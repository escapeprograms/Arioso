"""Build a BigVGAN fine-tuning corpus of (Arioso-generated mel, GT audio) chunk pairs.

BigVGAN-v2 is trained on natural mels; Arioso emits *its own* mels (the flow-matched
prior->target output), whose fine harmonic structure differs subtly from a real mel.
Fine-tuning the vocoder on Arioso's mels closes that train/deploy gap. This builder runs
Arioso inference once over a standard dataset root and lays the results out exactly the
way NVIDIA BigVGAN's ``train.py --fine_tuning`` expects (see
``external/BigVGAN/meldataset.py`` fine-tuning branch): for every wav ``<stem>.wav`` there
is a precomputed mel ``<base_mels_path>/<stem>.npy``, the wav is at sr 44100, and
``len(wav) == mel_T * 512`` exactly.

Two pair kinds are emitted per clip:

* ``_a`` chunks — (Arioso-generated mel, GT audio): the fine-tuning objective proper.
* ``_g`` chunks — (GT target_mel, GT audio): copy-synthesis pairs that anchor the vocoder
  to natural mels so fine-tuning does not drift the copy-synthesis quality. BigVGAN keys a
  mel by its wav stem, so the same audio chunk is duplicated under a distinct ``_g`` stem
  to carry the second (mel, audio) mapping.

**Loudness.** The GT audio is the project's voiced-RMS -20 dBFS track (the cross-producer
loudness contract in ``common.config``) copied through verbatim. It is deliberately NOT
peak-renormalized here: BigVGAN's fine-tuning branch does no renormalization either, so the
(mel, audio) pair stays self-consistent and the vocoder is fine-tuned at exactly the
loudness it will see at deployment. Renormalizing would desync the paired mel from its audio.

Run (from the repo root, so ``python -m`` resolves the sibling packages)::

    python -m Vocoder.build_vocoder_dataset --limit 1        # smoke test: one clip
    python -m Vocoder.build_vocoder_dataset                  # full build
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time

import numpy as np
import soundfile as sf
import torch

from Arioso.config import BoundaryCondSpec, cfg_from_dict
from Arioso.dataset import boundary_distances
from Arioso.infer import generate_mel
from Arioso.model import AriosoModel
from common.audio_io import write_pcm16
from common.config import HOP_SIZE, N_MELS, SR
from common.dataset_schema import DatasetRoot

WAVS_DIR = "wavs_nonspeech"          # BigVGAN input_wavs_dir
MELS_DIR = "arioso_mel"              # BigVGAN input_mels_dir (base_mels_path)
META_JSON = "build_meta.json"
DEFAULT_CKPT = os.path.join("Arioso", "models", "7-25-suzuki-3", "checkpoint_final.pt")


# --- model + conditioning (ported from Arioso.eval.metrics) ---------------------

def load_model(ckpt_path: str, weights: str, device: str):
    """Rebuild the Arioso model from the checkpoint's embedded cfg; return ``(model, cfg)``.

    Mirrors ``Arioso.eval.metrics``: ``torch.load`` -> ``cfg_from_dict(ckpt["cfg"])`` ->
    ``AriosoModel`` -> load the requested weights -> ``eval()``. If the requested weights key
    (``ema``) is absent, fall back to the ``model`` key with a warning.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = cfg_from_dict(ckpt.get("cfg") or {})
    model = AriosoModel(cfg).to(device)
    key = weights if weights in ckpt else "model"
    if key != weights:
        print(f"warning: checkpoint has no {weights!r} weights; falling back to {key!r}")
    model.load_state_dict(ckpt[key])
    model.eval()
    return model, cfg


def build_cond_frames(root: DatasetRoot, base: str, cfg, n_frames: int):
    """Per-frame conditioning tracks on the prior grid, exactly as ``Arioso.eval.metrics`` does.

    Boundary signals read the root's ``onsets``/``offsets`` frame array (all-``-1`` sentinel when
    the array is missing); categorical signals read their ``cond/<signal>`` track, or the CFG
    "unknown" fill (``spec.num_classes``) for a signal this root does not carry. Returns ``None``
    when the checkpoint has no conditioning (unconditioned baseline).
    """
    cond_frames: dict[str, np.ndarray] = {}
    for spec in cfg.conditioning:
        if isinstance(spec, BoundaryCondSpec):
            path = root.onsets_path(base) if spec.boundary == "onset" else root.offsets_path(base)
            if not os.path.isfile(path):
                cond_frames[spec.name] = np.full(n_frames, -1, dtype=np.int64)
            else:
                bounds = np.load(path).astype(np.int64)
                cond_frames[spec.name] = boundary_distances(bounds, 0, n_frames, spec.direction)
        elif spec.name not in root.signals:
            cond_frames[spec.name] = np.full(n_frames, spec.num_classes, dtype=np.int64)
        else:
            cond_frames[spec.name] = np.load(root.cond_path(spec.name, base))[:n_frames]
    return cond_frames or None


# --- GT audio -------------------------------------------------------------------

def load_gt_wav(path: str) -> np.ndarray:
    """Load a GT wav as a mono float32 array, asserting sr == 44100 (fail loud on a bad rate).

    Uses ``soundfile.read`` + an explicit rate assert rather than ``common.audio_io.load_mono``
    (which would silently resample) so a mis-rated file aborts the build instead of corrupting a
    pair. The project's GT wavs are already mono PCM16 @ SR.
    """
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == SR, f"{path}: sample rate {sr} != {SR}"
    if wav.ndim > 1:
        raise ValueError(f"{path}: expected mono audio, got shape {wav.shape}")
    return wav


# --- per-base processing --------------------------------------------------------

def _chunk_bounds(t_frames: int, chunk: int, min_chunk: int) -> list[tuple[int, int]]:
    """Contiguous ``[s, e)`` frame windows of ``chunk`` frames; drop a trailing short chunk."""
    bounds = []
    for s in range(0, t_frames, chunk):
        e = min(s + chunk, t_frames)
        if e - s >= min_chunk:
            bounds.append((s, e))
    return bounds


def process_base(root: DatasetRoot, base: str, model, cfg, device: str,
                 solver: str | None, steps: int | None,
                 chunk: int, min_chunk: int, wavs_dir: str, mels_dir: str) -> dict | None:
    """Run inference for one clip and write its ``_a``/``_g`` chunk pairs; return a meta entry.

    Returns ``None`` if the clip is missing a required artifact (prior/target mel or GT wav) or
    yields no chunks after trimming. The meta entry lists every stem written so a rerun can verify
    completeness.
    """
    prior_path = root.prior_mel_path(base)
    target_path = root.target_mel_path(base)
    gt_path = root.gt_path(base)
    if not (os.path.isfile(prior_path) and os.path.isfile(target_path) and os.path.isfile(gt_path)):
        print(f"  skip {base}: missing prior/target mel or GT wav")
        return None

    prior = np.load(prior_path).astype(np.float32)
    target = np.load(target_path).astype(np.float32)
    wav = load_gt_wav(gt_path)

    cond_frames = build_cond_frames(root, base, cfg, prior.shape[-1])
    with torch.no_grad():
        pred = generate_mel(model, prior, cfg, device, solver=solver, steps=steps,
                            cond_frames=cond_frames).astype(np.float32)

    # Common frame grid across all three artifacts; trim each (wav to whole frames).
    t = min(pred.shape[-1], target.shape[-1], len(wav) // HOP_SIZE)
    pred = pred[:, :t]
    target = target[:, :t]
    wav = wav[:t * HOP_SIZE]

    bounds = _chunk_bounds(t, chunk, min_chunk)
    if not bounds:
        print(f"  skip {base}: {t} frames < min chunk {min_chunk}")
        return None

    stems: list[str] = []
    for i, (s, e) in enumerate(bounds):
        wav_chunk = wav[s * HOP_SIZE:e * HOP_SIZE]           # (e - s) * 512 samples exactly
        for tag, mel in (("a", pred), ("g", target)):
            stem = f"{base}_{tag}{i:03d}"
            np.save(os.path.join(mels_dir, stem + ".npy"),
                    np.ascontiguousarray(mel[:, s:e], dtype=np.float32))
            write_pcm16(os.path.join(wavs_dir, stem + ".wav"), wav_chunk)
            stems.append(stem)

    print(f"  {base}: {t} frames -> {len(bounds)} chunks ({len(stems)} pairs)")
    return {"chunks": stems, "n_frames": int(t)}


# --- filelists + idempotency ----------------------------------------------------

def _is_tag(stem: str, base: str, tag: str) -> bool:
    """True if ``stem`` is a ``base``'s ``_a``/``_g`` chunk (suffix ``_<tag>###`` after ``base``)."""
    return stem.startswith(base) and stem[len(base):len(base) + 2] == f"_{tag}"


def write_filelists(meta: dict, out: str, no_copy_synth: bool) -> dict:
    """(Re)generate ``train/val/val_copysynth.txt`` from ``meta``; return chunk-count stats.

    Deterministic: bases sorted, chunks in stored order. Lines are LibriTTS-style ``"stem|"``
    (empty transcript, no ``.wav`` extension). ``train.txt`` = ``_a`` chunks of train bases plus,
    unless ``no_copy_synth``, their ``_g`` copy-synth chunks. ``val.txt`` = ``_a`` chunks of val
    bases; ``val_copysynth.txt`` = their ``_g`` chunks.
    """
    train, val, val_cs = [], [], []
    for base in sorted(meta["bases"]):
        entry = meta["bases"][base]
        a = [s for s in entry["chunks"] if _is_tag(s, base, "a")]
        g = [s for s in entry["chunks"] if _is_tag(s, base, "g")]
        if entry["split"] == "val":
            val.extend(a)
            val_cs.extend(g)
        else:
            train.extend(a)
            if not no_copy_synth:
                train.extend(g)

    for name, stems in (("train.txt", train), ("val.txt", val),
                        ("val_copysynth.txt", val_cs)):
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            f.write("".join(f"{s}|\n" for s in stems))
    return {"train": len(train), "val": len(val), "val_copysynth": len(val_cs)}


def _provenance(ckpt_path: str, solver: str, steps: int, chunk: int, min_chunk: int) -> dict:
    """Args-provenance block: ckpt identity (path + mtime + size) and the build params."""
    st = os.stat(ckpt_path)
    return {"ckpt": ckpt_path, "ckpt_mtime": st.st_mtime, "ckpt_size": st.st_size,
            "solver": solver, "steps": steps,
            "chunk_frames": chunk, "min_chunk_frames": min_chunk}


def _load_meta(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_meta(path: str, meta: dict) -> None:
    """Atomically write ``meta`` (same-dir tmp + os.replace) so a crash keeps a valid meta."""
    fd, tmp = tempfile.mkstemp(prefix=".tmp_meta_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _base_complete(entry: dict, wavs_dir: str, mels_dir: str) -> bool:
    """True if every stem in ``entry`` has both its mel and wav on disk (idempotency check)."""
    for stem in entry["chunks"]:
        if not (os.path.isfile(os.path.join(mels_dir, stem + ".npy"))
                and os.path.isfile(os.path.join(wavs_dir, stem + ".wav"))):
            return False
    return True


# --- end-of-run self-check ------------------------------------------------------

def self_check(meta: dict, wavs_dir: str, mels_dir: str) -> None:
    """Reload one random written pair and assert the BigVGAN fine-tuning contract holds."""
    all_stems = [s for e in meta["bases"].values() for s in e["chunks"]]
    if not all_stems:
        print("self-check: no chunks written")
        return
    stem = random.choice(all_stems)
    mel = np.load(os.path.join(mels_dir, stem + ".npy"))
    wav, sr = sf.read(os.path.join(wavs_dir, stem + ".wav"))
    assert sr == SR, f"self-check {stem}: sr {sr} != {SR}"
    assert mel.dtype == np.float32, f"self-check {stem}: mel dtype {mel.dtype} != float32"
    assert mel.shape[0] == N_MELS, f"self-check {stem}: mel bands {mel.shape[0]} != {N_MELS}"
    assert len(wav) == mel.shape[1] * HOP_SIZE, (
        f"self-check {stem}: {len(wav)} samples != {mel.shape[1]} frames * {HOP_SIZE}")
    print(f"self-check OK on {stem}: mel {mel.shape} {mel.dtype}, {len(wav)} samples")


# --- main -----------------------------------------------------------------------

def build(args) -> None:
    device = args.device
    wavs_dir = os.path.join(args.out, WAVS_DIR)
    mels_dir = os.path.join(args.out, MELS_DIR)
    os.makedirs(wavs_dir, exist_ok=True)
    os.makedirs(mels_dir, exist_ok=True)
    meta_path = os.path.join(args.out, META_JSON)

    model, cfg = load_model(args.ckpt, args.weights, device)
    # Effective solver/steps (None -> the cfg inference defaults) so provenance is stable.
    eff_solver = args.solver or cfg.solver
    eff_steps = args.steps or cfg.solver_steps
    provenance = _provenance(args.ckpt, eff_solver, eff_steps,
                             args.chunk_frames, args.min_chunk_frames)

    # Idempotency: reuse an existing meta unless --overwrite; abort on a provenance mismatch.
    existing = None if args.overwrite else _load_meta(meta_path)
    if existing is not None and existing.get("provenance") != provenance:
        raise SystemExit(
            f"provenance mismatch with existing {meta_path}:\n"
            f"  existing: {existing.get('provenance')}\n  requested: {provenance}\n"
            f"pass --overwrite to rebuild, or point --out at a fresh directory.")
    meta = existing or {"provenance": provenance, "bases": {}}

    root = DatasetRoot(args.data_root)
    with open(os.path.join(args.data_root, "split.json"), encoding="utf-8") as f:
        split = json.load(f)
    val_bases = set(split.get("val", []))
    known = set(split.get("train", [])) | val_bases

    bases = root.basenames()
    if args.limit:
        bases = bases[:args.limit]

    missing = [b for b in bases if b not in known]
    if missing:
        print(f"warning: {len(missing)} basename(s) absent from split.json -> assigned to train: "
              f"{', '.join(sorted(missing))}")

    n_done, n_skip = 0, 0
    for i, base in enumerate(bases, 1):
        split_name = "val" if base in val_bases else "train"
        if (not args.overwrite and base in meta["bases"]
                and _base_complete(meta["bases"][base], wavs_dir, mels_dir)):
            n_skip += 1
            continue
        print(f"[{i}/{len(bases)}] {base} ({split_name})")
        entry = process_base(root, base, model, cfg, device, args.solver, args.steps,
                             args.chunk_frames, args.min_chunk_frames, wavs_dir, mels_dir)
        if entry is None:
            continue
        entry["split"] = split_name
        meta["bases"][base] = entry
        _save_meta(meta_path, meta)                          # flush per base (resumable)
        n_done += 1

    _save_meta(meta_path, meta)
    stats = write_filelists(meta, args.out, args.no_copy_synth)
    self_check(meta, wavs_dir, mels_dir)

    total_frames = sum(e["n_frames"] * 2 for e in meta["bases"].values())  # _a + _g per frame
    total_sec = total_frames * HOP_SIZE / SR
    total_mb = _dir_size_mb(wavs_dir) + _dir_size_mb(mels_dir)
    n_val = sum(1 for e in meta["bases"].values() if e["split"] == "val")
    print(f"\nDone. bases processed {n_done}, skipped {n_skip}, in meta {len(meta['bases'])} "
          f"({n_val} val).\n"
          f"  filelist chunks: train {stats['train']}, val {stats['val']}, "
          f"val_copysynth {stats['val_copysynth']}\n"
          f"  total audio: {total_sec:.1f} s (both pair kinds), output size {total_mb:.1f} MB\n"
          f"  -> {args.out}")


def _dir_size_mb(path: str) -> float:
    total = 0
    for name in os.listdir(path):
        fp = os.path.join(path, name)
        if os.path.isfile(fp):
            total += os.path.getsize(fp)
    return total / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.path.join("Data", "datasets", "gt_arky"))
    ap.add_argument("--out", default=os.path.join("Data", "datasets", "vocoder_ft"))
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--weights", choices=("ema", "model"), default="ema",
                    help="checkpoint weights key (falls back to 'model' if absent)")
    ap.add_argument("--solver", default=None, help="ODE solver (default: cfg's, euler)")
    ap.add_argument("--steps", type=int, default=None, help="solver steps (default: cfg's, 24)")
    ap.add_argument("--chunk-frames", type=int, default=512)
    ap.add_argument("--min-chunk-frames", type=int, default=128)
    ap.add_argument("--no-copy-synth", action="store_true",
                    help="drop _g copy-synth pairs from train.txt only (the _g files and "
                         "val_copysynth.txt are always written)")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild all bases and ignore any existing build_meta.json")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N basenames")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.time()
    build(args)
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
