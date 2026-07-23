"""Compile verified Labeler clips into a standard Arioso dataset root.

The Labeler produces genuine human ground truth per clip (``clips/<id>/notes.json``:
per-note timing / pitch / velocity / articulation / vibrato, plus mute regions). This
module is the *producer* half of the shared dataset contract: it turns every clip the
human has marked ``verified`` into the on-disk layout Arioso trains from, exactly as
specified by :mod:`common.dataset_schema` — so the Labeler and the DataSynthesizer emit
byte-identical roots without ever importing each other.

Per verified clip ``<id>`` we write, under :attr:`CompileParams.root`:

* ``gt/<id>.wav`` — the chosen wav variant (cleaned by default), mute regions
  cosine-zeroed, then voiced-RMS-normalized to :data:`common.config.TARGET_RMS_DBFS`
  (mono PCM16 @ 44.1 kHz). This is the training target waveform.
* ``target_mel/<id>.npy`` — ``[128, T]`` mel of that GT (:func:`common.vocoder.mel_frames`).
* ``prior_mel/<id>.npy`` — ``[128, T]`` mel of the informed sawtooth prior rendered
  from the surviving notes (real velocities) at the GT's exact length.
* ``onsets/<id>.npy`` — ``[K]`` int32 onset frames.
* ``offsets/<id>.npy`` — ``[K]`` int32 note-offset frames (Arioso's sinusoidal
  boundary-distance conditioning reads these alongside ``onsets/``).
* ``cond/{articulation,vibrato}/<id>.npy`` — ``[T]`` uint8 conditioning tracks. (Per-note
  dynamics are baked into ``prior_mel`` via ``env_dct`` — there is no velocity cond track.)

**Notes already live in cleaned-audio time** (the align stage baked the offset into
``notes.json``), so every rasterizer runs at ``offset_s = 0``. Notes whose midpoint
falls inside a mute region are dropped before prior/onset/cond synthesis, so a muted
stretch carries neither audio nor labels.

**Incremental + idempotent.** Each clip's manifest entry records a provenance triple
``{rev, notes_hash, compile_hash}``. A clip is skipped when that triple matches AND all
of its artifact files exist (``--force`` overrides). A full (``--all``) run also prunes:
clips in the manifest that are no longer verified (or whose clip dir vanished) are removed
from the manifest and their artifacts deleted. The manifest is rewritten atomically after
every compiled/pruned clip, so an interrupted compile is resumable and training never
sees a half-written manifest.

CLI (from the project root, ai-violin env)::

    python -m Labeler.compile --all                 # every verified clip (incremental)
    python -m Labeler.compile scale_slur scale_spicc # only these (must be verified)
    python -m Labeler.compile --all --force          # ignore the skip cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading

import numpy as np

from common.audio_io import load_mono, voiced_rms_normalize, write_pcm16
from common.config import HOP_SIZE, N_MELS, SR
from common.dataset_schema import (ARTICULATIONS, DIR_GT, DIR_OFFSETS, DIR_ONSETS,
                                   DIR_PRIOR_MEL, DIR_TARGET_MEL,
                                   MANIFEST_JSON, MANIFEST_SCHEMA_VERSION,
                                   NoteEvent, cond_dir, load_manifest,
                                   offset_frames, onset_frames, rasterize_articulation,
                                   rasterize_vibrato,
                                   write_manifest)
from common.prior import quantized_prior
from common.vocoder import mel_frames

from . import notes_store
from .config import LabelerConfig, load_config
from .library import CLEANED_WAV, SOURCE_WAV, clip_file, scan_clips
from .midi_io import notes_to_pretty_midi

# Root-wide manifest identity for the human ground-truth corpus.
ROOT_NAME = "gt_arky"
SIGNALS = ["articulation", "vibrato"]


# --- mute regions ---------------------------------------------------------------

def apply_mute_regions(y: np.ndarray, regions: list[dict], sr: int = SR,
                       fade_ms: float = 5.0) -> np.ndarray:
    """Return a copy of ``y`` with each mute region silenced via cosine edges.

    Recovered from the deleted ``Labeler/export.py``: a ``fade_ms`` cosine ramp fades
    each region to zero at its edges and holds silence in between, so a muted stretch is
    truly silent without a click at the boundary.
    """
    y = np.array(y, dtype=np.float32, copy=True)
    n = len(y)
    f = max(1, int(round(fade_ms / 1000.0 * sr)))
    for reg in regions or []:
        s = max(0, int(round(float(reg["start_s"]) * sr)))
        e = min(n, int(round(float(reg["end_s"]) * sr)))
        if e <= s:
            continue
        gain = np.ones(e - s, dtype=np.float32)
        ff = min(f, (e - s) // 2)
        if ff > 0:
            ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, ff)))  # 1 -> 0
            gain[:ff] = ramp
            gain[-ff:] = ramp[::-1]
        gain[ff:len(gain) - ff] = 0.0
        y[s:e] *= gain
    return y


def _note_midpoint(note: dict) -> float:
    return 0.5 * (float(note["start_s"]) + float(note["end_s"]))


def _in_any_region(t: float, regions: list[dict]) -> bool:
    return any(float(r["start_s"]) <= t <= float(r["end_s"]) for r in (regions or []))


# --- provenance hashing ---------------------------------------------------------

def notes_hash(notes: list[dict], mute_regions: list[dict]) -> str:
    """SHA-1 over a canonical dump of the fields that determine the compiled output.

    Canonical form: the notes list reduced to ``{start_s, end_s, pitch, velocity,
    technique, vibrato, env_dct}`` (times rounded to 1e-6 s to match ``notes.json``'s own
    rounding; env_dct coeffs to 1e-4) plus the mute regions reduced to ``{start_s, end_s}``,
    serialized with
    ``sort_keys=True``. Any edit that changes what the compile emits changes this digest;
    cosmetic fields (``id``, ``auto`` block, ``human`` flags, ``slur_group``) do not.
    """
    canon = {
        "notes": [
            {
                "start_s": round(float(n["start_s"]), 6),
                "end_s": round(float(n["end_s"]), 6),
                "pitch": int(n["pitch"]),
                "velocity": int(n["velocity"]),
                "technique": str(n["technique"]),
                "vibrato": bool(n.get("vibrato", False)),
                "env_dct": [round(float(x), 4) for x in n["env_dct"]] if n.get("env_dct") else None,
            }
            for n in notes
        ],
        "mute_regions": [
            {"start_s": round(float(r["start_s"]), 6), "end_s": round(float(r["end_s"]), 6)}
            for r in (mute_regions or [])
        ],
    }
    blob = json.dumps(canon, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


# --- artifact paths -------------------------------------------------------------

def _artifact_paths(root: str, base: str) -> dict[str, str]:
    """Every on-disk artifact path for clip ``base`` under ``root``."""
    return {
        "gt": os.path.join(root, DIR_GT, base + ".wav"),
        "target_mel": os.path.join(root, DIR_TARGET_MEL, base + ".npy"),
        "prior_mel": os.path.join(root, DIR_PRIOR_MEL, base + ".npy"),
        "onsets": os.path.join(root, DIR_ONSETS, base + ".npy"),
        "offsets": os.path.join(root, DIR_OFFSETS, base + ".npy"),
        "articulation": os.path.join(root, cond_dir("articulation"), base + ".npy"),
        "vibrato": os.path.join(root, cond_dir("vibrato"), base + ".npy"),
    }


def _artifacts_exist(root: str, base: str) -> bool:
    return all(os.path.isfile(p) for p in _artifact_paths(root, base).values())


def _remove_clip(root: str, base: str, manifest: dict) -> None:
    """Delete ``base``'s artifacts and drop it from the manifest (in place)."""
    for p in _artifact_paths(root, base).values():
        try:
            os.remove(p)
        except OSError:
            pass
    manifest.get("clips", {}).pop(base, None)


# --- manifest -------------------------------------------------------------------

def _empty_manifest() -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "name": ROOT_NAME,
        "frame": {"sr": SR, "hop": HOP_SIZE, "n_mels": N_MELS},
        "signals": list(SIGNALS),
        "clips": {},
    }


def _load_or_init_manifest(root: str) -> dict:
    """Load ``<root>/manifest.json`` (validated) or start a fresh one."""
    if os.path.isfile(os.path.join(root, MANIFEST_JSON)):
        manifest = load_manifest(root)
        manifest.setdefault("clips", {})
        manifest["name"] = ROOT_NAME
        manifest["signals"] = list(SIGNALS)
        return manifest
    return _empty_manifest()


# --- vocab guard ----------------------------------------------------------------

def check_vocab(cfg: LabelerConfig) -> None:
    """Hard error unless every configured articulation name is a schema class.

    The compile rasterizes articulation ids via ``ARTICULATIONS.index(name)``; a config
    articulation outside the unified vocabulary would blow up mid-clip, so we reject it up
    front with a clear listing.
    """
    unknown = [n for n in cfg.articulation_names if n not in ARTICULATIONS]
    if unknown:
        raise ValueError(
            f"config articulation(s) {unknown} not in the unified vocabulary "
            f"{list(ARTICULATIONS)}; align labeler.yaml with common.dataset_schema.ARTICULATIONS")


# --- per-clip compile -----------------------------------------------------------

def compile_one(cfg: LabelerConfig, clip_id: str, manifest: dict,
                force: bool = False) -> bool:
    """Compile one verified clip into ``manifest``'s root; return True if it (re)built.

    Returns False when the clip is up to date (provenance matches + artifacts present)
    and ``force`` is not set — i.e. it was skipped. Raises on a validation failure
    (unknown articulation, missing wav) with the clip id in the message.
    """
    cp = cfg.compile
    root = cp.root
    doc = notes_store.load(cfg, clip_id)
    if doc is None:
        raise ValueError(f"{clip_id}: no notes.json to compile")
    notes = doc.get("notes", [])
    mute_regions = doc.get("mute_regions", [])
    rev = int(doc.get("rev", 0))
    n_hash = notes_hash(notes, mute_regions)
    c_hash = cfg.stage_params_hash("compile")

    # -- idempotent skip: provenance triple matches AND all artifacts on disk --
    entry = manifest.get("clips", {}).get(clip_id)
    if entry is not None and not force:
        prov = entry.get("provenance", {})
        if (prov.get("rev") == rev and prov.get("notes_hash") == n_hash
                and prov.get("compile_hash") == c_hash
                and _artifacts_exist(root, clip_id)):
            return False

    # -- validate every note's articulation (hard error listing note ids) --
    bad = [n.get("id", "?") for n in notes if str(n.get("technique")) not in ARTICULATIONS]
    if bad:
        raise ValueError(
            f"{clip_id}: {len(bad)} note(s) with an articulation outside "
            f"{list(ARTICULATIONS)}: {bad}")

    # -- drop notes whose midpoint falls inside a mute region --
    survivors = [n for n in notes if not _in_any_region(_note_midpoint(n), mute_regions)]
    events = [
        NoteEvent(start_s=float(n["start_s"]), end_s=float(n["end_s"]),
                  pitch=int(n["pitch"]), velocity=int(n["velocity"]),
                  articulation=str(n["technique"]), vibrato=bool(n.get("vibrato", False)),
                  env_dct=tuple(n["env_dct"]) if n.get("env_dct") else None)
        for n in survivors
    ]

    # -- GT wav: chosen variant -> cosine-zero mutes -> voiced-RMS normalize --
    variant_wav = CLEANED_WAV if cp.gt_variant == "cleaned" else SOURCE_WAV
    src_wav = clip_file(cfg, clip_id, variant_wav)
    if not os.path.isfile(src_wav):
        raise FileNotFoundError(
            f"{clip_id}: missing {variant_wav} (process the clip before compiling)")
    y = load_mono(src_wav, sr=SR)
    y = apply_mute_regions(y, mute_regions, sr=SR, fade_ms=cp.mute_fade_ms)
    y = voiced_rms_normalize(y, sr=SR)  # defaults track common.config TARGET_RMS_DBFS/VOICED_TOP_DB
    y = np.asarray(y, dtype=np.float32)
    n_samples = int(len(y))

    # -- target mel (defines T) --
    target_mel = mel_frames(y)
    n_frames = int(target_mel.shape[1])

    # -- prior: surviving notes -> in-memory score (env_dct baked per note) -> mel --
    pm = notes_to_pretty_midi(survivors, tempo=cp.tempo, program=cp.program, with_env=True)
    prior_wav = quantized_prior().render(pm, total_samples=n_samples)
    prior_mel = mel_frames(prior_wav)
    if prior_mel.shape[1] != n_frames:
        raise AssertionError(
            f"{clip_id}: prior_mel T={prior_mel.shape[1]} != target_mel T={n_frames}")

    # -- onsets + offsets + two conditioning tracks (notes already in cleaned-audio time) --
    onsets = onset_frames(events, n_frames)
    offs = offset_frames(events, n_frames)
    artic = rasterize_articulation(events, n_frames)
    vib = rasterize_vibrato(events, n_frames)

    # -- write artifacts (manifest is committed by the caller, after these succeed) --
    paths = _artifact_paths(root, clip_id)
    for p in paths.values():
        os.makedirs(os.path.dirname(p), exist_ok=True)
    write_pcm16(paths["gt"], y, SR)
    np.save(paths["target_mel"], target_mel.astype(np.float32))
    np.save(paths["prior_mel"], prior_mel.astype(np.float32))
    np.save(paths["onsets"], onsets.astype(np.int32))
    np.save(paths["offsets"], offs.astype(np.int32))
    np.save(paths["articulation"], artic.astype(np.uint8))
    np.save(paths["vibrato"], vib.astype(np.uint8))

    manifest.setdefault("clips", {})[clip_id] = {
        "n_frames": n_frames,
        "n_samples": n_samples,
        "duration_s": round(n_samples / float(SR), 6),
        "piece": clip_id,
        "provenance": {"rev": rev, "notes_hash": n_hash, "compile_hash": c_hash},
    }
    return True


# --- whole-corpus compile -------------------------------------------------------

def compile_all(cfg: LabelerConfig, clip_ids: list[str] | None = None,
                force: bool = False, progress=None, verbose: bool = True) -> dict:
    """Compile verified clips into ``cfg.compile.root``; return a run summary.

    ``clip_ids is None`` compiles every verified clip and prunes stale ones (a full run);
    an explicit list compiles just those (each must be verified — else a listing error)
    and prunes nothing. ``progress(**fields)`` is called with rolling
    ``total/done/skipped/failed/current/root`` so a UI can poll live counts.
    """
    check_vocab(cfg)
    root = cfg.compile.root
    os.makedirs(root, exist_ok=True)
    manifest = _load_or_init_manifest(root)
    infos = {info.id: info for info in scan_clips(cfg)}

    full = clip_ids is None
    if full:
        targets = sorted(cid for cid, info in infos.items() if info.verified)
    else:
        bad = [c for c in clip_ids if c not in infos or not infos[c].verified]
        if bad:
            raise ValueError(
                f"cannot compile — not verified (or unknown): {', '.join(sorted(bad))}")
        targets = list(clip_ids)

    total = len(targets)
    done = skipped = failed = 0

    def _emit(**kw):
        if progress:
            progress(**kw)

    _emit(root=root, total=total, done=0, skipped=0, failed=0, current=None)

    # Prune (full runs only): manifest clips that are no longer verified / gone.
    if full:
        for base in list(manifest.get("clips", {})):
            info = infos.get(base)
            if info is None or not info.verified:
                _remove_clip(root, base, manifest)
                write_manifest(root, manifest)
                if verbose:
                    print(f"[compile] pruned  {base} (no longer verified)")

    for cid in targets:
        _emit(current=cid, total=total, done=done, skipped=skipped, failed=failed)
        try:
            built = compile_one(cfg, cid, manifest, force=force)
            if built:
                write_manifest(root, manifest)  # atomic, per clip -> resumable
                done += 1
                if verbose:
                    e = manifest["clips"][cid]
                    print(f"[compile] built   {cid}  T={e['n_frames']}  {e['duration_s']:.2f}s")
            else:
                skipped += 1
                if verbose:
                    print(f"[compile] skipped {cid} (up to date)")
        except Exception as e:  # noqa: BLE001 - one bad clip must not abort the run
            failed += 1
            if verbose:
                print(f"[compile] FAILED  {cid}: {type(e).__name__}: {e}")
        _emit(current=None, total=total, done=done, skipped=skipped, failed=failed)

    summary = {"total": total, "done": done, "skipped": skipped, "failed": failed,
               "root": root}
    if verbose:
        print(f"[compile] done: {done} built, {skipped} skipped, {failed} failed "
              f"-> {root}")
    return summary


# --- background runner (server-side) --------------------------------------------

class CompileManager:
    """Runs :func:`compile_all` on a single background thread (one compile at a time)."""

    def __init__(self, cfg: LabelerConfig):
        self.cfg = cfg
        self._thread: threading.Thread | None = None
        self._guard = threading.Lock()
        self._state = {"running": False, "done": 0, "skipped": 0, "failed": 0,
                       "current": None, "total": 0, "root": cfg.compile.root}

    def status(self) -> dict:
        with self._guard:
            return dict(self._state)

    def start(self) -> bool:
        """Start a full compile in the background; False if one is already running."""
        with self._guard:
            if self._thread and self._thread.is_alive():
                return False
            self._state = {"running": True, "done": 0, "skipped": 0, "failed": 0,
                           "current": None, "total": 0, "root": self.cfg.compile.root}

            def _prog(**kw):
                with self._guard:
                    self._state.update(kw)

            def _run():
                try:
                    compile_all(self.cfg, clip_ids=None, progress=_prog, verbose=True)
                except Exception as e:  # noqa: BLE001 - surface, never crash the thread
                    print(f"[compile] run failed: {type(e).__name__}: {e}")
                finally:
                    with self._guard:
                        self._state["running"] = False
                        self._state["current"] = None

            self._thread = threading.Thread(target=_run, name="labeler-compile", daemon=True)
            self._thread.start()
            return True


# --- CLI ------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m Labeler.compile",
        description="Compile verified Labeler clips into a standard Arioso dataset root.")
    ap.add_argument("clip_ids", nargs="*",
                    help="explicit verified clip ids to compile (omit with --all)")
    ap.add_argument("--all", action="store_true",
                    help="compile every verified clip and prune stale ones")
    ap.add_argument("--config", default=None, help="path to a labeler.yaml override")
    ap.add_argument("--force", action="store_true",
                    help="recompile even when the provenance triple already matches")
    args = ap.parse_args(argv)

    if args.all and args.clip_ids:
        ap.error("give either --all or explicit clip ids, not both")
    if not args.all and not args.clip_ids:
        ap.error("nothing to compile: pass --all or one or more clip ids")

    cfg = load_config(args.config)
    clip_ids = None if args.all else args.clip_ids
    res = compile_all(cfg, clip_ids=clip_ids, force=args.force)
    return 1 if res["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
