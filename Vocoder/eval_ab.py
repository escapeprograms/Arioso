"""A/B evaluation of a fine-tuned BigVGAN generator vs the pretrained baseline.

Fine-tuning (``Vocoder.fine_tune``) adapts BigVGAN-v2 to Arioso's own mels. This
script scores how much that helped: it vocodes the held-out val chunks of a
``Vocoder.build_vocoder_dataset`` corpus with BOTH the pretrained baseline (from
``common.vocoder.load_vocoder``) and a fine-tuned generator snapshot, then compares
each output against the ground-truth audio on three vocoder-quality metrics:

* **MR-STFT** — auraloss multi-resolution STFT loss (default resolutions), the
  standard perceptual reconstruction metric for neural vocoders.
* **mel L1** — L1 between the BigVGAN mel of the output and of the GT audio
  (via ``common.vocoder.mel_frames``, the shared front-end).
* **MCD** — mel-cepstral distortion (``Arioso.eval.metrics.mcd``).

Lower is better for all three. Two val lists are scored separately: ``val.txt``
(Arioso-mel chunks — the fine-tuning objective proper) and ``val_copysynth.txt``
(GT-mel copy-synthesis chunks — the anchor that fine-tuning must not regress).

Per-chunk metrics land in ``<out>/metrics.csv``; a summary table of per-list means
plus an improved/regressed verdict per metric prints to the console. Paired wavs
``<stem>__baseline.wav`` / ``__finetuned.wav`` / ``__gt.wav`` are written for blind
listening.

Run (from the repo root)::

    python -m Vocoder.eval_ab --run-dir Vocoder/runs/ft_v1
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np
import soundfile as sf
import torch

from Arioso.eval.metrics import mcd
from common.audio_io import write_pcm16
from common.config import (FMAX, FMIN, HOP_SIZE, N_FFT, N_MELS, SR, WIN_SIZE)
from common.vocoder import _ensure_bigvgan_on_path, load_vocoder, mel_frames, vocode

WAVS_DIR = "wavs_nonspeech"
MELS_DIR = "arioso_mel"

# Which filelist maps to which human-readable list label in the summary.
_LISTS = (("arioso-val", "val.txt"), ("copysynth-val", "val_copysynth.txt"))

# Lower is better for every metric, so the verdict is "improved" when finetuned < baseline.
_METRICS = ("stft", "mel_l1", "mcd")


# --- fine-tuned generator load (mirrors common.vocoder.load_vocoder) -------------

def _newest_snapshot(run_dir: str) -> str | None:
    """Newest step-prefixed ``g_????????`` generator snapshot in ``run_dir``, or None."""
    snaps = sorted(glob.glob(os.path.join(run_dir, "g_" + "?" * 8)))
    return snaps[-1] if snaps else None


def load_finetuned(run_dir: str, checkpoint: str | None, device: str):
    """Load a fine-tuned BigVGAN generator from a run dir; return ``(model, ckpt_path)``.

    Config comes from ``<run_dir>/ft_config.json`` (falling back to ``config.json``);
    weights come from the explicit ``checkpoint`` if given, else the newest
    ``g_????????`` snapshot in the run dir. The mel-contract assert, weight-norm fold
    and ``eval().to(device)`` mirror ``common.vocoder.load_vocoder`` exactly — this
    build path exists only because the generator lives in a local run dir, not on the Hub.
    """
    _ensure_bigvgan_on_path()
    import bigvgan
    from env import AttrDict

    config_file = os.path.join(run_dir, "ft_config.json")
    if not os.path.isfile(config_file):
        config_file = os.path.join(run_dir, "config.json")
    if not os.path.isfile(config_file):
        raise FileNotFoundError(
            f"run dir {run_dir!r} has no ft_config.json or config.json")
    with open(config_file) as f:
        h = AttrDict(json.load(f))

    ckpt_path = checkpoint or _newest_snapshot(run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"run dir {run_dir!r} has no g_???????? generator snapshot "
            f"(and no --checkpoint given)")

    # Fail loudly if the checkpoint's mel params drift from our shared contract.
    expected = {
        "sampling_rate": SR, "hop_size": HOP_SIZE, "n_fft": N_FFT,
        "win_size": WIN_SIZE, "num_mels": N_MELS, "fmin": FMIN, "fmax": FMAX,
    }
    mismatch = {k: (h.get(k), v) for k, v in expected.items() if h.get(k) != v}
    if mismatch:
        raise ValueError(
            f"fine-tuned checkpoint {run_dir} mel params disagree with "
            f"common.config (got, expected): {mismatch}")

    model = bigvgan.BigVGAN(h, use_cuda_kernel=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # The freshly built model carries weight norm; the checkpoint may or may not.
    try:
        model.load_state_dict(ckpt["generator"])
        model.remove_weight_norm()
    except RuntimeError:
        model.remove_weight_norm()
        model.load_state_dict(ckpt["generator"])
    return model.eval().to(device), ckpt_path


# --- metrics --------------------------------------------------------------------

def _read_stems(list_path: str, limit: int | None) -> list[str]:
    """Parse a LibriTTS-style ``"stem|"`` filelist into stems (blank lines dropped)."""
    stems: list[str] = []
    with open(list_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                stems.append(line.split("|", 1)[0])
    return stems[:limit] if limit else stems


def compute_metrics(out_wav: np.ndarray, gt_wav: np.ndarray, stft_loss) -> dict:
    """MR-STFT, mel L1 and MCD of ``out_wav`` against ``gt_wav`` (trimmed to common length)."""
    t = min(len(out_wav), len(gt_wav))
    a = np.asarray(out_wav[:t], dtype=np.float32)
    b = np.asarray(gt_wav[:t], dtype=np.float32)

    # auraloss expects [batch, channel, time]; CPU tensors are fine.
    ta = torch.from_numpy(a).view(1, 1, -1)
    tb = torch.from_numpy(b).view(1, 1, -1)
    with torch.no_grad():
        stft = float(stft_loss(ta, tb))

    mel_a, mel_b = mel_frames(a), mel_frames(b)
    f = min(mel_a.shape[-1], mel_b.shape[-1])
    mel_l1 = float(np.mean(np.abs(mel_a[:, :f] - mel_b[:, :f])))
    return {"stft": stft, "mel_l1": mel_l1, "mcd": mcd(mel_a, mel_b)}


# --- main -----------------------------------------------------------------------

def evaluate(args) -> None:
    device = args.device
    out_dir = args.out or os.path.join(args.run_dir, "ab")
    os.makedirs(out_dir, exist_ok=True)

    import auraloss
    stft_loss = auraloss.freq.MultiResolutionSTFTLoss()

    print(f"loading baseline vocoder (device={device}) ...")
    # "hf" forces the stock pretrained checkpoint — config.VOCODER_DIR now points
    # at the fine-tune, and comparing the fine-tune against itself would be moot.
    baseline = load_vocoder(device=device, checkpoint_dir="hf")
    print(f"loading fine-tuned generator from {args.run_dir} ...")
    finetuned, ckpt_path = load_finetuned(args.run_dir, args.checkpoint, device)
    print(f"  fine-tuned checkpoint: {ckpt_path}")

    wavs_dir = os.path.join(args.dataset, WAVS_DIR)
    mels_dir = os.path.join(args.dataset, MELS_DIR)

    rows: list[dict] = []
    for list_label, list_name in _LISTS:
        list_path = os.path.join(args.dataset, list_name)
        if not os.path.isfile(list_path):
            print(f"warning: {list_path} not found; skipping {list_label}")
            continue
        stems = _read_stems(list_path, args.limit)
        print(f"\n{list_label} ({list_name}): {len(stems)} chunks")
        for stem in stems:
            mel_path = os.path.join(mels_dir, stem + ".npy")
            wav_path = os.path.join(wavs_dir, stem + ".wav")
            if not (os.path.isfile(mel_path) and os.path.isfile(wav_path)):
                print(f"  skip {stem}: missing mel or wav")
                continue

            mel = np.load(mel_path).astype(np.float32)          # [N_MELS, F]
            mel_t = torch.from_numpy(mel).unsqueeze(0)          # [1, N_MELS, F]
            gt_wav, sr = sf.read(wav_path, dtype="float32", always_2d=False)
            assert sr == SR, f"{wav_path}: sr {sr} != {SR}"

            base_wav = vocode(baseline, mel_t)
            ft_wav = vocode(finetuned, mel_t)

            base_m = compute_metrics(base_wav, gt_wav, stft_loss)
            ft_m = compute_metrics(ft_wav, gt_wav, stft_loss)

            row = {"stem": stem, "list": list_label}
            for m in _METRICS:
                row[f"{m}_baseline"] = base_m[m]
                row[f"{m}_finetuned"] = ft_m[m]
            rows.append(row)

            # Paired wavs for blind listening (trim to common length for A/B alignment).
            t = min(len(base_wav), len(ft_wav), len(gt_wav))
            write_pcm16(os.path.join(out_dir, f"{stem}__baseline.wav"),
                        base_wav[:t].astype(np.float32))
            write_pcm16(os.path.join(out_dir, f"{stem}__finetuned.wav"),
                        ft_wav[:t].astype(np.float32))
            write_pcm16(os.path.join(out_dir, f"{stem}__gt.wav"),
                        gt_wav[:t].astype(np.float32))
            print(f"  {stem}: "
                  + "  ".join(f"{m} {base_m[m]:.4f}->{ft_m[m]:.4f}" for m in _METRICS))

    if not rows:
        print("\nno chunks scored (empty val lists or missing artifacts).")
        return

    # Per-chunk CSV.
    csv_path = os.path.join(out_dir, "metrics.csv")
    fields = ["stem", "list"] + [f"{m}_{w}" for m in _METRICS
                                 for w in ("baseline", "finetuned")]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    _print_summary(rows, csv_path, out_dir)


def _print_summary(rows: list[dict], csv_path: str, out_dir: str) -> None:
    """Per-list means with an improved/regressed verdict per metric."""
    print("\n" + "=" * 72)
    print("A/B SUMMARY (lower is better; verdict = fine-tuned vs baseline)")
    print("=" * 72)
    for list_label, _ in _LISTS:
        sub = [r for r in rows if r["list"] == list_label]
        if not sub:
            continue
        print(f"\n{list_label}  ({len(sub)} chunks)")
        print(f"  {'metric':<8} {'baseline':>12} {'finetuned':>12} {'delta':>12}   verdict")
        for m in _METRICS:
            b = float(np.mean([r[f"{m}_baseline"] for r in sub]))
            ft = float(np.mean([r[f"{m}_finetuned"] for r in sub]))
            delta = ft - b
            verdict = "improved" if ft < b else ("regressed" if ft > b else "no change")
            print(f"  {m:<8} {b:>12.5f} {ft:>12.5f} {delta:>+12.5f}   {verdict}")
    print(f"\nper-chunk CSV : {csv_path}")
    print(f"paired wavs   : {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=os.path.join("Vocoder", "runs", "ft_v1"),
                    help="fine-tune run dir (config + g_???????? snapshots)")
    ap.add_argument("--checkpoint", default=None,
                    help="explicit g_ generator file (default: newest g_???????? in run dir)")
    ap.add_argument("--dataset", default=os.path.join("Data", "datasets", "vocoder_ft"),
                    help="vocoder_ft dataset dir (wavs_nonspeech/, arioso_mel/, val lists)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on chunks scored per val list")
    ap.add_argument("--out", default=None, help="output dir (default: <run-dir>/ab)")
    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
