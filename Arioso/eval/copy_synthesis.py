"""Step-0 sanity: real violin -> Section-3 mel -> frozen BigVGAN-v2 -> audio (Section 10.0).

Establishes the vocoder-plus-config ceiling, independent of Arioso. **Run this before any model
training.** If copy-synthesis is not near-transparent, the mel config or distribution is wrong and
no model work will fix it. Loading the vocoder also triggers ``common.vocoder``'s assertion that
the mel contract matches the checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch

from common.audio_io import load_mono, write_pcm16
from common.vocoder import load_vocoder, mel_spectrogram, vocode
from DataSynthesizer.config import DEFAULT_OUT

from ..config import SAMPLES_DIR, AriosoConfig
from ..splits import make_split


def _first_val_gt(out_dir: str) -> str:
    """Path to a held-out (val) GT wav, so the sanity is on a piece the model never sees."""
    split = make_split(out_dir)
    val = set(split["val"])
    with open(os.path.join(out_dir, "manifest.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") == "ok" and r["basename"] in val:
                return r["gt_path"]
    raise RuntimeError("no held-out GT clip found (run build_prior / splits first)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--input", help="GT wav (default: first held-out clip)")
    ap.add_argument("--out", default=os.path.join(SAMPLES_DIR, "copysynth.wav"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = AriosoConfig()
    gt_path = args.input or _first_val_gt(args.out_dir)
    print(f"copy-synthesis on: {gt_path}")

    y = load_mono(gt_path)
    mel = mel_spectrogram(y)                                  # [1, 128, T], Section-3 front-end
    voc = load_vocoder(device=args.device)                   # asserts mel contract vs checkpoint
    audio = vocode(voc, mel.to(args.device))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_pcm16(args.out, audio)

    n = min(len(y), len(audio))
    rms = float(np.sqrt(np.mean((y[:n] - audio[:n]) ** 2)))
    print(f"wrote {args.out}  ({len(audio) / cfg.sr:.1f} s)  waveform RMS error {rms:.4f}\n"
          f"Listen: copy-synthesis should be near-transparent (this is the vocoder ceiling).")


if __name__ == "__main__":
    main()
