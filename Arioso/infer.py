"""Arioso inference: score -> prior -> Euler-integrated mel -> audio (Section 9).

The prior is built from the input score **exactly as in training** (``DataSynthesizer.synthesizePrior``
via ``quantized_prior``: band-limited saw + masked-RMS to the fixed target level + mel). Starting
from ``x = x_0`` at t=0, integrate the
ODE ``x <- x + dt * v_theta(x, x_0, t)`` with 16-32 Euler steps (no CFG, single forward/step).
Long sequences are processed in overlapping chunks with a linear crossfade. The mel is turned to
audio with the **frozen** BigVGAN-v2 vocoder (listening only — never a selection arbiter).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from common.vocoder import load_vocoder, vocode
from DataSynthesizer.config import (PRIOR_ANTI_ALIAS, PRIOR_ENVELOPE, PRIOR_LEVEL_MATCH,
                                    TARGET_RMS_DBFS)
from DataSynthesizer.features import mel_for_training
from DataSynthesizer.synthesizePrior import quantized_prior

from .config import SAMPLES_DIR, AriosoConfig
from .model import AriosoModel


def build_prior_mel(midi_path: str) -> np.ndarray:
    """Score -> prior mel ``[N_MELS, T]``, identical to the training-time prior (Section 9.1).

    Built by the same DataSynthesizer pipeline the dataset's ``prior_mel_arioso`` was, so the
    inference prior matches training. No GT-alignment shift here: the score's onsets *are* t=0.
    """
    synth = quantized_prior(anti_alias=PRIOR_ANTI_ALIAS, envelope=PRIOR_ENVELOPE,
                            level_match=PRIOR_LEVEL_MATCH, target_rms_dbfs=TARGET_RMS_DBFS)
    return mel_for_training(synth.render(midi_path))


@torch.no_grad()
def integrate(model: AriosoModel, x0: torch.Tensor, cfg: AriosoConfig) -> torch.Tensor:
    """Euler-integrate the velocity field from t=0 (x=x0) to t=1. ``x0``: [1, 128, T]."""
    x = x0.clone()
    dt = 1.0 / cfg.euler_steps
    for i in range(cfg.euler_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        x = x + dt * model(x, x0, t)
    return x


@torch.no_grad()
def generate_mel(model: AriosoModel, prior_mel: np.ndarray, cfg: AriosoConfig,
                 device: str) -> np.ndarray:
    """Run the ODE over the whole prior mel, chunking long sequences with a linear crossfade."""
    x0 = torch.from_numpy(prior_mel[None]).float().to(device)   # [1, 128, T]
    t_total = x0.shape[-1]
    chunk, overlap = cfg.chunk_frames, cfg.overlap_frames

    if t_total <= chunk:
        return integrate(model, x0, cfg)[0].cpu().numpy()

    out = np.zeros((prior_mel.shape[0], t_total), dtype=np.float32)
    weight = np.zeros(t_total, dtype=np.float32)
    step = chunk - overlap
    for start in range(0, t_total, step):
        end = min(start + chunk, t_total)
        seg = integrate(model, x0[:, :, start:end], cfg)[0].cpu().numpy()
        w = np.ones(end - start, dtype=np.float32)
        if start > 0:                                            # fade-in over the overlap
            w[:overlap] = np.linspace(0.0, 1.0, overlap, endpoint=False)
        out[:, start:end] += seg * w
        weight[start:end] += w
        if end == t_total:
            break
    return out / np.maximum(weight, 1e-6)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("midi", help="input score (.mid)")
    ap.add_argument("-o", "--out", default=os.path.join(SAMPLES_DIR, "arioso_out.wav"))
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt (uses EMA weights)")
    ap.add_argument("--weights", choices=("ema", "model"), default="ema")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-mel", help="optional .npy path for the generated mel")
    args = ap.parse_args()

    cfg = AriosoConfig()
    model = AriosoModel(cfg).to(args.device)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(ckpt[args.weights])
    model.eval()

    prior_mel = build_prior_mel(args.midi)
    print(f"prior mel: {prior_mel.shape}  ({prior_mel.shape[-1] / cfg.sr * cfg.hop:.1f} s)")
    mel = generate_mel(model, prior_mel, cfg, args.device)
    if args.save_mel:
        np.save(args.save_mel, mel)

    voc = load_vocoder(device=args.device)                      # frozen; also asserts mel contract
    audio = vocode(voc, torch.from_numpy(mel[None]).float())
    from common.audio_io import write_pcm16
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_pcm16(args.out, audio)
    print(f"wrote {args.out}  ({len(audio) / cfg.sr:.1f} s)")


if __name__ == "__main__":
    main()
