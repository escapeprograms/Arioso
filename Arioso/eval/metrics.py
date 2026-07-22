"""Vocoder-independent metrics + Delta-mel diagnostics (Section 10).

Model selection runs on these, never on vocoded audio:

* **reconstruction MSE** — ``mean((pred_mel - target_mel)^2)`` on held-out pieces, reported
  alongside the raw transport baseline ``mean((prior_mel - target_mel)^2)``.
* **MCD** — mel-cepstral distortion: DCT-II of the (already log-) mel, drop c0, the standard
  ``(10/ln10) * sqrt(2 * sum_k dc_k^2)`` per frame, averaged. Reported for pred and prior.
* **Delta-target-mel plots** — ``(pred - target)`` and ``(prior - target)`` heatmaps. The latter
  is the raw transport problem; the former is residual error. Primary qualitative diagnostic:
  expect the largest residual in the upper harmonics (the un-removed sawtooth excess) — exactly
  the signal that motivates body EQ next.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from scipy.fft import dct

from common.dataset_schema import DatasetRoot

from ..config import BoundaryCondSpec, cfg_from_dict
from ..dataset import boundary_distances
from ..infer import generate_mel
from ..model import AriosoModel
from ..splits import make_split

_MCD_K = 24          # number of mel-cepstral coefficients (after dropping c0)
_MCD_COEF = 10.0 / np.log(10.0)


def mcd(a: np.ndarray, b: np.ndarray) -> float:
    """Mel-cepstral distortion (dB) between two ``[n_mels, T]`` log-mels over matched frames."""
    t = min(a.shape[-1], b.shape[-1])
    ca = dct(a[:, :t], type=2, norm="ortho", axis=0)[1:_MCD_K + 1]
    cb = dct(b[:, :t], type=2, norm="ortho", axis=0)[1:_MCD_K + 1]
    per_frame = np.sqrt(2.0 * np.sum((ca - cb) ** 2, axis=0))
    return float(_MCD_COEF * per_frame.mean())


def _val_basenames(root: DatasetRoot, limit: int | None) -> list[str]:
    split = make_split(root)
    bases = split["val"]
    return bases[:limit] if limit else bases


def _delta_plot(pred: np.ndarray, prior: np.ndarray, target: np.ndarray, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = min(pred.shape[-1], target.shape[-1])
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for a, (title, delta) in zip(ax, [("pred - target", pred[:, :t] - target[:, :t]),
                                      ("prior - target", prior[:, :t] - target[:, :t])]):
        im = a.imshow(delta, origin="lower", aspect="auto", cmap="coolwarm",
                      vmin=-2, vmax=2)
        a.set_title(title)
        a.set_ylabel("mel bin")
        fig.colorbar(im, ax=a)
    ax[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="Data",
                    help="dataset root to evaluate on (eval is single-root)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", choices=("ema", "model"), default="ema")
    ap.add_argument("--limit", type=int, default=8, help="number of held-out recordings to score")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--plot", help="path to save one Delta-mel figure (first recording)")
    args = ap.parse_args()

    # Build the model from the checkpoint's embedded config (pre-conditioning ckpts -> old arch).
    ckpt = torch.load(args.ckpt, map_location=args.device)
    cfg = cfg_from_dict(ckpt.get("cfg") or {})
    model = AriosoModel(cfg).to(args.device)
    model.load_state_dict(ckpt[args.weights])
    model.eval()

    root = DatasetRoot(args.data_root)
    recon_mse, prior_mse, mcd_pred, mcd_prior, n = 0.0, 0.0, 0.0, 0.0, 0
    for i, base in enumerate(_val_basenames(root, args.limit)):
        pp, tp = root.prior_mel_path(base), root.target_mel_path(base)
        if not (os.path.isfile(pp) and os.path.isfile(tp)):
            continue
        prior = np.load(pp).astype(np.float32)
        target = np.load(tp).astype(np.float32)

        # Per-frame conditioning tracks on the prior mel's frame grid (no slicing before
        # generate_mel). Categorical: a signal this root lacks gets the CFG "unknown" fill
        # (spec.num_classes), mirroring the training dataset's unknown-fill so conditioned eval on an
        # unlabelled root still runs; on-disk tracks are read for signals the root provides. Boundary
        # (mirrors Arioso.dataset:110-120): the root's onsets/offsets array over the full-recording
        # window -> per-frame distance; a missing array yields the all -1 (inactive) sentinel track.
        cond_frames = {}
        for spec in cfg.conditioning:
            if isinstance(spec, BoundaryCondSpec):
                path = (root.onsets_path(base) if spec.boundary == "onset"
                        else root.offsets_path(base))
                if not os.path.isfile(path):
                    cond_frames[spec.name] = np.full(prior.shape[-1], -1, dtype=np.int64)
                else:
                    bounds = np.load(path).astype(np.int64)
                    cond_frames[spec.name] = boundary_distances(
                        bounds, 0, prior.shape[-1], spec.direction)
            elif spec.name not in root.signals:
                cond_frames[spec.name] = np.full(prior.shape[-1], spec.num_classes, dtype=np.int64)
            else:
                cond_frames[spec.name] = np.load(root.cond_path(spec.name, base))[:prior.shape[-1]]
        cond_frames = cond_frames or None

        pred = generate_mel(model, prior, cfg, args.device,
                            cond_frames=cond_frames).astype(np.float32)
        t = min(pred.shape[-1], target.shape[-1])
        recon_mse += float(np.mean((pred[:, :t] - target[:, :t]) ** 2))
        prior_mse += float(np.mean((prior[:, :t] - target[:, :t]) ** 2))
        mcd_pred += mcd(pred, target)
        mcd_prior += mcd(prior, target)
        n += 1
        if args.plot and i == 0:
            _delta_plot(pred, prior, target, args.plot)

    if n == 0:
        print("no held-out recordings scored (run build_prior first).")
        return
    print(f"held-out recordings scored: {n}\n"
          f"  recon MSE  (pred  vs target): {recon_mse / n:.5f}\n"
          f"  transport MSE (prior vs target): {prior_mse / n:.5f}\n"
          f"  MCD        (pred  vs target): {mcd_pred / n:.3f} dB\n"
          f"  MCD        (prior vs target): {mcd_prior / n:.3f} dB")
    if args.plot:
        print(f"  Delta-mel figure: {args.plot}")


if __name__ == "__main__":
    main()
