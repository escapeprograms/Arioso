"""Regenerate Arioso's prior features over the built dataset (one-time pass).

The DataSynthesizer ``data/`` tree already has GT audio, ``target_mel`` and a ``manifest.csv``.
This script adds the two Arioso-specific outputs the model trains on, without re-downloading:

* ``data/prior_mel_arioso/<base>.npy`` — ``[N_MELS, T]`` float32 prior mel from the spec-faithful
  prior (anti-aliased saw + masked-RMS match), frame-aligned to ``target_mel`` (same ``T``).
* ``data/onsets_arioso/<base>.npy`` — ``[K]`` int32 aligned onset frame indices, used by the clip
  enumerator (Section 5).

It reuses the manifest's per-clip ``offset_ms`` (estimated once at build time from the onset
cross-correlation) so the prior lines up with the target exactly as the original prior did — no
re-estimation. Resumable + skip-existing, mirroring ``DataSynthesizer.build_dataset``.

Run::

    python -m Arioso.build_prior --limit 4      # smoke test
    python -m Arioso.build_prior                # full pass over status==ok clips
"""

from __future__ import annotations

import csv
import os
import sys
import traceback

import numpy as np

from DataSynthesizer.config import DEFAULT_DATASET, DEFAULT_OUT
from DataSynthesizer.features import mel_for_training
from DataSynthesizer.onset_align import shift_samples
from DataSynthesizer.synthesizePrior import note_onsets

from .config import ONSETS_DIR, PRIOR_MEL_DIR, AriosoConfig
from .prior import _sounding_mask, level_match, render_prior


def _midi_path(dataset_root: str, book: str, basename: str) -> str:
    return os.path.join(dataset_root, book, basename + ".mid")


def process_clip(row: dict, out_dir: str, dataset_root: str, cfg: AriosoConfig,
                 overwrite: bool = False) -> str:
    """Build prior mel + onset frames for one manifest row. Returns a status string."""
    base = row["basename"]
    prior_dir = os.path.join(out_dir, PRIOR_MEL_DIR)
    onset_dir = os.path.join(out_dir, ONSETS_DIR)
    os.makedirs(prior_dir, exist_ok=True)
    os.makedirs(onset_dir, exist_ok=True)
    prior_mel_path = os.path.join(prior_dir, base + ".npy")
    onset_path = os.path.join(onset_dir, base + ".npy")

    if not overwrite and os.path.isfile(prior_mel_path) and os.path.isfile(onset_path):
        return "exists"

    midi = _midi_path(dataset_root, row["book"], base)
    n_samples = int(row["n_samples"])            # == len(GT); no need to reload the audio
    applied = float(row["offset_ms"]) / 1000.0   # same shift the original prior used

    # Render -> masked-RMS level-match (on the score-aligned, unshifted prior) -> shift
    # into GT alignment. The gain is scale-invariant to the shift, so order is safe.
    prior = render_prior(midi, cfg, total_samples=n_samples)
    sounding = _sounding_mask(midi, n_samples)
    prior, _gain = level_match(prior, sounding, cfg)
    prior = shift_samples(prior, applied)

    prior_mel = mel_for_training(prior)
    np.save(prior_mel_path, prior_mel)

    n_frames = prior_mel.shape[-1]
    onsets = note_onsets(midi) + applied
    frames = np.round(onsets * cfg.sr / cfg.hop).astype(np.int64)
    frames = np.unique(frames[(frames >= 0) & (frames < n_frames)]).astype(np.int32)
    np.save(onset_path, frames)
    return "ok"


def build(out_dir: str = DEFAULT_OUT, dataset_root: str = DEFAULT_DATASET,
          cfg: AriosoConfig | None = None, limit: int | None = None,
          overwrite: bool = False) -> None:
    """Pass over ``manifest.csv`` (status==ok rows), building Arioso prior features."""
    cfg = cfg or AriosoConfig()
    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "ok"]
    if limit:
        rows = rows[:limit]

    n_ok = n_skip = n_fail = 0
    for i, row in enumerate(rows, 1):
        base = row["basename"]
        try:
            status = process_clip(row, out_dir, dataset_root, cfg, overwrite=overwrite)
            n_ok += status == "ok"
            n_skip += status == "exists"
            print(f"[{i}/{len(rows)}] {status:8s} {base}")
        except Exception as exc:  # noqa: BLE001 - log and continue, like build_dataset
            n_fail += 1
            print(f"[{i}/{len(rows)}] FAILED   {base}: {exc}", file=sys.stderr)
            traceback.print_exc(limit=1)

    print(f"\nDone: {n_ok} built, {n_skip} existed, {n_fail} failed -> "
          f"{os.path.join(out_dir, PRIOR_MEL_DIR)}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET)
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N ok clips (smoke testing)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-render even if outputs already exist")
    # Prior toggles (default = spec baseline) so ablations don't need code edits.
    ap.add_argument("--no-anti-alias", action="store_true",
                    help="use the naive aliased sawtooth instead of polyBLEP")
    ap.add_argument("--envelope", choices=("rect", "fade"), default="rect")
    ap.add_argument("--level-match", choices=("masked_rms", "peak"), default="masked_rms")
    args = ap.parse_args()

    cfg = AriosoConfig(anti_alias=not args.no_anti_alias, envelope=args.envelope,
                       level_match=args.level_match)
    build(out_dir=args.out_dir, dataset_root=args.dataset_root, cfg=cfg,
          limit=args.limit, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
