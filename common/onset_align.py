"""Onset alignment for the (prior, GT) pairs — the pipeline's step 3.

The dataset MIDI is already roughly time-aligned to the audio, but a small
residual global offset can remain. This module measures that offset from the
note onsets and shifts the prior to remove it — all on in-memory arrays:

  * ``estimate_offset_seconds`` cross-correlates the onset-strength envelopes of
    a (prior, GT) pair to find the residual global time offset,
  * ``shift_samples`` shifts a waveform in time (pad/truncate, same length), and
  * ``align_prior_to_gt`` ties them together: estimate, then shift the prior so it
    lines up with the GT, returning the aligned prior and the applied offset.

``build_dataset`` estimates the offset once from the quantized prior and applies the
same ``shift_samples`` to both prior variants, so they stay mutually aligned.

The onset method was manually verified on enough clips to trust as the automatic
alignment. (QC visualizations — mel-spectrogram + onset-envelope overlays — live
in ``visualizations.ipynb``.)

Example::

    python -m DataSynthesizer.onset_align prior.wav gt.wav            # report offset
    python -m DataSynthesizer.onset_align prior.wav gt.wav --apply    # write aligned prior
"""

from __future__ import annotations

import numpy as np
import librosa

from common.audio_io import load_mono, write_pcm16

from .config import HOP, SR


def estimate_offset_seconds(prior: np.ndarray, gt: np.ndarray, sr: int = SR,
                            max_lag_s: float = 1.0) -> float:
    """Estimate the global time offset between prior and GT.

    Cross-correlates their onset-strength envelopes. A positive return value
    means the prior's events arrive *later* than the GT's (prior lags), i.e.
    the prior should be shifted earlier by that amount to align.
    """
    op = librosa.onset.onset_strength(y=prior, sr=sr, hop_length=HOP)
    og = librosa.onset.onset_strength(y=gt, sr=sr, hop_length=HOP)
    n = min(len(op), len(og))
    op, og = op[:n], og[:n]
    op = (op - op.mean()) / (op.std() + 1e-9)
    og = (og - og.mean()) / (og.std() + 1e-9)
    corr = np.correlate(op, og, mode="full")
    lags = np.arange(-n + 1, n)
    max_lag_frames = int(round(max_lag_s * sr / HOP))
    keep = np.abs(lags) <= max_lag_frames
    best_lag = lags[keep][np.argmax(corr[keep])]
    return best_lag * HOP / sr


def shift_samples(y: np.ndarray, offset_seconds: float, sr: int = SR) -> np.ndarray:
    """Shift ``y`` in time by ``offset_seconds``, preserving length (pad/truncate).

    Positive ``offset_seconds`` delays the signal (pads silence at the front and
    truncates the tail); negative advances it (drops the front, pads the tail).
    """
    shift = int(round(offset_seconds * sr))
    if shift > 0:
        return np.concatenate([np.zeros(shift, y.dtype), y])[: len(y)]
    if shift < 0:
        return np.concatenate([y[-shift:], np.zeros(-shift, y.dtype)])
    return y


def align_prior_to_gt(prior: np.ndarray, gt: np.ndarray,
                      sr: int = SR) -> tuple[np.ndarray, float]:
    """Estimate the (prior, GT) onset offset and shift the prior to correct it.

    ``estimate_offset_seconds`` is positive when the prior lags, so we apply its
    negation to advance the prior into alignment. Returns ``(aligned_prior,
    applied)`` where ``applied`` is the shift in seconds (negative ⇒ prior advanced).
    """
    applied = -estimate_offset_seconds(prior, gt, sr)
    return shift_samples(prior, applied, sr), applied


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Onset-align a (prior, GT) pair: report the offset, or apply it.")
    ap.add_argument("prior", help="path to the prior wav")
    ap.add_argument("gt", help="path to the GT wav")
    ap.add_argument("--apply", action="store_true",
                    help="write the prior shifted into alignment (default: report only)")
    ap.add_argument("-o", "--out",
                    help="output wav for --apply (default: overwrite the prior)")
    args = ap.parse_args()

    prior, gt = load_mono(args.prior), load_mono(args.gt)
    if args.apply:
        aligned, applied = align_prior_to_gt(prior, gt)
        out = args.out or args.prior
        write_pcm16(out, aligned)
        print(f"aligned prior by {applied * 1000:+.0f} ms -> {out}")
        return

    offset = estimate_offset_seconds(prior, gt)
    print(f"estimated prior-vs-GT offset: {offset * 1000:+.0f} ms "
          f"({'prior lags' if offset > 0 else 'prior leads'})")


if __name__ == "__main__":
    main()
