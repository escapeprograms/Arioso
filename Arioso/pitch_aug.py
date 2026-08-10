"""Dataset-level augmentation — on-the-fly pitch shift of the prior/target pair.

Arioso now has **two** augmentation seams, and they never overlap:

  1. ``Arioso.augment`` — **tensor / step level**: post-batch transforms and auxiliary losses
     applied *inside the training step*, on tensors already on the device. No disk access, no
     waveforms, no per-sample RNG stream (the batch RNG is torch's).
  2. **this module** — **dataset level**: applied *inside* ``AriosoDataset.__getitem__``, before
     collation, working from the **waveforms on disk** (``prior_wav/`` + ``gt/``) rather than the
     precomputed mels. Anything that must change what a *sample* is (rather than how a batch is
     transformed) belongs here; future dataset-level augmentations extend this module.

The shift itself is :func:`common.keyshift.mel_frames_keyshift` — a DiffSinger nvSTFT keyshift,
duration-preserving, so the augmented mels stay frame-for-frame aligned with the score-rasterized
conditioning tracks and the clip's ``[start, end)`` framing is untouched. Both tracks of a sample
take the **same** shift, so the prior->target relation the model learns is preserved.

Everything here is CPU-pure numpy + soundfile, safe to run in dataloader workers. The RNG stream is
seeded from ``(seed, epoch, index)`` only — never from worker/process state — so ``num_workers``
does not change a single draw and a re-run of the same config reproduces the same shifts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from common.config import SR
from common.dataset_schema import DatasetRoot
from common.keyshift import margin_frames, mel_frames_keyshift, slice_read_range

from .config import AriosoConfig


@dataclass(frozen=True)
class PitchAug:
    """The pitch-shift augmentation policy for one split — draws the shift, nothing else.

    Fields mirror the config knobs ``pitch_aug_p`` / ``pitch_aug_cents`` plus the run ``seed``.
    Built once by :func:`build_pitch_aug` and carried by the dataset; frozen so a worker process
    can only ever hold an identical copy.
    """
    p: float
    max_cents: float
    seed: int

    @property
    def active(self) -> bool:
        """True iff a shift can actually happen. When False **no RNG is drawn and no wav is
        touched**, so an unaugmented run's items are byte-identical to the memmap path."""
        return self.p > 0.0 and self.max_cents > 0.0

    @property
    def margin(self) -> int:
        """Context frames a shifted slice needs on each side (``common.keyshift.margin_frames``)."""
        return margin_frames(self.max_cents)

    def cents_for(self, epoch: int, index: int) -> float | None:
        """The shift for sample ``index`` in ``epoch``, or ``None`` when this sample isn't shifted.

        The generator is seeded from ``(self.seed, epoch, index)`` and **nothing else** — not the
        worker id, not the process RNG state — so the stream is identical for any ``num_workers``
        and an exact re-run of the config reproduces every draw. The draw order is fixed: the
        Bernoulli(``p``) gate first, then (only if it passes) the uniform cents, so changing
        ``max_cents`` never re-rolls *which* samples are augmented.
        """
        if not self.active:
            return None                      # inactive: no generator constructed at all
        rng = np.random.default_rng([self.seed, epoch, index])
        if rng.random() >= self.p:
            return None
        return float(rng.uniform(-self.max_cents, self.max_cents))


def wavs_available(root: DatasetRoot, base: str) -> bool:
    """True iff ``base`` has BOTH source waveforms in ``root`` (``prior_wav/`` and ``gt/``).

    The graceful-fallback gate: roots predating ``prior_wav/`` (the synthetic ``Data/`` root) simply
    never get augmented instead of failing the run. Callers cache the answer per
    ``(root.path, base)`` — this is two ``os.path.isfile`` calls per clip otherwise.
    """
    return (os.path.isfile(root.prior_wav_path(base))
            and os.path.isfile(root.gt_path(base)))


def shifted_pair(root: DatasetRoot, base: str, start: int, end: int,
                 cents: float, margin: int) -> tuple[np.ndarray, np.ndarray]:
    """Re-mel ``base``'s prior and target frames ``[start, end)`` at a ``cents`` pitch shift.

    Both tracks go through the **identical** path — ``slice_read_range`` -> a *ranged*
    ``soundfile.read`` (never :func:`common.audio_io.load_mono`, which would decode the whole
    recording for a 10 s clip) -> :func:`common.keyshift.mel_frames_keyshift` -> drop the
    ``head_frames`` of margin context. A short read at EOF is fine: the slice then ends at the true
    file end, where its own reflect pad *is* the whole-file one (see ``slice_read_range``).

    Returns ``(x0, x1)``, each ``[N_MELS, end - start]`` float32 — prior from
    ``prior_wav/<base>.wav``, target from ``gt/<base>.wav``.
    """
    read_start, read_n, head = slice_read_range(start, end, margin)
    want = end - start

    def _mel(path: str) -> np.ndarray:
        wav, sr = sf.read(path, start=read_start, frames=read_n, dtype="float32")
        assert sr == SR, f"{path}: sample rate {sr} != {SR} (a root's wavs are never resampled)"
        mel = mel_frames_keyshift(wav, cents)[:, head:head + want]
        if mel.shape[1] != want:
            # Impossible by construction (the frame count is cents-invariant and the read range is
            # exact) — loud rather than silently padded, which would desync cond from mel.
            raise ValueError(
                f"pitch-aug framing bug: {path} base={base!r} frames [{start}, {end}) gave "
                f"{mel.shape[1]} frames, expected {want}")
        return mel

    return _mel(root.prior_wav_path(base)), _mel(root.gt_path(base))


def build_pitch_aug(cfg: AriosoConfig, split_name: str) -> PitchAug:
    """The single factory ``build_dataloader`` calls — the split's pitch-shift policy.

    Any split other than ``"train"`` gets a **structurally inactive** policy (``p=0.0``), so val can
    never be augmented no matter what the config says: the val metric always scores the real mels.
    """
    if split_name != "train":
        return PitchAug(p=0.0, max_cents=cfg.pitch_aug_cents, seed=cfg.seed)
    return PitchAug(p=cfg.pitch_aug_p, max_cents=cfg.pitch_aug_cents, seed=cfg.seed)
