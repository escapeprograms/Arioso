"""Shared audio I/O helpers for the Arioso project.

Wraps the idioms every package repeats — loading a file to a mono array at a
fixed sample rate, writing a 16-bit PCM wav, and peak-normalizing — so the
DataSynthesizer pipeline and the training code share one implementation.
"""

from __future__ import annotations

import numpy as np
import librosa
import soundfile as sf

from common.config import DEFAULT_PEAK, PCM_SUBTYPE, SR


def load_mono(path: str, sr: int = SR) -> np.ndarray:
    """Load ``path`` as a mono float array resampled to ``sr``."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def write_pcm16(path: str, y: np.ndarray, sr: int = SR) -> str:
    """Write ``y`` to ``path`` as a 16-bit PCM wav; return ``path``."""
    sf.write(path, y, sr, subtype=PCM_SUBTYPE)
    return path


def normalize(y: np.ndarray, target_peak: float = DEFAULT_PEAK) -> np.ndarray:
    """Scale ``y`` so its peak is ``target_peak`` (no-op on silence).

    Summed/overlapping sawtooth notes can land well above 1.0; rescaling to a
    fixed peak keeps the prior from clipping.
    """
    peak = float(np.max(np.abs(y)))
    if peak > 0.0:
        y = y * (target_peak / peak)
    return y


def voiced_rms_normalize(y: np.ndarray, sr: int = SR, target_rms_dbfs: float = -20.0,
                         top_db: float = 40.0,
                         peak_ceiling: float = DEFAULT_PEAK) -> np.ndarray:
    """Scale ``y`` so its RMS over voiced (non-silent) segments hits ``target_rms_dbfs``.

    The targets are scraped from many sources at different volumes. Measuring RMS
    only inside the non-silent intervals (``librosa.effects.split``) — where the
    instrument is actually playing — keeps rests from dragging the level down, so a
    single global gain equalizes the playing loudness across tracks. A final peak
    guard scales back down if the gain would otherwise clip (rare, very quiet src).
    """
    peak = float(np.max(np.abs(y)))
    if peak <= 0.0:
        return y  # all silence -> nothing to normalize
    # Detect voiced segments on a peak-normalized copy so the silence threshold is
    # scale-invariant: split's dB floor would otherwise fail to exclude true silence
    # in a globally-quiet track, dragging the RMS down and over-boosting it.
    intervals = librosa.effects.split(y / peak, top_db=top_db)
    if intervals.size == 0:
        return y
    voiced = np.concatenate([y[s:e] for s, e in intervals])
    rms = float(np.sqrt(np.mean(voiced ** 2)))
    if rms <= 0.0:
        return y
    y = y * (10.0 ** (target_rms_dbfs / 20.0)) / rms
    new_peak = float(np.max(np.abs(y)))
    if new_peak > peak_ceiling:
        y = y * (peak_ceiling / new_peak)
    return y.astype(np.float32)
