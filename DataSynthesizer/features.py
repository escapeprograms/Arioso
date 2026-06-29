"""Training-feature builders for the DataSynthesizer: mels + onset mask.

Keeps ``build_dataset`` thin. Two pieces:

* ``mel_for_training`` — turn an aligned waveform into the BigVGAN-compatible mel
  that the vocoder and training expect (the project's single source of truth is
  ``common.vocoder.mel_spectrogram``).
* ``build_onset_mask`` — turn the score onsets into the onset-mask training signal
  at the same mel granularity: 1 on each onset frame, exponential decay to ~0 over
  a short support window.
"""

from __future__ import annotations

import numpy as np

from common.vocoder import mel_spectrogram

from .config import HOP, ONSET_DECAY_FLOOR, ONSET_DECAY_MS, SR


def mel_for_training(wav: np.ndarray) -> np.ndarray:
    """BigVGAN mel of ``wav`` as a ``[N_MELS, T]`` float32 array.

    Uses ``common.vocoder.mel_spectrogram`` (which matches the vocoder checkpoint),
    then drops the leading batch dim and moves to numpy.
    """
    return mel_spectrogram(wav)[0].cpu().numpy().astype(np.float32)


def build_onset_mask(onset_times, applied: float, n_frames: int, sr: int = SR,
                     hop: int = HOP, decay_ms: float = ONSET_DECAY_MS,
                     floor: float = ONSET_DECAY_FLOOR) -> np.ndarray:
    """Onset-mask training signal aligned to the prior mel grid.

    Each onset spikes to 1 on its frame and decays exponentially to ``floor`` over a
    support window of ``decay_ms`` (hard 0 beyond it). ``onset_times`` (seconds, from
    the score) are shifted by ``applied`` — the same offset applied to the prior — so
    the mask lines up with the aligned prior mel. Overlapping decays combine with max.

    Returns a ``[n_frames]`` float32 array in [0, 1].
    """
    mask = np.zeros(n_frames, dtype=np.float32)
    onset_times = np.asarray(onset_times, dtype=np.float64)
    if onset_times.size == 0:
        return mask

    # Decay kernel: kernel[k] is the mask value k frames after an onset, within the
    # window. floor**(dt/window) == exp(-dt/tau) with tau set so the value reaches
    # `floor` at dt == window; kernel[0] == 1.
    window_s = decay_ms / 1000.0
    if window_s <= 0.0:
        kernel = np.ones(1, dtype=np.float32)
    else:
        n_k = int(np.floor(window_s * sr / hop)) + 1
        dt = np.arange(n_k) * hop / sr
        kernel = (float(floor) ** (dt / window_s)).astype(np.float32)
    klen = len(kernel)

    for t in onset_times:
        f0 = int(round((t + applied) * sr / hop))
        if f0 >= n_frames:
            continue
        lo = max(f0, 0)
        hi = min(f0 + klen, n_frames)
        if hi <= lo:
            continue
        k_lo = lo - f0  # skip the kernel head if the onset sits before frame 0
        seg = kernel[k_lo:k_lo + (hi - lo)]
        np.maximum(mask[lo:hi], seg, out=mask[lo:hi])
    return mask
