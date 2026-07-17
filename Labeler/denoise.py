"""Denoise a recording: high-pass -> spectral noise reduction -> voiced-RMS level.

The cleaned wav is the default ground truth + playback source in the editor. The
chain is:

  1. a 4th-order Butterworth **high-pass** at 120 Hz (zero-phase ``sosfiltfilt``)
     to drop room rumble / handling / HVAC below the violin's range (open G ~196 Hz);
  2. **noisereduce** in non-stationary mode (``prop_decrease`` 0.85, a 2 s time
     constant) to suppress broadband hiss while tracking a slowly changing floor; and
  3. ``common.audio_io.voiced_rms_normalize`` so the cleaned clip lands at the same
     -20 dBFS voiced-RMS convention as the rest of the project (the align prior is
     level-matched to that same target).

All parameters come from :class:`Labeler.config.DenoiseParams`.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from common.audio_io import voiced_rms_normalize
from common.config import SR

from .config import DenoiseParams


def highpass(y: np.ndarray, sr: int, cutoff: float, order: int) -> np.ndarray:
    """Zero-phase Butterworth high-pass (sosfiltfilt) at ``cutoff`` Hz."""
    sos = butter(order, cutoff, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def denoise(y: np.ndarray, params: DenoiseParams | None = None,
            sr: int = SR) -> np.ndarray:
    """Return the cleaned, level-normalized waveform for ``y`` (mono float @ ``sr``)."""
    import noisereduce as nr  # local: keep module import light

    params = params or DenoiseParams()
    y = np.asarray(y, dtype=np.float32)
    y = highpass(y, sr, params.hpf_hz, params.hpf_order)
    reduced = nr.reduce_noise(
        y=y, sr=sr, stationary=False,
        prop_decrease=params.prop_decrease,
        n_fft=params.n_fft, hop_length=params.hop_length,
        time_constant_s=params.time_constant_s,
        freq_mask_smooth_hz=params.freq_mask_smooth_hz,
        time_mask_smooth_ms=params.time_mask_smooth_ms)
    reduced = np.asarray(reduced, dtype=np.float32)
    return voiced_rms_normalize(reduced, sr=sr).astype(np.float32)
