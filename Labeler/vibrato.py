"""Detect vibrato from a note's per-frame pitch-bend (cents) trajectory.

MUSC emits a pitch-bend array per note (sampled at the model's ~172 fps). Vibrato
is a sustained ~4-9 Hz oscillation of the pitch; a slide or intonation drift is a
low-frequency trend and must NOT count. The detector:

  1. skip notes shorter than ``min_dur_s`` (0.3 s) — too short to sustain vibrato;
  2. keep the central ``middle_fraction`` (80%) of the note (drop attack/release);
  3. **linear-detrend** to remove slides/glissandi;
  4. take the windowed spectrum of the detrended segment and require most of its
     energy (``band_frac_min``, 40%) to sit in the 4-9 Hz band — this rejects a
     sub-4-Hz drift (whose energy is out of band) even though a band peak (rate)
     always exists, and it is the "sustained oscillation" evidence;
  5. call it vibrato iff the detrended oscillation has peak-to-peak
     ``extent >= 12 cents`` AND the in-band energy fraction clears ``band_frac_min``
     AND the in-band peak (the ``rate``) sits in 4-9 Hz.

Spectral band-energy (not a short-signal band-pass filter, and not a fixed
per-frame cents threshold) is the discriminator because real notes carry only a
few vibrato cycles at ~±8-10 cents — zero-phase filtering eats the amplitude and a
"fraction of frames above N cents" test needs many clean cycles to fire, so both
miss genuine short-note vibrato that band-energy detects. Returns the boolean plus
the measured ``rate_hz`` / ``extent_cents`` (kept in the note's ``auto`` block).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MODEL_FPS, VibratoParams


@dataclass
class VibratoResult:
    is_vibrato: bool
    rate_hz: float
    extent_cents: float


def bend_ints_to_cents(bend_ints) -> np.ndarray:
    """Convert MUSC MIDI pitch-bend integers to cents (x100/4096)."""
    return np.asarray(bend_ints, dtype=np.float64) * 100.0 / 4096.0


def detect_vibrato(bend_cents, dur_s: float, fps: float = MODEL_FPS,
                   params: VibratoParams | None = None) -> VibratoResult:
    """Analyze a per-frame bend-cents trajectory for sustained 4-9 Hz vibrato."""
    from scipy.signal import detrend

    params = params or VibratoParams()
    arr = np.asarray(bend_cents, dtype=np.float64)
    n = arr.size
    if dur_s < params.min_dur_s or n < 12:
        return VibratoResult(False, 0.0, 0.0)

    drop = (1.0 - params.middle_fraction) / 2.0
    i0 = int(np.floor(drop * n))
    i1 = int(np.ceil((1.0 - drop) * n))
    seg = arr[i0:i1]
    if seg.size < 12:
        return VibratoResult(False, 0.0, 0.0)

    seg = detrend(seg, type="linear")

    # Windowed spectrum of the detrended segment.
    win = np.hanning(seg.size)
    spec = np.abs(np.fft.rfft(seg * win))
    freqs = np.fft.rfftfreq(seg.size, d=1.0 / fps)
    valid = freqs > 0.5  # ignore DC / detrend residual for the energy ratio
    band = (freqs >= params.band_low_hz) & (freqs <= params.band_high_hz)
    if not band.any() or not valid.any():
        return VibratoResult(False, 0.0, 0.0)

    power = spec ** 2
    total = float(power[valid].sum())
    if total <= 0:
        return VibratoResult(False, 0.0, 0.0)
    band_frac = float(power[band].sum() / total)
    rate = float(freqs[band][int(np.argmax(spec[band]))])

    # Peak-to-peak extent of the detrended oscillation (~zero mean).
    extent = float(np.percentile(seg, 95) - np.percentile(seg, 5))

    is_vib = (extent >= params.extent_min_cents
              and band_frac >= params.band_frac_min
              and params.band_low_hz <= rate <= params.band_high_hz)
    return VibratoResult(bool(is_vib), rate, extent)
