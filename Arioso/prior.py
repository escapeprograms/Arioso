"""Arioso's informed prior — the spec-faithful sawtooth the velocity field transports from.

This is the single source of truth for the prior, reused **verbatim at train and inference**
(Section 4 / Section 9.1). It builds on ``DataSynthesizer.synthesizePrior`` but adds the two
spec-baseline changes that the dataset's ``prior_mel_quant`` lacks:

* **Anti-aliasing** — a polyBLEP band-limited sawtooth (toggle ``anti_alias``) instead of the
  naive ``scipy.signal.sawtooth`` that aliases on high notes. Anti-aliasing only removes the
  fold-back artifacts; the saw's native ~-6 dB/oct harmonic ladder (the "excess" the model must
  learn to remove) is kept on purpose.
* **Masked-RMS level match** — scale the whole prior by a single per-recording scalar so its RMS
  over *sounding* frames equals the target's (Section 4.3). The built ``prior_mel_quant`` is
  peak-normalized, which leaves an arbitrary prior/target level gap that derails OT-CFM transport.

Note gating is **rectangular** (hard on/off) by default per the build decision — ADSR is not
implemented this round. ``envelope="fade"`` reuses the 5 ms anti-click ramp as a toggle.

The prior is monophonic-summed across MIDI instrument tracks (reproducing double-stops), exactly
as ``render_prior``; pitch is quantized to the MIDI note number (bends ignored).
"""

from __future__ import annotations

import numpy as np
import pretty_midi
from scipy.signal import sawtooth

from common.config import SR
from DataSynthesizer.synthesizePrior import _fade_envelope, render_prior_bend

from .config import AriosoConfig


def _poly_blep(phase: np.ndarray, dt: float) -> np.ndarray:
    """polyBLEP residual for a sawtooth at fractional ``phase`` in [0, 1), step ``dt``.

    Subtracting this from the naive ramp band-limits the discontinuity at each wrap,
    removing most of the aliasing a hard sawtooth edge produces. ``dt = freq / sr`` is
    the per-sample phase increment (one period spans ``1/dt`` samples).
    """
    blep = np.zeros_like(phase)
    if dt <= 0.0:
        return blep
    # Just after the discontinuity (start of the period).
    lo = phase < dt
    t = phase[lo] / dt
    blep[lo] = t + t - t * t - 1.0
    # Just before the discontinuity (end of the period).
    hi = phase > 1.0 - dt
    t = (phase[hi] - 1.0) / dt
    blep[hi] = t * t + t + t + 1.0
    return blep


def _saw_blep(freq: float, n: int, sr: int) -> np.ndarray:
    """One band-limited (polyBLEP) sawtooth note of ``n`` samples at constant ``freq``.

    Matches the polarity/level of ``scipy.signal.sawtooth`` (ramp from -1 to +1) so the
    anti-aliased and naive paths are drop-in interchangeable.
    """
    dt = freq / sr
    phase = (np.arange(n) * dt) % 1.0
    naive = 2.0 * phase - 1.0
    return naive - _poly_blep(phase, dt)


def render_prior(midi_path: str, cfg: AriosoConfig, sr: int = SR,
                 total_samples: int | None = None) -> np.ndarray:
    """Render the quantized Arioso prior waveform (no level match yet).

    Mirrors ``DataSynthesizer.synthesizePrior.render_prior`` but honours the ``cfg``
    toggles (``anti_alias``, ``envelope``). Returns an *un-normalized* float32 mono
    waveform — level matching is applied separately by :func:`level_match` so the
    sounding-RMS computation sees the raw saw level.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    if total_samples is None:
        total_samples = int(round(pm.get_end_time() * sr))
    out = np.zeros(total_samples, dtype=np.float64)

    for inst in pm.instruments:
        for note in inst.notes:
            start = max(0, int(round(note.start * sr)))
            end = min(total_samples, int(round(note.end * sr)))
            n = end - start
            if n <= 1:
                continue
            freq = pretty_midi.note_number_to_hz(note.pitch)  # quantized; bends ignored
            amp = note.velocity / 127.0
            if cfg.anti_alias:
                y = amp * _saw_blep(freq, n, sr)
            else:
                t = np.arange(n) / sr
                y = amp * sawtooth(2.0 * np.pi * freq * t)
            if cfg.envelope == "fade":
                y *= _fade_envelope(n, sr, cfg.fade_ms)
            # "rect": hard on/off — no envelope, clicks accepted as baseline content.
            out[start:end] += y

    return out.astype(np.float32)


def _sounding_mask(midi_path: str, n_samples: int, sr: int = SR) -> np.ndarray:
    """Boolean [n_samples] mask, True where any score note is active (sounding)."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    mask = np.zeros(n_samples, dtype=bool)
    for inst in pm.instruments:
        for note in inst.notes:
            start = max(0, int(round(note.start * sr)))
            end = min(n_samples, int(round(note.end * sr)))
            if end > start:
                mask[start:end] = True
    return mask


def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0


def level_match(prior: np.ndarray, sounding: np.ndarray,
                cfg: AriosoConfig) -> tuple[np.ndarray, float]:
    """Scale ``prior`` to the target level. Returns ``(scaled_prior, gain)``.

    ``masked_rms`` (baseline): single per-recording gain so the prior's RMS over *sounding*
    samples (``sounding`` mask) hits ``cfg.target_rms_dbfs`` — the constant every GT was
    voiced-RMS-normalized to. This prevents the ~30x-hotter saw from dominating transport
    (Section 4.3) while staying fully score-determined, so the prior is identical at train and
    inference. ``peak``: legacy peak-normalize to ``DEFAULT_PEAK`` (matches the old prior).
    """
    if cfg.level_match == "peak":
        from common.audio_io import normalize
        return normalize(prior).astype(np.float32), float("nan")

    prior_rms = _rms(prior[sounding[: len(prior)]])
    if prior_rms <= 0.0:
        return prior.astype(np.float32), 0.0
    target_lin = 10.0 ** (cfg.target_rms_dbfs / 20.0)
    gain = target_lin / prior_rms
    return (prior * gain).astype(np.float32), gain
