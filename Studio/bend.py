"""Per-note pitch expression -> 14-bit pitch-wheel events for the saw prior.

A Studio note carries two expression layers, both in **semitones of deviation**
from its nominal pitch:

  * ``bend`` — a piecewise-linear list of ``{beat, semitones}`` control points
    (beats relative to the note onset). Empty, or a single zero point, means flat.
  * ``vibrato`` — ``{depth_semitones, rate_hz, onset_beats}``: a sine
    ``depth * sin(2π rate t)`` that fades in linearly over ~2 cycles starting at
    ``onset_beats`` (so it grows organically instead of snapping on).

This module samples the combined curve at a fixed **200 Hz control rate** over the
note's second span, clamps the total to the MIDI pitch-wheel range
(``±PB_RANGE_SEMITONES`` = ±2 st — what the prior's :class:`PitchBend` decodes),
and converts to 14-bit wheel ints (``semitones / range * 8192``, clamped to
``±8191``). It returns ``[(time_s, wheel)]`` events on the project's absolute
seconds timeline, ready to become ``pretty_midi.PitchBend`` objects.

Flat notes emit **no events** (:func:`has_expression` gates them) so the wheel is
left at whatever the previous note reset it to. Every expressive note therefore
appends a ``PitchBend(0)`` **reset** in the gap after it (just past its end, before
the next note) so a following flat note is not left bent. Torch-free (numpy only);
beats->seconds uses :mod:`Studio.timing` with the document BPM.
"""

from __future__ import annotations

import numpy as np

from common.prior import PB_RANGE_SEMITONES

from .timing import beats_to_seconds

# Dense control-point rate the curve is sampled at before quantizing to wheel ints.
CONTROL_RATE_HZ = 200.0
# 14-bit signed pitch-wheel magnitude (MIDI wheel spans -8192..+8191).
WHEEL_MAX = 8191
# Reset placed at most this long after an expressive note ends (or the gap midpoint,
# whichever is sooner), returning the wheel to centre before the next note.
_RESET_LEAD_S = 0.05


def has_expression(note: dict) -> bool:
    """True if ``note`` has any non-flat bend point or positive vibrato depth."""
    for pt in note.get("bend") or []:
        if abs(float(pt.get("semitones", 0.0))) > 0.0:
            return True
    vib = note.get("vibrato") or {}
    return float(vib.get("depth_semitones", 0.0)) > 0.0


def _bend_semitones(note: dict, t_rel_s: np.ndarray, bpm: float) -> np.ndarray:
    """Piecewise-linear bend deviation (semitones) at note-relative times ``t_rel_s``."""
    pts = note.get("bend") or []
    if not pts:
        return np.zeros_like(t_rel_s)
    beats = np.array([float(p["beat"]) for p in pts], dtype=np.float64)
    vals = np.array([float(p["semitones"]) for p in pts], dtype=np.float64)
    order = np.argsort(beats)
    xt = np.array([beats_to_seconds(b, bpm) for b in beats[order]], dtype=np.float64)
    # np.interp holds the endpoint value outside [xt[0], xt[-1]] and returns a
    # constant for a single control point -> a lone (or zero) point reads as flat.
    return np.interp(t_rel_s, xt, vals[order])


def _vibrato_semitones(note: dict, t_rel_s: np.ndarray, bpm: float) -> np.ndarray:
    """Ramped vibrato deviation (semitones) at note-relative times ``t_rel_s``."""
    vib = note.get("vibrato") or {}
    depth = float(vib.get("depth_semitones", 0.0))
    rate = float(vib.get("rate_hz", 0.0))
    if depth <= 0.0 or rate <= 0.0:
        return np.zeros_like(t_rel_s)
    onset_s = beats_to_seconds(float(vib.get("onset_beats", 0.0)), bpm)
    dt = t_rel_s - onset_s                       # time since vibrato onset
    ramp = np.clip(dt / (2.0 / rate), 0.0, 1.0)  # linear fade-in over ~2 cycles
    osc = depth * np.sin(2.0 * np.pi * rate * dt)
    return np.where(dt > 0.0, ramp * osc, 0.0)


def semitone_curve(note: dict, dur_s: float, bpm: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample the combined bend+vibrato curve for a note of ``dur_s`` seconds.

    Returns ``(t_rel_s, semitones)`` at the control rate, clamped to
    ``±PB_RANGE_SEMITONES``. ``t_rel_s`` is note-relative (0..dur_s).
    """
    n = max(2, int(round(dur_s * CONTROL_RATE_HZ)) + 1)
    t_rel = np.linspace(0.0, dur_s, n)
    semis = _bend_semitones(note, t_rel, bpm) + _vibrato_semitones(note, t_rel, bpm)
    return t_rel, np.clip(semis, -PB_RANGE_SEMITONES, PB_RANGE_SEMITONES)


def to_wheel(semitones: np.ndarray) -> np.ndarray:
    """Semitone deviations -> clamped 14-bit signed pitch-wheel ints."""
    wheel = np.round(semitones / PB_RANGE_SEMITONES * 8192.0)
    return np.clip(wheel, -WHEEL_MAX, WHEEL_MAX).astype(np.int64)


def reset_time(end_s: float, next_start_s: float | None = None) -> float | None:
    """Absolute time for the ``PitchBend(0)`` reset after a note ending at ``end_s``.

    Placed ``_RESET_LEAD_S`` past the end, or the midpoint of the gap to the next
    note if that gap is shorter. Returns ``None`` when there is effectively no gap
    (legato) so the reset never lands on the next note's first event.
    """
    if next_start_s is None:
        return end_s + _RESET_LEAD_S
    gap = next_start_s - end_s
    if gap <= 1e-4:
        return None
    return end_s + min(_RESET_LEAD_S, gap / 2.0)


def pitch_bend_events(note: dict, start_s: float, dur_s: float, bpm: float,
                      next_start_s: float | None = None) -> list[tuple[float, int]]:
    """``[(time_s, wheel)]`` pitch-wheel events for one note on the absolute timeline.

    Empty for a flat note. Otherwise the sampled curve (times ``start_s..start_s +
    dur_s``) followed by a single ``(reset_time, 0)`` event in the trailing gap
    (unless there is no gap). Times are non-decreasing.
    """
    if not has_expression(note):
        return []
    t_rel, semis = semitone_curve(note, dur_s, bpm)
    wheel = to_wheel(semis)
    events = [(start_s + float(t), int(w)) for t, w in zip(t_rel, wheel)]
    rt = reset_time(start_s + dur_s, next_start_s)
    if rt is not None:
        events.append((float(rt), 0))
    return events
