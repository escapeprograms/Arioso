"""Pure beats<->seconds authority + bar/beat and snap-grid math (no torch).

Note timing in a Studio project is stored in **beats**, where one beat is a
**quarter note** (matching quarter-note BPM and ``ppq`` = pulses per quarter).
Seconds only exist at the audio/MIDI boundary. All functions here are pure and
side-effect-free so they can be unit-tested and shared with the render pipeline.

Time signature is a ``{"num": int, "den": int}`` mapping. A "denominator beat"
is one ``1/den`` note, worth ``4/den`` quarter-note beats; a bar holds ``num`` of
them, i.e. ``num * 4 / den`` quarter-note beats. A "step" is a quarter of a
denominator beat (a 1/16 note in 4/4 — FL Studio's default step).
"""

from __future__ import annotations

from typing import Mapping

_MIN_BPM = 1e-6


# --- beats <-> seconds ------------------------------------------------------

def beats_to_seconds(beat: float, bpm: float) -> float:
    """Convert quarter-note beats to seconds at ``bpm`` (quarter notes / minute)."""
    if bpm <= _MIN_BPM:
        raise ValueError(f"bpm must be positive, got {bpm}")
    return float(beat) * 60.0 / float(bpm)


def seconds_to_beats(seconds: float, bpm: float) -> float:
    """Convert seconds to quarter-note beats at ``bpm``."""
    if bpm <= _MIN_BPM:
        raise ValueError(f"bpm must be positive, got {bpm}")
    return float(seconds) * float(bpm) / 60.0


# --- bar / beat math --------------------------------------------------------

def _sig(time_sig: Mapping) -> tuple[int, int]:
    num = int(time_sig["num"])
    den = int(time_sig["den"])
    if num <= 0 or den <= 0:
        raise ValueError(f"time_sig num/den must be positive, got {time_sig}")
    return num, den


def quarters_per_beat(time_sig: Mapping) -> float:
    """Quarter-note beats in one denominator beat (``4/den``)."""
    _, den = _sig(time_sig)
    return 4.0 / den


def quarters_per_bar(time_sig: Mapping) -> float:
    """Quarter-note beats in one bar (``num * 4 / den``)."""
    num, den = _sig(time_sig)
    return num * 4.0 / den


def bar_number(beat: float, time_sig: Mapping) -> int:
    """0-based bar index containing ``beat`` (quarter-note beats)."""
    qpb = quarters_per_bar(time_sig)
    return int(float(beat) // qpb)


def beat_within_bar(beat: float, time_sig: Mapping) -> float:
    """Position within the current bar, in quarter-note beats ``[0, quarters_per_bar)``."""
    qpb = quarters_per_bar(time_sig)
    return float(beat) - bar_number(beat, time_sig) * qpb


def bar_beat_tick(beat: float, time_sig: Mapping, ppq: int = 480) -> tuple[int, int, int]:
    """Decompose ``beat`` into ``(bar, denominator_beat, tick)``, all 0-based.

    ``bar`` and ``denominator_beat`` count from 0 (a UI adds 1 for the FL-style
    ``bar:beat`` readout); ``tick`` is the residue within the denominator beat in
    ``ppq`` pulses-per-quarter units (so ``tick < ppq * 4/den``).
    """
    num, den = _sig(time_sig)
    q_per_dbeat = 4.0 / den
    within = beat_within_bar(beat, time_sig)
    dbeat = int(within // q_per_dbeat)
    frac_q = within - dbeat * q_per_dbeat
    tick = int(round(frac_q * ppq))
    return bar_number(beat, time_sig), dbeat, tick


# --- snap grid --------------------------------------------------------------

def snap_grid(snap: str, time_sig: Mapping) -> float:
    """Grid size in quarter-note beats for a snap id (0.0 means no snapping).

    ``bar`` -> a full bar; ``beat`` -> one denominator beat; ``step`` -> a quarter
    of a beat; ``1/2step`` .. ``1/6step`` -> finer fractions of a step; ``line``
    tracks the step grid; ``none`` -> 0.
    """
    beat = quarters_per_beat(time_sig)          # one denominator beat
    step = beat / 4.0                            # 1/16 note in 4/4
    grids = {
        "none": 0.0,
        "line": step,
        "1/2step": step / 2.0,
        "1/3step": step / 3.0,
        "1/4step": step / 4.0,
        "1/6step": step / 6.0,
        "step": step,
        "beat": beat,
        "bar": quarters_per_bar(time_sig),
    }
    if snap not in grids:
        raise ValueError(f"unknown snap id {snap!r}; valid: {sorted(grids)}")
    return grids[snap]


def snap_beat(beat: float, grid: float) -> float:
    """Round ``beat`` to the nearest multiple of ``grid`` (``grid <= 0`` -> unchanged)."""
    if grid <= 0.0:
        return float(beat)
    return round(float(beat) / grid) * grid


def snap_to(beat: float, snap: str, time_sig: Mapping) -> float:
    """Convenience: snap ``beat`` using a snap id and time signature."""
    return snap_beat(beat, snap_grid(snap, time_sig))
