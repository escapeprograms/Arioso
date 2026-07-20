"""bend: expression detection, sampled curve, clamp, wheel ints, reset. Torch-free."""

from __future__ import annotations

import numpy as np

from Studio.bend import (WHEEL_MAX, has_expression, pitch_bend_events,
                         semitone_curve, to_wheel)

# bpm 60 -> 1 beat == 1 second, so beats and seconds coincide in these tests.
_BPM = 60.0


def _note(bend=None, vibrato=None):
    return {"id": "n0", "bend": bend or [],
            "vibrato": vibrato or {"depth_semitones": 0.0, "rate_hz": 5.5, "onset_beats": 0.0}}


def test_flat_note_has_no_expression():
    assert not has_expression(_note())
    assert not has_expression(_note(bend=[{"beat": 0.0, "semitones": 0.0}]))
    assert pitch_bend_events(_note(), 0.0, 1.0, _BPM) == []


def test_bend_point_is_expression():
    n = _note(bend=[{"beat": 0.0, "semitones": 0.0}, {"beat": 1.0, "semitones": 1.0}])
    assert has_expression(n)


def test_vibrato_depth_is_expression():
    assert has_expression(_note(vibrato={"depth_semitones": 0.3, "rate_hz": 5.0, "onset_beats": 0.0}))


def test_events_monotonic_and_reset():
    n = _note(bend=[{"beat": 0.0, "semitones": 0.0}, {"beat": 1.0, "semitones": 2.0}],
              vibrato={"depth_semitones": 0.2, "rate_hz": 6.0, "onset_beats": 0.1})
    events = pitch_bend_events(n, 5.0, 1.0, _BPM)
    times = [t for t, _ in events]
    assert times == sorted(times)                       # non-decreasing
    assert times[0] == 5.0                              # starts at note onset
    assert events[-1][1] == 0                           # trailing reset to centre
    assert events[-1][0] > 5.0 + 1.0                    # reset lands after the note


def test_curve_clamped_to_range():
    # A +5 st bend saturates at +2 st -> wheel at +WHEEL_MAX.
    n = _note(bend=[{"beat": 0.0, "semitones": 5.0}, {"beat": 1.0, "semitones": 5.0}])
    _t, semis = semitone_curve(n, 1.0, _BPM)
    assert np.max(semis) <= 2.0 + 1e-9
    assert to_wheel(semis).max() == WHEEL_MAX


def test_wheel_conversion_bounds():
    assert to_wheel(np.array([2.0]))[0] == WHEEL_MAX     # +2 st -> +8192 clamped to +8191
    assert to_wheel(np.array([-2.0]))[0] == -WHEEL_MAX
    assert to_wheel(np.array([0.0]))[0] == 0


def test_reset_suppressed_when_legato():
    # Next note starts immediately at the end -> no gap -> no reset appended, so the
    # final event is the (non-zero) curve endpoint rather than a PitchBend(0).
    n = _note(bend=[{"beat": 0.0, "semitones": 1.0}, {"beat": 1.0, "semitones": 1.0}])
    events = pitch_bend_events(n, 0.0, 1.0, _BPM, next_start_s=1.0)
    assert events[-1][0] == 1.0        # last event is the curve endpoint at the note end
    assert events[-1][1] != 0          # not a reset


def test_reset_present_with_gap():
    n = _note(bend=[{"beat": 0.0, "semitones": 1.0}, {"beat": 1.0, "semitones": 1.0}])
    events = pitch_bend_events(n, 0.0, 1.0, _BPM, next_start_s=2.0)
    assert events[-1][1] == 0          # reset placed in the gap
    assert 1.0 < events[-1][0] <= 2.0
