"""timing: beats<->seconds round-trips, bar/beat math, snap grid. Torch-free."""

from __future__ import annotations

import pytest

from Studio import timing

_44 = {"num": 4, "den": 4}
_68 = {"num": 6, "den": 8}
_34 = {"num": 3, "den": 4}


def test_beats_seconds_roundtrip():
    for bpm in (60.0, 120.0, 140.0, 200.0):
        for beat in (0.0, 1.0, 3.5, 42.0):
            s = timing.beats_to_seconds(beat, bpm)
            assert timing.seconds_to_beats(s, bpm) == pytest.approx(beat)


def test_beats_to_seconds_known_values():
    # At 120 bpm a quarter-note beat is 0.5 s.
    assert timing.beats_to_seconds(1.0, 120.0) == pytest.approx(0.5)
    assert timing.beats_to_seconds(4.0, 60.0) == pytest.approx(4.0)


def test_nonpositive_bpm_raises():
    with pytest.raises(ValueError):
        timing.beats_to_seconds(1.0, 0.0)
    with pytest.raises(ValueError):
        timing.seconds_to_beats(1.0, -5.0)


def test_quarters_per_bar():
    assert timing.quarters_per_bar(_44) == pytest.approx(4.0)
    assert timing.quarters_per_bar(_68) == pytest.approx(3.0)   # 6 * 4/8
    assert timing.quarters_per_bar(_34) == pytest.approx(3.0)


def test_bar_number_and_within_bar():
    # 4/4: bar boundaries every 4 quarter-beats.
    assert timing.bar_number(0.0, _44) == 0
    assert timing.bar_number(3.99, _44) == 0
    assert timing.bar_number(4.0, _44) == 1
    assert timing.bar_number(9.0, _44) == 2
    assert timing.beat_within_bar(5.5, _44) == pytest.approx(1.5)


def test_bar_beat_tick():
    # 2.5 quarter-beats into 4/4 = bar 0, denominator-beat 2, half a beat of ticks.
    bar, dbeat, tick = timing.bar_beat_tick(2.5, _44, ppq=480)
    assert (bar, dbeat) == (0, 2)
    assert tick == 240
    # 6/8: denominator beat = an eighth = 0.5 quarter; 3.5 quarters -> bar 1.
    bar, dbeat, tick = timing.bar_beat_tick(3.5, _68, ppq=480)
    assert bar == 1
    assert dbeat == 1          # 0.5 quarter into the bar = one eighth
    assert tick == 0


def test_snap_grid_sizes_44():
    assert timing.snap_grid("none", _44) == 0.0
    assert timing.snap_grid("bar", _44) == pytest.approx(4.0)
    assert timing.snap_grid("beat", _44) == pytest.approx(1.0)
    assert timing.snap_grid("step", _44) == pytest.approx(0.25)      # 1/16 note
    assert timing.snap_grid("1/2step", _44) == pytest.approx(0.125)
    assert timing.snap_grid("1/3step", _44) == pytest.approx(0.25 / 3)


def test_snap_grid_unknown_raises():
    with pytest.raises(ValueError):
        timing.snap_grid("triplet", _44)


def test_snap_beat_rounds_to_grid():
    assert timing.snap_beat(0.30, 0.25) == pytest.approx(0.25)
    assert timing.snap_beat(0.40, 0.25) == pytest.approx(0.50)
    # grid 0 -> no snapping.
    assert timing.snap_beat(0.37, 0.0) == pytest.approx(0.37)


def test_snap_to_convenience():
    assert timing.snap_to(1.1, "beat", _44) == pytest.approx(1.0)
    assert timing.snap_to(1.6, "bar", _44) == pytest.approx(0.0)
    assert timing.snap_to(2.6, "bar", _44) == pytest.approx(4.0)
