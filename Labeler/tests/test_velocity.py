"""Velocity mapping: monotonicity, MIDI range, and the degenerate fallback."""

from __future__ import annotations

import numpy as np

from Labeler.config import VelocityParams
from Labeler.velocity import map_velocities


def test_monotonic_nondecreasing():
    strengths = [0.0, 0.5, 1.0, 2.0, 5.0, 20.0]
    vel = map_velocities(strengths)
    assert vel == sorted(vel), "velocity must be non-decreasing in onset strength"


def test_range_within_midi_bounds():
    strengths = np.linspace(0, 100, 50).tolist()
    vel = map_velocities(strengths)
    assert all(1 <= v <= 127 for v in vel)
    # floor at x=0, floor+range at x=1 (before the 1..127 clamp).
    p = VelocityParams()
    assert min(vel) >= 1 and max(vel) <= p.vel_floor + p.vel_range


def test_degenerate_all_equal():
    p = VelocityParams()
    assert map_velocities([3.3, 3.3, 3.3, 3.3]) == [p.degenerate_velocity] * 4


def test_degenerate_single_note():
    p = VelocityParams()
    assert map_velocities([7.7]) == [p.degenerate_velocity]


def test_empty():
    assert map_velocities([]) == []


def test_spread_uses_full_curve():
    # A wide spread should reach both ends of the mapped band.
    vel = map_velocities([0.0, 0.01, 100.0])
    p = VelocityParams()
    assert vel[0] == p.vel_floor
    assert vel[-1] == min(127, p.vel_floor + p.vel_range)
