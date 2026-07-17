"""Vibrato detection on synthetic FM (bend-cents) trajectories."""

from __future__ import annotations

import numpy as np

from Labeler.config import MODEL_FPS
from Labeler.vibrato import detect_vibrato

FPS = MODEL_FPS


def _fm(freq_hz: float, amp_cents: float, dur_s: float, fps: float = FPS) -> np.ndarray:
    t = np.arange(int(round(dur_s * fps))) / fps
    return amp_cents * np.sin(2.0 * np.pi * freq_hz * t)


def test_true_vibrato_55hz():
    res = detect_vibrato(_fm(5.5, 25.0, 1.2), 1.2, fps=FPS)
    assert res.is_vibrato is True
    assert 4.0 <= res.rate_hz <= 9.0
    assert res.extent_cents >= 12.0


def test_flat_is_not_vibrato():
    res = detect_vibrato(np.zeros(int(1.0 * FPS)), 1.0, fps=FPS)
    assert res.is_vibrato is False


def test_slow_drift_1hz_is_not_vibrato():
    # A 1 Hz oscillation is below the 4-9 Hz band -> filtered out -> not vibrato.
    res = detect_vibrato(_fm(1.0, 40.0, 1.5), 1.5, fps=FPS)
    assert res.is_vibrato is False


def test_linear_glissando_is_not_vibrato():
    # A pure linear slide detrends to ~0 -> not vibrato.
    n = int(1.0 * FPS)
    ramp = np.linspace(-60.0, 60.0, n)
    res = detect_vibrato(ramp, 1.0, fps=FPS)
    assert res.is_vibrato is False


def test_short_note_is_not_vibrato():
    res = detect_vibrato(_fm(5.5, 25.0, 0.15), 0.15, fps=FPS)
    assert res.is_vibrato is False


def test_weak_extent_is_not_vibrato():
    # Right rate but tiny amplitude -> fails extent / sustained thresholds.
    res = detect_vibrato(_fm(6.0, 3.0, 1.0), 1.0, fps=FPS)
    assert res.is_vibrato is False
