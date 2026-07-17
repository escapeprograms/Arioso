"""Align sign convention: a delayed prior reads a positive offset (prior lags)."""

from __future__ import annotations

import numpy as np

from common.config import SR
from DataSynthesizer.onset_align import estimate_offset_seconds


def _click_track(onsets_s, dur_s=3.0, sr=SR):
    """A signal with short 440 Hz attack bursts at each onset (onset-detectable)."""
    n = int(dur_s * sr)
    y = np.zeros(n, dtype=np.float32)
    burst_len = int(0.03 * sr)
    t = np.arange(burst_len) / sr
    env = np.exp(-t * 40.0)
    burst = (np.sin(2 * np.pi * 440.0 * t) * env).astype(np.float32)
    for on in onsets_s:
        i = int(on * sr)
        if 0 <= i < n - burst_len:
            y[i:i + burst_len] += burst
    return y


def _delay(y, lag_s, sr=SR):
    lag = int(round(lag_s * sr))
    return np.concatenate([np.zeros(lag, dtype=y.dtype), y])[: len(y)]


def test_positive_offset_when_prior_lags():
    onsets = [0.3, 0.9, 1.5, 2.1]
    gt = _click_track(onsets)
    lag_s = 0.05
    prior = _delay(gt, lag_s)  # prior events arrive LATER than gt
    off = estimate_offset_seconds(prior, gt)
    assert off > 0, "a lagging prior must yield a positive offset"
    assert abs(off - lag_s) < 0.02


def test_negative_offset_when_prior_leads():
    onsets = [0.3, 0.9, 1.5, 2.1]
    gt = _click_track(onsets)
    prior = _delay(gt, 0.05)
    # Swapping roles: now the (delayed) track is the GT, the base leads.
    off = estimate_offset_seconds(gt, prior)
    assert off < 0, "a leading prior must yield a negative offset"


def test_aligned_offset_near_zero():
    onsets = [0.3, 0.9, 1.5, 2.1]
    gt = _click_track(onsets)
    off = estimate_offset_seconds(gt.copy(), gt.copy())
    assert abs(off) < 0.012  # within one hop
