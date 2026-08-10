"""Keyshift mel: identity at 0 cents, cents-invariant frame count, shift correctness, slice math.

Pins the four contracts ``common.keyshift`` promises (all pure CPU, no data root, no GPU):

* ``mel_frames_keyshift(wav, 0.0)`` is **bit-identical** to ``common.vocoder.mel_frames(wav)``
  — the re-implemented STFT can never silently drift from the vendored BigVGAN one.
* ``T == 1 + (len(wav) - HOP_SIZE) // HOP_SIZE`` for every ``cents`` — augmented mels stay
  frame-aligned with score-rasterized conditioning tracks.
* The shift really is a pitch shift: the keyshifted mel of a tone matches the mel of the
  *resynthesized* transposed tone, bin for bin.
* The known top-mel-bin artifact for ``cents < 0`` is locked to bins 126/127 so it cannot
  silently widen into the band the model actually learns.

Tolerances below are measured facts, not guesses; a failure means the DSP changed.
"""

from __future__ import annotations

import numpy as np
import pytest

from common.config import HOP_SIZE, N_MELS, SR
from common.keyshift import (keyshift_factor, margin_frames, mel_frames_keyshift,
                             scaled_win, slice_read_range)
from common.vocoder import mel_frames

FLOOR = float(np.log(1e-5))       # -11.512925…, the value a dead mel bin clamps to
AMPS = (1.0, 0.5, 0.25)           # a 3-partial "violin-ish" harmonic ladder


def _harmonic(f0: float, dur_s: float = 3.0) -> np.ndarray:
    """A ``dur_s``-second 3-partial tone at ``f0`` Hz, float32, safely inside [-1, 1]."""
    t = np.arange(int(dur_s * SR), dtype=np.float64) / SR
    y = sum(a * np.sin(2 * np.pi * f0 * i * t) for i, a in enumerate(AMPS, start=1))
    return (0.6 * y / np.abs(y).max()).astype(np.float32)


def _band_noise(cut_hz: float = 21500.0, dur_s: float = 3.0, seed: int = 0) -> np.ndarray:
    """Broadband noise brick-walled at ``cut_hz`` — a stand-in for a real (lowpassed) mic feed.

    Broadband so the top mel bins actually carry energy at 0 cents; band-limited just below
    Nyquist so the artifact test can tell "the shift killed bin 127" apart from "the signal
    never had anything up there".
    """
    n = int(dur_s * SR)
    spec = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    spec[np.fft.rfftfreq(n, 1.0 / SR) > cut_hz] = 0.0
    y = np.fft.irfft(spec, n)
    return (0.3 * y / np.abs(y).max()).astype(np.float32)


# --- the identity oracle ---------------------------------------------------------

def test_zero_cents_is_bit_identical_to_vocoder_mel_frames():
    # Not allclose — array_equal. At cents == 0 every keyshift step degenerates to the
    # vendored one (win 2048, pad 768/768, no bin crop, rescale exactly *1.0).
    y = _harmonic(440.0)
    ref = mel_frames(y)
    got = mel_frames_keyshift(y, 0.0)
    assert got.shape == ref.shape, f"shape drift: {got.shape} vs {ref.shape}"
    assert np.array_equal(got, ref), f"max|delta| = {np.abs(got - ref).max()}"


def test_output_is_float32_c_contiguous_n_mels_rows():
    got = mel_frames_keyshift(_harmonic(440.0, dur_s=1.0), 75.0)
    assert got.dtype == np.float32
    assert got.flags["C_CONTIGUOUS"]
    assert got.shape[0] == N_MELS


# --- frame count is independent of cents -----------------------------------------

@pytest.mark.parametrize("length", [132300, 132437, 220500, 100000])
@pytest.mark.parametrize("cents", [-100.0, -50.0, 0.0, 50.0, 100.0])
def test_frame_count_matches_the_hop_formula_for_every_shift(length, cents):
    y = _harmonic(440.0, dur_s=(length + SR) / SR)[:length]
    assert len(y) == length
    expected = 1 + (length - HOP_SIZE) // HOP_SIZE
    got = mel_frames_keyshift(y, cents)
    assert got.shape == (N_MELS, expected), f"cents={cents} L={length}: {got.shape}"


# --- the shift is a real pitch shift ---------------------------------------------

@pytest.mark.parametrize("cents", [-100.0, -50.0, 50.0, 100.0])
def test_keyshifted_tone_matches_a_resynthesized_transposed_tone(cents):
    # Ground truth for "shifted by `cents`" is the tone actually rendered at the shifted
    # f0 and melled the normal way; the keyshift mel must land on the same spectrum.
    shifted = mel_frames_keyshift(_harmonic(440.0), cents).mean(axis=1)
    ref = mel_frames(_harmonic(440.0 * keyshift_factor(cents))).mean(axis=1)
    assert shifted.argmax() == ref.argmax(), (
        f"cents={cents}: peak bin {shifted.argmax()} vs resynth {ref.argmax()}"
    )
    delta = np.abs(shifted[:116] - ref[:116]).max()
    assert delta < 0.25, f"cents={cents}: bins 0-115 max|delta| {delta:.4f} (worst measured 0.134)"


# --- shift geometry constants ----------------------------------------------------

def test_scaled_win_and_factor_are_the_pinned_values():
    assert scaled_win(0.0) == 2048
    assert scaled_win(100.0) == 2170
    assert scaled_win(-100.0) == 1933
    assert keyshift_factor(1200.0) == 2.0
    assert keyshift_factor(0.0) == 1.0


# --- the known top-bin artifact, locked ------------------------------------------

def test_negative_shift_kills_only_the_top_mel_bin():
    # Mel row 127 draws on FFT bins 961..1024; at -100 cents only 967 bins exist, so its
    # support all but vanishes. Row 126 (bins 931..991) is partly clipped; rows <= 125
    # (bins <= 960) are untouched.
    y = _band_noise()
    m0 = mel_frames_keyshift(y, 0.0)
    down = mel_frames_keyshift(y, -100.0)

    # The signal DOES excite bin 127 unshifted — so the collapse below is the shift's doing.
    assert m0[127].mean() > -5.0, f"test signal too dull at bin 127: {m0[127].mean():.3f}"
    # Shifted down, bin 127 is on the log(1e-5) floor for essentially every frame
    # (the handful of exceptions are float32 FFT round-off, ~1.6 log-units above it).
    assert np.median(down[127]) == pytest.approx(FLOOR, abs=1e-4)
    assert np.mean(down[127] <= FLOOR + 1e-6) > 0.95, "bin 127 not floored by the -100 c shift"

    # Bin 126: attenuated, but nowhere near dead.
    drop_126 = m0[126].mean() - down[126].mean()
    assert 1.0 < drop_126 < 5.0, f"bin 126 attenuation moved: {drop_126:.3f} (measured ~2.4)"
    assert down[126].mean() > FLOOR + 4.0, "bin 126 fell to the floor — artifact widened"

    # Bins <= 125: unaffected. Compared time-averaged, since a per-frame noise mel is
    # dominated by its own frame-to-frame variance, not by the shift.
    delta = np.abs(down[:126].mean(axis=1) - m0[:126].mean(axis=1))
    assert delta.mean() < 0.1, f"bins<=125 mean |delta| {delta.mean():.4f} (measured 0.040)"
    assert delta.max() < 0.25, f"bins<=125 max |delta| {delta.max():.4f} (measured 0.175)"


# --- margin_frames ---------------------------------------------------------------

def test_margin_frames_pinned_and_monotone():
    # (scaled_win(100) - 512)//2 = 829 samples -> ceil(829/512) + 1 = 3 frames.
    assert margin_frames(100.0) == 3
    # Even at 0 cents the vendored 768-sample pad spans 1.5 hops -> 2, +1 = 3.
    assert margin_frames(0.0) >= 2
    margins = [margin_frames(c) for c in (0.0, 25.0, 50.0, 100.0, 200.0)]
    assert margins == sorted(margins), f"margin_frames not monotone: {margins}"
    assert margin_frames(-100.0) == margin_frames(100.0), "margin must use |cents|"


# --- slice_read_range ------------------------------------------------------------

@pytest.mark.parametrize("start,end,margin", [(0, 100, 3), (1, 50, 3), (2, 9, 3),
                                              (3, 90, 3), (1000, 1256, 3), (7, 8, 2)])
def test_slice_read_start_is_always_a_hop_multiple(start, end, margin):
    read_start, read_n, _head = slice_read_range(start, end, margin)
    assert read_start % HOP_SIZE == 0, "frame j of the slice would no longer be frame f0+j"
    assert read_n % HOP_SIZE == 0
    assert read_n > 0


def test_slice_head_frames_follows_the_clamp():
    # Interior cut: the full margin is read ahead of `start` and later discarded.
    read_start, read_n, head = slice_read_range(1000, 1256, 3)
    assert head == 3
    assert read_start == (1000 - 3) * HOP_SIZE
    assert read_n == (1256 + 3 - 997) * HOP_SIZE

    # Clamped at the file head: nothing before frame 0 to read, so head_frames == start.
    for start in (0, 1, 2):
        _rs, _rn, head = slice_read_range(start, start + 20, 3)
        assert head == start, f"head clamp wrong at start={start}: {head}"
    assert slice_read_range(0, 20, 3)[0] == 0

    # General identity.
    for start, margin in ((0, 3), (2, 3), (5, 3), (40, 7)):
        assert slice_read_range(start, start + 10, margin)[2] == start - max(0, start - margin)


# --- degenerate input ------------------------------------------------------------

@pytest.mark.parametrize("cents", [-100.0, 0.0, 100.0])
def test_signal_shorter_than_the_window_does_not_raise(cents):
    # 600 samples < win_new (1933..2170) and < pad_left (710..829): torch's reflect pad
    # would reject this, so the implementation falls back to a constant pad.
    y = _harmonic(440.0, dur_s=600 / SR)[:600]
    got = mel_frames_keyshift(y, cents)
    assert got.shape == (N_MELS, 1 + (600 - HOP_SIZE) // HOP_SIZE)
    assert np.isfinite(got).all()
