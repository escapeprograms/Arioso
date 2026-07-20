"""peaks: SPK1 binary round-trip + header parse. Torch-free."""

from __future__ import annotations

import numpy as np

from Studio.peaks import MAGIC, compute_peaks, peaks_meta, write_peaks


def test_header_roundtrip(tmp_path):
    wav = np.sin(np.linspace(0, 40 * np.pi, 44100)).astype(np.float32)
    path = str(tmp_path / "x.peaks")
    write_peaks(wav, path, sr=44100, bins_per_sec=100)
    meta = peaks_meta(path)
    assert meta["magic"] == MAGIC.decode()
    assert meta["sr"] == 44100
    assert meta["bins_per_sec"] == 100
    # 1 s of audio at 100 bins/s -> ~100 bins.
    assert meta["n_bins"] == 100


def test_min_max_values():
    # A full-scale square-ish signal -> min ~ -127, max ~ +127.
    wav = np.concatenate([np.ones(500, np.float32), -np.ones(500, np.float32)])
    mn, mx, spb = compute_peaks(wav, sr=1000, bins_per_sec=1)
    assert spb == 1000
    assert mx[0] == 127
    assert mn[0] == -127


def test_data_length_matches_bins(tmp_path):
    wav = np.zeros(22050, dtype=np.float32)
    path = str(tmp_path / "z.peaks")
    write_peaks(wav, path, sr=44100, bins_per_sec=100)
    meta = peaks_meta(path)
    import os
    # header (16 bytes) + 2 int8 per bin.
    assert os.path.getsize(path) == 16 + meta["n_bins"] * 2


def test_empty_wav(tmp_path):
    path = str(tmp_path / "e.peaks")
    write_peaks(np.zeros(0, dtype=np.float32), path)
    assert peaks_meta(path)["n_bins"] == 0
