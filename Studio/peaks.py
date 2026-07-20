"""Waveform peak file (``mix.peaks``) for the frontend waveform lane.

A rendered ``mix.wav`` is too big to hand the browser raw for a zoomed-out
overview, so the render pipeline precomputes a compact **peaks** file: the signal
is bucketed into fixed-rate bins and each bin is summarized by its ``(min, max)``
sample as signed 8-bit ints. The frontend draws the filled envelope straight from
these pairs.

Binary layout (little-endian), parseable without guessing::

    magic       4 bytes  b"SPK1"
    sr          uint32   sample rate of the source wav
    bins_per_sec uint32  bins per second (== resolution)
    n_bins      uint32   number of (min,max) pairs that follow
    data        n_bins * 2 int8   interleaved min0,max0, min1,max1, ...

Samples are scaled by 127 and clamped to ``[-127, 127]``. Torch-free (numpy only).
"""

from __future__ import annotations

import struct

import numpy as np

from common.config import SR

MAGIC = b"SPK1"
HEADER_FMT = "<III"                     # sr, bins_per_sec, n_bins (after the magic)
HEADER_SIZE = len(MAGIC) + struct.calcsize(HEADER_FMT)


def compute_peaks(wav_f32: np.ndarray, sr: int = SR,
                  bins_per_sec: int = 100) -> tuple[np.ndarray, np.ndarray, int]:
    """Bin ``wav_f32`` and return ``(min_i8, max_i8, samples_per_bin)``.

    Each bin spans ``round(sr / bins_per_sec)`` samples; the final partial bin is
    zero-padded (a negligible pull toward 0 for one bin). Values are int8 in
    ``[-127, 127]``.
    """
    wav = np.asarray(wav_f32, dtype=np.float32).reshape(-1)
    spb = max(1, int(round(sr / bins_per_sec)))
    if wav.size == 0:
        empty = np.zeros(0, dtype=np.int8)
        return empty, empty, spb
    n_bins = int(np.ceil(wav.size / spb))
    pad = n_bins * spb - wav.size
    if pad:
        wav = np.concatenate([wav, np.zeros(pad, dtype=np.float32)])
    frames = wav.reshape(n_bins, spb)
    to_i8 = lambda a: np.clip(np.round(a * 127.0), -127, 127).astype(np.int8)
    return to_i8(frames.min(axis=1)), to_i8(frames.max(axis=1)), spb


def write_peaks(wav_f32: np.ndarray, path: str, sr: int = SR,
                bins_per_sec: int = 100) -> str:
    """Write the ``SPK1`` peaks file for ``wav_f32`` to ``path``; return ``path``."""
    mn, mx, _spb = compute_peaks(wav_f32, sr=sr, bins_per_sec=bins_per_sec)
    n_bins = int(mn.size)
    pairs = np.empty(n_bins * 2, dtype=np.int8)
    pairs[0::2] = mn
    pairs[1::2] = mx
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack(HEADER_FMT, int(sr), int(bins_per_sec), n_bins))
        f.write(pairs.tobytes())
    return path


def peaks_meta(path: str) -> dict:
    """Parse just the header of a peaks file -> ``{magic, sr, bins_per_sec, n_bins}``."""
    with open(path, "rb") as f:
        magic = f.read(len(MAGIC))
        sr, bins_per_sec, n_bins = struct.unpack(HEADER_FMT, f.read(struct.calcsize(HEADER_FMT)))
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic {magic!r} (expected {MAGIC!r})")
    return {"magic": magic.decode("ascii"), "sr": sr,
            "bins_per_sec": bins_per_sec, "n_bins": n_bins}
