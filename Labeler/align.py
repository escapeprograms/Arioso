"""Align the transcribed notes to the cleaned audio (residual global offset).

MUSC's note times can carry a small constant offset vs. the actual audio. We
measure it exactly the way the synthetic dataset build does: render the quantized
prior from ``musc_raw.mid`` (the pre-alignment flatten of the transcription) via
``common.prior.quantized_prior`` to the cleaned clip's length, then cross-correlate
onset-strength envelopes via ``common.onset_align.estimate_offset_seconds``.

Sign convention (inherited): ``estimate_offset_seconds`` is **positive when the
prior lags the audio** (the MIDI events arrive late), so we advance the notes by
subtracting the offset — ``offset_applied_s = -offset`` — and clamp starts at 0.
A re-estimate on the shifted notes should read ~0.
"""

from __future__ import annotations

import numpy as np

from common.config import SR
from common.onset_align import estimate_offset_seconds
from common.prior import quantized_prior


def estimate_midi_offset(midi_path: str, cleaned: np.ndarray, sr: int = SR) -> float:
    """Estimate the raw offset (seconds); positive => prior/MIDI lags the audio."""
    prior = quantized_prior().render(midi_path, total_samples=len(cleaned))
    n = min(len(prior), len(cleaned))
    if n == 0:
        return 0.0
    return float(estimate_offset_seconds(np.asarray(prior[:n], dtype=np.float32),
                                         np.asarray(cleaned[:n], dtype=np.float32), sr=sr))


def shift_note_times(start_s: float, end_s: float, shift_s: float) -> tuple[float, float]:
    """Shift one note's ``(start, end)`` by ``shift_s``, clamping start >= 0."""
    ns = max(0.0, float(start_s) + shift_s)
    ne = max(ns + 1e-6, float(end_s) + shift_s)
    return ns, ne


def apply_offset(notes: list[dict], shift_s: float) -> list[dict]:
    """Return copies of ``notes`` with ``start_s``/``end_s`` shifted by ``shift_s``.

    ``shift_s`` is ``offset_applied_s`` (i.e. ``-offset``): negative advances the
    notes into alignment when the prior lagged.
    """
    out = []
    for n in notes:
        ns, ne = shift_note_times(n["start_s"], n["end_s"], shift_s)
        m = dict(n)
        m["start_s"], m["end_s"] = ns, ne
        out.append(m)
    return out
