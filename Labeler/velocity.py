"""Estimate a MIDI velocity per note from the onset-strength envelope.

There is no amplitude->velocity model in the repo, so this is a simple,
per-clip-relative heuristic (a prefill the user can correct):

  1. ``librosa.onset.onset_strength`` on the cleaned audio (hop 512).
  2. Per note, take the **max** envelope value in a short window around its
     (aligned) onset frame: ``[f - pre_frames, f + post_frames)`` — a couple of
     frames of slack so a slightly-off onset still catches the attack peak.
  3. Normalize those per-note strengths across the clip between the 10th and 95th
     percentile (robust to one loud outlier), then bend with ``x ** 0.6`` and map
     onto ``[floor, floor + range]`` = ``[24, 120]``, clamped to the MIDI-legal 1..127.

If the percentile band collapses (all notes equally loud, or a single note) every
note gets ``degenerate_velocity`` (80). All knobs live in
:class:`Labeler.config.VelocityParams`.
"""

from __future__ import annotations

import numpy as np

from common.config import HOP_SIZE as HOP, SR

from .config import VelocityParams


def onset_env(cleaned: np.ndarray, sr: int = SR, hop: int = HOP) -> np.ndarray:
    """Onset-strength envelope of the cleaned audio (hop 512)."""
    import librosa
    return librosa.onset.onset_strength(
        y=np.asarray(cleaned, dtype=np.float32), sr=sr, hop_length=hop)


def note_onset_strengths(env: np.ndarray, onset_frames, pre: int, post: int) -> np.ndarray:
    """Per-note max envelope value over ``[f - pre, f + post)`` (clamped to bounds)."""
    vals = []
    n = len(env)
    for f in onset_frames:
        a = max(0, int(f) - pre)
        b = min(n, int(f) + post)
        vals.append(float(env[a:b].max()) if b > a else 0.0)
    return np.asarray(vals, dtype=np.float64)


def map_velocities(onset_strengths, params: VelocityParams | None = None) -> list[int]:
    """Map per-note onset strengths to integer velocities (monotone, 1..127).

    Degenerate band (``pct_high`` percentile <= ``pct_low`` percentile) ->
    ``degenerate_velocity`` for every note.
    """
    params = params or VelocityParams()
    vals = np.asarray(onset_strengths, dtype=np.float64)
    if vals.size == 0:
        return []
    lo = float(np.percentile(vals, params.pct_low))
    hi = float(np.percentile(vals, params.pct_high))
    if hi - lo <= 1e-12:
        return [int(params.degenerate_velocity)] * vals.size
    x = np.clip((vals - lo) / (hi - lo), 0.0, 1.0)
    vel = params.vel_floor + (x ** params.curve_gamma) * params.vel_range
    return np.clip(np.round(vel), 1, 127).astype(int).tolist()


def compute_velocities(cleaned: np.ndarray, onset_times_s, params: VelocityParams | None = None,
                       sr: int = SR, hop: int = HOP) -> tuple[list[int], list[float]]:
    """Return ``(velocities, per_note_onset_strengths)`` for the given onset times.

    ``onset_times_s`` are the notes' aligned onset seconds. Both lists are aligned
    to the input order.
    """
    params = params or VelocityParams()
    env = onset_env(cleaned, sr, hop)
    frames = [int(round(float(t) * sr / hop)) for t in onset_times_s]
    strengths = note_onset_strengths(env, frames, params.pre_frames, params.post_frames)
    return map_velocities(strengths, params), strengths.tolist()
