"""Sinusoidal boundary conditioning: offset frames, distance math, featurizer, config, loader.

Covers the five surfaces the boundary signals touch (all pure / CPU, no data root needed):

* :func:`common.dataset_schema.offset_frames` — clamp-not-drop tail policy.
* :func:`Arioso.dataset.boundary_distances` — the searchsorted since/until distance core.
* :class:`Arioso.model.conditioning.BoundarySinusoid` — compact-support sin/cos featurizer.
* ``AriosoConfig`` round-trip + ``__post_init__`` invariants for ``BoundaryCondSpec``.
* ``Arioso.run_config`` — boundary names / overrides resolve, unknown fields error.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from common.config import HOP_SIZE as HOP, SR
from common.dataset_schema import NoteEvent, frame_of, offset_frames

from Arioso.config import (FRAME_RATE, AriosoConfig, BoundaryCondSpec, CondSpec,
                           TIME_SINCE_ONSET, TIME_UNTIL_OFFSET, VELOCITY_COND,
                           cfg_from_dict, cfg_to_dict)
from Arioso.dataset import boundary_distances
from Arioso.model.conditioning import BoundarySinusoid
from Arioso.run_config import _parse_conditioning


# --- offset_frames --------------------------------------------------------------

def _note(end_s: float) -> NoteEvent:
    return NoteEvent(start_s=0.0, end_s=end_s, pitch=60, velocity=80)


def test_offset_frames_sorted_unique_int32():
    # Three notes ending on frames 5, 5 (dup), 8 -> sorted-unique {5, 8}.
    ends = [5 * HOP / SR, 5 * HOP / SR, 8 * HOP / SR]
    offs = offset_frames([_note(e) for e in ends], n_frames=100)
    assert offs.dtype == np.int32
    assert offs.tolist() == [5, 8]


def test_offset_frames_clamps_tail_not_dropped():
    # A note ending past the recording tail keeps an offset event on the LAST frame (clamp, not
    # drop) — the deliberate difference from onset_frames.
    n_frames = 10
    far = _note(1.0)                    # frame_of(1.0) == 86 >> n_frames
    assert frame_of(1.0) >= n_frames
    offs = offset_frames([far], n_frames=n_frames)
    assert offs.tolist() == [n_frames - 1]


def test_offset_frames_empty():
    offs = offset_frames([], n_frames=10)
    assert offs.dtype == np.int32
    assert offs.size == 0


# --- boundary_distances (dataset searchsorted core) -----------------------------

def test_since_zero_on_onset_and_minus_one_before_first():
    # arr = single onset at frame 5; clip frames [3, 8).
    d = boundary_distances(np.array([5], dtype=np.int64), 3, 8, "since")
    #        f=3   4   5   6   7
    assert d.tolist() == [-1, -1, 0, 1, 2]


def test_until_zero_on_offset_and_minus_one_after_last():
    d = boundary_distances(np.array([5], dtype=np.int64), 3, 8, "until")
    #        f=3  4  5   6    7
    assert d.tolist() == [2, 1, 0, -1, -1]


def test_since_boundary_before_clip_start_still_counts():
    # Onset at frame 2 sits before the clip window [5, 8) but still supplies the distance.
    d = boundary_distances(np.array([2], dtype=np.int64), 5, 8, "since")
    assert d.tolist() == [3, 4, 5]


def test_nearest_boundary_wins_since():
    # Two onsets [5, 8]; at frame 9 the nearer (8, dist 1) wins over the farther (5, dist 4).
    d = boundary_distances(np.array([5, 8], dtype=np.int64), 6, 10, "since")
    #        f=6  7  8  9    (behind: 5,5,8,8)
    assert d.tolist() == [1, 2, 0, 1]


def test_nearest_boundary_wins_until():
    # Two offsets [5, 8]; at frame 6 the nearer ahead (8, dist 2) wins.
    d = boundary_distances(np.array([5, 8], dtype=np.int64), 4, 8, "until")
    #        f=4  5  6  7    (ahead: 5,5,8,8)
    assert d.tolist() == [1, 0, 2, 1]


def test_empty_array_all_minus_one():
    d = boundary_distances(np.array([], dtype=np.int64), 0, 4, "since")
    assert d.tolist() == [-1, -1, -1, -1]


# --- BoundarySinusoid -----------------------------------------------------------

def test_window_frames_from_50ms():
    # 50 ms at ~86.13 fps rounds to 4 frames.
    mod = BoundarySinusoid(TIME_SINCE_ONSET, FRAME_RATE)
    assert mod.window_frames == 4


def test_sinusoid_shape_and_compact_support():
    mod = BoundarySinusoid(TIME_SINCE_ONSET, FRAME_RATE)  # window_frames == 4, emb_dim == 64
    d = torch.tensor([[-1, 0, 3, 4, 10]], dtype=torch.long)
    out = mod(d)
    assert out.shape == (1, 5, 64)
    assert torch.isfinite(out).all()
    row_norms = out[0].norm(dim=-1)
    # Zero vector where inactive (d < 0 or d >= window_frames): indices 0, 3, 4.
    assert row_norms[0] == 0.0
    assert row_norms[3] == 0.0
    assert row_norms[4] == 0.0
    # Non-zero (finite) where active: d == 0 (cos-half is all ones) and d == 3.
    assert row_norms[1] > 0.0
    assert row_norms[2] > 0.0


# --- config round-trip + invariants ---------------------------------------------

def _mixed_conditioning() -> tuple:
    return (VELOCITY_COND, TIME_SINCE_ONSET, TIME_UNTIL_OFFSET)


def test_cfg_dict_roundtrip_mixed():
    cfg = AriosoConfig(conditioning=_mixed_conditioning())
    back = cfg_from_dict(cfg_to_dict(cfg))
    assert back.conditioning == cfg.conditioning
    assert isinstance(back.conditioning[0], CondSpec)
    assert isinstance(back.conditioning[1], BoundaryCondSpec)
    assert isinstance(back.conditioning[2], BoundaryCondSpec)


def test_cond_dim_sum_mixed():
    cfg = AriosoConfig(conditioning=_mixed_conditioning())
    assert cfg.cond_dim == VELOCITY_COND.emb_dim + TIME_SINCE_ONSET.emb_dim + TIME_UNTIL_OFFSET.emb_dim


def test_post_init_rejects_odd_emb_dim():
    with pytest.raises(AssertionError):
        AriosoConfig(conditioning=(BoundaryCondSpec("bad", emb_dim=63),))


def test_post_init_rejects_bad_boundary_and_direction():
    with pytest.raises(AssertionError):
        AriosoConfig(conditioning=(BoundaryCondSpec("bad_b", boundary="middle"),))
    with pytest.raises(AssertionError):
        AriosoConfig(conditioning=(BoundaryCondSpec("bad_d", direction="between"),))


# --- run_config parsing ---------------------------------------------------------

def test_parse_boundary_names_resolve():
    specs = _parse_conditioning(["velocity", "time_since_onset", "time_until_offset"], "test.yaml")
    assert [s.name for s in specs] == ["velocity", "time_since_onset", "time_until_offset"]
    assert isinstance(specs[1], BoundaryCondSpec)
    assert isinstance(specs[2], BoundaryCondSpec)


def test_parse_boundary_override():
    specs = _parse_conditioning([{"name": "time_since_onset", "window_ms": 75.0}], "test.yaml")
    assert isinstance(specs[0], BoundaryCondSpec)
    assert specs[0].window_ms == 75.0
    assert specs[0].boundary == "onset"  # unchanged from the registered spec


def test_parse_boundary_unknown_field_errors():
    with pytest.raises(ValueError, match="unknown BoundaryCondSpec field"):
        _parse_conditioning([{"name": "time_since_onset", "nonsense": 1}], "test.yaml")


# --- end-to-end: the shipped config resolves to four specs, cond_dim 208 ---------

def test_conditioned_gt_arky_yaml_resolves():
    # Four signals, not five: the env_dct work (2026-07) dropped the velocity conditioning track —
    # per-note dynamics are baked into prior_mel instead. cond_dim = 64 (articulation) + 16
    # (vibrato) + 64 + 64 (the two boundary signals).
    import os
    from Arioso.run_config import load_config

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "configs", "conditioned_gt_arky.yaml")
    cfg, _run = load_config(path)
    assert len(cfg.conditioning) == 4
    assert cfg.cond_dim == 208
