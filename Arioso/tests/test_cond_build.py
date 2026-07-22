"""Inference-time conditioning builder: ``Arioso.infer.build_cond`` over hand-built notes.

Covers the shared cond builder the CLI (``Arioso.infer.main``) and the Studio renderer both drive,
using the exact five ``conditioned_gt_arky`` specs (resolved from the ``SIGNALS`` /
``BOUNDARY_SIGNALS`` registries — no data root needed, all pure / CPU):

* keys == spec names; categorical tracks uint8, boundary tracks int64 (``-1`` sentinel safe).
* semantics: distance 0 on onset/offset frames; pre-first-onset frame ``-1``; tail-offset clamp
  still yields a boundary on the last frame; simultaneous notes merge to one boundary.
* training equivalence: the boundary track equals ``boundary_distances`` over a saved-onsets array.
* refactor guard: constant-articulation notes reproduce the old ``build_cond_frames`` output
  (``expand_to_frames`` over ``note_groups`` with a constant id).
"""

from __future__ import annotations

import numpy as np

from common.config import HOP_SIZE as HOP, SR
from common.dataset_schema import (ARTIC_REST_ID, ARTICULATIONS, NoteEvent,
                                   expand_to_frames, note_groups, offset_frames,
                                   onset_frames)

from Arioso.config import BOUNDARY_SIGNALS, SIGNALS
from Arioso.dataset import boundary_distances
from Arioso.infer import build_cond

# The five conditioned_gt_arky signals, resolved through the registries.
SPECS = (SIGNALS["articulation"], SIGNALS["velocity"], SIGNALS["vibrato"],
         BOUNDARY_SIGNALS["time_since_onset"], BOUNDARY_SIGNALS["time_until_offset"])
SPEC_NAMES = {"articulation", "velocity", "vibrato",
              "time_since_onset", "time_until_offset"}


def _t(frame: int) -> float:
    """Seconds whose ``frame_of`` rounds exactly to ``frame`` (frame * HOP / SR)."""
    return frame * HOP / SR


def _note(onset_f: int, offset_f: int, *, pitch: int = 60, velocity: int = 80,
          articulation: str = "normal", vibrato: bool = False) -> NoteEvent:
    return NoteEvent(start_s=_t(onset_f), end_s=_t(offset_f), pitch=pitch,
                     velocity=velocity, articulation=articulation, vibrato=vibrato)


def test_keys_match_spec_names():
    notes = [_note(5, 15), _note(30, 40)]
    cond = build_cond(notes, 60, SPECS)
    assert set(cond) == SPEC_NAMES


def test_categorical_dtype_uint8_boundary_int64():
    notes = [_note(5, 15), _note(30, 40)]
    cond = build_cond(notes, 60, SPECS)
    for name in ("articulation", "velocity", "vibrato"):
        assert cond[name].dtype == np.uint8, name
    for name in ("time_since_onset", "time_until_offset"):
        assert cond[name].dtype == np.int64, name


def test_boundary_min_at_least_minus_one_and_pre_onset_sentinel():
    notes = [_note(5, 15), _note(30, 40)]
    cond = build_cond(notes, 60, SPECS)
    since = cond["time_since_onset"]
    assert since.min() >= -1
    # First onset is frame 5, so frames 0..4 have no onset behind them -> sentinel -1.
    assert since[0] == -1
    assert (since[:5] == -1).all()


def test_distance_zero_on_onset_and_offset_frames():
    notes = [_note(5, 15), _note(30, 40)]
    cond = build_cond(notes, 60, SPECS)
    # time_since_onset == 0 exactly on each onset frame.
    for f in onset_frames(notes, 60):
        assert cond["time_since_onset"][f] == 0
    # time_until_offset == 0 exactly on each offset frame.
    for f in offset_frames(notes, 60):
        assert cond["time_until_offset"][f] == 0


def test_tail_note_offset_clamps_to_last_frame():
    # A note whose end rounds past n_frames keeps an offset boundary on the LAST frame
    # (offset_frames clamps, not drops) -> time_until_offset == 0 there.
    n_frames = 20
    notes = [_note(2, 200)]                 # offset frame 200 >> n_frames
    cond = build_cond(notes, n_frames, SPECS)
    assert cond["time_until_offset"][n_frames - 1] == 0


def test_simultaneous_notes_merge_to_one_boundary():
    # Two notes sharing an onset frame (a double-stop) collapse to one onset boundary.
    notes = [_note(5, 15, pitch=60), _note(5, 15, pitch=64)]
    assert onset_frames(notes, 60).tolist() == [5]
    cond = build_cond(notes, 60, SPECS)
    since = cond["time_since_onset"]
    assert since[5] == 0
    # A single onset at 5 -> monotone ramp afterwards, sentinel before.
    assert since[6] == 1 and since[7] == 2


def test_boundary_track_equals_saved_onsets_path():
    # Training-path equivalence: build_cond's boundary track is exactly boundary_distances over
    # the saved (int32) onsets array cast to int64 — the dataset's on-disk boundary path.
    notes = [_note(5, 15), _note(30, 40), _note(45, 55)]
    n_frames = 60
    cond = build_cond(notes, n_frames, SPECS)
    saved = onset_frames(notes, n_frames)   # int32 array, as written to onsets/<base>.npy
    expected = boundary_distances(saved.astype(np.int64), 0, n_frames, "since")
    assert np.array_equal(cond["time_since_onset"], expected)


def test_constant_articulation_matches_old_build_cond_frames():
    # Refactor guard: a constant articulation over all notes reproduces the old build_cond_frames
    # path (expand_to_frames over note_groups with a constant id).
    n_frames = 60
    notes = [_note(5, 15, articulation="spiccato"),
             _note(30, 40, articulation="spiccato")]
    cond = build_cond(notes, n_frames, SPECS)

    groups = note_groups(notes, 0.0, n_frames)
    ids = np.full(len(groups), ARTICULATIONS.index("spiccato"), dtype=np.uint8)
    expected = expand_to_frames(groups, ids, n_frames, rest_id=ARTIC_REST_ID)
    assert np.array_equal(cond["articulation"], expected)


def test_velocity_and_vibrato_span_fill():
    # Sanity: velocity carries the note's MIDI velocity over its span (rest 0 outside);
    # vibrato flag flips on over a vibrato note's span.
    n_frames = 60
    notes = [_note(5, 15, velocity=99, vibrato=True)]
    cond = build_cond(notes, n_frames, SPECS)
    assert cond["velocity"][10] == 99
    assert cond["velocity"][0] == 0            # pre-onset rest
    assert cond["vibrato"][10] == 1            # VIBRATO_ON over the span
    assert cond["vibrato"][0] == 2             # VIBRATO_REST_ID outside


def test_empty_specs_gives_empty_dict():
    assert build_cond([_note(5, 15)], 60, ()) == {}
