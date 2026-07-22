"""MIDI round-trip: write_midi -> note_groups_from_midi onset frames == np.round."""

from __future__ import annotations

import numpy as np

from common.config import HOP_SIZE as HOP, SR
from common.dataset_schema import note_groups_from_midi
from Labeler.midi_io import write_midi


def test_onset_frames_match_np_round(tmp_path):
    # Starts a safe fraction off the frame grid so quantization can't flip np.round.
    ks = [5, 20, 51, 120, 240]
    starts = [(k + 0.3) * HOP / SR for k in ks]
    notes = [{"start_s": s, "end_s": s + 0.4, "pitch": 60 + i, "velocity": 90}
             for i, s in enumerate(starts)]
    path = str(tmp_path / "roundtrip.mid")
    write_midi(path, notes)

    n_frames = int(np.ceil((max(starts) + 1.0) * SR / HOP))
    groups = note_groups_from_midi(path, 0.0, n_frames)

    got = [g.onset_frame for g in groups]
    expected = sorted({int(np.round(s * SR / HOP)) for s in starts})
    assert got == expected == ks


def test_chord_merges_to_one_group(tmp_path):
    # Two notes sharing an onset frame collapse into a single group.
    s = (30 + 0.3) * HOP / SR
    notes = [{"start_s": s, "end_s": s + 0.4, "pitch": 60, "velocity": 90},
             {"start_s": s, "end_s": s + 0.4, "pitch": 64, "velocity": 90}]
    path = str(tmp_path / "chord.mid")
    write_midi(path, notes)
    n_frames = int(np.ceil((s + 1.0) * SR / HOP))
    groups = note_groups_from_midi(path, 0.0, n_frames)
    assert len(groups) == 1
    assert groups[0].onset_frame == 30
    assert groups[0].n_notes == 2
