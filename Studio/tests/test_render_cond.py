"""Studio render-side conditioning: ``_segment_note_events`` + build_cond composition.

Unit-tests the torch-free helper that turns a segment sub-doc's notes into
:class:`~common.dataset_schema.NoteEvent`\\ s for conditioning (beats->seconds, the ``_MIN_NOTE_S``
floor, velocity clamp, the vibrato-depth predicate, unknown-technique fallback + warning), then one
composition test that runs the events through ``Arioso.infer.build_cond`` with the five
conditioned_gt_arky specs and sanity-checks articulation ids over note spans and boundary zeros at
onsets.
"""

from __future__ import annotations

import pytest

from common.dataset_schema import ARTICULATIONS, frame_of, onset_frames

from Arioso.config import BOUNDARY_SIGNALS, SIGNALS
from Arioso.infer import build_cond
from Studio.config import load_config
from Studio.midi_export import _MIN_NOTE_S
from Studio.render import _segment_note_events
from Studio.timing import beats_to_seconds

SPECS = (SIGNALS["articulation"], SIGNALS["velocity"], SIGNALS["vibrato"],
         BOUNDARY_SIGNALS["time_since_onset"], BOUNDARY_SIGNALS["time_until_offset"])

# Identity technique->model vocab (matches the shipped config after A3).
TECH_MAP = load_config().technique_model_vocab()


def _note(start_beat: float, len_beats: float, *, pitch: int = 60, velocity: int = 80,
          technique: str = "normal", vibrato=None) -> dict:
    n = {"start_beat": start_beat, "len_beats": len_beats, "pitch": pitch,
         "velocity": velocity, "technique": technique}
    if vibrato is not None:
        n["vibrato"] = vibrato
    return n


def _sub(notes: list[dict], bpm: float = 120.0) -> dict:
    return {"bpm": bpm, "notes": notes}


def test_beats_to_seconds_conversion():
    warnings: list[str] = []
    ev = _segment_note_events(_sub([_note(2.0, 1.0)], bpm=120.0), TECH_MAP, warnings)
    assert len(ev) == 1
    assert ev[0].start_s == beats_to_seconds(2.0, 120.0) == 1.0
    assert ev[0].end_s == pytest.approx(1.5)
    assert warnings == []


def test_min_note_s_floor():
    warnings: list[str] = []
    ev = _segment_note_events(_sub([_note(1.0, 0.0)]), TECH_MAP, warnings)
    assert ev[0].end_s - ev[0].start_s == pytest.approx(_MIN_NOTE_S)


def test_velocity_clamped_to_1_127():
    warnings: list[str] = []
    ev = _segment_note_events(
        _sub([_note(0.0, 1.0, velocity=200), _note(1.0, 1.0, velocity=0),
              _note(2.0, 1.0, velocity=64)]),
        TECH_MAP, warnings)
    assert [e.velocity for e in ev] == [127, 1, 64]


def test_vibrato_depth_predicate():
    warnings: list[str] = []
    notes = [
        _note(0.0, 1.0, vibrato={"depth_semitones": 0.3, "rate_hz": 5.0}),  # on
        _note(1.0, 1.0, vibrato={"depth_semitones": 0.0}),                  # off (zero depth)
        _note(2.0, 1.0, vibrato=None),                                      # off (null)
        _note(3.0, 1.0),                                                    # off (missing)
    ]
    ev = _segment_note_events(_sub(notes), TECH_MAP, warnings)
    assert [e.vibrato for e in ev] == [True, False, False, False]


def test_unknown_technique_falls_back_to_normal_with_one_warning():
    warnings: list[str] = []
    notes = [_note(0.0, 1.0, technique="pizzicato"),
             _note(1.0, 1.0, technique="pizzicato"),   # same unknown -> no second warning
             _note(2.0, 1.0, technique="spiccato")]
    ev = _segment_note_events(_sub(notes), TECH_MAP, warnings)
    assert [e.articulation for e in ev] == ["normal", "normal", "spiccato"]
    assert len(warnings) == 1
    assert "pizzicato" in warnings[0]


def test_known_technique_maps_through_identity_vocab_no_warning():
    warnings: list[str] = []
    notes = [_note(0.0, 1.0, technique="slur"), _note(1.0, 1.0, technique="detache")]
    ev = _segment_note_events(_sub(notes), TECH_MAP, warnings)
    # After the A3 identity flip slur->slur, detache->detache.
    assert [e.articulation for e in ev] == ["slur", "detache"]
    assert warnings == []


def test_composition_build_cond_over_segment_events():
    # A small sub-doc -> events -> the five-spec cond dict; articulation ids over each note's onset
    # frame match the mapped articulation, and every onset frame is a time_since_onset zero.
    warnings: list[str] = []
    notes = [_note(1.0, 2.0, pitch=60, technique="spiccato"),
             _note(4.0, 1.0, pitch=64, technique="detache")]
    ev = _segment_note_events(_sub(notes, bpm=120.0), TECH_MAP, warnings)
    n_frames = 300
    cond = build_cond(ev, n_frames, SPECS)
    assert set(cond) == {s.name for s in SPECS}

    for e in ev:
        f = frame_of(e.start_s)
        assert 0 <= f < n_frames
        assert cond["articulation"][f] == ARTICULATIONS.index(e.articulation)
        assert cond["time_since_onset"][f] == 0

    for f in onset_frames(ev, n_frames):
        assert cond["time_since_onset"][f] == 0
