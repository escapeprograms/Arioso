"""midi_export: voices->instruments, bend gating by prior_mode, velocity clamp."""

from __future__ import annotations

import pretty_midi
import pytest

from Studio.midi_export import (project_to_pretty_midi, select_notes,
                                validate_notes, write_midi)


def _note(nid, start, length, pitch=67, velocity=100, bend=None, vibrato=None):
    return {"id": nid, "start_beat": start, "len_beats": length, "pitch": pitch,
            "velocity": velocity, "bend": bend or [],
            "vibrato": vibrato or {"depth_semitones": 0.0, "rate_hz": 5.5, "onset_beats": 0.15}}


def _doc(notes, bpm=120.0, ppq=480):
    return {"project_id": "t", "bpm": bpm, "ppq": ppq, "notes": notes}


def test_overlap_makes_two_instruments():
    doc = _doc([_note("n0", 0.0, 2.0, 60), _note("n1", 0.5, 2.0, 64)])
    pm = project_to_pretty_midi(doc)
    assert len(pm.instruments) == 2
    assert [i.name for i in pm.instruments] == ["violin_v0", "violin_v1"]
    assert all(i.program == 40 for i in pm.instruments)


def test_sequential_one_instrument():
    doc = _doc([_note("n0", 0.0, 1.0), _note("n1", 1.0, 1.0)])
    pm = project_to_pretty_midi(doc)
    assert len(pm.instruments) == 1
    assert len(pm.instruments[0].notes) == 2


def test_bend_events_only_in_bend_mode():
    bent = _note("n0", 0.0, 1.0, bend=[{"beat": 0.0, "semitones": 0.0},
                                       {"beat": 1.0, "semitones": 2.0}])
    doc = _doc([bent])
    pm_bend = project_to_pretty_midi(doc, prior_mode="bend")
    pm_quant = project_to_pretty_midi(doc, prior_mode="quantized")
    assert len(pm_bend.instruments[0].pitch_bends) > 0
    assert len(pm_quant.instruments[0].pitch_bends) == 0


def test_flat_note_no_bend_events():
    doc = _doc([_note("n0", 0.0, 1.0)])
    pm = project_to_pretty_midi(doc, prior_mode="bend")
    assert len(pm.instruments[0].pitch_bends) == 0


def test_velocity_clamped():
    doc = _doc([_note("n0", 0.0, 1.0, velocity=999), _note("n1", 1.0, 1.0, velocity=0)])
    pm = project_to_pretty_midi(doc)
    vels = sorted(n.velocity for n in pm.instruments[0].notes)
    assert vels == [1, 127]


def test_selection_subset():
    doc = _doc([_note("n0", 0.0, 1.0), _note("n1", 1.0, 1.0), _note("n2", 2.0, 1.0)])
    assert [n["id"] for n in select_notes(doc, ["n0", "n2"])] == ["n0", "n2"]
    pm = project_to_pretty_midi(doc, note_ids=["n0", "n2"])
    assert sum(len(i.notes) for i in pm.instruments) == 2


def test_zero_length_note_errors():
    errors, _w = validate_notes([_note("n0", 0.0, 0.0)])
    assert errors
    with pytest.raises(ValueError):
        project_to_pretty_midi(_doc([_note("n0", 0.0, 0.0)]))


def test_same_pitch_overlap_warns():
    _e, warnings = validate_notes([_note("n0", 0.0, 2.0, 60), _note("n1", 0.5, 2.0, 60)])
    assert warnings


def test_write_midi_roundtrip(tmp_path):
    doc = _doc([_note("n0", 0.0, 1.0, 60), _note("n1", 1.0, 1.0, 62)])
    path = str(tmp_path / "out.mid")
    written, warnings = write_midi(doc, path)
    assert warnings == []
    pm = pretty_midi.PrettyMIDI(written)
    assert sum(len(i.notes) for i in pm.instruments) == 2
