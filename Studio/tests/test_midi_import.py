"""midi_import: tempo-map beats, snap, pitch-bend/drum warnings, bad-midi."""

from __future__ import annotations

import io

import mido
import pretty_midi
import pytest

from Studio.midi_import import BadMidi, import_midi


def _simple_midi_bytes(bpm=120.0, resolution=480):
    """Two sequential quarter notes at ``bpm`` -> beats 0 and 1."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm, resolution=resolution)
    inst = pretty_midi.Instrument(program=40)
    spb = 60.0 / bpm
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=spb))
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=64, start=spb, end=2 * spb))
    pm.instruments.append(inst)
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()


def _tempo_change_midi_bytes(resolution=480):
    """One note per beat across a mid-file tempo change (120 -> 240 bpm at beat 1).

    Notes sit on ticks 0, 480, 960 (beats 0, 1, 2) regardless of the tempi, so a
    correct seconds->beats via the tempo map must recover exactly 0.0, 1.0, 2.0.
    """
    mid = mido.MidiFile(ticks_per_beat=resolution)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    tr.append(mido.Message("note_on", note=60, velocity=100, time=0))
    tr.append(mido.Message("note_off", note=60, velocity=0, time=resolution))
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(240), time=0))
    tr.append(mido.Message("note_on", note=62, velocity=100, time=0))
    tr.append(mido.Message("note_off", note=62, velocity=0, time=resolution))
    tr.append(mido.Message("note_on", note=64, velocity=100, time=0))
    tr.append(mido.Message("note_off", note=64, velocity=0, time=resolution))
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def test_import_simple_beats_from_bytes():
    res = import_midi(_simple_midi_bytes(bpm=120.0))
    beats = [n["start_beat"] for n in res["notes"]]
    assert beats == [0.0, 1.0]
    assert res["file_bpm"] == pytest.approx(120.0)
    assert all(n["technique"] == "normal" and n["bend"] == [] for n in res["notes"])


def test_import_from_path(tmp_path):
    p = tmp_path / "in.mid"
    p.write_bytes(_simple_midi_bytes(bpm=100.0))
    res = import_midi(str(p))
    assert [n["start_beat"] for n in res["notes"]] == [0.0, 1.0]


def test_tempo_map_preserves_beats():
    res = import_midi(_tempo_change_midi_bytes())
    beats = [n["start_beat"] for n in res["notes"]]
    assert beats == [0.0, 1.0, 2.0]
    # initial tempo is offered as file_bpm; a warning notes the tempo changes.
    assert res["file_bpm"] == pytest.approx(120.0)
    assert any("tempo change" in w for w in res["warnings"])


def test_snap_quantizes_starts():
    # A note a bit off the grid (0.2s @ 120bpm = beat 0.4) snaps to beat 0 at "beat".
    pm = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=480)
    inst = pretty_midi.Instrument(program=40)
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=67, start=0.2, end=0.7))
    pm.instruments.append(inst)
    buf = io.BytesIO()
    pm.write(buf)
    res = import_midi(buf.getvalue(), snap="beat")
    assert res["notes"][0]["start_beat"] == pytest.approx(0.0)


def test_pitch_bend_warns_and_notes_stay_flat():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=480)
    inst = pretty_midi.Instrument(program=40)
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=67, start=0.0, end=0.5))
    inst.pitch_bends.append(pretty_midi.PitchBend(pitch=2000, time=0.1))
    pm.instruments.append(inst)
    buf = io.BytesIO()
    pm.write(buf)
    res = import_midi(buf.getvalue())
    assert any("pitch-bend" in w for w in res["warnings"])
    assert res["notes"][0]["bend"] == []


def test_drum_track_skipped():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=480)
    drum = pretty_midi.Instrument(program=0, is_drum=True)
    drum.notes.append(pretty_midi.Note(velocity=100, pitch=38, start=0.0, end=0.25))
    mel = pretty_midi.Instrument(program=40)
    mel.notes.append(pretty_midi.Note(velocity=100, pitch=67, start=0.0, end=0.5))
    pm.instruments.append(drum)
    pm.instruments.append(mel)
    buf = io.BytesIO()
    pm.write(buf)
    res = import_midi(buf.getvalue())
    assert len(res["notes"]) == 1 and res["notes"][0]["pitch"] == 67
    assert any("drum" in w for w in res["warnings"])


def test_velocity_preserved_and_clamped():
    res = import_midi(_simple_midi_bytes())
    assert all(1 <= n["velocity"] <= 127 for n in res["notes"])
    assert res["notes"][0]["velocity"] == 90


def test_bad_midi_raises():
    with pytest.raises(BadMidi):
        import_midi(b"not a midi file at all")


def test_empty_midi_warns():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=480)
    buf = io.BytesIO()
    pm.write(buf)
    res = import_midi(buf.getvalue())
    assert res["notes"] == []
    assert any("no notes" in w for w in res["warnings"])
