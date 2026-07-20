"""voices: greedy monophonic partition of overlapping notes. Torch-free."""

from __future__ import annotations

import pytest

from Studio.voices import VoiceOverflow, partition_voices


def _n(nid, start, length, pitch=67):
    return {"id": nid, "start_beat": start, "len_beats": length, "pitch": pitch}


def test_sequential_notes_one_voice():
    notes = [_n("n0", 0.0, 1.0), _n("n1", 1.0, 1.0), _n("n2", 2.0, 1.0)]
    voices = partition_voices(notes)
    assert len(voices) == 1
    assert [n["id"] for n in voices[0]] == ["n0", "n1", "n2"]


def test_two_overlapping_notes_split():
    notes = [_n("n0", 0.0, 2.0, 60), _n("n1", 0.5, 2.0, 64)]
    voices = partition_voices(notes)
    assert len(voices) == 2
    # Each voice holds exactly one of the overlapping notes.
    assert {v[0]["id"] for v in voices} == {"n0", "n1"}


def test_chord_three_voices():
    notes = [_n("a", 0.0, 1.0, 60), _n("b", 0.0, 1.0, 64), _n("c", 0.0, 1.0, 67)]
    voices = partition_voices(notes)
    assert len(voices) == 3


def test_back_to_back_reuses_voice():
    # n1 starts exactly where n0 ends -> same voice (epsilon-tolerant).
    notes = [_n("n0", 0.0, 1.0), _n("n1", 1.0, 1.0)]
    voices = partition_voices(notes)
    assert len(voices) == 1


def test_deterministic_order():
    notes = [_n("z", 0.0, 2.0, 72), _n("a", 0.5, 2.0, 60), _n("m", 1.0, 2.0, 65)]
    v1 = partition_voices(notes)
    v2 = partition_voices(list(reversed(notes)))
    assert [[n["id"] for n in v] for v in v1] == [[n["id"] for n in v] for v in v2]


def test_voice_overflow_raises():
    # 3 fully-overlapping notes with a cap of 2 -> overflow.
    notes = [_n(f"n{i}", 0.0, 1.0, 60 + i) for i in range(3)]
    with pytest.raises(VoiceOverflow) as exc:
        partition_voices(notes, max_voices=2)
    assert "n2" in str(exc.value)
