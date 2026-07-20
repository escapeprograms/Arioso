"""Partition project notes into monophonic voices (greedy interval scheduling).

The saw prior renders per-note pitch bends off ``inst.pitch_bends`` — a single
pitch-wheel curve per instrument track. Overlapping notes (chords, double-stops)
would share one wheel and their bends would collide, so before MIDI export the
notes are split into **voices**: each voice is a monophonic stream (no two notes
overlap in time) and becomes its own ``pretty_midi.Instrument`` with its own
unambiguous bend curve. :mod:`Studio.midi_export` sums the voices back into
polyphony exactly as :class:`DataSynthesizer.synthesizePrior.PriorSynth` sums
across instrument tracks.

The partition is a classic greedy interval-graph coloring: sort by start, drop
each note into the first voice whose last note has already ended, else open a new
voice. It uses the minimum number of voices for the given overlaps and is fully
deterministic (a stable key breaks ties). A hard cap (``max_voices``) guards
against pathological dense chords; overflow raises :class:`VoiceOverflow` naming
the notes that could not be placed. Torch-free; operates on plain note dicts.
"""

from __future__ import annotations

# Notes whose gap is within this many beats are treated as non-overlapping, so a
# note may reuse a voice whose previous note ends exactly at (or a hair before)
# its start — the common back-to-back legato case.
_EPS = 1e-6

# Default ceiling on simultaneous voices (== max polyphony). Dense chords beyond
# this are almost always an editing mistake; surfacing them beats silently
# spawning dozens of instruments.
MAX_VOICES = 8


class VoiceOverflow(Exception):
    """Raised when notes overlap in more than ``max_voices`` simultaneous voices."""


def _end(note: dict) -> float:
    return float(note["start_beat"]) + float(note["len_beats"])


def _describe(note: dict) -> str:
    return (f"{note.get('id', '?')} (pitch {note.get('pitch', '?')} "
            f"@ {float(note['start_beat']):.3f}..{_end(note):.3f} beats)")


def partition_voices(notes: list[dict], max_voices: int = MAX_VOICES) -> list[list[dict]]:
    """Greedily partition ``notes`` into monophonic voices.

    Returns a list of voices (each a list of note dicts, ordered by start). Notes
    are sorted by ``(start_beat, end_beat, pitch, id)`` for determinism, then each
    is placed in the first voice whose last note ends ``<= start + _EPS``; if none
    qualifies a new voice is opened. Raises :class:`VoiceOverflow` (listing the
    offending note and the notes already sounding) if that would exceed
    ``max_voices``.
    """
    ordered = sorted(
        notes,
        key=lambda n: (float(n["start_beat"]), _end(n), int(n.get("pitch", 0)),
                       str(n.get("id", ""))),
    )
    voices: list[list[dict]] = []
    voice_end: list[float] = []          # last note end per voice (parallel to `voices`)
    for note in ordered:
        start = float(note["start_beat"])
        placed = False
        for i, vend in enumerate(voice_end):
            if vend <= start + _EPS:
                voices[i].append(note)
                voice_end[i] = _end(note)
                placed = True
                break
        if placed:
            continue
        if len(voices) >= max_voices:
            sounding = [v[-1] for v, e in zip(voices, voice_end) if e > start + _EPS]
            raise VoiceOverflow(
                f"more than {max_voices} overlapping voices at beat {start:.3f}: "
                f"cannot place {_describe(note)}; already sounding: "
                + ", ".join(_describe(n) for n in sounding))
        voices.append([note])
        voice_end.append(_end(note))
    return voices
