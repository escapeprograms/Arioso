""".mid file -> Studio notes (in **beats**) via the file's tempo map. Torch-free.

The inverse of :mod:`Studio.midi_export`: parse a MIDI file with ``pretty_midi``
and turn every (non-drum) note into a Studio note dict timed in **quarter-note
beats**, so an imported score lands on the same tempo-invariant timeline the editor
stores. The one subtle step is seconds -> beats: MIDI encodes onsets in *ticks*
(tempo-invariant, ``resolution`` ticks per quarter note), so a note's beat is
``pm.time_to_tick(seconds) / pm.resolution`` — this integrates the file's whole
tempo map exactly, so a file with mid-song tempo changes still imports onto the
right beats (a note two quarter notes in reads as beat 2.0 regardless of the tempi
in between). ``file_bpm`` is the file's initial tempo, offered to the caller as the
"adopt file BPM" value.

v1 scope (surfaced as ``warnings``): drum tracks are skipped; MIDI pitch-bend
events are **not** mapped onto per-note :mod:`Studio.bend` curves (imported notes
are flat); velocity carries over, technique defaults to ``normal``, vibrato to the
schema default. Bad/undecodable MIDI raises :class:`BadMidi` (the API maps it to a
400 ``bad_midi``).
"""

from __future__ import annotations

import io

import pretty_midi

from .timing import snap_beat, snap_grid

# A note shorter than one tick (or snapped to zero) is floored to this many beats
# so it survives validation (``len_beats > 0``) — 1/480 quarter note by default.
_MIN_LEN_BEATS = 1.0 / 480.0


class BadMidi(Exception):
    """Raised when the uploaded bytes / path is not a decodable MIDI file."""


def _load(src) -> pretty_midi.PrettyMIDI:
    """Parse ``src`` (bytes, a file-like, or a path string) into a ``PrettyMIDI``."""
    try:
        if isinstance(src, (bytes, bytearray)):
            return pretty_midi.PrettyMIDI(io.BytesIO(bytes(src)))
        if isinstance(src, str):
            return pretty_midi.PrettyMIDI(src)
        return pretty_midi.PrettyMIDI(src)          # already a file-like object
    except BadMidi:
        raise
    except Exception as exc:                        # mido / struct / value errors
        raise BadMidi(f"could not parse MIDI: {exc}") from exc


def import_midi(src, snap: str | None = None, adopt_bpm: bool = False,
                time_sig: dict | None = None) -> dict:
    """Parse a MIDI file into Studio notes (beats).

    ``src`` is raw MIDI ``bytes``, a file-like object, or a path. Returns
    ``{"notes": [...], "file_bpm": float, "warnings": [...]}`` where each note is a
    raw dict (``start_beat``/``len_beats``/``pitch``/``velocity``/``technique``/
    ``pan``/``bend``/``vibrato``) ready to feed :func:`Studio.project_store.build_note`.

    Notes from every non-drum instrument are merged and sorted by onset. Seconds are
    converted to beats through the file's tempo map (``time_to_tick / resolution``),
    so tempo changes are honoured. If ``snap`` is a grid id other than ``none``, each
    note's start and length are snapped to that grid (length floored to one grid
    cell). ``adopt_bpm`` does not alter beats (they are tempo-invariant) — it is
    echoed back so the caller can choose to set the project BPM to ``file_bpm``.
    """
    pm = _load(src)
    resolution = pm.resolution or 480
    ts = time_sig or {"num": 4, "den": 4}

    times, tempi = pm.get_tempo_changes()
    file_bpm = float(tempi[0]) if len(tempi) else 120.0

    warnings: list[str] = []
    if len(tempi) > 1:
        warnings.append(
            f"file has {len(tempi)} tempo changes; beats preserved via the tempo "
            f"map, project BPM set from the initial tempo ({file_bpm:.2f})")

    grid = 0.0
    if snap and snap != "none":
        grid = snap_grid(snap, ts)

    raw: list[dict] = []
    n_drum_tracks = 0
    n_bend_tracks = 0
    for inst in pm.instruments:
        if inst.is_drum:
            n_drum_tracks += 1
            continue
        if inst.pitch_bends:
            n_bend_tracks += 1
        for note in inst.notes:
            start_beat = pm.time_to_tick(note.start) / resolution
            end_beat = pm.time_to_tick(note.end) / resolution
            len_beats = end_beat - start_beat
            if grid > 0.0:
                start_beat = snap_beat(start_beat, grid)
                len_beats = snap_beat(len_beats, grid)
                if len_beats < _MIN_LEN_BEATS:
                    len_beats = grid
            if len_beats < _MIN_LEN_BEATS:
                len_beats = _MIN_LEN_BEATS
            raw.append({
                "start_beat": round(float(start_beat), 6),
                "len_beats": round(float(len_beats), 6),
                "pitch": int(note.pitch),
                "velocity": int(max(1, min(127, note.velocity))),
                "technique": "normal",
                "pan": 0.0,
                "bend": [],
                "vibrato": {"depth_semitones": 0.0, "rate_hz": 5.5, "onset_beats": 0.15},
            })

    raw.sort(key=lambda n: (n["start_beat"], n["pitch"]))

    if n_drum_tracks:
        warnings.append(f"skipped {n_drum_tracks} drum track(s)")
    if n_bend_tracks:
        warnings.append(
            f"{n_bend_tracks} track(s) had pitch-bend events; these are not imported "
            f"as per-note bends in this version (imported notes are flat)")
    if not raw:
        warnings.append("no notes found in the MIDI file")

    return {"notes": raw, "file_bpm": file_bpm, "warnings": warnings}
