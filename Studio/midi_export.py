"""Project document -> ``pretty_midi.PrettyMIDI`` for the render pipeline / export.

Turns a Studio ``project.json`` (notes in **beats**, per-note bend/vibrato) into a
MIDI score the ``common.prior`` saw prior consumes. The one non-trivial step is
polyphony: overlapping notes are split into monophonic **voices**
(:mod:`Studio.voices`), one ``pretty_midi.Instrument(program=40)`` each, so every
voice carries an unambiguous pitch-wheel curve; :class:`PriorSynth` sums the
instrument tracks back into polyphony at render time. Per-voice pitch bends
(:mod:`Studio.bend`) are attached only in ``prior_mode == "bend"`` — the quantized
dev mode emits a plain note grid with no wheel events, so its prior is the same one
training used.

Timing crosses to seconds here (the only place seconds enter, via
:mod:`Studio.timing` with the document BPM). ``PrettyMIDI`` is built at the
document's ``bpm``/``ppq`` so onset seconds survive the write/read round-trip.
Validation mirrors :func:`Labeler.midi_io.validate_for_export`: zero/negative
length notes are a hard error; same-pitch overlaps are a warning surfaced to the
caller. Torch-free.
"""

from __future__ import annotations

import pretty_midi

from common.envelope import velocity_to_env_dct

from .bend import pitch_bend_events
from .timing import beats_to_seconds
from .voices import partition_voices

# Notes shorter than this are stretched to it so the prior renders at least a couple
# of samples (a beats-length of exactly 0 is rejected as an error before this).
_MIN_NOTE_S = 1e-3

_VIOLIN_PROGRAM = 40  # General MIDI "Violin"


def select_notes(doc: dict, note_ids: list[str] | None) -> list[dict]:
    """All notes, or only those whose id is in ``note_ids`` (order preserved)."""
    notes = doc.get("notes", [])
    if note_ids is None:
        return list(notes)
    wanted = set(note_ids)
    return [n for n in notes if n["id"] in wanted]


def validate_notes(notes: list[dict]) -> tuple[list[str], list[str]]:
    """``(errors, warnings)`` for a note selection (in beats).

    * error: ``len_beats <= 0`` (zero/negative duration — the prior would drop it).
    * warning: two same-pitch notes whose beat spans overlap (summed oddly).
    """
    errors: list[str] = []
    for n in notes:
        if float(n["len_beats"]) <= 0.0:
            errors.append(
                f"note {n.get('id', '?')} pitch {n.get('pitch', '?')} has "
                f"non-positive length ({float(n['len_beats']):.4f} beats)")

    by_pitch: dict[int, list[dict]] = {}
    for n in notes:
        by_pitch.setdefault(int(n["pitch"]), []).append(n)
    warnings: list[str] = []
    for pitch, group in by_pitch.items():
        group = sorted(group, key=lambda x: float(x["start_beat"]))
        for a, b in zip(group, group[1:]):
            a_end = float(a["start_beat"]) + float(a["len_beats"])
            if float(b["start_beat"]) < a_end - 1e-6:
                warnings.append(
                    f"same-pitch overlap at pitch {pitch}: "
                    f"{a.get('id', '?')} & {b.get('id', '?')}")
    return errors, warnings


def project_to_pretty_midi(doc: dict, note_ids: list[str] | None = None,
                           prior_mode: str = "bend") -> pretty_midi.PrettyMIDI:
    """Build a ``PrettyMIDI`` from ``doc`` (all notes, or the ``note_ids`` subset).

    One instrument per voice (``violin_v0``, ``violin_v1``, ...), velocities clamped
    to 1..127, zero-length notes stretched to ``_MIN_NOTE_S``. In ``bend`` mode each
    voice gets per-note pitch-wheel events + a reset in each trailing gap. Raises
    ``ValueError`` if any selected note has non-positive length.
    """
    bpm = float(doc["bpm"])
    ppq = int(doc.get("ppq", 480))
    notes = select_notes(doc, note_ids)
    errors, _warnings = validate_notes(notes)
    if errors:
        raise ValueError("cannot build MIDI: " + "; ".join(errors))

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm, resolution=ppq)
    if not notes:
        return pm

    for k, voice in enumerate(partition_voices(notes)):
        inst = pretty_midi.Instrument(program=_VIOLIN_PROGRAM, name=f"violin_v{k}")
        vnotes = sorted(voice, key=lambda n: float(n["start_beat"]))
        for i, n in enumerate(vnotes):
            start_s = beats_to_seconds(float(n["start_beat"]), bpm)
            end_s = start_s + max(beats_to_seconds(float(n["len_beats"]), bpm), _MIN_NOTE_S)
            vel = max(1, min(127, int(round(float(n.get("velocity", 100))))))
            note_obj = pretty_midi.Note(
                velocity=vel, pitch=int(n["pitch"]), start=start_s, end=end_s)
            # Bake the per-note energy envelope into the prior: an explicit env_dct
            # lane (future) wins; otherwise derive a flat env from velocity. Set as a
            # dynamic attribute PriorSynth.render reads (common.prior:422).
            note_obj.env_dct = (list(n["env_dct"]) if n.get("env_dct")
                                else velocity_to_env_dct(vel))
            inst.notes.append(note_obj)
            if prior_mode == "bend":
                nxt = (beats_to_seconds(float(vnotes[i + 1]["start_beat"]), bpm)
                       if i + 1 < len(vnotes) else None)
                for t, wheel in pitch_bend_events(n, start_s, end_s - start_s, bpm, nxt):
                    inst.pitch_bends.append(pretty_midi.PitchBend(pitch=int(wheel), time=float(t)))
        inst.pitch_bends.sort(key=lambda pb: pb.time)
        pm.instruments.append(inst)
    return pm


def write_midi(doc: dict, path: str, note_ids: list[str] | None = None,
               prior_mode: str = "bend") -> tuple[str, list[str]]:
    """Write ``doc`` to ``path`` as MIDI; return ``(path, warnings)``.

    Raises ``ValueError`` on non-positive-length notes (via
    :func:`project_to_pretty_midi`); same-pitch-overlap warnings are returned for the
    caller to surface (e.g. into the render meta).
    """
    _errors, warnings = validate_notes(select_notes(doc, note_ids))
    pm = project_to_pretty_midi(doc, note_ids=note_ids, prior_mode=prior_mode)
    pm.write(path)
    return path, warnings
