"""Notes <-> MIDI: write a pretty_midi score the common.prior pipeline consumes.

Two roles:
  * write ``musc_raw.mid`` (the pre-alignment flatten of the raw transcription)
    that :mod:`Labeler.align` renders the offset-estimation prior from, and
  * write the export MIDI that ``common.prior.quantized_prior`` / ``note_onsets``
    / ``common.dataset_schema.note_groups_from_midi`` read back.

The score is deliberately plain — tempo 120, a single ``Instrument(program=40)``
"violin", integer velocities, **no pitch bends** — because the quantized prior and
the frame-export read only ``note.start``/``end``/``pitch``/``velocity`` and ignore
bends. ``validate_for_export`` flags the two conditions that break the downstream
math: zero/negative-duration notes (error) and same-pitch overlaps (warning).

Notes are passed as plain dicts (``start_s``, ``end_s``, ``pitch``, ``velocity``)
so this module never imports the notes-store schema or torch.
"""

from __future__ import annotations

import pretty_midi

from common.envelope import velocity_to_env_dct

from .config import CompileParams


def notes_to_pretty_midi(notes: list[dict], tempo: float = 120.0,
                         program: int = 40, with_env: bool = False,
                         with_vibrato: bool = False) -> pretty_midi.PrettyMIDI:
    """Build a single-instrument pretty_midi score from note dicts.

    Each note needs ``start_s``, ``end_s``, ``pitch``; ``velocity`` defaults to 80.
    Velocities are clamped to the MIDI-legal 1..127 (a 0 velocity would drop the
    note). Zero/negative-duration notes are skipped here (see ``validate_for_export``
    to surface them to the user first).

    When ``with_env`` is set, each in-memory ``Note`` gets a dynamic ``env_dct``
    attribute (the note's measured 3 DCT coeffs, or a flat env derived from velocity
    when the note has none) that ``common.prior``'s render loop reads to bake per-note
    dynamics into the prior. Default ``False`` leaves the ``Note`` objects (and every
    disk-MIDI writer) byte-identical to the legacy ``velocity/127`` path.

    When ``with_vibrato`` is set, a note that is flagged ``vibrato`` **and** carries
    measured ``vib_params`` for the current ``common.vibrato_model.VIB_MODEL`` gets
    ``vib_params``/``vib_model`` attributes that ``common.prior.QuantizedVibrato``
    replays. A vibrato-flagged note with no (or stale-model) params gets nothing —
    it renders flat rather than with an invented oscillation. Neither attribute
    survives a ``.write()``; both are in-memory only.
    """
    # High tick resolution keeps note.start/end faithful through the write/read
    # round-trip so onset frames (np.round(start*SR/HOP)) stay bit-stable.
    vib_model = None
    if with_vibrato:
        from common.vibrato_model import VIB_MODEL as vib_model  # lazy (scipy/librosa)
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo), resolution=480)
    inst = pretty_midi.Instrument(program=int(program), name="violin")
    for n in notes:
        start = float(n["start_s"])
        end = float(n["end_s"])
        if end <= start:
            continue
        vel = int(round(n.get("velocity", 80)))
        vel = max(1, min(127, vel))
        note_obj = pretty_midi.Note(
            velocity=vel, pitch=int(n["pitch"]), start=start, end=end)
        if with_env:
            env = n.get("env_dct")
            note_obj.env_dct = list(env) if env else velocity_to_env_dct(vel)
        if (with_vibrato and n.get("vibrato") and n.get("vib_params")
                and n.get("vib_model") == vib_model):
            note_obj.vib_params = list(n["vib_params"])
            note_obj.vib_model = vib_model
        inst.notes.append(note_obj)
    pm.instruments.append(inst)
    return pm


def write_midi(path: str, notes: list[dict], tempo: float = 120.0,
               program: int = 40) -> str:
    """Write ``notes`` to ``path`` as a MIDI file; return ``path``."""
    notes_to_pretty_midi(notes, tempo=tempo, program=program).write(path)
    return path


def write_export_midi(path: str, notes: list[dict], params: CompileParams) -> str:
    return write_midi(path, notes, tempo=params.tempo, program=params.program)


def validate_for_export(notes: list[dict]) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` for an export note list.

    * error: a note with ``end_s <= start_s`` (zero/negative duration) — the
      renderer would drop or mis-handle it.
    * warning: two notes of the same pitch whose spans overlap — the quantized
      prior would sum them oddly and the frame-export would merge onsets.
    """
    errors: list[str] = []
    warnings: list[str] = []
    for n in notes:
        if float(n["end_s"]) <= float(n["start_s"]):
            errors.append(
                f"note {n.get('id', '?')} pitch {n['pitch']} has non-positive "
                f"duration ({n['start_s']:.3f}..{n['end_s']:.3f}s)")

    by_pitch: dict[int, list[dict]] = {}
    for n in notes:
        by_pitch.setdefault(int(n["pitch"]), []).append(n)
    for pitch, group in by_pitch.items():
        group = sorted(group, key=lambda x: float(x["start_s"]))
        for a, b in zip(group, group[1:]):
            if float(b["start_s"]) < float(a["end_s"]) - 1e-6:
                warnings.append(
                    f"same-pitch overlap at pitch {pitch}: "
                    f"{a.get('id', '?')} ({a['start_s']:.3f}..{a['end_s']:.3f}) & "
                    f"{b.get('id', '?')} ({b['start_s']:.3f}..{b['end_s']:.3f})")
    return errors, warnings
