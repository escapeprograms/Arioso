"""project.json — the canonical Studio document: schema, atomic save, rev lock.

``project.json`` is the single source of truth for a song (notes + per-note
conditioning + tempo/meter + view + render metadata). The server owns
concurrency: every write goes through a per-project lock, is atomic (tmp +
``os.replace`` via :func:`Studio.library.atomic_write_json`), bumps a monotonic
``rev``, and rolls the previous version into ``project.backup/`` (last 20 revs).
PUTs from the editor carry the ``rev`` they were based on; a mismatch is a 409
``rev_conflict`` (last-writer-does-not-win). Note ids (``nNNNN``) are monotonic
and never reused (``next_note_ordinal``).

Timing is stored in **beats** (quarter notes) so editing is tempo-invariant — a
BPM change rescales seconds without rewriting notes (see :mod:`Studio.timing`).
This module is torch-free.
"""

from __future__ import annotations

import os
import threading
import time

from .config import SCHEMA_VERSION, StudioConfig
from .library import (BACKUP_DIR, atomic_write_json, project_dir, project_file,
                      read_json)

_KEEP_BACKUPS = 20         # rolling rev backups retained per project

# Per-project write locks (module-level so all requests share them).
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def lock_for(project_id: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lk = _LOCKS.get(project_id)
        if lk is None:
            lk = _LOCKS[project_id] = threading.Lock()
        return lk


# --- schema builders --------------------------------------------------------

def note_id(ordinal: int) -> str:
    return f"n{ordinal:04d}"


def default_vibrato() -> dict:
    """The default per-note vibrato block (a gentle 5.5 Hz, disabled by depth 0)."""
    return {"depth_semitones": 0.0, "rate_hz": 5.5, "onset_beats": 0.15}


def default_view() -> dict:
    """The default view/camera + editor-state block for a fresh project."""
    return {"px_per_beat": 48.0, "scroll_beat": 0.0, "top_pitch": 96,
            "lane": "velocity", "snap": "step"}


def default_render(cfg: StudioConfig) -> dict:
    """The default (empty) render-metadata block; filled once a render lands."""
    return {"model": cfg.default_model, "checkpoint": cfg.default_checkpoint,
            "prior_mode": cfg.default_prior_mode, "wav": None, "duration_s": None,
            "segments": []}


def build_note(ordinal: int, *, start_beat: float, len_beats: float, pitch: int,
               velocity: int = 100, technique: str = "normal", pan: float = 0.0,
               bend: list | None = None, vibrato: dict | None = None) -> dict:
    """Assemble one note dict with schema defaults.

    ``bend`` is a piecewise-linear list of ``{"beat", "semitones"}`` control points
    relative to the note start (empty = no bend). ``vibrato`` is a
    ``{depth_semitones, rate_hz, onset_beats}`` block. ``pan`` is a bipolar
    ``-1.0..+1.0`` stereo position (project-only for now, like ``technique``).
    """
    vib = default_vibrato()
    if vibrato:
        vib.update({k: float(v) for k, v in vibrato.items() if k in vib})
    return {
        "id": note_id(ordinal),
        "start_beat": round(float(start_beat), 6),
        "len_beats": round(float(len_beats), 6),
        "pitch": int(pitch),
        "velocity": int(max(1, min(127, round(velocity)))),
        "technique": str(technique),
        "pan": round(max(-1.0, min(1.0, float(pan))), 6),
        "bend": list(bend) if bend else [],
        "vibrato": vib,
    }


def build_document(cfg: StudioConfig, project_id: str, *, name: str,
                   notes: list[dict] | None = None, bpm: float = 140.0,
                   time_sig: dict | None = None, ppq: int = 480) -> dict:
    """Assemble a fresh project.json document (rev unset — 0).

    ``notes`` may be raw note dicts (with ``start_beat``/``len_beats``/``pitch`` and
    optional velocity/technique/bend/vibrato); they are normalized through
    :func:`build_note` and stamped with monotonic ids.
    """
    src = notes or []
    out_notes = [build_note(
        i,
        start_beat=n["start_beat"], len_beats=n["len_beats"], pitch=n["pitch"],
        velocity=n.get("velocity", 100), technique=n.get("technique", "normal"),
        pan=n.get("pan", 0.0), bend=n.get("bend"), vibrato=n.get("vibrato"),
    ) for i, n in enumerate(src)]
    return {
        "schema_version": SCHEMA_VERSION,
        "project_id": project_id,
        "rev": 0,
        "next_note_ordinal": len(out_notes),
        "name": str(name),
        "bpm": float(bpm),
        "time_sig": dict(time_sig) if time_sig else {"num": 4, "den": 4},
        "ppq": int(ppq),
        "render": default_render(cfg),
        "view": default_view(),
        "notes": out_notes,
    }


# --- load / backup / save ---------------------------------------------------

def load(cfg: StudioConfig, project_id: str) -> dict | None:
    return read_json(project_file(cfg, project_id))


def _backup_dir(cfg: StudioConfig, project_id: str) -> str:
    return os.path.join(project_dir(cfg, project_id), BACKUP_DIR)


def _roll_backup(cfg: StudioConfig, project_id: str, current: dict | None) -> None:
    """Copy the current project.json into project.backup/ before it is overwritten.

    Writes a ``rev_*`` backup and prunes old ``rev_*`` files to the newest 20.
    """
    if current is None:
        return
    bdir = _backup_dir(cfg, project_id)
    os.makedirs(bdir, exist_ok=True)
    rev = current.get("rev", 0)
    ts = time.strftime("%Y%m%d-%H%M%S")
    atomic_write_json(os.path.join(bdir, f"rev_{rev:05d}_{ts}.json"), current)
    revs = sorted(f for f in os.listdir(bdir) if f.startswith("rev_"))
    for stale in revs[:-_KEEP_BACKUPS]:
        try:
            os.remove(os.path.join(bdir, stale))
        except OSError:
            pass


def _write(cfg: StudioConfig, project_id: str, doc: dict) -> dict:
    atomic_write_json(project_file(cfg, project_id), doc)
    return doc


def save_new(cfg: StudioConfig, project_id: str, doc: dict) -> dict:
    """Write a brand-new document (rev -> 1) when none exists yet."""
    with lock_for(project_id):
        doc = dict(doc)
        doc["rev"] = 1
        doc["schema_version"] = SCHEMA_VERSION
        doc["project_id"] = project_id
        return _write(cfg, project_id, doc)


def create_project(cfg: StudioConfig, project_id: str, *, name: str) -> dict:
    """Build and persist a fresh empty project (rev 1)."""
    doc = build_document(cfg, project_id, name=name)
    return save_new(cfg, project_id, doc)


def put_with_rev(cfg: StudioConfig, project_id: str, incoming: dict) -> dict:
    """Apply an editor PUT with optimistic-concurrency rev check.

    Returns ``{"status": "ok", "rev": N, "doc": ...}`` on success, or
    ``{"status": "conflict", "server_rev": M}`` if the base rev is stale.
    """
    with lock_for(project_id):
        current = load(cfg, project_id)
        server_rev = current.get("rev", 0) if current else 0
        base_rev = int(incoming.get("rev", 0))
        if current is not None and base_rev != server_rev:
            return {"status": "conflict", "server_rev": server_rev}
        _roll_backup(cfg, project_id, current)
        doc = dict(incoming)
        doc["rev"] = server_rev + 1
        doc["schema_version"] = SCHEMA_VERSION
        doc["project_id"] = project_id
        _write(cfg, project_id, doc)
        return {"status": "ok", "rev": doc["rev"], "doc": doc}


def commit_rebuild(cfg: StudioConfig, project_id: str, new_doc: dict,
                   old_doc: dict | None) -> dict:
    """Persist a freshly (re)built document, bumping rev over any existing one."""
    with lock_for(project_id):
        current = load(cfg, project_id)
        _roll_backup(cfg, project_id, current)
        new_doc = dict(new_doc)
        new_doc["rev"] = (current.get("rev", 0) if current else 0) + 1
        new_doc["schema_version"] = SCHEMA_VERSION
        new_doc["project_id"] = project_id
        return _write(cfg, project_id, new_doc)


def summary(doc: dict) -> dict:
    """A compact listing summary of a loaded project document."""
    return {
        "id": doc.get("project_id"),
        "name": doc.get("name"),
        "rev": doc.get("rev", 0),
        "bpm": doc.get("bpm"),
        "n_notes": len(doc.get("notes", [])),
    }
