"""notes.json — the canonical annotation document: schema, atomic save, merge.

``notes.json`` is the single source of truth for a clip's labels. The server owns
concurrency: every write goes through a per-clip lock, is atomic (tmp +
``os.replace`` via :func:`Labeler.library.atomic_write_json`), bumps a monotonic
``rev``, and rolls the previous version into ``notes.backup/`` (last 20 revs +
named snapshots). PUTs from the editor carry the ``rev`` they were based on; a
mismatch is a 409 ``rev_conflict`` (last-writer-does-not-win).

**Label preservation is structural.** Note ids (``nNNNN``) are monotonic and never
reused (``next_note_ordinal``). Each note carries an ``auto`` block (the frozen
pipeline estimate) and a ``human`` block of per-facet edited-flags
(``technique``/``vibrato``/``velocity``/``timing``). :func:`merge_retranscribe`
re-labels a fresh transcription while carrying every human-flagged facet across by
matching old->new notes (equal pitch, ``|Δonset| <= 60 ms``, greedy nearest, once);
human-touched notes with no match land in ``orphaned_labels`` for a UI panel.
"""

from __future__ import annotations

import os
import shutil
import threading
import time

from .config import LabelerConfig, SCHEMA_VERSION
from .library import (BACKUP_DIR, NOTES_JSON, atomic_write_json, clip_dir,
                      clip_file, read_json)

_MATCH_TOL_S = 0.060       # onset tolerance for old<->new note matching
_KEEP_BACKUPS = 20         # rolling rev backups retained (snapshots are exempt)
_HUMAN_FACETS = ("technique", "vibrato", "velocity", "timing")

# Per-clip write locks (module-level so all requests share them).
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def lock_for(clip_id: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lk = _LOCKS.get(clip_id)
        if lk is None:
            lk = _LOCKS[clip_id] = threading.Lock()
        return lk


# --- schema builders --------------------------------------------------------

def note_id(ordinal: int) -> str:
    return f"n{ordinal:04d}"


def blank_human() -> dict:
    return {facet: False for facet in _HUMAN_FACETS}


def build_note(ordinal: int, *, start_s: float, end_s: float, pitch: int,
               velocity: int, technique: str, vibrato: bool, auto: dict) -> dict:
    return {
        "id": note_id(ordinal),
        "start_s": round(float(start_s), 6),
        "end_s": round(float(end_s), 6),
        "pitch": int(pitch),
        "velocity": int(max(1, min(127, round(velocity)))),
        "technique": str(technique),
        "vibrato": bool(vibrato),
        "slur_group": None,
        "auto": {
            "velocity": int(auto.get("velocity", velocity)),
            "technique": str(auto.get("technique", technique)),
            "vibrato": bool(auto.get("vibrato", vibrato)),
            "amplitude": float(auto.get("amplitude", 0.0)),
            "onset_strength": float(auto.get("onset_strength", 0.0)),
            "vibrato_rate_hz": float(auto.get("vibrato_rate_hz", 0.0)),
            "vibrato_extent_cents": float(auto.get("vibrato_extent_cents", 0.0)),
        },
        "human": blank_human(),
    }


def build_document(cfg: LabelerConfig, clip_id: str, notes: list[dict], *,
                   offset_applied_s: float, duration_s: float,
                   audio_meta: dict, pipeline_meta: dict) -> dict:
    """Assemble a fresh notes.json document (rev unset — 0) from finalized notes.

    ``notes`` are already-final note dicts (aligned times, velocity, technique,
    vibrato, and their ``auto`` blocks); this stamps ids/ordinals and the clip-level
    metadata / vocabulary snapshot.
    """
    out_notes = []
    for i, n in enumerate(notes):
        out_notes.append(build_note(
            i, start_s=n["start_s"], end_s=n["end_s"], pitch=n["pitch"],
            velocity=n["velocity"], technique=n["technique"], vibrato=n["vibrato"],
            auto=n["auto"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "clip_id": clip_id,
        "rev": 0,
        "next_note_ordinal": len(out_notes),
        "sr": audio_meta.get("sr"),
        "hop": audio_meta.get("hop"),
        "duration_s": round(float(duration_s), 6),
        "offset_applied_s": round(float(offset_applied_s), 6),
        "audio": audio_meta,
        "pipeline": pipeline_meta,
        "vocabulary": cfg.vocabulary_snapshot(),
        "notes": out_notes,
        "mute_regions": [],
        "orphaned_labels": [],
        "view": {"px_per_sec": 200.0, "scroll_s": 0.0, "gt_variant": "cleaned"},
    }


# --- load / backup / save ---------------------------------------------------

def load(cfg: LabelerConfig, clip_id: str) -> dict | None:
    return read_json(clip_file(cfg, clip_id, NOTES_JSON))


def _backup_dir(cfg: LabelerConfig, clip_id: str) -> str:
    return os.path.join(clip_dir(cfg, clip_id), BACKUP_DIR)


def _roll_backup(cfg: LabelerConfig, clip_id: str, current: dict | None,
                 snapshot_label: str | None = None) -> None:
    """Copy the current notes.json into notes.backup/ before it is overwritten.

    ``snapshot_label`` names a durable snapshot (exempt from rolling pruning);
    otherwise a ``rev_*`` backup is written and old ``rev_*`` files pruned to 20.
    """
    if current is None:
        return
    bdir = _backup_dir(cfg, clip_id)
    os.makedirs(bdir, exist_ok=True)
    rev = current.get("rev", 0)
    ts = time.strftime("%Y%m%d-%H%M%S")
    if snapshot_label:
        name = f"snapshot_{snapshot_label}_{ts}_rev{rev}.json"
    else:
        name = f"rev_{rev:05d}_{ts}.json"
    atomic_write_json(os.path.join(bdir, name), current)
    # Prune rolling rev_* backups (keep the newest _KEEP_BACKUPS).
    revs = sorted(f for f in os.listdir(bdir) if f.startswith("rev_"))
    for stale in revs[:-_KEEP_BACKUPS]:
        try:
            os.remove(os.path.join(bdir, stale))
        except OSError:
            pass


def _write(cfg: LabelerConfig, clip_id: str, doc: dict) -> dict:
    atomic_write_json(clip_file(cfg, clip_id, NOTES_JSON), doc)
    return doc


def commit_rebuild(cfg: LabelerConfig, clip_id: str, new_doc: dict,
                   old_doc: dict | None, snapshot_label: str | None = None) -> dict:
    """Persist a freshly (re)built document, bumping rev over any existing one."""
    with lock_for(clip_id):
        current = load(cfg, clip_id)
        _roll_backup(cfg, clip_id, current, snapshot_label=snapshot_label)
        new_doc = dict(new_doc)
        new_doc["rev"] = (current.get("rev", 0) if current else 0) + 1
        return _write(cfg, clip_id, new_doc)


def save_new(cfg: LabelerConfig, clip_id: str, doc: dict) -> dict:
    """Write a brand-new document (rev -> 1) when none exists yet."""
    with lock_for(clip_id):
        doc = dict(doc)
        doc["rev"] = 1
        return _write(cfg, clip_id, doc)


def put_with_rev(cfg: LabelerConfig, clip_id: str, incoming: dict) -> dict:
    """Apply an editor PUT with optimistic-concurrency rev check.

    Returns ``{"status": "ok", "rev": N, "doc": ...}`` on success, or
    ``{"status": "conflict", "server_rev": M}`` if the base rev is stale.
    """
    with lock_for(clip_id):
        current = load(cfg, clip_id)
        server_rev = current.get("rev", 0) if current else 0
        base_rev = int(incoming.get("rev", 0))
        if current is not None and base_rev != server_rev:
            return {"status": "conflict", "server_rev": server_rev}
        _roll_backup(cfg, clip_id, current)
        doc = dict(incoming)
        doc["rev"] = server_rev + 1
        doc["schema_version"] = SCHEMA_VERSION
        doc["clip_id"] = clip_id
        _write(cfg, clip_id, doc)
        return {"status": "ok", "rev": doc["rev"], "doc": doc}


# --- retranscribe merge -----------------------------------------------------

def _human_touched(note: dict) -> bool:
    return any(bool(v) for v in note.get("human", {}).values())


def merge_retranscribe(old_doc: dict, new_doc: dict,
                       tol_s: float = _MATCH_TOL_S) -> dict:
    """Carry human-flagged facets from ``old_doc`` onto a fresh ``new_doc``.

    Matching: for each human-touched old note, find the nearest still-unmatched new
    note of equal pitch within ``tol_s`` onset. Copy exactly the facets whose old
    ``human`` flag is set (technique/vibrato/velocity value, or timing = start/end),
    and set the same flags on the new note. Human-touched old notes with no match
    are appended to ``new_doc['orphaned_labels']``.
    """
    new_notes = new_doc["notes"]
    used = [False] * len(new_notes)
    orphans = list(new_doc.get("orphaned_labels", []))

    old_human = [n for n in old_doc.get("notes", []) if _human_touched(n)]
    old_human.sort(key=lambda n: float(n["start_s"]))

    for old in old_human:
        pitch = int(old["pitch"])
        onset = float(old["start_s"])
        best, best_d = -1, tol_s + 1.0
        for j, nn in enumerate(new_notes):
            if used[j] or int(nn["pitch"]) != pitch:
                continue
            d = abs(float(nn["start_s"]) - onset)
            if d <= tol_s and d < best_d:
                best, best_d = j, d
        if best < 0:
            orphans.append({
                "pitch": pitch, "start_s": onset, "end_s": float(old["end_s"]),
                "technique": old.get("technique"), "vibrato": old.get("vibrato"),
                "velocity": old.get("velocity"), "human": dict(old.get("human", {})),
                "reason": "no_match",
            })
            continue
        used[best] = True
        nn = new_notes[best]
        h_old = old.get("human", {})
        if h_old.get("technique"):
            nn["technique"] = old["technique"]
            nn["human"]["technique"] = True
        if h_old.get("vibrato"):
            nn["vibrato"] = bool(old["vibrato"])
            nn["human"]["vibrato"] = True
        if h_old.get("velocity"):
            nn["velocity"] = int(old["velocity"])
            nn["human"]["velocity"] = True
        if h_old.get("timing"):
            nn["start_s"] = round(float(old["start_s"]), 6)
            nn["end_s"] = round(float(old["end_s"]), 6)
            nn["human"]["timing"] = True

    new_doc["orphaned_labels"] = orphans
    # Carry mute regions across unchanged (they are time spans, not notes).
    if old_doc.get("mute_regions"):
        new_doc["mute_regions"] = old_doc["mute_regions"]
    return new_doc


def human_edit_count(doc: dict | None) -> int:
    if not doc:
        return 0
    return sum(1 for n in doc.get("notes", []) if _human_touched(n))
