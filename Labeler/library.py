"""Clip library: filesystem layout, clip-id sanitizing, scanning, atomic JSON.

The lowest-level Labeler module — everything else imports its path helpers and
the ``atomic_write_json`` / ``read_json`` pair (same-dir ``.tmp`` + ``os.replace``,
so a crash never leaves a half-written status/notes file). Owns the per-clip
directory convention and the canonical output filenames so every stage agrees on
where things live.

Layout under ``<dataset_root>``::

    raw/<name>.wav                  user drops recordings here
    clips/<clip_id>/                derived per clip (see the *_NAME constants)
    exports/                        export.py writes gt/ midi/ notes/ *_rec/ here
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass

import soundfile as sf

from .config import LabelerConfig, STAGES

# Canonical per-clip filenames (single source of truth across stages).
SOURCE_WAV = "source.wav"
CLEANED_WAV = "cleaned.wav"
TRANSCRIPTION_JSON = "transcription.json"
MUSC_RAW_MID = "musc_raw.mid"
AUTO_JSON = "_auto.json"          # internal per-note analysis accumulator
ONSET_ENV = "onset_env.f32"
NOTES_JSON = "notes.json"
STATUS_JSON = "status.json"
MEL_DIR = "mel"
MEL_META = "meta.json"
STRETCH_DIR = "stretch"
BACKUP_DIR = "notes.backup"

_ID_RE = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_ID = 80


class ClipNotFound(Exception):
    """Raised when a clip id has neither a processed dir nor a raw source wav."""


# --- path helpers -----------------------------------------------------------

def dataset_root(cfg: LabelerConfig) -> str:
    """Absolute dataset root (``cfg.dataset_root`` resolved against the cwd/project root)."""
    return os.path.abspath(cfg.dataset_root)


def raw_dir(cfg: LabelerConfig) -> str:
    return os.path.join(dataset_root(cfg), "raw")


def clips_root(cfg: LabelerConfig) -> str:
    return os.path.join(dataset_root(cfg), "clips")


def exports_dir(cfg: LabelerConfig) -> str:
    return os.path.join(dataset_root(cfg), "exports")


def clip_dir(cfg: LabelerConfig, clip_id: str) -> str:
    return os.path.join(clips_root(cfg), clip_id)


def clip_file(cfg: LabelerConfig, clip_id: str, name: str) -> str:
    return os.path.join(clip_dir(cfg, clip_id), name)


def sanitize_clip_id(stem: str) -> str:
    """Map a filename stem to a safe clip id: ``[A-Za-z0-9_-]``, <= 80 chars."""
    cid = _ID_RE.sub("_", stem).strip("_")
    if not cid:
        cid = "clip"
    return cid[:_MAX_ID]


def clip_id_for_path(path: str) -> str:
    return sanitize_clip_id(os.path.splitext(os.path.basename(path))[0])


# --- atomic JSON ------------------------------------------------------------

def read_json(path: str, default=None):
    """Read a JSON file; return ``default`` if it does not exist."""
    if not os.path.isfile(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str, obj) -> str:
    """Write ``obj`` as JSON to ``path`` atomically (same-dir tmp + os.replace)."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


# --- raw / clip discovery ---------------------------------------------------

_AUDIO_EXTS = (".wav", ".flac")


def scan_raw(cfg: LabelerConfig) -> dict[str, str]:
    """Map ``clip_id -> raw wav path`` for every audio file in ``raw/``.

    On a collision (two raw files sanitize to the same id) the first by sorted
    name wins; the others are ignored (documented limitation).
    """
    out: dict[str, str] = {}
    rd = raw_dir(cfg)
    if not os.path.isdir(rd):
        return out
    for name in sorted(os.listdir(rd)):
        if os.path.splitext(name)[1].lower() in _AUDIO_EXTS:
            cid = sanitize_clip_id(os.path.splitext(name)[0])
            out.setdefault(cid, os.path.join(rd, name))
    return out


def find_raw_wav(cfg: LabelerConfig, clip_id: str) -> str | None:
    """The raw source wav for ``clip_id`` (from status.json, else a raw/ scan)."""
    status = read_json(clip_file(cfg, clip_id, STATUS_JSON), {})
    rp = status.get("raw_path") if isinstance(status, dict) else None
    if rp and os.path.isfile(rp):
        return rp
    return scan_raw(cfg).get(clip_id)


def _audio_duration(path: str) -> float | None:
    try:
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


@dataclass
class ClipInfo:
    id: str
    name: str
    duration_s: float | None
    state: str            # raw | processing | ready | error
    has_notes: bool
    human_edits: int


def _human_edit_count(notes_obj) -> int:
    if not isinstance(notes_obj, dict):
        return 0
    n = 0
    for note in notes_obj.get("notes", []):
        human = note.get("human", {})
        if any(bool(v) for v in human.values()):
            n += 1
    return n


def clip_state(cfg: LabelerConfig, clip_id: str) -> str:
    """Static (disk-derived) state; the API overlays live JobManager state."""
    cdir = clip_dir(cfg, clip_id)
    status = read_json(os.path.join(cdir, STATUS_JSON), {}) or {}
    st = status.get("state")
    if st == "error":
        return "error"
    if st == "processing":
        return "processing"
    if os.path.isfile(os.path.join(cdir, NOTES_JSON)):
        return "ready"
    if os.path.isdir(cdir):
        # A clip dir exists but no notes yet -> mid/failed processing.
        return "processing" if st == "processing" else "raw"
    return "raw"


def get_clip_info(cfg: LabelerConfig, clip_id: str, raw_path: str | None = None) -> ClipInfo:
    cdir = clip_dir(cfg, clip_id)
    src = os.path.join(cdir, SOURCE_WAV)
    dur = _audio_duration(src) if os.path.isfile(src) else (
        _audio_duration(raw_path) if raw_path else None)
    notes_obj = read_json(os.path.join(cdir, NOTES_JSON))
    return ClipInfo(
        id=clip_id,
        name=clip_id,
        duration_s=dur,
        state=clip_state(cfg, clip_id),
        has_notes=notes_obj is not None,
        human_edits=_human_edit_count(notes_obj),
    )


def scan_clips(cfg: LabelerConfig) -> list[ClipInfo]:
    """All clips: processed dirs under ``clips/`` plus un-processed raw wavs."""
    raws = scan_raw(cfg)
    ids: set[str] = set(raws)
    croot = clips_root(cfg)
    if os.path.isdir(croot):
        for name in os.listdir(croot):
            if os.path.isdir(os.path.join(croot, name)):
                ids.add(name)
    return [get_clip_info(cfg, cid, raws.get(cid)) for cid in sorted(ids)]


def require_clip(cfg: LabelerConfig, clip_id: str) -> str:
    """Return the clip dir, raising ``ClipNotFound`` if the clip is unknown."""
    if os.path.isdir(clip_dir(cfg, clip_id)) or find_raw_wav(cfg, clip_id):
        return clip_dir(cfg, clip_id)
    raise ClipNotFound(clip_id)


# --- status.json helpers ----------------------------------------------------

def default_status(clip_id: str, raw_path: str | None) -> dict:
    return {
        "clip_id": clip_id,
        "raw_path": raw_path,
        "state": "pending",           # pending | processing | done | error
        "stage": None,
        "stage_index": 0,
        "n_stages": len(STAGES),
        "pct": -1,
        "message": "",
        "error": None,
        "stages": {},                 # stage -> {state, hash}
    }


def reconcile_interrupted(cfg: LabelerConfig) -> list[str]:
    """Reset clips wedged at ``state="processing"`` on disk back to an error.

    A hard crash (or a kill) during a stage — most likely the GPU transcribe — can
    take the process down without hitting the per-stage ``except`` that records an
    error, leaving ``status.json`` frozen at ``state="processing"``. The UI then
    shows a permanent spinner with no way to retry. Called once at server startup
    (before any job can run), this rewrites each such status to a re-runnable
    ``error`` while **preserving the per-stage skip cache** (the ``stages`` map), so
    the next process re-runs only the interrupted stage onward. Returns the ids reset.
    """
    reset: list[str] = []
    croot = clips_root(cfg)
    if not os.path.isdir(croot):
        return reset
    for cid in sorted(os.listdir(croot)):
        sp = os.path.join(croot, cid, STATUS_JSON)
        status = read_json(sp)
        if not isinstance(status, dict) or status.get("state") != "processing":
            continue
        status["state"] = "error"
        status["message"] = "interrupted"
        status["error"] = {
            "code": "interrupted",
            "detail": "processing was interrupted (crash or restart); press process to retry",
            "stage": status.get("stage"),
        }
        atomic_write_json(sp, status)
        reset.append(cid)
    return reset
