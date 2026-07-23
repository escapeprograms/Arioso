"""Project library: filesystem layout, id sanitizing, model scan, atomic JSON.

The lowest-level Studio module — everything else imports its path helpers and
the ``atomic_write_json`` / ``read_json`` pair (same-dir ``.tmp`` + ``os.replace``,
so a crash never leaves a half-written project/status file). Owns the per-project
directory convention and the canonical filenames so every module agrees on where
things live. Torch-free.

Layout under ``<projects_root>`` (the dir ``/media`` is served over)::

    <project_id>/project.json       canonical project document (notes + settings)
    <project_id>/render/mix.wav     stitched render output (+ seg_*.wav, status.json)
    <project_id>/render/status.json server-owned render job state
    <project_id>/exports/           WAV / MIDI exports
    <project_id>/project.backup/    rolling rev backups
"""

from __future__ import annotations

import json
import os
import re
import tempfile

from .config import StudioConfig

# Canonical per-project filenames / subdirs (single source of truth).
PROJECT_JSON = "project.json"
RENDER_DIR = "render"
SEG_CACHE_DIR = "cache"            # render/cache/ holds seg_<hash>.wav segment renders
MIX_WAV = "mix.wav"
MIX_PEAKS = "mix.peaks"
PRIOR_MIX_WAV = "prior_mix.wav"    # torch-free whole-doc saw-prior audition render
PRIOR_MIX_PEAKS = "prior_mix.peaks"
RENDER_MIDI = "render.mid"
RENDER_STATUS = "status.json"
RENDER_META = "meta.json"          # render bookkeeping (kept OUT of project.json)
EXPORTS_DIR = "exports"
BACKUP_DIR = "project.backup"

_ID_RE = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_ID = 80


class ProjectNotFound(Exception):
    """Raised when a project id has no directory / project.json."""


# --- path helpers -----------------------------------------------------------

def projects_root(cfg: StudioConfig) -> str:
    """Absolute projects root (``cfg.projects_root`` resolved against the cwd)."""
    return os.path.abspath(cfg.projects_root)


def models_root(cfg: StudioConfig) -> str:
    return os.path.abspath(cfg.models_root)


def project_dir(cfg: StudioConfig, project_id: str) -> str:
    return os.path.join(projects_root(cfg), project_id)


def project_file(cfg: StudioConfig, project_id: str) -> str:
    return os.path.join(project_dir(cfg, project_id), PROJECT_JSON)


def render_dir(cfg: StudioConfig, project_id: str) -> str:
    return os.path.join(project_dir(cfg, project_id), RENDER_DIR)


def render_cache_dir(cfg: StudioConfig, project_id: str) -> str:
    """``render/cache/`` — the segment-render cache (``seg_<hash>.wav``)."""
    return os.path.join(render_dir(cfg, project_id), SEG_CACHE_DIR)


def render_status_file(cfg: StudioConfig, project_id: str) -> str:
    return os.path.join(render_dir(cfg, project_id), RENDER_STATUS)


def render_meta_file(cfg: StudioConfig, project_id: str) -> str:
    return os.path.join(render_dir(cfg, project_id), RENDER_META)


def exports_dir(cfg: StudioConfig, project_id: str) -> str:
    return os.path.join(project_dir(cfg, project_id), EXPORTS_DIR)


def backup_dir(cfg: StudioConfig, project_id: str) -> str:
    return os.path.join(project_dir(cfg, project_id), BACKUP_DIR)


def sanitize_project_id(name: str) -> str:
    """Map a name to a safe project id: ``[A-Za-z0-9_-]``, <= 80 chars."""
    pid = _ID_RE.sub("_", name).strip("_")
    if not pid:
        pid = "project"
    return pid[:_MAX_ID]


def unique_project_id(cfg: StudioConfig, name: str) -> str:
    """A project id derived from ``name`` that does not collide on disk.

    Appends ``-2``, ``-3``, ... (within the 80-char cap) if the base id already
    has a directory, so two projects named the same never clobber each other.
    """
    base = sanitize_project_id(name)
    if not os.path.exists(project_dir(cfg, base)):
        return base
    for n in range(2, 10000):
        suffix = f"-{n}"
        cand = base[: _MAX_ID - len(suffix)] + suffix
        if not os.path.exists(project_dir(cfg, cand)):
            return cand
    raise RuntimeError(f"could not allocate a unique project id for {name!r}")


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


# --- project discovery ------------------------------------------------------

def scan_project_ids(cfg: StudioConfig) -> list[str]:
    """Sorted ids of every directory under the projects root holding a project.json."""
    root = projects_root(cfg)
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        if os.path.isfile(os.path.join(root, name, PROJECT_JSON)):
            out.append(name)
    return out


def project_exists(cfg: StudioConfig, project_id: str) -> bool:
    return os.path.isfile(project_file(cfg, project_id))


def require_project(cfg: StudioConfig, project_id: str) -> str:
    """Return the project dir, raising ``ProjectNotFound`` if it is unknown."""
    if project_exists(cfg, project_id):
        return project_dir(cfg, project_id)
    raise ProjectNotFound(project_id)


# --- model discovery (pure filesystem; the torch model cache is a later phase) --

def scan_models(cfg: StudioConfig) -> list[dict]:
    """Scan ``models_root`` for ``<run>/*.pt`` -> ``[{run, checkpoints, default}]``.

    Runs are sorted by name; each run's checkpoints are sorted filenames. The run
    equal to ``cfg.default_model`` is flagged ``default``. No torch — this only
    lists files, so ``/api/models`` stays fast and importable without a GPU.
    """
    root = models_root(cfg)
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        ckpts = sorted(f for f in os.listdir(d) if f.endswith(".pt"))
        if not ckpts:
            continue
        out.append({
            "run": name,
            "checkpoints": ckpts,
            "default": name == cfg.default_model,
        })
    return out
