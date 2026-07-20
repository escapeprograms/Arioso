"""WAV / MIDI export of a project to its ``exports/`` dir. Torch-free.

Two exports, both writing a timestamped file under ``<project>/exports/`` (served by
the ``/media`` StaticFiles mount) and returning its ``/media`` URL:

* **MIDI** (:func:`export_midi`) — :func:`Studio.midi_export.write_midi` with
  ``prior_mode="bend"`` always (an exported ``.mid`` should carry the full pitch-wheel
  expression regardless of the dev quantized toggle). Cheap and always available.

* **WAV** (:func:`export_wav`) — the exported audio must be the *current* render, so
  this never re-renders. It re-plans the render (:func:`Studio.cache.plan_render`,
  torch-free) against the existing ``render/meta.json`` params and checks that every
  planned segment hash matches the render's manifest and its cache WAV still exists
  (and ``mix.wav`` is present). If anything is missing/changed the render is stale and
  :class:`RenderStale` is raised — the API turns that into a 409 telling the client to
  render first (the frontend already knows how to render, so there is no double-render
  logic here). If fresh, ``mix.wav`` is copied to the timestamped export.

Kept torch-free: staleness detection is pure cache/plan bookkeeping and the WAV is a
file copy, so nothing here imports the model or vocoder.
"""

from __future__ import annotations

import os
import shutil
import time

from .cache import plan_render
from .library import (EXPORTS_DIR, MIX_WAV, RENDER_DIR, exports_dir,
                      render_cache_dir, render_meta_file, sanitize_project_id,
                      read_json)
from .midi_export import write_midi


class RenderStale(Exception):
    """Raised by :func:`export_wav` when no fresh render matches the current doc."""


def _project_id(doc: dict, project_dir: str) -> str:
    return doc.get("project_id") or os.path.basename(project_dir.rstrip("/\\"))


def _export_basename(doc: dict) -> str:
    """A filesystem-safe stem for an export: ``<name-or-id>-<YYYYmmdd-HHMMSS>``."""
    stem = sanitize_project_id(str(doc.get("name") or doc.get("project_id") or "project"))
    return f"{stem}-{time.strftime('%Y%m%d-%H%M%S')}"


def _media_url(project_id: str, *parts: str) -> str:
    return "/media/" + "/".join([project_id, *parts])


def export_midi(cfg, doc: dict, project_dir: str) -> dict:
    """Write ``doc`` to ``exports/<stem>.mid`` (bend-aware) and return its media path.

    Returns ``{"kind": "midi", "path": "/media/.../exports/<stem>.mid", "warnings": [...]}``.
    Raises ``ValueError`` (via :func:`write_midi`) if a note has non-positive length.
    """
    project_id = _project_id(doc, project_dir)
    out_dir = exports_dir(cfg, project_id)
    os.makedirs(out_dir, exist_ok=True)
    fname = _export_basename(doc) + ".mid"
    path = os.path.join(out_dir, fname)
    _written, warnings = write_midi(doc, path, note_ids=None, prior_mode="bend")
    return {"kind": "midi",
            "path": _media_url(project_id, EXPORTS_DIR, fname),
            "warnings": warnings}


def _render_is_fresh(cfg, doc: dict, project_id: str,
                     project_dir: str) -> tuple[bool, dict | None]:
    """``(fresh, meta)``: is the on-disk render current for ``doc``?

    Fresh means a ``render/meta.json`` exists, replanning the doc with that render's
    ``model``/``checkpoint``/``prior_mode`` yields exactly the manifest's segment
    hashes (in order), every planned segment's cache WAV is present, and ``mix.wav``
    exists. Any deviation (an edit changes a segment hash; a missing cache/mix file)
    means stale.
    """
    meta = read_json(render_meta_file(cfg, project_id))
    if not meta:
        return False, None
    plan = plan_render(doc, cfg, meta.get("model"), meta.get("checkpoint"),
                       meta.get("prior_mode"),
                       cache_dir=render_cache_dir(cfg, project_id))
    segs = plan["segments"]
    planned = [s["hash"] for s in segs]
    manifest = [s.get("hash") for s in meta.get("segments", [])]
    if planned != manifest:
        return False, meta
    if not all(s["cached"] for s in segs):
        return False, meta
    if not os.path.isfile(os.path.join(project_dir, RENDER_DIR, MIX_WAV)):
        return False, meta
    return True, meta


def export_wav(cfg, doc: dict, project_dir: str) -> dict:
    """Copy the current fresh render ``mix.wav`` to ``exports/<stem>.wav``.

    Returns ``{"kind": "wav", "path": "/media/.../exports/<stem>.wav", "warnings": [...]}``.
    Raises :class:`RenderStale` when there is no fresh render matching ``doc`` (the
    caller should render first, then retry).
    """
    project_id = _project_id(doc, project_dir)
    fresh, meta = _render_is_fresh(cfg, doc, project_id, project_dir)
    if not fresh:
        raise RenderStale("no fresh render for the current document; render first")

    src = os.path.join(project_dir, RENDER_DIR, MIX_WAV)
    if not os.path.isfile(src):
        raise RenderStale("rendered mix.wav is missing; render first")

    out_dir = exports_dir(cfg, project_id)
    os.makedirs(out_dir, exist_ok=True)
    fname = _export_basename(doc) + ".wav"
    dst = os.path.join(out_dir, fname)
    shutil.copyfile(src, dst)
    return {"kind": "wav",
            "path": _media_url(project_id, EXPORTS_DIR, fname),
            "warnings": list(meta.get("warnings") or []) if meta else []}
