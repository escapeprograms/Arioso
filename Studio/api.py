"""FastAPI routes (thin) — JSON over the project library, model scan, and jobs.

Handlers stay thin: they validate, delegate to ``library`` / ``project_store`` /
``jobs``, and shape the JSON. Errors are always
``JSONResponse({"error": code, "detail": ...})`` with a documented status:
404 (unknown project), 409 ``rev_conflict`` (stale PUT, carries ``server_rev``)
or ``job_running``, 400 (bad request), 501 ``not_implemented`` (render, wired in
a later phase).

``cfg`` and the shared :class:`Studio.jobs.JobManager` live on
``request.app.state`` (set by :mod:`Studio.server`). Media (rendered wav Range
support) is served by a StaticFiles mount in the server, not here. Torch-free.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from common.config import HOP_SIZE as HOP, SR

from . import project_store
from .config import FRAME_RATE, PRIOR_MODES
from .export import RenderStale, export_midi, export_wav
from .jobs import idle_status
from .library import (models_root, project_dir, project_exists, read_json,
                      render_meta_file, scan_models, scan_project_ids,
                      unique_project_id)
from .midi_import import BadMidi, import_midi

router = APIRouter()


def _err(status: int, code: str, detail: str = "", **extra) -> JSONResponse:
    return JSONResponse({"error": code, "detail": detail, **extra}, status_code=status)


def _cfg(request: Request):
    return request.app.state.cfg


def _jobs(request: Request):
    return request.app.state.jobs


# --- config -----------------------------------------------------------------

@router.get("/api/config")
def get_config(request: Request):
    cfg = _cfg(request)
    return {
        "sr": SR,
        "hop": HOP,
        "frame_rate": FRAME_RATE,
        "articulations": cfg.vocabulary_snapshot(),
        "keys": {a.key: a.name for a in cfg.articulations},
        "technique_model_vocab": cfg.technique_model_vocab(),
        "snap_options": cfg.snap_options(),
        "prior_modes": list(PRIOR_MODES),
        "default_model": cfg.default_model,
        "default_checkpoint": cfg.default_checkpoint,
        "default_prior_mode": cfg.default_prior_mode,
    }


# --- models -----------------------------------------------------------------

@router.get("/api/models")
def get_models(request: Request):
    cfg = _cfg(request)
    return {
        "models": scan_models(cfg),
        "default_model": cfg.default_model,
        "default_checkpoint": cfg.default_checkpoint,
    }


# --- projects list / create -------------------------------------------------

@router.get("/api/projects")
def list_projects(request: Request):
    cfg = _cfg(request)
    out = []
    for pid in scan_project_ids(cfg):
        doc = project_store.load(cfg, pid)
        if doc is not None:
            out.append(project_store.summary(doc))
    return out


@router.post("/api/projects")
async def create_project(request: Request):
    cfg = _cfg(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    body = body or {}
    name = body.get("name")
    if not isinstance(name, str) or not name.strip():
        return _err(400, "bad_request", "a non-empty 'name' string is required")
    pid = unique_project_id(cfg, name.strip())
    doc = project_store.create_project(cfg, pid, name=name.strip())
    return JSONResponse(doc, status_code=201)


# --- single project get / put -----------------------------------------------

@router.get("/api/projects/{project_id}")
def get_project(project_id: str, request: Request):
    cfg = _cfg(request)
    doc = project_store.load(cfg, project_id)
    if doc is None:
        return _err(404, "project_not_found", project_id)
    return doc


async def _put_project(project_id: str, request: Request):
    cfg = _cfg(request)
    if not project_exists(cfg, project_id):
        return _err(404, "project_not_found", project_id)
    try:
        incoming = await request.json()
    except Exception:
        return _err(400, "bad_request", "body must be a JSON project document")
    if not isinstance(incoming, dict) or "notes" not in incoming:
        return _err(400, "bad_request", "project document must be an object with a 'notes' list")
    result = project_store.put_with_rev(cfg, project_id, incoming)
    if result["status"] == "conflict":
        return _err(409, "rev_conflict", "project was modified by another writer",
                    server_rev=result["server_rev"])
    return {"status": "ok", "rev": result["rev"]}


@router.put("/api/projects/{project_id}")
async def put_project(project_id: str, request: Request):
    return await _put_project(project_id, request)


@router.post("/api/projects/{project_id}")
async def post_project(project_id: str, request: Request):
    # sendBeacon uses POST; alias to the PUT handler.
    return await _put_project(project_id, request)


# --- render ------------------------------------------------------------------

@router.post("/api/projects/{project_id}/render")
async def render_project(project_id: str, request: Request):
    cfg = _cfg(request)
    jobs = _jobs(request)
    if not project_exists(cfg, project_id):
        return _err(404, "project_not_found", project_id)
    try:
        body = await request.json()
    except Exception:
        body = {}
    body = body or {}

    scope = body.get("scope", "phrase")
    if scope not in ("phrase", "selection"):
        return _err(400, "bad_request", "scope must be 'phrase' or 'selection'")
    note_ids = body.get("note_ids")
    if scope == "selection" and not note_ids:
        return _err(400, "bad_request", "selection scope requires a non-empty note_ids")

    model = body.get("model") or cfg.default_model
    checkpoint = body.get("checkpoint") or cfg.default_checkpoint
    prior_mode = body.get("prior_mode") or cfg.default_prior_mode
    if prior_mode not in PRIOR_MODES:
        return _err(400, "bad_request", f"prior_mode must be one of {list(PRIOR_MODES)}")

    ckpt_path = os.path.join(models_root(cfg), model, checkpoint)
    if not os.path.isfile(ckpt_path):
        return _err(400, "model_not_found",
                    f"checkpoint {model}/{checkpoint} does not exist")

    doc = project_store.load(cfg, project_id)
    if doc is None:
        return _err(404, "project_not_found", project_id)
    pdir = project_dir(cfg, project_id)

    # Imported here (not at module top) so the api import graph stays torch-free.
    from .render import run_render

    def work(progress):
        run_render(cfg, doc, pdir, scope=scope, note_ids=note_ids, model=model,
                   checkpoint=checkpoint, prior_mode=prior_mode, progress=progress)

    initial = {"state": "processing", "stage": "serialize", "pct": 0,
               "message": "starting", "error": None}
    if not jobs.submit(project_id, work, initial=initial):
        return _err(409, "job_running", "a render is already in progress for this project")
    return JSONResponse(
        {"status": "accepted", "project_id": project_id, "scope": scope,
         "model": model, "checkpoint": checkpoint, "prior_mode": prior_mode},
        status_code=202)


@router.post("/api/projects/{project_id}/render-prior")
def render_prior_endpoint(project_id: str, request: Request):
    """Render the whole document's raw saw prior (torch-free, no GPU) for playback.

    A synchronous CPU render (FastAPI runs sync handlers in a threadpool, so the
    numpy prior synthesis never blocks the event loop) -> ``render/prior_mix.wav`` +
    ``.peaks``, returned as ``/media`` URLs for the frontend's PRIOR source. Optional
    ``prior_mode`` query param (``bend`` | ``quantized``) defaults to the config
    default. 404 unknown project; 400 on a bad ``prior_mode`` or a zero-length note.
    """
    cfg = _cfg(request)
    if not project_exists(cfg, project_id):
        return _err(404, "project_not_found", project_id)

    prior_mode = request.query_params.get("prior_mode") or cfg.default_prior_mode
    if prior_mode not in PRIOR_MODES:
        return _err(400, "bad_request", f"prior_mode must be one of {list(PRIOR_MODES)}")

    doc = project_store.load(cfg, project_id)
    if doc is None:
        return _err(404, "project_not_found", project_id)
    pdir = project_dir(cfg, project_id)

    # Imported here (not at module top) so the api import graph stays torch-free.
    from .render import render_prior_mix

    try:
        return render_prior_mix(cfg, doc, pdir, prior_mode=prior_mode)
    except ValueError as exc:                       # non-positive length note
        return _err(400, "bad_request", str(exc))


@router.get("/api/projects/{project_id}/render/status")
def get_render_status(project_id: str, request: Request):
    cfg = _cfg(request)
    jobs = _jobs(request)
    if not project_exists(cfg, project_id):
        return _err(404, "project_not_found", project_id)
    st = jobs.status(project_id)
    return st if st is not None else idle_status()


@router.get("/api/projects/{project_id}/render/meta")
def get_render_meta(project_id: str, request: Request):
    cfg = _cfg(request)
    if not project_exists(cfg, project_id):
        return _err(404, "project_not_found", project_id)
    meta = read_json(render_meta_file(cfg, project_id))
    if meta is None:
        return _err(404, "not_rendered", "no render exists for this project yet")
    return meta


# --- MIDI import -------------------------------------------------------------

def _truthy(value: str | None) -> bool:
    return str(value).lower() in ("1", "true", "yes", "on") if value is not None else False


@router.post("/api/projects/{project_id}/import-midi")
async def import_midi_endpoint(project_id: str, request: Request):
    """Import a ``.mid`` into the project (``mode=replace`` | ``merge``).

    The MIDI is uploaded as the **raw request body** (``application/octet-stream``);
    python-multipart is not installed, so there is no multipart form. Query params:
    ``mode`` (replace | merge, default replace), ``snap`` (a snap id, optional), and
    ``adopt_bpm`` (set the project BPM to the file's initial tempo). Notes are timed
    in beats through the file's tempo map, stamped with fresh monotonic ids, and the
    document is committed with a rev bump. Returns the updated doc plus a summary
    (``notes_added``, ``file_bpm``, ``warnings``). 400 ``bad_midi`` on parse failure.
    """
    cfg = _cfg(request)
    if not project_exists(cfg, project_id):
        return _err(404, "project_not_found", project_id)

    doc = project_store.load(cfg, project_id)
    if doc is None:
        return _err(404, "project_not_found", project_id)

    q = request.query_params
    mode = q.get("mode", "replace")
    if mode not in ("replace", "merge"):
        return _err(400, "bad_request", "mode must be 'replace' or 'merge'")
    snap = q.get("snap") or None
    adopt_bpm = _truthy(q.get("adopt_bpm"))

    body = await request.body()
    if not body:
        return _err(400, "bad_midi", "empty request body (upload the .mid as raw bytes)")

    try:
        result = import_midi(body, snap=snap, adopt_bpm=adopt_bpm,
                             time_sig=doc.get("time_sig"))
    except BadMidi as exc:
        return _err(400, "bad_midi", str(exc))
    except ValueError as exc:                       # e.g. unknown snap id
        return _err(400, "bad_request", str(exc))

    imported = result["notes"]
    if mode == "replace":
        base_notes: list[dict] = []
        start_ord = 0
    else:
        base_notes = list(doc.get("notes", []))
        start_ord = int(doc.get("next_note_ordinal", len(base_notes)))

    new_notes = [project_store.build_note(
        start_ord + i,
        start_beat=n["start_beat"], len_beats=n["len_beats"], pitch=n["pitch"],
        velocity=n.get("velocity", 100), technique=n.get("technique", "normal"),
        pan=n.get("pan", 0.0), bend=n.get("bend"), vibrato=n.get("vibrato"),
        env_dct=n.get("env_dct"),
    ) for i, n in enumerate(imported)]

    new_doc = dict(doc)
    new_doc["notes"] = base_notes + new_notes
    new_doc["next_note_ordinal"] = start_ord + len(new_notes)
    if adopt_bpm and result["file_bpm"]:
        new_doc["bpm"] = float(result["file_bpm"])

    saved = project_store.commit_rebuild(cfg, project_id, new_doc, doc)
    return {
        "rev": saved["rev"],
        "notes_added": len(new_notes),
        "file_bpm": result["file_bpm"],
        "warnings": result["warnings"],
        "doc": saved,
    }


# --- WAV / MIDI export -------------------------------------------------------

@router.post("/api/projects/{project_id}/export")
async def export_endpoint(project_id: str, request: Request):
    """Export the project as ``wav`` or ``midi`` under ``exports/`` (served by /media).

    Body ``{"kind": "wav" | "midi"}``. MIDI export always succeeds (200 with the media
    path). WAV export copies the *current* render's ``mix.wav`` — if no fresh render
    matches the document it returns 409 ``render_stale`` (the client should render,
    then retry). 400 ``bad_request`` on an unknown kind.
    """
    cfg = _cfg(request)
    if not project_exists(cfg, project_id):
        return _err(404, "project_not_found", project_id)
    try:
        body = await request.json()
    except Exception:
        body = {}
    body = body or {}
    kind = body.get("kind")
    if kind not in ("wav", "midi"):
        return _err(400, "bad_request", "kind must be 'wav' or 'midi'")

    doc = project_store.load(cfg, project_id)
    if doc is None:
        return _err(404, "project_not_found", project_id)
    pdir = project_dir(cfg, project_id)

    if kind == "midi":
        try:
            out = export_midi(cfg, doc, pdir)
        except ValueError as exc:                   # non-positive length note
            return _err(400, "bad_request", str(exc))
        return out

    try:
        return export_wav(cfg, doc, pdir)
    except RenderStale as exc:
        return _err(409, "render_stale", str(exc))
