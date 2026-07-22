"""FastAPI routes (thin) — JSON over the clip library and pipeline.

Handlers stay thin: they validate, delegate to ``library`` / ``processing`` /
``notes_store``, and shape the JSON. Errors are always
``JSONResponse({"error": code, "detail": ...})`` with the documented status:
404 (unknown clip / not-yet-processed), 409 ``job_running`` (mutate while a job
runs) or ``rev_conflict`` (stale PUT, carries ``server_rev``), 400 (bad request).

``cfg`` and the shared :class:`Labeler.processing.JobManager` live on
``request.app.state`` (set by :mod:`Labeler.server`). Media (wav Range support) is
served by a StaticFiles mount in the server, not here.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from common.config import HOP_SIZE as HOP, SR

from . import notes_store
from .config import STAGES
from .library import (MEL_DIR, MEL_META, clip_dir,
                      find_raw_wav, get_clip_info, read_json, scan_clips)
from .media import stretch_name
from .processing import NOTES_MODES

router = APIRouter()


def _err(status: int, code: str, detail: str = "", **extra) -> JSONResponse:
    return JSONResponse({"error": code, "detail": detail, **extra}, status_code=status)


def _cfg(request: Request):
    return request.app.state.cfg


def _jobs(request: Request):
    return request.app.state.jobs


def _clip_exists(cfg, clip_id: str) -> bool:
    import os
    return os.path.isdir(clip_dir(cfg, clip_id)) or bool(find_raw_wav(cfg, clip_id))


# --- config -----------------------------------------------------------------

@router.get("/api/config")
def get_config(request: Request):
    cfg = _cfg(request)
    return {
        "editing_enabled": cfg.editing_enabled,
        "sr": SR,
        "hop": HOP,
        "speeds": list(cfg.speeds),
        "articulations": cfg.vocabulary_snapshot(),
        "keys": {a.key: a.name for a in cfg.articulations},
        "stages": list(STAGES),
        "notes_modes": list(NOTES_MODES),
    }


# --- clips list -------------------------------------------------------------

@router.get("/api/clips")
def list_clips(request: Request):
    cfg = _cfg(request)
    jobs = _jobs(request)
    out = []
    for info in scan_clips(cfg):
        state = "processing" if jobs.is_running(info.id) else info.state
        out.append({"id": info.id, "name": info.name, "duration_s": info.duration_s,
                    "state": state, "has_notes": info.has_notes,
                    "human_edits": info.human_edits, "verified": info.verified})
    return out


# --- process ----------------------------------------------------------------

@router.post("/api/clips/{clip_id}/process")
async def process_clip(clip_id: str, request: Request):
    cfg = _cfg(request)
    jobs = _jobs(request)
    if not _clip_exists(cfg, clip_id):
        return _err(404, "clip_not_found", clip_id)
    try:
        body = await request.json()
    except Exception:
        body = {}
    body = body or {}
    force_from = body.get("force_from")
    notes_mode = body.get("notes_mode", "keep_labels")
    if force_from is not None and force_from not in STAGES:
        return _err(400, "bad_request", f"force_from must be one of {STAGES}")
    if notes_mode not in NOTES_MODES:
        return _err(400, "bad_request", f"notes_mode must be one of {NOTES_MODES}")
    if jobs.is_running(clip_id):
        return _err(409, "job_running", f"a job is already running for {clip_id}")
    raw_path = find_raw_wav(cfg, clip_id)
    ok = jobs.submit(clip_id, raw_path=raw_path, force_from=force_from, notes_mode=notes_mode)
    if not ok:
        return _err(409, "job_running", f"a job is already running for {clip_id}")
    return JSONResponse({"status": "accepted", "clip_id": clip_id}, status_code=202)


# --- status -----------------------------------------------------------------

@router.get("/api/clips/{clip_id}/status")
def get_status(clip_id: str, request: Request):
    cfg = _cfg(request)
    jobs = _jobs(request)
    if not _clip_exists(cfg, clip_id):
        return _err(404, "clip_not_found", clip_id)
    st = jobs.status(clip_id)
    if st is None:
        info = get_clip_info(cfg, clip_id, find_raw_wav(cfg, clip_id))
        return {"state": info.state, "stage": None, "stage_index": 0,
                "n_stages": len(STAGES), "pct": -1, "message": "", "error": None}
    return {
        "state": st.get("state"),
        "stage": st.get("stage"),
        "stage_index": st.get("stage_index", 0),
        "n_stages": st.get("n_stages", len(STAGES)),
        "pct": st.get("pct", -1),
        "message": st.get("message", ""),
        "error": st.get("error"),
    }


# --- meta -------------------------------------------------------------------

@router.get("/api/clips/{clip_id}/meta")
def get_meta(clip_id: str, request: Request):
    import os
    cfg = _cfg(request)
    if not _clip_exists(cfg, clip_id):
        return _err(404, "clip_not_found", clip_id)
    mel_meta = read_json(os.path.join(clip_dir(cfg, clip_id), MEL_DIR, MEL_META))
    if mel_meta is None:
        return _err(404, "not_processed", f"{clip_id} has no media yet (process it first)")
    doc = notes_store.load(cfg, clip_id)
    base = f"/media/{clip_id}"
    media = {
        "source": f"{base}/source.wav",
        "cleaned": f"{base}/cleaned.wav",
        "onset_env": f"{base}/onset_env.f32",
        "mel_tiles": [f"{base}/{MEL_DIR}/{t}" for t in mel_meta.get("tiles", [])],
        "stretch": {},
    }
    for variant in ("source", "cleaned"):
        media["stretch"][variant] = {
            str(s): f"{base}/stretch/{stretch_name(variant, s)}"
            for s in cfg.media.stretch_speeds}
    return {
        "clip_id": clip_id,
        "duration_s": (doc or {}).get("duration_s"),
        "offset_applied_s": (doc or {}).get("offset_applied_s"),
        "sr": SR, "hop": HOP,
        "mel": mel_meta,
        "media": media,
        "has_notes": doc is not None,
    }


# --- notes get/put ----------------------------------------------------------

@router.get("/api/clips/{clip_id}/notes")
def get_notes(clip_id: str, request: Request):
    cfg = _cfg(request)
    if not _clip_exists(cfg, clip_id):
        return _err(404, "clip_not_found", clip_id)
    doc = notes_store.load(cfg, clip_id)
    if doc is None:
        return _err(404, "no_notes", f"{clip_id} has no notes.json yet")
    return doc


async def _put_notes(clip_id: str, request: Request):
    cfg = _cfg(request)
    jobs = _jobs(request)
    if not _clip_exists(cfg, clip_id):
        return _err(404, "clip_not_found", clip_id)
    if jobs.is_running(clip_id):
        return _err(409, "job_running", f"a job is running for {clip_id}; try again after it finishes")
    try:
        incoming = await request.json()
    except Exception:
        return _err(400, "bad_request", "body must be a JSON notes document")
    if not isinstance(incoming, dict) or "notes" not in incoming:
        return _err(400, "bad_request", "notes document must be an object with a 'notes' list")
    result = notes_store.put_with_rev(cfg, clip_id, incoming)
    if result["status"] == "conflict":
        return _err(409, "rev_conflict", "notes were modified by another writer",
                    server_rev=result["server_rev"])
    return {"status": "ok", "rev": result["rev"]}


@router.put("/api/clips/{clip_id}/notes")
async def put_notes(clip_id: str, request: Request):
    return await _put_notes(clip_id, request)


@router.post("/api/clips/{clip_id}/notes")
async def post_notes(clip_id: str, request: Request):
    # sendBeacon uses POST; alias to the PUT handler.
    return await _put_notes(clip_id, request)


# --- verified flag ----------------------------------------------------------

@router.post("/api/clips/{clip_id}/verified")
async def set_verified(clip_id: str, request: Request):
    """Toggle a clip's human sign-off (gates compile). 409 while a job runs on it."""
    cfg = _cfg(request)
    jobs = _jobs(request)
    if not _clip_exists(cfg, clip_id):
        return _err(404, "clip_not_found", clip_id)
    if jobs.is_running(clip_id):
        return _err(409, "job_running", f"a job is running for {clip_id}; try again after it finishes")
    try:
        body = await request.json()
    except Exception:
        return _err(400, "bad_request", 'body must be {"verified": true|false}')
    if not isinstance(body, dict) or not isinstance(body.get("verified"), bool):
        return _err(400, "bad_request", 'body must be {"verified": true|false}')
    doc = notes_store.set_verified(cfg, clip_id, body["verified"])
    if doc is None:
        return _err(404, "no_notes", f"{clip_id} has no notes.json to verify yet")
    return {"status": "ok", "rev": doc["rev"], "verified": doc.get("verified", False)}


# --- compile (gt_arky -> standard dataset root) -----------------------------

def _compiler(request: Request):
    return request.app.state.compiler


@router.post("/api/compile")
def start_compile(request: Request):
    """Start a background compile of all verified clips; 409 if one already runs."""
    if not _compiler(request).start():
        return _err(409, "compile_running", "a compile is already running")
    return JSONResponse({"status": "accepted"}, status_code=202)


@router.get("/api/compile/status")
def compile_status(request: Request):
    """``{running, done, skipped, failed, current, total, root}`` for the last/active run."""
    return _compiler(request).status()
