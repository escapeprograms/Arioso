"""FastAPI app factory + uvicorn entrypoint (``python -m Labeler.server``).

Wires the config + a shared :class:`Labeler.processing.JobManager` onto
``app.state`` and mounts three surfaces:

  * ``/api/*`` — the JSON API (:mod:`Labeler.api`).
  * ``/media/<clip_id>/...`` — a StaticFiles mount over ``clips/`` (wav Range /
    206 support comes for free), used for audio + mel tiles + onset env.
  * ``/static`` + ``/`` — the vanilla-JS frontend under ``Labeler/static/``. The
    dir may not exist yet (a separate agent builds it); the server still boots and
    ``/`` returns a friendly placeholder so the API is testable.

Windows serves ``.js`` as ``text/plain`` by default, which breaks ES-module
imports, so the MIME registry is patched at startup. Binds 127.0.0.1:``cfg.port``
(8765), single process, no reload (the MUSC model is a per-process singleton).
"""

from __future__ import annotations

import mimetypes
import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .api import router
from .config import LabelerConfig, load_config
from .library import clips_root, reconcile_interrupted
from .processing import JobManager

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def _fix_windows_mimes() -> None:
    """Register the MIME types Windows gets wrong (breaks ES modules / CSS)."""
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("application/javascript", ".mjs")
    mimetypes.add_type("text/css", ".css")


def create_app(cfg: LabelerConfig | None = None) -> FastAPI:
    _fix_windows_mimes()
    cfg = cfg or load_config()
    app = FastAPI(title="Labeler", version="1.0")
    app.state.cfg = cfg
    app.state.jobs = JobManager(cfg)
    # Nothing is running yet, so it is safe to clear crash-wedged "processing"
    # statuses (e.g. a transcribe that hard-locked the machine) back to a
    # re-runnable error, rather than leaving a permanent UI spinner.
    interrupted = reconcile_interrupted(cfg)
    if interrupted:
        print(f"[labeler] reset {len(interrupted)} interrupted clip(s) to error: "
              + ", ".join(interrupted))
    app.include_router(router)

    croot = clips_root(cfg)
    os.makedirs(croot, exist_ok=True)
    # Range/206 support is free via StaticFiles' FileResponse.
    app.mount("/media", StaticFiles(directory=croot), name="media")

    if os.path.isdir(_STATIC_DIR):
        app.mount("/static", StaticFiles(directory=_STATIC_DIR, html=True), name="static")

    @app.get("/")
    def index():
        # index.html uses relative asset refs (./js, ./css), which only resolve
        # under the /static mount — serving it at "/" would 404 every module.
        idx = os.path.join(_STATIC_DIR, "index.html")
        if os.path.isfile(idx):
            return RedirectResponse(url="/static/")
        return HTMLResponse(
            "<!doctype html><meta charset=utf-8>"
            "<h1>Labeler backend is running</h1>"
            "<p>The frontend has not been built yet "
            "(<code>Labeler/static/index.html</code> is missing). "
            "The JSON API is live at <a href='/api/config'>/api/config</a>.</p>",
            status_code=200)

    return app


def main() -> None:
    cfg = load_config()
    app = create_app(cfg)
    uvicorn.run(app, host="127.0.0.1", port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
