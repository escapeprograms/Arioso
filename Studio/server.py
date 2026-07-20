"""FastAPI app factory + uvicorn entrypoint (``python -m Studio.server``).

Wires the config + a shared :class:`Studio.jobs.JobManager` onto ``app.state``
and mounts three surfaces:

  * ``/api/*`` — the JSON API (:mod:`Studio.api`).
  * ``/media/<project_id>/...`` — a StaticFiles mount over the projects root
    (rendered wav Range / 206 support comes for free), used for ``render/mix.wav``
    and future peaks/segment files.
  * ``/static`` + ``/`` — the vanilla-JS frontend under ``Studio/static/``. That
    dir is built by a separate agent and may not exist yet; the server still
    boots and ``/`` returns a friendly placeholder so the API is testable.

Windows serves ``.js`` as ``text/plain`` by default, which breaks ES-module
imports, so the MIME registry is patched at startup. Binds 127.0.0.1:``cfg.port``
(8766), single process, no reload (the model/vocoder are per-process singletons).

Module import is torch-free — nothing here (or under it) imports torch; the model
and vocoder are lazy singletons pulled in only inside the render pipeline.
"""

from __future__ import annotations

import mimetypes
import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .api import router
from .config import StudioConfig, load_config
from .jobs import JobManager
from .library import projects_root

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def _fix_windows_mimes() -> None:
    """Register the MIME types Windows gets wrong (breaks ES modules / CSS)."""
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("application/javascript", ".mjs")
    mimetypes.add_type("text/css", ".css")


def create_app(cfg: StudioConfig | None = None) -> FastAPI:
    _fix_windows_mimes()
    cfg = cfg or load_config()
    app = FastAPI(title="Studio", version="0.1")
    app.state.cfg = cfg
    app.state.jobs = JobManager(cfg)
    app.include_router(router)

    proot = projects_root(cfg)
    os.makedirs(proot, exist_ok=True)
    # Range/206 support is free via StaticFiles' FileResponse.
    app.mount("/media", StaticFiles(directory=proot), name="media")

    # The frontend is owned by a separate agent and may not exist yet; only mount
    # it when the directory is present so the API still boots on its own.
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
            "<h1>Studio backend is running</h1>"
            "<p>The frontend has not been built yet "
            "(<code>Studio/static/index.html</code> is missing). "
            "The JSON API is live at <a href='/api/config'>/api/config</a>.</p>",
            status_code=200)

    return app


def main() -> None:
    cfg = load_config()
    app = create_app(cfg)
    uvicorn.run(app, host="127.0.0.1", port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
