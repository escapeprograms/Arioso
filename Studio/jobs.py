"""JobManager — one background render job per project + live status polling.

Copies the :class:`Labeler.processing.JobManager` pattern but stays generic: a
job is any callable ``work(progress)`` where ``progress`` is a callback the work
invokes with a status dict as it advances. The manager runs it on a daemon
thread behind a per-project guard (at most one live job per project), mirrors the
latest status into an in-memory ``_live`` map, and persists it to the project's
``render/status.json`` so a fresh poll after the process restarts still sees the
last state. The concrete render pipeline is wired in a later phase; Phase 0 ships
the manager so ``/api/.../render/status`` has something to read.

Torch-free: the manager imports nothing heavy; the ``work`` callable owns any
lazy torch import.
"""

from __future__ import annotations

import threading
import traceback
from typing import Callable

from .config import StudioConfig
from .library import atomic_write_json, read_json, render_status_file

# A unit of work: receives a ``progress(status_dict)`` callback to report state.
Work = Callable[[Callable[[dict], None]], None]


def idle_status() -> dict:
    """The status reported for a project that has never run a job."""
    return {"state": "idle", "stage": None, "pct": -1, "message": "", "error": None}


class JobManager:
    """Runs project render jobs on background threads, one job per project."""

    def __init__(self, cfg: StudioConfig):
        self.cfg = cfg
        self._threads: dict[str, threading.Thread] = {}
        self._live: dict[str, dict] = {}
        self._guard = threading.Lock()

    def is_running(self, project_id: str) -> bool:
        with self._guard:
            t = self._threads.get(project_id)
            return bool(t and t.is_alive())

    def status(self, project_id: str) -> dict | None:
        """Live status if a job ran this process, else the persisted status.json."""
        with self._guard:
            live = self._live.get(project_id)
        if live is not None:
            return live
        return read_json(render_status_file(self.cfg, project_id))

    def submit(self, project_id: str, work: Work,
               initial: dict | None = None) -> bool:
        """Start ``work`` for ``project_id``; return ``False`` if one is running.

        ``initial`` seeds the live status shown until the work first reports
        progress. Any exception the work raises is caught and recorded as a
        ``state=error`` status (a truncated traceback), never crashing the thread.
        """
        with self._guard:
            t = self._threads.get(project_id)
            if t and t.is_alive():
                return False

            def _progress(st: dict) -> None:
                atomic_write_json(render_status_file(self.cfg, project_id), st)
                with self._guard:
                    self._live[project_id] = st

            def _run() -> None:
                try:
                    work(_progress)
                except Exception as e:  # noqa: BLE001 - record + surface, never crash
                    tb = "\n".join(traceback.format_exc().splitlines()[:30])
                    _progress({"state": "error", "stage": None, "pct": -1,
                               "message": f"error: {type(e).__name__}",
                               "error": {"code": type(e).__name__, "detail": tb}})
                finally:
                    persisted = read_json(render_status_file(self.cfg, project_id))
                    if persisted is not None:
                        with self._guard:
                            self._live[project_id] = persisted

            th = threading.Thread(target=_run, name=f"studio-render-{project_id}",
                                  daemon=True)
            self._threads[project_id] = th
            self._live[project_id] = initial or {"state": "processing", "stage": None,
                                                 "pct": -1, "message": "starting",
                                                 "error": None}
            th.start()
            return True
