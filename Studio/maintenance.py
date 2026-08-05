"""Periodic housekeeping sweep: bound per-project cache + backup footprint.

Studio accumulates two kinds of per-project bloat over a project's life: stale
``render/cache/seg_<hash>.wav`` segments left behind as notes are edited (each
edit re-hashes a segment, orphaning the old WAV until the next render's GC runs),
and ``project.backup/rev_*.json`` rolling backups. :func:`sweep_projects` walks
every project once at boot and trims both back to the configured budgets
(``cfg.cache`` for the segment cache, ``cfg.retention`` for backups), and clears
``.tmp*`` json files a crash mid-:func:`Studio.library.atomic_write_json` may have
left in a project dir.

The sweep is **filesystem-only** (no torch, no model/vocoder) and defensive: each
project's work is wrapped so one unreadable/corrupt project cannot abort the whole
pass. It never touches ``exports/`` (user-facing output the app never regenerates).
Run on a daemon thread from :func:`Studio.server.create_app` so boot stays instant.
"""

from __future__ import annotations

import logging
import os

from . import library
from .cache import prune_cache
from .config import StudioConfig

logger = logging.getLogger(__name__)


def _keep_hashes(cfg: StudioConfig, pid: str) -> set[str]:
    """Segment hashes in a project's ``render/meta.json`` manifest (empty if absent).

    Tolerates a missing or malformed meta file / segments list — anything other than
    a clean list of ``{"hash": ...}`` entries yields an empty keep set (so a broken
    manifest never protects stale WAVs, and never crashes the sweep).
    """
    meta = library.read_json(library.render_meta_file(cfg, pid), default={}) or {}
    try:
        return {s["hash"] for s in meta.get("segments", [])}
    except (TypeError, KeyError):
        return set()


def _prune_backups(cfg: StudioConfig, pid: str) -> int:
    """Trim ``project.backup/rev_*`` to the newest ``cfg.retention.backups_keep``.

    Same filename-sort convention as :func:`Studio.project_store._roll_backup`
    (``rev_<rev>_<ts>`` sorts oldest-first). Returns the number of files removed.
    """
    bdir = library.backup_dir(cfg, pid)
    if not os.path.isdir(bdir):
        return 0
    revs = sorted(f for f in os.listdir(bdir) if f.startswith("rev_"))
    removed = 0
    for stale in revs[:-cfg.retention.backups_keep]:
        try:
            os.remove(os.path.join(bdir, stale))
            removed += 1
        except OSError:
            pass
    return removed


def _clear_tmp(cfg: StudioConfig, pid: str) -> int:
    """Delete stray ``.tmp*.json`` files an interrupted atomic write left in the dir.

    :func:`Studio.library.atomic_write_json` writes to a same-dir ``.tmp_*.json``
    tempfile before ``os.replace``; a crash between create and replace orphans it.
    Only the top-level project dir is swept. Returns the number of files removed.
    """
    pdir = library.project_dir(cfg, pid)
    if not os.path.isdir(pdir):
        return 0
    removed = 0
    for f in os.listdir(pdir):
        if f.startswith(".tmp") and f.endswith(".json"):
            try:
                os.remove(os.path.join(pdir, f))
                removed += 1
            except OSError:
                pass
    return removed


def _sweep_one(cfg: StudioConfig, pid: str) -> None:
    """Trim one project's cache, backups and stray tmp files (best-effort)."""
    keep = _keep_hashes(cfg, pid)
    deleted = prune_cache(library.render_cache_dir(cfg, pid), keep,
                          cfg.cache.max_files, cfg.cache.max_mb)
    backups = _prune_backups(cfg, pid)
    tmps = _clear_tmp(cfg, pid)
    if deleted or backups or tmps:
        logger.info("maintenance %s: pruned %d cache, %d backups, %d tmp",
                    pid, len(deleted), backups, tmps)


def sweep_projects(cfg: StudioConfig) -> None:
    """Trim every project's cache + backups + stray tmp files back to budget.

    Iterates :func:`Studio.library.scan_project_ids`; each project's work is wrapped
    in try/except so one bad project cannot abort the sweep. Never touches
    ``exports/``. Safe to run on a daemon thread at startup.
    """
    for pid in library.scan_project_ids(cfg):
        try:
            _sweep_one(cfg, pid)
        except Exception:  # noqa: BLE001 — one bad project must not kill the sweep
            logger.exception("maintenance sweep failed for project %s", pid)
