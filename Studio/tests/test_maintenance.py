"""maintenance.sweep_projects: cache/backup/tmp pruning, exports untouched.

Pure filesystem (no torch, no GPU pipeline): a fake project tree is built on a
tmp projects_root and the sweep asserted to trim each footprint to budget while
respecting the render manifest and never touching ``exports/``.
"""

from __future__ import annotations

import dataclasses
import os

import pytest

from Studio import library
from Studio.cache import seg_wav_name
from Studio.config import CacheParams, RetentionParams, load_config
from Studio.maintenance import sweep_projects


@pytest.fixture()
def cfg(tmp_path):
    # Small budgets so the sweep actually prunes: cache keeps only manifest hashes
    # (max_files below the kept count), backups trimmed to the newest 3.
    return dataclasses.replace(
        load_config(), projects_root=str(tmp_path),
        cache=CacheParams(max_files=2, max_mb=100.0),
        retention=RetentionParams(backups_keep=3))


def _touch(path, data=b"\0"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _build_project(cfg, pid="proj"):
    # project.json
    library.atomic_write_json(library.project_file(cfg, pid),
                              {"project_id": pid, "rev": 5, "notes": []})
    # render/cache: two manifest segments (a, b) + three stale (x, y, z).
    cdir = library.render_cache_dir(cfg, pid)
    for h in ("a", "b", "x", "y", "z"):
        _touch(os.path.join(cdir, seg_wav_name(h)))
    library.atomic_write_json(
        library.render_meta_file(cfg, pid),
        {"segments": [{"hash": "a"}, {"hash": "b"}]})
    # project.backup: five rev_* files (rev_00001..rev_00005), sorted by name.
    bdir = library.backup_dir(cfg, pid)
    for i in range(1, 6):
        _touch(os.path.join(bdir, f"rev_{i:05d}_20260101-00000{i}.json"))
    # exports/: one user file that must never be touched.
    _touch(os.path.join(library.exports_dir(cfg, pid), "out.wav"))
    # A stray interrupted atomic-write tempfile in the project dir.
    _touch(os.path.join(library.project_dir(cfg, pid), ".tmp_deadbeef.json"))
    return pid


def test_sweep_prunes_cache_respecting_manifest(cfg):
    pid = _build_project(cfg)
    sweep_projects(cfg)
    cdir = library.render_cache_dir(cfg, pid)
    remaining = {f for f in os.listdir(cdir) if f.startswith("seg_")}
    # Manifest segments survive; all three stale segments are pruned.
    assert remaining == {seg_wav_name("a"), seg_wav_name("b")}


def test_sweep_prunes_backups_to_budget(cfg):
    pid = _build_project(cfg)
    sweep_projects(cfg)
    bdir = library.backup_dir(cfg, pid)
    revs = sorted(f for f in os.listdir(bdir) if f.startswith("rev_"))
    # Newest 3 kept (rev_00003..rev_00005), oldest 2 dropped.
    assert revs == ["rev_00003_20260101-000003.json",
                    "rev_00004_20260101-000004.json",
                    "rev_00005_20260101-000005.json"]


def test_sweep_leaves_exports_untouched(cfg):
    pid = _build_project(cfg)
    sweep_projects(cfg)
    assert os.path.isfile(os.path.join(library.exports_dir(cfg, pid), "out.wav"))


def test_sweep_clears_stray_tmp(cfg):
    pid = _build_project(cfg)
    sweep_projects(cfg)
    pdir = library.project_dir(cfg, pid)
    assert not os.path.isfile(os.path.join(pdir, ".tmp_deadbeef.json"))
    assert os.path.isfile(library.project_file(cfg, pid))   # real project.json kept


def test_sweep_tolerates_missing_meta(cfg, tmp_path):
    # A project with cache WAVs but no render/meta.json: empty keep set, count budget
    # still applies (max_files=2), and the sweep must not crash.
    pid = "nometa"
    library.atomic_write_json(library.project_file(cfg, pid),
                              {"project_id": pid, "rev": 1, "notes": []})
    cdir = library.render_cache_dir(cfg, pid)
    for h in ("p", "q", "r", "s"):
        _touch(os.path.join(cdir, seg_wav_name(h)))
    sweep_projects(cfg)
    remaining = {f for f in os.listdir(cdir) if f.startswith("seg_")}
    assert len(remaining) == 2   # count budget kept the 2 newest stale WAVs
