"""compile_all's post-run split reconciliation (``Labeler.compile.update_root_split``).

Every compile must leave ``<root>/split.json`` covering the whole manifest — the failure mode
this closes is a clip compiled after the split was written being in neither train nor val, and
so invisible to training. A full compile fixture (wavs + MUSC-shaped notes) would be
disproportionate here, so the hook is exercised through a no-op ``compile_all`` run (empty
``clip_ids``: nothing compiled, nothing pruned) over a hand-written manifest — which is exactly
the state the hook sees, since the manifest is on disk before it runs.
"""

from __future__ import annotations

import dataclasses
import json
import os

import pytest

from common.config import HOP_SIZE, N_MELS, SR
from common.dataset_schema import MANIFEST_SCHEMA_VERSION, SPLIT_JSON, write_manifest

from Labeler.compile import compile_all, update_root_split
from Labeler.config import load_config


@pytest.fixture()
def cfg(tmp_path):
    """A Labeler config whose dataset root (no clips) and compile root are both temporary."""
    base = load_config()
    return dataclasses.replace(base,
                               dataset_root=str(tmp_path / "labeler"),
                               compile=dataclasses.replace(base.compile,
                                                           root=str(tmp_path / "gt_root")))


def _seed_manifest(root: str, bases: list[str]) -> None:
    """Write a manifest listing ``bases`` (piece == clip id, the gt_arky convention)."""
    os.makedirs(root, exist_ok=True)
    write_manifest(root, {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "name": "gt_arky",
        "frame": {"sr": SR, "hop": HOP_SIZE, "n_mels": N_MELS},
        "signals": [],
        "clips": {b: {"n_frames": 100, "piece": b} for b in bases},
    })


def _read_split(root: str) -> dict:
    with open(os.path.join(root, SPLIT_JSON), encoding="utf-8") as f:
        return json.load(f)


def test_compile_all_creates_split_covering_the_manifest(cfg):
    root = cfg.compile.root
    _seed_manifest(root, [f"clip{i:02d}" for i in range(20)])

    summary = compile_all(cfg, clip_ids=[], verbose=False)

    assert summary["split"]["created"] is True
    split = _read_split(root)
    assert set(split["train"]) | set(split["val"]) == {f"clip{i:02d}" for i in range(20)}
    assert split["n_val_pieces"] == 2


def test_compile_all_adds_clips_compiled_after_the_split(cfg, capsys):
    """The regression: a split written when the corpus was smaller must gain the new clips."""
    root = cfg.compile.root
    _seed_manifest(root, [f"clip{i:02d}" for i in range(20)])
    compile_all(cfg, clip_ids=[], verbose=False)
    before = _read_split(root)

    _seed_manifest(root, [f"clip{i:02d}" for i in range(30)])       # 10 more compiled clips
    summary = compile_all(cfg, clip_ids=[], verbose=True)

    info = summary["split"]
    assert info["created"] is False
    assert info["added_train"] + info["added_val"] == 10
    assert info["pruned"] == 0
    split = _read_split(root)
    assert set(split["train"]) | set(split["val"]) == {f"clip{i:02d}" for i in range(30)}
    assert set(before["val"]) <= set(split["val"])                  # val never reshuffles
    assert "[compile] split" in capsys.readouterr().out


def test_split_failure_does_not_fail_the_compile(cfg, monkeypatch):
    import Arioso.splits

    def _boom(*a, **kw):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(Arioso.splits, "update_split", _boom)
    _seed_manifest(cfg.compile.root, ["clip00"])

    summary = compile_all(cfg, clip_ids=[], verbose=False)

    assert summary["failed"] == 0                                   # compile itself is fine
    assert "disk on fire" in summary["split"]["error"]
    assert not os.path.isfile(os.path.join(cfg.compile.root, SPLIT_JSON))


def test_no_manifest_is_skipped_not_an_error(cfg):
    # A first run that compiled nothing never wrote a manifest: nothing to reconcile.
    summary = compile_all(cfg, clip_ids=[], verbose=False)
    assert summary["split"] == {"skipped": "no manifest"}


def test_update_root_split_returns_the_arioso_summary(tmp_path):
    root = str(tmp_path / "root")
    _seed_manifest(root, [f"clip{i:02d}" for i in range(20)])
    info = update_root_split(root, verbose=False)
    assert info["created"] is True and info["n_pieces"] == 20
    assert set(info) == {"created", "added_train", "added_val", "pruned",
                         "n_pieces", "n_val_pieces"}
