"""Held-out-piece split: verbatim load + stale warning (``make_split``) and the additive
reconciliation (``update_split``).

The contract under test:

* ``make_split`` is unchanged — an existing ``split.json`` is still loaded byte-for-byte (a
  migrated root keeps its exact held-out split); it only *warns* when that split no longer
  covers the manifest.
* ``update_split`` never moves a surviving assignment (val stability across corpus growth),
  prunes basenames that left the manifest, assigns new ones by whole piece up to the val
  target, is deterministic, and writes the same four-key schema.
"""

from __future__ import annotations

import dataclasses
import json
import os

import pytest

from common.config import HOP_SIZE, N_MELS, SR
from common.dataset_schema import (MANIFEST_SCHEMA_VERSION, DatasetRoot, write_manifest)

from Arioso.config import SPLIT_FILE, AriosoConfig
from Arioso.splits import make_split, missing_from_split, update_split


def _write_root(path: str, clips: dict[str, str]) -> DatasetRoot:
    """Write a minimal manifest (``{basename: piece}``) and open it as a DatasetRoot."""
    os.makedirs(path, exist_ok=True)
    write_manifest(path, {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "name": "test_root",
        "frame": {"sr": SR, "hop": HOP_SIZE, "n_mels": N_MELS},
        "signals": [],
        "clips": {base: {"n_frames": 100, "piece": piece} for base, piece in clips.items()},
    })
    return DatasetRoot(path)


def _one_piece_each(n: int, prefix: str = "clip") -> dict[str, str]:
    """``n`` clips, each its own piece (the gt_arky convention: piece == clip id)."""
    return {f"{prefix}{i:03d}": f"{prefix}{i:03d}" for i in range(n)}


@pytest.fixture()
def root(tmp_path):
    return _write_root(str(tmp_path / "root"), _one_piece_each(20))


def _read_split(root: DatasetRoot) -> dict:
    with open(os.path.join(root.path, SPLIT_FILE), encoding="utf-8") as f:
        return json.load(f)


def _regrow(root: DatasetRoot, clips: dict[str, str]) -> DatasetRoot:
    """Rewrite the root's manifest with ``clips`` and reopen it (corpus grew / shrank)."""
    return _write_root(root.path, clips)


# --- make_split: unchanged semantics --------------------------------------------

def test_make_split_holds_out_val_frac_of_pieces(root):
    split = make_split(root)
    assert split["n_pieces"] == 20
    assert split["n_val_pieces"] == 2                    # max(1, round(20 * 0.10))
    assert len(split["val"]) == 2 and len(split["train"]) == 18
    assert set(split["train"]) | set(split["val"]) == set(root.basenames())
    assert split["train"] == sorted(split["train"]) and split["val"] == sorted(split["val"])


def test_make_split_loads_existing_verbatim(root):
    make_split(root)
    # Hand-edit the file: a migrated root's exact split must survive untouched.
    hand = {"train": ["clip000"], "val": ["clip001"], "n_pieces": 2, "n_val_pieces": 1}
    with open(os.path.join(root.path, SPLIT_FILE), "w", encoding="utf-8") as f:
        json.dump(hand, f)
    assert make_split(root) == hand


def test_make_split_overwrite_recomputes(root):
    make_split(root)
    with open(os.path.join(root.path, SPLIT_FILE), "w", encoding="utf-8") as f:
        json.dump({"train": [], "val": [], "n_pieces": 0, "n_val_pieces": 0}, f)
    split = make_split(root, overwrite=True)
    assert len(split["train"]) + len(split["val"]) == 20


def test_make_split_is_deterministic(tmp_path):
    a = make_split(_write_root(str(tmp_path / "a"), _one_piece_each(20)))
    b = make_split(_write_root(str(tmp_path / "b"), _one_piece_each(20)))
    assert a == b


def test_make_split_warns_on_stale_split(root, capsys):
    make_split(root)
    grown = _regrow(root, _one_piece_each(25))
    split = make_split(grown)                            # verbatim load, unchanged
    out = capsys.readouterr().out
    assert "WARNING" in out and "--update" in out
    assert "5 of 25" in out
    assert len(split["train"]) + len(split["val"]) == 20  # warning only: no behavior change


def test_make_split_quiet_when_complete(root, capsys):
    make_split(root)
    capsys.readouterr()
    make_split(root)
    assert capsys.readouterr().out == ""
    assert missing_from_split(root, _read_split(root)) == []


# --- update_split: fresh root ----------------------------------------------------

def test_update_on_fresh_root_delegates_to_make_split(tmp_path):
    a = _write_root(str(tmp_path / "a"), _one_piece_each(20))
    b = _write_root(str(tmp_path / "b"), _one_piece_each(20))
    split, info = update_split(a)
    assert split == make_split(b)
    assert info["created"] is True
    assert info["pruned"] == 0
    assert info["added_train"] == 18 and info["added_val"] == 2
    assert info["n_pieces"] == 20 and info["n_val_pieces"] == 2


# --- update_split: growth --------------------------------------------------------

def test_update_keeps_existing_assignments_and_assigns_new(root):
    before = make_split(root)
    grown = _regrow(root, _one_piece_each(40))
    split, info = update_split(grown)

    # (a) nothing moved: every old assignment is still on its old side.
    assert set(before["val"]) <= set(split["val"])
    assert set(before["train"]) <= set(split["train"])
    # (b) the manifest is fully covered, exactly once.
    assert set(split["train"]) | set(split["val"]) == set(grown.basenames())
    assert not set(split["train"]) & set(split["val"])
    # (c) the val target is met at the piece level: max(1, round(40 * 0.10)) == 4.
    assert split["n_pieces"] == 40 and split["n_val_pieces"] == 4
    assert len(split["val"]) == 4
    assert info["added_val"] == 2 and info["added_train"] == 18 and info["pruned"] == 0


def test_update_sends_new_clip_to_its_pieces_existing_side(root):
    """A new basename of an *already assigned* piece follows that piece (no piece in both)."""
    before = make_split(root)
    val_piece = root.piece(before["val"][0])
    clips = _one_piece_each(20)
    clips["newclip_val"] = val_piece                     # same piece as a held-out clip
    split, _ = update_split(_regrow(root, clips))
    assert "newclip_val" in split["val"]
    assert split["n_pieces"] == 20 and split["n_val_pieces"] == 2


def test_update_no_new_val_when_target_already_met(root):
    """Growth that leaves the target met adds only train clips (val never shrinks/grows blindly)."""
    make_split(root)                                     # 20 pieces -> 2 val
    split, info = update_split(_regrow(root, _one_piece_each(24)))   # target still round(2.4) == 2
    assert info["added_val"] == 0 and info["added_train"] == 4
    assert split["n_val_pieces"] == 2


# --- update_split: pruning -------------------------------------------------------

def test_update_prunes_departed_basenames(root):
    before = make_split(root)
    keep = {b: b for b in root.basenames() if b not in (before["train"][0], before["val"][0])}
    split, info = update_split(_regrow(root, keep))
    assert before["train"][0] not in split["train"]
    assert before["val"][0] not in split["val"]
    assert info["pruned"] == 2 and info["added_train"] == 0 and info["added_val"] == 0
    assert split["n_pieces"] == 18 and split["n_val_pieces"] == 1


# --- update_split: determinism + schema ------------------------------------------

def test_update_is_deterministic(tmp_path):
    def build(name):
        r = _write_root(str(tmp_path / name), _one_piece_each(20))
        make_split(r)
        return update_split(_write_root(str(tmp_path / name), _one_piece_each(40)))

    (split_a, info_a), (split_b, info_b) = build("a"), build("b")
    assert split_a == split_b and info_a == info_b


def test_update_is_idempotent(root):
    make_split(root)
    grown = _regrow(root, _one_piece_each(40))
    first, _ = update_split(grown)
    second, info = update_split(grown)
    assert first == second
    assert info["added_train"] == 0 and info["added_val"] == 0 and info["pruned"] == 0


def test_update_writes_the_same_schema(root):
    make_split(root)
    split, _ = update_split(_regrow(root, _one_piece_each(40)))
    on_disk = _read_split(root)
    assert on_disk == split
    assert set(on_disk) == {"train", "val", "n_pieces", "n_val_pieces"}
    assert on_disk["train"] == sorted(on_disk["train"])
    assert on_disk["val"] == sorted(on_disk["val"])


def test_update_respects_cfg_val_frac(root):
    make_split(root)
    cfg = dataclasses.replace(AriosoConfig(), val_frac=0.25)
    split, info = update_split(_regrow(root, _one_piece_each(40)), cfg)
    assert split["n_val_pieces"] == 10 and info["added_val"] == 8


def test_update_multi_clip_pieces_stay_whole(tmp_path):
    """Pieces with several recordings are assigned as a unit (the held-out-piece invariant)."""
    clips = {f"rec{i:02d}": f"piece{i // 4}" for i in range(40)}   # 10 pieces x 4 recordings
    root = _write_root(str(tmp_path / "multi"), clips)
    split, info = update_split(root)                               # fresh -> make_split
    assert info["created"] is True
    for side in ("train", "val"):
        pieces = {root.piece(b) for b in split[side]}
        for piece in pieces:
            assert len([b for b in split[side] if root.piece(b) == piece]) == 4
    assert split["n_val_pieces"] == 1 and len(split["val"]) == 4
