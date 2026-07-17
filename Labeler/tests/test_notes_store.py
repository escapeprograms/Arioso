"""notes_store: save, rev-conflict, and retranscribe-merge label preservation."""

from __future__ import annotations

import dataclasses

import pytest

from Labeler import notes_store
from Labeler.config import load_config


@pytest.fixture()
def cfg(tmp_path):
    return dataclasses.replace(load_config(), dataset_root=str(tmp_path))


def _note(start, pitch, tech="normal", vib=False, vel=80):
    return {"start_s": start, "end_s": start + 0.5, "pitch": pitch,
            "velocity": vel, "technique": tech, "vibrato": vib,
            "auto": {"amplitude": 0.5, "velocity": vel}}


def _doc(cfg, clip_id, notes):
    return notes_store.build_document(
        cfg, clip_id, notes, offset_applied_s=0.0, duration_s=3.0,
        audio_meta={"sr": 44100, "hop": 512}, pipeline_meta={})


def test_save_new_sets_rev_1(cfg):
    doc = _doc(cfg, "clipA", [_note(0.0, 60), _note(1.0, 62)])
    saved = notes_store.save_new(cfg, "clipA", doc)
    assert saved["rev"] == 1
    loaded = notes_store.load(cfg, "clipA")
    assert loaded["rev"] == 1 and len(loaded["notes"]) == 2
    assert loaded["notes"][0]["id"] == "n0000"


def test_put_with_rev_ok_then_conflict(cfg):
    notes_store.save_new(cfg, "clipB", _doc(cfg, "clipB", [_note(0.0, 60)]))
    cur = notes_store.load(cfg, "clipB")
    # Correct base rev -> ok, rev bumps to 2.
    res = notes_store.put_with_rev(cfg, "clipB", cur)
    assert res["status"] == "ok" and res["rev"] == 2
    # Stale base rev (still 1) -> conflict carrying the server rev.
    stale = dict(cur)
    stale["rev"] = 1
    res2 = notes_store.put_with_rev(cfg, "clipB", stale)
    assert res2["status"] == "conflict" and res2["server_rev"] == 2


def test_merge_retranscribe_preserves_human_facet_on_match(cfg):
    old = _doc(cfg, "clipC", [_note(0.0, 60, tech="spiccato")])
    old["notes"][0]["human"]["technique"] = True
    new = _doc(cfg, "clipC", [_note(0.01, 60, tech="normal"), _note(2.0, 64)])
    merged = notes_store.merge_retranscribe(old, new)
    # The near-onset same-pitch note inherits the human technique + flag.
    assert merged["notes"][0]["technique"] == "spiccato"
    assert merged["notes"][0]["human"]["technique"] is True
    # The unrelated new note stays auto.
    assert merged["notes"][1]["technique"] == "normal"
    assert merged["notes"][1]["human"]["technique"] is False
    assert merged["orphaned_labels"] == []


def test_merge_retranscribe_orphans_unmatched_human_note(cfg):
    old = _doc(cfg, "clipD", [_note(1.0, 62, tech="slur")])
    old["notes"][0]["human"]["technique"] = True
    new = _doc(cfg, "clipD", [_note(1.0, 60)])  # different pitch -> no match
    merged = notes_store.merge_retranscribe(old, new)
    assert merged["notes"][0]["technique"] == "normal"
    assert len(merged["orphaned_labels"]) == 1
    assert merged["orphaned_labels"][0]["pitch"] == 62


def test_merge_respects_onset_tolerance(cfg):
    # Same pitch but 200 ms away (> 60 ms tol) -> no match -> orphan.
    old = _doc(cfg, "clipE", [_note(0.0, 60, vel=120)])
    old["notes"][0]["human"]["velocity"] = True
    new = _doc(cfg, "clipE", [_note(0.2, 60)])
    merged = notes_store.merge_retranscribe(old, new)
    assert merged["notes"][0]["velocity"] != 120
    assert len(merged["orphaned_labels"]) == 1
