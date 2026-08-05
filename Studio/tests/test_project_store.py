"""project_store: build defaults, save/rev-conflict, backups. Torch-free."""

from __future__ import annotations

import dataclasses
import os

import pytest

from Studio import project_store
from Studio.config import SCHEMA_VERSION, load_config
from Studio.library import BACKUP_DIR, project_dir, render_cache_dir


@pytest.fixture()
def cfg(tmp_path):
    return dataclasses.replace(load_config(), projects_root=str(tmp_path))


def test_build_note_defaults():
    n = project_store.build_note(0, start_beat=0.0, len_beats=1.0, pitch=67)
    assert n["id"] == "n0000"
    assert n["velocity"] == 100
    assert n["technique"] == "normal"
    assert n["pan"] == 0.0
    assert n["bend"] == []
    assert n["vibrato"] == {"depth_semitones": 0.0, "rate_hz": 5.5, "onset_beats": 0.15}


def test_build_note_carries_and_clamps_pan():
    # pan is part of the schema (Phase 2) and must survive build_note, clamped to ±1.
    assert project_store.build_note(0, start_beat=0.0, len_beats=1.0, pitch=67,
                                    pan=-0.3)["pan"] == pytest.approx(-0.3)
    assert project_store.build_note(0, start_beat=0.0, len_beats=1.0, pitch=67,
                                    pan=5.0)["pan"] == 1.0


def test_build_document_propagates_pan(cfg):
    doc = project_store.build_document(
        cfg, "panned", name="P",
        notes=[{"start_beat": 0.0, "len_beats": 1.0, "pitch": 60, "pan": 0.4}])
    assert doc["notes"][0]["pan"] == pytest.approx(0.4)


def test_build_note_clamps_velocity_and_keeps_bend():
    n = project_store.build_note(
        3, start_beat=1.0, len_beats=0.5, pitch=60, velocity=999,
        technique="spiccato", bend=[{"beat": 0.0, "semitones": 1.0}],
        vibrato={"depth_semitones": 0.5, "rate_hz": 6.0, "onset_beats": 0.2})
    assert n["id"] == "n0003"
    assert n["velocity"] == 127
    assert n["technique"] == "spiccato"
    assert n["bend"] == [{"beat": 0.0, "semitones": 1.0}]
    assert n["vibrato"]["depth_semitones"] == pytest.approx(0.5)


def test_build_document_defaults(cfg):
    doc = project_store.build_document(cfg, "song", name="My Song")
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["project_id"] == "song"
    assert doc["rev"] == 0
    assert doc["name"] == "My Song"
    assert doc["bpm"] == 140.0
    assert doc["time_sig"] == {"num": 4, "den": 4}
    assert doc["ppq"] == 480
    assert doc["notes"] == []
    assert doc["next_note_ordinal"] == 0
    assert doc["render"]["model"] == cfg.default_model
    assert doc["render"]["prior_mode"] == cfg.default_prior_mode
    assert doc["view"]["snap"] == "step"


def test_create_and_load_sets_rev_1(cfg):
    saved = project_store.create_project(cfg, "songA", name="Song A")
    assert saved["rev"] == 1
    loaded = project_store.load(cfg, "songA")
    assert loaded["rev"] == 1 and loaded["name"] == "Song A"


def test_put_with_rev_ok_then_conflict(cfg):
    project_store.create_project(cfg, "songB", name="B")
    cur = project_store.load(cfg, "songB")
    res = project_store.put_with_rev(cfg, "songB", cur)
    assert res["status"] == "ok" and res["rev"] == 2
    # Stale base rev (still 1) -> conflict carrying the server rev.
    stale = dict(cur)
    stale["rev"] = 1
    res2 = project_store.put_with_rev(cfg, "songB", stale)
    assert res2["status"] == "conflict" and res2["server_rev"] == 2


def test_put_rolls_a_backup(cfg):
    project_store.create_project(cfg, "songC", name="C")
    cur = project_store.load(cfg, "songC")
    project_store.put_with_rev(cfg, "songC", cur)
    bdir = os.path.join(project_dir(cfg, "songC"), BACKUP_DIR)
    backups = [f for f in os.listdir(bdir) if f.startswith("rev_")]
    assert len(backups) == 1  # the rev-1 doc was rolled before writing rev 2


def test_set_name_bumps_rev_and_rolls_backup(cfg):
    project_store.create_project(cfg, "ren", name="Orig")
    doc = project_store.set_name(cfg, "ren", "Renamed")
    assert doc["rev"] == 2 and doc["name"] == "Renamed"
    assert project_store.load(cfg, "ren")["name"] == "Renamed"
    bdir = os.path.join(project_dir(cfg, "ren"), BACKUP_DIR)
    assert any(f.startswith("rev_") for f in os.listdir(bdir))
    # Unknown id -> None (api layer turns it into a 404).
    assert project_store.set_name(cfg, "nope", "x") is None


def test_duplicate_new_id_rev1_and_copies_cache(cfg):
    project_store.create_project(cfg, "orig", name="Orig")
    cache = render_cache_dir(cfg, "orig")
    os.makedirs(cache, exist_ok=True)
    with open(os.path.join(cache, "seg_deadbeef.wav"), "wb") as f:
        f.write(b"RIFF")
    dup = project_store.duplicate(cfg, "orig", "Orig copy")
    assert dup["project_id"] != "orig" and dup["rev"] == 1 and dup["name"] == "Orig copy"
    assert os.path.isfile(os.path.join(render_cache_dir(cfg, dup["project_id"]),
                                       "seg_deadbeef.wav"))
    assert project_store.duplicate(cfg, "nope", "x") is None


def test_summary_shape(cfg):
    doc = project_store.build_document(
        cfg, "songD", name="D",
        notes=[{"start_beat": 0.0, "len_beats": 1.0, "pitch": 60},
               {"start_beat": 1.0, "len_beats": 1.0, "pitch": 64}])
    s = project_store.summary(doc)
    assert s == {"id": "songD", "name": "D", "rev": 0, "bpm": 140.0, "n_notes": 2}
