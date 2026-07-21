"""library.reconcile_interrupted: crash-wedged 'processing' -> re-runnable error.

A hard crash during a stage (typically GPU transcribe) leaves status.json frozen
at state='processing'. The startup sweep must reset it to 'error' so the UI offers
process again, while preserving the per-stage skip cache so only the interrupted
stage re-runs. Done/error/ready statuses must be left untouched.
"""

from __future__ import annotations

import dataclasses
import os

import pytest

from Labeler.config import load_config
from Labeler.library import (STATUS_JSON, atomic_write_json, clip_dir,
                             default_status, reconcile_interrupted, read_json)


@pytest.fixture()
def cfg(tmp_path):
    return dataclasses.replace(load_config(), dataset_root=str(tmp_path))


def _write_status(cfg, clip_id, state, **extra):
    d = clip_dir(cfg, clip_id)
    os.makedirs(d, exist_ok=True)
    status = default_status(clip_id, None)
    status["state"] = state
    status.update(extra)
    atomic_write_json(os.path.join(d, STATUS_JSON), status)
    return os.path.join(d, STATUS_JSON)


def test_resets_processing_preserving_stages(cfg):
    stages = {"ingest": {"state": "done", "hash": "aaa"},
              "denoise": {"state": "done", "hash": "bbb"}}
    sp = _write_status(cfg, "clipP", "processing", stage="transcribe", stages=stages)

    reset = reconcile_interrupted(cfg)

    assert reset == ["clipP"]
    new = read_json(sp)
    assert new["state"] == "error"
    assert new["error"]["code"] == "interrupted"
    assert new["error"]["stage"] == "transcribe"      # remembers where it died
    assert new["stages"] == stages                    # skip-cache untouched


def test_leaves_terminal_states_untouched(cfg):
    sp_done = _write_status(cfg, "clipD", "done")
    sp_err = _write_status(cfg, "clipE", "error")

    assert reconcile_interrupted(cfg) == []
    assert read_json(sp_done)["state"] == "done"
    assert read_json(sp_err)["state"] == "error"


def test_missing_clips_dir_is_noop(cfg):
    # Fresh dataset root: no clips/ dir at all.
    assert reconcile_interrupted(cfg) == []


def test_resets_only_the_wedged_clips(cfg):
    _write_status(cfg, "aProc", "processing")
    _write_status(cfg, "bDone", "done")
    _write_status(cfg, "cProc", "processing")

    assert reconcile_interrupted(cfg) == ["aProc", "cProc"]   # sorted, processing-only
