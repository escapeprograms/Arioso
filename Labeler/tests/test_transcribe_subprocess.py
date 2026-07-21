"""Transcribe subprocess isolation: command/env wiring + failure -> clean error.

All model-free (subprocess.run is monkeypatched): the GPU forward pass itself is
covered elsewhere. Here we assert the isolation contract — the child is invoked
with the right module/config/cwd/allocator env, a non-zero exit or timeout is
raised as a RuntimeError the stage runner can record, and the in-process fallback
routes through the same shared core without spawning.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import types

import pytest

from Labeler import processing, transcribe_worker
from Labeler.config import load_config


@pytest.fixture()
def cfg(tmp_path):
    return dataclasses.replace(load_config(), dataset_root=str(tmp_path))


def _ok_proc(stdout=""):
    return types.SimpleNamespace(stdout=stdout, stderr="", returncode=0)


def test_subprocess_builds_cmd_env_and_cwd(cfg, monkeypatch):
    calls = {}

    def fake_run(cmd, **kw):
        calls["cmd"], calls["kw"] = cmd, kw
        return _ok_proc("[transcribe_worker] clipX: peak CUDA 500 MB\n")

    monkeypatch.setattr(processing.subprocess, "run", fake_run)
    processing.Pipeline(cfg, "clipX")._transcribe_subprocess()

    assert calls["cmd"][:4] == [sys.executable, "-m", "Labeler.transcribe_worker", "clipX"]
    assert "--config" in calls["cmd"]                    # cfg carries a config_path
    assert calls["kw"]["cwd"] == processing._PROJECT_ROOT
    assert calls["kw"]["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert calls["kw"]["timeout"] == cfg.transcribe.subprocess_timeout_s
    assert calls["kw"]["check"] is True


def test_subprocess_omits_config_when_none(cfg, monkeypatch):
    cfg2 = dataclasses.replace(cfg, config_path=None)
    calls = {}

    def fake_run(cmd, **kw):
        calls["cmd"] = cmd
        return _ok_proc()

    monkeypatch.setattr(processing.subprocess, "run", fake_run)
    processing.Pipeline(cfg2, "clipX")._transcribe_subprocess()
    assert "--config" not in calls["cmd"]


def test_nonzero_exit_maps_to_runtimeerror(cfg, monkeypatch):
    def fake_run(cmd, **kw):
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, stderr="Traceback\nRuntimeError: CUDA out of memory")

    monkeypatch.setattr(processing.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as ei:
        processing.Pipeline(cfg, "clipX")._transcribe_subprocess()
    msg = str(ei.value)
    assert "exited 1" in msg and "CUDA out of memory" in msg   # stderr tail surfaced


def test_timeout_maps_to_runtimeerror(cfg, monkeypatch):
    def fake_run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kw.get("timeout"), stderr="…")

    monkeypatch.setattr(processing.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="timed out"):
        processing.Pipeline(cfg, "clipX")._transcribe_subprocess()


def test_inprocess_fallback_uses_core_without_spawning(cfg, monkeypatch):
    cfg2 = dataclasses.replace(
        cfg, transcribe=dataclasses.replace(cfg.transcribe, use_subprocess=False))
    seen = {}
    monkeypatch.setattr(transcribe_worker, "run_transcribe",
                        lambda cleaned, tj, mid, params: seen.update(params=params))
    monkeypatch.setattr(processing.subprocess, "run",
                        lambda *a, **k: pytest.fail("must not spawn a subprocess"))
    processing.Pipeline(cfg2, "clipX")._stage_transcribe()
    assert seen["params"] is cfg2.transcribe


def test_worker_main_resolves_paths_and_calls_core(cfg, monkeypatch):
    captured = {}
    monkeypatch.setattr(transcribe_worker, "load_config", lambda p: cfg)
    monkeypatch.setattr(
        transcribe_worker, "run_transcribe",
        lambda cleaned, tj, mid, params: captured.update(
            cleaned=cleaned, tj=tj, mid=mid, params=params))

    rc = transcribe_worker.main(["clipZ"])

    assert rc == 0
    assert captured["cleaned"].endswith("cleaned.wav") and "clipZ" in captured["cleaned"]
    assert captured["tj"].endswith("transcription.json")
    assert captured["mid"].endswith("musc_raw.mid")
    assert captured["params"] is cfg.transcribe
