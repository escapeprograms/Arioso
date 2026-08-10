"""render endpoint contract — run_render monkeypatched out (no torch). """

from __future__ import annotations

import dataclasses
import threading
import time

import pytest
from fastapi.testclient import TestClient

import Studio.render as render_mod
from Studio.config import load_config
from Studio.server import create_app


@pytest.fixture()
def client(tmp_path):
    projects = tmp_path / "projects"
    models = tmp_path / "models"
    d = models / "7-9-adr"
    d.mkdir(parents=True)
    (d / "checkpoint_final.pt").write_bytes(b"")
    # Stub vocoders_root so the render endpoint's validation is hermetic (the real
    # Vocoder/models may hold any set of fine-tunes, or none on a fresh clone).
    vocoders = tmp_path / "vocoders"
    v = vocoders / "ft_v2"
    v.mkdir(parents=True)
    (v / "config.json").write_text("{}", encoding="utf-8")
    (v / "bigvgan_generator.pt").write_bytes(b"")
    cfg = dataclasses.replace(load_config(), projects_root=str(projects),
                              models_root=str(models),
                              vocoders_root=str(vocoders))
    return TestClient(create_app(cfg))


def _new_project(client, name):
    return client.post("/api/projects", json={"name": name}).json()["project_id"]


def _wait_idle(client, pid, timeout=5.0):
    jobs = client.app.state.jobs
    deadline = time.time() + timeout
    while jobs.is_running(pid) and time.time() < deadline:
        time.sleep(0.01)
    assert not jobs.is_running(pid), "render job did not finish in time"


def _done_stub(cfg, doc, project_dir, *, scope="phrase", note_ids=None, model=None,
               checkpoint=None, prior_mode=None, vocoder=None, device=None,
               progress=None):
    if progress is not None:
        progress({"state": "done", "stage": "write", "pct": 100,
                  "message": "done", "error": None})
    return {"ok": True, "model": model, "prior_mode": prior_mode,
            "vocoder": vocoder, "scope": scope}


def test_render_accepts_and_completes(client, monkeypatch):
    monkeypatch.setattr(render_mod, "run_render", _done_stub)
    pid = _new_project(client, "R")
    r = client.post(f"/api/projects/{pid}/render", json={})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "accepted"
    assert body["model"] == "7-9-adr" and body["prior_mode"] == "bend"
    assert body["vocoder"] == "ft_v2"          # cfg default when the body omits it

    _wait_idle(client, pid)
    st = client.get(f"/api/projects/{pid}/render/status").json()
    assert st["state"] == "done"


def test_render_accepts_explicit_and_hf_vocoder(client, monkeypatch):
    seen = []

    def _stub(cfg, doc, project_dir, **kw):
        seen.append(kw.get("vocoder"))
        return _done_stub(cfg, doc, project_dir, **kw)

    monkeypatch.setattr(render_mod, "run_render", _stub)
    pid = _new_project(client, "V")
    for name in ("ft_v2", "hf"):
        r = client.post(f"/api/projects/{pid}/render", json={"vocoder": name})
        assert r.status_code == 202, r.json()
        assert r.json()["vocoder"] == name
        _wait_idle(client, pid)
    assert seen == ["ft_v2", "hf"]


def test_render_unknown_vocoder_400(client):
    # Validated before any job is submitted, so no stub is needed.
    pid = _new_project(client, "VBad")
    r = client.post(f"/api/projects/{pid}/render", json={"vocoder": "nope"})
    assert r.status_code == 400
    assert r.json()["error"] == "vocoder_not_found"


def test_render_vocoder_path_traversal_400(client):
    pid = _new_project(client, "VEvil")
    for name in ("..\\evil", "../evil", "sub/dir"):
        r = client.post(f"/api/projects/{pid}/render", json={"vocoder": name})
        assert r.status_code == 400
        assert r.json()["error"] == "vocoder_not_found"


def test_render_conflict_when_running(client, monkeypatch):
    gate = threading.Event()

    def _blocking(cfg, doc, project_dir, *, progress=None, **kw):
        gate.wait(2.0)
        if progress is not None:
            progress({"state": "done", "stage": "write", "pct": 100,
                      "message": "done", "error": None})

    monkeypatch.setattr(render_mod, "run_render", _blocking)
    pid = _new_project(client, "Busy")
    assert client.post(f"/api/projects/{pid}/render", json={}).status_code == 202
    conflict = client.post(f"/api/projects/{pid}/render", json={})
    assert conflict.status_code == 409
    assert conflict.json()["error"] == "job_running"
    gate.set()
    _wait_idle(client, pid)


def test_render_meta_served_after_render(client, monkeypatch, tmp_path):
    # Stub writes a real meta.json so GET render/meta returns it.
    def _writes_meta(cfg, doc, project_dir, *, progress=None, **kw):
        import json
        import os
        rdir = os.path.join(project_dir, "render")
        os.makedirs(rdir, exist_ok=True)
        with open(os.path.join(rdir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"wav": "/media/x/render/mix.wav", "duration_s": 1.0}, f)
        if progress is not None:
            progress({"state": "done", "stage": "write", "pct": 100,
                      "message": "done", "error": None})

    monkeypatch.setattr(render_mod, "run_render", _writes_meta)
    pid = _new_project(client, "Meta")
    assert client.post(f"/api/projects/{pid}/render", json={}).status_code == 202
    _wait_idle(client, pid)
    meta = client.get(f"/api/projects/{pid}/render/meta")
    assert meta.status_code == 200
    assert meta.json()["duration_s"] == 1.0
