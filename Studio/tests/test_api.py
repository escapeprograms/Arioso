"""api: FastAPI TestClient over a tmp projects_root + fake models_root. Torch-free."""

from __future__ import annotations

import dataclasses
import os

import pytest
from fastapi.testclient import TestClient

from Studio import library
from Studio.config import load_config
from Studio.server import create_app


@pytest.fixture()
def client(tmp_path):
    projects = tmp_path / "projects"
    models = tmp_path / "models"
    # Fake model runs: <run>/<checkpoint>.pt (empty files; scan is filesystem-only).
    for run, ckpts in {"7-9-adr": ["checkpoint_final.pt", "checkpoint_step5000.pt"],
                       "lr_anneal": ["checkpoint_final.pt"],
                       "empty_run": []}.items():
        d = models / run
        d.mkdir(parents=True)
        for c in ckpts:
            (d / c).write_bytes(b"")
    cfg = dataclasses.replace(load_config(), projects_root=str(projects),
                              models_root=str(models))
    return TestClient(create_app(cfg))


def test_get_config(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert body["sr"] == 44100
    assert body["hop"] == 512
    assert body["frame_rate"] == pytest.approx(44100 / 512)
    assert body["default_model"] == "7-9-adr"
    assert body["default_checkpoint"] == "checkpoint_final.pt"
    assert body["default_prior_mode"] == "bend"
    assert body["prior_modes"] == ["bend", "quantized"]
    names = [a["name"] for a in body["articulations"]]
    assert names == ["normal", "slur", "spiccato", "detache"]
    assert body["technique_model_vocab"]["spiccato"] == "spiccato"
    assert body["technique_model_vocab"]["slur"] == "slur"
    assert any(o["id"] == "step" for o in body["snap_options"])


def test_get_models(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    body = r.json()
    runs = {m["run"]: m for m in body["models"]}
    # empty_run has no .pt files -> excluded.
    assert set(runs) == {"7-9-adr", "lr_anneal"}
    assert runs["7-9-adr"]["default"] is True
    assert runs["lr_anneal"]["default"] is False
    assert runs["7-9-adr"]["checkpoints"] == ["checkpoint_final.pt", "checkpoint_step5000.pt"]


def test_projects_crud_flow(client):
    # Empty to start.
    assert client.get("/api/projects").json() == []

    # Create -> 201 with a fresh doc.
    r = client.post("/api/projects", json={"name": "My Song"})
    assert r.status_code == 201
    doc = r.json()
    pid = doc["project_id"]
    assert doc["name"] == "My Song" and doc["rev"] == 1

    # Listed now.
    listing = client.get("/api/projects").json()
    assert len(listing) == 1 and listing[0]["id"] == pid

    # Fetch it back.
    got = client.get(f"/api/projects/{pid}")
    assert got.status_code == 200 and got.json()["project_id"] == pid


def test_list_includes_modified(client):
    client.post("/api/projects", json={"name": "Modded"})
    listing = client.get("/api/projects").json()
    assert len(listing) == 1
    assert isinstance(listing[0]["modified"], (int, float)) and listing[0]["modified"] > 0


def test_rename_bumps_rev_and_persists(client):
    pid = client.post("/api/projects", json={"name": "Old"}).json()["project_id"]
    r = client.post(f"/api/projects/{pid}/rename", json={"name": "New Name"})
    assert r.status_code == 200
    body = r.json()
    assert body == {"status": "ok", "rev": 2, "name": "New Name"}
    got = client.get(f"/api/projects/{pid}").json()
    assert got["name"] == "New Name" and got["rev"] == 2


def test_rename_unknown_404(client):
    r = client.post("/api/projects/nope/rename", json={"name": "X"})
    assert r.status_code == 404
    assert r.json()["error"] == "project_not_found"


def test_rename_requires_name(client):
    pid = client.post("/api/projects", json={"name": "R"}).json()["project_id"]
    r = client.post(f"/api/projects/{pid}/rename", json={"name": "   "})
    assert r.status_code == 400
    assert r.json()["error"] == "bad_request"


def test_duplicate_defaults_name_and_copies_cache_not_mix(client):
    cfg = client.app.state.cfg
    pid = client.post("/api/projects", json={"name": "Orig"}).json()["project_id"]
    # Seed a render tree: a content-keyed cache seg (copied) plus mix/meta/exports/backup
    # (all intentionally NOT copied).
    cache = library.render_cache_dir(cfg, pid)
    os.makedirs(cache, exist_ok=True)
    open(os.path.join(cache, "seg_abc123.wav"), "wb").write(b"RIFF")
    rdir = library.render_dir(cfg, pid)
    open(os.path.join(rdir, "mix.wav"), "wb").write(b"x")
    open(os.path.join(rdir, "meta.json"), "w").write("{}")
    exp = library.exports_dir(cfg, pid)
    os.makedirs(exp, exist_ok=True)
    open(os.path.join(exp, "a.wav"), "wb").write(b"x")
    bdir = library.backup_dir(cfg, pid)
    os.makedirs(bdir, exist_ok=True)
    open(os.path.join(bdir, "rev_00001.json"), "w").write("{}")

    r = client.post(f"/api/projects/{pid}/duplicate")
    assert r.status_code == 201
    dup = r.json()
    new_id = dup["project_id"]
    assert new_id != pid and dup["rev"] == 1 and dup["name"] == "Orig copy"
    # cache seg copied ...
    assert os.path.isfile(os.path.join(library.render_cache_dir(cfg, new_id), "seg_abc123.wav"))
    # ... but the stitched mix / meta / exports / backups were not.
    assert not os.path.exists(os.path.join(library.render_dir(cfg, new_id), "mix.wav"))
    assert not os.path.exists(os.path.join(library.render_dir(cfg, new_id), "meta.json"))
    assert not os.path.exists(library.exports_dir(cfg, new_id))
    assert not os.path.exists(library.backup_dir(cfg, new_id))


def test_duplicate_custom_name(client):
    pid = client.post("/api/projects", json={"name": "Base"}).json()["project_id"]
    dup = client.post(f"/api/projects/{pid}/duplicate", json={"name": "My Fork"}).json()
    assert dup["name"] == "My Fork" and dup["rev"] == 1


def test_duplicate_unknown_404(client):
    r = client.post("/api/projects/nope/duplicate")
    assert r.status_code == 404


def test_delete_removes_project(client):
    cfg = client.app.state.cfg
    pid = client.post("/api/projects", json={"name": "Gone"}).json()["project_id"]
    assert os.path.isdir(library.project_dir(cfg, pid))
    r = client.delete(f"/api/projects/{pid}")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert not os.path.exists(library.project_dir(cfg, pid))
    assert client.get(f"/api/projects/{pid}").status_code == 404


def test_delete_unknown_404(client):
    r = client.delete("/api/projects/nope")
    assert r.status_code == 404


def test_delete_conflict_while_job_running(client):
    pid = client.post("/api/projects", json={"name": "Busy"}).json()["project_id"]
    # The api reads the JobManager off app.state.jobs; shadow is_running to force a 409.
    client.app.state.jobs.is_running = lambda p: True
    r = client.delete(f"/api/projects/{pid}")
    assert r.status_code == 409
    assert r.json()["error"] == "job_running"


def test_create_requires_name(client):
    r = client.post("/api/projects", json={})
    assert r.status_code == 400
    assert r.json()["error"] == "bad_request"


def test_create_unique_ids_on_collision(client):
    a = client.post("/api/projects", json={"name": "Dup"}).json()
    b = client.post("/api/projects", json={"name": "Dup"}).json()
    assert a["project_id"] != b["project_id"]


def test_get_unknown_project_404(client):
    r = client.get("/api/projects/nope")
    assert r.status_code == 404
    assert r.json()["error"] == "project_not_found"


def test_put_rev_conflict(client):
    doc = client.post("/api/projects", json={"name": "Edit"}).json()
    pid = doc["project_id"]
    # Correct base rev -> ok.
    ok = client.put(f"/api/projects/{pid}", json=doc)
    assert ok.status_code == 200 and ok.json()["rev"] == 2
    # Re-put the stale rev-1 doc -> 409 with server_rev.
    conflict = client.put(f"/api/projects/{pid}", json=doc)
    assert conflict.status_code == 409
    body = conflict.json()
    assert body["error"] == "rev_conflict" and body["server_rev"] == 2


def test_put_requires_notes_field(client):
    doc = client.post("/api/projects", json={"name": "NoNotes"}).json()
    pid = doc["project_id"]
    r = client.put(f"/api/projects/{pid}", json={"rev": 1})
    assert r.status_code == 400


def test_render_bad_prior_mode(client):
    # Validation happens before any job is submitted (no torch touched).
    doc = client.post("/api/projects", json={"name": "R"}).json()
    pid = doc["project_id"]
    r = client.post(f"/api/projects/{pid}/render", json={"prior_mode": "nope"})
    assert r.status_code == 400
    assert r.json()["error"] == "bad_request"


def test_render_unknown_checkpoint(client):
    doc = client.post("/api/projects", json={"name": "R2"}).json()
    pid = doc["project_id"]
    r = client.post(f"/api/projects/{pid}/render",
                    json={"model": "7-9-adr", "checkpoint": "nope.pt"})
    assert r.status_code == 400
    assert r.json()["error"] == "model_not_found"


def test_render_selection_requires_note_ids(client):
    doc = client.post("/api/projects", json={"name": "R3"}).json()
    pid = doc["project_id"]
    r = client.post(f"/api/projects/{pid}/render", json={"scope": "selection"})
    assert r.status_code == 400


def test_render_meta_not_rendered(client):
    doc = client.post("/api/projects", json={"name": "R4"}).json()
    pid = doc["project_id"]
    r = client.get(f"/api/projects/{pid}/render/meta")
    assert r.status_code == 404
    assert r.json()["error"] == "not_rendered"


def test_render_status_idle(client):
    doc = client.post("/api/projects", json={"name": "S"}).json()
    pid = doc["project_id"]
    r = client.get(f"/api/projects/{pid}/render/status")
    assert r.status_code == 200
    assert r.json()["state"] == "idle"


def test_render_status_unknown_project_404(client):
    r = client.get("/api/projects/nope/render/status")
    assert r.status_code == 404
