"""import-midi + export endpoints over a TestClient (torch-free).

Import uploads raw MIDI bytes (python-multipart is not installed); export writes a
file under ``exports/`` served by the ``/media`` mount. The WAV export path is faked
fresh by planting a matching render manifest + cache/mix files (no GPU).
"""

from __future__ import annotations

import dataclasses
import io

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from Studio import project_store
from Studio.cache import plan_render
from Studio.config import load_config
from Studio.library import (atomic_write_json, render_cache_dir, render_dir,
                            render_meta_file)
from Studio.project_store import build_note
from Studio.server import create_app


@pytest.fixture()
def client(tmp_path):
    projects = tmp_path / "projects"
    models = tmp_path / "models"
    d = models / "7-9-adr"
    d.mkdir(parents=True)
    (d / "checkpoint_final.pt").write_bytes(b"")
    cfg = dataclasses.replace(load_config(), projects_root=str(projects),
                              models_root=str(models))
    return TestClient(create_app(cfg))


def _new_project(client, name):
    return client.post("/api/projects", json={"name": name}).json()["project_id"]


def _midi_bytes(pitches=(60, 62), bpm=120.0):
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm, resolution=480)
    inst = pretty_midi.Instrument(program=40)
    spb = 60.0 / bpm
    for i, p in enumerate(pitches):
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=p,
                                           start=i * spb, end=(i + 1) * spb))
    pm.instruments.append(inst)
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()


# ---------- import-midi ----------

def test_import_replace(client):
    pid = _new_project(client, "Imp")
    r = client.post(f"/api/projects/{pid}/import-midi",
                    content=_midi_bytes((60, 62, 64)),
                    headers={"content-type": "application/octet-stream"})
    assert r.status_code == 200
    body = r.json()
    assert body["notes_added"] == 3
    assert len(body["doc"]["notes"]) == 3
    assert body["doc"]["rev"] == 2
    assert body["file_bpm"] == pytest.approx(120.0)
    # persisted
    got = client.get(f"/api/projects/{pid}").json()
    assert len(got["notes"]) == 3


def test_import_merge_appends(client):
    pid = _new_project(client, "Merge")
    client.post(f"/api/projects/{pid}/import-midi", content=_midi_bytes((60, 62)),
                headers={"content-type": "application/octet-stream"})
    r = client.post(f"/api/projects/{pid}/import-midi", content=_midi_bytes((67,)),
                    params={"mode": "merge"},
                    headers={"content-type": "application/octet-stream"})
    assert r.status_code == 200
    body = r.json()
    assert body["notes_added"] == 1
    assert len(body["doc"]["notes"]) == 3
    ids = [n["id"] for n in body["doc"]["notes"]]
    assert len(set(ids)) == 3          # fresh, non-colliding ids


def test_import_adopt_bpm(client):
    pid = _new_project(client, "Bpm")
    r = client.post(f"/api/projects/{pid}/import-midi", content=_midi_bytes(bpm=90.0),
                    params={"adopt_bpm": "true"},
                    headers={"content-type": "application/octet-stream"})
    assert r.status_code == 200
    assert r.json()["doc"]["bpm"] == pytest.approx(90.0, abs=0.01)


def test_import_bad_midi_400(client):
    pid = _new_project(client, "Bad")
    r = client.post(f"/api/projects/{pid}/import-midi", content=b"garbage",
                    headers={"content-type": "application/octet-stream"})
    assert r.status_code == 400
    assert r.json()["error"] == "bad_midi"


def test_import_empty_body_400(client):
    pid = _new_project(client, "Empty")
    r = client.post(f"/api/projects/{pid}/import-midi", content=b"",
                    headers={"content-type": "application/octet-stream"})
    assert r.status_code == 400
    assert r.json()["error"] == "bad_midi"


def test_import_bad_mode_400(client):
    pid = _new_project(client, "Mode")
    r = client.post(f"/api/projects/{pid}/import-midi", content=_midi_bytes(),
                    params={"mode": "nope"},
                    headers={"content-type": "application/octet-stream"})
    assert r.status_code == 400
    assert r.json()["error"] == "bad_request"


# ---------- export MIDI ----------

def test_export_midi_writes_and_serves(client):
    pid = _new_project(client, "ExpMidi")
    client.post(f"/api/projects/{pid}/import-midi", content=_midi_bytes((60, 62)),
                headers={"content-type": "application/octet-stream"})
    r = client.post(f"/api/projects/{pid}/export", json={"kind": "midi"})
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "midi"
    assert body["path"].startswith(f"/media/{pid}/exports/")
    assert body["path"].endswith(".mid")
    # the /media mount serves the written file
    served = client.get(body["path"])
    assert served.status_code == 200
    assert served.content[:4] == b"MThd"     # MIDI header chunk


def test_export_bad_kind_400(client):
    pid = _new_project(client, "BadKind")
    r = client.post(f"/api/projects/{pid}/export", json={"kind": "flac"})
    assert r.status_code == 400
    assert r.json()["error"] == "bad_request"


# ---------- export WAV ----------

def test_export_wav_stale_409(client):
    pid = _new_project(client, "WavStale")
    client.post(f"/api/projects/{pid}/import-midi", content=_midi_bytes((60,)),
                headers={"content-type": "application/octet-stream"})
    r = client.post(f"/api/projects/{pid}/export", json={"kind": "wav"})
    assert r.status_code == 409
    assert r.json()["error"] == "render_stale"


def _plant_fresh_render(cfg, pid, vocoder=None):
    """Fake a fresh render: cache seg WAVs + mix.wav + a matching meta.json.

    ``vocoder`` is the render's vocoder name; ``None`` plants a *legacy* meta with
    no ``vocoder`` key at all (which must still read as fresh, since such renders
    used the config default).
    """
    doc = project_store.load(cfg, pid)
    plan = plan_render(doc, cfg, cfg.default_model, cfg.default_checkpoint,
                       cfg.default_prior_mode, vocoder=vocoder,
                       cache_dir=render_cache_dir(cfg, pid))
    cdir = render_cache_dir(cfg, pid)
    import os
    os.makedirs(cdir, exist_ok=True)
    tone = 0.1 * np.sin(2 * np.pi * 220 * np.arange(4410) / 44100).astype(np.float32)
    for s in plan["segments"]:
        sf.write(os.path.join(cdir, s["wav"]), tone, 44100, subtype="PCM_16")
    rdir = render_dir(cfg, pid)
    os.makedirs(rdir, exist_ok=True)
    sf.write(os.path.join(rdir, "mix.wav"), tone, 44100, subtype="PCM_16")
    meta = {
        "wav": f"/media/{pid}/render/mix.wav", "duration_s": 0.1,
        "model": cfg.default_model, "checkpoint": cfg.default_checkpoint,
        "prior_mode": cfg.default_prior_mode,
        "segments": [{"hash": s["hash"]} for s in plan["segments"]],
        "warnings": [],
    }
    if vocoder is not None:
        meta["vocoder"] = vocoder
    atomic_write_json(render_meta_file(cfg, pid), meta)


def test_export_wav_success_when_fresh(client):
    cfg = client.app.state.cfg
    pid = _new_project(client, "WavFresh")
    client.post(f"/api/projects/{pid}/import-midi", content=_midi_bytes((60, 62)),
                headers={"content-type": "application/octet-stream"})
    _plant_fresh_render(cfg, pid)
    r = client.post(f"/api/projects/{pid}/export", json={"kind": "wav"})
    assert r.status_code == 200, r.json()
    body = r.json()
    assert body["kind"] == "wav"
    assert body["path"].startswith(f"/media/{pid}/exports/")
    assert body["path"].endswith(".wav")
    served = client.get(body["path"])
    assert served.status_code == 200
    assert served.content[:4] == b"RIFF"


def test_export_wav_fresh_with_non_default_vocoder(client):
    # A render made with a non-default vocoder must re-plan under *its* vocoder,
    # or the export would spuriously 409 render_stale.
    cfg = client.app.state.cfg
    pid = _new_project(client, "WavVoc")
    client.post(f"/api/projects/{pid}/import-midi", content=_midi_bytes((60, 62)),
                headers={"content-type": "application/octet-stream"})
    _plant_fresh_render(cfg, pid, vocoder="hf")
    assert cfg.default_vocoder != "hf"            # genuinely a different identity
    r = client.post(f"/api/projects/{pid}/export", json={"kind": "wav"})
    assert r.status_code == 200, r.json()


def test_export_wav_fresh_with_legacy_meta_without_vocoder(client):
    # Metas written before per-render vocoder selection carry no "vocoder" key;
    # they used the config default, so the fallback keeps them hashing identically.
    cfg = client.app.state.cfg
    pid = _new_project(client, "WavLegacy")
    client.post(f"/api/projects/{pid}/import-midi", content=_midi_bytes((60,)),
                headers={"content-type": "application/octet-stream"})
    _plant_fresh_render(cfg, pid)                 # no "vocoder" key planted
    from Studio.library import read_json
    assert "vocoder" not in read_json(render_meta_file(cfg, pid))
    r = client.post(f"/api/projects/{pid}/export", json={"kind": "wav"})
    assert r.status_code == 200, r.json()


def test_export_wav_stale_after_edit(client):
    cfg = client.app.state.cfg
    pid = _new_project(client, "WavEdit")
    client.post(f"/api/projects/{pid}/import-midi", content=_midi_bytes((60,)),
                headers={"content-type": "application/octet-stream"})
    _plant_fresh_render(cfg, pid)
    # edit a note (changes its segment hash) -> render now stale
    doc = project_store.load(cfg, pid)
    doc["notes"].append(build_note(99, start_beat=8.0, len_beats=1.0, pitch=72))
    project_store.commit_rebuild(cfg, pid, doc, doc)
    r = client.post(f"/api/projects/{pid}/export", json={"kind": "wav"})
    assert r.status_code == 409
    assert r.json()["error"] == "render_stale"
