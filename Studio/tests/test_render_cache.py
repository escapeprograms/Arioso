"""run_render segment-cache behaviour with the GPU pipeline monkeypatched out.

Replaces :func:`Studio.render._render_segments` with a fake that writes a dummy
segment WAV per non-cached segment and records which hashes it was asked to render,
so we can assert: a first render renders every segment, a no-op re-render renders
none (all cache hits), and editing one note in one phrase re-renders exactly that
phrase's segment while the other stays a cache hit. Torch-free.
"""

from __future__ import annotations

import os

import numpy as np

import Studio.render as render_mod
from common.audio_io import write_pcm16
from common.config import SR
from Studio.config import load_config
from Studio.library import RENDER_DIR, SEG_CACHE_DIR, render_meta_file
from Studio.project_store import create_project, put_with_rev


class FakeRenderer:
    """Stand-in for ``_render_segments`` that writes dummy WAVs and logs calls."""

    def __init__(self):
        self.calls: list[list[str]] = []      # hashes rendered per invocation

    def __call__(self, cfg, doc, to_render, cache_dir, model, checkpoint,
                 prior_mode, device, progress):
        self.calls.append([s["hash"] for s in to_render])
        for seg in to_render:
            span = (seg["end_s"] + seg["pad_tail"]) - (seg["start_s"] - seg["pad_lead"])
            n = max(1, int(round(span * SR)))
            t = np.arange(n) / SR
            wav = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
            write_pcm16(os.path.join(cache_dir, seg["wav"]), wav)
        return "cpu", []

    @property
    def last(self) -> list[str]:
        return self.calls[-1]


def _two_phrase_notes():
    # Two phrases separated by a > 0.35 s silence gap at 120 bpm (0.5 s/beat):
    # phrase 1 = beats 0..2 (0..1.0 s), phrase 2 = beats 4..6 (2.0..3.0 s) -> 1.0 s gap.
    return [
        {"start_beat": 0.0, "len_beats": 1.0, "pitch": 60},
        {"start_beat": 1.0, "len_beats": 1.0, "pitch": 62},
        {"start_beat": 4.0, "len_beats": 1.0, "pitch": 64},
        {"start_beat": 5.0, "len_beats": 1.0, "pitch": 65},
    ]


def _project(tmp_path):
    cfg = load_config()
    import dataclasses
    cfg = dataclasses.replace(cfg, projects_root=str(tmp_path / "projects"))
    doc = create_project(cfg, "song", name="Song")
    doc = dict(doc)
    doc["bpm"] = 120.0
    # Restamp notes with monotonic ids via put_with_rev + build_document shape.
    from Studio.project_store import build_note
    doc["notes"] = [build_note(i, start_beat=n["start_beat"], len_beats=n["len_beats"],
                               pitch=n["pitch"]) for i, n in enumerate(_two_phrase_notes())]
    doc["next_note_ordinal"] = len(doc["notes"])
    res = put_with_rev(cfg, "song", doc)
    return cfg, res["doc"]


def _pdir(cfg, doc):
    return os.path.join(cfg.projects_root, doc["project_id"])


def test_first_render_renders_all_then_cache_hits(tmp_path, monkeypatch):
    fake = FakeRenderer()
    monkeypatch.setattr(render_mod, "_render_segments", fake)
    cfg, doc = _project(tmp_path)
    pdir = _pdir(cfg, doc)

    meta = render_mod.run_render(cfg, doc, pdir)
    assert meta["segments_total"] == 2
    assert meta["segments_rendered"] == 2
    assert len(fake.calls) == 1 and len(fake.last) == 2

    # mix.wav + peaks + meta.json written.
    rdir = os.path.join(pdir, RENDER_DIR)
    assert os.path.isfile(os.path.join(rdir, "mix.wav"))
    assert os.path.isfile(os.path.join(rdir, "mix.peaks"))
    assert os.path.isfile(render_meta_file(cfg, doc["project_id"]))
    # Two cache WAVs on disk.
    cache = os.path.join(rdir, SEG_CACHE_DIR)
    assert len([f for f in os.listdir(cache) if f.endswith(".wav")]) == 2

    # Re-render, nothing changed -> zero segments rendered (all cache hits).
    meta2 = render_mod.run_render(cfg, doc, pdir)
    assert meta2["segments_rendered"] == 0
    assert meta2["segments_cached"] == 2
    assert len(fake.calls) == 1      # fake not called again


def test_edit_one_note_rerenders_only_its_segment(tmp_path, monkeypatch):
    fake = FakeRenderer()
    monkeypatch.setattr(render_mod, "_render_segments", fake)
    cfg, doc = _project(tmp_path)
    pdir = _pdir(cfg, doc)

    render_mod.run_render(cfg, doc, pdir)          # prime both segments
    plan_before = render_mod.plan_render(doc, cfg, cfg.default_model,
                                         cfg.default_checkpoint, cfg.default_prior_mode,
                                         cache_dir=os.path.join(pdir, RENDER_DIR, SEG_CACHE_DIR))
    phrase1_hash = plan_before["segments"][0]["hash"]

    # Edit a note in phrase 2 (the second segment).
    edited = dict(doc)
    edited["notes"] = [dict(n) for n in doc["notes"]]
    edited["notes"][2]["pitch"] = 67               # was 64
    res = put_with_rev(cfg, "song", edited)
    edited_doc = res["doc"]

    render_mod.run_render(cfg, edited_doc, pdir)
    # Exactly one segment rendered, and it is NOT phrase 1 (unchanged -> cache hit).
    assert len(fake.last) == 1
    assert phrase1_hash not in fake.last

    # Meta from the last render reports 1 rendered, 1 cached.
    from Studio.library import read_json
    m = read_json(render_meta_file(cfg, edited_doc["project_id"]))
    assert m["segments_rendered"] == 1
    assert m["segments_cached"] == 1


def test_empty_project_renders_silence(tmp_path, monkeypatch):
    fake = FakeRenderer()
    monkeypatch.setattr(render_mod, "_render_segments", fake)
    import dataclasses
    cfg = dataclasses.replace(load_config(), projects_root=str(tmp_path / "projects"))
    doc = create_project(cfg, "empty", name="Empty")
    pdir = _pdir(cfg, doc)
    meta = render_mod.run_render(cfg, doc, pdir)
    assert meta["segments_total"] == 0
    assert meta["segments_rendered"] == 0
    assert len(fake.calls) == 0
    assert meta["duration_s"] > 0.0               # a short silence, not a crash
    assert os.path.isfile(os.path.join(pdir, RENDER_DIR, "mix.wav"))
