"""Segment cache: segmentation, content hashing, plan/cache-hit, stitch, GC.

All torch-free. The end-to-end render test monkeypatches out the GPU pipeline
(:func:`Studio.render._render_segments`) and asserts exactly the non-cached
segments get rendered.
"""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf

from common.audio_io import write_pcm16
from common.config import DEFAULT_PEAK, SR
from Studio.cache import (_vocoder_cache_id, plan_render, prune_cache,
                          seg_wav_name, segment_hash, segment_notes, stitch)
from Studio.config import load_config

# 120 bpm -> 1 quarter-note beat = 0.5 s (keeps the seconds arithmetic obvious).
BPM = 120.0
TIME_SIG = {"num": 4, "den": 4}
KNOBS = {"gap_s": 0.35, "pad_s": 0.1}


def _note(nid, start, length, pitch=67, velocity=100, technique="normal",
          bend=None, vibrato=None):
    return {"id": nid, "start_beat": float(start), "len_beats": float(length),
            "pitch": pitch, "velocity": velocity, "technique": technique,
            "bend": bend or [], "vibrato": vibrato or {"depth_semitones": 0.0,
            "rate_hz": 5.5, "onset_beats": 0.15}}


def _hash(notes, **over):
    p = {"bpm": BPM, "time_sig": TIME_SIG, "model": "m", "checkpoint": "c",
         "prior_mode": "bend", "schema_version": 1, "knobs": KNOBS,
         "vocoder": None}
    p.update(over)
    return segment_hash(notes, p["bpm"], p["time_sig"], p["model"], p["checkpoint"],
                        p["prior_mode"], p["schema_version"], p["knobs"],
                        vocoder=p["vocoder"])


# --- segmentation -----------------------------------------------------------

def test_single_run_no_gap():
    # Three back-to-back notes (no gap) -> one segment.
    notes = [_note("n0", 0, 1), _note("n1", 1, 1), _note("n2", 2, 1)]
    segs = segment_notes(notes, BPM, 0.35, 0.1)
    assert len(segs) == 1
    assert [n["id"] for n in segs[0]["notes"]] == ["n0", "n1", "n2"]


def test_gap_splits_into_two_segments():
    # n1 ends at 1.0 s, n2 starts at 1.5 s -> 0.5 s gap >= 0.35 -> split.
    notes = [_note("n0", 0, 1), _note("n1", 1, 1), _note("n2", 3, 1)]
    segs = segment_notes(notes, BPM, 0.35, 0.1)
    assert len(segs) == 2
    assert [n["id"] for n in segs[0]["notes"]] == ["n0", "n1"]
    assert [n["id"] for n in segs[1]["notes"]] == ["n2"]
    assert segs[0]["start_s"] == 0.0 and segs[0]["end_s"] == 1.0
    assert segs[1]["start_s"] == 1.5 and segs[1]["end_s"] == 2.0


def test_gap_below_threshold_stays_together():
    # 0.25 s gap (< 0.35) -> single segment.
    notes = [_note("n0", 0, 1), _note("n1", 1.5, 1)]  # 0..0.5, 0.75..1.25
    segs = segment_notes(notes, BPM, 0.35, 0.1)
    assert len(segs) == 1


def test_first_lead_pad_never_before_zero():
    # A note starting at t=0 cannot pad its lead into negative time.
    segs = segment_notes([_note("n0", 0, 1)], BPM, 0.35, 0.1)
    assert segs[0]["pad_lead"] == 0.0
    assert segs[0]["pad_tail"] == 0.1   # last segment: full pad into trailing silence


def test_internal_pads_clamped_to_half_gap():
    # Force a clamp: pad_s (0.3) > gap/2. n0 ends 0.5 s, n1 starts 1.0 s -> gap 0.5 s
    # -> half = 0.25 -> both pads clamp to 0.25.
    notes = [_note("n0", 0, 1), _note("n1", 2, 1)]   # 0..0.5 s, then 1.0..1.5 s
    segs = segment_notes(notes, BPM, 0.35, 0.3)
    assert abs(segs[0]["pad_tail"] - 0.25) < 1e-9
    assert abs(segs[1]["pad_lead"] - 0.25) < 1e-9
    # The two pads meet but never cross the gap midpoint.
    assert segs[0]["pad_tail"] + segs[1]["pad_lead"] <= 0.5 + 1e-9


def test_polyphony_stays_in_one_segment():
    # Overlapping notes (a chord) belong to the same segment.
    notes = [_note("n0", 0, 2, pitch=60), _note("n1", 0, 2, pitch=64)]
    segs = segment_notes(notes, BPM, 0.35, 0.1)
    assert len(segs) == 1
    assert len(segs[0]["notes"]) == 2


# --- content hashing --------------------------------------------------------

def test_same_content_same_hash():
    a = [_note("n0", 0, 1, pitch=60), _note("n1", 1, 1, pitch=62)]
    b = [_note("x9", 0, 1, pitch=60), _note("x8", 1, 1, pitch=62)]  # ids differ only
    assert _hash(a) == _hash(b)


def test_whole_segment_shift_same_hash():
    # Moving the entire segment in time (segment-relative hashing) -> same hash.
    a = [_note("n0", 0, 1, pitch=60), _note("n1", 1, 1, pitch=62)]
    shifted = [_note("n0", 10, 1, pitch=60), _note("n1", 11, 1, pitch=62)]
    assert _hash(a) == _hash(shifted)


def test_edit_one_note_changes_hash():
    a = [_note("n0", 0, 1, pitch=60), _note("n1", 1, 1, pitch=62)]
    edited = [_note("n0", 0, 1, pitch=60), _note("n1", 1, 1, pitch=63)]  # pitch change
    assert _hash(a) != _hash(edited)


def test_param_changes_change_hash():
    a = [_note("n0", 0, 1, pitch=60)]
    assert _hash(a) != _hash(a, bpm=121.0)
    assert _hash(a) != _hash(a, model="other")
    assert _hash(a) != _hash(a, prior_mode="quantized")
    assert _hash(a) != _hash(a, knobs={"gap_s": 0.4, "pad_s": 0.1})


def test_expression_fields_change_hash():
    flat = [_note("n0", 0, 1)]
    bent = [_note("n0", 0, 1, bend=[{"beat": 0.0, "semitones": 0.0},
                                    {"beat": 1.0, "semitones": 2.0}])]
    vib = [_note("n0", 0, 1, vibrato={"depth_semitones": 0.5, "rate_hz": 5.5,
                                      "onset_beats": 0.1})]
    assert _hash(flat) != _hash(bent)
    assert _hash(flat) != _hash(vib)


# --- vocoder identity + its effect on the hash -------------------------------

def _stub_vocoder_dir(root, name, nbytes):
    """A loadable-looking BigVGAN run dir (config.json + sized generator weights)."""
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "bigvgan_generator.pt").write_bytes(b"\0" * nbytes)
    return str(d)


def test_vocoder_cache_id_stock_is_none():
    # Both "no selection" and the explicit stock sentinel key as "no vocoder".
    assert _vocoder_cache_id(None) is None
    assert _vocoder_cache_id("hf") is None


def test_vocoder_cache_id_local_dir_is_name_and_size(tmp_path):
    d = _stub_vocoder_dir(tmp_path, "ft_v2", 123)
    assert _vocoder_cache_id(d) == "ft_v2:123"
    # A trailing separator must not change the identity (basename of the normpath).
    assert _vocoder_cache_id(d + os.sep) == "ft_v2:123"


def test_vocoder_cache_id_matches_the_module_global_era(tmp_path):
    """Hash-compat: the config-default path still yields the historical id.

    Before per-render selection the id came from ``common.config.VOCODER_DIR``
    directly; now it is derived from ``cfg.default_vocoder`` -> ``vocoder_checkpoint_dir``.
    Both must produce the same ``<name>:<size>`` string or every cached segment in
    every existing project would go stale.
    """
    from common.config import VOCODER_DIR
    from Studio.library import vocoder_checkpoint_dir

    cfg = load_config()
    old_style = _vocoder_cache_id(VOCODER_DIR)
    new_style = _vocoder_cache_id(vocoder_checkpoint_dir(cfg, cfg.default_vocoder))
    assert new_style == old_style


def test_hash_without_vocoder_matches_historical_payload():
    # vocoder=None omits the key entirely -> byte-identical to pre-selection hashes.
    notes = [_note("n0", 0, 1)]
    payload = {"notes": [{"pitch": 67, "start_beat": 0.0, "len_beats": 1.0,
                          "velocity": 100, "technique": "normal", "bend": [],
                          "vibrato": {"depth_semitones": 0.0, "rate_hz": 5.5,
                                      "onset_beats": 0.15},
                          "pan": 0.0}],
               "bpm": 120.0, "time_sig": {"num": 4, "den": 4}, "model": "m",
               "checkpoint": "c", "prior_mode": "bend", "schema_version": 1,
               "knobs": {"gap_s": 0.35, "pad_s": 0.1}}
    import hashlib
    import json
    expected = hashlib.sha1(json.dumps(payload, sort_keys=True,
                                       separators=(",", ":")).encode("utf-8")).hexdigest()
    assert _hash(notes) == expected
    assert _hash(notes, vocoder=None) == expected


def test_different_vocoder_ids_change_hash():
    notes = [_note("n0", 0, 1)]
    assert _hash(notes, vocoder="ft_v2:10") != _hash(notes, vocoder="ft_v3:10")
    assert _hash(notes, vocoder="ft_v2:10") != _hash(notes, vocoder="ft_v2:11")
    assert _hash(notes, vocoder="ft_v2:10") != _hash(notes, vocoder=None)


# --- plan_render cache-hit detection ----------------------------------------

def _doc(notes):
    return {"project_id": "p", "schema_version": 1, "bpm": BPM,
            "time_sig": TIME_SIG, "notes": notes}


def test_plan_marks_cache_hits(tmp_path):
    cfg = load_config()
    cache_dir = str(tmp_path / "cache")
    os.makedirs(cache_dir)
    doc = _doc([_note("n0", 0, 1), _note("n1", 1, 1), _note("n2", 3, 1)])
    plan = plan_render(doc, cfg, "m", "c", "bend", cache_dir=cache_dir)
    segs = plan["segments"]
    assert len(segs) == 2
    assert all(not s["cached"] for s in segs)

    # Touch the first segment's cache wav -> only it reports cached next plan.
    open(os.path.join(cache_dir, segs[0]["wav"]), "wb").close()
    plan2 = plan_render(doc, cfg, "m", "c", "bend", cache_dir=cache_dir)
    assert plan2["segments"][0]["cached"] is True
    assert plan2["segments"][1]["cached"] is False


def test_plan_vocoder_selection_changes_hashes(tmp_path):
    import dataclasses
    vroot = tmp_path / "vocoders"
    _stub_vocoder_dir(vroot, "A", 16)
    _stub_vocoder_dir(vroot, "B", 32)
    cfg = dataclasses.replace(load_config(), vocoders_root=str(vroot),
                              default_vocoder="A")
    doc = _doc([_note("n0", 0, 1)])
    cache_dir = str(tmp_path / "cache")

    a = plan_render(doc, cfg, "m", "c", "bend", vocoder="A", cache_dir=cache_dir)
    b = plan_render(doc, cfg, "m", "c", "bend", vocoder="B", cache_dir=cache_dir)
    assert a["segments"][0]["hash"] != b["segments"][0]["hash"]
    # Omitting the arg falls back to cfg.default_vocoder.
    dflt = plan_render(doc, cfg, "m", "c", "bend", cache_dir=cache_dir)
    assert dflt["segments"][0]["hash"] == a["segments"][0]["hash"]


def test_plan_hf_vocoder_hashes_like_no_vocoder_key(tmp_path):
    import dataclasses
    cfg = dataclasses.replace(load_config(), vocoders_root=str(tmp_path / "none"),
                              default_vocoder="hf")
    doc = _doc([_note("n0", 0, 1)])
    plan = plan_render(doc, cfg, "m", "c", "bend", vocoder="hf",
                       cache_dir=str(tmp_path))
    assert plan["segments"][0]["hash"] == _hash(
        doc["notes"], model="m", checkpoint="c", schema_version=1,
        knobs={"gap_s": cfg.cache.gap_s, "pad_s": cfg.cache.pad_s}, vocoder=None)


def test_plan_total_duration_covers_tail(tmp_path):
    cfg = load_config()
    doc = _doc([_note("n0", 0, 1)])   # 0..0.5 s + 0.1 s tail pad
    plan = plan_render(doc, cfg, "m", "c", "bend", cache_dir=str(tmp_path))
    assert abs(plan["total_duration_s"] - 0.6) < 1e-9


# --- stitch -----------------------------------------------------------------

def _write_sine(path, dur_s, freq=440.0, amp=0.5):
    n = int(round(dur_s * SR))
    t = np.arange(n) / SR
    write_pcm16(path, (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32))


def test_stitch_places_segments_with_silent_gap(tmp_path):
    cache_dir = str(tmp_path)
    # seg A at 0..0.5 s, seg B at 1.5..2.0 s -> silence in 0.5..1.5 s.
    wa, wb = seg_wav_name("a"), seg_wav_name("b")
    _write_sine(os.path.join(cache_dir, wa), 0.5)
    _write_sine(os.path.join(cache_dir, wb), 0.5)
    segments = [
        {"wav": wa, "start_s": 0.0, "end_s": 0.5, "pad_lead": 0.0, "pad_tail": 0.0},
        {"wav": wb, "start_s": 1.5, "end_s": 2.0, "pad_lead": 0.0, "pad_tail": 0.0},
    ]
    mix = stitch(segments, cache_dir, 2.0, SR)
    assert len(mix) == int(round(2.0 * SR))
    # Energy present in each segment window, silence in the gap.
    seg_a = mix[:int(0.5 * SR)]
    gap = mix[int(0.6 * SR):int(1.4 * SR)]
    seg_b = mix[int(1.5 * SR):int(2.0 * SR)]
    assert np.max(np.abs(seg_a)) > 0.1
    assert np.max(np.abs(seg_b)) > 0.1
    assert np.max(np.abs(gap)) < 1e-6


def test_stitch_honors_lead_pad_offset(tmp_path):
    cache_dir = str(tmp_path)
    w = seg_wav_name("a")
    # A segment whose content starts at 0.5 s but wav includes a 0.1 s lead pad.
    _write_sine(os.path.join(cache_dir, w), 0.5)   # wav is 0.5 s long
    segments = [{"wav": w, "start_s": 0.5, "end_s": 0.9, "pad_lead": 0.1,
                 "pad_tail": 0.0}]
    mix = stitch(segments, cache_dir, 1.5, SR)
    # Placed at start_s - pad_lead = 0.4 s.
    assert np.max(np.abs(mix[:int(0.35 * SR)])) < 1e-6
    assert np.max(np.abs(mix[int(0.45 * SR):int(0.85 * SR)])) > 0.1


def test_stitch_peak_safety_no_clip(tmp_path):
    cache_dir = str(tmp_path)
    w = seg_wav_name("a")
    _write_sine(os.path.join(cache_dir, w), 0.5, amp=0.99)
    segments = [{"wav": w, "start_s": 0.0, "end_s": 0.5, "pad_lead": 0.0,
                 "pad_tail": 0.0}]
    mix = stitch(segments, cache_dir, 0.5, SR)
    assert np.max(np.abs(mix)) <= DEFAULT_PEAK + 1e-6


def test_stitch_crossfade_overlap_no_clip(tmp_path):
    cache_dir = str(tmp_path)
    wa, wb = seg_wav_name("a"), seg_wav_name("b")
    _write_sine(os.path.join(cache_dir, wa), 0.5, amp=0.9)
    _write_sine(os.path.join(cache_dir, wb), 0.5, amp=0.9)
    # Overlapping placement (B starts inside A) -> crossfade must not sum to clip.
    segments = [
        {"wav": wa, "start_s": 0.0, "end_s": 0.5, "pad_lead": 0.0, "pad_tail": 0.0},
        {"wav": wb, "start_s": 0.3, "end_s": 0.8, "pad_lead": 0.0, "pad_tail": 0.0},
    ]
    mix = stitch(segments, cache_dir, 0.8, SR)
    assert np.max(np.abs(mix)) <= DEFAULT_PEAK + 1e-6


# --- garbage collection -----------------------------------------------------

def test_prune_keeps_manifest_and_caps_stale(tmp_path):
    cache_dir = str(tmp_path)
    keep = [f"k{i}" for i in range(3)]
    stale = [f"s{i}" for i in range(10)]
    for h in keep + stale:
        open(os.path.join(cache_dir, seg_wav_name(h)), "wb").close()
    # Budget 5: keep the 3 manifest files always + 2 newest stale, drop 8 stale.
    deleted = prune_cache(cache_dir, keep, max_files=5)
    remaining = {f for f in os.listdir(cache_dir) if f.startswith("seg_")}
    for h in keep:
        assert seg_wav_name(h) in remaining
    assert len(deleted) == 8
    assert len(remaining) == 5


def test_prune_keeps_all_manifest_even_over_budget(tmp_path):
    cache_dir = str(tmp_path)
    keep = [f"k{i}" for i in range(6)]
    for h in keep:
        open(os.path.join(cache_dir, seg_wav_name(h)), "wb").close()
    deleted = prune_cache(cache_dir, keep, max_files=3)
    assert deleted == []                       # manifest files are never deleted
    assert len(os.listdir(cache_dir)) == 6


def test_prune_missing_dir_is_noop(tmp_path):
    assert prune_cache(str(tmp_path / "nope"), [], max_files=5) == []


def _write_sized(path, nbytes, mtime):
    with open(path, "wb") as f:
        f.write(b"\0" * nbytes)
    os.utime(path, (mtime, mtime))


def test_prune_max_mb_deletes_oldest_stale_first(tmp_path):
    cache_dir = str(tmp_path)
    # Four 100 KB stale WAVs, ascending mtime (s0 oldest ... s3 newest).
    for i in range(4):
        _write_sized(os.path.join(cache_dir, seg_wav_name(f"s{i}")),
                     100_000, mtime=1_000_000.0 + i)
    # max_files high (count pass is a no-op); 0.25 MB budget keeps the 2 newest.
    deleted = prune_cache(cache_dir, [], max_files=64, max_mb=0.25)
    assert set(deleted) == {seg_wav_name("s0"), seg_wav_name("s1")}   # oldest first
    remaining = {f for f in os.listdir(cache_dir) if f.startswith("seg_")}
    assert remaining == {seg_wav_name("s2"), seg_wav_name("s3")}


def test_prune_max_mb_never_deletes_manifest_over_budget(tmp_path):
    cache_dir = str(tmp_path)
    keep = [f"k{i}" for i in range(3)]
    # Three 100 KB manifest WAVs (300 KB) against a 50 KB budget: all must survive.
    for i, h in enumerate(keep):
        _write_sized(os.path.join(cache_dir, seg_wav_name(h)),
                     100_000, mtime=1_000_000.0 + i)
    deleted = prune_cache(cache_dir, keep, max_files=64, max_mb=0.05)
    assert deleted == []
    assert len(os.listdir(cache_dir)) == 3
