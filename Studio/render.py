"""The offline render pipeline: project document -> ``mix.wav`` + peaks + meta.

Mirrors ``Arioso.infer`` end to end, driven by a Studio project instead of a MIDI
file on disk, and **segment-cached** (:mod:`Studio.cache`): the document is split
into silence-delimited segments, each rendered once to ``render/cache/seg_<hash>.wav``
and reused on later renders whose content did not change, then all segments are
stitched into ``mix.wav``. Editing one note re-renders only its segment.

Each non-cached segment runs the same five-stage pipeline the whole phrase used to,
reported to the polling UI through the ``progress(status)`` callback the
:class:`Studio.jobs.JobManager` hands in (now carrying ``segments_done`` /
``segments_total`` alongside the stage/pct):

  1. **serialize** — write the segment's (time-shifted) notes to a temp ``.mid``
     (:mod:`Studio.midi_export`: voices -> instruments, bend/vibrato -> pitch-wheel).
  2. **prior** — ``quantized_prior(pitch=...)`` -> ``synth.render(midi, total_samples)``
     (the segment span incl. pads, so the tail isn't clipped) -> ``mel_for_training``.
  3. **model** — lazy-load the checkpoint (:mod:`Studio.model_registry`) and run
     ``generate_mel`` (chunked CFM ODE, ``cond_frames=None`` — renders are always
     unconditioned for now).
  4. **vocoder** — the frozen BigVGAN-v2 vocoder -> float32 segment waveform.
  5. **write** — the segment WAV to the cache; then ``mix.wav`` (PCM16) + ``mix.peaks``
     + ``render/meta.json`` (with the segment manifest) after all segments exist.

Torch (and everything that transitively imports it) is imported **inside**
``run_render`` so importing this module — and the API above it — stays torch-free.
A module-level ``GPU_LOCK`` serializes the two GPU forwards (mel generation +
vocoding) across concurrently rendering projects, since the model/vocoder
singletons are not safe for concurrent forward passes (the ``Labeler.transcribe``
``GPU_LOCK`` pattern).

**Render bookkeeping lives in ``render/meta.json``, not ``project.json``.** The
frontend autosaves ``project.json`` under optimistic rev locking; writing render
results back into it from a background thread would race those saves and spuriously
bump the rev. Keeping render meta in its own server-owned file (like
``status.json``) sidesteps that entirely — the UI reads it via
``GET .../render/meta``.
"""

from __future__ import annotations

import os
import threading

import numpy as np

from common.config import SR

from .cache import plan_render, prune_cache, stitch
from .library import (MIX_PEAKS, MIX_WAV, RENDER_DIR, RENDER_META,
                      SEG_CACHE_DIR, atomic_write_json)
from .midi_export import write_midi
from .peaks import write_peaks
from .timing import seconds_to_beats

# Serializes GPU forwards (mel gen + vocode) across projects rendering at once.
GPU_LOCK = threading.Lock()

# Silence written for an empty project (no notes) so the mix.wav is a valid file.
_EMPTY_SILENCE_S = 0.05

# Cap on cached segment WAVs kept per project (GC drops the oldest stale ones).
_MAX_CACHE_FILES = 64


def _emit(progress, stage: str, pct: int, message: str = "",
          state: str = "processing", **extra) -> None:
    if progress is not None:
        st = {"state": state, "stage": stage, "pct": pct,
              "message": message, "error": None}
        st.update(extra)
        progress(st)


def _media_url(project_id: str, *parts: str) -> str:
    """URL under the ``/media`` StaticFiles mount for a per-project file."""
    return "/media/" + "/".join([project_id, *parts])


def _sub_doc(doc: dict, seg: dict) -> tuple[dict, int]:
    """Sub-document of one segment's notes shifted so its padded lead starts at t=0.

    Returns ``(sub_doc, total_samples)`` where notes are copies with ``start_beat``
    translated by the segment's origin (its first onset minus the lead pad, in beats)
    so the earliest note sits ``pad_lead`` seconds in, and ``total_samples`` is the
    full padded span (lead + content + tail) handed to ``PriorSynth.render`` so the
    trailing pad isn't clipped. Bend/vibrato are note-relative, so untouched.
    """
    bpm = float(doc["bpm"])
    origin_beat = seg["start_beat"] - seconds_to_beats(seg["pad_lead"], bpm)
    sub_notes = []
    for n in seg["notes"]:
        m = dict(n)
        m["start_beat"] = float(n["start_beat"]) - origin_beat
        sub_notes.append(m)
    sub = dict(doc)
    sub["notes"] = sub_notes
    span_s = (seg["end_s"] + seg["pad_tail"]) - (seg["start_s"] - seg["pad_lead"])
    total_samples = max(1, int(round(span_s * SR)))
    return sub, total_samples


def _render_segments(cfg, doc: dict, to_render: list[dict], cache_dir: str,
                     model: str, checkpoint: str, prior_mode: str,
                     device: str | None, progress) -> tuple[str, list[str]]:
    """Synthesize each non-cached segment in ``to_render`` to its cache WAV.

    Owns every torch import (kept inside the body so the module stays torch-free)
    and the lazy model/vocoder singletons; the two GPU forwards run under
    ``GPU_LOCK``. Returns ``(device, warnings)``. Extracted as a module-level
    function so the end-to-end tests can monkeypatch out the whole GPU pipeline and
    assert exactly the non-cached segments were rendered.
    """
    import torch

    from Arioso.infer import generate_mel
    from common.audio_io import write_pcm16
    from common.vocoder import vocode
    from DataSynthesizer.features import mel_for_training
    from DataSynthesizer.synthesizePrior import quantized_prior

    from .library import models_root as _models_root
    from .model_registry import get_model, get_vocoder

    n_render = len(to_render)
    _emit(progress, "model", 8, "loading model",
          segments_done=0, segments_total=n_render)
    loaded = get_model(_models_root(cfg), model, checkpoint, device=device)
    _emit(progress, "vocoder", 12, "loading vocoder",
          segments_done=0, segments_total=n_render)
    voc = get_vocoder(loaded.device)
    pitch = "bend" if prior_mode == "bend" else "quantized"

    warnings: list[str] = []
    for done, seg in enumerate(to_render, start=1):
        base_pct = int(15 + 80 * (done - 1) / n_render)
        _emit(progress, "prior", base_pct, f"segment {done}/{n_render}: prior",
              segments_done=done - 1, segments_total=n_render)

        sub, total_samples = _sub_doc(doc, seg)
        midi_path = os.path.join(cache_dir, f"{seg['hash']}.mid")
        _mid, seg_warn = write_midi(sub, midi_path, note_ids=None,
                                    prior_mode=prior_mode)
        warnings.extend(seg_warn)

        synth = quantized_prior(pitch=pitch)
        prior_mel = mel_for_training(synth.render(midi_path,
                                                  total_samples=total_samples))

        _emit(progress, "model", base_pct + 3,
              f"segment {done}/{n_render}: generating mel",
              segments_done=done - 1, segments_total=n_render)
        with GPU_LOCK:
            mel = generate_mel(loaded.model, prior_mel, loaded.cfg,
                               loaded.device, cond_frames=None)
            _emit(progress, "vocoder", base_pct + 6,
                  f"segment {done}/{n_render}: vocoding",
                  segments_done=done - 1, segments_total=n_render)
            audio = vocode(voc, torch.from_numpy(mel[None]).float())

        write_pcm16(os.path.join(cache_dir, seg["wav"]), audio)
        try:
            os.remove(midi_path)
        except OSError:
            pass
        _emit(progress, "model", int(15 + 80 * done / n_render),
              f"segment {done}/{n_render} done",
              segments_done=done, segments_total=n_render)
    return loaded.device, warnings


def run_render(cfg, doc: dict, project_dir: str, *, scope: str = "phrase",
               note_ids: list[str] | None = None, model: str | None = None,
               checkpoint: str | None = None, prior_mode: str | None = None,
               device: str | None = None, progress=None) -> dict:
    """Render ``doc`` to ``render/mix.wav`` (segment-cached) and return render meta.

    ``scope="phrase"`` renders the whole document; ``scope="selection"`` renders the
    same segmentation over **all** notes (the mix must reflect the true final audio)
    but only re-synthesizes segments missing from the cache — an edit changes a
    note's segment hash, so exactly the touched segment(s) miss and are re-rendered,
    while untouched cached segments are reused. Any non-cached untouched segment is
    still rendered so the mix is complete (e.g. a first-ever render). ``model`` /
    ``checkpoint`` / ``prior_mode`` default to the config defaults. ``progress`` is
    the JobManager status callback.
    """
    model = model or cfg.default_model
    checkpoint = checkpoint or cfg.default_checkpoint
    prior_mode = prior_mode or cfg.default_prior_mode
    sel_ids = set(note_ids) if (scope == "selection" and note_ids) else None
    project_id = doc.get("project_id") or os.path.basename(project_dir.rstrip("/\\"))

    rdir = os.path.join(project_dir, RENDER_DIR)
    cache_dir = os.path.join(rdir, SEG_CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)

    # -- plan (torch-free): segment, hash, detect cache hits ----------------
    _emit(progress, "serialize", 2, "planning segments")
    plan = plan_render(doc, cfg, model, checkpoint, prior_mode, cache_dir=cache_dir)
    segments = plan["segments"]
    total_segs = len(segments)
    to_render = [s for s in segments if not s["cached"]]
    n_render = len(to_render)

    warnings: list[str] = []
    loaded_device = device

    # -- render the non-cached segments -------------------------------------
    if n_render:
        loaded_device, seg_warn = _render_segments(
            cfg, doc, to_render, cache_dir, model, checkpoint, prior_mode,
            device, progress)
        warnings.extend(seg_warn)

    # -- stitch + write -----------------------------------------------------
    _emit(progress, "write", 95, "stitching mix",
          segments_done=n_render, segments_total=n_render)
    from common.audio_io import write_pcm16
    if total_segs:
        mix = stitch(segments, cache_dir, plan["total_duration_s"], SR)
    else:
        mix = np.zeros(int(_EMPTY_SILENCE_S * SR), dtype=np.float32)

    wav_path = os.path.join(rdir, MIX_WAV)
    write_pcm16(wav_path, mix)
    write_peaks(mix, os.path.join(rdir, MIX_PEAKS))

    # GC stale cache WAVs (keep the current manifest + a rolling budget).
    prune_cache(cache_dir, {s["hash"] for s in segments}, max_files=_MAX_CACHE_FILES)

    touched = 0
    seg_manifest = []
    for s in segments:
        is_touched = sel_ids is None or any(n["id"] in sel_ids for n in s["notes"])
        if sel_ids is not None and is_touched:
            touched += 1
        seg_manifest.append({
            "hash": s["hash"],
            "start_beat": round(s["start_beat"], 6),
            "end_beat": round(s["end_beat"], 6),
            "start_s": round(s["start_s"], 6),
            "end_s": round(s["end_s"], 6),
            "cached": bool(s["cached"]),
        })

    duration_s = float(len(mix)) / SR
    meta = {
        "wav": _media_url(project_id, RENDER_DIR, MIX_WAV),
        "peaks": _media_url(project_id, RENDER_DIR, MIX_PEAKS),
        "duration_s": duration_s,
        "model": model,
        "checkpoint": checkpoint,
        "prior_mode": prior_mode,
        "scope": scope,
        "note_ids": list(note_ids) if sel_ids is not None else None,
        "device": loaded_device,
        "segments": seg_manifest,
        "segments_total": total_segs,
        "segments_rendered": n_render,
        "segments_cached": total_segs - n_render,
        "segments_touched": touched if sel_ids is not None else None,
        "warnings": warnings,
    }
    atomic_write_json(os.path.join(rdir, RENDER_META), meta)

    _emit(progress, "write", 100, "done", state="done",
          segments_done=n_render, segments_total=n_render)
    return meta
