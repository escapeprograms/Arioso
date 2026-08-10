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
     (the segment span incl. pads, so the tail isn't clipped) -> ``mel_frames``.
  3. **model** — lazy-load the checkpoint (:mod:`Studio.model_registry`) and run
     ``generate_mel`` (chunked CFM ODE). When the checkpoint carries per-frame
     conditioning, the segment's notes are rasterized to cond tracks
     (``_segment_note_events`` -> ``Arioso.infer.build_cond``) on the prior mel's
     grid; an unconditioned checkpoint passes ``cond_frames=None`` unchanged.
  4. **vocoder** — the frozen BigVGAN-v2 vocoder the render selected (a fine-tuned
     generator under ``cfg.vocoders_root``, or the stock ``"hf"`` checkpoint)
     -> float32 segment waveform.
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
from common.dataset_schema import NoteEvent

from .cache import plan_render, prune_cache, stitch
from .library import (MIX_PEAKS, MIX_WAV, PRIOR_MIX_PEAKS, PRIOR_MIX_WAV,
                      RENDER_DIR, RENDER_META, SEG_CACHE_DIR, atomic_write_json)
from .midi_export import (_MIN_NOTE_S, project_to_pretty_midi, select_notes,
                          validate_notes)
from .peaks import write_peaks
from .timing import beats_to_seconds, seconds_to_beats

# Serializes GPU forwards (mel gen + vocode) across projects rendering at once.
GPU_LOCK = threading.Lock()

# Per-project locks serializing the torch-free prior-mix render (render_prior_mix)
# against concurrent writes of the same project's prior_mix.wav.
_PRIOR_LOCKS: dict[str, threading.Lock] = {}
_PRIOR_LOCKS_GUARD = threading.Lock()


def _prior_lock(project_id: str) -> threading.Lock:
    with _PRIOR_LOCKS_GUARD:
        lk = _PRIOR_LOCKS.get(project_id)
        if lk is None:
            lk = _PRIOR_LOCKS[project_id] = threading.Lock()
        return lk

# Silence written for an empty project (no notes) so the mix.wav is a valid file.
_EMPTY_SILENCE_S = 0.05


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


def _segment_note_events(sub: dict, tech_map: dict[str, str],
                         warnings: list[str]) -> list[NoteEvent]:
    """A segment sub-doc's notes as :class:`NoteEvent`\\ s for conditioning (torch-free).

    Mirrors :func:`Studio.midi_export.project_to_pretty_midi`'s timing so the cond spans line up
    frame-for-frame with the MIDI the prior is rendered from: ``start_s`` = beats->seconds at the
    doc BPM, ``end_s`` = ``start_s + max(len_s, _MIN_NOTE_S)`` (the same short-note floor). Velocity
    is clamped to ``[1, 127]``. Each note's ``technique`` is mapped through ``tech_map`` to the
    articulation name the conditioned model expects; an unknown / unmapped technique falls back to
    ``"normal"`` (a background render job must not crash — :func:`rasterize_articulation` hard-errors
    on an unknown name) with one warning per distinct unknown technique. ``vibrato`` is on when the
    note's vibrato depth is ``> 0`` (the :func:`Studio.bend.has_expression` predicate).
    """
    bpm = float(sub["bpm"])
    events: list[NoteEvent] = []
    unknown_seen: set[str] = set()
    for n in sub["notes"]:
        start_s = beats_to_seconds(float(n["start_beat"]), bpm)
        len_s = beats_to_seconds(float(n["len_beats"]), bpm)
        end_s = start_s + max(len_s, _MIN_NOTE_S)
        vel = max(1, min(127, int(round(float(n.get("velocity", 100))))))
        tech = str(n.get("technique", "normal"))
        artic = tech_map.get(tech)
        if artic is None:
            artic = "normal"
            if tech not in unknown_seen:
                unknown_seen.add(tech)
                warnings.append(
                    f"unknown technique {tech!r} has no model mapping; conditioned as 'normal'")
        vibrato = float((n.get("vibrato") or {}).get("depth_semitones", 0.0)) > 0.0
        events.append(NoteEvent(start_s=start_s, end_s=end_s, pitch=int(n["pitch"]),
                                velocity=vel, articulation=artic, vibrato=vibrato))
    return events


def _render_segments(cfg, doc: dict, to_render: list[dict], cache_dir: str,
                     model: str, checkpoint: str, prior_mode: str, vocoder: str,
                     device: str | None, progress) -> tuple[str, list[str]]:
    """Synthesize each non-cached segment in ``to_render`` to its cache WAV.

    Owns every torch import (kept inside the body so the module stays torch-free)
    and the lazy model/vocoder singletons; the two GPU forwards run under
    ``GPU_LOCK``. ``vocoder`` is a Studio vocoder name (a dir under
    ``cfg.vocoders_root``, or ``"hf"``), resolved to the loader's ``checkpoint_dir``
    here. Returns ``(device, warnings)``. Extracted as a module-level function so
    the end-to-end tests can monkeypatch out the whole GPU pipeline and assert
    exactly the non-cached segments were rendered.
    """
    import torch

    from Arioso.infer import build_cond, generate_mel
    from common.audio_io import write_pcm16
    from common.prior import quantized_prior
    from common.vocoder import mel_frames, vocode

    from .library import models_root as _models_root
    from .library import vocoder_checkpoint_dir
    from .model_registry import get_model, get_vocoder

    n_render = len(to_render)
    _emit(progress, "model", 8, "loading model",
          segments_done=0, segments_total=n_render)
    loaded = get_model(_models_root(cfg), model, checkpoint, device=device)
    _emit(progress, "vocoder", 12, "loading vocoder",
          segments_done=0, segments_total=n_render)
    voc = get_vocoder(loaded.device,
                      checkpoint_dir=vocoder_checkpoint_dir(cfg, vocoder))
    pitch = "bend" if prior_mode == "bend" else "quantized"
    # Per-frame conditioning tracks are built only when the checkpoint expects them; an
    # unconditioned checkpoint keeps cond_frames=None. tech_map resolves each note's technique
    # to the model's articulation vocab.
    cond_specs = loaded.cfg.conditioning
    tech_map = cfg.technique_model_vocab()

    warnings: list[str] = []
    for done, seg in enumerate(to_render, start=1):
        base_pct = int(15 + 80 * (done - 1) / n_render)
        _emit(progress, "prior", base_pct, f"segment {done}/{n_render}: prior",
              segments_done=done - 1, segments_total=n_render)

        sub, total_samples = _sub_doc(doc, seg)
        # Build the score in memory (no temp .mid round-trip) so per-note env_dct
        # attributes survive into the prior; keep write_midi's same-pitch-overlap
        # warning behavior by validating the same selection it did.
        _seg_errors, seg_warn = validate_notes(select_notes(sub, None))
        warnings.extend(seg_warn)
        pm = project_to_pretty_midi(sub, note_ids=None, prior_mode=prior_mode)

        synth = quantized_prior(pitch=pitch)
        prior_mel = mel_frames(synth.render(pm, total_samples=total_samples))

        cond_frames = None
        if cond_specs:
            ev = _segment_note_events(sub, tech_map, warnings)
            cond_frames = build_cond(ev, prior_mel.shape[-1], cond_specs)

        _emit(progress, "model", base_pct + 3,
              f"segment {done}/{n_render}: generating mel",
              segments_done=done - 1, segments_total=n_render)
        with GPU_LOCK:
            mel = generate_mel(loaded.model, prior_mel, loaded.cfg,
                               loaded.device, cond_frames=cond_frames)
            _emit(progress, "vocoder", base_pct + 6,
                  f"segment {done}/{n_render}: vocoding",
                  segments_done=done - 1, segments_total=n_render)
            audio = vocode(voc, torch.from_numpy(mel[None]).float())

        write_pcm16(os.path.join(cache_dir, seg["wav"]), audio)
        _emit(progress, "model", int(15 + 80 * done / n_render),
              f"segment {done}/{n_render} done",
              segments_done=done, segments_total=n_render)
    return loaded.device, warnings


def run_render(cfg, doc: dict, project_dir: str, *, scope: str = "phrase",
               note_ids: list[str] | None = None, model: str | None = None,
               checkpoint: str | None = None, prior_mode: str | None = None,
               vocoder: str | None = None, device: str | None = None,
               progress=None) -> dict:
    """Render ``doc`` to ``render/mix.wav`` (segment-cached) and return render meta.

    ``scope="phrase"`` renders the whole document; ``scope="selection"`` renders the
    same segmentation over **all** notes (the mix must reflect the true final audio)
    but only re-synthesizes segments missing from the cache — an edit changes a
    note's segment hash, so exactly the touched segment(s) miss and are re-rendered,
    while untouched cached segments are reused. Any non-cached untouched segment is
    still rendered so the mix is complete (e.g. a first-ever render). ``model`` /
    ``checkpoint`` / ``prior_mode`` / ``vocoder`` default to the config defaults
    (``vocoder`` is a name: a dir under ``cfg.vocoders_root``, or ``"hf"`` for the
    stock BigVGAN — it is folded into the segment hashes, so swapping it
    re-renders everything). ``progress`` is the JobManager status callback.
    """
    model = model or cfg.default_model
    checkpoint = checkpoint or cfg.default_checkpoint
    prior_mode = prior_mode or cfg.default_prior_mode
    vocoder = vocoder or cfg.default_vocoder
    sel_ids = set(note_ids) if (scope == "selection" and note_ids) else None
    project_id = doc.get("project_id") or os.path.basename(project_dir.rstrip("/\\"))

    rdir = os.path.join(project_dir, RENDER_DIR)
    cache_dir = os.path.join(rdir, SEG_CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)

    # -- plan (torch-free): segment, hash, detect cache hits ----------------
    _emit(progress, "serialize", 2, "planning segments")
    plan = plan_render(doc, cfg, model, checkpoint, prior_mode, vocoder=vocoder,
                       cache_dir=cache_dir)
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
            vocoder, device, progress)
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

    # GC stale cache WAVs (keep the current manifest + a rolling count/size budget).
    prune_cache(cache_dir, {s["hash"] for s in segments},
                max_files=cfg.cache.max_files, max_mb=cfg.cache.max_mb)

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
        "vocoder": vocoder,
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


def render_prior_mix(cfg, doc: dict, project_dir: str,
                     prior_mode: str | None = None) -> dict:
    """Render the whole document's raw saw prior to ``render/prior_mix.wav`` (+ peaks).

    The torch-free, GPU-free companion to :func:`run_render`: it synthesizes the
    corner-saw prior audio the diffusion model consumes (with each note's ``env_dct``
    baked in via :func:`Studio.midi_export.project_to_pretty_midi`) so the UI can
    audition the prior as a third playback source. No model / vocoder / ``GPU_LOCK``.
    Notes render at their absolute onsets, so the WAV starts at beat 0 like the model
    mix. A per-project lock serializes concurrent writes of the same ``prior_mix.wav``.

    Returns ``{wav, peaks, duration_s, prior_mode}`` with ``/media`` URLs (mirroring
    :func:`run_render`'s meta). Raises ``ValueError`` on a non-positive-length note.
    """
    from common.audio_io import write_pcm16
    from common.prior import quantized_prior

    prior_mode = prior_mode or cfg.default_prior_mode
    project_id = doc.get("project_id") or os.path.basename(project_dir.rstrip("/\\"))
    pitch = "bend" if prior_mode == "bend" else "quantized"

    rdir = os.path.join(project_dir, RENDER_DIR)
    os.makedirs(rdir, exist_ok=True)

    bpm = float(doc["bpm"])
    notes = doc.get("notes", [])
    end_s = max((beats_to_seconds(float(n["start_beat"]) + float(n["len_beats"]), bpm)
                 for n in notes), default=0.0)
    if notes:
        # Match the render mix's trailing pad so the prior tail isn't clipped.
        total_samples = max(1, int(round((end_s + cfg.cache.pad_s) * SR)))
    else:
        total_samples = int(_EMPTY_SILENCE_S * SR)

    with _prior_lock(project_id):
        pm = project_to_pretty_midi(doc, note_ids=None, prior_mode=prior_mode)
        synth = quantized_prior(pitch=pitch)
        y = np.asarray(synth.render(pm, total_samples=total_samples), dtype=np.float32)

        write_pcm16(os.path.join(rdir, PRIOR_MIX_WAV), y)
        write_peaks(y, os.path.join(rdir, PRIOR_MIX_PEAKS))

    return {
        "wav": _media_url(project_id, RENDER_DIR, PRIOR_MIX_WAV),
        "peaks": _media_url(project_id, RENDER_DIR, PRIOR_MIX_PEAKS),
        "duration_s": float(len(y)) / SR,
        "prior_mode": prior_mode,
    }
