"""Processing pipeline: the 8 stages, the skip cache, and the JobManager.

``STAGES = (ingest, denoise, transcribe, align, velocity, vibrato, media,
finalize)``. Each stage reads its inputs from disk and writes its outputs, so any
stage can run in isolation (``--force-from``). A stage is skipped when its recorded
``params_hash`` in ``status.json`` still matches AND its outputs exist AND none of
its (data-dependency) upstream stages re-ran this pass — a forced or param-changed
stage cascades along the dependency graph (the notes chain
transcribe->align->velocity->vibrato->finalize, and a separate denoise->media branch).

The GPU transcribe stage holds the module-level ``transcribe.GPU_LOCK``. The
:class:`JobManager` runs a clip's pipeline on a background thread behind a per-clip
lock (one job per clip), writing ``status.json`` after every stage so the polling
UI sees progress; any stage exception is caught and recorded as
``state=error {code, detail}`` (a truncated traceback). :func:`process_cli` runs the
same pipeline synchronously for headless testing.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback

import numpy as np
import soundfile as sf

from common.audio_io import load_mono, voiced_rms_normalize, write_pcm16
from common.config import HOP_SIZE as HOP, SR

from . import media as media_mod
from . import notes_store
from . import transcribe as transcribe_mod
from .align import estimate_midi_offset, shift_note_times
from .config import LabelerConfig, MODEL_FPS, STAGES, load_config
from .denoise import denoise
from .library import (AUTO_JSON, CLEANED_WAV, MEL_DIR, MEL_META, MUSC_RAW_MID,
                      NOTES_JSON, ONSET_ENV, SOURCE_WAV, STATUS_JSON, STRETCH_DIR,
                      TRANSCRIPTION_JSON, atomic_write_json, clip_dir, clip_file,
                      clip_id_for_path, default_status, find_raw_wav, read_json)
from .midi_io import write_midi
from .velocity import compute_velocities
from .vibrato import detect_vibrato

# Data-dependency graph (a stage re-runs when any listed dep re-ran this pass).
_DEPS = {
    "ingest": (),
    "denoise": ("ingest",),
    "transcribe": ("denoise",),
    "align": ("transcribe",),
    "velocity": ("align",),
    "vibrato": ("velocity",),
    "media": ("denoise",),
    "finalize": ("vibrato",),
}

NOTES_MODES = ("keep_labels", "retranscribe_merge", "reset_notes")


class Pipeline:
    """Runs one clip's stages with skip-caching; independent of the server."""

    def __init__(self, cfg: LabelerConfig, clip_id: str, raw_path: str | None = None):
        self.cfg = cfg
        self.clip_id = clip_id
        self.dir = clip_dir(cfg, clip_id)
        self.raw_path = raw_path
        self._status: dict = {}
        self._recorded: dict = {}
        self._progress = None

    # -- path helpers --
    def _p(self, name: str) -> str:
        return os.path.join(self.dir, name)

    # -- output existence (secondary guard for the skip cache) --
    def _outputs_exist(self, stage: str) -> bool:
        d = self.dir
        if stage == "ingest":
            return os.path.isfile(self._p(SOURCE_WAV))
        if stage == "denoise":
            return os.path.isfile(self._p(CLEANED_WAV))
        if stage == "transcribe":
            return os.path.isfile(self._p(TRANSCRIPTION_JSON)) and os.path.isfile(self._p(MUSC_RAW_MID))
        if stage in ("align", "velocity", "vibrato"):
            return os.path.isfile(self._p(AUTO_JSON))
        if stage == "media":
            ok = os.path.isfile(os.path.join(d, MEL_DIR, MEL_META)) and os.path.isfile(self._p(ONSET_ENV))
            for f in media_mod.stretch_paths(self.cfg.media):
                ok = ok and os.path.isfile(os.path.join(d, STRETCH_DIR, f))
            return ok
        if stage == "finalize":
            return os.path.isfile(self._p(NOTES_JSON))
        return False

    def _compute_dirty(self, fidx: int | None) -> dict[str, bool]:
        dirty: dict[str, bool] = {}

        def resolve(stage: str) -> bool:
            if stage in dirty:
                return dirty[stage]
            forced = fidx is not None and STAGES.index(stage) >= fidx
            rec = self._recorded.get(stage, {})
            fresh = (rec.get("state") == "done"
                     and rec.get("hash") == self.cfg.stage_params_hash(stage)
                     and self._outputs_exist(stage))
            dep_dirty = any(resolve(d) for d in _DEPS[stage])
            dirty[stage] = bool(forced or not fresh or dep_dirty)
            return dirty[stage]

        for s in STAGES:
            resolve(s)
        return dirty

    # -- intermediate loaders --
    def _load_transcription(self) -> dict:
        tj = read_json(self._p(TRANSCRIPTION_JSON))
        if tj is None:
            raise RuntimeError("transcription.json missing (run the transcribe stage)")
        return tj

    def _load_or_init_auto(self, tj: dict) -> dict:
        n = len(tj["notes"])
        auto = read_json(self._p(AUTO_JSON)) or {}
        notes = auto.get("notes", [])
        if len(notes) != n:
            default_tech = self.cfg.default_technique
            notes = [{
                "amplitude": float(tj["notes"][i].get("amplitude", 0.0)),
                "technique": default_tech,
                "velocity": int(self.cfg.velocity.degenerate_velocity),
                "onset_strength": 0.0,
                "vibrato": False,
                "vibrato_rate_hz": 0.0,
                "vibrato_extent_cents": 0.0,
            } for i in range(n)]
            auto = {"offset_applied_s": float(auto.get("offset_applied_s", 0.0)), "notes": notes}
        return auto

    def _save_auto(self, auto: dict) -> None:
        atomic_write_json(self._p(AUTO_JSON), auto)

    # -- stages --
    def _stage_ingest(self) -> None:
        raw = self.raw_path or find_raw_wav(self.cfg, self.clip_id)
        if not raw or not os.path.isfile(raw):
            raise FileNotFoundError(
                f"no raw source wav for clip '{self.clip_id}' (looked in raw/ and status.json)")
        self.raw_path = raw
        y = load_mono(raw, sr=SR)
        y = voiced_rms_normalize(y, sr=SR)
        os.makedirs(self.dir, exist_ok=True)
        write_pcm16(self._p(SOURCE_WAV), y, SR)

    def _stage_denoise(self) -> None:
        y = load_mono(self._p(SOURCE_WAV), sr=SR)
        cleaned = denoise(y, self.cfg.denoise, sr=SR)
        write_pcm16(self._p(CLEANED_WAV), cleaned, SR)

    def _stage_transcribe(self) -> None:
        y = load_mono(self._p(CLEANED_WAV), sr=SR)
        with transcribe_mod.GPU_LOCK:
            result = transcribe_mod.transcribe_notes(y, self.cfg.transcribe)
        tj = {
            "model_fps": result.model_fps,
            "sr": result.sr,
            "notes": [{"start_s": n.start_s, "end_s": n.end_s, "pitch": n.pitch,
                       "amplitude": n.amplitude, "bend_cents": n.bend_cents}
                      for n in result.notes],
        }
        atomic_write_json(self._p(TRANSCRIPTION_JSON), tj)
        raw_notes = [{"start_s": n.start_s, "end_s": n.end_s, "pitch": n.pitch,
                      "velocity": max(1, int(round(127 * n.amplitude)))}
                     for n in result.notes]
        write_midi(self._p(MUSC_RAW_MID), raw_notes)

    def _stage_align(self) -> None:
        tj = self._load_transcription()
        cleaned = load_mono(self._p(CLEANED_WAV), sr=SR)
        offset = estimate_midi_offset(self._p(MUSC_RAW_MID), cleaned, sr=SR)
        auto = self._load_or_init_auto(tj)
        auto["offset_applied_s"] = -float(offset)   # advance the notes into alignment
        auto["offset_raw_s"] = float(offset)
        for i, n in enumerate(tj["notes"]):
            auto["notes"][i]["amplitude"] = float(n.get("amplitude", 0.0))
            auto["notes"][i].setdefault("technique", self.cfg.default_technique)
        self._save_auto(auto)

    def _stage_velocity(self) -> None:
        tj = self._load_transcription()
        auto = self._load_or_init_auto(tj)
        offset = float(auto.get("offset_applied_s", 0.0))
        cleaned = load_mono(self._p(CLEANED_WAV), sr=SR)
        onset_times = [max(0.0, float(n["start_s"]) + offset) for n in tj["notes"]]
        vels, strengths = compute_velocities(cleaned, onset_times, self.cfg.velocity, sr=SR, hop=HOP)
        for i in range(len(tj["notes"])):
            auto["notes"][i]["velocity"] = int(vels[i]) if vels else int(self.cfg.velocity.degenerate_velocity)
            auto["notes"][i]["onset_strength"] = float(strengths[i]) if strengths else 0.0
        self._save_auto(auto)

    def _stage_vibrato(self) -> None:
        tj = self._load_transcription()
        auto = self._load_or_init_auto(tj)
        fps = float(tj.get("model_fps", MODEL_FPS))
        for i, n in enumerate(tj["notes"]):
            dur = float(n["end_s"]) - float(n["start_s"])
            res = detect_vibrato(n.get("bend_cents", []), dur, fps=fps, params=self.cfg.vibrato)
            auto["notes"][i]["vibrato"] = bool(res.is_vibrato)
            auto["notes"][i]["vibrato_rate_hz"] = float(res.rate_hz)
            auto["notes"][i]["vibrato_extent_cents"] = float(res.extent_cents)
        self._save_auto(auto)

    def _stage_media(self) -> None:
        source = load_mono(self._p(SOURCE_WAV), sr=SR)
        cleaned = load_mono(self._p(CLEANED_WAV), sr=SR)
        media_mod.mel_tiles(cleaned, os.path.join(self.dir, MEL_DIR), self.cfg.media, sr=SR, hop=HOP)
        media_mod.write_onset_env(cleaned, self._p(ONSET_ENV), sr=SR, hop=self.cfg.media.onset_hop)
        media_mod.make_stretches({"source": source, "cleaned": cleaned},
                                 os.path.join(self.dir, STRETCH_DIR), self.cfg.media, sr=SR)

    def _stage_finalize(self, notes_mode: str, chain_ran: bool) -> str:
        import dataclasses
        tj = self._load_transcription()
        auto = self._load_or_init_auto(tj)
        offset = float(auto.get("offset_applied_s", 0.0))
        default_tech = self.cfg.default_technique
        degen = int(self.cfg.velocity.degenerate_velocity)

        final_notes = []
        for i, n in enumerate(tj["notes"]):
            a = auto["notes"][i]
            ns, ne = shift_note_times(n["start_s"], n["end_s"], offset)
            vel = int(a.get("velocity", degen))
            tech = str(a.get("technique", default_tech))
            vib = bool(a.get("vibrato", False))
            final_notes.append({
                "start_s": ns, "end_s": ne, "pitch": int(n["pitch"]),
                "velocity": vel, "technique": tech, "vibrato": vib,
                "auto": {
                    "velocity": vel, "technique": tech, "vibrato": vib,
                    "amplitude": float(a.get("amplitude", n.get("amplitude", 0.0))),
                    "onset_strength": float(a.get("onset_strength", 0.0)),
                    "vibrato_rate_hz": float(a.get("vibrato_rate_hz", 0.0)),
                    "vibrato_extent_cents": float(a.get("vibrato_extent_cents", 0.0)),
                },
            })

        try:
            info = sf.info(self._p(CLEANED_WAV))
            duration_s = float(info.frames) / float(info.samplerate)
        except Exception:
            duration_s = (final_notes[-1]["end_s"] if final_notes else 0.0)

        audio_meta = {
            "sr": SR, "hop": HOP, "source_wav": SOURCE_WAV, "cleaned_wav": CLEANED_WAV,
            "denoise": dataclasses.asdict(self.cfg.denoise),
            "denoise_hash": self.cfg.stage_params_hash("denoise"),
        }
        pipeline_meta = {
            "musc": {
                "batch_size": int(self.cfg.transcribe.batch_size),
                "model_fps": float(tj.get("model_fps", MODEL_FPS)),
                "onset_thresh": float(self.cfg.transcribe.onset_thresh),
                "frame_thresh": float(self.cfg.transcribe.frame_thresh),
                "min_note_len_ms": float(self.cfg.transcribe.min_note_len_ms),
                "params_hash": self.cfg.stage_params_hash("transcribe"),
            },
            "align": {"offset_applied_s": offset, "offset_raw_s": float(auto.get("offset_raw_s", -offset))},
            "velocity": dataclasses.asdict(self.cfg.velocity),
            "vibrato": dataclasses.asdict(self.cfg.vibrato),
        }
        new_doc = notes_store.build_document(
            self.cfg, self.clip_id, final_notes, offset_applied_s=offset,
            duration_s=duration_s, audio_meta=audio_meta, pipeline_meta=pipeline_meta)

        existing = notes_store.load(self.cfg, self.clip_id)
        if notes_mode == "keep_labels":
            if existing is not None and notes_store.human_edit_count(existing) > 0:
                return "preserved"
            if existing is not None and not chain_ran:
                return "kept"
            if existing is None:
                notes_store.save_new(self.cfg, self.clip_id, new_doc)
                return "built"
            notes_store.commit_rebuild(self.cfg, self.clip_id, new_doc, existing)
            return "rebuilt"
        if notes_mode == "reset_notes":
            notes_store.commit_rebuild(self.cfg, self.clip_id, new_doc, existing, snapshot_label="reset")
            return "reset"
        if notes_mode == "retranscribe_merge":
            if existing is not None:
                new_doc = notes_store.merge_retranscribe(existing, new_doc)
            notes_store.commit_rebuild(self.cfg, self.clip_id, new_doc, existing, snapshot_label="retranscribe")
            return "merged"
        raise ValueError(f"unknown notes_mode {notes_mode!r} (expected one of {NOTES_MODES})")

    # -- status + run --
    def _write_status(self, **kw) -> None:
        self._status.update(kw)
        self._status["stages"] = self._recorded
        self._status["n_stages"] = len(STAGES)
        atomic_write_json(self._p(STATUS_JSON), self._status)
        if self._progress:
            self._progress(dict(self._status))

    def run(self, force_from: str | None = None, notes_mode: str = "keep_labels",
            progress=None) -> dict:
        """Run the pipeline; return ``{stage: 'ran'|'skip'|<finalize result>}``."""
        if notes_mode not in NOTES_MODES:
            raise ValueError(f"unknown notes_mode {notes_mode!r} (expected {NOTES_MODES})")
        if force_from is not None and force_from not in STAGES:
            raise ValueError(f"unknown force_from stage {force_from!r} (expected {STAGES})")
        self._progress = progress
        os.makedirs(self.dir, exist_ok=True)
        self._status = read_json(self._p(STATUS_JSON)) or default_status(self.clip_id, self.raw_path)
        if self.raw_path:
            self._status["raw_path"] = self.raw_path
        self._recorded = self._status.get("stages", {}) or {}

        fidx = STAGES.index(force_from) if force_from else None
        dirty = self._compute_dirty(fidx)
        chain = ("transcribe", "align", "velocity", "vibrato")
        chain_ran = False
        summary: dict = {}
        self._write_status(state="processing", error=None, pct=-1)

        for idx, stage in enumerate(STAGES):
            try:
                if stage == "finalize":
                    self._write_status(state="processing", stage=stage, stage_index=idx,
                                       pct=-1, message="finalize")
                    res = self._stage_finalize(notes_mode, chain_ran)
                    self._recorded[stage] = {"state": "done", "hash": self.cfg.stage_params_hash(stage)}
                    summary[stage] = res
                elif dirty[stage]:
                    self._write_status(state="processing", stage=stage, stage_index=idx,
                                       pct=-1, message=f"running {stage}")
                    getattr(self, f"_stage_{stage}")()
                    self._recorded[stage] = {"state": "done", "hash": self.cfg.stage_params_hash(stage)}
                    summary[stage] = "ran"
                    if stage in chain:
                        chain_ran = True
                else:
                    summary[stage] = "skip"
                    self._recorded.setdefault(stage, {"state": "done",
                                                      "hash": self.cfg.stage_params_hash(stage)})
            except Exception as e:  # noqa: BLE001 - record + surface per-stage failure
                tb = "\n".join(traceback.format_exc().splitlines()[:30])
                self._write_status(state="error", stage=stage, stage_index=idx, pct=-1,
                                   message=f"error in {stage}",
                                   error={"code": type(e).__name__, "detail": tb, "stage": stage})
                raise

        self._write_status(state="done", stage="finalize", stage_index=len(STAGES) - 1,
                           pct=100, message="done", error=None)
        return summary


# --- JobManager (server-side background runner) -----------------------------

class JobManager:
    """Runs clip pipelines on background threads, one job per clip."""

    def __init__(self, cfg: LabelerConfig):
        self.cfg = cfg
        self._threads: dict[str, threading.Thread] = {}
        self._live: dict[str, dict] = {}
        self._guard = threading.Lock()

    def is_running(self, clip_id: str) -> bool:
        with self._guard:
            t = self._threads.get(clip_id)
            return bool(t and t.is_alive())

    def status(self, clip_id: str) -> dict | None:
        with self._guard:
            live = self._live.get(clip_id)
        if live is not None:
            return live
        return read_json(clip_file(self.cfg, clip_id, STATUS_JSON))

    def submit(self, clip_id: str, raw_path: str | None = None,
               force_from: str | None = None, notes_mode: str = "keep_labels") -> bool:
        """Start a job; return ``False`` if one is already running for this clip."""
        with self._guard:
            t = self._threads.get(clip_id)
            if t and t.is_alive():
                return False

            def _run():
                pipe = Pipeline(self.cfg, clip_id, raw_path=raw_path)

                def _prog(st):
                    with self._guard:
                        self._live[clip_id] = st
                try:
                    pipe.run(force_from=force_from, notes_mode=notes_mode, progress=_prog)
                except Exception:  # noqa: BLE001 - already recorded in status.json
                    pass
                finally:
                    with self._guard:
                        self._live[clip_id] = read_json(
                            clip_file(self.cfg, clip_id, STATUS_JSON)) or self._live.get(clip_id)

            th = threading.Thread(target=_run, name=f"labeler-job-{clip_id}", daemon=True)
            self._threads[clip_id] = th
            self._live[clip_id] = default_status(clip_id, raw_path) | {"state": "processing"}
            th.start()
            return True


# --- CLI --------------------------------------------------------------------

def process_cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m Labeler.processing",
        description="Run the Labeler pipeline on a raw wav path or an existing clip id.")
    ap.add_argument("target", help="path to a raw wav (new clip) or an existing clip id")
    ap.add_argument("--force-from", choices=STAGES, default=None,
                    help="re-run from this stage onward (ignoring the skip cache)")
    ap.add_argument("--notes-mode", choices=NOTES_MODES, default="keep_labels",
                    help="finalize policy for an existing notes.json")
    ap.add_argument("--config", default=None, help="path to a labeler.yaml override")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    raw_path = None
    if os.path.isfile(args.target):
        raw_path = os.path.abspath(args.target)
        clip_id = clip_id_for_path(raw_path)
    else:
        clip_id = args.target
        raw_path = find_raw_wav(cfg, clip_id)

    print(f"[labeler] clip_id={clip_id}  raw={raw_path}  "
          f"force_from={args.force_from}  notes_mode={args.notes_mode}")

    def _prog(st):
        print(f"  [{st.get('stage_index', 0) + 1}/{len(STAGES)}] "
              f"{st.get('state'):10s} {st.get('stage')}: {st.get('message')}")

    pipe = Pipeline(cfg, clip_id, raw_path=raw_path)
    try:
        summary = pipe.run(force_from=args.force_from, notes_mode=args.notes_mode, progress=_prog)
    except Exception as e:  # noqa: BLE001
        print(f"[labeler] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print("[labeler] summary: " + ", ".join(f"{k}={v}" for k, v in summary.items()))
    notes = notes_store.load(cfg, clip_id)
    n_notes = len(notes["notes"]) if notes else 0
    n_vib = sum(1 for n in (notes["notes"] if notes else []) if n["vibrato"])
    print(f"[labeler] done: {n_notes} notes, {n_vib} vibrato, "
          f"offset_applied_s={notes.get('offset_applied_s') if notes else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(process_cli())
