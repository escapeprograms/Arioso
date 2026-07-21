"""Isolated transcribe worker: run one clip's MUSC forward pass in a fresh process.

The transcribe stage is the only GPU stage, and the MUSC model singleton
(:mod:`Labeler.transcribe`) stays resident on the GPU across clips with no
``empty_cache``. On a small (8 GB) WDDM display GPU that resident footprint plus
allocator fragmentation could tip a forward pass into a shared-memory spill and a
driver-timeout machine lock. Running the pass in a **separate process** makes the
OS reclaim 100% of the CUDA context (weights + fragmentation) on exit, so nothing
accumulates between clips, and a GPU OOM surfaces as a normal non-zero child exit
the parent can catch instead of a frozen server.

Disk-in / disk-out, exactly like every other stage: reads ``cleaned.wav`` and
writes ``transcription.json`` + ``musc_raw.mid``. It never touches ``status.json``
— the parent :class:`Labeler.processing.Pipeline` remains the single status writer.

:func:`run_transcribe` is the shared core: the parent's in-process fallback
(``use_subprocess=False``) calls it directly under ``GPU_LOCK``, and this module's
CLI (``python -m Labeler.transcribe_worker <clip_id> [--config <yaml>]``) calls it
in the child. One code path, two callers.
"""

from __future__ import annotations

import argparse
import sys

from common.audio_io import load_mono
from common.config import SR

from . import transcribe as transcribe_mod
from .config import TranscribeParams, load_config
from .library import (CLEANED_WAV, MUSC_RAW_MID, TRANSCRIPTION_JSON,
                      atomic_write_json, clip_file)
from .midi_io import write_midi


def run_transcribe(cleaned_wav: str, transcription_json: str, musc_raw_mid: str,
                   params: TranscribeParams) -> None:
    """Transcribe ``cleaned_wav`` and write ``transcription.json`` + ``musc_raw.mid``.

    The GPU forward pass runs here (via :func:`transcribe.transcribe_notes`). This
    function does NOT take ``GPU_LOCK`` — in the child it is the only thread, and
    the in-process caller wraps it in the lock itself. Mirrors the body of
    ``processing.Pipeline._stage_transcribe`` so the two paths never diverge.
    """
    y = load_mono(cleaned_wav, sr=SR)
    result = transcribe_mod.transcribe_notes(y, params)
    tj = {
        "model_fps": result.model_fps,
        "sr": result.sr,
        "notes": [{"start_s": n.start_s, "end_s": n.end_s, "pitch": n.pitch,
                   "amplitude": n.amplitude, "bend_cents": n.bend_cents}
                  for n in result.notes],
    }
    atomic_write_json(transcription_json, tj)
    raw_notes = [{"start_s": n.start_s, "end_s": n.end_s, "pitch": n.pitch,
                  "velocity": max(1, int(round(127 * n.amplitude)))}
                 for n in result.notes]
    write_midi(musc_raw_mid, raw_notes)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m Labeler.transcribe_worker",
        description="Transcribe one clip's cleaned.wav in an isolated process.")
    ap.add_argument("clip_id", help="clip id (its dir must already hold cleaned.wav)")
    ap.add_argument("--config", default=None, help="path to a labeler.yaml override")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cleaned = clip_file(cfg, args.clip_id, CLEANED_WAV)
    tj_path = clip_file(cfg, args.clip_id, TRANSCRIPTION_JSON)
    mid_path = clip_file(cfg, args.clip_id, MUSC_RAW_MID)

    # Peak-VRAM readout (verification aid; harmless if CUDA is absent).
    torch = None
    try:
        import torch  # noqa: PLC0415 - optional, only for the readout
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        torch = None

    run_transcribe(cleaned, tj_path, mid_path, cfg.transcribe)

    if torch is not None and torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"[transcribe_worker] {args.clip_id}: peak CUDA "
              f"{peak_mb:.0f} MB (batch_size={cfg.transcribe.batch_size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
