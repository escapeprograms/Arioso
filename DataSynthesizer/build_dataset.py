"""Build the Arioso training set: target + prior mel features per aligned clip.

Walks the violin-transcription MIDI dataset, and for each clip:
  1. downloads the GT audio (level-normalized once per video), trims it, and saves
     the target wav + its mel  (``download_audio.fetch_clip``, ``features``)
  2. renders both sawtooth priors — quantized and pitch-bend — to the SAME length
     (``synthesizePrior.render_prior`` / ``render_prior_bend``)
  3. estimates the residual onset offset from the quantized prior and shifts BOTH
     priors by it (``onset_align``), then saves each prior's mel
  4. saves the onset mask (1 at each onset, exponential decay) at mel granularity
For each clip it writes ``gt/<base>.wav``, ``target_mel/<base>.npy``,
``prior_mel_quant/<base>.npy``, ``prior_mel_bend/<base>.npy`` and
``prior_onset/<base>.npy``, recording one row (including the applied ``offset_ms``)
in ``manifest.csv``. Prior audio is not saved — only the mels are needed downstream.

Downloads are cached per YouTube id, so the many clips cut from one video only
download once. Already-finished clips are skipped, so the build is resumable and
clips whose video is unavailable are logged and skipped rather than aborting.

Example::

    # smoke test: one book, first 2 clips
    python -m DataSynthesizer.build_dataset --books Kayser --limit 2
    # full build
    python -m DataSynthesizer.build_dataset
"""

from __future__ import annotations

import csv
import glob
import os
import sys
import traceback

import numpy as np
import soundfile as sf

from .clip_name import parse_clip_name
from .config import BOOKS, DEFAULT_DATASET, DEFAULT_OUT, SR
from .download_audio import fetch_clip
from .features import build_onset_mask, mel_for_training
from .onset_align import estimate_offset_seconds, shift_samples
from .synthesizePrior import note_onsets, render_prior, render_prior_bend

MANIFEST_FIELDS = [
    "basename", "book", "composer", "catalog", "performer", "youtube_id",
    "start_sec", "end_sec", "duration_sec", "n_samples", "n_frames", "offset_ms",
    "gt_path", "target_mel_path", "prior_mel_quant_path", "prior_mel_bend_path",
    "prior_onset_path", "status",
]


def process_clip(midi_path: str, out_dir: str, cache_dir: str,
                 sr: int = SR, overwrite: bool = False) -> dict:
    """Produce the target + prior training features for one MIDI clip; return a row."""
    clip = parse_clip_name(midi_path)
    book = os.path.basename(os.path.dirname(midi_path))

    dirs = {
        "gt": os.path.join(out_dir, "gt"),
        "target_mel": os.path.join(out_dir, "target_mel"),
        "prior_mel_quant": os.path.join(out_dir, "prior_mel_quant"),
        "prior_mel_bend": os.path.join(out_dir, "prior_mel_bend"),
        "prior_onset": os.path.join(out_dir, "prior_onset"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    gt_path = os.path.join(dirs["gt"], clip.basename + ".wav")
    target_mel_path = os.path.join(dirs["target_mel"], clip.basename + ".npy")
    prior_mel_quant_path = os.path.join(dirs["prior_mel_quant"], clip.basename + ".npy")
    prior_mel_bend_path = os.path.join(dirs["prior_mel_bend"], clip.basename + ".npy")
    prior_onset_path = os.path.join(dirs["prior_onset"], clip.basename + ".npy")

    row = {
        "basename": clip.basename, "book": book, "composer": clip.composer,
        "catalog": clip.catalog, "performer": clip.performer,
        "youtube_id": clip.youtube_id, "start_sec": clip.start,
        "end_sec": clip.end, "duration_sec": clip.end - clip.start,
        "n_samples": "", "n_frames": "", "offset_ms": "", "gt_path": gt_path,
        "target_mel_path": target_mel_path,
        "prior_mel_quant_path": prior_mel_quant_path,
        "prior_mel_bend_path": prior_mel_bend_path,
        "prior_onset_path": prior_onset_path, "status": "",
    }

    outputs = (gt_path, target_mel_path, prior_mel_quant_path,
               prior_mel_bend_path, prior_onset_path)
    if not overwrite and all(os.path.isfile(p) for p in outputs):
        row["n_samples"] = sf.info(gt_path).frames
        row["n_frames"] = int(np.load(target_mel_path, mmap_mode="r").shape[-1])
        row["status"] = "exists"
        return row

    # 1) Target audio (download cached + level-normalized, then trim) + its mel. The
    #    download is the step that can fail when a video is private/removed/blocked.
    _, y_gt, _ = fetch_clip(midi_path, cache_dir, out_path=gt_path, sr=sr)
    np.save(target_mel_path, mel_for_training(y_gt))

    # 2) Both priors, rendered to exactly the GT length so each pair is sample-aligned.
    y_quant = render_prior(midi_path, sr=sr, total_samples=len(y_gt))
    y_bend = render_prior_bend(midi_path, sr=sr, total_samples=len(y_gt))

    # 3) Estimate the residual onset offset ONCE from the quantized prior (the
    #    canonical reference), then advance BOTH priors by that same shift so they
    #    line up with the GT and stay mutually comparable. Save each prior's mel.
    applied = -estimate_offset_seconds(y_quant, y_gt, sr=sr)
    y_quant = shift_samples(y_quant, applied, sr)
    y_bend = shift_samples(y_bend, applied, sr)
    prior_mel_quant = mel_for_training(y_quant)
    np.save(prior_mel_quant_path, prior_mel_quant)
    np.save(prior_mel_bend_path, mel_for_training(y_bend))

    # 4) Onset mask on the mel grid (shared; onset timings are identical for both).
    n_frames = prior_mel_quant.shape[-1]
    np.save(prior_onset_path,
            build_onset_mask(note_onsets(midi_path), applied, n_frames, sr=sr))

    row["n_samples"] = len(y_gt)
    row["n_frames"] = n_frames
    row["offset_ms"] = round(applied * 1000, 1)
    row["status"] = "ok"
    return row


def build(dataset_root: str = DEFAULT_DATASET, out_dir: str = DEFAULT_OUT,
          books=BOOKS, limit: int | None = None, sr: int = SR,
          overwrite: bool = False) -> str:
    """Run the full build; write ``{out_dir}/manifest.csv``; return its path."""
    cache_dir = os.path.join(out_dir, "_cache")
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.csv")

    midis = []
    for book in books:
        midis.extend(sorted(glob.glob(os.path.join(dataset_root, book, "*.mid"))))
    if limit:
        midis = midis[:limit]

    rows, n_ok, n_skip, n_fail = [], 0, 0, 0
    for i, midi in enumerate(midis, 1):
        base = os.path.splitext(os.path.basename(midi))[0]
        try:
            row = process_clip(midi, out_dir, cache_dir, sr=sr, overwrite=overwrite)
            if row["status"] == "ok":
                n_ok += 1
            else:
                n_skip += 1
            print(f"[{i}/{len(midis)}] {row['status']:8s} {base}")
        except Exception as exc:  # noqa: BLE001 - we want to log and continue
            n_fail += 1
            row = {f: "" for f in MANIFEST_FIELDS}
            row["basename"] = base
            row["book"] = os.path.basename(os.path.dirname(midi))
            row["status"] = f"FAILED: {type(exc).__name__}: {exc}"
            print(f"[{i}/{len(midis)}] FAILED   {base}: {exc}", file=sys.stderr)
            traceback.print_exc(limit=1)
        rows.append(row)
        _write_manifest(manifest_path, rows)  # flush each step (resumable, long runs)

    print(f"\nDone: {n_ok} built, {n_skip} skipped/exists, {n_fail} failed "
          f"-> {manifest_path}")
    return manifest_path


def _write_manifest(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--books", nargs="+", default=list(BOOKS), choices=list(BOOKS))
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N clips (smoke testing)")
    ap.add_argument("--sr", type=int, default=SR)
    ap.add_argument("--overwrite", action="store_true",
                    help="re-render even if outputs already exist")
    args = ap.parse_args()

    build(dataset_root=args.dataset_root, out_dir=args.out_dir, books=args.books,
          limit=args.limit, sr=args.sr, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
