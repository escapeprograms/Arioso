"""Classify each GT note's violin playing technique with VioPTT (one-time dataset pass).

Adds a per-clip technique track to the DataSynthesizer ``data/`` tree, alongside the priors and
onsets ``build_prior`` already wrote — no re-download, no re-render. For each ``status==ok``
clip it writes:

* ``data/technique_arioso/<base>.npy`` — ``[T]`` uint8 per-frame technique ids, frame-aligned to
  ``target_mel`` / ``prior_mel_arioso`` (frame t_sec = t*HOP/SR). Ids 0-4 are VioPTT's class ids
  (flageolet, normal, pizzicato, spiccato, no_technique); id 5 ("rest") is ours, filling frames
  where nothing is sounding.
* ``data/technique_arioso/<base>.notes.csv`` — one row per note-group (the per-note VioPTT
  prediction + its span), for verification / inspection.

**Note-group policy.** Notes are parsed from the score with pretty_midi (via
:func:`DataSynthesizer.technique.note_groups_from_midi`), grouped by rounded onset frame so
chords / double-stops collapse to one group. VioPTT classifies each group's audio slice
(min-onset → max-offset seconds); the id is then broadcast across the group's frames with a
**hybrid rest fill** (:func:`DataSynthesizer.technique.expand_to_frames`): a note extends to the
next onset when the gap is ``<= TECHNIQUE_REST_SNAP_FRAMES`` (legato / rounding slack), else it
stops at its own offset and the true gap becomes rest.

**Fixed 2 s note window.** Each group's slice is pad/truncated to :data:`TECHNIQUE_NOTE_SECONDS`
(default 2.0 s) and peak-normalized before classification — matching how VioPTT's note model was
trained (see :meth:`DataSynthesizer.technique.TechniqueClassifier.prepare_chunks`). Labels built
before this fix used variable-length un-normalized slices (VioPTT's off-distribution from-midi
inference path) and are STALE — they skew heavily to ``spiccato``; re-run with ``--overwrite`` to
replace them. ``--note-seconds 0`` restores the legacy variable-length path for comparison.

**Onsets tripwire.** Because the group onset frames are computed exactly as ``build_prior``
computed ``onsets_arioso``, they MUST match the saved onsets array. Each clip asserts this; a
mismatch means the artifacts are stale (rebuild ``build_prior`` first) and the clip fails loudly
rather than writing a misaligned track.

Resumable + skip-existing (both output files must exist to skip), mirroring
:mod:`DataSynthesizer.build_prior`.

Run::

    python -m DataSynthesizer.build_techniques --limit 2      # smoke test
    python -m DataSynthesizer.build_techniques                # full pass over status==ok clips
"""

from __future__ import annotations

import csv
import os
import sys
import traceback

import numpy as np

from common.audio_io import load_mono

from .config import (DEFAULT_DATASET, DEFAULT_OUT, ONSETS_DIR, TECHNIQUE_CLASSES,
                     TECHNIQUE_DIR, TECHNIQUE_PAD_S, TECHNIQUE_NOTE_SECONDS)
from .technique import (NOTE_CKPT, TRANSCRIPTOR_CKPT, VIOPTT_SR, TechniqueClassifier,
                        expand_to_frames, note_groups_from_midi)


def _midi_path(dataset_root: str, book: str, basename: str) -> str:
    return os.path.join(dataset_root, book, basename + ".mid")


def process_clip(row: dict, out_dir: str, dataset_root: str, clf: TechniqueClassifier, *,
                 pad_s: float = TECHNIQUE_PAD_S,
                 note_seconds: float | None = TECHNIQUE_NOTE_SECONDS,
                 overwrite: bool = False) -> str:
    """Classify one manifest row's note techniques → per-frame npy + per-note CSV.

    Returns a status string ("ok" / "exists"). Asserts the group onset frames match the clip's
    ``onsets_arioso`` array (stale-artifact guard) before writing anything.
    """
    base = row["basename"]
    tech_dir = os.path.join(out_dir, TECHNIQUE_DIR)
    os.makedirs(tech_dir, exist_ok=True)
    npy_path = os.path.join(tech_dir, base + ".npy")
    csv_path = os.path.join(tech_dir, base + ".notes.csv")

    if not overwrite and os.path.isfile(npy_path) and os.path.isfile(csv_path):
        return "exists"

    n_frames = int(row["n_frames"])
    offset_s = float(row["offset_ms"]) / 1000.0
    midi = _midi_path(dataset_root, row["book"], base)
    gt_path = os.path.join(out_dir, "gt", base + ".wav")

    groups = note_groups_from_midi(midi, offset_s, n_frames)

    # Stale-artifact tripwire: our onset frames must equal build_prior's onsets_arioso.
    onsets_path = os.path.join(out_dir, ONSETS_DIR, base + ".npy")
    onset_frames = np.array([g.onset_frame for g in groups], dtype=np.int32)
    if not np.array_equal(onset_frames, np.load(onsets_path)):
        raise AssertionError("onset mismatch vs onsets_arioso (stale artifacts?)")

    wav16 = load_mono(gt_path, sr=VIOPTT_SR)
    ids, conf = clf.classify(wav16, [(g.start_s, g.end_s) for g in groups],
                             pad_s, note_seconds=note_seconds)

    np.save(npy_path, expand_to_frames(groups, ids, n_frames))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["note_index", "onset_frame", "end_frame", "start_s", "end_s",
                         "n_notes", "technique", "confidence"])
        for i, g in enumerate(groups):
            label = (TECHNIQUE_CLASSES[int(ids[i])]
                     if int(ids[i]) < len(TECHNIQUE_CLASSES) else str(int(ids[i])))
            writer.writerow([i, g.onset_frame, g.end_frame,
                             f"{g.start_s:.6f}", f"{g.end_s:.6f}", g.n_notes,
                             label, f"{float(conf[i]):.6f}"])
    return "ok"


def build(out_dir: str = DEFAULT_OUT, dataset_root: str = DEFAULT_DATASET, *,
          device: str = "cuda", pad_s: float = TECHNIQUE_PAD_S,
          note_seconds: float | None = TECHNIQUE_NOTE_SECONDS,
          note_ckpt: str = NOTE_CKPT, transcriptor_ckpt: str = TRANSCRIPTOR_CKPT,
          limit: int | None = None, overwrite: bool = False) -> None:
    """Pass over ``manifest.csv`` (status==ok rows), classifying each clip's note techniques."""
    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "ok"]
    if limit:
        rows = rows[:limit]

    # Load both VioPTT checkpoints once and reuse across every clip.
    clf = TechniqueClassifier(device=device, note_ckpt=note_ckpt,
                              transcriptor_ckpt=transcriptor_ckpt)

    n_ok = n_skip = n_fail = 0
    for i, row in enumerate(rows, 1):
        base = row["basename"]
        try:
            status = process_clip(row, out_dir, dataset_root, clf,
                                  pad_s=pad_s, note_seconds=note_seconds, overwrite=overwrite)
            n_ok += status == "ok"
            n_skip += status == "exists"
            print(f"[{i}/{len(rows)}] {status:8s} {base}")
        except Exception as exc:  # noqa: BLE001 - log and continue, like build_prior
            n_fail += 1
            print(f"[{i}/{len(rows)}] FAILED   {base}: {exc}", file=sys.stderr)
            traceback.print_exc(limit=1)

    print(f"\nDone: {n_ok} built, {n_skip} existed, {n_fail} failed -> "
          f"{os.path.join(out_dir, TECHNIQUE_DIR)}")


def main() -> None:
    import argparse

    import torch

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET)
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N ok clips (smoke testing)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-classify even if outputs already exist")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="torch device for the VioPTT models (default: cuda if available)")
    ap.add_argument("--pad-seconds", type=float, default=TECHNIQUE_PAD_S,
                    help="per-note slice padding fed to VioPTT")
    ap.add_argument("--note-seconds", type=float,
                    default=(TECHNIQUE_NOTE_SECONDS if TECHNIQUE_NOTE_SECONDS else 0.0),
                    help="fixed per-note window (s) matching VioPTT training (default 2.0); "
                         "pass 0 for the legacy variable-length path")
    ap.add_argument("--note-ckpt", default=NOTE_CKPT)
    ap.add_argument("--transcriptor-ckpt", default=TRANSCRIPTOR_CKPT)
    args = ap.parse_args()

    build(out_dir=args.out_dir, dataset_root=args.dataset_root, device=args.device,
          pad_s=args.pad_seconds, note_seconds=(args.note_seconds or None),
          note_ckpt=args.note_ckpt, transcriptor_ckpt=args.transcriptor_ckpt,
          limit=args.limit, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
