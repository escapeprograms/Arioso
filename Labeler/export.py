"""Export a labeled clip into the artifacts the training pipeline consumes.

Writes, under ``<dataset_root>/exports/``:

  * ``midi/<id>.mid`` — the notes as a plain single-instrument score (what
    ``quantized_prior`` / ``note_onsets`` / ``note_groups_from_midi`` read).
  * ``gt/<id>.wav`` — the chosen wav variant (cleaned by default) with every
    ``mute_region`` cosine-zeroed (5 ms edges) so muted stretches are truly silent.
  * ``notes/<id>.notes.csv`` — the ``build_techniques`` per-note-group template
    (note_index, onset/end frame, span seconds, n_notes, technique) + ``velocity``
    and ``vibrato`` columns.
  * (optional) ``articulation_rec|vibrato_rec|velocity_rec/<id>.npy`` — ``[T]``
    uint8 per-frame tracks on the mel grid (``ceil(len(gt)/HOP)`` frames).

The articulation track is built exactly like the dataset does — ``note_groups_from_midi``
+ ``expand_to_frames`` (np.round rounding, rest-snap 10) — then the fill's ``REST_ID``
(5, DataSynthesizer's) is remapped to *our* local rest index (``len(articulations)``);
a config with an articulation that would collide with id 5 is rejected. Velocity and
vibrato are simple per-note span fills (rest = 0).
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np

from common.audio_io import load_mono, write_pcm16
from common.config import HOP_SIZE as HOP, SR
from DataSynthesizer.config import REST_ID
from DataSynthesizer.technique import expand_to_frames, note_groups_from_midi

from . import notes_store
from .config import LabelerConfig, load_config
from .library import (CLEANED_WAV, SOURCE_WAV, ClipNotFound, clip_dir, clip_file,
                      exports_dir)
from .midi_io import validate_for_export, write_export_midi

_REC_DIRS = {
    "articulation": "articulation_rec",
    "vibrato": "vibrato_rec",
    "velocity": "velocity_rec",
}


def apply_mute_regions(y: np.ndarray, regions: list[dict], sr: int = SR,
                       fade_ms: float = 5.0) -> np.ndarray:
    """Return a copy of ``y`` with each mute region silenced via cosine edges."""
    y = np.array(y, dtype=np.float32, copy=True)
    n = len(y)
    f = max(1, int(round(fade_ms / 1000.0 * sr)))
    for reg in regions or []:
        s = max(0, int(round(float(reg["start_s"]) * sr)))
        e = min(n, int(round(float(reg["end_s"]) * sr)))
        if e <= s:
            continue
        gain = np.ones(e - s, dtype=np.float32)
        ff = min(f, (e - s) // 2)
        if ff > 0:
            ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, ff)))  # 1 -> 0
            gain[:ff] = ramp
            gain[-ff:] = ramp[::-1]
        gain[ff:len(gain) - ff] = 0.0
        y[s:e] *= gain
    return y


def _local_tech_id(cfg: LabelerConfig, name: str) -> int:
    names = cfg.articulation_names
    return names.index(name) if name in names else 0  # unknown -> normal (0)


def _frame(t: float, n_frames: int) -> int:
    return int(np.clip(int(np.round(float(t) * SR / HOP)), 0, n_frames))


def export_clip(cfg: LabelerConfig, clip_id: str, gt_variant: str | None = None,
                include_frames: bool = True) -> dict:
    """Export ``clip_id``; return ``{written, warnings, errors, class_order}``."""
    doc = notes_store.load(cfg, clip_id)
    if doc is None:
        raise ClipNotFound(f"{clip_id} (no notes.json to export)")
    notes = doc.get("notes", [])
    gt_variant = gt_variant or cfg.export.gt_variant
    variant_wav = CLEANED_WAV if gt_variant == "cleaned" else SOURCE_WAV

    errors, warnings = validate_for_export(notes)

    out = exports_dir(cfg)
    written: dict[str, str] = {}

    # -- MIDI --
    midi_dir = os.path.join(out, "midi")
    os.makedirs(midi_dir, exist_ok=True)
    midi_path = os.path.join(midi_dir, f"{clip_id}.mid")
    write_export_midi(midi_path, notes, cfg.export)
    written["midi"] = midi_path

    # -- GT wav (chosen variant, mute regions cosine-zeroed) --
    src_wav = clip_file(cfg, clip_id, variant_wav)
    if not os.path.isfile(src_wav):
        raise ClipNotFound(f"{clip_id} (missing {variant_wav}; process the clip first)")
    y = load_mono(src_wav, sr=SR)
    y = apply_mute_regions(y, doc.get("mute_regions", []), sr=SR,
                           fade_ms=cfg.export.mute_fade_ms)
    gt_dir = os.path.join(out, "gt")
    os.makedirs(gt_dir, exist_ok=True)
    gt_path = os.path.join(gt_dir, f"{clip_id}.wav")
    write_pcm16(gt_path, y, SR)
    written["gt"] = gt_path

    n_frames = int(math.ceil(len(y) / HOP))

    # -- per-onset representative technique/velocity/vibrato (first note wins) --
    per_onset: dict[int, dict] = {}
    for nt in sorted(notes, key=lambda x: (float(x["start_s"]), int(x["pitch"]))):
        of = int(np.round(float(nt["start_s"]) * SR / HOP))
        if 0 <= of < n_frames:
            per_onset.setdefault(of, nt)

    groups = note_groups_from_midi(midi_path, 0.0, n_frames)

    # -- notes CSV (build_techniques template + velocity, vibrato) --
    notes_dir = os.path.join(out, "notes")
    os.makedirs(notes_dir, exist_ok=True)
    csv_path = os.path.join(notes_dir, f"{clip_id}.notes.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["note_index", "onset_frame", "end_frame", "start_s", "end_s",
                    "n_notes", "technique", "velocity", "vibrato"])
        for i, g in enumerate(groups):
            rep = per_onset.get(g.onset_frame, {})
            w.writerow([i, g.onset_frame, g.end_frame, f"{g.start_s:.6f}",
                        f"{g.end_s:.6f}", g.n_notes, rep.get("technique", cfg.default_technique),
                        int(rep.get("velocity", 80)), int(bool(rep.get("vibrato", False)))])
    written["notes_csv"] = csv_path

    class_order = [*cfg.articulation_names, "rest"]

    if include_frames:
        local_rest = len(cfg.articulation_names)
        tech_ids = np.array(
            [_local_tech_id(cfg, per_onset.get(g.onset_frame, {}).get("technique", cfg.default_technique))
             for g in groups], dtype=np.int64)
        if REST_ID in set(tech_ids.tolist()):
            raise ValueError(
                f"articulation id {REST_ID} collides with DataSynthesizer REST_ID; "
                f"frame export supports at most 5 articulations (got {local_rest})")
        artic = expand_to_frames(groups, tech_ids, n_frames).astype(np.int64)
        artic[artic == REST_ID] = local_rest
        artic = artic.astype(np.uint8)

        vib = np.zeros(n_frames, dtype=np.uint8)
        vel = np.zeros(n_frames, dtype=np.uint8)
        for nt in notes:
            f0 = _frame(nt["start_s"], n_frames)
            f1 = _frame(nt["end_s"], n_frames)
            if f1 <= f0:
                f1 = min(n_frames, f0 + 1)
            if nt.get("vibrato"):
                vib[f0:f1] = 1
            vel[f0:f1] = int(np.clip(int(nt.get("velocity", 80)), 1, 127))

        for kind, arr in (("articulation", artic), ("vibrato", vib), ("velocity", vel)):
            d = os.path.join(out, _REC_DIRS[kind])
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, f"{clip_id}.npy")
            np.save(p, arr)
            written[f"{kind}_rec"] = p

    return {"written": written, "warnings": warnings, "errors": errors,
            "class_order": class_order, "n_frames": n_frames}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m Labeler.export",
                                 description="Export a labeled clip's training artifacts.")
    ap.add_argument("clip_id")
    ap.add_argument("--gt-variant", choices=("cleaned", "source"), default=None)
    ap.add_argument("--no-frames", action="store_true", help="skip the per-frame npy tracks")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    res = export_clip(cfg, args.clip_id, gt_variant=args.gt_variant,
                      include_frames=not args.no_frames)
    for k, v in res["written"].items():
        print(f"  {k:16s} {v}")
    if res["errors"]:
        print("errors:", *res["errors"], sep="\n  ")
    if res["warnings"]:
        print("warnings:", *res["warnings"], sep="\n  ")
    print(f"class_order: {res['class_order']}  n_frames: {res['n_frames']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
