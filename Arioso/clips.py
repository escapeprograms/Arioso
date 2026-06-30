"""Deterministic onset-aligned clip enumeration (Section 5).

Training examples are short onset-aligned excerpts of the long recordings, not whole
recordings. We precompute, once, a fixed pool of overlapping clips (then the dataset shuffles
per epoch). For each recording, for **each** aligned onset ``o_i`` with at least ``L_min`` of
audio remaining: start at ``o_i``, target a ~10 s length, and end on the onset nearest to that
target (clamped to the recording end). Because note boundaries don't fall exactly on 10 s, clips
are variable length, roughly 5-10 s (~430-860 frames at hop 512).

Slicing is on precomputed mel frames (contiguous blocks) — no audio re-windowing. The fixed
enumeration gives full positional coverage, reproducibility, and easy debugging.
"""

from __future__ import annotations

import csv
import os
from typing import NamedTuple

import numpy as np

from .config import ONSETS_DIR, AriosoConfig


class Clip(NamedTuple):
    basename: str
    start: int        # start mel frame (inclusive)
    end: int          # end mel frame (exclusive)


def _n_frames_by_base(out_dir: str) -> dict[str, int]:
    with open(os.path.join(out_dir, "manifest.csv"), newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "ok"]
    return {r["basename"]: int(r["n_frames"]) for r in rows if r["n_frames"]}


def enumerate_clips(out_dir: str, basenames, cfg: AriosoConfig | None = None) -> list[Clip]:
    """Build the fixed clip pool for the given recordings (``basenames``).

    Reads aligned onset frames from ``data/onsets_arioso/<base>.npy``; needs ``n_frames`` per
    recording from the manifest. Onsets shorter than ``L_min`` of remaining audio are skipped.
    """
    cfg = cfg or AriosoConfig()
    l_min, target = cfg.l_min_frames, cfg.target_frames
    n_frames_by = _n_frames_by_base(out_dir)
    onset_dir = os.path.join(out_dir, ONSETS_DIR)

    clips: list[Clip] = []
    for base in basenames:
        n_frames = n_frames_by.get(base)
        onset_path = os.path.join(onset_dir, base + ".npy")
        if n_frames is None or not os.path.isfile(onset_path):
            continue
        onsets = np.sort(np.load(onset_path).astype(np.int64))
        if onsets.size == 0:
            continue
        for f_i in onsets:
            if n_frames - f_i < l_min:          # not enough audio remaining
                continue
            target_end = f_i + target
            later = onsets[onsets > f_i]
            if later.size:
                end = int(later[np.argmin(np.abs(later - target_end))])
            else:
                end = n_frames
            end = min(max(end, f_i + l_min), n_frames)   # >= L_min, within recording
            clips.append(Clip(base, int(f_i), int(end)))
    return clips


def main() -> None:
    import argparse

    from DataSynthesizer.config import DEFAULT_OUT

    from .splits import make_split

    ap = argparse.ArgumentParser(description="Enumerate the clip pool and report stats.")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--split", choices=("train", "val"), default="train")
    args = ap.parse_args()

    cfg = AriosoConfig()
    split = make_split(args.out_dir)
    clips = enumerate_clips(args.out_dir, split[args.split], cfg)
    if clips:
        lens = np.array([c.end - c.start for c in clips])
        print(f"{args.split}: {len(clips)} clips over {len(split[args.split])} recordings\n"
              f"frames min/median/max: {lens.min()}/{int(np.median(lens))}/{lens.max()}  "
              f"(~{lens.min()/cfg.sr*cfg.hop:.1f}-{lens.max()/cfg.sr*cfg.hop:.1f} s)")
    else:
        print(f"{args.split}: 0 clips (did you run `python -m DataSynthesizer.build_prior`?)")


if __name__ == "__main__":
    main()
