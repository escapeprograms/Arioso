"""Deterministic onset-aligned clip enumeration (Section 5).

Training examples are short onset-aligned excerpts of the long recordings, not whole
recordings. We precompute, once, a fixed pool of overlapping clips (then the dataset shuffles
per epoch). For each recording, for **each** aligned onset ``o_i`` with at least ``L_min`` of
audio remaining: start at ``o_i``, target a ~10 s length, and end on the onset nearest to that
target (clamped to the recording end). Because note boundaries don't fall exactly on 10 s, clips
are variable length, roughly 5-10 s (~430-860 frames at hop 512).

Slicing is on precomputed mel frames (contiguous blocks) — no audio re-windowing. The fixed
enumeration gives full positional coverage, reproducibility, and easy debugging.

**Multi-root.** A clip is keyed by ``(root, basename, start, end)`` where ``root`` is the index
into the loader's ordered :class:`~common.dataset_schema.DatasetRoot` list — the same ordering
:func:`Arioso.splits.make_splits` emits — so a basename shared by two roots stays distinct.
Per-recording ``n_frames`` comes from each root's ``manifest.json`` (not the old ``manifest.csv``)
and onset frames from ``<root>/onsets/<base>.npy``.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np

from common.dataset_schema import MANIFEST_JSON, DatasetRoot

from .config import AriosoConfig


def open_roots(paths: list[str]) -> list[DatasetRoot]:
    """Open each path as a :class:`DatasetRoot`, with a migration hint on a missing manifest.

    A standard root is defined by its ``manifest.json``; the raw ``FileNotFoundError`` from the
    reader only names the missing file, so this wrapper turns it into an actionable CLI-level error
    (the real ``Data/`` root must be migrated before Arioso can consume it).
    """
    roots: list[DatasetRoot] = []
    for path in paths:
        manifest_path = os.path.join(path, MANIFEST_JSON)
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"dataset root {path!r} has no {MANIFEST_JSON} (expected at {manifest_path}). "
                f"Arioso consumes standard roots only — generate the manifest first: "
                f"`python -m DataSynthesizer.migrate_root --root {path}` for the synthetic root, "
                f"or `python -m Labeler.compile --all` for a ground-truth root.")
        roots.append(DatasetRoot(path))
    return roots


class Clip(NamedTuple):
    root: int         # index into the loader's ordered DatasetRoot list
    basename: str
    start: int        # start mel frame (inclusive)
    end: int          # end mel frame (exclusive)


def enumerate_clips(roots: list[DatasetRoot], basenames_per_root: list[list[str]],
                    cfg: AriosoConfig | None = None) -> list[Clip]:
    """Build the fixed clip pool over ``roots`` (``basenames_per_root[i]`` are root ``i``'s bases).

    ``n_frames`` per recording comes from the root's ``manifest.json`` (``root.n_frames(base)``);
    aligned onset frames are read from ``root.onsets_path(base)``. Onsets with less than ``L_min``
    of remaining audio are skipped. Each emitted :class:`Clip` carries its root index so a basename
    present in more than one root is never conflated.
    """
    cfg = cfg or AriosoConfig()
    l_min, target = cfg.l_min_frames, cfg.target_frames

    clips: list[Clip] = []
    for ri, (root, basenames) in enumerate(zip(roots, basenames_per_root)):
        for base in basenames:
            n_frames = root.n_frames(base)                 # manifest.json (listed clip == complete)
            onset_path = root.onsets_path(base)
            if not os.path.isfile(onset_path):
                continue
            onsets = np.sort(np.load(onset_path).astype(np.int64))
            if onsets.size == 0:
                continue
            for f_i in onsets:
                if n_frames - f_i < l_min:                 # not enough audio remaining
                    continue
                target_end = f_i + target
                later = onsets[onsets > f_i]
                if later.size:
                    end = int(later[np.argmin(np.abs(later - target_end))])
                else:
                    end = n_frames
                end = min(max(end, f_i + l_min), n_frames)  # >= L_min, within recording
                clips.append(Clip(ri, base, int(f_i), int(end)))
    return clips


def _basenames_per_root(roots: list[DatasetRoot], entries: list) -> list[list[str]]:
    """Group a merged split list ``[(root_idx, base), ...]`` into a per-root basename list."""
    per_root: list[list[str]] = [[] for _ in roots]
    for root_idx, base in entries:
        per_root[root_idx].append(base)
    return per_root


def main() -> None:
    import argparse

    from .splits import make_splits

    ap = argparse.ArgumentParser(description="Enumerate the clip pool and report stats.")
    ap.add_argument("--data-root", action="append", dest="data_roots",
                    help="dataset root (repeatable; default: Data)")
    ap.add_argument("--split", choices=("train", "val"), default="train")
    args = ap.parse_args()

    cfg = AriosoConfig()
    root_paths = args.data_roots or ["Data"]
    roots = open_roots(root_paths)
    split = make_splits(roots, cfg)
    per_root = _basenames_per_root(roots, split[args.split])
    clips = enumerate_clips(roots, per_root, cfg)

    for ri, (root, bases) in enumerate(zip(roots, per_root)):
        rclips = [c for c in clips if c.root == ri]
        if rclips:
            lens = np.array([c.end - c.start for c in rclips])
            print(f"  root {ri} {root.path!r}: {len(rclips)} clips over {len(bases)} recordings  "
                  f"frames min/median/max: {lens.min()}/{int(np.median(lens))}/{lens.max()}")
        else:
            print(f"  root {ri} {root.path!r}: 0 clips over {len(bases)} recordings")
    if clips:
        lens = np.array([c.end - c.start for c in clips])
        print(f"{args.split}: {len(clips)} clips total over "
              f"{sum(len(b) for b in per_root)} recordings\n"
              f"frames min/median/max: {lens.min()}/{int(np.median(lens))}/{lens.max()}  "
              f"(~{lens.min()/cfg.sr*cfg.hop:.1f}-{lens.max()/cfg.sr*cfg.hop:.1f} s)")
    else:
        print(f"{args.split}: 0 clips (did you run `python -m DataSynthesizer.build_prior`?)")


if __name__ == "__main__":
    main()
