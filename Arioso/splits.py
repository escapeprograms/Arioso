"""Held-out-**piece** split (Section 5): no piece appears in both train and eval.

A "piece" is the manifest's ``piece`` field — the split-grouping key each producer stamps on a
clip (synthetic: ``"{composer}/{catalog}"`` e.g. Kayser Op20-01; gt_arky: the clip id). Splitting
on the piece, not the recording, is what distinguishes genuine timbre generalization from etude
memorization. Deterministic (sorted keys + seeded shuffle), persisted **per root** to
``<root>/split.json`` so train/eval/inference all read one split; an existing ``split.json`` is
loaded verbatim (this preserves a migrated root's held-out split exactly).

**Multi-root.** :func:`make_split` computes/loads one root's split; :func:`make_splits` merges an
ordered list of roots into ``{"train": [(root_idx, base), ...], "val": [...]}`` (root order
preserved), the keying Arioso's dataset + clip enumerator consume.
"""

from __future__ import annotations

import json
import os
import random

from common.dataset_schema import DatasetRoot

from .config import SPLIT_FILE, AriosoConfig


def make_split(root: DatasetRoot, cfg: AriosoConfig | None = None,
               overwrite: bool = False) -> dict:
    """Compute (or load) one root's held-out-piece split — ``{"train": [...], "val": [...]}``.

    Reserves ``cfg.val_frac`` of the *pieces* (rounded, >=1) for eval. The split file
    ``<root>/split.json`` is written once and reused; pass ``overwrite=True`` to recompute. An
    existing ``split.json`` is loaded as-is (a migrated root keeps its exact held-out split).
    Pieces are grouped from the manifest's ``piece`` field (``root.piece(base)``).
    """
    cfg = cfg or AriosoConfig()
    split_path = os.path.join(root.path, SPLIT_FILE)
    if os.path.isfile(split_path) and not overwrite:
        with open(split_path, encoding="utf-8") as f:
            return json.load(f)

    by_piece: dict[str, list[str]] = {}
    for base in root.basenames():
        by_piece.setdefault(root.piece(base), []).append(base)

    pieces = sorted(by_piece)                       # deterministic order
    rng = random.Random(cfg.seed)
    rng.shuffle(pieces)
    n_val = max(1, round(len(pieces) * cfg.val_frac))
    val_pieces = set(pieces[:n_val])

    train, val = [], []
    for piece, bases in by_piece.items():
        (val if piece in val_pieces else train).extend(sorted(bases))
    split = {"train": sorted(train), "val": sorted(val),
             "n_pieces": len(pieces), "n_val_pieces": n_val}

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    return split


def make_splits(roots: list[DatasetRoot], cfg: AriosoConfig | None = None) -> dict:
    """Merge each root's split into ``{"train": [(root_idx, base), ...], "val": [...]}``.

    Root order is preserved (the ``root_idx`` matches the ordering :class:`Arioso.clips.Clip`
    and :class:`Arioso.dataset.AriosoDataset` key off). Each root's split is computed/loaded via
    :func:`make_split`.
    """
    cfg = cfg or AriosoConfig()
    merged: dict[str, list] = {"train": [], "val": []}
    for root_idx, root in enumerate(roots):
        split = make_split(root, cfg)
        for name in ("train", "val"):
            merged[name].extend((root_idx, base) for base in split[name])
    return merged


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build the held-out-piece split(s).")
    ap.add_argument("--data-root", action="append", dest="data_roots",
                    help="dataset root (repeatable; default: Data)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from .clips import open_roots

    root_paths = args.data_roots or ["Data"]
    for root, root_path in zip(open_roots(root_paths), root_paths):
        split = make_split(root, overwrite=args.overwrite)
        print(f"root {root_path!r}: pieces {split['n_pieces']}  "
              f"held-out pieces {split['n_val_pieces']}  "
              f"train recordings {len(split['train'])}  val recordings {len(split['val'])}")


if __name__ == "__main__":
    main()
