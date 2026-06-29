"""Held-out-**piece** split (Section 5): no piece appears in both train and eval.

A "piece" is ``(composer, catalog)`` — e.g. Kayser Op20-01 — which may have several performers
(distinct YouTube recordings). Splitting on the piece, not the recording, is what distinguishes
genuine timbre generalization from etude memorization. Deterministic (sorted keys + seeded
shuffle), persisted to ``data/arioso_split.json`` so train/eval/inference all read one split.
"""

from __future__ import annotations

import csv
import json
import os
import random

from .config import SPLIT_FILE, AriosoConfig


def _ok_rows(manifest_path: str) -> list[dict]:
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("status") == "ok"]


def make_split(out_dir: str, cfg: AriosoConfig | None = None,
               overwrite: bool = False) -> dict:
    """Compute (or load) the held-out-piece split. Returns ``{"train": [...], "val": [...]}``.

    Reserves ``cfg.val_frac`` of the *pieces* (rounded, >=1) for eval. The split file is
    written once and reused; pass ``overwrite=True`` to recompute.
    """
    cfg = cfg or AriosoConfig()
    split_path = os.path.join(out_dir, SPLIT_FILE)
    if os.path.isfile(split_path) and not overwrite:
        with open(split_path, encoding="utf-8") as f:
            return json.load(f)

    rows = _ok_rows(os.path.join(out_dir, "manifest.csv"))
    by_piece: dict[tuple[str, str], list[str]] = {}
    for r in rows:
        by_piece.setdefault((r["composer"], r["catalog"]), []).append(r["basename"])

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


def main() -> None:
    import argparse

    from DataSynthesizer.config import DEFAULT_OUT

    ap = argparse.ArgumentParser(description="Build the held-out-piece split.")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    split = make_split(args.out_dir, overwrite=args.overwrite)
    print(f"pieces: {split['n_pieces']}  held-out pieces: {split['n_val_pieces']}\n"
          f"train recordings: {len(split['train'])}  val recordings: {len(split['val'])}")


if __name__ == "__main__":
    main()
