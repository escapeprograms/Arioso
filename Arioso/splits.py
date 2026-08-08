"""Held-out-**piece** split (Section 5): no piece appears in both train and eval.

A "piece" is the manifest's ``piece`` field — the split-grouping key each producer stamps on a
clip (synthetic: ``"{composer}/{catalog}"`` e.g. Kayser Op20-01; gt_arky: the clip id). Splitting
on the piece, not the recording, is what distinguishes genuine timbre generalization from etude
memorization. Deterministic (sorted keys + seeded shuffle), persisted **per root** to
``<root>/split.json`` so train/eval/inference all read one split; an existing ``split.json`` is
loaded verbatim (this preserves a migrated root's held-out split exactly).

**Growing corpora.** Loading verbatim means a root that gained clips after its split was written
has basenames in *neither* list — silently invisible to training. :func:`update_split` reconciles
an existing split against the current manifest **additively** (new basenames assigned, vanished
ones pruned, every surviving assignment left where it is), and :func:`make_split`'s verbatim-load
path prints a loud warning when it sees such a stale split. The Labeler's compile calls
:func:`update_split` on every run, so a compiled root's split always covers its manifest.

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


# --- shared internals (make_split / update_split both build on these) -------------

def _split_path(root: DatasetRoot) -> str:
    """Path of ``root``'s split file (``<root>/split.json``)."""
    return os.path.join(root.path, SPLIT_FILE)


def _group_by_piece(root: DatasetRoot, basenames: list[str] | None = None) -> dict[str, list[str]]:
    """``{piece: [basename, ...]}`` over ``basenames`` (default: the whole manifest)."""
    by_piece: dict[str, list[str]] = {}
    for base in (root.basenames() if basenames is None else basenames):
        by_piece.setdefault(root.piece(base), []).append(base)
    return by_piece


def _n_val_pieces(n_pieces: int, cfg: AriosoConfig) -> int:
    """How many pieces are held out for eval: ``max(1, round(n_pieces * val_frac))``."""
    return max(1, round(n_pieces * cfg.val_frac))


def _write_split(root: DatasetRoot, train: list[str], val: list[str],
                 n_pieces: int, n_val_pieces: int) -> dict:
    """Serialize the canonical split doc to ``<root>/split.json`` and return it.

    THE one place the on-disk schema is produced — ``{"train", "val", "n_pieces",
    "n_val_pieces"}`` with both basename lists sorted — so every writer stays byte-compatible
    with every existing consumer.
    """
    split = {"train": sorted(train), "val": sorted(val),
             "n_pieces": n_pieces, "n_val_pieces": n_val_pieces}
    with open(_split_path(root), "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    return split


def missing_from_split(root: DatasetRoot, split: dict) -> list[str]:
    """Manifest basenames present in neither split list (invisible to train AND eval)."""
    assigned = set(split.get("train", ())) | set(split.get("val", ()))
    return [base for base in root.basenames() if base not in assigned]


def make_split(root: DatasetRoot, cfg: AriosoConfig | None = None,
               overwrite: bool = False) -> dict:
    """Compute (or load) one root's held-out-piece split — ``{"train": [...], "val": [...]}``.

    Reserves ``cfg.val_frac`` of the *pieces* (rounded, >=1) for eval. The split file
    ``<root>/split.json`` is written once and reused; pass ``overwrite=True`` to recompute. An
    existing ``split.json`` is loaded as-is (a migrated root keeps its exact held-out split).
    Pieces are grouped from the manifest's ``piece`` field (``root.piece(base)``).

    A loaded split that does **not** cover the whole manifest prints a warning (only — the
    returned split is untouched): those basenames are in neither train nor val, so they are
    invisible to training. :func:`update_split` (or ``python -m Arioso.splits --update``) fixes it.
    """
    cfg = cfg or AriosoConfig()
    split_path = _split_path(root)
    if os.path.isfile(split_path) and not overwrite:
        with open(split_path, encoding="utf-8") as f:
            split = json.load(f)
        missing = missing_from_split(root, split)
        if missing:
            print(f"[splits] WARNING: {root.path!r}: {len(missing)} of "
                  f"{len(root.basenames())} manifest clips are in NEITHER train nor val "
                  f"(e.g. {', '.join(missing[:3])}{' ...' if len(missing) > 3 else ''}) — "
                  f"they are invisible to training. Run "
                  f"`python -m Arioso.splits --data-root {root.path} --update` "
                  f"(or recompile) to add them.")
        return split

    by_piece = _group_by_piece(root)
    pieces = sorted(by_piece)                       # deterministic order
    rng = random.Random(cfg.seed)
    rng.shuffle(pieces)
    n_val = _n_val_pieces(len(pieces), cfg)
    val_pieces = set(pieces[:n_val])

    train, val = [], []
    for piece, bases in by_piece.items():
        (val if piece in val_pieces else train).extend(bases)
    return _write_split(root, train, val, len(pieces), n_val)


def update_split(root: DatasetRoot, cfg: AriosoConfig | None = None) -> tuple[dict, dict]:
    """Additively reconcile ``<root>/split.json`` with the current manifest.

    Returns ``(split, summary)`` — the rewritten split doc (same schema as :func:`make_split`)
    and ``{"created", "added_train", "added_val", "pruned", "n_pieces", "n_val_pieces"}`` for the
    caller to log. With no split file yet this is exactly :func:`make_split` (``created=True``).

    The reconciliation, in order:

    1. **Prune** basenames no longer in the manifest (e.g. a clip de-verified and pruned by the
       Labeler's compile) from both lists.
    2. **Never move a surviving assignment.** Every basename already in train stays in train, and
       every basename already in val stays in val. Val stability across corpus growth is the whole
       point: an eval set that reshuffles when new clips land makes metrics incomparable between
       runs (and leaks previously-held-out material into train).
    3. **Assign the new basenames by piece.** A new basename whose piece is already assigned joins
       that piece's side (the held-out-piece invariant — no piece in both lists). Genuinely new
       pieces are sorted, shuffled with ``random.Random(cfg.seed)``, and sent to val only until the
       overall held-out piece count reaches ``max(1, round(total_pieces * cfg.val_frac))``; the
       rest go to train.
    4. Rewrite the file with recomputed ``n_pieces`` / ``n_val_pieces``.
    """
    cfg = cfg or AriosoConfig()
    split_path = _split_path(root)
    if not os.path.isfile(split_path):
        split = make_split(root, cfg)
        return split, {"created": True,
                       "added_train": len(split["train"]), "added_val": len(split["val"]),
                       "pruned": 0, "n_pieces": split["n_pieces"],
                       "n_val_pieces": split["n_val_pieces"]}

    with open(split_path, encoding="utf-8") as f:
        old = json.load(f)

    current = set(root.basenames())
    train = [b for b in old.get("train", []) if b in current]   # (1) prune
    val = [b for b in old.get("val", []) if b in current]       # (2) order preserved, side kept
    pruned = len(old.get("train", [])) + len(old.get("val", [])) - len(train) - len(val)

    train_pieces = {root.piece(b) for b in train}
    val_pieces = {root.piece(b) for b in val}
    n_train0, n_val0 = len(train), len(val)

    # (3) new basenames, grouped by piece; a piece already on a side keeps every clip on it.
    new_by_piece = _group_by_piece(root, [b for b in sorted(current)
                                          if b not in set(train) | set(val)])
    fresh_pieces = []
    for piece, bases in new_by_piece.items():
        if piece in val_pieces:
            val.extend(bases)
        elif piece in train_pieces:
            train.extend(bases)
        else:
            fresh_pieces.append(piece)

    fresh_pieces.sort()                                   # deterministic order before the shuffle
    random.Random(cfg.seed).shuffle(fresh_pieces)
    n_pieces = len(train_pieces | val_pieces) + len(fresh_pieces)
    target_val = _n_val_pieces(n_pieces, cfg)
    for piece in fresh_pieces:
        if len(val_pieces) < target_val:
            val_pieces.add(piece)
            val.extend(new_by_piece[piece])
        else:
            train_pieces.add(piece)
            train.extend(new_by_piece[piece])

    split = _write_split(root, train, val, n_pieces, len(val_pieces))   # (4)
    summary = {"created": False, "added_train": len(train) - n_train0,
               "added_val": len(val) - n_val0, "pruned": pruned,
               "n_pieces": n_pieces, "n_val_pieces": len(val_pieces)}
    return split, summary


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
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true",
                      help="recompute each split from scratch (reshuffles the held-out pieces)")
    mode.add_argument("--update", action="store_true",
                      help="additively reconcile each split with its manifest: assign new "
                           "clips, prune vanished ones, never move an existing assignment")
    args = ap.parse_args()

    from .clips import open_roots

    root_paths = args.data_roots or ["Data"]
    for root, root_path in zip(open_roots(root_paths), root_paths):
        if args.update:
            split, info = update_split(root)
            what = "created" if info["created"] else "updated"
            print(f"root {root_path!r}: {what} +{info['added_train']} train, "
                  f"+{info['added_val']} val, pruned {info['pruned']} "
                  f"(val pieces {info['n_val_pieces']}/{info['n_pieces']})")
        else:
            split = make_split(root, overwrite=args.overwrite)
        print(f"root {root_path!r}: pieces {split['n_pieces']}  "
              f"held-out pieces {split['n_val_pieces']}  "
              f"train recordings {len(split['train'])}  val recordings {len(split['val'])}")


if __name__ == "__main__":
    main()
