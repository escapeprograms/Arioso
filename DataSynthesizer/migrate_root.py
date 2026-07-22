"""One-shot migration of the existing synthetic ``Data/`` root to the standard layout.

The synthetic dataset predates :mod:`common.dataset_schema`: its priors/onsets live in
Arioso-flavoured dir names (``prior_mel_arioso`` / ``onsets_arioso``), it has no
``manifest.json``, its split is ``arioso_split.json``, and it still carries deprecated
VioPTT technique artifacts (``technique_arioso``). This script brings that root up to the
contract with **cheap renames only — no mel rebuilds**:

1. ``prior_mel_arioso`` -> ``prior_mel``, ``onsets_arioso`` -> ``onsets`` (``os.rename``).
2. generate ``manifest.json`` from ``manifest.csv`` (reuses :mod:`DataSynthesizer.export_manifest`).
3. copy ``arioso_split.json`` -> ``split.json`` (byte-identical; preserves the held-out split).
4. delete ``technique_arioso`` (deprecated VioPTT artifacts) — always.
5. ``--purge-legacy`` also deletes the pass-1 ``prior_mel_quant`` / ``prior_mel_bend`` /
   ``prior_onset`` dirs; without the flag their sizes are printed and left in place.

Every step is idempotent (skipped with a note if already done). ``--dry-run`` prints what
*would* happen and touches nothing. The producer's own trees — ``gt_arky``, ``recorded``,
``studio_projects``, ``_cache`` — and the kept ``gt`` / ``target_mel`` dirs are never touched.

Run::

    python -m DataSynthesizer.migrate_root --root Data --dry-run   # inspect first
    python -m DataSynthesizer.migrate_root --root Data             # apply
    python -m DataSynthesizer.migrate_root --root Data --purge-legacy
"""

from __future__ import annotations

import os
import shutil

from common.dataset_schema import DIR_ONSETS, DIR_PRIOR_MEL, MANIFEST_JSON, SPLIT_JSON

from . import export_manifest
from .export_manifest import MANIFEST_CSV

# Legacy -> standard directory renames (step 1).
RENAMES = (
    ("prior_mel_arioso", DIR_PRIOR_MEL),
    ("onsets_arioso", DIR_ONSETS),
)
LEGACY_SPLIT = "arioso_split.json"
TECHNIQUE_DIR = "technique_arioso"        # deprecated VioPTT artifacts, always removed
PURGE_DIRS = ("prior_mel_quant", "prior_mel_bend", "prior_onset")  # pass-1 legacy, opt-in


def _dir_size(path: str) -> int:
    """Total bytes of all files under ``path`` (0 if it does not exist)."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def _fmt_size(n: int) -> str:
    """Human-readable byte count (e.g. ``"1.2 GiB"``)."""
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{n} B"


def migrate(root: str, *, dry_run: bool = False, purge_legacy: bool = False) -> None:
    """Migrate ``root`` to the standard layout (see module docstring)."""
    verb = "would " if dry_run else ""  # action-line prefix: "would rename ..." vs "rename ..."

    # Guard rail: this must be a synthetic build root (has the CSV build log).
    if not os.path.isfile(os.path.join(root, MANIFEST_CSV)):
        raise SystemExit(
            f"refusing to migrate {root!r}: no {MANIFEST_CSV} (not a DataSynthesizer root)")

    print(f"Migrating root: {root}{' (dry-run, nothing will change)' if dry_run else ''}")

    # 1. Rename legacy prior/onset dirs to the standard names.
    for src_name, dst_name in RENAMES:
        src = os.path.join(root, src_name)
        dst = os.path.join(root, dst_name)
        if os.path.isdir(dst) and not os.path.isdir(src):
            print(f"  [1] rename {src_name} -> {dst_name}: already done (dest exists)")
        elif not os.path.isdir(src):
            print(f"  [1] rename {src_name} -> {dst_name}: SKIP (source missing, no dest)")
        elif os.path.exists(dst):
            print(f"  [1] rename {src_name} -> {dst_name}: SKIP (both exist — resolve manually)")
        else:
            print(f"  [1] {verb}rename {src_name} -> {dst_name}")
            if not dry_run:
                os.rename(src, dst)

    # 2. Generate manifest.json from the build log (reuse export_manifest).
    manifest, n_written, n_skipped = export_manifest.build_manifest(root)
    if dry_run:
        print(f"  [2] would write {MANIFEST_JSON}: {n_written} clips ({n_skipped} skipped non-ok)")
    else:
        path = export_manifest.write_manifest(root, manifest)
        print(f"  [2] wrote {MANIFEST_JSON}: {n_written} clips ({n_skipped} skipped non-ok) -> {path}")

    # 3. Copy arioso_split.json -> split.json (keep the original; preserves the split exactly).
    legacy_split = os.path.join(root, LEGACY_SPLIT)
    new_split = os.path.join(root, SPLIT_JSON)
    if os.path.exists(new_split):
        print(f"  [3] copy {LEGACY_SPLIT} -> {SPLIT_JSON}: already done (dest exists)")
    elif not os.path.isfile(legacy_split):
        print(f"  [3] copy {LEGACY_SPLIT} -> {SPLIT_JSON}: SKIP (no {LEGACY_SPLIT})")
    else:
        print(f"  [3] {verb}copy {LEGACY_SPLIT} -> {SPLIT_JSON}")
        if not dry_run:
            shutil.copy2(legacy_split, new_split)

    # 4. Delete deprecated VioPTT technique artifacts (always).
    tech = os.path.join(root, TECHNIQUE_DIR)
    if os.path.isdir(tech):
        size = _dir_size(tech)
        print(f"  [4] {verb}delete {TECHNIQUE_DIR} ({_fmt_size(size)})")
        if not dry_run:
            shutil.rmtree(tech)
    else:
        print(f"  [4] delete {TECHNIQUE_DIR}: already gone")

    # 5. Legacy pass-1 dirs: purge with the flag, else just report their sizes.
    for name in PURGE_DIRS:
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            print(f"  [5] legacy {name}: absent")
            continue
        size = _dir_size(path)
        if purge_legacy:
            print(f"  [5] {verb}purge {name} ({_fmt_size(size)})")
            if not dry_run:
                shutil.rmtree(path)
        else:
            print(f"  [5] legacy {name}: {_fmt_size(size)} kept (rerun with --purge-legacy to delete)")

    print("Done." if not dry_run else "Dry-run complete (nothing changed).")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the synthetic dataset root to migrate")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would happen; touch nothing")
    ap.add_argument("--purge-legacy", action="store_true",
                    help="also delete pass-1 prior_mel_quant / prior_mel_bend / prior_onset dirs")
    args = ap.parse_args()
    migrate(args.root, dry_run=args.dry_run, purge_legacy=args.purge_legacy)


if __name__ == "__main__":
    main()
