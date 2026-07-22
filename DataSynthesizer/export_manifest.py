"""Export the synthetic build log (``manifest.csv``) to the standard ``manifest.json``.

The DataSynthesizer's ``<out>/manifest.csv`` is the build's internal, append-only log
(one row per attempted clip, ``status`` recording success/failure of the download +
feature build). Arioso, however, consumes the *standard* dataset-root contract defined
in :mod:`common.dataset_schema` — a versioned ``manifest.json`` listing only complete,
trainable clips. This module bridges the two: it keeps the ``status == "ok"`` rows and
emits/updates ``<out>/manifest.json`` via :func:`common.dataset_schema.write_manifest`.

The synthetic root carries no conditioning tracks (``signals=[]``), the split key is
``"{composer}/{catalog}"`` and each clip's ``source`` block preserves the provenance
(book / performer / youtube id / start-end seconds / measured onset offset). Purely a
projection of the CSV — re-running regenerates identical content (idempotent), so it is
safe to run after every ``build_prior`` pass.

Run::

    python -m DataSynthesizer.export_manifest                 # default out-dir
    python -m DataSynthesizer.export_manifest --out-dir Data
"""

from __future__ import annotations

import csv
import os

from common.config import HOP_SIZE, N_MELS, SR
from common.dataset_schema import MANIFEST_SCHEMA_VERSION, write_manifest

from .config import DEFAULT_OUT

MANIFEST_CSV = "manifest.csv"  # DataSynthesizer's internal build log (input to this export)


def _num(s: str) -> int | float:
    """Parse a CSV numeric string to ``int`` when integral, else ``float``.

    The build log stores everything as strings; whole-second bounds (``start_sec`` =
    ``"4"``) become ``int``, fractional values (``offset_ms`` = ``"92.9"``) become
    ``float``. Keeps the JSON numerically faithful without forcing floats on integers.
    """
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        return float(s)


def build_manifest(out_dir: str = DEFAULT_OUT) -> tuple[dict, int, int]:
    """Project ``<out_dir>/manifest.csv`` into a standard manifest dict.

    Returns ``(manifest, n_written, n_skipped)`` where ``n_written`` is the count of
    ``status == "ok"`` clips placed in the manifest and ``n_skipped`` the count of
    non-ok rows dropped. Does not touch disk.
    """
    csv_path = os.path.join(out_dir, MANIFEST_CSV)
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    clips: dict[str, dict] = {}
    n_skipped = 0
    for row in rows:
        if row.get("status") != "ok":
            n_skipped += 1
            continue
        base = row["basename"]
        clips[base] = {
            "n_frames": int(row["n_frames"]),
            "n_samples": int(row["n_samples"]),
            "duration_s": _num(row["duration_sec"]),
            "piece": f"{row['composer']}/{row['catalog']}",
            "source": {
                "book": row["book"],
                "performer": row["performer"],
                "youtube_id": row["youtube_id"],
                "start_sec": _num(row["start_sec"]),
                "end_sec": _num(row["end_sec"]),
                "offset_ms": _num(row["offset_ms"]),
            },
        }

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "name": "synthetic",
        "frame": {"sr": SR, "hop": HOP_SIZE, "n_mels": N_MELS},
        "signals": [],
        "clips": clips,
    }
    return manifest, len(clips), n_skipped


def export_manifest(out_dir: str = DEFAULT_OUT) -> str:
    """Write ``<out_dir>/manifest.json`` from the build log; print a one-line summary."""
    manifest, n_written, n_skipped = build_manifest(out_dir)
    path = write_manifest(out_dir, manifest)
    print(f"Wrote {n_written} clips to {path} ({n_skipped} skipped non-ok)")
    return path


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_OUT,
                    help="dataset root holding manifest.csv (default: %(default)s)")
    args = ap.parse_args()
    export_manifest(out_dir=args.out_dir)


if __name__ == "__main__":
    main()
