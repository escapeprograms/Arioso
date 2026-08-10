"""Build the model's prior features over the dataset (one-time pass).

The DataSynthesizer ``data/`` tree already has GT audio, ``target_mel`` and a ``manifest.csv``.
This pass adds the two spec-faithful prior outputs the Arioso model trains on, without
re-downloading anything:

* ``<out>/prior_mel/<base>.npy`` — ``[N_MELS, T]`` float32 prior mel from the spec-faithful
  prior (shaped additive saw + masked-RMS level match), frame-aligned to ``target_mel`` (same ``T``).
* ``<out>/prior_wav/<base>.wav`` — that same GT-aligned prior as audio (mono PCM16 @ SR,
  ``n_samples`` long), for consumers that want to pitch-shift the waveform. An existing root
  without it needs no regeneration: a root lacking ``prior_wav`` is simply never augmented.
* ``<out>/onsets/<base>.npy`` — ``[K]`` int32 aligned onset frame indices, used by the clip
  enumerator.

The dir names come from the standard layout (``common.dataset_schema.DIR_PRIOR_MEL`` /
``DIR_PRIOR_WAV`` / ``DIR_ONSETS``). The prior is assembled by
:func:`common.prior.quantized_prior` from the ``PRIOR_*`` knobs (source / harmonic-law / tilt overridable per-run via the CLI flags below
for ablations). It reuses the
manifest's per-clip ``offset_ms`` (estimated once at build time from the onset cross-correlation) so
the prior lines up with the target exactly as the GT-length quantized prior did — no re-estimation.
Resumable + skip-existing, mirroring ``DataSynthesizer.build_dataset``.

Run::

    python -m DataSynthesizer.build_prior --limit 4      # smoke test
    python -m DataSynthesizer.build_prior                # full pass over status==ok clips
"""

from __future__ import annotations

import csv
import os
import sys
import traceback

import numpy as np

from common.audio_io import write_pcm16
from common.config import TARGET_RMS_DBFS
from common.dataset_schema import DIR_ONSETS, DIR_PRIOR_MEL, DIR_PRIOR_WAV
from common.onset_align import shift_samples
from common.prior import (PRIOR_ALPHA, PRIOR_CORNER_NC, PRIOR_CORNER_P, PRIOR_ENVELOPE,
                          PRIOR_HARMONIC_LAW, PRIOR_LEVEL_MATCH, PRIOR_SOURCE,
                          note_onsets, quantized_prior)
from common.vocoder import mel_frames

from .config import DEFAULT_DATASET, DEFAULT_OUT, HOP, SR


def _midi_path(dataset_root: str, book: str, basename: str) -> str:
    return os.path.join(dataset_root, book, basename + ".mid")


def process_clip(row: dict, out_dir: str, dataset_root: str, *,
                 source: str = PRIOR_SOURCE, harmonic_law: str = PRIOR_HARMONIC_LAW,
                 alpha: float = PRIOR_ALPHA, corner_nc: float = PRIOR_CORNER_NC,
                 corner_p: float = PRIOR_CORNER_P, envelope: str = PRIOR_ENVELOPE,
                 level_match: str = PRIOR_LEVEL_MATCH,
                 target_rms_dbfs: float = TARGET_RMS_DBFS,
                 overwrite: bool = False) -> str:
    """Build prior mel + prior wav + onset frames for one manifest row. Returns a status string."""
    base = row["basename"]
    prior_dir = os.path.join(out_dir, DIR_PRIOR_MEL)
    prior_wav_dir = os.path.join(out_dir, DIR_PRIOR_WAV)
    onset_dir = os.path.join(out_dir, DIR_ONSETS)
    os.makedirs(prior_dir, exist_ok=True)
    os.makedirs(prior_wav_dir, exist_ok=True)
    os.makedirs(onset_dir, exist_ok=True)
    prior_mel_path = os.path.join(prior_dir, base + ".npy")
    prior_wav_path = os.path.join(prior_wav_dir, base + ".wav")
    onset_path = os.path.join(onset_dir, base + ".npy")

    if (not overwrite and os.path.isfile(prior_mel_path) and os.path.isfile(prior_wav_path)
            and os.path.isfile(onset_path)):
        return "exists"

    midi = _midi_path(dataset_root, row["book"], base)
    n_samples = int(row["n_samples"])            # == len(GT); no need to reload the audio
    applied = float(row["offset_ms"]) / 1000.0   # same shift the original prior used

    # Render + level-match (on the score-aligned, unshifted prior) -> shift into GT
    # alignment. The gain is scale-invariant to the shift, so the order is safe.
    synth = quantized_prior(source=source, harmonic_law=harmonic_law, alpha=alpha,
                            corner_nc=corner_nc, corner_p=corner_p, envelope=envelope,
                            level_match=level_match, target_rms_dbfs=target_rms_dbfs)
    prior = synth.render(midi, total_samples=n_samples)
    prior = shift_samples(prior, applied)

    # Persist the POST-shift buffer only: the pre-shift render is in score time, so writing it
    # here would hand consumers a prior_wav silently misaligned with gt/ and prior_mel.
    # PCM16 clamps silently past ±1.0, which would make the wav a different signal from the
    # float the mel below is computed from — fail loudly instead.
    peak = float(np.max(np.abs(prior)))
    if peak > 1.0:
        raise AssertionError(f"{base}: prior peak {peak:.3f} > 1.0 would clip on PCM16 write")
    write_pcm16(prior_wav_path, prior, SR)

    prior_mel = mel_frames(prior)
    np.save(prior_mel_path, prior_mel)

    n_frames = prior_mel.shape[-1]
    onsets = note_onsets(midi) + applied
    frames = np.round(onsets * SR / HOP).astype(np.int64)
    frames = np.unique(frames[(frames >= 0) & (frames < n_frames)]).astype(np.int32)
    np.save(onset_path, frames)
    return "ok"


def build(out_dir: str = DEFAULT_OUT, dataset_root: str = DEFAULT_DATASET, *,
          source: str = PRIOR_SOURCE, harmonic_law: str = PRIOR_HARMONIC_LAW,
          alpha: float = PRIOR_ALPHA, corner_nc: float = PRIOR_CORNER_NC,
          corner_p: float = PRIOR_CORNER_P, envelope: str = PRIOR_ENVELOPE,
          level_match: str = PRIOR_LEVEL_MATCH,
          target_rms_dbfs: float = TARGET_RMS_DBFS,
          limit: int | None = None, overwrite: bool = False) -> None:
    """Pass over ``manifest.csv`` (status==ok rows), building the prior features."""
    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "ok"]
    if limit:
        rows = rows[:limit]

    n_ok = n_skip = n_fail = 0
    for i, row in enumerate(rows, 1):
        base = row["basename"]
        try:
            status = process_clip(row, out_dir, dataset_root, source=source,
                                  harmonic_law=harmonic_law, alpha=alpha, corner_nc=corner_nc,
                                  corner_p=corner_p, envelope=envelope, level_match=level_match,
                                  target_rms_dbfs=target_rms_dbfs, overwrite=overwrite)
            n_ok += status == "ok"
            n_skip += status == "exists"
            print(f"[{i}/{len(rows)}] {status:8s} {base}")
        except Exception as exc:  # noqa: BLE001 - log and continue, like build_dataset
            n_fail += 1
            print(f"[{i}/{len(rows)}] FAILED   {base}: {exc}", file=sys.stderr)
            traceback.print_exc(limit=1)

    print(f"\nDone: {n_ok} built, {n_skip} existed, {n_fail} failed -> "
          f"{os.path.join(out_dir, DIR_PRIOR_MEL)}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET)
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N ok clips (smoke testing)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-render even if outputs already exist")
    # Prior toggles (default = spec baseline) so ablations don't need code edits.
    ap.add_argument("--source", choices=("blep_saw", "naive_saw", "additive"), default=PRIOR_SOURCE,
                    help="prior source: polyBLEP saw | naive saw | shaped additive bank")
    ap.add_argument("--harmonic-law", choices=("alpha", "corner"), default=PRIOR_HARMONIC_LAW,
                    help="additive only: power-law tilt vs rounded-corner ladder")
    ap.add_argument("--alpha", type=float, default=PRIOR_ALPHA,
                    help="tilt exponent (also the corner law's below-corner tilt)")
    ap.add_argument("--corner-nc", type=float, default=PRIOR_CORNER_NC, help="corner harmonic n_c")
    ap.add_argument("--corner-p", type=float, default=PRIOR_CORNER_P, help="corner order p")
    ap.add_argument("--envelope", choices=("rect", "fade"), default=PRIOR_ENVELOPE)
    ap.add_argument("--level-match", choices=("masked_rms", "peak"), default=PRIOR_LEVEL_MATCH)
    args = ap.parse_args()

    build(out_dir=args.out_dir, dataset_root=args.dataset_root,
          source=args.source, harmonic_law=args.harmonic_law, alpha=args.alpha,
          corner_nc=args.corner_nc, corner_p=args.corner_p, envelope=args.envelope,
          level_match=args.level_match, limit=args.limit, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
