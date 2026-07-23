"""Byte-identity guard for the disk-MIDI prior render path.

DataSynthesizer builds 888 priors from on-disk score MIDIs; those MIDIs carry no
``env_dct`` attribute, so the env-bake change (common/prior.py render loop) must
leave their rendered samples *byte-identical*. This script pins that invariant:

* baseline mode (default): build a deterministic test MIDI (varied pitches /
  velocities / durations, incl. an overlap on separate instrument tracks), render
  it through ``quantized_prior().render`` from a disk path, and save both the MIDI
  and the rendered waveform (``.npy``) into the scratchpad dir.
* ``--check`` mode: re-render the saved MIDI and assert ``np.array_equal`` with the
  saved baseline (exact, no tolerance).

Run baseline once with the PRE-change code, then ``--check`` after editing.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pretty_midi

from common.config import SR
from common.prior import quantized_prior

SCRATCH = (
    r"C:\Users\archi\AppData\Local\Temp\claude"
    r"\c--Users-archi-Documents-Coding-stuff-AI-Violin"
    r"\701f7499-7f43-4e35-8b78-7843cae7e54a\scratchpad"
)
MIDI_PATH = os.path.join(SCRATCH, "prior_identity_guard.mid")
NPY_PATH = os.path.join(SCRATCH, "prior_identity_guard_baseline.npy")

# Deterministic score: (start_s, end_s, pitch, velocity, instrument_track).
# Track 1 overlaps track 0 in time (double-stop) to exercise the polyphony sum.
NOTES = [
    (0.00, 0.40, 55, 40, 0),
    (0.35, 1.10, 67, 100, 0),   # long, loud
    (0.50, 0.90, 71, 75, 1),    # overlaps the note above, separate track
    (1.20, 1.35, 60, 20, 0),    # very short, quiet
]
TOTAL_SAMPLES = int(round(1.6 * SR))


def build_midi(path: str) -> None:
    pm = pretty_midi.PrettyMIDI()
    tracks: dict[int, pretty_midi.Instrument] = {}
    for start, end, pitch, vel, tr in NOTES:
        inst = tracks.get(tr)
        if inst is None:
            inst = pretty_midi.Instrument(program=40)  # violin
            tracks[tr] = inst
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch,
                                           start=start, end=end))
    for tr in sorted(tracks):
        pm.instruments.append(tracks[tr])
    pm.write(path)


def render() -> np.ndarray:
    return quantized_prior().render(MIDI_PATH, total_samples=TOTAL_SAMPLES)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="re-render and assert byte-identity with the saved baseline")
    args = ap.parse_args()

    os.makedirs(SCRATCH, exist_ok=True)

    if args.check:
        if not os.path.exists(NPY_PATH):
            print("[guard] FAIL: no baseline .npy — run without --check first")
            return 1
        y = render()
        base = np.load(NPY_PATH)
        if y.shape == base.shape and np.array_equal(y, base):
            print(f"[guard] PASS: byte-identical ({y.shape[0]} samples, "
                  f"dtype={y.dtype})")
            return 0
        print(f"[guard] FAIL: differs. shapes {y.shape} vs {base.shape}; "
              f"max abs diff "
              f"{float(np.max(np.abs(y[:len(base)] - base[:len(y)]))) if y.size and base.size else 'n/a'}")
        return 1

    build_midi(MIDI_PATH)
    y = render()
    np.save(NPY_PATH, y)
    print(f"[guard] baseline saved: {NPY_PATH}")
    print(f"[guard] {y.shape[0]} samples, dtype={y.dtype}, "
          f"rms={float(np.sqrt(np.mean(y.astype(np.float64) ** 2))):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
