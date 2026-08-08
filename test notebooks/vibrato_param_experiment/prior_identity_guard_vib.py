"""Byte-identity guard for the vibrato-in-prior change (common/prior.py).

Disk score MIDIs carry no ``vib_params`` attribute, so the vibrato bake must leave
every disk-MIDI prior render *byte-identical* — both through the default
``pitch="quantized"`` path (untouched by design) and through the new
``pitch="quantized_vibrato"`` path (which must reduce exactly to ``Quantized``
when notes carry no params). Mirrors
``test notebooks/envelope_param_experiment/prior_identity_guard.py``:

* baseline mode (default): build a deterministic test MIDI, render via
  ``quantized_prior().render`` from a disk path, save MIDI + waveform ``.npy``.
  Run this ONCE with the PRE-change code.
* ``--check`` mode: re-render and assert ``np.array_equal`` with the baseline
  (exact, no tolerance). If the ``"quantized_vibrato"`` mode exists, also render
  through it and assert identity with the same baseline.
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
    r"\d22fd853-67b0-4041-a36d-46dc68bac1b7\scratchpad"
)
MIDI_PATH = os.path.join(SCRATCH, "prior_identity_guard_vib.mid")
NPY_PATH = os.path.join(SCRATCH, "prior_identity_guard_vib_baseline.npy")

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


def render(pitch: str = "quantized") -> np.ndarray:
    return quantized_prior(pitch=pitch).render(MIDI_PATH, total_samples=TOTAL_SAMPLES)


def _assert_identical(y: np.ndarray, base: np.ndarray, label: str) -> bool:
    if y.shape == base.shape and np.array_equal(y, base):
        print(f"[guard] PASS ({label}): byte-identical ({y.shape[0]} samples, "
              f"dtype={y.dtype})")
        return True
    diff = (float(np.max(np.abs(y[:len(base)] - base[:len(y)])))
            if y.size and base.size else "n/a")
    print(f"[guard] FAIL ({label}): differs. shapes {y.shape} vs {base.shape}; "
          f"max abs diff {diff}")
    return False


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
        base = np.load(NPY_PATH)
        ok = _assert_identical(render("quantized"), base, "pitch=quantized")
        try:
            y_vib = render("quantized_vibrato")
        except ValueError:
            print("[guard] SKIP (pitch=quantized_vibrato): mode not implemented yet")
        else:
            ok = _assert_identical(y_vib, base, "pitch=quantized_vibrato") and ok
        return 0 if ok else 1

    build_midi(MIDI_PATH)
    y = render()
    np.save(NPY_PATH, y)
    print(f"[guard] baseline saved: {NPY_PATH}")
    print(f"[guard] {y.shape[0]} samples, dtype={y.dtype}, "
          f"rms={float(np.sqrt(np.mean(y.astype(np.float64) ** 2))):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
