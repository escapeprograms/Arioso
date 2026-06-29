"""Synthesize the informed-prior sawtooth from an aligned violin MIDI file.

Arioso's flow-matching model maps an *informed prior* (a sawtooth rendered from
the MIDI score) to realistic violin audio. This module renders that prior in two
flavors and exposes the score onsets used to build the onset-mask training signal:

* ``render_prior`` — **quantized**: pitch = MIDI note number, constant frequency
  per note (the dense pitch bends are ignored). A clean, score-like sawtooth.
* ``render_prior_bend`` — **pitch-bend-following**: the instantaneous frequency
  tracks ``note.pitch + bend(t)`` (the interpolated pitch-wheel), so vibrato,
  slides and intonation are carried into the prior. Phase-accumulated so the
  time-varying frequency stays phase-continuous.
* ``note_onsets`` — the score onset times (seconds), shared by both renders.

Both use a **naive ``scipy.signal.sawtooth``** (no anti-aliasing; high notes
alias, acceptable for a prior) at **44.1 kHz mono**, matching the
violin-transcription dataset and the GT audio. The dataset's MIDI onset/offset
times are already time-aligned to the audio segment (t=0 == segment start), so the
rendered prior lines up with the GT audio.

Run as a script::

    python DataSynthesizer/synthesizePrior.py path/to/clip.mid -o clip_prior.wav
"""

from __future__ import annotations

import os

import numpy as np
import pretty_midi
from scipy.signal import sawtooth

from common.audio_io import normalize, write_pcm16

from .config import FADE_MS, SR

PB_RANGE_SEMITONES = 2.0  # MIDI default pitch-wheel range (the dataset sets no RPN)


def _fade_envelope(n: int, sr: int = SR, fade_ms: float = FADE_MS) -> np.ndarray:
    """Linear fade-in/out envelope of length ``n`` samples.

    A naive sawtooth that is simply gated on/off clicks at the note boundaries;
    short linear ramps at each end remove those transients. For notes shorter
    than two fade lengths the ramps are shrunk so they never overlap.
    """
    env = np.ones(n, dtype=np.float64)
    f = int(round(fade_ms / 1000.0 * sr))
    f = min(f, n // 2)
    if f > 0:
        ramp = np.linspace(0.0, 1.0, f, endpoint=False)
        env[:f] = ramp
        env[-f:] = ramp[::-1]
    return env


def _bend_semitones_at(inst: "pretty_midi.Instrument", t: np.ndarray) -> np.ndarray:
    """Interpolated pitch bend (in semitones) for instrument ``inst`` at times ``t``."""
    if not inst.pitch_bends:
        return np.zeros_like(t)
    bt = np.array([b.time for b in inst.pitch_bends])
    bv = np.array([b.pitch for b in inst.pitch_bends]) / 8192.0 * PB_RANGE_SEMITONES
    order = np.argsort(bt)
    return np.interp(t, bt[order], bv[order])


def render_prior(midi_path: str, sr: int = SR,
                 total_samples: int | None = None) -> np.ndarray:
    """Render a quantized naive-sawtooth prior from a MIDI file.

    Parameters
    ----------
    midi_path:
        Path to the aligned violin MIDI file.
    sr:
        Output sample rate.
    total_samples:
        Length of the output buffer in samples. Defaults to the MIDI end time;
        pass the GT audio length so the prior and target are sample-for-sample
        aligned (used by ``build_dataset.py``).

    Returns
    -------
    np.ndarray
        Mono ``float32`` waveform in [-1, 1].
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    if total_samples is None:
        total_samples = int(round(pm.get_end_time() * sr))
    out = np.zeros(total_samples, dtype=np.float64)

    # The dataset puts each note-stream on its own "instrument" track; summing
    # across all of them reproduces double-stops/polyphony.
    for inst in pm.instruments:
        for note in inst.notes:
            start = max(0, int(round(note.start * sr)))
            end = min(total_samples, int(round(note.end * sr)))
            n = end - start
            if n <= 1:
                continue
            freq = pretty_midi.note_number_to_hz(note.pitch)  # bends ignored
            t = np.arange(n) / sr
            amp = note.velocity / 127.0
            y = amp * sawtooth(2.0 * np.pi * freq * t)        # naive sawtooth
            y *= _fade_envelope(n, sr)
            out[start:end] += y

    return normalize(out).astype(np.float32)


def render_prior_bend(midi_path: str, sr: int = SR,
                      total_samples: int | None = None) -> np.ndarray:
    """Render a naive-sawtooth prior whose per-note frequency follows the pitch bend.

    The pitch-bend sibling of :func:`render_prior`: instead of a constant frequency
    per note, the instantaneous frequency tracks ``note.pitch + bend(t)`` (the
    linearly-interpolated pitch-wheel, +/-2 semitones), carrying vibrato, slides and
    intonation into the prior. The waveform is built by phase accumulation
    (``cumsum`` of the per-sample angular increment) so a time-varying frequency
    stays phase-continuous. Parameters and return value match :func:`render_prior`.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    if total_samples is None:
        total_samples = int(round(pm.get_end_time() * sr))
    out = np.zeros(total_samples, dtype=np.float64)

    for inst in pm.instruments:
        for note in inst.notes:
            start = max(0, int(round(note.start * sr)))
            end = min(total_samples, int(round(note.end * sr)))
            n = end - start
            if n <= 1:
                continue
            t = note.start + np.arange(n) / sr
            semis = note.pitch + _bend_semitones_at(inst, t)
            freq = 440.0 * 2.0 ** ((semis - 69.0) / 12.0)
            # phase accumulation keeps a time-varying freq phase-continuous
            phase = 2.0 * np.pi * np.cumsum(freq) / sr
            amp = note.velocity / 127.0
            y = amp * sawtooth(phase) * _fade_envelope(n, sr)
            out[start:end] += y

    return normalize(out).astype(np.float32)


def note_onsets(midi_path: str) -> np.ndarray:
    """Sorted, de-duplicated note onset times (seconds) across all instruments.

    These come straight from the score, so they are the exact onsets carried by the
    rendered prior; ``build_dataset`` uses them to build the onset-mask training
    signal at mel granularity.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    onsets = [note.start for inst in pm.instruments for note in inst.notes]
    return np.unique(np.asarray(onsets, dtype=np.float64))


def synthesize_to_file(midi_path: str, out_path: str, sr: int = SR,
                       total_samples: int | None = None) -> tuple[str, int, int]:
    """Render the prior for ``midi_path`` and write it to ``out_path`` (wav).

    Returns ``(out_path, n_samples, sr)``.
    """
    y = render_prior(midi_path, sr=sr, total_samples=total_samples)
    write_pcm16(out_path, y, sr)
    return out_path, len(y), sr


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Render a sawtooth prior from a violin MIDI file.")
    ap.add_argument("midi", help="path to the .mid file")
    ap.add_argument("-o", "--out",
                    help="output wav path (default: <midi>_prior.wav)")
    ap.add_argument("--sr", type=int, default=SR, help="sample rate (Hz)")
    args = ap.parse_args()

    out = args.out or (os.path.splitext(args.midi)[0] + "_prior.wav")
    path, n, sr = synthesize_to_file(args.midi, out, sr=args.sr)
    print(f"wrote {path}  ({n / sr:.2f} s @ {sr} Hz)")


if __name__ == "__main__":
    main()
