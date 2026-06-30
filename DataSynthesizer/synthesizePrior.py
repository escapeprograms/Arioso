"""Synthesize the informed-prior sawtooth from an aligned violin MIDI file.

Arioso's flow-matching model maps an *informed prior* (a sawtooth rendered from
the MIDI score) to realistic violin audio. This module renders that prior with a
small, composable pipeline rather than one monolithic function, so the several
independent axes of the prior (pitch flavor x source x envelope x body x level
match) can be swapped without a combinatorial explosion of variants.

The pipeline is the **Strategy** pattern: one concrete orchestrator
:class:`PriorSynth` keeps the per-note summation loop (so polyphony / double-stops
still sum) and delegates each step to an injected, ``Protocol``-typed component:

* :class:`PitchTrajectory` -> per-note f0 curve (Hz):  :class:`Quantized` (constant
  MIDI pitch, the baseline) | :class:`PitchBend` (pitch-wheel-following: vibrato,
  slides, intonation).
* :class:`SourceSynth` -> a unit-amplitude waveform from an f0 curve:
  :class:`NaiveSaw` (``scipy.signal.sawtooth``) | :class:`BandlimitedSaw` (polyBLEP
  band-limited; removes the fold-back aliasing a hard saw edge produces).
* :class:`Envelope` -> note shaping:  :class:`HardGate` ("rect", hard on/off) |
  :class:`Fade` (short linear anti-click ramp).
* :class:`BodyFilter` -> resonance shaping on the summed mix:  :class:`Identity`
  (the no-EQ baseline) | a future static body-EQ.
* :class:`Leveler` -> a single per-recording gain:  :class:`MaskedRMS` (scale so the
  RMS over *sounding* frames hits ``target_rms_dbfs`` -- the constant every GT was
  voiced-RMS-normalized to) | :class:`Peak` (legacy peak-normalize).

Both sources render from an f0 curve via an **exclusive prefix-sum phase**
(``phase[k] = sum(f0[:k]) / sr``), so a constant f0 reduces exactly to the old
``arange(n) * dt`` math: the named flavors below stay numerically identical to the
hand-written renderers they replace. Everything runs at **44.1 kHz mono**.

:func:`quantized_prior` assembles the spec-baseline pipeline from the
``PRIOR_*`` config constants. The module-level :func:`render_prior` /
:func:`render_prior_bend` wrappers reproduce the original peak-normalized priors
used by ``build_dataset`` and the wav CLI. The mel front-end is intentionally NOT
part of the pipeline: the dataset build shifts the waveform into GT alignment
*between* level-matching and mel, while inference mels directly, so the caller owns
that step (``DataSynthesizer.features.mel_for_training``).

Run as a script::

    python DataSynthesizer/synthesizePrior.py path/to/clip.mid -o clip_prior.wav
"""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np
import pretty_midi
from scipy.signal import sawtooth

from common.audio_io import write_pcm16
from common.config import DEFAULT_PEAK

from .config import (FADE_MS, PRIOR_ANTI_ALIAS, PRIOR_ENVELOPE, PRIOR_LEVEL_MATCH,
                     SR, TARGET_RMS_DBFS)

PB_RANGE_SEMITONES = 2.0  # MIDI default pitch-wheel range (the dataset sets no RPN)


# --- module-level helpers (flavor-independent) -----------------------------------

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


def _poly_blep(phase: np.ndarray, dt: np.ndarray | float) -> np.ndarray:
    """polyBLEP residual for a sawtooth at fractional ``phase`` in [0, 1), step ``dt``.

    Subtracting this from the naive ramp band-limits the discontinuity at each wrap,
    removing most of the aliasing a hard sawtooth edge produces. ``dt = f0 / sr`` is
    the per-sample phase increment (one period spans ``1/dt`` samples); it may be a
    scalar (constant pitch) or a per-sample array (pitch-bend), broadcast to ``phase``.
    """
    dt = np.broadcast_to(np.asarray(dt, dtype=phase.dtype), phase.shape)
    blep = np.zeros_like(phase)
    # Just after the discontinuity (start of the period).
    lo = (phase < dt) & (dt > 0.0)
    t = phase[lo] / dt[lo]
    blep[lo] = t + t - t * t - 1.0
    # Just before the discontinuity (end of the period).
    hi = (phase > 1.0 - dt) & (dt > 0.0)
    t = (phase[hi] - 1.0) / dt[hi]
    blep[hi] = t * t + t + t + 1.0
    return blep


def _phase_cycles(f0: np.ndarray, sr: int) -> np.ndarray:
    """Accumulated phase in *cycles* (turns) from an f0 curve, starting at 0.

    Exclusive prefix sum of the per-sample increment ``f0 / sr`` so the first sample
    sits at phase 0; a constant ``f0`` reduces exactly to ``arange(n) * (f0 / sr)``.
    """
    dt = f0 / sr
    return np.cumsum(dt) - dt


def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0


def note_onsets(midi_path: str) -> np.ndarray:
    """Sorted, de-duplicated note onset times (seconds) across all instruments.

    These come straight from the score, so they are the exact onsets carried by the
    rendered prior; the build passes use them to build the onset-mask training signal
    at mel granularity.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    onsets = [note.start for inst in pm.instruments for note in inst.notes]
    return np.unique(np.asarray(onsets, dtype=np.float64))


# --- pipeline component protocols -------------------------------------------------

class PitchTrajectory(Protocol):
    """Map one MIDI note to a per-sample f0 curve (Hz), length ``n``."""
    def __call__(self, inst: "pretty_midi.Instrument", note: "pretty_midi.Note",
                 n: int, sr: int) -> np.ndarray: ...


class SourceSynth(Protocol):
    """Render a unit-amplitude waveform from an f0 curve."""
    def __call__(self, f0: np.ndarray, sr: int) -> np.ndarray: ...


class Envelope(Protocol):
    """Shape one note's waveform (anti-click / gating)."""
    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray: ...


class BodyFilter(Protocol):
    """Resonance shaping applied to the summed mix."""
    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray: ...


class Leveler(Protocol):
    """A single per-recording gain: ``fit`` measures it, ``apply`` scales by it."""
    def fit(self, prior: np.ndarray, sounding: np.ndarray) -> float: ...
    def apply(self, prior: np.ndarray, gain: float) -> np.ndarray: ...


# --- pitch trajectories -----------------------------------------------------------

class Quantized:
    """Constant frequency = the MIDI note number (pitch bends ignored). The baseline."""
    def __call__(self, inst, note, n, sr) -> np.ndarray:
        return np.full(n, pretty_midi.note_number_to_hz(note.pitch), dtype=np.float64)


class PitchBend:
    """Pitch-wheel-following frequency: ``note.pitch + bend(t)`` (vibrato/slides)."""
    def __init__(self, pb_range_semitones: float = PB_RANGE_SEMITONES):
        self.pb_range_semitones = pb_range_semitones

    def _bend_semitones_at(self, inst, t: np.ndarray) -> np.ndarray:
        """Interpolated pitch bend (in semitones) for instrument ``inst`` at times ``t``."""
        if not inst.pitch_bends:
            return np.zeros_like(t)
        bt = np.array([b.time for b in inst.pitch_bends])
        bv = np.array([b.pitch for b in inst.pitch_bends]) / 8192.0 * self.pb_range_semitones
        order = np.argsort(bt)
        return np.interp(t, bt[order], bv[order])

    def __call__(self, inst, note, n, sr) -> np.ndarray:
        t = note.start + np.arange(n) / sr
        semis = note.pitch + self._bend_semitones_at(inst, t)
        return 440.0 * 2.0 ** ((semis - 69.0) / 12.0)


# --- source synths ----------------------------------------------------------------

class NaiveSaw:
    """Plain ``scipy.signal.sawtooth`` (no anti-aliasing; high notes alias)."""
    def __call__(self, f0: np.ndarray, sr: int) -> np.ndarray:
        return sawtooth(2.0 * np.pi * _phase_cycles(f0, sr))


class BandlimitedSaw:
    """polyBLEP band-limited sawtooth: same polarity/level as ``NaiveSaw`` (ramp -1->+1).

    Only the fold-back aliasing is removed; the saw's native ~-6 dB/oct harmonic ladder
    (the "excess" the model learns to remove) is kept on purpose.
    """
    def __call__(self, f0: np.ndarray, sr: int) -> np.ndarray:
        dt = f0 / sr
        phase = _phase_cycles(f0, sr) % 1.0
        naive = 2.0 * phase - 1.0
        return naive - _poly_blep(phase, dt)


# --- envelopes --------------------------------------------------------------------

class HardGate:
    """Rectangular note gating (hard on/off); clicks accepted as baseline content."""
    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        return wav


class Fade:
    """Short linear anti-click ramp at each note boundary."""
    def __init__(self, fade_ms: float = FADE_MS):
        self.fade_ms = fade_ms

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        return wav * _fade_envelope(len(wav), sr, self.fade_ms)


# --- body filters -----------------------------------------------------------------

class Identity:
    """No body shaping -- the no-EQ baseline. The seam for a future static body-EQ."""
    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        return wav


# --- levelers ---------------------------------------------------------------------

class MaskedRMS:
    """Scale so the prior's RMS over *sounding* frames hits ``target_rms_dbfs``.

    A single per-recording gain to the *constant* level every GT was voiced-RMS-
    normalized to. This prevents the ~30x-hotter saw from dominating OT-CFM transport
    while staying fully score-determined, so the prior is identical at train and
    inference.
    """
    def __init__(self, target_rms_dbfs: float = TARGET_RMS_DBFS):
        self.target_rms_dbfs = target_rms_dbfs

    def fit(self, prior: np.ndarray, sounding: np.ndarray) -> float:
        prior_rms = _rms(prior[sounding[: len(prior)]])
        if prior_rms <= 0.0:
            return 1.0
        return 10.0 ** (self.target_rms_dbfs / 20.0) / prior_rms

    def apply(self, prior: np.ndarray, gain: float) -> np.ndarray:
        return (prior * gain).astype(np.float32)


class Peak:
    """Legacy peak-normalize to ``target_peak`` (matches ``common.audio_io.normalize``)."""
    def __init__(self, target_peak: float = DEFAULT_PEAK):
        self.target_peak = target_peak

    def fit(self, prior: np.ndarray, sounding: np.ndarray) -> float:
        peak = float(np.max(np.abs(prior))) if prior.size else 0.0
        return self.target_peak / peak if peak > 0.0 else 1.0

    def apply(self, prior: np.ndarray, gain: float) -> np.ndarray:
        return (prior * gain).astype(np.float32)


# --- orchestrator -----------------------------------------------------------------

class PriorSynth:
    """Render a MIDI score to a level-matched prior waveform via injected components.

    One concrete class; the variation lives in the components passed to ``__init__``.
    ``render`` owns only the shared skeleton: iterate the score's notes (summing across
    instrument tracks for polyphony), synthesize and envelope each note, sum into the
    output buffer while recording the sounding mask, body-filter the mix, then fit and
    apply a single level-match gain.
    """
    def __init__(self, *, pitch: PitchTrajectory, source: SourceSynth,
                 envelope: Envelope, leveler: Leveler,
                 body: BodyFilter | None = None, sr: int = SR):
        self.pitch = pitch
        self.source = source
        self.envelope = envelope
        self.leveler = leveler
        self.body = body or Identity()
        self.sr = sr

    def render(self, midi_path: str, total_samples: int | None = None) -> np.ndarray:
        """Score -> mono float32 prior waveform.

        ``total_samples`` sets the output length; defaults to the MIDI end time. Pass
        the GT audio length so the prior and target are sample-for-sample aligned.
        """
        sr = self.sr
        pm = pretty_midi.PrettyMIDI(midi_path)
        if total_samples is None:
            total_samples = int(round(pm.get_end_time() * sr))
        out = np.zeros(total_samples, dtype=np.float64)
        sounding = np.zeros(total_samples, dtype=bool)

        # The dataset puts each note-stream on its own "instrument" track; summing
        # across all of them reproduces double-stops/polyphony.
        for inst in pm.instruments:
            for note in inst.notes:
                start = max(0, int(round(note.start * sr)))
                end = min(total_samples, int(round(note.end * sr)))
                n = end - start
                if n <= 1:
                    continue
                f0 = self.pitch(inst, note, n, sr)
                wav = self.source(f0, sr) * (note.velocity / 127.0)
                wav = self.envelope(wav, sr)
                out[start:end] += wav
                sounding[start:end] = True

        prior = self.body(out.astype(np.float32), sr)
        gain = self.leveler.fit(prior, sounding)
        return self.leveler.apply(prior, gain)


# --- factory + back-compat wrappers ----------------------------------------------

def quantized_prior(*, anti_alias: bool = PRIOR_ANTI_ALIAS,
                    envelope: str = PRIOR_ENVELOPE,
                    level_match: str = PRIOR_LEVEL_MATCH,
                    target_rms_dbfs: float = TARGET_RMS_DBFS,
                    sr: int = SR) -> PriorSynth:
    """Assemble the quantized prior pipeline (the spec baseline) from the config knobs."""
    return PriorSynth(
        pitch=Quantized(),
        source=BandlimitedSaw() if anti_alias else NaiveSaw(),
        envelope=Fade() if envelope == "fade" else HardGate(),
        leveler=MaskedRMS(target_rms_dbfs) if level_match == "masked_rms" else Peak(),
        sr=sr,
    )


def render_prior(midi_path: str, sr: int = SR,
                 total_samples: int | None = None) -> np.ndarray:
    """Quantized naive-sawtooth prior, peak-normalized (the legacy ``prior_mel_quant``)."""
    synth = quantized_prior(anti_alias=False, envelope="fade", level_match="peak", sr=sr)
    return synth.render(midi_path, total_samples)


def render_prior_bend(midi_path: str, sr: int = SR,
                      total_samples: int | None = None) -> np.ndarray:
    """Pitch-bend-following naive-sawtooth prior, peak-normalized (the legacy bend prior).

    The pitch-bend sibling of :func:`render_prior`: instantaneous frequency tracks
    ``note.pitch + bend(t)`` (the interpolated pitch-wheel, +/-2 semitones), carrying
    vibrato, slides and intonation into the prior.
    """
    synth = PriorSynth(pitch=PitchBend(), source=NaiveSaw(), envelope=Fade(),
                       leveler=Peak(), sr=sr)
    return synth.render(midi_path, total_samples)


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
