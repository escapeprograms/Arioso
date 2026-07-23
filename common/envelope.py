"""Per-note energy envelope — the compact ``env_dct`` descriptor baked into the prior.

The gt_arky MIDI velocity carries almost no dynamic information (it correlates at
only r~0.22-0.26 with actual note loudness). Instead we describe each note's
loudness *shape* with three coefficients of a cosine (DCT-I-style) basis fit to the
A-weighted power-dB loudness envelope over the note's frames — the fidelity knee
found by the ``envelope_param_experiment`` (~1.6 dB median reconstruction RMSE,
matching the MIDI-DDSP ~3-scalar precedent). The convention:

* The envelope is **power-dB**: ``10*log10`` of the A-weighting-filtered mean STFT
  power per frame (A-weighting applied in the *linear* power domain, DDSP-style).
* The basis is ``phi_k(i) = cos(k*pi*(i+0.5)/L)`` for ``k = 0, 1, 2`` over the note's
  ``L`` frames — so the three coefficients read as **c0 = level**, **c1 = tilt**
  (overall rise/fall across the note), **c2 = arch** (swell / dip curvature).
* :func:`env_gain` turns the coefficients back into a per-sample *linear amplitude*
  gain (``10**(dB/20)``) the prior render loop multiplies the source waveform by.
  Only cross-note level differences and within-note shape survive the prior's
  whole-clip ``MaskedRMS`` level match, which cancels the absolute dB offset.

Threading into the prior: producers set ``note.env_dct = [c0, c1, c2]`` on in-memory
``pretty_midi.Note`` objects; disk-loaded MIDIs have no such attribute and fall to
the exact legacy ``velocity/127`` path (so DataSynthesizer's priors stay identical).

numpy + librosa only — deliberately torch-free so any producer can import it.
"""

from __future__ import annotations

import librosa
import numpy as np

from common.config import HOP_SIZE as HOP, N_FFT, SR

EPS = 1e-10       # dB-domain floor (matches the envelope experiment)
ENV_K = 3         # number of cosine-basis coefficients (c0 level, c1 tilt, c2 arch)

# --- velocity -> env_dct (UI-anchored map, 2026-07-23) ---------------------------
# Studio (and MIDI fallbacks) derive a flat env from velocity when no measured
# envelope exists: env_dct = [VEL_C0_A * velocity + VEL_C0_B, 0, 0]. The map anchors
# velocity 1..127 onto the envelope lanes' full display range (-30 .. +16 dB) so a
# velocity-bar drag moves a note's level at exactly the same dB-per-pixel rate as the
# envelope control points (both lanes share that display range). Velocity is a UI
# *control*, not a measurement, so the perceptual span wins over the data-faithful
# OLS fit used previously (2026-07-22 calibration: A=0.043620, B=1.6462 from 24
# clips / 2589 notes, r(vel, c0)=0.251 — kept here for provenance; its ~5.5 dB total
# span made the velocity bar feel ~8x slower than the envelope handles). vel 100 ->
# ~6.1 dB, still near the corpus median c0 (5.4 dB). Only relative levels matter —
# the prior's MaskedRMS level match cancels the absolute c0 offset.
VEL_C0_A = 46.0 / 126.0            # dB per velocity step: (16 - -30) / (127 - 1)
VEL_C0_B = -30.0 - VEL_C0_A        # vel 1 -> -30 dB, vel 127 -> +16 dB


def loudness_db(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Per-frame A-weighted loudness in power-dB, ``[T]`` float (hop 512, n_fft 2048).

    STFT power, A-weighting applied in the **linear power** domain
    (``10**(A_weighting/10)``), mean over frequency bins, then ``10*log10(x + EPS)``.
    Mirrors the ``envelope_param_experiment`` extraction exactly.
    """
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP, center=True)
    power = np.abs(S) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    w_lin = 10.0 ** (librosa.A_weighting(freqs) / 10.0)     # A-weight in linear power
    avg_power = np.mean(power * w_lin[:, None], axis=0)      # DDSP-style A-weighted mean
    return 10.0 * np.log10(avg_power + EPS)


def fit_env_dct(env_db: np.ndarray) -> list[float]:
    """Least-squares fit of the 3 cosine coefficients to a note's dB envelope.

    Basis ``phi_k(i) = cos(k*pi*(i+0.5)/L)`` for ``k = 0..2`` over the note's ``L``
    frames. Always returns length 3 (zero-padded). Degenerate handling: ``L < 1`` ->
    ``[0, 0, 0]``; ``L == 1`` -> ``[env[0], 0, 0]``; otherwise fit only the
    ``k <= min(2, L-1)`` well-posed coefficients and zero-pad the rest.
    """
    env_db = np.asarray(env_db, dtype=np.float64)
    L = len(env_db)
    if L < 1:
        return [0.0, 0.0, 0.0]
    if L == 1:
        return [float(env_db[0]), 0.0, 0.0]
    k_max = min(2, L - 1)                       # fit k = 0..k_max
    i = np.arange(L, dtype=np.float64)
    k = np.arange(k_max + 1, dtype=np.float64)
    basis = np.cos(np.outer(i + 0.5, k) * np.pi / L)   # [L, k_max+1]
    coeffs, *_ = np.linalg.lstsq(basis, env_db, rcond=None)
    out = [0.0, 0.0, 0.0]
    for j in range(k_max + 1):
        out[j] = float(coeffs[j])
    return out


def eval_env_dct(coeffs, u: np.ndarray) -> np.ndarray:
    """Evaluate the fitted envelope (dB) at normalized positions ``u`` in ``[0, 1)``.

    ``c0 + c1*cos(pi*u) + c2*cos(2*pi*u)`` — the continuous form of the cosine basis.
    """
    c0, c1, c2 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    u = np.asarray(u, dtype=np.float64)
    return c0 + c1 * np.cos(np.pi * u) + c2 * np.cos(2.0 * np.pi * u)


def env_gain(coeffs, n_samples: int) -> np.ndarray:
    """Per-sample LINEAR amplitude gain of length ``n_samples`` (float64).

    Samples the fitted dB envelope at the ``n`` note-relative midpoints
    ``u = (arange(n)+0.5)/n`` and converts power-dB to amplitude (``10**(dB/20)``).
    float64 to match the prior render loop's float64 accumulation.
    """
    u = (np.arange(n_samples, dtype=np.float64) + 0.5) / n_samples
    return 10.0 ** (eval_env_dct(coeffs, u) / 20.0)


def note_env_from_wav(loud_db: np.ndarray, start_s: float, end_s: float,
                      sr: int = SR, hop: int = HOP) -> list[float]:
    """Fit a note's ``env_dct`` from a clip's precomputed ``loudness_db`` track.

    Frames are sliced with the repo's ``frame_of`` convention (``round(t*sr/hop)``),
    clamped to ``[0, len(loud_db)]``; a zero-width slice is widened to one frame. The
    slice is then fit with :func:`fit_env_dct`.
    """
    T = len(loud_db)
    f0 = int(round(start_s * sr / hop))
    f1 = int(round(end_s * sr / hop))
    f0 = int(np.clip(f0, 0, T))
    f1 = int(np.clip(f1, 0, T))
    if f1 <= f0:
        f1 = min(T, f0 + 1)
    return fit_env_dct(loud_db[f0:f1])


def velocity_to_env_dct(velocity: int) -> list[float]:
    """Flat env_dct derived from MIDI velocity: ``[A*vel + B, 0, 0]`` (see module note).

    Constants :data:`VEL_C0_A` / :data:`VEL_C0_B` are OLS-calibrated on the verified
    gt_arky corpus; only the slope is meaningful (MaskedRMS cancels the offset).
    """
    return [VEL_C0_A * float(velocity) + VEL_C0_B, 0.0, 0.0]
