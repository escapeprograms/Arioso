"""Per-note vibrato parameterization — the compact ``vib_params`` descriptor baked into the prior.

The prior's ``quantized`` pitch trajectory holds every note at its exact MIDI
frequency, so the one thing a violin recording always has and the prior never does
is vibrato. This module measures a note's F0 modulation (pyin) and compresses it to
**5 numbers** that ``common.prior.QuantizedVibrato`` replays as a per-sample
frequency ratio.

Production model — ``rampboth5`` (experiment id ``M5nd_rampboth5``), params
``[D0, gD, f0, s, phi]``::

    D(t)  = clip(D0 + gD*t, 0, 200)              half-depth, cents
    f(t)  = f0 + s*t                             instantaneous rate, Hz
    Phi(t)= 2*pi*(f0*t + s*t**2/2) + phi         phase = 2*pi * integral of f
    osc(t)= D(t) * sin(Phi(t))                   deviation from the score pitch, cents

Design decisions
----------------
**Absolute-time parameters.** Every parameter is in absolute units (cents, cents/s,
Hz, Hz/s, radians) measured from the note's onset — never note-duration-relative
fractions. Two reasons: a 5.5 Hz vibrato is 5.5 Hz whether the note is 0.3 s or 3 s
long, and, decisively, a note-boundary edit in the Labeler must not silently change
what the stored parameters *mean*. (Trimming a note's end does still truncate the
rendered oscillation, which is correct; the front-end therefore clears the params on
a timing edit rather than reinterpreting them.)

**Chirp phase is the integral of the instantaneous frequency.** With a non-constant
IF the argument of the sine is ``2*pi*integral(f)``, not ``2*pi*f(t)*t`` — the latter
double-counts the sweep and yields a systematically wrong rate.

**Jointly fitted but discarded nuisance baseline.** The measured cents trace also
carries the player's intonation offset and any slide/drift, which are not vibrato.
Every fit solves for ``osc(t) + b0 + b1*t`` and then **throws b0/b1 away** — they are
never stored, never rendered. Fitting them jointly (rather than detrending first)
matters: a pre-detrend lets a half-cycle of vibrato leak into the slope.

**Free per-note phase.** Vibrato phase at onset is not predictable from the score, so
``phi`` is fitted per note and stored like any other parameter.

**No onset-delay gate.** The experiment's gated variants pinned their delay to 0 on
the majority of notes, so the gate cost a parameter and suppressed the first 30 ms of
real signal for nothing.

Provenance and the model tiering
--------------------------------
Selected by ``test notebooks/vibrato_param_experiment`` over the 921 fittable
vibrato-marked notes of the verified gt_arky corpus (37 clips). Median in-sample
cents RMSE against a **7.72** no-model floor (the residual of the affine baseline
alone), i.e. what 3/4/5 numbers buy:

===========  ======  ===========  ==========================================
model        params  median RMSE  when to pick it
===========  ======  ===========  ==========================================
rampboth5         5         4.29  the default; best fit, highest raw-BIC win
                                  rate (21.7%) among production candidates
dramp4            4         4.95  preferred if a smaller model is ever
                                  wanted — costs 0.65 cents and gives a
                                  better-behaved rate estimate (one constant
                                  ``f`` instead of a chirp)
lfo3              3         5.57  minimal; also rampboth5's warm-start parent
===========  ======  ===========  ==========================================

Two caveats worth carrying forward:

* **The fitted ``f0``/``s`` are curve-fit parameters, not a vibrato-rate
  measurement.** Correlation between the fitted rate and the MUSC-based
  ``auto.vibrato_rate_hz`` detector is only 0.089, and ``f0`` sits pinned at one of
  its 3/11 Hz bounds on ~30% of notes — the chirp trades rate against depth-ramp to
  minimize residual. Anything that wants a *human-readable* rate (UI text, stats,
  heuristics) must read ``auto.vibrato_rate_hz``; ``vib_params`` is for rendering.
* **These parameters describe a note that was measured, not a forecast.** The
  block-holdout metric (fit the first 60% of a note, extrapolate onto the last 40%)
  rises monotonically with complexity for *every* model, so the unregularized ramps
  are only meaningful over the span they were fitted on. That is why the Labeler
  clears a note's params on any boundary edit rather than reinterpreting them.

numpy + librosa + scipy only — deliberately torch-free so any producer can import it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lombscargle, medfilt

from common.config import SR

# --- F0 measurement -------------------------------------------------------------
# hop 256 / frame 1024: vibrato lives at 4.5-7 Hz (140-220 ms period), so the usual
# 2048 frame (46 ms) averages over ~a quarter cycle and flattens ~14% of the depth.
# 1024 (23 ms) still resolves the open-G fundamental (196 Hz, ~4.5 periods/window).
# hop 256 -> 172.27 fps, ~31 samples per vibrato cycle.
F0_HOP, F0_FRAME = 256, 1024
F0_FMIN, F0_FMAX = 175.0, 2100.0      # below open G / above MIDI 83 with slip margin
# pyin decodes onto a discrete pitch grid of ``resolution`` semitones. The librosa
# default 0.1 is a 10-cent staircase — against a corpus half-depth of ~11 cents that
# squares off the modulation and puts a ~3-cent floor under every fit. 0.05 (5-cent
# bins, ~1.4 cent quantization sigma) is what the experiment measured with.
F0_RESOLUTION = 0.05

#: The single swap point: which model everything (fit, render, MIDI attach) uses.
VIB_MODEL = "rampboth5"
#: Wire-but-off fallback: synthesize params from the ``auto`` vibrato detector for
#: notes whose fit failed. Off — an unfitted note renders with no vibrato at all,
#: which is honest; a detector-shaped sine is a guess wearing a measurement's clothes.
VIB_FALLBACK = False

# --- fitting protocol constants (ported verbatim from the experiment) -------------
CENTS_ABS_MAX = 200.0      # |cents| beyond this is an octave/harmonic slip
DESPIKE_K = 5.0            # x 1.4826 x MAD of the 3-frame median-filter deviation
MIN_DUR_S = 0.25           # shorter notes carry <1.5 vibrato cycles
MIN_VALID = 12             # minimum unmasked frames
MIN_VOICED_FRAC = 0.5      # minimum unmasked fraction of the note's frames

D_MAX = 120.0              # half-depth bound, cents
D_CLIP = 200.0             # hard clip on the time-varying depth, cents
F_LO, F_HI = 3.0, 11.0     # vibrato rate bounds, Hz
S_MAX = 20.0               # rate-slope bound, Hz/s
GD_MAX = 400.0             # depth-slope bound, cents/s
PHI_MAX = 4.0 * math.pi
B0_MAX, B1_MAX = 300.0, 600.0     # nuisance baseline bounds (cents, cents/s)

# Per-unit solver scaling (least_squares x_scale): without it the Hz and cents/s
# parameters live on wildly different scales and trf's trust region is useless.
XS_CENTS, XS_CENTS_S, XS_HZ, XS_HZ_S, XS_PHI = 10.0, 100.0, 1.0, 2.0, 1.0

LS_GRID = np.arange(3.5, 10.0 + 1e-9, 0.05)   # Lomb-Scargle seed search grid, Hz
N_LS_PEAKS = 2                                 # multi-start: top-2 periodogram peaks
SEED_MIN_SEP_HZ = 0.3                          # dedupe seed frequencies closer than this
ROUND_DP = 4                                   # stored parameter precision


# --- F0 track / note slice ------------------------------------------------------

def f0_hz_track(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Whole-clip pyin F0: ``[T]`` float64 Hz, NaN where unvoiced.

    Run **once per clip**, not per note: a per-note extraction would restart pyin's
    Viterbi decode at every note boundary and produce edge artifacts exactly where
    the vibrato onset is measured. **Slow**: measured at ~3.5x the clip duration on
    this machine (a 24 s clip took 84 s) — hop 256 with the 5-cent pitch grid is
    ~4x the cost of librosa's defaults. Budget minutes, not seconds, per clip.
    """
    import librosa  # lazy: keeps import cost off consumers that only render
    f0, _voiced_flag, _voiced_prob = librosa.pyin(
        np.asarray(y, dtype=np.float32), fmin=F0_FMIN, fmax=F0_FMAX, sr=sr,
        frame_length=F0_FRAME, hop_length=F0_HOP, resolution=F0_RESOLUTION)
    return np.asarray(f0, dtype=np.float64)


def note_cents_from_track(f0_hz: np.ndarray, pitch: int, start_s: float, end_s: float,
                          sr: int = SR, hop: int = F0_HOP):
    """Slice one note out of a clip F0 track; return ``(t_rel_s, cents)``.

    Frames ``[round(start*sr/hop), round(end*sr/hop))`` clamped to the track. ``t_rel_s``
    is the *true* frame time minus ``start_s``, so the integer rounding of the slice
    bounds changes which frames are included but never shifts the timebase the
    parameters are anchored to. Cents are relative to the **scored** pitch (the
    intonation offset is absorbed by the discarded nuisance baseline, so the fitter
    should see the raw trace). Unvoiced frames stay NaN.
    """
    T = len(f0_hz)
    i0 = int(np.clip(round(start_s * sr / hop), 0, T))
    i1 = int(np.clip(round(end_s * sr / hop), 0, T))
    if i1 <= i0:
        return np.zeros(0), np.zeros(0)
    idx = np.arange(i0, i1)
    t = idx * (hop / sr) - start_s
    ref = 440.0 * 2.0 ** ((int(pitch) - 69) / 12.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cents = 1200.0 * np.log2(f0_hz[i0:i1] / ref)
    return t, cents


def mask_note(t: np.ndarray, cents: np.ndarray) -> np.ndarray:
    """Valid-frame boolean mask: drop unvoiced, drop octave slips, then despike.

    The despike step matters more than it looks: one pyin octave-halving frame inside
    a note is a 1200-cent outlier an L2 fit will chase with a bogus depth. Threshold
    is ``DESPIKE_K`` robust sigma (1.4826 x MAD) of the deviation from a 3-frame
    median filter.
    """
    cents = np.asarray(cents, dtype=np.float64)
    ok = np.isfinite(cents) & (np.abs(cents) <= CENTS_ABS_MAX)
    idx = np.where(ok)[0]
    if len(idx) < 3:
        return ok
    c = cents[idx]
    dev = c - medfilt(c, 3)
    mad = float(np.median(np.abs(dev - np.median(dev))))
    if mad > 1e-9:
        ok[idx[np.abs(dev) > DESPIKE_K * 1.4826 * mad]] = False
    return ok


# --- oscillator evaluators ------------------------------------------------------

def _eval_rampboth5(p, t):
    D0, gD, f0, s, phi = p
    dep = np.clip(D0 + gD * t, 0.0, D_CLIP)
    return dep * np.sin(2 * np.pi * (f0 * t + 0.5 * s * t * t) + phi)


def _eval_lfo3(p, t):
    D, f, phi = p
    return D * np.sin(2 * np.pi * f * t + phi)


def _eval_dramp4(p, t):
    D0, gD, f, phi = p
    dep = np.clip(D0 + gD * t, 0.0, D_CLIP)
    return dep * np.sin(2 * np.pi * f * t + phi)


# --- per-parameter bounds / scaling / seed padding ------------------------------

_BOUNDS = {
    "D": (0.0, D_MAX), "D0": (0.0, D_MAX),
    "gD": (-GD_MAX, GD_MAX),
    "f": (F_LO, F_HI), "f0": (F_LO, F_HI),
    "s": (-S_MAX, S_MAX),
    "phi": (-PHI_MAX, PHI_MAX),
}
_SCALES = {"D": XS_CENTS, "D0": XS_CENTS, "gD": XS_CENTS_S,
           "f": XS_HZ, "f0": XS_HZ, "s": XS_HZ_S, "phi": XS_PHI}
#: Value a parameter takes when it is absent from the seed / warm-start source.
#: Chosen to be *nesting-neutral*: ``gD = s = 0`` turns rampboth5 into lfo3 exactly,
#: so a warm start from lfo3 reproduces lfo3's curve rather than merely borrowing
#: its shared parameters.
_NEUTRAL = {"gD": 0.0, "s": 0.0}


def _seed_vector(param_names, nm: dict) -> np.ndarray:
    out = []
    for name in param_names:
        if name in nm:
            out.append(float(nm[name]))
        elif name in ("D", "D0"):
            out.append(float(nm.get("D", nm.get("D0", 20.0))))
        elif name in ("f", "f0"):
            out.append(float(nm.get("f", nm.get("f0", 5.5))))
        else:
            out.append(_NEUTRAL[name])
    return np.asarray(out, dtype=np.float64)


# --- seeds ----------------------------------------------------------------------

def _ols_baseline(t: np.ndarray, cents: np.ndarray):
    A = np.stack([np.ones_like(t), t], axis=1)
    coef, *_ = np.linalg.lstsq(A, cents, rcond=None)
    return float(coef[0]), float(coef[1]), cents - A @ coef


def _make_seeds(t: np.ndarray, cents: np.ndarray, auto_rate_hz=None):
    """Multi-start seeds ``[{f, D, phi}]`` + the OLS baseline start ``(b0, b1)``.

    Lomb-Scargle (not FFT) because despiking leaves the frames unevenly spaced, and
    notes are far too short for a periodogram with usable resolution anyway. The top
    ``N_LS_PEAKS`` peaks of the detrended residual seed the rate; the stored ``auto``
    detector rate is appended when it is in band; near-duplicates are dropped. Each
    kept rate gets its D/phi from a sin/cos projection of the residual.
    """
    b0, b1, resid = _ols_baseline(t, cents)
    freqs: list[float] = []
    try:
        # Mean-centered explicitly rather than via the deprecated ``precenter=True``
        # (scipy documents the two as exactly equivalent).
        y = resid.astype(float)
        pg = lombscargle(t.astype(float), y - y.mean(), 2 * np.pi * LS_GRID)
        pg = np.nan_to_num(pg, nan=0.0, posinf=0.0, neginf=0.0)
        loc = np.where((pg[1:-1] >= pg[:-2]) & (pg[1:-1] >= pg[2:]))[0] + 1
        if len(loc) == 0:
            loc = np.array([int(np.argmax(pg))])
        loc = loc[np.argsort(pg[loc])[::-1]][:N_LS_PEAKS]
        freqs = [float(LS_GRID[i]) for i in loc]
    except Exception:  # noqa: BLE001 - a degenerate trace must not kill the fit
        freqs = []
    if not freqs:
        freqs = [5.5]
    if auto_rate_hz is not None and np.isfinite(auto_rate_hz) and F_LO <= auto_rate_hz <= F_HI:
        freqs.append(float(auto_rate_hz))

    kept: list[float] = []
    for f in freqs:
        if all(abs(f - k) >= SEED_MIN_SEP_HZ for k in kept):
            kept.append(f)

    seeds = []
    for f in kept:
        A = np.stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)], axis=1)
        c, *_ = np.linalg.lstsq(A, resid, rcond=None)
        # resid ~ D*sin(2*pi*f*t + phi) = D cos(phi) sin(.) + D sin(phi) cos(.)
        seeds.append({"f": f, "D": min(float(np.hypot(c[0], c[1])), D_MAX),
                      "phi": float(np.arctan2(c[1], c[0]))})
    return seeds, (b0, b1)


# --- the model registry ---------------------------------------------------------

@dataclass(frozen=True)
class VibratoModel:
    """One vibrato parameterization: its evaluator, bounds and fitting protocol.

    ``fit`` takes the **already-sliced** note (its own masking is done by
    :func:`fit_note_vibrato`) and returns the oscillator parameters only — the two
    jointly-fitted nuisance baseline parameters are discarded here and never leave
    this module. ``warm_from`` names a simpler model whose converged solution is
    added as an extra start (padded with :data:`_NEUTRAL` values), which is what
    makes the nesting meaningful: a superset model that scores worse than the model
    it contains is a solver failure, not evidence.
    """
    name: str
    n_params: int
    param_names: tuple[str, ...]
    eval_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    warm_from: str | None = None

    def eval(self, params, t) -> np.ndarray:
        """Oscillator deviation in cents at times ``t`` (seconds since note onset)."""
        p = np.asarray(params, dtype=np.float64)
        if len(p) != self.n_params:
            raise ValueError(
                f"{self.name} takes {self.n_params} params, got {len(p)}")
        return self.eval_fn(p, np.asarray(t, dtype=np.float64))

    @property
    def bounds(self):
        lo = np.array([_BOUNDS[n][0] for n in self.param_names], dtype=np.float64)
        hi = np.array([_BOUNDS[n][1] for n in self.param_names], dtype=np.float64)
        return lo, hi

    @property
    def x_scale(self) -> np.ndarray:
        return np.array([_SCALES[n] for n in self.param_names], dtype=np.float64)

    def _solve(self, t, cents, starts):
        """Best (lowest-SSE) ``least_squares`` solution over the multi-start list.

        Solves for ``[osc..., b0, b1]`` jointly; returns ``(x, sse)`` with the
        baseline still attached (the caller strips it).
        """
        k = self.n_params
        lo_o, hi_o = self.bounds
        lo = np.concatenate([lo_o, [-B0_MAX, -B1_MAX]])
        hi = np.concatenate([hi_o, [B0_MAX, B1_MAX]])
        xs = np.concatenate([self.x_scale, [XS_CENTS, XS_CENTS_S]])
        n_fit = k + 2

        def resid(x):
            return cents - (self.eval_fn(x[:k], t) + x[k] + x[k + 1] * t)

        best_x, best_sse = None, np.inf
        eps = 1e-9
        for x0 in starts:
            x0 = np.clip(np.asarray(x0, dtype=np.float64), lo + eps, hi - eps)
            try:
                sol = least_squares(resid, x0, bounds=(lo, hi), method="trf",
                                    x_scale=xs, max_nfev=200 * n_fit, loss="linear")
                x, sse = sol.x, float(np.sum(resid(sol.x) ** 2))
            except Exception:  # noqa: BLE001 - a failed start must not kill the fit
                x, sse = x0, float(np.sum(resid(x0) ** 2))
            if sse < best_sse:
                best_x, best_sse = np.asarray(x, dtype=np.float64), sse
        return best_x, best_sse

    def fit(self, t, cents, dur_s: float, auto_rate_hz=None) -> list[float] | None:
        """Fit this model to one masked note; return its params (4 dp) or ``None``.

        ``t`` / ``cents`` are the surviving frames (see :func:`mask_note`), ``t`` in
        seconds since the note onset. ``dur_s`` is the note's full score duration.
        ``auto_rate_hz`` (the Labeler's stored detector estimate) seeds one extra
        start when it is in the 3-11 Hz band.
        """
        t = np.asarray(t, dtype=np.float64)
        cents = np.asarray(cents, dtype=np.float64)
        if len(t) < MIN_VALID or dur_s < MIN_DUR_S:
            return None
        seeds, base0 = _make_seeds(t, cents, auto_rate_hz)
        starts = [np.concatenate([_seed_vector(self.param_names, s), base0])
                  for s in seeds]
        if self.warm_from:
            parent = MODELS[self.warm_from]
            p_params = parent.fit(t, cents, dur_s, auto_rate_hz)
            if p_params is not None:
                nm = dict(zip(parent.param_names, p_params))
                starts.append(np.concatenate(
                    [_seed_vector(self.param_names, nm), base0]))
        if not starts:
            return None
        x, _sse = self._solve(t, cents, starts)
        if x is None or not np.all(np.isfinite(x)):
            return None
        # b0/b1 (x[-2:]) are the nuisance baseline: measured, then thrown away.
        return [round(float(v), ROUND_DP) for v in x[:self.n_params]]


MODELS: dict[str, VibratoModel] = {
    m.name: m for m in (
        # The production model. 5 params: half-depth ramp + rate ramp + phase.
        VibratoModel("rampboth5", 5, ("D0", "gD", "f0", "s", "phi"),
                     _eval_rampboth5, warm_from="lfo3"),
        # Plain constant-depth constant-rate LFO (experiment M1a) — the cheap
        # alternate, and rampboth5's warm-start parent (it is exactly nested at
        # gD = s = 0).
        VibratoModel("lfo3", 3, ("D", "f", "phi"), _eval_lfo3),
        # Depth ramp at constant rate (experiment M8) — rampboth5 minus the chirp.
        VibratoModel("dramp4", 4, ("D0", "gD", "f", "phi"), _eval_dramp4,
                     warm_from="lfo3"),
    )
}


def _model(name: str) -> VibratoModel:
    m = MODELS.get(name)
    if m is None:
        raise ValueError(
            f"unknown vibrato model {name!r} (expected one of {sorted(MODELS)})")
    return m


# --- public API -----------------------------------------------------------------

def fit_note_vibrato(f0_hz: np.ndarray, pitch: int, start_s: float, end_s: float,
                     sr: int = SR, model: str = VIB_MODEL,
                     auto_rate_hz=None) -> list[float] | None:
    """Fit one note's vibrato from a clip F0 track; ``None`` if it is unfittable.

    Rejects (returns ``None``) a note shorter than :data:`MIN_DUR_S`, one with fewer
    than :data:`MIN_VALID` surviving frames, or one whose surviving fraction is below
    :data:`MIN_VOICED_FRAC` — in all three cases the trace cannot support 5 parameters
    and a fit would be noise dressed as vibrato.
    """
    m = _model(model)
    dur_s = float(end_s) - float(start_s)
    if dur_s < MIN_DUR_S:
        return None
    t, cents = note_cents_from_track(f0_hz, pitch, start_s, end_s, sr=sr)
    if len(t) == 0:
        return None
    ok = mask_note(t, cents)
    n_valid = int(ok.sum())
    if n_valid < MIN_VALID or n_valid / len(t) < MIN_VOICED_FRAC:
        return None
    return m.fit(t[ok], cents[ok], dur_s, auto_rate_hz)


def vibrato_cents(params, t, model: str = VIB_MODEL) -> np.ndarray:
    """Pitch deviation in cents at times ``t`` (seconds since the note onset)."""
    return _model(model).eval(params, t)


def vibrato_ratio(params, n_samples: int, sr: int = SR,
                  model: str = VIB_MODEL) -> np.ndarray:
    """Per-sample frequency **ratio** ``2**(cents/1200)``, length ``n_samples`` (float64).

    Sampled at ``t = arange(n)/sr`` — the same onset-anchored timebase the parameters
    were fitted on (frame 0 == note onset == phase ``phi``), so the rendered phase
    matches the measured one. float64 to match the prior render loop's accumulation.
    """
    t = np.arange(int(n_samples), dtype=np.float64) / float(sr)
    return 2.0 ** (vibrato_cents(params, t, model=model) / 1200.0)


def vibrato_from_auto(rate_hz, extent_cents) -> list[float] | None:
    """:data:`VIB_MODEL` params synthesized from the ``auto`` detector's rate/extent.

    A flat sine at the detected rate with half-depth ``extent/2`` and zero phase — the
    fallback for vibrato-marked notes whose fit failed. Gated off by
    :data:`VIB_FALLBACK` (see its note); returns ``None`` when the detector output is
    missing or out of band.
    """
    try:
        rate = float(rate_hz)
        extent = float(extent_cents)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(rate) and np.isfinite(extent)):
        return None
    if not (F_LO <= rate <= F_HI) or extent <= 0.0:
        return None
    if VIB_MODEL != "rampboth5":       # keep the shape honest if the default moves
        raise ValueError(f"vibrato_from_auto has no mapping for {VIB_MODEL!r}")
    return [round(min(0.5 * extent, D_MAX), ROUND_DP), 0.0,
            round(rate, ROUND_DP), 0.0, 0.0]
