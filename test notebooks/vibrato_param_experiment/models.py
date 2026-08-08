"""Vibrato model registry + shared fitting machinery.

Prototype for a future ``common/vibrato_model.py``.  Only numpy/scipy; no torch,
no repo imports.

Design decisions
----------------
**Absolute-time parameters.**  Every oscillator parameter is in absolute units
(seconds, cents, Hz, Hz/s) rather than note-duration-relative fractions.  A
production conditioning signal has to be renderable from a score note *before*
the performance exists, and a 5.5 Hz vibrato is 5.5 Hz whether the note is 0.3 s
or 3 s long.  The single exception is ``M7_spline``, whose knots sit at
``tau in {0, T/3, 2T/3, T}`` -- that is deliberate: M7 is the complexity ceiling
used to bound how much structure is *in principle* recoverable, never a
production candidate.

**Jointly fitted affine nuisance baseline.**  The measured cents trace carries
the player's intonation offset and any slide/drift, which have nothing to do
with vibrato.  Every model is fitted as ``osc(t) + b0 + b1*t`` and the two
baseline parameters are reported separately and never counted as model params
(they enter ``k`` in the BIC identically for every model, so the comparison is
fair).  Fitting them jointly rather than detrending first matters: a
pre-detrend would let a half-cycle of vibrato leak into the slope.

**Free phase.**  Vibrato phase at note onset is not predictable from the score,
so phi is a free nuisance-like parameter -- but unlike the baseline it *is*
counted, because any production model that renders vibrato has to commit to
some phase (even if only "always 0"), and pretending it is free is cheating.

**Smoothstep gate.**  Real vibrato does not switch on discontinuously at the
onset delay ``d``; a hard step injects a spectral click and makes the residual
land on the delay in a non-smooth way, which trust-region solvers handle badly.
``g(t;d) = smoothstep(clamp((t-d)/w))`` with a fixed ``w = 30 ms`` gives a C1
ramp whose width is short compared to a vibrato period (~180 ms) so it barely
biases the depth, while making ``d`` a differentiable parameter.

**Chirp phase = integral of instantaneous frequency.**  For any model with a
non-constant IF, the argument of the sine is ``2*pi*integral(IF)``, not
``2*pi*IF(tau)*tau``.  The latter double-counts the sweep and produces a
systematically wrong rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lombscargle

# --- shared constants ---------------------------------------------------------
GATE_W = 0.030          # smoothstep gate ramp width, seconds (fixed, not fitted)
D_MAX = 120.0           # half-depth bound, cents
D_CLIP = 200.0          # hard clip on time-varying depth (M5), cents
F_LO, F_HI = 3.0, 11.0  # vibrato rate bounds, Hz
S_MAX = 20.0            # IF slope bound, Hz/s
GD_MAX = 400.0          # depth slope bound, cents/s
PHI_MAX = 4.0 * math.pi
B0_MAX, B1_MAX = 300.0, 600.0     # nuisance baseline bounds (cents, cents/s)
LD_LO, LD_HI = math.log(0.5), math.log(D_MAX)

# per-unit solver scaling
XS_TIME, XS_CENTS, XS_CENTS_S, XS_HZ, XS_HZ_S, XS_PHI, XS_LD = \
    0.1, 10.0, 100.0, 1.0, 2.0, 1.0, 1.0

#: corpus median fitted half-depth used by ``M0_global``; provisional value,
#: overwritten by :func:`set_global_depth` once M1 has been fitted everywhere.
GLOBAL_DEPTH = 14.0
GLOBAL_RATE = 5.5


def set_global_depth(d: float) -> None:
    """Set the fixed half-depth of ``M0_global`` (corpus median of M1's D)."""
    global GLOBAL_DEPTH
    GLOBAL_DEPTH = float(d)


# --- primitives ---------------------------------------------------------------

def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def gate(t: np.ndarray, d: float) -> np.ndarray:
    """C1 vibrato onset gate: 0 before ``d``, 1 after ``d + GATE_W``."""
    return smoothstep((t - d) / GATE_W)


def tau_of(t: np.ndarray, d: float) -> np.ndarray:
    return np.maximum(0.0, t - d)


def _pw_chirp_phase(tau, f0, s1, tk, s2):
    """Phase of a piecewise-linear-IF chirp, continuous at the knot ``tk``."""
    tk = max(float(tk), 1e-4)
    lo = f0 * tau + 0.5 * s1 * tau * tau
    dtk = tau - tk
    hi = f0 * tk + 0.5 * s1 * tk * tk + (f0 + s1 * tk) * dtk + 0.5 * s2 * dtk * dtk
    return np.where(tau < tk, lo, hi)


# --- model evaluators ---------------------------------------------------------

def _m0_none(p, t, T):
    return np.zeros_like(t)


def _m0_global(p, t, T):
    (phi,) = p
    return GLOBAL_DEPTH * np.sin(2 * np.pi * GLOBAL_RATE * t + phi)


def _m1a(p, t, T):
    D, f, phi = p
    return D * np.sin(2 * np.pi * f * t + phi)


def _m1(p, t, T):
    d, D, f, phi = p
    tau = tau_of(t, d)
    return D * gate(t, d) * np.sin(2 * np.pi * f * tau + phi)


def _m2(p, t, T):
    d, D, f0, s, phi = p
    tau = tau_of(t, d)
    ph = f0 * tau + 0.5 * s * tau * tau
    return D * gate(t, d) * np.sin(2 * np.pi * ph + phi)


def _m3(p, t, T):
    d, D, f0, s1, tk, s2, phi = p
    tau = tau_of(t, d)
    return D * gate(t, d) * np.sin(2 * np.pi * _pw_chirp_phase(tau, f0, s1, tk, s2) + phi)


def _m4(p, t, T):
    d, D, r, f, phi = p
    tau = tau_of(t, d)
    ramp = np.minimum(1.0, tau / max(r, 1e-3))
    return D * ramp * gate(t, d) * np.sin(2 * np.pi * f * tau + phi)


def _m5(p, t, T):
    d, D0, gD, f0, s, phi = p
    tau = tau_of(t, d)
    dep = np.clip(D0 + gD * tau, 0.0, D_CLIP)
    ph = f0 * tau + 0.5 * s * tau * tau
    return dep * gate(t, d) * np.sin(2 * np.pi * ph + phi)


def _m5nd(p, t, T):
    """M5 with the onset gate REMOVED (g == 1, tau == t).

    The delay ``d`` pins to 0 in the majority of M1 fits, so the gate mostly
    buys nothing while costing a parameter and suppressing the first 30 ms of
    real signal.  This is the production sweet-spot candidate.
    """
    D0, gD, f0, s, phi = p
    dep = np.clip(D0 + gD * t, 0.0, D_CLIP)
    return dep * np.sin(2 * np.pi * (f0 * t + 0.5 * s * t * t) + phi)


def _m8(p, t, T):
    """Linear depth ramp at CONSTANT frequency (no gate, no chirp).

    Isolates whether the frequency ramp ``s`` pays for itself once the depth is
    allowed to ramp -- M8 is M5nd minus ``s``.
    """
    D0, gD, f, phi = p
    dep = np.clip(D0 + gD * t, 0.0, D_CLIP)
    return dep * np.sin(2 * np.pi * f * t + phi)


def _m7(p, t, T):
    d = p[0]
    ld = np.asarray(p[1:5], dtype=float)
    fk = np.asarray(p[5:9], dtype=float)
    phi = p[9]
    tau = tau_of(t, d)
    knots = np.array([0.0, T / 3.0, 2.0 * T / 3.0, T])
    dseg = np.diff(knots)
    dseg = np.maximum(dseg, 1e-6)
    slopes = np.diff(fk) / dseg
    phi_k = np.concatenate([[0.0], np.cumsum(fk[:-1] * dseg + 0.5 * slopes * dseg ** 2)])
    j = np.clip(np.searchsorted(knots, tau, side="right") - 1, 0, 2)
    dt = tau - knots[j]
    ph = phi_k[j] + fk[j] * dt + 0.5 * slopes[j] * dt * dt
    dep = np.exp(np.interp(tau, knots, ld))
    return dep * gate(t, d) * np.sin(2 * np.pi * ph + phi)


# --- bounds / scaling by parameter name ---------------------------------------

def _bound(name: str, T: float):
    if name == "d":
        return 0.0, max(0.6 * T, 1e-3)
    if name in ("r", "tk"):
        return 0.01, max(T, 0.02)
    if name in ("D", "D0"):
        return 0.0, D_MAX
    if name == "gD":
        return -GD_MAX, GD_MAX
    if name in ("f", "f0", "f1", "f2", "f3"):
        return F_LO, F_HI
    if name in ("s", "s1", "s2"):
        return -S_MAX, S_MAX
    if name.startswith("ld"):
        return LD_LO, LD_HI
    if name == "phi":
        return -PHI_MAX, PHI_MAX
    raise KeyError(name)


def _scale(name: str) -> float:
    if name in ("d", "r", "tk"):
        return XS_TIME
    if name in ("D", "D0"):
        return XS_CENTS
    if name == "gD":
        return XS_CENTS_S
    if name in ("f", "f0", "f1", "f2", "f3"):
        return XS_HZ
    if name in ("s", "s1", "s2"):
        return XS_HZ_S
    if name.startswith("ld"):
        return XS_LD
    if name == "phi":
        return XS_PHI
    raise KeyError(name)


def knot_times(T: float) -> np.ndarray:
    """M7's IF/depth knots, in tau (see the module docstring on T-relativity)."""
    return np.array([0.0, T / 3.0, 2.0 * T / 3.0, T])


def _resolve(name: str, nm: dict, T: float) -> float:
    """Value for a child parameter absent from a parent's converged solution.

    The defaults are chosen to be *nesting-neutral*: whenever the child really
    does contain the parent, this reconstruction reproduces the parent's curve
    rather than merely borrowing its shared parameters.  That is the whole point
    of the warm-start ladder -- a child handed a degraded version of its
    parent's solution can lose to the parent for reasons that have nothing to do
    with the solver.  Concretely: the ramp time ``r`` pads to its lower bound
    (ramp off, not 150 ms on), the piecewise slopes ``s1/s2`` pad to the
    parent's single ``s``, and M7's knot values are the parent's linear IF and
    linear depth *sampled at the knots*.

    Fresh (Lomb-Scargle seeded) starts pass their exploratory values -- ``r =
    0.15``, ``tk = T/2`` -- in ``nm`` explicitly, so they are unaffected.
    """
    if name in nm:
        return float(nm[name])
    if name == "d":
        return 0.0
    if name in ("D", "D0"):
        return float(nm.get("D", nm.get("D0", 20.0)))
    if name in ("f", "f0"):
        return float(nm.get("f", nm.get("f0", GLOBAL_RATE)))
    if name in ("s", "s1", "s2"):
        return float(nm.get("s", nm.get("s1", 0.0)))
    if name == "gD":
        return 0.0
    if name == "r":
        return 0.01                      # lower bound == depth ramp disabled
    if name == "tk":
        return T / 2.0
    if name in ("f1", "f2", "f3"):       # parent's linear IF sampled at the knot
        f0 = float(nm.get("f", nm.get("f0", GLOBAL_RATE)))
        s = float(nm.get("s", nm.get("s1", 0.0)))
        return min(max(f0 + s * knot_times(T)[int(name[1])], F_LO), F_HI)
    if name.startswith("ld"):            # parent's linear depth sampled at the knot
        D = float(nm.get("D", nm.get("D0", 20.0)))
        gD = float(nm.get("gD", 0.0))
        v = D + gD * knot_times(T)[int(name[2])]
        return math.log(min(max(v, 0.5), D_MAX))
    if name == "phi":
        return float(nm.get("phi", 0.0))
    raise KeyError(name)


# --- init builders ------------------------------------------------------------
# ``seeds`` is a list of dicts {"f":Hz, "D":cents, "phi":rad} -- one per
# multi-start frequency candidate (Lomb-Scargle peaks + the stored auto rate).

def _init_from_seeds(param_names: Sequence[str]) -> Callable:
    def init(t, cents, seeds, T):
        out = []
        for s in seeds:
            # exploratory (non-neutral) starts for the shape params, per protocol
            nm = {"f": s["f"], "f0": s["f"], "D": s["D"], "D0": s["D"],
                  "phi": s["phi"], "r": 0.15, "tk": T / 2.0}
            out.append(np.array([_resolve(n, nm, T) for n in param_names], float))
        return out
    return init


@dataclass
class VibModel:
    name: str
    n_osc: int
    param_names: tuple
    eval_osc: Callable
    init: Callable = field(repr=False, default=None)

    def __post_init__(self):
        if self.init is None:
            self.init = _init_from_seeds(self.param_names)
        assert len(self.param_names) == self.n_osc, self.name

    def bounds(self, T: float):
        if not self.param_names:
            return np.zeros(0), np.zeros(0)
        b = [_bound(n, T) for n in self.param_names]
        return np.array([x[0] for x in b]), np.array([x[1] for x in b])

    @property
    def x_scale(self):
        return np.array([_scale(n) for n in self.param_names]) if self.param_names \
            else np.zeros(0)

    def named(self, osc) -> dict:
        return dict(zip(self.param_names, [float(v) for v in osc]))

    def from_named(self, nm: dict, T: float) -> np.ndarray:
        lo, hi = self.bounds(T)
        x = np.array([_resolve(n, nm, T) for n in self.param_names], float)
        return np.clip(x, lo, hi) if len(x) else x


REGISTRY: dict[str, VibModel] = {}


def _reg(m: VibModel) -> VibModel:
    REGISTRY[m.name] = m
    return m


_reg(VibModel("M0_none", 0, (), _m0_none, init=lambda t, c, s, T: [np.zeros(0)]))
_reg(VibModel("M0_global", 1, ("phi",), _m0_global))
_reg(VibModel("M1a_lfo3", 3, ("D", "f", "phi"), _m1a))
_reg(VibModel("M1_lfo4", 4, ("d", "D", "f", "phi"), _m1))
_reg(VibModel("M2_framp", 5, ("d", "D", "f0", "s", "phi"), _m2))
_reg(VibModel("M3_fpwl", 7, ("d", "D", "f0", "s1", "tk", "s2", "phi"), _m3))
_reg(VibModel("M4_dramp", 5, ("d", "D", "r", "f", "phi"), _m4))
_reg(VibModel("M5_both", 6, ("d", "D0", "gD", "f0", "s", "phi"), _m5))
_reg(VibModel("M5nd_rampboth5", 5, ("D0", "gD", "f0", "s", "phi"), _m5nd))
_reg(VibModel("M8_dramp4", 4, ("D0", "gD", "f", "phi"), _m8))
_reg(VibModel("M7_spline", 10,
              ("d", "ld0", "ld1", "ld2", "ld3", "f0", "f1", "f2", "f3", "phi"), _m7))

# ordered by complexity; ALSO the fit order, so every parent precedes its child
MODEL_ORDER = ["M0_none", "M0_global", "M1a_lfo3", "M1_lfo4", "M8_dramp4",
               "M2_framp", "M4_dramp", "M5nd_rampboth5", "M5_both",
               "M3_fpwl", "M7_spline"]

#: warm-start ladder: child -> parent.  Each child gets its parent's converged
#: solution (padded with neutral values for the extra params) as an EXTRA start,
#: which is what makes the nesting meaningful: a superset model that scores
#: worse than its parent is a solver failure, not evidence.
LADDER = {
    "M1_lfo4": "M1a_lfo3",
    "M2_framp": "M1_lfo4",
    "M3_fpwl": "M2_framp",
    "M4_dramp": "M1_lfo4",
    "M5_both": "M4_dramp",
    "M7_spline": "M5_both",
    # round 2: the gate-free branch warm-starts from the gate-free M1a
    "M5nd_rampboth5": "M1a_lfo3",
    "M8_dramp4": "M1a_lfo3",
}

#: Ladder pairs where the child genuinely CONTAINS the parent, so a worse
#: in-sample score really is a solver failure.  ``M1a -> M1`` is excluded on
#: purpose: M1's smoothstep onset gate is active even at ``d = 0`` (it ramps
#: over the first 30 ms), so M1 cannot reproduce M1a's ungated sine.  ``M4 ->
#: M5`` and ``M5 -> M7`` are excluded because M5's unsaturated linear depth
#: cannot express M4's saturating ramp, and M7's log-linear depth cannot
#: express M5's linear one.  ``M1 -> M4`` is included as *near*-nested: M4's
#: ramp time pads to its 0.01 s lower bound, leaving a 10 ms ramp that sits
#: entirely inside the 30 ms onset gate; measured worst-case disagreement is
#: ~0.6 cents on a 20-cent depth (~3%), far below the residuals in play.
#: The round-2 gate-free pair IS exactly nested in M1a: ``M8`` reduces to M1a at
#: ``gD = 0``, and ``M5nd`` reduces to M1a at ``gD = s = 0`` (and to M8 at
#: ``s = 0``), with no gate anywhere to break the identity.
NESTED_PAIRS = {("M1_lfo4", "M2_framp"), ("M2_framp", "M3_fpwl"),
                ("M1_lfo4", "M4_dramp"),
                ("M1a_lfo3", "M8_dramp4"), ("M1a_lfo3", "M5nd_rampboth5")}


# --- seeds --------------------------------------------------------------------

def ols_baseline(t: np.ndarray, cents: np.ndarray):
    A = np.stack([np.ones_like(t), t], axis=1)
    coef, *_ = np.linalg.lstsq(A, cents, rcond=None)
    return float(coef[0]), float(coef[1]), cents - A @ coef


LS_GRID = np.arange(3.5, 10.0 + 1e-9, 0.05)


def make_seeds(t, cents, auto_rate=None, n_ls=2, min_sep=0.3):
    """Multi-start seeds: top-``n_ls`` Lomb-Scargle peaks + the stored auto rate.

    Lomb-Scargle (not FFT) because the frame mask makes the samples unevenly
    spaced after despiking, and notes are far too short for a periodogram with
    usable resolution anyway.
    """
    b0, b1, r = ols_baseline(t, cents)
    freqs = []
    try:
        pg = lombscargle(t.astype(float), r.astype(float), 2 * np.pi * LS_GRID,
                         precenter=True)
        pg = np.nan_to_num(pg, nan=0.0, posinf=0.0, neginf=0.0)
        loc = np.where((pg[1:-1] >= pg[:-2]) & (pg[1:-1] >= pg[2:]))[0] + 1
        if len(loc) == 0:
            loc = np.array([int(np.argmax(pg))])
        loc = loc[np.argsort(pg[loc])[::-1]][:n_ls]
        freqs = [float(LS_GRID[i]) for i in loc]
    except Exception:  # noqa: BLE001
        freqs = []
    if not freqs:
        freqs = [GLOBAL_RATE]
    if auto_rate is not None and np.isfinite(auto_rate) and F_LO <= auto_rate <= F_HI:
        freqs.append(float(auto_rate))

    kept = []
    for f in freqs:
        if all(abs(f - k) >= min_sep for k in kept):
            kept.append(f)

    seeds = []
    for f in kept:
        A = np.stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)], axis=1)
        c, *_ = np.linalg.lstsq(A, r, rcond=None)
        # r ~ D*sin(2*pi*f*t + phi) = D cos(phi) sin(.) + D sin(phi) cos(.)
        D = float(np.hypot(c[0], c[1]))
        phi = float(np.arctan2(c[1], c[0]))
        seeds.append({"f": f, "D": min(D, D_MAX), "phi": phi})
    return seeds, (b0, b1)


# --- fitting ------------------------------------------------------------------

def fit(model: VibModel, t, cents, T, seeds, base0=(0.0, 0.0),
        extra_starts=(), loss="linear", f_scale=1.0):
    """Fit ``osc(params) + b0 + b1*t`` to ``cents`` on the already-masked frames.

    Returns a dict with the osc params, the (separately reported) baseline, SSE,
    RMSE and solver bookkeeping.  ``extra_starts`` are full ``[osc..., b0, b1]``
    vectors (used for the warm-start ladder).
    """
    t = np.asarray(t, float)
    cents = np.asarray(cents, float)
    n = len(t)
    k = model.n_osc
    lo_o, hi_o = model.bounds(T)
    lo = np.concatenate([lo_o, [-B0_MAX, -B1_MAX]])
    hi = np.concatenate([hi_o, [B0_MAX, B1_MAX]])
    xs = np.concatenate([model.x_scale, [XS_CENTS, XS_CENTS_S]])
    n_fit = k + 2

    def resid(x):
        osc = model.eval_osc(x[:k], t, T) if k else 0.0
        return cents - (osc + x[k] + x[k + 1] * t)

    starts = []
    for x0o in model.init(t, cents, seeds, T):
        starts.append(np.concatenate([np.asarray(x0o, float), np.asarray(base0, float)]))
    starts.extend(np.asarray(s, float) for s in extra_starts)

    best = None
    eps = 1e-9
    for x0 in starts:
        x0 = np.clip(x0, lo + eps, hi - eps)
        try:
            sol = least_squares(resid, x0, bounds=(lo, hi), method="trf",
                                x_scale=xs, max_nfev=200 * n_fit,
                                loss=loss, f_scale=f_scale)
            sse = float(np.sum(resid(sol.x) ** 2))
            x, nfev, ok = sol.x, int(sol.nfev), bool(sol.status > 0)
        except Exception:  # noqa: BLE001
            sse, x, nfev, ok = float(np.sum(resid(x0) ** 2)), x0, 0, False
        if best is None or sse < best["sse"]:
            best = {"sse": sse, "x": np.asarray(x, float), "nfev": nfev, "success": ok}

    x = best["x"]
    return {
        "model": model.name,
        "osc": x[:k].copy(),
        "baseline": (float(x[k]), float(x[k + 1])),
        "x": x,
        "sse": best["sse"],
        "rmse": math.sqrt(best["sse"] / max(n, 1)),
        "n": n,
        "k_osc": k,
        "nfev": best["nfev"],
        "success": best["success"],
        "n_starts": len(starts),
    }


def predict(model: VibModel, res: dict, t, T, with_baseline=True):
    osc = model.eval_osc(res["osc"], np.asarray(t, float), T) if model.n_osc \
        else np.zeros_like(np.asarray(t, float))
    if not with_baseline:
        return osc
    b0, b1 = res["baseline"]
    return osc + b0 + b1 * np.asarray(t, float)


def bic(sse: float, n: int, k_osc: int) -> float:
    """BIC with k = n_osc + 2 (the affine nuisance baseline counts for everyone)."""
    k = k_osc + 2
    return n * math.log(max(sse, 1e-12) / n) + k * math.log(n)


def lag1_rho(resid: np.ndarray) -> float:
    """Lag-1 autocorrelation of a fit residual, clipped to [0, 0.99]."""
    r = np.asarray(resid, float)
    if len(r) < 3:
        return 0.0
    r = r - r.mean()
    den = float(np.dot(r, r))
    if den <= 1e-12:
        return 0.0
    return float(min(max(float(np.dot(r[:-1], r[1:])) / den, 0.0), 0.99))


def n_effective(n: int, rho: float) -> float:
    """Autocorrelation-corrected sample size ``n (1-rho)/(1+rho)``."""
    return max(n * (1.0 - rho) / (1.0 + rho), 4.0)


def bic_eff(sse: float, n: int, k_osc: int, n_eff: float) -> float:
    """BIC evaluated on an EFFECTIVE sample size instead of the frame count.

    At 172 fps a 5.5 Hz vibrato is oversampled ~31x, so consecutive residuals
    are strongly correlated and plain BIC counts each frame as an independent
    observation.  That systematically under-prices parameters and is exactly why
    the most flexible models sweep the raw-BIC win rate.

    IMPORTANT -- ``n_eff`` must be COMMON to every model being compared on a
    given note, not each model's own residual autocorrelation.  Because
    ``ln(SSE/n) > 0`` whenever the residual exceeds 1 cent (always, here), the
    first term *shrinks* as ``n_eff`` shrinks, so a per-model ``n_eff`` hands
    the lowest score to whichever model leaves the MOST autocorrelated residual
    -- i.e. it rewards not fitting the vibrato at all.  Measured directly: with
    per-model rho, the zero-parameter ``M0_none`` won 92% of notes despite the
    worst RMSE of any model.  We therefore take rho from the residual of the
    richest model in the comparison (the best available estimate of the *noise*
    correlation rather than the leftover *signal* correlation) and apply the
    resulting ``n_eff`` to every model, which makes the pairwise difference a
    proper likelihood ratio scaled by the independent information available.
    """
    k = k_osc + 2
    return n_eff * math.log(max(sse, 1e-12) / n) + k * math.log(max(n_eff, 1.0001))


def aicc(sse: float, n: int, k_osc: int) -> float:
    k = k_osc + 2
    a = n * math.log(max(sse, 1e-12) / n) + 2 * k
    denom = n - k - 1
    return a + (2 * k * (k + 1) / denom if denom > 0 else float("inf"))


def pinned(model: VibModel, osc, T, tol=1e-3):
    """Names of parameters sitting on a bound (diagnostic for broken fits)."""
    lo, hi = model.bounds(T)
    out = []
    for nm, v, a, b in zip(model.param_names, osc, lo, hi):
        span = max(b - a, 1e-9)
        if (v - a) / span < tol or (b - v) / span < tol:
            out.append(nm)
    return out
