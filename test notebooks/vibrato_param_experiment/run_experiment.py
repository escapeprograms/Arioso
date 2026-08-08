"""Offline vibrato-parameterization experiment (read-only analysis).

Question: what is the cheapest per-note vibrato parameterization that explains
the measured F0 modulation of the gt_arky corpus?  The answer picks the
production conditioning signal (a future ``common/vibrato_model.py``), so the
criterion is BIC -- in-sample fit bought at a stated parameter price -- not raw
RMSE, which monotonically favours the biggest model.

Data: the 37 verified clips of ``Data/datasets/gt_arky/manifest.json``, notes
with the top-level ``"vibrato": true`` flag (the human label, not ``auto``).
F0 comes from :mod:`f0_cache` (pyin, hop 256 / frame 1024); models and the
shared fitter live in :mod:`models`.

Nothing in the pipeline is modified.  Outputs (results.json + 4 PNGs) land next
to this script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, PROJ)
sys.path.insert(0, HERE)

import f0_cache as F0                                              # noqa: E402
import models as M                                                 # noqa: E402
from f0_cache import CLIPS_DIR, MANIFEST, SR, F0_HOP, FPS          # noqa: E402

# --- experiment constants -----------------------------------------------------
CENTS_ABS_MAX = 200.0      # |cents| beyond this is an octave/harmonic slip
DESPIKE_K = 5.0            # x 1.4826 x MAD on the median-filter deviation
MIN_DUR_S = 0.25
MIN_VALID = 12
MIN_VOICED_FRAC = 0.5
BUCKETS = ["<0.35s", "0.35-0.7s", ">0.7s", "all"]
SOFT_L1_FSCALE = 10.0
BLOCK_FRAC = 0.6           # block-holdout train fraction (first 60% of frames)


def bucket_of(dur: float) -> str:
    if dur < 0.35:
        return "<0.35s"
    if dur <= 0.7:
        return "0.35-0.7s"
    return ">0.7s"


# --- masking ------------------------------------------------------------------

def mask_note(t, cents):
    """Valid-frame mask: drop unvoiced, octave slips, then median-filter spikes.

    The despike step matters more than it looks: a single pyin octave-halving
    frame inside a note is a 1200-cent outlier that an L2 fit will chase with a
    bogus depth.  Threshold is 5 robust sigma of the 3-point median deviation.
    """
    ok = np.isfinite(cents) & (np.abs(cents) <= CENTS_ABS_MAX)
    idx = np.where(ok)[0]
    if len(idx) < 3:
        return ok, 0
    c = cents[idx]
    dev = c - medfilt(c, 3)
    mad = float(np.median(np.abs(dev - np.median(dev))))
    n_spike = 0
    if mad > 1e-9:
        bad = np.abs(dev) > DESPIKE_K * 1.4826 * mad
        n_spike = int(bad.sum())
        ok[idx[bad]] = False
    return ok, n_spike


# --- per-note fitting ---------------------------------------------------------

def fit_ladder(model_names, t, cents, T, auto_rate, loss="linear", f_scale=1.0):
    """Fit every requested model to one masked note, honouring the warm-start ladder."""
    seeds, base0 = M.make_seeds(t, cents, auto_rate)
    out = {}
    for name in M.MODEL_ORDER:
        if name not in model_names:
            continue
        mod = M.REGISTRY[name]
        extras = []
        par = M.LADDER.get(name)
        if par and par in out:
            osc0 = mod.from_named(M.REGISTRY[par].named(out[par]["osc"]), T)
            extras.append(np.concatenate([osc0, np.asarray(out[par]["baseline"])]))
        t0 = time.perf_counter()
        res = M.fit(mod, t, cents, T, seeds, base0, extras, loss=loss, f_scale=f_scale)
        res["fit_ms"] = 1000.0 * (time.perf_counter() - t0)
        out[name] = res
    return out


def eval_holdout(model_names, t, cents, T, auto_rate):
    """Fit on even-indexed valid frames, score RMSE on the odd-indexed ones.

    CAVEAT (repeated in the report): at 172 fps a ~5.5 Hz vibrato is sampled
    ~31x per cycle, so an interleaved split leaves the held-out frames almost
    perfectly interpolable from the training frames.  This holdout therefore
    barely penalizes complexity -- it detects catastrophic overfit (a model
    chasing pyin noise) and nothing subtler.  BIC is the primary criterion.
    """
    ev, od = t[0::2], t[1::2]
    if len(ev) < MIN_VALID or len(od) < 3:
        return {}
    fits = fit_ladder(model_names, ev, cents[0::2], T, auto_rate)
    out = {}
    for name, res in fits.items():
        pred = M.predict(M.REGISTRY[name], res, od, T)
        out[name] = float(np.sqrt(np.mean((cents[1::2] - pred) ** 2)))
    return out


def eval_block_holdout(model_names, t, cents, T, auto_rate, frac=BLOCK_FRAC):
    """Fit on the FIRST 60% of a note's valid frames, score on the LAST 40%.

    This is the metric that actually discriminates.  The interleaved holdout
    above and plain BIC are both compromised by the ~31x oversampling of a
    vibrato cycle: neighbouring frames are nearly the same observation, so an
    over-flexible model can interpolate the held-out frames and buy its extra
    parameters at a discount.  A contiguous block forces genuine EXTRAPOLATION
    in time -- the model must have captured the note's oscillatory structure,
    not memorized pyin's noise.

    Note ``T`` stays the FULL note duration (it is score-known a priori), which
    is deliberate: M7's knots sit at T/3, 2T/3, T, so two of its four knots fall
    in the unseen block.  A model whose parameterization is duration-relative
    rather than absolute-time should be punished here, and that is the point.
    """
    n = len(t)
    cut = int(round(frac * n))
    if cut < MIN_VALID or n - cut < 4:
        return {}
    fits = fit_ladder(model_names, t[:cut], cents[:cut], T, auto_rate)
    out = {}
    for name, res in fits.items():
        pred = M.predict(M.REGISTRY[name], res, t[cut:], T)
        out[name] = float(np.sqrt(np.mean((cents[cut:] - pred) ** 2)))
    return out


def fit_one_note(nd, ladder_names):
    """All three fits for a single note (parallelizable unit of work)."""
    T = float(nd["dur"])
    t, c, ar = nd["t"], nd["cents"], nd["auto_rate"]
    return (fit_ladder(ladder_names, t, c, T, ar),
            eval_holdout(ladder_names, t, c, T, ar),
            eval_block_holdout(ladder_names, t, c, T, ar))


def _fit_chunk(args):
    chunk, ladder_names, depth = args
    M.set_global_depth(depth)
    return [fit_one_note(nd, ladder_names) for nd in chunk]


# --- aggregation --------------------------------------------------------------

def _agg(notes, model_names, key_full="rmse_full"):
    """Per-model aggregates for every duration bucket."""
    table = {}
    base_med = {}
    for bk in BUCKETS:
        sub = [n for n in notes if bk == "all" or n["bucket"] == bk]
        if sub and "M0_none" in model_names:
            base_med[bk] = float(np.median([n["fits"]["M0_none"][key_full] for n in sub]))
    for name in model_names:
        mod = M.REGISTRY[name]
        entry = {"n_osc": mod.n_osc, "k_bic": mod.n_osc + 2, "by_bucket": {}}
        for bk in BUCKETS:
            sub = [n for n in notes if bk == "all" or n["bucket"] == bk]
            if not sub:
                continue
            rf = np.array([n["fits"][name][key_full] for n in sub])
            ho = np.array([n["fits"][name]["rmse_holdout"] for n in sub
                           if n["fits"][name].get("rmse_holdout") is not None])
            bh = np.array([n["fits"][name]["rmse_block_holdout"] for n in sub
                           if n["fits"][name].get("rmse_block_holdout") is not None])
            bics = np.array([n["fits"][name]["bic"] for n in sub])
            beff = np.array([n["fits"][name]["bic_eff"] for n in sub])
            rho = np.array([n["fits"][name]["rho"] for n in sub])
            aic = np.array([n["fits"][name]["aicc"] for n in sub])
            wins = np.mean([n["bic_winner"] == name for n in sub])
            wins_eff = np.mean([n["bic_eff_winner"] == name for n in sub])
            ms = np.array([n["fits"][name]["fit_ms"] for n in sub])
            entry["by_bucket"][bk] = {
                "n": len(sub),
                "median_rmse_full": float(np.median(rf)),
                "p90_rmse_full": float(np.percentile(rf, 90)),
                "median_rmse_holdout": float(np.median(ho)) if len(ho) else None,
                "n_holdout": int(len(ho)),
                "median_rmse_block_holdout": float(np.median(bh)) if len(bh) else None,
                "p90_rmse_block_holdout": (float(np.percentile(bh, 90))
                                           if len(bh) else None),
                "n_block_holdout": int(len(bh)),
                "mean_bic": float(np.mean(bics)),
                "median_bic": float(np.median(bics)),
                "mean_bic_eff": float(np.mean(beff)),
                "bic_eff_win_rate": float(wins_eff),
                "median_resid_lag1_rho": float(np.median(rho)),
                "mean_aicc": float(np.mean(aic)),
                "bic_win_rate": float(wins),
                "pct_explained": (float(1.0 - np.median(rf) / base_med[bk])
                                  if base_med.get(bk) else None),
                "median_fit_ms": float(np.median(ms)),
            }
        table[name] = entry
    return table


def _dist(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if not len(v):
        return None
    return {"n": int(len(v)), "p10": float(np.percentile(v, 10)),
            "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)),
            "mean": float(np.mean(v))}


# --- main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", default=None, help="comma-separated clip ids")
    ap.add_argument("--limit", type=int, default=None, help="first N clips")
    ap.add_argument("--cache-dir", default=F0.DEFAULT_CACHE_DIR)
    ap.add_argument("--models", default=None,
                    help="comma-separated model ids (default: all)")
    ap.add_argument("--jobs", type=int, default=6, help="parallel pyin workers")
    args = ap.parse_args()

    manifest = json.load(open(MANIFEST))
    clip_ids = sorted(manifest["clips"].keys())
    if args.clips:
        want = [c.strip() for c in args.clips.split(",") if c.strip()]
        clip_ids = [c for c in clip_ids if c in want]
    if args.limit:
        clip_ids = clip_ids[:args.limit]

    model_names = list(M.MODEL_ORDER)
    if args.models:
        want = {c.strip() for c in args.models.split(",") if c.strip()}
        model_names = [m for m in model_names if m in want]
    ladder_names = [m for m in model_names if m != "M0_global"]

    print(f"[vib] {len(clip_ids)} clips | models: {','.join(model_names)}")
    print(f"[vib] SR={SR} hop={F0_HOP} fps={FPS:.2f} cache={args.cache_dir}")

    F0.warm_cache(clip_ids, args.cache_dir, CLIPS_DIR, jobs=max(1, args.jobs))

    # ---- collect notes -------------------------------------------------------
    notes, reject = [], {"short_dur": 0, "few_valid": 0, "unvoiced": 0, "empty": 0}
    n_vib = 0
    for cid in clip_ids:
        f0 = F0.clip_f0(cid, args.cache_dir, CLIPS_DIR)
        nj = json.load(open(os.path.join(CLIPS_DIR, cid, "notes.json")))
        for nt in nj["notes"]:
            if not nt.get("vibrato"):
                continue
            n_vib += 1
            s, e = float(nt["start_s"]), float(nt["end_s"])
            dur = e - s
            t_all, c_all = F0.note_cents(f0, int(nt["pitch"]), s, e)
            if len(t_all) == 0:
                reject["empty"] += 1
                continue
            ok, n_spike = mask_note(t_all, c_all)
            n_valid = int(ok.sum())
            vfrac = n_valid / len(t_all)
            if dur < MIN_DUR_S:
                reject["short_dur"] += 1
                continue
            if n_valid < MIN_VALID:
                reject["few_valid"] += 1
                continue
            if vfrac < MIN_VOICED_FRAC:
                reject["unvoiced"] += 1
                continue
            ar = nt.get("auto", {}).get("vibrato_rate_hz")
            ar = float(ar) if ar else float("nan")
            notes.append({
                "cid": cid, "nid": nt["id"], "pitch": int(nt["pitch"]),
                "dur": dur, "bucket": bucket_of(dur),
                "t": t_all[ok], "cents": c_all[ok],
                "n_valid": n_valid, "n_frames": len(t_all), "voiced_frac": vfrac,
                "n_spike": n_spike, "auto_rate": ar,
            })

    print(f"[vib] vibrato-marked notes={n_vib} usable={len(notes)} rejected={reject}")
    if not notes:
        print("[vib] nothing to fit"); return 1

    # ---- pass 1: ladder fits (3 fits/note: full, interleaved, block) ---------
    t_start = time.perf_counter()
    nw = max(1, args.jobs)
    if nw > 1:
        from concurrent.futures import ProcessPoolExecutor
        slim = [{"t": n["t"], "cents": n["cents"], "dur": n["dur"],
                 "auto_rate": n["auto_rate"]} for n in notes]
        chunks = [slim[i::nw * 4] for i in range(nw * 4)]
        order = [list(range(i, len(slim), nw * 4)) for i in range(nw * 4)]
        print(f"[vib] fitting {len(notes)} notes x {len(ladder_names)} models "
              f"x 3 splits across {nw} processes", flush=True)
        with ProcessPoolExecutor(max_workers=nw) as ex:
            done = 0
            results = list(ex.map(_fit_chunk,
                                  [(c, ladder_names, M.GLOBAL_DEPTH) for c in chunks]))
        for idxs, res in zip(order, results):
            for i, (fits, ho, bho) in zip(idxs, res):
                notes[i]["fits"], notes[i]["ho_raw"], notes[i]["bho_raw"] = fits, ho, bho
    else:
        for i, nd in enumerate(notes):
            nd["fits"], nd["ho_raw"], nd["bho_raw"] = fit_one_note(nd, ladder_names)
            if (i + 1) % 100 == 0:
                print(f"[vib] fitted {i+1}/{len(notes)} notes "
                      f"({time.perf_counter()-t_start:.0f}s)", flush=True)
    print(f"[vib] pass 1 done in {time.perf_counter()-t_start:.0f}s", flush=True)

    # ---- pass 2: M0_global (needs the corpus median half-depth from M1) ------
    d_med = None
    if "M1_lfo4" in ladder_names:
        d_med = float(np.median([nd["fits"]["M1_lfo4"]["osc"][1] for nd in notes]))
        M.set_global_depth(d_med)
        print(f"[vib] M0_global fixed half-depth D_bar = {d_med:.2f} cents "
              f"(corpus median of M1's D; provisional default was 14.0)")
    if "M0_global" in model_names:
        for nd in notes:
            g, hg, bg = fit_one_note(nd, ["M0_global"])
            nd["fits"]["M0_global"] = g["M0_global"]
            nd["ho_raw"].update(hg)
            nd["bho_raw"].update(bg)

    # ---- derived per-note metrics -------------------------------------------
    for nd in notes:
        n, T = nd["n_valid"], float(nd["dur"])
        for name, r in nd["fits"].items():
            r["rmse_full"] = r["rmse"]
            resid = nd["cents"] - M.predict(M.REGISTRY[name], r, nd["t"], T)
            r["rho"] = M.lag1_rho(resid)
        # rho of the RICHEST model's residual == best available noise-correlation
        # estimate; shared by every model so bic_eff stays a valid comparison.
        ref = max(nd["fits"], key=lambda m: M.REGISTRY[m].n_osc)
        nd["rho_ref"] = nd["fits"][ref]["rho"]
        nd["n_eff"] = M.n_effective(n, nd["rho_ref"])
        for name, r in nd["fits"].items():
            r["bic"] = M.bic(r["sse"], n, r["k_osc"])
            r["bic_eff"] = M.bic_eff(r["sse"], n, r["k_osc"], nd["n_eff"])
            r["n_eff"] = nd["n_eff"]
            r["aicc"] = M.aicc(r["sse"], n, r["k_osc"])
            r["rmse_holdout"] = nd["ho_raw"].get(name)
            r["rmse_block_holdout"] = nd["bho_raw"].get(name)
        nd["bic_winner"] = min(model_names, key=lambda m: nd["fits"][m]["bic"])
        nd["bic_eff_winner"] = min(model_names, key=lambda m: nd["fits"][m]["bic_eff"])

    # ---- non-monotonicity (superset scoring worse in-sample than its parent) --
    # Only STRICTLY NESTED pairs are solver-failure evidence.  The rest of the
    # ladder is a warm-start convenience, not a nesting: M1's 30 ms onset gate
    # is active even at d=0 so M1a is not inside M1; M5's unsaturated linear
    # depth cannot reproduce M4's saturating ramp; M7's log-linear depth cannot
    # reproduce M5's linear one.  Counting those as failures is meaningless.
    # A bare "child > parent" test is useless here: the near-nested warm start
    # reproduces the parent to ~1e-3 cents, so sign-level differences that are
    # physically meaningless get counted.  We report the shortfall MAGNITUDE and
    # treat only >= MEANINGFUL_CENTS as a real solver failure.
    MEANINGFUL_CENTS = 0.01
    nonmono = {}
    n_nonmono = 0
    for child, par in M.LADDER.items():
        if child not in model_names or par not in model_names:
            continue
        d = np.array([nd["fits"][child]["rmse_full"] - nd["fits"][par]["rmse_full"]
                      for nd in notes])
        worse = d[d > 0]
        real = int((d > MEANINGFUL_CENTS).sum())
        nested = (par, child) in M.NESTED_PAIRS
        nonmono[f"{par}->{child}"] = {
            "n_any_worse": int((d > 1e-9).sum()),
            "n_worse_by_ge_0.01c": real,
            "median_shortfall_cents": float(np.median(worse)) if len(worse) else 0.0,
            "p90_shortfall_cents": float(np.percentile(worse, 90)) if len(worse) else 0.0,
            "strictly_nested": nested,
        }
        if nested:
            n_nonmono += real

    table = _agg(notes, model_names)

    # ---- parameter distributions --------------------------------------------
    def param_summary(name):
        mod = M.REGISTRY[name]
        cols = {p: [] for p in mod.param_names}
        pins = {}
        for nd in notes:
            osc = nd["fits"][name]["osc"]
            for p, v in zip(mod.param_names, osc):
                cols[p].append(float(v))
            for p in M.pinned(mod, osc, float(nd["dur"])):
                pins[p] = pins.get(p, 0) + 1
        return {"dist": {p: _dist(v) for p, v in cols.items()},
                "pinned_counts": pins, "n": len(notes)}

    winner_all = max(model_names,
                     key=lambda m: table[m]["by_bucket"]["all"]["bic_win_rate"])
    bic_best_mean = min(model_names,
                        key=lambda m: table[m]["by_bucket"]["all"]["mean_bic"])
    par_models = sorted({m for m in ["M1_lfo4", "M5nd_rampboth5", "M8_dramp4",
                                     winner_all, bic_best_mean]
                         if m in model_names and M.REGISTRY[m].n_osc > 0})
    params = {m: param_summary(m) for m in par_models}

    # ---- fitted rate vs stored auto.vibrato_rate_hz --------------------------
    rate_corr = {}
    for m in par_models:
        mod = M.REGISTRY[m]
        if "f" in mod.param_names:
            fi = mod.param_names.index("f")
        elif "f0" in mod.param_names:
            fi = mod.param_names.index("f0")
        else:
            continue
        pairs = [(nd["fits"][m]["osc"][fi], nd["auto_rate"]) for nd in notes
                 if np.isfinite(nd["auto_rate"])]
        if len(pairs) > 3:
            a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
            rate_corr[m] = {"n": len(pairs), "pearson_r": float(pearsonr(a, b)[0]),
                            "median_fitted": float(np.median(a)),
                            "median_auto": float(np.median(b)),
                            "median_abs_diff": float(np.median(np.abs(a - b)))}

    # ---- soft_l1 sensitivity pass (M1 vs M1a) -------------------------------
    soft = None
    if "M1_lfo4" in model_names and "M1a_lfo3" in model_names:
        s_rmse = {"M1a_lfo3": [], "M1_lfo4": []}
        for nd in notes:
            f = fit_ladder(["M1a_lfo3", "M1_lfo4"], nd["t"], nd["cents"],
                           float(nd["dur"]), nd["auto_rate"],
                           loss="soft_l1", f_scale=SOFT_L1_FSCALE)
            for k in s_rmse:
                s_rmse[k].append(f[k]["rmse"])
        lin = {k: table[k]["by_bucket"]["all"]["median_rmse_full"] for k in s_rmse}
        sl = {k: float(np.median(v)) for k, v in s_rmse.items()}
        flip = ((lin["M1_lfo4"] < lin["M1a_lfo3"]) !=
                (sl["M1_lfo4"] < sl["M1a_lfo3"]))
        soft = {"f_scale": SOFT_L1_FSCALE, "median_rmse_linear_loss": lin,
                "median_rmse_soft_l1_loss": sl, "ranking_flips": bool(flip),
                "note": "rmse is still plain L2; only the fitting loss changed"}

    # ---- results.json --------------------------------------------------------
    results = {
        "config": {
            "clips": clip_ids, "n_clips": len(clip_ids), "models": model_names,
            "sr": SR, "f0_hop": F0_HOP, "f0_frame": F0.F0_FRAME, "fps": FPS,
            "f0_fmin": F0.F0_FMIN, "f0_fmax": F0.F0_FMAX,
            "pyin_resolution_semitones": F0.F0_RESOLUTION,
            "pitch_grid_cents": 100.0 * F0.F0_RESOLUTION,
            "quantization_sigma_cents": F0.QUANT_SIGMA_CENTS,
            "cache_dir": args.cache_dir,
            "cents_abs_max": CENTS_ABS_MAX, "despike_k": DESPIKE_K,
            "min_dur_s": MIN_DUR_S, "min_valid_frames": MIN_VALID,
            "min_voiced_frac": MIN_VOICED_FRAC,
            "gate_w_s": M.GATE_W, "buckets": BUCKETS,
            "loss": "linear (L2, matches the RMSE metric)",
            "global_depth_cents": d_med,
            "holdout_interleaved": (
                "fit on even-indexed valid frames, score odd-indexed. WEAK: at "
                "172 fps a 5.5 Hz vibrato is ~31x oversampled, so held-out "
                "frames are nearly interpolable and complexity is barely "
                "penalized. Detects only catastrophic overfit."),
            "holdout_block": (
                f"fit on the FIRST {int(100*BLOCK_FRAC)}% of a note's valid "
                f"frames, score cents RMSE on the LAST {int(100*(1-BLOCK_FRAC))}%. "
                "Contiguous => genuine time EXTRAPOLATION; the discriminating "
                "metric. T stays the full score-known note duration, so M7's "
                "T-relative knots must extrapolate into unseen territory."),
            "bic_eff": (
                "BIC on n_eff = n(1-rho)/(1+rho), rho = lag-1 autocorrelation of "
                "that model's residual clipped to [0,0.99]; corrects raw BIC's "
                "assumption that the ~31x-oversampled frames are independent."),
            "block_frac": BLOCK_FRAC,
        },
        "counts": {"n_vibrato_marked": n_vib, "n_fitted": len(notes),
                   "n_rejected": sum(reject.values()), "reject_reasons": reject,
                   "by_bucket": {bk: sum(1 for n in notes if n["bucket"] == bk)
                                 for bk in BUCKETS[:-1]},
                   "median_valid_frames": float(np.median([n["n_valid"] for n in notes])),
                   "median_voiced_frac": float(np.median([n["voiced_frac"] for n in notes])),
                   "total_despiked_frames": int(sum(n["n_spike"] for n in notes)),
                   "median_rho_ref": float(np.median([n["rho_ref"] for n in notes])),
                   "median_n_eff": float(np.median([n["n_eff"] for n in notes])),
                   "median_n_valid_over_n_eff": float(np.median(
                       [n["n_valid"] / n["n_eff"] for n in notes]))},
        "table": table,
        "n_nonmonotone": n_nonmono,
        "nonmonotone_by_pair": nonmono,
        "param_distributions": params,
        "rate_vs_auto_correlation": rate_corr,
        "soft_l1_sensitivity": soft,
        "bic_winner_all": winner_all,
        "lowest_mean_bic_all": bic_best_mean,
        "bic_eff_winner_all": max(
            model_names,
            key=lambda m: table[m]["by_bucket"]["all"]["bic_eff_win_rate"]),
        "best_block_holdout_all": min(
            [m for m in model_names
             if table[m]["by_bucket"]["all"]["median_rmse_block_holdout"]],
            key=lambda m: table[m]["by_bucket"]["all"]["median_rmse_block_holdout"],
            default=None),
    }
    out_json = os.path.join(HERE, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # ---- figures -------------------------------------------------------------
    fig_a(notes, model_names)
    fig_b(table, model_names)
    fig_c(table, model_names)
    fig_d(notes, params, par_models)

    # ---- console -------------------------------------------------------------
    for bk in BUCKETS:
        print(f"\n=== bucket {bk} "
              f"(n={table[model_names[0]]['by_bucket'].get(bk, {}).get('n', 0)}) ===")
        print(f"{'model':16s} {'k':>2s} {'rmse_full':>10s} {'rmse_intlv':>10s} "
              f"{'rmse_BLOCK':>10s} {'meanBIC':>9s} {'BICwin%':>8s} "
              f"{'BICeffW%':>9s} {'expl%':>7s} {'fit_ms':>7s}")
        for name in model_names:
            b = table[name]["by_bucket"].get(bk)
            if not b:
                continue
            ho = f"{b['median_rmse_holdout']:10.3f}" if b["median_rmse_holdout"] \
                else f"{'--':>10s}"
            bh = f"{b['median_rmse_block_holdout']:10.3f}" \
                if b["median_rmse_block_holdout"] else f"{'--':>10s}"
            ex = f"{100*b['pct_explained']:7.1f}" if b["pct_explained"] is not None \
                else f"{'--':>7s}"
            print(f"{name:16s} {table[name]['n_osc']:>2d} {b['median_rmse_full']:10.3f} "
                  f"{ho} {bh} {b['mean_bic']:9.1f} {100*b['bic_win_rate']:8.1f} "
                  f"{100*b['bic_eff_win_rate']:9.1f} {ex} {b['median_fit_ms']:7.1f}")

    print(f"\nrejections: {reject}  (of {n_vib} vibrato-marked notes)")
    print(f"n_nonmonotone (nested pairs, shortfall >= 0.01 cents): {n_nonmono} "
          f"(n={len(notes)} notes per pair)")
    print(f"   {'pair':24s} {'any>':>5s} {'>=.01c':>7s} {'medShort':>9s} "
          f"{'p90Short':>9s}  nesting")
    for pair, v in nonmono.items():
        tag = "nested" if v["strictly_nested"] else "NOT nested (warm-start only)"
        print(f"   {pair:24s} {v['n_any_worse']:5d} {v['n_worse_by_ge_0.01c']:7d} "
              f"{v['median_shortfall_cents']:9.2e} {v['p90_shortfall_cents']:9.2e}  {tag}")
    print(f"pyin pitch grid {100*F0.F0_RESOLUTION:.0f} cents "
          f"-> quantization sigma {F0.QUANT_SIGMA_CENTS:.2f} cents "
          f"(irreducible floor under every model's RMSE)")
    print(f"BIC win-rate leader (all): {winner_all} | lowest mean BIC: {bic_best_mean}")
    for m, p in params.items():
        print(f"\n[{m}] param p10/p50/p90:")
        for k, v in p["dist"].items():
            if v:
                print(f"   {k:5s} {v['p10']:9.3f} {v['p50']:9.3f} {v['p90']:9.3f}")
        print(f"   pinned-at-bound counts (of {p['n']}): {p['pinned_counts']}")
    print(f"\nrate vs auto.vibrato_rate_hz: {json.dumps(rate_corr, indent=1)}")
    print(f"soft_l1 sensitivity: {json.dumps(soft, indent=1)}")
    print(f"\n[vib] wrote {out_json} + 4 PNGs to {HERE}")
    return 0


# --- figures ------------------------------------------------------------------

def fig_a(notes, model_names):
    """6 example notes (2 per duration bucket) with the M1 baseline removed.

    Black = measured cents minus M1's fitted affine baseline; each overlay is
    that model's full prediction minus the *same* M1 baseline, so the visual gap
    to the black curve is exactly each model's residual.
    """
    show = [m for m in ["M0_global", "M1a_lfo3", "M1_lfo4", "M2_framp",
                        "M4_dramp", "M5_both", "M7_spline"] if m in model_names]
    ref = "M1_lfo4" if "M1_lfo4" in model_names else model_names[-1]
    picks = []
    for bk in BUCKETS[:-1]:
        sub = sorted([n for n in notes if n["bucket"] == bk],
                     key=lambda n: n["fits"][ref]["rmse_full"])
        if not sub:
            continue
        for q in (0.25, 0.6):
            picks.append(sub[min(len(sub) - 1, int(q * len(sub)))])
    picks = picks[:6]
    if not picks:
        return
    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 2.5 * len(picks)))
    axes = np.atleast_1d(axes)
    for ax, nd in zip(axes, picks):
        t, c, T = nd["t"], nd["cents"], float(nd["dur"])
        b0, b1 = nd["fits"][ref]["baseline"]
        base = b0 + b1 * t
        ax.plot(t, c - base, "k.-", lw=1.4, ms=3, label="measured (M1 baseline removed)")
        tf = np.linspace(t.min(), t.max(), 600)
        basef = b0 + b1 * tf
        for m in show:
            y = M.predict(M.REGISTRY[m], nd["fits"][m], tf, T) - basef
            ax.plot(tf, y, lw=1.0, alpha=0.85,
                    label=f"{m} ({nd['fits'][m]['rmse_full']:.1f})")
        o = nd["fits"][ref]["osc"]
        ax.set_title(f"{nd['cid']} {nd['nid']} pitch={nd['pitch']} dur={T:.2f}s "
                     f"n={nd['n_valid']} | M1: d={o[0]*1000:.0f}ms D={o[1]:.1f}c "
                     f"f={o[2]:.2f}Hz", fontsize=8)
        ax.set_xlabel("t since note onset (s)", fontsize=7)
        ax.set_ylabel("cents", fontsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=5.5, ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_a_example_fits.png"), dpi=110)
    plt.close(fig)


def fig_b(table, model_names):
    fig, axes = plt.subplots(1, len(BUCKETS), figsize=(4.2 * len(BUCKETS), 4.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, bk in zip(axes, BUCKETS):
        pts = [(table[m]["n_osc"], table[m]["by_bucket"].get(bk), m) for m in model_names]
        pts = [p for p in pts if p[1]]
        if not pts:
            ax.set_title(f"{bk} (empty)", fontsize=9); ax.axis("off"); continue
        pts.sort(key=lambda p: p[0])
        x = [p[0] for p in pts]
        ax.plot(x, [p[1]["median_rmse_full"] for p in pts], "o-", label="in-sample")
        ho = [(p[0], p[1]["median_rmse_holdout"]) for p in pts
              if p[1]["median_rmse_holdout"] is not None]
        if ho:
            ax.plot([h[0] for h in ho], [h[1] for h in ho], "s--",
                    label="interleaved holdout")
        bh = [(p[0], p[1]["median_rmse_block_holdout"]) for p in pts
              if p[1]["median_rmse_block_holdout"] is not None]
        if bh:
            ax.plot([h[0] for h in bh], [h[1] for h in bh], "^:", color="C3",
                    label="BLOCK holdout (last 40%)")
        for p in pts:
            ax.annotate(p[2].split("_")[0], (p[0], p[1]["median_rmse_full"]),
                        fontsize=6, xytext=(2, 4), textcoords="offset points")
        ax.set_title(f"{bk} (n={pts[0][1]['n']})", fontsize=9)
        ax.set_xlabel("n oscillator params")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("median cents RMSE")
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Vibrato F0 fit error vs model complexity.  In-sample and INTERLEAVED "
        "holdout are indistinguishable (~31 samples/cycle -> held-out frames are "
        "interpolable, so complexity is not penalized at all); the BLOCK holdout "
        "(extrapolate onto the last 40% of the note) rises with complexity.",
        fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_b_rmse_vs_params.png"), dpi=110)
    plt.close(fig)


def fig_c(table, model_names):
    """Raw BIC vs autocorrelation-corrected BIC win rates.

    The two panels are the headline caveat of the whole study: raw BIC treats
    every one of the ~31 frames per vibrato cycle as an independent
    observation, so it massively under-prices parameters.  bic_eff deflates n to
    the effective sample size and the ranking changes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.2), sharey=True)
    w = 0.8 / len(BUCKETS)
    x = np.arange(len(model_names))
    for ax, key, title in [
            (axes[0], "bic_win_rate", "raw BIC (n = all frames; UNDER-prices params)"),
            (axes[1], "bic_eff_win_rate",
             "bic_eff (n_eff = n(1-rho)/(1+rho); autocorrelation-corrected)")]:
        for i, bk in enumerate(BUCKETS):
            v = [100 * table[m]["by_bucket"].get(bk, {}).get(key, 0.0)
                 for m in model_names]
            ax.bar(x + i * w - 0.4 + w / 2, v, w, label=bk)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{m}\n({table[m]['n_osc']}p)" for m in model_names],
                           fontsize=6.5, rotation=30, ha="right")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("% of notes where this model wins")
    axes[0].legend(title="duration", fontsize=7)
    fig.suptitle("Model-selection win rate (k = n_osc + 2 nuisance baseline params)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_c_bic_winrate.png"), dpi=110)
    plt.close(fig)


def fig_d(notes, params, par_models):
    m0 = "M1_lfo4" if "M1_lfo4" in params else (par_models[0] if par_models else None)
    if m0 is None:
        return
    mod = M.REGISTRY[m0]
    osc = np.array([nd["fits"][m0]["osc"] for nd in notes])
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    spec = [("f", "vibrato rate (Hz)", (3, 11)),
            ("D", "half-depth (cents)", (0, 60)),
            ("d", "onset delay d (s)", (0, 0.6))]
    for ax, (p, lab, rng) in zip(axes, spec):
        if p not in mod.param_names:
            ax.axis("off"); continue
        v = osc[:, mod.param_names.index(p)]
        ax.hist(v, bins=40, range=rng, color="C0", alpha=0.85)
        ax.axvline(np.median(v), color="k", ls="--",
                   label=f"median {np.median(v):.3f}")
        ax.set_xlabel(lab); ax.set_ylabel("notes"); ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    ax = axes[3]
    fi = mod.param_names.index("f") if "f" in mod.param_names \
        else mod.param_names.index("f0")
    ar = np.array([nd["auto_rate"] for nd in notes])
    ok = np.isfinite(ar)
    if ok.sum() > 3:
        r = pearsonr(osc[ok, fi], ar[ok])[0]
        ax.scatter(ar[ok], osc[ok, fi], s=7, alpha=0.35)
        ax.plot([3, 11], [3, 11], "k--", lw=0.8)
        ax.set_xlabel("auto.vibrato_rate_hz (stored)")
        ax.set_ylabel(f"fitted rate ({m0})")
        ax.set_title(f"Pearson r = {r:.3f} (n={int(ok.sum())})", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{m0} fitted parameter distributions (n={len(notes)} vibrato notes)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_d_param_distributions.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
