"""Per-note dynamic-envelope parameterization experiment (read-only analysis).

Compares compact per-note parameterizations of a violin note's loudness/energy
envelope against reconstruction error, to inform replacing the current single
scalar MIDI velocity conditioning with a richer per-note envelope descriptor.

Data: the 22 VERIFIED Labeler clips (== the compiled gt_arky manifest clips).
Per clip we read Data/gt_arky/clips/<id>/cleaned.wav (mono @ SR, hop 512) and
notes.json (aligned per-note start_s/end_s + stored MIDI velocity).

Nothing in the pipeline is modified. Outputs (figs + results.json) land next to
this script.
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
from scipy.optimize import least_squares
from scipy.stats import pearsonr, spearmanr

# --- constants (mirror common.config; do NOT import the pipeline) --------------
SR = 44100
HOP = 512
N_FFT = 2048
FPS = SR / HOP                      # ~86.13 frames/s
FRAME_MS = 1000.0 * HOP / SR        # ~11.61 ms/frame
EPS = 1e-10

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))
CLIPS_DIR = os.path.join(PROJ, "Data", "gt_arky", "clips")
MANIFEST = os.path.join(PROJ, "Data", "datasets", "gt_arky", "manifest.json")

RNG = np.random.default_rng(0)
MAX_NOTES = 5000

PRE_FRAMES = int(round(0.025 * FPS))        # ~25 ms pre-onset attack context (2)
ATTACK_FRAMES = int(round(0.060 * FPS))     # first 60 ms window (5)
DCT_NFIX = 64                               # fixed length for DCT descriptor
SHORT_FRAMES = 3                            # < this many frames => degenerate


def frame_of(t_s: float) -> int:
    return int(round(t_s * SR / HOP))


# --- envelope extraction -------------------------------------------------------

def clip_envelopes(y: np.ndarray):
    """Return (a_weighted_db[T], log_rms_db[T]) frame-wise loudness, hop 512."""
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)          # [1+N_FFT/2, T]
    power = np.abs(S) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    a_w_db = librosa.A_weighting(freqs)                        # dB per bin (0 Hz -> min_db)
    w_lin = 10.0 ** (a_w_db / 10.0)
    avg_power = np.mean(power * w_lin[:, None], axis=0)        # DDSP-style A-weighted mean power
    a_db = 10.0 * np.log10(avg_power + EPS)

    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    rms_db = 20.0 * np.log10(rms + EPS)
    T = min(len(a_db), len(rms_db))
    return a_db[:T], rms_db[:T]


# --- parameterization fits (each returns reconstruction over the L note frames) -

def fit_K0(env):
    return np.full_like(env, env.mean()), 1


def fit_K1_attack_mean(env, env_pre):
    """peak dB in first 60ms (incl. pre-onset context) + mean dB of remainder."""
    L = len(env)
    a = min(ATTACK_FRAMES, L)
    peak = float(np.concatenate([env_pre, env[:a]]).max()) if a > 0 else float(env.max())
    rem = env[a:]
    mean_rem = float(rem.mean()) if len(rem) else peak
    recon = np.empty_like(env)
    recon[:a] = peak
    recon[a:] = mean_rem
    return recon, 2


def fit_linear(env):
    L = len(env)
    if L < 2:
        return np.full_like(env, env.mean()), 2
    x = np.arange(L, dtype=float)
    A = np.vstack([x, np.ones(L)]).T
    coef, *_ = np.linalg.lstsq(A, env, rcond=None)
    return A @ coef, 2


def fit_dct(env, k):
    """First k DCT-II coeffs of the env resampled to DCT_NFIX; recon back to L."""
    L = len(env)
    xf = np.linspace(0, 1, DCT_NFIX)
    xs = np.linspace(0, 1, L) if L > 1 else np.array([0.0])
    up = np.interp(xf, xs, env) if L > 1 else np.full(DCT_NFIX, env[0])
    c = dct(up, type=2, norm="ortho")
    c[k:] = 0.0
    rec_fix = idct(c, type=2, norm="ortho")
    recon = np.interp(xs, xf, rec_fix) if L > 1 else np.array([rec_fix.mean()])
    return recon, k


def fit_pwl3(env):
    """Continuous piecewise-linear: start, free interior breakpoint, end (~4 params)."""
    L = len(env)
    if L < 3:
        return fit_linear(env)
    x = np.arange(L, dtype=float)
    best = None
    for m in range(1, L - 1):
        # continuous PWL with knots at 0, m, L-1 -> linear in the 3 knot heights.
        b0 = np.where(x <= m, 1.0 - x / m, 0.0)                  # hat at 0
        b1 = np.where(x <= m, x / m, (L - 1 - x) / (L - 1 - m))  # hat at m
        b2 = np.where(x >= m, (x - m) / (L - 1 - m), 0.0)        # hat at L-1
        A = np.vstack([b0, b1, b2]).T
        coef, *_ = np.linalg.lstsq(A, env, rcond=None)
        rec = A @ coef
        err = np.mean((rec - env) ** 2)
        if best is None or err < best[0]:
            best = (err, rec)
    return best[1], 4


def _adsr_curve(p, L, L0, Lend):
    ta, td, tr, Lp, Ls = p
    t = np.linspace(0.0, 1.0, L)
    d_end = ta + td
    r_start = max(1.0 - tr, d_end)
    xs = np.array([0.0, ta, d_end, r_start, 1.0])
    ys = np.array([L0, Lp, Ls, Ls, Lend])
    order = np.argsort(xs)
    return np.interp(t, xs[order], ys[order])


def fit_adsr(env):
    """5-param ADSR-style fit (attack/decay/release fractions + peak/sustain level)."""
    L = len(env)
    if L < 4:
        return fit_linear(env)[0], 5, "short_fallback_linear"
    L0, Lend = float(env[0]), float(env[-1])
    Lp0 = float(env.max())
    Ls0 = float(np.median(env))
    p0 = np.array([0.1, 0.2, 0.2, Lp0, Ls0])
    lo = np.array([0.01, 0.01, 0.01, env.min() - 5, env.min() - 5])
    hi = np.array([0.6, 0.7, 0.6, env.max() + 5, env.max() + 5])
    p0 = np.clip(p0, lo + 1e-6, hi - 1e-6)

    def resid(p):
        return _adsr_curve(p, L, L0, Lend) - env

    try:
        sol = least_squares(resid, p0, bounds=(lo, hi), max_nfev=200, method="trf")
        recon = _adsr_curve(sol.x, L, L0, Lend)
        status = "ok" if sol.success else "no_converge"
    except Exception as e:  # noqa: BLE001
        recon = np.full_like(env, env.mean())
        status = f"except:{type(e).__name__}"
    return recon, 5, status


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# --- main ----------------------------------------------------------------------

def main():
    manifest = json.load(open(MANIFEST))
    clip_ids = sorted(manifest["clips"].keys())
    print(f"[exp] {len(clip_ids)} verified clips; FPS={FPS:.3f} frame_ms={FRAME_MS:.3f}")
    print(f"[exp] PRE_FRAMES={PRE_FRAMES} ATTACK_FRAMES={ATTACK_FRAMES}")

    records = []          # per-note dicts
    method_pairs = []     # (a_db, rms_db) pooled per-frame for env-method correlation
    per_note_method_r = []
    n_total_notes = 0
    n_short = 0
    n_out_of_range = 0

    for cid in clip_ids:
        cdir = os.path.join(CLIPS_DIR, cid)
        wav = os.path.join(cdir, "cleaned.wav")
        nj = json.load(open(os.path.join(cdir, "notes.json")))
        y, _ = librosa.load(wav, sr=SR, mono=True)
        a_db, rms_db = clip_envelopes(y)
        T = len(a_db)
        for nt in nj["notes"]:
            n_total_notes += 1
            f0 = frame_of(float(nt["start_s"]))
            f1 = frame_of(float(nt["end_s"]))
            f0c = int(np.clip(f0, 0, T))
            f1c = int(np.clip(f1, 0, T))
            if f1c <= f0c:
                f1c = min(T, f0c + 1)
            if f0c >= T:
                n_out_of_range += 1
                continue
            env = a_db[f0c:f1c]
            envr = rms_db[f0c:f1c]
            if len(env) < 1:
                n_out_of_range += 1
                continue
            L = len(env)
            pre0 = max(0, f0c - PRE_FRAMES)
            env_pre = a_db[pre0:f0c]
            dur_s = float(nt["end_s"]) - float(nt["start_s"])
            short = L < SHORT_FRAMES
            if short:
                n_short += 1
            records.append({
                "cid": cid, "L": L, "dur_s": dur_s, "short": short,
                "velocity": int(nt["velocity"]),
                "onset_strength": float(nt.get("auto", {}).get("onset_strength", 0.0)),
                "peak_db": float(env.max()), "mean_db": float(env.mean()),
                "env": env, "env_pre": env_pre, "envr": envr,
            })
            method_pairs.append((env, envr))
            if L >= 3:
                per_note_method_r.append(np.corrcoef(env, envr)[0, 1])

    print(f"[exp] total notes={n_total_notes} usable={len(records)} "
          f"short(<{SHORT_FRAMES}f)={n_short} out_of_range={n_out_of_range}")

    # subsample for the per-note fitting if huge
    if len(records) > MAX_NOTES:
        idx = RNG.choice(len(records), MAX_NOTES, replace=False)
        fit_records = [records[i] for i in sorted(idx)]
        print(f"[exp] subsampled {MAX_NOTES}/{len(records)} notes for fitting")
    else:
        fit_records = records

    schemes = ["K0", "K1_attack_mean", "linear", "dct2", "dct3", "dct4",
               "dct5", "pwl3", "adsr"]
    param_count = {"K0": 1, "K1_attack_mean": 2, "linear": 2, "dct2": 2, "dct3": 3,
                   "dct4": 4, "dct5": 5, "pwl3": 4, "adsr": 5}

    # per-note RMSE for every scheme
    adsr_status = {}
    example_bank = {}  # for figure (a): store a few notes' env + all recons
    for r in fit_records:
        env, env_pre = r["env"], r["env_pre"]
        recons = {}
        recons["K0"] = fit_K0(env)[0]
        recons["K1_attack_mean"] = fit_K1_attack_mean(env, env_pre)[0]
        recons["linear"] = fit_linear(env)[0]
        for k in (2, 3, 4, 5):
            recons[f"dct{k}"] = fit_dct(env, k)[0]
        recons["pwl3"] = fit_pwl3(env)[0]
        ad_rec, _, ad_stat = fit_adsr(env)
        recons["adsr"] = ad_rec
        adsr_status[ad_stat] = adsr_status.get(ad_stat, 0) + 1
        r["rmse"] = {s: rmse(recons[s], env) for s in schemes}

    # duration buckets
    def bucket(d):
        if d < 0.25:
            return "<0.25s"
        if d <= 1.0:
            return "0.25-1s"
        return ">1s"

    buckets = ["<0.25s", "0.25-1s", ">1s", "all"]
    summary = {}
    for s in schemes:
        summary[s] = {"params": param_count[s], "by_bucket": {}}
        for bk in buckets:
            rs = [r for r in fit_records if (bk == "all" or bucket(r["dur_s"]) == bk)]
            if not rs:
                continue
            errs = np.array([r["rmse"][s] for r in rs])
            durs = np.array([r["dur_s"] for r in rs])
            dw = float(np.sum(errs * durs) / np.sum(durs))
            summary[s]["by_bucket"][bk] = {
                "n": len(rs),
                "median_rmse_db": float(np.median(errs)),
                "p90_rmse_db": float(np.percentile(errs, 90)),
                "dur_weighted_mean_rmse_db": dw,
            }

    # short-note behavior: median rmse on short vs non-short (all schemes)
    short_stats = {}
    for s in schemes:
        se = np.array([r["rmse"][s] for r in fit_records if r["short"]])
        ne = np.array([r["rmse"][s] for r in fit_records if not r["short"]])
        short_stats[s] = {
            "n_short": int(len(se)),
            "median_rmse_short": float(np.median(se)) if len(se) else None,
            "median_rmse_nonshort": float(np.median(ne)) if len(ne) else None,
        }

    # velocity correlations (over ALL usable notes, not just fit subset)
    vel = np.array([r["velocity"] for r in records], dtype=float)
    peak = np.array([r["peak_db"] for r in records])
    mean_db = np.array([r["mean_db"] for r in records])
    onset_str = np.array([r["onset_strength"] for r in records])
    cids = np.array([r["cid"] for r in records])

    def _within_clip_center(vals):
        """Subtract each clip's mean (removes cross-clip level / percentile offset)."""
        out = np.array(vals, dtype=float)
        for c in np.unique(cids):
            m = cids == c
            out[m] = out[m] - out[m].mean()
        return out

    vel_c = _within_clip_center(vel)
    peak_c = _within_clip_center(peak)
    mean_c = _within_clip_center(mean_db)
    vel_corr = {
        "n": len(records),
        "note_cross_clip": {
            "velocity_vs_peak_db_pearson": float(pearsonr(vel, peak)[0]),
            "velocity_vs_peak_db_spearman": float(spearmanr(vel, peak)[0]),
            "velocity_vs_mean_db_pearson": float(pearsonr(vel, mean_db)[0]),
            "velocity_vs_mean_db_spearman": float(spearmanr(vel, mean_db)[0]),
            "onset_strength_vs_peak_db_pearson": float(pearsonr(onset_str, peak)[0]),
            "onset_strength_vs_mean_db_pearson": float(pearsonr(onset_str, mean_db)[0]),
            "peak_vs_mean_db_pearson": float(pearsonr(peak, mean_db)[0]),
        },
        "within_clip_centered": {
            "velocity_vs_peak_db_pearson": float(pearsonr(vel_c, peak_c)[0]),
            "velocity_vs_mean_db_pearson": float(pearsonr(vel_c, mean_c)[0]),
        },
        # kept flat for the scatter-fig annotation
        "velocity_vs_peak_db_pearson": float(pearsonr(vel, peak)[0]),
        "velocity_vs_mean_db_pearson": float(pearsonr(vel, mean_db)[0]),
    }

    # envelope method comparison
    pooled_a = np.concatenate([p[0] for p in method_pairs])
    pooled_r = np.concatenate([p[1] for p in method_pairs])
    method_cmp = {
        "pooled_frame_pearson_Aweighted_vs_logRMS": float(pearsonr(pooled_a, pooled_r)[0]),
        "pooled_frame_spearman": float(spearmanr(pooled_a, pooled_r)[0]),
        "per_note_pearson_median": float(np.median(per_note_method_r)),
        "per_note_pearson_p10": float(np.percentile(per_note_method_r, 10)),
        "mean_offset_db_A_minus_RMS": float(np.mean(pooled_a - pooled_r)),
    }

    results = {
        "meta": {
            "n_clips": len(clip_ids), "clip_ids": clip_ids,
            "n_total_notes": n_total_notes, "n_usable": len(records),
            "n_fit": len(fit_records), "n_short": n_short,
            "n_out_of_range": n_out_of_range,
            "fps": FPS, "frame_ms": FRAME_MS, "hop": HOP, "sr": SR,
            "envelope_target": "A_weighted_dB", "audio": "cleaned.wav",
            "short_threshold_frames": SHORT_FRAMES,
        },
        "scheme_summary": summary,
        "short_note_stats": short_stats,
        "adsr_fit_status": adsr_status,
        "velocity_correlations": vel_corr,
        "envelope_method_comparison": method_cmp,
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- figures -------------------------------------------------------------
    _fig_examples(fit_records)
    _fig_rmse_vs_params(summary, param_count, schemes)
    _fig_velocity_scatter(vel, peak, mean_db, vel_corr)

    # ---- console report ------------------------------------------------------
    print("\n=== SCHEME SUMMARY (median / p90 RMSE dB, by duration bucket) ===")
    hdr = f"{'scheme':16s} {'p':>2s} | "
    for bk in buckets:
        hdr += f"{bk:>10s}          "
    print(hdr)
    for s in schemes:
        line = f"{s:16s} {param_count[s]:>2d} | "
        for bk in buckets:
            b = summary[s]["by_bucket"].get(bk)
            if b:
                line += f"{b['median_rmse_db']:5.2f}/{b['p90_rmse_db']:5.2f}(n{b['n']:<4d}) "
            else:
                line += f"{'--':>18s} "
        print(line)
    print("\nADSR fit status:", adsr_status)
    print("Velocity corr:", json.dumps(vel_corr, indent=1))
    print("Envelope method cmp:", json.dumps(method_cmp, indent=1))
    print("\n[exp] wrote results.json + 3 PNGs to", HERE)


def _fig_examples(fit_records):
    # pick varied examples: shortest, a mid, a long, and highest-range note
    rs = [r for r in fit_records if r["L"] >= 2]
    by_len = sorted(rs, key=lambda r: r["L"])
    ranges = sorted(rs, key=lambda r: r["env"].max() - r["env"].min())
    picks = []
    seen = set()
    for cand in [by_len[len(by_len)//20], by_len[len(by_len)//2],
                 by_len[-1], ranges[-1], ranges[-5]]:
        key = id(cand)
        if key not in seen:
            seen.add(key)
            picks.append(cand)
    picks = picks[:4]
    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 3 * len(picks)))
    if len(picks) == 1:
        axes = [axes]
    for ax, r in zip(axes, picks):
        env, env_pre = r["env"], r["env_pre"]
        x = np.arange(len(env))
        ax.plot(x, env, "k-", lw=2.2, label="A-weighted dB (true)")
        ax.plot(x, fit_K0(env)[0], label="K0 const")
        ax.plot(x, fit_K1_attack_mean(env, env_pre)[0], label="K1 attack+mean")
        ax.plot(x, fit_linear(env)[0], label="linear")
        ax.plot(x, fit_dct(env, 3)[0], label="dct3")
        ax.plot(x, fit_pwl3(env)[0], label="pwl3")
        ax.plot(x, fit_adsr(env)[0], "--", label="adsr")
        ax.set_title(f"{r['cid']}  L={r['L']}f  dur={r['dur_s']:.3f}s  vel={r['velocity']}")
        ax.set_xlabel("frame (hop 512)")
        ax.set_ylabel("loudness dB")
        ax.legend(fontsize=6, ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_a_example_notes.png"), dpi=110)
    plt.close(fig)


def _fig_rmse_vs_params(summary, param_count, schemes):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    buckets = ["<0.25s", "0.25-1s", ">1s", "all"]
    for bk in buckets:
        xs, ys, labels = [], [], []
        for s in schemes:
            b = summary[s]["by_bucket"].get(bk)
            if b:
                xs.append(param_count[s] + 0.02 * hash(bk) % 3 * 0.0)
                ys.append(b["median_rmse_db"])
                labels.append(s)
        order = np.argsort(xs)
        xs = np.array(xs)[order]; ys = np.array(ys)[order]
        ax.plot(xs, ys, "o-", label=bk, alpha=0.8)
    # annotate scheme names at the 'all' bucket
    for s in schemes:
        b = summary[s]["by_bucket"].get("all")
        if b:
            ax.annotate(s, (param_count[s], b["median_rmse_db"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("parameter count")
    ax.set_ylabel("median RMSE (dB) vs A-weighted envelope")
    ax.set_title("Reconstruction error vs param count (by duration bucket)")
    ax.legend(title="duration")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_b_rmse_vs_params.png"), dpi=110)
    plt.close(fig)


def _fig_velocity_scatter(vel, peak, mean_db, vel_corr):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(vel, peak, s=6, alpha=0.3)
    axes[0].set_xlabel("stored MIDI velocity")
    axes[0].set_ylabel("note peak A-weighted dB")
    axes[0].set_title(f"velocity vs peak dB  (r={vel_corr['velocity_vs_peak_db_pearson']:.3f})")
    axes[1].scatter(vel, mean_db, s=6, alpha=0.3, color="C1")
    axes[1].set_xlabel("stored MIDI velocity")
    axes[1].set_ylabel("note mean A-weighted dB")
    axes[1].set_title(f"velocity vs mean dB  (r={vel_corr['velocity_vs_mean_db_pearson']:.3f})")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_c_velocity_scatter.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
