"""Whole-clip F0 extraction + on-disk cache for the vibrato parameterization study.

Design decisions worth stating up front:

* **hop 256 / frame 1024.**  Vibrato lives at ~4.5-7 Hz, i.e. a period of
  140-220 ms.  ``frame_length=2048`` (46 ms analysis window at 44.1 kHz) averages
  over ~a quarter of a vibrato period and measurably flattens the modulation
  (~14% of the depth is lost).  1024 (23 ms) still resolves the G string
  fundamental (196 Hz -> ~4.5 periods per window) while keeping the modulation
  intact.  hop 256 gives 172.27 fps, ~31 samples per vibrato cycle.
* **pyin, whole clip at once.**  Per-note extraction would restart pyin's HMM at
  every note boundary and produce edge artifacts exactly where the vibrato onset
  delay is measured.  We run once per clip and slice.
* **pitch-bin resolution.**  librosa's ``pyin`` decodes onto a discrete pitch
  grid whose spacing is ``resolution`` semitones; the default ``0.1`` means a
  **10-cent staircase**, and the Viterbi transition prior holds the state in a
  bin for tens of milliseconds at a time.  Against a corpus half-depth of order
  10 cents that is not a small perturbation -- it squares off the modulation and
  puts a ~3-cent floor under every model's residual, compressing exactly the
  differences this experiment is trying to measure.  We use ``resolution=0.05``
  (5-cent bins, ~1.4 cent quantization sigma); 0.025 was measured to buy almost
  nothing more for ~4x the runtime.  See ``results.json ->
  config.quantization_sigma_cents``.
* **cents relative to the *scored* pitch**, not to a per-note median: the
  intonation offset and any slide are absorbed by the jointly fitted affine
  nuisance baseline in ``models.fit``, so the raw cents trace is what the fitter
  should see.

Nothing in the repo pipeline is touched; this module only reads
``Data/gt_arky/clips/<id>/cleaned.wav``.
"""

from __future__ import annotations

import os

import librosa
import numpy as np

# --- constants (mirrored locally; do NOT import the pipeline) -----------------
SR = 44100
F0_HOP = 256
F0_FRAME = 1024
F0_FMIN = 175.0          # below the open G (196 Hz) with margin
F0_FMAX = 2100.0         # above MIDI 83 (987 Hz) with harmonic-slip margin
F0_RESOLUTION = 0.05     # pyin pitch-bin spacing in semitones -> 5-cent grid
FPS = SR / F0_HOP        # 172.266 frames/s
QUANT_SIGMA_CENTS = 100.0 * F0_RESOLUTION / (12 ** 0.5)   # uniform-quantization sigma

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))
CLIPS_DIR = os.path.join(PROJ, "Data", "gt_arky", "clips")
MANIFEST = os.path.join(PROJ, "Data", "datasets", "gt_arky", "manifest.json")

DEFAULT_CACHE_DIR = (
    r"C:\Users\archi\AppData\Local\Temp\claude"
    r"\c--Users-archi-Documents-Coding-stuff-AI-Violin"
    r"\d22fd853-67b0-4041-a36d-46dc68bac1b7\scratchpad\f0_cache"
)

_META = dict(sr=SR, hop=F0_HOP, frame=F0_FRAME, fmin=F0_FMIN, fmax=F0_FMAX,
             resolution=F0_RESOLUTION)


def cache_path(clip_id: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"{clip_id}.npz")


def _meta_ok(z) -> bool:
    try:
        return all(float(z[k]) == float(v) for k, v in _META.items())
    except (KeyError, ValueError, TypeError):
        return False


def clip_f0(clip_id: str, cache_dir: str = DEFAULT_CACHE_DIR,
            clips_dir: str = CLIPS_DIR, verbose: bool = False) -> np.ndarray:
    """Return [T] float64 F0 in Hz for the whole clip, NaN where unvoiced.

    Cached as one ``.npz`` per clip; the analysis settings are stored alongside
    the array so a stale cache (different hop/frame/band) is detected and
    recomputed rather than silently reused.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path(clip_id, cache_dir)
    if os.path.exists(path):
        try:
            with np.load(path, allow_pickle=False) as z:
                if _meta_ok(z) and str(z["clip_id"]) == clip_id:
                    return z["f0"].astype(np.float64)
        except Exception:  # noqa: BLE001 - corrupt cache -> recompute
            pass

    wav = os.path.join(clips_dir, clip_id, "cleaned.wav")
    y, _ = librosa.load(wav, sr=SR, mono=True)
    f0, _voiced_flag, _voiced_prob = librosa.pyin(
        y, fmin=F0_FMIN, fmax=F0_FMAX, sr=SR,
        frame_length=F0_FRAME, hop_length=F0_HOP, resolution=F0_RESOLUTION,
    )
    f0 = np.asarray(f0, dtype=np.float64)
    np.savez_compressed(path, f0=f0.astype(np.float32), clip_id=clip_id, **_META)
    if verbose:
        vf = float(np.mean(np.isfinite(f0)))
        print(f"[f0] {clip_id}: {len(f0)} frames ({len(y)/SR:.1f}s), voiced {vf:.2%}")
    return f0


def note_cents(f0_hz: np.ndarray, pitch: int, start_s: float, end_s: float):
    """Slice a note out of the clip F0 trace and convert to cents vs the score pitch.

    Returns ``(t_rel_s, cents)``.  ``t_rel_s`` uses the *true* frame times minus
    ``start_s``, so the integer rounding of the slice bounds shifts which frames
    are included but never shifts the timebase the vibrato-onset delay ``d`` is
    measured against.  Unvoiced frames stay NaN and are dropped by the caller.
    """
    T = len(f0_hz)
    i0 = int(np.clip(round(start_s * SR / F0_HOP), 0, T))
    i1 = int(np.clip(round(end_s * SR / F0_HOP), 0, T))
    if i1 <= i0:
        return np.zeros(0), np.zeros(0)
    idx = np.arange(i0, i1)
    t = idx * (F0_HOP / SR) - start_s
    ref = 440.0 * 2.0 ** ((pitch - 69) / 12.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cents = 1200.0 * np.log2(f0_hz[i0:i1] / ref)
    return t, cents


def is_cached(clip_id: str, cache_dir: str) -> bool:
    """True only if a cache file exists AND was built with the current settings."""
    p = cache_path(clip_id, cache_dir)
    if not os.path.exists(p):
        return False
    try:
        with np.load(p, allow_pickle=False) as z:
            return _meta_ok(z) and str(z["clip_id"]) == clip_id
    except Exception:  # noqa: BLE001
        return False


def _warm(args):
    cid, cache_dir, clips_dir = args
    try:
        f0 = clip_f0(cid, cache_dir, clips_dir)
        return cid, len(f0), None
    except Exception as e:  # noqa: BLE001
        return cid, 0, f"{type(e).__name__}: {e}"


def warm_cache(clip_ids, cache_dir: str = DEFAULT_CACHE_DIR,
               clips_dir: str = CLIPS_DIR, jobs: int = 1) -> None:
    """Populate the cache, optionally across processes (pyin is the bottleneck)."""
    todo = [c for c in clip_ids if not is_cached(c, cache_dir)]
    if not todo:
        return
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[f0] computing pyin for {len(todo)}/{len(clip_ids)} clips "
          f"(jobs={jobs})", flush=True)
    if jobs <= 1:
        for i, cid in enumerate(todo, 1):
            cid, n, err = _warm((cid, cache_dir, clips_dir))
            print(f"[f0] {i}/{len(todo)} {cid}: {n} frames {err or ''}", flush=True)
        return
    from concurrent.futures import ProcessPoolExecutor
    args = [(c, cache_dir, clips_dir) for c in todo]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for i, (cid, n, err) in enumerate(ex.map(_warm, args), 1):
            print(f"[f0] {i}/{len(todo)} {cid}: {n} frames {err or ''}", flush=True)


if __name__ == "__main__":  # pre-warm the cache: python f0_cache.py [jobs]
    import json
    import sys

    ids = sorted(json.load(open(MANIFEST))["clips"].keys())
    warm_cache(ids, jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 1)
