"""Segment caching: silence-gap segmentation, content hashing, stitch + GC.

The render unit is a **segment**: a maximal run of notes with no inter-note
silence gap ``>= cfg.cache.gap_s`` seconds, padded ``cfg.cache.pad_s`` seconds into
the surrounding silence on each side. Each segment is hashed over its (position-
independent) note content + render params into ``seg_<hash>.wav`` under
``render/cache/``; on a re-render only segments whose content changed miss the
cache and are re-synthesized, and ``mix.wav`` is re-stitched from the cached +
fresh segment WAVs. Editing one note therefore re-renders only its segment.

Two design decisions worth stating up front:

* **Segment-relative hashing.** The hash uses each note's beat position *relative
  to its segment's first onset* (``start_beat - segment_start_beat``), so shifting a
  whole phrase in time without changing its content still hits cache — the rendered
  WAV is position-independent and gets placed at the right absolute offset only at
  stitch time. Bend/vibrato are already note-relative, so they need no adjustment.
  A note's ``env_dct`` (energy envelope) is folded into the hash *conditionally* —
  only when present — so pre-envelope projects keep their exact segment hashes.

* **No per-segment renormalization.** :class:`common.prior.MaskedRMS`
  already normalizes each rendered segment's prior to the same ``-20`` dBFS voiced
  level, and the model + vocoder roughly preserve that level, so per-segment levels
  are already consistent. Stitching therefore sums the segments as-is and applies
  only a single **peak-safety clamp** on the mix (scale down if it would clip, never
  boost) — an aggressive per-mix RMS renormalization would change the perceived
  loudness on every re-render, which is exactly what the segment cache exists to
  avoid. The residual risk (a segment whose voiced level drifts) is accepted for v1.

Torch-free (numpy + soundfile only); the heavy render lives in :mod:`Studio.render`.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import soundfile as sf

from common.config import DEFAULT_PEAK, SR

from .config import SCHEMA_VERSION
from .timing import beats_to_seconds

# Cache WAV naming: ``seg_<sha1hex>.wav`` under ``render/cache/``.
SEG_PREFIX = "seg_"
SEG_SUFFIX = ".wav"


def seg_wav_name(seg_hash: str) -> str:
    """Cache filename for a segment hash (``seg_<hash>.wav``)."""
    return f"{SEG_PREFIX}{seg_hash}{SEG_SUFFIX}"


# --- segmentation -----------------------------------------------------------

def _note_span_s(note: dict, bpm: float) -> tuple[float, float]:
    """``(start_s, end_s)`` of a note in seconds (length clamped non-negative)."""
    s = beats_to_seconds(float(note["start_beat"]), bpm)
    e = s + max(0.0, beats_to_seconds(float(note["len_beats"]), bpm))
    return s, e


def segment_notes(notes: list[dict], bpm: float, gap_s: float,
                  pad_s: float) -> list[dict]:
    """Group ``notes`` into cache segments split on silence gaps ``>= gap_s``.

    Notes are ordered by onset and merged into maximal runs: a new segment begins
    when the next note's onset is ``>= gap_s`` seconds past the running end of the
    current run (the union end, so overlapping/polyphonic notes stay together).
    Each segment gets ``pad_lead``/``pad_tail`` seconds clamped into the adjacent
    silence — an internal boundary's silence is split in half between the two
    segments touching it (so their pads never cross), the first segment's lead pad
    is clamped to ``[0, start_s]`` (never before ``t=0``), and the last segment's
    tail pad is the full ``pad_s`` (unbounded trailing silence).

    Returns ``[{notes, start_s, end_s, pad_lead, pad_tail}]`` in time order.
    """
    if not notes:
        return []
    ordered = sorted(
        notes,
        key=lambda n: (float(n["start_beat"]),
                       float(n["start_beat"]) + float(n["len_beats"]),
                       int(n.get("pitch", 0)), str(n.get("id", ""))),
    )
    raw: list[tuple[list[dict], float, float]] = []   # (notes, start_s, end_s)
    cur: list[dict] = []
    cur_start = cur_end = 0.0
    for n in ordered:
        s, e = _note_span_s(n, bpm)
        if not cur:
            cur, cur_start, cur_end = [n], s, e
        elif s - cur_end >= gap_s:
            raw.append((cur, cur_start, cur_end))
            cur, cur_start, cur_end = [n], s, e
        else:
            cur.append(n)
            cur_end = max(cur_end, e)
    if cur:
        raw.append((cur, cur_start, cur_end))

    out: list[dict] = []
    last = len(raw) - 1
    for i, (snotes, s_s, e_s) in enumerate(raw):
        if i == 0:
            pad_lead = min(pad_s, s_s)                       # never before t=0
        else:
            pad_lead = min(pad_s, (s_s - raw[i - 1][2]) / 2.0)
        if i == last:
            pad_tail = pad_s                                 # unbounded trailing silence
        else:
            pad_tail = min(pad_s, (raw[i + 1][1] - e_s) / 2.0)
        out.append({
            "notes": snotes,
            "start_s": s_s,
            "end_s": e_s,
            "pad_lead": max(0.0, pad_lead),
            "pad_tail": max(0.0, pad_tail),
        })
    return out


# --- content hashing --------------------------------------------------------

def _canon_bend(bend) -> list[dict]:
    pts = [{"beat": round(float(p.get("beat", 0.0)), 6),
            "semitones": round(float(p.get("semitones", 0.0)), 6)}
           for p in (bend or [])]
    pts.sort(key=lambda p: (p["beat"], p["semitones"]))
    return pts


def _canon_env(env):
    """Canonicalize a ``[c0, c1, c2]`` energy envelope, or ``None`` if absent/invalid.

    Mirrors :func:`_canon_bend`'s style: a valid 3-element list rounds to 6 dp; any
    other value (``None``, wrong length) returns ``None`` so the caller can omit the
    key entirely (see :func:`segment_hash` for why the key is conditional).
    """
    if not isinstance(env, (list, tuple)) or len(env) != 3:
        return None
    return [round(float(x), 6) for x in env]


def _canon_vibrato(vib) -> dict:
    v = vib or {}
    return {"depth_semitones": round(float(v.get("depth_semitones", 0.0)), 6),
            "rate_hz": round(float(v.get("rate_hz", 0.0)), 6),
            "onset_beats": round(float(v.get("onset_beats", 0.0)), 6)}


def _vocoder_cache_id(checkpoint_dir: str | None) -> str | None:
    """Identity of a vocoder for cache keying, or None for the HF stock baseline.

    ``checkpoint_dir`` is the loader-facing selection (:func:`common.vocoder.load_vocoder`'s
    argument, produced by :func:`Studio.library.vocoder_checkpoint_dir`): a local
    fine-tuned generator dir, or ``None`` / ``"hf"`` for the stock checkpoint. A
    local generator's audio genuinely differs from the stock one, so it must
    invalidate cached segments: it returns ``"<dirname>:<generator size>"`` (the
    size catches re-exported weights under the same name). The stock baseline has
    no id — see :func:`segment_hash` for why that keeps historical hashes intact.
    """
    if not checkpoint_dir or checkpoint_dir == "hf":
        return None
    gen = os.path.join(checkpoint_dir, "bigvgan_generator.pt")
    size = os.path.getsize(gen) if os.path.isfile(gen) else 0
    return f"{os.path.basename(os.path.normpath(checkpoint_dir))}:{size}"


def segment_hash(segment_notes_: list[dict], bpm: float, time_sig: dict,
                 model: str, checkpoint: str, prior_mode: str,
                 schema_version: int, knobs: dict,
                 vocoder: str | None = None) -> str:
    """SHA1 hex over a segment's canonical content + render params.

    Note beat positions are made **segment-relative** (each ``start_beat`` minus the
    segment's earliest onset) so an unchanged phrase moved in time keeps its hash.
    Notes are sorted deterministically; floats rounded to 6 dp to kill FP noise.
    Params folded in: ``bpm``, ``time_sig``, ``model``, ``checkpoint``,
    ``prior_mode``, ``schema_version``, the cache ``knobs`` (gap/pad) and — when
    the render uses a local fine-tuned vocoder — ``vocoder``, so any of them
    changing invalidates every segment (expected — e.g. a BPM change or a vocoder
    swap). ``vocoder`` is a **precomputed cache id** (:func:`_vocoder_cache_id`),
    not a Studio vocoder name; ``None`` omits the key entirely, which is the stock
    HF baseline (and the historical, pre-vocoder-selection payload).

    Per-note fields hashed: ``pitch``, ``start_beat`` (segment-relative),
    ``len_beats``, ``velocity``, ``technique``, ``bend``, ``vibrato``, ``pan`` and —
    *conditionally* — ``env`` (the ``[c0, c1, c2]`` energy envelope), which is
    included **only** for notes that carry an ``env_dct`` (D6). Omitting the key on
    env-less notes keeps their segment hashes byte-identical to pre-envelope
    projects; env-carrying notes intentionally re-hash.
    """
    notes = list(segment_notes_)
    base = min((float(n["start_beat"]) for n in notes), default=0.0)
    ordered = sorted(
        notes,
        key=lambda n: (float(n["start_beat"]), int(n.get("pitch", 0)),
                       float(n["len_beats"]), str(n.get("id", ""))),
    )
    canon_notes = []
    for n in ordered:
        cn = {
            "pitch": int(n["pitch"]),
            "start_beat": round(float(n["start_beat"]) - base, 6),
            "len_beats": round(float(n["len_beats"]), 6),
            "velocity": int(n.get("velocity", 100)),
            "technique": str(n.get("technique", "normal")),
            "bend": _canon_bend(n.get("bend")),
            "vibrato": _canon_vibrato(n.get("vibrato")),
            "pan": round(float(n.get("pan", 0.0)), 6),
        }
        # Conditional "env" key (D6): only notes that actually carry an env_dct add
        # it to the hash payload, so env-less projects keep their existing segment
        # hashes (no global re-render). Env-carrying notes intentionally re-hash —
        # their audio genuinely differs.
        env = _canon_env(n.get("env_dct"))
        if env is not None:
            cn["env"] = env
        canon_notes.append(cn)

    payload = {
        "notes": canon_notes,
        "bpm": round(float(bpm), 6),
        "time_sig": {"num": int(time_sig["num"]), "den": int(time_sig["den"])},
        "model": str(model),
        "checkpoint": str(checkpoint),
        "prior_mode": str(prior_mode),
        "schema_version": int(schema_version),
        "knobs": {"gap_s": round(float(knobs["gap_s"]), 6),
                  "pad_s": round(float(knobs["pad_s"]), 6)},
    }
    # Conditional "vocoder" key (same pattern as per-note "env"): only a local
    # fine-tuned vocoder adds it, so stock-vocoder hashes stay byte-identical to
    # historical ones (and revive their caches whenever a render selects "hf"
    # again). A fine-tuned vocoder intentionally re-hashes every segment — all
    # cached audio predating it is stale.
    voc = vocoder
    if voc is not None:
        payload["vocoder"] = voc
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


# --- planning ---------------------------------------------------------------

def plan_render(doc: dict, cfg, model: str, checkpoint: str, prior_mode: str,
                vocoder: str | None = None, cache_dir: str | None = None) -> dict:
    """Plan a render: segment ``doc``, hash each segment, mark cache hits.

    Returns ``{"segments": [...], "cache_dir": str, "total_duration_s": float}``
    where each segment entry is ``{hash, wav, cached, start_s, end_s, pad_lead,
    pad_tail, start_beat, end_beat, notes}``. ``cached`` is whether
    ``render/cache/seg_<hash>.wav`` already exists. ``cache_dir`` defaults to the
    project's ``render/cache/`` (derived from ``cfg`` + ``doc['project_id']``).

    ``vocoder`` is a Studio vocoder **name** (a dir under ``vocoders_root``, or
    ``"hf"``), defaulting to ``cfg.default_vocoder``; it is resolved to its cache
    id once here and folded into every segment hash, so switching vocoders
    invalidates the whole cache (and switching back revives it).
    """
    bpm = float(doc["bpm"])
    time_sig = doc.get("time_sig") or {"num": 4, "den": 4}
    schema_version = int(doc.get("schema_version", SCHEMA_VERSION))
    knobs = {"gap_s": cfg.cache.gap_s, "pad_s": cfg.cache.pad_s}
    # Lazy (like render_cache_dir below): library imports config, not the reverse.
    from .library import vocoder_checkpoint_dir
    voc_id = _vocoder_cache_id(
        vocoder_checkpoint_dir(cfg, vocoder or cfg.default_vocoder))
    if cache_dir is None:
        from .library import render_cache_dir
        cache_dir = render_cache_dir(cfg, doc["project_id"])

    segs = segment_notes(doc.get("notes", []), bpm, cfg.cache.gap_s, cfg.cache.pad_s)
    entries: list[dict] = []
    for seg in segs:
        h = segment_hash(seg["notes"], bpm, time_sig, model, checkpoint,
                         prior_mode, schema_version, knobs, vocoder=voc_id)
        wav = seg_wav_name(h)
        start_beat = min(float(n["start_beat"]) for n in seg["notes"])
        end_beat = max(float(n["start_beat"]) + float(n["len_beats"])
                       for n in seg["notes"])
        entries.append({
            "hash": h,
            "wav": wav,
            "cached": os.path.isfile(os.path.join(cache_dir, wav)),
            "start_s": seg["start_s"],
            "end_s": seg["end_s"],
            "pad_lead": seg["pad_lead"],
            "pad_tail": seg["pad_tail"],
            "start_beat": start_beat,
            "end_beat": end_beat,
            "notes": seg["notes"],
        })
    total_duration_s = max((e["end_s"] + e["pad_tail"] for e in entries), default=0.0)
    return {"segments": entries, "cache_dir": cache_dir,
            "total_duration_s": total_duration_s}


# --- stitching --------------------------------------------------------------

def _read_wav_mono(path: str) -> np.ndarray:
    """Read a cache WAV as a mono float32 array at its native rate (no resample)."""
    y, _sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return np.asarray(y, dtype=np.float32)


def stitch(segments: list[dict], cache_dir: str, total_duration_s: float,
           sr: int = SR) -> np.ndarray:
    """Place each segment WAV at its absolute offset into one float32 mix.

    A segment's WAV begins at ``start_s - pad_lead`` seconds (its padded lead), so
    it is written starting at that sample; gaps between segments stay zero. Where two
    consecutive placements overlap (their pads touch) the overlap is linearly
    crossfaded (outgoing down, incoming up). Finally one **peak-safety** pass scales
    the mix down if it exceeds ``DEFAULT_PEAK`` (never boosts) — see the module
    docstring for why there is no per-segment / per-mix RMS renormalization.
    """
    total = max(0, int(round(total_duration_s * sr)))
    mix = np.zeros(total, dtype=np.float32)
    filled_until = 0
    ordered = sorted(segments, key=lambda s: s["start_s"] - s["pad_lead"])
    for seg in ordered:
        y = _read_wav_mono(os.path.join(cache_dir, seg["wav"]))
        s0 = max(0, int(round((seg["start_s"] - seg["pad_lead"]) * sr)))
        if s0 >= total:
            continue
        L = min(len(y), total - s0)
        if L <= 0:
            continue
        y = y[:L]
        overlap = min(max(0, filled_until - s0), L)
        if overlap > 0:
            if overlap > 1:
                fade = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
            else:
                fade = np.array([0.5], dtype=np.float32)
            mix[s0:s0 + overlap] = mix[s0:s0 + overlap] * (1.0 - fade) + y[:overlap] * fade
            if L > overlap:
                mix[s0 + overlap:s0 + L] = y[overlap:]
        else:
            mix[s0:s0 + L] = y
        filled_until = max(filled_until, s0 + L)

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > DEFAULT_PEAK:
        mix = mix * (DEFAULT_PEAK / peak)
    return mix.astype(np.float32)


# --- garbage collection -----------------------------------------------------

def prune_cache(cache_dir: str, keep_hashes, max_files: int = 64,
                max_mb: float | None = None) -> list[str]:
    """Delete stale ``seg_*.wav`` beyond the file-count / size budget; keep manifest always.

    Segments in ``keep_hashes`` (the current manifest) are never deleted. Pruning runs
    in two passes over the remaining (stale) WAVs, oldest-mtime dropped first:

    * **count** — the newest stale WAVs are kept up to the ``max_files`` total budget
      (manifest + stale) and the rest are removed.
    * **size** — when ``max_mb`` is not ``None``, the remaining ``seg_*.wav`` total is
      summed and stale WAVs are dropped oldest-first while it exceeds ``max_mb`` MB.
      Manifest files are never dropped, so the total can legitimately stay over budget
      when the manifest alone exceeds it.

    Returns the list of deleted filenames (both passes).
    """
    if not os.path.isdir(cache_dir):
        return []
    keep = set(keep_hashes)
    kept: list[str] = []
    stale: list[str] = []
    for f in os.listdir(cache_dir):
        if not (f.startswith(SEG_PREFIX) and f.endswith(SEG_SUFFIX)):
            continue
        h = f[len(SEG_PREFIX):-len(SEG_SUFFIX)]
        (kept if h in keep else stale).append(f)
    # Newest first, so slicing/keeping from the front retains the freshest WAVs.
    stale.sort(key=lambda f: os.path.getmtime(os.path.join(cache_dir, f)), reverse=True)
    allowed = max(0, max_files - len(kept))
    deleted: list[str] = []
    for f in stale[allowed:]:
        try:
            os.remove(os.path.join(cache_dir, f))
            deleted.append(f)
        except OSError:
            pass
    surviving_stale = stale[:allowed]

    # Size pass: drop the oldest surviving stale WAV until the total (manifest +
    # stale) is under budget. Manifest files are never touched.
    if max_mb is not None:
        def _size(f: str) -> int:
            try:
                return os.path.getsize(os.path.join(cache_dir, f))
            except OSError:
                return 0

        budget = max_mb * 1e6
        total = sum(_size(f) for f in kept) + sum(_size(f) for f in surviving_stale)
        # Oldest last so we can pop() the oldest surviving stale WAV each iteration.
        while total > budget and surviving_stale:
            f = surviving_stale.pop()      # oldest-mtime stale (list is newest-first)
            sz = _size(f)
            try:
                os.remove(os.path.join(cache_dir, f))
                deleted.append(f)
                total -= sz
            except OSError:
                pass
    return deleted
