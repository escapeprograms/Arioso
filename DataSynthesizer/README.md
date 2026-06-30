# DataSynthesizer — memory palace

Builds Arioso's training set: per clip, the **ground-truth violin audio + its mel** and
two **aligned informed-prior mels** (quantized and pitch-bend) plus an **onset mask** —
the features the flow-matching model learns to map between (prior mel → target mel).

Raw material is the vendored `violin-transcription` repo (paper *High-Resolution Violin
Transcription using Weak Labels*, ISMIR 2023), cloned under `external/violin-transcription`
(a peer of `external/BigVGAN`). It ships **MIDI weak-labels only, no audio**:
`dataset/{Kayser,Paganini,Wohlfahrt}/*.mid`, 1,021 clips. Each filename encodes where the
audio lives and which slice to use:

```
{Composer}_{Catalog}_{Performer}_{YouTubeID}-{startSec}-{endSec}.mid
Kayser_Op20-01_AlexandrosIakovou_O105paQOHCE-0004-0064.mid
```

The MIDI note times are already in real seconds, time-aligned to that `[start,end]` audio
slice (t=0 == startSec). MIDI is multi-track/polyphonic (one note-stream per "instrument")
and carries dense pitch bends — rendered into the **pitch-bend** prior and dropped from the
**quantized** prior; we save a mel of each (see design choices).

This directory is a **Python package** (`__init__.py`); run its modules with `python -m
DataSynthesizer.<module>` from the project root so intra-package imports resolve.

## Design choices (confirmed with the user)

- **Two prior flavors, saved as mels.** *Quantized* (one constant frequency per note, bends
  ignored — clean and score-like) and *pitch-bend* (instantaneous frequency follows the
  pitch-wheel, carrying vibrato/slides). Both may help training; the model maps a prior mel
  to the target mel, so we save mels — **not prior audio**.
- **Naive `scipy.signal.sawtooth`** (no anti-aliasing). High notes alias; acceptable for a prior.
- **Mels via `common.vocoder.mel_spectrogram`** — the single source of truth that matches the
  BigVGAN checkpoint (`[N_MELS=128, T]`, hop 512). Target and both prior mels share frame count `T`.
- **Targets are level-normalized.** Clips come from many YouTube channels at different volumes,
  so each downloaded track is normalized **once** to a target RMS measured over voiced
  (non-silent) segments, re-saved before any clip is trimmed (`voiced_rms_normalize`).
- **44.1 kHz mono**; the GT/target wav is 16-bit PCM, features are `.npy` float32.
- **Onset alignment is automatic.** The MIDI is only roughly aligned, so we measure the residual
  offset from the note onsets (cross-correlation, from the quantized prior) and shift **both**
  priors by it. The onset method was manually verified on enough clips to trust without a
  per-clip human check.
- **Onset mask** training signal: 1 on each onset frame, exponential decay to ~0 over a
  `ONSET_DECAY_MS` (50 ms) support window, at mel granularity.

## Pipeline / data flow

```
.mid ──parse (clip_name)──> (youtube_id, start, end, composer/catalog/performer)
   │
   ├ download_audio.fetch_clip ─> data/gt/<base>.wav            (download cached+normalized, trim)
   │        └ features.mel_for_training(y_gt) ───────────────> data/target_mel/<base>.npy
   │
   ├ synthesizePrior.render_prior(total_samples=len(y_gt)) ─┐  (quantized)
   ├ synthesizePrior.render_prior_bend(  …len(y_gt)) ───────┤  (pitch-bend)
   │   offset = -onset_align.estimate_offset_seconds(quant, gt)│  shift BOTH by `offset`
   │        ├ mel_for_training(quant) ──────────────────────> data/prior_mel_quant/<base>.npy
   │        └ mel_for_training(bend)  ──────────────────────> data/prior_mel_bend/<base>.npy
   │
   └ features.build_onset_mask(note_onsets, offset, T) ─────> data/prior_onset/<base>.npy
            │
        build_dataset orchestrates all clips ──────────────> data/manifest.csv
```

Both priors are rendered to the GT's exact sample count, so each pair starts sample-for-sample
aligned; the residual global offset (estimated once from the quantized prior) is applied to both
so they stay mutually aligned. Prior audio is never written — only the mels.

## Shared infrastructure

Audio I/O, the canonical sample rate, and the mel contract live in the **top-level
`common/` package** (shared with the training code), not in this package — see
`common/README.md`. Import them, don't re-implement:
`from common.audio_io import load_mono, write_pcm16, normalize, voiced_rms_normalize`,
`from common.vocoder import mel_spectrogram` (the only correct mel for the vocoder), and
`from common.config import SR`.

### config.py — pipeline constants (+ re-exported SR)
Re-exports `SR`/`HOP` from `common.config` (so modules keep one `from .config import SR, ...`
line) and owns the build-specific constants: `FADE_MS=5.0`, `BOOKS`, `DEFAULT_DATASET`,
`DEFAULT_OUT`, the target-normalization knobs `TARGET_RMS_DBFS=-20.0` / `VOICED_TOP_DB=40.0`,
and the onset-mask knobs `ONSET_DECAY_MS=50.0` / `ONSET_DECAY_FLOOR=0.01`. The peak target
lives in `common` (`DEFAULT_PEAK`, also the post-normalization clip guard). It also owns the
**Arioso prior** knobs — `PRIOR_ANTI_ALIAS=True` / `PRIOR_ENVELOPE="rect"` /
`PRIOR_LEVEL_MATCH="masked_rms"` (assembled by `synthesizePrior.quantized_prior`) and the prior
build's output dirs `PRIOR_MEL_DIR` / `ONSETS_DIR` — so `TARGET_RMS_DBFS` is the **single source
of truth** shared between GT loudness normalization and the prior's masked-RMS level match.

### clip_name.py — the one filename parser
- `parse_clip_name(path)` → `ClipName(youtube_id, start, end, basename, composer, catalog,
  performer)`. The id is the **11 chars** before the trailing `-start-end` (YouTube ids are
  always 11 chars and may contain BOTH `_` and `-`, so splitting on `_` is wrong — validated
  across all 1,021 files). Centralizes the convention that download + build both rely on.

## Files

### synthesizePrior.py — MIDI → sawtooth prior via a composable `PriorSynth`
A **Strategy**-pattern pipeline: one concrete `PriorSynth` orchestrator keeps the per-note
summation loop (summing across instrument tracks reproduces double-stops/polyphony) and delegates
each step to an injected, `Protocol`-typed component — swap any axis without a subclass explosion:
- `PitchTrajectory` → per-note f0 curve: `Quantized` (constant MIDI pitch, baseline) | `PitchBend`
  (pitch-wheel-following: vibrato/slides, ±2 semitones).
- `SourceSynth` → unit saw from an f0 curve: `NaiveSaw` (`scipy.signal.sawtooth`) | `BandlimitedSaw`
  (polyBLEP, removes fold-back aliasing). Both phase-accumulate via an **exclusive prefix-sum phase**
  so a constant f0 reduces exactly to the old `arange(n)·dt` math (outputs stay numerically identical).
- `Envelope` → `HardGate` ("rect", hard on/off) | `Fade` (~5 ms anti-click ramp via `_fade_envelope`).
- `BodyFilter` → `Identity` (the no-EQ baseline; the seam for a future static body-EQ).
- `Leveler` (`fit`/`apply`) → `MaskedRMS(target_rms_dbfs)` (scale so sounding-frame RMS hits the
  target) | `Peak` (legacy peak-normalize to `DEFAULT_PEAK`). The level match is a component, not
  baked into render. The mel front-end is **not** in the pipeline — callers mel after any alignment shift.
- `quantized_prior(anti_alias=, envelope=, level_match=, target_rms_dbfs=, sr=)` — factory that
  assembles the spec-baseline quantized pipeline from the `PRIOR_*` config knobs.
- `render_prior(midi_path, sr=44100, total_samples=None)` / `render_prior_bend(...)` — thin wrappers
  for the legacy **peak-normalized** quantized / pitch-bend priors (used by `build_dataset`).
  `total_samples` defaults to the MIDI end time; pass the GT length to force exact pair alignment.
- `note_onsets(midi_path)` — sorted, de-duplicated note onset times (seconds) across instruments;
  the exact onsets the prior carries, used to build the onset mask.
- `synthesize_to_file(midi_path, out_path, ...)` — `render_prior` + `common.audio_io.write_pcm16`.
- CLI: `python -m DataSynthesizer.synthesizePrior clip.mid -o clip_prior.wav` (quantized).

### build_prior.py — Arioso prior features over the dataset (one-time pass)
- `build(...)` / `process_clip(row, ...)` — pass over `manifest.csv` (status==ok): assemble the
  spec-faithful prior via `quantized_prior` (anti-aliased saw + masked-RMS to `TARGET_RMS_DBFS`),
  shift by the manifest `offset_ms`, mel it → `data/prior_mel_arioso/<base>.npy`, and write aligned
  onset frames → `data/onsets_arioso/<base>.npy`. Reuses the manifest offset (no re-estimation);
  resumable + skip-existing. CLI flags `--no-anti-alias` / `--envelope` / `--level-match` for ablations.
- CLI: `python -m DataSynthesizer.build_prior --limit 4` (smoke) | `python -m DataSynthesizer.build_prior` (full).

### download_audio.py — obtain, level-normalize + trim the GT violin audio (step 1)
- `download_full_audio(youtube_id, cache_dir, sr=44100, audio_codec="wav")` — download a whole
  video's audio once via `yt_dlp` + ffmpeg, cached as `cache_dir/{id}.wav`; reused on repeat
  calls (one video → many clips ⇒ one download). A *freshly* downloaded track is
  `voiced_rms_normalize`d and re-saved in place (mono @ sr) **before** trimming, so all its
  clips share one loudness. Mirrors `download_youtube` in `external/violin-transcription/musc/model.py`.
- `fetch_clip(midi_path, cache_dir, out_path=None, sr=44100)` — parse the clip name, download
  (cached + normalized), then `librosa.load(..., offset=start, duration=end-start)` to mono @ sr,
  and `common.audio_io.write_pcm16`. Returns `(out_path, y, sr)`; feed `len(y)` to the renderers.
- CLI: `python -m DataSynthesizer.download_audio clip.mid -o clip_gt.wav`.

### features.py — training-feature builders (mel + onset mask)
- `mel_for_training(wav)` — BigVGAN mel of `wav` as a `[N_MELS, T]` float32 array, via
  `common.vocoder.mel_spectrogram` (drops the batch dim, to numpy).
- `build_onset_mask(onset_times, applied, n_frames, sr, hop, decay_ms, floor)` — onset-mask
  signal on the mel grid: 1 on each onset frame, exponential decay to `floor` over `decay_ms`
  then hard 0. `onset_times` are shifted by `applied` (the prior's offset) and combined with max.

### onset_align.py — onset alignment (all in-memory, array-first)
- `estimate_offset_seconds(prior, gt, sr=44100, max_lag_s=1.0)` — cross-correlate
  onset-strength envelopes; positive ⇒ prior lags GT.
- `shift_samples(y, offset_seconds, sr=44100)` — shift a waveform in time (pad/truncate, same
  length). Positive delays; negative advances. Pure array op (no I/O).
- `align_prior_to_gt(prior, gt, sr=44100)` — estimate the offset and apply its **negation**
  (advance a lagging prior into alignment); returns `(aligned_prior, applied_seconds)`.
  `build_dataset` inlines `estimate_offset_seconds` + `shift_samples` so the one offset drives
  both priors; this fn is the convenience single-prior / CLI path.
- CLI: `python -m DataSynthesizer.onset_align prior.wav gt.wav` (report only; add `--apply`
  to write the shifted prior, `-o` for the output path).
- The f0/pitch ("frequency alignment") path and the QC plotting were removed from this module;
  the mel-spectrogram + onset plots live in `visualizations.ipynb`.

### build_dataset.py — orchestrate all clips → features + manifest
- `build(dataset_root, out_dir="data", books=(...), limit=None, sr=44100, overwrite=False)` —
  walk the books, process each clip, write `data/manifest.csv` (flushed every clip, so runs are
  **resumable** and an unavailable video is logged + skipped, never fatal).
- `process_clip(midi_path, out_dir, cache_dir, ...)` — for one clip: fetch + level-normalize +
  save GT and its mel; render both priors to the GT length; estimate the offset from the
  quantized prior and shift both; save both prior mels; save the onset mask. Returns the
  manifest row (incl. `n_frames`, `offset_ms`); skips clips already done (`status="exists"`,
  detected by all five outputs existing) unless `overwrite`.
- `_write_manifest(path, rows)` — (re)write the CSV.
- Manifest columns: `basename, book, composer, catalog, performer, youtube_id, start_sec,
  end_sec, duration_sec, n_samples, n_frames, offset_ms, gt_path, target_mel_path,
  prior_mel_quant_path, prior_mel_bend_path, prior_onset_path, status`.
- CLI: `--books`, `--limit` (smoke test), `--out-dir`, `--overwrite`.
- Output layout: `data/gt/`, `data/target_mel/`, `data/prior_mel_quant/`, `data/prior_mel_bend/`,
  `data/prior_onset/`, `data/_cache/` (normalized downloads), `data/manifest.csv`. `data/` is
  generated and large — not intended for version control.

### visualizations.ipynb — alignment QC plots (manual eyeballing)
Notebook for spot-checking a clip: loads the saved `.npy` features (`target_mel`,
`prior_mel_quant`, `prior_mel_bend`, `prior_onset`) and overlays the onset mask on the target
mel so the spikes can be checked against note attacks. Launch Jupyter from the project root so
`import DataSynthesizer...` resolves.

## Run

```bash
PY="C:/Users/archi/Miniconda3/envs/ai-violin/python.exe"   # use the ai-violin env
"$PY" -m DataSynthesizer.build_dataset --books Kayser --limit 2   # smoke test
"$PY" -m DataSynthesizer.build_dataset                            # full build (~1021 clips)
"$PY" -m DataSynthesizer.onset_align prior.wav gt.wav             # report a (prior, GT) offset
```

> **Adopting the level-normalization change:** existing `data/_cache/` downloads were saved
> before normalization and won't be re-normalized on a cache hit. Clear `data/_cache/` once (and
> rebuild with `--overwrite`) so every track is normalized.

## Dependencies & caveats

- Env **ai-violin**: `pretty_midi`, `yt-dlp`, plus `scipy`, `numpy`, `soundfile`, `librosa`,
  `matplotlib` (notebook only). **ffmpeg** must be on PATH. (parselmouth is no longer needed
  now that the f0 path is gone.)
- **yt-dlp JS runtime**: yt-dlp warns "No supported JavaScript runtime" and downloads a basic
  audio stream. It worked in testing, but for a full build some videos may need a JS runtime
  (install `deno`) to expose all formats. Unavailable/region-blocked videos are skipped + logged.
- **Two prior mels**: the *quantized* prior has no vibrato by design (the GT supplies it); the
  *pitch-bend* prior carries the pitch-wheel's vibrato/slides. Both mels are saved so training
  can use either/both.
- **Mels must come from `common.vocoder.mel_spectrogram`** — it matches the BigVGAN checkpoint;
  a re-implemented mel (e.g. plain librosa) would not vocode correctly. This pulls in `torch` +
  the vendored BigVGAN at build time.
- **Naive sawtooth** ⇒ aliasing above the upper register; acceptable for a prior, swap in a
  band-limited oscillator if it matters.
- GT clips can be slightly shorter than `end-start` if the source video is short; both priors
  are matched to the actual GT length, so the mels stay aligned.
- **Onset offset sign**: `estimate_offset_seconds` is positive when the prior lags; we apply its
  negation to advance both priors. A re-estimate on an aligned prior should be ≈0 ms.
- **Voiced-RMS normalization** measures level over non-silent segments (`librosa.effects.split`,
  `VOICED_TOP_DB`) and applies one global gain, with a `DEFAULT_PEAK` clip guard; a very quiet
  source may land below `TARGET_RMS_DBFS` rather than clip.
