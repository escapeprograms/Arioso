# DataSynthesizer — memory palace

The **synthetic** producer of Arioso training data: it turns the vendored score-MIDI
weak-labels into a **standard dataset root** (`common.dataset_schema` layout) that Arioso
consumes. Per clip it builds the **ground-truth violin audio + its mel** and an **aligned
informed-prior mel** plus an **onset frame index** — the features the flow-matching model
learns to map between (prior mel → target mel). It is one of two producers (the other is
`Labeler/`, which emits genuine human ground truth); the two never import each other and
agree only through `common/`.

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
and carries dense pitch bends — dropped from the (quantized) Arioso prior, carried in the
optional pitch-bend prior variant.

This directory is a **Python package** (`__init__.py`); run its modules with `python -m
DataSynthesizer.<module>` from the project root so intra-package imports resolve.

## Role: synthetic standard-root producer

The build emits everything into one dataset root (default `data/`, migrated to the standard
layout in `Data/`). Two of the shared building blocks that used to live here have **moved to
`common/`** (so the Labeler and Arioso reuse the exact same code without importing this
package) — import them from there, do not re-add them:

- **Prior synthesis** → `common.prior` (`PriorSynth`, `quantized_prior`, `render_prior*`,
  `note_onsets`, and the `PRIOR_*` / `FADE_MS` knobs). See `common/README.md`.
- **Onset alignment** → `common.onset_align` (`estimate_offset_seconds`, `shift_samples`,
  `align_prior_to_gt`).
- **The mel front-end** → `common.vocoder.mel_frames` (`[N_MELS, T]` float32, matches the
  BigVGAN checkpoint). `features.py` keeps only the pass-1 onset-**mask** builder.
- **The dataset-root contract** (dir names, `manifest.json` schema, encodings) →
  `common.dataset_schema`. This module never hardcodes a layout string.

## Design choices (confirmed with the user)

- **Prior saved as a mel, not audio.** The model maps a prior mel to the target mel, so we
  save mels. The Arioso prior is a *quantized* saw (one constant frequency per note, bends
  ignored — clean and score-like); a *pitch-bend* variant (instantaneous frequency follows
  the pitch-wheel) is also rendered by `build_dataset` as a legacy pass-1 output.
- **Mels via `common.vocoder.mel_frames`** — the single source of truth that matches the
  BigVGAN checkpoint (`[N_MELS=128, T]`, hop 512). Target and prior mels share frame count `T`.
- **Targets are level-normalized.** Clips come from many YouTube channels at different volumes,
  so each downloaded track is normalized **once** to `TARGET_RMS_DBFS` measured over voiced
  (non-silent) segments, re-saved before any clip is trimmed (`voiced_rms_normalize`). This is
  the cross-producer loudness contract in `common.config` — the prior is masked-RMS matched to
  the **same** target so prior/target levels agree.
- **44.1 kHz mono**; the GT/target wav is 16-bit PCM, features are `.npy` float32.
- **Onset alignment is automatic.** The MIDI is only roughly aligned, so `build_dataset`
  measures the residual offset from the note onsets (cross-correlation, from the quantized
  prior, `common.onset_align`) once and records it as `offset_ms` in the manifest; `build_prior`
  reuses that offset (no re-estimation). The onset method was manually verified on enough clips
  to trust without a per-clip human check.
- **Onset mask** (pass-1 legacy): 1 on each onset frame, exponential decay to ~0 over
  `ONSET_DECAY_MS` (50 ms), at mel granularity.

## Pipeline / data flow

Two passes then a manifest export (no VioPTT technique pass — that was removed):

```
.mid ──parse (clip_name)──> (youtube_id, start, end, composer/catalog/performer)

Pass 1  build_dataset (per clip):
   ├ download_audio.fetch_clip ─> data/gt/<base>.wav        (download cached+normalized, trim)
   │        └ mel_frames(y_gt) ─────────────────────────────> data/target_mel/<base>.npy
   ├ common.prior.render_prior(total_samples=len(y_gt))  ─┐  (quantized, naive saw, legacy)
   ├ common.prior.render_prior_bend( …len(y_gt))  ────────┤  (pitch-bend, legacy)
   │   offset = -onset_align.estimate_offset_seconds(quant, gt)│ shift BOTH by `offset`
   │        ├ mel ──> data/prior_mel_quant/<base>.npy · data/prior_mel_bend/<base>.npy
   └ features.build_onset_mask(...) ────────────────────────> data/prior_onset/<base>.npy
        │  all clips ──────────────────────────────────────> data/manifest.csv   (build log)

Pass 2  build_prior (per ok row of manifest.csv):
   └ common.prior.quantized_prior() (shaped additive saw + masked-RMS -> TARGET_RMS_DBFS)
        + reuse manifest offset_ms shift + mel ─────────────> data/prior_mel/<base>.npy
        │                              └ same signal as wav ─> data/prior_wav/<base>.wav
        + aligned onset frames ─────────────────────────────> data/onsets/<base>.npy

Export  export_manifest:  manifest.csv (ok rows) ────────────> data/manifest.json  (schema v1)
```

`prior_mel/` + `onsets/` are what **Arioso trains on**; `prior_mel_quant` / `prior_mel_bend` /
`prior_onset` are the naive-saw *pass-1 legacy* outputs (the level-mismatched, aliased prior the
Arioso spec replaced) — kept for reference, purgeable via `migrate_root --purge-legacy`.
`manifest.csv` stays the build's **internal log**; Arioso reads only the exported `manifest.json`.

## Shared infrastructure

Audio I/O, the sample rate, the mel/vocoder front-end, the prior synth, the onset aligner, and
the dataset-root standard all live in the **top-level `common/` package** — see
`common/README.md`. Import them, don't re-implement:
`from common.audio_io import load_mono, write_pcm16, voiced_rms_normalize`,
`from common.vocoder import mel_frames`, `from common.prior import quantized_prior, note_onsets`,
`from common.onset_align import estimate_offset_seconds, shift_samples`,
`from common.dataset_schema import DIR_PRIOR_MEL, DIR_PRIOR_WAV, DIR_ONSETS, write_manifest`, and
`from common.config import SR, TARGET_RMS_DBFS`.

### config.py — pipeline-specific constants (+ re-exported SR/HOP)
Re-exports `SR`/`HOP` from `common.config` (so modules keep one `from .config import SR, ...`
line) and owns only what remains **specific to building the synthetic dataset**: the onset-mask
knobs `ONSET_DECAY_MS=50.0` / `ONSET_DECAY_FLOOR=0.01`, the dataset paths `BOOKS`,
`DEFAULT_DATASET`, `DEFAULT_OUT`. The prior-shape knobs (`PRIOR_*`, `FADE_MS`) moved to
`common.prior`; the loudness contract (`TARGET_RMS_DBFS`, `VOICED_TOP_DB`) and the mel contract
to `common.config`; the standard on-disk dir names to `common.dataset_schema`. (VioPTT technique
knobs are gone entirely.)

### clip_name.py — the one filename parser
- `parse_clip_name(path)` → `ClipName(youtube_id, start, end, basename, composer, catalog,
  performer)`. The id is the **11 chars** before the trailing `-start-end` (YouTube ids are
  always 11 chars and may contain BOTH `_` and `-`, so splitting on `_` is wrong — validated
  across all 1,021 files). Centralizes the convention that download + build both rely on.

## Files

### build_dataset.py — pass 1: orchestrate all clips → GT/target + legacy priors + manifest.csv
- `build(dataset_root, out_dir="data", books=(...), limit=None, sr=44100, overwrite=False)` —
  walk the books, process each clip, write `data/manifest.csv` (flushed every clip, so runs are
  **resumable** and an unavailable video is logged + skipped, never fatal).
- `process_clip(midi_path, out_dir, cache_dir, ...)` — for one clip: fetch + level-normalize +
  save GT and its mel; render the quantized + pitch-bend priors to the GT length (`common.prior`);
  estimate the offset from the quantized prior and shift both; save both prior mels; save the
  onset mask. Returns the manifest row (incl. `n_frames`, `offset_ms`); skips clips already done
  (`status="exists"`, detected by all five outputs existing) unless `overwrite`.
- Manifest columns: `basename, book, composer, catalog, performer, youtube_id, start_sec,
  end_sec, duration_sec, n_samples, n_frames, offset_ms, gt_path, target_mel_path,
  prior_mel_quant_path, prior_mel_bend_path, prior_onset_path, status`.
- CLI: `--books`, `--limit` (smoke), `--out-dir`, `--overwrite`.
- Output layout: `data/gt/`, `data/target_mel/`, `data/prior_mel_quant/`, `data/prior_mel_bend/`,
  `data/prior_onset/`, `data/_cache/` (normalized downloads), `data/manifest.csv`.

### build_prior.py — pass 2: the Arioso prior features (one-time pass)
- `build(...)` / `process_clip(row, out_dir, dataset_root, ...)` — pass over `manifest.csv`
  (status==ok): assemble the spec-faithful prior via `common.prior.quantized_prior` (shaped
  additive saw + masked-RMS to `TARGET_RMS_DBFS`), shift by the manifest `offset_ms`, mel it via
  `common.vocoder.mel_frames` → `data/prior_mel/<base>.npy` (dir name `DIR_PRIOR_MEL`), write that
  same buffer as audio → `data/prior_wav/<base>.wav` (`DIR_PRIOR_WAV`), and write
  aligned onset frames → `data/onsets/<base>.npy` (`DIR_ONSETS`). Reuses the manifest offset (no
  re-estimation); resumable + skip-existing (all three outputs must be present). CLI flags
  `--source` / `--harmonic-law` / `--alpha` / `--corner-nc` / `--corner-p` / `--envelope` /
  `--level-match` for ablations.
- **`prior_wav/` — the rendered prior as audio.** The **post-shift** buffer only: the pre-shift
  render is in score time, so writing that would hand consumers a wav silently misaligned with
  `gt/` and `prior_mel/`. It is the same signal the mel is computed from, mono PCM16 @ `SR`, so a
  consumer can re-mel the prior/target pair from audio — the one consumer today is Arioso's
  pitch-shift augmentation (`Arioso.pitch_aug`). Guarded by a **peak assert**: PCM16 clamps
  silently past ±1.0, which would make the wav a different signal from the float the mel came
  from, so `peak > 1.0` raises rather than quietly diverging.
- **The synthetic root needs no regeneration.** `prior_wav/` is optional in the dataset-root
  contract and a root lacking it is simply never augmented (`Arioso.pitch_aug.wavs_available` is
  the graceful-fallback gate), so the existing `Data/` root stays fully trainable as-is — it has
  **not** been backfilled. Backfilling is just a plain re-run of `build_prior`: the skip check
  fires only when *all three* outputs exist, so every clip missing `prior_wav` re-renders (no
  `--overwrite` needed) — a full ~888-clip pass, which is why it has not been spent.
- CLI: `python -m DataSynthesizer.build_prior --limit 4` (smoke) | `python -m DataSynthesizer.build_prior` (full).

### export_manifest.py — build log → standard `manifest.json`
- `build_manifest(out_dir)` / `export_manifest(out_dir)` — project the `status=="ok"` rows of
  `manifest.csv` into a standard `manifest.json` via `common.dataset_schema.write_manifest`:
  `name="synthetic"`, `signals=[]` (the synthetic corpus carries no conditioning tracks), split
  key `piece = "{composer}/{catalog}"`, and each clip's `source` block preserving provenance
  (book / performer / youtube id / start-end / measured `offset_ms`). Idempotent — safe to run
  after every `build_prior`. Only complete/trainable clips are listed.
- CLI: `python -m DataSynthesizer.export_manifest [--out-dir Data]`.

### migrate_root.py — one-shot migration of the existing `Data/` root (already run)
Brings the pre-schema synthetic root up to the standard layout with **cheap renames only, no mel
rebuilds**: `prior_mel_arioso → prior_mel`, `onsets_arioso → onsets`; generate `manifest.json`
(reusing `export_manifest`); copy `arioso_split.json → split.json` byte-identical (preserves the
held-out split exactly); **delete `technique_arioso`** (deprecated VioPTT artifacts, always). The
pass-1 legacy dirs (`prior_mel_quant` / `prior_mel_bend` / `prior_onset`) are kept and their sizes
printed unless `--purge-legacy`. Every step idempotent; `--dry-run` touches nothing; guarded to a
real DataSynthesizer root (must have `manifest.csv`); the producer trees `gt_arky` / `recorded` /
`studio_projects` and the kept `gt` / `target_mel` are never touched.
- CLI: `python -m DataSynthesizer.migrate_root --root Data --dry-run` then without `--dry-run`.

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

### features.py — pass-1 onset-mask builder
The mel front-end moved to `common.vocoder.mel_frames`; what remains is the pass-1 onset-mask:
- `build_onset_mask(onset_times, applied, n_frames, sr, hop, decay_ms, floor)` — onset-mask
  signal on the mel grid: 1 on each onset frame, exponential decay to `floor` over `decay_ms`
  then hard 0. `onset_times` are shifted by `applied` (the prior's offset) and combined with max.
  (The Arioso model trains on `build_prior`'s onset frame **indices** in `onsets/`, not this
  soft mask — the mask is a `build_dataset` pass-1 artifact.)

### visualizations.ipynb — alignment QC plots (manual eyeballing)
Notebook for spot-checking a clip: loads the saved `.npy` features and overlays the onset mask on
the target mel so the spikes can be checked against note attacks. Launch Jupyter from the project
root so `import DataSynthesizer...` resolves.

## Run

```bash
PY="C:/Users/archi/Miniconda3/envs/ai-violin/python.exe"   # use the ai-violin env
"$PY" -m DataSynthesizer.build_dataset --books Kayser --limit 2   # pass 1: smoke test
"$PY" -m DataSynthesizer.build_dataset                            # pass 1: full build (~1021 clips)
"$PY" -m DataSynthesizer.build_prior --limit 4                    # pass 2: Arioso prior features (smoke)
"$PY" -m DataSynthesizer.build_prior                              # pass 2: full (~888 ok clips)
"$PY" -m DataSynthesizer.export_manifest                         # manifest.csv -> standard manifest.json
```

> **Pass ordering:** `build_prior` needs `build_dataset`'s `manifest.csv` (it reads the `ok`
> rows + each clip's `offset_ms` / `n_samples`). Run `export_manifest` after any `build_prior`
> pass so `manifest.json` reflects the current clip set.

> **Adopting the level-normalization change:** existing `data/_cache/` downloads saved before
> normalization won't be re-normalized on a cache hit. Clear `data/_cache/` once (and rebuild
> with `--overwrite`) so every track is normalized.

## Dependencies & caveats

- Env **ai-violin**: `pretty_midi`, `yt-dlp`, plus `scipy`, `numpy`, `soundfile`, `librosa`,
  `matplotlib` (notebook only). **ffmpeg** must be on PATH.
- **yt-dlp JS runtime**: yt-dlp warns "No supported JavaScript runtime" and downloads a basic
  audio stream. It worked in testing, but for a full build some videos may need a JS runtime
  (install `deno`) to expose all formats. Unavailable/region-blocked videos are skipped + logged.
- **Mels must come from `common.vocoder.mel_frames`** — it matches the BigVGAN checkpoint;
  a re-implemented mel (e.g. plain librosa) would not vocode correctly. This pulls in `torch` +
  the vendored BigVGAN at build time.
- **Naive sawtooth** (pass-1 legacy priors) ⇒ aliasing above the upper register; the Arioso prior
  (`build_prior`) uses the shaped band-limited additive saw instead.
- GT clips can be slightly shorter than `end-start` if the source video is short; the prior is
  matched to the actual GT length, so the mels stay aligned.
- **Onset offset sign**: `estimate_offset_seconds` is positive when the prior lags; we apply its
  negation to advance the prior. A re-estimate on an aligned prior should be ≈0 ms.
- **Voiced-RMS normalization** measures level over non-silent segments (`librosa.effects.split`,
  `VOICED_TOP_DB`) and applies one global gain, with a `DEFAULT_PEAK` clip guard; a very quiet
  source may land below `TARGET_RMS_DBFS` rather than clip.
