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
Finally it owns the **VioPTT technique** knobs (consumed by `build_techniques`): `TECHNIQUE_DIR`
(the output dir), `TECHNIQUE_CLASSES` — the id→name table below — `NO_TECHNIQUE_ID=4` /
`REST_ID=5`, `TECHNIQUE_PAD_S=0.02` (per-note slice padding fed to VioPTT),
`TECHNIQUE_REST_SNAP_FRAMES=10` (the legato/rest gap threshold), plus the **fixed-window** knobs
`TECHNIQUE_NOTE_SECONDS=2.0` and `TECHNIQUE_PEAK_NORMALIZE=True`. VioPTT's note model was **trained
on single notes pad/truncated to a fixed 2.0 s window then peak-normalized** (its RWC/MOSAPT note
datasets; `--mosapt_fixed_seconds` default 2.0), so we reproduce that construction rather than
VioPTT's own from-midi inference, which feeds variable-length un-normalized slices and skews short
notes to `spiccato`. `TECHNIQUE_NOTE_SECONDS=None` (CLI `--note-seconds 0`) selects the legacy
variable-length path (A/B only).

**Technique class ids** (`TECHNIQUE_CLASSES`, index = id): `0 flageolet`, `1 normal`,
`2 pizzicato`, `3 spiccato`, `4 no_technique`. Ids 0–4 are VioPTT's own
`DEFAULT_TECHNIQUE_CLASSES` in order (its class ids); `5 rest` is **ours** — VioPTT never
predicts it, it is only produced by the per-frame fill for stretches where nothing is sounding.

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

### technique.py — VioPTT playing-technique classifier wrapper + pure helpers
Thin wrapper around the pretrained **VioPTT** note-technique model (repo vendored under
`external/VioPTT`) so the build reuses VioPTT's own inference code (`infer_note_technique_from_midi`)
instead of reimplementing it, plus torch-free helpers to turn a MIDI score into per-note slice
spans and expand VioPTT's per-note predictions into a per-frame id track on the `target_mel` grid.
- **Two note sources, kept bit-identical.** The emitted onset frames MUST equal `onsets_arioso`
  (the clip enumerator keys off it), so `note_groups_from_midi` parses notes with **pretty_midi**
  and rounds with **np.round** exactly like `build_prior` — **not** VioPTT's raw-MIDI-event parser
  (`parse_midi_to_note_events`), which would drift. VioPTT is used only for the audio→technique step.
- **Mandatory transcription features.** The shipped note model was trained WITH transcription
  features, so it MUST be built `use_trans_features=True` / `trans_feat_dim=352` and fed features
  from the transcriptor checkpoint — the transcriptor is therefore also **mandatory** (a missing
  one is fatal). Building without trans-features makes `load_state_dict` fail on a shape mismatch.
- `_ensure_vioptt_on_path()` / `_import_vioptt()` — VioPTT's modules use **flat** imports, so both
  its `piano_transcription/pytorch` and `.../utils` dirs go on `sys.path` (mirrors
  `common.vocoder`'s BigVGAN vendoring), and importing them binds top-level `config` / `utilities`
  / `models_contrast` / `pytorch_utils` to VioPTT's copies (inert — repo code only imports
  package-qualified). Lazy (keeps the module torch-free for the pure helpers). Tripwire asserts
  after import: VioPTT `config.sample_rate == 16000` and its class list matches `TECHNIQUE_CLASSES[:5]`
  (catches sys.path shadowing / upstream drift).
- `NoteGroup` (NamedTuple: `onset_frame, end_frame, start_s, end_s, n_notes`) — one distinct
  rounded-onset frame's notes (chords / double-stops / sub-frame onsets merge into one group).
- `note_groups_from_midi(midi_path, offset_s, n_frames, sr, hop)` — parse notes (same source as
  `synthesizePrior.note_onsets`: pretty_midi `note.start`/`note.end`, all instrument tracks, no
  filtering), shift by `offset_s`, group by `round(start*sr/hop)`, drop frames outside
  `[0, n_frames)`. Per group `start_s`=min onset, `end_s`=max offset, `end_frame`=clip of
  `round(end_s*sr/hop)` into `[onset_frame+1, n_frames]`. **INVARIANT:** `[g.onset_frame …]` equals
  the saved `onsets_arioso` array (build asserts it).
- `expand_to_frames(groups, tech_ids, n_frames, rest_snap)` — `[n_frames]` uint8, all-`REST_ID`
  first (pre-first-onset frames stay rest). Each group fills from its onset with its technique id;
  **hybrid rest policy**: if the gap to the next onset is `<= rest_snap`, the note extends to the
  next onset (legato/rounding slack, no spurious rest sliver); else it stops at `end_frame` and the
  gap stays rest.
- `TechniqueClassifier(device="cuda", note_ckpt=NOTE_CKPT, transcriptor_ckpt=TRANSCRIPTOR_CKPT)` —
  loads both checkpoints **once** (construct one, reuse across clips); falls back to CPU if cuda
  is unavailable; raises `FileNotFoundError` if either ckpt is missing or the transcriptor fails to
  build.
- `prepare_chunks(wav_16k, spans_s, pad_s, note_seconds=TECHNIQUE_NOTE_SECONDS, peak_normalize=True)`
  → `list[np.ndarray]`: builds the **exact** per-note arrays the model consumes. Onset-anchored
  slice (`slice_waveform_by_notes`, `pad_s` context) → if `note_seconds` set, pad/truncate to
  `round(note_seconds*16000)` samples with VioPTT's policy (**short → zeros appended at the end**,
  note stays at the front; **long → first N samples**), matching `pad_truncate_sequence`
  (`utils/utilities.py:84-88`) → optional peak-normalize `x/(max|x|+1e-8)`. `note_seconds=None` →
  raw variable-length slices (legacy). All arrays contiguous float32. This is what the overlay
  notebook auditions per note.
- `classify(wav_16k, spans_s, pad_s, note_seconds=TECHNIQUE_NOTE_SECONDS, peak_normalize=True)` →
  `(ids [K] int64, conf [K] float32)`: `prepare_chunks` → `infer_batch` (uniform fixed-window
  lengths make its batch-max padding a no-op, so the model sees the training-time input); empty
  spans → two empty arrays. The fixed 2 s window is the fix for the blanket-`spiccato` mislabeling.
- Module constants: `VIOPTT_SR=16000` (VioPTT runs at 16 kHz — GT is resampled on load), `NOTE_CKPT`
  / `TRANSCRIPTOR_CKPT` (`external/VioPTT/checkpoints/*.pth`). Training-window facts cite
  `external/VioPTT/piano_transcription/utils/data_generator.py` (RWC/MOSAPT note datasets) and
  `pytorch/main_note_technique.py`.

### build_techniques.py — VioPTT technique labels over the dataset (one-time pass)
Mirrors `build_prior.py`. For each `status==ok` clip, writes `data/technique_arioso/<base>.npy`
(`[T]` uint8 per-frame technique ids, frame-aligned to `target_mel`) and
`data/technique_arioso/<base>.notes.csv` (one row per note-group: the VioPTT prediction + span).
- `process_clip(row, out_dir, dataset_root, clf, *, pad_s, note_seconds, overwrite)` —
  `note_groups_from_midi` from the clip's MIDI + manifest `offset_ms`/`n_frames`; **onsets tripwire**
  asserts the group onset frames equal `onsets_arioso` (else `AssertionError` — stale artifacts,
  rebuild `build_prior`); `load_mono(gt, sr=VIOPTT_SR)` → `clf.classify(..., note_seconds=...)` →
  `expand_to_frames` → save npy + CSV. GT path is built as `data/gt/<base>.wav` (the manifest's
  `gt_path` is backslash-formatted, not parsed). Skip-existing (both files) unless `overwrite`. CSV
  columns: `note_index, onset_frame, end_frame, start_s, end_s, n_notes, technique` (class-name
  string), `confidence` (6-dp floats).
- `build(out_dir, dataset_root, *, device, pad_s, note_seconds, note_ckpt, transcriptor_ckpt,
  limit, overwrite)` — read ok rows, construct `TechniqueClassifier` **once**, loop with per-clip
  try/except (`[i/N] status base` prints, failures logged to stderr + 1-frame traceback, continue),
  summary.
- CLI flags `--out-dir --dataset-root --limit --overwrite --device --pad-seconds --note-seconds
  --note-ckpt --transcriptor-ckpt` (device default `cuda` if available else `cpu`; `--note-seconds`
  default 2.0, `0` = legacy variable-length).
- CLI: `python -m DataSynthesizer.build_techniques --limit 2` (smoke) | `python -m DataSynthesizer.build_techniques` (full).
- **Fixed-window fix / stale labels:** each note is now classified on a fixed 2.0 s onset-anchored,
  peak-normalized window matching VioPTT's training (`clf.classify` default). Labels written
  **before** this change used VioPTT's variable-length un-normalized from-midi path and are
  **stale** — they over-report `spiccato` on fast legato notes. Re-run with `--overwrite` to
  replace them.
- **Note:** VioPTT is still a pretrained black box — it labels each note with whatever class it
  predicts; the pipeline broadcasts those predictions, it does not second-guess them. Matching the
  training input distribution just makes those predictions trustworthy.

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
"$PY" -m DataSynthesizer.build_prior --limit 4                    # Arioso prior features (smoke)
"$PY" -m DataSynthesizer.build_techniques --limit 2              # VioPTT technique labels (smoke)
"$PY" -m DataSynthesizer.build_techniques                        # full technique pass (status==ok)
"$PY" -m DataSynthesizer.onset_align prior.wav gt.wav             # report a (prior, GT) offset
```

> **Technique pass ordering:** `build_techniques` requires `build_prior`'s `onsets_arioso` to
> already exist (it asserts the two agree per clip). Run `build_prior` first; if `onsets_arioso`
> is stale, rebuild it (`--overwrite`) before the technique pass.

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
- **VioPTT technique pass** (`build_techniques`) needs two extra deps in **ai-violin**:
  `torchlibrosa==0.1.0` and `h5py` (everything else — torch, librosa, pretty_midi — is already
  present). It reuses VioPTT's vendored inference code (`external/VioPTT`, not on PyPI) and its two
  checkpoints under `external/VioPTT/checkpoints/`. VioPTT runs at **16 kHz** (GT is resampled on
  load, not the repo's 44.1 kHz), but the emitted per-frame track is on the **44.1 kHz / hop-512**
  mel grid, so it aligns with `target_mel` / `prior_mel_arioso`. The note model **requires**
  transcription features, so the transcriptor checkpoint is mandatory.
