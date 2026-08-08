# Labeler — memory palace

Local web app for labeling the **new recorded violin dataset**: drop a wav in, it is
denoised, auto-transcribed to notes by the vendored **MUSC** model
(`external/violin-transcription`, the same model that produced the etude dataset's MIDIs),
onset-aligned with the usual `common.onset_align` convention, prefilled with
onset-energy **velocities** and pitch-bend-derived **vibrato** flags — then hand-corrected
and labeled (articulation / vibrato / velocity, full note editing) in a canvas piano-roll
drawn **over the BigVGAN-contract mel**, with synchronized playback of the original audio
and a simple oscillator MIDI synth. Saves canonical per-note annotations and compiles
**verified** clips into a standard Arioso dataset root (GT wav + target/prior mel + onsets
+ per-frame conditioning arrays, `common.dataset_schema` layout) that the prior/conditioning
pipeline consumes unchanged.

Run **from the project root** (so `import common` resolves):

```bash
PY="C:/Users/archi/Miniconda3/envs/ai-violin/python.exe"
"$PY" -m Labeler.server                     # http://127.0.0.1:8765  (/ redirects to /static/)
"$PY" -m Labeler.processing <wav-or-clip-id> [--force-from STAGE] [--notes-mode MODE]   # headless
"$PY" -m Labeler.compile [--all | <clip_id> ...] [--config PATH] [--force]              # compile verified clips
"$PY" -m pytest Labeler/tests -q            # 55 tests, no GPU/network
```

Pipeline (per clip, cached per stage by params-hash):
`ingest → denoise → transcribe → align → velocity → vibrato → media → finalize`

## Dependencies

Installed into **ai-violin** on top of the training stack: `torchaudio==2.6.0+cu124`
(matches torch; MUSC imports it), `fastapi`, `uvicorn[standard]`, `noisereduce`, `gdown`,
`mir_eval`, `pytest`. MUSC checkpoint auto-downloads on first use to
`external/violin-transcription/musc/violin_model.pt` (219 MB, Drive id
`1FfUjC3usmZBoTxNT6rNVIcYw4pol4K1g`); if gdown fails, download manually to that path.

## Data layout (`Data/recorded/`, configurable via `labeler.yaml: dataset_root`)

- `raw/` — user drops recordings (`*.wav`, `*.flac`; anything soundfile reads — m4a is not).
- `clips/<clip_id>/` — derived: `source.wav` + `cleaned.wav` (44.1k mono PCM16, voiced-RMS
  −20 dBFS), `stretch/{source,cleaned}_x050|x025.wav` (pitch-preserving phase vocoder),
  `musc_raw.mid` (pre-alignment transcription), `transcription.json` (notes + amplitudes +
  bend-cents; lets velocity/vibrato re-run without GPU), `_auto.json` (per-note analysis
  accumulator for stage isolation), `onset_env.f32` (raw f32 LE onset strength, hop 512),
  `mel/meta.json + tile_NNNN.png` (magma, 2048 frames/tile, p1–p99), `notes.json`
  (CANONICAL annotations), `notes.backup/` (rolling 20 revs + named snapshots),
  `status.json` (server-owned stage state/hashes).
- compile output → `CompileParams.root` (default `Data/datasets/gt_arky`), the standard
  dataset-root layout (`manifest.json`, `gt/<id>.wav`, `target_mel/`, `prior_mel/`, `onsets/`,
  `cond/{articulation,velocity,vibrato}/<id>.npy`) — the contract + encodings are documented in
  `common/README.md` (this root sets `signals: [articulation, velocity, vibrato]`).
- `clip_id` = raw filename stem sanitized to `[A-Za-z0-9_-]`.

## notes.json schema (v1) + label preservation

Per note `{id "nNNNN" (monotonic via next_note_ordinal), start_s, end_s, pitch, velocity,
technique: str, vibrato: bool, slur_group: null (reserved), env_dct, vib_params, vib_model,
auto: {velocity, technique,
vibrato, amplitude, onset_strength, vibrato_rate_hz, vibrato_extent_cents},
human: {technique, vibrato, velocity, timing}}`.

The last three are **measured shape descriptors baked into the compile prior**, `null`
until their measurement pass runs, and are labels-derived data rather than labels:
- `env_dct: [c0, c1, c2] | null` — the note's energy envelope (`common/envelope.py`;
  level/tilt/arch in power-dB), written by `POST .../envelope`.
- `vib_params: [float] | null` + `vib_model: str | null` — the note's fitted vibrato
  oscillator (`common/vibrato_model.py`; for the current `rampboth5`,
  `[D0, gD, f0, s, phi]` = half-depth cents, cents/s, Hz, Hz/s, radians), written by
  `POST .../vibrato-fit`. Only ever set on notes with `vibrato: true`; `vib_model` is
  emitted only alongside params. **Onset-anchored**, so the front-end clears both on
  any boundary/pitch edit or when the vibrato flag is cleared — a fresh pass re-measures.
  For a human-readable vibrato rate read `auto.vibrato_rate_hz`, **not** `vib_params[2]`:
  the fitted rate is a curve-fit parameter (r = 0.089 vs the detector; pinned at a bound
  on ~30% of notes). Model tiering if it is ever swapped: `rampboth5` (5 params, median
  4.29 cents), `dramp4` (4, 4.95, better-behaved rate), `lfo3` (3, 5.57), against a 7.72
  no-model floor.

Clip level: audio/denoise + pipeline param
blocks (with hashes), vocabulary snapshot, `mute_regions [{start_s, end_s, label}]`,
`orphaned_labels`, `view {px_per_sec, scroll_s, gt_variant}` (snake_case), `rev`, and the
top-level **`verified: false`** sign-off flag (missing → `False`; additive, no schema bump —
the 59 existing clips need no migration). `verified` **gates compilation** (only verified
clips are compiled) and is set via `POST /api/clips/{id}/verified` (UI badge + toggle).

Reprocessing (`notes_mode`): **keep_labels** (default) — finalize never touches an existing
notes.json that has human edits; **retranscribe_merge** — snapshot, then base = fresh auto
notes, old human-flagged facets copied onto matches (equal pitch, |Δonset| ≤ 60 ms, greedy
nearest, once), unmatched human-touched notes → `orphaned_labels`, and **`verified` is NOT
carried** (a rebuild invalidates the sign-off, so it resets to `False`); **reset_notes** —
snapshot then rebuild. Saves are atomic (`.tmp` + `os.replace`) with optimistic `rev`
locking (409 on stale).

## API (localhost JSON; errors `{error, detail}`)

`GET /api/config` (vocabulary/keys/colors, speeds, stages) · `GET /api/clips` ·
`POST /api/clips/{id}/process {force_from?, notes_mode?}` → 202/409 ·
`GET .../status` (poll 500 ms; state/stage/pct) · `GET .../meta` (durations,
offset_applied_s, `mel.mel_bin_of_midi[128]`, `media` URL map incl. per-tile widths) ·
`GET|PUT .../notes` (+POST alias for sendBeacon; 409 rev_conflict/job_running) ·
`POST .../verified {verified}` (rev-bumping sign-off; 409 job_running) ·
`POST .../envelope` → `{status, rev, n_notes}` (fit every note's `env_dct` from
`cleaned.wav` + the CURRENT boundaries; rev-bumping; 409 job_running / 404
not_processed) ·
`POST .../vibrato-fit` → `{status, rev, model, n_notes, n_vibrato, n_fit, n_failed,
elapsed_s}` (one whole-clip pyin pass, then fit every `vibrato: true` note and
**clear** every other note's params; same guards. `n_failed` = vibrato notes the fitter
rejected as unfittable. **Slow — ~3.5× the clip duration** (a 24 s clip measured 84 s);
it is a synchronous request with no server-side timeout, so clients must not impose a
short one) ·
`POST /api/compile` + `GET /api/compile/status` (background compile of verified clips;
409 compile_running) · `/media/<clip>/…` StaticFiles (wav
Range/206). `/` **redirects to `/static/`** — index.html uses relative `./js` refs that
only resolve under the mount.

## Keyboard map (window keydown, skipped in form fields)

Click note = select + playhead to its start; click empty = playhead only; Esc deselect.
`1..9` assign technique (order from `labeler.yaml`) + auto-advance to next note.
`space` play from playhead / pause + snap playhead to selected note. `v` vibrato toggle.
`←`/`→` prev/next note. `↑`/`↓` pitch ±1 semitone, `shift+↑`/`↓` velocity ±5.
`ctrl+z`/`ctrl+y` undo/redo. `[`/`]` speed. `m`/`n` mute Original/MIDI. `f`
follow-playhead. `s` save. Editing: drag body move (semitone snap) / right edge resize,
`Del` delete, double-click add, `x` split selected note at playhead, `c` split all notes
at playhead, `g` mute-region paint on ruler. Wheel pans time; ctrl+wheel zooms (40–1000 px/s).

## Files (backend)

- **`__init__.py`** — package marker; run modules from the project root.
- **`labeler.yaml`** — user config: `dataset_root`, `port`, `editing_enabled`, articulation
  vocabulary `[{name,key,color,abbrev}]` (default normal/slur/spiccato/detache on 1–4;
  adding a class = one entry, keys auto-bind in order), `speeds`, and the
  denoise/transcribe/velocity/vibrato/compile/media parameter groups. The `transcribe` group
  carries the recall-biased decode knobs (`onset_thresh`/`frame_thresh`/`min_note_len_ms`)
  plus `batch_size`. Partial override of code defaults.
- **`config.py`** — `LabelerConfig` dataclass tree + YAML loader (`load_config`; unknown key
  = hard error listing valid keys, mirroring `Arioso/run_config.py`). Owns `STAGES`,
  `SCHEMA_VERSION`, `MODEL_FPS` (44100/256), `stage_params_hash` (drives the skip cache).
  Re-exports `SR`/`HOP` from `common.config`. `CompileParams` = `{root, gt_variant, tempo,
  program, mute_fade_ms, prior_pitch}`; `prior_pitch` (default `quantized_vibrato`) is the
  `common.prior` pitch trajectory the compile prior renders through. `_STAGE_PARAMS["compile"]`
  hashes the whole dataclass, so touching any of the six changes `compile_hash` and forces a
  full recompile.
- **`library.py`** — filesystem layout (`clip_dir`, `raw_dir`, `exports_dir`, filename
  constants), `sanitize_clip_id`, `scan_clips`/`get_clip_info` (state raw/processing/
  ready/error, duration via `soundfile.info`), `ClipNotFound`, atomic `read_json`/
  `atomic_write_json` — the primitives every stage uses.
- **`transcribe.py`** — MUSC vendoring (sys.path, mirrors `common.vocoder`'s BigVGAN
  vendoring pattern) + lazy model singleton `get_model` (cuda→cpu fallback, `GPU_LOCK`,
  torch-2.6 `weights_only` retry-shim, `CheckpointMissing` with the manual download recipe).
  `transcribe_notes(y_44k, params)` runs `model.predict` (caller holds `GPU_LOCK`), then
  **decodes the posteriors itself** — it does NOT use MUSC's `model.transcribe`/`out2note`,
  whose decoder is hardcoded to a 127.7 ms minimum note length that silently deleted most
  short notes (spiccato scale kept 10 of ~39). `decode_posteriors(out, (note_low, note_high),
  params)` is the pure numpy/scipy core (no torch/GPU/model, hence unit-testable): it calls
  `musc.postprocessing.spotify_create_notes` with the three recall-biased `TranscribeParams`
  knobs — `onset_thresh` 0.3, `frame_thresh` 0.2, `min_note_len_ms` 45 (frames =
  `round(ms/1000·sr/hop)`, fps recovered from `out["time"]`) — and everything else exactly
  as `out2note`'s 'spotify' path (`infer_onsets`, `melodia_trick`, default `energy_tol`),
  returning `(start_f, end_f, start_s, end_s, pitch, amp)` sorted by `(start_s, pitch)`.
  WHY recall-biased: the dataset MIDIs were score-aligned, so a blind re-transcription must
  over-recall and let the human trim. `transcribe_notes` then runs `model.get_pitch_bends`
  on the frame-index tuples BEFORE the second-mapping (as `out2note` does; order-preserving,
  1:1 at the default `timing_refinement_range=0`), yielding sorted plain-Python events with
  bends → cents (×100/4096). `batch_size` lives in `TranscribeParams` too. Torch-free at import.
- **`denoise.py`** — `highpass` (4th-order Butterworth 120 Hz, `sosfiltfilt` zero-phase) →
  `denoise` (noisereduce non-stationary spectral gating, prop_decrease 0.85, 2 s
  time-constant) → `voiced_rms_normalize` to −20 dBFS (matches the dataset convention).
  Removes buzz/hiss; discrete page-turn thumps are handled by mute regions instead.
- **`align.py`** — `estimate_midi_offset`: render quantized prior from `musc_raw.mid`
  (`common.prior.quantized_prior`) → `estimate_offset_seconds` vs the
  cleaned audio (positive ⇒ MIDI lags) → `apply_offset` shifts note times by −offset
  (clamp ≥ 0); `offset_applied_s = -offset` recorded. ≈ 0 for self-transcribed clips by
  construction; matters for imported MIDI.
- **`velocity.py`** — `onset_env` (librosa onset_strength, hop 512), per-note max over
  frames `[f−2, f+4)`, per-clip percentile 10→95 normalization, `vel = 24 + x^0.6·96`
  clamp 1–127 (degenerate clip → 80). Raw strength kept in `auto.onset_strength`.
- **`vibrato.py`** — `detect_vibrato(bend_cents, dur_s, fps≈172.27)`: skip < 0.3 s →
  central 80 % → linear detrend → windowed spectrum; vibrato iff extent (p95−p5) ≥ 12 c
  AND 4–9 Hz band-energy fraction ≥ 0.4 AND peak rate ∈ [4, 9] Hz. (Band-energy fraction
  replaced the planned per-frame sustain gate — robust on short real notes.) Thresholds
  tunable in `labeler.yaml`; conservative on fast passages by design.
- **`notes_store.py`** — schema v1 builders (`build_document`/`build_note`), per-clip
  locks, `put_with_rev` (409 stale), rolling + named-snapshot backups,
  `merge_retranscribe`, `commit_rebuild`/`save_new`. Two measurement-pass writers with the
  same lock / `_roll_backup` / rev-bump discipline (so the editor's next full-doc PUT does
  not 409): `set_envelopes(cfg, clip_id, env_by_id)` and
  `set_vibrato_params(cfg, clip_id, params_by_id, model)`. They differ deliberately on the
  `None` case — `set_envelopes` ignores it, while for `set_vibrato_params` an id present
  with `None` **clears** that note's `vib_params`/`vib_model` (the fit pass sweeps every
  note, so a note that lost its vibrato flag must lose params it can no longer justify).
  In both, an **absent** id leaves that note untouched.
- **`midi_io.py`** — `notes_to_pretty_midi`/`write_midi`: tempo 120, single Instrument
  program 40 ("violin"), int velocities, **no pitch bends**, resolution 480 (so
  `note_groups_from_midi` onset frames survive the write/read round-trip);
  `validate_for_export` (zero/neg duration = error, same-pitch overlap = warning).
  This is the repo's only MIDI **writer**; consumers (`common.prior` `render`/`note_onsets`,
  `common.dataset_schema.note_groups_from_midi`) read it identically to the etude MIDIs.
  Two opt-in flags attach **in-memory-only** per-note attributes the prior render loop
  reads (neither survives `.write()`, so disk MIDIs stay byte-identical): `with_env=True`
  → `note.env_dct` (measured, else the velocity-derived flat env) and `with_vibrato=True`
  → `note.vib_params`/`note.vib_model`, attached **only** when the note is flagged
  `vibrato`, carries params, and its `vib_model` matches the current
  `common.vibrato_model.VIB_MODEL` (imported lazily inside the call). No fallback for an
  unfitted vibrato note — it renders flat.
- **`media.py`** — viewer derivatives (matplotlib Agg, lazy torch): `mel_tiles`
  (`common.vocoder.mel_spectrogram` → p1–p99 → magma PNGs + `meta.json` incl.
  `mel_bin_of_midi[128]` from `librosa.mel_frequencies` interpolation), `write_onset_env`,
  `make_stretches` (0.5× n_fft 2048, 0.25× n_fft 4096; eager so `/media` always has files).
- **`processing.py`** — the 8-stage `Pipeline` with per-stage params-hash skip cache
  (+ cascade + `--force-from`), the `_auto.json` intermediate, `JobManager` (background
  thread, per-clip lock, live `status.json`), `process_cli`.
- **`compile.py`** — `compile_all`/`compile_one`: the gt_arky → standard-root producer.
  Per **verified** clip: validate every note articulation ∈ `common.dataset_schema.ARTICULATIONS`,
  drop notes whose midpoint is inside a mute region, cosine-zero mutes (5 ms) + voiced-RMS
  normalize the chosen wav → `gt/`, mel → `target_mel/`, prior (surviving notes →
  `notes_to_pretty_midi(with_env=True, with_vibrato=True)` →
  `common.prior.quantized_prior(pitch=cp.prior_pitch).render`, so each note's measured
  envelope **and** vibrato are baked in) → `prior_mel/`, `onsets/` + `offsets/`
  + `cond/{articulation,vibrato}/` via the schema rasterizers (the rasterizers see
  only the `vibrato` bool — `vib_params` affects the prior audio, never the cond tracks;
  per-note dynamics ride in the prior via `env_dct`, so there is no velocity cond track).
  Incremental/idempotent
  on a `{rev, notes_hash, compile_hash}` provenance triple; a full `--all` run prunes clips no
  longer verified (manifest + artifacts). `CompileManager` runs it on a background thread for
  the API. `notes_hash` = sha1 over a canonical dump of the notes (`start_s/end_s/pitch/velocity/
  technique/vibrato/env_dct/vib_params/vib_model`, times rounded 1e-6 and coefficients 1e-4)
  + mute regions (`start_s/end_s`) — so re-running either measurement pass invalidates the
  affected clips and nothing else.
  - **Auto split update.** Every run (full or partial) ends in `update_root_split(root)` →
    `Arioso.splits.update_split`, which additively reconciles `<root>/split.json` with the new
    manifest (new clips assigned by piece, vanished ones pruned, existing train/val assignments
    never moved), prints one `[compile] split +N train, +N val, pruned N (val pieces K/N)` line
    and returns its summary under the run summary's `"split"` key. Without it a clip compiled
    after the split was written sits in neither list and is invisible to training. It is wrapped
    in a try/except (the artifacts are already on disk — a split failure must not fail the
    compile) and imports `Arioso.splits` **lazily inside the function**, so `Labeler.server`'s
    startup import graph never reaches into `Arioso`. The progress/status schema is unchanged.
- **`server.py`** — `create_app`: cfg + JobManager + CompileManager on `app.state`; mounts `/api`,
  `/media` (Range for free), `/static` (html=True); `/` → RedirectResponse(`/static/`);
  Windows `.js/.mjs/.css` MIME fix at startup. `main`: uvicorn 127.0.0.1:8765, single
  process, **no reload** (MUSC singleton).
- **`api.py`** — thin routes onto library/processing/notes_store/compile; documented error
  codes; reads cfg/jobs/compiler from `request.app.state`. The two measurement passes
  (`compute_envelopes`, `fit_vibrato`) are plain sync handlers sharing one guard set
  (404 clip_not_found / no_notes / not_processed, 409 job_running) — they run in the request
  thread rather than through `JobManager` because they are read-then-write-notes, not
  pipeline stages.
- **`tests/`** — 55 pytest cases, GPU/network-free: velocity mapping, vibrato synthetic FM
  (true/flat/drift/short), notes_store rev-conflict + merge semantics, MIDI round-trip
  through `note_groups_from_midi`, align sign convention on synthetic click tracks,
  `test_vibrato_fit` (synthetic rampboth5 round-trip incl. NaN/octave-spike robustness and
  baseline-absorption, the three rejection paths, `vibrato_ratio`/unknown-model contracts,
  `set_vibrato_params` set/clear/absent + backup, and `notes_hash` sensitivity), and
  `test_transcribe_decode` — `decode_posteriors` on a synthetic 60 ms + 200 ms posterior
  grid: both survive `min_note_len_ms` 45, only the long one survives 127.7, onset order +
  frame→second mapping exact (no torch/model needed, `spotify_create_notes` is numpy/scipy).

## Files (frontend, `static/` — vanilla ES modules, no build step)

- **`index.html` / `css/app.css`** — DOM skeleton (transport, sidebar, roll grid, velocity/
  onset lanes, minimap, region popover, orphan panel, rev-conflict modal) + dark theme.
  Header holds the three clip-level action buttons, all hidden until a doc is loaded:
  `#btn-envelope` (♫ Envelopes), `#btn-vibrato` (∯ Vibrato) and `#btn-verified`.
- **`js/state.js`** — single store + command-pattern undo/redo (cap 500; captures
  selection+playhead so undoing a label-key press steps back), sorted-notes cache
  (`invalidateSorted` MUST be called after any direct `store.doc` assignment — the load
  paths do), id allocation seeded from `next_note_ordinal`, dirty→autosave hook, helpers.
- **`js/timeline.js`** — geometry only: time↔px, bin↔y (origin lower), `binForPitch` via
  server `mel_bin_of_midi`, note/velocity rects, pan/zoom clamp, binary-search culling
  (`visibleNotes`).
- **`js/api.js`** — one function per endpoint (incl. `envelopePass`/`vibratoPass`);
  `?mock=1` diverts to `mock.js`; resolves
  `meta.media` URLs; decodes audio + onset env; `beaconNotes` for `beforeunload`. `j()` is a
  plain `fetch` with **no client timeout** — required, since `vibratoPass` can run minutes.
- **`js/render.js`** — all drawing, DPR-correct: mel tiles (source-crop, LRU 30), open-string
  gridlines, mute hatching, notes (technique color, vibrato squiggle, human tick, selection),
  velocity/onset lanes, minimap, rAF-only overlay (playhead/hover/ghosts). The vibrato
  squiggle is **scaled from `vib_params`** when present (`VIB_PARAM_IX` maps model → depth/rate
  slots; spatial frequency converted through the note's px/s so the wiggle count matches the
  real cycle count, amplitude from the half-depth clamped to the note band) — decorative
  only, no ramps or phase; notes without params keep the legacy fixed squiggle.
- **`js/interact.js`** — wheel pan / ctrl+wheel zoom; select/scrub; drags: move+re-pitch
  (semitone snap), edge resize (5 px zones), velocity bars, ruler region-paint, dbl-click
  create, minimap jump. Every mutation is an undoable command.
- **`js/keymap.js`** — pure keydown→action dispatch (form-field-guarded, injected actions).
- **`js/player.js` / `js/synth.js`** — one 44.1 kHz AudioContext; lazy `[variant][speed]`
  buffers (prestretched files played at `offset = playhead/s`); shared clock
  `clipTime = t0 + (now−anchor)·s` drives playhead AND the 25 ms/120 ms lookahead
  scheduler; voice = triangle + saw(−8 dB) → lowpass(4·f0 ≤ 12 kHz) → 8 ms attack /
  60 ms exp release, peak `0.25·(vel/127)^1.5`, pool 8; speed/variant change pauses first.
- **`js/editing.js`** — command factories with exact inverses (fields/technique/vibrato/
  velocity, move/resize/add/delete/split, mute-region add/delete). Because `vib_params` are
  onset-anchored, `setVibrato(id, false)`, `moveNote`, `resizeNote` and `splitNote` all
  **clear** `vib_params`/`vib_model` (snapshotting them first, so undo restores them); the
  split's right half is created without params.
- **`js/main.js`** — bootstrap + orchestration: config/clips load, clip loading
  (meta → notes → onset; `invalidateSorted` after doc assignment), keymap actions, rAF
  loop + follow-playhead, autosave (1.5 s debounce, retry backoff, sendBeacon), rev-conflict
  modal, 500 ms processing poll, sidebar/inspector/transport wiring. `runServerPass(btnId,
  apiFn, {idle, busy, done})` is the shared driver behind both measurement buttons —
  disable → POST → re-GET the doc (the server bumped `rev`, so a stale full-doc PUT would
  409) → `invalidateSorted`/`initIdCounter`/`clearHistory` → repaint → flash the result on
  the label, with 409 `job_running` flashed rather than raised.
- **`js/mock.js`** — dev backend (`?mock=1`) mirroring the real API shapes (incl.
  `envelopePass` and `vibratoPass`, which stamps fixed `rampboth5` params on vibrato notes
  and nulls the rest) + the `?selftest=1` in-page test routine (result in `document.title`).

## Windows gotchas

- Registry maps `.js` → text/plain; `server.py` patches mimetypes at startup (ES modules
  hard-fail otherwise).
- No `--reload`: uvicorn reload uses multiprocessing and would re-load the 219 MB MUSC
  singleton per worker.
- PowerShell: `curl` is an alias — use `curl.exe`; `$null` as a native-command arg becomes
  an empty string.
- If the torch/numba OpenMP clash ever bites, set `KMP_DUPLICATE_LIB_OK=TRUE` for the
  server process only.

## Limitations / deferred

- Vibrato prefill is conservative (< 0.3 s notes never flagged); tune
  `labeler.yaml: vibrato`. Compile hard-errors on any note articulation outside the unified
  `common.dataset_schema.ARTICULATIONS` vocab, and on a config articulation name not in it.
- Transcription is deliberately recall-biased (`labeler.yaml: transcribe`, `min_note_len_ms`
  45 vs MUSC's 127.7): it over-segments (extra short/spurious notes) so the human trims
  rather than re-adds. Lowering thresholds further raises the false-positive cleanup cost;
  raising `min_note_len_ms` back toward 127.7 drops genuine short (spiccato) notes.
- `--force-from` is linear (re-runs media even for notes-only changes). Eager stretches
  slow first-process of very long takes; recommend ≤ ~10 min per take.
- Follow-ups (not built): bulk-assign technique on selection, auto-play-on-select,
  loop-selected-note, long-take splitter.
