# Studio — memory palace

FL Studio–style piano-roll web app for **Arioso violin inference**: compose violin
phrases on a piano roll, attach per-note conditioning (articulation, velocity/dynamics,
pitch-bend, vibrato), and render audio offline through the Arioso model (MIDI → saw prior
→ CFM ODE → BigVGAN). Architecturally a sibling of `Labeler/`: FastAPI app factory +
`JobManager` background threads + atomic rev-versioned JSON document store + a vanilla-JS
zero-build canvas frontend. Note timing is stored in **beats** (quarter notes) so editing
is tempo-invariant; seconds exist only at the audio/MIDI boundary.

## Quickstart

Run **from the project root** (so `import common` / `import Arioso` resolve):

```bash
PY="C:/Users/archi/miniconda3/envs/ai-violin/python.exe"
KMP_DUPLICATE_LIB_OK=TRUE "$PY" -m Studio.server   # → http://127.0.0.1:8766  (/ redirects to /static/)
"$PY" -m pytest Studio/tests -q                    # torch-free unit + API tests (106 green)
```

- **URL**: open <http://127.0.0.1:8766/> — it redirects to `/static/`, the editor.
- **Mock mode** (no backend/torch): <http://127.0.0.1:8766/static/?mock=1> (or open
  `index.html` over `file://`) runs the whole editor against an in-browser fake backend
  (`js/mock.js`, localStorage) — seeded demo phrase, simulated render with a synthesized
  waveform/peaks. Ideal for frontend work.
- **Vocoder weights**: the default vocoder is the local violin fine-tune
  (`Vocoder/models/ft_v2`, via `common.config.VOCODER_DIR`), so a normal first render loads
  from disk. Picking **`stock (HF)`** in the DEV drawer downloads the pretrained **BigVGAN-v2**
  weights from HuggingFace (`nvidia/bigvgan_v2_44khz_128band_512x`) — a one-time network cost
  surfaced as the `vocoder` job stage; later renders use the HF cache.
  `KMP_DUPLICATE_LIB_OK=TRUE` is needed for the torch/numba OpenMP clash on this machine.

## Status — all phases DONE (Phases 0–7)

A phased build (see the plan), now complete and integration-verified end to end (compose →
technique/velocity/pan/bend/vibrato → real GPU render on `7-9-adr/checkpoint_final.pt` →
fetch `mix.wav`/`mix.peaks` over `/media` → export WAV/MIDI → import `.mid`). **Phase 0 =
backend skeleton** (config, library, project store, timing, `JobManager`, API). **Phase 3 =
render backend** (`voices`/`bend`/`midi_export`/`peaks`/`model_registry`/`render`, rendering the
prior via `common.prior.quantized_prior(pitch=)`). **Phase 5 =
segment caching** (`cache.py` + segment-aware `render.py`): the document is split into
silence-delimited segments, each rendered once to `render/cache/seg_<hash>.wav` and reused
on later renders whose content did not change, then stitched into `mix.wav` (editing one
note in a two-phrase project → re-render only its segment, ~0.4 s, other segment a cache
HIT). **Phase 6 = MIDI import + WAV/MIDI export**. **Phases 1/2/4/7 = the frontend** (piano
roll, conditioning UI, render UX, polish) — see *Frontend files* below.

**The server/API import graph is torch-free** and must stay that way — torch/model/vocoder
are lazy singletons pulled in only inside `render._render_segments` / `model_registry`
function bodies (`cache.py`, `midi_*`, `export.py` are numpy/pretty_midi/soundfile, all
torch-free).

## project.json schema (v1)

Per project: `{schema_version, project_id, rev (monotonic, optimistic-lock), next_note_ordinal,
name, bpm (140), time_sig {num,den} (4/4), ppq (480), render {...}, view {...}, notes [...]}`.
Per note: `{id "nNNNN" (monotonic, never reused), start_beat, len_beats, pitch, velocity (100),
technique (normal|slur|spiccato|detache — conditions renders on conditioned checkpoints),
pan (0.0, bipolar −1..+1 — project-only), bend [] (piecewise-linear {beat, semitones} control points
relative to note start), vibrato {depth_semitones 0, rate_hz 5.5, onset_beats 0.15}}`. Every
schema producer agrees on this field set — `project_store.build_note` (backend canonical),
`editing.makeNote` (frontend), `mock.js` seeds, `midi_import`, and `cache.segment_hash` all
include `pan`. `render {model, checkpoint, prior_mode, wav, duration_s, segments[]}`
tracks the last render + its cache segments; `view {px_per_beat 48, scroll_beat 0, top_pitch 96,
lane "velocity", snap "step"}` is the persisted camera/editor state. Saves are atomic
(`.tmp` + `os.replace`) with rev locking (409 on stale) and rolling 20-rev backups in
`project.backup/`.

**Articulation conditions renders on conditioned checkpoints**: with an unconditioned
checkpoint (`conditioning: []`) it only colors the note in the UI, but when the loaded
checkpoint carries per-frame conditioning, each note's technique is mapped through
`technique_model_vocab` (identity: `normal→normal, slur→slur, spiccato→spiccato,
detache→detache`, config-overridable) onto the model's articulation signal, and per-note
velocity / vibrato-depth>0 / note-boundary tracks are rasterized alongside it
(`render._segment_note_events` → `Arioso.infer.build_cond`).

## Data layout (`Data/studio_projects/`, configurable via `studio.yaml: projects_root`)

The projects root is the directory `/media` is served over, so a project's rendered wav at
`<projects_root>/<id>/render/mix.wav` is reachable at `/media/<id>/render/mix.wav`.

- `<project_id>/project.json` — canonical document (CANONICAL; the source of truth).
- `<project_id>/render/` — `mix.wav` (stitched render output), `mix.peaks` (`SPK1` waveform
  peaks), `meta.json` (render bookkeeping incl. the segment manifest; read by
  `/api/.../render/meta`), `status.json` (server-owned render-job state; read by
  `/api/.../render/status`), and `cache/seg_<hash>.wav` (per-segment render cache — the
  reuse unit; GC'd to a rolling 64-file budget, current manifest always kept).
- `<project_id>/exports/` — timestamped WAV / MIDI exports (`<name>-<YYYYmmdd-HHMMSS>.{wav,mid}`),
  served over `/media/<id>/exports/…`.
- `<project_id>/project.backup/` — rolling `rev_*.json` backups (last 20).
- `project_id` = the project name sanitized to `[A-Za-z0-9_-]` (≤ 80 chars), with a
  `-2`/`-3` suffix on collision.

## API (localhost JSON; errors `{error, detail}`)

`GET /api/config` (sr/hop/frame_rate, articulations + colors + keys + model-vocab map, snap
options, prior modes, default model/checkpoint/**vocoder**/prior_mode) · `GET /api/models`
(filesystem scan of `models_root/<run>/*.pt` → `[{run, checkpoints, default}]`, `7-9-adr`
flagged default, plus `vocoders: [{name, default}]` — the scan of `vocoders_root/<name>/`
dirs holding `config.json` + `bigvgan_generator.pt`, with the dir-less `{"name":"hf"}` stock
baseline appended — and `default_model`/`default_checkpoint`/`default_vocoder`;
**no torch**) · `GET /api/projects` (summaries) · `POST /api/projects {name}` → **201** (fresh
doc) · `GET /api/projects/{id}` (404 unknown) · `PUT|POST /api/projects/{id}` (full doc; POST
is the sendBeacon alias; **409 `rev_conflict`** carrying `server_rev` on stale rev) ·
`POST /api/projects/{id}/render` `{scope "phrase"|"selection", note_ids?, model?, checkpoint?,
prior_mode?, vocoder?}` → **202 `accepted`** (job on a `JobManager` thread, echoing the
resolved `model`/`checkpoint`/`prior_mode`/`vocoder`) / **409 `job_running`** /
**400** (bad scope/prior_mode, `model_not_found` on a missing checkpoint,
**`vocoder_not_found`** on a name that is neither `"hf"` nor a loadable dir under
`vocoders_root` — names containing `/`, `\` or `..` are rejected outright, so a request can
never load weights outside that root) ·
`GET /api/projects/{id}/render/status` (live job status or `{state:"idle"}`; while running
also carries `segments_done`/`segments_total`) ·
`GET /api/projects/{id}/render/meta` → `render/meta.json`
(`{wav, peaks, duration_s, model, checkpoint, prior_mode, vocoder, scope, note_ids, device,
segments[{hash, start_beat, end_beat, start_s, end_s, cached}], segments_total,
segments_rendered, segments_cached, segments_touched, warnings}`) or **404 `not_rendered`**. `/media/<id>/…` is a StaticFiles mount (wav Range/206 for free), so
the render's `mix.wav`/`mix.peaks` are reachable at `/media/<id>/render/…`. `/` **redirects to
`/static/`**.

`POST /api/projects/{id}/import-midi` — the `.mid` is the **raw request body**
(`application/octet-stream`; no multipart, python-multipart is not installed). Query params
`mode` (`replace`|`merge`, default replace), `snap` (a snap id, optional), `adopt_bpm`
(truthy → set project BPM to the file's initial tempo). Notes are timed in beats through the
file's tempo map, stamped with fresh monotonic ids, committed with a rev bump → `{rev,
notes_added, file_bpm, warnings, doc}` (the full updated doc). **400 `bad_midi`** (unparseable/
empty body) / **400 `bad_request`** (bad mode/snap). ·
`POST /api/projects/{id}/export` `{kind:"wav"|"midi"}` → **200** `{kind, path (/media URL),
warnings}`. MIDI always succeeds (bend-aware, `prior_mode="bend"`). WAV copies the *current*
render's `mix.wav` and returns **409 `render_stale`** when no fresh render matches the doc
(replanning the doc with the render's own params must reproduce the manifest's segment hashes
and every cache WAV + `mix.wav` must exist). **400 `bad_request`** on an unknown kind.

**Render meta lives in `render/meta.json`, NOT `project.json`.** The frontend autosaves
`project.json` under optimistic rev locking; writing render results back into it from the
render thread would race those saves and bump the rev. Meta is a separate server-owned file
(like `status.json`), read via `GET .../render/meta`.

## Files (backend)

- **`__init__.py`** — package marker; run modules from the project root. Notes the torch-free
  import contract.
- **`studio.yaml`** — user config: `projects_root`, `port` (8766, distinct from Labeler's 8765),
  `models_root`, `vocoders_root` (`Vocoder/models`),
  `default_model`/`default_checkpoint`/`default_vocoder`/`default_prior_mode`, the articulation
  vocabulary `[{name,key,color,abbrev,model_vocab}]` (normal/slur/spiccato/detache on keys 1–4,
  normal = FL green `#6cc04a`), and the `cache` group (`gap_s` 0.35, `pad_s` 0.1). Partial
  override of code defaults; unknown key = hard error. The model/vocoder default keys ship
  **commented out** — they are derived from `common/config.py` (see below), so uncomment only to
  pin Studio to something other than the project default.
- **`config.py`** — `StudioConfig` dataclass tree + `load_config` YAML loader (unknown key =
  hard error listing valid keys, mirroring `Labeler/config.py`). Owns `SCHEMA_VERSION`,
  `FRAME_RATE` (SR/HOP ≈ 86.13), `PRIOR_MODES` (`bend`/`quantized`), `SNAP_OPTIONS`, the
  `Articulation`/`CacheParams` dataclasses, and derived helpers (`vocabulary_snapshot`,
  `technique_model_vocab` stub table, `snap_options`, `default_technique`). Re-exports `SR`/`HOP`
  from `common.config` — and **derives its model/vocoder defaults from there** rather than
  hardcoding them: `default_model`/`default_checkpoint` are the run-dir and file basenames of
  `common.config.ACOUSTIC_CKPT`, `default_vocoder` is `basename(VOCODER_DIR)` (`"hf"` when that
  is `None`). Promoting a checkpoint project-wide is therefore a one-line edit in
  `common/config.py`. **Torch-free.**
- **`library.py`** — lowest-level filesystem module: per-project path helpers (`project_dir`,
  `project_file`, `render_dir`, `render_status_file`, `exports_dir`, `backup_dir`), filename
  constants (`PROJECT_JSON`, `RENDER_DIR`, `MIX_WAV`, …), `sanitize_project_id` /
  `unique_project_id` (collision `-N` suffix), `scan_project_ids` / `project_exists` /
  `require_project` (`ProjectNotFound`), atomic `read_json` / `atomic_write_json` (`.tmp` +
  `os.replace`), **`scan_models`** (pure filesystem `<run>/*.pt` scan for `/api/models`) and the
  matching vocoder trio: `vocoders_root(cfg)`, **`scan_vocoders`** (dirs under it holding *both*
  `config.json` and `bigvgan_generator.pt`, so the UI never offers a vocoder that would fail at
  load; the dir-less `"hf"` sentinel is appended by the API, not the scan) and
  `vocoder_checkpoint_dir(cfg, name)` mapping a Studio vocoder name onto
  `common.vocoder.load_vocoder`'s `checkpoint_dir` (`"hf"` passes straight through).
- **`timing.py`** — pure beats↔seconds authority + bar/beat + snap math, **no torch**. One
  beat = one quarter note (matches quarter-note BPM and `ppq`). `beats_to_seconds` /
  `seconds_to_beats`; `quarters_per_beat` (`4/den`) / `quarters_per_bar` (`num·4/den`);
  `bar_number` / `beat_within_bar` / `bar_beat_tick`; `snap_grid` (id → grid size in
  quarter-beats: bar/beat/step = 1/16-note/finer-fractions/none) + `snap_beat` / `snap_to`.
- **`project_store.py`** — schema-v1 builders (`note_id`, `build_note` with defaults velocity
  100 / technique normal / **pan 0.0 (clamped ±1)** / bend `[]` / vibrato `{0, 5.5, 0.15}`,
  `build_document` with bpm 140 /
  4/4 / ppq 480 / default `render` + `view` blocks), per-project locks, `create_project` /
  `save_new` (rev → 1), `put_with_rev` (optimistic rev, 409 conflict → `server_rev`),
  `commit_rebuild`, rolling backups (`_roll_backup`, keep 20), and `summary`. **Torch-free.**
- **`jobs.py`** — generic **`JobManager`** copied from `Labeler/processing.py`: a daemon thread
  per project, per-project one-job guard (`submit` returns `False` when busy), live status dict
  mirrored to `render/status.json` via a `progress(status)` callback, exceptions recorded as
  `state=error` (truncated traceback). `work` is any `Callable[[progress], None]` — the concrete
  render pipeline is injected in Phase 3. `idle_status()` is the never-run default. **Torch-free.**
- **`server.py`** — `create_app(cfg)`: `_fix_windows_mimes()` (`.js`/`.mjs`/`.css`), cfg +
  `JobManager` on `app.state`, mounts `/api`, `/media` (StaticFiles over `projects_root`, Range
  for free), and `/static` (`Studio/static`, `html=True`, **only if the dir exists** — a separate
  agent builds it). `/` → `RedirectResponse("/static/")`, else a placeholder that points at
  `/api/config`. `main()`: uvicorn 127.0.0.1:8766, single process, **no reload**. **Torch-free.**
- **`api.py`** — thin `APIRouter` with uniform `_err(status, code, detail, **extra)`; reads
  `cfg`/`jobs` from `request.app.state`. Endpoints per the API section above; the render POST
  validates scope/prior_mode/checkpoint and the requested vocoder (name traversal rejected, then
  the dir must hold `config.json` + `bigvgan_generator.pt`) then `jobs.submit`s
  `render.run_render` (`from .render
  import run_render` is done **inside** the handler so the api import stays torch-free);
  render/status reads `JobManager`, render/meta reads `render/meta.json`. **Torch-free.**

### Files (Phase 3 — render backend)

- **`voices.py`** — `partition_voices(notes, max_voices=8)`: greedy interval-graph coloring
  splitting overlapping notes into monophonic **voices** (each → its own instrument with an
  unambiguous pitch-wheel), sorted `(start, end, pitch, id)` for determinism, `_EPS`-tolerant
  back-to-back reuse. Dense chords past the cap raise `VoiceOverflow` naming the offending +
  sounding notes. **Torch-free.**
- **`bend.py`** — per-note expression → 14-bit pitch-wheel events. `has_expression(note)` gates
  flat notes (no events). `pitch_bend_events(note, start_s, dur_s, bpm, next_start_s=None)`:
  piecewise-linear `bend` control points (beats rel note start) + ramped vibrato
  (`depth·sin(2π rate t)`, linear fade-in over ~2 cycles from `onset_beats`), sampled at
  **200 Hz**, clamped to ±2 st (`PB_RANGE_SEMITONES`, imported from `common.prior`), converted to wheel ints
  (`semis/range·8192`, clamp ±8191), plus a `PitchBend(0)` **reset** in the trailing gap
  (≤50 ms past the end / gap-midpoint; suppressed when legato). **Torch-free** (numpy).
- **`midi_export.py`** — `project_to_pretty_midi(doc, note_ids=None, prior_mode="bend")`:
  select notes → `validate_notes` (zero/neg length = error, same-pitch overlap = warning) →
  `partition_voices` → one `Instrument(program=40, name="violin_vK")` per voice, beats→seconds
  via `timing` at `doc.bpm`, velocity clamp 1..127, min-length guard, per-voice bends only in
  `prior_mode=="bend"`. `write_midi(doc, path, …) → (path, warnings)`. `PrettyMIDI(initial_tempo=
  bpm, resolution=ppq)`. **Torch-free.**
- **`peaks.py`** — `write_peaks(wav_f32, path, bins_per_sec=100)`: `SPK1` binary — magic +
  `uint32 sr/bins_per_sec/n_bins` header + interleaved int8 min/max pairs per bin — for the
  frontend waveform lane; `compute_peaks` / `peaks_meta` (header parse) helpers. **Torch-free.**
- **`model_registry.py`** — lazy torch. `get_model(models_root, run, checkpoint, device=None)`:
  double-checked-locking singleton keyed `(run, checkpoint, device)` → `LoadedModel(model, cfg,
  device)` loaded the `Arioso.infer.main` way (`torch.load` map_location → `cfg_from_dict` →
  `AriosoModel` → `load_state_dict(ema||model)` → eval); `ModelNotFound` on a missing ckpt.
  `get_vocoder(device=None, checkpoint_dir=None)`: BigVGAN singleton
  (`common.vocoder.load_vocoder`, HF download first call, native-GPU fallback) keyed
  `(checkpoint_dir, device)` — **not per-device only** — so a project rendering through a
  fine-tuned generator and one on the stock baseline can both stay resident.
  `list_models` = `library.scan_models` pass-through.
- **`render.py`** — `run_render(cfg, doc, project_dir, *, scope, note_ids, model, checkpoint,
  prior_mode, vocoder, device, progress)`: **segment-aware**. `plan_render` (torch-free) segments +
  hashes the doc and detects cache hits, then only the non-cached segments run the five-stage
  pipeline (`_render_segments`, module-level so it can be monkeypatched out in tests):
  per-segment `write_midi` (notes time-shifted so the padded lead = t 0) →
  `common.prior.quantized_prior(pitch=bend|quantized)` +
  `common.vocoder.mel_frames(synth.render(total_samples=span))` →
  `generate_mel(cond_frames=None)` → `vocode` → `write_pcm16(cache/seg_<hash>.wav)`.
  `model`/`checkpoint`/`prior_mode`/`vocoder` all default to the `cfg` defaults; `vocoder` is a
  **name** (a dir under `cfg.vocoders_root`, or `"hf"`), resolved to the loader's
  `checkpoint_dir` via `library.vocoder_checkpoint_dir` in `_render_segments` and folded into
  every segment hash — so swapping it re-renders the whole project. Then
  `cache.stitch` places every segment (cached + fresh) into `mix.wav`, `write_peaks(mix.peaks)`,
  `prune_cache`, and writes `render/meta.json` with the segment manifest. Module-level
  `GPU_LOCK` serializes the two GPU forwards across projects; all torch imports are **inside**
  `_render_segments` (torch-free at module load). Progress carries `segments_done`/`_total`.
  Both `phrase` and `selection` scope render exactly the non-cached segments (content hashing
  makes an edited note's segment miss automatically, so the mix is always complete); scope
  only affects the reported `segments_touched` count. Empty project → a short silence.
  Note: `common.dataset_schema.NoteEvent` gained optional `vib_params`/`vib_model` fields
  (Labeler's measured per-note vibrato, `common/vibrato_model.py`) — Studio is **unaffected**.
  It builds its `NoteEvent`s without them, and its `PRIOR_MODES` are only `bend`/`quantized`,
  never the new `quantized_vibrato`; Studio expresses vibrato through its own bend curves.
- **`cache.py`** — segment cache logic (numpy + soundfile, **torch-free**).
  `segment_notes(notes, bpm, gap_s, pad_s)`: maximal runs split on silence gaps `>= gap_s`,
  each padded into the adjacent silence (internal boundaries split the gap in half so pads
  never cross; first lead clamped to `[0, start_s]`; last tail = full `pad_s`).
  `segment_hash(...)`: SHA1 over canonical JSON of the notes' content
  (`pitch/start_beat/len_beats/velocity/technique/bend/vibrato/pan`, plus `env` — the
  `[c0,c1,c2]` energy envelope — **conditionally**, only for notes that carry an `env_dct`, so
  env-less projects keep their historical hashes; beats made
  **segment-relative** so a whole phrase can move in time and still hit cache) + params
  (`bpm/time_sig/model/checkpoint/prior_mode/schema_version/gap_s/pad_s`) + a **conditional
  `vocoder`** key. The vocoder identity is a precomputed cache id (`_vocoder_cache_id`):
  `"<dirname>:<generator file size>"` for a local fine-tune (the size catches a re-export under
  the same name), and **omitted entirely** for the stock `"hf"` baseline — same trick as `env`,
  so stock-vocoder hashes stay byte-identical to the pre-selection ones and selecting `hf` again
  revives those caches, while switching to a fine-tuned generator intentionally invalidates
  everything. `plan_render(...)` takes the vocoder **name** (default `cfg.default_vocoder`),
  resolves it to that id once, and returns →
  `{segments[{hash, wav, cached, start_s, end_s, pad_lead, pad_tail, start_beat, end_beat,
  notes}], cache_dir, total_duration_s}`. `stitch(segments, cache_dir, total_duration_s, sr)`:
  places each cache WAV at `start_s - pad_lead`, linear crossfade on any pad overlap, zeros in
  gaps, then **one peak-safety clamp** on the mix (scale down if it would clip, never boost —
  no per-segment or per-mix RMS renormalization, so re-render loudness is stable; residual
  per-segment level drift accepted for v1). `prune_cache(cache_dir, keep_hashes, max_files=64)`
  GC. `seg_wav_name(hash)` = `seg_<hash>.wav`.
### Files (Phase 6 — MIDI import + WAV/MIDI export)

- **`midi_import.py`** — the inverse of `midi_export`, **torch-free**. `import_midi(src, snap=
  None, adopt_bpm=False, time_sig=None) → {notes, file_bpm, warnings}`. `src` is raw MIDI
  `bytes`, a file-like, or a path (parse via `pretty_midi`; `BadMidi` on failure). Seconds →
  beats through the file's **tempo map** (`pm.time_to_tick(s) / pm.resolution`), so mid-file
  tempo changes are honoured exactly. Merges every non-drum instrument's notes (drums skipped
  with a warning), velocity carried over, technique `normal` / bend `[]` / default vibrato.
  Optional `snap` (a snap id — `none/line/1/2step/1/3step/1/4step/1/6step/step/beat/bar`, the
  **same ids `config.SNAP_OPTIONS` / `timing.snap_grid` / the frontend `snap.js` use**, so the
  UI's persisted `view.snap` is a valid `snap=` param) quantizes start + length to the grid.
  v1 non-goals surfaced as
  warnings: pitch-bend events are **not** mapped to per-note bends (imported notes are flat);
  multiple tempi note the initial-tempo `file_bpm` choice. `adopt_bpm` is echoed for the caller
  (beats are tempo-invariant, so it does not change them).
- **`export.py`** — `export_midi(cfg, doc, project_dir)` → `write_midi` (`prior_mode="bend"`
  always) to `exports/<stem>.mid` → `{kind, path, warnings}`. `export_wav(cfg, doc, project_dir)`
  never re-renders: it re-plans the doc with the existing `render/meta.json` params — including
  that render's **own** `vocoder` (a legacy meta predating the key falls back to
  `cfg.default_vocoder`, which is exactly what such a render used), or a non-default-vocoder
  render would hash differently and look spuriously stale — and, if every
  planned segment hash matches the manifest, its cache WAV exists, and `mix.wav` exists, copies
  `mix.wav` to `exports/<stem>.wav`; otherwise raises `RenderStale` (→ 409, the client renders
  first). `<stem>` = sanitized name + `-YYYYmmdd-HHMMSS`. **Torch-free** (cache bookkeeping +
  file copy — no model/vocoder).

- **`tests/`** — 157 pytest cases, GPU/network/torch-free: `test_timing`,
  `test_project_store` (build defaults incl. **pan carry/clamp**, rev-conflict, backups),
  `test_api` (config/models/CRUD/rev-conflict + render validation 400s and
  render/meta 404), `test_voices` (overlap/chord/legato/determinism/overflow), `test_bend`
  (flat→no events, monotonic times + clamp + reset, wheel bounds), `test_midi_export`
  (voices→instruments, bend gating by prior_mode, velocity clamp, selection, error/warn),
  `test_peaks` (SPK1 round-trip), `test_render_api` (`run_render` monkeypatched: 202/409/meta),
  `test_cache` (segmentation/pads, hash stability + segment-relative + one-note-edit invalidates,
  plan cache-hit detection, stitch placement/offset/no-clip/crossfade, GC), `test_render_cache`
  (`_render_segments` monkeypatched: first render → all segments, re-render → all cache hits,
  edit one note → only its segment re-rendered, empty project → silence), `test_midi_import`
  (tempo-map beats via a mido tempo-change file, snap, pitch-bend/drum warnings, bad/empty midi),
  `test_export_api` (import replace/merge/adopt-bpm + bad-body/mode 400s; MIDI export written +
  served by `/media`; WAV export 409 when stale, success against a planted fresh render, stale
  again after an edit, freshness with a non-default/legacy-meta vocoder), `test_config`
  (yaml overrides incl. the vocoder keys, defaults derived from `common.config`),
  `test_library` (`scan_vocoders` filtering/sorting, `vocoder_checkpoint_dir` hf passthrough).

## Frontend additions (Phase 6, `static/`)

- **`js/io.js`** (new) — the toolbar **File group** (Import / WAV / MIDI) + the import-options
  modal + the status-bar download link. Import opens a `<input type=file accept=.mid,.midi>`,
  then a modal (replace vs merge, adopt-file-BPM, snap-on-import → uses the current grid) →
  `api.importMidi` (raw bytes) → on success reloads the doc into the store (same reset as project
  load) and flashes the note count + warnings. Export WAV auto-renders first on `render_stale`
  (`rendermgr.renderAndWait()`) and retries once; on success shows a `download` anchor at the
  `/media` path.
- **`js/api.js`** — `importMidi(id, file, opts)` now POSTs the file as the raw body with
  `{mode, snap, adopt_bpm}` query params (no `FormData`). `exportProject` unchanged (`{kind}`).
- **`js/rendermgr.js`** — added `renderAndWait()` (promise settled on render completion/failure)
  + `isRunning()` for the export auto-render flow.
- **`js/mock.js`** — `importMidi` returns `not_implemented` (client-side `.mid` parsing is out of
  scope for mock mode); `exportProject` returns a fake `{kind, path, filename, mock:true}`.
- **`js/main.js`** — `loadProject` refactored into `applyLoadedDoc` + a `reloadProject` handed to
  `io.init`. **`index.html`/`css`** — the File group, hidden file input, import modal, and the
  status-bar `.export-link`.

## Frontend files (`static/`, Phases 1/2/4/7 — zero-build ES modules)

`index.html` (transport bar → toolbar → DEV drawer → roll `#ruler/#keys/#grid/#overlay/#lane/
#wave/#minimap` → vibrato inspector → status footer → import modal) + `css/studio.css` (FL
charcoal theme). All canvases are DPR-scaled. Modules:

- **`js/state.js`** — the single app `store` + command-pattern undo/redo (cap 500), the
  multi-select `Set`, the sorted-notes cache, monotonic id allocation, and the **`markDirty`
  choke point** every doc mutation passes through (flips `save.state`→`unsaved`, `renderInfo.
  fresh`→`false`, fires `onMutation` handlers + the autosave `dirtyHandler`). Notes live in
  **beats**. `barBeatTick`, `pitchName`, `isBlackKey` helpers.
- **`js/timeline.js`** — beat↔px and pitch↔px transforms (linear semitone rows, high pitch
  top), `layout` geometry, pan/zoom clamps (`clampView`, `zoomAbout`), `noteRect` (shared by
  render + hit-test), and binary-search `visibleNotes` culling. `PITCH_LO/HI` = 33..100.
- **`js/render.js`** — all canvas drawing. `drawStage` (ruler/keys/grid+notes/lane/wave/
  minimap, on view/data change) + `drawOverlay` (hover, drag ghosts, bend editor, playhead —
  every rAF). FL flat-beveled notes; three-tier bar/beat/step gridlines; alternating black/
  white key row shading; per-note vibrato/bend glyphs. `techColor(name)` reads
  `config.articulations`; `shade`/`mix` color helpers. Owns the **minimap click/drag
  jump-scroll** (`wireMinimap` — centers the main view on the clicked beat).
- **`js/interact.js`** — grid/keys pointer + wheel state machine: draw/select/delete/slice
  tools, marquee, move, edge-resize, wheel-pan, ctrl-wheel zoom-about-cursor, **ctrl+drag =
  marquee multi-select even over notes** (shift+ctrl adds to the selection), **alt = temp free
  snap for any drag** (toggleable mid-drag; ctrl wins when both are held), **alt+drag on a note
  body = free unsnapped end-resize**, right-click delete, double-click → open bend editor,
  **piano-key click = preview note** (`player.previewNote`). Every mutation goes through an
  `editing.js` command.
- **`js/editing.js`** — command factories (each `{label, mutate, invert, selAfter?, phAfter?}`):
  create/delete/move/resize, `setVelocity`/`setVelocities`, `setPans`, `setBendsFlat`,
  `sliceNote`, `setTechnique`, `setBend`, `setVibrato`, `paste`. `makeNote` stamps the full
  schema (incl. `pan`, default bend/vibrato). Inverses are exact.
- **`js/snap.js`** — snap engine. `SNAP_OPTIONS` + `gridSize`/`snapBeat`/`snapLen`/`defaultLen`.
  **Keys are the backend snap ids** (`none/line/1/2step/1/3step/1/4step/1/6step/step/beat/bar`)
  so the id round-trips through `view.snap` and is valid as the `import-midi` `snap=` param.
- **`js/tools.js`** — tool state + toolbar buttons + the snap `<select>` (populated from
  `SNAP_OPTIONS`). **`js/clipboard.js`** — copy/cut/paste, beat-relative to the earliest
  selected note, pastes at the playhead (carries pan/bend/vibrato). **`js/keymap.js`** —
  window keydown → action dispatch: Ctrl+Z/Y/C/X/V/A/S/R, Space, Delete, Esc, tool letters
  P/E/D/C + digits 1–4, **Shift+1–4 = technique** (via `e.code`, so digits-alone stay tools),
  L loop, F follow, F9 render. (No duplicate bindings — ctrl-gated vs plain are distinct.)
- **`js/player.js`** — one `AudioContext`, a shared **beat-clock** (`clipBeat`). Two sources
  on the clock: the oscillator preview synth (`synth.js`) and the decoded rendered `mix.wav`
  buffer; `useRender()` picks the buffer only when it is loaded **and** `renderInfo.fresh`
  **and** not force-synth, else the synth (so any edit falls back to preview until re-render).
  Loop re-arms at each wrap. **`js/synth.js`** — lookahead oscillator preview (Labeler clone).
- **`js/technique.js`** (Phase 2) — Slur/Spiccato/Detache toolbar buttons + `applyByKey`.
  Selection semantics: assign to all selected, or **revert all to normal** if every selected
  note already has it (toggle). Buttons carry the articulation color + light when the whole
  selection shares the technique.
- **`js/vellane.js`** (Phase 2) — the bottom control lane: switchable **Velocity / Pan / Pitch**
  modes, bipolar bars for pan/pitch from a center line, drag-paint ramps → one undoable
  command (`setVelocities`/`setPans`/`setBendsFlat`). Pitch mode is read-only on notes carrying
  a multi-point bend curve (edit those in the bend editor; marked with a glyph).
- **`js/bendedit.js`** (Phase 2) — per-note pitch-bend polyline editor on `#overlay` (open on
  double-click). Control points `{beat, semitones}` rel note start, semitones clamped ±2 with a
  saturation warning glyph; drag/add(double-click)/remove(right-click, never the anchor); Esc /
  click-outside commits. Each gesture is one undoable `setBend`.
- **`js/vibrato.js`** (Phase 2) — the selection inspector strip: depth/rate/onset mini-sliders
  showing the common value or "mixed"; release writes one undoable partial `setVibrato` to
  every selected note.
- **`js/transport.js`** (Phase 2) — transport bar: play/pause/stop, loop toggle + **ruler
  loop-region drag** (top half) / playhead scrub (bottom half), BPM LCD (drag/dblclick/wheel),
  time-sig editing, bar:beat readout, master volume. BPM/time-sig write the doc via `markDirty`
  (so they invalidate a fresh render).
- **`js/rendermgr.js`** (Phase 4) — render orchestration: Render button → POST (`selection` when
  a selection exists, else `phrase`) with the DEV drawer's model/checkpoint/vocoder/prior_mode, polls
  `render/status` every 500 ms → progress strip, on done loads peaks + decodes the wav and marks
  the render fresh. `onMutation` hook dims the lane + falls the live source back to synth.
  `renderAndWait()` powers the export-WAV auto-render. `onProjectLoaded` adopts any on-disk
  render as **stale** (can't cheaply prove it matches the doc).
- **`js/waveform.js`** (Phase 4) — the `#wave` lane: fetches/parses `mix.peaks` (`SPK1`), draws
  the filled FL envelope beat-aligned (seconds→beats via bpm, so it re-aligns on zoom/scroll/
  bpm-change), dimmed + "STALE — re-render" hint when `renderInfo.fresh` is false.
- **`js/devopts.js`** (Phase 4) — the DEV drawer: model + checkpoint + **vocoder** selectors
  (all from `/api/models`; the vocoder dropdown lists every scanned `Vocoder/models/*` export
  and shows `"hf"` as **`stock (HF)`**), bend/quantized prior toggle, readouts (the last-render
  line reports `model/checkpoint · prior_mode · voc <name>`); selection persisted in localStorage
  and read by `rendermgr` for the render POST body. Tolerant of the real + mock `/api/models`
  shapes — a backend without a vocoder scan keeps the current selection selectable.
- **`js/main.js`** — bootstrap: loads config/models/project (creates a default if none), wires
  every module **once** (no double-subscribe), the rAF draw loop with follow-playhead, autosave
  (`putProject`, 1.5 s debounce, 409 → reload-server-copy recovery), `beforeunload` sendBeacon
  flush, and the status footer. `applyLoadedDoc` / `reloadProject` are the single reset path
  (project load **and** post-import).

## Future work

- **~~Conditioned checkpoints hookup~~ — DONE**: `_render_segments` builds all of the loaded
  checkpoint's conditioning tracks (articulation via `technique_model_vocab`, velocity, vibrato,
  note-boundary distances) per segment via `_segment_note_events` → `Arioso.infer.build_cond`;
  unconditioned checkpoints still pass `cond_frames=None`.
- **Pan → audio** — `pan` is stored + edited + folded into the segment hash but is **not** yet
  applied at render (renders are mono). Add a stereo pan stage to `stitch`/`vocode` output.
- **Per-note bend import** — `midi_import` currently drops MIDI pitch-bend events (imported
  notes are flat, warned); map them onto per-note `bend` curves.
- **Electron wrap** — package the frontend + a bundled server for a desktop app (the eventual
  VST/standalone), as decided with the user.

## Windows gotchas

- Registry maps `.js` → text/plain; `server.py` patches mimetypes at startup (ES modules
  hard-fail otherwise).
- No `--reload`: the model + vocoder are per-process singletons; uvicorn reload would reload
  them per worker.
- PowerShell: `curl` is an alias — use `curl.exe`.
- If the torch/numba OpenMP clash bites once the render phase lands, set
  `KMP_DUPLICATE_LIB_OK=TRUE` for the server process only.
