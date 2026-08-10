# common — memory palace

Project-wide shared code: the constants, audio I/O, mel/vocoder front-end (plus the
pitch-shifted `keyshift` variant), prior synthesis, onset alignment, **and the
on-disk dataset-root standard** that every top-level package agrees on. It is a
peer of `DataSynthesizer/`, `Labeler/`, `Arioso/`, and `Studio/` — owned by none of
them. The two data **producers**
(DataSynthesizer, Labeler) never import each other; they agree only through this
package (chiefly `dataset_schema.py`), and Arioso **consumes** what they emit.

## ⚠️ Reuse this — do not rewrite it

Before adding audio loading, wav writing, normalization, a sample-rate/mel
constant, a prior synth, an onset aligner, or *any* dataset-layout / vocab /
rasterizer logic to **any** module, import it from here instead:

```python
from common.config import SR, HOP_SIZE, TARGET_RMS_DBFS
from common.audio_io import load_mono, write_pcm16, normalize, voiced_rms_normalize
from common.vocoder import mel_spectrogram, mel_frames
from common.prior import quantized_prior
from common.dataset_schema import ARTICULATIONS, NoteEvent, DatasetRoot, frame_of
```

Do **not** redefine `SR = 44100`, re-implement PCM-16 writing / normalization,
re-derive the mel STFT, hardcode a `cond/velocity` dir name, or re-invent the
`frame = round(t*SR/HOP)` rounding rule in a consumer. If a genuinely shared
helper is missing, **add it to `common/`** (and document it here). Two invariants
depend on this single-source discipline: (1) the sample rate/mel params must be
one value everywhere so training reads audio at the exact rate/params the dataset
was written at, and (2) `frame_of` must be *the* rounding rule so the load-bearing
identity `onsets == [g.onset_frame for g in note_groups(...)]` holds for every
producer (Arioso's clip enumerator keys off it).

Import mechanics: run packages as modules from the project root
(`python -m DataSynthesizer.build_prior`, `python -m Arioso.train`, …) so
`import common` resolves.

---

## The standard dataset-root layout (the contract)

A **standard dataset root** is a self-describing directory any producer emits and
Arioso consumes. Arioso is a **multi-root** consumer: it can train over several
roots at once (e.g. the synthetic corpus `Data/` + the recorded `Data/datasets/gt_arky/`).
The canonical definition lives in [dataset_schema.py](dataset_schema.py); this is
its prose spec.

```
<root>/                          # e.g. Data/ (synthetic) and Data/datasets/gt_arky/
  manifest.json                  # canonical, schema_version 1
  split.json                     # written by Arioso.splits (NOT by producers)
  gt/<base>.wav                  # mono PCM16 @ 44100, voiced-RMS at TARGET_RMS_DBFS (-20 dBFS)
  target_mel/<base>.npy          # [128, T] float32 (common.vocoder mel)
  prior_mel/<base>.npy           # [128, T] float32, same T
  prior_wav/<base>.wav           # the rendered prior as audio, mono PCM16 @ 44100, len == gt/
  onsets/<base>.npy              # [K] int32 onset frame indices, sorted unique
  cond/articulation/<base>.npy   # [T] uint8 — only if listed in manifest "signals"
  cond/velocity/<base>.npy       # [T] uint8 — optional likewise
  cond/vibrato/<base>.npy        # [T] uint8 — optional likewise
```

Directory/file names are constants in the schema (`DIR_GT`, `DIR_TARGET_MEL`,
`DIR_PRIOR_MEL`, `DIR_PRIOR_WAV`, `DIR_ONSETS`, `cond_dir(signal)`, `MANIFEST_JSON`,
`SPLIT_JSON`) — never hardcode the strings. `split.json` is produced by
`Arioso.splits`, not the data producers, and is per-root (grouped by manifest `piece`).

**`prior_wav/` is optional and additive.** Both producers write it (Labeler
`compile`, DataSynthesizer `build_prior`) as the float prior *before* melling, so
it is the same signal `prior_mel/` was computed from, sample-for-sample aligned
with `gt/`. It exists so a consumer can **re-mel the pair from audio** — today
that is Arioso's pitch-shift augmentation (`Arioso.pitch_aug` +
`common.keyshift`), which needs waveforms, not mels. A root written before it
existed simply lacks the directory; consumers must check `os.path.isfile` and fall
back rather than fail (`Arioso.pitch_aug.wavs_available` is that gate), so no root
needs regeneration to stay trainable. It is deliberately **not** required by the
"a listed clip is complete" rule below.

### `manifest.json` shape

```json
{
  "schema_version": 1,
  "name": "synthetic",
  "frame": {"sr": 44100, "hop": 512, "n_mels": 128},
  "signals": ["articulation", "velocity", "vibrato"],
  "clips": {
    "<base>": {
      "n_frames": 1234,
      "n_samples": 631808,
      "duration_s": 14.3,
      "piece": "Kayser/Op20-01",
      "source": { "...": "producer-specific provenance" },
      "provenance": {"rev": 7, "notes_hash": "…", "compile_hash": "…"}
    }
  }
}
```

- **A listed clip is complete and trainable** — if a `base` appears in `clips`,
  all of its required artifacts (gt, target_mel, prior_mel, onsets, and every
  track in `signals`) exist on disk. Producers write the manifest entry **last**,
  atomically, so training never sees a half-built clip.
- **`frame`** is asserted against `common.config` on load (`load_manifest`); mels
  built at a different sr/hop/n_mels are unusable and rejected loudly.
- **`signals`** is **root-wide** — every listed clip in a root carries exactly
  that set of `cond/` tracks. The synthetic corpus has `signals: []`; `gt_arky`
  has all three. Arioso fills a signal a root lacks with the "unknown" id (below).
- **`piece`** is the split-grouping key (no piece in both train and val).
  Synthetic: `"{composer}/{catalog}"`; gt_arky: the clip_id. This is what
  `Arioso.splits` groups on.
- **`provenance`** is producer-specific (Labeler uses `{rev, notes_hash,
  compile_hash}` for incremental/idempotent recompile; synthetic clips carry the
  build's `source` block). Old `manifest.csv` (DataSynthesizer) / `notes.json`
  (Labeler) remain each producer's **internal** log — never read by Arioso.

### Encodings (the three conditioning tracks)

Every track is `[T]` uint8, one id per mel frame, `frame = round(t*44100/512)`
(`frame_of`). Class counts / rest ids are the schema constants
(`SIGNAL_NUM_CLASSES`, `SIGNAL_REST_ID`); Arioso's `CondSpec` registry derives its
embeddings from them so the on-disk encoding and the embedding tables cannot drift.

- **articulation**: `normal=0, slur=1, spiccato=2, detache=3, rest=4`
  (`num_classes=5`, pad=4). The CFG **unknown** id is `5` — added as an extra
  embedding row by the model, **never written to disk**. Unknown articulation
  *name* at rasterization is a hard `ValueError` (no silent fallback).
- **velocity**: `0=rest, 1..127=MIDI velocity` (`num_classes=128`, pad=0);
  unknown=128.
- **vibrato**: `0=off, 1=on, 2=rest` (`num_classes=3`, pad=2); unknown=3 — rest
  is kept **distinct** from "played without vibrato" (off).

`ARTICULATIONS = ("normal", "slur", "spiccato", "detache")` is the unified vocab
(the Labeler's vocab + rest); its ids are the tuple index.

### The cross-producer loudness contract

Every GT wav — YouTube rips for the synthetic corpus, the Labeler's recorded
clips — is scaled so its RMS over **voiced** (non-silent) segments hits
`TARGET_RMS_DBFS` (−20 dBFS), equalizing playing loudness across recordings
(`common.audio_io.voiced_rms_normalize`, measured over segments split by
`VOICED_TOP_DB`). The Arioso prior is masked-RMS level-matched to the **same**
`TARGET_RMS_DBFS` (`common.prior.MaskedRMS`), so prior and target levels agree —
which keeps the prior fully score-determined and **identical at train and
inference** (a hard requirement OT-CFM transport relies on). Both constants live
in `common/config.py` so the two normalizations can never disagree.

---

## Files

### config.py — canonical audio constants + contracts
- `SR = 44100` — the project sample rate; the single source of truth.
- `DEFAULT_PEAK = 0.95` — default target peak for `audio_io.normalize` and the
  post-normalization clip guard.
- `PCM_SUBTYPE = "PCM_16"` — the wav subtype every output uses. Canonical format:
  **mono, 16-bit PCM, `SR` Hz**.
- **Loudness contract**: `TARGET_RMS_DBFS = -20.0`, `VOICED_TOP_DB = 40.0` (see
  above) — shared by every GT normalization and the prior's level match.
- **Mel-spectrogram contract**: `HOP_SIZE = 512`, `N_FFT = 2048`, `WIN_SIZE = 2048`,
  `N_MELS = 128`, `FMIN = 0`, `FMAX = None`. These **must** match the BigVGAN-v2
  checkpoint (`nvidia/bigvgan_v2_44khz_128band_512x`); mels at any other values
  feed the vocoder garbage, so they live here as the single source of truth for
  data prep, training, and inference. `DataSynthesizer/config.py` re-exports
  `HOP_SIZE as HOP`; `Arioso.config` derives `FRAME_RATE = SR/HOP_SIZE`. Truly
  pipeline-only constants (`BOOKS`, book/dataset paths) stay in
  `DataSynthesizer/config.py`.
- **Project-default model checkpoints** — the two paths inference starts from, both
  anchored to the repo root so they resolve from any cwd:
  - `VOCODER_DIR = <repo>/Vocoder/models/ft_v2` — the active BigVGAN generator dir
    (`config.json` + `bigvgan_generator.pt`); the violin fine-tune. Set to `None` for
    the stock HF checkpoint. Overridable per call via `load_vocoder(checkpoint_dir=…)`.
  - `ACOUSTIC_CKPT = <repo>/Arioso/models/7-9-adr/checkpoint_final.pt` — the default
    Arioso checkpoint: `Arioso.infer`'s `--ckpt` default and Studio's initial
    selection. Keep it pointing inside `Arioso/models/<run>/` so Studio's model scan
    can resolve it (`Studio.config` splits it into `default_model` /
    `default_checkpoint` by basename, and derives `default_vocoder` from
    `VOCODER_DIR`'s basename — `"hf"` when it is `None`).
  Both live here so promoting a checkpoint is a one-line change that the app and every
  CLI pick up together.

### audio_io.py — shared audio I/O
- `load_mono(path, sr=SR)` — `librosa.load` to a mono array at `sr`.
- `write_pcm16(path, y, sr=SR)` — write a 16-bit PCM wav; returns `path`.
- `normalize(y, target_peak=DEFAULT_PEAK)` — rescale to a fixed peak so
  summed/loud audio doesn't clip; no-op on silence.
- `voiced_rms_normalize(y, target_dbfs=TARGET_RMS_DBFS, top_db=VOICED_TOP_DB)` —
  one global gain so the RMS over voiced (non-silent) segments hits the target,
  with a `DEFAULT_PEAK` clip guard. The dataset loudness convention.

### dataset_schema.py — the dataset-root standard (single source of truth)
The prose above **is** this module's contract; it is deliberately **torch-free**
(numpy + stdlib, `pretty_midi` only inside the one MIDI helper) so any producer
imports it without a GPU stack.
- **Layout constants** — `DIR_GT`/`DIR_TARGET_MEL`/`DIR_PRIOR_MEL`/`DIR_PRIOR_WAV`/
  `DIR_ONSETS`, `COND_ROOT`, `cond_dir(signal)`, `MANIFEST_JSON`, `SPLIT_JSON`,
  `MANIFEST_SCHEMA_VERSION = 1`. `DIR_PRIOR_WAV = "prior_wav"` is the optional
  rendered-prior audio described above.
- **Vocab / encoding constants** — `ARTICULATIONS`, `ARTIC_REST_ID`/`ARTIC_NUM_CLASSES`,
  the velocity/vibrato ids, and the derived `SIGNAL_NUM_CLASSES` / `SIGNAL_REST_ID`
  maps (Arioso's `CondSpec` registry reads these — the literals live once, here).
  `REST_SNAP_FRAMES = 10` — the legato/rest-gap threshold.
- `NoteEvent` — frozen dataclass `{start_s, end_s, pitch, velocity, articulation,
  vibrato, env_dct, vib_params, vib_model}`: the canonical per-note record every
  producer rasterizes from. The last three default to `None` and are *measured shape*
  descriptors baked into the rendered prior (`env_dct` → `envelope.py`,
  `vib_params`/`vib_model` → `vibrato_model.py`); **no rasterizer or conditioning
  encoding reads them**, so a producer that has not measured them emits identical
  `cond/` tracks. Only `Labeler.compile` fills them today.
- `frame_of(t_s)` — **THE** rounding rule `round(t_s*SR/HOP_SIZE)` (matches
  `build_prior`'s `np.round`); keeping it in one place is what makes the
  onset≡group-onset identity hold.
- `NoteGroup` + `note_groups(notes, offset_s, n_frames)` /
  `note_groups_from_midi(midi_path, offset_s, n_frames)` — group notes by rounded
  onset frame (chords/sub-frame onsets merge); the group onset frames are
  bit-identical to `onset_frames` and to `common.prior.note_onsets` rounded.
- `expand_to_frames(groups, ids, n_frames, *, rest_id, rest_snap=REST_SNAP_FRAMES)`
  — per-group ids → `[T]` uint8 track with the hybrid rest-snap fill (legato notes
  bridge a `<= rest_snap` gap; real gaps stay `rest_id`).
- **Rasterizers** — `rasterize_articulation` (onset-group first-note wins;
  unknown name → `ValueError`; rest-snap fill), `rasterize_velocity` / `rasterize_vibrato`
  (per-note span fill), `onset_frames(notes, n_frames, offset_s=0)` → `[K]` int32.
- **Manifest IO** — `load_manifest(root)` / `write_manifest(root, manifest)`
  (atomic `.tmp` + `os.replace`, both validate schema_version + frame block).
- `DatasetRoot(path)` — read-only accessor Arioso and eval consume a root through:
  `.name`, `.signals` (frozenset), `.clips`, `.basenames()`, `.n_frames(base)`,
  `.piece(base)`, `.gt_path/target_mel_path/prior_mel_path/onsets_path/offsets_path(base)`,
  `.prior_wav_path(base)` (the **optional** rendered-prior audio — the path is always
  returned, so the caller `isfile`-checks it), and `.cond_path(signal, base)`
  (→ `None` when the root lacks that signal).

### prior.py — MIDI → sawtooth informed-prior via a composable `PriorSynth`
The prior synthesis (moved here from DataSynthesizer, so Arioso inference, the
Labeler compile step, and the Studio renderer all render the *same* prior without
importing a producer). A **Strategy**-pattern pipeline: one `PriorSynth` keeps the
per-note summation loop (polyphony/double-stops sum) and delegates each axis to an
injected `Protocol`-typed component — pitch (`Quantized` | `PitchBend`), source
(`NaiveSaw` | `BandlimitedSaw` | `AdditiveSaw`), harmonic law (`AlphaTilt` |
`RoundedCorner`, the `n_c=8, p=2` sweep winner), envelope (`HardGate` | `Fade`),
body (`Identity`, the no-EQ seam), and leveler (`MaskedRMS` | `Peak`). The
`PRIOR_*` / `FADE_MS` knobs live **here** now (the single source of truth for the
prior shape).
- `quantized_prior(...)` — factory assembling the spec-baseline pipeline from the
  `PRIOR_*` knobs (additive rounded-corner saw + masked-RMS to `TARGET_RMS_DBFS`).
  `pitch=` selects the trajectory via `_make_pitch`: `"quantized"` (default;
  byte-identical train/inference prior), `"quantized_vibrato"`, or `"bend"`
  (Studio bend-mode). The single prior source of truth reused at inference.
- `QuantizedVibrato` (`pitch="quantized_vibrato"`) — `Quantized`'s constant note
  frequency multiplied by `vibrato_model.vibrato_ratio(note.vib_params, n, sr)` when
  an in-memory producer attached params, and **exactly** `Quantized` when it did not
  (so every disk-loaded MIDI renders byte-identically — guarded by
  `test notebooks/vibrato_param_experiment/prior_identity_guard_vib.py --check`).
  An unrecognized `note.vib_model` also falls back to the constant rather than
  raising: one stale note must not abort a whole render. `Labeler.compile` uses this
  mode (`CompileParams.prior_pitch`).
- **Per-note attributes the render loop reads off in-memory `pretty_midi.Note`s**:
  `note.env_dct` (else the legacy `velocity/127` gain) and `note.vib_params` /
  `note.vib_model` (only in `quantized_vibrato` mode). Both are attached by
  `Labeler.midi_io.notes_to_pretty_midi(..., with_env=, with_vibrato=)` and neither
  survives a `.write()` — disk MIDIs are unaffected by design.
- `PriorSynth.render(midi, sr=SR, total_samples=None)` and `note_onsets(midi)`
  accept **either a path to a `.mid` or an in-memory `pretty_midi.PrettyMIDI`** —
  so the Labeler renders a prior straight from a note list
  (`midi_io.notes_to_pretty_midi`) with no disk round-trip.
- `render_prior` / `render_prior_bend` — legacy peak-normalized wrappers (the
  synthetic `build_dataset` pass + wav CLI). `PB_RANGE_SEMITONES` (±2) is the
  pitch-wheel range Studio's bend export clamps to.
- The mel front-end is deliberately **not** in the pipeline — the caller mels
  after any alignment shift, via `common.vocoder.mel_frames`.
- CLI: `python -m common.prior clip.mid -o clip_prior.wav`.

### envelope.py — per-note energy envelope (`env_dct`), the prior's dynamics
Describes a note's loudness *shape* with 3 cosine (DCT-I-style) coefficients fit to
its A-weighted power-dB envelope — the fidelity knee found by
`test notebooks/envelope_param_experiment` (~1.6 dB median reconstruction RMSE).
gt_arky's MIDI velocity carries almost no dynamic information (r ≈ 0.22–0.26 with
actual loudness), so this replaces it in the prior. numpy + librosa only (torch-free).
- `loudness_db(y, sr=SR)` — `[T]` per-frame A-weighted loudness in power-dB (STFT
  power, A-weighting applied in the **linear power** domain, mean over bins,
  `10*log10`). Uses the repo's `N_FFT`/`HOP_SIZE` (2048/512), not the F0 hop.
- `fit_env_dct(env_db)` → `[c0, c1, c2]` = **level, tilt, arch** over basis
  `cos(k*pi*(i+0.5)/L)`; always length 3 (degenerate `L` zero-pads).
- `eval_env_dct(coeffs, u)` — the continuous form at normalized position `u ∈ [0,1)`.
- `env_gain(coeffs, n_samples)` — per-sample **linear amplitude** gain
  (`10**(dB/20)`) sampled at note-relative midpoints; what `PriorSynth.render`
  multiplies the source by. Only cross-note level and within-note shape survive the
  whole-clip `MaskedRMS` match, which cancels the absolute dB offset.
- `note_env_from_wav(loud_db, start_s, end_s, ...)` — slice a note out of a clip
  loudness track (repo `frame_of` rounding) and fit it.
- `velocity_to_env_dct(velocity)` → `[A*vel + B, 0, 0]` — the flat fallback for notes
  with no measurement. `VEL_C0_A`/`VEL_C0_B` are a **UI-anchored** map (vel 1..127 →
  −30..+16 dB, the Labeler/Studio envelope-lane display range) so a velocity drag and
  an envelope-handle drag move level at the same dB-per-pixel rate; mirrored in
  `Labeler/static/js/state.js`.

### vibrato_model.py — per-note vibrato params (`vib_params`), the prior's vibrato
The `quantized` prior holds every note at its exact MIDI frequency, so vibrato is the
one thing a violin recording always has and the prior never does. This measures a
note's F0 modulation (pyin) and compresses it to a handful of numbers `prior.py`
replays as a per-sample frequency ratio. numpy + librosa + scipy (torch-free).
- **Registry** `MODELS: {name: VibratoModel}` with `VIB_MODEL = "rampboth5"` the
  single swap point. `rampboth5` `[D0, gD, f0, s, phi]`: half-depth ramp
  `D(t)=clip(D0+gD·t, 0, 200)` cents, rate ramp `f(t)=f0+s·t` Hz, phase
  `Φ(t)=2π(f0·t + s·t²/2)+phi`, `osc = D(t)·sin(Φ)`. Alternates: `dramp4`
  `[D0, gD, f, phi]` and `lfo3` `[D, f, phi]` (also `rampboth5`'s warm-start parent).
  Unknown model name → `ValueError` from every public entry point.
- **Selection + tiering** (`test notebooks/vibrato_param_experiment`, 921 fittable
  vibrato notes over the 37 verified gt_arky clips; median in-sample cents RMSE
  against a **7.72** no-model floor): `rampboth5` 5 params → **4.29**; `dramp4` 4 →
  4.95 (better-behaved rate, pick this if a smaller model is ever wanted); `lfo3`
  3 → 5.57. Caveat: the fitted `f0`/`s` are **curve-fit parameters, not a rate
  measurement** (r = 0.089 vs the `auto.vibrato_rate_hz` detector; `f0` pinned at a
  bound on ~30% of notes) — read `auto.vibrato_rate_hz` for anything human-readable.
- **Parameters are absolute-time and onset-anchored**, never duration-relative, so a
  note-boundary edit cannot silently change their meaning — and the unregularized
  ramps extrapolate badly, so a producer must *clear* params on a boundary edit and
  re-measure (the Labeler front-end does).
- **The affine nuisance baseline `b0 + b1·t` is fitted jointly and discarded** — it
  absorbs intonation offset and slide, which are not vibrato. Pre-detrending instead
  would let a half-cycle of vibrato leak into the slope. It is never stored.
- `f0_hz_track(y, sr=SR)` — whole-clip `librosa.pyin` (hop 256 / frame 1024 / 175–2100
  Hz / `resolution=0.05`), `[T]` Hz with NaN unvoiced. Per-note extraction is wrong
  here (Viterbi restarts would land artifacts on the note boundaries). **Slow: ~3.5×
  the clip duration** (a measured 24 s clip took 84 s).
- `note_cents_from_track(f0_hz, pitch, start_s, end_s, ...)` → `(t_rel, cents)`; cents
  vs the **scored** pitch, `t_rel` from true frame times so slice rounding never
  shifts the timebase.
- `mask_note(t, cents)` — drop unvoiced, drop `|cents| > 200` (octave slips), then
  despike at 5 robust sigma of the 3-frame median deviation.
- `fit_note_vibrato(f0_hz, pitch, start_s, end_s, ...)` → params (4 dp) or `None`.
  Rejects `dur < 0.25 s`, `< 12` surviving frames, or `< 50%` surviving. Multi-start:
  top-2 Lomb-Scargle peaks (3.5–10 Hz) + the optional `auto_rate_hz` seed, deduped at
  0.3 Hz, plus the `lfo3` warm start; `scipy.least_squares` trf with finite bounds and
  explicit `x_scale`; lowest SSE wins.
- `vibrato_cents(params, t, model=)` / `vibrato_ratio(params, n_samples, sr, model=)`
  — the render path; `t = arange(n)/sr` keeps rendered phase on the fitted timebase.
- `vibrato_from_auto(rate_hz, extent_cents)` — detector-shaped fallback params, gated
  off by `VIB_FALLBACK = False` (an unfitted note renders flat, which is honest).

### onset_align.py — (prior, GT) onset alignment (in-memory, array-first)
Measures the residual global time offset between a prior and its GT so priors can
be shifted into alignment (moved here so `Labeler.align` uses it without importing
DataSynthesizer).
- `estimate_offset_seconds(prior, gt, sr=SR, max_lag_s=1.0)` — cross-correlate
  onset-strength envelopes; positive ⇒ prior lags GT.
- `shift_samples(y, offset_seconds, sr=SR)` — time-shift a waveform (pad/truncate,
  same length). Pure array op.
- `align_prior_to_gt(prior, gt, sr=SR)` — estimate then apply the negation
  (advance a lagging prior); returns `(aligned_prior, applied_seconds)`.
- CLI: `python -m common.onset_align prior.wav gt.wav [--apply -o out.wav]`.

### vocoder.py — BigVGAN-v2 mel ↔ waveform vocoder
- Thin wrapper over NVIDIA's BigVGAN-v2, vendored at `external/BigVGAN` (no PyPI
  package; weights pull from the HF Hub on first load). Checkpoint:
  `nvidia/bigvgan_v2_44khz_128band_512x`.
- `load_vocoder(device="cpu", use_cuda_kernel=False, checkpoint_dir=None)` — load +
  assert the checkpoint's mel params equal the `config.py` contract (fails loudly on
  drift). `use_cuda_kernel=True` selects the fused kernel; if it can't be built the
  loader warns and falls back to the PyTorch-native path — **still on the GPU**, slower.
  Checkpoint resolution order: the `checkpoint_dir` argument > `config.VOCODER_DIR`
  (the project default — the violin fine-tune) > the HF Hub baseline;
  `checkpoint_dir="hf"` forces the stock baseline regardless of the config (what the
  A/B tooling and Studio's `stock (HF)` option use). A local dir must hold
  `config.json` (or `ft_config.json`) plus `bigvgan_generator.pt` or the newest
  step-prefixed `g_????????` snapshot.
- `mel_spectrogram(wav)` — mel via BigVGAN's own `meldataset.mel_spectrogram`, fed
  the `config.py` params, so it can never drift from the checkpoint. **Use this for
  any mel computation — do not re-implement an STFT.**
- `mel_frames(wav)` — `mel_spectrogram(wav)[0]` as a `[N_MELS, T]` float32 numpy
  array (batch dim dropped, to numpy): the one feature front-end every producer
  mels through (DataSynthesizer build, Labeler compile, Studio render, Arioso
  inference), so the training feature can never drift.
- `vocode(model, mel)` — run the model, returns a 1-D float waveform.
- Self-test: `python -m common.vocoder --selftest` round-trips a `Data/gt` clip.

#### Enabling the fused BigVGAN CUDA kernel (optional speedup)

`use_cuda_kernel=True` JIT-compiles a fused CUDA kernel at load time, which needs
a full CUDA **toolkit** (not just PyTorch's bundled runtime) plus a host C++
compiler. On a machine that lacks them, `torch.utils.cpp_extension.CUDA_HOME` is
`None` and the build fails, so the loader falls back to the native path. To get
the speedup (one-time, computer-level setup — several GB):

1. **CUDA Toolkit 12.x** matching your PyTorch build (`python -c "import torch;
   print(torch.version.cuda)"` — e.g. `12.4` for `torch 2.6.0+cu124`). Install
   from NVIDIA; default path `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`.
   Ensure `%CUDA_PATH%\bin` is on `PATH` so `cpp_extension.CUDA_HOME` resolves
   (it reads `CUDA_HOME` / `CUDA_PATH`); set `CUDA_HOME` to the same path if needed.
2. **MSVC C++ build tools** — "Visual Studio 2022 Build Tools" with the "Desktop
   development with C++" workload (provides `cl.exe`, required by
   `cpp_extension.load()` on Windows).
3. **Launch the process from a shell where `cl.exe` is on `PATH`** — e.g. the
   "x64 Native Tools Command Prompt for VS 2022" (or run `vcvars64.bat` first).
   Otherwise the build can't find the compiler.
4. First load with `use_cuda_kernel=True` compiles the kernel (verbose, ~a minute)
   into `external/BigVGAN/alias_free_activation/cuda/build/` and caches it; later
   loads are fast. Success = no fallback `RuntimeWarning`.

No BigVGAN source changes are needed: its hardcoded `sm_70`/`sm_80` arch flags are
forward-compatible with newer GPUs (e.g. sm_89 Ada).

### keyshift.py — pitch-shifted mel extraction (DiffSinger nvSTFT "keyshift")
A mel of a waveform **as if it had been transposed**, without resampling it. The
shift happens *inside the STFT*: run it with a window/FFT of
`WIN_SIZE * 2**(cents/1200)` while holding `HOP_SIZE` fixed, then read the first
`N_FFT//2+1` bins through the ordinary 2048-point mel basis. Bin `i` of a
`win_new`-point FFT sits at `i*SR/win_new` Hz but the basis reads it as
`i*SR/N_FFT` Hz, so every partial lands a factor `win_new/N_FFT` higher — a pitch
shift for the cost of one STFT, with the frame grid untouched. Pure DSP + integer
math, numpy in / numpy out, CPU-only and torch-free at the API surface, so it is
safe to call from dataloader workers. Sole consumer today: `Arioso.pitch_aug`.
- `mel_frames_keyshift(wav, cents)` → `[N_MELS, T]` float32; positive `cents`
  shifts up.
  - **Bit-identity contract**: at `cents == 0.0` this is `np.array_equal` to
    `common.vocoder.mel_frames` — *identical*, not close (`win_new` collapses to
    2048, the pad to the vendored 768/768, the bin crop/pad to a no-op, the level
    rescale to `* 1.0`). This is what lets augmented and memmapped training items
    share one feature manifold; pinned by `Arioso/tests/test_keyshift.py` and
    re-checked on real audio by `test notebooks/pitch_shift_validation.ipynb`.
  - **Duration-preserving**: `T = 1 + (len(wav) - HOP_SIZE) // HOP_SIZE` for
    *every* `cents`, because the pre-STFT reflect pad is sized from the scaled
    window (`win_new - HOP_SIZE`) rather than held at 768/768, so `win_new` cancels
    out of the `center=False` frame count. That is what keeps a shifted mel
    frame-aligned with score-rasterized conditioning tracks.
- `keyshift_factor(cents)` / `scaled_win(cents)` — the frequency multiplier and the
  window+FFT length that realize it (2048 → 1933 at −100 cents, 2170 at +100).
- `margin_frames(max_cents)` / `slice_read_range(start, end, margin)` — slice-read
  geometry for windowed reads of long files: how many context frames a shifted
  slice needs on each side (3 at ±100 cents) and the exact sample range to read for
  frames `[start, end)`, returned as `(read_start_sample, read_n_samples,
  head_frames)`. Both file edges are **exact**, not approximate — at frame 0 the
  slice's own reflect pad *is* the whole-file one, likewise at EOF.
- **Trap — never route this through the vendored BigVGAN mel.**
  `external/BigVGAN/meldataset.mel_spectrogram` caches the mel basis *and* the Hann
  window under one key built from `n_fft`, so calling it with a scaled `n_fft`
  would also build a mel **basis** for that scaled `n_fft` — exactly the wrong
  thing, since the whole mechanism depends on the basis staying at `N_FFT=2048`
  while only the window scales. That vendored file must never be modified either;
  this module re-implements its (small, fixed) math, and the `cents == 0`
  bit-identity assertion is what stops the copy from drifting.
- **Known artifact, `cents < 0` only** (`win_new < 2048`): the shifted STFT yields
  fewer than `N_FFT//2+1` bins and the absent top rows are zero-filled, so mel bin
  127 (~22 kHz) loses all support and sits on the `log(1e-5)` floor while bin 126 is
  partially attenuated. Measured at −100 cents on real audio: bin 127 pinned on
  100 % of frames, bin 126 on 72 %, bins ≤ 125 unaffected. Harmless for a 44.1 kHz
  violin corpus — but **re-measure before widening the shift range**, since
  `win_new` keeps shrinking and the floored band walks downward.
- **Cost**: ~6-7 ms per 10 s of audio per track on CPU (≈13 ms for a prior+target
  pair, ~19 ms end-to-end through `Arioso.pitch_aug.shifted_pair` including the two
  ranged wav reads) — hidden behind the GPU step by `run.num_workers`.

## Consumers

- `DataSynthesizer/` — `build_prior` (prior mel/wav/onsets via `common.prior` /
  `common.onset_align` / `common.vocoder`), `export_manifest` / `migrate_root`
  (standard-root manifest via `dataset_schema`), `build_dataset` / `download_audio`
  (audio I/O + loudness contract).
- `Labeler/` — `compile` (standard-root output via `dataset_schema` rasterizers +
  `common.prior` + `common.vocoder`), `align` (`common.onset_align` / `common.prior`),
  `media` (`common.vocoder.mel_spectrogram`).
- `Arioso/` — `config`/`clips`/`splits`/`dataset` (multi-root consume via
  `DatasetRoot` + the schema constants), `pitch_aug` (the **only** consumer of
  `keyshift` + `DIR_PRIOR_WAV`), `infer` (prior + mel + conditioning).
- `Studio/` — `render`/`bend` (prior + mel + `PB_RANGE_SEMITONES` from
  `common.prior` / `common.vocoder`).
