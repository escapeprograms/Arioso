# common — memory palace

Project-wide shared code: the constants, audio I/O, mel/vocoder front-end, prior
synthesis, onset alignment, **and the on-disk dataset-root standard** that every
top-level package agrees on. It is a peer of `DataSynthesizer/`, `Labeler/`,
`Arioso/`, and `Studio/` — owned by none of them. The two data **producers**
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
  onsets/<base>.npy              # [K] int32 onset frame indices, sorted unique
  cond/articulation/<base>.npy   # [T] uint8 — only if listed in manifest "signals"
  cond/velocity/<base>.npy       # [T] uint8 — optional likewise
  cond/vibrato/<base>.npy        # [T] uint8 — optional likewise
```

Directory/file names are constants in the schema (`DIR_GT`, `DIR_TARGET_MEL`,
`DIR_PRIOR_MEL`, `DIR_ONSETS`, `cond_dir(signal)`, `MANIFEST_JSON`, `SPLIT_JSON`) —
never hardcode the strings. `split.json` is produced by `Arioso.splits`, not the
data producers, and is per-root (grouped by manifest `piece`).

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
- **Layout constants** — `DIR_GT`/`DIR_TARGET_MEL`/`DIR_PRIOR_MEL`/`DIR_ONSETS`,
  `COND_ROOT`, `cond_dir(signal)`, `MANIFEST_JSON`, `SPLIT_JSON`,
  `MANIFEST_SCHEMA_VERSION = 1`.
- **Vocab / encoding constants** — `ARTICULATIONS`, `ARTIC_REST_ID`/`ARTIC_NUM_CLASSES`,
  the velocity/vibrato ids, and the derived `SIGNAL_NUM_CLASSES` / `SIGNAL_REST_ID`
  maps (Arioso's `CondSpec` registry reads these — the literals live once, here).
  `REST_SNAP_FRAMES = 10` — the legato/rest-gap threshold.
- `NoteEvent` — frozen dataclass `{start_s, end_s, pitch, velocity, articulation,
  vibrato}`: the canonical per-note record every producer rasterizes from.
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
  `.piece(base)`, `.gt_path/target_mel_path/prior_mel_path/onsets_path(base)`, and
  `.cond_path(signal, base)` (→ `None` when the root lacks that signal).

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
  `PRIOR_*` knobs (additive rounded-corner saw + masked-RMS to `TARGET_RMS_DBFS`);
  `pitch="quantized"` (default; byte-identical train/inference prior) or `"bend"`
  (Studio bend-mode). The single prior source of truth reused at inference.
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
- `load_vocoder(device="cpu", use_cuda_kernel=False)` — load + assert the
  checkpoint's mel params equal the `config.py` contract (fails loudly on drift).
  `use_cuda_kernel=True` selects the fused kernel; if it can't be built the loader
  warns and falls back to the PyTorch-native path — **still on the GPU**, slower.
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

## Consumers

- `DataSynthesizer/` — `build_prior` (prior/onsets via `common.prior` /
  `common.onset_align` / `common.vocoder`), `export_manifest` / `migrate_root`
  (standard-root manifest via `dataset_schema`), `build_dataset` / `download_audio`
  (audio I/O + loudness contract).
- `Labeler/` — `compile` (standard-root output via `dataset_schema` rasterizers +
  `common.prior` + `common.vocoder`), `align` (`common.onset_align` / `common.prior`),
  `media` (`common.vocoder.mel_spectrogram`).
- `Arioso/` — `config`/`clips`/`splits`/`dataset` (multi-root consume via
  `DatasetRoot` + the schema constants), `infer` (prior + mel + conditioning).
- `Studio/` — `render`/`bend` (prior + mel + `PB_RANGE_SEMITONES` from
  `common.prior` / `common.vocoder`).
