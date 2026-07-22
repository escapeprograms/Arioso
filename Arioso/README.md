# Arioso — memory palace

The acoustic model at the center of the violin-synthesis system: an **OT-CFM velocity field** that
transports a score-synthesized **sawtooth-prior mel** toward a **real-violin target mel**, entirely
in the mel domain. This is the **no-EQ baseline** (`SPEC_Arioso_v1_baseline.md`): the prior is an
unshaped quantized sawtooth, so the model's job is almost entirely **timbre** (body resonance,
spectral envelope, bow noise, attack/decay shape). Every later addition (body EQ, vibrato priors,
articulation conditioning, energy-balanced loss, vocoder fine-tuning) is measured against it.

The single question it answers: *does a quantized-pitch sawtooth prior, with no spectral shaping,
carry enough information for the velocity field to learn violin timbre?*

This directory is a **Python package** (`__init__.py`); run its modules with `python -m
Arioso.<module>` from the project root so `import common` resolves.

## What was already built vs. what Arioso adds

The data layer (spec Sections 3-5) is **pre-built** by the two producers (`DataSynthesizer/`,
`Labeler/`) + `common/`, emitted as **standard dataset roots** (`common.dataset_schema` layout:
`manifest.json` + `gt/` + `target_mel/` + `prior_mel/` + `onsets/` + optional `cond/`). The
synthetic root `Data/` is 888 usable clips / 134 pieces. Arioso is a **multi-root consumer** and
reuses all of it. It adds:

- consumption of a **spec-faithful prior**, built by `DataSynthesizer.build_prior`
  (`common.prior`'s composable `PriorSynth`: **shaped additive saw** (rounded-corner harmonic
  ladder, `n_c=8`) + **masked-RMS level match**) into each root's `prior_mel/`. (The synthetic
  build's legacy `prior_mel_quant` is a *naive aliased*, *peak-normalized* saw; the level mismatch
  derails OT-CFM transport, which the masked-RMS fixes. The corner law drops the saw's HF excess
  the targets don't carry — mel RMSE 3.070→2.718.) The prior is a **dataset artifact**; Arioso
  just consumes the mels and rebuilds the same prior at inference via `common.prior.quantized_prior`.
  **Checkpoints trained on the old polyBLEP prior get a mismatched inference prior after this flip
  and must be retrained** — retrain against a freshly rebuilt `prior_mel/`.
- the **train-time clip/dataset/split layer** (`clips.py`, `dataset.py`, `splits.py`) — **multi-root**.
- the **WaveNet->DiT model** (`model/`), the **OT-CFM loop** (`cfm.py`, `train.py`), **inference**
  (`infer.py`), and **evaluation** (`eval/`).

## Design decisions (this baseline)

- **Mel contract is imported, not re-pulled.** `common.config` is the single source of truth and
  `common.vocoder.load_vocoder()` asserts it equals the BigVGAN checkpoint `config.json` at load —
  so Arioso imports `common.config` rather than re-reading the checkpoint (the spec's "pull from
  checkpoint" intent is satisfied by that assertion).
- **Rectangular note gating, no ADSR** (build decision). `envelope="rect"` is the baseline;
  `"fade"` reuses the 5 ms anti-click ramp as a toggle. ADSR is deferred.
- **Masked-RMS matches a fixed level, not a per-recording target.** Every GT (whatever its
  producer) is voiced-RMS normalized to -20 dBFS, so the prior's sounding-frame RMS is scaled to
  that **constant** (`common.config.TARGET_RMS_DBFS`, the cross-producer loudness contract shared
  with the GT normalization). This keeps the prior fully score-determined => **identical at train
  and inference** (a hard spec requirement that "match the target's RMS" would otherwise break,
  since inference has no target).
- **Held-out-piece split** by the manifest `piece` key (synthetic: `composer/catalog`; gt_arky:
  clip id) — no piece in both train and eval, per root.
- **Selection metrics are vocoder-independent:** velocity/recon MSE, MCD, Delta-mel. FAD/MUSHRA are
  deferred (standard FAD needs an audio-embedding net, re-introducing the vocoder the spec bars as
  arbiter). The frozen vocoder is for listening checks only.

## Pipeline / data flow

```
score .mid ─ common.prior.quantized_prior().render (additive saw, corner ladder, masked-RMS -> -20 dBFS)
   │                                                              │
   │  (train) DataSynthesizer.build_prior: + manifest offset shift + mel ─> <root>/prior_mel/<base>.npy
   │                                       + aligned onset frames ────────> <root>/onsets/<base>.npy
   │  (conditioning, per labelled root) ────────────────────────────────> <root>/cond/{articulation,velocity,vibrato}/<base>.npy
   │
   ├ splits.make_splits(roots) ─> per-root <root>/split.json   (held-out pieces, merged)
   ├ clips.enumerate_clips(roots) ─> fixed 5-10 s onset-aligned clip pool, keyed (root, base, start, end)
   └ dataset.build_dataloader ─> length-bucketed batches + frame masks (+ batch["cond"], unknown-filled per root)
        │
   train.py: x_t,v_target = cfm.interpolate(x0,x1,t); v = model(x_t,x0,t,mask,cond); masked_mse
        │                                            (AdamW, warmup+cosine, bf16, EMA)
   infer.py: x0 (+cond from build_cond) -> Euler -> mel -> frozen BigVGAN -> wav
   eval/: copy_synthesis (vocoder ceiling) · metrics (MSE/MCD/Delta-mel, cond-aware)
```

## Files

- **config.py** — `AriosoConfig` (one frozen dataclass: model + training + clip + infer hparams;
  spec-baseline defaults). Mel contract imported from `common.config`; **prior knobs live in
  `common.prior`** (the prior is a dataset artifact). Output-dir names: `SPLIT_FILE`, `CKPT_DIR`,
  `SAMPLES_DIR`; `PRIOR_MEL_DIR`/`ONSETS_DIR` re-exported from `common.dataset_schema`
  (`"prior_mel"`/`"onsets"` — the standard-root dir names).
  - **Per-frame conditioning framework.** `CondSpec` (frozen dataclass) describes one per-frame
    categorical conditioning signal: `name`, `num_classes`, `emb_dim`, `dir` (the `cond/<signal>`
    subdir of `<base>.npy` `[T]` uint8 id tracks, via `dataset_schema.cond_dir`), and `pad_id` (id
    used for batch padding). The **three standard signals** are constants whose class counts / rest
    (pad) ids are DERIVED from `common.dataset_schema` (so the on-disk encoding and the embedding
    tables can never drift — the encodings are documented once in `common/README.md`):
    `ARTICULATION_COND` (5 cls, emb 64, pad 4), `VELOCITY_COND` (128 cls, emb 64, pad 0),
    `VIBRATO_COND` (3 cls, emb 16, pad 2). The `SIGNALS` registry `{articulation, velocity,
    vibrato}` maps a name → its canonical `CondSpec` (a YAML `conditioning: [articulation]` names a
    signal and the loader resolves it here; adding a signal is one CondSpec constant + one registry
    entry, mirroring the `SOLVERS`/`SCHEDULES` pattern). `AriosoConfig.conditioning: tuple[CondSpec,
    ...]` defaults to `()` — **the unconditioned baseline** (the main synthetic corpus carries no
    labels); a labelled run opts in via `conditioning: [articulation, velocity, vibrato]`.
    `cond_dropout` (default 0.0, a deferred CFG toggle, OFF) is the fraction of samples whose ENTIRE
    conditioning is swapped for the "unknown" id during training. `cond_dim` (property) = sum of the
    specs' `emb_dim` (0 when none). `__post_init__` also checks: spec names unique, each
    `emb_dim > 0` and `0 <= pad_id < num_classes`, `0 <= cond_dropout < 1`.
  - **Checkpoint config (de)serialization.** `cfg_to_dict(cfg)` = `dataclasses.asdict` (CondSpec ->
    plain dict) so checkpoints stay **`torch.load(weights_only=True)`-safe** under torch 2.6 (no
    pickled dataclass instances). `cfg_from_dict(d)` filters `d` to current field names (ignoring
    removed/foreign keys), rebuilds `conditioning` from CondSpec dicts (accepts CondSpec instances
    too) and coerces list-typed tuple fields; a **missing `conditioning` key -> `()`**, so
    pre-conditioning checkpoints reconstruct the exact old (no-conditioning) architecture. Both
    `train._save` and the W&B config use `cfg_to_dict`; `infer`/`eval`/the notebooks rebuild via
    `cfg_from_dict`.
- **(prior + build_prior live in `common.prior` / `DataSynthesizer/`)** — `common.prior.PriorSynth`
  (composable: pitch / source / envelope / body / leveler components) + `quantized_prior(...)` factory
  is the single prior source of truth, reused at inference; `DataSynthesizer.build_prior` is the
  one-time dataset pass writing each root's `prior_mel/` + `onsets/`.
- **clips.py** — **multi-root.** `open_roots(paths) -> list[DatasetRoot]` (actionable error on a
  missing `manifest.json`); `Clip(root, basename, start, end)` where `root` is the index into the
  ordered `DatasetRoot` list; `enumerate_clips(roots, basenames_per_root, cfg)` deterministic
  onset-aligned 5-10 s pool, using each root's manifest `n_frames` + `onsets/<base>.npy` (the old
  `manifest.csv` `n_frames` reader is gone). A basename shared by two roots stays distinct via `root`.
- **splits.py** — held-out-**piece** split (group key = manifest `piece`). `make_split(root, cfg)`
  computes/loads **one** root's `<root>/split.json` (an existing file is loaded verbatim, preserving
  a migrated root's exact split); `make_splits(roots, cfg)` merges an ordered root list into
  `{"train": [(root_idx, base), ...], "val": [...]}` (root order preserved).
- **dataset.py** — `AriosoDataset` (mmap mel slices), `LengthBucketBatchSampler`, `collate`
  (frame masks), `build_dataloader`.
  - **Multi-root + conditioning flow.** `AriosoDataset(roots, clips, specs=())` — a clip's `root`
    index selects its `DatasetRoot`, whose accessors give the prior/target mel paths. When `specs`
    is non-empty each item also carries `item["cond"][spec.name]`, a `[T]` int64 id track mmap-sliced
    identically to the mels (from `root.cond_path(spec.name, base)`); **a signal the clip's root does
    not provide** (`spec.name not in root.signals`) is filled with the CFG "unknown" id
    (`spec.num_classes` — always in-range, the embedding has `num_classes + 1` rows). With no specs
    the item dict is unchanged (no `"cond"` key). `collate(batch, specs=())` packs each spec into
    `out["cond"][name]` `[B, T_max]` long, pre-filled with `spec.pad_id`. `build_dataloader(data_roots,
    split_name, batch_size, cfg, ...)` opens the roots once, `make_splits` → `enumerate_clips`
    preserving root order, and binds `collate_fn=functools.partial(collate, specs=cfg.conditioning)`
    (partial stays picklable for future `num_workers>0`).
- **cfm.py** — `interpolate` (OT-CFM x_t + v_target), `masked_mse`. `sigma=1e-4`. Training
  augmentations (off-path perturbation, ADR) live in **augment.py**, which imports `interpolate`
  from here (no import back — no cycle).
- **augment.py** — extensible training augmentations along two axes: **path transforms** (pre-forward
  `(x_t, v_target)` transforms — `perturb_off_path` / `PathAug`) and **auxiliary losses** (need the
  model — Anti-Drift Rectification). `build_augmentor(cfg) -> Augmentor` is the single factory
  `train.py` calls.
  - **Off-path augmentation** (`perturb_off_path`, deferred toggle default OFF). For a
    Bernoulli(`path_aug_p`) subset of train samples, nudge `x_t` off the straight OT line by
    `std*t(1-t)*z` and re-aim the target to `v_target - std*t*z`, which preserves the endpoint
    (`x_tilde + (1-t)*v_hat == x_t + (1-t)*v_target`) so constant velocity from the perturbed point
    still lands on the path at `t=1` — teaching the field a **restoring component** off the line
    (the v1 diagnosis found none). Two config keys: `path_aug_p` (fraction perturbed, 0 = off) and
    `path_aug_std` (noise scale; effective displacement std is `path_aug_std*t(1-t) <= std/4`). Unlike
    simply raising `sigma` (whose noise is *not* re-aimed), the re-aimed target here teaches
    restoration. `masked_mse_per_sample` backs the per-`t` eval breakdown below.
  - **Anti-Drift Rectification** (`adr_per_sample` / ADR, deferred toggle default OFF; DEFAR,
    arXiv 2606.28226; ADR only, no Frequency Compensation). One extra model forward on the
    drift-simulated state `x_hat = x_t + (t1-t0)*v_t` (`t1 >= t0`, grads flow through `v_t`) whose
    direction is penalized toward the data anchor `v_adr = x1 - (1-sigma)*x_hat` — a masked, fp32,
    unit-vector squared error. `L = L_FM + adr_beta*L_ADR`. Two config keys: `adr_beta` (weight,
    0 = off) and `adr_p` (per-step gate, paper-faithful 1.0). Logged as `val/adr_loss` for all runs.
- **model/** — `timestep.py` (sinusoidal t_emb 256), `wavenet.py` (20 DiffSinger blocks, dilations
  `[1..512]x2`, gated act, skip-sum), `dit.py` (3 AdaLN-Zero **zero-init** + RoPE blocks, 6x64, FFN
  1536), `arioso.py`, `conditioning.py`.
  - **arioso.py** — `AriosoModel`: input proj `(256 + cond_dim)->384` -> wavenet -> dit -> head
    ->128. Builds `self.cond_enc = ConditioningEncoder(cfg)` iff `cfg.conditioning` is non-empty.
    **Forward contract** is now `forward(x_t, x_0, t, frame_mask=None, cond=None)`: raises
    `ValueError` unless `cond` is provided **iff** the model has conditioning; the conditioning
    embedding `[B, cond_dim, T]` is concatenated to `[x_t, x_0]` before `in_proj`.
  - **conditioning.py** — `ConditioningEncoder(cfg)`: one `nn.Embedding(num_classes + 1, emb_dim)`
    per `CondSpec` (in a `ModuleDict` keyed by name). `forward(cond)` validates the keys equal the
    spec names, embeds each `[B, T]` id track, and concatenates over channels -> `[B, cond_dim, T]`.
    The extra row `num_classes` is the **"unknown" embedding**: in `training` mode, with probability
    `cfg.cond_dropout`, ONE per-sample Bernoulli mask (shared across all specs) swaps that sample's
    entire conditioning for the unknown row — classifier-free-guidance style. Never present in data.
- **train.py** — OT-CFM loop, AdamW(2e-4, wd 0.01), warmup 4000 -> cosine, grad-clip 1.0, bf16,
  EMA (`delta=min(0.9999,(s+1)/(s+10))`), raw+EMA checkpoints. Train loop + `evaluate()` build
  `cond = {k: v.to(device) …}` from `batch["cond"]` (or `None`) and pass `cond=` to the model.
  Checkpoints and the W&B config embed the config via `cfg_to_dict` (weights_only-safe). `--smoke`
  for a short validation run. Logs loss/lr/val-MSE to W&B (`archimedesli/Arioso`) when
  `WANDB_API_KEY` is set; `--no-wandb` opts out. `--resume` continues a named run from its highest
  step checkpoint; optimizer state is not saved so AdamW restarts cold (a brief loss bump that
  washes out), and it requires an explicit run name. The train step applies `perturb_off_path` when
  `cfg.path_aug_p > 0`. `evaluate()` reports velocity MSE over a 10-point `T_GRID` (round-robin per
  sample) both **on-path** (`val/velocity_mse`) and **off-path** (`val/offpath_velocity_mse`, a
  fixed-seed `OFFPATH_EVAL_STD` perturbation — a restoration probe comparable across runs), plus
  per-`t` curves.
- **infer.py** — `build_prior_mel`, `build_cond`, `integrate` (Euler), `generate_mel`
  (chunk+crossfade), frozen BigVGAN to wav. `main()` loads the checkpoint **first** and rebuilds the
  model from its embedded config (`cfg_from_dict`), so pre-conditioning checkpoints load unchanged.
  - **build_cond(notes, n_frames, specs)** — `{name: [T] track}` for every spec the checkpoint
    expects, all rasterized from the same `NoteEvent` list (offset 0.0, matching `build_prior_mel`'s
    no-shift convention). Categorical specs are `[T]` uint8 id tracks via the `common.dataset_schema`
    rasterizers (`articulation` from each note's stamped name, `velocity` from the MIDI's own
    per-note velocities, `vibrato` from the per-note flag); boundary specs are `[T]` **int64**
    distance tracks — `onset_frames`/`offset_frames` (drop vs clamp, the training policy) through
    `Arioso.dataset.boundary_distances` (sentinel −1; must stay int64 — uint8 would corrupt it).
  - **--articulation** (`choices=ARTICULATIONS`, default `normal`) and **--vibrato /--no-vibrato**
    CLI args apply one constant per signal to every note (per-note control is a future extension).
    `main()` assembles `cond_frames` generically over `cfg.conditioning`; the deprecated `"technique"`
    signal is a clear `NotImplementedError` (its classifier was removed — retrain with
    articulation/velocity/vibrato). `integrate(..., cond=None)` and `generate_mel(..., cond_frames=None)`
    thread the fixed conditioning through the ODE — each track is uploaded once and sliced per chunk;
    a length != prior-mel-frames mismatch is fatal.
- **eval/copy_synthesis.py** — step-0 vocoder-ceiling sanity (run first); vocodes GT audio directly,
  builds no model, so it needs no conditioning handling. **eval/metrics.py** — recon/transport MSE,
  MCD, Delta-mel plots. Rebuilds the model from the checkpoint's embedded config (`cfg_from_dict`);
  per held-out recording it loads each categorical spec's track via the root's `cond_path` (or the
  unknown fill for a root lacking that signal), builds boundary tracks from the root's on-disk
  `onsets`/`offsets` arrays via `boundary_distances` (missing array → all −1 inactive), and passes
  `cond_frames=` through `generate_mel`.

## Run

```bash
PY="C:/Users/archi/Miniconda3/envs/ai-violin/python.exe"   # the ai-violin env
"$PY" -m DataSynthesizer.build_prior --limit 4   # smoke: regenerate 4 prior mels
"$PY" -m DataSynthesizer.build_prior             # full pass (888 clips)
"$PY" -m Arioso.splits --data-root Data      # held-out-piece split (per root)
"$PY" -m Arioso.clips --data-root Data --split train   # clip-pool stats
"$PY" -m Arioso.eval.copy_synthesis          # step 0: vocoder ceiling (run before training)
"$PY" -m Arioso.train --smoke                # short pipeline validation
"$PY" -m Arioso.train                        # full run (~1e5 steps; tune to convergence)
"$PY" -m Arioso.train --no-wandb             # ...same, without W&B logging
"$PY" -m Arioso.infer score.mid --ckpt Arioso/models/checkpoint_final.pt   # -> Arioso/samples/
"$PY" -m Arioso.infer score.mid --ckpt Arioso/models/checkpoint_final.pt --articulation spiccato --vibrato
"$PY" -m Arioso.eval.metrics --ckpt Arioso/models/checkpoint_final.pt --plot delta.png
```

## Run configs (YAML)

Ablations no longer require editing code: a YAML run-config (see `Arioso/configs/`) describes a run
over the frozen `AriosoConfig` + a new `RunSettings` (in `run_config.py`). Two sections map 1:1 onto
those dataclasses — `model:` (the science, embedded in checkpoints) and `run:` (runtime/env knobs,
never embedded) — plus an optional `config_version:`.

```bash
"$PY" -m Arioso.train --config Arioso/configs/default.yaml        # loads to exactly the code defaults
"$PY" -m Arioso.train --config Arioso/configs/conditioned_gt.yaml # multi-root + all 3 signals (unknown-filled synthetic)
"$PY" -m Arioso.train --config Arioso/configs/unconditioned.yaml  # a partial-override ablation
"$PY" -m Arioso.train --config Arioso/configs/default.yaml --batch-size 2   # CLI overrides YAML
"$PY" -m Arioso.train --data-root Data --data-root Data/datasets/gt_arky    # repeatable multi-root CLI
"$PY" -m Arioso.train                                             # no --config -> exact old behavior
```

A run-config is a **partial override**: the loader starts from the code defaults and applies only
the keys present, so missing keys keep their defaults (this *is* the backward-compat mechanism) and
a mistyped key is a hard error listing the valid keys. `sr`/`hop`/`n_mels`/`in_ch` are **locked**
(the mel contract lives in `common.config`) and rejected in YAML. **Data roots** are the
`run.data_roots` list (ordered root paths; CLI `--data-root` repeatable), default `("Data",)`; the
legacy single-root `run.out_dir: X` YAML key is accepted as an alias for `data_roots: [X]` (warns,
`_RENAMED`). `configs/conditioned_gt.yaml` is the worked example: `data_roots: [Data,
Data/datasets/gt_arky]` + `conditioning: [articulation, velocity, vibrato]`. Each run writes its
fully-resolved config to `Arioso/models/config_<run-name>.yaml` (uploaded to W&B) for reproducibility.

**Precedence:** smoke clamps > CLI flags > YAML > code defaults.

**Backward-compatibility rules** (a YAML written today must still train correctly later):
1. New hyperparameter → add an `AriosoConfig` field **with a behavior-preserving default**; it is
   automatically a valid YAML key with no other change.
2. Never rename/remove/repurpose an existing YAML key. If unavoidable, add it to `_RENAMED` in
   `run_config.py` (the old key keeps working and warns).
3. New loss/solver/schedule → a new function + one registry entry (`LOSSES`/`SOLVERS`/`SCHEDULES`);
   never change an existing entry's semantics.
4. New conditioning signal → add its encoding constants to `common.dataset_schema`
   (`SIGNAL_NUM_CLASSES`/`SIGNAL_REST_ID` + a rasterizer) so producers emit it under
   `cond/<signal>/`, then a `CondSpec` constant deriving its classes/pad from those schema
   constants + one `SIGNALS` registry entry + an inference-time branch in `infer.build_cond`.
   (A YAML `conditioning: [...]` then names it; a root lacking it is unknown-filled automatically.)

## Dependencies & caveats

- Env **ai-violin**: `torch` (2.6, CUDA), `numpy`, `scipy`, `librosa`, `soundfile`, `pretty_midi`,
  `matplotlib` (metrics plot), plus the vendored BigVGAN (pulled in via `common.vocoder`). The
  vocoder checkpoint downloads from HF Hub on first use (cached).
- **W&B logging** (optional): `pip install wandb`, then copy `.env.example` -> `.env` (gitignored)
  and set `WANDB_API_KEY` (or export it). Runs land in `archimedesli/Arioso`. A missing key/package
  is non-fatal — training just skips logging. Disable explicitly with `--no-wandb`.
- **Prior must come from `common.prior`** (`quantized_prior`) and mels from
  `common.vocoder.mel_spectrogram`/`mel_frames` — a re-implemented prior/mel would break
  train/inference identity or the vocoder.
- A **dataset root** holds training data only (gitignored): per-root `prior_mel/`, `onsets/`,
  `manifest.json`, `split.json`, optional `cond/`. Model **artifacts** live under the package and
  are gitignored: checkpoints in `Arioso/models/checkpoint_<step>.pt`, listening wavs in
  `Arioso/samples/`.
- **Per-frame conditioning is baseline-OFF** (`AriosoConfig.conditioning` defaults to `()` — the
  main synthetic corpus carries no labels). A labelled run opts in with `conditioning:
  [articulation, velocity, vibrato]` (concatenating those per-frame embeddings to `[x_t, x_0]`);
  the id tracks live under each root's `cond/<signal>/` (built by `Labeler.compile` for gt_arky,
  unknown-filled for a root that lacks a signal). The three encodings are documented in
  `common/README.md`.
- **Out of scope** (deferred, toggles default OFF): body EQ / tilt / rolloff, vibrato/LFO prior,
  F0/voicing conditioning, CFG **conditioning dropout** (`cond_dropout`, the framework is wired but
  defaults 0.0), energy-balanced loss, vocoder fine-tuning, polyphony.
- Receptive field of the 20-block WaveNet is ~4093 frames (~47 s) — comfortably covers any clip.
