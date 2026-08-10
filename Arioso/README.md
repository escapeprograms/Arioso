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
  - **Pitch-shift augmentation knobs** (deferred toggle, OFF). `pitch_aug_p` (default **0.0** — the
    fraction of TRAIN samples re-melled at a random shift; 0 = off) and `pitch_aug_cents` (default
    **100.0** — the half-width of the uniform cents draw, i.e. ±1 semitone). They live on
    `AriosoConfig` (the science half, embedded in checkpoints); `build_pitch_aug` reads them plus
    `seed`. See **pitch_aug.py**.
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
  - **Growing corpora.** `update_split(root, cfg) -> (split, summary)` reconciles an existing split
    with the current manifest **additively**: prune basenames that left the manifest, keep every
    surviving assignment on its current side (val stability across corpus growth), and assign the
    new ones by whole piece — a new clip of an already-assigned piece follows that piece, genuinely
    new pieces are sorted + seed-shuffled and sent to val only until the piece-level val target
    `max(1, round(n_pieces * val_frac))` is met. No split file yet ⇒ plain `make_split`. The file
    schema is unchanged (`train`/`val` sorted, `n_pieces`, `n_val_pieces`), so every consumer is
    untouched; `summary` = `{created, added_train, added_val, pruned, n_pieces, n_val_pieces}`.
  - **Stale-split warning.** Because the verbatim load is load-bearing, a split that no longer
    covers its manifest would silently hide those clips from *both* train and val. `make_split`
    now prints one `[splits] WARNING: ...` line naming the count when it loads such a split
    (warning only — no behavior change). `missing_from_split(root, split)` is the check itself.
  - CLI: `python -m Arioso.splits --data-root <root> [--update | --overwrite]` — `--update` is the
    additive reconcile (prints `+N train, +N val, pruned N (val pieces K/N)`), `--overwrite`
    recomputes from scratch (reshuffles the held-out pieces). The Labeler's compile calls
    `update_split` at the end of every run, so a compiled root's split always covers its manifest.
- **dataset.py** — `AriosoDataset` (mmap mel slices), `LengthBucketBatchSampler`, `collate`
  (frame masks), `build_dataloader`.
  - **Pitch-shift augmentation flow.** `AriosoDataset(roots, clips, specs=(), pitch_aug=None)` — a
    `PitchAug` policy (from **pitch_aug.py**) replaces the memmap read of `x0`/`x1` with a re-mel
    from `prior_wav/` + `gt/` for the drawn subset. The dataset carries an `epoch` token set by
    `set_epoch(...)`, mirroring `LengthBucketBatchSampler.set_epoch` — `train.py` calls **both**
    with the global step at the start of each pass over the pool, a unique per-epoch token, so the
    shifts are redrawn every epoch. The order inside `__getitem__` is deliberate:
    **draw first, check availability second**, so the RNG stream depends only on
    `(seed, epoch, index)` and never on which roots happen to carry `prior_wav/`. `wavs_available`
    is memoized per `(root.path, base)`. When the policy is `None` or inactive **no RNG is drawn
    and no waveform is opened** — items are byte-identical to the pure-memmap path. Because the
    shift is duration-preserving, `length` and every cond/boundary track are untouched.
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
- **pitch_aug.py** — **dataset-level** augmentation: on-the-fly pitch shift of the prior/target pair.
  Arioso now has **two** augmentation seams and they never overlap. `augment.py` is **tensor / step
  level** — post-batch transforms and auxiliary losses applied *inside the training step*, on tensors
  already on the device, no disk access and no per-sample RNG stream. This module is **dataset
  level** — applied *inside* `AriosoDataset.__getitem__`, before collation, working from the
  **waveforms on disk** (`prior_wav/` + `gt/`) rather than the precomputed mels. Anything that
  changes what a *sample is* (rather than how a batch is transformed) belongs here; future
  dataset-level augmentations extend this module. CPU-pure numpy + soundfile, safe in dataloader
  workers.
  - `PitchAug(p, max_cents, seed)` — the frozen per-split policy; `.active` is `p > 0 and
    max_cents > 0`, `.margin` is `common.keyshift.margin_frames(max_cents)`.
    `cents_for(epoch, index)` returns the shift or `None`. Its generator is seeded from
    `(seed, epoch, index)` and **nothing else** — not the worker id, not process RNG state — so
    `num_workers` cannot change a single draw and a re-run of a config reproduces every shift. Draw
    order is fixed (Bernoulli gate first, uniform cents only if it passes), so changing `max_cents`
    never re-rolls *which* samples are augmented.
  - `shifted_pair(root, base, start, end, cents, margin)` — both tracks through the identical path:
    `slice_read_range` → a **ranged** `soundfile.read` (never `common.audio_io.load_mono`, which
    would decode a whole recording for a 10 s clip) → `common.keyshift.mel_frames_keyshift` → drop
    the margin context. A frame-count mismatch raises rather than pads (silent padding would desync
    cond from mel). ~19 ms for a 10 s pair.
  - `wavs_available(root, base)` — the graceful-fallback gate (both wavs present?). A root
    predating `prior_wav/` — the synthetic `Data/` root — is simply never augmented instead of
    failing the run.
  - `build_pitch_aug(cfg, split_name)` — the single factory `build_dataloader` calls. Any split but
    `"train"` gets a **structurally inactive** policy (`p=0.0`), so val can never be augmented no
    matter what the config says: the val metric always scores the real mels.
  - The shift itself is duration-preserving (`common/README.md` → `keyshift.py`), so the augmented
    mels stay frame-for-frame aligned with the score-rasterized cond tracks and the clip's
    `[start, end)` framing is untouched. Both tracks of a sample take the **same** shift, so the
    prior→target relation the model learns is preserved.
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
  - **Checkpoint flags** — `--ckpt` is **optional**, defaulting to `common.config.ACOUSTIC_CKPT`
    (the project-default Arioso checkpoint, currently `Arioso/models/7-9-adr/checkpoint_final.pt`);
    pass it only to run a different one. **`--vocoder DIR|hf`** picks the BigVGAN checkpoint for
    the listening render — a local generator dir, or `hf` for the stock pretrained baseline;
    default is `common.config.VOCODER_DIR` (the violin fine-tune). `eval/copy_synthesis.py` takes
    the same `--vocoder` flag. Both defaults live in `common/config.py`, so the CLI, the eval
    scripts and Studio agree on one project default.
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
"$PY" -m Arioso.infer score.mid               # -> Arioso/samples/ (project-default ckpt + vocoder)
"$PY" -m Arioso.infer score.mid --ckpt Arioso/models/<run>/checkpoint_final.pt --articulation spiccato --vibrato
"$PY" -m Arioso.infer score.mid --vocoder hf  # ...listen through the stock BigVGAN baseline
"$PY" -m Arioso.eval.metrics --ckpt Arioso/models/<run>/checkpoint_final.pt --plot delta.png   # --ckpt required here
"$PY" -m pytest Arioso/tests -q                  # unit tests (no GPU/network)
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
"$PY" -m Arioso.train --config Arioso/configs/8-8-pitch-aug.yaml  # gt_arky only + pitch-shift aug (p=0.5)
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

**Pitch-shift augmentation in YAML** — two `model:` keys plus one `run:` key:

```yaml
model:
  pitch_aug_p: 0.5          # fraction of TRAIN samples re-melled at a random shift (0.0 = off, the default)
  pitch_aug_cents: 100.0    # half-width of the uniform cents draw (default 100 = +/- 1 semitone)
run:
  num_workers: 4            # TRAIN-loader workers only; hides the pitch-aug STFT cost behind the GPU step
```

`num_workers` is a `RunSettings` field (runtime, never embedded in a checkpoint) and applies to the
**train loader only** — `build_dataloader` is called without it for val, which stays at 0. It is
safe to raise because the augmentation's RNG is seeded from `(seed, epoch, index)` alone, so worker
count cannot change a draw; `persistent_workers` deliberately stays `False`, since a persistent
worker would hold a stale dataset copy and never see `set_epoch`, freezing the shifts at epoch 0.
`configs/8-8-pitch-aug.yaml` is the worked example (gt_arky alone, `p=0.5`, 4 workers), validated by
`test notebooks/pitch_shift_validation.ipynb`.

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

## Tests (`Arioso/tests/` — 107 cases, no GPU / network / data root)

`"$PY" -m pytest Arioso/tests -q`. Everything is pure CPU and builds whatever roots it needs under
`tmp_path`, so the suite runs anywhere.

- **test_keyshift.py** (39) — the four contracts `common.keyshift` promises: `cents == 0` is
  **bit-identical** to `common.vocoder.mel_frames` (so the re-implemented STFT can never silently
  drift from the vendored BigVGAN one); `T` is cents-invariant; the shift really is a pitch shift
  (the keyshifted mel of a tone matches the mel of the *resynthesized* transposed tone, bin for
  bin); and the known top-bin artifact for `cents < 0` is locked to bins 126/127 so it cannot widen
  into the band the model learns. Tolerances are measured facts — a failure means the DSP changed.
- **test_pitch_aug.py** (23) — `Arioso.pitch_aug` + the dataset wiring: **off is off** (`p == 0` and
  `pitch_aug=None` give items byte-identical to the memmap path and draw no RNG at all, so a
  baseline run cannot change); **train only** (`build_pitch_aug` is structurally inactive for any
  other split); **graceful fallback** (a root without `prior_wav/`/`gt/` is never augmented even at
  `p == 1.0`); **alignment** (an augmented item keeps its frame count and its cond/boundary tracks
  bit-for-bit — only `x0`/`x1` move); and the `shifted_pair` slice geometry / RNG-stream contract.
- **test_boundary_cond.py** (19) — offset frames, the `boundary_distances` core, the
  `BoundarySinusoid` featurizer, `BoundaryCondSpec` config round-trip, YAML resolution.
- **test_splits.py** (16) — `make_split`'s verbatim load + stale warning, and `update_split`'s
  additive reconciliation (never moves a surviving assignment, prunes, assigns by whole piece).
- **test_cond_build.py** (10) — `Arioso.infer.build_cond` over hand-built notes: key/dtype
  contract, boundary semantics, train/inference equivalence, refactor guard.

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
  `manifest.json`, `split.json`, optional `cond/` and optional `prior_wav/`. Model **artifacts**
  live under the package and are gitignored: checkpoints in `Arioso/models/checkpoint_<step>.pt`, listening wavs in
  `Arioso/samples/`.
- **Per-frame conditioning is baseline-OFF** (`AriosoConfig.conditioning` defaults to `()` — the
  main synthetic corpus carries no labels). A labelled run opts in with `conditioning:
  [articulation, velocity, vibrato]` (concatenating those per-frame embeddings to `[x_t, x_0]`);
  the id tracks live under each root's `cond/<signal>/` (built by `Labeler.compile` for gt_arky,
  unknown-filled for a root that lacks a signal). The three encodings are documented in
  `common/README.md`.
- **Pitch-shift augmentation is baseline-OFF** (`pitch_aug_p` defaults to 0.0) and needs
  `prior_wav/` in the root — the recorded `Data/datasets/gt_arky` root has it, the synthetic `Data/`
  root does **not**, and a root without it is silently never augmented rather than erroring. It
  costs two STFTs per augmented sample (~19 ms per 10 s pair), so raise `run.num_workers` to hide
  it behind the GPU step. Validated end-to-end in `test notebooks/pitch_shift_validation.ipynb`.
- **Out of scope** (deferred, toggles default OFF): body EQ / tilt / rolloff, vibrato/LFO prior,
  F0/voicing conditioning, CFG **conditioning dropout** (`cond_dropout`, the framework is wired but
  defaults 0.0), energy-balanced loss, vocoder fine-tuning, polyphony.
- Receptive field of the 20-block WaveNet is ~4093 frames (~47 s) — comfortably covers any clip.
