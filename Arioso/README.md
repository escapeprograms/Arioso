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
Arioso.<module>` from the project root so `import common` / `import DataSynthesizer` resolve.

## What was already built vs. what Arioso adds

The data layer (spec Sections 3-5) is **~80% pre-built** by `DataSynthesizer/` + `common/`:
`data/manifest.csv` (888 usable clips / 134 pieces), GT audio, `target_mel`, the existing
`prior_mel_quant`, and the BigVGAN-matched mel front-end. Arioso **reuses** all of it. It adds:

- a **spec-faithful prior**, built by `DataSynthesizer.build_prior` (`DataSynthesizer.synthesizePrior`'s
  composable `PriorSynth`: anti-aliased polyBLEP saw + **masked-RMS level match**) into
  `data/prior_mel_arioso/`. (The built `prior_mel_quant` is a *naive aliased*, *peak-normalized* saw;
  the level mismatch derails OT-CFM transport, which the masked-RMS fixes.) The prior is a **dataset
  artifact owned by DataSynthesizer**; Arioso just consumes the mels and rebuilds the same prior at
  inference via `synthesizePrior.quantized_prior`.
- the **train-time clip/dataset/split layer** (`clips.py`, `dataset.py`, `splits.py`).
- the **WaveNet->DiT model** (`model/`), the **OT-CFM loop** (`cfm.py`, `train.py`), **inference**
  (`infer.py`), and **evaluation** (`eval/`).

## Design decisions (this baseline)

- **Mel contract is imported, not re-pulled.** `common.config` is the single source of truth and
  `common.vocoder.load_vocoder()` asserts it equals the BigVGAN checkpoint `config.json` at load —
  so Arioso imports `common.config` rather than re-reading the checkpoint (the spec's "pull from
  checkpoint" intent is satisfied by that assertion).
- **Rectangular note gating, no ADSR** (build decision). `envelope="rect"` is the baseline;
  `"fade"` reuses the 5 ms anti-click ramp as a toggle. ADSR is deferred.
- **Masked-RMS matches a fixed level, not a per-recording target.** Every GT was voiced-RMS
  normalized to -20 dBFS by DataSynthesizer, so the prior's sounding-frame RMS is scaled to that
  **constant** (`DataSynthesizer.config.TARGET_RMS_DBFS`, the single source of truth shared with the
  GT normalization). This keeps the prior fully score-determined => **identical at train and
  inference** (a hard spec requirement that "match the target's RMS" would otherwise break, since
  inference has no target).
- **Held-out-piece split** by `(composer, catalog)` — no piece in both train and eval.
- **Selection metrics are vocoder-independent:** velocity/recon MSE, MCD, Delta-mel. FAD/MUSHRA are
  deferred (standard FAD needs an audio-embedding net, re-introducing the vocoder the spec bars as
  arbiter). The frozen vocoder is for listening checks only.

## Pipeline / data flow

```
score .mid ─ synthesizePrior.quantized_prior().render (polyBLEP saw, quantized, masked-RMS -> -20 dBFS)
   │                                                              │
   │  (train) DataSynthesizer.build_prior: + manifest offset shift + mel ─> data/prior_mel_arioso/<base>.npy
   │                                       + aligned onset frames ────────> data/onsets_arioso/<base>.npy
   │
   ├ splits.make_split ─> data/arioso_split.json   (held-out pieces)
   ├ clips.enumerate_clips ─> fixed 5-10 s onset-aligned clip pool
   ├ (conditioning) data/technique_arioso/<base>.npy  [T] per-frame technique ids
   └ dataset.build_dataloader ─> length-bucketed batches + frame masks (+ batch["cond"])
        │
   train.py: x_t,v_target = cfm.interpolate(x0,x1,t); v = model(x_t,x0,t,mask,cond); masked_mse
        │                                            (AdamW, warmup+cosine, bf16, EMA)
   infer.py: x0 (+cond from build_technique_frames) -> Euler -> mel -> frozen BigVGAN -> wav
   eval/: copy_synthesis (vocoder ceiling) · metrics (MSE/MCD/Delta-mel, cond-aware)
```

## Files

- **config.py** — `AriosoConfig` (one frozen dataclass: model + training + clip + infer hparams;
  spec-baseline defaults). Mel contract imported from `common.config`; **prior knobs live in
  `DataSynthesizer.config`** (the prior is a dataset artifact). Output-dir names: `SPLIT_FILE`,
  `CKPT_DIR`, `SAMPLES_DIR`; `PRIOR_MEL_DIR`/`ONSETS_DIR` re-exported from `DataSynthesizer.config`.
  - **Per-frame conditioning framework.** `CondSpec` (frozen dataclass) describes one per-frame
    categorical conditioning signal: `name`, `num_classes`, `emb_dim`, `dir` (the `data/` subdir of
    `<base>.npy` `[T]` uint8 id tracks), and `pad_id` (id used for batch padding). `TECHNIQUE_COND`
    is the default signal (`num_classes=6`, `emb_dim=64`, `dir=technique_arioso`, `pad_id=REST_ID`),
    re-exporting `TECHNIQUE_DIR`/`TECHNIQUE_CLASSES`/`REST_ID` from `DataSynthesizer.config`.
    `AriosoConfig.conditioning: tuple[CondSpec, ...]` is `(TECHNIQUE_COND,)` — **default ON**; the
    unconditioned ablation is `AriosoConfig(conditioning=())`. `cond_dropout` (default 0.0, a
    deferred CFG toggle, OFF) is the fraction of samples whose ENTIRE conditioning is swapped for
    the "unknown" id during training. `cond_dim` (property) = sum of the specs' `emb_dim` (0 when
    none). `__post_init__` also checks: spec names unique, each `emb_dim > 0` and
    `0 <= pad_id < num_classes`, `0 <= cond_dropout < 1`.
  - **Checkpoint config (de)serialization.** `cfg_to_dict(cfg)` = `dataclasses.asdict` (CondSpec ->
    plain dict) so checkpoints stay **`torch.load(weights_only=True)`-safe** under torch 2.6 (no
    pickled dataclass instances). `cfg_from_dict(d)` filters `d` to current field names (ignoring
    removed/foreign keys), rebuilds `conditioning` from CondSpec dicts (accepts CondSpec instances
    too) and coerces list-typed tuple fields; a **missing `conditioning` key -> `()`**, so
    pre-conditioning checkpoints reconstruct the exact old (no-conditioning) architecture. Both
    `train._save` and the W&B config use `cfg_to_dict`; `infer`/`eval`/the notebooks rebuild via
    `cfg_from_dict`.
- **(prior + build_prior now live in `DataSynthesizer/`)** — `synthesizePrior.PriorSynth`
  (composable: pitch / source / envelope / body / leveler components) + `quantized_prior(...)` factory
  is the single prior source of truth, reused at inference; `DataSynthesizer.build_prior` is the
  one-time dataset pass writing `prior_mel_arioso/` + `onsets_arioso/`.
- **splits.py** — `make_split(out_dir, cfg)` held-out-**piece** split -> `arioso_split.json`.
- **clips.py** — `enumerate_clips(out_dir, basenames, cfg)` deterministic onset-aligned 5-10 s pool.
- **dataset.py** — `AriosoDataset` (mmap mel slices), `LengthBucketBatchSampler`, `collate`
  (frame masks), `build_dataloader`.
  - **Conditioning flow.** `AriosoDataset(out_dir, clips, specs=())` — when `specs` is non-empty each
    item also carries `item["cond"][spec.name]`, a `[T]` int64 id track mmap-sliced identically to
    the mels (from `data/<spec.dir>/<base>.npy`); with no specs the item dict is unchanged (no
    `"cond"` key). `collate(batch, specs=())` packs each spec into `out["cond"][name]` `[B, T_max]`
    long, pre-filled with `spec.pad_id` (pad frames are masked by `frame_mask` anyway).
    `build_dataloader` passes `cfg.conditioning` to the dataset and binds
    `collate_fn=functools.partial(collate, specs=cfg.conditioning)` (partial stays picklable for
    future `num_workers>0`).
- **cfm.py** — `interpolate` (OT-CFM x_t + v_target), `masked_mse`. `sigma=1e-4`.
  - **Off-path augmentation** (`perturb_off_path`, deferred toggle default OFF). For a
    Bernoulli(`path_aug_p`) subset of train samples, nudge `x_t` off the straight OT line by
    `std*t(1-t)*z` and re-aim the target to `v_target - std*t*z`, which preserves the endpoint
    (`x_tilde + (1-t)*v_hat == x_t + (1-t)*v_target`) so constant velocity from the perturbed point
    still lands on the path at `t=1` — teaching the field a **restoring component** off the line
    (the v1 diagnosis found none). Two config keys: `path_aug_p` (fraction perturbed, 0 = off) and
    `path_aug_std` (noise scale; effective displacement std is `path_aug_std*t(1-t) <= std/4`). Unlike
    simply raising `sigma` (whose noise is *not* re-aimed), the re-aimed target here teaches
    restoration. `masked_mse_per_sample` backs the per-`t` eval breakdown below.
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
  `WANDB_API_KEY` is set; `--no-wandb` opts out. The train step applies `perturb_off_path` when
  `cfg.path_aug_p > 0`. `evaluate()` reports velocity MSE over a 10-point `T_GRID` (round-robin per
  sample) both **on-path** (`val/velocity_mse`) and **off-path** (`val/offpath_velocity_mse`, a
  fixed-seed `OFFPATH_EVAL_STD` perturbation — a restoration probe comparable across runs), plus
  per-`t` curves.
- **infer.py** — `build_prior_mel`, `build_technique_frames`, `integrate` (Euler), `generate_mel`
  (chunk+crossfade), frozen BigVGAN to wav. `main()` loads the checkpoint **first** and rebuilds the
  model from its embedded config (`cfg_from_dict`), so pre-conditioning checkpoints load unchanged.
  - **build_technique_frames(midi, n_frames, technique)** — `[T]` uint8 technique-id track for a
    score: every note gets the requested `technique`, gaps/lead-in/tail get `rest`, built from the
    same `note_groups_from_midi` (offset 0.0) + `expand_to_frames` as the dataset labels.
  - **--technique** CLI arg (`choices=TECHNIQUE_CLASSES[:5]`, default `normal`) fills every note
    ("rest" auto-fills gaps; per-note control is a future extension). `main()` assembles
    `cond_frames` generically over `cfg.conditioning` (technique builder wired; other signals raise
    `NotImplementedError`). `integrate(..., cond=None)` and `generate_mel(..., cond_frames=None)`
    thread the fixed conditioning through the ODE — each track is uploaded once and sliced per chunk;
    a length != prior-mel-frames mismatch is fatal.
- **eval/copy_synthesis.py** — step-0 vocoder-ceiling sanity (run first); vocodes GT audio directly,
  builds no model, so it needs no conditioning handling. **eval/metrics.py** — recon/transport MSE,
  MCD, Delta-mel plots. Rebuilds the model from the checkpoint's embedded config (`cfg_from_dict`);
  per held-out recording it loads each conditioning spec's `data/<dir>/<base>.npy`, skips the
  recording (same skip style as a missing prior/target) if any track is absent, and passes
  `cond_frames=` through `generate_mel`.

## Run

```bash
PY="C:/Users/archi/Miniconda3/envs/ai-violin/python.exe"   # the ai-violin env
"$PY" -m DataSynthesizer.build_prior --limit 4   # smoke: regenerate 4 prior mels
"$PY" -m DataSynthesizer.build_prior             # full pass (888 clips)
"$PY" -m Arioso.splits                       # held-out-piece split
"$PY" -m Arioso.clips --split train          # clip-pool stats
"$PY" -m Arioso.eval.copy_synthesis          # step 0: vocoder ceiling (run before training)
"$PY" -m Arioso.train --smoke                # short pipeline validation
"$PY" -m Arioso.train                        # full run (~1e5 steps; tune to convergence)
"$PY" -m Arioso.train --no-wandb             # ...same, without W&B logging
"$PY" -m Arioso.infer score.mid --ckpt Arioso/models/checkpoint_final.pt   # -> Arioso/samples/
"$PY" -m Arioso.infer score.mid --ckpt Arioso/models/checkpoint_final.pt --technique pizzicato
"$PY" -m Arioso.eval.metrics --ckpt Arioso/models/checkpoint_final.pt --plot delta.png
```

## Run configs (YAML)

Ablations no longer require editing code: a YAML run-config (see `Arioso/configs/`) describes a run
over the frozen `AriosoConfig` + a new `RunSettings` (in `run_config.py`). Two sections map 1:1 onto
those dataclasses — `model:` (the science, embedded in checkpoints) and `run:` (runtime/env knobs,
never embedded) — plus an optional `config_version:`.

```bash
"$PY" -m Arioso.train --config Arioso/configs/default.yaml        # loads to exactly the code defaults
"$PY" -m Arioso.train --config Arioso/configs/unconditioned.yaml  # a partial-override ablation
"$PY" -m Arioso.train --config Arioso/configs/default.yaml --batch-size 2   # CLI overrides YAML
"$PY" -m Arioso.train                                             # no --config -> exact old behavior
```

A run-config is a **partial override**: the loader starts from the code defaults and applies only
the keys present, so missing keys keep their defaults (this *is* the backward-compat mechanism) and
a mistyped key is a hard error listing the valid keys. `sr`/`hop`/`n_mels`/`in_ch` are **locked**
(the mel contract lives in `common.config`) and rejected in YAML. Each run writes its fully-resolved
config to `Arioso/models/config_<run-name>.yaml` (uploaded to W&B) for reproducibility.

**Precedence:** smoke clamps > CLI flags > YAML > code defaults.

**Backward-compatibility rules** (a YAML written today must still train correctly later):
1. New hyperparameter → add an `AriosoConfig` field **with a behavior-preserving default**; it is
   automatically a valid YAML key with no other change.
2. Never rename/remove/repurpose an existing YAML key. If unavoidable, add it to `_RENAMED` in
   `run_config.py` (the old key keeps working and warns).
3. New loss/solver/schedule → a new function + one registry entry (`LOSSES`/`SOLVERS`/`SCHEDULES`);
   never change an existing entry's semantics.
4. New conditioning signal → dataset-artifact dir (DataSynthesizer) + a `CondSpec` constant + a
   `SIGNALS` entry + an inference-time track builder in `infer.py`.

## Dependencies & caveats

- Env **ai-violin**: `torch` (2.6, CUDA), `numpy`, `scipy`, `librosa`, `soundfile`, `pretty_midi`,
  `matplotlib` (metrics plot), plus the vendored BigVGAN (pulled in via `common.vocoder`). The
  vocoder checkpoint downloads from HF Hub on first use (cached).
- **W&B logging** (optional): `pip install wandb`, then copy `.env.example` -> `.env` (gitignored)
  and set `WANDB_API_KEY` (or export it). Runs land in `archimedesli/Arioso`. A missing key/package
  is non-fatal — training just skips logging. Disable explicitly with `--no-wandb`.
- **Prior must come from `DataSynthesizer.synthesizePrior`** (`quantized_prior`) and mels from
  `common.vocoder.mel_spectrogram` — a re-implemented prior/mel would break train/inference identity
  or the vocoder.
- `data/` holds **training data only** (gitignored): `prior_mel_arioso/`, `onsets_arioso/`,
  `arioso_split.json`. Model **artifacts** live under the package and are gitignored:
  checkpoints in `Arioso/models/checkpoint_<step>.pt`, listening wavs in `Arioso/samples/`.
- **Per-frame technique conditioning is now baseline-ON** (`AriosoConfig.conditioning`): the model
  concatenates a per-frame violin-technique embedding to `[x_t, x_0]`. Ablate with
  `AriosoConfig(conditioning=())`. Its `data/technique_arioso/` id tracks are built by
  `DataSynthesizer.build_techniques`.
- **Out of scope** (deferred, toggles default OFF): body EQ / tilt / rolloff, vibrato/LFO prior,
  F0/voicing conditioning, CFG **conditioning dropout** (`cond_dropout`, the framework is wired but
  defaults 0.0), energy-balanced loss, vocoder fine-tuning, polyphony.
- Receptive field of the 20-block WaveNet is ~4093 frames (~47 s) — comfortably covers any clip.
