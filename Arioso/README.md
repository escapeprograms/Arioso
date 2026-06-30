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
   └ dataset.build_dataloader ─> length-bucketed batches + frame masks
        │
   train.py: x_t,v_target = cfm.interpolate(x0,x1,t); v = model(x_t,x0,t,mask); masked_mse
        │                                            (AdamW, warmup+cosine, bf16, EMA)
   infer.py: x0 -> Euler 16-32 steps -> mel -> frozen BigVGAN -> wav   (listening only)
   eval/: copy_synthesis (vocoder ceiling) · metrics (MSE/MCD/Delta-mel)
```

## Files

- **config.py** — `AriosoConfig` (one frozen dataclass: model + training + clip + infer hparams;
  spec-baseline defaults). Mel contract imported from `common.config`; **prior knobs live in
  `DataSynthesizer.config`** (the prior is a dataset artifact). Output-dir names: `SPLIT_FILE`,
  `CKPT_DIR`, `SAMPLES_DIR`; `PRIOR_MEL_DIR`/`ONSETS_DIR` re-exported from `DataSynthesizer.config`.
- **(prior + build_prior now live in `DataSynthesizer/`)** — `synthesizePrior.PriorSynth`
  (composable: pitch / source / envelope / body / leveler components) + `quantized_prior(...)` factory
  is the single prior source of truth, reused at inference; `DataSynthesizer.build_prior` is the
  one-time dataset pass writing `prior_mel_arioso/` + `onsets_arioso/`.
- **splits.py** — `make_split(out_dir, cfg)` held-out-**piece** split -> `arioso_split.json`.
- **clips.py** — `enumerate_clips(out_dir, basenames, cfg)` deterministic onset-aligned 5-10 s pool.
- **dataset.py** — `AriosoDataset` (mmap mel slices), `LengthBucketBatchSampler`, `collate`
  (frame masks), `build_dataloader`.
- **cfm.py** — `interpolate` (OT-CFM x_t + v_target), `masked_mse`. `sigma=1e-4`.
- **model/** — `timestep.py` (sinusoidal t_emb 256), `wavenet.py` (20 DiffSinger blocks, dilations
  `[1..512]x2`, gated act, skip-sum), `dit.py` (3 AdaLN-Zero **zero-init** + RoPE blocks, 6x64, FFN
  1536), `arioso.py` (input proj 256->384 -> wavenet -> dit -> head ->128).
- **train.py** — OT-CFM loop, AdamW(2e-4, wd 0.01), warmup 4000 -> cosine, grad-clip 1.0, bf16,
  EMA (`delta=min(0.9999,(s+1)/(s+10))`), raw+EMA checkpoints. `--smoke` for a short validation run.
  Logs loss/lr/val-MSE to W&B (`archimedesli/Arioso`) when `WANDB_API_KEY` is set; `--no-wandb` opts out.
- **infer.py** — `build_prior_mel`, `integrate` (Euler), `generate_mel` (chunk+crossfade), frozen
  BigVGAN to wav.
- **eval/copy_synthesis.py** — step-0 vocoder-ceiling sanity (run first). **eval/metrics.py** —
  recon/transport MSE, MCD, Delta-mel plots.

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
"$PY" -m Arioso.eval.metrics --ckpt Arioso/models/checkpoint_final.pt --plot delta.png
```

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
- **Out of scope** (deferred, toggles default OFF): body EQ / tilt / rolloff, vibrato/LFO prior,
  articulation/F0/voicing conditioning, CFG, energy-balanced loss, vocoder fine-tuning, polyphony.
- Receptive field of the 20-block WaveNet is ~4093 frames (~47 s) — comfortably covers any clip.
