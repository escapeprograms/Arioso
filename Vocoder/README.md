# Vocoder — BigVGAN fine-tuning

Fine-tunes NVIDIA BigVGAN-v2 (`nvidia/bigvgan_v2_44khz_128band_512x`) on **Arioso's own
generated mels** so the vocoder matches what it actually sees at deployment, instead of
the natural mels it was pretrained on. Wraps the vendored trainer at
`external/BigVGAN/train.py` with an 8 GB VRAM-friendly, Windows-safe driver.

## The full workflow

All commands run from the repo root in the `ai-violin` env (call `python.exe` directly;
`conda` is not on PATH).

### 1. Build the fine-tuning corpus

```
python -m Vocoder.build_vocoder_dataset --limit 1   # smoke: one clip
python -m Vocoder.build_vocoder_dataset             # full build
```

Runs Arioso inference (default ckpt `Arioso/models/7-25-suzuki-3/checkpoint_final.pt`)
over the standard dataset root and lays out `Data/datasets/vocoder_ft/` exactly as
BigVGAN's `--fine_tuning` branch expects: `wavs_nonspeech/` (44.1 kHz GT audio,
`len(wav) == mel_T * 512` exactly), `arioso_mel/` (one `.npy` per wav stem), and
`train.txt` / `val.txt` / `val_copysynth.txt`. Two chunk kinds per clip:

* `_a` chunks — (Arioso mel, GT audio): the fine-tuning objective proper.
* `_g` chunks — (GT mel, GT audio): copy-synthesis anchors so fine-tuning doesn't
  drift natural-mel quality.

GT audio is the project's voiced-RMS −20 dBFS track copied verbatim (deliberately not
renormalized — the pair must stay self-consistent).

### 2. Smoke test, then fine-tune

```
python -m Vocoder.fine_tune --smoke      # few steps into Vocoder/runs/smoke
python -m Vocoder.fine_tune --dry-run    # print the exact trainer command
python -m Vocoder.fine_tune --batch-size 1 --segment-size 16384   # real run (8 GB card)
```

Note: add --run-dir Vocoder/runs/XXXX to start from scratch on a new run


> **VRAM warning (RTX 2000 Ada 8 GB, Windows/WDDM):** the CLI *defaults*
> (`--batch-size 2 --segment-size 32768`) **thrash on this card**. When the torch
> allocator pool nears ~7.9 GiB, WDDM silently pages GPU memory to system RAM — steps go
> from ~0.6 s to 100–230 s with the GPU showing "100 %" util, and **no OOM is raised**.
> The proven config is `--batch-size 1 --segment-size 16384` (fp32): ~0.6–0.8 s/step,
> peak ~7.5 GiB, survives full-chunk validation passes. Both shipped runs (ft_v1, ft_v2)
> used it.

Useful knobs: `--extra-steps` (default 10000 added steps), `--run-dir`
(default `Vocoder/runs/ft_v1`), `--checkpoint-interval` (2500), `--validation-interval`
(1000), `--no-validation` (maps to trainer `--debug True`; skips ALL validation loops —
diagnostic only), `--freeze-extra N` (freeze the discriminator for the first N added
steps).

**Resuming / extending:** just rerun with the same `--run-dir`. If step-prefixed
`g_XXXXXXXX` checkpoints exist there, the driver skips the HF copy and the trainer
resumes from the newest pair. ft_v2 was produced exactly this way — same run dir,
10k more steps.

### 3. A/B evaluate

```
python -m Vocoder.eval_ab --run-dir Vocoder/runs/ft_v1
```

Vocodes the held-out val chunks with both the pretrained baseline and the newest
fine-tuned snapshot; scores MR-STFT, mel L1, and MCD (lower = better) separately for
`val.txt` (Arioso mels — must improve) and `val_copysynth.txt` (copy-synthesis — must
not regress badly). Writes `metrics.csv` + paired
`__baseline/__finetuned/__gt` wavs for blind listening under the run dir.

### 4. Export & promote a checkpoint

1. **Export is automatic**: when a non-smoke run exits cleanly, the driver copies the
   newest `g_XXXXXXXX` snapshot to `Vocoder/models/<run-dir-basename>/bigvgan_generator.pt`
   with the run's `config.json` alongside it (`export_newest_snapshot` in
   `fine_tune.py`). Pass `--export-name <name>` when extending a run dir in place so
   the previous export isn't overwritten, or `--no-export` to skip. Manual export for
   an arbitrary snapshot is the same copy done by hand.
2. **Promotion stays manual** (gate on an `eval_ab` listen first): point
   `common/config.py::VOCODER_DIR` at the models dir (set `None` for stock HF).
   Resolution order everywhere is `load_vocoder(checkpoint_dir=...)` arg >
   `config.VOCODER_DIR` > HF hub; `checkpoint_dir="hf"` forces the pretrained baseline
   (that's what `eval_ab` uses for the A side). Studio's render cache auto-invalidates
   (cache key includes vocoder dirname + file size).

## Run history

| Run | Steps | Settings | Result |
|---|---|---|---|
| `models/ft_v1` | 5.00M → 5.01M (10k FT) | batch 1, seg 16384, fp32 | Arioso-val MCD 33.4→30.8, mel L1 −6.7 %; copy-synth MCD 8.8→11.0 (acceptable) |
| `models/ft_v2` (**active default**) | → 5.02M (20k FT total; `g_05020000`) | batch 1, seg 16384, fp32 (same run dir continued) | Arioso-val MCD 30.62 vs baseline 33.38; copy-synth 11.46 vs 8.79 |
| `models/ft_v3` | 5.00M → 5.01M (10k FT, fresh from HF baseline; `g_05010000`) | batch 1, seg 16384, fp32, `runs/ft_v3/` | trained 2026-08-06; not yet eval'd/promoted |

ft_v1 and ft_v2 both trained in `Vocoder/runs/ft_v1/`. Note: a run dir's `ft_config.json` is
rewritten on every driver launch with whatever CLI args were passed — it reflects the
*last launch attempt*, not necessarily what trained the exported checkpoints. The
authoritative record of what a shipped model used is `Vocoder/models/<name>/config.json`.

## Traps the driver handles (don't fight them)

* **Silent-from-scratch:** the trainer resumes only when BOTH `bigvgan_generator.pt`
  AND `bigvgan_discriminator_optimizer.pt` (or step-prefixed `g_`/`do_` files) are in
  the run dir — otherwise it silently trains from random init. `prepare_run` stages
  both from HF (~4 GB first download) and hard-fails if either is missing.
* **Inert `learning_rate`:** on resume, lr is restored from the optimizer state in the
  `do_` file (~1.35e-5 at 5M steps); the config value is ignored. The driver prints the
  derived lr up front.
* **PESQ on non-speech:** dodged by launching with `cwd` = dataset root and passing the
  wavs dir as the relative `wavs_nonspeech` so the dataset "name" carries the marker.
* **`--fine_tuning True`** is the literal string (trainer argparse `type=bool` quirk).
* Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and runs a background
  `nvidia-smi` peak-VRAM monitor, reported at exit.
