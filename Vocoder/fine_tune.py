"""One-command driver to fine-tune NVIDIA BigVGAN from the pretrained
``nvidia/bigvgan_v2_44khz_128band_512x`` checkpoint on a prepared dataset dir.

This wraps the vendored trainer at ``external/BigVGAN/train.py`` with an 8 GB
VRAM-friendly config and a Windows-safe subprocess launch. It downloads the HF
checkpoint (generator + discriminator/optimizer + config), stages it into the
run dir, writes a fine-tuning config, and launches training as a child process
whose stdio is inherited so logs stream straight to the console.

Usage
-----
Smoke test (prep everything, run a handful of steps)::

    python -m Vocoder.fine_tune --smoke

Dry run (prep + print the exact command, do not launch)::

    python -m Vocoder.fine_tune --dry-run

Full fine-tune (10k added steps, defaults)::

    python -m Vocoder.fine_tune

Traps this driver works around
------------------------------
* **Silent-from-scratch trap.** ``train.py`` resumes only when BOTH
  ``bigvgan_generator.pt`` AND ``bigvgan_discriminator_optimizer.pt`` are present
  in ``--checkpoint_path`` (or step-prefixed ``g_``/``do_`` files). If either is
  missing it silently trains from random init. ``prepare_run`` guarantees both,
  and hard-fails otherwise.
* **Inert learning_rate.** On resume the lr is restored from the optimizer state
  in the ``do_`` file; the ``learning_rate`` in the config is ignored. We pre-read
  the ``do_`` checkpoint and print the restored ``steps``/``epoch`` plus the
  derived lr so there are no surprises.
* **PESQ on non-speech.** The trainer only skips PESQ when the dataset "name"
  (first path component of the wavs dir it is handed) contains ``"nonspeech"``.
  We launch with ``cwd`` = dataset root and pass the wavs dirs as the RELATIVE
  ``wavs_nonspeech`` so the name carries the marker; all other paths are absolute.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

# Pretrained checkpoint we fine-tune from.
HF_REPO_ID = "nvidia/bigvgan_v2_44khz_128band_512x"
HF_GENERATOR = "bigvgan_generator.pt"
HF_DISCRIMINATOR = "bigvgan_discriminator_optimizer.pt"
HF_CONFIG = "config.json"

# Absolute path to the vendored trainer and its dir (its own dir goes on sys.path).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
BIGVGAN_DIR = os.path.join(_REPO_ROOT, "external", "BigVGAN")
BIGVGAN_TRAIN = os.path.join(BIGVGAN_DIR, "train.py")

# Only these config keys are overridden for the fine-tune; mel/audio/lr untouched.
_CONFIG_OVERRIDE_KEYS = ("batch_size", "segment_size", "num_workers")

_DEP_CHECK_SKIP_ENV = "VOCODER_SKIP_DEP_CHECK"


# --------------------------------------------------------------------------- #
# Dependency pre-check
# --------------------------------------------------------------------------- #
def check_deps() -> None:
    """Fail fast if the trainer's third-party deps are missing.

    Honors ``VOCODER_SKIP_DEP_CHECK=1`` (documented escape hatch used while the
    conda env is still installing deps) — prints a warning and returns.
    """
    if os.environ.get(_DEP_CHECK_SKIP_ENV) == "1":
        print(f"[WARN] {_DEP_CHECK_SKIP_ENV}=1 set; skipping dependency pre-check.")
        return

    required = ["tensorboard", "pesq", "auraloss", "nnAudio"]
    missing: List[str] = []
    import importlib

    for mod in required:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)

    if missing:
        # pip package names match import names for these four.
        pip_cmd = f'"{sys.executable}" -m pip install ' + " ".join(missing)
        print("[ERROR] Missing training dependencies: " + ", ".join(missing))
        print("[ERROR] Install them with:\n    " + pip_cmd)
        sys.exit(1)


# --------------------------------------------------------------------------- #
# Dataset validation
# --------------------------------------------------------------------------- #
@dataclass
class DatasetPaths:
    root: str
    wavs_rel: str  # relative wavs dir name (carries the "nonspeech" marker)
    mels_dir: str  # absolute
    train_file: str  # absolute
    val_file: str  # absolute
    val_copysynth_file: str  # absolute
    n_train_lines: int


def _count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def validate_dataset(dataset_dir: str) -> DatasetPaths:
    """Check the prepared dataset layout and count training chunks."""
    root = os.path.abspath(dataset_dir)
    if not os.path.isdir(root):
        print(f"[ERROR] Dataset dir not found: {root}")
        sys.exit(1)

    wavs_rel = "wavs_nonspeech"
    wavs_dir = os.path.join(root, wavs_rel)
    mels_dir = os.path.join(root, "arioso_mel")
    train_file = os.path.join(root, "train.txt")
    val_file = os.path.join(root, "val.txt")
    val_copysynth_file = os.path.join(root, "val_copysynth.txt")

    required = {
        "wavs_nonspeech/": wavs_dir,
        "arioso_mel/": mels_dir,
        "train.txt": train_file,
        "val.txt": val_file,
        "val_copysynth.txt": val_copysynth_file,
    }
    missing = [name for name, p in required.items() if not os.path.exists(p)]
    if missing:
        print(f"[ERROR] Dataset dir {root} is missing: " + ", ".join(missing))
        sys.exit(1)

    n_train_lines = _count_lines(train_file)
    if n_train_lines == 0:
        print(f"[ERROR] train.txt is empty: {train_file}")
        sys.exit(1)

    return DatasetPaths(
        root=root,
        wavs_rel=wavs_rel,
        mels_dir=mels_dir,
        train_file=train_file,
        val_file=val_file,
        val_copysynth_file=val_copysynth_file,
        n_train_lines=n_train_lines,
    )


# --------------------------------------------------------------------------- #
# Checkpoint staging
# --------------------------------------------------------------------------- #
def _has_pattern_checkpoints(run_dir: str) -> bool:
    """True if step-prefixed ``g_????????`` files exist (our own resume)."""
    # Mirrors scan_checkpoint's glob so "pattern wins" matches the trainer.
    return len(glob.glob(os.path.join(run_dir, "g_" + "?" * 8))) > 0


def prepare_run(run_dir: str) -> str:
    """Create the run dir and guarantee both HF checkpoints + config are present.

    Downloads the three HF files (cached by ``huggingface_hub``; ~4 GB on first
    call) and COPIES the generator + discriminator into ``run_dir`` — unless
    step-prefixed ``g_????????`` files already exist there, in which case we are
    resuming our own run and the pattern checkpoints win in scan_checkpoint.

    Returns the absolute local path to the downloaded HF ``config.json``.
    """
    from huggingface_hub import hf_hub_download

    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    print(f"[INFO] Downloading HF checkpoint files from {HF_REPO_ID} (cached) ...")
    gen_src = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_GENERATOR)
    disc_src = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_DISCRIMINATOR)
    config_src = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_CONFIG)

    if _has_pattern_checkpoints(run_dir):
        print(
            "[INFO] Found step-prefixed g_ checkpoints in run dir; resuming our own "
            "run (skipping HF checkpoint copy)."
        )
    else:
        gen_dst = os.path.join(run_dir, HF_GENERATOR)
        disc_dst = os.path.join(run_dir, HF_DISCRIMINATOR)
        print(f"[INFO] Staging generator     -> {gen_dst}")
        shutil.copyfile(gen_src, gen_dst)
        print(f"[INFO] Staging discriminator -> {disc_dst}")
        shutil.copyfile(disc_src, disc_dst)

    # Hard-fail unless the trainer will be able to resume (pattern OR renamed).
    has_gen = _has_pattern_checkpoints(run_dir) or os.path.isfile(
        os.path.join(run_dir, HF_GENERATOR)
    )
    has_disc = len(glob.glob(os.path.join(run_dir, "do_" + "?" * 8))) > 0 or os.path.isfile(
        os.path.join(run_dir, HF_DISCRIMINATOR)
    )
    if not (has_gen and has_disc):
        print(
            "[ERROR] Run dir is missing a resumable checkpoint pair after prepare_run. "
            "The trainer would silently start FROM SCRATCH. Aborting.\n"
            f"        run_dir={run_dir}\n"
            f"        generator present={has_gen}, discriminator present={has_disc}"
        )
        sys.exit(1)

    return config_src


def _torch_load(path: str):
    """Load a checkpoint dict, tolerating torch 2.6's weights_only default.

    These checkpoints are plain dicts of tensors/ints, but weights_only=True can
    reject non-tensor objects; fall back to the trusted full unpickler.
    """
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Trusted vendored NVIDIA checkpoint; full unpickle is safe here.
        return torch.load(path, map_location="cpu", weights_only=False)


@dataclass
class CkptState:
    steps: int
    epoch: int
    lr: float


def read_do_checkpoint(run_dir: str, hf_config: dict) -> CkptState:
    """Pre-read the discriminator/optimizer checkpoint for steps/epoch/lr.

    Prefers the newest step-prefixed ``do_????????`` (our own resume), else the
    HF-renamed ``bigvgan_discriminator_optimizer.pt``.
    """
    run_dir = os.path.abspath(run_dir)
    pattern = sorted(glob.glob(os.path.join(run_dir, "do_" + "?" * 8)))
    if pattern:
        do_path = pattern[-1]
    else:
        do_path = os.path.join(run_dir, HF_DISCRIMINATOR)

    state = _torch_load(do_path)
    steps = int(state["steps"])
    epoch = int(state["epoch"])

    base_lr = float(hf_config.get("learning_rate", 1e-4))
    decay = float(hf_config.get("lr_decay", 0.9999996))
    lr = base_lr * (decay ** steps)

    print(
        f"[INFO] Restored from '{os.path.basename(do_path)}': "
        f"steps={steps}, epoch={epoch}, derived lr={lr:.3e} "
        f"(= {base_lr:g} * {decay:.10g}**{steps})"
    )
    return CkptState(steps=steps, epoch=epoch, lr=lr)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def write_ft_config(
    run_dir: str,
    hf_config_path: str,
    batch_size: int,
    segment_size: int,
    num_workers: int,
) -> str:
    """Write ``ft_config.json`` into the run dir.

    Loads the HF ``config.json`` and overrides ONLY batch_size, segment_size and
    num_workers. Mel/audio params and the (inert) learning_rate are untouched.
    Returns the absolute path to the written config.
    """
    with open(hf_config_path, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    config["batch_size"] = batch_size
    config["segment_size"] = segment_size
    config["num_workers"] = num_workers

    out_path = os.path.join(os.path.abspath(run_dir), "ft_config.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=4)
    print(f"[INFO] Wrote fine-tune config -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# Model export
# --------------------------------------------------------------------------- #
def export_newest_snapshot(run_dir: str, name: Optional[str] = None) -> Optional[str]:
    """Copy the newest ``g_`` snapshot + config into ``Vocoder/models/<name>/``.

    The snapshot is copied as ``bigvgan_generator.pt`` — the filename
    ``common.vocoder.load_vocoder`` prefers over step-prefixed snapshots — with the
    trainer-written ``config.json`` alongside as the authoritative record of the
    settings that produced it. ``name`` defaults to the run dir's basename.
    Returns the export dir, or None if the run dir holds no snapshot.

    Deliberately does NOT touch ``common.config.VOCODER_DIR``: promotion to the
    active default stays a manual step, gated on an eval_ab listen — or just select
    it in Studio's DEV drawer (exports under Vocoder/models/ are picked up
    automatically).
    """
    run_dir = os.path.abspath(run_dir)
    snaps = sorted(glob.glob(os.path.join(run_dir, "g_" + "?" * 8)))
    if not snaps:
        print("[WARN] No g_???????? snapshot in run dir; nothing to export.")
        return None
    snap = snaps[-1]

    config_src = os.path.join(run_dir, "config.json")
    if not os.path.isfile(config_src):
        config_src = os.path.join(run_dir, "ft_config.json")

    export_dir = os.path.join(_THIS_DIR, "models", name or os.path.basename(run_dir))
    gen_dst = os.path.join(export_dir, "bigvgan_generator.pt")
    if os.path.isfile(gen_dst):
        print(f"[WARN] Overwriting existing export {gen_dst} (pass --export-name to keep it).")
    os.makedirs(export_dir, exist_ok=True)
    print(f"[INFO] Exporting {os.path.basename(snap)} -> {gen_dst}")
    shutil.copyfile(snap, gen_dst)
    shutil.copyfile(config_src, os.path.join(export_dir, "config.json"))
    print(
        "[INFO] Export complete. To make it the active default, point "
        f"common/config.py::VOCODER_DIR at {export_dir} (after an eval_ab check) "
        "— or just select it in Studio's DEV drawer (exports under Vocoder/models/ "
        "are picked up automatically)."
    )
    return export_dir


# --------------------------------------------------------------------------- #
# Smoke helpers
# --------------------------------------------------------------------------- #
def _write_first_n_lines(src: str, dst: str, n: int) -> None:
    with open(src, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()][:n]
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write("\n".join(ln.rstrip("\n") for ln in lines) + "\n")


# --------------------------------------------------------------------------- #
# GPU memory monitor
# --------------------------------------------------------------------------- #
class GpuMemoryMonitor(threading.Thread):
    """Daemon thread polling ``nvidia-smi`` for peak GPU memory (MiB)."""

    def __init__(self, interval: float = 2.0) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_mib = 0
        # NB: must not be named `_stop` — that would shadow Thread._stop(),
        # which threading's join() machinery calls internally.
        self._stop_event = threading.Event()
        self._available = shutil.which("nvidia-smi") is not None

    def run(self) -> None:
        if not self._available:
            return
        while not self._stop_event.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                for line in out.strip().splitlines():
                    line = line.strip()
                    if line:
                        self.peak_mib = max(self.peak_mib, int(float(line)))
            except Exception:
                # nvidia-smi hiccup; keep polling.
                pass
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()

    def report(self) -> None:
        if not self._available:
            print("[INFO] nvidia-smi not available; peak GPU memory unknown.")
        else:
            print(f"[INFO] peak GPU memory: {self.peak_mib} MiB")


# --------------------------------------------------------------------------- #
# Command construction
# --------------------------------------------------------------------------- #
def build_command(
    ft_config: str,
    run_dir: str,
    ds: DatasetPaths,
    training_file: str,
    validation_file: str,
    unseen_validation_file: str,
    training_epochs: int,
    checkpoint_interval: int,
    validation_interval: int,
    summary_interval: int,
    eval_subsample: int,
    freeze_step: int,
) -> List[str]:
    """Assemble the argv for the trainer subprocess.

    wavs dirs are passed RELATIVE (``wavs_nonspeech``) so the dataset "name"
    carries the nonspeech marker (PESQ is skipped); everything else is absolute.
    ``--fine_tuning True`` is the literal string (argparse type=bool quirk).
    """
    return [
        sys.executable,
        os.path.abspath(BIGVGAN_TRAIN),
        "--config", os.path.abspath(ft_config),
        "--checkpoint_path", os.path.abspath(run_dir),
        "--input_wavs_dir", ds.wavs_rel,
        "--input_mels_dir", os.path.abspath(ds.mels_dir),
        "--input_training_file", os.path.abspath(training_file),
        "--input_validation_file", os.path.abspath(validation_file),
        "--list_input_unseen_wavs_dir", ds.wavs_rel,
        "--list_input_unseen_validation_file", os.path.abspath(unseen_validation_file),
        "--fine_tuning", "True",
        "--training_epochs", str(training_epochs),
        "--checkpoint_interval", str(checkpoint_interval),
        "--validation_interval", str(validation_interval),
        "--summary_interval", str(summary_interval),
        "--eval_subsample", str(eval_subsample),
        "--freeze_step", str(freeze_step),
    ]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
_DEFAULT_RUN_DIR = "Vocoder/runs/ft_v1"
_SMOKE_RUN_DIR = "Vocoder/runs/smoke"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m Vocoder.fine_tune",
        description="Fine-tune BigVGAN from the nvidia bigvgan_v2_44khz_128band_512x checkpoint.",
    )
    parser.add_argument("--dataset", default="Data/datasets/vocoder_ft")
    parser.add_argument("--run-dir", default=_DEFAULT_RUN_DIR)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--segment-size", type=int, default=16384)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--extra-steps", type=int, default=20000)
    parser.add_argument("--checkpoint-interval", type=int, default=2500)
    parser.add_argument("--validation-interval", type=int, default=1000)
    parser.add_argument("--summary-interval", type=int, default=25)
    parser.add_argument("--eval-subsample", type=int, default=3)
    parser.add_argument(
        "--freeze-extra",
        type=int,
        default=0,
        help="if >0, freeze D until ckpt_steps + freeze_extra; else freeze_step=0",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--export-name",
        default=None,
        help="name under Vocoder/models/ for the end-of-run export "
        "(default: run dir basename); use when extending a run dir in place "
        "so the previous export is not overwritten",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="skip the automatic end-of-run export to Vocoder/models/",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="pass --debug True to the trainer (skips ALL validation loops; "
        "diagnostic use, e.g. isolating training-only VRAM footprint)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    # --smoke redirects the default run dir (only if the user didn't override it).
    run_dir = args.run_dir
    if args.smoke and run_dir == _DEFAULT_RUN_DIR:
        run_dir = _SMOKE_RUN_DIR
    run_dir = os.path.abspath(run_dir)

    check_deps()
    ds = validate_dataset(args.dataset)

    # Stage HF checkpoints + config into the run dir.
    hf_config_path = prepare_run(run_dir)
    with open(hf_config_path, "r", encoding="utf-8") as fh:
        hf_config = json.load(fh)

    ckpt = read_do_checkpoint(run_dir, hf_config)

    # Resolve smoke vs. real inputs.
    if args.smoke:
        num_workers = 0
        training_file = os.path.join(run_dir, "train_smoke.txt")
        validation_file = os.path.join(run_dir, "val_smoke.txt")
        unseen_validation_file = os.path.join(run_dir, "val_copysynth_smoke.txt")
        _write_first_n_lines(ds.train_file, training_file, 4)
        _write_first_n_lines(ds.val_file, validation_file, 2)
        _write_first_n_lines(ds.val_copysynth_file, unseen_validation_file, 2)
        n_train_used = _count_lines(training_file)
    else:
        num_workers = args.num_workers
        training_file = ds.train_file
        validation_file = ds.val_file
        unseen_validation_file = ds.val_copysynth_file
        n_train_used = ds.n_train_lines

    steps_per_epoch = n_train_used // max(1, args.batch_size)

    if args.smoke:
        training_epochs = ckpt.epoch + 2
    else:
        training_epochs = ckpt.epoch + math.ceil(
            args.extra_steps / max(1, steps_per_epoch)
        )

    added_steps = max(0, training_epochs - ckpt.epoch) * steps_per_epoch
    freeze_step = (ckpt.steps + args.freeze_extra) if args.freeze_extra > 0 else 0

    # ft_config.json (batch_size / segment_size / num_workers overrides only).
    ft_config = write_ft_config(
        run_dir,
        hf_config_path,
        batch_size=args.batch_size,
        segment_size=args.segment_size,
        num_workers=num_workers,
    )

    cmd = build_command(
        ft_config=ft_config,
        run_dir=run_dir,
        ds=ds,
        training_file=training_file,
        validation_file=validation_file,
        unseen_validation_file=unseen_validation_file,
        training_epochs=training_epochs,
        checkpoint_interval=args.checkpoint_interval,
        validation_interval=args.validation_interval,
        summary_interval=args.summary_interval,
        eval_subsample=args.eval_subsample,
        freeze_step=freeze_step,
    )
    if args.smoke:
        # Per-step timing visibility (default stdout_interval of 5 would print
        # nothing over a handful of smoke steps since steps resume at ~5M).
        cmd += ["--stdout_interval", "1"]
    if args.no_validation:
        cmd += ["--debug", "True"]

    # One-block summary.
    print("=" * 70)
    print("BigVGAN fine-tune" + ("  [SMOKE]" if args.smoke else ""))
    print(f"  dataset          : {ds.root}")
    print(f"  run dir          : {run_dir}")
    print(f"  batch / segment  : {args.batch_size} / {args.segment_size}")
    print(f"  n train chunks   : {n_train_used}")
    print(f"  steps / epoch    : {steps_per_epoch}")
    print(f"  ckpt steps/epoch : {ckpt.steps} / {ckpt.epoch}")
    print(f"  target epochs    : {training_epochs}")
    print(f"  est. added steps : {added_steps}")
    print(f"  freeze_step      : {freeze_step}")
    print("=" * 70)

    if args.dry_run:
        print("[DRY-RUN] Would launch (cwd=%s):" % ds.root)
        print("    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\")
        print("    " + " \\\n        ".join(cmd))
        return 0

    # Launch the trainer as a child; inherit stdio so logs stream through.
    child_env = dict(os.environ)
    child_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    monitor = GpuMemoryMonitor(interval=2.0)
    monitor.start()
    print(f"[INFO] Launching trainer (cwd={ds.root}) ...")
    try:
        proc = subprocess.Popen(cmd, cwd=ds.root, env=child_env)
        returncode = proc.wait()
    finally:
        monitor.stop()
        monitor.join(timeout=5.0)
        monitor.report()

    print(f"[INFO] Trainer exited with code {returncode}.")

    # Auto-export the newest snapshot so the fine-tuned weights land in
    # Vocoder/models/ (never on smoke or failed runs).
    if returncode == 0 and not args.smoke and not args.no_export:
        export_newest_snapshot(run_dir, args.export_name)

    return returncode


if __name__ == "__main__":
    sys.exit(main())
