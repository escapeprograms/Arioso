"""Arioso OT-CFM training loop (Section 8).

AdamW (lr 2e-4, wd 0.01), linear warmup (4000 steps) -> LR annealing (``Arioso.schedules``,
cosine by default), grad-norm clip 1.0, bf16
mixed precision, and an EMA of the weights for inference (warmup schedule prevents the EMA from
retaining random init early). Checkpoints both raw and EMA weights. ``--smoke`` runs a short loop
on a tiny subset to validate the pipeline end-to-end.

``--resume`` continues a named run from the highest step checkpoint in its folder (raw + EMA weights
restored, step continued at N+1). Optimizer state is not checkpointed, so AdamW restarts cold — a
brief loss bump that washes out. A resume starts a fresh W&B run with the same name, continuing the
step axis so charts align.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import os
import re

import torch

from DataSynthesizer.config import DEFAULT_OUT

from .augment import adr_per_sample, build_augmentor, perturb_off_path, sample_t1
from .cfm import (LOSSES, interpolate, masked_mse_per_sample)
from .config import CKPT_DIR, WANDB_ENTITY, WANDB_PROJECT, AriosoConfig, cfg_to_dict
from .dataset import build_dataloader
from .model import AriosoModel
from .run_config import (RunSettings, auto_run_name, load_config, merge_cli_overrides,
                         save_resolved_config)
from .schedules import lr_at

# Validation t-grid: 10 evenly spaced points (0.05, 0.15, ..., 0.95). ``evaluate`` assigns t to
# each val sample round-robin over this grid (deterministic, one forward per batch), so the reported
# velocity MSE is a grid mean rather than a single-t snapshot — strictly more informative than the
# old t=0.5 metric (which was also contaminated by the known x1-leak at t=0.5).
T_GRID = tuple(round(0.05 + 0.1 * i, 2) for i in range(10))
# Off-path eval perturbation scale — fixed and independent of cfg.path_aug_std, so the off-path
# robustness metric is comparable across augmented and baseline (aug-off) runs. Calibrated to the
# measured inference drift (diagnose_field Test 4: drift ~ 1.34*t per-element RMS): the probe's
# peak displacement std/4 at t=0.5 matches the observed mid-path drift of ~0.67.
OFFPATH_EVAL_STD = 2.7


def _load_dotenv(path: str = ".env") -> None:
    """Minimal ``KEY=VALUE`` loader so the W&B API key can live in a gitignored ``.env``
    (see ``.env.example``). Existing env vars win; no dependency on python-dotenv."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip("'\""))


def _init_wandb(cfg: AriosoConfig, run_cfg: dict, name: str):
    """Return an initialized W&B run, or ``None`` if disabled/unavailable (training proceeds
    regardless). Reads ``WANDB_API_KEY`` from the environment / ``.env``. ``run_cfg`` is merged flat
    with ``cfg_to_dict(cfg)`` (flat dict preserved for dashboard continuity with existing runs)."""
    _load_dotenv()
    try:
        import wandb
    except ImportError:
        print("  [wandb] package not installed — skipping (pip install wandb)")
        return None
    if not os.environ.get("WANDB_API_KEY"):
        print("  [wandb] WANDB_API_KEY not set (.env or env var) — skipping")
        return None
    run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=name,
                     config={**cfg_to_dict(cfg), **run_cfg})
    print(f"  [wandb] logging to {run.url}")
    return run


class EMA:
    """Exponential moving average of model params with the spec's warmup decay schedule."""

    def __init__(self, model: torch.nn.Module, ema_max: float):
        self.ema_max = ema_max
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        decay = min(self.ema_max, (step + 1) / (step + 10))      # delta = min(0.9999,(s+1)/(s+10))
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(decay).add_(p.detach(), alpha=1 - decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


@torch.no_grad()
def evaluate(model, loader, cfg, device, max_batches: int = 20) -> dict[str, float]:
    """Velocity MSE over up to ``max_batches`` val batches, on a ``T_GRID`` and off-path.

    Three metrics over the same forwards:
    - **on-path** ``val/velocity_mse``: the model's velocity error on the true OT interpolant.
    - **off-path** ``val/offpath_velocity_mse``: the error on a ``perturb_off_path``-nudged input
      (``p=1.0``, ``std=OFFPATH_EVAL_STD``), a fixed-seed robustness probe measuring whether the
      field restores toward the path (low error) or not.
    - **ADR** ``val/adr_loss``: the directional drift-rectification loss (``adr_per_sample``), logged
      for ALL runs (not gated on ``adr_beta``) so it is comparable across ADR and baseline runs —
      one extra forward per batch, reusing the on-path ``v``. Its own fixed-seed generator (``gen_adr``,
      independent of the off-path ``gen``) keeps the off-path probe's RNG stream untouched.

    Each val sample is assigned a grid ``t`` **round-robin** over ``T_GRID`` using a global sample
    index across batches (deterministic; three forwards per batch — on-path, off-path, ADR).
    Per-sample errors (``masked_mse_per_sample``) are accumulated
    grouped by grid ``t``, so both the grid mean and the per-``t`` curves are reported. Returns a flat
    dict: the grid means plus ``val/velocity_mse_t/<t>`` / ``val/offpath_velocity_mse_t/<t>`` /
    ``val/adr_loss_t/<t>`` for each grid point (mean over the samples that landed on that ``t``).

    Determinism: the off-path perturbation uses a fixed-seed ``torch.Generator``. On CUDA a generator
    must be created with ``device=device`` (a CPU generator cannot drive CUDA ``randn``), so it is
    built here from ``device``.
    """
    model.eval()
    gen = torch.Generator(device=device).manual_seed(0)     # fixed-seed off-path perturbation
    gen_adr = torch.Generator(device=device).manual_seed(1)  # independent seed for the ADR t1 draw
    # Per-grid-t running (sum, count) of per-sample MSE, for on-path, off-path, and ADR.
    on_sum = [0.0] * len(T_GRID)
    on_cnt = [0] * len(T_GRID)
    off_sum = [0.0] * len(T_GRID)
    off_cnt = [0] * len(T_GRID)
    adr_sum = [0.0] * len(T_GRID)
    adr_cnt = [0] * len(T_GRID)
    sample_idx = 0                                          # global sample index -> round-robin t
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x0, x1 = batch["x0"].to(device), batch["x1"].to(device)
        mask = batch["frame_mask"].to(device)
        cond = {k: v.to(device) for k, v in batch["cond"].items()} if "cond" in batch else None
        b = x0.shape[0]
        grid_pos = [(sample_idx + j) % len(T_GRID) for j in range(b)]   # per-sample grid index
        sample_idx += b
        t = torch.tensor([T_GRID[g] for g in grid_pos], device=device)
        x_t, v_target = interpolate(x0, x1, t, cfg.sigma)

        # On-path forward.
        v = model(x_t, x0, t, mask, cond=cond)
        per = masked_mse_per_sample(v, v_target, mask)                  # [B]
        for j, g in enumerate(grid_pos):
            on_sum[g] += per[j].item()
            on_cnt[g] += 1

        # Off-path forward: perturb every sample (p=1.0) with the fixed-seed generator. x0 (model
        # input channel 2) and t are unchanged; only x_t/v_target move off the line.
        x_t_off, v_target_off = perturb_off_path(
            x_t, v_target, t, p=1.0, std=OFFPATH_EVAL_STD, generator=gen)
        v_off = model(x_t_off, x0, t, mask, cond=cond)
        per_off = masked_mse_per_sample(v_off, v_target_off, mask)      # [B]
        for j, g in enumerate(grid_pos):
            off_sum[g] += per_off[j].item()
            off_cnt[g] += 1

        # ADR probe: drift-simulate from the on-path v and score the drifted state's direction. One
        # extra forward at the per-sample t1 (fixed-seed gen_adr); reuses the on-path v (no re-forward
        # at t). Logged for all runs for cross-run comparability (see docstring).
        t1 = sample_t1(t, generator=gen_adr)
        per_adr = adr_per_sample(model, x_t, v, x0, x1, t, t1, mask, cond, cfg.sigma)  # [B]
        for j, g in enumerate(grid_pos):
            adr_sum[g] += per_adr[j].item()
            adr_cnt[g] += 1
    model.train()

    metrics: dict[str, float] = {}
    on_means, off_means, adr_means = [], [], []
    for g, tval in enumerate(T_GRID):
        if on_cnt[g]:
            m = on_sum[g] / on_cnt[g]
            metrics[f"val/velocity_mse_t/{tval}"] = m
            on_means.append(m)
        if off_cnt[g]:
            m = off_sum[g] / off_cnt[g]
            metrics[f"val/offpath_velocity_mse_t/{tval}"] = m
            off_means.append(m)
        if adr_cnt[g]:
            m = adr_sum[g] / adr_cnt[g]
            metrics[f"val/adr_loss_t/{tval}"] = m
            adr_means.append(m)
    # Grid mean = unweighted mean over grid points (each point ~equally sampled by round-robin).
    metrics["val/velocity_mse"] = sum(on_means) / max(1, len(on_means))
    metrics["val/offpath_velocity_mse"] = sum(off_means) / max(1, len(off_means))
    metrics["val/adr_loss"] = sum(adr_means) / max(1, len(adr_means))
    return metrics


def train(cfg: AriosoConfig, run_s: RunSettings) -> None:
    torch.manual_seed(cfg.seed)
    out_dir = run_s.out_dir
    batch_size = run_s.batch_size
    device = run_s.device
    log_every, ckpt_every, val_every = run_s.log_every, run_s.ckpt_every, run_s.val_every
    total_steps = run_s.steps if run_s.steps is not None else cfg.total_steps
    loss_fn = LOSSES[cfg.loss]                # velocity loss, chosen once (validated at load time)
    if run_s.resume and run_s.name is None:
        # Auto-generated run names embed a launch timestamp, so they can never match an existing
        # folder; resume must target a known folder by name.
        raise ValueError("--resume requires an explicit run name (--run-name or run.name in YAML)")
    # Each run gets its own subfolder under Arioso/models/ (named for the run) so its checkpoints
    # + resolved config stay grouped and don't collide across runs. Re-running the same run name
    # reuses the folder (step-named checkpoints overwrite), which is the intended behavior.
    run_name = run_s.name or auto_run_name(cfg)
    run_s = dataclasses.replace(run_s, name=run_name)   # record the resolved name in saved config
    ckpt_dir = os.path.join(CKPT_DIR, run_name)   # Arioso/models/<run_name>/ — model artifacts
    # Resolve the resume checkpoint before creating anything, so a mistyped run name fails clean
    # instead of leaving an empty run folder + resolved config behind.
    resume_ckpt = _latest_checkpoint(ckpt_dir) if run_s.resume else None
    if run_s.resume and resume_ckpt is None:
        raise FileNotFoundError(
            f"--resume: no checkpoint_step<N>.pt in {ckpt_dir} (only checkpoint_final.pt may exist "
            f"if the run finished, which is not resumable).")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Reproducibility: dump the fully-resolved config into the run folder, upload to W&B.
    run = (_init_wandb(cfg, {**dataclasses.asdict(run_s), "resolved_total_steps": total_steps},
                       run_name) if run_s.wandb else None)
    resolved_path = os.path.join(ckpt_dir, "config.yaml")
    # A resolved config reproduces the run from scratch, never implicitly resumes -> force resume=False.
    save_resolved_config(cfg, dataclasses.replace(run_s, resume=False), resolved_path)
    print(f"  resolved config -> {resolved_path}")
    if run:
        run.save(resolved_path, policy="now")

    train_loader = build_dataloader(out_dir, "train", batch_size, cfg, shuffle=True)
    val_loader = build_dataloader(out_dir, "val", batch_size, cfg, shuffle=False)
    print(f"train clips: {len(train_loader.dataset)}  val clips: {len(val_loader.dataset)}  "
          f"batches/epoch: {len(train_loader)}")

    model = AriosoModel(cfg).to(device)
    print(f"model params: {model.num_params() / 1e6:.1f} M")
    aug = build_augmentor(cfg)     # path transforms + ADR, resolved once (see Arioso.augment)
    ema = EMA(model, cfg.ema_max)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = device.startswith("cuda")

    step = _load_resume(resume_ckpt, model, ema, cfg, device) if resume_ckpt else 0
    if run_s.resume and step >= total_steps:
        print(f"  [resume] checkpoint step {step} is at/past total_steps ({total_steps}); the loop "
              f"will just save a final checkpoint and exit")

    model.train()
    while step < total_steps:
        sampler = train_loader.batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(step)                              # reshuffle buckets
        for batch in train_loader:
            if step >= total_steps:
                break
            x0, x1 = batch["x0"].to(device), batch["x1"].to(device)
            mask = batch["frame_mask"].to(device)
            cond = ({k: v.to(device) for k, v in batch["cond"].items()}
                    if "cond" in batch else None)
            t = torch.rand(x0.shape[0], device=device)          # t ~ U(0, 1) per sample
            # Interpolate + path-aug stay OUTSIDE autocast (byte-identical to the baseline stream).
            x_t, v_target = interpolate(x0, x1, t, cfg.sigma)
            if aug.path_aug.active:                              # off-path augmentation (default OFF)
                x_t, v_target = aug.path_aug(x_t, v_target, t)

            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                v = model(x_t, x0, t, mask, cond=cond)
                loss_fm = loss_fn(v, v_target, mask, batch)
                loss, l_adr = loss_fm, None
                # ADR (default OFF): one extra forward on the drift-simulated state (grads flow
                # through v — do NOT detach). adr_p==1.0 skips the gate RNG draw; when ADR is off no
                # RNG is drawn at all, so the baseline stream is untouched (byte-identity).
                if aug.adr_active and (aug.adr_p >= 1.0 or float(torch.rand(())) < aug.adr_p):
                    t1 = sample_t1(t)
                    l_adr = adr_per_sample(
                        model, x_t, v, x0, x1, t, t1, mask, cond, cfg.sigma).mean()
                    loss = loss_fm + aug.adr_beta * l_adr
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ema.update(model, step)

            if step % log_every == 0:
                cur_lr = lr_at(step, cfg)
                # train/loss stays the total (dashboard continuity); loss_fm / loss_adr break it out.
                logs = {"train/loss": loss.item(), "train/loss_fm": loss_fm.item(),
                        "train/lr": cur_lr}
                msg = f"step {step:7d}  loss {loss.item():.5f}  fm {loss_fm.item():.5f}"
                if l_adr is not None:
                    logs["train/loss_adr"] = l_adr.item()
                    msg += f"  adr {l_adr.item():.5f}"
                print(f"{msg}  lr {cur_lr:.2e}")
                if run:
                    run.log(logs, step=step)
            if val_every and step > 0 and step % val_every == 0:
                val = evaluate(model, val_loader, cfg, device)
                print(f"  [val] velocity MSE: {val['val/velocity_mse']:.5f}  "
                      f"off-path: {val['val/offpath_velocity_mse']:.5f}  "
                      f"adr: {val['val/adr_loss']:.5f}")
                if run:
                    run.log(val, step=step)
            if ckpt_every and step > 0 and step % ckpt_every == 0:
                _save(ckpt_dir, model, ema, cfg, step)
            step += 1

    _save(ckpt_dir, model, ema, cfg, step, final=True)
    final = evaluate(model, val_loader, cfg, device)
    print(f"  [val] final velocity MSE: {final['val/velocity_mse']:.5f}  "
          f"off-path: {final['val/offpath_velocity_mse']:.5f}  "
          f"adr: {final['val/adr_loss']:.5f}")
    if run:
        run.log(final, step=step)
        run.finish()


def _save(ckpt_dir, model, ema, cfg, step, final=False) -> None:
    name = "final" if final else f"step{step}"
    path = os.path.join(ckpt_dir, f"checkpoint_{name}.pt")
    torch.save({"step": step, "model": model.state_dict(),
                "ema": ema.shadow.state_dict(), "cfg": cfg_to_dict(cfg)}, path)
    print(f"  saved {path}")


def _latest_checkpoint(ckpt_dir: str) -> tuple[str, int] | None:
    """Return ``(path, N)`` for the highest-step ``checkpoint_step<N>.pt`` in ``ckpt_dir``, else None.

    ``checkpoint_final.pt`` is deliberately excluded: a finished run has nothing to resume.
    """
    if not os.path.isdir(ckpt_dir):
        return None
    best: tuple[str, int] | None = None
    for fname in os.listdir(ckpt_dir):
        m = re.fullmatch(r"checkpoint_step(\d+)\.pt", fname)
        if m and (best is None or int(m.group(1)) > best[1]):
            best = (os.path.join(ckpt_dir, fname), int(m.group(1)))
    return best


def _load_resume(resume_ckpt: tuple[str, int], model, ema, cfg, device) -> int:
    """Restore the ``(path, N)`` step checkpoint into ``model``/``ema``; return the step to continue at.

    Optimizer state is not checkpointed, so AdamW moments restart cold (a brief loss bump). A config
    drift between the checkpoint and the current run is warned (not hard-erroring — extending a run
    by raising total_steps is legitimate; a real architecture mismatch fails in load_state_dict).
    """
    path, n = resume_ckpt
    ckpt = torch.load(path, map_location=device)
    cur = cfg_to_dict(cfg)
    old = ckpt["cfg"]
    differ = sorted(k for k in set(cur) | set(old) if cur.get(k) != old.get(k))
    if differ:
        print("  [resume] WARNING: config differs from the checkpoint (continuing anyway):")
        for k in differ:
            print(f"    {k}: checkpoint={old.get(k)!r} -> current={cur.get(k)!r}")
    model.load_state_dict(ckpt["model"])
    ema.shadow.load_state_dict(ckpt["ema"])
    print(f"  [resume] loaded checkpoint_step{n}.pt -> continuing at step {ckpt['step'] + 1}")
    print("  [resume] optimizer state restarts cold (no opt state checkpointed; brief loss bump)")
    # step-N ckpt is saved after step N's opt.step()/ema.update() and before step += 1 -> resume N+1.
    return ckpt["step"] + 1


def main() -> None:
    _load_dotenv()   # load PYTORCH_CUDA_ALLOC_CONF / WANDB_API_KEY before any CUDA init
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # --config selects the YAML run-config; the run-level flags below default to None sentinels so
    # precedence is smoke clamps > CLI > YAML > code defaults (only flags actually passed override).
    ap.add_argument("--config", default=None,
                    help="YAML run-config (see Arioso/configs/); omit for code defaults")
    ap.add_argument("--out-dir", default=None,
                    help=f"data root (default: YAML/{DEFAULT_OUT})")
    ap.add_argument("--batch-size", type=int, default=None, help="default: YAML/8")
    ap.add_argument("--steps", type=int, default=None,
                    help="override total_steps (default: YAML/cfg.total_steps)")
    ap.add_argument("--log-every", type=int, default=None, help="default: YAML/50")
    ap.add_argument("--ckpt-every", type=int, default=None, help="default: YAML/5000")
    ap.add_argument("--val-every", type=int, default=None, help="default: YAML/2000")
    ap.add_argument("--device", default=None,
                    help="default: YAML, else cuda if available else cpu")
    ap.add_argument("--run-name", dest="name", default=None,
                    help="W&B run name (default: YAML, else auto_run_name(cfg))")
    ap.add_argument("--smoke", action="store_true",
                    help="short run on a tiny subset to validate the pipeline")
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None,
                    help=f"log to W&B ({WANDB_ENTITY}/{WANDB_PROJECT}); needs WANDB_API_KEY "
                         "in env or .env (default: YAML/on)")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None,
                    help="continue a named run from its highest step checkpoint (optimizer state "
                         "not restored; requires --run-name) (default: YAML/off)")
    args = ap.parse_args()

    # Resolve precedence: code defaults -> YAML -> CLI -> (device fallback) -> smoke clamps.
    cfg, run_s = load_config(args.config)
    run_s = merge_cli_overrides(run_s, args)
    if run_s.device is None:
        run_s = dataclasses.replace(
            run_s, device="cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        run_s = dataclasses.replace(
            run_s, steps=run_s.steps or 300, batch_size=min(run_s.batch_size, 4),
            log_every=25, ckpt_every=0, val_every=150)
    train(cfg, run_s)


if __name__ == "__main__":
    main()
