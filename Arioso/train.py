"""Arioso OT-CFM training loop (Section 8).

AdamW (lr 2e-4, wd 0.01), linear warmup (4000 steps) -> cosine decay, grad-norm clip 1.0, bf16
mixed precision, and an EMA of the weights for inference (warmup schedule prevents the EMA from
retaining random init early). Checkpoints both raw and EMA weights. ``--smoke`` runs a short loop
on a tiny subset to validate the pipeline end-to-end.
"""

from __future__ import annotations

import argparse
import copy
import math
import os

import torch

from DataSynthesizer.config import DEFAULT_OUT

from .cfm import interpolate, masked_mse
from .config import CKPT_DIR, AriosoConfig
from .dataset import build_dataloader
from .model import AriosoModel


def lr_at(step: int, cfg: AriosoConfig) -> float:
    """Linear warmup over ``warmup_steps`` then cosine decay to 0 at ``total_steps``."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


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
def evaluate(model, loader, cfg, device, max_batches: int = 20) -> float:
    """Mean masked velocity MSE over up to ``max_batches`` val batches (fixed t=0.5 grid)."""
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x0, x1 = batch["x0"].to(device), batch["x1"].to(device)
        mask = batch["frame_mask"].to(device)
        t = torch.full((x0.shape[0],), 0.5, device=device)
        x_t, v_target = interpolate(x0, x1, t, cfg.sigma)
        v = model(x_t, x0, t, mask)
        total += masked_mse(v, v_target, mask).item()
        n += 1
    model.train()
    return total / max(1, n)


def train(out_dir: str, cfg: AriosoConfig, batch_size: int, steps: int | None,
          log_every: int, ckpt_every: int, val_every: int, device: str) -> None:
    torch.manual_seed(cfg.seed)
    total_steps = steps if steps is not None else cfg.total_steps
    ckpt_dir = os.path.join(out_dir, CKPT_DIR)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader = build_dataloader(out_dir, "train", batch_size, cfg, shuffle=True)
    val_loader = build_dataloader(out_dir, "val", batch_size, cfg, shuffle=False)
    print(f"train clips: {len(train_loader.dataset)}  val clips: {len(val_loader.dataset)}  "
          f"batches/epoch: {len(train_loader)}")

    model = AriosoModel(cfg).to(device)
    print(f"model params: {model.num_params() / 1e6:.1f} M")
    ema = EMA(model, cfg.ema_max)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = device.startswith("cuda")

    model.train()
    step = 0
    while step < total_steps:
        sampler = train_loader.batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(step)                              # reshuffle buckets
        for batch in train_loader:
            if step >= total_steps:
                break
            x0, x1 = batch["x0"].to(device), batch["x1"].to(device)
            mask = batch["frame_mask"].to(device)
            t = torch.rand(x0.shape[0], device=device)          # t ~ U(0, 1) per sample
            x_t, v_target = interpolate(x0, x1, t, cfg.sigma)

            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                v = model(x_t, x0, t, mask)
                loss = masked_mse(v, v_target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ema.update(model, step)

            if step % log_every == 0:
                print(f"step {step:7d}  loss {loss.item():.5f}  lr {lr_at(step, cfg):.2e}")
            if val_every and step > 0 and step % val_every == 0:
                print(f"  [val] velocity MSE: {evaluate(model, val_loader, cfg, device):.5f}")
            if ckpt_every and step > 0 and step % ckpt_every == 0:
                _save(ckpt_dir, model, ema, cfg, step)
            step += 1

    _save(ckpt_dir, model, ema, cfg, step, final=True)
    print(f"  [val] final velocity MSE: {evaluate(model, val_loader, cfg, device):.5f}")


def _save(ckpt_dir, model, ema, cfg, step, final=False) -> None:
    name = "final" if final else f"step{step}"
    path = os.path.join(ckpt_dir, f"arioso_{name}.pt")
    torch.save({"step": step, "model": model.state_dict(),
                "ema": ema.shadow.state_dict(), "cfg": vars(cfg)}, path)
    print(f"  saved {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=None, help="override total_steps")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true",
                    help="short run on a tiny subset to validate the pipeline")
    args = ap.parse_args()

    cfg = AriosoConfig()
    if args.smoke:
        steps = args.steps if args.steps is not None else 300
        train(args.out_dir, cfg, batch_size=min(args.batch_size, 4), steps=steps,
              log_every=25, ckpt_every=0, val_every=150, device=args.device)
    else:
        train(args.out_dir, cfg, batch_size=args.batch_size, steps=args.steps,
              log_every=args.log_every, ckpt_every=args.ckpt_every,
              val_every=args.val_every, device=args.device)


if __name__ == "__main__":
    main()
