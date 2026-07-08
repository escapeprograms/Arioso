"""OT-CFM interpolation + masked MSE objective (Section 7).

Optimal-Transport Conditional Flow Matching with a near-straight path between the prior mel
``x_0`` and the target mel ``x_1``. Constants: ``sigma = 1e-4``. Plain masked MSE on the
velocity — no energy-balanced reweighting in this baseline.
"""

from __future__ import annotations

import torch


def interpolate(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor,
                sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(x_t, v_target)`` for OT-CFM.

    ``x_t = (1 - (1 - sigma) t) x_0 + t x_1`` ; ``v_target = x_1 - (1 - sigma) x_0``.
    ``t`` is per-sample ``[B]``; mels are ``[B, 128, T]`` (t broadcasts over channels/frames).
    """
    tb = t.view(-1, 1, 1)
    x_t = (1.0 - (1.0 - sigma) * tb) * x0 + tb * x1
    v_target = x1 - (1.0 - sigma) * x0
    return x_t, v_target


def perturb_off_path(x_t: torch.Tensor, v_target: torch.Tensor, t: torch.Tensor,
                     p: float, std: float,
                     generator: torch.Generator | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Off-path augmentation: perturb ``x_t`` off the OT line and re-aim the velocity target.

    For a per-sample Bernoulli(``p``) subset of the batch, draw ``z ~ N(0, I)`` (per element) and::

        x_tilde = x_t + std * t(1-t) * z          # nudge off the straight path
        v_hat   = v_target - std * t * z          # re-aim so it still lands on the endpoint

    The ``t(1-t)`` displacement scaling is zero at both ends and maximal mid-path (mimicking
    solver-drift accumulation), and keeps the re-aimed target bounded — avoiding the ``/(1-t)``
    singularity as ``t -> 1``. The endpoint is preserved exactly: ``x_tilde + (1-t)*v_hat ==
    x_t + (1-t)*v_target`` for every element, so constant velocity from the perturbed point still
    reaches the path endpoint at ``t=1``. This teaches the field a restoring component off the line.

    ``x_t``, ``v_target``: ``[B, 128, T]``; ``t`` is per-sample ``[B]``. ``generator`` seeds BOTH the
    ``randn`` and the Bernoulli keep-mask, so ``evaluate`` gets a deterministic metric (training
    passes ``None``). Returns ``(x_t, v_target)`` unchanged when ``p == 0`` or ``std == 0`` (fast
    path: no RNG draw, so a baseline run's numerics are byte-for-byte untouched).
    """
    if p == 0.0 or std == 0.0:
        return x_t, v_target
    b = x_t.shape[0]
    tb = t.view(-1, 1, 1)                                       # [B, 1, 1]
    z = torch.randn(x_t.shape, generator=generator, device=x_t.device, dtype=x_t.dtype)
    keep = (torch.rand(b, generator=generator, device=x_t.device) < p).view(-1, 1, 1)
    x_tilde = torch.where(keep, x_t + std * tb * (1.0 - tb) * z, x_t)
    v_hat = torch.where(keep, v_target - std * tb * z, v_target)
    return x_tilde, v_hat


def masked_mse(v: torch.Tensor, v_target: torch.Tensor, frame_mask: torch.Tensor,
               batch: dict | None = None) -> torch.Tensor:
    """Masked MSE of ``(v - v_target)^2`` over real frames only (Section 7 step 5).

    ``v``, ``v_target``: ``[B, 128, T]``; ``frame_mask``: ``[B, T]`` (1 real / 0 pad). The mean is
    over real frames x mel bins x batch; padded frames are excluded entirely.

    ``batch`` is the full collated batch dict (unused here) — a forward-compatible hook so future
    losses (e.g. energy-weighted) can read extra tensors without a signature change. Existing 3-arg
    callers (``evaluate``) are unaffected.
    """
    m = frame_mask[:, None, :]                       # [B, 1, T]
    sq = (v - v_target) ** 2 * m
    denom = m.sum() * v.shape[1]                     # real frames * mel bins
    return sq.sum() / denom.clamp_min(1.0)


def masked_mse_per_sample(v: torch.Tensor, v_target: torch.Tensor,
                          frame_mask: torch.Tensor) -> torch.Tensor:
    """Per-sample masked velocity MSE — same reduction as ``masked_mse`` but returned as ``[B]``.

    Each entry is the mean of ``(v - v_target)^2`` over that sample's real frames x mel bins (padded
    frames excluded). Needed for the per-``t`` breakdown in ``evaluate``: accumulating per-sample
    values lets samples be grouped by their grid ``t`` before averaging. The batch mean of the return,
    weighted by each sample's real-frame count, reproduces ``masked_mse``'s global value.
    """
    m = frame_mask[:, None, :]                       # [B, 1, T]
    sq = ((v - v_target) ** 2 * m).sum(dim=(1, 2))   # [B] sum over mel bins x real frames
    denom = frame_mask.sum(dim=1) * v.shape[1]       # [B] real frames * mel bins
    return sq / denom.clamp_min(1.0)


# name -> loss fn ``(v, v_target, frame_mask, batch)``. Mirrors Arioso.solvers.SOLVERS /
# Arioso.schedules.SCHEDULES: pick by name via ``AriosoConfig.loss``; the run-config loader
# validates membership at load time. ``masked_mse`` is the spec baseline.
LOSSES = {"masked_mse": masked_mse}
