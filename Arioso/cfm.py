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


def masked_mse(v: torch.Tensor, v_target: torch.Tensor,
               frame_mask: torch.Tensor) -> torch.Tensor:
    """Masked MSE of ``(v - v_target)^2`` over real frames only (Section 7 step 5).

    ``v``, ``v_target``: ``[B, 128, T]``; ``frame_mask``: ``[B, T]`` (1 real / 0 pad). The mean is
    over real frames x mel bins x batch; padded frames are excluded entirely.
    """
    m = frame_mask[:, None, :]                       # [B, 1, T]
    sq = (v - v_target) ** 2 * m
    denom = m.sum() * v.shape[1]                     # real frames * mel bins
    return sq.sum() / denom.clamp_min(1.0)
