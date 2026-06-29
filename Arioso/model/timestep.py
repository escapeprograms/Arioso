"""Shared sinusoidal flow-time embedding (Section 6.0).

A single ``t_emb`` of dim 256 is computed once per forward from the per-sample flow-time
``t in [0, 1]`` and reused by both the WaveNet (per-block time conditioning) and the DiT
(AdaLN conditioning vector).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Map scalar flow-time ``t`` ``[B]`` -> sinusoidal embedding ``[B, dim]``.

    Standard transformer/diffusion sinusoidal features (concatenated sin & cos over a
    geometric range of frequencies). ``dim`` must be even.
    """

    def __init__(self, dim: int = 256, max_period: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "time embedding dim must be even"
        self.dim = dim
        half = dim // 2
        # freqs[k] = max_period^(-k/half); registered so it follows .to(device)/dtype.
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] (flow-time). Compute in float32 for precision under AMP, return float32.
        args = t.float()[:, None] * self.freqs[None, :]   # [B, half]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
