"""Non-causal WaveNet stack — 20 DiffSinger-style residual blocks (Section 6.2).

Operates channels-first ``[B, hidden, T]`` (hidden = 384). Each block: per-block time
conditioning added before a dilated conv, a gated activation, a residual 1x1 conv, and a
parallel skip 1x1 conv. The 20 dilations cycle ``[1, 2, ..., 512]`` twice (receptive field
~4093 frames, comfortably covering any clip). All 20 skip outputs are summed (skip-sum) and
projected to the DiT hidden dim.

Non-causal = the full sequence is available at every ODE step (no autoregression); "same"
padding (= dilation for kernel 3) keeps the frame count fixed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AriosoConfig


class WaveNetBlock(nn.Module):
    """One DiffSinger residual block (Section 6.2 steps 1-5)."""

    def __init__(self, hidden: int, t_emb_dim: int, kernel: int, dilation: int):
        super().__init__()
        self.time_proj = nn.Linear(t_emb_dim, hidden)            # per-block time conditioning
        pad = dilation * (kernel - 1) // 2                       # "same" padding (=dilation, k=3)
        self.dilated = nn.Conv1d(hidden, 2 * hidden, kernel, padding=pad, dilation=dilation)
        self.res = nn.Conv1d(hidden, hidden, 1)
        self.skip = nn.Conv1d(hidden, hidden, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, hidden, T]; t_emb: [B, t_emb_dim].
        h = x + self.time_proj(t_emb).unsqueeze(-1)             # add time before the conv/gate
        h = self.dilated(h)                                    # [B, 2*hidden, T]
        a, b = h.chunk(2, dim=1)
        h = torch.tanh(a) * torch.sigmoid(b)                   # gated activation -> [B, hidden, T]
        return x + self.res(h), self.skip(h)                   # (residual, skip)


class WaveNetStack(nn.Module):
    """The full 20-block stack. ``[B, hidden, T]`` -> ``[B, T, hidden]`` (transposed for DiT)."""

    def __init__(self, cfg: AriosoConfig):
        super().__init__()
        self.blocks = nn.ModuleList([
            WaveNetBlock(cfg.hidden, cfg.t_emb_dim, cfg.wn_kernel, d)
            for d in cfg.dilations
        ])
        self.skip_proj = nn.Conv1d(cfg.hidden, cfg.hidden, 1)   # skip-sum -> DiT hidden

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        skip_sum = 0.0
        for block in self.blocks:
            x, skip = block(x, t_emb)
            skip_sum = skip_sum + skip
        out = self.skip_proj(F.relu(skip_sum))                 # [B, hidden, T]
        return out.transpose(1, 2)                             # [B, T, hidden]
