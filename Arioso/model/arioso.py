"""The Arioso velocity field: input proj -> WaveNet -> DiT -> output head (Section 6).

Forward signature is the OT-CFM contract:
``forward(x_t, x_0, t, frame_mask, cond) -> v_theta``, all mels channels-first ``[B, 128, T]``.
``x_t`` (noisy/interpolated mel) and ``x_0`` (raw prior mel) are concatenated along channels
(=> 256). When conditioning is configured (``cfg.conditioning`` non-empty), the per-frame
conditioning embeddings ``[B, cond_dim, T]`` are concatenated too (=> 256 + cond_dim), projected
to hidden 384, run through the WaveNet then DiT, and read out to a 128-band predicted velocity.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import AriosoConfig
from .conditioning import ConditioningEncoder
from .dit import DiTStack
from .timestep import SinusoidalTimeEmbedding
from .wavenet import WaveNetStack


class AriosoModel(nn.Module):
    def __init__(self, cfg: AriosoConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or AriosoConfig()
        self.time_embed = SinusoidalTimeEmbedding(cfg.t_emb_dim)
        # Optional per-frame conditioning embedded and concatenated to [x_t, x_0].
        self.cond_enc = ConditioningEncoder(cfg) if cfg.conditioning else None
        self.in_proj = nn.Conv1d(cfg.in_ch + cfg.cond_dim, cfg.hidden, 1)  # [x_t, x_0(, cond)]->384
        self.wavenet = WaveNetStack(cfg)
        self.dit = DiTStack(cfg)
        self.out_norm = nn.LayerNorm(cfg.hidden)
        self.out_proj = nn.Linear(cfg.hidden, cfg.n_mels)         # 384 -> 128

    def forward(self, x_t: torch.Tensor, x_0: torch.Tensor, t: torch.Tensor,
                frame_mask: torch.Tensor | None = None,
                cond: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        # x_t, x_0: [B, 128, T]; t: [B]; frame_mask: [B, T] (1 real / 0 pad);
        # cond: {spec_name: [B, T] ids} iff the model was built with conditioning.
        if (cond is None) != (self.cond_enc is None):
            raise ValueError(
                "cond must be provided iff the model has conditioning "
                f"(cfg.conditioning={'set' if self.cond_enc is not None else 'empty'}, "
                f"cond={'given' if cond is not None else 'None'})")
        t_emb = self.time_embed(t)                                # [B, t_emb_dim]
        parts = [x_t, x_0]                                        # [B, 256, T]
        if self.cond_enc is not None:
            parts.append(self.cond_enc(cond))                    # + [B, cond_dim, T]
        h = torch.cat(parts, dim=1)                               # [B, 256(+cond_dim), T]
        h = self.in_proj(h)                                       # [B, 384, T]
        h = self.wavenet(h, t_emb)                                # [B, T, 384]
        h = self.dit(h, t_emb, frame_mask)                        # [B, T, 384]
        v = self.out_proj(self.out_norm(h))                       # [B, T, 128]
        return v.transpose(1, 2)                                  # [B, 128, T]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
