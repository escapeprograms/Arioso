"""Per-frame categorical conditioning -> channels-first embedding stream (Section 6).

Arioso conditions the velocity field on per-frame categorical signals (technique is the baseline;
the framework is generic over ``cfg.conditioning``). Each signal is a ``[B, T]`` track of integer
class ids; this module embeds every configured signal and concatenates them into a single
``[B, cond_dim, T]`` tensor ready to sit alongside ``[x_t, x_0]`` before the input projection.

Each ``CondSpec`` gets an ``nn.Embedding(num_classes + 1, emb_dim)``: the extra final row (index
``num_classes``) is the "unknown" embedding used for **classifier-free-guidance dropout**. During
training, with probability ``cfg.cond_dropout``, a sample's ENTIRE conditioning (every spec at
once, one Bernoulli draw per sample) is replaced by that unknown row, teaching the model an
unconditioned mode. It is off by default (``cond_dropout == 0.0``) and never appears in the data.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import AriosoConfig


class ConditioningEncoder(nn.Module):
    """Embed per-frame categorical conditioning signals into ``[B, cond_dim, T]``.

    One ``nn.Embedding(spec.num_classes + 1, spec.emb_dim)`` per ``CondSpec`` (keyed by name in a
    ``ModuleDict``); the ``num_classes`` row is the CFG "unknown" embedding. ``forward`` takes a
    dict of ``[B, T]`` id tracks (keys must be exactly the spec names), applies the shared
    per-sample conditioning dropout in training, embeds each, and concatenates over the channel
    axis in spec order.
    """

    def __init__(self, cfg: AriosoConfig):
        super().__init__()
        self.specs = tuple(cfg.conditioning)
        self.cond_dropout = cfg.cond_dropout
        self.embeddings = nn.ModuleDict({
            spec.name: nn.Embedding(spec.num_classes + 1, spec.emb_dim)
            for spec in self.specs
        })

    def forward(self, cond: dict[str, torch.Tensor]) -> torch.Tensor:
        # Validate keys are exactly the configured spec names (fail loud on drift).
        expected = {spec.name for spec in self.specs}
        got = set(cond)
        if got != expected:
            raise ValueError(
                f"conditioning keys {sorted(got)} != configured spec names {sorted(expected)}")

        # One shared CFG-dropout mask per sample: when a sample is dropped, ALL of its specs are
        # replaced by their "unknown" row (index num_classes). Only active while training.
        drop = None
        if self.training and self.cond_dropout > 0.0:
            any_ids = cond[self.specs[0].name]
            drop = torch.rand(any_ids.shape[0], device=any_ids.device) < self.cond_dropout

        embs = []
        for spec in self.specs:
            ids = cond[spec.name].long()                      # [B, T]
            if drop is not None:
                unknown = torch.full_like(ids, spec.num_classes)
                ids = torch.where(drop[:, None], unknown, ids)
            embs.append(self.embeddings[spec.name](ids))      # [B, T, emb_dim]

        emb = torch.cat(embs, dim=-1)                         # [B, T, cond_dim]
        return emb.transpose(1, 2)                            # [B, cond_dim, T]
