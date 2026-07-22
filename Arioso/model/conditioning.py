"""Per-frame conditioning -> channels-first embedding stream (Section 6).

Arioso conditions the velocity field on per-frame signals (the framework is generic over
``cfg.conditioning``). Each signal is a ``[B, T]`` per-frame track; this module maps every
configured signal to a ``[B, T, emb_dim]`` feature and concatenates them into a single
``[B, cond_dim, T]`` tensor ready to sit alongside ``[x_t, x_0]`` before the input projection.

Two signal kinds share the stream:

* **Categorical** (``CondSpec``): a ``[B, T]`` id track embedded by an
  ``nn.Embedding(num_classes + 1, emb_dim)``. The extra final row (index ``num_classes``) is the
  "unknown" embedding used for classifier-free guidance.
* **Sinusoidal boundary distance** (``BoundaryCondSpec`` -> :class:`BoundarySinusoid`): a ``[B, T]``
  distance-in-frames track (sentinel ``-1``) featurized with sin/cos over a geometric frequency
  band. Its "off" state is the exact **zero vector** (no unknown row needed).

**Classifier-free-guidance dropout.** During training, with probability ``cfg.cond_dropout``, a
sample's ENTIRE conditioning (every spec at once, one shared Bernoulli draw per sample) is dropped:
categorical specs swap to their unknown row, boundary specs are zeroed. This teaches the model an
unconditioned mode. It is off by default (``cond_dropout == 0.0``) and never appears in the data.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config import FRAME_RATE, AriosoConfig, BoundaryCondSpec, CondSpec


class BoundarySinusoid(nn.Module):
    """Featurize a per-frame note-boundary distance into ``[B, T, emb_dim]`` sin/cos features.

    Input ``d`` is a ``[B, T]`` **distance in frames** to the nearest boundary (an onset behind or an
    offset ahead, computed upstream in the dataset — nearest-boundary-wins is settled there, not
    here). A frame with no boundary in range carries the sentinel ``-1`` (any negative, or any
    distance ``>= window_frames``, is treated identically).

    **Compact support.** Only ``0 <= d < window_frames`` is active; every inactive frame maps to the
    exact **zero vector**, so there is no learned "off" state and CFG drop is just a zero (see
    :class:`ConditioningEncoder`). Active frames normalize ``u = d / window_frames`` into ``[0, 1)``
    and feed ``sin/cos`` of ``pi * u`` scaled by a geometric frequency band (``freqs``, one cycle at
    the low end up to ``window_frames`` cycles at the high end) — the flow-time-embedding recipe
    (see :class:`~Arioso.model.timestep.SinusoidalTimeEmbedding`) applied to a spatial distance.
    """

    def __init__(self, spec: BoundaryCondSpec, frame_rate: float = FRAME_RATE):
        super().__init__()
        assert spec.emb_dim % 2 == 0, "boundary emb_dim must be even (sin+cos halves)"
        # Compact-support width in mel frames (at least one frame so d==0 is always active).
        self.window_frames = max(1, int(round(spec.window_ms * frame_rate / 1000.0)))
        half = spec.emb_dim // 2
        # Geometric band from 1 cycle to <=window_frames cycles over the normalized [0,1) window;
        # non-persistent so it follows .to(device)/dtype but never bloats the checkpoint.
        freqs = torch.exp(torch.linspace(math.log(1.0),
                                         math.log(max(2.0, float(self.window_frames))), half))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        # d: [B, T] long distance-in-frames (-1 sentinel). Compute in float32 for AMP stability.
        active = (d >= 0) & (d < self.window_frames)          # [B, T] compact-support mask
        u = d.clamp(min=0).float() / self.window_frames       # [B, T] in [0, 1) where active
        args = math.pi * u.unsqueeze(-1) * self.freqs         # [B, T, half]
        feat = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, T, emb_dim]
        return feat * active.unsqueeze(-1)                    # zero vector outside the window


class ConditioningEncoder(nn.Module):
    """Embed per-frame conditioning signals (categorical + boundary) into ``[B, cond_dim, T]``.

    Categorical ``CondSpec``\\ s get one ``nn.Embedding(num_classes + 1, emb_dim)`` each (the
    ``num_classes`` row is the CFG "unknown" embedding); boundary ``BoundaryCondSpec``\\ s get one
    :class:`BoundarySinusoid` each. Both live in name-keyed ``ModuleDict``\\ s. ``forward`` takes a
    dict of ``[B, T]`` tracks (keys must be exactly the spec names), applies one shared per-sample
    CFG dropout in training (categorical -> unknown row, boundary -> zeroed), maps each signal, and
    concatenates over the channel axis in spec order.
    """

    def __init__(self, cfg: AriosoConfig):
        super().__init__()
        self.specs = tuple(cfg.conditioning)
        self.cond_dropout = cfg.cond_dropout
        self.embeddings = nn.ModuleDict({
            spec.name: nn.Embedding(spec.num_classes + 1, spec.emb_dim)
            for spec in self.specs if isinstance(spec, CondSpec)
        })
        self.boundaries = nn.ModuleDict({
            spec.name: BoundarySinusoid(spec, FRAME_RATE)
            for spec in self.specs if isinstance(spec, BoundaryCondSpec)
        })

    def forward(self, cond: dict[str, torch.Tensor]) -> torch.Tensor:
        # Validate keys are exactly the configured spec names (fail loud on drift).
        expected = {spec.name for spec in self.specs}
        got = set(cond)
        if got != expected:
            raise ValueError(
                f"conditioning keys {sorted(got)} != configured spec names {sorted(expected)}")

        # One shared CFG-dropout mask per sample: a dropped sample loses ALL of its specs at once
        # (categorical -> unknown row, boundary -> zero vector). Only active while training. Either
        # kind's track is [B, T] long, so specs[0] gives the batch size / device.
        drop = None
        if self.training and self.cond_dropout > 0.0:
            any_ids = cond[self.specs[0].name]
            drop = torch.rand(any_ids.shape[0], device=any_ids.device) < self.cond_dropout

        feats = []
        for spec in self.specs:
            if isinstance(spec, CondSpec):
                ids = cond[spec.name].long()                  # [B, T]
                if drop is not None:
                    unknown = torch.full_like(ids, spec.num_classes)
                    ids = torch.where(drop[:, None], unknown, ids)
                feats.append(self.embeddings[spec.name](ids))  # [B, T, emb_dim]
            else:
                feat = self.boundaries[spec.name](cond[spec.name].long())  # [B, T, emb_dim]
                if drop is not None:
                    feat = feat * (~drop)[:, None, None].to(feat.dtype)     # dropped -> zero vector
                feats.append(feat)

        emb = torch.cat(feats, dim=-1)                        # [B, T, cond_dim]
        return emb.transpose(1, 2)                            # [B, cond_dim, T]
