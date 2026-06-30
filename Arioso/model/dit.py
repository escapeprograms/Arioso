"""DiT stack — 3 pre-norm transformer blocks with AdaLN-Zero + RoPE (Section 6.3).

Operates ``[B, T, hidden]`` (hidden = 384). Each block: AdaLN-Zero modulation from the flow-time
conditioning vector ``c`` (a **zero-initialized** Linear so every block starts as identity, which
keeps deep AdaLN training stable), multi-head self-attention with RoPE on q,k (6 heads x 64) that
respects the frame mask as a key-padding mask, and a GELU FFN (dim 1536). Both sub-layers are
gated residuals scaled by the AdaLN alpha params.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AriosoConfig


def _rope_tables(t: int, head_dim: int, base: float, device, dtype):
    """Return (cos, sin) of shape ``[T, head_dim]`` for rotary position embedding."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(t, device=device).float()
    freqs = torch.outer(pos, inv_freq)                 # [T, half]
    emb = torch.cat([freqs, freqs], dim=-1)            # [T, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, heads, T, head_dim]; cos/sin: [T, head_dim].
    return x * cos[None, None] + _rotate_half(x) * sin[None, None]


class RoPESelfAttention(nn.Module):
    """Multi-head self-attention with RoPE on q,k and a key-padding mask."""

    def __init__(self, hidden: int, heads: int, head_dim: int, base: float):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.base = base
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(hidden, 3 * heads * head_dim)
        self.out = nn.Linear(heads * head_dim, hidden)

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor | None) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)           # each [B, heads, T, head_dim]
        cos, sin = _rope_tables(t, self.head_dim, self.base, x.device, x.dtype)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)

        attn_mask = None
        if frame_mask is not None:
            # SDPA boolean mask: True = PARTICIPATE (inverse of the old pad mask). frame_mask is
            # [B, T] (1 real / 0 pad); broadcast over heads + query dim to forbid pad keys.
            attn_mask = (frame_mask != 0)[:, None, None, :]          # [B, 1, 1, T] bool
        # SDPA's default scale is head_dim**-0.5 (== self.scale); its mem-efficient kernel avoids
        # materializing the [B, heads, T, T] scores, the main fragmentation source under varying T.
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # [B, heads, T, head_dim]
        h = h.transpose(1, 2).reshape(b, t, self.heads * self.head_dim)
        return self.out(h)


class DiTBlock(nn.Module):
    """One AdaLN-Zero + RoPE transformer block."""

    def __init__(self, cfg: AriosoConfig):
        super().__init__()
        h = cfg.hidden
        self.norm1 = nn.LayerNorm(h, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(h, elementwise_affine=False)
        self.attn = RoPESelfAttention(h, cfg.dit_heads, cfg.dit_head_dim, cfg.rope_base)
        self.ffn = nn.Sequential(
            nn.Linear(h, cfg.dit_ffn), nn.GELU(), nn.Linear(cfg.dit_ffn, h),
        )
        # Zero-init AdaLN modulation: each block starts as identity. Do not remove.
        self.adaLN = nn.Linear(h, 6 * h)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                frame_mask: torch.Tensor | None) -> torch.Tensor:
        g1, b1, a1, g2, b2, a2 = self.adaLN(c).chunk(6, dim=-1)   # each [B, hidden]
        g1, b1, a1 = g1[:, None], b1[:, None], a1[:, None]        # broadcast over T
        g2, b2, a2 = g2[:, None], b2[:, None], a2[:, None]
        h = self.norm1(x) * (1 + g1) + b1
        x = x + a1 * self.attn(h, frame_mask)
        h = self.norm2(x) * (1 + g2) + b2
        x = x + a2 * self.ffn(h)
        return x


class DiTStack(nn.Module):
    """The conditioning MLP + 3 DiT blocks. ``[B, T, hidden]`` -> ``[B, T, hidden]``."""

    def __init__(self, cfg: AriosoConfig):
        super().__init__()
        self.cond = nn.Sequential(
            nn.Linear(cfg.t_emb_dim, cfg.hidden), nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
        )
        self.blocks = nn.ModuleList([DiTBlock(cfg) for _ in range(cfg.dit_blocks)])

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                frame_mask: torch.Tensor | None) -> torch.Tensor:
        c = self.cond(t_emb)
        for block in self.blocks:
            x = block(x, c, frame_mask)
        return x
