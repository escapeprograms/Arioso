"""Arioso model package: shared time embedding, WaveNet stack, DiT stack, full model.

The velocity field is a hybrid WaveNet (20 blocks) -> DiT (3 blocks), hidden dim 384 throughout
(Section 6). WaveNet runs channels-first ``[B, C, T]``; DiT runs ``[B, T, C]``; the stages
transpose between them. ``AriosoModel`` ties them together. ``ConditioningEncoder`` embeds the
per-frame categorical conditioning signals concatenated to ``[x_t, x_0]``.
"""

from .arioso import AriosoModel
from .conditioning import ConditioningEncoder

__all__ = ["AriosoModel", "ConditioningEncoder"]
