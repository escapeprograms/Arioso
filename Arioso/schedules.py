"""Interchangeable LR annealing schedules for training (Section 8).

Each schedule is a pure decay-shape function of ``progress`` in [0, 1] (fraction of the
post-warmup run elapsed) returning a multiplier in [0, 1], with no knowledge of warmup,
step counts, or absolute LR values — those are composed once in ``lr_at``. The registry
mirrors ``Arioso.solvers.SOLVERS``: pick by name via ``AriosoConfig.lr_schedule``; unknown
names raise KeyError at lookup (loud enough). ``cosine`` is the spec baseline; the LR
anneals from ``cfg.lr`` down to ``cfg.lr_min`` (0 by default, i.e. decay to zero).
"""

from __future__ import annotations

import math


def constant(progress: float) -> float:
    """No annealing — hold the post-warmup LR flat at ``cfg.lr``."""
    return 1.0


def linear(progress: float) -> float:
    """Linear anneal from 1 to 0 over the post-warmup run."""
    return 1.0 - progress


def cosine(progress: float) -> float:
    """Cosine anneal (half-cosine from 1 to 0) — the baseline schedule."""
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# name -> decay-shape fn. Unknown names raise KeyError at lookup (loud enough).
SCHEDULES = {
    "constant": constant,
    "linear": linear,
    "cosine": cosine,
}


def lr_at(step: int, cfg) -> float:
    """LR for ``step``: linear warmup over ``warmup_steps``, then the ``cfg.lr_schedule``
    shape annealing from ``cfg.lr`` to ``cfg.lr_min`` at ``total_steps``."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    decay = SCHEDULES[cfg.lr_schedule](progress)
    return cfg.lr_min + (cfg.lr - cfg.lr_min) * decay
