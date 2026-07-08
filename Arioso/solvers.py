"""Interchangeable ODE solvers for the flow-matching sampler (Section 9).

Each step advances ``x`` over ``[t, t + dt]`` given a velocity closure ``v(x, t)`` that
already captures the fixed prior conditioning, so a solver is a pure function of
``(v, x, t, dt)`` with no knowledge of the model, conditioning, or chunking. The registry's
NFE-per-step lets callers match function-eval budgets across methods (Euler 1/step; Heun and
midpoint 2/step), e.g. Euler@32 ~ Heun@16. ``euler`` is the spec baseline; the others are
higher-order toggles for the discretization sweep.
"""

from __future__ import annotations


def euler_step(v, x, t, dt):
    """Forward Euler (1 eval) — the baseline first-order step."""
    return x + dt * v(x, t)


def midpoint_step(v, x, t, dt):
    """Explicit midpoint / RK2 (2 evals): evaluate at the half-step, advance with it."""
    k1 = v(x, t)
    return x + dt * v(x + 0.5 * dt * k1, t + 0.5 * dt)


def heun_step(v, x, t, dt):
    """Heun / trapezoidal RK2 (2 evals): average the slope at both endpoints."""
    k1 = v(x, t)
    k2 = v(x + dt * k1, t + dt)
    return x + 0.5 * dt * (k1 + k2)


# name -> (step_fn, nfe_per_step). Unknown names raise KeyError at lookup (loud enough).
SOLVERS = {
    "euler": (euler_step, 1),
    "midpoint": (midpoint_step, 2),
    "heun": (heun_step, 2),
}
