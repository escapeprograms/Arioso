"""Training augmentations for the OT-CFM objective — path transforms and auxiliary losses.

The v1 diagnosis found the field has **no restoring component** off the OT line (exposure bias:
solver drift at inference is never corrected). This module houses the mitigations, kept extensible
along two axes so new ones plug in without touching ``train.py``'s step body:

  1. **Path transforms** (pre-forward, no model access): transform ``(x_t, v_target)`` *before* the
     model forward. ``perturb_off_path`` / ``PathAug`` is the current one — nudge ``x_t`` off the
     straight path and re-aim the target so constant velocity still lands on the endpoint. A future
     transform is a new callable + a field on ``Augmentor`` wired in ``build_augmentor``.
  2. **Auxiliary losses** (need the model): losses that call the model a second time on a derived
     input. ADR (below) is the current one — it drift-simulates from the model's own velocity and
     penalizes the drifted state's direction. A future aux loss is a new function taking ``model``
     + tensors and a weight field on ``Augmentor``.

``build_augmentor(cfg)`` is the single factory ``train.py`` calls; ``Augmentor`` bundles the active
transforms/losses so the step body stays a few readable lines.

**Anti-Drift Rectification (ADR)** — from *"Exposure Bias Can Alleviate Itself via Directional and
Frequency Rectification in Flow Matching"* (DEFAR, `arXiv 2606.28226
<https://arxiv.org/abs/2606.28226>`_). We take **ADR only, not the paper's Frequency Compensation**
(user decision). Paper Algorithm 1 mapped to Arioso conventions (``x_t = (1-(1-sigma)t) x0 + t x1``,
``v_target = x1 - (1-sigma) x0``, ``t=1`` is data, prior mel ``x0`` plays the paper's noise ``eps``):

  1. ``t0`` = the existing per-sample ``t ~ U(0,1)``; ``t1 = t0 + (1-t0) U(0,1)`` per sample (``t1 >= t0``).
  2. ``v_t0 = model(x_t0, x0, t0, ...)`` — **reuses the existing FM forward** (grads on). If off-path
     aug is active ``x_t0`` is the augmented input; one forward serves both FM and ADR (paper-faithful:
     the drift is simulated from the model's own ``v_t0``).
  3. Drift simulation, **gradient flows through** (paper — do NOT detach ``v_t0``): ``x_hat = x_t0 +
     (t1-t0) v_t0``.
  4. ``v_hat = model(x_hat, x0, t1, ...)`` — the **one extra forward** (no FC => no stop-grad forward
     at the true ``x_t1``).
  5. ADR anchor: ``v_adr = x1 - (1-sigma) x_hat`` (drifted state -> data). Degeneracy: if ``x_hat``
     equals the true interpolant ``x_t1`` then ``v_adr == (1-(1-sigma) t1) v_target`` — same direction
     as the FM target (see ``sample_t1`` / verification).
  6. ``L_ADR = mean_B || v_hat/||v_hat|| - v_adr/||v_adr|| ||^2`` — per-sample L2 norms and the
     squared-diff sum over **masked (real-frame) elements only**, in **fp32** (cast out of bf16
     autocast), ``eps=1e-8`` on the norms.

Total loss (assembled in ``train.py``): ``L = L_FM + adr_beta * L_ADR`` (``L_FM`` is the unchanged
``masked_mse``). ``adr_p`` gates ADR **per step** (not per sample — a per-sample gate would mean
ragged masking through two graphs); paper-faithful default is 1.0 (every step).

``cfm.py`` owns ``interpolate``/the losses and must NOT import from this module (no cycle).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import AriosoConfig


def perturb_off_path(x_t: torch.Tensor, v_target: torch.Tensor, t: torch.Tensor,
                     p: float, std: float,
                     generator: torch.Generator | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Off-path augmentation: perturb ``x_t`` off the OT line and re-aim the velocity target.

    For a per-sample Bernoulli(``p``) subset of the batch, draw ``z ~ N(0, I)`` (per element) and::

        x_tilde = x_t + std * t(1-t) * z          # nudge off the straight path
        v_hat   = v_target - std * t * z          # re-aim so it still lands on the endpoint

    The ``t(1-t)`` displacement scaling is zero at both ends and maximal mid-path (mimicking
    solver-drift accumulation), and keeps the re-aimed target bounded — avoiding the ``/(1-t)``
    singularity as ``t -> 1``. The endpoint is preserved exactly: ``x_tilde + (1-t)*v_hat ==
    x_t + (1-t)*v_target`` for every element, so constant velocity from the perturbed point still
    reaches the path endpoint at ``t=1``. This teaches the field a restoring component off the line.

    ``x_t``, ``v_target``: ``[B, 128, T]``; ``t`` is per-sample ``[B]``. ``generator`` seeds BOTH the
    ``randn`` and the Bernoulli keep-mask, so ``evaluate`` gets a deterministic metric (training
    passes ``None``). Returns ``(x_t, v_target)`` unchanged when ``p == 0`` or ``std == 0`` (fast
    path: no RNG draw, so a baseline run's numerics are byte-for-byte untouched).
    """
    if p == 0.0 or std == 0.0:
        return x_t, v_target
    b = x_t.shape[0]
    tb = t.view(-1, 1, 1)                                       # [B, 1, 1]
    z = torch.randn(x_t.shape, generator=generator, device=x_t.device, dtype=x_t.dtype)
    keep = (torch.rand(b, generator=generator, device=x_t.device) < p).view(-1, 1, 1)
    x_tilde = torch.where(keep, x_t + std * tb * (1.0 - tb) * z, x_t)
    v_hat = torch.where(keep, v_target - std * tb * z, v_target)
    return x_tilde, v_hat


# --- Extension point 1: path transforms (pre-forward, no model access) -----------

@dataclass(frozen=True)
class PathAug:
    """Off-path path transform on ``(x_t, v_target)`` before the model forward.

    Thin dataclass wrapper over ``perturb_off_path`` so ``Augmentor`` can carry it and the step body
    reads ``if aug.path_aug.active: x_t, v_target = aug.path_aug(x_t, v_target, t)``. Fields mirror
    the config knobs ``path_aug_p`` / ``path_aug_std``.
    """
    p: float
    std: float

    @property
    def active(self) -> bool:
        """True iff the transform actually perturbs — matches ``perturb_off_path``'s fast path (it
        no-ops when ``p == 0`` or ``std == 0``), so an inactive transform draws no RNG."""
        return self.p > 0.0 and self.std > 0.0

    def __call__(self, x_t: torch.Tensor, v_target: torch.Tensor, t: torch.Tensor,
                 generator: torch.Generator | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        return perturb_off_path(x_t, v_target, t, self.p, self.std, generator)


# --- Extension point 2: auxiliary losses (need the model) ------------------------

def sample_t1(t0: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """ADR drift-target time: ``t1 = t0 + (1-t0) * U(0,1)`` per sample, so ``t0 <= t1 <= 1``.

    ``t0`` is the existing per-sample ``t ~ U(0,1)`` ``[B]``; the return is ``[B]`` on the same device.
    ``generator`` seeds the draw (``evaluate`` passes a fixed-seed one; training passes ``None``).
    """
    u = torch.rand(t0.shape, generator=generator, device=t0.device)
    return t0 + (1.0 - t0) * u


def _masked_l2_normalize(v: torch.Tensor, frame_mask: torch.Tensor,
                         eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample masked L2 norm of ``v`` and the mask-broadcast, both in fp32.

    ``v``: ``[B, 128, T]``; ``frame_mask``: ``[B, T]`` (1 real / 0 pad). Returns ``(vf * m, norm)``
    where ``vf`` is ``v`` cast to fp32 with padded frames zeroed and ``norm`` is ``[B, 1, 1]`` — the
    per-sample L2 over real-frame elements only, clamped away from 0 by ``eps``. The caller divides
    ``vf * m`` by ``norm`` to get the masked unit direction (padded elements stay 0). Casting to fp32
    here makes the norm/normalization safe under the bf16 autocast the two forwards run in.
    """
    m = frame_mask[:, None, :].to(torch.float32)                 # [B, 1, T]
    vf = v.to(torch.float32) * m                                 # padded frames zeroed
    norm = torch.sqrt((vf ** 2).sum(dim=(1, 2), keepdim=True)).clamp_min(eps)   # [B, 1, 1]
    return vf, norm


def adr_per_sample(model, x_t: torch.Tensor, v_t: torch.Tensor, x0: torch.Tensor,
                   x1: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor,
                   mask: torch.Tensor, cond: dict | None, sigma: float) -> torch.Tensor:
    """Per-sample ADR loss ``[B]`` — steps 3-6 of the module math (drift-sim + directional penalty).

    ``x_t`` / ``v_t`` are the (possibly path-augmented) FM input and the model's velocity at ``t0``
    (**reused** — no re-forward at ``t0``); ``v_t``'s graph is kept so the drift-sim gradient flows
    (paper: do NOT detach). ``x0`` = prior mel, ``x1`` = target mel, ``mask`` = ``[B, T]`` frame mask,
    ``cond`` = the per-frame conditioning dict (or ``None``), ``sigma`` = OT-CFM constant.

    Steps: ``x_hat = x_t + (t1-t0) v_t`` (drift sim) -> ``v_hat = model(x_hat, x0, t1, ...)`` (the one
    extra forward) -> anchor ``v_adr = x1 - (1-sigma) x_hat`` -> ``|| v_hat/||v_hat|| -
    v_adr/||v_adr|| ||^2`` summed over masked elements, per sample. Returned as ``[B]`` so the batch
    mean (train) and per-``t`` accumulation (eval) share one implementation.
    """
    dt = (t1 - t0).view(-1, 1, 1)                                # [B, 1, 1]
    x_hat = x_t + dt * v_t                                        # drift simulation (grads flow)
    v_hat = model(x_hat, x0, t1, mask, cond=cond)                # the one extra forward at t1
    v_adr = x1 - (1.0 - sigma) * x_hat                           # drifted state -> data anchor

    vh, nh = _masked_l2_normalize(v_hat, mask)                   # masked fp32 dir + norm
    va, na = _masked_l2_normalize(v_adr, mask)
    diff = vh / nh - va / na                                     # [B, 128, T], padded elems == 0
    return (diff ** 2).sum(dim=(1, 2))                           # [B] masked squared-diff sum


# --- Factory + bundle ------------------------------------------------------------

@dataclass(frozen=True)
class Augmentor:
    """Bundle of the active augmentations for one run, built once by ``build_augmentor``.

    ``path_aug`` is the pre-forward transform (extension point 1); ``adr_beta`` / ``adr_p`` drive the
    ADR auxiliary loss (extension point 2). Future augmentations add a field here + wiring in
    ``build_augmentor``, keeping ``train.py``'s step body stable.
    """
    path_aug: PathAug
    adr_beta: float
    adr_p: float

    @property
    def adr_active(self) -> bool:
        """True iff ADR contributes to the loss (weight and gate both nonzero). When False the step
        body draws no ADR RNG, so a baseline run's numerics are untouched."""
        return self.adr_beta > 0.0 and self.adr_p > 0.0


def build_augmentor(cfg: AriosoConfig) -> Augmentor:
    """The single factory ``train.py`` calls — maps config knobs to the active ``Augmentor``.

    Path transforms (``path_aug_*``) and auxiliary losses (``adr_*``) both get wired here, so adding
    a new augmentation is one config field + one line in this factory (no step-body change).
    """
    return Augmentor(
        path_aug=PathAug(p=cfg.path_aug_p, std=cfg.path_aug_std),
        adr_beta=cfg.adr_beta,
        adr_p=cfg.adr_p,
    )
