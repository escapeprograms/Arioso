"""Arioso inference: score -> prior -> Euler-integrated mel -> audio (Section 9).

The prior is built from the input score **exactly as in training** (``DataSynthesizer.synthesizePrior``
via ``quantized_prior``: band-limited saw + masked-RMS to the fixed target level + mel). Starting
from ``x = x_0`` at t=0, integrate the ODE ``x <- x + dt * v_theta(x, x_0, t, cond)`` with a
selectable solver (``Arioso.solvers.SOLVERS``: Euler / Heun / midpoint) over ``solver_steps`` steps
(no CFG). When the checkpoint's config has per-frame conditioning (technique by default), a
``[T]`` id track is built from the SAME score (``build_technique_frames``: ``--technique`` on every
note, rest in the gaps) and passed as fixed conditioning through the whole integration. The
architecture is reconstructed from the checkpoint's embedded config (``cfg_from_dict``), so
pre-conditioning checkpoints load unchanged. Long sequences are processed in overlapping chunks
with a linear crossfade. The mel is turned to audio with the **frozen** BigVGAN-v2 vocoder
(listening only — never a selection arbiter).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from common.vocoder import load_vocoder, vocode
from DataSynthesizer.config import (PRIOR_ANTI_ALIAS, PRIOR_ENVELOPE, PRIOR_LEVEL_MATCH,
                                    TARGET_RMS_DBFS, TECHNIQUE_CLASSES)
from DataSynthesizer.features import mel_for_training
from DataSynthesizer.synthesizePrior import quantized_prior
from DataSynthesizer.technique import expand_to_frames, note_groups_from_midi

from .config import SAMPLES_DIR, AriosoConfig, cfg_from_dict
from .model import AriosoModel
from .solvers import SOLVERS


def build_prior_mel(midi_path: str) -> np.ndarray:
    """Score -> prior mel ``[N_MELS, T]``, identical to the training-time prior (Section 9.1).

    Built by the same DataSynthesizer pipeline the dataset's ``prior_mel_arioso`` was, so the
    inference prior matches training. No GT-alignment shift here: the score's onsets *are* t=0.
    """
    synth = quantized_prior(anti_alias=PRIOR_ANTI_ALIAS, envelope=PRIOR_ENVELOPE,
                            level_match=PRIOR_LEVEL_MATCH, target_rms_dbfs=TARGET_RMS_DBFS)
    return mel_for_training(synth.render(midi_path))


def build_technique_frames(midi_path: str, n_frames: int, technique: str = "normal") -> np.ndarray:
    """``[T]`` uint8 technique-id track for a score at inference time.

    Every note is assigned the requested ``technique``; gaps / lead-in / tail become ``rest``.
    Built from the SAME note grouping as the dataset labels (``note_groups_from_midi`` with
    ``offset_s=0.0`` — matching ``build_prior_mel``'s no-shift convention, since the score's
    onsets are t=0) so the id grid lines up frame-for-frame with the prior mel.
    """
    groups = note_groups_from_midi(midi_path, offset_s=0.0, n_frames=n_frames)
    tech_ids = np.full(len(groups), TECHNIQUE_CLASSES.index(technique), dtype=np.uint8)
    return expand_to_frames(groups, tech_ids, n_frames)


@torch.no_grad()
def integrate(model: AriosoModel, x0: torch.Tensor, cfg: AriosoConfig,
              solver: str | None = None, steps: int | None = None,
              cond: dict | None = None) -> torch.Tensor:
    """Integrate the velocity field from t=0 (x=x0) to t=1 with the chosen solver.

    ``x0``: [1, 128, T]. ``solver``/``steps`` default to ``cfg.solver``/``cfg.solver_steps``;
    the prior ``x0`` (and the per-frame ``cond`` id tracks, when conditioning is on) is the fixed
    conditioning for the whole integration.
    """
    step_fn, _ = SOLVERS[solver or cfg.solver]
    steps = steps or cfg.solver_steps
    x = x0.clone()

    def v(x_in: torch.Tensor, t: float) -> torch.Tensor:        # captures fixed prior x0 + cond
        tt = torch.full((x_in.shape[0],), float(t), device=x_in.device)
        return model(x_in, x0, tt, cond=cond)

    dt = 1.0 / steps
    for i in range(steps):
        x = step_fn(v, x, i * dt, dt)
    return x


@torch.no_grad()
def generate_mel(model: AriosoModel, prior_mel: np.ndarray, cfg: AriosoConfig,
                 device: str, solver: str | None = None, steps: int | None = None,
                 cond_frames: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Run the ODE over the whole prior mel, chunking long sequences with a linear crossfade.

    ``cond_frames`` (required iff the model has conditioning) maps each spec name to a ``[T]``
    id array on the prior mel's frame grid; each is uploaded once as a ``[1, T]`` long tensor and
    sliced to match every chunk. Fails loud if any track's length != prior mel frame count.
    """
    x0 = torch.from_numpy(prior_mel[None]).float().to(device)   # [1, 128, T]
    t_total = x0.shape[-1]
    chunk, overlap = cfg.chunk_frames, cfg.overlap_frames

    cond_t = None
    if cond_frames is not None:
        cond_t = {}
        for name, arr in cond_frames.items():
            if arr.shape[-1] != t_total:
                raise ValueError(
                    f"conditioning track {name!r} length {arr.shape[-1]} != prior mel frames "
                    f"{t_total}")
            cond_t[name] = torch.from_numpy(np.ascontiguousarray(arr)).long()[None].to(device)

    def cond_slice(start: int, end: int) -> dict | None:
        return None if cond_t is None else {k: v[:, start:end] for k, v in cond_t.items()}

    if t_total <= chunk:
        return integrate(model, x0, cfg, solver, steps,
                         cond=cond_slice(0, t_total))[0].cpu().numpy()

    out = np.zeros((prior_mel.shape[0], t_total), dtype=np.float32)
    weight = np.zeros(t_total, dtype=np.float32)
    step = chunk - overlap
    for start in range(0, t_total, step):
        end = min(start + chunk, t_total)
        seg = integrate(model, x0[:, :, start:end], cfg, solver, steps,
                        cond=cond_slice(start, end))[0].cpu().numpy()
        w = np.ones(end - start, dtype=np.float32)
        if start > 0:                                            # fade-in over the overlap
            w[:overlap] = np.linspace(0.0, 1.0, overlap, endpoint=False)
        out[:, start:end] += seg * w
        weight[start:end] += w
        if end == t_total:
            break
    return out / np.maximum(weight, 1e-6)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("midi", help="input score (.mid)")
    ap.add_argument("-o", "--out", default=os.path.join(SAMPLES_DIR, "arioso_out.wav"))
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt (uses EMA weights)")
    ap.add_argument("--weights", choices=("ema", "model"), default="ema")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--solver", choices=tuple(SOLVERS), default=None,
                    help="integration method (default: config's 'euler')")
    ap.add_argument("--steps", type=int, default=None,
                    help="solver steps; NFE = steps * nfe_per_step (default: config's 24)")
    ap.add_argument("--technique", choices=TECHNIQUE_CLASSES[:5], default="normal",
                    help="technique conditioning applied to EVERY note ('rest' auto-fills gaps); "
                         "per-note control is a future extension (only used if the checkpoint's "
                         "config has technique conditioning)")
    ap.add_argument("--save-mel", help="optional .npy path for the generated mel")
    args = ap.parse_args()

    # Load the checkpoint FIRST so the model is built from ITS config: pre-conditioning ckpts
    # reconstruct the exact old 256-ch architecture (conditioning=()), conditioned ckpts add cond.
    ckpt = torch.load(args.ckpt, map_location=args.device)
    cfg = cfg_from_dict(ckpt.get("cfg") or {})
    model = AriosoModel(cfg).to(args.device)
    model.load_state_dict(ckpt[args.weights])
    model.eval()

    prior_mel = build_prior_mel(args.midi)
    print(f"prior mel: {prior_mel.shape}  ({prior_mel.shape[-1] / cfg.sr * cfg.hop:.1f} s)")

    # Assemble per-frame conditioning id tracks for whatever signals this checkpoint expects.
    cond_frames = {}
    for spec in cfg.conditioning:
        if spec.name == "technique":
            cond_frames[spec.name] = build_technique_frames(
                args.midi, prior_mel.shape[-1], args.technique)
        else:
            raise NotImplementedError(
                f"no inference-time builder for conditioning signal {spec.name!r}")
        # Both derive from the same MIDI render, but fail loud if they ever disagree.
        assert cond_frames[spec.name].shape[-1] == prior_mel.shape[-1], (
            f"conditioning track {spec.name!r} length {cond_frames[spec.name].shape[-1]} != "
            f"prior mel frames {prior_mel.shape[-1]}")
    cond_frames = cond_frames or None

    mel = generate_mel(model, prior_mel, cfg, args.device, args.solver, args.steps,
                       cond_frames=cond_frames)
    if args.save_mel:
        np.save(args.save_mel, mel)

    voc = load_vocoder(device=args.device)                      # frozen; also asserts mel contract
    audio = vocode(voc, torch.from_numpy(mel[None]).float())
    from common.audio_io import write_pcm16
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_pcm16(args.out, audio)
    print(f"wrote {args.out}  ({len(audio) / cfg.sr:.1f} s)")


if __name__ == "__main__":
    main()
