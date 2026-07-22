"""Arioso inference: score -> prior -> Euler-integrated mel -> audio (Section 9).

The prior is built from the input score **exactly as in training** (``common.prior`` via
``quantized_prior``: shaped additive saw + masked-RMS to the fixed target level + mel). Starting
from ``x = x_0`` at t=0, integrate the ODE ``x <- x + dt * v_theta(x, x_0, t, cond)`` with a
selectable solver (``Arioso.solvers.SOLVERS``: Euler / Heun / midpoint) over ``solver_steps`` steps
(no CFG). When the checkpoint's config has per-frame conditioning, a ``[T]`` id track per signal is
built from the SAME score (``build_cond_frames``: a constant ``--articulation`` over note groups, a
constant ``--vibrato`` flag over note spans, and velocity rasterized from the MIDI's own velocities)
and passed as fixed conditioning through the whole integration. The architecture is reconstructed
from the checkpoint's embedded config (``cfg_from_dict``), so pre-conditioning checkpoints load
unchanged. Long sequences are processed in overlapping chunks with a linear crossfade. The mel is
turned to audio with the **frozen** BigVGAN-v2 vocoder (listening only — never a selection arbiter).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from common.dataset_schema import (ARTIC_REST_ID, ARTICULATIONS, NoteEvent,
                                   expand_to_frames, note_groups_from_midi,
                                   rasterize_velocity, rasterize_vibrato)
from common.prior import quantized_prior
from common.vocoder import load_vocoder, mel_frames, vocode

from .config import SAMPLES_DIR, AriosoConfig, cfg_from_dict
from .model import AriosoModel
from .solvers import SOLVERS


def build_prior_mel(midi_path: str) -> np.ndarray:
    """Score -> prior mel ``[N_MELS, T]``, identical to the training-time prior (Section 9.1).

    Built by the same ``common.prior`` pipeline the dataset's ``prior_mel`` was, so the inference
    prior matches training. No GT-alignment shift here: the score's onsets *are* t=0.
    """
    synth = quantized_prior()   # factory defaults ARE the PRIOR_* knobs (one source of truth)
    return mel_frames(synth.render(midi_path))


def _note_events(midi_path: str, *, vibrato: bool = False) -> list[NoteEvent]:
    """All score notes as :class:`NoteEvent`\\ s (real MIDI velocities; a constant vibrato flag)."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(midi_path)
    return [NoteEvent(start_s=note.start, end_s=note.end, pitch=note.pitch,
                      velocity=note.velocity, vibrato=vibrato)
            for inst in pm.instruments for note in inst.notes]


def build_cond_frames(midi_path: str, n_frames: int, spec_name: str,
                      args: argparse.Namespace) -> np.ndarray:
    """``[T]`` uint8 conditioning-id track for one signal at inference time.

    All tracks are built from the SAME score (``offset_s=0.0`` — the score's onsets are t=0, matching
    ``build_prior_mel``'s no-shift convention) so the id grid lines up frame-for-frame with the prior
    mel. Per the standard encodings:

    * ``articulation`` — the constant ``args.articulation`` id assigned to every note group
      (``note_groups_from_midi`` + ``expand_to_frames``); gaps / lead-in / tail become rest.
    * ``velocity`` — the MIDI's own per-note velocities span-filled (rest = 0).
    * ``vibrato`` — the constant ``args.vibrato`` flag span-filled over note spans (rest = 2).

    An unknown signal (e.g. the deprecated ``"technique"`` classifier signal) is a
    ``NotImplementedError``.
    """
    if spec_name == "articulation":
        groups = note_groups_from_midi(midi_path, offset_s=0.0, n_frames=n_frames)
        artic_id = ARTICULATIONS.index(args.articulation)
        ids = np.full(len(groups), artic_id, dtype=np.uint8)
        return expand_to_frames(groups, ids, n_frames, rest_id=ARTIC_REST_ID)
    if spec_name == "velocity":
        return rasterize_velocity(_note_events(midi_path), n_frames)
    if spec_name == "vibrato":
        return rasterize_vibrato(_note_events(midi_path, vibrato=args.vibrato), n_frames)
    raise NotImplementedError(
        f"no inference-time builder for conditioning signal {spec_name!r}; 'technique' is a "
        "deprecated signal (its classifier was removed) — retrain with the "
        "articulation/velocity/vibrato signals instead")


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
    ap.add_argument("--articulation", choices=ARTICULATIONS, default="normal",
                    help="articulation conditioning applied to EVERY note (rest auto-fills gaps); "
                         "only used if the checkpoint's config has articulation conditioning")
    ap.add_argument("--vibrato", action=argparse.BooleanOptionalAction, default=False,
                    help="vibrato conditioning applied to EVERY note span (--vibrato / --no-vibrato); "
                         "only used if the checkpoint's config has vibrato conditioning")
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
        cond_frames[spec.name] = build_cond_frames(
            args.midi, prior_mel.shape[-1], spec.name, args)
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
