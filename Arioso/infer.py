"""Arioso inference: score -> prior -> Euler-integrated mel -> audio (Section 9).

The prior is built from the input score **exactly as in training** (``common.prior`` via
``quantized_prior``: shaped additive saw + masked-RMS to the fixed target level + mel). Starting
from ``x = x_0`` at t=0, integrate the ODE ``x <- x + dt * v_theta(x, x_0, t, cond)`` with a
selectable solver (``Arioso.solvers.SOLVERS``: Euler / Heun / midpoint) over ``solver_steps`` steps
(no CFG). When the checkpoint's config has per-frame conditioning, a ``[T]`` track per signal is
built from the SAME score (``build_cond``: categorical id tracks — a constant ``--articulation`` over
note spans, a constant ``--vibrato`` flag over note spans, velocity rasterized from the MIDI's own
velocities — plus raw int64 boundary-distance tracks for the time-since-onset / time-until-offset
signals, sinusoid-featurized in-model) and passed as fixed conditioning through the whole
integration. The architecture is reconstructed
from the checkpoint's embedded config (``cfg_from_dict``), so pre-conditioning checkpoints load
unchanged. Long sequences are processed in overlapping chunks with a linear crossfade. The mel is
turned to audio with the **frozen** BigVGAN-v2 vocoder (listening only — never a selection arbiter).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from common.dataset_schema import (ARTICULATIONS, NoteEvent, offset_frames,
                                   onset_frames, rasterize_articulation,
                                   rasterize_velocity, rasterize_vibrato)
from common.prior import quantized_prior
from common.vocoder import load_vocoder, mel_frames, vocode

from .config import (SAMPLES_DIR, AriosoConfig, BoundaryCondSpec, CondSpec,
                     cfg_from_dict)
from .dataset import boundary_distances
from .model import AriosoModel
from .solvers import SOLVERS


def build_prior_mel(midi_path: str) -> np.ndarray:
    """Score -> prior mel ``[N_MELS, T]``, identical to the training-time prior (Section 9.1).

    Built by the same ``common.prior`` pipeline the dataset's ``prior_mel`` was, so the inference
    prior matches training. No GT-alignment shift here: the score's onsets *are* t=0.
    """
    synth = quantized_prior()   # factory defaults ARE the PRIOR_* knobs (one source of truth)
    return mel_frames(synth.render(midi_path))


def _note_events(midi_path: str, *, articulation: str = "normal",
                 vibrato: bool = False) -> list[NoteEvent]:
    """All score notes as :class:`NoteEvent`\\ s (real MIDI velocities; a constant articulation/vibrato).

    Every note is stamped with the same ``articulation`` name and ``vibrato`` flag (the CLI applies
    one constant technique / vibrato state to the whole score); MIDI velocities are kept per-note.
    """
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(midi_path)
    return [NoteEvent(start_s=note.start, end_s=note.end, pitch=note.pitch,
                      velocity=note.velocity, articulation=articulation, vibrato=vibrato)
            for inst in pm.instruments for note in inst.notes]


def build_cond(notes: list[NoteEvent], n_frames: int,
               specs: tuple[CondSpec | BoundaryCondSpec, ...]) -> dict[str, np.ndarray]:
    """``{name: [T] track}`` per-frame conditioning at inference time, for whatever ``specs`` expect.

    All tracks are rasterized from the SAME ``notes`` (``offset_s=0.0`` — the score's onsets are t=0,
    matching ``build_prior_mel``'s no-shift convention) so the grid lines up frame-for-frame with the
    prior mel. The two spec kinds are built differently:

    * **Categorical** (:class:`~Arioso.config.CondSpec`), ``[T]`` uint8 id tracks — ``articulation``
      (:func:`rasterize_articulation`, each note's stamped name; gaps / lead-in / tail become rest),
      ``velocity`` (:func:`rasterize_velocity`, the MIDI's own per-note velocities; rest = 0) and
      ``vibrato`` (:func:`rasterize_vibrato`, the per-note flag; rest = 2). An unknown categorical
      name (e.g. the deprecated ``"technique"`` classifier signal) is a ``NotImplementedError``.
    * **Boundary** (:class:`~Arioso.config.BoundaryCondSpec`), ``[T]`` **int64** distance tracks —
      the root's ``onsets``/``offsets`` frame array (:func:`onset_frames` drops out-of-range,
      :func:`offset_frames` clamps the tail — the deliberate training policy) turned into a per-frame
      distance via :func:`~Arioso.dataset.boundary_distances` (sentinel ``-1``). These MUST stay
      int64: the ``-1`` sentinel would become 255 under uint8 and corrupt the compact support.
    """
    cond: dict[str, np.ndarray] = {}
    for spec in specs:
        if isinstance(spec, BoundaryCondSpec):
            arr = (onset_frames if spec.boundary == "onset" else offset_frames)(notes, n_frames)
            cond[spec.name] = boundary_distances(arr.astype(np.int64), 0, n_frames, spec.direction)
        elif spec.name == "articulation":
            cond[spec.name] = rasterize_articulation(notes, n_frames)
        elif spec.name == "velocity":
            cond[spec.name] = rasterize_velocity(notes, n_frames)
        elif spec.name == "vibrato":
            cond[spec.name] = rasterize_vibrato(notes, n_frames)
        else:
            raise NotImplementedError(
                f"no inference-time builder for conditioning signal {spec.name!r}; 'technique' is a "
                "deprecated signal (its classifier was removed) — retrain with the "
                "articulation/velocity/vibrato signals instead")
    return cond


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

    # Assemble per-frame conditioning tracks for whatever signals this checkpoint expects, all from
    # the one score (constant --articulation / --vibrato, per-note MIDI velocities + note boundaries).
    notes = _note_events(args.midi, articulation=args.articulation, vibrato=args.vibrato)
    cond_frames = build_cond(notes, prior_mel.shape[-1], cfg.conditioning) or None
    if cond_frames is not None:
        for name, arr in cond_frames.items():
            # Both derive from the same score, but fail loud if they ever disagree.
            assert arr.shape[-1] == prior_mel.shape[-1], (
                f"conditioning track {name!r} length {arr.shape[-1]} != "
                f"prior mel frames {prior_mel.shape[-1]}")

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
