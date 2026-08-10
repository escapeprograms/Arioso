"""Arioso hyperparameters — the single, toggleable config for the model + training.

One frozen dataclass (``AriosoConfig``) so a run is fully described by one object and
ablations are a one-field change. The **mel contract** is *not* redefined here — it is
imported from ``common.config`` (the project's single source of truth, asserted against
the BigVGAN checkpoint at vocoder-load time). The **prior** is a dataset artifact built
by ``DataSynthesizer.build_prior``; its shape knobs (anti-alias, envelope, level match,
RMS target) live in ``common.prior`` (shared with Arioso inference and the Studio
renderer). Only Arioso-specific model/training knobs live here.

Defaults are the **spec baseline** (``SPEC_Arioso_v1_baseline.md``); every deferred or
out-of-scope feature is a toggle that defaults OFF so the baseline is the default run.

**Per-frame conditioning** (``AriosoConfig.conditioning``) comes in two kinds, both per-frame and
both embedded then concatenated to ``[x_t, x_0]`` before the input projection:

* **Categorical** (``CondSpec``): the three standard signals — articulation, velocity, vibrato —
  each a ``[T]`` id track embedded via a lookup table, optional per dataset root, resolved from the
  ``SIGNALS`` registry.
* **Sinusoidal boundary distance** (``BoundaryCondSpec``): a continuous per-frame distance to the
  nearest note boundary (time since onset / time until offset), sinusoidally featurized over a
  compact support window and zero outside it; resolved from the ``BOUNDARY_SIGNALS`` registry.

``AriosoConfig.conditioning`` defaults to ``()`` (the main synthetic corpus carries no labels); a
labelled root opts in via ``conditioning: [articulation, velocity, vibrato, time_since_onset,
time_until_offset]``. Classifier-free-guidance dropout (``cond_dropout``) is a deferred toggle and
defaults **OFF**. Checkpoints embed the config via ``cfg_to_dict`` / ``cfg_from_dict`` (plain dicts
only, so ``torch.load(weights_only=True)`` under torch 2.6 stays safe, and pre-conditioning
checkpoints reconstruct the exact old architecture).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from common.config import HOP_SIZE, N_MELS, SR
from common.dataset_schema import (DIR_ONSETS, DIR_PRIOR_MEL, SIGNAL_NUM_CLASSES,
                                   SIGNAL_REST_ID, SPLIT_JSON, cond_dir)

# The standard prior/onsets dir names, re-exported under the historical constant names so
# Arioso readers (clips, dataset, eval) keep a single ``from .config import ...`` line. The
# values come from common.dataset_schema ("prior_mel" / "onsets") — the on-disk names every
# standard root uses (the synthetic root was migrated from the old "prior_mel_arioso" /
# "onsets_arioso" by DataSynthesizer.migrate_root).
PRIOR_MEL_DIR = DIR_PRIOR_MEL
ONSETS_DIR = DIR_ONSETS

# Frames per second of the mel grid (~86.13 at SR=44100, hop=512).
FRAME_RATE = SR / HOP_SIZE


@dataclass(frozen=True)
class CondSpec:
    """One per-frame categorical conditioning signal (a dataset artifact dir of [T] uint8 id
    tracks) and how to embed it. The embedding table gets one extra row at index
    ``num_classes`` — the "unknown" id used by conditioning dropout (CFG), never in data."""
    name: str
    num_classes: int
    emb_dim: int
    dir: str          # data/ subdir holding <base>.npy [T] uint8 id tracks
    pad_id: int = 0   # id for batch padding (padded frames are masked by frame_mask anyway)


# The three standard conditioning signals, each optional per root. Their id tracks live under
# ``cond/<signal>/`` on a standard dataset root; class counts / rest (pad) ids are DERIVED from
# common.dataset_schema so the on-disk encoding and the embedding tables can never drift. Padded
# frames use each signal's rest id (masked by frame_mask anyway). The main synthetic corpus carries
# none of these (AriosoConfig.conditioning defaults to () — the unconditioned baseline); a labelled
# root (e.g. gt_arky) opts in via ``conditioning: [articulation, velocity, vibrato]``.
ARTICULATION_COND = CondSpec("articulation", num_classes=SIGNAL_NUM_CLASSES["articulation"],
                             emb_dim=64, dir=cond_dir("articulation"),
                             pad_id=SIGNAL_REST_ID["articulation"])
VELOCITY_COND = CondSpec("velocity", num_classes=SIGNAL_NUM_CLASSES["velocity"],
                         emb_dim=64, dir=cond_dir("velocity"),
                         pad_id=SIGNAL_REST_ID["velocity"])
VIBRATO_COND = CondSpec("vibrato", num_classes=SIGNAL_NUM_CLASSES["vibrato"],
                        emb_dim=16, dir=cond_dir("vibrato"),
                        pad_id=SIGNAL_REST_ID["vibrato"])

# Conditioning-signal registry: name -> the canonical CondSpec for that signal. A YAML run-config
# names a signal (e.g. ``conditioning: [articulation]``) and the loader resolves it here, so adding
# a new signal is one CondSpec constant + one entry (mirrors the SOLVERS/SCHEDULES registry pattern).
SIGNALS: dict[str, CondSpec] = {
    "articulation": ARTICULATION_COND,
    "velocity": VELOCITY_COND,
    "vibrato": VIBRATO_COND,
}


@dataclass(frozen=True)
class BoundaryCondSpec:
    """One per-frame *continuous* conditioning signal: sinusoidal distance to a note boundary.

    Where a :class:`CondSpec` embeds a categorical id per frame, this signal encodes a real-valued
    distance (in frames) from each frame to the nearest note boundary — the onset it follows
    (``direction="since"``) or the offset it precedes (``direction="until"``). The dataset turns the
    root's ``onsets/`` / ``offsets/`` frame arrays into a per-frame distance track (sentinel ``-1``
    where no boundary is in range); :class:`~Arioso.model.conditioning.BoundarySinusoid` featurizes
    it with sin/cos over a geometric frequency band (like the flow-time embedding).

    **Compact support.** The signal is active only within ``window_ms`` of the boundary; a frame
    farther away (or the ``-1`` sentinel) maps to the exact **zero vector**, so there is no
    "unknown" class and no ``num_classes`` / ``pad_id`` — inactivity is representable natively (batch
    padding uses ``-1``). This makes it a smooth, local attack/decay-shaped cue rather than a global
    ramp.

    The 50 ms default mirrors the legacy ``DataSynthesizer`` ``ONSET_DECAY_MS`` — the width of the
    old scalar onset-emphasis mask this per-frame signal supersedes. It is a **deliberate literal**,
    not an import: producers (DataSynthesizer / Labeler) and consumers (Arioso) never cross-import,
    so the provenance is documented here rather than coupled by reference.

    Fields:
        name:       unique signal name (registry key + cond-dict key).
        emb_dim:    output width per frame; ``emb_dim // 2`` sin + ``emb_dim // 2`` cos (must be even).
        boundary:   which per-root frame array feeds the distance — ``"onset"`` or ``"offset"``.
        direction:  ``"since"`` (frame - previous boundary) or ``"until"`` (next boundary - frame).
        window_ms:  compact-support half-life X in milliseconds (distance normalizer; active window).
    """
    name: str
    emb_dim: int = 64          # 32 sin + 32 cos; must be even
    boundary: str = "onset"    # which per-root artifact array: "onset" | "offset"
    direction: str = "since"   # "since" (frame - prev boundary) | "until" (next boundary - frame)
    window_ms: float = 50.0    # X: compact support in ms; default = legacy DataSynthesizer ONSET_DECAY_MS


# Boundary-signal registry: name -> canonical BoundarySinusoid spec (mirrors SIGNALS). The two
# standard signals bracket a note — "time since its onset" and "time until its offset" — each a
# compact 50 ms attack/decay-shaped cue resolved by the run-config loader.
TIME_SINCE_ONSET = BoundaryCondSpec("time_since_onset", emb_dim=64, boundary="onset",
                                    direction="since", window_ms=50.0)
TIME_UNTIL_OFFSET = BoundaryCondSpec("time_until_offset", emb_dim=64, boundary="offset",
                                     direction="until", window_ms=50.0)
BOUNDARY_SIGNALS: dict[str, BoundaryCondSpec] = {
    "time_since_onset": TIME_SINCE_ONSET,
    "time_until_offset": TIME_UNTIL_OFFSET,
}


@dataclass(frozen=True)
class AriosoConfig:
    """Everything that defines an Arioso run. Spec-baseline defaults."""

    # --- Mel contract (mirrored from common.config; do not override lightly) ------
    sr: int = SR
    hop: int = HOP_SIZE
    n_mels: int = N_MELS

    # The prior (Section 4) is a dataset artifact built by DataSynthesizer.build_prior;
    # its shape knobs live in common.prior, not here.

    # --- Model architecture (Section 6) ------------------------------------------
    hidden: int = 384
    in_ch: int = 2 * N_MELS          # [x_t, x_0] concatenated => 256
    t_emb_dim: int = 256
    # WaveNet
    wn_blocks: int = 20
    wn_kernel: int = 3
    wn_dilation_cycle: tuple = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    wn_dilation_repeats: int = 2     # cycle repeated this many times => 20 blocks
    # DiT
    dit_blocks: int = 3
    dit_heads: int = 6
    dit_head_dim: int = 64           # heads * head_dim == hidden (6*64 == 384)
    dit_ffn: int = 1536
    rope_base: float = 10000.0
    # Per-frame conditioning: a mix of categorical CondSpecs (embedding-table id tracks) and
    # continuous BoundaryCondSpecs (sinusoidal note-boundary distance), each embedded and
    # concatenated to [x_t, x_0] before in_proj. Defaults to () — the main synthetic corpus carries
    # no labels; a labelled root opts in via
    # ``conditioning: [articulation, velocity, vibrato, time_since_onset, time_until_offset]``.
    conditioning: tuple[CondSpec | BoundaryCondSpec, ...] = ()
    # CFG lever (deferred, OFF): fraction of samples whose ENTIRE conditioning is swapped for the
    # per-spec "unknown" id (row num_classes) during training. See model.conditioning.
    cond_dropout: float = 0.0

    # --- OT-CFM objective (Section 7) --------------------------------------------
    sigma: float = 1e-4
    loss: str = "masked_mse"         # velocity loss; see Arioso.cfm.LOSSES
    # Off-path augmentation (deferred toggle, OFF): for a Bernoulli(path_aug_p) subset of train
    # samples, perturb x_t off the OT line and re-aim the target so constant velocity still lands on
    # the path endpoint at t=1 — teaching the field a restoring component. See Arioso.augment.perturb_off_path.
    path_aug_p: float = 0.0          # fraction of train samples perturbed off the OT line (0 = off)
    path_aug_std: float = 1.0        # noise scale; effective displacement std is path_aug_std*t(1-t) <= std/4
    # Anti-Drift Rectification (deferred toggle, OFF): a second forward drift-simulates from the
    # model's own velocity and penalizes the drifted state's direction toward data — teaching a
    # restoring component (DEFAR, arXiv 2606.28226; ADR only, no Frequency Compensation). See
    # Arioso.augment.
    adr_beta: float = 0.0            # ADR loss weight in L = L_FM + adr_beta*L_ADR (0 = off)
    adr_p: float = 1.0               # fraction of steps ADR is applied (per-step Bernoulli gate; paper: 1.0)
    # Pitch-shift augmentation (deferred toggle, OFF): for a Bernoulli(pitch_aug_p) subset of TRAIN
    # samples, both the prior and the target mel are recomputed from their waveforms with a
    # DiffSinger keyshift of k ~ U(-pitch_aug_cents, +pitch_aug_cents), drawn fresh per sample per
    # epoch. The shift is duration-preserving, so the frame grid — and with it every cond / onset /
    # boundary track and the clip framing — is untouched. Requires ``prior_wav/`` in the root; a root
    # predating it is never augmented (graceful fallback to the memmapped mels). See Arioso.pitch_aug
    # + common.keyshift.
    pitch_aug_p: float = 0.0         # fraction of train samples pitch-shifted (0 = off)
    pitch_aug_cents: float = 100.0   # half-width of the uniform cents draw (100 = +/- 1 semitone)

    # --- Training (Section 8) ----------------------------------------------------
    lr: float = 2e-4
    lr_schedule: str = "cosine"      # post-warmup annealing shape; see Arioso.schedules.SCHEDULES
    lr_min: float = 0.0              # LR floor the schedule anneals to at total_steps
    weight_decay: float = 0.01
    warmup_steps: int = 4000
    total_steps: int = 200_000
    grad_clip: float = 1.0
    ema_max: float = 0.9999
    seed: int = 0
    val_frac: float = 0.10           # fraction of *pieces* held out for eval

    # --- Clip enumeration (Section 5) --------------------------------------------
    l_min_s: float = 5.0
    target_s: float = 10.0

    # --- Inference (Section 9) ---------------------------------------------------
    solver: str = "euler"            # integration method; see Arioso.solvers.SOLVERS
    solver_steps: int = 24           # outer steps (NFE = solver_steps * nfe_per_step)
    chunk_frames: int = 860
    overlap_frames: int = 16

    @property
    def dilations(self) -> list[int]:
        """The 20-entry dilation schedule: the cycle repeated ``wn_dilation_repeats`` times."""
        return list(self.wn_dilation_cycle) * self.wn_dilation_repeats

    @property
    def cond_dim(self) -> int:
        """Total conditioning embedding width concatenated to [x_t, x_0] (0 when no specs)."""
        return sum(spec.emb_dim for spec in self.conditioning)

    @property
    def l_min_frames(self) -> int:
        return int(round(self.l_min_s * FRAME_RATE))

    @property
    def target_frames(self) -> int:
        return int(round(self.target_s * FRAME_RATE))

    def __post_init__(self) -> None:
        # Fail loud on the invariants the model code relies on.
        assert self.dit_heads * self.dit_head_dim == self.hidden, \
            "dit_heads * dit_head_dim must equal hidden"
        assert len(self.dilations) == self.wn_blocks, \
            "dilation cycle * repeats must equal wn_blocks"
        assert self.in_ch == 2 * self.n_mels, "in_ch must be 2 * n_mels ([x_t, x_0])"
        # Conditioning invariants: unique names across BOTH kinds, then per-kind checks.
        names = [spec.name for spec in self.conditioning]
        assert len(names) == len(set(names)), f"conditioning spec names must be unique: {names}"
        for spec in self.conditioning:
            if isinstance(spec, CondSpec):
                # Categorical: positive even-free emb dim + an in-range pad id.
                assert spec.emb_dim > 0, f"conditioning spec {spec.name!r} emb_dim must be > 0"
                assert 0 <= spec.pad_id < spec.num_classes, \
                    f"conditioning spec {spec.name!r} pad_id must be in [0, num_classes)"
            else:
                # Boundary: sin/cos split needs an even, positive emb dim + a positive window;
                # boundary/direction must name a real array and distance sign.
                assert spec.emb_dim > 0 and spec.emb_dim % 2 == 0, \
                    f"boundary spec {spec.name!r} emb_dim must be even and > 0 (sin+cos halves)"
                assert spec.window_ms > 0.0, f"boundary spec {spec.name!r} window_ms must be > 0"
                assert spec.boundary in {"onset", "offset"}, \
                    f"boundary spec {spec.name!r} boundary must be 'onset' or 'offset'"
                assert spec.direction in {"since", "until"}, \
                    f"boundary spec {spec.name!r} direction must be 'since' or 'until'"
        assert 0.0 <= self.cond_dropout < 1.0, "cond_dropout must be in [0.0, 1.0)"
        # Off-path augmentation invariants.
        assert 0.0 <= self.path_aug_p <= 1.0, "path_aug_p must be in [0.0, 1.0]"
        assert self.path_aug_std >= 0.0, "path_aug_std must be >= 0.0"
        # Anti-Drift Rectification invariants.
        assert self.adr_beta >= 0.0, "adr_beta must be >= 0.0"
        assert 0.0 <= self.adr_p <= 1.0, "adr_p must be in [0.0, 1.0]"
        # Pitch-shift augmentation invariants.
        assert 0.0 <= self.pitch_aug_p <= 1.0, "pitch_aug_p must be in [0.0, 1.0]"
        assert self.pitch_aug_cents >= 0.0, "pitch_aug_cents must be >= 0.0"


# --- Checkpoint (de)serialization ------------------------------------------------
# A run is fully described by its AriosoConfig, embedded in each checkpoint under "cfg". torch 2.6
# defaults ``torch.load(weights_only=True)``, which refuses pickled dataclass instances, so we
# store PLAIN dicts (CondSpec -> dict too) and reconstruct on load. ``cfg_from_dict`` filters to
# known fields (so removed/foreign keys are ignored) and treats a missing "conditioning" key as
# the empty tuple, letting PRE-conditioning checkpoints rebuild the exact old architecture.

def cfg_to_dict(cfg: AriosoConfig) -> dict:
    """AriosoConfig -> plain nested dict (``dataclasses.asdict``; each CondSpec becomes a dict).

    Keeps checkpoints ``torch.load(weights_only=True)``-safe: no pickled dataclass instances.
    """
    return dataclasses.asdict(cfg)


def cfg_from_dict(d: dict) -> AriosoConfig:
    """Rebuild an AriosoConfig from a (checkpoint-embedded) dict.

    Robust to schema drift: only keys matching current field names are used (removed/foreign keys
    are dropped). A missing ``conditioning`` key becomes ``()`` so pre-conditioning checkpoints
    reconstruct the exact old (no-conditioning) architecture. List-typed tuple fields
    (``wn_dilation_cycle``, ``conditioning``) are coerced back to tuples, and each conditioning
    entry is rebuilt into its spec: an existing ``CondSpec`` / ``BoundaryCondSpec`` passes through;
    a dict is a ``CondSpec`` when it carries ``num_classes`` (the categorical discriminator) else a
    ``BoundaryCondSpec`` — so a mixed conditioning tuple round-trips through ``cfg_to_dict``.
    """
    field_names = {f.name for f in dataclasses.fields(AriosoConfig)}
    kw = {k: v for k, v in d.items() if k in field_names}

    # A pre-conditioning checkpoint has no "conditioning" key -> the old architecture had none.
    specs = kw.get("conditioning", ())

    def _rebuild(s):
        if isinstance(s, (CondSpec, BoundaryCondSpec)):
            return s
        return CondSpec(**s) if "num_classes" in s else BoundaryCondSpec(**s)

    kw["conditioning"] = tuple(_rebuild(s) for s in specs)
    if "wn_dilation_cycle" in kw:
        kw["wn_dilation_cycle"] = tuple(kw["wn_dilation_cycle"])
    return AriosoConfig(**kw)


# --- Output layout ---------------------------------------------------------------
# Training *data* (prior mels, onset frames, split) lives under a standard dataset root;
# model *artifacts* (checkpoints, listening samples) live under the Arioso package.
# PRIOR_MEL_DIR / ONSETS_DIR are re-exported from common.dataset_schema (above); the split
# file name is the standard SPLIT_JSON ("split.json", written by Arioso.splits per root).
SPLIT_FILE = SPLIT_JSON              # held-out-piece split (train/val basenames), in the root
CKPT_DIR = "Arioso/models"           # raw + EMA checkpoints (project-relative, gitignored)
SAMPLES_DIR = "Arioso/samples"       # listening artifacts (copy-synthesis, inference wavs)

# --- Experiment tracking ---------------------------------------------------------
# Weights & Biases destination for training runs. The API key is read from the env
# (WANDB_API_KEY) or a gitignored .env file (see .env.example), never hardcoded.
WANDB_ENTITY = "archimedesli"
WANDB_PROJECT = "Arioso"
