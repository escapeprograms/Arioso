"""Arioso hyperparameters — the single, toggleable config for the model + training.

One frozen dataclass (``AriosoConfig``) so a run is fully described by one object and
ablations are a one-field change. The **mel contract** is *not* redefined here — it is
imported from ``common.config`` (the project's single source of truth, asserted against
the BigVGAN checkpoint at vocoder-load time). The **prior** is a dataset artifact built
by ``DataSynthesizer.build_prior``; its knobs (anti-alias, envelope, level match, RMS
target) live in ``DataSynthesizer.config`` (the single source of truth shared with the
GT loudness normalization). Only Arioso-specific model/training knobs live here.

Defaults are the **spec baseline** (``SPEC_Arioso_v1_baseline.md``); every deferred or
out-of-scope feature is a toggle that defaults OFF so the baseline is the default run.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.config import HOP_SIZE, N_MELS, SR
# Prior-build output layout is owned by DataSynthesizer (it writes these dirs); re-export
# so Arioso readers (clips, dataset, eval) keep a single ``from .config import ...`` line.
from DataSynthesizer.config import ONSETS_DIR, PRIOR_MEL_DIR

# Frames per second of the mel grid (~86.13 at SR=44100, hop=512).
FRAME_RATE = SR / HOP_SIZE


@dataclass(frozen=True)
class AriosoConfig:
    """Everything that defines an Arioso run. Spec-baseline defaults."""

    # --- Mel contract (mirrored from common.config; do not override lightly) ------
    sr: int = SR
    hop: int = HOP_SIZE
    n_mels: int = N_MELS

    # The prior (Section 4) is a dataset artifact built by DataSynthesizer.build_prior;
    # its knobs live in DataSynthesizer.config, not here.

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

    # --- OT-CFM objective (Section 7) --------------------------------------------
    sigma: float = 1e-4

    # --- Training (Section 8) ----------------------------------------------------
    lr: float = 2e-4
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
    euler_steps: int = 24
    chunk_frames: int = 860
    overlap_frames: int = 16

    @property
    def dilations(self) -> list[int]:
        """The 20-entry dilation schedule: the cycle repeated ``wn_dilation_repeats`` times."""
        return list(self.wn_dilation_cycle) * self.wn_dilation_repeats

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


# --- Output layout ---------------------------------------------------------------
# Training *data* (prior mels, onset frames, split) lives under the DataSynthesizer `data/`
# root; model *artifacts* (checkpoints, listening samples) live under the Arioso package.
# PRIOR_MEL_DIR / ONSETS_DIR are re-exported from DataSynthesizer.config (above), which
# owns the prior build and writes those dirs.
SPLIT_FILE = "arioso_split.json"     # held-out-piece split (train/val basenames) (in data/)
CKPT_DIR = "Arioso/models"           # raw + EMA checkpoints (project-relative, gitignored)
SAMPLES_DIR = "Arioso/samples"       # listening artifacts (copy-synthesis, inference wavs)

# --- Experiment tracking ---------------------------------------------------------
# Weights & Biases destination for training runs. The API key is read from the env
# (WANDB_API_KEY) or a gitignored .env file (see .env.example), never hardcoded.
WANDB_ENTITY = "archimedesli"
WANDB_PROJECT = "Arioso"
