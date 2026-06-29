"""Arioso hyperparameters — the single, toggleable config for prior + model + training.

One frozen dataclass (``AriosoConfig``) so a run is fully described by one object and
ablations are a one-field change. The **mel contract** is *not* redefined here — it is
imported from ``common.config`` (the project's single source of truth, asserted against
the BigVGAN checkpoint at vocoder-load time). Only Arioso-specific knobs live here.

Defaults are the **spec baseline** (``SPEC_Arioso_v1_baseline.md``); every deferred or
out-of-scope feature is a toggle that defaults OFF so the baseline is the default run.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.config import HOP_SIZE, N_MELS, SR

# Frames per second of the mel grid (~86.13 at SR=44100, hop=512).
FRAME_RATE = SR / HOP_SIZE


@dataclass(frozen=True)
class AriosoConfig:
    """Everything that defines an Arioso run. Spec-baseline defaults."""

    # --- Mel contract (mirrored from common.config; do not override lightly) ------
    sr: int = SR
    hop: int = HOP_SIZE
    n_mels: int = N_MELS

    # --- Prior generation (Section 4) --------------------------------------------
    # pitch_source: "quantized" = constant MIDI-pitch saw (baseline); "bend" = pitch-
    #   wheel-following (deferred ablation, rendered by DataSynthesizer.render_prior_bend).
    pitch_source: str = "quantized"
    # anti_alias: True = polyBLEP band-limited saw; False = naive scipy sawtooth.
    anti_alias: bool = True
    # envelope: "rect" = hard on/off note gating (baseline, per build decision);
    #   "fade" = 5 ms linear anti-click ramp (DataSynthesizer._fade_envelope).
    #   ADSR is intentionally NOT implemented this round.
    envelope: str = "rect"
    fade_ms: float = 5.0
    # level_match: "masked_rms" = scale prior so its sounding-frame RMS hits target_rms_dbfs
    #   (single per-recording scalar, Section 4.3); "peak" = peak-normalize.
    level_match: str = "masked_rms"
    # The level all GTs were voiced-RMS-normalized to by DataSynthesizer (TARGET_RMS_DBFS).
    # Matching the prior to this *constant* (not a per-recording target) keeps prior generation
    # fully score-determined => identical at train and inference (Section 9.1 / checklist).
    target_rms_dbfs: float = -20.0

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
        assert self.pitch_source in ("quantized", "bend")
        assert self.envelope in ("rect", "fade")
        assert self.level_match in ("masked_rms", "peak")


# --- Output layout (new dirs under the DataSynthesizer `data/` root) -------------
PRIOR_MEL_DIR = "prior_mel_arioso"   # [N_MELS, T] float32, masked-RMS-matched prior mel
ONSETS_DIR = "onsets_arioso"         # [K] int32 aligned onset frame indices per recording
SPLIT_FILE = "arioso_split.json"     # held-out-piece split (train/val basenames)
CKPT_DIR = "checkpoints_arioso"      # raw + EMA checkpoints
