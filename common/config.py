"""Canonical audio constants shared across the Arioso project.

Single source of truth for values that every package must agree on — most
importantly the sample rate (training must read audio at the exact rate the
dataset was written at), the mel contract, and the cross-producer loudness
contract (``TARGET_RMS_DBFS`` / ``VOICED_TOP_DB``, shared by every GT
normalization and the prior's masked-RMS level match). It also names the
project-default model checkpoints — which BigVGAN generator (``VOCODER_DIR``)
and which Arioso checkpoint (``ACOUSTIC_CKPT``) inference starts from — so the
app and the CLIs agree on one default without hardcoding paths apiece. Truly
pipeline-only constants (e.g. ``BOOKS``, book paths) stay in
``DataSynthesizer/config.py``; the prior-shape knobs live in ``common/prior.py``.
"""

from __future__ import annotations

import os

# Canonical audio format for every wav in the project: mono, 16-bit PCM, SR Hz.
SR = 44100          # sample rate (Hz)
DEFAULT_PEAK = 0.95  # default peak for audio_io.normalize (avoids clipping)
PCM_SUBTYPE = "PCM_16"

# --- Cross-producer loudness contract -----------------------------------
# Every GT wav (whatever its source — YouTube rips for the synthetic corpus, the
# Labeler's recorded clips) is scaled so its RMS over voiced (non-silent) segments
# hits TARGET_RMS_DBFS, equalizing playing loudness across recordings. Measuring
# only over voiced segments keeps rests from dragging the level down. The Arioso
# prior is masked-RMS level-matched to the SAME TARGET_RMS_DBFS so prior/target
# levels agree (see common.prior.MaskedRMS, common.audio_io.voiced_rms_normalize).
TARGET_RMS_DBFS = -20.0   # voiced-RMS target for every GT track (dBFS)
VOICED_TOP_DB = 40.0      # librosa.effects.split: dB below peak counted as silence

# --- Mel-spectrogram contract -------------------------------------------
# These MUST match the BigVGAN-v2 vocoder checkpoint
# (nvidia/bigvgan_v2_44khz_128band_512x). Mels computed with any other values
# produce garbage when fed to the vocoder, so they live here as the single
# source of truth shared by data prep, training, and inference.
HOP_SIZE = 512        # hop length (samples); vocoder upsampling ratio
N_FFT    = 2048       # STFT FFT size
WIN_SIZE = 2048       # STFT window length (samples)
N_MELS   = 128        # number of mel bands
FMIN     = 0          # mel lower bound (Hz)
FMAX     = None       # mel upper bound; None => SR / 2 (22050 Hz)

# --- Vocoder checkpoint selection -----------------------------------------
# Directory holding the active BigVGAN generator (config.json +
# bigvgan_generator.pt). Points at the violin fine-tune (Arioso-mel adapted,
# 20k steps 2026-07-28, +2.8 dB MCD on Arioso-mel val) by default; set to None
# to use the stock HF checkpoint (nvidia/bigvgan_v2_44khz_128band_512x).
# Overridable per call via load_vocoder(checkpoint_dir=...). Missing dir fails
# loudly at load — silently vocoding with the wrong checkpoint would just sound
# mysteriously worse.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCODER_DIR: str | None = os.path.join(_REPO_ROOT, "Vocoder", "models", "ft_v2")

# --- Acoustic model checkpoint selection -----------------------------------
# Default Arioso checkpoint for inference (Studio's initial selection and the
# CLI fallback). Runs live under Arioso/models/<run>/; keep this pointing at a
# checkpoint inside that tree so Studio's scan can resolve it.
ACOUSTIC_CKPT: str = os.path.join(_REPO_ROOT, "Arioso", "models",
                                  "7-9-adr", "checkpoint_final.pt")
