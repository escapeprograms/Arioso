"""Canonical audio constants shared across the Arioso project.

Single source of truth for values that every package must agree on — most
importantly the sample rate, since training must read audio at the exact rate
the dataset was written at. Pipeline-specific constants do **not** belong here
(see ``DataSynthesizer/config.py``).
"""

from __future__ import annotations

# Canonical audio format for every wav in the project: mono, 16-bit PCM, SR Hz.
SR = 44100          # sample rate (Hz)
DEFAULT_PEAK = 0.95  # default peak for audio_io.normalize (avoids clipping)
PCM_SUBTYPE = "PCM_16"

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
