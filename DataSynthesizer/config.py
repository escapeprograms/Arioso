"""Pipeline-specific constants for the DataSynthesizer build.

The canonical sample rate lives in ``common.config`` (shared with training) and
is re-exported here so the pipeline modules can keep a single ``from .config
import SR, ...`` line. Everything else below is specific to building the dataset.
"""

from __future__ import annotations

# Canonical rate and hop are defined in common.config (shared with training and
# the vocoder) and re-exported here so pipeline modules keep a single import line.
from common.config import SR, HOP_SIZE as HOP

# --- Prior synthesis / alignment ----------------------------------------
FADE_MS = 5.0       # per-note linear fade in/out, to avoid click artifacts

# --- Target loudness normalization (voiced-segment RMS) -----------------
# Each downloaded track is scaled so its RMS over voiced (non-silent) segments
# hits TARGET_RMS_DBFS, equalizing volume across the different YouTube channels.
# Measuring only over voiced segments keeps rests from dragging the level down.
TARGET_RMS_DBFS = -20.0   # voiced-RMS target for each downloaded track (dBFS)
VOICED_TOP_DB = 40.0      # librosa.effects.split: dB below peak counted as silence
# (the post-gain peak-clip guard reuses common.config.DEFAULT_PEAK = 0.95)

# --- Onset mask ---------------------------------------------------------
# Training signal: 1 on each onset frame, exponential decay to ~0 over a support
# window of ONSET_DECAY_MS, then hard 0. ONSET_DECAY_FLOOR is the value reached at
# the window edge (it sets the decay time constant tau).
ONSET_DECAY_MS = 50.0     # X: exp-decay support window (ms); mask is ~0 by X then 0
ONSET_DECAY_FLOOR = 0.01  # mask value at dt = X (sets tau), before the hard 0

# --- Dataset / output layout --------------------------------------------
BOOKS = ("Kayser", "Paganini", "Wohlfahrt")
DEFAULT_DATASET = "external/violin-transcription/dataset"  # relative to the project root
DEFAULT_OUT = "data"

__all__ = [
    "SR", "HOP", "FADE_MS",
    "TARGET_RMS_DBFS", "VOICED_TOP_DB",
    "ONSET_DECAY_MS", "ONSET_DECAY_FLOOR",
    "BOOKS", "DEFAULT_DATASET", "DEFAULT_OUT",
]
