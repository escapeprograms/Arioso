"""Pipeline-specific constants for the DataSynthesizer build.

The canonical sample rate lives in ``common.config`` (shared with training) and
is re-exported here so the pipeline modules can keep a single ``from .config
import SR, ...`` line. The prior-synthesis knobs and the cross-producer loudness
contract moved out (they are shared by more than this build): the ``PRIOR_*`` /
``FADE_MS`` knobs live in ``common.prior``, and ``TARGET_RMS_DBFS`` /
``VOICED_TOP_DB`` in ``common.config``. The standard on-disk dir names (prior mel,
onsets, cond tracks) live in ``common.dataset_schema``. Everything below is what
remains specific to building the synthetic dataset.
"""

from __future__ import annotations

# Canonical rate and hop are defined in common.config (shared with training and
# the vocoder) and re-exported here so pipeline modules keep a single import line.
from common.config import SR, HOP_SIZE as HOP

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
    "SR", "HOP",
    "ONSET_DECAY_MS", "ONSET_DECAY_FLOOR",
    "BOOKS", "DEFAULT_DATASET", "DEFAULT_OUT",
]
