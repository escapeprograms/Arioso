"""Degrade the violin recordings with sms-tools spectral models.

For each source recording we run two classic analysis/resynthesis models from
MTG's sms-tools and write the results back out as MP3:

* ``sineModel``  - sinusoidal model. Tracks and resynthesizes only the partials,
  discarding the residual (bow noise, breathiness). The result sounds thin/pure.
* ``spsModel``   - sinusoidal-plus-stochastic model. Adds a coarse stochastic
  approximation of that residual on top of the sinusoids, so it's fuller but
  still degraded relative to the original.

Run with the ai-violin env's interpreter, e.g.::

    "C:/Users/archi/Miniconda3/envs/ai-violin/python.exe" degrade_violin.py
"""

import os

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import get_window

from smstools.models import sineModel, spsModel

# --- Sources to degrade (project root, alongside this script) -----------------
HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ["D_violin.mp3", "DA_violin.mp3"]

# --- Standard analysis parameters --------------------------------------------
WINDOW = "blackman"   # analysis window type
M = 2001              # window length (odd)
N = 4096              # FFT size (power of 2, >= M)
H = 128               # hop size for the sps stochastic analysis (matches Ns=512)
T = -80               # peak-detection threshold (dB)


def normalize(y):
    """Scale to <= 1.0 peak so the MP3 encode doesn't clip."""
    peak = np.max(np.abs(y))
    return y / peak if peak > 1.0 else y


def degrade(path):
    """Load one source and return {suffix: (audio, sr)} for each model."""
    x, sr = librosa.load(path, sr=None, mono=True)
    x = x.astype(np.float64)
    w = get_window(WINDOW, M)

    # sineModel: sinusoids only (analysis + synthesis in one call).
    y_sine = sineModel.sineModel(x, sr, w, N, T)

    # spsModel: full sines + stochastic residual mix (first return value).
    y_sps, _ys, _yst = spsModel.spsModel(x, sr, w, N, H, T)

    return {"sine": (normalize(y_sine), sr), "sps": (normalize(y_sps), sr)}


def main():
    for src in SOURCES:
        src_path = os.path.join(HERE, src)
        stem, _ext = os.path.splitext(src)
        print(f"Processing {src} ...")

        for suffix, (y, sr) in degrade(src_path).items():
            out_path = os.path.join(HERE, f"{stem}_{suffix}.mp3")
            sf.write(out_path, y, sr, format="MP3")
            print(f"  -> {os.path.basename(out_path)}  "
                  f"({len(y) / sr:.2f} s @ {sr} Hz)")


if __name__ == "__main__":
    main()
