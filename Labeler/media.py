"""Media derivatives for the viewer: mel tiles, onset envelope, time-stretches.

All display artifacts the frontend loads:

  * **mel tiles** — the BigVGAN-contract log-mel (via
    ``common.vocoder.mel_spectrogram``, the only mel that matches the vocoder),
    per-clip percentile-scaled (p1..p99 -> 0..1), colormapped through magma and
    sliced into ``tile_frames``-wide PNGs (last tile ragged) so a multi-minute clip
    draws only the visible window. ``meta.json`` records the tiling + the
    ``mel_bin_of_midi[128]`` lookup (fractional mel bin of each MIDI pitch's
    fundamental) the frontend uses to place note rects and open-string gridlines.
  * **onset_env.f32** — raw little-endian float32 ``librosa.onset.onset_strength``
    (hop 512), the edge-snap signal for resize.
  * **time-stretches** — pitch-preserving phase-vocoder renders of both wav variants
    at each configured speed (0.5 -> n_fft 2048, 0.25 -> n_fft 4096) for slow playback.

torch is only pulled in (lazily) inside :func:`mel_tiles`; matplotlib runs headless
(Agg backend, set before any pyplot/cm import).
"""

from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")  # headless; MUST precede pyplot/cm import
import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402

from common.audio_io import write_pcm16  # noqa: E402
from common.config import FMIN, HOP_SIZE as HOP, N_MELS, SR  # noqa: E402

from .config import MediaParams  # noqa: E402
from .library import MEL_META, STRETCH_DIR  # noqa: E402


def stretch_name(variant: str, speed: float) -> str:
    """Filename for a stretched variant, e.g. ``cleaned_x050.wav`` for speed 0.5."""
    return f"{variant}_x{int(round(speed * 100)):03d}.wav"


def write_onset_env(cleaned: np.ndarray, path: str, sr: int = SR, hop: int = HOP) -> str:
    """Write ``onset_strength`` as raw little-endian float32 bytes to ``path``."""
    import librosa
    env = librosa.onset.onset_strength(
        y=np.asarray(cleaned, dtype=np.float32), sr=sr, hop_length=hop)
    with open(path, "wb") as f:
        f.write(env.astype("<f4").tobytes())
    return path


def mel_bin_of_midi() -> list[float]:
    """Fractional mel bin of each MIDI pitch 0..127's fundamental (clamped 0..N-1)."""
    import librosa
    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=FMIN, fmax=SR / 2, htk=False)
    midi_hz = 440.0 * 2.0 ** ((np.arange(128) - 69) / 12.0)
    bins = np.interp(midi_hz, mel_freqs, np.arange(N_MELS))  # np.interp clamps ends
    return [float(b) for b in np.clip(bins, 0, N_MELS - 1)]


def mel_tiles(cleaned: np.ndarray, mel_dir: str, params: MediaParams | None = None,
              sr: int = SR, hop: int = HOP) -> dict:
    """Render magma mel tiles + ``meta.json`` for ``cleaned``; return the meta dict."""
    from common.vocoder import mel_spectrogram  # lazy: pulls torch + BigVGAN

    params = params or MediaParams()
    os.makedirs(mel_dir, exist_ok=True)

    mel = mel_spectrogram(np.asarray(cleaned, dtype=np.float32))  # torch [1,128,T] or [128,T]
    mel = mel.detach().cpu().numpy()
    if mel.ndim == 3:
        mel = mel[0]
    # Already log-mel (BigVGAN ln convention); scale per clip by p_low..p_high.
    p_lo = float(np.percentile(mel, params.mel_p_low))
    p_hi = float(np.percentile(mel, params.mel_p_high))
    denom = (p_hi - p_lo) or 1.0
    norm = np.clip((mel - p_lo) / denom, 0.0, 1.0)

    n_frames = norm.shape[1]
    tw = params.tile_frames
    n_tiles = max(1, math.ceil(n_frames / tw))
    tiles, widths = [], []
    for i in range(n_tiles):
        a, b = i * tw, min((i + 1) * tw, n_frames)
        tile = norm[:, a:b]
        fname = f"tile_{i:04d}.png"
        # origin='lower' -> mel bin 0 (lowest freq) at the image bottom.
        mpimg.imsave(os.path.join(mel_dir, fname), tile, cmap="magma",
                     vmin=0.0, vmax=1.0, origin="lower")
        tiles.append(fname)
        widths.append(int(b - a))

    meta = {
        "n_frames": int(n_frames),
        "n_mels": int(N_MELS),
        "tile_frames": int(tw),
        "n_tiles": int(n_tiles),
        "tiles": tiles,
        "tile_widths": widths,
        "frame_rate": float(sr) / float(hop),
        "sr": int(sr),
        "hop": int(hop),
        "scale": {"p_low": params.mel_p_low, "p_high": params.mel_p_high,
                  "p_low_val": p_lo, "p_high_val": p_hi},
        "y_axis": ("tiles use origin=lower: image row 0 (top) = mel bin n_mels-1 "
                   "(highest); pixel_row(bin) = (n_mels-1) - bin"),
        "mel_bin_of_midi": mel_bin_of_midi(),
    }
    from .library import atomic_write_json
    atomic_write_json(os.path.join(mel_dir, MEL_META), meta)
    return meta


def time_stretch(y: np.ndarray, speed: float, n_fft: int) -> np.ndarray:
    """Pitch-preserving phase-vocoder stretch. ``speed`` < 1 slows (longer)."""
    import librosa
    return librosa.effects.time_stretch(
        np.asarray(y, dtype=np.float32), rate=float(speed), n_fft=int(n_fft)).astype(np.float32)


def make_stretches(variants: dict[str, np.ndarray], stretch_dir: str,
                   params: MediaParams | None = None, sr: int = SR) -> list[str]:
    """Write pitch-preserving stretches for each ``{variant_name: waveform}``.

    Returns the list of written filenames (relative to ``stretch_dir``).
    """
    params = params or MediaParams()
    os.makedirs(stretch_dir, exist_ok=True)
    written = []
    for variant, y in variants.items():
        for speed, nfft in zip(params.stretch_speeds, params.stretch_nfft):
            out = time_stretch(y, speed, nfft)
            fname = stretch_name(variant, speed)
            write_pcm16(os.path.join(stretch_dir, fname), out, sr)
            written.append(fname)
    return written


def stretch_paths(params: MediaParams | None = None) -> list[str]:
    """Expected stretch filenames for both variants (for existence checks)."""
    params = params or MediaParams()
    return [stretch_name(v, s) for v in ("source", "cleaned") for s in params.stretch_speeds]
