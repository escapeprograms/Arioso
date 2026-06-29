"""Download and trim the ground-truth violin audio for a dataset MIDI clip.

The violin-transcription dataset ships MIDI only; the audio lives on YouTube.
The MIDI filename encodes the video id and the clip's [start, end] in seconds
(see ``clip_name.py`` for the convention). We download a whole video's audio
once (cached per id), then slice each clip out, resample to 44.1 kHz mono, and
write a GT wav whose timeline matches the (already time-aligned) MIDI. The
download approach mirrors ``PretrainedModel.download_youtube`` in
``external/violin-transcription/musc/model.py``.

Requires the ``yt_dlp`` Python package and ``ffmpeg`` on PATH.
"""

from __future__ import annotations

import os

import librosa
import numpy as np

from common.audio_io import load_mono, voiced_rms_normalize, write_pcm16

from .clip_name import parse_clip_name
from .config import SR, TARGET_RMS_DBFS, VOICED_TOP_DB


def download_full_audio(youtube_id: str, cache_dir: str, sr: int = SR,
                        audio_codec: str = "wav") -> str:
    """Download a whole video's audio to ``cache_dir/{id}.{codec}``.

    Reuses the cached file if it already exists, so many clips cut from the same
    video only trigger one download. A *freshly* downloaded track is level-normalized
    (voiced-segment RMS) and re-saved in place — mono @ ``sr`` — **before** any clip
    is trimmed from it, so every clip inherits one consistent loudness across the
    different source channels. (Cache hits are assumed already normalized.)
    """
    from yt_dlp import YoutubeDL

    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"{youtube_id}.{audio_codec}")
    if os.path.isfile(out_path):
        return out_path

    url = f"https://www.youtube.com/watch?v={youtube_id}"
    ydl_opts = {
        "noplaylist": True,
        "quiet": True,
        "format": "bestaudio/best",
        "outtmpl": os.path.join(cache_dir, "%(id)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": audio_codec,
            "preferredquality": "192",
        }],
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    if not os.path.isfile(out_path):
        raise RuntimeError(f"download did not produce {out_path}")

    # Level-normalize the whole track once and re-save in place, so each clip
    # trimmed from it shares the same playing loudness.
    y = load_mono(out_path, sr)
    y = voiced_rms_normalize(y, sr, target_rms_dbfs=TARGET_RMS_DBFS,
                             top_db=VOICED_TOP_DB)
    write_pcm16(out_path, y, sr)
    return out_path


def fetch_clip(midi_path: str, cache_dir: str, out_path: str | None = None,
               sr: int = SR) -> tuple[str, np.ndarray, int]:
    """Download (cached) and trim the GT audio for one MIDI clip; write a wav.

    Returns ``(out_path, y, sr)`` where ``y`` is the trimmed clip (mono float @ sr),
    so callers can compute features without re-reading. ``len(y)`` is what
    ``synthesizePrior.render_prior`` should use as ``total_samples`` so the prior and
    GT line up exactly. The downloaded track is already level-normalized (see
    ``download_full_audio``), so the clip inherits that gain.
    """
    clip = parse_clip_name(midi_path)
    full = download_full_audio(clip.youtube_id, cache_dir, sr=sr)

    y, _ = librosa.load(full, sr=sr, mono=True,
                        offset=float(clip.start), duration=float(clip.end - clip.start))
    if out_path is None:
        out_path = os.path.splitext(midi_path)[0] + "_gt.wav"
    write_pcm16(out_path, y, sr)
    return out_path, y, sr


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Download + trim the GT violin audio for a dataset MIDI clip.")
    ap.add_argument("midi", help="path to the .mid file (name encodes id/times)")
    ap.add_argument("--cache-dir", default="data/_cache",
                    help="where to cache full-video audio (default: data/_cache)")
    ap.add_argument("-o", "--out", help="output wav path (default: <midi>_gt.wav)")
    ap.add_argument("--sr", type=int, default=SR, help="sample rate (Hz)")
    args = ap.parse_args()

    clip = parse_clip_name(args.midi)
    print(f"clip {clip.basename}: id={clip.youtube_id} "
          f"{clip.start}-{clip.end}s ({clip.end - clip.start}s)")
    path, y, sr = fetch_clip(args.midi, args.cache_dir, out_path=args.out, sr=args.sr)
    print(f"wrote {path}  ({len(y) / sr:.2f} s @ {sr} Hz)")


if __name__ == "__main__":
    main()
