import numpy as np
import librosa


def load_audio(file_path: str, sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    y, sample_rate = librosa.load(file_path, sr=sr, mono=mono)
    return y, sample_rate

