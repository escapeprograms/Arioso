"""BigVGAN-v2 vocoder wrapper for the Arioso project.

Thin loader around NVIDIA's BigVGAN-v2 neural vocoder — the project's chosen
mel-spectrogram -> waveform model. Every mel parameter is sourced from
``common.config`` so the mel contract lives in exactly one place and is checked,
at load time, against the checkpoint (``nvidia/bigvgan_v2_44khz_128band_512x``).

**Reuse this — do not re-implement mel extraction or model loading.** Mels are
computed by BigVGAN's own ``meldataset.mel_spectrogram`` (not a re-implemented
STFT) so they can never drift from what the checkpoint was trained on.

BigVGAN ships no PyPI package; the repo is vendored under ``external/BigVGAN``
and added to ``sys.path`` here. Weights download from the HF Hub on first load
and are cached thereafter.

Self-test (round-trips a real clip mel -> waveform):

    python -m common.vocoder --selftest
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

from common.config import SR, HOP_SIZE, N_FFT, WIN_SIZE, N_MELS, FMIN, FMAX

# The 44.1 kHz / 128-band / hop-512 checkpoint — the exact match for our config.
CHECKPOINT = "nvidia/bigvgan_v2_44khz_128band_512x"

# Vendored BigVGAN clone (no PyPI package); add to path for ``import bigvgan``.
_BIGVGAN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "external", "BigVGAN",
)


def _ensure_bigvgan_on_path() -> None:
    if _BIGVGAN_DIR not in sys.path:
        sys.path.insert(0, _BIGVGAN_DIR)


def load_vocoder(device: str = "cpu", use_cuda_kernel: bool = False):
    """Load the BigVGAN-v2 checkpoint, asserting it matches ``common.config``.

    Downloads the weights from the HF Hub on first call (cached thereafter).
    ``use_cuda_kernel`` selects the fused anti-aliased-activation CUDA kernel; we
    default to ``False`` (the PyTorch-native path) since it needs no build step.

    We download config + weights directly rather than via BigVGAN's
    ``from_pretrained`` mixin: that mixin's ``_from_pretrained`` signature is
    pinned to an older ``huggingface_hub`` contract (it requires ``proxies`` /
    ``resume_download`` kwargs that newer hub releases no longer pass), so calling
    it breaks on modern hub. This path replicates the same load and is
    version-independent.
    """
    import json

    from huggingface_hub import hf_hub_download

    _ensure_bigvgan_on_path()
    import bigvgan
    from env import AttrDict

    config_file = hf_hub_download(repo_id=CHECKPOINT, filename="config.json")
    with open(config_file) as f:
        h = AttrDict(json.load(f))

    # Fail loudly if the checkpoint's mel params drift from our shared contract.
    expected = {
        "sampling_rate": SR, "hop_size": HOP_SIZE, "n_fft": N_FFT,
        "win_size": WIN_SIZE, "num_mels": N_MELS, "fmin": FMIN, "fmax": FMAX,
    }
    mismatch = {
        k: (h.get(k), v) for k, v in expected.items() if h.get(k) != v
    }
    if mismatch:
        raise ValueError(
            f"BigVGAN checkpoint {CHECKPOINT} mel params disagree with "
            f"common.config (got, expected): {mismatch}"
        )

    model = bigvgan.BigVGAN(h, use_cuda_kernel=use_cuda_kernel)

    weight_file = hf_hub_download(
        repo_id=CHECKPOINT, filename="bigvgan_generator.pt")
    ckpt = torch.load(weight_file, map_location="cpu")

    # The freshly built model carries weight norm; the checkpoint may or may not.
    # Either way, end up weight-norm-free (folded) for inference.
    try:
        model.load_state_dict(ckpt["generator"])
        model.remove_weight_norm()
    except RuntimeError:
        model.remove_weight_norm()
        model.load_state_dict(ckpt["generator"])

    return model.eval().to(device)


def mel_spectrogram(wav, device: str = "cpu") -> torch.Tensor:
    """Mel-spectrogram matching the vocoder, via BigVGAN's own meldataset fn.

    ``wav`` is a 1-D float waveform in [-1, 1] at ``SR`` (numpy array or tensor).
    Returns a ``[1, N_MELS, frames]`` tensor.
    """
    _ensure_bigvgan_on_path()
    from meldataset import mel_spectrogram as _bigvgan_mel

    if not torch.is_tensor(wav):
        wav = torch.from_numpy(np.asarray(wav, dtype=np.float32))
    wav = wav.float().to(device)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # [1, T]
    return _bigvgan_mel(wav, N_FFT, N_MELS, SR, HOP_SIZE, WIN_SIZE, FMIN, FMAX)


def vocode(model, mel: torch.Tensor) -> np.ndarray:
    """Run the vocoder: ``[1, N_MELS, frames]`` mel -> 1-D float waveform."""
    device = next(model.parameters()).device
    with torch.no_grad():
        wav = model(mel.to(device))  # [1, 1, T]
    return wav.squeeze().cpu().numpy()


def _selftest() -> None:
    import argparse
    import glob

    from common.audio_io import load_mono, write_pcm16

    ap = argparse.ArgumentParser(
        description="BigVGAN-v2 mel->waveform round-trip self-test.")
    ap.add_argument("--selftest", action="store_true",
                    help="(accepted for symmetry; the script always self-tests)")
    ap.add_argument("--input", default=None,
                    help="wav to round-trip (default: first clip under Data/gt)")
    ap.add_argument("--out", default="bigvgan_selftest_out.wav",
                    help="path for the reconstructed wav")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    src = args.input or next(iter(sorted(glob.glob("Data/gt/*.wav"))), None)
    if src is None:
        raise SystemExit("no input wav found (pass --input PATH)")

    print(f"device={args.device}  input={src}")
    y = load_mono(src)                       # mono float32 @ SR
    y = y[: SR * 4]                          # cap to ~4 s for a quick test
    model = load_vocoder(device=args.device)
    mel = mel_spectrogram(y, device=args.device)
    print(f"mel shape: {tuple(mel.shape)}  (expect [1, {N_MELS}, frames])")
    rec = vocode(model, mel)

    expected_len = mel.shape[-1] * HOP_SIZE
    print(f"in samples={len(y)}  out samples={len(rec)}  "
          f"mel_frames*HOP={expected_len}")
    write_pcm16(args.out, rec.astype(np.float32), SR)
    print(f"wrote {args.out}  ->  listen to confirm it matches the input")


if __name__ == "__main__":
    _selftest()
