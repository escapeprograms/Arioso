"""Pitch-shifted mel extraction (DiffSinger nvSTFT "keyshift"), for pitch augmentation.

Transposing training audio the obvious way — resample the waveform, re-mel it — is
both slow (a high-quality resampler per augmented clip) and *duration-changing*, so
the augmented mel no longer lines up frame-for-frame with the conditioning tracks
(onsets, envelopes, articulation) rasterized from the score. DiffSinger's trick,
reproduced here, does the shift **inside the STFT** instead:

    run the STFT with a window/FFT of ``WIN_SIZE * 2**(cents/1200)`` while keeping
    ``HOP_SIZE`` fixed, then read the first ``N_FFT//2+1`` bins as if they were the
    ordinary 2048-point bins.

Bin ``i`` of a ``win_new``-point FFT sits at ``i*SR/win_new`` Hz but the standard mel
basis reads it as ``i*SR/N_FFT`` Hz, so every partial lands a factor
``win_new/N_FFT == 2**(cents/1200)`` higher — a pitch shift, with the frame grid
untouched. Cost is one STFT: no resampling, no realignment, and the *same* number of
frames for every ``cents`` (see the padding note in :func:`mel_frames_keyshift`).

**Trap — never route this through the vendored BigVGAN mel.** ``external/BigVGAN/
meldataset.mel_spectrogram`` caches the mel basis *and* the Hann window under one key
built from ``n_fft`` (``meldataset.py`` ~lines 86/93). Calling it with a scaled
``n_fft`` would therefore also build a mel basis for that scaled ``n_fft`` — exactly
the wrong thing, since the whole mechanism depends on the basis staying at
``N_FFT=2048`` while the window scales. That vendored file must never be modified
either; this module re-implements its (small, fixed) math instead, and
:func:`mel_frames_keyshift` at ``cents == 0`` is asserted bit-identical to
:func:`common.vocoder.mel_frames` so the re-implementation can never drift.

Everything here is pure DSP + integer math: CPU-only, torch-free at the API surface
(numpy in, numpy out), safe to call from dataloader workers.
"""

from __future__ import annotations

import math

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from common.config import SR, HOP_SIZE, N_FFT, WIN_SIZE, N_MELS, FMIN, FMAX

# --- caches -----------------------------------------------------------------------
# The mel basis is keyed by DEVICE ONLY: it is always the standard N_FFT=2048 basis,
# never a scaled one (that is the whole point — see the module docstring). The Hann
# window is the only thing that varies with ``cents``, so it gets its own cache keyed
# by (length, device). Both mirror the vendored construction exactly.
_MEL_BASIS: dict[str, torch.Tensor] = {}
_HANN: dict[tuple[int, str], torch.Tensor] = {}


def _mel_basis(device: str = "cpu") -> torch.Tensor:
    """The standard ``[N_MELS, N_FFT//2+1]`` slaney mel basis (cached per device)."""
    if device not in _MEL_BASIS:
        mel = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
        _MEL_BASIS[device] = torch.from_numpy(mel).float().to(device)
    return _MEL_BASIS[device]


def _hann(win: int, device: str = "cpu") -> torch.Tensor:
    """A ``win``-point periodic Hann window (cached), as ``torch.hann_window`` builds."""
    key = (win, device)
    if key not in _HANN:
        _HANN[key] = torch.hann_window(win).to(device)
    return _HANN[key]


# --- shift geometry ---------------------------------------------------------------

def keyshift_factor(cents: float) -> float:
    """Frequency multiplier for a shift of ``cents``; positive ``cents`` shifts UP."""
    return 2.0 ** (float(cents) / 1200.0)


def scaled_win(cents: float) -> int:
    """Window / FFT length that realizes a ``cents`` shift under the standard mel basis.

    ``N_FFT == WIN_SIZE == 2048``, so this one value serves as both ``n_fft`` and
    ``win_length`` for the shifted STFT.
    """
    return int(round(WIN_SIZE * keyshift_factor(cents)))


# --- the shifted mel --------------------------------------------------------------

def mel_frames_keyshift(wav, cents: float) -> np.ndarray:
    """Log-mel of ``wav`` pitch-shifted by ``cents``, as ``[N_MELS, T]`` float32 numpy.

    ``wav`` is a 1-D float waveform in [-1, 1] at ``SR`` (numpy array or tensor);
    positive ``cents`` shifts up. Duration-preserving: the number of frames is

        ``T = 1 + (len(wav) - HOP_SIZE) // HOP_SIZE``

    for **every** ``cents``, because the pre-STFT reflect pad is sized from the scaled
    window (``pad_left + pad_right == win_new - HOP_SIZE``) rather than held at the
    fixed 768/768. So the padded length is always ``len(wav) + win_new - HOP_SIZE``
    and the ``center=False`` frame count cancels ``win_new`` out entirely — the
    augmented mel stays frame-aligned with score-rasterized conditioning tracks.

    At ``cents == 0.0`` this is **bit-identical** to :func:`common.vocoder.mel_frames`
    (``np.array_equal``, not merely close): ``win_new`` collapses to 2048, the pad to
    the vendored 768/768, the bin crop/pad to a no-op and the ``WIN_SIZE/win_new``
    rescale to an exact ``* 1.0``. Pinned by ``Arioso/tests/test_keyshift.py``.

    Known artifact, for ``cents < 0`` only (``win_new < 2048``): the shifted STFT has
    fewer than ``N_FFT//2+1`` bins, so the missing top rows are zero-filled. Mel bin
    127 loses all of its support and sits on the ``log(1e-5) == -11.5129`` floor, bin
    126 is mildly attenuated, and bins <= 125 are unaffected for ``|cents| <= 100``.
    Harmless for a 44.1 kHz violin corpus (mel bin 127 is ~21-22 kHz), but do not
    widen the shift range without re-measuring it.
    """
    win_new = scaled_win(cents)
    device = "cpu"

    if not torch.is_tensor(wav):
        wav = torch.from_numpy(np.asarray(wav, dtype=np.float32))
    y = wav.detach().float().to(device).reshape(1, -1)   # [1, L]

    with torch.no_grad():
        # Pad so the padded length is L + win_new - HOP_SIZE, i.e. T is cents-invariant.
        pad_left = (win_new - HOP_SIZE) // 2
        pad_right = (win_new - HOP_SIZE) - pad_left
        # torch's reflect pad requires pad < dim; short signals fall back to zero pad.
        mode = "reflect" if max(pad_left, pad_right) < y.shape[-1] else "constant"
        y = F.pad(y.unsqueeze(1), (pad_left, pad_right), mode=mode).squeeze(1)

        spec = torch.stft(
            y,
            win_new,
            hop_length=HOP_SIZE,
            win_length=win_new,
            window=_hann(win_new, device),
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

        # A longer window spreads the same energy over more bins (and vice versa);
        # undo that so levels match the unshifted mel. Exactly 1.0 at cents == 0.
        spec = spec * (WIN_SIZE / win_new)

        # Read the low bins as if they were the standard 2048-point bins: crop when
        # the window grew (shift up), zero-fill the absent top rows when it shrank.
        n_bins = N_FFT // 2 + 1
        if spec.shape[1] > n_bins:
            spec = spec[:, :n_bins, :]
        elif spec.shape[1] < n_bins:
            spec = F.pad(spec, (0, 0, 0, n_bins - spec.shape[1]))

        mel = torch.matmul(_mel_basis(device), spec)
        mel = torch.log(torch.clamp(mel, min=1e-5))

    return np.ascontiguousarray(mel[0].numpy().astype(np.float32))


# --- slice-read geometry (for windowed reads of long files) -----------------------

def margin_frames(max_cents: float) -> int:
    """Frames of context a shifted slice needs on each side to match a whole-file mel.

    The scaled window reaches ``pad_left_max = (scaled_win(|max_cents|) - HOP_SIZE)//2``
    samples back from the first frame's start; ``ceil(pad_left_max / HOP_SIZE) + 1``
    frames covers that plus the frame the window itself spans. 3 at the shipped
    +/-100 cents range (and 3 already at 0, since 768/512 = 1.5 -> 2, +1).
    """
    pad_left_max = (scaled_win(abs(float(max_cents))) - HOP_SIZE) // 2
    return math.ceil(pad_left_max / HOP_SIZE) + 1


def slice_read_range(start: int, end: int, margin: int) -> tuple[int, int, int]:
    """Sample range to read for frames ``[start, end)`` plus ``margin`` context frames.

    Returns ``(read_start_sample, read_n_samples, head_frames)``. Pure integer math —
    no I/O, no file length needed.

    ``read_start_sample`` is always a multiple of ``HOP_SIZE``, so melling the slice
    puts whole-file frame ``f0 + j`` at slice frame ``j``: the caller keeps
    ``mel[:, head_frames : head_frames + (end - start)]``.

    Both edge cases are **exact**, not approximate. Near frame 0 the head clamp
    (``head_frames < margin``) leaves the slice starting at the true file start, where
    its own reflect pad *is* the whole-file reflect pad; likewise a short read at EOF
    ends at the true file end. Only interior cuts need the margin, and there the
    discarded ``margin`` frames absorb the difference.
    """
    f0 = max(0, start - margin)
    read_start_sample = f0 * HOP_SIZE
    read_n_samples = (end + margin - f0) * HOP_SIZE
    head_frames = start - f0
    return read_start_sample, read_n_samples, head_frames
