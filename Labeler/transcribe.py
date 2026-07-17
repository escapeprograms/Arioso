"""MUSC violin-transcription wrapper: lazy model singleton -> note events.

Vendors MTG's MUSC model (``external/violin-transcription``) the same way
``DataSynthesizer.technique`` vendors VioPTT — the repo dir goes on ``sys.path``
so ``import musc`` resolves, and torch is imported *lazily* inside the functions
that need it (so importing this module for its dataclasses stays torch-free and
pytest collection is fast).

``PretrainedModel('violin')`` reads ``musc/violin.json`` (sr 44100, hop 256 =>
~172.27 fps) and loads ``musc/violin_model.pt``. The checkpoint is already
present; if it is ever absent and the gdown auto-download fails, ``get_model``
raises :class:`CheckpointMissing` with the manual Drive recipe (``external/`` is
gitignored). A dormant ``torch.load`` retry-shim (plain load, then
``weights_only=False`` on an ``UnpicklingError``) guards against the torch 2.6
``weights_only`` default without editing the vendored code.

``transcribe_notes`` runs the model forward pass (``model.predict``) and then
**decodes the posteriors itself** via :func:`decode_posteriors` — it does not use
MUSC's hardcoded 'spotify' decoder, whose 127.7 ms minimum note length silently
deleted most short notes. The three decode knobs (onset/frame thresholds, min
note length) come from ``config.TranscribeParams`` and are recall-biased because
the dataset MIDIs were score-aligned; everything else mirrors ``out2note``'s
'spotify' path exactly (infer_onsets, melodia_trick, default energy_tol, and
``model.get_pitch_bends`` on the frame-index tuples before the second-mapping).
It returns plain-Python note events (no numpy leaking into JSON):
``(start_s, end_s, pitch, amplitude, bend_cents[])`` sorted by ``(start_s,
pitch)``. Pitch-bend ints are converted to cents (x100/4096) so the vibrato
analysis can rerun off ``transcription.json`` with no GPU.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass, field

# GPU serialization: the model singleton is not safe for concurrent forward
# passes, so the transcribe stage holds this lock across jobs.
GPU_LOCK = threading.Lock()

_MUSC_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "external", "violin-transcription")
_CKPT = os.path.join(_MUSC_ROOT, "musc", "violin_model.pt")
_DRIVE_ID = "1FfUjC3usmZBoTxNT6rNVIcYw4pol4K1g"

_MODEL = None
_MODEL_LOCK = threading.Lock()


class CheckpointMissing(Exception):
    """The MUSC checkpoint is absent and could not be downloaded."""

    def __init__(self, cause: object = None):
        msg = (
            f"MUSC violin checkpoint not found at {_CKPT} and the automatic "
            f"gdown download failed.\n"
            f"Download it manually (external/ is gitignored):\n"
            f"  1) https://drive.google.com/uc?export=download&id={_DRIVE_ID}\n"
            f"  2) save as: {_CKPT}\n"
            f"  or:  gdown {_DRIVE_ID} -O \"{_CKPT}\"")
        if cause is not None:
            msg += f"\n(cause: {type(cause).__name__}: {cause})"
        super().__init__(msg)


def _ensure_musc_on_path() -> None:
    if _MUSC_ROOT not in sys.path:
        sys.path.insert(0, _MUSC_ROOT)


def _install_weights_only_shim(torch):
    """Wrap ``torch.load`` so a weights_only UnpicklingError retries full-unpickle.

    Returns a restore callable. The vendored MUSC code calls ``torch.load(path)``
    with no ``weights_only`` kwarg; under torch >= 2.6 the default flipped to
    ``True``, which can reject a legit full-object checkpoint. Dormant here (the
    load currently succeeds plain) but keeps us robust without touching musc.
    """
    import pickle
    orig = torch.load

    def _patched(*args, **kwargs):
        try:
            return orig(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - narrow to the weights_only failure
            if kwargs.get("weights_only") is False:
                raise
            if isinstance(e, pickle.UnpicklingError) or "weights_only" in str(e).lower():
                kwargs["weights_only"] = False
                return orig(*args, **kwargs)
            raise

    torch.load = _patched
    return lambda: setattr(torch, "load", orig)


def get_model(device: str | None = None):
    """Load (once) and return the MUSC ``PretrainedModel('violin')`` on ``device``.

    Thread-safe singleton. ``device`` defaults to cuda when available else cpu.
    The model prints verbose config lines on first construction (expected).
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        import torch
        _ensure_musc_on_path()
        ckpt_present = os.path.isfile(_CKPT)
        restore = _install_weights_only_shim(torch)
        try:
            from musc.model import PretrainedModel
            model = PretrainedModel("violin")
        except CheckpointMissing:
            raise
        except Exception as e:  # noqa: BLE001
            if not ckpt_present and not os.path.isfile(_CKPT):
                raise CheckpointMissing(e) from e
            raise
        finally:
            restore()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        _MODEL = model
        return _MODEL


@dataclass
class TranscribedNote:
    start_s: float
    end_s: float
    pitch: int
    amplitude: float
    bend_cents: list = field(default_factory=list)


@dataclass
class TranscriptionResult:
    notes: list
    model_fps: float
    sr: int


def decode_posteriors(out: dict, labeling_range, params) -> list:
    """Decode MUSC posteriors into frame+second note events (model-free, no bends).

    Mirrors ``Transcriber.out2note``'s 'spotify' path — note creation followed by
    the frame->second mapping — with three recall-biased knobs from ``params``
    (``onset_thresh`` / ``frame_thresh`` / ``min_note_len_ms``). Everything else is
    left at the vendored default: ``infer_onsets=True``, ``melodia_trick=True`` and
    ``spotify_create_notes``'s default ``energy_tol``. Pure numpy/scipy — no torch,
    GPU or model — so it is unit-testable on a synthetic posterior grid.

    ``out`` must carry ``note`` (n_frames, n_freqs), ``onset`` (same shape) and
    ``time`` (n_frames,). ``labeling_range`` is ``(note_low, note_high)`` as ints
    (``model.labeling.midi_centers[0]`` / ``[-1]``). The frame rate is recovered
    from ``time`` (``1/(time[1]-time[0]) == sr/hop``), so the frame minimum matches
    ``out2note``'s ``round(min_note_len_ms/1000 * sr/hop)`` exactly.

    Returns note events ``(start_frame, end_frame, start_s, end_s, pitch,
    amplitude)`` sorted by ``(start_s, pitch)``. The frame indices are kept so the
    caller can run the model's pitch-bend estimator on the raw frame tuples (as
    ``out2note`` does) before discarding them.
    """
    import numpy as np

    _ensure_musc_on_path()
    from musc.postprocessing import spotify_create_notes

    note_low, note_high = int(labeling_range[0]), int(labeling_range[1])
    times = np.asarray(out["time"], dtype=np.float64)
    fps = 1.0 / (float(times[1]) - float(times[0]))
    min_note_len = int(np.round(params.min_note_len_ms / 1000.0 * fps))

    frame_notes = spotify_create_notes(
        out["note"],
        out["onset"],
        onset_thresh=params.onset_thresh,
        frame_thresh=params.frame_thresh,
        min_note_len=min_note_len,
        infer_onsets=True,
        note_low=note_low,
        note_high=note_high,
        melodia_trick=True,
    )

    decoded = []
    for start_f, end_f, pitch, amplitude in frame_notes:
        sf, ef = int(start_f), int(end_f)
        decoded.append((sf, ef, float(times[sf]), float(times[ef]),
                        int(pitch), float(amplitude)))
    decoded.sort(key=lambda e: (e[2], e[4]))
    return decoded


def transcribe_notes(y, params, device: str | None = None) -> TranscriptionResult:
    """Transcribe a mono float waveform at 44.1 kHz into note events.

    ``y`` is a numpy float array already at ``common.config.SR`` (the model's own
    sample rate — no resampling). ``params`` is a ``config.TranscribeParams`` (the
    three decode knobs + ``batch_size``). Runs the model forward pass
    (``model.predict``; the caller holds :data:`GPU_LOCK`), decodes the posteriors
    itself via :func:`decode_posteriors`, then estimates per-note pitch bends on the
    frame-index tuples (``model.get_pitch_bends``, exactly as ``out2note`` does)
    before the second-mapping. Returns plain-Python note events sorted by
    ``(start_s, pitch)`` with pitch bends converted to cents.
    """
    import numpy as np

    model = get_model(device)
    y = np.asarray(y, dtype=np.float32)
    out = model.predict(y, params.batch_size)

    note_low = int(model.labeling.midi_centers[0])
    note_high = int(model.labeling.midi_centers[-1])
    decoded = decode_posteriors(out, (note_low, note_high), params)

    # Pitch bends run on the frame-index tuples, before the second-mapping, exactly
    # as out2note does (get_pitch_bends is order-preserving and 1:1 at the default
    # timing_refinement_range=0, so it aligns positionally with ``decoded``).
    frame_tuples = [(sf, ef, pitch, amp) for sf, ef, _ss, _es, pitch, amp in decoded]
    with_bends = model.get_pitch_bends(out["f0"], frame_tuples)

    fps = float(model.sr) / float(model.hop_length)
    notes: list[TranscribedNote] = []
    for (sf, ef, ss, es, pitch, amplitude), bend_ev in zip(decoded, with_bends):
        bend = bend_ev[4] if len(bend_ev) > 4 else None
        if bend is None:
            bend_cents: list = []
        else:
            bend_cents = (np.asarray(bend, dtype=np.float64) * 100.0 / 4096.0).tolist()
        notes.append(TranscribedNote(
            start_s=float(ss), end_s=float(es), pitch=int(pitch),
            amplitude=float(amplitude), bend_cents=[float(c) for c in bend_cents]))

    notes.sort(key=lambda n: (n.start_s, n.pitch))
    return TranscriptionResult(notes=notes, model_fps=fps, sr=int(model.sr))
