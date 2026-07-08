"""Classify the violin playing technique of every score note with pretrained VioPTT.

Arioso conditions on *what technique* each note is played with (flageolet, pizzicato,
spiccato, …). VioPTT (paper *Violin Playing Technique Detection*, its repo vendored under
``external/VioPTT``) ships a pretrained note-level classifier that, given a per-note audio
slice, predicts one of five technique classes. This module is the thin wrapper that lets the
DataSynthesizer build reuse VioPTT's own inference code instead of reimplementing it, plus the
pure (torch-free) helpers that turn a MIDI score into per-note slice spans and expand VioPTT's
per-note predictions into a per-frame id track aligned to the ``target_mel`` grid.

**Two note sources, kept bit-identical.** The onset frames we emit MUST equal the
``onsets_arioso`` array built by :mod:`DataSynthesizer.build_prior` (the clip enumerator keys
off it). So :func:`note_groups_from_midi` parses notes with **pretty_midi** and rounds onsets
with **np.round**, exactly as ``build_prior`` does — NOT VioPTT's raw-MIDI-event parser
(``parse_midi_to_note_events``), which would drift. VioPTT is used only for the audio→technique
step; note timing is owned by us.

**Mandatory transcription features.** The shipped VioPTT note model was trained WITH
transcription features, so it MUST be constructed with ``use_trans_features=True`` /
``trans_feat_dim=352`` and fed features from the transcriptor checkpoint (which is therefore
also mandatory — a missing transcriptor is fatal). Constructing without trans-features makes
``load_state_dict`` fail on a shape mismatch.

**Fixed 2 s training window (the label-quality fix).** VioPTT's note model was TRAINED on a
single note sliced from its onset, then pad/truncated to a fixed 2.0 s (32000-sample) window and
peak-normalized — see ``RWCTechNoteWavDataset`` / ``MOSAPTNoteDataset`` in
``external/VioPTT/piano_transcription/utils/data_generator.py`` (``segment_seconds=2.0`` /
``fixed_seconds`` -> ``pad_truncate_sequence`` at utils/utilities.py:84-88, then
``wav /= max(abs)+1e-8``); ``--mosapt_fixed_seconds`` defaults to 2.0 in
``pytorch/main_note_technique.py`` and the transcription features are computed on that SAME padded
window (main_note_technique.py:198,211). Their shipped from-midi inference
(``infer_note_technique_from_midi.py``) instead slices VARIABLE-length chunks and pads only to the
batch max (``infer_batch``), with NO peak-normalization — an off-distribution input that made fast
short notes (0.15 s) read as short/bouncy → blanket ``spiccato``. :meth:`prepare_chunks`
reproduces the training construction so :meth:`classify` feeds in-distribution windows; the legacy
variable-length behavior stays reachable via ``note_seconds=None`` for A/B comparison.

**Deliberate namespace pollution.** VioPTT's modules use *flat* imports (``import config``,
``from models_contrast import …``, ``from utilities import …``), so importing them binds the
top-level names ``config`` / ``utilities`` / ``models_contrast`` / ``pytorch_utils`` to VioPTT's
copies for the rest of the process. This is inert for the repo: our code only ever imports via
package-qualified paths (``DataSynthesizer.config``, ``common.…``). Tripwire asserts after the
import catch any sys.path shadowing (e.g. a stray top-level ``config``) or upstream drift.
"""

from __future__ import annotations

import os
import sys
from typing import NamedTuple

import numpy as np

from .config import (HOP, SR, TECHNIQUE_CLASSES, TECHNIQUE_PAD_S,
                     TECHNIQUE_REST_SNAP_FRAMES, REST_ID,
                     TECHNIQUE_NOTE_SECONDS, TECHNIQUE_PEAK_NORMALIZE)
from .synthesizePrior import note_onsets  # noqa: F401 - re-exported / shared MIDI source

# VioPTT operates at 16 kHz mono (its config.sample_rate); the repo's canonical SR is 44.1 kHz,
# so audio is resampled on load (common.audio_io.load_mono(sr=VIOPTT_SR)).
VIOPTT_SR = 16000

# Vendored VioPTT clone (no PyPI package). Both its ``pytorch`` and ``utils`` dirs must be on
# sys.path because its modules import each other flatly (see module docstring).
_VIOPTT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "external", "VioPTT",
)
_VIOPTT_PYTORCH = os.path.join(_VIOPTT_ROOT, "piano_transcription", "pytorch")
_VIOPTT_UTILS = os.path.join(_VIOPTT_ROOT, "piano_transcription", "utils")
NOTE_CKPT = os.path.join(_VIOPTT_ROOT, "checkpoints", "note_tech_model.pth")
TRANSCRIPTOR_CKPT = os.path.join(_VIOPTT_ROOT, "checkpoints", "transcriptor_model.pth")

# The note model was trained with transcription features = 4 heads x 88 piano classes.
_TRANS_FEAT_DIM = 88 * 4  # == 352


def _ensure_vioptt_on_path() -> None:
    """Idempotently put VioPTT's ``pytorch`` + ``utils`` dirs at the front of sys.path."""
    for d in (_VIOPTT_UTILS, _VIOPTT_PYTORCH):  # pytorch ends up first (inserted last)
        if d not in sys.path:
            sys.path.insert(0, d)


def _import_vioptt():
    """Lazily import the VioPTT pieces we reuse, returning them in a namespace-ish dict.

    Deferred so merely importing this module (e.g. for the pure helpers) neither drags in torch
    nor pollutes the top-level namespace. The tripwire asserts guard against sys.path shadowing
    (a stray top-level ``config``) and upstream drift in the class list / sample rate.
    """
    _ensure_vioptt_on_path()
    import config  # VioPTT's utils/config.py (flat import)
    from infer_note_technique_from_midi import (DEFAULT_TECHNIQUE_CLASSES,
                                                build_transcriptor, infer_batch,
                                                load_note_model, slice_waveform_by_notes)

    assert config.sample_rate == VIOPTT_SR, (
        f"VioPTT config.sample_rate={config.sample_rate} != {VIOPTT_SR}; sys.path may be "
        "shadowing a different 'config' module")
    assert tuple(DEFAULT_TECHNIQUE_CLASSES) == TECHNIQUE_CLASSES[:5], (
        f"VioPTT technique classes {DEFAULT_TECHNIQUE_CLASSES} drifted from "
        f"{TECHNIQUE_CLASSES[:5]}; update DataSynthesizer.config.TECHNIQUE_CLASSES")
    return {
        "config": config,
        "load_note_model": load_note_model,
        "build_transcriptor": build_transcriptor,
        "slice_waveform_by_notes": slice_waveform_by_notes,
        "infer_batch": infer_batch,
    }


# --- pure helpers (no torch) ------------------------------------------------------

class NoteGroup(NamedTuple):
    """One distinct rounded-onset frame's worth of notes (chords / sub-frame onsets merged).

    ``onset_frame`` keys the group and — across all groups — must equal ``onsets_arioso``.
    ``start_s``/``end_s`` are the min-onset / max-offset *seconds* of the merged notes, fed to
    VioPTT as the slice span; ``end_frame`` is the mel frame that span ends on (clamped to at
    least ``onset_frame + 1`` and at most ``n_frames``), used only by the per-frame expansion.
    """
    onset_frame: int
    end_frame: int
    start_s: float
    end_s: float
    n_notes: int


def note_groups_from_midi(midi_path: str, offset_s: float, n_frames: int,
                          sr: int = SR, hop: int = HOP) -> list[NoteGroup]:
    """Parse ``midi_path`` into onset-frame-grouped note spans, aligned like ``onsets_arioso``.

    Uses the SAME note source as :func:`DataSynthesizer.synthesizePrior.note_onsets`
    (pretty_midi ``note.start``/``note.end`` across all instruments — the dataset puts each
    note-stream on its own instrument track, so no filtering) and the SAME rounding as
    ``build_prior`` (``np.round(t * sr / hop)``, then drop frames outside ``[0, n_frames)``), so
    ``[g.onset_frame for g in groups]`` is bit-identical to the saved onsets array. Notes sharing
    a rounded onset frame (chords, double-stops, sub-frame-distinct onsets) merge into one group:
    ``start_s`` = earliest onset, ``end_s`` = latest offset, ``n_notes`` = how many merged.
    """
    import pretty_midi  # local: keep the module import-light and consistent with note_onsets

    pm = pretty_midi.PrettyMIDI(midi_path)
    groups: dict[int, dict] = {}
    for inst in pm.instruments:
        for note in inst.notes:
            start = note.start + offset_s
            end = note.end + offset_s
            onset_frame = int(np.round(start * sr / hop))
            if onset_frame < 0 or onset_frame >= n_frames:
                continue  # same range filter build_prior applies to the onset array
            g = groups.get(onset_frame)
            if g is None:
                groups[onset_frame] = {"start_s": start, "end_s": end, "n_notes": 1}
            else:
                g["start_s"] = min(g["start_s"], start)
                g["end_s"] = max(g["end_s"], end)
                g["n_notes"] += 1

    out: list[NoteGroup] = []
    for onset_frame in sorted(groups):
        g = groups[onset_frame]
        end_frame = int(np.clip(np.round(g["end_s"] * sr / hop), onset_frame + 1, n_frames))
        out.append(NoteGroup(onset_frame=onset_frame, end_frame=end_frame,
                             start_s=g["start_s"], end_s=g["end_s"], n_notes=g["n_notes"]))
    return out


def expand_to_frames(groups: list[NoteGroup], tech_ids: np.ndarray, n_frames: int,
                     rest_snap: int = TECHNIQUE_REST_SNAP_FRAMES) -> np.ndarray:
    """Expand per-group technique ids into a ``[n_frames]`` uint8 per-frame id track.

    Frames start all-``REST_ID`` (so pre-first-onset frames stay rest automatically). For each
    group we fill from its onset frame forward with that group's technique id. The end of the
    fill uses a **hybrid rest policy** against the next onset (or ``n_frames`` for the last
    group): if the gap between this note's ``end_frame`` and the next onset is small
    (``<= rest_snap`` frames — a legato transition / rounding slack), the note is extended to
    the next onset so no spurious rest sliver appears; otherwise the note stops at ``end_frame``
    and the real gap ``[end_frame, next_onset)`` stays rest.
    """
    ids = np.full(n_frames, REST_ID, dtype=np.uint8)
    n = len(groups)
    for i, g in enumerate(groups):
        next_boundary = groups[i + 1].onset_frame if i + 1 < n else n_frames
        fill_end = next_boundary if (next_boundary - g.end_frame) <= rest_snap else g.end_frame
        fill_end = min(fill_end, n_frames)
        if fill_end > g.onset_frame:
            ids[g.onset_frame:fill_end] = tech_ids[i]
    return ids


# --- torch wrapper ----------------------------------------------------------------

class TechniqueClassifier:
    """Pretrained VioPTT note-technique model + its mandatory transcriptor, ready to classify.

    Loads both checkpoints once (construct ONE instance and reuse it across clips). ``classify``
    takes a 16 kHz waveform and a list of per-note ``(start_s, end_s)`` spans and returns the
    predicted class id + softmax confidence for each. By default it reconstructs VioPTT's
    training-time note window (fixed :data:`TECHNIQUE_NOTE_SECONDS` seconds, onset-anchored, end
    zero-padded, peak-normalized — see :meth:`prepare_chunks`) so the model sees in-distribution
    inputs; pass ``note_seconds=None`` for the legacy variable-length slicing. Both the
    note-technique checkpoint AND the transcriptor checkpoint are required (the model needs
    transcription features); a missing file — or a transcriptor that fails to build — raises
    ``FileNotFoundError``.
    """

    def __init__(self, device: str = "cuda", note_ckpt: str = NOTE_CKPT,
                 transcriptor_ckpt: str = TRANSCRIPTOR_CKPT):
        import torch

        v = _import_vioptt()
        self._v = v
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        if not os.path.isfile(note_ckpt):
            raise FileNotFoundError(f"VioPTT note-technique checkpoint not found: {note_ckpt}")
        if not os.path.isfile(transcriptor_ckpt):
            raise FileNotFoundError(
                f"VioPTT transcriptor checkpoint not found: {transcriptor_ckpt} "
                "(the note model was trained WITH transcription features, so it is mandatory)")

        fps = v["config"].frames_per_second  # 100
        self.note_model = v["load_note_model"](
            note_ckpt, self.device, frames_per_second=fps,
            use_trans_features=True, trans_feat_dim=_TRANS_FEAT_DIM)
        self.transcriptor = v["build_transcriptor"](
            self.device, fps, classes_num=v["config"].classes_num,
            checkpoint_path=transcriptor_ckpt)
        if self.transcriptor is None:  # build_transcriptor returns None on a missing/bad ckpt
            raise FileNotFoundError(
                f"VioPTT transcriptor failed to load from {transcriptor_ckpt}")

    def prepare_chunks(self, wav_16k: np.ndarray, spans_s: list[tuple[float, float]],
                       pad_s: float = TECHNIQUE_PAD_S,
                       note_seconds: float | None = TECHNIQUE_NOTE_SECONDS,
                       peak_normalize: bool = TECHNIQUE_PEAK_NORMALIZE) -> list[np.ndarray]:
        """Build the EXACT per-note waveform arrays fed to the model — one float32 array per span.

        This reproduces VioPTT's *training-time* note construction (so the notebook can audition
        precisely what the model hears):

        * Onset-anchored slice via VioPTT's own ``slice_waveform_by_notes`` (``pad_s`` seconds of
          context on each side; identical to their inference slicing).
        * If ``note_seconds`` is set (default :data:`TECHNIQUE_NOTE_SECONDS` = 2.0), pad/truncate
          the slice to ``round(note_seconds * VIOPTT_SR)`` samples with VioPTT's
          ``pad_truncate_sequence`` policy: shorter → **zeros appended at the end** (note stays at
          the front); longer → **kept from the start** (first N samples). This is the fixed
          32000-sample window the RWC/MOSAPT note datasets train on.
        * If ``peak_normalize`` (default True), scale each window by ``1 / (max|x| + 1e-8)``,
          exactly as those datasets do AFTER pad/truncate.

        ``note_seconds=None`` returns the raw variable-length slices (legacy behavior, for A/B).
        Empty ``spans_s`` → empty list. All arrays are contiguous float32.
        """
        if not spans_s:
            return []
        notes = [{"onset_time": float(s), "offset_time": float(e)} for s, e in spans_s]
        chunks, _ = self._v["slice_waveform_by_notes"](
            wav_16k, VIOPTT_SR, notes, pad_seconds=pad_s)

        seg_samples = int(round(note_seconds * VIOPTT_SR)) if note_seconds else None
        out: list[np.ndarray] = []
        for c in chunks:
            x = np.asarray(c, dtype=np.float32)
            if seg_samples is not None:
                # VioPTT's pad_truncate_sequence (utils/utilities.py:84-88): end-pad or head-crop.
                if x.shape[0] < seg_samples:
                    x = np.concatenate([x, np.zeros(seg_samples - x.shape[0], dtype=np.float32)])
                else:
                    x = x[:seg_samples]
            if peak_normalize and x.size > 0:
                x = x / (np.max(np.abs(x)) + 1e-8)
            out.append(np.ascontiguousarray(x, dtype=np.float32))
        return out

    def classify(self, wav_16k: np.ndarray, spans_s: list[tuple[float, float]],
                 pad_s: float = TECHNIQUE_PAD_S,
                 note_seconds: float | None = TECHNIQUE_NOTE_SECONDS,
                 peak_normalize: bool = TECHNIQUE_PEAK_NORMALIZE) -> tuple[np.ndarray, np.ndarray]:
        """Classify one note per span. Returns ``(ids [K] int64, conf [K] float32)``.

        ``spans_s`` are ``(onset_s, offset_s)`` per note. Chunks are built by
        :meth:`prepare_chunks` (fixed :data:`TECHNIQUE_NOTE_SECONDS` window + peak-normalize by
        default; ``note_seconds=None`` for legacy variable-length) and run through VioPTT's
        ``infer_batch``. With a fixed window every chunk has the same length, so ``infer_batch``'s
        batch-max zero-padding is a no-op and the model sees exactly the training-time input.
        Empty ``spans_s`` → two empty arrays.
        """
        if not spans_s:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        chunks = self.prepare_chunks(wav_16k, spans_s, pad_s=pad_s,
                                     note_seconds=note_seconds, peak_normalize=peak_normalize)
        ids, conf = self._v["infer_batch"](
            self.note_model, self.device, chunks, self.transcriptor,
            use_trans_features=True, trans_features_list=None)
        return ids.astype(np.int64), conf.astype(np.float32)
