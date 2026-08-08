"""The standard dataset-root layout — the single source of truth every producer emits.

Two producers feed Arioso training: the **DataSynthesizer** (synthetic weak-label data
from score MIDIs) and the **Labeler** (genuine human ground truth). They must never
import each other, yet they must agree, byte-for-byte, on *where* artifacts live and
*how* the categorical conditioning tracks are encoded. This module is that contract:

* **Layout constants** (``DIR_*``, :func:`cond_dir`, ``MANIFEST_JSON``, ``SPLIT_JSON``,
  ``MANIFEST_SCHEMA_VERSION``) — the on-disk directory/file names of a standard root.
* **Vocab / encoding constants** (``ARTICULATIONS``, ``SIGNAL_NUM_CLASSES``,
  ``SIGNAL_REST_ID`` …) — the id alphabets Arioso's conditioning embeddings expect.
* :class:`NoteEvent` — the canonical per-note record every producer rasterizes from.
* :func:`frame_of` — THE one rounding rule (mel frame index of a time in seconds). The
  identity ``onsets == [g.onset_frame for g in note_groups(...)]`` is load-bearing for
  Arioso's clip enumeration, so onset frames, group onsets and every rasterizer round
  through this exact function.
* The note-grouping helpers (:func:`note_groups`, :func:`note_groups_from_midi`,
  :func:`expand_to_frames`) and the per-frame rasterizers (:func:`rasterize_articulation`
  / :func:`rasterize_velocity` / :func:`rasterize_vibrato`) plus the boundary-frame arrays
  (:func:`onset_frames` -> ``onsets/``, :func:`offset_frames` -> ``offsets/``) that Arioso's
  sinusoidal boundary-distance conditioning consumes.
* The manifest reader/writer (:func:`load_manifest`, :func:`write_manifest`) and the
  :class:`DatasetRoot` accessor that Arioso and eval consume a root through.

**Encodings** (every track is ``[T]`` uint8; ``frame = round(t * SR / HOP_SIZE)``):

* articulation: normal=0, slur=1, spiccato=2, detache=3, rest=4 (``num_classes=5``,
  pad=4); the CFG "unknown" row is 5, produced by the model, never written to disk.
* velocity: 0=rest, 1..127=MIDI velocity (``num_classes=128``, pad=0); unknown=128.
* vibrato: 0=off, 1=on, 2=rest (``num_classes=3``, pad=2); unknown=3 — rest kept
  distinct from "played without vibrato".

Deliberately **torch-free** (numpy + stdlib, plus ``pretty_midi`` only inside the
MIDI-parsing helper) so any producer can import it without dragging in a GPU stack.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from common.config import HOP_SIZE, N_MELS, SR

# --- standard root layout (directory / file names) ------------------------------

MANIFEST_JSON = "manifest.json"      # canonical per-root manifest (schema_version 1)
SPLIT_JSON = "split.json"            # train/val split (written by Arioso.splits, not producers)
DIR_GT = "gt"                        # <base>.wav mono PCM16 @ SR, voiced-RMS at TARGET_RMS_DBFS
DIR_TARGET_MEL = "target_mel"        # <base>.npy [N_MELS, T] float32 (common.vocoder mel)
DIR_PRIOR_MEL = "prior_mel"          # <base>.npy [N_MELS, T] float32, same T
DIR_ONSETS = "onsets"                # <base>.npy [K] int32 onset frame indices, sorted unique
DIR_OFFSETS = "offsets"              # <base>.npy [K] int32 note-offset frame indices, sorted unique
COND_ROOT = "cond"                   # per-signal conditioning tracks live under cond/<signal>/


def cond_dir(signal: str) -> str:
    """Relative dir holding a conditioning signal's ``[T]`` uint8 tracks (``cond/<signal>``)."""
    return f"{COND_ROOT}/{signal}"


MANIFEST_SCHEMA_VERSION = 1


# --- vocab / encoding constants -------------------------------------------------

# Unified articulation vocabulary = the Labeler vocab + rest. Ids are the tuple index;
# rest is the next id after the last named class. The CFG "unknown" row is one past rest.
ARTICULATIONS = ("normal", "slur", "spiccato", "detache")
ARTIC_REST_ID = 4                    # len(ARTICULATIONS): nothing sounding (gaps / lead-in / tail)
ARTIC_NUM_CLASSES = 5                # named classes + rest (embedding gets one more "unknown" row)

VELOCITY_REST_ID = 0                 # 0 = rest; 1..127 = MIDI velocity
VELOCITY_NUM_CLASSES = 128

VIBRATO_OFF = 0
VIBRATO_ON = 1
VIBRATO_REST_ID = 2                  # rest kept distinct from "played without vibrato" (off)
VIBRATO_NUM_CLASSES = 3

# Per-signal class counts + rest/pad ids, DERIVED from the constants above (do not
# duplicate the literals — Arioso's CondSpec registry reads these).
SIGNAL_NUM_CLASSES = {
    "articulation": ARTIC_NUM_CLASSES,
    "velocity": VELOCITY_NUM_CLASSES,
    "vibrato": VIBRATO_NUM_CLASSES,
}
SIGNAL_REST_ID = {
    "articulation": ARTIC_REST_ID,
    "velocity": VELOCITY_REST_ID,
    "vibrato": VIBRATO_REST_ID,
}

# Legato / rounding-slack window: a note whose gap to the next onset is <= this many
# frames extends to that onset (no spurious rest sliver); a wider gap stays rest.
REST_SNAP_FRAMES = 10


# --- canonical per-note record --------------------------------------------------

@dataclass(frozen=True)
class NoteEvent:
    """One played note — the canonical record every producer rasterizes from.

    ``articulation`` is a name in :data:`ARTICULATIONS` (validated at rasterization,
    unknown names are a hard error); ``vibrato`` is a played-with-vibrato flag.

    ``env_dct`` (3 cosine coefficients, :mod:`common.envelope`) and ``vib_params`` /
    ``vib_model`` (:mod:`common.vibrato_model`) are the measured per-note *shape*
    descriptors baked into the rendered prior. Both are optional: the rasterizers and
    conditioning encodings ignore them entirely, so a producer that has not measured
    them emits identical cond tracks.
    """
    start_s: float
    end_s: float
    pitch: int
    velocity: int = 80
    articulation: str = "normal"
    vibrato: bool = False
    env_dct: tuple[float, ...] | None = None
    vib_params: tuple[float, ...] | None = None
    vib_model: str | None = None


# --- the one rounding rule ------------------------------------------------------

def frame_of(t_s: float) -> int:
    """Mel frame index of a time in seconds: ``round(t_s * SR / HOP_SIZE)``.

    THE single rounding rule shared by onset frames, note-group onsets and every
    rasterizer. Keeping it in one place is what makes the load-bearing identity
    ``onset_frames(notes) == [g.onset_frame for g in note_groups(notes)]`` hold, which
    Arioso's clip enumerator relies on. Matches ``DataSynthesizer.build_prior``'s
    ``np.round(t * SR / HOP)`` (numpy and Python both round half-to-even).
    """
    return int(round(t_s * SR / HOP_SIZE))


def _frame_clip(t_s: float, n_frames: int) -> int:
    """Frame index of ``t_s`` clamped to ``[0, n_frames]`` (for per-note span fills)."""
    return int(np.clip(frame_of(t_s), 0, n_frames))


# --- note grouping (moved from DataSynthesizer.technique) ------------------------

class NoteGroup(NamedTuple):
    """One distinct rounded-onset frame's worth of notes (chords / sub-frame onsets merged).

    ``onset_frame`` keys the group and — across all groups — equals the saved onsets
    array (the clip enumerator keys off it). ``start_s``/``end_s`` are the min-onset /
    max-offset *seconds* of the merged notes; ``end_frame`` is the mel frame that span
    ends on (clamped to at least ``onset_frame + 1`` and at most ``n_frames``), used by
    the per-frame expansion.
    """
    onset_frame: int
    end_frame: int
    start_s: float
    end_s: float
    n_notes: int


def note_groups(notes: list[NoteEvent], offset_s: float, n_frames: int) -> list[NoteGroup]:
    """Group ``notes`` by rounded onset frame, aligned like the saved onsets array.

    Each note's onset is rounded with :func:`frame_of` (after adding ``offset_s``) and
    dropped if it falls outside ``[0, n_frames)`` — the same range filter the prior
    build applies. Notes sharing a rounded onset frame (chords, double-stops, sub-frame
    onsets) merge into one group: ``start_s`` = earliest onset, ``end_s`` = latest
    offset, ``n_notes`` = how many merged. ``[g.onset_frame for g in groups]`` is
    bit-identical to :func:`onset_frames`.
    """
    groups: dict[int, dict] = {}
    for note in notes:
        start = note.start_s + offset_s
        end = note.end_s + offset_s
        onset_frame = frame_of(start)
        if onset_frame < 0 or onset_frame >= n_frames:
            continue
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
        end_frame = int(np.clip(frame_of(g["end_s"]), onset_frame + 1, n_frames))
        out.append(NoteGroup(onset_frame=onset_frame, end_frame=end_frame,
                             start_s=g["start_s"], end_s=g["end_s"], n_notes=g["n_notes"]))
    return out


def note_groups_from_midi(midi_path: str, offset_s: float, n_frames: int) -> list[NoteGroup]:
    """Parse ``midi_path`` into onset-frame-grouped note spans (see :func:`note_groups`).

    Uses **pretty_midi** ``note.start``/``note.end`` across all instruments (the dataset
    puts each note-stream on its own instrument track, so no filtering), converts them to
    :class:`NoteEvent`\\ s, then delegates to :func:`note_groups` — so the emitted onset
    frames are bit-identical to :func:`common.prior.note_onsets` rounded with
    :func:`frame_of`.
    """
    import pretty_midi  # local: keep the module import-light and torch/pretty_midi-optional

    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = [NoteEvent(start_s=note.start, end_s=note.end, pitch=note.pitch,
                       velocity=note.velocity)
             for inst in pm.instruments for note in inst.notes]
    return note_groups(notes, offset_s, n_frames)


def expand_to_frames(groups: list[NoteGroup], ids: np.ndarray, n_frames: int, *,
                     rest_id: int, rest_snap: int = REST_SNAP_FRAMES) -> np.ndarray:
    """Expand per-group ids into a ``[n_frames]`` uint8 per-frame id track.

    Frames start all-``rest_id`` (so pre-first-onset frames stay rest automatically). For
    each group we fill from its onset frame forward with that group's id (``ids[i]``). The
    end of the fill uses a **hybrid rest policy** against the next onset (or ``n_frames``
    for the last group): if the gap between this note's ``end_frame`` and the next onset
    is small (``<= rest_snap`` frames — a legato transition / rounding slack), the note is
    extended to the next onset so no spurious rest sliver appears; otherwise the note stops
    at ``end_frame`` and the real gap ``[end_frame, next_onset)`` stays rest.
    """
    out = np.full(n_frames, rest_id, dtype=np.uint8)
    n = len(groups)
    for i, g in enumerate(groups):
        next_boundary = groups[i + 1].onset_frame if i + 1 < n else n_frames
        fill_end = next_boundary if (next_boundary - g.end_frame) <= rest_snap else g.end_frame
        fill_end = min(fill_end, n_frames)
        if fill_end > g.onset_frame:
            out[g.onset_frame:fill_end] = ids[i]
    return out


# --- per-frame rasterizers ------------------------------------------------------

def _onset_representatives(notes: list[NoteEvent], offset_s: float,
                           n_frames: int) -> dict[int, NoteEvent]:
    """Map each in-range onset frame to its first note (earliest start, then lowest pitch)."""
    per_onset: dict[int, NoteEvent] = {}
    for nt in sorted(notes, key=lambda x: (float(x.start_s), int(x.pitch))):
        of = frame_of(nt.start_s + offset_s)
        if 0 <= of < n_frames:
            per_onset.setdefault(of, nt)
    return per_onset


def rasterize_articulation(notes: list[NoteEvent], n_frames: int,
                           offset_s: float = 0.0) -> np.ndarray:
    """``[T]`` uint8 articulation-id track (rest = :data:`ARTIC_REST_ID`).

    Per onset group the first note wins (earliest start, then lowest pitch); its
    articulation name is mapped to an id via ``ARTICULATIONS.index`` — an **unknown name
    is a hard ``ValueError``** (no silent fallback) — and expanded with the rest-snap
    hybrid fill. Frames with nothing sounding are rest.
    """
    groups = note_groups(notes, offset_s, n_frames)
    reps = _onset_representatives(notes, offset_s, n_frames)
    ids = np.empty(len(groups), dtype=np.int64)
    for i, g in enumerate(groups):
        name = reps[g.onset_frame].articulation
        ids[i] = ARTICULATIONS.index(name)  # ValueError on an unknown articulation name
    return expand_to_frames(groups, ids, n_frames, rest_id=ARTIC_REST_ID)


def rasterize_velocity(notes: list[NoteEvent], n_frames: int,
                       offset_s: float = 0.0) -> np.ndarray:
    """``[T]`` uint8 velocity track: per-note span fill of MIDI velocity, rest = 0.

    Each note fills ``[frame_of(start), frame_of(end))`` (at least one frame) with its
    velocity clipped to ``[1, 127]``; overlapping notes overwrite in iteration order.
    Frames covered by no note stay :data:`VELOCITY_REST_ID` (0).
    """
    vel = np.full(n_frames, VELOCITY_REST_ID, dtype=np.uint8)
    for nt in notes:
        f0 = _frame_clip(nt.start_s + offset_s, n_frames)
        f1 = _frame_clip(nt.end_s + offset_s, n_frames)
        if f1 <= f0:
            f1 = min(n_frames, f0 + 1)
        vel[f0:f1] = int(np.clip(int(nt.velocity), 1, 127))
    return vel


def rasterize_vibrato(notes: list[NoteEvent], n_frames: int,
                      offset_s: float = 0.0) -> np.ndarray:
    """``[T]`` uint8 vibrato track: per-note span fill of on/off, rest = 2.

    Each note fills its span with :data:`VIBRATO_ON` (1) if played with vibrato else
    :data:`VIBRATO_OFF` (0); frames covered by no note stay :data:`VIBRATO_REST_ID` (2),
    so rest is distinct from a played-but-not-vibrato note.
    """
    vib = np.full(n_frames, VIBRATO_REST_ID, dtype=np.uint8)
    for nt in notes:
        f0 = _frame_clip(nt.start_s + offset_s, n_frames)
        f1 = _frame_clip(nt.end_s + offset_s, n_frames)
        if f1 <= f0:
            f1 = min(n_frames, f0 + 1)
        vib[f0:f1] = VIBRATO_ON if nt.vibrato else VIBRATO_OFF
    return vib


def onset_frames(notes: list[NoteEvent], n_frames: int, offset_s: float = 0.0) -> np.ndarray:
    """``[K]`` int32 onset frame indices: sorted, unique, clipped to ``[0, n_frames)``.

    Each note's onset is rounded with :func:`frame_of` (after ``offset_s``); frames
    outside ``[0, n_frames)`` are dropped. Identical to ``[g.onset_frame for g in
    note_groups(notes, offset_s, n_frames)]`` — the identity Arioso enumeration depends on.
    """
    frames = np.array([frame_of(nt.start_s + offset_s) for nt in notes], dtype=np.int64)
    frames = frames[(frames >= 0) & (frames < n_frames)]
    return np.unique(frames).astype(np.int32)


def offset_frames(notes: list[NoteEvent], n_frames: int, offset_s: float = 0.0) -> np.ndarray:
    """``[K]`` int32 note-offset frame indices: sorted, unique, **clamped** to ``[0, n_frames-1]``.

    Each note's ``end_s`` is rounded with :func:`frame_of` (after ``offset_s``) — the mel
    frame that note stops sounding on — and, unlike :func:`onset_frames`, out-of-range
    frames are **clamped rather than dropped** (``np.clip`` into ``[0, n_frames-1]``). This
    is deliberate: a note ending at the recording tail rounds to ``n_frames`` (or beyond),
    and dropping it would erase a real boundary — Arioso's "time until offset" signal must
    still see an offset event on the last frame. Onsets drop because a note starting past
    the tail was never in the clip; offsets clamp because the note *is* in the clip and only
    its end fell off the grid.
    """
    frames = np.array([frame_of(nt.end_s + offset_s) for nt in notes], dtype=np.int64)
    frames = np.clip(frames, 0, n_frames - 1)
    return np.unique(frames).astype(np.int32)


# --- manifest read / write ------------------------------------------------------

def _frame_block() -> dict:
    """The mel-contract block asserted into every manifest (from :mod:`common.config`)."""
    return {"sr": SR, "hop": HOP_SIZE, "n_mels": N_MELS}


def _validate_manifest(manifest: dict, where: str) -> dict:
    """Assert ``manifest``'s schema_version and frame block agree with common.config."""
    if not isinstance(manifest, dict):
        raise ValueError(f"{where}: manifest must be a JSON object")
    ver = manifest.get("schema_version")
    if ver != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"{where}: manifest schema_version {ver!r} != {MANIFEST_SCHEMA_VERSION}")
    frame = manifest.get("frame")
    expected = _frame_block()
    if frame != expected:
        raise ValueError(
            f"{where}: manifest frame block {frame!r} disagrees with common.config "
            f"{expected!r} (mels built at a different sr/hop/n_mels are unusable)")
    return manifest


def load_manifest(root: str) -> dict:
    """Load + validate ``<root>/manifest.json`` (schema_version + frame block)."""
    path = os.path.join(root, MANIFEST_JSON)
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    return _validate_manifest(manifest, path)


def write_manifest(root: str, manifest: dict) -> str:
    """Atomically write ``manifest`` to ``<root>/manifest.json`` (validated first).

    Same-dir ``.tmp`` + ``os.replace`` (as ``Labeler.library.atomic_write_json``), so a
    crash never leaves a half-written manifest that training would read.
    """
    _validate_manifest(manifest, "write_manifest")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, MANIFEST_JSON)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_manifest_", suffix=".json", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


# --- root accessor --------------------------------------------------------------

class DatasetRoot:
    """Read-only accessor over one standard dataset root (loaded + validated manifest).

    Attributes:
        name: the manifest's ``name`` (e.g. ``"synthetic"`` / a gt root id).
        path: the root directory.
        signals: ``frozenset`` of the root-wide conditioning signals present on disk.
        clips: ``{base: clip_entry}`` — a listed clip is complete and trainable.

    Path helpers return absolute-or-root-relative paths (joined on ``path``);
    :meth:`cond_path` returns ``None`` when the signal is not in this root's ``signals``.
    """

    def __init__(self, path: str):
        self.path = path
        self.manifest = load_manifest(path)
        self.name = self.manifest.get("name")
        self.signals = frozenset(self.manifest.get("signals", []))
        self.clips = dict(self.manifest.get("clips", {}))

    def basenames(self) -> list[str]:
        """Sorted list of the root's clip basenames."""
        return sorted(self.clips)

    def _clip(self, base: str) -> dict:
        try:
            return self.clips[base]
        except KeyError:
            raise KeyError(f"{base!r} is not a listed clip of root {self.name!r}")

    def n_frames(self, base: str) -> int:
        return int(self._clip(base)["n_frames"])

    def piece(self, base: str) -> str:
        """The split-grouping key for ``base`` (synthetic: ``composer/catalog``)."""
        return self._clip(base)["piece"]

    def gt_path(self, base: str) -> str:
        return os.path.join(self.path, DIR_GT, base + ".wav")

    def target_mel_path(self, base: str) -> str:
        return os.path.join(self.path, DIR_TARGET_MEL, base + ".npy")

    def prior_mel_path(self, base: str) -> str:
        return os.path.join(self.path, DIR_PRIOR_MEL, base + ".npy")

    def onsets_path(self, base: str) -> str:
        return os.path.join(self.path, DIR_ONSETS, base + ".npy")

    def offsets_path(self, base: str) -> str:
        return os.path.join(self.path, DIR_OFFSETS, base + ".npy")

    def cond_path(self, signal: str, base: str) -> str | None:
        """Path to ``base``'s ``signal`` track, or ``None`` if the root lacks that signal."""
        if signal not in self.signals:
            return None
        return os.path.join(self.path, COND_ROOT, signal, base + ".npy")
