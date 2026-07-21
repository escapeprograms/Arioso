"""LabelerConfig — dataclass defaults + ``labeler.yaml`` partial override.

Mirrors the philosophy of ``Arioso/run_config.py`` (partial override, unknown
keys are a hard error), kept deliberately simpler: the config is a small tree of
frozen param dataclasses plus an articulation vocabulary list. The YAML file
only needs to carry the keys it changes; everything else keeps its code default,
so a config written today still loads after new fields are added.

The sample rate / hop are NOT config knobs — they come from ``common.config``
(the mel + audio contract shared with the rest of the project) and are re-exported
here so pipeline modules keep one import line.

``stage_params_hash(stage)`` returns a short digest of the parameters that affect
one pipeline stage's output; ``processing`` records it per stage so a param change
(or a forced rerun) invalidates exactly the downstream stages.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field, fields

import yaml

from common.config import SR, HOP_SIZE as HOP  # canonical audio/mel contract

# Pipeline stages, in execution order (status.json / JobManager key off this).
STAGES = ("ingest", "denoise", "transcribe", "align", "velocity", "vibrato",
          "media", "finalize")

# notes.json schema version (bumped when the on-disk note shape changes).
SCHEMA_VERSION = 1

# The MUSC model's native frame rate (hop 256 @ 44.1 kHz) — the rate the
# per-note pitch-bend arrays are sampled at (used by the vibrato analysis).
MODEL_HOP = 256
MODEL_FPS = SR / MODEL_HOP  # ~172.27 fps


# --- articulation vocabulary ------------------------------------------------

@dataclass(frozen=True)
class Articulation:
    """One playing-technique class: display name, hotkey, color, compact abbrev."""
    name: str
    key: str
    color: str
    abbrev: str


_DEFAULT_ARTICULATIONS = (
    Articulation("normal", "1", "#4c72b0", "norm"),
    Articulation("slur", "2", "#dd8452", "slur"),
    Articulation("spiccato", "3", "#55a868", "spic"),
    Articulation("detache", "4", "#c44e52", "det"),
)


# --- per-stage parameter groups ---------------------------------------------

@dataclass(frozen=True)
class DenoiseParams:
    hpf_hz: float = 120.0
    hpf_order: int = 4
    prop_decrease: float = 0.85
    n_fft: int = 2048
    hop_length: int = 512
    time_constant_s: float = 2.0
    freq_mask_smooth_hz: float = 500.0
    time_mask_smooth_ms: float = 50.0


@dataclass(frozen=True)
class TranscribeParams:
    """MUSC posterior-decode knobs (recall-biased vs. the vendored defaults).

    ``transcribe_notes`` decodes the model posteriors itself rather than using
    MUSC's hardcoded 'spotify' decoder. The dataset MIDIs were score-aligned, so a
    blind re-transcription of a recording needs recall-biased thresholds and a
    short minimum note length — the vendored 127.7 ms floor deleted most short
    (e.g. spiccato) notes. ``batch_size`` is the number of ~3 s MUSC windows run on
    the GPU per forward pass — the direct VRAM lever (kept low so the peak stays
    well under an 8 GB card; the whole clip is still transcribed, just in more,
    smaller batches).

    ``use_subprocess`` runs the GPU forward pass in a fresh ``transcribe_worker``
    process so all VRAM (and its fragmentation) is reclaimed by the OS on exit and
    a GPU OOM becomes a catchable child failure instead of hanging the server; it
    is a pure-execution knob (does not change the transcription output).
    ``subprocess_timeout_s`` kills a soft-wedged child. Both execution knobs are
    excluded from the transcribe stage hash (see ``stage_params_hash``).
    """
    onset_thresh: float = 0.3
    frame_thresh: float = 0.2
    min_note_len_ms: float = 45.0
    batch_size: int = 8
    use_subprocess: bool = True
    subprocess_timeout_s: float = 900.0


@dataclass(frozen=True)
class VelocityParams:
    pre_frames: int = 2
    post_frames: int = 4
    pct_low: float = 10.0
    pct_high: float = 95.0
    vel_floor: int = 24
    vel_range: int = 96
    curve_gamma: float = 0.6
    degenerate_velocity: int = 80


@dataclass(frozen=True)
class VibratoParams:
    min_dur_s: float = 0.3
    middle_fraction: float = 0.8
    band_low_hz: float = 4.0
    band_high_hz: float = 9.0
    extent_min_cents: float = 12.0
    band_frac_min: float = 0.4    # min fraction of (>0.5 Hz) energy inside the band


@dataclass(frozen=True)
class ExportParams:
    gt_variant: str = "cleaned"
    tempo: float = 120.0
    program: int = 40
    mute_fade_ms: float = 5.0


@dataclass(frozen=True)
class MediaParams:
    tile_frames: int = 2048
    mel_p_low: float = 1.0
    mel_p_high: float = 99.0
    onset_hop: int = 512
    stretch_speeds: tuple[float, ...] = (0.5, 0.25)
    stretch_nfft: tuple[int, ...] = (2048, 4096)


# Which param group(s) each stage's output depends on (for stage invalidation).
# Stages absent here have no config-dependent output (a version tag covers them).
_STAGE_PARAMS = {
    "denoise": ("denoise",),
    "transcribe": ("transcribe",),
    "velocity": ("velocity",),
    "vibrato": ("vibrato",),
    "media": ("media",),
    "finalize": ("articulations",),
}

# Fields that live on a param group but only affect *execution*, not the stage's
# output — excluded from the stage hash so toggling them never forces a rerun.
_HASH_EXCLUDE = {
    "transcribe": ("use_subprocess", "subprocess_timeout_s"),
}


@dataclass
class LabelerConfig:
    """Top-level config: dataset layout, server, vocabulary, and stage params."""
    dataset_root: str = "Data/recorded"
    port: int = 8765
    editing_enabled: bool = True
    speeds: tuple[float, ...] = (1.0, 0.5, 0.25)
    articulations: tuple[Articulation, ...] = _DEFAULT_ARTICULATIONS
    denoise: DenoiseParams = field(default_factory=DenoiseParams)
    transcribe: TranscribeParams = field(default_factory=TranscribeParams)
    velocity: VelocityParams = field(default_factory=VelocityParams)
    vibrato: VibratoParams = field(default_factory=VibratoParams)
    export: ExportParams = field(default_factory=ExportParams)
    media: MediaParams = field(default_factory=MediaParams)
    # Resolved path of the YAML this config was loaded from (None => code defaults).
    # Set by load_config; the transcribe subprocess re-loads it so the child sees the
    # exact same config. Not a tunable — kept out of repr/equality/hashing.
    config_path: str | None = field(default=None, compare=False, repr=False)

    # -- derived helpers --
    @property
    def articulation_names(self) -> list[str]:
        return [a.name for a in self.articulations]

    def articulation(self, name: str) -> Articulation | None:
        for a in self.articulations:
            if a.name == name:
                return a
        return None

    @property
    def default_technique(self) -> str:
        """The auto-assigned technique for freshly transcribed notes."""
        return self.articulations[0].name if self.articulations else "normal"

    def vocabulary_snapshot(self) -> list[dict]:
        """The vocabulary embedded into notes.json (name/key/color/abbrev)."""
        return [dataclasses.asdict(a) for a in self.articulations]

    def stage_params_hash(self, stage: str) -> str:
        """Short digest of the params governing ``stage``'s output.

        A stable value across processes: same params -> same hash, so a stage can
        be skipped when its recorded hash matches. Version-tagged so an algorithm
        change (not captured by a param) can force a rebuild by bumping the tag.
        """
        payload: dict = {"_v": 2}
        for key in _STAGE_PARAMS.get(stage, ()):  # empty -> version-only hash
            if key == "articulations":
                payload[key] = self.vocabulary_snapshot()
            else:
                group = dataclasses.asdict(getattr(self, key))
                for drop in _HASH_EXCLUDE.get(stage, ()):  # execution-only knobs
                    group.pop(drop, None)
                payload[key] = group
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


# --- YAML loading (partial override) ----------------------------------------

def _coerce(value, default, where: str):
    """Coerce ``value`` to match ``default``'s type (PyYAML gives strings for 2e-4)."""
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{where}: expected a boolean (true/false), got {value!r}")
        return value
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(f"{where}: expected a number, got {value!r}")
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"{where}: expected a number, got {value!r}")
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where}: expected an integer, got {value!r}")
        return value
    if isinstance(default, tuple):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{where}: expected a list, got {value!r}")
        return tuple(value)
    return value  # str / other: pass through


def _apply_group(group_cls, defaults, section, where: str):
    """Build a frozen param dataclass from ``defaults``, overriding with ``section``."""
    if not isinstance(section, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(section).__name__}")
    valid = {f.name: getattr(defaults, f.name) for f in fields(defaults)}
    kw = {}
    for key, val in section.items():
        if key not in valid:
            raise ValueError(
                f"{where}: unknown key '{key}'. Valid keys: {sorted(valid)}")
        kw[key] = _coerce(val, valid[key], f"{where}.{key}")
    return dataclasses.replace(defaults, **kw)


def _parse_articulations(raw, where: str) -> tuple[Articulation, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{where}: 'articulations' must be a non-empty list")
    valid = {f.name for f in fields(Articulation)}
    out = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{where}[{i}]: each articulation must be a mapping")
        unknown = set(entry) - valid
        if unknown:
            raise ValueError(
                f"{where}[{i}]: unknown key(s) {sorted(unknown)}. Valid: {sorted(valid)}")
        missing = valid - set(entry)
        if missing:
            raise ValueError(f"{where}[{i}]: missing key(s) {sorted(missing)}")
        out.append(Articulation(name=str(entry["name"]), key=str(entry["key"]),
                                color=str(entry["color"]), abbrev=str(entry["abbrev"])))
    names = [a.name for a in out]
    if len(set(names)) != len(names):
        raise ValueError(f"{where}: duplicate articulation name(s) in {names}")
    return tuple(out)


def load_config(yaml_path: str | None = None) -> LabelerConfig:
    """Load a ``LabelerConfig``: code defaults overridden by ``yaml_path`` if given.

    ``None`` -> the packaged ``Labeler/labeler.yaml`` if present, else pure code
    defaults. Unknown top-level or nested keys raise a listing ``ValueError``.
    """
    if yaml_path is None:
        packaged = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labeler.yaml")
        yaml_path = packaged if os.path.isfile(packaged) else None
    if yaml_path is None:
        return LabelerConfig()

    with open(yaml_path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"{yaml_path}: top-level YAML must be a mapping")

    base = LabelerConfig()
    top_scalars = {"dataset_root", "port", "editing_enabled", "speeds"}
    groups = {"denoise", "transcribe", "velocity", "vibrato", "export", "media"}
    valid_top = top_scalars | groups | {"articulations"}
    unknown = set(doc) - valid_top
    if unknown:
        raise ValueError(
            f"{yaml_path}: unknown top-level key(s) {sorted(unknown)}. "
            f"Valid: {sorted(valid_top)}")

    kw: dict = {}
    for key in top_scalars:
        if key in doc:
            kw[key] = _coerce(doc[key], getattr(base, key), f"{yaml_path}:{key}")
    if "articulations" in doc:
        kw["articulations"] = _parse_articulations(doc["articulations"], f"{yaml_path}:articulations")
    for key in groups:
        if key in doc:
            kw[key] = _apply_group(type(getattr(base, key)), getattr(base, key),
                                   doc[key], f"{yaml_path}:{key}")

    cfg = dataclasses.replace(base, config_path=os.path.abspath(yaml_path), **kw)
    # Sanity: stretch_speeds and stretch_nfft must be parallel.
    if len(cfg.media.stretch_speeds) != len(cfg.media.stretch_nfft):
        raise ValueError(
            f"{yaml_path}:media: stretch_speeds and stretch_nfft must have equal length")
    return cfg
