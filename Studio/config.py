"""StudioConfig — dataclass defaults + ``studio.yaml`` partial override.

Mirrors :mod:`Labeler.config` (and the philosophy of ``Arioso/run_config.py``):
the config is a small tree of frozen param dataclasses; the YAML file only
carries the keys it changes, and any unknown key is a hard error listing the
valid ones, so a config written today still loads after new fields are added.

Sample rate / hop / mel frame rate are NOT config knobs — they come from
``common.config`` (the audio/mel contract shared with the rest of the project)
and are re-exported here so pipeline modules keep one import line. This module is
torch-free and must stay that way (torch is a lazy import in the render phase).
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field, fields

import yaml

from common.config import HOP_SIZE as HOP, SR  # canonical audio/mel contract

# project.json schema version (bumped when the on-disk project shape changes).
# v2: one-time cache invalidation for the env-shaped prior (velocity->env_dct baked in).
# v3: velocity->c0 map re-anchored to the envelope display range (-30..+16 dB); env-less
#     notes' fallback priors change level, and velocity is hashed but the map is not, so
#     cached segments must be invalidated once.
SCHEMA_VERSION = 3

# Mel frame rate the model + saw prior are sampled at (hop 512 @ 44.1 kHz).
FRAME_RATE = SR / HOP  # ~86.13 fps

# Prior synthesis modes (bend = pitch-bends baked into the saw prior;
# quantized = ignore per-note bends/vibrato, one pitch per note — a dev toggle).
PRIOR_MODES = ("bend", "quantized")

# Snap-grid options surfaced to the toolbar magnet dropdown (id -> label). The
# id is resolved to a grid size (in quarter-note beats) by ``timing.snap_grid``.
SNAP_OPTIONS = (
    ("none", "None"),
    ("line", "Line"),
    ("1/2step", "½ step"),
    ("1/3step", "⅓ step"),
    ("1/4step", "¼ step"),
    ("1/6step", "⅙ step"),
    ("step", "Step"),
    ("beat", "Beat"),
    ("bar", "Bar"),
)


# --- articulation vocabulary ------------------------------------------------

@dataclass(frozen=True)
class Articulation:
    """One playing-technique class: display name, hotkey, note-fill color, abbrev.

    Project-only for now: articulation is stored in project.json and colors the
    note in the UI, but it never affects a render while the loaded checkpoint is
    unconditioned (``cfg.conditioning == []``). ``model_vocab`` is the stub name
    this class maps onto once conditioned checkpoints exist.
    """
    name: str
    key: str
    color: str
    abbrev: str
    model_vocab: str


# Normal = FL Studio's default lime-green note; the rest get distinct fills.
_DEFAULT_ARTICULATIONS = (
    Articulation("normal", "1", "#6cc04a", "norm", "normal"),
    Articulation("slur", "2", "#dd8452", "slur", "slur"),
    Articulation("spiccato", "3", "#4c9be8", "spic", "spiccato"),
    Articulation("detache", "4", "#c46bd0", "det", "detache"),
)


# --- param groups -----------------------------------------------------------

@dataclass(frozen=True)
class CacheParams:
    """Segment-cache knobs (used by the render/cache phase, declared here now).

    A cache segment is a maximal run of notes with no inter-note gap ``>= gap_s``,
    padded ``pad_s`` into the silence on each side before rendering. ``max_files`` /
    ``max_mb`` bound the per-project ``render/cache/`` footprint — GC drops the
    oldest stale (off-manifest) segment WAVs first by count, then by total size.
    """
    gap_s: float = 0.35
    pad_s: float = 0.1
    max_files: int = 32
    max_mb: float = 100.0


@dataclass(frozen=True)
class RetentionParams:
    """Housekeeping knobs for the periodic maintenance sweep (:mod:`Studio.maintenance`).

    ``backups_keep`` is the number of rolling ``project.backup/rev_*`` files retained
    per project (both by the save-time roll and the sweep).
    """
    backups_keep: int = 8


@dataclass
class StudioConfig:
    """Top-level config: server, project/model layout, vocabulary, cache params."""
    projects_root: str = "Data/studio_projects"
    port: int = 8766
    models_root: str = "Arioso/models"
    default_model: str = "7-9-adr"
    default_checkpoint: str = "checkpoint_final.pt"
    default_prior_mode: str = "bend"
    articulations: tuple[Articulation, ...] = _DEFAULT_ARTICULATIONS
    cache: CacheParams = field(default_factory=CacheParams)
    retention: RetentionParams = field(default_factory=RetentionParams)

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
        """Technique assigned to a freshly created note."""
        return self.articulations[0].name if self.articulations else "normal"

    def vocabulary_snapshot(self) -> list[dict]:
        """Articulation vocab embedded into /api/config (name/key/color/abbrev/...)."""
        return [dataclasses.asdict(a) for a in self.articulations]

    def technique_model_vocab(self) -> dict[str, str]:
        """Technique -> model-vocab (articulation) mapping table.

        Kept as a derived view of the articulation vocab so there is one source of
        truth. Feeds real conditioned renders: ``Studio.render`` maps each note's
        technique through this table to the articulation name the conditioned
        checkpoint's articulation signal expects (unmapped names fall back to
        ``normal``). With an unconditioned checkpoint it is simply unused.
        """
        return {a.name: a.model_vocab for a in self.articulations}

    def snap_options(self) -> list[dict]:
        return [{"id": sid, "label": label} for sid, label in SNAP_OPTIONS]


# --- YAML loading (partial override) ----------------------------------------

def _coerce(value, default, where: str):
    """Coerce ``value`` to ``default``'s type (PyYAML can hand back odd shapes)."""
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


def _apply_group(defaults, section, where: str):
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
        out.append(Articulation(
            name=str(entry["name"]), key=str(entry["key"]), color=str(entry["color"]),
            abbrev=str(entry["abbrev"]), model_vocab=str(entry["model_vocab"])))
    names = [a.name for a in out]
    if len(set(names)) != len(names):
        raise ValueError(f"{where}: duplicate articulation name(s) in {names}")
    return tuple(out)


def load_config(yaml_path: str | None = None) -> StudioConfig:
    """Load a ``StudioConfig``: code defaults overridden by ``yaml_path`` if given.

    ``None`` -> the packaged ``Studio/studio.yaml`` if present, else pure code
    defaults. Unknown top-level or nested keys raise a listing ``ValueError``.
    """
    if yaml_path is None:
        packaged = os.path.join(os.path.dirname(os.path.abspath(__file__)), "studio.yaml")
        yaml_path = packaged if os.path.isfile(packaged) else None
    if yaml_path is None:
        return StudioConfig()

    with open(yaml_path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"{yaml_path}: top-level YAML must be a mapping")

    base = StudioConfig()
    top_scalars = {"projects_root", "port", "models_root", "default_model",
                   "default_checkpoint", "default_prior_mode"}
    valid_top = top_scalars | {"cache", "retention", "articulations"}
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
    if "cache" in doc:
        kw["cache"] = _apply_group(base.cache, doc["cache"], f"{yaml_path}:cache")
    if "retention" in doc:
        kw["retention"] = _apply_group(base.retention, doc["retention"], f"{yaml_path}:retention")

    cfg = dataclasses.replace(base, **kw)
    if cfg.default_prior_mode not in PRIOR_MODES:
        raise ValueError(
            f"{yaml_path}:default_prior_mode: must be one of {PRIOR_MODES}, "
            f"got {cfg.default_prior_mode!r}")
    return cfg
