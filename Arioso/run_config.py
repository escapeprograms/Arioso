"""YAML run-config framework for Arioso — plain PyYAML over the frozen ``AriosoConfig``.

A run is fully described by two dataclasses: ``AriosoConfig`` (the *science*, embedded in every
checkpoint) and ``RunSettings`` (runtime/env knobs, never embedded). A YAML file maps 1:1 onto them
via two top-level sections — ``model:`` (AriosoConfig fields) and ``run:`` (RunSettings fields) —
plus an optional ``config_version:`` (accepted and reserved). Keys within a section are exactly the
dataclass field names, so any field added later is automatically a valid YAML key with zero
registration code.

**Partial-override semantics** (the backward-compat mechanism): the loader starts from code
defaults and applies only the keys present in the YAML. Missing keys keep their code default — so a
YAML written today still trains correctly after new fields are added. Unknown keys are a hard error
listing the valid keys (a typo catcher). Precedence at the call site is
**smoke clamps > CLI > YAML > code defaults**.

**Locked keys** (the mel contract: ``sr``, ``hop``, ``n_mels``, ``in_ch``) are rejected in YAML with
a pointer to ``common.config`` — they are asserted against the BigVGAN checkpoint and ``in_ch`` is
derived, so overriding them in a run-config is meaningless.

Design notes:
- No Hydra/OmegaConf: PyYAML is already installed and the two-dataclass mapping needs no framework.
- Registry validation at load time (``loss`` in ``LOSSES``, ``lr_schedule`` in ``SCHEDULES``,
  ``solver`` in ``SOLVERS``) turns a mistyped registry name into a readable error before training.
- No circular imports: ``cfm``/``schedules``/``solvers`` do not import this module.
- PyYAML gotcha: bare ``2e-4`` parses as a **string**, not a float. Type coercion against each
  field's default fixes this (and ``bool`` is checked before ``int`` because ``bool`` subclasses
  ``int``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime

import yaml

from .cfm import LOSSES
from .config import (AriosoConfig, CondSpec, SIGNALS, cfg_to_dict)
from .schedules import SCHEDULES
from .solvers import SOLVERS

# Mel-contract fields that must not be set in a run-config (asserted against the BigVGAN checkpoint
# in common.config; in_ch is derived). Naming one in YAML is an error pointing back there.
_LOCKED = ("sr", "hop", "n_mels", "in_ch")

# Deprecation-alias table applied before validation: {old_yaml_key: new_field_name}. Empty today —
# the escape hatch for the rare case a key must be renamed without breaking existing YAMLs. When a
# renamed key is used, the loader remaps it and warns.
_RENAMED: dict[str, str] = {}


@dataclass(frozen=True)
class RunSettings:
    """Runtime/env knobs for one training run — *never* embedded in checkpoints.

    Defaults match today's argparse defaults exactly, so an empty ``run:`` section (or no YAML at
    all) reproduces the current CLI behavior.
    """
    data_roots: tuple[str, ...] = ("Data",)   # ordered dataset roots (multi-root loader)
    batch_size: int = 8
    steps: int | None = None       # total-steps cap; None -> cfg.total_steps
    log_every: int = 50
    ckpt_every: int = 5000
    val_every: int = 2000
    device: str | None = None      # None -> "cuda" if available else "cpu" (resolved in main)
    wandb: bool = True
    name: str | None = None        # W&B run name; None -> auto_run_name(cfg)
    resume: bool = False           # load latest step ckpt from this run's folder; optimizer state not restored


# --- YAML -> dataclass loading ---------------------------------------------------

def _coerce(value, default, field_name: str, yaml_path: str):
    """Coerce a raw YAML value to match the type of ``default`` (a dataclass field's code default).

    Handles the PyYAML gotchas the loader must survive: bare ``2e-4`` parsed as a string, integers
    where floats are expected, and lists where tuples are expected. ``bool`` is checked *before*
    ``int``/``float`` because ``bool`` is a subclass of ``int`` in Python.
    """
    # bool fields require real YAML booleans (true/false), never 0/1/"yes".
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError(
                f"{yaml_path}: key '{field_name}' must be a boolean (true/false), "
                f"got {value!r} ({type(value).__name__})")
        return value
    # float fields: accept ints and strings ("2e-4"); reject bools.
    if isinstance(default, float):
        if isinstance(value, bool):
            raise ValueError(f"{yaml_path}: key '{field_name}' must be a number, got a boolean")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise ValueError(
                    f"{yaml_path}: key '{field_name}' must be a number, got {value!r}")
        raise ValueError(
            f"{yaml_path}: key '{field_name}' must be a number, got {type(value).__name__}")
    # int fields: reject bools; accept only real ints (no silent float truncation).
    if isinstance(default, int):
        if isinstance(value, bool):
            raise ValueError(f"{yaml_path}: key '{field_name}' must be an integer, got a boolean")
        if isinstance(value, int):
            return value
        raise ValueError(
            f"{yaml_path}: key '{field_name}' must be an integer, got {value!r}")
    # tuple fields (e.g. wn_dilation_cycle): accept a YAML list.
    if isinstance(default, tuple):
        if isinstance(value, (list, tuple)):
            return tuple(value)
        raise ValueError(
            f"{yaml_path}: key '{field_name}' must be a list, got {type(value).__name__}")
    # str / None-typed / everything else: pass through as-is.
    return value


def _parse_conditioning(raw, yaml_path: str) -> tuple[CondSpec, ...]:
    """Parse the ``conditioning:`` list into a tuple of ``CondSpec``.

    - ``[]`` / ``null`` -> ``()`` (the unconditioned ablation).
    - a string ``"technique"`` -> ``SIGNALS`` registry lookup.
    - a dict with a **known** ``name`` -> ``dataclasses.replace(SIGNALS[name], **overrides)``
      (tweak emb_dim etc. of a registered signal).
    - a dict with a **novel** ``name`` -> an inline custom ``CondSpec(**entry)`` (requires at least
      ``num_classes``, ``emb_dim``, ``dir``).
    """
    if raw is None or raw == []:
        return ()
    if not isinstance(raw, list):
        raise ValueError(
            f"{yaml_path}: 'conditioning' must be a list (of signal names or specs), "
            f"got {type(raw).__name__}")
    specs: list[CondSpec] = []
    for entry in raw:
        if isinstance(entry, str):
            if entry not in SIGNALS:
                raise ValueError(
                    f"{yaml_path}: unknown conditioning signal '{entry}'. "
                    f"Known signals: {sorted(SIGNALS)}. Define a custom one inline as a mapping "
                    f"with name/num_classes/emb_dim/dir.")
            specs.append(SIGNALS[entry])
        elif isinstance(entry, dict):
            entry = dict(entry)
            name = entry.get("name")
            if name is None:
                raise ValueError(
                    f"{yaml_path}: a conditioning entry mapping must have a 'name' key, "
                    f"got {entry!r}")
            if name in SIGNALS:
                # Known signal with overrides: start from the registered spec.
                overrides = {k: v for k, v in entry.items() if k != "name"}
                unknown = set(overrides) - {f.name for f in dataclasses.fields(CondSpec)}
                if unknown:
                    raise ValueError(
                        f"{yaml_path}: unknown CondSpec field(s) {sorted(unknown)} for signal "
                        f"'{name}'. Valid fields: {[f.name for f in dataclasses.fields(CondSpec)]}")
                specs.append(dataclasses.replace(SIGNALS[name], **overrides))
            else:
                # Novel inline signal: build a fresh CondSpec (name/num_classes/emb_dim/dir req'd).
                valid = {f.name for f in dataclasses.fields(CondSpec)}
                unknown = set(entry) - valid
                if unknown:
                    raise ValueError(
                        f"{yaml_path}: unknown CondSpec field(s) {sorted(unknown)} for inline "
                        f"signal '{name}'. Valid fields: {sorted(valid)}")
                try:
                    specs.append(CondSpec(**entry))
                except TypeError as e:
                    raise ValueError(
                        f"{yaml_path}: inline conditioning signal '{name}' is missing required "
                        f"fields (need num_classes, emb_dim, dir): {e}")
        else:
            raise ValueError(
                f"{yaml_path}: each conditioning entry must be a signal name (string) or a "
                f"spec mapping, got {type(entry).__name__}")
    return tuple(specs)


def _apply_section(section: dict, defaults, yaml_path: str, kind: str) -> dict:
    """Validate one YAML section against a dataclass's fields and return coercion kwargs.

    ``defaults`` is a default instance of the target dataclass (``AriosoConfig()`` /
    ``RunSettings()``). ``kind`` names the section ("model"/"run") for error messages. Applies the
    ``_RENAMED`` alias table, rejects unknown keys, and coerces each value's type. Locked mel-contract
    keys are rejected *unless* set to their canonical value — so a user override (e.g. ``n_mels: 80``)
    errors, but the locked keys that ``save_resolved_config`` writes at their contract value (via
    ``cfg_to_dict``) pass through as no-ops, letting a saved config round-trip. The ``conditioning``
    key is handled by the caller (special parsing), so it is skipped here.
    """
    field_defaults = {f.name: getattr(defaults, f.name) for f in dataclasses.fields(defaults)}
    valid_keys = set(field_defaults)
    kw: dict = {}
    for raw_key, value in section.items():
        key = _RENAMED.get(raw_key, raw_key)
        if key != raw_key:
            print(f"  [run_config] '{raw_key}' is deprecated; using '{key}' instead")
        if kind == "model" and key == "conditioning":
            continue  # parsed separately by the caller
        if key not in valid_keys:
            raise ValueError(
                f"{yaml_path}: unknown '{kind}' key '{key}'. Valid keys: {sorted(valid_keys)}")
        coerced = _coerce(value, field_defaults[key], key, yaml_path)
        if kind == "model" and key in _LOCKED:
            # A locked key at its contract value is a harmless no-op (round-tripped saved configs
            # write it); any other value is a real override attempt and is rejected.
            if coerced != field_defaults[key]:
                raise ValueError(
                    f"{yaml_path}: '{key}' is a locked mel-contract key and cannot be overridden "
                    f"in a run-config (it is {field_defaults[key]!r}). The mel contract is defined "
                    f"in common.config (asserted against the BigVGAN checkpoint; in_ch is derived).")
            continue  # don't pass locked keys into the dataclass constructor
        kw[key] = coerced
    return kw


def load_config(yaml_path: str | None) -> tuple[AriosoConfig, RunSettings]:
    """Load ``(AriosoConfig, RunSettings)`` from a YAML run-config, or code defaults if ``None``.

    ``None`` -> ``(AriosoConfig(), RunSettings())`` — exactly today's behavior. Otherwise the file's
    ``model:`` / ``run:`` sections partially override the code defaults (missing keys keep their
    default; unknown/locked keys raise a listing ``ValueError``). ``AriosoConfig`` construction is
    wrapped so ``__post_init__`` invariant asserts become a readable file-level ``ValueError``, and
    the resolved ``loss``/``lr_schedule``/``solver`` are validated against their registries.
    """
    if yaml_path is None:
        return AriosoConfig(), RunSettings()

    with open(yaml_path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if doc is None:
        doc = {}
    if not isinstance(doc, dict):
        raise ValueError(f"{yaml_path}: top-level YAML must be a mapping, got {type(doc).__name__}")

    valid_top = {"config_version", "model", "run"}
    unknown_top = set(doc) - valid_top
    if unknown_top:
        raise ValueError(
            f"{yaml_path}: unknown top-level key(s) {sorted(unknown_top)}. "
            f"Valid: {sorted(valid_top)}")

    model_section = doc.get("model") or {}
    run_section = doc.get("run") or {}
    if not isinstance(model_section, dict):
        raise ValueError(f"{yaml_path}: 'model' must be a mapping, got {type(model_section).__name__}")
    if not isinstance(run_section, dict):
        raise ValueError(f"{yaml_path}: 'run' must be a mapping, got {type(run_section).__name__}")

    model_kw = _apply_section(model_section, AriosoConfig(), yaml_path, "model")
    if "conditioning" in model_section:
        model_kw["conditioning"] = _parse_conditioning(model_section["conditioning"], yaml_path)

    # Legacy back-compat: a single-root ``out_dir: X`` maps to ``data_roots: [X]`` (deprecation note).
    # ``data_roots`` (a YAML list) is the current key; if both are present ``data_roots`` wins.
    run_section = dict(run_section)
    if "out_dir" in run_section:
        legacy = run_section.pop("out_dir")
        if not isinstance(legacy, str):
            raise ValueError(
                f"{yaml_path}: legacy 'out_dir' must be a single root path string, "
                f"got {type(legacy).__name__}; use 'data_roots: [...]' for multiple roots")
        print(f"  [run_config] 'out_dir' is deprecated; use 'data_roots: [...]'. "
              f"Mapping out_dir={legacy!r} -> data_roots=[{legacy!r}]")
        run_section.setdefault("data_roots", [legacy])

    run_kw = _apply_section(run_section, RunSettings(), yaml_path, "run")

    # Build AriosoConfig; turn __post_init__ AssertionError into a readable file-level error.
    try:
        cfg = AriosoConfig(**model_kw)
    except AssertionError as e:
        raise ValueError(f"{yaml_path}: invalid model config: {e}") from e

    # Registry validation — a mistyped registry name is a readable error, not a KeyError at runtime.
    if cfg.loss not in LOSSES:
        raise ValueError(
            f"{yaml_path}: unknown loss '{cfg.loss}'. Valid losses: {sorted(LOSSES)}")
    if cfg.lr_schedule not in SCHEDULES:
        raise ValueError(
            f"{yaml_path}: unknown lr_schedule '{cfg.lr_schedule}'. "
            f"Valid schedules: {sorted(SCHEDULES)}")
    if cfg.solver not in SOLVERS:
        raise ValueError(
            f"{yaml_path}: unknown solver '{cfg.solver}'. Valid solvers: {sorted(SOLVERS)}")

    run = RunSettings(**run_kw)
    return cfg, run


# --- CLI merge / run naming / round-trip ----------------------------------------

def merge_cli_overrides(run: RunSettings, args) -> RunSettings:
    """Overlay CLI args onto ``run``: only args that are not ``None`` win (YAML/defaults otherwise).

    Implements the CLI > YAML > defaults slice of the precedence chain. ``args`` is the argparse
    namespace whose run-level flags default to ``None`` sentinels, so an unspecified flag leaves the
    YAML/default value untouched. Repeatable ``--data-root`` maps to ``data_roots`` (a tuple);
    ``--steps`` maps to ``steps``; ``--wandb/--no-wandb`` to ``wandb``; ``--resume/--no-resume`` to
    ``resume``.
    """
    overrides = {}
    for field in ("batch_size", "steps", "log_every", "ckpt_every",
                  "val_every", "device", "wandb", "name", "resume"):
        val = getattr(args, field, None)
        if val is not None:
            overrides[field] = val
    data_roots = getattr(args, "data_roots", None)
    if data_roots:                                       # repeatable --data-root -> tuple of roots
        overrides["data_roots"] = tuple(data_roots)
    return dataclasses.replace(run, **overrides)


def auto_run_name(cfg: AriosoConfig) -> str:
    """A descriptive default W&B run name summarizing the run's science + a timestamp.

    e.g. ``arioso_h384_wn20_dit3_masked_mse_cosine_0706-1430``. The timestamp is computed *inside*
    this function (never at import time) so each call reflects the actual launch time.
    """
    stamp = datetime.now().strftime("%m%d-%H%M")
    return (f"arioso_h{cfg.hidden}_wn{cfg.wn_blocks}_dit{cfg.dit_blocks}"
            f"_{cfg.loss}_{cfg.lr_schedule}_{stamp}")


def save_resolved_config(cfg: AriosoConfig, run: RunSettings, path: str) -> None:
    """Dump the fully-resolved ``(cfg, run)`` to a YAML that round-trips through ``load_config``.

    Emits ``{config_version: 1, model: cfg_to_dict(cfg), run: asdict(run)}``. Conditioning specs are
    written as CondSpec dicts (each carries ``name`` + full fields), which ``_parse_conditioning``
    reads back via the known-name-with-overrides / novel-inline paths — so a saved config reproduces
    the same ``AriosoConfig`` on reload (reproducibility artifact for every run).
    """
    doc = {
        "config_version": 1,
        "model": cfg_to_dict(cfg),
        "run": dataclasses.asdict(run),
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
