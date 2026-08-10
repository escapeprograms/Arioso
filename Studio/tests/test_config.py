"""load_config: defaults + partial YAML override for cache/retention budgets."""

from __future__ import annotations

import pytest

from Studio.config import CacheParams, RetentionParams, StudioConfig, load_config


def test_model_and_vocoder_defaults_derive_from_common_config():
    """StudioConfig's model/vocoder defaults are basenames of common.config's paths."""
    import os

    from common.config import ACOUSTIC_CKPT, VOCODER_DIR

    cfg = StudioConfig()
    assert cfg.default_model == os.path.basename(os.path.dirname(ACOUSTIC_CKPT))
    assert cfg.default_checkpoint == os.path.basename(ACOUSTIC_CKPT)
    assert cfg.default_vocoder == (
        "hf" if not VOCODER_DIR else os.path.basename(os.path.normpath(VOCODER_DIR)))
    assert cfg.vocoders_root == "Vocoder/models"


def test_vocoder_keys_override(tmp_path):
    y = tmp_path / "studio.yaml"
    y.write_text("vocoders_root: some/where\ndefault_vocoder: hf\n", encoding="utf-8")
    cfg = load_config(str(y))
    assert cfg.vocoders_root == "some/where"
    assert cfg.default_vocoder == "hf"
    assert cfg.models_root == "Arioso/models"     # untouched keys keep their defaults


def test_new_budget_defaults():
    cfg = StudioConfig()
    assert cfg.cache.max_files == 32
    assert cfg.cache.max_mb == 100.0
    assert cfg.retention.backups_keep == 8


def test_retention_and_cache_override(tmp_path):
    y = tmp_path / "studio.yaml"
    y.write_text("retention:\n  backups_keep: 3\ncache:\n  max_mb: 10\n",
                 encoding="utf-8")
    cfg = load_config(str(y))
    assert cfg.retention.backups_keep == 3
    assert cfg.cache.max_mb == 10.0
    assert cfg.cache.max_files == 32          # untouched cache key keeps its default
    assert isinstance(cfg.cache, CacheParams)
    assert isinstance(cfg.retention, RetentionParams)


def test_unknown_top_level_key_raises(tmp_path):
    y = tmp_path / "studio.yaml"
    y.write_text("bogus: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(y))


def test_unknown_retention_key_raises(tmp_path):
    y = tmp_path / "studio.yaml"
    y.write_text("retention:\n  nope: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(y))
