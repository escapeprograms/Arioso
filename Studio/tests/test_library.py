"""library: the torch-free vocoder discovery helpers behind /api/models.

The model scan is covered end-to-end through the API (``test_api``); these unit
tests pin the vocoder side, whose selection rules (both files required, the ``hf``
sentinel passing through to the loader) are what keep the render endpoint from
offering a checkpoint that would fail at load time.
"""

from __future__ import annotations

import dataclasses
import os

from Studio.config import load_config
from Studio.library import (scan_vocoders, vocoder_checkpoint_dir,
                            vocoders_root)


def _cfg(root, default="ft_v2"):
    return dataclasses.replace(load_config(), vocoders_root=str(root),
                               default_vocoder=default)


def _voc_dir(root, name, files=("config.json", "bigvgan_generator.pt")):
    d = root / name
    d.mkdir(parents=True)
    for f in files:
        (d / f).write_bytes(b"")
    return d


def test_scan_vocoders_sorted_with_default_flag(tmp_path):
    root = tmp_path / "vocoders"
    for name in ("ft_v3", "ft_v1", "ft_v2"):
        _voc_dir(root, name)
    cfg = _cfg(root)
    assert scan_vocoders(cfg) == [
        {"name": "ft_v1", "default": False},
        {"name": "ft_v2", "default": True},
        {"name": "ft_v3", "default": False},
    ]


def test_scan_vocoders_excludes_incomplete_dirs(tmp_path):
    root = tmp_path / "vocoders"
    _voc_dir(root, "complete")
    _voc_dir(root, "no_weights", files=("config.json",))
    _voc_dir(root, "no_config", files=("bigvgan_generator.pt",))
    (root / "loose.pt").write_bytes(b"")
    assert [v["name"] for v in scan_vocoders(_cfg(root))] == ["complete"]


def test_scan_vocoders_missing_root_is_empty(tmp_path):
    assert scan_vocoders(_cfg(tmp_path / "nope")) == []


def test_scan_vocoders_never_includes_hf(tmp_path):
    # "hf" has no dir; the API appends it, so the scan must not invent it.
    root = tmp_path / "vocoders"
    _voc_dir(root, "ft_v2")
    assert "hf" not in [v["name"] for v in scan_vocoders(_cfg(root, default="hf"))]


def test_vocoder_checkpoint_dir_maps_name_to_dir_and_passes_hf(tmp_path):
    cfg = _cfg(tmp_path / "vocoders")
    assert vocoder_checkpoint_dir(cfg, "hf") == "hf"          # loader sentinel
    assert vocoder_checkpoint_dir(cfg, "ft_v2") == os.path.join(
        vocoders_root(cfg), "ft_v2")
