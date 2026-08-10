"""Lazy, cached Arioso model + BigVGAN vocoder singletons for the renderer.

The server import graph is torch-free; this module is where torch finally lands,
but only **inside** function bodies (imports at call time), so merely importing it
— or anything above it — stays cheap and GPU-free. It owns two process-wide caches:

  * ``get_model(models_root, run, checkpoint, device)`` — loads a checkpoint the
    way ``Arioso.infer.main`` does (``torch.load`` with ``map_location`` ->
    ``cfg_from_dict(ckpt["cfg"])`` -> ``AriosoModel(cfg)`` ->
    ``load_state_dict(ckpt["ema"])`` -> ``eval``), keyed by ``(run, checkpoint,
    device)`` behind a lock so the first render pays the load and the rest reuse it.
  * ``get_vocoder(device, checkpoint_dir)`` — the frozen BigVGAN-v2 vocoder
    (:func:`common.vocoder.load_vocoder`), keyed by ``(checkpoint_dir, device)`` so a
    render can pick a fine-tuned generator per call and each stays loaded. Stock-HF
    weights download from the HF Hub on first use (a one-time cost the render
    pipeline reports as its own stage).

Both mirror :func:`Labeler.transcribe.get_model`'s double-checked-locking singleton
pattern. ``list_models`` is a thin pass-through to :func:`Studio.library.scan_models`
(the pure-filesystem scan behind ``/api/models``) so there is one model-listing path.
"""

from __future__ import annotations

import os
import threading
from typing import NamedTuple

from .config import StudioConfig
from .library import scan_models

_MODEL_CACHE: dict[tuple, "LoadedModel"] = {}
_MODEL_LOCK = threading.Lock()

_VOC_CACHE: dict[tuple, object] = {}
_VOC_LOCK = threading.Lock()


class ModelNotFound(Exception):
    """Raised when a requested ``<run>/<checkpoint>.pt`` does not exist on disk."""


class LoadedModel(NamedTuple):
    """A loaded checkpoint + the config it was built from + the device it lives on."""
    model: object          # Arioso.model.AriosoModel (eval mode)
    cfg: object            # Arioso.config.AriosoConfig reconstructed from the ckpt
    device: str


def list_models(cfg: StudioConfig) -> list[dict]:
    """Filesystem model scan (reuses :func:`Studio.library.scan_models`)."""
    return scan_models(cfg)


def _pick_device(device: str | None) -> str:
    if device is not None:
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model(models_root: str, run: str, checkpoint: str,
              device: str | None = None) -> LoadedModel:
    """Load (once) and return the ``run``/``checkpoint`` model on ``device``.

    Thread-safe singleton keyed by ``(run, checkpoint, device)``. ``device`` defaults
    to cuda when available else cpu. Raises :class:`ModelNotFound` if the checkpoint
    file is missing.
    """
    device = _pick_device(device)
    key = (run, checkpoint, device)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        import torch

        from Arioso.config import cfg_from_dict
        from Arioso.model import AriosoModel

        ckpt_path = os.path.join(models_root, run, checkpoint)
        if not os.path.isfile(ckpt_path):
            raise ModelNotFound(
                f"checkpoint not found: {ckpt_path} (run={run!r}, checkpoint={checkpoint!r})")

        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = cfg_from_dict(ckpt.get("cfg") or {})
        model = AriosoModel(cfg).to(device)
        # EMA weights are the inference default (Arioso.infer); fall back to raw.
        weights = ckpt.get("ema") or ckpt.get("model")
        if weights is None:
            raise KeyError(
                f"{ckpt_path} has neither 'ema' nor 'model' weights (keys: {sorted(ckpt)})")
        model.load_state_dict(weights)
        model.eval()

        loaded = LoadedModel(model=model, cfg=cfg, device=device)
        _MODEL_CACHE[key] = loaded
        return loaded


def get_vocoder(device: str | None = None, checkpoint_dir: str | None = None):
    """Load (once) and return a frozen BigVGAN-v2 vocoder on ``device``.

    Thread-safe singleton keyed by ``(checkpoint_dir, device)``, so a project
    rendering with a fine-tuned generator and one on the stock baseline can both
    stay resident. ``checkpoint_dir`` is passed straight to
    :func:`common.vocoder.load_vocoder`: ``None`` takes that loader's default
    (``common.config.VOCODER_DIR``), ``"hf"`` forces the stock HF checkpoint, and
    anything else is a local run dir. Stock weights download from the HF Hub on
    first call (cached thereafter); the loader falls back to the PyTorch-native
    path when the fused CUDA kernel can't build (it can't in this environment).
    """
    device = _pick_device(device)
    key = (checkpoint_dir, device)
    cached = _VOC_CACHE.get(key)
    if cached is not None:
        return cached
    with _VOC_LOCK:
        cached = _VOC_CACHE.get(key)
        if cached is not None:
            return cached
        from common.vocoder import load_vocoder
        voc = load_vocoder(device=device, checkpoint_dir=checkpoint_dir)
        _VOC_CACHE[key] = voc
        return voc
