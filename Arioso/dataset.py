"""Torch dataset + length-bucketed batching with frame masks (Section 5).

Each example is a contiguous mel-frame slice ``[:, start:end]`` of a recording's prior and
target mels (memory-mapped, no audio re-windowing). Clips are variable length, so we:

* **length-bucket** the fixed clip pool into batches of similar frame count to minimize padding
  (all lengths are known up front from enumeration), then shuffle batch order per epoch;
* pad each batch to its max length and carry a **frame mask** ``[B, T]`` (1 real / 0 pad); the
  loss averages over real frames only, and the mask is the DiT attention key-padding mask.
"""

from __future__ import annotations

import functools
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from common.dataset_schema import DatasetRoot

from .clips import Clip, _basenames_per_root, enumerate_clips, open_roots
from .config import AriosoConfig, BoundaryCondSpec, CondSpec
from .pitch_aug import PitchAug, build_pitch_aug, shifted_pair, wavs_available
from .splits import make_splits


def boundary_distances(arr: np.ndarray, start: int, end: int, direction: str) -> np.ndarray:
    """Per-frame distance (in frames) from clip frames ``[start, end)`` to the nearest boundary.

    ``arr`` is a root's ``onsets``/``offsets`` array — ``[K]`` **absolute** (whole-recording) frame
    indices, sorted and unique. For each frame ``f`` in the clip window:

    * ``direction="since"`` -> ``f - (greatest boundary <= f)``: distance behind, 0 on the boundary
      frame. A boundary *before* the clip start still counts (distances stay absolute-frame based).
    * ``direction="until"`` -> ``(least boundary >= f) - f``: distance ahead, 0 on the boundary frame.

    Frames with no boundary on the required side (before the first onset / after the last offset,
    or an empty ``arr``) get the sentinel ``-1``. Returns ``[end - start]`` int64. This is the pure
    core of the "time since onset" / "time until offset" signal; :class:`BoundarySinusoid`
    featurizes it and applies the compact-support window.
    """
    frames = np.arange(start, end, dtype=np.int64)
    if arr.size == 0:
        return np.full(frames.shape, -1, dtype=np.int64)
    if direction == "since":
        idx = np.searchsorted(arr, frames, side="right") - 1
        d = np.where(idx >= 0, frames - arr[np.clip(idx, 0, None)], -1)
    else:
        idx = np.searchsorted(arr, frames, side="left")
        d = np.where(idx < arr.size, arr[np.clip(idx, None, arr.size - 1)] - frames, -1)
    return d.astype(np.int64)


class AriosoDataset(Dataset):
    """Returns ``(x0, x1)`` prior/target mel slices for a fixed clip pool over multiple roots.

    Each :class:`~Arioso.clips.Clip` carries a ``root`` index into ``roots`` (the ordered
    :class:`~common.dataset_schema.DatasetRoot` list); prior/target mel paths come from that root's
    accessors, so a basename shared by two roots never collides.

    When ``specs`` is non-empty each item also carries a ``"cond"`` dict mapping each spec name to
    that clip's ``[T]`` int64 per-frame track. The two spec kinds fill it differently:

    * **Categorical** (:class:`~Arioso.config.CondSpec`): the on-disk ``[T]`` id track (mmap-sliced
      identically to the mels). A signal the clip's root does *not* provide (``spec.name not in
      root.signals``) is filled with the CFG "unknown" id (``spec.num_classes`` — one past the last
      class, always in-range because the embedding table has ``num_classes + 1`` rows).
    * **Boundary** (:class:`~Arioso.config.BoundaryCondSpec`): a per-frame distance track derived
      from the root's ``onsets``/``offsets`` array via :func:`boundary_distances` (sentinel ``-1``).
      A root missing that array (or an empty one) yields all ``-1`` (the inactive/zero state).

    With no specs the item dict is exactly as before (no ``"cond"`` key).

    **Pitch-shift augmentation** (``pitch_aug``, see :mod:`Arioso.pitch_aug`): for a per-sample
    Bernoulli subset, ``x0``/``x1`` are re-melled from the clip's ``prior_wav/`` / ``gt/`` waveforms
    at a random shift instead of read from the mel memmaps. The shift is duration-preserving, so
    ``length`` and every cond/boundary track below are unaffected. Inactive by default (and for
    ``None``): no RNG is drawn, no waveform is opened, and every item is **byte-identical** to the
    pure-memmap path.
    """

    def __init__(self, roots: list[DatasetRoot], clips: list[Clip],
                 specs: tuple[CondSpec | BoundaryCondSpec, ...] = (),
                 pitch_aug: PitchAug | None = None):
        self.roots = roots
        self.clips = clips
        self.specs = tuple(specs)
        self.pitch_aug = pitch_aug
        self.epoch = 0
        self._wav_ok: dict[tuple[str, str], bool] = {}

    def __len__(self) -> int:
        return len(self.clips)

    def lengths(self) -> list[int]:
        return [c.end - c.start for c in self.clips]

    def set_epoch(self, epoch: int) -> None:
        """Set the per-epoch token seeding the pitch-shift draws (fresh shifts each epoch).

        Mirrors :meth:`LengthBucketBatchSampler.set_epoch`, which the training loop already calls
        with the same value. Purely bookkeeping when the augmentation is inactive.
        """
        self.epoch = epoch

    def _wavs_ok(self, root: DatasetRoot, base: str) -> bool:
        """Cached :func:`~Arioso.pitch_aug.wavs_available` per ``(root.path, base)``."""
        key = (root.path, base)
        ok = self._wav_ok.get(key)
        if ok is None:
            ok = wavs_available(root, base)
            self._wav_ok[key] = ok
        return ok

    def __getitem__(self, i: int) -> dict:
        c = self.clips[i]
        root = self.roots[c.root]
        x0 = x1 = None
        if self.pitch_aug is not None and self.pitch_aug.active:
            # Draw FIRST, availability second: the RNG stream depends only on (seed, epoch, index),
            # never on which roots happen to carry prior_wav/.
            cents = self.pitch_aug.cents_for(self.epoch, i)
            if cents is not None and self._wavs_ok(root, c.basename):
                x0, x1 = shifted_pair(root, c.basename, c.start, c.end,
                                      cents, self.pitch_aug.margin)
        if x0 is None:
            x0 = np.load(root.prior_mel_path(c.basename), mmap_mode="r")[:, c.start:c.end]
            x1 = np.load(root.target_mel_path(c.basename), mmap_mode="r")[:, c.start:c.end]
        item = {
            "x0": torch.from_numpy(np.ascontiguousarray(x0, dtype=np.float32)),
            "x1": torch.from_numpy(np.ascontiguousarray(x1, dtype=np.float32)),
            "length": c.end - c.start,
        }
        if self.specs:
            cond = {}
            for spec in self.specs:
                if isinstance(spec, CondSpec):
                    if spec.name not in root.signals:
                        # Root lacks this signal -> CFG "unknown" row (in-range: num_classes+1 rows).
                        arr = np.full(c.end - c.start, spec.num_classes, dtype=np.int64)
                    else:
                        arr = np.load(root.cond_path(spec.name, c.basename),
                                      mmap_mode="r")[c.start:c.end]
                        arr = np.ascontiguousarray(arr).astype(np.int64)
                    cond[spec.name] = torch.from_numpy(arr)
                else:
                    # Boundary signal: absolute-frame onset/offset array -> per-frame distance track.
                    path = (root.onsets_path(c.basename) if spec.boundary == "onset"
                            else root.offsets_path(c.basename))
                    if not os.path.isfile(path):
                        # Root lacks the boundary array -> inactive everywhere (all -1 sentinel).
                        d = np.full(c.end - c.start, -1, dtype=np.int64)
                    else:
                        bounds = np.load(path).astype(np.int64)   # [K] absolute frames, sorted-unique
                        d = boundary_distances(bounds, c.start, c.end, spec.direction)
                    cond[spec.name] = torch.from_numpy(d)
            item["cond"] = cond
        return item


class LengthBucketBatchSampler(Sampler):
    """Group clips of similar length into batches (min padding); shuffle batch order/epoch.

    Sorting by length then chunking keeps padding small, which matters because the WaveNet is
    non-causal with a large receptive field, so tail padding can otherwise leak into valid frames.
    """

    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool = True, seed: int = 0):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        batches = self.batches
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in order]
        yield from batches

    def __len__(self) -> int:
        return len(self.batches)


def collate(batch: list[dict], specs: tuple[CondSpec | BoundaryCondSpec, ...] = ()) -> dict:
    """Pad a batch to its max length; return tensors + a ``[B, T]`` frame mask (1 real / 0 pad).

    When ``specs`` is non-empty, also pack each spec's per-frame track into ``out["cond"]``:
    ``[B, T_max]`` long, filled per item. The pad value depends on the spec kind — a categorical
    :class:`~Arioso.config.CondSpec` pads with its ``pad_id`` (a valid embedding index), a boundary
    :class:`~Arioso.config.BoundaryCondSpec` pads with ``-1`` (its inactive sentinel). Either way
    the pad frames are masked out by ``frame_mask``, so the value only needs to stay in-range.
    """
    n_mels = batch[0]["x0"].shape[0]
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    t_max = int(lengths.max())
    b = len(batch)
    x0 = torch.zeros(b, n_mels, t_max, dtype=torch.float32)
    x1 = torch.zeros(b, n_mels, t_max, dtype=torch.float32)
    mask = torch.zeros(b, t_max, dtype=torch.float32)
    for i, item in enumerate(batch):
        ln = item["length"]
        x0[i, :, :ln] = item["x0"]
        x1[i, :, :ln] = item["x1"]
        mask[i, :ln] = 1.0
    out = {"x0": x0, "x1": x1, "frame_mask": mask, "lengths": lengths}
    if specs:
        cond = {}
        for spec in specs:
            pad_value = spec.pad_id if isinstance(spec, CondSpec) else -1
            packed = torch.full((b, t_max), pad_value, dtype=torch.long)
            for i, item in enumerate(batch):
                ln = item["length"]
                packed[i, :ln] = item["cond"][spec.name]
            cond[spec.name] = packed
        out["cond"] = cond
    return out


def build_dataloader(data_roots, split_name: str, batch_size: int,
                     cfg: AriosoConfig | None = None, shuffle: bool = True,
                     num_workers: int = 0) -> DataLoader:
    """DataLoader over the multi-root clip pool for a split ('train'|'val'), length-bucketed + masked.

    ``data_roots`` is the ordered list of root path strings; a :class:`DatasetRoot` is constructed
    once per root here (loading/validating each ``manifest.json``). The per-root splits are merged
    (``make_splits``) and enumerated (``enumerate_clips``) preserving root order, so a clip's ``root``
    index keys the right :class:`DatasetRoot` in the dataset.

    Conditioning (``cfg.conditioning``) flows through: the dataset loads each spec's id track (or the
    unknown fill for a root lacking that signal) and ``collate`` (bound to the specs via
    ``functools.partial``, kept picklable for num_workers>0) packs it into ``batch["cond"]``.

    Pitch-shift augmentation is wired here too, via :func:`~Arioso.pitch_aug.build_pitch_aug` — which
    returns an inactive policy for every split but ``"train"``, so only the train loader can ever
    augment (the val metric always scores the real mels).
    """
    cfg = cfg or AriosoConfig()
    roots = open_roots(list(data_roots))
    split = make_splits(roots, cfg)
    per_root = _basenames_per_root(roots, split[split_name])
    clips = enumerate_clips(roots, per_root, cfg)
    ds = AriosoDataset(roots, clips, specs=cfg.conditioning,
                       pitch_aug=build_pitch_aug(cfg, split_name))
    sampler = LengthBucketBatchSampler(ds.lengths(), batch_size, shuffle=shuffle, seed=cfg.seed)
    collate_fn = functools.partial(collate, specs=cfg.conditioning)
    # persistent_workers stays at its default False: a persistent worker would hold a stale copy of
    # the dataset and never see set_epoch, freezing the pitch-shift draws at epoch 0.
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=num_workers)
    return loader
