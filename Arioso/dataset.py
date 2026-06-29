"""Torch dataset + length-bucketed batching with frame masks (Section 5).

Each example is a contiguous mel-frame slice ``[:, start:end]`` of a recording's prior and
target mels (memory-mapped, no audio re-windowing). Clips are variable length, so we:

* **length-bucket** the fixed clip pool into batches of similar frame count to minimize padding
  (all lengths are known up front from enumeration), then shuffle batch order per epoch;
* pad each batch to its max length and carry a **frame mask** ``[B, T]`` (1 real / 0 pad); the
  loss averages over real frames only, and the mask is the DiT attention key-padding mask.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .clips import Clip, enumerate_clips
from .config import PRIOR_MEL_DIR, AriosoConfig
from .splits import make_split


class AriosoDataset(Dataset):
    """Returns ``(x0, x1)`` prior/target mel slices for a fixed clip pool."""

    def __init__(self, out_dir: str, clips: list[Clip]):
        self.out_dir = out_dir
        self.clips = clips
        self.prior_dir = os.path.join(out_dir, PRIOR_MEL_DIR)
        self.target_dir = os.path.join(out_dir, "target_mel")

    def __len__(self) -> int:
        return len(self.clips)

    def lengths(self) -> list[int]:
        return [c.end - c.start for c in self.clips]

    def __getitem__(self, i: int) -> dict:
        c = self.clips[i]
        x0 = np.load(os.path.join(self.prior_dir, c.basename + ".npy"),
                     mmap_mode="r")[:, c.start:c.end]
        x1 = np.load(os.path.join(self.target_dir, c.basename + ".npy"),
                     mmap_mode="r")[:, c.start:c.end]
        return {
            "x0": torch.from_numpy(np.ascontiguousarray(x0, dtype=np.float32)),
            "x1": torch.from_numpy(np.ascontiguousarray(x1, dtype=np.float32)),
            "length": c.end - c.start,
        }


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


def collate(batch: list[dict]) -> dict:
    """Pad a batch to its max length; return tensors + a ``[B, T]`` frame mask (1 real / 0 pad)."""
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
    return {"x0": x0, "x1": x1, "frame_mask": mask, "lengths": lengths}


def build_dataloader(out_dir: str, split_name: str, batch_size: int,
                     cfg: AriosoConfig | None = None, shuffle: bool = True,
                     num_workers: int = 0) -> DataLoader:
    """DataLoader over the clip pool for a split ('train'|'val'), length-bucketed + masked."""
    cfg = cfg or AriosoConfig()
    split = make_split(out_dir, cfg)
    clips = enumerate_clips(out_dir, split[split_name], cfg)
    ds = AriosoDataset(out_dir, clips)
    sampler = LengthBucketBatchSampler(ds.lengths(), batch_size, shuffle=shuffle, seed=cfg.seed)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=num_workers)
    return loader
