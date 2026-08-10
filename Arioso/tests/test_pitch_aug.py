"""On-the-fly pitch-shift augmentation: byte-identity when off, wav path when on, RNG contract.

Pins what ``Arioso.pitch_aug`` + the dataset wiring promise (all pure CPU, tmp_path roots only):

* **Off is off.** ``pitch_aug_p == 0`` (and ``pitch_aug=None``) yields items byte-identical to the
  memmap path, and draws no RNG at all — a baseline run cannot change.
* **Train only.** ``build_pitch_aug`` returns a structurally inactive policy for any split but
  ``"train"``, whatever the config says.
* **Graceful fallback.** A root missing ``prior_wav/`` (or ``gt/``) is never augmented, even at
  ``p == 1.0`` — the roots predating the rendered prior waveform still train.
* **Alignment.** An augmented item keeps its exact frame count and its cond/boundary tracks
  bit-for-bit; the shift is duration-preserving, so only ``x0``/``x1`` move.
* **Slice geometry.** ``shifted_pair`` at ``cents == 0`` reproduces the whole-file mel's slice
  (exactly at the file head, to float round-off in the interior).
* **RNG.** The stream is a function of ``(seed, epoch, index)`` alone — reproducible, independent of
  global numpy state, gate rate and support as configured.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest
import soundfile as sf
import torch

from common.config import HOP_SIZE, N_MELS, SR
from common.dataset_schema import (DIR_GT, DIR_ONSETS, DIR_PRIOR_MEL, DIR_PRIOR_WAV,
                                   DIR_TARGET_MEL, MANIFEST_SCHEMA_VERSION, DatasetRoot,
                                   cond_dir, write_manifest)
from common.audio_io import write_pcm16
from common.keyshift import margin_frames
from common.vocoder import mel_frames

from Arioso.clips import Clip
from Arioso.config import (BOUNDARY_SIGNALS, SIGNALS, AriosoConfig, cfg_from_dict,
                           cfg_to_dict)
from Arioso.dataset import AriosoDataset
from Arioso.pitch_aug import PitchAug, build_pitch_aug, shifted_pair, wavs_available

T_FRAMES = 60                       # clip length in mel frames; L = T_FRAMES * HOP_SIZE samples
BASE = "clip000"
SPECS = (SIGNALS["articulation"], BOUNDARY_SIGNALS["time_since_onset"])
AMPS = (1.0, 0.5, 0.25)             # a 3-partial "violin-ish" harmonic ladder


# --- tmp_path root fixtures ------------------------------------------------------

def _signal(n_samples: int, f0: float, seed: int) -> np.ndarray:
    """A seeded ``n_samples``-long 3-partial tone at ``f0`` Hz, float32, safely inside [-1, 1]."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64) / SR
    y = sum(a * np.sin(2 * np.pi * f0 * i * t + rng.uniform(0.0, 2 * np.pi))
            for i, a in enumerate(AMPS, start=1))
    y = y + 0.02 * rng.standard_normal(n_samples)
    return (0.6 * y / np.abs(y).max()).astype(np.float32)


def _write_root(path: str, *, prior_wav: bool = True, gt_wav: bool = True) -> DatasetRoot:
    """Write a one-clip standard root (manifest + mels + onsets + cond + wavs) and open it.

    The mels are melled from the **PCM16 round-tripped** waveforms (write, read back, mel), so a
    ``cents == 0`` re-mel of the wav reproduces the saved npy and the byte-identity comparisons are
    about the augmentation, not about a 16-bit quantization gap. ``prior_wav`` / ``gt_wav`` drop the
    corresponding waveform to exercise the graceful fallback (the npy mel is written either way).
    """
    n_samples = T_FRAMES * HOP_SIZE            # T = 1 + (L - HOP) // HOP == T_FRAMES
    for sub in (DIR_GT, DIR_PRIOR_WAV, DIR_TARGET_MEL, DIR_PRIOR_MEL, DIR_ONSETS,
                cond_dir("articulation")):
        os.makedirs(os.path.join(path, sub), exist_ok=True)

    for kind, f0, seed, keep in (("prior", 220.0, 0, prior_wav), ("gt", 223.0, 1, gt_wav)):
        wav_path = os.path.join(path, DIR_PRIOR_WAV if kind == "prior" else DIR_GT, BASE + ".wav")
        write_pcm16(wav_path, _signal(n_samples, f0, seed))
        round_tripped, _sr = sf.read(wav_path, dtype="float32")
        mel = mel_frames(round_tripped)
        np.save(os.path.join(path, DIR_PRIOR_MEL if kind == "prior" else DIR_TARGET_MEL,
                             BASE + ".npy"), mel)
        if not keep:
            os.remove(wav_path)

    np.save(os.path.join(path, DIR_ONSETS, BASE + ".npy"),
            np.array([0, 20, 40], dtype=np.int32))
    np.save(os.path.join(path, cond_dir("articulation"), BASE + ".npy"),
            np.arange(T_FRAMES, dtype=np.uint8) % 4)

    write_manifest(path, {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "name": "pitch_aug_test_root",
        "frame": {"sr": SR, "hop": HOP_SIZE, "n_mels": N_MELS},
        "signals": ["articulation"],
        "clips": {BASE: {"n_frames": T_FRAMES, "piece": BASE}},
    })
    return DatasetRoot(path)


CLIPS = [Clip(0, BASE, 0, 30), Clip(0, BASE, 10, 40), Clip(0, BASE, 25, T_FRAMES)]


@pytest.fixture()
def root(tmp_path) -> DatasetRoot:
    return _write_root(str(tmp_path / "root"))


def _dataset(root: DatasetRoot, aug: PitchAug | None) -> AriosoDataset:
    return AriosoDataset([root], CLIPS, specs=SPECS, pitch_aug=aug)


# --- off is off ------------------------------------------------------------------

def test_p_zero_items_are_byte_identical_to_the_no_aug_dataset(root):
    plain = _dataset(root, None)
    off = _dataset(root, PitchAug(p=0.0, max_cents=100.0, seed=0))
    for i in range(len(CLIPS)):
        a, b = plain[i], off[i]
        assert torch.equal(a["x0"], b["x0"]), f"clip {i}: x0 moved with p=0"
        assert torch.equal(a["x1"], b["x1"]), f"clip {i}: x1 moved with p=0"
        assert a["length"] == b["length"]


def test_p_zero_draws_no_rng(root, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("an inactive PitchAug must not construct a generator")

    monkeypatch.setattr("Arioso.pitch_aug.np.random.default_rng", boom)
    aug = PitchAug(p=0.0, max_cents=100.0, seed=0)
    assert aug.cents_for(0, 0) is None
    ds = _dataset(root, aug)
    assert ds[0]["x0"].shape == (N_MELS, 30)     # __getitem__ still works


def test_max_cents_zero_is_inactive():
    assert PitchAug(p=1.0, max_cents=0.0, seed=0).active is False
    assert PitchAug(p=1.0, max_cents=0.0, seed=0).cents_for(3, 7) is None


# --- train-only wiring ------------------------------------------------------------

def test_build_pitch_aug_is_inactive_for_val():
    cfg = AriosoConfig(pitch_aug_p=1.0)
    assert build_pitch_aug(cfg, "val").active is False
    assert build_pitch_aug(cfg, "val").p == 0.0


def test_build_pitch_aug_carries_the_train_knobs():
    cfg = AriosoConfig(pitch_aug_p=0.5, pitch_aug_cents=50.0, seed=7)
    aug = build_pitch_aug(cfg, "train")
    assert (aug.p, aug.max_cents, aug.seed) == (0.5, 50.0, 7)
    assert aug.active and aug.margin == margin_frames(50.0)


# --- graceful fallback on roots without the waveforms -----------------------------

@pytest.mark.parametrize("missing", ["prior_wav", "gt"])
def test_root_without_a_waveform_is_never_augmented(tmp_path, missing):
    root = _write_root(str(tmp_path / missing),
                       prior_wav=(missing != "prior_wav"), gt_wav=(missing != "gt"))
    assert wavs_available(root, BASE) is False
    plain = _dataset(root, None)
    aug = _dataset(root, PitchAug(p=1.0, max_cents=100.0, seed=0))
    for i in range(len(CLIPS)):
        assert torch.equal(plain[i]["x0"], aug[i]["x0"]), f"clip {i}: x0 augmented anyway"
        assert torch.equal(plain[i]["x1"], aug[i]["x1"]), f"clip {i}: x1 augmented anyway"


# --- an augmented item keeps its framing and its conditioning ---------------------

def test_augmented_item_keeps_shape_dtype_length_and_cond(root):
    plain = _dataset(root, None)
    ds = _dataset(root, PitchAug(p=1.0, max_cents=100.0, seed=0))
    for i, clip in enumerate(CLIPS):
        want = clip.end - clip.start
        item, ref = ds[i], plain[i]
        assert item["x0"].shape == (N_MELS, want)
        assert item["x1"].shape == (N_MELS, want)
        assert item["x0"].dtype == torch.float32 and item["x1"].dtype == torch.float32
        assert item["length"] == ref["length"] == want
        # The shift is duration-preserving: every cond/boundary track is untouched.
        assert set(item["cond"]) == set(ref["cond"])
        for name, track in item["cond"].items():
            assert torch.equal(track, ref["cond"][name]), f"clip {i}: cond {name} moved"
        # ...and the mels really did change (this is the augmented path, not a silent fallback).
        assert not torch.equal(item["x0"], ref["x0"])
        assert not torch.equal(item["x1"], ref["x1"])


def test_epoch_changes_the_shifts(root):
    ds = _dataset(root, PitchAug(p=1.0, max_cents=100.0, seed=0))
    first = ds[0]["x0"].clone()
    ds.set_epoch(1)
    assert not torch.equal(first, ds[0]["x0"])
    ds.set_epoch(0)
    assert torch.equal(first, ds[0]["x0"])          # epoch is the only state; same epoch, same mel


# --- shifted_pair slice geometry at cents == 0 ------------------------------------

def _whole_file_mel(path: str) -> np.ndarray:
    """Mel of the entire (PCM16 round-tripped) file — the oracle a slice must reproduce."""
    wav, _sr = sf.read(path, dtype="float32")
    return mel_frames(wav)


def test_zero_cents_interior_slice_matches_the_whole_file_mel(root):
    start, end = 10, 40
    x0, x1 = shifted_pair(root, BASE, start, end, 0.0, margin_frames(100.0))
    for got, path in ((x0, root.prior_wav_path(BASE)), (x1, root.gt_path(BASE))):
        ref = _whole_file_mel(path)[:, start:end]
        assert got.shape == ref.shape
        delta = np.abs(got - ref).max()
        assert delta < 1e-5, f"{path}: interior slice max|delta| {delta}"


def test_zero_cents_head_slice_is_exact(root):
    # start == 0: the slice's own reflect pad IS the whole file's, so this is exact, not close.
    x0, x1 = shifted_pair(root, BASE, 0, 30, 0.0, margin_frames(100.0))
    assert np.array_equal(x0, _whole_file_mel(root.prior_wav_path(BASE))[:, 0:30])
    assert np.array_equal(x1, _whole_file_mel(root.gt_path(BASE))[:, 0:30])


def test_shifted_pair_returns_float32_pair_of_the_requested_width(root):
    x0, x1 = shifted_pair(root, BASE, 25, T_FRAMES, 75.0, margin_frames(100.0))
    for arr in (x0, x1):
        assert arr.shape == (N_MELS, T_FRAMES - 25)
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()


# --- the RNG contract -------------------------------------------------------------

def test_cents_for_is_a_pure_function_of_seed_epoch_index():
    a = PitchAug(p=0.5, max_cents=100.0, seed=3)
    b = PitchAug(p=0.5, max_cents=100.0, seed=3)
    draws_a = [a.cents_for(e, i) for e in range(3) for i in range(50)]
    draws_b = [b.cents_for(e, i) for e in range(3) for i in range(50)]
    assert draws_a == draws_b


def test_different_epochs_give_different_draws():
    aug = PitchAug(p=1.0, max_cents=100.0, seed=0)
    e0 = [aug.cents_for(0, i) for i in range(50)]
    e1 = [aug.cents_for(1, i) for i in range(50)]
    assert e0 != e1
    assert sum(x == y for x, y in zip(e0, e1)) == 0


def test_global_numpy_seed_does_not_perturb_the_stream():
    aug = PitchAug(p=0.5, max_cents=100.0, seed=0)
    before = [aug.cents_for(0, i) for i in range(100)]
    np.random.seed(12345)
    np.random.random(10)
    after = [aug.cents_for(0, i) for i in range(100)]
    assert before == after


@pytest.mark.parametrize("p", [0.25, 0.5, 0.9])
def test_gate_rate_and_support_over_20k_draws(p):
    aug = PitchAug(p=p, max_cents=100.0, seed=0)
    draws = [aug.cents_for(0, i) for i in range(20_000)]
    hits = [d for d in draws if d is not None]
    rate = len(hits) / len(draws)
    assert abs(rate - p) < 0.02, f"gate rate {rate:.4f} vs p={p}"
    arr = np.array(hits)
    assert arr.min() >= -100.0 and arr.max() <= 100.0
    assert abs(float(arr.mean())) < 2.0, f"cents mean {arr.mean():.3f} (expected ~0)"


# --- config knobs ------------------------------------------------------------------

def test_cfg_dict_roundtrips_the_pitch_aug_knobs():
    cfg = AriosoConfig(pitch_aug_p=0.5, pitch_aug_cents=42.0)
    back = cfg_from_dict(cfg_to_dict(cfg))
    assert back.pitch_aug_p == 0.5
    assert back.pitch_aug_cents == 42.0


def test_post_init_rejects_out_of_range_knobs():
    with pytest.raises(AssertionError):
        AriosoConfig(pitch_aug_p=1.5)
    with pytest.raises(AssertionError):
        AriosoConfig(pitch_aug_cents=-1.0)


def test_default_yaml_still_resolves_to_the_code_defaults():
    # The default.yaml == defaults guard, extended by the two new model keys + run.num_workers.
    from Arioso.run_config import RunSettings, load_config

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg, run = load_config(os.path.join(here, "configs", "default.yaml"))
    assert cfg == AriosoConfig()
    assert run == RunSettings()
    assert cfg.pitch_aug_p == 0.0 and cfg.pitch_aug_cents == 100.0
    assert run.num_workers == 0


def test_shipped_pitch_aug_yaml_resolves():
    from Arioso.run_config import load_config

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg, run = load_config(os.path.join(here, "configs", "8-8-pitch-aug.yaml"))
    assert cfg.pitch_aug_p == 0.5 and cfg.pitch_aug_cents == 100.0
    assert run.num_workers == 4
    assert build_pitch_aug(cfg, "train").active is True


def test_dataclasses_replace_keeps_the_knobs():
    # train.py rebuilds RunSettings via dataclasses.replace; the new field must survive.
    from Arioso.run_config import RunSettings

    run = dataclasses.replace(RunSettings(num_workers=4), name="x")
    assert run.num_workers == 4
