"""Vibrato fitting: synthetic round-trip, robustness, rejection, store + hash wiring."""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

from common import vibrato_model as V
from common.config import SR
from Labeler import notes_store
from Labeler.compile import notes_hash
from Labeler.config import load_config
from Labeler.library import BACKUP_DIR, clip_dir

FPS = SR / V.F0_HOP           # 172.27 frames/s — the timebase every fit sees


@pytest.fixture()
def cfg(tmp_path):
    return dataclasses.replace(load_config(), dataset_root=str(tmp_path))


def _synth_cents(params, dur_s=0.6, noise=1.0, seed=0):
    """A rampboth5 cents trace on the real frame grid + gaussian measurement noise."""
    t = np.arange(int(round(dur_s * FPS))) / FPS
    cents = V.vibrato_cents(params, t, model="rampboth5")
    if noise:
        cents = cents + np.random.default_rng(seed).normal(0.0, noise, size=len(t))
    return t, cents


# --- synthetic round-trip -------------------------------------------------------

TRUE = [18.0, -12.0, 5.6, 1.5, 0.7]      # D0, gD, f0, s, phi


def test_roundtrip_recovers_rate_and_depth():
    t, cents = _synth_cents(TRUE)
    got = V.MODELS["rampboth5"].fit(t, cents, 0.6)
    assert got is not None and len(got) == 5
    assert abs(got[2] - TRUE[2]) < 0.15, f"rate off: {got}"
    assert abs(got[0] - TRUE[0]) < 1.5, f"depth off: {got}"
    # The recovered curve is what actually matters — sub-noise agreement.
    resid = V.vibrato_cents(got, t) - V.vibrato_cents(TRUE, t)
    assert float(np.sqrt(np.mean(resid ** 2))) < 1.5


def test_roundtrip_params_are_rounded_to_4dp():
    t, cents = _synth_cents(TRUE)
    got = V.MODELS["rampboth5"].fit(t, cents, 0.6)
    assert all(v == round(v, 4) for v in got)


def test_alternate_models_also_fit():
    t, cents = _synth_cents([18.0, 0.0, 5.6, 0.0, 0.7])   # no ramps -> lfo3 territory
    lfo = V.MODELS["lfo3"].fit(t, cents, 0.6)
    dr = V.MODELS["dramp4"].fit(t, cents, 0.6)
    assert lfo is not None and abs(lfo[1] - 5.6) < 0.15   # [D, f, phi]
    assert dr is not None and abs(dr[2] - 5.6) < 0.15     # [D0, gD, f, phi]


# --- robustness -----------------------------------------------------------------

def test_survives_unvoiced_gaps_and_an_octave_spike():
    t, cents = _synth_cents(TRUE)
    rng = np.random.default_rng(7)
    cents = cents.copy()
    cents[rng.choice(len(cents), size=int(0.2 * len(cents)), replace=False)] = np.nan
    cents[len(cents) // 2] += 1200.0            # pyin octave slip
    ok = V.mask_note(t, cents)
    assert not ok[len(cents) // 2], "despike/abs-max must drop the octave spike"
    got = V.MODELS["rampboth5"].fit(t[ok], cents[ok], 0.6)
    assert got is not None
    assert abs(got[2] - TRUE[2]) < 0.15 and abs(got[0] - TRUE[0]) < 2.0


def test_baseline_offset_and_slide_are_absorbed_not_stored():
    """A large intonation offset + slide must not leak into the stored params."""
    t, cents = _synth_cents(TRUE)
    got = V.MODELS["rampboth5"].fit(t, cents + 40.0 - 60.0 * t, 0.6)
    assert got is not None
    assert abs(got[2] - TRUE[2]) < 0.15 and abs(got[0] - TRUE[0]) < 2.0


# --- degenerate input -----------------------------------------------------------

def _f0_track(dur_s, pitch=69, value=440.0):
    n = int(round(dur_s * FPS))
    return np.full(n, float(value))


def test_rejects_too_short_note():
    assert V.fit_note_vibrato(_f0_track(2.0), 69, 0.0, 0.1) is None


def test_rejects_all_unvoiced_note():
    f0 = np.full(int(round(2.0 * FPS)), np.nan)
    assert V.fit_note_vibrato(f0, 69, 0.0, 1.0) is None


def test_rejects_too_few_frames():
    # A 2-frame slice of a long track (the duration floor catches this one first).
    f0 = _f0_track(2.0)
    assert V.fit_note_vibrato(f0, 69, 0.0, 2 / FPS) is None
    # ... and a note that is long enough but mostly unvoiced (the voiced-fraction floor).
    f0 = _f0_track(2.0)
    f0[: int(0.8 * len(f0))] = np.nan
    assert V.fit_note_vibrato(f0, 69, 0.0, 1.0) is None


def test_fit_note_vibrato_end_to_end_on_a_synthetic_track():
    """Whole path: Hz track -> cents slice -> mask -> fit."""
    dur, start = 1.0, 0.5
    n = int(round(2.0 * FPS))
    t_abs = np.arange(n) / FPS
    cents = np.zeros(n)
    inside = (t_abs >= start) & (t_abs < start + dur)
    cents[inside] = V.vibrato_cents(TRUE, t_abs[inside] - start)
    f0 = 440.0 * 2.0 ** (cents / 1200.0)
    got = V.fit_note_vibrato(f0, 69, start, start + dur)
    assert got is not None and abs(got[2] - TRUE[2]) < 0.1


# --- eval / render helpers ------------------------------------------------------

def test_vibrato_ratio_shape_and_center():
    r = V.vibrato_ratio([20.0, 0.0, 5.0, 0.0, 0.0], 1000, sr=44100)
    assert r.shape == (1000,) and r.dtype == np.float64
    assert r[0] == pytest.approx(1.0)                       # phase 0 -> no deviation
    # 20 cents half-depth -> ratio within 2**(+-20/1200)
    assert float(np.max(r)) <= 2.0 ** (20.0 / 1200.0) + 1e-9
    assert float(np.min(r)) >= 2.0 ** (-20.0 / 1200.0) - 1e-9


def test_depth_clip_is_applied_in_eval():
    # D0 + gD*t runs past D_CLIP; the envelope must saturate at 200 cents.
    t = np.linspace(0.0, 1.0, 200)
    c = V.vibrato_cents([100.0, 400.0, 5.0, 0.0, 0.0], t)
    assert float(np.max(np.abs(c))) <= V.D_CLIP + 1e-9


def test_unknown_model_raises_everywhere():
    for call in (lambda: V.vibrato_ratio([1, 2, 3], 10, model="nope"),
                 lambda: V.vibrato_cents([1, 2, 3], np.zeros(4), model="nope"),
                 lambda: V.fit_note_vibrato(_f0_track(2.0), 69, 0.0, 1.0, model="nope")):
        with pytest.raises(ValueError, match="unknown vibrato model"):
            call()


def test_wrong_param_count_raises():
    with pytest.raises(ValueError, match="takes 5 params"):
        V.vibrato_cents([1.0, 2.0], np.zeros(4))


def test_vibrato_from_auto_is_gated_and_bounded():
    p = V.vibrato_from_auto(5.5, 24.0)
    assert p == [12.0, 0.0, 5.5, 0.0, 0.0]
    assert V.vibrato_from_auto(0.0, 24.0) is None        # out of the 3-11 Hz band
    assert V.vibrato_from_auto(5.5, 0.0) is None         # no extent
    assert V.vibrato_from_auto(None, None) is None
    assert V.VIB_FALLBACK is False                        # wired but off


# --- notes_store.set_vibrato_params --------------------------------------------

def _note(start, pitch=60, vib=False):
    return {"start_s": start, "end_s": start + 0.8, "pitch": pitch, "velocity": 80,
            "technique": "normal", "vibrato": vib, "auto": {"velocity": 80}}


def _saved_doc(cfg, clip_id, notes):
    doc = notes_store.build_document(
        cfg, clip_id, notes, offset_applied_s=0.0, duration_s=5.0,
        audio_meta={"sr": 44100, "hop": 512}, pipeline_meta={})
    return notes_store.save_new(cfg, clip_id, doc)


def test_build_note_defaults_vib_fields_to_absent(cfg):
    doc = _saved_doc(cfg, "v0", [_note(0.0)])
    n = doc["notes"][0]
    assert n["vib_params"] is None and "vib_model" not in n


def test_set_vibrato_params_sets_clears_and_leaves_absent_alone(cfg):
    _saved_doc(cfg, "v1", [_note(0.0, vib=True), _note(1.0), _note(2.0)])
    # Seed all three so the "absent" case has something to preserve.
    notes_store.set_vibrato_params(
        cfg, "v1", {"n0000": [1, 2, 3, 4, 5], "n0001": [9, 9, 9, 9, 9],
                    "n0002": [8, 8, 8, 8, 8]}, "rampboth5")
    doc = notes_store.set_vibrato_params(
        cfg, "v1", {"n0000": [1.234567, 2.0, 5.5, 0.0, 0.1], "n0001": None},
        "rampboth5")
    by_id = {n["id"]: n for n in doc["notes"]}
    # set (rounded to 4 dp) + model stamped
    assert by_id["n0000"]["vib_params"] == [1.2346, 2.0, 5.5, 0.0, 0.1]
    assert by_id["n0000"]["vib_model"] == "rampboth5"
    # explicit None clears BOTH fields
    assert by_id["n0001"]["vib_params"] is None
    assert by_id["n0001"]["vib_model"] is None
    # absent id untouched
    assert by_id["n0002"]["vib_params"] == [8, 8, 8, 8, 8]


def test_set_vibrato_params_bumps_rev_and_backs_up(cfg):
    _saved_doc(cfg, "v2", [_note(0.0, vib=True)])
    before = notes_store.load(cfg, "v2")["rev"]
    doc = notes_store.set_vibrato_params(cfg, "v2", {"n0000": [1, 2, 3, 4, 5]},
                                         "rampboth5")
    assert doc["rev"] == before + 1
    assert notes_store.load(cfg, "v2")["rev"] == before + 1
    bdir = os.path.join(clip_dir(cfg, "v2"), BACKUP_DIR)
    assert any(f.startswith("rev_") for f in os.listdir(bdir))


def test_set_vibrato_params_on_missing_clip_returns_none(cfg):
    assert notes_store.set_vibrato_params(cfg, "nope", {"n0000": None}, "rampboth5") is None


def test_build_document_carries_vib_fields_through_a_rebuild(cfg):
    n = _note(0.0, vib=True)
    n["vib_params"] = [1.0, 2.0, 5.5, 0.0, 0.1]
    n["vib_model"] = "rampboth5"
    doc = notes_store.build_document(
        cfg, "v3", [n], offset_applied_s=0.0, duration_s=5.0,
        audio_meta={"sr": 44100, "hop": 512}, pipeline_meta={})
    assert doc["notes"][0]["vib_params"] == [1.0, 2.0, 5.5, 0.0, 0.1]
    assert doc["notes"][0]["vib_model"] == "rampboth5"


# --- compile provenance ---------------------------------------------------------

def _hash_note(**kw):
    base = {"start_s": 0.0, "end_s": 1.0, "pitch": 60, "velocity": 80,
            "technique": "normal", "vibrato": True}
    base.update(kw)
    return base


def test_notes_hash_is_sensitive_to_vib_params():
    a = notes_hash([_hash_note()], [])
    b = notes_hash([_hash_note(vib_params=[1, 2, 3, 4, 5], vib_model="rampboth5")], [])
    c = notes_hash([_hash_note(vib_params=[1, 2, 3, 4, 5.0001], vib_model="rampboth5")], [])
    d = notes_hash([_hash_note(vib_params=[1, 2, 3, 4, 5], vib_model="lfo3")], [])
    assert len({a, b, c, d}) == 4, "each vibrato change must change the digest"
    # 5 dp below the 4 dp rounding is invisible (matches env_dct's contract).
    e = notes_hash([_hash_note(vib_params=[1, 2, 3, 4, 5.000001],
                               vib_model="rampboth5")], [])
    assert e == b
