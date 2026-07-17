"""Model-free posterior decode: min-note-len knob + frame->second mapping.

Builds a synthetic MUSC posterior dict (numpy only — no torch/GPU/model) with a
short (~60 ms) and a long (~200 ms) note, then exercises ``decode_posteriors``:
the short note survives the recall-biased 45 ms floor but is deleted by MUSC's old
127.7 ms floor, while onset ordering and the frame->second conversion stay exact.
"""

from __future__ import annotations

import numpy as np

from Labeler.config import TranscribeParams
from Labeler.transcribe import decode_posteriors

SR = 44100
HOP = 256
FPS = SR / HOP                       # 172.265625, the MUSC violin frame rate

NOTE_LOW = 40                        # stands in for labeling.midi_centers[0]
NOTE_HIGH = 127                      # ... and midi_centers[-1]
N_FREQS = 88


def _ms_to_frames(ms: float) -> int:
    return int(round(ms / 1000.0 * FPS))


def _make_posteriors():
    """A (n_frames, 88) note/onset grid: one 60 ms note then one 200 ms note."""
    short_len = _ms_to_frames(60.0)      # 10 frames
    long_len = _ms_to_frames(200.0)      # 34 frames
    short_start, short_col = 20, 20      # -> pitch 60
    long_start, long_col = 60, 44        # -> pitch 84
    n_frames = 120                       # >= 11 trailing zero-frames after each note

    note = np.zeros((n_frames, N_FREQS), dtype=np.float64)
    onset = np.zeros((n_frames, N_FREQS), dtype=np.float64)
    note[short_start:short_start + short_len, short_col] = 1.0
    note[long_start:long_start + long_len, long_col] = 1.0
    onset[short_start, short_col] = 1.0
    onset[long_start, long_col] = 1.0

    time = np.arange(n_frames) * (HOP / SR)
    out = {"note": note, "onset": onset, "time": time}
    spec = {
        "short": (short_start, short_len, short_col + NOTE_LOW),
        "long": (long_start, long_len, long_col + NOTE_LOW),
    }
    return out, spec


def test_short_note_survives_recall_biased_floor():
    out, spec = _make_posteriors()
    params = TranscribeParams(onset_thresh=0.3, frame_thresh=0.2, min_note_len_ms=45.0)
    notes = decode_posteriors(out, (NOTE_LOW, NOTE_HIGH), params)

    # Both notes survive the 45 ms floor.
    assert len(notes) == 2

    # Sorted by (start_s, pitch): the short note (earlier onset) comes first.
    starts = [n[2] for n in notes]
    assert starts == sorted(starts)
    assert [n[4] for n in notes] == [spec["short"][2], spec["long"][2]] == [60, 84]

    # Frame -> second conversion is exact off out["time"] at the onset frame.
    assert notes[0][2] == out["time"][spec["short"][0]]
    assert notes[1][2] == out["time"][spec["long"][0]]
    # End maps the decoded end frame; each note spans ~its length (>=; <=1 frame).
    for note_ev, key in ((notes[0], "short"), (notes[1], "long")):
        _start_f, expect_len, _pitch = spec[key]
        assert abs((note_ev[1] - note_ev[0]) - expect_len) <= 1
        assert note_ev[3] == out["time"][note_ev[1]]
        assert note_ev[3] > note_ev[2]


def test_old_127ms_floor_deletes_short_note():
    out, spec = _make_posteriors()
    params = TranscribeParams(onset_thresh=0.3, frame_thresh=0.2, min_note_len_ms=127.7)
    notes = decode_posteriors(out, (NOTE_LOW, NOTE_HIGH), params)

    # Only the long note clears MUSC's original 127.7 ms floor.
    assert len(notes) == 1
    assert notes[0][4] == spec["long"][2] == 84
    assert notes[0][2] == out["time"][spec["long"][0]]


def test_min_note_len_frame_conversion():
    # The ms->frame conversion the decode uses internally: 45 ms -> 8, 127.7 -> 22.
    assert _ms_to_frames(45.0) == 8
    assert _ms_to_frames(127.7) == 22
    # ... and it is the min-length knob alone that flips the short note's fate.
    out, _ = _make_posteriors()
    lax = decode_posteriors(out, (NOTE_LOW, NOTE_HIGH),
                            TranscribeParams(min_note_len_ms=45.0))
    strict = decode_posteriors(out, (NOTE_LOW, NOTE_HIGH),
                               TranscribeParams(min_note_len_ms=127.7))
    assert len(lax) == 2 and len(strict) == 1
