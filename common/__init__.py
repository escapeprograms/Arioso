"""Shared code for the Arioso project.

Cross-cutting helpers that more than one top-level package needs (the
DataSynthesizer pipeline today, the training code next): the canonical audio
constants in ``config`` and the audio I/O helpers in ``audio_io``.

**Reuse this — do not re-implement.** New modules should import
``common.audio_io`` / ``common.config`` rather than writing their own
``librosa.load`` / PCM-16 write / normalization / ``SR`` constant. See
``common/README.md``.
"""
