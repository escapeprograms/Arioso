"""Labeler — interactive violin dataset labeling app (FastAPI backend).

Runs MTG's MUSC violin-transcription model on fresh recordings, then serves a
canvas editor for correcting notes / articulation / vibrato / velocity and
compiles **verified** clips into a standard Arioso dataset root (the
``common.dataset_schema`` layout Arioso trains on). Run from the project root as
``python -m Labeler.server`` (so ``import common`` resolves).

See ``Labeler/README.md`` for the memory-palace overview of each module.
"""
