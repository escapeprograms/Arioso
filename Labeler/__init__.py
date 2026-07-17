"""Labeler — interactive violin dataset labeling app (FastAPI backend).

Runs MTG's MUSC violin-transcription model on fresh recordings, then serves a
canvas editor for correcting notes / articulation / vibrato / velocity and
exports artifacts the DataSynthesizer prior + Arioso conditioning pipeline
consumes. Run from the project root as ``python -m Labeler.server`` (so
``import common`` / ``import DataSynthesizer`` resolve).

See ``Labeler/README.md`` for the memory-palace overview of each module.
"""
