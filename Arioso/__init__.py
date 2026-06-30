"""Arioso — the OT-CFM acoustic model at the center of the violin-synthesis system.

Learns a velocity field ``v_theta`` that transports a score-synthesized sawtooth-prior
mel toward a real-violin target mel via Optimal-Transport Conditional Flow Matching,
entirely in the mel domain. This is the **no-EQ baseline** (``SPEC_Arioso_v1_baseline.md``):
the prior is an unshaped quantized sawtooth, so the model's job is almost entirely timbre.

Built on the shared infrastructure — ``common`` (audio I/O, mel contract, vocoder) and the
``DataSynthesizer`` dataset (``data/manifest.csv`` + per-clip target/prior mels). Run the
modules as a package so intra-package imports resolve, e.g.::

    python -m DataSynthesizer.build_prior --limit 4   # prior features (owned by DataSynthesizer)
    python -m Arioso.train --smoke
    python -m Arioso.infer <score.mid> -o out.wav

See ``README.md`` (the module memory palace) for the full pipeline.
"""
