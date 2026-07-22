"""DataSynthesizer — build Arioso's target + prior mel training features.

Run the modules as a package so intra-package imports resolve, e.g.::

    python -m DataSynthesizer.build_dataset --books Kayser --limit 2
    python -m DataSynthesizer.build_prior --limit 4

The prior synthesis and onset-alignment code now lives in ``common`` (``common.prior``,
``common.onset_align``), shared with Arioso inference / the Labeler / Studio.
See ``README.md`` (the module memory palace) for the full pipeline.
"""
