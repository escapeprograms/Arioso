"""DataSynthesizer — build Arioso's target + prior mel training features.

Run the modules as a package so intra-package imports resolve, e.g.::

    python -m DataSynthesizer.build_dataset --books Kayser --limit 2
    python -m DataSynthesizer.onset_align prior.wav gt.wav

See ``README.md`` (the module memory palace) for the full pipeline.
"""
