"""Arioso evaluation (Section 10).

Model selection runs on **vocoder-independent** signals (a frozen vocoder must not be the
arbiter): velocity / reconstruction MSE, MCD (mel-cepstral distortion), and Delta-target-mel
inspection. The copy-synthesis sanity (step 0) establishes the vocoder ceiling before any
training. FAD + MUSHRA are deferred to the later perceptual gate.
"""
