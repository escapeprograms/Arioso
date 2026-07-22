"""Studio — FL Studio–style piano-roll web app for Arioso violin inference.

Package marker. Run the backend as a module from the project root so
``import common`` / ``import Arioso`` resolve::

    python -m Studio.server

Import of the server / api / config surface is deliberately torch-free; the
model and vocoder are lazy singletons pulled in only inside the render pipeline.
The prior + mel front-end come from ``common.prior`` / ``common.vocoder``.
"""
