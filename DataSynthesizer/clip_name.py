"""Parse a dataset clip's filename — the single place the convention lives.

The violin-transcription dataset encodes everything in the MIDI filename::

    {Composer}_{Catalog}_{Performer}_{YouTubeID}-{startSec}-{endSec}.mid
    e.g. Kayser_Op20-01_AlexandrosIakovou_O105paQOHCE-0004-0064.mid

The YouTube id is the 11 characters immediately before the ``-start-end``
suffix: ids are always exactly 11 chars and use the base64url alphabet
``[A-Za-z0-9_-]``, so they may contain BOTH ``_`` and ``-`` (e.g.
``DE-KEDN8f8A``). Splitting on ``_`` would truncate such ids, so we slice a
fixed 11 chars and validate the charset instead.
"""

from __future__ import annotations

import os
import re
from typing import NamedTuple

_TIMES_RE = re.compile(r"-(\d+)-(\d+)$")
_YTID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


class ClipName(NamedTuple):
    youtube_id: str
    start: int
    end: int
    basename: str
    composer: str
    catalog: str
    performer: str


def parse_clip_name(path: str) -> ClipName:
    """Parse a clip path/filename into its :class:`ClipName` fields."""
    base = os.path.splitext(os.path.basename(path))[0]
    m = _TIMES_RE.search(base)
    if not m:
        raise ValueError(f"could not parse -start-end from: {base}")
    start, end = int(m.group(1)), int(m.group(2))

    prefix = base[:m.start()]          # "{Composer}_{Catalog}_{Performer}_{id}"
    youtube_id = prefix[-11:]          # ids are exactly 11 chars
    if not _YTID_RE.match(youtube_id):
        raise ValueError(f"bad YouTube id {youtube_id!r} parsed from: {base}")

    head = prefix[:-11].rstrip("_")    # drop the 11-char id (+ separator)
    parts = head.split("_")
    composer = parts[0] if parts else ""
    catalog = parts[1] if len(parts) > 1 else ""
    performer = "_".join(parts[2:]) if len(parts) > 2 else ""

    return ClipName(youtube_id, start, end, base, composer, catalog, performer)
