"""Stable server-side identifier generation.

Content identifiers are derived deterministically from the content identity
(algorithm + digest), so the same bytes always map to the same stable id and
ids are not guessable sequences. The actor id is client-supplied and is not
generated here.
"""

from __future__ import annotations

import hashlib


def content_id(digest_algorithm: str, digest_hex: str) -> str:
    """Return a stable ``cnt_``-prefixed identifier for a content identity."""
    material = f"content:{digest_algorithm}:{digest_hex}".encode("ascii")
    return "cnt_" + hashlib.sha256(material).hexdigest()
