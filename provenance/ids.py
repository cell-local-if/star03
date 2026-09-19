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


def claim_id(content_id: str, actor_id: str, claim_type: str, payload_digest: str) -> str:
    """Return a stable ``clm_`` identifier for a claim identity.

    The identity is the tuple (content, actor, claim type, canonical payload
    digest); identical resubmissions therefore resolve to the same id. Fields
    are length-prefixed so client-supplied separators can never cause
    ambiguity between different tuples.
    """
    parts = (content_id, actor_id, claim_type, payload_digest)
    material = bytearray(b"claim-v1:")
    for part in parts:
        encoded = part.encode("utf-8")
        material.extend(str(len(encoded)).encode("ascii"))
        material.append(ord(":"))
        material.extend(encoded)
    return "clm_" + hashlib.sha256(bytes(material)).hexdigest()
