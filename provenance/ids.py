"""Stable server-side identifier generation.

Content identifiers are derived deterministically from the content identity
(algorithm + digest), so the same bytes always map to the same stable id and
ids are not guessable sequences. The actor id is client-supplied and is not
generated here.
"""

from __future__ import annotations

import hashlib
import json


def content_id(digest_algorithm: str, digest_hex: str) -> str:
    """Return a stable ``cnt_``-prefixed identifier for a content identity."""
    material = f"content:{digest_algorithm}:{digest_hex}".encode("ascii")
    return "cnt_" + hashlib.sha256(material).hexdigest()


def claim_id(
    content_id: str,
    actor_id: str,
    claim_type: str,
    payload_digest_hex: str,
) -> str:
    """Return a stable ``clm_``-prefixed identifier for a claim identity.

    The material is a canonical JSON array so fields containing separators
    cannot collide with different field splits.
    """
    material = json.dumps(
        ["claim", content_id, actor_id, claim_type, payload_digest_hex],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "clm_" + hashlib.sha256(material).hexdigest()


def evidence_bundle_id(
    claim_id: str,
    evidence_type: str,
    digest_algorithm: str,
    digest_hex: str,
) -> str:
    """Return a stable ``evb_``-prefixed identifier for an evidence identity.

    As with claims, the material is a canonical JSON array so fields
    containing separators cannot collide with different field splits.
    """
    material = json.dumps(
        [
            "evidence_bundle",
            claim_id,
            evidence_type,
            digest_algorithm,
            digest_hex,
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "evb_" + hashlib.sha256(material).hexdigest()
