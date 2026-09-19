"""Deterministic canonical JSON for claim payload digests.

A claim commits to its payload through a SHA-256 digest of a canonical
serialization, so the digest is stable across key orderings, insignificant
whitespace, and Unicode escaping choices. The raw payload itself is never
persisted or echoed back.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

CANONICAL_DIGEST_ALGORITHM = "sha256"


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize ``value`` deterministically: sorted keys, minimal separators.

    ``ensure_ascii=False`` plus UTF-8 encoding keeps the canonical form
    independent of how the input escaped non-ASCII characters.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def payload_digest_hex(payload: Any) -> str:
    """Return the SHA-256 hex digest of the canonical claim payload bytes."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


#: Domain separator for attestation signature messages.
ATTESTATION_DOMAIN = "provenance-attestation-v1"


def attestation_message_bytes(
    target_type: str, target_id: str, signer_actor_id: str
) -> bytes:
    """Return the canonical UTF-8 message an attestation signature commits to.

    The message is the compact JSON array
    ``["provenance-attestation-v1", target_type, target_id, signer_actor_id]``
    with no ASCII escaping, encoded as UTF-8, so signer and verifier derive
    byte-identical messages regardless of platform or escaping choices.
    """
    return json.dumps(
        [ATTESTATION_DOMAIN, target_type, target_id, signer_actor_id],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
