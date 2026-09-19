"""Attestation signing contract: the exact bytes a signer commits to.

An attestation proves that an existing signing actor endorses an existing
claim or evidence bundle. The signed message is a UTF-8 JSON array with
compact separators and non-ASCII characters left unescaped, so the bytes are
identical regardless of which language or client produced the signature::

    ["provenance-attestation-v1", target_type, target_id, signer_actor_id]

Only Ed25519 signatures over these exact bytes are accepted; see
:mod:`provenance.ed25519`.
"""

from __future__ import annotations

import json

#: Domain-separation prefix and protocol version of the signed array.
ATTESTATION_MESSAGE_PREFIX = "provenance-attestation-v1"

#: Attestable target types, matched verbatim in the signed message.
TARGET_CLAIM = "claim"
TARGET_EVIDENCE_BUNDLE = "evidence_bundle"
ATTESTATION_TARGET_TYPES = frozenset({TARGET_CLAIM, TARGET_EVIDENCE_BUNDLE})


def attestation_message_bytes(
    target_type: str, target_id: str, signer_actor_id: str
) -> bytes:
    """Return the exact canonical bytes the signer's signature is over."""
    return json.dumps(
        [ATTESTATION_MESSAGE_PREFIX, target_type, target_id, signer_actor_id],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


#: Domain-separation prefix and protocol version of a signed access request.
ACCESS_MESSAGE_PREFIX = "provenance-access-v1"

#: Maximum absolute age, in seconds, between the signed timestamp and now.
ACCESS_TIMESTAMP_TOLERANCE_SECONDS = 300


def access_message_bytes(
    method: str, path: str, timestamp: str, body_sha256: str
) -> bytes:
    """Return the exact canonical bytes an access signature is over.

    The five elements are the fixed protocol prefix, the HTTP method, the
    request path (exactly as served, including any route prefix), the
    RFC 3339 UTC timestamp from the request, and the lowercase hex SHA-256
    digest of the actual request body (zero bytes for an empty body).
    """
    return json.dumps(
        [ACCESS_MESSAGE_PREFIX, method, path, timestamp, body_sha256],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
