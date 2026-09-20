"""Authentication for protected proof access (the ``X-PA``/``X-PT``/``X-PS`` contract).

Both ``POST /v1/attestation-access-grants`` and
``GET /v1/protected/attestations/{attestation_id}`` authenticate the caller
with three headers:

``X-PA``
    The calling actor id.
``X-PT``
    The request timestamp: an RFC 3339 UTC instant no more than 300 seconds
    from the server's current time.
``X-PS``
    A standard-Base64 Ed25519 signature (64 raw bytes) over the exact UTF-8
    compact JSON array::

        ["provenance-access-v1", method, path, timestamp, body_sha256]

    where ``method`` is the uppercase HTTP method, ``path`` is the request
    path (no query string), ``timestamp`` is the exact ``X-PT`` header
    value, and ``body_sha256`` is the lowercase-hex SHA-256 of the actual
    request body bytes (zero bytes for an empty body).

The signature is accepted if it verifies under **any** public key of a
non-revoked attestation created by the calling actor, or under an active
authentication key the actor introduced through a key rotation (a retired
rotation key never authenticates). Revoked attestation keys never
authenticate, regardless of the target the request operates on.

Failures split into two categories:

* *Malformed* credentials (an unparseable/out-of-window timestamp or a
  signature that is not canonical Base64 of exactly 64 bytes) are client
  input errors (``422``) on both routes.
* *Missing or unauthenticated* credentials (any header absent, or a
  well-formed signature no current non-revoked key of the actor verifies)
  are unauthenticated requests. The write route rejects them as ``422``;
  the read route answers with the same opaque ``404`` it uses for a missing
  target or an unauthorized actor, so existence is never revealed to a
  caller without access.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from datetime import datetime, timezone

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from provenance import ed25519
from provenance.models import (
    Attestation,
    AttestationRevocation,
    AuthenticationKeyRotation,
)

#: Required headers, in the order documented by the contract.
HEADER_ACTOR = "X-PA"
HEADER_TIMESTAMP = "X-PT"
HEADER_SIGNATURE = "X-PS"

#: Domain-separation prefix and protocol version of the signed array.
ACCESS_MESSAGE_PREFIX = "provenance-access-v1"

#: A request timestamp may differ from the server clock by at most this much.
MAX_TIMESTAMP_SKEW_SECONDS = 300

#: Strict RFC 3339 timestamp denoting UTC: ``Z`` or an explicit ``+00:00``.
#: Naive timestamps, non-UTC offsets, whitespace, and lowercase ``z`` fail.
_RFC3339_UTC_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)$"
)

# Failure reasons. ``MALFORMED_REASONS`` are client-input errors (422 even on
# reads); the rest are authentication failures (opaque 404 on reads).
REASON_MISSING_CREDENTIALS = "missing_credentials"
REASON_INVALID_TIMESTAMP = "invalid_timestamp"
REASON_TIMESTAMP_OUT_OF_WINDOW = "timestamp_out_of_window"
REASON_INVALID_SIGNATURE = "invalid_signature"
REASON_SIGNATURE_VERIFICATION_FAILED = "signature_verification_failed"

#: Reasons that render as ``422 validation_error`` even on the read route.
MALFORMED_REASONS = frozenset(
    {
        REASON_INVALID_TIMESTAMP,
        REASON_TIMESTAMP_OUT_OF_WINDOW,
        REASON_INVALID_SIGNATURE,
    }
)


class AccessAuthError(Exception):
    """A protected-request authentication failure, categorized by reason."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason

    @property
    def malformed(self) -> bool:
        """True for invalid input (422 on reads); False for unauthenticated."""
        return self.reason in MALFORMED_REASONS


def access_message_bytes(
    method: str, path: str, timestamp: str, body_sha256: str
) -> bytes:
    """Return the exact canonical bytes the access signature is committed to."""
    return json.dumps(
        [ACCESS_MESSAGE_PREFIX, method, path, timestamp, body_sha256],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def body_sha256_hex(body: bytes) -> str:
    """Return the lowercase SHA-256 hex of the actual request body bytes."""
    return hashlib.sha256(body).hexdigest()


def _parse_timestamp(raw: str) -> datetime:
    if not _RFC3339_UTC_RE.fullmatch(raw):
        raise AccessAuthError(REASON_INVALID_TIMESTAMP)
    try:
        # Python 3.11+ accepts the trailing "Z"; impossible calendar values
        # still raise ValueError.
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        raise AccessAuthError(REASON_INVALID_TIMESTAMP) from None
    skew = abs((parsed - datetime.now(timezone.utc)).total_seconds())
    if skew > MAX_TIMESTAMP_SKEW_SECONDS:
        raise AccessAuthError(REASON_TIMESTAMP_OUT_OF_WINDOW)
    return parsed


def _decode_signature(raw: str) -> bytes:
    """Decode canonical standard Base64 (RFC 4648) and require 64 bytes."""
    try:
        signature = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        raise AccessAuthError(REASON_INVALID_SIGNATURE) from None
    if base64.b64encode(signature).decode("ascii") != raw:
        raise AccessAuthError(REASON_INVALID_SIGNATURE)
    if len(signature) != ed25519.SIGNATURE_LENGTH:
        raise AccessAuthError(REASON_INVALID_SIGNATURE)
    return signature


def _actor_public_keys(session: Session, actor_id: str) -> list[bytes]:
    """Distinct current authentication keys of the actor.

    The set is the union of:

    * public keys of the actor's attestations carrying no revocation, and
    * public keys of the actor's active (non-retired) key rotations.

    A rotated key therefore authenticates without any attestation, an
    existing non-revoked attestation key keeps working, and a retired
    rotation key -- like a revoked attestation key -- never authenticates.
    """
    revoked = exists().where(
        AttestationRevocation.attestation_id == Attestation.id
    )
    attestation_keys = (
        session.execute(
            select(Attestation.public_key).where(
                Attestation.signer_actor_id == actor_id,
                ~revoked,
            )
        )
        .scalars()
        .all()
    )
    rotation_keys = (
        session.execute(
            select(AuthenticationKeyRotation.public_key).where(
                AuthenticationKeyRotation.actor_id == actor_id,
                AuthenticationKeyRotation.active.is_(True),
            )
        )
        .scalars()
        .all()
    )
    # De-duplicate: the same key bytes may be carried by both an attestation
    # and a rotation; verifying twice adds no security and costs a scalar
    # multiplication.
    return list(dict.fromkeys([*attestation_keys, *rotation_keys]))


def authenticate(session: Session, request, body: bytes) -> str:
    """Authenticate a protected request, returning the verified actor id.

    Raises :class:`AccessAuthError` (see its ``malformed`` flag) on any
    failure.
    """
    actor = request.headers.get(HEADER_ACTOR)
    timestamp = request.headers.get(HEADER_TIMESTAMP)
    signature_b64 = request.headers.get(HEADER_SIGNATURE)

    # Absent or blank credential material is an unauthenticated request,
    # distinct from present-but-malformed material.
    if not actor or not actor.strip():
        raise AccessAuthError(REASON_MISSING_CREDENTIALS)
    if not timestamp:
        raise AccessAuthError(REASON_MISSING_CREDENTIALS)
    if not signature_b64:
        raise AccessAuthError(REASON_MISSING_CREDENTIALS)

    _parse_timestamp(timestamp)
    signature = _decode_signature(signature_b64)

    message = access_message_bytes(
        request.method,
        request.url.path,
        timestamp,
        body_sha256_hex(body),
    )

    # Any one current (non-revoked) attestation key of the actor that
    # verifies the signature authenticates the request as that actor.
    for public_key in _actor_public_keys(session, actor):
        if ed25519.verify(public_key, message, signature):
            return actor

    raise AccessAuthError(REASON_SIGNATURE_VERIFICATION_FAILED)
