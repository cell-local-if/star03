"""Signed-access authentication for protected attestation routes.

A caller authenticates a protected request with three headers:

``X-PA``
    The acting actor id.
``X-PT``
    The request timestamp as an RFC 3339 UTC string, no more than 300
    seconds from the current time in either direction.
``X-PS``
    A Base64 (RFC 4648) Ed25519 signature, exactly 64 bytes when decoded.

The signature is verified over the exact UTF-8 canonical JSON bytes
``["provenance-access-v1", method, path, timestamp, body_sha256]`` (compact
separators, non-ASCII unescaped), where ``body_sha256`` is the lowercase
hex SHA-256 digest of the actual request body (the SHA-256 of zero bytes
for an empty body). The signature must verify under the public key of any
attestation by that actor that carries no revocation; an attestation
whose key was revoked no longer authenticates.

The write path treats any malformed element as a client validation error
(422). The read path never distinguishes "missing target",
"unauthenticated", and "forbidden": every such failure is reported as a
missing resource (404), so the endpoint neither authenticates nor
reveals a protected resource.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from datetime import datetime, timedelta, timezone

from sqlalchemy import exists, select
from sqlalchemy.orm import Session
from starlette.requests import Request

from provenance import ed25519
from provenance.errors import AccessValidationError
from provenance.models import Attestation, AttestationRevocation
from provenance.signing import (
    ACCESS_TIMESTAMP_TOLERANCE_SECONDS,
    access_message_bytes,
)

#: Header carrying the acting actor id.
HEADER_ACTOR = "X-PA"
#: Header carrying the RFC 3339 UTC timestamp.
HEADER_TIMESTAMP = "X-PT"
#: Header carrying the Base64 Ed25519 signature.
HEADER_SIGNATURE = "X-PS"

# Strict RFC 3339 with a mandatory UTC designator (``Z`` or a zero
# offset). Date/time digit shape is validated here; calendar validity and
# the timestamp window are checked after parsing.
_RFC3339_UTC_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]00:00)$"
)


def _single_header(request: Request, name: str) -> str | None:
    """Return a header provided exactly once, else ``None``."""
    values = request.headers.getlist(name)
    if len(values) != 1:
        return None
    value = values[0]
    return value or None


def _parse_timestamp(raw: str) -> datetime:
    """Parse a strict RFC 3339 UTC timestamp or raise ``AccessValidationError``."""
    if not _RFC3339_UTC_RE.fullmatch(raw):
        raise AccessValidationError("invalid_timestamp")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        # Syntactically shaped but not a real calendar date/time.
        raise AccessValidationError("invalid_timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise AccessValidationError("invalid_timestamp")
    delta = abs(parsed - datetime.now(timezone.utc))
    if delta > timedelta(seconds=ACCESS_TIMESTAMP_TOLERANCE_SECONDS):
        raise AccessValidationError("invalid_timestamp")
    return parsed


def _decode_signature(raw: str) -> bytes:
    """Strictly decode the standard-Base64 64-byte Ed25519 signature."""
    try:
        signature = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        raise AccessValidationError("invalid_signature") from None
    if base64.b64encode(signature).decode("ascii") != raw:
        raise AccessValidationError("invalid_signature")
    if len(signature) != ed25519.SIGNATURE_LENGTH:
        raise AccessValidationError("invalid_signature")
    return signature


def _non_revoked_public_keys(session: Session, actor_id: str) -> list[bytes]:
    """Return distinct public keys on the actor's non-revoked attestations."""
    revoked = exists().where(
        AttestationRevocation.attestation_id == Attestation.id
    )
    return list(
        session.execute(
            select(Attestation.public_key)
            .where(Attestation.signer_actor_id == actor_id, ~revoked)
            .distinct()
        )
        .scalars()
        .all()
    )


async def authenticate(request: Request, session: Session) -> str:
    """Authenticate a protected request, returning the acting actor id.

    Raises :class:`AccessValidationError` (422) when any access header is
    missing, repeated, or blank; the timestamp is malformed or outside the
    tolerance window; the signature is malformed; or no non-revoked
    attestation key of the actor verifies it.
    """
    actor_id = _single_header(request, HEADER_ACTOR)
    timestamp = _single_header(request, HEADER_TIMESTAMP)
    signature_raw = _single_header(request, HEADER_SIGNATURE)
    if actor_id is None or timestamp is None or signature_raw is None:
        raise AccessValidationError("missing_access_headers")

    _parse_timestamp(timestamp)
    signature = _decode_signature(signature_raw)

    body = await request.body()
    body_sha256 = hashlib.sha256(body).hexdigest()
    message = access_message_bytes(
        request.method, request.url.path, timestamp, body_sha256
    )

    for public_key in _non_revoked_public_keys(session, actor_id):
        if ed25519.verify(public_key, message, signature):
            return actor_id
    raise AccessValidationError("signature_verification_failed")


async def authenticate_or_none(
    request: Request, session: Session
) -> str | None:
    """Authenticate a protected read, returning the actor id or ``None``.

    A protected read never reports *why* access is unavailable: any
    malformed or unverifiable request is indistinguishable from a missing
    target to the caller.
    """
    try:
        return await authenticate(request, session)
    except AccessValidationError:
        return None
