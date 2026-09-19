"""Versioned, signed, opaque pagination cursors for lineage queries.

A cursor captures the exact position within one filtered, depth-bounded
lineage result set and binds to every effective query parameter of the
request that produced it. Cursors are opaque to clients: the payload is
JSON wrapped in URL-safe Base64 and authenticated with HMAC-SHA256 using a
server-held secret, so a cursor cannot be forged or tampered with. Any
malformed, unsigned, wrong-version, or structurally invalid token is
rejected by the caller as a ``422 validation_error`` rather than trusted.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from typing import Any

#: Cursor format generation. Bump only when the claim set/encoding changes;
#: cursors carrying any other version are rejected as expired/unknown.
CURSOR_VERSION = "v1"

#: Claims whose exact values must match the request carrying the cursor, in
#: stable serialized order. ``offset`` is the position state itself.
_CLAIM_FIELDS = (
    "content_id",
    "direction",
    "max_depth",
    "min_depth",
    "relation_type",
    "limit",
    "offset",
)


class InvalidCursorError(ValueError):
    """The cursor token is absent, malformed, expired, or unverifiable."""


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    try:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode(value + padding)
    except (binascii.Error, ValueError) as exc:
        raise InvalidCursorError("cursor is not valid base64") from exc


def _sign(secret: bytes, version: str, payload: str) -> bytes:
    return hmac.new(
        secret, f"{version}.{payload}".encode("utf-8"), hashlib.sha256
    ).digest()


def encode_cursor(secret: bytes, claims: dict[str, Any]) -> str:
    """Return an opaque, signed token for normalized pagination ``claims``."""
    payload = _b64encode(
        json.dumps(
            {field: claims[field] for field in _CLAIM_FIELDS},
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    signature = _b64encode(_sign(secret, CURSOR_VERSION, payload))
    return f"{CURSOR_VERSION}.{payload}.{signature}"


def decode_cursor(secret: bytes, token: str) -> dict[str, Any]:
    """Verify and decode ``token``, returning its normalized claims dict.

    Raises :class:`InvalidCursorError` for an empty token, wrong number of
    segments, unknown version, bad Base64/JSON, a missing- or wrongly-typed
    claim, an out-of-range offset, or a signature that does not authenticate
    the payload.
    """
    if not isinstance(token, str) or not token:
        raise InvalidCursorError("cursor must not be empty")
    parts = token.split(".")
    if len(parts) != 3:
        raise InvalidCursorError("cursor has an invalid structure")
    version, payload, signature = parts
    if version != CURSOR_VERSION:
        raise InvalidCursorError("cursor uses an unknown or expired version")

    expected_signature = _b64encode(_sign(secret, version, payload))
    if not hmac.compare_digest(signature, expected_signature):
        raise InvalidCursorError("cursor signature is invalid")

    try:
        raw = _b64decode(payload)
        claims = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidCursorError("cursor payload is not valid JSON") from exc

    if not isinstance(claims, dict) or set(claims) != set(_CLAIM_FIELDS):
        raise InvalidCursorError("cursor payload has an invalid claim set")
    if not isinstance(claims["content_id"], str) or not claims["content_id"]:
        raise InvalidCursorError("cursor content_id is invalid")
    if claims["direction"] not in ("ancestors", "descendants"):
        raise InvalidCursorError("cursor direction is invalid")
    if not isinstance(claims["relation_type"], str | None) or (
        isinstance(claims["relation_type"], str)
        and claims["relation_type"] not in ("version_of", "derived_from")
    ):
        raise InvalidCursorError("cursor relation_type is invalid")
    for field in ("max_depth", "min_depth", "limit", "offset"):
        if not isinstance(claims[field], int) or isinstance(claims[field], bool):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["max_depth"] <= 32):
        raise InvalidCursorError("cursor max_depth is out of range")
    if not (1 <= claims["min_depth"] <= claims["max_depth"]):
        raise InvalidCursorError("cursor min_depth is out of range")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    # An offset of 0 is never issued (a first page carries no cursor); a
    # negative offset is structurally invalid. No tight upper bound is
    # imposed: a wide graph can legitimately page past any small constant,
    # and a client cannot forge a large value without the server secret.
    if not isinstance(claims["offset"], int) or claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")
    return claims
