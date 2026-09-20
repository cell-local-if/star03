"""Versioned, signed, opaque pagination cursors.

A cursor captures the exact position within one filtered result set and
binds to every effective query parameter of the request that produced it.
Cursors are opaque to clients: the payload is JSON wrapped in URL-safe
Base64 and authenticated with HMAC-SHA256 using a server-held secret, so a
cursor cannot be forged or tampered with. Any malformed, unsigned,
wrong-version/kind, or structurally invalid token is rejected by the
caller as a ``422 validation_error`` rather than trusted.

Each cursor family (lineage traversal, a content's evidence-bundle list,
...) defines its own :class:`CursorKind`: a distinct format marker, claim
set, and claim validation. A cursor minted for one family can never resume
another family's query, even though all families share the server secret.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any, Callable

from provenance.time_utils import parse_rfc3339_utc

#: Legacy lineage cursor format generation. Bump only when the lineage claim
#: set/encoding changes; cursors carrying any other marker are rejected as
#: expired/unknown.
LINEAGE_CURSOR_VERSION = "v1"

#: Marker for cursors that page through one content's evidence bundles.
CONTENT_EVIDENCE_CURSOR_VERSION = "ce1"

#: Marker for cursors that page through the audit-event search.
AUDIT_EVENTS_CURSOR_VERSION = "ae1"


class InvalidCursorError(ValueError):
    """The cursor token is absent, malformed, expired, or unverifiable."""


@dataclass(frozen=True)
class CursorKind:
    """One cursor family: its wire marker, claim fields, and validation."""

    #: First token segment; namespaces the family and its format generation.
    version: str
    #: Claims whose exact values must match the request carrying the cursor,
    #: in stable serialized order. ``offset`` is the position state itself.
    claim_fields: tuple[str, ...]
    #: Raises :class:`InvalidCursorError` for a wrongly-typed/out-of-range
    #: claim of a structurally decoded payload.
    validate: Callable[[dict[str, Any]], None]


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


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _optional_nonempty_str(claims: dict[str, Any], field: str) -> None:
    value = claims[field]
    if value is not None and (not isinstance(value, str) or not value):
        raise InvalidCursorError(f"cursor {field} is invalid")


def _validate_lineage_claims(claims: dict[str, Any]) -> None:
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
        if not _is_int(claims[field]):
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
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Lineage traversal cursor family (kept under the original ``v1`` marker).
LINEAGE_CURSOR = CursorKind(
    version=LINEAGE_CURSOR_VERSION,
    claim_fields=(
        "content_id",
        "direction",
        "max_depth",
        "min_depth",
        "relation_type",
        "limit",
        "offset",
    ),
    validate=_validate_lineage_claims,
)


def _validate_content_evidence_claims(claims: dict[str, Any]) -> None:
    if not isinstance(claims["content_id"], str) or not claims["content_id"]:
        raise InvalidCursorError("cursor content_id is invalid")
    # The exact-match filters are either absent (null) or non-empty strings.
    _optional_nonempty_str(claims, "evidence_type")
    _optional_nonempty_str(claims, "media_type")
    for field in ("limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Cursor family for ``GET /v1/contents/{id}/evidence-bundles``.
CONTENT_EVIDENCE_CURSOR = CursorKind(
    version=CONTENT_EVIDENCE_CURSOR_VERSION,
    claim_fields=(
        "content_id",
        "evidence_type",
        "media_type",
        "limit",
        "offset",
    ),
    validate=_validate_content_evidence_claims,
)


def _validate_audit_events_claims(claims: dict[str, Any]) -> None:
    # The exact-match filters are either absent (null) or non-empty strings.
    _optional_nonempty_str(claims, "event_type")
    _optional_nonempty_str(claims, "resource_id")
    # The time bounds are absent (null) or canonical RFC 3339 UTC strings.
    for field in ("from", "to"):
        value = claims[field]
        if value is not None and (
            not isinstance(value, str) or parse_rfc3339_utc(value) is None
        ):
            raise InvalidCursorError(f"cursor {field} is invalid")
    for field in ("limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Cursor family for ``GET /v1/audit-events``.
AUDIT_EVENTS_CURSOR = CursorKind(
    version=AUDIT_EVENTS_CURSOR_VERSION,
    claim_fields=(
        "event_type",
        "resource_id",
        "from",
        "to",
        "limit",
        "offset",
    ),
    validate=_validate_audit_events_claims,
)


def encode_typed_cursor(
    secret: bytes, kind: CursorKind, claims: dict[str, Any]
) -> str:
    """Return an opaque, signed token for normalized pagination ``claims``."""
    payload = _b64encode(
        json.dumps(
            {field: claims[field] for field in kind.claim_fields},
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    signature = _b64encode(_sign(secret, kind.version, payload))
    return f"{kind.version}.{payload}.{signature}"


def decode_typed_cursor(
    secret: bytes, kind: CursorKind, token: str
) -> dict[str, Any]:
    """Verify and decode ``token``, returning its normalized claims dict.

    Raises :class:`InvalidCursorError` for an empty token, wrong number of
    segments, an unknown family/version marker, bad Base64/JSON, a missing-
    or wrongly-typed claim, an out-of-range offset/limit, or a signature
    that does not authenticate the payload.
    """
    if not isinstance(token, str) or not token:
        raise InvalidCursorError("cursor must not be empty")
    parts = token.split(".")
    if len(parts) != 3:
        raise InvalidCursorError("cursor has an invalid structure")
    version, payload, signature = parts
    if version != kind.version:
        raise InvalidCursorError("cursor uses an unknown or expired version")

    expected_signature = _b64encode(_sign(secret, version, payload))
    if not hmac.compare_digest(signature, expected_signature):
        raise InvalidCursorError("cursor signature is invalid")

    try:
        raw = _b64decode(payload)
        claims = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidCursorError("cursor payload is not valid JSON") from exc

    if not isinstance(claims, dict) or set(claims) != set(kind.claim_fields):
        raise InvalidCursorError("cursor payload has an invalid claim set")
    kind.validate(claims)
    return claims


def encode_cursor(secret: bytes, claims: dict[str, Any]) -> str:
    """Encode a lineage cursor (backward-compatible wrapper)."""
    return encode_typed_cursor(secret, LINEAGE_CURSOR, claims)


def decode_cursor(secret: bytes, token: str) -> dict[str, Any]:
    """Decode a lineage cursor (backward-compatible wrapper)."""
    return decode_typed_cursor(secret, LINEAGE_CURSOR, token)
