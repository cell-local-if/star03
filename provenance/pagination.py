"""Versioned, signed, opaque pagination cursors.

A cursor captures the exact position within one filtered result set and binds
to every effective query parameter of the request that produced it. Cursors
are opaque to clients: the payload is JSON wrapped in URL-safe Base64 and
authenticated with HMAC-SHA256 using a server-held secret, so a cursor cannot
be forged or tampered with. Any malformed, unsigned, wrong-version, or
structurally invalid token is rejected by the caller as a
``422 validation_error`` rather than trusted.

Cursors are namespaced into *kinds*. Each kind has its own format marker,
claim set, and claim validator, so a token minted for one endpoint family can
never resume another: the marker is checked before the signature is trusted
and the decoded claim set must match that kind exactly. The lineage family
keeps its historical ``v1`` token format verbatim.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from typing import Any, Callable

#: Cursor format markers per family. Bump only when that family's claim
#: set/encoding changes; cursors carrying any other marker are rejected as
#: expired/unknown (a foreign marker is a cross-family replay attempt).
LINEAGE_CURSOR_VERSION = "v1"
EVIDENCE_BUNDLE_CURSOR_VERSION = "v1eb"

#: Backwards-compatible alias for the lineage marker.
CURSOR_VERSION = LINEAGE_CURSOR_VERSION

#: Claims whose exact values must match the request carrying the cursor, in
#: stable serialized order, per cursor family. ``offset`` is the position
#: state itself and is present in every family.
_LINEAGE_CLAIM_FIELDS = (
    "content_id",
    "direction",
    "max_depth",
    "min_depth",
    "relation_type",
    "limit",
    "offset",
)
_EVIDENCE_BUNDLE_CLAIM_FIELDS = (
    "content_id",
    "evidence_type",
    "media_type",
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


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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


def _optional_nonempty_str(value: Any, field: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value:
        raise InvalidCursorError(f"cursor {field} is invalid")


def _validate_evidence_bundle_claims(claims: dict[str, Any]) -> None:
    if not isinstance(claims["content_id"], str) or not claims["content_id"]:
        raise InvalidCursorError("cursor content_id is invalid")
    # Both filters are either absent (unfiltered) or non-empty exact values.
    _optional_nonempty_str(claims["evidence_type"], "evidence_type")
    _optional_nonempty_str(claims["media_type"], "media_type")
    for field in ("limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    # As for lineage, only server-minted continuation tokens carry an offset,
    # and the first page carries no cursor; a non-positive offset is invalid.
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Marker, serialized claim order, and structural validator per family.
_KIND_SPECS: dict[str, tuple[str, tuple[str, ...], Callable[[dict[str, Any]], None]]] = {
    "lineage": (
        LINEAGE_CURSOR_VERSION,
        _LINEAGE_CLAIM_FIELDS,
        _validate_lineage_claims,
    ),
    "evidence_bundle": (
        EVIDENCE_BUNDLE_CURSOR_VERSION,
        _EVIDENCE_BUNDLE_CLAIM_FIELDS,
        _validate_evidence_bundle_claims,
    ),
}


def encode_cursor(
    secret: bytes, claims: dict[str, Any], kind: str = "lineage"
) -> str:
    """Return an opaque, signed token for normalized pagination ``claims``.

    ``kind`` selects the cursor family (its marker and claim set); it
    defaults to the lineage family for backwards compatibility.
    """
    version, fields, _ = _KIND_SPECS[kind]
    payload = _b64encode(
        json.dumps(
            {field: claims[field] for field in fields},
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    signature = _b64encode(_sign(secret, version, payload))
    return f"{version}.{payload}.{signature}"


def decode_cursor(
    secret: bytes, token: str, kind: str = "lineage"
) -> dict[str, Any]:
    """Verify and decode ``token``, returning its normalized claims dict.

    The token must belong to the ``kind`` family: a marker from another
    family is rejected before any claim is trusted. Raises
    :class:`InvalidCursorError` for an empty token, wrong number of
    segments, an unknown/foreign marker, bad Base64/JSON, a missing- or
    wrongly-typed claim, an out-of-range value, or a signature that does not
    authenticate the payload.
    """
    expected_version, expected_fields, validate = _KIND_SPECS[kind]
    if not isinstance(token, str) or not token:
        raise InvalidCursorError("cursor must not be empty")
    parts = token.split(".")
    if len(parts) != 3:
        raise InvalidCursorError("cursor has an invalid structure")
    version, payload, signature = parts
    if version != expected_version:
        # Covers unknown versions, expired formats, and tokens minted for a
        # different cursor family (cross-endpoint replay).
        raise InvalidCursorError("cursor uses an unknown or expired version")

    expected_signature = _b64encode(_sign(secret, version, payload))
    if not hmac.compare_digest(signature, expected_signature):
        raise InvalidCursorError("cursor signature is invalid")

    try:
        raw = _b64decode(payload)
        claims = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidCursorError("cursor payload is not valid JSON") from exc

    if not isinstance(claims, dict) or set(claims) != set(expected_fields):
        raise InvalidCursorError("cursor payload has an invalid claim set")
    validate(claims)
    return claims
