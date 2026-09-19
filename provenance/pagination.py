"""Stateless, tamper-evident pagination cursors for lineage queries.

A cursor encodes the exact query position (the effective filter parameters
and a non-negative offset) and is authenticated with HMAC-SHA256. Nothing is
stored server-side, so verification is fully offline and deterministic. A
cursor is accepted only if its signature verifies under the configured
secret, its payload shape/version is recognized, and every bound query
parameter matches the request.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Any

#: Payload schema version carried inside every cursor. A cursor minted by a
#: newer/older encoding is rejected rather than guessed at.
CURSOR_VERSION = 1

#: Defensive upper bound on an accepted token; genuine cursors are far
#: shorter than this.
_MAX_TOKEN_LENGTH = 2048

#: Required payload keys and their accepted primitive types. The offset
#: ``f`` ("from") is the zero-based index into the filtered result list.
_PAYLOAD_FIELDS = {
    "v": int,
    "o": str,  # origin content id
    "d": str,  # direction
    "x": int,  # effective max_depth
    "m": int,  # effective min_depth
    "r": str,  # relation type, or "" when unfiltered
    "l": int,  # effective limit
    "f": int,  # offset of the page the cursor begins at
}


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes | None:
    # Only the URL-safe alphabet with stripped canonical padding is accepted.
    # ``urlsafe_b64decode`` has no validate kwarg, so the alphabet is checked
    # explicitly (rejecting '+', '/', '=', whitespace, and any other byte).
    if not value or "=" in value or not all(
        ch.isalnum() or ch in "-_" for ch in value
    ):
        return None
    padded = value + "=" * (-len(value) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
    except (ValueError, TypeError):
        return None
    if _b64encode(raw) != value:
        return None
    return raw


def encode_lineage_cursor(
    secret: str,
    *,
    origin_id: str,
    direction: str,
    max_depth: int,
    min_depth: int,
    relation_type: str | None,
    limit: int,
    offset: int,
) -> str:
    """Return a signed, URL-safe cursor for one position in a lineage query."""
    payload = {
        "v": CURSOR_VERSION,
        "o": origin_id,
        "d": direction,
        "x": max_depth,
        "m": min_depth,
        "r": relation_type or "",
        "l": limit,
        "f": offset,
    }
    body = _b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signature = hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256)
    return f"{body}.{_b64encode(signature.digest())}"


def decode_lineage_cursor(secret: str, token: str) -> dict[str, Any] | None:
    """Validate and decode a cursor, returning its payload.

    Returns ``None`` for an empty, malformed, tampered, or
    unrecognized-version token. The signature is compared in constant time.
    """
    if not isinstance(token, str) or "." not in token or len(token) > _MAX_TOKEN_LENGTH:
        return None
    body, _, signature = token.partition(".")
    raw_body = _b64decode(body)
    raw_signature = _b64decode(signature)
    if raw_body is None or raw_signature is None:
        return None
    expected = hmac.new(
        secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256
    ).digest()
    if not hmac.compare_digest(raw_signature, expected):
        return None
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != set(_PAYLOAD_FIELDS):
        return None
    for key, expected_type in _PAYLOAD_FIELDS.items():
        # bool is a subclass of int: never accept booleans for integer slots.
        if not isinstance(payload[key], expected_type) or (
            expected_type is int and isinstance(payload[key], bool)
        ):
            return None
    if payload["v"] != CURSOR_VERSION:
        return None
    if payload["f"] < 0 or payload["l"] < 1 or payload["x"] < 1 or payload["m"] < 1:
        return None
    return payload
