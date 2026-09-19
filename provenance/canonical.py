"""Deterministic canonical JSON and payload digests.

Claims persist only a SHA-256 digest of their payload, never the payload
itself. The digest is reproducible across submissions, so canonicalization is
the contract that makes idempotency well-defined.

The canonical form follows the JCS (RFC 8785) rules:

* UTF-8 output with no insignificant whitespace;
* object members sorted by UTF-16 code unit order (RFC 8785 section 3.2.3);
* arrays preserve order;
* numbers use the shortest round-trip representation (Python 3's
  repr-based :func:`json.dumps` output), and finite integer-valued numbers
  render without a fractional part, so ``1`` and ``1.0`` canonicalize alike;
* non-finite numbers (``NaN``/``Infinity``) and bytes are rejected, since
  they are not representable in JSON. Only values parsed from a JSON request
  body are ever canonicalized here.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

#: Algorithm recorded alongside every claim payload digest.
DIGEST_ALGORITHM = "sha256"

#: Integer-valued floats up to this magnitude have an exact integral value
#: (2**53) and are serialized without a fractional part, per JCS.
_MAX_EXACT_INTEGER = 2**53


def _utf16_key_order(key: str) -> tuple[int, ...]:
    """Return the UTF-16 code units of *key* for RFC 8785 ordering.

    Supplementary-plane characters (code point > U+FFFF) appear as surrogate
    pairs (0xD800-0xDBFF high, 0xDC00-0xDFFF low); Python's code-point sort
    does not reproduce that ordering, so keys are encoded as UTF-16-BE and
    compared code unit by code unit.
    """
    encoded = key.encode("utf-16-be")
    return tuple(
        int.from_bytes(encoded[index : index + 2], "big")
        for index in range(0, len(encoded), 2)
    )


def _normalize(value: Any, path: str = "$") -> Any:
    """Validate *value* and return its canonicalizable Python form.

    Objects are rebuilt with UTF-16-sorted keys; finite integer-valued floats
    within the exact-integer range become ints; anything that cannot appear in
    finite JSON (non-finite floats, bytes, ...) raises :class:`ValueError`.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float at {path} is not valid JSON")
        if value.is_integer() and abs(value) < _MAX_EXACT_INTEGER:
            # ``int(-0.0)`` is ``0``; JCS serializes negative zero as "0".
            return int(value)
        return value
    if isinstance(value, list):
        return [
            _normalize(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"non-string object key at {path}")
            normalized[key] = _normalize(item, f"{path}.{key}")
        return {
            key: normalized[key]
            for key in sorted(normalized, key=_utf16_key_order)
        }
    raise ValueError(f"unsupported value of type {type(value).__name__} at {path}")


def canonical_json(value: Any) -> bytes:
    """Return the deterministic canonical byte sequence for *value*."""
    normalized = _normalize(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=False,
    ).encode("utf-8")


def canonical_digest(value: Any) -> tuple[str, str]:
    """Return ``(algorithm, hex digest)`` of the canonical form of *value*."""
    canonical = canonical_json(value)
    return DIGEST_ALGORITHM, hashlib.sha256(canonical).hexdigest()
