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
