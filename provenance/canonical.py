"""Deterministic canonical JSON for claim payload and exchange-manifest digests.

A claim commits to its payload through a SHA-256 digest of a canonical
serialization, so the digest is stable across key orderings, insignificant
whitespace, and Unicode escaping choices. The exchange manifest uses the
same canonical form for its bundle snapshot. The raw payload itself is
never persisted or echoed back.
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


def _ordered_root_canonical_bytes(value: Any) -> bytes:
    """Serialize a snapshot whose root keeps its own member order.

    Only nested object members are sorted by Unicode code point; the root
    object keeps its received member order and arrays keep their element
    order. Serialization is otherwise identical to
    :func:`canonical_json_bytes` (compact separators, non-ASCII emitted
    unescaped, UTF-8 encoded), so the bytes are reproducible by an external
    verifier from the snapshot JSON alone.
    """

    def canonicalize(member: Any, *, sort_root: bool) -> Any:
        if isinstance(member, dict):
            items = sorted(member.items()) if sort_root else member.items()
            # Every nested object is sorted; the root alone keeps its order.
            return {key: canonicalize(child, sort_root=True) for key, child in items}
        if isinstance(member, list):
            return [canonicalize(item, sort_root=True) for item in member]
        return member

    normalized = canonicalize(value, sort_root=False)
    return json.dumps(
        normalized,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_exchange_snapshot_bytes(snapshot: Any) -> bytes:
    """Serialize an exchange snapshot under the manifest canonical rules.

    Unlike :func:`canonical_json_bytes`, the snapshot root keeps its member
    names in the snapshot's own order (``content``, ``claim``,
    ``evidence_bundle``, ``attestations``) and arrays keep their element
    order; only nested object members are sorted by Unicode code point.
    Serialization is otherwise identical (compact separators, non-ASCII
    emitted unescaped, UTF-8 encoded), so the bytes are reproducible by an
    external verifier from the exchange JSON alone.
    """
    return _ordered_root_canonical_bytes(snapshot)


def canonical_content_export_snapshot_bytes(snapshot: Any) -> bytes:
    """Serialize a content export snapshot under its verification rules.

    The snapshot root keeps its received member order (``content`` then
    ``claims``) and arrays (the claims and each claim's evidence bundles)
    keep their element order; every nested object member is sorted by
    Unicode code point. Serialization is otherwise identical to the
    exchange snapshot rules (compact separators, non-ASCII emitted
    unescaped, UTF-8 encoded), so the digest is reproducible offline from
    the content export JSON alone.
    """
    return _ordered_root_canonical_bytes(snapshot)


def exchange_manifest_digest_hex(snapshot: Any) -> str:
    """Return the SHA-256 hex digest of the canonical exchange snapshot bytes."""
    return hashlib.sha256(canonical_exchange_snapshot_bytes(snapshot)).hexdigest()


def content_export_digest_hex(snapshot: Any) -> str:
    """Return the SHA-256 hex digest of the canonical content export bytes."""
    return hashlib.sha256(canonical_content_export_snapshot_bytes(snapshot)).hexdigest()


def audit_events_digest_hex(events: Any) -> str:
    """Return the SHA-256 hex digest of the canonical audit-event array bytes.

    The array keeps its element order; each element's object members are
    sorted by Unicode code point, with compact separators, non-ASCII
    emitted unescaped, and UTF-8 encoding -- the same canonical form as
    :func:`canonical_json_bytes`, so the digest is reproducible offline
    from the filtered audit-event listing alone.
    """
    return hashlib.sha256(canonical_json_bytes(events)).hexdigest()
