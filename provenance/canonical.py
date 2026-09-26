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
    """Serialize with the root member order kept and nested keys sorted.

    The root object keeps its member names in the received order and arrays
    keep their element order; every nested object's members are sorted by
    Unicode code point. Serialization is otherwise identical to
    :func:`canonical_json_bytes` (compact separators, non-ASCII emitted
    unescaped, UTF-8 encoded), so the bytes are reproducible by an external
    verifier from the received JSON alone.
    """

    def canonicalize(value: Any, *, sort_root: bool) -> Any:
        if isinstance(value, dict):
            items = sorted(value.items()) if sort_root else value.items()
            # Every nested object is sorted; the root alone keeps its order.
            return {key: canonicalize(member, sort_root=True) for key, member in items}
        if isinstance(value, list):
            return [canonicalize(item, sort_root=True) for item in value]
        return value

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


def exchange_manifest_digest_hex(snapshot: Any) -> str:
    """Return the SHA-256 hex digest of the canonical exchange snapshot bytes."""
    return hashlib.sha256(canonical_exchange_snapshot_bytes(snapshot)).hexdigest()


def canonical_content_export_snapshot_bytes(snapshot: Any) -> bytes:
    """Serialize a content export snapshot under its offline digest rules.

    Exactly the same canonical form as
    :func:`canonical_exchange_snapshot_bytes`: the snapshot root keeps its
    member order (``content`` then ``claims``), arrays keep their element
    order, and every nested object's members sort by Unicode code point,
    with compact separators, unescaped non-ASCII, and UTF-8 encoding -- so
    an offline verifier reproduces the bytes from the export JSON alone,
    including an empty ``claims`` array.
    """
    return _ordered_root_canonical_bytes(snapshot)


def content_export_snapshot_digest_hex(snapshot: Any) -> str:
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


def revocation_impacts_digest_hex(impacts: Any) -> str:
    """Return the SHA-256 hex digest of the canonical impact-array bytes.

    The impacts array keeps its element order exactly as exported (the
    revocations' stable creation order); each impact object's members are
    sorted by Unicode code point, with compact separators, non-ASCII
    emitted unescaped, and UTF-8 encoding -- the same canonical form as
    :func:`audit_events_digest_hex`, so the digest is reproducible offline
    from the exported ``impacts`` member alone, including an empty array
    (the digest of ``[]``).
    """
    return hashlib.sha256(canonical_json_bytes(impacts)).hexdigest()


def impact_recon_entries_digest_hex(entries: Any) -> str:
    """Return the SHA-256 hex digest of the canonical recon-entries bytes.

    The entries array keeps its element order exactly as exported (the
    receipts' stable creation order); each entry object's members are
    sorted recursively by Unicode code point (including the nested
    ``local_checkpoint``), with compact separators, non-ASCII emitted
    unescaped, and UTF-8 encoding -- the same canonical form as
    :func:`revocation_impacts_digest_hex`, so the digest is reproducible
    offline from the exported ``entries`` member alone, including an empty
    array (the digest of ``[]``).
    """
    return hashlib.sha256(canonical_json_bytes(entries)).hexdigest()


def impact_recon_audit_entries_digest_hex(entries: Any) -> str:
    """Return the SHA-256 hex digest of the canonical audit-entries bytes.

    The entries array keeps its element order exactly as exported (the
    signed exchange-import receipts' stable creation order); each entry
    object's members are sorted by Unicode code point, with compact
    separators, non-ASCII emitted unescaped, and UTF-8 encoding -- the
    same canonical form as :func:`impact_recon_entries_digest_hex`, so
    the digest is reproducible offline from the exported ``entries``
    member alone, including an empty array (the digest of ``[]``).
    """
    return hashlib.sha256(canonical_json_bytes(entries)).hexdigest()


def canonical_impact_recon_package_bytes(package: Any) -> bytes:
    """Serialize an impact-recon package under the package digest rules.

    Exactly the same canonical form as the other package/snapshot rules:
    the package root keeps its member order (``checkpoint`` then
    ``entries``) and arrays keep their element order, while every nested
    object's members are sorted by Unicode code point, with compact
    separators, unescaped non-ASCII, and UTF-8 encoding -- so the digest
    is reproducible by an external verifier from the received package
    JSON alone. (The two root members are already in code-point order, so
    this coincides with fully sorted canonicalization for this shape.)
    """
    return _ordered_root_canonical_bytes(package)


def impact_recon_package_digest_hex(package: Any) -> str:
    """Return the SHA-256 hex digest of the canonical recon-package bytes."""
    return hashlib.sha256(canonical_impact_recon_package_bytes(package)).hexdigest()
