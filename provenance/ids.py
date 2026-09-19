"""Stable server-side identifier generation.

Content identifiers are derived deterministically from the content identity
(algorithm + digest), so the same bytes always map to the same stable id and
ids are not guessable sequences. The actor id is client-supplied and is not
generated here.
"""

from __future__ import annotations

import hashlib
import json


def content_id(digest_algorithm: str, digest_hex: str) -> str:
    """Return a stable ``cnt_``-prefixed identifier for a content identity."""
    material = f"content:{digest_algorithm}:{digest_hex}".encode("ascii")
    return "cnt_" + hashlib.sha256(material).hexdigest()


def claim_id(
    content_id: str,
    actor_id: str,
    claim_type: str,
    payload_digest_hex: str,
) -> str:
    """Return a stable ``clm_``-prefixed identifier for a claim identity.

    The material is a canonical JSON array so fields containing separators
    cannot collide with different field splits.
    """
    material = json.dumps(
        ["claim", content_id, actor_id, claim_type, payload_digest_hex],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "clm_" + hashlib.sha256(material).hexdigest()


def evidence_bundle_id(
    claim_id: str,
    evidence_type: str,
    digest_algorithm: str,
    digest_hex: str,
) -> str:
    """Return a stable ``evb_``-prefixed identifier for an evidence identity.

    As with claims, the material is a canonical JSON array so fields
    containing separators cannot collide with different field splits.
    """
    material = json.dumps(
        [
            "evidence_bundle",
            claim_id,
            evidence_type,
            digest_algorithm,
            digest_hex,
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "evb_" + hashlib.sha256(material).hexdigest()


def content_relation_id(
    content_id: str,
    parent_content_id: str,
    relation_type: str,
) -> str:
    """Return a stable ``rel_``-prefixed identifier for a content relation.

    The identity is exactly the idempotency key: the two endpoints and the
    relation type. As with claims, the material is a canonical JSON array so
    fields containing separators cannot collide with different field splits.
    """
    material = json.dumps(
        ["content_relation", content_id, parent_content_id, relation_type],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "rel_" + hashlib.sha256(material).hexdigest()


def attestation_id(
    target_type: str,
    target_id: str,
    signer_actor_id: str,
    public_key_hex: str,
    signature_digest_hex: str,
) -> str:
    """Return a stable ``att_``-prefixed identifier for an attestation.

    The identity is exactly the idempotency key: target (type + id), signing
    actor, public key, and the SHA-256 digest of the signature. The raw
    signature never enters the identifier material directly.
    """
    material = json.dumps(
        [
            "attestation",
            target_type,
            target_id,
            signer_actor_id,
            public_key_hex,
            signature_digest_hex,
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "att_" + hashlib.sha256(material).hexdigest()


def attestation_access_grant_id(
    attestation_id: str, grantee_actor_id: str
) -> str:
    """Return a stable ``aag_``-prefixed identifier for an access grant.

    The identity is exactly the idempotency key: the attested proof and the
    grantee actor. The material is a canonical JSON array so fields
    containing separators cannot collide with different field splits.
    """
    material = json.dumps(
        ["attestation_access_grant", attestation_id, grantee_actor_id],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "aag_" + hashlib.sha256(material).hexdigest()


def attestation_revocation_id(
    attestation_id: str,
    revoker_actor_id: str,
    reason: str,
) -> str:
    """Return a stable ``rev_``-prefixed identifier for a revocation.

    The identity is exactly the idempotency key: the revoked attestation,
    the revoking actor, and the (trimmed) reason text. The material is a
    canonical JSON array so fields containing separators cannot collide
    with different field splits.
    """
    material = json.dumps(
        ["attestation_revocation", attestation_id, revoker_actor_id, reason],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "rev_" + hashlib.sha256(material).hexdigest()
