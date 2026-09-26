"""Attestation signing contract: the exact bytes a signer commits to.

An attestation proves that an existing signing actor endorses an existing
claim or evidence bundle. The signed message is a UTF-8 JSON array with
compact separators and non-ASCII characters left unescaped, so the bytes are
identical regardless of which language or client produced the signature::

    ["provenance-attestation-v1", target_type, target_id, signer_actor_id]

Only Ed25519 signatures over these exact bytes are accepted; see
:mod:`provenance.ed25519`.
"""

from __future__ import annotations

import json

#: Domain-separation prefix and protocol version of the signed array.
ATTESTATION_MESSAGE_PREFIX = "provenance-attestation-v1"

#: Attestable target types, matched verbatim in the signed message.
TARGET_CLAIM = "claim"
TARGET_EVIDENCE_BUNDLE = "evidence_bundle"
ATTESTATION_TARGET_TYPES = frozenset({TARGET_CLAIM, TARGET_EVIDENCE_BUNDLE})


def attestation_message_bytes(
    target_type: str, target_id: str, signer_actor_id: str
) -> bytes:
    """Return the exact canonical bytes the signer's signature is over."""
    return json.dumps(
        [ATTESTATION_MESSAGE_PREFIX, target_type, target_id, signer_actor_id],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


#: Domain-separation prefix and protocol version of a signed audit
#: checkpoint exchange package.
AUDIT_EXCHANGE_MESSAGE_PREFIX = "provenance-audit-exchange-v1"


def audit_exchange_message_bytes(
    signer_subject: str,
    package_digest_algorithm: str,
    package_digest_hex: str,
) -> bytes:
    """Return the exact canonical bytes an audit exchange signature is over.

    The signature binds, in order, the fixed exchange version, the signing
    subject, the package digest algorithm, and the whole-package digest
    value:

        ["provenance-audit-exchange-v1", subject, "sha256", digest]

    The bytes are a UTF-8 compact JSON array (compact separators, non-ASCII
    emitted unescaped), identical regardless of which client produced the
    signature.
    """
    return json.dumps(
        [
            AUDIT_EXCHANGE_MESSAGE_PREFIX,
            signer_subject,
            package_digest_algorithm,
            package_digest_hex,
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


#: Domain-separation prefix and protocol version of a signed impact-recon
#: exchange package.
IMPACT_RECON_EXCHANGE_MESSAGE_PREFIX = (
    "provenance-impact-recon-exchange-v1"
)


def impact_recon_exchange_message_bytes(
    signer_subject: str,
    package_digest_algorithm: str,
    package_digest_hex: str,
) -> bytes:
    """Return the exact canonical bytes an exchange signature is over.

    The signature binds, in order, the fixed exchange version, the signing
    subject, the package digest algorithm, and the package digest value:

        ["provenance-impact-recon-exchange-v1", subject, "sha256", digest]

    The bytes are a UTF-8 compact JSON array (compact separators, non-ASCII
    emitted unescaped), identical regardless of which client produced the
    signature.
    """
    return json.dumps(
        [
            IMPACT_RECON_EXCHANGE_MESSAGE_PREFIX,
            signer_subject,
            package_digest_algorithm,
            package_digest_hex,
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
