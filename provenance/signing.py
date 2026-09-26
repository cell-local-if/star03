"""Attestation signing contract: the exact bytes a signer commits to.

An attestation proves that an existing signing actor endorses an existing
claim or evidence bundle. The signed message is a UTF-8 JSON array with
compact separators and non-ASCII characters left unescaped, so the bytes are
identical regardless of which language or client produced the signature::

    ["provenance-attestation-v1", target_type, target_id, signer_actor_id]

Only Ed25519 signatures over these exact bytes are accepted; see
:mod:`provenance.ed25519`.

The impact-recon exchange import signs an analogous array binding the
signature version, signing subject, package digest algorithm, and package
digest; see :func:`impact_recon_exchange_message_bytes`.
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


#: Domain-separation prefix and protocol version of the signed exchange
#: array; identical to the signature metadata's ``signature_version``.
IMPACT_RECON_EXCHANGE_MESSAGE_PREFIX = "provenance-impact-recon-exchange-v1"


def impact_recon_exchange_message_bytes(
    subject: str, digest_algorithm: str, package_digest_hex: str
) -> bytes:
    """Return the exact canonical bytes an exchange signature is over.

    The signed message is a UTF-8 compact JSON array binding, in order,
    the signature version, the signing subject, the package digest
    algorithm, and the package digest::

        ["provenance-impact-recon-exchange-v1", subject,
         digest_algorithm, package_digest_hex]
    """
    return json.dumps(
        [
            IMPACT_RECON_EXCHANGE_MESSAGE_PREFIX,
            subject,
            digest_algorithm,
            package_digest_hex,
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
