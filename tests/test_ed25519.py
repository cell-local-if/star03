"""Known-answer and boundary tests for the stdlib-only Ed25519 verifier.

The verifier is the only cryptographic trust boundary in the attestation
flow, so it is checked against published RFC 8032 vectors in addition to the
offline round-trip signatures used by the HTTP tests.
"""

from __future__ import annotations

import hashlib

from provenance.ed25519 import PUBLIC_KEY_LENGTH, SIGNATURE_LENGTH, verify

# RFC 8032 Section 7.1, test vectors 1-3 (public key, message, signature).
_RFC_VECTORS = [
    (
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
        "",
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
    ),
    (
        "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
        "72",
        "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da"
        "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
    ),
    (
        "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
        "af82",
        "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac"
        "18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a",
    ),
]


def test_rfc8032_known_answer_vectors_verify():
    for pk_hex, msg_hex, sig_hex in _RFC_VECTORS:
        assert verify(
            bytes.fromhex(pk_hex), bytes.fromhex(msg_hex), bytes.fromhex(sig_hex)
        )


def test_modified_message_or_signature_or_key_fails():
    pk_hex, msg_hex, sig_hex = _RFC_VECTORS[1]
    pk, msg, sig = (
        bytes.fromhex(pk_hex),
        bytes.fromhex(msg_hex),
        bytes.fromhex(sig_hex),
    )
    flipped_sig = bytearray(sig)
    flipped_sig[63] ^= 1
    flipped_pk = bytearray(pk)
    flipped_pk[0] ^= 1
    assert not verify(pk, msg + b"\x00", sig)
    assert not verify(pk, msg, bytes(flipped_sig))
    assert not verify(bytes(flipped_pk), msg, sig)


def test_scalar_out_of_range_is_rejected():
    pk_hex, _, sig_hex = _RFC_VECTORS[0]
    big_s = bytes.fromhex(sig_hex[:32]) + (2**255 - 1).to_bytes(32, "little")
    assert not verify(bytes.fromhex(pk_hex), b"", big_s)


def test_small_subgroup_and_identity_public_keys_are_rejected():
    # All-zero identity point.
    assert not verify(b"\x00" * PUBLIC_KEY_LENGTH, b"", b"\x00" * SIGNATURE_LENGTH)
    # A documented low-order point (non-identity) on Ed25519.
    low_order = bytes.fromhex(
        "c7176a703d4dd84fba3c0b760d10670f2a2053fa2dc4ed942ee81fe1daeaf2e2"
    )
    assert not verify(low_order, b"x", b"\x00" * SIGNATURE_LENGTH)


def test_offline_round_trip_signatures_verify_and_tamper_fails():
    # The test-only signer must interoperate with the verifier.
    from tests.helpers import ed25519_public_key, ed25519_sign

    for seed, message in [
        (b"0" * 32, b""),
        (b"1" * 32, b"compact-json-with-\xe8\xaf\x81\xe6\x8d\xae"),
        (b"2" * 32, hashlib.sha512(b"bulk").digest()),
    ]:
        public = ed25519_public_key(seed)
        signature = ed25519_sign(seed, message)
        assert len(public) == PUBLIC_KEY_LENGTH
        assert len(signature) == SIGNATURE_LENGTH
        assert verify(public, message, signature)
        assert not verify(public, message + b"!", signature)
