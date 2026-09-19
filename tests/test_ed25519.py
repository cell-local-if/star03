"""Tests for the pure-Python Ed25519 implementation (RFC 8032).

Uses the published RFC 8032 test vectors plus deterministic negative cases.
All tests are offline and deterministic.
"""

from __future__ import annotations

from provenance import ed25519

# RFC 8032, section 7.1, TEST 1 (empty message).
TEST1_PUBLIC = bytes.fromhex(
    "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
)
TEST1_SIGNATURE = bytes.fromhex(
    "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
    "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
)

# RFC 8032, section 7.1, TEST 2 (one-byte message 0x72).
TEST2_SECRET = bytes.fromhex(
    "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb"
)
TEST2_PUBLIC = bytes.fromhex(
    "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c"
)
TEST2_MESSAGE = b"\x72"
TEST2_SIGNATURE = bytes.fromhex(
    "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da"
    "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"
)

# RFC 8032, section 7.1, TEST 3 (two-byte message 0xaf82).
TEST3_PUBLIC = bytes.fromhex(
    "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025"
)
TEST3_MESSAGE = bytes.fromhex("af82")
TEST3_SIGNATURE = bytes.fromhex(
    "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac"
    "18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"
)


def test_rfc8032_test1_signature_verifies():
    assert ed25519.verify(TEST1_PUBLIC, TEST1_SIGNATURE, b"")


def test_rfc8032_test2_public_key_and_signature():
    assert ed25519.public_key_from_seed(TEST2_SECRET) == TEST2_PUBLIC
    assert ed25519.sign(TEST2_SECRET, TEST2_MESSAGE) == TEST2_SIGNATURE
    assert ed25519.verify(TEST2_PUBLIC, TEST2_SIGNATURE, TEST2_MESSAGE)


def test_rfc8032_test3_signature_verifies():
    assert ed25519.verify(TEST3_PUBLIC, TEST3_SIGNATURE, TEST3_MESSAGE)


def test_signing_is_deterministic():
    seed = b"\x2a" * 32
    message = "deterministic-证据".encode("utf-8")
    assert ed25519.sign(seed, message) == ed25519.sign(seed, message)


def test_tampered_message_or_signature_fails():
    seed = b"\x07" * 32
    public_key = ed25519.public_key_from_seed(seed)
    signature = ed25519.sign(seed, b"message")
    assert not ed25519.verify(public_key, signature, b"message!")
    tampered = bytearray(signature)
    tampered[10] ^= 0x01
    assert not ed25519.verify(public_key, bytes(tampered), b"message")
    other_key = ed25519.public_key_from_seed(b"\x08" * 32)
    assert not ed25519.verify(other_key, signature, b"message")


def test_malformed_inputs_are_invalid_not_exceptions():
    public_key = ed25519.public_key_from_seed(b"\x01" * 32)
    signature = ed25519.sign(b"\x01" * 32, b"m")
    # Wrong lengths.
    assert not ed25519.verify(b"", signature, b"m")
    assert not ed25519.verify(public_key, b"", b"m")
    assert not ed25519.verify(public_key[:-1], signature, b"m")
    assert not ed25519.verify(public_key, signature[:-1], b"m")
    # Off-curve / non-canonical encodings.
    assert not ed25519.verify(b"\xff" * 32, signature, b"m")
    assert not ed25519.verify(public_key, b"\x00" * 64, b"m")
    # S >= L (group order) is rejected.
    group_order = 2**252 + 27742317777372353535851937790883648493
    bad_s = signature[:32] + group_order.to_bytes(32, "little")
    assert not ed25519.verify(public_key, bad_s, b"m")


def test_seed_must_be_32_bytes():
    import pytest

    with pytest.raises(ValueError):
        ed25519.public_key_from_seed(b"\x00" * 31)
