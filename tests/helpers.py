"""Deterministic payload helpers for tests."""

from __future__ import annotations

import hashlib

from provenance.ed25519 import B, L, _encode_point, _scalarmult  # type: ignore

# sha256 of a fixed, tiny input: stable and offline.
DIGEST_A = hashlib.sha256(b"content-a").hexdigest()
DIGEST_B = hashlib.sha256(b"content-b").hexdigest()
DIGEST_C = hashlib.sha256(b"content-c").hexdigest()


def actor_payload(actor_id="org-1", name="Example Org", type="organization"):
    return {"id": actor_id, "name": name, "type": type}


def create_actor(client, **overrides):
    payload = actor_payload(**overrides)
    response = client.post("/v1/actors", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def content_payload(
    actor_id="org-1",
    digest=DIGEST_A,
    media_type="image/png",
    title=None,
    algorithm="sha256",
):
    return {
        "digest_algorithm": algorithm,
        "digest_hex": digest,
        "media_type": media_type,
        "title": title,
        "actor_id": actor_id,
    }


# --- Offline Ed25519 signing (tests only) -----------------------------------
#
# The production package verifies but never signs. Tests construct standard
# RFC 8032 signatures using the curve primitives in ``provenance.ed25519``,
# keeping everything deterministic and offline (no third-party crypto).


def ed25519_public_key(seed: bytes) -> bytes:
    """Derive the 32-byte public key for a 32-byte deterministic seed."""
    scalar = _ed25519_secret_scalar(seed)
    return _encode_point(_scalarmult(B, scalar))


def ed25519_sign(seed: bytes, message: bytes) -> bytes:
    """Return the 64-byte Ed25519 signature of ``message`` under ``seed``."""
    scalar = _ed25519_secret_scalar(seed)
    prefix = hashlib.sha512(seed).digest()[32:]
    public = _encode_point(_scalarmult(B, scalar))
    r = int.from_bytes(hashlib.sha512(prefix + message).digest(), "little") % L
    r_bytes = _encode_point(_scalarmult(B, r))
    k = int.from_bytes(
        hashlib.sha512(r_bytes + public + message).digest(), "little"
    ) % L
    s = (r + k * scalar) % L
    return r_bytes + s.to_bytes(32, "little")


def _ed25519_secret_scalar(seed: bytes) -> int:
    a = int.from_bytes(hashlib.sha512(seed).digest()[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a


#: Fixed test seeds so signatures and signature digests are reproducible.
SEED_A = b"test-ed25519-seed-a-00000000000000"[:32]
SEED_B = b"test-ed25519-seed-b-00000000000000"[:32]

