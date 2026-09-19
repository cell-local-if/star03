"""Pure-Python Ed25519 signatures (RFC 8032), no external dependencies.

The service only needs verification, which is implemented here over the
edwards25519 curve exactly as specified in RFC 8032. Deterministic signing
and public-key derivation from a 32-byte seed are provided for offline
tests and local tooling; the API never sees a private key.

This is a compact reference implementation favoring clarity and
determinism over side-channel hardening; verification of untrusted
signatures is not a secret-bearing operation.
"""

from __future__ import annotations

import hashlib

# Curve and group constants for edwards25519 (RFC 8032, section 5.1).
_Q = 2**255 - 19  # field prime
_L = 2**252 + 27742317777372353535851937790883648493  # group order
_D = (-121665 * pow(121666, _Q - 2, _Q)) % _Q
_I = pow(2, (_Q - 1) // 4, _Q)  # square root of -1 mod q

PUBLIC_KEY_LENGTH = 32
SIGNATURE_LENGTH = 64
SEED_LENGTH = 32


def _sha512(data: bytes) -> bytes:
    return hashlib.sha512(data).digest()


def _xrecover(y: int) -> int:
    xx = (y * y - 1) * pow(_D * y * y + 1, _Q - 2, _Q) % _Q
    x = pow(xx, (_Q + 3) // 8, _Q)
    if (x * x - xx) % _Q != 0:
        x = (x * _I) % _Q
    if x % 2 != 0:
        x = _Q - x
    return x


# Base point B: y = 4/5 with even x.
_B = (_xrecover(4 * pow(5, _Q - 2, _Q) % _Q), 4 * pow(5, _Q - 2, _Q) % _Q)
_IDENTITY = (0, 1)


def _isoncurve(point: tuple[int, int]) -> bool:
    x, y = point
    return (-x * x + y * y - 1 - _D * x * x * y * y) % _Q == 0


def _edwards_add(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int]:
    x1, y1 = p
    x2, y2 = q
    common = _D * x1 * x2 * y1 * y2 % _Q
    x3 = (x1 * y2 + x2 * y1) * pow(1 + common, _Q - 2, _Q) % _Q
    y3 = (y1 * y2 + x1 * x2) * pow(1 - common, _Q - 2, _Q) % _Q
    return (x3, y3)


def _scalarmult(point: tuple[int, int], scalar: int) -> tuple[int, int]:
    result = _IDENTITY
    addend = point
    while scalar > 0:
        if scalar & 1:
            result = _edwards_add(result, addend)
        addend = _edwards_add(addend, addend)
        scalar >>= 1
    return result


def _encode_point(point: tuple[int, int]) -> bytes:
    x, y = point
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def _decode_point(data: bytes) -> tuple[int, int]:
    if len(data) != 32:
        raise ValueError("encoded point must be 32 bytes")
    y = int.from_bytes(data, "little") & ((1 << 255) - 1)
    if y >= _Q:
        raise ValueError("point y-coordinate out of range")
    x = _xrecover(y)
    if x & 1 != data[31] >> 7:
        x = _Q - x
    point = (x, y)
    if not _isoncurve(point):
        raise ValueError("point is not on the curve")
    return point


def verify(public_key: bytes, signature: bytes, message: bytes) -> bool:
    """Return True iff ``signature`` is a valid Ed25519 signature.

    Pure verification per RFC 8032 section 5.1.7: ``S*B == R + H(R,A,M)*A``
    with canonicality checks on the encoded points and the scalar. Any
    malformed input (wrong lengths, off-curve points, ``S >= L``) is simply
    invalid, never an exception.
    """
    if (
        len(public_key) != PUBLIC_KEY_LENGTH
        or len(signature) != SIGNATURE_LENGTH
    ):
        return False
    try:
        a = _decode_point(public_key)
        r = _decode_point(signature[:32])
    except ValueError:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _L:
        return False
    h = int.from_bytes(_sha512(signature[:32] + public_key + message), "little") % _L
    return _scalarmult(_B, s) == _edwards_add(r, _scalarmult(a, h))


def _secret_expand(seed: bytes) -> tuple[int, bytes]:
    digest = _sha512(seed)
    a = int.from_bytes(digest[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, digest[32:]


def public_key_from_seed(seed: bytes) -> bytes:
    """Derive the 32-byte Ed25519 public key for a 32-byte seed."""
    if len(seed) != SEED_LENGTH:
        raise ValueError("seed must be 32 bytes")
    a, _ = _secret_expand(seed)
    return _encode_point(_scalarmult(_B, a))


def sign(seed: bytes, message: bytes) -> bytes:
    """Deterministically sign ``message`` with a 32-byte seed (RFC 8032)."""
    a, prefix = _secret_expand(seed)
    public_key = _encode_point(_scalarmult(_B, a))
    r = int.from_bytes(_sha512(prefix + message), "little") % _L
    encoded_r = _encode_point(_scalarmult(_B, r))
    h = int.from_bytes(_sha512(encoded_r + public_key + message), "little") % _L
    s = (r + h * a) % _L
    return encoded_r + s.to_bytes(32, "little")
