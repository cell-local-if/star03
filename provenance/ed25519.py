"""Minimal Ed25519 signature verification (RFC 8032), standard library only.

Only :func:`verify` is provided: attestations are self-contained proofs, so
the service never holds or generates private keys. Point arithmetic uses
extended twisted-Edwards coordinates so a verification needs no per-step
modular inversion; there is no third-party dependency, keeping verification
deterministic and fully offline.
"""

from __future__ import annotations

import hashlib

#: Valid Ed25519 public key length.
PUBLIC_KEY_LENGTH = 32
#: Valid Ed25519 signature length.
SIGNATURE_LENGTH = 64

#: Prime of the base field.
Q = 2**255 - 19
#: Order of the prime-order subgroup (and of the standard base point).
L = 2**252 + 27742317777372353535851937790883648493

#: Curve constant ``d`` and the quartic root of -1 used for x recovery.
D = -121665 * pow(121666, Q - 2, Q) % Q
_D2 = D * 2 % Q
_I = pow(2, (Q - 1) // 4, Q)


class _InvalidPoint(ValueError):
    """Raised when 32 bytes do not encode a valid curve point."""


def _xrecover(y: int) -> int:
    """Recover the even-parity x coordinate from y (RFC 8032 §5.1.3).

    ``x`` satisfies ``x^2 = (y^2 - 1) / (d*y^2 + 1)``.
    """
    u = y * y - 1
    v = D * y * y + 1
    xx = u * pow(v, Q - 2, Q) % Q
    x = pow(xx, (Q + 3) // 8, Q)
    if (x * x - xx) % Q != 0:
        x = x * _I % Q
    if (x * x - xx) % Q != 0:
        raise _InvalidPoint("no square root for x")
    if x & 1:
        x = Q - x
    return x


def _decompress(encoded: bytes) -> tuple[int, int, int, int]:
    """Decode 32 bytes to extended coordinates ``(X, Y, Z, T)``."""
    if len(encoded) != PUBLIC_KEY_LENGTH:
        raise _InvalidPoint("bad point length")
    y = int.from_bytes(encoded, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    if y >= Q:
        raise _InvalidPoint("y out of field")
    x = _xrecover(y)
    if (x & 1) != sign:
        x = Q - x
    if (-x * x + y * y - 1 - D * x * x * y * y) % Q != 0:
        raise _InvalidPoint("point not on curve")
    return x, y, 1, x * y % Q


def _add(
    p: tuple[int, int, int, int], q: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Extended-coordinate twisted-Edwards addition (add ``d`` = -1)."""
    x1, y1, z1, t1 = p
    x2, y2, z2, t2 = q
    a = (y1 - x1) * (y2 - x2) % Q
    b = (y1 + x1) * (y2 + x2) % Q
    c = t1 * _D2 * t2 % Q
    dd = z1 * 2 * z2 % Q
    e = b - a
    f = dd - c
    g = dd + c
    h = b + a
    return e * f % Q, g * h % Q, f * g % Q, e * h % Q


def _double(
    p: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Extended-coordinate doubling."""
    x1, y1, z1, _ = p
    a = x1 * x1 % Q
    b = y1 * y1 % Q
    c = 2 * z1 * z1 % Q
    h = a + b
    e = h - (x1 + y1) * (x1 + y1) % Q
    g = a - b
    f = c + g
    return e * f % Q, g * h % Q, f * g % Q, e * h % Q


def _scalarmult(
    point: tuple[int, int, int, int], scalar: int
) -> tuple[int, int, int, int]:
    """Scalar multiplication via double-and-add."""
    result = (0, 1, 1, 0)
    addend = point
    while scalar:
        if scalar & 1:
            result = _add(result, addend)
        addend = _double(addend)
        scalar >>= 1
    return result


def _is_identity(p: tuple[int, int, int, int]) -> bool:
    x, y, z, _ = p
    return x % Q == 0 and (y - z) % Q == 0


def _points_equal(
    p: tuple[int, int, int, int], q: tuple[int, int, int, int]
) -> bool:
    # Equality in extended coordinates: X1*Z2 == X2*Z1 and Y1*Z2 == Y2*Z1.
    x1, y1, z1, _ = p
    x2, y2, z2, _ = q
    return (
        (x1 * z2 - x2 * z1) % Q == 0
        and (y1 * z2 - y2 * z1) % Q == 0
    )


#: Standard Ed25519 base point (has prime order L).
B = _decompress(
    bytes.fromhex(
        "5866666666666666666666666666666666666666666666666666666666666666"
    )
)


def _encode_point(p: tuple[int, int, int, int]) -> bytes:
    """Encode an extended point in the compressed 32-byte form.

    Used by tests to construct offline signatures; production verification
    never encodes points.
    """
    x, y, z, _ = p
    zi = pow(z, Q - 2, Q)
    x = x * zi % Q
    y = y * zi % Q
    encoded = y | ((x & 1) << 255)
    return encoded.to_bytes(PUBLIC_KEY_LENGTH, "little")


def verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
    """Verify an Ed25519 signature, returning ``True`` only when valid.

    :param public_key: exactly 32 raw bytes (caller must have length-checked).
    :param message: the exact signed bytes.
    :param signature: exactly 64 raw bytes (caller must have length-checked).

    Decoding follows RFC 8032 strictly: encodings off the curve, non-canonical
    scalars (``S >= L``), and low-order public keys are rejected.
    """
    if len(public_key) != PUBLIC_KEY_LENGTH or len(signature) != SIGNATURE_LENGTH:
        return False
    try:
        a = _decompress(public_key)
        r_point = _decompress(signature[:PUBLIC_KEY_LENGTH])
    except _InvalidPoint:
        return False

    s = int.from_bytes(signature[PUBLIC_KEY_LENGTH:], "little")
    if s >= L:
        return False

    # A valid key lies in the prime-order subgroup: [L]A must be the
    # identity. This rejects all small-subgroup points except the identity,
    # which is rejected explicitly (it would make signatures forgeable).
    if _is_identity(a) or not _is_identity(_scalarmult(a, L)):
        return False

    k = int.from_bytes(
        hashlib.sha512(
            signature[:PUBLIC_KEY_LENGTH] + public_key + message
        ).digest(),
        "little",
    )
    # [S]B == R + [k]A
    return _points_equal(
        _scalarmult(B, s),
        _add(r_point, _scalarmult(a, k)),
    )
