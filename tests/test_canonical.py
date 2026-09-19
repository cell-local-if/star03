"""Unit tests for deterministic canonical JSON and payload digests."""

from __future__ import annotations

import hashlib

import pytest

from provenance.canonical import canonical_digest, canonical_json


def test_object_keys_are_sorted_recursively():
    assert canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'
    assert canonical_json({"outer": {"z": 0, "a": {"y": 1, "x": 2}}}) == (
        b'{"outer":{"a":{"x":2,"y":1},"z":0}}'
    )


def test_arrays_keep_their_order():
    assert canonical_json([3, 1, 2]) == b"[3,1,2]"
    assert canonical_json([{"b": 1}, {"a": 2}]) == b'[{"b":1},{"a":2}]'


def test_no_insignificant_whitespace_and_utf8_output():
    assert canonical_json({"a": "中"}) == '{"a":"中"}'.encode("utf-8")
    # Non-ASCII is never \u-escaped.
    assert b"\\u" not in canonical_json({"k": "😀"})


def test_key_reordering_does_not_change_digest():
    first = canonical_digest({"a": 1, "b": [1, 2, {"c": True}]})
    second = canonical_digest({"b": [1, 2, {"c": True}], "a": 1})
    assert first == second
    assert first[0] == "sha256"
    assert len(first[1]) == 64 and all(c in "0123456789abcdef" for c in first[1])


def test_digest_matches_independent_sha256_of_canonical_bytes():
    payload = {"b": 2, "a": "x"}
    algorithm, digest = canonical_digest(payload)
    assert algorithm == "sha256"
    assert digest == hashlib.sha256(b'{"a":"x","b":2}').hexdigest()


def test_array_order_is_significant():
    assert canonical_digest([1, 2]) != canonical_digest([2, 1])


def test_integer_valued_float_canonicalizes_like_integer():
    assert canonical_json({"v": 1.0}) == b'{"v":1}'
    assert canonical_digest({"v": 1.0}) == canonical_digest({"v": 1})


def test_negative_zero_canonicalizes_to_zero():
    assert canonical_json({"v": -0.0}) == b'{"v":0}'


def test_fractional_float_keeps_round_trip_form():
    assert canonical_json({"v": 0.5}) == b'{"v":0.5}'
    assert canonical_digest({"v": 1.5}) != canonical_digest({"v": 1})


def test_bool_is_distinct_from_integer():
    assert canonical_digest({"v": True}) != canonical_digest({"v": 1})


def test_empty_object_and_nested_nulls():
    assert canonical_json({}) == b"{}"
    assert canonical_json({"a": None}) == b'{"a":null}'


def test_non_finite_floats_are_rejected():
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            canonical_json({"v": value})
        with pytest.raises(ValueError):
            canonical_json([value])


def test_non_json_types_are_rejected():
    with pytest.raises(ValueError):
        canonical_json({"v": b"raw"})
    with pytest.raises(ValueError):
        canonical_json({1: "x"})


def test_unicode_key_order_is_deterministic():
    # RFC 8785 orders by UTF-16 code units; the result must at minimum be
    # stable and byte-identical across repeated calls.
    payload = {"😀": 1, "中": 2, "a": 3}
    assert canonical_json(payload) == canonical_json(payload)
    # BMP "中" (U+4E2D -> 0x4E2D) sorts before supplementary "😀"
    # (0xD83D, 0xDE00 high surrogate 0xD83D > 0x4E2D).
    assert canonical_json(payload) == b'{"a":3,"\xe4\xb8\xad":2,"\xf0\x9f\x98\x80":1}'
