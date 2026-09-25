"""Tests for the stateless revocation-impact checkpoint verification endpoint.

Covers POST /v1/impact-verifications: the request body is exactly
{"checkpoint", "impacts"}; the checkpoint is exactly
{"checkpoint_version", "digest_algorithm", "impact_count",
"impacts_digest_hex"} with the version pinned to
"provenance-revocation-impact-checkpoint-v1", the algorithm to "sha256",
the count a non-negative integer, and the digest 64 lowercase hex
characters. The impacts array must contain exactly ``impact_count``
elements, each exactly the revocation-impact export public view with the
declared types, a strict RFC 3339 UTC ``created_at``, and the internal
association ``before - after == delta`` (zero or one). The digest is the
SHA-256 of the canonical impact array (array order preserved, object
keys sorted by Unicode code point, compact separators, non-ASCII
unescaped, UTF-8 encoded) computed over the impacts exactly as received.
Any malformed JSON or structural, field, type, association, count, or
digest-input violation is a 422 validation_error; a structurally valid
request whose digest differs is 200 {"valid": false,
"computed_digest_hex": ...}; a match is exactly {"valid": true}. The
route is fully stateless: it needs no persisted records, resolves no id
against local state, and creates, modifies, and queries nothing; unknown
resources, repeated requests, and differing local state never move the
verdict. All fixtures are deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json
import re

from sqlalchemy import select

from provenance.models import AuditEvent
from tests.test_revocation_impacts import _world

URL = "/v1/impact-verifications"
CHECKPOINT_VERSION = "provenance-revocation-impact-checkpoint-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

ITEM_FIELDS = (
    "id",
    "attestation_id",
    "revoker_actor_id",
    "reason",
    "created_at",
    "content_id",
    "target_type",
    "signer_actor_id",
    "qualified_signer_count_after",
    "coverage_status_after",
    "qualified_signer_count_before",
    "coverage_status_before",
    "qualified_signer_count_delta",
)


def _digest_of(impacts: list) -> str:
    """Independently canonicalize the impact array and digest it."""
    canonical = json.dumps(
        impacts,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _item(**overrides) -> dict:
    item = {
        "id": "rev_00000000000000000000000000000001",
        "attestation_id": "att_00000000000000000000000000000001",
        "revoker_actor_id": "org-1",
        "reason": "no longer relied upon",
        "created_at": "2026-01-02T03:04:05Z",
        "content_id": "cnt_00000000000000000000000000000001",
        "target_type": "claim",
        "signer_actor_id": "org-1",
        "qualified_signer_count_after": 0,
        "coverage_status_after": "partial",
        "qualified_signer_count_before": 1,
        "coverage_status_before": "covered",
        "qualified_signer_count_delta": 1,
    }
    item.update(overrides)
    return item


def _request(impacts: list | None = None, *, digest: str | None = None) -> dict:
    impacts = [_item()] if impacts is None else impacts
    return {
        "checkpoint": {
            "checkpoint_version": CHECKPOINT_VERSION,
            "digest_algorithm": "sha256",
            "impact_count": len(impacts),
            "impacts_digest_hex": (
                _digest_of(impacts) if digest is None else digest
            ),
        },
        "impacts": impacts,
    }


def _served_package(client) -> dict:
    return client.get("/v1/revocation-impact-package").json()


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- Verdicts -----------------------------------------------------------------


def test_verification_of_served_package_is_valid(client):
    _world(client)
    resp = client.post(URL, json=_served_package(client))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_valid_success_body_has_no_other_fields(client):
    _world(client)
    resp = client.post(URL, json=_served_package(client))
    assert resp.status_code == 200
    assert set(resp.json()) == {"valid"}
    assert resp.json()["valid"] is True


def test_digest_mismatch_is_200_invalid_with_computed_digest(client):
    _world(client)
    package = _served_package(client)
    package["impacts"][0]["reason"] = "tampered"
    resp = client.post(URL, json=package)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "valid": False,
        "computed_digest_hex": _digest_of(package["impacts"]),
    }
    assert _HEX64.fullmatch(resp.json()["computed_digest_hex"])


def test_reordered_array_changes_the_digest(client):
    _world(client)
    package = _served_package(client)
    impacts = package["impacts"]
    swapped = [impacts[1], impacts[0], *impacts[2:]]
    resp = client.post(
        URL,
        json={
            "checkpoint": {
                "checkpoint_version": CHECKPOINT_VERSION,
                "digest_algorithm": "sha256",
                "impact_count": len(swapped),
                "impacts_digest_hex": package["checkpoint"][
                    "impacts_digest_hex"
                ],
            },
            "impacts": swapped,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["valid"] is False
    assert resp.json()["computed_digest_hex"] == _digest_of(swapped)


def test_empty_array_snapshot_is_valid(client):
    empty = {
        "checkpoint": {
            "checkpoint_version": CHECKPOINT_VERSION,
            "digest_algorithm": "sha256",
            "impact_count": 0,
            "impacts_digest_hex": hashlib.sha256(b"[]").hexdigest(),
        },
        "impacts": [],
    }
    resp = client.post(URL, json=empty)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_non_ascii_canonical_form_verifies(client):
    impacts = [
        _item(
            id="rev-émoji-✓",
            reason="Képek ⛄",
            created_at="2026-01-02T03:04:05.500Z",
        )
    ]
    resp = client.post(URL, json=_request(impacts))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}
    assert "\\u" not in resp.text


def test_verdict_response_is_compact_utf8_json_with_one_newline(client):
    valid = client.post(URL, json=_request())
    assert valid.content == b'{"valid":true}\n'

    mismatch = _request(digest="0" * 64)
    invalid = client.post(URL, json=mismatch)
    raw = invalid.content
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw and b": " not in raw
    assert raw.startswith(b'{"valid":false,"computed_digest_hex":"')


# --- Statelessness ------------------------------------------------------------


def test_route_works_with_no_persisted_state(client):
    resp = client.post(URL, json=_request())
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_unknown_resource_ids_still_verify(client):
    _world(client)
    package = _served_package(client)
    package["impacts"][0].update(
        {
            "id": "rev_does_not_exist",
            "attestation_id": "att_does_not_exist",
            "revoker_actor_id": "actor_does_not_exist",
            "content_id": "cnt_does_not_exist",
            "signer_actor_id": "signer_does_not_exist",
        }
    )
    package["checkpoint"]["impacts_digest_hex"] = _digest_of(
        package["impacts"]
    )
    resp = client.post(URL, json=package)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_repeated_requests_have_identical_verdicts(client):
    _world(client)
    package = _served_package(client)
    first = client.post(URL, json=package)
    second = client.post(URL, json=package)
    assert first.content == second.content


def test_differing_local_state_does_not_move_the_verdict(client, db_session):
    _world(client)
    package = _served_package(client)
    audits_before = len(db_session.execute(select(AuditEvent)).scalars().all())

    assert client.post(URL, json=package).json() == {"valid": True}
    assert client.post(URL, json=package).json() == {"valid": True}
    # No session is injected: the route cannot write an audit event.
    audits_after = len(db_session.execute(select(AuditEvent)).scalars().all())
    assert audits_after == audits_before


# --- Malformed JSON and structure ----------------------------------------------


def test_malformed_json_is_422(client):
    for raw in (
        b"not json",
        b"{",
        b"",
        b'{"checkpoint":',
        b"[]",
        b'"a string"',
        b"42",
        b"null",
        b"true",
    ):
        resp = client.post(
            URL, content=raw, headers={"content-type": "application/json"}
        )
        _assert_validation_error(resp)


def test_root_must_be_an_object_with_exactly_two_members(client):
    good = _request()
    for payload in (
        [],
        "string",
        42,
        None,
        {},
        {"checkpoint": good["checkpoint"]},
        {"impacts": good["impacts"]},
        {**good, "extra": 1},
    ):
        _assert_validation_error(client.post(URL, json=payload))


def test_checkpoint_must_be_an_object_with_exactly_four_members(client):
    good = _request()
    base = good["checkpoint"]
    for checkpoint in (
        None,
        [],
        "x",
        42,
        {},
        {k: v for k, v in base.items() if k != "checkpoint_version"},
        {k: v for k, v in base.items() if k != "impacts_digest_hex"},
        {**base, "extra": 1},
    ):
        _assert_validation_error(
            client.post(URL, json={"checkpoint": checkpoint, "impacts": []})
        )


def test_impacts_must_be_an_array(client):
    good = _request()
    for impacts in (None, {}, "x", 42, True):
        _assert_validation_error(
            client.post(
                URL,
                json={"checkpoint": good["checkpoint"], "impacts": impacts},
            )
        )


# --- Checkpoint fields ---------------------------------------------------------


def test_checkpoint_version_and_algorithm_are_pinned(client):
    good = _request()
    for key, value in (
        ("checkpoint_version", "provenance-audit-checkpoint-v1"),
        ("checkpoint_version", ""),
        ("checkpoint_version", None),
        ("checkpoint_version", 42),
        ("digest_algorithm", "sha512"),
        ("digest_algorithm", "SHA256"),
        ("digest_algorithm", None),
    ):
        checkpoint = dict(good["checkpoint"])
        checkpoint[key] = value
        _assert_validation_error(
            client.post(
                URL, json={"checkpoint": checkpoint, "impacts": good["impacts"]}
            )
        )


def test_impact_count_validation(client):
    good = _request()
    for count in (-1, -100, "1", 1.0, None, True, False):
        checkpoint = dict(good["checkpoint"])
        checkpoint["impact_count"] = count
        _assert_validation_error(
            client.post(
                URL,
                json={"checkpoint": checkpoint, "impacts": good["impacts"]},
            )
        )


def test_impact_count_must_equal_array_length(client):
    impacts = [_item(), _item(id="rev_2")]
    for count in (0, 1, 3):
        _assert_validation_error(
            client.post(
                URL,
                json={
                    "checkpoint": {
                        "checkpoint_version": CHECKPOINT_VERSION,
                        "digest_algorithm": "sha256",
                        "impact_count": count,
                        "impacts_digest_hex": _digest_of(impacts),
                    },
                    "impacts": impacts,
                },
            )
        )


def test_digest_hex_must_be_64_lowercase_hex(client):
    good = _request()
    valid = good["checkpoint"]["impacts_digest_hex"]
    for value in (
        valid.upper(),
        valid[:-1],
        valid + "00",
        "g" * 64,
        " " + valid,
        valid + " ",
        "",
        None,
        42,
        [],
    ):
        checkpoint = dict(good["checkpoint"])
        checkpoint["impacts_digest_hex"] = value
        _assert_validation_error(
            client.post(
                URL,
                json={"checkpoint": checkpoint, "impacts": good["impacts"]},
            )
        )


# --- Impact items --------------------------------------------------------------


def test_item_must_have_exactly_the_declared_fields(client):
    good = _item()
    for field in ITEM_FIELDS:
        item = {k: v for k, v in good.items() if k != field}
        _assert_validation_error(client.post(URL, json=_request([item])))
    item = {**good, "extra": 1}
    _assert_validation_error(client.post(URL, json=_request([item])))


def test_string_fields_must_be_nonempty_strings(client):
    for field in (
        "id",
        "attestation_id",
        "revoker_actor_id",
        "reason",
        "content_id",
        "signer_actor_id",
    ):
        for value in ("", "   ", 1, [], None, True):
            _assert_validation_error(
                client.post(URL, json=_request([_item(**{field: value})]))
            )


def test_enumerated_fields_are_strict(client):
    for value in ("", "Claim", "CLAIM", "content", None, 1):
        _assert_validation_error(
            client.post(
                URL, json=_request([_item(target_type=value)])
            )
        )
    for field in ("coverage_status_after", "coverage_status_before"):
        for value in ("", "Covered", "unknown", None, 1):
            _assert_validation_error(
                client.post(URL, json=_request([_item(**{field: value})]))
            )


def test_created_at_must_be_strict_rfc3339_utc(client):
    for value in (
        "",
        "2026-01-01",
        "2026-01-01T00:00",
        "2026-01-01T00:00:00",  # naive
        "2026-01-01T00:00:00z",  # lowercase z
        "2026-01-01T01:00:00+01:00",  # non-UTC offset
        "not-a-timestamp",
        "2026-13-01T00:00:00Z",
        None,
        42,
    ):
        _assert_validation_error(
            client.post(URL, json=_request([_item(created_at=value)]))
        )


def test_count_fields_must_be_non_negative_integers(client):
    for field in (
        "qualified_signer_count_after",
        "qualified_signer_count_before",
    ):
        for value in (-1, "0", 1.5, None, True, []):
            _assert_validation_error(
                client.post(URL, json=_request([_item(**{field: value})]))
            )


def test_delta_must_be_zero_or_one(client):
    # _item serves before=1/after=0, so delta 0 and 1 are both structurally
    # possible depending on consistency; 2, -1, and non-integers are not.
    for value in (2, -1, "1", 1.0, None, True):
        item = _item(
            qualified_signer_count_delta=value,
            qualified_signer_count_before=max(value if isinstance(value, int) and value >= 2 else 2, 2),
            qualified_signer_count_after=0,
        )
        _assert_validation_error(client.post(URL, json=_request([item])))


def test_delta_must_equal_before_minus_after(client):
    # Structurally well-typed but internally inconsistent rows are 422
    # even if the claimed digest matches the submitted array.
    inconsistent = [
        # delta 1 while before == after
        _item(
            qualified_signer_count_before=1,
            qualified_signer_count_after=1,
            qualified_signer_count_delta=1,
            coverage_status_after="covered",
        ),
        # delta 0 while before == after+1
        _item(
            qualified_signer_count_before=1,
            qualified_signer_count_after=0,
            qualified_signer_count_delta=0,
        ),
    ]
    for impacts in ([item] for item in inconsistent):
        request = _request(impacts)
        # A digest matching the submitted bytes does not rescue the
        # malformed association: still a 422.
        _assert_validation_error(client.post(URL, json=request))


def test_every_array_element_is_validated(client):
    impacts = [_item(), _item(id="")]
    _assert_validation_error(client.post(URL, json=_request(impacts)))


# --- Query parameters and methods ----------------------------------------------


def test_query_parameters_are_422(client):
    resp = client.post(f"{URL}?foo=1", json=_request())
    _assert_validation_error(resp)


def test_non_post_methods_are_405(client):
    for method in ("get", "put", "patch", "delete"):
        resp = getattr(client, method)(URL)
        assert resp.status_code == 405, resp.text
        assert resp.json()["error"]["code"] == "method_not_allowed"
