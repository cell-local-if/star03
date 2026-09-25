"""Tests for the read-only revocation-impact checkpoint package export.

Covers GET /v1/revocation-impact-package: the success body is exactly
{"checkpoint", "impacts"}; ``checkpoint`` is exactly
{"checkpoint_version", "digest_algorithm", "impact_count",
"impacts_digest_hex"} with the version pinned to
"provenance-revocation-impact-checkpoint-v1" and the algorithm to
"sha256", and ``impacts`` lists the same revocation-impact public views
the impact search serves for the same effective filter, in the
revocations' stable creation order, each with exactly the existing
13-field impact view. The checkpoint digest is independently recomputed
from the impacts member of the same response under the checkpoint
canonical rules (array order kept, nested object keys sorted by Unicode
code point, compact separators, unescaped non-ASCII, UTF-8), so both
halves are bound to one read state; the pair verifies offline via the
stateless POST /v1/impact-verifications route. attestation_id/
revoker_actor_id/reason are non-empty exact combinable filters and
from/to are strict RFC 3339 UTC inclusive bounds exactly as on the
impact search; limit/cursor and any other undeclared, blank, repeated,
or malformed parameter is a 422 validation_error, as is any non-empty
request body; an empty match set returns "impacts": [] together with
the deterministic digest of the empty array; and neither successful nor
failed package reads write any resource or audit rows. All fixtures
are deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance.app import create_app
from provenance.config import Settings
from provenance.models import (
    Actor,
    AttestationRevocation,
    AuditEvent,
    Claim,
    Content,
    EvidenceBundle,
)
from tests.test_revocation_impacts import _world

PACKAGE_PATH = "/v1/revocation-impact-package"
SEARCH_PATH = "/v1/revocation-impacts"
VERIFY_PATH = "/v1/impact-verifications"

CHECKPOINT_VERSION = "provenance-revocation-impact-checkpoint-v1"
DIGEST_ALGORITHM = "sha256"

PACKAGE_FIELDS = {"checkpoint", "impacts"}
CHECKPOINT_FIELDS = {
    "checkpoint_version",
    "digest_algorithm",
    "impact_count",
    "impacts_digest_hex",
}
IMPACT_FIELDS = {
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
}

EMPTY_ARRAY_DIGEST = hashlib.sha256(b"[]").hexdigest()


def _package(client, **params):
    return client.get(PACKAGE_PATH, params=params)


def _search_items(client, **params):
    """The full unpaginated impact-search wire view for ``params``."""
    items = []
    cursor = None
    for _ in range(100):
        query = {**params}
        if cursor is not None:
            query["cursor"] = cursor
        resp = client.get(SEARCH_PATH, params=query)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        items.extend(body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            return items
    raise AssertionError("pagination did not terminate")


def _expected_digest(impacts):
    """Independently canonicalize the served impact items and digest them."""
    canonical = json.dumps(
        impacts,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- Response shape ---------------------------------------------------------------


def test_empty_database_yields_empty_array_and_checkpoint(client):
    resp = _package(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == PACKAGE_FIELDS
    assert list(body) == ["checkpoint", "impacts"]
    assert set(body["checkpoint"]) == CHECKPOINT_FIELDS
    assert list(body["checkpoint"]) == [
        "checkpoint_version",
        "digest_algorithm",
        "impact_count",
        "impacts_digest_hex",
    ]
    assert body["checkpoint"]["checkpoint_version"] == CHECKPOINT_VERSION
    assert body["checkpoint"]["digest_algorithm"] == DIGEST_ALGORITHM
    assert body["checkpoint"]["impact_count"] == 0
    assert body["checkpoint"]["impacts_digest_hex"] == EMPTY_ARRAY_DIGEST
    assert body["impacts"] == []


def test_success_body_has_exact_shape(client):
    world = _world(client)
    body = _package(client).json()
    assert set(body) == PACKAGE_FIELDS
    assert set(body["checkpoint"]) == CHECKPOINT_FIELDS
    assert body["checkpoint"]["checkpoint_version"] == CHECKPOINT_VERSION
    assert body["checkpoint"]["digest_algorithm"] == DIGEST_ALGORITHM
    assert body["checkpoint"]["impact_count"] == len(world["revocations"]) == 6
    assert len(body["impacts"]) == 6
    # Every item carries exactly the existing 13-field impact view, in the
    # public-view member order.
    for impact in body["impacts"]:
        assert set(impact) == IMPACT_FIELDS
        assert list(impact) == [
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
        ]


def test_no_raw_material_or_internal_fields_are_echoed(client):
    _world(client)
    rendered = _package(client).content.decode()
    # No proof bytes, public key, claim payload, ordering surrogate, or raw
    # material field can ever appear (target_type's "evidence_bundle"
    # literal is the public association, not evidence bytes).
    for forbidden in (
        "signature",
        "public_key",
        "payload",
        '"seq"',
        '"evidence"',
        "content_bytes",
    ):
        assert forbidden not in rendered


def test_response_is_compact_json_with_one_newline(client):
    _world(client)
    raw = _package(client).content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    assert raw.startswith(b'{"checkpoint":{"checkpoint_version":"')
    expected = (
        json.dumps(
            json.loads(raw),
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    assert raw == expected


def test_non_ascii_is_emitted_unescaped(client):
    from tests.test_revocation_impacts import (
        DIGEST_D,
        _attest,
        _make_claim,
        _make_content,
        _revoke,
    )

    from tests.helpers import create_actor

    create_actor(client)
    c = _make_content(client, DIGEST_D)
    cl = _make_claim(client, c["id"])
    att = _attest(client, "claim", cl["id"])
    _revoke(client, att["id"], reason="Képek ⛄")
    raw = _package(client).content.decode("utf-8")
    assert "Képek ⛄" in raw
    assert "\\u" not in raw


# --- Consistency with the existing impact search ----------------------------------


def test_impacts_member_equals_impact_search_view(client):
    _world(client)
    package = _package(client).json()
    searched = _search_items(client)
    assert package["impacts"] == searched
    assert package["checkpoint"]["impact_count"] == len(searched)


def test_filters_match_the_impact_search_route(client):
    world = _world(client)
    rev1, rev2, rev4 = world["revocations"][0], world["revocations"][1], world["revocations"][3]
    for params in (
        {"attestation_id": rev1["attestation_id"]},
        {"revoker_actor_id": "p-1"},
        {"reason": "fourth"},
        {"attestation_id": rev2["attestation_id"], "reason": "second"},
        {"revoker_actor_id": "nobody"},
        {"reason": "FIRST"},
        {"reason": "first "},
        {"revoker_actor_id": rev4["revoker_actor_id"], "reason": "fourth"},
    ):
        package = _package(client, **params).json()
        searched = _search_items(client, **params)
        assert package["impacts"] == searched, params
        assert package["checkpoint"]["impact_count"] == len(searched), params
        assert package["checkpoint"]["impacts_digest_hex"] == _expected_digest(
            package["impacts"]
        ), params


def test_time_bounds_match_the_impact_search_route(client):
    world = _world(client)
    first_created = world["revocations"][0]["created_at"]
    last_created = world["revocations"][-1]["created_at"]
    for params in (
        {"from": first_created},
        {"to": last_created},
        {"from": first_created, "to": last_created},
        # Equivalent Z vs +00:00 spellings canonicalize to the same instant.
        {"from": first_created.replace("Z", "+00:00")},
        {"to": last_created.replace("Z", "+00:00")},
    ):
        package = _package(client, **params).json()
        searched = _search_items(client, **params)
        assert package["impacts"] == searched, params
        assert package["checkpoint"]["impact_count"] == len(searched)


def test_package_accepts_no_pagination_parameters(client):
    # The package snapshots the entire filtered set in one read; the
    # search route's limit/cursor are not part of its contract and are
    # rejected like any other undeclared parameter.
    _world(client)
    _assert_validation_error(_package(client, limit=1))
    _assert_validation_error(_package(client, cursor="abc"))
    _assert_validation_error(_package(client, limit=1, cursor="abc"))


# --- Digest binding ---------------------------------------------------------------


def test_checkpoint_digest_binds_to_returned_impacts(client):
    _world(client)
    body = _package(client).json()
    assert body["checkpoint"]["impacts_digest_hex"] == _expected_digest(
        body["impacts"]
    )

    tampered = json.loads(json.dumps(body["impacts"]))
    tampered[0], tampered[1] = tampered[1], tampered[0]
    assert body["checkpoint"]["impacts_digest_hex"] != _expected_digest(tampered)

    augmented = list(body["impacts"])
    augmented.append(dict(body["impacts"][0]))
    assert body["checkpoint"]["impacts_digest_hex"] != _expected_digest(augmented)

    dropped = body["impacts"][:-1]
    assert body["checkpoint"]["impacts_digest_hex"] != _expected_digest(dropped)


def test_nested_object_key_order_does_not_change_digest(client):
    # The digest sorts nested object keys by Unicode code point, so a
    # verifier reproducing the canonical form gets the same value even
    # though the wire view keeps the public-view field order.
    body = _package(client).json()
    reordered = []
    for impact in body["impacts"]:
        reordered.append({k: impact[k] for k in sorted(impact, reverse=True)})
    assert body["checkpoint"]["impacts_digest_hex"] == _expected_digest(reordered)


def test_package_pair_verifies_offline(client):
    _world(client)
    body = _package(client, reason="second").json()
    resp = client.post(
        VERIFY_PATH,
        json={"checkpoint": body["checkpoint"], "impacts": body["impacts"]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_empty_pair_verifies_offline(client):
    body = _package(client, revoker_actor_id="nobody").json()
    assert body["impacts"] == []
    resp = client.post(
        VERIFY_PATH,
        json={"checkpoint": body["checkpoint"], "impacts": body["impacts"]},
    )
    assert resp.status_code == 200
    assert resp.json() == {"valid": True}


# --- Parameter and body validation ------------------------------------------------


def test_limit_and_cursor_are_undeclared_validation_errors(client):
    _world(client)
    for suffix in ("limit=1", "cursor=x", "limit=50&cursor=abc"):
        resp = client.get(f"{PACKAGE_PATH}?{suffix}")
        _assert_validation_error(resp)


def test_undeclared_parameters_are_validation_errors(client):
    _world(client)
    for suffix in (
        "actor_id=org-1",
        "attestation_ids=att_x",
        "offset=2",
        "FROM=2026-01-01T00:00:00Z",
        "checkpoint_version=x",
        "count=true",
    ):
        resp = client.get(f"{PACKAGE_PATH}?{suffix}")
        _assert_validation_error(resp)


def test_repeated_parameters_are_validation_errors(client):
    _world(client)
    for suffix in (
        "attestation_id=a&attestation_id=b",
        "revoker_actor_id=a&revoker_actor_id=b",
        "reason=a&reason=b",
        "from=2026-01-01T00:00:00Z&from=2026-01-02T00:00:00Z",
        "to=2026-01-01T00:00:00Z&to=2026-01-02T00:00:00Z",
    ):
        resp = client.get(f"{PACKAGE_PATH}?{suffix}")
        _assert_validation_error(resp)


def test_blank_filters_are_validation_errors(client):
    _world(client)
    for field in ("attestation_id", "revoker_actor_id", "reason"):
        for value in ("", "   ", "\t"):
            _assert_validation_error(_package(client, **{field: value}))


def test_invalid_time_values_are_validation_errors(client):
    for field in ("from", "to"):
        for value in (
            "",
            "   ",
            "2026-01-02",
            "2026-01-02T12:30:00",
            "2026-01-02 12:30:00Z",
            "2026-01-02T12:30Z",
            "2026-01-02T12:30:00+01:00",
            "2026-13-02T12:30:00Z",
            "not-a-time",
        ):
            _assert_validation_error(_package(client, **{field: value}))


def test_from_later_than_to_is_validation_error(client):
    _world(client)
    resp = _package(
        client,
        **{"from": "2027-01-03T00:00:00Z", "to": "2026-01-02T00:00:00Z"},
    )
    _assert_validation_error(resp)


def test_any_request_body_is_a_validation_error(client):
    _world(client)
    for body_bytes in (b" ", b"\t\n", b"{}", b"not json", b"\xff\xfe"):
        resp = client.request("GET", PACKAGE_PATH, content=body_bytes)
        _assert_validation_error(resp)


def test_package_validation_boundaries_match_search_route(client):
    # The package route shares the search route's filter contract: the
    # invalid queries they have in common must render the same 422
    # validation_error JSON. (limit/cursor are valid on the search route
    # and rejected on the package route, so they are covered separately.)
    cases = (
        "unknown=1",
        "attestation_id=a&attestation_id=b",
        "reason=",
        "from=not-a-time",
        "from=2027-01-01T00:00:00Z&to=2026-01-01T00:00:00Z",
        "revoker_actor_id=",
    )
    for suffix in cases:
        package_resp = client.get(f"{PACKAGE_PATH}?{suffix}")
        search_resp = client.get(f"{SEARCH_PATH}?{suffix}")
        assert package_resp.status_code == search_resp.status_code == 422
        assert package_resp.json() == search_resp.json()


def test_non_get_methods_are_allowed_then_405(client):
    # The export is a GET; POST (despite the sibling verification route)
    # and the mutating verbs are not part of its contract.
    for method in ("post", "put", "patch", "delete"):
        resp = getattr(client, method)(PACKAGE_PATH)
        assert resp.status_code == 405, (method, resp.text)
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Read-only guarantee ----------------------------------------------------------


def test_reads_and_failures_write_no_rows_or_audit_events(client, db_session):
    _world(client)
    models = (
        Actor,
        Content,
        Claim,
        EvidenceBundle,
        AttestationRevocation,
        AuditEvent,
    )

    def counts():
        return tuple(
            db_session.scalar(select(func.count()).select_from(model))
            for model in models
        )

    before = counts()

    _package(client)
    _package(client, revoker_actor_id="p-1")
    _package(client, reason="no.such.reason")
    _package(client, attestation_id="att_missing")
    _package(
        client,
        **{
            "revoker_actor_id": "org-1",
            "from": "2026-01-01T00:00:00Z",
            "to": "2027-01-01T00:00:00Z",
        },
    )

    # Failures must not write anything either.
    _package(client, reason=" ")
    _package(client, limit=1)
    _package(client, cursor="tampered")
    _package(client, **{"from": "not-a-time"})
    _package(
        client,
        **{"from": "2027-01-01T00:00:00Z", "to": "2026-01-01T00:00:00Z"},
    )
    client.get(f"{PACKAGE_PATH}?reason=a&reason=b")
    client.get(f"{PACKAGE_PATH}?unknown=1")
    client.request("GET", PACKAGE_PATH, content=b" ")

    db_session.expire_all()
    assert counts() == before


def test_package_is_deterministic_across_repeated_reads(client):
    _world(client)
    first = _package(client).content
    second = _package(client).content
    assert first == second


def test_package_is_deterministic_across_restart(file_client, tmp_db_url):
    _world(file_client)
    expected = _package(file_client)
    assert expected.status_code == 200, expected.text
    expected_body = expected.content

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        resp = _package(client)
        assert resp.status_code == 200, resp.text
        assert resp.content == expected_body
        assert _package(client).content == expected_body
