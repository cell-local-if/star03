"""Tests for the read-only revocation-impact checkpoint package export.

Covers GET /v1/revocation-impact-package: the success body is exactly
{"checkpoint", "impacts"}; ``checkpoint`` is exactly
checkpoint_version/digest_algorithm/impact_count/impacts_digest_hex with
the version pinned to "provenance-revocation-impact-checkpoint-v1" and the
algorithm to "sha256", and ``impacts`` lists, in stable revocation
creation order, the same filtered impact public views the revocation-impact
search serves (unpaginated: no limit/cursor is accepted). The checkpoint
digest is independently recomputed from the impacts member of the same
response under the checkpoint canonical rules (array order kept, object
keys sorted by Unicode code point, compact separators, unescaped
non-ASCII, UTF-8), so both halves are bound to one read state; the pair
even verifies offline via the stateless impact-verification route.
attestation_id/revoker_actor_id/reason are non-empty exact combinable
filters and from/to are strict RFC 3339 UTC inclusive bounds exactly as on
the search route; limit/cursor and any other undeclared, blank, repeated,
or malformed parameter is a 422 validation_error; an empty match set
returns "impacts": [] together with the digest of the empty array; and
neither successful nor failed package reads write any resource or audit
rows. All fixtures are deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json

from sqlalchemy import select

from provenance.models import AuditEvent
from tests.helpers import DIGEST_A, create_actor
from tests.test_revocation_impacts import (
    _attest,
    _make_claim,
    _make_content,
    _revoke,
    _world,
)

PACKAGE_PATH = "/v1/revocation-impact-package"
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
        resp = client.get("/v1/revocation-impacts", params=query)
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


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


# --- Response shape ---------------------------------------------------------


def test_success_body_has_exact_shape_and_member_order(client):
    _world(client)
    resp = _package(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert list(body) == ["checkpoint", "impacts"]
    assert list(body["checkpoint"]) == [
        "checkpoint_version",
        "digest_algorithm",
        "impact_count",
        "impacts_digest_hex",
    ]
    assert body["checkpoint"]["checkpoint_version"] == CHECKPOINT_VERSION
    assert body["checkpoint"]["digest_algorithm"] == DIGEST_ALGORITHM
    assert body["checkpoint"]["impact_count"] == 6
    assert len(body["impacts"]) == 6
    for impact in body["impacts"]:
        assert set(impact) == IMPACT_FIELDS


def test_response_is_compact_json_with_one_newline(client):
    _world(client)
    resp = _package(client)
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    assert raw.startswith(b'{"checkpoint":{')
    expected = (
        json.dumps(
            resp.json(), separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        + b"\n"
    )
    assert raw == expected


def test_checkpoint_digest_binds_the_served_impacts(client):
    _world(client)
    body = _package(client).json()
    assert body["checkpoint"]["impacts_digest_hex"] == _expected_digest(
        body["impacts"]
    )


def test_impacts_match_the_search_wire_view(client):
    _world(client)
    body = _package(client).json()
    assert body["impacts"] == _search_items(client)


def test_package_verifies_via_the_stateless_route(client):
    _world(client)
    body = _package(client).json()
    resp = client.post("/v1/impact-verifications", json=body)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_non_ascii_is_emitted_unescaped_and_digested(client):
    create_actor(client)
    content = _make_content(client, DIGEST_A)
    claim = _make_claim(client, content["id"])
    attestation = _attest(client, "claim", claim["id"])
    _revoke(client, attestation["id"], reason="Képek ⛄")
    resp = _package(client)
    raw = resp.content.decode("utf-8")
    assert "Képek ⛄" in raw
    assert "\\u" not in raw
    body = resp.json()
    assert body["checkpoint"]["impacts_digest_hex"] == _expected_digest(
        body["impacts"]
    )


# --- Filtering ---------------------------------------------------------------


def test_filters_match_the_search_filters(client):
    world = _world(client)
    a5 = world["attestations"]["a5"]["id"]
    for params in (
        {"attestation_id": a5},
        {"revoker_actor_id": "p-1"},
        {"reason": "first"},
        {"revoker_actor_id": "p-1", "reason": "fourth"},
        {"from": "2000-01-01T00:00:00Z"},
        {"to": "2999-01-01T00:00:00Z"},
    ):
        body = _package(client, **params).json()
        expected = _search_items(client, **params)
        assert body["impacts"] == expected
        assert body["checkpoint"]["impact_count"] == len(expected)
        assert body["checkpoint"]["impacts_digest_hex"] == _expected_digest(
            expected
        )


def test_unknown_filter_values_are_empty_packages_not_404(client):
    _world(client)
    for params in (
        {"attestation_id": "att-does-not-exist"},
        {"revoker_actor_id": "no-such-actor"},
        {"reason": "no such reason"},
    ):
        resp = _package(client, **params)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["impacts"] == []
        assert body["checkpoint"]["impact_count"] == 0


def test_from_later_than_to_is_422(client):
    _world(client)
    resp = _package(
        client,
        **{"from": "2026-01-02T00:00:00Z", "to": "2026-01-01T00:00:00Z"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Empty result -------------------------------------------------------------


def test_empty_match_is_a_legal_snapshot(client):
    _world(client)
    resp = _package(client, reason="no such reason")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["impacts"] == []
    assert body["checkpoint"] == {
        "checkpoint_version": CHECKPOINT_VERSION,
        "digest_algorithm": DIGEST_ALGORITHM,
        "impact_count": 0,
        "impacts_digest_hex": hashlib.sha256(b"[]").hexdigest(),
    }
    # The empty snapshot verifies too.
    verify = client.post("/v1/impact-verifications", json=body)
    assert verify.json() == {"valid": True}


def test_empty_registry_is_an_empty_package(client):
    body = _package(client).json()
    assert body["impacts"] == []
    assert body["checkpoint"]["impact_count"] == 0
    assert body["checkpoint"]["impacts_digest_hex"] == hashlib.sha256(
        b"[]"
    ).hexdigest()


# --- Parameter validation -----------------------------------------------------


def test_limit_and_cursor_are_undeclared_parameters(client):
    _world(client)
    cursor = client.get("/v1/revocation-impacts", params={"limit": "2"}).json()[
        "next_cursor"
    ]
    assert cursor is not None
    for params in ({"limit": "2"}, {"cursor": cursor}, {"limit": "2", "cursor": cursor}):
        resp = _package(client, **params)
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "validation_error"


def test_unknown_and_repeated_parameters_are_422(client):
    _world(client)
    resp = _package(client, actor_id="org-1")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    for query in (
        "attestation_id=a&attestation_id=b",
        "revoker_actor_id=org-1&revoker_actor_id=p-1",
        "reason=a&reason=b",
        "from=2026-01-01T00:00:00Z&from=2026-01-02T00:00:00Z",
        "to=2026-01-01T00:00:00Z&to=2026-01-02T00:00:00Z",
    ):
        resp = client.get(f"{PACKAGE_PATH}?{query}")
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "validation_error"


def test_blank_and_malformed_filters_are_422(client):
    _world(client)
    for field in ("attestation_id", "revoker_actor_id", "reason"):
        for value in ("", " "):
            resp = _package(client, **{field: value})
            assert resp.status_code == 422, resp.text
            assert resp.json()["error"]["code"] == "validation_error"
    for field, value in (
        ("from", ""),
        ("from", "2026-01-01"),
        ("from", "2026-01-01T00:00:00"),
        ("from", "2026-01-01T00:00:00z"),
        ("from", "2026-01-01T01:00:00+01:00"),
        ("to", "not-a-timestamp"),
    ):
        resp = _package(client, **{field: value})
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "validation_error"


def test_any_request_body_is_422(client):
    _world(client)
    for body in (b"{}", b" ", b"\n", b"not json", b"[]"):
        resp = client.request("GET", PACKAGE_PATH, content=body)
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_get_methods_are_405(client):
    _world(client)
    for method in ("put", "patch", "delete", "post"):
        resp = getattr(client, method)(PACKAGE_PATH)
        assert resp.status_code == 405, resp.text
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Read-only guarantee ------------------------------------------------------


def test_export_is_strictly_read_only(client, db_session):
    _world(client)
    audits_before = _audit_count(db_session)

    _package(client)
    _package(client, revoker_actor_id="org-1")
    _package(client, reason="no such reason")
    _package(client, limit="2")  # a rejected read
    client.request("GET", PACKAGE_PATH, content=b"{}")  # a rejected read

    assert _audit_count(db_session) == audits_before


def test_checkpoint_is_stable_across_repeated_reads(client):
    _world(client)
    first = _package(client).json()
    second = _package(client).json()
    assert first == second
