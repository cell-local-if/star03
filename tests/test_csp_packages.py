"""Tests for the immutable-declaration correction checkpoint package export.

Covers GET /v1/csp/package: the success body is exactly
{"checkpoint", "corrections"}; ``checkpoint`` is exactly
{"checkpoint_version", "digest_algorithm", "correction_count",
"corrections_digest_hex"} with the version pinned to
"provenance-csp-checkpoint-v1" and the algorithm to "sha256", and
``corrections`` lists the same claim-supersession (correction) public
views the supersession search serves for the same effective filter, in
the supersessions' stable creation order, each with exactly the existing
five-field view (id, superseded_claim_id, replacement_claim_id, reason,
created_at). The checkpoint digest is independently recomputed from the
corrections member of the same response under the checkpoint canonical
rules (array order kept, nested object keys sorted by Unicode code
point, compact separators, unescaped non-ASCII, UTF-8), so both halves
are bound to one read state; the pair verifies offline via the
stateless POST /v1/csp/verify route. id/superseded_claim_id/
replacement_claim_id/reason are non-empty, case/whitespace-sensitive
exact combinable filters and from/to are strict RFC 3339 UTC inclusive
bounds exactly as on the supersession search; limit/cursor and any
other undeclared, blank, repeated, or malformed parameter is a 422
validation_error, as is any non-empty request body; an empty match set
returns "corrections": [] together with the deterministic digest of
the empty array; a from later than to is a 422 returning no partial
results; and neither successful nor failed package reads write any
resource or audit rows. The existing correction (supersession)
interfaces are unchanged. All fixtures are deterministic and offline.
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
    AuditEvent,
    Claim,
    ClaimSupersession,
    Content,
)
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    content_payload,
    create_actor,
)

PACKAGE_PATH = "/v1/csp/package"
SEARCH_PATH = "/v1/claim-supersessions"
VERIFY_PATH = "/v1/csp/verify"

CHECKPOINT_VERSION = "provenance-csp-checkpoint-v1"
DIGEST_ALGORITHM = "sha256"

PACKAGE_FIELDS = {"checkpoint", "corrections"}
CHECKPOINT_FIELDS = {
    "checkpoint_version",
    "digest_algorithm",
    "correction_count",
    "corrections_digest_hex",
}
CORRECTION_FIELDS = {
    "id",
    "superseded_claim_id",
    "replacement_claim_id",
    "reason",
    "created_at",
}

REASON_1 = "corrected source claim"
REASON_2 = "new evidence attached"
REASON_3 = "撤回与勘误"

EMPTY_ARRAY_DIGEST = hashlib.sha256(b"[]").hexdigest()


# --- Fixture-style world ------------------------------------------------------


def _create_content(client, digest=DIGEST_A, actor_id="org-1"):
    resp = client.post(
        "/v1/contents",
        json=content_payload(actor_id=actor_id, digest=digest),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, statement, claim_type="authorship"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": "org-1",
            "claim_type": claim_type,
            "payload": {"statement": statement},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_supersession(client, old, new, reason):
    resp = client.post(
        SEARCH_PATH,
        json={
            "superseded_claim_id": old,
            "replacement_claim_id": new,
            "reason": reason,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _world(client):
    """Two contents and three corrections, in a fixed creation order."""
    create_actor(client)
    a = _create_content(client, digest=DIGEST_A)
    b = _create_content(client, digest=DIGEST_B)
    a1 = _create_claim(client, a["id"], "first")
    a2 = _create_claim(client, a["id"], "second")
    a3 = _create_claim(client, a["id"], "third")
    b1 = _create_claim(client, b["id"], "b-first")
    b2 = _create_claim(client, b["id"], "b-second")

    s1 = _create_supersession(client, a1["id"], a2["id"], REASON_1)
    s2 = _create_supersession(client, a2["id"], a3["id"], REASON_2)
    s3 = _create_supersession(client, b1["id"], b2["id"], REASON_3)
    return [s1, s2, s3]


def _package(client, **params):
    return client.get(PACKAGE_PATH, params=params)


def _search_items(client, **params):
    """The full supersession-search wire view for ``params."""
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


def _expected_digest(corrections):
    """Independently canonicalize the served corrections and digest them."""
    canonical = json.dumps(
        corrections,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- Response shape -----------------------------------------------------------


def test_empty_database_yields_empty_array_and_checkpoint(client):
    resp = _package(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == PACKAGE_FIELDS
    assert list(body) == ["checkpoint", "corrections"]
    assert set(body["checkpoint"]) == CHECKPOINT_FIELDS
    assert list(body["checkpoint"]) == [
        "checkpoint_version",
        "digest_algorithm",
        "correction_count",
        "corrections_digest_hex",
    ]
    assert body["checkpoint"]["checkpoint_version"] == CHECKPOINT_VERSION
    assert body["checkpoint"]["digest_algorithm"] == DIGEST_ALGORITHM
    assert body["checkpoint"]["correction_count"] == 0
    assert body["checkpoint"]["corrections_digest_hex"] == EMPTY_ARRAY_DIGEST
    assert body["corrections"] == []


def test_success_body_has_exact_shape(client):
    corrections = _world(client)
    body = _package(client).json()
    assert set(body) == PACKAGE_FIELDS
    assert set(body["checkpoint"]) == CHECKPOINT_FIELDS
    assert body["checkpoint"]["checkpoint_version"] == CHECKPOINT_VERSION
    assert body["checkpoint"]["digest_algorithm"] == DIGEST_ALGORITHM
    assert body["checkpoint"]["correction_count"] == len(corrections) == 3
    assert len(body["corrections"]) == 3
    for correction in body["corrections"]:
        assert set(correction) == CORRECTION_FIELDS
        assert list(correction) == [
            "id",
            "superseded_claim_id",
            "replacement_claim_id",
            "reason",
            "created_at",
        ]


def test_no_raw_material_or_internal_fields_are_echoed(client):
    _world(client)
    rendered = _package(client).content.decode()
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
    _world(client)
    raw = _package(client, reason=REASON_3).content.decode("utf-8")
    assert REASON_3 in raw
    assert "\\u" not in raw


# --- Consistency with the existing correction search ---------------------------


def test_corrections_member_equals_supersession_search_view(client):
    _world(client)
    package = _package(client).json()
    searched = _search_items(client)
    assert package["corrections"] == searched
    assert package["checkpoint"]["correction_count"] == len(searched)


def test_filters_match_the_supersession_search_route(client):
    corrections = _world(client)
    first = corrections[0]
    for params in (
        {"id": first["id"]},
        {"superseded_claim_id": first["superseded_claim_id"]},
        {"replacement_claim_id": first["replacement_claim_id"]},
        {"reason": REASON_1},
        {"id": first["id"], "reason": REASON_1},
        {"superseded_claim_id": "clm_missing"},
        {"reason": "nobody"},
        {"reason": "CORRECTED SOURCE CLAIM"},
        {"reason": "corrected source claim "},
    ):
        package = _package(client, **params).json()
        searched = _search_items(client, **params)
        assert package["corrections"] == searched, params
        assert package["checkpoint"]["correction_count"] == len(searched), params
        assert package["checkpoint"]["corrections_digest_hex"] == (
            _expected_digest(package["corrections"])
        ), params


def test_text_filters_are_case_and_whitespace_sensitive(client):
    corrections = _world(client)
    # Exact spelling hits; whitespace- or case-different spellings miss.
    assert (
        _package(client, reason=REASON_1).json()["checkpoint"]["correction_count"]
        == 1
    )
    empty = _package(client, reason=f" {REASON_1}")
    assert empty.json()["checkpoint"]["correction_count"] == 0
    by_id = _package(client, id=corrections[0]["id"])
    assert by_id.json()["corrections"][0]["id"] == corrections[0]["id"]
    # The id filter is not trimmed or normalized: padded spelling misses.
    assert (
        _package(client, id=f" {corrections[0]['id']}")
        .json()["checkpoint"]["correction_count"]
        == 0
    )


def test_time_bounds_match_the_supersession_search_route(client):
    corrections = _world(client)
    first_created = corrections[0]["created_at"]
    last_created = corrections[-1]["created_at"]
    for params in (
        {"from": first_created},
        {"to": last_created},
        {"from": first_created, "to": last_created},
        # Equivalent Z vs +00:00 spellings canonicalize to the same instant.
        {"from": first_created.replace("Z", "+00:00")},
        {"to": last_created.replace("Z", "+00:00")},
        # Inclusive at both boundaries.
        {"from": last_created, "to": last_created},
    ):
        package = _package(client, **params).json()
        searched = _search_items(client, **params)
        assert package["corrections"] == searched, params
        assert package["checkpoint"]["correction_count"] == len(searched)


def test_unknown_values_yield_empty_snapshot_not_an_error(client):
    _world(client)
    body = _package(
        client,
        id="csp_missing",
        superseded_claim_id="clm_missing",
        replacement_claim_id="clm_missing_too",
        reason="no such reason",
    ).json()
    assert body["corrections"] == []
    assert body["checkpoint"]["correction_count"] == 0
    assert body["checkpoint"]["corrections_digest_hex"] == EMPTY_ARRAY_DIGEST


def test_package_accepts_no_pagination_parameters(client):
    _world(client)
    _assert_validation_error(_package(client, limit=1))
    _assert_validation_error(_package(client, cursor="abc"))
    _assert_validation_error(_package(client, limit=1, cursor="abc"))


# --- Digest binding ------------------------------------------------------------


def test_checkpoint_digest_binds_to_returned_corrections(client):
    _world(client)
    body = _package(client).json()
    assert body["checkpoint"]["corrections_digest_hex"] == _expected_digest(
        body["corrections"]
    )

    tampered = list(body["corrections"])
    tampered[0], tampered[1] = tampered[1], tampered[0]
    assert body["checkpoint"]["corrections_digest_hex"] != _expected_digest(
        tampered
    )

    augmented = list(body["corrections"])
    augmented.append(dict(body["corrections"][0]))
    assert body["checkpoint"]["corrections_digest_hex"] != _expected_digest(
        augmented
    )

    dropped = body["corrections"][:-1]
    assert body["checkpoint"]["corrections_digest_hex"] != _expected_digest(
        dropped
    )


def test_nested_object_key_order_does_not_change_digest(client):
    # The digest sorts nested object keys by Unicode code point, so a
    # verifier reproducing the canonical form gets the same value even
    # though the wire view keeps the public-view field order.
    body = _package(client).json()
    reordered = [
        {k: correction[k] for k in sorted(correction, reverse=True)}
        for correction in body["corrections"]
    ]
    assert body["checkpoint"]["corrections_digest_hex"] == _expected_digest(
        reordered
    )


def test_package_pair_verifies_offline(client):
    _world(client)
    body = _package(client, reason=REASON_2).json()
    resp = client.post(
        VERIFY_PATH,
        json={
            "checkpoint": body["checkpoint"],
            "corrections": body["corrections"],
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_empty_pair_verifies_offline(client):
    body = _package(client, reason="no.such.reason").json()
    assert body["corrections"] == []
    resp = client.post(
        VERIFY_PATH,
        json={
            "checkpoint": body["checkpoint"],
            "corrections": body["corrections"],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"valid": True}


# --- Parameter and body validation ---------------------------------------------


def test_undeclared_parameters_are_validation_errors(client):
    _world(client)
    for suffix in (
        "limit=1",
        "cursor=x",
        "supersession_id=csp_1",
        "actor_id=org-1",
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
        "id=a&id=b",
        "superseded_claim_id=a&superseded_claim_id=b",
        "replacement_claim_id=a&replacement_claim_id=b",
        "reason=a&reason=b",
        "from=2026-01-01T00:00:00Z&from=2026-01-02T00:00:00Z",
        "to=2026-01-01T00:00:00Z&to=2026-01-02T00:00:00Z",
    ):
        resp = client.get(f"{PACKAGE_PATH}?{suffix}")
        _assert_validation_error(resp)


def test_blank_filters_are_validation_errors(client):
    _world(client)
    for field in (
        "id",
        "superseded_claim_id",
        "replacement_claim_id",
        "reason",
    ):
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
            "2026-01-02T12:30:00z",
            "2026-13-02T12:30:00Z",
            "not-a-time",
        ):
            _assert_validation_error(_package(client, **{field: value}))


def test_from_later_than_to_is_validation_error_with_no_partial_results(client):
    _world(client)
    resp = _package(
        client,
        **{"from": "2027-01-03T00:00:00Z", "to": "2026-01-02T00:00:00Z"},
    )
    _assert_validation_error(resp)
    # No partial snapshot is ever returned: the error envelope carries no
    # checkpoint or corrections member.
    assert "checkpoint" not in resp.json()["error"]
    assert "corrections" not in resp.json()["error"]


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
        "id=a&id=b",
        "reason=",
        "from=not-a-time",
        "from=2027-01-01T00:00:00Z&to=2026-01-01T00:00:00Z",
        "superseded_claim_id=",
    )
    for suffix in cases:
        package_resp = client.get(f"{PACKAGE_PATH}?{suffix}")
        search_resp = client.get(f"{SEARCH_PATH}?{suffix}")
        assert package_resp.status_code == search_resp.status_code == 422
        assert package_resp.json() == search_resp.json()


def test_non_get_methods_are_405(client):
    # The export is a GET; POST (despite the sibling verification route)
    # and the mutating verbs are not part of its contract.
    for method in ("post", "put", "patch", "delete"):
        resp = getattr(client, method)(PACKAGE_PATH)
        assert resp.status_code == 405, (method, resp.text)
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Read-only and determinism guarantees --------------------------------------


def test_reads_and_failures_write_no_rows_or_audit_events(client, db_session):
    _world(client)
    models = (
        Actor,
        Content,
        Claim,
        ClaimSupersession,
        AuditEvent,
    )

    def counts():
        return tuple(
            db_session.scalar(select(func.count()).select_from(model))
            for model in models
        )

    before = counts()

    _package(client)
    _package(client, reason=REASON_1)
    _package(client, id="csp_missing")
    _package(client, superseded_claim_id="clm_missing")
    _package(
        client,
        **{
            "reason": REASON_2,
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


def test_existing_correction_interfaces_are_unchanged(client):
    corrections = _world(client)
    # Creation idempotency still returns the original record with 200.
    retry = client.post(
        SEARCH_PATH,
        json={
            "superseded_claim_id": corrections[0]["superseded_claim_id"],
            "replacement_claim_id": corrections[0]["replacement_claim_id"],
            "reason": REASON_1,
        },
    )
    assert retry.status_code == 200, retry.text
    assert retry.json() == corrections[0]
    # Detail read and per-claim listing are unchanged.
    detail = client.get(f"{SEARCH_PATH}/{corrections[0]['id']}")
    assert detail.status_code == 200
    assert detail.json() == corrections[0]
    per_claim = client.get(
        f"/v1/claims/{corrections[0]['replacement_claim_id']}/supersessions"
    )
    assert per_claim.status_code == 200
    # a2 is the replacement of s1 and the superseded endpoint of s2.
    assert per_claim.json()["count"] == 2
