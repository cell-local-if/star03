"""Tests for the read-only correction checkpoint package export.

Covers GET /v1/csp/package: the success body is exactly
{"checkpoint", "corrections"}; ``checkpoint`` is exactly
{"checkpoint_version", "digest_algorithm", "correction_count",
"corrections_digest_hex"} with the version pinned to
"provenance-csp-checkpoint-v1" and the algorithm to "sha256", and
``corrections`` lists the same claim-supersession (correction) public
views the supersession search serves for the same effective filter, in
the supersessions' stable creation order, each with exactly the existing
5-field public view. The checkpoint digest is independently recomputed
from the corrections member of the same response under the checkpoint
canonical rules (array order kept, nested object keys sorted by Unicode
code point, compact separators, unescaped non-ASCII, UTF-8), so both
halves are bound to one read state; the pair verifies offline via the
stateless POST /v1/csp/verify route. id/superseded_claim_id/
replacement_claim_id/reason are non-empty exact combinable filters and
from/to are strict RFC 3339 UTC inclusive bounds exactly as on the
supersession search; limit/cursor and any other undeclared, blank,
repeated, or malformed parameter is a 422 validation_error, as is any
non-empty request body; an empty match set returns "corrections": []
together with the deterministic digest of the empty array; and neither
successful nor failed package reads write any resource or audit rows.
All fixtures are deterministic and offline.
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
    DIGEST_C,
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

EMPTY_ARRAY_DIGEST = hashlib.sha256(b"[]").hexdigest()


# --- Fixture-style setup ------------------------------------------------------


def _make_content(client, digest, actor_id="org-1"):
    resp = client.post(
        "/v1/contents",
        json=content_payload(digest=digest, actor_id=actor_id),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make_claim(client, content_id, statement, actor_id="org-1"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": "authorship",
            "payload": {"statement": statement},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _supersede(client, superseded_claim_id, replacement_claim_id, reason):
    resp = client.post(
        "/v1/claim-supersessions",
        json={
            "superseded_claim_id": superseded_claim_id,
            "replacement_claim_id": replacement_claim_id,
            "reason": reason,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _world(client):
    """Three contents with five correction records of distinct shapes.

    * content A: a two-step correction chain (a1 -> a2 -> a3) with
      reasons "first"/"second";
    * content B: a single correction with reason "third";
    * content C: a two-step chain whose first record carries the
      non-ASCII reason "Képek ⛄" and whose second carries "fifth".
    """
    create_actor(client, actor_id="org-1", name="Example Org", type="organization")

    ca = _make_content(client, DIGEST_A)
    a1 = _make_claim(client, ca["id"], "a1")
    a2 = _make_claim(client, ca["id"], "a2")
    a3 = _make_claim(client, ca["id"], "a3")
    s1 = _supersede(client, a1["id"], a2["id"], "first")
    s2 = _supersede(client, a2["id"], a3["id"], "second")

    cb = _make_content(client, DIGEST_B)
    b1 = _make_claim(client, cb["id"], "b1")
    b2 = _make_claim(client, cb["id"], "b2")
    s3 = _supersede(client, b1["id"], b2["id"], "third")

    cc = _make_content(client, DIGEST_C)
    c1 = _make_claim(client, cc["id"], "c1")
    c2 = _make_claim(client, cc["id"], "c2")
    c3 = _make_claim(client, cc["id"], "c3")
    s4 = _supersede(client, c1["id"], c2["id"], "Képek ⛄")
    s5 = _supersede(client, c2["id"], c3["id"], "fifth")

    return {
        "contents": [ca, cb, cc],
        "claims": [a1, a2, a3, b1, b2, c1, c2, c3],
        "supersessions": [s1, s2, s3, s4, s5],
    }


def _package(client, **params):
    return client.get(PACKAGE_PATH, params=params)


def _search_items(client, **params):
    """The full unpaginated supersession-search wire view for ``params``."""
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
    """Independently canonicalize the served correction items and digest them."""
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
    world = _world(client)
    body = _package(client).json()
    assert set(body) == PACKAGE_FIELDS
    assert set(body["checkpoint"]) == CHECKPOINT_FIELDS
    assert body["checkpoint"]["checkpoint_version"] == CHECKPOINT_VERSION
    assert body["checkpoint"]["digest_algorithm"] == DIGEST_ALGORITHM
    assert body["checkpoint"]["correction_count"] == len(
        world["supersessions"]
    ) == 5
    assert len(body["corrections"]) == 5
    # Every item carries exactly the existing 5-field public view, in the
    # public-view member order.
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
    # No claim payload, signature, public key, ordering surrogate, or raw
    # material field can ever appear.
    for forbidden in (
        "signature",
        "public_key",
        "payload",
        '"seq"',
        '"evidence"',
        "content_bytes",
        "statement",
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
    raw = _package(client).content.decode("utf-8")
    assert "Képek ⛄" in raw
    assert "\\u" not in raw


# --- Consistency with the existing supersession search -------------------------


def test_corrections_member_equals_supersession_search_view(client):
    _world(client)
    package = _package(client).json()
    searched = _search_items(client)
    assert package["corrections"] == searched
    assert package["checkpoint"]["correction_count"] == len(searched)


def test_filters_match_the_supersession_search_route(client):
    world = _world(client)
    s1, s2, s4 = (
        world["supersessions"][0],
        world["supersessions"][1],
        world["supersessions"][3],
    )
    for params in (
        {"id": s1["id"]},
        {"superseded_claim_id": s2["superseded_claim_id"]},
        {"replacement_claim_id": s2["replacement_claim_id"]},
        {"reason": "third"},
        {"reason": "Képek ⛄"},
        {
            "superseded_claim_id": s1["superseded_claim_id"],
            "reason": "first",
        },
        {"id": "csp_" + "0" * 64},
        {"reason": "FIRST"},
        {"reason": "first "},
        {
            "superseded_claim_id": s4["superseded_claim_id"],
            "replacement_claim_id": s4["replacement_claim_id"],
        },
    ):
        package = _package(client, **params).json()
        searched = _search_items(client, **params)
        assert package["corrections"] == searched, params
        assert package["checkpoint"]["correction_count"] == len(searched), params
        assert package["checkpoint"]["corrections_digest_hex"] == (
            _expected_digest(package["corrections"])
        ), params


def test_time_bounds_match_the_supersession_search_route(client):
    world = _world(client)
    first_created = world["supersessions"][0]["created_at"]
    last_created = world["supersessions"][-1]["created_at"]
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
        assert package["corrections"] == searched, params
        assert package["checkpoint"]["correction_count"] == len(searched)


def test_package_accepts_no_pagination_parameters(client):
    # The package snapshots the entire filtered set in one read; the
    # search route's limit/cursor are not part of its contract and are
    # rejected like any other undeclared parameter.
    _world(client)
    _assert_validation_error(_package(client, limit=1))
    _assert_validation_error(_package(client, cursor="abc"))
    _assert_validation_error(_package(client, limit=1, cursor="abc"))


# --- Digest binding -------------------------------------------------------------


def test_checkpoint_digest_binds_to_returned_corrections(client):
    _world(client)
    body = _package(client).json()
    assert body["checkpoint"]["corrections_digest_hex"] == _expected_digest(
        body["corrections"]
    )

    tampered = json.loads(json.dumps(body["corrections"]))
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
    reordered = []
    for correction in body["corrections"]:
        reordered.append(
            {k: correction[k] for k in sorted(correction, reverse=True)}
        )
    assert body["checkpoint"]["corrections_digest_hex"] == _expected_digest(
        reordered
    )


def test_package_pair_verifies_offline(client):
    _world(client)
    body = _package(client, reason="second").json()
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


# --- Parameter and body validation ----------------------------------------------


def test_limit_and_cursor_are_undeclared_validation_errors(client):
    _world(client)
    for suffix in ("limit=1", "cursor=x", "limit=50&cursor=abc"):
        resp = client.get(f"{PACKAGE_PATH}?{suffix}")
        _assert_validation_error(resp)


def test_undeclared_parameters_are_validation_errors(client):
    _world(client)
    for suffix in (
        "actor_id=org-1",
        "supersession_id=csp_x",
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


# --- Read-only guarantee ---------------------------------------------------------


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
    _package(client, reason="first")
    _package(client, reason="no.such.reason")
    _package(client, id="csp_" + "0" * 64)
    _package(
        client,
        **{
            "reason": "first",
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
