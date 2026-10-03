"""Tests for the trust-policy-revocation read and search routes.

Covers the two reviewer read routes added on top of the existing immutable
write route, without changing any creation semantics:

* ``GET /v1/trust-policy-revocations/{revocation_id}`` -- one existing
  revocation's public view, or an explicit 404 ``not_found`` for an unknown
  id;
* ``GET /v1/trust-policy-revocations`` -- a read-only search with the
  ``id``/``policy_id``/``actor_id``/``reason`` exact-match filters, the
  inclusive RFC 3339 UTC ``from``/``to`` bounds, and ``limit``/``cursor``
  pagination over the stable creation order.

The tests pin down, deterministically and offline:

* the exact five public fields per record (``id``, ``policy_id``,
  ``actor_id``, verbatim ``reason``, UTC ``created_at``) and the strict
  ``items``/``count``/``next_cursor`` envelope in compact UTF-8 JSON
  terminated by exactly one newline -- no private key, raw signature,
  claim payload, content, or evidence byte can appear;
* lookup isolation: the individual route keys solely on the revocation id
  and unknown filter values are empty collections, never 404s;
* parameter precedence: a non-empty body, an undeclared/blank/repeated
  parameter, a bad limit or time, or an invalid cursor is a 422 validated
  before any record is read;
* cursor discipline: the opaque cursor only resumes the same family, the
  same effective filters, and the same limit; the final page carries a null
  cursor and an out-of-range page is empty with the original count;
* strict zero-write behavior for successful, empty, missing, and invalid
  reads -- no revocation, policy, other resource, or audit event is created
  or modified;
* only GET is served (PUT/PATCH/DELETE are 405), no credentials are
  required, and the records persist unchanged across an app restart while
  outstanding cursors are invalidated by the secret rotation.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import select

from provenance import pagination
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import ActorTrustPolicyRevocation, AuditEvent
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    DIGEST_C,
    SEED_A,
    SEED_B,
    create_actor,
)
from tests.test_trust_policy_revocations import (
    POLICIES_PATH,
    REVOCATIONS_PATH,
    _attest,
    _create_policy,
    _make_claim,
    _post_revocation,
    _revoke,
    _world,
)

SEED_C = b"test-ed25519-seed-c-00000000000000"[:32]
SEED_D = b"test-ed25519-seed-d-00000000000000"[:32]
SEED_E = b"test-ed25519-seed-e-00000000000000"[:32]

DIGEST_D = hashlib.sha256(b"content-d").hexdigest()
DIGEST_E = hashlib.sha256(b"content-e").hexdigest()

PUBLIC_FIELDS = {"id", "policy_id", "actor_id", "reason", "created_at"}

REASON_A = "compromised root key"
REASON_B = "policy superseded"
REASON_C = "org migrated to a new root"


# --- Fixture-style setup ------------------------------------------------------


def _make_revoker(client, actor_id, seed, digest, *, threshold=1):
    """An actor with a current key and its own trust policy."""
    create_actor(client, actor_id=actor_id, name=f"{actor_id} Org",
                 type="organization")
    _attest(client, _make_claim(client, actor_id, digest=digest)["id"],
            actor_id, seed)
    return _create_policy(client, actor=actor_id, seed=seed, threshold=threshold)


def _world_revocations(client):
    """Five revocations across five subjects, in creation order."""
    _world(client)  # org-1 and org-2, each holding a current key
    p1 = _create_policy(client, actor="org-1", seed=SEED_A, threshold=1)
    p2 = _create_policy(client, actor="org-2", seed=SEED_B, threshold=1)
    p3 = _make_revoker(client, "org-3", SEED_C, DIGEST_C)
    p4 = _make_revoker(client, "org-4", SEED_D, DIGEST_D)
    p5 = _make_revoker(client, "org-5", SEED_E, DIGEST_E)
    r1 = _revoke(client, p1["id"], reason=REASON_A)
    r2 = _revoke(client, p2["id"], reason=REASON_B, actor="org-2", seed=SEED_B)
    r3 = _revoke(client, p3["id"], reason=REASON_A, actor="org-3", seed=SEED_C)
    r4 = _revoke(client, p4["id"], reason=REASON_C, actor="org-4", seed=SEED_D)
    r5 = _revoke(client, p5["id"], reason=REASON_A, actor="org-5", seed=SEED_E)
    return [r1, r2, r3, r4, r5]


def _pin_times(db_session, revocations, instants):
    """Overwrite created_at per revocation id with a fixed UTC instant."""
    by_id = {r["id"]: instant for r, instant in zip(revocations, instants)}
    for row in db_session.execute(
        select(ActorTrustPolicyRevocation)
    ).scalars():
        row.created_at = by_id[row.id]
    db_session.commit()


def _one_revocation(client):
    """The baseline world plus one revoked policy; returns (policy, record)."""
    _world(client)
    policy = _create_policy(client, threshold=2)
    record = _revoke(client, policy["id"], reason="key material retired")
    return policy, record


def _revocation_url(revocation_id: str) -> str:
    return f"{REVOCATIONS_PATH}/{revocation_id}"


def _list(client, **params):
    return client.get(REVOCATIONS_PATH, params=params)


def _walk_pages(client, **params):
    """Follow next_cursor until exhausted; return (all_items, pages, count)."""
    pages = []
    all_items = []
    count = None
    cursor = None
    for _ in range(100):
        query = {**params}
        if cursor is not None:
            query["cursor"] = cursor
        resp = _list(client, **query)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        count = body["count"]
        pages.append(body["items"])
        all_items.extend(body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            break
    return all_items, pages, count


def _assert_public_item(item: dict) -> None:
    assert set(item) == PUBLIC_FIELDS
    assert item["id"].startswith("tpr_")
    created_at = datetime.fromisoformat(item["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert item["created_at"].endswith(("Z", "+00:00"))


def _revocation_count(session) -> int:
    return len(session.execute(select(ActorTrustPolicyRevocation)).scalars().all())


def _audit_count(session) -> int:
    return len(session.execute(select(AuditEvent)).scalars().all())


# --- Individual resource: exact public view -----------------------------------


def test_get_revocation_returns_the_existing_public_view(client):
    policy, created = _one_revocation(client)

    resp = client.get(_revocation_url(created["id"]))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == created
    assert list(body) == ["id", "policy_id", "actor_id", "reason", "created_at"]
    _assert_public_item(body)
    assert body["policy_id"] == policy["id"]
    assert body["actor_id"] == "org-1"
    assert body["reason"] == "key material retired"


def test_get_revocation_response_is_compact_json_with_one_newline(client):
    _, created = _one_revocation(client)
    resp = client.get(_revocation_url(created["id"]))
    assert resp.status_code == 200
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    assert raw.index(b'"id"') < raw.index(b'"policy_id"')
    assert raw.index(b'"policy_id"') < raw.index(b'"actor_id"')
    assert raw.index(b'"actor_id"') < raw.index(b'"reason"')
    assert raw.index(b'"reason"') < raw.index(b'"created_at"')
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    assert raw == expected


def test_get_revocation_requires_no_authentication_headers(client):
    _, created = _one_revocation(client)
    # Reviewer reads are public: no X-PA/X-PT/X-PS headers are sent.
    resp = client.get(_revocation_url(created["id"]))
    assert resp.status_code == 200
    assert resp.json() == created


def test_get_revocation_is_byte_for_byte_stable_across_reads(client):
    _, created = _one_revocation(client)
    first = client.get(_revocation_url(created["id"]))
    for _ in range(3):
        again = client.get(_revocation_url(created["id"]))
        assert again.status_code == 200
        assert again.content == first.content


def test_reason_is_returned_verbatim(client):
    _world(client)
    policy = _create_policy(client)
    reason = "  撤回原因:policy moved to a new root 🔒\n"
    created = _revoke(client, policy["id"], reason=reason)
    resp = client.get(_revocation_url(created["id"]))
    assert resp.status_code == 200
    assert resp.json()["reason"] == reason


def test_unknown_revocation_id_is_404_not_found(client):
    _world(client)
    resp = client.get(_revocation_url("tpr_" + "0" * 64))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_get_revocation_never_reverse_resolves_other_ids(client):
    policy, created = _one_revocation(client)
    # A policy id, an actor id, or a reason never resolves a revocation.
    for other in (policy["id"], "org-1", created["reason"]):
        resp = client.get(_revocation_url(other))
        assert resp.status_code == 404, other
        assert resp.json()["error"]["code"] == "not_found"


def test_get_revocation_rejects_any_query_parameter_before_lookup(client):
    _, created = _one_revocation(client)
    for params in (
        {"actor_id": "org-1"},
        {"limit": "1"},
        {"unknown": "x"},
        {"id": ""},
    ):
        resp = client.get(_revocation_url(created["id"]), params=params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error", params
    # A bad parameter outranks even an unknown id: 422, never 404.
    resp = client.get(_revocation_url("tpr_" + "0" * 64), params={"x": "y"})
    assert resp.status_code == 422


def test_get_revocation_requires_an_empty_body(client):
    _, created = _one_revocation(client)
    for body in (b"{}", b" ", b"null"):
        resp = client.request(
            "GET", _revocation_url(created["id"]), content=body
        )
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error", body


def test_get_revocation_writes_nothing(client, db_session):
    _, created = _one_revocation(client)
    revocations_before = _revocation_count(db_session)
    audits_before = _audit_count(db_session)

    assert client.get(_revocation_url(created["id"])).status_code == 200
    assert client.get(_revocation_url("tpr_" + "0" * 64)).status_code == 404
    assert (
        client.get(_revocation_url(created["id"]), params={"x": "y"}).status_code
        == 422
    )
    assert (
        client.request(
            "GET", _revocation_url(created["id"]), content=b"{}"
        ).status_code
        == 422
    )

    assert _revocation_count(db_session) == revocations_before
    assert _audit_count(db_session) == audits_before


def test_item_route_put_patch_and_delete_are_405(client):
    _, created = _one_revocation(client)
    for method in (client.put, client.patch, client.delete):
        resp = method(_revocation_url(created["id"]))
        assert resp.status_code == 405
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Search: envelope and ordering --------------------------------------------


def test_empty_store_is_an_empty_collection(client):
    resp = _list(client)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_search_requires_no_authentication_headers(client):
    _world_revocations(client)
    # A plain GET with no X-PA/X-PT/X-PS credentials answers the search.
    resp = client.get(REVOCATIONS_PATH)
    assert resp.status_code == 200
    assert resp.json()["count"] == 5


def test_search_lists_every_revocation_in_stable_creation_order(client):
    revocations = _world_revocations(client)
    resp = _list(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert list(body) == ["items", "count", "next_cursor"]
    assert body["count"] == 5
    assert body["next_cursor"] is None
    assert [item["id"] for item in body["items"]] == [
        r["id"] for r in revocations
    ]
    for item in body["items"]:
        _assert_public_item(item)


def test_search_response_is_compact_json_with_one_newline(client):
    _world_revocations(client)
    resp = _list(client)
    assert resp.status_code == 200
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    assert raw.index(b'"items"') < raw.index(b'"count"')
    assert raw.index(b'"count"') < raw.index(b'"next_cursor"')
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    assert raw == expected


def test_creation_order_uses_the_persistent_seq_tiebreaker(client, db_session):
    revocations = _world_revocations(client)
    # Identical (and out-of-insertion) timestamps still order by insertion.
    same = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    _pin_times(db_session, revocations, [same, same, same, same, same])
    body = _list(client).json()
    assert [item["id"] for item in body["items"]] == [
        r["id"] for r in revocations
    ]


# --- Search: exact-match filters ------------------------------------------------


def test_filter_by_id_policy_id_actor_id_and_reason(client):
    revocations = _world_revocations(client)
    r1, r2, r3, r4, r5 = revocations

    body = _list(client, id=r3["id"]).json()
    assert [item["id"] for item in body["items"]] == [r3["id"]]
    assert body["count"] == 1

    body = _list(client, policy_id=r4["policy_id"]).json()
    assert [item["id"] for item in body["items"]] == [r4["id"]]
    assert body["count"] == 1

    body = _list(client, actor_id="org-2").json()
    assert [item["id"] for item in body["items"]] == [r2["id"]]
    assert body["count"] == 1

    body = _list(client, reason=REASON_A).json()
    assert [item["id"] for item in body["items"]] == [
        r1["id"],
        r3["id"],
        r5["id"],
    ]
    assert body["count"] == 3

    # Filters combine as logical AND.
    body = _list(client, actor_id="org-3", reason=REASON_A).json()
    assert [item["id"] for item in body["items"]] == [r3["id"]]
    body = _list(client, actor_id="org-3", reason=REASON_B).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


def test_filter_matching_is_exact_case_and_whitespace_sensitive(client):
    _world_revocations(client)
    for params in (
        {"reason": REASON_A.upper()},
        {"reason": "Compromised Root Key"},
        {"reason": f" {REASON_A}"},
        {"reason": f"{REASON_A} "},
        {"actor_id": "ORG-1"},
        {"actor_id": "org-1 "},
    ):
        resp = _list(client, **params)
        assert resp.status_code == 200, params
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_unknown_filter_values_are_empty_collections_never_404(client):
    _world_revocations(client)
    for params in (
        {"id": "tpr_" + "0" * 64},
        {"policy_id": "atp_" + "0" * 64},
        {"actor_id": "org-99"},
        {"reason": "never recorded"},
    ):
        resp = _list(client, **params)
        assert resp.status_code == 200, params
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


# --- Search: time bounds --------------------------------------------------------


def _pinned_world(client, db_session):
    revocations = _world_revocations(client)
    instants = [
        datetime(2026, 1, day, 0, 0, 0, tzinfo=timezone.utc)
        for day in (1, 2, 3, 4, 5)
    ]
    _pin_times(db_session, revocations, instants)
    return revocations


def test_from_and_to_are_inclusive_bounds(client, db_session):
    revocations = _pinned_world(client, db_session)
    body = _list(
        client,
        **{
            "from": "2026-01-02T00:00:00Z",
            "to": "2026-01-04T00:00:00Z",
        },
    ).json()
    assert [item["id"] for item in body["items"]] == [
        r["id"] for r in revocations[1:4]
    ]
    assert body["count"] == 3

    # Equal bounds select exactly that instant.
    body = _list(
        client,
        **{"from": "2026-01-03T00:00:00Z", "to": "2026-01-03T00:00:00Z"},
    ).json()
    assert [item["id"] for item in body["items"]] == [revocations[2]["id"]]

    # One-sided bounds are open on the other side.
    assert _list(client, **{"from": "2026-01-04T00:00:00Z"}).json()["count"] == 2
    assert _list(client, **{"to": "2026-01-02T00:00:00Z"}).json()["count"] == 2

    # "+00:00" and "Z" spell the same instant.
    assert (
        _list(client, **{"from": "2026-01-02T00:00:00+00:00"}).json()["count"]
        == 4
    )


def test_from_later_than_to_is_422(client, db_session):
    _pinned_world(client, db_session)
    resp = _list(
        client,
        **{"from": "2026-01-04T00:00:00Z", "to": "2026-01-02T00:00:00Z"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_time_values_are_422(client):
    _world_revocations(client)
    for value in (
        "",
        "   ",
        "2026-01-01",
        "2026-01-01T00:00:00",
        "2026-01-01T00:00:00+01:00",
        "2026-01-01t00:00:00z",
        "2026-13-01T00:00:00Z",
        "not-a-time",
    ):
        for field in ("from", "to"):
            resp = _list(client, **{field: value})
            assert resp.status_code == 422, (field, value)
            assert resp.json()["error"]["code"] == "validation_error", (
                field,
                value,
            )


# --- Search: limit ----------------------------------------------------------------


def test_limit_defaults_to_50_and_bounds_the_page(client):
    _world_revocations(client)
    assert len(_list(client).json()["items"]) == 5
    body = _list(client, limit="2").json()
    assert len(body["items"]) == 2
    assert body["count"] == 5
    assert body["next_cursor"] is not None
    # Boundary limits are accepted.
    assert _list(client, limit="1").status_code == 200
    assert _list(client, limit="100").status_code == 200


def test_invalid_limits_are_422(client):
    _world_revocations(client)
    for value in (
        "",
        " ",
        "0",
        "101",
        "-1",
        "+5",
        " 5",
        "5 ",
        "1.5",
        "1e2",
        "abc",
        "0x10",
    ):
        resp = _list(client, limit=value)
        assert resp.status_code == 422, repr(value)
        assert resp.json()["error"]["code"] == "validation_error", repr(value)


# --- Search: parameter shape ------------------------------------------------------


def test_unknown_blank_and_repeated_parameters_are_422(client):
    _world_revocations(client)
    # Undeclared parameters are rejected rather than ignored.
    for params in (
        {"policy": "x"},
        {"actor": "org-1"},
        {"offset": "1"},
        {"page": "1"},
    ):
        resp = _list(client, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error", params

    # Blank filters are invalid rather than matched against nothing.
    for field in ("id", "policy_id", "actor_id", "reason"):
        for value in ("", "   "):
            resp = _list(client, **{field: value})
            assert resp.status_code == 422, (field, value)

    # A repeated scalar is rejected instead of silently taking the last value.
    for query in (
        "limit=1&limit=2",
        "reason=a&reason=b",
        "actor_id=org-1&actor_id=org-2",
        "cursor=a&cursor=b",
        "from=2026-01-01T00:00:00Z&from=2026-01-02T00:00:00Z",
    ):
        resp = client.get(f"{REVOCATIONS_PATH}?{query}")
        assert resp.status_code == 422, query
        assert resp.json()["error"]["code"] == "validation_error", query


def test_search_requires_an_empty_body(client):
    _world_revocations(client)
    for body in (b"{}", b" ", b"null", b"not json"):
        resp = client.request("GET", REVOCATIONS_PATH, content=body)
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error", body


# --- Search: pagination -----------------------------------------------------------


def test_cursor_pagination_walks_every_page_without_duplication(client):
    revocations = _world_revocations(client)
    all_items, pages, count = _walk_pages(client, limit="2")
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    assert [item["id"] for item in all_items] == [r["id"] for r in revocations]
    assert len({item["id"] for item in all_items}) == 5

    # The final page carries a null cursor; every page keeps the total count.
    cursor = None
    for expected in (2, 2, 1):
        query = {"limit": "2"}
        if cursor is not None:
            query["cursor"] = cursor
        body = _list(client, **query).json()
        assert body["count"] == 5
        assert len(body["items"]) == expected
        cursor = body["next_cursor"]
    assert cursor is None


def test_filtered_search_paginates_within_the_filter(client):
    revocations = _world_revocations(client)
    all_items, pages, count = _walk_pages(client, reason=REASON_A, limit="1")
    assert count == 3
    assert [len(page) for page in pages] == [1, 1, 1]
    assert [item["id"] for item in all_items] == [
        revocations[0]["id"],
        revocations[2]["id"],
        revocations[4]["id"],
    ]


def test_cursor_at_or_past_the_tail_returns_empty_page_with_count(client, app):
    _world_revocations(client)
    claims = {
        "id": None,
        "policy_id": None,
        "actor_id": None,
        "reason": None,
        "from": None,
        "to": None,
        "limit": 50,
    }
    tail = pagination.encode_typed_cursor(
        app.state.trust_policy_revocations_cursor_secret,
        pagination.TRUST_POLICY_REVOCATIONS_CURSOR,
        {**claims, "offset": 5},
    )
    resp = _list(client, cursor=tail)
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 5, "next_cursor": None}

    past = pagination.encode_typed_cursor(
        app.state.trust_policy_revocations_cursor_secret,
        pagination.TRUST_POLICY_REVOCATIONS_CURSOR,
        {**claims, "offset": 99},
    )
    resp = _list(client, cursor=past)
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 5, "next_cursor": None}


def test_cursor_binds_every_effective_filter_and_limit(client):
    revocations = _world_revocations(client)
    cursor = _list(client, limit="2").json()["next_cursor"]
    assert cursor is not None
    # A different limit, a new filter, or a dropped filter all mismatch.
    for params in (
        {"limit": "3", "cursor": cursor},
        {"cursor": cursor},
        {"actor_id": "org-1", "limit": "2", "cursor": cursor},
        {"reason": REASON_A, "limit": "2", "cursor": cursor},
        {"id": revocations[0]["id"], "limit": "2", "cursor": cursor},
        {
            "policy_id": revocations[0]["policy_id"],
            "limit": "2",
            "cursor": cursor,
        },
        {"from": "2026-01-01T00:00:00Z", "limit": "2", "cursor": cursor},
        {"to": "2030-01-01T00:00:00Z", "limit": "2", "cursor": cursor},
    ):
        resp = _list(client, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"

    # A cursor minted under one exact filter cannot resume another.
    a_cursor = _list(client, reason=REASON_A, limit="1").json()["next_cursor"]
    assert a_cursor is not None
    assert (
        _list(client, reason=REASON_B, limit="1", cursor=a_cursor).status_code
        == 422
    )
    actor_cursor = _list(client, actor_id="org-1", limit="1").json()[
        "next_cursor"
    ]
    # org-1 has a single revocation, so no continuation cursor exists; use a
    # filtered cursor that does have one instead.
    if actor_cursor is not None:
        assert (
            _list(
                client, actor_id="org-2", limit="1", cursor=actor_cursor
            ).status_code
            == 422
        )

    # A time-bound cursor is bound to the exact instant; equivalent UTC
    # notation resumes, a different second does not.
    time_cursor = _list(
        client, **{"from": "2000-01-01T00:00:00Z", "limit": "1"}
    ).json()["next_cursor"]
    assert time_cursor is not None
    same_bound = _list(
        client,
        **{
            "from": "2000-01-01T00:00:00+00:00",
            "limit": "1",
            "cursor": time_cursor,
        },
    )
    assert same_bound.status_code == 200
    changed_bound = _list(
        client,
        **{
            "from": "2000-01-01T00:00:01Z",
            "limit": "1",
            "cursor": time_cursor,
        },
    )
    assert changed_bound.status_code == 422


def test_blank_malformed_and_tampered_cursors_are_422(client):
    _world_revocations(client)
    valid = _list(client, limit="1").json()["next_cursor"]
    assert valid is not None
    tampered = valid[:-1] + ("A" if valid[-1] != "A" else "B")
    for bad in ("", "   ", "not-a-cursor", "tr1", "tr1.abc", tampered):
        resp = _list(client, cursor=bad)
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_foreign_family_cursor_is_422(client, app):
    _world_revocations(client)
    # A well-formed cursor minted by another endpoint family -- including the
    # closely named trust-policy retrieval -- never resumes this search.
    trust_policies = pagination.encode_typed_cursor(
        app.state.trust_policies_cursor_secret,
        pagination.TRUST_POLICIES_CURSOR,
        {"actor_id": None, "limit": 50, "offset": 1},
    )
    assert _list(client, cursor=trust_policies).status_code == 422

    grant_revocations = pagination.encode_typed_cursor(
        app.state.attestation_access_grant_revocations_cursor_secret,
        pagination.ATTESTATION_ACCESS_GRANT_REVOCATIONS_CURSOR,
        {
            "grant_id": None,
            "revoker_actor_id": None,
            "reason": None,
            "from": None,
            "to": None,
            "limit": 50,
            "offset": 1,
        },
    )
    assert _list(client, cursor=grant_revocations).status_code == 422


def test_cursor_with_wrong_claim_set_is_422(client, app):
    _world_revocations(client)
    # Correct family marker, HMAC, and base64, but a payload missing a bound
    # claim is rejected rather than trusted.
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "id": None,
                "policy_id": None,
                "actor_id": None,
                "reason": None,
                "limit": 50,
                "offset": 1,
            },
            separators=(",", ":"),
        ).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(
        hmac.new(
            app.state.trust_policy_revocations_cursor_secret,
            f"tr1.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    resp = _list(client, cursor=f"tr1.{payload}.{sig}")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Search: read-only and method boundary ----------------------------------------


def test_search_writes_nothing(client, db_session):
    _world_revocations(client)
    revocations_before = _revocation_count(db_session)
    audits_before = _audit_count(db_session)

    assert _list(client).status_code == 200
    assert _list(client, reason=REASON_A, limit="1").status_code == 200
    assert _list(client, reason="never recorded").status_code == 200
    assert _list(client, limit="0").status_code == 422
    assert _list(client, unknown="x").status_code == 422
    assert client.request("GET", REVOCATIONS_PATH, content=b"{}").status_code == 422

    assert _revocation_count(db_session) == revocations_before
    assert _audit_count(db_session) == audits_before


def test_collection_put_patch_and_delete_are_405(client):
    _world_revocations(client)
    for method in (client.put, client.patch, client.delete):
        resp = method(REVOCATIONS_PATH)
        assert resp.status_code == 405
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Persistence across restart ---------------------------------------------------


def test_reads_survive_restart_but_cursors_do_not(tmp_db_url):
    app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app) as client:
        revocations = _world_revocations(client)
        cursor = _list(client, limit="2").json()["next_cursor"]
        assert cursor is not None

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        # The records persist: the individual read and the search are
        # unchanged across the restart.
        first = revocations[0]
        resp = client.get(_revocation_url(first["id"]))
        assert resp.status_code == 200
        assert resp.json() == first

        body = _list(client).json()
        assert body["count"] == 5
        assert [item["id"] for item in body["items"]] == [
            r["id"] for r in revocations
        ]

        # The cursor secret rotated with the process: the old cursor is a
        # 422, not a silently wrong page.
        resp = _list(client, limit="2", cursor=cursor)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"
