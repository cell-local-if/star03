"""Tests for the read-only trust-policy retrieval.

Covers ``GET /v1/trust-policies``: the response is compact UTF-8 JSON with
members in ``items``, ``count``, ``next_cursor`` order terminated by exactly
one newline; each item is the existing public policy view (identifier,
subject, threshold, enabled flag, UTC creation timestamp — never a private
key, raw signature, claim payload, or content/evidence bytes) and policies
follow stable creation order. The optional ``actor_id`` filter is a
non-empty, case- and whitespace-sensitive exact match (absent means
unfiltered; an unknown subject is an empty collection, never a 404).
``limit`` is a pure decimal integer from 1 to 100 defaulting to 50; the
opaque HMAC cursor binds the effective ``actor_id`` and ``limit`` so pages
concatenate without gaps or duplicates, a page at or past the tail returns
empty ``items`` with the original ``count``, and the final cursor is null.
A non-empty request body, blank/illegal/repeated/undeclared parameters, and
blank/malformed/tampered/foreign-family/query-mismatching cursors are all
``422 validation_error``; neither queries nor failures write any policy,
resource, or audit row. All fixtures are deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime

from sqlalchemy import func, select

from provenance import pagination
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import ActorTrustPolicy, AuditEvent
from tests.helpers import create_actor
from tests.test_trust_policies import (
    POLICIES_PATH,
    _make_attestation,
    _make_claim,
    _post_policy,
    _policy_body,
)

URL = POLICIES_PATH

POLICY_KEYS = {"id", "actor_id", "threshold", "enabled", "created_at"}


def _list(client, **params):
    return client.get(URL, params=params)


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


def _seed(index: int) -> bytes:
    return hashlib.sha256(b"trust-policy-list-%d" % index).digest()


def _make_policy(client, actor_id: str, seed: bytes, threshold: int = 2):
    """Create an actor with a current key and register its policy."""
    create_actor(client, actor_id=actor_id, name=f"Org {actor_id}",
                 type="organization")
    digest = hashlib.sha256(b"content-%s" % actor_id.encode()).hexdigest()
    claim = _make_claim(client, actor_id, digest=digest)
    _make_attestation(client, actor_id, seed, claim["id"])
    resp = _post_policy(
        client, _policy_body(actor_id, threshold), actor=actor_id, seed=seed
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make_policies(client, count: int):
    """Register ``count`` policies for distinct subjects, in creation order."""
    return [
        _make_policy(client, f"org-{i}", _seed(i), threshold=(i % 100) + 1)
        for i in range(1, count + 1)
    ]


# --- Response shape, fields, and ordering -------------------------------------


def test_empty_database_is_an_empty_collection(client):
    resp = _list(client)
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}
    assert resp.content == b'{"items":[],"count":0,"next_cursor":null}\n'


def test_response_is_compact_json_with_ordered_members_and_one_newline(client):
    _make_policies(client, 2)
    resp = _list(client)
    assert resp.status_code == 200
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    # Members appear exactly in items, count, next_cursor order.
    assert list(resp.json()) == ["items", "count", "next_cursor"]
    assert raw.find(b'"items"') < raw.find(b'"count"') < raw.find(
        b'"next_cursor"'
    )
    # Compact separators: no insignificant whitespace anywhere.
    assert b", " not in raw
    assert b": " not in raw
    # Numbers render as plain integers, never floats or strings.
    assert b'"count":2,' in raw
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    assert raw == expected


def test_item_is_the_public_policy_view_and_matches_the_registration(client):
    policies = _make_policies(client, 2)
    body = _list(client).json()
    assert len(body["items"]) == 2
    for item, policy in zip(body["items"], policies):
        assert set(item) == POLICY_KEYS
        assert item == policy
        assert item["id"].startswith("atp_")
        assert item["enabled"] is True
        assert isinstance(item["threshold"], int)


def test_created_at_is_utc(client):
    _make_policies(client, 1)
    (item,) = _list(client).json()["items"]
    created_at = datetime.fromisoformat(item["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert item["created_at"].endswith(("Z", "+00:00"))


def test_policies_follow_stable_creation_order(client):
    policies = _make_policies(client, 5)
    body = _list(client).json()
    assert [i["id"] for i in body["items"]] == [p["id"] for p in policies]
    assert [i["actor_id"] for i in body["items"]] == [
        f"org-{i}" for i in range(1, 6)
    ]


def test_repeat_registration_adds_no_search_position(client):
    policy = _make_policy(client, "org-1", _seed(1), threshold=2)
    # A same-threshold retry is idempotent: no new row, no new list entry.
    retry = _post_policy(
        client, _policy_body("org-1", 2), actor="org-1", seed=_seed(1)
    )
    assert retry.status_code == 200
    body = _list(client).json()
    assert body["count"] == 1
    assert [i["id"] for i in body["items"]] == [policy["id"]]


# --- Filtering ------------------------------------------------------------------


def test_actor_id_exact_match(client):
    _make_policies(client, 3)
    body = _list(client, actor_id="org-2").json()
    assert body["count"] == 1
    assert [i["actor_id"] for i in body["items"]] == ["org-2"]
    assert body["next_cursor"] is None


def test_actor_id_match_is_case_and_whitespace_sensitive(client):
    _make_policy(client, "org-1", _seed(1))
    for value in ("Org-1", "ORG-1", "org-1 ", " org-1", "org-1\t"):
        body = _list(client, actor_id=value).json()
        assert body["items"] == [], value
        assert body["count"] == 0, value
        assert body["next_cursor"] is None, value


def test_unknown_subject_filter_is_empty_collection_not_not_found(client):
    _make_policies(client, 2)
    resp = _list(client, actor_id="ghost")
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_search_does_not_require_actor_existence(client):
    # No actors, no policies: a filtered read is still a 200 empty page.
    assert _list(client, actor_id="ghost").status_code == 200


# --- Pagination -----------------------------------------------------------------


def test_pagination_concatenates_without_gaps_or_duplicates(client):
    policies = _make_policies(client, 5)
    all_items, pages, count = _walk_pages(client, limit=2)
    # 5 policies -> pages of 2, 2, 1.
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    ids = [i["id"] for i in all_items]
    assert len(ids) == len(set(ids)) == 5
    assert ids == [p["id"] for p in policies]
    assert all_items == _list(client).json()["items"]


def test_count_is_filtered_total_on_every_page(client):
    _make_policies(client, 4)
    all_items, pages, count = _walk_pages(client, actor_id="org-3", limit=1)
    assert count == 1
    assert [len(page) for page in pages] == [1]
    assert [i["actor_id"] for i in all_items] == ["org-3"]


def test_last_page_cursor_null_on_exact_division(client):
    _make_policies(client, 4)
    first = _list(client, limit=2).json()
    assert first["next_cursor"] is not None
    second = _list(client, limit=2, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 2
    assert second["count"] == 4
    assert second["next_cursor"] is None


def test_default_limit_is_fifty(client):
    _make_policies(client, 55)
    first = _list(client).json()
    assert len(first["items"]) == 50
    assert first["count"] == 55
    assert first["next_cursor"] is not None
    second = _list(client, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 5
    assert second["count"] == 55
    assert second["next_cursor"] is None


def test_limit_boundaries_accepted(client):
    _make_policies(client, 2)
    for value in (1, 100):
        assert _list(client, limit=value).status_code == 200


def test_reusing_a_cursor_replays_the_same_page(client):
    _make_policies(client, 4)
    cursor = _list(client, limit=2).json()["next_cursor"]
    replay_one = _list(client, limit=2, cursor=cursor).json()
    replay_two = _list(client, limit=2, cursor=cursor).json()
    assert replay_one == replay_two


def test_cursor_past_end_returns_empty_page_with_total_count(client, app):
    _make_policies(client, 3)
    token = pagination.encode_typed_cursor(
        app.state.trust_policies_cursor_secret,
        pagination.TRUST_POLICIES_CURSOR,
        {"actor_id": None, "limit": 50, "offset": 99},
    )
    body = _list(client, cursor=token).json()
    assert body["items"] == []
    assert body["count"] == 3
    assert body["next_cursor"] is None


# --- Cursor integrity ------------------------------------------------------------


def test_tampered_or_malformed_cursors_are_validation_errors(client):
    _make_policies(client, 2)
    good = _list(client, limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    foreign = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.TRUST_POLICIES_CURSOR,
        {"actor_id": None, "limit": 1, "offset": 1},
    )
    for token in (
        "",
        "   ",
        "not-a-cursor",
        "tp1.onlytwoparts",
        "tp1.too.many.parts",
        "tp0.x.y",
        "tp2.x.y",
        "v1.x.y",
        "cl1.x.y",
        "ae1.x.y",
        tampered,
        foreign,
    ):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422, repr(token)
        assert resp.json()["error"]["code"] == "validation_error"
        assert "items" not in resp.json()


def test_cursor_from_other_families_is_rejected(client, app):
    _make_policies(client, 2)
    claims_cursor = pagination.encode_typed_cursor(
        app.state.claims_cursor_secret,
        pagination.CLAIMS_CURSOR,
        {
            "content_id": None,
            "actor_id": None,
            "claim_type": None,
            "payload_digest_hex": None,
            "limit": 1,
            "offset": 1,
        },
    )
    audit_cursor = pagination.encode_typed_cursor(
        app.state.audit_events_cursor_secret,
        pagination.AUDIT_EVENTS_CURSOR,
        {
            "event_type": None,
            "resource_id": None,
            "from": None,
            "to": None,
            "limit": 1,
            "offset": 1,
        },
    )
    for token in (claims_cursor, audit_cursor):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


def test_trust_policies_cursor_is_rejected_by_other_endpoints(client, app):
    _make_policies(client, 2)
    cursor = _list(client, limit=1).json()["next_cursor"]

    # The claims search uses its own family and claim set.
    resp = client.get("/v1/claims", params={"limit": 1, "cursor": cursor})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    # As does the audit-event search.
    resp = client.get(
        "/v1/audit-events", params={"limit": 1, "cursor": cursor}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_bound_to_actor_id_and_limit(client):
    _make_policies(client, 4)

    cursor = _list(client, limit=2).json()["next_cursor"]
    # A filter/limit present in the resume request but absent from (or
    # different in) the cursor mismatches.
    for params in ({"limit": 3}, {"actor_id": "org-1"}):
        resp = _list(client, cursor=cursor, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"

    # A cursor minted under a filter cannot resume without it or under a
    # different filter value.
    filtered = _list(client, actor_id="org-1", limit=1).json()
    assert filtered["count"] == 1
    unfiltered_cursor = _list(client, limit=1).json()["next_cursor"]
    assert unfiltered_cursor is not None
    resp = _list(client, cursor=unfiltered_cursor, actor_id="org-2", limit=1)
    assert resp.status_code == 422


def test_cursor_secret_rotation_invalidates_outstanding_cursors(
    tmp_db_url, file_app, file_client
):
    _make_policies(file_client, 3)
    token = _list(file_client, limit=1).json()["next_cursor"]
    assert token is not None

    restarted = create_app(Settings(database_url=tmp_db_url))
    from fastapi.testclient import TestClient

    with TestClient(restarted) as second_client:
        stale = _list(second_client, limit=1, cursor=token)
        assert stale.status_code == 422
        # The policies themselves survive the restart and still list.
        fresh = _list(second_client).json()
        assert fresh["count"] == 3
        assert len(fresh["items"]) == 3


# --- Parameter validation --------------------------------------------------------


def test_blank_actor_id_is_a_validation_error(client):
    _make_policies(client, 1)
    for value in ("", "   ", "\t"):
        resp = _list(client, actor_id=value)
        assert resp.status_code == 422, repr(value)
        assert resp.json()["error"]["code"] == "validation_error"


def test_illegal_limit_values_are_validation_errors(client):
    _make_policies(client, 1)
    for value in ("0", "101", "-1", "1.5", "abc", "8.0", "  2", "+1", ""):
        resp = _list(client, limit=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_validation_errors(client):
    _make_policies(client, 1)
    for suffix in (
        "actor_id=a&actor_id=b",
        "limit=1&limit=2",
        "cursor=x&cursor=y",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameters_are_validation_errors(client):
    _make_policies(client, 1)
    for suffix in (
        "actor=org-1",
        "threshold=2",
        "enabled=true",
        "actor_ids=org-1",
        "limit=1&offset=2",
        "CURSOR=x",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_cursor_with_otherwise_valid_params_is_422(client):
    _make_policies(client, 1)
    resp = _list(client, actor_id="org-1", limit=10, cursor="garbage")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_validation_failure_locates_the_query_field(client):
    _make_policies(client, 1)
    resp = _list(client, limit="nope")
    assert resp.status_code == 422
    issue = resp.json()["error"]["details"]["issues"][0]
    assert issue["loc"] == ["query", "limit"]


# --- Request body ----------------------------------------------------------------


def test_non_empty_body_is_rejected_before_any_parameter_is_read(client):
    _make_policies(client, 1)
    bodies = (
        b"{}",
        b'{"actor_id": "org-1"}',
        b'{"actor_id": "a", "actor_id": "b"}',
        b"{not valid json",
        b" ",
    )
    for body in bodies:
        # Even an otherwise valid query is rejected when a body is present...
        resp = client.request("GET", URL, content=body)
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"
        assert (
            resp.json()["error"]["details"]["reason"] == "body_must_be_empty"
        )
        # ...and the body check precedes parameter validation.
        resp = client.request("GET", f"{URL}?limit=abc", content=body)
        assert resp.status_code == 422, body
        assert (
            resp.json()["error"]["details"]["reason"] == "body_must_be_empty"
        )


# --- Read-only guarantee ----------------------------------------------------------


def test_queries_and_failures_write_no_policies_or_audit_events(
    client, db_session
):
    _make_policies(client, 3)

    def counts():
        return (
            db_session.scalar(
                select(func.count()).select_from(ActorTrustPolicy)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    policies_before, events_before = counts()
    assert policies_before == 3

    # Successful unfiltered, filtered, and paginated reads.
    cursor = None
    for _ in range(6):
        params = {"limit": 1}
        if cursor is not None:
            params["cursor"] = cursor
        resp = _list(client, **params)
        assert resp.status_code == 200, resp.text
        cursor = resp.json()["next_cursor"]
        if cursor is None:
            break
    _list(client)
    _list(client, actor_id="org-2")
    _list(client, actor_id="ghost")

    # Failed reads must not write anything either.
    _list(client, actor_id=" ")
    _list(client, limit=0)
    _list(client, limit=101)
    _list(client, actor_id="a", cursor="garbage")
    client.get(f"{URL}?actor_id=a&actor_id=b")
    client.get(f"{URL}?unknown=1")
    client.request("GET", URL, content=b"{}")

    assert counts() == (policies_before, events_before)


def test_repeated_reads_and_empty_results_leave_no_state_change(
    client, db_session
):
    def counts():
        return (
            db_session.scalar(
                select(func.count()).select_from(ActorTrustPolicy)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    before = counts()
    for _ in range(3):
        assert _list(client).status_code == 200
        assert _list(client, actor_id="ghost").json()["count"] == 0
    assert counts() == before


# --- Existing routes stay compatible ----------------------------------------------


def test_existing_registration_route_remains_unchanged(client):
    policy = _make_policy(client, "org-1", _seed(1), threshold=3)
    assert policy["actor_id"] == "org-1"
    assert policy["threshold"] == 3
    # Idempotent retry still returns the original record with 200.
    retry = _post_policy(
        client, _policy_body("org-1", 3), actor="org-1", seed=_seed(1)
    )
    assert retry.status_code == 200
    assert retry.json() == policy
    # A different threshold for the same subject is still a 409.
    conflict = _post_policy(
        client, _policy_body("org-1", 4), actor="org-1", seed=_seed(1)
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "actor_trust_policy_conflict"
