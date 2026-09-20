"""Tests for the read-only audit-event search endpoint.

Covers GET /v1/audit-events: items expose exactly
``{event_type, resource_id, created_at}`` in stable creation order with
timezone-aware UTC timestamps; ``event_type``/``resource_id`` are non-empty
exact, combinable filters; ``from``/``to`` are inclusive RFC 3339 UTC bounds
with ``from <= to``; ``limit`` is 1..100 defaulting to 50; pages concatenate
without gaps or duplicates while ``count`` stays the filtered total and the
final cursor is null; cursors are opaque/HMAC-bound to every effective
filter and limit (tampered, foreign-family, or mismatched cursors are 422);
blank/illegal/repeated/undeclared parameters are 422; no match is an empty
collection; and neither successful nor failed queries write any resource or
audit rows. All fixtures are deterministic and offline.
"""

from __future__ import annotations

import base64
import hashlib
import hmac as hmac_mod
import json
import secrets
from datetime import datetime, timezone

from sqlalchemy import func, select

from provenance import pagination
from provenance.models import Actor, AuditEvent
from tests.helpers import DIGEST_A, content_payload, create_actor


def _list(client, **params):
    return client.get("/v1/audit-events", params=params)


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


def _insert_event(db_session, event_type, resource_id, created_at):
    event = AuditEvent(
        event_type=event_type, resource_id=resource_id, created_at=created_at
    )
    db_session.add(event)
    db_session.commit()
    return event


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def _seed_timeline(db_session):
    """Five events at fixed instants; e3/e4 share a timestamp (seq breaks)."""
    rows = [
        ("actor.created", "res-1", "2026-01-01T00:00:00Z"),
        ("content.created", "res-2", "2026-01-02T00:00:00Z"),
        ("claim.created", "res-3", "2026-01-03T00:00:00Z"),
        ("actor.created", "res-4", "2026-01-03T00:00:00Z"),
        ("content.created", "res-5", "2026-01-04T00:00:00Z"),
    ]
    return [
        _insert_event(db_session, event_type, resource_id, _utc(ts))
        for event_type, resource_id, ts in rows
    ]


#: The exact public audit-event field set.
PUBLIC_FIELDS = {"event_type", "resource_id", "created_at"}


# --- Shape, field set, ordering, UTC rendering --------------------------------


def test_response_shape_and_public_fields_only(client):
    create_actor(client)
    body = _list(client).json()
    assert set(body) == {"items", "count", "next_cursor"}
    assert body["count"] == 1
    assert body["next_cursor"] is None
    (item,) = body["items"]
    assert set(item) == PUBLIC_FIELDS
    assert item == {
        "event_type": "actor.created",
        "resource_id": "org-1",
        "created_at": item["created_at"],
    }


def test_created_at_keeps_utc_timezone(client):
    create_actor(client)
    (item,) = _list(client).json()["items"]
    parsed = datetime.fromisoformat(item["created_at"])
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_stable_creation_order_across_event_types(client):
    create_actor(client)
    created = client.post("/v1/contents", json=content_payload()).json()
    claim = client.post(
        "/v1/claims",
        json={
            "content_id": created["id"],
            "actor_id": "org-1",
            "claim_type": "authorship",
            "payload": {"statement": "x"},
        },
    ).json()

    body = _list(client).json()
    assert body["count"] == 3
    assert [item["event_type"] for item in body["items"]] == [
        "actor.created",
        "content.created",
        "claim.created",
    ]
    assert [item["resource_id"] for item in body["items"]] == [
        "org-1",
        created["id"],
        claim["id"],
    ]


def test_same_timestamp_events_follow_insertion_order(client, db_session):
    _seed_timeline(db_session)
    body = _list(client).json()
    assert [item["resource_id"] for item in body["items"]] == [
        "res-1",
        "res-2",
        "res-3",
        "res-4",
        "res-5",
    ]


# --- Exact-match filtering ------------------------------------------------------


def test_event_type_exact_match(client, db_session):
    _seed_timeline(db_session)
    body = _list(client, event_type="actor.created").json()
    assert [item["resource_id"] for item in body["items"]] == ["res-1", "res-4"]
    assert body["count"] == 2
    assert body["next_cursor"] is None


def test_resource_id_exact_match(client, db_session):
    _seed_timeline(db_session)
    body = _list(client, resource_id="res-3").json()
    assert [item["event_type"] for item in body["items"]] == ["claim.created"]
    assert body["count"] == 1


def test_filters_combine_as_logical_and(client, db_session):
    _seed_timeline(db_session)
    body = _list(
        client, event_type="actor.created", resource_id="res-4"
    ).json()
    assert [item["resource_id"] for item in body["items"]] == ["res-4"]
    assert body["count"] == 1

    # Same type, different resource: no intersection.
    body = _list(
        client, event_type="actor.created", resource_id="res-2"
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


def test_exact_match_is_case_and_whitespace_sensitive(client, db_session):
    _seed_timeline(db_session)
    for params in (
        {"event_type": "ACTOR.CREATED"},
        {"event_type": "actor.created "},
        {"event_type": " actor.created"},
        {"resource_id": "RES-1"},
        {"resource_id": "res-1 "},
    ):
        body = _list(client, **params).json()
        assert body == {"items": [], "count": 0, "next_cursor": None}, params


def test_no_match_returns_empty_collection(client, db_session):
    _seed_timeline(db_session)
    body = _list(client, event_type="no.such_event").json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


# --- Time-range filtering -------------------------------------------------------


def test_from_and_to_bounds_are_inclusive(client, db_session):
    _seed_timeline(db_session)
    body = _list(
        client,
        **{
            "from": "2026-01-02T00:00:00Z",
            "to": "2026-01-03T00:00:00Z",
        },
    ).json()
    assert [item["resource_id"] for item in body["items"]] == [
        "res-2",
        "res-3",
        "res-4",
    ]
    assert body["count"] == 3


def test_from_equal_to_selects_exact_instant(client, db_session):
    _seed_timeline(db_session)
    body = _list(
        client,
        **{
            "from": "2026-01-03T00:00:00Z",
            "to": "2026-01-03T00:00:00Z",
        },
    ).json()
    assert [item["resource_id"] for item in body["items"]] == ["res-3", "res-4"]
    assert body["count"] == 2


def test_from_and_to_combine_with_exact_filters(client, db_session):
    _seed_timeline(db_session)
    body = _list(
        client,
        event_type="actor.created",
        **{
            "from": "2026-01-02T00:00:00Z",
            "to": "2026-01-04T00:00:00Z",
        },
    ).json()
    assert [item["resource_id"] for item in body["items"]] == ["res-4"]
    assert body["count"] == 1


def test_zero_offset_and_fractional_forms_are_accepted(client, db_session):
    _seed_timeline(db_session)
    body = _list(
        client,
        **{
            "from": "2026-01-02T00:00:00+00:00",
            "to": "2026-01-03T00:00:00.000Z",
        },
    ).json()
    assert [item["resource_id"] for item in body["items"]] == [
        "res-2",
        "res-3",
        "res-4",
    ]


def test_from_later_than_to_is_validation_error(client, db_session):
    _seed_timeline(db_session)
    resp = _list(
        client,
        **{
            "from": "2026-01-04T00:00:00Z",
            "to": "2026-01-03T00:00:00Z",
        },
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_malformed_timestamps_are_validation_errors(client, db_session):
    _seed_timeline(db_session)
    for field in ("from", "to"):
        for value in (
            "",
            "   ",
            "2026-01-02",                      # date only
            "2026-01-02 03:04:05Z",            # space separator
            "2026-01-02T03:04Z",               # missing seconds
            "2026-01-02T03:04:05",             # naive, no designator
            "2026-01-02T03:04:05+05:00",       # non-UTC offset
            "2026-01-02T03:04:05-00:00",       # unknown-offset convention
            "2026-13-02T03:04:05Z",            # not a real month
            "2026-01-02T25:04:05Z",            # not a real hour
            "not-a-timestamp",
        ):
            resp = _list(client, **{field: value})
            assert resp.status_code == 422, (field, value)
            assert resp.json()["error"]["code"] == "validation_error"


# --- Pagination -----------------------------------------------------------------


def _seed_events(db_session, count):
    for i in range(count):
        _insert_event(
            db_session,
            "actor.created",
            f"res-{i:03d}",
            _utc(f"2026-02-01T00:00:{i % 60:02d}Z"),
        )


def test_pagination_concatenates_without_gaps_or_duplicates(client, db_session):
    _seed_events(db_session, 12)
    all_items, pages, count = _walk_pages(client, limit=3)
    assert [len(page) for page in pages] == [3, 3, 3, 3]
    assert count == 12
    ids = [item["resource_id"] for item in all_items]
    assert len(ids) == len(set(ids)) == 12
    assert ids == [f"res-{i:03d}" for i in range(12)]


def test_count_is_filtered_total_on_every_page(client, db_session):
    _seed_events(db_session, 7)
    all_items, pages, count = _walk_pages(
        client, event_type="actor.created", limit=3
    )
    assert count == 7
    assert [len(page) for page in pages] == [3, 3, 1]
    assert len(all_items) == 7


def test_last_page_cursor_null_on_exact_division(client, db_session):
    _seed_events(db_session, 4)
    first = _list(client, limit=2).json()
    assert first["next_cursor"] is not None
    second = _list(client, limit=2, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 2
    assert second["count"] == 4
    assert second["next_cursor"] is None


def test_default_limit_is_fifty(client, db_session):
    _seed_events(db_session, 55)
    first = _list(client).json()
    assert len(first["items"]) == 50
    assert first["count"] == 55
    assert first["next_cursor"] is not None
    second = _list(client, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 5
    assert second["count"] == 55
    assert second["next_cursor"] is None


def test_limit_boundaries_accepted(client, db_session):
    _seed_events(db_session, 2)
    for value in (1, 100):
        assert _list(client, limit=value).status_code == 200


def test_pagination_is_stable_and_deterministic(client, db_session):
    _seed_events(db_session, 10)
    first_items, _, _ = _walk_pages(client, limit=4)
    second_items, _, _ = _walk_pages(client, limit=4)
    assert first_items == second_items


def test_reusing_a_cursor_replays_the_same_page(client, db_session):
    _seed_events(db_session, 6)
    cursor = _list(client, limit=2).json()["next_cursor"]
    replay_one = _list(client, limit=2, cursor=cursor).json()
    replay_two = _list(client, limit=2, cursor=cursor).json()
    assert replay_one == replay_two
    assert [item["resource_id"] for item in replay_one["items"]] == [
        "res-002",
        "res-003",
    ]


def test_filtered_pagination_binds_the_filter(client, db_session):
    _seed_timeline(db_session)
    all_items, pages, count = _walk_pages(
        client, event_type="content.created", limit=1
    )
    assert count == 2
    assert [len(page) for page in pages] == [1, 1]
    assert [item["resource_id"] for item in all_items] == ["res-2", "res-5"]


def test_cursor_past_end_returns_empty_page_with_total_count(client, app, db_session):
    _seed_events(db_session, 3)
    token = pagination.encode_typed_cursor(
        app.state.audit_events_cursor_secret,
        pagination.AUDIT_EVENTS_CURSOR,
        {
            "event_type": None,
            "resource_id": None,
            "from": None,
            "to": None,
            "limit": 50,
            "offset": 99,
        },
    )
    body = _list(client, cursor=token).json()
    assert body["items"] == []
    assert body["count"] == 3
    assert body["next_cursor"] is None


# --- Cursor integrity -----------------------------------------------------------


def test_tampered_or_malformed_cursors_are_validation_errors(client, app, db_session):
    _seed_events(db_session, 3)
    good = _list(client, limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    foreign = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
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
    for token in (
        "",
        "   ",
        "not-a-cursor",
        "ae1.onlytwoparts",
        "ae1.too.many.parts",
        "ae0.x.y",
        "ae2.x.y",
        tampered,
        foreign,
    ):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422, repr(token)
        assert resp.json()["error"]["code"] == "validation_error"
        assert "items" not in resp.json()


def test_cursor_signed_with_old_format_marker_is_rejected(client, app, db_session):
    _seed_events(db_session, 2)
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "event_type": None,
                "resource_id": None,
                "from": None,
                "to": None,
                "limit": 50,
                "offset": 1,
            }
        ).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(
        hmac_mod.new(
            app.state.audit_events_cursor_secret,
            f"ae0.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    resp = _list(client, cursor=f"ae0.{payload}.{sig}")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursors_from_other_families_are_rejected(client, app, db_session):
    _seed_events(db_session, 3)
    lineage_cursor = pagination.encode_cursor(
        app.state.lineage_cursor_secret,
        {
            "content_id": "cnt_x",
            "direction": "ancestors",
            "max_depth": 8,
            "min_depth": 1,
            "relation_type": None,
            "limit": 1,
            "offset": 1,
        },
    )
    evidence_cursor = pagination.encode_typed_cursor(
        app.state.content_evidence_cursor_secret,
        pagination.CONTENT_EVIDENCE_CURSOR,
        {
            "content_id": "cnt_x",
            "evidence_type": None,
            "media_type": None,
            "limit": 1,
            "offset": 1,
        },
    )
    for token in (lineage_cursor, evidence_cursor):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


def test_audit_cursor_is_rejected_by_other_endpoints(client, db_session):
    _seed_events(db_session, 3)
    audit_cursor = _list(client, limit=1).json()["next_cursor"]
    create_actor(client)
    created = client.post("/v1/contents", json=content_payload()).json()
    resp = client.get(
        f"/v1/contents/{created['id']}/evidence-bundles",
        params={"limit": 1, "cursor": audit_cursor},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_bound_to_every_effective_query_parameter(client, db_session):
    _seed_timeline(db_session)
    cursor = _list(client, limit=1).json()["next_cursor"]

    # A filter present in the request but absent from the cursor mismatches,
    # as does a changed limit.
    mismatches = [
        {"limit": 2},
        {"event_type": "actor.created"},
        {"resource_id": "res-1"},
        {"from": "2026-01-02T00:00:00Z"},
        {"to": "2026-01-03T00:00:00Z"},
    ]
    for params in mismatches:
        resp = _list(client, cursor=cursor, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"

    # A filtered cursor resumed without its filter (or with a different
    # value) also mismatches.
    filtered = _list(
        client, event_type="actor.created", limit=1
    ).json()["next_cursor"]
    assert _list(client, limit=1, cursor=filtered).status_code == 422
    assert (
        _list(client, event_type="content.created", limit=1, cursor=filtered)
        .status_code
        == 422
    )

    bounded = _list(
        client, **{"from": "2026-01-02T00:00:00Z"}, limit=1
    ).json()["next_cursor"]
    assert _list(client, limit=1, cursor=bounded).status_code == 422
    assert (
        _list(
            client,
            **{"from": "2026-01-03T00:00:00Z"},
            limit=1,
            cursor=bounded,
        ).status_code
        == 422
    )


def test_cursor_accepts_explicit_params_equal_to_issued_query(client, db_session):
    _seed_timeline(db_session)
    cursor = _list(
        client, event_type="actor.created", limit=1
    ).json()["next_cursor"]
    resp = _list(client, event_type="actor.created", limit=1, cursor=cursor)
    assert resp.status_code == 200
    assert [item["resource_id"] for item in resp.json()["items"]] == ["res-4"]


# --- Parameter validation -------------------------------------------------------


def test_blank_filters_are_validation_errors(client, db_session):
    _seed_events(db_session, 2)
    for field in ("event_type", "resource_id"):
        for value in ("", "   ", "\t"):
            resp = _list(client, **{field: value})
            assert resp.status_code == 422, (field, value)
            assert resp.json()["error"]["code"] == "validation_error"


def test_illegal_limit_values_are_validation_errors(client, db_session):
    _seed_events(db_session, 2)
    for value in ("0", "101", "-1", "1.5", "abc", "8.0", "  2", ""):
        resp = _list(client, limit=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_validation_errors(client, db_session):
    _seed_events(db_session, 2)
    for suffix in (
        "event_type=actor.created&event_type=content.created",
        "resource_id=res-000&resource_id=res-001",
        "from=2026-01-01T00:00:00Z&from=2026-01-02T00:00:00Z",
        "to=2026-01-01T00:00:00Z&to=2026-01-02T00:00:00Z",
        "limit=1&limit=2",
        "cursor=x&cursor=y",
    ):
        resp = client.get(f"/v1/audit-events?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameters_are_validation_errors(client, db_session):
    _seed_events(db_session, 2)
    for params in (
        {"actor_id": "org-1"},
        {"event": "actor.created"},
        {"offset": "1"},
        {"event_type": "actor.created", "bogus": "x"},
    ):
        resp = _list(client, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_cursor_with_otherwise_valid_params_is_422(client, db_session):
    _seed_events(db_session, 2)
    resp = _list(
        client,
        event_type="actor.created",
        resource_id="res-000",
        limit=10,
        cursor="garbage",
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only guarantee ---------------------------------------------------------


def test_queries_write_no_rows_or_audit_events(client, db_session):
    _seed_timeline(db_session)

    def counts():
        return (
            db_session.scalar(select(func.count()).select_from(Actor)),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    before = counts()

    # Successful filtered and paginated reads.
    cursor = None
    for _ in range(6):
        params = {"event_type": "actor.created", "limit": 1}
        if cursor is not None:
            params["cursor"] = cursor
        resp = _list(client, **params)
        assert resp.status_code == 200, resp.text
        cursor = resp.json()["next_cursor"]
        if cursor is None:
            break
    _list(client)
    _list(client, resource_id="res-2")
    _list(
        client,
        **{"from": "2026-01-02T00:00:00Z", "to": "2026-01-03T00:00:00Z"},
    )

    # Failed reads must not write anything either.
    _list(client, event_type=" ")
    _list(client, limit=0)
    _list(client, cursor="tampered")
    _list(client, **{"from": "not-a-timestamp"})
    _list(
        client,
        **{"from": "2026-01-04T00:00:00Z", "to": "2026-01-03T00:00:00Z"},
    )
    _list(client, unknown_param="x")
    client.get("/v1/audit-events?limit=1&limit=2")

    db_session.expire_all()
    assert counts() == before
