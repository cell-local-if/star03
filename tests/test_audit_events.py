"""Tests for the read-only audit-event search endpoint.

Covers GET /v1/audit-events: the response shape carries exactly
{"items", "count", "next_cursor"} with each item exposing only
event_type/resource_id/created_at (timezone-aware UTC), events follow the
stable creation order, event_type/resource_id are non-empty exact
combinable filters, from/to are strict RFC 3339 UTC instants applied as
inclusive bounds (from must not exceed to), limit is 1..100 defaulting to
50, opaque HMAC cursors bind every effective filter and the limit so pages
concatenate without gaps or duplicates and the final cursor is null,
blank/illegal/repeated/undeclared parameters and blank/tampered/
foreign-family/mismatching cursors are 422 validation_error, no match is
an empty collection, and neither queries nor failures write any resource
or audit rows. All fixtures are deterministic and offline.
"""

from __future__ import annotations

import base64
import hashlib
import hmac as hmac_mod
import json
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from provenance import pagination
from provenance.models import (
    EVENT_ACTOR_CREATED,
    EVENT_CONTENT_CREATED,
    AuditEvent,
    Content,
)
from tests.helpers import DIGEST_A, DIGEST_B, DIGEST_C, create_actor


def _content_payload(digest, actor_id="org-1"):
    return {
        "digest_algorithm": "sha256",
        "digest_hex": digest,
        "media_type": "image/png",
        "actor_id": actor_id,
    }


def _create_content(client, digest, actor_id="org-1"):
    resp = client.post("/v1/contents", json=_content_payload(digest, actor_id))
    assert resp.status_code == 201, resp.text
    return resp.json()


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
    """Append one audit row at a fixed instant (deterministic time tests)."""
    db_session.add(
        AuditEvent(
            event_type=event_type,
            resource_id=resource_id,
            created_at=created_at,
        )
    )
    db_session.commit()


#: The exact public audit-event field set.
PUBLIC_FIELDS = {"event_type", "resource_id", "created_at"}


def _setup_events(client):
    """Interleaved actor/content creations -> a known audit sequence."""
    create_actor(client, actor_id="org-1")
    c1 = _create_content(client, DIGEST_A)
    create_actor(client, actor_id="org-2", name="Other", type="person")
    c2 = _create_content(client, DIGEST_B, actor_id="org-2")
    c3 = _create_content(client, DIGEST_C)
    return [c1, c2, c3]


# --- Response shape, fields, and ordering -------------------------------------


def test_response_shape_and_item_fields(client):
    _setup_events(client)
    resp = _list(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"items", "count", "next_cursor"}
    assert body["count"] == 5
    assert body["next_cursor"] is None
    assert len(body["items"]) == 5
    for item in body["items"]:
        assert set(item) == PUBLIC_FIELDS


def test_created_at_preserves_utc_timezone(client):
    _setup_events(client)
    for item in _list(client).json()["items"]:
        parsed = datetime.fromisoformat(item["created_at"])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)


def test_events_follow_stable_creation_order(client):
    c1, c2, c3 = _setup_events(client)
    items = _list(client).json()["items"]
    assert [(i["event_type"], i["resource_id"]) for i in items] == [
        (EVENT_ACTOR_CREATED, "org-1"),
        (EVENT_CONTENT_CREATED, c1["id"]),
        (EVENT_ACTOR_CREATED, "org-2"),
        (EVENT_CONTENT_CREATED, c2["id"]),
        (EVENT_CONTENT_CREATED, c3["id"]),
    ]


def test_empty_database_is_an_empty_collection(client):
    assert _list(client).json() == {"items": [], "count": 0, "next_cursor": None}


# --- Exact-match filtering -----------------------------------------------------


def test_event_type_exact_match(client):
    _setup_events(client)
    body = _list(client, event_type=EVENT_CONTENT_CREATED).json()
    assert body["count"] == 3
    assert {i["event_type"] for i in body["items"]} == {EVENT_CONTENT_CREATED}


def test_resource_id_exact_match(client):
    c1, _, _ = _setup_events(client)
    body = _list(client, resource_id=c1["id"]).json()
    assert body["count"] == 1
    assert body["items"][0]["resource_id"] == c1["id"]
    assert body["items"][0]["event_type"] == EVENT_CONTENT_CREATED


def test_filters_combine_as_logical_and(client):
    c1, c2, _ = _setup_events(client)
    body = _list(
        client,
        event_type=EVENT_CONTENT_CREATED,
        resource_id=c2["id"],
    ).json()
    assert [i["resource_id"] for i in body["items"]] == [c2["id"]]
    assert body["count"] == 1

    # A combination that no single event satisfies is empty, not an error.
    body = _list(
        client,
        event_type=EVENT_ACTOR_CREATED,
        resource_id=c1["id"],
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


def test_exact_match_is_case_and_whitespace_sensitive(client):
    _setup_events(client)
    for params in (
        {"event_type": "ACTOR.CREATED"},
        {"event_type": "actor.created "},
        {"event_type": " actor.created"},
        {"resource_id": "ORG-1"},
        {"resource_id": "org-1 "},
    ):
        body = _list(client, **params).json()
        assert body == {"items": [], "count": 0, "next_cursor": None}, params


def test_no_matching_events_is_empty_collection(client):
    _setup_events(client)
    body = _list(client, event_type="claim.created").json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


# --- Time-bound filtering -------------------------------------------------------

T1 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc)
T3 = datetime(2026, 1, 3, 23, 59, 59, tzinfo=timezone.utc)


def _setup_timed_events(db_session):
    _insert_event(db_session, EVENT_ACTOR_CREATED, "org-1", T1)
    _insert_event(db_session, EVENT_CONTENT_CREATED, "cnt_t2", T2)
    _insert_event(db_session, EVENT_CONTENT_CREATED, "cnt_t3", T3)


def test_from_bound_is_inclusive(client, db_session):
    _setup_timed_events(db_session)
    body = _list(client, **{"from": "2026-01-02T12:30:00Z"}).json()
    assert [i["resource_id"] for i in body["items"]] == ["cnt_t2", "cnt_t3"]
    assert body["count"] == 2


def test_to_bound_is_inclusive(client, db_session):
    _setup_timed_events(db_session)
    body = _list(client, to="2026-01-02T12:30:00Z").json()
    assert [i["resource_id"] for i in body["items"]] == ["org-1", "cnt_t2"]
    assert body["count"] == 2


def test_from_equal_to_matches_exact_instant(client, db_session):
    _setup_timed_events(db_session)
    body = _list(
        client,
        **{"from": "2026-01-02T12:30:00Z", "to": "2026-01-02T12:30:00Z"},
    ).json()
    assert [i["resource_id"] for i in body["items"]] == ["cnt_t2"]
    assert body["count"] == 1


def test_time_window_between_instants(client, db_session):
    _setup_timed_events(db_session)
    body = _list(
        client,
        **{"from": "2026-01-01T00:00:01Z", "to": "2026-01-03T00:00:00Z"},
    ).json()
    assert [i["resource_id"] for i in body["items"]] == ["cnt_t2"]
    assert body["count"] == 1


def test_time_filters_combine_with_exact_filters(client, db_session):
    _setup_timed_events(db_session)
    body = _list(
        client,
        event_type=EVENT_CONTENT_CREATED,
        to="2026-01-02T12:30:00Z",
    ).json()
    assert [i["resource_id"] for i in body["items"]] == ["cnt_t2"]

    body = _list(
        client,
        event_type=EVENT_ACTOR_CREATED,
        **{"from": "2026-01-02T00:00:00Z"},
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


def test_utc_offset_notation_and_fractional_seconds_accepted(client, db_session):
    _setup_timed_events(db_session)
    body = _list(
        client,
        **{
            "from": "2026-01-02T12:30:00+00:00",
            "to": "2026-01-03T23:59:59.000Z",
        },
    ).json()
    assert [i["resource_id"] for i in body["items"]] == ["cnt_t2", "cnt_t3"]


def test_from_later_than_to_is_validation_error(client, db_session):
    _setup_timed_events(db_session)
    resp = _list(
        client,
        **{"from": "2026-01-03T00:00:00Z", "to": "2026-01-02T00:00:00Z"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


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
            "2026-01-02T12:30:00-00:01",
            "2026-13-02T12:30:00Z",
            "2026-01-02T25:30:00Z",
            "2026-01-02t12:30:00z",
            "not-a-time",
        ):
            resp = _list(client, **{field: value})
            assert resp.status_code == 422, (field, value)
            assert resp.json()["error"]["code"] == "validation_error"


# --- Pagination -----------------------------------------------------------------


def _setup_many_events(client, count=7):
    """``count`` content creations on one actor -> count+1 audit events."""
    create_actor(client)
    contents = []
    for i in range(count):
        digest = hashlib.sha256(f"audit-page-{i}".encode()).hexdigest()
        contents.append(_create_content(client, digest))
    return contents


def test_pagination_concatenates_without_gaps_or_duplicates(client):
    _setup_many_events(client, 7)
    all_items, pages, count = _walk_pages(client, limit=3)
    # 1 actor + 7 contents = 8 events -> pages of 3, 3, 2.
    assert count == 8
    assert [len(page) for page in pages] == [3, 3, 2]
    keys = [(i["event_type"], i["resource_id"]) for i in all_items]
    assert len(keys) == len(set(keys)) == 8
    # Concatenated pages equal the unpaginated listing exactly.
    assert all_items == _list(client).json()["items"]


def test_count_is_filtered_total_on_every_page(client):
    _setup_many_events(client, 7)
    all_items, pages, count = _walk_pages(
        client, event_type=EVENT_CONTENT_CREATED, limit=2
    )
    assert count == 7
    assert [len(page) for page in pages] == [2, 2, 2, 1]
    assert {i["event_type"] for i in all_items} == {EVENT_CONTENT_CREATED}


def test_last_page_cursor_null_on_exact_division(client):
    _setup_many_events(client, 3)  # 4 events total
    first = _list(client, limit=2).json()
    assert first["next_cursor"] is not None
    second = _list(client, limit=2, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 2
    assert second["count"] == 4
    assert second["next_cursor"] is None


def test_default_limit_is_fifty(client):
    _setup_many_events(client, 55)  # 56 events total
    first = _list(client).json()
    assert len(first["items"]) == 50
    assert first["count"] == 56
    assert first["next_cursor"] is not None
    second = _list(client, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 6
    assert second["count"] == 56
    assert second["next_cursor"] is None


def test_limit_boundaries_accepted(client):
    _setup_many_events(client, 2)
    for value in (1, 100):
        assert _list(client, limit=value).status_code == 200


def test_reusing_a_cursor_replays_the_same_page(client):
    _setup_many_events(client, 5)
    cursor = _list(client, limit=2).json()["next_cursor"]
    replay_one = _list(client, limit=2, cursor=cursor).json()
    replay_two = _list(client, limit=2, cursor=cursor).json()
    assert replay_one == replay_two


def test_cursor_past_end_returns_empty_page_with_total_count(client, app):
    _setup_many_events(client, 2)  # 3 events total
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


# --- Cursor integrity ------------------------------------------------------------


def test_tampered_or_malformed_cursors_are_validation_errors(client, app):
    _setup_many_events(client, 3)
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
        "v1.x.y",
        "ce1.x.y",
        tampered,
        foreign,
    ):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422, repr(token)
        assert resp.json()["error"]["code"] == "validation_error"
        assert "items" not in resp.json()


def test_cursor_signed_with_old_format_marker_is_rejected(client, app):
    _setup_many_events(client, 2)
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


def test_cursor_from_other_families_is_rejected(client, app):
    _setup_many_events(client, 3)
    lineage_cursor = pagination.encode_cursor(
        secrets.token_bytes(32),
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
        secrets.token_bytes(32),
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


def test_audit_cursor_is_rejected_by_other_endpoints(client, app):
    _setup_many_events(client, 3)
    audit_cursor = _list(client, limit=1).json()["next_cursor"]
    content = _create_content(client, DIGEST_B)
    resp = client.get(
        f"/v1/contents/{content['id']}/evidence-bundles",
        params={"limit": 1, "cursor": audit_cursor},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_bound_to_every_effective_query_parameter(client, db_session):
    _setup_timed_events(db_session)
    _setup_many_events(client, 5)
    cursor = _list(client, limit=2).json()["next_cursor"]

    # A filter present in the request but absent from the cursor mismatches.
    mismatches = [
        {"limit": 3},
        {"event_type": EVENT_CONTENT_CREATED},
        {"resource_id": "org-1"},
        {"from": "2026-01-01T00:00:00Z"},
        {"to": "2027-01-01T00:00:00Z"},
    ]
    for params in mismatches:
        resp = _list(client, cursor=cursor, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"

    # A filtered cursor resumed without (or with a changed) filter mismatches.
    filtered = _list(
        client, event_type=EVENT_CONTENT_CREATED, limit=2
    ).json()["next_cursor"]
    assert _list(client, limit=2, cursor=filtered).status_code == 422
    assert (
        _list(client, event_type=EVENT_ACTOR_CREATED, limit=2,
              cursor=filtered).status_code
        == 422
    )

    timed = _list(
        client, **{"from": "2026-01-01T00:00:00Z"}, limit=2
    ).json()["next_cursor"]
    assert _list(client, limit=2, cursor=timed).status_code == 422
    assert (
        _list(client, **{"from": "2026-01-02T00:00:00Z"}, limit=2,
              cursor=timed).status_code
        == 422
    )


def test_cursor_binds_the_instant_not_its_notation(client, db_session):
    _setup_timed_events(db_session)
    first = _list(client, **{"from": "2026-01-02T00:00:00Z"}, limit=1).json()
    assert first["next_cursor"] is not None
    # The same instant spelled with +00:00 resumes the "Z"-minted cursor.
    resp = _list(
        client,
        **{"from": "2026-01-02T00:00:00+00:00"},
        limit=1,
        cursor=first["next_cursor"],
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["count"] == 2


# --- Parameter validation --------------------------------------------------------


def test_blank_filters_are_validation_errors(client):
    _setup_many_events(client, 1)
    for field in ("event_type", "resource_id"):
        for value in ("", "   ", "\t"):
            resp = _list(client, **{field: value})
            assert resp.status_code == 422, (field, value)
            assert resp.json()["error"]["code"] == "validation_error"


def test_illegal_limit_values_are_validation_errors(client):
    _setup_many_events(client, 1)
    for value in ("0", "101", "-1", "1.5", "abc", "8.0", "  2", ""):
        resp = _list(client, limit=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_validation_errors(client):
    _setup_many_events(client, 1)
    for suffix in (
        "event_type=actor.created&event_type=content.created",
        "resource_id=a&resource_id=b",
        "from=2026-01-01T00:00:00Z&from=2026-01-02T00:00:00Z",
        "to=2026-01-01T00:00:00Z&to=2026-01-02T00:00:00Z",
        "limit=1&limit=2",
        "cursor=x&cursor=y",
    ):
        resp = client.get(f"/v1/audit-events?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameters_are_validation_errors(client):
    _setup_many_events(client, 1)
    for suffix in (
        "actor_id=org-1",
        "event_types=actor.created",
        "limit=1&offset=2",
        "CURSOR=x",
    ):
        resp = client.get(f"/v1/audit-events?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_cursor_with_otherwise_valid_params_is_422(client):
    _setup_many_events(client, 1)
    resp = _list(
        client,
        event_type=EVENT_CONTENT_CREATED,
        limit=10,
        cursor="garbage",
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only guarantee ----------------------------------------------------------


def test_queries_and_failures_write_no_rows_or_audit_events(client, db_session):
    _setup_many_events(client, 4)

    def counts():
        return (
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
            db_session.scalar(select(func.count()).select_from(Content)),
        )

    before = counts()

    # Successful filtered and paginated reads.
    cursor = None
    for _ in range(6):
        params = {"event_type": EVENT_CONTENT_CREATED, "limit": 1}
        if cursor is not None:
            params["cursor"] = cursor
        resp = _list(client, **params)
        assert resp.status_code == 200, resp.text
        cursor = resp.json()["next_cursor"]
        if cursor is None:
            break
    _list(client)
    _list(client, event_type="no.such.event")

    # Failed reads must not write anything either.
    _list(client, event_type=" ")
    _list(client, limit=0)
    _list(client, cursor="tampered")
    _list(client, **{"from": "not-a-time"})
    _list(client, **{"from": "2027-01-01T00:00:00Z",
                     "to": "2026-01-01T00:00:00Z"})
    client.get("/v1/audit-events?limit=1&limit=2")
    client.get("/v1/audit-events?unknown=1")

    db_session.expire_all()
    assert counts() == before
