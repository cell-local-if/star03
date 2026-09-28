"""Tests for the read-only audit checkpoint job search endpoint.

Covers GET /v1/audit-checkpoint-jobs: the response shape carries exactly
{"items", "count", "next_cursor"} with each item being the existing
single-job public view (the twelve fields, including a settled result
snapshot with only public checkpoint/event material), jobs follow stable
creation order, request_id/event_type/resource_id are exact combinable
case/whitespace-sensitive filters (status accepts only the four lifecycle
literals and is absent by default), from/to are strict RFC 3339 UTC
inclusive created_at bounds (from must not exceed to), limit is 1..100
defaulting to 50, opaque HMAC cursors (their own acj1 family) bind every
effective filter and the limit so pages concatenate without gaps or
duplicates and the final cursor is null, blank/illegal/repeated/undeclared
parameters, an inverted range, an illegal status, and blank/tampered/
foreign-family/mismatching cursors are 422 validation_error rejected before
any job is read, any GET request body is 422, no match is an empty
collection, the success body is compact UTF-8 JSON ending in one newline,
and neither queries nor failures write any job or audit rows. All fixtures
are deterministic and offline.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from provenance import pagination
from provenance.models import AuditCheckpointJob, AuditEvent
from tests.helpers import create_actor


JOBS_URL = "/v1/audit-checkpoint-jobs"

#: The exact public single-job field set.
JOB_KEYS = {
    "id",
    "request_id",
    "event_type",
    "resource_id",
    "from",
    "to",
    "status",
    "created_at",
    "started_at",
    "finished_at",
    "result",
    "error",
}

T1 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc)
T3 = datetime(2026, 1, 3, 23, 59, 59, tzinfo=timezone.utc)


def _create_job(client, request_id, **filters):
    resp = client.post(
        JOBS_URL, json={"request_id": request_id, **filters}
    )
    assert resp.status_code in (200, 201), resp.text
    return resp.json()


def _list(client, **params):
    return client.get(JOBS_URL, params=params)


def _walk_pages(client, **params):
    """Follow next_cursor until exhausted; return (all_items, pages, count)."""
    pages = []
    all_items = []
    count = None
    cursor = None
    for _ in range(200):
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


def _insert_job(db_session, job_id, request_id, created_at, status="pending"):
    """Insert one job row directly at a fixed instant (time-range tests)."""
    db_session.add(
        AuditCheckpointJob(
            id=job_id,
            request_id=request_id,
            status=status,
            created_at=created_at,
        )
    )
    db_session.commit()


# --- Response shape, fields, and ordering -------------------------------------


def test_response_shape_and_item_fields(client):
    job = _create_job(client, "req-1", event_type="actor.created")

    resp = _list(client)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/json"
    body = resp.json()
    assert set(body) == {"items", "count", "next_cursor"}
    assert body["count"] == 1
    assert body["next_cursor"] is None
    assert len(body["items"]) == 1
    assert set(body["items"][0]) == JOB_KEYS
    assert body["items"][0] == job


def test_item_is_the_single_job_public_view(client):
    job = _create_job(client, "req-view", event_type="actor.created")
    settled = client.post(f"{JOBS_URL}/{job['id']}/run").json()

    listed = _list(client).json()["items"][0]
    # The list item is exactly the GET single-resource view.
    single = client.get(f"{JOBS_URL}/{job['id']}").json()
    assert listed == single == settled
    # A settled job carries its checkpoint snapshot, never raw event bytes.
    assert set(listed["result"]) == {"checkpoint", "events"}
    for event in listed["result"]["events"]:
        assert set(event) == {"event_type", "resource_id", "created_at"}


def test_jobs_follow_stable_creation_order(client):
    j1 = _create_job(client, "req-1", event_type="actor.created")
    j2 = _create_job(client, "req-2", resource_id="org-1")
    j3 = _create_job(client, "req-3")

    items = _list(client).json()["items"]
    assert [i["id"] for i in items] == [j1["id"], j2["id"], j3["id"]]
    for earlier, later in zip(items, items[1:]):
        assert earlier["created_at"] <= later["created_at"]


def test_empty_database_is_an_empty_collection(client):
    resp = _list(client)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_success_body_is_compact_json_ending_in_one_newline(client):
    _create_job(client, "req-nl")
    resp = _list(client)
    assert resp.status_code == 200
    assert resp.content.endswith(b"\n")
    assert not resp.content.endswith(b"\n\n")
    # Compact separators: no whitespace after delimiters.
    assert b": " not in resp.content.rstrip(b"\n")
    assert b", " not in resp.content.rstrip(b"\n")
    # Members are ordered items, count, next_cursor.
    text = resp.text
    assert text.index('"items"') < text.index('"count"') < text.index(
        '"next_cursor"'
    )


# --- Exact-match filtering -----------------------------------------------------


def test_filter_by_request_id(client):
    _create_job(client, "one")
    other = _create_job(client, "two")

    body = _list(client, request_id="two").json()
    assert [i["id"] for i in body["items"]] == [other["id"]]
    assert body["count"] == 1


def test_filter_by_event_type(client):
    match = _create_job(client, "req-a", event_type="actor.created")
    _create_job(client, "req-b", event_type="content.created")

    body = _list(client, event_type="actor.created").json()
    assert [i["id"] for i in body["items"]] == [match["id"]]
    assert body["count"] == 1


def test_filter_by_resource_id(client):
    match = _create_job(client, "req-a", resource_id="org-1")
    _create_job(client, "req-b", resource_id="org-2")

    body = _list(client, resource_id="org-1").json()
    assert [i["id"] for i in body["items"]] == [match["id"]]
    assert body["count"] == 1


def test_filters_combine_as_logical_and(client):
    match = _create_job(
        client, "req-match", event_type="actor.created", resource_id="org-1"
    )
    _create_job(client, "req-other-type", event_type="content.created",
                resource_id="org-1")
    _create_job(client, "req-other-resource", event_type="actor.created",
                resource_id="org-2")

    body = _list(
        client, event_type="actor.created", resource_id="org-1"
    ).json()
    assert [i["id"] for i in body["items"]] == [match["id"]]
    assert body["count"] == 1

    # A combination nothing satisfies is empty, not an error.
    assert _list(
        client, request_id="req-match", status="failed"
    ).json() == {"items": [], "count": 0, "next_cursor": None}


def test_exact_match_is_case_and_whitespace_sensitive(client):
    _create_job(client, "Request-1", event_type="actor.created",
                resource_id="Org-1")

    for params in (
        {"request_id": "request-1"},
        {"request_id": "Request-1 "},
        {"request_id": " Request-1"},
        {"event_type": "Actor.Created"},
        {"event_type": "actor.created "},
        {"resource_id": "org-1"},
        {"resource_id": " Org-1"},
    ):
        body = _list(client, **params).json()
        assert body == {"items": [], "count": 0, "next_cursor": None}, params


def test_unknown_filter_value_is_empty_not_error(client):
    _create_job(client, "known")
    assert _list(client, request_id="no-such-request").json() == {
        "items": [], "count": 0, "next_cursor": None
    }
    assert _list(client, event_type="no.such.event").json() == {
        "items": [], "count": 0, "next_cursor": None
    }
    assert _list(client, resource_id="res_unknown").json() == {
        "items": [], "count": 0, "next_cursor": None
    }


@pytest.mark.parametrize("field", ["request_id", "event_type", "resource_id"])
@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_blank_text_filter_is_validation_error(client, field, value):
    resp = _list(client, **{field: value})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Status filter --------------------------------------------------------------


@pytest.mark.parametrize("state", ["pending", "running", "succeeded", "failed"])
def test_status_accepts_each_lifecycle_literal(client, state):
    _create_job(client, f"req-{state}")
    # Each literal is accepted structurally (unknown state simply matches 0;
    # pending matches the one row).
    resp = _list(client, status=state)
    assert resp.status_code == 200
    assert resp.json()["count"] == (1 if state == "pending" else 0)


def test_status_filter_matches_exact_lifecycle_state(client):
    pending = _create_job(client, "p")
    succeeded = _create_job(client, "s")
    client.post(f"{JOBS_URL}/{succeeded['id']}/run")

    pending_ids = [i["id"] for i in _list(client, status="pending").json()["items"]]
    assert pending_ids == [pending["id"]]
    succeeded_ids = [
        i["id"] for i in _list(client, status="succeeded").json()["items"]
    ]
    assert succeeded_ids == [succeeded["id"]]
    # Absent status means unfiltered across all four states.
    assert _list(client).json()["count"] == 2


@pytest.mark.parametrize(
    "value",
    ["", "   ", "PENDING", "Pending", "done", "pending ", " pending", "settled"],
)
def test_illegal_status_is_validation_error(client, value):
    resp = _list(client, status=value)
    assert resp.status_code == 422, value
    assert resp.json()["error"]["code"] == "validation_error"


def test_status_can_change_between_queries(client):
    job = _create_job(client, "req-change")
    assert _list(client, status="pending").json()["count"] == 1
    client.post(f"{JOBS_URL}/{job['id']}/run")
    assert _list(client, status="pending").json()["count"] == 0
    succeeded = _list(client, status="succeeded").json()
    assert [i["id"] for i in succeeded["items"]] == [job["id"]]


# --- Time-bound filtering -------------------------------------------------------


def test_from_bound_is_inclusive(client, db_session):
    _insert_job(db_session, "acj_t1", "r1", T1)
    _insert_job(db_session, "acj_t2", "r2", T2)
    _insert_job(db_session, "acj_t3", "r3", T3)

    body = _list(client, **{"from": "2026-01-02T12:30:00Z"}).json()
    assert [i["id"] for i in body["items"]] == ["acj_t2", "acj_t3"]
    assert body["count"] == 2


def test_to_bound_is_inclusive(client, db_session):
    _insert_job(db_session, "acj_t1", "r1", T1)
    _insert_job(db_session, "acj_t2", "r2", T2)
    _insert_job(db_session, "acj_t3", "r3", T3)

    body = _list(client, to="2026-01-02T12:30:00Z").json()
    assert [i["id"] for i in body["items"]] == ["acj_t1", "acj_t2"]
    assert body["count"] == 2


def test_from_equal_to_matches_exact_instant(client, db_session):
    _insert_job(db_session, "acj_t1", "r1", T1)
    _insert_job(db_session, "acj_t2", "r2", T2)
    body = _list(
        client,
        **{"from": "2026-01-02T12:30:00Z", "to": "2026-01-02T12:30:00Z"},
    ).json()
    assert [i["id"] for i in body["items"]] == ["acj_t2"]
    assert body["count"] == 1


def test_utc_offset_notation_and_fractional_seconds_accepted(client, db_session):
    _insert_job(db_session, "acj_t2", "r2", T2)
    _insert_job(db_session, "acj_t3", "r3", T3)
    body = _list(
        client,
        **{
            "from": "2026-01-02T12:30:00+00:00",
            "to": "2026-01-03T23:59:59.000Z",
        },
    ).json()
    assert [i["id"] for i in body["items"]] == ["acj_t2", "acj_t3"]


def test_from_later_than_to_is_validation_error(client, db_session):
    _insert_job(db_session, "acj_t1", "r1", T1)
    resp = _list(
        client,
        **{"from": "2026-01-03T00:00:00Z", "to": "2026-01-02T00:00:00Z"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "2026-01-02",
        "2026-01-02T12:30:00",
        "2026-01-02 12:30:00Z",
        "2026-01-02T12:30Z",
        "2026-01-02T12:30:00+01:00",
        "2026-01-02T12:30:00-00:01",
        "2026-13-02T12:30:00Z",
        "2026-01-02t12:30:00z",
        "not-a-time",
    ],
)
def test_invalid_time_values_are_validation_errors(client, value):
    for field in ("from", "to"):
        resp = _list(client, **{field: value})
        assert resp.status_code == 422, (field, value)
        assert resp.json()["error"]["code"] == "validation_error"


# --- Pagination -----------------------------------------------------------------


def _setup_many_jobs(client, count):
    return [_create_job(client, f"req-{i:03d}") for i in range(count)]


def test_pagination_concatenates_without_gaps_or_duplicates(client):
    jobs = _setup_many_jobs(client, 7)
    all_items, pages, count = _walk_pages(client, limit=3)
    assert count == 7
    assert [len(page) for page in pages] == [3, 3, 1]
    ids = [i["id"] for i in all_items]
    assert len(ids) == len(set(ids)) == 7
    assert ids == [j["id"] for j in jobs]


def test_count_is_filtered_total_on_every_page(client):
    _setup_many_jobs(client, 5)
    all_items, pages, count = _walk_pages(client, status="pending", limit=2)
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    assert {i["status"] for i in all_items} == {"pending"}


def test_last_page_cursor_null_on_exact_division(client):
    _setup_many_jobs(client, 4)
    first = _list(client, limit=2).json()
    assert first["next_cursor"] is not None
    second = _list(client, limit=2, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 2
    assert second["count"] == 4
    assert second["next_cursor"] is None


def test_default_limit_is_fifty(client):
    _setup_many_jobs(client, 53)
    first = _list(client).json()
    assert len(first["items"]) == 50
    assert first["count"] == 53
    assert first["next_cursor"] is not None
    second = _list(client, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 3
    assert second["count"] == 53
    assert second["next_cursor"] is None


@pytest.mark.parametrize("value", [1, 100])
def test_limit_boundaries_accepted(client, value):
    _setup_many_jobs(client, 1)
    assert _list(client, limit=value).status_code == 200


@pytest.mark.parametrize(
    "value",
    ["0", "101", "-1", "1.0", "1e1", " 1", "1 ", "", "   ", "abc", "+1"],
)
def test_illegal_limit_is_validation_error(client, value):
    _setup_many_jobs(client, 1)
    resp = _list(client, limit=value)
    assert resp.status_code == 422, value
    assert resp.json()["error"]["code"] == "validation_error"


def test_reusing_a_cursor_replays_the_same_page(client):
    _setup_many_jobs(client, 5)
    cursor = _list(client, limit=2).json()["next_cursor"]
    one = _list(client, limit=2, cursor=cursor).json()
    two = _list(client, limit=2, cursor=cursor).json()
    assert one == two


def test_cursor_past_end_returns_empty_page_with_total_count(client, app):
    _setup_many_jobs(client, 2)
    token = pagination.encode_typed_cursor(
        app.state.audit_checkpoint_jobs_cursor_secret,
        pagination.AUDIT_CHECKPOINT_JOBS_CURSOR,
        {
            "request_id": None,
            "event_type": None,
            "resource_id": None,
            "status": None,
            "from": None,
            "to": None,
            "limit": 50,
            "offset": 99,
        },
    )
    body = _list(client, cursor=token).json()
    assert body == {"items": [], "count": 2, "next_cursor": None}


def test_filtered_empty_first_page_has_null_cursor(client):
    _setup_many_jobs(client, 2)
    body = _list(client, request_id="matches-nothing").json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


# --- Cursor integrity ------------------------------------------------------------


def test_tampered_or_malformed_cursors_are_validation_errors(client):
    _setup_many_jobs(client, 3)
    good = _list(client, limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    for token in (
        "",
        "   ",
        "not-a-cursor",
        "acj1.onlytwoparts",
        "acj1.too.many.parts",
        "acj0.x.y",
        "acj2.x.y",
        tampered,
    ):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422, repr(token)
        assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_signed_with_wrong_secret_is_rejected(client):
    _setup_many_jobs(client, 2)
    foreign = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.AUDIT_CHECKPOINT_JOBS_CURSOR,
        {
            "request_id": None,
            "event_type": None,
            "resource_id": None,
            "status": None,
            "from": None,
            "to": None,
            "limit": 1,
            "offset": 1,
        },
    )
    resp = _list(client, cursor=foreign)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_from_other_families_is_rejected(client):
    _setup_many_jobs(client, 3)
    # The content export job family (cx1) shares the exact same filter
    # shape except content_id instead of event_type/resource_id and is the
    # most likely cross-entry confusion; it must never resume this query.
    token = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.CONTENT_EXPORT_JOBS_CURSOR,
        {
            "content_id": None,
            "request_id": None,
            "status": None,
            "from": None,
            "to": None,
            "limit": 1,
            "offset": 1,
        },
    )
    resp = _list(client, limit=1, cursor=token)
    assert resp.status_code == 422


def test_audit_jobs_cursor_is_rejected_by_other_endpoints(client):
    _setup_many_jobs(client, 3)
    cursor = _list(client, limit=1).json()["next_cursor"]
    assert client.get(
        "/v1/audit-events", params={"limit": 1, "cursor": cursor}
    ).status_code == 422
    assert client.get(
        "/v1/content-export-jobs", params={"limit": 1, "cursor": cursor}
    ).status_code == 422
    assert client.get(
        "/v1/claims", params={"limit": 1, "cursor": cursor}
    ).status_code == 422


def test_cursor_bound_to_every_effective_query_parameter(client):
    _setup_many_jobs(client, 4)
    cursor = _list(client, limit=2).json()["next_cursor"]

    # A cursor minted by an unfiltered query must not be replayed under any
    # different effective filter or limit.
    for params in (
        {"limit": 3},
        {"request_id": "req-000"},
        {"event_type": "actor.created"},
        {"resource_id": "org-1"},
        {"status": "pending"},
        {"from": "2026-01-01T00:00:00Z"},
        {"to": "2027-01-01T00:00:00Z"},
    ):
        resp = _list(client, cursor=cursor, **params)
        assert resp.status_code == 422, params


def test_cursor_replayed_with_equivalent_utc_spelling_resumes(client):
    # The cursor binds the effective instant, so "Z" on the first page and
    # "+00:00" on the continuation name the same query.
    _setup_many_jobs(client, 3)
    first = _list(client, limit=1, **{"from": "2000-01-01T00:00:00Z"}).json()
    resp = _list(
        client,
        limit=1,
        cursor=first["next_cursor"],
        **{"from": "2000-01-01T00:00:00+00:00"},
    )
    assert resp.status_code == 200, resp.text


# --- Parameter validation --------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        {"unknown": "1"},
        {"request_id": ["a", "b"]},
        {"event_id": "x"},
        {"limit": "1&limit=2"},
    ],
)
def test_undeclared_or_repeated_parameters_are_validation_errors(
    client, params
):
    resp = client.get(JOBS_URL, params=params)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameter_is_validation_error(client):
    resp = client.get(JOBS_URL + "?status=pending&status=failed")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Empty body ------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [b" ", b"\t", b"\n", b"{not json", b'{"a":1}', b"\x00", b"abc"],
)
def test_any_request_body_is_validation_error_before_reads(
    client, db_session, body
):
    _create_job(client, "req-body")
    before = len(db_session.execute(select(AuditEvent)).scalars().all())

    resp = client.request("GET", JOBS_URL, content=body)
    assert resp.status_code == 422, body
    assert resp.json()["error"]["code"] == "validation_error"

    # The body is rejected before any job is read or anything is written.
    db_session.expire_all()
    assert len(db_session.execute(select(AuditEvent)).scalars().all()) == before
    jobs = db_session.execute(select(AuditCheckpointJob)).scalars().all()
    assert len(jobs) == 1


# --- Read-only --------------------------------------------------------------------


def test_successful_query_writes_nothing(client, db_session):
    create_actor(client)
    _create_job(client, "req-ro", event_type="actor.created")
    before_events = len(db_session.execute(select(AuditEvent)).scalars().all())
    before_jobs = len(
        db_session.execute(select(AuditCheckpointJob)).scalars().all()
    )

    assert _list(client, event_type="actor.created", limit=1).status_code == 200
    assert _list(client, request_id="missing").status_code == 200

    db_session.expire_all()
    assert len(db_session.execute(select(AuditEvent)).scalars().all()) == before_events
    assert (
        len(db_session.execute(select(AuditCheckpointJob)).scalars().all())
        == before_jobs
    )


def test_failed_validation_writes_nothing(client, db_session):
    _create_job(client, "req-fail-ro")
    before = len(db_session.execute(select(AuditEvent)).scalars().all())
    assert _list(client, status="nope").status_code == 422
    db_session.expire_all()
    assert len(db_session.execute(select(AuditEvent)).scalars().all()) == before
    assert (
        len(db_session.execute(select(AuditCheckpointJob)).scalars().all())
        == 1
    )


# --- Method handling --------------------------------------------------------------


@pytest.mark.parametrize("method", ["put", "patch", "delete"])
def test_other_methods_are_405(client, method):
    resp = getattr(client, method)(JOBS_URL)
    assert resp.status_code == 405, resp.text
    assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Stable order across restart --------------------------------------------------


def test_stable_order_survives_restart(tmp_db_url):
    from fastapi.testclient import TestClient
    from provenance.app import create_app
    from provenance.config import Settings

    app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app) as client:
        jobs = _setup_many_jobs(client, 3)

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        body = _list(client).json()
        assert [i["id"] for i in body["items"]] == [j["id"] for j in jobs]
