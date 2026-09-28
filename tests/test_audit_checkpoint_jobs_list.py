"""Tests for the read-only audit checkpoint job search endpoint.

Covers GET /v1/audit-checkpoint-jobs: the response shape carries exactly
{"items", "count", "next_cursor"} with each item being the existing
single-job public view (eleven fields, including a settled result package),
jobs follow stable creation order (created_at with the monotonic seq
tiebreaker, stable across a restart), request_id/event_type/resource_id are
exact combinable case/whitespace-sensitive filters (status accepts only the
four lifecycle literals and is absent by default), from/to are strict RFC
3339 UTC inclusive created_at bounds (from must not exceed to), limit is
1..100 defaulting to 50, opaque HMAC cursors (their own aj1 family) bind
every effective filter and the limit so pages concatenate without gaps or
duplicates and the final cursor is null, blank/illegal/repeated/undeclared
parameters, an inverted range, an illegal status, and blank/tampered/
foreign-family/mismatching cursors are 422 validation_error rejected before
any job is read, any GET request body is 422, non-GET non-POST methods are
405, no match is an empty collection, the success body is compact UTF-8
JSON ending in one newline, and neither queries nor failures write any job
or audit rows. All fixtures are deterministic and offline.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select, update as sa_update

from provenance import pagination, service
from provenance.models import AuditCheckpointJob, AuditEvent

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


def _insert_job(
    db_session,
    job_id,
    request_id,
    created_at,
    event_type=None,
    resource_id=None,
    status="pending",
):
    """Insert one job row directly at a fixed instant (time-range tests)."""
    db_session.add(
        AuditCheckpointJob(
            id=job_id,
            request_id=request_id,
            event_type=event_type,
            resource_id=resource_id,
            status=status,
            created_at=created_at,
        )
    )
    db_session.commit()


# --- Response shape, fields, and ordering -------------------------------------


def test_response_shape_and_item_fields(client):
    _create_job(client, "req-1", event_type="actor.created")

    resp = _list(client)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/json"
    body = resp.json()
    assert set(body) == {"items", "count", "next_cursor"}
    assert body["count"] == 1
    assert body["next_cursor"] is None
    assert len(body["items"]) == 1
    assert set(body["items"][0]) == JOB_KEYS


def test_item_is_the_single_job_public_view_with_settled_result(client):
    job = _create_job(client, "req-view", event_type="actor.created")
    settled = client.post(f"{JOBS_URL}/{job['id']}/run").json()

    listed = _list(client).json()["items"][0]
    # The list item is exactly the GET single-resource view.
    single = client.get(f"{JOBS_URL}/{job['id']}").json()
    assert listed == single == settled
    # A settled job carries its checkpoint package, never raw material.
    assert set(listed["result"]) == {"checkpoint", "events"}


def test_failed_job_item_carries_error_and_null_result(client, monkeypatch):
    job = _create_job(client, "req-fail")

    def _boom(session, job):
        raise RuntimeError("simulated checkpoint failure")

    monkeypatch.setattr(service, "list_audit_events", _boom)
    assert client.post(f"{JOBS_URL}/{job['id']}/run").status_code == 200
    monkeypatch.undo()

    listed = _list(client, status="failed").json()["items"][0]
    assert listed["result"] is None
    assert listed["error"] == "audit_checkpoint_export_failed"
    # The failed view is exactly the single-resource view.
    assert listed == client.get(f"{JOBS_URL}/{job['id']}").json()


def test_jobs_follow_stable_creation_order(client):
    j1 = _create_job(client, "req-1", event_type="actor.created")
    j2 = _create_job(client, "req-2", event_type="content.created")
    j3 = _create_job(client, "req-3", resource_id="res-1")

    items = _list(client).json()["items"]
    assert [i["id"] for i in items] == [j1["id"], j2["id"], j3["id"]]
    for earlier, later in zip(items, items[1:]):
        assert earlier["created_at"] <= later["created_at"]


def test_empty_database_is_an_empty_collection(client):
    resp = _list(client)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


# --- Exact-match filtering -----------------------------------------------------


def test_filter_by_request_id(client):
    _create_job(client, "req-a", event_type="actor.created")
    other = _create_job(client, "req-b", event_type="actor.created")
    _create_job(client, "req-c", event_type="actor.created")

    body = _list(client, request_id="req-b").json()
    assert [i["id"] for i in body["items"]] == [other["id"]]
    assert body["count"] == 1


def test_filter_by_event_type(client):
    j1 = _create_job(client, "req-1", event_type="actor.created")
    _create_job(client, "req-2", event_type="content.created")
    j3 = _create_job(client, "req-3", event_type="actor.created")

    body = _list(client, event_type="actor.created").json()
    assert [i["id"] for i in body["items"]] == [j1["id"], j3["id"]]
    assert body["count"] == 2


def test_filter_by_resource_id(client):
    j1 = _create_job(client, "req-1", resource_id="res-a")
    _create_job(client, "req-2", resource_id="res-b")
    j3 = _create_job(client, "req-3", resource_id="res-a")

    body = _list(client, resource_id="res-a").json()
    assert [i["id"] for i in body["items"]] == [j1["id"], j3["id"]]
    assert body["count"] == 2


def test_filters_combine_as_logical_and(client):
    match = _create_job(
        client,
        "req-match",
        event_type="actor.created",
        resource_id="res-1",
    )
    _create_job(
        client, "req-other-type", event_type="content.created",
        resource_id="res-1",
    )
    _create_job(
        client, "req-other-res", event_type="actor.created",
        resource_id="res-2",
    )

    body = _list(
        client,
        request_id="req-match",
        event_type="actor.created",
        resource_id="res-1",
    ).json()
    assert [i["id"] for i in body["items"]] == [match["id"]]
    assert body["count"] == 1

    # A combination nothing satisfies is empty, not an error.
    assert _list(
        client,
        request_id="req-match",
        event_type="content.created",
    ).json() == {"items": [], "count": 0, "next_cursor": None}
    assert _list(
        client, event_type="actor.created", status="failed"
    ).json() == {"items": [], "count": 0, "next_cursor": None}


def test_exact_match_is_case_and_whitespace_sensitive(client):
    _create_job(
        client,
        "Request-1",
        event_type="Actor.Created",
        resource_id="Res-1",
    )

    for params in (
        {"request_id": "request-1"},
        {"request_id": "Request-1 "},
        {"request_id": " Request-1"},
        {"event_type": "actor.created"},
        {"event_type": " Actor.Created"},
        {"resource_id": "res-1"},
        {"resource_id": "Res-1\t"},
    ):
        body = _list(client, **params).json()
        assert body == {"items": [], "count": 0, "next_cursor": None}, params


def test_unknown_value_is_empty_not_error(client):
    _create_job(client, "known", event_type="actor.created")

    assert _list(client, request_id="no-such-request").json() == {
        "items": [], "count": 0, "next_cursor": None
    }
    assert _list(client, event_type="nothing.happened").json() == {
        "items": [], "count": 0, "next_cursor": None
    }
    assert _list(client, resource_id="res_does_not_exist").json() == {
        "items": [], "count": 0, "next_cursor": None
    }


# --- Status filter --------------------------------------------------------------


@pytest.mark.parametrize("state", ["pending", "running", "succeeded", "failed"])
def test_status_accepts_each_lifecycle_literal(client, state):
    _create_job(client, f"req-{state}")
    body = _list(client, status=state).json()
    # A created job is pending; the other literals simply match 0.
    assert body["count"] == (1 if state == "pending" else 0)


def test_status_filter_matches_exact_lifecycle_state(client, db_session):
    pending = _create_job(client, "p")
    succeeded = _create_job(client, "s")
    running = _create_job(client, "r")
    failed = _create_job(client, "f")
    client.post(f"{JOBS_URL}/{succeeded['id']}/run")

    def _set_state(job, state):
        db_session.execute(
            sa_update(AuditCheckpointJob)
            .where(AuditCheckpointJob.id == job["id"])
            .values(status=state)
        )
        db_session.commit()

    _set_state(running, "running")
    _set_state(failed, "failed")

    assert [
        i["id"] for i in _list(client, status="pending").json()["items"]
    ] == [pending["id"]]
    assert [
        i["id"] for i in _list(client, status="running").json()["items"]
    ] == [running["id"]]
    assert [
        i["id"] for i in _list(client, status="succeeded").json()["items"]
    ] == [succeeded["id"]]
    assert [
        i["id"] for i in _list(client, status="failed").json()["items"]
    ] == [failed["id"]]
    # Absent status means unfiltered across all four states.
    assert _list(client).json()["count"] == 4


@pytest.mark.parametrize(
    "value",
    ["", "   ", "PENDING", "Pending", "done", "pending ", " pending", "settled"],
)
def test_illegal_status_is_validation_error(client, value):
    resp = _list(client, status=value)
    assert resp.status_code == 422, value
    assert resp.json()["error"]["code"] == "validation_error"


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


def test_time_filters_combine_with_exact_filters(client, db_session):
    _insert_job(db_session, "acj_a1", "ra", T1, event_type="actor.created")
    _insert_job(
        db_session, "acj_b2", "rb", T2, event_type="content.created",
        resource_id="res-b",
    )

    body = _list(
        client, event_type="actor.created",
        **{"from": "2026-01-02T00:00:00Z"},
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}

    body = _list(
        client, request_id="rb", to="2026-01-03T00:00:00Z"
    ).json()
    assert [i["id"] for i in body["items"]] == ["acj_b2"]


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
    return [
        _create_job(client, f"req-{i:03d}", event_type="actor.created")
        for i in range(count)
    ]


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
    all_items, pages, count = _walk_pages(
        client, event_type="actor.created", limit=2
    )
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    assert {i["event_type"] for i in all_items} == {"actor.created"}


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


def test_stable_order_and_cursors_survive_restart(tmp_db_url):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app) as client:
        jobs = _setup_many_jobs(client, 5)
        first = _list(client, limit=2).json()
        assert first["next_cursor"] is not None
        page_two_ids = [i["id"] for i in first["items"]]

    # A fresh process over the same file keeps the same creation order; the
    # old cursor is invalidated by the per-process secret rotation.
    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        body = _list(client, limit=2).json()
        assert [i["id"] for i in body["items"]] == page_two_ids
        assert [i["id"] for i in body["items"]] == [
            j["id"] for j in jobs[:2]
        ]


# --- Cursor integrity ------------------------------------------------------------


def test_tampered_or_malformed_cursors_are_validation_errors(client):
    _setup_many_jobs(client, 3)
    good = _list(client, limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    for token in (
        "",
        "   ",
        "not-a-cursor",
        "aj1.onlytwoparts",
        "aj1.too.many.parts",
        "aj0.x.y",
        "aj2.x.y",
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
    for marker, kind in (
        ("ae1", pagination.AUDIT_EVENTS_CURSOR),
        ("cx1", pagination.CONTENT_EXPORT_JOBS_CURSOR),
        ("cl1", pagination.CLAIMS_CURSOR),
        ("ce1", pagination.CONTENT_EVIDENCE_CURSOR),
        ("v1", pagination.LINEAGE_CURSOR),
    ):
        token = pagination.encode_typed_cursor(
            secrets.token_bytes(32), kind, _cursor_claims_for(marker)
        )
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422, marker


def _cursor_claims_for(marker: str) -> dict:
    if marker == "ae1":
        return {
            "event_type": None, "resource_id": None, "from": None, "to": None,
            "limit": 1, "offset": 1,
        }
    if marker == "cx1":
        return {
            "content_id": None, "request_id": None, "status": None,
            "from": None, "to": None, "limit": 1, "offset": 1,
        }
    if marker == "cl1":
        return {
            "content_id": None, "actor_id": None, "claim_type": None,
            "payload_digest_hex": None, "limit": 1, "offset": 1,
        }
    if marker == "ce1":
        return {
            "content_id": "cnt_x", "evidence_type": None, "media_type": None,
            "limit": 1, "offset": 1,
        }
    return {
        "content_id": "cnt_x",
        "direction": "ancestors",
        "max_depth": 8,
        "min_depth": 1,
        "relation_type": None,
        "limit": 1,
        "offset": 1,
    }


def test_checkpoint_jobs_cursor_is_rejected_by_other_endpoints(client):
    _setup_many_jobs(client, 3)
    cursor = _list(client, limit=1).json()["next_cursor"]
    assert client.get(
        "/v1/audit-events", params={"limit": 1, "cursor": cursor}
    ).status_code == 422
    assert client.get(
        "/v1/content-export-jobs", params={"limit": 1, "cursor": cursor}
    ).status_code == 422


def test_cursor_bound_to_every_effective_query_parameter(client):
    _setup_many_jobs(client, 4)
    cursor = _list(client, limit=2).json()["next_cursor"]

    # A request that mints no binding filter but carries one mismatches.
    for params in (
        {"limit": 3},
        {"request_id": "req-000"},
        {"event_type": "actor.created"},
        {"resource_id": "res-1"},
        {"status": "pending"},
        {"from": "2026-01-01T00:00:00Z"},
        {"to": "2027-01-01T00:00:00Z"},
    ):
        resp = _list(client, cursor=cursor, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"

    # A filtered cursor resumed without (or with a changed) filter mismatches.
    filtered = _list(
        client, event_type="actor.created", limit=2
    ).json()["next_cursor"]
    assert _list(client, limit=2, cursor=filtered).status_code == 422
    assert _list(
        client, event_type="content.created", limit=2, cursor=filtered
    ).status_code == 422

    timed = _list(
        client, **{"from": "2026-01-01T00:00:00Z"}, limit=2
    ).json()["next_cursor"]
    assert _list(client, limit=2, cursor=timed).status_code == 422
    assert _list(
        client, **{"from": "2026-01-02T00:00:00Z"}, limit=2, cursor=timed
    ).status_code == 422


def test_cursor_binds_the_instant_not_its_notation(client, db_session):
    _insert_job(db_session, "acj_t2", "r2", T2)
    _insert_job(db_session, "acj_t3", "r3", T3)
    first = _list(client, **{"from": "2026-01-02T00:00:00Z"}, limit=1).json()
    assert first["next_cursor"] is not None
    resp = _list(
        client,
        **{"from": "2026-01-02T00:00:00+00:00"},
        limit=1,
        cursor=first["next_cursor"],
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["count"] == 2


# --- Parameter validation --------------------------------------------------------


@pytest.mark.parametrize("field", ["request_id", "event_type", "resource_id"])
@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_blank_filters_are_validation_errors(client, field, value):
    resp = _list(client, **{field: value})
    assert resp.status_code == 422, (field, value)
    assert resp.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize(
    "value", ["0", "101", "-1", "1.5", "abc", "8.0", "  2", "", "+1", "1e2"]
)
def test_illegal_limit_values_are_validation_errors(client, value):
    resp = _list(client, limit=value)
    assert resp.status_code == 422, value
    assert resp.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize(
    "suffix",
    [
        "request_id=a&request_id=b",
        "event_type=a&event_type=b",
        "resource_id=a&resource_id=b",
        "status=pending&status=failed",
        "from=2026-01-01T00:00:00Z&from=2026-01-02T00:00:00Z",
        "to=2026-01-01T00:00:00Z&to=2026-01-02T00:00:00Z",
        "limit=1&limit=2",
        "cursor=x&cursor=y",
    ],
)
def test_repeated_parameters_are_validation_errors(client, suffix):
    resp = client.get(f"{JOBS_URL}?{suffix}")
    assert resp.status_code == 422, suffix
    assert resp.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize(
    "suffix",
    [
        "content_id=cnt_1",
        "id=acj_1",
        "job_id=acj_1",
        "event_types=actor.created",
        "stat=pending",
        "limit=1&offset=2",
        "CURSOR=x",
    ],
)
def test_undeclared_parameters_are_validation_errors(client, suffix):
    resp = client.get(f"{JOBS_URL}?{suffix}")
    assert resp.status_code == 422, suffix
    assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_cursor_with_otherwise_valid_params_is_422(client):
    _setup_many_jobs(client, 2)
    resp = _list(client, status="pending", limit=10, cursor="garbage")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Empty request body -----------------------------------------------------------


@pytest.mark.parametrize(
    "body,headers",
    [
        (b'{"a":1}', {"content-type": "application/json"}),
        (b" ", {"content-type": "application/json"}),
        (b"x=1", {"content-type": "application/x-www-form-urlencoded"}),
        (b"\x00", {}),
        (b"{}", {"content-type": "application/json"}),
    ],
)
def test_any_get_body_is_validation_error(client, body, headers):
    resp = client.request("GET", JOBS_URL, content=body, headers=headers)
    assert resp.status_code == 422, repr(body)
    assert resp.json()["error"]["code"] == "validation_error"


def test_empty_get_body_succeeds(client):
    resp = client.request("GET", JOBS_URL)
    assert resp.status_code == 200, resp.text


@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE"])
def test_non_get_or_post_methods_are_405(client, method):
    resp = client.request(method, JOBS_URL)
    assert resp.status_code == 405, method
    assert resp.json()["error"]["code"] == "method_not_allowed"


def test_post_still_creates(client):
    resp = client.post(JOBS_URL, json={"request_id": "req-post"})
    assert resp.status_code == 201, resp.text
    assert resp.json()["request_id"] == "req-post"


# --- Body framing ------------------------------------------------------------------


def test_success_body_is_compact_json_with_single_trailing_newline(client):
    _create_job(client, "req-x")

    raw = _list(client).content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    # Compact separators: no whitespace after commas or colons.
    assert b", " not in raw
    assert b": " not in raw
    # The same payload parses to the compact JSON the client sees.
    assert json.loads(raw.decode("utf-8"))["count"] == 1


def test_non_ascii_is_emitted_unescaped_as_utf8(client):
    _create_job(client, "req-é-û")

    raw = _list(client).content
    assert "req-é-û".encode("utf-8") in raw
    assert b"\\u" not in raw
    assert json.loads(raw.decode("utf-8"))["items"][0]["request_id"] == "req-é-û"


def test_booleans_null_and_integers_rendered_literally(client):
    _create_job(client, "req-l")
    body = _list(client).json()
    item = body["items"][0]
    assert item["started_at"] is None
    assert item["finished_at"] is None
    assert item["result"] is None
    assert item["error"] is None
    assert isinstance(body["count"], int)
    assert body["next_cursor"] is None


# --- Read-only guarantee -----------------------------------------------------------


def test_queries_and_failures_write_no_jobs_or_audit_events(client, db_session):
    jobs = _setup_many_jobs(client, 4)
    request_id = jobs[0]["request_id"]

    def counts():
        return (
            db_session.scalar(
                select(func.count()).select_from(AuditCheckpointJob)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    before = counts()

    # Successful filtered, timed, and paginated reads.
    cursor = None
    for _ in range(8):
        params = {"event_type": "actor.created", "limit": 1}
        if cursor is not None:
            params["cursor"] = cursor
        resp = _list(client, **params)
        assert resp.status_code == 200, resp.text
        cursor = resp.json()["next_cursor"]
        if cursor is None:
            break
    _list(client)
    _list(client, request_id=request_id)
    _list(client, request_id="missing")
    _list(client, resource_id="res_missing")
    _list(client, **{"from": "2026-01-01T00:00:00Z", "to": "2026-01-02T00:00:00Z"})

    # Failed reads must not write anything either.
    _list(client, status="nope")
    _list(client, request_id=" ")
    _list(client, limit=0)
    _list(client, cursor="tampered")
    _list(client, **{"from": "not-a-time"})
    _list(client, **{"from": "2027-01-01T00:00:00Z",
                     "to": "2026-01-01T00:00:00Z"})
    client.get(f"{JOBS_URL}?limit=1&limit=2")
    client.get(f"{JOBS_URL}?unknown=1")
    client.request("GET", JOBS_URL, content=b"body")
    client.request("DELETE", JOBS_URL)

    db_session.expire_all()
    assert counts() == before
