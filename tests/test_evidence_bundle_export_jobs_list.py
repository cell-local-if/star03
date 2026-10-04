"""Tests for the read-only evidence bundle export job search endpoint.

Covers GET /v1/evidence-bundle-export-jobs: the response shape carries
exactly {"items", "count", "next_cursor"} with each item being the existing
single-job public view (nine fields, including a settled result package),
jobs follow stable creation order (created_at with the monotonic seq
tiebreaker, stable across a restart), evidence_bundle_id/request_id are
exact combinable case/whitespace-sensitive filters (status accepts only the
four lifecycle literals and is absent by default), from/to are strict RFC
3339 UTC inclusive created_at bounds (from must not exceed to), limit is
1..100 defaulting to 50, opaque HMAC cursors (their own ej1 family) bind
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
from provenance.models import AuditEvent, EvidenceBundleExportJob
from tests.test_evidence_bundle_exchange import _setup_bundle

JOBS_URL = "/v1/evidence-bundle-export-jobs"

#: The exact public single-job field set.
JOB_KEYS = {
    "id",
    "evidence_bundle_id",
    "request_id",
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


def _create_job(client, evidence_bundle_id, request_id):
    resp = client.post(
        JOBS_URL,
        json={
            "evidence_bundle_id": evidence_bundle_id,
            "request_id": request_id,
        },
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
    evidence_bundle_id,
    request_id,
    created_at,
    status="pending",
):
    """Insert one job row directly at a fixed instant (time-range tests)."""
    db_session.add(
        EvidenceBundleExportJob(
            id=job_id,
            evidence_bundle_id=evidence_bundle_id,
            request_id=request_id,
            status=status,
            created_at=created_at,
        )
    )
    db_session.commit()


# --- Response shape, fields, and ordering -------------------------------------


def test_response_shape_and_item_fields(client):
    _, _, bundle = _setup_bundle(client)
    _create_job(client, bundle["id"], "req-1")

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
    _, _, bundle = _setup_bundle(client)
    job = _create_job(client, bundle["id"], "req-view")
    settled = client.post(f"{JOBS_URL}/{job['id']}/run").json()

    listed = _list(client).json()["items"][0]
    # The list item is exactly the GET single-resource view.
    single = client.get(f"{JOBS_URL}/{job['id']}").json()
    assert listed == single == settled
    # A settled job carries its exchange package, never raw material.
    assert set(listed["result"]) == {"snapshot", "manifest"}


def test_failed_job_item_carries_error_and_null_result(client, monkeypatch):
    _, _, bundle = _setup_bundle(client)
    job = _create_job(client, bundle["id"], "req-fail")

    def _boom(session, evidence_bundle_id):
        raise RuntimeError("simulated export failure")

    monkeypatch.setattr(service, "get_evidence_bundle_exchange", _boom)
    assert client.post(f"{JOBS_URL}/{job['id']}/run").status_code == 200
    monkeypatch.undo()

    listed = _list(client, status="failed").json()["items"][0]
    assert listed["result"] is None
    assert listed["error"] == "evidence_bundle_export_failed"
    # The failed view is exactly the single-resource view.
    assert listed == client.get(f"{JOBS_URL}/{job['id']}").json()


def test_jobs_follow_stable_creation_order(client):
    _, _, bundle = _setup_bundle(client)
    j1 = _create_job(client, bundle["id"], "req-1")
    j2 = _create_job(client, bundle["id"], "req-2")
    j3 = _create_job(client, bundle["id"], "req-3")

    items = _list(client).json()["items"]
    assert [i["id"] for i in items] == [j1["id"], j2["id"], j3["id"]]
    for earlier, later in zip(items, items[1:]):
        assert earlier["created_at"] <= later["created_at"]


def test_empty_database_is_an_empty_collection(client):
    resp = _list(client)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


# --- Exact-match filtering -----------------------------------------------------


def test_filter_by_evidence_bundle_id(client):
    _, claim, bundle = _setup_bundle(client)
    from tests.test_evidence_bundle_exchange import _create_bundle

    other_bundle = _create_bundle(client, claim["id"], "other-evidence")
    j1 = _create_job(client, bundle["id"], "req-1")
    j2 = _create_job(client, other_bundle["id"], "req-2")
    j3 = _create_job(client, bundle["id"], "req-3")

    body = _list(client, evidence_bundle_id=bundle["id"]).json()
    assert [i["id"] for i in body["items"]] == [j1["id"], j3["id"]]
    assert body["count"] == 2

    body = _list(client, evidence_bundle_id=other_bundle["id"]).json()
    assert [i["id"] for i in body["items"]] == [j2["id"]]
    assert body["count"] == 1


def test_filter_by_request_id(client):
    _, _, bundle = _setup_bundle(client)
    _create_job(client, bundle["id"], "req-a")
    other = _create_job(client, bundle["id"], "req-b")
    _create_job(client, bundle["id"], "req-c")

    body = _list(client, request_id="req-b").json()
    assert [i["id"] for i in body["items"]] == [other["id"]]
    assert body["count"] == 1


def test_filters_combine_as_logical_and(client):
    _, claim, bundle = _setup_bundle(client)
    from tests.test_evidence_bundle_exchange import _create_bundle

    other_bundle = _create_bundle(client, claim["id"], "other-evidence")
    match = _create_job(client, bundle["id"], "req-match")
    _create_job(client, other_bundle["id"], "req-other-bundle")
    _create_job(client, bundle["id"], "req-other-key")

    body = _list(
        client,
        evidence_bundle_id=bundle["id"],
        request_id="req-match",
    ).json()
    assert [i["id"] for i in body["items"]] == [match["id"]]
    assert body["count"] == 1

    # A combination nothing satisfies is empty, not an error.
    assert _list(
        client,
        evidence_bundle_id=bundle["id"],
        request_id="req-other-bundle",
    ).json() == {"items": [], "count": 0, "next_cursor": None}
    assert _list(
        client, evidence_bundle_id=bundle["id"], status="failed"
    ).json() == {"items": [], "count": 0, "next_cursor": None}


def test_exact_match_is_case_and_whitespace_sensitive(client):
    _, _, bundle = _setup_bundle(client)
    _create_job(client, bundle["id"], "Request-1")

    for params in (
        {"request_id": "request-1"},
        {"request_id": "Request-1 "},
        {"request_id": " Request-1"},
        {"evidence_bundle_id": bundle["id"].upper()},
        {"evidence_bundle_id": bundle["id"] + " "},
        {"evidence_bundle_id": " " + bundle["id"]},
    ):
        body = _list(client, **params).json()
        assert body == {"items": [], "count": 0, "next_cursor": None}, params


def test_unknown_value_is_empty_not_error(client):
    _, _, bundle = _setup_bundle(client)
    _create_job(client, bundle["id"], "known")

    assert _list(client, request_id="no-such-request").json() == {
        "items": [], "count": 0, "next_cursor": None
    }
    assert _list(client, evidence_bundle_id="evb_does_not_exist").json() == {
        "items": [], "count": 0, "next_cursor": None
    }


# --- Status filter --------------------------------------------------------------


@pytest.mark.parametrize("state", ["pending", "running", "succeeded", "failed"])
def test_status_accepts_each_lifecycle_literal(client, state):
    _, _, bundle = _setup_bundle(client)
    _create_job(client, bundle["id"], f"req-{state}")
    body = _list(client, status=state).json()
    # A created job is pending; the other literals simply match 0.
    assert body["count"] == (1 if state == "pending" else 0)


def test_status_filter_matches_exact_lifecycle_state(client, db_session):
    _, _, bundle = _setup_bundle(client)
    pending = _create_job(client, bundle["id"], "p")
    succeeded = _create_job(client, bundle["id"], "s")
    running = _create_job(client, bundle["id"], "r")
    failed = _create_job(client, bundle["id"], "f")
    client.post(f"{JOBS_URL}/{succeeded['id']}/run")

    def _set_state(job, state):
        db_session.execute(
            sa_update(EvidenceBundleExportJob)
            .where(EvidenceBundleExportJob.id == job["id"])
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
    _insert_job(db_session, "exj_t1", "evb_a", "r1", T1)
    _insert_job(db_session, "exj_t2", "evb_a", "r2", T2)
    _insert_job(db_session, "exj_t3", "evb_a", "r3", T3)

    body = _list(client, **{"from": "2026-01-02T12:30:00Z"}).json()
    assert [i["id"] for i in body["items"]] == ["exj_t2", "exj_t3"]
    assert body["count"] == 2


def test_to_bound_is_inclusive(client, db_session):
    _insert_job(db_session, "exj_t1", "evb_a", "r1", T1)
    _insert_job(db_session, "exj_t2", "evb_a", "r2", T2)
    _insert_job(db_session, "exj_t3", "evb_a", "r3", T3)

    body = _list(client, to="2026-01-02T12:30:00Z").json()
    assert [i["id"] for i in body["items"]] == ["exj_t1", "exj_t2"]
    assert body["count"] == 2


def test_from_equal_to_matches_exact_instant(client, db_session):
    _insert_job(db_session, "exj_t1", "evb_a", "r1", T1)
    _insert_job(db_session, "exj_t2", "evb_a", "r2", T2)
    body = _list(
        client,
        **{"from": "2026-01-02T12:30:00Z", "to": "2026-01-02T12:30:00Z"},
    ).json()
    assert [i["id"] for i in body["items"]] == ["exj_t2"]
    assert body["count"] == 1


def test_utc_offset_notation_and_fractional_seconds_accepted(client, db_session):
    _insert_job(db_session, "exj_t2", "evb_a", "r2", T2)
    _insert_job(db_session, "exj_t3", "evb_a", "r3", T3)
    body = _list(
        client,
        **{
            "from": "2026-01-02T12:30:00+00:00",
            "to": "2026-01-03T23:59:59.000Z",
        },
    ).json()
    assert [i["id"] for i in body["items"]] == ["exj_t2", "exj_t3"]


def test_time_filters_combine_with_exact_filters(client, db_session):
    _insert_job(db_session, "exj_a1", "evb_a", "ra", T1)
    _insert_job(db_session, "exj_b2", "evb_b", "rb", T2)

    body = _list(
        client, evidence_bundle_id="evb_a",
        **{"from": "2026-01-02T00:00:00Z"},
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}

    body = _list(
        client, request_id="rb", to="2026-01-03T00:00:00Z"
    ).json()
    assert [i["id"] for i in body["items"]] == ["exj_b2"]


def test_from_later_than_to_is_validation_error(client, db_session):
    _insert_job(db_session, "exj_t1", "evb_a", "r1", T1)
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
    _, _, bundle = _setup_bundle(client)
    return [
        _create_job(client, bundle["id"], f"req-{i:03d}")
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
    _, _, bundle = _setup_bundle(client)
    jobs = [
        _create_job(client, bundle["id"], f"req-{i:03d}") for i in range(5)
    ]
    all_items, pages, count = _walk_pages(
        client, evidence_bundle_id=bundle["id"], limit=2
    )
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    assert {i["evidence_bundle_id"] for i in all_items} == {bundle["id"]}
    assert [i["id"] for i in all_items] == [j["id"] for j in jobs]


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
        app.state.evidence_bundle_export_jobs_cursor_secret,
        pagination.EVIDENCE_BUNDLE_EXPORT_JOBS_CURSOR,
        {
            "evidence_bundle_id": None,
            "request_id": None,
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
        page_one_ids = [i["id"] for i in first["items"]]

    # A fresh process over the same file keeps the same creation order; the
    # old cursor is invalidated by the per-process secret rotation.
    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        body = _list(client, limit=2).json()
        assert [i["id"] for i in body["items"]] == page_one_ids
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
        "ej1.onlytwoparts",
        "ej1.too.many.parts",
        "ej0.x.y",
        "ej2.x.y",
        tampered,
    ):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422, repr(token)
        assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_signed_with_wrong_secret_is_rejected(client):
    _setup_many_jobs(client, 2)
    foreign = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.EVIDENCE_BUNDLE_EXPORT_JOBS_CURSOR,
        {
            "evidence_bundle_id": None,
            "request_id": None,
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
        ("aj1", pagination.AUDIT_CHECKPOINT_JOBS_CURSOR),
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
    if marker == "aj1":
        return {
            "request_id": None, "event_type": None, "resource_id": None,
            "status": None, "from": None, "to": None, "limit": 1, "offset": 1,
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


def test_export_jobs_cursor_is_rejected_by_other_endpoints(client):
    _setup_many_jobs(client, 3)
    cursor = _list(client, limit=1).json()["next_cursor"]
    assert client.get(
        "/v1/audit-events", params={"limit": 1, "cursor": cursor}
    ).status_code == 422
    assert client.get(
        "/v1/content-export-jobs", params={"limit": 1, "cursor": cursor}
    ).status_code == 422
    assert client.get(
        "/v1/audit-checkpoint-jobs", params={"limit": 1, "cursor": cursor}
    ).status_code == 422


def test_cursor_bound_to_every_effective_query_parameter(client):
    jobs = _setup_many_jobs(client, 4)
    bundle_id = jobs[0]["evidence_bundle_id"]
    cursor = _list(client, limit=2).json()["next_cursor"]

    # A request that mints no binding filter but carries one mismatches.
    for params in (
        {"limit": 3},
        {"evidence_bundle_id": "evb_x"},
        {"request_id": "req-000"},
        {"status": "pending"},
        {"from": "2026-01-01T00:00:00Z"},
        {"to": "2027-01-01T00:00:00Z"},
    ):
        resp = _list(client, cursor=cursor, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"

    # A filtered cursor resumed without (or with a changed) filter mismatches.
    filtered = _list(
        client, evidence_bundle_id=bundle_id, limit=2
    ).json()["next_cursor"]
    assert filtered is not None
    assert _list(client, limit=2, cursor=filtered).status_code == 422
    assert _list(
        client, evidence_bundle_id="evb_other", limit=2, cursor=filtered
    ).status_code == 422

    timed = _list(
        client, **{"from": "2026-01-01T00:00:00Z"}, limit=2
    ).json()["next_cursor"]
    assert _list(client, limit=2, cursor=timed).status_code == 422
    assert _list(
        client, **{"from": "2026-01-02T00:00:00Z"}, limit=2, cursor=timed
    ).status_code == 422


def test_cursor_binds_the_instant_not_its_notation(client, db_session):
    _insert_job(db_session, "exj_t2", "evb_a", "r2", T2)
    _insert_job(db_session, "exj_t3", "evb_a", "r3", T3)
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


@pytest.mark.parametrize("field", ["evidence_bundle_id", "request_id"])
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
        "evidence_bundle_id=a&evidence_bundle_id=b",
        "request_id=a&request_id=b",
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
        "id=exj_1",
        "job_id=exj_1",
        "evidence_bundle_ids=evb_1",
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
    _, _, bundle = _setup_bundle(client)
    resp = client.post(
        JOBS_URL,
        json={"evidence_bundle_id": bundle["id"], "request_id": "req-post"},
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["request_id"] == "req-post"


# --- Body framing ------------------------------------------------------------------


def test_success_body_is_compact_json_with_single_trailing_newline(client):
    _, _, bundle = _setup_bundle(client)
    _create_job(client, bundle["id"], "req-x")

    raw = _list(client).content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    # Compact separators: no whitespace after commas or colons.
    assert b", " not in raw
    assert b": " not in raw
    # The same payload parses to the compact JSON the client sees.
    assert json.loads(raw.decode("utf-8"))["count"] == 1


def test_non_ascii_is_emitted_unescaped_as_utf8(client):
    _, _, bundle = _setup_bundle(client)
    _create_job(client, bundle["id"], "req-é-û")

    raw = _list(client).content
    assert "req-é-û".encode("utf-8") in raw
    assert b"\\u" not in raw
    assert json.loads(raw.decode("utf-8"))["items"][0]["request_id"] == "req-é-û"


def test_booleans_null_and_integers_rendered_literally(client):
    _, _, bundle = _setup_bundle(client)
    _create_job(client, bundle["id"], "req-l")
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
                select(func.count()).select_from(EvidenceBundleExportJob)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    before = counts()

    # Successful filtered, timed, and paginated reads.
    cursor = None
    for _ in range(8):
        params = {"status": "pending", "limit": 1}
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
    _list(client, evidence_bundle_id="evb_missing")
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
