"""Tests for the audit checkpoint job queue summary.

Covers ``GET /v1/audit-checkpoint-jobs/summary``:

* empty queue -> all four counts zero, oldest-pending id and wait null;
* mixed states -> counts cover every existing job (settled jobs stay in
  their ``succeeded``/``failed`` counts) and the oldest *pending* job is
  picked in stable creation order;
* the wait time is the current UTC instant minus the job's creation time,
  floored to whole integer seconds;
* the same persisted state read across an app restart keeps the counts and
  the stable oldest id, while the wait time is recomputed per read;
* the success body is compact UTF-8 JSON terminated by exactly one newline
  with integral numbers only;
* any non-empty body (whitespace, malformed JSON, an object, extra fields)
  or any query parameter (unknown, blank, repeated) is a 422
  ``validation_error``, rejected before any job is read;
* non-GET methods are 405 method_not_allowed;
* an unreadable database or an internal summary failure is a 503
  service_unavailable carrying a machine-readable reason;
* success, empty results, 422/405/503 failures are strictly read-only: no
  job, resource, or audit row is created or modified.

All fixtures are deterministic and offline (in-memory and temporary-file
SQLite, no network).
"""

from __future__ import annotations

import json
import math
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update as sa_update

from provenance.app import create_app
from provenance.api import get_db
from provenance.config import Settings
from provenance.models import (
    EVENT_AUDIT_CHECKPOINT_JOB_CREATED,
    AuditCheckpointJob,
    AuditEvent,
)
from provenance import service
from provenance.time_utils import utc_now

SUMMARY_URL = "/v1/audit-checkpoint-jobs/summary"
JOBS_URL = "/v1/audit-checkpoint-jobs"


def _create_job(client, request_id, **filters):
    resp = client.post(
        JOBS_URL, json={"request_id": request_id, **filters}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _set_status(db_session, job_id, status):
    db_session.execute(
        sa_update(AuditCheckpointJob)
        .where(AuditCheckpointJob.id == job_id)
        .values(status=status)
    )
    db_session.commit()


# --- empty queue ------------------------------------------------------------


def test_summary_empty_queue_is_all_zero_and_null(client, db_session):
    resp = client.get(SUMMARY_URL)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/json")
    # Compact JSON terminated by exactly one newline.
    assert resp.content == (
        b'{"pending":0,"running":0,"succeeded":0,"failed":0,'
        b'"oldest_pending_id":null,"oldest_pending_wait_seconds":null}\n'
    )
    body = resp.json()
    assert body == {
        "pending": 0,
        "running": 0,
        "succeeded": 0,
        "failed": 0,
        "oldest_pending_id": None,
        "oldest_pending_wait_seconds": None,
    }
    # The read writes nothing at all.
    assert db_session.execute(select(AuditCheckpointJob)).scalars().all() == []
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


# --- counts over mixed states ------------------------------------------------


def test_summary_counts_cover_all_states_including_settled(client, db_session):
    pending_a = _create_job(client, "req-pending-a")
    pending_b = _create_job(client, "req-pending-b")
    running = _create_job(client, "req-running")
    succeeded = _create_job(client, "req-succeeded")
    failed = _create_job(client, "req-failed")

    _set_status(db_session, running["id"], "running")
    run = client.post(f"{JOBS_URL}/{succeeded['id']}/run")
    assert run.status_code == 200
    assert run.json()["status"] == "succeeded"

    def _boom(session, job):
        raise RuntimeError("simulated checkpoint failure")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(service, "list_audit_events", _boom)
    try:
        failed_run = client.post(f"{JOBS_URL}/{failed['id']}/run")
        assert failed_run.status_code == 200
        assert failed_run.json()["status"] == "failed"
    finally:
        monkeypatch.undo()

    events_before = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    resp = client.get(SUMMARY_URL)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Settled jobs stay in their counts; every existing job is covered.
    assert body["pending"] == 2
    assert body["running"] == 1
    assert body["succeeded"] == 1
    assert body["failed"] == 1
    for key in ("pending", "running", "succeeded", "failed"):
        assert isinstance(body[key], int)

    # The oldest pending job in stable creation order is reported.
    assert body["oldest_pending_id"] == pending_a["id"]
    wait = body["oldest_pending_wait_seconds"]
    assert isinstance(wait, int)
    assert wait >= 0

    # Strictly read-only: no job state changed and no audit event was added.
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )
    assert (
        client.get(f"{JOBS_URL}/{pending_a['id']}").json()["status"]
        == "pending"
    )
    assert (
        client.get(f"{JOBS_URL}/{pending_b['id']}").json()["status"]
        == "pending"
    )


def test_summary_oldest_pending_skips_non_pending_jobs(client, db_session):
    oldest = _create_job(client, "req-oldest-settled")
    middle = _create_job(client, "req-middle-running")
    youngest_pending = _create_job(client, "req-youngest")

    _set_status(db_session, oldest["id"], "succeeded")
    _set_status(db_session, middle["id"], "running")

    body = client.get(SUMMARY_URL).json()
    assert body["pending"] == 1
    # Older non-pending jobs are never reported as the queue head.
    assert body["oldest_pending_id"] == youngest_pending["id"]


def test_summary_no_pending_reports_null_oldest(client):
    job = _create_job(client, "req-settled-only")
    run = client.post(f"{JOBS_URL}/{job['id']}/run")
    assert run.status_code == 200

    body = client.get(SUMMARY_URL).json()
    assert body["pending"] == 0
    assert body["succeeded"] == 1
    assert body["oldest_pending_id"] is None
    assert body["oldest_pending_wait_seconds"] is None


# --- wait time ---------------------------------------------------------------


def test_summary_wait_seconds_floor_to_whole_seconds(client, db_session):
    job = _create_job(client, "req-wait")

    # Backdate the creation time by a non-integral number of seconds.
    created_at = utc_now() - timedelta(seconds=3723.6)
    db_session.execute(
        sa_update(AuditCheckpointJob)
        .where(AuditCheckpointJob.id == job["id"])
        .values(created_at=created_at)
    )
    db_session.commit()

    before = utc_now()
    body = client.get(SUMMARY_URL).json()
    after = utc_now()

    assert body["oldest_pending_id"] == job["id"]
    wait = body["oldest_pending_wait_seconds"]
    assert isinstance(wait, int)
    # Floor of (read instant - created_at) in whole seconds, bounded by the
    # instants bracketing the request.
    assert wait >= math.floor((before - created_at).total_seconds())
    assert wait <= math.floor((after - created_at).total_seconds())
    assert 3723 <= wait <= 3725


def test_summary_wait_seconds_grows_between_reads(client, db_session):
    job = _create_job(client, "req-wait-grows")

    created_at = utc_now() - timedelta(seconds=10)
    db_session.execute(
        sa_update(AuditCheckpointJob)
        .where(AuditCheckpointJob.id == job["id"])
        .values(created_at=created_at)
    )
    db_session.commit()

    first = client.get(SUMMARY_URL).json()
    second = client.get(SUMMARY_URL).json()
    # Nothing is frozen: each read recomputes the wait from its own instant.
    assert second["oldest_pending_wait_seconds"] >= first[
        "oldest_pending_wait_seconds"
    ]
    assert first["oldest_pending_id"] == second["oldest_pending_id"] == job["id"]


# --- restart stability -------------------------------------------------------


def test_summary_counts_and_oldest_id_stable_across_restart(tmp_db_url):
    app1 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app1) as first:
        oldest = _create_job(first, "req-r-oldest")
        _create_job(first, "req-r-youngest")
        settled = _create_job(first, "req-r-settled")
        run = first.post(f"{JOBS_URL}/{settled['id']}/run")
        assert run.status_code == 200
        before_restart = first.get(SUMMARY_URL).json()

    # A brand-new process/app over the same database file.
    app2 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app2) as second:
        after_restart = second.get(SUMMARY_URL).json()

    assert after_restart["pending"] == before_restart["pending"] == 2
    assert after_restart["running"] == before_restart["running"] == 0
    assert after_restart["succeeded"] == before_restart["succeeded"] == 1
    assert after_restart["failed"] == before_restart["failed"] == 0
    # The stable identifier survives the restart; the wait time is
    # recomputed at each read's own instant.
    assert after_restart["oldest_pending_id"] == oldest["id"]
    assert (
        after_restart["oldest_pending_wait_seconds"]
        >= before_restart["oldest_pending_wait_seconds"]
    )


# --- wire format -------------------------------------------------------------


def test_summary_body_is_compact_json_with_single_trailing_newline(client):
    _create_job(client, "req-wire")

    resp = client.get(SUMMARY_URL)
    assert resp.status_code == 200, resp.text
    raw = resp.content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    # Compact separators: no incidental whitespace inside the document.
    assert b" " not in raw
    assert b": " not in raw
    body = json.loads(raw.decode("utf-8"))
    assert list(body) == [
        "pending",
        "running",
        "succeeded",
        "failed",
        "oldest_pending_id",
        "oldest_pending_wait_seconds",
    ]
    # Integral numbers only: no floats, no -0.0, no non-finite values.
    for key in ("pending", "running", "succeeded", "failed"):
        assert isinstance(body[key], int)
    assert isinstance(body["oldest_pending_wait_seconds"], int)


# --- request validation ------------------------------------------------------


@pytest.mark.parametrize(
    "send",
    [
        # Any non-empty body is rejected, even whitespace.
        lambda c: c.request("GET", SUMMARY_URL, content=b"   "),
        # Malformed JSON is a validation error, not a parser crash.
        lambda c: c.request(
            "GET",
            SUMMARY_URL,
            content=b"{not json",
            headers={"content-type": "application/json"},
        ),
        # An empty JSON object is still a non-empty body.
        lambda c: c.request("GET", SUMMARY_URL, content=b"{}"),
        # Any extra field rides a non-empty body.
        lambda c: c.request("GET", SUMMARY_URL, content=b'{"unexpected": true}'),
        # No query parameters are accepted in any shape.
        lambda c: c.get(SUMMARY_URL + "?x=1"),
        lambda c: c.get(SUMMARY_URL + "?status=pending"),
        lambda c: c.get(SUMMARY_URL + "?limit=10"),
        lambda c: c.get(SUMMARY_URL + "?x="),
        lambda c: c.get(SUMMARY_URL + "?x=1&x=2"),
        # A body together with a query parameter is still a 422.
        lambda c: c.request("GET", SUMMARY_URL + "?x=1", content=b"{}"),
    ],
)
def test_summary_validation_errors_422_write_nothing(client, db_session, send):
    job = _create_job(client, "req-guarded")
    events_before = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    resp = send(client)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"

    # A rejected request reads and writes nothing: the queue is intact and
    # no audit event was added.
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )
    view = client.get(f"{JOBS_URL}/{job['id']}").json()
    assert view["status"] == "pending"

    # A well-formed read still succeeds afterwards.
    ok = client.get(SUMMARY_URL)
    assert ok.status_code == 200
    assert ok.json()["pending"] == 1
    assert ok.json()["oldest_pending_id"] == job["id"]


# --- method handling ---------------------------------------------------------


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_non_get_methods_are_405_and_write_nothing(client, db_session, method):
    job = _create_job(client, "req-method")
    events_before = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    resp = getattr(client, method)(SUMMARY_URL)
    assert resp.status_code == 405, resp.text
    assert resp.json()["error"]["code"] == "method_not_allowed"

    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )
    assert client.get(f"{JOBS_URL}/{job['id']}").json()["status"] == "pending"


# --- 503 service_unavailable --------------------------------------------------


def _client_with_dropped_table(file_app):
    # A TestClient created without entering its context never runs the
    # lifespan startup, so dropping a table here leaves the schema unreadable
    # for the summary instead of being silently recreated.
    from sqlalchemy import text

    with file_app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE audit_checkpoint_jobs"))
    return TestClient(file_app)


def test_unreadable_database_returns_503_with_reason(file_app):
    client = _client_with_dropped_table(file_app)
    resp = client.get(SUMMARY_URL)
    assert resp.status_code == 503, resp.text
    error = resp.json()["error"]
    assert error["code"] == "service_unavailable"
    assert "message" in error
    assert error["details"]["reason"] == "database_unavailable"
    # The failure created no audit event; the audit table still exists.
    from sqlalchemy import text

    with file_app.state.engine.connect() as conn:
        audit_total = conn.execute(
            text("SELECT COUNT(*) FROM audit_events")
        ).scalar_one()
    assert audit_total == 0


def test_summary_validation_runs_before_the_unreadable_database(file_app):
    # Empty or illegal input is rejected first; the database is never read,
    # so an unreadable database still answers these with 422 rather than
    # 503. The valid empty request is the one that reaches the queries and
    # gets the 503.
    client = _client_with_dropped_table(file_app)

    whitespace = client.request("GET", SUMMARY_URL, content=b"  ")
    assert whitespace.status_code == 422, whitespace.text
    unknown_param = client.get(SUMMARY_URL + "?x=1")
    assert unknown_param.status_code == 422, unknown_param.text
    repeated = client.get(SUMMARY_URL + "?x=1&x=2")
    assert repeated.status_code == 422, repeated.text

    valid = client.get(SUMMARY_URL)
    assert valid.status_code == 503, valid.text


class _BrokenSummarySession:
    """A session whose summary reads raise a non-SQLAlchemy failure."""

    def execute(self, *_args, **_kwargs):
        raise RuntimeError("summary machinery broken")

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_internal_summary_failure_is_503_with_reason(app, client):
    # A non-database internal failure while assembling the summary is still
    # the existing-structure 503 carrying a reason, never a partial body or
    # a 500.
    app.dependency_overrides[get_db] = lambda: _BrokenSummarySession()
    try:
        resp = client.get(SUMMARY_URL)
    finally:
        app.dependency_overrides.pop(get_db, None)
    assert resp.status_code == 503, resp.text
    error = resp.json()["error"]
    assert error["code"] == "service_unavailable"
    assert error["details"]["reason"] == "internal_error"


# --- read-only consistency with the checkpoint package -----------------------


def test_summary_counts_match_lifecycle_states_seen_on_package_runs(client):
    # Two pending jobs and one settled (succeeded) job: the summary counts
    # reflect exactly the states the job detail/package routes serve, and
    # the summary never returns any job result or raw material.
    pending = _create_job(client, "req-pkg-pending")
    settled = _create_job(client, "req-pkg-settled")
    run = client.post(f"{JOBS_URL}/{settled['id']}/run")
    assert run.status_code == 200
    assert run.json()["result"]["checkpoint"] is not None

    body = client.get(SUMMARY_URL).json()
    assert body["pending"] == 1
    assert body["succeeded"] == 1
    assert body["oldest_pending_id"] == pending["id"]
    # The summary carries only counts, an id, and a wait: no result, filter,
    # checkpoint, event, or other raw material ever appears.
    assert set(body) == {
        "pending",
        "running",
        "succeeded",
        "failed",
        "oldest_pending_id",
        "oldest_pending_wait_seconds",
    }
    # Creation events are untouched by the summary read.
    assert client.get(
        f"/v1/audit-events?resource_id={pending['id']}"
    ).json()["count"] == 1
    assert client.get(
        f"/v1/audit-events?resource_id={pending['id']}"
        f"&event_type={EVENT_AUDIT_CHECKPOINT_JOB_CREATED}"
    ).json()["count"] == 1
