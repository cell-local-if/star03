"""Tests for the content export job queue summary.

Covers ``GET /v1/content-export-jobs/summary``:

* empty queue -> all four counts zero, oldest-pending id and wait null;
* counts cover every persisted job in each of the four states, with
  settled jobs retained in their terminal-state counts;
* the oldest pending job's stable id and floored whole-second wait time
  are returned in the same response, selected in stable creation order;
* the body is compact UTF-8 JSON terminated by exactly one newline, with
  integral counts and wait seconds (no -0.0, no non-finite values);
* any request body (whitespace, malformed JSON, an object) or any query
  parameter (unknown, blank, repeated) is a 422 ``validation_error``
  raised before any job is read -- no partial result, no writes;
* the read is strictly read-only: success, empty, and rejected reads
  create or modify no job, resource, or audit event;
* counts and the stable oldest-pending id survive an app restart over
  the same database, while the wait time is recomputed per read.

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

from provenance import service
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import AuditEvent, ContentExportJob
from provenance.time_utils import utc_now
from tests.helpers import create_actor

SUMMARY_URL = "/v1/content-export-jobs/summary"


def _digest(name: str) -> str:
    import hashlib

    return hashlib.sha256(f"summary-{name}".encode()).hexdigest()


def _create_content(client, name, actor_id="org-1"):
    resp = client.post(
        "/v1/contents",
        json={
            "digest_algorithm": "sha256",
            "digest_hex": _digest(f"content-{name}"),
            "media_type": "image/png",
            "title": name,
            "actor_id": actor_id,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_job(client, content_id, request_id):
    resp = client.post(
        "/v1/content-export-jobs",
        json={"content_id": content_id, "request_id": request_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _set_status(db_session, job_id, status):
    db_session.execute(
        sa_update(ContentExportJob)
        .where(ContentExportJob.id == job_id)
        .values(status=status)
    )
    db_session.commit()


def _set_created_at(db_session, job_id, created_at):
    db_session.execute(
        sa_update(ContentExportJob)
        .where(ContentExportJob.id == job_id)
        .values(created_at=created_at)
    )
    db_session.commit()


def _audit_events(db_session):
    return db_session.execute(select(AuditEvent)).scalars().all()


# --- empty queue ------------------------------------------------------------


def test_summary_empty_queue_all_zero_and_null(client, db_session):
    resp = client.get(SUMMARY_URL)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {
        "pending": 0,
        "running": 0,
        "succeeded": 0,
        "failed": 0,
        "oldest_pending_job_id": None,
        "oldest_pending_wait_seconds": None,
    }
    # The empty read wrote nothing at all.
    assert db_session.execute(select(ContentExportJob)).scalars().all() == []
    assert _audit_events(db_session) == []


def test_summary_body_is_compact_json_with_single_trailing_newline(client):
    resp = client.get(SUMMARY_URL)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    raw = resp.content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    # Compact separators: no insignificant whitespace inside the payload.
    payload = raw[:-1]
    assert b" " not in payload
    assert b"\n" not in payload
    parsed = json.loads(payload.decode("utf-8"))
    assert list(parsed) == [
        "pending",
        "running",
        "succeeded",
        "failed",
        "oldest_pending_job_id",
        "oldest_pending_wait_seconds",
    ]


# --- counts over mixed states -----------------------------------------------


def test_summary_counts_all_states_and_keeps_settled_jobs(client, db_session):
    create_actor(client)
    content = _create_content(client, "mixed")
    pending_a = _create_job(client, content["id"], "req-pending-a")
    _create_job(client, content["id"], "req-pending-b")
    running = _create_job(client, content["id"], "req-running")
    succeeded = _create_job(client, content["id"], "req-succeeded")
    failed = _create_job(client, content["id"], "req-failed")

    _set_status(db_session, running["id"], "running")
    _set_status(db_session, succeeded["id"], "succeeded")
    _set_status(db_session, failed["id"], "failed")

    resp = client.get(SUMMARY_URL)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["pending"] == 2
    assert body["running"] == 1
    assert body["succeeded"] == 1
    assert body["failed"] == 1
    # The oldest pending job is reported by its stable id.
    assert body["oldest_pending_job_id"] == pending_a["id"]
    assert isinstance(body["oldest_pending_wait_seconds"], int)

    # A repeated read still counts the settled jobs: they never disappear
    # from their terminal-state counts.
    again = client.get(SUMMARY_URL).json()
    assert again["succeeded"] == 1
    assert again["failed"] == 1


def test_summary_counts_jobs_settled_through_real_runs(client, db_session):
    create_actor(client)
    content = _create_content(client, "run")
    ok = _create_job(client, content["id"], "req-ok")
    other = _create_job(client, content["id"], "req-still-queued")

    run = client.post(f"/v1/content-export-jobs/{ok['id']}/run")
    assert run.status_code == 200
    assert run.json()["status"] == "succeeded"

    body = client.get(SUMMARY_URL).json()
    assert body["pending"] == 1
    assert body["succeeded"] == 1
    assert body["running"] == 0
    assert body["failed"] == 0
    assert body["oldest_pending_job_id"] == other["id"]


def test_summary_oldest_pending_follows_stable_creation_order(
    client, db_session
):
    create_actor(client)
    content = _create_content(client, "order")
    oldest = _create_job(client, content["id"], "req-o1")
    middle = _create_job(client, content["id"], "req-o2")
    youngest = _create_job(client, content["id"], "req-o3")

    # Once the oldest settles, the next oldest pending job is reported.
    _set_status(db_session, oldest["id"], "succeeded")
    body = client.get(SUMMARY_URL).json()
    assert body["oldest_pending_job_id"] == middle["id"]

    _set_status(db_session, middle["id"], "failed")
    body = client.get(SUMMARY_URL).json()
    assert body["oldest_pending_job_id"] == youngest["id"]

    # No pending jobs left: both oldest-pending members are null.
    _set_status(db_session, youngest["id"], "succeeded")
    body = client.get(SUMMARY_URL).json()
    assert body["oldest_pending_job_id"] is None
    assert body["oldest_pending_wait_seconds"] is None
    assert body["pending"] == 0


def test_summary_only_pending_jobs_have_no_oldest_when_none_pending(
    client, db_session
):
    create_actor(client)
    content = _create_content(client, "settled")
    job = _create_job(client, content["id"], "req-settled")
    _set_status(db_session, job["id"], "succeeded")

    body = client.get(SUMMARY_URL).json()
    assert body["succeeded"] == 1
    assert body["oldest_pending_job_id"] is None
    assert body["oldest_pending_wait_seconds"] is None


# --- wait time ---------------------------------------------------------------


def test_summary_wait_seconds_is_floored_whole_seconds(client, db_session):
    create_actor(client)
    content = _create_content(client, "wait")
    job = _create_job(client, content["id"], "req-wait")

    # Backdate creation by 90.7 seconds: the wait must floor to 90.
    created_at = utc_now() - timedelta(seconds=90, microseconds=700_000)
    _set_created_at(db_session, job["id"], created_at)

    before = utc_now()
    body = client.get(SUMMARY_URL).json()
    after = utc_now()

    assert body["oldest_pending_job_id"] == job["id"]
    wait = body["oldest_pending_wait_seconds"]
    assert isinstance(wait, int)
    lower = math.floor((before - created_at).total_seconds())
    upper = math.floor((after - created_at).total_seconds())
    assert lower <= wait <= upper
    assert wait >= 90


def test_summary_wait_is_recomputed_per_read(client, db_session):
    create_actor(client)
    content = _create_content(client, "rewait")
    job = _create_job(client, content["id"], "req-rewait")
    _set_created_at(db_session, job["id"], utc_now() - timedelta(seconds=5))

    first = client.get(SUMMARY_URL).json()
    # Move the creation further into the past; the next read reflects the
    # new elapsed time rather than a frozen snapshot.
    _set_created_at(db_session, job["id"], utc_now() - timedelta(seconds=30))
    second = client.get(SUMMARY_URL).json()

    assert first["oldest_pending_job_id"] == second["oldest_pending_job_id"]
    assert second["oldest_pending_wait_seconds"] >= 30
    assert (
        second["oldest_pending_wait_seconds"]
        > first["oldest_pending_wait_seconds"]
    )


# --- validation --------------------------------------------------------------


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
        # Any declared-value body is rejected too.
        lambda c: c.request(
            "GET", SUMMARY_URL, json={"status": "pending"}
        ),
        # No query parameters are accepted in any shape.
        lambda c: c.get(SUMMARY_URL + "?x=1"),
        lambda c: c.get(SUMMARY_URL + "?status=pending"),
        lambda c: c.get(SUMMARY_URL + "?x="),
        lambda c: c.get(SUMMARY_URL + "?x=1&x=2"),
        lambda c: c.get(SUMMARY_URL + "?limit=10"),
        # A body together with a query parameter is still a 422.
        lambda c: c.request("GET", SUMMARY_URL + "?x=1", content=b"{}"),
    ],
)
def test_summary_validation_errors_422_before_any_read(
    client, db_session, monkeypatch, send
):
    create_actor(client)
    content = _create_content(client, "guarded")
    _create_job(client, content["id"], "req-guarded")
    events_before = len(_audit_events(db_session))

    # Validation must run before the summary read: any attempt to read
    # jobs for a rejected request fails the test.
    def _forbidden_read(session):
        raise AssertionError("summary read must not run for a 422 request")

    monkeypatch.setattr(
        service, "summarize_content_export_jobs", _forbidden_read
    )

    resp = send(client)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"

    # A rejected request writes nothing.
    assert len(_audit_events(db_session)) == events_before
    jobs = db_session.execute(select(ContentExportJob)).scalars().all()
    assert [j.status for j in jobs] == ["pending"]

    # The queue is still served normally by a well-formed request.
    monkeypatch.undo()
    ok = client.get(SUMMARY_URL)
    assert ok.status_code == 200
    assert ok.json()["pending"] == 1


# --- read-only guarantees ----------------------------------------------------


def test_summary_successful_read_writes_nothing(client, db_session):
    create_actor(client)
    content = _create_content(client, "readonly")
    _create_job(client, content["id"], "req-readonly")
    events_before = len(_audit_events(db_session))
    jobs_before = db_session.execute(select(ContentExportJob)).scalars().all()

    resp = client.get(SUMMARY_URL)
    assert resp.status_code == 200

    assert len(_audit_events(db_session)) == events_before
    jobs_after = db_session.execute(select(ContentExportJob)).scalars().all()
    assert [j.id for j in jobs_after] == [j.id for j in jobs_before]
    assert [j.status for j in jobs_after] == [j.status for j in jobs_before]


# --- restart stability -------------------------------------------------------


def test_summary_counts_and_oldest_id_stable_across_restart(tmp_db_url):
    app1 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app1) as first:
        create_actor(first)
        content = _create_content(first, "restart")
        oldest = _create_job(first, content["id"], "req-r-oldest")
        _create_job(first, content["id"], "req-r-youngest")
        settled = _create_job(first, content["id"], "req-r-settled")
        run = first.post(f"/v1/content-export-jobs/{settled['id']}/run")
        assert run.status_code == 200
        before_restart = first.get(SUMMARY_URL).json()

    # A brand-new app over the same database file: counts and the stable
    # oldest-pending id are identical; the wait is recomputed (>= before).
    app2 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app2) as second:
        after_restart = second.get(SUMMARY_URL).json()

    assert after_restart["pending"] == before_restart["pending"] == 2
    assert after_restart["succeeded"] == before_restart["succeeded"] == 1
    assert after_restart["running"] == before_restart["running"] == 0
    assert after_restart["failed"] == before_restart["failed"] == 0
    assert (
        after_restart["oldest_pending_job_id"]
        == before_restart["oldest_pending_job_id"]
        == oldest["id"]
    )
    assert (
        after_restart["oldest_pending_wait_seconds"]
        >= before_restart["oldest_pending_wait_seconds"]
    )
