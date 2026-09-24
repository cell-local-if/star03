"""Tests for the content export job queue summary.

Covers ``GET /v1/content-export-jobs/summary``:

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
* success, empty results, and rejected requests are strictly read-only: no
  job, resource, or audit row is created or modified.

All fixtures are deterministic and offline (in-memory and temporary-file
SQLite, no network).
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update as sa_update

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
    assert db_session.execute(select(ContentExportJob)).scalars().all() == []
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


# --- counts over mixed states ------------------------------------------------


def test_summary_counts_cover_all_states_including_settled(
    client, db_session
):
    create_actor(client)
    content = _create_content(client, "mixed")
    pending_a = _create_job(client, content["id"], "req-pending-a")
    pending_b = _create_job(client, content["id"], "req-pending-b")
    running = _create_job(client, content["id"], "req-running")
    succeeded = _create_job(client, content["id"], "req-succeeded")
    failed = _create_job(client, content["id"], "req-failed")

    _set_status(db_session, running["id"], "running")
    run = client.post(f"/v1/content-export-jobs/{succeeded['id']}/run")
    assert run.status_code == 200
    assert run.json()["status"] == "succeeded"
    _set_status(db_session, failed["id"], "failed")

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
        client.get(f"/v1/content-export-jobs/{pending_a['id']}").json()[
            "status"
        ]
        == "pending"
    )
    assert (
        client.get(f"/v1/content-export-jobs/{pending_b['id']}").json()[
            "status"
        ]
        == "pending"
    )


def test_summary_oldest_pending_skips_non_pending_jobs(client, db_session):
    create_actor(client)
    content = _create_content(client, "order")
    oldest = _create_job(client, content["id"], "req-oldest-settled")
    middle = _create_job(client, content["id"], "req-middle-running")
    youngest_pending = _create_job(client, content["id"], "req-youngest")

    _set_status(db_session, oldest["id"], "succeeded")
    _set_status(db_session, middle["id"], "running")

    body = client.get(SUMMARY_URL).json()
    assert body["pending"] == 1
    # Older non-pending jobs are never reported as the queue head.
    assert body["oldest_pending_id"] == youngest_pending["id"]


def test_summary_no_pending_reports_null_oldest(client, db_session):
    create_actor(client)
    content = _create_content(client, "no-pending")
    job = _create_job(client, content["id"], "req-settled-only")
    run = client.post(f"/v1/content-export-jobs/{job['id']}/run")
    assert run.status_code == 200

    body = client.get(SUMMARY_URL).json()
    assert body["pending"] == 0
    assert body["succeeded"] == 1
    assert body["oldest_pending_id"] is None
    assert body["oldest_pending_wait_seconds"] is None


# --- wait time ---------------------------------------------------------------


def test_summary_wait_seconds_floor_to_whole_seconds(client, db_session):
    create_actor(client)
    content = _create_content(client, "wait")
    job = _create_job(client, content["id"], "req-wait")

    # Backdate the creation time by a non-integral number of seconds.
    created_at = utc_now() - timedelta(seconds=3723.6)
    db_session.execute(
        sa_update(ContentExportJob)
        .where(ContentExportJob.id == job["id"])
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
    import math

    assert wait >= math.floor((before - created_at).total_seconds())
    assert wait <= math.floor((after - created_at).total_seconds())
    assert 3723 <= wait <= 3725


def test_summary_wait_seconds_grows_between_reads(client, db_session):
    create_actor(client)
    content = _create_content(client, "wait-grows")
    job = _create_job(client, content["id"], "req-wait-grows")

    created_at = utc_now() - timedelta(seconds=10)
    db_session.execute(
        sa_update(ContentExportJob)
        .where(ContentExportJob.id == job["id"])
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
        create_actor(first)
        content = _create_content(first, "restart")
        oldest = _create_job(first, content["id"], "req-r-oldest")
        _create_job(first, content["id"], "req-r-youngest")
        settled = _create_job(first, content["id"], "req-r-settled")
        run = first.post(f"/v1/content-export-jobs/{settled['id']}/run")
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
    create_actor(client)
    content = _create_content(client, "wire")
    _create_job(client, content["id"], "req-wire")

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
    create_actor(client)
    content = _create_content(client, "guarded")
    job = _create_job(client, content["id"], "req-guarded")
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
    view = client.get(f"/v1/content-export-jobs/{job['id']}").json()
    assert view["status"] == "pending"

    # A well-formed read still succeeds afterwards.
    ok = client.get(SUMMARY_URL)
    assert ok.status_code == 200
    assert ok.json()["pending"] == 1
    assert ok.json()["oldest_pending_id"] == job["id"]
