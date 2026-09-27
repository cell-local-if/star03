"""Tests for server-side audit checkpoint job queue claiming.

Covers ``POST /v1/audit-checkpoint-jobs/run-next``:

* empty queue -> ``404 audit_checkpoint_job_not_found`` with zero writes;
* the single oldest ``pending`` job is claimed in stable creation order,
  non-pending jobs are skipped, younger pending jobs stay queued, and the
  queue drains oldest-first across calls (also across an app restart);
* the claimed run uses the atomic run semantics: succeeded with the
  checkpoint package for the job's filter and UTC timestamps, or failed with
  null result and ``audit_checkpoint_export_failed``, either way recording
  one ``audit_checkpoint_job.run`` audit event;
* any non-empty body or any query parameter is a 422 ``validation_error``
  that writes nothing and leaves the queue intact;
* a caller whose picked oldest job was claimed first gets ``409 conflict``
  without touching another job or writing an audit event;
* concurrent calls claim each job at most once.

All fixtures are deterministic and offline (in-memory and temporary-file
SQLite, no network).
"""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update as sa_update

from provenance import service
from provenance.api import _audit_checkpoint_package_payload
from provenance.app import create_app
from provenance.config import Settings
from provenance.errors import AuditCheckpointJobConflictError
from provenance.models import (
    EVENT_AUDIT_CHECKPOINT_JOB_RUN,
    AuditCheckpointJob,
    AuditEvent,
)
from tests.helpers import create_actor

RUN_NEXT_URL = "/v1/audit-checkpoint-jobs/run-next"
JOBS_URL = "/v1/audit-checkpoint-jobs"


def _create_job(client, request_id, **filters):
    resp = client.post(JOBS_URL, json={"request_id": request_id, **filters})
    assert resp.status_code == 201, resp.text
    return resp.json()


def _view(client, job_id):
    return client.get(f"{JOBS_URL}/{job_id}").json()


# --- empty queue ------------------------------------------------------------


def test_run_next_empty_queue_is_404_and_writes_nothing(client, db_session):
    assert db_session.execute(select(AuditCheckpointJob)).scalars().all() == []

    resp = client.post(RUN_NEXT_URL)
    assert resp.status_code == 404, resp.text
    err = resp.json()["error"]
    assert err["code"] == "audit_checkpoint_job_not_found"
    assert "job_id" not in err.get("details", {})

    assert db_session.execute(select(AuditCheckpointJob)).scalars().all() == []
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_run_next_with_only_settled_jobs_is_404_zero_writes(client, db_session):
    job = _create_job(client, "req-settled")
    run = client.post(f"{JOBS_URL}/{job['id']}/run")
    assert run.status_code == 200

    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())

    resp = client.post(RUN_NEXT_URL)
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "audit_checkpoint_job_not_found"

    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )
    assert _view(client, job["id"])["status"] == "succeeded"


# --- stable oldest-first claiming -------------------------------------------


def test_run_next_claims_oldest_pending_and_leaves_rest_queued(client):
    oldest = _create_job(client, "req-oldest")
    middle = _create_job(client, "req-middle")
    youngest = _create_job(client, "req-youngest")

    first = client.post(RUN_NEXT_URL)
    assert first.status_code == 200, first.text
    assert first.json()["id"] == oldest["id"]
    assert first.json()["status"] == "succeeded"

    for queued in (middle, youngest):
        body = _view(client, queued["id"])
        assert body["status"] == "pending"
        assert body["started_at"] is None
        assert body["finished_at"] is None
        assert body["result"] is None

    second = client.post(RUN_NEXT_URL)
    assert second.json()["id"] == middle["id"]
    third = client.post(RUN_NEXT_URL)
    assert third.json()["id"] == youngest["id"]
    assert client.post(RUN_NEXT_URL).status_code == 404


def test_run_next_skips_non_pending_jobs_to_oldest_pending(client, db_session):
    settled = _create_job(client, "req-settled")
    failed = _create_job(client, "req-failed-state")
    pending = _create_job(client, "req-pending")

    db_session.execute(
        sa_update(AuditCheckpointJob)
        .where(AuditCheckpointJob.id == settled["id"])
        .values(status="succeeded")
    )
    db_session.execute(
        sa_update(AuditCheckpointJob)
        .where(AuditCheckpointJob.id == failed["id"])
        .values(status="failed")
    )
    db_session.commit()

    resp = client.post(RUN_NEXT_URL)
    assert resp.status_code == 200, resp.text
    assert resp.json()["id"] == pending["id"]
    assert client.post(RUN_NEXT_URL).status_code == 404


def test_run_next_each_job_uses_its_own_filter(client):
    create_actor(client)
    typed = _create_job(client, "req-typed", event_type="actor.created")
    unfiltered = _create_job(client, "req-all")

    first = client.post(RUN_NEXT_URL).json()
    assert first["id"] == typed["id"]
    assert first["result"]["checkpoint"]["event_count"] == 1
    assert all(
        e["event_type"] == "actor.created" for e in first["result"]["events"]
    )

    second = client.post(RUN_NEXT_URL).json()
    assert second["id"] == unfiltered["id"]
    # The setup wrote one actor.created plus the two created-job events, so
    # the unfiltered package covers a different (larger) sequence.
    assert second["result"]["checkpoint"]["event_count"] > 1


def test_run_next_claims_in_stable_order_across_restart(tmp_db_url):
    # Stable creation order (created_at plus the monotonic seq) is persisted,
    # so a restarted server still claims the oldest pending job first; the
    # ``python -m provenance`` entry point itself is unchanged.
    app1 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app1) as first:
        oldest = _create_job(first, "req-r-oldest")
        youngest = _create_job(first, "req-r-youngest")

    app2 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app2) as second:
        run_one = second.post(RUN_NEXT_URL)
        assert run_one.status_code == 200, run_one.text
        assert run_one.json()["id"] == oldest["id"]

        run_two = second.post(RUN_NEXT_URL)
        assert run_two.status_code == 200, run_two.text
        assert run_two.json()["id"] == youngest["id"]

        assert second.post(RUN_NEXT_URL).status_code == 404


# --- run settlement ---------------------------------------------------------


def test_run_next_success_result_matches_package(client, db_session):
    create_actor(client)
    job = _create_job(client, "req-runnext-ok", event_type="actor.created")

    resp = client.post(RUN_NEXT_URL)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == job["id"]
    assert body["status"] == "succeeded"

    expected = client.get(
        "/v1/audit-events/checkpoint/package",
        params={"event_type": "actor.created"},
    ).json()
    assert body["result"] == expected

    events = db_session.execute(
        select(AuditEvent.event_type)
        .where(AuditEvent.resource_id == job["id"])
        .order_by(AuditEvent.seq)
    ).scalars().all()
    assert events == [
        "audit_checkpoint_job.created",
        "audit_checkpoint_job.run",
    ]


def test_run_next_failure_settles_failed_then_queue_continues(
    client, db_session, monkeypatch
):
    first_job = _create_job(client, "req-fail")
    second_job = _create_job(client, "req-next-after-fail")

    def _boom(session, job):
        raise RuntimeError("simulated checkpoint failure")

    monkeypatch.setattr(service, "list_audit_events", _boom)

    resp = client.post(RUN_NEXT_URL)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == first_job["id"]
    assert body["status"] == "failed"
    assert body["started_at"] is not None
    assert body["finished_at"] is not None
    assert body["result"] is None
    assert body["error"] == "audit_checkpoint_export_failed"

    monkeypatch.undo()
    follow_up = client.post(RUN_NEXT_URL)
    assert follow_up.status_code == 200, follow_up.text
    assert follow_up.json()["id"] == second_job["id"]
    assert follow_up.json()["status"] == "succeeded"

    first_events = db_session.execute(
        select(AuditEvent.event_type)
        .where(AuditEvent.resource_id == first_job["id"])
        .order_by(AuditEvent.seq)
    ).scalars().all()
    assert first_events == [
        "audit_checkpoint_job.created",
        "audit_checkpoint_job.run",
    ]


# --- request validation -----------------------------------------------------


@pytest.mark.parametrize(
    "send",
    [
        lambda c: c.post(RUN_NEXT_URL, content=b"   "),
        lambda c: c.post(
            RUN_NEXT_URL,
            content=b"{not json",
            headers={"content-type": "application/json"},
        ),
        lambda c: c.post(RUN_NEXT_URL, json={}),
        lambda c: c.post(RUN_NEXT_URL, json={"unexpected": True}),
        lambda c: c.post(RUN_NEXT_URL + "?x=1"),
        lambda c: c.post(RUN_NEXT_URL + "?status=pending"),
        lambda c: c.post(RUN_NEXT_URL + "?x="),
        lambda c: c.post(RUN_NEXT_URL + "?x=1&x=2"),
        lambda c: c.post(RUN_NEXT_URL + "?x=1", json={}),
    ],
)
def test_run_next_validation_errors_422_write_nothing(client, db_session, send):
    job = _create_job(client, "req-guarded")

    resp = send(client)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"

    assert _view(client, job["id"])["status"] == "pending"
    run_events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.resource_id == job["id"],
            AuditEvent.event_type == EVENT_AUDIT_CHECKPOINT_JOB_RUN,
        )
    ).scalars().all()
    assert run_events == []

    claimed = client.post(RUN_NEXT_URL)
    assert claimed.status_code == 200
    assert claimed.json()["id"] == job["id"]


# --- conflict losers --------------------------------------------------------


def test_run_next_loser_does_not_touch_other_jobs(tmp_db_url):
    """Deterministic interleaving over a file-backed database.

    Caller A selects the oldest pending job; before A's conditional UPDATE,
    caller B claims and settles that same job. A's compare-and-set then
    matches zero rows: A must surface a 409 conflict, must not modify any
    job, must not write an audit event, and must NOT fall through to the
    younger pending job.
    """
    application = create_app(Settings(database_url=tmp_db_url))
    with TestClient(application) as client:
        oldest = _create_job(client, "req-race-oldest")
        younger = _create_job(client, "req-race-younger")

    factory = application.state.session_factory
    session_a = factory()
    session_b = factory()
    try:
        picked = session_a.execute(
            select(AuditCheckpointJob)
            .where(AuditCheckpointJob.status == "pending")
            .order_by(
                AuditCheckpointJob.created_at.asc(),
                AuditCheckpointJob.seq.asc(),
            )
            .limit(1)
        ).scalar_one_or_none()
        assert picked.id == oldest["id"]
        session_a.commit()

        settled = service.run_audit_checkpoint_job(
            session_b, oldest["id"], _audit_checkpoint_package_payload
        )
        assert settled.status == "succeeded"

        with pytest.raises(AuditCheckpointJobConflictError) as exc_info:
            service._claim_and_run_audit_checkpoint_job(
                session_a, picked, _audit_checkpoint_package_payload
            )
        assert exc_info.value.code == "conflict"
        assert exc_info.value.details["job_id"] == oldest["id"]
        assert exc_info.value.details["status"] == "succeeded"
    finally:
        session_a.close()
        session_b.close()
        application.state.engine.dispose()

    verifier = create_app(Settings(database_url=tmp_db_url))
    with TestClient(verifier) as client:
        assert _view(client, oldest["id"])["status"] == "succeeded"
        young_view = _view(client, younger["id"])
        assert young_view["status"] == "pending"
        assert young_view["started_at"] is None

    session = verifier.state.session_factory()
    try:
        run_events = session.execute(
            select(AuditEvent).where(
                AuditEvent.event_type == EVENT_AUDIT_CHECKPOINT_JOB_RUN
            )
        ).scalars().all()
        assert [e.resource_id for e in run_events] == [oldest["id"]]
        younger_events = session.execute(
            select(AuditEvent.event_type)
            .where(AuditEvent.resource_id == younger["id"])
            .order_by(AuditEvent.seq)
        ).scalars().all()
        assert younger_events == ["audit_checkpoint_job.created"]
    finally:
        session.close()


def test_concurrent_run_next_calls_claim_exactly_once(file_app):
    """Six simultaneous run-next calls against one pending job: one winner."""
    setup = TestClient(file_app)
    job_id = _create_job(setup, "req-runnext-concurrent")["id"]

    outcomes: list[tuple[int, str]] = []
    lock = threading.Lock()

    def _fire() -> None:
        worker = TestClient(file_app)
        resp = worker.post(RUN_NEXT_URL)
        marker = resp.json().get("status") or resp.json()["error"]["code"]
        with lock:
            outcomes.append((resp.status_code, marker))

    threads = [threading.Thread(target=_fire) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    winners = [o for o in outcomes if o[0] == 200]
    conflicts = [o for o in outcomes if o[0] == 409]
    assert len(winners) == 1, outcomes
    assert winners[0][1] == "succeeded"
    assert len(conflicts) == 5, outcomes

    reader = TestClient(file_app)
    assert reader.get(f"{JOBS_URL}/{job_id}").json()["status"] == "succeeded"
    session = file_app.state.session_factory()
    try:
        run_events = session.execute(
            select(AuditEvent).where(
                AuditEvent.resource_id == job_id,
                AuditEvent.event_type == EVENT_AUDIT_CHECKPOINT_JOB_RUN,
            )
        ).scalars().all()
        assert len(run_events) == 1
    finally:
        session.close()
