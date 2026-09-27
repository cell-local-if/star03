"""Tests for server-side audit checkpoint job queue claiming.

Covers ``POST /v1/audit-checkpoint-jobs/run-next``:

* empty queue -> ``404 audit_checkpoint_job_not_found`` with zero writes (no
  job or audit row);
* the single oldest ``pending`` job is claimed in stable creation order,
  non-pending jobs are skipped, younger pending jobs stay queued, and the
  queue drains oldest-first across calls (also across an app restart);
* the claimed run uses the existing atomic run semantics: succeeded with the
  checkpoint package and UTC timestamps, or failed with null result and
  ``audit_checkpoint_export_failed``, either way recording one
  ``audit_checkpoint_job.run`` audit event;
* any non-empty body (whitespace, malformed JSON, an object, extra fields)
  or any query parameter (unknown, blank, repeated) is a 422
  ``validation_error`` that writes nothing and leaves the queue intact;
* a caller whose picked oldest job was claimed first gets ``409 conflict``
  without touching another job or writing an audit event, and never falls
  through to a younger queued job;
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
from provenance.models import AuditCheckpointJob, AuditEvent
from provenance.time_utils import parse_rfc3339_utc
from tests.helpers import create_actor

COLLECTION_URL = "/v1/audit-checkpoint-jobs"
RUN_NEXT_URL = "/v1/audit-checkpoint-jobs/run-next"


def _create_job(client, request_id, **filters):
    resp = client.post(
        COLLECTION_URL, json={"request_id": request_id, **filters}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# --- empty queue ------------------------------------------------------------


def test_run_next_empty_queue_is_404_and_writes_nothing(client, db_session):
    assert db_session.execute(select(AuditCheckpointJob)).scalars().all() == []

    resp = client.post(RUN_NEXT_URL)
    assert resp.status_code == 404, resp.text
    err = resp.json()["error"]
    assert err["code"] == "audit_checkpoint_job_not_found"
    # No job id exists to report for a queue-wide miss.
    assert "job_id" not in err.get("details", {})

    assert db_session.execute(select(AuditCheckpointJob)).scalars().all() == []
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_run_next_with_only_settled_jobs_is_404_zero_writes(
    client, db_session
):
    job = _create_job(client, "req-settled")
    run = client.post(f"{COLLECTION_URL}/{job['id']}/run")
    assert run.status_code == 200

    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())

    resp = client.post(RUN_NEXT_URL)
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "audit_checkpoint_job_not_found"

    # The miss adds no job and no audit event.
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )
    assert (
        client.get(f"{COLLECTION_URL}/{job['id']}").json()["status"]
        == "succeeded"
    )


# --- stable oldest-first claiming -------------------------------------------


def test_run_next_claims_oldest_pending_and_leaves_rest_queued(client):
    oldest = _create_job(client, "req-oldest")
    middle = _create_job(client, "req-middle")
    youngest = _create_job(client, "req-youngest")

    first = client.post(RUN_NEXT_URL)
    assert first.status_code == 200, first.text
    assert first.json()["id"] == oldest["id"]
    assert first.json()["status"] == "succeeded"

    # The other jobs are untouched: still pending with null run fields.
    for queued in (middle, youngest):
        view = client.get(f"{COLLECTION_URL}/{queued['id']}").json()
        assert view["status"] == "pending"
        assert view["started_at"] is None
        assert view["finished_at"] is None
        assert view["result"] is None

    second = client.post(RUN_NEXT_URL)
    assert second.status_code == 200
    assert second.json()["id"] == middle["id"]

    third = client.post(RUN_NEXT_URL)
    assert third.status_code == 200
    assert third.json()["id"] == youngest["id"]

    # The queue is now drained.
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

    # Nothing left to claim.
    assert client.post(RUN_NEXT_URL).status_code == 404


def test_run_next_claims_in_stable_order_across_restart(tmp_db_url):
    # Stable creation order (created_at plus the monotonic seq) is persisted,
    # so a restarted server still claims the oldest pending job first; the
    # ``python -m provenance`` entry point itself is unchanged.
    app1 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app1) as first:
        oldest = _create_job(first, "req-r-oldest")
        youngest = _create_job(first, "req-r-youngest")

    # A brand-new process/app over the same database file.
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


def test_run_next_success_result_matches_checkpoint_package(
    client, db_session
):
    create_actor(client)
    job = _create_job(client, "req-runnext-ok")
    # Snapshot from the read state immediately before the claim: the run
    # records its own audit event only after the result is built.
    expected = client.get(
        "/v1/audit-events/checkpoint/package"
    ).json()

    resp = client.post(RUN_NEXT_URL)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == job["id"]
    assert body["status"] == "succeeded"

    started = parse_rfc3339_utc(body["started_at"])
    finished = parse_rfc3339_utc(body["finished_at"])
    assert started.tzinfo is not None
    assert finished.tzinfo is not None
    assert started.utcoffset().total_seconds() == 0
    assert started <= finished
    assert body["error"] is None

    assert body["result"] == expected

    events = db_session.execute(
        select(AuditEvent.event_type)
        .where(AuditEvent.resource_id == job["id"])
        .order_by(AuditEvent.seq)
    ).scalars().all()
    assert events == ["audit_checkpoint_job.created", "audit_checkpoint_job.run"]


def test_run_next_failure_settles_failed_then_queue_continues(
    client, db_session, monkeypatch
):
    first_job = _create_job(client, "req-fail")
    second_job = _create_job(client, "req-next-after-fail")

    def _boom(session, event_type, resource_id, from_dt, to_dt):
        raise RuntimeError("simulated checkpoint export failure")

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

    # Once the fault clears, the next claim takes the following queued job;
    # the failed job is settled and never retried.
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
        # Any non-empty body is rejected, even whitespace.
        lambda c: c.post(RUN_NEXT_URL, content=b"   "),
        # Malformed JSON is a validation error, not a parser crash.
        lambda c: c.post(
            RUN_NEXT_URL,
            content=b"{not json",
            headers={"content-type": "application/json"},
        ),
        # An empty JSON object is still a non-empty body.
        lambda c: c.post(RUN_NEXT_URL, json={}),
        # Any extra field rides a non-empty body.
        lambda c: c.post(RUN_NEXT_URL, json={"unexpected": True}),
        # No query parameters are accepted in any shape.
        lambda c: c.post(RUN_NEXT_URL + "?x=1"),
        lambda c: c.post(RUN_NEXT_URL + "?status=pending"),
        lambda c: c.post(RUN_NEXT_URL + "?x="),
        lambda c: c.post(RUN_NEXT_URL + "?x=1&x=2"),
        # A body together with a query parameter is still a 422.
        lambda c: c.post(RUN_NEXT_URL + "?x=1", json={}),
    ],
)
def test_run_next_validation_errors_422_write_nothing(
    client, db_session, send
):
    job = _create_job(client, "req-guarded")

    resp = send(client)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"

    # A rejected request leaves the queue intact: the pending job is still
    # claimable by a well-formed call, and no run audit event exists.
    view = client.get(f"{COLLECTION_URL}/{job['id']}").json()
    assert view["status"] == "pending"
    run_events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.resource_id == job["id"],
            AuditEvent.event_type == "audit_checkpoint_job.run",
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
        # A reads the queue candidate first.
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
        # End A's read transaction so B can commit on this file-backed
        # SQLite database; the picked candidate and its (stale) pending
        # status are retained on the object.
        session_a.commit()

        # B wins the same job outright through the real run path.
        settled = service.run_audit_checkpoint_job(
            session_b, oldest["id"], _audit_checkpoint_package_payload
        )
        assert settled.status == "succeeded"

        # A now runs its claim-and-settle against the stale pick. The CAS
        # finds the row no longer pending and must refuse.
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

    # Reopen to assert final state.
    verifier = create_app(Settings(database_url=tmp_db_url))
    with TestClient(verifier) as client:
        assert (
            client.get(f"{COLLECTION_URL}/{oldest['id']}").json()["status"]
            == "succeeded"
        )
        # The loser never fell through: the younger job is still queued.
        young_view = client.get(
            f"{COLLECTION_URL}/{younger['id']}"
        ).json()
        assert young_view["status"] == "pending"
        assert young_view["started_at"] is None

    session = verifier.state.session_factory()
    try:
        # Exactly one run audit event (the winner's), and none for the
        # younger job beyond its creation event.
        run_events = session.execute(
            select(AuditEvent).where(
                AuditEvent.event_type == "audit_checkpoint_job.run"
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
    """Real concurrency over a file-backed SQLite database.

    Six simultaneous run-next calls against one pending job: exactly one
    returns 200 and settles succeeded; every other is a 409 conflict. The
    job ends succeeded with a single run audit event.
    """
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
    final = reader.get(f"{COLLECTION_URL}/{job_id}").json()
    assert final["status"] == "succeeded"
    session = file_app.state.session_factory()
    try:
        run_events = session.execute(
            select(AuditEvent).where(
                AuditEvent.resource_id == job_id,
                AuditEvent.event_type == "audit_checkpoint_job.run",
            )
        ).scalars().all()
        assert len(run_events) == 1
    finally:
        session.close()


@pytest.mark.parametrize("method", ("put", "patch", "delete"))
def test_other_methods_on_run_next_are_405(client, method):
    resp = getattr(client, method)(RUN_NEXT_URL)
    assert resp.status_code == 405
    assert resp.json()["error"]["code"] == "method_not_allowed"


def test_get_on_run_next_is_treated_as_job_detail_404(client):
    # GET is registered on the detail path ("/.../{job_id}"), so the literal
    # "run-next" segment is read as a job id rather than matching the queue
    # route: no such job exists, so it is the standard 404 -- exactly as for
    # the content export job family.
    resp = client.get(RUN_NEXT_URL)
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "audit_checkpoint_job_not_found"
