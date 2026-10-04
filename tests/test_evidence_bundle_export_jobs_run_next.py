"""Tests for server-side evidence bundle export job queue claiming.

Covers ``POST /v1/evidence-bundle-export-jobs/run-next``:

* empty queue -> ``404 evidence_bundle_export_job_not_found`` with zero
  writes;
* the single oldest ``pending`` job is claimed in stable creation order,
  non-pending jobs are skipped, younger pending jobs stay queued, and the
  queue drains oldest-first across calls (also across an app restart);
* the claimed run uses the atomic single-job run semantics: succeeded with
  the exchange package (``{"snapshot", "manifest"}``) for the job's bundle
  and UTC timestamps, or failed with null result and
  ``evidence_bundle_export_failed``;
* any non-empty body or any query parameter is a 422 ``validation_error``
  that writes nothing and leaves the queue intact;
* a caller whose picked oldest job was claimed first gets ``409
  evidence_bundle_export_job_state_conflict`` without touching another job
  or writing an audit event;
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
from provenance.api import _evidence_bundle_export_result_payload
from provenance.app import create_app
from provenance.config import Settings
from provenance.errors import EvidenceBundleExportJobConflictError
from provenance.models import AuditEvent, EvidenceBundleExportJob
from tests.test_evidence_bundle_exchange import (
    _create_bundle,
    _setup_bundle,
)

RUN_NEXT_URL = "/v1/evidence-bundle-export-jobs/run-next"
JOBS_URL = "/v1/evidence-bundle-export-jobs"


def _create_job(client, evidence_bundle_id, request_id):
    resp = client.post(
        JOBS_URL,
        json={
            "evidence_bundle_id": evidence_bundle_id,
            "request_id": request_id,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _view(client, job_id):
    return client.get(f"{JOBS_URL}/{job_id}").json()


# --- empty queue ------------------------------------------------------------


def test_run_next_empty_queue_is_404_and_writes_nothing(client, db_session):
    assert (
        db_session.execute(select(EvidenceBundleExportJob)).scalars().all()
        == []
    )

    resp = client.post(RUN_NEXT_URL)
    assert resp.status_code == 404, resp.text
    err = resp.json()["error"]
    assert err["code"] == "evidence_bundle_export_job_not_found"
    assert "job_id" not in err.get("details", {})

    assert (
        db_session.execute(select(EvidenceBundleExportJob)).scalars().all()
        == []
    )
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_run_next_with_only_settled_jobs_is_404_zero_writes(client, db_session):
    _, _, bundle = _setup_bundle(client)
    job = _create_job(client, bundle["id"], "req-settled")
    run = client.post(f"{JOBS_URL}/{job['id']}/run")
    assert run.status_code == 200

    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())

    resp = client.post(RUN_NEXT_URL)
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "evidence_bundle_export_job_not_found"

    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )
    assert _view(client, job["id"])["status"] == "succeeded"


# --- stable oldest-first claiming -------------------------------------------


def test_run_next_claims_oldest_pending_and_leaves_rest_queued(client):
    _, _, bundle = _setup_bundle(client)
    oldest = _create_job(client, bundle["id"], "req-oldest")
    middle = _create_job(client, bundle["id"], "req-middle")
    youngest = _create_job(client, bundle["id"], "req-youngest")

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
    _, _, bundle = _setup_bundle(client)
    settled = _create_job(client, bundle["id"], "req-settled")
    failed = _create_job(client, bundle["id"], "req-failed-state")
    pending = _create_job(client, bundle["id"], "req-pending")

    db_session.execute(
        sa_update(EvidenceBundleExportJob)
        .where(EvidenceBundleExportJob.id == settled["id"])
        .values(status="succeeded")
    )
    db_session.execute(
        sa_update(EvidenceBundleExportJob)
        .where(EvidenceBundleExportJob.id == failed["id"])
        .values(status="failed")
    )
    db_session.commit()

    resp = client.post(RUN_NEXT_URL)
    assert resp.status_code == 200, resp.text
    assert resp.json()["id"] == pending["id"]
    assert client.post(RUN_NEXT_URL).status_code == 404


def test_run_next_each_job_exports_its_own_bundle(client):
    _, claim, bundle = _setup_bundle(client)
    other = _create_bundle(client, claim["id"], "other-evidence")
    first_job = _create_job(client, bundle["id"], "req-first")
    second_job = _create_job(client, other["id"], "req-second")

    first = client.post(RUN_NEXT_URL).json()
    assert first["id"] == first_job["id"]
    assert first["result"]["snapshot"]["evidence_bundle"]["id"] == bundle["id"]
    assert first["result"]["manifest"]["evidence_bundle_id"] == bundle["id"]

    second = client.post(RUN_NEXT_URL).json()
    assert second["id"] == second_job["id"]
    assert second["result"]["snapshot"]["evidence_bundle"]["id"] == other["id"]
    assert second["result"]["manifest"]["evidence_bundle_id"] == other["id"]


def test_run_next_claims_in_stable_order_across_restart(tmp_db_url):
    # Stable creation order (created_at plus the monotonic seq) is persisted,
    # so a restarted server still claims the oldest pending job first; the
    # ``python -m provenance`` entry point itself is unchanged.
    app1 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app1) as first:
        _, _, bundle = _setup_bundle(first)
        oldest = _create_job(first, bundle["id"], "req-r-oldest")
        youngest = _create_job(first, bundle["id"], "req-r-youngest")

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


def test_run_next_success_result_matches_exchange_package(client, db_session):
    _, _, bundle = _setup_bundle(client)
    job = _create_job(client, bundle["id"], "req-runnext-ok")

    resp = client.post(RUN_NEXT_URL)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == job["id"]
    assert body["status"] == "succeeded"
    assert body["started_at"] is not None
    assert body["finished_at"] is not None
    assert body["error"] is None

    # The result is exactly the existing read-only exchange package.
    expected = client.get(
        f"/v1/evidence-bundles/{bundle['id']}/exchange/package"
    ).json()
    assert body["result"] == expected
    assert set(body["result"]) == {"snapshot", "manifest"}

    # Like the single-job run, the queue claim writes no run audit event.
    events = db_session.execute(
        select(AuditEvent.event_type)
        .where(AuditEvent.resource_id == job["id"])
        .order_by(AuditEvent.seq)
    ).scalars().all()
    assert events == ["evidence_bundle_export_job.created"]


def test_run_next_failure_settles_failed_then_queue_continues(
    client, db_session, monkeypatch
):
    _, _, bundle = _setup_bundle(client)
    first_job = _create_job(client, bundle["id"], "req-fail")
    second_job = _create_job(client, bundle["id"], "req-next-after-fail")

    def _boom(session, evidence_bundle_id):
        raise RuntimeError("simulated export failure")

    monkeypatch.setattr(service, "get_evidence_bundle_exchange", _boom)

    resp = client.post(RUN_NEXT_URL)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == first_job["id"]
    assert body["status"] == "failed"
    assert body["started_at"] is not None
    assert body["finished_at"] is not None
    assert body["result"] is None
    assert body["error"] == "evidence_bundle_export_failed"

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
    assert first_events == ["evidence_bundle_export_job.created"]


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
    _, _, bundle = _setup_bundle(client)
    job = _create_job(client, bundle["id"], "req-guarded")

    resp = send(client)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"

    assert _view(client, job["id"])["status"] == "pending"
    events = db_session.execute(
        select(AuditEvent.event_type)
        .where(AuditEvent.resource_id == job["id"])
        .order_by(AuditEvent.seq)
    ).scalars().all()
    assert events == ["evidence_bundle_export_job.created"]

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
        _, _, bundle = _setup_bundle(client)
        oldest = _create_job(client, bundle["id"], "req-race-oldest")
        younger = _create_job(client, bundle["id"], "req-race-younger")

    factory = application.state.session_factory
    session_a = factory()
    session_b = factory()
    try:
        picked = session_a.execute(
            select(EvidenceBundleExportJob)
            .where(EvidenceBundleExportJob.status == "pending")
            .order_by(
                EvidenceBundleExportJob.created_at.asc(),
                EvidenceBundleExportJob.seq.asc(),
            )
            .limit(1)
        ).scalar_one_or_none()
        assert picked.id == oldest["id"]
        session_a.commit()

        settled = service.run_evidence_bundle_export_job(
            session_b, oldest["id"], _evidence_bundle_export_result_payload
        )
        assert settled.status == "succeeded"

        with pytest.raises(EvidenceBundleExportJobConflictError) as exc_info:
            service._claim_and_run_evidence_bundle_export_job(
                session_a, picked, _evidence_bundle_export_result_payload
            )
        assert exc_info.value.code == "evidence_bundle_export_job_state_conflict"
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
        younger_events = session.execute(
            select(AuditEvent.event_type)
            .where(AuditEvent.resource_id == younger["id"])
            .order_by(AuditEvent.seq)
        ).scalars().all()
        assert younger_events == ["evidence_bundle_export_job.created"]
        oldest_events = session.execute(
            select(AuditEvent.event_type)
            .where(AuditEvent.resource_id == oldest["id"])
            .order_by(AuditEvent.seq)
        ).scalars().all()
        assert oldest_events == ["evidence_bundle_export_job.created"]
    finally:
        session.close()


def test_concurrent_run_next_calls_claim_exactly_once(file_app):
    """Six simultaneous run-next calls against one pending job: one winner."""
    setup = TestClient(file_app)
    _, _, bundle = _setup_bundle(setup)
    job_id = _create_job(setup, bundle["id"], "req-runnext-concurrent")["id"]

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
        events = session.execute(
            select(AuditEvent.event_type)
            .where(AuditEvent.resource_id == job_id)
            .order_by(AuditEvent.seq)
        ).scalars().all()
        # Exactly one claim won; the export job family records no run event.
        assert events == ["evidence_bundle_export_job.created"]
    finally:
        session.close()
