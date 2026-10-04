"""Tests for asynchronous evidence bundle exchange package export jobs.

Covers:

* ``POST /v1/evidence-bundle-export-jobs`` — required non-empty
  ``evidence_bundle_id`` and ``request_id`` (422 ``validation_error``
  otherwise), unknown bundle ``404 evidence_bundle_not_found``, first
  creation ``201`` as ``pending`` with null
  ``started_at``/``finished_at``/``result``/``error``, idempotent retry of
  the same key ``200`` (and no second audit row), and the same
  ``request_id`` reused for a different bundle ``409
  evidence_bundle_export_request_conflict``.
* ``GET /v1/evidence-bundle-export-jobs/{job_id}`` — the job view, or
  ``404 evidence_bundle_export_job_not_found``.
* ``POST /v1/evidence-bundle-export-jobs/{job_id}/run`` — atomic claim of a
  pending job (``running`` + UTC ``started_at``) settling to ``succeeded``
  with UTC ``finished_at`` and a result equal to the existing read-only
  exchange package (``{"snapshot", "manifest"}``); a repeated, concurrent,
  or otherwise non-pending run is ``409
  evidence_bundle_export_job_state_conflict`` and overwrites nothing; a
  failed run settles ``failed`` with UTC ``finished_at``, null ``result``
  and ``evidence_bundle_export_failed``; an unknown id is ``404
  evidence_bundle_export_job_not_found``. The job row and its
  ``evidence_bundle_export_job.created`` audit event commit in one
  transaction.

All fixtures are deterministic and offline (in-memory SQLite, plus a
temporary-file SQLite database for the real concurrent-claim test).
"""

from __future__ import annotations

import threading
from datetime import timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update as sa_update

from provenance import service
from provenance.models import (
    AuditEvent,
    EvidenceBundleExportJob,
)
from provenance.time_utils import parse_rfc3339_utc
from tests.test_evidence_bundle_exchange import (
    _create_bundle,
    _create_claim,
    _create_content,
    _setup_bundle,
)


def _create_job(client, evidence_bundle_id, request_id):
    return client.post(
        "/v1/evidence-bundle-export-jobs",
        json={
            "evidence_bundle_id": evidence_bundle_id,
            "request_id": request_id,
        },
    )


_JOB_KEYS = {
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


# --- creation ---------------------------------------------------------------


def test_create_job_201_pending_with_null_fields(client):
    _, _, bundle = _setup_bundle(client)

    resp = _create_job(client, bundle["id"], "req-1")
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == _JOB_KEYS
    assert body["id"].startswith("exj_")
    assert body["evidence_bundle_id"] == bundle["id"]
    assert body["request_id"] == "req-1"
    assert body["status"] == "pending"
    assert body["started_at"] is None
    assert body["finished_at"] is None
    assert body["result"] is None
    assert body["error"] is None
    assert body["created_at"].endswith("Z")


def test_create_job_writes_created_audit_event(client, db_session):
    _, _, bundle = _setup_bundle(client)

    body = _create_job(client, bundle["id"], "req-audit").json()
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.resource_id == body["id"],
            AuditEvent.event_type == "evidence_bundle_export_job.created",
        )
    ).scalars().all()
    assert len(events) == 1
    assert events[0].created_at.tzinfo is not None


def test_create_job_idempotent_same_key_returns_200(client, db_session):
    _, _, bundle = _setup_bundle(client)

    first = _create_job(client, bundle["id"], "retry-key")
    assert first.status_code == 201, first.text
    original = first.json()

    second = _create_job(client, bundle["id"], "retry-key")
    assert second.status_code == 200, second.text
    assert second.json() == original

    # The retry writes neither a second job row nor a second audit event, in
    # any lifecycle state.
    jobs = db_session.execute(
        select(EvidenceBundleExportJob).where(
            EvidenceBundleExportJob.request_id == "retry-key"
        )
    ).scalars().all()
    assert len(jobs) == 1
    events = db_session.execute(
        select(AuditEvent).where(AuditEvent.resource_id == original["id"])
    ).scalars().all()
    assert [e.event_type for e in events] == [
        "evidence_bundle_export_job.created"
    ]


def test_create_job_same_request_id_different_bundle_is_409(client, db_session):
    _, claim, bundle = _setup_bundle(client)
    other = _create_bundle(client, claim["id"], "other-evidence")

    ok = _create_job(client, bundle["id"], "shared-key")
    assert ok.status_code == 201, ok.text

    conflict = _create_job(client, other["id"], "shared-key")
    assert conflict.status_code == 409, conflict.text
    err = conflict.json()["error"]
    assert err["code"] == "evidence_bundle_export_request_conflict"
    assert err["details"]["request_id"] == "shared-key"

    # The conflict writes no job and no audit event; the original job alone
    # remains bound to that request_id.
    jobs = db_session.execute(
        select(EvidenceBundleExportJob).where(
            EvidenceBundleExportJob.request_id == "shared-key"
        )
    ).scalars().all()
    assert len(jobs) == 1
    assert jobs[0].evidence_bundle_id == bundle["id"]


def test_distinct_request_ids_for_same_bundle_are_independent_jobs(client):
    _, _, bundle = _setup_bundle(client)

    one = _create_job(client, bundle["id"], "req-one")
    two = _create_job(client, bundle["id"], "req-two")
    assert one.status_code == 201
    assert two.status_code == 201
    assert one.json()["id"] != two.json()["id"]


def test_re_register_same_key_after_settlement_is_200_unchanged(client):
    _, _, bundle = _setup_bundle(client)
    created = _create_job(client, bundle["id"], "req-settled").json()
    settled = client.post(
        f"/v1/evidence-bundle-export-jobs/{created['id']}/run"
    ).json()
    assert settled["status"] == "succeeded"

    # Re-registering the same (request_id, evidence_bundle_id) after
    # settlement is still idempotent: 200 with the original settled job, no
    # new row/event.
    retry = _create_job(client, bundle["id"], "req-settled")
    assert retry.status_code == 200, retry.text
    assert retry.json() == settled


@pytest.mark.parametrize(
    "payload",
    [
        {"request_id": "r"},  # missing evidence_bundle_id
        {"evidence_bundle_id": "evb_x"},  # missing request_id
        {"evidence_bundle_id": "", "request_id": "r"},  # empty bundle id
        {"evidence_bundle_id": "evb_x", "request_id": ""},  # empty request_id
        {"evidence_bundle_id": "   ", "request_id": "r"},  # whitespace id
        {"evidence_bundle_id": "evb_x", "request_id": "   "},  # whitespace key
        {"evidence_bundle_id": None, "request_id": "r"},  # null bundle id
        {"evidence_bundle_id": "evb_x", "request_id": None},  # null request_id
        {  # undeclared field
            "evidence_bundle_id": "evb_x",
            "request_id": "r",
            "unexpected": True,
        },
    ],
)
def test_create_job_validation_errors_422(client, payload):
    resp = client.post("/v1/evidence-bundle-export-jobs", json=payload)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def test_create_job_malformed_json_is_422(client):
    resp = client.post(
        "/v1/evidence-bundle-export-jobs",
        content="{not json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def test_create_job_unknown_bundle_is_404(client):
    _setup_bundle(client)
    resp = _create_job(client, "evb_does_not_exist", "req-x")
    assert resp.status_code == 404, resp.text
    err = resp.json()["error"]
    assert err["code"] == "evidence_bundle_not_found"
    assert err["details"]["evidence_bundle_id"] == "evb_does_not_exist"


# --- read -------------------------------------------------------------------


def test_get_job_returns_job(client):
    _, _, bundle = _setup_bundle(client)
    created = _create_job(client, bundle["id"], "req-get").json()

    resp = client.get(f"/v1/evidence-bundle-export-jobs/{created['id']}")
    assert resp.status_code == 200, resp.text
    assert resp.json() == created


def test_get_unknown_job_is_404(client):
    resp = client.get("/v1/evidence-bundle-export-jobs/exj_does_not_exist")
    assert resp.status_code == 404, resp.text
    err = resp.json()["error"]
    assert err["code"] == "evidence_bundle_export_job_not_found"
    assert err["details"]["job_id"] == "exj_does_not_exist"


# --- run: success -----------------------------------------------------------


def test_run_pending_job_succeeds_with_package_result(client):
    _, _, bundle = _setup_bundle(client)
    # A second bundle on another claim must never appear in this job's
    # result.
    other_content = _create_content(client, "other")
    other_claim = _create_claim(client, other_content["id"], "other-claim")
    _create_bundle(client, other_claim["id"], "other-evidence")

    job = _create_job(client, bundle["id"], "req-run").json()
    resp = client.post(f"/v1/evidence-bundle-export-jobs/{job['id']}/run")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert set(body) == _JOB_KEYS
    assert body["status"] == "succeeded"
    started = parse_rfc3339_utc(body["started_at"])
    finished = parse_rfc3339_utc(body["finished_at"])
    assert started is not None and finished is not None
    assert started.tzinfo == timezone.utc
    assert finished.tzinfo == timezone.utc
    assert started <= finished
    assert body["error"] is None

    # The result is exactly the existing read-only exchange package.
    expected = client.get(
        f"/v1/evidence-bundles/{bundle['id']}/exchange/package"
    ).json()
    assert body["result"] == expected
    assert set(body["result"]) == {"snapshot", "manifest"}
    assert body["result"]["snapshot"]["evidence_bundle"]["id"] == bundle["id"]
    assert (
        body["result"]["manifest"]["evidence_bundle_id"] == bundle["id"]
    )

    # The settled state is what a subsequent read observes.
    assert (
        client.get(f"/v1/evidence-bundle-export-jobs/{job['id']}").json()
        == body
    )


def test_run_result_manifest_digests_the_same_snapshot(client):
    _, _, bundle = _setup_bundle(client)
    job = _create_job(client, bundle["id"], "req-digest").json()

    body = client.post(
        f"/v1/evidence-bundle-export-jobs/{job['id']}/run"
    ).json()
    assert body["status"] == "succeeded"
    result = body["result"]
    # The stored manifest matches the manifest route computed over the same
    # snapshot, and the snapshot matches the exchange route.
    assert result["manifest"] == client.get(
        f"/v1/evidence-bundles/{bundle['id']}/exchange/manifest"
    ).json()
    assert result["snapshot"] == client.get(
        f"/v1/evidence-bundles/{bundle['id']}/exchange"
    ).json()


def test_run_writes_no_run_audit_event(client, db_session):
    _, _, bundle = _setup_bundle(client)
    job = _create_job(client, bundle["id"], "req-audit-run").json()

    run = client.post(f"/v1/evidence-bundle-export-jobs/{job['id']}/run")
    assert run.status_code == 200, run.text

    events = db_session.execute(
        select(AuditEvent.event_type)
        .where(AuditEvent.resource_id == job["id"])
        .order_by(AuditEvent.seq)
    ).scalars().all()
    assert events == ["evidence_bundle_export_job.created"]


# --- run: conflicts and missing ---------------------------------------------


def test_run_unknown_job_is_404(client):
    resp = client.post("/v1/evidence-bundle-export-jobs/exj_missing/run")
    assert resp.status_code == 404, resp.text
    err = resp.json()["error"]
    assert err["code"] == "evidence_bundle_export_job_not_found"
    assert err["details"]["job_id"] == "exj_missing"


def test_run_twice_only_first_runs(client, db_session):
    _, _, bundle = _setup_bundle(client)
    job = _create_job(client, bundle["id"], "req-rerun").json()

    first = client.post(f"/v1/evidence-bundle-export-jobs/{job['id']}/run")
    assert first.status_code == 200, first.text
    assert first.json()["status"] == "succeeded"

    second = client.post(f"/v1/evidence-bundle-export-jobs/{job['id']}/run")
    assert second.status_code == 409, second.text
    assert (
        second.json()["error"]["code"]
        == "evidence_bundle_export_job_state_conflict"
    )

    # The rejected repeat changes no state and writes no second run event.
    final = client.get(f"/v1/evidence-bundle-export-jobs/{job['id']}").json()
    assert final == first.json()


@pytest.mark.parametrize("state", ["running", "succeeded", "failed"])
def test_run_non_pending_job_is_409(client, db_session, state):
    _, _, bundle = _setup_bundle(client)
    job = _create_job(client, bundle["id"], f"req-{state}").json()

    # Move the job directly into a non-pending lifecycle state.
    db_session.execute(
        sa_update(EvidenceBundleExportJob)
        .where(EvidenceBundleExportJob.id == job["id"])
        .values(status=state)
    )
    db_session.commit()

    resp = client.post(f"/v1/evidence-bundle-export-jobs/{job['id']}/run")
    assert resp.status_code == 409, resp.text
    assert (
        resp.json()["error"]["code"]
        == "evidence_bundle_export_job_state_conflict"
    )

    # The rejected run leaves the state untouched.
    assert (
        client.get(f"/v1/evidence-bundle-export-jobs/{job['id']}").json()[
            "status"
        ]
        == state
    )


def test_run_settles_failed_when_export_raises(client, db_session, monkeypatch):
    _, _, bundle = _setup_bundle(client)
    job = _create_job(client, bundle["id"], "req-fail").json()

    def _boom(session, evidence_bundle_id):
        raise RuntimeError("simulated export failure")

    monkeypatch.setattr(service, "get_evidence_bundle_exchange", _boom)

    resp = client.post(f"/v1/evidence-bundle-export-jobs/{job['id']}/run")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "failed"
    assert body["started_at"] is not None
    assert body["finished_at"] is not None
    assert body["result"] is None
    assert body["error"] == "evidence_bundle_export_failed"

    # The failure is durable; a failed job can never be (re)claimed, even
    # after the fault clears.
    monkeypatch.undo()
    assert (
        client.post(
            f"/v1/evidence-bundle-export-jobs/{job['id']}/run"
        ).status_code
        == 409
    )
    final = client.get(f"/v1/evidence-bundle-export-jobs/{job['id']}").json()
    assert final["status"] == "failed"
    assert final["error"] == "evidence_bundle_export_failed"


def test_concurrent_runs_claim_exactly_once(file_app):
    """Real concurrency over a file-backed SQLite database.

    The pending->running claim is a single conditional UPDATE, so of many
    simultaneous run calls exactly one must return 200 (settling succeeded)
    and every other must be a 409; the final state is succeeded, regardless
    of thread interleaving.
    """
    # Tables and app.state are created eagerly by create_app, so plain
    # TestClients (without the lifespan context) serve requests; they are
    # deliberately not closed inside workers so the shared engine is not
    # disposed while sibling threads still hold connections.
    setup = TestClient(file_app)
    _, _, bundle = _setup_bundle(setup)
    job_id = _create_job(setup, bundle["id"], "req-concurrent").json()["id"]

    outcomes: list[tuple[int, str]] = []
    lock = threading.Lock()

    def _fire() -> None:
        worker = TestClient(file_app)
        resp = worker.post(f"/v1/evidence-bundle-export-jobs/{job_id}/run")
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
    final = reader.get(f"/v1/evidence-bundle-export-jobs/{job_id}").json()
    assert final["status"] == "succeeded"
