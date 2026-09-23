"""Tests for asynchronous content export jobs.

Covers:

* ``POST /v1/content-export-jobs`` — required non-empty ``content_id`` and
  ``request_id`` (422 ``validation_error`` otherwise), unknown content
  ``404 content_not_found``, first creation ``201`` as ``pending`` with null
  ``started_at``/``finished_at``/``result``/``error``, idempotent retry of the
  same key ``200`` (and no second audit row), and the same ``request_id``
  reused for a different content ``409 content_export_request_conflict``.
* ``GET /v1/content-export-jobs/{job_id}`` — the job view, or
  ``404 content_export_job_not_found``.
* ``POST /v1/content-export-jobs/{job_id}/run`` — atomic claim of a pending
  job (``200``, ``running`` + UTC ``started_at``) settling to ``succeeded``
  with UTC ``finished_at`` and a result equal to the existing read-only
  export (``{"content", "claims"}``); a non-pending job is ``409 conflict``;
  a failed run settles ``failed`` with ``finished_at``, null ``result`` and
  ``content_export_failed``; an unknown id is ``404
  content_export_job_not_found``. The lifecycle and run audit events commit in
  the same transaction as their state changes.

All fixtures are deterministic and offline (in-memory SQLite, plus a
temporary-file SQLite database for the real concurrent-claim test).
"""

from __future__ import annotations

import hashlib
import threading
from datetime import timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update as sa_update

from provenance import service
from provenance.models import (
    AuditEvent,
    ContentExportJob,
)
from provenance.time_utils import parse_rfc3339_utc
from tests.helpers import create_actor


def _digest(name: str) -> str:
    return hashlib.sha256(f"job-{name}".encode()).hexdigest()


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


def _create_claim(client, content_id, marker, actor_id="org-1", claim_type="authorship"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": claim_type,
            "payload": {"marker": marker},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_bundle(client, claim_id, name, evidence_type="raw_capture"):
    resp = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim_id,
            "evidence_type": evidence_type,
            "digest_algorithm": "sha256",
            "digest_hex": _digest(f"evidence-{name}"),
            "media_type": "image/jpeg",
            "metadata": {"name": name},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_job(client, content_id, request_id):
    return client.post(
        "/v1/content-export-jobs",
        json={"content_id": content_id, "request_id": request_id},
    )


_JOB_KEYS = {
    "id",
    "content_id",
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
    create_actor(client)
    content = _create_content(client, "main")

    resp = _create_job(client, content["id"], "req-1")
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == _JOB_KEYS
    assert body["id"].startswith("cxj_")
    assert body["content_id"] == content["id"]
    assert body["request_id"] == "req-1"
    assert body["status"] == "pending"
    assert body["started_at"] is None
    assert body["finished_at"] is None
    assert body["result"] is None
    assert body["error"] is None
    assert body["created_at"].endswith("Z")


def test_create_job_writes_created_audit_event(client, db_session):
    create_actor(client)
    content = _create_content(client, "audited")

    body = _create_job(client, content["id"], "req-audit").json()
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.resource_id == body["id"],
            AuditEvent.event_type == "content_export_job.created",
        )
    ).scalars().all()
    assert len(events) == 1
    assert events[0].created_at.tzinfo is not None


def test_create_job_idempotent_same_key_returns_200(client, db_session):
    create_actor(client)
    content = _create_content(client, "idem")

    first = _create_job(client, content["id"], "retry-key")
    assert first.status_code == 201, first.text
    original = first.json()

    second = _create_job(client, content["id"], "retry-key")
    assert second.status_code == 200, second.text
    assert second.json() == original

    # The retry writes neither a second job row nor a second audit event, in
    # any lifecycle state.
    jobs = db_session.execute(
        select(ContentExportJob).where(
            ContentExportJob.request_id == "retry-key"
        )
    ).scalars().all()
    assert len(jobs) == 1
    events = db_session.execute(
        select(AuditEvent).where(AuditEvent.resource_id == original["id"])
    ).scalars().all()
    assert [e.event_type for e in events] == ["content_export_job.created"]


def test_create_job_same_request_id_different_content_is_409(client, db_session):
    create_actor(client)
    first = _create_content(client, "first")
    second = _create_content(client, "second")

    ok = _create_job(client, first["id"], "shared-key")
    assert ok.status_code == 201, ok.text

    conflict = _create_job(client, second["id"], "shared-key")
    assert conflict.status_code == 409, conflict.text
    err = conflict.json()["error"]
    assert err["code"] == "content_export_request_conflict"
    assert err["details"]["request_id"] == "shared-key"

    # The conflict writes no job and no audit event; the original job alone
    # remains bound to that request_id.
    jobs = db_session.execute(
        select(ContentExportJob).where(
            ContentExportJob.request_id == "shared-key"
        )
    ).scalars().all()
    assert len(jobs) == 1
    assert jobs[0].content_id == first["id"]


def test_distinct_request_ids_for_same_content_are_independent_jobs(client):
    create_actor(client)
    content = _create_content(client, "multi")

    one = _create_job(client, content["id"], "req-one")
    two = _create_job(client, content["id"], "req-two")
    assert one.status_code == 201
    assert two.status_code == 201
    assert one.json()["id"] != two.json()["id"]


def test_re_register_same_key_after_settlement_is_200_unchanged(client):
    create_actor(client)
    content = _create_content(client, "resettled")
    created = _create_job(client, content["id"], "req-settled").json()
    settled = client.post(f"/v1/content-export-jobs/{created['id']}/run").json()
    assert settled["status"] == "succeeded"

    # Re-registering the same (request_id, content_id) after settlement is
    # still idempotent: 200 with the original settled job, no new row/event.
    retry = _create_job(client, content["id"], "req-settled")
    assert retry.status_code == 200, retry.text
    assert retry.json() == settled


@pytest.mark.parametrize(
    "payload",
    [
        {"request_id": "r"},  # missing content_id
        {"content_id": "cnt_x"},  # missing request_id
        {"content_id": "", "request_id": "r"},  # empty content_id
        {"content_id": "cnt_x", "request_id": ""},  # empty request_id
        {"content_id": "   ", "request_id": "r"},  # whitespace content_id
        {"content_id": "cnt_x", "request_id": "   "},  # whitespace request_id
        {"content_id": None, "request_id": "r"},  # null content_id
        {"content_id": "cnt_x", "request_id": None},  # null request_id
        {  # undeclared field
            "content_id": "cnt_x",
            "request_id": "r",
            "unexpected": True,
        },
    ],
)
def test_create_job_validation_errors_422(client, payload):
    resp = client.post("/v1/content-export-jobs", json=payload)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def test_create_job_malformed_json_is_422(client):
    resp = client.post(
        "/v1/content-export-jobs",
        content="{not json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def test_create_job_unknown_content_is_404(client):
    create_actor(client)
    resp = _create_job(client, "cnt_does_not_exist", "req-x")
    assert resp.status_code == 404, resp.text
    err = resp.json()["error"]
    assert err["code"] == "content_not_found"
    assert err["details"]["content_id"] == "cnt_does_not_exist"


# --- read -------------------------------------------------------------------


def test_get_job_returns_job(client):
    create_actor(client)
    content = _create_content(client, "read")
    created = _create_job(client, content["id"], "req-get").json()

    resp = client.get(f"/v1/content-export-jobs/{created['id']}")
    assert resp.status_code == 200, resp.text
    assert resp.json() == created


def test_get_unknown_job_is_404(client):
    resp = client.get("/v1/content-export-jobs/cjx_does_not_exist")
    assert resp.status_code == 404, resp.text
    err = resp.json()["error"]
    assert err["code"] == "content_export_job_not_found"
    assert err["details"]["job_id"] == "cjx_does_not_exist"


# --- run: success -----------------------------------------------------------


def test_run_pending_job_succeeds_with_export_result(client):
    create_actor(client)
    content = _create_content(client, "exported")
    other = _create_content(client, "other")
    claim_a = _create_claim(client, content["id"], "a")
    claim_b = _create_claim(client, content["id"], "b", claim_type="review")
    # A claim on another content must never appear in this job's result.
    foreign = _create_claim(client, other["id"], "foreign")
    bundle_a2 = _create_bundle(client, claim_a["id"], "a2")
    bundle_a1 = _create_bundle(client, claim_a["id"], "a1")
    _create_bundle(client, foreign["id"], "foreign")

    job = _create_job(client, content["id"], "req-run").json()
    resp = client.post(f"/v1/content-export-jobs/{job['id']}/run")
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

    # The result is exactly the existing read-only export.
    expected = client.get(f"/v1/contents/{content['id']}/export").json()
    assert body["result"] == expected
    assert body["result"]["content"] == content
    assert [c["id"] for c in body["result"]["claims"]] == [
        claim_a["id"],
        claim_b["id"],
    ]
    assert [b["id"] for b in body["result"]["claims"][0]["evidence_bundles"]] == [
        bundle_a2["id"],
        bundle_a1["id"],
    ]
    assert body["result"]["claims"][1]["evidence_bundles"] == []

    # The settled state is what a subsequent read observes.
    assert (
        client.get(f"/v1/content-export-jobs/{job['id']}").json() == body
    )


def test_run_content_without_claims_has_empty_claims_result(client):
    create_actor(client)
    content = _create_content(client, "lonely")
    job = _create_job(client, content["id"], "req-empty").json()

    body = client.post(f"/v1/content-export-jobs/{job['id']}/run").json()
    assert body["status"] == "succeeded"
    expected = client.get(f"/v1/contents/{content['id']}/export").json()
    assert body["result"] == expected
    assert body["result"]["claims"] == []


def test_run_writes_one_run_audit_event(client, db_session):
    create_actor(client)
    content = _create_content(client, "audit-run")
    job = _create_job(client, content["id"], "req-audit-run").json()

    run = client.post(f"/v1/content-export-jobs/{job['id']}/run")
    assert run.status_code == 200, run.text

    events = db_session.execute(
        select(AuditEvent.event_type)
        .where(AuditEvent.resource_id == job["id"])
        .order_by(AuditEvent.seq)
    ).scalars().all()
    assert events == ["content_export_job.created", "content_export_job.run"]


# --- run: conflicts and missing ---------------------------------------------


def test_run_unknown_job_is_404(client):
    resp = client.post("/v1/content-export-jobs/cjx_missing/run")
    assert resp.status_code == 404, resp.text
    err = resp.json()["error"]
    assert err["code"] == "content_export_job_not_found"
    assert err["details"]["job_id"] == "cjx_missing"


def test_run_twice_only_first_runs(client, db_session):
    create_actor(client)
    content = _create_content(client, "rerun")
    job = _create_job(client, content["id"], "req-rerun").json()

    first = client.post(f"/v1/content-export-jobs/{job['id']}/run")
    assert first.status_code == 200, first.text
    assert first.json()["status"] == "succeeded"

    second = client.post(f"/v1/content-export-jobs/{job['id']}/run")
    assert second.status_code == 409, second.text
    assert second.json()["error"]["code"] == "conflict"

    # The rejected repeat changes no state and writes no second run event.
    final = client.get(f"/v1/content-export-jobs/{job['id']}").json()
    assert final == first.json()
    run_events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.resource_id == job["id"],
            AuditEvent.event_type == "content_export_job.run",
        )
    ).scalars().all()
    assert len(run_events) == 1


@pytest.mark.parametrize("state", ["running", "succeeded", "failed"])
def test_run_non_pending_job_is_409(client, db_session, state):
    create_actor(client)
    content = _create_content(client, f"state-{state}")
    job = _create_job(client, content["id"], f"req-{state}").json()

    # Move the job directly into a non-pending lifecycle state.
    db_session.execute(
        sa_update(ContentExportJob)
        .where(ContentExportJob.id == job["id"])
        .values(status=state)
    )
    db_session.commit()

    resp = client.post(f"/v1/content-export-jobs/{job['id']}/run")
    assert resp.status_code == 409, resp.text
    assert resp.json()["error"]["code"] == "conflict"

    # The rejected run leaves the state untouched and writes no run event.
    assert (
        client.get(f"/v1/content-export-jobs/{job['id']}").json()["status"]
        == state
    )


def test_run_settles_failed_when_export_raises(client, db_session, monkeypatch):
    create_actor(client)
    content = _create_content(client, "failing")
    job = _create_job(client, content["id"], "req-fail").json()

    def _boom(session, content_id):
        raise RuntimeError("simulated export failure")

    monkeypatch.setattr(service, "get_content_export", _boom)

    resp = client.post(f"/v1/content-export-jobs/{job['id']}/run")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "failed"
    assert body["started_at"] is not None
    assert body["finished_at"] is not None
    assert body["result"] is None
    assert body["error"] == "content_export_failed"

    # The failure is durable; a failed job can never be (re)claimed, even
    # after the fault clears.
    monkeypatch.undo()
    assert (
        client.post(f"/v1/content-export-jobs/{job['id']}/run").status_code
        == 409
    )
    final = client.get(f"/v1/content-export-jobs/{job['id']}").json()
    assert final["status"] == "failed"
    assert final["error"] == "content_export_failed"

    # The failed run still records exactly one run audit event.
    events = db_session.execute(
        select(AuditEvent.event_type)
        .where(AuditEvent.resource_id == job["id"])
        .order_by(AuditEvent.seq)
    ).scalars().all()
    assert events == ["content_export_job.created", "content_export_job.run"]


def test_concurrent_runs_claim_exactly_once(file_app):
    """Real concurrency over a file-backed SQLite database.

    The pending->running claim is a single conditional UPDATE, so of many
    simultaneous run calls exactly one must return 200 (settling succeeded)
    and every other must be a 409; the final state is succeeded with a
    single run audit event, regardless of thread interleaving.
    """
    # Tables and app.state are created eagerly by create_app, so plain
    # TestClients (without the lifespan context) serve requests; they are
    # deliberately not closed inside workers so the shared engine is not
    # disposed while sibling threads still hold connections.
    setup = TestClient(file_app)
    create_actor(setup)
    content = _create_content(setup, "concurrent")
    job_id = _create_job(setup, content["id"], "req-concurrent").json()["id"]

    outcomes: list[tuple[int, str]] = []
    lock = threading.Lock()

    def _fire() -> None:
        worker = TestClient(file_app)
        resp = worker.post(f"/v1/content-export-jobs/{job_id}/run")
        marker = (
            resp.json().get("status") or resp.json()["error"]["code"]
        )
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
    final = reader.get(f"/v1/content-export-jobs/{job_id}").json()
    assert final["status"] == "succeeded"
    session = file_app.state.session_factory()
    try:
        run_events = session.execute(
            select(AuditEvent).where(
                AuditEvent.resource_id == job_id,
                AuditEvent.event_type == "content_export_job.run",
            )
        ).scalars().all()
        assert len(run_events) == 1
    finally:
        session.close()
