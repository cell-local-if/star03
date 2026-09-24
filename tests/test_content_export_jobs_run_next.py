"""Tests for server-side queue claiming.

Covers ``POST /v1/content-export-jobs/run-next``:

* empty queue -> ``404 content_export_job_not_found`` with zero writes (no
  job, resource, or audit row);
* the single oldest ``pending`` job is claimed in stable creation order,
  non-pending jobs are skipped, younger pending jobs stay queued, and the
  queue drains oldest-first across calls (also across an app restart);
* the claimed run uses the existing atomic run semantics: succeeded with the
  export snapshot and UTC timestamps, or failed with null result and
  ``content_export_failed``, either way recording one
  ``content_export_job.run`` audit event;
* any non-empty body (whitespace, malformed JSON, an object, extra fields)
  or any query parameter (unknown, blank, repeated) is a 422
  ``validation_error`` that writes nothing and leaves the queue intact;
* a caller whose picked oldest job was claimed first gets ``409 conflict``
  without touching another job, creating a resource, or writing an audit
  event, and never falls through to a younger queued job;
* concurrent calls claim each job at most once.

All fixtures are deterministic and offline (in-memory and temporary-file
SQLite, no network).
"""

from __future__ import annotations

import threading
from datetime import timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update as sa_update

from provenance import service
from provenance.api import _content_export_result_payload
from provenance.app import create_app
from provenance.config import Settings
from provenance.errors import ContentExportJobConflictError
from provenance.models import AuditEvent, ContentExportJob
from provenance.time_utils import parse_rfc3339_utc
from tests.helpers import create_actor

RUN_NEXT_URL = "/v1/content-export-jobs/run-next"


def _digest(name: str) -> str:
    import hashlib

    return hashlib.sha256(f"runnext-{name}".encode()).hexdigest()


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


def _create_job(client, content_id, request_id):
    resp = client.post(
        "/v1/content-export-jobs",
        json={"content_id": content_id, "request_id": request_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# --- empty queue ------------------------------------------------------------


def test_run_next_empty_queue_is_404_and_writes_nothing(client, db_session):
    # No jobs at all: the queue is empty, so no job/resource/audit row may
    # exist before or after the rejected claim.
    assert db_session.execute(select(ContentExportJob)).scalars().all() == []

    resp = client.post(RUN_NEXT_URL)
    assert resp.status_code == 404, resp.text
    err = resp.json()["error"]
    assert err["code"] == "content_export_job_not_found"
    # No job id exists to report for a queue-wide miss.
    assert "job_id" not in err.get("details", {})

    assert db_session.execute(select(ContentExportJob)).scalars().all() == []
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_run_next_with_only_settled_jobs_is_404_zero_writes(client, db_session):
    create_actor(client)
    content = _create_content(client, "settled-only")
    job = _create_job(client, content["id"], "req-settled")
    run = client.post(f"/v1/content-export-jobs/{job['id']}/run")
    assert run.status_code == 200

    events_before = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    resp = client.post(RUN_NEXT_URL)
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "content_export_job_not_found"

    # The miss adds no job and no audit event.
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )
    assert (
        client.get(f"/v1/content-export-jobs/{job['id']}").json()["status"]
        == "succeeded"
    )


# --- stable oldest-first claiming -------------------------------------------


def test_run_next_claims_oldest_pending_and_leaves_rest_queued(client):
    create_actor(client)
    content = _create_content(client, "queue")
    oldest = _create_job(client, content["id"], "req-oldest")
    middle = _create_job(client, content["id"], "req-middle")
    youngest = _create_job(client, content["id"], "req-youngest")

    first = client.post(RUN_NEXT_URL)
    assert first.status_code == 200, first.text
    assert first.json()["id"] == oldest["id"]
    assert first.json()["status"] == "succeeded"

    # The other jobs are untouched: still pending with null run fields.
    for queued in (middle, youngest):
        view = client.get(f"/v1/content-export-jobs/{queued['id']}").json()
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
    create_actor(client)
    content = _create_content(client, "mixed")
    settled = _create_job(client, content["id"], "req-settled")
    failed = _create_job(client, content["id"], "req-failed-state")
    pending = _create_job(client, content["id"], "req-pending")

    db_session.execute(
        sa_update(ContentExportJob)
        .where(ContentExportJob.id == settled["id"])
        .values(status="succeeded")
    )
    db_session.execute(
        sa_update(ContentExportJob)
        .where(ContentExportJob.id == failed["id"])
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
        create_actor(first)
        content = _create_content(first, "restart")
        oldest = _create_job(first, content["id"], "req-r-oldest")
        youngest = _create_job(first, content["id"], "req-r-youngest")

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


def test_run_next_success_result_matches_export_snapshot(client, db_session):
    create_actor(client)
    content = _create_content(client, "exported")
    other = _create_content(client, "other")
    claim_a = _create_claim(client, content["id"], "a")
    claim_b = _create_claim(client, content["id"], "b", claim_type="review")
    _create_claim(client, other["id"], "foreign")
    job = _create_job(client, content["id"], "req-runnext-ok")

    resp = client.post(RUN_NEXT_URL)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == job["id"]
    assert body["status"] == "succeeded"

    started = parse_rfc3339_utc(body["started_at"])
    finished = parse_rfc3339_utc(body["finished_at"])
    assert started.tzinfo == timezone.utc
    assert finished.tzinfo == timezone.utc
    assert started <= finished
    assert body["error"] is None

    expected = client.get(f"/v1/contents/{content['id']}/export").json()
    assert body["result"] == expected
    assert [c["id"] for c in body["result"]["claims"]] == [
        claim_a["id"],
        claim_b["id"],
    ]

    events = db_session.execute(
        select(AuditEvent.event_type)
        .where(AuditEvent.resource_id == job["id"])
        .order_by(AuditEvent.seq)
    ).scalars().all()
    assert events == ["content_export_job.created", "content_export_job.run"]


def test_run_next_failure_settles_failed_then_queue_continues(
    client, db_session, monkeypatch
):
    create_actor(client)
    first_content = _create_content(client, "failing")
    second_content = _create_content(client, "healthy")
    first_job = _create_job(client, first_content["id"], "req-fail")
    second_job = _create_job(client, second_content["id"], "req-next-after-fail")

    def _boom(session, content_id):
        raise RuntimeError("simulated export failure")

    monkeypatch.setattr(service, "get_content_export", _boom)

    resp = client.post(RUN_NEXT_URL)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == first_job["id"]
    assert body["status"] == "failed"
    assert body["started_at"] is not None
    assert body["finished_at"] is not None
    assert body["result"] is None
    assert body["error"] == "content_export_failed"

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
        "content_export_job.created",
        "content_export_job.run",
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
def test_run_next_validation_errors_422_write_nothing(client, db_session, send):
    create_actor(client)
    content = _create_content(client, "guarded")
    job = _create_job(client, content["id"], "req-guarded")

    resp = send(client)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"

    # A rejected request leaves the queue intact: the pending job is still
    # claimable by a well-formed call, and no run audit event exists.
    view = client.get(f"/v1/content-export-jobs/{job['id']}").json()
    assert view["status"] == "pending"
    run_events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.resource_id == job["id"],
            AuditEvent.event_type == "content_export_job.run",
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
        create_actor(client)
        content = _create_content(client, "race")
        oldest = _create_job(client, content["id"], "req-race-oldest")
        younger = _create_job(client, content["id"], "req-race-younger")

    factory = application.state.session_factory

    session_a = factory()
    session_b = factory()
    try:
        # A reads the queue candidate first.
        picked = session_a.execute(
            select(ContentExportJob)
            .where(ContentExportJob.status == "pending")
            .order_by(
                ContentExportJob.created_at.asc(), ContentExportJob.seq.asc()
            )
            .limit(1)
        ).scalar_one_or_none()
        assert picked.id == oldest["id"]
        # End A's read transaction so B can commit on this file-backed
        # SQLite database; the picked candidate and its (stale) pending
        # status are retained on the object.
        session_a.commit()

        # B wins the same job outright through the real run path.
        settled = service.run_content_export_job(
            session_b, oldest["id"], _content_export_result_payload
        )
        assert settled.status == "succeeded"

        # A now runs its claim-and-settle against the stale pick. The CAS
        # finds the row no longer pending and must refuse.
        with pytest.raises(ContentExportJobConflictError) as exc_info:
            service._claim_and_run_content_export_job(
                session_a, picked, _content_export_result_payload
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
            client.get(f"/v1/content-export-jobs/{oldest['id']}").json()[
                "status"
            ]
            == "succeeded"
        )
        # The loser never fell through: the younger job is still queued.
        young_view = client.get(
            f"/v1/content-export-jobs/{younger['id']}"
        ).json()
        assert young_view["status"] == "pending"
        assert young_view["started_at"] is None

    session = verifier.state.session_factory()
    try:
        # Exactly one run audit event (the winner's), and none for the
        # younger job beyond its creation event.
        run_events = session.execute(
            select(AuditEvent).where(
                AuditEvent.event_type == "content_export_job.run"
            )
        ).scalars().all()
        assert [e.resource_id for e in run_events] == [oldest["id"]]
        younger_events = session.execute(
            select(AuditEvent.event_type)
            .where(AuditEvent.resource_id == younger["id"])
            .order_by(AuditEvent.seq)
        ).scalars().all()
        assert younger_events == ["content_export_job.created"]
    finally:
        session.close()


def test_concurrent_run_next_calls_claim_exactly_once(file_app):
    """Real concurrency over a file-backed SQLite database.

    Six simultaneous run-next calls against one pending job: exactly one
    returns 200 and settles succeeded; every other is a 409 conflict. The
    job ends succeeded with a single run audit event.
    """
    setup = TestClient(file_app)
    create_actor(setup)
    content = _create_content(setup, "concurrent")
    job_id = _create_job(setup, content["id"], "req-runnext-concurrent")["id"]

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
