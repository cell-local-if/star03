"""Tests for the read-only asynchronous task-queue observability summary.

Covers ``GET /v1/observability/task-queues``:

* the success body is compact UTF-8 JSON terminated by exactly one newline,
  with root members in the fixed order ``task_queues`` then ``checked_at``;
  ``task_queues`` holds exactly the fixed keys ``content_export``,
  ``audit_checkpoint``, and ``evidence_bundle_export``, each with the eight
  fixed members (four counts, oldest pending id/created_at, oldest running
  id/started_at); timestamps stay UTC and every number renders as an
  integer;
* counts cover every existing task of the family; the oldest pending and
  running members name the task earliest in the stable creation order, with
  null oldest members for an empty state, no started_at for pending, and a
  non-null started_at for running;
* an empty database yields three all-zero queues with null oldest members,
  never a missing-resource error;
* any non-empty body (whitespace, arbitrary bytes, malformed JSON) and any
  query parameter (unknown, blank, repeated) is a 422 validation_error,
  rejected before any state is read -- even when the database is
  unreadable; non-GET methods are 405 method_not_allowed;
* an unreadable database or an internal summary-query failure is the
  existing-structure 503 service_unavailable carrying a reason, never a
  partial body, and the read transaction rolls back;
* repeated, concurrent, and post-restart reads return identical queues for
  unchanged persisted state; only the check time changes;
* the endpoint is strictly read-only: success, empty results, rejections,
  and the 503 path create, modify, or delete no resource, task, or audit
  event;
* the existing ``GET /v1/observability/summary`` and the three task
  families' create/query/run/run-next/summary routes are unchanged.

All fixtures are deterministic and offline (in-memory and temporary-file
SQLite, no network).
"""

from __future__ import annotations

import json
import threading
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, update as sa_update

from provenance.app import create_app
from provenance.config import Settings
from provenance.models import (
    AuditCheckpointJob,
    AuditEvent,
    ContentExportJob,
    EvidenceBundleExportJob,
)
from provenance.time_utils import parse_rfc3339_utc, utc_now
from tests.helpers import DIGEST_A, DIGEST_B, content_payload, create_actor

URL = "/v1/observability/task-queues"
SUMMARY_URL = "/v1/observability/summary"

QUEUE_KEYS = ["content_export", "audit_checkpoint", "evidence_bundle_export"]
QUEUE_MEMBERS = [
    "pending",
    "running",
    "succeeded",
    "failed",
    "oldest_pending_id",
    "oldest_pending_created_at",
    "oldest_running_id",
    "oldest_running_started_at",
]
EMPTY_QUEUE = {
    "pending": 0,
    "running": 0,
    "succeeded": 0,
    "failed": 0,
    "oldest_pending_id": None,
    "oldest_pending_created_at": None,
    "oldest_running_id": None,
    "oldest_running_started_at": None,
}


# --- World construction ------------------------------------------------------


def _make_content(client, name, actor_id="org-1", digest=DIGEST_A):
    resp = client.post(
        "/v1/contents",
        json=content_payload(actor_id=actor_id, digest=digest, title=name),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make_bundle(client, actor_id="org-1"):
    content = _make_content(
        client, "bundle-content", actor_id=actor_id, digest=DIGEST_B
    )
    claim = client.post(
        "/v1/claims",
        json={
            "content_id": content["id"],
            "actor_id": actor_id,
            "claim_type": "authorship",
            "payload": {"statement": "observed"},
        },
    )
    assert claim.status_code == 201, claim.text
    bundle = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim.json()["id"],
            "evidence_type": "raw_capture",
            "digest_algorithm": "sha256",
            "digest_hex": DIGEST_A,
            "media_type": "image/jpeg",
            "metadata": {"source": "camera"},
        },
    )
    assert bundle.status_code == 201, bundle.text
    return bundle.json()


def _make_content_job(client, content_id, request_id):
    resp = client.post(
        "/v1/content-export-jobs",
        json={"content_id": content_id, "request_id": request_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make_audit_job(client, request_id):
    resp = client.post(
        "/v1/audit-checkpoint-jobs", json={"request_id": request_id}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make_bundle_job(client, bundle_id, request_id):
    resp = client.post(
        "/v1/evidence-bundle-export-jobs",
        json={"evidence_bundle_id": bundle_id, "request_id": request_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _set_status(db_session, model, job_id, status_value, started_at=None):
    db_session.execute(
        sa_update(model)
        .where(model.id == job_id)
        .values(status=status_value, started_at=started_at)
    )
    db_session.commit()


def _populate(client, db_session):
    """Create tasks in all three families covering all four lifecycle states.

    Each family gets: two pending (the first is the oldest), two running
    (the first created is the oldest even though the second started
    earlier), one succeeded through the real run route, and one failed
    (direct state set).
    """
    create_actor(client)
    content = _make_content(client, "content")
    bundle = _make_bundle(client)

    makers = {
        "content_export": (
            ContentExportJob,
            lambda rid: _make_content_job(client, content["id"], rid),
            f"/v1/content-export-jobs/{{}}/run",
        ),
        "audit_checkpoint": (
            AuditCheckpointJob,
            lambda rid: _make_audit_job(client, rid),
            f"/v1/audit-checkpoint-jobs/{{}}/run",
        ),
        "evidence_bundle_export": (
            EvidenceBundleExportJob,
            lambda rid: _make_bundle_job(client, bundle["id"], rid),
            f"/v1/evidence-bundle-export-jobs/{{}}/run",
        ),
    }

    world = {}
    now = utc_now()
    for key, (model, make, run_path) in makers.items():
        pending_a = make(f"req-{key}-pending-a")
        make(f"req-{key}-pending-b")
        running_a = make(f"req-{key}-running-a")
        running_b = make(f"req-{key}-running-b")
        # The second running task started earlier; the oldest running member
        # still follows the stable creation order, not the start time.
        _set_status(
            db_session, model, running_a["id"], "running", started_at=now
        )
        _set_status(
            db_session,
            model,
            running_b["id"],
            "running",
            started_at=now - timedelta(hours=1),
        )
        succeeded = make(f"req-{key}-succeeded")
        run = client.post(run_path.format(succeeded["id"]))
        assert run.status_code == 200, run.text
        assert run.json()["status"] == "succeeded"
        failed = make(f"req-{key}-failed")
        _set_status(db_session, model, failed["id"], "failed")
        world[key] = {
            "oldest_pending_id": pending_a["id"],
            "oldest_pending_created_at": pending_a["created_at"],
            "oldest_running_id": running_a["id"],
        }
    return world


def _audit_total(db_session):
    return db_session.scalar(select(func.count()).select_from(AuditEvent))


def _job_totals(db_session):
    return {
        model.__tablename__: db_session.scalar(
            select(func.count()).select_from(model)
        )
        for model in (
            ContentExportJob,
            AuditCheckpointJob,
            EvidenceBundleExportJob,
        )
    }


# --- Empty database ----------------------------------------------------------


def test_empty_database_returns_three_zeroed_queues(client):
    before = utc_now()
    resp = client.get(URL)
    after = utc_now()
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()

    assert list(body) == ["task_queues", "checked_at"]
    assert list(body["task_queues"]) == QUEUE_KEYS
    for key in QUEUE_KEYS:
        assert body["task_queues"][key] == EMPTY_QUEUE

    checked_at = parse_rfc3339_utc(body["checked_at"])
    assert checked_at is not None
    assert before <= checked_at <= after


def test_empty_body_is_compact_json_with_one_newline(client):
    resp = client.get(URL)
    raw = resp.content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    # Compact document: no incidental whitespace.
    assert b" " not in raw
    body = json.loads(raw.decode("utf-8"))
    assert list(body) == ["task_queues", "checked_at"]
    for key in QUEUE_KEYS:
        assert list(body["task_queues"][key]) == QUEUE_MEMBERS


# --- Populated database ------------------------------------------------------


def test_populated_queues_cover_every_family_and_lifecycle(client, db_session):
    world = _populate(client, db_session)

    resp = client.get(URL)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert list(body["task_queues"]) == QUEUE_KEYS
    for key in QUEUE_KEYS:
        queue = body["task_queues"][key]
        assert list(queue) == QUEUE_MEMBERS
        for state in ("pending", "running", "succeeded", "failed"):
            assert isinstance(queue[state], int)
            assert queue[state] >= 0
        assert queue["pending"] == 2
        assert queue["running"] == 2
        assert queue["succeeded"] == 1
        assert queue["failed"] == 1

        expected = world[key]
        assert queue["oldest_pending_id"] == expected["oldest_pending_id"]
        # The pending oldest carries its UTC creation time, no start time.
        pending_created = parse_rfc3339_utc(queue["oldest_pending_created_at"])
        assert pending_created is not None
        assert (
            pending_created
            == parse_rfc3339_utc(expected["oldest_pending_created_at"])
        )
        # Creation order, not start order, picks the oldest running task.
        assert queue["oldest_running_id"] == expected["oldest_running_id"]
        running_started = parse_rfc3339_utc(queue["oldest_running_started_at"])
        assert running_started is not None

    checked_at = parse_rfc3339_utc(body["checked_at"])
    assert checked_at is not None


def test_counts_match_independent_totals(client, db_session):
    _populate(client, db_session)
    body = client.get(URL).json()
    totals = _job_totals(db_session)
    assert totals == {
        "content_export_jobs": 6,
        "audit_checkpoint_jobs": 6,
        "evidence_bundle_export_jobs": 6,
    }
    for key in QUEUE_KEYS:
        queue = body["task_queues"][key]
        assert (
            queue["pending"]
            + queue["running"]
            + queue["succeeded"]
            + queue["failed"]
            == 6
        )


# --- Read-only guarantee -----------------------------------------------------


def test_reads_create_no_tasks_or_audit_events(client, db_session):
    _populate(client, db_session)
    audit_before = _audit_total(db_session)
    jobs_before = _job_totals(db_session)

    for _ in range(3):
        resp = client.get(URL)
        assert resp.status_code == 200, resp.text

    assert _audit_total(db_session) == audit_before
    assert _job_totals(db_session) == jobs_before


def test_repeated_reads_are_identical_except_checked_at(client, db_session):
    _populate(client, db_session)
    first = client.get(URL).json()
    second = client.get(URL).json()
    first_checked = parse_rfc3339_utc(first.pop("checked_at"))
    second_checked = parse_rfc3339_utc(second.pop("checked_at"))
    assert first_checked is not None and second_checked is not None
    assert second_checked >= first_checked
    assert first == second


def test_concurrent_reads_return_identical_queues(file_app):
    setup = TestClient(file_app)
    create_actor(setup)
    content = _make_content(setup, "concurrent")
    _make_content_job(setup, content["id"], "req-concurrent")

    outcomes: list[dict] = []
    errors: list[object] = []
    lock = threading.Lock()

    def _read() -> None:
        worker = TestClient(file_app)
        resp = worker.get(URL)
        with lock:
            if resp.status_code != 200:
                errors.append((resp.status_code, resp.text))
            else:
                outcomes.append(resp.json())

    threads = [threading.Thread(target=_read) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(outcomes) == 6
    for item in outcomes:
        item.pop("checked_at")
    first = outcomes[0]
    for other in outcomes[1:]:
        assert other == first
    assert first["task_queues"]["content_export"]["pending"] == 1
    assert first["task_queues"]["audit_checkpoint"] == EMPTY_QUEUE
    assert first["task_queues"]["evidence_bundle_export"] == EMPTY_QUEUE

    session = file_app.state.session_factory()
    try:
        # One actor + one content + one job creation; reads add nothing.
        assert (
            session.scalar(select(func.count()).select_from(AuditEvent)) == 3
        )
    finally:
        session.close()


def test_queues_stable_across_restart(tmp_db_url):
    app1 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app1) as first:
        session = app1.state.session_factory()
        try:
            _populate(first, session)
            before = first.get(URL).json()
        finally:
            session.close()

    app2 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app2) as second:
        after = second.get(URL).json()

    assert after["task_queues"] == before["task_queues"]
    # Only the check time changes: it stays UTC and does not move backward.
    before_checked = parse_rfc3339_utc(before["checked_at"])
    after_checked = parse_rfc3339_utc(after["checked_at"])
    assert after_checked >= before_checked


# --- Request validation ------------------------------------------------------


@pytest.mark.parametrize(
    "send",
    [
        lambda c: c.request("GET", URL, content=b"   "),
        lambda c: c.request("GET", URL, content=b"\t\n"),
        lambda c: c.request(
            "GET",
            URL,
            content=b"{not json",
            headers={"content-type": "application/json"},
        ),
        lambda c: c.request("GET", URL, content=b"{}"),
        lambda c: c.request("GET", URL, content=b'{"unexpected": true}'),
        lambda c: c.get(URL + "?x=1"),
        lambda c: c.get(URL + "?x="),
        lambda c: c.get(URL + "?status=pending"),
        lambda c: c.get(URL + "?limit=1"),
        lambda c: c.get(URL + "?x=1&x=2"),
        lambda c: c.get(URL + "?x=1&y=2"),
        lambda c: c.request("GET", URL + "?x=1", content=b"{}"),
    ],
)
def test_invalid_requests_are_422_and_write_nothing(client, db_session, send):
    _populate(client, db_session)
    audit_before = _audit_total(db_session)
    jobs_before = _job_totals(db_session)

    resp = send(client)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"

    assert _audit_total(db_session) == audit_before
    assert _job_totals(db_session) == jobs_before
    # A valid read still works after the rejection.
    assert client.get(URL).status_code == 200


@pytest.mark.parametrize("method", ("post", "put", "patch", "delete"))
def test_non_get_methods_are_405_and_write_nothing(client, db_session, method):
    _populate(client, db_session)
    audit_before = _audit_total(db_session)
    jobs_before = _job_totals(db_session)

    resp = getattr(client, method)(URL)
    assert resp.status_code == 405, (method, resp.text)
    assert resp.json()["error"]["code"] == "method_not_allowed"

    assert _audit_total(db_session) == audit_before
    assert _job_totals(db_session) == jobs_before
    # The 405 response itself is the existing JSON error structure.
    assert set(resp.json()["error"]) == {"code", "message"}


# --- 503 service_unavailable -------------------------------------------------


def _client_with_dropped_table(file_app, table="content_export_jobs"):
    # A TestClient created without entering its context never runs the
    # lifespan startup, so dropping a table here leaves the schema unreadable
    # for the summary instead of being silently recreated.
    from sqlalchemy import text

    with file_app.state.engine.begin() as conn:
        conn.execute(text(f"DROP TABLE {table}"))
    return TestClient(file_app)


def test_unreadable_database_returns_503_with_reason(file_app):
    client = _client_with_dropped_table(file_app)
    resp = client.get(URL)
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


def test_validation_runs_before_the_unreadable_database(file_app):
    # Empty or illegal input is rejected first; the database is never read,
    # so an unreadable database still answers these with 422 rather than
    # 503. The valid empty request is the one that reaches the queries and
    # gets the 503.
    client = _client_with_dropped_table(file_app)

    whitespace = client.request("GET", URL, content=b"  ")
    assert whitespace.status_code == 422, whitespace.text
    malformed = client.request(
        "GET",
        URL,
        content=b"{bad",
        headers={"content-type": "application/json"},
    )
    assert malformed.status_code == 422, malformed.text
    unknown_param = client.get(URL + "?x=1")
    assert unknown_param.status_code == 422, unknown_param.text
    repeated = client.get(URL + "?x=1&x=2")
    assert repeated.status_code == 422, repeated.text

    valid = client.get(URL)
    assert valid.status_code == 503, valid.text


def test_503_rolls_back_and_leaves_database_usable(file_app):
    client = _client_with_dropped_table(
        file_app, table="evidence_bundle_export_jobs"
    )
    # The summary reads the dropped table; the failed read must roll its
    # transaction back, leaving the connection usable for subsequent
    # validation and for unrelated tables.
    first = client.get(URL)
    assert first.status_code == 503, first.text
    # A rejected request never opens the failing read.
    assert client.get(URL + "?x=1").status_code == 422
    # Unrelated tables remain readable and writable through their routes.
    resp = client.post(
        "/v1/actors",
        json={"id": "org-x", "name": "X", "type": "organization"},
    )
    assert resp.status_code == 201, resp.text


class _BrokenSummarySession:
    """A session whose summary reads raise a non-SQLAlchemy failure."""

    def scalar(self, *_args, **_kwargs):
        raise RuntimeError("summary machinery broken")

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
    from provenance.api import get_db

    app.dependency_overrides[get_db] = lambda: _BrokenSummarySession()
    try:
        resp = client.get(URL)
    finally:
        app.dependency_overrides.pop(get_db, None)
    assert resp.status_code == 503, resp.text
    error = resp.json()["error"]
    assert error["code"] == "service_unavailable"
    assert error["details"]["reason"] == "internal_error"


# --- Existing routes stay compatible -----------------------------------------


def test_existing_summary_and_task_routes_unchanged(client, db_session):
    world = _populate(client, db_session)

    # The existing running-state summary still answers with its own shape.
    summary = client.get(SUMMARY_URL)
    assert summary.status_code == 200, summary.text
    assert list(summary.json()) == [
        "service_status",
        "database_status",
        "resource_counts",
        "task_counts",
        "audit_status",
        "checked_at",
    ]
    assert summary.json()["task_counts"] == {
        "pending": 2,
        "running": 2,
        "succeeded": 1,
        "failed": 1,
    }

    # The three families' existing summary routes still answer.
    for path in (
        "/v1/content-export-jobs/summary",
        "/v1/audit-checkpoint-jobs/summary",
        "/v1/evidence-bundle-export-jobs/summary",
    ):
        resp = client.get(path)
        assert resp.status_code == 200, (path, resp.text)
        assert resp.json()["pending"] == 2

    # The existing by-id query routes still answer for the oldest pending.
    content_pending = world["content_export"]["oldest_pending_id"]
    resp = client.get(f"/v1/content-export-jobs/{content_pending}")
    assert resp.status_code == 200, resp.text
    assert resp.json()["id"] == content_pending
