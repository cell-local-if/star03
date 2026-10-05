"""Tests for the read-only asynchronous task-queue summary.

Covers ``GET /v1/observability/task-queues``:

* the success body is compact UTF-8 JSON terminated by exactly one newline,
  with fixed root members ``task_queues`` and ``checked_at``; ``task_queues``
  holds exactly the ``content_export``, ``audit_checkpoint``, and
  ``evidence_bundle_export`` queues, each with exactly the four lifecycle
  counts and the four oldest-pending/oldest-running members;
* counts cover every existing task of a queue (settled ones included) as
  non-negative integers; an empty database yields three all-zero queues
  with every oldest member null, never a missing-resource error;
* the oldest pending and oldest running task of each queue is the first row
  in the stable creation order (``created_at`` plus the monotonic insertion
  tiebreaker), reported with its stable id and its UTC creation/start time;
  an empty state makes both members of the pair null;
* the read is strictly read-only: success, empty results, concurrency,
  rejections, and the 503 path create, modify, or delete no task, resource,
  or audit event, and unchanged state yields identical bodies except for
  ``checked_at``;
* any request-body byte (whitespace, arbitrary bytes, malformed JSON, an
  object) and any query parameter (unknown, blank, repeated, or multiple)
  is a 422 validation_error rejected before any database query; POST, PUT,
  PATCH, and DELETE are 405 method_not_allowed;
* an unreadable database or an internal summary-query failure is the
  existing-structure 503 service_unavailable carrying a
  ``details.reason``, never a partial queue summary;
* the existing ``GET /v1/observability/summary`` and the three task
  families' creation/query/run/run-next/summary routes stay compatible.

All fixtures are deterministic and offline (in-memory and temporary-file
SQLite, no network).
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text, update as sa_update

from provenance.app import create_app
from provenance.api import get_db
from provenance.config import Settings
from provenance.models import (
    AuditCheckpointJob,
    AuditEvent,
    ContentExportJob,
    EvidenceBundleExportJob,
)
from provenance.time_utils import parse_rfc3339_utc, utc_now
from tests.helpers import (
    DIGEST_A,
    content_payload,
    create_actor,
)

URL = "/v1/observability/task-queues"
SUMMARY_URL = "/v1/observability/summary"

_QUEUE_KEYS = ("content_export", "audit_checkpoint", "evidence_bundle_export")
_COUNT_KEYS = ("pending", "running", "succeeded", "failed")
_QUEUE_MEMBERS = (
    "pending",
    "running",
    "succeeded",
    "failed",
    "oldest_pending_id",
    "oldest_pending_created_at",
    "oldest_running_id",
    "oldest_running_started_at",
)

_EMPTY_QUEUE = {
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


def _digest(name: str) -> str:
    return hashlib.sha256(f"task-queues-{name}".encode()).hexdigest()


def _make_content(client, name="q", actor_id="org-1"):
    resp = client.post(
        "/v1/contents",
        json=content_payload(
            actor_id=actor_id, digest=_digest(name), title=name
        ),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make_claim(client, content_id):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": "org-1",
            "claim_type": "authorship",
            "payload": {"statement": "observed"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make_bundle(client, claim_id, name="q"):
    resp = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim_id,
            "evidence_type": "raw_capture",
            "digest_algorithm": "sha256",
            "digest_hex": _digest(f"bundle-{name}"),
            "media_type": "image/jpeg",
            "metadata": {"source": "camera"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_content_job(client, content_id, request_id):
    resp = client.post(
        "/v1/content-export-jobs",
        json={"content_id": content_id, "request_id": request_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_checkpoint_job(client, request_id):
    resp = client.post(
        "/v1/audit-checkpoint-jobs", json={"request_id": request_id}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_evidence_job(client, evidence_bundle_id, request_id):
    resp = client.post(
        "/v1/evidence-bundle-export-jobs",
        json={
            "evidence_bundle_id": evidence_bundle_id,
            "request_id": request_id,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _set_state(db_session, model, job_id, status_value, **timestamps):
    db_session.execute(
        sa_update(model)
        .where(model.id == job_id)
        .values(status=status_value, **timestamps)
    )
    db_session.commit()


def _audit_total(db_session) -> int:
    return db_session.scalar(select(func.count()).select_from(AuditEvent))


def _populate(client, db_session):
    """Put every queue into a known mixed-state world.

    content_export: two pending, one running (fixed started_at), one
    succeeded via the real run route, one failed (direct state set).
    audit_checkpoint: one pending, one running (fixed started_at), one
    succeeded via the real run route, one failed (direct state set).
    evidence_bundle_export: one pending, one running (fixed started_at), one
    succeeded via the real run route, one failed (direct state set).
    """
    create_actor(client)
    content = _make_content(client, "world")
    claim = _make_claim(client, content["id"])
    bundle = _make_bundle(client, claim["id"], "world")

    # --- content export: 2 pending / 1 running / 1 succeeded / 1 failed.
    ce_pending_a = _create_content_job(client, content["id"], "ce-pending-a")
    ce_pending_b = _create_content_job(client, content["id"], "ce-pending-b")
    ce_running = _create_content_job(client, content["id"], "ce-running")
    ce_running_started = utc_now() - timedelta(seconds=5)
    _set_state(
        db_session,
        ContentExportJob,
        ce_running["id"],
        "running",
        started_at=ce_running_started,
    )
    ce_succeeded = _create_content_job(client, content["id"], "ce-succeeded")
    run = client.post(f"/v1/content-export-jobs/{ce_succeeded['id']}/run")
    assert run.status_code == 200, run.text
    assert run.json()["status"] == "succeeded"
    ce_failed = _create_content_job(client, content["id"], "ce-failed")
    _set_state(
        db_session,
        ContentExportJob,
        ce_failed["id"],
        "failed",
        finished_at=utc_now(),
    )

    # --- audit checkpoint: 1 pending / 1 running / 1 succeeded / 1 failed.
    ac_pending = _create_checkpoint_job(client, "ac-pending")
    ac_running = _create_checkpoint_job(client, "ac-running")
    ac_running_started = utc_now() - timedelta(seconds=7)
    _set_state(
        db_session,
        AuditCheckpointJob,
        ac_running["id"],
        "running",
        started_at=ac_running_started,
    )
    ac_succeeded = _create_checkpoint_job(client, "ac-succeeded")
    run = client.post(f"/v1/audit-checkpoint-jobs/{ac_succeeded['id']}/run")
    assert run.status_code == 200, run.text
    assert run.json()["status"] == "succeeded"
    ac_failed = _create_checkpoint_job(client, "ac-failed")
    _set_state(
        db_session,
        AuditCheckpointJob,
        ac_failed["id"],
        "failed",
        finished_at=utc_now(),
    )

    # --- evidence bundle export: 1 pending / 1 running / 1 succeeded / 1.
    eb_pending = _create_evidence_job(client, bundle["id"], "eb-pending")
    eb_running = _create_evidence_job(client, bundle["id"], "eb-running")
    eb_running_started = utc_now() - timedelta(seconds=9)
    _set_state(
        db_session,
        EvidenceBundleExportJob,
        eb_running["id"],
        "running",
        started_at=eb_running_started,
    )
    eb_succeeded = _create_evidence_job(client, bundle["id"], "eb-succeeded")
    run = client.post(f"/v1/evidence-bundle-export-jobs/{eb_succeeded['id']}/run")
    assert run.status_code == 200, run.text
    assert run.json()["status"] == "succeeded"
    eb_failed = _create_evidence_job(client, bundle["id"], "eb-failed")
    _set_state(
        db_session,
        EvidenceBundleExportJob,
        eb_failed["id"],
        "failed",
        finished_at=utc_now(),
    )

    return {
        "content_export": {
            "counts": {"pending": 2, "running": 1, "succeeded": 1, "failed": 1},
            "oldest_pending": (ce_pending_a["id"], ce_pending_a["created_at"]),
            "oldest_running": (ce_running["id"], ce_running_started),
        },
        "audit_checkpoint": {
            "counts": {"pending": 1, "running": 1, "succeeded": 1, "failed": 1},
            "oldest_pending": (ac_pending["id"], ac_pending["created_at"]),
            "oldest_running": (ac_running["id"], ac_running_started),
        },
        "evidence_bundle_export": {
            "counts": {"pending": 1, "running": 1, "succeeded": 1, "failed": 1},
            "oldest_pending": (eb_pending["id"], eb_pending["created_at"]),
            "oldest_running": (eb_running["id"], eb_running_started),
        },
    }


# --- Empty database ----------------------------------------------------------


def test_empty_database_three_zero_queues_with_null_metrics(client):
    before = utc_now()
    resp = client.get(URL)
    after = utc_now()
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()

    assert list(body) == ["task_queues", "checked_at"]
    assert list(body["task_queues"]) == list(_QUEUE_KEYS)
    for key in _QUEUE_KEYS:
        assert body["task_queues"][key] == _EMPTY_QUEUE
        assert list(body["task_queues"][key]) == list(_QUEUE_MEMBERS)

    checked_at = parse_rfc3339_utc(body["checked_at"])
    assert checked_at is not None
    assert before <= checked_at <= after


def test_empty_body_is_compact_json_with_single_trailing_newline(client):
    resp = client.get(URL)
    raw = resp.content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    # Compact document: no incidental whitespace anywhere.
    assert b" " not in raw
    # The whole queue block is byte-stable when the database is empty; only
    # checked_at differs between reads.
    expected_prefix = (
        b'{"task_queues":{'
        b'"content_export":'
        + json.dumps(_EMPTY_QUEUE, separators=(",", ":")).encode()
        + b',"audit_checkpoint":'
        + json.dumps(_EMPTY_QUEUE, separators=(",", ":")).encode()
        + b',"evidence_bundle_export":'
        + json.dumps(_EMPTY_QUEUE, separators=(",", ":")).encode()
        + b'},"checked_at":"'
    )
    assert raw.startswith(expected_prefix)
    assert raw.endswith(b'"}\n')
    stamped = raw[len(expected_prefix) : -len(b'"}\n')].decode()
    assert parse_rfc3339_utc(stamped) is not None


def test_empty_database_is_not_a_missing_resource_and_writes_nothing(
    client, db_session
):
    # Repeated reads of an empty database stay successful and create no row.
    for _ in range(3):
        resp = client.get(URL)
        assert resp.status_code == 200, resp.text
    assert _audit_total(db_session) == 0
    for model in (ContentExportJob, AuditCheckpointJob, EvidenceBundleExportJob):
        assert db_session.scalar(select(func.count()).select_from(model)) == 0


# --- Populated counts and oldest metrics -------------------------------------


def test_populated_counts_and_oldest_metrics(client, db_session):
    expected = _populate(client, db_session)

    resp = client.get(URL)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    queues = body["task_queues"]

    for key in _QUEUE_KEYS:
        queue = queues[key]
        assert list(queue) == list(_QUEUE_MEMBERS)
        for state in _COUNT_KEYS:
            value = queue[state]
            assert isinstance(value, int)
            assert value >= 0
            assert value == expected[key]["counts"][state]

        pending_id, pending_created = expected[key]["oldest_pending"]
        assert queue["oldest_pending_id"] == pending_id
        assert queue["oldest_pending_created_at"] == pending_created
        running_id, running_started = expected[key]["oldest_running"]
        assert queue["oldest_running_id"] == running_id
        # The oldest running task reports its own non-null claim time.
        served_started = parse_rfc3339_utc(queue["oldest_running_started_at"])
        assert served_started == running_started
        assert parse_rfc3339_utc(queue["oldest_pending_created_at"]) is not None


def test_counts_cover_every_existing_row(client, db_session):
    _populate(client, db_session)
    queues = client.get(URL).json()["task_queues"]
    table_rows = {
        "content_export": db_session.scalar(
            select(func.count()).select_from(ContentExportJob)
        ),
        "audit_checkpoint": db_session.scalar(
            select(func.count()).select_from(AuditCheckpointJob)
        ),
        "evidence_bundle_export": db_session.scalar(
            select(func.count()).select_from(EvidenceBundleExportJob)
        ),
    }
    for key in _QUEUE_KEYS:
        assert sum(queues[key][state] for state in _COUNT_KEYS) == table_rows[key]


def test_oldest_pending_skips_older_non_pending_tasks(client, db_session):
    create_actor(client)
    content = _make_content(client, "skip")
    oldest = _create_content_job(client, content["id"], "oldest-settled")
    middle = _create_content_job(client, content["id"], "middle-running")
    youngest = _create_content_job(client, content["id"], "youngest-pending")
    _set_state(db_session, ContentExportJob, oldest["id"], "succeeded")
    _set_state(
        db_session,
        ContentExportJob,
        middle["id"],
        "running",
        started_at=utc_now(),
    )

    queue = client.get(URL).json()["task_queues"]["content_export"]
    assert queue["pending"] == 1
    assert queue["oldest_pending_id"] == youngest["id"]
    assert queue["oldest_pending_created_at"] == youngest["created_at"]
    # The older running task is still the running head.
    assert queue["oldest_running_id"] == middle["id"]


def test_oldest_running_follows_creation_order_not_started_at(
    client, db_session
):
    # The oldest running task is the creation-order-first running row, even
    # though a later-created task was claimed earlier in wall-clock time.
    create_actor(client)
    content = _make_content(client, "running-order")
    first_created = _create_content_job(client, content["id"], "first-created")
    second_created = _create_content_job(client, content["id"], "second-created")
    earlier_start = utc_now() - timedelta(seconds=30)
    later_start = utc_now() - timedelta(seconds=2)
    _set_state(
        db_session,
        ContentExportJob,
        first_created["id"],
        "running",
        started_at=later_start,
    )
    _set_state(
        db_session,
        ContentExportJob,
        second_created["id"],
        "running",
        started_at=earlier_start,
    )

    queue = client.get(URL).json()["task_queues"]["content_export"]
    assert queue["running"] == 2
    assert queue["oldest_running_id"] == first_created["id"]
    assert parse_rfc3339_utc(queue["oldest_running_started_at"]) == later_start


def test_oldest_tie_on_identical_created_at_uses_insertion_order(
    client, db_session
):
    # Equal created_at values resolve by the monotonic insertion surrogate:
    # the first-inserted row wins, deterministically across reads.
    create_actor(client)
    content = _make_content(client, "tie")
    first = _create_content_job(client, content["id"], "tie-first")
    second = _create_content_job(client, content["id"], "tie-second")
    tied_created = utc_now() - timedelta(seconds=12)
    _set_state(
        db_session,
        ContentExportJob,
        first["id"],
        "pending",
        created_at=tied_created,
    )
    _set_state(
        db_session,
        ContentExportJob,
        second["id"],
        "pending",
        created_at=tied_created,
    )

    queue = client.get(URL).json()["task_queues"]["content_export"]
    assert queue["oldest_pending_id"] == first["id"]
    assert parse_rfc3339_utc(queue["oldest_pending_created_at"]) == tied_created


def test_null_metrics_when_state_is_empty(client, db_session):
    # Only a succeeded job exists: both pending and running pairs are null.
    _create_checkpoint_job_empty_world(client)
    queue = client.get(URL).json()["task_queues"]["audit_checkpoint"]
    assert queue["succeeded"] == 1
    assert queue["pending"] == queue["running"] == queue["failed"] == 0
    assert queue["oldest_pending_id"] is None
    assert queue["oldest_pending_created_at"] is None
    assert queue["oldest_running_id"] is None
    assert queue["oldest_running_started_at"] is None


def _create_checkpoint_job_empty_world(client):
    job = client.post(
        "/v1/audit-checkpoint-jobs", json={"request_id": "only-settled"}
    )
    assert job.status_code == 201, job.text
    job_id = job.json()["id"]
    run = client.post(f"/v1/audit-checkpoint-jobs/{job_id}/run")
    assert run.status_code == 200, run.text
    return job_id


# --- Read-only / repeatability / concurrency --------------------------------


def test_success_reads_create_no_tasks_resources_or_audit_events(
    client, db_session
):
    _populate(client, db_session)
    audit_before = _audit_total(db_session)
    totals_before = {
        model: db_session.scalar(select(func.count()).select_from(model))
        for model in (ContentExportJob, AuditCheckpointJob, EvidenceBundleExportJob)
    }

    for _ in range(4):
        resp = client.get(URL)
        assert resp.status_code == 200, resp.text

    assert _audit_total(db_session) == audit_before
    for model, before in totals_before.items():
        assert (
            db_session.scalar(select(func.count()).select_from(model))
            == before
        )


def test_repeated_reads_identical_except_checked_at(client, db_session):
    expected = _populate(client, db_session)
    first = client.get(URL).json()
    second = client.get(URL).json()
    first_checked = parse_rfc3339_utc(first.pop("checked_at"))
    second_checked = parse_rfc3339_utc(second.pop("checked_at"))
    assert second_checked >= first_checked
    assert first == second


def test_concurrent_reads_return_identical_queues(file_app):
    with TestClient(file_app) as setup:
        create_actor(setup)
        content = _make_content(setup, "concurrent")
        _create_content_job(setup, content["id"], "ce-concurrent")
        _create_checkpoint_job(setup, "ac-concurrent")
        claim = _make_claim(setup, content["id"])
        bundle = _make_bundle(setup, claim["id"], "concurrent")
        _create_evidence_job(setup, bundle["id"], "eb-concurrent")

    outcomes: list[dict] = []
    errors: list[tuple] = []
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
    checked_times = {item.pop("checked_at") for item in outcomes}
    first = outcomes[0]
    for other in outcomes[1:]:
        assert other == first
    assert first["task_queues"]["content_export"]["pending"] == 1
    assert first["task_queues"]["audit_checkpoint"]["pending"] == 1
    assert first["task_queues"]["evidence_bundle_export"]["pending"] == 1
    assert len(checked_times) >= 1

    # Concurrent reads wrote nothing: one actor, one content, one claim,
    # one bundle, and three job creations account for all audit events.
    with file_app.state.engine.connect() as conn:
        audit_total = conn.execute(
            text("SELECT COUNT(*) FROM audit_events")
        ).scalar_one()
    assert audit_total == 7


def test_queues_stable_across_restart(tmp_db_url):
    app1 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app1) as first:
        session = app1.state.session_factory()
        try:
            expected = _populate(first, session)
            before = first.get(URL).json()
        finally:
            session.close()

    app2 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app2) as second:
        after = second.get(URL).json()

    assert after["task_queues"] == before["task_queues"]
    before_checked = parse_rfc3339_utc(before["checked_at"])
    after_checked = parse_rfc3339_utc(after["checked_at"])
    assert after_checked >= before_checked
    # The stable queue heads survive the restart unchanged.
    for key in _QUEUE_KEYS:
        assert (
            after["task_queues"][key]["oldest_pending_id"]
            == expected[key]["oldest_pending"][0]
        )


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
        lambda c: c.request(
            "GET", URL, content=b'{"unexpected": true}'
        ),
        lambda c: c.request("GET", URL, content=b"\x00\x01\x02"),
        lambda c: c.get(URL + "?x=1"),
        lambda c: c.get(URL + "?x="),
        lambda c: c.get(URL + "?status=pending"),
        lambda c: c.get(URL + "?queue=content_export"),
        lambda c: c.get(URL + "?x=1&x=2"),
        lambda c: c.get(URL + "?a=1&b=2"),
        lambda c: c.request("GET", URL + "?x=1", content=b"{}"),
    ],
)
def test_invalid_requests_are_422_and_write_nothing(
    client, db_session, send
):
    _populate(client, db_session)
    audit_before = _audit_total(db_session)

    resp = send(client)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"

    assert _audit_total(db_session) == audit_before
    # A valid read still succeeds after the rejection with the same state.
    ok = client.get(URL)
    assert ok.status_code == 200
    assert ok.json()["task_queues"]["content_export"]["pending"] == 2


@pytest.mark.parametrize("method", ("post", "put", "patch", "delete"))
def test_non_get_methods_are_405_and_write_nothing(client, db_session, method):
    _populate(client, db_session)
    audit_before = _audit_total(db_session)

    resp = getattr(client, method)(URL)
    assert resp.status_code == 405, (method, resp.text)
    assert resp.json()["error"]["code"] == "method_not_allowed"

    assert _audit_total(db_session) == audit_before
    assert set(resp.json()["error"]) == {"code", "message"}


# --- 503 service_unavailable -------------------------------------------------


def _client_with_dropped_table(file_app, table):
    # A TestClient created without entering its context never runs the
    # lifespan startup, so dropping a table here leaves the schema
    # unreadable for the summary instead of being silently recreated.
    with file_app.state.engine.begin() as conn:
        conn.execute(text(f"DROP TABLE {table}"))
    return TestClient(file_app)


def test_unreadable_database_returns_503_with_reason(file_app):
    client = _client_with_dropped_table(file_app, "content_export_jobs")
    resp = client.get(URL)
    assert resp.status_code == 503, resp.text
    error = resp.json()["error"]
    assert error["code"] == "service_unavailable"
    assert "message" in error
    assert error["details"]["reason"] == "database_unavailable"


def test_failure_in_a_later_queue_is_503_not_a_partial_summary(file_app):
    # The first two queues read fine; the missing third table must still
    # produce a full 503, never a body carrying the first two queues.
    client = _client_with_dropped_table(
        file_app, "evidence_bundle_export_jobs"
    )
    resp = client.get(URL)
    assert resp.status_code == 503, resp.text
    error = resp.json()["error"]
    assert error["code"] == "service_unavailable"
    assert error["details"]["reason"] == "database_unavailable"
    assert "task_queues" not in resp.text


def test_validation_runs_before_the_unreadable_database(file_app):
    # Empty or illegal input is rejected first; the database is never read,
    # so an unreadable database answers these with 422 rather than 503. The
    # single valid empty request reaches the queries and gets the 503.
    client = _client_with_dropped_table(file_app, "audit_checkpoint_jobs")

    assert client.request("GET", URL, content=b"  ").status_code == 422
    assert client.get(URL + "?x=1").status_code == 422
    assert client.get(URL + "?x=1&x=2").status_code == 422

    valid = client.get(URL)
    assert valid.status_code == 503, valid.text


def test_503_rolls_back_and_leaves_database_usable(file_app):
    # Missing middle queue: the failed read must roll its transaction back,
    # leaving the connection usable for subsequent validation and writes.
    client = _client_with_dropped_table(file_app, "audit_checkpoint_jobs")
    first = client.get(URL)
    assert first.status_code == 503, first.text
    assert client.get(URL + "?x=1").status_code == 422
    # Unrelated tables remain readable and writable through their routes.
    resp = client.post(
        "/v1/actors",
        json={"id": "org-x", "name": "X", "type": "organization"},
    )
    assert resp.status_code == 201, resp.text


class _BrokenQueueSession:
    """A session whose queue reads raise a non-SQLAlchemy failure."""

    def scalar(self, *_args, **_kwargs):
        raise RuntimeError("queue machinery broken")

    def execute(self, *_args, **_kwargs):
        raise RuntimeError("queue machinery broken")

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_internal_summary_failure_is_503_with_reason(app, client):
    # A non-database internal failure while assembling the summary is the
    # existing-structure 503 carrying a reason, never a partial body or 500.
    app.dependency_overrides[get_db] = lambda: _BrokenQueueSession()
    try:
        resp = client.get(URL)
    finally:
        app.dependency_overrides.pop(get_db, None)
    assert resp.status_code == 503, resp.text
    error = resp.json()["error"]
    assert error["code"] == "service_unavailable"
    assert error["details"]["reason"] == "internal_error"


# --- Compatibility -----------------------------------------------------------


def test_task_queues_and_existing_summary_coexist(client, db_session):
    _populate(client, db_session)

    queues = client.get(URL)
    assert queues.status_code == 200, queues.text

    # The existing summary is unchanged in shape and still summarizes only
    # the content export task family.
    summary = client.get(SUMMARY_URL)
    assert summary.status_code == 200, summary.text
    body = summary.json()
    assert list(body) == [
        "service_status",
        "database_status",
        "resource_counts",
        "task_counts",
        "audit_status",
        "checked_at",
    ]
    assert body["task_counts"] == {
        "pending": 2,
        "running": 1,
        "succeeded": 1,
        "failed": 1,
    }
    # The new route reports all three families, content matching the summary.
    assert (
        queues.json()["task_queues"]["content_export"]["pending"]
        == body["task_counts"]["pending"]
    )


def test_existing_per_queue_summary_and_job_routes_unchanged(client, db_session):
    _populate(client, db_session)

    content_summary = client.get("/v1/content-export-jobs/summary")
    assert content_summary.status_code == 200
    assert content_summary.json()["pending"] == 2
    assert content_summary.json()["running"] == 1

    checkpoint_summary = client.get("/v1/audit-checkpoint-jobs/summary")
    assert checkpoint_summary.status_code == 200
    assert checkpoint_summary.json()["pending"] == 1

    evidence_summary = client.get("/v1/evidence-bundle-export-jobs/summary")
    assert evidence_summary.status_code == 200
    assert evidence_summary.json()["pending"] == 1

    # run-next on a pending queue still claims the head without error.
    claimed = client.post("/v1/audit-checkpoint-jobs/run-next")
    assert claimed.status_code == 200, claimed.text
    assert claimed.json()["status"] == "succeeded"
    after = client.get(URL).json()["task_queues"]["audit_checkpoint"]
    assert after["pending"] == 0
    assert after["succeeded"] == 2
