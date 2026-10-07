"""Tests for stalled content export job recovery.

Covers ``POST /v1/content-export-jobs/recover-stalled``:

* request validation: an empty body, whitespace, malformed JSON, invalid
  UTF-8, a non-object document, a missing/extra/duplicate field, a
  non-integer or out-of-range ``older_than_seconds``, and any query
  parameter are all ``422 validation_error`` rejected before any job is
  read -- a stalled job in the queue stays ``running`` and no audit event
  is written;
* an empty or non-stalled queue is ``200`` with an empty ``items`` array
  and ``count`` 0, and no job in another state (or not yet old enough) is
  touched;
* every still-``running`` job whose ``started_at`` is strictly older than
  now-UTC minus ``older_than_seconds`` is settled ``failed`` in one
  transaction: null result, stable ``content_export_stalled`` error, the
  recovery instant as ``finished_at``, unchanged id/request_id/content_id/
  started_at, and one ``content_export_job.stalled_recovery`` audit event
  per job, all in stable ``started_at``/``seq`` order;
* a concurrent settler that changes any selected job after the selection
  is a ``409 content_export_job_recovery_conflict`` carrying the first
  contested job id, with the entire pass rolled back;
* non-POST methods on the path are ``405 method_not_allowed``;
* the recovered state persists across an app restart.

All fixtures are deterministic and offline (in-memory and temporary-file
SQLite, no network).
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update as sa_update

from provenance import service
from provenance.app import create_app
from provenance.config import Settings
from provenance.errors import ContentExportJobRecoveryConflictError
from provenance.models import AuditEvent, ContentExportJob
from provenance.time_utils import parse_rfc3339_utc, utc_now
from tests.helpers import create_actor

RECOVER_URL = "/v1/content-export-jobs/recover-stalled"


def _digest(name: str) -> str:
    import hashlib

    return hashlib.sha256(f"recover-{name}".encode()).hexdigest()


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


def _set_running(db_session, job_id, started_at):
    db_session.execute(
        sa_update(ContentExportJob)
        .where(ContentExportJob.id == job_id)
        .values(status="running", started_at=started_at)
    )
    db_session.commit()


def _job_row(db_session, job_id):
    return db_session.execute(
        select(ContentExportJob).where(ContentExportJob.id == job_id)
    ).scalar_one()


def _audit_events(db_session, resource_id):
    return (
        db_session.execute(
            select(AuditEvent.event_type)
            .where(AuditEvent.resource_id == resource_id)
            .order_by(AuditEvent.seq)
        )
        .scalars()
        .all()
    )


# --- validation: 422 before any job is read ----------------------------------


def _assert_validation_error(resp):
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


@pytest.fixture
def stalled_queue(client, db_session):
    """One stalled running job plus one pending job in the queue."""
    create_actor(client)
    content = _create_content(client, "stalled")
    stalled = _create_job(client, content["id"], "req-stalled")
    pending = _create_job(client, content["id"], "req-pending")
    # Older than the maximum allowed window (86400s), so every in-range
    # older_than_seconds value selects it.
    _set_running(
        db_session, stalled["id"], utc_now() - timedelta(seconds=90000)
    )
    return {"stalled": stalled, "pending": pending}


def _assert_queue_untouched(db_session, stalled, pending):
    stalled_row = _job_row(db_session, stalled["id"])
    assert stalled_row.status == "running"
    assert stalled_row.finished_at is None
    assert stalled_row.error is None
    assert _job_row(db_session, pending["id"]).status == "pending"
    # Only the two creation events exist: the rejection wrote nothing.
    assert _audit_events(db_session, stalled["id"]) == [
        "content_export_job.created"
    ]
    assert _audit_events(db_session, pending["id"]) == [
        "content_export_job.created"
    ]


def test_empty_body_is_422(client, db_session, stalled_queue):
    resp = client.post(RECOVER_URL)
    _assert_validation_error(resp)
    _assert_queue_untouched(db_session, **stalled_queue)


def test_whitespace_body_is_422(client, db_session, stalled_queue):
    resp = client.post(
        RECOVER_URL, content=b"  \n ", headers={"content-type": "application/json"}
    )
    _assert_validation_error(resp)
    _assert_queue_untouched(db_session, **stalled_queue)


def test_malformed_json_is_422(client, db_session, stalled_queue):
    resp = client.post(
        RECOVER_URL,
        content=b'{"older_than_seconds":',
        headers={"content-type": "application/json"},
    )
    _assert_validation_error(resp)
    _assert_queue_untouched(db_session, **stalled_queue)


def test_invalid_utf8_is_422(client, db_session, stalled_queue):
    resp = client.post(
        RECOVER_URL,
        content=b'\xff\xfe{"older_than_seconds": 10}',
        headers={"content-type": "application/json"},
    )
    _assert_validation_error(resp)
    _assert_queue_untouched(db_session, **stalled_queue)


@pytest.mark.parametrize("document", ["[1]", '"3600"', "5", "null", "true"])
def test_non_object_json_is_422(client, db_session, stalled_queue, document):
    resp = client.post(
        RECOVER_URL,
        content=document.encode(),
        headers={"content-type": "application/json"},
    )
    _assert_validation_error(resp)
    _assert_queue_untouched(db_session, **stalled_queue)


def test_missing_field_is_422(client, db_session, stalled_queue):
    resp = client.post(RECOVER_URL, json={})
    _assert_validation_error(resp)
    _assert_queue_untouched(db_session, **stalled_queue)


def test_extra_field_is_422(client, db_session, stalled_queue):
    resp = client.post(
        RECOVER_URL, json={"older_than_seconds": 10, "include_pending": True}
    )
    _assert_validation_error(resp)
    _assert_queue_untouched(db_session, **stalled_queue)


def test_duplicate_field_is_422(client, db_session, stalled_queue):
    resp = client.post(
        RECOVER_URL,
        content=b'{"older_than_seconds": 10, "older_than_seconds": 20}',
        headers={"content-type": "application/json"},
    )
    _assert_validation_error(resp)
    _assert_queue_untouched(db_session, **stalled_queue)


@pytest.mark.parametrize(
    "value", ["3600", 1.5, True, None, [10], {"seconds": 10}, 10.0]
)
def test_non_integer_value_is_422(client, db_session, stalled_queue, value):
    resp = client.post(RECOVER_URL, json={"older_than_seconds": value})
    _assert_validation_error(resp)
    _assert_queue_untouched(db_session, **stalled_queue)


@pytest.mark.parametrize("value", [0, -1, -86400, 86401, 100000])
def test_out_of_range_value_is_422(client, db_session, stalled_queue, value):
    resp = client.post(RECOVER_URL, json={"older_than_seconds": value})
    _assert_validation_error(resp)
    _assert_queue_untouched(db_session, **stalled_queue)


@pytest.mark.parametrize("query", ["older_than_seconds=10", "foo=bar", "foo="])
def test_any_query_param_is_422(client, db_session, stalled_queue, query):
    resp = client.post(
        f"{RECOVER_URL}?{query}", json={"older_than_seconds": 3600}
    )
    _assert_validation_error(resp)
    _assert_queue_untouched(db_session, **stalled_queue)


@pytest.mark.parametrize("value", [1, 86400])
def test_boundary_values_are_accepted(client, db_session, stalled_queue, value):
    resp = client.post(RECOVER_URL, json={"older_than_seconds": value})
    assert resp.status_code == 200, resp.text
    assert resp.json()["count"] == 1


# --- method not allowed -------------------------------------------------------


@pytest.mark.parametrize("method", ["get", "put", "patch", "delete"])
def test_non_post_methods_are_405(client, method):
    resp = getattr(client, method)(RECOVER_URL)
    assert resp.status_code == 405, resp.text
    assert resp.json()["error"]["code"] == "method_not_allowed"


# --- empty and non-stalled queues --------------------------------------------


def test_empty_queue_is_200_empty_items(client, db_session):
    resp = client.post(RECOVER_URL, json={"older_than_seconds": 3600})
    assert resp.status_code == 200, resp.text
    # Exactly the two members, compact JSON, one trailing newline.
    assert resp.content == b'{"items":[],"count":0}\n'
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_non_stalled_jobs_are_untouched(client, db_session):
    create_actor(client)
    content = _create_content(client, "fresh")
    pending = _create_job(client, content["id"], "req-pending")
    succeeded = _create_job(client, content["id"], "req-succeeded")
    run = client.post(f"/v1/content-export-jobs/{succeeded['id']}/run")
    assert run.status_code == 200
    failed = _create_job(client, content["id"], "req-failed")
    db_session.execute(
        sa_update(ContentExportJob)
        .where(ContentExportJob.id == failed["id"])
        .values(status="failed", error="content_export_failed")
    )
    young = _create_job(client, content["id"], "req-young-running")
    # Running but only 30 seconds old: younger than any allowed cutoff
    # (older_than_seconds >= 1 but the job is not "strictly older" than an
    # hour).
    _set_running(db_session, young["id"], utc_now() - timedelta(seconds=30))
    db_session.commit()

    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())

    resp = client.post(RECOVER_URL, json={"older_than_seconds": 3600})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0}

    assert _job_row(db_session, pending["id"]).status == "pending"
    assert _job_row(db_session, succeeded["id"]).status == "succeeded"
    failed_row = _job_row(db_session, failed["id"])
    assert failed_row.status == "failed"
    assert failed_row.error == "content_export_failed"
    young_row = _job_row(db_session, young["id"])
    assert young_row.status == "running"
    assert young_row.finished_at is None
    assert young_row.error is None
    # No recovery audit event was written for anyone.
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )


# --- recovery ----------------------------------------------------------------


def test_recovers_single_stalled_job(client, db_session):
    create_actor(client)
    content = _create_content(client, "single")
    job = _create_job(client, content["id"], "req-single")
    started_at = utc_now() - timedelta(seconds=7200)
    _set_running(db_session, job["id"], started_at)

    before = utc_now()
    resp = client.post(RECOVER_URL, json={"older_than_seconds": 3600})
    after = utc_now()
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 1
    assert len(body["items"]) == 1
    item = body["items"][0]

    # The public view: exactly the existing single-job fields.
    assert set(item) == {
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
    # Identity and associations are unchanged.
    assert item["id"] == job["id"]
    assert item["content_id"] == content["id"]
    assert item["request_id"] == "req-single"
    assert item["created_at"] == job["created_at"]
    assert parse_rfc3339_utc(item["started_at"]) == started_at
    # Settled as failed with the stable stalled error and a null result.
    assert item["status"] == "failed"
    assert item["error"] == "content_export_stalled"
    assert item["result"] is None
    finished_at = parse_rfc3339_utc(item["finished_at"])
    assert finished_at is not None
    assert before <= finished_at <= after

    # The persisted row matches the returned public view exactly.
    row = _job_row(db_session, job["id"])
    assert row.status == "failed"
    assert row.error == "content_export_stalled"
    assert row.result is None
    assert row.started_at == started_at
    assert row.finished_at == finished_at

    # One recovery audit event appended after the creation event.
    assert _audit_events(db_session, job["id"]) == [
        "content_export_job.created",
        "content_export_job.stalled_recovery",
    ]

    # The single-job read view reflects the settlement.
    view = client.get(f"/v1/content-export-jobs/{job['id']}").json()
    assert view == item


def test_cutoff_is_strict(client, db_session):
    create_actor(client)
    content = _create_content(client, "cutoff")
    older = _create_job(client, content["id"], "req-older")
    younger = _create_job(client, content["id"], "req-younger")
    # 3630 seconds old: strictly older than now minus 3600 -> recovered.
    _set_running(
        db_session, older["id"], utc_now() - timedelta(seconds=3630)
    )
    # 3570 seconds old: NOT strictly older than the cutoff -> untouched.
    _set_running(
        db_session, younger["id"], utc_now() - timedelta(seconds=3570)
    )

    resp = client.post(RECOVER_URL, json={"older_than_seconds": 3600})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 1
    assert [item["id"] for item in body["items"]] == [older["id"]]

    younger_row = _job_row(db_session, younger["id"])
    assert younger_row.status == "running"
    assert younger_row.finished_at is None
    assert younger_row.error is None
    assert _audit_events(db_session, younger["id"]) == [
        "content_export_job.created"
    ]


def test_recovers_in_started_at_then_seq_order(client, db_session):
    create_actor(client)
    content = _create_content(client, "order")
    jobs = [
        _create_job(client, content["id"], f"req-order-{i}") for i in range(5)
    ]
    # started_at ages deliberately shuffled against creation order.
    ages = [5000, 7000, 6000, 8000, 8000]
    for job, age in zip(jobs, ages):
        _set_running(
            db_session, job["id"], utc_now() - timedelta(seconds=age)
        )

    resp = client.post(RECOVER_URL, json={"older_than_seconds": 3600})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 5
    # started_at ascending; the two equally-old jobs keep creation (seq)
    # order.
    expected = [jobs[3], jobs[4], jobs[1], jobs[2], jobs[0]]
    assert [item["id"] for item in body["items"]] == [j["id"] for j in expected]

    # One recovery instant stamps every job settled by this pass.
    finished = {item["finished_at"] for item in body["items"]}
    assert len(finished) == 1

    # Each job carries exactly its creation event followed by its recovery
    # event, and the recovery events were appended in processing order.
    recovery_events = db_session.execute(
        select(AuditEvent.resource_id)
        .where(AuditEvent.event_type == "content_export_job.stalled_recovery")
        .order_by(AuditEvent.seq)
    ).scalars().all()
    assert recovery_events == [j["id"] for j in expected]
    for job in jobs:
        assert _audit_events(db_session, job["id"]) == [
            "content_export_job.created",
            "content_export_job.stalled_recovery",
        ]


def test_second_pass_recovers_nothing(client, db_session):
    create_actor(client)
    content = _create_content(client, "twice")
    job = _create_job(client, content["id"], "req-twice")
    _set_running(
        db_session, job["id"], utc_now() - timedelta(seconds=7200)
    )

    first = client.post(RECOVER_URL, json={"older_than_seconds": 3600})
    assert first.status_code == 200
    assert first.json()["count"] == 1

    # The settled job is failed, not running: a repeat pass is an empty 200
    # and writes no further audit event.
    second = client.post(RECOVER_URL, json={"older_than_seconds": 3600})
    assert second.status_code == 200
    assert second.json() == {"items": [], "count": 0}
    assert _audit_events(db_session, job["id"]) == [
        "content_export_job.created",
        "content_export_job.stalled_recovery",
    ]


def test_recovered_state_survives_restart(file_app, tmp_db_url):
    setup = TestClient(file_app)
    create_actor(setup)
    content = _create_content(setup, "restart")
    job = _create_job(setup, content["id"], "req-restart")
    session = file_app.state.session_factory()
    try:
        _set_running(
            session, job["id"], utc_now() - timedelta(seconds=7200)
        )
    finally:
        session.close()

    resp = setup.post(RECOVER_URL, json={"older_than_seconds": 3600})
    assert resp.status_code == 200, resp.text
    assert resp.json()["count"] == 1
    file_app.state.engine.dispose()

    # Reopen the same database file: the settlement and its audit event
    # persist, and a further pass recovers nothing.
    verifier = create_app(Settings(database_url=tmp_db_url))
    with TestClient(verifier) as client:
        view = client.get(f"/v1/content-export-jobs/{job['id']}").json()
        assert view["status"] == "failed"
        assert view["error"] == "content_export_stalled"
        assert view["result"] is None
        assert view["started_at"] is not None
        assert view["finished_at"] is not None
        again = client.post(RECOVER_URL, json={"older_than_seconds": 3600})
        assert again.status_code == 200
        assert again.json() == {"items": [], "count": 0}
    session = verifier.state.session_factory()
    try:
        assert _audit_events(session, job["id"]) == [
            "content_export_job.created",
            "content_export_job.stalled_recovery",
        ]
    finally:
        session.close()
        verifier.state.engine.dispose()


# --- conflict: a concurrent settler wins --------------------------------------


def test_concurrent_settler_rolls_back_entire_pass(tmp_db_url):
    """Two-session race over a file-backed SQLite database.

    Session A selects two stalled jobs; before A settles them, session B
    (a concurrent runner or recovery) settles the second one. A's pass must
    roll back entirely -- its first settlement included -- and raise the
    recovery conflict carrying the second job's id.
    """
    application = create_app(Settings(database_url=tmp_db_url))
    setup = TestClient(application)
    create_actor(setup)
    content = _create_content(setup, "conflict")
    first = _create_job(setup, content["id"], "req-conflict-first")
    second = _create_job(setup, content["id"], "req-conflict-second")
    factory = application.state.session_factory

    seed = factory()
    try:
        # Both stalled; "first" is the older of the two, so the pass
        # processes it before "second".
        _set_running(
            seed, first["id"], utc_now() - timedelta(seconds=7200)
        )
        _set_running(
            seed, second["id"], utc_now() - timedelta(seconds=5400)
        )
    finally:
        seed.close()

    session_a = factory()
    session_b = factory()
    try:
        # A reads the recovery selection first.
        picked = (
            session_a.execute(
                select(ContentExportJob)
                .where(
                    ContentExportJob.status == "running",
                    ContentExportJob.started_at
                    < utc_now() - timedelta(seconds=3600),
                )
                .order_by(
                    ContentExportJob.started_at.asc(),
                    ContentExportJob.seq.asc(),
                )
            )
            .scalars()
            .all()
        )
        assert [job.id for job in picked] == [first["id"], second["id"]]
        # End A's read transaction so B can commit on this file-backed
        # SQLite database; the stale selection is retained on the objects.
        session_a.commit()

        # B (a concurrent runner) settles the second job first.
        session_b.execute(
            sa_update(ContentExportJob)
            .where(ContentExportJob.id == second["id"])
            .values(
                status="failed",
                finished_at=utc_now(),
                error="content_export_failed",
            )
        )
        session_b.commit()

        # A's pass settles the first job, then its compare-and-set on the
        # second matches zero rows: the whole pass rolls back.
        with pytest.raises(ContentExportJobRecoveryConflictError) as exc_info:
            service._settle_stalled_content_export_jobs(session_a, picked)
        assert exc_info.value.code == "content_export_job_recovery_conflict"
        assert exc_info.value.details["job_id"] == second["id"]
    finally:
        session_a.close()
        session_b.close()
        application.state.engine.dispose()

    # Reopen to assert the final state: A's first settlement was rolled
    # back, B's settlement stands, and no recovery audit event exists.
    verifier = create_app(Settings(database_url=tmp_db_url))
    session = verifier.state.session_factory()
    try:
        first_row = _job_row(session, first["id"])
        assert first_row.status == "running"
        assert first_row.finished_at is None
        assert first_row.error is None
        second_row = _job_row(session, second["id"])
        assert second_row.status == "failed"
        assert second_row.error == "content_export_failed"
        assert _audit_events(session, first["id"]) == [
            "content_export_job.created"
        ]
        assert _audit_events(session, second["id"]) == [
            "content_export_job.created"
        ]
    finally:
        session.close()
        verifier.state.engine.dispose()
