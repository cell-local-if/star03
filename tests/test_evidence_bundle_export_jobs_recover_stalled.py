"""Tests for stalled evidence bundle export job recovery.

Covers ``POST /v1/evidence-bundle-export-jobs/recover-stalled``:

* the body is exactly one field, ``older_than_seconds`` (a decimal integer
  between 1 and 86400): an empty body, whitespace, malformed JSON, invalid
  UTF-8, a non-object document, a missing/extra/duplicate field, a
  non-integer or out-of-range value, and any query parameter are all a 422
  ``validation_error`` raised before any job is read, writing nothing;
* with no timed-out ``running`` job the response is 200 with an empty
  ``items`` and ``count`` zero, and no job or audit row changes;
* every ``running`` job whose ``started_at`` is strictly older than now
  minus ``older_than_seconds`` is settled ``failed`` in one transaction --
  ``finished_at`` is the recovery instant, ``result`` is null, ``error`` is
  ``evidence_bundle_export_stalled``, and
  id/evidence_bundle_id/request_id/started_at are unchanged -- with one
  ``evidence_bundle_export_job.stalled_recovery`` audit event per job,
  items returned in started_at/seq order;
* jobs in any other state and running jobs younger than the cutoff are
  never touched;
* a concurrent change to any selected job rolls the whole pass back into a
  ``409 evidence_bundle_export_job_recovery_conflict`` naming the first
  contested job, leaving every other state as it was;
* any non-POST method on the path is ``405 method_not_allowed``.

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
from provenance.errors import EvidenceBundleExportJobRecoveryConflictError
from provenance.models import AuditEvent, EvidenceBundleExportJob
from provenance.time_utils import parse_rfc3339_utc, utc_now
from tests.test_evidence_bundle_exchange import _setup_bundle

RECOVER_URL = "/v1/evidence-bundle-export-jobs/recover-stalled"

# Fixed instants far enough apart to be unambiguous under any allowed
# older_than_seconds value (1..86400 seconds).
STALLED_AT = "2026-01-01T00:00:00Z"


def _create_job(client, evidence_bundle_id, request_id):
    resp = client.post(
        "/v1/evidence-bundle-export-jobs",
        json={
            "evidence_bundle_id": evidence_bundle_id,
            "request_id": request_id,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _force_state(db_session, job_id, status, started_at=None):
    """Directly persist a lifecycle state a public route cannot produce."""
    db_session.execute(
        sa_update(EvidenceBundleExportJob)
        .where(EvidenceBundleExportJob.id == job_id)
        .values(status=status, started_at=started_at)
    )
    db_session.commit()


def _force_stalled(db_session, job_id, started_at=STALLED_AT):
    _force_state(
        db_session, job_id, "running", parse_rfc3339_utc(started_at)
    )


def _recovery_events(db_session):
    return db_session.execute(
        select(AuditEvent)
        .where(
            AuditEvent.event_type
            == "evidence_bundle_export_job.stalled_recovery"
        )
        .order_by(AuditEvent.seq)
    ).scalars().all()


def _audit_types(db_session, job_id):
    return db_session.execute(
        select(AuditEvent.event_type)
        .where(AuditEvent.resource_id == job_id)
        .order_by(AuditEvent.seq)
    ).scalars().all()


# --- empty and non-matching passes ------------------------------------------


def test_recover_with_no_jobs_is_200_empty(client, db_session):
    resp = client.post(RECOVER_URL, json={"older_than_seconds": 60})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0}
    # Exactly the two root members, in order, and one trailing newline.
    assert list(resp.json().keys()) == ["items", "count"]
    assert resp.text.endswith("}\n")
    assert _recovery_events(db_session) == []


def test_recover_ignores_other_states_and_young_running(client, db_session):
    _, _, bundle = _setup_bundle(client)
    pending = _create_job(client, bundle["id"], "req-pending")
    young = _create_job(client, bundle["id"], "req-young-running")
    succeeded = _create_job(client, bundle["id"], "req-succeeded")
    failed = _create_job(client, bundle["id"], "req-failed")
    # A running job that has not reached the cutoff stays untouched.
    _force_state(db_session, young["id"], "running", utc_now())
    _force_state(
        db_session,
        succeeded["id"],
        "succeeded",
        parse_rfc3339_utc(STALLED_AT),
    )
    _force_state(
        db_session, failed["id"], "failed", parse_rfc3339_utc(STALLED_AT)
    )

    resp = client.post(RECOVER_URL, json={"older_than_seconds": 3600})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0}

    for job, expected in (
        (pending, "pending"),
        (young, "running"),
        (succeeded, "succeeded"),
        (failed, "failed"),
    ):
        view = client.get(
            f"/v1/evidence-bundle-export-jobs/{job['id']}"
        ).json()
        assert view["status"] == expected
        assert view["error"] is None
    assert _recovery_events(db_session) == []


# --- successful recovery -----------------------------------------------------


def test_recover_settles_stalled_running_job(client, db_session):
    _, _, bundle = _setup_bundle(client)
    job = _create_job(client, bundle["id"], "req-stalled")
    _force_stalled(db_session, job["id"])

    before = client.get(f"/v1/evidence-bundle-export-jobs/{job['id']}").json()
    assert before["status"] == "running"

    resp = client.post(RECOVER_URL, json={"older_than_seconds": 60})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 1
    assert len(body["items"]) == 1

    item = body["items"][0]
    # Exactly the existing single-job public view members.
    assert set(item.keys()) == {
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
    # Identity, association, and execution start are preserved verbatim.
    assert item["id"] == job["id"]
    assert item["evidence_bundle_id"] == bundle["id"]
    assert item["request_id"] == "req-stalled"
    assert item["created_at"] == before["created_at"]
    assert item["started_at"] == before["started_at"]
    # Settled as failed by the recovery: null result, stable stalled error,
    # and a UTC finished_at at or after the (old) started_at.
    assert item["status"] == "failed"
    assert item["result"] is None
    assert item["error"] == "evidence_bundle_export_stalled"
    assert item["finished_at"] is not None
    assert item["finished_at"] > item["started_at"]

    # The detail route reflects the same settled public view.
    view = client.get(f"/v1/evidence-bundle-export-jobs/{job['id']}").json()
    assert view == item

    # Exactly one recovery audit event, appended after the creation event.
    assert _audit_types(db_session, job["id"]) == [
        "evidence_bundle_export_job.created",
        "evidence_bundle_export_job.stalled_recovery",
    ]


def test_recover_processes_stalled_jobs_in_started_order(client, db_session):
    _, _, bundle = _setup_bundle(client)
    # Created newest-started first so creation order and started order
    # disagree; processing must follow started_at (seq tiebreaker).
    newest = _create_job(client, bundle["id"], "req-order-newest")
    oldest = _create_job(client, bundle["id"], "req-order-oldest")
    middle = _create_job(client, bundle["id"], "req-order-middle")
    _force_stalled(db_session, newest["id"], "2026-01-01T00:02:00Z")
    _force_stalled(db_session, oldest["id"], "2026-01-01T00:00:00Z")
    _force_stalled(db_session, middle["id"], "2026-01-01T00:01:00Z")

    resp = client.post(RECOVER_URL, json={"older_than_seconds": 60})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 3
    assert [item["id"] for item in body["items"]] == [
        oldest["id"],
        middle["id"],
        newest["id"],
    ]
    assert all(item["status"] == "failed" for item in body["items"])
    assert all(
        item["error"] == "evidence_bundle_export_stalled"
        for item in body["items"]
    )
    # One recovery instant is shared by the whole pass.
    assert len({item["finished_at"] for item in body["items"]}) == 1

    # One recovery audit event per job, appended in processing order.
    events = _recovery_events(db_session)
    assert [e.resource_id for e in events] == [
        oldest["id"],
        middle["id"],
        newest["id"],
    ]


def test_recover_recovers_only_jobs_older_than_cutoff(client, db_session):
    _, _, bundle = _setup_bundle(client)
    stalled = _create_job(client, bundle["id"], "req-cutoff-stalled")
    young = _create_job(client, bundle["id"], "req-cutoff-young")
    _force_stalled(db_session, stalled["id"])
    _force_state(db_session, young["id"], "running", utc_now())

    resp = client.post(RECOVER_URL, json={"older_than_seconds": 86400})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 1
    assert [item["id"] for item in body["items"]] == [stalled["id"]]

    young_view = client.get(
        f"/v1/evidence-bundle-export-jobs/{young['id']}"
    ).json()
    assert young_view["status"] == "running"
    assert young_view["finished_at"] is None
    assert young_view["error"] is None


def test_recover_cutoff_is_strict(client, db_session, monkeypatch):
    """started_at exactly at the cutoff is not recovered (strictly earlier)."""
    fixed_now = parse_rfc3339_utc("2026-06-01T12:00:00Z")
    monkeypatch.setattr(service, "utc_now", lambda: fixed_now)
    _, _, bundle = _setup_bundle(client)
    at_cutoff = _create_job(client, bundle["id"], "req-strict-at")
    just_before = _create_job(client, bundle["id"], "req-strict-before")
    _force_state(
        db_session,
        at_cutoff["id"],
        "running",
        fixed_now - timedelta(seconds=60),
    )
    _force_state(
        db_session,
        just_before["id"],
        "running",
        fixed_now - timedelta(seconds=61),
    )

    recovered = service.recover_stalled_evidence_bundle_export_jobs(
        db_session, 60
    )
    assert [job.id for job in recovered] == [just_before["id"]]

    at_view = client.get(
        f"/v1/evidence-bundle-export-jobs/{at_cutoff['id']}"
    ).json()
    assert at_view["status"] == "running"
    assert at_view["finished_at"] is None
    before_view = client.get(
        f"/v1/evidence-bundle-export-jobs/{just_before['id']}"
    ).json()
    assert before_view["status"] == "failed"
    assert before_view["error"] == "evidence_bundle_export_stalled"


def test_recover_accepts_boundary_older_than_seconds(client):
    for value in (1, 86400):
        resp = client.post(RECOVER_URL, json={"older_than_seconds": value})
        assert resp.status_code == 200, (value, resp.text)
        assert resp.json() == {"items": [], "count": 0}


def test_recover_state_survives_restart(tmp_db_url):
    app1 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app1) as client:
        _, _, bundle = _setup_bundle(client)
        job = _create_job(client, bundle["id"], "req-durable")
        session = app1.state.session_factory()
        try:
            _force_stalled(session, job["id"])
        finally:
            session.close()
        resp = client.post(RECOVER_URL, json={"older_than_seconds": 60})
        assert resp.status_code == 200, resp.text
        assert resp.json()["count"] == 1
    app1.state.engine.dispose()

    app2 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app2) as client:
        view = client.get(
            f"/v1/evidence-bundle-export-jobs/{job['id']}"
        ).json()
        assert view["status"] == "failed"
        assert view["error"] == "evidence_bundle_export_stalled"
        assert view["result"] is None
        assert view["started_at"] is not None
        assert view["finished_at"] is not None
    session = app2.state.session_factory()
    try:
        assert _audit_types(session, job["id"]) == [
            "evidence_bundle_export_job.created",
            "evidence_bundle_export_job.stalled_recovery",
        ]
    finally:
        session.close()
        app2.state.engine.dispose()


# --- validation failures -----------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        b"",  # empty body
        b"   ",  # whitespace only
        b"{",  # malformed JSON
        b"\xff\xfe{}",  # invalid UTF-8
        b"[1]",  # non-object: array
        b'"x"',  # non-object: string
        b"60",  # non-object: number
        b"null",  # non-object: null
        b"true",  # non-object: boolean
    ],
)
def test_recover_malformed_bodies_are_422(client, db_session, body):
    _, _, bundle = _setup_bundle(client)
    job = _create_job(client, bundle["id"], "req-guarded-body")
    _force_stalled(db_session, job["id"])

    resp = client.post(
        RECOVER_URL, content=body, headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"

    # The stalled job is untouched and no recovery audit event exists.
    view = client.get(f"/v1/evidence-bundle-export-jobs/{job['id']}").json()
    assert view["status"] == "running"
    assert _recovery_events(db_session) == []


@pytest.mark.parametrize(
    "payload",
    [
        {},  # missing older_than_seconds
        {"older_than_seconds": 60, "extra": 1},  # undeclared field
        {"older_than_seconds": 0},  # below the minimum
        {"older_than_seconds": -1},
        {"older_than_seconds": 86401},  # above the maximum
        {"older_than_seconds": 1.5},  # not an integer
        {"older_than_seconds": 60.0},  # float spelling is not an integer
        {"older_than_seconds": "60"},  # string is not an integer
        {"older_than_seconds": True},  # boolean is not an integer
        {"older_than_seconds": None},
    ],
)
def test_recover_invalid_fields_are_422(client, db_session, payload):
    _, _, bundle = _setup_bundle(client)
    job = _create_job(client, bundle["id"], "req-guarded-field")
    _force_stalled(db_session, job["id"])

    resp = client.post(RECOVER_URL, json=payload)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"

    view = client.get(f"/v1/evidence-bundle-export-jobs/{job['id']}").json()
    assert view["status"] == "running"
    assert _recovery_events(db_session) == []


def test_recover_duplicate_field_is_422(client, db_session):
    _, _, bundle = _setup_bundle(client)
    job = _create_job(client, bundle["id"], "req-guarded-dup")
    _force_stalled(db_session, job["id"])

    resp = client.post(
        RECOVER_URL,
        content=b'{"older_than_seconds": 60, "older_than_seconds": 60}',
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"
    assert _recovery_events(db_session) == []


@pytest.mark.parametrize(
    "query",
    ["?older_than_seconds=60", "?x=1", "?x=", "?x=1&x=2"],
)
def test_recover_any_query_param_is_422(client, db_session, query):
    _, _, bundle = _setup_bundle(client)
    job = _create_job(client, bundle["id"], "req-guarded-query")
    _force_stalled(db_session, job["id"])

    resp = client.post(RECOVER_URL + query, json={"older_than_seconds": 60})
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"

    view = client.get(f"/v1/evidence-bundle-export-jobs/{job['id']}").json()
    assert view["status"] == "running"
    assert _recovery_events(db_session) == []


# --- conflict ---------------------------------------------------------------


def test_recover_conflict_rolls_back_entire_pass(tmp_db_url):
    """Deterministic interleaving over a file-backed database.

    Recovery selects three stalled jobs; before its compare-and-set, a
    concurrent caller settles the middle one. The whole pass must roll
    back: the first job stays running, no job is settled by the loser, and
    no recovery audit event is written.
    """
    application = create_app(Settings(database_url=tmp_db_url))
    with TestClient(application) as client:
        _, _, bundle = _setup_bundle(client)
        first = _create_job(client, bundle["id"], "req-race-first")
        middle = _create_job(client, bundle["id"], "req-race-middle")
        last = _create_job(client, bundle["id"], "req-race-last")

    factory = application.state.session_factory
    session_a = factory()
    session_b = factory()
    try:
        for job in (first, middle, last):
            _force_state(
                session_a,
                job["id"],
                "running",
                parse_rfc3339_utc(STALLED_AT),
            )

        # A selects the stalled set (oldest-started first), then ends its
        # read transaction so B can commit on this file-backed database.
        selected = session_a.execute(
            select(EvidenceBundleExportJob)
            .where(
                EvidenceBundleExportJob.status == "running",
                EvidenceBundleExportJob.started_at < utc_now(),
            )
            .order_by(
                EvidenceBundleExportJob.started_at.asc(),
                EvidenceBundleExportJob.seq.asc(),
            )
        ).scalars().all()
        assert [job.id for job in selected] == [
            first["id"],
            middle["id"],
            last["id"],
        ]
        session_a.commit()

        # B wins the middle job first (a concurrent recovery or runner).
        session_b.execute(
            sa_update(EvidenceBundleExportJob)
            .where(EvidenceBundleExportJob.id == middle["id"])
            .values(
                status="failed",
                finished_at=utc_now(),
                result=None,
                error="evidence_bundle_export_stalled",
            )
        )
        session_b.commit()

        # A's pass now fails its compare-and-set on the middle job: the
        # whole recovery is a conflict naming that job.
        with pytest.raises(
            EvidenceBundleExportJobRecoveryConflictError
        ) as exc_info:
            service._settle_stalled_evidence_bundle_export_jobs(
                session_a, list(selected)
            )
        assert (
            exc_info.value.code
            == "evidence_bundle_export_job_recovery_conflict"
        )
        assert exc_info.value.details["job_id"] == middle["id"]
    finally:
        session_a.close()
        session_b.close()
        application.state.engine.dispose()

    # Reopen to assert the rolled-back final state.
    verifier = create_app(Settings(database_url=tmp_db_url))
    with TestClient(verifier) as client:
        # The first job was settled inside the loser's transaction and must
        # have been rolled back to its pre-request running state.
        first_view = client.get(
            f"/v1/evidence-bundle-export-jobs/{first['id']}"
        ).json()
        assert first_view["status"] == "running"
        assert first_view["finished_at"] is None
        assert first_view["error"] is None
        # The middle job carries exactly B's settlement.
        middle_view = client.get(
            f"/v1/evidence-bundle-export-jobs/{middle['id']}"
        ).json()
        assert middle_view["status"] == "failed"
        assert middle_view["error"] == "evidence_bundle_export_stalled"
        # The last job was never reached.
        last_view = client.get(
            f"/v1/evidence-bundle-export-jobs/{last['id']}"
        ).json()
        assert last_view["status"] == "running"
    session = verifier.state.session_factory()
    try:
        # The loser wrote no recovery audit event at all.
        assert _recovery_events(session) == []
        for job in (first, last):
            assert _audit_types(session, job["id"]) == [
                "evidence_bundle_export_job.created"
            ]
    finally:
        session.close()
        verifier.state.engine.dispose()


def test_recover_conflict_is_409_over_http(client, monkeypatch):
    _, _, bundle = _setup_bundle(client)
    job = _create_job(client, bundle["id"], "req-conflict")

    def _lose_race(session, older_than_seconds):
        raise EvidenceBundleExportJobRecoveryConflictError(job["id"])

    monkeypatch.setattr(
        service, "recover_stalled_evidence_bundle_export_jobs", _lose_race
    )
    resp = client.post(RECOVER_URL, json={"older_than_seconds": 60})
    assert resp.status_code == 409, resp.text
    error = resp.json()["error"]
    assert error["code"] == "evidence_bundle_export_job_recovery_conflict"
    assert error["details"]["job_id"] == job["id"]
    monkeypatch.undo()

    # The job itself was never touched by the rejected request.
    view = client.get(f"/v1/evidence-bundle-export-jobs/{job['id']}").json()
    assert view["status"] == "pending"


# --- method handling ---------------------------------------------------------


@pytest.mark.parametrize("method", ["get", "put", "patch", "delete"])
def test_non_post_methods_are_405_method_not_allowed(client, method):
    resp = getattr(client, method)(RECOVER_URL)
    assert resp.status_code == 405, (method, resp.text)
    assert resp.json()["error"]["code"] == "method_not_allowed"
