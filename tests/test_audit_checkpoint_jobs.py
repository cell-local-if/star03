"""Tests for asynchronous audit checkpoint export jobs.

Covers:

* ``POST /v1/audit-checkpoint-jobs`` — required non-empty ``request_id`` plus
  the existing audit checkpoint filters (422 ``validation_error`` otherwise),
  first creation ``201`` as ``pending`` with a stable ``acj_`` id and null
  ``started_at``/``finished_at``/``result``/``error``, idempotent retry of the
  same key and filters ``200`` (and no second audit row), and the same
  ``request_id`` reused for different filters ``409
  audit_checkpoint_request_conflict``.
* ``GET /v1/audit-checkpoint-jobs/{job_id}`` — the job view, or
  ``404 audit_checkpoint_job_not_found``; any query parameter is a 422.
* ``POST /v1/audit-checkpoint-jobs/{job_id}/run`` — atomic claim of a pending
  job (``200`` settling ``succeeded`` with UTC timestamps and a result equal
  to the existing checkpoint package); a non-pending job is ``409 conflict``;
  a failed run settles ``failed`` with null result and
  ``audit_checkpoint_export_failed``; an unknown id is ``404
  audit_checkpoint_job_not_found``. The lifecycle and run audit events commit
  in the same transaction as their state changes.
* ``POST /v1/audit-checkpoint-jobs/run-next`` — empty queue 404, oldest-first
  stable claiming across a restart, body/parameter validation, and single
  winner under concurrency.

All fixtures are deterministic and offline (in-memory SQLite, plus a
temporary-file SQLite database for the real concurrent-claim tests).
"""

from __future__ import annotations

import threading
from datetime import timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update as sa_update

from provenance import service
from provenance.api import _audit_checkpoint_package_payload
from provenance.app import create_app
from provenance.config import Settings
from provenance.errors import AuditCheckpointJobConflictError
from provenance.models import (
    EVENT_ACTOR_CREATED,
    EVENT_CONTENT_CREATED,
    AuditCheckpointJob,
    AuditEvent,
)
from provenance.time_utils import parse_rfc3339_utc
from tests.helpers import DIGEST_A, DIGEST_B, DIGEST_C, create_actor

COLLECTION_URL = "/v1/audit-checkpoint-jobs"
RUN_NEXT_URL = "/v1/audit-checkpoint-jobs/run-next"

_JOB_KEYS = {
    "id",
    "request_id",
    "event_type",
    "resource_id",
    "from",
    "to",
    "status",
    "created_at",
    "started_at",
    "finished_at",
    "result",
    "error",
}


def _create_content(client, digest, actor_id="org-1"):
    resp = client.post(
        "/v1/contents",
        json={
            "digest_algorithm": "sha256",
            "digest_hex": digest,
            "media_type": "image/png",
            "actor_id": actor_id,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_job(client, request_id, **filters):
    return client.post(
        COLLECTION_URL, json={"request_id": request_id, **filters}
    )


def _package(client, **params):
    return client.get("/v1/audit-events/checkpoint/package", params=params)


def _setup_events(client):
    """Interleaved actor/content creations -> a known audit sequence."""
    create_actor(client, actor_id="org-1")
    c1 = _create_content(client, DIGEST_A)
    create_actor(client, actor_id="org-2", name="Other", type="person")
    c2 = _create_content(client, DIGEST_B, actor_id="org-2")
    c3 = _create_content(client, DIGEST_C)
    return [c1, c2, c3]


# --- creation ---------------------------------------------------------------


def test_create_job_201_pending_with_null_fields(client):
    resp = _create_job(client, "req-1")
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == _JOB_KEYS
    assert body["id"].startswith("acj_")
    assert len(body["id"]) == len("acj_") + 64
    assert body["request_id"] == "req-1"
    assert body["event_type"] is None
    assert body["resource_id"] is None
    assert body["from"] is None
    assert body["to"] is None
    assert body["status"] == "pending"
    assert body["started_at"] is None
    assert body["finished_at"] is None
    assert body["result"] is None
    assert body["error"] is None
    assert body["created_at"].endswith("Z")


def test_create_job_with_filters_echoes_them(client):
    resp = _create_job(
        client,
        "req-f",
        event_type=EVENT_CONTENT_CREATED,
        resource_id="cnt_x",
        **{
            "from": "2026-01-01T00:00:00Z",
            "to": "2026-02-01T00:00:00Z",
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["event_type"] == EVENT_CONTENT_CREATED
    assert body["resource_id"] == "cnt_x"
    assert body["from"] == "2026-01-01T00:00:00Z"
    assert body["to"] == "2026-02-01T00:00:00Z"


def test_create_job_id_is_stable(client):
    first = _create_job(client, "stable").json()
    second = _create_job(client, "stable").json()
    assert first["id"] == second["id"]
    # Different request ids yield different stable ids.
    other = _create_job(client, "other").json()
    assert other["id"] != first["id"]


def test_create_job_writes_created_audit_event(client, db_session):
    body = _create_job(client, "req-audit").json()
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.resource_id == body["id"],
            AuditEvent.event_type == "audit_checkpoint_job.created",
        )
    ).scalars().all()
    assert len(events) == 1
    assert events[0].created_at.tzinfo is not None


def test_create_job_idempotent_same_key_returns_200(client, db_session):
    first = _create_job(client, "retry-key")
    assert first.status_code == 201, first.text
    original = first.json()

    second = _create_job(client, "retry-key")
    assert second.status_code == 200, second.text
    assert second.json() == original

    # The retry writes neither a second job row nor a second audit event, in
    # any lifecycle state.
    jobs = db_session.execute(
        select(AuditCheckpointJob).where(
            AuditCheckpointJob.request_id == "retry-key"
        )
    ).scalars().all()
    assert len(jobs) == 1
    events = db_session.execute(
        select(AuditEvent).where(AuditEvent.resource_id == original["id"])
    ).scalars().all()
    assert [e.event_type for e in events] == ["audit_checkpoint_job.created"]


def test_create_job_equivalent_utc_spellings_are_same_filter(client, db_session):
    first = _create_job(
        client, "time-key", **{"from": "2026-01-01T00:00:00Z"}
    )
    assert first.status_code == 201, first.text
    # The "+00:00" spelling names the same UTC instant and filter: it is an
    # idempotent retry of the same job, not a second job or a conflict.
    second = _create_job(
        client, "time-key", **{"from": "2026-01-01T00:00:00+00:00"}
    )
    assert second.status_code == 200, second.text
    assert second.json()["id"] == first.json()["id"]
    jobs = db_session.execute(
        select(AuditCheckpointJob).where(
            AuditCheckpointJob.request_id == "time-key"
        )
    ).scalars().all()
    assert len(jobs) == 1


@pytest.mark.parametrize(
    "first_filters,second_filters",
    [
        ({}, {"event_type": EVENT_CONTENT_CREATED}),
        ({"event_type": EVENT_ACTOR_CREATED}, {"event_type": EVENT_CONTENT_CREATED}),
        ({}, {"resource_id": "cnt_a"}),
        ({"resource_id": "cnt_a"}, {"resource_id": "cnt_b"}),
        # Case- and whitespace-sensitive exact match: a different spelling is
        # a different filter.
        ({"event_type": "actor.created"}, {"event_type": "ACTOR.CREATED"}),
        ({"resource_id": "org-1"}, {"resource_id": "org-1 "}),
    ],
)
def test_create_job_same_request_id_different_filter_is_409(
    client, db_session, first_filters, second_filters
):
    ok = _create_job(client, "shared-key", **first_filters)
    assert ok.status_code == 201, ok.text
    original_id = ok.json()["id"]

    conflict = _create_job(client, "shared-key", **second_filters)
    assert conflict.status_code == 409, conflict.text
    err = conflict.json()["error"]
    assert err["code"] == "audit_checkpoint_request_conflict"
    assert err["details"]["request_id"] == "shared-key"

    # The conflict writes no job and no audit event; the original job alone
    # remains bound to that request_id.
    jobs = db_session.execute(
        select(AuditCheckpointJob).where(
            AuditCheckpointJob.request_id == "shared-key"
        )
    ).scalars().all()
    assert len(jobs) == 1
    assert jobs[0].id == original_id


def test_create_job_different_time_bound_is_409(client):
    ok = _create_job(
        client, "time-conflict", **{"from": "2026-01-01T00:00:00Z"}
    )
    assert ok.status_code == 201
    conflict = _create_job(
        client, "time-conflict", **{"from": "2026-01-02T00:00:00Z"}
    )
    assert conflict.status_code == 409
    assert (
        conflict.json()["error"]["code"]
        == "audit_checkpoint_request_conflict"
    )


def test_distinct_request_ids_are_independent_jobs(client):
    one = _create_job(client, "req-one")
    two = _create_job(client, "req-two")
    assert one.status_code == 201
    assert two.status_code == 201
    assert one.json()["id"] != two.json()["id"]


def test_re_register_same_key_after_settlement_is_200_unchanged(client):
    created = _create_job(client, "req-settled").json()
    settled = client.post(
        f"{COLLECTION_URL}/{created['id']}/run"
    ).json()
    assert settled["status"] == "succeeded"

    retry = _create_job(client, "req-settled")
    assert retry.status_code == 200, retry.text
    assert retry.json() == settled


@pytest.mark.parametrize(
    "payload",
    [
        {},  # missing request_id
        {"request_id": ""},  # empty request_id
        {"request_id": "   "},  # whitespace request_id
        {"request_id": None},  # null request_id
        {"request_id": 7},  # wrong type
        {"request_id": "r", "event_type": ""},  # blank filter
        {"request_id": "r", "event_type": "  "},
        {"request_id": "r", "resource_id": ""},
        {"request_id": "r", "resource_id": "\t"},
        {"request_id": "r", "event_type": None, "resource_id": 9},
        {"request_id": "r", "event_type": ["actor.created"]},
        {"request_id": "r", "from": "2026-01-02"},  # malformed times
        {"request_id": "r", "from": "2026-01-02T12:30:00+01:00"},
        {"request_id": "r", "to": "2026-01-02t12:30:00z"},
        {"request_id": "r", "from": "not-a-time"},
        {"request_id": "r", "from": 1_700_000_000},
        {"request_id": "r", "to": True},
        {
            # from later than to
            "request_id": "r",
            "from": "2026-01-03T00:00:00Z",
            "to": "2026-01-02T00:00:00Z",
        },
        {  # undeclared field
            "request_id": "r",
            "unexpected": True,
        },
        {"request_id": "r", "limit": 1},
        {"request_id": "r", "cursor": "x"},
    ],
)
def test_create_job_validation_errors_422(client, payload):
    resp = client.post(COLLECTION_URL, json=payload)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def test_create_job_malformed_json_is_422(client):
    resp = client.post(
        COLLECTION_URL,
        content="{not json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def test_create_job_query_param_is_422_and_writes_nothing(client, db_session):
    resp = client.post(COLLECTION_URL + "?x=1", json={"request_id": "r"})
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"
    assert db_session.execute(select(AuditCheckpointJob)).scalars().all() == []


# --- read -------------------------------------------------------------------


def test_get_job_returns_job(client):
    created = _create_job(client, "req-get").json()
    resp = client.get(f"{COLLECTION_URL}/{created['id']}")
    assert resp.status_code == 200, resp.text
    assert resp.json() == created


def test_get_unknown_job_is_404(client):
    resp = client.get(f"{COLLECTION_URL}/acj_does_not_exist")
    assert resp.status_code == 404, resp.text
    err = resp.json()["error"]
    assert err["code"] == "audit_checkpoint_job_not_found"
    assert err["details"]["job_id"] == "acj_does_not_exist"


def test_get_job_with_query_param_is_422(client):
    created = _create_job(client, "req-get-qp").json()
    resp = client.get(f"{COLLECTION_URL}/{created['id']}?x=1")
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- run: success -----------------------------------------------------------


def test_run_pending_job_succeeds_with_checkpoint_package(client):
    contents = _setup_events(client)
    job = _create_job(client, "req-run").json()
    # The package is captured from the read state immediately before the run:
    # the run appends its ``audit_checkpoint_job.run`` audit event only after
    # the result snapshot is built, so the result equals this same-filter
    # package (covering the full matched sequence as of the run), which also
    # already contains the job's creation event.
    expected = _package(client).json()
    resp = client.post(f"{COLLECTION_URL}/{job['id']}/run")
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

    # The result is exactly the read-only checkpoint package for the same
    # filter at the run's read state.
    assert body["result"] == expected
    assert set(body["result"]) == {"checkpoint", "events"}
    assert set(body["result"]["checkpoint"]) == {
        "checkpoint_version",
        "digest_algorithm",
        "event_count",
        "events_digest_hex",
    }
    # The full hit sequence is covered (the five setup events plus the job's
    # own creation event), and each event exposes only the three public
    # fields -- never raw material.
    assert body["result"]["checkpoint"]["event_count"] == 6
    assert [e["resource_id"] for e in body["result"]["events"]] == [
        "org-1",
        contents[0]["id"],
        "org-2",
        contents[1]["id"],
        contents[2]["id"],
        job["id"],
    ]
    for event in body["result"]["events"]:
        assert set(event) == {"event_type", "resource_id", "created_at"}

    # The settled state is what a subsequent read observes.
    assert client.get(f"{COLLECTION_URL}/{job['id']}").json() == body


def test_run_result_under_filter_is_stable_after_run(client):
    # With a filter that never matches job lifecycle events, the settled
    # result equals the same-filter checkpoint package even when read again
    # after the run audit event exists.
    _setup_events(client)
    job = _create_job(
        client, "req-run-stable", event_type=EVENT_CONTENT_CREATED
    ).json()
    body = client.post(f"{COLLECTION_URL}/{job['id']}/run").json()
    later = _package(client, event_type=EVENT_CONTENT_CREATED).json()
    assert body["result"] == later


def test_run_filtered_job_covers_only_matching_events(client):
    _setup_events(client)
    job = _create_job(
        client, "req-run-filtered", event_type=EVENT_CONTENT_CREATED
    ).json()
    body = client.post(f"{COLLECTION_URL}/{job['id']}/run").json()
    expected = _package(client, event_type=EVENT_CONTENT_CREATED).json()
    assert body["result"] == expected
    assert body["result"]["checkpoint"]["event_count"] == 3
    assert all(
        e["event_type"] == EVENT_CONTENT_CREATED
        for e in body["result"]["events"]
    )


def test_run_empty_match_has_empty_events_and_digest(client):
    import hashlib

    # A filter matching nothing (job lifecycle events included) yields the
    # deterministic empty package: zero events and the digest of the empty
    # array.
    _setup_events(client)
    created = _create_job(
        client, "req-empty", event_type="no.such.event"
    ).json()
    body = client.post(f"{COLLECTION_URL}/{created['id']}/run").json()
    expected = _package(client, event_type="no.such.event").json()
    assert body["result"] == expected
    assert body["result"]["events"] == []
    assert body["result"]["checkpoint"]["event_count"] == 0
    assert (
        body["result"]["checkpoint"]["events_digest_hex"]
        == hashlib.sha256(b"[]").hexdigest()
    )


def test_run_writes_one_run_audit_event(client, db_session):
    job = _create_job(client, "req-audit-run").json()
    run = client.post(f"{COLLECTION_URL}/{job['id']}/run")
    assert run.status_code == 200, run.text

    events = db_session.execute(
        select(AuditEvent.event_type)
        .where(AuditEvent.resource_id == job["id"])
        .order_by(AuditEvent.seq)
    ).scalars().all()
    assert events == ["audit_checkpoint_job.created", "audit_checkpoint_job.run"]


# --- run: conflicts and missing ---------------------------------------------


def test_run_unknown_job_is_404(client):
    resp = client.post(f"{COLLECTION_URL}/acj_missing/run")
    assert resp.status_code == 404, resp.text
    err = resp.json()["error"]
    assert err["code"] == "audit_checkpoint_job_not_found"
    assert err["details"]["job_id"] == "acj_missing"


def test_run_twice_only_first_runs(client, db_session):
    job = _create_job(client, "req-rerun").json()

    first = client.post(f"{COLLECTION_URL}/{job['id']}/run")
    assert first.status_code == 200, first.text
    assert first.json()["status"] == "succeeded"

    second = client.post(f"{COLLECTION_URL}/{job['id']}/run")
    assert second.status_code == 409, second.text
    assert second.json()["error"]["code"] == "conflict"

    # The rejected repeat changes no state and writes no second run event.
    final = client.get(f"{COLLECTION_URL}/{job['id']}").json()
    assert final == first.json()
    run_events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.resource_id == job["id"],
            AuditEvent.event_type == "audit_checkpoint_job.run",
        )
    ).scalars().all()
    assert len(run_events) == 1


@pytest.mark.parametrize("state", ["running", "succeeded", "failed"])
def test_run_non_pending_job_is_409(client, db_session, state):
    job = _create_job(client, f"req-{state}").json()

    db_session.execute(
        sa_update(AuditCheckpointJob)
        .where(AuditCheckpointJob.id == job["id"])
        .values(status=state)
    )
    db_session.commit()

    resp = client.post(f"{COLLECTION_URL}/{job['id']}/run")
    assert resp.status_code == 409, resp.text
    assert resp.json()["error"]["code"] == "conflict"
    assert (
        client.get(f"{COLLECTION_URL}/{job['id']}").json()["status"] == state
    )


def test_run_settles_failed_when_build_raises(client, db_session, monkeypatch):
    job = _create_job(client, "req-fail").json()

    def _boom(session, event_type, resource_id, from_dt, to_dt):
        raise RuntimeError("simulated checkpoint export failure")

    monkeypatch.setattr(service, "list_audit_events", _boom)

    resp = client.post(f"{COLLECTION_URL}/{job['id']}/run")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "failed"
    assert body["started_at"] is not None
    assert body["finished_at"] is not None
    assert body["result"] is None
    assert body["error"] == "audit_checkpoint_export_failed"

    # The failure is durable; a failed job can never be (re)claimed, even
    # after the fault clears.
    monkeypatch.undo()
    assert (
        client.post(f"{COLLECTION_URL}/{job['id']}/run").status_code == 409
    )
    final = client.get(f"{COLLECTION_URL}/{job['id']}").json()
    assert final["status"] == "failed"
    assert final["error"] == "audit_checkpoint_export_failed"

    events = db_session.execute(
        select(AuditEvent.event_type)
        .where(AuditEvent.resource_id == job["id"])
        .order_by(AuditEvent.seq)
    ).scalars().all()
    assert events == ["audit_checkpoint_job.created", "audit_checkpoint_job.run"]


@pytest.mark.parametrize(
    "send",
    [
        lambda c, url: c.post(url, content=b"   "),
        lambda c, url: c.post(
            url,
            content=b"{not json",
            headers={"content-type": "application/json"},
        ),
        lambda c, url: c.post(url, json={}),
        lambda c, url: c.post(url, json={"unexpected": True}),
        lambda c, url: c.post(url + "?x=1"),
        lambda c, url: c.post(url + "?x="),
        lambda c, url: c.post(url + "?x=1&x=2"),
    ],
)
def test_run_rejects_body_and_params_422(client, db_session, send):
    job = _create_job(client, "req-run-guarded").json()
    resp = send(client, f"{COLLECTION_URL}/{job['id']}/run")
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"

    view = client.get(f"{COLLECTION_URL}/{job['id']}").json()
    assert view["status"] == "pending"
    assert view["started_at"] is None
    run_events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.resource_id == job["id"],
            AuditEvent.event_type == "audit_checkpoint_job.run",
        )
    ).scalars().all()
    assert run_events == []

    # The well-formed call still claims it exactly once.
    claimed = client.post(f"{COLLECTION_URL}/{job['id']}/run")
    assert claimed.status_code == 200


# --- method not allowed -----------------------------------------------------


@pytest.mark.parametrize("method", ("put", "patch", "delete"))
def test_non_post_methods_on_collection_are_405(client, method):
    resp = getattr(client, method)(COLLECTION_URL)
    assert resp.status_code == 405
    assert resp.json()["error"]["code"] == "method_not_allowed"


@pytest.mark.parametrize("method", ("put", "patch", "delete", "post"))
def test_other_methods_on_detail_are_405(client, method):
    job = _create_job(client, "req-405").json()
    resp = getattr(client, method)(f"{COLLECTION_URL}/{job['id']}")
    assert resp.status_code == 405
    assert resp.json()["error"]["code"] == "method_not_allowed"


def test_get_on_run_is_405(client):
    job = _create_job(client, "req-405-run").json()
    resp = client.get(f"{COLLECTION_URL}/{job['id']}/run")
    assert resp.status_code == 405
    assert resp.json()["error"]["code"] == "method_not_allowed"


# --- concurrency ------------------------------------------------------------


def test_concurrent_runs_claim_exactly_once(file_app):
    """Real concurrency over a file-backed SQLite database.

    Of many simultaneous run calls exactly one returns 200 (settling
    succeeded) and every other is a 409; the final state is succeeded with a
    single run audit event.
    """
    setup = TestClient(file_app)
    job_id = _create_job(setup, "req-concurrent").json()["id"]

    outcomes: list[tuple[int, str]] = []
    lock = threading.Lock()

    def _fire() -> None:
        worker = TestClient(file_app)
        resp = worker.post(f"{COLLECTION_URL}/{job_id}/run")
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
