"""Tests for asynchronous audit checkpoint export jobs.

Covers:

* ``POST /v1/audit-checkpoint-jobs`` — a required non-empty ``request_id``
  plus the four existing audit-checkpoint filters
  (``event_type``/``resource_id`` exact matches, strict UTC ``from``/``to``
  bounds); first creation ``201`` as ``pending`` with a stable ``acj_`` id
  and null ``started_at``/``finished_at``/``result``/``error``; an idempotent
  retry of the same request id and filter is ``200`` with the original job
  and no second audit row; the same request id with a different filter is
  ``409 audit_checkpoint_request_conflict`` with zero writes.
* ``GET /v1/audit-checkpoint-jobs/{job_id}`` — the job view, or
  ``404 audit_checkpoint_job_not_found``.
* ``POST /v1/audit-checkpoint-jobs/{job_id}/run`` — atomic claim of a
  pending job settling to ``succeeded`` with UTC timestamps and a result
  equal to the read-only checkpoint package for the same filter (full
  matched sequence, public checkpoint/events structure only), or ``failed``
  with null result and ``audit_checkpoint_export_failed``; a non-pending,
  concurrent, or repeated run is ``409 conflict`` with zero writes; an
  unknown id is a 404. The lifecycle and run audit events commit in the same
  transaction as their state changes.

All fixtures are deterministic and offline (in-memory SQLite, plus a
temporary-file SQLite database for the concurrent-claim test).
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update as sa_update

from provenance import service
from provenance.models import (
    EVENT_AUDIT_CHECKPOINT_JOB_CREATED,
    EVENT_AUDIT_CHECKPOINT_JOB_RUN,
    AuditCheckpointJob,
    AuditEvent,
)
from provenance.time_utils import parse_rfc3339_utc
from tests.helpers import DIGEST_A, DIGEST_B, DIGEST_C, create_actor

JOBS_URL = "/v1/audit-checkpoint-jobs"

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


def _setup_events(client):
    """Interleaved actor/content creations -> a known audit sequence."""
    create_actor(client, actor_id="org-1")
    c1 = _create_content(client, DIGEST_A)
    create_actor(client, actor_id="org-2", name="Other", type="person")
    c2 = _create_content(client, DIGEST_B, actor_id="org-2")
    c3 = _create_content(client, DIGEST_C)
    return [c1, c2, c3]


def _create_job(client, request_id, **filters):
    return client.post(
        JOBS_URL, json={"request_id": request_id, **filters}
    )


def _package(client, **params):
    return client.get(
        "/v1/audit-events/checkpoint/package", params=params
    ).json()


# --- creation ---------------------------------------------------------------


def test_create_job_201_pending_with_null_fields(client):
    _setup_events(client)
    resp = _create_job(client, "req-1", event_type="actor.created")
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == _JOB_KEYS
    assert body["id"].startswith("acj_")
    assert len(body["id"]) == len("acj_") + 64
    assert body["request_id"] == "req-1"
    assert body["event_type"] == "actor.created"
    assert body["resource_id"] is None
    assert body["from"] is None
    assert body["to"] is None
    assert body["status"] == "pending"
    assert body["started_at"] is None
    assert body["finished_at"] is None
    assert body["result"] is None
    assert body["error"] is None
    assert body["created_at"].endswith("Z")


def test_create_job_id_is_stable_for_same_identity(client):
    _setup_events(client)
    first = _create_job(client, "stable", event_type="actor.created")
    second = _create_job(client, "stable", event_type="actor.created")
    assert first.json()["id"] == second.json()["id"]


def test_create_job_success_body_ends_with_single_newline(client):
    resp = _create_job(client, "req-nl")
    assert resp.status_code == 201
    assert resp.content.endswith(b"\n")
    assert not resp.content.endswith(b"\n\n")


def test_create_job_writes_created_audit_event(client, db_session):
    body = _create_job(client, "req-audit").json()
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.resource_id == body["id"],
            AuditEvent.event_type == EVENT_AUDIT_CHECKPOINT_JOB_CREATED,
        )
    ).scalars().all()
    assert len(events) == 1
    assert events[0].created_at.tzinfo is not None


def test_create_job_idempotent_same_filter_returns_200(client, db_session):
    filters = {
        "event_type": "actor.created",
        "resource_id": "org-1",
        "from": "2026-01-01T00:00:00Z",
        "to": "2027-01-01T00:00:00Z",
    }
    first = _create_job(client, "retry-key", **filters)
    assert first.status_code == 201, first.text
    original = first.json()

    second = _create_job(client, "retry-key", **filters)
    assert second.status_code == 200, second.text
    assert second.json() == original

    jobs = db_session.execute(
        select(AuditCheckpointJob).where(
            AuditCheckpointJob.request_id == "retry-key"
        )
    ).scalars().all()
    assert len(jobs) == 1
    events = db_session.execute(
        select(AuditEvent).where(AuditEvent.resource_id == original["id"])
    ).scalars().all()
    assert [e.event_type for e in events] == [
        EVENT_AUDIT_CHECKPOINT_JOB_CREATED
    ]


def test_create_job_equivalent_utc_spelling_is_same_filter(client):
    # "Z" and "+00:00" name the same UTC instant, so the effective filter is
    # identical and the retry returns the original job.
    first = _create_job(client, "utc-spelling", **{"from": "2026-01-01T00:00:00Z"})
    assert first.status_code == 201
    retry = _create_job(
        client, "utc-spelling", **{"from": "2026-01-01T00:00:00+00:00"}
    )
    assert retry.status_code == 200
    assert retry.json()["id"] == first.json()["id"]


@pytest.mark.parametrize(
    "first,second",
    [
        ({"event_type": "actor.created"}, {"event_type": "content.created"}),
        ({"resource_id": "a"}, {"resource_id": "b"}),
        ({}, {"event_type": "actor.created"}),
        ({"event_type": "actor.created"}, {}),
        ({"from": "2026-01-01T00:00:00Z"}, {"from": "2026-01-02T00:00:00Z"}),
        ({"to": "2027-01-01T00:00:00Z"}, {"to": "2028-01-01T00:00:00Z"}),
        ({"event_type": "actor.created"}, {"resource_id": "actor.created"}),
    ],
)
def test_create_job_same_request_id_different_filter_is_409(
    client, db_session, first, second
):
    ok = _create_job(client, "shared-key", **first)
    assert ok.status_code == 201, ok.text

    conflict = _create_job(client, "shared-key", **second)
    assert conflict.status_code == 409, conflict.text
    err = conflict.json()["error"]
    assert err["code"] == "audit_checkpoint_request_conflict"
    assert err["details"]["request_id"] == "shared-key"

    # Zero writes: exactly one job remains bound to the request id and only
    # the original creation audit event exists.
    jobs = db_session.execute(
        select(AuditCheckpointJob).where(
            AuditCheckpointJob.request_id == "shared-key"
        )
    ).scalars().all()
    assert len(jobs) == 1
    assert jobs[0].event_type == first.get("event_type")
    assert jobs[0].resource_id == first.get("resource_id")
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.resource_id == jobs[0].id
        )
    ).scalars().all()
    assert [e.event_type for e in events] == [
        EVENT_AUDIT_CHECKPOINT_JOB_CREATED
    ]


def test_distinct_request_ids_are_independent_jobs(client):
    one = _create_job(client, "req-one", event_type="actor.created")
    two = _create_job(client, "req-two", event_type="actor.created")
    assert one.status_code == 201
    assert two.status_code == 201
    assert one.json()["id"] != two.json()["id"]


def test_re_register_same_key_after_settlement_is_200_unchanged(client):
    created = _create_job(client, "req-settled").json()
    settled = client.post(f"{JOBS_URL}/{created['id']}/run").json()
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
        {"request_id": 5},  # wrong type
        {"request_id": "r", "event_type": ""},  # blank filter
        {"request_id": "r", "event_type": "  "},  # whitespace filter
        {"request_id": "r", "resource_id": ""},
        {"request_id": "r", "event_type": 1},  # wrong filter type
        {"request_id": "r", "resource_id": []},
        {"request_id": "r", "from": {}},
        {"request_id": "r", "from": "2026-01-02"},  # not RFC 3339
        {"request_id": "r", "from": "2026-01-02T12:30:00"},  # naive
        {"request_id": "r", "from": "2026-01-02T12:30:00+01:00"},  # non-UTC
        {"request_id": "r", "to": "not-a-time"},
        {  # undeclared field
            "request_id": "r",
            "unexpected": True,
        },
    ],
)
def test_create_job_validation_errors_422(client, payload):
    resp = client.post(JOBS_URL, json=payload)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def test_create_job_from_later_than_to_is_422(client):
    resp = _create_job(
        client,
        "req-range",
        **{"from": "2026-01-03T00:00:00Z", "to": "2026-01-01T00:00:00Z"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_create_job_malformed_json_is_422(client):
    resp = client.post(
        JOBS_URL,
        content="{not json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def test_create_job_with_query_parameter_is_422(client, db_session):
    resp = client.post(JOBS_URL + "?x=1", json={"request_id": "r"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert db_session.execute(select(AuditCheckpointJob)).scalars().all() == []


# --- read -------------------------------------------------------------------


def test_get_job_returns_job(client):
    created = _create_job(client, "req-get").json()
    resp = client.get(f"{JOBS_URL}/{created['id']}")
    assert resp.status_code == 200, resp.text
    assert resp.json() == created
    assert resp.content.endswith(b"\n")


def test_get_unknown_job_is_404(client):
    resp = client.get(f"{JOBS_URL}/acj_does_not_exist")
    assert resp.status_code == 404, resp.text
    err = resp.json()["error"]
    assert err["code"] == "audit_checkpoint_job_not_found"
    assert err["details"]["job_id"] == "acj_does_not_exist"


def test_get_job_with_extra_parameter_is_422_zero_writes(client, db_session):
    created = _create_job(client, "req-param").json()
    before = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )
    resp = client.get(f"{JOBS_URL}/{created['id']}?x=1")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    db_session.expire_all()
    assert len(db_session.execute(select(AuditEvent)).scalars().all()) == before


# --- run: success -----------------------------------------------------------


def test_run_pending_job_succeeds_with_checkpoint_package(client):
    contents = _setup_events(client)
    resource_id = contents[0]["id"]
    job = _create_job(client, "req-run", resource_id=resource_id).json()

    resp = client.post(f"{JOBS_URL}/{job['id']}/run")
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
    # filter: checkpoint plus the full matched event sequence in order.
    expected = _package(client, resource_id=resource_id)
    assert body["result"] == expected
    assert set(body["result"]) == {"checkpoint", "events"}
    assert [e["resource_id"] for e in body["result"]["events"]] == [resource_id]
    assert (
        body["result"]["checkpoint"]["event_count"]
        == len(body["result"]["events"])
    )
    assert body["result"]["checkpoint"]["events_digest_hex"] == expected[
        "checkpoint"
    ]["events_digest_hex"]

    # Only public event material is carried: each event is the three-field
    # public view, nothing else.
    for event in body["result"]["events"]:
        assert set(event) == {"event_type", "resource_id", "created_at"}

    # The settled state is what a subsequent read observes.
    assert client.get(f"{JOBS_URL}/{job['id']}").json() == body


def test_run_unfiltered_result_covers_full_audit_sequence(client):
    _setup_events(client)
    job = _create_job(client, "req-full").json()

    body = client.post(f"{JOBS_URL}/{job['id']}/run").json()
    # The result equals the checkpoint package for the same filter. The run
    # audit event is written (and flushed) before the package is read in the
    # same transaction, so the post-run package covers the five setup events
    # plus this job's created and run events, in stable creation order.
    expected = _package(client)
    assert body["result"] == expected
    event_types = [e["event_type"] for e in body["result"]["events"]]
    assert event_types[:5] == [
        "actor.created",
        "content.created",
        "actor.created",
        "content.created",
        "content.created",
    ]
    assert event_types[5:] == [
        EVENT_AUDIT_CHECKPOINT_JOB_CREATED,
        EVENT_AUDIT_CHECKPOINT_JOB_RUN,
    ]
    assert body["result"]["checkpoint"]["event_count"] == 7


def test_run_empty_match_has_empty_events_and_its_digest(client):
    _setup_events(client)
    job = _create_job(
        client, "req-empty", event_type="no.such.event"
    ).json()

    body = client.post(f"{JOBS_URL}/{job['id']}/run").json()
    expected = _package(client, event_type="no.such.event")
    assert body["result"] == expected
    assert body["result"]["events"] == []
    assert body["result"]["checkpoint"]["event_count"] == 0


def test_run_result_respects_strict_utc_bounds(client, db_session):
    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 2, 12, 30, tzinfo=timezone.utc)
    db_session.add(AuditEvent(event_type="actor.created", resource_id="r1", created_at=t1))
    db_session.add(AuditEvent(event_type="actor.created", resource_id="r2", created_at=t2))
    db_session.commit()

    job = _create_job(
        client,
        "req-bounds",
        **{"from": "2026-01-02T12:30:00Z", "to": "2026-01-02T12:30:00Z"},
    ).json()
    body = client.post(f"{JOBS_URL}/{job['id']}/run").json()
    expected = _package(
        client,
        **{"from": "2026-01-02T12:30:00Z", "to": "2026-01-02T12:30:00Z"},
    )
    assert body["result"] == expected
    assert [e["resource_id"] for e in body["result"]["events"]] == ["r2"]


def test_run_writes_one_run_audit_event(client, db_session):
    job = _create_job(client, "req-audit-run").json()
    run = client.post(f"{JOBS_URL}/{job['id']}/run")
    assert run.status_code == 200, run.text

    events = db_session.execute(
        select(AuditEvent.event_type)
        .where(AuditEvent.resource_id == job["id"])
        .order_by(AuditEvent.seq)
    ).scalars().all()
    assert events == [
        EVENT_AUDIT_CHECKPOINT_JOB_CREATED,
        EVENT_AUDIT_CHECKPOINT_JOB_RUN,
    ]


# --- run: conflicts, failure, missing ---------------------------------------


def test_run_unknown_job_is_404(client):
    resp = client.post(f"{JOBS_URL}/acj_missing/run")
    assert resp.status_code == 404, resp.text
    err = resp.json()["error"]
    assert err["code"] == "audit_checkpoint_job_not_found"
    assert err["details"]["job_id"] == "acj_missing"


def test_run_twice_only_first_runs(client, db_session):
    job = _create_job(client, "req-rerun").json()
    first = client.post(f"{JOBS_URL}/{job['id']}/run")
    assert first.status_code == 200
    assert first.json()["status"] == "succeeded"

    second = client.post(f"{JOBS_URL}/{job['id']}/run")
    assert second.status_code == 409, second.text
    assert second.json()["error"]["code"] == "conflict"

    final = client.get(f"{JOBS_URL}/{job['id']}").json()
    assert final == first.json()
    run_events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.resource_id == job["id"],
            AuditEvent.event_type == EVENT_AUDIT_CHECKPOINT_JOB_RUN,
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

    resp = client.post(f"{JOBS_URL}/{job['id']}/run")
    assert resp.status_code == 409, resp.text
    err = resp.json()["error"]
    assert err["code"] == "conflict"
    assert err["details"]["job_id"] == job["id"]
    assert err["details"]["status"] == state

    assert client.get(f"{JOBS_URL}/{job['id']}").json()["status"] == state
    run_events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.resource_id == job["id"],
            AuditEvent.event_type == EVENT_AUDIT_CHECKPOINT_JOB_RUN,
        )
    ).scalars().all()
    assert run_events == []


def test_run_settles_failed_when_export_raises(client, db_session, monkeypatch):
    job = _create_job(client, "req-fail").json()

    def _boom(session, job):
        raise RuntimeError("simulated checkpoint failure")

    monkeypatch.setattr(service, "list_audit_events", _boom)

    resp = client.post(f"{JOBS_URL}/{job['id']}/run")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "failed"
    assert body["started_at"] is not None
    assert body["finished_at"] is not None
    started = parse_rfc3339_utc(body["started_at"])
    finished = parse_rfc3339_utc(body["finished_at"])
    assert started.tzinfo == timezone.utc
    assert finished.tzinfo == timezone.utc
    assert body["result"] is None
    assert body["error"] == "audit_checkpoint_export_failed"

    monkeypatch.undo()
    assert client.post(f"{JOBS_URL}/{job['id']}/run").status_code == 409
    final = client.get(f"{JOBS_URL}/{job['id']}").json()
    assert final["status"] == "failed"
    assert final["error"] == "audit_checkpoint_export_failed"

    events = db_session.execute(
        select(AuditEvent.event_type)
        .where(AuditEvent.resource_id == job["id"])
        .order_by(AuditEvent.seq)
    ).scalars().all()
    assert events == [
        EVENT_AUDIT_CHECKPOINT_JOB_CREATED,
        EVENT_AUDIT_CHECKPOINT_JOB_RUN,
    ]


def test_failed_job_settlement_timestamps_are_utc_after_restart(tmp_db_url):
    # A failed job keeps both started_at and finished_at as timezone-aware
    # UTC instants, including when the settled row is re-read by a fresh
    # process/app over the same database file.
    from provenance.app import create_app
    from provenance.config import Settings

    app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app) as client:
        job = _create_job(client, "req-fail-utc").json()

        def _boom(session, job):
            raise RuntimeError("simulated checkpoint failure")

        import provenance.service as svc

        original = svc.list_audit_events
        svc.list_audit_events = _boom
        try:
            settled = client.post(f"{JOBS_URL}/{job['id']}/run")
        finally:
            svc.list_audit_events = original
        assert settled.status_code == 200
        assert settled.json()["status"] == "failed"

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        view = client.get(f"{JOBS_URL}/{job['id']}").json()
        started = parse_rfc3339_utc(view["started_at"])
        finished = parse_rfc3339_utc(view["finished_at"])
        assert started is not None and finished is not None
        assert started.tzinfo == timezone.utc
        assert finished.tzinfo == timezone.utc
        assert started <= finished


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
def test_run_validation_errors_422_write_nothing(client, db_session, send):
    job = _create_job(client, "req-guarded").json()
    resp = send(client, f"{JOBS_URL}/{job['id']}/run")
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"

    view = client.get(f"{JOBS_URL}/{job['id']}").json()
    assert view["status"] == "pending"
    assert view["started_at"] is None
    run_events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.resource_id == job["id"],
            AuditEvent.event_type == EVENT_AUDIT_CHECKPOINT_JOB_RUN,
        )
    ).scalars().all()
    assert run_events == []

    claimed = client.post(f"{JOBS_URL}/{job['id']}/run")
    assert claimed.status_code == 200
    assert claimed.json()["id"] == job["id"]


def test_concurrent_runs_claim_exactly_once(file_app):
    """Real concurrency: of many runs exactly one settles, the rest 409."""
    setup = TestClient(file_app)
    job_id = setup.post(JOBS_URL, json={"request_id": "req-concurrent"}).json()[
        "id"
    ]

    outcomes: list[tuple[int, str]] = []
    lock = threading.Lock()

    def _fire() -> None:
        worker = TestClient(file_app)
        resp = worker.post(f"{JOBS_URL}/{job_id}/run")
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
    final = reader.get(f"{JOBS_URL}/{job_id}").json()
    assert final["status"] == "succeeded"
    session = file_app.state.session_factory()
    try:
        run_events = session.execute(
            select(AuditEvent).where(
                AuditEvent.resource_id == job_id,
                AuditEvent.event_type == EVENT_AUDIT_CHECKPOINT_JOB_RUN,
            )
        ).scalars().all()
        assert len(run_events) == 1
    finally:
        session.close()


# --- method handling --------------------------------------------------------


@pytest.mark.parametrize("method", ["put", "patch", "delete"])
def test_unsupported_methods_on_collection_are_405(client, method):
    resp = getattr(client, method)(JOBS_URL)
    assert resp.status_code == 405, resp.text
    assert resp.json()["error"]["code"] == "method_not_allowed"


@pytest.mark.parametrize("method", ["put", "patch", "delete"])
def test_unsupported_methods_on_detail_are_405(client, method):
    created = _create_job(client, "req-method").json()
    resp = getattr(client, method)(f"{JOBS_URL}/{created['id']}")
    assert resp.status_code == 405, resp.text
    assert resp.json()["error"]["code"] == "method_not_allowed"
