"""Tests for asynchronous content export jobs.

Covers POST /v1/content-export-jobs, GET /v1/content-export-jobs/{job_id},
and POST /v1/content-export-jobs/{job_id}/run:

* ``content_id``/``request_id`` are required and non-empty; malformed input
  is 422 validation_error and an unknown content is 404 content_not_found.
* Idempotency: the same ``(content_id, request_id)`` pair is 200 returning
  the original job; reusing a ``request_id`` for different content is 409
  content_export_request_conflict.
* First creation is 201 with exactly id/content_id/request_id/status/
  created_at/started_at/finished_at/result/error, initially pending with the
  three times/result/error null.
* An unknown job id is 404 content_export_job_not_found on read and run.
* Run atomically claims a pending job: the winning call is 200 and leaves the
  job succeeded with UTC started_at/finished_at and result equal to the
  existing read-only export (claims/evidence_bundles); a concurrent or
  repeat run of a non-pending job is 409; an export error leaves the job
  failed with finished_at, null result, and error content_export_failed.
* Job creation and run each write one audit event in the same transaction;
  idempotent and conflicting writes add none.

All fixtures are deterministic and offline.
"""

from __future__ import annotations

import hashlib
import threading
import time
from datetime import datetime

from sqlalchemy import func, select

from provenance import service
from provenance.models import (
    AuditEvent,
    ContentExportJob,
    EVENT_CONTENT_EXPORT_JOB_CREATED,
    EVENT_CONTENT_EXPORT_JOB_RUN,
)
from tests.helpers import create_actor


def _digest(name: str) -> str:
    return hashlib.sha256(f"export-job-{name}".encode()).hexdigest()


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


def _create_claim(client, content_id, marker, actor_id="org-1",
                  claim_type="authorship"):
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
    resp = client.post(
        "/v1/content-export-jobs",
        json={"content_id": content_id, "request_id": request_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _is_utc(value: str) -> bool:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.utcoffset().total_seconds() == 0


# --- creation ---------------------------------------------------------------


def test_create_job_is_201_pending_with_exact_fields_and_nulls(client):
    create_actor(client)
    content = _create_content(client, "main")

    resp = client.post(
        "/v1/content-export-jobs",
        json={"content_id": content["id"], "request_id": "req-1"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == {
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
    assert body["id"].startswith("cej_")
    assert body["content_id"] == content["id"]
    assert body["request_id"] == "req-1"
    assert body["status"] == "pending"
    assert body["started_at"] is None
    assert body["finished_at"] is None
    assert body["result"] is None
    assert body["error"] is None
    assert _is_utc(body["created_at"])


def test_create_job_does_not_run_it(client):
    create_actor(client)
    content = _create_content(client, "main")
    job = _create_job(client, content["id"], "req-1")

    read = client.get(f"/v1/content-export-jobs/{job['id']}")
    assert read.status_code == 200, read.text
    assert read.json()["status"] == "pending"
    assert read.json()["started_at"] is None


def test_create_job_requires_nonempty_fields(client):
    create_actor(client)
    content = _create_content(client, "main")

    for payload in (
        {"request_id": "r"},  # missing content_id
        {"content_id": content["id"]},  # missing request_id
        {"content_id": "", "request_id": "r"},
        {"content_id": "   ", "request_id": "r"},
        {"content_id": content["id"], "request_id": ""},
        {"content_id": content["id"], "request_id": "   "},
        {"content_id": content["id"], "request_id": 42},
        {"content_id": ["x"], "request_id": "r"},
        {},
    ):
        resp = client.post("/v1/content-export-jobs", json=payload)
        assert resp.status_code == 422, payload
        assert resp.json()["error"]["code"] == "validation_error"


def test_create_job_rejects_undeclared_fields(client):
    create_actor(client)
    content = _create_content(client, "main")
    resp = client.post(
        "/v1/content-export-jobs",
        json={"content_id": content["id"], "request_id": "r", "extra": 1},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_create_job_unknown_content_is_404(client):
    resp = client.post(
        "/v1/content-export-jobs",
        json={"content_id": "cnt_doesnotexist", "request_id": "r"},
    )
    assert resp.status_code == 404, resp.text
    err = resp.json()["error"]
    assert err["code"] == "content_not_found"
    assert err["details"]["content_id"] == "cnt_doesnotexist"


def test_create_job_rejects_any_query_parameter(client):
    create_actor(client)
    content = _create_content(client, "main")
    resp = client.post(
        f"/v1/content-export-jobs?x=1",
        json={"content_id": content["id"], "request_id": "r"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- idempotency and conflict ----------------------------------------------


def test_same_pair_is_idempotent_200(client, db_session):
    create_actor(client)
    content = _create_content(client, "main")
    first = _create_job(client, content["id"], "req-1")

    repeat = client.post(
        "/v1/content-export-jobs",
        json={"content_id": content["id"], "request_id": "req-1"},
    )
    assert repeat.status_code == 200, repeat.text
    assert repeat.json()["id"] == first["id"]
    assert repeat.json()["status"] == "pending"

    # Only the first creation wrote a job row.
    count = db_session.execute(
        select(func.count()).select_from(ContentExportJob)
    ).scalar_one()
    assert count == 1


def test_same_request_id_different_content_is_409(client, db_session):
    create_actor(client)
    first = _create_content(client, "first")
    second = _create_content(client, "second")
    _create_job(client, first["id"], "shared-request")

    resp = client.post(
        "/v1/content-export-jobs",
        json={"content_id": second["id"], "request_id": "shared-request"},
    )
    assert resp.status_code == 409, resp.text
    err = resp.json()["error"]
    assert err["code"] == "content_export_request_conflict"
    assert err["details"]["request_id"] == "shared-request"

    # The conflict wrote no second job.
    count = db_session.execute(
        select(func.count()).select_from(ContentExportJob)
    ).scalar_one()
    assert count == 1


def test_distinct_request_ids_are_independent_jobs(client):
    create_actor(client)
    content = _create_content(client, "main")
    a = _create_job(client, content["id"], "req-a")
    b = _create_job(client, content["id"], "req-b")
    assert a["id"] != b["id"]


# --- read -------------------------------------------------------------------


def test_get_unknown_job_is_404(client):
    resp = client.get("/v1/content-export-jobs/cej_doesnotexist")
    assert resp.status_code == 404, resp.text
    err = resp.json()["error"]
    assert err["code"] == "content_export_job_not_found"
    assert err["details"]["job_id"] == "cej_doesnotexist"


def test_get_job_rejects_any_query_parameter(client):
    create_actor(client)
    content = _create_content(client, "main")
    job = _create_job(client, content["id"], "req-1")
    resp = client.get(f"/v1/content-export-jobs/{job['id']}?x=1")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_get_job_is_read_only(client, db_session):
    create_actor(client)
    content = _create_content(client, "main")
    job = _create_job(client, content["id"], "req-1")

    before = db_session.execute(
        select(func.count()).select_from(AuditEvent)
    ).scalar_one()
    for _ in range(2):
        resp = client.get(f"/v1/content-export-jobs/{job['id']}")
        assert resp.status_code == 200
        assert resp.json()["id"] == job["id"]
    after = db_session.execute(
        select(func.count()).select_from(AuditEvent)
    ).scalar_one()
    assert after == before


# --- run --------------------------------------------------------------------


def test_run_succeeds_with_utc_times_and_existing_export_result(client):
    create_actor(client)
    content = _create_content(client, "main")
    other = _create_content(client, "other")
    claim_a = _create_claim(client, content["id"], "a")
    claim_b = _create_claim(client, content["id"], "b", claim_type="review")
    foreign = _create_claim(client, other["id"], "foreign")
    bundle = _create_bundle(client, claim_a["id"], "b1")
    job = _create_job(client, content["id"], "req-1")

    resp = client.post(f"/v1/content-export-jobs/{job['id']}/run")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "succeeded"
    assert body["started_at"] is not None
    assert body["finished_at"] is not None
    assert _is_utc(body["started_at"])
    assert _is_utc(body["finished_at"])
    assert body["started_at"] <= body["finished_at"]
    assert body["error"] is None

    # The result is exactly the existing read-only content export.
    export = client.get(f"/v1/contents/{content['id']}/export")
    assert export.status_code == 200
    assert body["result"] == export.json()

    result = body["result"]
    assert result["content"]["id"] == content["id"]
    # Direct claims in stable creation order; the foreign content's claim and
    # any lineage never appear.
    assert [c["id"] for c in result["claims"]] == [
        claim_a["id"],
        claim_b["id"],
    ]
    first_claim = result["claims"][0]
    assert [b["id"] for b in first_claim["evidence_bundles"]] == [
        bundle["id"]
    ]
    # A claim without evidence bundles exports an empty array.
    assert result["claims"][1]["evidence_bundles"] == []
    assert foreign["id"] not in [c["id"] for c in result["claims"]]


def test_run_empty_export_succeeds_with_empty_claims(client):
    create_actor(client)
    content = _create_content(client, "lonely")
    job = _create_job(client, content["id"], "req-1")

    resp = client.post(f"/v1/content-export-jobs/{job['id']}/run")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "succeeded"
    export = client.get(f"/v1/contents/{content['id']}/export")
    assert body["result"] == export.json()
    assert body["result"]["claims"] == []


def test_run_unknown_job_is_404(client):
    resp = client.post("/v1/content-export-jobs/cej_doesnotexist/run")
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "content_export_job_not_found"


def test_run_rejects_any_query_parameter(client):
    create_actor(client)
    content = _create_content(client, "main")
    job = _create_job(client, content["id"], "req-1")
    resp = client.post(f"/v1/content-export-jobs/{job['id']}/run?x=1")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_run_is_idempotent_only_once_repeat_is_409(client):
    create_actor(client)
    content = _create_content(client, "main")
    job = _create_job(client, content["id"], "req-1")

    first = client.post(f"/v1/content-export-jobs/{job['id']}/run")
    assert first.status_code == 200
    assert first.json()["status"] == "succeeded"

    for _ in range(2):
        repeat = client.post(f"/v1/content-export-jobs/{job['id']}/run")
        assert repeat.status_code == 409, repeat.text
        assert repeat.json()["error"]["code"] == "content_export_job_conflict"

    # The succeeded outcome is unchanged by the rejected repeats.
    read = client.get(f"/v1/content-export-jobs/{job['id']}")
    assert read.json()["status"] == "succeeded"
    assert read.json()["result"] is not None


def test_concurrent_runs_claim_pending_exactly_once(file_client):
    # File-backed SQLite serializes writers; the conditional pending claim
    # must let exactly one run win and the other receive 409.
    create_actor(file_client)
    content = _create_content(file_client, "main")
    job_id = _create_job(file_client, content["id"], "req-1")["id"]

    entered = threading.Event()
    release = threading.Event()
    original = service.get_content_export

    def patched(session, content_id):
        entered.set()
        release.wait(timeout=10)
        return original(session, content_id)

    service.get_content_export = patched
    results: dict[str, tuple] = {}
    try:
        def winner():
            r = file_client.post(f"/v1/content-export-jobs/{job_id}/run")
            results["winner"] = (r.status_code, r.json()["status"])

        def loser():
            entered.wait(timeout=10)
            time.sleep(0.5)  # run while the winner holds the claim
            r = file_client.post(f"/v1/content-export-jobs/{job_id}/run")
            results["loser"] = (r.status_code, r.json()["error"]["code"])

        t_winner = threading.Thread(target=winner)
        t_loser = threading.Thread(target=loser)
        t_winner.start()
        t_loser.start()
        entered.wait(timeout=10)
        release.set()
        t_winner.join(timeout=10)
        t_loser.join(timeout=10)
    finally:
        service.get_content_export = original

    assert results == {
        "winner": (200, "succeeded"),
        "loser": (409, "content_export_job_conflict"),
    }


def test_run_export_error_marks_job_failed(client, monkeypatch):
    create_actor(client)
    content = _create_content(client, "main")
    job = _create_job(client, content["id"], "req-1")

    def boom(session, content_id):
        raise RuntimeError("synthetic export failure")

    monkeypatch.setattr(service, "get_content_export", boom)

    resp = client.post(f"/v1/content-export-jobs/{job['id']}/run")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "failed"
    assert body["result"] is None
    assert body["error"] == "content_export_failed"
    assert body["started_at"] is not None
    assert body["finished_at"] is not None
    assert body["started_at"] <= body["finished_at"]

    # A failed job is terminal: a further run is a 409 and it stays failed.
    repeat = client.post(f"/v1/content-export-jobs/{job['id']}/run")
    assert repeat.status_code == 409
    read = client.get(f"/v1/content-export-jobs/{job['id']}").json()
    assert read["status"] == "failed"
    assert read["result"] is None
    assert read["error"] == "content_export_failed"


# --- audit ------------------------------------------------------------------


def _audit_events(session):
    rows = session.execute(
        select(AuditEvent).order_by(AuditEvent.seq.asc())
    ).scalars().all()
    return [(r.event_type, r.resource_id) for r in rows]


def test_job_lifecycle_writes_one_create_and_one_run_audit_event(client, db_session):
    create_actor(client)
    content = _create_content(client, "main")
    job = _create_job(client, content["id"], "req-1")

    assert EVENT_CONTENT_EXPORT_JOB_CREATED in [
        e[0] for e in _audit_events(db_session)
    ]
    create_events = [
        e for e in _audit_events(db_session)
        if e[0] == EVENT_CONTENT_EXPORT_JOB_CREATED
    ]
    assert create_events == [(EVENT_CONTENT_EXPORT_JOB_CREATED, job["id"])]

    client.post(f"/v1/content-export-jobs/{job['id']}/run")

    run_events = [
        e for e in _audit_events(db_session) if e[0] == EVENT_CONTENT_EXPORT_JOB_RUN
    ]
    assert run_events == [(EVENT_CONTENT_EXPORT_JOB_RUN, job["id"])]


def test_idempotent_and_conflicting_creates_write_no_audit(client, db_session):
    create_actor(client)
    first = _create_content(client, "first")
    second = _create_content(client, "second")
    _create_job(client, first["id"], "shared")
    after_create = len(_audit_events(db_session))

    # Same pair: no new event.
    same = client.post(
        "/v1/content-export-jobs",
        json={"content_id": first["id"], "request_id": "shared"},
    )
    assert same.status_code == 200
    # Cross-content conflict: no new event.
    conflict = client.post(
        "/v1/content-export-jobs",
        json={"content_id": second["id"], "request_id": "shared"},
    )
    assert conflict.status_code == 409
    # Unknown content: no new event.
    missing = client.post(
        "/v1/content-export-jobs",
        json={"content_id": "cnt_missing", "request_id": "shared"},
    )
    assert missing.status_code == 404

    assert len(_audit_events(db_session)) == after_create


def test_rejected_run_writes_no_audit_event(client, db_session):
    create_actor(client)
    content = _create_content(client, "main")
    job = _create_job(client, content["id"], "req-1")
    client.post(f"/v1/content-export-jobs/{job['id']}/run")
    after_first = len(_audit_events(db_session))

    repeat = client.post(f"/v1/content-export-jobs/{job['id']}/run")
    assert repeat.status_code == 409
    unknown = client.post("/v1/content-export-jobs/cej_missing/run")
    assert unknown.status_code == 404
    assert len(_audit_events(db_session)) == after_first


def test_failed_run_still_writes_run_audit_event(client, db_session, monkeypatch):
    create_actor(client)
    content = _create_content(client, "main")
    job = _create_job(client, content["id"], "req-1")

    def boom(session, content_id):
        raise RuntimeError("synthetic export failure")

    monkeypatch.setattr(service, "get_content_export", boom)
    client.post(f"/v1/content-export-jobs/{job['id']}/run")

    run_events = [
        e for e in _audit_events(db_session) if e[0] == EVENT_CONTENT_EXPORT_JOB_RUN
    ]
    assert run_events == [(EVENT_CONTENT_EXPORT_JOB_RUN, job["id"])]
