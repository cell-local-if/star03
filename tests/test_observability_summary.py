"""Tests for the read-only runtime status summary.

Covers ``GET /v1/observability/summary``:

* an empty database returns zero resource/task counts, a definite empty
  audit status (zero events, null latest time), ``service`` ``ok`` and
  ``database`` ``ready`` -- never a missing-resource error;
* the resource counts cover the ten existing collections (actors,
  contents, claims, evidence bundles, attestations, attestation
  revocations, attestation access grants, content relations, audit
  events, content export jobs) and the task counts cover the four
  lifecycle states (pending/running/succeeded/failed); the audit status
  reports the total event count and the latest event time;
* the body is compact UTF-8 JSON in fixed member order terminated by
  exactly one newline, with UTC times and integral numbers only;
* repeated reads, concurrent reads, and reads across an app restart
  return the same counts, statuses, and audit result; only the check
  time changes;
* an empty/whitespace/malformed-JSON body and unknown, blank, or
  repeated query parameters are 422 ``validation_error`` rejected before
  any state is read;
* non-GET methods are 405 ``method_not_allowed`` and create, modify, or
  delete nothing;
* an unreadable database or an internal aggregation failure is a 503
  ``service_unavailable`` carrying a reason under the existing error
  JSON structure;
* every read (and every failure) is strictly read-only: no resource,
  task, or audit event is created, modified, or deleted.

All fixtures are deterministic and offline (in-memory and temporary-file
SQLite, no network).
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update as sa_update
from sqlalchemy.exc import OperationalError

from provenance import service
from provenance.access_signing import access_message_bytes
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import (
    AttestationAccessGrant,
    AttestationRevocation,
    AuditEvent,
    ContentExportJob,
)
from provenance.signing import attestation_message_bytes
from provenance.time_utils import parse_rfc3339_utc, utc_now
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    content_payload,
    create_actor,
    ed25519_public_key,
    ed25519_sign,
    SEED_A,
)

SUMMARY_URL = "/v1/observability/summary"
GRANTS_PATH = "/v1/attestation-access-grants"
EVIDENCE_DIGEST = "f" * 64


# --- Deterministic world -----------------------------------------------------


def _create_content(client, name, *, actor_id="org-1", digest=DIGEST_A):
    resp = client.post(
        "/v1/contents",
        json=content_payload(actor_id=actor_id, digest=digest, title=name),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, *, actor_id="org-1", claim_type="authorship"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": claim_type,
            "payload": {"statement": "summarized"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_bundle(client, claim_id):
    resp = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim_id,
            "evidence_type": "raw_capture",
            "digest_algorithm": "sha256",
            "digest_hex": EVIDENCE_DIGEST,
            "media_type": "image/jpeg",
            "metadata": {"source": "camera-1"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_attestation(client, target_id):
    signature = ed25519_sign(
        SEED_A, attestation_message_bytes("claim", target_id, "org-1")
    )
    import base64

    resp = client.post(
        "/v1/attestations",
        json={
            "target_type": "claim",
            "target_id": target_id,
            "signer_actor_id": "org-1",
            "public_key": base64.b64encode(ed25519_public_key(SEED_A)).decode(),
            "signature": base64.b64encode(signature).decode(),
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _signed_grant_headers(body: bytes) -> dict:
    import base64
    import hashlib

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    message = access_message_bytes(
        "POST", GRANTS_PATH, ts, hashlib.sha256(body).hexdigest()
    )
    signature = base64.b64encode(ed25519_sign(SEED_A, message)).decode("ascii")
    return {
        "Content-Type": "application/json",
        "X-PA": "org-1",
        "X-PT": ts,
        "X-PS": signature,
    }


def _create_grant(client, attestation_id):
    body = json.dumps(
        {"attestation_id": attestation_id, "grantee_actor_id": "org-2"}
    ).encode("utf-8")
    resp = client.post(GRANTS_PATH, content=body, headers=_signed_grant_headers(body))
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_revocation(client, attestation_id):
    resp = client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": attestation_id,
            "revoker_actor_id": "org-2",
            "reason": "key compromise drill",
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


def build_world(client, db_session) -> dict:
    """Create at least one row of every counted collection and task state."""
    create_actor(client)  # org-1
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    parent = _create_content(client, "parent", digest=DIGEST_B)
    content = _create_content(client, "main", digest=DIGEST_A)
    claim = _create_claim(client, content["id"])
    _create_bundle(client, claim["id"])
    attestation = _create_attestation(client, claim["id"])
    _create_grant(client, attestation["id"])
    _create_revocation(client, attestation["id"])

    rel = client.post(
        "/v1/content-relations",
        json={
            "content_id": content["id"],
            "parent_content_id": parent["id"],
            "relation_type": "version_of",
        },
    )
    assert rel.status_code == 201, rel.text

    pending = _create_job(client, content["id"], "obs-req-pending")
    running = _create_job(client, content["id"], "obs-req-running")
    succeeded = _create_job(client, content["id"], "obs-req-succeeded")
    failed = _create_job(client, content["id"], "obs-req-failed")

    run = client.post(f"/v1/content-export-jobs/{succeeded['id']}/run")
    assert run.status_code == 200
    db_session.execute(
        sa_update(ContentExportJob)
        .where(ContentExportJob.id == running["id"])
        .values(status="running")
    )
    db_session.execute(
        sa_update(ContentExportJob)
        .where(ContentExportJob.id == failed["id"])
        .values(status="failed", error="content_export_failed")
    )
    db_session.commit()

    return {
        "pending_id": pending["id"],
        "running_id": running["id"],
        "succeeded_id": succeeded["id"],
        "failed_id": failed["id"],
    }


def _expected_resource_counts() -> dict:
    return {
        "actors": 2,
        "contents": 2,
        "claims": 1,
        "evidence_bundles": 1,
        "attestations": 1,
        "attestation_revocations": 1,
        "attestation_access_grants": 1,
        "content_relations": 1,
        "content_export_jobs": 4,
        # audit_events is asserted from the database directly.
    }


# --- Empty database ----------------------------------------------------------


def test_empty_database_reports_zero_counts_and_ok(client, db_session):
    resp = client.get(SUMMARY_URL)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    assert list(body) == [
        "service",
        "database",
        "resources",
        "tasks",
        "audit",
        "checked_at",
    ]
    assert body["service"] == {"status": "ok"}
    assert body["database"] == {"status": "ready"}
    assert list(body["resources"]) == [
        "actors",
        "contents",
        "claims",
        "evidence_bundles",
        "attestations",
        "attestation_revocations",
        "attestation_access_grants",
        "content_relations",
        "audit_events",
        "content_export_jobs",
    ]
    assert body["resources"] == {
        "actors": 0,
        "contents": 0,
        "claims": 0,
        "evidence_bundles": 0,
        "attestations": 0,
        "attestation_revocations": 0,
        "attestation_access_grants": 0,
        "content_relations": 0,
        "audit_events": 0,
        "content_export_jobs": 0,
    }
    assert list(body["tasks"]) == ["pending", "running", "succeeded", "failed"]
    assert body["tasks"] == {
        "pending": 0,
        "running": 0,
        "succeeded": 0,
        "failed": 0,
    }
    # An empty trail is a definite empty status, not a missing member.
    assert list(body["audit"]) == ["event_count", "latest_event_at"]
    assert body["audit"] == {"event_count": 0, "latest_event_at": None}
    assert parse_rfc3339_utc(body["checked_at"]) is not None

    # The read creates nothing: the audit trail stays empty.
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


# --- Wire format -------------------------------------------------------------


def test_body_is_compact_json_fixed_order_single_newline(client):
    before = utc_now()
    raw = client.get(SUMMARY_URL).content
    after = utc_now()
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    # Compact separators: no incidental whitespace in the document.
    assert b" " not in raw
    assert b": " not in raw
    body = json.loads(raw.decode("utf-8"))
    assert list(body) == [
        "service",
        "database",
        "resources",
        "tasks",
        "audit",
        "checked_at",
    ]
    # Every count is rendered as an integer (no floats, no -0).
    for value in body["resources"].values():
        assert isinstance(value, int) and not isinstance(value, bool)
    for value in body["tasks"].values():
        assert isinstance(value, int) and not isinstance(value, bool)
    assert isinstance(body["audit"]["event_count"], int)
    # checked_at is timezone-aware UTC stamped at read time.
    checked_at = parse_rfc3339_utc(body["checked_at"])
    assert checked_at is not None
    assert checked_at.utcoffset().total_seconds() == 0
    assert before <= checked_at <= after


# --- Populated counts --------------------------------------------------------


def test_counts_cover_every_resource_and_task_state(client, db_session):
    build_world(client, db_session)

    body = client.get(SUMMARY_URL).json()
    expected = _expected_resource_counts()
    audit_total = db_session.scalar(select(AuditEvent.seq).order_by(
        AuditEvent.seq.desc()
    ).limit(1))
    # The monotonic seq is dense from 1, so the highest seq is the total.
    expected["audit_events"] = audit_total or 0
    assert body["resources"] == expected
    assert body["tasks"] == {
        "pending": 1,
        "running": 1,
        "succeeded": 1,
        "failed": 1,
    }
    assert body["service"] == {"status": "ok"}
    assert body["database"] == {"status": "ready"}

    # The audit status reports the total and the latest event time, which
    # equals the latest audit row under the stable creation order.
    events = db_session.execute(
        select(AuditEvent).order_by(AuditEvent.seq)
    ).scalars().all()
    assert body["audit"]["event_count"] == len(events)
    assert body["audit"]["event_count"] == body["resources"]["audit_events"]
    latest = events[-1].created_at
    assert body["audit"]["latest_event_at"] == latest.isoformat()
    assert parse_rfc3339_utc(body["audit"]["latest_event_at"]) is not None


def test_status_literals_are_restricted(client, db_session):
    build_world(client, db_session)
    body = client.get(SUMMARY_URL).json()
    assert body["service"]["status"] in ("ok", "degraded")
    assert body["service"]["status"] == "ok"
    assert body["database"]["status"] in ("ready", "unavailable")
    assert body["database"]["status"] == "ready"


# --- Read-only and determinism ----------------------------------------------


def test_summary_read_is_strictly_read_only(client, db_session):
    build_world(client, db_session)
    grants_before = len(
        db_session.execute(select(AttestationAccessGrant)).scalars().all()
    )
    revocations_before = len(
        db_session.execute(select(AttestationRevocation)).scalars().all()
    )
    jobs_before = {
        job.id: job.status
        for job in db_session.execute(select(ContentExportJob)).scalars().all()
    }
    events_before = db_session.scalar(select(AuditEvent.seq).order_by(
        AuditEvent.seq.desc()
    ).limit(1))

    for _ in range(3):
        resp = client.get(SUMMARY_URL)
        assert resp.status_code == 200

    assert len(
        db_session.execute(select(AttestationAccessGrant)).scalars().all()
    ) == grants_before
    assert len(
        db_session.execute(select(AttestationRevocation)).scalars().all()
    ) == revocations_before
    jobs_after = {
        job.id: job.status
        for job in db_session.execute(select(ContentExportJob)).scalars().all()
    }
    assert jobs_after == jobs_before
    events_after = db_session.scalar(select(AuditEvent.seq).order_by(
        AuditEvent.seq.desc()
    ).limit(1))
    assert events_after == events_before


def test_repeated_reads_share_everything_but_checked_at(client, db_session):
    build_world(client, db_session)
    first = client.get(SUMMARY_URL).json()
    second = client.get(SUMMARY_URL).json()
    assert first["service"] == second["service"]
    assert first["database"] == second["database"]
    assert first["resources"] == second["resources"]
    assert first["tasks"] == second["tasks"]
    assert first["audit"] == second["audit"]
    # Only the check time may change, and it never moves backwards.
    assert parse_rfc3339_utc(first["checked_at"]) is not None
    assert parse_rfc3339_utc(second["checked_at"]) is not None
    assert first["checked_at"] <= second["checked_at"]


def test_concurrent_reads_agree(file_app):
    setup = TestClient(file_app)
    build_world(setup, file_app.state.session_factory())

    snapshots: list[dict] = []
    statuses: list[int] = []
    lock = threading.Lock()

    def _read() -> None:
        worker = TestClient(file_app)
        resp = worker.get(SUMMARY_URL)
        with lock:
            statuses.append(resp.status_code)
            snapshots.append(resp.json())

    threads = [threading.Thread(target=_read) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert statuses == [200] * 8
    assert len(snapshots) == 8
    reference = snapshots[0]
    for snapshot in snapshots[1:]:
        assert snapshot["resources"] == reference["resources"]
        assert snapshot["tasks"] == reference["tasks"]
        assert snapshot["audit"] == reference["audit"]
        assert snapshot["service"] == reference["service"]
        assert snapshot["database"] == reference["database"]
    # Only the check time is allowed to differ between reads.
    assert len({s["checked_at"] for s in snapshots}) >= 1


def test_counts_and_audit_are_stable_across_restart(tmp_db_url):
    app1 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app1) as first:
        build_world(first, app1.state.session_factory())
        before = first.get(SUMMARY_URL).json()

    app2 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app2) as second:
        after = second.get(SUMMARY_URL).json()

    assert after["service"] == before["service"]
    assert after["database"] == before["database"]
    assert after["resources"] == before["resources"]
    assert after["tasks"] == before["tasks"]
    assert after["audit"] == before["audit"]


# --- Request validation ------------------------------------------------------


@pytest.mark.parametrize(
    "send",
    [
        lambda c: c.request("GET", SUMMARY_URL, content=b"   "),
        lambda c: c.request(
            "GET",
            SUMMARY_URL,
            content=b"{not json",
            headers={"content-type": "application/json"},
        ),
        lambda c: c.request("GET", SUMMARY_URL, content=b"{}"),
        lambda c: c.request(
            "GET", SUMMARY_URL, content=b'{"unexpected": true}'
        ),
        lambda c: c.get(SUMMARY_URL + "?x=1"),
        lambda c: c.get(SUMMARY_URL + "?status=ok"),
        lambda c: c.get(SUMMARY_URL + "?x="),
        lambda c: c.get(SUMMARY_URL + "?x=1&x=2"),
        lambda c: c.request("GET", SUMMARY_URL + "?x=1", content=b"{}"),
    ],
)
def test_invalid_inputs_are_422_and_read_nothing(client, db_session, send):
    build_world(client, db_session)
    expected = client.get(SUMMARY_URL).json()
    audit_seq = db_session.scalar(select(AuditEvent.seq).order_by(
        AuditEvent.seq.desc()
    ).limit(1))

    resp = send(client)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"

    # Rejected before any state read/write: the summary is unchanged and no
    # audit event was added.
    assert client.get(SUMMARY_URL).json()["resources"] == expected["resources"]
    assert db_session.scalar(select(AuditEvent.seq).order_by(
        AuditEvent.seq.desc()
    ).limit(1)) == audit_seq


def test_validation_runs_before_the_state_read(client, monkeypatch):
    # If the aggregation itself could not run, an invalid request must still
    # be the 422 from input validation -- never the 503 from the read.
    def _boom(session):
        raise OperationalError("select 1", {}, Exception("unreadable"))

    monkeypatch.setattr(service, "summarize_observability", _boom)
    resp = client.request("GET", SUMMARY_URL, content=b" ")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    resp = client.get(SUMMARY_URL + "?x=1")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Method boundary ---------------------------------------------------------


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_non_get_methods_are_405_and_write_nothing(
    client, db_session, method
):
    build_world(client, db_session)
    audit_seq = db_session.scalar(select(AuditEvent.seq).order_by(
        AuditEvent.seq.desc()
    ).limit(1))
    grants = len(
        db_session.execute(select(AttestationAccessGrant)).scalars().all()
    )

    resp = getattr(client, method)(SUMMARY_URL)
    assert resp.status_code == 405, resp.text
    assert resp.json()["error"]["code"] == "method_not_allowed"

    # Even a non-GET request carrying a body never reaches a write path.
    assert len(
        db_session.execute(select(AttestationAccessGrant)).scalars().all()
    ) == grants
    assert db_session.scalar(select(AuditEvent.seq).order_by(
        AuditEvent.seq.desc()
    ).limit(1)) == audit_seq
    # A subsequent GET still serves the same summary.
    assert client.get(SUMMARY_URL).status_code == 200


# --- Unreadable database / aggregation failure ------------------------------


def test_unreadable_database_is_503_service_unavailable(tmp_db_url):
    app = create_app(Settings(database_url="sqlite:///:memory:"))
    with TestClient(app) as client:
        # The shared in-memory connection holds the schema; disposing the
        # engine drops it, so the next read hits a missing-table failure.
        app.state.engine.dispose()
        resp = client.get(SUMMARY_URL)
    assert resp.status_code == 503, resp.text
    error = resp.json()["error"]
    assert error["code"] == "service_unavailable"
    assert "message" in error
    assert error["details"]["reason"] == "database_unavailable"


def test_missing_file_database_is_503_service_unavailable(tmp_db_url):
    """A file-backed database that disappears is unreadable, not degraded."""
    import os

    app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app) as client:
        assert client.get(SUMMARY_URL).status_code == 200
        app.state.engine.dispose()
        raw_path = tmp_db_url.removeprefix("sqlite:///")
        os.remove(raw_path)
        resp = client.get(SUMMARY_URL)
    assert resp.status_code == 503, resp.text
    error = resp.json()["error"]
    assert error["code"] == "service_unavailable"
    assert error["details"]["reason"] == "database_unavailable"


def test_internal_aggregation_failure_is_503_and_read_only(
    client, db_session, monkeypatch
):
    build_world(client, db_session)
    audit_seq = db_session.scalar(select(AuditEvent.seq).order_by(
        AuditEvent.seq.desc()
    ).limit(1))

    def _boom(session):
        raise OperationalError("select count(1)", {}, Exception("boom"))

    monkeypatch.setattr(service, "summarize_observability", _boom)
    resp = client.get(SUMMARY_URL)
    assert resp.status_code == 503, resp.text
    error = resp.json()["error"]
    assert error["code"] == "service_unavailable"
    assert error["details"]["reason"] == "database_unavailable"

    # The failure writes nothing; the world is intact afterwards.
    db_session.rollback()
    assert db_session.scalar(select(AuditEvent.seq).order_by(
        AuditEvent.seq.desc()
    ).limit(1)) == audit_seq
    monkeypatch.undo()
    assert client.get(SUMMARY_URL).status_code == 200


# --- Empty audit status edge case --------------------------------------------


def test_audit_latest_time_is_null_without_events(client):
    body = client.get(SUMMARY_URL).json()
    assert body["audit"] == {"event_count": 0, "latest_event_at": None}


def test_audit_latest_time_tracks_the_newest_event(client, db_session):
    first = client.get(SUMMARY_URL).json()
    assert first["audit"]["event_count"] == 0

    create_actor(client)
    second = client.get(SUMMARY_URL).json()
    newest = db_session.execute(
        select(AuditEvent).order_by(AuditEvent.seq.desc())
    ).scalars().first()
    assert second["audit"]["event_count"] == 1
    assert second["audit"]["latest_event_at"] == newest.created_at.isoformat()

    # A later event moves the reported latest time with it.
    create_actor(client, actor_id="org-2", name="Other", type="organization")
    third = client.get(SUMMARY_URL).json()
    newest = db_session.execute(
        select(AuditEvent).order_by(AuditEvent.seq.desc())
    ).scalars().first()
    assert third["audit"]["event_count"] == 2
    assert third["audit"]["latest_event_at"] == newest.created_at.isoformat()
    assert third["audit"]["latest_event_at"] >= second["audit"]["latest_event_at"]


# --- Non-database internal failure -------------------------------------------


def test_non_database_failure_is_503_with_reason(
    client, db_session, monkeypatch
):
    audit_seq = db_session.scalar(
        select(AuditEvent.seq).order_by(AuditEvent.seq.desc()).limit(1)
    )

    def _boom(session):
        raise RuntimeError("aggregation blew up")

    monkeypatch.setattr(service, "summarize_observability", _boom)
    resp = client.get(SUMMARY_URL)
    assert resp.status_code == 503, resp.text
    error = resp.json()["error"]
    assert error["code"] == "service_unavailable"
    assert error["details"]["reason"] == "summary_query_failed"
    db_session.rollback()
    assert (
        db_session.scalar(
            select(AuditEvent.seq).order_by(AuditEvent.seq.desc()).limit(1)
        )
        == audit_seq
    )
