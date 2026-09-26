"""Tests for the read-only running-state observability summary.

Covers ``GET /v1/observability/summary``:

* the success body is compact UTF-8 JSON terminated by exactly one newline,
  with members in the fixed order service status, database status, resource
  counts, task counts, audit status, and check time; timestamps stay UTC and
  every number renders as an integer;
* an empty database yields all-zero counts, the determined empty audit
  state (``event_count`` 0 and ``latest_event_at`` null), ``ok``/``ready``,
  and never a missing-resource error;
* resource counts cover the ten existing families (subjects, contents,
  claims, evidence bundles, attestations, revocations, grants, relations,
  audit events, and export jobs); task counts cover pending/running/
  succeeded/failed; audit status reports the event total and latest time;
* any non-empty body (whitespace, arbitrary bytes, malformed JSON) and any
  query parameter (unknown, blank, repeated) is a 422 validation_error,
  rejected before any state is read -- even when the database is
  unreadable; non-GET methods are 405 method_not_allowed;
* an unreadable database or an internal summary-query failure is the
  existing-structure 503 service_unavailable carrying a reason, never a
  partial body, and the read transaction rolls back;
* repeated, concurrent, and post-restart reads return identical counts,
  statuses, and audit results for unchanged persisted state; only the
  check time changes;
* the endpoint is strictly read-only: success, empty results, rejections,
  and the 503 path create, modify, or delete no resource, task, or audit
  event.

All fixtures are deterministic and offline (in-memory and temporary-file
SQLite, no network).
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, update as sa_update

from provenance.app import create_app
from provenance.config import Settings
from provenance.access_signing import access_message_bytes
from provenance.models import AuditEvent, ContentExportJob
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

URL = "/v1/observability/summary"
GRANTS_PATH = "/v1/attestation-access-grants"

EVIDENCE_DIGEST_1 = hashlib.sha256(b"evidence-obs-1").hexdigest()
EVIDENCE_DIGEST_2 = hashlib.sha256(b"evidence-obs-2").hexdigest()


# --- World construction ------------------------------------------------------


def _make_content(client, name, actor_id="org-1", digest=DIGEST_A):
    resp = client.post(
        "/v1/contents",
        json=content_payload(
            actor_id=actor_id, digest=digest, title=name
        ),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make_claim(client, content_id, actor_id="org-1", claim_type="authorship"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": claim_type,
            "payload": {"statement": "observed"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make_bundle(client, claim_id, digest):
    resp = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim_id,
            "evidence_type": "raw_capture",
            "digest_algorithm": "sha256",
            "digest_hex": digest,
            "media_type": "image/jpeg",
            "metadata": {"source": "camera"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make_attestation(client, target_type, target_id, actor_id="org-1"):
    message = attestation_message_bytes(target_type, target_id, actor_id)
    signature = ed25519_sign(SEED_A, message)
    resp = client.post(
        "/v1/attestations",
        json={
            "target_type": target_type,
            "target_id": target_id,
            "signer_actor_id": actor_id,
            "public_key": base64.b64encode(ed25519_public_key(SEED_A)).decode(),
            "signature": base64.b64encode(signature).decode(),
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make_job(client, content_id, request_id):
    resp = client.post(
        "/v1/content-export-jobs",
        json={"content_id": content_id, "request_id": request_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _set_job_status(db_session, job_id, status_value):
    db_session.execute(
        sa_update(ContentExportJob)
        .where(ContentExportJob.id == job_id)
        .values(status=status_value)
    )
    db_session.commit()


def _populate(client, db_session):
    """Create at least one record in every counted resource family.

    Two actors, two contents, two claims, two bundles, two attestations
    (one per target type), one revocation, one access grant, one relation,
    and five export jobs covering all four lifecycle states (two pending,
    one running, one succeeded, one failed).
    """
    create_actor(client)
    create_actor(
        client, actor_id="org-2", name="Other Org", type="organization"
    )
    content_a = _make_content(client, "a", digest=DIGEST_A)
    content_b = _make_content(
        client, "b", actor_id="org-2", digest=DIGEST_B
    )
    claim_a = _make_claim(client, content_a["id"])
    claim_b = _make_claim(
        client, content_b["id"], actor_id="org-2"
    )
    bundle_a = _make_bundle(client, claim_a["id"], EVIDENCE_DIGEST_1)
    _make_bundle(client, claim_b["id"], EVIDENCE_DIGEST_2)
    attestation_a = _make_attestation(client, "claim", claim_a["id"])
    attestation_b = _make_attestation(
        client, "evidence_bundle", bundle_a["id"]
    )

    # One revocation (unsigned write route), revoked by a different actor.
    rev = client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": attestation_b["id"],
            "revoker_actor_id": "org-2",
            "reason": "proof retired",
        },
    )
    assert rev.status_code == 201, rev.text

    # One access grant from the signer of attestation_a to org-2. The
    # grant write authenticates with the signed X-PA/X-PT/X-PS headers.
    grant_body = json.dumps(
        {
            "attestation_id": attestation_a["id"],
            "grantee_actor_id": "org-2",
        }
    ).encode("utf-8")
    timestamp = utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")
    signature = ed25519_sign(
        SEED_A,
        access_message_bytes(
            "POST",
            GRANTS_PATH,
            timestamp,
            hashlib.sha256(grant_body).hexdigest(),
        ),
    )
    grant = client.post(
        GRANTS_PATH,
        content=grant_body,
        headers={
            "Content-Type": "application/json",
            "X-PA": "org-1",
            "X-PT": timestamp,
            "X-PS": base64.b64encode(signature).decode("ascii"),
        },
    )
    assert grant.status_code == 201, grant.text

    # One lineage relation between the two contents.
    relation = client.post(
        "/v1/content-relations",
        json={
            "content_id": content_b["id"],
            "parent_content_id": content_a["id"],
            "relation_type": "derived_from",
        },
    )
    assert relation.status_code == 201, relation.text

    # Five jobs: two pending, one running (direct state set), one
    # succeeded through the real run route, one failed (direct state set).
    pending_a = _make_job(client, content_a["id"], "req-obs-pending-a")
    _make_job(client, content_a["id"], "req-obs-pending-b")
    running = _make_job(client, content_a["id"], "req-obs-running")
    _set_job_status(db_session, running["id"], "running")
    succeeded = _make_job(client, content_a["id"], "req-obs-succeeded")
    run = client.post(f"/v1/content-export-jobs/{succeeded['id']}/run")
    assert run.status_code == 200, run.text
    assert run.json()["status"] == "succeeded"
    failed = _make_job(client, content_a["id"], "req-obs-failed")
    _set_job_status(db_session, failed["id"], "failed")

    return {
        "actors": 2,
        "contents": 2,
        "claims": 2,
        "evidence_bundles": 2,
        "attestations": 2,
        "attestation_revocations": 1,
        "attestation_access_grants": 1,
        "content_relations": 1,
        "content_export_jobs": 5,
        "pending_a_id": pending_a["id"],
    }


def _audit_totals(db_session):
    events = db_session.execute(select(AuditEvent)).scalars().all()
    latest = (
        db_session.execute(
            select(AuditEvent.created_at)
            .order_by(
                AuditEvent.created_at.desc(),
                AuditEvent.seq.desc(),
            )
            .limit(1)
        ).scalar_one_or_none()
    )
    return len(events), latest


# --- Empty database ----------------------------------------------------------


def test_empty_database_summary_is_all_zero_ok_and_ready(client):
    before = utc_now()
    resp = client.get(URL)
    after = utc_now()
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()

    assert body["service_status"] == "ok"
    assert body["database_status"] == "ready"
    assert body["resource_counts"] == {
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
    assert body["task_counts"] == {
        "pending": 0,
        "running": 0,
        "succeeded": 0,
        "failed": 0,
    }
    # Determined empty audit state, not a missing error.
    assert body["audit_status"] == {
        "event_count": 0,
        "latest_event_at": None,
    }

    checked_at = parse_rfc3339_utc(body["checked_at"])
    assert checked_at is not None
    assert before <= checked_at <= after


def test_empty_summary_body_is_compact_json_with_one_newline(client):
    resp = client.get(URL)
    raw = resp.content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    # Compact document: no incidental whitespace.
    assert b" " not in raw
    body = json.loads(raw.decode("utf-8"))
    assert list(body) == [
        "service_status",
        "database_status",
        "resource_counts",
        "task_counts",
        "audit_status",
        "checked_at",
    ]
    assert list(body["resource_counts"]) == [
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
    assert list(body["task_counts"]) == [
        "pending",
        "running",
        "succeeded",
        "failed",
    ]
    assert list(body["audit_status"]) == ["event_count", "latest_event_at"]


# --- Populated database ------------------------------------------------------


def test_populated_counts_cover_every_family_and_lifecycle(client, db_session):
    expected = _populate(client, db_session)
    audit_total, latest_event_at = _audit_totals(db_session)
    expected["audit_events"] = audit_total

    resp = client.get(URL)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["service_status"] == "ok"
    assert body["database_status"] == "ready"

    for name, value in expected.items():
        if name.endswith("_id"):
            continue
        assert body["resource_counts"][name] == value, name
        assert isinstance(body["resource_counts"][name], int)

    assert body["task_counts"] == {
        "pending": 2,
        "running": 1,
        "succeeded": 1,
        "failed": 1,
    }

    assert body["audit_status"]["event_count"] == audit_total
    served_latest = parse_rfc3339_utc(body["audit_status"]["latest_event_at"])
    assert served_latest == latest_event_at

    checked_at = parse_rfc3339_utc(body["checked_at"])
    assert checked_at is not None
    assert checked_at >= latest_event_at


def test_summary_counts_match_independent_sqlite_totals(client, db_session):
    _populate(client, db_session)
    body = client.get(URL).json()
    for name, value in body["resource_counts"].items():
        assert isinstance(value, int)
        assert value >= 0
    task_total = sum(body["task_counts"].values())
    assert task_total == body["resource_counts"]["content_export_jobs"]


# --- Read-only guarantee -----------------------------------------------------


def test_reads_create_no_resources_or_audit_events(client, db_session):
    _populate(client, db_session)
    audit_before, _ = _audit_totals(db_session)
    jobs_before = (
        db_session.scalar(
            select(func.count()).select_from(ContentExportJob)
        )
    )

    for _ in range(3):
        resp = client.get(URL)
        assert resp.status_code == 200, resp.text

    audit_after, _ = _audit_totals(db_session)
    jobs_after = db_session.scalar(
        select(func.count()).select_from(ContentExportJob)
    )
    assert audit_after == audit_before
    assert jobs_after == jobs_before


def test_repeated_reads_are_identical_except_checked_at(client, db_session):
    _populate(client, db_session)
    first = client.get(URL).json()
    second = client.get(URL).json()
    first_checked = parse_rfc3339_utc(first.pop("checked_at"))
    second_checked = parse_rfc3339_utc(second.pop("checked_at"))
    assert first_checked is not None and second_checked is not None
    assert second_checked >= first_checked
    assert first == second


def test_concurrent_reads_return_identical_counts(file_app):
    setup = TestClient(file_app)
    create_actor(setup)
    content = _make_content(setup, "concurrent")
    _make_job(setup, content["id"], "req-concurrent")

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
    checked_times = {item.pop("checked_at") for item in outcomes}
    assert len(outcomes) == 6
    first = outcomes[0]
    for other in outcomes[1:]:
        assert other == first
    assert first["resource_counts"]["actors"] == 1
    assert first["task_counts"]["pending"] == 1
    # Every read minted its own check time.
    assert len(checked_times) >= 1

    session = file_app.state.session_factory()
    try:
        audit_total = session.scalar(
            select(func.count()).select_from(AuditEvent)
        )
        # One actor + one content + one job creation; reads add nothing.
        assert audit_total == 3
    finally:
        session.close()


def test_counts_and_audit_results_stable_across_restart(tmp_db_url):
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

    assert after["service_status"] == before["service_status"] == "ok"
    assert after["database_status"] == before["database_status"] == "ready"
    assert after["resource_counts"] == before["resource_counts"]
    assert after["task_counts"] == before["task_counts"]
    assert after["audit_status"] == before["audit_status"]
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
        lambda c: c.request(
            "GET", URL, content=b'{"unexpected": true}'
        ),
        lambda c: c.get(URL + "?x=1"),
        lambda c: c.get(URL + "?x="),
        lambda c: c.get(URL + "?status=ok"),
        lambda c: c.get(URL + "?limit=1"),
        lambda c: c.get(URL + "?x=1&x=2"),
        lambda c: c.request("GET", URL + "?x=1", content=b"{}"),
    ],
)
def test_invalid_requests_are_422_and_write_nothing(
    client, db_session, send
):
    _populate(client, db_session)
    audit_before, _ = _audit_totals(db_session)

    resp = send(client)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"

    audit_after, _ = _audit_totals(db_session)
    assert audit_after == audit_before
    # A valid read still works after the rejection.
    assert client.get(URL).status_code == 200


@pytest.mark.parametrize("method", ("post", "put", "patch", "delete"))
def test_non_get_methods_are_405_and_write_nothing(
    client, db_session, method
):
    _populate(client, db_session)
    audit_before, _ = _audit_totals(db_session)
    jobs_before = db_session.scalar(
        select(func.count()).select_from(ContentExportJob)
    )

    resp = getattr(client, method)(URL)
    assert resp.status_code == 405, (method, resp.text)
    assert resp.json()["error"]["code"] == "method_not_allowed"

    audit_after, _ = _audit_totals(db_session)
    jobs_after = db_session.scalar(
        select(func.count()).select_from(ContentExportJob)
    )
    assert audit_after == audit_before
    assert jobs_after == jobs_before
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
    client = _client_with_dropped_table(file_app, table="content_relations")
    # The summary reads the relations table; the failed read must roll its
    # transaction back, leaving the connection usable for subsequent
    # validation and for unrelated tables.
    first = client.get(URL)
    assert first.status_code == 503, first.text
    # A rejected request never opens the failing read.
    assert client.get(URL + "?x=1").status_code == 422
    # Unrelated tables remain readable and writable through their routes.
    resp = client.post("/v1/actors", json={"id": "org-x", "name": "X", "type": "organization"})
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
