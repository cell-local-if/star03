"""Tests for immutable evidence bundle revocation records.

Covers POST /v1/evidence-bundle-revocations, GET by id, and the per-bundle
list, including: first-creation 201 response fields and UTC timestamps,
reason trimming, idempotent 200 retries that add no record or audit,
independent records for different revoker/reason combinations, per-bundle
isolation and stable creation ordering, the 404 boundary (unknown bundle,
revocation, revoker), the 422 boundary (missing/blank/wrong-typed/extra/
duplicate fields, malformed JSON, GET query parameters and bodies) with no
writes, the 405 boundary (PUT/PATCH/DELETE), append-only preservation of
the revoked bundle, single-transaction audit commit, persistence across
restarts, and the legacy v3 -> v4 schema upgrade. All tests are
deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime

import pytest
from sqlalchemy import func, select

from provenance import migrations
from provenance.app import create_app
from provenance.config import Settings
from provenance.database import make_engine
from provenance.models import (
    EVENT_EVIDENCE_BUNDLE_REVOKED,
    AuditEvent,
    EvidenceBundleRevocation,
)
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    content_payload,
    create_actor,
)

REASON_A = "superseded by a newer capture"
REASON_B = "captured from an unverified source"

EVIDENCE_DIGEST_A = hashlib.sha256(b"evidence-rev-a").hexdigest()
EVIDENCE_DIGEST_B = hashlib.sha256(b"evidence-rev-b").hexdigest()


# --- Setup helpers ----------------------------------------------------------


def _create_claim(client, actor_id="org-1", digest=DIGEST_A):
    resp = client.post(
        "/v1/contents", json=content_payload(actor_id=actor_id, digest=digest)
    )
    assert resp.status_code == 201, resp.text
    content_id = resp.json()["id"]
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": "authorship",
            "payload": {"statement": "captured"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_bundle(client, claim_id, digest=EVIDENCE_DIGEST_A):
    resp = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim_id,
            "evidence_type": "raw_capture",
            "digest_algorithm": "sha256",
            "digest_hex": digest,
            "media_type": "image/jpeg",
            "metadata": {"source": "camera-1"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _setup_bundle(client, digest=EVIDENCE_DIGEST_A):
    create_actor(client)
    claim = _create_claim(client)
    return _create_bundle(client, claim["id"], digest=digest)


def _revocation_payload(
    evidence_bundle_id, *, revoker_actor_id="org-1", reason=REASON_A
):
    return {
        "evidence_bundle_id": evidence_bundle_id,
        "revoker_actor_id": revoker_actor_id,
        "reason": reason,
    }


def _revoke(client, *args, **kwargs):
    resp = client.post(
        "/v1/evidence-bundle-revocations",
        json=_revocation_payload(*args, **kwargs),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# --- Creation ---------------------------------------------------------------


def test_create_revocation_returns_full_public_fields(client):
    bundle = _setup_bundle(client)
    body = _revoke(client, bundle["id"])

    assert set(body) == {
        "id",
        "evidence_bundle_id",
        "revoker_actor_id",
        "reason",
        "created_at",
    }
    assert body["id"].startswith("ebr_")
    assert len(body["id"]) == len("ebr_") + 64
    assert body["evidence_bundle_id"] == bundle["id"]
    assert body["revoker_actor_id"] == "org-1"
    assert body["reason"] == REASON_A
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))


def test_revocation_id_is_stable_and_deterministic(client):
    bundle = _setup_bundle(client)
    created = _revoke(client, bundle["id"])
    expected_material = json.dumps(
        ["evidence_bundle_revocation", bundle["id"], "org-1", REASON_A],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert created["id"] == "ebr_" + hashlib.sha256(expected_material).hexdigest()


def test_reason_surrounding_whitespace_is_trimmed(client):
    bundle = _setup_bundle(client)
    resp = client.post(
        "/v1/evidence-bundle-revocations",
        json=_revocation_payload(bundle["id"], reason=f"   {REASON_A}\t "),
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["reason"] == REASON_A


def test_revoker_need_not_be_the_claim_actor(client):
    bundle = _setup_bundle(client)
    create_actor(client, actor_id="org-2", name="Reviewer Org", type="organization")
    body = _revoke(client, bundle["id"], revoker_actor_id="org-2", reason=REASON_B)
    assert body["revoker_actor_id"] == "org-2"
    assert body["reason"] == REASON_B


def test_get_revocation_returns_the_created_record(client):
    bundle = _setup_bundle(client)
    created = _revoke(client, bundle["id"])
    resp = client.get(f"/v1/evidence-bundle-revocations/{created['id']}")
    assert resp.status_code == 200
    assert resp.json() == created


# --- Idempotency and distinct combinations ----------------------------------


def test_repeat_submission_is_idempotent_200_and_adds_no_record_or_audit(
    client, db_session
):
    bundle = _setup_bundle(client)
    first = _revoke(client, bundle["id"])
    events_after_create = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    for _ in range(3):
        repeat = client.post(
            "/v1/evidence-bundle-revocations",
            json=_revocation_payload(bundle["id"]),
        )
        assert repeat.status_code == 200
        assert repeat.json() == first

    rows = db_session.execute(select(EvidenceBundleRevocation)).scalars().all()
    assert [r.id for r in rows] == [first["id"]]
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_EVIDENCE_BUNDLE_REVOKED
        )
    ).scalars().all()
    assert len(events) == 1
    assert events[0].resource_id == first["id"]
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_after_create
    )


def test_retry_with_whitespace_padded_reason_matches_trimmed_record(client):
    bundle = _setup_bundle(client)
    first = _revoke(client, bundle["id"], reason=REASON_A)
    repeat = client.post(
        "/v1/evidence-bundle-revocations",
        json=_revocation_payload(bundle["id"], reason=f"  {REASON_A}  "),
    )
    assert repeat.status_code == 200
    assert repeat.json()["id"] == first["id"]


def test_different_revoker_and_different_reason_form_independent_records(
    client, db_session
):
    bundle = _setup_bundle(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")

    first = _revoke(client, bundle["id"], revoker_actor_id="org-1", reason=REASON_A)
    # Same bundle and revoker, different reason: independent record.
    second = _revoke(client, bundle["id"], revoker_actor_id="org-1", reason=REASON_B)
    # Same reason, different revoker: independent record.
    third = _revoke(client, bundle["id"], revoker_actor_id="org-2", reason=REASON_A)

    ids = {first["id"], second["id"], third["id"]}
    assert len(ids) == 3
    rows = db_session.execute(select(EvidenceBundleRevocation)).scalars().all()
    assert len(rows) == 3


# --- Per-bundle listing, ordering, and isolation ----------------------------


def test_list_for_bundle_returns_its_records_in_creation_order(client):
    bundle = _setup_bundle(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    first = _revoke(client, bundle["id"], revoker_actor_id="org-1", reason=REASON_A)
    second = _revoke(client, bundle["id"], revoker_actor_id="org-2", reason=REASON_B)

    resp = client.get(f"/v1/evidence-bundles/{bundle['id']}/revocations")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [first["id"], second["id"]]
    assert body["items"][0] == first
    assert body["items"][1] == second


def test_listing_is_isolated_per_bundle_and_empty_for_none(client):
    create_actor(client)
    claim = _create_claim(client)
    bundle_one = _create_bundle(client, claim["id"], digest=EVIDENCE_DIGEST_A)
    bundle_two = _create_bundle(client, claim["id"], digest=EVIDENCE_DIGEST_B)

    revocation = _revoke(client, bundle_one["id"])

    only_one = client.get(
        f"/v1/evidence-bundles/{bundle_one['id']}/revocations"
    ).json()
    assert [item["id"] for item in only_one["items"]] == [revocation["id"]]
    assert only_one["count"] == 1

    # The other bundle has no revocations: an empty collection, not 404.
    none = client.get(f"/v1/evidence-bundles/{bundle_two['id']}/revocations")
    assert none.status_code == 200
    assert none.json() == {"items": [], "count": 0}


def test_revocation_preserves_the_bundle_unchanged(client):
    bundle = _setup_bundle(client)
    _revoke(client, bundle["id"])

    # The original bundle is still fetchable and listed, byte-for-byte.
    assert client.get(f"/v1/evidence-bundles/{bundle['id']}").json() == bundle
    listed = client.get(
        "/v1/evidence-bundles", params={"claim_id": bundle["claim_id"]}
    ).json()
    assert [item["id"] for item in listed["items"]] == [bundle["id"]]


def test_revocation_does_not_change_evidence_coverage(client):
    bundle = _setup_bundle(client)
    content_id = client.get(f"/v1/claims/{bundle['claim_id']}").json()[
        "content_id"
    ]
    before = client.get(f"/v1/contents/{content_id}/evidence-coverage").json()

    _revoke(client, bundle["id"])

    # A bundle revocation is a statement about the bundle, not a proof
    # revocation: the coverage summary is computed exactly as before.
    after = client.get(f"/v1/contents/{content_id}/evidence-coverage").json()
    assert after == before


def test_revocation_does_not_change_content_export(client):
    bundle = _setup_bundle(client)
    content_id = client.get(f"/v1/claims/{bundle['claim_id']}").json()[
        "content_id"
    ]
    before = client.get(f"/v1/contents/{content_id}/export").json()

    _revoke(client, bundle["id"])

    assert client.get(f"/v1/contents/{content_id}/export").json() == before


# --- Missing-resource boundary ----------------------------------------------


def test_create_revocation_unknown_bundle_is_404(client):
    create_actor(client)
    resp = client.post(
        "/v1/evidence-bundle-revocations",
        json=_revocation_payload("evb_ghost"),
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "evidence_bundle_not_found"
    assert error["details"]["evidence_bundle_id"] == "evb_ghost"


def test_create_revocation_unknown_revoker_is_404(client):
    bundle = _setup_bundle(client)
    resp = client.post(
        "/v1/evidence-bundle-revocations",
        json=_revocation_payload(bundle["id"], revoker_actor_id="ghost"),
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "unknown_actor"
    assert resp.json()["error"]["details"]["actor_id"] == "ghost"


def test_get_unknown_revocation_is_404(client):
    resp = client.get("/v1/evidence-bundle-revocations/ebr_does_not_exist")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "evidence_bundle_revocation_not_found"
    assert error["details"]["revocation_id"] == "ebr_does_not_exist"


def test_list_for_unknown_bundle_is_404(client):
    resp = client.get("/v1/evidence-bundles/evb_ghost/revocations")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "evidence_bundle_not_found"
    assert error["details"]["evidence_bundle_id"] == "evb_ghost"


# --- Validation boundary -----------------------------------------------------


def test_reject_missing_required_fields(client):
    resp = client.post("/v1/evidence-bundle-revocations", json={})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    issue_fields = {
        ".".join(part for part in issue["loc"] if part != "body")
        for issue in resp.json()["error"]["details"]["issues"]
    }
    assert {"evidence_bundle_id", "revoker_actor_id", "reason"}.issubset(
        issue_fields
    )


def test_reject_blank_fields(client):
    bundle = _setup_bundle(client)
    for field in ("evidence_bundle_id", "revoker_actor_id", "reason"):
        payload = _revocation_payload(bundle["id"])
        payload[field] = "   "
        resp = client.post("/v1/evidence-bundle-revocations", json=payload)
        assert resp.status_code == 422, field
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_wrong_field_types(client):
    bundle = _setup_bundle(client)
    for field, value in (
        ("evidence_bundle_id", 42),
        ("revoker_actor_id", ["org-1"]),
        ("reason", {"text": REASON_A}),
    ):
        payload = _revocation_payload(bundle["id"])
        payload[field] = value
        resp = client.post("/v1/evidence-bundle-revocations", json=payload)
        assert resp.status_code == 422, field
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_undeclared_fields(client):
    bundle = _setup_bundle(client)
    payload = _revocation_payload(bundle["id"])
    payload["unexpected"] = "value"
    resp = client.post("/v1/evidence-bundle-revocations", json=payload)
    assert resp.status_code == 422
    issues = resp.json()["error"]["details"]["issues"]
    assert any("unexpected" in issue["loc"] for issue in issues)


def test_reject_duplicate_fields(client, db_session):
    bundle = _setup_bundle(client)
    raw = json.dumps(_revocation_payload(bundle["id"]))
    # Rebuild the document with the reason member repeated.
    duplicated = raw[:-1] + f',"reason":"{REASON_B}"' + "}"
    resp = client.post(
        "/v1/evidence-bundle-revocations",
        content=duplicated.encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert (
        db_session.scalar(select(func.count()).select_from(EvidenceBundleRevocation))
        == 0
    )


def test_reject_malformed_and_non_object_json_body(client):
    for content in (b"{not valid json", b"[1, 2]", b'"just a string"', b""):
        resp = client.post(
            "/v1/evidence-bundle-revocations",
            content=content,
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422, content
        assert resp.json()["error"]["code"] == "validation_error"


def test_validation_failures_write_no_records_or_audit(client, db_session):
    bundle = _setup_bundle(client)
    revocations_before = db_session.scalar(
        select(func.count()).select_from(EvidenceBundleRevocation)
    )
    audit_before = db_session.scalar(select(func.count()).select_from(AuditEvent))

    attempts = [
        client.post("/v1/evidence-bundle-revocations", json={}),
        client.post(
            "/v1/evidence-bundle-revocations",
            json=_revocation_payload(bundle["id"], reason="   "),
        ),
        client.post(
            "/v1/evidence-bundle-revocations",
            content=b"{bad json",
            headers={"content-type": "application/json"},
        ),
    ]
    assert [r.status_code for r in attempts] == [422, 422, 422]
    assert (
        db_session.scalar(select(func.count()).select_from(EvidenceBundleRevocation))
        == revocations_before
    )
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audit_before
    )


# --- GET request validation ---------------------------------------------------


def test_get_endpoints_reject_any_query_parameter(client):
    bundle = _setup_bundle(client)
    created = _revoke(client, bundle["id"])
    for url in (
        f"/v1/evidence-bundle-revocations/{created['id']}",
        f"/v1/evidence-bundles/{bundle['id']}/revocations",
    ):
        for query in ("foo=bar", "foo=bar&foo=baz", "limit=1"):
            resp = client.get(f"{url}?{query}")
            assert resp.status_code == 422, (url, query)
            assert resp.json()["error"]["code"] == "validation_error"


def test_get_endpoints_reject_a_non_empty_body(client):
    bundle = _setup_bundle(client)
    created = _revoke(client, bundle["id"])
    for url in (
        f"/v1/evidence-bundle-revocations/{created['id']}",
        f"/v1/evidence-bundles/{bundle['id']}/revocations",
    ):
        for content in (b"   ", b"{}", b"{bad json"):
            resp = client.request(
                "GET",
                url,
                content=content,
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 422, (url, content)
            assert resp.json()["error"]["code"] == "validation_error"


def test_get_validation_runs_before_the_404_lookup(client):
    # An unknown id with an illegal query parameter is a 422, not a 404:
    # validation completes before any record is read.
    resp = client.get("/v1/evidence-bundle-revocations/ebr_ghost?foo=bar")
    assert resp.status_code == 422
    resp = client.get("/v1/evidence-bundles/evb_ghost/revocations?foo=bar")
    assert resp.status_code == 422


# --- Method boundary ----------------------------------------------------------


def test_unsupported_methods_are_405(client):
    bundle = _setup_bundle(client)
    created = _revoke(client, bundle["id"])
    urls = [
        "/v1/evidence-bundle-revocations",
        f"/v1/evidence-bundle-revocations/{created['id']}",
        f"/v1/evidence-bundles/{bundle['id']}/revocations",
    ]
    for url in urls:
        for method, kwargs in (
            ("put", {"json": _revocation_payload(bundle["id"])}),
            ("patch", {"json": {"reason": "changed"}}),
            ("delete", {}),
        ):
            resp = getattr(client, method)(url, **kwargs)
            assert resp.status_code == 405, (method, url)
            assert resp.json()["error"]["code"] == "method_not_allowed"

    # The record is unchanged and remains fetchable.
    assert client.get(urls[1]).json() == created


# --- Transactional guarantees -------------------------------------------------


def test_revocation_and_audit_event_commit_atomically(client, db_session):
    bundle = _setup_bundle(client)
    created = _revoke(client, bundle["id"])

    rows = db_session.execute(select(EvidenceBundleRevocation)).scalars().all()
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_EVIDENCE_BUNDLE_REVOKED
        )
    ).scalars().all()
    assert [r.id for r in rows] == [created["id"]]
    assert [e.resource_id for e in events] == [created["id"]]
    assert events[0].created_at.utcoffset().total_seconds() == 0


def test_concurrent_identical_revocations_yield_one_record_and_audit(
    tmp_db_url, file_app, file_client
):
    bundle = _setup_bundle(file_client)

    from provenance.schemas import EvidenceBundleRevocationCreate
    from provenance import service

    factory = file_app.state.session_factory
    request = EvidenceBundleRevocationCreate(
        **_revocation_payload(bundle["id"])
    )
    results: list[tuple[str, bool]] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        session = factory()
        try:
            barrier.wait()
            record, created = service.create_evidence_bundle_revocation(
                session, request
            )
            results.append((record.id, created))
        except Exception as exc:  # pragma: no cover - fails the test below
            errors.append(exc)
        finally:
            session.close()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(results) == 4
    assert {record_id for record_id, _ in results} == {results[0][0]}
    assert sum(1 for _, created in results if created) == 1

    audit_session = factory()
    try:
        rows = audit_session.execute(
            select(EvidenceBundleRevocation)
        ).scalars().all()
        events = audit_session.execute(
            select(AuditEvent).where(
                AuditEvent.event_type == EVENT_EVIDENCE_BUNDLE_REVOKED
            )
        ).scalars().all()
        assert [r.id for r in rows] == [results[0][0]]
        assert [e.resource_id for e in events] == [results[0][0]]
    finally:
        audit_session.close()


# --- Persistence across restarts ----------------------------------------------


def test_revocations_persist_across_app_restarts(tmp_db_url):
    from fastapi.testclient import TestClient

    app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app) as client:
        bundle = _setup_bundle(client)
        create_actor(
            client, actor_id="org-2", name="Other Org", type="organization"
        )
        first = _revoke(client, bundle["id"], reason=REASON_A)
        second = _revoke(
            client, bundle["id"], revoker_actor_id="org-2", reason=REASON_B
        )

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        # Identifiers, UTC timestamps, and creation order survive the restart.
        assert (
            client.get(f"/v1/evidence-bundle-revocations/{first['id']}").json()
            == first
        )
        assert (
            client.get(f"/v1/evidence-bundle-revocations/{second['id']}").json()
            == second
        )
        listed = client.get(
            f"/v1/evidence-bundles/{bundle['id']}/revocations"
        ).json()
        assert [item["id"] for item in listed["items"]] == [
            first["id"],
            second["id"],
        ]
        assert listed["count"] == 2

        # A retry after the restart is the idempotent 200, not a new record.
        retry = client.post(
            "/v1/evidence-bundle-revocations",
            json=_revocation_payload(bundle["id"]),
        )
        assert retry.status_code == 200
        assert retry.json() == first


# --- Legacy schema upgrade -----------------------------------------------------


def _downgrade_to_v3(db_path):
    """Shape a current database as a legacy version-3 database."""
    con = sqlite3.connect(db_path)
    try:
        con.execute("DROP TABLE evidence_bundle_revocations")
        con.execute("DELETE FROM schema_migrations WHERE version > 3")
        con.commit()
    finally:
        con.close()


def _snapshot(con, table):
    return sorted(con.execute(f"SELECT * FROM {table}").fetchall())


def test_legacy_v3_upgrade_preserves_bundles_and_audits(tmp_path):
    from fastapi.testclient import TestClient

    db_path = tmp_path / "legacy.db"
    url = f"sqlite:///{db_path.as_posix()}"

    app = create_app(Settings(database_url=url))
    with TestClient(app) as client:
        bundle = _setup_bundle(client)
    _downgrade_to_v3(db_path)

    con = sqlite3.connect(db_path)
    try:
        bundles_before = _snapshot(con, "evidence_bundles")
        audits_before = _snapshot(con, "audit_events")
        assert [
            row[0] for row in con.execute("SELECT version FROM schema_migrations")
        ] == [1, 2, 3]
    finally:
        con.close()

    # The upgrade adds only the revocation table; existing rows are untouched.
    upgraded = create_app(Settings(database_url=url))
    with TestClient(upgraded) as client:
        con = sqlite3.connect(db_path)
        try:
            assert _snapshot(con, "evidence_bundles") == bundles_before
            assert _snapshot(con, "audit_events") == audits_before
            assert [
                row[0]
                for row in con.execute("SELECT version FROM schema_migrations")
            ] == [1, 2, 3, 4, 5]
            assert "evidence_bundle_revocations" in {
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        finally:
            con.close()

        # The upgraded database serves revocations consistently.
        created = _revoke(client, bundle["id"], reason="post-upgrade")
        assert created["evidence_bundle_id"] == bundle["id"]
        assert (
            client.get(f"/v1/evidence-bundle-revocations/{created['id']}").json()
            == created
        )


def test_failed_v4_migration_rolls_back_and_keeps_old_data(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    db_path = tmp_path / "fail.db"
    url = f"sqlite:///{db_path.as_posix()}"

    app = create_app(Settings(database_url=url))
    with TestClient(app) as client:
        _setup_bundle(client)
    _downgrade_to_v3(db_path)

    con = sqlite3.connect(db_path)
    try:
        bundles_before = _snapshot(con, "evidence_bundles")
        audits_before = _snapshot(con, "audit_events")
    finally:
        con.close()

    def _failing_up(cursor, engine, metadata):
        cursor.execute("CREATE TABLE evidence_bundle_revocations (seq INTEGER)")
        raise RuntimeError("simulated migration failure")

    monkeypatch.setattr(
        migrations,
        "MIGRATIONS",
        migrations.MIGRATIONS[:3]
        + (migrations.Migration(version=4, up=_failing_up),),
    )
    engine = make_engine(url)
    with pytest.raises(RuntimeError, match="simulated migration failure"):
        migrations.run_migrations(engine)

    con = sqlite3.connect(db_path)
    try:
        # No half-finished version row and no partial table; the old data is
        # byte-for-byte intact.
        assert [
            row[0] for row in con.execute("SELECT version FROM schema_migrations")
        ] == [1, 2, 3]
        assert "evidence_bundle_revocations" not in {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert _snapshot(con, "evidence_bundles") == bundles_before
        assert _snapshot(con, "audit_events") == audits_before
    finally:
        con.close()
