"""Tests for immutable evidence bundle revocation records.

Covers POST /v1/evidence-bundle-revocations, GET by id, and the per-bundle
list, including: first-creation 201 response fields and UTC timestamps,
reason trimming, idempotent 200 retries that add no record or audit,
independent records for different combinations, per-bundle isolation and
stable ordering, the 404 boundary (unknown bundle, revocation, revoker),
the 422 boundary (blank/missing/undeclared fields, malformed JSON, GET
query parameters and bodies) with no writes, append-only immutability
(no update/delete), single-transaction audit commit, a
concurrent-identical race, persistence across restarts, and the guarantee
that the revoked bundle itself and every existing read are unchanged. All
tests are deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime

from sqlalchemy import func, select

from provenance.models import (
    EVENT_EVIDENCE_BUNDLE_REVOKED,
    AuditEvent,
    EvidenceBundle,
    EvidenceBundleRevocation,
)
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    content_payload,
    create_actor,
)

REASON_A = "bundle superseded by a verified re-capture"
REASON_B = "reviewer no longer relies on this evidence"

EVIDENCE_DIGEST_A = hashlib.sha256(b"evidence-a").hexdigest()
EVIDENCE_DIGEST_B = hashlib.sha256(b"evidence-b").hexdigest()


# --- Setup helpers ----------------------------------------------------------


def _create_content(client, actor_id="org-1", digest=DIGEST_A):
    resp = client.post(
        "/v1/contents", json=content_payload(actor_id=actor_id, digest=digest)
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, actor_id="org-1", claim_type="authorship"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": claim_type,
            "payload": {"statement": "evidenced"},
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
    claim = _create_claim(client, _create_content(client)["id"])
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


def test_revoker_need_not_be_a_bundle_signer(client):
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


def test_listing_is_isolated_per_bundle_and_empty_for_none(client):
    bundle_one = _setup_bundle(client)
    claim_two = _create_claim(client, _create_content(client, digest=DIGEST_B)["id"])
    bundle_two = _create_bundle(client, claim_two["id"], digest=EVIDENCE_DIGEST_B)

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
    per_claim = client.get(
        f"/v1/claims/{bundle['claim_id']}/evidence-bundles"
    ).json()
    assert [item["id"] for item in per_claim["items"]] == [bundle["id"]]


def test_bundle_creation_idempotency_is_unaffected_by_revocation(client):
    bundle = _setup_bundle(client)
    _revoke(client, bundle["id"])

    # Re-submitting the same bundle identity is still the idempotent 200
    # with the original record; the revocation changed nothing about it.
    repeat = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": bundle["claim_id"],
            "evidence_type": "raw_capture",
            "digest_algorithm": "sha256",
            "digest_hex": EVIDENCE_DIGEST_A,
            "media_type": "image/jpeg",
            "metadata": {"source": "camera-1"},
        },
    )
    assert repeat.status_code == 200
    assert repeat.json() == bundle


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


def test_reject_wrong_typed_fields(client):
    bundle = _setup_bundle(client)
    for field in ("evidence_bundle_id", "revoker_actor_id", "reason"):
        payload = _revocation_payload(bundle["id"])
        payload[field] = 42
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


def test_reject_duplicate_json_fields(client, db_session):
    bundle = _setup_bundle(client)
    body = (
        '{"evidence_bundle_id": "%s", "revoker_actor_id": "org-1",'
        ' "reason": "first", "reason": "second"}' % bundle["id"]
    )
    resp = client.post(
        "/v1/evidence-bundle-revocations",
        content=body,
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    # The ambiguous document created nothing.
    assert (
        db_session.scalar(
            select(func.count()).select_from(EvidenceBundleRevocation)
        )
        == 0
    )


def test_reject_malformed_json_body(client):
    resp = client.post(
        "/v1/evidence-bundle-revocations",
        content="{not valid json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_get_endpoints_reject_query_params_and_bodies(client):
    bundle = _setup_bundle(client)
    created = _revoke(client, bundle["id"])
    urls = (
        f"/v1/evidence-bundle-revocations/{created['id']}",
        f"/v1/evidence-bundles/{bundle['id']}/revocations",
    )
    for url in urls:
        resp = client.get(url, params={"unexpected": "1"})
        assert resp.status_code == 422, url
        assert resp.json()["error"]["code"] == "validation_error"

        resp = client.get(f"{url}?reason=a&reason=b")
        assert resp.status_code == 422, url
        assert resp.json()["error"]["code"] == "validation_error"

        resp = client.request("GET", url, content=b"{}")
        assert resp.status_code == 422, url
        assert resp.json()["error"]["code"] == "validation_error"

        resp = client.request("GET", url, content=b"   ")
        assert resp.status_code == 422, url
        assert resp.json()["error"]["code"] == "validation_error"

    # The failures wrote nothing and the records are still served.
    assert (
        client.get(f"/v1/evidence-bundle-revocations/{created['id']}").json()
        == created
    )


def test_validation_failures_write_no_records_or_audit(client, db_session):
    create_actor(client)
    revocations_before = db_session.scalar(
        select(func.count()).select_from(EvidenceBundleRevocation)
    )
    audit_before = db_session.scalar(select(func.count()).select_from(AuditEvent))

    attempts = [
        client.post("/v1/evidence-bundle-revocations", json={}),
        client.post(
            "/v1/evidence-bundle-revocations",
            json=_revocation_payload("evb_ghost"),
        ),  # 404, also writes nothing
        client.post(
            "/v1/evidence-bundle-revocations",
            content="{bad json",
            headers={"content-type": "application/json"},
        ),
    ]
    assert [r.status_code for r in attempts] == [422, 404, 422]
    assert (
        db_session.scalar(
            select(func.count()).select_from(EvidenceBundleRevocation)
        )
        == revocations_before
    )
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audit_before
    )


# --- Append-only immutability ------------------------------------------------


def test_revocation_has_no_update_or_delete_path(client):
    bundle = _setup_bundle(client)
    created = _revoke(client, bundle["id"])
    urls = (
        f"/v1/evidence-bundle-revocations/{created['id']}",
        f"/v1/evidence-bundles/{bundle['id']}/revocations",
    )
    for url in urls:
        for method, kwargs in (
            ("put", {"json": {"reason": "changed"}}),
            ("patch", {"json": {"reason": "changed"}}),
            ("delete", {}),
        ):
            resp = getattr(client, method)(url, **kwargs)
            assert resp.status_code == 405, (url, method)
            assert resp.json()["error"]["code"] == "method_not_allowed"

    # The record is unchanged and remains fetchable.
    assert (
        client.get(f"/v1/evidence-bundle-revocations/{created['id']}").json()
        == created
    )


def test_re_post_with_a_new_reason_does_not_overwrite(client):
    bundle = _setup_bundle(client)
    first = _revoke(client, bundle["id"], reason=REASON_A)
    second = _revoke(client, bundle["id"], reason=REASON_B)

    assert first["id"] != second["id"]
    # The first record keeps its original reason.
    assert client.get(
        f"/v1/evidence-bundle-revocations/{first['id']}"
    ).json()["reason"] == REASON_A
    listing = client.get(
        f"/v1/evidence-bundles/{bundle['id']}/revocations"
    ).json()
    assert [item["reason"] for item in listing["items"]] == [REASON_A, REASON_B]


# --- Transactional guarantees ------------------------------------------------


def test_revocation_and_audit_event_commit_atomically(client, db_session):
    bundle = _setup_bundle(client)
    created = _revoke(client, bundle["id"])

    rows = db_session.execute(select(EvidenceBundleRevocation)).scalars().all()
    events = db_session.execute(
        select(AuditEvent)
        .where(AuditEvent.event_type == EVENT_EVIDENCE_BUNDLE_REVOKED)
        .order_by(AuditEvent.seq.asc())
    ).scalars().all()
    assert [r.id for r in rows] == [created["id"]]
    assert [e.resource_id for e in events] == [created["id"]]
    assert events[0].created_at.tzinfo.utcoffset(
        events[0].created_at
    ).total_seconds() == 0


def test_revocation_never_stores_evidence_bytes(client, db_session):
    # The revocation table carries no evidence material of any kind.
    columns = {c.name for c in EvidenceBundleRevocation.__table__.columns}
    assert columns == {
        "seq",
        "id",
        "evidence_bundle_id",
        "revoker_actor_id",
        "reason",
        "created_at",
    }
    assert "digest_hex" not in columns
    assert "metadata" not in columns


def test_read_endpoints_write_nothing(client, db_session):
    bundle = _setup_bundle(client)
    created = _revoke(client, bundle["id"])

    revocations_before = db_session.scalar(
        select(func.count()).select_from(EvidenceBundleRevocation)
    )
    bundles_before = db_session.scalar(
        select(func.count()).select_from(EvidenceBundle)
    )
    audit_before = db_session.scalar(select(func.count()).select_from(AuditEvent))

    client.get(f"/v1/evidence-bundle-revocations/{created['id']}")
    client.get(f"/v1/evidence-bundles/{bundle['id']}/revocations")
    client.get("/v1/evidence-bundle-revocations/ebr_ghost")  # 404, read-only
    client.get("/v1/evidence-bundles/evb_ghost/revocations")  # 404, read-only

    assert (
        db_session.scalar(
            select(func.count()).select_from(EvidenceBundleRevocation)
        )
        == revocations_before
    )
    assert (
        db_session.scalar(select(func.count()).select_from(EvidenceBundle))
        == bundles_before
    )
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audit_before
    )


# --- Concurrent race boundary ------------------------------------------------


def test_concurrent_identical_revocation_requests_yield_one_record_and_audit(
    tmp_db_url, file_app, file_client
):
    bundle = _setup_bundle(file_client)

    from provenance.schemas import EvidenceBundleRevocationCreate
    from provenance import service

    factory = file_app.state.session_factory
    request = EvidenceBundleRevocationCreate(**_revocation_payload(bundle["id"]))
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


# --- Persistence across restarts ---------------------------------------------


def test_revocations_persist_across_app_restarts(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    bundle = _setup_bundle(file_client)
    revocation = file_client.post(
        "/v1/evidence-bundle-revocations",
        json=_revocation_payload(bundle["id"]),
    ).json()

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        fetched = client.get(
            f"/v1/evidence-bundle-revocations/{revocation['id']}"
        )
        assert fetched.status_code == 200
        assert fetched.json() == revocation

        listing = client.get(
            f"/v1/evidence-bundles/{bundle['id']}/revocations"
        ).json()
        assert listing["count"] == 1
        assert [i["id"] for i in listing["items"]] == [revocation["id"]]

        # The bundle itself is untouched by the revocation and the restart.
        assert client.get(f"/v1/evidence-bundles/{bundle['id']}").json() == bundle

        import sqlite3

        path = tmp_db_url.removeprefix("sqlite:///")
        con = sqlite3.connect(path)
        audit_count = con.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = ?",
            (EVENT_EVIDENCE_BUNDLE_REVOKED,),
        ).fetchone()[0]
        table_cols = {
            row[1]
            for row in con.execute(
                "PRAGMA table_info(evidence_bundle_revocations)"
            )
        }
        con.close()
        assert audit_count == 1
        assert table_cols == {
            "seq",
            "id",
            "evidence_bundle_id",
            "revoker_actor_id",
            "reason",
            "created_at",
        }
