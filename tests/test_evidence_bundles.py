"""Tests for verifiable evidence bundles.

Covers POST/GET /v1/evidence-bundles and the per-claim list, including
normal creation, idempotent dedup (first metadata retained, no extra
audit), claim isolation and stable ordering, single-transaction audit
commit, raw-byte exclusion, and the missing-vs-validation error boundary.
All tests are deterministic and offline.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import select

from provenance.app import create_app
from provenance.config import Settings
from provenance.models import (
    EVENT_EVIDENCE_BUNDLE_CREATED,
    AuditEvent,
    EvidenceBundle,
)
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    content_payload,
    create_actor,
)

EVIDENCE_DIGEST_1 = hashlib.sha256(b"evidence-a").hexdigest()
EVIDENCE_DIGEST_2 = hashlib.sha256(b"evidence-b").hexdigest()
EVIDENCE_DIGEST_3 = hashlib.sha256(b"evidence-c").hexdigest()

METADATA_1 = {"source": "camera-1", "captured_at": "2026-01-01T00:00:00Z"}
METADATA_2 = {"source": "scanner-2", "pages": 3}


def _create_content(client, digest=DIGEST_A, actor_id="org-1"):
    resp = client.post(
        "/v1/contents", json=content_payload(actor_id=actor_id, digest=digest)
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(
    client,
    content_id,
    actor_id="org-1",
    claim_type="authorship",
    payload=None,
):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": claim_type,
            "payload": payload if payload is not None else {"statement": "x"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _setup_claim(client, digest=DIGEST_A):
    create_actor(client)
    content = _create_content(client, digest=digest)
    return _create_claim(client, content_id=content["id"])


_UNSET = object()


def _bundle_payload(
    claim_id,
    evidence_type="raw_capture",
    digest=EVIDENCE_DIGEST_1,
    algorithm="sha256",
    media_type="image/jpeg",
    metadata=_UNSET,
):
    return {
        "claim_id": claim_id,
        "evidence_type": evidence_type,
        "digest_algorithm": algorithm,
        "digest_hex": digest,
        "media_type": media_type,
        "metadata": METADATA_1 if metadata is _UNSET else metadata,
    }


def _create_bundle(client, **overrides):
    resp = client.post("/v1/evidence-bundles", json=_bundle_payload(**overrides))
    assert resp.status_code == 201, resp.text
    return resp.json()


# --- Normal creation -------------------------------------------------------


def test_create_bundle_returns_full_public_fields(client):
    claim = _setup_claim(client)
    body = _create_bundle(
        client, claim_id=claim["id"], metadata=METADATA_1
    )
    assert set(body) == {
        "id",
        "claim_id",
        "evidence_type",
        "digest_algorithm",
        "digest_hex",
        "media_type",
        "metadata",
        "created_at",
    }
    assert body["id"].startswith("evb_")
    assert body["claim_id"] == claim["id"]
    assert body["evidence_type"] == "raw_capture"
    assert body["digest_algorithm"] == "sha256"
    assert body["digest_hex"] == EVIDENCE_DIGEST_1
    assert body["media_type"] == "image/jpeg"
    assert body["metadata"] == METADATA_1
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))


def test_metadata_round_trips_nested_json(client):
    claim = _setup_claim(client)
    rich = {
        "nested": {"a": [1, 2, {"b": True}], "n": None},
        "unicode": "证据",
        "float": 1.5,
    }
    body = _create_bundle(client, claim_id=claim["id"], metadata=rich)
    assert body["metadata"] == rich


def test_bundle_id_is_deterministic_and_stable(client):
    claim = _setup_claim(client)
    first = _create_bundle(client, claim_id=claim["id"])
    repeat = client.post(
        "/v1/evidence-bundles", json=_bundle_payload(claim_id=claim["id"])
    )
    assert repeat.status_code == 200
    assert repeat.json()["id"] == first["id"]


def test_get_bundle_returns_full_fields(client):
    claim = _setup_claim(client)
    created = _create_bundle(client, claim_id=claim["id"])
    resp = client.get(f"/v1/evidence-bundles/{created['id']}")
    assert resp.status_code == 200
    assert resp.json() == created


def test_distinct_identities_create_distinct_bundles(client):
    claim = _setup_claim(client)
    base = _create_bundle(client, claim_id=claim["id"])
    by_type = _create_bundle(
        client, claim_id=claim["id"], evidence_type="signature"
    )
    by_digest = _create_bundle(
        client, claim_id=claim["id"], digest=EVIDENCE_DIGEST_2
    )
    ids = {base["id"], by_type["id"], by_digest["id"]}
    assert len(ids) == 3


def test_same_type_and_digest_on_another_claim_is_independent(client):
    claim_one = _setup_claim(client, digest=DIGEST_A)
    create_actor(client, actor_id="org-2", name="Other", type="person")
    content_two = _create_content(
        client, digest=DIGEST_B, actor_id="org-2"
    )
    claim_two = _create_claim(
        client, content_id=content_two["id"], actor_id="org-2"
    )

    first = _create_bundle(client, claim_id=claim_one["id"])
    second = _create_bundle(client, claim_id=claim_two["id"])
    assert first["id"] != second["id"]
    assert first["claim_id"] == claim_one["id"]
    assert second["claim_id"] == claim_two["id"]
    assert first["digest_hex"] == second["digest_hex"]


# --- Idempotency ------------------------------------------------------------


def test_repeat_returns_existing_retains_first_metadata_and_media_type(
    client, db_session
):
    claim = _setup_claim(client)
    first = _create_bundle(
        client, claim_id=claim["id"], media_type="image/jpeg",
        metadata=METADATA_1,
    )
    events_after_create = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    # Same claim + type + digest, but different media type and metadata: the
    # existing resource must win, unchanged.
    for _ in range(3):
        repeat = client.post(
            "/v1/evidence-bundles",
            json=_bundle_payload(
                claim_id=claim["id"],
                media_type="application/pdf",
                metadata=METADATA_2,
            ),
        )
        assert repeat.status_code == 200
        assert repeat.json() == first
        assert repeat.json()["metadata"] == METADATA_1
        assert repeat.json()["media_type"] == "image/jpeg"

    events = db_session.execute(
        select(AuditEvent).order_by(AuditEvent.seq.asc())
    ).scalars().all()
    assert len(events) == events_after_create
    bundle_events = [
        e for e in events if e.event_type == EVENT_EVIDENCE_BUNDLE_CREATED
    ]
    assert len(bundle_events) == 1
    assert bundle_events[0].resource_id == first["id"]


def test_duplicate_with_uppercase_hex_is_idempotent(client):
    claim = _setup_claim(client)
    first = _create_bundle(client, claim_id=claim["id"])
    resp = client.post(
        "/v1/evidence-bundles",
        json=_bundle_payload(
            claim_id=claim["id"], digest=EVIDENCE_DIGEST_1.upper()
        ),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == first["id"]
    # The stored digest keeps the normalized lowercase form.
    assert body["digest_hex"] == EVIDENCE_DIGEST_1


def test_existing_bundle_is_never_updated(client):
    claim = _setup_claim(client)
    first = _create_bundle(client, claim_id=claim["id"], metadata=METADATA_1)
    other = _create_bundle(
        client, claim_id=claim["id"], digest=EVIDENCE_DIGEST_2,
        metadata=METADATA_2,
    )
    again = client.get(f"/v1/evidence-bundles/{first['id']}")
    assert again.json() == first
    assert again.json()["digest_hex"] != other["digest_hex"]


# --- Isolation and ordering -------------------------------------------------


def test_list_bundles_is_isolated_and_stably_ordered(client):
    claim_one = _setup_claim(client, digest=DIGEST_A)
    content_two = _create_content(client, digest=DIGEST_B)
    claim_two = _create_claim(client, content_id=content_two["id"])

    a1 = _create_bundle(client, claim_id=claim_one["id"])
    b1 = _create_bundle(
        client, claim_id=claim_two["id"], evidence_type="other"
    )
    a2 = _create_bundle(
        client, claim_id=claim_one["id"], digest=EVIDENCE_DIGEST_2
    )
    a3 = _create_bundle(
        client, claim_id=claim_one["id"], evidence_type="third",
        digest=EVIDENCE_DIGEST_3,
    )

    resp = client.get(f"/v1/claims/{claim_one['id']}/evidence-bundles")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    assert [item["id"] for item in body["items"]] == [
        a1["id"], a2["id"], a3["id"]
    ]
    assert all(
        item["claim_id"] == claim_one["id"] for item in body["items"]
    )

    resp_b = client.get(f"/v1/claims/{claim_two['id']}/evidence-bundles")
    assert resp_b.json()["count"] == 1
    assert [item["id"] for item in resp_b.json()["items"]] == [b1["id"]]


def test_list_bundles_empty_for_claim_without_bundles(client):
    claim = _setup_claim(client)
    resp = client.get(f"/v1/claims/{claim['id']}/evidence-bundles")
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 0}


# --- Missing-resource boundary (distinct from validation) -------------------


def test_get_unknown_bundle_is_distinct_not_found(client):
    resp = client.get("/v1/evidence-bundles/evb_does_not_exist")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "evidence_bundle_not_found"
    assert error["details"]["evidence_bundle_id"] == "evb_does_not_exist"


def test_create_bundle_unknown_claim_is_not_found(client):
    create_actor(client)
    resp = client.post(
        "/v1/evidence-bundles", json=_bundle_payload(claim_id="clm_ghost")
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "claim_not_found"
    assert error["details"]["claim_id"] == "clm_ghost"


def test_list_bundles_unknown_claim_is_not_found_not_empty(client):
    resp = client.get("/v1/claims/clm_ghost/evidence-bundles")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "claim_not_found"


# --- Validation boundary ----------------------------------------------------


def test_reject_blank_evidence_type(client):
    claim = _setup_claim(client)
    resp = client.post(
        "/v1/evidence-bundles",
        json=_bundle_payload(claim_id=claim["id"], evidence_type="   "),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_reject_blank_media_type(client):
    claim = _setup_claim(client)
    resp = client.post(
        "/v1/evidence-bundles",
        json=_bundle_payload(claim_id=claim["id"], media_type="  "),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_reject_blank_claim_id(client):
    _setup_claim(client)
    resp = client.post(
        "/v1/evidence-bundles", json=_bundle_payload(claim_id="  ")
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_reject_non_sha256_algorithm(client):
    claim = _setup_claim(client)
    resp = client.post(
        "/v1/evidence-bundles",
        json=_bundle_payload(claim_id=claim["id"], algorithm="sha512"),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_reject_short_long_and_non_hex_digests(client):
    claim = _setup_claim(client)
    for bad in ("a" * 63, "a" * 65, EVIDENCE_DIGEST_1[:-1] + "z"):
        resp = client.post(
            "/v1/evidence-bundles",
            json=_bundle_payload(claim_id=claim["id"], digest=bad),
        )
        assert resp.status_code == 422, bad
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_non_object_metadata(client):
    claim = _setup_claim(client)
    for bad in ([1, 2], "text", 42, 3.14, True, None):
        resp = client.post(
            "/v1/evidence-bundles",
            json=_bundle_payload(claim_id=claim["id"], metadata=bad),
        )
        assert resp.status_code == 422, bad
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_missing_required_fields(client):
    resp = client.post("/v1/evidence-bundles", json={})
    assert resp.status_code == 422
    issue_fields = {
        ".".join(part for part in issue["loc"] if part != "body")
        for issue in resp.json()["error"]["details"]["issues"]
    }
    assert {
        "claim_id",
        "evidence_type",
        "digest_algorithm",
        "digest_hex",
        "media_type",
        "metadata",
    }.issubset(issue_fields)


def test_reject_malformed_json_body(client):
    _setup_claim(client)
    resp = client.post(
        "/v1/evidence-bundles",
        content="{not valid json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Transactional and no-write guarantees ----------------------------------


def test_failed_requests_write_no_rows_or_audit_events(client, db_session):
    create_actor(client)
    content = _create_content(client)
    events_before = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    attempts = [
        client.post(
            "/v1/evidence-bundles", json=_bundle_payload(claim_id="clm_ghost")
        ),
        client.post(
            "/v1/evidence-bundles",
            json=_bundle_payload(claim_id=content["id"]),
        ),  # claim_id is a content id -> unknown claim
        client.post(
            "/v1/evidence-bundles",
            json=_bundle_payload(
                claim_id="clm_ghost", evidence_type=" "
            ),
        ),
        client.post(
            "/v1/evidence-bundles",
            json=_bundle_payload(claim_id="clm_ghost", metadata=[]),
        ),
    ]
    assert [r.status_code for r in attempts] == [404, 404, 422, 422]
    assert db_session.execute(select(EvidenceBundle)).scalars().all() == []
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )


def test_bundle_and_audit_event_commit_atomically(client, db_session):
    claim = _setup_claim(client)
    created = _create_bundle(client, claim_id=claim["id"])

    bundles = db_session.execute(select(EvidenceBundle)).scalars().all()
    events = db_session.execute(
        select(AuditEvent)
        .where(AuditEvent.event_type == EVENT_EVIDENCE_BUNDLE_CREATED)
        .order_by(AuditEvent.seq.asc())
    ).scalars().all()
    # Both the bundle row and its audit event are visible after one request:
    # they committed in the same transaction.
    assert [b.id for b in bundles] == [created["id"]]
    assert [e.resource_id for e in events] == [created["id"]]
    created_at = events[0].created_at
    assert created_at.tzinfo.utcoffset(created_at).total_seconds() == 0


def test_each_first_bundle_creation_writes_one_audit_event(client, db_session):
    claim = _setup_claim(client)
    _create_bundle(client, claim_id=claim["id"])
    _create_bundle(
        client, claim_id=claim["id"], digest=EVIDENCE_DIGEST_2
    )
    # Repeat: no new event.
    repeat = client.post(
        "/v1/evidence-bundles", json=_bundle_payload(claim_id=claim["id"])
    )
    assert repeat.status_code == 200

    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_EVIDENCE_BUNDLE_CREATED
        )
    ).scalars().all()
    assert len(events) == 2


# --- Raw evidence bytes never enter the service ------------------------------


def test_raw_evidence_bytes_are_never_persisted(client, db_session):
    claim = _setup_claim(client)
    secret_marker = "raw-evidence-byte-marker-9c2e"
    payload = _bundle_payload(claim_id=claim["id"], metadata={"k": "v"})
    # Extra, non-schema fields that might smuggle bytes must be ignored.
    payload["data"] = secret_marker
    payload["evidence"] = secret_marker.encode().hex()
    resp = client.post("/v1/evidence-bundles", json=payload)
    assert resp.status_code == 201, resp.text
    assert secret_marker not in resp.text

    row = db_session.execute(select(EvidenceBundle)).scalars().one()
    columns = [c.name for c in EvidenceBundle.__table__.columns]
    # There is no column capable of carrying the evidence itself.
    assert "data" not in columns and "evidence" not in columns
    persisted = " ".join(
        str(getattr(row, attr))
        for attr in (
            "id",
            "claim_id",
            "evidence_type",
            "digest_algorithm",
            "digest_hex",
            "media_type",
        )
    ) + " " + repr(row.metadata_)
    assert secret_marker not in persisted


# --- Concurrent race boundary ------------------------------------------------


def test_concurrent_identical_requests_yield_one_bundle_and_audit(
    tmp_db_url, file_app, file_client
):
    # Concurrent identical creates must collapse into one resource and one
    # audit event (one wins the unique constraint, the other is deduped),
    # never a 500 -- regardless of whether dedup happens before insert or
    # via the IntegrityError fallback.
    import threading

    create_actor(file_client)
    content = file_client.post(
        "/v1/contents", json=content_payload()
    ).json()
    claim = file_client.post(
        "/v1/claims",
        json={
            "content_id": content["id"],
            "actor_id": "org-1",
            "claim_type": "authorship",
            "payload": {"statement": "x"},
        },
    ).json()

    from provenance.schemas import EvidenceBundleCreate
    from provenance import service

    factory = file_app.state.session_factory
    request = EvidenceBundleCreate(
        **_bundle_payload(claim_id=claim["id"], metadata=METADATA_1)
    )
    results: list[tuple[str, bool]] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        session = factory()
        try:
            barrier.wait()
            bundle, created = service.create_evidence_bundle(
                session, request
            )
            results.append((bundle.id, created))
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
    assert {bundle_id for bundle_id, _ in results} == {results[0][0]}
    assert sum(1 for _, created in results if created) == 1

    audit_session = factory()
    try:
        rows = audit_session.execute(select(EvidenceBundle)).scalars().all()
        events = audit_session.execute(
            select(AuditEvent).where(
                AuditEvent.event_type == EVENT_EVIDENCE_BUNDLE_CREATED
            )
        ).scalars().all()
        assert [r.id for r in rows] == [results[0][0]]
        assert [e.resource_id for e in events] == [results[0][0]]
    finally:
        audit_session.close()


# --- Persistence across restarts ---------------------------------------------
def test_bundles_persist_across_app_restarts(tmp_db_url, file_client):
    create_actor(file_client)
    content = file_client.post(
        "/v1/contents", json=content_payload()
    ).json()
    claim = file_client.post(
        "/v1/claims",
        json={
            "content_id": content["id"],
            "actor_id": "org-1",
            "claim_type": "authorship",
            "payload": {"statement": "x"},
        },
    ).json()
    bundle = file_client.post(
        "/v1/evidence-bundles",
        json=_bundle_payload(claim_id=claim["id"], metadata=METADATA_1),
    ).json()

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        fetched = client.get(f"/v1/evidence-bundles/{bundle['id']}")
        assert fetched.status_code == 200
        assert fetched.json() == bundle

        listing = client.get(
            f"/v1/claims/{claim['id']}/evidence-bundles"
        ).json()
        assert listing["count"] == 1
        assert listing["items"][0]["id"] == bundle["id"]
        assert listing["items"][0]["metadata"] == METADATA_1

        # Audit history (actor, content, claim, bundle) survived the restart.
        import sqlite3

        path = tmp_db_url.removeprefix("sqlite:///")
        con = sqlite3.connect(path)
        audit_count = con.execute(
            "SELECT COUNT(*) FROM audit_events"
        ).fetchone()[0]
        bundle_audit = con.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = ?",
            (EVENT_EVIDENCE_BUNDLE_CREATED,),
        ).fetchone()[0]
        con.close()
        assert audit_count == 4
        assert bundle_audit == 1
