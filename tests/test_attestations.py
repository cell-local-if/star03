"""Tests for verifiable attestations.

Covers POST/GET /v1/attestations and the filtered list, including Ed25519
signature verification over the canonical UTF-8 message, idempotent dedup,
single-transaction audit commit, raw-signature exclusion, stable ordering,
and the missing-vs-validation error boundary. All tests are deterministic
and offline: keys are derived from fixed seeds and signatures are produced
by the same RFC 8032 implementation the service verifies with.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import select

from provenance import ed25519
from provenance.app import create_app
from provenance.canonical import attestation_message_bytes
from provenance.config import Settings
from provenance.models import (
    EVENT_ATTESTATION_CREATED,
    Attestation,
    AuditEvent,
)
from tests.helpers import DIGEST_A, DIGEST_B, content_payload, create_actor

SEED_1 = bytes.fromhex(
    "9d61b19deffebc3a9ba12a4f14309e05a501b1c5c24cfcf0b8d8a90e5b0e9f8d"
)
SEED_2 = hashlib.sha256(b"attestation-seed-2").digest()
SEED_3 = hashlib.sha256(b"attestation-seed-3").digest()

PUBLIC_KEY_1 = ed25519.public_key_from_seed(SEED_1)
PUBLIC_KEY_2 = ed25519.public_key_from_seed(SEED_2)


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _signed_payload(
    target_type,
    target_id,
    signer_actor_id,
    seed=SEED_1,
    public_key=None,
):
    """Build a create request with a valid signature over the canonical message."""
    if public_key is None:
        public_key = ed25519.public_key_from_seed(seed)
    message = attestation_message_bytes(target_type, target_id, signer_actor_id)
    signature = ed25519.sign(seed, message)
    return {
        "target_type": target_type,
        "target_id": target_id,
        "signer_actor_id": signer_actor_id,
        "public_key": _b64(public_key),
        "signature": _b64(signature),
    }


def _create_content(client, digest=DIGEST_A, actor_id="org-1"):
    resp = client.post(
        "/v1/contents", json=content_payload(actor_id=actor_id, digest=digest)
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, actor_id="org-1"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": "authorship",
            "payload": {"statement": "x"},
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
            "digest_hex": hashlib.sha256(b"evidence-a").hexdigest(),
            "media_type": "image/jpeg",
            "metadata": {"source": "camera-1"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _setup_claim(client, actor_id="org-1", digest=DIGEST_A):
    create_actor(client, actor_id=actor_id)
    content = _create_content(client, digest=digest, actor_id=actor_id)
    return _create_claim(client, content_id=content["id"], actor_id=actor_id)


def _create_attestation(client, **overrides):
    resp = client.post("/v1/attestations", json=_signed_payload(**overrides))
    assert resp.status_code == 201, resp.text
    return resp.json()


# --- Normal creation ---------------------------------------------------------


def test_create_attestation_on_claim_returns_full_public_fields(client):
    claim = _setup_claim(client)
    payload = _signed_payload("claim", claim["id"], "org-1")
    resp = client.post("/v1/attestations", json=payload)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == {
        "id",
        "target_type",
        "target_id",
        "signer_actor_id",
        "public_key",
        "signature_digest_algorithm",
        "signature_digest_hex",
        "verified",
        "created_at",
    }
    assert body["id"].startswith("att_")
    assert len(body["id"]) == 4 + 64
    assert body["target_type"] == "claim"
    assert body["target_id"] == claim["id"]
    assert body["signer_actor_id"] == "org-1"
    assert body["public_key"] == payload["public_key"]
    assert body["signature_digest_algorithm"] == "sha256"
    expected_digest = hashlib.sha256(
        base64.b64decode(payload["signature"])
    ).hexdigest()
    assert body["signature_digest_hex"] == expected_digest
    assert body["verified"] is True
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))
    # The raw signature is never echoed back.
    assert payload["signature"] not in resp.text
    assert "signature" not in body


def test_create_attestation_on_evidence_bundle(client):
    claim = _setup_claim(client)
    bundle = _create_bundle(client, claim_id=claim["id"])
    body = _create_attestation(
        client, target_type="evidence_bundle", target_id=bundle["id"],
        signer_actor_id="org-1",
    )
    assert body["target_type"] == "evidence_bundle"
    assert body["target_id"] == bundle["id"]
    assert body["verified"] is True


def test_attestation_id_is_deterministic_and_stable(client):
    claim = _setup_claim(client)
    first = _create_attestation(
        client, target_type="claim", target_id=claim["id"],
        signer_actor_id="org-1",
    )
    repeat = client.post(
        "/v1/attestations",
        json=_signed_payload("claim", claim["id"], "org-1"),
    )
    assert repeat.status_code == 200
    assert repeat.json() == first


def test_get_attestation_returns_full_fields(client):
    claim = _setup_claim(client)
    created = _create_attestation(
        client, target_type="claim", target_id=claim["id"],
        signer_actor_id="org-1",
    )
    resp = client.get(f"/v1/attestations/{created['id']}")
    assert resp.status_code == 200
    assert resp.json() == created


def test_distinct_signers_and_keys_create_distinct_attestations(client):
    claim = _setup_claim(client)
    create_actor(client, actor_id="org-2", name="Other", type="person")
    first = _create_attestation(
        client, target_type="claim", target_id=claim["id"],
        signer_actor_id="org-1",
    )
    by_signer = _create_attestation(
        client, target_type="claim", target_id=claim["id"],
        signer_actor_id="org-2", seed=SEED_2,
    )
    by_key = _create_attestation(
        client, target_type="claim", target_id=claim["id"],
        signer_actor_id="org-1", seed=SEED_3,
    )
    ids = {first["id"], by_signer["id"], by_key["id"]}
    assert len(ids) == 3


def test_non_ascii_signer_id_uses_utf8_canonical_message(client):
    claim = _setup_claim(client, actor_id="组织-1")
    body = _create_attestation(
        client, target_type="claim", target_id=claim["id"],
        signer_actor_id="组织-1",
    )
    assert body["signer_actor_id"] == "组织-1"
    assert body["verified"] is True


# --- Idempotency --------------------------------------------------------------


def test_repeat_adds_no_row_or_audit_event(client, db_session):
    claim = _setup_claim(client)
    first = _create_attestation(
        client, target_type="claim", target_id=claim["id"],
        signer_actor_id="org-1",
    )
    events_after_create = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )
    for _ in range(3):
        repeat = client.post(
            "/v1/attestations",
            json=_signed_payload("claim", claim["id"], "org-1"),
        )
        assert repeat.status_code == 200
        assert repeat.json() == first
    rows = db_session.execute(select(Attestation)).scalars().all()
    assert len(rows) == 1
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_after_create
    )


# --- Verification boundary -----------------------------------------------------


def test_invalid_signature_is_422_without_writes(client, db_session):
    claim = _setup_claim(client)
    events_before = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )
    payload = _signed_payload("claim", claim["id"], "org-1")
    # Sign a different message than the request claims.
    forged = ed25519.sign(SEED_1, b"not-the-canonical-message")
    payload["signature"] = _b64(forged)
    resp = client.post("/v1/attestations", json=payload)
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "attestation_verification_failed"
    assert db_session.execute(select(Attestation)).scalars().all() == []
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )


def test_signature_from_wrong_key_is_422(client):
    claim = _setup_claim(client)
    payload = _signed_payload("claim", claim["id"], "org-1", seed=SEED_1)
    # Present a different public key than the one that signed.
    payload["public_key"] = _b64(PUBLIC_KEY_2)
    resp = client.post("/v1/attestations", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "attestation_verification_failed"


def test_signature_for_other_target_is_422(client):
    claim = _setup_claim(client)
    other_content = _create_content(client, digest=DIGEST_B)
    other_claim = _create_claim(client, content_id=other_content["id"])
    # Valid signature, but over the other claim's canonical message.
    payload = _signed_payload("claim", other_claim["id"], "org-1")
    payload["target_id"] = claim["id"]
    resp = client.post("/v1/attestations", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "attestation_verification_failed"


# --- Missing-resource boundary (distinct from validation) ----------------------


def test_unknown_claim_target_is_not_found(client):
    create_actor(client)
    resp = client.post(
        "/v1/attestations",
        json=_signed_payload("claim", "clm_ghost", "org-1"),
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "claim_not_found"
    assert error["details"]["claim_id"] == "clm_ghost"


def test_unknown_bundle_target_is_not_found(client):
    create_actor(client)
    resp = client.post(
        "/v1/attestations",
        json=_signed_payload("evidence_bundle", "evb_ghost", "org-1"),
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "evidence_bundle_not_found"
    assert error["details"]["evidence_bundle_id"] == "evb_ghost"


def test_unknown_signer_is_not_found(client):
    claim = _setup_claim(client)
    resp = client.post(
        "/v1/attestations",
        json=_signed_payload("claim", claim["id"], "ghost-actor"),
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "unknown_actor"
    assert error["details"]["actor_id"] == "ghost-actor"


def test_get_unknown_attestation_is_distinct_not_found(client):
    resp = client.get("/v1/attestations/att_does_not_exist")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "attestation_not_found"
    assert error["details"]["attestation_id"] == "att_does_not_exist"


# --- Validation boundary --------------------------------------------------------


def test_reject_invalid_target_type(client):
    create_actor(client)
    resp = client.post(
        "/v1/attestations",
        json=_signed_payload("content", "cnt_x", "org-1"),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_reject_invalid_base64_and_wrong_lengths(client):
    claim = _setup_claim(client)
    good = _signed_payload("claim", claim["id"], "org-1")
    cases = []
    for field, values in (
        ("public_key", ["not-base64!!", _b64(b"\x00" * 31), _b64(b"\x00" * 33)]),
        ("signature", ["not-base64!!", _b64(b"\x00" * 63), _b64(b"\x00" * 65)]),
    ):
        for bad in values:
            payload = dict(good)
            payload[field] = bad
            cases.append((field, bad, client.post("/v1/attestations", json=payload)))
    for field, bad, resp in cases:
        assert resp.status_code == 422, (field, bad)
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_missing_required_fields(client):
    resp = client.post("/v1/attestations", json={})
    assert resp.status_code == 422
    issue_fields = {
        ".".join(part for part in issue["loc"] if part != "body")
        for issue in resp.json()["error"]["details"]["issues"]
    }
    assert {
        "target_type",
        "target_id",
        "signer_actor_id",
        "public_key",
        "signature",
    }.issubset(issue_fields)


def test_reject_undeclared_fields(client):
    claim = _setup_claim(client)
    payload = _signed_payload("claim", claim["id"], "org-1")
    payload["comment"] = "not part of the schema"
    resp = client.post("/v1/attestations", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Listing ---------------------------------------------------------------------


def test_list_attestations_filters_and_stably_orders(client):
    claim_one = _setup_claim(client)
    bundle = _create_bundle(client, claim_id=claim_one["id"])
    content_two = _create_content(client, digest=DIGEST_B)
    claim_two = _create_claim(client, content_id=content_two["id"])

    a1 = _create_attestation(
        client, target_type="claim", target_id=claim_one["id"],
        signer_actor_id="org-1",
    )
    a2 = _create_attestation(
        client, target_type="evidence_bundle", target_id=bundle["id"],
        signer_actor_id="org-1",
    )
    a3 = _create_attestation(
        client, target_type="claim", target_id=claim_two["id"],
        signer_actor_id="org-1",
    )
    a4 = _create_attestation(
        client, target_type="claim", target_id=claim_one["id"],
        signer_actor_id="org-1", seed=SEED_2,
    )

    resp = client.get("/v1/attestations")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 4
    assert [item["id"] for item in body["items"]] == [
        a1["id"], a2["id"], a3["id"], a4["id"]
    ]

    by_type = client.get("/v1/attestations?target_type=claim").json()
    assert [item["id"] for item in by_type["items"]] == [
        a1["id"], a3["id"], a4["id"]
    ]
    assert by_type["count"] == 3

    by_target = client.get(
        f"/v1/attestations?target_id={claim_one['id']}"
    ).json()
    assert [item["id"] for item in by_target["items"]] == [
        a1["id"], a4["id"]
    ]
    assert by_target["count"] == 2

    by_both = client.get(
        f"/v1/attestations?target_type=claim&target_id={claim_two['id']}"
    ).json()
    assert [item["id"] for item in by_both["items"]] == [a3["id"]]
    assert by_both["count"] == 1

    empty = client.get("/v1/attestations?target_id=clm_ghost").json()
    assert empty == {"items": [], "count": 0}


def test_list_attestations_empty(client):
    resp = client.get("/v1/attestations")
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 0}


# --- Transactional and storage guarantees ----------------------------------------


def test_attestation_and_audit_event_commit_atomically(client, db_session):
    claim = _setup_claim(client)
    created = _create_attestation(
        client, target_type="claim", target_id=claim["id"],
        signer_actor_id="org-1",
    )
    rows = db_session.execute(select(Attestation)).scalars().all()
    events = db_session.execute(
        select(AuditEvent)
        .where(AuditEvent.event_type == EVENT_ATTESTATION_CREATED)
        .order_by(AuditEvent.seq.asc())
    ).scalars().all()
    # Both the attestation row and its audit event are visible after one
    # request: they committed in the same transaction.
    assert [r.id for r in rows] == [created["id"]]
    assert [e.resource_id for e in events] == [created["id"]]
    created_at = events[0].created_at
    assert created_at.tzinfo.utcoffset(created_at).total_seconds() == 0


def test_raw_signature_is_never_persisted(client, db_session):
    claim = _setup_claim(client)
    payload = _signed_payload("claim", claim["id"], "org-1")
    resp = client.post("/v1/attestations", json=payload)
    assert resp.status_code == 201, resp.text

    # There is no column capable of carrying the raw signature.
    columns = [c.name for c in Attestation.__table__.columns]
    assert "signature" not in columns

    row = db_session.execute(select(Attestation)).scalars().one()
    persisted = " ".join(
        str(getattr(row, attr))
        for attr in (
            "id",
            "target_type",
            "target_id",
            "signer_actor_id",
            "public_key_b64",
            "signature_digest_algorithm",
            "signature_digest_hex",
        )
    )
    assert payload["signature"] not in persisted


# --- Concurrent race boundary ------------------------------------------------


def test_concurrent_identical_requests_yield_one_attestation_and_audit(
    tmp_db_url, file_app, file_client
):
    # Concurrent identical creates must collapse into one resource and one
    # audit event (one wins the unique constraint, the others are deduped),
    # never a 500.
    import threading

    claim = _setup_claim(file_client)
    payload = _signed_payload("claim", claim["id"], "org-1")

    from provenance import service
    from provenance.schemas import AttestationCreate

    factory = file_app.state.session_factory
    request = AttestationCreate(**payload)
    results: list[tuple[str, bool]] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        session = factory()
        try:
            barrier.wait()
            attestation, created = service.create_attestation(session, request)
            results.append((attestation.id, created))
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
    assert {att_id for att_id, _ in results} == {results[0][0]}
    assert sum(1 for _, created in results if created) == 1

    audit_session = factory()
    try:
        rows = audit_session.execute(select(Attestation)).scalars().all()
        events = audit_session.execute(
            select(AuditEvent).where(
                AuditEvent.event_type == EVENT_ATTESTATION_CREATED
            )
        ).scalars().all()
        assert [r.id for r in rows] == [results[0][0]]
        assert [e.resource_id for e in events] == [results[0][0]]
    finally:
        audit_session.close()


# --- Persistence across restarts ----------------------------------------------


def test_attestations_persist_across_app_restarts(tmp_db_url, file_client):
    claim = _setup_claim(file_client)
    created = file_client.post(
        "/v1/attestations",
        json=_signed_payload("claim", claim["id"], "org-1"),
    ).json()

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        fetched = client.get(f"/v1/attestations/{created['id']}")
        assert fetched.status_code == 200
        assert fetched.json() == created

        listing = client.get(
            f"/v1/attestations?target_type=claim&target_id={claim['id']}"
        ).json()
        assert listing["count"] == 1
        assert listing["items"][0]["id"] == created["id"]

        # A repeat submission after the restart is still idempotent.
        repeat = client.post(
            "/v1/attestations",
            json=_signed_payload("claim", claim["id"], "org-1"),
        )
        assert repeat.status_code == 200
        assert repeat.json() == created

        # Audit history (actor, content, claim, attestation) survived.
        import sqlite3

        path = tmp_db_url.removeprefix("sqlite:///")
        con = sqlite3.connect(path)
        audit_count = con.execute(
            "SELECT COUNT(*) FROM audit_events"
        ).fetchone()[0]
        attestation_audit = con.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = ?",
            (EVENT_ATTESTATION_CREATED,),
        ).fetchone()[0]
        con.close()
        assert audit_count == 4
        assert attestation_audit == 1
    # Only the digest of the signature is stored.
    expected_digest = hashlib.sha256(
        base64.b64decode(payload["signature"])
    ).hexdigest()
    assert row.signature_digest_hex == expected_digest


# --- Persistence across restarts ---------------------------------------------------


def test_attestations_persist_across_app_restarts(tmp_db_url, file_client):
    claim = _setup_claim(file_client)
    created = file_client.post(
        "/v1/attestations",
        json=_signed_payload("claim", claim["id"], "org-1"),
    ).json()

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        fetched = client.get(f"/v1/attestations/{created['id']}")
        assert fetched.status_code == 200
        assert fetched.json() == created

        listing = client.get(
            f"/v1/attestations?target_id={claim['id']}"
        ).json()
        assert listing["count"] == 1
        assert listing["items"][0]["id"] == created["id"]

        # Audit history (actor, content, claim, attestation) survived.
        import sqlite3

        path = tmp_db_url.removeprefix("sqlite:///")
        con = sqlite3.connect(path)
        audit_count = con.execute(
            "SELECT COUNT(*) FROM audit_events"
        ).fetchone()[0]
        attestation_audit = con.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = ?",
            (EVENT_ATTESTATION_CREATED,),
        ).fetchone()[0]
        con.close()
        assert audit_count == 4
        assert attestation_audit == 1
