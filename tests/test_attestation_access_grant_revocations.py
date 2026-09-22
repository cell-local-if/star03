"""Tests for immutable revocations of read-only proof-access grants.

Covers ``POST /v1/attestation-access-grant-revocations`` and its effect on
``GET /v1/protected/attestations/{attestation_id}``, including:

* the shared ``X-PA``/``X-PT``/``X-PS`` signed-header contract on the new
  write route (missing/malformed credentials, body/method/path binding);
* signer-only revocation: only the signing subject of the attestation the
  grant covers may revoke it; an unknown grant and a non-signer are 422;
* first creation (201, stable ``agr_`` id, exact public fields) and the
  single-transaction commit of the row with its
  ``attestation.access_grant_revoked`` audit event;
* idempotency: the same grant, revoking subject, and reason retries to the
  original record (200) with no new row or audit event, while a different
  reason is an independent immutable record;
* the post-revocation authorization boundary: the signer keeps reading,
  the revoked grant's grantee is denied through the opaque 404, and other
  unrevoked grants (other grantees, other attestations) are unaffected;
* append-only immutability (no update or delete path) and preservation of
  the original grant and its ``attestation.access_granted`` audit;
* no writes from any failed request, no persisted signing material, and a
  concurrent-identical-revocation race collapsing to one record/audit.

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures).
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from provenance.access_signing import access_message_bytes
from provenance.ids import attestation_access_grant_revocation_id
from provenance.models import (
    EVENT_ATTESTATION_ACCESS_GRANTED,
    EVENT_ATTESTATION_ACCESS_GRANT_REVOKED,
    AttestationAccessGrant,
    AttestationAccessGrantRevocation,
    AuditEvent,
)
from provenance.signing import attestation_message_bytes
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    content_payload,
    create_actor,
    ed25519_public_key,
    ed25519_sign,
    SEED_A,
    SEED_B,
)

REVOCATIONS_PATH = "/v1/attestation-access-grant-revocations"
GRANTS_PATH = "/v1/attestation-access-grants"
SEED_C = b"test-ed25519-seed-c-00000000000000"[:32]
REASON = "investigation closed; access withdrawn"


# --- Fixture-style setup ------------------------------------------------------


def _make_claim(client, actor_id, digest=DIGEST_A):
    content = client.post(
        "/v1/contents",
        json=content_payload(actor_id=actor_id, digest=digest),
    )
    assert content.status_code == 201, content.text
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content.json()["id"],
            "actor_id": actor_id,
            "claim_type": "authorship",
            "payload": {"statement": "made"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make_attestation(client, actor_id, seed, target_id):
    signature = ed25519_sign(
        seed, attestation_message_bytes("claim", target_id, actor_id)
    )
    resp = client.post(
        "/v1/attestations",
        json={
            "target_type": "claim",
            "target_id": target_id,
            "signer_actor_id": actor_id,
            "public_key": base64.b64encode(ed25519_public_key(seed)).decode(),
            "signature": base64.b64encode(signature).decode(),
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _world(client):
    """Create the signer plus two grantees; return the signer's attestation.

    Each non-signer holds its own current attestation, hence a current
    public key, so its access signature authenticates on the read route.
    """
    create_actor(client)  # org-1, the signer
    for actor_id, seed, digest in (
        ("org-2", SEED_B, DIGEST_B),
        ("org-3", SEED_C, hashlib.sha256(b"content-c3").hexdigest()),
    ):
        create_actor(client, actor_id=actor_id, name=actor_id, type="organization")
        _make_attestation(
            client, actor_id, seed, _make_claim(client, actor_id, digest)["id"]
        )
    attestation = _make_attestation(
        client, "org-1", SEED_A, _make_claim(client, "org-1")["id"]
    )
    return attestation


def _second_signer_attestation(client):
    """A second attestation signed by org-1 under the same current key."""
    return _make_attestation(
        client,
        "org-1",
        SEED_A,
        _make_claim(
            client, "org-1", digest=hashlib.sha256(b"second-signer-content").hexdigest()
        )["id"],
    )


# --- Signed-header helpers ---------------------------------------------------


def _signed_headers(
    method,
    path,
    body,
    *,
    actor="org-1",
    seed=SEED_A,
    timestamp=None,
    signed_timestamp=None,
    signed_method=None,
    signed_path=None,
    signed_body=None,
):
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    signed_ts = signed_timestamp if signed_timestamp is not None else ts
    body_digest = hashlib.sha256(
        signed_body if signed_body is not None else body
    ).hexdigest()
    message = access_message_bytes(
        signed_method or method,
        signed_path if signed_path is not None else path,
        signed_ts,
        body_digest,
    )
    signature = base64.b64encode(ed25519_sign(seed, message)).decode("ascii")
    return {"X-PA": actor, "X-PT": ts, "X-PS": signature}


def _grant_body(attestation_id, grantee_actor_id):
    return {"attestation_id": attestation_id, "grantee_actor_id": grantee_actor_id}


def _revocation_body(grant_id, reason=REASON):
    return {"grant_id": grant_id, "reason": reason}


def _post_json_signed(client, path, body_obj, *, actor="org-1", seed=SEED_A,
                      **sign_kwargs):
    body = json.dumps(body_obj).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", path, body, actor=actor, seed=seed,
                          **sign_kwargs),
    }
    return client.post(path, content=body, headers=headers)


def _post_grant(client, attestation_id, grantee_actor_id, *, actor="org-1",
                seed=SEED_A):
    return _post_json_signed(
        client, GRANTS_PATH, _grant_body(attestation_id, grantee_actor_id),
        actor=actor, seed=seed,
    )


def _post_revocation(client, grant_id, *, reason=REASON, actor="org-1",
                     seed=SEED_A, **sign_kwargs):
    return _post_json_signed(
        client, REVOCATIONS_PATH, _revocation_body(grant_id, reason),
        actor=actor, seed=seed, **sign_kwargs,
    )


def _get_protected(client, attestation_id, *, actor="org-1", seed=SEED_A):
    path = f"/v1/protected/attestations/{attestation_id}"
    headers = _signed_headers("GET", path, b"", actor=actor, seed=seed)
    return client.get(path, headers=headers)


def _granted(client, attestation_id, grantee_actor_id):
    resp = _post_grant(client, attestation_id, grantee_actor_id)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


# --- First creation -----------------------------------------------------------


def test_first_revocation_returns_201_with_exact_public_fields(client):
    attestation = _world(client)
    grant = _granted(client, attestation["id"], "org-2")

    resp = _post_revocation(client, grant["id"])
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == {
        "id", "grant_id", "revoker_actor_id", "reason", "created_at",
    }
    assert body["id"].startswith("agr_")
    assert len(body["id"]) == len("agr_") + 64
    assert body["grant_id"] == grant["id"]
    assert body["revoker_actor_id"] == "org-1"
    assert body["reason"] == REASON
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))


def test_revocation_id_is_the_stable_triple_derived_identifier(client):
    attestation = _world(client)
    grant = _granted(client, attestation["id"], "org-2")
    body = _post_revocation(client, grant["id"]).json()
    assert body["id"] == attestation_access_grant_revocation_id(
        grant["id"], "org-1", REASON
    )


def test_revocation_and_audit_event_commit_in_one_transaction(client, db_session):
    attestation = _world(client)
    grant = _granted(client, attestation["id"], "org-2")
    created = _post_revocation(client, grant["id"]).json()

    rows = db_session.execute(
        select(AttestationAccessGrantRevocation)
    ).scalars().all()
    assert [(r.id, r.grant_id, r.revoker_actor_id, r.reason) for r in rows] == [
        (created["id"], grant["id"], "org-1", REASON)
    ]
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ATTESTATION_ACCESS_GRANT_REVOKED
        )
    ).scalars().all()
    assert [e.resource_id for e in events] == [created["id"]]
    assert events[0].created_at.tzinfo.utcoffset(
        events[0].created_at
    ).total_seconds() == 0


# --- Idempotency ---------------------------------------------------------------


def test_same_grant_and_reason_retry_returns_200_original_and_no_audit(
    client, db_session
):
    attestation = _world(client)
    grant = _granted(client, attestation["id"], "org-2")
    first = _post_revocation(client, grant["id"])
    assert first.status_code == 201
    events_after_first = _audit_count(db_session)

    for _ in range(3):
        retry = _post_revocation(client, grant["id"])
        assert retry.status_code == 200
        assert retry.json() == first.json()

    rows = db_session.execute(
        select(AttestationAccessGrantRevocation)
    ).scalars().all()
    assert [r.id for r in rows] == [first.json()["id"]]
    assert _audit_count(db_session) == events_after_first
    revoked_audit = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ATTESTATION_ACCESS_GRANT_REVOKED
        )
    ).scalars().all()
    assert len(revoked_audit) == 1


def test_different_reason_forms_an_independent_immutable_record(client):
    attestation = _world(client)
    grant = _granted(client, attestation["id"], "org-2")
    first = _post_revocation(client, grant["id"], reason="reason one")
    assert first.status_code == 201
    second = _post_revocation(client, grant["id"], reason="reason two")
    assert second.status_code == 201
    assert second.json()["id"] != first.json()["id"]
    assert second.json()["grant_id"] == grant["id"]

    # Each triple retries to its own original record.
    assert _post_revocation(client, grant["id"], reason="reason one").json() == (
        first.json()
    )
    assert _post_revocation(client, grant["id"], reason="reason two").status_code == 200


def test_surrounding_whitespace_on_reason_is_trimmed_for_identity(client):
    attestation = _world(client)
    grant = _granted(client, attestation["id"], "org-2")
    first = _post_revocation(client, grant["id"], reason="  trimmed reason  ")
    assert first.status_code == 201
    assert first.json()["reason"] == "trimmed reason"
    # The trimmed text defines the identity triple.
    retry = _post_revocation(client, grant["id"], reason="trimmed reason")
    assert retry.status_code == 200
    assert retry.json() == first.json()


# --- Permission isolation ------------------------------------------------------


def test_grantee_cannot_revoke_even_with_a_valid_key(client, db_session):
    attestation = _world(client)
    grant = _granted(client, attestation["id"], "org-2")
    events_before = _audit_count(db_session)

    resp = _post_revocation(client, grant["id"], actor="org-2", seed=SEED_B)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert resp.json()["error"]["details"]["reason"] == "caller_not_signer"
    assert (
        db_session.execute(
            select(AttestationAccessGrantRevocation)
        ).scalars().all()
        == []
    )
    assert _audit_count(db_session) == events_before


def test_stranger_actor_cannot_revoke(client, db_session):
    attestation = _world(client)
    grant = _granted(client, attestation["id"], "org-2")
    events_before = _audit_count(db_session)

    resp = _post_revocation(client, grant["id"], actor="org-3", seed=SEED_C)
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "caller_not_signer"
    assert _audit_count(db_session) == events_before


def test_unknown_grant_is_422_and_writes_nothing(client, db_session):
    _world(client)
    events_before = _audit_count(db_session)

    resp = _post_revocation(client, "aag_ghost")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert resp.json()["error"]["details"]["reason"] == "grant_not_found"
    assert (
        db_session.execute(
            select(AttestationAccessGrantRevocation)
        ).scalars().all()
        == []
    )
    assert _audit_count(db_session) == events_before


# --- Post-revocation read isolation --------------------------------------------


def test_signer_still_reads_after_revocation(client):
    attestation = _world(client)
    grant = _granted(client, attestation["id"], "org-2")
    assert _post_revocation(client, grant["id"]).status_code == 201

    resp = _get_protected(client, attestation["id"])
    assert resp.status_code == 200
    assert resp.json() == attestation


def test_revoked_grant_no_longer_authorizes_the_grantee(client, db_session):
    attestation = _world(client)
    grant = _granted(client, attestation["id"], "org-2")

    before = _get_protected(client, attestation["id"], actor="org-2", seed=SEED_B)
    assert before.status_code == 200

    assert _post_revocation(client, grant["id"]).status_code == 201

    denied = _get_protected(client, attestation["id"], actor="org-2", seed=SEED_B)
    assert denied.status_code == 404
    assert denied.json()["error"]["code"] == "not_found"
    # Indistinguishable from a caller who never had a grant.
    assert denied.json() == _get_protected(
        client, attestation["id"], actor="org-3", seed=SEED_C
    ).json()


def test_revoking_one_grant_leaves_another_grantee_of_same_attestation(client):
    attestation = _world(client)
    grant_org2 = _granted(client, attestation["id"], "org-2")
    grant_org3 = _granted(client, attestation["id"], "org-3")
    assert grant_org2["id"] != grant_org3["id"]

    assert _post_revocation(client, grant_org2["id"]).status_code == 201

    # org-2 is cut off; org-3's independent grant still authorizes.
    assert _get_protected(
        client, attestation["id"], actor="org-2", seed=SEED_B
    ).status_code == 404
    assert _get_protected(
        client, attestation["id"], actor="org-3", seed=SEED_C
    ).status_code == 200


def test_revoking_one_attestation_grant_leaves_same_grantee_elsewhere(client):
    first = _world(client)
    second = _second_signer_attestation(client)
    grant_a = _granted(client, first["id"], "org-2")
    grant_b = _granted(client, second["id"], "org-2")

    assert _post_revocation(client, grant_a["id"]).status_code == 201

    # Access to the revoked attestation is gone; the other, unrevoked grant
    # for the same grantee still authorizes its own attestation.
    assert _get_protected(
        client, first["id"], actor="org-2", seed=SEED_B
    ).status_code == 404
    other = _get_protected(client, second["id"], actor="org-2", seed=SEED_B)
    assert other.status_code == 200
    assert other.json() == second


def test_revocation_preserves_the_grant_and_its_original_audit(client, db_session):
    attestation = _world(client)
    grant = _granted(client, attestation["id"], "org-2")
    assert _post_revocation(client, grant["id"]).status_code == 201

    # The grant row is neither mutated nor deleted.
    stored = db_session.execute(
        select(AttestationAccessGrant).where(
            AttestationAccessGrant.id == grant["id"]
        )
    ).scalar_one()
    assert stored.attestation_id == attestation["id"]
    assert stored.grantee_actor_id == "org-2"
    assert stored.created_at.isoformat()  # original timestamp intact

    granted_audit = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ATTESTATION_ACCESS_GRANTED,
            AuditEvent.resource_id == grant["id"],
        )
    ).scalars().all()
    assert len(granted_audit) == 1


# --- Credential contract on the write route ------------------------------------


def test_revocation_without_headers_is_422(client, db_session):
    attestation = _world(client)
    grant = _granted(client, attestation["id"], "org-2")
    body = json.dumps(_revocation_body(grant["id"])).encode()
    events_before = _audit_count(db_session)

    resp = client.post(
        REVOCATIONS_PATH, content=body,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "missing_credentials"
    assert _audit_count(db_session) == events_before


def test_revocation_with_unverifiable_signature_is_422(client, db_session):
    attestation = _world(client)
    grant = _granted(client, attestation["id"], "org-2")
    events_before = _audit_count(db_session)

    # SEED_B is not a current key of org-1, so nothing verifies.
    resp = _post_revocation(client, grant["id"], actor="org-1", seed=SEED_B)
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )
    assert _audit_count(db_session) == events_before


def test_revocation_rejects_stale_and_malformed_timestamps(client):
    attestation = _world(client)
    grant = _granted(client, attestation["id"], "org-2")
    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    future = (
        datetime.now(timezone.utc) + timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    cases = {
        stale: "timestamp_out_of_window",
        future: "timestamp_out_of_window",
        "2026-09-20T12:00:00": "invalid_timestamp",
        "2026-09-20 12:00:00Z": "invalid_timestamp",
        "2026-09-20T12:00:00+01:00": "invalid_timestamp",
        "not-a-timestamp": "invalid_timestamp",
    }
    for raw, reason in cases.items():
        resp = _post_revocation(client, grant["id"], timestamp=raw)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["details"]["reason"] == reason, raw


def test_revocation_signature_binds_body_method_path_and_timestamp(client):
    attestation = _world(client)
    grant = _granted(client, attestation["id"], "org-2")

    body_bound = _post_revocation(
        client, grant["id"],
        signed_body=b'{"grant_id":"aag_other","reason":"x"}',
    )
    assert body_bound.status_code == 422
    assert body_bound.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )

    assert _post_revocation(client, grant["id"], signed_method="GET").status_code == 422
    assert _post_revocation(
        client, grant["id"], signed_path=REVOCATIONS_PATH + "/"
    ).status_code == 422
    assert _post_revocation(
        client, grant["id"], signed_path=GRANTS_PATH
    ).status_code == 422

    sent = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    signed = (
        datetime.now(timezone.utc) - timedelta(seconds=10)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert _post_revocation(
        client, grant["id"], timestamp=sent, signed_timestamp=signed
    ).status_code == 422


# --- Body validation and no-write guarantees ------------------------------------


def test_revocation_body_validation_failures_are_422_and_write_nothing(
    client, db_session
):
    attestation = _world(client)
    grant = _granted(client, attestation["id"], "org-2")
    revocations_before = db_session.scalar(
        select(func.count()).select_from(AttestationAccessGrantRevocation)
    )
    events_before = _audit_count(db_session)

    def post_obj(obj):
        body = json.dumps(obj).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            **_signed_headers("POST", REVOCATIONS_PATH, body),
        }
        return client.post(REVOCATIONS_PATH, content=body, headers=headers)

    blank_grant = post_obj({"grant_id": "  ", "reason": REASON})
    blank_reason = post_obj({"grant_id": grant["id"], "reason": "   "})
    missing_reason = post_obj({"grant_id": grant["id"]})
    missing_grant = post_obj({"reason": REASON})
    empty = post_obj({})
    extra = post_obj({"grant_id": grant["id"], "reason": REASON, "nope": True})
    wrong_type = post_obj({"grant_id": 123, "reason": REASON})
    for resp in (
        blank_grant, blank_reason, missing_reason, missing_grant, empty,
        extra, wrong_type,
    ):
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "validation_error"

    # Malformed JSON is rejected at the boundary even though the signature
    # commits to those exact (malformed) bytes.
    raw_body = b"{not valid json"
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", REVOCATIONS_PATH, raw_body),
    }
    malformed = client.post(REVOCATIONS_PATH, content=raw_body, headers=headers)
    assert malformed.status_code == 422

    assert (
        db_session.scalar(
            select(func.count()).select_from(AttestationAccessGrantRevocation)
        )
        == revocations_before
    )
    assert _audit_count(db_session) == events_before


def test_all_failure_modes_write_neither_rows_nor_audit(client, db_session):
    attestation = _world(client)
    grant = _granted(client, attestation["id"], "org-2")
    rows_before = db_session.scalar(
        select(func.count()).select_from(AttestationAccessGrantRevocation)
    )
    grants_before = db_session.scalar(
        select(func.count()).select_from(AttestationAccessGrant)
    )
    events_before = _audit_count(db_session)

    failures = (
        _post_revocation(client, "aag_ghost"),
        _post_revocation(client, grant["id"], actor="org-2", seed=SEED_B),
        _post_revocation(client, grant["id"], actor="org-3", seed=SEED_C),
        _post_revocation(client, grant["id"], actor="org-1", seed=SEED_B),
    )
    assert {r.status_code for r in failures} == {422}

    db_session.expire_all()
    assert (
        db_session.scalar(
            select(func.count()).select_from(AttestationAccessGrantRevocation)
        )
        == rows_before
    )
    assert (
        db_session.scalar(
            select(func.count()).select_from(AttestationAccessGrant)
        )
        == grants_before
    )
    assert _audit_count(db_session) == events_before


# --- No sensitive material is persisted -----------------------------------------


def test_persisted_revocation_carries_no_signing_or_content_material(
    client, db_session
):
    attestation = _world(client)
    grant = _granted(client, attestation["id"], "org-2")
    created = _post_revocation(client, grant["id"]).json()

    row = db_session.execute(
        select(AttestationAccessGrantRevocation).where(
            AttestationAccessGrantRevocation.id == created["id"]
        )
    ).scalar_one()
    assert set(
        c.name for c in AttestationAccessGrantRevocation.__table__.columns
    ) == {"seq", "id", "grant_id", "revoker_actor_id", "reason", "created_at"}
    # Nothing capable of holding a private key, raw signature, payload,
    # content, or evidence byte appears on the row or in the response.
    blob = json.dumps(
        {c.name: getattr(row, c.name) for c in row.__table__.columns},
        default=str,
    )
    assert "BEGIN" not in blob
    assert set(created) == {
        "id", "grant_id", "revoker_actor_id", "reason", "created_at",
    }


# --- Append-only immutability ----------------------------------------------------


def test_revocation_has_no_update_or_delete_path(client):
    attestation = _world(client)
    grant = _granted(client, attestation["id"], "org-2")
    created = _post_revocation(client, grant["id"])
    assert created.status_code == 201

    for method, kwargs in (
        ("put", {"json": {"reason": "changed"}}),
        ("patch", {"json": {"reason": "changed"}}),
        ("delete", {}),
    ):
        resp = getattr(client, method)(REVOCATIONS_PATH, **kwargs)
        assert resp.status_code == 405, method
        assert resp.json()["error"]["code"] == "method_not_allowed"

    # A retry still returns the single original record.
    retry = _post_revocation(client, grant["id"])
    assert retry.status_code == 200
    assert retry.json() == created.json()


# --- Concurrent race --------------------------------------------------------------


def test_concurrent_identical_revocations_yield_one_record_and_audit(
    tmp_db_url, file_app, file_client
):
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
            "payload": {"s": 1},
        },
    ).json()
    attestation = _make_attestation(file_client, "org-1", SEED_A, claim["id"])
    create_actor(
        file_client, actor_id="org-2", name="Other", type="organization"
    )
    grant = _granted(file_client, attestation["id"], "org-2")

    from provenance import service
    from provenance.schemas import AttestationAccessGrantRevocationCreate

    factory = file_app.state.session_factory
    payload = AttestationAccessGrantRevocationCreate(
        grant_id=grant["id"], reason=REASON
    )
    results: list[tuple[str, bool]] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        session = factory()
        try:
            barrier.wait()
            record, created = service.create_attestation_access_grant_revocation(
                session, payload, "org-1"
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
    assert {rid for rid, _ in results} == {results[0][0]}
    assert sum(1 for _, created in results if created) == 1

    audit_session = factory()
    try:
        rows = audit_session.execute(
            select(AttestationAccessGrantRevocation)
        ).scalars().all()
        events = audit_session.execute(
            select(AuditEvent).where(
                AuditEvent.event_type
                == EVENT_ATTESTATION_ACCESS_GRANT_REVOKED
            )
        ).scalars().all()
        assert [r.id for r in rows] == [results[0][0]]
        assert [e.resource_id for e in events] == [results[0][0]]
    finally:
        audit_session.close()


# --- Persistence -------------------------------------------------------------------


def test_revocations_persist_and_keep_enforcing_across_restart(
    tmp_db_url, file_client
):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    attestation = _world(file_client)
    grant = _granted(file_client, attestation["id"], "org-2")
    created = _post_revocation(file_client, grant["id"]).json()

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        # The signer still reads; the grantee stays cut off after restart.
        assert _get_protected(client, attestation["id"]).status_code == 200
        assert _get_protected(
            client, attestation["id"], actor="org-2", seed=SEED_B
        ).status_code == 404

        import sqlite3

        con = sqlite3.connect(tmp_db_url.removeprefix("sqlite:///"))
        rows = con.execute(
            "SELECT id, grant_id, revoker_actor_id, reason "
            "FROM attestation_access_grant_revocations"
        ).fetchall()
        audit = con.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = ?",
            (EVENT_ATTESTATION_ACCESS_GRANT_REVOKED,),
        ).fetchone()[0]
        con.close()
        assert rows == [(created["id"], grant["id"], "org-1", REASON)]
        assert audit == 1
