"""Tests for authentication public-key rotation.

Covers ``POST /v1/authentication-key-rotations`` and
``POST /v1/authentication-key-rotations/{rotation_id}/retire``, including:

* the protected-route ``X-PA``/``X-PT``/``X-PS`` signed-header contract
  (inherited from the access-grant routes);
* first-creation 201 with the exact public fields, the stable ``akr_``
  identifier, and a single-transaction ``authentication_key.rotated``
  audit commit;
* same-actor/same-key idempotent retries (200, original record, no audit)
  and independent records for distinct keys;
* the caller-is-subject authorization boundary and body validation;
* activation: a rotated key authenticates the protected routes with no new
  attestation, alongside still-valid non-revoked attestation keys;
* retirement: subject-only, ``active: false`` + UTC ``retired_at``, a
  single-transaction ``authentication_key.retired`` audit commit, and
  immediate key invalidation -- with unknown record, non-subject, and
  duplicate retire all 422 writing nothing;
* no private key or raw signature material is ever persisted.

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures).
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from provenance import service
from provenance.access_signing import access_message_bytes
from provenance.errors import ProtectedAccessValidationError
from provenance.ids import authentication_key_rotation_id
from provenance.models import (
    EVENT_AUTHENTICATION_KEY_RETIRED,
    EVENT_AUTHENTICATION_KEY_ROTATED,
    AuditEvent,
    AuthenticationKeyRotation,
)
from provenance.schemas import AuthenticationKeyRotationCreate
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

ROTATIONS_PATH = "/v1/authentication-key-rotations"
GRANTS_PATH = "/v1/attestation-access-grants"

#: Keys that only ever exist as rotated keys (never on an attestation).
SEED_NEW = b"test-ed25519-seed-new-000000000000"[:32]
SEED_EXTRA = b"test-ed25519-seed-extra-00000000"[:32]


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
    """Two actors, each holding a current attestation key (SEED_A / SEED_B)."""
    create_actor(client)  # org-1
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    attestation = _make_attestation(
        client, "org-1", SEED_A, _make_claim(client, "org-1")["id"]
    )
    _make_attestation(
        client, "org-2", SEED_B, _make_claim(client, "org-2", digest=DIGEST_B)["id"]
    )
    return attestation


# --- Signed-header helpers -----------------------------------------------------


def _signed_headers(
    method,
    path,
    body,
    *,
    actor="org-1",
    seed=SEED_A,
    timestamp=None,
    signed_timestamp=None,
    signed_path=None,
):
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    signed_ts = signed_timestamp if signed_timestamp is not None else ts
    message = access_message_bytes(
        method,
        signed_path if signed_path is not None else path,
        signed_ts,
        hashlib.sha256(body).hexdigest(),
    )
    signature = base64.b64encode(ed25519_sign(seed, message)).decode("ascii")
    return {"X-PA": actor, "X-PT": ts, "X-PS": signature}


def _rotation_body(actor_id="org-1", seed=SEED_NEW):
    return {
        "actor_id": actor_id,
        "new_public_key": base64.b64encode(ed25519_public_key(seed)).decode(),
    }


def _post_rotation(client, body_obj, *, actor="org-1", seed=SEED_A, **sign_kwargs):
    body = json.dumps(body_obj).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", ROTATIONS_PATH, body, actor=actor, seed=seed,
                          **sign_kwargs),
    }
    return client.post(ROTATIONS_PATH, content=body, headers=headers)


def _retire(client, rotation_id, *, actor="org-1", seed=SEED_A, **sign_kwargs):
    path = f"{ROTATIONS_PATH}/{rotation_id}/retire"
    headers = _signed_headers("POST", path, b"", actor=actor, seed=seed,
                              **sign_kwargs)
    return client.post(path, headers=headers)


def _post_grant(client, attestation_id, grantee, *, actor, seed):
    body = json.dumps(
        {"attestation_id": attestation_id, "grantee_actor_id": grantee}
    ).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", GRANTS_PATH, body, actor=actor, seed=seed),
    }
    return client.post(GRANTS_PATH, content=body, headers=headers)


def _audit_events(session, event_type):
    return session.execute(
        select(AuditEvent).where(AuditEvent.event_type == event_type)
    ).scalars().all()


def _rotation_rows(session):
    return session.execute(select(AuthenticationKeyRotation)).scalars().all()


# --- First creation -------------------------------------------------------------


def test_first_rotation_returns_201_with_exact_public_fields(client):
    _world(client)
    resp = _post_rotation(client, _rotation_body())
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == {
        "id", "actor_id", "public_key", "active", "created_at", "retired_at",
    }
    assert body["id"].startswith("akr_")
    assert len(body["id"]) == len("akr_") + 64
    assert body["actor_id"] == "org-1"
    assert body["public_key"] == _rotation_body()["new_public_key"]
    assert body["active"] is True
    assert body["retired_at"] is None
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))


def test_rotation_id_is_the_stable_derived_identifier(client):
    _world(client)
    body = _post_rotation(client, _rotation_body()).json()
    expected = authentication_key_rotation_id(
        "org-1", ed25519_public_key(SEED_NEW).hex()
    )
    assert body["id"] == expected


def test_rotation_and_audit_event_commit_in_one_transaction(client, db_session):
    _world(client)
    created = _post_rotation(client, _rotation_body()).json()

    rows = _rotation_rows(db_session)
    assert [(r.id, r.actor_id, r.active, r.retired_at) for r in rows] == [
        (created["id"], "org-1", True, None)
    ]
    assert rows[0].public_key == ed25519_public_key(SEED_NEW)
    events = _audit_events(db_session, EVENT_AUTHENTICATION_KEY_ROTATED)
    assert [e.resource_id for e in events] == [created["id"]]
    assert events[0].created_at.tzinfo.utcoffset(
        events[0].created_at
    ).total_seconds() == 0


# --- Idempotency and independence -----------------------------------------------


def test_same_actor_same_key_retry_returns_200_original_and_no_audit(
    client, db_session
):
    _world(client)
    first = _post_rotation(client, _rotation_body())
    assert first.status_code == 201
    events_after_first = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    for _ in range(3):
        retry = _post_rotation(client, _rotation_body())
        assert retry.status_code == 200
        assert retry.json() == first.json()

    assert [r.id for r in _rotation_rows(db_session)] == [first.json()["id"]]
    assert len(db_session.execute(select(AuditEvent)).scalars().all()) == (
        events_after_first
    )
    assert len(_audit_events(db_session, EVENT_AUTHENTICATION_KEY_ROTATED)) == 1


def test_different_key_forms_an_independent_record(client, db_session):
    _world(client)
    first = _post_rotation(client, _rotation_body(seed=SEED_NEW))
    second = _post_rotation(client, _rotation_body(seed=SEED_EXTRA))
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] != second.json()["id"]

    # Retrying each key returns its own original record.
    assert _post_rotation(client, _rotation_body(seed=SEED_NEW)).json()["id"] == (
        first.json()["id"]
    )
    assert _post_rotation(client, _rotation_body(seed=SEED_EXTRA)).json()["id"] == (
        second.json()["id"]
    )
    assert len(_rotation_rows(db_session)) == 2
    assert len(_audit_events(db_session, EVENT_AUTHENTICATION_KEY_ROTATED)) == 2


def test_retry_of_a_retired_record_returns_it_unchanged(client, db_session):
    _world(client)
    created = _post_rotation(client, _rotation_body()).json()
    retired = _retire(client, created["id"])
    assert retired.status_code == 200
    events_after_retire = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    # A same-actor/same-key retry is idempotent: the retired record is
    # returned as-is and is not reactivated.
    retry = _post_rotation(client, _rotation_body())
    assert retry.status_code == 200
    assert retry.json() == retired.json()
    assert retry.json()["active"] is False
    assert len(_rotation_rows(db_session)) == 1
    assert len(db_session.execute(select(AuditEvent)).scalars().all()) == (
        events_after_retire
    )


# --- Authorization and validation boundary --------------------------------------


def test_caller_must_be_the_subject(client, db_session):
    _world(client)
    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())
    # org-2 authenticates validly but names org-1 as the subject.
    resp = _post_rotation(
        client, _rotation_body(actor_id="org-1"), actor="org-2", seed=SEED_B
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert resp.json()["error"]["details"]["reason"] == "caller_not_subject"
    assert _rotation_rows(db_session) == []
    assert len(db_session.execute(select(AuditEvent)).scalars().all()) == (
        events_before
    )


def test_rotation_body_validation_failures_are_422_and_write_nothing(
    client, db_session
):
    _world(client)
    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())

    def post_obj(obj):
        body = json.dumps(obj).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            **_signed_headers("POST", ROTATIONS_PATH, body),
        }
        return client.post(ROTATIONS_PATH, content=body, headers=headers)

    valid_key = _rotation_body()["new_public_key"]
    cases = [
        {"actor_id": "  ", "new_public_key": valid_key},
        {"actor_id": "org-1"},
        {"new_public_key": valid_key},
        {"actor_id": "org-1", "new_public_key": "@@@"},
        {"actor_id": "org-1", "new_public_key": "aGVsbG8"},  # bad padding
        # Decodes to 31 bytes, not 32.
        {"actor_id": "org-1",
         "new_public_key": base64.b64encode(b"k" * 31).decode()},
        {"actor_id": "org-1", "new_public_key": valid_key, "private_key": "x"},
    ]
    for obj in cases:
        resp = post_obj(obj)
        assert resp.status_code == 422, obj
        assert resp.json()["error"]["code"] == "validation_error"

    # Malformed JSON is rejected at the boundary even when the signature
    # commits to those exact bytes.
    raw_body = b"{not valid json"
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", ROTATIONS_PATH, raw_body),
    }
    assert client.post(
        ROTATIONS_PATH, content=raw_body, headers=headers
    ).status_code == 422

    assert _rotation_rows(db_session) == []
    assert len(db_session.execute(select(AuditEvent)).scalars().all()) == (
        events_before
    )


def test_rotation_requires_valid_credentials(client, db_session):
    _world(client)
    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())

    # No headers at all.
    body = json.dumps(_rotation_body()).encode()
    resp = client.post(
        ROTATIONS_PATH, content=body,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "missing_credentials"

    # A key that is not a current key of the caller.
    resp = _post_rotation(client, _rotation_body(), actor="org-1", seed=SEED_B)
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )

    # A stale timestamp.
    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    resp = _post_rotation(client, _rotation_body(), timestamp=stale)
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == (
        "timestamp_out_of_window"
    )

    # The signature binds to the exact path.
    resp = _post_rotation(
        client, _rotation_body(), signed_path=ROTATIONS_PATH + "/"
    )
    assert resp.status_code == 422

    assert _rotation_rows(db_session) == []
    assert len(db_session.execute(select(AuditEvent)).scalars().all()) == (
        events_before
    )


def test_unknown_subject_is_422_at_the_service_boundary(db_session):
    # Unreachable through the API (an unknown caller cannot authenticate),
    # so the defensive service-level check is exercised directly.
    payload = AuthenticationKeyRotationCreate(
        actor_id="ghost",
        new_public_key=base64.b64encode(
            ed25519_public_key(SEED_NEW)
        ).decode(),
    )
    with pytest.raises(ProtectedAccessValidationError) as excinfo:
        service.create_authentication_key_rotation(db_session, payload, "ghost")
    assert excinfo.value.details["reason"] == "unknown_actor"


# --- Activation: the rotated key authenticates -----------------------------------


def test_rotated_key_authenticates_protected_routes_without_attestation(client):
    attestation = _world(client)
    created = _post_rotation(client, _rotation_body()).json()
    assert created["active"] is True

    # The brand-new key (never on any attestation) now authenticates.
    grant = _post_grant(client, attestation["id"], "org-2",
                        actor="org-1", seed=SEED_NEW)
    assert grant.status_code == 201, grant.text

    # The existing non-revoked attestation key remains valid too.
    path = f"/v1/protected/attestations/{attestation['id']}"
    headers = _signed_headers("GET", path, b"", actor="org-1", seed=SEED_A)
    assert client.get(path, headers=headers).status_code == 200


def test_rotated_key_survives_attestation_revocation(client):
    attestation = _world(client)
    _post_rotation(client, _rotation_body())

    revoked = client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": attestation["id"],
            "revoker_actor_id": "org-1",
            "reason": "superseded by rotated key",
        },
    )
    assert revoked.status_code == 201

    # The revoked attestation key no longer authenticates...
    denied = _post_grant(client, attestation["id"], "org-2",
                         actor="org-1", seed=SEED_A)
    assert denied.status_code == 422
    # ...but the active rotated key still does.
    granted = _post_grant(client, attestation["id"], "org-2",
                          actor="org-1", seed=SEED_NEW)
    assert granted.status_code == 201, granted.text


# --- Retirement -------------------------------------------------------------------


def test_retire_returns_inactive_record_and_commits_audit(client, db_session):
    _world(client)
    created = _post_rotation(client, _rotation_body()).json()

    resp = _retire(client, created["id"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {
        "id", "actor_id", "public_key", "active", "created_at", "retired_at",
    }
    assert body["id"] == created["id"]
    assert body["actor_id"] == "org-1"
    assert body["public_key"] == created["public_key"]
    assert body["active"] is False
    assert body["created_at"] == created["created_at"]
    retired_at = datetime.fromisoformat(body["retired_at"])
    assert retired_at.utcoffset().total_seconds() == 0
    assert body["retired_at"].endswith(("Z", "+00:00"))

    row = _rotation_rows(db_session)[0]
    assert row.active is False
    assert row.retired_at is not None
    assert row.retired_at.tzinfo.utcoffset(
        row.retired_at
    ).total_seconds() == 0
    events = _audit_events(db_session, EVENT_AUTHENTICATION_KEY_RETIRED)
    assert [e.resource_id for e in events] == [created["id"]]


def test_retired_key_stops_authenticating_immediately(client):
    attestation = _world(client)
    created = _post_rotation(client, _rotation_body()).json()
    # Sanity: the rotated key authenticates before retirement.
    path = f"/v1/protected/attestations/{attestation['id']}"
    headers = _signed_headers("GET", path, b"", actor="org-1", seed=SEED_NEW)
    assert client.get(path, headers=headers).status_code == 200

    assert _retire(client, created["id"]).status_code == 200

    # The retired key is immediately invalid on both protected routes...
    headers = _signed_headers("GET", path, b"", actor="org-1", seed=SEED_NEW)
    assert client.get(path, headers=headers).status_code == 404
    denied = _post_grant(client, attestation["id"], "org-2",
                         actor="org-1", seed=SEED_NEW)
    assert denied.status_code == 422
    assert denied.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )
    # ...while the still-current attestation key is unaffected.
    headers = _signed_headers("GET", path, b"", actor="org-1", seed=SEED_A)
    assert client.get(path, headers=headers).status_code == 200


def test_retire_only_deactivates_the_target_key(client):
    attestation = _world(client)
    first = _post_rotation(client, _rotation_body(seed=SEED_NEW)).json()
    second = _post_rotation(client, _rotation_body(seed=SEED_EXTRA)).json()
    assert first["id"] != second["id"]

    assert _retire(client, first["id"]).status_code == 200

    # The retired key no longer authenticates...
    denied = _post_grant(client, attestation["id"], "org-2",
                         actor="org-1", seed=SEED_NEW)
    assert denied.status_code == 422
    # ...but the other active rotation key still does.
    granted = _post_grant(client, attestation["id"], "org-2",
                          actor="org-1", seed=SEED_EXTRA)
    assert granted.status_code == 201, granted.text


def test_duplicate_retire_is_422_and_writes_nothing(client, db_session):
    _world(client)
    created = _post_rotation(client, _rotation_body()).json()
    assert _retire(client, created["id"]).status_code == 200
    events_after_retire = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    resp = _retire(client, created["id"])
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert resp.json()["error"]["details"]["reason"] == (
        "rotation_already_retired"
    )
    assert len(_audit_events(db_session, EVENT_AUTHENTICATION_KEY_RETIRED)) == 1
    assert len(db_session.execute(select(AuditEvent)).scalars().all()) == (
        events_after_retire
    )


def test_retire_by_non_subject_is_422_and_writes_nothing(client, db_session):
    _world(client)
    created = _post_rotation(client, _rotation_body()).json()
    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())

    # org-2 authenticates validly but is not the rotation's subject.
    resp = _retire(client, created["id"], actor="org-2", seed=SEED_B)
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "caller_not_subject"

    row = _rotation_rows(db_session)[0]
    assert row.active is True
    assert row.retired_at is None
    assert _audit_events(db_session, EVENT_AUTHENTICATION_KEY_RETIRED) == []
    assert len(db_session.execute(select(AuditEvent)).scalars().all()) == (
        events_before
    )


def test_retire_unknown_rotation_is_422_and_writes_nothing(client, db_session):
    _world(client)
    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())
    resp = _retire(client, "akr_" + "0" * 64)
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == (
        "authentication_key_rotation_not_found"
    )
    assert len(db_session.execute(select(AuditEvent)).scalars().all()) == (
        events_before
    )


def test_retire_requires_valid_credentials(client, db_session):
    _world(client)
    created = _post_rotation(client, _rotation_body()).json()
    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())
    path = f"{ROTATIONS_PATH}/{created['id']}/retire"

    # No headers at all.
    resp = client.post(path)
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "missing_credentials"

    # A key that is not a current key of the caller.
    resp = _retire(client, created["id"], actor="org-1", seed=SEED_B)
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )

    # The signature binds to the exact retire path.
    resp = _retire(client, created["id"], signed_path=ROTATIONS_PATH)
    assert resp.status_code == 422

    row = _rotation_rows(db_session)[0]
    assert row.active is True
    assert len(db_session.execute(select(AuditEvent)).scalars().all()) == (
        events_before
    )


# --- Storage hygiene and persistence ----------------------------------------------


def test_no_private_key_or_signature_material_is_stored(client, db_session):
    _world(client)
    created = _post_rotation(client, _rotation_body()).json()

    # The persisted row carries only the public key bytes -- no column can
    # hold a private key or a signature.
    row = _rotation_rows(db_session)[0]
    columns = {column.name for column in row.__table__.columns}
    assert columns == {
        "seq", "id", "actor_id", "public_key",
        "active", "created_at", "retired_at",
    }
    assert row.public_key == ed25519_public_key(SEED_NEW)
    # The response exposes no signature or private material either.
    assert "signature" not in created
    assert "private_key" not in created


def test_rotations_persist_across_app_restarts(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    attestation = _world(file_client)
    created = _post_rotation(file_client, _rotation_body()).json()
    retired = _retire(file_client, created["id"]).json()
    assert retired["active"] is False

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        # The retired key still does not authenticate after a restart.
        path = f"/v1/protected/attestations/{attestation['id']}"
        headers = _signed_headers("GET", path, b"", actor="org-1", seed=SEED_NEW)
        assert client.get(path, headers=headers).status_code == 404
        # The non-revoked attestation key still does.
        headers = _signed_headers("GET", path, b"", actor="org-1", seed=SEED_A)
        assert client.get(path, headers=headers).status_code == 200

        import sqlite3

        con = sqlite3.connect(tmp_db_url.removeprefix("sqlite:///"))
        rows = con.execute(
            "SELECT id, actor_id, active, retired_at IS NOT NULL "
            "FROM authentication_key_rotations"
        ).fetchall()
        audit = con.execute(
            "SELECT event_type FROM audit_events WHERE resource_id = ? "
            "ORDER BY created_at, seq",
            (created["id"],),
        ).fetchall()
        con.close()
        assert rows == [(created["id"], "org-1", 0, 1)]
        assert [event for (event,) in audit] == [
            EVENT_AUTHENTICATION_KEY_ROTATED,
            EVENT_AUTHENTICATION_KEY_RETIRED,
        ]
