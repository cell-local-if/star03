"""Tests for authentication public-key rotation and retirement.

Covers ``POST /v1/authentication-key-rotations`` and
``POST /v1/authentication-key-rotations/{rotation_id}/retire``, including:

* the shared ``X-PA``/``X-PT``/``X-PS`` access authentication (a subject
  bootstraps with an existing non-revoked attestation key);
* first creation (201) with the exact public fields, the stable ``akr_``
  id, and a single-transaction ``authentication_key.rotated`` audit row;
* 200 idempotency for the same subject and public key (including after
  retirement), independent records for distinct keys and distinct
  subjects, and no audit event on a retry;
* the rotated key immediately joining the subject's non-revoked
  authentication set without any attestation, existing attestation keys
  remaining valid, and a retired key failing immediately;
* subject-only retirement of an active record with the
  ``authentication_key.retired`` audit event, the empty-body requirement,
  and opaque 422s for an unknown record, a non-owner, and a repeat retire;
* request/credential validation failures writing nothing, persistence
  across restarts, and a concurrent-create race;
* the absence of any stored private key or raw signature.

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures); only fixed seed-derived public keys are used.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from provenance.access_signing import access_message_bytes
from provenance.ids import authentication_key_rotation_id
from provenance.models import (
    EVENT_AUTHENTICATION_KEY_RETIRED,
    EVENT_AUTHENTICATION_KEY_ROTATED,
    AuthenticationKeyRotation,
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

ROTATIONS_PATH = "/v1/authentication-key-rotations"
# Deterministic seeds for rotated keys; these keys start with NO attestation.
SEED_R1 = b"test-ed25519-rotate-r1-000000000000"[:32]
SEED_R2 = b"test-ed25519-rotate-r2-000000000000"[:32]


# --- World setup ---------------------------------------------------------------


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
    """Two actors, each bootstrapped with one non-revoked attestation key."""
    create_actor(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    att1 = _make_attestation(
        client, "org-1", SEED_A, _make_claim(client, "org-1")["id"]
    )
    att2 = _make_attestation(
        client, "org-2", SEED_B,
        _make_claim(client, "org-2", digest=DIGEST_B)["id"],
    )
    return att1, att2


# --- Signed-request helpers ---------------------------------------------------


def _signed_headers(method, path, body, *, actor, seed, timestamp=None,
                    signed_timestamp=None, signed_method=None,
                    signed_path=None, signed_body=None):
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


def _key_b64(seed) -> str:
    return base64.b64encode(ed25519_public_key(seed)).decode()


def _rotation_body(actor_id="org-1", seed=SEED_R1):
    return {"actor_id": actor_id, "new_public_key": _key_b64(seed)}


def _post_rotation(client, body_obj, *, actor="org-1", seed=SEED_A, raw=None,
                   **sign_kwargs):
    body = raw if raw is not None else json.dumps(body_obj).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", ROTATIONS_PATH, body,
                          actor=actor, seed=seed, **sign_kwargs),
    }
    return client.post(ROTATIONS_PATH, content=body, headers=headers)


def _retire_path(rotation_id):
    return f"{ROTATIONS_PATH}/{rotation_id}/retire"


def _post_retire(client, rotation_id, *, actor="org-1", seed=SEED_A,
                 body=b"", path=None, **sign_kwargs):
    path = path or _retire_path(rotation_id)
    headers = _signed_headers("POST", path, body, actor=actor, seed=seed,
                              **sign_kwargs)
    return client.post(path, content=body, headers=headers)


def _protected_get(client, attestation_id, *, actor, seed):
    path = f"/v1/protected/attestations/{attestation_id}"
    headers = _signed_headers("GET", path, b"", actor=actor, seed=seed)
    return client.get(path, headers=headers)


def _audit_events(session):
    return session.execute(select(AuditEvent)).scalars().all()


def _rotation_by_id(session, rotation_id):
    return session.execute(
        select(AuthenticationKeyRotation).where(
            AuthenticationKeyRotation.id == rotation_id
        )
    ).scalar_one_or_none()


# --- First creation ------------------------------------------------------------


def test_first_rotation_returns_201_with_exact_public_fields(client):
    _world(client)
    resp = _post_rotation(client, _rotation_body())
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == {
        "id", "actor_id", "new_public_key", "active", "created_at", "retired_at"
    }
    assert body["id"].startswith("akr_")
    assert len(body["id"]) == len("akr_") + 64
    assert body["actor_id"] == "org-1"
    assert body["new_public_key"] == _key_b64(SEED_R1)
    assert body["active"] is True
    assert body["retired_at"] is None
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0


def test_rotation_id_is_the_stable_pair_derived_identifier(client):
    _world(client)
    body = _post_rotation(client, _rotation_body()).json()
    assert body["id"] == authentication_key_rotation_id(
        "org-1", ed25519_public_key(SEED_R1).hex()
    )


def test_rotation_and_rotated_audit_commit_in_one_transaction(client, db_session):
    _world(client)
    created = _post_rotation(client, _rotation_body()).json()

    rows = db_session.execute(
        select(AuthenticationKeyRotation)
    ).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert (row.id, row.actor_id, row.active, row.retired_at) == (
        created["id"], "org-1", True, None
    )
    assert row.public_key == ed25519_public_key(SEED_R1)
    assert len(row.public_key) == 32
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_AUTHENTICATION_KEY_ROTATED
        )
    ).scalars().all()
    assert [e.resource_id for e in events] == [created["id"]]
    assert events[0].created_at.utcoffset().total_seconds() == 0


def test_no_private_key_or_signature_material_is_persisted(tmp_db_url, file_client):
    _world(file_client)
    created = _post_rotation(file_client, _rotation_body()).json()
    con = sqlite3.connect(tmp_db_url.removeprefix("sqlite:///"))
    try:
        cols = {
            row[1]
            for row in con.execute("PRAGMA table_info(authentication_key_rotations)")
        }
        assert cols == {
            "seq", "id", "actor_id", "public_key", "active",
            "created_at", "retired_at",
        }
        assert "private_key" not in cols and "signature" not in cols
        stored = con.execute(
            "SELECT public_key FROM authentication_key_rotations WHERE id = ?",
            (created["id"],),
        ).fetchone()[0]
        assert stored == ed25519_public_key(SEED_R1)
    finally:
        con.close()


# --- The rotated key immediately authenticates, without an attestation ---------


def test_rotated_key_authenticates_protected_routes_without_an_attestation(
    client, db_session
):
    att1, _ = _world(client)
    # SEED_R1 has no attestation anywhere; the rotation is the only binding.
    created = _post_rotation(client, _rotation_body(seed=SEED_R1))
    assert created.status_code == 201

    # The rotated key reads the subject's own protected attestation.
    read = _protected_get(client, att1["id"], actor="org-1", seed=SEED_R1)
    assert read.status_code == 200, read.text
    assert read.json() == att1

    # And it can create an access grant (the caller is still the signer).
    grant_body = json.dumps(
        {"attestation_id": att1["id"], "grantee_actor_id": "org-2"}
    ).encode()
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", "/v1/attestation-access-grants", grant_body,
                          actor="org-1", seed=SEED_R1),
    }
    grant = client.post(
        "/v1/attestation-access-grants", content=grant_body, headers=headers
    )
    assert grant.status_code == 201, grant.text


def test_rotated_key_can_authenticate_a_further_rotation(client):
    _world(client)
    assert _post_rotation(client, _rotation_body(seed=SEED_R1)).status_code == 201
    # The brand-new SEED_R2 key is introduced using only SEED_R1, which
    # itself has no attestation: the chain never returns to SEED_A.
    second = _post_rotation(
        client, _rotation_body(seed=SEED_R2), seed=SEED_R1
    )
    assert second.status_code == 201, second.text
    assert second.json()["new_public_key"] == _key_b64(SEED_R2)


def test_existing_attestation_key_still_authenticates_after_rotation(client):
    att1, _ = _world(client)
    assert _post_rotation(client, _rotation_body()).status_code == 201
    # The pre-rotation attestation key remains in the non-revoked set.
    assert _protected_get(client, att1["id"], actor="org-1",
                          seed=SEED_A).status_code == 200


def test_subject_without_any_current_key_cannot_rotate(client, db_session):
    _world(client)
    # org-1's key cannot speak for a brand-new actor with no attestation.
    create_actor(client, actor_id="org-9", name="No Keys", type="device")
    events_before = len(_audit_events(db_session))
    resp = _post_rotation(
        client, _rotation_body(actor_id="org-9", seed=SEED_R1),
        actor="org-9", seed=SEED_R1,
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )
    assert db_session.execute(
        select(AuthenticationKeyRotation)
    ).scalars().all() == []
    assert len(_audit_events(db_session)) == events_before


# --- Idempotency and independence ----------------------------------------------


def test_same_subject_and_key_retry_returns_200_original_and_no_audit(
    client, db_session
):
    _world(client)
    first = _post_rotation(client, _rotation_body())
    assert first.status_code == 201
    events_after_first = len(_audit_events(db_session))

    for _ in range(3):
        retry = _post_rotation(client, _rotation_body())
        assert retry.status_code == 200
        assert retry.json() == first.json()

    rows = db_session.execute(select(AuthenticationKeyRotation)).scalars().all()
    assert [r.id for r in rows] == [first.json()["id"]]
    rotated = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_AUTHENTICATION_KEY_ROTATED
        )
    ).scalars().all()
    assert len(rotated) == 1
    assert len(_audit_events(db_session)) == events_after_first


def test_distinct_keys_for_one_subject_form_independent_records(client):
    _world(client)
    one = _post_rotation(client, _rotation_body(seed=SEED_R1))
    two = _post_rotation(
        client, _rotation_body(seed=SEED_R2), seed=SEED_R1
    )
    assert one.status_code == 201
    assert two.status_code == 201
    assert one.json()["id"] != two.json()["id"]

    # Each pair retries to its own original record.
    assert _post_rotation(client, _rotation_body(seed=SEED_R1)).json()["id"] == (
        one.json()["id"]
    )
    assert _post_rotation(
        client, _rotation_body(seed=SEED_R2), seed=SEED_R1
    ).json()["id"] == two.json()["id"]


def test_same_key_bytes_for_distinct_subjects_form_independent_records(client):
    _world(client)
    first = _post_rotation(client, _rotation_body(actor_id="org-1", seed=SEED_R1))
    # org-2 introduces the identical 32 key bytes under its own identity.
    second = _post_rotation(
        client, _rotation_body(actor_id="org-2", seed=SEED_R1),
        actor="org-2", seed=SEED_B,
    )
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] != second.json()["id"]
    assert first.json()["new_public_key"] == second.json()["new_public_key"]


def test_retry_after_retirement_returns_original_retired_record_no_audit(
    client, db_session
):
    _world(client)
    first = _post_rotation(client, _rotation_body()).json()
    assert _post_retire(client, first["id"]).status_code == 200
    events_after_retire = len(_audit_events(db_session))

    retry = _post_rotation(client, _rotation_body())
    assert retry.status_code == 200
    body = retry.json()
    assert body["id"] == first["id"]
    assert body["active"] is False
    assert body["retired_at"] is not None
    # A post-retirement retry resurrects nothing and writes no audit.
    assert len(_audit_events(db_session)) == events_after_retire


# --- Retirement ----------------------------------------------------------------


def test_retire_success_returns_inactive_record_with_utc_retired_at(client):
    _world(client)
    created = _post_rotation(client, _rotation_body()).json()
    resp = _post_retire(client, created["id"], seed=SEED_R1)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == created["id"]
    assert body["actor_id"] == "org-1"
    assert body["new_public_key"] == created["new_public_key"]
    assert body["active"] is False
    retired_at = datetime.fromisoformat(body["retired_at"])
    assert retired_at.utcoffset().total_seconds() == 0
    assert retired_at >= datetime.fromisoformat(created["created_at"])


def test_retire_writes_retired_audit_in_the_same_transaction(client, db_session):
    _world(client)
    created = _post_rotation(client, _rotation_body()).json()
    assert _post_retire(client, created["id"]).status_code == 200

    row = db_session.execute(
        select(AuthenticationKeyRotation).where(
            AuthenticationKeyRotation.id == created["id"]
        )
    ).scalar_one()
    assert row.active is False
    assert row.retired_at is not None
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_AUTHENTICATION_KEY_RETIRED
        )
    ).scalars().all()
    assert [e.resource_id for e in events] == [created["id"]]


def test_retired_key_fails_authentication_immediately_but_other_keys_remain(client):
    att1, _ = _world(client)
    created = _post_rotation(client, _rotation_body(seed=SEED_R1)).json()
    assert _protected_get(client, att1["id"], actor="org-1",
                          seed=SEED_R1).status_code == 200

    # Retire authenticated by the rotated key itself; zero-byte body.
    assert _post_retire(client, created["id"], seed=SEED_R1).status_code == 200

    # A protected WRITE with the dead key is a 422 verification failure...
    dead_write = _post_rotation(
        client, _rotation_body(seed=SEED_R2), seed=SEED_R1
    )
    assert dead_write.status_code == 422
    assert dead_write.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )
    # ...and a protected READ collapses the same failure into the opaque 404.
    dead_read = _protected_get(client, att1["id"], actor="org-1", seed=SEED_R1)
    assert dead_read.status_code == 404
    assert dead_read.json()["error"]["code"] == "not_found"

    # The original attestation key is unaffected and still current.
    assert _protected_get(client, att1["id"], actor="org-1",
                          seed=SEED_A).status_code == 200
    # A second, still-active rotation key also continues to authenticate.
    second = _post_rotation(
        client, _rotation_body(seed=SEED_R2), seed=SEED_A
    ).json()
    assert _protected_get(client, att1["id"], actor="org-1",
                          seed=SEED_R2).status_code == 200
    assert _post_retire(client, second["id"], seed=SEED_R2).status_code == 200
    assert _protected_get(client, att1["id"], actor="org-1",
                          seed=SEED_R2).status_code == 404
    # The attestation key still authenticates after both rotations retire.
    assert _protected_get(client, att1["id"], actor="org-1",
                          seed=SEED_A).status_code == 200


def test_retire_requires_an_empty_body(client, db_session):
    _world(client)
    created = _post_rotation(client, _rotation_body()).json()
    events_before = len(_audit_events(db_session))

    # A properly-signed but non-empty body is a validation error.
    resp = _post_retire(client, created["id"], body=b"{}")
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "body_must_be_empty"

    # The record is untouched and no audit event was written.
    assert len(_audit_events(db_session)) == events_before
    again = _post_retire(client, created["id"])
    assert again.status_code == 200
    assert again.json()["active"] is False


def test_only_the_owning_subject_may_retire_and_existence_is_not_revealed(
    client, db_session
):
    _world(client)
    created = _post_rotation(client, _rotation_body()).json()
    events_before = len(_audit_events(db_session))

    # org-2 is a real, authenticated actor but not the rotation's subject.
    wrong_owner = _post_retire(
        client, created["id"], actor="org-2", seed=SEED_B
    )
    assert wrong_owner.status_code == 422
    assert wrong_owner.json()["error"]["code"] == "validation_error"
    assert wrong_owner.json()["error"]["details"]["reason"] == (
        "rotation_not_found"
    )
    # The non-owner sees exactly the same body as for an unknown id.
    unknown = _post_retire(
        client, "akr_ghost", actor="org-2", seed=SEED_B
    )
    assert unknown.status_code == 422
    assert unknown.json() == wrong_owner.json()

    # No state change, no audit.
    row = _rotation_by_id(db_session, created["id"])
    assert row.active is True and row.retired_at is None
    assert len(_audit_events(db_session)) == events_before

    # The owner can still retire it afterwards.
    assert _post_retire(client, created["id"]).status_code == 200


def test_unknown_rotation_id_is_422_and_writes_nothing(client, db_session):
    _world(client)
    events_before = len(_audit_events(db_session))
    resp = _post_retire(client, "akr_does_not_exist")
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "rotation_not_found"
    assert len(_audit_events(db_session)) == events_before
    assert db_session.execute(
        select(AuthenticationKeyRotation)
    ).scalars().all() == []


def test_repeat_retirement_is_422_and_writes_no_second_audit(client, db_session):
    _world(client)
    created = _post_rotation(client, _rotation_body()).json()
    first = _post_retire(client, created["id"])
    assert first.status_code == 200
    first_retired_at = first.json()["retired_at"]
    events_after_first = len(_audit_events(db_session))

    for _ in range(2):
        repeat = _post_retire(client, created["id"])
        assert repeat.status_code == 422
        assert repeat.json()["error"]["details"]["reason"] == (
            "rotation_already_retired"
        )

    retired_events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_AUTHENTICATION_KEY_RETIRED
        )
    ).scalars().all()
    assert len(retired_events) == 1
    assert len(_audit_events(db_session)) == events_after_first
    db_session.expire_all()
    row = _rotation_by_id(db_session, created["id"])
    # The stamped retirement time from the first call is preserved exactly.
    assert row.retired_at == datetime.fromisoformat(first_retired_at)


# --- Credential failures on the protected rotation routes ----------------------


def test_rotation_without_credentials_is_422(client, db_session):
    _world(client)
    events_before = len(_audit_events(db_session))
    body = json.dumps(_rotation_body()).encode()
    resp = client.post(
        ROTATIONS_PATH, content=body,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "missing_credentials"
    assert len(_audit_events(db_session)) == events_before


def test_rotation_wrong_signature_is_422(client, db_session):
    _world(client)
    resp = _post_rotation(
        client, _rotation_body(), actor="org-1", seed=SEED_B
    )
    # SEED_B is not a current key of org-1 (and no rotation exists yet).
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )


def test_retire_wrong_signature_is_422_and_writes_nothing(client, db_session):
    _world(client)
    created = _post_rotation(client, _rotation_body()).json()
    events_before = len(_audit_events(db_session))
    resp = _post_retire(client, created["id"], actor="org-1", seed=SEED_B)
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )
    row = _rotation_by_id(db_session, created["id"])
    assert row.active is True
    assert len(_audit_events(db_session)) == events_before


def test_rotation_rejects_stale_and_malformed_timestamps(client):
    _world(client)
    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    for raw, reason in (
        (stale, "timestamp_out_of_window"),
        ("2026-09-20T12:00:00", "invalid_timestamp"),
        ("not-a-timestamp", "invalid_timestamp"),
    ):
        resp = _post_rotation(client, _rotation_body(), timestamp=raw)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["details"]["reason"] == reason, raw


def test_rotation_signature_binds_to_method_path_and_body(client):
    _world(client)
    assert _post_rotation(
        client, _rotation_body(), signed_method="GET"
    ).status_code == 422
    assert _post_rotation(
        client, _rotation_body(), signed_path="/v1/attestations"
    ).status_code == 422
    assert _post_rotation(
        client, _rotation_body(), signed_body=b'{"actor_id":"x"}'
    ).status_code == 422


def test_malformed_credentials_are_rejected_before_retire_lookup(client, db_session):
    # A malformed timestamp is a credential 422 whether or not the record
    # exists; nothing is written either way.
    _world(client)
    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    resp = _post_retire(client, "akr_ghost", timestamp=stale)
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == (
        "timestamp_out_of_window"
    )


# --- Request-body validation ---------------------------------------------------


def test_rotation_body_validation_failures_are_422_and_write_nothing(
    client, db_session
):
    _world(client)
    events_before = len(_audit_events(db_session))
    valid_key = _key_b64(SEED_R1)

    def post_obj(obj):
        return _post_rotation(client, obj)

    blank_actor = post_obj({"actor_id": "  ", "new_public_key": valid_key})
    missing_actor = post_obj({"new_public_key": valid_key})
    missing_key = post_obj({"actor_id": "org-1"})
    extra = post_obj(
        {"actor_id": "org-1", "new_public_key": valid_key, "nope": True}
    )
    not_b64 = post_obj({"actor_id": "org-1", "new_public_key": "@@@"})
    short = post_obj(
        {"actor_id": "org-1",
         "new_public_key": base64.b64encode(b"k" * 31).decode()}
    )
    long = post_obj(
        {"actor_id": "org-1",
         "new_public_key": base64.b64encode(b"k" * 33).decode()}
    )
    non_string = post_obj({"actor_id": "org-1", "new_public_key": 1234})
    # 32 bytes whose standard Base64 contains '+' and '/': the urlsafe
    # alphabet ('-'/'_') must be rejected rather than silently accepted.
    urlsafe_raw = bytes([0xFB, 0xFF]) * 16
    urlsafe_value = base64.urlsafe_b64encode(urlsafe_raw).decode()
    assert urlsafe_value != base64.b64encode(urlsafe_raw).decode()
    urlsafe = post_obj({"actor_id": "org-1", "new_public_key": urlsafe_value})
    for resp in (blank_actor, missing_actor, missing_key, extra, not_b64,
                 short, long, non_string, urlsafe):
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "validation_error"

    # Malformed JSON is rejected at the boundary even with a signature over
    # those exact bytes.
    raw = b"{not valid json"
    malformed = _post_rotation(client, None, raw=raw)
    assert malformed.status_code == 422

    assert db_session.execute(
        select(AuthenticationKeyRotation)
    ).scalars().all() == []
    assert len(_audit_events(db_session)) == events_before


def test_rotation_for_unknown_subject_is_422_and_writes_nothing(client, db_session):
    _world(client)
    events_before = len(_audit_events(db_session))
    # Authenticated as org-1 (valid signature), naming an unknown subject.
    resp = _post_rotation(
        client, _rotation_body(actor_id="ghost", seed=SEED_R1)
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "unknown_actor"
    assert db_session.execute(
        select(AuthenticationKeyRotation)
    ).scalars().all() == []
    assert len(_audit_events(db_session)) == events_before


def test_rotation_caller_must_match_the_body_subject(client, db_session):
    _world(client)
    events_before = len(_audit_events(db_session))
    # org-1 authenticates but the body names org-2.
    resp = _post_rotation(
        client, _rotation_body(actor_id="org-2", seed=SEED_R1)
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "actor_mismatch"
    assert db_session.execute(
        select(AuthenticationKeyRotation)
    ).scalars().all() == []
    assert len(_audit_events(db_session)) == events_before


# --- Persistence and concurrency -----------------------------------------------


def test_rotations_survive_restart_and_retirement_is_durable(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    _world(file_client)
    created = _post_rotation(
        file_client, _rotation_body(seed=SEED_R1)
    ).json()

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        # The rotated key still authenticates after a restart.
        path = f"{ROTATIONS_PATH}/{created['id']}/retire"
        retire = _post_retire(client, created["id"], seed=SEED_R1)
        assert retire.status_code == 200
        assert retire.json()["active"] is False

    restarted_again = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted_again) as client:
        # The retired key must stay dead after another restart (a protected
        # read renders the unauthenticated signature as the opaque 404).
        att1 = client.get("/v1/attestations").json()["items"]
        att1 = next(a for a in att1 if a["signer_actor_id"] == "org-1")
        dead = _protected_get(client, att1["id"], actor="org-1", seed=SEED_R1)
        assert dead.status_code == 404
        alive = _protected_get(client, att1["id"], actor="org-1", seed=SEED_A)
        assert alive.status_code == 200


def test_concurrent_identical_rotations_yield_one_record_and_audit(
    tmp_db_url, file_app, file_client
):
    _world(file_client)

    from provenance import service
    from provenance.schemas import AuthenticationKeyRotationCreate

    factory = file_app.state.session_factory
    payload = AuthenticationKeyRotationCreate(
        actor_id="org-1", new_public_key=_key_b64(SEED_R1)
    )
    results: list[tuple[str, bool]] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        session = factory()
        try:
            barrier.wait()
            rotation, created = service.create_authentication_key_rotation(
                session, payload, "org-1"
            )
            results.append((rotation.id, created))
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
            select(AuthenticationKeyRotation)
        ).scalars().all()
        events = audit_session.execute(
            select(AuditEvent).where(
                AuditEvent.event_type == EVENT_AUTHENTICATION_KEY_ROTATED
            )
        ).scalars().all()
        assert [r.id for r in rows] == [results[0][0]]
        assert [e.resource_id for e in events] == [results[0][0]]
    finally:
        audit_session.close()


def test_concurrent_retirements_yield_one_flip_and_one_retired_audit(
    tmp_db_url, file_app, file_client
):
    from provenance import service

    _world(file_client)
    created = _post_rotation(
        file_client, _rotation_body(seed=SEED_R1)
    ).json()
    factory = file_app.state.session_factory

    outcomes: list[str] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        session = factory()
        try:
            barrier.wait()
            service.retire_authentication_key_rotation(
                session, created["id"], "org-1"
            )
            outcomes.append("retired")
        except Exception:
            outcomes.append("rejected")
        finally:
            session.close()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert outcomes.count("retired") == 1
    assert outcomes.count("rejected") == 3

    audit_session = factory()
    try:
        row = audit_session.execute(
            select(AuthenticationKeyRotation).where(
                AuthenticationKeyRotation.id == created["id"]
            )
        ).scalar_one()
        assert row.active is False
        assert row.retired_at is not None
        retired_events = audit_session.execute(
            select(AuditEvent).where(
                AuditEvent.event_type == EVENT_AUTHENTICATION_KEY_RETIRED
            )
        ).scalars().all()
        assert [e.resource_id for e in retired_events] == [created["id"]]
    finally:
        audit_session.close()
