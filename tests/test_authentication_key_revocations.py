"""Tests for emergency authentication public-key revocation.

Covers ``POST /v1/authentication-key-revocations`` and
``GET /v1/authentication-key-revocations/{revocation_id}``, including:

* the shared ``X-PA``/``X-PT``/``X-PS`` access authentication, with the key
  being revoked allowed to sign its own revocation request;
* first creation (201) with the exact public fields, the stable ``akv_``
  id, the UTC ``revoked_at``, and a single-transaction
  ``authentication_key.revoked`` audit row;
* 200 idempotency for the same subject/key/reason triple, a 409
  ``authentication_key_revocation_conflict`` for the same subject and key
  with a different reason, and exactly one record and one audit event
  under concurrent identical submissions;
* the revoked key bytes immediately leaving the subject's authentication
  set for protected reads and writes -- whether carried by a non-revoked
  attestation or an active rotation -- while other keys keep working and a
  rotation idempotent replay cannot restore eligibility;
* every input, identity, authentication, or unknown-target failure
  rendering as ``422 validation_error`` distinguished only by
  ``details.reason`` and writing no state;
* the by-id read returning the record and rendering an unknown id as the
  ``authentication_key_revocation_not_found`` 404;
* attestation, trust, and grant semantics staying unchanged, persistence
  across restarts, and the absence of any stored private key or raw
  signature.

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures); only fixed seed-derived public keys are used.
"""

from __future__ import annotations

import base64
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from provenance.ids import authentication_key_revocation_id
from provenance.models import (
    EVENT_ATTESTATION_REVOKED,
    EVENT_AUTHENTICATION_KEY_REVOKED,
    AuthenticationKeyRevocation,
    AuditEvent,
)
from tests.helpers import SEED_A, SEED_B, ed25519_public_key
from tests.test_authentication_key_rotations import (
    ROTATIONS_PATH,
    SEED_R1,
    SEED_R2,
    _key_b64,
    _post_retire,
    _post_rotation,
    _protected_get,
    _rotation_body,
    _signed_headers,
    _world,
)

REVOCATIONS_PATH = "/v1/authentication-key-revocations"


# --- Signed-request helpers ---------------------------------------------------


def _revocation_body(actor_id="org-1", seed=SEED_R1, reason="compromised"):
    return {"actor_id": actor_id, "public_key": _key_b64(seed), "reason": reason}


def _post_revocation(client, body_obj, *, actor="org-1", seed=SEED_A, raw=None,
                     **sign_kwargs):
    body = raw if raw is not None else json.dumps(body_obj).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", REVOCATIONS_PATH, body,
                          actor=actor, seed=seed, **sign_kwargs),
    }
    return client.post(REVOCATIONS_PATH, content=body, headers=headers)


def _get_revocation(client, revocation_id):
    return client.get(f"{REVOCATIONS_PATH}/{revocation_id}")


def _audit_events(session):
    return session.execute(select(AuditEvent)).scalars().all()


def _revocations(session):
    return session.execute(select(AuthenticationKeyRevocation)).scalars().all()


def _rotate(client, seed=SEED_R1, *, actor="org-1", sign_seed=SEED_A):
    resp = _post_rotation(
        client, _rotation_body(actor_id=actor, seed=seed),
        actor=actor, seed=sign_seed,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# --- First creation ------------------------------------------------------------


def test_first_revocation_returns_201_with_exact_public_fields(client):
    _world(client)
    _rotate(client, SEED_R1)
    resp = _post_revocation(client, _revocation_body())
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == {"id", "actor_id", "public_key", "reason", "revoked_at"}
    assert body["id"].startswith("akv_")
    assert len(body["id"]) == len("akv_") + 64
    assert body["actor_id"] == "org-1"
    assert body["public_key"] == _key_b64(SEED_R1)
    assert body["reason"] == "compromised"
    revoked_at = datetime.fromisoformat(body["revoked_at"])
    assert revoked_at.utcoffset().total_seconds() == 0


def test_each_supported_reason_is_accepted(client):
    _world(client)
    _rotate(client, SEED_R1)
    _rotate(client, SEED_R2, sign_seed=SEED_R1)
    for seed, reason in (
        (SEED_R1, "compromised"),
        (SEED_R2, "superseded"),
        (SEED_A, "policy"),
    ):
        resp = _post_revocation(client, _revocation_body(seed=seed, reason=reason))
        assert resp.status_code == 201, resp.text
        assert resp.json()["reason"] == reason


def test_revocation_id_is_the_stable_triple_derived_identifier(client):
    _world(client)
    _rotate(client, SEED_R1)
    body = _post_revocation(client, _revocation_body()).json()
    assert body["id"] == authentication_key_revocation_id(
        "org-1", ed25519_public_key(SEED_R1).hex(), "compromised"
    )


def test_revocation_and_revoked_audit_commit_in_one_transaction(
    client, db_session
):
    _world(client)
    _rotate(client, SEED_R1)
    created = _post_revocation(client, _revocation_body()).json()

    rows = _revocations(db_session)
    assert len(rows) == 1
    row = rows[0]
    assert (row.id, row.actor_id, row.reason) == (
        created["id"], "org-1", "compromised"
    )
    assert row.public_key == ed25519_public_key(SEED_R1)
    assert len(row.public_key) == 32
    assert row.revoked_at.utcoffset().total_seconds() == 0
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_AUTHENTICATION_KEY_REVOKED
        )
    ).scalars().all()
    assert [e.resource_id for e in events] == [created["id"]]
    assert events[0].created_at.utcoffset().total_seconds() == 0


def test_no_private_key_or_signature_material_is_persisted(
    tmp_db_url, file_client
):
    _world(file_client)
    _rotate(file_client, SEED_R1)
    created = _post_revocation(file_client, _revocation_body()).json()
    con = sqlite3.connect(tmp_db_url.removeprefix("sqlite:///"))
    try:
        cols = {
            row[1]
            for row in con.execute(
                "PRAGMA table_info(authentication_key_revocations)"
            )
        }
        assert cols == {
            "seq", "id", "actor_id", "public_key", "reason", "revoked_at",
        }
        assert "private_key" not in cols and "signature" not in cols
        stored = con.execute(
            "SELECT public_key FROM authentication_key_revocations"
            " WHERE id = ?",
            (created["id"],),
        ).fetchone()[0]
        assert stored == ed25519_public_key(SEED_R1)
    finally:
        con.close()


# --- The revoked key immediately leaves the authentication set ------------------


def test_revoked_rotation_key_fails_immediately_but_other_keys_remain(client):
    att1, _ = _world(client)
    _rotate(client, SEED_R1)
    assert _protected_get(client, att1["id"], actor="org-1",
                          seed=SEED_R1).status_code == 200

    assert _post_revocation(client, _revocation_body()).status_code == 201

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


def test_revoked_attestation_key_fails_even_though_attestation_survives(
    client, db_session
):
    att1, _ = _world(client)
    # Revoke the very key carried by org-1's non-revoked attestation.
    resp = _post_revocation(client, _revocation_body(seed=SEED_A))
    assert resp.status_code == 201, resp.text

    # The key bytes left the authentication set...
    assert _protected_get(client, att1["id"], actor="org-1",
                          seed=SEED_A).status_code == 404
    # ...but the attestation itself is untouched: no attestation.revoked
    # audit event, and the proof still reads back through the public route.
    assert db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ATTESTATION_REVOKED
        )
    ).scalars().all() == []
    assert client.get(f"/v1/attestations/{att1['id']}").status_code == 200


def test_grant_authorization_is_unchanged_by_an_emergency_revocation(client):
    att1, _ = _world(client)
    # org-1 grants org-2 read access before the emergency revocation.
    grant_body = json.dumps(
        {"attestation_id": att1["id"], "grantee_actor_id": "org-2"}
    ).encode()
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", "/v1/attestation-access-grants", grant_body,
                          actor="org-1", seed=SEED_A),
    }
    grant = client.post(
        "/v1/attestation-access-grants", content=grant_body, headers=headers
    )
    assert grant.status_code == 201, grant.text

    # org-1's only key is emergency-revoked; org-2's grant-based read is
    # unaffected.
    assert _post_revocation(client, _revocation_body(seed=SEED_A)).status_code == 201
    read = _protected_get(client, att1["id"], actor="org-2", seed=SEED_B)
    assert read.status_code == 200, read.text
    assert read.json() == att1


def test_key_being_revoked_may_sign_its_own_revocation(client):
    _world(client)
    _rotate(client, SEED_R1)
    # The compromised key itself authenticates the emergency revocation.
    resp = _post_revocation(client, _revocation_body(), seed=SEED_R1)
    assert resp.status_code == 201, resp.text
    assert resp.json()["public_key"] == _key_b64(SEED_R1)


def test_rotation_idempotent_replay_cannot_restore_a_revoked_key(client):
    att1, _ = _world(client)
    rotation = _rotate(client, SEED_R1)
    assert _post_revocation(client, _revocation_body()).status_code == 201

    # Re-submitting the same rotation is the existing idempotent 200 of the
    # original record; it resurrects nothing.
    replay = _post_rotation(client, _rotation_body(seed=SEED_R1))
    assert replay.status_code == 200
    assert replay.json()["id"] == rotation["id"]
    assert replay.json()["active"] is True
    assert _protected_get(client, att1["id"], actor="org-1",
                          seed=SEED_R1).status_code == 404


def test_same_key_bytes_revoked_for_one_subject_do_not_affect_another(client):
    _, att2 = _world(client)
    _rotate(client, SEED_R1)
    # org-2 introduces the identical 32 key bytes under its own identity.
    _post_rotation(
        client, _rotation_body(actor_id="org-2", seed=SEED_R1),
        actor="org-2", seed=SEED_B,
    )
    assert _post_revocation(client, _revocation_body()).status_code == 201

    # org-2's distinct revocation-free binding of the same bytes still
    # authenticates org-2.
    assert _protected_get(client, att2["id"], actor="org-2",
                          seed=SEED_R1).status_code == 200


# --- Idempotency, conflict, and concurrency -------------------------------------


def test_same_triple_retry_returns_200_original_and_no_audit(
    client, db_session
):
    _world(client)
    _rotate(client, SEED_R1)
    first = _post_revocation(client, _revocation_body())
    assert first.status_code == 201
    events_after_first = len(_audit_events(db_session))

    for _ in range(3):
        retry = _post_revocation(client, _revocation_body())
        assert retry.status_code == 200
        assert retry.json() == first.json()

    assert [r.id for r in _revocations(db_session)] == [first.json()["id"]]
    revoked = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_AUTHENTICATION_KEY_REVOKED
        )
    ).scalars().all()
    assert len(revoked) == 1
    assert len(_audit_events(db_session)) == events_after_first


def test_same_key_with_a_different_reason_is_a_409_conflict(
    client, db_session
):
    _world(client)
    _rotate(client, SEED_R1)
    first = _post_revocation(client, _revocation_body(reason="compromised"))
    assert first.status_code == 201
    events_after_first = len(_audit_events(db_session))

    conflict = _post_revocation(client, _revocation_body(reason="policy"))
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == (
        "authentication_key_revocation_conflict"
    )

    # The conflict wrote nothing: one record, one audit event, and the
    # original reason is preserved.
    rows = _revocations(db_session)
    assert [r.id for r in rows] == [first.json()["id"]]
    assert rows[0].reason == "compromised"
    assert len(_audit_events(db_session)) == events_after_first

    # The original triple still idempotently returns the first record.
    retry = _post_revocation(client, _revocation_body(reason="compromised"))
    assert retry.status_code == 200
    assert retry.json() == first.json()


def test_distinct_keys_and_subjects_form_independent_revocations(client):
    _world(client)
    _rotate(client, SEED_R1)
    _rotate(client, SEED_R2, sign_seed=SEED_R1)
    one = _post_revocation(client, _revocation_body(seed=SEED_R1))
    two = _post_revocation(
        client, _revocation_body(seed=SEED_R2), seed=SEED_A
    )
    assert one.status_code == 201 and two.status_code == 201
    assert one.json()["id"] != two.json()["id"]

    # org-2 revoking the same key bytes is an independent record.
    _post_rotation(
        client, _rotation_body(actor_id="org-2", seed=SEED_R1),
        actor="org-2", seed=SEED_B,
    )
    three = _post_revocation(
        client, _revocation_body(actor_id="org-2", seed=SEED_R1),
        actor="org-2", seed=SEED_B,
    )
    assert three.status_code == 201
    assert three.json()["id"] not in {one.json()["id"], two.json()["id"]}


def test_concurrent_identical_revocations_yield_one_record_and_audit(
    tmp_db_url, file_app, file_client
):
    _world(file_client)
    _rotate(file_client, SEED_R1)

    from provenance import service
    from provenance.schemas import AuthenticationKeyRevocationCreate

    factory = file_app.state.session_factory
    payload = AuthenticationKeyRevocationCreate(
        actor_id="org-1", public_key=_key_b64(SEED_R1), reason="compromised"
    )
    results: list[tuple[str, bool]] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        session = factory()
        try:
            barrier.wait()
            revocation, created = (
                service.create_authentication_key_revocation(
                    session, payload, "org-1"
                )
            )
            results.append((revocation.id, created))
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
            select(AuthenticationKeyRevocation)
        ).scalars().all()
        events = audit_session.execute(
            select(AuditEvent).where(
                AuditEvent.event_type == EVENT_AUTHENTICATION_KEY_REVOKED
            )
        ).scalars().all()
        assert [r.id for r in rows] == [results[0][0]]
        assert [e.resource_id for e in events] == [results[0][0]]
    finally:
        audit_session.close()


# --- Validation failures: 422 validation_error, no state written ----------------


def test_revocation_body_validation_failures_are_422_and_write_nothing(
    client, db_session
):
    _world(client)
    events_before = len(_audit_events(db_session))
    valid_key = _key_b64(SEED_R1)

    def post_obj(obj):
        return _post_revocation(client, obj)

    blank_actor = post_obj(
        {"actor_id": "  ", "public_key": valid_key, "reason": "policy"}
    )
    missing_actor = post_obj({"public_key": valid_key, "reason": "policy"})
    missing_key = post_obj({"actor_id": "org-1", "reason": "policy"})
    missing_reason = post_obj({"actor_id": "org-1", "public_key": valid_key})
    extra = post_obj(
        {"actor_id": "org-1", "public_key": valid_key,
         "reason": "policy", "nope": True}
    )
    bad_reason = post_obj(
        {"actor_id": "org-1", "public_key": valid_key, "reason": "lost"}
    )
    not_b64 = post_obj(
        {"actor_id": "org-1", "public_key": "@@@", "reason": "policy"}
    )
    short = post_obj(
        {"actor_id": "org-1",
         "public_key": base64.b64encode(b"k" * 31).decode(),
         "reason": "policy"}
    )
    long = post_obj(
        {"actor_id": "org-1",
         "public_key": base64.b64encode(b"k" * 33).decode(),
         "reason": "policy"}
    )
    non_string = post_obj(
        {"actor_id": "org-1", "public_key": 1234, "reason": "policy"}
    )
    for resp in (blank_actor, missing_actor, missing_key, missing_reason,
                 extra, bad_reason, not_b64, short, long, non_string):
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "validation_error"

    # Malformed JSON is rejected at the boundary even with a signature over
    # those exact bytes.
    malformed = _post_revocation(client, None, raw=b"{not valid json")
    assert malformed.status_code == 422

    assert _revocations(db_session) == []
    assert len(_audit_events(db_session)) == events_before


def test_revocation_for_unknown_subject_is_422_and_writes_nothing(
    client, db_session
):
    _world(client)
    events_before = len(_audit_events(db_session))
    resp = _post_revocation(client, _revocation_body(actor_id="ghost"))
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "unknown_actor"
    assert _revocations(db_session) == []
    assert len(_audit_events(db_session)) == events_before


def test_revocation_caller_must_match_the_body_subject(client, db_session):
    _world(client)
    _rotate(client, SEED_R1)
    events_before = len(_audit_events(db_session))
    # org-1 authenticates but the body names org-2.
    resp = _post_revocation(client, _revocation_body(actor_id="org-2"))
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "actor_mismatch"
    assert _revocations(db_session) == []
    assert len(_audit_events(db_session)) == events_before


def test_revocation_of_an_unregistered_key_is_422_and_writes_nothing(
    client, db_session
):
    _world(client)
    events_before = len(_audit_events(db_session))
    # SEED_R1 is neither attested nor rotated for org-1.
    resp = _post_revocation(client, _revocation_body(seed=SEED_R1))
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "key_not_registered"
    assert _revocations(db_session) == []
    assert len(_audit_events(db_session)) == events_before


def test_key_registered_only_for_another_subject_is_not_a_target(
    client, db_session
):
    _world(client)
    # The key bytes are registered for org-2 but never for org-1.
    _post_rotation(
        client, _rotation_body(actor_id="org-2", seed=SEED_R1),
        actor="org-2", seed=SEED_B,
    )
    resp = _post_revocation(client, _revocation_body(seed=SEED_R1))
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "key_not_registered"
    assert _revocations(db_session) == []


def test_retired_rotation_key_remains_a_registered_target(client):
    _world(client)
    rotation = _rotate(client, SEED_R1)
    assert _post_retire(client, rotation["id"]).status_code == 200
    resp = _post_revocation(client, _revocation_body(seed=SEED_R1))
    assert resp.status_code == 201, resp.text


# --- Credential failures on the protected revocation route ----------------------


def test_revocation_without_credentials_is_422(client, db_session):
    _world(client)
    events_before = len(_audit_events(db_session))
    body = json.dumps(_revocation_body()).encode()
    resp = client.post(
        REVOCATIONS_PATH, content=body,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "missing_credentials"
    assert len(_audit_events(db_session)) == events_before


def test_revocation_wrong_signature_is_422(client):
    _world(client)
    _rotate(client, SEED_R1)
    # SEED_B is not a current key of org-1.
    resp = _post_revocation(client, _revocation_body(), seed=SEED_B)
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )


def test_revocation_rejects_stale_and_malformed_timestamps(client):
    _world(client)
    _rotate(client, SEED_R1)
    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    for raw, reason in (
        (stale, "timestamp_out_of_window"),
        ("2026-09-20T12:00:00", "invalid_timestamp"),
        ("not-a-timestamp", "invalid_timestamp"),
    ):
        resp = _post_revocation(client, _revocation_body(), timestamp=raw)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["details"]["reason"] == reason, raw


def test_revocation_signature_binds_to_method_path_and_body(client):
    _world(client)
    _rotate(client, SEED_R1)
    assert _post_revocation(
        client, _revocation_body(), signed_method="GET"
    ).status_code == 422
    assert _post_revocation(
        client, _revocation_body(), signed_path=ROTATIONS_PATH
    ).status_code == 422
    assert _post_revocation(
        client, _revocation_body(), signed_body=b'{"actor_id":"x"}'
    ).status_code == 422


# --- By-id retrieval -------------------------------------------------------------


def test_get_revocation_returns_the_record(client):
    _world(client)
    _rotate(client, SEED_R1)
    created = _post_revocation(client, _revocation_body()).json()
    resp = _get_revocation(client, created["id"])
    assert resp.status_code == 200, resp.text
    assert resp.json() == created


def test_get_unknown_revocation_is_404_not_found(client, db_session):
    _world(client)
    events_before = len(_audit_events(db_session))
    resp = _get_revocation(client, "akv_" + "0" * 64)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == (
        "authentication_key_revocation_not_found"
    )
    # The read is strictly read-only.
    assert len(_audit_events(db_session)) == events_before


def test_get_revocation_never_exposes_sensitive_material(client):
    _world(client)
    _rotate(client, SEED_R1)
    created = _post_revocation(client, _revocation_body()).json()
    body = _get_revocation(client, created["id"]).json()
    assert set(body) == {"id", "actor_id", "public_key", "reason", "revoked_at"}
    assert body["public_key"] == _key_b64(SEED_R1)


# --- Persistence across restarts ---------------------------------------------------


def test_revocation_survives_restart_and_stays_effective(
    tmp_db_url, file_client
):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    att1, _ = _world(file_client)
    _rotate(file_client, SEED_R1)
    created = _post_revocation(file_client, _revocation_body()).json()

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        # The record and its effect are durable: the key stays dead (the
        # protected read renders the unauthenticated signature as the
        # opaque 404) and the by-id read returns the original record.
        dead = _protected_get(client, att1["id"], actor="org-1", seed=SEED_R1)
        assert dead.status_code == 404
        alive = _protected_get(client, att1["id"], actor="org-1", seed=SEED_A)
        assert alive.status_code == 200
        fetched = _get_revocation(client, created["id"])
        assert fetched.status_code == 200
        assert fetched.json() == created


def test_legacy_database_gains_the_revocation_table(tmp_path):
    # A database migrated from before version 3 keeps every existing row
    # and gains exactly the new empty table.
    import sqlite3 as _sqlite3

    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    db_path = tmp_path / "legacy.db"
    url = f"sqlite:///{db_path.as_posix()}"
    with TestClient(create_app(Settings(database_url=url))) as client:
        _world(client)
        _rotate(client, SEED_R1)

    con = _sqlite3.connect(db_path)
    try:
        con.execute("DROP TABLE authentication_key_revocations")
        con.execute("DELETE FROM schema_migrations WHERE version = 3")
        con.commit()
    finally:
        con.close()

    with TestClient(create_app(Settings(database_url=url))) as client:
        # Startup migrates the simulated pre-v3 database to version 3; the
        # pre-existing rotation is untouched and revocations work.
        resp = _post_revocation(client, _revocation_body())
        assert resp.status_code == 201, resp.text
    con = _sqlite3.connect(db_path)
    try:
        tables = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "authentication_key_revocations" in tables
        assert [
            r[0]
            for r in con.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            )
        ] == [1, 2, 3]
    finally:
        con.close()
