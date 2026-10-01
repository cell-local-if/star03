"""Tests for emergency authentication public-key revocation.

Covers ``POST /v1/authentication-key-revocations`` and
``GET /v1/authentication-key-revocations/{revocation_id}``, including:

* the shared ``X-PA``/``X-PT``/``X-PS`` access authentication (the key
  being revoked may itself sign the request);
* first creation (201) with the exact public fields, the stable ``akv_``
  id, and a single-transaction ``authentication_key.revoked`` audit row;
* 200 idempotency for the same subject/key/reason triple, a 409
  ``authentication_key_revocation_conflict`` for a different reason, and
  independent records for distinct subjects over the same key bytes;
* the revoked key bytes immediately leaving the subject's authentication
  set -- whether registered by an attestation or a rotation -- while
  unrevoked keys, attestations, trust, and authorization are unchanged,
  and a rotation idempotent retry cannot restore eligibility;
* every input, identity, authentication, or unknown-target failure
  rendering as a 422 ``validation_error`` distinguished only by
  ``details.reason`` and writing nothing;
* the single-record read, the ``authentication_key_revocation_not_found``
  404, persistence across restarts, a legacy-database migration, and
  concurrent-create races.

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
    EVENT_AUTHENTICATION_KEY_REVOKED,
    AuthenticationKeyRevocation,
    AuditEvent,
)
from tests.helpers import (
    ed25519_public_key,
    SEED_A,
    SEED_B,
)
from tests.test_authentication_key_rotations import (
    ROTATIONS_PATH,
    SEED_R1,
    SEED_R2,
    _key_b64,
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


def _revocation_rows(session):
    return session.execute(select(AuthenticationKeyRevocation)).scalars().all()


def _world_with_rotation(client):
    """The two-actor world plus an active SEED_R1 rotation for org-1."""
    att1, att2 = _world(client)
    rotation = _post_rotation(client, _rotation_body(seed=SEED_R1)).json()
    return att1, att2, rotation


# --- First creation ------------------------------------------------------------


def test_first_revocation_returns_201_with_exact_public_fields(client):
    _world_with_rotation(client)
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


def test_every_reason_literal_is_accepted(client):
    _world(client)
    seeds = [SEED_R1, SEED_R2]
    for seed, reason in zip(
        seeds, ("compromised", "superseded")
    ):
        assert _post_rotation(client, _rotation_body(seed=seed)).status_code == 201
        resp = _post_revocation(client, _revocation_body(seed=seed, reason=reason))
        assert resp.status_code == 201, resp.text
        assert resp.json()["reason"] == reason
    # A third key for the remaining literal.
    seed_p = b"test-ed25519-revoke-policy-00000000"[:32]
    assert _post_rotation(client, _rotation_body(seed=seed_p)).status_code == 201
    resp = _post_revocation(client, _revocation_body(seed=seed_p, reason="policy"))
    assert resp.status_code == 201, resp.text
    assert resp.json()["reason"] == "policy"


def test_revocation_id_is_the_stable_pair_derived_identifier(client):
    _world_with_rotation(client)
    body = _post_revocation(client, _revocation_body()).json()
    assert body["id"] == authentication_key_revocation_id(
        "org-1", ed25519_public_key(SEED_R1).hex()
    )


def test_revocation_and_revoked_audit_commit_in_one_transaction(
    client, db_session
):
    _world_with_rotation(client)
    created = _post_revocation(client, _revocation_body()).json()

    rows = _revocation_rows(db_session)
    assert len(rows) == 1
    row = rows[0]
    assert (row.id, row.actor_id, row.reason) == (
        created["id"], "org-1", "compromised"
    )
    assert row.public_key == ed25519_public_key(SEED_R1)
    assert len(row.public_key) == 32
    assert row.created_at.utcoffset().total_seconds() == 0
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
    _world_with_rotation(file_client)
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
            "seq", "id", "actor_id", "public_key", "reason", "created_at",
        }
        assert "private_key" not in cols and "signature" not in cols
        stored = con.execute(
            "SELECT public_key FROM authentication_key_revocations WHERE id = ?",
            (created["id"],),
        ).fetchone()[0]
        assert stored == ed25519_public_key(SEED_R1)
    finally:
        con.close()


# --- The revoked key immediately leaves the authentication set -----------------


def test_revoked_rotation_key_fails_immediately_but_other_keys_remain(client):
    att1, _, _ = _world_with_rotation(client)
    # SEED_R1 authenticates before the revocation.
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


def test_key_being_revoked_may_sign_the_request(client):
    att1, _, _ = _world_with_rotation(client)
    # The compromised key signs its own emergency revocation.
    resp = _post_revocation(client, _revocation_body(), seed=SEED_R1)
    assert resp.status_code == 201, resp.text
    # From the commit on, the key no longer authenticates anything.
    assert _protected_get(client, att1["id"], actor="org-1",
                          seed=SEED_R1).status_code == 404


def test_revoked_attestation_key_fails_immediately_but_attestation_survives(
    client,
):
    att1, _, rotation = _world_with_rotation(client)
    # Revoke the SEED_A attestation key, signed by the rotation key.
    resp = _post_revocation(
        client, _revocation_body(seed=SEED_A), seed=SEED_R1
    )
    assert resp.status_code == 201, resp.text

    # SEED_A no longer authenticates; the rotation key still does.
    assert _protected_get(client, att1["id"], actor="org-1",
                          seed=SEED_A).status_code == 404
    assert _protected_get(client, att1["id"], actor="org-1",
                          seed=SEED_R1).status_code == 200

    # The attestation itself is untouched: no attestation revocation was
    # created and the proof's public view is unchanged.
    assert client.get(f"/v1/attestations/{att1['id']}").json() == att1
    assert client.get(
        f"/v1/attestations/{att1['id']}/revocations"
    ).json() == {"items": [], "count": 0}
    # The rotation record is likewise untouched.
    assert client.get(
        "/v1/actors/org-1/authentication-key-rotations"
    ).json()["items"] == [rotation]


def test_rotation_idempotent_retry_cannot_restore_a_revoked_key(client):
    att1, _, rotation = _world_with_rotation(client)
    assert _post_revocation(client, _revocation_body()).status_code == 201

    # The rotation retry is idempotent: the original active record returns,
    # but the key stays revoked.
    retry = _post_rotation(client, _rotation_body(seed=SEED_R1))
    assert retry.status_code == 200
    assert retry.json() == rotation
    assert retry.json()["active"] is True
    assert _protected_get(client, att1["id"], actor="org-1",
                          seed=SEED_R1).status_code == 404


def test_retired_rotation_key_is_still_a_registered_target(client):
    _, _, rotation = _world_with_rotation(client)
    path = f"{ROTATIONS_PATH}/{rotation['id']}/retire"
    headers = _signed_headers("POST", path, b"", actor="org-1", seed=SEED_R1)
    assert client.post(path, content=b"", headers=headers).status_code == 200
    # The key left the authentication set by retirement, yet it remains
    # registered and can still be emergency-revoked.
    resp = _post_revocation(client, _revocation_body(reason="policy"))
    assert resp.status_code == 201, resp.text
    assert resp.json()["reason"] == "policy"


def test_key_of_an_already_revoked_attestation_is_still_a_registered_target(
    client,
):
    att1, _, _ = _world_with_rotation(client)
    # Revoke the attestation itself first (an unauthenticated route).
    resp = client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": att1["id"],
            "revoker_actor_id": "org-1",
            "reason": "no longer relied upon",
        },
    )
    assert resp.status_code == 201, resp.text
    # The SEED_A key is still registered, so it remains a valid target; the
    # request is authenticated by the rotation key.
    resp = _post_revocation(
        client, _revocation_body(seed=SEED_A), seed=SEED_R1
    )
    assert resp.status_code == 201, resp.text


def test_revocation_does_not_touch_other_subjects_using_the_same_key_bytes(
    client,
):
    att1, att2 = _world(client)
    # Both subjects rotate in the identical 32 key bytes.
    assert _post_rotation(client, _rotation_body(seed=SEED_R1)).status_code == 201
    assert _post_rotation(
        client, _rotation_body(actor_id="org-2", seed=SEED_R1),
        actor="org-2", seed=SEED_B,
    ).status_code == 201

    # org-1 revokes its registration of the key.
    assert _post_revocation(client, _revocation_body()).status_code == 201
    # org-2's identical key bytes still authenticate org-2.
    assert _protected_get(client, att2["id"], actor="org-2",
                          seed=SEED_R1).status_code == 200
    # org-2's own revocation is an independent record with its own id.
    second = _post_revocation(
        client, _revocation_body(actor_id="org-2"), actor="org-2", seed=SEED_B
    )
    assert second.status_code == 201, second.text
    first = _get_revocation(
        client, authentication_key_revocation_id(
            "org-1", ed25519_public_key(SEED_R1).hex()
        )
    ).json()
    assert second.json()["id"] != first["id"]


# --- Idempotency and conflict ---------------------------------------------------


def test_same_triple_retry_returns_200_original_and_no_audit(
    client, db_session
):
    _world_with_rotation(client)
    first = _post_revocation(client, _revocation_body())
    assert first.status_code == 201
    events_after_first = len(_audit_events(db_session))

    for _ in range(3):
        retry = _post_revocation(client, _revocation_body())
        assert retry.status_code == 200
        assert retry.json() == first.json()

    assert [r.id for r in _revocation_rows(db_session)] == [first.json()["id"]]
    revoked = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_AUTHENTICATION_KEY_REVOKED
        )
    ).scalars().all()
    assert len(revoked) == 1
    assert len(_audit_events(db_session)) == events_after_first


def test_different_reason_is_409_conflict_and_writes_nothing(
    client, db_session
):
    _world_with_rotation(client)
    first = _post_revocation(client, _revocation_body(reason="compromised"))
    assert first.status_code == 201
    events_after_first = len(_audit_events(db_session))

    for reason in ("superseded", "policy"):
        conflict = _post_revocation(client, _revocation_body(reason=reason))
        assert conflict.status_code == 409
        error = conflict.json()["error"]
        assert error["code"] == "authentication_key_revocation_conflict"
        assert error["details"]["actor_id"] == "org-1"

    rows = _revocation_rows(db_session)
    assert [r.id for r in rows] == [first.json()["id"]]
    assert rows[0].reason == "compromised"
    assert len(_audit_events(db_session)) == events_after_first
    # The original record is returned unchanged by its read route.
    assert _get_revocation(client, first.json()["id"]).json() == first.json()


# --- Unknown target, identity, and credential failures --------------------------


def test_unregistered_target_key_is_422_and_writes_nothing(client, db_session):
    _world_with_rotation(client)
    events_before = len(_audit_events(db_session))
    # SEED_R2 was never registered for org-1 (no attestation, no rotation).
    resp = _post_revocation(client, _revocation_body(seed=SEED_R2))
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert resp.json()["error"]["details"]["reason"] == "unknown_public_key"
    assert _revocation_rows(db_session) == []
    assert len(_audit_events(db_session)) == events_before


def test_key_registered_only_for_another_subject_is_an_unknown_target(
    client, db_session
):
    _world(client)
    # The key is registered -- but for org-2, not for the caller org-1.
    assert _post_rotation(
        client, _rotation_body(actor_id="org-2", seed=SEED_R1),
        actor="org-2", seed=SEED_B,
    ).status_code == 201
    events_before = len(_audit_events(db_session))
    resp = _post_revocation(client, _revocation_body(seed=SEED_R1))
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "unknown_public_key"
    assert _revocation_rows(db_session) == []
    assert len(_audit_events(db_session)) == events_before


def test_revocation_for_unknown_subject_is_422_and_writes_nothing(
    client, db_session
):
    _world_with_rotation(client)
    events_before = len(_audit_events(db_session))
    resp = _post_revocation(client, _revocation_body(actor_id="ghost"))
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "unknown_actor"
    assert _revocation_rows(db_session) == []
    assert len(_audit_events(db_session)) == events_before


def test_revocation_caller_must_match_the_body_subject(client, db_session):
    _world_with_rotation(client)
    events_before = len(_audit_events(db_session))
    # org-1 authenticates but the body names org-2.
    resp = _post_revocation(client, _revocation_body(actor_id="org-2"))
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "actor_mismatch"
    assert _revocation_rows(db_session) == []
    assert len(_audit_events(db_session)) == events_before


def test_revocation_without_credentials_is_422(client, db_session):
    _world_with_rotation(client)
    events_before = len(_audit_events(db_session))
    body = json.dumps(_revocation_body()).encode()
    resp = client.post(
        REVOCATIONS_PATH, content=body,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "missing_credentials"
    assert _revocation_rows(db_session) == []
    assert len(_audit_events(db_session)) == events_before


def test_revocation_wrong_signature_is_422(client, db_session):
    _world_with_rotation(client)
    events_before = len(_audit_events(db_session))
    # SEED_B is not a current key of org-1.
    resp = _post_revocation(client, _revocation_body(), seed=SEED_B)
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )
    assert _revocation_rows(db_session) == []
    assert len(_audit_events(db_session)) == events_before


def test_revocation_rejects_stale_and_malformed_timestamps(client):
    _world_with_rotation(client)
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
    _world_with_rotation(client)
    assert _post_revocation(
        client, _revocation_body(), signed_method="GET"
    ).status_code == 422
    assert _post_revocation(
        client, _revocation_body(), signed_path=ROTATIONS_PATH
    ).status_code == 422
    assert _post_revocation(
        client, _revocation_body(), signed_body=b'{"actor_id":"x"}'
    ).status_code == 422


# --- Request-body validation -----------------------------------------------------


def test_revocation_body_validation_failures_are_422_and_write_nothing(
    client, db_session
):
    _world_with_rotation(client)
    events_before = len(_audit_events(db_session))
    valid_key = _key_b64(SEED_R1)

    def post_obj(obj):
        return _post_revocation(client, obj)

    cases = [
        {"actor_id": "  ", "public_key": valid_key, "reason": "compromised"},
        {"public_key": valid_key, "reason": "compromised"},
        {"actor_id": "org-1", "reason": "compromised"},
        {"actor_id": "org-1", "public_key": valid_key},
        {"actor_id": "org-1", "public_key": valid_key,
         "reason": "compromised", "nope": True},
        {"actor_id": "org-1", "public_key": "@@@", "reason": "compromised"},
        {"actor_id": "org-1",
         "public_key": base64.b64encode(b"k" * 31).decode(),
         "reason": "compromised"},
        {"actor_id": "org-1",
         "public_key": base64.b64encode(b"k" * 33).decode(),
         "reason": "compromised"},
        {"actor_id": "org-1", "public_key": 1234, "reason": "compromised"},
        {"actor_id": "org-1", "public_key": valid_key, "reason": "lost"},
        {"actor_id": "org-1", "public_key": valid_key, "reason": ""},
        {"actor_id": "org-1", "public_key": valid_key, "reason": "COMPROMISED"},
        {"actor_id": "org-1", "public_key": valid_key, "reason": None},
    ]
    # 32 bytes whose standard Base64 contains '+' and '/': the urlsafe
    # alphabet must be rejected rather than silently accepted.
    urlsafe_raw = bytes([0xFB, 0xFF]) * 16
    urlsafe_value = base64.urlsafe_b64encode(urlsafe_raw).decode()
    assert urlsafe_value != base64.b64encode(urlsafe_raw).decode()
    cases.append(
        {"actor_id": "org-1", "public_key": urlsafe_value,
         "reason": "compromised"}
    )

    for obj in cases:
        resp = post_obj(obj)
        assert resp.status_code == 422, obj
        assert resp.json()["error"]["code"] == "validation_error", obj

    # Malformed JSON is rejected at the boundary even with a signature over
    # those exact bytes.
    malformed = _post_revocation(client, None, raw=b"{not valid json")
    assert malformed.status_code == 422

    assert _revocation_rows(db_session) == []
    assert len(_audit_events(db_session)) == events_before


# --- Single-record read ----------------------------------------------------------


def test_get_revocation_returns_the_record(client, db_session):
    _world_with_rotation(client)
    created = _post_revocation(client, _revocation_body()).json()
    events_before = len(_audit_events(db_session))

    resp = _get_revocation(client, created["id"])
    assert resp.status_code == 200
    assert resp.json() == created
    assert set(resp.json()) == {
        "id", "actor_id", "public_key", "reason", "revoked_at"
    }
    # The read is strictly read-only.
    assert len(_audit_events(db_session)) == events_before


def test_get_unknown_revocation_is_404_not_found(client, db_session):
    _world_with_rotation(client)
    events_before = len(_audit_events(db_session))
    resp = _get_revocation(client, "akv_" + "0" * 64)
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "authentication_key_revocation_not_found"
    assert error["details"]["revocation_id"] == "akv_" + "0" * 64
    assert len(_audit_events(db_session)) == events_before


# --- Persistence, legacy migration, and concurrency ------------------------------


def test_revocations_survive_restart_and_stay_enforced(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    att1, _, _ = _world_with_rotation(file_client)
    created = _post_revocation(file_client, _revocation_body()).json()

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        # The record and its UTC fields survive the restart.
        assert _get_revocation(client, created["id"]).json() == created
        # The revoked key is still dead; the attestation key still works.
        assert _protected_get(client, att1["id"], actor="org-1",
                              seed=SEED_R1).status_code == 404
        assert _protected_get(client, att1["id"], actor="org-1",
                              seed=SEED_A).status_code == 200


def test_legacy_v2_database_gains_the_revocation_table(tmp_path):
    import sqlite3 as _sqlite3
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    # A minimal legacy database: build the full current schema, then remove
    # the revocation table and its v3 ledger row to simulate a pre-v3
    # database that already carries rotations and attestations.
    db_path = tmp_path / "legacy.db"
    url = f"sqlite:///{db_path.as_posix()}"
    with TestClient(create_app(Settings(database_url=url))) as client:
        att1, _, rotation = _world_with_rotation(client)
    con = _sqlite3.connect(db_path)
    try:
        con.execute("DROP TABLE authentication_key_revocations")
        con.execute("DELETE FROM schema_migrations WHERE version = 3")
        con.commit()
    finally:
        con.close()

    with TestClient(create_app(Settings(database_url=url))) as client:
        # Startup migrates the simulated pre-v3 database to version 3 while
        # the pre-existing records and the unrevoked key behavior survive.
        assert _protected_get(client, att1["id"], actor="org-1",
                              seed=SEED_R1).status_code == 200
        assert client.get(
            "/v1/actors/org-1/authentication-key-rotations"
        ).json()["items"] == [rotation]
        # The new API works against the migrated database.
        created = _post_revocation(client, _revocation_body())
        assert created.status_code == 201, created.text
        assert _protected_get(client, att1["id"], actor="org-1",
                              seed=SEED_R1).status_code == 404
    con = _sqlite3.connect(db_path)
    try:
        tables = {
            r[0]
            for r in con.execute(
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


def test_concurrent_identical_revocations_yield_one_record_and_audit(
    tmp_db_url, file_app, file_client
):
    _world_with_rotation(file_client)

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
            revocation, created = service.create_authentication_key_revocation(
                session, payload, "org-1"
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


def test_concurrent_different_reasons_one_writer_rest_conflict(
    tmp_db_url, file_app, file_client
):
    _world_with_rotation(file_client)

    from provenance import service
    from provenance.errors import AuthenticationKeyRevocationConflictError
    from provenance.schemas import AuthenticationKeyRevocationCreate

    factory = file_app.state.session_factory
    reasons = ["compromised", "superseded", "policy", "compromised"]
    outcomes: list[str] = []
    barrier = threading.Barrier(len(reasons))

    def worker(reason: str) -> None:
        session = factory()
        try:
            barrier.wait()
            payload = AuthenticationKeyRevocationCreate(
                actor_id="org-1", public_key=_key_b64(SEED_R1), reason=reason
            )
            _, created = service.create_authentication_key_revocation(
                session, payload, "org-1"
            )
            outcomes.append("created" if created else "idempotent")
        except AuthenticationKeyRevocationConflictError:
            outcomes.append("conflict")
        finally:
            session.close()

    threads = [
        threading.Thread(target=worker, args=(reason,)) for reason in reasons
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one writer commits; every different-reason submission is a
    # conflict, and any same-reason retry is an idempotent return.
    assert outcomes.count("created") == 1
    assert outcomes.count("conflict") + outcomes.count("idempotent") == 3

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
        assert len(rows) == 1
        assert [e.resource_id for e in events] == [rows[0].id]
    finally:
        audit_session.close()
