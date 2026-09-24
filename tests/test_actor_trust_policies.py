"""Tests for subject trust-policy registration.

Covers ``POST /v1/trust-policies``:

* first registration (201) with the exact public fields, the stable
  ``atp_`` id, and a single-transaction ``actor_trust_policy.created``
  audit row;
* 200 idempotency for the same subject and threshold (original record, no
  audit), and 409 ``actor_trust_policy_conflict`` for a different threshold
  with no write;
* the shared ``X-PA``/``X-PT``/``X-PS`` access authentication using the
  read-style boundary: missing/unverifiable credentials are the opaque
  404 ``not_found``, malformed timestamps/signature encoding are 422, and a
  validly authenticated caller submitting another subject gets 404 rather
  than 422;
* threshold and request-body validation (422), the absence of any update
  or delete path, persistence across a restart, and the absence of any
  stored private key or raw signature.

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures); only fixed seed-derived public keys are used.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from provenance.access_signing import access_message_bytes
from provenance.ids import actor_trust_policy_id
from provenance.models import (
    EVENT_ACTOR_TRUST_POLICY_CREATED,
    ActorTrustPolicy,
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

POLICIES_PATH = "/v1/trust-policies"


# --- World setup --------------------------------------------------------------


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
    _make_attestation(client, "org-1", SEED_A, _make_claim(client, "org-1")["id"])
    _make_attestation(
        client, "org-2", SEED_B,
        _make_claim(client, "org-2", digest=DIGEST_B)["id"],
    )


# --- Signed-request helper ----------------------------------------------------


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


def _post_policy(client, body_obj=None, *, actor="org-1", seed=SEED_A,
                 raw=None, **sign_kwargs):
    body = raw if raw is not None else json.dumps(body_obj).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", POLICIES_PATH, body,
                          actor=actor, seed=seed, **sign_kwargs),
    }
    return client.post(POLICIES_PATH, content=body, headers=headers)


def _audit_events(session):
    return session.execute(select(AuditEvent)).scalars().all()


# --- First registration -------------------------------------------------------


def test_first_policy_returns_201_with_exact_public_fields(client):
    _world(client)
    resp = _post_policy(client, {"actor_id": "org-1", "threshold": 2})
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == {"id", "actor_id", "threshold", "active", "created_at"}
    assert body["id"].startswith("atp_")
    assert len(body["id"]) == len("atp_") + 64
    assert body["actor_id"] == "org-1"
    assert body["threshold"] == 2
    assert body["active"] is True
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0


def test_policy_id_is_the_stable_subject_derived_identifier(client):
    _world(client)
    body = _post_policy(client, {"actor_id": "org-1", "threshold": 3}).json()
    assert body["id"] == actor_trust_policy_id("org-1")


def test_policy_and_created_audit_commit_in_one_transaction(client, db_session):
    _world(client)
    created = _post_policy(client, {"actor_id": "org-1", "threshold": 2}).json()

    rows = db_session.execute(select(ActorTrustPolicy)).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert (row.id, row.actor_id, row.threshold, row.active) == (
        created["id"], "org-1", 2, True
    )
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ACTOR_TRUST_POLICY_CREATED
        )
    ).scalars().all()
    assert [e.resource_id for e in events] == [created["id"]]
    assert events[0].created_at.utcoffset().total_seconds() == 0


def test_no_private_key_or_signature_material_is_persisted(tmp_db_url, file_client):
    _world(file_client)
    created = _post_policy(
        file_client, {"actor_id": "org-1", "threshold": 2}
    ).json()
    con = sqlite3.connect(tmp_db_url.removeprefix("sqlite:///"))
    try:
        cols = {
            row[1]
            for row in con.execute("PRAGMA table_info(actor_trust_policies)")
        }
        assert cols == {"seq", "id", "actor_id", "threshold", "active",
                        "created_at"}
        assert "private_key" not in cols and "signature" not in cols
        stored = con.execute(
            "SELECT actor_id, threshold, active FROM actor_trust_policies"
            " WHERE id = ?",
            (created["id"],),
        ).fetchone()
        assert stored == ("org-1", 2, 1)
    finally:
        con.close()


# --- Idempotency and conflict -------------------------------------------------


def test_same_subject_and_threshold_retry_returns_200_original_no_audit(
    client, db_session
):
    _world(client)
    first = _post_policy(client, {"actor_id": "org-1", "threshold": 2})
    assert first.status_code == 201
    events_after_first = len(_audit_events(db_session))

    for _ in range(3):
        retry = _post_policy(client, {"actor_id": "org-1", "threshold": 2})
        assert retry.status_code == 200
        assert retry.json() == first.json()

    rows = db_session.execute(select(ActorTrustPolicy)).scalars().all()
    assert [r.id for r in rows] == [first.json()["id"]]
    created_events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ACTOR_TRUST_POLICY_CREATED
        )
    ).scalars().all()
    assert len(created_events) == 1
    assert len(_audit_events(db_session)) == events_after_first


def test_different_threshold_is_409_conflict_and_writes_nothing(
    client, db_session
):
    _world(client)
    first = _post_policy(client, {"actor_id": "org-1", "threshold": 2})
    assert first.status_code == 201
    events_after_first = len(_audit_events(db_session))

    conflict = _post_policy(client, {"actor_id": "org-1", "threshold": 3})
    assert conflict.status_code == 409, conflict.text
    error = conflict.json()["error"]
    assert error["code"] == "actor_trust_policy_conflict"
    assert error["details"]["actor_id"] == "org-1"

    # The original policy is untouched and nothing was written.
    rows = db_session.execute(select(ActorTrustPolicy)).scalars().all()
    assert len(rows) == 1
    assert rows[0].threshold == 2
    assert len(_audit_events(db_session)) == events_after_first

    # The original threshold still retries as 200.
    retry = _post_policy(client, {"actor_id": "org-1", "threshold": 2})
    assert retry.status_code == 200
    assert retry.json() == first.json()


def test_distinct_subjects_register_independent_policies(client):
    _world(client)
    one = _post_policy(client, {"actor_id": "org-1", "threshold": 2})
    two = _post_policy(
        client, {"actor_id": "org-2", "threshold": 2},
        actor="org-2", seed=SEED_B,
    )
    assert one.status_code == 201
    assert two.status_code == 201
    assert one.json()["id"] != two.json()["id"]
    # Same threshold value for two subjects is not a conflict.
    assert one.json()["threshold"] == two.json()["threshold"] == 2


# --- Credential boundaries ----------------------------------------------------


def test_policy_without_credentials_is_opaque_404(client, db_session):
    _world(client)
    events_before = len(_audit_events(db_session))
    body = json.dumps({"actor_id": "org-1", "threshold": 2}).encode()
    resp = client.post(
        POLICIES_PATH, content=body,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"
    assert db_session.execute(select(ActorTrustPolicy)).scalars().all() == []
    assert len(_audit_events(db_session)) == events_before


def test_policy_unverifiable_signature_is_opaque_404(client, db_session):
    _world(client)
    events_before = len(_audit_events(db_session))
    # SEED_B is not a current key of org-1.
    resp = _post_policy(
        client, {"actor_id": "org-1", "threshold": 2}, seed=SEED_B
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"
    assert db_session.execute(select(ActorTrustPolicy)).scalars().all() == []
    assert len(_audit_events(db_session)) == events_before


def test_policy_malformed_timestamp_and_signature_are_422(client, db_session):
    _world(client)
    events_before = len(_audit_events(db_session))
    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    stale_ts = _post_policy(
        client, {"actor_id": "org-1", "threshold": 2}, timestamp=stale
    )
    assert stale_ts.status_code == 422
    assert stale_ts.json()["error"]["code"] == "validation_error"

    body = json.dumps({"actor_id": "org-1", "threshold": 2}).encode()
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", POLICIES_PATH, body,
                          actor="org-1", seed=SEED_A),
    }
    headers["X-PS"] = base64.b64encode(b"s" * 63).decode()
    bad_sig = client.post(POLICIES_PATH, content=body, headers=headers)
    assert bad_sig.status_code == 422
    assert bad_sig.json()["error"]["code"] == "validation_error"

    assert db_session.execute(select(ActorTrustPolicy)).scalars().all() == []
    assert len(_audit_events(db_session)) == events_before


def test_policy_signature_binds_to_method_path_and_body(client):
    _world(client)
    assert _post_policy(
        client, {"actor_id": "org-1", "threshold": 2}, signed_method="GET"
    ).status_code == 404
    assert _post_policy(
        client, {"actor_id": "org-1", "threshold": 2},
        signed_path="/v1/attestations",
    ).status_code == 404
    assert _post_policy(
        client, {"actor_id": "org-1", "threshold": 2},
        signed_body=b'{"actor_id":"x"}',
    ).status_code == 404


def test_authenticated_caller_submitting_another_subject_is_404_not_422(
    client, db_session
):
    _world(client)
    events_before = len(_audit_events(db_session))
    # A valid org-1 signature over a body naming org-2: unauthorized, and
    # indistinguishable from an unauthenticated request -- 404, never 422.
    resp = _post_policy(client, {"actor_id": "org-2", "threshold": 2})
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "not_found"
    assert db_session.execute(select(ActorTrustPolicy)).scalars().all() == []
    assert len(_audit_events(db_session)) == events_before


# --- Request-body validation --------------------------------------------------


def test_policy_body_validation_failures_are_422_and_write_nothing(
    client, db_session
):
    _world(client)
    events_before = len(_audit_events(db_session))

    cases = (
        {"threshold": 2},
        {"actor_id": "org-1"},
        {"actor_id": "  ", "threshold": 2},
        {"actor_id": "org-1", "threshold": 0},
        {"actor_id": "org-1", "threshold": 101},
        {"actor_id": "org-1", "threshold": 2.0},
        {"actor_id": "org-1", "threshold": "2"},
        {"actor_id": "org-1", "threshold": True},
        {"actor_id": "org-1", "threshold": -1},
        {"actor_id": "org-1", "threshold": 2, "nope": True},
    )
    for obj in cases:
        resp = _post_policy(client, obj)
        assert resp.status_code == 422, obj
        assert resp.json()["error"]["code"] == "validation_error"

    # Malformed JSON is rejected at the boundary even with a signature over
    # those exact bytes.
    raw = b"{not valid json"
    malformed = _post_policy(client, None, raw=raw)
    assert malformed.status_code == 422

    assert db_session.execute(select(ActorTrustPolicy)).scalars().all() == []
    assert len(_audit_events(db_session)) == events_before


def test_policy_threshold_boundaries_1_and_100_are_accepted(client):
    _world(client)
    # org-1 takes threshold 1; org-2 independently takes threshold 100.
    low = _post_policy(client, {"actor_id": "org-1", "threshold": 1})
    assert low.status_code == 201, low.text
    high = _post_policy(
        client, {"actor_id": "org-2", "threshold": 100},
        actor="org-2", seed=SEED_B,
    )
    assert high.status_code == 201, high.text


# --- Append-only surface ------------------------------------------------------


def test_policy_has_no_update_delete_or_deactivate_path(client):
    _world(client)
    created = _post_policy(client, {"actor_id": "org-1", "threshold": 2}).json()
    body = json.dumps({"threshold": 5}).encode()
    for method in ("PUT", "PATCH", "DELETE"):
        resp = client.request(method, POLICIES_PATH, content=body)
        assert resp.status_code == 405, method
    # No deactivate sub-resource exists either.
    retire = client.post(f"{POLICIES_PATH}/{created['id']}/retire", content=b"")
    assert retire.status_code in (404, 405)


# --- Persistence --------------------------------------------------------------


def test_policies_survive_restart(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    _world(file_client)
    created = _post_policy(
        file_client, {"actor_id": "org-1", "threshold": 2}
    ).json()

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        retry = _post_policy(client, {"actor_id": "org-1", "threshold": 2})
        assert retry.status_code == 200
        assert retry.json() == created
        conflict = _post_policy(client, {"actor_id": "org-1", "threshold": 3})
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == (
            "actor_trust_policy_conflict"
        )
