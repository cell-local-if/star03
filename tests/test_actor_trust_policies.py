"""Tests for subject-level trust-policy registration.

Covers ``POST /v1/trust-policies``:

* the shared ``X-PA``/``X-PT``/``X-PS`` protected-request contract (a
  subject bootstraps with an existing non-revoked attestation key);
* first registration (201) with the exact public fields, the stable
  ``atp_`` id, ``enabled: true``, a UTC ``created_at``, and a single
  ``actor_trust_policy.created`` audit row committed in one transaction;
* 200 idempotency for a retried submission with the same subject and
  threshold (same id, no second row or audit event), independent policies
  for distinct subjects, and the 409 ``actor_trust_policy_conflict`` on a
  different threshold for an already-registered subject (no write);
* the authorization boundary: missing/unverifiable credentials are the
  opaque 404 ``not_found``, malformed timestamp/signature encodings are
  422, and a valid signature that names another subject in the body is the
  opaque 404 (never a 422);
* body validation: missing/blank/extra fields, bad JSON, non-integer and
  out-of-range thresholds are all 422 and write nothing;
* persistence across a restart and the absence of any update/delete path.

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures).
"""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from datetime import datetime, timezone

from sqlalchemy import func, select

from provenance.access_signing import access_message_bytes
from provenance.app import create_app
from provenance.config import Settings
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


# --- World setup ------------------------------------------------------------


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


# --- Signed-request helper --------------------------------------------------


def _signed_headers(method, path, body, *, actor, seed, timestamp=None,
                    signed_timestamp=None, signed_method=None,
                    signed_path=None, signed_body=None):
    ts = timestamp or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
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


def _policy_body(actor_id="org-1", threshold=1):
    return {"actor_id": actor_id, "threshold": threshold}


def _post_policy(client, body_obj=None, *, actor="org-1", seed=SEED_A,
                 raw=None, headers_extra=None, **sign_kwargs):
    body = raw if raw is not None else json.dumps(
        body_obj if body_obj is not None else _policy_body(actor)
    ).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", POLICIES_PATH, body, actor=actor, seed=seed,
                          **sign_kwargs),
        **(headers_extra or {}),
    }
    return client.post(POLICIES_PATH, content=body, headers=headers)


# --- First registration -----------------------------------------------------


def test_first_registration_returns_201_with_exact_public_fields(client):
    _world(client)
    resp = _post_policy(client, _policy_body(threshold=2))
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == {"id", "actor_id", "threshold", "enabled", "created_at"}
    assert body["id"].startswith("atp_")
    assert len(body["id"]) == len("atp_") + 64
    assert body["actor_id"] == "org-1"
    assert body["threshold"] == 2
    assert body["enabled"] is True
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0


def test_policy_id_is_the_stable_pair_derived_identifier(client):
    _world(client)
    body = _post_policy(client, _policy_body(threshold=3)).json()
    assert body["id"] == actor_trust_policy_id("org-1", 3)


def test_policy_and_audit_commit_in_one_transaction(client, db_session):
    _world(client)
    created = _post_policy(client, _policy_body(threshold=1)).json()

    rows = db_session.execute(select(ActorTrustPolicy)).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert (row.id, row.actor_id, row.threshold, row.enabled) == (
        created["id"], "org-1", 1, True
    )
    assert row.created_at.utcoffset().total_seconds() == 0
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ACTOR_TRUST_POLICY_CREATED
        )
    ).scalars().all()
    assert [e.resource_id for e in events] == [created["id"]]
    assert events[0].created_at.utcoffset().total_seconds() == 0


# --- Idempotency and conflict -----------------------------------------------


def test_retry_same_subject_and_threshold_returns_200_and_writes_nothing(
    client, db_session
):
    _world(client)
    first = _post_policy(client, _policy_body(threshold=2))
    assert first.status_code == 201

    audit_before = db_session.scalar(select(func.count()).select_from(AuditEvent))
    second = _post_policy(client, _policy_body(threshold=2))
    assert second.status_code == 200, second.text
    assert second.json() == first.json()

    assert db_session.scalar(select(func.count()).select_from(ActorTrustPolicy)) == 1
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audit_before
    )


def test_distinct_subjects_register_independent_policies(client):
    _world(client)
    one = _post_policy(client, _policy_body("org-1", 1), actor="org-1", seed=SEED_A)
    two = _post_policy(client, _policy_body("org-2", 2), actor="org-2", seed=SEED_B)
    assert one.status_code == 201
    assert two.status_code == 201
    assert one.json()["id"] != two.json()["id"]
    assert (one.json()["actor_id"], one.json()["threshold"]) == ("org-1", 1)
    assert (two.json()["actor_id"], two.json()["threshold"]) == ("org-2", 2)


def test_different_threshold_is_409_conflict_and_writes_nothing(client, db_session):
    _world(client)
    first = _post_policy(client, _policy_body(threshold=2))
    assert first.status_code == 201

    policies_before = db_session.scalar(
        select(func.count()).select_from(ActorTrustPolicy)
    )
    audit_before = db_session.scalar(select(func.count()).select_from(AuditEvent))

    conflict = _post_policy(client, _policy_body(threshold=3))
    assert conflict.status_code == 409, conflict.text
    error = conflict.json()["error"]
    assert error["code"] == "actor_trust_policy_conflict"
    assert error["details"]["actor_id"] == "org-1"

    assert (
        db_session.scalar(select(func.count()).select_from(ActorTrustPolicy))
        == policies_before
    )
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audit_before
    )
    # The original policy is unchanged.
    assert (
        db_session.execute(select(ActorTrustPolicy)).scalar_one().threshold == 2
    )


def test_threshold_boundaries_1_and_100_are_accepted(client):
    _world(client)
    low = _post_policy(client, _policy_body("org-1", 1))
    assert low.status_code == 201
    # org-2 is still free at the upper boundary.
    high = _post_policy(client, _policy_body("org-2", 100), actor="org-2",
                        seed=SEED_B)
    assert high.status_code == 201
    assert high.json()["threshold"] == 100


# --- Authentication boundary ------------------------------------------------


def test_missing_credentials_are_opaque_404(client):
    _world(client)
    body = json.dumps(_policy_body()).encode("utf-8")
    # No X-PA/X-PT/X-PS headers at all.
    resp = client.post(POLICIES_PATH, content=body,
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_signature_no_current_key_verifies_is_opaque_404(client):
    _world(client)
    # org-2's signature over an org-1 body: well-formed credentials, but the
    # body names a different subject -> opaque 404 (the unauthorized path).
    other = _post_policy(client, _policy_body("org-1", 1), actor="org-2",
                         seed=SEED_B)
    assert other.status_code == 404
    assert other.json()["error"]["code"] == "not_found"


def test_valid_signature_for_other_subject_is_404_not_422_even_when_absent(
    client, db_session
):
    # org-2 authenticates validly and names a subject that does not exist at
    # all. The valid-signature path is still the opaque 404, never a 422, and
    # nothing is written.
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    _make_attestation(
        client, "org-2", SEED_B,
        _make_claim(client, "org-2", digest=DIGEST_B)["id"],
    )
    audit_before = db_session.scalar(select(func.count()).select_from(AuditEvent))
    resp = _post_policy(client, {"actor_id": "ghost", "threshold": 1},
                        actor="org-2", seed=SEED_B)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audit_before
    )


def test_unparseable_timestamp_is_422(client):
    _world(client)
    resp = _post_policy(client, timestamp="not-a-timestamp")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_timestamp_out_of_window_is_422(client):
    _world(client)
    resp = _post_policy(client, timestamp="2000-01-01T00:00:00Z")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_malformed_signature_encoding_is_422(client):
    _world(client)
    for value in ("not-base64!!!", "AAAA", base64.b64encode(b"x" * 32).decode()):
        resp = _post_policy(
            client, headers_extra={"X-PS": value}
        )
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_signature_over_wrong_body_is_opaque_404_and_writes_nothing(
    client, db_session
):
    # A structurally valid signature that commits to different bytes cannot be
    # verified by any current key; "signature content invalid" collapses into
    # the same opaque 404 as missing credentials, and nothing is written.
    _world(client)
    audit_before = db_session.scalar(select(func.count()).select_from(AuditEvent))
    body = json.dumps(_policy_body()).encode("utf-8")
    resp = _post_policy(client, raw=body, signed_body=b"different")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"
    assert db_session.scalar(
        select(func.count()).select_from(ActorTrustPolicy)
    ) == 0
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audit_before
    )


# --- Body validation --------------------------------------------------------


def test_missing_fields_are_422(client):
    _world(client)
    for body_obj in ({}, {"actor_id": "org-1"}, {"threshold": 1}):
        resp = _post_policy(client, body_obj)
        assert resp.status_code == 422, body_obj
        assert resp.json()["error"]["code"] == "validation_error"


def test_extra_fields_are_422(client):
    _world(client)
    resp = _post_policy(client, {"actor_id": "org-1", "threshold": 1, "enabled": False})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_bad_json_is_422(client):
    _world(client)
    resp = _post_policy(client, raw=b"{not json")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_threshold_must_be_an_integer_in_1_to_100(client):
    _world(client)
    for threshold in (0, -1, 101, 1.5, "1", True, None, [1]):
        resp = _post_policy(client, {"actor_id": "org-1", "threshold": threshold})
        assert resp.status_code == 422, threshold
        assert resp.json()["error"]["code"] == "validation_error"


def test_blank_or_whitespace_actor_id_is_422(client):
    _world(client)
    for value in ("", "   ", "\t"):
        resp = _post_policy(client, {"actor_id": value, "threshold": 1})
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_failed_body_validation_writes_nothing(client, db_session):
    _world(client)
    policies_before = db_session.scalar(
        select(func.count()).select_from(ActorTrustPolicy)
    )
    audit_before = db_session.scalar(select(func.count()).select_from(AuditEvent))
    for body_obj in (
        {},
        {"actor_id": "org-1", "threshold": 0},
        {"actor_id": "org-1", "threshold": 1, "extra": True},
    ):
        assert _post_policy(client, body_obj).status_code == 422
    assert (
        db_session.scalar(select(func.count()).select_from(ActorTrustPolicy))
        == policies_before
    )
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audit_before
    )


# --- Persistence and append-only surface ------------------------------------


def test_policy_persists_across_restart(tmp_db_url, file_client):
    _world(file_client)
    created = _post_policy(file_client, _policy_body(threshold=2)).json()

    # A second process over the same file sees the registered policy and a
    # retry is the idempotent 200 (the bootstrapping attestation is also
    # persisted, so the same key still authenticates).
    app_two = create_app(Settings(database_url=tmp_db_url))
    from fastapi.testclient import TestClient

    with TestClient(app_two) as client_two:
        retry = _post_policy(client_two, _policy_body(threshold=2))
        assert retry.status_code == 200
        assert retry.json() == created

        # The immutable threshold still conflicts across the restart.
        assert _post_policy(
            client_two, _policy_body(threshold=3)
        ).status_code == 409

    con = sqlite3.connect(tmp_db_url.removeprefix("sqlite:///"))
    try:
        cols = {
            row[1]
            for row in con.execute("PRAGMA table_info(actor_trust_policies)")
        }
        # Only public policy material is stored; no credential/key columns.
        assert cols == {
            "seq", "id", "actor_id", "threshold", "enabled", "created_at"
        }
    finally:
        con.close()


def test_no_update_or_delete_route_exists(client):
    _world(client)
    _post_policy(client, _policy_body(threshold=1))
    row_id = actor_trust_policy_id("org-1", 1)
    for method in ("PUT", "PATCH", "DELETE"):
        resp = client.request(method, f"{POLICIES_PATH}/{row_id}")
        assert resp.status_code in (404, 405)
