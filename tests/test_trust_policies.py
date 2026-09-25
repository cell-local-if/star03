"""Tests for subject trust-policy registration.

Covers ``POST /v1/trust-policies``: the protected signed-header contract,
first-creation 201 with the exact public view (compact UTF-8 JSON terminated
by exactly one newline), the stable ``atp_`` identifier, single-transaction
audit commit, same-threshold retry idempotency (200, no new row/audit/id),
the different-threshold 409 ``actor_trust_policy_conflict``, the
unknown-subject and caller/subject-mismatch 422s, strict pure-decimal-integer
threshold validation, immutability (no update/delete path), and the
no-write guarantee on every failure.

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures).
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
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
    """Create two actors, each holding a current authentication key."""
    create_actor(client)  # org-1
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    _make_attestation(client, "org-1", SEED_A, _make_claim(client, "org-1")["id"])
    _make_attestation(
        client, "org-2", SEED_B, _make_claim(client, "org-2", digest=DIGEST_B)["id"]
    )


# --- Signed-header helper -----------------------------------------------------


def _signed_headers(
    method,
    path,
    body,
    *,
    actor="org-1",
    seed=SEED_A,
    timestamp=None,
    signed_method=None,
    signed_path=None,
    signed_body=None,
):
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    body_digest = hashlib.sha256(
        signed_body if signed_body is not None else body
    ).hexdigest()
    message = access_message_bytes(
        signed_method or method,
        signed_path if signed_path is not None else path,
        ts,
        body_digest,
    )
    signature = base64.b64encode(ed25519_sign(seed, message)).decode("ascii")
    return {"X-PA": actor, "X-PT": ts, "X-PS": signature}


def _post_policy(client, body_obj, *, actor="org-1", seed=SEED_A, **sign_kwargs):
    body = json.dumps(body_obj).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", POLICIES_PATH, body, actor=actor, seed=seed,
                          **sign_kwargs),
    }
    return client.post(POLICIES_PATH, content=body, headers=headers)


def _policy_body(actor_id="org-1", threshold=2):
    return {"actor_id": actor_id, "threshold": threshold}


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


def _policy_rows(session):
    return session.execute(select(ActorTrustPolicy)).scalars().all()


# --- First creation -----------------------------------------------------------


def test_first_policy_returns_201_with_exact_public_fields(client):
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
    assert body["created_at"].endswith(("Z", "+00:00"))


def test_policy_response_is_compact_utf8_json_with_one_trailing_newline(client):
    _world(client)
    resp = _post_policy(client, _policy_body(threshold=2))
    assert resp.status_code == 201
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    # Compact separators: no insignificant whitespace anywhere.
    assert b", " not in raw
    assert b": " not in raw
    # The threshold renders as a plain integer, never a float or string.
    assert b'"threshold":2,' in raw
    assert b'"enabled":true,' in raw
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    assert raw == expected


def test_policy_id_is_the_stable_subject_derived_identifier(client):
    _world(client)
    body = _post_policy(client, _policy_body()).json()
    assert body["id"] == actor_trust_policy_id("org-1")


def test_threshold_boundaries_one_and_one_hundred_are_accepted(client):
    _world(client)
    assert _post_policy(client, _policy_body(threshold=1)).status_code == 201
    assert (
        _post_policy(client, _policy_body("org-2", 100), actor="org-2", seed=SEED_B)
        .status_code
        == 201
    )


def test_policy_and_audit_event_commit_in_one_transaction(client, db_session):
    _world(client)
    created = _post_policy(client, _policy_body(threshold=3)).json()

    rows = _policy_rows(db_session)
    assert [(r.id, r.actor_id, r.threshold, r.enabled) for r in rows] == [
        (created["id"], "org-1", 3, True)
    ]
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ACTOR_TRUST_POLICY_CREATED
        )
    ).scalars().all()
    assert [e.resource_id for e in events] == [created["id"]]
    assert events[0].created_at.tzinfo.utcoffset(
        events[0].created_at
    ).total_seconds() == 0


# --- Idempotency, conflict, and immutability -----------------------------------


def test_same_threshold_retry_returns_200_original_and_no_new_writes(
    client, db_session
):
    _world(client)
    first = _post_policy(client, _policy_body(threshold=2))
    assert first.status_code == 201
    events_after_first = _audit_count(db_session)

    for _ in range(3):
        retry = _post_policy(client, _policy_body(threshold=2))
        assert retry.status_code == 200
        assert retry.content == first.content

    rows = _policy_rows(db_session)
    assert [r.id for r in rows] == [first.json()["id"]]
    assert _audit_count(db_session) == events_after_first
    policy_events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ACTOR_TRUST_POLICY_CREATED
        )
    ).scalars().all()
    assert len(policy_events) == 1


def test_different_threshold_for_same_subject_is_409_and_writes_nothing(
    client, db_session
):
    _world(client)
    first = _post_policy(client, _policy_body(threshold=2))
    assert first.status_code == 201
    events_before = _audit_count(db_session)

    conflict = _post_policy(client, _policy_body(threshold=5))
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "actor_trust_policy_conflict"
    assert conflict.json()["error"]["details"]["actor_id"] == "org-1"

    rows = _policy_rows(db_session)
    assert [(r.id, r.threshold) for r in rows] == [(first.json()["id"], 2)]
    assert _audit_count(db_session) == events_before


def test_subjects_have_independent_policies(client):
    _world(client)
    one = _post_policy(client, _policy_body("org-1", 2))
    two = _post_policy(
        client, _policy_body("org-2", 2), actor="org-2", seed=SEED_B
    )
    assert one.status_code == 201
    assert two.status_code == 201
    assert one.json()["id"] != two.json()["id"]


def test_policy_has_no_update_delete_or_deactivate_path(client):
    _world(client)
    assert _post_policy(client, _policy_body()).status_code == 201
    for method in (client.put, client.patch, client.delete):
        resp = method(POLICIES_PATH)
        assert resp.status_code == 405
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Subject validation ---------------------------------------------------------


def test_unknown_subject_is_422_and_writes_nothing(client, db_session):
    _world(client)
    events_before = _audit_count(db_session)
    resp = _post_policy(client, _policy_body("ghost", 2))
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert resp.json()["error"]["details"]["reason"] == "unknown_actor"
    assert _policy_rows(db_session) == []
    assert _audit_count(db_session) == events_before


def test_subject_mismatch_is_422_and_writes_nothing(client, db_session):
    _world(client)
    events_before = _audit_count(db_session)
    # org-1 authenticates but tries to register org-2's policy.
    resp = _post_policy(client, _policy_body("org-2", 2))
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "actor_mismatch"
    assert _policy_rows(db_session) == []
    assert _audit_count(db_session) == events_before


# --- Field and threshold validation ---------------------------------------------


def test_threshold_must_be_a_pure_decimal_integer(client, db_session):
    _world(client)
    events_before = _audit_count(db_session)
    for bad in (0, 101, -1, 5.0, 5.5, 1e2, "5", "05", " 5", True, None):
        resp = _post_policy(client, _policy_body(threshold=bad))
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)
    assert _policy_rows(db_session) == []
    assert _audit_count(db_session) == events_before


def test_policy_body_validation_failures_are_422_and_write_nothing(
    client, db_session
):
    _world(client)
    events_before = _audit_count(db_session)

    missing_threshold = _post_policy(client, {"actor_id": "org-1"})
    missing_actor = _post_policy(client, {"threshold": 2})
    blank_actor = _post_policy(client, {"actor_id": "  ", "threshold": 2})
    extra = _post_policy(
        client, {"actor_id": "org-1", "threshold": 2, "enabled": False}
    )
    for resp in (missing_threshold, missing_actor, blank_actor, extra):
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "validation_error"

    # Malformed JSON is rejected at the boundary regardless of a signature
    # that commits to those exact (malformed) bytes.
    raw_body = b"{not valid json"
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", POLICIES_PATH, raw_body),
    }
    malformed = client.post(POLICIES_PATH, content=raw_body, headers=headers)
    assert malformed.status_code == 422

    assert _policy_rows(db_session) == []
    assert _audit_count(db_session) == events_before


# --- Signed-header contract -----------------------------------------------------


def test_policy_without_headers_is_422(client, db_session):
    _world(client)
    events_before = _audit_count(db_session)
    body = json.dumps(_policy_body()).encode()
    resp = client.post(
        POLICIES_PATH, content=body, headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "missing_credentials"
    assert _audit_count(db_session) == events_before


def test_policy_with_wrong_signature_is_422(client):
    _world(client)
    # SEED_B is not a current key of org-1, so nothing verifies.
    resp = _post_policy(client, _policy_body(), actor="org-1", seed=SEED_B)
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )


def test_policy_rejects_stale_and_malformed_timestamps(client):
    _world(client)
    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    cases = {
        stale: "timestamp_out_of_window",
        "2026-09-20T12:00:00": "invalid_timestamp",
        "2026-09-20T12:00:00+01:00": "invalid_timestamp",
        "not-a-timestamp": "invalid_timestamp",
    }
    for raw, reason in cases.items():
        resp = _post_policy(client, _policy_body(), timestamp=raw)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["details"]["reason"] == reason, raw


def test_policy_rejects_malformed_signature_encoding(client):
    _world(client)
    for raw in ("@@@", "aGVsbG8", base64.b64encode(b"s" * 63).decode()):
        body = json.dumps(_policy_body()).encode()
        headers = {
            "Content-Type": "application/json",
            **_signed_headers("POST", POLICIES_PATH, body),
        }
        headers["X-PS"] = raw
        resp = client.post(POLICIES_PATH, content=body, headers=headers)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["details"]["reason"] == "invalid_signature"


def test_policy_signature_binds_to_method_path_and_body(client):
    _world(client)
    wrong_method = _post_policy(client, _policy_body(), signed_method="GET")
    assert wrong_method.status_code == 422
    wrong_path = _post_policy(
        client, _policy_body(), signed_path="/v1/trust-decisions"
    )
    assert wrong_path.status_code == 422
    wrong_body = _post_policy(
        client, _policy_body(), signed_body=b'{"actor_id":"org-1","threshold":9}'
    )
    assert wrong_body.status_code == 422


# --- Concurrent race and persistence ---------------------------------------------


def test_concurrent_identical_policies_yield_one_record_and_audit(
    tmp_db_url, file_app, file_client
):
    create_actor(file_client)
    content = file_client.post("/v1/contents", json=content_payload()).json()
    claim = file_client.post(
        "/v1/claims",
        json={
            "content_id": content["id"],
            "actor_id": "org-1",
            "claim_type": "authorship",
            "payload": {"s": 1},
        },
    ).json()
    _make_attestation(file_client, "org-1", SEED_A, claim["id"])

    from provenance import service
    from provenance.schemas import ActorTrustPolicyCreate

    factory = file_app.state.session_factory
    payload = ActorTrustPolicyCreate(actor_id="org-1", threshold=2)
    results: list[tuple[str, bool]] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        session = factory()
        try:
            barrier.wait()
            policy, created = service.create_actor_trust_policy(
                session, payload, "org-1"
            )
            results.append((policy.id, created))
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
    assert {pid for pid, _ in results} == {results[0][0]}
    assert sum(1 for _, created in results if created) == 1

    audit_session = factory()
    try:
        rows = audit_session.execute(select(ActorTrustPolicy)).scalars().all()
        events = audit_session.execute(
            select(AuditEvent).where(
                AuditEvent.event_type == EVENT_ACTOR_TRUST_POLICY_CREATED
            )
        ).scalars().all()
        assert [r.id for r in rows] == [results[0][0]]
        assert [e.resource_id for e in events] == [results[0][0]]
    finally:
        audit_session.close()


def test_policies_persist_across_app_restarts(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    _world(file_client)
    created = _post_policy(file_client, _policy_body(threshold=4)).json()

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted):
        import sqlite3

        con = sqlite3.connect(tmp_db_url.removeprefix("sqlite:///"))
        rows = con.execute(
            "SELECT id, actor_id, threshold, enabled FROM actor_trust_policies"
        ).fetchall()
        audit = con.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = ?",
            (EVENT_ACTOR_TRUST_POLICY_CREATED,),
        ).fetchone()[0]
        con.close()
        assert rows == [(created["id"], "org-1", 4, 1)]
        assert audit == 1
