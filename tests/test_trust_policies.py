"""Tests for subject-level immutable trust policy registration.

Covers protected ``POST /v1/trust-policies``:

* first creation returns 201 with the stable ``atp_`` id, subject,
  threshold, ``enabled: true``, and a UTC ``created_at``; compact UTF-8
  JSON terminated by exactly one newline;
* the policy row and the ``actor_trust_policy.created`` audit event commit
  in a single transaction;
* a retry for the same subject and threshold returns 200 with the original
  policy (same id, threshold, and creation time) and writes no second row
  or audit event;
* the same subject with a different threshold is
  ``409 actor_trust_policy_conflict`` and writes nothing;
* independent subjects register independent policies;
* there is no update, delete, disable, or replace path (405);
* an unknown subject and a caller/body mismatch are 422
  ``validation_error``;
* the threshold must be a plain JSON decimal integer in 1..100 (strings,
  floats, booleans, signs, out-of-range, extra fields, malformed JSON are
  all 422);
* the protected X-PA/X-PT/X-PS credential contract matches the other
  protected write routes (missing/unverified -> 422, malformed -> 422);
* persistence across a restart and a concurrent-create race.

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures).
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
from datetime import datetime, timezone

from sqlalchemy import func, select

from provenance.access_signing import access_message_bytes
from provenance.ids import actor_trust_policy_id
from provenance.models import (
    EVENT_ACTOR_TRUST_POLICY_CREATED,
    ActorTrustPolicy,
    AuditEvent,
)
from provenance.signing import attestation_message_bytes
from provenance.time_utils import parse_rfc3339_utc
from tests.helpers import (
    DIGEST_A,
    content_payload,
    create_actor,
    ed25519_public_key,
    ed25519_sign,
    SEED_A,
    SEED_B,
)

POLICIES_PATH = "/v1/trust-policies"


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
    # A non-revoked attestation gives the subject a current authentication
    # key for the protected contract without any key rotation.
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
    """Two actors, each holding a usable non-revoked attestation key."""
    create_actor(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    _make_attestation(client, "org-1", SEED_A, _make_claim(client, "org-1")["id"])
    _make_attestation(
        client,
        "org-2",
        SEED_B,
        _make_claim(
            client, "org-2", digest=hashlib.sha256(b"content-b").hexdigest()
        )["id"],
    )


# --- Signed-request helpers ---------------------------------------------------


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


def _post_policy(
    client,
    body_obj,
    *,
    actor="org-1",
    seed=SEED_A,
    raw=None,
    content_type="application/json",
    **sign_kwargs,
):
    body = raw if raw is not None else json.dumps(body_obj).encode("utf-8")
    headers = {
        "Content-Type": content_type,
        **_signed_headers(
            "POST", POLICIES_PATH, body, actor=actor, seed=seed, **sign_kwargs
        ),
    }
    return client.post(POLICIES_PATH, content=body, headers=headers)


def _policy_body(subject_id="org-1", threshold=2):
    return {"subject_id": subject_id, "threshold": threshold}


# --- First creation ------------------------------------------------------------


def test_first_creation_returns_201_with_the_public_policy_view(client):
    _world(client)
    resp = _post_policy(client, _policy_body(threshold=3))
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body == {
        "id": actor_trust_policy_id("org-1"),
        "subject_id": "org-1",
        "threshold": 3,
        "enabled": True,
        "created_at": body["created_at"],
    }
    # The timestamp is a strict RFC 3339 UTC instant.
    assert parse_rfc3339_utc(body["created_at"]) is not None
    assert set(body) == {"id", "subject_id", "threshold", "enabled", "created_at"}


def test_response_body_is_compact_utf8_json_ending_in_one_newline(client):
    _world(client)
    resp = _post_policy(client, _policy_body())
    raw = resp.content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert b": " not in raw
    assert b", " not in raw
    raw.decode("utf-8")
    # The numeric threshold renders as a plain JSON integer token.
    assert b'"threshold":2' in raw


def test_policy_and_audit_event_commit_together(client, db_session):
    _world(client)
    resp = _post_policy(client, _policy_body())
    policy_id = resp.json()["id"]

    policies = db_session.execute(select(ActorTrustPolicy)).scalars().all()
    assert len(policies) == 1
    policy = policies[0]
    assert policy.id == policy_id
    assert policy.subject_actor_id == "org-1"
    assert policy.threshold == 2
    assert policy.enabled is True

    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ACTOR_TRUST_POLICY_CREATED,
            AuditEvent.resource_id == policy_id,
        )
    ).scalars().all()
    assert len(events) == 1


def test_threshold_boundaries_1_and_100(client):
    _world(client)
    low = _post_policy(client, _policy_body(threshold=1))
    assert low.status_code == 201
    assert low.json()["threshold"] == 1

    # org-2 independently registers its own policy at the upper bound.
    high = _post_policy(
        client,
        _policy_body(subject_id="org-2", threshold=100),
        actor="org-2",
        seed=SEED_B,
    )
    assert high.status_code == 201, high.text
    assert high.json()["threshold"] == 100
    assert high.json()["id"] != low.json()["id"]


# --- Idempotency and conflict --------------------------------------------------


def test_same_subject_and_threshold_retry_returns_200_original_policy(
    client, db_session
):
    _world(client)
    first = _post_policy(client, _policy_body(threshold=2))
    assert first.status_code == 201

    audit_before = db_session.scalar(
        select(func.count()).select_from(AuditEvent)
    )
    policies_before = db_session.scalar(
        select(func.count()).select_from(ActorTrustPolicy)
    )

    second = _post_policy(client, _policy_body(threshold=2))
    assert second.status_code == 200, second.text
    assert second.json() == first.json()

    assert (
        db_session.scalar(select(func.count()).select_from(ActorTrustPolicy))
        == policies_before
    )
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audit_before
    )


def test_same_subject_different_threshold_is_409_and_writes_nothing(
    client, db_session
):
    _world(client)
    first = _post_policy(client, _policy_body(threshold=2))
    assert first.status_code == 201

    audit_before = db_session.scalar(
        select(func.count()).select_from(AuditEvent)
    )
    policies_before = db_session.scalar(
        select(func.count()).select_from(ActorTrustPolicy)
    )

    for threshold in (1, 3, 100):
        resp = _post_policy(client, _policy_body(threshold=threshold))
        assert resp.status_code == 409, threshold
        error = resp.json()["error"]
        assert error["code"] == "actor_trust_policy_conflict"
        assert error["details"]["subject_actor_id"] == "org-1"

    assert (
        db_session.scalar(select(func.count()).select_from(ActorTrustPolicy))
        == policies_before
    )
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audit_before
    )
    # The original policy is untouched. Reload from the database rather
    # than the session's identity map, which predates the API writes.
    original = db_session.execute(
        select(ActorTrustPolicy).where(
            ActorTrustPolicy.id == first.json()["id"]
        )
    ).scalar_one()
    assert original.threshold == 2


def test_conflict_then_retry_still_returns_the_original_policy(client):
    _world(client)
    first = _post_policy(client, _policy_body(threshold=2))
    assert _post_policy(client, _policy_body(threshold=3)).status_code == 409
    retry = _post_policy(client, _policy_body(threshold=2))
    assert retry.status_code == 200
    assert retry.json() == first.json()


# --- Immutability ---------------------------------------------------------------


def test_policy_has_no_update_delete_disable_or_replace_path(client):
    _world(client)
    created = _post_policy(client, _policy_body())
    policy_id = created.json()["id"]

    # The collection path exists only for POST: another verb there is 405.
    replacement = json.dumps(_policy_body(threshold=4)).encode("utf-8")
    for method in ("PUT", "PATCH", "DELETE"):
        resp = client.request(method, POLICIES_PATH, content=replacement)
        assert resp.status_code == 405, method
        assert resp.json()["error"]["code"] == "method_not_allowed"

    # No item route is registered at all: there is no read, update, delete,
    # disable, or replace endpoint for a policy, so item paths are 404 and
    # can never mutate the immutable row.
    item_paths = [
        f"{POLICIES_PATH}/{policy_id}",
        f"{POLICIES_PATH}/{policy_id}/disable",
        f"{POLICIES_PATH}/{policy_id}/retire",
        f"{POLICIES_PATH}/{policy_id}/replace",
    ]
    for url in item_paths:
        for method, kwargs in (
            ("get", {}),
            ("put", {"json": {"threshold": 4}}),
            ("patch", {"json": {"threshold": 4}}),
            ("delete", {}),
            ("post", {"json": {}}),
        ):
            resp = getattr(client, method)(url, **kwargs)
            assert resp.status_code == 404, (method, url, resp.status_code)


# --- Subject validation ---------------------------------------------------------


def test_unknown_subject_is_422_and_writes_nothing(client, db_session):
    _world(client)
    policies_before = db_session.scalar(
        select(func.count()).select_from(ActorTrustPolicy)
    )
    audit_before = db_session.scalar(
        select(func.count()).select_from(AuditEvent)
    )

    resp = _post_policy(client, _policy_body(subject_id="org-ghost"))
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"
    reason = resp.json()["error"]["details"]["reason"]
    assert reason == "unknown_actor"

    assert (
        db_session.scalar(select(func.count()).select_from(ActorTrustPolicy))
        == policies_before
    )
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audit_before
    )


def test_caller_subject_mismatch_is_422(client):
    _world(client)
    # org-1's key signs a policy that names org-2 as the subject.
    resp = _post_policy(client, _policy_body(subject_id="org-2"))
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"]["reason"] == "actor_mismatch"

    # The inverse is rejected identically and writes nothing.
    resp = _post_policy(
        client,
        _policy_body(subject_id="org-1"),
        actor="org-2",
        seed=SEED_B,
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "actor_mismatch"


# --- Request-body validation ----------------------------------------------------


def test_threshold_must_be_a_plain_decimal_integer(client):
    _world(client)
    for threshold in ("2", 2.0, 2.5, True, False, [2]):
        resp = _post_policy(client, {"subject_id": "org-1", "threshold": threshold})
        assert resp.status_code == 422, threshold
        assert resp.json()["error"]["code"] == "validation_error"
    # An explicit null is invalid just like a missing/typed-wrong value.
    resp = _post_policy(client, {"subject_id": "org-1", "threshold": None})
    assert resp.status_code == 422


def test_threshold_must_be_in_1_to_100(client):
    _world(client)
    for threshold in (0, -1, 101, 1000):
        resp = _post_policy(client, _policy_body(threshold=threshold))
        assert resp.status_code == 422, threshold
        assert resp.json()["error"]["code"] == "validation_error"


def test_missing_fields_are_422(client):
    _world(client)
    for body_obj in ({}, {"subject_id": "org-1"}, {"threshold": 2}):
        resp = _post_policy(client, body_obj)
        assert resp.status_code == 422, body_obj
        assert resp.json()["error"]["code"] == "validation_error"


def test_blank_subject_is_422(client):
    _world(client)
    for value in ("", "   ", "\t"):
        resp = _post_policy(client, _policy_body(subject_id=value))
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_extra_fields_are_422(client):
    _world(client)
    resp = _post_policy(
        client,
        {"subject_id": "org-1", "threshold": 2, "enabled": False, "extra": 1},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_malformed_or_non_object_json_is_422(client):
    _world(client)
    for raw in (
        b"{",
        b'{"subject_id": "org-1", "threshold":',
        b"not json",
        b"[1, 2, 3]",
        b'"a string"',
        b"2",
        b"null",
    ):
        resp = _post_policy(client, None, raw=raw)
        assert resp.status_code == 422, raw


# --- Credential boundary ---------------------------------------------------------


def test_missing_credentials_are_422(client):
    _world(client)
    body = json.dumps(_policy_body()).encode("utf-8")
    assert client.post(POLICIES_PATH, content=body).status_code == 422
    assert (
        client.post(
            POLICIES_PATH,
            content=body,
            headers={"X-PA": "org-1", "X-PT": "2026-09-25T00:00:00Z"},
        ).status_code
        == 422
    )


def test_signature_by_another_actor_is_422(client):
    _world(client)
    # org-2's key authenticates org-2 only; it cannot sign as org-1.
    resp = _post_policy(client, _policy_body(), actor="org-1", seed=SEED_B)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_unknown_actor_header_is_422(client):
    _world(client)
    resp = _post_policy(client, _policy_body(subject_id="org-ghost"),
                        actor="org-ghost", seed=SEED_A)
    # Authentication fails first (no keys for the header actor); either way
    # the boundary is a 422 validation_error and nothing is written.
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_malformed_timestamp_and_signature_are_422(client):
    _world(client)
    bad_ts = _post_policy(
        client, _policy_body(), timestamp="not-a-timestamp"
    )
    assert bad_ts.status_code == 422

    old = datetime(2000, 1, 1, tzinfo=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    out_of_window = _post_policy(client, _policy_body(), timestamp=old)
    assert out_of_window.status_code == 422

    body = json.dumps(_policy_body()).encode("utf-8")
    good = _signed_headers("POST", POLICIES_PATH, body)
    good["X-PS"] = "not-base64!"
    resp = client.post(
        POLICIES_PATH,
        content=body,
        headers={"Content-Type": "application/json", **good},
    )
    assert resp.status_code == 422


def test_signature_must_bind_the_exact_body_and_path(client):
    _world(client)
    body = json.dumps(_policy_body(threshold=2)).encode("utf-8")

    # The signature covers the exact body bytes: signing different bytes is
    # a 422 rather than acceptance of the parsed model.
    resp = _post_policy(
        client,
        None,
        raw=body,
        signed_body=json.dumps(_policy_body(threshold=3)).encode("utf-8"),
    )
    assert resp.status_code == 422

    # The signature binds the request path.
    resp = _post_policy(client, None, raw=body, signed_path="/v1/other")
    assert resp.status_code == 422


def test_failed_requests_write_no_audit_events(client, db_session):
    _world(client)
    audit_before = db_session.scalar(
        select(func.count()).select_from(AuditEvent)
    )
    _post_policy(client, _policy_body(threshold=200))
    _post_policy(client, _policy_body(subject_id="org-ghost"))
    _post_policy(client, _policy_body(subject_id="org-2"))
    _post_policy(client, _policy_body(), actor="org-1", seed=SEED_B)
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audit_before
    )


# --- Persistence and concurrency -------------------------------------------------


def test_policy_persists_across_restart(file_client, tmp_db_url):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    _world(file_client)
    resp = _post_policy(file_client, _policy_body(threshold=4))
    assert resp.status_code == 201
    expected = resp.json()

    second_app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(second_app) as restarted:
        again = _post_policy(
            restarted,
            _policy_body(threshold=4),
            actor="org-1",
            seed=SEED_A,
        )
        assert again.status_code == 200
        assert again.json() == expected


def test_concurrent_first_creations_serialize_to_one_policy(
    file_app, file_client
):
    from provenance import service
    from provenance.schemas import TrustPolicyCreate

    _world(file_client)
    factory = file_app.state.session_factory
    payload = TrustPolicyCreate(subject_id="org-1", threshold=2)
    results: list[tuple[str, bool]] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        session = factory()
        try:
            barrier.wait()
            policy, created = service.create_trust_policy(
                session, payload, "org-1"
            )
            results.append((policy.id, created))
        except Exception as exc:  # pragma: no cover - fails the test below
            errors.append(exc)
        finally:
            session.close()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 4
    assert {policy_id for policy_id, _ in results} == {results[0][0]}
    assert sum(1 for _, created in results if created) == 1

    audit_session = factory()
    try:
        rows = audit_session.execute(select(ActorTrustPolicy)).scalars().all()
        events = audit_session.execute(
            select(AuditEvent).where(
                AuditEvent.event_type == EVENT_ACTOR_TRUST_POLICY_CREATED
            )
        ).scalars().all()
        assert [row.id for row in rows] == [results[0][0]]
        assert [event.resource_id for event in events] == [results[0][0]]
    finally:
        audit_session.close()
