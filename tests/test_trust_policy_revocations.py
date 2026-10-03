"""Tests for subject trust-policy revocations.

Covers ``POST /v1/trust-policy-revocations``: the protected signed-header
contract, first-revocation 201 with the exact public view (compact UTF-8
JSON terminated by exactly one newline), the stable ``tpr_`` identifier,
single-transaction ``actor_trust_policy.revoked`` audit commit, same-reason
retry idempotency (200, no new row/audit/id), the different-reason 409
``trust_policy_revocation_conflict``, the opaque 404 for an unknown policy
or a non-subject caller, strict body validation (missing/extra/duplicate
fields, wrong types, blank or out-of-range reason, non-object JSON), the
verbatim (untrimmed) reason, the no-write guarantee on every failure, the
post-revocation ``policy_missing`` decision that never reads the target,
and the unchanged policy list, creation-idempotency, evaluation, and
migration semantics.

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
from provenance.ids import actor_trust_policy_revocation_id
from provenance.models import (
    EVENT_ACTOR_TRUST_POLICY_CREATED,
    EVENT_ACTOR_TRUST_POLICY_REVOKED,
    ActorTrustPolicy,
    ActorTrustPolicyRevocation,
    AuditEvent,
)
from provenance.signing import attestation_message_bytes
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    DIGEST_C,
    content_payload,
    create_actor,
    ed25519_public_key,
    ed25519_sign,
    SEED_A,
    SEED_B,
)

POLICIES_PATH = "/v1/trust-policies"
REVOCATIONS_PATH = "/v1/trust-policy-revocations"
DECISIONS_PATH = "/v1/trust-decisions"
EVALUATIONS_PATH = "/v1/trust-evaluations"


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


def _post_policy(client, actor="org-1", seed=SEED_A, threshold=2):
    body = json.dumps({"actor_id": actor, "threshold": threshold}).encode()
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", POLICIES_PATH, body, actor=actor, seed=seed),
    }
    resp = client.post(POLICIES_PATH, content=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _post_revocation(client, body_obj, *, actor="org-1", seed=SEED_A, **sign_kwargs):
    body = (
        body_obj
        if isinstance(body_obj, bytes)
        else json.dumps(body_obj, ensure_ascii=False).encode("utf-8")
    )
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", REVOCATIONS_PATH, body, actor=actor, seed=seed,
                          **sign_kwargs),
    }
    return client.post(REVOCATIONS_PATH, content=body, headers=headers)


def _revoke(client, policy_id, reason="no longer needed", **kwargs):
    resp = _post_revocation(
        client, {"policy_id": policy_id, "reason": reason}, **kwargs
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _get_decision(client, target_type, target_id, *, actor="org-1", seed=SEED_A):
    headers = _signed_headers("GET", DECISIONS_PATH, b"", actor=actor, seed=seed)
    return client.get(
        DECISIONS_PATH,
        params={"target_type": target_type, "target_id": target_id},
        headers=headers,
    )


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


def _revocation_rows(session):
    return session.execute(select(ActorTrustPolicyRevocation)).scalars().all()


def _policy_missing_body(target_type, target_id):
    return {
        "target_type": target_type,
        "target_id": target_id,
        "policy_id": None,
        "threshold": None,
        "qualified_signer_count": 0,
        "decision": "untrusted",
        "reason": "policy_missing",
    }


# --- First revocation -----------------------------------------------------------


def test_first_revocation_returns_201_with_exact_public_fields(client):
    _world(client)
    policy = _post_policy(client)
    resp = _post_revocation(
        client, {"policy_id": policy["id"], "reason": "superseded"}
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert list(body) == ["id", "policy_id", "actor_id", "reason", "created_at"]
    assert body["id"].startswith("tpr_")
    assert len(body["id"]) == len("tpr_") + 64
    assert body["policy_id"] == policy["id"]
    assert body["actor_id"] == "org-1"
    assert body["reason"] == "superseded"
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))


def test_revocation_response_is_compact_utf8_json_with_one_trailing_newline(
    client,
):
    _world(client)
    policy = _post_policy(client)
    resp = _post_revocation(
        client, {"policy_id": policy["id"], "reason": "done"}
    )
    assert resp.status_code == 201
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    assert raw == expected


def test_revocation_id_is_the_stable_policy_derived_identifier(client):
    _world(client)
    policy = _post_policy(client)
    body = _revoke(client, policy["id"])
    assert body["id"] == actor_trust_policy_revocation_id(policy["id"])


def test_reason_boundaries_and_unicode_are_stored_verbatim(client, db_session):
    _world(client)
    policy = _post_policy(client)
    # Exactly 1000 Unicode characters, with untrimmed surrounding spaces.
    reason = "  " + "撤" * 996 + "🙂 "
    assert len(reason) == 1000
    body = _revoke(client, policy["id"], reason)
    assert body["reason"] == reason
    rows = _revocation_rows(db_session)
    assert [r.reason for r in rows] == [reason]


def test_single_character_reason_is_accepted(client):
    _world(client)
    policy = _post_policy(client)
    body = _revoke(client, policy["id"], "x")
    assert body["reason"] == "x"


def test_revocation_and_audit_event_commit_in_one_transaction(
    client, db_session
):
    _world(client)
    policy = _post_policy(client, threshold=3)
    created = _revoke(client, policy["id"], "retired")

    rows = _revocation_rows(db_session)
    assert [
        (r.id, r.policy_id, r.actor_id, r.reason) for r in rows
    ] == [(created["id"], policy["id"], "org-1", "retired")]
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ACTOR_TRUST_POLICY_REVOKED
        )
    ).scalars().all()
    assert [e.resource_id for e in events] == [created["id"]]
    assert events[0].created_at.tzinfo.utcoffset(
        events[0].created_at
    ).total_seconds() == 0
    # The policy itself is preserved, unmutated.
    policies = db_session.execute(select(ActorTrustPolicy)).scalars().all()
    assert [(p.id, p.actor_id, p.threshold, p.enabled) for p in policies] == [
        (policy["id"], "org-1", 3, True)
    ]


# --- Idempotency and conflict ----------------------------------------------------


def test_same_reason_retry_returns_200_original_and_no_new_writes(
    client, db_session
):
    _world(client)
    policy = _post_policy(client)
    first = _post_revocation(
        client, {"policy_id": policy["id"], "reason": "done"}
    )
    assert first.status_code == 201
    events_after_first = _audit_count(db_session)

    for _ in range(3):
        retry = _post_revocation(
            client, {"policy_id": policy["id"], "reason": "done"}
        )
        assert retry.status_code == 200
        assert retry.content == first.content

    rows = _revocation_rows(db_session)
    assert [r.id for r in rows] == [first.json()["id"]]
    assert _audit_count(db_session) == events_after_first
    revoked_events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ACTOR_TRUST_POLICY_REVOKED
        )
    ).scalars().all()
    assert len(revoked_events) == 1


def test_different_reason_for_same_policy_is_409_and_writes_nothing(
    client, db_session
):
    _world(client)
    policy = _post_policy(client)
    first = _revoke(client, policy["id"], "first reason")
    events_before = _audit_count(db_session)

    conflict = _post_revocation(
        client, {"policy_id": policy["id"], "reason": "second reason"}
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "trust_policy_revocation_conflict"
    assert conflict.json()["error"]["details"]["policy_id"] == policy["id"]

    rows = _revocation_rows(db_session)
    assert [(r.id, r.reason) for r in rows] == [(first["id"], "first reason")]
    assert _audit_count(db_session) == events_before


def test_reason_is_case_and_whitespace_sensitive_for_idempotency(client):
    _world(client)
    policy = _post_policy(client)
    first = _revoke(client, policy["id"], "done")
    # A trailing space makes it a different reason: conflict, not a retry.
    conflict = _post_revocation(
        client, {"policy_id": policy["id"], "reason": "done "}
    )
    assert conflict.status_code == 409
    conflict = _post_revocation(
        client, {"policy_id": policy["id"], "reason": "Done"}
    )
    assert conflict.status_code == 409
    assert first["reason"] == "done"


def test_subjects_have_independent_revocations(client):
    _world(client)
    one = _post_policy(client, "org-1", SEED_A, 2)
    two = _post_policy(client, "org-2", SEED_B, 2)
    revoked_one = _revoke(client, one["id"], "one")
    revoked_two = _revoke(client, two["id"], "two", actor="org-2", seed=SEED_B)
    assert revoked_one["id"] != revoked_two["id"]
    assert revoked_one["actor_id"] == "org-1"
    assert revoked_two["actor_id"] == "org-2"


# --- Unknown policy and non-subject caller: the shared opaque 404 ----------------


def test_unknown_policy_id_is_404_not_found_and_writes_nothing(
    client, db_session
):
    _world(client)
    events_before = _audit_count(db_session)
    resp = _post_revocation(
        client, {"policy_id": "atp_" + "0" * 64, "reason": "x"}
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"
    assert _revocation_rows(db_session) == []
    assert _audit_count(db_session) == events_before


def test_non_subject_caller_gets_the_same_404_and_writes_nothing(
    client, db_session
):
    _world(client)
    policy = _post_policy(client)  # org-1's policy
    events_before = _audit_count(db_session)
    # org-2 authenticates validly but is not the policy's subject.
    resp = _post_revocation(
        client,
        {"policy_id": policy["id"], "reason": "x"},
        actor="org-2",
        seed=SEED_B,
    )
    assert resp.status_code == 404
    assert resp.json() == {
        "error": {
            "code": "not_found",
            "message": "The requested resource does not exist.",
        }
    }
    assert _revocation_rows(db_session) == []
    assert _audit_count(db_session) == events_before


def test_unknown_policy_and_non_subject_render_identically(client):
    _world(client)
    policy = _post_policy(client)
    unknown = _post_revocation(
        client,
        {"policy_id": "atp_" + "f" * 64, "reason": "x"},
        actor="org-2",
        seed=SEED_B,
    )
    non_subject = _post_revocation(
        client,
        {"policy_id": policy["id"], "reason": "x"},
        actor="org-2",
        seed=SEED_B,
    )
    assert unknown.status_code == 404
    assert non_subject.status_code == 404
    assert unknown.content == non_subject.content


# --- Body validation --------------------------------------------------------------


def test_field_validation_failures_are_422_and_write_nothing(client, db_session):
    _world(client)
    policy = _post_policy(client)
    events_before = _audit_count(db_session)

    cases = [
        {"reason": "x"},  # missing policy_id
        {"policy_id": policy["id"]},  # missing reason
        {},  # both missing
        {"policy_id": policy["id"], "reason": "x", "extra": 1},  # extra field
        {"policy_id": 123, "reason": "x"},  # wrong policy_id type
        {"policy_id": None, "reason": "x"},
        {"policy_id": True, "reason": "x"},
        {"policy_id": ["atp_" + "0" * 64], "reason": "x"},
        {"policy_id": "  ", "reason": "x"},  # blank policy_id
        {"policy_id": "", "reason": "x"},
        {"policy_id": policy["id"], "reason": 5},  # wrong reason type
        {"policy_id": policy["id"], "reason": None},
        {"policy_id": policy["id"], "reason": True},
        {"policy_id": policy["id"], "reason": ""},  # empty reason
        {"policy_id": policy["id"], "reason": "   \t\n  "},  # blank reason
        {"policy_id": policy["id"], "reason": "y" * 1001},  # too long
    ]
    for body in cases:
        resp = _post_revocation(client, body)
        assert resp.status_code == 422, repr(body)
        assert resp.json()["error"]["code"] == "validation_error", repr(body)

    assert _revocation_rows(db_session) == []
    assert _audit_count(db_session) == events_before


def test_duplicate_body_fields_are_422(client, db_session):
    _world(client)
    policy = _post_policy(client)
    events_before = _audit_count(db_session)
    raw = (
        b'{"policy_id":"' + policy["id"].encode()
        + b'","reason":"x","reason":"x"}'
    )
    resp = _post_revocation(client, raw)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    raw = (
        b'{"policy_id":"' + policy["id"].encode()
        + b'","policy_id":"' + policy["id"].encode()
        + b'","reason":"x"}'
    )
    resp = _post_revocation(client, raw)
    assert resp.status_code == 422
    assert _revocation_rows(db_session) == []
    assert _audit_count(db_session) == events_before


def test_non_object_and_malformed_json_bodies_are_422(client, db_session):
    _world(client)
    events_before = _audit_count(db_session)
    for raw in (
        b"{not valid json",
        b"[1,2]",
        b'"just a string"',
        b"42",
        b"null",
        b"true",
        b"",
    ):
        resp = _post_revocation(client, raw)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["code"] == "validation_error", raw
    assert _revocation_rows(db_session) == []
    assert _audit_count(db_session) == events_before


# --- Signed-header contract --------------------------------------------------------


def test_revocation_without_headers_is_422(client, db_session):
    _world(client)
    policy = _post_policy(client)
    events_before = _audit_count(db_session)
    body = json.dumps({"policy_id": policy["id"], "reason": "x"}).encode()
    resp = client.post(
        REVOCATIONS_PATH,
        content=body,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert resp.json()["error"]["details"]["reason"] == "missing_credentials"
    assert _audit_count(db_session) == events_before


def test_revocation_with_wrong_signature_is_422(client):
    _world(client)
    policy = _post_policy(client)
    # SEED_B is not a current key of org-1, so nothing verifies.
    resp = _post_revocation(
        client, {"policy_id": policy["id"], "reason": "x"}, seed=SEED_B
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )


def test_revocation_rejects_stale_and_malformed_timestamps(client):
    _world(client)
    policy = _post_policy(client)
    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    cases = {
        stale: "timestamp_out_of_window",
        "2026-09-20T12:00:00": "invalid_timestamp",
        "not-a-timestamp": "invalid_timestamp",
    }
    for raw, reason in cases.items():
        resp = _post_revocation(
            client, {"policy_id": policy["id"], "reason": "x"}, timestamp=raw
        )
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["details"]["reason"] == reason, raw


def test_revocation_signature_binds_to_method_path_and_body(client):
    _world(client)
    policy = _post_policy(client)
    wrong_method = _post_revocation(
        client, {"policy_id": policy["id"], "reason": "x"}, signed_method="GET"
    )
    assert wrong_method.status_code == 422
    wrong_path = _post_revocation(
        client,
        {"policy_id": policy["id"], "reason": "x"},
        signed_path="/v1/trust-policies",
    )
    assert wrong_path.status_code == 422
    wrong_body = _post_revocation(
        client,
        {"policy_id": policy["id"], "reason": "x"},
        signed_body=b'{"policy_id":"atp_other","reason":"x"}',
    )
    assert wrong_body.status_code == 422


# --- Post-revocation decisions and preserved surfaces -------------------------------


def test_revoked_subject_decision_is_policy_missing_without_target_read(client):
    _world(client)
    policy = _post_policy(client, threshold=1)
    target = _make_claim(client, "org-1", digest=DIGEST_C)
    _make_attestation(client, "org-1", SEED_A, target["id"])
    _make_attestation(client, "org-2", SEED_B, target["id"])

    # Before revocation the policy decides normally.
    before = _get_decision(client, "claim", target["id"])
    assert before.status_code == 200
    assert before.json()["decision"] == "trusted"
    assert before.json()["reason"] == "threshold_met"

    _revoke(client, policy["id"], "retired")

    # An existing, well-attested target now yields the policy_missing shape.
    after = _get_decision(client, "claim", target["id"])
    assert after.status_code == 200
    assert after.json() == _policy_missing_body("claim", target["id"])

    # A nonexistent target is never looked up: no 404, same shape.
    ghost = _get_decision(client, "claim", "clm_ghost")
    assert ghost.status_code == 200
    assert ghost.json() == _policy_missing_body("claim", "clm_ghost")


def test_revocation_writes_no_decision_side_effects(client, db_session):
    _world(client)
    policy = _post_policy(client)
    _revoke(client, policy["id"])
    events_before = _audit_count(db_session)
    resp = _get_decision(client, "claim", "clm_ghost")
    assert resp.status_code == 200
    assert _audit_count(db_session) == events_before


def test_unrevoked_subject_decisions_are_unchanged(client):
    _world(client)
    one = _post_policy(client, "org-1", SEED_A, 1)
    _post_policy(client, "org-2", SEED_B, 2)
    target = _make_claim(client, "org-1", digest=DIGEST_C)
    _make_attestation(client, "org-1", SEED_A, target["id"])

    _revoke(client, one["id"], "retired")

    # org-2's unrevoked policy still decides normally.
    resp = _get_decision(client, "claim", target["id"], actor="org-2", seed=SEED_B)
    assert resp.status_code == 200
    body = resp.json()
    assert body["policy_id"] is not None
    assert body["threshold"] == 2
    assert body["qualified_signer_count"] == 1
    assert body["decision"] == "untrusted"
    assert body["reason"] == "below_threshold"


def test_policy_list_and_creation_idempotency_survive_revocation(client):
    _world(client)
    policy = _post_policy(client, threshold=2)
    _revoke(client, policy["id"], "retired")

    # The paginated view still returns the policy, unchanged.
    listing = client.get(POLICIES_PATH, params={"actor_id": "org-1"})
    assert listing.status_code == 200
    body = listing.json()
    assert body["count"] == 1
    assert [item["id"] for item in body["items"]] == [policy["id"]]
    assert body["items"][0]["threshold"] == 2
    assert body["items"][0]["enabled"] is True

    # Creation idempotency is unchanged: same threshold -> 200 original.
    retry_body = json.dumps({"actor_id": "org-1", "threshold": 2}).encode()
    retry = client.post(
        POLICIES_PATH,
        content=retry_body,
        headers={
            "Content-Type": "application/json",
            **_signed_headers("POST", POLICIES_PATH, retry_body),
        },
    )
    assert retry.status_code == 200
    assert retry.json()["id"] == policy["id"]

    # A different threshold is still the 409 conflict.
    other_body = json.dumps({"actor_id": "org-1", "threshold": 5}).encode()
    conflict = client.post(
        POLICIES_PATH,
        content=other_body,
        headers={
            "Content-Type": "application/json",
            **_signed_headers("POST", POLICIES_PATH, other_body),
        },
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "actor_trust_policy_conflict"


def test_trust_evaluations_are_unaffected_by_revocation(client):
    _world(client)
    policy = _post_policy(client, threshold=1)
    target = _make_claim(client, "org-1", digest=DIGEST_C)
    _make_attestation(client, "org-1", SEED_A, target["id"])
    _revoke(client, policy["id"])

    # The unauthenticated evaluation surface does not consult policies.
    resp = client.get(
        EVALUATIONS_PATH,
        params={"target_type": "claim", "target_id": target["id"]},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "target_type": "claim",
        "target_id": target["id"],
        "min_signers": 1,
        "qualified_signer_count": 1,
        "decision": "trusted",
    }

    batch = client.post(
        "/v1/trust-evaluation-batches",
        json={
            "items": [
                {"target_type": "claim", "target_id": target["id"]},
            ]
        },
    )
    assert batch.status_code == 200
    assert batch.json()["items"][0]["decision"] == "trusted"


def test_historical_audit_events_are_preserved(client, db_session):
    _world(client)
    policy = _post_policy(client)
    _revoke(client, policy["id"], "retired")
    created_events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ACTOR_TRUST_POLICY_CREATED
        )
    ).scalars().all()
    revoked_events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ACTOR_TRUST_POLICY_REVOKED
        )
    ).scalars().all()
    assert [e.resource_id for e in created_events] == [policy["id"]]
    assert len(revoked_events) == 1


# --- Concurrency and persistence ----------------------------------------------------


def test_concurrent_identical_revocations_yield_one_201(
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
    policy = _post_policy(file_client)

    from provenance import service
    from provenance.schemas import ActorTrustPolicyRevocationCreate

    factory = file_app.state.session_factory
    payload = ActorTrustPolicyRevocationCreate(
        policy_id=policy["id"], reason="retired"
    )
    results: list[tuple[str, bool]] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        session = factory()
        try:
            barrier.wait()
            record, created = service.create_actor_trust_policy_revocation(
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
            select(ActorTrustPolicyRevocation)
        ).scalars().all()
        events = audit_session.execute(
            select(AuditEvent).where(
                AuditEvent.event_type == EVENT_ACTOR_TRUST_POLICY_REVOKED
            )
        ).scalars().all()
        assert [r.id for r in rows] == [results[0][0]]
        assert [e.resource_id for e in events] == [results[0][0]]
    finally:
        audit_session.close()


def test_revocation_and_decision_persist_across_restart(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    _world(file_client)
    policy = _post_policy(file_client, threshold=1)
    target = _make_claim(file_client, "org-1", digest=DIGEST_C)
    _make_attestation(file_client, "org-1", SEED_A, target["id"])
    created = _revoke(file_client, policy["id"], "retired")

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        # The revocation record survives, byte-identical on the idempotent
        # retry (200, not a new 201).
        retry = _post_revocation(
            client, {"policy_id": policy["id"], "reason": "retired"}
        )
        assert retry.status_code == 200
        assert retry.json() == created

        # The decision is still the policy_missing shape after restart.
        decision = _get_decision(client, "claim", target["id"])
        assert decision.status_code == 200
        assert decision.json() == _policy_missing_body("claim", target["id"])

        # The policy list still shows the preserved policy.
        listing = client.get(POLICIES_PATH, params={"actor_id": "org-1"})
        assert [item["id"] for item in listing.json()["items"]] == [
            policy["id"]
        ]


# --- Legacy migration ----------------------------------------------------------------


def test_legacy_v2_database_gains_the_revocation_table_untouched(tmp_path):
    import sqlite3 as _sqlite3

    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    # A simulated pre-v3 database: build the current schema once, then drop
    # the revocation table and its v3 ledger row. Existing policies and
    # audit rows stay exactly as they were.
    db_path = tmp_path / "legacy.db"
    url = f"sqlite:///{db_path.as_posix()}"
    app = create_app(Settings(database_url=url))
    with TestClient(app) as client:
        _world(client)
        policy = _post_policy(client, threshold=4)
    con = _sqlite3.connect(db_path)
    try:
        policies_before = con.execute(
            "SELECT id, actor_id, threshold, enabled, created_at "
            "FROM actor_trust_policies"
        ).fetchall()
        audits_before = con.execute(
            "SELECT event_type, resource_id, created_at FROM audit_events"
        ).fetchall()
        con.execute("DROP TABLE actor_trust_policy_revocations")
        con.execute("DELETE FROM schema_migrations WHERE version = 3")
        con.commit()
    finally:
        con.close()

    restarted = create_app(Settings(database_url=url))
    with TestClient(restarted) as client:
        # The upgrade added the table without rewriting policies or audit.
        con = _sqlite3.connect(db_path)
        try:
            assert "actor_trust_policy_revocations" in {
                r[0]
                for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            assert con.execute(
                "SELECT id, actor_id, threshold, enabled, created_at "
                "FROM actor_trust_policies"
            ).fetchall() == policies_before
            assert con.execute(
                "SELECT event_type, resource_id, created_at FROM audit_events"
            ).fetchall() == audits_before
            assert [
                r[0]
                for r in con.execute(
                    "SELECT version FROM schema_migrations ORDER BY version"
                )
            ] == [1, 2, 3]
        finally:
            con.close()

        # The upgraded database serves the new endpoint immediately.
        created = _revoke(client, policy["id"], "after upgrade")
        assert created["policy_id"] == policy["id"]
        decision = _get_decision(client, "claim", "clm_ghost")
        assert decision.json() == _policy_missing_body("claim", "clm_ghost")
