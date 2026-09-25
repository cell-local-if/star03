"""Tests for protected authorization decisions.

Covers ``GET /v1/trust-decisions``: decisions computed only from the calling
subject's current immutable policy, the ``trusted``/``untrusted`` outcomes
with ``threshold_met``/``below_threshold``/``policy_missing`` reasons,
qualified-signer counting (verified, non-revoked, deduplicated by signer),
the no-policy path that never looks up or leaks the target, the type-matched
404s once a policy exists, the opaque-404/422 credential boundary, strict
query-parameter and body validation, the compact one-newline JSON wire
format, and the strict read-only guarantee.

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures).
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from provenance.access_signing import access_message_bytes
from provenance.models import ActorTrustPolicy, Attestation, AuditEvent
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

DECISIONS_PATH = "/v1/trust-decisions"
POLICIES_PATH = "/v1/trust-policies"
SEED_C = b"test-ed25519-seed-c-00000000000000"[:32]
SEED_D = b"test-ed25519-seed-d-00000000000000"[:32]
EVIDENCE_DIGEST = hashlib.sha256(b"evidence-decision").hexdigest()


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


def _make_bundle(client, claim_id):
    resp = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim_id,
            "evidence_type": "raw_capture",
            "digest_algorithm": "sha256",
            "digest_hex": EVIDENCE_DIGEST,
            "media_type": "image/jpeg",
            "metadata": {"source": "camera-1"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _attest(client, target_type, target_id, actor_id, seed):
    signature = ed25519_sign(
        seed, attestation_message_bytes(target_type, target_id, actor_id)
    )
    resp = client.post(
        "/v1/attestations",
        json={
            "target_type": target_type,
            "target_id": target_id,
            "signer_actor_id": actor_id,
            "public_key": base64.b64encode(ed25519_public_key(seed)).decode(),
            "signature": base64.b64encode(signature).decode(),
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _world(client):
    """Three actors; org-1 (the usual caller) holds a current key."""
    create_actor(client)  # org-1
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    create_actor(client, actor_id="org-3", name="Third Org", type="organization")
    # org-1's own attestation gives org-1 a current authentication key.
    _attest(client, "claim", _make_claim(client, "org-1")["id"], "org-1", SEED_A)


def _target_claim(client):
    """A second claim (the decision target), distinct from org-1's key claim."""
    return _make_claim(client, "org-1", digest=DIGEST_C)


# --- Signed-request helpers ----------------------------------------------------


def _signed_headers(
    method,
    path,
    body,
    *,
    actor="org-1",
    seed=SEED_A,
    timestamp=None,
):
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    message = access_message_bytes(
        method, path, ts, hashlib.sha256(body).hexdigest()
    )
    signature = base64.b64encode(ed25519_sign(seed, message)).decode("ascii")
    return {"X-PA": actor, "X-PT": ts, "X-PS": signature}


def _create_policy(client, actor, seed, threshold):
    body = json.dumps({"actor_id": actor, "threshold": threshold}).encode()
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", POLICIES_PATH, body, actor=actor, seed=seed),
    }
    resp = client.post(POLICIES_PATH, content=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _get_decision(client, target_type, target_id, *, actor="org-1", seed=SEED_A,
                  headers=None, params=None):
    if headers is None:
        headers = _signed_headers("GET", DECISIONS_PATH, b"", actor=actor, seed=seed)
    if params is None:
        params = {"target_type": target_type, "target_id": target_id}
    return client.get(DECISIONS_PATH, params=params, headers=headers)


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


# --- No policy: policy_missing without any target lookup ------------------------


def test_no_policy_is_policy_missing_and_never_looks_up_the_target(client):
    _world(client)
    # The target does not even exist: without a policy there is no lookup,
    # so the response is 200 policy_missing rather than any 404.
    resp = _get_decision(client, "claim", "clm_ghost")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "target_type": "claim",
        "target_id": "clm_ghost",
        "policy_id": None,
        "threshold": None,
        "qualified_signer_count": 0,
        "decision": "untrusted",
        "reason": "policy_missing",
    }


def test_no_policy_reports_zero_signers_even_when_proofs_exist(client):
    _world(client)
    target = _target_claim(client)
    _attest(client, "claim", target["id"], "org-2", SEED_B)
    _attest(client, "claim", target["id"], "org-3", SEED_C)
    resp = _get_decision(client, "claim", target["id"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["reason"] == "policy_missing"
    assert body["qualified_signer_count"] == 0
    assert body["decision"] == "untrusted"


def test_decision_response_is_compact_utf8_json_with_one_newline(client):
    _world(client)
    resp = _get_decision(client, "claim", "clm_ghost")
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    assert b'"policy_id":null,' in raw
    assert b'"threshold":null,' in raw
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    assert raw == expected


# --- Decisions under a policy ----------------------------------------------------


def test_below_threshold_is_untrusted_with_policy_fields(client):
    _world(client)
    policy = _create_policy(client, "org-1", SEED_A, 2)
    target = _target_claim(client)
    _attest(client, "claim", target["id"], "org-2", SEED_B)

    resp = _get_decision(client, "claim", target["id"])
    assert resp.status_code == 200
    assert resp.json() == {
        "target_type": "claim",
        "target_id": target["id"],
        "policy_id": policy["id"],
        "threshold": 2,
        "qualified_signer_count": 1,
        "decision": "untrusted",
        "reason": "below_threshold",
    }


def test_met_threshold_is_trusted(client):
    _world(client)
    _create_policy(client, "org-1", SEED_A, 2)
    target = _target_claim(client)
    _attest(client, "claim", target["id"], "org-2", SEED_B)
    _attest(client, "claim", target["id"], "org-3", SEED_C)

    resp = _get_decision(client, "claim", target["id"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["qualified_signer_count"] == 2
    assert body["decision"] == "trusted"
    assert body["reason"] == "threshold_met"


def test_same_signer_under_two_keys_counts_once(client):
    _world(client)
    _create_policy(client, "org-1", SEED_A, 2)
    target = _target_claim(client)
    # org-2 attests the target under two different keys: one qualified signer.
    _attest(client, "claim", target["id"], "org-2", SEED_B)
    _attest(client, "claim", target["id"], "org-2", SEED_D)

    resp = _get_decision(client, "claim", target["id"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["qualified_signer_count"] == 1
    assert body["decision"] == "untrusted"
    assert body["reason"] == "below_threshold"


def test_revoked_attestation_no_longer_counts(client):
    _world(client)
    _create_policy(client, "org-1", SEED_A, 2)
    target = _target_claim(client)
    att_b = _attest(client, "claim", target["id"], "org-2", SEED_B)
    _attest(client, "claim", target["id"], "org-3", SEED_C)
    revoked = client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": att_b["id"],
            "revoker_actor_id": "org-2",
            "reason": "key rotation",
        },
    )
    assert revoked.status_code == 201

    resp = _get_decision(client, "claim", target["id"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["qualified_signer_count"] == 1
    assert body["decision"] == "untrusted"
    assert body["reason"] == "below_threshold"


def test_evidence_bundle_target_is_decided(client):
    _world(client)
    _create_policy(client, "org-1", SEED_A, 2)
    bundle = _make_bundle(client, _target_claim(client)["id"])
    _attest(client, "evidence_bundle", bundle["id"], "org-2", SEED_B)
    _attest(client, "evidence_bundle", bundle["id"], "org-3", SEED_C)

    resp = _get_decision(client, "evidence_bundle", bundle["id"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["target_type"] == "evidence_bundle"
    assert body["decision"] == "trusted"
    assert body["reason"] == "threshold_met"


def test_decision_uses_only_the_callers_own_policy(client):
    _world(client)
    # org-1 requires two signers; org-2 requires one. org-2 attests its own
    # claim so it holds a current key; org-3 alone attests the target.
    _create_policy(client, "org-1", SEED_A, 2)
    _attest(client, "claim", _make_claim(client, "org-2", digest=DIGEST_B)["id"],
            "org-2", SEED_B)
    _create_policy(client, "org-2", SEED_B, 1)
    target = _target_claim(client)
    _attest(client, "claim", target["id"], "org-3", SEED_C)

    as_org1 = _get_decision(client, "claim", target["id"])
    assert as_org1.json()["decision"] == "untrusted"
    assert as_org1.json()["reason"] == "below_threshold"

    as_org2 = _get_decision(client, "claim", target["id"], actor="org-2", seed=SEED_B)
    assert as_org2.json()["decision"] == "trusted"
    assert as_org2.json()["reason"] == "threshold_met"
    assert as_org1.json()["policy_id"] != as_org2.json()["policy_id"]


# --- Missing targets once a policy exists ----------------------------------------


def test_unknown_claim_is_404_claim_not_found_with_policy(client, db_session):
    _world(client)
    _create_policy(client, "org-1", SEED_A, 1)
    events_before = _audit_count(db_session)
    resp = _get_decision(client, "claim", "clm_ghost")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "claim_not_found"
    assert _audit_count(db_session) == events_before


def test_unknown_evidence_bundle_is_404_with_policy(client):
    _world(client)
    _create_policy(client, "org-1", SEED_A, 1)
    resp = _get_decision(client, "evidence_bundle", "evb_ghost")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "evidence_bundle_not_found"


def test_target_types_never_cross_match(client):
    _world(client)
    _create_policy(client, "org-1", SEED_A, 1)
    claim = _target_claim(client)
    bundle = _make_bundle(client, claim["id"])

    # An existing bundle id declared as a claim, and vice versa, is a 404.
    as_claim = _get_decision(client, "claim", bundle["id"])
    assert as_claim.status_code == 404
    assert as_claim.json()["error"]["code"] == "claim_not_found"
    as_bundle = _get_decision(client, "evidence_bundle", claim["id"])
    assert as_bundle.status_code == 404
    assert as_bundle.json()["error"]["code"] == "evidence_bundle_not_found"


# --- Credential boundary ------------------------------------------------------------


def test_missing_or_unverifiable_credentials_are_the_opaque_404(client):
    _world(client)
    _create_policy(client, "org-1", SEED_A, 1)
    target = _target_claim(client)

    # No headers at all.
    assert client.get(
        DECISIONS_PATH,
        params={"target_type": "claim", "target_id": target["id"]},
    ).status_code == 404
    # A well-formed signature no current key of the claimed actor verifies.
    assert _get_decision(
        client, "claim", target["id"], actor="org-1", seed=SEED_B
    ).status_code == 404
    # A ghost actor id with a well-formed signature.
    assert _get_decision(
        client, "claim", target["id"], actor="ghost", seed=SEED_A
    ).status_code == 404

    resp = _get_decision(client, "claim", target["id"], actor="ghost", seed=SEED_A)
    assert resp.json()["error"]["code"] == "not_found"


def test_malformed_credentials_are_422(client):
    _world(client)
    target = _target_claim(client)
    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    for ts in (stale, "not-a-timestamp", "2026-09-20T12:00:00+01:00"):
        headers = _signed_headers("GET", DECISIONS_PATH, b"", timestamp=ts)
        resp = _get_decision(client, "claim", target["id"], headers=headers)
        assert resp.status_code == 422, ts
        assert resp.json()["error"]["code"] == "validation_error"

    for raw in ("@@@", "aGVsbG8", base64.b64encode(b"x" * 63).decode()):
        headers = _signed_headers("GET", DECISIONS_PATH, b"")
        headers["X-PS"] = raw
        resp = _get_decision(client, "claim", target["id"], headers=headers)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["code"] == "validation_error"


# --- Parameter and body validation ----------------------------------------------------


def test_parameter_validation_failures_are_422_even_without_credentials(client):
    _world(client)
    target = _target_claim(client)
    base = {"target_type": "claim", "target_id": target["id"]}
    cases = [
        {"target_id": target["id"]},  # missing target_type
        {"target_type": "claim"},  # missing target_id
        {**base, "target_type": "Claim"},
        {**base, "target_type": "attestation"},
        {**base, "target_id": "  "},
        {**base, "min_signers": "2"},  # undeclared parameter
        {**base, "target_types": "claim"},
    ]
    for params in cases:
        # No credentials at all: validation still precedes authentication.
        resp = client.get(DECISIONS_PATH, params=params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error", params


def test_repeated_parameter_is_422(client):
    _world(client)
    resp = client.get(
        DECISIONS_PATH + "?target_type=claim&target_type=claim&target_id=x"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_non_empty_body_is_422_even_when_well_signed(client):
    _world(client)
    target = _target_claim(client)
    for body in (b"{not valid json", b"{}", b" "):
        headers = {
            "Content-Type": "application/json",
            **_signed_headers("GET", DECISIONS_PATH, body),
        }
        resp = client.request(
            "GET",
            DECISIONS_PATH,
            params={"target_type": "claim", "target_id": target["id"]},
            content=body,
            headers=headers,
        )
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error", body


# --- Read-only guarantee ---------------------------------------------------------------


def test_decisions_write_no_resources_or_audit(client, db_session):
    _world(client)
    _create_policy(client, "org-1", SEED_A, 1)
    target = _target_claim(client)
    _attest(client, "claim", target["id"], "org-2", SEED_B)
    # org-3 holds a current key (its own attestation) but no policy.
    _attest(client, "claim", _make_claim(client, "org-3", digest=DIGEST_B)["id"],
            "org-3", SEED_C)

    events_before = _audit_count(db_session)
    policies_before = len(db_session.execute(select(ActorTrustPolicy)).scalars().all())
    attestations_before = len(db_session.execute(select(Attestation)).scalars().all())

    responses = [
        _get_decision(client, "claim", target["id"]),  # trusted
        _get_decision(client, "claim", "clm_ghost"),  # 404 claim_not_found
        _get_decision(client, "evidence_bundle", "evb_ghost"),  # 404
        client.get(DECISIONS_PATH),  # 422 missing params
        # A no-policy caller decision (org-3 has a key but no policy).
        _get_decision(client, "claim", target["id"], actor="org-3", seed=SEED_C),
    ]
    assert [r.status_code for r in responses] == [200, 404, 404, 422, 200]

    db_session.expire_all()
    assert _audit_count(db_session) == events_before
    assert len(db_session.execute(select(ActorTrustPolicy)).scalars().all()) == (
        policies_before
    )
    assert len(db_session.execute(select(Attestation)).scalars().all()) == (
        attestations_before
    )
