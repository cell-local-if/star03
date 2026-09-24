"""Tests for subject-scoped trust authorization decisions.

Covers ``GET /v1/trust-decisions?actor_id=...&target_type=...&target_id=...``:

* the response shape and compact UTF-8 JSON body terminated by one newline;
* qualified signers following the attestation semantics (verified and
  non-revoked attestations of the exact target, deduplicated by signing
  actor), the ``trusted``/``untrusted`` threshold comparison, and
  recomputation from current policy and revocation state across a restart;
* the no-policy branch: 200 ``untrusted`` / ``policy_missing`` / empty
  policy id, without any target lookup (a missing target gives the same
  result), ahead of the resource-missing branch;
* unknown targets once a policy exists (404 claim_not_found /
  evidence_bundle_not_found), and no partial results for evidence bundles;
* the protected-read credential boundary (malformed credentials 422,
  missing/unverifiable credentials and a caller querying another subject
  are the opaque 404), parameter validation (422), read-only behavior (no
  resource or audit writes on success, failure, or the empty-policy
  branch), and the absence of any raw material in the response.

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures); only fixed seed-derived public keys are used.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from provenance.access_signing import access_message_bytes
from provenance.models import (
    ActorTrustPolicy,
    AttestationRevocation,
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

DECISIONS_PATH = "/v1/trust-decisions"
POLICIES_PATH = "/v1/trust-policies"

# Additional fixed signers, each with their own bootstrap key.
SEED_S2 = b"test-ed25519-signer-s2-0000000000"[:32]
SEED_S3 = b"test-ed25519-signer-s3-0000000000"[:32]
# A second key held by the same actor as SEED_A (different key pair).
SEED_A2 = b"test-ed25519-second-key-a2-00000000"[:32]


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


def _make_attestation(client, signer, seed, target_type, target_id):
    signature = ed25519_sign(
        seed, attestation_message_bytes(target_type, target_id, signer)
    )
    resp = client.post(
        "/v1/attestations",
        json={
            "target_type": target_type,
            "target_id": target_id,
            "signer_actor_id": signer,
            "public_key": base64.b64encode(ed25519_public_key(seed)).decode(),
            "signature": base64.b64encode(signature).decode(),
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _bootstrap_signer(client, signer, seed, digest=DIGEST_A):
    """Create an actor with one non-revoked attestation (auth key bootstrap)."""
    create_actor(
        client, actor_id=signer, name=signer, type="organization"
    )
    claim = _make_claim(client, signer, digest=digest)
    _make_attestation(client, signer, seed, "claim", claim["id"])
    return claim


def _make_bundle(client, claim_id, evidence_type="measurement",
                 digest=DIGEST_B, media_type="application/octet-stream"):
    resp = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim_id,
            "evidence_type": evidence_type,
            "digest_algorithm": "sha256",
            "digest_hex": digest,
            "media_type": media_type,
            "metadata": {},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make_policy(client, actor_id, threshold, *, seed):
    body = (
        f'{{"actor_id":"{actor_id}","threshold":{threshold}}}'
    ).encode()
    ts = _now()
    message = access_message_bytes(
        "POST", POLICIES_PATH, ts, hashlib.sha256(body).hexdigest()
    )
    headers = {
        "Content-Type": "application/json",
        "X-PA": actor_id,
        "X-PT": ts,
        "X-PS": base64.b64encode(ed25519_sign(seed, message)).decode("ascii"),
    }
    resp = client.post(POLICIES_PATH, content=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _signed_get(client, actor, seed, *, query="", timestamp=None,
                path=DECISIONS_PATH, signed_timestamp=None, signed_path=None):
    ts = timestamp or _now()
    signed_ts = signed_timestamp if signed_timestamp is not None else ts
    message = access_message_bytes(
        "GET",
        signed_path if signed_path is not None else path,
        signed_ts,
        hashlib.sha256(b"").hexdigest(),
    )
    headers = {
        "X-PA": actor,
        "X-PT": ts,
        "X-PS": base64.b64encode(ed25519_sign(seed, message)).decode("ascii"),
    }
    return client.get(f"{path}?{query}" if query else path, headers=headers)


def _decide(client, actor, seed, target_type, target_id):
    query = (
        f"actor_id={actor}&target_type={target_type}&target_id={target_id}"
    )
    return _signed_get(client, actor, seed, query=query)


def _revoke(client, attestation_id, revoker, reason="no longer relied upon"):
    resp = client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": attestation_id,
            "revoker_actor_id": revoker,
            "reason": reason,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# --- Response shape and wire format -------------------------------------------


def test_decision_success_body_shape_and_wire_format(client, db_session):
    _bootstrap_signer(client, "org-1", SEED_A, digest=DIGEST_A)
    policy = _make_policy(client, "org-1", 1, seed=SEED_A)
    target = _make_claim(client, "org-1", digest=DIGEST_B)
    _make_attestation(client, "org-1", SEED_A, "claim", target["id"])

    resp = _decide(client, "org-1", SEED_A, "claim", target["id"])
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/json")
    raw = resp.content
    # Compact separators, exactly one trailing newline, no whitespace padding.
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b": " not in raw and b", " not in raw
    body = resp.json()
    assert set(body) == {
        "policy_id",
        "threshold",
        "qualified_signer_count",
        "decision",
    }
    assert body["policy_id"] == policy["id"]
    assert body["threshold"] == 1
    assert body["qualified_signer_count"] == 1
    assert body["decision"] == "trusted"
    # No material, credential, or key fields ever appear.
    text = raw.decode("utf-8")
    for forbidden in ("signature", "public_key", "payload", "X-PA"):
        assert forbidden not in text


# --- Qualified signer semantics ------------------------------------------------


def test_qualified_signers_are_deduplicated_by_signing_actor(client):
    _bootstrap_signer(client, "org-1", SEED_A)
    _make_policy(client, "org-1", 2, seed=SEED_A)
    target = _make_claim(client, "org-1", digest=DIGEST_B)
    # Two attestations by the same actor under DIFFERENT keys still count
    # once: the threshold counts distinct signing subjects, not keys.
    first = _make_attestation(client, "org-1", SEED_A, "claim", target["id"])
    second_key = _make_attestation(
        client, "org-1", SEED_A2, "claim", target["id"]
    )
    assert first["id"] != second_key["id"]
    assert first["public_key"] != second_key["public_key"]
    assert first["signer_actor_id"] == second_key["signer_actor_id"] == "org-1"

    resp = _decide(client, "org-1", SEED_A, "claim", target["id"])
    assert resp.status_code == 200
    assert resp.json()["qualified_signer_count"] == 1
    assert resp.json()["decision"] == "untrusted"


def test_independent_signers_reach_the_threshold(client):
    _bootstrap_signer(client, "org-1", SEED_A)
    _bootstrap_signer(client, "s2", SEED_S2, digest=DIGEST_B)
    _bootstrap_signer(client, "s3", SEED_S3, digest=DIGEST_C)
    _make_policy(client, "org-1", 2, seed=SEED_A)
    # The evaluated target is a separate claim owned by org-1.
    target = _make_claim(
        client,
        "org-1",
        digest=hashlib.sha256(b"eval-target").hexdigest(),
    )
    _make_attestation(client, "org-1", SEED_A, "claim", target["id"])

    one = _decide(client, "org-1", SEED_A, "claim", target["id"])
    assert one.json() == {
        "policy_id": one.json()["policy_id"],
        "threshold": 2,
        "qualified_signer_count": 1,
        "decision": "untrusted",
    }

    _make_attestation(client, "s2", SEED_S2, "claim", target["id"])
    two = _decide(client, "org-1", SEED_A, "claim", target["id"])
    assert two.json()["qualified_signer_count"] == 2
    assert two.json()["decision"] == "trusted"

    # A third independent signer does not change the verdict.
    _make_attestation(client, "s3", SEED_S3, "claim", target["id"])
    three = _decide(client, "org-1", SEED_A, "claim", target["id"])
    assert three.json()["qualified_signer_count"] == 3
    assert three.json()["decision"] == "trusted"


def test_revocation_drops_a_qualified_signer_and_flips_the_decision(client):
    _bootstrap_signer(client, "org-1", SEED_A)
    _bootstrap_signer(client, "s2", SEED_S2, digest=DIGEST_B)
    _make_policy(client, "org-1", 2, seed=SEED_A)
    target = _make_claim(
        client, "org-1", digest=hashlib.sha256(b"revoke-target").hexdigest()
    )
    att1 = _make_attestation(client, "org-1", SEED_A, "claim", target["id"])
    _make_attestation(client, "s2", SEED_S2, "claim", target["id"])
    trusted = _decide(client, "org-1", SEED_A, "claim", target["id"])
    assert trusted.json()["decision"] == "trusted"

    # Revoke one of the two signers' attestations.
    _revoke(client, att1["id"], "org-1")
    untrusted = _decide(client, "org-1", SEED_A, "claim", target["id"])
    assert untrusted.json()["qualified_signer_count"] == 1
    assert untrusted.json()["decision"] == "untrusted"
    # The revoked attestation is retained in the store.
    assert client.get(f"/v1/attestations/{att1['id']}").status_code == 200


def test_evidence_bundle_target_counts_attestations_of_the_bundle(client):
    _bootstrap_signer(client, "org-1", SEED_A)
    _bootstrap_signer(client, "s2", SEED_S2, digest=DIGEST_B)
    _make_policy(client, "org-1", 2, seed=SEED_A)
    claim = _make_claim(client, "org-1", digest=DIGEST_C)
    bundle = _make_bundle(client, claim["id"])
    # Attestations of the claim do not count toward the bundle target.
    _make_attestation(client, "org-1", SEED_A, "claim", claim["id"])
    before = _decide(client, "org-1", SEED_A, "evidence_bundle", bundle["id"])
    assert before.json()["qualified_signer_count"] == 0
    assert before.json()["decision"] == "untrusted"

    _make_attestation(
        client, "org-1", SEED_A, "evidence_bundle", bundle["id"]
    )
    _make_attestation(
        client, "s2", SEED_S2, "evidence_bundle", bundle["id"]
    )
    after = _decide(client, "org-1", SEED_A, "evidence_bundle", bundle["id"])
    assert after.json()["qualified_signer_count"] == 2
    assert after.json()["decision"] == "trusted"


def test_decision_is_recomputed_after_restart(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    _bootstrap_signer(file_client, "org-1", SEED_A)
    _bootstrap_signer(file_client, "s2", SEED_S2, digest=DIGEST_B)
    _make_policy(file_client, "org-1", 2, seed=SEED_A)
    target = _make_claim(
        file_client, "org-1",
        digest=hashlib.sha256(b"restart-target").hexdigest(),
    )
    _make_attestation(file_client, "org-1", SEED_A, "claim", target["id"])
    _make_attestation(file_client, "s2", SEED_S2, "claim", target["id"])

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        resp = _decide(client, "org-1", SEED_A, "claim", target["id"])
        assert resp.status_code == 200
        body = resp.json()
        assert body["qualified_signer_count"] == 2
        assert body["decision"] == "trusted"
        assert body["threshold"] == 2


# --- The policy-missing branch -------------------------------------------------


def test_no_policy_returns_200_policy_missing_without_target_lookup(client):
    _bootstrap_signer(client, "org-1", SEED_A)
    # Even an unknown target id never reaches the resource-missing branch:
    # the absent policy short-circuits first.
    resp = _decide(client, "org-1", SEED_A, "claim", "clm_does_not_exist")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {
        "policy_id",
        "threshold",
        "qualified_signer_count",
        "decision",
        "reason",
    }
    assert body["policy_id"] is None
    assert body["threshold"] is None
    assert body["qualified_signer_count"] == 0
    assert body["decision"] == "untrusted"
    assert body["reason"] == "policy_missing"
    assert resp.content.endswith(b"\n")


def test_no_policy_with_unknown_bundle_is_still_policy_missing(client):
    _bootstrap_signer(client, "org-1", SEED_A)
    resp = _decide(
        client, "org-1", SEED_A, "evidence_bundle", "evb_does_not_exist"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["reason"] == "policy_missing"
    assert body["decision"] == "untrusted"
    assert body["policy_id"] is None


def test_decision_with_policy_and_unknown_targets_is_404(client):
    _bootstrap_signer(client, "org-1", SEED_A)
    _make_policy(client, "org-1", 1, seed=SEED_A)

    missing_claim = _decide(
        client, "org-1", SEED_A, "claim", "clm_does_not_exist"
    )
    assert missing_claim.status_code == 404
    claim_error = missing_claim.json()["error"]
    assert claim_error["code"] == "claim_not_found"
    assert claim_error["details"]["claim_id"] == "clm_does_not_exist"

    missing_bundle = _decide(
        client, "org-1", SEED_A, "evidence_bundle", "evb_does_not_exist"
    )
    assert missing_bundle.status_code == 404
    bundle_error = missing_bundle.json()["error"]
    assert bundle_error["code"] == "evidence_bundle_not_found"
    assert bundle_error["details"]["evidence_bundle_id"] == (
        "evb_does_not_exist"
    )


def test_declaring_a_bundle_id_as_a_claim_and_vice_versa_is_404(client):
    _bootstrap_signer(client, "org-1", SEED_A)
    _make_policy(client, "org-1", 1, seed=SEED_A)
    claim = _make_claim(client, "org-1", digest=DIGEST_B)
    bundle = _make_bundle(client, claim["id"])

    cross_one = _decide(
        client, "org-1", SEED_A, "claim", bundle["id"]
    )
    assert cross_one.status_code == 404
    assert cross_one.json()["error"]["code"] == "claim_not_found"
    cross_two = _decide(
        client, "org-1", SEED_A, "evidence_bundle", claim["id"]
    )
    assert cross_two.status_code == 404
    assert cross_two.json()["error"]["code"] == "evidence_bundle_not_found"


# --- Credential boundaries -----------------------------------------------------


def test_decision_without_credentials_is_opaque_404(client, db_session):
    _bootstrap_signer(client, "org-1", SEED_A)
    _make_policy(client, "org-1", 1, seed=SEED_A)
    target = _make_claim(client, "org-1", digest=DIGEST_B)
    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())
    resp = client.get(
        f"{DECISIONS_PATH}?actor_id=org-1&target_type=claim"
        f"&target_id={target['id']}"
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"
    assert len(db_session.execute(select(AuditEvent)).scalars().all()) == (
        events_before
    )


def test_decision_unverifiable_signature_is_opaque_404(client):
    _bootstrap_signer(client, "org-1", SEED_A)
    _make_policy(client, "org-1", 1, seed=SEED_A)
    target = _make_claim(client, "org-1", digest=DIGEST_B)
    # SEED_B is not a current key of org-1.
    resp = _decide(client, "org-1", SEED_B, "claim", target["id"])
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_decision_malformed_credentials_are_422(client):
    _bootstrap_signer(client, "org-1", SEED_A)
    _make_policy(client, "org-1", 1, seed=SEED_A)
    target = _make_claim(client, "org-1", digest=DIGEST_B)
    query = f"actor_id=org-1&target_type=claim&target_id={target['id']}"

    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    stale_resp = _signed_get(
        client, "org-1", SEED_A, query=query, timestamp=stale
    )
    assert stale_resp.status_code == 422
    assert stale_resp.json()["error"]["code"] == "validation_error"

    not_ts = _signed_get(
        client, "org-1", SEED_A, query=query, timestamp="not-a-timestamp"
    )
    assert not_ts.status_code == 422

    # A signature header that is not canonical Base64 of exactly 64 bytes.
    headers = {
        "X-PA": "org-1",
        "X-PT": _now(),
        "X-PS": base64.b64encode(b"x" * 63).decode(),
    }
    resp = client.get(f"{DECISIONS_PATH}?{query}", headers=headers)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_decision_for_another_subject_is_404_not_422(client):
    _bootstrap_signer(client, "org-1", SEED_A)
    _bootstrap_signer(client, "org-2", SEED_B, digest=DIGEST_B)
    _make_policy(client, "org-1", 1, seed=SEED_A)
    target = _make_claim(client, "org-1", digest=DIGEST_C)
    # org-2 authenticates validly but asks for org-1's decision.
    query = f"actor_id=org-1&target_type=claim&target_id={target['id']}"
    resp = _signed_get(client, "org-2", SEED_B, query=query)
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "not_found"


# --- Parameter validation ------------------------------------------------------


def test_decision_parameter_validation_failures_are_422(client):
    _bootstrap_signer(client, "org-1", SEED_A)
    target = _make_claim(client, "org-1", digest=DIGEST_B)
    base = target["id"]

    cases = (
        f"target_type=claim&target_id={base}",
        "actor_id=org-1&target_id=" + base,
        "actor_id=org-1&target_type=claim",
        f"actor_id=org-1&target_type=&target_id={base}",
        f"actor_id=org-1&target_type=bundle&target_id={base}",
        f"actor_id=org-1&target_type=Claim&target_id={base}",
        f"actor_id=&target_type=claim&target_id={base}",
        f"actor_id=org-1&target_type=claim&target_id={base}&extra=1",
        f"actor_id=org-1&actor_id=org-2&target_type=claim&target_id={base}",
        f"actor_id=org-1&target_type=claim&target_type=evidence_bundle"
        f"&target_id={base}",
    )
    ts = _now()
    headers = {
        "X-PA": "org-1",
        "X-PT": ts,
        "X-PS": base64.b64encode(
            ed25519_sign(
                SEED_A,
                access_message_bytes(
                    "GET", DECISIONS_PATH, ts,
                    hashlib.sha256(b"").hexdigest(),
                ),
            )
        ).decode("ascii"),
    }
    for query in cases:
        resp = client.get(f"{DECISIONS_PATH}?{query}", headers=headers)
        assert resp.status_code == 422, query
        assert resp.json()["error"]["code"] == "validation_error"


def test_decision_signature_path_excludes_query_string(client):
    _bootstrap_signer(client, "org-1", SEED_A)
    _make_policy(client, "org-1", 1, seed=SEED_A)
    target = _make_claim(client, "org-1", digest=DIGEST_B)
    # Signing a path that includes the query string must fail
    # verification: the contract signs the bare path.
    query = f"actor_id=org-1&target_type=claim&target_id={target['id']}"
    signed_path = f"{DECISIONS_PATH}?{query}"
    resp = _signed_get(
        client, "org-1", SEED_A, query=query, signed_path=signed_path
    )
    assert resp.status_code == 404


# --- Read-only -----------------------------------------------------------------


def test_decisions_never_write_resources_or_audit(client, db_session):
    _bootstrap_signer(client, "org-1", SEED_A)
    _make_policy(client, "org-1", 2, seed=SEED_A)
    target = _make_claim(client, "org-1", digest=DIGEST_B)
    _make_attestation(client, "org-1", SEED_A, "claim", target["id"])

    policies_before = len(
        db_session.execute(select(ActorTrustPolicy)).scalars().all()
    )
    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())
    revocations_before = len(
        db_session.execute(select(AttestationRevocation)).scalars().all()
    )

    # Success (untrusted), the target-exists path...
    resp = _decide(client, "org-1", SEED_A, "claim", target["id"])
    assert resp.status_code == 200 and resp.json()["decision"] == "untrusted"
    # ...a failed request (unknown target)...
    failed = _decide(
        client, "org-1", SEED_A, "claim", "clm_does_not_exist"
    )
    assert failed.status_code == 404
    # ...and an unauthenticated request (no credentials at all).
    unauth = client.get(
        f"{DECISIONS_PATH}?actor_id=org-9&target_type=claim"
        f"&target_id={target['id']}"
    )
    assert unauth.status_code == 404

    assert len(
        db_session.execute(select(ActorTrustPolicy)).scalars().all()
    ) == policies_before
    assert len(
        db_session.execute(select(AuditEvent)).scalars().all()
    ) == events_before
    assert len(
        db_session.execute(select(AttestationRevocation)).scalars().all()
    ) == revocations_before


def test_decision_with_existing_target_and_no_attestations_is_untrusted_zero(
    client,
):
    _bootstrap_signer(client, "org-1", SEED_A)
    _make_policy(client, "org-1", 1, seed=SEED_A)
    target = _make_claim(client, "org-1", digest=DIGEST_B)
    resp = _decide(client, "org-1", SEED_A, "claim", target["id"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["qualified_signer_count"] == 0
    assert body["decision"] == "untrusted"
    # A normal (policy-present) decision carries no reason field.
    assert "reason" not in body
