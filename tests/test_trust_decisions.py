"""Tests for subject-level protected authorization decisions.

Covers protected ``GET /v1/trust-decisions``:

* the decision uses only the calling subject's current policy and returns
  target type/id, policy id, threshold, and the qualified signer count;
* no policy -> 200 ``untrusted``/``policy_missing`` with null policy id and
  threshold, and the target is never looked up (so an unknown target does
  not leak as a 404);
* with a policy, an unknown claim is ``404 claim_not_found`` and an unknown
  evidence bundle is ``404 evidence_bundle_not_found``; claim ids and
  bundle ids never cross-match;
* ``trusted``/``threshold_met`` at or above the threshold and
  ``untrusted``/``below_threshold`` below it, keeping the requested target
  type and id in the response;
* qualified signers count only verified, non-revoked attestations of the
  exact target, deduplicated by signing subject; revocation drops a
  signer;
* each subject's decisions are governed by its own policy;
* the credential boundary: missing/unauthenticated credentials are the
  opaque ``404 not_found`` (existence is never revealed), while malformed
  timestamps/signatures and malformed query parameters are 422;
* decisions, the policy-missing answer, missing resources, and failed
  requests are all strictly read-only;
* compact UTF-8 JSON terminated by exactly one newline.

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures).
"""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timezone
from urllib.parse import quote

from sqlalchemy import func, select

from provenance.access_signing import access_message_bytes
from provenance.ids import actor_trust_policy_id
from provenance.models import ActorTrustPolicy, AuditEvent
from provenance.signing import attestation_message_bytes
from tests.helpers import (
    DIGEST_A,
    DIGEST_C,
    content_payload,
    create_actor,
    ed25519_public_key,
    ed25519_sign,
    SEED_A,
    SEED_B,
)

DECISIONS_PATH = "/v1/trust-decisions"
SEED_C = b"test-ed25519-seed-c-00000000000000"[:32]
EVIDENCE_DIGEST = hashlib.sha256(b"evidence-decision").hexdigest()


# --- World setup ---------------------------------------------------------------


def _make_content(client, actor_id="org-1", digest=DIGEST_A):
    resp = client.post(
        "/v1/contents",
        json=content_payload(actor_id=actor_id, digest=digest),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make_claim(client, content_id, actor_id="org-1", claim_type="authorship"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": claim_type,
            "payload": {"statement": "decide on me"},
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


def _attest(client, target_type, target_id, *, actor, seed):
    signature = ed25519_sign(
        seed, attestation_message_bytes(target_type, target_id, actor)
    )
    resp = client.post(
        "/v1/attestations",
        json={
            "target_type": target_type,
            "target_id": target_id,
            "signer_actor_id": actor,
            "public_key": base64.b64encode(ed25519_public_key(seed)).decode(),
            "signature": base64.b64encode(signature).decode(),
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _revoke(client, attestation_id, *, revoker="org-1", reason="no longer relied on"):
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


def _world(client, actors=("org-1", "org-2", "org-3")):
    """Create actors and give each a current key by attesting their own claim."""
    seeds = {"org-1": SEED_A, "org-2": SEED_B, "org-3": SEED_C}
    digests = {
        "org-1": DIGEST_A,
        "org-2": hashlib.sha256(b"content-b").hexdigest(),
        "org-3": hashlib.sha256(b"content-c2").hexdigest(),
    }
    for actor in actors:
        create_actor(
            client,
            **(
                {}
                if actor == "org-1"
                else {"actor_id": actor, "name": actor, "type": "organization"}
            ),
        )
        claim = _make_claim(
            client, _make_content(client, actor, digests[actor])["id"], actor
        )
        _attest(client, "claim", claim["id"], actor=actor, seed=seeds[actor])
    return seeds


def _target_claim(client, actor="org-1", digest=DIGEST_C):
    return _make_claim(client, _make_content(client, actor, digest)["id"], actor)


# --- Signed-request helpers ---------------------------------------------------


def _signed_headers(
    path,
    *,
    query="",
    actor="org-1",
    seed=SEED_A,
    timestamp=None,
    signed_timestamp=None,
    signed_path=None,
):
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    signed_ts = signed_timestamp if signed_timestamp is not None else ts
    # A GET carries no body; the signed body_sha256 is the digest of zero
    # bytes. The signature binds the path WITHOUT the query string.
    message = access_message_bytes(
        "GET",
        signed_path if signed_path is not None else path,
        signed_ts,
        hashlib.sha256(b"").hexdigest(),
    )
    signature = base64.b64encode(ed25519_sign(seed, message)).decode("ascii")
    return {"X-PA": actor, "X-PT": ts, "X-PS": signature}


def _decide(
    client,
    target_type,
    target_id,
    *,
    actor="org-1",
    seed=SEED_A,
    path=DECISIONS_PATH,
    query_string=None,
    **sign_kwargs,
):
    if query_string is None:
        query_string = (
            "target_type="
            + quote(target_type, safe="")
            + "&target_id="
            + quote(target_id, safe="")
        )
    url = path + ("?" + query_string if query_string else "")
    headers = _signed_headers(
        path, query=query_string, actor=actor, seed=seed, **sign_kwargs
    )
    return client.get(url, headers=headers)


def _create_policy(client, subject, threshold, seed):
    body = (
        b'{"subject_id":"'
        + subject.encode()
        + b'","threshold":'
        + str(threshold).encode()
        + b"}"
    )
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    message = access_message_bytes(
        "POST",
        "/v1/trust-policies",
        ts,
        hashlib.sha256(body).hexdigest(),
    )
    headers = {
        "Content-Type": "application/json",
        "X-PA": subject,
        "X-PT": ts,
        "X-PS": base64.b64encode(ed25519_sign(seed, message)).decode("ascii"),
    }
    resp = client.post("/v1/trust-policies", content=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


# --- No policy: policy_missing, target never looked up -------------------------


def test_no_policy_returns_untrusted_policy_missing_without_target_lookup(client):
    seeds = _world(client)
    claim = _target_claim(client)

    # An existing target and an unknown target answer identically: the
    # absence of a policy never reveals whether the target exists.
    for target_id in (claim["id"], "clm_ghost"):
        resp = _decide(client, "claim", target_id)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "target_type": "claim",
            "target_id": target_id,
            "policy_id": None,
            "threshold": None,
            "qualified_signer_count": 0,
            "decision": "untrusted",
            "reason": "policy_missing",
        }

    # Same for evidence bundles, including an unknown bundle id.
    bundle = _make_bundle(client, claim["id"])
    for target_id in (bundle["id"], "evb_ghost"):
        resp = _decide(client, "evidence_bundle", target_id)
        assert resp.status_code == 200
        body = resp.json()
        assert body["reason"] == "policy_missing"
        assert body["policy_id"] is None
        assert body["threshold"] is None
        assert body["decision"] == "untrusted"
        assert body["target_id"] == target_id


def test_policy_missing_body_is_compact_json_with_one_newline(client):
    _world(client)
    claim = _target_claim(client)
    resp = _decide(client, "claim", claim["id"])
    raw = resp.content
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b": " not in raw and b", " not in raw
    raw.decode("utf-8")
    assert b'"policy_id":null' in raw


# --- Target existence after a policy exists ------------------------------------


def test_unknown_claim_with_a_policy_is_404(client):
    seeds = _world(client)
    _create_policy(client, "org-1", 1, seeds["org-1"])
    resp = _decide(client, "claim", "clm_ghost")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "claim_not_found"
    assert error["details"]["claim_id"] == "clm_ghost"


def test_unknown_bundle_with_a_policy_is_404(client):
    seeds = _world(client)
    _create_policy(client, "org-1", 1, seeds["org-1"])
    resp = _decide(client, "evidence_bundle", "evb_ghost")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "evidence_bundle_not_found"
    assert error["details"]["evidence_bundle_id"] == "evb_ghost"


def test_claim_and_bundle_ids_never_cross_match(client):
    seeds = _world(client)
    _create_policy(client, "org-1", 1, seeds["org-1"])
    claim = _target_claim(client)
    bundle = _make_bundle(client, claim["id"])

    as_claim = _decide(client, "claim", bundle["id"])
    assert as_claim.status_code == 404
    assert as_claim.json()["error"]["code"] == "claim_not_found"

    as_bundle = _decide(client, "evidence_bundle", claim["id"])
    assert as_bundle.status_code == 404
    assert as_bundle.json()["error"]["code"] == "evidence_bundle_not_found"


def test_missing_resource_404_produces_no_partial_decision(client):
    seeds = _world(client)
    _create_policy(client, "org-1", 1, seeds["org-1"])
    resp = _decide(client, "claim", "clm_ghost")
    # The error envelope carries no decision fields.
    assert set(resp.json()) == {"error"}


# --- Decisions with a policy ----------------------------------------------------


def test_below_threshold_is_untrusted_and_keeps_the_target(client):
    seeds = _world(client)
    _create_policy(client, "org-1", 2, seeds["org-1"])
    claim = _target_claim(client)
    _attest(client, "claim", claim["id"], actor="org-2", seed=SEED_B)

    resp = _decide(client, "claim", claim["id"])
    assert resp.status_code == 200
    assert resp.json() == {
        "target_type": "claim",
        "target_id": claim["id"],
        "policy_id": actor_trust_policy_id("org-1"),
        "threshold": 2,
        "qualified_signer_count": 1,
        "decision": "untrusted",
        "reason": "below_threshold",
    }


def test_threshold_boundary_equality_is_trusted(client):
    seeds = _world(client)
    _create_policy(client, "org-1", 2, seeds["org-1"])
    claim = _target_claim(client)
    _attest(client, "claim", claim["id"], actor="org-1", seed=SEED_A)
    _attest(client, "claim", claim["id"], actor="org-2", seed=SEED_B)

    resp = _decide(client, "claim", claim["id"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["qualified_signer_count"] == 2
    assert body["decision"] == "trusted"
    assert body["reason"] == "threshold_met"


def test_zero_qualified_signers_is_below_threshold(client):
    seeds = _world(client)
    _create_policy(client, "org-1", 1, seeds["org-1"])
    claim = _target_claim(client)
    resp = _decide(client, "claim", claim["id"])
    assert resp.json()["qualified_signer_count"] == 0
    assert resp.json()["decision"] == "untrusted"
    assert resp.json()["reason"] == "below_threshold"


def test_same_signer_under_distinct_keys_counts_once(client):
    seeds = _world(client)
    _create_policy(client, "org-1", 1, seeds["org-1"])
    claim = _target_claim(client)
    _attest(client, "claim", claim["id"], actor="org-2", seed=SEED_B)
    # org-2 attests the same target again under a second, independent key.
    _attest(client, "claim", claim["id"], actor="org-2", seed=SEED_C)
    resp = _decide(client, "claim", claim["id"])
    assert resp.json()["qualified_signer_count"] == 1
    assert resp.json()["decision"] == "trusted"


def test_revoked_attestation_no_longer_qualifies(client):
    seeds = _world(client)
    _create_policy(client, "org-1", 2, seeds["org-1"])
    claim = _target_claim(client)
    attested = _attest(
        client, "claim", claim["id"], actor="org-2", seed=SEED_B
    )

    before = _decide(client, "claim", claim["id"])
    assert before.json()["qualified_signer_count"] == 1
    assert before.json()["reason"] == "below_threshold"

    _revoke(client, attested["id"])
    after = _decide(client, "claim", claim["id"])
    assert after.json()["qualified_signer_count"] == 0
    assert after.json()["decision"] == "untrusted"
    assert after.json()["reason"] == "below_threshold"


def test_only_attestations_of_the_exact_target_count(client):
    seeds = _world(client)
    _create_policy(client, "org-1", 1, seeds["org-1"])
    claim_one = _target_claim(client, digest=DIGEST_C)
    claim_two = _target_claim(
        client, digest=hashlib.sha256(b"content-other").hexdigest()
    )
    bundle = _make_bundle(client, claim_one["id"])
    _attest(client, "claim", claim_two["id"], actor="org-2", seed=SEED_B)
    _attest(client, "evidence_bundle", bundle["id"], actor="org-3", seed=SEED_C)

    claim_decision = _decide(client, "claim", claim_one["id"])
    assert claim_decision.json()["qualified_signer_count"] == 0
    assert claim_decision.json()["reason"] == "below_threshold"

    bundle_decision = _decide(client, "evidence_bundle", bundle["id"])
    assert bundle_decision.json()["qualified_signer_count"] == 1
    assert bundle_decision.json()["reason"] == "threshold_met"


def test_decisions_use_only_the_calling_subjects_policy(client):
    seeds = _world(client)
    # org-1 requires two signers; org-2 requires one.
    _create_policy(client, "org-1", 2, seeds["org-1"])
    _create_policy(client, "org-2", 1, seeds["org-2"])
    claim = _target_claim(client)
    _attest(client, "claim", claim["id"], actor="org-2", seed=SEED_B)

    for_org1 = _decide(client, "claim", claim["id"], actor="org-1", seed=SEED_A)
    assert for_org1.json()["threshold"] == 2
    assert for_org1.json()["reason"] == "below_threshold"

    for_org2 = _decide(client, "claim", claim["id"], actor="org-2", seed=SEED_B)
    assert for_org2.json()["threshold"] == 1
    assert for_org2.json()["policy_id"] == actor_trust_policy_id("org-2")
    assert for_org2.json()["reason"] == "threshold_met"


def test_decision_is_computed_live(client):
    seeds = _world(client)
    _create_policy(client, "org-1", 1, seeds["org-1"])
    claim = _target_claim(client)
    assert _decide(client, "claim", claim["id"]).json()["reason"] == (
        "below_threshold"
    )
    _attest(client, "claim", claim["id"], actor="org-2", seed=SEED_B)
    assert _decide(client, "claim", claim["id"]).json()["reason"] == (
        "threshold_met"
    )


def test_policy_registered_after_missing_answer_then_governs(client):
    seeds = _world(client)
    claim = _target_claim(client)
    first = _decide(client, "claim", claim["id"])
    assert first.json()["reason"] == "policy_missing"
    _create_policy(client, "org-1", 1, seeds["org-1"])
    second = _decide(client, "claim", claim["id"])
    assert second.json()["reason"] == "below_threshold"
    assert second.json()["policy_id"] == actor_trust_policy_id("org-1")


# --- Credential boundary ---------------------------------------------------------


def test_missing_credentials_are_an_opaque_404(client):
    _world(client)
    claim = _target_claim(client)
    url = f"{DECISIONS_PATH}?target_type=claim&target_id={claim['id']}"
    resp = client.get(url)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_unverifiable_signature_is_an_opaque_404(client):
    _world(client)
    claim = _target_claim(client)
    # org-2's key cannot authenticate a request naming org-1.
    resp = _decide(
        client, "claim", claim["id"], actor="org-1", seed=SEED_B
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_unknown_header_actor_is_an_opaque_404(client):
    _world(client)
    claim = _target_claim(client)
    resp = _decide(
        client, "claim", claim["id"], actor="org-ghost", seed=SEED_A
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_malformed_credentials_are_422(client):
    _world(client)
    claim = _target_claim(client)

    bad_ts = _decide(
        client, "claim", claim["id"], timestamp="not-a-timestamp"
    )
    assert bad_ts.status_code == 422
    assert bad_ts.json()["error"]["code"] == "validation_error"

    old = datetime(2000, 1, 1, tzinfo=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    out_of_window = _decide(client, "claim", claim["id"], timestamp=old)
    assert out_of_window.status_code == 422

    headers = _signed_headers(DECISIONS_PATH, actor="org-1", seed=SEED_A)
    headers["X-PS"] = "not-base64!"
    resp = client.get(
        f"{DECISIONS_PATH}?target_type=claim&target_id={claim['id']}",
        headers=headers,
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_signature_binds_the_path_without_the_query_string(client):
    _world(client)
    claim = _target_claim(client)
    # Signing the path WITH the query string must fail: the contract signs
    # the bare path.
    query = f"target_type=claim&target_id={claim['id']}"
    resp = _decide(
        client,
        "claim",
        claim["id"],
        signed_path=DECISIONS_PATH + "?" + query,
    )
    assert resp.status_code == 404


# --- Query validation -------------------------------------------------------------


def test_missing_target_parameters_are_422(client):
    _world(client)
    for query_string in (
        "",
        "target_id=clm_x",
        "target_type=claim",
        "target_type=&target_id=clm_x",
    ):
        headers = _signed_headers(DECISIONS_PATH, actor="org-1", seed=SEED_A)
        resp = client.get(
            DECISIONS_PATH + ("?" + query_string if query_string else ""),
            headers=headers,
        )
        assert resp.status_code == 422, query_string
        assert resp.json()["error"]["code"] == "validation_error"


def test_blank_or_unknown_target_type_is_422(client):
    _world(client)
    claim = _target_claim(client)
    for value in ("", "   ", "Claim", "claim ", "evidence", "attestation"):
        resp = _decide(client, value, claim["id"])
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_blank_target_id_is_422(client):
    _world(client)
    for value in ("", "   ", "\t"):
        resp = _decide(client, "claim", value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_and_unknown_parameters_are_422(client):
    _world(client)
    claim = _target_claim(client)
    base = f"{DECISIONS_PATH}?target_id={claim['id']}"
    for url in (
        f"{base}&target_type=claim&target_type=evidence_bundle",
        f"{base}&target_type=claim&target_type=claim",
        f"{base}&target_type=claim&foo=bar",
        f"{base}&target_type=claim&min_signers=1",
    ):
        headers = _signed_headers(DECISIONS_PATH, actor="org-1", seed=SEED_A)
        resp = client.get(url, headers=headers)
        assert resp.status_code == 422, url
        assert resp.json()["error"]["code"] == "validation_error"


def test_query_validation_precedes_authentication_and_lookup(client):
    _world(client)
    # No credentials at all: malformed query parameters are still 422, not
    # the opaque 404, and no target lookup happens.
    resp = client.get(
        f"{DECISIONS_PATH}?target_type=nope&target_id=clm_ghost"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only guarantee -----------------------------------------------------------


def test_decisions_write_no_resources_or_audit_events(client, db_session):
    seeds = _world(client)
    _create_policy(client, "org-1", 2, seeds["org-1"])
    claim = _target_claim(client)
    bundle = _make_bundle(client, claim["id"])
    _attest(client, "claim", claim["id"], actor="org-2", seed=SEED_B)

    policies_before = db_session.scalar(
        select(func.count()).select_from(ActorTrustPolicy)
    )
    audit_before = db_session.scalar(
        select(func.count()).select_from(AuditEvent)
    )

    assert _decide(client, "claim", claim["id"]).status_code == 200
    assert _decide(client, "evidence_bundle", bundle["id"]).status_code == 200
    assert _decide(client, "claim", "clm_ghost").status_code == 404
    assert (
        _decide(client, "evidence_bundle", "evb_ghost").status_code == 404
    )
    # An actor without a policy also produces only a read.
    assert (
        _decide(client, "claim", claim["id"], actor="org-3", seed=SEED_C).status_code
        == 200
    )

    assert (
        db_session.scalar(select(func.count()).select_from(ActorTrustPolicy))
        == policies_before
    )
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audit_before
    )
