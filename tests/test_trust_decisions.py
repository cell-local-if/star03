"""Tests for subject trust decisions.

Covers ``GET /v1/trust-decisions``:

* the shared ``X-PA``/``X-PT``/``X-PS`` protected-request contract over the
  empty GET body, with the signed path carrying no query string;
* no policy -> 200 ``untrusted`` / ``policy_missing`` / null policy id /
  zero threshold and signer count, with the target never inspected (the
  branch precedes target lookup, so a missing target still returns it);
* a policy followed by target lookup: unknown claim -> 404
  ``claim_not_found``; unknown evidence bundle -> 404
  ``evidence_bundle_not_found`` (no partial result);
* the live decision: threshold, distinct qualified-signer counting
  (verified, non-revoked proofs, deduplicated by signing subject), the
  trusted/untrusted boundary, revocation recomputation, and recomputation
  across a restart;
* 422 validation for missing/blank/unknown/repeated parameters and illegal
  target types, validated before credentials and before target lookup;
* the opaque 404 for missing/unverifiable credentials and for a valid
  signature querying another subject (never 422), while malformed timestamp
  or signature encodings stay 422;
* compact UTF-8 JSON terminated by one newline and the no-write guarantee
  for every success, failure, and empty-policy request.

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures).
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone

from sqlalchemy import func, select

from provenance.access_signing import access_message_bytes
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import ActorTrustPolicy, AuditEvent
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

DECISIONS_PATH = "/v1/trust-decisions"
EVIDENCE_DIGEST = hashlib.sha256(b"evidence-decision").hexdigest()


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


def _attest(client, target_type, target_id, *, actor_id, seed):
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


def _two_subjects(client):
    create_actor(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    claim = _make_claim(client, "org-1")
    # Each subject holds a usable non-revoked attestation key: org-1 attests
    # the claim under evaluation; org-2 bootstraps on its own claim so its
    # SEED_B signature can authenticate (it later also attests claim).
    _attest(client, "claim", claim["id"], actor_id="org-1", seed=SEED_A)
    org2_claim = _make_claim(client, "org-2", digest=DIGEST_B)
    _attest(client, "claim", org2_claim["id"], actor_id="org-2", seed=SEED_B)
    return claim


def _signed_headers(query_string, *, actor, seed, timestamp=None,
                    headers_extra=None):
    ts = timestamp or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    # The signed path never carries the query string; the GET body is empty.
    message = access_message_bytes(
        "GET",
        DECISIONS_PATH,
        ts,
        hashlib.sha256(b"").hexdigest(),
    )
    signature = base64.b64encode(ed25519_sign(seed, message)).decode("ascii")
    return {
        "X-PA": actor,
        "X-PT": ts,
        "X-PS": signature,
        **(headers_extra or {}),
    }


def _decide(client, actor_id, target_type, target_id, *, actor=None,
            seed=SEED_A, timestamp=None, headers_extra=None,
            extra_query="", include_credentials=True):
    query = (
        f"actor_id={actor_id}&target_type={target_type}"
        f"&target_id={target_id}{extra_query}"
    )
    url = f"{DECISIONS_PATH}?{query}"
    if not include_credentials:
        return client.get(url)
    headers = _signed_headers(
        query, actor=actor or actor_id, seed=seed, timestamp=timestamp,
        headers_extra=headers_extra,
    )
    return client.get(url, headers=headers)


def _register_policy(client, actor_id, threshold, *, seed):
    """Register a policy through the public signed endpoint."""
    path = "/v1/trust-policies"
    body = json.dumps({"actor_id": actor_id, "threshold": threshold}).encode()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    message = access_message_bytes(
        "POST", path, ts, hashlib.sha256(body).hexdigest()
    )
    headers = {
        "Content-Type": "application/json",
        "X-PA": actor_id,
        "X-PT": ts,
        "X-PS": base64.b64encode(ed25519_sign(seed, message)).decode(),
    }
    resp = client.post(path, content=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


# --- Policy-missing branch --------------------------------------------------


def test_no_policy_returns_200_untrusted_policy_missing(client):
    claim = _two_subjects(client)
    resp = _decide(client, "org-1", "claim", claim["id"])
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "policy_id": None,
        "actor_id": "org-1",
        "target_type": "claim",
        "target_id": claim["id"],
        "threshold": 0,
        "qualified_signer_count": 0,
        "decision": "untrusted",
        "reason": "policy_missing",
    }


def test_policy_missing_never_inspects_the_target_even_when_absent(client):
    _two_subjects(client)
    # The policy-missing branch precedes the target lookup: an unknown claim
    # or bundle id still returns the 200 policy_missing body, not a 404.
    for target_type, ghost in (
        ("claim", "clm_ghost"),
        ("evidence_bundle", "evb_ghost"),
    ):
        resp = _decide(client, "org-1", target_type, ghost)
        assert resp.status_code == 200, target_type
        body = resp.json()
        assert body["decision"] == "untrusted"
        assert body["reason"] == "policy_missing"
        assert body["policy_id"] is None
        assert body["qualified_signer_count"] == 0


def test_policy_missing_body_is_compact_json_with_one_newline(client):
    _two_subjects(client)
    resp = _decide(client, "org-1", "claim", "clm_anything")
    raw = resp.content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    # Compact separators: no whitespace after the ':' or ',' delimiters.
    assert b'": ' not in raw
    assert b", " not in raw
    # Round-trips and carries the expected reason marker.
    assert json.loads(raw)["reason"] == "policy_missing"


# --- Target lookup after a policy exists ------------------------------------


def test_unknown_claim_after_policy_is_404(client):
    _two_subjects(client)
    _register_policy(client, "org-1", 1, seed=SEED_A)
    resp = _decide(client, "org-1", "claim", "clm_ghost")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "claim_not_found"
    assert error["details"]["claim_id"] == "clm_ghost"


def test_unknown_evidence_bundle_after_policy_is_404(client):
    _two_subjects(client)
    _register_policy(client, "org-1", 1, seed=SEED_A)
    resp = _decide(client, "org-1", "evidence_bundle", "evb_ghost")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "evidence_bundle_not_found"
    assert error["details"]["evidence_bundle_id"] == "evb_ghost"


def test_wrong_target_type_for_existing_resource_is_404(client):
    claim = _two_subjects(client)
    bundle = _make_bundle(client, claim["id"])
    _register_policy(client, "org-1", 1, seed=SEED_A)
    r1 = _decide(client, "org-1", "claim", bundle["id"])
    r2 = _decide(client, "org-1", "evidence_bundle", claim["id"])
    assert r1.status_code == 404
    assert r1.json()["error"]["code"] == "claim_not_found"
    assert r2.status_code == 404
    assert r2.json()["error"]["code"] == "evidence_bundle_not_found"


# --- Decision semantics ------------------------------------------------------


def test_trusted_when_qualified_signers_reach_threshold(client):
    claim = _two_subjects(client)
    _register_policy(client, "org-1", 2, seed=SEED_A)
    # Only org-1's attestation exists so far.
    one = _decide(client, "org-1", "claim", claim["id"])
    assert one.json()["qualified_signer_count"] == 1
    assert one.json()["decision"] == "untrusted"
    assert "reason" not in one.json()
    assert one.json()["threshold"] == 2

    # A second distinct signing subject attests the same target.
    _attest(client, "claim", claim["id"], actor_id="org-2", seed=SEED_B)
    two = _decide(client, "org-1", "claim", claim["id"])
    assert two.status_code == 200
    assert two.json()["qualified_signer_count"] == 2
    assert two.json()["decision"] == "trusted"
    assert "reason" not in two.json()


def test_same_signer_with_multiple_keys_counts_once(client):
    claim = _two_subjects(client)
    _register_policy(client, "org-1", 2, seed=SEED_A)
    # org-1 attests again under a *different* key (SEED_B, which it does not
    # use to authenticate) -- same signing subject, so still one signer.
    _attest(client, "claim", claim["id"], actor_id="org-1", seed=SEED_B)
    resp = _decide(client, "org-1", "claim", claim["id"])
    assert resp.json()["qualified_signer_count"] == 1
    assert resp.json()["decision"] == "untrusted"


def test_revocation_drops_the_signer_and_recomputes(client):
    claim = _two_subjects(client)
    _attest(client, "claim", claim["id"], actor_id="org-2", seed=SEED_B)
    _register_policy(client, "org-1", 2, seed=SEED_A)
    assert _decide(client, "org-1", "claim", claim["id"]).json()[
        "qualified_signer_count"
    ] == 2

    # Revoke org-2's attestation: its subject drops out of the count.
    atts = client.get(
        "/v1/attestations",
        params={"target_type": "claim", "target_id": claim["id"]},
    ).json()["items"]
    org2_att = next(a for a in atts if a["signer_actor_id"] == "org-2")
    rev = client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": org2_att["id"],
            "revoker_actor_id": "org-1",
            "reason": "key compromise",
        },
    )
    assert rev.status_code == 201, rev.text

    after = _decide(client, "org-1", "claim", claim["id"])
    assert after.json()["qualified_signer_count"] == 1
    assert after.json()["decision"] == "untrusted"


def test_evidence_bundle_target_is_evaluated(client):
    claim = _two_subjects(client)
    bundle = _make_bundle(client, claim["id"])
    _attest(client, "evidence_bundle", bundle["id"], actor_id="org-1",
            seed=SEED_A)
    _register_policy(client, "org-1", 1, seed=SEED_A)
    resp = _decide(client, "org-1", "evidence_bundle", bundle["id"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["qualified_signer_count"] == 1
    assert body["decision"] == "trusted"
    assert body["threshold"] == 1


def test_decision_recomputes_live_across_restart(tmp_db_url):
    from fastapi.testclient import TestClient

    app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app) as client:
        claim = _two_subjects(client)
        _register_policy(client, "org-1", 2, seed=SEED_A)
        assert _decide(client, "org-1", "claim", claim["id"]).json()[
            "decision"
        ] == "untrusted"

    # Restart: the policy and the first attestation persist; the decision is
    # recomputed from current state, then a revocation flips it back.
    app_two = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app_two) as client:
        _attest(client, "claim", claim["id"], actor_id="org-2", seed=SEED_B)
        trusted = _decide(client, "org-1", "claim", claim["id"])
        assert trusted.json()["decision"] == "trusted"

        atts = client.get(
            "/v1/attestations",
            params={"target_type": "claim", "target_id": claim["id"]},
        ).json()["items"]
        org2_att = next(a for a in atts if a["signer_actor_id"] == "org-2")
        client.post(
            "/v1/attestation-revocations",
            json={
                "attestation_id": org2_att["id"],
                "revoker_actor_id": "org-1",
                "reason": "rotate",
            },
        )
        recomputed = _decide(client, "org-1", "claim", claim["id"])
        assert recomputed.json()["decision"] == "untrusted"
        assert recomputed.json()["qualified_signer_count"] == 1


# --- Validation boundary ----------------------------------------------------


def test_missing_required_parameters_are_422(client):
    _two_subjects(client)
    for url in (
        DECISIONS_PATH,
        f"{DECISIONS_PATH}?target_type=claim&target_id=clm_x",
        f"{DECISIONS_PATH}?actor_id=org-1&target_id=clm_x",
        f"{DECISIONS_PATH}?actor_id=org-1&target_type=claim",
    ):
        resp = client.get(url)
        assert resp.status_code == 422, url
        assert resp.json()["error"]["code"] == "validation_error"


def test_blank_or_unknown_target_type_and_blank_ids_are_422(client):
    _two_subjects(client)
    bad_urls = [
        f"{DECISIONS_PATH}?actor_id=org-1&target_type=&target_id=clm_x",
        f"{DECISIONS_PATH}?actor_id=org-1&target_type=Claim&target_id=clm_x",
        f"{DECISIONS_PATH}?actor_id=org-1&target_type=attestation&target_id=x",
        f"{DECISIONS_PATH}?actor_id=&target_type=claim&target_id=clm_x",
        f"{DECISIONS_PATH}?actor_id=org-1&target_type=claim&target_id=",
        f"{DECISIONS_PATH}?actor_id=%20%20&target_type=claim&target_id=x",
    ]
    for url in bad_urls:
        resp = client.get(url)
        assert resp.status_code == 422, url
        assert resp.json()["error"]["code"] == "validation_error"


def test_unknown_and_repeated_parameters_are_422(client):
    claim = _two_subjects(client)
    base = (
        f"{DECISIONS_PATH}?actor_id=org-1&target_type=claim"
        f"&target_id={claim['id']}"
    )
    for url in (
        f"{base}&threshold=1",
        f"{base}&foo=bar",
        f"{base}&actor_id=org-2",
        f"{base}&target_type=evidence_bundle",
        f"{base}&target_id={claim['id']}",
    ):
        resp = client.get(url)
        assert resp.status_code == 422, url


def test_validation_precedes_credentials_and_target_lookup(client):
    _two_subjects(client)
    # No credentials at all, but the query is malformed: 422, not 404.
    resp = client.get(
        f"{DECISIONS_PATH}?actor_id=org-1&target_type=bogus&target_id=clm_x"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    # A structurally valid query for an absent policy and absent target is
    # 200 policy_missing -- but without credentials it is the opaque 404.
    resp = client.get(
        f"{DECISIONS_PATH}?actor_id=org-1&target_type=claim&target_id=clm_ghost"
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


# --- Authentication boundary ------------------------------------------------


def test_missing_or_unverifiable_credentials_are_opaque_404(client):
    claim = _two_subjects(client)
    _register_policy(client, "org-1", 1, seed=SEED_A)
    url = (
        f"{DECISIONS_PATH}?actor_id=org-1&target_type=claim"
        f"&target_id={claim['id']}"
    )
    # No headers.
    assert client.get(url).status_code == 404
    # A well-formed signature under org-2's key over org-1's query is an
    # unauthorized request: the opaque 404.
    resp = _decide(client, "org-1", "claim", claim["id"], actor="org-2",
                   seed=SEED_B)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_valid_signature_querying_another_subject_is_404_not_422(client):
    # Even when neither a policy nor the target exists, an authenticated
    # caller naming a different subject is the opaque 404 (authorization),
    # never a 422.
    claim = _two_subjects(client)
    resp = _decide(client, "org-1", "claim", claim["id"], actor="org-2",
                   seed=SEED_B)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"

    # ...and the same holds against a missing target (no 404 claim branch is
    # leaked to an unauthorized caller).
    resp = _decide(client, "org-1", "claim", "clm_ghost", actor="org-2",
                   seed=SEED_B)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_unparseable_or_old_timestamp_is_422(client):
    _two_subjects(client)
    for bad_ts in ("not-a-timestamp", "2000-01-01T00:00:00Z"):
        resp = _decide(
            client, "org-1", "claim", "clm_x", timestamp=bad_ts
        )
        assert resp.status_code == 422, bad_ts
        assert resp.json()["error"]["code"] == "validation_error"


def test_malformed_signature_encoding_is_422(client):
    _two_subjects(client)
    for value in ("not-base64!!!", "AAAA", base64.b64encode(b"x" * 32).decode()):
        resp = _decide(
            client, "org-1", "claim", "clm_x", headers_extra={"X-PS": value}
        )
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only guarantee -----------------------------------------------------


def test_decisions_write_no_resources_or_audit_events(client, db_session):
    claim = _two_subjects(client)
    _register_policy(client, "org-1", 2, seed=SEED_A)
    _attest(client, "claim", claim["id"], actor_id="org-2", seed=SEED_B)

    def counts():
        return (
            db_session.scalar(
                select(func.count()).select_from(ActorTrustPolicy)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    policies_before, audit_before = counts()

    # Successful, untrusted, policy-missing, and missing-target decisions.
    assert _decide(client, "org-1", "claim", claim["id"]).status_code == 200
    # org-2 has a usable key but no policy -> 200 policy_missing (target not
    # inspected), authenticated as org-2 itself.
    assert _decide(client, "org-2", "claim", "clm_ghost", actor="org-2",
                   seed=SEED_B).status_code == 200
    assert _decide(client, "org-1", "claim", "clm_ghost").status_code == 404
    # A malformed request and an unauthenticated one also write nothing.
    assert client.get(
        f"{DECISIONS_PATH}?actor_id=org-1&target_type=x&target_id=y"
    ).status_code == 422
    assert client.get(
        f"{DECISIONS_PATH}?actor_id=org-1&target_type=claim"
        f"&target_id={claim['id']}"
    ).status_code == 404

    policies_after, audit_after = counts()
    assert policies_after == policies_before
    assert audit_after == audit_before
