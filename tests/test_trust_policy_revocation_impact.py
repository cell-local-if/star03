"""Tests for the trust-policy-revocation impact read.

Covers ``GET /v1/trust-policy-revocations/{revocation_id}/impact``: the 200
impact view (compact UTF-8 JSON terminated by exactly one newline, members
in revocation, threshold, before_authorized_claim_count,
before_authorized_evidence_bundle_count, after_authorized_claim_count,
after_authorized_evidence_bundle_count order, and the revocation members in
id, policy_id, actor_id, reason, created_at order), the counterfactual
"before" counts under the current evidence state (verified, non-revoked
attestations of the exact target, deduplicated by signing actor, compared
against the revoked policy's threshold, with every other revocation still
in effect), the always-zero "after" counts, the all-zero empty and
below-threshold states, the verbatim-id 404
``actor_trust_policy_revocation_not_found``, the empty-body and no-query
422 boundary, the 405 on non-GET methods, the strict read-only guarantee,
and restart consistency.

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures).
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from provenance.access_signing import access_message_bytes
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import (
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
SEED_C = b"test-ed25519-seed-c-00000000000000"[:32]
EVIDENCE_DIGEST = hashlib.sha256(b"evidence-impact").hexdigest()

IMPACT_ORDER = [
    "revocation",
    "threshold",
    "before_authorized_claim_count",
    "before_authorized_evidence_bundle_count",
    "after_authorized_claim_count",
    "after_authorized_evidence_bundle_count",
]
REVOCATION_ORDER = ["id", "policy_id", "actor_id", "reason", "created_at"]


# --- Fixture-style setup ------------------------------------------------------


def _impact_path(revocation_id):
    return f"{REVOCATIONS_PATH}/{revocation_id}/impact"


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
    """Three actors; org-1 (the policy subject) holds a current key."""
    create_actor(client)  # org-1
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    create_actor(client, actor_id="org-3", name="Third Org", type="organization")
    # org-1's own attestation gives org-1 a current authentication key.
    _attest(client, "claim", _make_claim(client, "org-1")["id"], "org-1", SEED_A)


# --- Signed-request helpers ---------------------------------------------------


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


def _create_policy(client, actor="org-1", seed=SEED_A, threshold=2):
    body = json.dumps({"actor_id": actor, "threshold": threshold}).encode()
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", POLICIES_PATH, body, actor=actor, seed=seed),
    }
    resp = client.post(POLICIES_PATH, content=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _revoke(client, policy_id, reason="no longer relied upon"):
    body = json.dumps({"policy_id": policy_id, "reason": reason}).encode()
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", REVOCATIONS_PATH, body),
    }
    resp = client.post(REVOCATIONS_PATH, content=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _get_impact(client, revocation_id, **kwargs):
    return client.get(_impact_path(revocation_id), **kwargs)


# --- Impact counts ------------------------------------------------------------


def test_impact_response_shape_and_member_order(client):
    _world(client)
    policy = _create_policy(client, threshold=2)
    revocation = _revoke(client, policy["id"], reason="key material retired")

    resp = _get_impact(client, revocation["id"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert list(body) == IMPACT_ORDER
    assert list(body["revocation"]) == REVOCATION_ORDER
    assert body["revocation"] == revocation
    assert body["threshold"] == 2
    for field in IMPACT_ORDER[2:]:
        assert isinstance(body[field], int)
        assert body[field] >= 0
    assert body["after_authorized_claim_count"] == 0
    assert body["after_authorized_evidence_bundle_count"] == 0


def test_impact_response_is_compact_utf8_json_with_one_newline(client):
    _world(client)
    policy = _create_policy(client)
    revocation = _revoke(client, policy["id"], reason="done")

    resp = _get_impact(client, revocation["id"])
    assert resp.status_code == 200
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    # Members appear in the declared public order.
    assert raw.index(b'"revocation"') < raw.index(b'"threshold"')
    assert raw.index(b'"threshold"') < raw.index(
        b'"before_authorized_claim_count"'
    )
    assert raw.index(b'"before_authorized_claim_count"') < raw.index(
        b'"before_authorized_evidence_bundle_count"'
    )
    assert raw.index(b'"before_authorized_evidence_bundle_count"') < raw.index(
        b'"after_authorized_claim_count"'
    )
    assert raw.index(b'"after_authorized_claim_count"') < raw.index(
        b'"after_authorized_evidence_bundle_count"'
    )
    # The nested revocation view keeps its own declared order.
    assert raw.index(b'"id"') < raw.index(b'"policy_id"')
    assert raw.index(b'"policy_id"') < raw.index(b'"actor_id"')
    assert raw.index(b'"actor_id"') < raw.index(b'"reason"')
    assert raw.index(b'"reason"') < raw.index(b'"created_at"')
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    assert raw == expected


def test_before_counts_follow_threshold_per_target_type(client):
    _world(client)
    policy = _create_policy(client, threshold=2)

    # Two claims and one bundle above the threshold, one of each below.
    met_claim = _make_claim(client, "org-2", digest=DIGEST_B)
    _attest(client, "claim", met_claim["id"], "org-2", SEED_B)
    _attest(client, "claim", met_claim["id"], "org-3", SEED_C)
    below_claim = _make_claim(client, "org-2", digest=DIGEST_C)
    _attest(client, "claim", below_claim["id"], "org-2", SEED_B)
    met_bundle = _make_bundle(client, met_claim["id"])
    _attest(client, "evidence_bundle", met_bundle["id"], "org-2", SEED_B)
    _attest(client, "evidence_bundle", met_bundle["id"], "org-3", SEED_C)
    below_bundle = _make_bundle(client, below_claim["id"])
    _attest(client, "evidence_bundle", below_bundle["id"], "org-3", SEED_C)

    revocation = _revoke(client, policy["id"])
    body = _get_impact(client, revocation["id"]).json()
    # org-1's setup claim from _world also has exactly one signer, so only
    # the two-signer targets reach the threshold of 2.
    assert body["before_authorized_claim_count"] == 1
    assert body["before_authorized_evidence_bundle_count"] == 1
    assert body["after_authorized_claim_count"] == 0
    assert body["after_authorized_evidence_bundle_count"] == 0


def test_repeated_signer_counts_once(client):
    _world(client)
    policy = _create_policy(client, threshold=2)

    # Two attestations from the same signer (under different keys) never
    # reach a threshold of 2.
    claim = _make_claim(client, "org-2", digest=DIGEST_B)
    _attest(client, "claim", claim["id"], "org-2", SEED_B)
    _attest(client, "claim", claim["id"], "org-2", SEED_C)

    revocation = _revoke(client, policy["id"])
    body = _get_impact(client, revocation["id"]).json()
    assert body["before_authorized_claim_count"] == 0
    assert body["before_authorized_evidence_bundle_count"] == 0


def test_attestation_revocations_still_apply_in_before_counts(client):
    _world(client)
    policy = _create_policy(client, threshold=2)

    claim = _make_claim(client, "org-2", digest=DIGEST_B)
    first = _attest(client, "claim", claim["id"], "org-2", SEED_B)
    _attest(client, "claim", claim["id"], "org-3", SEED_C)
    # A pre-existing attestation revocation removes one of the two signers:
    # the counterfactual ignores only the policy revocation itself.
    resp = client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": first["id"],
            "revoker_actor_id": "org-2",
            "reason": "proof withdrawn",
        },
    )
    assert resp.status_code == 201, resp.text

    revocation = _revoke(client, policy["id"])
    body = _get_impact(client, revocation["id"]).json()
    assert body["before_authorized_claim_count"] == 0
    assert body["before_authorized_evidence_bundle_count"] == 0


def test_all_zero_without_claims_or_bundles(client):
    _world(client)
    policy = _create_policy(client, threshold=1)
    revocation = _revoke(client, policy["id"])

    body = _get_impact(client, revocation["id"]).json()
    # Only org-1's setup claim exists, with one signer: threshold 1 counts
    # exactly that one claim and no bundle.
    assert body["before_authorized_claim_count"] == 1
    assert body["before_authorized_evidence_bundle_count"] == 0
    assert body["after_authorized_claim_count"] == 0
    assert body["after_authorized_evidence_bundle_count"] == 0


def test_all_zero_when_everything_is_below_threshold(client):
    _world(client)
    policy = _create_policy(client, threshold=100)
    _make_claim(client, "org-2", digest=DIGEST_B)
    revocation = _revoke(client, policy["id"])

    body = _get_impact(client, revocation["id"]).json()
    assert body["before_authorized_claim_count"] == 0
    assert body["before_authorized_evidence_bundle_count"] == 0
    assert body["after_authorized_claim_count"] == 0
    assert body["after_authorized_evidence_bundle_count"] == 0


def test_impact_reflects_current_evidence_state(client):
    _world(client)
    policy = _create_policy(client, threshold=2)
    claim = _make_claim(client, "org-2", digest=DIGEST_B)
    _attest(client, "claim", claim["id"], "org-2", SEED_B)
    revocation = _revoke(client, policy["id"])

    before = _get_impact(client, revocation["id"]).json()
    assert before["before_authorized_claim_count"] == 0

    # Evidence added after the revocation still counts: the impact is
    # computed from the current state, not reconstructed from history.
    _attest(client, "claim", claim["id"], "org-3", SEED_C)
    after = _get_impact(client, revocation["id"]).json()
    assert after["before_authorized_claim_count"] == 1
    assert after["after_authorized_claim_count"] == 0


# --- Lookup and request boundary ----------------------------------------------


def test_unknown_revocation_is_404(client):
    _world(client)
    policy = _create_policy(client)
    revocation = _revoke(client, policy["id"])

    resp = _get_impact(client, "tpr_" + "0" * 64)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == (
        "actor_trust_policy_revocation_not_found"
    )
    assert resp.json()["error"]["details"]["revocation_id"] == (
        "tpr_" + "0" * 64
    )

    # The id match is verbatim: a case variant of an existing id is unknown.
    variant = revocation["id"].upper()
    assert variant != revocation["id"]
    resp = _get_impact(client, variant)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == (
        "actor_trust_policy_revocation_not_found"
    )


def test_revocation_id_is_not_matched_against_policy_id(client):
    _world(client)
    policy = _create_policy(client)
    _revoke(client, policy["id"])

    # The policy id is not a revocation id.
    resp = _get_impact(client, policy["id"])
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == (
        "actor_trust_policy_revocation_not_found"
    )


@pytest.mark.parametrize("body", [b"{}", b" ", b"\n", b"\x00"])
def test_get_body_must_be_empty(client, body):
    _world(client)
    policy = _create_policy(client)
    revocation = _revoke(client, policy["id"])

    resp = client.request("GET", _impact_path(revocation["id"]), content=body)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize(
    "params",
    [
        {"foo": "1"},  # unknown parameter
        {"foo": ""},  # blank value
        {"threshold": "2"},  # a field name is still undeclared here
        [("foo", "1"), ("foo", "2")],  # repeated parameter
    ],
)
def test_any_query_parameter_is_422(client, params):
    _world(client)
    policy = _create_policy(client)
    revocation = _revoke(client, policy["id"])

    resp = client.get(_impact_path(revocation["id"]), params=params)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_query_parameter_is_rejected_before_unknown_id(client):
    # Validation precedes existence: a malformed request is a 422 even when
    # the revocation id is also unknown.
    resp = client.get(_impact_path("tpr_" + "0" * 64), params={"foo": "1"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_non_get_methods_are_405(client, method):
    _world(client)
    policy = _create_policy(client)
    revocation = _revoke(client, policy["id"])

    resp = getattr(client, method)(_impact_path(revocation["id"]))
    assert resp.status_code == 405
    assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Read-only guarantee and stability ----------------------------------------


def test_impact_read_is_read_only_and_repeatable(client, db_session):
    _world(client)
    policy = _create_policy(client, threshold=1)
    revocation = _revoke(client, policy["id"])

    def _state():
        return (
            db_session.execute(select(AuditEvent)).scalars().all(),
            db_session.execute(select(ActorTrustPolicyRevocation))
            .scalars()
            .all(),
        )

    events_before, revocations_before = _state()
    first = _get_impact(client, revocation["id"])
    second = _get_impact(client, revocation["id"])
    assert first.status_code == 200
    assert first.content == second.content
    events_after, revocations_after = _state()
    assert [(e.seq, e.event_type, e.resource_id) for e in events_after] == [
        (e.seq, e.event_type, e.resource_id) for e in events_before
    ]
    assert [r.id for r in revocations_after] == [
        r.id for r in revocations_before
    ]


def test_impact_survives_restart(tmp_db_url):
    app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app) as client:
        _world(client)
        policy = _create_policy(client, threshold=2)
        claim = _make_claim(client, "org-2", digest=DIGEST_B)
        _attest(client, "claim", claim["id"], "org-2", SEED_B)
        _attest(client, "claim", claim["id"], "org-3", SEED_C)
        revocation = _revoke(client, policy["id"])
        first = _get_impact(client, revocation["id"])
        assert first.status_code == 200

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        second = _get_impact(client, revocation["id"])
        assert second.status_code == 200
        assert second.content == first.content
