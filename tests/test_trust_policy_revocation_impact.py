"""Tests for the read-only trust-policy-revocation impact endpoint.

Covers ``GET /v1/trust-policy-revocations/{revocation_id}/impact``: the 200
impact view (compact UTF-8 JSON terminated by exactly one newline, members in
revocation, threshold, before_authorized_claim_count,
before_authorized_evidence_bundle_count, after_authorized_claim_count,
after_authorized_evidence_bundle_count order; the revocation member in id,
policy_id, actor_id, reason, created_at order), the counterfactual
before-counts from the current evidence state (verified, non-revoked
attestations of the exact target, deduplicated by signing actor; attestation
revocations still apply), the always-zero after-counts, the all-zero results
for empty databases and below-threshold targets, the 404
``actor_trust_policy_revocation_not_found`` for unknown identifiers, the 422
``validation_error`` for any request body or any query parameter, the
framework's 405 ``method_not_allowed`` for non-GET methods, the strict
no-write guarantee, and restart consistency.

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures).
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone

from sqlalchemy import func, select

from provenance.app import create_app
from provenance.config import Settings
from provenance.models import (
    EVENT_ACTOR_TRUST_POLICY_REVOKED,
    ActorTrustPolicyRevocation,
    AuditEvent,
)
from provenance.access_signing import access_message_bytes
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

EVIDENCE_DIGEST = hashlib.sha256(b"evidence-impact").hexdigest()
SEED_C = b"test-ed25519-seed-c-00000000000000"[:32]

IMPACT_KEYS = [
    "revocation",
    "threshold",
    "before_authorized_claim_count",
    "before_authorized_evidence_bundle_count",
    "after_authorized_claim_count",
    "after_authorized_evidence_bundle_count",
]
REVOCATION_KEYS = ["id", "policy_id", "actor_id", "reason", "created_at"]


# --- Fixture-style setup ------------------------------------------------------


def _make_claim(client, actor_id="org-1", digest=DIGEST_A):
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
    """Two actors, each holding a current authentication key."""
    create_actor(client)  # org-1
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    _attest(client, "claim", _make_claim(client, "org-1")["id"], "org-1", SEED_A)
    _attest(
        client,
        "claim",
        _make_claim(client, "org-2", digest=DIGEST_B)["id"],
        "org-2",
        SEED_B,
    )


# --- Signed-request helpers ---------------------------------------------------


def _signed_headers(method, path, body, *, actor="org-1", seed=SEED_A):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
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


def _revoke_policy(client, policy_id, reason="no longer relied upon", **kwargs):
    actor = kwargs.pop("actor", "org-1")
    seed = kwargs.pop("seed", SEED_A)
    body = json.dumps({"policy_id": policy_id, "reason": reason}).encode()
    headers = {
        "Content-Type": "application/json",
        **_signed_headers(
            "POST", REVOCATIONS_PATH, body, actor=actor, seed=seed
        ),
    }
    resp = client.post(REVOCATIONS_PATH, content=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _impact_path(revocation_id):
    return f"{REVOCATIONS_PATH}/{revocation_id}/impact"


def _impact(client, revocation_id, **kwargs):
    return client.get(_impact_path(revocation_id), **kwargs)


def _setup_impact(client, threshold=2, reason="key material retired"):
    """A world with a revoked policy; returns ``(policy, revocation)``."""
    _world(client)
    policy = _create_policy(client, threshold=threshold)
    revocation = _revoke_policy(client, policy["id"], reason=reason)
    return policy, revocation


# --- Happy path -----------------------------------------------------------------


def test_impact_counts_targets_reaching_the_threshold(client):
    policy, revocation = _setup_impact(client, threshold=2)

    # Two distinct signers -> meets the threshold of 2.
    met_claim = _make_claim(client, "org-1", digest=DIGEST_C)
    _attest(client, "claim", met_claim["id"], "org-1", SEED_A)
    _attest(client, "claim", met_claim["id"], "org-2", SEED_B)
    # One signer only -> below the threshold.
    below_claim = _make_claim(
        client,
        "org-1",
        digest=hashlib.sha256(b"content-below").hexdigest(),
    )
    _attest(client, "claim", below_claim["id"], "org-1", SEED_A)
    # A bundle with two distinct signers -> meets the threshold.
    bundle = _make_bundle(client, met_claim["id"])
    _attest(client, "evidence_bundle", bundle["id"], "org-1", SEED_A)
    _attest(client, "evidence_bundle", bundle["id"], "org-2", SEED_B)

    resp = _impact(client, revocation["id"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert list(body) == IMPACT_KEYS
    assert list(body["revocation"]) == REVOCATION_KEYS
    assert body["revocation"] == revocation
    assert body["threshold"] == policy["threshold"] == 2
    # The world's two setup claims carry one signer each: below threshold.
    assert body["before_authorized_claim_count"] == 1
    assert body["before_authorized_evidence_bundle_count"] == 1
    assert body["after_authorized_claim_count"] == 0
    assert body["after_authorized_evidence_bundle_count"] == 0
    for key in IMPACT_KEYS[1:]:
        assert isinstance(body[key], int)
        assert body[key] >= 0


def test_impact_response_is_compact_utf8_json_with_one_newline(client):
    _, revocation = _setup_impact(client, threshold=1)
    resp = _impact(client, revocation["id"])
    assert resp.status_code == 200
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    for earlier, later in zip(IMPACT_KEYS, IMPACT_KEYS[1:]):
        assert raw.index(f'"{earlier}"'.encode()) < raw.index(
            f'"{later}"'.encode()
        )
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    assert raw == expected


def test_revocation_member_is_the_exact_public_view(client):
    _, revocation = _setup_impact(client, reason="  撤回原因 🔒\n")
    resp = _impact(client, revocation["id"])
    assert resp.status_code == 200
    view = resp.json()["revocation"]
    assert view["id"] == revocation["id"]
    assert view["id"].startswith("tpr_")
    assert view["reason"] == "  撤回原因 🔒\n"
    created_at = datetime.fromisoformat(view["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    # No evidence byte, signature, or key material ever appears.
    assert set(view) == set(REVOCATION_KEYS)


def test_distinct_signers_deduplicated_by_signing_actor(client):
    _, revocation = _setup_impact(client, threshold=2)
    claim = _make_claim(client, "org-1", digest=DIGEST_C)
    # The same signing actor under two different keys qualifies once.
    _attest(client, "claim", claim["id"], "org-1", SEED_A)
    _attest(client, "claim", claim["id"], "org-1", SEED_C)

    body = _impact(client, revocation["id"]).json()
    assert body["before_authorized_claim_count"] == 0

    # A second distinct actor pushes the same target over the threshold.
    _attest(client, "claim", claim["id"], "org-2", SEED_B)
    body = _impact(client, revocation["id"]).json()
    assert body["before_authorized_claim_count"] == 1


def test_counts_reflect_the_current_evidence_state(client):
    _, revocation = _setup_impact(client, threshold=2)
    claim = _make_claim(client, "org-1", digest=DIGEST_C)
    attestation = _attest(client, "claim", claim["id"], "org-2", SEED_B)
    _attest(client, "claim", claim["id"], "org-1", SEED_A)

    # The setup claims carry one signer each (below the threshold of 2);
    # only the new claim, with two distinct signers, qualifies.
    body = _impact(client, revocation["id"]).json()
    assert body["before_authorized_claim_count"] == 1

    # A later attestation revocation shrinks the counterfactual: the impact
    # is computed live, never reconstructed from history.
    resp = client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": attestation["id"],
            "revoker_actor_id": "org-1",
            "reason": "superseded",
        },
    )
    assert resp.status_code == 201, resp.text
    body = _impact(client, revocation["id"]).json()
    assert body["before_authorized_claim_count"] == 0


def test_no_claims_and_no_bundles_is_all_zero(client):
    # No claims or bundles beyond the world setup, and the threshold of 2
    # puts every one-signer setup claim below it.
    _, revocation = _setup_impact(client, threshold=2)
    body = _impact(client, revocation["id"]).json()
    assert body["before_authorized_claim_count"] == 0
    assert body["before_authorized_evidence_bundle_count"] == 0
    assert body["after_authorized_claim_count"] == 0
    assert body["after_authorized_evidence_bundle_count"] == 0


def test_empty_database_impact_is_all_zero(client, db_session):
    # Only the actor, its policy, and the revocation exist; no claim, no
    # evidence bundle, no attestation at all. (Bootstrapping an
    # authentication key through the API requires an attestation, so the
    # rows are inserted directly to reach a truly empty evidence state.)
    from provenance import ids
    from provenance.models import Actor, ActorTrustPolicy

    db_session.add(Actor(id="org-1", name="Example Org", type="organization"))
    policy = ActorTrustPolicy(
        id=ids.actor_trust_policy_id("org-1"),
        actor_id="org-1",
        threshold=1,
        enabled=True,
    )
    db_session.add(policy)
    revocation = ActorTrustPolicyRevocation(
        id=ids.actor_trust_policy_revocation_id(policy.id),
        policy_id=policy.id,
        actor_id="org-1",
        reason="withdrawing",
    )
    db_session.add(revocation)
    db_session.commit()

    resp = _impact(client, revocation.id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert list(body) == IMPACT_KEYS
    assert body["threshold"] == 1
    assert body["before_authorized_claim_count"] == 0
    assert body["before_authorized_evidence_bundle_count"] == 0
    assert body["after_authorized_claim_count"] == 0
    assert body["after_authorized_evidence_bundle_count"] == 0


def test_after_counts_are_zero_even_when_threshold_is_met(client):
    _, revocation = _setup_impact(client, threshold=1)
    claim = _make_claim(client, "org-1", digest=DIGEST_C)
    _attest(client, "claim", claim["id"], "org-1", SEED_A)

    body = _impact(client, revocation["id"]).json()
    # The world's two setup claims also carry one signer each: with a
    # threshold of 1 every attested claim qualifies.
    assert body["before_authorized_claim_count"] == 3
    # The revoked policy no longer participates in decide_trust: it
    # authorizes nothing, whatever the evidence state.
    assert body["after_authorized_claim_count"] == 0
    assert body["after_authorized_evidence_bundle_count"] == 0

    headers = _signed_headers("GET", DECISIONS_PATH, b"")
    decision = client.get(
        DECISIONS_PATH,
        params={"target_type": "claim", "target_id": claim["id"]},
        headers=headers,
    )
    assert decision.status_code == 200
    assert decision.json()["reason"] == "policy_missing"


# --- Unknown revocation ---------------------------------------------------------


def test_unknown_revocation_is_a_specific_404(client):
    _, revocation = _setup_impact(client)
    ghost = "tpr_" + "0" * 64
    resp = _impact(client, ghost)
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "actor_trust_policy_revocation_not_found"
    assert error["details"]["revocation_id"] == ghost

    # The identifier match is verbatim: a real id with different casing or
    # surrounding whitespace is equally unknown.
    assert _impact(client, revocation["id"].upper()).status_code == 404
    assert _impact(client, " " + revocation["id"]).status_code == 404


# --- Request-shape validation -----------------------------------------------------


def test_any_request_body_is_422(client):
    _, revocation = _setup_impact(client)
    for body in (b"{}", b" ", b"null", b"\x00"):
        resp = client.request("GET", _impact_path(revocation["id"]), content=body)
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"


def test_any_query_parameter_is_422(client):
    _, revocation = _setup_impact(client)
    path = _impact_path(revocation["id"])
    for url in (
        f"{path}?unknown=1",
        f"{path}?threshold=2",
        f"{path}?blank=",
        f"{path}?x=1&x=2",
    ):
        resp = client.get(url)
        assert resp.status_code == 422, url
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_get_methods_are_405(client):
    _, revocation = _setup_impact(client)
    path = _impact_path(revocation["id"])
    for method in ("post", "put", "patch", "delete"):
        resp = getattr(client, method)(path)
        assert resp.status_code == 405, method
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Read-only guarantee and stability --------------------------------------------


def test_impact_is_strictly_read_only_and_repeatable(client, db_session):
    _, revocation = _setup_impact(client, threshold=1)
    claim = _make_claim(client, "org-1", digest=DIGEST_C)
    _attest(client, "claim", claim["id"], "org-1", SEED_A)

    def _state():
        return (
            db_session.execute(select(func.count()).select_from(AuditEvent)).scalar_one(),
            db_session.execute(
                select(func.count()).select_from(ActorTrustPolicyRevocation)
            ).scalar_one(),
            db_session.execute(
                select(AuditEvent).where(
                    AuditEvent.event_type == EVENT_ACTOR_TRUST_POLICY_REVOKED
                )
            ).scalars().all(),
        )

    audits_before, revocations_before, events_before = _state()
    first = _impact(client, revocation["id"])
    assert first.status_code == 200
    for _ in range(3):
        repeat = _impact(client, revocation["id"])
        assert repeat.status_code == 200
        assert repeat.content == first.content
    audits_after, revocations_after, events_after = _state()
    assert audits_after == audits_before
    assert revocations_after == revocations_before
    assert [e.seq for e in events_after] == [e.seq for e in events_before]


def test_impact_survives_restart(tmp_db_url):
    from fastapi.testclient import TestClient

    app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app) as client:
        _, revocation = _setup_impact(client, threshold=1)
        claim = _make_claim(client, "org-1", digest=DIGEST_C)
        _attest(client, "claim", claim["id"], "org-1", SEED_A)
        first = _impact(client, revocation["id"])
        assert first.status_code == 200

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        again = _impact(client, revocation["id"])
        assert again.status_code == 200
        assert again.content == first.content
        # Two setup claims plus the later one, each with one signer.
        assert again.json()["before_authorized_claim_count"] == 3
