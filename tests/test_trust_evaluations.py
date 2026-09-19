"""Tests for the read-only reviewer trust evaluation endpoint.

Covers GET /v1/trust-evaluations: live decision from verified attestations,
distinct-signer counting across multiple keys, exact-target scoping, both
claim and evidence-bundle targets, threshold defaults and boundaries, the
404/422 boundary (validation precedes existence), repeated/unknown/blank
parameter rejection, and the no-write guarantee. All tests are deterministic
and offline (signatures are produced by the stdlib test signer).
"""

from __future__ import annotations

import hashlib

from sqlalchemy import func, select

from provenance.models import Attestation, AuditEvent
from tests.helpers import (
    DIGEST_B,
    DIGEST_C,
    content_payload,
    create_actor,
    ed25519_public_key,
    ed25519_sign,
    SEED_A,
    SEED_B,
)
from provenance.signing import attestation_message_bytes

EVIDENCE_DIGEST = hashlib.sha256(b"evidence-trust").hexdigest()


# --- Setup helpers ----------------------------------------------------------


def _create_content(client, actor_id="org-1", digest=None):
    resp = client.post(
        "/v1/contents",
        json=content_payload(actor_id=actor_id, digest=digest or DIGEST_B),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, actor_id="org-1", claim_type="authorship"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": claim_type,
            "payload": {"statement": "trust me"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_bundle(client, claim_id):
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


def _setup_claim(client):
    create_actor(client)
    return _create_claim(client, _create_content(client)["id"])


def _setup_claim_and_bundle(client):
    claim = _setup_claim(client)
    return claim, _create_bundle(client, claim["id"])


def _attestation_body(target_type, target_id, *, seed=SEED_A, signer_actor_id="org-1"):
    import base64

    signature = ed25519_sign(
        seed, attestation_message_bytes(target_type, target_id, signer_actor_id)
    )
    return {
        "target_type": target_type,
        "target_id": target_id,
        "signer_actor_id": signer_actor_id,
        "public_key": base64.b64encode(ed25519_public_key(seed)).decode("ascii"),
        "signature": base64.b64encode(signature).decode("ascii"),
    }


def _attest(client, target_type, target_id, **kwargs):
    resp = client.post(
        "/v1/attestations",
        json=_attestation_body(target_type, target_id, **kwargs),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _evaluate(client, target_type, target_id, **params):
    return client.get(
        "/v1/trust-evaluations",
        params={"target_type": target_type, "target_id": target_id, **params},
    )


# --- Successful evaluations -------------------------------------------------


def test_existing_claim_without_attestations_is_untrusted(client):
    claim = _setup_claim(client)
    resp = _evaluate(client, "claim", claim["id"])
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "target_type": "claim",
        "target_id": claim["id"],
        "min_signers": 1,
        "qualified_signer_count": 0,
        "decision": "untrusted",
    }


def test_single_verified_signer_meets_the_default_threshold(client):
    claim = _setup_claim(client)
    _attest(client, "claim", claim["id"])
    resp = _evaluate(client, "claim", claim["id"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["qualified_signer_count"] == 1
    assert body["decision"] == "trusted"
    assert body["min_signers"] == 1


def test_threshold_boundary_equality_is_trusted(client):
    claim = _setup_claim(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    _attest(client, "claim", claim["id"], seed=SEED_A, signer_actor_id="org-1")
    _attest(client, "claim", claim["id"], seed=SEED_B, signer_actor_id="org-2")

    at_two = _evaluate(client, "claim", claim["id"], min_signers=2)
    assert at_two.json()["qualified_signer_count"] == 2
    assert at_two.json()["decision"] == "trusted"

    at_three = _evaluate(client, "claim", claim["id"], min_signers=3)
    assert at_three.json()["qualified_signer_count"] == 2
    assert at_three.json()["decision"] == "untrusted"


def test_same_signer_with_distinct_keys_counts_once(client):
    claim = _setup_claim(client)
    # Same target, same signing actor, two independent key/signature pairs.
    _attest(client, "claim", claim["id"], seed=SEED_A)
    _attest(client, "claim", claim["id"], seed=SEED_B)

    resp = _evaluate(client, "claim", claim["id"])
    assert resp.json()["qualified_signer_count"] == 1
    assert resp.json()["decision"] == "trusted"

    # A second distinct actor raises the count by exactly one.
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    _attest(client, "claim", claim["id"], seed=SEED_A, signer_actor_id="org-2")
    resp = _evaluate(client, "claim", claim["id"])
    assert resp.json()["qualified_signer_count"] == 2


def test_evidence_bundle_target_is_evaluated(client):
    _claim, bundle = _setup_claim_and_bundle(client)
    _attest(client, "evidence_bundle", bundle["id"])

    resp = _evaluate(client, "evidence_bundle", bundle["id"])
    assert resp.status_code == 200
    assert resp.json() == {
        "target_type": "evidence_bundle",
        "target_id": bundle["id"],
        "min_signers": 1,
        "qualified_signer_count": 1,
        "decision": "trusted",
    }


def test_only_attestations_of_the_exact_target_count(client):
    claim_one = _setup_claim(client)
    content_two = _create_content(client, digest=DIGEST_C)
    claim_two = _create_claim(client, content_two["id"])
    bundle = _create_bundle(client, claim_one["id"])
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")

    _attest(client, "claim", claim_two["id"], seed=SEED_B, signer_actor_id="org-2")
    _attest(client, "evidence_bundle", bundle["id"])

    # Neither the other claim's signer nor the bundle's signer attest claim one.
    resp = _evaluate(client, "claim", claim_one["id"])
    assert resp.json()["qualified_signer_count"] == 0
    assert resp.json()["decision"] == "untrusted"

    # The bundle's own attestation still qualifies the bundle.
    resp = _evaluate(client, "evidence_bundle", bundle["id"])
    assert resp.json()["qualified_signer_count"] == 1


def test_evaluation_is_computed_live(client):
    claim = _setup_claim(client)
    assert _evaluate(client, "claim", claim["id"]).json()["qualified_signer_count"] == 0
    _attest(client, "claim", claim["id"])
    assert _evaluate(client, "claim", claim["id"]).json()["qualified_signer_count"] == 1

    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    _attest(client, "claim", claim["id"], seed=SEED_B, signer_actor_id="org-2")
    assert _evaluate(client, "claim", claim["id"]).json()["qualified_signer_count"] == 2


def test_min_signers_boundaries_1_and_100_are_accepted(client):
    claim = _setup_claim(client)
    low = _evaluate(client, "claim", claim["id"], min_signers=1)
    assert low.status_code == 200
    high = _evaluate(client, "claim", claim["id"], min_signers=100)
    assert high.status_code == 200
    assert high.json()["min_signers"] == 100
    assert high.json()["decision"] == "untrusted"


# --- Missing-resource boundary ----------------------------------------------


def test_unknown_claim_target_is_404(client):
    _setup_claim(client)
    resp = _evaluate(client, "claim", "clm_ghost")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "claim_not_found"
    assert error["details"]["claim_id"] == "clm_ghost"


def test_unknown_evidence_bundle_target_is_404(client):
    _setup_claim(client)
    resp = _evaluate(client, "evidence_bundle", "evb_ghost")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "evidence_bundle_not_found"
    assert error["details"]["evidence_bundle_id"] == "evb_ghost"


def test_wrong_target_type_for_existing_resource_is_404(client):
    claim, bundle = _setup_claim_and_bundle(client)
    r1 = _evaluate(client, "claim", bundle["id"])
    r2 = _evaluate(client, "evidence_bundle", claim["id"])
    assert r1.status_code == 404
    assert r1.json()["error"]["code"] == "claim_not_found"
    assert r2.status_code == 404
    assert r2.json()["error"]["code"] == "evidence_bundle_not_found"


# --- Validation boundary -----------------------------------------------------


def test_missing_target_type_or_target_id_is_422(client):
    _setup_claim(client)
    no_type = client.get(
        "/v1/trust-evaluations", params={"target_id": "clm_x"}
    )
    assert no_type.status_code == 422
    assert no_type.json()["error"]["code"] == "validation_error"

    no_id = client.get(
        "/v1/trust-evaluations", params={"target_type": "claim"}
    )
    assert no_id.status_code == 422
    assert no_id.json()["error"]["code"] == "validation_error"

    neither = client.get("/v1/trust-evaluations")
    assert neither.status_code == 422
    assert neither.json()["error"]["code"] == "validation_error"


def test_blank_or_unknown_target_type_is_422(client):
    claim = _setup_claim(client)
    for value in ("", "   ", "Claim", "claim ", " content", "evidence", "attestation"):
        resp = _evaluate(client, value, claim["id"])
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_blank_target_id_is_422(client):
    for value in ("", "   ", "\t"):
        resp = _evaluate(client, "claim", value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_min_signers_must_be_a_plain_integer_in_range(client):
    claim = _setup_claim(client)
    for value in (
        "",
        "   ",
        "abc",
        "1.5",
        "0",
        "-1",
        "101",
        "+1",
        " 1",
        "1 ",
        "1.0",
        "8.0",
        "0x1",
    ):
        resp = _evaluate(client, "claim", claim["id"], min_signers=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_422(client):
    claim = _setup_claim(client)
    base = f"/v1/trust-evaluations?target_id={claim['id']}"
    for url in (
        f"{base}&target_type=claim&target_type=evidence_bundle",
        f"{base}&target_type=claim&target_type=claim",
        f"{base}&target_type=claim&target_id={claim['id']}",
        f"{base}&target_type=claim&min_signers=1&min_signers=2",
    ):
        resp = client.get(url)
        assert resp.status_code == 422, url
        assert resp.json()["error"]["code"] == "validation_error"


def test_unknown_parameters_are_422(client):
    claim = _setup_claim(client)
    for url in (
        f"/v1/trust-evaluations?target_type=claim&target_id={claim['id']}&min_signer=1",
        f"/v1/trust-evaluations?target_type=claim&target_id={claim['id']}&foo=bar",
    ):
        resp = client.get(url)
        assert resp.status_code == 422, url
        assert resp.json()["error"]["code"] == "validation_error"


def test_validation_errors_take_precedence_over_missing_target(client):
    # A non-existent target with an otherwise malformed request is 422, not
    # 404: parameters are validated before any existence lookup.
    resp = client.get(
        "/v1/trust-evaluations",
        params={"target_type": "claim", "target_id": "clm_ghost", "min_signers": "abc"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    resp = client.get(
        "/v1/trust-evaluations",
        params={"target_type": "nope", "target_id": "clm_ghost"},
    )
    assert resp.status_code == 422

    resp = client.get(
        "/v1/trust-evaluations",
        params={"target_type": "claim", "target_id": "   "},
    )
    assert resp.status_code == 422


# --- Read-only guarantee -----------------------------------------------------


def test_evaluation_writes_no_resources_or_audit_events(client, db_session):
    claim = _setup_claim(client)
    _attest(client, "claim", claim["id"])

    attestations_before = db_session.scalar(
        select(func.count()).select_from(Attestation)
    )
    audit_before = db_session.scalar(select(func.count()).select_from(AuditEvent))

    for params in (
        {},
        {"min_signers": 1},
        {"min_signers": 2},
    ):
        resp = _evaluate(client, "claim", claim["id"], **params)
        assert resp.status_code == 200

    # Including evaluations of missing targets, which also write nothing.
    assert _evaluate(client, "claim", "clm_ghost").status_code == 404

    assert (
        db_session.scalar(select(func.count()).select_from(Attestation))
        == attestations_before
    )
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audit_before
    )
