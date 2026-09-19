"""Tests for reviewer trust evaluations.

Covers GET /v1/trust-evaluations: distinct-signer counting over verified
attestations (multiple attestations from one subject count once), the
``trusted``/``untrusted`` threshold boundary (``min_signers`` defaults to 1,
strictly 1..100), per-target-type 404s, the full 422 boundary (blank,
unknown, and repeated parameters; non-integer, out-of-range, plus-prefixed,
or whitespace thresholds), validation-before-existence ordering, and the
read-only guarantee (no resource or audit rows). All tests are deterministic
and offline.
"""

from __future__ import annotations

import base64
import hashlib

from sqlalchemy import func, select

from provenance.models import Attestation, AuditEvent
from provenance.signing import attestation_message_bytes
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    SEED_A,
    SEED_B,
    content_payload,
    create_actor,
    ed25519_public_key,
    ed25519_sign,
)

EVIDENCE_DIGEST = hashlib.sha256(b"evidence-trust").hexdigest()

#: Deterministic seeds beyond the shared pair, one per distinct signer.
SEED_C = b"test-ed25519-seed-c-00000000000000"[:32]


# --- Setup helpers ----------------------------------------------------------


def _create_content(client, actor_id="org-1", digest=DIGEST_A):
    resp = client.post(
        "/v1/contents", json=content_payload(actor_id=actor_id, digest=digest)
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
            "payload": {"statement": "endorsed"},
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


def _setup_claim(client, actors=("org-1",)):
    for actor_id in actors:
        create_actor(client, actor_id=actor_id)
    return _create_claim(client, _create_content(client)["id"])


def _setup_claim_and_bundle(client, actors=("org-1",)):
    for actor_id in actors:
        create_actor(client, actor_id=actor_id)
    claim = _create_claim(client, _create_content(client)["id"])
    return claim, _create_bundle(client, claim["id"])


def _attest(client, target_type, target_id, signer_actor_id, seed):
    public_key = base64.b64encode(ed25519_public_key(seed)).decode("ascii")
    signature = base64.b64encode(
        ed25519_sign(
            seed,
            attestation_message_bytes(target_type, target_id, signer_actor_id),
        )
    ).decode("ascii")
    resp = client.post(
        "/v1/attestations",
        json={
            "target_type": target_type,
            "target_id": target_id,
            "signer_actor_id": signer_actor_id,
            "public_key": public_key,
            "signature": signature,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _evaluate(client, **params):
    return client.get("/v1/trust-evaluations", params=params)


# --- Response shape and defaults --------------------------------------------


def test_claim_with_one_verified_signer_is_trusted_by_default(client):
    claim = _setup_claim(client)
    _attest(client, "claim", claim["id"], "org-1", SEED_A)

    resp = _evaluate(client, target_type="claim", target_id=claim["id"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {
        "target_type",
        "target_id",
        "min_signers",
        "qualified_signer_count",
        "decision",
    }
    assert body == {
        "target_type": "claim",
        "target_id": claim["id"],
        "min_signers": 1,
        "qualified_signer_count": 1,
        "decision": "trusted",
    }


def test_claim_without_attestations_is_untrusted_at_zero(client):
    claim = _setup_claim(client)
    resp = _evaluate(client, target_type="claim", target_id=claim["id"])
    assert resp.status_code == 200
    assert resp.json() == {
        "target_type": "claim",
        "target_id": claim["id"],
        "min_signers": 1,
        "qualified_signer_count": 0,
        "decision": "untrusted",
    }


# --- Distinct-signer dedup --------------------------------------------------


def test_repeated_attestations_from_one_subject_count_once(client):
    claim = _setup_claim(client)
    # Same signing actor, two different key/signature pairs: two stored
    # attestation rows but one independent signing subject.
    _attest(client, "claim", claim["id"], "org-1", SEED_A)
    _attest(client, "claim", claim["id"], "org-1", SEED_B)

    resp = _evaluate(client, target_type="claim", target_id=claim["id"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["qualified_signer_count"] == 1
    assert body["decision"] == "trusted"


def test_distinct_signers_are_counted_independently(client):
    claim = _setup_claim(client, actors=("org-1", "org-2", "org-3"))
    _attest(client, "claim", claim["id"], "org-1", SEED_A)
    _attest(client, "claim", claim["id"], "org-2", SEED_B)
    _attest(client, "claim", claim["id"], "org-3", SEED_C)

    resp = _evaluate(
        client, target_type="claim", target_id=claim["id"], min_signers=3
    )
    assert resp.status_code == 200
    assert resp.json()["qualified_signer_count"] == 3
    assert resp.json()["decision"] == "trusted"


def test_only_verified_attestations_of_the_exact_target_count(client):
    claim = _setup_claim(client, actors=("org-1", "org-2"))
    other = _create_claim(
        client, _create_content(client, digest=DIGEST_B)["id"],
        claim_type="capture",
    )
    bundle = _create_bundle(client, claim["id"])
    _attest(client, "claim", claim["id"], "org-1", SEED_A)
    _attest(client, "claim", claim["id"], "org-2", SEED_B)
    # An attestation of a different claim never enters this target's count.
    _attest(client, "claim", other["id"], "org-1", SEED_A)
    # Neither does an evidence-bundle endorsement of this claim's bundle:
    # target_type and target_id must both match.
    _attest(client, "evidence_bundle", bundle["id"], "org-1", SEED_A)

    resp = _evaluate(
        client, target_type="claim", target_id=claim["id"], min_signers=3
    )
    assert resp.status_code == 200
    assert resp.json()["qualified_signer_count"] == 2
    assert resp.json()["decision"] == "untrusted"

    # The bundle evaluation, conversely, sees only its own attestation.
    resp = _evaluate(
        client, target_type="evidence_bundle", target_id=bundle["id"]
    )
    assert resp.status_code == 200
    assert resp.json()["qualified_signer_count"] == 1
    assert resp.json()["decision"] == "trusted"


# --- Threshold boundaries ----------------------------------------------------


def test_threshold_boundary_and_range(client):
    claim = _setup_claim(client, actors=("org-1", "org-2"))
    _attest(client, "claim", claim["id"], "org-1", SEED_A)
    _attest(client, "claim", claim["id"], "org-2", SEED_B)

    for threshold, decision in (
        ("1", "trusted"),
        ("2", "trusted"),  # count reaches the threshold exactly
        ("3", "untrusted"),
        ("100", "untrusted"),  # upper bound is accepted
    ):
        resp = _evaluate(
            client,
            target_type="claim",
            target_id=claim["id"],
            min_signers=threshold,
        )
        assert resp.status_code == 200, threshold
        assert resp.json()["decision"] == decision
        assert resp.json()["min_signers"] == int(threshold)
        assert resp.json()["qualified_signer_count"] == 2


def test_evidence_bundle_target_is_evaluated(client):
    _, bundle = _setup_claim_and_bundle(client, actors=("org-1", "org-2"))
    _attest(client, "evidence_bundle", bundle["id"], "org-1", SEED_A)

    resp = _evaluate(
        client,
        target_type="evidence_bundle",
        target_id=bundle["id"],
        min_signers=2,
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "target_type": "evidence_bundle",
        "target_id": bundle["id"],
        "min_signers": 2,
        "qualified_signer_count": 1,
        "decision": "untrusted",
    }


# --- Missing targets ---------------------------------------------------------


def test_missing_claim_is_claim_not_found(client):
    _setup_claim(client)
    resp = _evaluate(
        client, target_type="claim", target_id="clm_ghost", min_signers=1
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "claim_not_found"
    assert error["details"]["claim_id"] == "clm_ghost"


def test_missing_evidence_bundle_is_evidence_bundle_not_found(client):
    create_actor(client)
    resp = _evaluate(
        client, target_type="evidence_bundle", target_id="evb_ghost"
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "evidence_bundle_not_found"
    assert error["details"]["evidence_bundle_id"] == "evb_ghost"


def test_target_lookup_uses_the_declared_type(client):
    # A real bundle id declared as a claim is a missing claim, not a match.
    _, bundle = _setup_claim_and_bundle(client)
    _attest(client, "evidence_bundle", bundle["id"], "org-1", SEED_A)
    resp = _evaluate(client, target_type="claim", target_id=bundle["id"])
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "claim_not_found"


# --- Parameter validation ----------------------------------------------------


def test_missing_required_parameters_are_validation_errors(client):
    for url in (
        "/v1/trust-evaluations",
        "/v1/trust-evaluations?target_id=clm_x",
        "/v1/trust-evaluations?target_type=claim",
    ):
        resp = client.get(url)
        assert resp.status_code == 422, url
        assert resp.json()["error"]["code"] == "validation_error"


def test_blank_or_unknown_target_type_is_validation_error(client):
    claim = _setup_claim(client)
    for value in ("", "   ", "claims", "Claim", "content", "evidence"):
        resp = _evaluate(
            client, target_type=value, target_id=claim["id"]
        )
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_blank_target_id_is_validation_error(client):
    for value in ("", "   ", "\t"):
        resp = _evaluate(
            client, target_type="claim", target_id=value
        )
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_illegal_min_signers_values_are_validation_errors(client):
    claim = _setup_claim(client)
    for value in ("0", "101", "-1", "1.5", "abc", "8.0", "", "  2", "2  "):
        resp = _evaluate(
            client, target_type="claim", target_id=claim["id"],
            min_signers=value,
        )
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_plus_prefixed_threshold_is_validation_error(client):
    claim = _setup_claim(client)
    # Percent-encoded '+' must not be accepted as a positive-integer sign.
    resp = client.get(
        f"/v1/trust-evaluations?target_type=claim"
        f"&target_id={claim['id']}&min_signers=%2B1"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_validation_errors(client):
    claim = _setup_claim(client)
    base = "/v1/trust-evaluations"
    cid = claim["id"]
    for suffix in (
        f"target_type=claim&target_type=evidence_bundle&target_id={cid}",
        f"target_type=claim&target_type=claim&target_id={cid}",
        f"target_type=claim&target_id={cid}&target_id={cid}",
        f"target_type=claim&target_id={cid}&min_signers=1&min_signers=2",
        f"target_type=claim&target_id={cid}&min_signers=1&min_signers=1",
    ):
        resp = client.get(f"{base}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_unknown_parameters_are_validation_errors(client):
    claim = _setup_claim(client)
    for suffix in (
        f"target_type=claim&target_id={claim['id']}&extra=1",
        f"target_type=claim&target_id={claim['id']}&min_signers=1&bogus=x",
        "unexpected=1",
    ):
        resp = client.get(f"/v1/trust-evaluations?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_validation_errors_take_precedence_over_missing_target(client):
    # Parameters are fully validated before the target is ever looked up.
    for suffix in (
        "target_type=claim&target_id=clm_ghost&min_signers=0",
        "target_type=claim&target_id=clm_ghost&min_signers=%2B1",
        "target_type=bogus&target_id=clm_ghost",
        "target_type=claim&target_id=%20%20",
        "target_type=claim&target_id=clm_ghost&target_id=clm_other",
        "target_type=claim&target_id=clm_ghost&unknown=1",
    ):
        resp = client.get(f"/v1/trust-evaluations?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"

    # The same ghost target with structurally valid parameters is a 404.
    resp = _evaluate(client, target_type="claim", target_id="clm_ghost")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "claim_not_found"


# --- Read-only guarantee -----------------------------------------------------


def test_evaluations_write_no_resources_or_audit_events(client, db_session):
    claim = _setup_claim(client, actors=("org-1", "org-2"))
    _attest(client, "claim", claim["id"], "org-1", SEED_A)

    def attestation_rows():
        return db_session.scalar(
            select(func.count()).select_from(Attestation)
        )

    def audit_rows():
        return db_session.scalar(select(func.count()).select_from(AuditEvent))

    before_attestations = attestation_rows()
    before_audit = audit_rows()

    # Both decisions, both target types (incl. their 404s): nothing mutates.
    _evaluate(client, target_type="claim", target_id=claim["id"])
    _evaluate(
        client, target_type="claim", target_id=claim["id"], min_signers=5
    )
    _evaluate(client, target_type="claim", target_id="clm_ghost")
    _evaluate(
        client, target_type="evidence_bundle", target_id="evb_ghost"
    )

    assert attestation_rows() == before_attestations
    assert audit_rows() == before_audit
