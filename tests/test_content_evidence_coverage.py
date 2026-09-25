"""Tests for the read-only content evidence-coverage summary endpoint.

Covers GET /v1/contents/{content_id}/evidence-coverage: the three coverage
states, direct-claim and associated-bundle scoping, distinct qualified
signer counting, retention of revoked proofs in the proof count, bundle
dedup across the content's claims, the 404/422 boundary, the compact wire
format (member order, single trailing newline), no-lineage traversal,
no-write guarantees, idempotent-resource retries, repeat reads, and stable
counts across an app restart. All tests are deterministic and offline
(signatures are produced by the stdlib test signer).
"""

from __future__ import annotations

import hashlib

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance.app import create_app
from provenance.config import Settings
from provenance.models import Attestation, AuditEvent
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

EVIDENCE_DIGEST_1 = hashlib.sha256(b"evidence-coverage-1").hexdigest()
EVIDENCE_DIGEST_2 = hashlib.sha256(b"evidence-coverage-2").hexdigest()


# --- Setup helpers ----------------------------------------------------------


def _create_content(client, actor_id="org-1", digest=DIGEST_A):
    resp = client.post(
        "/v1/contents",
        json=content_payload(actor_id=actor_id, digest=digest),
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
            "payload": {"statement": "coverage claim"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_bundle(client, claim_id, digest=EVIDENCE_DIGEST_1, evidence_type="raw_capture"):
    resp = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim_id,
            "evidence_type": evidence_type,
            "digest_algorithm": "sha256",
            "digest_hex": digest,
            "media_type": "image/jpeg",
            "metadata": {"source": "camera-1"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


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


def _revoke(client, attestation_id, *, revoker_actor_id="org-1", reason="key compromise"):
    resp = client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": attestation_id,
            "revoker_actor_id": revoker_actor_id,
            "reason": reason,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _coverage(client, content_id):
    return client.get(f"/v1/contents/{content_id}/evidence-coverage")


# --- Wire format ------------------------------------------------------------


def test_success_body_is_compact_ordered_json_with_one_newline(client):
    create_actor(client)
    content = _create_content(client)
    resp = _coverage(client, content["id"])
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/json"
    assert resp.content.endswith(b"\n")
    assert not resp.content.endswith(b"\n\n")
    assert resp.content == (
        b'{"content_id":"'
        + content["id"].encode("ascii")
        + b'","claim_count":0,"bundle_count":0,'
        b'"attestation_count":0,"qualified_signer_count":0,'
        b'"coverage_status":"uncovered"}\n'
    )
    # Exactly the six declared members in order.
    assert list(resp.json().keys()) == [
        "content_id",
        "claim_count",
        "bundle_count",
        "attestation_count",
        "qualified_signer_count",
        "coverage_status",
    ]


# --- Coverage states --------------------------------------------------------


def test_content_without_claims_is_all_zero_uncovered(client):
    create_actor(client)
    content = _create_content(client)
    body = _coverage(client, content["id"]).json()
    assert body == {
        "content_id": content["id"],
        "claim_count": 0,
        "bundle_count": 0,
        "attestation_count": 0,
        "qualified_signer_count": 0,
        "coverage_status": "uncovered",
    }


def test_claim_without_proofs_is_partial_with_zero_signers(client):
    create_actor(client)
    content = _create_content(client)
    _create_claim(client, content["id"])
    body = _coverage(client, content["id"]).json()
    assert body["claim_count"] == 1
    assert body["bundle_count"] == 0
    assert body["attestation_count"] == 0
    assert body["qualified_signer_count"] == 0
    assert body["coverage_status"] == "partial"


def test_claim_and_bundle_without_proofs_is_partial(client):
    create_actor(client)
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    _create_bundle(client, claim["id"])
    body = _coverage(client, content["id"]).json()
    assert body["claim_count"] == 1
    assert body["bundle_count"] == 1
    assert body["attestation_count"] == 0
    assert body["qualified_signer_count"] == 0
    assert body["coverage_status"] == "partial"


def test_one_verified_claim_attestation_makes_covered(client):
    create_actor(client)
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    _attest(client, "claim", claim["id"])
    body = _coverage(client, content["id"]).json()
    assert body == {
        "content_id": content["id"],
        "claim_count": 1,
        "bundle_count": 0,
        "attestation_count": 1,
        "qualified_signer_count": 1,
        "coverage_status": "covered",
    }


def test_one_verified_bundle_attestation_makes_covered(client):
    create_actor(client)
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    bundle = _create_bundle(client, claim["id"])
    _attest(client, "evidence_bundle", bundle["id"])
    body = _coverage(client, content["id"]).json()
    assert body["claim_count"] == 1
    assert body["bundle_count"] == 1
    assert body["attestation_count"] == 1
    assert body["qualified_signer_count"] == 1
    assert body["coverage_status"] == "covered"


def test_counts_aggregate_over_multiple_claims_and_bundles(client):
    create_actor(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    content = _create_content(client)
    claim_one = _create_claim(client, content["id"], claim_type="authorship")
    claim_two = _create_claim(client, content["id"], claim_type="integrity")
    bundle_one = _create_bundle(client, claim_one["id"], digest=EVIDENCE_DIGEST_1)
    bundle_two = _create_bundle(client, claim_two["id"], digest=EVIDENCE_DIGEST_2)

    # Two distinct subjects over the claim/bundle union.
    _attest(client, "claim", claim_one["id"], seed=SEED_A, signer_actor_id="org-1")
    _attest(
        client, "evidence_bundle", bundle_one["id"], seed=SEED_B,
        signer_actor_id="org-2",
    )
    _attest(
        client, "evidence_bundle", bundle_two["id"], seed=SEED_A,
        signer_actor_id="org-1",
    )

    body = _coverage(client, content["id"]).json()
    assert body["claim_count"] == 2
    assert body["bundle_count"] == 2
    assert body["attestation_count"] == 3
    # org-1 attests two different targets but is one distinct subject.
    assert body["qualified_signer_count"] == 2
    assert body["coverage_status"] == "covered"


def test_same_signer_with_distinct_keys_counts_once(client):
    create_actor(client)
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    bundle = _create_bundle(client, claim["id"])
    _attest(client, "claim", claim["id"], seed=SEED_A)
    _attest(client, "claim", claim["id"], seed=SEED_B)
    _attest(client, "evidence_bundle", bundle["id"], seed=SEED_B)

    body = _coverage(client, content["id"]).json()
    assert body["attestation_count"] == 3
    assert body["qualified_signer_count"] == 1
    assert body["coverage_status"] == "covered"


def test_duplicate_resource_retries_do_not_inflate_counts(client):
    create_actor(client)
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    bundle = _create_bundle(client, claim["id"])

    # Idempotent retries return 200 and must not add claims, bundles, or
    # attestations.
    retry_claim = client.post(
        "/v1/claims",
        json={
            "content_id": content["id"],
            "actor_id": "org-1",
            "claim_type": "authorship",
            "payload": {"statement": "coverage claim"},
        },
    )
    assert retry_claim.status_code == 200
    retry_bundle = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim["id"],
            "evidence_type": "raw_capture",
            "digest_algorithm": "sha256",
            "digest_hex": EVIDENCE_DIGEST_1,
            "media_type": "image/jpeg",
            "metadata": {"source": "camera-1"},
        },
    )
    assert retry_bundle.status_code == 200
    body = _attestation_body("evidence_bundle", bundle["id"])
    first = client.post("/v1/attestations", json=body)
    assert first.status_code == 201
    retry_att = client.post("/v1/attestations", json=body)
    assert retry_att.status_code == 200

    summary = _coverage(client, content["id"]).json()
    assert summary["claim_count"] == 1
    assert summary["bundle_count"] == 1
    assert summary["attestation_count"] == 1
    assert summary["qualified_signer_count"] == 1
    assert summary["coverage_status"] == "covered"


# --- Revocations ------------------------------------------------------------


def test_revoked_attestation_still_counts_but_does_not_qualify(client):
    create_actor(client)
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    attestation = _attest(client, "claim", claim["id"])

    covered = _coverage(client, content["id"]).json()
    assert covered["coverage_status"] == "covered"

    _revoke(client, attestation["id"])
    body = _coverage(client, content["id"]).json()
    # The proof is retained and still counted; it just no longer qualifies.
    assert body["attestation_count"] == 1
    assert body["qualified_signer_count"] == 0
    assert body["coverage_status"] == "partial"


def test_all_attestations_revoked_is_partial_with_zero_qualified(client):
    create_actor(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    bundle = _create_bundle(client, claim["id"])
    first = _attest(client, "claim", claim["id"], seed=SEED_A, signer_actor_id="org-1")
    second = _attest(
        client, "evidence_bundle", bundle["id"], seed=SEED_B,
        signer_actor_id="org-2",
    )
    _revoke(client, first["id"])
    _revoke(client, second["id"])

    body = _coverage(client, content["id"]).json()
    assert body["attestation_count"] == 2
    assert body["qualified_signer_count"] == 0
    assert body["coverage_status"] == "partial"


def test_one_remaining_qualified_signer_keeps_covered(client):
    create_actor(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    first = _attest(client, "claim", claim["id"], seed=SEED_A, signer_actor_id="org-1")
    _attest(client, "claim", claim["id"], seed=SEED_B, signer_actor_id="org-2")
    _revoke(client, first["id"])

    body = _coverage(client, content["id"]).json()
    assert body["attestation_count"] == 2
    assert body["qualified_signer_count"] == 1
    assert body["coverage_status"] == "covered"


# --- Scoping ----------------------------------------------------------------


def test_only_direct_claims_and_their_bundles_count(client):
    create_actor(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    target = _create_content(client, digest=DIGEST_A)
    other = _create_content(client, digest=DIGEST_C)
    target_claim = _create_claim(client, target["id"])
    other_claim = _create_claim(client, other["id"])
    target_bundle = _create_bundle(client, target_claim["id"], digest=EVIDENCE_DIGEST_1)
    _create_bundle(client, other_claim["id"], digest=EVIDENCE_DIGEST_2)

    # Proofs of the other content's claim/bundle never leak into the summary.
    _attest(client, "claim", other_claim["id"], seed=SEED_B, signer_actor_id="org-2")
    _attest(
        client, "evidence_bundle",
        _create_bundle(
            client, other_claim["id"],
            digest=hashlib.sha256(b"other-third").hexdigest(),
            evidence_type="signature_record",
        )["id"],
        seed=SEED_B,
        signer_actor_id="org-2",
    )

    body = _coverage(client, target["id"]).json()
    assert body["claim_count"] == 1
    assert body["bundle_count"] == 1
    assert body["attestation_count"] == 0
    assert body["qualified_signer_count"] == 0
    assert body["coverage_status"] == "partial"

    # A proof of the target bundle covers the target only.
    _attest(client, "evidence_bundle", target_bundle["id"])
    body = _coverage(client, target["id"]).json()
    assert body["attestation_count"] == 1
    assert body["qualified_signer_count"] == 1
    assert body["coverage_status"] == "covered"


def test_lineage_relations_are_not_traversed(client):
    create_actor(client)
    parent = _create_content(client, digest=DIGEST_A)
    child = _create_content(client, digest=DIGEST_B)
    parent_claim = _create_claim(client, parent["id"])
    _create_claim(client, child["id"])
    _attest(client, "claim", parent_claim["id"])
    relation = client.post(
        "/v1/content-relations",
        json={
            "content_id": child["id"],
            "parent_content_id": parent["id"],
            "relation_type": "derived_from",
        },
    )
    assert relation.status_code == 201

    # The child has its own claim but inherits nothing through the edge.
    child_body = _coverage(client, child["id"]).json()
    assert child_body["claim_count"] == 1
    assert child_body["bundle_count"] == 0
    assert child_body["attestation_count"] == 0
    assert child_body["qualified_signer_count"] == 0
    assert child_body["coverage_status"] == "partial"

    parent_body = _coverage(client, parent["id"]).json()
    assert parent_body["claim_count"] == 1
    assert parent_body["attestation_count"] == 1
    assert parent_body["qualified_signer_count"] == 1
    assert parent_body["coverage_status"] == "covered"


# --- Missing-resource boundary ----------------------------------------------


def test_unknown_content_is_404(client):
    create_actor(client)
    _create_content(client)
    resp = _coverage(client, "cnt_ghost")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "content_not_found"
    assert error["details"]["content_id"] == "cnt_ghost"


def test_blank_path_content_id_is_422(client):
    create_actor(client)
    content = _create_content(client)
    for raw_path in (
        "/v1/contents/%20/evidence-coverage",
        "/v1/contents/%20%20/evidence-coverage",
        "/v1/contents/%09/evidence-coverage",
    ):
        resp = client.get(raw_path)
        assert resp.status_code == 422, raw_path
        assert resp.json()["error"]["code"] == "validation_error"
    # A well-formed id on the same route still succeeds.
    assert _coverage(client, content["id"]).status_code == 200


# --- Request-shape boundary --------------------------------------------------


def test_any_query_parameter_is_422_before_lookup(client):
    create_actor(client)
    content = _create_content(client)
    base = f"/v1/contents/{content['id']}/evidence-coverage"
    for url in (
        f"{base}?foo=bar",
        f"{base}?foo=",
        f"{base}?x=1&x=2",
    ):
        resp = client.get(url)
        assert resp.status_code == 422, url
        assert resp.json()["error"]["code"] == "validation_error"
    # An unknown id together with a query parameter is the 422, since
    # parameters are checked first.
    resp = client.get("/v1/contents/cnt_ghost/evidence-coverage?foo=bar")
    assert resp.status_code == 422


def test_any_body_is_422_including_whitespace_and_malformed_json(client):
    create_actor(client)
    content = _create_content(client)
    url = f"/v1/contents/{content['id']}/evidence-coverage"
    for raw in (b"{}", b" ", b"\n\t ", b"{bad json", b"null"):
        resp = client.request("GET", url, content=raw)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["code"] == "validation_error"


def test_body_is_rejected_before_query_and_lookup(client):
    create_actor(client)
    url = "/v1/contents/cnt_ghost/evidence-coverage?foo=bar"
    resp = client.request("GET", url, content=b" ")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only and determinism guarantees ------------------------------------


def test_coverage_writes_no_resources_or_audit_events(client, db_session):
    create_actor(client)
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    _create_bundle(client, claim["id"])
    _attest(client, "claim", claim["id"])

    attestations_before = db_session.scalar(
        select(func.count()).select_from(Attestation)
    )
    audit_before = db_session.scalar(select(func.count()).select_from(AuditEvent))

    assert _coverage(client, content["id"]).status_code == 200
    assert _coverage(client, content["id"]).status_code == 200
    # Failures write nothing either.
    assert _coverage(client, "cnt_ghost").status_code == 404
    assert client.get(
        f"/v1/contents/{content['id']}/evidence-coverage?x=1"
    ).status_code == 422

    assert (
        db_session.scalar(select(func.count()).select_from(Attestation))
        == attestations_before
    )
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audit_before
    )


def test_repeated_reads_are_identical(client):
    create_actor(client)
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    bundle = _create_bundle(client, claim["id"])
    _attest(client, "evidence_bundle", bundle["id"])

    first = _coverage(client, content["id"])
    second = _coverage(client, content["id"])
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_counts_are_stable_across_app_restart(tmp_db_url, file_client):
    create_actor(file_client)
    content = _create_content(file_client)
    claim = _create_claim(file_client, content["id"])
    bundle = _create_bundle(file_client, claim["id"])
    _attest(file_client, "claim", claim["id"])
    _attest(file_client, "evidence_bundle", bundle["id"])

    before = _coverage(file_client, content["id"])
    assert before.status_code == 200
    expected = before.content

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        resp = _coverage(client, content["id"])
        assert resp.status_code == 200
        assert resp.content == expected
        body = resp.json()
        assert body["claim_count"] == 1
        assert body["bundle_count"] == 1
        assert body["attestation_count"] == 2
        assert body["qualified_signer_count"] == 1
        assert body["coverage_status"] == "covered"
