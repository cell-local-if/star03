"""Tests for the read-only evidence-bundle interoperability snapshot.

Covers GET /v1/evidence-bundles/{evidence_bundle_id}/exchange: the success
body is exactly {"content", "claim", "evidence_bundle", "attestations"};
content is the single content the bundle's directly associated claim
asserts (no lineage traversal, no foreign contents/claims/bundles); claim
and evidence_bundle are the existing full public views; attestations list
only existing attestations targeting this exact evidence bundle in stable
creation order, each identical to its attestation detail, including revoked
ones, with no raw signature/payload/bytes; an empty set serializes as [];
any query parameter (including a repeated one) is 422 validation_error; an
unknown bundle is 404 evidence_bundle_not_found; and the route writes no
resource or audit rows on success, empty results, or failure. All fixtures
are deterministic and offline.
"""

from __future__ import annotations

import base64
import hashlib
import json

from sqlalchemy import func, select

from provenance.models import (
    Attestation,
    AttestationRevocation,
    AuditEvent,
    Claim,
    Content,
    EvidenceBundle,
)
from provenance.signing import attestation_message_bytes
from tests.helpers import (
    create_actor,
    ed25519_public_key,
    ed25519_sign,
    SEED_A,
    SEED_B,
)

REVOKE_REASON = "key compromised during incident"


def _digest(name: str) -> str:
    return hashlib.sha256(f"exchange-{name}".encode()).hexdigest()


def _create_content(client, name, actor_id="org-1"):
    resp = client.post(
        "/v1/contents",
        json={
            "digest_algorithm": "sha256",
            "digest_hex": _digest(f"content-{name}"),
            "media_type": "image/png",
            "title": name,
            "actor_id": actor_id,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, marker, actor_id="org-1", claim_type="authorship"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": claim_type,
            "payload": {"marker": marker, "secret": marker + "-payload"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_bundle(client, claim_id, name, evidence_type="raw_capture"):
    resp = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim_id,
            "evidence_type": evidence_type,
            "digest_algorithm": "sha256",
            "digest_hex": _digest(f"evidence-{name}"),
            "media_type": "image/jpeg",
            "metadata": {"name": name},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _attest(client, target_type, target_id, *, seed=SEED_A, signer_actor_id="org-1"):
    signature = ed25519_sign(
        seed,
        attestation_message_bytes(target_type, target_id, signer_actor_id),
    )
    resp = client.post(
        "/v1/attestations",
        json={
            "target_type": target_type,
            "target_id": target_id,
            "signer_actor_id": signer_actor_id,
            "public_key": base64.b64encode(ed25519_public_key(seed)).decode("ascii"),
            "signature": base64.b64encode(signature).decode("ascii"),
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json(), signature


def _revoke(client, attestation_id, *, revoker_actor_id="org-1", reason=REVOKE_REASON):
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


def _setup_bundle(client):
    create_actor(client)
    content = _create_content(client, "main")
    claim = _create_claim(client, content["id"], "main-claim")
    bundle = _create_bundle(client, claim["id"], "main-evidence")
    return content, claim, bundle


# --- Success shape and existing public views --------------------------------


def test_exchange_without_attestations_has_exact_shape_and_views(client):
    content, claim, bundle = _setup_bundle(client)

    resp = client.get(f"/v1/evidence-bundles/{bundle['id']}/exchange")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"content", "claim", "evidence_bundle", "attestations"}
    assert body["attestations"] == []

    # The first three members are the existing full public views, fetched
    # independently through their own detail routes.
    assert body["content"] == client.get(f"/v1/contents/{content['id']}").json()
    assert body["claim"] == client.get(f"/v1/claims/{claim['id']}").json()
    assert (
        body["evidence_bundle"]
        == client.get(f"/v1/evidence-bundles/{bundle['id']}").json()
    )

    # Snapshot reads are deterministic and idempotent.
    assert (
        client.get(f"/v1/evidence-bundles/{bundle['id']}/exchange").json() == body
    )


def test_exchange_attestations_target_only_this_bundle_in_stable_order(client):
    content, claim, bundle = _setup_bundle(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")

    # Two attestations of the target bundle by distinct signers/keys.
    att_one, _ = _attest(client, "evidence_bundle", bundle["id"])
    att_two, _ = _attest(
        client,
        "evidence_bundle",
        bundle["id"],
        seed=SEED_B,
        signer_actor_id="org-2",
    )

    # Excluded: an attestation of the bundle's own claim ...
    claim_att, _ = _attest(client, "claim", claim["id"])
    # ... a second bundle under the same claim with its own attestation ...
    other_bundle = _create_bundle(client, claim["id"], "other-evidence")
    other_bundle_att, _ = _attest(
        client, "evidence_bundle", other_bundle["id"], seed=SEED_B,
        signer_actor_id="org-2",
    )
    # ... and an unrelated content/claim/bundle graph.
    other_content = _create_content(client, "other", actor_id="org-2")
    other_claim = _create_claim(
        client, other_content["id"], "foreign", actor_id="org-2"
    )
    foreign_bundle = _create_bundle(client, other_claim["id"], "foreign")
    _attest(
        client, "evidence_bundle", foreign_bundle["id"], seed=SEED_B,
        signer_actor_id="org-2",
    )

    resp = client.get(f"/v1/evidence-bundles/{bundle['id']}/exchange")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["content"]["id"] == content["id"]
    assert body["claim"]["id"] == claim["id"]
    assert body["evidence_bundle"]["id"] == bundle["id"]

    attestations = body["attestations"]
    assert [a["id"] for a in attestations] == [att_one["id"], att_two["id"]]

    expected_keys = {
        "id",
        "target_type",
        "target_id",
        "signer_actor_id",
        "public_key",
        "signature_digest_algorithm",
        "signature_digest_hex",
        "verified",
        "created_at",
    }
    for index, att_id in enumerate((att_one["id"], att_two["id"])):
        item = attestations[index]
        # Each item is identical to the existing attestation detail and
        # carries no extra (e.g. revocation) fields.
        assert set(item) == expected_keys
        assert item == client.get(f"/v1/attestations/{att_id}").json()
        assert item["target_type"] == "evidence_bundle"
        assert item["target_id"] == bundle["id"]

    # The excluded proofs are genuinely absent from the serialized body.
    serialized = json.dumps(body)
    for excluded_id in (claim_att["id"], other_bundle_att["id"]):
        assert excluded_id not in serialized


def test_exchange_retains_revoked_attestations_for_audit_history(client):
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    _content, _claim, bundle = _setup_bundle(client)
    att_active, _ = _attest(client, "evidence_bundle", bundle["id"])
    att_revoked, _ = _attest(
        client,
        "evidence_bundle",
        bundle["id"],
        seed=SEED_B,
        signer_actor_id="org-2",
    )
    # org-1 (need not be the signer) records the revocation of the second.
    _revoke(client, att_revoked["id"])

    body = client.get(
        f"/v1/evidence-bundles/{bundle['id']}/exchange"
    ).json()
    # The revoked proof is retained, in its original creation position, and
    # its fields are exactly the attestation detail (no revocation marker).
    assert [a["id"] for a in body["attestations"]] == [
        att_active["id"],
        att_revoked["id"],
    ]
    revoked_item = body["attestations"][1]
    assert revoked_item == client.get(
        f"/v1/attestations/{att_revoked['id']}"
    ).json()
    assert "revoked" not in revoked_item
    assert "revocation" not in json.dumps(revoked_item)


def test_exchange_does_not_traverse_lineage(client):
    create_actor(client)
    parent = _create_content(client, "parent")
    child = _create_content(client, "child")
    relation = client.post(
        "/v1/content-relations",
        json={
            "content_id": child["id"],
            "parent_content_id": parent["id"],
            "relation_type": "derived_from",
        },
    )
    assert relation.status_code == 201, relation.text

    parent_claim = _create_claim(client, parent["id"], "parent-claim")
    parent_bundle = _create_bundle(client, parent_claim["id"], "parent-evidence")
    _attest(client, "evidence_bundle", parent_bundle["id"])

    child_claim = _create_claim(client, child["id"], "child-claim")
    child_bundle = _create_bundle(client, child_claim["id"], "child-evidence")

    body = client.get(
        f"/v1/evidence-bundles/{child_bundle['id']}/exchange"
    ).json()
    assert body["content"]["id"] == child["id"]
    assert body["claim"]["id"] == child_claim["id"]
    assert body["evidence_bundle"]["id"] == child_bundle["id"]
    assert body["attestations"] == []


def test_exchange_never_echoes_raw_signature_payload_or_bytes(client):
    _content, claim, bundle = _setup_bundle(client)
    _att, raw_signature = _attest(client, "evidence_bundle", bundle["id"])

    body = client.get(f"/v1/evidence-bundles/{bundle['id']}/exchange").json()
    serialized = json.dumps(body)

    # The raw 64-byte signature, in either encoding, is never present.
    assert "signature" not in body["attestations"][0]
    assert base64.b64encode(raw_signature).decode("ascii") not in serialized
    # The claim payload is committed to by digest only.
    assert "payload" not in body["claim"]
    assert "main-claim-payload" not in serialized
    # Content/evidence are digest-referenced; no byte-carrying fields exist.
    assert set(body["content"]) == {
        "id",
        "digest_algorithm",
        "digest_hex",
        "media_type",
        "title",
        "actor_id",
        "created_at",
    }
    assert set(body["evidence_bundle"]) == {
        "id",
        "claim_id",
        "evidence_type",
        "digest_algorithm",
        "digest_hex",
        "media_type",
        "metadata",
        "created_at",
    }


# --- Error boundary ----------------------------------------------------------


def test_exchange_unknown_bundle_is_404(client):
    resp = client.get("/v1/evidence-bundles/evb_doesnotexist/exchange")
    assert resp.status_code == 404, resp.text
    error = resp.json()["error"]
    assert error["code"] == "evidence_bundle_not_found"
    assert error["details"]["evidence_bundle_id"] == "evb_doesnotexist"


def test_exchange_rejects_any_query_parameter(client):
    _content, _claim, bundle = _setup_bundle(client)
    base = f"/v1/evidence-bundles/{bundle['id']}/exchange"

    for url in (
        f"{base}?limit=10",
        f"{base}?cursor=abc",
        f"{base}?unknown=",
        f"{base}?a=1&b=2",
        # The same parameter repeated must be rejected, not collapsed.
        f"{base}?a=1&a=2",
    ):
        resp = client.get(url)
        assert resp.status_code == 422, (url, resp.text)
        error = resp.json()["error"]
        assert error["code"] == "validation_error"
        issue = error["details"]["issues"][0]
        assert issue["loc"][0] == "query"
        assert issue["type"] == "value_error.unknown"


# --- Read-only guarantees ----------------------------------------------------


def test_exchange_writes_no_resources_or_audit_events(client, db_session):
    content, claim, bundle = _setup_bundle(client)
    _attest(client, "evidence_bundle", bundle["id"])
    # A second bundle on the same claim with no attestations (empty case).
    lonely_bundle = _create_bundle(client, claim["id"], "lonely-evidence")

    models = (
        Content,
        Claim,
        EvidenceBundle,
        Attestation,
        AttestationRevocation,
        AuditEvent,
    )
    counts_before = {
        model: db_session.execute(select(func.count()).select_from(model)).scalar_one()
        for model in models
    }
    audit_before = [
        (row.event_type, row.resource_id)
        for row in db_session.execute(select(AuditEvent)).scalars().all()
    ]

    # Success with an attestation, an empty-attestation bundle, a 422, and a
    # 404 are all strictly read-only.
    assert (
        client.get(f"/v1/evidence-bundles/{bundle['id']}/exchange").status_code
        == 200
    )
    assert (
        client.get(
            f"/v1/evidence-bundles/{lonely_bundle['id']}/exchange"
        ).status_code
        == 200
    )
    assert (
        client.get(
            f"/v1/evidence-bundles/{bundle['id']}/exchange?x=1"
        ).status_code
        == 422
    )
    assert (
        client.get("/v1/evidence-bundles/evb_missing/exchange").status_code == 404
    )

    for model, before in counts_before.items():
        after = db_session.execute(
            select(func.count()).select_from(model)
        ).scalar_one()
        assert after == before, model.__name__
    audit_after = [
        (row.event_type, row.resource_id)
        for row in db_session.execute(select(AuditEvent)).scalars().all()
    ]
    assert audit_after == audit_before
