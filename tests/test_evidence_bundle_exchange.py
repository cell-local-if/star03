"""Tests for the read-only evidence-bundle interoperability snapshot.

Covers GET /v1/evidence-bundles/{evidence_bundle_id}/exchange: the success
body is exactly {"content", "claim", "evidence_bundle", "attestations"};
the first three are the bundle's owning content, its directly associated
claim, and the bundle's existing full public view; no lineage expansion and
no other content, claim, bundle, or attestation is included; attestations
list only evidence_bundle-targeted attestations of this exact bundle in
stable creation order, each identical to the existing attestation detail
view (raw signatures, claim payloads, content bytes, and evidence bytes
never appear); an empty attestation collection is an empty array; revoked
attestations are retained; any or repeated query parameter is 422
validation_error; an unknown bundle is 404 evidence_bundle_not_found; the
route is read-only (no resource or audit rows on success, empty results,
or failure); and snapshots are deterministic, including across an app
restart. All fixtures are deterministic and offline.
"""

from __future__ import annotations

import base64
import hashlib

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance.app import create_app
from provenance.config import Settings
from provenance.models import (
    Actor,
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


def _digest(name: str) -> str:
    return hashlib.sha256(f"exchange-{name}".encode()).hexdigest()


def _create_actor(client, actor_id):
    return create_actor(
        client,
        actor_id=actor_id,
        name=f"Actor {actor_id}",
        type="organization",
    )


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


def _create_claim(client, content_id, marker, actor_id="org-1",
                  claim_type="authorship"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": claim_type,
            "payload": {"marker": marker},
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


def _create_attestation(client, target_type, target_id, *, seed, signer):
    message = attestation_message_bytes(target_type, target_id, signer)
    payload = {
        "target_type": target_type,
        "target_id": target_id,
        "signer_actor_id": signer,
        "public_key": base64.b64encode(ed25519_public_key(seed)).decode("ascii"),
        "signature": base64.b64encode(ed25519_sign(seed, message)).decode("ascii"),
    }
    resp = client.post("/v1/attestations", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _revoke(client, attestation_id, revoker="org-1", reason="no longer relied upon"):
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


def _setup_bundle(client):
    _create_actor(client, "org-1")
    _create_actor(client, "org-2")
    content = _create_content(client, "main")
    claim = _create_claim(client, content["id"], "main-claim")
    bundle = _create_bundle(client, claim["id"], "main-evidence")
    return content, claim, bundle


# --- Success shape and public views -----------------------------------------


def test_exchange_empty_attestations_has_exact_shape_and_existing_views(client):
    content, claim, bundle = _setup_bundle(client)

    resp = client.get(f"/v1/evidence-bundles/{bundle['id']}/exchange")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"content", "claim", "evidence_bundle", "attestations"}

    # The first three members are exactly the existing full public views.
    assert body["content"] == client.get(
        f"/v1/contents/{content['id']}"
    ).json()
    assert body["claim"] == client.get(f"/v1/claims/{claim['id']}").json()
    assert body["evidence_bundle"] == client.get(
        f"/v1/evidence-bundles/{bundle['id']}"
    ).json()
    assert body["content"] == content
    assert body["claim"] == claim
    assert body["evidence_bundle"] == bundle
    assert body["attestations"] == []


def test_exchange_never_echoes_raw_material(client):
    _, _, bundle = _setup_bundle(client)

    body = client.get(f"/v1/evidence-bundles/{bundle['id']}/exchange").json()
    # Claim public view carries only the payload digest; bundle only its
    # digest/metadata; no field can carry raw bytes or a raw signature.
    assert set(body["claim"]) == {
        "id",
        "content_id",
        "actor_id",
        "claim_type",
        "payload_digest_algorithm",
        "payload_digest_hex",
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
    assert "payload" not in body["claim"]
    assert "data" not in body["evidence_bundle"]
    assert "evidence" not in body["evidence_bundle"]


# --- Attestation selection and ordering --------------------------------------


def test_exchange_lists_only_bundle_attestations_in_stable_creation_order(client):
    _, claim, bundle = _setup_bundle(client)

    # Two attestations of this exact bundle by distinct signers, created in a
    # deliberate order; both must appear in creation order.
    first = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_B, signer="org-2"
    )
    second = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    # An attestation of the bundle's claim must be excluded.
    _create_attestation(
        client, "claim", claim["id"], seed=SEED_A, signer="org-1"
    )
    # An attestation of a different bundle (other content/claim) is excluded.
    other_content = _create_content(client, "other")
    other_claim = _create_claim(client, other_content["id"], "other-claim")
    other_bundle = _create_bundle(client, other_claim["id"], "other-evidence")
    _create_attestation(
        client, "evidence_bundle", other_bundle["id"], seed=SEED_A,
        signer="org-1",
    )

    body = client.get(f"/v1/evidence-bundles/{bundle['id']}/exchange").json()
    attestations = body["attestations"]
    assert [a["id"] for a in attestations] == [first["id"], second["id"]]

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
    for item in attestations:
        assert set(item) == expected_keys
        # Each entry is identical to the existing attestation detail view.
        detail = client.get(f"/v1/attestations/{item['id']}").json()
        assert item == detail
        assert item["target_type"] == "evidence_bundle"
        assert item["target_id"] == bundle["id"]
        assert item["verified"] is True
        # The raw signature is never present, only its digest.
        assert "signature" not in item


def test_exchange_retains_revoked_attestations(client):
    _, _, bundle = _setup_bundle(client)
    attested = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    _revoke(client, attested["id"])

    body = client.get(f"/v1/evidence-bundles/{bundle['id']}/exchange").json()
    assert [a["id"] for a in body["attestations"]] == [attested["id"]]
    # The retained entry is still the unchanged attestation detail view: the
    # revocation neither removes the row nor alters any of its fields.
    assert body["attestations"][0] == client.get(
        f"/v1/attestations/{attested['id']}"
    ).json()


def test_exchange_does_not_expand_lineage_or_other_resources(client):
    content, claim, bundle = _setup_bundle(client)

    # A second bundle on the same direct claim must not appear.
    sibling_bundle = _create_bundle(client, claim["id"], "sibling-evidence")
    # A second claim on the same content (with its own bundle) must not appear.
    other_claim = _create_claim(client, content["id"], "other-claim",
                                claim_type="review")
    _create_bundle(client, other_claim["id"], "other-claim-evidence")

    # Lineage neighbors with their own claims and bundles must not appear.
    parent = _create_content(client, "parent")
    parent_claim = _create_claim(client, parent["id"], "parent-claim")
    _create_bundle(client, parent_claim["id"], "parent-evidence")
    relation = client.post(
        "/v1/content-relations",
        json={
            "content_id": content["id"],
            "parent_content_id": parent["id"],
            "relation_type": "derived_from",
        },
    )
    assert relation.status_code == 201, relation.text

    body = client.get(f"/v1/evidence-bundles/{bundle['id']}/exchange").json()
    assert set(body) == {"content", "claim", "evidence_bundle", "attestations"}
    assert body["content"]["id"] == content["id"]
    assert body["claim"]["id"] == claim["id"]
    assert body["evidence_bundle"]["id"] == bundle["id"]
    # Exactly one each: no arrays of other claims or bundles exist anywhere.
    assert not isinstance(body["claim"], list)
    assert not isinstance(body["evidence_bundle"], list)
    assert sibling_bundle["id"] != bundle["id"]


# --- Query validation and missing resources ----------------------------------


def test_exchange_unknown_bundle_is_404(client):
    resp = client.get("/v1/evidence-bundles/evb_doesnotexist/exchange")
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "evidence_bundle_not_found"


def test_exchange_rejects_any_query_parameter(client):
    _, _, bundle = _setup_bundle(client)
    base = f"/v1/evidence-bundles/{bundle['id']}/exchange"

    for url in (
        f"{base}?limit=10",
        f"{base}?cursor=abc",
        f"{base}?unknown=",
        f"{base}?a=1&b=2",
        # The same parameter repeated is also rejected rather than collapsed.
        f"{base}?a=1&a=2",
    ):
        resp = client.get(url)
        assert resp.status_code == 422, (url, resp.text)
        assert resp.json()["error"]["code"] == "validation_error"


def test_exchange_unknown_bundle_with_query_param_is_422(client):
    # Parameters are validated before any existence lookup, matching the
    # export route's boundary.
    resp = client.get(
        "/v1/evidence-bundles/evb_doesnotexist/exchange?anything=1"
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def _exchange_url(bundle) -> str:
    return f"/v1/evidence-bundles/{bundle['id']}/exchange"


# --- Read-only and determinism -----------------------------------------------


def test_exchange_writes_no_resources_or_audit_events(client, db_session):
    content, claim, bundle = _setup_bundle(client)
    attested = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    _revoke(client, attested["id"])
    # A second bundle on the same claim exercises the empty-attestations
    # path; it is created before the snapshot so its write is included.
    plain_bundle = _create_bundle(client, claim["id"], "plain")

    models = (
        Actor,
        Content,
        Claim,
        EvidenceBundle,
        Attestation,
        AttestationRevocation,
        AuditEvent,
    )

    def _counts():
        return {
            model: db_session.execute(
                select(func.count()).select_from(model)
            ).scalar_one()
            for model in models
        }

    before = _counts()

    # Success (with attestations), an empty-attestation bundle, a 422, and a
    # 404 all leave every table untouched.
    assert client.get(_exchange_url(bundle)).status_code == 200
    assert client.get(_exchange_url(plain_bundle)).status_code == 200
    assert client.get(f"{_exchange_url(bundle)}?x=1").status_code == 422
    assert client.get(
        "/v1/evidence-bundles/evb_missing/exchange"
    ).status_code == 404

    db_session.expire_all()
    assert _counts() == before


def test_exchange_is_deterministic_across_repeated_reads(client):
    _, _, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )

    first = client.get(f"/v1/evidence-bundles/{bundle['id']}/exchange")
    second = client.get(f"/v1/evidence-bundles/{bundle['id']}/exchange")
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


def test_exchange_snapshot_is_deterministic_across_restart(file_client, tmp_db_url):
    content, claim, bundle = _setup_bundle(file_client)
    attested = _create_attestation(
        file_client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    expected = file_client.get(
        f"/v1/evidence-bundles/{bundle['id']}/exchange"
    ).json()

    # A brand-new app/engine over the same file reproduces the snapshot
    # byte-for-byte (stable ids, UTC timestamps, and creation order).
    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        resp = client.get(f"/v1/evidence-bundles/{bundle['id']}/exchange")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body == expected
        assert body["content"]["id"] == content["id"]
        assert body["claim"]["id"] == claim["id"]
        assert body["evidence_bundle"]["id"] == bundle["id"]
        assert [a["id"] for a in body["attestations"]] == [attested["id"]]

        # The read after restart created nothing.
        resp2 = client.get(f"/v1/evidence-bundles/{bundle['id']}/exchange")
        assert resp2.json() == expected
