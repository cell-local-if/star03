"""Tests for the read-only evidence-bundle exchange package export.

Covers GET /v1/evidence-bundles/{evidence_bundle_id}/exchange/package:
the success body is exactly {"snapshot", "manifest"}; the snapshot is
byte-for-byte the bundle's existing GET /exchange public view and the
manifest is byte-for-byte the existing GET /exchange/manifest public
view; the manifest digest is independently reproducible from the
snapshot member served in the *same* response under the stated canonical
rules (root members and arrays keep snapshot order, nested object keys
sort by Unicode code point, compact separators, unescaped non-ASCII,
UTF-8), so the pair is self-binding; no lineage traversal is performed
and no other resource, internal sequence, raw signature, claim payload,
content bytes, or evidence bytes appear; any or repeated query parameter
is 422 validation_error before the bundle lookup; an unknown bundle is
404 evidence_bundle_not_found; the route is read-only (no resource or
audit rows on success, on an empty attestation list, or on failure); and
the package is deterministic across repeated reads and an app restart.
All fixtures are deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json

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
from tests.helpers import SEED_A, SEED_B
from tests.test_evidence_bundle_exchange import (
    _create_attestation,
    _create_bundle,
    _create_claim,
    _create_content,
    _revoke,
    _setup_bundle,
)

MANIFEST_VERSION = "provenance-exchange-manifest-v1"


def _package_url(bundle) -> str:
    return f"/v1/evidence-bundles/{bundle['id']}/exchange/package"


def _exchange_url(bundle) -> str:
    return f"/v1/evidence-bundles/{bundle['id']}/exchange"


def _manifest_url(bundle) -> str:
    return f"/v1/evidence-bundles/{bundle['id']}/exchange/manifest"


def _canonical_manifest_bytes(snapshot: dict, *, ascii_escape: bool = False) -> bytes:
    """Independently apply the manifest canonicalization to a parsed snapshot.

    Root members keep their snapshot order; every nested object sorts its
    members by Unicode code point; arrays keep element order; compact
    separators; non-ASCII emitted unescaped (or escaped when checking the
    escaping requirement); UTF-8 encoded.
    """

    def normalize(value, sort_keys: bool):
        if isinstance(value, dict):
            items = sorted(value.items()) if sort_keys else value.items()
            return {key: normalize(member, True) for key, member in items}
        if isinstance(value, list):
            return [normalize(item, True) for item in value]
        return value

    return json.dumps(
        normalize(snapshot, False),
        separators=(",", ":"),
        ensure_ascii=ascii_escape,
        allow_nan=False,
    ).encode("utf-8")


def _expected_manifest_digest(snapshot: dict) -> str:
    return hashlib.sha256(_canonical_manifest_bytes(snapshot)).hexdigest()


# --- Success shape and consistency with the existing views -------------------


def test_package_success_has_exact_two_member_shape(client):
    _, _, bundle = _setup_bundle(client)

    resp = client.get(_package_url(bundle))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"snapshot", "manifest"}
    assert set(body["snapshot"]) == {
        "content",
        "claim",
        "evidence_bundle",
        "attestations",
    }
    assert set(body["manifest"]) == {
        "manifest_version",
        "evidence_bundle_id",
        "digest_algorithm",
        "manifest_digest_hex",
    }


def test_package_snapshot_equals_existing_exchange_view(client):
    content, claim, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )

    package = client.get(_package_url(bundle)).json()

    # The snapshot member is exactly what GET /exchange serves, and each
    # nested view matches the existing detail endpoints.
    assert package["snapshot"] == client.get(_exchange_url(bundle)).json()
    assert package["snapshot"]["content"] == client.get(
        f"/v1/contents/{content['id']}"
    ).json()
    assert package["snapshot"]["claim"] == client.get(
        f"/v1/claims/{claim['id']}"
    ).json()
    assert package["snapshot"]["evidence_bundle"] == client.get(
        f"/v1/evidence-bundles/{bundle['id']}"
    ).json()
    for item in package["snapshot"]["attestations"]:
        assert item == client.get(
            f"/v1/attestations/{item['id']}"
        ).json()


def test_package_manifest_equals_existing_manifest_view(client):
    _, _, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )

    package = client.get(_package_url(bundle)).json()
    assert package["manifest"] == client.get(_manifest_url(bundle)).json()
    assert package["manifest"]["manifest_version"] == MANIFEST_VERSION
    assert package["manifest"]["digest_algorithm"] == "sha256"
    assert package["manifest"]["evidence_bundle_id"] == bundle["id"]


def test_package_empty_attestations_is_empty_array(client):
    _, _, bundle = _setup_bundle(client)

    package = client.get(_package_url(bundle)).json()
    assert package["snapshot"]["attestations"] == []
    # The manifest still binds the empty-attestations snapshot.
    assert package["manifest"]["manifest_digest_hex"] == _expected_manifest_digest(
        package["snapshot"]
    )


# --- Digest binding within one response ---------------------------------------


def test_package_manifest_binds_the_snapshot_in_the_same_response(client):
    _, claim, bundle = _setup_bundle(client)
    first = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_B, signer="org-2"
    )
    second = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    # Excluded attestations must not influence the bound digest.
    _create_attestation(client, "claim", claim["id"], seed=SEED_A, signer="org-1")

    body = client.get(_package_url(bundle)).json()
    snapshot = body["snapshot"]
    assert [a["id"] for a in snapshot["attestations"]] == [
        first["id"],
        second["id"],
    ]

    # The digest is independently reproducible from only the snapshot
    # member of this same response (no service helpers involved).
    assert body["manifest"]["manifest_digest_hex"] == _expected_manifest_digest(
        snapshot
    )

    # Tampering with the served snapshot in any way breaks the binding:
    # reordered attestation array, dropped member, or added member.
    reordered = dict(snapshot)
    reordered["attestations"] = list(reversed(snapshot["attestations"]))
    assert (
        _expected_manifest_digest(reordered)
        != body["manifest"]["manifest_digest_hex"]
    )
    dropped = dict(snapshot)
    dropped["attestations"] = []
    assert (
        _expected_manifest_digest(dropped)
        != body["manifest"]["manifest_digest_hex"]
    )
    augmented = dict(snapshot)
    augmented["extra"] = {"unrelated": True}
    assert (
        _expected_manifest_digest(augmented)
        != body["manifest"]["manifest_digest_hex"]
    )


def test_package_binding_covers_unescaped_non_ascii_utf8(client):
    content, _, _ = _setup_bundle(client)
    claim = _create_claim(client, content["id"], "unicode-claim")
    resp = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim["id"],
            "evidence_type": "unicode_capture",
            "digest_algorithm": "sha256",
            "digest_hex": hashlib.sha256(b"unicode-evidence-package").hexdigest(),
            "media_type": "image/jpeg",
            "metadata": {"label": "日本語の証拠", "é": 1, "a": {"中": True}},
        },
    )
    assert resp.status_code == 201, resp.text
    unicode_bundle = resp.json()

    body = client.get(_package_url(unicode_bundle)).json()
    assert body["manifest"]["manifest_digest_hex"] == _expected_manifest_digest(
        body["snapshot"]
    )
    # ASCII-escaping the same canonical object must not match: the digest
    # hashes unescaped non-ASCII as raw UTF-8 bytes.
    escaped = hashlib.sha256(
        _canonical_manifest_bytes(body["snapshot"], ascii_escape=True)
    ).hexdigest()
    assert escaped != body["manifest"]["manifest_digest_hex"]


def test_package_retains_revoked_attestation_in_binding(client):
    _, _, bundle = _setup_bundle(client)
    attested = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    _revoke(client, attested["id"])

    body = client.get(_package_url(bundle)).json()
    assert [a["id"] for a in body["snapshot"]["attestations"]] == [attested["id"]]
    assert body["manifest"]["manifest_digest_hex"] == _expected_manifest_digest(
        body["snapshot"]
    )


# --- Isolation: no lineage, no other resources, no raw material ---------------


def test_package_does_not_expand_lineage_or_other_resources(client):
    content, claim, bundle = _setup_bundle(client)
    sibling = _create_bundle(client, claim["id"], "sibling-evidence")
    other_claim = _create_claim(
        client, content["id"], "other-claim", claim_type="review"
    )
    _create_bundle(client, other_claim["id"], "other-claim-evidence")

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

    body = client.get(_package_url(bundle)).json()
    assert set(body) == {"snapshot", "manifest"}
    snapshot = body["snapshot"]
    assert set(snapshot) == {"content", "claim", "evidence_bundle", "attestations"}
    assert snapshot["content"]["id"] == content["id"]
    assert snapshot["claim"]["id"] == claim["id"]
    assert snapshot["evidence_bundle"]["id"] == bundle["id"]
    assert not isinstance(snapshot["claim"], list)
    assert not isinstance(snapshot["evidence_bundle"], list)
    assert sibling["id"] != bundle["id"]

    # Another bundle's package is independently bound and different.
    other = client.get(_package_url(sibling)).json()
    assert other["snapshot"]["evidence_bundle"]["id"] == sibling["id"]
    assert (
        other["manifest"]["manifest_digest_hex"]
        != body["manifest"]["manifest_digest_hex"]
    )


def test_package_carries_no_internal_or_raw_material_fields(client):
    _, _, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )

    body = client.get(_package_url(bundle)).json()
    assert set(body["snapshot"]["claim"]) == {
        "id",
        "content_id",
        "actor_id",
        "claim_type",
        "payload_digest_algorithm",
        "payload_digest_hex",
        "created_at",
    }
    assert set(body["snapshot"]["evidence_bundle"]) == {
        "id",
        "claim_id",
        "evidence_type",
        "digest_algorithm",
        "digest_hex",
        "media_type",
        "metadata",
        "created_at",
    }
    for item in body["snapshot"]["attestations"]:
        assert "signature" not in item
        assert "public_key" in item  # public material only, Base64
    serialized = json.dumps(body)
    # No raw signatures, claim payloads, content bytes, evidence bytes, or
    # internal sequence numbers anywhere in the package.
    for forbidden in (
        '"signature"',
        '"payload"',
        '"data"',
        '"evidence"',
        '"_sa_instance_state"',
        '"sequence"',
        '"seq"',
        '"rowid"',
    ):
        assert forbidden not in serialized


# --- Query validation and missing resources ----------------------------------


def test_package_unknown_bundle_is_404(client):
    resp = client.get("/v1/evidence-bundles/evb_doesnotexist/exchange/package")
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "evidence_bundle_not_found"


def test_package_rejects_any_query_parameter(client):
    _, _, bundle = _setup_bundle(client)
    base = _package_url(bundle)

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


def test_package_unknown_bundle_with_query_param_is_422(client):
    # Parameters are validated before any existence lookup, matching the
    # exchange and manifest routes' boundary.
    resp = client.get(
        "/v1/evidence-bundles/evb_doesnotexist/exchange/package?anything=1"
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only and determinism -----------------------------------------------


def test_package_writes_no_resources_or_audit_events(client, db_session):
    content, claim, bundle = _setup_bundle(client)
    attested = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    _revoke(client, attested["id"])
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
    assert client.get(_package_url(bundle)).status_code == 200
    assert client.get(_package_url(plain_bundle)).status_code == 200
    assert client.get(f"{_package_url(bundle)}?x=1").status_code == 422
    assert client.get(
        "/v1/evidence-bundles/evb_missing/exchange/package"
    ).status_code == 404

    db_session.expire_all()
    assert _counts() == before
    assert content["id"]  # fixtures remain readable and unchanged


def test_package_is_deterministic_across_repeated_reads(client):
    _, _, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )

    first = client.get(_package_url(bundle))
    second = client.get(_package_url(bundle))
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


def test_package_is_deterministic_across_restart(file_client, tmp_db_url):
    content, claim, bundle = _setup_bundle(file_client)
    attested = _create_attestation(
        file_client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    expected = file_client.get(_package_url(bundle))
    assert expected.status_code == 200, expected.text
    expected_body = expected.json()

    # A brand-new app/engine over the same file reproduces the package
    # byte-for-byte (stable ids, UTC timestamps, and creation order), and
    # the manifest still binds the snapshot after restart.
    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        resp = client.get(_package_url(bundle))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body == expected_body
        assert body["snapshot"]["content"]["id"] == content["id"]
        assert body["snapshot"]["claim"]["id"] == claim["id"]
        assert body["snapshot"]["evidence_bundle"]["id"] == bundle["id"]
        assert [a["id"] for a in body["snapshot"]["attestations"]] == [
            attested["id"]
        ]
        assert body["manifest"]["manifest_digest_hex"] == _expected_manifest_digest(
            body["snapshot"]
        )

        # The read after restart created nothing and stays stable.
        assert client.get(_package_url(bundle)).json() == expected_body
