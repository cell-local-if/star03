"""Tests for the read-only evidence-bundle exchange package export.

Covers GET /v1/evidence-bundles/{evidence_bundle_id}/exchange/package: the
success body is exactly {"snapshot", "manifest"}; ``snapshot`` is byte-for-byte
the body of the existing ``.../exchange`` route (its four existing public
views, no lineage expansion, no other resources, no internal sequence numbers,
no raw signatures, claim payloads, content bytes, or evidence bytes) and
``manifest`` is byte-for-byte the body of the existing
``.../exchange/manifest`` route; the manifest digest is independently
recomputed from the snapshot member of the same response under the existing
canonical rules, so both halves are bound to one read state; any or repeated
query parameter is 422 validation_error before the bundle lookup; an unknown
bundle is 404 evidence_bundle_not_found; the route is read-only (no resource,
record, or audit rows on success, empty attestation lists, or failure); and
the package is deterministic across repeated reads and an app restart. All
fixtures are deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json
import re

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
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

SNAPSHOT_KEYS = {"content", "claim", "evidence_bundle", "attestations"}
MANIFEST_KEYS = {
    "manifest_version",
    "evidence_bundle_id",
    "digest_algorithm",
    "manifest_digest_hex",
}


def _package_url(bundle) -> str:
    return f"/v1/evidence-bundles/{bundle['id']}/exchange/package"


def _exchange_url(bundle) -> str:
    return f"/v1/evidence-bundles/{bundle['id']}/exchange"


def _manifest_url(bundle) -> str:
    return f"/v1/evidence-bundles/{bundle['id']}/exchange/manifest"


def _expected_manifest_digest(snapshot: dict) -> str:
    """Independently apply the manifest canonicalization to a parsed snapshot.

    Root members keep their snapshot order; every nested object sorts its
    members by Unicode code point; arrays keep element order; compact
    separators; non-ASCII emitted unescaped; UTF-8 encoded.
    """

    def normalize(value, sort_keys: bool):
        if isinstance(value, dict):
            items = sorted(value.items()) if sort_keys else value.items()
            return {key: normalize(member, True) for key, member in items}
        if isinstance(value, list):
            return [normalize(item, True) for item in value]
        return value

    canonical = json.dumps(
        normalize(snapshot, False),
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


# --- Response consistency with the existing routes ---------------------------


def test_package_success_has_exact_shape(client):
    _, _, bundle = _setup_bundle(client)

    resp = client.get(_package_url(bundle))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"snapshot", "manifest"}
    assert set(body["snapshot"]) == SNAPSHOT_KEYS
    assert set(body["manifest"]) == MANIFEST_KEYS


def test_package_snapshot_equals_existing_exchange_view(client):
    content, claim, bundle = _setup_bundle(client)
    first = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_B, signer="org-2"
    )
    second = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )

    package = client.get(_package_url(bundle)).json()
    exchange = client.get(_exchange_url(bundle)).json()

    # The snapshot member is byte-for-byte the existing exchange body.
    assert package["snapshot"] == exchange
    assert [a["id"] for a in package["snapshot"]["attestations"]] == [
        first["id"],
        second["id"],
    ]
    # The three single-resource members are the existing public detail views.
    assert package["snapshot"]["content"] == client.get(
        f"/v1/contents/{content['id']}"
    ).json()
    assert package["snapshot"]["claim"] == client.get(
        f"/v1/claims/{claim['id']}"
    ).json()
    assert package["snapshot"]["evidence_bundle"] == client.get(
        f"/v1/evidence-bundles/{bundle['id']}"
    ).json()


def test_package_manifest_equals_existing_manifest_view(client):
    _, _, bundle = _setup_bundle(client)

    package = client.get(_package_url(bundle)).json()
    manifest = client.get(_manifest_url(bundle)).json()

    # The manifest member is byte-for-byte the existing four-field manifest.
    assert package["manifest"] == manifest
    assert package["manifest"]["manifest_version"] == MANIFEST_VERSION
    assert package["manifest"]["digest_algorithm"] == "sha256"
    assert package["manifest"]["evidence_bundle_id"] == bundle["id"]
    assert _HEX64.fullmatch(package["manifest"]["manifest_digest_hex"])


def test_package_empty_attestations_snapshot_and_digest(client):
    _, _, bundle = _setup_bundle(client)

    body = client.get(_package_url(bundle)).json()
    assert body["snapshot"]["attestations"] == []
    # The manifest binds the empty attestation array, not an omitted one.
    assert body["manifest"]["manifest_digest_hex"] == _expected_manifest_digest(
        body["snapshot"]
    )


# --- Digest binding to the same response's snapshot --------------------------


def test_package_manifest_digest_binds_to_returned_snapshot(client):
    _, claim, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    # An attestation of the bundle's claim is outside the snapshot; the
    # digest must nonetheless match exactly the snapshot returned here.
    _create_attestation(
        client, "claim", claim["id"], seed=SEED_B, signer="org-2"
    )

    body = client.get(_package_url(bundle)).json()

    # Independently canonicalize the snapshot member of THIS response: the
    # digest in the same response must equal it.
    assert body["manifest"]["manifest_digest_hex"] == _expected_manifest_digest(
        body["snapshot"]
    )
    # The manifest references the snapshot's own bundle.
    assert (
        body["manifest"]["evidence_bundle_id"]
        == body["snapshot"]["evidence_bundle"]["id"]
    )

    # Mutating any returned snapshot member breaks the binding, proving the
    # digest commits to exactly the four served members.
    tampered = json.loads(json.dumps(body["snapshot"]))
    tampered["attestations"] = []
    assert body["manifest"]["manifest_digest_hex"] != _expected_manifest_digest(
        tampered
    )
    augmented = dict(body["snapshot"])
    augmented["extra"] = {"unrelated": True}
    assert body["manifest"]["manifest_digest_hex"] != _expected_manifest_digest(
        augmented
    )


def test_package_digest_reflects_attestation_array_order(client):
    _, _, bundle = _setup_bundle(client)
    first = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_B, signer="org-2"
    )
    second = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )

    body = client.get(_package_url(bundle)).json()
    assert [a["id"] for a in body["snapshot"]["attestations"]] == [
        first["id"],
        second["id"],
    ]
    assert body["manifest"]["manifest_digest_hex"] == _expected_manifest_digest(
        body["snapshot"]
    )
    reversed_order = dict(body["snapshot"])
    reversed_order["attestations"] = list(
        reversed(body["snapshot"]["attestations"])
    )
    assert body["manifest"]["manifest_digest_hex"] != _expected_manifest_digest(
        reversed_order
    )


def test_package_retains_revoked_attestation_in_snapshot_and_digest(client):
    _, _, bundle = _setup_bundle(client)
    attested = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    _revoke(client, attested["id"])

    body = client.get(_package_url(bundle)).json()
    assert [a["id"] for a in body["snapshot"]["attestations"]] == [
        attested["id"]
    ]
    assert body["manifest"]["manifest_digest_hex"] == _expected_manifest_digest(
        body["snapshot"]
    )


# --- Isolation: no lineage, other resources, internals, or raw material ------


def test_package_does_not_expand_lineage_or_other_resources(client):
    content, claim, bundle = _setup_bundle(client)

    # A second bundle on the same direct claim must not appear.
    _create_bundle(client, claim["id"], "sibling-evidence")
    # A second claim on the same content (with its own bundle) must not appear.
    other_claim = _create_claim(
        client, content["id"], "other-claim", claim_type="review"
    )
    _create_bundle(client, other_claim["id"], "other-claim-evidence")

    # Lineage neighbors with their own claims and bundles must not appear.
    parent = _create_content(client, "parent")
    parent_claim = _create_claim(client, parent["id"], "public-claim")
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
    snapshot = body["snapshot"]
    assert set(snapshot) == SNAPSHOT_KEYS
    assert snapshot["content"]["id"] == content["id"]
    assert snapshot["claim"]["id"] == claim["id"]
    assert snapshot["evidence_bundle"]["id"] == bundle["id"]
    assert not isinstance(snapshot["claim"], list)
    assert not isinstance(snapshot["evidence_bundle"], list)
    # Equality with the isolated exchange view is the full isolation check.
    assert snapshot == client.get(_exchange_url(bundle)).json()


def test_package_carries_no_raw_material_or_internal_fields(client):
    _, _, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )

    body = client.get(_package_url(bundle)).json()
    serialized = json.dumps(body)
    # No raw signature, claim payload, or content/evidence bytes anywhere.
    for forbidden in ('"signature"', '"payload"', '"data"', '"evidence"'):
        assert forbidden not in serialized
    # No internal sequence/primary-key bookkeeping fields leak into any view.
    claim_keys = set(body["snapshot"]["claim"])
    bundle_keys = set(body["snapshot"]["evidence_bundle"])
    attestation_keys = set(body["snapshot"]["attestations"][0])
    for internal in ("seq", "sequence", "rowid", "_id", "internal_id"):
        assert internal not in claim_keys
        assert internal not in bundle_keys
        assert internal not in attestation_keys
    assert "signature" not in attestation_keys


# --- Query validation and missing resources ----------------------------------


def test_package_unknown_bundle_is_404(client):
    resp = client.get(
        "/v1/evidence-bundles/evb_doesnotexist/exchange/package"
    )
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
    # A second bundle on the same claim exercises the empty-attestations
    # path; it is created before the reads so its write is included.
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
    assert content["id"]  # fixtures remain the only persisted state


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
    # byte-for-byte (stable ids, UTC timestamps, and creation order).
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
