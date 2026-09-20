"""Tests for the read-only evidence-bundle exchange integrity manifest.

Covers GET /v1/evidence-bundles/{evidence_bundle_id}/exchange/manifest: the
success body is exactly
``{"manifest_version", "evidence_bundle_id", "digest_algorithm",
"manifest_digest_hex"}`` with the version fixed at
``provenance-exchange-manifest-v1`` and the algorithm fixed at ``sha256``;
the digest is the SHA-256 of the canonical JSON object of exactly the
exchange snapshot's ``content``, ``claim``, ``evidence_bundle``, and
``attestations`` members (object keys sorted by Unicode code point, array
order preserved, compact separators, unescaped non-ASCII, UTF-8); no other
resource or raw material enters the digest; revoked attestations remain in
it; any or repeated query parameter is 422 validation_error (also for an
unknown bundle); an unknown bundle is 404 evidence_bundle_not_found; the
route is read-only (no resource or audit rows on success or failure); and
manifests are deterministic for the same persisted state, including across
an app restart. All fixtures are deterministic and offline.
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
from provenance.schemas import EXCHANGE_MANIFEST_VERSION
from tests.helpers import SEED_A, SEED_B
from tests.test_evidence_bundle_exchange import (
    _create_attestation,
    _create_bundle,
    _create_claim,
    _create_content,
    _digest,
    _revoke,
    _setup_bundle,
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _manifest_url(bundle) -> str:
    return f"/v1/evidence-bundles/{bundle['id']}/exchange/manifest"


def _independent_canonical_hex(snapshot: dict) -> str:
    """Recompute the manifest digest with stdlib json, independently.

    Mirrors the contract rather than importing the production serializer:
    sorted object keys, array order untouched, compact separators,
    unescaped non-ASCII, UTF-8.
    """
    raw = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# --- Success shape and fixed values -----------------------------------------


def test_manifest_has_exact_shape_and_fixed_values(client):
    _, _, bundle = _setup_bundle(client)

    resp = client.get(_manifest_url(bundle))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {
        "manifest_version",
        "evidence_bundle_id",
        "digest_algorithm",
        "manifest_digest_hex",
    }
    assert body["manifest_version"] == EXCHANGE_MANIFEST_VERSION
    assert body["manifest_version"] == "provenance-exchange-manifest-v1"
    assert body["evidence_bundle_id"] == bundle["id"]
    assert body["digest_algorithm"] == "sha256"
    assert _HEX64.fullmatch(body["manifest_digest_hex"])


def test_manifest_digest_is_sha256_of_canonical_snapshot_with_attestations(client):
    _, _, bundle = _setup_bundle(client)
    first = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_B, signer="org-2"
    )
    second = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )

    snapshot = client.get(
        f"/v1/evidence-bundles/{bundle['id']}/exchange"
    ).json()
    manifest = client.get(_manifest_url(bundle)).json()

    assert [a["id"] for a in snapshot["attestations"]] == [
        first["id"],
        second["id"],
    ]
    assert manifest["manifest_digest_hex"] == _independent_canonical_hex(snapshot)


def test_manifest_digest_for_empty_attestations(client):
    _, _, bundle = _setup_bundle(client)

    snapshot = client.get(
        f"/v1/evidence-bundles/{bundle['id']}/exchange"
    ).json()
    assert snapshot["attestations"] == []
    manifest = client.get(_manifest_url(bundle)).json()
    assert manifest["manifest_digest_hex"] == _independent_canonical_hex(snapshot)


# --- Canonicalization details ------------------------------------------------


def test_manifest_canonical_form_sorts_keys_and_emits_utf8_unescaped(client):
    _setup_bundle(client)
    # A fresh claim/bundle whose metadata carries deliberately unsorted keys,
    # a non-ASCII value, and a nested array (whose order must be preserved).
    content = _create_content(client, "unicode")
    claim = _create_claim(client, content["id"], "unicode-claim")
    metadata = {"zulu": 1, "étoile": ["Ω", "b"], "alpha": "日本語"}
    resp = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim["id"],
            "evidence_type": "raw_capture",
            "digest_algorithm": "sha256",
            "digest_hex": _digest("evidence-unicode"),
            "media_type": "image/jpeg",
            "metadata": metadata,
        },
    )
    assert resp.status_code == 201, resp.text
    bundle = resp.json()
    assert bundle["metadata"] == metadata

    snapshot = client.get(
        f"/v1/evidence-bundles/{bundle['id']}/exchange"
    ).json()
    manifest = client.get(_manifest_url(bundle)).json()

    canonical = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert manifest["manifest_digest_hex"] == hashlib.sha256(canonical).hexdigest()

    # Non-ASCII is emitted unescaped: the raw UTF-8 bytes of the value appear.
    assert "日本語".encode("utf-8") in canonical
    assert b"\\u" not in canonical
    # Object keys are ordered by Unicode code point: "alpha" (U+0061) precedes
    # "zulu" (U+007A) which precedes "étoile" (U+00E9), regardless of the
    # submitted insertion order.
    assert canonical.index(b'"alpha"') < canonical.index(b'"zulu"')
    assert canonical.index(b'"zulu"') < canonical.index("étoile".encode("utf-8"))
    # The nested array keeps its element order.
    assert canonical.index("Ω".encode("utf-8")) < canonical.index(b'"b"')

    # Key sorting is load-bearing: an unsorted serialization digests
    # differently even though it carries the same values.
    unsorted = json.dumps(
        snapshot,
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(unsorted).hexdigest() != manifest["manifest_digest_hex"]


def test_manifest_attestation_array_order_is_snapshot_creation_order(client):
    _, _, bundle = _setup_bundle(client)
    first = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    second = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_B, signer="org-2"
    )

    snapshot = client.get(
        f"/v1/evidence-bundles/{bundle['id']}/exchange"
    ).json()
    assert [a["id"] for a in snapshot["attestations"]] == [
        first["id"],
        second["id"],
    ]
    manifest = client.get(_manifest_url(bundle)).json()

    swapped = dict(snapshot)
    swapped["attestations"] = list(reversed(swapped["attestations"]))
    # Reversing the array changes the digest: order is preserved, not sorted.
    assert _independent_canonical_hex(swapped) != manifest["manifest_digest_hex"]
    assert _independent_canonical_hex(snapshot) == manifest["manifest_digest_hex"]


def test_manifest_digest_input_is_only_the_four_snapshot_members(client):
    content, claim, bundle = _setup_bundle(client)
    attestation = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )

    manifest = client.get(_manifest_url(bundle)).json()
    # The digest equals that of the four-member object assembled explicitly
    # from the existing public views; adding an extra top-level member
    # changes it.
    explicit = {
        "content": client.get(f"/v1/contents/{content['id']}").json(),
        "claim": client.get(f"/v1/claims/{claim['id']}").json(),
        "evidence_bundle": client.get(
            f"/v1/evidence-bundles/{bundle['id']}"
        ).json(),
        "attestations": [
            client.get(f"/v1/attestations/{attestation['id']}").json()
        ],
    }
    assert manifest["manifest_digest_hex"] == _independent_canonical_hex(explicit)
    with_extra = dict(explicit)
    with_extra["audit_events"] = []
    assert _independent_canonical_hex(with_extra) != manifest["manifest_digest_hex"]


def test_manifest_never_includes_raw_material(client):
    _, _, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )

    snapshot = client.get(
        f"/v1/evidence-bundles/{bundle['id']}/exchange"
    ).json()
    manifest = client.get(_manifest_url(bundle)).json()

    # The manifest only exposes the digest; the digested object is the public
    # snapshot, whose views carry no raw payload, bytes, or signature.
    assert "payload" not in snapshot["claim"]
    for item in snapshot["attestations"]:
        assert "signature" not in item
    assert set(manifest) == {
        "manifest_version",
        "evidence_bundle_id",
        "digest_algorithm",
        "manifest_digest_hex",
    }


# --- State binding and revocation --------------------------------------------


def test_manifest_tracks_persisted_state_change(client):
    _, _, bundle = _setup_bundle(client)
    before = client.get(_manifest_url(bundle)).json()

    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    after = client.get(_manifest_url(bundle)).json()
    assert after["manifest_digest_hex"] != before["manifest_digest_hex"]
    assert after["manifest_version"] == before["manifest_version"]


def test_manifest_revoked_attestation_remains_in_digest(client):
    _, _, bundle = _setup_bundle(client)
    attested = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    digest_with = client.get(_manifest_url(bundle)).json()["manifest_digest_hex"]
    _revoke(client, attested["id"])

    snapshot = client.get(
        f"/v1/evidence-bundles/{bundle['id']}/exchange"
    ).json()
    # The revocation neither removes the attestation from the snapshot nor
    # alters its fields, so the manifest digest is unchanged by revocation.
    assert [a["id"] for a in snapshot["attestations"]] == [attested["id"]]
    manifest = client.get(_manifest_url(bundle)).json()
    assert manifest["manifest_digest_hex"] == digest_with
    assert manifest["manifest_digest_hex"] == _independent_canonical_hex(snapshot)


def test_manifest_ignores_other_resources(client):
    content, claim, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    baseline = client.get(_manifest_url(bundle)).json()["manifest_digest_hex"]

    # Resources outside this bundle's snapshot must not move the digest:
    # a sibling bundle, a second claim with its own bundle, and an attestation
    # of the bundle's claim rather than the bundle itself.
    _create_bundle(client, claim["id"], "sibling")
    other_claim = _create_claim(
        client, content["id"], "other", claim_type="review"
    )
    other_bundle = _create_bundle(client, other_claim["id"], "other-evidence")
    _create_attestation(
        client, "claim", claim["id"], seed=SEED_A, signer="org-1"
    )
    _create_attestation(
        client, "evidence_bundle", other_bundle["id"], seed=SEED_A,
        signer="org-1",
    )

    assert (
        client.get(_manifest_url(bundle)).json()["manifest_digest_hex"]
        == baseline
    )
    # ... while the unrelated bundle has its own distinct manifest.
    other_manifest = client.get(_manifest_url(other_bundle)).json()
    assert other_manifest["manifest_digest_hex"] != baseline
    assert other_manifest["evidence_bundle_id"] == other_bundle["id"]


# --- Query validation and missing resources ----------------------------------


def test_manifest_unknown_bundle_is_404(client):
    resp = client.get(
        "/v1/evidence-bundles/evb_doesnotexist/exchange/manifest"
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "evidence_bundle_not_found"


def test_manifest_rejects_any_query_parameter(client):
    _, _, bundle = _setup_bundle(client)
    base = _manifest_url(bundle)

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


def test_manifest_unknown_bundle_with_query_param_is_422(client):
    # Parameters are validated before any existence lookup, matching the
    # exchange route's boundary.
    resp = client.get(
        "/v1/evidence-bundles/evb_doesnotexist/exchange/manifest?anything=1"
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only and determinism -----------------------------------------------


def test_manifest_writes_no_resources_or_audit_events(client, db_session):
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
    assert client.get(_manifest_url(bundle)).status_code == 200
    assert client.get(_manifest_url(plain_bundle)).status_code == 200
    assert client.get(f"{_manifest_url(bundle)}?x=1").status_code == 422
    assert client.get(
        "/v1/evidence-bundles/evb_missing/exchange/manifest"
    ).status_code == 404

    db_session.expire_all()
    assert _counts() == before


def test_manifest_is_deterministic_across_repeated_reads(client):
    _, _, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )

    first = client.get(_manifest_url(bundle))
    second = client.get(_manifest_url(bundle))
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


def test_manifest_is_deterministic_across_restart(file_client, tmp_db_url):
    content, claim, bundle = _setup_bundle(file_client)
    attested = _create_attestation(
        file_client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    _revoke(file_client, attested["id"])
    expected = file_client.get(_manifest_url(bundle)).json()

    # A brand-new app/engine over the same file reproduces the manifest
    # exactly (stable ids, UTC timestamps, attestation creation order).
    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        resp = client.get(_manifest_url(bundle))
        assert resp.status_code == 200, resp.text
        assert resp.json() == expected

        # The read after restart created nothing and stays identical.
        assert client.get(_manifest_url(bundle)).json() == expected
