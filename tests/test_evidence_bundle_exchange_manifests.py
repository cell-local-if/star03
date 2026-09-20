"""Tests for the read-only evidence-bundle exchange integrity manifest.

Covers GET /v1/evidence-bundles/{evidence_bundle_id}/exchange/manifest:
the success body is exactly {"manifest_version", "evidence_bundle_id",
"digest_algorithm", "manifest_digest_hex"} with fixed version
``provenance-exchange-manifest-v1`` and fixed algorithm ``sha256``; the
digest is independently recomputed from the existing exchange snapshot
under the stated canonical rules (root members and arrays keep the
snapshot order, nested object keys sort by Unicode code point, compact
separators, unescaped non-ASCII, UTF-8); only the four snapshot members
contribute, so unrelated attestations/resources do not and revoked
attestations still do; any or repeated query parameter is 422
validation_error before the bundle lookup; an unknown bundle is 404
evidence_bundle_not_found; the route is read-only (no resource or audit
rows on success or failure); and the manifest is deterministic across
repeated reads and an app restart. All fixtures are deterministic and
offline.
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


def _manifest_url(bundle) -> str:
    return f"/v1/evidence-bundles/{bundle['id']}/exchange/manifest"


def _exchange_url(bundle) -> str:
    return f"/v1/evidence-bundles/{bundle['id']}/exchange"


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


# --- Success shape and digest ------------------------------------------------


def test_manifest_success_has_exact_fixed_shape(client):
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
    assert body["manifest_version"] == MANIFEST_VERSION
    assert body["digest_algorithm"] == "sha256"
    assert body["evidence_bundle_id"] == bundle["id"]
    assert _HEX64.fullmatch(body["manifest_digest_hex"])


def test_manifest_digest_matches_independent_canonicalization_of_snapshot(client):
    _, claim, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    # Attestation of the bundle's claim contributes nothing to the snapshot.
    _create_attestation(
        client, "claim", claim["id"], seed=SEED_B, signer="org-2"
    )

    snapshot = client.get(_exchange_url(bundle)).json()
    manifest = client.get(_manifest_url(bundle)).json()

    assert manifest["manifest_digest_hex"] == _expected_manifest_digest(snapshot)


def test_manifest_canonicalization_sorts_nested_object_keys_only(client):
    _, _, bundle = _setup_bundle(client)
    snapshot = client.get(_exchange_url(bundle)).json()
    manifest = client.get(_manifest_url(bundle)).json()

    # The canonical digest sorts nested object keys (e.g. the content view's
    # keys are not in Unicode order on the wire) while keeping the four root
    # members in snapshot order.
    assert manifest["manifest_digest_hex"] == _expected_manifest_digest(snapshot)
    unsorted_nested = hashlib.sha256(
        json.dumps(
            snapshot, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    assert manifest["manifest_digest_hex"] != unsorted_nested
    root_sorted = {key: snapshot[key] for key in sorted(snapshot)}
    assert manifest["manifest_digest_hex"] != _expected_manifest_digest(root_sorted)


def test_manifest_digest_covers_exactly_the_four_snapshot_members(client):
    _, claim, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    other_content = _create_content(client, "other")
    other_claim = _create_claim(client, other_content["id"], "other-claim")
    other_bundle = _create_bundle(client, other_claim["id"], "other-evidence")
    _create_attestation(
        client, "evidence_bundle", other_bundle["id"], seed=SEED_B,
        signer="org-2",
    )

    snapshot = client.get(_exchange_url(bundle)).json()
    served = client.get(_manifest_url(bundle)).json()["manifest_digest_hex"]

    # Adding any fifth member changes the digest: nothing beyond the four
    # snapshot members may participate.
    augmented = dict(snapshot)
    augmented["extra"] = {"unrelated": True}
    assert served != _expected_manifest_digest(augmented)

    # A different bundle (its own resources and attestations) has a
    # different manifest.
    other_snapshot = client.get(_exchange_url(other_bundle)).json()
    other_manifest = client.get(_manifest_url(other_bundle)).json()
    assert other_manifest["manifest_digest_hex"] == _expected_manifest_digest(
        other_snapshot
    )
    assert other_manifest["manifest_digest_hex"] != served


def test_manifest_attestation_array_order_participates(client):
    _, _, bundle = _setup_bundle(client)
    first = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_B, signer="org-2"
    )
    second = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )

    snapshot = client.get(_exchange_url(bundle)).json()
    assert [a["id"] for a in snapshot["attestations"]] == [
        first["id"],
        second["id"],
    ]
    served = client.get(_manifest_url(bundle)).json()["manifest_digest_hex"]

    reversed_order = dict(snapshot)
    reversed_order["attestations"] = list(reversed(snapshot["attestations"]))
    assert served != _expected_manifest_digest(reversed_order)


def test_manifest_retains_revoked_attestation_in_digest(client):
    _, _, bundle = _setup_bundle(client)
    attested = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    _revoke(client, attested["id"])

    # The revocation neither removes the attestation nor alters its view; it
    # still participates in the digest.
    snapshot = client.get(_exchange_url(bundle)).json()
    assert [a["id"] for a in snapshot["attestations"]] == [attested["id"]]
    served = client.get(_manifest_url(bundle)).json()["manifest_digest_hex"]
    assert served == _expected_manifest_digest(snapshot)

    without_revoked = dict(snapshot)
    without_revoked["attestations"] = []
    assert served != _expected_manifest_digest(without_revoked)


def test_manifest_digest_uses_unescaped_non_ascii_utf8(client):
    content, _, _ = _setup_bundle(client)
    # A second, distinct bundle identity with non-ASCII metadata keys and
    # values (bundles are immutable, so this is a fresh bundle).
    claim = _create_claim(client, content["id"], "unicode-claim")
    resp = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim["id"],
            "evidence_type": "unicode_capture",
            "digest_algorithm": "sha256",
            "digest_hex": hashlib.sha256(b"unicode-evidence-2").hexdigest(),
            "media_type": "image/jpeg",
            "metadata": {"label": "日本語の証拠", "é": 1, "a": {"中": True}},
        },
    )
    assert resp.status_code == 201, resp.text
    unicode_bundle = resp.json()

    snapshot = client.get(_exchange_url(unicode_bundle)).json()
    served = client.get(
        _manifest_url(unicode_bundle)
    ).json()["manifest_digest_hex"]

    assert served == _expected_manifest_digest(snapshot)
    # ASCII-escaping the same canonical object must produce a different
    # digest: the manifest hashes unescaped non-ASCII as raw UTF-8 bytes.
    escaped = hashlib.sha256(
        _canonical_manifest_bytes(snapshot, ascii_escape=True)
    ).hexdigest()
    assert served != escaped



def test_manifest_carries_no_raw_material_and_four_fields_only(client):
    _, _, bundle = _setup_bundle(client)
    body = client.get(_manifest_url(bundle)).json()
    assert set(body) == {
        "manifest_version",
        "evidence_bundle_id",
        "digest_algorithm",
        "manifest_digest_hex",
    }
    snapshot = client.get(_exchange_url(bundle)).json()
    # The hashed view is the existing public snapshot: no raw signatures,
    # claim payloads, content bytes, or evidence bytes anywhere in it.
    serialized = json.dumps(snapshot)
    for forbidden in ('"signature"', '"payload"', '"data"', '"evidence"'):
        assert forbidden not in serialized


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
    expected = file_client.get(_manifest_url(bundle))
    assert expected.status_code == 200, expected.text
    expected_body = expected.json()

    # A brand-new app/engine over the same file reproduces the manifest
    # byte-for-byte (stable ids, UTC timestamps, and creation order).
    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        resp = client.get(_manifest_url(bundle))
        assert resp.status_code == 200, resp.text
        assert resp.json() == expected_body
        assert resp.json()["evidence_bundle_id"] == bundle["id"]

        # The read after restart created nothing and stays stable.
        assert client.get(_manifest_url(bundle)).json() == expected_body
        assert attested["id"]  # fixture bundle carries one attestation
        assert content["id"] and claim["id"]
