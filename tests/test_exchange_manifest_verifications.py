"""Tests for the stateless offline exchange-manifest verification endpoint.

Covers POST /v1/exchange-manifest-verifications: the request body is exactly
{"manifest_version", "evidence_bundle_id", "digest_algorithm",
"manifest_digest_hex", "snapshot"}; the version is fixed at
``provenance-exchange-manifest-v1``, the algorithm at ``sha256``, and the
digest must be 64 lowercase hexadecimal characters. The snapshot is exactly
{"content", "claim", "evidence_bundle", "attestations"} with the existing
public-view fields and self-consistent associations (bundle id matches the
manifest, the claim asserts the content, the bundle is attached to the
claim, every attestation targets this bundle). The digest is recomputed
under the existing manifest canonical rules (root members and arrays keep
the submitted order, nested object keys sort by Unicode code point, compact
separators, unescaped non-ASCII, UTF-8): a structural, field, association,
or raw-material violation is 422 validation_error; a structurally valid
mismatch is 200 {"valid": false, "computed_digest_hex": ...}; a match is
200 with "valid": true. The route is fully stateless: it queries, creates,
and modifies no local resource or audit record. All fixtures are
deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json
import re

from sqlalchemy import func, select

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
    _setup_bundle,
)

URL = "/v1/exchange-manifest-verifications"
MANIFEST_VERSION = "provenance-exchange-manifest-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_manifest_bytes(snapshot: dict) -> bytes:
    """Independently apply the manifest canonicalization to a parsed snapshot.

    Root members keep their submitted order; every nested object sorts its
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

    return json.dumps(
        normalize(snapshot, False),
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _expected_digest(snapshot: dict) -> str:
    return hashlib.sha256(_canonical_manifest_bytes(snapshot)).hexdigest()


def _snapshot_and_manifest(client, *, attest: bool = True):
    _, claim, bundle = _setup_bundle(client)
    if attest:
        _create_attestation(
            client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
        )
    snapshot = client.get(
        f"/v1/evidence-bundles/{bundle['id']}/exchange"
    ).json()
    manifest = client.get(
        f"/v1/evidence-bundles/{bundle['id']}/exchange/manifest"
    ).json()
    return claim, bundle, snapshot, manifest


def _payload(snapshot, manifest, **overrides):
    body = {
        "manifest_version": manifest["manifest_version"],
        "evidence_bundle_id": manifest["evidence_bundle_id"],
        "digest_algorithm": manifest["digest_algorithm"],
        "manifest_digest_hex": manifest["manifest_digest_hex"],
        "snapshot": snapshot,
    }
    body.update(overrides)
    return body


def _other_hex(digest: str) -> str:
    # A well-formed 64-lowercase-hex digest that differs from ``digest``.
    candidate = "0" * 64
    return candidate if candidate != digest else "1" * 64


# --- Verdicts -----------------------------------------------------------------


def test_valid_manifest_and_snapshot_returns_valid_true(client):
    _, _, snapshot, manifest = _snapshot_and_manifest(client)

    resp = client.post(URL, json=_payload(snapshot, manifest))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"valid", "computed_digest_hex"}
    assert body["valid"] is True
    assert body["computed_digest_hex"] == manifest["manifest_digest_hex"]
    assert _HEX64.fullmatch(body["computed_digest_hex"])


def test_computed_digest_matches_independent_canonicalization(client):
    _, _, snapshot, manifest = _snapshot_and_manifest(client)

    body = client.post(URL, json=_payload(snapshot, manifest)).json()
    assert body["computed_digest_hex"] == _expected_digest(snapshot)
    assert body["valid"] is True


def test_empty_attestations_snapshot_verifies(client):
    _, _, snapshot, manifest = _snapshot_and_manifest(client, attest=False)
    assert snapshot["attestations"] == []

    body = client.post(URL, json=_payload(snapshot, manifest)).json()
    assert body == {
        "valid": True,
        "computed_digest_hex": manifest["manifest_digest_hex"],
    }


def test_mismatched_digest_returns_200_valid_false(client):
    _, _, snapshot, manifest = _snapshot_and_manifest(client)
    tampered = _other_hex(manifest["manifest_digest_hex"])

    resp = client.post(
        URL, json=_payload(snapshot, manifest, manifest_digest_hex=tampered)
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert body["computed_digest_hex"] == manifest["manifest_digest_hex"]


def test_tampered_snapshot_returns_valid_false(client):
    _, _, snapshot, manifest = _snapshot_and_manifest(client)
    tampered_snapshot = json.loads(json.dumps(snapshot))
    tampered_snapshot["claim"]["claim_type"] = "endorsement"

    body = client.post(
        URL, json=_payload(tampered_snapshot, manifest)
    ).json()
    assert body["valid"] is False
    assert body["computed_digest_hex"] == _expected_digest(tampered_snapshot)
    assert body["computed_digest_hex"] != manifest["manifest_digest_hex"]


def test_multiple_attestations_keep_array_order(client):
    _, claim, bundle = _setup_bundle(client)
    first = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_B, signer="org-2"
    )
    second = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    snapshot = client.get(
        f"/v1/evidence-bundles/{bundle['id']}/exchange"
    ).json()
    manifest = client.get(
        f"/v1/evidence-bundles/{bundle['id']}/exchange/manifest"
    ).json()
    assert [a["id"] for a in snapshot["attestations"]] == [
        first["id"],
        second["id"],
    ]

    body = client.post(URL, json=_payload(snapshot, manifest)).json()
    assert body["valid"] is True

    # Reversing the attestation array changes the canonical bytes: the
    # recomputed digest no longer matches the manifest.
    reordered = json.loads(json.dumps(snapshot))
    reordered["attestations"] = list(reversed(reordered["attestations"]))
    body = client.post(URL, json=_payload(reordered, manifest)).json()
    assert body["valid"] is False
    assert body["computed_digest_hex"] == _expected_digest(reordered)


# --- Canonicalization semantics ------------------------------------------------


def test_root_member_order_participates_in_digest(client):
    _, _, snapshot, manifest = _snapshot_and_manifest(client)

    # Same members, different root order: the digest commits to the
    # submitted order, so the server-issued manifest digest no longer
    # matches.
    reordered = {
        "attestations": snapshot["attestations"],
        "evidence_bundle": snapshot["evidence_bundle"],
        "claim": snapshot["claim"],
        "content": snapshot["content"],
    }
    body = client.post(URL, json=_payload(reordered, manifest)).json()
    assert body["valid"] is False
    assert body["computed_digest_hex"] == _expected_digest(reordered)

    # A manifest digest recomputed over that same root order verifies:
    # the root order is preserved, never re-sorted.
    resp = client.post(
        URL,
        json=_payload(
            reordered, manifest, manifest_digest_hex=_expected_digest(reordered)
        ),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["valid"] is True


def test_nested_member_order_does_not_affect_digest(client):
    _, _, snapshot, manifest = _snapshot_and_manifest(client)

    # Nested object members are sorted by Unicode code point during
    # canonicalization, so their submitted order is irrelevant.
    reordered = json.loads(json.dumps(snapshot))
    reordered["claim"] = dict(reversed(list(reordered["claim"].items())))
    reordered["evidence_bundle"] = dict(
        reversed(list(reordered["evidence_bundle"].items()))
    )
    body = client.post(URL, json=_payload(reordered, manifest)).json()
    assert body["valid"] is True
    assert body["computed_digest_hex"] == manifest["manifest_digest_hex"]


def test_non_ascii_metadata_verifies_unescaped(client):
    _, claim, _ = _setup_bundle(client)
    resp = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim["id"],
            "evidence_type": "unicode_capture",
            "digest_algorithm": "sha256",
            "digest_hex": hashlib.sha256(b"unicode-evidence").hexdigest(),
            "media_type": "image/jpeg",
            "metadata": {"label": "日本語の証拠", "é": 1},
        },
    )
    assert resp.status_code == 201, resp.text
    bundle = resp.json()
    snapshot = client.get(
        f"/v1/evidence-bundles/{bundle['id']}/exchange"
    ).json()
    manifest = client.get(
        f"/v1/evidence-bundles/{bundle['id']}/exchange/manifest"
    ).json()

    body = client.post(URL, json=_payload(snapshot, manifest)).json()
    assert body["valid"] is True
    assert body["computed_digest_hex"] == _expected_digest(snapshot)


# --- Structural and field validation (422) -------------------------------------


def _assert_422(resp):
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def test_missing_or_extra_root_members_are_422(client):
    _, _, snapshot, manifest = _snapshot_and_manifest(client)
    base = _payload(snapshot, manifest)

    for field in (
        "manifest_version",
        "evidence_bundle_id",
        "digest_algorithm",
        "manifest_digest_hex",
        "snapshot",
    ):
        body = {key: value for key, value in base.items() if key != field}
        _assert_422(client.post(URL, json=body))

    for extra in ("payload", "data", "evidence", "signature", "computed"):
        body = dict(base, **{extra: "smuggled"})
        _assert_422(client.post(URL, json=body))


def test_fixed_version_and_algorithm_are_enforced(client):
    _, _, snapshot, manifest = _snapshot_and_manifest(client)

    for bad_version in (
        "provenance-exchange-manifest-v2",
        "PROVENANCE-EXCHANGE-MANIFEST-V1",
        "",
    ):
        _assert_422(
            client.post(
                URL,
                json=_payload(snapshot, manifest, manifest_version=bad_version),
            )
        )

    for bad_algorithm in ("SHA256", "sha512", ""):
        _assert_422(
            client.post(
                URL,
                json=_payload(snapshot, manifest, digest_algorithm=bad_algorithm),
            )
        )


def test_manifest_digest_hex_must_be_64_lowercase_hex(client):
    _, _, snapshot, manifest = _snapshot_and_manifest(client)
    valid = manifest["manifest_digest_hex"]

    for bad in (
        valid.upper(),
        valid[:-1],
        valid + "0",
        "g" * 64,
        "",
        " " + valid,
    ):
        _assert_422(
            client.post(
                URL, json=_payload(snapshot, manifest, manifest_digest_hex=bad)
            )
        )


def test_snapshot_must_be_an_object_with_exactly_four_members(client):
    _, _, snapshot, manifest = _snapshot_and_manifest(client)

    _assert_422(client.post(URL, json=_payload([], manifest)))
    _assert_422(client.post(URL, json=_payload(None, manifest)))

    for member in ("content", "claim", "evidence_bundle", "attestations"):
        broken = {key: value for key, value in snapshot.items() if key != member}
        _assert_422(client.post(URL, json=_payload(broken, manifest)))

    extra = dict(snapshot, extra_member={})
    _assert_422(client.post(URL, json=_payload(extra, manifest)))


def test_nested_members_use_exact_public_view_fields(client):
    _, _, snapshot, manifest = _snapshot_and_manifest(client)

    # A missing declared field in any nested public view is a 422.
    broken = json.loads(json.dumps(snapshot))
    del broken["content"]["id"]
    _assert_422(client.post(URL, json=_payload(broken, manifest)))

    broken = json.loads(json.dumps(snapshot))
    del broken["evidence_bundle"]["digest_hex"]
    _assert_422(client.post(URL, json=_payload(broken, manifest)))

    broken = json.loads(json.dumps(snapshot))
    del broken["attestations"][0]["verified"]
    _assert_422(client.post(URL, json=_payload(broken, manifest)))

    # Undeclared nested members — including any raw-material field — are 422.
    for member, field in (
        ("claim", "payload"),
        ("evidence_bundle", "data"),
        ("evidence_bundle", "evidence"),
        ("content", "bytes"),
    ):
        broken = json.loads(json.dumps(snapshot))
        broken[member][field] = {"raw": "material"}
        _assert_422(client.post(URL, json=_payload(broken, manifest)))

    broken = json.loads(json.dumps(snapshot))
    broken["attestations"][0]["signature"] = "AAAA"
    _assert_422(client.post(URL, json=_payload(broken, manifest)))


def test_malformed_nested_field_values_are_422(client):
    _, _, snapshot, manifest = _snapshot_and_manifest(client)

    broken = json.loads(json.dumps(snapshot))
    broken["content"]["created_at"] = "not-a-timestamp"
    _assert_422(client.post(URL, json=_payload(broken, manifest)))

    broken = json.loads(json.dumps(snapshot))
    broken["attestations"] = {}
    _assert_422(client.post(URL, json=_payload(broken, manifest)))


def test_invalid_json_body_is_422(client):
    resp = client.post(
        URL, content=b"{not json", headers={"content-type": "application/json"}
    )
    _assert_422(resp)


# --- Association consistency (422) --------------------------------------------


def test_bundle_id_must_match_manifest(client):
    _, _, snapshot, manifest = _snapshot_and_manifest(client)

    broken = json.loads(json.dumps(snapshot))
    broken["evidence_bundle"]["id"] = "evb_other"
    _assert_422(client.post(URL, json=_payload(broken, manifest)))

    _assert_422(
        client.post(
            URL, json=_payload(snapshot, manifest, evidence_bundle_id="evb_other")
        )
    )


def test_claim_must_assert_the_snapshot_content(client):
    _, _, snapshot, manifest = _snapshot_and_manifest(client)

    broken = json.loads(json.dumps(snapshot))
    broken["claim"]["content_id"] = "cnt_other"
    _assert_422(client.post(URL, json=_payload(broken, manifest)))


def test_bundle_must_be_attached_to_the_snapshot_claim(client):
    _, _, snapshot, manifest = _snapshot_and_manifest(client)

    broken = json.loads(json.dumps(snapshot))
    broken["evidence_bundle"]["claim_id"] = "clm_other"
    _assert_422(client.post(URL, json=_payload(broken, manifest)))


def test_every_attestation_must_target_this_bundle(client):
    _, _, snapshot, manifest = _snapshot_and_manifest(client)

    broken = json.loads(json.dumps(snapshot))
    broken["attestations"][0]["target_type"] = "claim"
    _assert_422(client.post(URL, json=_payload(broken, manifest)))

    broken = json.loads(json.dumps(snapshot))
    broken["attestations"][0]["target_id"] = "evb_other"
    _assert_422(client.post(URL, json=_payload(broken, manifest)))


# --- Stateless and offline -----------------------------------------------------


def test_verification_is_fully_offline_and_stateless(client):
    # A hand-built snapshot for identifiers that exist nowhere locally still
    # verifies: the endpoint performs no lookup of any kind.
    snapshot = {
        "content": {
            "id": "cnt_external",
            "digest_algorithm": "sha256",
            "digest_hex": hashlib.sha256(b"external-content").hexdigest(),
            "media_type": "image/png",
            "title": "External",
            "actor_id": "org-external",
            "created_at": "2026-01-02T03:04:05Z",
        },
        "claim": {
            "id": "clm_external",
            "content_id": "cnt_external",
            "actor_id": "org-external",
            "claim_type": "authorship",
            "payload_digest_algorithm": "sha256",
            "payload_digest_hex": hashlib.sha256(b"external-payload").hexdigest(),
            "created_at": "2026-01-02T03:05:05Z",
        },
        "evidence_bundle": {
            "id": "evb_external",
            "claim_id": "clm_external",
            "evidence_type": "raw_capture",
            "digest_algorithm": "sha256",
            "digest_hex": hashlib.sha256(b"external-evidence").hexdigest(),
            "media_type": "image/jpeg",
            "metadata": {"origin": "offline"},
            "created_at": "2026-01-02T03:06:05Z",
        },
        "attestations": [],
    }
    body = {
        "manifest_version": MANIFEST_VERSION,
        "evidence_bundle_id": "evb_external",
        "digest_algorithm": "sha256",
        "manifest_digest_hex": _expected_digest(snapshot),
        "snapshot": snapshot,
    }

    resp = client.post(URL, json=body)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "valid": True,
        "computed_digest_hex": body["manifest_digest_hex"],
    }


def test_verification_writes_no_resources_or_audit_events(client, db_session):
    _, _, snapshot, manifest = _snapshot_and_manifest(client)

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

    # A valid verdict, an invalid verdict, and a 422 all leave every table
    # untouched.
    assert client.post(URL, json=_payload(snapshot, manifest)).status_code == 200
    tampered = _payload(
        snapshot,
        manifest,
        manifest_digest_hex=_other_hex(manifest["manifest_digest_hex"]),
    )
    assert client.post(URL, json=tampered).status_code == 200
    _assert_422(client.post(URL, json=_payload(snapshot, manifest, extra=1)))

    db_session.expire_all()
    assert _counts() == before


def test_verification_is_deterministic_across_repeated_posts(client):
    _, _, snapshot, manifest = _snapshot_and_manifest(client)

    first = client.post(URL, json=_payload(snapshot, manifest))
    second = client.post(URL, json=_payload(snapshot, manifest))
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
