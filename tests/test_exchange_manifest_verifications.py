"""Tests for the stateless offline exchange-manifest verification endpoint.

Covers POST /v1/exchange-manifest-verifications: the request body is exactly
{"manifest_version", "evidence_bundle_id", "digest_algorithm",
"manifest_digest_hex", "snapshot"}; the version is fixed at
``provenance-exchange-manifest-v1``, the algorithm at ``sha256``, and the
claimed digest must be 64 lowercase hex characters. The snapshot is exactly
{"content", "claim", "evidence_bundle", "attestations"} with the existing
public-view fields; the bundle id must match the manifest, the claim/content
associations must be consistent, and every attestation must target this
evidence bundle. Structural, field, association, or raw-material violations
are 422 validation_error; a structurally valid request whose digest differs
is 200 {"valid": false, "computed_digest_hex": ...}; a match is exactly
{"valid": true}. The route is fully stateless: it needs no persisted
resources and creates, modifies, and queries nothing (no resource or audit
rows, even for unknown ids). All fixtures are deterministic and offline.
"""

from __future__ import annotations

import base64
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
from tests.helpers import DIGEST_A, DIGEST_B, DIGEST_C, SEED_A, SEED_B
from tests.helpers import ed25519_public_key
from tests.test_evidence_bundle_exchange import (
    _create_attestation,
    _create_bundle,
    _create_claim,
    _create_content,
    _revoke,
    _setup_bundle,
)

URL = "/v1/exchange-manifest-verifications"
MANIFEST_VERSION = "provenance-exchange-manifest-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _exchange_url(bundle) -> str:
    return f"/v1/evidence-bundles/{bundle['id']}/exchange"


def _manifest_url(bundle) -> str:
    return f"/v1/evidence-bundles/{bundle['id']}/exchange/manifest"


def _canonical_snapshot_bytes(snapshot: dict, *, ascii_escape: bool = False) -> bytes:
    """Independently apply the manifest canonicalization to a parsed snapshot.

    Root members keep their snapshot order; every nested object sorts its
    members by Unicode code point; arrays keep element order; compact
    separators; non-ASCII emitted unescaped (unless checking the escaping
    requirement); UTF-8 encoded.
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


def _digest_of(snapshot: dict) -> str:
    return hashlib.sha256(_canonical_snapshot_bytes(snapshot)).hexdigest()


def _served_request(client, bundle) -> dict:
    """A verification request built from the service's own exchange views."""
    snapshot = client.get(_exchange_url(bundle)).json()
    manifest = client.get(_manifest_url(bundle)).json()
    return {**manifest, "snapshot": snapshot}


def _offline_snapshot() -> dict:
    """A fully self-consistent snapshot fabricated without any service state."""
    content = {
        "id": "cnt_" + hashlib.sha256(b"offline-content").hexdigest(),
        "digest_algorithm": "sha256",
        "digest_hex": DIGEST_A,
        "media_type": "image/png",
        "title": "offline 快照",
        "actor_id": "org-offline",
        "created_at": "2026-01-02T03:04:05Z",
    }
    claim = {
        "id": "clm_" + hashlib.sha256(b"offline-claim").hexdigest(),
        "content_id": content["id"],
        "actor_id": "org-offline",
        "claim_type": "authorship",
        "payload_digest_algorithm": "sha256",
        "payload_digest_hex": DIGEST_B,
        "created_at": "2026-01-02T03:04:06Z",
    }
    bundle_id = "evb_" + hashlib.sha256(b"offline-bundle").hexdigest()
    bundle = {
        "id": bundle_id,
        "claim_id": claim["id"],
        "evidence_type": "raw_capture",
        "digest_algorithm": "sha256",
        "digest_hex": DIGEST_C,
        "media_type": "image/jpeg",
        "metadata": {"origin": "offline", "标签": {"中": True}},
        "created_at": "2026-01-02T03:04:07Z",
    }
    attestation = {
        "id": "att_" + hashlib.sha256(b"offline-attestation").hexdigest(),
        "target_type": "evidence_bundle",
        "target_id": bundle_id,
        "signer_actor_id": "org-offline",
        "public_key": base64.b64encode(ed25519_public_key(SEED_A)).decode("ascii"),
        "signature_digest_algorithm": "sha256",
        "signature_digest_hex": hashlib.sha256(b"offline-signature").hexdigest(),
        "verified": True,
        "created_at": "2026-01-02T03:04:08Z",
    }
    return {
        "content": content,
        "claim": claim,
        "evidence_bundle": bundle,
        "attestations": [attestation],
    }


def _offline_request(snapshot: dict | None = None) -> dict:
    snapshot = _offline_snapshot() if snapshot is None else snapshot
    return {
        "manifest_version": MANIFEST_VERSION,
        "evidence_bundle_id": snapshot["evidence_bundle"]["id"],
        "digest_algorithm": "sha256",
        "manifest_digest_hex": _digest_of(snapshot),
        "snapshot": snapshot,
    }


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- Verdicts -----------------------------------------------------------------


def test_verification_of_served_manifest_is_valid(client):
    _, _, bundle = _setup_bundle(client)
    attested = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    # A revoked attestation stays in the snapshot and still verifies.
    _revoke(client, attested["id"])

    resp = client.post(URL, json=_served_request(client, bundle))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_digest_mismatch_returns_computed_digest(client):
    _, _, bundle = _setup_bundle(client)
    request = _served_request(client, bundle)
    snapshot = request["snapshot"]

    # Flip the last hex character of the claimed digest.
    claimed = request["manifest_digest_hex"]
    request["manifest_digest_hex"] = claimed[:-1] + (
        "0" if claimed[-1] != "0" else "1"
    )

    resp = client.post(URL, json=request)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"valid", "computed_digest_hex"}
    assert body["valid"] is False
    assert body["computed_digest_hex"] == _digest_of(snapshot)
    assert body["computed_digest_hex"] == claimed


def test_verification_tampered_snapshot_is_invalid(client):
    _, _, bundle = _setup_bundle(client)
    request = _served_request(client, bundle)
    # The manifest digest still commits to the untouched snapshot.
    request["snapshot"]["content"]["title"] = "forged title"

    resp = client.post(URL, json=request)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert body["computed_digest_hex"] == _digest_of(request["snapshot"])
    assert body["computed_digest_hex"] != request["manifest_digest_hex"]


def test_verification_attestation_order_participates(client):
    _, _, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_B, signer="org-2"
    )
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    request = _served_request(client, bundle)

    # Arrays keep their order: the same attestations in another order are a
    # different snapshot and no longer match the manifest.
    reordered = json.loads(json.dumps(request))
    reordered["snapshot"]["attestations"] = list(
        reversed(reordered["snapshot"]["attestations"])
    )
    resp = client.post(URL, json=reordered)
    assert resp.status_code == 200, resp.text
    assert resp.json()["valid"] is False
    assert resp.json()["computed_digest_hex"] == _digest_of(reordered["snapshot"])


def test_verification_root_member_order_is_preserved(client):
    snapshot = _offline_snapshot()
    # Root members keep the request's order: a manifest computed over a
    # differently ordered root verifies against that same order.
    reordered_root = {
        key: snapshot[key]
        for key in ("attestations", "evidence_bundle", "claim", "content")
    }
    request = _offline_request(reordered_root)

    resp = client.post(URL, json=request)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}

    # The digest of the canonically ordered root would not have matched.
    assert request["manifest_digest_hex"] != _digest_of(snapshot)


def test_verification_uses_unescaped_non_ascii_utf8(client):
    snapshot = _offline_snapshot()
    assert any(
        ord(ch) > 127 for ch in json.dumps(snapshot, ensure_ascii=False)
    )
    resp = client.post(URL, json=_offline_request(snapshot))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}

    # A digest computed over ASCII-escaped bytes would not match.
    escaped = hashlib.sha256(
        _canonical_snapshot_bytes(snapshot, ascii_escape=True)
    ).hexdigest()
    assert escaped != _digest_of(snapshot)


def test_verification_is_deterministic_across_repeated_requests(client):
    _, _, bundle = _setup_bundle(client)
    request = _served_request(client, bundle)

    first = client.post(URL, json=request)
    second = client.post(URL, json=request)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json() == {"valid": True}


# --- Statelessness -------------------------------------------------------------


def test_verification_needs_no_persisted_resources(client):
    # The database is empty and every identifier is unknown to the service:
    # verification is decided by the request body alone.
    resp = client.post(URL, json=_offline_request())
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_writes_no_resources_or_audit_events(client, db_session):
    _, _, bundle = _setup_bundle(client)
    valid_request = _served_request(client, bundle)
    invalid_request = json.loads(json.dumps(valid_request))
    invalid_request["snapshot"]["content"]["title"] = "tampered"
    malformed_request = _offline_request()
    del malformed_request["snapshot"]["claim"]

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

    assert client.post(URL, json=valid_request).status_code == 200
    assert client.post(URL, json=invalid_request).status_code == 200
    _assert_validation_error(client.post(URL, json=malformed_request))
    # A wholly offline request against unknown ids also writes nothing.
    assert client.post(URL, json=_offline_request()).status_code == 200

    db_session.expire_all()
    assert _counts() == before


# --- Top-level structure and manifest fields -----------------------------------


def test_verification_requires_exactly_the_five_top_level_fields(client):
    request = _offline_request()

    for field in (
        "manifest_version",
        "evidence_bundle_id",
        "digest_algorithm",
        "manifest_digest_hex",
        "snapshot",
    ):
        incomplete = {k: v for k, v in request.items() if k != field}
        _assert_validation_error(client.post(URL, json=incomplete))

    augmented = dict(request)
    augmented["computed_digest_hex"] = request["manifest_digest_hex"]
    _assert_validation_error(client.post(URL, json=augmented))


def test_verification_rejects_non_object_and_empty_bodies(client):
    for body in (None, [], "manifest", 42):
        _assert_validation_error(client.post(URL, json=body))
    _assert_validation_error(client.post(URL, json={}))


def test_verification_requires_fixed_manifest_version(client):
    request = _offline_request()
    for version in (
        "provenance-exchange-manifest-v2",
        "Provenance-Exchange-Manifest-V1",
        "",
    ):
        bad = dict(request, manifest_version=version)
        _assert_validation_error(client.post(URL, json=bad))


def test_verification_requires_sha256_digest_algorithm(client):
    request = _offline_request()
    for algorithm in ("sha512", "SHA256", "sha-256", ""):
        bad = dict(request, digest_algorithm=algorithm)
        _assert_validation_error(client.post(URL, json=bad))


def test_verification_requires_64_lowercase_hex_manifest_digest(client):
    request = _offline_request()
    valid = request["manifest_digest_hex"]
    for digest in (
        valid.upper(),
        valid[:-1],
        valid + "0",
        "g" + valid[1:],
        " " + valid,
        123,
        None,
    ):
        bad = dict(request, manifest_digest_hex=digest)
        _assert_validation_error(client.post(URL, json=bad))


# --- Snapshot structure and fields ----------------------------------------------


def test_verification_snapshot_has_exactly_the_four_members(client):
    request = _offline_request()

    for member in ("content", "claim", "evidence_bundle", "attestations"):
        incomplete = json.loads(json.dumps(request))
        del incomplete["snapshot"][member]
        _assert_validation_error(client.post(URL, json=incomplete))

    augmented = json.loads(json.dumps(request))
    augmented["snapshot"]["lineage"] = []
    _assert_validation_error(client.post(URL, json=augmented))


def test_verification_snapshot_members_have_correct_types(client):
    request = _offline_request()
    for member, wrong in (
        ("content", []),
        ("claim", None),
        ("evidence_bundle", "evb"),
        ("attestations", {}),
    ):
        bad = json.loads(json.dumps(request))
        bad["snapshot"][member] = wrong
        _assert_validation_error(client.post(URL, json=bad))


def test_verification_nested_views_forbid_undeclared_fields(client):
    request = _offline_request()

    cases = []
    extra_content = json.loads(json.dumps(request))
    extra_content["snapshot"]["content"]["data"] = "AAAA"
    cases.append(extra_content)
    extra_claim = json.loads(json.dumps(request))
    extra_claim["snapshot"]["claim"]["payload"] = {"raw": True}
    cases.append(extra_claim)
    extra_bundle = json.loads(json.dumps(request))
    extra_bundle["snapshot"]["evidence_bundle"]["evidence"] = "AAAA"
    cases.append(extra_bundle)
    extra_attestation = json.loads(json.dumps(request))
    extra_attestation["snapshot"]["attestations"][0]["signature"] = "AAAA"
    cases.append(extra_attestation)

    for case in cases:
        _assert_validation_error(client.post(URL, json=case))


def test_verification_nested_views_require_every_public_field(client):
    request = _offline_request()
    snapshot = request["snapshot"]

    missing_cases = []
    for member, field in (
        ("content", "created_at"),
        ("claim", "payload_digest_hex"),
        ("evidence_bundle", "metadata"),
    ):
        case = json.loads(json.dumps(request))
        del case["snapshot"][member][field]
        missing_cases.append(case)
    attestation_case = json.loads(json.dumps(request))
    del attestation_case["snapshot"]["attestations"][0]["signature_digest_hex"]
    missing_cases.append(attestation_case)

    for case in missing_cases:
        _assert_validation_error(client.post(URL, json=case))
    assert snapshot  # the source request itself was well-formed


def test_verification_nested_field_types_are_enforced(client):
    request = _offline_request()

    cases = []
    bad = json.loads(json.dumps(request))
    bad["snapshot"]["content"]["created_at"] = "not-a-timestamp"
    cases.append(bad)
    bad = json.loads(json.dumps(request))
    bad["snapshot"]["claim"]["payload_digest_hex"] = 123
    cases.append(bad)
    bad = json.loads(json.dumps(request))
    bad["snapshot"]["evidence_bundle"]["metadata"] = ["not", "an", "object"]
    cases.append(bad)
    bad = json.loads(json.dumps(request))
    bad["snapshot"]["attestations"][0]["target_type"] = "content"
    cases.append(bad)
    bad = json.loads(json.dumps(request))
    bad["snapshot"]["attestations"][0]["verified"] = "not-a-bool"
    cases.append(bad)

    for case in cases:
        _assert_validation_error(client.post(URL, json=case))


def test_verification_metadata_must_have_finite_numbers(client):
    # Python's json emits NaN/Infinity literals, which some clients can
    # produce; they have no canonical JSON form and are rejected.
    for non_finite in ("NaN", "Infinity", "-Infinity"):
        request = _offline_request()
        raw = json.dumps(request).replace(
            '"origin": "offline"', f'"origin": {non_finite}'
        )
        resp = client.post(
            URL, content=raw, headers={"content-type": "application/json"}
        )
        _assert_validation_error(resp)


# --- Associations ----------------------------------------------------------------


def test_verification_bundle_id_must_match_manifest(client):
    request = _offline_request()
    request["evidence_bundle_id"] = "evb_" + hashlib.sha256(b"other").hexdigest()
    _assert_validation_error(client.post(URL, json=request))


def test_verification_claim_must_match_bundle(client):
    request = _offline_request()
    request["snapshot"]["claim"]["id"] = "clm_" + hashlib.sha256(b"x").hexdigest()
    _assert_validation_error(client.post(URL, json=request))


def test_verification_content_must_match_claim(client):
    request = _offline_request()
    request["snapshot"]["content"]["id"] = "cnt_" + hashlib.sha256(b"x").hexdigest()
    _assert_validation_error(client.post(URL, json=request))


def test_verification_attestations_must_target_the_bundle(client):
    request = _offline_request()
    bundle_id = request["evidence_bundle_id"]

    wrong_type = json.loads(json.dumps(request))
    wrong_type["snapshot"]["attestations"][0]["target_type"] = "claim"
    wrong_type["snapshot"]["attestations"][0]["target_id"] = request["snapshot"][
        "claim"
    ]["id"]
    _assert_validation_error(client.post(URL, json=wrong_type))

    wrong_id = json.loads(json.dumps(request))
    wrong_id["snapshot"]["attestations"][0]["target_id"] = (
        "evb_" + hashlib.sha256(b"other").hexdigest()
    )
    _assert_validation_error(client.post(URL, json=wrong_id))

    # A well-formed empty attestation list is fine.
    empty = json.loads(json.dumps(request))
    empty["snapshot"]["attestations"] = []
    empty["manifest_digest_hex"] = _digest_of(empty["snapshot"])
    resp = client.post(URL, json=empty)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}
    assert bundle_id


def test_verification_recomputed_digest_matches_served_manifest(client):
    # End-to-end agreement: the digest this endpoint computes for the served
    # snapshot equals the digest the manifest endpoint issued for it.
    _, _, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    request = _served_request(client, bundle)
    request["manifest_digest_hex"] = "0" * 64

    resp = client.post(URL, json=request)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert _HEX64.fullmatch(body["computed_digest_hex"])
    assert body["computed_digest_hex"] == client.get(_manifest_url(bundle)).json()[
        "manifest_digest_hex"
    ]
