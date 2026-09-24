"""Tests for the stateless offline content-export verification endpoint.

Covers POST /v1/content-export-verifications: the request body is exactly
{"snapshot", "digest_algorithm", "digest_hex"}; the algorithm is fixed at
``sha256`` and the claimed digest must be 64 lowercase hex characters. The
snapshot is exactly {"content", "claims"} with the existing public-view
fields (each claim carrying its ``evidence_bundles``); every claim's
``content_id`` must equal the snapshot content's id and every bundle's
``claim_id`` must equal its claim's id. Structural, field, association, or
raw-material violations are 422 validation_error; a structurally valid
request whose digest differs is 200 {"valid": false,
"computed_digest_hex": ...}; a match is exactly {"valid": true}. The route
is fully stateless: it needs no persisted resources and creates, modifies,
and queries nothing (no resource or audit rows, even for unknown ids).
All fixtures are deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json
import re

from sqlalchemy import func, select

from provenance.models import (
    Actor,
    Attestation,
    AuditEvent,
    Claim,
    Content,
    ContentExportJob,
    EvidenceBundle,
)
from tests.helpers import DIGEST_A, DIGEST_B, DIGEST_C, create_actor
from tests.test_content_export import (
    _create_bundle,
    _create_claim,
    _create_content,
)

URL = "/v1/content-export-verifications"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_snapshot_bytes(snapshot: dict, *, ascii_escape: bool = False) -> bytes:
    """Independently apply the export canonicalization to a parsed snapshot.

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


def _request_for(snapshot: dict) -> dict:
    return {
        "snapshot": snapshot,
        "digest_algorithm": "sha256",
        "digest_hex": _digest_of(snapshot),
    }


def _export_url(content) -> str:
    return f"/v1/contents/{content['id']}/export"


def _served_request(client, content) -> dict:
    """A verification request built from the service's own export view."""
    snapshot = client.get(_export_url(content)).json()
    return _request_for(snapshot)


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
        "evidence_bundles": [
            {
                "id": "evb_" + hashlib.sha256(b"offline-bundle").hexdigest(),
                "claim_id": "clm_" + hashlib.sha256(b"offline-claim").hexdigest(),
                "evidence_type": "raw_capture",
                "digest_algorithm": "sha256",
                "digest_hex": DIGEST_C,
                "media_type": "image/jpeg",
                "metadata": {"origin": "offline", "标签": {"中": True}},
                "created_at": "2026-01-02T03:04:07Z",
            }
        ],
    }
    return {"content": content, "claims": [claim]}


def _offline_request(snapshot: dict | None = None) -> dict:
    snapshot = _offline_snapshot() if snapshot is None else snapshot
    return _request_for(snapshot)


def _setup_export(client):
    """One content with two claims and evidence bundles, served for export."""
    create_actor(client)
    content = _create_content(client, "main")
    claim_a = _create_claim(client, content["id"], "a")
    _create_claim(client, content["id"], "b", claim_type="review")
    _create_bundle(client, claim_a["id"], "a1")
    _create_bundle(client, claim_a["id"], "a2")
    return content


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- Verdicts -----------------------------------------------------------------


def test_verification_of_served_export_is_valid(client):
    content = _setup_export(client)

    resp = client.post(URL, json=_served_request(client, content))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_of_empty_claims_export_is_valid(client):
    create_actor(client)
    content = _create_content(client, "lonely")

    resp = client.post(URL, json=_served_request(client, content))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_digest_mismatch_returns_computed_digest(client):
    content = _setup_export(client)
    request = _served_request(client, content)
    snapshot = request["snapshot"]

    # Flip the last hex character of the claimed digest.
    claimed = request["digest_hex"]
    request["digest_hex"] = claimed[:-1] + ("0" if claimed[-1] != "0" else "1")

    resp = client.post(URL, json=request)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"valid", "computed_digest_hex"}
    assert body["valid"] is False
    assert body["computed_digest_hex"] == _digest_of(snapshot)
    assert body["computed_digest_hex"] == claimed


def test_verification_tampered_snapshot_is_invalid(client):
    content = _setup_export(client)
    request = _served_request(client, content)
    # The claimed digest still commits to the untouched snapshot.
    request["snapshot"]["content"]["title"] = "forged title"

    resp = client.post(URL, json=request)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert body["computed_digest_hex"] == _digest_of(request["snapshot"])
    assert body["computed_digest_hex"] != request["digest_hex"]


def test_verification_claim_and_bundle_order_participates(client):
    content = _setup_export(client)
    request = _served_request(client, content)

    # Arrays keep their order: the same claims in another order are a
    # different snapshot and no longer match the claimed digest.
    reordered = json.loads(json.dumps(request))
    reordered["snapshot"]["claims"] = list(reversed(reordered["snapshot"]["claims"]))
    resp = client.post(URL, json=reordered)
    assert resp.status_code == 200, resp.text
    assert resp.json()["valid"] is False
    assert resp.json()["computed_digest_hex"] == _digest_of(reordered["snapshot"])

    # The same holds inside a claim's evidence bundle array.
    reordered_bundles = json.loads(json.dumps(request))
    bundles = reordered_bundles["snapshot"]["claims"][0]["evidence_bundles"]
    assert len(bundles) == 2
    reordered_bundles["snapshot"]["claims"][0]["evidence_bundles"] = list(
        reversed(bundles)
    )
    resp = client.post(URL, json=reordered_bundles)
    assert resp.status_code == 200, resp.text
    assert resp.json()["valid"] is False
    assert resp.json()["computed_digest_hex"] == _digest_of(
        reordered_bundles["snapshot"]
    )


def test_verification_root_member_order_is_preserved(client):
    snapshot = _offline_snapshot()
    # Root members keep the request's order: a digest computed over a
    # differently ordered root verifies against that same order.
    reordered_root = {"claims": snapshot["claims"], "content": snapshot["content"]}
    request = _offline_request(reordered_root)

    resp = client.post(URL, json=request)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}

    # The digest of the canonically ordered root would not have matched.
    assert request["digest_hex"] != _digest_of(snapshot)


def test_verification_uses_unescaped_non_ascii_utf8(client):
    snapshot = _offline_snapshot()
    assert any(ord(ch) > 127 for ch in json.dumps(snapshot, ensure_ascii=False))
    resp = client.post(URL, json=_offline_request(snapshot))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}

    # A digest computed over ASCII-escaped bytes would not match.
    escaped = hashlib.sha256(
        _canonical_snapshot_bytes(snapshot, ascii_escape=True)
    ).hexdigest()
    assert escaped != _digest_of(snapshot)


def test_verification_is_deterministic_across_repeated_requests(client):
    content = _setup_export(client)
    request = _served_request(client, content)

    first = client.post(URL, json=request)
    second = client.post(URL, json=request)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json() == {"valid": True}


def test_verification_ignores_query_parameters(client):
    # The route takes no query parameters, and none can influence the
    # verdict: the same body verifies identically with or without them.
    content = _setup_export(client)
    request = _served_request(client, content)

    plain = client.post(URL, json=request)
    queried = client.post(f"{URL}?content_id=unknown", json=request)
    assert plain.status_code == queried.status_code == 200
    assert plain.json() == queried.json() == {"valid": True}


# --- Statelessness -------------------------------------------------------------


def test_verification_needs_no_persisted_resources(client):
    # The database is empty and every identifier is unknown to the service:
    # verification is decided by the request body alone.
    resp = client.post(URL, json=_offline_request())
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_writes_no_resources_or_audit_events(client, db_session):
    content = _setup_export(client)
    valid_request = _served_request(client, content)
    invalid_request = json.loads(json.dumps(valid_request))
    invalid_request["snapshot"]["content"]["title"] = "tampered"
    malformed_request = _offline_request()
    del malformed_request["snapshot"]["content"]

    models = (
        Actor,
        Content,
        Claim,
        EvidenceBundle,
        Attestation,
        ContentExportJob,
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


# --- Top-level structure and digest fields -------------------------------------


def test_verification_requires_exactly_the_three_top_level_fields(client):
    request = _offline_request()

    for field in ("snapshot", "digest_algorithm", "digest_hex"):
        incomplete = {k: v for k, v in request.items() if k != field}
        _assert_validation_error(client.post(URL, json=incomplete))

    augmented = dict(request)
    augmented["computed_digest_hex"] = request["digest_hex"]
    _assert_validation_error(client.post(URL, json=augmented))


def test_verification_rejects_non_object_and_empty_bodies(client):
    for body in (None, [], "snapshot", 42):
        _assert_validation_error(client.post(URL, json=body))
    _assert_validation_error(client.post(URL, json={}))


def test_verification_rejects_malformed_json(client):
    resp = client.post(
        URL,
        content=b'{"snapshot": ',
        headers={"content-type": "application/json"},
    )
    _assert_validation_error(resp)


def test_verification_requires_sha256_digest_algorithm(client):
    request = _offline_request()
    for algorithm in ("sha512", "SHA256", "sha-256", ""):
        bad = dict(request, digest_algorithm=algorithm)
        _assert_validation_error(client.post(URL, json=bad))


def test_verification_requires_64_lowercase_hex_digest(client):
    request = _offline_request()
    valid = request["digest_hex"]
    for digest in (
        valid.upper(),
        valid[:-1],
        valid + "0",
        "g" + valid[1:],
        " " + valid,
        123,
        None,
    ):
        bad = dict(request, digest_hex=digest)
        _assert_validation_error(client.post(URL, json=bad))


# --- Snapshot structure and fields ----------------------------------------------


def test_verification_snapshot_has_exactly_the_two_members(client):
    request = _offline_request()

    for member in ("content", "claims"):
        incomplete = json.loads(json.dumps(request))
        del incomplete["snapshot"][member]
        _assert_validation_error(client.post(URL, json=incomplete))

    augmented = json.loads(json.dumps(request))
    augmented["snapshot"]["attestations"] = []
    _assert_validation_error(client.post(URL, json=augmented))


def test_verification_snapshot_members_have_correct_types(client):
    request = _offline_request()
    for member, wrong in (
        ("content", []),
        ("content", None),
        ("claims", {}),
        ("claims", "clm"),
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
    extra_claim["snapshot"]["claims"][0]["payload"] = {"raw": True}
    cases.append(extra_claim)
    extra_bundle = json.loads(json.dumps(request))
    extra_bundle["snapshot"]["claims"][0]["evidence_bundles"][0]["evidence"] = "AAAA"
    cases.append(extra_bundle)

    for case in cases:
        _assert_validation_error(client.post(URL, json=case))


def test_verification_raw_material_is_never_echoed(client):
    # An undeclared raw-material field is a 422, and the material itself
    # appears nowhere in the error response.
    secret = "c2VjcmV0LXJhdy1ieXRlcw=="
    request = _offline_request()
    request["snapshot"]["claims"][0]["payload"] = {"raw": secret}

    resp = client.post(URL, json=request)
    _assert_validation_error(resp)
    assert secret not in resp.text


def test_verification_nested_views_require_every_public_field(client):
    request = _offline_request()

    missing_cases = []
    for member_path, field in (
        (("content",), "created_at"),
        (("claims", 0), "payload_digest_hex"),
        (("claims", 0), "evidence_bundles"),
        (("claims", 0, "evidence_bundles", 0), "metadata"),
    ):
        case = json.loads(json.dumps(request))
        target = case["snapshot"]
        for part in member_path:
            target = target[part]
        del target[field]
        missing_cases.append(case)

    for case in missing_cases:
        _assert_validation_error(client.post(URL, json=case))


def test_verification_nested_field_types_are_enforced(client):
    request = _offline_request()

    cases = []
    bad = json.loads(json.dumps(request))
    bad["snapshot"]["content"]["created_at"] = "not-a-timestamp"
    cases.append(bad)
    bad = json.loads(json.dumps(request))
    bad["snapshot"]["claims"][0]["payload_digest_hex"] = 123
    cases.append(bad)
    bad = json.loads(json.dumps(request))
    bad["snapshot"]["claims"][0]["evidence_bundles"] = {}
    cases.append(bad)
    bad = json.loads(json.dumps(request))
    bad["snapshot"]["claims"][0]["evidence_bundles"][0]["metadata"] = [
        "not",
        "an",
        "object",
    ]
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


def test_verification_claims_must_assert_the_snapshot_content(client):
    request = _offline_request()
    request["snapshot"]["claims"][0]["content_id"] = (
        "cnt_" + hashlib.sha256(b"other").hexdigest()
    )
    _assert_validation_error(client.post(URL, json=request))


def test_verification_bundles_must_attach_to_their_claim(client):
    request = _offline_request()
    request["snapshot"]["claims"][0]["evidence_bundles"][0]["claim_id"] = (
        "clm_" + hashlib.sha256(b"other").hexdigest()
    )
    _assert_validation_error(client.post(URL, json=request))


def test_verification_empty_claims_and_bundles_are_well_formed(client):
    snapshot = _offline_snapshot()
    # A claim with no evidence bundles is fine.
    snapshot["claims"][0]["evidence_bundles"] = []
    resp = client.post(URL, json=_offline_request(snapshot))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}

    # And so is a content with no claims at all.
    snapshot["claims"] = []
    resp = client.post(URL, json=_offline_request(snapshot))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_recomputed_digest_matches_served_export(client):
    # End-to-end agreement: the digest this endpoint computes for the served
    # snapshot equals the digest of the export body exactly as served.
    content = _setup_export(client)
    request = _served_request(client, content)
    served_digest = request["digest_hex"]
    request["digest_hex"] = "0" * 64

    resp = client.post(URL, json=request)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert _HEX64.fullmatch(body["computed_digest_hex"])
    assert body["computed_digest_hex"] == served_digest
