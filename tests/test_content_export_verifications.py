"""Tests for the stateless offline content export verification endpoint.

Covers POST /v1/content-export-verifications: the request body is exactly
{"snapshot", "digest_algorithm", "digest_hex"}; the algorithm is fixed at
``sha256`` and the claimed digest must be 64 lowercase hex characters. The
snapshot is exactly {"content", "claims"} with the existing public-view
fields, each claim directly asserting the snapshot content and each bundle
attached to its enclosing claim. Structural, field, association, or
raw-material violations are 422 validation_error; a structurally valid
request whose recomputed digest differs is 200
{"valid": false, "computed_digest_hex": ...}; a match is exactly
{"valid": true}. The digest keeps root member order and array order, sorts
nested object keys by Unicode code point, uses compact separators,
unescaped non-ASCII, and UTF-8; an empty claims array is covered by the
digest. The route is fully stateless: it needs no persisted resources and
creates, modifies, and queries nothing (no resource, snapshot, audit, or
log rows, even for unknown ids), and it saves or echoes no raw material.
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
    EvidenceBundle,
)
from tests.helpers import DIGEST_A, DIGEST_B, DIGEST_C
from tests.test_content_export import (
    _create_bundle,
    _create_claim,
    _create_content,
)
from tests.helpers import create_actor

URL = "/v1/content-export-verifications"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_snapshot_bytes(snapshot: dict, *, ascii_escape: bool = False) -> bytes:
    """Independently apply the snapshot canonicalization to a parsed snapshot.

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


def _offline_content(content_id: str | None = None) -> dict:
    return {
        "id": content_id
        or ("cnt_" + hashlib.sha256(b"offline-content").hexdigest()),
        "digest_algorithm": "sha256",
        "digest_hex": DIGEST_A,
        "media_type": "image/png",
        "title": "offline 快照",
        "actor_id": "org-offline",
        "created_at": "2026-01-02T03:04:05Z",
    }


def _offline_claim(content_id: str, *, marker: bytes = b"offline-claim") -> dict:
    return {
        "id": "clm_" + hashlib.sha256(marker).hexdigest(),
        "content_id": content_id,
        "actor_id": "org-offline",
        "claim_type": "authorship",
        "payload_digest_algorithm": "sha256",
        "payload_digest_hex": DIGEST_B,
        "created_at": "2026-01-02T03:04:06Z",
        "evidence_bundles": [],
    }


def _offline_bundle(claim_id: str, *, marker: bytes = b"offline-bundle") -> dict:
    return {
        "id": "evb_" + hashlib.sha256(marker).hexdigest(),
        "claim_id": claim_id,
        "evidence_type": "raw_capture",
        "digest_algorithm": "sha256",
        "digest_hex": DIGEST_C,
        "media_type": "image/jpeg",
        "metadata": {"origin": "offline", "标签": {"中": True}},
        "created_at": "2026-01-02T03:04:07Z",
    }


def _offline_snapshot(*, empty_claims: bool = False) -> dict:
    """A fully self-consistent snapshot fabricated without any service state."""
    content = _offline_content()
    claims: list[dict] = []
    if not empty_claims:
        claim = _offline_claim(content["id"])
        claim["evidence_bundles"] = [_offline_bundle(claim["id"])]
        claims.append(claim)
    return {"content": content, "claims": claims}


def _request(snapshot: dict, *, digest_hex: str | None = None) -> dict:
    return {
        "snapshot": snapshot,
        "digest_algorithm": "sha256",
        "digest_hex": _digest_of(snapshot) if digest_hex is None else digest_hex,
    }


def _offline_request(*, empty_claims: bool = False) -> dict:
    return _request(_offline_snapshot(empty_claims=empty_claims))


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def _setup_served_export(client) -> dict:
    """Create one content with two ordered claims/bundles and GET its export."""
    create_actor(client)
    content = _create_content(client, "main")
    other = _create_content(client, "other")
    claim_a = _create_claim(client, content["id"], "a")
    claim_b = _create_claim(client, content["id"], "b", claim_type="review")
    foreign = _create_claim(client, other["id"], "foreign")
    _create_bundle(client, claim_a["id"], "a2")
    _create_bundle(client, claim_a["id"], "a1")
    _create_bundle(client, foreign["id"], "foreign")
    # claim_b deliberately has no bundles.
    resp = client.get(f"/v1/contents/{content['id']}/export")
    assert resp.status_code == 200, resp.text
    return resp.json()


# --- Verdicts -----------------------------------------------------------------


def test_verification_of_served_export_snapshot_is_valid(client):
    snapshot = _setup_served_export(client)
    resp = client.post(URL, json=_request(snapshot))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_digest_mismatch_returns_computed_digest(client):
    snapshot = _setup_served_export(client)

    # Flip the last hex character of the claimed digest.
    claimed = _digest_of(snapshot)
    bad_claim = claimed[:-1] + ("0" if claimed[-1] != "0" else "1")

    resp = client.post(URL, json=_request(snapshot, digest_hex=bad_claim))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"valid", "computed_digest_hex"}
    assert body["valid"] is False
    assert body["computed_digest_hex"] == claimed
    assert _HEX64.fullmatch(body["computed_digest_hex"])


def test_verification_tampered_snapshot_is_invalid(client):
    snapshot = _setup_served_export(client)
    claimed_digest = _digest_of(snapshot)
    snapshot["content"]["title"] = "forged title"

    resp = client.post(URL, json=_request(snapshot, digest_hex=claimed_digest))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert body["computed_digest_hex"] == _digest_of(snapshot)
    assert body["computed_digest_hex"] != claimed_digest


def test_verification_claim_order_participates(client):
    snapshot = _setup_served_export(client)
    claimed_digest = _digest_of(snapshot)

    reordered = json.loads(json.dumps(snapshot))
    reordered["claims"] = list(reversed(reordered["claims"]))

    resp = client.post(URL, json=_request(reordered, digest_hex=claimed_digest))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert body["computed_digest_hex"] == _digest_of(reordered)


def test_verification_bundle_order_participates(client):
    snapshot = _setup_served_export(client)
    claimed_digest = _digest_of(snapshot)

    reordered = json.loads(json.dumps(snapshot))
    reordered["claims"][0]["evidence_bundles"] = list(
        reversed(reordered["claims"][0]["evidence_bundles"])
    )

    resp = client.post(URL, json=_request(reordered, digest_hex=claimed_digest))
    assert resp.status_code == 200, resp.text
    assert resp.json()["valid"] is False
    assert resp.json()["computed_digest_hex"] == _digest_of(reordered)


def test_verification_root_member_order_is_preserved(client):
    snapshot = _offline_snapshot()
    # Root members keep the request's order: a digest over the reordered root
    # verifies against that same order.
    reordered_root = {"claims": snapshot["claims"], "content": snapshot["content"]}

    resp = client.post(URL, json=_request(reordered_root))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}

    # The digest of the canonically ordered root would not have matched.
    assert _digest_of(reordered_root) != _digest_of(snapshot)


def test_verification_timestamp_spelling_participates(client):
    # The digest commits to the raw received spelling: an equivalent RFC 3339
    # instant spelled differently changes the recomputed digest.
    snapshot_z = _offline_snapshot()
    snapshot_offset = json.loads(json.dumps(snapshot_z))
    snapshot_offset["content"]["created_at"] = "2026-01-02T03:04:05+00:00"

    resp = client.post(
        URL, json=_request(snapshot_offset, digest_hex=_digest_of(snapshot_offset))
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}

    mismatch = client.post(
        URL, json=_request(snapshot_offset, digest_hex=_digest_of(snapshot_z))
    )
    assert mismatch.status_code == 200
    assert mismatch.json()["valid"] is False


def test_verification_uses_unescaped_non_ascii_utf8(client):
    snapshot = _offline_snapshot()
    assert any(
        ord(ch) > 127 for ch in json.dumps(snapshot, ensure_ascii=False)
    )
    resp = client.post(URL, json=_request(snapshot))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}

    # A digest computed over ASCII-escaped bytes would not match.
    escaped = hashlib.sha256(
        _canonical_snapshot_bytes(snapshot, ascii_escape=True)
    ).hexdigest()
    assert escaped != _digest_of(snapshot)


def test_verification_metadata_number_participates(client):
    snapshot = _offline_snapshot()
    snapshot["content"] = json.loads(json.dumps(snapshot["content"]))
    snapshot["claims"][0]["evidence_bundles"][0]["metadata"]["score"] = 1.5
    resp = client.post(URL, json=_request(snapshot))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_is_deterministic_across_repeated_requests(client):
    snapshot = _offline_snapshot()
    request = _request(snapshot)
    first = client.post(URL, json=request)
    second = client.post(URL, json=request)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json() == {"valid": True}


# --- Wire format ---------------------------------------------------------------


def test_verification_success_body_is_compact_json_with_one_newline(client):
    request = _offline_request()
    resp = client.post(URL, json=request)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"
    assert resp.content == b'{"valid":true}\n'


def test_verification_mismatch_body_is_compact_json_with_one_newline(client):
    snapshot = _offline_snapshot()
    resp = client.post(URL, json=_request(snapshot, digest_hex="0" * 64))
    assert resp.status_code == 200
    computed = _digest_of(snapshot)
    assert resp.content == (
        f'{{"valid":false,"computed_digest_hex":"{computed}"}}\n'
    ).encode("utf-8")


# --- Empty claims --------------------------------------------------------------


def test_verification_empty_claims_list_is_valid_and_covers_empty_array(client):
    snapshot = _offline_snapshot(empty_claims=True)
    resp = client.post(URL, json=_request(snapshot))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}

    # The digest genuinely covers the empty array: a wrong claim is detected.
    mismatch = client.post(URL, json=_request(snapshot, digest_hex="0" * 64))
    assert mismatch.status_code == 200
    body = mismatch.json()
    assert body["valid"] is False
    assert body["computed_digest_hex"] == _digest_of(snapshot)


def test_verification_empty_evidence_bundles_list_is_valid(client):
    snapshot = _offline_snapshot()
    snapshot["claims"][0]["evidence_bundles"] = []
    resp = client.post(URL, json=_request(snapshot))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


# --- Statelessness -------------------------------------------------------------


def test_verification_needs_no_persisted_resources(client):
    # The database is empty and every identifier is unknown to the service:
    # verification is decided by the request body alone.
    resp = client.post(URL, json=_offline_request())
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}
    assert client.post(URL, json=_offline_request(empty_claims=True)).status_code == 200


def test_verification_writes_no_resources_or_audit_events(client, db_session):
    snapshot = _setup_served_export(client)
    valid_request = _request(snapshot)
    invalid_request = _request(
        json.loads(json.dumps(snapshot)), digest_hex="0" * 64
    )
    malformed_request = _offline_request()
    del malformed_request["snapshot"]["claims"]

    models = (
        Actor,
        Content,
        Claim,
        EvidenceBundle,
        Attestation,
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


# --- Top-level structure --------------------------------------------------------


def test_verification_requires_exactly_the_three_top_level_fields(client):
    request = _offline_request()

    for field in ("snapshot", "digest_algorithm", "digest_hex"):
        incomplete = {k: v for k, v in request.items() if k != field}
        _assert_validation_error(client.post(URL, json=incomplete))

    for extra_field in (
        "computed_digest_hex",
        "payload",
        "content",
        "evidence",
        "signature",
        "manifest",
    ):
        augmented = dict(request)
        augmented[extra_field] = "raw-material"
        _assert_validation_error(client.post(URL, json=augmented))


def test_verification_rejects_non_object_and_empty_bodies(client):
    for body in (None, [], "snapshot", 42):
        _assert_validation_error(client.post(URL, json=body))
    _assert_validation_error(client.post(URL, json={}))
    _assert_validation_error(
        client.post(
            URL, content="", headers={"content-type": "application/json"}
        )
    )


def test_verification_rejects_malformed_json(client):
    resp = client.post(
        URL,
        content='{"snapshot":',
        headers={"content-type": "application/json"},
    )
    _assert_validation_error(resp)


def test_verification_requires_sha256_digest_algorithm(client):
    request = _offline_request()
    for algorithm in ("sha512", "SHA256", "sha-256", "", 256, None):
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
        " " + valid[:-1],
        123,
        None,
    ):
        bad = dict(request, digest_hex=digest)
        _assert_validation_error(client.post(URL, json=bad))


def test_verification_rejects_any_query_parameter(client):
    request = _offline_request()
    for url in (
        URL + "?x=1",
        URL + "?digest_hex=" + "0" * 64,
        URL + "?a=1&b=2",
    ):
        resp = client.post(url, json=request)
        _assert_validation_error(resp)


# --- Snapshot structure and fields ----------------------------------------------


def test_verification_snapshot_has_exactly_the_two_members(client):
    request = _offline_request()

    for member in ("content", "claims"):
        incomplete = json.loads(json.dumps(request))
        del incomplete["snapshot"][member]
        _assert_validation_error(client.post(URL, json=incomplete))

    augmented = json.loads(json.dumps(request))
    augmented["snapshot"]["lineage"] = []
    _assert_validation_error(client.post(URL, json=augmented))


def test_verification_snapshot_members_have_correct_types(client):
    request = _offline_request()
    for member, wrong in (("content", []), ("claims", {}), ("claims", None)):
        bad = json.loads(json.dumps(request))
        bad["snapshot"][member] = wrong
        _assert_validation_error(client.post(URL, json=bad))

    # A claim entry must itself be an object.
    bad = json.loads(json.dumps(request))
    bad["snapshot"]["claims"] = ["not-an-object"]
    _assert_validation_error(client.post(URL, json=bad))


def test_verification_nested_views_forbid_undeclared_fields(client):
    request = _offline_request()

    extra_content = json.loads(json.dumps(request))
    extra_content["snapshot"]["content"]["data"] = "SECRETBYTES1"
    _assert_validation_error(client.post(URL, json=extra_content))

    extra_claim = json.loads(json.dumps(request))
    extra_claim["snapshot"]["claims"][0]["payload"] = {"raw": "SECRETBYTES2"}
    _assert_validation_error(client.post(URL, json=extra_claim))

    extra_bundle = json.loads(json.dumps(request))
    extra_bundle["snapshot"]["claims"][0]["evidence_bundles"][0][
        "evidence"
    ] = "SECRETBYTES3"
    _assert_validation_error(client.post(URL, json=extra_bundle))

    extra_signature = json.loads(json.dumps(request))
    extra_signature["snapshot"]["claims"][0]["signature"] = "SECRETBYTES4"
    _assert_validation_error(client.post(URL, json=extra_signature))


def test_verification_validation_errors_do_not_echo_raw_material(client):
    request = _offline_request()
    request["snapshot"]["claims"][0]["payload"] = {"secret": "SECRETVALUE"}
    resp = client.post(URL, json=request)
    _assert_validation_error(resp)
    assert "SECRETVALUE" not in resp.text


def test_verification_nested_views_require_every_public_field(client):
    request = _offline_request()
    snapshot = request["snapshot"]

    missing_cases = []
    for member, field in (
        ("content", "created_at"),
        ("content", "actor_id"),
    ):
        case = json.loads(json.dumps(request))
        del case["snapshot"][member][field]
        missing_cases.append(case)
    claim_case = json.loads(json.dumps(request))
    del claim_case["snapshot"]["claims"][0]["payload_digest_hex"]
    missing_cases.append(claim_case)
    bundle_case = json.loads(json.dumps(request))
    del bundle_case["snapshot"]["claims"][0]["evidence_bundles"][0][
        "media_type"
    ]
    missing_cases.append(bundle_case)

    for case in missing_cases:
        _assert_validation_error(client.post(URL, json=case))
    assert snapshot  # the source request itself was well-formed


def test_verification_nested_field_types_are_enforced(client):
    request = _offline_request()

    bad = json.loads(json.dumps(request))
    bad["snapshot"]["content"]["created_at"] = "not-a-timestamp"
    _assert_validation_error(client.post(URL, json=bad))

    bad = json.loads(json.dumps(request))
    bad["snapshot"]["content"]["digest_hex"] = 123
    _assert_validation_error(client.post(URL, json=bad))

    bad = json.loads(json.dumps(request))
    bad["snapshot"]["claims"][0]["claim_type"] = 7
    _assert_validation_error(client.post(URL, json=bad))

    bad = json.loads(json.dumps(request))
    bad["snapshot"]["claims"][0]["evidence_bundles"] = {}
    _assert_validation_error(client.post(URL, json=bad))

    bad = json.loads(json.dumps(request))
    bad["snapshot"]["claims"][0]["evidence_bundles"][0]["metadata"] = [
        "not",
        "an",
        "object",
    ]
    _assert_validation_error(client.post(URL, json=bad))


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


def test_verification_content_title_may_be_null(client):
    snapshot = _offline_snapshot()
    snapshot["content"]["title"] = None
    resp = client.post(URL, json=_request(snapshot))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


# --- Associations ----------------------------------------------------------------


def test_verification_claim_must_directly_assert_snapshot_content(client):
    request = _offline_request()
    other_content = "cnt_" + hashlib.sha256(b"other-content").hexdigest()
    request["snapshot"]["claims"][0]["content_id"] = other_content
    _assert_validation_error(client.post(URL, json=request))


def test_verification_bundle_must_attach_to_its_enclosing_claim(client):
    request = _offline_request()
    other_claim = "clm_" + hashlib.sha256(b"other-claim").hexdigest()
    request["snapshot"]["claims"][0]["evidence_bundles"][0][
        "claim_id"
    ] = other_claim
    _assert_validation_error(client.post(URL, json=request))


def test_verification_multiple_consistent_claims_and_bundles_are_valid(client):
    content = _offline_content()
    claim_a = _offline_claim(content["id"], marker=b"claim-a")
    claim_b = _offline_claim(content["id"], marker=b"claim-b")
    claim_a["evidence_bundles"] = [
        _offline_bundle(claim_a["id"], marker=b"bundle-a1"),
        _offline_bundle(claim_a["id"], marker=b"bundle-a2"),
    ]
    claim_b["evidence_bundles"] = [
        _offline_bundle(claim_b["id"], marker=b"bundle-b1"),
    ]
    snapshot = {"content": content, "claims": [claim_a, claim_b]}

    resp = client.post(URL, json=_request(snapshot))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_digest_mismatch_is_not_a_validation_error(client):
    # Association/structure failures are 422, but a digest mismatch on a
    # structurally valid snapshot is always the 200 false verdict.
    snapshot = _offline_snapshot()
    resp = client.post(URL, json=_request(snapshot, digest_hex="0" * 64))
    assert resp.status_code == 200
    assert resp.json()["valid"] is False
