"""Tests for the stateless offline content-export verification endpoint.

Covers POST /v1/content-export-verifications: the request body is exactly
{"snapshot", "digest_algorithm", "digest_hex"}; the algorithm is fixed at
``sha256`` and the claimed digest must be 64 lowercase hex characters. The
snapshot is exactly {"content", "claims"} with the existing public-view
fields (each claim carrying its ``evidence_bundles``); every claim must
directly assert the snapshot's content and every bundle must belong to its
enclosing claim. Structural, field, association, or raw-material violations
are 422 validation_error; a structurally valid request whose digest differs
is 200 {"valid": false, "computed_digest_hex": ...}; a match is exactly
{"valid": true}. The route is fully stateless: it needs no persisted
resources and creates, modifies, and queries nothing (no resource or audit
rows, even for unknown ids). All fixtures are deterministic and offline.
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
from tests.helpers import DIGEST_A, DIGEST_B, DIGEST_C, create_actor
from tests.test_content_export import (
    _create_bundle,
    _create_claim,
    _create_content,
)

URL = "/v1/content-export-verifications"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _export_url(content) -> str:
    return f"/v1/contents/{content['id']}/export"


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


def _request_for(snapshot: dict, digest_hex: str | None = None) -> dict:
    return {
        "snapshot": snapshot,
        "digest_algorithm": "sha256",
        "digest_hex": _digest_of(snapshot) if digest_hex is None else digest_hex,
    }


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


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- Verdicts -----------------------------------------------------------------


def test_verification_of_served_export_is_valid(client):
    create_actor(client)
    content = _create_content(client, "main")
    claim = _create_claim(client, content["id"], "a")
    _create_bundle(client, claim["id"], "a1")

    resp = client.post(URL, json=_served_request(client, content))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_digest_mismatch_returns_computed_digest(client):
    create_actor(client)
    content = _create_content(client, "main")
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
    create_actor(client)
    content = _create_content(client, "main")
    request = _served_request(client, content)
    # The claimed digest still commits to the untouched snapshot.
    request["snapshot"]["content"]["title"] = "forged title"

    resp = client.post(URL, json=request)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert body["computed_digest_hex"] == _digest_of(request["snapshot"])
    assert body["computed_digest_hex"] != request["digest_hex"]


def test_verification_array_order_participates(client):
    create_actor(client)
    content = _create_content(client, "main")
    _create_claim(client, content["id"], "a")
    _create_claim(client, content["id"], "b", claim_type="review")
    request = _served_request(client, content)

    # Arrays keep their order: the same claims in another order are a
    # different snapshot and no longer match the claimed digest.
    reordered = json.loads(json.dumps(request))
    reordered["snapshot"]["claims"] = list(reversed(reordered["snapshot"]["claims"]))
    resp = client.post(URL, json=reordered)
    assert resp.status_code == 200, resp.text
    assert resp.json()["valid"] is False
    assert resp.json()["computed_digest_hex"] == _digest_of(reordered["snapshot"])


def test_verification_root_member_order_is_preserved(client):
    snapshot = _offline_snapshot()
    # Root members keep the request's order: a digest computed over a
    # differently ordered root verifies against that same order.
    reordered_root = {key: snapshot[key] for key in ("claims", "content")}
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
    create_actor(client)
    content = _create_content(client, "main")
    request = _served_request(client, content)

    first = client.post(URL, json=request)
    second = client.post(URL, json=request)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json() == {"valid": True}
    assert first.content == second.content


def test_verification_response_is_compact_json_with_one_newline(client):
    resp = client.post(URL, json=_offline_request())
    assert resp.status_code == 200, resp.text
    assert resp.content == b'{"valid":true}\n'

    mismatch = _offline_request()
    mismatch["digest_hex"] = "0" * 64
    resp = client.post(URL, json=mismatch)
    assert resp.status_code == 200, resp.text
    raw = resp.content
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b", " not in raw and b'": ' not in raw
    assert json.loads(raw.decode("utf-8")) == {
        "valid": False,
        "computed_digest_hex": _digest_of(mismatch["snapshot"]),
    }


# --- Empty claims ---------------------------------------------------------------


def test_verification_empty_claims_list_is_legal(client):
    snapshot = _offline_snapshot()
    snapshot["claims"] = []

    resp = client.post(URL, json=_offline_request(snapshot))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}

    # The digest still covers the empty array: a digest computed over a
    # snapshot without the claims member at all would differ.
    assert _digest_of(snapshot) != _digest_of({"content": snapshot["content"]})


def test_verification_served_export_with_empty_claims(client):
    create_actor(client)
    content = _create_content(client, "lonely")

    resp = client.post(URL, json=_served_request(client, content))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


# --- Statelessness -------------------------------------------------------------


def test_verification_needs_no_persisted_resources(client):
    # The database is empty and every identifier is unknown to the service:
    # verification is decided by the request body alone.
    resp = client.post(URL, json=_offline_request())
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_writes_no_resources_or_audit_events(client, db_session):
    create_actor(client)
    content = _create_content(client, "main")
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


# --- Top-level structure and digest fields --------------------------------------


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


def test_verification_rejects_any_query_parameter(client):
    request = _offline_request()
    for url in (
        f"{URL}?limit=10",
        f"{URL}?unknown=",
        f"{URL}?a=1&b=2",
        f"{URL}?digest_hex={request['digest_hex']}",
    ):
        resp = client.post(url, json=request)
        assert resp.status_code == 422, (url, resp.text)
        assert resp.json()["error"]["code"] == "validation_error"


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
    for member, wrong in (
        ("content", []),
        ("content", None),
        ("claims", {}),
        ("claims", "claims"),
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
    extra_content_raw = json.loads(json.dumps(request))
    extra_content_raw["snapshot"]["content"]["content"] = "AAAA"
    cases.append(extra_content_raw)
    extra_claim = json.loads(json.dumps(request))
    extra_claim["snapshot"]["claims"][0]["payload"] = {"raw": True}
    cases.append(extra_claim)
    extra_bundle = json.loads(json.dumps(request))
    extra_bundle["snapshot"]["claims"][0]["evidence_bundles"][0]["evidence"] = "AAAA"
    cases.append(extra_bundle)
    extra_signature = json.loads(json.dumps(request))
    extra_signature["snapshot"]["claims"][0]["signature"] = "AAAA"
    cases.append(extra_signature)

    for case in cases:
        resp = client.post(URL, json=case)
        _assert_validation_error(resp)
        # Raw material is never echoed back.
        assert "AAAA" not in resp.text


def test_verification_nested_views_require_every_public_field(client):
    request = _offline_request()

    missing_cases = []
    for path in (
        ("content", "created_at"),
        ("claims", "payload_digest_hex"),
    ):
        case = json.loads(json.dumps(request))
        if path[0] == "content":
            del case["snapshot"]["content"][path[1]]
        else:
            del case["snapshot"]["claims"][0][path[1]]
        missing_cases.append(case)
    bundle_case = json.loads(json.dumps(request))
    del bundle_case["snapshot"]["claims"][0]["evidence_bundles"][0]["metadata"]
    missing_cases.append(bundle_case)
    claim_bundles_case = json.loads(json.dumps(request))
    del claim_bundles_case["snapshot"]["claims"][0]["evidence_bundles"]
    missing_cases.append(claim_bundles_case)

    for case in missing_cases:
        _assert_validation_error(client.post(URL, json=case))
    assert request["snapshot"]  # the source request itself was well-formed


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
    bad["snapshot"]["claims"][0]["evidence_bundles"][0]["metadata"] = [
        "not",
        "an",
        "object",
    ]
    cases.append(bad)
    bad = json.loads(json.dumps(request))
    bad["snapshot"]["claims"][0]["evidence_bundles"] = {}
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


def test_verification_bundles_must_belong_to_their_claim(client):
    request = _offline_request()
    request["snapshot"]["claims"][0]["evidence_bundles"][0]["claim_id"] = (
        "clm_" + hashlib.sha256(b"other").hexdigest()
    )
    _assert_validation_error(client.post(URL, json=request))


def test_verification_claim_without_bundles_is_legal(client):
    snapshot = _offline_snapshot()
    snapshot["claims"][0]["evidence_bundles"] = []

    resp = client.post(URL, json=_offline_request(snapshot))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_recomputed_digest_matches_served_export(client):
    # End-to-end agreement: the digest this endpoint computes for the served
    # snapshot equals the independently canonicalized digest of that export.
    create_actor(client)
    content = _create_content(client, "main")
    claim = _create_claim(client, content["id"], "a")
    _create_bundle(client, claim["id"], "a1")
    request = _served_request(client, content)
    request["digest_hex"] = "0" * 64

    resp = client.post(URL, json=request)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert _HEX64.fullmatch(body["computed_digest_hex"])
    assert body["computed_digest_hex"] == _digest_of(request["snapshot"])
