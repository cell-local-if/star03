"""Tests for the stateless offline content-export verification endpoint.

Covers POST /v1/content-export-verifications: the request body is exactly
{"snapshot", "digest_algorithm", "digest_hex"}; the algorithm is fixed at
``sha256`` and the claimed digest must be 64 lowercase hex characters. The
snapshot is exactly {"content", "claims"} with the existing public-view
fields -- content is the full public content view and claims lists only
claims directly asserting that content, each with its evidence bundles;
every claim's ``content_id`` must equal the content id and every bundle's
``claim_id`` must equal its enclosing claim's id. The digest is recomputed
over the raw received snapshot (root member order and claim/bundle array
order kept, nested object keys sorted by Unicode code point, compact
separators, unescaped non-ASCII, UTF-8, SHA-256). Structural, field,
association, or raw-material violations are 422 validation_error; a
structurally valid request whose computed digest differs is 200
{"valid": false, "computed_digest_hex": ...}; a match is exactly
{"valid": true}. The route is fully stateless: it needs no persisted
resources, ignores query conditions, and creates, modifies, and queries
nothing (no resource or audit rows, even for unknown ids). All fixtures
are deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json
import re

from sqlalchemy import func, select

from provenance.models import AuditEvent, Claim, Content, EvidenceBundle
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
    """Independently apply the content-export canonicalization.

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


def _served_request(client, content) -> dict:
    """A verification request built from the service's own export view."""
    snapshot = client.get(_export_url(content)).json()
    return {
        "snapshot": snapshot,
        "digest_algorithm": "sha256",
        "digest_hex": _digest_of(snapshot),
    }


def _offline_snapshot() -> dict:
    """A fully self-consistent content export fabricated without state."""
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
    return {
        "snapshot": snapshot,
        "digest_algorithm": "sha256",
        "digest_hex": _digest_of(snapshot),
    }


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- Verdicts -----------------------------------------------------------------


def test_verification_of_served_export_is_valid(client):
    create_actor(client)
    content = _create_content(client, "main")
    claim = _create_claim(client, content["id"], "a")
    _create_bundle(client, claim["id"], "a1")
    _create_claim(client, content["id"], "b")

    resp = client.post(URL, json=_served_request(client, content))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_empty_claims_is_valid(client):
    create_actor(client)
    content = _create_content(client, "lonely")
    resp = client.post(URL, json=_served_request(client, content))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_digest_mismatch_returns_computed_digest(client):
    create_actor(client)
    content = _create_content(client, "main")
    request = _served_request(client, content)
    snapshot = request["snapshot"]

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


def test_verification_claim_and_bundle_order_participates(client):
    create_actor(client)
    content = _create_content(client, "main")
    claim_a = _create_claim(client, content["id"], "a")
    claim_b = _create_claim(client, content["id"], "b")
    _create_bundle(client, claim_a["id"], "a1")
    _create_bundle(client, claim_a["id"], "a2")
    _create_bundle(client, claim_b["id"], "b1")
    request = _served_request(client, content)

    # Reversing the claim array is a different snapshot even though every
    # association stays consistent.
    reordered_claims = json.loads(json.dumps(request))
    reordered_claims["snapshot"]["claims"] = list(
        reversed(reordered_claims["snapshot"]["claims"])
    )
    resp = client.post(URL, json=reordered_claims)
    assert resp.status_code == 200, resp.text
    assert resp.json()["valid"] is False
    assert resp.json()["computed_digest_hex"] == _digest_of(
        reordered_claims["snapshot"]
    )

    # Reversing one claim's evidence bundles likewise changes the digest.
    reordered_bundles = json.loads(json.dumps(request))
    claim_a_export = next(
        c
        for c in reordered_bundles["snapshot"]["claims"]
        if c["id"] == claim_a["id"]
    )
    claim_a_export["evidence_bundles"] = list(
        reversed(claim_a_export["evidence_bundles"])
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
    create_actor(client)
    content = _create_content(client, "main")
    request = _served_request(client, content)

    first = client.post(URL, json=request)
    second = client.post(URL, json=request)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json() == {"valid": True}


def test_verification_query_conditions_do_not_affect_verdict(client):
    # The pure function ignores query parameters: they neither validate nor
    # change the verdict (unlike the read-only export GET route).
    request = _offline_request()
    for url in (
        URL + "?limit=10",
        URL + "?unknown=",
        URL + "?a=1&b=2",
    ):
        resp = client.post(url, json=request)
        assert resp.status_code == 200, (url, resp.text)
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
    claim = _create_claim(client, content["id"], "a")
    _create_bundle(client, claim["id"], "a1")
    valid_request = _served_request(client, content)
    invalid_request = json.loads(json.dumps(valid_request))
    invalid_request["snapshot"]["content"]["title"] = "tampered"
    malformed_request = _offline_request()
    del malformed_request["snapshot"]["claims"]

    models = (Content, Claim, EvidenceBundle, AuditEvent)

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


# --- Top-level structure -------------------------------------------------------


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
    for raw in ("", "{", "{not json", "null", "[]", '"snapshot"'):
        resp = client.post(
            URL, content=raw, headers={"content-type": "application/json"}
        )
        _assert_validation_error(resp)


def test_verification_requires_sha256_digest_algorithm(client):
    request = _offline_request()
    for algorithm in ("sha512", "SHA256", "sha-256", "", None, 256):
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
    augmented["snapshot"]["lineage"] = []
    _assert_validation_error(client.post(URL, json=augmented))


def test_verification_snapshot_members_have_correct_types(client):
    request = _offline_request()
    for member, wrong in (("content", []), ("claims", {}), ("claims", None)):
        bad = json.loads(json.dumps(request))
        bad["snapshot"][member] = wrong
        _assert_validation_error(client.post(URL, json=bad))


def test_verification_nested_views_forbid_undeclared_fields(client):
    secret = "SECRET-RAW-MATERIAL-MUST-NOT-ECHO"
    cases = []

    extra_content = json.loads(json.dumps(_offline_request()))
    extra_content["snapshot"]["content"]["data"] = secret
    cases.append(extra_content)

    extra_claim = json.loads(json.dumps(_offline_request()))
    extra_claim["snapshot"]["claims"][0]["payload"] = {"raw": secret}
    cases.append(extra_claim)

    extra_bundle = json.loads(json.dumps(_offline_request()))
    extra_bundle["snapshot"]["claims"][0]["evidence_bundles"][0]["evidence"] = secret
    cases.append(extra_bundle)

    for case in cases:
        resp = client.post(URL, json=case)
        _assert_validation_error(resp)
        # The undeclared raw material must never be echoed back.
        assert secret not in resp.text


def test_verification_nested_views_require_every_public_field(client):
    request = _offline_request()

    missing_cases = []
    for field in ("created_at", "actor_id"):
        case = json.loads(json.dumps(request))
        del case["snapshot"]["content"][field]
        missing_cases.append(case)

    claim_case = json.loads(json.dumps(request))
    del claim_case["snapshot"]["claims"][0]["payload_digest_hex"]
    missing_cases.append(claim_case)

    bundle_case = json.loads(json.dumps(request))
    del bundle_case["snapshot"]["claims"][0]["evidence_bundles"][0]["metadata"]
    missing_cases.append(bundle_case)

    for case in missing_cases:
        _assert_validation_error(client.post(URL, json=case))


def test_verification_nested_field_types_are_enforced(client):
    request = _offline_request()

    cases = []
    bad = json.loads(json.dumps(request))
    bad["snapshot"]["content"]["created_at"] = "not-a-timestamp"
    cases.append(bad)
    bad = json.loads(json.dumps(request))
    bad["snapshot"]["content"]["digest_hex"] = 123
    cases.append(bad)
    bad = json.loads(json.dumps(request))
    bad["snapshot"]["claims"][0]["payload_digest_hex"] = 123
    cases.append(bad)
    bad = json.loads(json.dumps(request))
    bad["snapshot"]["claims"] = ["not-an-object"]
    cases.append(bad)
    bad = json.loads(json.dumps(request))
    bad["snapshot"]["claims"][0]["evidence_bundles"] = {}
    cases.append(bad)
    bad = json.loads(json.dumps(request))
    bad["snapshot"]["claims"][0]["evidence_bundles"][0]["media_type"] = 7
    cases.append(bad)

    for case in cases:
        _assert_validation_error(client.post(URL, json=case))


def test_verification_metadata_must_have_finite_numbers(client):
    # Python's json emits NaN/Infinity literals; they have no canonical JSON
    # form and are rejected.
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


def test_verification_claim_must_assert_the_snapshot_content(client):
    request = _offline_request()
    request["snapshot"]["claims"][0]["content_id"] = (
        "cnt_" + hashlib.sha256(b"other-content").hexdigest()
    )
    _assert_validation_error(client.post(URL, json=request))

    # A foreign claim on another content is rejected even though its own
    # content id is a well-formed string.
    foreign = _offline_snapshot()
    foreign_claim = json.loads(json.dumps(foreign["claims"][0]))
    foreign_claim["id"] = "clm_" + hashlib.sha256(b"foreign").hexdigest()
    foreign_claim["content_id"] = (
        "cnt_" + hashlib.sha256(b"foreign-content").hexdigest()
    )
    foreign_claim["evidence_bundles"] = []
    foreign["claims"].append(foreign_claim)
    _assert_validation_error(client.post(URL, json=_offline_request(foreign)))


def test_verification_bundle_must_attach_to_its_claim(client):
    request = _offline_request()
    request["snapshot"]["claims"][0]["evidence_bundles"][0]["claim_id"] = (
        "clm_" + hashlib.sha256(b"other-claim").hexdigest()
    )
    _assert_validation_error(client.post(URL, json=request))


def test_verification_recomputed_digest_agrees_offline_and_served(client):
    # End-to-end agreement: the digest the endpoint reports for the served
    # snapshot equals an independent SHA-256 over the canonical bytes.
    create_actor(client)
    content = _create_content(client, "main")
    claim = _create_claim(client, content["id"], "a", claim_type="review")
    _create_bundle(client, claim["id"], "a1")
    request = _served_request(client, content)
    request["digest_hex"] = "0" * 64

    resp = client.post(URL, json=request)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert _HEX64.fullmatch(body["computed_digest_hex"])
    assert body["computed_digest_hex"] == _digest_of(request["snapshot"])

    # The fully offline fabricated snapshot reaches the same verdict with no
    # server state at all.
    offline = _offline_request()
    offline["digest_hex"] = "0" * 64
    resp = client.post(URL, json=offline)
    assert resp.status_code == 200, resp.text
    assert resp.json()["computed_digest_hex"] == _digest_of(
        offline["snapshot"]
    )
