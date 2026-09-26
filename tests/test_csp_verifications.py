"""Tests for the stateless correction checkpoint verification route.

Covers POST /v1/csp/verify: the request body is exactly
{"checkpoint", "corrections"}; the checkpoint is exactly
{"checkpoint_version", "digest_algorithm", "correction_count",
"corrections_digest_hex"} with the version pinned to
"provenance-csp-checkpoint-v1", the algorithm to "sha256", the count a
non-negative integer, and the digest 64 lowercase hex characters. The
corrections array must contain exactly ``correction_count`` elements,
each exactly the exported five-field correction public view
(``id``, ``superseded_claim_id``, ``replacement_claim_id``, ``reason``,
strict RFC 3339 UTC ``created_at``; non-empty identifiers and reason;
no undeclared member). The digest is the SHA-256 of the canonical
corrections array (array order preserved, nested object keys sorted by
Unicode code point, compact separators, non-ASCII unescaped, UTF-8)
computed over the corrections exactly as received. Any structural,
field, type, count, timestamp, digest-input, or malformed-JSON
violation is a 422 validation_error; a structurally valid request whose
digest differs is 200 {"valid": false, "computed_digest_hex": ...}; a
match is exactly {"valid": true}. The route accepts no query
parameters and is fully stateless: it needs no persisted correction and
creates, modifies, logs, and queries nothing, so unknown resources,
repeated requests, and differing local state verify identically.
Non-POST methods are 405. All fixtures are deterministic and offline.
"""

from __future__ import annotations

import copy
import hashlib
import json

from sqlalchemy import func, select

from provenance.models import (
    Actor,
    AuditEvent,
    Claim,
    ClaimSupersession,
    Content,
)
from tests.test_csp_packages import _world

URL = "/v1/csp/verify"
PACKAGE_PATH = "/v1/csp/package"
CHECKPOINT_VERSION = "provenance-csp-checkpoint-v1"

CHECKPOINT_FIELDS = {
    "checkpoint_version",
    "digest_algorithm",
    "correction_count",
    "corrections_digest_hex",
}
CORRECTION_FIELDS = {
    "id",
    "superseded_claim_id",
    "replacement_claim_id",
    "reason",
    "created_at",
}


def _digest_of(corrections: list) -> str:
    """Independently canonicalize the corrections array and digest it."""
    canonical = json.dumps(
        corrections,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _correction(
    *,
    id="csp_01hztest000000000000000001",
    superseded_claim_id="clm_01hztest00000000000000001",
    replacement_claim_id="clm_01hztest00000000000000002",
    reason="corrected source claim",
    created_at="2026-01-02T03:04:05Z",
):
    return {
        "id": id,
        "superseded_claim_id": superseded_claim_id,
        "replacement_claim_id": replacement_claim_id,
        "reason": reason,
        "created_at": created_at,
    }


def _offline_corrections() -> list:
    """A self-contained correction sequence fabricated without state."""
    return [
        _correction(
            id="csp_01hztest000000000000000001",
            superseded_claim_id="clm_01hztest00000000000000001",
            replacement_claim_id="clm_01hztest00000000000000002",
            reason="corrected source claim",
            created_at="2026-01-02T03:04:05Z",
        ),
        _correction(
            id="csp_01hztest000000000000000002",
            superseded_claim_id="clm_01hztest00000000000000002",
            replacement_claim_id="clm_01hztest00000000000000003",
            reason="new evidence attached 勘误",
            created_at="2026-01-02T03:04:06.500+00:00",
        ),
        _correction(
            id="csp_01hztest000000000000000003",
            superseded_claim_id="clm_01hztest00000000000000004",
            replacement_claim_id="clm_01hztest00000000000000005",
            reason="retracted statement",
            created_at="2026-01-03T00:00:00Z",
        ),
    ]


def _request(corrections: list | None = None, *, digest=None) -> dict:
    corrections = _offline_corrections() if corrections is None else corrections
    return {
        "checkpoint": {
            "checkpoint_version": CHECKPOINT_VERSION,
            "digest_algorithm": "sha256",
            "correction_count": len(corrections),
            "corrections_digest_hex": _digest_of(corrections)
            if digest is None
            else digest,
        },
        "corrections": corrections,
    }


def _served_request(client, **params) -> dict:
    package = client.get(PACKAGE_PATH, params=params).json()
    return {
        "checkpoint": package["checkpoint"],
        "corrections": package["corrections"],
    }


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- Verdicts -----------------------------------------------------------------------


def test_verification_of_offline_package_is_valid_on_empty_database(client):
    # No resource exists locally and every identifier is unknown: the
    # verdict is decided by the request body alone.
    resp = client.post(URL, json=_request())
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_of_served_package_is_valid(client):
    _world(client)
    resp = client.post(URL, json=_served_request(client))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_of_empty_snapshot_is_valid(client):
    for payload in (
        _request(corrections=[]),
        _served_request(client, reason="nobody"),
    ):
        assert payload["checkpoint"]["correction_count"] == 0
        assert payload["checkpoint"]["corrections_digest_hex"] == (
            hashlib.sha256(b"[]").hexdigest()
        )
        resp = client.post(URL, json=payload)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"valid": True}


def test_digest_mismatch_returns_exactly_valid_false_and_computed_digest(client):
    request = _request()
    claimed = request["checkpoint"]["corrections_digest_hex"]
    request["checkpoint"]["corrections_digest_hex"] = claimed[:-1] + (
        "0" if claimed[-1] != "0" else "1"
    )
    resp = client.post(URL, json=request)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"valid", "computed_digest_hex"}
    assert body["valid"] is False
    assert body["computed_digest_hex"] == _digest_of(_offline_corrections())
    assert body["computed_digest_hex"] == claimed


def test_mismatch_response_is_compact_json_with_one_newline(client):
    request = _request()
    request["checkpoint"]["corrections_digest_hex"] = "0" * 64
    raw = client.post(URL, json=request).content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    body = json.loads(raw)
    assert body["valid"] is False
    assert len(body["computed_digest_hex"]) == 64

    matched = client.post(URL, json=_request()).content
    assert matched == b'{"valid":true}\n'


def test_tampered_correction_is_invalid(client):
    request = _request()
    request["corrections"][0]["reason"] = "forged"
    resp = client.post(URL, json=request)
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert body["computed_digest_hex"] == _digest_of(request["corrections"])


def test_array_order_participates_in_the_digest(client):
    request = _request()
    reordered = copy.deepcopy(request)
    reordered["corrections"] = list(reversed(reordered["corrections"]))
    resp = client.post(URL, json=reordered)
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert body["computed_digest_hex"] == _digest_of(reordered["corrections"])


def test_nested_object_key_spelling_is_irrelevant_to_the_digest(client):
    # The canonical form sorts nested object keys by Unicode code point:
    # reserializing each correction with keys in another order must still
    # match, while the array order itself stays significant.
    request = _request()
    reshuffled = copy.deepcopy(request)
    reshuffled["corrections"] = [
        {k: correction[k] for k in sorted(correction, reverse=True)}
        for correction in reshuffled["corrections"]
    ]
    resp = client.post(URL, json=reshuffled)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_non_ascii_is_canonicalized_unescaped(client):
    request = _request()
    assert any(
        ord(ch) > 127
        for ch in json.dumps(request["corrections"], ensure_ascii=False)
    )
    assert client.post(URL, json=request).json() == {"valid": True}
    escaped = json.dumps(
        request["corrections"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    assert hashlib.sha256(escaped).hexdigest() != _digest_of(
        request["corrections"]
    )


def test_equivalent_utc_spellings_are_accepted(client):
    # Both Z and +00:00 are strict RFC 3339 UTC spellings; the digest
    # commits to the exact received spelling (the route normalizes
    # nothing), but each is a legal digest input and verifies.
    for spelling in (
        "2026-01-02T03:04:05Z",
        "2026-01-02T03:04:05+00:00",
        "2026-01-02T03:04:05.000Z",
    ):
        corrections = [_correction(created_at=spelling)]
        assert client.post(URL, json=_request(corrections)).json() == {
            "valid": True
        }


# --- Statelessness ------------------------------------------------------------------


def test_verification_is_deterministic_across_repeated_requests(client):
    _world(client)
    request = _served_request(client)
    first = client.post(URL, json=request)
    second = client.post(URL, json=request)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content == b'{"valid":true}\n'


def test_local_state_changes_do_not_change_the_verdict(client):
    # First verify a served package, then add more corrections: the
    # earlier package must still verify byte-for-byte because the route
    # never queries local state.
    corrections = _world(client)
    package = _served_request(client)
    assert client.post(URL, json=package).json() == {"valid": True}

    # Fabricated but structurally consistent identifiers verify on an
    # empty database and after local state exists, exactly alike.
    assert client.post(URL, json=_request()).json() == {"valid": True}

    # Additional local resources cannot influence a body-only verdict.
    from tests.test_csp_packages import (
        _create_claim,
        _create_content,
        _create_supersession,
    )

    more_content = _create_content(
        client, digest=hashlib.sha256(b"more").hexdigest()
    )
    m1 = _create_claim(client, more_content["id"], "more-1")
    m2 = _create_claim(client, more_content["id"], "more-2")
    _create_supersession(client, m1["id"], m2["id"], "later correction")
    assert client.post(URL, json=package).json() == {"valid": True}
    assert client.post(URL, json=_request()).json() == {"valid": True}


def test_verification_writes_no_resources_or_audit_events(client, db_session):
    _world(client)
    valid_request = _served_request(client)
    tampered = copy.deepcopy(valid_request)
    tampered["corrections"][0]["reason"] = "forged"
    offline = _request()

    models = (
        Actor,
        Content,
        Claim,
        ClaimSupersession,
        AuditEvent,
    )

    def _counts():
        return {
            model: db_session.scalar(select(func.count()).select_from(model))
            for model in models
        }

    before = _counts()
    assert client.post(URL, json=valid_request).status_code == 200
    assert client.post(URL, json=tampered).status_code == 200
    assert client.post(URL, json=offline).status_code == 200
    _assert_validation_error(client.post(URL, json={}))
    _assert_validation_error(
        client.post(
            URL,
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
    )
    db_session.expire_all()
    assert _counts() == before


# --- Top-level structure -------------------------------------------------------------


def test_requires_exactly_checkpoint_and_corrections(client):
    request = _request()
    for field in ("checkpoint", "corrections"):
        incomplete = {k: v for k, v in request.items() if k != field}
        _assert_validation_error(client.post(URL, json=incomplete))

    augmented = dict(request)
    augmented["extra"] = "nope"
    _assert_validation_error(client.post(URL, json=augmented))
    for bad in ([], "checkpoint", None, 42, {}):
        _assert_validation_error(client.post(URL, json=bad))


def test_members_have_correct_types(client):
    request = _request()
    for member, wrong in (
        ("checkpoint", []),
        ("checkpoint", None),
        ("checkpoint", "x"),
        ("corrections", {}),
        ("corrections", None),
        ("corrections", "corrections"),
    ):
        bad = copy.deepcopy(request)
        bad[member] = wrong
        _assert_validation_error(client.post(URL, json=bad))


def test_malformed_json_is_a_validation_error(client):
    for raw in (b"", b" ", b"{", b"{not json", b"[", b"null", b"42"):
        resp = client.post(
            URL, content=raw, headers={"Content-Type": "application/json"}
        )
        _assert_validation_error(resp)


# --- Checkpoint structure -------------------------------------------------------------


def test_checkpoint_requires_exactly_its_four_fields(client):
    request = _request()
    for field in CHECKPOINT_FIELDS:
        bad = copy.deepcopy(request)
        del bad["checkpoint"][field]
        _assert_validation_error(client.post(URL, json=bad))
    bad = copy.deepcopy(request)
    bad["checkpoint"]["extra"] = "nope"
    _assert_validation_error(client.post(URL, json=bad))


def test_checkpoint_version_and_algorithm_are_pinned(client):
    request = _request()
    for field, wrong in (
        ("checkpoint_version", "provenance-csp-checkpoint-v2"),
        ("checkpoint_version", "provenance-csp-checkpoint-v0"),
        ("checkpoint_version", "provenance-audit-checkpoint-v1"),
        ("checkpoint_version", ""),
        ("checkpoint_version", None),
        ("digest_algorithm", "sha512"),
        ("digest_algorithm", "SHA256"),
        ("digest_algorithm", ""),
        ("digest_algorithm", None),
    ):
        bad = copy.deepcopy(request)
        bad["checkpoint"][field] = wrong
        _assert_validation_error(client.post(URL, json=bad))


def test_correction_count_validation(client):
    request = _request()
    for wrong in (-1, "3", 3.0, True, None, {}, []):
        bad = copy.deepcopy(request)
        bad["checkpoint"]["correction_count"] = wrong
        _assert_validation_error(client.post(URL, json=bad))

    # A count that disagrees with the array length is structural: never a
    # digest verdict.
    for wrong_count in (0, 1, 2, 4, 99):
        bad = copy.deepcopy(request)
        bad["checkpoint"]["correction_count"] = wrong_count
        _assert_validation_error(client.post(URL, json=bad))


def test_digest_hex_validation(client):
    request = _request()
    valid = request["checkpoint"]["corrections_digest_hex"]
    for wrong in (
        valid.upper(),
        "g" * 64,
        valid[:-1],
        valid + "0",
        " " + valid,
        valid + " ",
        "",
        None,
        123,
    ):
        bad = copy.deepcopy(request)
        bad["checkpoint"]["corrections_digest_hex"] = wrong
        _assert_validation_error(client.post(URL, json=bad))


# --- Correction structure -------------------------------------------------------------


def test_correction_requires_exactly_its_five_fields(client):
    request = _request()
    for field in CORRECTION_FIELDS:
        bad = copy.deepcopy(request)
        del bad["corrections"][0][field]
        _assert_validation_error(client.post(URL, json=bad))
    bad = copy.deepcopy(request)
    bad["corrections"][0]["extra"] = "nope"
    _assert_validation_error(client.post(URL, json=bad))
    # Raw-material fields can never enter the request.
    for field in ("payload", "signature", "public_key", "content"):
        bad = copy.deepcopy(request)
        bad["corrections"][0][field] = "x"
        _assert_validation_error(client.post(URL, json=bad))


def test_correction_field_types(client):
    request = _request()
    for field in ("id", "superseded_claim_id", "replacement_claim_id", "reason"):
        for wrong in (None, 42, [], {}, True):
            bad = copy.deepcopy(request)
            bad["corrections"][0][field] = wrong
            _assert_validation_error(client.post(URL, json=bad))
    for wrong in (None, 42, [], {}, True, 1234567890):
        bad = copy.deepcopy(request)
        bad["corrections"][0]["created_at"] = wrong
        _assert_validation_error(client.post(URL, json=bad))


def test_blank_identifiers_and_reason_are_validation_errors(client):
    request = _request()
    for field in ("id", "superseded_claim_id", "replacement_claim_id", "reason"):
        for blank in ("", "   ", "\t"):
            bad = copy.deepcopy(request)
            bad["corrections"][0][field] = blank
            _assert_validation_error(client.post(URL, json=bad))


def test_created_at_must_be_strict_rfc3339_utc(client):
    request = _request()
    for wrong in (
        "",
        "   ",
        "2026-01-02",
        "2026-01-02T03:04:05",
        "2026-01-02 03:04:05Z",
        "2026-01-02T03:04Z",
        "2026-01-02T03:04:05z",
        "2026-01-02T03:04:05+01:00",
        "2026-01-02T03:04:05-00:00",
        "2026-13-02T03:04:05Z",
        "not-a-time",
    ):
        bad = copy.deepcopy(request)
        bad["corrections"][0]["created_at"] = wrong
        _assert_validation_error(client.post(URL, json=bad))


# --- Query parameters and methods ------------------------------------------------------


def test_query_parameters_are_validation_errors(client):
    request = _request()
    for suffix in ("?x=1", "?limit=1", "?cursor=x", "?checkpoint=1"):
        resp = client.post(f"{URL}{suffix}", json=request)
        _assert_validation_error(resp)


def test_non_post_methods_are_405(client):
    for method in ("get", "put", "patch", "delete"):
        resp = getattr(client, method)(URL)
        assert resp.status_code == 405, (method, resp.text)
        assert resp.json()["error"]["code"] == "method_not_allowed"
