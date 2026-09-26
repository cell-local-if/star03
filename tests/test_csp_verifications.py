"""Tests for the stateless correction checkpoint verification route.

Covers POST /v1/csp/verify: the request body is exactly
{"checkpoint", "corrections"}; the checkpoint is exactly
{"checkpoint_version", "digest_algorithm", "correction_count",
"corrections_digest_hex"} with the version pinned to
"provenance-csp-checkpoint-v1", the algorithm to "sha256", the count a
non-negative integer, and the digest 64 lowercase hex characters. The
corrections array must contain exactly ``correction_count`` elements,
each exactly the exported 5-field claim-supersession public view (strict
RFC 3339 UTC ``created_at``; non-empty identifiers and reason; no
undeclared member). The digest is the SHA-256 of the canonical
corrections array (array order preserved, nested object keys sorted by
Unicode code point, compact separators, non-ASCII unescaped, UTF-8)
computed over the corrections exactly as received. Any structural,
field, type, count, timestamp, digest-input, or malformed-JSON
violation is a 422 validation_error; a structurally valid request whose
digest differs is 200 {"valid": false, "computed_digest_hex": ...}; a
match is exactly {"valid": true}. The route accepts no query parameters
and is fully stateless: it needs no persisted supersession and creates,
modifies, logs, and queries nothing, so unknown resources, repeated
requests, and differing local state verify identically. All fixtures
are deterministic and offline.
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


def _digest_of(corrections: list) -> str:
    """Independently canonicalize the correction array and digest it."""
    canonical = json.dumps(
        corrections,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _correction(
    *,
    id="csp_" + "0" * 64,
    superseded_claim_id="clm_" + "1" * 64,
    replacement_claim_id="clm_" + "2" * 64,
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
    """A self-contained correction sequence fabricated without service state."""
    return [
        _correction(
            id="csp_" + "0" * 64,
            superseded_claim_id="clm_" + "1" * 64,
            replacement_claim_id="clm_" + "2" * 64,
            created_at="2026-01-02T03:04:05Z",
        ),
        _correction(
            id="csp_" + "a" * 64,
            superseded_claim_id="clm_" + "2" * 64,
            replacement_claim_id="clm_" + "3" * 64,
            reason="Képek ⛄",
            created_at="2026-01-02T03:04:06.500+00:00",
        ),
        _correction(
            id="csp_" + "f" * 64,
            superseded_claim_id="clm_" + "4" * 64,
            replacement_claim_id="clm_" + "5" * 64,
            reason="third correction",
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


# --- Verdicts -----------------------------------------------------------------


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
        _served_request(client, reason="no.such.reason"),
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


# --- Statelessness --------------------------------------------------------------


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
    _world(client)
    package = _served_request(client)
    assert client.post(URL, json=package).json() == {"valid": True}

    # Creating new local resources changes the current correction listing
    # but cannot influence a verdict computed from the request body.
    import hashlib as _hashlib

    from tests.helpers import content_payload
    from tests.test_csp_packages import _make_claim, _supersede

    content = client.post(
        "/v1/contents",
        json=content_payload(
            digest=_hashlib.sha256(b"state-change").hexdigest(),
            actor_id="org-1",
        ),
    ).json()
    k1 = _make_claim(client, content["id"], "later-1")
    k2 = _make_claim(client, content["id"], "later-2")
    _supersede(client, k1["id"], k2["id"], "later change")
    assert client.post(URL, json=package).json() == {"valid": True}
    # And a package carrying unknown/fabricated ids verifies regardless.
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


# --- Top-level structure ---------------------------------------------------------


def test_requires_exactly_checkpoint_and_corrections(client):
    request = _request()
    for field in ("checkpoint", "corrections"):
        incomplete = {k: v for k, v in request.items() if k != field}
        _assert_validation_error(client.post(URL, json=incomplete))

    augmented = dict(request)
    augmented["extra"] = "nope"
    _assert_validation_error(client.post(URL, json=augmented))
    _assert_validation_error(client.post(URL, json=[]))
    _assert_validation_error(client.post(URL, json="checkpoint"))
    _assert_validation_error(client.post(URL, json=None))
    _assert_validation_error(client.post(URL, json=42))
    _assert_validation_error(client.post(URL, json={}))


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


def test_malformed_and_empty_bodies_are_validation_errors(client):
    for body_bytes in (
        b"",
        b"   ",
        b"\n\t",
        b"{not json",
        b'{"checkpoint":',
        b"[]",
        b"null",
        b"\xff\xfe",
    ):
        resp = client.post(
            URL,
            content=body_bytes,
            headers={"Content-Type": "application/json"},
        )
        _assert_validation_error(resp)


def test_query_parameters_are_rejected(client):
    request = _request()
    for suffix in ("?x=1", "?limit=1", "?cursor=x", "?valid=true"):
        resp = client.post(URL + suffix, json=request)
        _assert_validation_error(resp)


def test_non_post_methods_are_405(client):
    for method in ("get", "put", "patch", "delete"):
        resp = getattr(client, method)(URL)
        assert resp.status_code == 405, (method, resp.text)
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Checkpoint fields ------------------------------------------------------------


def test_checkpoint_requires_exactly_its_four_fields(client):
    request = _request()
    for field in CHECKPOINT_FIELDS:
        bad = copy.deepcopy(request)
        del bad["checkpoint"][field]
        _assert_validation_error(client.post(URL, json=bad))
    bad = copy.deepcopy(request)
    bad["checkpoint"]["extra"] = 1
    _assert_validation_error(client.post(URL, json=bad))


def test_checkpoint_version_and_algorithm_are_pinned(client):
    request = _request()
    for field, wrong in (
        ("checkpoint_version", "provenance-csp-checkpoint-v2"),
        ("checkpoint_version", ""),
        ("checkpoint_version", None),
        ("checkpoint_version", 1),
        ("digest_algorithm", "sha512"),
        ("digest_algorithm", "SHA256"),
        ("digest_algorithm", None),
    ):
        bad = copy.deepcopy(request)
        bad["checkpoint"][field] = wrong
        _assert_validation_error(client.post(URL, json=bad))


def test_correction_count_must_be_a_non_negative_integer(client):
    request = _request()
    for wrong in (-1, 1.0, "3", None, True, False):
        bad = copy.deepcopy(request)
        bad["checkpoint"]["correction_count"] = wrong
        _assert_validation_error(client.post(URL, json=bad))


def test_correction_count_must_match_the_array_length(client):
    request = _request()
    for claimed in (0, 1, 2, 4, 99):
        bad = copy.deepcopy(request)
        bad["checkpoint"]["correction_count"] = claimed
        _assert_validation_error(client.post(URL, json=bad))
    # The exact length is accepted (regardless of the digest, which is
    # checked independently of the count).
    okay = copy.deepcopy(request)
    okay["checkpoint"]["corrections_digest_hex"] = "0" * 64
    resp = client.post(URL, json=okay)
    assert resp.status_code == 200
    assert resp.json()["valid"] is False


def test_digest_must_be_64_lowercase_hex(client):
    request = _request()
    good = request["checkpoint"]["corrections_digest_hex"]
    for wrong in (
        good.upper(),
        good[:-1],
        good + "0",
        "g" * 64,
        " " + good,
        good + " ",
        "",
        None,
        64,
        True,
    ):
        bad = copy.deepcopy(request)
        bad["checkpoint"]["corrections_digest_hex"] = wrong
        _assert_validation_error(client.post(URL, json=bad))


# --- Correction item structure ----------------------------------------------------


def test_correction_requires_exactly_its_five_fields(client):
    corrections = _offline_corrections()
    required = set(corrections[0])
    assert len(required) == 5
    for field in required:
        bad_corrections = copy.deepcopy(corrections)
        del bad_corrections[0][field]
        _assert_validation_error(
            client.post(URL, json=_request(bad_corrections))
        )
    for extra in ("signature", "public_key", "payload", "seq", "content_id"):
        bad_corrections = copy.deepcopy(corrections)
        bad_corrections[0][extra] = "abcd"
        _assert_validation_error(
            client.post(URL, json=_request(bad_corrections))
        )


def test_correction_string_fields_must_be_nonempty_strings(client):
    corrections = _offline_corrections()
    for field in (
        "id",
        "superseded_claim_id",
        "replacement_claim_id",
        "reason",
    ):
        for wrong in ("", "   ", "\t", None, 7, [], {}):
            bad_corrections = copy.deepcopy(corrections)
            bad_corrections[0][field] = wrong
            _assert_validation_error(
                client.post(URL, json=_request(bad_corrections))
            )


def test_created_at_must_be_strict_rfc3339_utc(client):
    corrections = _offline_corrections()
    for wrong in (
        "",
        "   ",
        "2026-01-02",
        "2026-01-02T03:04:05",
        "2026-01-02 03:04:05Z",
        "2026-01-02T03:04Z",
        "2026-01-02T03:04:05+01:00",
        "2026-13-02T03:04:05Z",
        "2026-01-02t03:04:05z",
        None,
        1735786000,
    ):
        bad_corrections = copy.deepcopy(corrections)
        bad_corrections[0]["created_at"] = wrong
        _assert_validation_error(
            client.post(URL, json=_request(bad_corrections))
        )


def test_structurally_valid_but_fabricated_identifiers_verify(client):
    # The route never resolves ids: endpoints that do not exist locally
    # are not 422s -- they verify as long as the structure and digest
    # hold.
    corrections = [
        _correction(
            superseded_claim_id="clm_fabricated_old",
            replacement_claim_id="clm_fabricated_new",
        )
    ]
    resp = client.post(URL, json=_request(corrections))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


# --- End-to-end with the export ----------------------------------------------------


def test_every_served_package_round_trips_through_verification(client):
    world = _world(client)
    s1 = world["supersessions"][0]
    for params in (
        {},
        {"reason": "third"},
        {"reason": "Képek ⛄"},
        {"id": s1["id"]},
        {"superseded_claim_id": "clm_nonexistent"},
        {"from": "2026-01-01T00:00:00Z"},
        {
            "from": "2026-01-01T00:00:00Z",
            "to": "2027-01-01T00:00:00Z",
        },
    ):
        package = client.get(PACKAGE_PATH, params=params).json()
        assert package["checkpoint"]["correction_count"] == len(
            package["corrections"]
        )
        resp = client.post(
            URL,
            json={
                "checkpoint": package["checkpoint"],
                "corrections": package["corrections"],
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"valid": True}, params
