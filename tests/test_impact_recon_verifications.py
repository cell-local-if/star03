"""Tests for the stateless impact-recon package verification route.

Covers POST /v1/impact-recon-verifications: the request body is exactly
{"checkpoint", "entries"}; the checkpoint is exactly
{"checkpoint_version", "digest_algorithm", "entry_count",
"entries_digest_hex"} with the version pinned to "pir-checkpoint-v1",
the algorithm to "sha256", the count a non-negative integer, and the
digest 64 lowercase hex characters. The entries array must contain
exactly ``entry_count`` elements, each exactly the exported 8-field
reconciliation public view (strict RFC 3339 UTC ``received_at``;
non-empty ``id``/``checkpoint_version``; non-negative integral
``impact_count``; 64 lowercase hex ``impacts_digest_hex``; strict
boolean ``local_available``/``matches``; no undeclared member) with a
``local_checkpoint`` of exactly its four fields, pinned to
"provenance-revocation-impact-checkpoint-v1"/"sha256" with a
non-negative impact count and 64 lowercase hex impacts digest. No
cross-field association between the flags, receipt identity, and the
embedded local checkpoint is enforced: verification is decided by
structure and digest alone and never resolves any id. The digest is
the SHA-256 of the canonical entries array (array order preserved,
nested object keys sorted recursively by Unicode code point, compact
separators, non-ASCII unescaped, UTF-8) computed over the entries
exactly as received. Any structural, field, type, count, timestamp,
digest-format, or malformed-JSON violation is a 422 validation_error; a
structurally valid request whose digest differs is 200 {"valid":
false, "computed_digest_hex": ...}; a match is exactly {"valid": true}.
The route accepts no query parameters and is fully stateless: it needs
no persisted receipt and creates, modifies, logs, and queries nothing,
so unknown resources, repeated requests, and differing local state
verify identically. All fixtures are deterministic and offline.
"""

from __future__ import annotations

import copy
import hashlib
import json

from sqlalchemy import func, select

from provenance.models import (
    Actor,
    AttestationRevocation,
    AuditEvent,
    Claim,
    Content,
    EvidenceBundle,
    ImpactImportRecord,
)
from tests.test_impact_imports import _offline_request
from tests.test_revocation_impacts import _world

URL = "/v1/impact-recon-verifications"
PACKAGE_PATH = "/v1/impact-recon-package"
IMPORTS_URL = "/v1/impact-imports"
PACKAGE_VERSION = "pir-checkpoint-v1"
LOCAL_CHECKPOINT_VERSION = "provenance-revocation-impact-checkpoint-v1"

CHECKPOINT_FIELDS = {
    "checkpoint_version",
    "digest_algorithm",
    "entry_count",
    "entries_digest_hex",
}
LOCAL_CHECKPOINT_FIELDS = {
    "checkpoint_version",
    "digest_algorithm",
    "impact_count",
    "impacts_digest_hex",
}
ENTRY_FIELDS = {
    "id",
    "checkpoint_version",
    "impact_count",
    "impacts_digest_hex",
    "received_at",
    "local_available",
    "local_checkpoint",
    "matches",
}


def _digest_of(entries: list) -> str:
    """Independently canonicalize the entries array and digest it."""
    canonical = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _local_checkpoint(
    *,
    version=LOCAL_CHECKPOINT_VERSION,
    algorithm="sha256",
    impact_count=0,
    impacts_digest_hex=None,
):
    return {
        "checkpoint_version": version,
        "digest_algorithm": algorithm,
        "impact_count": impact_count,
        "impacts_digest_hex": hashlib.sha256(b"local").hexdigest()
        if impacts_digest_hex is None
        else impacts_digest_hex,
    }


def _entry(
    *,
    id="rii_01hztest000000000000000001",
    checkpoint_version=LOCAL_CHECKPOINT_VERSION,
    impact_count=2,
    impacts_digest_hex=None,
    received_at="2026-01-02T03:04:05Z",
    local_available=True,
    local_checkpoint=None,
    matches=False,
):
    return {
        "id": id,
        "checkpoint_version": checkpoint_version,
        "impact_count": impact_count,
        "impacts_digest_hex": hashlib.sha256(b"receipt").hexdigest()
        if impacts_digest_hex is None
        else impacts_digest_hex,
        "received_at": received_at,
        "local_available": local_available,
        "local_checkpoint": (
            _local_checkpoint() if local_checkpoint is None else local_checkpoint
        ),
        "matches": matches,
    }


def _offline_entries() -> list:
    """A self-contained reconciliation sequence with no service state."""
    return [
        _entry(
            id="rii_01hztest000000000000000001",
            checkpoint_version="v-快照-é",
            impact_count=2,
            impacts_digest_hex=hashlib.sha256(b"receipt-1").hexdigest(),
            received_at="2026-01-02T03:04:05Z",
            local_available=False,
            local_checkpoint=_local_checkpoint(impact_count=0),
            matches=True,
        ),
        _entry(
            id="rii_01hztest000000000000000002",
            checkpoint_version=LOCAL_CHECKPOINT_VERSION,
            impact_count=3,
            impacts_digest_hex=hashlib.sha256(b"receipt-2").hexdigest(),
            received_at="2026-01-02T03:04:06.500+00:00",
            local_available=True,
            local_checkpoint=_local_checkpoint(impact_count=4),
            matches=False,
        ),
        _entry(
            id="rii_01hztest000000000000000003",
            checkpoint_version="other-version",
            impact_count=9,
            impacts_digest_hex=hashlib.sha256(b"receipt-3").hexdigest(),
            received_at="2026-01-03T00:00:00Z",
            local_available=True,
            local_checkpoint=_local_checkpoint(impact_count=4),
            matches=False,
        ),
    ]


def _request(entries: list | None = None, *, digest=None) -> dict:
    entries = _offline_entries() if entries is None else entries
    return {
        "checkpoint": {
            "checkpoint_version": PACKAGE_VERSION,
            "digest_algorithm": "sha256",
            "entry_count": len(entries),
            "entries_digest_hex": _digest_of(entries)
            if digest is None
            else digest,
        },
        "entries": entries,
    }


def _served_request(client, **params) -> dict:
    package = client.get(PACKAGE_PATH, params=params).json()
    return {
        "checkpoint": package["checkpoint"],
        "entries": package["entries"],
    }


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- Verdicts ---------------------------------------------------------------------


def test_verification_of_offline_package_is_valid_on_empty_database(client):
    # No resource exists locally and every identifier is unknown: the
    # verdict is decided by the request body alone.
    resp = client.post(URL, json=_request())
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_of_served_package_is_valid(client):
    _world(client)
    # Register one receipt so the served package has an entry.
    client.post(IMPORTS_URL, json=_offline_request())
    resp = client.post(URL, json=_served_request(client))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_of_empty_snapshot_is_valid(client):
    for payload in (
        _request(entries=[]),
        _served_request(client),
    ):
        assert payload["checkpoint"]["entry_count"] == 0
        assert payload["checkpoint"]["entries_digest_hex"] == hashlib.sha256(
            b"[]"
        ).hexdigest()
        resp = client.post(URL, json=payload)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"valid": True}


def test_digest_mismatch_returns_exactly_valid_false_and_computed_digest(client):
    request = _request()
    claimed = request["checkpoint"]["entries_digest_hex"]
    request["checkpoint"]["entries_digest_hex"] = claimed[:-1] + (
        "0" if claimed[-1] != "0" else "1"
    )
    resp = client.post(URL, json=request)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"valid", "computed_digest_hex"}
    assert body["valid"] is False
    assert body["computed_digest_hex"] == _digest_of(_offline_entries())
    assert body["computed_digest_hex"] == claimed
    assert len(body["computed_digest_hex"]) == 64


def test_mismatch_response_is_compact_json_with_one_newline(client):
    request = _request()
    request["checkpoint"]["entries_digest_hex"] = "0" * 64
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


def test_tampered_entry_is_invalid(client):
    request = _request()
    request["entries"][0]["matches"] = not request["entries"][0]["matches"]
    resp = client.post(URL, json=request)
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert body["computed_digest_hex"] == _digest_of(request["entries"])


def test_array_order_participates_in_the_digest(client):
    request = _request()
    reordered = copy.deepcopy(request)
    reordered["entries"] = list(reversed(reordered["entries"]))
    resp = client.post(URL, json=reordered)
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert body["computed_digest_hex"] == _digest_of(reordered["entries"])


def test_nested_object_key_spelling_is_irrelevant_to_the_digest(client):
    # The canonical form sorts nested object keys recursively (the
    # embedded local_checkpoint included): reserializing the entry with
    # keys in another order must still match, while the array order
    # itself stays significant.
    request = _request()
    reshuffled = copy.deepcopy(request)
    reshuffled["entries"] = [
        {k: entry[k] for k in sorted(entry, reverse=True)}
        for entry in reshuffled["entries"]
    ]
    resp = client.post(URL, json=reshuffled)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_non_ascii_is_canonicalized_unescaped(client):
    request = _request()
    assert any(
        ord(ch) > 127
        for ch in json.dumps(request["entries"], ensure_ascii=False)
    )
    assert client.post(URL, json=request).json() == {"valid": True}
    escaped = json.dumps(
        request["entries"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    assert hashlib.sha256(escaped).hexdigest() != _digest_of(request["entries"])


def test_equivalent_utc_spellings_are_accepted(client):
    # Both Z and +00:00 are strict RFC 3339 UTC spellings; the digest
    # commits to the exact received spelling (the route normalizes
    # nothing), but each is a legal digest input and verifies.
    for spelling in (
        "2026-01-02T03:04:05Z",
        "2026-01-02T03:04:05+00:00",
        "2026-01-02T03:04:05.000Z",
    ):
        entries = [_entry(received_at=spelling)]
        assert client.post(URL, json=_request(entries)).json() == {"valid": True}


def test_structurally_valid_but_self_inconsistent_associations_verify(client):
    # The route never reconciles the entry against itself or local
    # state: an unavailable flag paired with a positive local impact
    # count, or a matches flag contradicting the receipt/local identity,
    # is not a 422 -- it verifies as long as structure and digest hold.
    entries = [
        _entry(
            local_available=False,
            local_checkpoint=_local_checkpoint(impact_count=5),
            matches=True,
        )
    ]
    resp = client.post(URL, json=_request(entries))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


# --- Statelessness ----------------------------------------------------------------


def test_verification_is_deterministic_across_repeated_requests(client):
    _world(client)
    client.post(IMPORTS_URL, json=_offline_request())
    request = _served_request(client)
    first = client.post(URL, json=request)
    second = client.post(URL, json=request)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content == b'{"valid":true}\n'


def test_local_state_changes_do_not_change_the_verdict(client):
    # First verify an offline package, then create local resources and
    # receipts: the earlier package must still verify byte-for-byte
    # because the route never queries local state.
    request = _request()
    assert client.post(URL, json=request).json() == {"valid": True}
    _world(client)
    client.post(IMPORTS_URL, json=_offline_request())
    assert client.post(URL, json=request).json() == {"valid": True}


def test_verification_writes_no_resources_or_audit_events(client, db_session):
    _world(client)
    client.post(IMPORTS_URL, json=_offline_request())
    valid_request = _served_request(client)
    tampered = copy.deepcopy(valid_request)
    if tampered["entries"]:
        tampered["entries"][0]["matches"] = not tampered["entries"][0]["matches"]
    offline = _request()

    models = (
        Actor,
        Content,
        Claim,
        EvidenceBundle,
        AttestationRevocation,
        ImpactImportRecord,
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


# --- Top-level structure -----------------------------------------------------------


def test_requires_exactly_checkpoint_and_entries(client):
    request = _request()
    for field in ("checkpoint", "entries"):
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
        ("entries", {}),
        ("entries", None),
        ("entries", "entries"),
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


# --- Checkpoint fields -------------------------------------------------------------


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
        ("checkpoint_version", "provenance-revocation-impact-checkpoint-v1"),
        ("checkpoint_version", "pir-checkpoint-v2"),
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


def test_entry_count_must_be_a_non_negative_integer(client):
    request = _request()
    for wrong in (-1, 1.0, "3", None, True, False):
        bad = copy.deepcopy(request)
        bad["checkpoint"]["entry_count"] = wrong
        _assert_validation_error(client.post(URL, json=bad))


def test_entry_count_must_match_the_array_length(client):
    request = _request()
    for claimed in (0, 1, 2, 4, 99):
        bad = copy.deepcopy(request)
        bad["checkpoint"]["entry_count"] = claimed
        _assert_validation_error(client.post(URL, json=bad))
    # The exact length is accepted (regardless of the digest, which is
    # checked independently of the count).
    okay = copy.deepcopy(request)
    okay["checkpoint"]["entries_digest_hex"] = "0" * 64
    resp = client.post(URL, json=okay)
    assert resp.status_code == 200
    assert resp.json()["valid"] is False


def test_digest_must_be_64_lowercase_hex(client):
    request = _request()
    good = request["checkpoint"]["entries_digest_hex"]
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
        bad["checkpoint"]["entries_digest_hex"] = wrong
        _assert_validation_error(client.post(URL, json=bad))


# --- Entry structure ---------------------------------------------------------------


def test_entry_requires_exactly_its_eight_fields(client):
    entries = _offline_entries()
    assert set(entries[0]) == ENTRY_FIELDS
    for field in ENTRY_FIELDS:
        bad_entries = copy.deepcopy(entries)
        del bad_entries[0][field]
        _assert_validation_error(client.post(URL, json=_request(bad_entries)))
    # Undeclared raw-material fields are refused.
    for extra in (
        "signature",
        "public_key",
        "payload",
        "seq",
        "evidence",
        "impacts",
        "content_bytes",
    ):
        bad_entries = copy.deepcopy(entries)
        bad_entries[0][extra] = "abcd"
        _assert_validation_error(client.post(URL, json=_request(bad_entries)))


def test_entry_string_fields_must_be_nonempty_strings(client):
    entries = _offline_entries()
    for field in ("id", "checkpoint_version"):
        for wrong in ("", "   ", "\t", None, 7, [], {}):
            bad_entries = copy.deepcopy(entries)
            bad_entries[0][field] = wrong
            _assert_validation_error(client.post(URL, json=_request(bad_entries)))


def test_entry_count_fields_must_be_non_negative_integers(client):
    entries = _offline_entries()
    for wrong in (-1, 1.5, "1", None, True, False):
        bad_entries = copy.deepcopy(entries)
        bad_entries[0]["impact_count"] = wrong
        _assert_validation_error(client.post(URL, json=_request(bad_entries)))


def test_receipt_digest_must_be_64_lowercase_hex(client):
    entries = _offline_entries()
    good = entries[0]["impacts_digest_hex"]
    for wrong in (
        good.upper(),
        good[:-1],
        "g" * 64,
        " " + good,
        "",
        None,
        64,
    ):
        bad_entries = copy.deepcopy(entries)
        bad_entries[0]["impacts_digest_hex"] = wrong
        _assert_validation_error(client.post(URL, json=_request(bad_entries)))


def test_flags_must_be_strict_booleans(client):
    entries = _offline_entries()
    for field in ("local_available", "matches"):
        for wrong in ("true", "false", 1, 0, None, "yes", []):
            bad_entries = copy.deepcopy(entries)
            bad_entries[0][field] = wrong
            _assert_validation_error(client.post(URL, json=_request(bad_entries)))
    # Genuine JSON booleans (both values) are accepted.
    for value in (True, False):
        good_entries = copy.deepcopy(entries)
        good_entries[0]["local_available"] = value
        good_entries[0]["matches"] = value
        resp = client.post(URL, json=_request(good_entries))
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"valid": True}


def test_received_at_must_be_strict_rfc3339_utc(client):
    entries = _offline_entries()
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
        bad_entries = copy.deepcopy(entries)
        bad_entries[0]["received_at"] = wrong
        _assert_validation_error(client.post(URL, json=_request(bad_entries)))


def test_local_checkpoint_must_be_an_object(client):
    entries = _offline_entries()
    for wrong in (None, [], "x", 1):
        bad_entries = copy.deepcopy(entries)
        bad_entries[0]["local_checkpoint"] = wrong
        _assert_validation_error(client.post(URL, json=_request(bad_entries)))


# --- Local checkpoint structure ----------------------------------------------------


def test_local_checkpoint_requires_exactly_its_four_fields(client):
    entries = _offline_entries()
    for field in LOCAL_CHECKPOINT_FIELDS:
        bad_entries = copy.deepcopy(entries)
        del bad_entries[0]["local_checkpoint"][field]
        _assert_validation_error(client.post(URL, json=_request(bad_entries)))
    bad_entries = copy.deepcopy(entries)
    bad_entries[0]["local_checkpoint"]["extra"] = 1
    _assert_validation_error(client.post(URL, json=_request(bad_entries)))


def test_local_checkpoint_version_and_algorithm_are_pinned(client):
    entries = _offline_entries()
    for field, wrong in (
        ("checkpoint_version", "pir-checkpoint-v1"),
        ("checkpoint_version", "provenance-revocation-impact-checkpoint-v2"),
        ("checkpoint_version", ""),
        ("checkpoint_version", None),
        ("digest_algorithm", "sha512"),
        ("digest_algorithm", "SHA256"),
        ("digest_algorithm", None),
    ):
        bad_entries = copy.deepcopy(entries)
        bad_entries[0]["local_checkpoint"][field] = wrong
        _assert_validation_error(client.post(URL, json=_request(bad_entries)))


def test_local_checkpoint_count_must_be_non_negative_integer(client):
    entries = _offline_entries()
    for wrong in (-1, 1.0, "0", None, True, False):
        bad_entries = copy.deepcopy(entries)
        bad_entries[0]["local_checkpoint"]["impact_count"] = wrong
        _assert_validation_error(client.post(URL, json=_request(bad_entries)))


def test_local_checkpoint_digest_must_be_64_lowercase_hex(client):
    entries = _offline_entries()
    good = entries[0]["local_checkpoint"]["impacts_digest_hex"]
    for wrong in (
        good.upper(),
        good[:-1],
        "g" * 64,
        good + " ",
        "",
        None,
        64,
    ):
        bad_entries = copy.deepcopy(entries)
        bad_entries[0]["local_checkpoint"]["impacts_digest_hex"] = wrong
        _assert_validation_error(client.post(URL, json=_request(bad_entries)))


# --- End-to-end with the export ----------------------------------------------------


def test_every_served_package_round_trips_through_verification(client):
    _world(client)
    client.post(IMPORTS_URL, json=_offline_request())
    for params in (
        {},
        {"local_available": "true"},
        {"matches": "true"},
        {"matches": "false"},
        {"local_available": "false", "matches": "true"},
        {"local_available": "true", "matches": "false"},
        # Empty match sets still carry the deterministic empty digest.
        {"local_available": "false"},
    ):
        package = client.get(PACKAGE_PATH, params=params).json()
        assert package["checkpoint"]["entry_count"] == len(package["entries"])
        resp = client.post(
            URL,
            json={
                "checkpoint": package["checkpoint"],
                "entries": package["entries"],
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"valid": True}, params
