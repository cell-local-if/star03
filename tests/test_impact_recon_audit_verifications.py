"""Tests for the stateless impact-recon audit package verification route.

Covers POST /v1/impact-recon-audit-verifications: the request body is
exactly {"checkpoint", "entries"}; the checkpoint is exactly
{"checkpoint_version", "digest_algorithm", "entry_count",
"entries_digest_hex"} with the version pinned to "pir-audit-checkpoint-v1",
the algorithm to "sha256", the count a non-negative integer, and the
digest 64 lowercase hex characters. The entries array must contain
exactly ``entry_count`` elements, each exactly the exported nine-field
audit entry (non-empty ``id``/``signer_subject``/``public_key``; 64
lowercase hex ``package_digest_hex``/``signature_digest_hex``/
``local_package_digest_hex``; strict RFC 3339 UTC ``received_at``;
strict booleans ``local_available``/``matches``; no undeclared member).
Verification is decided by structure and digest alone and never resolves
any id against local state. The digest is the SHA-256 of the canonical
entries array (array order preserved, nested object keys sorted by
Unicode code point, compact separators, non-ASCII unescaped, UTF-8)
computed over the entries exactly as received. Any structural, field,
type, count, timestamp, digest-format, or malformed-JSON violation is a
422 validation_error; a structurally valid request whose digest differs
is 200 {"valid": false, "computed_digest_hex": ...}; a match is exactly
{"valid": true}. The route accepts no query parameters and is fully
stateless: it needs no persisted receipt and creates, modifies, logs,
and queries nothing, so unknown resources, repeated requests, and
differing local state verify identically. Non-POST methods are 405
method_not_allowed. All fixtures are deterministic and offline.
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
from sqlalchemy import func, select

from provenance import canonical
from provenance.models import AuditEvent, ImpactReconExchangeImportRecord
from tests.test_impact_recon_exchange_imports_list import _three_receipts

URL = "/v1/impact-recon-audit-verifications"
PACKAGE_PATH = "/v1/impact-recon-audit-packages"
PACKAGE_VERSION = "pir-audit-checkpoint-v1"
DIGEST_ALGORITHM = "sha256"

CHECKPOINT_FIELDS = {
    "checkpoint_version",
    "digest_algorithm",
    "entry_count",
    "entries_digest_hex",
}
ENTRY_FIELDS = {
    "id",
    "signer_subject",
    "public_key",
    "package_digest_hex",
    "signature_digest_hex",
    "received_at",
    "local_available",
    "local_package_digest_hex",
    "matches",
}
EMPTY_ARRAY_DIGEST = hashlib.sha256(b"[]").hexdigest()


def _digest_of(entries: list) -> str:
    """Independently canonicalize the entries array and digest it."""
    canonical_bytes = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def _entry(
    *,
    id="irx_" + "0" * 64,
    signer_subject="remote-system",
    public_key="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
    package_digest_hex=None,
    signature_digest_hex=None,
    received_at="2026-01-02T03:04:05Z",
    local_available=True,
    local_package_digest_hex=None,
    matches=False,
):
    return {
        "id": id,
        "signer_subject": signer_subject,
        "public_key": public_key,
        "package_digest_hex": hashlib.sha256(b"package").hexdigest()
        if package_digest_hex is None
        else package_digest_hex,
        "signature_digest_hex": hashlib.sha256(b"signature").hexdigest()
        if signature_digest_hex is None
        else signature_digest_hex,
        "received_at": received_at,
        "local_available": local_available,
        "local_package_digest_hex": hashlib.sha256(b"local").hexdigest()
        if local_package_digest_hex is None
        else local_package_digest_hex,
        "matches": matches,
    }


def _offline_entries() -> list:
    """A self-contained audit-entry sequence with no service state."""
    return [
        _entry(),
        _entry(
            id="irx_" + "1" * 64,
            signer_subject="other-subject",
            received_at="2026-01-02T03:04:06+00:00",
            local_available=False,
            matches=True,
        ),
    ]


def _checkpoint(entries: list, **overrides) -> dict:
    checkpoint = {
        "checkpoint_version": PACKAGE_VERSION,
        "digest_algorithm": DIGEST_ALGORITHM,
        "entry_count": len(entries),
        "entries_digest_hex": _digest_of(entries),
    }
    checkpoint.update(overrides)
    return checkpoint


def _request_body(entries=None, **checkpoint_overrides) -> dict:
    if entries is None:
        entries = _offline_entries()
    return {
        "checkpoint": _checkpoint(entries, **checkpoint_overrides),
        "entries": entries,
    }


def _verify(client, body) -> object:
    return client.post(URL, json=body)


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def _audit_count(db_session) -> int:
    return db_session.scalar(select(func.count()).select_from(AuditEvent))


def _receipt_count(db_session) -> int:
    return db_session.scalar(
        select(func.count()).select_from(ImpactReconExchangeImportRecord)
    )


# --- Verdicts ---------------------------------------------------------------------


def test_matching_checkpoint_is_valid(client):
    resp = _verify(client, _request_body())
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}
    # A match carries no field besides valid.
    assert resp.content == b'{"valid":true}\n'


def test_empty_entries_match_the_empty_array_digest(client):
    body = {
        "checkpoint": {
            "checkpoint_version": PACKAGE_VERSION,
            "digest_algorithm": DIGEST_ALGORITHM,
            "entry_count": 0,
            "entries_digest_hex": EMPTY_ARRAY_DIGEST,
        },
        "entries": [],
    }
    resp = _verify(client, body)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_mismatched_checkpoint_reports_the_computed_digest(client):
    body = _request_body()
    body["checkpoint"]["entries_digest_hex"] = "0" * 64
    resp = _verify(client, body)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "valid": False,
        "computed_digest_hex": _digest_of(body["entries"]),
    }
    assert resp.content.endswith(b"\n") and b", " not in resp.content


def test_digest_binds_array_order_and_timestamp_spelling(client):
    entries = _offline_entries()
    # Reordering the entries changes the digest.
    reordered = list(reversed(entries))
    body = _request_body(entries)
    body["entries"] = reordered
    resp = _verify(client, body)
    assert resp.status_code == 200, resp.text
    assert resp.json()["valid"] is False
    assert resp.json()["computed_digest_hex"] == _digest_of(reordered)
    # A different spelling of the same instant is a different commitment.
    respelled = copy.deepcopy(entries)
    respelled[0]["received_at"] = "2026-01-02T03:04:05+00:00"
    body = _request_body(entries)
    body["entries"] = respelled
    resp = _verify(client, body)
    assert resp.status_code == 200, resp.text
    assert resp.json()["valid"] is False
    assert resp.json()["computed_digest_hex"] == _digest_of(respelled)


def test_verification_uses_the_request_body_alone(client):
    # The same offline body verifies identically before and after local
    # receipts exist, and needs no persisted state at all.
    body = _request_body()
    assert _verify(client, body).json() == {"valid": True}
    _three_receipts(client)
    assert _verify(client, body).json() == {"valid": True}
    assert _verify(client, body).json() == {"valid": True}


def test_exported_package_verifies(client):
    _three_receipts(client)
    package = client.get(PACKAGE_PATH).json()
    assert _verify(client, package).json() == {"valid": True}


# --- Structure validation -----------------------------------------------------------


def test_malformed_json_is_422(client):
    resp = client.post(
        URL, content=b"{not json", headers={"Content-Type": "application/json"}
    )
    _assert_validation_error(resp)


@pytest.mark.parametrize("body", [b"", b"[]", b'"x"', b"1", b"null", b"true"])
def test_non_object_body_is_422(client, body):
    resp = client.post(
        URL, content=body, headers={"Content-Type": "application/json"}
    )
    _assert_validation_error(resp)


@pytest.mark.parametrize("missing", ["checkpoint", "entries"])
def test_missing_top_level_member_is_422(client, missing):
    body = _request_body()
    del body[missing]
    _assert_validation_error(_verify(client, body))


def test_extra_top_level_member_is_422(client):
    body = _request_body()
    body["extra"] = True
    _assert_validation_error(_verify(client, body))


@pytest.mark.parametrize(
    "override",
    [
        {"checkpoint_version": "pir-checkpoint-v1"},
        {"checkpoint_version": "provenance-audit-checkpoint-v1"},
        {"digest_algorithm": "sha512"},
        {"entry_count": -1},
        {"entry_count": 1.5},
        {"entry_count": "2"},
        {"entries_digest_hex": "0" * 63},
        {"entries_digest_hex": "0" * 65},
        {"entries_digest_hex": "A" * 64},
        {"entries_digest_hex": "g" * 64},
    ],
)
def test_bad_checkpoint_field_is_422(client, override):
    _assert_validation_error(_verify(client, _request_body(**override)))


def test_checkpoint_missing_and_extra_members_are_422(client):
    body = _request_body()
    del body["checkpoint"]["entry_count"]
    _assert_validation_error(_verify(client, body))
    body = _request_body()
    body["checkpoint"]["extra"] = 1
    _assert_validation_error(_verify(client, body))


def test_entry_count_must_match_entries_length(client):
    entries = _offline_entries()
    body = _request_body(entries, entry_count=len(entries) + 1)
    _assert_validation_error(_verify(client, body))
    body = _request_body(entries, entry_count=len(entries) - 1)
    _assert_validation_error(_verify(client, body))


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", ""),
        ("id", "   "),
        ("id", 12),
        ("signer_subject", ""),
        ("signer_subject", "  "),
        ("signer_subject", None),
        ("public_key", ""),
        ("public_key", "   "),
        ("public_key", 64),
        ("package_digest_hex", "0" * 63),
        ("package_digest_hex", "F" * 64),
        ("signature_digest_hex", ""),
        ("signature_digest_hex", "0" * 65),
        ("received_at", "2026-01-02"),
        ("received_at", "2026-01-02T03:04:05"),
        ("received_at", "2026-01-02T03:04:05+01:00"),
        ("received_at", 0),
        ("local_available", "true"),
        ("local_available", 1),
        ("local_package_digest_hex", "z" * 64),
        ("local_package_digest_hex", None),
        ("matches", "false"),
        ("matches", 0),
    ],
)
def test_bad_entry_field_is_422(client, field, value):
    entries = _offline_entries()
    entries[0][field] = value
    body = _request_body(entries)
    _assert_validation_error(_verify(client, body))


def test_entry_missing_and_extra_members_are_422(client):
    entries = _offline_entries()
    del entries[0]["public_key"]
    _assert_validation_error(_verify(client, _request_body(entries)))
    entries = _offline_entries()
    entries[0]["signature"] = "AAAA"
    _assert_validation_error(_verify(client, _request_body(entries)))
    entries = _offline_entries()
    entries[0]["package"] = {}
    _assert_validation_error(_verify(client, _request_body(entries)))


def test_any_query_parameter_is_422(client):
    resp = client.post(f"{URL}?anything=1", json=_request_body())
    _assert_validation_error(resp)


@pytest.mark.parametrize("method", ["get", "put", "patch", "delete"])
def test_non_post_methods_are_405(client, method):
    resp = getattr(client, method)(URL)
    assert resp.status_code == 405, resp.text
    assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Statelessness -------------------------------------------------------------------


def test_verification_writes_and_queries_nothing(client, db_session):
    body = _request_body()
    assert _verify(client, body).json() == {"valid": True}
    mismatched = _request_body()
    mismatched["checkpoint"]["entries_digest_hex"] = "0" * 64
    assert _verify(client, mismatched).json()["valid"] is False
    _assert_validation_error(_verify(client, {"checkpoint": {}}))
    db_session.expire_all()
    assert _receipt_count(db_session) == 0
    assert _audit_count(db_session) == 0


# --- Identifier length bounds removed ----------------------------------------------


def test_long_identifiers_have_no_upper_length_bound(client):
    # The previous arbitrary upper bounds (id 80, signer_subject/public_key
    # 255) are removed: non-empty identifiers of any length verify as long
    # as the structure, types, and recomputed digest agree.
    long_id = "irx_" + "a" * 512
    long_subject = "subject-" + "x" * 512
    long_public_key = "k" * 512
    entries = [
        _entry(
            id=long_id,
            signer_subject=long_subject,
            public_key=long_public_key,
        )
    ]
    resp = _verify(client, _request_body(entries=entries))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}
