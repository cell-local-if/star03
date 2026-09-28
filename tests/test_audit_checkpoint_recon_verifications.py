"""Tests for the stateless audit checkpoint recon package verification route.

Covers POST /v1/audit-checkpoint-recon-verifications: the request body
is exactly ``{"checkpoint", "entries"}``; the checkpoint is exactly
``{"checkpoint_version", "digest_algorithm", "entry_count",
"entries_digest_hex"}`` with the version pinned to
"provenance-audit-checkpoint-recon-v1", the algorithm to "sha256", the
count a non-negative integer, and the digest 64 lowercase hex
characters. The entries array must contain exactly ``entry_count``
elements, each exactly the exported seven-field reconciliation entry
(non-empty ``id``/``checkpoint_version``; 64 lowercase hex
``events_digest_hex``; non-negative integer ``event_count``; strict RFC
3339 UTC ``received_at``; strict boolean ``matches``; and a
``local_checkpoint`` exactly the four-field local audit checkpoint with
its version pinned to "provenance-audit-checkpoint-v1"); no undeclared
member is accepted on any object. Verification is decided by structure
and digest alone and never resolves any id against local state. The
digest is the SHA-256 of the canonical entries array (array order
preserved, nested object keys sorted recursively by Unicode code point,
compact separators, non-ASCII unescaped, UTF-8) computed over the
entries exactly as received. Any structural, field, type, count,
timestamp, digest-format, or malformed-JSON violation is a 422
validation_error; a structurally valid request whose digest differs is
200 ``{"valid": false, "computed_digest_hex": ...}``; a match is
exactly ``{"valid": true}``. The route accepts no query parameters and
is fully stateless: it needs no persisted receipt and creates,
modifies, logs, and queries nothing, so unknown resources, repeated
requests, and differing local state verify identically. Non-POST
methods are 405 method_not_allowed. All fixtures are deterministic and
offline.
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
from sqlalchemy import func, select

from provenance import canonical
from provenance.models import AuditEvent, CheckpointImportRecord

PACKAGE_PATH = "/v1/audit-checkpoint-recon-package"
URL = "/v1/audit-checkpoint-recon-verifications"
PACKAGE_VERSION = "provenance-audit-checkpoint-recon-v1"
LOCAL_VERSION = "provenance-audit-checkpoint-v1"
DIGEST_ALGORITHM = "sha256"

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


def _local_checkpoint(
    *,
    checkpoint_version=LOCAL_VERSION,
    digest_algorithm=DIGEST_ALGORITHM,
    event_count=1,
    events_digest_hex=None,
):
    return {
        "checkpoint_version": checkpoint_version,
        "digest_algorithm": digest_algorithm,
        "event_count": event_count,
        "events_digest_hex": hashlib.sha256(b"local-events").hexdigest()
        if events_digest_hex is None
        else events_digest_hex,
    }


def _entry(
    *,
    id="aci_" + "0" * 64,
    checkpoint_version=LOCAL_VERSION,
    events_digest_hex=None,
    event_count=2,
    received_at="2026-01-02T03:04:05Z",
    local_checkpoint=None,
    matches=False,
):
    return {
        "id": id,
        "checkpoint_version": checkpoint_version,
        "events_digest_hex": hashlib.sha256(b"receipt-events").hexdigest()
        if events_digest_hex is None
        else events_digest_hex,
        "event_count": event_count,
        "received_at": received_at,
        "local_checkpoint": _local_checkpoint()
        if local_checkpoint is None
        else local_checkpoint,
        "matches": matches,
    }


def _offline_entries() -> list:
    """A self-contained reconciliation-entry sequence with no service state."""
    return [
        _entry(),
        _entry(
            id="aci_" + "1" * 64,
            checkpoint_version="provenance-audit-checkpoint-v2",
            event_count=99,
            events_digest_hex=hashlib.sha256(b"other-events").hexdigest(),
            received_at="2026-01-02T03:04:06+00:00",
            local_checkpoint=_local_checkpoint(event_count=7),
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


def _verify(client, body):
    return client.post(URL, json=body)


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def _audit_count(db_session) -> int:
    return db_session.scalar(select(func.count()).select_from(AuditEvent))


def _receipt_count(db_session) -> int:
    return db_session.scalar(
        select(func.count()).select_from(CheckpointImportRecord)
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
    body = _request_body(entries)
    body["entries"] = list(reversed(entries))
    resp = _verify(client, body)
    assert resp.status_code == 200, resp.text
    assert resp.json()["valid"] is False
    assert resp.json()["computed_digest_hex"] == _digest_of(
        list(reversed(entries))
    )
    # A different spelling of the same instant is a different commitment.
    respelled = copy.deepcopy(entries)
    respelled[0]["received_at"] = "2026-01-02T03:04:05+00:00"
    body = _request_body(entries)
    body["entries"] = respelled
    resp = _verify(client, body)
    assert resp.status_code == 200, resp.text
    assert resp.json()["valid"] is False
    assert resp.json()["computed_digest_hex"] == _digest_of(respelled)


def test_nested_object_key_order_is_canonically_irrelevant(client):
    entries = _offline_entries()
    reordered = copy.deepcopy(entries)
    nested = reordered[1]["local_checkpoint"]
    reordered[1]["local_checkpoint"] = {
        key: nested[key] for key in reversed(list(nested))
    }
    body = _request_body(entries)
    body["entries"] = reordered
    # Nested keys sort by Unicode code point, so the wire key order of
    # local_checkpoint cannot change the recomputed digest.
    assert _verify(client, body).json() == {"valid": True}


def test_non_ascii_is_digested_unescaped(client):
    entries = [
        _entry(id="aci_" + "a" * 64, checkpoint_version="système-✓")
    ]
    resp = _verify(client, _request_body(entries=entries))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}
    # The ASCII-escaped canonicalization is a different commitment.
    escaped = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    escaped_digest = hashlib.sha256(escaped).hexdigest()
    body = _request_body(entries=entries, entries_digest_hex=escaped_digest)
    verdict = _verify(client, body).json()
    assert verdict["valid"] is False
    assert verdict["computed_digest_hex"] == _digest_of(entries)


def test_verification_uses_the_request_body_alone(client, db_session):
    # The same offline body verifies identically before and after local
    # receipts exist, and needs no persisted state at all.
    body = _request_body()
    assert _verify(client, body).json() == {"valid": True}
    # Materialize an unrelated receipt directly.
    from datetime import datetime, timezone

    from tests.test_audit_checkpoint_import_reconciliation import (
        _insert_receipt,
    )

    _insert_receipt(
        db_session, LOCAL_VERSION, 0, "0" * 64
    )
    assert _verify(client, body).json() == {"valid": True}
    assert _verify(client, body).json() == {"valid": True}
    # The fabricated entry ids are never resolved: unknown resources do
    # not change the verdict.
    assert _receipt_count(db_session) == 1


def test_exported_package_verifies(client, db_session):
    from datetime import datetime, timezone

    from tests.test_audit_checkpoint_import_reconciliation import (
        _insert_event,
        _insert_receipt,
    )

    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    _insert_receipt(db_session, "provenance-audit-checkpoint-v2", 3, "0" * 64)
    package = client.get(PACKAGE_PATH).json()
    assert _verify(client, package).json() == {"valid": True}


# --- Structure validation -----------------------------------------------------------


def test_malformed_json_is_422(client):
    resp = client.post(
        URL, content=b"{not json", headers={"Content-Type": "application/json"}
    )
    _assert_validation_error(resp)


@pytest.mark.parametrize(
    "body", [b"", b"[]", b'"x"', b"1", b"null", b"true"]
)
def test_non_object_or_empty_body_is_422(client, body):
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
        {"checkpoint_version": "provenance-audit-checkpoint-v1"},
        {"checkpoint_version": "pir-checkpoint-v1"},
        {"digest_algorithm": "sha512"},
        {"entry_count": -1},
        {"entry_count": 1.5},
        {"entry_count": "2"},
        {"entry_count": True},
        {"entries_digest_hex": "0" * 63},
        {"entries_digest_hex": "0" * 65},
        {"entries_digest_hex": "A" * 64},
        {"entries_digest_hex": "g" * 64},
        {"entries_digest_hex": None},
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
        ("id", None),
        ("checkpoint_version", ""),
        ("checkpoint_version", "  "),
        ("checkpoint_version", 5),
        ("events_digest_hex", "0" * 63),
        ("events_digest_hex", "F" * 64),
        ("events_digest_hex", ""),
        ("event_count", -1),
        ("event_count", 1.5),
        ("event_count", "2"),
        ("event_count", True),
        ("received_at", "2026-01-02"),
        ("received_at", "2026-01-02T03:04:05"),
        ("received_at", "2026-01-02T03:04:05+01:00"),
        ("received_at", "2026-01-02T03:04:05z"),
        ("received_at", 0),
        ("matches", "false"),
        ("matches", 0),
        ("matches", None),
        ("local_checkpoint", None),
        ("local_checkpoint", {}),
        ("local_checkpoint", "not-an-object"),
    ],
)
def test_bad_entry_field_is_422(client, field, value):
    entries = _offline_entries()
    entries[0][field] = value
    body = _request_body(entries)
    _assert_validation_error(_verify(client, body))


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_version", "provenance-audit-checkpoint-v2"),
        ("checkpoint_version", LOCAL_VERSION.upper()),
        ("digest_algorithm", "sha512"),
        ("event_count", -1),
        ("event_count", 2.0),
        ("event_count", "1"),
        ("events_digest_hex", "z" * 64),
        ("events_digest_hex", "A" * 64),
        ("events_digest_hex", "0" * 63),
    ],
)
def test_bad_local_checkpoint_field_is_422(client, field, value):
    entries = _offline_entries()
    entries[0]["local_checkpoint"][field] = value
    body = _request_body(entries)
    _assert_validation_error(_verify(client, body))


def test_local_checkpoint_missing_and_extra_members_are_422(client):
    entries = _offline_entries()
    del entries[0]["local_checkpoint"]["event_count"]
    _assert_validation_error(_verify(client, _request_body(entries)))
    entries = _offline_entries()
    entries[0]["local_checkpoint"]["extra"] = 1
    _assert_validation_error(_verify(client, _request_body(entries)))


def test_entry_missing_and_extra_members_are_422(client):
    entries = _offline_entries()
    del entries[0]["matches"]
    _assert_validation_error(_verify(client, _request_body(entries)))
    # Raw-material or undeclared fields never enter the request, under any
    # plausible name.
    for name in ("events", "signature", "payload", "content", "evidence"):
        entries = _offline_entries()
        entries[0][name] = {} if name != "signature" else "AAAA"
        _assert_validation_error(_verify(client, _request_body(entries)))
    # An extra nested raw-material field is rejected just the same.
    entries = _offline_entries()
    entries[0]["local_checkpoint"]["evidence_bytes"] = "AAAA"
    _assert_validation_error(_verify(client, _request_body(entries)))


def test_entries_must_be_an_array(client):
    body = _request_body()
    body["entries"] = {"0": _offline_entries()[0]}
    _assert_validation_error(_verify(client, body))


def test_any_query_parameter_is_422(client):
    resp = client.post(f"{URL}?anything=1", json=_request_body())
    _assert_validation_error(resp)
    resp = client.post(f"{URL}?a=1&a=2", json=_request_body())
    _assert_validation_error(resp)


@pytest.mark.parametrize("method", ["get", "put", "patch", "delete"])
def test_non_post_methods_are_405(client, method):
    resp = getattr(client, method)(URL)
    assert resp.status_code == 405, resp.text
    assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Statelessness -------------------------------------------------------------------


def test_verification_writes_and_queries_nothing(client, db_session):
    assert _verify(client, _request_body()).json() == {"valid": True}
    mismatched = _request_body()
    mismatched["checkpoint"]["entries_digest_hex"] = "0" * 64
    assert _verify(client, mismatched).json()["valid"] is False
    _assert_validation_error(_verify(client, {"checkpoint": {}}))
    _assert_validation_error(
        client.post(
            URL, content=b"{bad", headers={"Content-Type": "application/json"}
        )
    )
    db_session.expire_all()
    assert _receipt_count(db_session) == 0
    assert _audit_count(db_session) == 0


# --- Identifier length bounds removed ----------------------------------------------


def test_long_identifiers_have_no_upper_length_bound(client):
    # Non-empty identifiers of any length verify as long as the
    # structure, types, and recomputed digest agree.
    entries = [
        _entry(
            id="aci_" + "a" * 512,
            checkpoint_version="subject-" + "x" * 512,
        )
    ]
    resp = _verify(client, _request_body(entries=entries))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}
