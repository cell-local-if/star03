"""Tests for the read-only global checkpoint-import reconciliation package.

Covers GET /v1/audit-checkpoint-recon-package: the success body is exactly
{"checkpoint", "entries"} in that member order; ``checkpoint`` is exactly
{"checkpoint_version", "digest_algorithm", "entry_count",
"entries_digest_hex"} with the version pinned to "pacr-checkpoint-v1"
and the algorithm to "sha256"; ``entries`` lists every existing
audit checkpoint import receipt's global reconciliation view in stable
creation order -- each with exactly the seven fields
id/checkpoint_version/events_digest_hex/event_count/received_at plus
the read-time local_checkpoint and matches served by the existing
GET /v1/audit-events/checkpoint-import-reconciliations listing (the
imported event array is never carried). The checkpoint digest is
independently recomputed from the entries member of the same response
under the checkpoint canonical rules (array order kept, nested object
keys sorted recursively by Unicode code point, compact separators,
unescaped non-ASCII, UTF-8); an empty database returns "entries": []
with entry_count 0 and the deterministic digest of the empty array;
repeated reads and reads across a restart return the same package for
the same persisted state; and neither successful nor failed exports
write any receipt, resource, or audit row. Any query parameter and any
non-empty request body are 422 validation_error. Non-GET methods are
405 method_not_allowed. All fixtures are deterministic and offline.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance import canonical
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import AuditEvent, CheckpointImportRecord
from tests.test_audit_checkpoint_import_reconciliation import (
    CHECKPOINT_URL,
    CHECKPOINT_VERSION,
    EMPTY_DIGEST,
    IMPORTS_URL,
    _digest_of,
    _insert_event,
    _insert_receipt,
    _offline_request,
    _served_checkpoint,
)

PACKAGE_PATH = "/v1/audit-checkpoint-recon-package"
VERIFY_PATH = "/v1/audit-checkpoint-recon-verifications"
RECONCILIATIONS_PATH = (
    "/v1/audit-events/checkpoint-import-reconciliations"
)

PACKAGE_VERSION = "pacr-checkpoint-v1"
DIGEST_ALGORITHM = "sha256"

PACKAGE_FIELDS = ["checkpoint", "entries"]
CHECKPOINT_FIELDS = [
    "checkpoint_version",
    "digest_algorithm",
    "entry_count",
    "entries_digest_hex",
]
ENTRY_FIELDS = [
    "id",
    "checkpoint_version",
    "events_digest_hex",
    "event_count",
    "received_at",
    "local_checkpoint",
    "matches",
]
EMPTY_ARRAY_DIGEST = hashlib.sha256(b"[]").hexdigest()


def _package(client, **params):
    return client.get(PACKAGE_PATH, params=params)


def _register(client, request: dict) -> dict:
    resp = client.post(IMPORTS_URL, json=request)
    assert resp.status_code in (200, 201), resp.text
    return resp.json()


def _walk_reconciliations(client):
    """Every item of the existing global reconciliation listing."""
    items = []
    cursor = None
    for _ in range(100):
        params = {"limit": 2}
        if cursor is not None:
            params["cursor"] = cursor
        resp = client.get(RECONCILIATIONS_PATH, params=params)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        items.extend(body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            return items
    raise AssertionError("pagination did not terminate")


def _receipt_count(db_session) -> int:
    return db_session.scalar(
        select(func.count()).select_from(CheckpointImportRecord)
    )


def _audit_count(db_session) -> int:
    return db_session.scalar(select(func.count()).select_from(AuditEvent))


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- Baseline export -------------------------------------------------------------


def test_empty_state_returns_empty_entries_and_empty_array_digest(client):
    resp = _package(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert list(body) == PACKAGE_FIELDS
    checkpoint = body["checkpoint"]
    assert list(checkpoint) == CHECKPOINT_FIELDS
    assert checkpoint["checkpoint_version"] == PACKAGE_VERSION
    assert checkpoint["digest_algorithm"] == DIGEST_ALGORITHM
    assert checkpoint["entry_count"] == 0
    assert checkpoint["entries_digest_hex"] == EMPTY_ARRAY_DIGEST
    assert body["entries"] == []


def test_response_is_compact_json_with_exactly_one_trailing_newline(client):
    _register(client, _offline_request())
    resp = _package(client)
    assert resp.status_code == 200, resp.text
    raw = resp.content
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b", " not in raw
    assert b": " not in raw
    assert raw.startswith(b'{"checkpoint":{')


def _three_receipts(client) -> list[dict]:
    """Register three distinct receipts (distinct claimed identities)."""
    receipts = []
    for i in range(3):
        events = [
            {
                "event_type": "actor.created",
                "resource_id": f"res-recon-{i}",
                "created_at": f"2026-01-02T03:04:{i:02d}Z",
            }
        ]
        receipts.append(_register(client, _offline_request(events)))
    return receipts


def test_entries_are_exactly_the_global_reconciliation_listing_views(client):
    receipts = _three_receipts(client)
    body = _package(client).json()
    assert body["checkpoint"]["entry_count"] == 3
    entries = body["entries"]
    listing = _walk_reconciliations(client)
    assert entries == listing
    # Stable creation order, one entry per receipt.
    assert [entry["id"] for entry in entries] == [r["id"] for r in receipts]
    served = _served_checkpoint(client)
    for entry, receipt in zip(entries, receipts):
        assert list(entry) == ENTRY_FIELDS
        # The receipt half is exactly the existing receipt public view.
        assert entry["id"] == receipt["id"]
        assert entry["checkpoint_version"] == receipt["checkpoint_version"]
        assert entry["events_digest_hex"] == receipt["events_digest_hex"]
        assert entry["event_count"] == receipt["event_count"]
        assert entry["received_at"] == receipt["received_at"]
        # The local reconciliation triple agrees with the single view.
        assert entry["local_checkpoint"] == served
        assert entry["matches"] is (
            entry["checkpoint_version"] == served["checkpoint_version"]
            and entry["event_count"] == served["event_count"]
            and entry["events_digest_hex"] == served["events_digest_hex"]
        )
    # The imported event array and checkpoint envelope never appear.
    for entry in entries:
        assert "events" not in entry
        assert "import_id" not in entry


def test_entry_digest_is_independently_reproducible_and_non_ascii_is_unescaped(
    client, db_session
):
    # A receipt with a non-ASCII claimed version inserted directly: the
    # canonical digest must reproduce from the entries alone and the wire
    # bytes carry the non-ASCII characters unescaped.
    _insert_receipt(db_session, "v-快照-é", 2, "a" * 64)
    resp = _package(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    recomputed = canonical.audit_checkpoint_recon_entries_digest_hex(
        body["entries"]
    )
    assert body["checkpoint"]["entries_digest_hex"] == recomputed
    assert body["checkpoint"]["entry_count"] == len(body["entries"])
    assert "快照".encode("utf-8") in resp.content
    assert "\\u" not in resp.text


def test_checkpoint_digest_binds_exactly_the_served_entries(client):
    _register(client, _offline_request())
    _register(client, _offline_request())
    body = _package(client).json()
    recomputed = canonical.audit_checkpoint_recon_entries_digest_hex(
        body["entries"]
    )
    assert body["checkpoint"]["entries_digest_hex"] == recomputed
    assert body["checkpoint"]["entry_count"] == len(body["entries"])


def test_export_verifies_offline(client):
    _register(client, _offline_request())
    package = _package(client).json()
    resp = client.post(VERIFY_PATH, json=package)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_local_checkpoint_in_entries_is_the_unfiltered_served_checkpoint(
    client, db_session
):
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    _insert_receipt(db_session, CHECKPOINT_VERSION, 99, "0" * 64)
    served = _served_checkpoint(client)
    entries = _package(client).json()["entries"]
    assert len(entries) == 1
    assert entries[0]["local_checkpoint"] == served
    assert entries[0]["matches"] is False
    # The local checkpoint is independently reproducible from the audit
    # event listing, including its canonical events digest.
    assert served["events_digest_hex"]


def test_repeated_reads_return_the_same_package(client):
    _register(client, _offline_request())
    assert _package(client).content == _package(client).content


def test_restart_returns_the_same_export(client, tmp_db_url):
    producer = create_app(Settings(database_url=tmp_db_url))
    with TestClient(producer) as first:
        _register(first, _offline_request())
        _register(first, _offline_request())
        before = first.get(PACKAGE_PATH)
        assert before.status_code == 200, before.text
    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as second:
        after = second.get(PACKAGE_PATH)
        assert after.status_code == 200, after.text
        assert after.content == before.content


# --- Validation ------------------------------------------------------------------


@pytest.mark.parametrize("body", [b"{}", b" ", b"x", b"\x00"])
def test_non_empty_body_is_422(client, body):
    resp = client.request("GET", PACKAGE_PATH, content=body)
    _assert_validation_error(resp)


@pytest.mark.parametrize(
    "suffix",
    [
        "unknown=1",
        "limit=10",
        "cursor=abc",
        "matches=true",
        "checkpoint_version=" + CHECKPOINT_VERSION,
        "a=1&a=2",
    ],
)
def test_any_query_parameter_is_422(client, suffix):
    _assert_validation_error(client.get(f"{PACKAGE_PATH}?{suffix}"))


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_non_get_methods_are_405(client, method):
    resp = getattr(client, method)(PACKAGE_PATH)
    assert resp.status_code == 405, resp.text
    assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Read-only guarantees ---------------------------------------------------------


def test_successful_and_failed_exports_write_nothing(client, db_session):
    _register(client, _offline_request())
    db_session.expire_all()
    receipts_before = _receipt_count(db_session)
    audits_before = _audit_count(db_session)
    assert _package(client).status_code == 200
    _assert_validation_error(client.get(f"{PACKAGE_PATH}?limit=1"))
    _assert_validation_error(client.request("GET", PACKAGE_PATH, content=b"{}"))
    _assert_validation_error(client.get(f"{PACKAGE_PATH}?unknown=1"))
    db_session.expire_all()
    assert _receipt_count(db_session) == receipts_before
    assert _audit_count(db_session) == audits_before


def test_empty_database_export_matches_empty_array_digest(client):
    # The empty snapshot is the deterministic digest of [] and still
    # verifies offline.
    body = _package(client).json()
    assert body == {
        "checkpoint": {
            "checkpoint_version": PACKAGE_VERSION,
            "digest_algorithm": DIGEST_ALGORITHM,
            "entry_count": 0,
            "entries_digest_hex": EMPTY_DIGEST,
        },
        "entries": [],
    }
    assert EMPTY_ARRAY_DIGEST == EMPTY_DIGEST
