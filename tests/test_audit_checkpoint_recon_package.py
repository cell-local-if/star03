"""Tests for the read-only audit checkpoint recon package export.

Covers GET /v1/audit-checkpoint-recon-package: the success body is
exactly ``{"checkpoint", "entries"}`` in that member order; ``checkpoint``
is exactly ``{"checkpoint_version", "digest_algorithm", "entry_count",
"entries_digest_hex"}`` with the version pinned to
"provenance-audit-checkpoint-recon-v1" and the algorithm to "sha256";
``entries`` lists every checkpoint-import receipt's current reconciliation
in stable creation order, each with exactly the seven fields of the
existing global reconciliation public view
(``id``/``checkpoint_version``/``events_digest_hex``/``event_count``/
``received_at`` plus the read-time ``local_checkpoint`` and ``matches``);
the imported event array is never carried. The checkpoint digest is
independently recomputed from the entries member of the same response
under the checkpoint canonical rules (array order kept, nested object
keys sorted recursively by Unicode code point, compact separators,
unescaped non-ASCII, UTF-8); an empty set returns ``"entries": []`` with
``entry_count`` 0 and the deterministic digest of the empty array;
repeated reads and reads across a restart return the same package for the
same persisted state; and neither successful nor failed reads write any
receipt, resource, or audit row. The route takes no body and no query
parameters: any body byte or any (including repeated) parameter is a 422
validation_error. Non-GET methods are 405 method_not_allowed. All
fixtures are deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance import canonical
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import AuditEvent, CheckpointImportRecord
from tests.test_audit_checkpoint_import_reconciliation import (
    CHECKPOINT_VERSION,
    EMPTY_DIGEST,
    IMPORTS_URL,
    _insert_event,
    _insert_receipt,
    _offline_request,
)

PACKAGE_PATH = "/v1/audit-checkpoint-recon-package"
VERIFY_PATH = "/v1/audit-checkpoint-recon-verifications"
RECON_LIST_PATH = "/v1/audit-events/checkpoint-import-reconciliations"

PACKAGE_VERSION = "provenance-audit-checkpoint-recon-v1"
DIGEST_ALGORITHM = "sha256"

PACKAGE_FIELDS = ["checkpoint", "entries"]
CHECKPOINT_FIELDS = [
    "checkpoint_version",
    "digest_algorithm",
    "entry_count",
    "entries_digest_hex",
]
ENTRY_FIELDS = {
    "id",
    "checkpoint_version",
    "events_digest_hex",
    "event_count",
    "received_at",
    "local_checkpoint",
    "matches",
}
LOCAL_CHECKPOINT_FIELDS = {
    "checkpoint_version",
    "digest_algorithm",
    "event_count",
    "events_digest_hex",
}
EMPTY_ARRAY_DIGEST = hashlib.sha256(b"[]").hexdigest()


def _package(client, **params):
    return client.get(PACKAGE_PATH, params=params)


def _recon_list_items(client) -> list:
    """The full global reconciliation list (the collection is unpaginated)."""
    resp = client.get(RECON_LIST_PATH)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == len(body["items"])
    assert body["next_cursor"] is None
    return body["items"]


def _register(client, request: dict) -> dict:
    resp = client.post(IMPORTS_URL, json=request)
    assert resp.status_code in (200, 201), resp.text
    return resp.json()


def _import(client, label: str, count: int = 2) -> dict:
    """Register one receipt for a fabricated offline checkpoint."""
    events = [
        {
            "event_type": "actor.created",
            "resource_id": f"res-{label}-{i}",
            "created_at": f"2026-01-02T03:04:{i:02d}Z",
        }
        for i in range(count)
    ]
    return _register(client, _offline_request(events))


def _import_many(client, count: int):
    return [_import(client, f"bulk-{i:03d}", count=1) for i in range(count)]


def _receipt_count(db_session) -> int:
    return db_session.scalar(
        select(func.count()).select_from(CheckpointImportRecord)
    )


def _audit_count(db_session) -> int:
    return db_session.scalar(select(func.count()).select_from(AuditEvent))


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- Baseline export ------------------------------------------------------------


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
    _import_many(client, 3)
    resp = _package(client)
    assert resp.status_code == 200, resp.text
    raw = resp.content
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b", " not in raw
    assert b": " not in raw
    assert raw.startswith(b'{"checkpoint":{')


def test_entries_are_exactly_the_global_reconciliation_public_view(client):
    r1, r2, r3 = _import_many(client, 3)
    body = _package(client).json()
    assert body["checkpoint"]["entry_count"] == 3
    entries = body["entries"]
    list_items = _recon_list_items(client)
    # The entries member is exactly the global reconciliation view, in
    # stable creation order.
    assert entries == list_items
    assert [entry["id"] for entry in entries] == [r1["id"], r2["id"], r3["id"]]
    for entry, receipt in zip(entries, (r1, r2, r3)):
        assert set(entry) == ENTRY_FIELDS
        assert entry["checkpoint_version"] == receipt["checkpoint_version"]
        assert entry["events_digest_hex"] == receipt["events_digest_hex"]
        assert entry["event_count"] == receipt["event_count"]
        assert entry["received_at"] == receipt["received_at"]
        assert set(entry["local_checkpoint"]) == LOCAL_CHECKPOINT_FIELDS
        assert isinstance(entry["matches"], bool)
    # The imported event array never appears in any shape.
    serialized = json.dumps(body)
    assert '"events"' not in serialized
    assert "bulk-" not in serialized


def test_local_checkpoint_and_matches_agree_with_single_reconciliation(client):
    receipts = _import_many(client, 3)
    entries = {e["id"]: e for e in _package(client).json()["entries"]}
    for receipt in receipts:
        single = client.get(
            f"{IMPORTS_URL}/{receipt['id']}/reconciliation"
        ).json()
        entry = entries[receipt["id"]]
        assert entry["local_checkpoint"] == single["local_checkpoint"]
        assert entry["matches"] == single["matches"]


def test_directly_inserted_receipts_match_against_current_state(
    client, db_session
):
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    local = client.get("/v1/audit-events/checkpoint").json()
    matching = _insert_receipt(
        db_session,
        local["checkpoint_version"],
        local["event_count"],
        local["events_digest_hex"],
    )
    wrong_state = _insert_receipt(db_session, CHECKPOINT_VERSION, 99, "0" * 64)
    empty = _insert_receipt(db_session, CHECKPOINT_VERSION, 0, EMPTY_DIGEST)

    entries = {e["id"]: e for e in _package(client).json()["entries"]}
    assert set(entries) == {matching, wrong_state, empty}
    assert len(entries) == 3
    verdicts = {rid: e["matches"] for rid, e in entries.items()}
    assert verdicts[matching] is True
    assert verdicts[wrong_state] is False
    assert verdicts[empty] is False
    assert all(
        e["local_checkpoint"] == local for e in entries.values()
    )


def test_entries_follow_stable_creation_order(client):
    receipts = [_import(client, f"order-{i}") for i in range(5)]
    entries = _package(client).json()["entries"]
    assert [e["id"] for e in entries] == [r["id"] for r in receipts]


def test_checkpoint_digest_binds_exactly_the_served_entries(client):
    _import_many(client, 3)
    body = _package(client).json()
    recomputed = canonical.audit_checkpoint_recon_entries_digest_hex(
        body["entries"]
    )
    assert body["checkpoint"]["entries_digest_hex"] == recomputed
    assert body["checkpoint"]["entry_count"] == len(body["entries"])

    # Mutating the served array breaks the binding.
    reordered = list(reversed(body["entries"]))
    assert recomputed != canonical.audit_checkpoint_recon_entries_digest_hex(
        reordered
    )
    dropped = body["entries"][:-1]
    assert recomputed != canonical.audit_checkpoint_recon_entries_digest_hex(
        dropped
    )


def test_digest_sorts_nested_keys_and_keeps_non_ascii(client):
    # The digest canonicalizes recursively: the nested local_checkpoint
    # object's keys participate in code-point order regardless of wire
    # order, and non-ASCII stays unescaped.
    _import_many(client, 1)
    body = _package(client).json()
    entries = body["entries"]
    expected = hashlib.sha256(
        json.dumps(
            entries,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert body["checkpoint"]["entries_digest_hex"] == expected
    # A reorder of the nested object's keys leaves the digest unchanged.
    respelled = [dict(entry) for entry in entries]
    nested = respelled[0]["local_checkpoint"]
    respelled[0]["local_checkpoint"] = {
        key: nested[key] for key in reversed(list(nested))
    }
    assert (
        canonical.audit_checkpoint_recon_entries_digest_hex(respelled)
        == expected
    )


def test_export_verifies_offline(client):
    _import_many(client, 3)
    package = _package(client).json()
    resp = client.post(VERIFY_PATH, json=package)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_repeated_reads_return_the_same_package(client):
    _import_many(client, 3)
    assert _package(client).content == _package(client).content


def test_restart_returns_the_same_export(client, tmp_db_url):
    producer = create_app(Settings(database_url=tmp_db_url))
    with TestClient(producer) as first:
        # Direct, fixed-instant receipts so the snapshot is identical
        # across a restart.
        session = first.app.state.session_factory()
        try:
            _insert_receipt(session, CHECKPOINT_VERSION, 0, EMPTY_DIGEST)
            _insert_receipt(
                session, "provenance-audit-checkpoint-v2", 0, EMPTY_DIGEST
            )
        finally:
            session.close()
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
        "local_available=false",
        "checkpoint_version=" + CHECKPOINT_VERSION,
    ],
)
def test_any_query_parameter_is_422(client, suffix):
    _assert_validation_error(client.get(f"{PACKAGE_PATH}?{suffix}"))


def test_repeated_query_parameter_is_422(client):
    _assert_validation_error(client.get(f"{PACKAGE_PATH}?a=1&a=2"))


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_non_get_methods_are_405(client, method):
    resp = getattr(client, method)(PACKAGE_PATH)
    assert resp.status_code == 405, resp.text
    assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Read-only guarantees ---------------------------------------------------------


def test_successful_and_failed_exports_write_nothing(client, db_session):
    _import_many(client, 3)
    db_session.expire_all()
    receipts_before = _receipt_count(db_session)
    audits_before = _audit_count(db_session)
    assert _package(client).status_code == 200
    _assert_validation_error(client.get(f"{PACKAGE_PATH}?limit=1"))
    _assert_validation_error(client.request("GET", PACKAGE_PATH, content=b"{}"))
    db_session.expire_all()
    assert _receipt_count(db_session) == receipts_before
    assert _audit_count(db_session) == audits_before


def test_existing_reconciliation_routes_are_unchanged(client):
    receipt = _import(client, "compat")
    package = _package(client).json()
    # The single and paginated reconciliation views keep their shapes and
    # agree with the new package's entries.
    single = client.get(f"{IMPORTS_URL}/{receipt['id']}/reconciliation")
    assert single.status_code == 200
    assert set(single.json()) == {"import_id", "local_checkpoint", "matches"}

    page = client.get(RECON_LIST_PATH).json()
    assert set(page) == {"items", "count", "next_cursor"}
    assert page["items"] == package["entries"]
