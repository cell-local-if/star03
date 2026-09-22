"""Tests for the read-only checkpoint-import reconciliation listing endpoint.

Covers GET /v1/audit-events/checkpoint-import-reconciliations: the response
shape carries exactly ``{"items", "count", "next_cursor"}``. Each item is the
existing single-receipt public view
(``id``/``checkpoint_version``/``events_digest_hex``/``event_count``/
``received_at``) plus ``local_checkpoint`` and ``matches``. Receipts follow
stable creation order and ``count`` is the total number of receipts. Every
item is reconciled against the current local audit checkpoint computed over
the complete, unfiltered local sequence under the existing
``GET /v1/audit-events/checkpoint`` rules; ``matches`` is true only when the
receipt's version, event count, and digest all equal that checkpoint.
``limit`` is 1..100 defaulting to 50; the opaque HMAC cursor binds the limit
so pages concatenate without gaps or duplicates and the final cursor is
null. Any other, repeated, blank, or illegal parameter and any
malformed/tampered/foreign/mismatching cursor is 422 validation_error; no
receipts is an empty collection. The route is strictly read-only -- no
resource, receipt, or audit row is written -- and the imported event array
is never echoed. All fixtures are deterministic and offline.
"""

from __future__ import annotations

import base64
import hashlib
import hmac as hmac_mod
import json
import secrets
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance import pagination
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import AuditEvent, CheckpointImportRecord
from tests.test_audit_checkpoint_import_reconciliation import (
    CHECKPOINT_KEYS,
    CHECKPOINT_URL,
    CHECKPOINT_VERSION,
    EMPTY_DIGEST,
    EVENTS_URL,
    IMPORTS_URL,
    _digest_of,
    _insert_event,
    _insert_receipt,
    _offline_request,
    _served_checkpoint,
    _served_events,
)
from tests.test_audit_checkpoint_imports import RECEIPT_KEYS

URL = "/v1/audit-events/checkpoint-import-reconciliations"
RECONCILIATION_ITEM_KEYS = RECEIPT_KEYS | {"local_checkpoint", "matches"}


def _list(client, **params):
    return client.get(URL, params=params)


def _walk_pages(client, **params):
    """Follow next_cursor until exhausted; return (all_items, pages, count)."""
    pages = []
    all_items = []
    count = None
    cursor = None
    for _ in range(100):
        query = {**params}
        if cursor is not None:
            query["cursor"] = cursor
        resp = _list(client, **query)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        count = body["count"]
        pages.append(body["items"])
        all_items.extend(body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            break
    return all_items, pages, count


def _register(client, request: dict) -> dict:
    resp = client.post(IMPORTS_URL, json=request)
    assert resp.status_code in (200, 201), resp.text
    return resp.json()


def _events(label: str, count: int) -> list:
    """A deterministic, self-contained event sequence of ``count`` events."""
    return [
        {
            "event_type": "actor.created",
            "resource_id": f"res-{label}-{i}",
            "created_at": f"2026-01-02T03:04:{i:02d}Z",
        }
        for i in range(count)
    ]


def _import(client, label: str, count: int = 2) -> dict:
    """Register one receipt for a fabricated offline checkpoint."""
    return _register(client, _offline_request(_events(label, count)))


def _import_many(client, count: int):
    return [_import(client, f"bulk-{i:03d}", count=1) for i in range(count)]


def _single_reconciliation(client, import_id: str) -> dict:
    resp = client.get(f"{IMPORTS_URL}/{import_id}/reconciliation")
    assert resp.status_code == 200, resp.text
    return resp.json()


# --- Response shape, fields, and ordering -------------------------------------


def test_empty_database_is_an_empty_collection(client):
    assert _list(client).json() == {
        "items": [],
        "count": 0,
        "next_cursor": None,
    }


def test_response_shape_and_item_fields(client, db_session):
    receipt_id = _insert_receipt(
        db_session, CHECKPOINT_VERSION, 0, EMPTY_DIGEST
    )
    resp = _list(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"items", "count", "next_cursor"}
    assert body["count"] == 1
    assert body["next_cursor"] is None
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert set(item) == RECONCILIATION_ITEM_KEYS
    assert item["id"] == receipt_id
    assert item["id"].startswith("aci_")
    assert item["checkpoint_version"] == CHECKPOINT_VERSION
    assert set(item["local_checkpoint"]) == CHECKPOINT_KEYS


def test_item_carries_the_existing_receipt_public_view(client):
    receipt = _import(client, "public-view")
    item = _list(client).json()["items"][0]
    # The receipt half of the item is exactly the existing single-receipt
    # view, including its original received_at.
    assert {key: item[key] for key in RECEIPT_KEYS} == receipt
    assert (
        {key: item[key] for key in RECEIPT_KEYS}
        == client.get(f"{IMPORTS_URL}/{receipt['id']}").json()
    )
    # The event array and checkpoint envelope stay absent from the item.
    assert "events" not in item
    assert "checkpoint" not in item
    assert "digest_algorithm" not in item
    assert "import_id" not in item


def test_received_at_is_utc(client):
    _import(client, "utc")
    for item in _list(client).json()["items"]:
        parsed = datetime.fromisoformat(item["received_at"])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)
        assert item["received_at"].endswith(("Z", "+00:00"))


def test_receipts_follow_stable_creation_order(client):
    receipts = [_import(client, f"title-{i}") for i in range(5)]
    items = _list(client).json()["items"]
    assert [i["id"] for i in items] == [r["id"] for r in receipts]


def test_count_is_total_receipt_count_on_every_page(client):
    _import_many(client, 5)
    all_items, pages, count = _walk_pages(client, limit=2)
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    assert len(all_items) == 5


# --- Per-item reconciliation semantics -----------------------------------------


def test_item_local_checkpoint_is_the_unfiltered_served_checkpoint(
    client, db_session
):
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    _insert_event(
        db_session,
        "content.created",
        "cnt_x",
        datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    _insert_receipt(db_session, "provenance-audit-checkpoint-v2", 99, "0" * 64)

    served = _served_checkpoint(client)
    items = _list(client).json()["items"]
    assert len(items) == 1
    # The local checkpoint is exactly the unfiltered served checkpoint, and
    # is independently reproducible from the audit-event listing.
    assert items[0]["local_checkpoint"] == served
    served_events = _served_events(client)
    assert items[0]["local_checkpoint"]["event_count"] == len(served_events)
    assert items[0]["local_checkpoint"]["events_digest_hex"] == _digest_of(
        served_events
    )
    # A filter on the checkpoint route excludes an event but never narrows
    # the reconciliation's unfiltered sequence.
    filtered = client.get(CHECKPOINT_URL, params={"event_type": "actor.created"})
    assert filtered.json()["event_count"] == 1
    assert _list(client).json()["items"][0]["local_checkpoint"] == served


def test_matching_receipt_item(client, db_session):
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    checkpoint = _served_checkpoint(client)
    _insert_receipt(
        db_session,
        checkpoint["checkpoint_version"],
        checkpoint["event_count"],
        checkpoint["events_digest_hex"],
    )
    item = _list(client).json()["items"][0]
    assert item["local_checkpoint"] == checkpoint
    assert item["matches"] is True


def test_each_identity_field_is_required_for_a_match(client, db_session):
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    current = _served_checkpoint(client)

    version_id = _insert_receipt(
        db_session,
        "provenance-audit-checkpoint-v2",
        current["event_count"],
        current["events_digest_hex"],
    )
    count_id = _insert_receipt(
        db_session,
        current["checkpoint_version"],
        current["event_count"] + 1,
        current["events_digest_hex"],
    )
    digest_id = _insert_receipt(
        db_session,
        current["checkpoint_version"],
        current["event_count"],
        "0" * 64,
    )
    items = {item["id"]: item for item in _list(client).json()["items"]}
    for receipt_id in (version_id, count_id, digest_id):
        assert items[receipt_id]["local_checkpoint"] == current
        assert items[receipt_id]["matches"] is False


def test_items_agree_with_the_single_reconciliation_route(client, db_session):
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    _insert_event(
        db_session,
        "content.created",
        "cnt_x",
        datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    current = _served_checkpoint(client)

    # One receipt matching the current local checkpoint exactly.
    matching = _insert_receipt(
        db_session,
        current["checkpoint_version"],
        current["event_count"],
        current["events_digest_hex"],
    )
    # Two receipts that cannot match.
    wrong_version = _insert_receipt(
        db_session,
        "provenance-audit-checkpoint-v2",
        current["event_count"],
        current["events_digest_hex"],
    )
    wrong_state = _insert_receipt(
        db_session, CHECKPOINT_VERSION, 99, "a" * 64
    )

    items = {item["id"]: item for item in _list(client).json()["items"]}
    assert set(items) == {matching, wrong_version, wrong_state}
    for receipt_id, item in items.items():
        single = _single_reconciliation(client, receipt_id)
        assert item["local_checkpoint"] == single["local_checkpoint"]
        assert item["matches"] == single["matches"]

    assert items[matching]["matches"] is True
    assert items[wrong_version]["matches"] is False
    assert items[wrong_state]["matches"] is False


def test_local_trail_growing_after_registration_flips_match_to_false(
    client, db_session
):
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    checkpoint = _served_checkpoint(client)
    receipt_id = _insert_receipt(
        db_session,
        checkpoint["checkpoint_version"],
        checkpoint["event_count"],
        checkpoint["events_digest_hex"],
    )
    assert _list(client).json()["items"][0]["matches"] is True

    _insert_event(
        db_session,
        "content.created",
        "cnt_later",
        datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    item = _list(client).json()["items"][0]
    assert item["id"] == receipt_id
    assert item["local_checkpoint"] == _served_checkpoint(client)
    assert item["matches"] is False


def test_reconciliation_reflects_current_state_on_every_page(client):
    # Reconciliation happens per item on every page; a bulk set imported via
    # the API leaves the most recent local sequence one event ahead of each
    # claimed (pre-import) checkpoint, so no item spuriously matches.
    receipts = _import_many(client, 3)
    all_items, _, _ = _walk_pages(client, limit=2)
    assert [i["id"] for i in all_items] == [r["id"] for r in receipts]
    assert all(item["matches"] is False for item in all_items)
    # Every page still carries the same current local checkpoint.
    current = _served_checkpoint(client)
    assert all(item["local_checkpoint"] == current for item in all_items)


def test_response_never_echoes_the_imported_events(client):
    marker = "reconciliation-list-no-echo-marker-7b2e"
    request = _offline_request()
    request["events"][0]["resource_id"] = marker
    request["checkpoint"]["events_digest_hex"] = _digest_of(request["events"])
    _register(client, request)

    serialized = json.dumps(_list(client).json(), ensure_ascii=False)
    assert marker not in serialized
    assert '"events"' not in serialized
    for event in request["events"]:
        assert event["resource_id"] not in serialized
        assert event["created_at"] not in serialized


# --- Pagination ----------------------------------------------------------------


def test_pagination_concatenates_without_gaps_or_duplicates(client):
    receipts = _import_many(client, 5)
    all_items, pages, count = _walk_pages(client, limit=2)
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    ids = [i["id"] for i in all_items]
    assert len(ids) == len(set(ids)) == 5
    assert ids == [r["id"] for r in receipts]


def test_last_page_cursor_null_on_exact_division(client):
    _import_many(client, 4)
    first = _list(client, limit=2).json()
    assert first["next_cursor"] is not None
    second = _list(client, limit=2, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 2
    assert second["count"] == 4
    assert second["next_cursor"] is None


def test_default_limit_is_fifty(client):
    _import_many(client, 55)
    first = _list(client).json()
    assert len(first["items"]) == 50
    assert first["count"] == 55
    assert first["next_cursor"] is not None
    second = _list(client, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 5
    assert second["count"] == 55
    assert second["next_cursor"] is None


def test_limit_boundaries_accepted(client):
    _import_many(client, 2)
    for value in (1, 100):
        assert _list(client, limit=value).status_code == 200


def test_reusing_a_cursor_replays_the_same_page(client):
    _import_many(client, 5)
    cursor = _list(client, limit=2).json()["next_cursor"]
    replay_one = _list(client, limit=2, cursor=cursor).json()
    replay_two = _list(client, limit=2, cursor=cursor).json()
    assert replay_one == replay_two


def test_cursor_past_end_returns_empty_page_with_total_count(client, app):
    _import_many(client, 5)
    token = pagination.encode_typed_cursor(
        app.state.checkpoint_import_reconciliations_cursor_secret,
        pagination.CHECKPOINT_IMPORT_RECONCILIATIONS_CURSOR,
        {"limit": 50, "offset": 99},
    )
    body = _list(client, cursor=token).json()
    assert body == {"items": [], "count": 5, "next_cursor": None}


# --- Cursor integrity ------------------------------------------------------------


def test_tampered_or_malformed_cursors_are_validation_errors(client):
    _import_many(client, 3)
    good = _list(client, limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    foreign = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.CHECKPOINT_IMPORT_RECONCILIATIONS_CURSOR,
        {"limit": 1, "offset": 1},
    )
    for token in (
        "",
        "   ",
        "not-a-cursor",
        "cr1.onlytwoparts",
        "cr1.too.many.parts",
        "cr0.x.y",
        "cr2.x.y",
        "v1.x.y",
        "ce1.x.y",
        "ae1.x.y",
        "ei1.x.y",
        "ir1.x.y",
        "ci1.x.y",
        tampered,
        foreign,
    ):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422, repr(token)
        assert resp.json()["error"]["code"] == "validation_error"
        assert "items" not in resp.json()


def test_cursor_signed_with_old_format_marker_is_rejected(client, app):
    _import_many(client, 2)
    payload = base64.urlsafe_b64encode(
        json.dumps({"limit": 50, "offset": 1}).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(
        hmac_mod.new(
            app.state.checkpoint_import_reconciliations_cursor_secret,
            f"cr0.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    resp = _list(client, cursor=f"cr0.{payload}.{sig}")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_from_other_families_is_rejected(client):
    _import_many(client, 3)
    # Real cursors minted by sibling endpoints (same process) must never
    # resume this collection.
    checkpoint_imports_cursor = client.get(
        IMPORTS_URL, params={"limit": 1}
    ).json()["next_cursor"]
    audit_cursor = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.AUDIT_EVENTS_CURSOR,
        {
            "event_type": None,
            "resource_id": None,
            "from": None,
            "to": None,
            "limit": 1,
            "offset": 1,
        },
    )
    exchange_cursor = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.EXCHANGE_IMPORTS_CURSOR,
        {
            "manifest_version": None,
            "evidence_bundle_id": None,
            "manifest_digest_hex": None,
            "limit": 1,
            "offset": 1,
        },
    )
    exchange_recon_cursor = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.EXCHANGE_IMPORT_RECONCILIATIONS_CURSOR,
        {"limit": 1, "offset": 1},
    )
    for token in (
        checkpoint_imports_cursor,
        audit_cursor,
        exchange_cursor,
        exchange_recon_cursor,
    ):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


def test_foreign_family_cursor_signed_with_this_secret_is_rejected(client, app):
    # Family separation is by format marker, not by secret: a checkpoint-
    # imports cursor minted under this endpoint's own secret is rejected.
    _import_many(client, 2)
    foreign = pagination.encode_typed_cursor(
        app.state.checkpoint_import_reconciliations_cursor_secret,
        pagination.CHECKPOINT_IMPORTS_CURSOR,
        {
            "checkpoint_version": None,
            "events_digest_hex": None,
            "event_count": None,
            "limit": 1,
            "offset": 1,
        },
    )
    resp = _list(client, limit=1, cursor=foreign)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_reconciliations_cursor_is_rejected_by_other_endpoints(client):
    _import_many(client, 2)
    cursor = _list(client, limit=1).json()["next_cursor"]

    resp = client.get(IMPORTS_URL, params={"limit": 1, "cursor": cursor})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    resp = client.get(
        "/v1/audit-events", params={"limit": 1, "cursor": cursor}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_bound_to_limit(client):
    _import_many(client, 5)
    cursor = _list(client, limit=2).json()["next_cursor"]
    resp = _list(client, limit=3, cursor=cursor)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    # Dropping the explicit limit is also a mismatch: the cursor binds the
    # effective limit, not its spelling.
    assert _list(client, cursor=cursor).status_code == 422
    # Replaying with the bound limit still works.
    assert _list(client, limit=2, cursor=cursor).status_code == 200


def test_cursor_secret_rotation_invalidates_outstanding_cursors(
    client, tmp_db_url
):
    _import(client, "restart")
    cursor = _list(client, limit=1).json()["next_cursor"]
    client.app.state.checkpoint_import_reconciliations_cursor_secret = (
        secrets.token_bytes(32)
    )
    assert _list(client, limit=1, cursor=cursor).status_code == 422

    first_app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(first_app) as first_client:
        receipt = _import(first_client, "persisted")
        token = _list(first_client, limit=1).json()["next_cursor"]
    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as second_client:
        stale = _list(second_client, limit=1, cursor=token)
        assert stale.status_code == 422
        # The receipts themselves survive the restart and still list.
        fresh = _list(second_client).json()
        assert [i["id"] for i in fresh["items"]] == [receipt["id"]]


# --- Parameter validation --------------------------------------------------------


def test_illegal_limit_values_are_validation_errors(client):
    _import_many(client, 2)
    for value in ("0", "101", "-1", "1.5", "abc", "8.0", "  2", ""):
        resp = _list(client, limit=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_validation_errors(client):
    _import_many(client, 2)
    for suffix in ("limit=1&limit=2", "cursor=x&cursor=y"):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameters_are_validation_errors(client):
    _import_many(client, 2)
    for suffix in (
        "import_id=aci_x",
        "checkpoint_version=" + CHECKPOINT_VERSION,
        "events_digest_hex=" + "0" * 64,
        "event_count=2",
        "matches=true",
        "local_checkpoint=x",
        "limit=1&offset=2",
        "CURSOR=x",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_unknown_parameter_is_422_on_empty_database(client):
    # Parameters are validated before any read: a typo is 422 with no rows.
    resp = _list(client, unknown=1)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_cursor_with_otherwise_valid_params_is_422(client):
    _import_many(client, 2)
    resp = _list(client, limit=10, cursor="garbage")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only guarantee ----------------------------------------------------------


def test_reads_and_failures_write_no_rows_or_audit_events(client, db_session):
    _import_many(client, 3)

    def counts():
        return (
            db_session.scalar(
                select(func.count()).select_from(CheckpointImportRecord)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    receipts_before, events_before = counts()
    assert receipts_before == 3
    assert events_before == 3

    # Successful reads, including a full page walk.
    cursor = None
    for _ in range(10):
        params = {"limit": 1}
        if cursor is not None:
            params["cursor"] = cursor
        resp = _list(client, **params)
        assert resp.status_code == 200, resp.text
        cursor = resp.json()["next_cursor"]
        if cursor is None:
            break
    _list(client)
    _list(client, limit=100)

    # Failed reads must not write anything either.
    _list(client, limit=0)
    _list(client, limit=101)
    _list(client, cursor="tampered")
    _list(client, limit=1, cursor="ci1.x.y")
    client.get(f"{URL}?limit=1&limit=2")
    client.get(f"{URL}?unknown=1")
    _list(client, event_count=2)

    db_session.expire_all()
    assert counts() == (receipts_before, events_before)


# --- Determinism and compatibility ------------------------------------------------


def test_listing_is_deterministic_across_restart(file_client, tmp_db_url):
    # Direct, event-free fixture state at a fixed instant so the verdict and
    # ordering survive identically across a process restart.
    session = file_client.app.state.session_factory()
    try:
        matching = _insert_receipt(
            session, CHECKPOINT_VERSION, 0, EMPTY_DIGEST
        )
        non_matching = _insert_receipt(
            session, "provenance-audit-checkpoint-v2", 0, EMPTY_DIGEST
        )
    finally:
        session.close()

    expected = _list(file_client)
    assert expected.status_code == 200, expected.text

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        resp = _list(client)
        assert resp.status_code == 200, resp.text
        assert resp.json() == expected.json()
        items = {item["id"]: item for item in resp.json()["items"]}
        assert items[matching]["matches"] is True
        assert items[non_matching]["matches"] is False


def test_existing_single_reconciliation_checkpoint_and_list_unchanged(client):
    receipt = _import(client, "compat")

    # The single reconciliation route keeps its three-member body.
    single = client.get(f"{IMPORTS_URL}/{receipt['id']}/reconciliation")
    assert single.status_code == 200
    single_body = single.json()
    assert set(single_body) == {"import_id", "local_checkpoint", "matches"}
    assert single_body["import_id"] == receipt["id"]

    # The unfiltered checkpoint route is unchanged.
    checkpoint = client.get(CHECKPOINT_URL)
    assert checkpoint.status_code == 200
    assert single_body["local_checkpoint"] == checkpoint.json()

    # The checkpoint-imports receipt search and single read are unchanged.
    imports_list = client.get(IMPORTS_URL)
    assert imports_list.status_code == 200
    assert imports_list.json()["items"] == [receipt]
    assert client.get(f"{IMPORTS_URL}/{receipt['id']}").json() == receipt

    missing = client.get(f"{IMPORTS_URL}/aci_{'0' * 64}")
    assert missing.status_code == 404
    assert (
        missing.json()["error"]["code"]
        == "audit_checkpoint_import_not_found"
    )
