"""Tests for the read-only checkpoint-import reconciliation listing endpoint.

Covers GET /v1/audit-events/checkpoint-import-reconciliations: the response
shape carries exactly ``{"items", "count", "next_cursor"}``. Each item is the
existing single-receipt public view (``id``/``checkpoint_version``/
``events_digest_hex``/``event_count``/``received_at``) plus
``local_checkpoint`` and ``matches``. Receipts follow stable creation order
and ``count`` is the total number of receipts. ``local_checkpoint`` is the
four-field checkpoint computed over the complete, unfiltered local audit
sequence under the existing ``GET /v1/audit-events/checkpoint`` rules, and
``matches`` is true only when the receipt's ``checkpoint_version``,
``event_count``, and ``events_digest_hex`` all equal the local checkpoint
fields. ``limit`` is 1..100 defaulting to 50; the opaque HMAC cursor binds
the limit so pages concatenate without gaps or duplicates and the final
cursor is null. Any other, repeated, blank, or illegal parameter and any
malformed/tampered/foreign/mismatching cursor is 422 validation_error; no
receipts is an empty collection. The route is strictly read-only — no
resource, receipt, or audit row is written — and no imported event array is
echoed. All fixtures are deterministic and offline.
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
from provenance.models import (
    EVENT_CONTENT_CREATED,
    AuditEvent,
    CheckpointImportRecord,
)
from tests.test_audit_checkpoint_import_reconciliation import (
    CHECKPOINT_KEYS,
    CHECKPOINT_VERSION,
    EMPTY_DIGEST,
    IMPORTS_URL,
    _digest_of,
    _insert_event,
    _insert_receipt,
    _matching_receipt,
    _offline_request,
    _served_checkpoint,
)

URL = "/v1/audit-events/checkpoint-import-reconciliations"
CHECKPOINT_URL = "/v1/audit-events/checkpoint"
RECEIPT_KEYS = {
    "id",
    "checkpoint_version",
    "events_digest_hex",
    "event_count",
    "received_at",
}
ITEM_KEYS = RECEIPT_KEYS | {"local_checkpoint", "matches"}


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


def _bulk_receipts(db_session, count: int):
    """Persist ``count`` distinct receipts directly, in stable order.

    Direct insertion keeps the local audit sequence exactly controlled (no
    import audit events), and each receipt carries a distinct digest so its
    receiving identity is unique.
    """
    return [
        _insert_receipt(
            db_session,
            CHECKPOINT_VERSION,
            i,
            hashlib.sha256(f"bulk-{i:03d}".encode()).hexdigest(),
        )
        for i in range(count)
    ]


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
    _insert_receipt(db_session, CHECKPOINT_VERSION, 0, EMPTY_DIGEST)
    resp = _list(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"items", "count", "next_cursor"}
    assert body["count"] == 1
    assert body["next_cursor"] is None
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert set(item) == ITEM_KEYS
    assert set(item["local_checkpoint"]) == CHECKPOINT_KEYS
    assert item["id"].startswith("aci_")
    assert item["checkpoint_version"] == CHECKPOINT_VERSION


def test_item_carries_the_existing_receipt_public_view(client, db_session):
    receipt_id = _insert_receipt(db_session, CHECKPOINT_VERSION, 3, "ab" * 32)
    item = _list(client).json()["items"][0]
    # The receipt half of the item is exactly the existing single-receipt
    # view, including its original received_at.
    served = client.get(f"{IMPORTS_URL}/{receipt_id}").json()
    assert {key: item[key] for key in RECEIPT_KEYS} == served


def test_received_at_is_utc(client, db_session):
    _bulk_receipts(db_session, 3)
    for item in _list(client).json()["items"]:
        parsed = datetime.fromisoformat(item["received_at"])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)
        assert item["received_at"].endswith(("Z", "+00:00"))


def test_receipts_follow_stable_creation_order(client, db_session):
    receipt_ids = _bulk_receipts(db_session, 5)
    items = _list(client).json()["items"]
    assert [i["id"] for i in items] == receipt_ids


def test_count_is_total_receipt_count_on_every_page(client, db_session):
    _bulk_receipts(db_session, 5)
    all_items, pages, count = _walk_pages(client, limit=2)
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    assert len(all_items) == 5


# --- Per-item reconciliation semantics -----------------------------------------


def test_empty_trail_receipt_matches_the_empty_checkpoint(client, db_session):
    _insert_receipt(db_session, CHECKPOINT_VERSION, 0, EMPTY_DIGEST)
    item = _list(client).json()["items"][0]
    assert item["local_checkpoint"] == {
        "checkpoint_version": CHECKPOINT_VERSION,
        "digest_algorithm": "sha256",
        "event_count": 0,
        "events_digest_hex": EMPTY_DIGEST,
    }
    assert item["matches"] is True


def test_item_agrees_with_the_single_reconciliation_route(client, db_session):
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    matching_id, _ = _matching_receipt(db_session, client)
    stale_id = _insert_receipt(db_session, CHECKPOINT_VERSION, 99, "a" * 64)
    other_version_id = _insert_receipt(
        db_session, "provenance-audit-checkpoint-v2", 1, "b" * 64
    )

    items = {item["id"]: item for item in _list(client).json()["items"]}
    assert set(items) == {matching_id, stale_id, other_version_id}
    for receipt_id, item in items.items():
        single = _single_reconciliation(client, receipt_id)
        assert item["local_checkpoint"] == single["local_checkpoint"]
        assert item["matches"] == single["matches"]

    assert items[matching_id]["matches"] is True
    assert items[stale_id]["matches"] is False
    assert items[other_version_id]["matches"] is False


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


def test_local_checkpoint_is_unfiltered_when_filters_would_exclude_events(
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
        EVENT_CONTENT_CREATED,
        "cnt_x",
        datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    receipt_id, checkpoint = _matching_receipt(db_session, client)
    assert checkpoint["event_count"] == 2

    filtered = client.get(CHECKPOINT_URL, params={"event_type": "actor.created"})
    assert filtered.json()["event_count"] == 1

    item = _list(client).json()["items"][0]
    assert item["id"] == receipt_id
    assert item["local_checkpoint"]["event_count"] == 2
    assert item["local_checkpoint"] == checkpoint
    assert item["matches"] is True


def test_local_checkpoint_reflects_current_state_on_each_read(
    client, db_session
):
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    receipt_id, checkpoint = _matching_receipt(db_session, client)
    assert _list(client).json()["items"][0]["matches"] is True

    # The local sequence changes after registration: the receipt no longer
    # matches, but the current checkpoint is still reported in full.
    _insert_event(
        db_session,
        EVENT_CONTENT_CREATED,
        "cnt_later",
        datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    current = _served_checkpoint(client)
    assert current["event_count"] == checkpoint["event_count"] + 1

    item = _list(client).json()["items"][0]
    assert item["id"] == receipt_id
    assert item["local_checkpoint"] == current
    assert item["matches"] is False


def test_response_carries_no_imported_events(client, db_session):
    marker = "reconciliations-no-echo-marker-7b21"
    request = _offline_request()
    request["events"][0]["resource_id"] = marker
    request["checkpoint"]["events_digest_hex"] = _digest_of(request["events"])
    resp = client.post(IMPORTS_URL, json=request)
    assert resp.status_code == 201, resp.text

    body = _list(client).json()
    assert len(body["items"]) == 1
    assert set(body["items"][0]) == ITEM_KEYS
    serialized = json.dumps(body, ensure_ascii=False)
    # The imported event array and its values are absent in every form.
    assert '"events"' not in serialized
    assert marker not in serialized
    for event in request["events"]:
        assert event["resource_id"] not in serialized
        assert event["created_at"] not in serialized


# --- Pagination ----------------------------------------------------------------


def test_pagination_concatenates_without_gaps_or_duplicates(client, db_session):
    receipt_ids = _bulk_receipts(db_session, 5)
    all_items, pages, count = _walk_pages(client, limit=2)
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    ids = [i["id"] for i in all_items]
    assert len(ids) == len(set(ids)) == 5
    assert ids == receipt_ids


def test_last_page_cursor_null_on_exact_division(client, db_session):
    _bulk_receipts(db_session, 4)
    first = _list(client, limit=2).json()
    assert first["next_cursor"] is not None
    second = _list(client, limit=2, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 2
    assert second["count"] == 4
    assert second["next_cursor"] is None


def test_default_limit_is_fifty(client, db_session):
    _bulk_receipts(db_session, 55)
    first = _list(client).json()
    assert len(first["items"]) == 50
    assert first["count"] == 55
    assert first["next_cursor"] is not None
    second = _list(client, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 5
    assert second["count"] == 55
    assert second["next_cursor"] is None


def test_limit_boundaries_accepted(client, db_session):
    _bulk_receipts(db_session, 2)
    for value in (1, 100):
        assert _list(client, limit=value).status_code == 200


def test_reusing_a_cursor_replays_the_same_page(client, db_session):
    _bulk_receipts(db_session, 5)
    cursor = _list(client, limit=2).json()["next_cursor"]
    replay_one = _list(client, limit=2, cursor=cursor).json()
    replay_two = _list(client, limit=2, cursor=cursor).json()
    assert replay_one == replay_two


def test_cursor_past_end_returns_empty_page_with_total_count(client, app, db_session):
    _bulk_receipts(db_session, 5)
    token = pagination.encode_typed_cursor(
        app.state.checkpoint_import_reconciliations_cursor_secret,
        pagination.CHECKPOINT_IMPORT_RECONCILIATIONS_CURSOR,
        {"limit": 50, "offset": 99},
    )
    body = _list(client, cursor=token).json()
    assert body == {"items": [], "count": 5, "next_cursor": None}


def test_every_page_item_is_reconciled(client, db_session):
    # Reconciliation happens per item on every page, not just the first page.
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    matching_id, _ = _matching_receipt(db_session, client)
    _bulk_receipts(db_session, 3)
    all_items, _, _ = _walk_pages(client, limit=2)
    assert len(all_items) == 4
    matching = [item for item in all_items if item["matches"]]
    assert [item["id"] for item in matching] == [matching_id]


# --- Cursor integrity ------------------------------------------------------------


def test_tampered_or_malformed_cursors_are_validation_errors(client, db_session):
    _bulk_receipts(db_session, 3)
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


def test_cursor_signed_with_old_format_marker_is_rejected(client, app, db_session):
    _bulk_receipts(db_session, 2)
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


def test_cursor_from_other_families_is_rejected(client, db_session):
    _bulk_receipts(db_session, 3)
    # A real checkpoint-imports cursor is signed by the same process but uses
    # the ci1 family; it must never resume this collection.
    imports_cursor = client.get(
        IMPORTS_URL, params={"limit": 1}
    ).json()["next_cursor"]
    assert imports_cursor is not None
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
    for token in (imports_cursor, audit_cursor):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


def test_reconciliations_cursor_is_rejected_by_other_endpoints(client, db_session):
    _bulk_receipts(db_session, 2)
    cursor = _list(client, limit=1).json()["next_cursor"]

    resp = client.get(IMPORTS_URL, params={"limit": 1, "cursor": cursor})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    resp = client.get(
        "/v1/audit-events", params={"limit": 1, "cursor": cursor}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_bound_to_limit(client, db_session):
    _bulk_receipts(db_session, 5)
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
    client, tmp_db_url, db_session
):
    _bulk_receipts(db_session, 2)
    cursor = _list(client, limit=1).json()["next_cursor"]
    client.app.state.checkpoint_import_reconciliations_cursor_secret = (
        secrets.token_bytes(32)
    )
    assert _list(client, limit=1, cursor=cursor).status_code == 422

    first_app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(first_app) as first_client:
        session = first_app.state.session_factory()
        try:
            receipt_ids = _bulk_receipts(session, 2)
        finally:
            session.close()
        token = _list(first_client, limit=1).json()["next_cursor"]
    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as second_client:
        stale = _list(second_client, limit=1, cursor=token)
        assert stale.status_code == 422
        # The receipts themselves survive the restart and still list.
        fresh = _list(second_client).json()
        assert [i["id"] for i in fresh["items"]] == receipt_ids


# --- Parameter validation --------------------------------------------------------


def test_illegal_limit_values_are_validation_errors(client, db_session):
    _bulk_receipts(db_session, 2)
    for value in ("0", "101", "-1", "1.5", "abc", "8.0", "  2", ""):
        resp = _list(client, limit=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_validation_errors(client, db_session):
    _bulk_receipts(db_session, 2)
    for suffix in ("limit=1&limit=2", "cursor=x&cursor=y"):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameters_are_validation_errors(client, db_session):
    _bulk_receipts(db_session, 2)
    for suffix in (
        "import_id=aci_x",
        "checkpoint_version=" + CHECKPOINT_VERSION,
        "events_digest_hex=" + "0" * 64,
        "event_count=1",
        "matches=true",
        "offset=2",
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


def test_invalid_cursor_with_otherwise_valid_params_is_422(client, db_session):
    _bulk_receipts(db_session, 2)
    resp = _list(client, limit=10, cursor="garbage")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only guarantee ----------------------------------------------------------


def test_reads_and_failures_write_no_rows_or_audit_events(client, db_session):
    _bulk_receipts(db_session, 3)
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    _matching_receipt(db_session, client)

    def counts():
        return (
            db_session.scalar(
                select(func.count()).select_from(CheckpointImportRecord)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    receipts_before, events_before = counts()
    assert receipts_before == 4

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
    _list(client, event_count=1)

    db_session.expire_all()
    assert counts() == (receipts_before, events_before)


# --- Determinism and compatibility ------------------------------------------------


def test_listing_is_deterministic_across_restart(file_client, tmp_db_url):
    session = file_client.app.state.session_factory()
    try:
        _bulk_receipts(session, 3)
        _insert_event(
            session,
            "actor.created",
            "org-1",
            datetime(2026, 1, 1, tzinfo=timezone.utc),
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


def test_existing_single_reconciliation_and_imports_list_remain_unchanged(
    client, db_session
):
    receipt_id = _insert_receipt(db_session, CHECKPOINT_VERSION, 0, EMPTY_DIGEST)
    single = client.get(f"{IMPORTS_URL}/{receipt_id}/reconciliation")
    assert single.status_code == 200
    assert single.json() == {
        "import_id": receipt_id,
        "local_checkpoint": {
            "checkpoint_version": CHECKPOINT_VERSION,
            "digest_algorithm": "sha256",
            "event_count": 0,
            "events_digest_hex": EMPTY_DIGEST,
        },
        "matches": True,
    }

    imports_list = client.get(IMPORTS_URL)
    assert imports_list.status_code == 200
    assert [item["id"] for item in imports_list.json()["items"]] == [receipt_id]

    # The single receipt read and its 404 contract are untouched.
    served = client.get(f"{IMPORTS_URL}/{receipt_id}")
    assert served.status_code == 200
    assert served.json()["id"] == receipt_id
    missing = client.get(f"{IMPORTS_URL}/aci_{'0' * 64}")
    assert missing.status_code == 404
    assert (
        missing.json()["error"]["code"] == "audit_checkpoint_import_not_found"
    )

    # The unfiltered checkpoint route is untouched.
    assert client.get(CHECKPOINT_URL).json() == single.json()["local_checkpoint"]
