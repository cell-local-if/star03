"""Tests for the read-only exchange-import receipt search endpoint.

Covers GET /v1/evidence-bundle-exchange-imports: the response shape carries
exactly {"items", "count", "next_cursor"} and each item exposes only the
single-receipt public fields (id, manifest_version, evidence_bundle_id,
manifest_digest_hex, received_at in UTC), receipts follow the stable
creation order, manifest_version/evidence_bundle_id/manifest_digest_hex are
non-empty exact, case- and whitespace-sensitive combinable filters, limit
is 1..100 defaulting to 50, opaque HMAC cursors bind every effective filter
and the limit so pages concatenate without gaps or duplicates and the
final cursor is null, blank/illegal/repeated/undeclared parameters and
blank/tampered/malformed/foreign-family/mismatching cursors are 422
validation_error, no match is an empty collection, and neither queries nor
failures write any resource or audit rows. All fixtures are deterministic
and offline.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta

from sqlalchemy import func, select

from provenance import pagination
from provenance.models import AuditEvent, ExchangeImportRecord
from tests.test_evidence_bundle_exchange_imports import _import_request, URL
from tests.test_exchange_manifest_verifications import (
    MANIFEST_VERSION,
    _offline_snapshot,
)

LIST_URL = URL
RECEIPT_KEYS = {
    "id",
    "manifest_version",
    "evidence_bundle_id",
    "manifest_digest_hex",
    "received_at",
}
OTHER_BUNDLE_ID = "evb_" + hashlib.sha256(b"other-bundle").hexdigest()


def _snapshot(*, other_bundle: bool = False, title: str | None = None) -> dict:
    """A self-consistent offline snapshot, optionally for a second bundle."""
    snapshot = _offline_snapshot()
    if other_bundle:
        snapshot["evidence_bundle"]["id"] = OTHER_BUNDLE_ID
        snapshot["attestations"][0]["target_id"] = OTHER_BUNDLE_ID
    if title is not None:
        snapshot["content"]["title"] = title
    return snapshot


def _import(client, snapshot: dict) -> dict:
    resp = client.post(LIST_URL, json=_import_request(snapshot))
    assert resp.status_code == 201, resp.text
    return resp.json()


def _list(client, **params):
    return client.get(LIST_URL, params=params)


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


def _setup_receipts(client) -> tuple[list[dict], list[dict]]:
    """Register three receipts for the offline bundle and two for another."""
    bundle_a = [
        _import(client, _snapshot(title=f"offline title {i}"))
        for i in range(3)
    ]
    bundle_b = [
        _import(
            client,
            _snapshot(other_bundle=True, title=f"other title {i}"),
        )
        for i in range(2)
    ]
    return bundle_a, bundle_b


# --- Response shape, fields, and ordering -------------------------------------


def test_empty_database_is_an_empty_collection(client):
    assert _list(client).json() == {
        "items": [],
        "count": 0,
        "next_cursor": None,
    }


def test_response_shape_and_item_fields(client):
    _setup_receipts(client)
    resp = _list(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"items", "count", "next_cursor"}
    assert body["count"] == 5
    assert body["next_cursor"] is None
    assert len(body["items"]) == 5
    for item in body["items"]:
        assert set(item) == RECEIPT_KEYS


def test_listed_items_equal_the_single_receipt_views(client):
    bundle_a, bundle_b = _setup_receipts(client)
    items = _list(client).json()["items"]
    for expected, item in zip(bundle_a + bundle_b, items):
        assert item == expected
        assert item == client.get(f"{LIST_URL}/{item['id']}").json()
    # The snapshot is never part of any listed item.
    assert all("snapshot" not in item and "manifest" not in item for item in items)


def test_received_at_preserves_utc_timezone(client):
    _setup_receipts(client)
    for item in _list(client).json()["items"]:
        parsed = datetime.fromisoformat(item["received_at"])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)


def test_receipts_follow_stable_creation_order(client):
    bundle_a, bundle_b = _setup_receipts(client)
    expected_ids = [r["id"] for r in bundle_a + bundle_b]
    assert [i["id"] for i in _list(client).json()["items"]] == expected_ids


def test_idempotent_retry_does_not_change_the_listing(client):
    first = _import(client, _snapshot())
    assert _list(client).json()["count"] == 1
    # Re-posting the same package is a 200 retry and adds no list entry.
    retry = client.post(LIST_URL, json=_import_request(_snapshot()))
    assert retry.status_code == 200
    body = _list(client).json()
    assert body["count"] == 1
    assert body["items"] == [first]


# --- Exact-match filtering -----------------------------------------------------


def test_filter_by_evidence_bundle_id(client):
    bundle_a, bundle_b = _setup_receipts(client)
    body = _list(
        client, evidence_bundle_id=bundle_a[0]["evidence_bundle_id"]
    ).json()
    assert body["count"] == 3
    assert [i["id"] for i in body["items"]] == [r["id"] for r in bundle_a]

    body = _list(client, evidence_bundle_id=OTHER_BUNDLE_ID).json()
    assert body["count"] == 2
    assert [i["id"] for i in body["items"]] == [r["id"] for r in bundle_b]


def test_filter_by_manifest_digest_hex(client):
    bundle_a, _ = _setup_receipts(client)
    target = bundle_a[1]
    body = _list(
        client, manifest_digest_hex=target["manifest_digest_hex"]
    ).json()
    assert body["count"] == 1
    assert body["items"][0] == target


def test_filter_by_manifest_version(client):
    _setup_receipts(client)
    body = _list(client, manifest_version=MANIFEST_VERSION).json()
    assert body["count"] == 5

    # Every receipt commits to the single fixed version; any other literal
    # is a valid (non-empty) filter that simply matches nothing.
    body = _list(
        client, manifest_version="provenance-exchange-manifest-v2"
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


def test_filters_combine_as_logical_and(client):
    bundle_a, bundle_b = _setup_receipts(client)
    target = bundle_b[0]
    body = _list(
        client,
        evidence_bundle_id=target["evidence_bundle_id"],
        manifest_digest_hex=target["manifest_digest_hex"],
        manifest_version=MANIFEST_VERSION,
    ).json()
    assert body["count"] == 1
    assert body["items"][0] == target

    # A digest from bundle A combined with bundle B's id matches nothing.
    body = _list(
        client,
        evidence_bundle_id=target["evidence_bundle_id"],
        manifest_digest_hex=bundle_a[0]["manifest_digest_hex"],
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


def test_exact_match_is_case_and_whitespace_sensitive(client):
    bundle_a, _ = _setup_receipts(client)
    bid = bundle_a[0]["evidence_bundle_id"]
    digest = bundle_a[0]["manifest_digest_hex"]
    for params in (
        {"evidence_bundle_id": bid.upper()},
        {"evidence_bundle_id": bid + " "},
        {"evidence_bundle_id": " " + bid},
        {"manifest_digest_hex": digest.upper()},
        {"manifest_digest_hex": digest + " "},
        {"manifest_version": MANIFEST_VERSION.upper()},
        {"manifest_version": MANIFEST_VERSION + " "},
    ):
        body = _list(client, **params).json()
        assert body == {"items": [], "count": 0, "next_cursor": None}, params


def test_no_matching_receipts_is_empty_collection(client):
    _setup_receipts(client)
    body = _list(client, evidence_bundle_id="evb_" + "0" * 64).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


# --- Pagination -----------------------------------------------------------------


def test_pagination_concatenates_without_gaps_or_duplicates(client):
    _setup_receipts(client)
    all_items, pages, count = _walk_pages(client, limit=2)
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    ids = [item["id"] for item in all_items]
    assert len(ids) == len(set(ids)) == 5
    assert all_items == _list(client).json()["items"]


def test_count_is_filtered_total_on_every_page(client):
    _setup_receipts(client)
    all_items, pages, count = _walk_pages(
        client, evidence_bundle_id=OTHER_BUNDLE_ID, limit=1
    )
    assert count == 2
    assert [len(page) for page in pages] == [1, 1]
    assert {i["evidence_bundle_id"] for i in all_items} == {OTHER_BUNDLE_ID}


def test_last_page_cursor_null_on_exact_division(client):
    bundle_a, _ = _setup_receipts(client)
    # The offline bundle has exactly three receipts; a page of three is the
    # full filtered set, so the cursor is already null (no trailing empty page).
    first = _list(
        client,
        evidence_bundle_id=bundle_a[0]["evidence_bundle_id"],
        limit=3,
    ).json()
    assert first["count"] == 3
    assert len(first["items"]) == 3
    assert first["next_cursor"] is None

    # Walking the same set a page at a time ends on a full final page with a
    # null cursor rather than an empty remainder page.
    all_items, pages, count = _walk_pages(
        client, evidence_bundle_id=bundle_a[0]["evidence_bundle_id"], limit=1
    )
    assert count == 3
    assert [len(page) for page in pages] == [1, 1, 1]
    assert len(all_items) == 3


def test_default_limit_is_fifty(client):
    for i in range(55):
        _import(client, _snapshot(title=f"bulk title {i}"))
    first = _list(client).json()
    assert len(first["items"]) == 50
    assert first["count"] == 55
    assert first["next_cursor"] is not None
    second = _list(client, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 5
    assert second["count"] == 55
    assert second["next_cursor"] is None


def test_limit_boundaries_accepted(client):
    _setup_receipts(client)
    for value in (1, 100):
        assert _list(client, limit=value).status_code == 200


def test_reusing_a_cursor_replays_the_same_page(client):
    _setup_receipts(client)
    cursor = _list(client, limit=2).json()["next_cursor"]
    replay_one = _list(client, limit=2, cursor=cursor).json()
    replay_two = _list(client, limit=2, cursor=cursor).json()
    assert replay_one == replay_two


def test_cursor_past_end_returns_empty_page_with_total_count(client, app):
    _setup_receipts(client)
    token = pagination.encode_typed_cursor(
        app.state.exchange_imports_cursor_secret,
        pagination.EXCHANGE_IMPORTS_CURSOR,
        {
            "manifest_version": None,
            "evidence_bundle_id": None,
            "manifest_digest_hex": None,
            "limit": 50,
            "offset": 99,
        },
    )
    body = _list(client, cursor=token).json()
    assert body["items"] == []
    assert body["count"] == 5
    assert body["next_cursor"] is None


# --- Cursor integrity ------------------------------------------------------------


def test_tampered_or_malformed_cursors_are_validation_errors(client, app):
    _setup_receipts(client)
    good = _list(client, limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    foreign_secret = pagination.encode_typed_cursor(
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
    for token in (
        "",
        "   ",
        "not-a-cursor",
        "ei1.onlytwoparts",
        "ei1.too.many.parts",
        "ei0.x.y",
        "ei2.x.y",
        "v1.x.y",
        "ce1.x.y",
        "ae1.x.y",
        tampered,
        foreign_secret,
    ):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422, repr(token)
        assert resp.json()["error"]["code"] == "validation_error"
        assert "items" not in resp.json()


def test_cursors_from_other_families_are_rejected(client, app):
    _setup_receipts(client)
    lineage_cursor = pagination.encode_cursor(
        app.state.lineage_cursor_secret,
        {
            "content_id": "cnt_x",
            "direction": "ancestors",
            "max_depth": 8,
            "min_depth": 1,
            "relation_type": None,
            "limit": 1,
            "offset": 1,
        },
    )
    evidence_cursor = pagination.encode_typed_cursor(
        app.state.content_evidence_cursor_secret,
        pagination.CONTENT_EVIDENCE_CURSOR,
        {
            "content_id": "cnt_x",
            "evidence_type": None,
            "media_type": None,
            "limit": 1,
            "offset": 1,
        },
    )
    audit_cursor = pagination.encode_typed_cursor(
        app.state.audit_events_cursor_secret,
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
    for token in (lineage_cursor, evidence_cursor, audit_cursor):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


def test_exchange_imports_cursor_is_rejected_by_other_endpoints(client):
    _setup_receipts(client)
    cursor = _list(client, limit=1).json()["next_cursor"]
    # The collection cursor is family-namespaced: other paginated read
    # routes reject it rather than interpret it.
    resp = client.get("/v1/audit-events", params={"limit": 1, "cursor": cursor})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    resp = client.get(
        f"/v1/contents/cnt_{'0' * 64}/evidence-bundles",
        params={"limit": 1, "cursor": cursor},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_bound_to_every_effective_query_parameter(client):
    bundle_a, _ = _setup_receipts(client)
    cursor = _list(client, limit=2).json()["next_cursor"]

    # An unfiltered cursor resumed with any filter mismatches.
    for params in (
        {"limit": 3},
        {"manifest_version": MANIFEST_VERSION},
        {"evidence_bundle_id": OTHER_BUNDLE_ID},
        {"manifest_digest_hex": bundle_a[0]["manifest_digest_hex"]},
    ):
        resp = _list(client, cursor=cursor, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"

    # A filtered cursor resumed without (or with a changed) filter
    # mismatches -- pages never silently repeat or omit a result set.
    filtered = _list(
        client, evidence_bundle_id=OTHER_BUNDLE_ID, limit=1
    ).json()["next_cursor"]
    assert _list(client, limit=1, cursor=filtered).status_code == 422
    assert (
        _list(
            client,
            evidence_bundle_id=bundle_a[0]["evidence_bundle_id"],
            limit=1,
            cursor=filtered,
        ).status_code
        == 422
    )


# --- Parameter validation --------------------------------------------------------


def test_blank_filters_are_validation_errors(client):
    _setup_receipts(client)
    for field in (
        "manifest_version",
        "evidence_bundle_id",
        "manifest_digest_hex",
    ):
        for value in ("", "   ", "\t"):
            resp = _list(client, **{field: value})
            assert resp.status_code == 422, (field, value)
            assert resp.json()["error"]["code"] == "validation_error"


def test_illegal_limit_values_are_validation_errors(client):
    _setup_receipts(client)
    for value in ("0", "101", "-1", "1.5", "abc", "8.0", "  2", ""):
        resp = _list(client, limit=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_validation_errors(client):
    _setup_receipts(client)
    for suffix in (
        "manifest_version=a&manifest_version=b",
        "evidence_bundle_id=a&evidence_bundle_id=b",
        "manifest_digest_hex=a&manifest_digest_hex=b",
        "limit=1&limit=2",
        "cursor=x&cursor=y",
    ):
        resp = client.get(f"{LIST_URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameters_are_validation_errors(client):
    _setup_receipts(client)
    for suffix in (
        "id=eir_x",
        "received_at=2026-01-01T00:00:00Z",
        "evidence_type=raw_capture",
        "bundle_id=evb_x",
        "limit=1&offset=2",
        "CURSOR=x",
    ):
        resp = client.get(f"{LIST_URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_cursor_with_otherwise_valid_params_is_422(client):
    _setup_receipts(client)
    resp = _list(
        client,
        evidence_bundle_id=OTHER_BUNDLE_ID,
        limit=10,
        cursor="garbage",
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only guarantee ----------------------------------------------------------


def test_queries_and_failures_write_no_rows_or_audit_events(client, db_session):
    _setup_receipts(client)

    def counts():
        return (
            db_session.scalar(
                select(func.count()).select_from(ExchangeImportRecord)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    before = counts()

    # Successful filtered, empty, and paginated reads.
    cursor = None
    for _ in range(6):
        params = {"evidence_bundle_id": OTHER_BUNDLE_ID, "limit": 1}
        if cursor is not None:
            params["cursor"] = cursor
        resp = _list(client, **params)
        assert resp.status_code == 200, resp.text
        cursor = resp.json()["next_cursor"]
        if cursor is None:
            break
    _list(client)
    _list(client, evidence_bundle_id="evb_" + "f" * 64)
    _list(client, manifest_version="provenance-exchange-manifest-v9")

    # Failed reads must not write anything either.
    _list(client, evidence_bundle_id=" ")
    _list(client, limit=0)
    _list(client, limit=101)
    _list(client, cursor="tampered")
    _list(client, manifest_version=" ")
    client.get(f"{LIST_URL}?limit=1&limit=2")
    client.get(f"{LIST_URL}?unknown=1")

    db_session.expire_all()
    assert counts() == before
