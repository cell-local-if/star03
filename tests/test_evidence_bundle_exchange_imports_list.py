"""Tests for the read-only exchange-import receipt search endpoint.

Covers GET /v1/evidence-bundle-exchange-imports: the response shape carries
exactly ``{"items", "count", "next_cursor"}`` with each item exposing only
the existing single-receipt fields ``id``/``manifest_version``/
``evidence_bundle_id``/``manifest_digest_hex``/``received_at`` (timezone-
aware UTC), receipts follow stable creation order, and the three
receiving-identity fields are non-empty, case- and whitespace-sensitive
exact matches that combine as logical AND (absent means unfiltered).
``limit`` is 1..100 defaulting to 50; the opaque HMAC cursor binds every
effective filter and the limit so pages concatenate without gaps or
duplicates and the final cursor is null. Blank/illegal/repeated/undeclared
parameters and blank/malformed/tampered/foreign-family/query-mismatching
cursors are 422 validation_error; no match is an empty collection. Neither
queries nor failures write any receipt or audit row. All fixtures are
deterministic and offline.
"""

from __future__ import annotations

import base64
import hashlib
import hmac as hmac_mod
import json
import secrets
from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance import pagination
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import AuditEvent, ExchangeImportRecord
from tests.test_evidence_bundle_exchange import _setup_bundle
from tests.test_evidence_bundle_exchange_imports import (
    RECEIPT_KEYS,
    URL,
    _import_request,
)
from tests.test_exchange_manifest_verifications import (
    MANIFEST_VERSION,
    _offline_snapshot,
)


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


def _import_with_title(client, title: str) -> dict:
    """Register one receipt for the default offline bundle under a title."""
    snapshot = _offline_snapshot()
    snapshot["content"]["title"] = title
    resp = client.post(URL, json=_import_request(snapshot))
    assert resp.status_code == 201, resp.text
    return resp.json()


def _alt_bundle_id(label: str) -> str:
    return "evb_" + hashlib.sha256(f"alt-bundle-{label}".encode()).hexdigest()


def _import_for_bundle(
    client, bundle_label: str, *, variation: str | None = None
) -> dict:
    """Register one receipt for a fabricated bundle id (offline, no state).

    ``bundle_label`` fixes the bundle id; ``variation`` changes the snapshot
    (hence the digest) so one bundle can carry several distinct receipts.
    """
    snapshot = _offline_snapshot()
    bundle_id = _alt_bundle_id(bundle_label)
    # Keep the snapshot internally consistent: the manifest takes the bundle
    # id from the snapshot, and every attestation must target that bundle.
    snapshot["evidence_bundle"]["id"] = bundle_id
    snapshot["attestations"][0]["target_id"] = bundle_id
    snapshot["content"]["title"] = (
        f"offline alt 快照 {variation or bundle_label}"
    )
    resp = client.post(URL, json=_import_request(snapshot))
    assert resp.status_code == 201, resp.text
    return resp.json()


def _setup_grouped(client):
    """Three receipts for the default bundle and two for a second bundle.

    Returns ``(default_bundle_id, alt_bundle_id, receipts_in_order)``.
    """
    receipts = [_import_with_title(client, f"title-{i}") for i in range(3)]
    default_bundle_id = receipts[0]["evidence_bundle_id"]

    alt_bundle_id = _alt_bundle_id("x")
    for variation in ("x", "y"):
        receipts.append(_import_for_bundle(client, "x", variation=variation))
    return default_bundle_id, alt_bundle_id, receipts


def _import_many(client, count: int):
    return [_import_with_title(client, f"bulk-{i:03d}") for i in range(count)]


# --- Response shape, fields, and ordering -------------------------------------


def test_empty_database_is_an_empty_collection(client):
    assert _list(client).json() == {
        "items": [],
        "count": 0,
        "next_cursor": None,
    }


def test_response_shape_and_item_fields(client):
    _setup_grouped(client)
    resp = _list(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"items", "count", "next_cursor"}
    assert body["count"] == 5
    assert body["next_cursor"] is None
    assert len(body["items"]) == 5
    for item in body["items"]:
        assert set(item) == RECEIPT_KEYS
        assert item["id"].startswith("eir_")
        assert item["manifest_version"] == MANIFEST_VERSION


def test_item_is_the_public_receipt_and_matches_the_single_read(client):
    created = _import_with_title(client, "shape")
    item = _list(client).json()["items"][0]
    assert item == created
    assert item == client.get(f"{URL}/{created['id']}").json()
    # The snapshot and every raw material stay absent from the list view.
    assert "snapshot" not in item
    assert "manifest" not in item


def test_received_at_is_utc(client):
    _setup_grouped(client)
    for item in _list(client).json()["items"]:
        parsed = datetime.fromisoformat(item["received_at"])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)
        assert item["received_at"].endswith(("Z", "+00:00"))


def test_receipts_follow_stable_creation_order(client):
    _, _, receipts = _setup_grouped(client)
    items = _list(client).json()["items"]
    assert [i["id"] for i in items] == [r["id"] for r in receipts]
    assert items == receipts


def test_listing_after_retry_keeps_the_original_position(client):
    receipt = _import_with_title(client, "once")
    # Re-presenting the same package is the idempotent 200 retry, not a new
    # receipt and not a new position in the listing.
    snapshot = _offline_snapshot()
    snapshot["content"]["title"] = "once"
    retry = client.post(URL, json=_import_request(snapshot))
    assert retry.status_code == 200
    later = _import_with_title(client, "later")
    items = _list(client).json()["items"]
    assert [i["id"] for i in items] == [receipt["id"], later["id"]]


# --- Exact-match filtering -----------------------------------------------------


def test_manifest_version_exact_match(client):
    default_bundle_id, _, _ = _setup_grouped(client)
    body = _list(client, manifest_version=MANIFEST_VERSION).json()
    assert body["count"] == 5
    assert {i["manifest_version"] for i in body["items"]} == {MANIFEST_VERSION}

    body = _list(
        client, manifest_version="provenance-exchange-manifest-v2"
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


def test_evidence_bundle_id_exact_match(client):
    _, alt_bundle_id, _ = _setup_grouped(client)
    body = _list(client, evidence_bundle_id=alt_bundle_id).json()
    assert body["count"] == 2
    assert {i["evidence_bundle_id"] for i in body["items"]} == {alt_bundle_id}


def test_manifest_digest_hex_exact_match(client):
    receipts = _import_many(client, 3)
    target = receipts[1]
    body = _list(
        client, manifest_digest_hex=target["manifest_digest_hex"]
    ).json()
    assert body["count"] == 1
    assert body["items"][0] == target


def test_filters_combine_as_logical_and(client):
    default_bundle_id, alt_bundle_id, receipts = _setup_grouped(client)
    target = receipts[0]
    body = _list(
        client,
        manifest_version=MANIFEST_VERSION,
        evidence_bundle_id=default_bundle_id,
        manifest_digest_hex=target["manifest_digest_hex"],
    ).json()
    assert [i["id"] for i in body["items"]] == [target["id"]]
    assert body["count"] == 1

    # A bundle id paired with another bundle's digest satisfies nothing:
    # the filters are ANDed, not coerced.
    body = _list(
        client,
        evidence_bundle_id=alt_bundle_id,
        manifest_digest_hex=target["manifest_digest_hex"],
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


def test_exact_match_is_case_and_whitespace_sensitive(client):
    default_bundle_id, _, receipts = _setup_grouped(client)
    digest = receipts[0]["manifest_digest_hex"]
    for params in (
        {"manifest_version": MANIFEST_VERSION.upper()},
        {"manifest_version": f" {MANIFEST_VERSION}"},
        {"evidence_bundle_id": default_bundle_id.swapcase()},
        {"evidence_bundle_id": f"{default_bundle_id} "},
        {"manifest_digest_hex": digest.upper()},
        {"manifest_digest_hex": f" {digest}"},
    ):
        body = _list(client, **params).json()
        assert body == {"items": [], "count": 0, "next_cursor": None}, params


def test_no_matching_receipts_is_empty_collection(client):
    _setup_grouped(client)
    body = _list(
        client, evidence_bundle_id="evb_" + "0" * 64
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


# --- Pagination -----------------------------------------------------------------


def test_pagination_concatenates_without_gaps_or_duplicates(client):
    _setup_grouped(client)
    all_items, pages, count = _walk_pages(client, limit=2)
    # 5 receipts -> pages of 2, 2, 1.
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    ids = [i["id"] for i in all_items]
    assert len(ids) == len(set(ids)) == 5
    assert all_items == _list(client).json()["items"]


def test_count_is_filtered_total_on_every_page(client):
    default_bundle_id, _, _ = _setup_grouped(client)
    all_items, pages, count = _walk_pages(
        client, evidence_bundle_id=default_bundle_id, limit=2
    )
    assert count == 3
    assert [len(page) for page in pages] == [2, 1]
    assert {i["evidence_bundle_id"] for i in all_items} == {default_bundle_id}


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
    _setup_grouped(client)
    cursor = _list(client, limit=2).json()["next_cursor"]
    replay_one = _list(client, limit=2, cursor=cursor).json()
    replay_two = _list(client, limit=2, cursor=cursor).json()
    assert replay_one == replay_two


def test_cursor_past_end_returns_empty_page_with_total_count(client, app):
    _setup_grouped(client)
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


def test_tampered_or_malformed_cursors_are_validation_errors(client):
    _setup_grouped(client)
    good = _list(client, limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    foreign = pagination.encode_typed_cursor(
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
        foreign,
    ):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422, repr(token)
        assert resp.json()["error"]["code"] == "validation_error"
        assert "items" not in resp.json()


def test_cursor_signed_with_old_format_marker_is_rejected(client, app):
    _setup_grouped(client)
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "manifest_version": None,
                "evidence_bundle_id": None,
                "manifest_digest_hex": None,
                "limit": 50,
                "offset": 1,
            }
        ).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(
        hmac_mod.new(
            app.state.exchange_imports_cursor_secret,
            f"ei0.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    resp = _list(client, cursor=f"ei0.{payload}.{sig}")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_from_other_families_is_rejected(client):
    _setup_grouped(client)
    lineage_cursor = pagination.encode_cursor(
        secrets.token_bytes(32),
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
        secrets.token_bytes(32),
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
    for token in (lineage_cursor, evidence_cursor, audit_cursor):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


def test_exchange_imports_cursor_is_rejected_by_other_endpoints(client):
    _setup_grouped(client)
    cursor = _list(client, limit=1).json()["next_cursor"]

    # The audit-events family uses its own marker and claim set.
    resp = client.get("/v1/audit-events", params={"limit": 1, "cursor": cursor})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    # As does the per-content evidence-bundle listing.
    _, _, bundle = _setup_bundle(client)
    content_id = client.get(
        f"/v1/evidence-bundles/{bundle['id']}/exchange"
    ).json()["content"]["id"]
    resp = client.get(
        f"/v1/contents/{content_id}/evidence-bundles",
        params={"limit": 1, "cursor": cursor},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_bound_to_every_effective_query_parameter(client):
    default_bundle_id, alt_bundle_id, receipts = _setup_grouped(client)
    digest = receipts[0]["manifest_digest_hex"]

    cursor = _list(client, limit=2).json()["next_cursor"]
    # A filter/limit present in the resume request but absent from the
    # cursor mismatches.
    for params in (
        {"limit": 3},
        {"manifest_version": MANIFEST_VERSION},
        {"evidence_bundle_id": default_bundle_id},
        {"manifest_digest_hex": digest},
    ):
        resp = _list(client, cursor=cursor, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"

    # A cursor minted under a filter cannot be resumed without it or with a
    # different value: no filter may be dropped or substituted.
    filtered = _list(
        client, evidence_bundle_id=default_bundle_id, limit=2
    ).json()["next_cursor"]
    assert _list(client, limit=2, cursor=filtered).status_code == 422
    assert (
        _list(
            client,
            evidence_bundle_id=alt_bundle_id,
            limit=2,
            cursor=filtered,
        ).status_code
        == 422
    )

    versioned = _list(
        client, manifest_version=MANIFEST_VERSION, limit=2
    ).json()["next_cursor"]
    assert _list(client, limit=2, cursor=versioned).status_code == 422

    digested = _list(
        client, manifest_digest_hex=digest, limit=2
    ).json()["next_cursor"]
    assert _list(client, limit=2, cursor=digested).status_code == 422


def test_cursor_secret_rotation_invalidates_outstanding_cursors(client, tmp_db_url):
    # A restart rotates the per-process HMAC secret, so a cursor minted by
    # the old process is rejected rather than trusted; the stored receipts
    # themselves survive.
    _import_with_title(client, "restart")
    cursor = _list(client, limit=1).json()["next_cursor"]
    client.app.state.exchange_imports_cursor_secret = secrets.token_bytes(32)
    assert _list(client, limit=1, cursor=cursor).status_code == 422

    first_app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(first_app) as first_client:
        receipt = _import_with_title(first_client, "persisted")
        token = _list(first_client, limit=1).json()["next_cursor"]
    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as second_client:
        stale = _list(second_client, limit=1, cursor=token)
        assert stale.status_code == 422
        # The receipts themselves survive the restart and still list.
        fresh = _list(second_client).json()
        assert [i["id"] for i in fresh["items"]] == [receipt["id"]]


# --- Parameter validation --------------------------------------------------------


def test_blank_filters_are_validation_errors(client):
    _setup_grouped(client)
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
    _setup_grouped(client)
    for value in ("0", "101", "-1", "1.5", "abc", "8.0", "  2", ""):
        resp = _list(client, limit=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_validation_errors(client):
    _setup_grouped(client)
    for suffix in (
        "manifest_version=a&manifest_version=b",
        "evidence_bundle_id=a&evidence_bundle_id=b",
        "manifest_digest_hex=a&manifest_digest_hex=b",
        "limit=1&limit=2",
        "cursor=x&cursor=y",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameters_are_validation_errors(client):
    _setup_grouped(client)
    for suffix in (
        "import_id=eir_x",
        "manifest_versions=provenance-exchange-manifest-v1",
        "evidence_bundle=evb_x",
        "digest_hex=" + "0" * 64,
        "limit=1&offset=2",
        "CURSOR=x",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_cursor_with_otherwise_valid_params_is_422(client):
    _setup_grouped(client)
    resp = _list(
        client,
        evidence_bundle_id=_alt_bundle_id("x"),
        limit=10,
        cursor="garbage",
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only guarantee ----------------------------------------------------------


def test_queries_and_failures_write_no_rows_or_audit_events(client, db_session):
    _setup_grouped(client)

    def counts():
        return (
            db_session.scalar(
                select(func.count()).select_from(ExchangeImportRecord)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    receipts_before, events_before = counts()
    assert receipts_before == 5
    assert events_before == 5

    # Successful unfiltered, filtered, and paginated reads.
    cursor = None
    for _ in range(6):
        params = {"limit": 1}
        if cursor is not None:
            params["cursor"] = cursor
        resp = _list(client, **params)
        assert resp.status_code == 200, resp.text
        cursor = resp.json()["next_cursor"]
        if cursor is None:
            break
    _list(client)
    _list(client, evidence_bundle_id=_alt_bundle_id("x"))
    _list(client, manifest_version="provenance-exchange-manifest-v2")
    _list(client, manifest_digest_hex="0" * 64)

    # Failed reads must not write anything either.
    _list(client, evidence_bundle_id=" ")
    _list(client, limit=0)
    _list(client, limit=101)
    _list(client, cursor="tampered")
    _list(client, limit=1, cursor="ae1.x.y")
    client.get(f"{URL}?limit=1&limit=2")
    client.get(f"{URL}?unknown=1")
    _list(client, evidence_bundle_id="a", cursor="garbage")

    db_session.expire_all()
    assert counts() == (5, 5)


# --- Compatibility with the existing single-receipt routes -----------------------


def test_existing_post_and_single_get_remain_unchanged(client):
    created = _import_with_title(client, "compat")
    single = client.get(f"{URL}/{created['id']}")
    assert single.status_code == 200
    assert single.json() == created

    missing = client.get(f"{URL}/eir_{'0' * 64}")
    assert missing.status_code == 404
    assert (
        missing.json()["error"]["code"]
        == "evidence_bundle_exchange_import_not_found"
    )
