"""Tests for the read-only impact-import reconciliation listing endpoint.

Covers GET /v1/impact-import-reconciliations: the success body is compact
UTF-8 JSON terminated by exactly one newline, with members in exactly the
order ``items``, ``count``, ``next_cursor``. Each item is the existing
single-receipt public view (``id``/``checkpoint_version``/``impact_count``/
``impacts_digest_hex``/``received_at``) plus ``local_available``,
``local_checkpoint``, and ``matches``. Every item is reconciled against the
current local checkpoint computed over the complete, unfiltered local
revocation-impact set under the existing ``GET /v1/revocation-impact-package``
rules; ``local_available`` is true exactly when that local impact set is
non-empty, and ``matches`` is true only when the receipt's
``checkpoint_version``, ``impact_count``, and ``impacts_digest_hex`` all
equal the local checkpoint fields. The optional ``local_available`` and
``matches`` filters accept only the lowercase literals ``true``/``false``,
are each provided at most once, and combine as logical AND against the
read-time reconciliation. ``count`` is the filtered total; ``limit`` is a
pure decimal integer 1..100 defaulting to 50; the opaque HMAC cursor (its
own ``iir1`` family) binds both effective filters and the effective limit,
so pages concatenate without gaps or duplicates and the final cursor is
null. Blank/illegal/repeated/undeclared parameters, any GET body, and any
malformed/tampered/foreign/mismatching cursor are 422 validation_error;
non-GET methods are 405. The route is strictly read-only -- no resource,
receipt, or audit row is written -- and the imported impacts array is never
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
from provenance.models import AuditEvent, ImpactImportRecord
from tests.test_impact_import_recon import (
    CHECKPOINT_KEYS,
    CHECKPOINT_VERSION,
    EMPTY_DIGEST,
    IMPORTS_URL,
    PACKAGE_URL,
    _digest_of,
    _insert_receipt,
    _matching_receipt,
    _served_checkpoint,
)
from tests.test_impact_imports import RECEIPT_KEYS, _offline_request
from tests.test_revocation_impacts import _world

URL = "/v1/impact-import-reconciliations"
RECONCILIATION_ITEM_KEYS = RECEIPT_KEYS | {
    "local_available",
    "local_checkpoint",
    "matches",
}


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


def _import(client, label: str) -> dict:
    """Register one receipt for a fabricated offline checkpoint."""
    request = _offline_request()
    marker = f"offline-{label}"
    request["impacts"][0]["reason"] = marker
    request["checkpoint"]["impacts_digest_hex"] = _digest_of(request["impacts"])
    return _register(client, request)


def _import_many(client, count: int):
    return [_import(client, f"bulk-{i:03d}") for i in range(count)]


def _single_reconciliation(client, import_id: str) -> dict:
    resp = client.get(f"{IMPORTS_URL}/{import_id}/recon")
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
    receipt_id = _insert_receipt(db_session, CHECKPOINT_VERSION, 0, EMPTY_DIGEST)
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
    assert item["id"].startswith("rii_")
    assert item["checkpoint_version"] == CHECKPOINT_VERSION
    assert set(item["local_checkpoint"]) == CHECKPOINT_KEYS


def test_success_body_is_compact_json_with_one_trailing_newline(client):
    _import(client, "compact")
    resp = _list(client)
    assert resp.status_code == 200
    raw = resp.content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    text = raw.decode("utf-8")[:-1]
    assert text == json.dumps(
        json.loads(text), separators=(",", ":"), ensure_ascii=False
    )
    # Top-level member order is exactly items, count, next_cursor.
    assert list(json.loads(text)) == ["items", "count", "next_cursor"]


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
    # The impacts array and checkpoint envelope stay absent from the item.
    assert "impacts" not in item
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


def test_local_available_reflects_whether_the_local_impact_set_is_non_empty(
    client, db_session
):
    # Empty local impact set: local_available false, but a deterministic
    # checkpoint is still reported.
    empty_receipt = _insert_receipt(
        db_session, CHECKPOINT_VERSION, 0, EMPTY_DIGEST
    )
    item = _list(client).json()["items"][0]
    assert item["id"] == empty_receipt
    assert item["local_available"] is False
    assert item["local_checkpoint"] == {
        "checkpoint_version": CHECKPOINT_VERSION,
        "digest_algorithm": "sha256",
        "impact_count": 0,
        "impacts_digest_hex": EMPTY_DIGEST,
    }
    assert item["matches"] is True

    # A non-empty local impact set flips local_available to true.
    _world(client)
    assert _served_checkpoint(client)["impact_count"] > 0
    item = _list(client).json()["items"][0]
    assert item["local_available"] is True
    assert item["matches"] is False


def test_item_local_checkpoint_is_the_unfiltered_served_checkpoint(
    client, db_session
):
    _world(client)
    _insert_receipt(db_session, "provenance-revocation-impact-checkpoint-v2", 99, "0" * 64)

    served = _served_checkpoint(client)
    items = _list(client).json()["items"]
    assert len(items) == 1
    # The local checkpoint is exactly the unfiltered served checkpoint, and
    # is independently reproducible from the package impacts.
    assert items[0]["local_checkpoint"] == served
    served_impacts = client.get(PACKAGE_URL).json()["impacts"]
    assert items[0]["local_checkpoint"]["impact_count"] == len(served_impacts)
    assert items[0]["local_checkpoint"]["impacts_digest_hex"] == _digest_of(
        served_impacts
    )
    # A filter on the package route never narrows the reconciliation's
    # unfiltered impact set.
    filtered = client.get(
        PACKAGE_URL, params={"reason": "first"}
    ).json()["checkpoint"]
    assert filtered["impact_count"] == 1
    assert _list(client).json()["items"][0]["local_checkpoint"] == served


def test_matching_receipt_item(client, db_session):
    _world(client)
    receipt_id, checkpoint = _matching_receipt(db_session, client)
    item = _list(client).json()["items"][0]
    assert item["id"] == receipt_id
    assert item["local_checkpoint"] == checkpoint
    assert item["local_available"] is True
    assert item["matches"] is True


def test_each_identity_field_is_required_for_a_match(client, db_session):
    _world(client)
    current = _served_checkpoint(client)

    version_id = _insert_receipt(
        db_session,
        "provenance-revocation-impact-checkpoint-v2",
        current["impact_count"],
        current["impacts_digest_hex"],
    )
    count_id = _insert_receipt(
        db_session,
        current["checkpoint_version"],
        current["impact_count"] + 1,
        current["impacts_digest_hex"],
    )
    digest_id = _insert_receipt(
        db_session,
        current["checkpoint_version"],
        current["impact_count"],
        "0" * 64,
    )
    items = {item["id"]: item for item in _list(client).json()["items"]}
    for receipt_id in (version_id, count_id, digest_id):
        assert items[receipt_id]["local_checkpoint"] == current
        assert items[receipt_id]["matches"] is False


def test_items_agree_with_the_single_reconciliation_route(client, db_session):
    _world(client)
    current = _served_checkpoint(client)

    # One receipt matching the current local checkpoint exactly.
    matching = _insert_receipt(
        db_session,
        current["checkpoint_version"],
        current["impact_count"],
        current["impacts_digest_hex"],
    )
    # Two receipts that cannot match.
    wrong_version = _insert_receipt(
        db_session,
        "provenance-revocation-impact-checkpoint-v2",
        current["impact_count"],
        current["impacts_digest_hex"],
    )
    wrong_state = _insert_receipt(db_session, CHECKPOINT_VERSION, 99, "a" * 64)

    items = {item["id"]: item for item in _list(client).json()["items"]}
    assert set(items) == {matching, wrong_version, wrong_state}
    for receipt_id, item in items.items():
        single = _single_reconciliation(client, receipt_id)
        assert item["local_checkpoint"] == single["local_checkpoint"]
        assert item["matches"] == single["matches"]

    assert items[matching]["matches"] is True
    assert items[wrong_version]["matches"] is False
    assert items[wrong_state]["matches"] is False


def test_local_impacts_growing_after_registration_flips_match_to_false(
    client, db_session
):
    _world(client)
    receipt_id, checkpoint = _matching_receipt(db_session, client)
    assert _list(client).json()["items"][0]["matches"] is True

    # Grow the local impact set with one more revocation.
    from tests.test_revocation_impacts import _revoke

    impacts = client.get(PACKAGE_URL).json()["impacts"]
    remaining = [
        a["id"]
        for a in client.get("/v1/attestations").json()["items"]
        if a["id"] not in {imp["attestation_id"] for imp in impacts}
    ]
    _revoke(client, remaining[0], reason="late revocation")
    assert (
        _served_checkpoint(client)["impact_count"] == checkpoint["impact_count"] + 1
    )

    item = _list(client).json()["items"][0]
    assert item["id"] == receipt_id
    assert item["local_checkpoint"] == _served_checkpoint(client)
    assert item["matches"] is False


def test_reconciliation_reflects_current_state_on_every_page(client):
    # Reconciliation happens per item on every page; the fabricated offline
    # receipts never match the (empty) local impact set.
    receipts = _import_many(client, 3)
    all_items, _, _ = _walk_pages(client, limit=2)
    assert [i["id"] for i in all_items] == [r["id"] for r in receipts]
    assert all(item["matches"] is False for item in all_items)
    assert all(item["local_available"] is False for item in all_items)
    # Every page still carries the same current local checkpoint.
    current = _served_checkpoint(client)
    assert all(item["local_checkpoint"] == current for item in all_items)


def test_response_never_echoes_the_imported_impacts(client):
    marker = "reconciliation-list-no-echo-marker-7b2e"
    request = _offline_request()
    request["impacts"][0]["reason"] = marker
    request["checkpoint"]["impacts_digest_hex"] = _digest_of(request["impacts"])
    _register(client, request)

    serialized = json.dumps(_list(client).json(), ensure_ascii=False)
    assert marker not in serialized
    assert '"impacts"' not in serialized
    for impact in request["impacts"]:
        assert impact["id"] not in serialized
        assert impact["created_at"] not in serialized


# --- Filters -----------------------------------------------------------------


def _mixed(client, db_session):
    """A non-empty local impact set plus one matching and two non-matching
    receipts, in stable creation order."""
    _world(client)
    matching, checkpoint = _matching_receipt(db_session, client)
    wrong_version = _insert_receipt(
        db_session,
        "provenance-revocation-impact-checkpoint-v2",
        checkpoint["impact_count"],
        checkpoint["impacts_digest_hex"],
    )
    wrong_state = _insert_receipt(db_session, CHECKPOINT_VERSION, 99, "a" * 64)
    return [matching, wrong_version, wrong_state]


def test_empty_database_with_each_filter_is_empty_collection(client):
    for params in (
        {"local_available": "true"},
        {"matches": "true"},
        {"local_available": "true", "matches": "true"},
    ):
        assert _list(client, **params).json() == {
            "items": [],
            "count": 0,
            "next_cursor": None,
        }, params


def test_matches_true_returns_only_matching_receipts(client, db_session):
    matching, wrong_version, wrong_state = _mixed(client, db_session)
    body = _list(client, matches="true").json()
    assert body["count"] == 1
    assert body["next_cursor"] is None
    assert [item["id"] for item in body["items"]] == [matching]
    assert body["items"][0]["matches"] is True


def test_matches_false_returns_only_non_matching_receipts(client, db_session):
    matching, wrong_version, wrong_state = _mixed(client, db_session)
    body = _list(client, matches="false").json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [wrong_version, wrong_state]
    assert all(item["matches"] is False for item in body["items"])


def test_local_available_filters_on_the_current_local_set(client, db_session):
    matching, wrong_version, wrong_state = _mixed(client, db_session)
    # The local impact set is non-empty: every item reconciles as available.
    body = _list(client, local_available="true").json()
    assert body["count"] == 3
    assert [item["id"] for item in body["items"]] == [
        matching,
        wrong_version,
        wrong_state,
    ]
    assert all(item["local_available"] is True for item in body["items"])

    assert _list(client, local_available="false").json() == {
        "items": [],
        "count": 0,
        "next_cursor": None,
    }


def test_local_available_false_and_matches_true_can_match(client, db_session):
    # An empty local impact set still yields a deterministic checkpoint; a
    # receipt for exactly that empty checkpoint is both unavailable and
    # matching.
    receipt_id = _insert_receipt(db_session, CHECKPOINT_VERSION, 0, EMPTY_DIGEST)
    body = _list(client, local_available="false", matches="true").json()
    assert body["count"] == 1
    assert [item["id"] for item in body["items"]] == [receipt_id]
    item = body["items"][0]
    assert item["local_available"] is False
    assert item["matches"] is True


def test_filters_combine_as_logical_and(client, db_session):
    matching, wrong_version, wrong_state = _mixed(client, db_session)
    body = _list(client, local_available="true", matches="false").json()
    assert [item["id"] for item in body["items"]] == [wrong_version, wrong_state]
    assert body["count"] == 2

    body = _list(client, local_available="true", matches="true").json()
    assert [item["id"] for item in body["items"]] == [matching]
    assert body["count"] == 1


def test_filtered_count_is_the_filtered_total_on_every_page(client, db_session):
    _world(client)
    checkpoint = _served_checkpoint(client)
    _insert_receipt(
        db_session,
        checkpoint["checkpoint_version"],
        checkpoint["impact_count"],
        checkpoint["impacts_digest_hex"],
    )
    for i in range(4):
        _insert_receipt(
            db_session, CHECKPOINT_VERSION, 100 + i, f"{i:064d}"
        )
    all_items, pages, count = _walk_pages(client, matches="false", limit=2)
    assert count == 4
    assert [len(page) for page in pages] == [2, 2]
    assert len(all_items) == 4
    assert all(item["matches"] is False for item in all_items)


def test_filter_values_are_never_resolved_for_existence(client):
    # A filter spelling that cannot match anything is an empty collection,
    # not an error.
    assert _list(client, matches="true").json()["count"] == 0


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
        app.state.impact_import_reconciliations_cursor_secret,
        pagination.IMPACT_IMPORT_RECONCILIATIONS_CURSOR,
        {"local_available": None, "matches": None, "limit": 50, "offset": 99},
    )
    body = _list(client, cursor=token).json()
    assert body == {"items": [], "count": 5, "next_cursor": None}


def test_filtered_cursor_past_end_keeps_the_filtered_count(client, app):
    _import_many(client, 3)
    token = pagination.encode_typed_cursor(
        app.state.impact_import_reconciliations_cursor_secret,
        pagination.IMPACT_IMPORT_RECONCILIATIONS_CURSOR,
        {"local_available": None, "matches": False, "limit": 2, "offset": 99},
    )
    body = _list(client, matches="false", limit=2, cursor=token).json()
    assert body == {"items": [], "count": 3, "next_cursor": None}


# --- Cursor integrity ------------------------------------------------------------


def test_tampered_or_malformed_cursors_are_validation_errors(client):
    _import_many(client, 3)
    good = _list(client, limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    foreign = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.IMPACT_IMPORT_RECONCILIATIONS_CURSOR,
        {"local_available": None, "matches": None, "limit": 1, "offset": 1},
    )
    for token in (
        "",
        "   ",
        "not-a-cursor",
        "iir1.onlytwoparts",
        "iir1.too.many.parts",
        "iir0.x.y",
        "iir2.x.y",
        "v1.x.y",
        "ii1.x.y",
        "ir1.x.y",
        "cr1.x.y",
        "ri1.x.y",
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
        json.dumps(
            {
                "local_available": None,
                "matches": None,
                "limit": 50,
                "offset": 1,
            }
        ).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(
        hmac_mod.new(
            app.state.impact_import_reconciliations_cursor_secret,
            f"iir0.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    resp = _list(client, cursor=f"iir0.{payload}.{sig}")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_from_other_families_is_rejected(client):
    _import_many(client, 3)
    # Real cursors minted by sibling endpoints (same process) must never
    # resume this collection.
    impact_imports_cursor = client.get(
        IMPORTS_URL, params={"limit": 1}
    ).json()["next_cursor"]
    impacts_cursor = client.get(
        "/v1/revocation-impacts", params={"limit": 1}
    ).json()["next_cursor"]
    exchange_recon_cursor = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.EXCHANGE_IMPORT_RECONCILIATIONS_CURSOR,
        {
            "local_available": None,
            "matches": None,
            "limit": 1,
            "offset": 1,
        },
    )
    checkpoint_recon_cursor = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.CHECKPOINT_IMPORT_RECONCILIATIONS_CURSOR,
        {"limit": 1, "offset": 1},
    )
    for token in (
        impact_imports_cursor,
        impacts_cursor,
        exchange_recon_cursor,
        checkpoint_recon_cursor,
    ):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


def test_foreign_family_cursor_signed_with_this_secret_is_rejected(client, app):
    # Family separation is by format marker, not by secret: an impact-imports
    # cursor minted under this endpoint's own secret is rejected.
    _import_many(client, 2)
    foreign = pagination.encode_typed_cursor(
        app.state.impact_import_reconciliations_cursor_secret,
        pagination.IMPACT_IMPORTS_CURSOR,
        {
            "checkpoint_version": None,
            "impacts_digest_hex": None,
            "impact_count": None,
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
        "/v1/revocation-impacts", params={"limit": 1, "cursor": cursor}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_bound_to_filters_and_limit(client):
    _import_many(client, 5)
    cursor = _list(client, limit=2).json()["next_cursor"]
    # A different limit, a dropped limit, or an added/changed filter all
    # mismatch the bound cursor.
    assert _list(client, limit=3, cursor=cursor).status_code == 422
    assert _list(client, cursor=cursor).status_code == 422
    assert (
        _list(client, limit=2, matches="false", cursor=cursor).status_code
        == 422
    )
    assert (
        _list(client, limit=2, local_available="false", cursor=cursor).status_code
        == 422
    )
    # Replaying with the bound conditions still works.
    assert _list(client, limit=2, cursor=cursor).status_code == 200

    filtered_cursor = _list(client, matches="false", limit=2).json()[
        "next_cursor"
    ]
    assert filtered_cursor is not None
    # Dropping the filter or flipping its value mismatches.
    assert _list(client, limit=2, cursor=filtered_cursor).status_code == 422
    assert (
        _list(client, limit=2, matches="true", cursor=filtered_cursor).status_code
        == 422
    )
    assert (
        _list(client, limit=2, matches="false", cursor=filtered_cursor).status_code
        == 200
    )


def test_cursor_secret_rotation_invalidates_outstanding_cursors(
    client, tmp_db_url
):
    _import(client, "restart")
    cursor = _list(client, limit=1).json()["next_cursor"]
    client.app.state.impact_import_reconciliations_cursor_secret = (
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


def test_blank_filters_are_validation_errors(client):
    _import_many(client, 2)
    for params in (
        {"local_available": ""},
        {"local_available": " "},
        {"matches": ""},
        {"matches": "  "},
    ):
        resp = _list(client, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_lowercase_boolean_filters_are_validation_errors(client):
    _import_many(client, 2)
    for field in ("local_available", "matches"):
        for value in ("True", "FALSE", "1", "0", "yes", "true ", " true"):
            resp = _list(client, **{field: value})
            assert resp.status_code == 422, (field, value)
            assert resp.json()["error"]["code"] == "validation_error"


def test_illegal_limit_values_are_validation_errors(client):
    _import_many(client, 2)
    for value in ("0", "101", "-1", "1.5", "abc", "8.0", "  2", "", "+2", "2 "):
        resp = _list(client, limit=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_validation_errors(client):
    _import_many(client, 2)
    for suffix in (
        "limit=1&limit=2",
        "cursor=x&cursor=y",
        "local_available=true&local_available=false",
        "matches=true&matches=true",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameters_are_validation_errors(client):
    _import_many(client, 2)
    for suffix in (
        "import_id=rii_x",
        "checkpoint_version=" + CHECKPOINT_VERSION,
        "impacts_digest_hex=" + "0" * 64,
        "impact_count=2",
        "local_checkpoint=x",
        "offset=2",
        "CURSOR=x",
        "Limit=1",
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


def test_non_empty_get_body_is_422(client):
    _import_many(client, 2)
    for content in (b"{}", b" ", b"\n", b"not-json", b"\x00"):
        resp = client.request("GET", URL, content=content)
        assert resp.status_code == 422, content
        assert resp.json()["error"]["code"] == "validation_error"
    # A body is rejected even on an otherwise valid filtered query.
    resp = client.request("GET", URL, params={"limit": 1}, content=b"{}")
    assert resp.status_code == 422


def test_non_get_methods_are_405(client):
    for method in ("put", "patch", "delete", "post"):
        resp = getattr(client, method)(URL)
        assert resp.status_code == 405, method
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Read-only guarantee ----------------------------------------------------------


def test_reads_and_failures_write_no_rows_or_audit_events(client, db_session):
    _import_many(client, 3)

    def counts():
        return (
            db_session.scalar(
                select(func.count()).select_from(ImpactImportRecord)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    receipts_before, events_before = counts()
    assert receipts_before == 3
    assert events_before == 3

    # Successful reads, including a full page walk and filtered reads.
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
    _list(client, matches="false")
    _list(client, local_available="false", matches="true")

    # Failed reads must not write anything either.
    _list(client, limit=0)
    _list(client, limit=101)
    _list(client, matches="TRUE")
    _list(client, cursor="tampered")
    _list(client, limit=1, cursor="ii1.x.y")
    client.get(f"{URL}?limit=1&limit=2")
    client.get(f"{URL}?unknown=1")
    client.request("GET", URL, content=b"{}")

    db_session.expire_all()
    assert counts() == (receipts_before, events_before)


# --- Determinism and compatibility ------------------------------------------------


def test_listing_is_deterministic_across_restart(file_client, tmp_db_url):
    # Direct, event-free fixture state at a fixed instant so the verdict and
    # ordering survive identically across a process restart.
    session = file_client.app.state.session_factory()
    try:
        matching = _insert_receipt(session, CHECKPOINT_VERSION, 0, EMPTY_DIGEST)
        non_matching = _insert_receipt(
            session, "provenance-revocation-impact-checkpoint-v2", 0, EMPTY_DIGEST
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
        assert items[matching]["local_available"] is False
        assert items[non_matching]["matches"] is False


def test_existing_import_recon_and_package_routes_remain_unchanged(client):
    receipt = _import(client, "compat")

    # The single reconciliation route keeps its three-member body.
    single = client.get(f"{IMPORTS_URL}/{receipt['id']}/recon")
    assert single.status_code == 200
    single_body = single.json()
    assert set(single_body) == {"import_id", "local_checkpoint", "matches"}
    assert single_body["import_id"] == receipt["id"]

    # The unfiltered package route is unchanged.
    package = client.get(PACKAGE_URL)
    assert package.status_code == 200
    assert single_body["local_checkpoint"] == package.json()["checkpoint"]

    # The impact-imports receipt search and single read are unchanged.
    imports_list = client.get(IMPORTS_URL)
    assert imports_list.status_code == 200
    assert imports_list.json()["items"] == [receipt]
    assert client.get(f"{IMPORTS_URL}/{receipt['id']}").json() == receipt

    missing = client.get(f"{IMPORTS_URL}/rii_{'0' * 64}")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "impact_import_not_found"
