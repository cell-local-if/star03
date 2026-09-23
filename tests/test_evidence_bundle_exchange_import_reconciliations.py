"""Tests for the read-only exchange-import reconciliation listing endpoint.

Covers GET /v1/evidence-bundle-exchange-import-reconciliations: the
response shape carries exactly ``{"items", "count", "next_cursor"}``. Each
item is the existing single-receipt public view
(``id``/``manifest_version``/``evidence_bundle_id``/
``manifest_digest_hex``/``received_at``) plus ``local_available``,
``local_manifest_digest_hex``, and ``matches``. Receipts follow stable
creation order and ``count`` is the total number of receipts. Every item is
reconciled using only its own ``evidence_bundle_id``: with no local bundle
the triple is ``false``/``null``/``false``; with a local bundle the current
digest is computed under the existing exchange manifest rules and matches
only character for character. ``limit`` is 1..100 defaulting to 50; the
opaque HMAC cursor binds the effective filters and limit so pages
concatenate without gaps or duplicates and the final cursor is null. Any
other, repeated, blank, or illegal parameter and any
malformed/tampered/foreign/mismatching cursor is 422 validation_error; no
receipts is an empty collection. The optional ``local_available`` and
``matches`` filters (lowercase ``true``/``false`` only) are covered by
``test_evidence_bundle_exchange_import_reconciliation_filters``; with
neither filter present every behavior below is exactly the original
unfiltered contract. The route is strictly read-only — no resource,
receipt, or audit row is written — and no snapshot or raw material is
echoed. All fixtures are deterministic and offline.
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
from tests.helpers import SEED_A
from tests.test_evidence_bundle_exchange import (
    _create_attestation,
    _revoke,
    _setup_bundle,
)
from tests.test_evidence_bundle_exchange_imports import (
    RECEIPT_KEYS,
    _import_request,
    _served_import_request,
)
from tests.test_evidence_bundle_exchange_import_reconciliation import (
    IMPORTS_URL,
)
from tests.test_exchange_manifest_verifications import (
    MANIFEST_VERSION,
    _digest_of,
    _offline_snapshot,
)

URL = "/v1/evidence-bundle-exchange-import-reconciliations"
RECONCILIATION_ITEM_KEYS = RECEIPT_KEYS | {
    "local_available",
    "local_manifest_digest_hex",
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
    assert resp.status_code == 201, resp.text
    return resp.json()


def _offline_receipt(client, title: str) -> dict:
    """Register a receipt whose bundle id is unknown locally."""
    snapshot = _offline_snapshot()
    snapshot["content"]["title"] = title
    return _register(client, _import_request(snapshot))


def _offline_many(client, count: int):
    return [_offline_receipt(client, f"bulk-{i:03d}") for i in range(count)]


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


def test_response_shape_and_item_fields(client):
    _offline_receipt(client, "shape")
    resp = _list(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"items", "count", "next_cursor"}
    assert body["count"] == 1
    assert body["next_cursor"] is None
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert set(item) == RECONCILIATION_ITEM_KEYS
    assert item["id"].startswith("eir_")
    assert item["manifest_version"] == MANIFEST_VERSION


def test_item_carries_the_existing_receipt_public_view(client):
    receipt = _offline_receipt(client, "public-view")
    item = _list(client).json()["items"][0]
    # The receipt half of the item is exactly the existing single-receipt
    # view, including its original received_at.
    assert {key: item[key] for key in RECEIPT_KEYS} == receipt
    assert (
        {key: item[key] for key in RECEIPT_KEYS}
        == client.get(f"{IMPORTS_URL}/{receipt['id']}").json()
    )


def test_received_at_is_utc(client):
    _offline_receipt(client, "utc")
    for item in _list(client).json()["items"]:
        parsed = datetime.fromisoformat(item["received_at"])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)
        assert item["received_at"].endswith(("Z", "+00:00"))


def test_receipts_follow_stable_creation_order(client):
    receipts = [_offline_receipt(client, f"title-{i}") for i in range(5)]
    items = _list(client).json()["items"]
    assert [i["id"] for i in items] == [r["id"] for r in receipts]


def test_count_is_total_receipt_count_on_every_page(client):
    _offline_many(client, 5)
    all_items, pages, count = _walk_pages(client, limit=2)
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    assert len(all_items) == 5


# --- Per-item reconciliation semantics -----------------------------------------


def test_item_without_local_bundle_is_unavailable(client):
    receipt = _offline_receipt(client, "offline")
    item = _list(client).json()["items"][0]
    assert item["local_available"] is False
    assert item["local_manifest_digest_hex"] is None
    assert item["matches"] is False


def test_item_agrees_with_the_single_reconciliation_route(client):
    offline = _offline_receipt(client, "offline")

    _, _, bundle = _setup_bundle(client)
    # Receipt registered against the unattested snapshot; it becomes stale
    # once a later attestation changes the current snapshot.
    stale = _register(client, _served_import_request(client, bundle))
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    # A fresh receipt taken from the now-current snapshot matches.
    matching = _register(client, _served_import_request(client, bundle))
    stale_digest = _single_reconciliation(
        client, stale["id"]
    )["local_manifest_digest_hex"]

    items = {item["id"]: item for item in _list(client).json()["items"]}
    assert set(items) == {offline["id"], stale["id"], matching["id"]}
    for receipt_id, item in items.items():
        single = _single_reconciliation(client, receipt_id)
        assert (
            item["local_available"],
            item["local_manifest_digest_hex"],
            item["matches"],
        ) == (
            single["local_available"],
            single["local_manifest_digest_hex"],
            single["matches"],
        )

    assert items[offline["id"]]["local_available"] is False
    assert items[matching["id"]]["local_available"] is True
    assert items[matching["id"]]["matches"] is True
    assert items[stale["id"]]["local_available"] is True
    assert items[stale["id"]]["local_manifest_digest_hex"] == stale_digest
    assert items[stale["id"]]["matches"] is False


def test_each_item_uses_only_its_own_evidence_bundle_id(client):
    # A local bundle exists for the served receipt only; the unrelated
    # offline receipt must never resolve to that bundle.
    offline = _offline_receipt(client, "offline")
    _, _, bundle = _setup_bundle(client)
    served = _register(client, _served_import_request(client, bundle))

    items = {item["id"]: item for item in _list(client).json()["items"]}
    assert items[offline["id"]]["local_available"] is False
    assert items[offline["id"]]["local_manifest_digest_hex"] is None
    assert items[offline["id"]]["matches"] is False
    assert items[served["id"]]["local_available"] is True
    assert items[served["id"]]["matches"] is True


def test_two_receipts_for_one_bundle_only_the_current_digest_matches(client):
    _, _, bundle = _setup_bundle(client)
    served = _served_import_request(client, bundle)
    current_receipt = _register(client, served)

    stale_request = json.loads(json.dumps(served))
    stale_request["snapshot"]["content"]["title"] = "superseded 标题"
    stale_request["manifest"]["manifest_digest_hex"] = _digest_of(
        stale_request["snapshot"]
    )
    stale_receipt = _register(client, stale_request)

    items = {item["id"]: item for item in _list(client).json()["items"]}
    current_digest = items[current_receipt["id"]]["local_manifest_digest_hex"]
    assert items[current_receipt["id"]]["matches"] is True
    assert items[stale_receipt["id"]]["local_available"] is True
    assert items[stale_receipt["id"]]["local_manifest_digest_hex"] == (
        current_digest
    )
    assert items[stale_receipt["id"]]["matches"] is False


def test_reconciliation_reflects_current_state_on_each_page(client):
    # The current digest is computed at read time: adding an attestation
    # after the import flips a previously matching receipt to non-matching.
    _, _, bundle = _setup_bundle(client)
    receipt = _register(client, _served_import_request(client, bundle))
    assert _list(client).json()["items"][0]["matches"] is True

    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    item = _list(client).json()["items"][0]
    assert item["id"] == receipt["id"]
    assert item["local_available"] is True
    assert item["matches"] is False
    assert item["local_manifest_digest_hex"] != receipt["manifest_digest_hex"]


def test_revocation_does_not_change_the_snapshot_digest(client):
    _, _, bundle = _setup_bundle(client)
    attested = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    receipt = _register(client, _served_import_request(client, bundle))
    _revoke(client, attested["id"])
    item = _list(client).json()["items"][0]
    assert item["local_available"] is True
    assert item["matches"] is True
    assert item["local_manifest_digest_hex"] == receipt["manifest_digest_hex"]


def test_response_carries_no_snapshot_or_raw_material(client):
    _, _, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    _register(client, _served_import_request(client, bundle))
    _offline_receipt(client, "offline")

    serialized = json.dumps(_list(client).json())
    for forbidden in (
        '"snapshot"',
        '"manifest"',
        '"signature"',
        '"payload"',
        '"data"',
        '"evidence"',
        '"public_key"',
    ):
        assert forbidden not in serialized


# --- Pagination ----------------------------------------------------------------


def test_pagination_concatenates_without_gaps_or_duplicates(client):
    receipts = _offline_many(client, 5)
    all_items, pages, count = _walk_pages(client, limit=2)
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    ids = [i["id"] for i in all_items]
    assert len(ids) == len(set(ids)) == 5
    assert ids == [r["id"] for r in receipts]


def test_last_page_cursor_null_on_exact_division(client):
    _offline_many(client, 4)
    first = _list(client, limit=2).json()
    assert first["next_cursor"] is not None
    second = _list(client, limit=2, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 2
    assert second["count"] == 4
    assert second["next_cursor"] is None


def test_default_limit_is_fifty(client):
    _offline_many(client, 55)
    first = _list(client).json()
    assert len(first["items"]) == 50
    assert first["count"] == 55
    assert first["next_cursor"] is not None
    second = _list(client, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 5
    assert second["count"] == 55
    assert second["next_cursor"] is None


def test_limit_boundaries_accepted(client):
    _offline_many(client, 2)
    for value in (1, 100):
        assert _list(client, limit=value).status_code == 200


def test_reusing_a_cursor_replays_the_same_page(client):
    _offline_many(client, 5)
    cursor = _list(client, limit=2).json()["next_cursor"]
    replay_one = _list(client, limit=2, cursor=cursor).json()
    replay_two = _list(client, limit=2, cursor=cursor).json()
    assert replay_one == replay_two


def test_cursor_past_end_returns_empty_page_with_total_count(client, app):
    _offline_many(client, 5)
    token = pagination.encode_typed_cursor(
        app.state.exchange_import_reconciliations_cursor_secret,
        pagination.EXCHANGE_IMPORT_RECONCILIATIONS_CURSOR,
        {
            "local_available": None,
            "matches": None,
            "limit": 50,
            "offset": 99,
        },
    )
    body = _list(client, cursor=token).json()
    assert body == {"items": [], "count": 5, "next_cursor": None}


def test_every_page_item_is_reconciled(client):
    # Reconciliation happens per item on every page, not just the first page.
    _, _, bundle = _setup_bundle(client)
    _register(client, _served_import_request(client, bundle))
    _offline_many(client, 3)
    all_items, _, _ = _walk_pages(client, limit=2)
    assert len(all_items) == 4
    available = [item for item in all_items if item["local_available"]]
    assert len(available) == 1
    assert available[0]["matches"] is True


# --- Cursor integrity ------------------------------------------------------------


def test_tampered_or_malformed_cursors_are_validation_errors(client):
    _offline_many(client, 3)
    good = _list(client, limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    foreign = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.EXCHANGE_IMPORT_RECONCILIATIONS_CURSOR,
        {
            "local_available": None,
            "matches": None,
            "limit": 1,
            "offset": 1,
        },
    )
    for token in (
        "",
        "   ",
        "not-a-cursor",
        "ir1.onlytwoparts",
        "ir1.too.many.parts",
        "ir0.x.y",
        "ir2.x.y",
        "v1.x.y",
        "ce1.x.y",
        "ae1.x.y",
        "ei1.x.y",
        tampered,
        foreign,
    ):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422, repr(token)
        assert resp.json()["error"]["code"] == "validation_error"
        assert "items" not in resp.json()


def test_cursor_signed_with_old_format_marker_is_rejected(client, app):
    _offline_many(client, 2)
    payload = base64.urlsafe_b64encode(
        json.dumps({"limit": 50, "offset": 1}).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(
        hmac_mod.new(
            app.state.exchange_import_reconciliations_cursor_secret,
            f"ir0.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    resp = _list(client, cursor=f"ir0.{payload}.{sig}")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_from_other_families_is_rejected(client):
    _offline_many(client, 3)
    # A real exchange-imports cursor is signed by the same process but uses
    # the ei1 family; it must never resume this collection.
    imports_cursor = client.get(
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
    for token in (imports_cursor, audit_cursor):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


def test_reconciliations_cursor_is_rejected_by_other_endpoints(client):
    _offline_many(client, 2)
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
    _offline_many(client, 5)
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
    _offline_receipt(client, "restart")
    cursor = _list(client, limit=1).json()["next_cursor"]
    client.app.state.exchange_import_reconciliations_cursor_secret = (
        secrets.token_bytes(32)
    )
    assert _list(client, limit=1, cursor=cursor).status_code == 422

    first_app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(first_app) as first_client:
        receipt = _offline_receipt(first_client, "persisted")
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
    _offline_many(client, 2)
    for value in ("0", "101", "-1", "1.5", "abc", "8.0", "  2", ""):
        resp = _list(client, limit=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_validation_errors(client):
    _offline_many(client, 2)
    for suffix in ("limit=1&limit=2", "cursor=x&cursor=y"):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameters_are_validation_errors(client):
    _offline_many(client, 2)
    for suffix in (
        "import_id=eir_x",
        "manifest_version=" + MANIFEST_VERSION,
        "evidence_bundle_id=evb_x",
        "manifest_digest_hex=" + "0" * 64,
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
    _offline_many(client, 2)
    resp = _list(client, limit=10, cursor="garbage")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only guarantee ----------------------------------------------------------


def test_reads_and_failures_write_no_rows_or_audit_events(client, db_session):
    _offline_many(client, 3)
    _, _, bundle = _setup_bundle(client)
    _register(client, _served_import_request(client, bundle))

    def counts():
        return (
            db_session.scalar(
                select(func.count()).select_from(ExchangeImportRecord)
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
    _list(client, limit=1, cursor="ei1.x.y")
    client.get(f"{URL}?limit=1&limit=2")
    client.get(f"{URL}?unknown=1")
    _list(client, evidence_bundle_id="a")

    db_session.expire_all()
    assert counts() == (receipts_before, events_before)


# --- Determinism and compatibility ------------------------------------------------


def test_listing_is_deterministic_across_restart(file_client, tmp_db_url):
    _offline_receipt(file_client, "offline")
    _, _, bundle = _setup_bundle(file_client)
    _register(file_client, _served_import_request(file_client, bundle))
    expected = _list(file_client)
    assert expected.status_code == 200, expected.text

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        resp = _list(client)
        assert resp.status_code == 200, resp.text
        assert resp.json() == expected.json()


def test_existing_single_reconciliation_and_imports_list_remain_unchanged(
    client,
):
    receipt = _offline_receipt(client, "compat")
    single = client.get(f"{IMPORTS_URL}/{receipt['id']}/reconciliation")
    assert single.status_code == 200
    assert single.json() == {
        "import_id": receipt["id"],
        "local_available": False,
        "local_manifest_digest_hex": None,
        "matches": False,
    }

    imports_list = client.get(IMPORTS_URL)
    assert imports_list.status_code == 200
    assert imports_list.json()["items"] == [receipt]

    # The single receipt read and its 404 contract are untouched.
    assert (
        client.get(f"{IMPORTS_URL}/{receipt['id']}").json() == receipt
    )
    missing = client.get(f"{IMPORTS_URL}/eir_{'0' * 64}")
    assert missing.status_code == 404
    assert (
        missing.json()["error"]["code"]
        == "evidence_bundle_exchange_import_not_found"
    )
