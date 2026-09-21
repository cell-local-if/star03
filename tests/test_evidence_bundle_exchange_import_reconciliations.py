"""Tests for the read-only exchange-import reconciliation search endpoint.

Covers GET /v1/evidence-bundle-exchange-import-reconciliations: the response
shape carries exactly ``{"items", "count", "next_cursor"}`` with every item
exposing the existing single-receipt public fields (``id``/
``manifest_version``/``evidence_bundle_id``/``manifest_digest_hex``/
``received_at``) plus the three reconciliation fields of the existing
per-receipt reconciliation route (``local_available``/
``local_manifest_digest_hex``/``matches``). Each item is reconciled using
only its own ``evidence_bundle_id`` against the local bundle: no local
bundle is ``(false, null, false)``; a local bundle contributes the current
exchange-manifest digest and matches only character-for-character. Items
follow stable receipt creation order. ``limit`` is 1..100 defaulting to 50
and the opaque HMAC cursor binds the limit so pages concatenate without
gaps or duplicates and the final cursor is null. Any other, repeated,
blank, or illegal parameter and any blank/malformed/tampered/foreign-
family/query-mismatching cursor is 422 validation_error; no receipts is an
empty collection. The route is strictly read-only — no resource, receipt,
or audit row is written — and never echoes a snapshot or raw material. All
fixtures are deterministic and offline.
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
from provenance.models import (
    Actor,
    Attestation,
    AttestationRevocation,
    AuditEvent,
    Claim,
    Content,
    ContentRelation,
    EvidenceBundle,
    ExchangeImportRecord,
)
from tests.helpers import SEED_A
from tests.test_evidence_bundle_exchange import (
    _create_attestation,
    _setup_bundle,
)
from tests.test_evidence_bundle_exchange_imports import (
    _import_request,
    _served_import_request,
)
from tests.test_evidence_bundle_exchange_imports_list import (
    _import_for_bundle,
    _import_with_title,
)
from tests.test_exchange_manifest_verifications import (
    MANIFEST_VERSION,
    _digest_of,
)

URL = "/v1/evidence-bundle-exchange-import-reconciliations"
IMPORTS_URL = "/v1/evidence-bundle-exchange-imports"
RECEIPT_KEYS = {
    "id",
    "manifest_version",
    "evidence_bundle_id",
    "manifest_digest_hex",
    "received_at",
}
ITEM_KEYS = RECEIPT_KEYS | {
    "local_available",
    "local_manifest_digest_hex",
    "matches",
}
_DOMAIN_MODELS = (
    Actor,
    Content,
    Claim,
    EvidenceBundle,
    Attestation,
    AttestationRevocation,
    ContentRelation,
    ExchangeImportRecord,
    AuditEvent,
)


def _list(client, **params):
    return client.get(URL, params=params)


def _register(client, request: dict) -> dict:
    resp = client.post(IMPORTS_URL, json=request)
    assert resp.status_code == 201, resp.text
    return resp.json()


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


def _manifest_digest_of(client, bundle) -> str:
    resp = client.get(f"/v1/evidence-bundles/{bundle['id']}/exchange/manifest")
    assert resp.status_code == 200, resp.text
    return resp.json()["manifest_digest_hex"]


def _import_many(client, count: int):
    return [_import_with_title(client, f"bulk-{i:03d}") for i in range(count)]


def _counts(db_session):
    return {
        model: db_session.execute(
            select(func.count()).select_from(model)
        ).scalar_one()
        for model in _DOMAIN_MODELS
    }


# --- Response shape, fields, and ordering -------------------------------------


def test_empty_database_is_an_empty_collection(client):
    assert _list(client).json() == {
        "items": [],
        "count": 0,
        "next_cursor": None,
    }


def test_response_shape_and_item_fields(client):
    receipts = _import_many(client, 3)
    resp = _list(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"items", "count", "next_cursor"}
    assert body["count"] == 3
    assert body["next_cursor"] is None
    assert len(body["items"]) == 3
    for item in body["items"]:
        assert set(item) == ITEM_KEYS
        assert item["id"].startswith("eir_")
        assert item["manifest_version"] == MANIFEST_VERSION


def test_items_follow_stable_creation_order_and_embed_the_receipt(client):
    receipts = _import_many(client, 4)
    items = _list(client).json()["items"]
    assert [i["id"] for i in items] == [r["id"] for r in receipts]
    for item, receipt in zip(items, receipts):
        assert {k: item[k] for k in RECEIPT_KEYS} == receipt


def test_received_at_is_utc(client):
    _import_many(client, 2)
    for item in _list(client).json()["items"]:
        parsed = datetime.fromisoformat(item["received_at"])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)
        assert item["received_at"].endswith(("Z", "+00:00"))


def test_item_matches_single_receipt_and_single_reconciliation(client):
    receipt = _register(client, _import_request())
    item = _list(client).json()["items"][0]
    # The embedded receipt is exactly the single-receipt public view.
    assert {k: item[k] for k in RECEIPT_KEYS} == client.get(
        f"{IMPORTS_URL}/{receipt['id']}"
    ).json()
    # The three reconciliation fields match the per-receipt route.
    single = client.get(
        f"{IMPORTS_URL}/{receipt['id']}/reconciliation"
    ).json()
    assert item["local_available"] == single["local_available"]
    assert item["local_manifest_digest_hex"] == single[
        "local_manifest_digest_hex"
    ]
    assert item["matches"] == single["matches"]


# --- Per-item reconciliation ----------------------------------------------------


def test_unavailable_receipts_report_false_null_false(client):
    # Offline packages reference bundle ids that do not exist locally.
    _import_for_bundle(client, "a", variation="a")
    _import_with_title(client, "offline")
    for item in _list(client).json()["items"]:
        assert item["local_available"] is False
        assert item["local_manifest_digest_hex"] is None
        assert item["matches"] is False


def test_available_matching_receipt_reconciles_true(client):
    _, _, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    receipt = _register(client, _served_import_request(client, bundle))
    # An unrelated offline receipt sits alongside it.
    offline = _register(client, _import_request())

    by_id = {i["id"]: i for i in _list(client).json()["items"]}
    current = _manifest_digest_of(client, bundle)
    assert by_id[receipt["id"]] == {
        **receipt,
        "local_available": True,
        "local_manifest_digest_hex": current,
        "matches": True,
    }
    assert by_id[offline["id"]]["local_available"] is False
    assert by_id[offline["id"]]["local_manifest_digest_hex"] is None
    assert by_id[offline["id"]]["matches"] is False


def test_diverged_local_state_reports_current_digest_without_match(client):
    _, _, bundle = _setup_bundle(client)
    receipt = _register(client, _served_import_request(client, bundle))
    # A later attestation changes the current snapshot digest.
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    current = _manifest_digest_of(client, bundle)
    assert current != receipt["manifest_digest_hex"]

    item = next(
        i for i in _list(client).json()["items"] if i["id"] == receipt["id"]
    )
    assert item == {
        **receipt,
        "local_available": True,
        "local_manifest_digest_hex": current,
        "matches": False,
    }


def test_each_item_uses_only_its_own_bundle_id(client):
    # One local bundle exists; a receipt for a different unknown id must not
    # be reconciled against the unrelated bundle.
    _, _, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    matching = _register(client, _served_import_request(client, bundle))
    foreign = _register(client, _import_request())

    by_id = {i["id"]: i for i in _list(client).json()["items"]}
    assert by_id[matching["id"]]["local_available"] is True
    assert by_id[foreign["id"]] == {
        **foreign,
        "local_available": False,
        "local_manifest_digest_hex": None,
        "matches": False,
    }


def test_two_receipts_same_bundle_only_current_one_matches(client):
    _, _, bundle = _setup_bundle(client)
    served = _served_import_request(client, bundle)
    current_receipt = _register(client, served)

    stale_request = json.loads(json.dumps(served))
    stale_request["snapshot"]["content"]["title"] = "superseded 标题"
    stale_request["manifest"]["manifest_digest_hex"] = _digest_of(
        stale_request["snapshot"]
    )
    stale_receipt = _register(client, stale_request)

    current_digest = _manifest_digest_of(client, bundle)
    by_id = {i["id"]: i for i in _list(client).json()["items"]}
    assert by_id[current_receipt["id"]]["matches"] is True
    assert by_id[current_receipt["id"]]["local_manifest_digest_hex"] == current_digest
    assert by_id[stale_receipt["id"]]["matches"] is False
    assert by_id[stale_receipt["id"]]["local_available"] is True
    assert by_id[stale_receipt["id"]]["local_manifest_digest_hex"] == current_digest


# --- Pagination -----------------------------------------------------------------


def test_pagination_concatenates_without_gaps_or_duplicates(client):
    _import_many(client, 5)
    all_items, pages, count = _walk_pages(client, limit=2)
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    ids = [i["id"] for i in all_items]
    assert len(ids) == len(set(ids)) == 5
    assert all_items == _list(client).json()["items"]


def test_count_is_total_on_every_page(client):
    _import_many(client, 5)
    _, pages, count = _walk_pages(client, limit=3)
    assert count == 5
    assert [len(page) for page in pages] == [3, 2]


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
    _import_many(client, 1)
    for value in (1, 100):
        assert _list(client, limit=value).status_code == 200


def test_reusing_a_cursor_replays_the_same_page(client):
    _import_many(client, 5)
    cursor = _list(client, limit=2).json()["next_cursor"]
    assert _list(client, limit=2, cursor=cursor).json() == _list(
        client, limit=2, cursor=cursor
    ).json()


def test_cursor_past_end_returns_empty_page_with_total_count(client, app):
    _import_many(client, 5)
    token = pagination.encode_typed_cursor(
        app.state.exchange_import_reconciliations_cursor_secret,
        pagination.EXCHANGE_IMPORT_RECONCILIATIONS_CURSOR,
        {"limit": 50, "offset": 99},
    )
    body = _list(client, cursor=token).json()
    assert body == {"items": [], "count": 5, "next_cursor": None}


def test_pagination_survives_local_state_per_item(client):
    # Reconciliation happens per page; walking all pages reconciles every
    # receipt exactly once regardless of local availability.
    _, _, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    matching = _register(client, _served_import_request(client, bundle))
    offline_ids = [r["id"] for r in _import_many(client, 4)]
    all_items, _, count = _walk_pages(client, limit=2)
    assert count == 5
    assert {i["id"] for i in all_items} == {matching["id"], *offline_ids}
    by_id = {i["id"]: i for i in all_items}
    assert by_id[matching["id"]]["matches"] is True
    assert all(by_id[i]["local_available"] is False for i in offline_ids)


def test_listing_is_deterministic_across_restart(file_client, tmp_db_url):
    _, _, bundle = _setup_bundle(file_client)
    _register(file_client, _served_import_request(file_client, bundle))
    _register(file_client, _import_request())
    expected = file_client.get(URL)
    assert expected.status_code == 200, expected.text

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        resp = client.get(URL)
        assert resp.status_code == 200, resp.text
        assert resp.json() == expected.json()


# --- Cursor integrity ------------------------------------------------------------


def test_tampered_or_malformed_cursors_are_validation_errors(client):
    _import_many(client, 3)
    good = _list(client, limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    foreign = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.EXCHANGE_IMPORT_RECONCILIATIONS_CURSOR,
        {"limit": 1, "offset": 1},
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
        "ei1.x.y",
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
    _import_many(client, 3)
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
    _import_many(client, 3)
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
    imports_cursor = pagination.encode_typed_cursor(
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
    for token in (lineage_cursor, imports_cursor):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


def test_reconciliations_cursor_is_rejected_by_other_endpoints(client):
    _import_many(client, 3)
    cursor = _list(client, limit=1).json()["next_cursor"]
    resp = client.get(
        IMPORTS_URL, params={"limit": 1, "cursor": cursor}
    )
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
    # Resuming under a different effective limit is a mismatch.
    resp = _list(client, limit=3, cursor=cursor)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    # The default limit (50) also differs from the bound limit (2).
    resp = _list(client, cursor=cursor)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_secret_rotation_invalidates_outstanding_cursors(
    client, tmp_db_url
):
    _import_many(client, 2)
    cursor = _list(client, limit=1).json()["next_cursor"]
    client.app.state.exchange_import_reconciliations_cursor_secret = (
        secrets.token_bytes(32)
    )
    assert _list(client, limit=1, cursor=cursor).status_code == 422

    first_app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(first_app) as first_client:
        _import_with_title(first_client, "persisted")
        token = _list(first_client, limit=1).json()["next_cursor"]
    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as second_client:
        assert _list(second_client, limit=1, cursor=token).status_code == 422
        # The receipts survive the restart and still reconcile.
        fresh = _list(second_client).json()
        assert fresh["count"] == 1


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
        "manifest_version=" + MANIFEST_VERSION,
        "evidence_bundle_id=evb_x",
        "manifest_digest_hex=" + "0" * 64,
        "import_id=eir_x",
        "limit=1&offset=2",
        "CURSOR=x",
        "unknown=",
        "a=1&b=2",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_cursor_with_otherwise_valid_params_is_422(client):
    _import_many(client, 2)
    resp = _list(client, limit=10, cursor="garbage")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only and no-echo boundary ---------------------------------------------


def test_queries_and_failures_write_no_rows_or_audit_events(client, db_session):
    _, _, bundle = _setup_bundle(client)
    matching = _register(client, _served_import_request(client, bundle))
    offline = _register(client, _import_request())
    assert matching["id"] != offline["id"]

    before = _counts(db_session)

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
    # Failed reads must not write anything either.
    _list(client, limit=0)
    _list(client, limit=101)
    _list(client, cursor="tampered")
    _list(client, limit=1, cursor="ir1.x.y")
    client.get(f"{URL}?limit=1&limit=2")
    client.get(f"{URL}?unknown=1")
    client.get(f"{URL}?evidence_bundle_id=a")

    db_session.expire_all()
    assert _counts(db_session) == before


def test_response_carries_no_snapshot_or_raw_material(client):
    _, _, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    _register(client, _served_import_request(client, bundle))
    _register(client, _import_request())

    body = _list(client).json()
    serialized = json.dumps(body)
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
