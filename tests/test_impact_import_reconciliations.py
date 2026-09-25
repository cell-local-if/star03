"""Tests for the read-only global impact-import reconciliation listing.

Covers GET /v1/impact-import-reconciliations: the success body is compact
UTF-8 JSON terminated by exactly one newline with members in exactly the
order ``items``, ``count``, ``next_cursor``. Each item reuses the existing
single impact-import receipt public view
(``id``/``checkpoint_version``/``impact_count``/``impacts_digest_hex``/
``received_at``) followed by ``local_available``, ``local_checkpoint``, and
``matches``. ``local_available`` is true exactly when the current local
revocation-impact set is non-empty (an empty set still yields its
deterministic checkpoint); ``local_checkpoint`` is the four-field
checkpoint of the complete, unfiltered local impact set under the existing
single-receipt reconciliation rules, and ``matches`` is true only when the
receipt's version, impact count, and digest all equal it. Receipts follow
stable creation order and ``count`` is the filtered total. The optional
``local_available`` and ``matches`` filters accept only lowercase
``true``/``false`` and combine as logical AND against the read-time
results; ``local_available=false&matches=true`` never matches. ``limit`` is
1..100 defaulting to 50; the opaque HMAC cursor (its own ``rr1`` family)
binds both effective filters and the limit so pages concatenate without
gaps or duplicates and the final cursor is null. A non-empty GET body,
blank/illegal/repeated/undeclared parameters, and
blank/malformed/tampered/cross-family/query-mismatching cursors are 422
validation_error; non-GET methods are 405. No receipts is an empty
collection. The route is strictly read-only -- no resource, receipt, or
audit row is written -- and the impacts array and every raw signature,
payload, content, or evidence byte are absent. All fixtures are
deterministic and offline.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import json
import secrets
from datetime import datetime, timedelta

import base64

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
    CheckpointImportRecord,
    Claim,
    Content,
    ContentRelation,
    EvidenceBundle,
    ExchangeImportRecord,
    ImpactImportRecord,
)
from tests.test_impact_import_recon import (
    CHECKPOINT_VERSION,
    EMPTY_DIGEST,
    _insert_receipt,
)
from tests.test_impact_imports import RECEIPT_KEYS, _offline_request
from tests.test_revocation_impacts import _revoke, _world

URL = "/v1/impact-import-reconciliations"
IMPORTS_URL = "/v1/impact-imports"
PACKAGE_URL = "/v1/revocation-impact-package"

RECONCILIATION_ITEM_KEYS = RECEIPT_KEYS | {
    "local_available",
    "local_checkpoint",
    "matches",
}
CHECKPOINT_KEYS = {
    "checkpoint_version",
    "digest_algorithm",
    "impact_count",
    "impacts_digest_hex",
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
    CheckpointImportRecord,
    ImpactImportRecord,
    AuditEvent,
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


def _digest_of(impacts: list) -> str:
    canonical = json.dumps(
        impacts,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _register(client, request: dict) -> dict:
    resp = client.post(IMPORTS_URL, json=request)
    assert resp.status_code in (200, 201), resp.text
    return resp.json()


def _offline_receipt(client, marker: str) -> dict:
    """Register a fabricated receipt with a distinct identity per marker."""
    request = _offline_request()
    request["impacts"][0]["reason"] = f"offline-{marker}"
    request["checkpoint"]["impacts_digest_hex"] = _digest_of(request["impacts"])
    return _register(client, request)


def _offline_many(client, count: int):
    return [_offline_receipt(client, f"bulk-{i:03d}") for i in range(count)]


def _served_receipt(client) -> dict:
    """Register a receipt of the current served local package."""
    package = client.get(PACKAGE_URL).json()
    return _register(client, package)


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


def test_success_body_is_compact_json_with_one_trailing_newline(client):
    _offline_receipt(client, "shape")
    resp = _list(client)
    assert resp.status_code == 200
    raw = resp.content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    text = raw.decode("utf-8")[:-1]
    assert ", " not in text and ": " not in text
    parsed = json.loads(text)
    assert list(parsed) == ["items", "count", "next_cursor"]


def test_response_shape_and_item_fields(client):
    _offline_receipt(client, "shape")
    body = _list(client).json()
    assert set(body) == {"items", "count", "next_cursor"}
    assert body["count"] == 1
    assert body["next_cursor"] is None
    item = body["items"][0]
    assert set(item) == RECONCILIATION_ITEM_KEYS
    assert item["id"].startswith("rii_")
    assert set(item["local_checkpoint"]) == CHECKPOINT_KEYS


def test_item_carries_the_existing_receipt_public_view(client):
    receipt = _offline_receipt(client, "public-view")
    item = _list(client).json()["items"][0]
    assert {key: item[key] for key in RECEIPT_KEYS} == receipt
    assert {key: item[key] for key in RECEIPT_KEYS} == client.get(
        f"{IMPORTS_URL}/{receipt['id']}"
    ).json()


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


def test_count_is_filtered_total_on_every_page(client):
    _offline_many(client, 5)
    all_items, pages, count = _walk_pages(client, limit=2)
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    assert len(all_items) == 5


# --- Per-item reconciliation semantics -----------------------------------------


def test_empty_local_set_yields_deterministic_checkpoint_and_unavailable(client):
    receipt = _offline_receipt(client, "offline")
    item = _list(client).json()["items"][0]
    assert item["local_available"] is False
    assert item["local_checkpoint"] == {
        "checkpoint_version": CHECKPOINT_VERSION,
        "digest_algorithm": "sha256",
        "impact_count": 0,
        "impacts_digest_hex": EMPTY_DIGEST,
    }
    assert item["matches"] is False
    # The offline receipt claims two impacts; the local set has none.
    assert receipt["impact_count"] == 2


def test_item_agrees_with_the_single_reconciliation_route(client, db_session):
    offline = _offline_receipt(client, "offline")
    _world(client)
    # A receipt matching the current non-empty local checkpoint.
    matching = _served_receipt(client)
    # A stale receipt for a count/digest that no longer exists locally.
    stale = _insert_receipt(db_session, CHECKPOINT_VERSION, 99, "a" * 64)

    items = {item["id"]: item for item in _list(client).json()["items"]}
    assert set(items) == {offline["id"], matching["id"], stale}
    for receipt_id, item in items.items():
        single = _single_reconciliation(client, receipt_id)
        assert item["local_checkpoint"] == single["local_checkpoint"]
        assert item["matches"] == single["matches"]

    assert items[offline["id"]]["local_available"] is True
    assert items[matching["id"]]["local_available"] is True
    assert items[matching["id"]]["matches"] is True
    assert items[stale]["matches"] is False


def test_local_available_reflects_a_non_empty_local_set(client):
    _offline_receipt(client, "before-world")
    assert _list(client).json()["items"][0]["local_available"] is False
    _world(client)
    item = _list(client).json()["items"][0]
    assert item["local_available"] is True
    assert item["local_checkpoint"]["impact_count"] == 6


def test_served_package_receipt_matches_immediately(client):
    _world(client)
    receipt = _served_receipt(client)
    item = _list(client).json()["items"][0]
    assert item["id"] == receipt["id"]
    assert item["local_available"] is True
    assert item["matches"] is True


def test_local_set_growing_after_registration_flips_match_to_false(client):
    _world(client)
    receipt = _served_receipt(client)
    assert _list(client).json()["items"][0]["matches"] is True

    impacts = client.get(PACKAGE_URL).json()["impacts"]
    _revoke(client, impacts[1]["attestation_id"], reason="late revocation")

    item = _list(client).json()["items"][0]
    assert item["id"] == receipt["id"]
    assert item["local_available"] is True
    assert item["matches"] is False
    assert (
        item["local_checkpoint"]["impacts_digest_hex"]
        != receipt["impacts_digest_hex"]
    )


def test_each_identity_field_is_required_for_a_match(client, db_session):
    _world(client)
    current = client.get(PACKAGE_URL).json()["checkpoint"]
    versioned = _insert_receipt(
        db_session,
        "provenance-revocation-impact-checkpoint-v2",
        current["impact_count"],
        current["impacts_digest_hex"],
    )
    counted = _insert_receipt(
        db_session,
        current["checkpoint_version"],
        current["impact_count"] + 1,
        current["impacts_digest_hex"],
    )
    digested = _insert_receipt(
        db_session,
        current["checkpoint_version"],
        current["impact_count"],
        "0" * 64,
    )
    items = {item["id"]: item for item in _list(client).json()["items"]}
    assert items[versioned]["matches"] is False
    assert items[counted]["matches"] is False
    assert items[digested]["matches"] is False


def test_reconciliation_is_recomputed_at_read_time_on_every_page(client):
    receipts = _offline_many(client, 3)
    # Empty local set: everything unavailable and non-matching.
    first = _list(client, limit=2).json()
    assert all(i["local_available"] is False for i in first["items"])

    _world(client)
    # A subsequent read (and later page) reflects the now-non-empty set.
    all_items, _, _ = _walk_pages(client, limit=2)
    assert all(i["local_available"] is True for i in all_items)
    assert [i["id"] for i in all_items] == [r["id"] for r in receipts]


def test_response_carries_no_impacts_or_raw_material(client):
    marker = "listing-no-echo-marker-91d2"
    request = _offline_request()
    request["impacts"][0]["reason"] = marker
    request["checkpoint"]["impacts_digest_hex"] = _digest_of(request["impacts"])
    _register(client, request)

    serialized = json.dumps(_list(client).json())
    assert '"impacts"' not in serialized
    assert marker not in serialized
    for forbidden in (
        '"signature"',
        '"payload"',
        '"data"',
        '"evidence"',
        '"public_key"',
    ):
        assert forbidden not in serialized


# --- Filters --------------------------------------------------------------------


def test_local_available_false_when_local_set_is_empty(client, db_session):
    # local_available reflects the single shared local impact set: with an
    # empty local set every receipt is unavailable, regardless of identity.
    one = _offline_receipt(client, "empty-1")
    two = _offline_receipt(client, "empty-2")

    body = _list(client, local_available="false").json()
    assert [i["id"] for i in body["items"]] == [one["id"], two["id"]]
    assert body["count"] == 2
    assert all(i["local_available"] is False for i in body["items"])

    # The true filter matches no receipt while the local set is empty.
    assert _list(client, local_available="true").json() == {
        "items": [],
        "count": 0,
        "next_cursor": None,
    }


def test_local_available_true_when_local_set_is_nonempty(client, db_session):
    # Once the local set is non-empty, EVERY receipt (even a fabricated
    # offline one) reports local_available=true.
    offline = _offline_receipt(client, "offline")
    _world(client)
    matching = _served_receipt(client)

    body = _list(client, local_available="true").json()
    assert [i["id"] for i in body["items"]] == [offline["id"], matching["id"]]
    assert body["count"] == 2
    assert all(i["local_available"] is True for i in body["items"])


def test_matches_true_keeps_only_matching(client):
    _offline_receipt(client, "offline")
    _world(client)
    matching = _served_receipt(client)

    body = _list(client, matches="true").json()
    assert [i["id"] for i in body["items"]] == [matching["id"]]
    assert body["count"] == 1
    assert all(i["matches"] is True for i in body["items"])


def test_matches_false_keeps_offline_and_diverged(client, db_session):
    offline = _offline_receipt(client, "offline")
    _world(client)
    matching = _served_receipt(client)
    diverged = _insert_receipt(db_session, CHECKPOINT_VERSION, 99, "a" * 64)

    body = _list(client, matches="false").json()
    # At read time the local set is non-empty, so both non-matching
    # receipts are also available; the matching receipt is excluded.
    assert {i["id"] for i in body["items"]} == {offline["id"], diverged}
    assert body["count"] == 2
    assert all(i["matches"] is False for i in body["items"])
    assert all(i["local_available"] is True for i in body["items"])


def test_filters_combine_as_logical_and(client, db_session):
    offline = _offline_receipt(client, "offline")  # available, non-matching
    _world(client)
    _served_receipt(client)  # available and matching
    diverged = _insert_receipt(
        db_session, CHECKPOINT_VERSION, 99, "a" * 64
    )  # available, non-matching

    # Available but non-matching -> the offline and diverged receipts.
    body = _list(client, local_available="true", matches="false").json()
    assert {i["id"] for i in body["items"]} == {offline["id"], diverged}
    assert body["count"] == 2
    assert all(i["local_available"] is True for i in body["items"])
    assert all(i["matches"] is False for i in body["items"])

    # Available and matching -> only the served receipt.
    body = _list(client, local_available="true", matches="true").json()
    assert body["count"] == 1
    assert all(i["matches"] is True for i in body["items"])


def test_local_available_false_and_matches_true_is_an_empty_collection(client):
    _offline_receipt(client, "offline")
    _world(client)
    _served_receipt(client)
    # An unavailable local set cannot match: deterministic 200 empty set.
    assert _list(
        client, local_available="false", matches="true"
    ).json() == {"items": [], "count": 0, "next_cursor": None}


def test_filter_pages_concatenate_without_gaps_or_duplicates(client, db_session):
    offline = _offline_many(client, 3)  # non-matching fabricated receipts
    _world(client)
    _served_receipt(client)  # available and matching

    # matches=false selects exactly the three offline receipts; the served
    # matching receipt is excluded.
    items, pages, count = _walk_pages(client, matches="false", limit=2)
    assert count == 3
    assert [len(page) for page in pages] == [2, 1]
    assert [i["id"] for i in items] == [r["id"] for r in offline]
    assert len({i["id"] for i in items}) == 3


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
    one = _list(client, limit=2, cursor=cursor).json()
    two = _list(client, limit=2, cursor=cursor).json()
    assert one == two


def test_cursor_past_end_returns_empty_page_with_total_count(client, app):
    _offline_many(client, 5)
    token = pagination.encode_typed_cursor(
        app.state.impact_import_reconciliations_cursor_secret,
        pagination.IMPACT_IMPORT_RECONCILIATIONS_CURSOR,
        {"local_available": None, "matches": None, "limit": 50, "offset": 99},
    )
    assert _list(client, cursor=token).json() == {
        "items": [],
        "count": 5,
        "next_cursor": None,
    }


# --- Cursor integrity ------------------------------------------------------------


def test_tampered_or_malformed_cursors_are_validation_errors(client):
    _offline_many(client, 3)
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
        "rr1.onlytwoparts",
        "rr1.too.many.parts",
        "rr0.x.y",
        "rr2.x.y",
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
    _offline_many(client, 2)
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {"local_available": None, "matches": None, "limit": 50, "offset": 1}
        ).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(
        hmac_mod.new(
            app.state.impact_import_reconciliations_cursor_secret,
            f"rr0.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    resp = _list(client, cursor=f"rr0.{payload}.{sig}")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_from_other_families_is_rejected(client):
    _offline_many(client, 3)
    imports_cursor = client.get(
        IMPORTS_URL, params={"limit": 1}
    ).json()["next_cursor"]
    exchange_cursor = client.get(
        "/v1/evidence-bundle-exchange-import-reconciliations",
        params={"limit": 1},
    ).json()["next_cursor"]
    for token in (imports_cursor, exchange_cursor):
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
        "/v1/evidence-bundle-exchange-import-reconciliations",
        params={"limit": 1, "cursor": cursor},
    )
    assert resp.status_code == 422


def test_cursor_bound_to_limit(client):
    _offline_many(client, 5)
    cursor = _list(client, limit=2).json()["next_cursor"]
    assert _list(client, limit=3, cursor=cursor).status_code == 422
    # Dropping the explicit limit is also a mismatch: the cursor binds the
    # effective limit, not its spelling.
    assert _list(client, cursor=cursor).status_code == 422
    assert _list(client, limit=2, cursor=cursor).status_code == 200


def test_cursor_bound_to_filters(client):
    _offline_many(client, 3)
    cursor = _list(
        client, limit=1, local_available="false"
    ).json()["next_cursor"]
    # A different effective filter rejects the cursor.
    assert _list(client, limit=1, cursor=cursor).status_code == 422
    assert _list(
        client, limit=1, local_available="true", cursor=cursor
    ).status_code == 422
    assert _list(
        client, limit=1, matches="false", cursor=cursor
    ).status_code == 422
    # Replaying with the bound filter works.
    assert _list(
        client, limit=1, local_available="false", cursor=cursor
    ).status_code == 200


def test_cursor_secret_rotation_invalidates_outstanding_cursors(
    client, tmp_db_url
):
    _offline_receipt(client, "restart")
    cursor = _list(client, limit=1).json()["next_cursor"]
    client.app.state.impact_import_reconciliations_cursor_secret = (
        secrets.token_bytes(32)
    )
    assert _list(client, limit=1, cursor=cursor).status_code == 422

    first_app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(first_app) as first_client:
        receipt = _offline_receipt(first_client, "persisted")
        token = _list(first_client, limit=1).json()["next_cursor"]
    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as second_client:
        assert _list(second_client, limit=1, cursor=token).status_code == 422
        fresh = _list(second_client).json()
        assert [i["id"] for i in fresh["items"]] == [receipt["id"]]


# --- Parameter validation --------------------------------------------------------


def test_illegal_limit_values_are_validation_errors(client):
    _offline_many(client, 2)
    for value in ("0", "101", "-1", "1.5", "abc", "8.0", "  2", ""):
        resp = _list(client, limit=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_boolean_filters_accept_only_lowercase_literals(client):
    _offline_many(client, 2)
    for field in ("local_available", "matches"):
        for value in ("True", "FALSE", "1", "0", "yes", "", "  ", "true "):
            resp = _list(client, **{field: value})
            assert resp.status_code == 422, (field, value)
            assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_validation_errors(client):
    _offline_many(client, 2)
    for suffix in (
        "limit=1&limit=2",
        "cursor=x&cursor=y",
        "local_available=true&local_available=false",
        "matches=true&matches=false",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameters_are_validation_errors(client):
    _offline_many(client, 2)
    for suffix in (
        "import_id=rii_x",
        "checkpoint_version=" + CHECKPOINT_VERSION,
        "impacts_digest_hex=" + "0" * 64,
        "impact_count=2",
        "limit=1&offset=2",
        "CURSOR=x",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_unknown_parameter_is_422_on_empty_database(client):
    resp = _list(client, unknown=1)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_cursor_with_otherwise_valid_params_is_422(client):
    _offline_many(client, 2)
    assert _list(client, limit=10, cursor="garbage").status_code == 422


# --- Empty body boundary ---------------------------------------------------------


def test_any_get_body_is_422_before_any_read(client):
    _offline_receipt(client, "body")
    for payload in (b"x", b" ", b"\n", b"{", b"\xef\xbb\xbf"):
        resp = client.request("GET", URL, content=payload)
        assert resp.status_code == 422, payload
        assert resp.json()["error"]["code"] == "validation_error"


# --- Method boundary -------------------------------------------------------------


def test_non_get_methods_are_405(client):
    for method in (client.post, client.put, client.patch, client.delete):
        resp = method(URL)
        assert resp.status_code == 405, resp.text
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Read-only guarantee ----------------------------------------------------------


def test_reads_and_failures_write_no_rows_or_audit_events(client, db_session):
    _offline_many(client, 3)
    _world(client)
    _served_receipt(client)

    def counts():
        return {
            model: db_session.scalar(
                select(func.count()).select_from(model)
            )
            for model in _DOMAIN_MODELS
        }

    before = counts()

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
    _list(client, local_available="false")
    _list(client, matches="true")
    _list(client, local_available="true", matches="false")

    # Failed reads must not write anything either.
    _list(client, limit=0)
    _list(client, limit=101)
    _list(client, cursor="tampered")
    _list(client, limit=1, cursor="ii1.x.y")
    client.get(f"{URL}?limit=1&limit=2")
    client.get(f"{URL}?unknown=1")
    _list(client, local_available="TRUE")
    client.request("GET", URL, content=b"x")
    client.post(URL)

    db_session.expire_all()
    assert counts() == before


# --- Determinism and compatibility ------------------------------------------------


def test_listing_is_deterministic_across_restart(file_client, tmp_db_url):
    _offline_receipt(file_client, "offline")
    _world(file_client)
    _served_receipt(file_client)
    expected = _list(file_client)
    assert expected.status_code == 200, expected.text

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        resp = _list(client)
        assert resp.status_code == 200, resp.text
        assert resp.json() == expected.json()


def test_existing_import_routes_remain_unchanged(client):
    receipt = _offline_receipt(client, "compat")
    single = client.get(f"{IMPORTS_URL}/{receipt['id']}")
    assert single.status_code == 200
    assert single.json() == receipt

    recon = client.get(f"{IMPORTS_URL}/{receipt['id']}/recon")
    assert recon.status_code == 200
    assert recon.json() == {
        "import_id": receipt["id"],
        "local_checkpoint": {
            "checkpoint_version": CHECKPOINT_VERSION,
            "digest_algorithm": "sha256",
            "impact_count": 0,
            "impacts_digest_hex": EMPTY_DIGEST,
        },
        "matches": False,
    }

    imports_list = client.get(IMPORTS_URL)
    assert imports_list.status_code == 200
    assert imports_list.json()["items"] == [receipt]

    missing = client.get(f"{IMPORTS_URL}/rii_{'0' * 64}")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "impact_import_not_found"
