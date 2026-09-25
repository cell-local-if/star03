"""Tests for the read-only impact-import receipt search endpoint.

Covers GET /v1/impact-imports: the success body is compact UTF-8 JSON
terminated by exactly one newline with members exactly ``items``,
``count``, ``next_cursor``; each item reuses the existing single-receipt
public view (``id``/``checkpoint_version``/``impact_count``/
``impacts_digest_hex``/timezone-aware UTC ``received_at``) and never
echoes the impacts array or any raw signature, payload, content, or
evidence byte. Receipts follow stable creation order. The three
filters -- ``checkpoint_version`` and ``impacts_digest_hex`` as
non-empty, case-/whitespace-sensitive exact matches and ``impact_count``
as a non-negative pure decimal integer -- are each provided at most once
and combine as logical AND (absent means unfiltered); unknown values
yield an empty collection. ``limit`` is 1..100 defaulting to 50 and the
opaque HMAC cursor binds every effective filter and the effective limit
so pages concatenate without gaps or duplicates, the final cursor is
null, and a cursor past the tail keeps the total count with an empty
page. Empty/whitespace bodies, blank/illegal/repeated/undeclared
parameters, and blank/malformed/tampered/wrong-signature/cross-family/
cross-entry/query-mismatching cursors are 422 validation_error;
non-GET methods are 405. Neither queries nor failures write any receipt
or audit row. All fixtures are deterministic and offline.
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
from provenance.models import AuditEvent, ImpactImportRecord
from tests.test_impact_imports import (
    CHECKPOINT_VERSION,
    RECEIPT_KEYS,
    URL,
    _digest_of,
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


def _impact(label: str, index: int) -> dict:
    """One structurally valid, internally consistent fabricated impact."""
    revoked = index % 2 == 0
    return {
        "id": "rev_" + hashlib.sha256(f"rev-{label}-{index}".encode()).hexdigest(),
        "attestation_id": "att_"
        + hashlib.sha256(f"att-{label}-{index}".encode()).hexdigest(),
        "revoker_actor_id": "org-émoji-✓",
        "reason": f"reason-{label}-{index}",
        "created_at": (
            f"2026-01-02T03:{index // 60:02d}:{index % 60:02d}Z"
            if index < 3600
            else "2026-01-02T04:00:00Z"
        ),
        "content_id": "cnt_"
        + hashlib.sha256(f"cnt-{label}-{index}".encode()).hexdigest(),
        "target_type": "claim" if index % 2 == 0 else "evidence_bundle",
        "signer_actor_id": f"org-signer-{label}-{index}",
        "qualified_signer_count_after": 0 if revoked else 1,
        "coverage_status_after": "partial" if revoked else "covered",
        "qualified_signer_count_before": 1,
        "coverage_status_before": "covered",
        "qualified_signer_count_delta": 1 if revoked else 0,
    }


def _impacts(label: str, count: int) -> list:
    return [_impact(label, i) for i in range(count)]


def _request(label: str, count: int) -> dict:
    impacts = _impacts(label, count)
    return {
        "checkpoint": {
            "checkpoint_version": CHECKPOINT_VERSION,
            "digest_algorithm": "sha256",
            "impact_count": len(impacts),
            "impacts_digest_hex": _digest_of(impacts),
        },
        "impacts": impacts,
    }


def _import(client, label: str, count: int = 2) -> dict:
    resp = client.post(URL, json=_request(label, count))
    assert resp.status_code == 201, resp.text
    return resp.json()


def _setup_grouped(client):
    """Three receipts with impact_count 2 and two with impact_count 3."""
    receipts = [_import(client, f"pair-{i}", count=2) for i in range(3)]
    receipts += [_import(client, f"trio-{i}", count=3) for i in range(2)]
    return receipts


def _import_many(client, count: int):
    return [_import(client, f"bulk-{i:03d}", count=1) for i in range(count)]


# --- Response shape, fields, and ordering -------------------------------------


def test_empty_database_is_an_empty_collection(client):
    assert _list(client).json() == {
        "items": [],
        "count": 0,
        "next_cursor": None,
    }


def test_success_body_is_compact_json_terminated_by_one_newline(client):
    _import(client, "raw")
    resp = _list(client)
    assert resp.status_code == 200
    raw = resp.content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    # Compact separators: no whitespace after JSON punctuation.
    assert b", " not in raw and b": " not in raw
    assert json.loads(raw.decode("utf-8")) is not None
    # Top-level member order is items, count, next_cursor.
    text = raw.decode("utf-8")
    assert text.index('"items"') < text.index('"count"')
    assert text.index('"count"') < text.index('"next_cursor"')


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
        assert item["id"].startswith("rii_")
        assert item["checkpoint_version"] == CHECKPOINT_VERSION
        assert isinstance(item["impact_count"], int)


def test_item_is_the_public_receipt_and_matches_the_single_read(client):
    created = _import(client, "shape")
    item = _list(client).json()["items"][0]
    assert item == created
    assert item == client.get(f"{URL}/{created['id']}").json()
    # The impacts array and the checkpoint envelope stay absent.
    assert "impacts" not in item
    assert "checkpoint" not in item
    assert "digest_algorithm" not in item
    # No raw signature/payload/content/evidence field can appear.
    for forbidden in ("signature", "payload", "content", "evidence"):
        assert forbidden not in item


def test_received_at_is_utc(client):
    _setup_grouped(client)
    for item in _list(client).json()["items"]:
        parsed = datetime.fromisoformat(item["received_at"])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)
        assert item["received_at"].endswith(("Z", "+00:00"))


def test_receipts_follow_stable_creation_order(client):
    receipts = _setup_grouped(client)
    items = _list(client).json()["items"]
    assert [i["id"] for i in items] == [r["id"] for r in receipts]
    assert items == receipts


def test_listing_after_retry_keeps_the_original_position(client):
    receipt = _import(client, "once")
    retry = client.post(URL, json=_request("once", 2))
    assert retry.status_code == 200
    later = _import(client, "later")
    items = _list(client).json()["items"]
    assert [i["id"] for i in items] == [receipt["id"], later["id"]]


# --- Exact-match filtering -----------------------------------------------------


def test_checkpoint_version_exact_match(client):
    _setup_grouped(client)
    body = _list(client, checkpoint_version=CHECKPOINT_VERSION).json()
    assert body["count"] == 5
    assert {i["checkpoint_version"] for i in body["items"]} == {
        CHECKPOINT_VERSION
    }

    body = _list(
        client, checkpoint_version="provenance-revocation-impact-checkpoint-v2"
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


def test_impacts_digest_hex_exact_match(client):
    receipts = _setup_grouped(client)
    target = receipts[1]
    body = _list(
        client, impacts_digest_hex=target["impacts_digest_hex"]
    ).json()
    assert body["count"] == 1
    assert body["items"][0] == target


def test_impact_count_exact_match(client):
    _setup_grouped(client)
    body = _list(client, impact_count=2).json()
    assert body["count"] == 3
    assert {i["impact_count"] for i in body["items"]} == {2}

    body = _list(client, impact_count=3).json()
    assert body["count"] == 2
    assert {i["impact_count"] for i in body["items"]} == {3}

    body = _list(client, impact_count=0).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


def test_impact_count_zero_matches_empty_sequence_receipts(client):
    empty = _import(client, "empty", count=0)
    assert empty["impact_count"] == 0
    _import(client, "nonempty", count=2)
    body = _list(client, impact_count=0).json()
    assert body["count"] == 1
    assert body["items"][0] == empty


def test_filters_combine_as_logical_and(client):
    receipts = _setup_grouped(client)
    target = receipts[0]
    body = _list(
        client,
        checkpoint_version=CHECKPOINT_VERSION,
        impacts_digest_hex=target["impacts_digest_hex"],
        impact_count=2,
    ).json()
    assert [i["id"] for i in body["items"]] == [target["id"]]
    assert body["count"] == 1

    # A digest paired with another receipt's count satisfies nothing.
    body = _list(
        client,
        impacts_digest_hex=target["impacts_digest_hex"],
        impact_count=3,
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


def test_exact_match_is_case_and_whitespace_sensitive(client):
    receipts = _setup_grouped(client)
    digest = receipts[0]["impacts_digest_hex"]
    for params in (
        {"checkpoint_version": CHECKPOINT_VERSION.upper()},
        {"checkpoint_version": f" {CHECKPOINT_VERSION}"},
        {"checkpoint_version": f"{CHECKPOINT_VERSION} "},
        {"impacts_digest_hex": digest.upper()},
        {"impacts_digest_hex": f" {digest}"},
        {"impacts_digest_hex": f"{digest} "},
    ):
        body = _list(client, **params).json()
        assert body == {"items": [], "count": 0, "next_cursor": None}, params


def test_no_matching_receipts_is_empty_collection(client):
    _setup_grouped(client)
    body = _list(client, impacts_digest_hex="0" * 64).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


# --- Pagination -----------------------------------------------------------------


def test_pagination_concatenates_without_gaps_or_duplicates(client):
    _setup_grouped(client)
    all_items, pages, count = _walk_pages(client, limit=2)
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    ids = [i["id"] for i in all_items]
    assert len(ids) == len(set(ids)) == 5
    assert all_items == _list(client).json()["items"]


def test_count_is_filtered_total_on_every_page(client):
    _setup_grouped(client)
    all_items, pages, count = _walk_pages(client, impact_count=2, limit=2)
    assert count == 3
    assert [len(page) for page in pages] == [2, 1]
    assert {i["impact_count"] for i in all_items} == {2}


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
        app.state.impact_imports_cursor_secret,
        pagination.IMPACT_IMPORTS_CURSOR,
        {
            "checkpoint_version": None,
            "impacts_digest_hex": None,
            "impact_count": None,
            "limit": 50,
            "offset": 99,
        },
    )
    body = _list(client, cursor=token).json()
    assert body["items"] == []
    assert body["count"] == 5
    assert body["next_cursor"] is None


def test_past_tail_differs_from_no_match(client, app):
    # A past-the-tail cursor keeps the filtered total; a no-match filter has
    # count 0.
    _setup_grouped(client)
    token = pagination.encode_typed_cursor(
        app.state.impact_imports_cursor_secret,
        pagination.IMPACT_IMPORTS_CURSOR,
        {
            "checkpoint_version": None,
            "impacts_digest_hex": None,
            "impact_count": None,
            "limit": 50,
            "offset": 7,
        },
    )
    body = _list(client, cursor=token).json()
    assert body == {"items": [], "count": 5, "next_cursor": None}

    no_match = _list(client, impacts_digest_hex="0" * 64).json()
    assert no_match == {"items": [], "count": 0, "next_cursor": None}


# --- Cursor integrity ------------------------------------------------------------


def test_tampered_or_malformed_cursors_are_validation_errors(client):
    _setup_grouped(client)
    good = _list(client, limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    foreign = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.IMPACT_IMPORTS_CURSOR,
        {
            "checkpoint_version": None,
            "impacts_digest_hex": None,
            "impact_count": None,
            "limit": 1,
            "offset": 1,
        },
    )
    for token in (
        "",
        "   ",
        "not-a-cursor",
        "ii1.onlytwoparts",
        "ii1.too.many.parts",
        "ii0.x.y",
        "ii2.x.y",
        "v1.x.y",
        "ci1.x.y",
        "ri1.x.y",
        "ei1.x.y",
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
                "checkpoint_version": None,
                "impacts_digest_hex": None,
                "impact_count": None,
                "limit": 50,
                "offset": 1,
            }
        ).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(
        hmac_mod.new(
            app.state.impact_imports_cursor_secret,
            f"ii0.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    resp = _list(client, cursor=f"ii0.{payload}.{sig}")
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
    checkpoint_cursor = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.CHECKPOINT_IMPORTS_CURSOR,
        {
            "checkpoint_version": None,
            "events_digest_hex": None,
            "event_count": None,
            "limit": 1,
            "offset": 1,
        },
    )
    revocation_impact_cursor = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.REVOCATION_IMPACTS_CURSOR,
        {
            "attestation_id": None,
            "revoker_actor_id": None,
            "reason": None,
            "from": None,
            "to": None,
            "limit": 1,
            "offset": 1,
        },
    )
    for token in (lineage_cursor, checkpoint_cursor, revocation_impact_cursor):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


def test_foreign_family_cursor_signed_with_this_secret_is_rejected(client, app):
    # Family separation is by format marker, not by secret: a
    # checkpoint-imports cursor minted under this endpoint's own secret is
    # still rejected.
    _setup_grouped(client)
    foreign = pagination.encode_typed_cursor(
        app.state.impact_imports_cursor_secret,
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


def test_impact_imports_cursor_is_rejected_by_other_entries(client):
    _setup_grouped(client)
    cursor = _list(client, limit=1).json()["next_cursor"]

    # The audit checkpoint-imports search uses its own family.
    resp = client.get(
        "/v1/audit-events/checkpoint-imports",
        params={"limit": 1, "cursor": cursor},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    # As does the revocation-impact search.
    resp = client.get(
        "/v1/revocation-impacts", params={"limit": 1, "cursor": cursor}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_bound_to_every_effective_query_parameter(client):
    receipts = _setup_grouped(client)
    digest = receipts[0]["impacts_digest_hex"]

    cursor = _list(client, limit=2).json()["next_cursor"]
    # A filter/limit present in the resume request but absent from the
    # cursor mismatches.
    for params in (
        {"limit": 3},
        {"checkpoint_version": CHECKPOINT_VERSION},
        {"impacts_digest_hex": digest},
        {"impact_count": 2},
    ):
        resp = _list(client, cursor=cursor, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"

    # A cursor minted under a filter cannot be resumed without it or with a
    # different value: no filter may be dropped or substituted.
    versioned = _list(
        client, checkpoint_version=CHECKPOINT_VERSION, limit=2
    ).json()["next_cursor"]
    assert _list(client, limit=2, cursor=versioned).status_code == 422
    assert (
        _list(
            client,
            checkpoint_version="provenance-revocation-impact-checkpoint-v2",
            limit=2,
            cursor=versioned,
        ).status_code
        == 422
    )

    digested = _list(client, impacts_digest_hex=digest, limit=2).json()[
        "next_cursor"
    ]
    assert _list(client, limit=2, cursor=digested).status_code == 422

    counted = _list(client, impact_count=2, limit=2).json()["next_cursor"]
    assert _list(client, limit=2, cursor=counted).status_code == 422
    assert (
        _list(client, impact_count=3, limit=2, cursor=counted).status_code
        == 422
    )


def test_cursor_with_bad_claim_types_is_422(client, app):
    _setup_grouped(client)
    # A structurally signed cursor whose claims violate the family's claim
    # set is rejected rather than trusted.
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "checkpoint_version": 5,
                "impacts_digest_hex": None,
                "impact_count": None,
                "limit": 1,
                "offset": 1,
            }
        ).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(
        hmac_mod.new(
            app.state.impact_imports_cursor_secret,
            f"ii1.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    resp = _list(client, cursor=f"ii1.{payload}.{sig}")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_secret_rotation_invalidates_outstanding_cursors(
    client, tmp_db_url
):
    _import(client, "restart")
    cursor = _list(client, limit=1).json()["next_cursor"]
    client.app.state.impact_imports_cursor_secret = secrets.token_bytes(32)
    assert _list(client, limit=1, cursor=cursor).status_code == 422

    first_app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(first_app) as first_client:
        receipt = _import(first_client, "persisted")
        token = _list(first_client, limit=1).json()["next_cursor"]
    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as second_client:
        stale = _list(second_client, limit=1, cursor=token)
        assert stale.status_code == 422
        fresh = _list(second_client).json()
        assert [i["id"] for i in fresh["items"]] == [receipt["id"]]


# --- Parameter validation --------------------------------------------------------


def test_blank_filters_are_validation_errors(client):
    _setup_grouped(client)
    for field in ("checkpoint_version", "impacts_digest_hex", "impact_count"):
        for value in ("", "   ", "\t"):
            resp = _list(client, **{field: value})
            assert resp.status_code == 422, (field, value)
            assert resp.json()["error"]["code"] == "validation_error"


def test_illegal_impact_count_values_are_validation_errors(client):
    _setup_grouped(client)
    for value in ("-1", "+1", "1.5", "2.0", "abc", "  2", "2 ", "1e3", "0x2"):
        resp = _list(client, impact_count=value)
        assert resp.status_code == 422, value
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
        "checkpoint_version=a&checkpoint_version=b",
        "impacts_digest_hex=a&impacts_digest_hex=b",
        "impact_count=1&impact_count=2",
        "limit=1&limit=2",
        "cursor=x&cursor=y",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameters_are_validation_errors(client):
    _setup_grouped(client)
    for suffix in (
        "import_id=rii_x",
        "checkpoint_versions=" + CHECKPOINT_VERSION,
        "digest_hex=" + "0" * 64,
        "impact_counts=2",
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
        checkpoint_version=CHECKPOINT_VERSION,
        impact_count=2,
        limit=10,
        cursor="garbage",
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_any_request_body_is_422_before_reading_receipts(client, db_session):
    _setup_grouped(client)
    for body in (b"{}", b" ", b"\n", b"not json", b"[]"):
        resp = client.request("GET", URL, content=body)
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"

    # Validation precedes the receipt read and writes nothing.
    assert (
        db_session.execute(
            select(func.count()).select_from(ImpactImportRecord)
        ).scalar_one()
        == 5
    )


# --- Method handling -------------------------------------------------------------


def test_non_get_methods_are_405(client):
    _setup_grouped(client)
    for method in ("put", "patch", "delete"):
        resp = getattr(client, method)(URL)
        assert resp.status_code == 405, resp.text
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Read-only guarantee ----------------------------------------------------------


def test_queries_and_failures_write_no_rows_or_audit_events(client, db_session):
    _setup_grouped(client)

    def counts():
        return (
            db_session.scalar(
                select(func.count()).select_from(ImpactImportRecord)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    receipts_before, events_before = counts()
    assert receipts_before == 5
    assert events_before == 5

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
    _list(client, checkpoint_version=CHECKPOINT_VERSION)
    _list(client, impacts_digest_hex="0" * 64)
    _list(client, impact_count=2)
    _list(client, impact_count=0)

    _list(client, checkpoint_version=" ")
    _list(client, impact_count="-1")
    _list(client, limit=0)
    _list(client, limit=101)
    _list(client, cursor="tampered")
    _list(client, limit=1, cursor="ri1.x.y")
    client.get(f"{URL}?limit=1&limit=2")
    client.get(f"{URL}?unknown=1")
    _list(client, impact_count=2, cursor="garbage")
    client.request("GET", URL, content=b"{}")

    db_session.expire_all()
    assert counts() == (5, 5)


# --- Compatibility with the existing impact-import routes -------------------------


def test_existing_post_single_get_and_recon_remain_unchanged(client):
    created = _import(client, "compat")
    single = client.get(f"{URL}/{created['id']}")
    assert single.status_code == 200
    assert single.json() == created

    recon = client.get(f"{URL}/{created['id']}/recon")
    assert recon.status_code == 200
    assert recon.json()["import_id"] == created["id"]
    assert recon.json()["matches"] is False

    missing = client.get(f"{URL}/rii_{'0' * 64}")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "impact_import_not_found"


def test_import_creation_and_export_verification_remain_unchanged(client):
    # The new search never materializes described impacts: importing an
    # offline checkpoint and the local package export stay consistent.
    request = _request("offline", 2)
    assert client.post("/v1/impact-verifications", json=request).json() == {
        "valid": True
    }
    assert client.post(URL, json=request).status_code == 201

    package = client.get("/v1/revocation-impact-package")
    assert package.status_code == 200
    assert package.json()["impacts"] == []
