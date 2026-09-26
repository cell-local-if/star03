"""Tests for the read-only signed exchange-import receipt retrieval.

Covers GET /v1/impact-recon-exchange-imports: receipts follow stable
creation order and each item is exactly the existing single-receipt public
view (``id``, ``signature_version``, ``signer_subject``, ``public_key``,
``package_digest_hex``, ``signature_digest_hex``, UTC ``received_at``).
``signature_version``, ``signer_subject``, ``public_key`` (the exact
Base64 spelling), and ``package_digest_hex`` are non-empty, case- and
whitespace-sensitive exact matches; ``from``/``to`` are strict RFC 3339
UTC bounds, both inclusive, with ``from`` never later than ``to``. All
filters combine as logical AND; an unknown value is an empty collection,
never an error. ``count`` is the filtered total covering every page;
``limit`` is a pure decimal integer in 1..100 defaulting to 50; the
opaque HMAC cursor (its own ``rx1`` family) binds every effective filter
and the limit, so pages concatenate without gaps or duplicates and the
final cursor is null. A non-empty GET body, a blank/illegal/repeated/
undeclared parameter, and a blank/malformed/tampered/foreign-family/
query-mismatching cursor are all 422 validation_error; non-GET methods
(other than the import POST) are 405 method_not_allowed. Queries and
failures write no receipt and no audit row. All fixtures are
deterministic and offline.
"""

from __future__ import annotations

import base64
import hashlib
import hmac as hmac_mod
import json
import secrets
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance import pagination
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import AuditEvent, ImpactReconExchangeImportRecord
from tests.helpers import SEED_A, SEED_B
from tests.test_impact_imports import _offline_request
from tests.test_impact_recon_exchange_imports import (
    POST_PATH,
    RECEIPT_FIELDS,
    RECEIPT_ORDER,
    SIGNATURE_VERSION,
    _request,
)

LIST_PATH = "/v1/impact-recon-exchange-imports"
IMPACT_IMPORTS_PATH = "/v1/impact-imports"
RECON_PACKAGE_PATH = "/v1/impact-recon-package"


# --- Fixture construction ----------------------------------------------------


def _impact(label: str) -> dict:
    """One deterministic, self-contained offline impact."""
    return {
        "id": "rev_" + hashlib.sha256(f"{label}-rev".encode()).hexdigest(),
        "attestation_id": "att_"
        + hashlib.sha256(f"{label}-att".encode()).hexdigest(),
        "revoker_actor_id": "org-revoker-1",
        "reason": f"reason {label}",
        "created_at": "2026-01-02T03:04:05Z",
        "content_id": "cnt_"
        + hashlib.sha256(f"{label}-cnt".encode()).hexdigest(),
        "target_type": "claim",
        "signer_actor_id": "org-signer-1",
        "qualified_signer_count_after": 0,
        "coverage_status_after": "partial",
        "qualified_signer_count_before": 1,
        "coverage_status_before": "covered",
        "qualified_signer_count_delta": 1,
    }


def _grow_and_import(client, label, *, subject="remote-system", seed=SEED_A):
    """Grow local recon state, then exchange-import the new package.

    Each distinct label yields a distinct local recon package and hence a
    distinct receipt; the receipt JSON is returned.
    """
    resp = client.post(IMPACT_IMPORTS_PATH, json=_offline_request([_impact(label)]))
    assert resp.status_code in (200, 201), resp.text
    package = client.get(RECON_PACKAGE_PATH).json()
    request, _ = _request(package, subject=subject, seed=seed)
    resp = client.post(POST_PATH, json=request)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _three_receipts(client):
    """Three distinct receipts: two under subject/key A, one under B."""
    r1 = _grow_and_import(client, "one")
    r2 = _grow_and_import(client, "two")
    r3 = _grow_and_import(client, "three", subject="other-subject", seed=SEED_B)
    return r1, r2, r3


def _list(client, **params):
    return client.get(LIST_PATH, params=params)


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


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"
    # A failed query never returns partial results.
    assert "items" not in resp.json()


def _stamp(db_session, instants_by_id: dict) -> None:
    """Overwrite created_at per receipt id with a fixed UTC instant."""
    for receipt_id, instant in instants_by_id.items():
        row = db_session.execute(
            select(ImpactReconExchangeImportRecord).where(
                ImpactReconExchangeImportRecord.id == receipt_id
            )
        ).scalar_one()
        row.created_at = instant
    db_session.commit()


# --- Baseline listing ---------------------------------------------------------


def test_empty_collection_returns_zero_count(client):
    resp = _list(client)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_receipts_follow_stable_creation_order(client):
    r1, r2, r3 = _three_receipts(client)
    body = _list(client).json()
    assert body["count"] == 3
    assert body["next_cursor"] is None
    assert [item["id"] for item in body["items"]] == [r1["id"], r2["id"], r3["id"]]


def test_items_are_exactly_the_receipt_public_view(client):
    r1, _, _ = _three_receipts(client)
    body = _list(client).json()
    for item in body["items"]:
        assert set(item) == RECEIPT_FIELDS
        assert list(item) == RECEIPT_ORDER
        assert item["received_at"].endswith(("Z", "+00:00"))
    detail = client.get(f"{LIST_PATH}/{r1['id']}").json()
    assert body["items"][0] == detail


def test_response_is_compact_utf8_json_with_one_newline(client):
    _three_receipts(client)
    resp = _list(client)
    raw = resp.content
    assert resp.headers["content-type"].startswith("application/json")
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    text = raw.decode("utf-8")[:-1]
    assert '", "' not in text and '": ' not in text
    assert list(resp.json()) == ["items", "count", "next_cursor"]


# --- Exact-match filters -------------------------------------------------------


def test_filter_by_signature_version(client):
    _three_receipts(client)
    body = _list(client, signature_version=SIGNATURE_VERSION).json()
    assert body["count"] == 3
    assert _list(client, signature_version="no-such-version").json()["count"] == 0
    # Case- and whitespace-sensitive: near-miss spellings match nothing.
    assert (
        _list(client, signature_version=SIGNATURE_VERSION.upper()).json()["count"]
        == 0
    )
    assert (
        _list(client, signature_version=f" {SIGNATURE_VERSION}").json()["count"]
        == 0
    )


def test_filter_by_signer_subject(client):
    _, _, r3 = _three_receipts(client)
    body = _list(client, signer_subject="remote-system").json()
    assert body["count"] == 2
    assert all(item["signer_subject"] == "remote-system" for item in body["items"])
    body = _list(client, signer_subject="other-subject").json()
    assert [item["id"] for item in body["items"]] == [r3["id"]]
    assert _list(client, signer_subject="REMOTE-SYSTEM").json()["count"] == 0
    assert _list(client, signer_subject="remote-system ").json()["count"] == 0
    assert _list(client, signer_subject="unknown-subject").json()["count"] == 0


def test_filter_by_public_key(client):
    r1, _, r3 = _three_receipts(client)
    body = _list(client, public_key=r1["public_key"]).json()
    assert body["count"] == 2
    assert all(item["public_key"] == r1["public_key"] for item in body["items"])
    body = _list(client, public_key=r3["public_key"]).json()
    assert [item["id"] for item in body["items"]] == [r3["id"]]
    # Non-canonical or foreign Base64 spellings are plain non-matching
    # text (an empty collection), never a 422 and never decoded.
    assert _list(client, public_key=r1["public_key"].rstrip("=")).json()["count"] == 0
    assert _list(client, public_key=r1["public_key"].swapcase()).json()["count"] == 0
    assert _list(client, public_key="not-base64!!").json()["count"] == 0


def test_filter_by_package_digest_hex(client):
    r1, _, _ = _three_receipts(client)
    body = _list(client, package_digest_hex=r1["package_digest_hex"]).json()
    assert [item["id"] for item in body["items"]] == [r1["id"]]
    assert body["count"] == 1
    unknown = hashlib.sha256(b"no-such-package").hexdigest()
    assert _list(client, package_digest_hex=unknown).json()["count"] == 0
    # Case-sensitive: the uppercased digest matches nothing.
    assert (
        _list(client, package_digest_hex=r1["package_digest_hex"].upper()).json()[
            "count"
        ]
        == 0
    )


def test_filters_combine_as_logical_and(client):
    r1, _, r3 = _three_receipts(client)
    body = _list(
        client,
        signature_version=SIGNATURE_VERSION,
        signer_subject="remote-system",
        public_key=r1["public_key"],
        package_digest_hex=r1["package_digest_hex"],
    ).json()
    assert [item["id"] for item in body["items"]] == [r1["id"]]
    # One contradictory conjunct empties the set.
    body = _list(
        client,
        signer_subject="remote-system",
        package_digest_hex=r3["package_digest_hex"],
    ).json()
    assert body["count"] == 0 and body["items"] == []


# --- Time filters --------------------------------------------------------------


def test_time_filters_are_inclusive(client, db_session):
    r1, r2, r3 = _three_receipts(client)
    _stamp(
        db_session,
        {
            r1["id"]: datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc),
            r2["id"]: datetime(2026, 2, 1, 11, 0, 0, tzinfo=timezone.utc),
            r3["id"]: datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc),
        },
    )
    # An instant equal to both bounds is included (boundaries inclusive).
    body = _list(
        client, **{"from": "2026-02-01T11:00:00Z", "to": "2026-02-01T11:00:00Z"}
    ).json()
    assert [item["id"] for item in body["items"]] == [r2["id"]]
    assert body["count"] == 1
    body = _list(client, **{"from": "2026-02-01T11:00:00Z"}).json()
    assert [item["id"] for item in body["items"]] == [r2["id"], r3["id"]]
    body = _list(client, to="2026-02-01T11:00:00Z").json()
    assert [item["id"] for item in body["items"]] == [r1["id"], r2["id"]]
    body = _list(
        client, **{"from": "2026-02-01T10:00:00Z", "to": "2026-02-01T12:00:00Z"}
    ).json()
    assert body["count"] == 3
    # The "+00:00" spelling is the same instant as "Z".
    body = _list(client, **{"from": "2026-02-01T11:00:00+00:00"}).json()
    assert [item["id"] for item in body["items"]] == [r2["id"], r3["id"]]
    # A range matching nothing is an empty collection with a zero total.
    body = _list(
        client, **{"from": "2027-01-01T00:00:00Z", "to": "2027-01-02T00:00:00Z"}
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


def test_from_later_than_to_is_422(client):
    resp = _list(
        client, **{"from": "2026-02-02T00:00:00Z", "to": "2026-02-01T00:00:00Z"}
    )
    _assert_validation_error(resp)


@pytest.mark.parametrize(
    "value",
    [
        "2026-01-01",
        "2026-01-01T00:00:00",
        "2026-01-01 00:00:00Z",
        "2026-01-01T00:00Z",
        "2026-01-01T00:00:00z",
        "2026-01-01T00:00:00+01:00",
        "2026-13-01T00:00:00Z",
        "not-a-time",
    ],
)
def test_malformed_time_filters_are_422(client, value):
    _assert_validation_error(_list(client, **{"from": value}))
    _assert_validation_error(_list(client, to=value))


# --- Limit and parameter validation --------------------------------------------


def test_limit_defaults_to_50(client):
    _three_receipts(client)
    body = _list(client).json()
    assert len(body["items"]) == 3 and body["count"] == 3


@pytest.mark.parametrize("value", [1, 2, 100])
def test_limit_boundaries_accepted(client, value):
    _three_receipts(client)
    resp = _list(client, limit=value)
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["items"]) == min(value, 3)


@pytest.mark.parametrize(
    "value", ["0", "101", "-1", "+1", "1.0", "1e2", "abc", " 1", "1 ", ""]
)
def test_bad_limits_are_422(client, value):
    _assert_validation_error(_list(client, limit=value))


@pytest.mark.parametrize(
    "field",
    [
        "signature_version",
        "signer_subject",
        "public_key",
        "package_digest_hex",
        "from",
        "to",
        "limit",
    ],
)
def test_blank_values_are_422(client, field):
    _assert_validation_error(_list(client, **{field: ""}))
    _assert_validation_error(_list(client, **{field: "   "}))


@pytest.mark.parametrize(
    "query",
    [
        "unknown=1",
        "signer_subjects=x",
        "limit=1&limit=2",
        "cursor=a&cursor=b",
        "signer_subject=a&signer_subject=b",
        "from=2026-01-01T00:00:00Z&from=2026-01-02T00:00:00Z",
        "to=2026-01-01T00:00:00Z&to=2026-01-02T00:00:00Z",
    ],
)
def test_repeated_or_undeclared_parameters_are_422(client, query):
    resp = client.get(f"{LIST_PATH}?{query}")
    _assert_validation_error(resp)


@pytest.mark.parametrize(
    "body",
    [b"{}", b"null", b"[1]", b'"s"', b"1", b"not json", b"{", b" ", b"\n"],
)
def test_non_empty_get_body_is_422(client, body):
    resp = client.request("GET", LIST_PATH, content=body)
    _assert_validation_error(resp)


# --- Pagination -----------------------------------------------------------------


def test_pages_concatenate_without_gaps_or_duplicates(client):
    r1, r2, r3 = _three_receipts(client)
    expected = [r1["id"], r2["id"], r3["id"]]
    for limit in (1, 2, 3, 100):
        items, pages, count = _walk_pages(client, limit=limit)
        assert [item["id"] for item in items] == expected
        assert count == 3
        assert all(len(page) <= limit for page in pages)


def test_count_covers_every_page(client):
    _three_receipts(client)
    first = _list(client, limit=1).json()
    assert first["count"] == 3 and len(first["items"]) == 1
    second = _list(client, limit=1, cursor=first["next_cursor"]).json()
    assert second["count"] == 3
    third = _list(client, limit=1, cursor=second["next_cursor"]).json()
    assert third["count"] == 3 and third["next_cursor"] is None


def test_filtered_pagination(client):
    _three_receipts(client)
    items, pages, count = _walk_pages(client, signer_subject="remote-system", limit=1)
    assert count == 2
    assert [item["signer_subject"] for item in items] == ["remote-system"] * 2


def test_reusing_a_cursor_replays_the_same_page(client):
    _three_receipts(client)
    cursor = _list(client, limit=2).json()["next_cursor"]
    assert _list(client, limit=2, cursor=cursor).json() == _list(
        client, limit=2, cursor=cursor
    ).json()


def test_cursor_past_end_returns_empty_page_with_total_count(client, app):
    _three_receipts(client)
    token = pagination.encode_typed_cursor(
        app.state.impact_recon_exchange_imports_cursor_secret,
        pagination.IMPACT_RECON_EXCHANGE_IMPORTS_CURSOR,
        {
            "signature_version": None,
            "signer_subject": None,
            "public_key": None,
            "package_digest_hex": None,
            "from": None,
            "to": None,
            "limit": 50,
            "offset": 99,
        },
    )
    body = _list(client, cursor=token).json()
    assert body["items"] == []
    assert body["count"] == 3
    assert body["next_cursor"] is None


# --- Cursor integrity -------------------------------------------------------------


def test_cursor_binds_every_filter_and_the_limit(client):
    _three_receipts(client)
    cursor = _list(client, limit=1).json()["next_cursor"]
    assert _list(client, limit=1, cursor=cursor).status_code == 200
    # A different limit (including the defaulted one) or any added,
    # dropped, or changed filter is a 422, not a new query.
    _assert_validation_error(_list(client, limit=2, cursor=cursor))
    _assert_validation_error(_list(client, cursor=cursor))
    _assert_validation_error(
        _list(client, limit=1, signer_subject="remote-system", cursor=cursor)
    )
    _assert_validation_error(
        _list(client, limit=1, **{"from": "2026-01-01T00:00:00Z"}, cursor=cursor)
    )

    filtered = _list(client, limit=1, signer_subject="remote-system").json()
    assert _list(
        client, limit=1, signer_subject="remote-system",
        cursor=filtered["next_cursor"],
    ).status_code == 200
    _assert_validation_error(
        _list(
            client, limit=1, signer_subject="other-subject",
            cursor=filtered["next_cursor"],
        )
    )


def test_cursor_time_claim_canonicalizes_utc_spellings(client, db_session):
    r1, r2, r3 = _three_receipts(client)
    _stamp(
        db_session,
        {
            r1["id"]: datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc),
            r2["id"]: datetime(2026, 2, 1, 11, 0, 0, tzinfo=timezone.utc),
            r3["id"]: datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc),
        },
    )
    first = _list(client, limit=1, **{"from": "2026-02-01T10:00:00Z"}).json()
    # The same instant spelled "+00:00" resumes the "Z"-minted cursor.
    resp = _list(
        client, limit=1,
        **{"from": "2026-02-01T10:00:00+00:00"},
        cursor=first["next_cursor"],
    )
    assert resp.status_code == 200, resp.text
    # A genuinely different instant does not.
    _assert_validation_error(
        _list(
            client, limit=1,
            **{"from": "2026-02-01T10:00:01Z"},
            cursor=first["next_cursor"],
        )
    )


def test_tampered_or_malformed_cursors_are_validation_errors(client):
    _three_receipts(client)
    good = _list(client, limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    foreign = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.IMPACT_RECON_EXCHANGE_IMPORTS_CURSOR,
        {
            "signature_version": None,
            "signer_subject": None,
            "public_key": None,
            "package_digest_hex": None,
            "from": None,
            "to": None,
            "limit": 1,
            "offset": 1,
        },
    )
    for token in (
        "",
        "   ",
        "not-a-cursor",
        "rx1.onlytwoparts",
        "rx1.too.many.parts",
        "rx0.x.y",
        "rx2.x.y",
        "v1.x.y",
        "ii1.x.y",
        "ei1.x.y",
        tampered,
        foreign,
    ):
        _assert_validation_error(_list(client, limit=1, cursor=token))


def test_cursor_signed_with_old_format_marker_is_rejected(client, app):
    _three_receipts(client)
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "signature_version": None,
                "signer_subject": None,
                "public_key": None,
                "package_digest_hex": None,
                "from": None,
                "to": None,
                "limit": 50,
                "offset": 1,
            }
        ).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(
        hmac_mod.new(
            app.state.impact_recon_exchange_imports_cursor_secret,
            f"rx0.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    _assert_validation_error(_list(client, cursor=f"rx0.{payload}.{sig}"))


def test_cursor_from_other_families_is_rejected(client):
    _three_receipts(client)
    impacts_cursor = pagination.encode_typed_cursor(
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
    for token in (impacts_cursor, exchange_cursor):
        _assert_validation_error(_list(client, limit=1, cursor=token))


def test_foreign_family_cursor_signed_with_this_secret_is_rejected(client, app):
    # Family separation is by format marker, not by secret: an
    # impact-imports cursor minted under this endpoint's own secret is
    # still rejected.
    _three_receipts(client)
    foreign = pagination.encode_typed_cursor(
        app.state.impact_recon_exchange_imports_cursor_secret,
        pagination.IMPACT_IMPORTS_CURSOR,
        {
            "checkpoint_version": None,
            "impacts_digest_hex": None,
            "impact_count": None,
            "limit": 1,
            "offset": 1,
        },
    )
    _assert_validation_error(_list(client, limit=1, cursor=foreign))


def test_cursor_secret_rotation_invalidates_outstanding_cursors(tmp_db_url):
    app_one = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app_one) as client_one:
        _three_receipts(client_one)
        cursor = _list(client_one, limit=1).json()["next_cursor"]
    # A restart rotates the per-process secret; the old cursor is a 422.
    app_two = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app_two) as client_two:
        _assert_validation_error(_list(client_two, limit=1, cursor=cursor))


# --- Read-only guarantees ---------------------------------------------------------


def test_queries_and_failures_write_nothing(client, db_session):
    _three_receipts(client)
    audits_before = db_session.scalar(select(func.count()).select_from(AuditEvent))
    receipts_before = db_session.scalar(
        select(func.count()).select_from(ImpactReconExchangeImportRecord)
    )
    assert _list(client).status_code == 200
    _walk_pages(client, limit=1)
    _list(client, signer_subject="no-such-subject")
    _list(client, limit=0)
    _list(client, unknown="1")
    client.request("GET", LIST_PATH, content=b"{}")
    _list(client, cursor="not-a-cursor")
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audits_before
    )
    assert (
        db_session.scalar(
            select(func.count()).select_from(ImpactReconExchangeImportRecord)
        )
        == receipts_before
    )
