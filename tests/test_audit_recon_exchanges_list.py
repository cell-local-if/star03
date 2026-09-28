"""Tests for the read-only audit recon exchange receipt retrieval.

Covers GET /v1/audit-recon-exchanges: receipts follow stable creation
order (``received_at`` with the persistent insertion-order tiebreaker
that survives a restart) and each item is exactly the existing
single-receipt public view (``id``, ``signature_version``,
``signer_subject``, ``public_key``, ``package_digest_hex``,
``signature_digest_hex``, UTC ``received_at``); the package, raw
signature, and private keys never appear. ``signer_subject``,
``public_key`` (the exact standard-Base64 spelling, never decoded or
normalized and never reverse-resolved by digest), and
``package_digest_hex`` are non-empty, case- and whitespace-sensitive
exact matches; ``from``/``to`` are strict RFC 3339 UTC inclusive bounds
on ``received_at`` with ``from`` never later than ``to``. All filters
combine as logical AND; an unknown value is an empty collection, never
an error. ``count`` is the filtered total covering every page;
``limit`` is a pure decimal integer in 1..100 defaulting to 50; the
opaque HMAC cursor (its own ``arx1`` family) binds every effective
filter and the limit, so pages concatenate without gaps or duplicates
and the final cursor is null. Any GET body bytes, a blank/illegal/
repeated/undeclared parameter (``signature_version`` is not accepted on
this route), and a blank/malformed/tampered/foreign-family/
query-mismatching cursor are all 422 validation_error, validated before
any receipt is read; non-GET methods other than the unchanged import
POST are 405 method_not_allowed. Queries and failures write no receipt
and no audit row. All fixtures are deterministic and offline.
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
from provenance.models import AuditEvent, AuditReconExchangeImportRecord
from tests.helpers import SEED_A, SEED_B
from tests.test_audit_recon_exchanges import (
    POST_PATH,
    RECEIPT_FIELDS,
    RECEIPT_ORDER,
    _entries,
    _package,
    _request,
)

LIST_PATH = POST_PATH
EVENT_TYPE = "audit.recon_imported"


# --- Fixture construction ----------------------------------------------------


def _import(client, n, *, subject="remote-audit", seed=SEED_A):
    """Register one distinct signed recon package; return its receipt."""
    request, _ = _request(_package(_entries(n)), subject=subject, seed=seed)
    resp = client.post(POST_PATH, json=request)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _three_receipts(client):
    """Three distinct receipts: two under subject/key A, one under B."""
    r1 = _import(client, 1)
    r2 = _import(client, 2)
    r3 = _import(client, 3, subject="other-subject", seed=SEED_B)
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
            select(AuditReconExchangeImportRecord).where(
                AuditReconExchangeImportRecord.id == receipt_id
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
    assert [item["id"] for item in body["items"]] == [
        r1["id"],
        r2["id"],
        r3["id"],
    ]


def test_items_are_exactly_the_receipt_public_view(client):
    r1, _, _ = _three_receipts(client)
    body = _list(client).json()
    for item in body["items"]:
        assert set(item) == RECEIPT_FIELDS
        assert list(item) == RECEIPT_ORDER
        assert item["received_at"].endswith(("Z", "+00:00"))
    detail = client.get(f"{LIST_PATH}/{r1['id']}").json()
    assert body["items"][0] == detail
    # Raw material is never echoed on the collection route.
    rendered = _list(client).text
    assert "entries" not in rendered
    assert "checkpoint" not in rendered
    assert '"signature"' not in rendered


def test_response_is_compact_utf8_json_with_one_newline(client):
    _three_receipts(client)
    resp = _list(client)
    raw = resp.content
    assert resp.headers["content-type"].startswith("application/json")
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    text = raw.decode("utf-8")[:-1]
    assert '", "' not in text and '": ' not in text
    assert list(resp.json()) == ["items", "count", "next_cursor"]


def test_same_instant_tiebreaker_is_insertion_order_and_survives_restart(
    tmp_db_url,
):
    app_one = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app_one) as client_one:
        r1 = _import(client_one, 1)
        r2 = _import(client_one, 2)
        session = app_one.state.session_factory()
        try:
            same = datetime(2026, 3, 1, 9, 0, 0, tzinfo=timezone.utc)
            _stamp(session, {r1["id"]: same, r2["id"]: same})
        finally:
            session.close()
    # Across a restart the same-instant order stays the insertion order.
    app_two = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app_two) as client_two:
        body = _list(client_two).json()
        assert [item["id"] for item in body["items"]] == [r1["id"], r2["id"]]


# --- Exact-match filters -------------------------------------------------------


def test_filter_by_signer_subject(client):
    _, _, r3 = _three_receipts(client)
    body = _list(client, signer_subject="remote-audit").json()
    assert body["count"] == 2
    assert all(item["signer_subject"] == "remote-audit" for item in body["items"])
    body = _list(client, signer_subject="other-subject").json()
    assert [item["id"] for item in body["items"]] == [r3["id"]]
    # Case- and whitespace-sensitive: near-miss spellings match nothing.
    assert _list(client, signer_subject="REMOTE-AUDIT").json()["count"] == 0
    assert _list(client, signer_subject="remote-audit ").json()["count"] == 0
    assert _list(client, signer_subject=" remote-audit").json()["count"] == 0
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
    assert (
        _list(client, public_key=r1["public_key"].rstrip("=")).json()["count"]
        == 0
    )
    assert (
        _list(client, public_key=r1["public_key"].swapcase()).json()["count"]
        == 0
    )
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
        _list(
            client, package_digest_hex=r1["package_digest_hex"].upper()
        ).json()["count"]
        == 0
    )


def test_filters_combine_as_logical_and(client):
    r1, _, r3 = _three_receipts(client)
    body = _list(
        client,
        signer_subject="remote-audit",
        public_key=r1["public_key"],
        package_digest_hex=r1["package_digest_hex"],
    ).json()
    assert [item["id"] for item in body["items"]] == [r1["id"]]
    # One contradictory conjunct empties the set.
    body = _list(
        client,
        signer_subject="remote-audit",
        package_digest_hex=r3["package_digest_hex"],
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


def test_signature_version_is_not_an_accepted_parameter(client):
    _three_receipts(client)
    _assert_validation_error(
        _list(client, signature_version="acr-exchange-v1")
    )


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
    ["signer_subject", "public_key", "package_digest_hex", "from", "to", "limit"],
)
def test_blank_values_are_422(client, field):
    _assert_validation_error(_list(client, **{field: ""}))
    _assert_validation_error(_list(client, **{field: "   "}))


@pytest.mark.parametrize(
    "query",
    [
        "unknown=1",
        "signer_subjects=x",
        "id=arx_x",
        "limit=1&limit=2",
        "cursor=a&cursor=b",
        "signer_subject=a&signer_subject=b",
        "public_key=a&public_key=b",
        "package_digest_hex=a&package_digest_hex=b",
        "from=2026-01-01T00:00:00Z&from=2026-01-02T00:00:00Z",
        "to=2026-01-01T00:00:00Z&to=2026-01-02T00:00:00Z",
    ],
)
def test_repeated_or_undeclared_parameters_are_422(client, query):
    resp = client.get(f"{LIST_PATH}?{query}")
    _assert_validation_error(resp)


@pytest.mark.parametrize(
    "body",
    [b"{}", b"null", b"[1]", b'"s"', b"1", b"not json", b"{", b" ", b"\n", b"\x00\xff"],
)
def test_any_get_body_bytes_are_422(client, body):
    resp = client.request("GET", LIST_PATH, content=body)
    _assert_validation_error(resp)


def test_body_is_validated_before_query_parameters(client):
    # Both a non-empty body and an unknown query parameter: the empty-body
    # rule fails first and is still a 422 validation_error.
    resp = client.request(
        "GET", f"{LIST_PATH}?unknown=1", content=b"x"
    )
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
    items, pages, count = _walk_pages(
        client, signer_subject="remote-audit", limit=1
    )
    assert count == 2
    assert [item["signer_subject"] for item in items] == ["remote-audit"] * 2


def test_first_page_carries_no_cursor_input_and_final_page_cursor_is_null(
    client,
):
    _three_receipts(client)
    first = _list(client, limit=1).json()
    assert first["next_cursor"].startswith("arx1.")
    last = _walk_pages(client, limit=1)[1][-1]
    assert len(last) == 1


def test_reusing_a_cursor_replays_the_same_page(client):
    _three_receipts(client)
    cursor = _list(client, limit=2).json()["next_cursor"]
    assert _list(client, limit=2, cursor=cursor).json() == _list(
        client, limit=2, cursor=cursor
    ).json()


def test_cursor_past_end_returns_empty_page_with_total_count(client, app):
    _three_receipts(client)
    token = pagination.encode_typed_cursor(
        app.state.audit_recon_exchange_imports_cursor_secret,
        pagination.AUDIT_RECON_EXCHANGE_IMPORTS_CURSOR,
        {
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
    r1, _, _ = _three_receipts(client)
    cursor = _list(client, limit=1).json()["next_cursor"]
    assert _list(client, limit=1, cursor=cursor).status_code == 200
    # A different limit (including the defaulted one) or any added,
    # dropped, or changed filter is a 422, not a new query.
    _assert_validation_error(_list(client, limit=2, cursor=cursor))
    _assert_validation_error(_list(client, cursor=cursor))
    _assert_validation_error(
        _list(client, limit=1, signer_subject="remote-audit", cursor=cursor)
    )
    _assert_validation_error(
        _list(client, limit=1, public_key=r1["public_key"], cursor=cursor)
    )
    _assert_validation_error(
        _list(
            client, limit=1, **{"from": "2026-01-01T00:00:00Z"}, cursor=cursor
        )
    )

    filtered = _list(client, limit=1, signer_subject="remote-audit").json()
    assert _list(
        client,
        limit=1,
        signer_subject="remote-audit",
        cursor=filtered["next_cursor"],
    ).status_code == 200
    _assert_validation_error(
        _list(
            client,
            limit=1,
            signer_subject="other-subject",
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
    first = _list(
        client, limit=1, **{"from": "2026-02-01T10:00:00Z"}
    ).json()
    # The same instant spelled "+00:00" resumes the "Z"-minted cursor.
    resp = _list(
        client,
        limit=1,
        **{"from": "2026-02-01T10:00:00+00:00"},
        cursor=first["next_cursor"],
    )
    assert resp.status_code == 200, resp.text
    # A genuinely different instant does not.
    _assert_validation_error(
        _list(
            client,
            limit=1,
            **{"from": "2026-02-01T10:00:01Z"},
            cursor=first["next_cursor"],
        )
    )


def test_tampered_or_malformed_cursors_are_validation_errors(client):
    _three_receipts(client)
    good = _list(client, limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    # A foreign-family token even when correctly signed for its family.
    foreign_ax = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.AUDIT_EXCHANGE_IMPORTS_CURSOR,
        {
            "signature_version": None,
            "signer_subject": None,
            "public_key": None,
            "package_digest_hex": None,
            "limit": 1,
            "offset": 1,
        },
    )
    foreign_rx = pagination.encode_typed_cursor(
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
        "arx1.onlytwoparts",
        "arx1.too.many.parts",
        "arx0.x.y",
        "arx2.x.y",
        "ax1.x.y",
        "rx1.x.y",
        "v1.x.y",
        tampered,
        foreign_ax,
        foreign_rx,
    ):
        _assert_validation_error(_list(client, limit=1, cursor=token))


def test_cursor_signed_with_old_format_marker_is_rejected(client, app):
    _three_receipts(client)
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
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
            app.state.audit_recon_exchange_imports_cursor_secret,
            f"arx0.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    _assert_validation_error(_list(client, cursor=f"arx0.{payload}.{sig}"))


def test_foreign_family_cursor_signed_with_this_secret_is_rejected(client, app):
    # Family separation is by format marker, not by secret: an
    # audit-exchanges (ax1) cursor minted under this endpoint's own secret
    # is still rejected.
    _three_receipts(client)
    foreign = pagination.encode_typed_cursor(
        app.state.audit_recon_exchange_imports_cursor_secret,
        pagination.AUDIT_EXCHANGE_IMPORTS_CURSOR,
        {
            "signature_version": None,
            "signer_subject": None,
            "public_key": None,
            "package_digest_hex": None,
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


# --- Method handling and read-only guarantees ------------------------------------


@pytest.mark.parametrize("method", ("put", "patch", "delete"))
def test_non_get_post_methods_on_collection_are_405(client, method):
    response = getattr(client, method)(LIST_PATH)
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"


def test_post_registration_on_the_collection_path_is_unchanged(client):
    request, _ = _request(_package(_entries(1)))
    response = client.post(POST_PATH, json=request)
    assert response.status_code == 201, response.text
    # The new receipt is immediately visible to the collection read.
    listed = _list(client).json()
    assert listed["count"] == 1
    assert listed["items"][0] == response.json()


def test_unknown_detail_id_still_404(client):
    import_id = "arx_" + "00" * 32
    response = client.get(f"{LIST_PATH}/{import_id}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "audit_recon_not_found"


def test_queries_and_failures_write_nothing(client, db_session):
    _three_receipts(client)
    audits_before = db_session.scalar(select(func.count()).select_from(AuditEvent))
    receipts_before = db_session.scalar(
        select(func.count()).select_from(AuditReconExchangeImportRecord)
    )
    assert _list(client).status_code == 200
    _walk_pages(client, limit=1)
    _list(client, signer_subject="no-such-subject")
    _list(client, public_key="not-base64!!")
    _list(client, package_digest_hex="00" * 32)
    _list(
        client, **{"from": "2027-01-01T00:00:00Z", "to": "2027-02-01T00:00:00Z"}
    )
    _list(client, limit=0)
    _list(client, unknown="1")
    client.request("GET", LIST_PATH, content=b"{}")
    _list(client, cursor="not-a-cursor")
    db_session.expire_all()
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audits_before
    )
    assert (
        db_session.scalar(
            select(func.count()).select_from(AuditReconExchangeImportRecord)
        )
        == receipts_before
    )
    # Only the three import events exist; the reads added no event type.
    assert (
        db_session.scalar(
            select(func.count())
            .select_from(AuditEvent)
            .where(AuditEvent.event_type == EVENT_TYPE)
        )
        == 3
    )
