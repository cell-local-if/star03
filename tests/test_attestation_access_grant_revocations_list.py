"""Tests for the global read-only access-grant-revocation search.

Covers ``GET /v1/attestation-access-grant-revocations``: the exact
``items``/``count``/``next_cursor`` member order in compact UTF-8 JSON
(no ASCII escaping) terminated by one newline, each item reusing the single
revocation public view (no raw signature, private key, payload, content, or
evidence byte), stable creation ordering (``created_at`` with the persistent
insertion-order ``seq`` tiebreaker that survives an app restart), the exact
``grant_id``/``revoker_actor_id``/``reason`` filters and the strict
inclusive RFC 3339 UTC ``from``/``to`` bounds (from must not be later than
to), unknown values yielding an empty collection, pure-decimal ``limit``
validation (1..100, default 50), the opaque HMAC-signed ``gr1`` cursor
family binding every effective filter and the limit, paging without
duplication or omission, a null cursor on the final page and an empty page
with the original count at or past the tail, every 422 validation boundary
(blank/bad-time/out-of-range/repeated/undeclared parameters, non-empty
bodies, blank/tampered/foreign-family/mismatched cursors), the 405
rejection of PUT/PATCH/DELETE, POST remaining the authenticated creation
route, no authentication headers on the search, and the strictly read-only
guarantee on success, empty results, and failures.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timezone

from sqlalchemy import func, select

from provenance import pagination
from provenance.models import (
    AttestationAccessGrantRevocation,
    AuditEvent,
)
from tests.test_attestation_access_grant_revocations import (
    GRANTS_PATH,
    REASON_A,
    REASON_B,
    REVOCATIONS_PATH,
    SEED_B,
    _grant,
    _make_attestation,
    _make_claim,
    _post_revocation,
    _revocation_body,
    _signed_headers,
    _world,
)

REASON_D = "offboarding completed"
REASON_UNICODE = "访问权限撤销 — λήμμα 🌐"

SEARCH_PATH = REVOCATIONS_PATH


# --- Fixture-style setup ------------------------------------------------------


def _grant_as(client, attestation_id, grantee_actor_id, *, actor, seed):
    """Create a grant signed by the attestation's actual signer."""
    body = json.dumps(
        {
            "attestation_id": attestation_id,
            "grantee_actor_id": grantee_actor_id,
        }
    ).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **_signed_headers(
            "POST", GRANTS_PATH, body, actor=actor, seed=seed
        ),
    }
    resp = client.post(GRANTS_PATH, content=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _revoke(client, grant, reason, *, actor="org-1", seed=None):
    kwargs = {}
    if actor != "org-1" or seed is not None:
        kwargs["actor"] = actor
        kwargs["seed"] = seed
    resp = _post_revocation(
        client, _revocation_body(grant["id"], reason), **kwargs
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _world_revocations(client):
    """Two signers, three grants, five interleaved revocations in order."""
    att_one = _world(client)
    # org-1's attestation: two grants to two grantees.
    g1 = _grant(client, att_one["id"], "org-2")
    g2 = _grant(client, att_one["id"], "org-3")

    # org-2 signs a second attestation on fresh content (the baseline world
    # already registered DIGEST_B) and grants read access to org-3.
    fresh_digest = hashlib.sha256(b"search-grant-revocations").hexdigest()
    att_two = _make_attestation(
        client,
        "org-2",
        SEED_B,
        _make_claim(client, "org-2", digest=fresh_digest)["id"],
    )
    g3 = _grant_as(
        client, att_two["id"], "org-3", actor="org-2", seed=SEED_B
    )

    r1 = _revoke(client, g1, REASON_A)
    r2 = _revoke(client, g1, REASON_B)
    r3 = _revoke(client, g2, REASON_A)
    r4 = _revoke(client, g3, REASON_A, actor="org-2", seed=SEED_B)
    r5 = _revoke(client, g3, REASON_D, actor="org-2", seed=SEED_B)
    return att_one, att_two, (g1, g2, g3), [r1, r2, r3, r4, r5]


def _pin_times(db_session, revocations, instants):
    """Overwrite created_at per revocation id with a fixed UTC instant."""
    by_id = {r["id"]: instant for r, instant in zip(revocations, instants)}
    for row in db_session.execute(
        select(AttestationAccessGrantRevocation)
    ).scalars():
        row.created_at = by_id[row.id]
    db_session.commit()


def _list(client, **params):
    return client.get(SEARCH_PATH, params=params)


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


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


# --- Empty collection and response shape --------------------------------------


def test_empty_store_is_an_empty_collection(client):
    resp = _list(client)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_search_requires_no_authentication_headers(client):
    _world_revocations(client)
    # A plain GET with no X-PA/X-PT/X-PS credentials answers the search.
    resp = client.get(SEARCH_PATH)
    assert resp.status_code == 200
    assert resp.json()["count"] == 5


def test_response_is_compact_json_with_ordered_members_and_one_newline(client):
    _world_revocations(client)
    resp = _list(client)
    assert resp.status_code == 200
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    assert raw.startswith(b'{"items":[')
    assert b'],"count":5,"next_cursor":null}' in raw
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        .encode("utf-8")
        + b"\n"
    )
    assert raw == expected


def test_non_ascii_text_is_utf8_not_ascii_escaped(client):
    _att_one, _att_two, grants, _revocations = _world_revocations(client)
    created = _revoke(client, grants[0], REASON_UNICODE)

    resp = _list(client, grant_id=grants[0]["id"])
    assert resp.status_code == 200
    raw = resp.content
    # Compact JSON with ensure_ascii=False: the reason bytes appear verbatim
    # and no ``\\u`` escape sequence is emitted anywhere.
    assert REASON_UNICODE.encode("utf-8") in raw
    assert b"\\u" not in raw
    assert created["reason"] == REASON_UNICODE
    assert any(
        item["reason"] == REASON_UNICODE for item in resp.json()["items"]
    )


def test_items_carry_exactly_the_revocation_public_view(client):
    _att_one, _att_two, _grants, revocations = _world_revocations(client)
    resp = _list(client)
    body = resp.json()
    assert body["count"] == 5
    assert body["next_cursor"] is None
    assert [item["id"] for item in body["items"]] == [
        r["id"] for r in revocations
    ]
    for item, revocation in zip(body["items"], revocations):
        assert list(item) == [
            "id",
            "grant_id",
            "revoker_actor_id",
            "reason",
            "created_at",
        ]
        assert item == revocation
        created_at = datetime.fromisoformat(item["created_at"])
        assert created_at.utcoffset().total_seconds() == 0
        assert item["created_at"].endswith(("Z", "+00:00"))
    # No signature, key, payload, content, evidence, or ordering column is
    # ever echoed.
    rendered = resp.content.decode()
    for forbidden in (
        "signature",
        "public_key",
        "private_key",
        "authorization",
        "payload",
        "content_bytes",
        "evidence",
        "seq",
    ):
        assert forbidden not in rendered


# --- Stable ordering -----------------------------------------------------------


def test_items_follow_stable_creation_order(client):
    _a, _t, _g, revocations = _world_revocations(client)
    body = _list(client).json()
    assert [item["id"] for item in body["items"]] == [
        r["id"] for r in revocations
    ]
    created = [datetime.fromisoformat(item["created_at"]) for item in body["items"]]
    assert created == sorted(created)


def test_same_timestamp_ties_break_by_persistence_order(client, db_session):
    _a, _t, _g, revocations = _world_revocations(client)
    tie = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    _pin_times(db_session, revocations, [tie] * 5)

    body = _list(client).json()
    assert [item["id"] for item in body["items"]] == [
        r["id"] for r in revocations
    ]


def test_ordering_is_identical_across_an_app_restart(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    _a, _t, _g, revocations = _world_revocations(file_client)
    first = _list(file_client)
    assert first.status_code == 200

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as restarted_client:
        second = _list(restarted_client)
        assert second.status_code == 200
        assert second.json() == first.json()
        assert [item["id"] for item in second.json()["items"]] == [
            r["id"] for r in revocations
        ]


# --- Exact-match filters --------------------------------------------------------


def test_grant_id_filter_is_an_exact_match(client):
    _a, _t, grants, revocations = _world_revocations(client)
    body = _list(client, grant_id=grants[0]["id"]).json()
    assert body["count"] == 2
    assert body["next_cursor"] is None
    assert [item["id"] for item in body["items"]] == [
        revocations[0]["id"],
        revocations[1]["id"],
    ]
    assert {item["grant_id"] for item in body["items"]} == {grants[0]["id"]}


def test_revoker_actor_id_filter_is_an_exact_match(client):
    _a, _t, _g, revocations = _world_revocations(client)
    org_one = _list(client, revoker_actor_id="org-1").json()
    assert org_one["count"] == 3
    assert [item["id"] for item in org_one["items"]] == [
        revocations[0]["id"],
        revocations[1]["id"],
        revocations[2]["id"],
    ]
    org_two = _list(client, revoker_actor_id="org-2").json()
    assert org_two["count"] == 2
    assert [item["id"] for item in org_two["items"]] == [
        revocations[3]["id"],
        revocations[4]["id"],
    ]


def test_reason_filter_is_an_exact_match(client):
    _a, _t, _g, revocations = _world_revocations(client)
    body = _list(client, reason=REASON_A).json()
    assert body["count"] == 3
    assert [item["id"] for item in body["items"]] == [
        revocations[0]["id"],
        revocations[2]["id"],
        revocations[3]["id"],
    ]


def test_filters_combine_as_logical_and(client):
    _a, _t, grants, revocations = _world_revocations(client)
    body = _list(
        client,
        grant_id=grants[1]["id"],
        revoker_actor_id="org-1",
        reason=REASON_A,
    ).json()
    assert body["count"] == 1
    assert [item["id"] for item in body["items"]] == [revocations[2]["id"]]

    miss = _list(
        client,
        grant_id=grants[1]["id"],
        revoker_actor_id="org-2",
        reason=REASON_A,
    ).json()
    assert miss == {"items": [], "count": 0, "next_cursor": None}


def test_each_filter_is_accepted_at_most_once(client):
    _world_revocations(client)
    for field, value in (
        ("grant_id", "aag_x"),
        ("revoker_actor_id", "org-1"),
        ("reason", REASON_A),
        ("from", "2026-01-01T00:00:00Z"),
        ("to", "2026-01-01T00:00:00Z"),
        ("limit", "1"),
        ("cursor", "x"),
    ):
        resp = client.get(
            SEARCH_PATH, params=[(field, value), (field, value)]
        )
        assert resp.status_code == 422, field
        assert resp.json()["error"]["code"] == "validation_error"


def test_filters_are_case_and_whitespace_sensitive(client):
    _world_revocations(client)
    for params in (
        {"grant_id": "AAG_X"},
        {"revoker_actor_id": "Org-1"},
        {"revoker_actor_id": " org-1"},
        {"revoker_actor_id": "org-1 "},
        {"reason": "ACCESS WITHDRAWN AFTER OFFBOARDING"},
        {"reason": f" {REASON_A}"},
        {"reason": f"{REASON_A}\t"},
    ):
        resp = _list(client, **params)
        assert resp.status_code == 200, params
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_unknown_or_nonexistent_filter_value_is_an_empty_collection(client):
    _world_revocations(client)
    for params in (
        {"grant_id": "aag_ghost"},
        {"revoker_actor_id": "ghost"},
        {"reason": "no such reason was ever recorded"},
        {
            "grant_id": "aag_ghost",
            "revoker_actor_id": "ghost",
            "reason": REASON_A,
        },
    ):
        resp = _list(client, **params)
        assert resp.status_code == 200, params
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_unknown_filter_parameters_do_not_404(client):
    # An undeclared parameter is a 422 validation_error, never a 404; a
    # declared-but-unmatched filter is an empty 200 collection.
    resp = _list(client, nonexistent="x")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_blank_filters_are_422(client):
    _world_revocations(client)
    for field in ("grant_id", "revoker_actor_id", "reason"):
        for blank in ("", "   ", "\t"):
            resp = _list(client, **{field: blank})
            assert resp.status_code == 422, (field, repr(blank))
            assert resp.json()["error"]["code"] == "validation_error"


# --- Time bounds ----------------------------------------------------------------


def test_from_bound_is_inclusive(client, db_session):
    _a, _t, _g, revocations = _world_revocations(client)
    _pin_times(
        db_session,
        revocations,
        [
            datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 3, 9, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 4, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 5, 6, 0, 0, tzinfo=timezone.utc),
        ],
    )
    body = _list(client, **{"from": "2026-01-02T12:30:00Z"}).json()
    assert [item["id"] for item in body["items"]] == [
        r["id"] for r in revocations[1:]
    ]
    assert body["count"] == 4


def test_to_bound_is_inclusive(client, db_session):
    _a, _t, _g, revocations = _world_revocations(client)
    _pin_times(
        db_session,
        revocations,
        [
            datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 3, 9, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 4, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 5, 6, 0, 0, tzinfo=timezone.utc),
        ],
    )
    body = _list(client, to="2026-01-02T12:30:00Z").json()
    assert [item["id"] for item in body["items"]] == [
        r["id"] for r in revocations[:2]
    ]
    assert body["count"] == 2


def test_equal_from_and_to_selects_exactly_that_instant(client, db_session):
    _a, _t, _g, revocations = _world_revocations(client)
    _pin_times(
        db_session,
        revocations,
        [
            datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 4, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 5, 6, 0, 0, tzinfo=timezone.utc),
        ],
    )
    body = _list(
        client, **{"from": "2026-01-02T12:30:00Z", "to": "2026-01-02T12:30:00Z"}
    ).json()
    assert [item["id"] for item in body["items"]] == [
        revocations[1]["id"],
        revocations[2]["id"],
    ]
    assert body["count"] == 2


def test_equivalent_utc_notations_are_the_same_bound(client, db_session):
    _a, _t, _g, revocations = _world_revocations(client)
    instant = datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc)
    _pin_times(
        db_session,
        revocations,
        [
            datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            instant,
            datetime(2026, 1, 3, 9, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 4, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 5, 6, 0, 0, tzinfo=timezone.utc),
        ],
    )
    zed = _list(client, **{"from": "2026-01-02T12:30:00Z"}).json()
    offset = _list(client, **{"from": "2026-01-02T12:30:00+00:00"}).json()
    assert zed == offset
    assert [item["id"] for item in zed["items"]] == [
        r["id"] for r in revocations[1:]
    ]


def test_bad_time_bounds_and_inverted_range_are_422(client):
    _world_revocations(client)
    for bad in (
        "2026-01-02",
        "2026-01-02T12:30:00",
        "2026-01-02 12:30:00Z",
        "2026-01-02T12:30Z",
        "2026-01-02T12:30:00+01:00",
        "2026-13-02T12:30:00Z",
        "2026-01-02T25:30:00Z",
        "2026-01-02t12:30:00z",
        "soon",
    ):
        resp = _list(client, **{"from": bad})
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error"
        resp = _list(client, to=bad)
        assert resp.status_code == 422, repr(bad)
    inverted = _list(
        client, **{"from": "2026-01-03T00:00:00Z", "to": "2026-01-02T00:00:00Z"}
    )
    assert inverted.status_code == 422
    assert inverted.json()["error"]["code"] == "validation_error"
    for blank in ("", "   "):
        assert _list(client, **{"from": blank}).status_code == 422
        assert _list(client, to=blank).status_code == 422


# --- limit validation -----------------------------------------------------------


def test_limit_boundaries_one_and_one_hundred_are_accepted(client):
    _world_revocations(client)
    assert _list(client, limit="1").status_code == 200
    assert _list(client, limit="100").status_code == 200


def test_limit_must_be_a_pure_decimal_integer_in_range(client):
    _world_revocations(client)
    for bad in ("0", "101", "-1", "5.0", "5e0", " 5", "5 ", "five", "+5", ""):
        resp = _list(client, limit=bad)
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_default_limit_is_fifty(client):
    att_one = _world(client)
    grant = _grant(client, att_one["id"], "org-2")
    for index in range(51):
        # Distinct reasons form independent immutable records; a repeated
        # triple would collapse to the original row.
        _revoke(client, grant, f"reason-{index:03d}")
    resp = _list(client, grant_id=grant["id"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 51
    assert len(body["items"]) == 50
    assert body["next_cursor"] is not None


# --- Parameter strictness -------------------------------------------------------


def test_undeclared_parameters_are_422(client):
    _world_revocations(client)
    for params in (
        {"id": "agr_1"},
        {"revoker": "org-1"},
        {"grantee_actor_id": "org-2"},
        {"offset": "1"},
        {"q": "x"},
        {"cursor": "x", "bogus": "y"},
    ):
        resp = _list(client, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_empty_body_is_422_rejected_before_any_read(client, db_session):
    _world_revocations(client)
    events_before = _audit_count(db_session)
    revocations_before = db_session.scalar(
        select(func.count()).select_from(AttestationAccessGrantRevocation)
    )
    for body in (b"{}", b" ", b"{not valid json", b"[]", b"\n", b"\x00\xff"):
        resp = client.request(
            "GET",
            SEARCH_PATH,
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"
    assert _audit_count(db_session) == events_before
    assert (
        db_session.scalar(
            select(func.count()).select_from(AttestationAccessGrantRevocation)
        )
        == revocations_before
    )


# --- Pagination -----------------------------------------------------------------


def test_pages_resume_without_duplication_or_omission(client):
    _a, _t, _g, revocations = _world_revocations(client)
    all_items, pages, count = _walk_pages(client, limit="2")
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    assert [item["id"] for item in all_items] == [
        r["id"] for r in revocations
    ]

    first = _list(client, limit="2").json()
    assert first["next_cursor"] is not None
    assert first["next_cursor"].startswith("gr1.")
    second = _list(client, limit="2", cursor=first["next_cursor"]).json()
    assert [item["id"] for item in second["items"]] == [
        revocations[2]["id"],
        revocations[3]["id"],
    ]
    third = _list(client, limit="2", cursor=second["next_cursor"]).json()
    assert [item["id"] for item in third["items"]] == [revocations[4]["id"]]
    assert third["next_cursor"] is None


def test_count_covers_every_filtered_page(client):
    _a, _t, grants, _revocations = _world_revocations(client)
    all_items, _pages, count = _walk_pages(
        client, grant_id=grants[0]["id"], limit="1"
    )
    assert count == 2
    assert len(all_items) == 2
    assert {item["grant_id"] for item in all_items} == {grants[0]["id"]}


def test_replayed_cursor_returns_the_same_page(client):
    _world_revocations(client)
    cursor = _list(client, limit="1").json()["next_cursor"]
    assert cursor is not None
    page = _list(client, limit="1", cursor=cursor)
    replay = _list(client, limit="1", cursor=cursor)
    assert page.status_code == 200
    assert replay.status_code == 200
    assert replay.content == page.content


def test_cursor_at_or_past_the_tail_returns_empty_page_with_count(client, app):
    _world_revocations(client)
    claims = {
        "grant_id": None,
        "revoker_actor_id": None,
        "reason": None,
        "from": None,
        "to": None,
        "limit": 50,
    }
    tail = pagination.encode_typed_cursor(
        app.state.attestation_access_grant_revocations_cursor_secret,
        pagination.ATTESTATION_ACCESS_GRANT_REVOCATIONS_CURSOR,
        {**claims, "offset": 5},
    )
    resp = _list(client, cursor=tail)
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 5, "next_cursor": None}

    past = pagination.encode_typed_cursor(
        app.state.attestation_access_grant_revocations_cursor_secret,
        pagination.ATTESTATION_ACCESS_GRANT_REVOCATIONS_CURSOR,
        {**claims, "offset": 99},
    )
    resp = _list(client, cursor=past)
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 5, "next_cursor": None}


def test_cursor_binds_every_effective_filter_and_limit(client):
    _a, _t, grants, _revocations = _world_revocations(client)
    cursor = _list(client, limit="2").json()["next_cursor"]
    assert cursor is not None
    # A different limit, a new filter, or a dropped filter all mismatch.
    for params in (
        {"limit": "3", "cursor": cursor},
        {"cursor": cursor},
        {"revoker_actor_id": "org-1", "limit": "2", "cursor": cursor},
        {"reason": REASON_A, "limit": "2", "cursor": cursor},
        {
            "grant_id": grants[0]["id"],
            "limit": "2",
            "cursor": cursor,
        },
        {
            "from": "2026-01-01T00:00:00Z",
            "limit": "2",
            "cursor": cursor,
        },
        {
            "to": "2030-01-01T00:00:00Z",
            "limit": "2",
            "cursor": cursor,
        },
    ):
        resp = _list(client, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"

    # A cursor minted under one exact filter cannot resume another.
    g1_cursor = _list(
        client, grant_id=grants[0]["id"], limit="1"
    ).json()["next_cursor"]
    assert g1_cursor is not None
    switched = _list(
        client, grant_id=grants[1]["id"], limit="1", cursor=g1_cursor
    )
    assert switched.status_code == 422
    revoker_cursor = _list(
        client, revoker_actor_id="org-1", limit="1"
    ).json()["next_cursor"]
    assert _list(
        client, revoker_actor_id="org-2", limit="1", cursor=revoker_cursor
    ).status_code == 422

    # A time-bound cursor is bound to the exact instant; equivalent UTC
    # notation resumes, a different second does not.
    time_cursor = _list(
        client, **{"from": "2000-01-01T00:00:00Z", "limit": "1"}
    ).json()["next_cursor"]
    assert time_cursor is not None
    same_bound = _list(
        client,
        **{
            "from": "2000-01-01T00:00:00+00:00",
            "limit": "1",
            "cursor": time_cursor,
        },
    )
    assert same_bound.status_code == 200
    changed_bound = _list(
        client,
        **{
            "from": "2000-01-01T00:00:01Z",
            "limit": "1",
            "cursor": time_cursor,
        },
    )
    assert changed_bound.status_code == 422


def test_blank_malformed_and_tampered_cursors_are_422(client):
    _world_revocations(client)
    valid = _list(client, limit="1").json()["next_cursor"]
    assert valid is not None
    tampered = valid[:-1] + ("A" if valid[-1] != "A" else "B")
    for bad in ("", "   ", "not-a-cursor", "gr1", "gr1.abc", tampered):
        resp = _list(client, cursor=bad)
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_foreign_family_cursor_is_422(client, app):
    _world_revocations(client)
    # A well-formed cursor minted by another endpoint family -- including the
    # closely named proof-revocation search and the per-grant collection --
    # never resumes this cross-grant retrieval.
    foreign = pagination.encode_typed_cursor(
        app.state.actors_cursor_secret,
        pagination.ACTORS_CURSOR,
        {"actor_id": None, "name": None, "actor_type": None, "limit": 50,
         "offset": 1},
    )
    assert _list(client, cursor=foreign).status_code == 422

    attestation_revocations = pagination.encode_typed_cursor(
        app.state.attestation_revocations_cursor_secret,
        pagination.ATTESTATION_REVOCATIONS_CURSOR,
        {
            "attestation_id": None,
            "revoker_actor_id": None,
            "reason": None,
            "from": None,
            "to": None,
            "limit": 50,
            "offset": 1,
        },
    )
    assert _list(client, cursor=attestation_revocations).status_code == 422


def test_cursor_with_wrong_claim_set_is_422(client, app):
    _world_revocations(client)
    # Correct family marker, HMAC, and base64, but a payload missing a bound
    # claim is rejected rather than trusted.
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "grant_id": None,
                "revoker_actor_id": None,
                "reason": None,
                "limit": 50,
                "offset": 1,
            },
            separators=(",", ":"),
        ).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(
        hmac.new(
            app.state.attestation_access_grant_revocations_cursor_secret,
            f"gr1.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    resp = _list(client, cursor=f"gr1.{payload}.{sig}")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_secret_rotates_on_restart_invalidating_old_cursors(
    tmp_db_url, file_client
):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    _world_revocations(file_client)
    cursor = _list(file_client, limit="1").json()["next_cursor"]
    assert cursor is not None

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as restarted_client:
        resp = _list(restarted_client, limit="1", cursor=cursor)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


# --- Method boundary ------------------------------------------------------------


def test_put_patch_and_delete_are_405_method_not_allowed(client):
    _world_revocations(client)
    for method in (client.put, client.patch, client.delete):
        resp = method(SEARCH_PATH)
        assert resp.status_code == 405
        assert resp.json()["error"]["code"] == "method_not_allowed"


def test_post_remains_the_authenticated_creation_route(client):
    # The new read entry does not shadow or alter authenticated revocation
    # creation: first write 201, identical retry 200 with the same record.
    att_one = _world(client)
    grant = _grant(client, att_one["id"], "org-2")
    first = _post_revocation(client, _revocation_body(grant["id"], REASON_A))
    assert first.status_code == 201
    repeat = _post_revocation(client, _revocation_body(grant["id"], REASON_A))
    assert repeat.status_code == 200
    assert repeat.json() == first.json()
    # And the search immediately reflects the new record.
    body = _list(client, grant_id=grant["id"]).json()
    assert body["count"] == 1
    assert body["items"] == [first.json()]


# --- Read-only guarantee ---------------------------------------------------------


def test_queries_and_failures_write_nothing(client, db_session):
    _a, _t, _g, revocations = _world_revocations(client)
    events_before = _audit_count(db_session)
    revocations_before = db_session.scalar(
        select(func.count()).select_from(AttestationAccessGrantRevocation)
    )

    assert _list(client).status_code == 200
    assert _list(client, revoker_actor_id="ghost").status_code == 200
    assert _list(client, reason="nobody said this").status_code == 200
    assert _list(client, grant_id="aag_ghost").status_code == 200
    assert (
        _list(
            client, limit="2", **{"from": "2026-01-01T00:00:00Z"}
        ).status_code
        == 200
    )
    all_items, _pages, count = _walk_pages(client, limit="1")
    assert count == len(revocations) == 5
    assert len(all_items) == 5

    # Failures are read-only as well.
    assert _list(client, limit="0").status_code == 422
    assert _list(client, grant_id=" ").status_code == 422
    assert _list(client, **{"from": "not-a-time"}).status_code == 422
    assert _list(client, cursor="bad").status_code == 422
    assert (
        client.request("GET", SEARCH_PATH, content=b"{}").status_code == 422
    )

    assert (
        db_session.scalar(
            select(func.count()).select_from(AttestationAccessGrantRevocation)
        )
        == revocations_before
    )
    assert _audit_count(db_session) == events_before


def test_all_numbers_in_the_body_are_integers(client):
    _world_revocations(client)
    raw = _list(client, limit="2").content.decode("utf-8")
    # No non-finite token is ever serialized (allow_nan=False).
    assert "NaN" not in raw and "Infinity" not in raw

    def assert_integral(value):
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return
        if isinstance(value, int):
            assert value != -0  # -0 == 0 in Python; emitted only as 0
            return
        if isinstance(value, float):  # pragma: no cover - fails the test
            raise AssertionError(f"non-integral number rendered: {value!r}")
        if isinstance(value, list):
            for member in value:
                assert_integral(member)
        elif isinstance(value, dict):
            for member in value.values():
                assert_integral(member)

    body = _list(client, limit="2").json()
    assert isinstance(body["count"], int)
    assert_integral(body)
