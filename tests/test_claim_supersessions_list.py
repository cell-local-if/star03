"""Tests for the global read-only claim-supersession search.

Covers ``GET /v1/claim-supersessions``: the exact
``items``/``count``/``next_cursor`` member order in compact UTF-8 JSON
terminated by one newline, each item reusing the supersession public view
(``id``, both endpoint claim ids, ``reason``, and UTC ``created_at``),
stable creation ordering (``created_at`` with the persistent insertion-order
tiebreaker that survives an app restart), the exact ``id``/
``superseded_claim_id``/``replacement_claim_id``/``reason`` filters and the
strict inclusive RFC 3339 UTC ``from``/``to`` bounds (from must not be later
than to), unknown values yielding an empty collection, pure-decimal
``limit`` validation (1..100, default 50), the opaque HMAC-signed ``cs1``
cursor family binding every effective filter and the limit, paging without
duplication or omission, a null cursor on the final page and an empty page
with the original count at or past the tail, every 422 validation boundary
(blank/bad-time/out-of-range/repeated/undeclared parameters, non-empty
bodies, blank/tampered/foreign-family/mismatched cursors), the 405
rejection of PUT/PATCH/DELETE, POST remaining the creation route, and the
strictly read-only guarantee on success, empty results, and failures.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timezone

from sqlalchemy import func, select

from provenance import pagination
from provenance.models import AuditEvent, ClaimSupersession
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    content_payload,
    create_actor,
)

SUPERSESSIONS_PATH = "/v1/claim-supersessions"

REASON_1 = "corrected source claim"
REASON_2 = "new evidence attached"
#: Non-ASCII rationale exercising unescaped compact UTF-8 rendering.
REASON_3 = "撤回与勘误"


# --- Fixture-style setup ------------------------------------------------------


def _create_content(client, digest=DIGEST_A, actor_id="org-1"):
    resp = client.post(
        "/v1/contents",
        json=content_payload(actor_id=actor_id, digest=digest),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, statement, claim_type="authorship"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": "org-1",
            "claim_type": claim_type,
            "payload": {"statement": statement},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_supersession(client, old, new, reason):
    resp = client.post(
        SUPERSESSIONS_PATH,
        json={
            "superseded_claim_id": old,
            "replacement_claim_id": new,
            "reason": reason,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _world(client):
    """Two contents and three supersessions, in a fixed creation order."""
    create_actor(client)
    a = _create_content(client, digest=DIGEST_A)
    b = _create_content(client, digest=DIGEST_B)
    a1 = _create_claim(client, a["id"], "first")
    a2 = _create_claim(client, a["id"], "second")
    a3 = _create_claim(client, a["id"], "third")
    b1 = _create_claim(client, b["id"], "b-first")
    b2 = _create_claim(client, b["id"], "b-second")

    s1 = _create_supersession(client, a1["id"], a2["id"], REASON_1)
    s2 = _create_supersession(client, a2["id"], a3["id"], REASON_2)
    s3 = _create_supersession(client, b1["id"], b2["id"], REASON_3)
    return a, b, [a1, a2, a3], [b1, b2], [s1, s2, s3]


def _pin_times(db_session, supersessions, instants):
    """Overwrite created_at per supersession id with a fixed UTC instant."""
    by_id = {s["id"]: instant for s, instant in zip(supersessions, instants)}
    for row in db_session.execute(select(ClaimSupersession)).scalars():
        row.created_at = by_id[row.id]
    db_session.commit()


def _list(client, **params):
    return client.get(SUPERSESSIONS_PATH, params=params)


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


def test_response_is_compact_json_with_ordered_members_and_one_newline(client):
    _world(client)
    resp = _list(client)
    assert resp.status_code == 200
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    assert raw.startswith(b'{"items":[')
    assert b'],"count":3,"next_cursor":null}' in raw
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        .encode("utf-8")
        + b"\n"
    )
    assert raw == expected


def test_non_ascii_reason_is_emitted_unescaped(client):
    _world(client)
    raw = _list(client).content
    # Compact UTF-8: the non-ASCII rationale is never emitted as a \u escape.
    assert REASON_3.encode("utf-8") in raw
    assert b"\\u" not in raw


def test_items_carry_exactly_the_supersession_public_view(client):
    _a, _b, claims_a, claims_b, supersessions = _world(client)
    body = _list(client).json()
    assert body["count"] == 3
    assert body["next_cursor"] is None
    assert [item["id"] for item in body["items"]] == [
        s["id"] for s in supersessions
    ]
    for item in body["items"]:
        assert list(item) == [
            "id",
            "superseded_claim_id",
            "replacement_claim_id",
            "reason",
            "created_at",
        ]
        created_at = datetime.fromisoformat(item["created_at"])
        assert created_at.utcoffset().total_seconds() == 0
        assert item["created_at"].endswith(("Z", "+00:00"))
    assert [item["superseded_claim_id"] for item in body["items"]] == [
        claims_a[0]["id"],
        claims_a[1]["id"],
        claims_b[0]["id"],
    ]
    assert [item["replacement_claim_id"] for item in body["items"]] == [
        claims_a[1]["id"],
        claims_a[2]["id"],
        claims_b[1]["id"],
    ]
    assert [item["reason"] for item in body["items"]] == [
        REASON_1,
        REASON_2,
        REASON_3,
    ]
    for item, supersession in zip(body["items"], supersessions):
        assert item == supersession
    # No internal ordering surrogate, payload, signature, or byte is echoed.
    rendered = _list(client).content.decode()
    assert "seq" not in rendered


# --- Stable ordering -----------------------------------------------------------


def test_items_follow_stable_creation_order(client):
    _a, _b, _ca, _cb, supersessions = _world(client)
    body = _list(client).json()
    assert [item["id"] for item in body["items"]] == [
        s["id"] for s in supersessions
    ]
    created = [
        datetime.fromisoformat(item["created_at"]) for item in body["items"]
    ]
    assert created == sorted(created)


def test_same_timestamp_ties_break_by_persistence_order(client, db_session):
    _a, _b, _ca, _cb, supersessions = _world(client)
    tie = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    _pin_times(db_session, supersessions, [tie] * 3)

    body = _list(client).json()
    assert [item["id"] for item in body["items"]] == [
        s["id"] for s in supersessions
    ]


def test_ordering_is_identical_across_an_app_restart(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    _world(file_client)
    first = _list(file_client)
    assert first.status_code == 200

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as restarted_client:
        second = _list(restarted_client)
        assert second.status_code == 200
        assert second.json() == first.json()
        assert [item["id"] for item in second.json()["items"]] == [
            s["id"] for s in first.json()["items"]
        ]


# --- Exact-match filters --------------------------------------------------------


def test_id_filter_is_an_exact_match(client):
    _a, _b, _ca, _cb, supersessions = _world(client)
    body = _list(client, id=supersessions[1]["id"]).json()
    assert body == {
        "items": [supersessions[1]],
        "count": 1,
        "next_cursor": None,
    }


def test_superseded_claim_filter_is_an_exact_match(client):
    _a, _b, claims_a, _cb, supersessions = _world(client)
    body = _list(
        client, superseded_claim_id=claims_a[1]["id"]
    ).json()
    assert body["count"] == 1
    assert [item["id"] for item in body["items"]] == [
        supersessions[1]["id"]
    ]


def test_replacement_claim_filter_is_an_exact_match(client):
    _a, _b, claims_a, _cb, supersessions = _world(client)
    body = _list(
        client, replacement_claim_id=claims_a[1]["id"]
    ).json()
    assert body["count"] == 1
    assert [item["id"] for item in body["items"]] == [
        supersessions[0]["id"]
    ]


def test_a_claim_matches_records_on_either_endpoint(client):
    _a, _b, claims_a, _cb, supersessions = _world(client)
    # The middle claim is replaced by s1 and supersedes via s2: the global
    # filter keeps every record naming it on either endpoint.
    body = _list(client, superseded_claim_id=claims_a[1]["id"]).json()
    assert [item["id"] for item in body["items"]] == [
        supersessions[1]["id"]
    ]
    body = _list(
        client,
        superseded_claim_id=claims_a[1]["id"],
        replacement_claim_id=claims_a[2]["id"],
    ).json()
    assert [item["id"] for item in body["items"]] == [
        supersessions[1]["id"]
    ]


def test_reason_filter_is_an_exact_match(client):
    _world(client)
    body = _list(client, reason=REASON_2).json()
    assert body["count"] == 1
    assert body["items"][0]["reason"] == REASON_2
    # A different reason is an independently retained, independently
    # matchable record.
    assert _list(client, reason=REASON_1).json()["count"] == 1
    assert _list(client, reason=REASON_3).json()["count"] == 1


def test_filters_combine_as_logical_and(client):
    _a, _b, claims_a, claims_b, supersessions = _world(client)
    hit = _list(
        client,
        superseded_claim_id=claims_b[0]["id"],
        replacement_claim_id=claims_b[1]["id"],
        reason=REASON_3,
    ).json()
    assert hit["count"] == 1
    assert [item["id"] for item in hit["items"]] == [
        supersessions[2]["id"]
    ]

    # The right endpoints with a different reason cannot match.
    miss = _list(
        client,
        superseded_claim_id=claims_b[0]["id"],
        replacement_claim_id=claims_b[1]["id"],
        reason=REASON_1,
    ).json()
    assert miss == {"items": [], "count": 0, "next_cursor": None}


def test_filters_are_case_and_whitespace_sensitive(client):
    _a, _b, claims_a, _cb, supersessions = _world(client)
    for params in (
        {"id": supersessions[0]["id"].upper()},
        {"id": f" {supersessions[0]['id']}"},
        {"superseded_claim_id": "clm_ghost"},
        {"replacement_claim_id": f"{claims_a[1]['id']} "},
        {"reason": REASON_1.upper()},
        {"reason": f" {REASON_1}"},
        {"reason": REASON_1 + " "},
    ):
        resp = _list(client, **params)
        assert resp.status_code == 200, params
        assert resp.json() == {
            "items": [],
            "count": 0,
            "next_cursor": None,
        }, params


def test_unknown_or_nonexistent_filter_value_is_an_empty_collection(client):
    _world(client)
    for params in (
        {"id": "csp_ghost"},
        {"superseded_claim_id": "clm_ghost"},
        {"replacement_claim_id": "clm_ghost"},
        {"reason": "no such rationale"},
        {
            "superseded_claim_id": "clm_ghost",
            "replacement_claim_id": "clm_ghost",
        },
    ):
        resp = _list(client, **params)
        assert resp.status_code == 200, params
        assert resp.json() == {
            "items": [],
            "count": 0,
            "next_cursor": None,
        }


def test_blank_filters_are_422(client):
    _world(client)
    for field in (
        "id",
        "superseded_claim_id",
        "replacement_claim_id",
        "reason",
    ):
        for blank in ("", "   ", "\t"):
            resp = _list(client, **{field: blank})
            assert resp.status_code == 422, (field, repr(blank))
            assert resp.json()["error"]["code"] == "validation_error"


# --- Time bounds ----------------------------------------------------------------


def _pin_three_times(db_session, supersessions):
    _pin_times(
        db_session,
        supersessions,
        [
            datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 3, 9, 0, 0, tzinfo=timezone.utc),
        ],
    )


def test_from_bound_is_inclusive(client, db_session):
    _a, _b, _ca, _cb, supersessions = _world(client)
    _pin_three_times(db_session, supersessions)
    body = _list(client, **{"from": "2026-01-02T12:30:00Z"}).json()
    assert [item["id"] for item in body["items"]] == [
        s["id"] for s in supersessions[1:]
    ]
    assert body["count"] == 2


def test_to_bound_is_inclusive(client, db_session):
    _a, _b, _ca, _cb, supersessions = _world(client)
    _pin_three_times(db_session, supersessions)
    body = _list(client, to="2026-01-02T12:30:00Z").json()
    assert [item["id"] for item in body["items"]] == [
        s["id"] for s in supersessions[:2]
    ]
    assert body["count"] == 2


def test_equal_from_and_to_selects_exactly_that_instant(client, db_session):
    _a, _b, _ca, _cb, supersessions = _world(client)
    _pin_times(
        db_session,
        supersessions,
        [
            datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc),
        ],
    )
    body = _list(
        client,
        **{"from": "2026-01-02T12:30:00Z", "to": "2026-01-02T12:30:00Z"},
    ).json()
    assert [item["id"] for item in body["items"]] == [
        supersessions[1]["id"],
        supersessions[2]["id"],
    ]
    assert body["count"] == 2


def test_equivalent_utc_notations_are_the_same_bound(client, db_session):
    _a, _b, _ca, _cb, supersessions = _world(client)
    _pin_three_times(db_session, supersessions)
    zed = _list(client, **{"from": "2026-01-02T12:30:00Z"}).json()
    offset = _list(
        client, **{"from": "2026-01-02T12:30:00+00:00"}
    ).json()
    assert zed == offset
    assert [item["id"] for item in zed["items"]] == [
        s["id"] for s in supersessions[1:]
    ]


def test_bad_time_bounds_and_inverted_range_are_422(client):
    _world(client)
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
        client,
        **{"from": "2026-01-03T00:00:00Z", "to": "2026-01-02T00:00:00Z"},
    )
    assert inverted.status_code == 422
    assert inverted.json()["error"]["code"] == "validation_error"
    for blank in ("", "   "):
        assert _list(client, **{"from": blank}).status_code == 422
        assert _list(client, to=blank).status_code == 422


# --- limit validation -----------------------------------------------------------


def test_limit_boundaries_one_and_one_hundred_are_accepted(client):
    _world(client)
    assert _list(client, limit="1").status_code == 200
    assert _list(client, limit="100").status_code == 200


def test_limit_must_be_a_pure_decimal_integer_in_range(client):
    _world(client)
    for bad in ("0", "101", "-1", "5.0", "5e0", " 5", "5 ", "five", "+5", ""):
        resp = _list(client, limit=bad)
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_default_limit_is_fifty(client):
    # A chain of 52 claims carries 51 supersessions; default paging caps
    # the first page at fifty and issues a continuation cursor.
    create_actor(client)
    content = _create_content(client)
    chain = [
        _create_claim(client, content["id"], f"version-{i}")
        for i in range(52)
    ]
    for earlier, later in zip(chain, chain[1:]):
        resp = client.post(
            SUPERSESSIONS_PATH,
            json={
                "superseded_claim_id": earlier["id"],
                "replacement_claim_id": later["id"],
                "reason": REASON_1,
            },
        )
        assert resp.status_code == 201, resp.text
    resp = _list(client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 51
    assert len(body["items"]) == 50
    assert body["next_cursor"] is not None
    assert body["next_cursor"].startswith("cs1.")


# --- Parameter strictness -------------------------------------------------------


def test_repeated_and_undeclared_parameters_are_422(client):
    _world(client)
    assert (
        client.get(
            SUPERSESSIONS_PATH,
            params=[
                ("superseded_claim_id", "clm_a"),
                ("superseded_claim_id", "clm_b"),
            ],
        ).status_code
        == 422
    )
    assert (
        client.get(
            SUPERSESSIONS_PATH,
            params=[("reason", "one"), ("reason", "two")],
        ).status_code
        == 422
    )
    assert (
        client.get(
            SUPERSESSIONS_PATH,
            params=[("limit", "1"), ("limit", "2")],
        ).status_code
        == 422
    )
    assert (
        client.get(
            SUPERSESSIONS_PATH,
            params=[
                ("from", "2026-01-01T00:00:00Z"),
                ("from", "2026-01-02T00:00:00Z"),
            ],
        ).status_code
        == 422
    )
    for params in (
        {"supersession_id": "csp_1"},
        {"claim_id": "clm_1"},
        {"offset": "1"},
        {"q": "x"},
        {"cursor": "x", "bogus": "y"},
    ):
        resp = _list(client, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_empty_body_is_422_rejected_before_any_read(client, db_session):
    _world(client)
    events_before = _audit_count(db_session)
    for body in (b"{}", b" ", b"{not valid json", b"[]", b"\n", b"\x00\xff"):
        resp = client.request(
            "GET",
            SUPERSESSIONS_PATH,
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"
    assert _audit_count(db_session) == events_before


# --- Pagination -----------------------------------------------------------------


def test_pages_resume_without_duplication_or_omission(client):
    _a, _b, _ca, _cb, supersessions = _world(client)
    all_items, pages, count = _walk_pages(client, limit="2")
    assert count == 3
    assert [len(page) for page in pages] == [2, 1]
    assert [item["id"] for item in all_items] == [
        s["id"] for s in supersessions
    ]

    first = _list(client, limit="2").json()
    assert first["next_cursor"] is not None
    assert first["next_cursor"].startswith("cs1.")
    second = _list(
        client, limit="2", cursor=first["next_cursor"]
    ).json()
    assert second["next_cursor"] is None


def test_count_covers_every_filtered_page(client):
    _a, _b, claims_a, _cb, _s = _world(client)
    all_items, _pages, count = _walk_pages(
        client, reason=REASON_1, limit="1"
    )
    assert count == 1
    assert len(all_items) == 1
    assert {item["reason"] for item in all_items} == {REASON_1}

    # A filter matching the first two records pages both.
    all_items, _pages, count = _walk_pages(
        client,
        superseded_claim_id=claims_a[0]["id"],
        limit="1",
    )
    assert count == 1
    assert len(all_items) == 1


def test_replayed_cursor_returns_the_same_page(client):
    _world(client)
    cursor = _list(client, limit="1").json()["next_cursor"]
    assert cursor is not None
    page = _list(client, limit="1", cursor=cursor)
    replay = _list(client, limit="1", cursor=cursor)
    assert page.status_code == 200
    assert replay.status_code == 200
    assert replay.content == page.content


def test_cursor_at_or_past_the_tail_returns_empty_page_with_count(
    client, app
):
    _world(client)
    claims = {
        "supersession_id": None,
        "superseded_claim_id": None,
        "replacement_claim_id": None,
        "reason": None,
        "from": None,
        "to": None,
        "limit": 50,
    }
    tail = pagination.encode_typed_cursor(
        app.state.claim_supersessions_cursor_secret,
        pagination.CLAIM_SUPERSESSIONS_CURSOR,
        {**claims, "offset": 3},
    )
    resp = _list(client, cursor=tail)
    assert resp.status_code == 200
    assert resp.json() == {
        "items": [],
        "count": 3,
        "next_cursor": None,
    }

    past = pagination.encode_typed_cursor(
        app.state.claim_supersessions_cursor_secret,
        pagination.CLAIM_SUPERSESSIONS_CURSOR,
        {**claims, "offset": 99},
    )
    resp = _list(client, cursor=past)
    assert resp.status_code == 200
    assert resp.json() == {
        "items": [],
        "count": 3,
        "next_cursor": None,
    }


def test_cursor_binds_every_effective_filter_and_limit(client):
    _a, _b, claims_a, claims_b, _s = _world(client)
    cursor = _list(client, limit="2").json()["next_cursor"]
    assert cursor is not None
    # A different limit, a new filter, or a dropped filter all mismatch.
    for params in (
        {"limit": "3", "cursor": cursor},
        {"cursor": cursor},
        {
            "superseded_claim_id": claims_a[0]["id"],
            "limit": "2",
            "cursor": cursor,
        },
        {
            "replacement_claim_id": claims_b[1]["id"],
            "limit": "2",
            "cursor": cursor,
        },
        {"reason": REASON_1, "limit": "2", "cursor": cursor},
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

    # A cursor minted under one exact filter cannot resume another. A
    # second, independently retained record between the same two endpoints
    # (a different reason) gives both the endpoint and the REASON_1 filter
    # two matches.
    extra = _create_supersession(
        client, claims_a[0]["id"], claims_a[1]["id"], REASON_2
    )
    assert extra["id"] != _s[0]["id"]
    old_cursor = _list(
        client, superseded_claim_id=claims_a[0]["id"], limit="1"
    ).json()["next_cursor"]
    assert old_cursor is not None
    assert (
        _list(
            client,
            superseded_claim_id=claims_b[0]["id"],
            limit="1",
            cursor=old_cursor,
        ).status_code
        == 422
    )
    # Two records now share REASON_2 (the parallel record and s2), so the
    # reason filter issues a continuation cursor.
    reason_cursor = _list(
        client, reason=REASON_2, limit="1"
    ).json()["next_cursor"]
    assert reason_cursor is not None
    assert (
        _list(
            client, reason=REASON_1, limit="1", cursor=reason_cursor
        ).status_code
        == 422
    )

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
    _world(client)
    valid = _list(client, limit="1").json()["next_cursor"]
    assert valid is not None
    tampered = valid[:-1] + ("A" if valid[-1] != "A" else "B")
    for bad in ("", "   ", "not-a-cursor", "cs1", "cs1.abc", tampered):
        resp = _list(client, cursor=bad)
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_foreign_family_cursor_is_422(client, app):
    _a, _b, claims_a, claims_b, supersessions = _world(client)
    # A well-formed cursor minted by another endpoint family never resumes
    # this retrieval.
    foreign = pagination.encode_typed_cursor(
        app.state.actors_cursor_secret,
        pagination.ACTORS_CURSOR,
        {
            "actor_id": None,
            "name": None,
            "actor_type": None,
            "limit": 50,
            "offset": 1,
        },
    )
    resp = _list(client, cursor=foreign)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    # The supersession-lineage family shares the domain but not the entry:
    # its sl1 cursor is a cross-entry cursor and is rejected here.
    lineage = pagination.encode_typed_cursor(
        app.state.claim_supersession_lineage_cursor_secret,
        pagination.CLAIM_SUPERSESSION_LINEAGE_CURSOR,
        {
            "claim_id": claims_a[0]["id"],
            "direction": "newer",
            "max_depth": 8,
            "min_depth": 1,
            "limit": 50,
            "offset": 1,
        },
    )
    assert _list(client, cursor=lineage).status_code == 422

    relations = pagination.encode_typed_cursor(
        app.state.content_relations_cursor_secret,
        pagination.CONTENT_RELATIONS_CURSOR,
        {
            "relation_id": None,
            "content_id": None,
            "parent_content_id": None,
            "relation_type": None,
            "from": None,
            "to": None,
            "limit": 50,
            "offset": 1,
        },
    )
    assert _list(client, cursor=relations).status_code == 422


def test_cursor_with_wrong_claim_set_is_422(client, app):
    _world(client)
    # Correct family marker, HMAC, and base64, but a payload missing a bound
    # claim is rejected rather than trusted.
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "supersession_id": None,
                "superseded_claim_id": None,
                "replacement_claim_id": None,
                "reason": None,
                "limit": 50,
                "offset": 1,
            },
            separators=(",", ":"),
        ).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(
        hmac.new(
            app.state.claim_supersessions_cursor_secret,
            f"cs1.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    resp = _list(client, cursor=f"cs1.{payload}.{sig}")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_secret_rotates_on_restart_invalidating_old_cursors(
    tmp_db_url, file_client
):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    _world(file_client)
    cursor = _list(file_client, limit="1").json()["next_cursor"]
    assert cursor is not None

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as restarted_client:
        resp = _list(restarted_client, limit="1", cursor=cursor)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


# --- Method boundary ------------------------------------------------------------


def test_put_patch_and_delete_are_405_method_not_allowed(client):
    _world(client)
    for method in (client.put, client.patch, client.delete):
        resp = method(SUPERSESSIONS_PATH)
        assert resp.status_code == 405
        assert resp.json()["error"]["code"] == "method_not_allowed"


def test_post_remains_the_supersession_creation_route(client):
    # The new read entry does not shadow or alter supersession creation,
    # its idempotency, or its 201/200 distinction.
    create_actor(client)
    content = _create_content(
        client, digest=hashlib.sha256(b"post-route").hexdigest()
    )
    old = _create_claim(client, content["id"], "earlier")
    new = _create_claim(client, content["id"], "later")
    payload = {
        "superseded_claim_id": old["id"],
        "replacement_claim_id": new["id"],
        "reason": REASON_1,
    }
    first = client.post(SUPERSESSIONS_PATH, json=payload)
    assert first.status_code == 201
    repeat = client.post(SUPERSESSIONS_PATH, json=payload)
    assert repeat.status_code == 200
    assert repeat.json() == first.json()


# --- Read-only guarantee ---------------------------------------------------------


def test_queries_and_failures_write_nothing(client, db_session):
    _a, _b, _ca, _cb, supersessions = _world(client)
    events_before = _audit_count(db_session)
    rows_before = db_session.scalar(
        select(func.count()).select_from(ClaimSupersession)
    )

    assert _list(client).status_code == 200
    assert (
        _list(client, superseded_claim_id="clm_ghost").status_code == 200
    )
    assert _list(client, id="csp_nobody").status_code == 200
    assert (
        _list(client, limit="2", **{"from": "2026-01-01T00:00:00Z"}).status_code
        == 200
    )
    all_items, _pages, count = _walk_pages(client, limit="1")
    assert count == len(supersessions) == 3
    assert len(all_items) == 3

    # Failures are read-only as well.
    assert _list(client, limit="0").status_code == 422
    assert _list(client, reason=" ").status_code == 422
    assert _list(client, **{"from": "not-a-time"}).status_code == 422
    assert _list(client, cursor="bad").status_code == 422
    assert (
        client.request(
            "GET", SUPERSESSIONS_PATH, content=b"{}"
        ).status_code
        == 422
    )

    assert (
        db_session.scalar(
            select(func.count()).select_from(ClaimSupersession)
        )
        == rows_before
    )
    assert _audit_count(db_session) == events_before
