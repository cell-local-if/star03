"""Tests for the global read-only content-relation search.

Covers ``GET /v1/content-relations``: the exact
``items``/``count``/``next_cursor`` member order in compact UTF-8 JSON
terminated by one newline, each item reusing the relation public view
(stable id, both content ids, relation type, UTC ``created_at``; never the
internal insertion sequence), stable creation ordering (``created_at`` with
the persistent insertion-order tiebreaker that survives an app restart),
the exact ``id``/``content_id``/``parent_content_id`` filters, the strict
``relation_type`` literal filter, the strict inclusive RFC 3339 UTC
``from``/``to`` bounds (from must not be later than to), unknown values
yielding an empty collection, pure-decimal ``limit`` validation (1..100,
default 50), the opaque HMAC-signed ``rl1`` cursor family binding every
effective filter and the limit, paging without duplication or omission, a
null cursor on the final page and an empty page with the original count at
or past the tail, every 422 validation boundary (blank/bad-type/bad-time/
out-of-range/repeated/undeclared parameters, non-empty bodies,
blank/tampered/foreign-family/mismatched cursors), the 405 rejection of
PUT/PATCH/DELETE, POST remaining the creation route, and the strictly
read-only guarantee on success, empty results, and failures.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timezone

from sqlalchemy import func, select

from provenance import pagination
from provenance.models import AuditEvent, ContentRelation
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    DIGEST_C,
    content_payload,
    create_actor,
)

RELATIONS_PATH = "/v1/content-relations"

DIGEST_D = hashlib.sha256(b"content-d").hexdigest()


# --- Fixture-style setup ------------------------------------------------------


def _create_content(client, digest, actor_id="org-1"):
    resp = client.post(
        "/v1/contents", json=content_payload(actor_id=actor_id, digest=digest)
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_relation(client, content_id, parent_content_id, relation_type):
    resp = client.post(
        RELATIONS_PATH,
        json={
            "content_id": content_id,
            "parent_content_id": parent_content_id,
            "relation_type": relation_type,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _world(client):
    """Four contents and four interleaved relations, in creation order."""
    create_actor(client)
    a = _create_content(client, DIGEST_A)
    b = _create_content(client, DIGEST_B)
    c = _create_content(client, DIGEST_C)
    d = _create_content(client, DIGEST_D)

    r1 = _create_relation(client, b["id"], a["id"], "version_of")
    r2 = _create_relation(client, c["id"], a["id"], "derived_from")
    r3 = _create_relation(client, c["id"], b["id"], "version_of")
    r4 = _create_relation(client, d["id"], c["id"], "derived_from")
    return (a, b, c, d), [r1, r2, r3, r4]


def _pin_times(db_session, relations, instants):
    """Overwrite created_at per relation id with a fixed UTC instant."""
    by_id = {r["id"]: instant for r, instant in zip(relations, instants)}
    for row in db_session.execute(select(ContentRelation)).scalars():
        row.created_at = by_id[row.id]
    db_session.commit()


def _list(client, **params):
    return client.get(RELATIONS_PATH, params=params)


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
    assert b'],"count":4,"next_cursor":null}' in raw
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        .encode("utf-8")
        + b"\n"
    )
    assert raw == expected


def test_items_carry_exactly_the_relation_public_view(client):
    _, relations = _world(client)
    resp = _list(client)
    body = resp.json()
    assert body["count"] == 4
    assert body["next_cursor"] is None
    assert [item["id"] for item in body["items"]] == [r["id"] for r in relations]
    for item, relation in zip(body["items"], relations):
        assert list(item) == [
            "id",
            "content_id",
            "parent_content_id",
            "relation_type",
            "created_at",
        ]
        assert item == relation
        created_at = datetime.fromisoformat(item["created_at"])
        assert created_at.utcoffset().total_seconds() == 0
        assert item["created_at"].endswith(("Z", "+00:00"))
    # The internal monotonic insertion sequence is never echoed.
    assert "seq" not in resp.content.decode()


# --- Stable ordering -----------------------------------------------------------


def test_items_follow_stable_creation_order(client):
    _, relations = _world(client)
    body = _list(client).json()
    assert [item["id"] for item in body["items"]] == [
        r["id"] for r in relations
    ]
    created = [datetime.fromisoformat(item["created_at"]) for item in body["items"]]
    assert created == sorted(created)


def test_same_timestamp_ties_break_by_persistence_order(client, db_session):
    _, relations = _world(client)
    tie = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    _pin_times(db_session, relations, [tie] * 4)

    body = _list(client).json()
    assert [item["id"] for item in body["items"]] == [
        r["id"] for r in relations
    ]


def test_ordering_is_identical_across_an_app_restart(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    _, relations = _world(file_client)
    first = _list(file_client)
    assert first.status_code == 200

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as restarted_client:
        second = _list(restarted_client)
        assert second.status_code == 200
        assert second.json() == first.json()
        assert [item["id"] for item in second.json()["items"]] == [
            r["id"] for r in relations
        ]


# --- Exact-match filters --------------------------------------------------------


def test_id_filter_is_an_exact_match(client):
    _, relations = _world(client)
    body = _list(client, id=relations[2]["id"]).json()
    assert body == {
        "items": [relations[2]],
        "count": 1,
        "next_cursor": None,
    }


def test_content_id_filter_is_an_exact_match(client):
    (a, b, c, d), relations = _world(client)
    body = _list(client, content_id=c["id"]).json()
    assert body["count"] == 2
    assert body["next_cursor"] is None
    assert [item["id"] for item in body["items"]] == [
        relations[1]["id"],
        relations[2]["id"],
    ]


def test_parent_content_id_filter_is_an_exact_match(client):
    (a, b, c, d), relations = _world(client)
    body = _list(client, parent_content_id=a["id"]).json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [
        relations[0]["id"],
        relations[1]["id"],
    ]


def test_relation_type_filter_matches_only_the_literal(client):
    _, relations = _world(client)
    versions = _list(client, relation_type="version_of").json()
    assert [item["id"] for item in versions["items"]] == [
        relations[0]["id"],
        relations[2]["id"],
    ]
    derived = _list(client, relation_type="derived_from").json()
    assert [item["id"] for item in derived["items"]] == [
        relations[1]["id"],
        relations[3]["id"],
    ]


def test_filters_combine_as_logical_and(client):
    (a, b, c, d), relations = _world(client)
    body = _list(
        client, content_id=c["id"], relation_type="version_of"
    ).json()
    assert body["count"] == 1
    assert [item["id"] for item in body["items"]] == [relations[2]["id"]]

    miss = _list(
        client, content_id=c["id"], relation_type="derived_from",
        parent_content_id=b["id"],
    ).json()
    assert miss == {"items": [], "count": 0, "next_cursor": None}


def test_filters_are_case_and_whitespace_sensitive(client):
    (a, b, c, d), relations = _world(client)
    for params in (
        {"id": relations[0]["id"].upper()},
        {"id": f" {relations[0]['id']}"},
        {"content_id": c["id"].upper()},
        {"content_id": f"{c['id']} "},
        {"parent_content_id": f" {a['id']}"},
        {"parent_content_id": f"{a['id']}\t"},
    ):
        resp = _list(client, **params)
        assert resp.status_code == 200, params
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_unknown_or_nonexistent_filter_value_is_an_empty_collection(client):
    _world(client)
    for params in (
        {"id": "rel_ghost"},
        {"content_id": "cnt_ghost"},
        {"parent_content_id": "cnt_ghost"},
        {"id": "rel_ghost", "content_id": "cnt_ghost"},
    ):
        resp = _list(client, **params)
        assert resp.status_code == 200, params
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_blank_filters_are_422(client):
    _world(client)
    for field in ("id", "content_id", "parent_content_id"):
        for blank in ("", "   ", "\t"):
            resp = _list(client, **{field: blank})
            assert resp.status_code == 422, (field, repr(blank))
            assert resp.json()["error"]["code"] == "validation_error"


def test_unknown_relation_type_is_422_not_an_empty_collection(client):
    _world(client)
    for bad in ("copied_from", "VERSION_OF", "Derived_From", "", " ", "1"):
        resp = _list(client, relation_type=bad)
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error"


# --- Time bounds ----------------------------------------------------------------


def test_from_bound_is_inclusive(client, db_session):
    _, relations = _world(client)
    _pin_times(
        db_session,
        relations,
        [
            datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 3, 9, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 4, 0, 0, 0, tzinfo=timezone.utc),
        ],
    )
    body = _list(client, **{"from": "2026-01-02T12:30:00Z"}).json()
    assert [item["id"] for item in body["items"]] == [
        r["id"] for r in relations[1:]
    ]
    assert body["count"] == 3


def test_to_bound_is_inclusive(client, db_session):
    _, relations = _world(client)
    _pin_times(
        db_session,
        relations,
        [
            datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 3, 9, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 4, 0, 0, 0, tzinfo=timezone.utc),
        ],
    )
    body = _list(client, to="2026-01-02T12:30:00Z").json()
    assert [item["id"] for item in body["items"]] == [
        r["id"] for r in relations[:2]
    ]
    assert body["count"] == 2


def test_equal_from_and_to_selects_exactly_that_instant(client, db_session):
    _, relations = _world(client)
    _pin_times(
        db_session,
        relations,
        [
            datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 4, 0, 0, 0, tzinfo=timezone.utc),
        ],
    )
    body = _list(
        client, **{"from": "2026-01-02T12:30:00Z", "to": "2026-01-02T12:30:00Z"}
    ).json()
    assert [item["id"] for item in body["items"]] == [
        relations[1]["id"],
        relations[2]["id"],
    ]
    assert body["count"] == 2


def test_equivalent_utc_notations_are_the_same_bound(client, db_session):
    _, relations = _world(client)
    instant = datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc)
    _pin_times(
        db_session,
        relations,
        [
            datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            instant,
            datetime(2026, 1, 3, 9, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 4, 0, 0, 0, tzinfo=timezone.utc),
        ],
    )
    zed = _list(client, **{"from": "2026-01-02T12:30:00Z"}).json()
    offset = _list(client, **{"from": "2026-01-02T12:30:00+00:00"}).json()
    assert zed == offset
    assert [item["id"] for item in zed["items"]] == [
        r["id"] for r in relations[1:]
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
        client, **{"from": "2026-01-03T00:00:00Z", "to": "2026-01-02T00:00:00Z"}
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
    create_actor(client)
    digests = [
        hashlib.sha256(f"bulk-{index}".encode()).hexdigest()
        for index in range(8)
    ]
    contents = [_create_content(client, digest) for digest in digests]
    # Edges always point from a later content to an earlier one, so the
    # graph stays acyclic; two types per pair give 56 possible relations.
    created = 0
    for child in range(8):
        for parent in range(child):
            for relation_type in ("version_of", "derived_from"):
                if created >= 51:
                    break
                _create_relation(
                    client,
                    contents[child]["id"],
                    contents[parent]["id"],
                    relation_type,
                )
                created += 1
    assert created == 51
    resp = _list(client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 51
    assert len(body["items"]) == 50
    assert body["next_cursor"] is not None


# --- Parameter strictness -------------------------------------------------------


def test_repeated_and_undeclared_parameters_are_422(client):
    _world(client)
    assert (
        client.get(
            RELATIONS_PATH,
            params=[("id", "rel_a"), ("id", "rel_b")],
        ).status_code
        == 422
    )
    assert (
        client.get(
            RELATIONS_PATH,
            params=[("content_id", "cnt_a"), ("content_id", "cnt_b")],
        ).status_code
        == 422
    )
    assert (
        client.get(
            RELATIONS_PATH,
            params=[("relation_type", "version_of"),
                    ("relation_type", "derived_from")],
        ).status_code
        == 422
    )
    assert (
        client.get(
            RELATIONS_PATH,
            params=[("limit", "1"), ("limit", "2")],
        ).status_code
        == 422
    )
    assert (
        client.get(
            RELATIONS_PATH,
            params=[("from", "2026-01-01T00:00:00Z"),
                    ("from", "2026-01-02T00:00:00Z")],
        ).status_code
        == 422
    )
    for params in (
        {"relation_id": "rel_1"},
        {"type": "version_of"},
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
            RELATIONS_PATH,
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"
    assert _audit_count(db_session) == events_before


# --- Pagination -----------------------------------------------------------------


def test_pages_resume_without_duplication_or_omission(client):
    _, relations = _world(client)
    all_items, pages, count = _walk_pages(client, limit="2")
    assert count == 4
    assert [len(page) for page in pages] == [2, 2]
    assert [item["id"] for item in all_items] == [r["id"] for r in relations]

    first = _list(client, limit="2").json()
    assert first["next_cursor"] is not None
    assert first["next_cursor"].startswith("rl1.")
    second = _list(client, limit="2", cursor=first["next_cursor"]).json()
    assert second["next_cursor"] is None


def test_count_covers_every_filtered_page(client):
    (a, b, c, d), _ = _world(client)
    all_items, _pages, count = _walk_pages(
        client, parent_content_id=a["id"], limit="1"
    )
    assert count == 2
    assert len(all_items) == 2
    assert {item["parent_content_id"] for item in all_items} == {a["id"]}


def test_replayed_cursor_returns_the_same_page(client):
    _world(client)
    cursor = _list(client, limit="1").json()["next_cursor"]
    assert cursor is not None
    page = _list(client, limit="1", cursor=cursor)
    replay = _list(client, limit="1", cursor=cursor)
    assert page.status_code == 200
    assert replay.status_code == 200
    assert replay.content == page.content


def test_cursor_at_or_past_the_tail_returns_empty_page_with_count(client, app):
    _world(client)
    claims = {
        "id": None,
        "content_id": None,
        "parent_content_id": None,
        "relation_type": None,
        "from": None,
        "to": None,
        "limit": 50,
    }
    tail = pagination.encode_typed_cursor(
        app.state.content_relations_cursor_secret,
        pagination.CONTENT_RELATIONS_CURSOR,
        {**claims, "offset": 4},
    )
    resp = _list(client, cursor=tail)
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 4, "next_cursor": None}

    past = pagination.encode_typed_cursor(
        app.state.content_relations_cursor_secret,
        pagination.CONTENT_RELATIONS_CURSOR,
        {**claims, "offset": 99},
    )
    resp = _list(client, cursor=past)
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 4, "next_cursor": None}


def test_cursor_binds_every_effective_filter_and_limit(client):
    (a, b, c, d), relations = _world(client)
    cursor = _list(client, limit="2").json()["next_cursor"]
    assert cursor is not None
    # A different limit, a new filter, or a dropped filter all mismatch.
    for params in (
        {"limit": "3", "cursor": cursor},
        {"cursor": cursor},
        {"id": relations[0]["id"], "limit": "2", "cursor": cursor},
        {"content_id": c["id"], "limit": "2", "cursor": cursor},
        {"parent_content_id": a["id"], "limit": "2", "cursor": cursor},
        {"relation_type": "version_of", "limit": "2", "cursor": cursor},
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
    filtered_cursor = _list(
        client, content_id=c["id"], limit="1"
    ).json()["next_cursor"]
    assert filtered_cursor is not None
    switched = _list(
        client, content_id=b["id"], limit="1", cursor=filtered_cursor
    )
    assert switched.status_code == 422
    type_cursor = _list(
        client, relation_type="version_of", limit="1"
    ).json()["next_cursor"]
    assert _list(
        client, relation_type="derived_from", limit="1", cursor=type_cursor
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
    _world(client)
    valid = _list(client, limit="1").json()["next_cursor"]
    assert valid is not None
    tampered = valid[:-1] + ("A" if valid[-1] != "A" else "B")
    for bad in ("", "   ", "not-a-cursor", "rl1", "rl1.abc", tampered):
        resp = _list(client, cursor=bad)
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_foreign_family_cursor_is_422(client, app):
    _world(client)
    # A well-formed cursor minted by another endpoint family never resumes
    # this retrieval.
    foreign = pagination.encode_typed_cursor(
        app.state.actors_cursor_secret,
        pagination.ACTORS_CURSOR,
        {"actor_id": None, "name": None, "actor_type": None, "limit": 50,
         "offset": 1},
    )
    resp = _list(client, cursor=foreign)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    audit_cursor = pagination.encode_typed_cursor(
        app.state.audit_events_cursor_secret,
        pagination.AUDIT_EVENTS_CURSOR,
        {
            "event_type": None,
            "resource_id": None,
            "from": None,
            "to": None,
            "limit": 50,
            "offset": 1,
        },
    )
    assert _list(client, cursor=audit_cursor).status_code == 422


def test_cursor_with_wrong_claim_set_is_422(client, app):
    _world(client)
    # Correct family marker, HMAC, and base64, but a payload missing a bound
    # claim is rejected rather than trusted.
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "id": None,
                "content_id": None,
                "relation_type": None,
                "limit": 50,
                "offset": 1,
            },
            separators=(",", ":"),
        ).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(
        hmac.new(
            app.state.content_relations_cursor_secret,
            f"rl1.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    resp = _list(client, cursor=f"rl1.{payload}.{sig}")
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
        resp = method(RELATIONS_PATH)
        assert resp.status_code == 405
        assert resp.json()["error"]["code"] == "method_not_allowed"


def test_post_remains_the_relation_creation_route(client):
    # The new read entry does not shadow or alter relation creation.
    create_actor(client)
    a = _create_content(client, DIGEST_A)
    b = _create_content(client, DIGEST_B)
    payload = {
        "content_id": b["id"],
        "parent_content_id": a["id"],
        "relation_type": "version_of",
    }
    first = client.post(RELATIONS_PATH, json=payload)
    assert first.status_code == 201
    repeat = client.post(RELATIONS_PATH, json=payload)
    assert repeat.status_code == 200
    assert repeat.json() == first.json()


def test_detail_and_per_content_reads_are_unchanged(client):
    (a, b, c, d), relations = _world(client)
    detail = client.get(f"{RELATIONS_PATH}/{relations[0]['id']}")
    assert detail.status_code == 200
    assert detail.json() == relations[0]
    missing = client.get(f"{RELATIONS_PATH}/rel_ghost")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "content_relation_not_found"

    per_content = client.get(f"/v1/contents/{c['id']}/relations")
    assert per_content.status_code == 200
    assert [item["id"] for item in per_content.json()["items"]] == [
        relations[1]["id"],
        relations[2]["id"],
        relations[3]["id"],
    ]


# --- Read-only guarantee ---------------------------------------------------------


def test_queries_and_failures_write_nothing(client, db_session):
    (a, b, c, d), relations = _world(client)
    events_before = _audit_count(db_session)
    relations_before = db_session.scalar(
        select(func.count()).select_from(ContentRelation)
    )

    assert _list(client).status_code == 200
    assert _list(client, id="rel_ghost").status_code == 200
    assert _list(client, content_id="cnt_ghost").status_code == 200
    assert (
        _list(
            client, limit="2", **{"from": "2026-01-01T00:00:00Z"}
        ).status_code
        == 200
    )
    all_items, _pages, count = _walk_pages(client, limit="1")
    assert count == len(relations) == 4
    assert len(all_items) == 4

    # Failures are read-only as well.
    assert _list(client, limit="0").status_code == 422
    assert _list(client, id=" ").status_code == 422
    assert _list(client, relation_type="copied_from").status_code == 422
    assert _list(client, **{"from": "not-a-time"}).status_code == 422
    assert _list(client, cursor="bad").status_code == 422
    assert (
        client.request("GET", RELATIONS_PATH, content=b"{}").status_code == 422
    )

    assert (
        db_session.scalar(select(func.count()).select_from(ContentRelation))
        == relations_before
    )
    assert _audit_count(db_session) == events_before
