"""Tests for read-only actor retrieval.

Covers ``GET /v1/actors``: the empty-body contract, the exact
``items``/``count``/``next_cursor`` member order in compact UTF-8 JSON
terminated by one newline, the optional exact ``id``/``name``/``type``
filters (an unknown or nonexistent value is an empty collection, never a
404), stable creation ordering (``created_at`` with the persistent
insertion-order tiebreaker that survives an app restart), pure-decimal
``limit`` validation, the opaque HMAC-signed ``ac1`` cursor family
(binding every effective filter and the limit, resuming without
duplication or omission, a null cursor on the final page, and an empty
page with the original count at or past the tail), every 422 validation
boundary, the 405 rejection of PUT/PATCH/DELETE, and the strictly
read-only guarantee (no actor, resource, or audit writes on success,
empty results, repeated reads, or failures).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import select

from provenance import pagination
from provenance.models import Actor, AuditEvent
from tests.helpers import create_actor

ACTORS_PATH = "/v1/actors"


# --- Fixture-style setup ------------------------------------------------------


def _world(client):
    """Register four distinct actors, in stable creation order."""
    one = create_actor(
        client, actor_id="org-1", name="Example Org", type="organization"
    )
    two = create_actor(client, actor_id="p-1", name="Alice", type="person")
    three = create_actor(
        client, actor_id="org-2", name="Other Org", type="organization"
    )
    four = create_actor(client, actor_id="dev-1", name="Sensor", type="device")
    return [one, two, three, four]


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


def _actor_ids(session):
    return {row.id for row in session.execute(select(Actor.id)).all()}


# --- Empty collection and response shape --------------------------------------


def test_empty_registry_is_an_empty_collection(client):
    resp = client.get(ACTORS_PATH)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_response_is_compact_json_with_ordered_members_and_one_newline(client):
    _world(client)
    resp = client.get(ACTORS_PATH)
    assert resp.status_code == 200
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    # The top-level members appear in exactly this order.
    assert raw.startswith(b'{"items":[')
    assert b'],"count":4,"next_cursor":null}' in raw
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        .encode("utf-8")
        + b"\n"
    )
    assert raw == expected


def test_items_carry_exactly_the_actor_public_view(client):
    actors = _world(client)
    resp = client.get(ACTORS_PATH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 4
    assert body["next_cursor"] is None
    # Stable creation order, and each item is exactly the public view.
    assert [item["id"] for item in body["items"]] == [a["id"] for a in actors]
    for item, actor in zip(body["items"], actors):
        assert list(item) == ["id", "name", "type", "created_at"]
        assert item == actor
        created_at = datetime.fromisoformat(item["created_at"])
        assert created_at.tzinfo is not None
        assert created_at.utcoffset().total_seconds() == 0
        assert item["created_at"].endswith(("Z", "+00:00"))
    # No private, signature, payload, content, or byte field can ever appear.
    rendered = resp.content.decode()
    for forbidden in (
        "private",
        "signature",
        "payload",
        "public_key",
        "digest",
        "seq",
    ):
        assert forbidden not in rendered


# --- Stable ordering -----------------------------------------------------------


def test_items_follow_stable_creation_order(client):
    actors = _world(client)
    body = client.get(ACTORS_PATH).json()
    assert [item["id"] for item in body["items"]] == [a["id"] for a in actors]
    created = [datetime.fromisoformat(item["created_at"]) for item in body["items"]]
    assert created == sorted(created)


def test_same_timestamp_ties_break_by_persistence_order(client, db_session):
    # Force every actor onto one instant: the persistent insertion order
    # (rowid) must still yield a stable, restart-durable order.
    actors = _world(client)
    tie = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    for row in db_session.execute(select(Actor)).scalars():
        row.created_at = tie
    db_session.commit()

    body = client.get(ACTORS_PATH).json()
    assert [item["id"] for item in body["items"]] == [a["id"] for a in actors]


def test_ordering_is_identical_across_an_app_restart(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    actors = _world(file_client)
    first = file_client.get(ACTORS_PATH)
    assert first.status_code == 200

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as restarted_client:
        second = restarted_client.get(ACTORS_PATH)
        assert second.status_code == 200
        assert second.json() == first.json()
        assert [item["id"] for item in second.json()["items"]] == [
            a["id"] for a in actors
        ]


# --- Filtering -----------------------------------------------------------------


def test_id_filter_is_an_exact_match(client):
    actors = _world(client)
    resp = client.get(ACTORS_PATH, params={"id": "org-2"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["next_cursor"] is None
    assert [item["id"] for item in body["items"]] == [actors[2]["id"]]


def test_name_and_type_filters_match_exactly(client):
    _world(client)
    by_name = client.get(ACTORS_PATH, params={"name": "Alice"}).json()
    assert [item["id"] for item in by_name["items"]] == ["p-1"]
    assert by_name["count"] == 1
    by_type = client.get(ACTORS_PATH, params={"type": "organization"}).json()
    assert [item["id"] for item in by_type["items"]] == ["org-1", "org-2"]
    assert by_type["count"] == 2


def test_filters_combine_as_logical_and(client):
    _world(client)
    resp = client.get(
        ACTORS_PATH, params={"type": "organization", "name": "Other Org"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert [item["id"] for item in body["items"]] == ["org-2"]

    # A valid type combined with a non-matching name is an empty set.
    miss = client.get(
        ACTORS_PATH, params={"type": "organization", "name": "Alice"}
    ).json()
    assert miss == {"items": [], "count": 0, "next_cursor": None}


def test_filters_are_case_and_whitespace_sensitive(client):
    _world(client)
    for params in (
        {"id": "Org-1"},
        {"id": "org-1 "},
        {"id": " org-1"},
        {"name": "alice"},
        {"name": " Alice"},
        {"type": "Person"},
        {"type": "person "},
    ):
        resp = client.get(ACTORS_PATH, params=params)
        assert resp.status_code == 200, params
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_unknown_or_nonexistent_filter_value_is_an_empty_collection(client):
    _world(client)
    for params in (
        {"id": "ghost"},
        {"name": "Nobody"},
        {"type": "unicorn"},
        {"id": "ghost", "name": "Alice", "type": "person"},
    ):
        resp = client.get(ACTORS_PATH, params=params)
        assert resp.status_code == 200, params
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_blank_filters_are_422(client):
    _world(client)
    for field in ("id", "name", "type"):
        for blank in ("", "   ", "\t"):
            resp = client.get(ACTORS_PATH, params={field: blank})
            assert resp.status_code == 422, (field, repr(blank))
            assert resp.json()["error"]["code"] == "validation_error"


# --- limit validation -----------------------------------------------------------


def test_limit_boundaries_one_and_one_hundred_are_accepted(client):
    _world(client)
    assert client.get(ACTORS_PATH, params={"limit": "1"}).status_code == 200
    assert client.get(ACTORS_PATH, params={"limit": "100"}).status_code == 200


def test_limit_must_be_a_pure_decimal_integer_in_range(client):
    _world(client)
    for bad in ("0", "101", "-1", "5.0", "5e0", " 5", "5 ", "five", "+5", ""):
        resp = client.get(ACTORS_PATH, params={"limit": bad})
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_default_limit_is_fifty(client):
    for index in range(51):
        create_actor(client, actor_id=f"org-bulk-{index:03d}", name=f"N{index}")
    resp = client.get(ACTORS_PATH)
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
            ACTORS_PATH, params=[("id", "org-1"), ("id", "org-2")]
        ).status_code
        == 422
    )
    assert (
        client.get(
            ACTORS_PATH, params=[("type", "person"), ("type", "device")]
        ).status_code
        == 422
    )
    assert (
        client.get(
            ACTORS_PATH, params=[("limit", "1"), ("limit", "2")]
        ).status_code
        == 422
    )
    for params in (
        {"actor_id": "org-1"},
        {"types": "person"},
        {"offset": "1"},
        {"q": "x"},
    ):
        resp = client.get(ACTORS_PATH, params=params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_empty_body_is_422_rejected_before_any_read(client, db_session):
    _world(client)
    events_before = _audit_count(db_session)
    for body in (b"{}", b" ", b"{not valid json", b"[]", b"\n"):
        resp = client.request(
            "GET",
            ACTORS_PATH,
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"
    assert _audit_count(db_session) == events_before


# --- Pagination -----------------------------------------------------------------


def test_pages_resume_without_duplication_or_omission(client):
    actors = _world(client)
    first = client.get(ACTORS_PATH, params={"limit": "2"})
    assert first.status_code == 200
    page_one = first.json()
    assert page_one["count"] == 4
    assert [item["id"] for item in page_one["items"]] == [
        actors[0]["id"],
        actors[1]["id"],
    ]
    cursor = page_one["next_cursor"]
    assert cursor is not None
    assert cursor.startswith("ac1.")

    second = client.get(ACTORS_PATH, params={"limit": "2", "cursor": cursor})
    assert second.status_code == 200
    page_two = second.json()
    assert page_two["count"] == 4
    assert [item["id"] for item in page_two["items"]] == [
        actors[2]["id"],
        actors[3]["id"],
    ]
    assert page_two["next_cursor"] is None

    seen = [item["id"] for item in page_one["items"] + page_two["items"]]
    assert seen == [a["id"] for a in actors]


def test_count_covers_every_page_including_a_filtered_one(client):
    _world(client)
    page = client.get(
        ACTORS_PATH, params={"type": "organization", "limit": "1"}
    ).json()
    assert page["count"] == 2
    assert len(page["items"]) == 1
    cursor = page["next_cursor"]
    assert cursor is not None
    final = client.get(
        ACTORS_PATH,
        params={"type": "organization", "limit": "1", "cursor": cursor},
    ).json()
    assert final["count"] == 2
    assert len(final["items"]) == 1
    assert final["next_cursor"] is None


def test_replayed_cursor_returns_the_same_page(client):
    _world(client)
    first = client.get(ACTORS_PATH, params={"limit": "1"}).json()
    cursor = first["next_cursor"]
    assert cursor is not None
    page = client.get(ACTORS_PATH, params={"limit": "1", "cursor": cursor})
    replay = client.get(ACTORS_PATH, params={"limit": "1", "cursor": cursor})
    assert page.status_code == 200
    assert replay.status_code == 200
    assert replay.content == page.content


def test_cursor_at_or_past_the_tail_returns_empty_page_with_count(client, app):
    _world(client)
    cursor = pagination.encode_typed_cursor(
        app.state.actors_cursor_secret,
        pagination.ACTORS_CURSOR,
        {"actor_id": None, "name": None, "actor_type": None, "limit": 50,
         "offset": 4},
    )
    resp = client.get(ACTORS_PATH, params={"cursor": cursor})
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 4, "next_cursor": None}

    past = pagination.encode_typed_cursor(
        app.state.actors_cursor_secret,
        pagination.ACTORS_CURSOR,
        {"actor_id": None, "name": None, "actor_type": None, "limit": 50,
         "offset": 99},
    )
    resp = client.get(ACTORS_PATH, params={"cursor": past})
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 4, "next_cursor": None}


def test_cursor_binds_every_effective_filter_and_limit(client):
    _world(client)
    cursor = client.get(ACTORS_PATH, params={"limit": "2"}).json()["next_cursor"]
    assert cursor is not None
    # A different limit, a new filter, or a dropped filter all mismatch.
    for params in (
        {"limit": "3", "cursor": cursor},
        {"type": "person", "limit": "2", "cursor": cursor},
        {"cursor": cursor},
    ):
        resp = client.get(ACTORS_PATH, params=params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"

    # A cursor minted under a filter cannot resume a different filter.
    person_cursor = client.get(
        ACTORS_PATH, params={"type": "person", "limit": "1"}
    ).json()["next_cursor"]
    # Single match -> final page, no cursor; forge a mid-filter cursor instead.
    mid = client.get(
        ACTORS_PATH, params={"type": "organization", "limit": "1"}
    ).json()["next_cursor"]
    assert person_cursor is None
    assert mid is not None
    mismatch = client.get(
        ACTORS_PATH, params={"type": "device", "limit": "1", "cursor": mid}
    )
    assert mismatch.status_code == 422
    changed_id = client.get(
        ACTORS_PATH, params={"id": "org-1", "limit": "1", "cursor": mid}
    )
    assert changed_id.status_code == 422


def test_blank_malformed_and_tampered_cursors_are_422(client):
    _world(client)
    valid = client.get(ACTORS_PATH, params={"limit": "1"}).json()["next_cursor"]
    assert valid is not None
    tampered = valid[:-1] + ("A" if valid[-1] != "A" else "B")
    for bad in ("", "   ", "not-a-cursor", "ac1", "ac1.abc", tampered):
        resp = client.get(ACTORS_PATH, params={"cursor": bad})
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_foreign_family_cursor_is_422(client, app):
    _world(client)
    # A well-formed cursor minted by another endpoint family never resumes
    # this retrieval.
    foreign = pagination.encode_typed_cursor(
        app.state.trust_policies_cursor_secret,
        pagination.TRUST_POLICIES_CURSOR,
        {"actor_id": None, "limit": 50, "offset": 1},
    )
    resp = client.get(ACTORS_PATH, params={"cursor": foreign})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    claims_cursor = pagination.encode_typed_cursor(
        app.state.claims_cursor_secret,
        pagination.CLAIMS_CURSOR,
        {
            "content_id": None,
            "actor_id": None,
            "claim_type": None,
            "payload_digest_hex": None,
            "limit": 50,
            "offset": 1,
        },
    )
    resp = client.get(ACTORS_PATH, params={"cursor": claims_cursor})
    assert resp.status_code == 422


def test_cursor_with_wrong_claim_set_is_422(client, app):
    _world(client)
    # Correct family marker, HMAC, and base64, but a foreign claim payload.
    token = pagination.encode_typed_cursor(
        app.state.actors_cursor_secret,
        pagination.ACTORS_CURSOR,
        {"actor_id": None, "name": None, "actor_type": None, "limit": 50,
         "offset": 1},
    )
    assert client.get(ACTORS_PATH, params={"cursor": token}).status_code == 200
    # Hand-built token of the same family whose payload drops a bound claim.
    import base64
    import hashlib
    import hmac
    import json

    payload = base64.urlsafe_b64encode(
        json.dumps(
            {"actor_id": None, "name": None, "limit": 50, "offset": 1},
            separators=(",", ":"),
        ).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(
        hmac.new(
            app.state.actors_cursor_secret,
            f"ac1.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    resp = client.get(ACTORS_PATH, params={"cursor": f"ac1.{payload}.{sig}"})
    assert resp.status_code == 422


# --- Method boundary ------------------------------------------------------------


def test_put_patch_and_delete_are_405_method_not_allowed(client):
    _world(client)
    for method in (client.put, client.patch, client.delete):
        resp = method(ACTORS_PATH)
        assert resp.status_code == 405
        assert resp.json()["error"]["code"] == "method_not_allowed"


def test_post_remains_the_actor_creation_route(client):
    # The new read entry does not shadow or alter actor creation semantics.
    resp = client.post(
        ACTORS_PATH, json={"id": "org-1", "name": "Example Org",
                           "type": "organization"}
    )
    assert resp.status_code == 201
    again = client.post(
        ACTORS_PATH, json={"id": "org-1", "name": "Example Org",
                           "type": "organization"}
    )
    assert again.status_code == 409
    assert again.json()["error"]["code"] == "actor_already_exists"


# --- Read-only guarantee ---------------------------------------------------------


def test_queries_and_failures_write_nothing(client, db_session):
    actors = _world(client)
    events_before = _audit_count(db_session)
    ids_before = _actor_ids(db_session)

    assert client.get(ACTORS_PATH).status_code == 200
    assert client.get(ACTORS_PATH, params={"id": "ghost"}).status_code == 200
    assert client.get(ACTORS_PATH, params={"type": "person", "limit": "1"}).status_code == 200
    assert client.get(ACTORS_PATH, params={"limit": "0"}).status_code == 422
    assert client.get(ACTORS_PATH, params={"id": " "}).status_code == 422
    assert client.get(ACTORS_PATH, params={"cursor": "bad"}).status_code == 422
    assert client.request("GET", ACTORS_PATH, content=b"{}").status_code == 422

    assert _actor_ids(db_session) == ids_before == {a["id"] for a in actors}
    assert _audit_count(db_session) == events_before
