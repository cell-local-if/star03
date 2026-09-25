"""Tests for the read-only source-actor search ``GET /v1/actors``.

Covers the optional exact ``id``/``name``/``type`` filters (non-empty, case-
and whitespace-sensitive, combined as logical AND; an unknown value is an
empty collection, never a 404), the compact UTF-8 JSON body with members in
exactly ``items``/``count``/``next_cursor`` order terminated by one newline,
the existing actor public view (never a private key, signature, payload,
content, or byte), stable creation order (``created_at`` with the monotonic
``seq`` tiebreaker, preserved across a restart), pure-decimal ``limit``
validation, the opaque HMAC-signed ``ac1`` cursor family (binding every
effective filter and the limit, resuming without duplication or omission, a
null cursor on the final page, and an empty page with the original count at
or past the tail), every 422 validation boundary, the strictly read-only
guarantee, and 405 method_not_allowed for change/deactivate/delete/replace
attempts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import select

from provenance import pagination
from provenance.models import Actor, AuditEvent
from tests.helpers import actor_payload, create_actor

ACTORS_PATH = "/v1/actors"


def _world(client):
    """Register three distinct source actors, in stable creation order."""
    one = create_actor(client, actor_id="org-1", name="Alpha Org", type="organization")
    two = create_actor(client, actor_id="org-2", name="Bob", type="person")
    three = create_actor(client, actor_id="org-3", name="Gamma Org", type="organization")
    return [one, two, three]


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


def _actor_rows(session):
    return session.execute(select(Actor)).scalars().all()


# --- Empty collection and response shape ---------------------------------------


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
    assert b'],"count":3,"next_cursor":null}' in raw
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    assert raw == expected


def test_items_carry_exactly_the_actor_public_view(client):
    actors = _world(client)
    resp = client.get(ACTORS_PATH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
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


def test_public_view_never_carries_private_or_raw_material_fields(client):
    _world(client)
    item = client.get(ACTORS_PATH).json()["items"][0]
    forbidden = {
        "private_key",
        "signature",
        "raw_signature",
        "payload",
        "content",
        "bytes",
        "seq",
    }
    assert not (forbidden & set(item))


# --- Filtering ------------------------------------------------------------------


def test_id_filter_is_an_exact_match(client):
    actors = _world(client)
    resp = client.get(ACTORS_PATH, params={"id": "org-2"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert [item["id"] for item in body["items"]] == [actors[1]["id"]]


def test_name_filter_is_an_exact_match(client):
    _world(client)
    resp = client.get(ACTORS_PATH, params={"name": "Bob"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["items"][0]["id"] == "org-2"


def test_type_filter_matches_every_actor_of_that_type(client):
    _world(client)
    resp = client.get(ACTORS_PATH, params={"type": "organization"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == ["org-1", "org-3"]


def test_filters_are_case_and_whitespace_sensitive(client):
    _world(client)
    for spelling in ("Org-1", "org-1 ", " org-1"):
        resp = client.get(ACTORS_PATH, params={"id": spelling})
        assert resp.status_code == 200, spelling
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}
    for spelling in ("bob", "Bob ", " Bob"):
        resp = client.get(ACTORS_PATH, params={"name": spelling})
        assert resp.status_code == 200, spelling
        assert resp.json()["count"] == 0
    for spelling in ("Person", " person", "Person "):
        resp = client.get(ACTORS_PATH, params={"type": spelling})
        assert resp.status_code == 200, spelling
        assert resp.json()["count"] == 0


def test_filters_combine_as_logical_and(client):
    _world(client)
    match = client.get(
        ACTORS_PATH,
        params={"id": "org-1", "name": "Alpha Org", "type": "organization"},
    )
    assert match.status_code == 200
    assert [item["id"] for item in match.json()["items"]] == ["org-1"]
    # A contradictory combination matches nothing rather than erroring.
    contradiction = client.get(
        ACTORS_PATH, params={"id": "org-1", "type": "person"}
    )
    assert contradiction.status_code == 200
    assert contradiction.json() == {"items": [], "count": 0, "next_cursor": None}


def test_unknown_filter_values_are_an_empty_collection_not_a_404(client):
    _world(client)
    for params in (
        {"id": "ghost"},
        {"name": "Nobody"},
        {"type": "device"},
        {"id": "org-1", "name": "Nope"},
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
    for bad in ("0", "101", "-1", "5.0", "5e0", " 5", "5 ", "five", "+5"):
        resp = client.get(ACTORS_PATH, params={"limit": bad})
        assert resp.status_code == 422, bad
        assert resp.json()["error"]["code"] == "validation_error", bad


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
        client.get(ACTORS_PATH, params=[("limit", "1"), ("limit", "2")]).status_code
        == 422
    )
    # ``actor_id`` is the creation field name, not a declared search parameter.
    for params in (
        {"actor_id": "org-1"},
        {"ids": "org-1"},
        {"names": "Bob"},
        {"offset": "1"},
    ):
        resp = client.get(ACTORS_PATH, params=params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_empty_body_is_422_rejected_before_any_read(client, db_session):
    _world(client)
    events_before = _audit_count(db_session)
    for body in (b"{}", b" ", b"{not valid json", b"[]", b"\t"):
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
    assert page_one["count"] == 3
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
    assert page_two["count"] == 3
    assert [item["id"] for item in page_two["items"]] == [actors[2]["id"]]
    assert page_two["next_cursor"] is None

    seen = [item["id"] for item in page_one["items"] + page_two["items"]]
    assert seen == [a["id"] for a in actors]


def test_filtered_pages_bind_all_filters(client):
    _world(client)
    resp = client.get(
        ACTORS_PATH,
        params={"type": "organization", "name": "Alpha Org", "id": "org-1", "limit": "1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["id"] == "org-1"
    assert body["next_cursor"] is None


def test_replayed_cursor_returns_the_same_page(client):
    _world(client)
    cursor = client.get(ACTORS_PATH, params={"limit": "1"}).json()["next_cursor"]
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
        {"id": None, "name": None, "type": None, "limit": 50, "offset": 3},
    )
    resp = client.get(ACTORS_PATH, params={"cursor": cursor})
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 3, "next_cursor": None}

    past = pagination.encode_typed_cursor(
        app.state.actors_cursor_secret,
        pagination.ACTORS_CURSOR,
        {"id": None, "name": None, "type": None, "limit": 50, "offset": 99},
    )
    resp = client.get(ACTORS_PATH, params={"cursor": past})
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 3, "next_cursor": None}


def test_cursor_binds_the_effective_filters_and_limit(client):
    _world(client)
    cursor = client.get(ACTORS_PATH, params={"limit": "2"}).json()["next_cursor"]
    assert cursor is not None
    for params in (
        {"limit": "3", "cursor": cursor},
        {"id": "org-1", "limit": "2", "cursor": cursor},
        {"name": "Bob", "limit": "2", "cursor": cursor},
        {"type": "person", "limit": "2", "cursor": cursor},
        {"cursor": cursor},
    ):
        resp = client.get(ACTORS_PATH, params=params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_blank_malformed_and_tampered_cursors_are_422(client):
    _world(client)
    valid = client.get(ACTORS_PATH, params={"limit": "1"}).json()["next_cursor"]
    assert valid is not None
    tampered = valid[:-1] + ("A" if valid[-1] != "A" else "B")
    for bad in ("", "   ", "not-a-cursor", tampered):
        resp = client.get(ACTORS_PATH, params={"cursor": bad})
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_foreign_family_cursor_is_422(client, app):
    _world(client)
    foreign = pagination.encode_typed_cursor(
        app.state.trust_policies_cursor_secret,
        pagination.TRUST_POLICIES_CURSOR,
        {"actor_id": None, "limit": 50, "offset": 1},
    )
    resp = client.get(ACTORS_PATH, params={"cursor": foreign})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Stable ordering ------------------------------------------------------------


def test_same_timestamp_ties_break_by_persistent_insertion_order(
    client, db_session
):
    # Two actors committed at exactly the same UTC timestamp: the monotonic
    # persistent sequence, not the id or the timestamp, fixes the order.
    instant = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    later_id = "zzz-actor"
    earlier_id = "aaa-actor"
    db_session.add(Actor(id=later_id, name="Z", type="person", created_at=instant))
    db_session.add(Actor(id=earlier_id, name="A", type="person", created_at=instant))
    db_session.commit()

    resp = client.get(ACTORS_PATH)
    assert resp.status_code == 200
    # Insertion (persistent) order wins over alphabetical id order.
    assert [item["id"] for item in resp.json()["items"]] == [later_id, earlier_id]


def test_actors_list_identically_across_an_app_restart(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    _world(file_client)
    first = file_client.get(ACTORS_PATH)
    assert first.status_code == 200

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as restarted_client:
        second = restarted_client.get(ACTORS_PATH)
        assert second.status_code == 200
        assert second.json() == first.json()


# --- Read-only guarantee ---------------------------------------------------------


def test_queries_and_failures_write_nothing(client, db_session):
    _world(client)
    events_before = _audit_count(db_session)
    ids_before = sorted(r.id for r in _actor_rows(db_session))

    assert client.get(ACTORS_PATH).status_code == 200
    assert client.get(ACTORS_PATH, params={"id": "ghost"}).status_code == 200
    assert client.get(ACTORS_PATH, params={"limit": "1"}).status_code == 200
    assert client.get(ACTORS_PATH, params={"limit": "0"}).status_code == 422
    assert client.get(ACTORS_PATH, params={"cursor": "bad"}).status_code == 422
    assert (
        client.request("PUT", ACTORS_PATH, json=actor_payload()).status_code == 405
    )

    assert sorted(r.id for r in _actor_rows(db_session)) == ids_before
    assert _audit_count(db_session) == events_before


# --- Method boundary -------------------------------------------------------------


def test_change_methods_on_the_collection_are_method_not_allowed(client):
    _world(client)
    for method in ("PUT", "PATCH", "DELETE"):
        resp = client.request(
            method,
            ACTORS_PATH,
            json={"id": "org-1", "name": "New", "type": "person"},
        )
        assert resp.status_code == 405, method
        assert resp.json()["error"]["code"] == "method_not_allowed", method


def test_post_creation_semantics_are_unchanged(client):
    # Creation still works and still rejects a duplicate id with a 409.
    first = client.post("/v1/actors", json=actor_payload())
    assert first.status_code == 201
    duplicate = client.post("/v1/actors", json=actor_payload())
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "actor_already_exists"
