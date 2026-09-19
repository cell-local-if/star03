"""Tests for filtered, cursor-paginated lineage queries.

Covers GET /v1/contents/{content_id}/lineage with ``relation_type``,
``min_depth``, ``limit``, and ``cursor``: filtering never prunes traversal
reachability or changes shortest depths, convergence keeps the first-
discovery edge's type, cycles still terminate, pages concatenate without
gaps or duplicates while ``count`` stays the filtered total, cursors are
opaque/HMAC-bound to every effective query parameter (tampered, expired,
foreign, or mismatched cursors are 422), empty results and boundaries are
deterministic, and the extended endpoint stays strictly read-only. All
fixtures are deterministic and offline.
"""

from __future__ import annotations

import base64
import hashlib
import hmac as hmac_mod
import json
import secrets

from sqlalchemy import func, select

from provenance import pagination
from provenance.ids import content_relation_id
from provenance.models import AuditEvent, Content, ContentRelation
from tests.helpers import create_actor


def _digest_for(name: str) -> str:
    return hashlib.sha256(f"lineage-page-{name}".encode()).hexdigest()


def _create_content(client, name, actor_id="org-1"):
    resp = client.post(
        "/v1/contents",
        json={
            "digest_algorithm": "sha256",
            "digest_hex": _digest_for(name),
            "media_type": "image/png",
            "title": name,
            "actor_id": actor_id,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _edge(client, child, parent, relation_type="version_of"):
    resp = client.post(
        "/v1/content-relations",
        json={
            "content_id": child,
            "parent_content_id": parent,
            "relation_type": relation_type,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _lineage(client, content_id, direction="ancestors", **params):
    return client.get(
        f"/v1/contents/{content_id}/lineage",
        params={"direction": direction, **params},
    )


def _titles(items):
    return [item["title"] for item in items]


def _titles_and_depths(items):
    return [(item["title"], item["depth"]) for item in items]


def _chain(client, names, types=()):
    """Edges names[i+1] -> names[i]; ``types`` optionally fixes each type."""
    create_actor(client)
    ids = {name: _create_content(client, name)["id"] for name in names}
    for index, (child, parent) in enumerate(zip(names[1:], names[0:])):
        relation_type = types[index] if index < len(types) else "version_of"
        _edge(client, ids[child], ids[parent], relation_type)
    return ids


def _alternating_types(count):
    return ["version_of" if i % 2 == 0 else "derived_from" for i in range(count)]


def _walk_pages(client, content_id, direction="ancestors", **params):
    """Follow next_cursor until exhausted; return (all_items, pages, count)."""
    pages = []
    all_items = []
    count = None
    cursor = None
    seen_cursors = []
    for _ in range(100):
        query = {**params}
        if cursor is not None:
            query["cursor"] = cursor
        resp = _lineage(client, content_id, direction, **query)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        count = body["count"]
        pages.append(body["items"])
        all_items.extend(body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            break
        seen_cursors.append(cursor)
    return all_items, pages, count, seen_cursors


# --- relation_type filtering -------------------------------------------------


def test_unfiltered_response_shape_now_carries_null_cursor(client):
    ids = _chain(client, ["a", "b"])
    body = _lineage(client, ids["b"]).json()
    assert set(body) == {"items", "count", "next_cursor"}
    assert body["count"] == 1
    assert body["next_cursor"] is None


def test_relation_type_filters_returned_items_only(client):
    # Diamond: edges are created in an order that fixes c's first-discovery
    # edge to a -> c (version_of), regardless of b -> c's type.
    ids = _chain(client, ["x"])
    for name in ("a", "b", "c", "d"):
        ids[name] = _create_content(client, name)["id"]
    _edge(client, ids["a"], ids["c"], "version_of")
    _edge(client, ids["b"], ids["c"], "derived_from")
    _edge(client, ids["d"], ids["a"], "version_of")
    _edge(client, ids["d"], ids["b"], "derived_from")

    body = _lineage(client, ids["d"], relation_type="derived_from").json()
    assert _titles_and_depths(body["items"]) == [("b", 1)]
    assert body["count"] == 1
    assert body["next_cursor"] is None

    body = _lineage(client, ids["d"], relation_type="version_of").json()
    # c survives the filter at depth 2: its first-discovery edge (a -> c,
    # created before b -> c) is version_of.
    assert _titles_and_depths(body["items"]) == [("a", 1), ("c", 2)]
    assert body["count"] == 2


def test_filter_does_not_prune_traversal_through_filtered_nodes(client):
    # o -> a is derived_from; a -> x is version_of. Even when only
    # version_of nodes are returned, the walk must still pass through the
    # filtered a and reach x at its true depth of 2.
    ids = _chain(client, ["x", "a", "o"], ["version_of", "derived_from"])
    body = _lineage(client, ids["o"], relation_type="version_of").json()
    assert _titles_and_depths(body["items"]) == [("x", 2)]


def test_filter_uses_first_discovery_edge_not_any_path(client):
    # x is directly reachable through a derived_from edge at depth 1 and
    # also through a longer version_of path (o -> a -> x). Shortest depth
    # wins and the discovering edge is derived_from, so version_of filtering
    # excludes x even though a version_of path exists.
    create_actor(client)
    ids = {name: _create_content(client, name)["id"] for name in ("o", "a", "x")}
    # Direct edges are created first so x's first-discovery edge is the
    # derived_from one; the depth-2 version_of edge is added last.
    _edge(client, ids["o"], ids["x"], "derived_from")
    _edge(client, ids["o"], ids["a"], "derived_from")
    _edge(client, ids["a"], ids["x"], "version_of")

    assert _titles_and_depths(_lineage(client, ids["o"]).json()["items"]) == [
        ("x", 1),
        ("a", 1),
    ]
    body = _lineage(client, ids["o"], relation_type="version_of").json()
    assert body["items"] == []
    assert body["count"] == 0
    derived = _lineage(client, ids["o"], relation_type="derived_from").json()
    assert _titles_and_depths(derived["items"]) == [("x", 1), ("a", 1)]


def test_filter_works_in_descendants_direction(client):
    ids = _chain(client, ["x", "a", "o"], ["version_of", "derived_from"])
    # From x: descendants are a (edge a -> x version_of) at depth 1 and
    # o (edge o -> a derived_from) at depth 2.
    version = _lineage(
        client, ids["x"], "descendants", relation_type="version_of"
    ).json()
    assert _titles_and_depths(version["items"]) == [("a", 1)]
    derived = _lineage(
        client, ids["x"], "descendants", relation_type="derived_from"
    ).json()
    assert _titles_and_depths(derived["items"]) == [("o", 2)]


# --- min_depth filtering ------------------------------------------------------


def test_min_depth_truncates_the_shallow_side_only(client):
    names = [f"n{i}" for i in range(12)]  # n11 -> ... -> n0
    ids = _chain(client, names)
    body = _lineage(client, ids["n11"], max_depth=4, min_depth=2).json()
    assert _titles_and_depths(body["items"]) == [
        ("n9", 2),
        ("n8", 3),
        ("n7", 4),
    ]
    assert body["count"] == 3


def test_min_depth_equal_to_max_depth_returns_single_level(client):
    ids = _chain(client, ["a", "b", "c"])
    body = _lineage(client, ids["c"], max_depth=2, min_depth=2).json()
    assert _titles_and_depths(body["items"]) == [("a", 2)]


def test_min_depth_defaults_to_one_and_matches_unfiltered(client):
    names = [f"n{i}" for i in range(12)]
    ids = _chain(client, names)
    defaulted = _lineage(client, ids["n11"]).json()
    explicit = _lineage(client, ids["n11"], min_depth=1).json()
    assert defaulted == explicit


def test_min_depth_combined_with_relation_type(client):
    # n11 -> n10 ... -> n0 with alternating edge types. The discovering edge
    # to nk is types[11-k]; version_of at even indices.
    names = [f"n{i}" for i in range(12)]
    ids = _chain(client, names, _alternating_types(11))
    body = _lineage(
        client,
        ids["n11"],
        max_depth=32,
        min_depth=4,
        relation_type="version_of",
    ).json()
    assert _titles_and_depths(body["items"]) == [
        ("n6", 5),
        ("n4", 7),
        ("n2", 9),
        ("n0", 11),
    ]
    assert body["count"] == 4


# --- convergence + cycles under filters --------------------------------------


def test_convergence_first_discovery_edge_type_determines_membership(client):
    ids = _chain(client, ["x"])
    for name in ("a", "b", "c", "d"):
        ids[name] = _create_content(client, name)["id"]
    # Creation order: d -> a (derived), d -> b (version), then b -> c
    # (version) precedes a -> c (derived), so c is discovered via b.
    _edge(client, ids["d"], ids["a"], "derived_from")
    _edge(client, ids["d"], ids["b"], "version_of")
    _edge(client, ids["b"], ids["c"], "version_of")
    _edge(client, ids["a"], ids["c"], "derived_from")

    derived = _lineage(client, ids["d"], relation_type="derived_from").json()
    assert _titles(derived["items"]) == ["a"]
    version = _lineage(client, ids["d"], relation_type="version_of").json()
    assert _titles_and_depths(version["items"]) == [("b", 1), ("c", 2)]


def test_cycle_terminates_and_filters_normally(client, db_session):
    create_actor(client)
    ids = {name: _create_content(client, name)["id"] for name in ("a", "b", "c")}
    _edge(client, ids["b"], ids["a"], "version_of")  # b -> a
    _edge(client, ids["c"], ids["b"], "derived_from")  # c -> b
    db_session.add(
        ContentRelation(
            id=content_relation_id(ids["a"], ids["c"], "version_of"),
            content_id=ids["a"],
            parent_content_id=ids["c"],
            relation_type="version_of",
        )
    )
    db_session.commit()

    derived = _lineage(client, ids["c"], relation_type="derived_from").json()
    assert _titles_and_depths(derived["items"]) == [("b", 1)]
    version = _lineage(client, ids["c"], relation_type="version_of").json()
    assert _titles_and_depths(version["items"]) == [("a", 2)]
    paged = _lineage(
        client, ids["c"], relation_type="version_of", limit=1
    ).json()
    assert _titles(paged["items"]) == ["a"]
    assert paged["next_cursor"] is None


# --- pagination continuity ----------------------------------------------------


def test_pagination_concatenates_without_gaps_or_duplicates(client):
    names = [f"n{i}" for i in range(12)]
    ids = _chain(client, names, _alternating_types(11))
    all_items, pages, count, _ = _walk_pages(
        client, ids["n11"], max_depth=32, limit=3
    )

    assert [len(page) for page in pages] == [3, 3, 3, 2]
    assert count == 11
    returned_ids = [item["id"] for item in all_items]
    assert len(returned_ids) == len(set(returned_ids)) == 11
    # Concatenated pages reproduce the canonical depth/discovery order.
    assert _titles_and_depths(all_items) == _titles_and_depths(
        _lineage(client, ids["n11"], max_depth=32).json()["items"]
    )


def test_count_is_the_filtered_total_on_every_page(client):
    names = [f"n{i}" for i in range(12)]
    ids = _chain(client, names, _alternating_types(11))
    all_items, pages, count, _ = _walk_pages(
        client,
        ids["n11"],
        max_depth=32,
        relation_type="version_of",
        min_depth=3,
        limit=2,
    )
    assert count == 5  # n8,n6,n4,n2,n0
    assert [len(page) for page in pages] == [2, 2, 1]
    assert _titles(all_items) == ["n8", "n6", "n4", "n2", "n0"]


def test_last_page_cursor_is_null_when_pages_divide_exactly(client):
    ids = _chain(client, ["a", "b", "c", "d", "e"])
    body = _lineage(client, ids["e"], limit=2).json()
    assert body["count"] == 4
    assert body["next_cursor"] is not None
    second = _lineage(client, ids["e"], limit=2, cursor=body["next_cursor"]).json()
    assert len(second["items"]) == 2
    assert second["count"] == 4
    # Four items at two per page: the second page is the final page.
    assert second["next_cursor"] is None


def test_partial_final_page_cursor_is_null(client):
    ids = _chain(client, ["a", "b", "c", "d", "e"])
    body = _lineage(client, ids["e"], limit=3).json()
    assert _titles(body["items"]) == ["d", "c", "b"]
    second = _lineage(client, ids["e"], limit=3, cursor=body["next_cursor"]).json()
    assert _titles(second["items"]) == ["a"]
    assert second["next_cursor"] is None


def test_default_limit_is_fifty(client):
    create_actor(client)
    origin = _create_content(client, "origin")
    names = [f"p{i:02d}" for i in range(60)]
    ids = {name: _create_content(client, name)["id"] for name in names}
    for name in names:
        _edge(client, origin["id"], ids[name])

    first = _lineage(client, origin["id"]).json()
    assert len(first["items"]) == 50
    assert first["count"] == 60
    assert first["next_cursor"] is not None
    second = _lineage(
        client, origin["id"], cursor=first["next_cursor"]
    ).json()
    assert len(second["items"]) == 10
    assert second["count"] == 60
    assert second["next_cursor"] is None
    assert _titles(first["items"]) == names[:50]
    assert _titles(second["items"]) == names[50:]


def test_limit_one_pages_every_item(client):
    ids = _chain(client, ["a", "b", "c"])
    all_items, pages, count, cursors = _walk_pages(
        client, ids["c"], limit=1
    )
    assert [len(page) for page in pages] == [1, 1]
    assert count == 2
    assert _titles(all_items) == ["b", "a"]
    assert len(cursors) == 1


def test_pagination_is_stable_and_deterministic_across_walks(client):
    names = [f"n{i}" for i in range(12)]
    ids = _chain(client, names, _alternating_types(11))
    first_items, _, _, first_cursors = _walk_pages(
        client, ids["n11"], relation_type="version_of", limit=4
    )
    second_items, _, _, second_cursors = _walk_pages(
        client, ids["n11"], relation_type="version_of", limit=4
    )
    # Stateless HMAC cursors: identical requests mint identical tokens.
    assert first_cursors == second_cursors
    assert first_items == second_items


def test_reusing_a_cursor_replays_the_same_page(client):
    names = [f"n{i}" for i in range(12)]
    ids = _chain(client, names)
    first = _lineage(client, ids["n11"], max_depth=32, limit=2).json()
    cursor = first["next_cursor"]
    replay_one = _lineage(
        client, ids["n11"], max_depth=32, limit=2, cursor=cursor
    ).json()
    replay_two = _lineage(
        client, ids["n11"], max_depth=32, limit=2, cursor=cursor
    ).json()
    assert replay_one == replay_two
    assert _titles(replay_one["items"]) == ["n8", "n7"]


def test_empty_filtered_result_has_no_cursor(client):
    ids = _chain(client, ["a", "b"], ["derived_from"])
    body = _lineage(
        client, ids["b"], relation_type="version_of", limit=1
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


def test_cursor_past_end_returns_empty_page_with_total_count(client, app):
    ids = _chain(client, ["a", "b", "c"])
    token = pagination.encode_cursor(
        app.state.lineage_cursor_secret,
        {
            "content_id": ids["c"],
            "direction": "ancestors",
            "max_depth": 8,
            "min_depth": 1,
            "relation_type": None,
            "limit": 50,
            "offset": 99,
        },
    )
    body = _lineage(client, ids["c"], cursor=token).json()
    assert body["items"] == []
    assert body["count"] == 2
    assert body["next_cursor"] is None


# --- cursor integrity ---------------------------------------------------------


def test_tampered_or_malformed_cursors_are_validation_errors(client, app):
    ids = _chain(client, ["a", "b", "c"])
    good = _lineage(client, ids["c"], limit=1).json()["next_cursor"]
    tampered_payload = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    foreign = pagination.encode_cursor(
        secrets.token_bytes(32),
        {
            "content_id": ids["c"],
            "direction": "ancestors",
            "max_depth": 8,
            "min_depth": 1,
            "relation_type": None,
            "limit": 1,
            "offset": 1,
        },
    )
    for token in (
        "",
        "   ",
        "not-a-cursor",
        "v1.onlytwoparts",
        "v1.too.many.parts",
        "v0.x.y",
        "v2.x.y",
        tampered_payload,
        foreign,
    ):
        resp = _lineage(client, ids["c"], limit=1, cursor=token)
        assert resp.status_code == 422, repr(token)
        assert resp.json()["error"]["code"] == "validation_error"
        assert "items" not in resp.json()


def test_cursor_signed_with_old_format_marker_is_rejected(client, app):
    ids = _chain(client, ["a", "b"])
    # Manually mint a structurally valid token tagged with an old version.
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "content_id": ids["b"],
                "direction": "ancestors",
                "max_depth": 8,
                "min_depth": 1,
                "relation_type": None,
                "limit": 50,
                "offset": 1,
            }
        ).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(
        hmac_mod.new(
            app.state.lineage_cursor_secret,
            f"v0.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    resp = _lineage(client, ids["b"], cursor=f"v0.{payload}.{sig}")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_bound_to_every_effective_query_parameter(client):
    names = [f"n{i}" for i in range(12)]
    ids = _chain(client, names)
    cursor = _lineage(client, ids["n11"], limit=2).json()["next_cursor"]

    # Every mismatch (including a different path origin) is rejected.
    mismatches = [
        ("descendants", {"limit": 2}),
        ("ancestors", {"limit": 3}),
        ("ancestors", {"limit": 2, "max_depth": 4}),
        ("ancestors", {"limit": 2, "min_depth": 2}),
        ("ancestors", {"limit": 2, "relation_type": "version_of"}),
    ]
    for direction, params in mismatches:
        resp = _lineage(
            client, ids["n11"], direction, cursor=cursor, **params
        )
        assert resp.status_code == 422, (direction, params)
        assert resp.json()["error"]["code"] == "validation_error"

    resp = _lineage(client, ids["n10"], limit=2, cursor=cursor)
    assert resp.status_code == 422


def test_cursor_accepts_explicit_params_equal_to_cursor_defaults(client):
    names = [f"n{i}" for i in range(6)]
    ids = _chain(client, names)
    # First page uses all defaults (max_depth=8, min_depth=1, no filter).
    cursor = _lineage(client, ids["n5"], limit=2).json()["next_cursor"]
    # Repeating the same *effective* parameters explicitly must resume.
    resp = _lineage(
        client,
        ids["n5"],
        max_depth=8,
        min_depth=1,
        limit=2,
        cursor=cursor,
    )
    assert resp.status_code == 200, resp.text
    assert _titles(resp.json()["items"]) == ["n2", "n1"]

# --- parameter validation ----------------------------------------------------


def test_illegal_relation_type_is_validation_error(client):
    ids = _chain(client, ["a"])
    for value in ("", "   ", "version", "VERSION_OF", "version_of ", "parent_of"):
        resp = _lineage(client, ids["a"], relation_type=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_min_depth_boundaries_and_incompatibility(client):
    ids = _chain(client, ["a", "b", "c"])
    for value in ("0", "33", "-1", "1.5", "abc", "8.0", "  2", ""):
        resp = _lineage(client, ids["c"], min_depth=value)
        assert resp.status_code == 422, value
    resp = _lineage(client, ids["c"], max_depth=2, min_depth=3)
    assert resp.status_code == 422
    # Equality is allowed.
    assert (
        _lineage(client, ids["c"], max_depth=2, min_depth=2).status_code == 200
    )


def test_limit_boundaries(client):
    ids = _chain(client, ["a"])
    for value in ("0", "101", "-1", "1.0", "abc", ""):
        resp = _lineage(client, ids["a"], limit=value)
        assert resp.status_code == 422, value
    for value in (1, 100):
        assert _lineage(client, ids["a"], limit=value).status_code == 200


def test_repeated_new_parameters_are_validation_errors(client):
    ids = _chain(client, ["a", "b"])
    base = f"/v1/contents/{ids['b']}/lineage"
    for suffix in (
        "direction=ancestors&relation_type=version_of&relation_type=derived_from",
        "direction=ancestors&min_depth=1&min_depth=1",
        "direction=ancestors&limit=1&limit=2",
        "direction=ancestors&cursor=x&cursor=y",
    ):
        resp = client.get(f"{base}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_cursor_with_otherwise_valid_params_is_422(client):
    ids = _chain(client, ["a", "b"])
    resp = _lineage(
        client,
        ids["b"],
        relation_type="version_of",
        min_depth=1,
        limit=10,
        cursor="garbage",
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_unknown_origin_still_404_with_new_params(client):
    create_actor(client)
    resp = _lineage(
        client,
        "cnt_ghost",
        relation_type="version_of",
        min_depth=1,
        limit=10,
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "content_not_found"


# --- read-only guarantee ------------------------------------------------------


def test_filtered_paginated_queries_write_nothing(client, db_session):
    names = [f"n{i}" for i in range(6)]
    ids = _chain(client, names, _alternating_types(5))

    def counts():
        return (
            db_session.scalar(select(func.count()).select_from(Content)),
            db_session.scalar(select(func.count()).select_from(ContentRelation)),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    before = counts()
    cursor = None
    for _ in range(5):
        params = {
            "relation_type": "version_of",
            "min_depth": 1,
            "max_depth": 8,
            "limit": 2,
        }
        if cursor is not None:
            params["cursor"] = cursor
        resp = _lineage(client, ids["n5"], **params)
        assert resp.status_code == 200, resp.text
        cursor = resp.json()["next_cursor"]
        if cursor is None:
            break
    # Invalid extensions must not write anything either.
    _lineage(client, ids["n5"], relation_type="nope")
    _lineage(client, ids["n5"], min_depth=9, max_depth=2)
    _lineage(client, ids["n5"], limit=0)
    _lineage(client, ids["n5"], cursor="tampered")
    client.get(
        f"/v1/contents/{ids['n5']}/lineage?direction=ancestors&limit=1&limit=2"
    )
    db_session.expire_all()
    assert counts() == before
