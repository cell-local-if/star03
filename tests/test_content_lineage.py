"""Tests for the read-only multi-hop lineage endpoint.

Covers GET /v1/contents/{content_id}/lineage: bidirectional traversal,
convergence/deduplication at shortest depth, depth truncation, depth-then-
creation ordering, bounded termination over anomalous cyclic history,
parameter validation, the missing-origin 404, and the read-only guarantee.
All fixtures are deterministic and offline.
"""

from __future__ import annotations

import hashlib

from sqlalchemy import func, select

from provenance.ids import content_relation_id
from provenance.models import AuditEvent, Content, ContentRelation
from tests.helpers import create_actor

# Fixed, distinct, offline digests derived deterministically from node names.
def _digest_for(name: str) -> str:
    return hashlib.sha256(f"lineage-{name}".encode()).hexdigest()


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


def _lineage(client, content_id, direction, **params):
    return client.get(
        f"/v1/contents/{content_id}/lineage",
        params={"direction": direction, **params},
    )


def _setup_chain(client, names):
    """Create contents and edges each -> the previous; return id-by-name."""
    create_actor(client)
    ids = {name: _create_content(client, name)["id"] for name in names}
    for child, parent in zip(names[1:], names[:-1]):
        _edge(client, ids[child], ids[parent])
    return ids


def _ordered(items):
    return [(item["title"], item["depth"]) for item in items]


# --- Bidirectional traversal ------------------------------------------------


def test_ancestors_walk_content_to_parent_excluding_origin(client):
    ids = _setup_chain(client, ["a", "b", "c"])  # c -> b -> a
    resp = _lineage(client, ids["c"], "ancestors")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert _ordered(body["items"]) == [("b", 1), ("a", 2)]
    assert all(item["id"] != ids["c"] for item in body["items"])


def test_descendants_traverse_reversed_edges_excluding_origin(client):
    ids = _setup_chain(client, ["a", "b", "c"])  # c -> b -> a
    resp = _lineage(client, ids["a"], "descendants")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert _ordered(body["items"]) == [("b", 1), ("c", 2)]
    assert all(item["id"] != ids["a"] for item in body["items"])


def test_lineage_item_is_full_public_content_view_plus_depth(client):
    ids = _setup_chain(client, ["a", "b"])
    resp = _lineage(client, ids["b"], "ancestors")
    item = resp.json()["items"][0]
    assert set(item) == {
        "id",
        "digest_algorithm",
        "digest_hex",
        "media_type",
        "title",
        "actor_id",
        "created_at",
        "depth",
    }
    fetched = client.get(f"/v1/contents/{ids['a']}").json()
    for field in (
        "id",
        "digest_algorithm",
        "digest_hex",
        "media_type",
        "title",
        "actor_id",
        "created_at",
    ):
        assert item[field] == fetched[field]
    assert item["depth"] == 1


def test_empty_traversal_returns_empty_set_not_404(client):
    ids = _setup_chain(client, ["a", "b"])
    # a is a root: no ancestors; b is a leaf: no descendants.
    for origin, direction in ((ids["a"], "ancestors"), (ids["b"], "descendants")):
        resp = _lineage(client, origin, direction)
        assert resp.status_code == 200
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_unknown_origin_is_content_not_found(client):
    create_actor(client)
    for direction in ("ancestors", "descendants"):
        resp = _lineage(client, "cnt_ghost", direction)
        assert resp.status_code == 404
        error = resp.json()["error"]
        assert error["code"] == "content_not_found"
        assert error["details"]["content_id"] == "cnt_ghost"


# --- Convergence and shortest depth -----------------------------------------


def test_converging_paths_dedup_content_at_shortest_depth(client):
    # Diamond: d -> a -> c and d -> b -> c, with a -> c created before b -> c.
    ids = _setup_chain(client, ["x"])  # actor + an unrelated content node
    for name in ("a", "b", "c", "d"):
        ids[name] = _create_content(client, name)["id"]
    _edge(client, ids["a"], ids["c"])
    _edge(client, ids["b"], ids["c"], "derived_from")
    _edge(client, ids["d"], ids["a"])
    _edge(client, ids["d"], ids["b"], "derived_from")

    resp = _lineage(client, ids["d"], "ancestors")
    assert resp.status_code == 200
    assert _ordered(resp.json()["items"]) == [
        ("a", 1),
        ("b", 1),
        ("c", 2),
    ]

    # Reverse direction converges too: c reaches a, b at depth 1 and d at 2.
    resp = _lineage(client, ids["c"], "descendants")
    assert _ordered(resp.json()["items"]) == [
        ("a", 1),
        ("b", 1),
        ("d", 2),
    ]


def test_content_reachable_at_two_depths_keeps_the_shortest_once(client):
    # o derives directly from a (depth 1) and indirectly o -> x -> a.
    create_actor(client)
    ids = {name: _create_content(client, name)["id"] for name in ("o", "a", "x")}
    _edge(client, ids["o"], ids["x"])
    _edge(client, ids["o"], ids["a"], "derived_from")
    _edge(client, ids["x"], ids["a"])

    resp = _lineage(client, ids["o"], "ancestors")
    items = resp.json()["items"]
    a_entries = [item for item in items if item["id"] == ids["a"]]
    assert len(a_entries) == 1
    assert a_entries[0]["depth"] == 1
    assert [(item["title"], item["depth"]) for item in items] == [
        ("x", 1),
        ("a", 1),
    ]


# --- Truncation --------------------------------------------------------------


def test_max_depth_truncates_beyond_the_limit(client):
    ids = _setup_chain(client, ["a", "b", "c", "d", "e"])  # e .. -> a
    resp = _lineage(client, ids["e"], "ancestors", max_depth=2)
    assert resp.status_code == 200
    assert _ordered(resp.json()["items"]) == [("d", 1), ("c", 2)]


def test_max_depth_one_returns_only_direct_neighbors(client):
    ids = _setup_chain(client, ["a", "b", "c"])
    resp = _lineage(client, ids["c"], "ancestors", max_depth=1)
    assert _ordered(resp.json()["items"]) == [("b", 1)]


def test_default_max_depth_is_eight(client):
    names = [f"0{i}"[-2:] for i in range(11)]  # 11 nodes, 10 edges
    ids = _setup_chain(client, names)
    resp = _lineage(client, ids[names[-1]], "ancestors")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 8
    assert [item["depth"] for item in items] == list(range(1, 9))


def test_max_depth_boundaries_are_accepted(client):
    ids = _setup_chain(client, ["a", "b"])
    for depth in (1, 32):
        resp = _lineage(client, ids["b"], "ancestors", max_depth=depth)
        assert resp.status_code == 200, resp.text


# --- Ordering ----------------------------------------------------------------


def test_within_level_order_uses_edge_creation_order_across_siblings(client):
    create_actor(client)
    # Contents are deliberately created in an order that differs from the
    # edge order: b exists before a, and x before y.
    ids = {
        name: _create_content(client, name)["id"] for name in ("o", "b", "a", "x", "y")
    }
    # Edges, in strict creation order:
    _edge(client, ids["o"], ids["a"])  # depth 1: a discovered first
    _edge(client, ids["o"], ids["b"])  # depth 1: b second
    _edge(client, ids["b"], ids["x"])  # b's depth-2 edge is created FIRST
    _edge(client, ids["a"], ids["y"])  # a's depth-2 edge is created after

    resp = _lineage(client, ids["o"], "ancestors")
    items = resp.json()["items"]
    # Depth 1 follows the edges out of o (a before b), not content order.
    # Depth 2 is globally ordered by discovering-edge creation: b -> x was
    # created before a -> y, so x precedes y even though a precedes b.
    assert _ordered(items) == [
        ("a", 1),
        ("b", 1),
        ("x", 2),
        ("y", 2),
    ]


# --- Cycles in anomalous history ---------------------------------------------


def test_cycle_in_anomalous_history_terminates_deduped(client, db_session):
    create_actor(client)
    ids = {name: _create_content(client, name)["id"] for name in ("a", "b", "c")}
    _edge(client, ids["b"], ids["a"])  # b -> a
    _edge(client, ids["c"], ids["b"])  # c -> b
    # Bypass the service cycle guard to simulate anomalous history: a -> c
    # closes c -> b -> a -> c.
    cyclic = ContentRelation(
        id=content_relation_id(ids["a"], ids["c"], "version_of"),
        content_id=ids["a"],
        parent_content_id=ids["c"],
        relation_type="version_of",
    )
    db_session.add(cyclic)
    db_session.commit()

    ancestors = _lineage(client, ids["c"], "ancestors")
    assert ancestors.status_code == 200
    assert _ordered(ancestors.json()["items"]) == [("b", 1), ("a", 2)]

    descendants = _lineage(client, ids["a"], "descendants")
    assert descendants.status_code == 200
    assert _ordered(descendants.json()["items"]) == [("b", 1), ("c", 2)]

    # Every node is reported at most once and the origin is never included.
    for body, origin in (
        (ancestors.json(), ids["c"]),
        (descendants.json(), ids["a"]),
    ):
        returned = [item["id"] for item in body["items"]]
        assert len(returned) == len(set(returned))
        assert origin not in returned


def test_max_depth_bounds_walk_even_with_a_cycle(client, db_session):
    create_actor(client)
    ids = {name: _create_content(client, name)["id"] for name in ("a", "b")}
    _edge(client, ids["b"], ids["a"])
    db_session.add(
        ContentRelation(
            id=content_relation_id(ids["a"], ids["b"], "derived_from"),
            content_id=ids["a"],
            parent_content_id=ids["b"],
            relation_type="derived_from",
        )
    )
    db_session.commit()
    # A direct a <-> b loop: depth limiting alone would keep the walk finite
    # even without the visited set.
    resp = _lineage(client, ids["a"], "ancestors", max_depth=32)
    assert resp.status_code == 200
    assert _ordered(resp.json()["items"]) == [("b", 1)]


# --- Parameter validation ----------------------------------------------------


def test_direction_is_required(client):
    ids = _setup_chain(client, ["a"])
    resp = client.get(f"/v1/contents/{ids['a']}/lineage")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_blank_or_illegal_direction_is_validation_error(client):
    ids = _setup_chain(client, ["a"])
    for value in ("", "   ", "up", "ANCESTORS", "ancestor", "descendant"):
        resp = _lineage(client, ids["a"], value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_max_depth_empty_non_integer_or_out_of_range_is_validation_error(client):
    ids = _setup_chain(client, ["a"])
    for value in ("", "abc", "1.5", "0", "-1", "33", "8.0", "  8"):
        resp = _lineage(client, ids["a"], "ancestors", max_depth=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_validation_error(client):
    ids = _setup_chain(client, ["a", "b"])
    base = f"/v1/contents/{ids['b']}/lineage"
    for url in (
        f"{base}?direction=ancestors&max_depth=1&max_depth=2",
        f"{base}?direction=ancestors&max_depth=1&max_depth=1",
        f"{base}?direction=ancestors&direction=descendants",
        f"{base}?direction=ancestors&direction=ancestors",
    ):
        resp = client.get(url)
        assert resp.status_code == 422, url
        assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_parameters_are_never_silently_defaulted(client):
    ids = _setup_chain(client, ["a", "b"])
    # A bad direction must not fall back to either traversal; the response is
    # an error, not a defaulted result.
    resp = _lineage(client, ids["b"], "sideways")
    assert resp.status_code == 422
    assert "items" not in resp.json()


# --- Read-only guarantee -----------------------------------------------------


def test_lineage_queries_write_no_resources_or_audit_events(client, db_session):
    ids = _setup_chain(client, ["a", "b", "c"])

    def counts():
        return (
            db_session.scalar(select(func.count()).select_from(Content)),
            db_session.scalar(
                select(func.count()).select_from(ContentRelation)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    before = counts()
    _lineage(client, ids["c"], "ancestors")
    _lineage(client, ids["a"], "descendants", max_depth=1)
    _lineage(client, ids["b"], "ancestors")  # empty result
    _lineage(client, "cnt_ghost", "ancestors")  # 404
    _lineage(client, ids["c"], "bad")  # 422
    client.get(
        f"/v1/contents/{ids['c']}/lineage?direction=ancestors&max_depth=1&max_depth=2"
    )  # 422
    db_session.expire_all()
    assert counts() == before
