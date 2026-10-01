"""Tests for the deterministic shortest source-path endpoint.

Covers GET /v1/content-lineage-paths: bidirectional shortest-path search,
same-node and unreachable results, lexicographic relation-id tie-breaking
among equal-length paths (independent of edge creation order), depth
bounding, the two distinct 404s, strict query/body validation, compact wire
format, and the read-only/no-audit guarantee. All fixtures are deterministic
and offline.
"""

from __future__ import annotations

import hashlib
import json

from sqlalchemy import func, select

from provenance.ids import content_relation_id
from provenance.models import AuditEvent, Content, ContentRelation
from tests.helpers import create_actor


def _digest_for(name: str) -> str:
    return hashlib.sha256(f"path-{name}".encode()).hexdigest()


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


def _path(client, start, end, direction, **params):
    return client.get(
        "/v1/content-lineage-paths",
        params={
            "start_content_id": start,
            "end_content_id": end,
            "direction": direction,
            **params,
        },
    )


def _setup_chain(client, names):
    """Create contents and edges each -> the previous; return id-by-name."""
    create_actor(client)
    ids = {name: _create_content(client, name)["id"] for name in names}
    for child, parent in zip(names[1:], names[:-1]):
        _edge(client, ids[child], ids[parent])
    return ids


def _node_titles(body):
    return [node["title"] for node in body["nodes"]]


# --- Basic traversal ---------------------------------------------------------


def test_ancestors_path_walks_content_to_parent(client):
    ids = _setup_chain(client, ["a", "b", "c"])  # c -> b -> a
    resp = _path(client, ids["c"], ids["a"], "ancestors")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["start_content_id"] == ids["c"]
    assert body["end_content_id"] == ids["a"]
    assert body["direction"] == "ancestors"
    assert body["found"] is True
    assert body["depth"] == 2
    assert _node_titles(body) == ["c", "b", "a"]
    assert [rel["content_id"] for rel in body["relations"]] == [ids["c"], ids["b"]]
    assert [rel["parent_content_id"] for rel in body["relations"]] == [
        ids["b"],
        ids["a"],
    ]
    assert len(body["nodes"]) == body["depth"] + 1
    assert len(body["relations"]) == body["depth"]


def test_descendants_path_walks_reversed_edges(client):
    ids = _setup_chain(client, ["a", "b", "c"])  # c -> b -> a
    resp = _path(client, ids["a"], ids["c"], "descendants")
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["depth"] == 2
    assert _node_titles(body) == ["a", "b", "c"]
    assert [rel["content_id"] for rel in body["relations"]] == [ids["b"], ids["c"]]


def test_nodes_and_relations_reuse_public_views(client):
    ids = _setup_chain(client, ["a", "b"])
    edge = _edge(client, ids["b"], ids["a"], "derived_from")
    resp = _path(client, ids["b"], ids["a"], "ancestors")
    body = resp.json()

    assert set(body["nodes"][0]) == {
        "id",
        "digest_algorithm",
        "digest_hex",
        "media_type",
        "title",
        "actor_id",
        "created_at",
    }
    fetched_node = client.get(f"/v1/contents/{ids['b']}").json()
    assert body["nodes"][0] == fetched_node
    assert body["nodes"][1] == client.get(f"/v1/contents/{ids['a']}").json()

    assert set(body["relations"][0]) == {
        "id",
        "content_id",
        "parent_content_id",
        "relation_type",
        "created_at",
    }
    assert body["relations"][0] == edge


def test_direct_relation_is_depth_one(client):
    ids = _setup_chain(client, ["a", "b", "c"])
    resp = _path(client, ids["c"], ids["b"], "ancestors")
    body = resp.json()
    assert body["found"] is True
    assert body["depth"] == 1
    assert _node_titles(body) == ["c", "b"]


# --- Same node ---------------------------------------------------------------


def test_same_start_and_end_is_depth_zero_single_node(client):
    ids = _setup_chain(client, ["a", "b"])
    for direction in ("ancestors", "descendants"):
        resp = _path(client, ids["b"], ids["b"], direction)
        assert resp.status_code == 200
        body = resp.json()
        assert body["start_content_id"] == body["end_content_id"] == ids["b"]
        assert body["direction"] == direction
        assert body["found"] is True
        assert body["depth"] == 0
        assert _node_titles(body) == ["b"]
        assert body["relations"] == []
        assert len(body["nodes"]) == body["depth"] + 1


# --- Unreachable -------------------------------------------------------------


def test_unreachable_within_bound_is_found_false_not_an_error(client):
    ids = _setup_chain(client, ["a", "b"])
    for extra in ("x", "y"):
        ids[extra] = _create_content(client, extra)["id"]
    _edge(client, ids["y"], ids["x"])
    # Disjoint components in both directions.
    for start, end, direction in (
        (ids["a"], ids["x"], "descendants"),
        (ids["x"], ids["a"], "descendants"),
        (ids["b"], ids["y"], "ancestors"),
        (ids["y"], ids["b"], "ancestors"),
    ):
        resp = _path(client, start, end, direction)
        assert resp.status_code == 200, (start, end, direction)
        body = resp.json()
        assert body["found"] is False
        assert body["depth"] is None
        assert body["nodes"] == []
        assert body["relations"] == []


def test_wrong_direction_makes_reachable_pair_unreachable(client):
    ids = _setup_chain(client, ["a", "b", "c"])  # c -> b -> a
    # Ancestors only flow c -> a; asking ancestors a -> c cannot follow.
    resp = _path(client, ids["a"], ids["c"], "ancestors")
    body = resp.json()
    assert resp.status_code == 200
    assert body["found"] is False
    assert body["depth"] is None
    assert body["nodes"] == []
    assert body["relations"] == []


# --- Shortest path and tie-breaking ------------------------------------------


def test_shortest_path_is_preferred_over_a_longer_one(client):
    create_actor(client)
    ids = {
        name: _create_content(client, name)["id"] for name in ("s", "e", "x", "y")
    }
    _edge(client, ids["s"], ids["e"], "derived_from")  # direct, depth 1
    _edge(client, ids["s"], ids["x"])  # detour depth 3
    _edge(client, ids["x"], ids["y"])
    _edge(client, ids["y"], ids["e"])
    resp = _path(client, ids["s"], ids["e"], "ancestors")
    body = resp.json()
    assert body["depth"] == 1
    assert _node_titles(body) == ["s", "e"]
    assert body["relations"][0]["relation_type"] == "derived_from"


def test_tie_break_uses_relation_ids_not_creation_order(client):
    create_actor(client)
    ids = {name: _create_content(client, name)["id"] for name in ("s", "x", "y", "e")}
    # Create the s->x edge BEFORE s->y; the diamond then offers two
    # equal-length paths s-x-e and s-y-e. The chosen path must follow the
    # smaller relation id regardless of which edge was created first.
    edge_sx = _edge(client, ids["s"], ids["x"])
    edge_sy = _edge(client, ids["s"], ids["y"], "derived_from")
    edge_xe = _edge(client, ids["x"], ids["e"])
    edge_ye = _edge(client, ids["y"], ids["e"], "derived_from")

    resp = _path(client, ids["s"], ids["e"], "ancestors")
    body = resp.json()
    assert body["depth"] == 2
    expected_first = min(edge_sx["id"], edge_sy["id"])
    assert body["relations"][0]["id"] == expected_first
    via_x = [edge_sx["id"], edge_xe["id"]]
    via_y = [edge_sy["id"], edge_ye["id"]]
    assert [rel["id"] for rel in body["relations"]] == min(via_x, via_y)
    assert _node_titles(body) == (
        ["s", "x", "e"] if via_x < via_y else ["s", "y", "e"]
    )
    assert edge_sx["id"] != edge_sy["id"]  # tie genuinely possible here


def test_tie_break_falls_through_to_later_edge(client):
    # Two paths share the smallest first edge; the tie must then be resolved
    # on the second edge id.
    create_actor(client)
    ids = {
        name: _create_content(client, name)["id"]
        for name in ("s", "m", "x", "y", "e")
    }
    edge_sm = _edge(client, ids["s"], ids["m"])
    edge_mx = _edge(client, ids["m"], ids["x"])
    edge_my = _edge(client, ids["m"], ids["y"], "derived_from")
    edge_xe = _edge(client, ids["x"], ids["e"])
    edge_ye = _edge(client, ids["y"], ids["e"])

    resp = _path(client, ids["s"], ids["e"], "ancestors")
    body = resp.json()
    assert body["depth"] == 3
    via_x = [edge_sm["id"], edge_mx["id"], edge_xe["id"]]
    via_y = [edge_sm["id"], edge_my["id"], edge_ye["id"]]
    assert [rel["id"] for rel in body["relations"]] == min(via_x, via_y)
    assert _node_titles(body) == (
        ["s", "m", "x", "e"] if via_x < via_y else ["s", "m", "y", "e"]
    )


def test_parallel_edges_of_different_types_compare_by_relation_id(client):
    # The same node pair can carry both edge types; the two parallel
    # relations form distinct depth-1 paths and the smallest id wins.
    create_actor(client)
    ids = {name: _create_content(client, name)["id"] for name in ("s", "e")}
    edge_version = _edge(client, ids["s"], ids["e"], "version_of")
    edge_derived = _edge(client, ids["s"], ids["e"], "derived_from")
    assert edge_version["id"] != edge_derived["id"]

    resp = _path(client, ids["s"], ids["e"], "ancestors")
    body = resp.json()
    assert body["found"] is True
    assert body["depth"] == 1
    assert body["relations"][0]["id"] == min(
        edge_version["id"], edge_derived["id"]
    )
    assert _node_titles(body) == ["s", "e"]


def test_descendants_tie_break_is_equally_deterministic(client):
    create_actor(client)
    ids = {name: _create_content(client, name)["id"] for name in ("e", "x", "y", "s")}
    # Edges point child -> parent: x -> e and y -> e, s -> x, s -> y.
    edge_xe = _edge(client, ids["x"], ids["e"])
    edge_ye = _edge(client, ids["y"], ids["e"], "derived_from")
    edge_sx = _edge(client, ids["s"], ids["x"])
    edge_sy = _edge(client, ids["s"], ids["y"], "derived_from")

    resp = _path(client, ids["e"], ids["s"], "descendants")
    body = resp.json()
    assert body["depth"] == 2
    via_x = [edge_xe["id"], edge_sx["id"]]
    via_y = [edge_ye["id"], edge_sy["id"]]
    assert [rel["id"] for rel in body["relations"]] == min(via_x, via_y)
    assert _node_titles(body) == (
        ["e", "x", "s"] if via_x < via_y else ["e", "y", "s"]
    )


# --- Depth bound -------------------------------------------------------------


def test_path_at_exactly_max_depth_is_found(client):
    ids = _setup_chain(client, ["a", "b", "c", "d"])  # d -> c -> b -> a
    resp = _path(client, ids["d"], ids["a"], "ancestors", max_depth=3)
    body = resp.json()
    assert body["found"] is True
    assert body["depth"] == 3


def test_path_one_edge_beyond_max_depth_is_not_found(client):
    ids = _setup_chain(client, ["a", "b", "c", "d"])
    resp = _path(client, ids["d"], ids["a"], "ancestors", max_depth=2)
    body = resp.json()
    assert body["found"] is False
    assert body["depth"] is None
    assert body["nodes"] == []
    assert body["relations"] == []


def test_max_depth_boundaries_accepted(client):
    ids = _setup_chain(client, ["a", "b"])
    for value in (1, 32):
        resp = _path(client, ids["b"], ids["a"], "ancestors", max_depth=value)
        assert resp.status_code == 200, resp.text


def test_default_max_depth_is_eight(client):
    names = [f"n{i:02d}" for i in range(10)]  # 10 nodes, 9 edges
    ids = _setup_chain(client, names)
    resp = _path(client, ids[names[-1]], ids[names[0]], "ancestors")
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is False
    assert body["depth"] is None


# --- Cycles in anomalous history ---------------------------------------------


def test_cycle_in_anomalous_history_still_resolves_path(client, db_session):
    create_actor(client)
    ids = {name: _create_content(client, name)["id"] for name in ("s", "m", "e")}
    _edge(client, ids["s"], ids["m"])
    _edge(client, ids["m"], ids["e"])
    # Bypass the service cycle guard: e -> s closes s -> m -> e -> s.
    db_session.add(
        ContentRelation(
            id=content_relation_id(ids["e"], ids["s"], "version_of"),
            content_id=ids["e"],
            parent_content_id=ids["s"],
            relation_type="version_of",
        )
    )
    db_session.commit()
    resp = _path(client, ids["s"], ids["e"], "ancestors")
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["depth"] == 2
    assert _node_titles(body) == ["s", "m", "e"]


# --- Missing endpoints --------------------------------------------------------


def test_missing_start_is_content_not_found(client):
    ids = _setup_chain(client, ["a"])
    resp = _path(client, "cnt_ghost", ids["a"], "ancestors")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "content_not_found"
    assert error["details"]["content_id"] == "cnt_ghost"


def test_missing_end_is_content_not_found(client):
    ids = _setup_chain(client, ["a"])
    resp = _path(client, ids["a"], "cnt_ghost", "ancestors")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "content_not_found"
    assert error["details"]["content_id"] == "cnt_ghost"


def test_missing_start_is_reported_before_missing_end(client):
    _setup_chain(client, ["a"])
    resp = _path(client, "cnt_start", "cnt_end", "ancestors")
    assert resp.status_code == 404
    assert resp.json()["error"]["details"]["content_id"] == "cnt_start"


def test_same_unknown_id_is_one_not_found(client):
    _setup_chain(client, ["a"])
    resp = _path(client, "cnt_ghost", "cnt_ghost", "ancestors")
    assert resp.status_code == 404
    assert resp.json()["error"]["details"]["content_id"] == "cnt_ghost"


# --- Validation ---------------------------------------------------------------


def test_direction_is_required(client):
    ids = _setup_chain(client, ["a", "b"])
    resp = client.get(
        "/v1/content-lineage-paths",
        params={
            "start_content_id": ids["b"],
            "end_content_id": ids["a"],
        },
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_start_and_end_are_required(client):
    ids = _setup_chain(client, ["a"])
    for missing, params in (
        ("start_content_id", {"end_content_id": ids["a"], "direction": "ancestors"}),
        ("end_content_id", {"start_content_id": ids["a"], "direction": "ancestors"}),
    ):
        resp = client.get("/v1/content-lineage-paths", params=params)
        assert resp.status_code == 422, missing
        error = resp.json()["error"]
        assert error["code"] == "validation_error"
        assert error["details"]["issues"][0]["loc"] == ["query", missing]


def test_blank_or_illegal_direction_is_validation_error(client):
    ids = _setup_chain(client, ["a", "b"])
    for value in ("", "   ", "up", "ANCESTORS", "ancestor", "descendant"):
        resp = _path(client, ids["b"], ids["a"], value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_empty_content_ids_are_validation_error(client):
    ids = _setup_chain(client, ["a"])
    for params in (
        {"start_content_id": "", "end_content_id": ids["a"], "direction": "ancestors"},
        {"start_content_id": "   ", "end_content_id": ids["a"], "direction": "ancestors"},
        {"start_content_id": ids["a"], "end_content_id": "", "direction": "ancestors"},
        {"start_content_id": ids["a"], "end_content_id": "  ", "direction": "ancestors"},
    ):
        resp = client.get("/v1/content-lineage-paths", params=params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_max_depth_non_decimal_or_out_of_range_is_validation_error(client):
    ids = _setup_chain(client, ["a", "b"])
    for value in ("", "abc", "1.5", "0", "-1", "33", "8.0", "  8", "8 "):
        resp = _path(client, ids["b"], ids["a"], "ancestors", max_depth=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_unknown_parameter_is_validation_error(client):
    ids = _setup_chain(client, ["a", "b"])
    resp = _path(client, ids["b"], ids["a"], "ancestors", relation_type="version_of")
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"]["issues"][0]["loc"][-1] == "relation_type"


def test_repeated_parameters_are_validation_error(client):
    ids = _setup_chain(client, ["a", "b"])
    base = "/v1/content-lineage-paths"
    urls = (
        f"{base}?start_content_id={ids['b']}&end_content_id={ids['a']}"
        f"&direction=ancestors&direction=descendants",
        f"{base}?start_content_id={ids['b']}&start_content_id={ids['b']}"
        f"&end_content_id={ids['a']}&direction=ancestors",
        f"{base}?start_content_id={ids['b']}&end_content_id={ids['a']}"
        f"&end_content_id={ids['a']}&direction=ancestors",
        f"{base}?start_content_id={ids['b']}&end_content_id={ids['a']}"
        f"&direction=ancestors&max_depth=1&max_depth=2",
    )
    for url in urls:
        resp = client.get(url)
        assert resp.status_code == 422, url
        assert resp.json()["error"]["code"] == "validation_error"


def test_nonempty_body_bytes_are_validation_error(client):
    ids = _setup_chain(client, ["a", "b"])
    url = (
        "/v1/content-lineage-paths?start_content_id="
        f"{ids['b']}&end_content_id={ids['a']}&direction=ancestors"
    )
    for body in (b"{}", b" ", b"\n", b"null"):
        resp = client.request("GET", url, content=body)
        assert resp.status_code == 422, body
        error = resp.json()["error"]
        assert error["code"] == "validation_error"
        assert error["details"]["issues"][0]["loc"] == ["query", "body"]


def test_validation_failure_never_reaches_the_404(client):
    _setup_chain(client, ["a"])
    # Nonexistent endpoints, but the malformed parameter must win: 422
    # before any existence check.
    resp = client.get(
        "/v1/content-lineage-paths",
        params={
            "start_content_id": "cnt_ghost",
            "end_content_id": "cnt_ghost",
            "direction": "sideways",
        },
    )
    assert resp.status_code == 422
    resp = client.request(
        "GET",
        "/v1/content-lineage-paths?start_content_id=cnt_ghost"
        "&end_content_id=cnt_ghost&direction=ancestors",
        content=b"x",
    )
    assert resp.status_code == 422


# --- Wire format --------------------------------------------------------------


def test_response_is_compact_utf8_json_with_single_trailing_newline(client):
    ids = _setup_chain(client, ["a", "b"])
    resp = _path(client, ids["b"], ids["a"], "ancestors")
    raw = resp.content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    # Compact separators: no whitespace after commas or colons.
    assert b": " not in raw and b", " not in raw
    # The bytes parse back to the same document.
    assert json.loads(raw.decode("utf-8")) == resp.json()
    assert resp.headers["content-type"].startswith("application/json")


def test_not_found_response_shape_is_stable(client):
    ids = _setup_chain(client, ["zz"])
    resp = _path(client, "cnt_x", ids["zz"], "ancestors")
    assert resp.status_code == 404
    assert set(resp.json()["error"]) == {"code", "message", "details"}
    assert resp.json()["error"]["details"] == {"content_id": "cnt_x"}


# --- Read-only guarantee ------------------------------------------------------


def test_path_queries_write_no_resources_or_audit_events(client, db_session):
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
    _path(client, ids["c"], ids["a"], "ancestors")  # found
    _path(client, ids["a"], ids["c"], "descendants")  # found
    _path(client, ids["a"], ids["c"], "ancestors")  # not found
    _path(client, ids["b"], ids["b"], "ancestors")  # depth 0
    _path(client, "cnt_ghost", ids["a"], "ancestors")  # 404
    _path(client, ids["a"], "cnt_ghost", "ancestors")  # 404
    _path(client, ids["c"], ids["a"], "sideways")  # 422
    client.get(
        "/v1/content-lineage-paths?start_content_id="
        f"{ids['c']}&end_content_id={ids['a']}"
        "&direction=ancestors&max_depth=1&max_depth=2"
    )  # 422
    db_session.expire_all()
    assert counts() == before
