"""Tests for the deterministic shortest lineage path endpoint.

Covers GET /v1/content-lineage-paths: bidirectional shortest-path search,
lexicographic relation-id tie-breaking, the trivial same-content path,
depth truncation and the not-found result, parameter validation, the
missing-endpoint 404s, the exact success-body shape and encoding, and the
read-only guarantee. All fixtures are deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json

from sqlalchemy import func, select

from provenance.models import AuditEvent
from tests.helpers import create_actor

# Fixed, distinct, offline digests derived deterministically from node names.
def _digest_for(name: str) -> str:
    return hashlib.sha256(f"lineage-path-{name}".encode()).hexdigest()


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


# --- Shortest-path traversal ---------------------------------------------------


def test_ancestors_path_follows_content_to_parent(client):
    ids = _setup_chain(client, ["a", "b", "c"])  # c -> b -> a
    resp = _path(client, ids["c"], ids["a"], "ancestors")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["found"] is True
    assert body["depth"] == 2
    assert [node["id"] for node in body["nodes"]] == [
        ids["c"],
        ids["b"],
        ids["a"],
    ]
    assert len(body["relations"]) == 2
    assert body["relations"][0]["content_id"] == ids["c"]
    assert body["relations"][0]["parent_content_id"] == ids["b"]
    assert body["relations"][1]["content_id"] == ids["b"]
    assert body["relations"][1]["parent_content_id"] == ids["a"]


def test_descendants_path_follows_edges_in_reverse(client):
    ids = _setup_chain(client, ["a", "b", "c"])  # c -> b -> a
    resp = _path(client, ids["a"], ids["c"], "descendants")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["found"] is True
    assert body["depth"] == 2
    assert [node["id"] for node in body["nodes"]] == [
        ids["a"],
        ids["b"],
        ids["c"],
    ]
    # The same relations, reported in traversal order.
    assert [rel["content_id"] for rel in body["relations"]] == [
        ids["b"],
        ids["c"],
    ]


def test_direction_requiring_against_the_grain_is_not_found(client):
    ids = _setup_chain(client, ["a", "b", "c"])  # c -> b -> a
    resp = _path(client, ids["a"], ids["c"], "ancestors")
    assert resp.status_code == 200
    assert resp.json() == {
        "start_content_id": ids["a"],
        "end_content_id": ids["c"],
        "direction": "ancestors",
        "found": False,
        "depth": None,
        "nodes": [],
        "relations": [],
    }


def test_same_start_and_end_is_trivial_path(client):
    ids = _setup_chain(client, ["a", "b"])
    for direction in ("ancestors", "descendants"):
        resp = _path(client, ids["a"], ids["a"], direction)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["found"] is True
        assert body["depth"] == 0
        assert [node["id"] for node in body["nodes"]] == [ids["a"]]
        assert body["relations"] == []


def test_shortest_path_wins_over_a_longer_route(client):
    # o derives directly from a (depth 1) and indirectly o -> x -> a.
    create_actor(client)
    ids = {name: _create_content(client, name)["id"] for name in ("o", "a", "x")}
    _edge(client, ids["o"], ids["x"])
    direct = _edge(client, ids["o"], ids["a"], "derived_from")
    _edge(client, ids["x"], ids["a"])

    resp = _path(client, ids["o"], ids["a"], "ancestors")
    body = resp.json()
    assert body["found"] is True
    assert body["depth"] == 1
    assert [node["id"] for node in body["nodes"]] == [ids["o"], ids["a"]]
    assert [rel["id"] for rel in body["relations"]] == [direct["id"]]


def test_equal_length_paths_pick_smallest_relation_id_sequence(client):
    # Diamond: d -> x -> c and d -> y -> c, two ancestor paths of depth 2.
    create_actor(client)
    ids = {
        name: _create_content(client, name)["id"] for name in ("d", "x", "y", "c")
    }
    rel_dx = _edge(client, ids["d"], ids["x"])
    rel_xc = _edge(client, ids["x"], ids["c"])
    rel_dy = _edge(client, ids["d"], ids["y"], "derived_from")
    rel_yc = _edge(client, ids["y"], ids["c"], "derived_from")

    expected = min(
        [rel_dx["id"], rel_xc["id"]],
        [rel_dy["id"], rel_yc["id"]],
    )
    resp = _path(client, ids["d"], ids["c"], "ancestors")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["depth"] == 2
    assert [rel["id"] for rel in body["relations"]] == expected
    # The node sequence is the one belonging to the chosen relation chain.
    via = expected[0]
    middle = {rel_dx["id"]: ids["x"], rel_dy["id"]: ids["y"]}[via]
    assert [node["id"] for node in body["nodes"]] == [ids["d"], middle, ids["c"]]
    # Deterministic: a repeat query returns the identical path.
    assert _path(client, ids["d"], ids["c"], "ancestors").json() == body


def test_parallel_edges_between_same_pair_pick_smaller_relation_id(client):
    create_actor(client)
    ids = {name: _create_content(client, name)["id"] for name in ("b", "a")}
    rel1 = _edge(client, ids["b"], ids["a"])
    rel2 = _edge(client, ids["b"], ids["a"], "derived_from")

    resp = _path(client, ids["b"], ids["a"], "ancestors")
    body = resp.json()
    assert body["depth"] == 1
    assert [rel["id"] for rel in body["relations"]] == [
        min(rel1["id"], rel2["id"])
    ]


# --- Depth limit -----------------------------------------------------------------


def test_max_depth_truncates_longer_paths(client):
    ids = _setup_chain(client, ["a", "b", "c", "d"])  # d -> c -> b -> a
    resp = _path(client, ids["d"], ids["a"], "ancestors", max_depth=2)
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is False
    assert body["depth"] is None
    assert body["nodes"] == []
    assert body["relations"] == []

    resp = _path(client, ids["d"], ids["a"], "ancestors", max_depth=3)
    assert resp.json()["found"] is True
    assert resp.json()["depth"] == 3


def test_default_max_depth_is_eight(client):
    names = [f"n{i:02d}" for i in range(11)]  # 11 nodes, 10 edges
    ids = _setup_chain(client, names)
    resp = _path(client, ids[names[-1]], ids[names[0]], "ancestors")
    assert resp.json()["found"] is False
    resp = _path(
        client, ids[names[-1]], ids[names[0]], "ancestors", max_depth=10
    )
    assert resp.json()["found"] is True
    assert resp.json()["depth"] == 10


def test_max_depth_boundaries_are_accepted(client):
    ids = _setup_chain(client, ["a", "b"])
    for depth in (1, 32):
        resp = _path(
            client, ids["b"], ids["a"], "ancestors", max_depth=depth
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["found"] is True


# --- Response shape and encoding ---------------------------------------------------


def test_success_body_shape_reuses_public_views(client):
    ids = _setup_chain(client, ["a", "b"])  # b -> a
    resp = _path(client, ids["b"], ids["a"], "ancestors")
    assert resp.status_code == 200
    body = resp.json()
    assert list(body) == [
        "start_content_id",
        "end_content_id",
        "direction",
        "found",
        "depth",
        "nodes",
        "relations",
    ]
    assert body["start_content_id"] == ids["b"]
    assert body["end_content_id"] == ids["a"]
    assert body["direction"] == "ancestors"
    assert body["depth"] == len(body["relations"]) == len(body["nodes"]) - 1

    content_view = client.get(f"/v1/contents/{ids['a']}").json()
    assert body["nodes"][1] == content_view
    relation_view = client.get(
        f"/v1/content-relations/{body['relations'][0]['id']}"
    ).json()
    assert body["relations"][0] == relation_view


def test_response_is_compact_json_with_single_trailing_newline(client):
    ids = _setup_chain(client, ["a", "b"])
    resp = _path(client, ids["b"], ids["a"], "ancestors")
    raw = resp.content
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert raw.decode("utf-8")[:-1] == json.dumps(
        resp.json(), separators=(",", ":"), ensure_ascii=False
    )


# --- Missing endpoints ----------------------------------------------------------


def test_unknown_start_content_is_404(client):
    ids = _setup_chain(client, ["a"])
    resp = _path(client, "cnt_ghost", ids["a"], "ancestors")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "content_not_found"
    assert error["details"]["content_id"] == "cnt_ghost"


def test_unknown_end_content_is_404(client):
    ids = _setup_chain(client, ["a"])
    resp = _path(client, ids["a"], "cnt_ghost", "descendants")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "content_not_found"
    assert error["details"]["content_id"] == "cnt_ghost"


def test_both_unknown_reports_the_start_content(client):
    create_actor(client)
    resp = _path(client, "cnt_ghost_start", "cnt_ghost_end", "ancestors")
    assert resp.status_code == 404
    assert resp.json()["error"]["details"]["content_id"] == "cnt_ghost_start"


# --- Parameter validation ----------------------------------------------------------


def test_non_empty_body_is_422(client):
    ids = _setup_chain(client, ["a"])
    for body in (b"{}", b" ", b"not-json"):
        resp = client.request(
            "GET",
            "/v1/content-lineage-paths"
            f"?start_content_id={ids['a']}&end_content_id={ids['a']}"
            "&direction=ancestors",
            content=body,
        )
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"


def test_missing_required_parameters_are_422(client):
    ids = _setup_chain(client, ["a"])
    base = {
        "start_content_id": ids["a"],
        "end_content_id": ids["a"],
        "direction": "ancestors",
    }
    for omitted in base:
        resp = client.get(
            "/v1/content-lineage-paths",
            params={k: v for k, v in base.items() if k != omitted},
        )
        assert resp.status_code == 422, omitted
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_422(client):
    ids = _setup_chain(client, ["a"])
    url = (
        "/v1/content-lineage-paths"
        f"?start_content_id={ids['a']}&start_content_id={ids['a']}"
        f"&end_content_id={ids['a']}&direction=ancestors"
    )
    resp = client.get(url)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_unknown_parameter_is_422(client):
    ids = _setup_chain(client, ["a"])
    resp = _path(client, ids["a"], ids["a"], "ancestors", limit=5)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_blank_required_values_are_422(client):
    ids = _setup_chain(client, ["a"])
    for params in (
        {"start_content_id": ""},
        {"end_content_id": "   "},
        {"direction": ""},
    ):
        resp = client.get(
            "/v1/content-lineage-paths",
            params={
                "start_content_id": ids["a"],
                "end_content_id": ids["a"],
                "direction": "ancestors",
                **params,
            },
        )
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_illegal_direction_is_422(client):
    ids = _setup_chain(client, ["a"])
    for direction in ("up", "ANCESTORS", "ancestor", "descendant"):
        resp = _path(client, ids["a"], ids["a"], direction)
        assert resp.status_code == 422, direction
        assert resp.json()["error"]["code"] == "validation_error"


def test_illegal_max_depth_is_422(client):
    ids = _setup_chain(client, ["a", "b"])
    for value in ("", "abc", "8.0", "1.5", "0", "-1", "33", "  8", "1e2"):
        resp = _path(
            client, ids["b"], ids["a"], "ancestors", max_depth=value
        )
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only guarantee ----------------------------------------------------------


def test_queries_write_no_resources_or_audit_events(client, db_session):
    ids = _setup_chain(client, ["a", "b", "c"])

    def audit_count():
        return db_session.scalar(select(func.count()).select_from(AuditEvent))

    before = audit_count()
    _path(client, ids["c"], ids["a"], "ancestors")  # found
    _path(client, ids["a"], ids["c"], "ancestors")  # not found
    _path(client, ids["a"], ids["a"], "descendants")  # trivial path
    _path(client, "cnt_ghost", ids["a"], "ancestors")  # 404
    _path(client, ids["c"], ids["a"], "sideways")  # 422
    client.get(
        "/v1/content-lineage-paths"
        f"?start_content_id={ids['c']}&end_content_id={ids['a']}"
        "&direction=ancestors&max_depth=0"  # 422
    )
    db_session.expire_all()
    assert audit_count() == before
