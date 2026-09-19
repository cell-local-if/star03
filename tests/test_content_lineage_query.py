"""Tests for filtered, cursor-paginated read-only lineage queries.

Covers the ``relation_type`` / ``min_depth`` / ``limit`` / ``cursor``
extensions of GET /v1/contents/{content_id}/lineage: filtering that never
changes traversal reachability or shortest-depth computation, convergence
under mixed edge types, depth/edge ordering preserved across pages,
page-to-page continuity with no duplicates or gaps, empty filtered results,
parameter boundaries and incompatibilities, malformed/tampered/expired/
mismatched cursors, cyclic history, the missing-origin 404, and the
read-only/no-audit guarantee. Every fixture is deterministic and offline.
"""

from __future__ import annotations

import hashlib

from sqlalchemy import func, select

from provenance import pagination
from provenance.app import create_app
from provenance.config import DEFAULT_LINEAGE_CURSOR_SECRET, Settings
from provenance.ids import content_relation_id
from provenance.models import AuditEvent, Content, ContentRelation
from tests.helpers import create_actor


def _digest_for(name: str) -> str:
    return hashlib.sha256(f"lineage-query-{name}".encode()).hexdigest()


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


def _ordered(items):
    return [(item["title"], item["depth"]) for item in items]


def _walk_all(client, content_id, direction, **params):
    """Follow every cursor page; return (all items in order, per-page count)."""
    pages, cursor = [], None
    while True:
        query = {"direction": direction, **params}
        if cursor is not None:
            query["cursor"] = cursor
        resp = _lineage(client, content_id, **query)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        pages.append(body)
        cursor = body["next_cursor"]
        if cursor is None:
            break
    items = [item for page in pages for item in page["items"]]
    return items, pages


# --- relation_type filtering ------------------------------------------------


def test_relation_type_filter_returns_only_first_discovery_edge_type(client):
    # o -> a (version_of), o -> b (derived_from), a -> c (derived_from),
    # b -> c (version_of, created second): c's first-discovery edge is
    # a -> c and therefore derived_from.
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("o", "a", "b", "c")}
    _edge(client, ids["o"], ids["a"], "version_of")
    _edge(client, ids["o"], ids["b"], "derived_from")
    _edge(client, ids["a"], ids["c"], "derived_from")
    _edge(client, ids["b"], ids["c"], "version_of")

    version = _lineage(client, ids["o"], relation_type="version_of").json()
    assert _ordered(version["items"]) == [("a", 1)]
    assert version["count"] == 1
    assert version["next_cursor"] is None

    derived = _lineage(client, ids["o"], relation_type="derived_from").json()
    assert _ordered(derived["items"]) == [("b", 1), ("c", 2)]
    assert derived["count"] == 2


def test_filtering_does_not_restrict_traversal_reachability(client):
    # o -derived-> p -version-> q: q is reachable only THROUGH a derived edge.
    # With a version_of filter the walk still traverses the derived edge and
    # returns q (whose first-discovery edge is version_of).
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("o", "p", "q")}
    _edge(client, ids["o"], ids["p"], "derived_from")
    _edge(client, ids["p"], ids["q"], "version_of")

    resp = _lineage(client, ids["o"], relation_type="version_of")
    assert _ordered(resp.json()["items"]) == [("q", 2)]
    # The derived-only filter keeps the intermediary but not the version edge
    # target; nothing about the walk was pruned to compute this.
    resp = _lineage(client, ids["o"], relation_type="derived_from")
    assert _ordered(resp.json()["items"]) == [("p", 1)]


def test_filtering_does_not_restrict_traversal_in_descendant_direction(client):
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("o", "p", "q")}
    _edge(client, ids["q"], ids["p"], "version_of")
    _edge(client, ids["p"], ids["o"], "derived_from")  # p -derived-> o

    resp = _lineage(client, ids["o"], "descendants", relation_type="version_of")
    assert _ordered(resp.json()["items"]) == [("q", 2)]


def test_filtered_convergence_uses_shortest_depth_of_full_graph(client):
    # x reaches a directly (derived, depth 1) and via x -> v -> a (version
    # edges, depth 2). The version_of filter must not surface a at depth 2:
    # shortest depth is computed over the unfiltered graph, and a is simply
    # excluded because its first-discovery edge is derived.
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("x", "v", "a")}
    _edge(client, ids["x"], ids["a"], "derived_from")
    _edge(client, ids["x"], ids["v"], "version_of")
    _edge(client, ids["v"], ids["a"], "version_of")

    resp = _lineage(client, ids["x"], relation_type="version_of")
    assert _ordered(resp.json()["items"]) == [("v", 1)]


def test_omitted_relation_type_disables_filtering(client):
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("o", "a", "b")}
    _edge(client, ids["o"], ids["a"], "version_of")
    _edge(client, ids["o"], ids["b"], "derived_from")
    resp = _lineage(client, ids["o"])
    assert _ordered(resp.json()["items"]) == [("a", 1), ("b", 1)]


# --- min_depth filtering -----------------------------------------------------


def test_min_depth_floors_the_returned_items_only(client):
    # Chain e -> d -> c -> b -> a; the shallower edges are still walked, so
    # deeper contents remain reachable despite the depth-1/2 items being
    # filtered out of the response.
    names = ["a", "b", "c", "d", "e"]
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in names}
    for child, parent in zip(names[1:], names[:-1]):
        _edge(client, ids[child], ids[parent])

    resp = _lineage(client, ids["e"], min_depth=3)
    assert _ordered(resp.json()["items"]) == [("b", 3), ("a", 4)]
    assert resp.json()["count"] == 2


def test_min_depth_equal_to_max_depth_returns_one_level(client):
    names = ["a", "b", "c"]
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in names}
    for child, parent in zip(names[1:], names[:-1]):
        _edge(client, ids[child], ids[parent])

    resp = _lineage(client, ids["c"], min_depth=2, max_depth=2)
    assert _ordered(resp.json()["items"]) == [("a", 2)]


def test_min_depth_beyond_reachable_depth_is_empty(client):
    names = ["a", "b", "c"]
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in names}
    for child, parent in zip(names[1:], names[:-1]):
        _edge(client, ids[child], ids[parent])

    resp = _lineage(client, ids["c"], min_depth=3, max_depth=8)
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_min_depth_and_relation_type_compose(client):
    # o -d-> p -v-> q: depth 1's p is derived; depth 2's q is version.
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("o", "p", "q")}
    _edge(client, ids["o"], ids["p"], "derived_from")
    _edge(client, ids["p"], ids["q"], "version_of")

    resp = _lineage(client, ids["o"], min_depth=2, relation_type="version_of")
    assert _ordered(resp.json()["items"]) == [("q", 2)]
    # Floor removes q even though its edge type matches.
    resp = _lineage(client, ids["o"], min_depth=3, relation_type="version_of")
    assert resp.json()["items"] == []


def test_default_min_depth_is_one(client):
    names = ["a", "b"]
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in names}
    _edge(client, ids["b"], ids["a"])
    explicit = _lineage(client, ids["b"], min_depth=1).json()
    defaulted = _lineage(client, ids["b"]).json()
    assert explicit == defaulted
    assert _ordered(defaulted["items"]) == [("a", 1)]


# --- limit / pagination shape ------------------------------------------------


def test_response_carries_null_cursor_when_everything_fits(client):
    names = ["a", "b", "c"]
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in names}
    for child, parent in zip(names[1:], names[:-1]):
        _edge(client, ids[child], ids[parent])
    body = _lineage(client, ids["c"], limit=100).json()
    assert body["count"] == 2
    assert len(body["items"]) == 2
    assert body["next_cursor"] is None


def test_limit_one_pages_one_item_at_a_time_in_order(client):
    names = ["a", "b", "c", "d"]  # d -> c -> b -> a
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in names}
    for child, parent in zip(names[1:], names[:-1]):
        _edge(client, ids[child], ids[parent])

    items, pages = _walk_all(client, ids["d"], "ancestors", limit=1)
    assert _ordered(items) == [("c", 1), ("b", 2), ("a", 3)]
    assert [len(page["items"]) for page in pages] == [1, 1, 1]
    # count is the filtered total and is identical on every page.
    assert {page["count"] for page in pages} == {3}
    assert pages[-1]["next_cursor"] is None


def test_pagination_is_continuous_without_duplicates_or_gaps(client):
    # Mixed, converging graph; pages must concatenate back to the exact
    # unfiltered depth/edge ordering.
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("r", "a", "b", "c", "x", "y", "d")}
    _edge(client, ids["r"], ids["a"], "version_of")
    _edge(client, ids["r"], ids["b"], "derived_from")
    _edge(client, ids["r"], ids["c"], "version_of")
    _edge(client, ids["a"], ids["x"], "derived_from")
    _edge(client, ids["b"], ids["x"], "version_of")  # converges: x already depth 2
    _edge(client, ids["b"], ids["y"], "version_of")
    _edge(client, ids["x"], ids["d"], "version_of")
    _edge(client, ids["y"], ids["d"], "derived_from")  # converges: d already depth 3

    full = _lineage(client, ids["r"]).json()
    expected = [("a", 1), ("b", 1), ("c", 1), ("x", 2), ("y", 2), ("d", 3)]
    assert _ordered(full["items"]) == expected

    items, pages = _walk_all(client, ids["r"], "ancestors", limit=2)
    assert _ordered(items) == expected
    assert len(items) == len({item["id"] for item in items})
    assert [len(page["items"]) for page in pages] == [2, 2, 2]
    assert {page["count"] for page in pages} == {6}


def test_pagination_over_filtered_results_preserves_order_and_count(client):
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("r", "a", "b", "c", "x", "y", "d")}
    _edge(client, ids["r"], ids["a"], "version_of")
    _edge(client, ids["r"], ids["b"], "derived_from")
    _edge(client, ids["r"], ids["c"], "version_of")
    _edge(client, ids["a"], ids["x"], "derived_from")
    _edge(client, ids["b"], ids["x"], "version_of")
    _edge(client, ids["b"], ids["y"], "version_of")
    _edge(client, ids["x"], ids["d"], "version_of")
    _edge(client, ids["y"], ids["d"], "derived_from")

    items, pages = _walk_all(
        client, ids["r"], "ancestors", relation_type="version_of", limit=2
    )
    # a,c (depth1), y (depth2 via b->y), d (depth3 via x->d); x is excluded
    # because its first-discovery edge is derived.
    assert _ordered(items) == [("a", 1), ("c", 1), ("y", 2), ("d", 3)]
    assert {page["count"] for page in pages} == {4}

    items, pages = _walk_all(
        client, ids["r"], "ancestors", min_depth=2, limit=2
    )
    assert _ordered(items) == [("x", 2), ("y", 2), ("d", 3)]
    assert {page["count"] for page in pages} == {3}


def test_default_limit_is_fifty(client):
    # 52 direct parents at depth 1 (reachable well within the default
    # max_depth of 8): the default page holds 50 items plus a cursor, and the
    # cursor resumes with the remaining 2; count is always 52.
    create_actor(client)
    origin = _create_content(client, "origin")["id"]
    names = [f"p{i:02d}" for i in range(52)]
    ids = {n: _create_content(client, n)["id"] for n in names}
    # Edges in stable creation order; within depth 1 that order is the
    # lineage ordering, so page concatenation must follow it exactly.
    for name in names:
        _edge(client, origin, ids[name])

    items, pages = _walk_all(client, origin, "ancestors")
    assert [len(page["items"]) for page in pages] == [50, 2]
    assert {page["count"] for page in pages} == {52}
    assert _titles(items) == names
    assert {item["depth"] for item in items} == {1}


def test_cursor_is_replayable_and_opaque(client):
    names = ["a", "b", "c", "d"]
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in names}
    for child, parent in zip(names[1:], names[:-1]):
        _edge(client, ids[child], ids[parent])

    first = _lineage(client, ids["d"], limit=2).json()
    cursor = first["next_cursor"]
    assert isinstance(cursor, str) and cursor.count(".") == 1
    # Replaying the same cursor returns the same page again (no skipping).
    replay_a = _lineage(client, ids["d"], limit=2, cursor=cursor).json()
    replay_b = _lineage(client, ids["d"], limit=2, cursor=cursor).json()
    assert _titles(replay_a["items"]) == ["a"]
    assert replay_a == replay_b


# --- cursor validation -------------------------------------------------------


def test_empty_or_blank_cursor_is_validation_error(client):
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("a", "b")}
    _edge(client, ids["b"], ids["a"])
    for value in ("", "   ", "\t"):
        resp = _lineage(client, ids["b"], cursor=value)
        assert resp.status_code == 422, repr(value)
        assert resp.json()["error"]["code"] == "validation_error"
        assert resp.json()["error"]["details"]["issues"][0]["loc"] == [
            "query",
            "cursor",
        ]


def test_malformed_or_tampered_cursor_is_validation_error(client):
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("a", "b", "c")}
    _edge(client, ids["b"], ids["a"])
    _edge(client, ids["c"], ids["b"])
    cursor = _lineage(client, ids["c"], limit=1).json()["next_cursor"]
    body, _, signature = cursor.partition(".")
    for bad in (
        "garbage",
        "garbage.garbage",
        cursor[:-1],
        cursor[:-2] + "aa",
        body + "." + signature[:-2] + "aa",
        body.upper() + "." + signature,
        "." + signature,
        body + ".",
        body + "." + signature + ".extra",
    ):
        resp = _lineage(client, ids["c"], limit=1, cursor=bad)
        assert resp.status_code == 422, bad
        assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_with_unknown_version_is_validation_error(client, monkeypatch):
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("a", "b", "c")}
    _edge(client, ids["b"], ids["a"])
    _edge(client, ids["c"], ids["b"])
    # Mint a structurally valid, correctly signed cursor from a future
    # payload version; it must be rejected rather than interpreted.
    monkeypatch.setattr(pagination, "CURSOR_VERSION", 2)
    future = pagination.encode_lineage_cursor(
        DEFAULT_LINEAGE_CURSOR_SECRET,
        origin_id=ids["c"],
        direction="ancestors",
        max_depth=8,
        min_depth=1,
        relation_type=None,
        limit=1,
        offset=1,
    )
    monkeypatch.undo()
    resp = _lineage(client, ids["c"], limit=1, cursor=future)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_must_match_every_query_parameter(client):
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("o", "a", "b", "c", "z")}
    _edge(client, ids["o"], ids["a"], "version_of")
    _edge(client, ids["o"], ids["b"], "derived_from")
    _edge(client, ids["b"], ids["c"], "version_of")
    _edge(client, ids["z"], ids["a"], "version_of")  # a different origin

    cursor = _lineage(
        client,
        ids["o"],
        max_depth=4,
        min_depth=1,
        relation_type="version_of",
        limit=1,
    ).json()["next_cursor"]
    base = dict(max_depth=4, min_depth=1, relation_type="version_of", limit=1)

    # Change one bound parameter per request; every mismatch is a 422 rather
    # than a silently re-anchored traversal.
    mismatches = [
        {"direction": "descendants", **base},
        {"max_depth": 3, **{k: v for k, v in base.items() if k != "max_depth"}},
        {"min_depth": 2, **{k: v for k, v in base.items() if k != "min_depth"}},
        {"relation_type": "derived_from",
         **{k: v for k, v in base.items() if k != "relation_type"}},
        {"limit": 2, **{k: v for k, v in base.items() if k != "limit"}},
    ]
    for params in mismatches:
        resp = _lineage(client, ids["o"], cursor=cursor, **params)
        assert resp.status_code == 422, params
        issue = resp.json()["error"]["details"]["issues"][0]
        assert issue["loc"] == ["query", "cursor"]

    # Same parameters, different origin path id: also a mismatch.
    resp = _lineage(client, ids["z"], cursor=cursor, **base)
    assert resp.status_code == 422


def test_cursor_signed_under_another_secret_is_rejected(tmp_db_url):
    # A cursor is invalid once its signing secret is gone (expired/rotated);
    # the same cursor works again under an instance with the same secret,
    # proving cursors are stateless and deterministic.
    from fastapi.testclient import TestClient

    def client_for(secret):
        application = create_app(
            Settings(database_url=tmp_db_url, lineage_cursor_secret=secret)
        )
        return TestClient(application), application

    with TestClient(create_app(Settings(database_url=tmp_db_url, lineage_cursor_secret="s1"))) as c1:
        create_actor(c1)
        ids = {n: _create_content(c1, n)["id"] for n in ("a", "b", "c")}
        _edge(c1, ids["b"], ids["a"])
        _edge(c1, ids["c"], ids["b"])
        cursor = c1.get(
            f"/v1/contents/{ids['c']}/lineage",
            params={"direction": "ancestors", "limit": 1},
        ).json()["next_cursor"]

    c2, app2 = client_for("s2-rotated")
    with c2:
        resp = _lineage(c2, ids["c"], limit=1, cursor=cursor)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"
    app2.state.engine.dispose()

    c3, app3 = client_for("s1")
    with c3:
        resp = _lineage(c3, ids["c"], limit=1, cursor=cursor)
        assert resp.status_code == 200
        assert _titles(resp.json()["items"]) == ["a"]
    app3.state.engine.dispose()


# --- cycles under filtering and pagination -----------------------------------


def test_cyclic_history_paginates_and_filters_deterministically(client, db_session):
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("a", "b", "c")}
    _edge(client, ids["b"], ids["a"], "version_of")  # b -> a
    _edge(client, ids["c"], ids["b"], "version_of")  # c -> b
    # Anomalous history closes c -> b -> a -> c with a derived edge.
    db_session.add(
        ContentRelation(
            id=content_relation_id(ids["a"], ids["c"], "derived_from"),
            content_id=ids["a"],
            parent_content_id=ids["c"],
            relation_type="derived_from",
        )
    )
    db_session.commit()

    full, pages = _walk_all(client, ids["c"], "ancestors", limit=1)
    assert _ordered(full) == [("b", 1), ("a", 2)]
    assert [len(page["items"]) for page in pages] == [1, 1]
    assert all(page["count"] == 2 for page in pages)

    # The closing edge's type never becomes a first-discovery edge from c's
    # walk (b and a are already reached through version edges), so the
    # derived_from filter yields an empty, cursorless page -- the cycle still
    # terminates.
    resp = _lineage(client, ids["c"], relation_type="derived_from")
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}

    versioned, _ = _walk_all(
        client, ids["c"], "ancestors", relation_type="version_of", limit=1
    )
    assert _ordered(versioned) == [("b", 1), ("a", 2)]


# --- empty results ------------------------------------------------------------


def test_filter_matching_nothing_is_empty_without_cursor(client):
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("o", "p")}
    _edge(client, ids["o"], ids["p"], "version_of")
    resp = _lineage(client, ids["o"], relation_type="derived_from")
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_cursor_past_the_end_returns_empty_final_page(client):
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("a", "b", "c")}
    _edge(client, ids["b"], ids["a"])
    _edge(client, ids["c"], ids["b"])
    # The server only mints a cursor while items remain; a well-signed cursor
    # positioned exactly at the total (e.g. replayed after concurrent
    # history-independent recomputation) is accepted and yields an empty
    # final page rather than an error, with no further cursor.
    end_cursor = pagination.encode_lineage_cursor(
        DEFAULT_LINEAGE_CURSOR_SECRET,
        origin_id=ids["c"],
        direction="ancestors",
        max_depth=8,
        min_depth=1,
        relation_type=None,
        limit=2,
        offset=2,
    )
    resp = _lineage(client, ids["c"], limit=2, cursor=end_cursor)
    assert resp.status_code == 200
    assert resp.json()["items"] == []
    assert resp.json()["count"] == 2
    assert resp.json()["next_cursor"] is None


# --- parameter validation ----------------------------------------------------


def test_relation_type_must_be_one_of_the_two_literals(client):
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("a", "b")}
    _edge(client, ids["b"], ids["a"])
    for value in ("version", "Version_Of", "derived", "copy_of", " "):
        resp = _lineage(client, ids["b"], relation_type=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_min_depth_boundaries(client):
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("a", "b")}
    _edge(client, ids["b"], ids["a"])
    for value in (1, 32):
        resp = _lineage(client, ids["b"], min_depth=value, max_depth=32)
        assert resp.status_code == 200, value
    for value in (0, 33, -1, "1.5", "8.0", "abc", "", " 2", "2 "):
        resp = _lineage(client, ids["b"], min_depth=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_limit_boundaries(client):
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("a", "b")}
    _edge(client, ids["b"], ids["a"])
    for value in (1, 100):
        resp = _lineage(client, ids["b"], limit=value)
        assert resp.status_code == 200, value
    for value in (0, 101, -1, "1.0", "abc", "", " 5", "5 "):
        resp = _lineage(client, ids["b"], limit=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_min_depth_greater_than_max_depth_is_validation_error(client):
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("a", "b")}
    _edge(client, ids["b"], ids["a"])
    # Explicit incompatibility.
    resp = _lineage(client, ids["b"], min_depth=3, max_depth=2)
    assert resp.status_code == 422
    issue = resp.json()["error"]["details"]["issues"][0]
    assert issue["loc"] == ["query", "min_depth"]
    # min_depth above the default max_depth of 8 is incompatible too, not
    # silently clamped.
    resp = _lineage(client, ids["b"], min_depth=9)
    assert resp.status_code == 422
    # Equality is allowed.
    assert _lineage(client, ids["b"], min_depth=8, max_depth=8).status_code == 200


def test_repeated_new_parameters_are_validation_error(client):
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("a", "b")}
    _edge(client, ids["b"], ids["a"])
    base = f"/v1/contents/{ids['b']}/lineage?direction=ancestors"
    for suffix in (
        "&relation_type=version_of&relation_type=derived_from",
        "&min_depth=1&min_depth=2",
        "&limit=1&limit=2",
        "&cursor=x&cursor=y",
    ):
        resp = client.get(base + suffix)
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_new_parameter_is_not_defaulted_and_has_no_items(client):
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("a", "b")}
    _edge(client, ids["b"], ids["a"])
    resp = _lineage(client, ids["b"], min_depth=2, max_depth=1, limit=0)
    assert resp.status_code == 422
    assert "items" not in resp.json()


# --- 404 semantics preserved -------------------------------------------------


def test_unknown_origin_is_404_even_with_filters_and_cursor(client):
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("a", "b")}
    _edge(client, ids["b"], ids["a"])
    for params in (
        {"relation_type": "version_of"},
        {"min_depth": 1, "max_depth": 2, "limit": 10},
    ):
        resp = _lineage(client, "cnt_ghost", **params)
        assert resp.status_code == 404, params
        assert resp.json()["error"]["code"] == "content_not_found"

    # A cursor whose bound parameters all match the request -- including the
    # origin -- still yields 404 once traversal finds that the origin is
    # missing; cursor validation precedes, but does not replace, the lookup.
    matching_cursor = pagination.encode_lineage_cursor(
        DEFAULT_LINEAGE_CURSOR_SECRET,
        origin_id="cnt_ghost",
        direction="ancestors",
        max_depth=8,
        min_depth=1,
        relation_type=None,
        limit=1,
        offset=1,
    )
    resp = _lineage(client, "cnt_ghost", limit=1, cursor=matching_cursor)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "content_not_found"

    # Conversely, a cursor minted for a different origin is a query mismatch
    # (422), exactly like any other changed bound parameter.
    foreign_cursor = _lineage(client, ids["b"], limit=1).json()["next_cursor"]
    resp = _lineage(client, "cnt_ghost", limit=1, cursor=foreign_cursor)
    assert resp.status_code == 422


# --- read-only / no-audit guarantee ------------------------------------------


def test_filtered_paginated_queries_write_nothing(client, db_session):
    create_actor(client)
    ids = {n: _create_content(client, n)["id"] for n in ("a", "b", "c")}
    _edge(client, ids["b"], ids["a"])
    _edge(client, ids["c"], ids["b"])
    cursor = _lineage(client, ids["c"], limit=1).json()["next_cursor"]

    def counts():
        return (
            db_session.scalar(select(func.count()).select_from(Content)),
            db_session.scalar(
                select(func.count()).select_from(ContentRelation)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    before = counts()
    _lineage(client, ids["c"], relation_type="version_of")
    _lineage(client, ids["c"], relation_type="derived_from", min_depth=2)
    _lineage(client, ids["c"], limit=1, cursor=cursor)
    _lineage(client, ids["c"], relation_type="nope")          # 422
    _lineage(client, ids["c"], min_depth=9)                  # 422
    _lineage(client, ids["c"], limit=0)                      # 422
    _lineage(client, ids["c"], cursor="tampered.pair")       # 422
    _lineage(client, "cnt_ghost", limit=1)                   # 404
    db_session.expire_all()
    assert counts() == before
