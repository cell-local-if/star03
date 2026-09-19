"""Tests for the read-only multi-hop lineage endpoint.

Covers GET /v1/contents/{content_id}/lineage in both directions: bidirectional
reachability, convergence with shortest-depth dedup, max-depth truncation,
within-depth ordering by first-discovery relation creation order, bounded
termination when anomalous history contains a cycle, query parameter
validation, 404 handling, empty results, and the no-write guarantee.
"""

from __future__ import annotations

import time

import pytest
from sqlalchemy import select

from provenance.ids import content_relation_id
from provenance.models import AuditEvent, Content, ContentRelation
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    content_payload,
    create_actor,
)

# Extra deterministic digests for wider graphs.
import hashlib

DIGEST_D = hashlib.sha256(b"content-d").hexdigest()
DIGEST_E = hashlib.sha256(b"content-e").hexdigest()
DIGEST_F = hashlib.sha256(b"content-f").hexdigest()
DIGEST_G = hashlib.sha256(b"content-g").hexdigest()


def _create_content(client, digest, actor_id="org-1", title=None):
    resp = client.post(
        "/v1/contents",
        json=content_payload(
            actor_id=actor_id, digest=digest, title=title
        ),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _relation(client, content_id, parent_content_id, relation_type="version_of"):
    resp = client.post(
        "/v1/content-relations",
        json={
            "content_id": content_id,
            "parent_content_id": parent_content_id,
            "relation_type": relation_type,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _setup(client, digests):
    create_actor(client)
    contents = [_create_content(client, digest=d) for d in digests]
    return contents


def _lineage(client, content_id, **params):
    return client.get(f"/v1/contents/{content_id}/lineage", params=params)


# --- happy path: fields, empty results, 404 --------------------------------


def test_lineage_item_has_full_public_content_fields_plus_depth(client):
    a, b = _setup(client, [DIGEST_A, DIGEST_B])
    _relation(client, b["id"], a["id"])
    resp = _lineage(client, b["id"], direction="ancestors")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 1
    item = body["items"][0]
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
    # Every existing public content field matches the canonical resource.
    expected = client.get(f"/v1/contents/{a['id']}").json()
    for key, value in expected.items():
        assert item[key] == value
    assert item["id"] == a["id"]
    assert item["depth"] == 1
    assert isinstance(item["depth"], int)


def test_lineage_excludes_origin(client):
    a, b = _setup(client, [DIGEST_A, DIGEST_B])
    _relation(client, b["id"], a["id"])
    for direction in ("ancestors", "descendants"):
        body = _lineage(client, a["id"], direction=direction).json()
        assert all(item["id"] != a["id"] for item in body["items"])


def test_lineage_empty_collection_when_nothing_reachable(client):
    a, b = _setup(client, [DIGEST_A, DIGEST_B])
    _relation(client, b["id"], a["id"])
    # a has no ancestors; b has no descendants.
    assert _lineage(client, a["id"], direction="ancestors").json() == {
        "items": [],
        "count": 0,
    }
    assert _lineage(client, b["id"], direction="descendants").json() == {
        "items": [],
        "count": 0,
    }


def test_lineage_unknown_content_is_not_found(client):
    resp = _lineage(client, "cnt_missing", direction="ancestors")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "content_not_found"
    assert error["details"]["content_id"] == "cnt_missing"


# --- bidirectional traversal -----------------------------------------------


def test_ancestors_and_descendants_are_reverse_traversals(client):
    # Chain: a <- b <- d <- e
    a, b, d, e = _setup(client, [DIGEST_A, DIGEST_B, DIGEST_D, DIGEST_E])
    _relation(client, b["id"], a["id"])
    _relation(client, d["id"], b["id"])
    _relation(client, e["id"], d["id"])

    ancestors = _lineage(client, e["id"], direction="ancestors").json()
    assert [(i["id"], i["depth"]) for i in ancestors["items"]] == [
        (d["id"], 1),
        (b["id"], 2),
        (a["id"], 3),
    ]
    assert ancestors["count"] == 3

    descendants = _lineage(client, a["id"], direction="descendants").json()
    assert [(i["id"], i["depth"]) for i in descendants["items"]] == [
        (b["id"], 1),
        (d["id"], 2),
        (e["id"], 3),
    ]
    assert descendants["count"] == 3

def test_lineage_both_directions_over_branching_graph(client):
    # a is parent of b and d; b is parent of e.
    a, b, d, e = _setup(client, [DIGEST_A, DIGEST_B, DIGEST_D, DIGEST_E])
    _relation(client, b["id"], a["id"])
    _relation(client, d["id"], a["id"], "derived_from")
    _relation(client, e["id"], b["id"])

    assert [i["id"] for i in _lineage(client, a["id"], direction="descendants").json()["items"]] == [
        b["id"],
        d["id"],
        e["id"],
    ]
    assert [i["id"] for i in _lineage(client, e["id"], direction="ancestors").json()["items"]] == [
        b["id"],
        a["id"],
    ]
    # d shares the a ancestor but is not on e's path.
    assert [
        i["id"]
        for i in _lineage(client, d["id"], direction="ancestors").json()["items"]
    ] == [a["id"]]


# --- convergence / dedup / shortest depth ----------------------------------


def test_converging_paths_dedup_and_take_shortest_depth(client):
    # Diamond: a <- b, a <- d, b <- e, d <- e.
    # e reaches a at depth 2 via both b and d; a appears once with depth 2.
    a, b, d, e = _setup(client, [DIGEST_A, DIGEST_B, DIGEST_D, DIGEST_E])
    _relation(client, b["id"], a["id"])
    _relation(client, d["id"], a["id"], "derived_from")
    _relation(client, e["id"], b["id"])
    _relation(client, e["id"], d["id"], "derived_from")

    ancestors = _lineage(client, e["id"], direction="ancestors").json()
    by_id = {item["id"]: item["depth"] for item in ancestors["items"]}
    assert by_id == {b["id"]: 1, d["id"]: 1, a["id"]: 2}
    assert ancestors["count"] == 3
    assert [item["id"] for item in ancestors["items"]].count(a["id"]) == 1

    # Same diamond in reverse from a: e reached at depth 2 via both parents,
    # listed once.
    descendants = _lineage(client, a["id"], direction="descendants").json()
    by_id = {item["id"]: item["depth"] for item in descendants["items"]}
    assert by_id == {b["id"]: 1, d["id"]: 1, e["id"]: 2}
    assert descendants["count"] == 3


def test_longer_converging_path_does_not_change_shortest_depth(client):
    # a <- b <- d ; a <- f <- g <- d (d reaches a at depth 2 via b and at
    # depth 3 via g/f; the reported depth must stay 2).
    a, b, d, f, g = _setup(
        client, [DIGEST_A, DIGEST_B, DIGEST_D, DIGEST_F, DIGEST_G]
    )
    _relation(client, b["id"], a["id"])
    _relation(client, d["id"], b["id"])
    _relation(client, f["id"], a["id"])
    _relation(client, g["id"], f["id"])
    _relation(client, d["id"], g["id"], "derived_from")

    depths = {
        item["id"]: item["depth"]
        for item in _lineage(client, d["id"], direction="ancestors").json()["items"]
    }
    assert depths == {b["id"]: 1, g["id"]: 1, a["id"]: 2, f["id"]: 2}


def test_two_relation_types_between_same_pair_dedup_to_one_item(client):
    # Both relation types are independent relation rows between the same two
    # contents, but lineage traverses contents and must list the neighbor
    # once at depth 1 in both directions.
    a, b = _setup(client, [DIGEST_A, DIGEST_B])
    _relation(client, b["id"], a["id"], "version_of")
    _relation(client, b["id"], a["id"], "derived_from")

    ancestors = _lineage(client, b["id"], direction="ancestors").json()
    assert ancestors == {
        "items": [
            {
                **client.get(f"/v1/contents/{a['id']}").json(),
                "depth": 1,
            }
        ],
        "count": 1,
    }
    descendants = _lineage(client, a["id"], direction="descendants").json()
    assert descendants["count"] == 1
    assert descendants["items"][0]["id"] == b["id"]
    assert descendants["items"][0]["depth"] == 1


# --- truncation ------------------------------------------------------------


def test_max_depth_truncates_traversal(client):
    # Chain a <- b <- d <- e.
    a, b, d, e = _setup(client, [DIGEST_A, DIGEST_B, DIGEST_D, DIGEST_E])
    _relation(client, b["id"], a["id"])
    _relation(client, d["id"], b["id"])
    _relation(client, e["id"], d["id"])

    for depth, expected in (
        (1, [(d["id"], 1)]),
        (2, [(d["id"], 1), (b["id"], 2)]),
        (3, [(d["id"], 1), (b["id"], 2), (a["id"], 3)]),
        (32, [(d["id"], 1), (b["id"], 2), (a["id"], 3)]),
    ):
        body = _lineage(
            client, e["id"], direction="ancestors", max_depth=depth
        ).json()
        assert [(i["id"], i["depth"]) for i in body["items"]] == expected
        assert body["count"] == len(expected)


def test_max_depth_defaults_to_eight(client):
    # A ten-deep chain from a root: the 8 nearest ancestors are returned and
    # the two furthest are truncated.
    digests = [
        hashlib.sha256(f"chain-{n}".encode()).hexdigest() for n in range(10)
    ]
    contents = _setup(client, digests)
    for child, parent in zip(contents[1:], contents[:-1]):
        _relation(client, child["id"], parent["id"])

    body = _lineage(client, contents[-1]["id"], direction="ancestors").json()
    assert body["count"] == 8
    assert [item["depth"] for item in body["items"]] == list(range(1, 9))
    assert body["items"][-1]["id"] == contents[1]["id"]
    assert contents[0]["id"] not in {item["id"] for item in body["items"]}


def test_max_depth_boundaries_accepted(client):
    a, b = _setup(client, [DIGEST_A, DIGEST_B])
    _relation(client, b["id"], a["id"])
    for value in (1, 32):
        resp = _lineage(
            client, b["id"], direction="ancestors", max_depth=value
        )
        assert resp.status_code == 200, value


# --- ordering --------------------------------------------------------------


def test_within_depth_ordered_by_first_discovery_relation_creation_order(
    client,
):
    # Three independent parents of e; their order must follow the stable
    # creation order of the edges discovering them, regardless of content
    # creation order or relation type.
    a, b, d, e = _setup(client, [DIGEST_A, DIGEST_B, DIGEST_D, DIGEST_E])

    # Create edges with distinct, guaranteed timestamps in a fixed order.
    edge_e_b = _relation(client, e["id"], b["id"])
    time.sleep(0.01)
    edge_e_a = _relation(client, e["id"], a["id"], "derived_from")
    time.sleep(0.01)
    edge_e_d = _relation(client, e["id"], d["id"])
    assert edge_e_b["created_at"] < edge_e_a["created_at"] < edge_e_d["created_at"]

    body = _lineage(client, e["id"], direction="ancestors").json()
    assert [item["id"] for item in body["items"]] == [
        b["id"],
        a["id"],
        d["id"],
    ]
    assert {item["depth"] for item in body["items"]} == {1}

    # Descendants of a single parent with three children follow the same
    # edge order.
    body = _lineage(client, a["id"], direction="descendants").json()
    assert [item["id"] for item in body["items"]] == [e["id"]]


def test_ordering_is_depth_ascending_with_edge_order_inside_depth(client):
    # Layered graph from origin a:
    #   depth 1: b, d   depth 2: e (via b), f (via d)
    # The depth-2 edges are inserted interleaved with the depth-1 edges:
    # f's discovering edge is created before e's. Depth ordering must
    # dominate, and inside depth 2 the first-discovery edges' stable
    # creation order puts f before e.
    a, b, d, e, f = _setup(
        client, [DIGEST_A, DIGEST_B, DIGEST_D, DIGEST_E, DIGEST_F]
    )
    _relation(client, b["id"], a["id"])  # r1: b discovered at depth 1
    time.sleep(0.01)
    edge_f_d = _relation(client, f["id"], d["id"])  # r2: f discovered
    time.sleep(0.01)
    _relation(
        client, d["id"], a["id"], "derived_from"
    )  # r3: d discovered at depth 1
    time.sleep(0.01)
    edge_e_b = _relation(client, e["id"], b["id"])  # r4: e discovered

    body = _lineage(client, a["id"], direction="descendants").json()
    assert [(i["id"], i["depth"]) for i in body["items"]] == [
        (b["id"], 1),
        (d["id"], 1),
        (f["id"], 2),
        (e["id"], 2),
    ]
    # f's discovering edge (r2) predates e's (r4), so f wins the within-
    # depth ordering even though f is reached through the later depth-1
    # node d.
    assert edge_f_d["created_at"] < edge_e_b["created_at"]


# --- cycle safety (anomalous history) --------------------------------------


def _insert_raw_relation(db_session, child_id, parent_id, relation_type, seq):
    """Insert a relation edge bypassing cycle checks (anomalous history)."""
    relation = ContentRelation(
        id=content_relation_id(child_id, parent_id, relation_type)
        + f"x{seq}",
        content_id=child_id,
        parent_content_id=parent_id,
        relation_type=relation_type,
    )
    db_session.add(relation)
    db_session.commit()
    return relation


def test_traversal_terminates_on_cycle_in_anomalous_history(client, db_session):
    # Build a normal chain a <- b <- d through the API, then force a closing
    # edge a -> d directly into storage, producing the cycle
    # a -> d -> b -> a (edges point content -> parent).
    a, b, d = _setup(client, [DIGEST_A, DIGEST_B, DIGEST_D])
    _relation(client, b["id"], a["id"])
    _relation(client, d["id"], b["id"])
    _insert_raw_relation(db_session, a["id"], d["id"], "version_of", 1)

    # Every direction terminates with a finite, cycle-deduped result set.
    ancestors = _lineage(client, d["id"], direction="ancestors").json()
    assert ancestors["count"] == 2
    assert {item["id"] for item in ancestors["items"]} == {b["id"], a["id"]}
    assert {item["depth"] for item in ancestors["items"]} == {1, 2}

    descendants = _lineage(client, a["id"], direction="descendants").json()
    assert descendants["count"] == 2
    assert {item["id"] for item in descendants["items"]} == {d["id"], b["id"]}
    assert {item["depth"] for item in descendants["items"]} == {1, 2}

    # Starting on the forced-edge node terminates as well and never returns
    # the origin.
    body = _lineage(client, a["id"], direction="ancestors").json()
    assert a["id"] not in {item["id"] for item in body["items"]}
    assert body["count"] == 2

    # A tiny max_depth on a cyclic graph is still bounded.
    body = _lineage(
        client, a["id"], direction="ancestors", max_depth=1
    ).json()
    assert [item["id"] for item in body["items"]] == [d["id"]]


def test_self_loop_in_anomalous_history_is_ignored(client, db_session):
    (a,) = _setup(client, [DIGEST_A])
    _insert_raw_relation(db_session, a["id"], a["id"], "version_of", 2)

    for direction in ("ancestors", "descendants"):
        body = _lineage(client, a["id"], direction=direction).json()
        assert body == {"items": [], "count": 0}


# --- parameter validation --------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "",  # direction missing entirely
        "?direction=",  # empty
        "?direction=%20%20%09",  # whitespace only
        "?direction=upstream",
        "?direction=ANCESTORS",
        "?direction=ancestor",
        "?direction=descendant",
        "?direction=ancestors%20",
    ],
)
def test_invalid_direction_is_validation_error(client, query):
    resp = client.get(f"/v1/contents/x/lineage{query}")
    assert resp.status_code == 422, query
    assert resp.json()["error"]["code"] == "validation_error"
    issue_locs = [
        tuple(issue["loc"]) for issue in resp.json()["error"]["details"]["issues"]
    ]
    assert ("query", "direction") in issue_locs


@pytest.mark.parametrize(
    "value",
    [
        "",  # empty
        "abc",  # non-integer
        "1.5",
        "true",
        "1e2",
        "0",  # out of range
        "-1",
        "33",
        "100",
        "999999999999999999999999",
        "+1",
        "01",
        " 1",
    ],
)
def test_invalid_max_depth_is_validation_error(client, value):
    resp = client.get(
        f"/v1/contents/x/lineage?direction=ancestors&max_depth={value}"
    )
    assert resp.status_code == 422, value
    assert resp.json()["error"]["code"] == "validation_error"
    issue_locs = [
        tuple(issue["loc"]) for issue in resp.json()["error"]["details"]["issues"]
    ]
    assert ("query", "max_depth") in issue_locs


def test_duplicate_max_depth_is_validation_error_even_if_values_equal(client):
    resp = client.get(
        "/v1/contents/x/lineage?direction=ancestors&max_depth=2&max_depth=2"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_duplicate_max_depth_different_values_is_validation_error(client):
    resp = client.get(
        "/v1/contents/x/lineage?direction=ancestors&max_depth=1&max_depth=3"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_duplicate_direction_is_validation_error(client):
    resp = client.get(
        "/v1/contents/x/lineage"
        "?direction=ancestors&direction=descendants"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_unknown_content_takes_precedence_over_valid_query(client):
    # Sanity: a well-formed query on a missing id is the documented 404, not
    # a validation error or an empty collection.
    resp = _lineage(client, "cnt_ghost", direction="descendants")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "content_not_found"


# --- no writes -------------------------------------------------------------


def test_lineage_writes_no_rows_or_audit_events(client, db_session):
    a, b, d, e = _setup(client, [DIGEST_A, DIGEST_B, DIGEST_D, DIGEST_E])
    _relation(client, b["id"], a["id"])
    _relation(client, d["id"], b["id"])
    _relation(client, e["id"], d["id"])

    relations_before = {
        (r.content_id, r.parent_content_id)
        for r in db_session.execute(select(ContentRelation)).scalars().all()
    }
    events_before = list(
        db_session.execute(select(AuditEvent)).scalars().all()
    )
    contents_before = list(db_session.execute(select(Content)).scalars().all())

    for direction in ("ancestors", "descendants"):
        for depth in (1, 2, 8):
            resp = _lineage(
                client, e["id"], direction=direction, max_depth=depth
            )
            assert resp.status_code == 200
    # Empty and 404 cases must not write either.
    _lineage(client, a["id"], direction="ancestors")
    _lineage(client, "cnt_ghost", direction="ancestors")

    relations_after = {
        (r.content_id, r.parent_content_id)
        for r in db_session.execute(select(ContentRelation)).scalars().all()
    }
    assert relations_after == relations_before
    assert list(db_session.execute(select(Content)).scalars().all()) == contents_before
    events_after = list(db_session.execute(select(AuditEvent)).scalars().all())
    assert len(events_after) == len(events_before)
    assert [(e.seq, e.event_type, e.resource_id) for e in events_after] == [
        (e.seq, e.event_type, e.resource_id) for e in events_before
    ]


def test_invalid_lineage_requests_write_nothing(client, db_session):
    a, b = _setup(client, [DIGEST_A, DIGEST_B])
    _relation(client, b["id"], a["id"])
    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())
    relations_before = len(
        db_session.execute(select(ContentRelation)).scalars().all()
    )

    bad_requests = [
        client.get(f"/v1/contents/{b['id']}/lineage"),
        client.get(f"/v1/contents/{b['id']}/lineage?direction="),
        client.get(f"/v1/contents/{b['id']}/lineage?direction=sideways"),
        client.get(
            f"/v1/contents/{b['id']}/lineage?direction=ancestors&max_depth=0"
        ),
        client.get(
            f"/v1/contents/{b['id']}/lineage?direction=ancestors&max_depth=33"
        ),
        client.get(
            f"/v1/contents/{b['id']}/lineage?direction=ancestors&max_depth=x"
        ),
        client.get(
            f"/v1/contents/{b['id']}/lineage"
            "?direction=ancestors&max_depth=1&max_depth=2"
        ),
    ]
    assert all(r.status_code == 422 for r in bad_requests)
    assert len(db_session.execute(select(ContentRelation)).scalars().all()) == relations_before
    assert len(db_session.execute(select(AuditEvent)).scalars().all()) == events_before
