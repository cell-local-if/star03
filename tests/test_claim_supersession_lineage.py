"""Tests for the read-only multi-hop claim supersession lineage endpoint.

Covers GET /v1/claims/{claim_id}/supersession-lineage: bidirectional
traversal (newer/older), convergence/deduplication at shortest depth, depth
truncation, depth-then-creation ordering, bounded termination over anomalous
cyclic history, parameter validation, the missing-origin 404, and the
read-only zero-write guarantee. All fixtures are deterministic and offline.
"""

from __future__ import annotations

from sqlalchemy import func, select

from provenance.ids import claim_supersession_id
from provenance.models import AuditEvent, Claim, ClaimSupersession
from tests.helpers import DIGEST_A, content_payload, create_actor

SUPERSESSIONS_PATH = "/v1/claim-supersessions"


def _create_content(client, digest=DIGEST_A, actor_id="org-1"):
    resp = client.post(
        "/v1/contents",
        json=content_payload(actor_id=actor_id, digest=digest),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, name, claim_type="authorship"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": "org-1",
            "claim_type": claim_type,
            "payload": {"statement": name},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _supersede(client, old, new, reason="corrected source claim"):
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


def _lineage(client, claim_id, direction, **params):
    return client.get(
        f"/v1/claims/{claim_id}/supersession-lineage",
        params={"direction": direction, **params},
    )


def _setup_chain(client, names):
    """Create claims on one content; each claim supersedes the previous.

    Returns a name -> claim id map. The first name is the oldest claim, the
    last is the newest.
    """
    create_actor(client)
    content = _create_content(client)
    ids = {name: _create_claim(client, content["id"], name)["id"] for name in names}
    for old, new in zip(names[:-1], names[1:]):
        _supersede(client, ids[old], ids[new])
    return ids


def _ordered(items):
    return [(item["payload_digest_hex"], item["depth"]) for item in items]


def _name_digest(client, claim_id):
    return client.get(f"/v1/claims/{claim_id}").json()["payload_digest_hex"]


# --- Bidirectional traversal ------------------------------------------------


def test_newer_walks_superseded_to_replacement_excluding_origin(client):
    ids = _setup_chain(client, ["a", "b", "c"])  # b replaces a, c replaces b
    resp = _lineage(client, ids["a"], "newer")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"items", "count"}
    assert body["count"] == 2
    digests = {name: _name_digest(client, claim_id) for name, claim_id in ids.items()}
    assert _ordered(body["items"]) == [(digests["b"], 1), (digests["c"], 2)]
    assert all(item["id"] != ids["a"] for item in body["items"])


def test_older_walks_replacement_back_to_superseded_excluding_origin(client):
    ids = _setup_chain(client, ["a", "b", "c"])
    resp = _lineage(client, ids["c"], "older")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    digests = {name: _name_digest(client, claim_id) for name, claim_id in ids.items()}
    assert _ordered(body["items"]) == [(digests["b"], 1), (digests["a"], 2)]
    assert all(item["id"] != ids["c"] for item in body["items"])


def test_lineage_item_is_full_public_claim_view_plus_depth(client):
    ids = _setup_chain(client, ["a", "b"])
    resp = _lineage(client, ids["a"], "newer")
    item = resp.json()["items"][0]
    assert set(item) == {
        "id",
        "content_id",
        "actor_id",
        "claim_type",
        "payload_digest_algorithm",
        "payload_digest_hex",
        "created_at",
        "depth",
    }
    # The raw payload is never stored, so it can never be echoed.
    assert "payload" not in item
    fetched = client.get(f"/v1/claims/{ids['b']}").json()
    for field in (
        "id",
        "content_id",
        "actor_id",
        "claim_type",
        "payload_digest_algorithm",
        "payload_digest_hex",
        "created_at",
    ):
        assert item[field] == fetched[field]
    assert item["depth"] == 1


def test_empty_traversal_returns_empty_set_not_404(client):
    ids = _setup_chain(client, ["a", "b"])
    # The oldest claim has no older history; the newest has nothing newer.
    for origin, direction in ((ids["a"], "older"), (ids["b"], "newer")):
        resp = _lineage(client, origin, direction)
        assert resp.status_code == 200
        assert resp.json() == {"items": [], "count": 0}


def test_unknown_origin_is_claim_not_found(client):
    create_actor(client)
    for direction in ("newer", "older"):
        resp = _lineage(client, "clm_ghost", direction)
        assert resp.status_code == 404
        error = resp.json()["error"]
        assert error["code"] == "claim_not_found"
        assert error["details"]["claim_id"] == "clm_ghost"


# --- Convergence and shortest depth -----------------------------------------


def test_converging_paths_dedup_claim_at_shortest_depth(client):
    # Diamond: b and c both replace a; d replaces both b and c.
    ids = _setup_chain(client, ["x"])  # actor + one unrelated claim
    for name in ("a", "b", "c", "d"):
        content_id = client.get(f"/v1/claims/{ids['x']}").json()["content_id"]
        ids[name] = _create_claim(client, content_id, name)["id"]
    _supersede(client, ids["a"], ids["b"])
    _supersede(client, ids["a"], ids["c"])
    _supersede(client, ids["b"], ids["d"])
    _supersede(client, ids["c"], ids["d"])

    digests = {
        name: _name_digest(client, claim_id) for name, claim_id in ids.items()
    }

    resp = _lineage(client, ids["a"], "newer")
    assert resp.status_code == 200
    assert _ordered(resp.json()["items"]) == [
        (digests["b"], 1),
        (digests["c"], 1),
        (digests["d"], 2),
    ]

    # Reverse direction converges too: d reaches b, c at depth 1 and a at 2.
    resp = _lineage(client, ids["d"], "older")
    assert _ordered(resp.json()["items"]) == [
        (digests["b"], 1),
        (digests["c"], 1),
        (digests["a"], 2),
    ]


def test_claim_reachable_at_two_depths_keeps_the_shortest_once(client):
    # o is directly superseded by a (depth 1) and indirectly o -> x -> a.
    create_actor(client)
    content = _create_content(client)
    ids = {
        name: _create_claim(client, content["id"], name)["id"]
        for name in ("o", "x", "a")
    }
    _supersede(client, ids["o"], ids["x"])
    _supersede(client, ids["o"], ids["a"])
    _supersede(client, ids["x"], ids["a"])

    resp = _lineage(client, ids["o"], "newer")
    assert resp.status_code == 200
    items = resp.json()["items"]
    a_entries = [item for item in items if item["id"] == ids["a"]]
    assert len(a_entries) == 1
    assert a_entries[0]["depth"] == 1
    digests = {
        name: _name_digest(client, claim_id) for name, claim_id in ids.items()
    }
    assert _ordered(items) == [(digests["x"], 1), (digests["a"], 1)]


# --- Truncation --------------------------------------------------------------


def test_max_depth_truncates_beyond_the_limit(client):
    ids = _setup_chain(client, ["a", "b", "c", "d", "e"])
    resp = _lineage(client, ids["a"], "newer", max_depth=2)
    assert resp.status_code == 200
    digests = {name: _name_digest(client, claim_id) for name, claim_id in ids.items()}
    assert _ordered(resp.json()["items"]) == [(digests["b"], 1), (digests["c"], 2)]


def test_max_depth_one_returns_only_direct_replacements(client):
    ids = _setup_chain(client, ["a", "b", "c"])
    resp = _lineage(client, ids["a"], "newer", max_depth=1)
    digests = {name: _name_digest(client, claim_id) for name, claim_id in ids.items()}
    assert _ordered(resp.json()["items"]) == [(digests["b"], 1)]


def test_default_max_depth_is_eight(client):
    names = [f"claim-{i:02d}" for i in range(11)]  # 11 claims, 10 edges
    ids = _setup_chain(client, names)
    resp = _lineage(client, ids[names[0]], "newer")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 8
    assert [item["depth"] for item in items] == list(range(1, 9))


def test_max_depth_boundaries_are_accepted(client):
    ids = _setup_chain(client, ["a", "b"])
    for depth in (1, 32):
        resp = _lineage(client, ids["a"], "newer", max_depth=depth)
        assert resp.status_code == 200, resp.text


# --- Ordering ----------------------------------------------------------------


def test_within_level_order_uses_supersession_creation_order(client):
    create_actor(client)
    content = _create_content(client)
    # Claims are deliberately created in an order that differs from the
    # supersession edge order below.
    ids = {
        name: _create_claim(client, content["id"], name)["id"]
        for name in ("o", "a", "b", "y", "x")
    }
    # Supersession edges, in strict creation order:
    _supersede(client, ids["o"], ids["a"])  # depth 1: a discovered first
    _supersede(client, ids["o"], ids["b"])  # depth 1: b second
    _supersede(client, ids["b"], ids["x"])  # b's depth-2 edge is created FIRST
    _supersede(client, ids["a"], ids["y"])  # a's depth-2 edge is created after

    resp = _lineage(client, ids["o"], "newer")
    items = resp.json()["items"]
    # Depth 1 follows the edges out of o (a before b), not claim order.
    # Depth 2 is globally ordered by discovering-edge creation: b -> x was
    # created before a -> y, so x precedes y even though a precedes b.
    digests = {
        name: _name_digest(client, claim_id) for name, claim_id in ids.items()
    }
    assert _ordered(items) == [
        (digests["a"], 1),
        (digests["b"], 1),
        (digests["x"], 2),
        (digests["y"], 2),
    ]


# --- Cycles in anomalous history ---------------------------------------------


def test_cycle_in_anomalous_history_terminates_deduped(client, db_session):
    ids = _setup_chain(client, ["a", "b", "c"])
    # Bypass the service cycle guard to simulate anomalous history: a being
    # superseded by c closes a -> b -> c -> a.
    cyclic = ClaimSupersession(
        id=claim_supersession_id(ids["c"], ids["a"], "anomalous correction"),
        superseded_claim_id=ids["c"],
        replacement_claim_id=ids["a"],
        reason="anomalous correction",
    )
    db_session.add(cyclic)
    db_session.commit()

    digests = {name: _name_digest(client, claim_id) for name, claim_id in ids.items()}

    newer = _lineage(client, ids["a"], "newer")
    assert newer.status_code == 200
    assert _ordered(newer.json()["items"]) == [
        (digests["b"], 1),
        (digests["c"], 2),
    ]

    older = _lineage(client, ids["c"], "older")
    assert older.status_code == 200
    assert _ordered(older.json()["items"]) == [
        (digests["b"], 1),
        (digests["a"], 2),
    ]

    # Every claim is reported at most once and the origin is never included.
    for body, origin in (
        (newer.json(), ids["a"]),
        (older.json(), ids["c"]),
    ):
        returned = [item["id"] for item in body["items"]]
        assert len(returned) == len(set(returned))
        assert origin not in returned


def test_max_depth_bounds_walk_even_with_a_cycle(client, db_session):
    ids = _setup_chain(client, ["a", "b"])
    db_session.add(
        ClaimSupersession(
            id=claim_supersession_id(ids["b"], ids["a"], "anomalous correction"),
            superseded_claim_id=ids["b"],
            replacement_claim_id=ids["a"],
            reason="anomalous correction",
        )
    )
    db_session.commit()
    # A direct a <-> b loop: depth limiting alone would keep the walk finite
    # even without the visited set.
    resp = _lineage(client, ids["a"], "newer", max_depth=32)
    assert resp.status_code == 200
    digests = {name: _name_digest(client, claim_id) for name, claim_id in ids.items()}
    assert _ordered(resp.json()["items"]) == [(digests["b"], 1)]


# --- Parameter validation ----------------------------------------------------


def test_direction_is_required(client):
    ids = _setup_chain(client, ["a"])
    resp = client.get(f"/v1/claims/{ids['a']}/supersession-lineage")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_blank_or_illegal_direction_is_validation_error(client):
    ids = _setup_chain(client, ["a"])
    for value in ("", "   ", "up", "NEWER", "newest", "ancestors", "old"):
        resp = _lineage(client, ids["a"], value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_max_depth_empty_non_integer_or_out_of_range_is_validation_error(client):
    ids = _setup_chain(client, ["a"])
    for value in ("", "abc", "1.5", "0", "-1", "33", "8.0", "  8"):
        resp = _lineage(client, ids["a"], "newer", max_depth=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_validation_error(client):
    ids = _setup_chain(client, ["a", "b"])
    base = f"/v1/claims/{ids['a']}/supersession-lineage"
    for url in (
        f"{base}?direction=newer&max_depth=1&max_depth=2",
        f"{base}?direction=newer&max_depth=1&max_depth=1",
        f"{base}?direction=newer&direction=older",
        f"{base}?direction=newer&direction=newer",
    ):
        resp = client.get(url)
        assert resp.status_code == 422, url
        assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_parameters_are_never_silently_defaulted(client):
    ids = _setup_chain(client, ["a", "b"])
    # A bad direction must not fall back to either traversal; the response is
    # an error, not a defaulted result.
    resp = _lineage(client, ids["a"], "sideways")
    assert resp.status_code == 422
    assert "items" not in resp.json()


# --- Read-only guarantee -----------------------------------------------------


def test_lineage_queries_write_no_resources_or_audit_events(client, db_session):
    ids = _setup_chain(client, ["a", "b", "c"])

    def counts():
        return (
            db_session.scalar(select(func.count()).select_from(Claim)),
            db_session.scalar(
                select(func.count()).select_from(ClaimSupersession)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    before = counts()
    _lineage(client, ids["a"], "newer")
    _lineage(client, ids["c"], "older", max_depth=1)
    _lineage(client, ids["a"], "older")  # empty result
    _lineage(client, "clm_ghost", "newer")  # 404
    _lineage(client, ids["a"], "bad")  # 422
    client.get(
        f"/v1/claims/{ids['a']}/supersession-lineage"
        "?direction=newer&max_depth=1&max_depth=2"
    )  # 422
    db_session.expire_all()
    assert counts() == before
