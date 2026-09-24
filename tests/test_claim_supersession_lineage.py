"""Tests for the read-only multi-hop supersession-lineage endpoint.

Covers GET /v1/claims/{claim_id}/supersession-lineage: bidirectional
traversal (newer/older), convergence/deduplication at shortest depth, depth
truncation, depth-then-creation ordering, bounded termination over anomalous
cyclic history, parameter validation, the missing-origin 404, and the
read-only/zero-write guarantee. All fixtures are deterministic and offline.
"""

from __future__ import annotations

from sqlalchemy import func, select

from provenance.ids import claim_supersession_id
from provenance.models import AuditEvent, Claim, ClaimSupersession
from tests.helpers import DIGEST_A, content_payload, create_actor

PAYLOADS = [{"statement": f"version {i}"} for i in range(12)]


def _create_content(client, digest=DIGEST_A, actor_id="org-1"):
    resp = client.post(
        "/v1/contents",
        json=content_payload(actor_id=actor_id, digest=digest),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, payload, claim_type="authorship"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": "org-1",
            "claim_type": claim_type,
            "payload": payload,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _supersede(client, old, new, reason="corrected source claim"):
    resp = client.post(
        "/v1/claim-supersessions",
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
    """Create one content and one claim per name; edge each claim -> previous."""
    create_actor(client)
    content = _create_content(client)
    claims = {
        name: _create_claim(client, content["id"], PAYLOADS[i])
        for i, name in enumerate(names)
    }
    for older, newer in zip(names[:-1], names[1:]):
        _supersede(client, claims[older]["id"], claims[newer]["id"])
    return claims


def _ordered(items):
    return [item["depth"] for item in items]


# --- Bidirectional traversal ------------------------------------------------


def test_newer_walks_superseded_to_replacement_excluding_origin(client):
    claims = _setup_chain(client, ["a", "b", "c"])  # b replaces a, c replaces b
    resp = _lineage(client, claims["a"]["id"], "newer")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"items", "count", "next_cursor"}
    assert body["next_cursor"] is None
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [
        claims["b"]["id"],
        claims["c"]["id"],
    ]
    assert _ordered(body["items"]) == [1, 2]
    assert all(item["id"] != claims["a"]["id"] for item in body["items"])


def test_older_walks_replacement_to_superseded_excluding_origin(client):
    claims = _setup_chain(client, ["a", "b", "c"])
    resp = _lineage(client, claims["c"]["id"], "older")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [
        claims["b"]["id"],
        claims["a"]["id"],
    ]
    assert _ordered(body["items"]) == [1, 2]
    assert all(item["id"] != claims["c"]["id"] for item in body["items"])


def test_lineage_item_is_full_public_claim_view_plus_depth(client):
    claims = _setup_chain(client, ["a", "b"])
    item = _lineage(client, claims["a"]["id"], "newer").json()["items"][0]
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
    fetched = client.get(f"/v1/claims/{claims['b']['id']}").json()
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
    # The raw payload is never stored and therefore never echoed.
    assert "payload" not in item


def test_empty_traversal_returns_empty_set_not_404(client):
    claims = _setup_chain(client, ["a", "b"])
    # b is the newest: nothing newer; a is the oldest: nothing older.
    for origin, direction in (
        (claims["b"]["id"], "newer"),
        (claims["a"]["id"], "older"),
    ):
        resp = _lineage(client, origin, direction)
        assert resp.status_code == 200
        assert resp.json() == {
            "items": [],
            "count": 0,
            "next_cursor": None,
        }


def test_unknown_origin_is_claim_not_found(client):
    create_actor(client)
    for direction in ("newer", "older"):
        resp = _lineage(client, "clm_ghost", direction)
        assert resp.status_code == 404
        error = resp.json()["error"]
        assert error["code"] == "claim_not_found"
        assert error["details"]["claim_id"] == "clm_ghost"


# --- Convergence and shortest depth ----------------------------------------


def test_converging_paths_dedup_claim_at_shortest_depth(client):
    # Diamond: d replaces both b and c; b and c each replace a.
    claims = _setup_chain(client, ["x"])  # actor + content + one lone claim
    for name in ("a", "b", "c", "d"):
        claims[name] = _create_claim(
            client, claims["x"]["content_id"], PAYLOADS[len(claims)]
        )
    _supersede(client, claims["a"]["id"], claims["b"]["id"])
    _supersede(client, claims["a"]["id"], claims["c"]["id"], "second correction")
    _supersede(client, claims["b"]["id"], claims["d"]["id"])
    _supersede(client, claims["c"]["id"], claims["d"]["id"], "second correction")

    resp = _lineage(client, claims["a"]["id"], "newer")
    assert resp.status_code == 200
    body = resp.json()
    assert [item["id"] for item in body["items"]] == [
        claims["b"]["id"],
        claims["c"]["id"],
        claims["d"]["id"],
    ]
    assert _ordered(body["items"]) == [1, 1, 2]

    # Reverse direction converges too: d reaches b, c at depth 1 and a at 2.
    resp = _lineage(client, claims["d"]["id"], "older")
    body = resp.json()
    assert [item["id"] for item in body["items"]] == [
        claims["b"]["id"],
        claims["c"]["id"],
        claims["a"]["id"],
    ]
    assert _ordered(body["items"]) == [1, 1, 2]


def test_claim_reachable_at_two_depths_keeps_the_shortest_once(client):
    create_actor(client)
    content = _create_content(client)
    claims = {
        name: _create_claim(client, content["id"], PAYLOADS[i])
        for i, name in enumerate(("o", "x", "a"))
    }
    # a replaces o directly (depth 1) and also via x (depth 2).
    _supersede(client, claims["o"]["id"], claims["x"]["id"])
    _supersede(client, claims["o"]["id"], claims["a"]["id"], "direct correction")
    _supersede(client, claims["x"]["id"], claims["a"]["id"])

    items = _lineage(client, claims["o"]["id"], "newer").json()["items"]
    a_entries = [item for item in items if item["id"] == claims["a"]["id"]]
    assert len(a_entries) == 1
    assert a_entries[0]["depth"] == 1
    assert [item["id"] for item in items] == [
        claims["x"]["id"],
        claims["a"]["id"],
    ]


def test_parallel_supersessions_between_same_claims_do_not_duplicate(client):
    # Two independent records (different reasons) connect the same two
    # claims; the neighbor is still one item at depth 1.
    claims = _setup_chain(client, ["a", "b"])
    _supersede(client, claims["a"]["id"], claims["b"]["id"], "other rationale")
    body = _lineage(client, claims["a"]["id"], "newer").json()
    assert body["count"] == 1
    assert body["items"][0]["id"] == claims["b"]["id"]
    assert body["items"][0]["depth"] == 1


# --- Truncation -------------------------------------------------------------


def test_max_depth_truncates_beyond_the_limit(client):
    claims = _setup_chain(client, ["a", "b", "c", "d", "e"])
    resp = _lineage(client, claims["a"]["id"], "newer", max_depth=2)
    assert resp.status_code == 200
    assert [item["id"] for item in resp.json()["items"]] == [
        claims["b"]["id"],
        claims["c"]["id"],
    ]


def test_max_depth_one_returns_only_direct_replacements(client):
    claims = _setup_chain(client, ["a", "b", "c"])
    resp = _lineage(client, claims["a"]["id"], "newer", max_depth=1)
    assert [item["id"] for item in resp.json()["items"]] == [claims["b"]["id"]]


def test_default_max_depth_is_eight(client):
    names = [f"c{i:02d}" for i in range(10)]  # 10 claims, 9 edges
    claims = _setup_chain(client, names)
    resp = _lineage(client, claims[names[0]]["id"], "newer")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 8
    assert _ordered(items) == list(range(1, 9))


def test_max_depth_boundaries_are_accepted(client):
    claims = _setup_chain(client, ["a", "b"])
    for depth in (1, 32):
        resp = _lineage(client, claims["a"]["id"], "newer", max_depth=depth)
        assert resp.status_code == 200, resp.text


# --- Ordering ---------------------------------------------------------------


def test_within_level_order_uses_supersession_creation_order(client):
    create_actor(client)
    content = _create_content(client)
    # Claims are deliberately created in a different order than the edges.
    claims = {
        name: _create_claim(client, content["id"], PAYLOADS[i])
        for i, name in enumerate(("o", "b", "a", "x", "y"))
    }
    # Supersession records, in strict creation order:
    _supersede(client, claims["o"]["id"], claims["a"]["id"])  # depth 1: a first
    _supersede(
        client, claims["o"]["id"], claims["b"]["id"], "second"
    )  # depth 1: b second
    _supersede(
        client, claims["b"]["id"], claims["x"]["id"], "second"
    )  # b's depth-2 edge first
    _supersede(
        client, claims["a"]["id"], claims["y"]["id"], "third"
    )  # a's depth-2 edge after

    items = _lineage(client, claims["o"]["id"], "newer").json()["items"]
    # Depth 1 follows the edges out of o (a before b). Depth 2 is globally
    # ordered by discovering-edge creation: b -> x was created before
    # a -> y, so x precedes y even though a precedes b.
    assert [item["id"] for item in items] == [
        claims["a"]["id"],
        claims["b"]["id"],
        claims["x"]["id"],
        claims["y"]["id"],
    ]
    assert _ordered(items) == [1, 1, 2, 2]


# --- Cycles in anomalous history -------------------------------------------


def _insert_cyclic_supersession(db_session, old_id, new_id, reason="anomaly"):
    db_session.add(
        ClaimSupersession(
            id=claim_supersession_id(old_id, new_id, reason),
            superseded_claim_id=old_id,
            replacement_claim_id=new_id,
            reason=reason,
        )
    )
    db_session.commit()


def test_cycle_in_anomalous_history_terminates_deduped(client, db_session):
    claims = _setup_chain(client, ["a", "b", "c"])  # a->b, b->c
    # Bypass the service cycle guard: c->a closes a->b->c->a.
    _insert_cyclic_supersession(db_session, claims["c"]["id"], claims["a"]["id"])

    newer = _lineage(client, claims["a"]["id"], "newer")
    assert newer.status_code == 200
    assert [item["id"] for item in newer.json()["items"]] == [
        claims["b"]["id"],
        claims["c"]["id"],
    ]

    older = _lineage(client, claims["a"]["id"], "older")
    assert older.status_code == 200
    assert [item["id"] for item in older.json()["items"]] == [
        claims["c"]["id"],
        claims["b"]["id"],
    ]

    for body, origin in (
        (newer.json(), claims["a"]["id"]),
        (older.json(), claims["a"]["id"]),
    ):
        returned = [item["id"] for item in body["items"]]
        assert len(returned) == len(set(returned))
        assert origin not in returned


def test_max_depth_bounds_walk_even_with_a_direct_cycle(client, db_session):
    claims = _setup_chain(client, ["a", "b"])  # a->b
    # Anomalous reverse edge b->a forms a direct a <-> b loop.
    _insert_cyclic_supersession(db_session, claims["b"]["id"], claims["a"]["id"])
    for direction in ("newer", "older"):
        resp = _lineage(client, claims["a"]["id"], direction, max_depth=32)
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 1
        assert body["items"][0]["id"] == claims["b"]["id"]
        assert body["items"][0]["depth"] == 1


# --- Parameter validation ---------------------------------------------------


def test_direction_is_required(client):
    claims = _setup_chain(client, ["a"])
    resp = client.get(f"/v1/claims/{claims['a']['id']}/supersession-lineage")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_blank_or_illegal_direction_is_validation_error(client):
    claims = _setup_chain(client, ["a"])
    for value in ("", "   ", "up", "NEWER", "new", "old"):
        resp = _lineage(client, claims["a"]["id"], value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_max_depth_empty_non_integer_or_out_of_range_is_validation_error(client):
    claims = _setup_chain(client, ["a"])
    for value in ("", "abc", "1.5", "0", "-1", "33", "8.0", "  8"):
        resp = _lineage(client, claims["a"]["id"], "newer", max_depth=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_validation_error(client):
    claims = _setup_chain(client, ["a", "b"])
    base = f"/v1/claims/{claims['a']['id']}/supersession-lineage"
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
    claims = _setup_chain(client, ["a", "b"])
    resp = _lineage(client, claims["a"]["id"], "sideways")
    assert resp.status_code == 422
    assert "items" not in resp.json()


def test_bad_parameters_are_422_even_for_unknown_origin(client):
    # Validation precedes the origin lookup, mirroring the content lineage
    # boundary: a malformed request never renders as claim_not_found.
    resp = client.get(
        "/v1/claims/clm_ghost/supersession-lineage?direction=sideways"
    )
    assert resp.status_code == 422
    resp = client.get(
        "/v1/claims/clm_ghost/supersession-lineage?direction=newer&max_depth=0"
    )
    assert resp.status_code == 422


# --- Read-only / zero-write guarantee --------------------------------------


def test_lineage_queries_write_no_rows_or_audit_events(client, db_session):
    claims = _setup_chain(client, ["a", "b", "c"])

    def counts():
        return (
            db_session.scalar(select(func.count()).select_from(Claim)),
            db_session.scalar(
                select(func.count()).select_from(ClaimSupersession)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    before = counts()
    _lineage(client, claims["a"]["id"], "newer")
    _lineage(client, claims["c"]["id"], "older", max_depth=1)
    _lineage(client, claims["c"]["id"], "newer")  # empty result
    _lineage(client, "clm_ghost", "newer")  # 404
    _lineage(client, claims["a"]["id"], "bad")  # 422
    client.get(
        f"/v1/claims/{claims['a']['id']}/supersession-lineage"
        "?direction=newer&max_depth=1&max_depth=2"
    )  # 422
    db_session.expire_all()
    assert counts() == before
