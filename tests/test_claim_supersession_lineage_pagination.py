"""Tests for min_depth filtering and cursor pagination of the read-only
multi-hop supersession-lineage endpoint.

Covers GET /v1/claims/{claim_id}/supersession-lineage with ``min_depth``,
``limit``, and ``cursor``: filtering never prunes traversal reachability or
changes shortest-depth dedup/layer order, pages concatenate without gaps or
duplicates while ``count`` stays the filtered total, cursors are opaque and
HMAC-bound to the origin, direction, and every effective depth/limit value
(tampered, foreign, cross-endpoint, or mismatching cursors are 422), empty
results and tail boundaries are deterministic, and the extended endpoint
stays strictly read-only. All fixtures are deterministic and offline.
"""

from __future__ import annotations

import base64
import hashlib
import hmac as hmac_mod
import json
import secrets

from sqlalchemy import func, select

from provenance import pagination
from provenance.models import AuditEvent, Claim, ClaimSupersession
from tests.helpers import DIGEST_A, content_payload, create_actor

PAYLOADS = [{"statement": f"version {i}"} for i in range(70)]


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


def _ids(items):
    return [item["id"] for item in items]


def _walk_pages(client, claim_id, direction="newer", **params):
    """Follow next_cursor until exhausted; return (all_items, pages, count)."""
    pages = []
    all_items = []
    count = None
    seen_cursors = []
    cursor = None
    for _ in range(100):
        query = {**params}
        if cursor is not None:
            query["cursor"] = cursor
        resp = _lineage(client, claim_id, direction, **query)
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


# --- response shape ----------------------------------------------------------


def test_unfiltered_response_shape_now_carries_null_cursor(client):
    claims = _setup_chain(client, ["a", "b"])
    body = _lineage(client, claims["a"]["id"], "newer").json()
    assert set(body) == {"items", "count", "next_cursor"}
    assert body["count"] == 1
    assert body["next_cursor"] is None


# --- min_depth filtering ------------------------------------------------------


def test_min_depth_truncates_the_shallow_side_only(client):
    names = [f"c{i:02d}" for i in range(6)]
    claims = _setup_chain(client, names)
    body = _lineage(
        client, claims[names[0]]["id"], "newer", max_depth=4, min_depth=2
    ).json()
    assert _ids(body["items"]) == [
        claims[names[2]]["id"],
        claims[names[3]]["id"],
        claims[names[4]]["id"],
    ]
    assert _ordered(body["items"]) == [2, 3, 4]
    assert body["count"] == 3
    assert body["next_cursor"] is None


def test_min_depth_equal_to_max_depth_returns_single_level(client):
    claims = _setup_chain(client, ["a", "b", "c"])
    body = _lineage(
        client, claims["a"]["id"], "newer", max_depth=2, min_depth=2
    ).json()
    assert _ids(body["items"]) == [claims["c"]["id"]]
    assert body["count"] == 1


def test_min_depth_defaults_to_one_and_matches_unfiltered(client):
    claims = _setup_chain(client, ["a", "b", "c"])
    defaulted = _lineage(client, claims["a"]["id"], "newer").json()
    explicit = _lineage(
        client, claims["a"]["id"], "newer", min_depth=1
    ).json()
    assert defaulted == explicit


def test_min_depth_does_not_change_reachability_or_shortest_depths(client):
    # a -> b, a -> c, b -> d, c -> d (diamond). Even when only depth 2 is
    # returned, the walk still reaches d through both paths and dedups it at
    # the shortest depth exactly as without the filter.
    create_actor(client)
    content = _create_content(client)
    claims = {
        name: _create_claim(client, content["id"], PAYLOADS[i])
        for i, name in enumerate(("a", "b", "c", "d"))
    }
    _supersede(client, claims["a"]["id"], claims["b"]["id"])
    _supersede(client, claims["a"]["id"], claims["c"]["id"], "second")
    _supersede(client, claims["b"]["id"], claims["d"]["id"])
    _supersede(client, claims["c"]["id"], claims["d"]["id"], "second")

    body = _lineage(
        client, claims["a"]["id"], "newer", min_depth=2
    ).json()
    assert _ids(body["items"]) == [claims["d"]["id"]]
    assert _ordered(body["items"]) == [2]
    assert body["count"] == 1

    full = _lineage(client, claims["a"]["id"], "newer").json()
    assert [item["depth"] for item in full["items"] if item["id"] == claims["d"]["id"]] == [2]


def test_min_depth_filtering_works_in_older_direction(client):
    names = ["a", "b", "c", "d"]
    claims = _setup_chain(client, names)
    body = _lineage(
        client, claims["d"]["id"], "older", min_depth=2
    ).json()
    assert _ids(body["items"]) == [claims["b"]["id"], claims["a"]["id"]]
    assert _ordered(body["items"]) == [2, 3]


def test_min_depth_beyond_reachable_set_returns_empty(client):
    claims = _setup_chain(client, ["a", "b"])
    body = _lineage(
        client, claims["a"]["id"], "newer", min_depth=2
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


# --- pagination continuity ----------------------------------------------------


def test_pagination_concatenates_without_gaps_or_duplicates(client):
    names = [f"c{i:02d}" for i in range(12)]
    claims = _setup_chain(client, names)
    all_items, pages, count, _ = _walk_pages(
        client, claims[names[0]]["id"], max_depth=32, limit=3
    )
    assert [len(page) for page in pages] == [3, 3, 3, 2]
    assert count == 11
    returned = _ids(all_items)
    assert len(returned) == len(set(returned)) == 11
    # Concatenated pages reproduce the canonical depth/discovery order.
    assert _ids(all_items) == _ids(
        _lineage(client, claims[names[0]]["id"], "newer", max_depth=32)
        .json()["items"]
    )


def test_count_is_the_filtered_total_on_every_page(client):
    names = [f"c{i:02d}" for i in range(8)]
    claims = _setup_chain(client, names)
    all_items, pages, count, _ = _walk_pages(
        client,
        claims[names[0]]["id"],
        max_depth=32,
        min_depth=3,
        limit=2,
    )
    assert count == 5  # c03..c07
    assert [len(page) for page in pages] == [2, 2, 1]
    assert _ordered(all_items) == [3, 4, 5, 6, 7]


def test_last_page_cursor_is_null_when_pages_divide_exactly(client):
    claims = _setup_chain(client, ["a", "b", "c", "d", "e"])
    body = _lineage(client, claims["a"]["id"], "newer", limit=2).json()
    assert body["count"] == 4
    assert body["next_cursor"] is not None
    second = _lineage(
        client, claims["a"]["id"], "newer", limit=2, cursor=body["next_cursor"]
    ).json()
    assert len(second["items"]) == 2
    assert second["count"] == 4
    assert second["next_cursor"] is None


def test_partial_final_page_cursor_is_null(client):
    claims = _setup_chain(client, ["a", "b", "c", "d", "e"])
    body = _lineage(client, claims["a"]["id"], "newer", limit=3).json()
    assert _ids(body["items"]) == [
        claims["b"]["id"],
        claims["c"]["id"],
        claims["d"]["id"],
    ]
    second = _lineage(
        client, claims["a"]["id"], "newer", limit=3, cursor=body["next_cursor"]
    ).json()
    assert _ids(second["items"]) == [claims["e"]["id"]]
    assert second["next_cursor"] is None


def test_default_limit_is_fifty(client):
    # One origin and 60 direct replacements, all at depth 1: two pages.
    create_actor(client)
    content = _create_content(client)
    origin = _create_claim(client, content["id"], PAYLOADS[60])
    replacements = [_create_claim(client, content["id"], PAYLOADS[i]) for i in range(60)]
    for replacement in replacements:
        _supersede(client, origin["id"], replacement["id"])

    first = _lineage(client, origin["id"], "newer").json()
    assert len(first["items"]) == 50
    assert first["count"] == 60
    assert first["next_cursor"] is not None
    second = _lineage(
        client, origin["id"], "newer", cursor=first["next_cursor"]
    ).json()
    assert len(second["items"]) == 10
    assert second["count"] == 60
    assert second["next_cursor"] is None
    assert _ids(first["items"]) == [r["id"] for r in replacements[:50]]
    assert _ids(second["items"]) == [r["id"] for r in replacements[50:]]


def test_limit_one_pages_every_item(client):
    claims = _setup_chain(client, ["a", "b", "c"])
    all_items, pages, count, cursors = _walk_pages(
        client, claims["a"]["id"], limit=1
    )
    assert [len(page) for page in pages] == [1, 1]
    assert count == 2
    assert _ids(all_items) == [claims["b"]["id"], claims["c"]["id"]]
    assert len(cursors) == 1


def test_pagination_is_stable_and_deterministic_across_walks(client):
    names = [f"c{i:02d}" for i in range(8)]
    claims = _setup_chain(client, names)
    first_items, _, _, first_cursors = _walk_pages(
        client, claims[names[0]]["id"], max_depth=32, limit=3
    )
    second_items, _, _, second_cursors = _walk_pages(
        client, claims[names[0]]["id"], max_depth=32, limit=3
    )
    # Stateless HMAC cursors: identical requests mint identical tokens.
    assert first_cursors == second_cursors
    assert _ids(first_items) == _ids(second_items)


def test_reusing_a_cursor_replays_the_same_page(client):
    names = [f"c{i:02d}" for i in range(6)]
    claims = _setup_chain(client, names)
    first = _lineage(
        client, claims[names[0]]["id"], "newer", max_depth=32, limit=2
    ).json()
    cursor = first["next_cursor"]
    params = dict(max_depth=32, limit=2, cursor=cursor)
    replay_one = _lineage(
        client, claims[names[0]]["id"], "newer", **params
    ).json()
    replay_two = _lineage(
        client, claims[names[0]]["id"], "newer", **params
    ).json()
    assert replay_one == replay_two
    assert _ids(replay_one["items"]) == [claims[names[3]]["id"], claims[names[4]]["id"]]


def test_cursor_past_end_returns_empty_page_with_total_count(client, app):
    claims = _setup_chain(client, ["a", "b", "c"])
    token = pagination.encode_typed_cursor(
        app.state.supersession_lineage_cursor_secret,
        pagination.SUPERSESSION_LINEAGE_CURSOR,
        {
            "claim_id": claims["a"]["id"],
            "direction": "newer",
            "max_depth": 8,
            "min_depth": 1,
            "limit": 50,
            "offset": 99,
        },
    )
    body = _lineage(client, claims["a"]["id"], "newer", cursor=token).json()
    assert body["items"] == []
    assert body["count"] == 2
    assert body["next_cursor"] is None


# --- cursor integrity ---------------------------------------------------------


def test_tampered_or_malformed_cursors_are_validation_errors(client, app):
    claims = _setup_chain(client, ["a", "b", "c"])
    good = _lineage(client, claims["a"]["id"], "newer", limit=1).json()[
        "next_cursor"
    ]
    tampered_payload = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    foreign = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.SUPERSESSION_LINEAGE_CURSOR,
        {
            "claim_id": claims["a"]["id"],
            "direction": "newer",
            "max_depth": 8,
            "min_depth": 1,
            "limit": 1,
            "offset": 1,
        },
    )
    for token in (
        "",
        "   ",
        "not-a-cursor",
        "sl1.onlytwoparts",
        "sl1.too.many.parts",
        "sl0.x.y",
        "sl2.x.y",
        tampered_payload,
        foreign,
    ):
        resp = _lineage(client, claims["a"]["id"], "newer", limit=1, cursor=token)
        assert resp.status_code == 422, repr(token)
        assert resp.json()["error"]["code"] == "validation_error"
        assert "items" not in resp.json()


def test_cursor_signed_with_old_format_marker_is_rejected(client, app):
    claims = _setup_chain(client, ["a", "b"])
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "claim_id": claims["a"]["id"],
                "direction": "newer",
                "max_depth": 8,
                "min_depth": 1,
                "limit": 50,
                "offset": 1,
            }
        ).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(
        hmac_mod.new(
            app.state.supersession_lineage_cursor_secret,
            f"sl0.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    resp = _lineage(
        client, claims["a"]["id"], "newer", cursor=f"sl0.{payload}.{sig}"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_from_another_endpoint_family_is_rejected(client, app):
    # A content-lineage ("v1") cursor, even hand-signed with the
    # supersession-lineage secret, must never resume this endpoint.
    claims = _setup_chain(client, ["a", "b"])
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "content_id": "cnt_other",
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
            app.state.supersession_lineage_cursor_secret,
            f"v1.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    resp = _lineage(
        client, claims["a"]["id"], "newer", cursor=f"v1.{payload}.{sig}"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_bound_to_every_effective_query_parameter(client):
    names = [f"c{i:02d}" for i in range(12)]
    claims = _setup_chain(client, names)
    origin = claims[names[0]]["id"]
    cursor = _lineage(client, origin, "newer", max_depth=32, limit=2).json()[
        "next_cursor"
    ]

    # Every mismatch (including a different origin) is rejected.
    mismatches = [
        ("older", dict(max_depth=32, limit=2)),
        ("newer", dict(max_depth=32, limit=3)),
        ("newer", dict(max_depth=4, limit=2)),
        ("newer", dict(max_depth=32, min_depth=2, limit=2)),
    ]
    for direction, params in mismatches:
        resp = _lineage(
            client, origin, direction, cursor=cursor, **params
        )
        assert resp.status_code == 422, (direction, params)
        assert resp.json()["error"]["code"] == "validation_error"

    resp = _lineage(
        client, claims[names[1]]["id"], "newer", max_depth=32, limit=2,
        cursor=cursor,
    )
    assert resp.status_code == 422


def test_cursor_accepts_explicit_params_equal_to_cursor_defaults(client):
    names = [f"c{i:02d}" for i in range(6)]
    claims = _setup_chain(client, names)
    # First page uses all defaults (max_depth=8, min_depth=1).
    cursor = _lineage(
        client, claims[names[0]]["id"], "newer", limit=2
    ).json()["next_cursor"]
    # Repeating the same *effective* parameters explicitly must resume.
    resp = _lineage(
        client,
        claims[names[0]]["id"],
        "newer",
        max_depth=8,
        min_depth=1,
        limit=2,
        cursor=cursor,
    )
    assert resp.status_code == 200, resp.text
    assert _ids(resp.json()["items"]) == [claims[names[3]]["id"], claims[names[4]]["id"]]


# --- parameter validation -----------------------------------------------------


def test_min_depth_boundaries_and_incompatibility(client):
    claims = _setup_chain(client, ["a", "b", "c"])
    for value in ("0", "33", "-1", "1.5", "abc", "8.0", "  2", ""):
        resp = _lineage(client, claims["a"]["id"], "newer", min_depth=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"
    resp = _lineage(
        client, claims["a"]["id"], "newer", max_depth=2, min_depth=3
    )
    assert resp.status_code == 422
    # Equality is allowed.
    assert (
        _lineage(
            client, claims["a"]["id"], "newer", max_depth=2, min_depth=2
        ).status_code
        == 200
    )


def test_limit_boundaries(client):
    claims = _setup_chain(client, ["a"])
    for value in ("0", "101", "-1", "1.0", "1.5", "abc", "  2", ""):
        resp = _lineage(client, claims["a"]["id"], "newer", limit=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"
    for value in (1, 100):
        assert (
            _lineage(client, claims["a"]["id"], "newer", limit=value).status_code
            == 200
        )


def test_repeated_new_parameters_are_validation_errors(client):
    claims = _setup_chain(client, ["a", "b"])
    base = f"/v1/claims/{claims['a']['id']}/supersession-lineage"
    for suffix in (
        "direction=newer&min_depth=1&min_depth=1",
        "direction=newer&limit=1&limit=2",
        "direction=newer&cursor=x&cursor=y",
    ):
        resp = client.get(f"{base}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameter_is_validation_error(client):
    claims = _setup_chain(client, ["a"])
    resp = _lineage(
        client, claims["a"]["id"], "newer", relation_type="version_of"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    resp = client.get(
        f"/v1/claims/{claims['a']['id']}/supersession-lineage"
        "?direction=newer&unexpected=1"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_cursor_with_otherwise_valid_params_is_422(client):
    claims = _setup_chain(client, ["a", "b"])
    resp = _lineage(
        client,
        claims["a"]["id"],
        "newer",
        min_depth=1,
        max_depth=8,
        limit=10,
        cursor="garbage",
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_new_param_failures_are_422_even_for_unknown_origin(client):
    create_actor(client)
    for suffix in (
        "direction=newer&min_depth=9&max_depth=2",
        "direction=newer&limit=0",
        "direction=newer&min_depth=33",
        "direction=newer&unexpected=1",
        "direction=newer&cursor=garbage",
    ):
        resp = client.get(
            "/v1/claims/clm_ghost/supersession-lineage?" + suffix
        )
        assert resp.status_code == 422, suffix


def test_unknown_origin_still_404_with_new_params(client):
    create_actor(client)
    resp = _lineage(
        client, "clm_ghost", "newer", min_depth=1, max_depth=8, limit=10
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "claim_not_found"


# --- read-only guarantee ------------------------------------------------------


def test_filtered_paginated_queries_write_nothing(client, db_session):
    names = [f"c{i:02d}" for i in range(6)]
    claims = _setup_chain(client, names)

    def counts():
        return (
            db_session.scalar(select(func.count()).select_from(Claim)),
            db_session.scalar(
                select(func.count()).select_from(ClaimSupersession)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    before = counts()
    cursor = None
    for _ in range(5):
        params = {"max_depth": 8, "min_depth": 1, "limit": 2}
        if cursor is not None:
            params["cursor"] = cursor
        resp = _lineage(client, claims[names[0]]["id"], "newer", **params)
        assert resp.status_code == 200, resp.text
        cursor = resp.json()["next_cursor"]
        if cursor is None:
            break
    # Invalid extensions must not write anything either.
    _lineage(client, claims[names[0]]["id"], "newer", min_depth=9, max_depth=2)
    _lineage(client, claims[names[0]]["id"], "newer", limit=0)
    _lineage(client, claims[names[0]]["id"], "newer", cursor="tampered")
    client.get(
        f"/v1/claims/{claims[names[0]]['id']}/supersession-lineage"
        "?direction=newer&limit=1&limit=2"
    )
    db_session.expire_all()
    assert counts() == before
