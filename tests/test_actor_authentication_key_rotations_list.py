"""Offline deterministic tests for the reviewer read-only rotation listing.

Covers ``GET /v1/actors/{actor_id}/authentication-key-rotations``:

* the success body carries exactly ``{"items", "count", "next_cursor"}`` and
  each item exposes only the existing rotation public fields (``id``,
  ``actor_id``, ``new_public_key``, ``active``, ``created_at``,
  ``retired_at``) -- never the internal ordering surrogate, a private key, a
  raw signature, or an authentication header;
* records follow stable creation order, an existing subject without
  rotations is an empty collection, and an unknown subject is the existing
  ``unknown_actor`` 404;
* ``limit`` is strictly a decimal integer in 1..100 (default 50); pages
  concatenate without gaps or duplicates, the final cursor is null, a
  cursor at/past the tail returns an empty page with the original count, and
  a cursor replays identically;
* blank, malformed, tampered, wrongly-signed, wrong-family, wrong-claim-set,
  cross-endpoint, and subject/limit-mismatching cursors are all
  ``422 validation_error`` -- never silently normalized -- and every query
  validation failure (including repeated or undeclared parameters) is
  decided before the subject lookup, so it is a 422 even for an unknown
  actor;
* the route is strictly read-only: no rotation, resource, or audit row is
  written by a success or a failure, and the existing rotation/retire
  routes and their identifiers and UTC fields remain compatible.

All fixtures are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures); only fixed seed-derived public keys are used.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime

from sqlalchemy import func, select

from provenance import pagination
from provenance.models import AuditEvent, AuthenticationKeyRotation
from tests.helpers import SEED_B
from tests.test_authentication_key_rotations import (
    SEED_A,
    SEED_R1,
    _key_b64,
    _post_retire,
    _post_rotation,
    _rotation_body,
    _world,
)

URL = "/v1/actors/org-1/authentication-key-rotations"

ITEM_KEYS = {
    "id",
    "actor_id",
    "new_public_key",
    "active",
    "created_at",
    "retired_at",
}
PAGE_KEYS = {"items", "count", "next_cursor"}


# --- World setup ----------------------------------------------------------------


def _seed_for(i: int) -> bytes:
    # Fixed, distinct, 32-byte seeds; each derives a distinct public key.
    return (b"test-ed25519-rotate-list-%06d" % i).ljust(32, b"x")[:32]


def _create_rotations(client, count: int, *, actor: str = "org-1"):
    """Create ``count`` distinct rotations for one subject.

    Every request is signed by the subject's original non-revoked
    attestation key, which stays in the authentication set across
    rotations, so no rotation key needs to be chained.
    """
    seed = SEED_A if actor == "org-1" else SEED_B
    created = []
    for i in range(count):
        body = {
            "actor_id": actor,
            "new_public_key": _key_b64(_seed_for(i)),
        }
        resp = _post_rotation(client, body, actor=actor, seed=seed)
        assert resp.status_code == 201, resp.text
        created.append(resp.json())
    return created


def _list(client, actor_id="org-1", **params):
    return client.get(f"/v1/actors/{actor_id}/authentication-key-rotations",
                      params=params)


def _walk_pages(client, actor_id="org-1", **params):
    """Follow next_cursor until exhausted; return (all_items, pages, count)."""
    pages = []
    all_items = []
    count = None
    cursor = None
    for _ in range(100):
        query = {**params}
        if cursor is not None:
            query["cursor"] = cursor
        resp = _list(client, actor_id, **query)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        count = body["count"]
        pages.append(body["items"])
        all_items.extend(body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            break
    return all_items, pages, count


def _hand_cursor(secret: bytes, version: str, claims) -> str:
    """Sign a hand-built token so claim-set/format failures are exercised.

    ``claims`` is serialized compactly as JSON unless it is already a
    string, in which case it is Base64-encoded verbatim (used to mint a
    correctly-signed token whose payload is not JSON).
    """
    raw = (
        claims
        if isinstance(claims, str)
        else json.dumps(claims, separators=(",", ":"))
    )
    payload = pagination._b64encode(raw.encode("utf-8"))
    signature = pagination._b64encode(pagination._sign(secret, version, payload))
    return f"{version}.{payload}.{signature}"


# --- Response shape, fields, and ordering --------------------------------------


def test_success_body_has_exactly_the_three_page_members(client):
    _world(client)
    _post_rotation(client, _rotation_body())
    body = _list(client).json()
    assert set(body) == PAGE_KEYS


def test_item_has_exactly_the_existing_public_rotation_fields(client):
    _world(client)
    created = _post_rotation(client, _rotation_body()).json()
    body = _list(client).json()
    assert body["count"] == 1
    item = body["items"][0]
    assert set(item) == ITEM_KEYS
    assert item == created
    # Internal/secret material never appears, even under different spellings.
    serialized = json.dumps(body)
    for leak in ("seq", "private", "signature", "X-PA", "X-PT", "X-PS"):
        assert leak not in serialized


def test_created_at_and_retired_at_are_utc(client):
    _world(client)
    created = _post_rotation(client, _rotation_body()).json()
    assert _post_retire(client, created["id"], seed=SEED_R1).status_code == 200
    item = _list(client).json()["items"][0]
    assert item["active"] is False
    assert item["retired_at"] is not None
    assert datetime.fromisoformat(item["created_at"]).utcoffset().total_seconds() == 0
    assert datetime.fromisoformat(item["retired_at"]).utcoffset().total_seconds() == 0


def test_rotations_follow_stable_creation_order(client):
    _world(client)
    created = _create_rotations(client, 5)
    body = _list(client).json()
    assert [i["id"] for i in body["items"]] == [r["id"] for r in created]
    # created_at is non-decreasing, and every id is present exactly once.
    ids = [i["id"] for i in body["items"]]
    assert len(ids) == len(set(ids)) == 5


def test_listing_matches_the_single_rotation_views(client):
    _world(client)
    created = _create_rotations(client, 3)
    items = _list(client).json()["items"]
    assert items == created


def test_listing_is_scoped_to_the_path_subject(client):
    _world(client)
    _create_rotations(client, 3, actor="org-1")
    _create_rotations(client, 2, actor="org-2")
    org1 = _list(client, "org-1").json()
    org2 = _list(client, "org-2").json()
    assert org1["count"] == 3
    assert org2["count"] == 2
    assert {i["actor_id"] for i in org1["items"]} == {"org-1"}
    assert {i["actor_id"] for i in org2["items"]} == {"org-2"}
    assert not {i["id"] for i in org1["items"]} & {
        i["id"] for i in org2["items"]
    }


def test_existing_subject_without_rotations_is_an_empty_collection(client):
    _world(client)
    body = _list(client, "org-2").json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


def test_retired_and_active_records_are_listed_together(client):
    _world(client)
    created = _create_rotations(client, 3)
    assert _post_retire(client, created[1]["id"], seed=_seed_for(1)).status_code == 200
    items = _list(client).json()["items"]
    assert [i["active"] for i in items] == [True, False, True]
    assert items[1]["retired_at"] is not None
    assert items[0]["retired_at"] is None
    assert items[2]["retired_at"] is None


def test_reading_requires_no_authentication_headers(client):
    _world(client)
    _post_rotation(client, _rotation_body())
    # The reviewer route is public, exactly like the other read-only lists.
    resp = client.get(URL)
    assert resp.status_code == 200
    assert "X-PA" not in resp.request.headers


# --- Unknown subject -------------------------------------------------------------


def test_unknown_subject_is_404_unknown_actor(client):
    _world(client)
    resp = _list(client, "ghost")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "unknown_actor"
    assert error["details"]["actor_id"] == "ghost"


def test_unknown_subject_with_limit_is_still_404(client):
    _world(client)
    assert _list(client, "ghost", limit=10).status_code == 404


# --- Pagination ------------------------------------------------------------------


def test_pagination_concatenates_without_gaps_or_duplicates(client):
    _world(client)
    created = _create_rotations(client, 5)
    all_items, pages, count = _walk_pages(client, limit=2)
    # 5 rotations -> pages of 2, 2, 1.
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    ids = [i["id"] for i in all_items]
    assert len(ids) == len(set(ids)) == 5
    assert ids == [r["id"] for r in created]
    assert all_items == _list(client).json()["items"]


def test_count_is_total_on_every_page(client):
    _world(client)
    _create_rotations(client, 5)
    _, pages, _ = _walk_pages(client, limit=2)
    all_items, _, count = _walk_pages(client, limit=3)
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    assert len(all_items) == 5


def test_last_page_cursor_is_null_on_exact_division(client):
    _world(client)
    _create_rotations(client, 4)
    first = _list(client, limit=2).json()
    assert first["next_cursor"] is not None
    second = _list(client, limit=2, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 2
    assert second["count"] == 4
    assert second["next_cursor"] is None


def test_default_limit_is_fifty(client):
    _world(client)
    _create_rotations(client, 55)
    first = _list(client).json()
    assert len(first["items"]) == 50
    assert first["count"] == 55
    assert first["next_cursor"] is not None
    second = _list(client, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 5
    assert second["count"] == 55
    assert second["next_cursor"] is None


def test_limit_boundaries_accepted(client):
    _world(client)
    _create_rotations(client, 1)
    for value in (1, 100):
        assert _list(client, limit=value).status_code == 200


def test_reusing_a_cursor_replays_the_same_page(client):
    _world(client)
    _create_rotations(client, 4)
    cursor = _list(client, limit=2).json()["next_cursor"]
    replay_one = _list(client, limit=2, cursor=cursor).json()
    replay_two = _list(client, limit=2, cursor=cursor).json()
    assert replay_one == replay_two


def test_issued_cursors_use_the_dedicated_opaque_family_marker(client):
    _world(client)
    _create_rotations(client, 2)
    token = _list(client, limit=1).json()["next_cursor"]
    assert token.startswith(pagination.AUTHENTICATION_KEY_ROTATIONS_CURSOR_VERSION + ".")
    assert len(token.split(".")) == 3


def test_cursor_past_end_returns_empty_page_with_original_count(client, app):
    _world(client)
    _create_rotations(client, 5)
    token = pagination.encode_typed_cursor(
        app.state.authentication_key_rotations_cursor_secret,
        pagination.AUTHENTICATION_KEY_ROTATIONS_CURSOR,
        {"actor_id": "org-1", "limit": 50, "offset": 99},
    )
    body = _list(client, cursor=token).json()
    assert body["items"] == []
    assert body["count"] == 5
    assert body["next_cursor"] is None


def test_tail_cursor_returns_empty_items_and_same_count(client):
    _world(client)
    _create_rotations(client, 4)
    # A cursor pointing exactly at the tail (offset == total) is validly
    # signed but yields an empty page; the response still reports count 4.
    first = _list(client, limit=2).json()
    tail = _list(client, limit=2, cursor=first["next_cursor"]).json()
    assert tail["next_cursor"] is None
    # Re-requesting with a freshly minted offset==total token behaves the
    # same as the natural end of the walk.
    app = client.app
    token = pagination.encode_typed_cursor(
        app.state.authentication_key_rotations_cursor_secret,
        pagination.AUTHENTICATION_KEY_ROTATIONS_CURSOR,
        {"actor_id": "org-1", "limit": 2, "offset": 4},
    )
    body = _list(client, limit=2, cursor=token).json()
    assert body["items"] == []
    assert body["count"] == 4
    assert body["next_cursor"] is None


# --- Cursor integrity ------------------------------------------------------------


def test_blank_malformed_or_tampered_cursors_are_validation_errors(client, app):
    _world(client)
    _create_rotations(client, 2)
    good = _list(client, limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    foreign = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.AUTHENTICATION_KEY_ROTATIONS_CURSOR,
        {"actor_id": "org-1", "limit": 1, "offset": 1},
    )
    secret = app.state.authentication_key_rotations_cursor_secret
    for token in (
        "",
        "   ",
        "\t",
        "not-a-cursor",
        "ak1.onlytwoparts",
        "ak1.too.many.parts",
        "ak0.x.y",
        "ak2.x.y",
        # Every other family marker, even signed with this family's secret,
        # is a version mismatch rather than a trusted token.
        _hand_cursor(secret, "v1", {"x": 1}),
        _hand_cursor(secret, "ce1", {"x": 1}),
        _hand_cursor(secret, "ae1", {"x": 1}),
        _hand_cursor(secret, "ei1", {"x": 1}),
        _hand_cursor(secret, "ir1", {"x": 1}),
        _hand_cursor(secret, "ci1", {"x": 1}),
        _hand_cursor(secret, "cr1", {"x": 1}),
        _hand_cursor(secret, "cl1", {"x": 1}),
        _hand_cursor(secret, "eb1", {"x": 1}),
        # Correctly signed, but structurally invalid claim payloads.
        _hand_cursor(secret, "ak1", "not-json"),
        _hand_cursor(
            secret, "ak1",
            {"actor_id": "org-1", "limit": 1, "offset": 1, "extra": 2},
        ),
        _hand_cursor(secret, "ak1", {"actor_id": "org-1", "limit": 1}),
        _hand_cursor(secret, "ak1", {"actor_id": "", "limit": 1, "offset": 1}),
        _hand_cursor(secret, "ak1", {"actor_id": "org-1", "limit": 101, "offset": 1}),
        _hand_cursor(secret, "ak1", {"actor_id": "org-1", "limit": 0, "offset": 1}),
        _hand_cursor(secret, "ak1", {"actor_id": "org-1", "limit": 1, "offset": 0}),
        _hand_cursor(secret, "ak1", {"actor_id": 7, "limit": 1, "offset": 1}),
        _hand_cursor(
            secret, "ak1", {"actor_id": "org-1", "limit": "1", "offset": 1}
        ),
        tampered,
        foreign,
    ):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422, repr(token)
        assert resp.json()["error"]["code"] == "validation_error", repr(token)
        assert "items" not in resp.json()


def test_cursor_from_other_endpoints_is_rejected(client, app):
    _world(client)
    _create_rotations(client, 2)
    secret = app.state.authentication_key_rotations_cursor_secret
    # Tokens minted by other families (under this process's own secret) can
    # never resume an actor-rotation page.
    others = [
        pagination.encode_typed_cursor(
            secret,
            pagination.CLAIMS_CURSOR,
            {
                "content_id": None,
                "actor_id": None,
                "claim_type": None,
                "payload_digest_hex": None,
                "limit": 1,
                "offset": 1,
            },
        ),
        pagination.encode_typed_cursor(
            secret,
            pagination.AUDIT_EVENTS_CURSOR,
            {
                "event_type": None,
                "resource_id": None,
                "from": None,
                "to": None,
                "limit": 1,
                "offset": 1,
            },
        ),
        pagination.encode_typed_cursor(
            secret,
            pagination.CONTENT_EVIDENCE_CURSOR,
            {
                "content_id": "cnt_x",
                "evidence_type": None,
                "media_type": None,
                "limit": 1,
                "offset": 1,
            },
        ),
    ]
    for token in others:
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


def test_actor_rotation_cursor_is_rejected_by_other_endpoints(client):
    _world(client)
    _create_rotations(client, 2)
    cursor = _list(client, limit=1).json()["next_cursor"]
    for path in (
        "/v1/claims",
        "/v1/evidence-bundles",
        "/v1/audit-events",
        "/v1/evidence-bundle-exchange-imports",
    ):
        resp = client.get(path, params={"limit": 1, "cursor": cursor})
        assert resp.status_code == 422, path
        assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_is_bound_to_its_subject(client, app):
    _world(client)
    _create_rotations(client, 3, actor="org-1")
    _create_rotations(client, 3, actor="org-2")
    cursor = _list(client, "org-1", limit=2).json()["next_cursor"]

    # Same cursor on another existing subject is a mismatch, not a page.
    resp = _list(client, "org-2", limit=2, cursor=cursor)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    # A correctly-signed cursor naming org-1 cannot be presented on the
    # org-2 path.
    secret = app.state.authentication_key_rotations_cursor_secret
    org1_token = pagination.encode_typed_cursor(
        secret,
        pagination.AUTHENTICATION_KEY_ROTATIONS_CURSOR,
        {"actor_id": "org-1", "limit": 2, "offset": 2},
    )
    assert _list(client, "org-2", limit=2, cursor=org1_token).status_code == 422


def test_cursor_is_bound_to_its_effective_limit(client):
    _world(client)
    _create_rotations(client, 4)
    cursor = _list(client, limit=2).json()["next_cursor"]
    for params in ({"limit": 3}, {"limit": 1}, {}):
        # Presenting the limit-2 cursor with another limit (or with the
        # default 50) is a mismatch rather than a re-paging.
        resp = _list(client, cursor=cursor, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_mismatch_is_rejected_before_actor_lookup(client):
    _world(client)
    _create_rotations(client, 2)
    cursor = _list(client, "org-1", limit=1).json()["next_cursor"]
    # The subject in the path does not exist; the bound cursor still
    # mismatches and validation wins over the unknown_actor lookup.
    resp = _list(client, "ghost", limit=1, cursor=cursor)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_secret_rotation_invalidates_outstanding_cursors(client):
    _world(client)
    _create_rotations(client, 2)
    cursor = _list(client, limit=1).json()["next_cursor"]
    client.app.state.authentication_key_rotations_cursor_secret = (
        secrets.token_bytes(32)
    )
    resp = _list(client, limit=1, cursor=cursor)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    # A fresh cursor under the new secret pages normally.
    fresh = _list(client, limit=1).json()
    assert len(fresh["items"]) == 1


# --- Parameter validation --------------------------------------------------------


def test_illegal_limit_values_are_validation_errors(client):
    _world(client)
    for value in (
        "0", "101", "-1", "1.5", "8.0", "abc", "", "  2", "2  ",
        "+1", "1e2", "０１", "0x1",
    ):
        resp = _list(client, limit=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_validation_errors(client):
    _world(client)
    for suffix in ("limit=1&limit=2", "cursor=x&cursor=y"):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameters_are_validation_errors(client):
    _world(client)
    for suffix in (
        "active=true",
        "actor_id=org-2",
        "offset=1",
        "page=1",
        "Limit=1",
        "limit=1&include_retired=1",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_validation_failures_are_decided_before_actor_lookup(client):
    # Even with an unknown subject, every malformed query is a 422 rather
    # than the unknown_actor 404: validation precedes the lookup.
    _world(client)
    for suffix in (
        "limit=0",
        "limit=101",
        "limit=abc",
        "limit=1&limit=2",
        "cursor=garbage",
        "cursor=x&cursor=y",
        "bogus=1",
        "active=false",
    ):
        resp = client.get(
            "/v1/actors/ghost/authentication-key-rotations?" + suffix
        )
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"
    # A structurally valid request for the same unknown subject is the 404.
    plain = client.get("/v1/actors/ghost/authentication-key-rotations")
    assert plain.status_code == 404


def test_validation_failure_locates_the_query_field(client):
    _world(client)
    resp = _list(client, limit="nope")
    issue = resp.json()["error"]["details"]["issues"][0]
    assert issue["loc"] == ["query", "limit"]
    resp = _list(client, cursor="garbage")
    issue = resp.json()["error"]["details"]["issues"][0]
    assert issue["loc"] == ["query", "cursor"]


# --- Read-only guarantee ---------------------------------------------------------


def test_reads_and_failures_write_nothing(client, db_session):
    _world(client)
    created = _create_rotations(client, 5)

    def counts():
        return (
            db_session.scalar(
                select(func.count()).select_from(AuthenticationKeyRotation)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    rotations_before, events_before = counts()
    assert rotations_before == 5

    # Successful reads: unfiltered, paginated through the tail, and the
    # empty-collection case for another subject.
    _walk_pages(client, limit=2)
    _list(client)
    _list(client, "org-2")

    # Failed reads: malformed limit/cursor, repeats, unknown params, unknown
    # subject, and a bound-cursor mismatch.
    _list(client, limit=0)
    _list(client, limit=101)
    _list(client, cursor="tampered")
    _list(client, "ghost")
    client.get(f"{URL}?limit=1&limit=2")
    client.get(f"{URL}?unknown=1")
    cursor = _list(client, "org-1", limit=1).json()["next_cursor"]
    _list(client, "org-2", limit=1, cursor=cursor)

    db_session.expire_all()
    assert counts() == (rotations_before, events_before)
    # The rotations themselves are untouched.
    rows = db_session.execute(
        select(AuthenticationKeyRotation).order_by(AuthenticationKeyRotation.seq)
    ).scalars().all()
    assert [r.id for r in rows] == [r["id"] for r in created]
    assert all(r.active for r in rows)
    assert all(r.retired_at is None for r in rows)


# --- Compatibility with the existing rotation routes -----------------------------


def test_existing_rotation_routes_remain_unchanged(client):
    _world(client)
    created = _post_rotation(client, _rotation_body())
    assert created.status_code == 201
    # A repeat submission is still the idempotent 200 with no new record.
    repeat = _post_rotation(client, _rotation_body())
    assert repeat.status_code == 200
    assert repeat.json() == created.json()
    listing = _list(client).json()
    assert listing["count"] == 1
    assert listing["items"] == [created.json()]
