"""Offline deterministic tests for the reviewer key-revocation listing.

Covers ``GET /v1/actors/{actor_id}/authentication-key-revocations``:

* the success body carries exactly ``{"items", "count", "next_cursor"}``
  and each item exposes only the revocation public fields (``id``,
  ``actor_id``, ``public_key``, ``reason``, ``revoked_at``) -- never the
  internal ordering surrogate, a private key, a raw signature, or an
  authentication header;
* records follow stable creation order, an existing subject without
  revocations is an empty collection, and an unknown subject is the
  existing ``unknown_actor`` 404;
* the GET body must be empty; ``limit`` is strictly a decimal integer in
  1..100 (default 50); pages concatenate without gaps or duplicates, the
  final cursor is null, a cursor at/past the tail returns an empty page
  with the original count, and a cursor replays identically;
* blank, malformed, tampered, wrongly-signed, wrong-family,
  cross-endpoint, and subject/limit-mismatching cursors are all
  ``422 validation_error`` -- never silently normalized -- and every
  query validation failure (including repeated or undeclared parameters)
  is decided before the subject lookup, so it is a 422 even for an
  unknown actor;
* the route is strictly read-only: no revocation, resource, or audit row
  is written by a success or a failure.

All fixtures are deterministic and offline (the stdlib test signer
produces the Ed25519 signatures); only fixed seed-derived public keys are
used.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime

from sqlalchemy import select

from provenance import pagination
from provenance.models import AuditEvent
from tests.test_authentication_key_revocations import (
    REVOCATIONS_PATH,
    _post_revocation,
    _revocation_body,
)
from tests.test_authentication_key_rotations import (
    SEED_A,
    SEED_B,
    _key_b64,
    _post_rotation,
    _rotation_body,
    _world,
)

URL = "/v1/actors/org-1/authentication-key-revocations"

ITEM_KEYS = {"id", "actor_id", "public_key", "reason", "revoked_at"}
PAGE_KEYS = {"items", "count", "next_cursor"}


# --- World setup ----------------------------------------------------------------


def _seed_for(i: int) -> bytes:
    # Fixed, distinct, 32-byte seeds; each derives a distinct public key.
    return (b"test-ed25519-revoke-list-%06d" % i).ljust(32, b"x")[:32]


def _create_revocations(client, count: int, *, actor: str = "org-1"):
    """Rotate in and then revoke ``count`` distinct keys for one subject.

    Every request is signed by the subject's original non-revoked
    attestation key, which stays in the authentication set throughout.
    """
    seed = SEED_A if actor == "org-1" else SEED_B
    created = []
    for i in range(count):
        key_seed = _seed_for(i)
        assert _post_rotation(
            client, _rotation_body(actor_id=actor, seed=key_seed),
            actor=actor, seed=seed,
        ).status_code == 201
        resp = _post_revocation(
            client, _revocation_body(actor_id=actor, seed=key_seed),
            actor=actor, seed=seed,
        )
        assert resp.status_code == 201, resp.text
        created.append(resp.json())
    return created


def _list(client, actor_id="org-1", **params):
    return client.get(
        f"/v1/actors/{actor_id}/authentication-key-revocations",
        params=params,
    )


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
    """Sign a hand-built token so claim-set/format failures are exercised."""
    raw = (
        claims
        if isinstance(claims, str)
        else json.dumps(claims, separators=(",", ":"))
    )
    payload = pagination._b64encode(raw.encode("utf-8"))
    signature = pagination._b64encode(pagination._sign(secret, version, payload))
    return f"{version}.{payload}.{signature}"


# --- Response shape, fields, and ordering ---------------------------------------


def test_success_body_has_exactly_the_three_page_members(client):
    _world(client)
    body = _list(client).json()
    assert set(body) == PAGE_KEYS


def test_item_has_exactly_the_public_revocation_fields(client):
    _world(client)
    created = _create_revocations(client, 1)[0]
    body = _list(client).json()
    assert body["count"] == 1
    item = body["items"][0]
    assert set(item) == ITEM_KEYS
    assert item == created
    assert datetime.fromisoformat(
        item["revoked_at"]
    ).utcoffset().total_seconds() == 0
    # Internal/secret material never appears, even under different spellings.
    serialized = json.dumps(body)
    for leak in ("seq", "private", "signature", "X-PA", "X-PT", "X-PS"):
        assert leak not in serialized


def test_revocations_follow_stable_creation_order(client):
    _world(client)
    created = _create_revocations(client, 5)
    body = _list(client).json()
    assert [i["id"] for i in body["items"]] == [r["id"] for r in created]
    ids = [i["id"] for i in body["items"]]
    assert len(ids) == len(set(ids)) == 5


def test_listing_is_scoped_to_the_path_subject(client):
    _world(client)
    _create_revocations(client, 3, actor="org-1")
    _create_revocations(client, 2, actor="org-2")
    org1 = _list(client, "org-1").json()
    org2 = _list(client, "org-2").json()
    assert org1["count"] == 3
    assert org2["count"] == 2
    assert {i["actor_id"] for i in org1["items"]} == {"org-1"}
    assert {i["actor_id"] for i in org2["items"]} == {"org-2"}
    assert not {i["id"] for i in org1["items"]} & {
        i["id"] for i in org2["items"]
    }


def test_existing_subject_without_revocations_is_an_empty_collection(client):
    _world(client)
    body = _list(client, "org-2").json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


def test_reading_requires_no_authentication_headers(client):
    _world(client)
    _create_revocations(client, 1)
    # The reviewer route is public, exactly like the other read-only lists.
    resp = client.get(URL)
    assert resp.status_code == 200
    assert "X-PA" not in resp.request.headers


# --- Unknown subject --------------------------------------------------------------


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


# --- Pagination -------------------------------------------------------------------


def test_pagination_concatenates_without_gaps_or_duplicates(client):
    _world(client)
    created = _create_revocations(client, 5)
    all_items, pages, count = _walk_pages(client, limit=2)
    # 5 revocations -> pages of 2, 2, 1.
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    ids = [i["id"] for i in all_items]
    assert len(ids) == len(set(ids)) == 5
    assert ids == [r["id"] for r in created]
    assert all_items == _list(client).json()["items"]


def test_count_is_total_on_every_page(client):
    _world(client)
    _create_revocations(client, 5)
    _, pages, _ = _walk_pages(client, limit=2)
    all_items, _, count = _walk_pages(client, limit=3)
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    assert len(all_items) == 5


def test_last_page_cursor_is_null_on_exact_division(client):
    _world(client)
    _create_revocations(client, 4)
    first = _list(client, limit=2).json()
    assert first["next_cursor"] is not None
    second = _list(client, limit=2, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 2
    assert second["count"] == 4
    assert second["next_cursor"] is None


def test_default_limit_is_fifty(client):
    _world(client)
    _create_revocations(client, 55)
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
    _create_revocations(client, 1)
    assert _list(client, limit=1).status_code == 200
    assert _list(client, limit=100).status_code == 200


def test_cursor_at_or_past_tail_returns_empty_page_and_no_cursor(client, app):
    _world(client)
    _create_revocations(client, 2)
    secret = app.state.authentication_key_revocations_cursor_secret
    for offset in (2, 3, 100):
        cursor = _hand_cursor(
            secret,
            pagination.AUTHENTICATION_KEY_REVOCATIONS_CURSOR_VERSION,
            {"actor_id": "org-1", "limit": 50, "offset": offset},
        )
        body = _list(client, cursor=cursor).json()
        assert body == {"items": [], "count": 2, "next_cursor": None}


def test_cursor_replays_identically(client):
    _world(client)
    _create_revocations(client, 3)
    first = _list(client, limit=2).json()
    replayed = _list(client, limit=2, cursor=first["next_cursor"])
    again = _list(client, limit=2, cursor=first["next_cursor"])
    assert replayed.status_code == 200
    assert replayed.json() == again.json()


# --- Query-parameter validation ----------------------------------------------------


def test_invalid_limit_values_are_422(client):
    _world(client)
    for bad in ("0", "101", "-1", "2.0", "x", "", " 1", "+1", "01 "):
        resp = _list(client, limit=bad)
        assert resp.status_code == 422, bad
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_or_unknown_parameters_are_422(client):
    _world(client)
    repeated = client.get(f"{URL}?limit=1&limit=2")
    assert repeated.status_code == 422
    repeated_cursor = client.get(f"{URL}?cursor=a&cursor=b")
    assert repeated_cursor.status_code == 422
    for param in ("subject", "reason", "offset", "limit2"):
        resp = _list(client, **{param: "1"})
        assert resp.status_code == 422, param
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_empty_body_is_422(client):
    _world(client)
    for body in (b"{}", b" ", b"not json"):
        resp = client.request("GET", URL, content=body)
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"


def test_validation_failures_precede_the_subject_lookup(client):
    _world(client)
    # A malformed request for an unknown subject is a 422, never a 404.
    assert _list(client, "ghost", limit="x").status_code == 422
    assert _list(client, "ghost", bogus="1").status_code == 422
    assert client.request(
        "GET", "/v1/actors/ghost/authentication-key-revocations",
        content=b"{}",
    ).status_code == 422


# --- Cursor validation --------------------------------------------------------------


def test_blank_malformed_and_tampered_cursors_are_422(client, app):
    _world(client)
    _create_revocations(client, 2)
    first = _list(client, limit=1).json()
    good = first["next_cursor"]

    assert _list(client, cursor="").status_code == 422
    assert _list(client, cursor="not-a-cursor").status_code == 422
    # A flipped character in the payload invalidates the signature.
    tampered = good[:-4] + ("A" if good[-4] != "A" else "B") + good[-3:]
    assert _list(client, cursor=tampered).status_code == 422
    # A token signed with a different secret is unverifiable.
    foreign = _hand_cursor(
        secrets.token_bytes(32),
        pagination.AUTHENTICATION_KEY_REVOCATIONS_CURSOR_VERSION,
        {"actor_id": "org-1", "limit": 1, "offset": 1},
    )
    assert _list(client, cursor=foreign).status_code == 422


def test_wrong_family_and_cross_endpoint_cursors_are_422(client, app):
    _world(client)
    _create_revocations(client, 2)
    secret = app.state.authentication_key_revocations_cursor_secret
    claims = {"actor_id": "org-1", "limit": 1, "offset": 1}

    # Same claims, wrong family marker.
    wrong_version = _hand_cursor(secret, "zz9", claims)
    assert _list(client, cursor=wrong_version).status_code == 422

    # A genuine rotations-list cursor is a different family.
    rotation_cursor = client.get(
        "/v1/actors/org-1/authentication-key-rotations", params={"limit": 1}
    ).json()["next_cursor"]
    assert rotation_cursor is not None
    assert _list(client, cursor=rotation_cursor).status_code == 422


def test_subject_and_limit_mismatching_cursors_are_422(client, app):
    _world(client)
    _create_revocations(client, 3)
    secret = app.state.authentication_key_revocations_cursor_secret

    other_subject = _hand_cursor(
        secret,
        pagination.AUTHENTICATION_KEY_REVOCATIONS_CURSOR_VERSION,
        {"actor_id": "org-2", "limit": 1, "offset": 1},
    )
    assert _list(client, cursor=other_subject).status_code == 422

    other_limit = _hand_cursor(
        secret,
        pagination.AUTHENTICATION_KEY_REVOCATIONS_CURSOR_VERSION,
        {"actor_id": "org-1", "limit": 2, "offset": 1},
    )
    assert _list(client, limit=1, cursor=other_limit).status_code == 422

    # A structurally invalid claim set is rejected even when well-signed.
    for bad_claims in (
        {"actor_id": "", "limit": 1, "offset": 1},
        {"actor_id": "org-1", "limit": 0, "offset": 1},
        {"actor_id": "org-1", "limit": 101, "offset": 1},
        {"actor_id": "org-1", "limit": 1, "offset": 0},
        {"actor_id": "org-1", "limit": "1", "offset": 1},
    ):
        cursor = _hand_cursor(
            secret,
            pagination.AUTHENTICATION_KEY_REVOCATIONS_CURSOR_VERSION,
            bad_claims,
        )
        assert _list(client, cursor=cursor).status_code == 422, bad_claims


# --- Read-only guarantee -------------------------------------------------------------


def test_listing_writes_no_state(client, db_session):
    _world(client)
    _create_revocations(client, 3)
    events_before = db_session.execute(select(AuditEvent)).scalars().all()

    assert _list(client, limit=2).status_code == 200
    assert _list(client, "ghost").status_code == 404
    assert _list(client, limit="x").status_code == 422
    assert client.request("GET", URL, content=b"{}").status_code == 422

    assert (
        db_session.execute(select(AuditEvent)).scalars().all()
        == events_before
    )
