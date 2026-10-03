"""Offline deterministic tests for the public authentication-key listing.

Covers ``GET /v1/actors/{actor_id}/authentication-keys``:

* the success body carries exactly ``{"actor_id", "items", "count",
  "next_cursor"}``; each item is exactly ``{"public_key", "sources"}`` with
  the standard-Base64 32-byte key spelling, and each source is exactly
  ``{"source_type", "source_id", "created_at"}`` with ``source_type`` in
  ``{"attestation", "rotation"}`` -- never a private key, a raw signature,
  an authentication header, a payload, or an internal ordering surrogate;
* the result is exactly the current authentication set: non-revoked
  attestation keys and active rotation keys, deduplicated by key bytes with
  every valid source merged into one item, items ordered by their first
  valid source and sources in the same stable order;
* revoked attestation keys and retired rotation keys disappear (a key
  carried by another still-valid source stays with that source only) and
  never resurrect; an existing subject without valid keys is an empty
  collection with ``count`` 0; an unknown subject is the existing
  ``unknown_actor`` 404;
* ``limit`` is strictly a decimal integer in 1..100 (default 50); pages
  concatenate without gaps or duplicates, ``count`` is the deduplicated
  total on every page, the final cursor is null, a cursor at/past the tail
  returns an empty page with the original count, and a cursor replays
  identically;
* a non-empty body, repeated or undeclared parameters, an illegal limit,
  and a blank, malformed, tampered, wrongly-signed, wrong-family,
  cross-endpoint, cross-subject, or limit-mismatching cursor are all
  ``422 validation_error`` decided before the subject lookup; non-GET
  methods are ``405 method_not_allowed``;
* the route is strictly read-only: no attestation, rotation, resource, or
  audit row is written by a success or a failure, repeated reads are
  identical, and the existing rotation, retirement, history, and protected
  read routes are unchanged.

All fixtures are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures); only fixed seed-derived public keys are used.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime

from sqlalchemy import func, select

from provenance import pagination
from provenance.models import (
    Attestation,
    AuditEvent,
    AuthenticationKeyRotation,
)
from tests.helpers import SEED_A, SEED_B, create_actor
from tests.test_authentication_key_rotations import (
    SEED_R1,
    SEED_R2,
    _key_b64,
    _make_attestation,
    _make_claim,
    _post_retire,
    _post_rotation,
    _protected_get,
    _rotation_body,
    _world,
)

URL = "/v1/actors/org-1/authentication-keys"

PAGE_KEYS = {"actor_id", "items", "count", "next_cursor"}
ITEM_KEYS = {"public_key", "sources"}
SOURCE_KEYS = {"source_type", "source_id", "created_at"}

# Deterministic seed for an extra attestation key.
SEED_C = b"test-ed25519-authkeys-c-0000000000"[:32]

# Extra content digests for additional claims (DIGEST_A/DIGEST_B are taken
# by the ``_world`` bootstrap contents).
DIGEST_KEYS_C = hashlib.sha256(b"content-authkeys-c").hexdigest()
DIGEST_KEYS_D = hashlib.sha256(b"content-authkeys-d").hexdigest()


# --- World setup ----------------------------------------------------------------


def _seed_for(i: int) -> bytes:
    # Fixed, distinct, 32-byte seeds; each derives a distinct public key.
    return (b"test-ed25519-authkeys-%06d" % i).ljust(32, b"x")[:32]


def _create_rotations(client, count: int, *, actor: str = "org-1"):
    """Create ``count`` distinct active rotations for one subject."""
    seed = SEED_A if actor == "org-1" else SEED_B
    created = []
    for i in range(count):
        body = {"actor_id": actor, "new_public_key": _key_b64(_seed_for(i))}
        resp = _post_rotation(client, body, actor=actor, seed=seed)
        assert resp.status_code == 201, resp.text
        created.append(resp.json())
    return created


def _revoke(client, attestation_id, *, revoker="org-1"):
    resp = client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": attestation_id,
            "revoker_actor_id": revoker,
            "reason": "no longer relied upon",
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _list(client, actor_id="org-1", **params):
    return client.get(f"/v1/actors/{actor_id}/authentication-keys",
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
    """Sign a hand-built token so claim-set/format failures are exercised."""
    raw = (
        claims
        if isinstance(claims, str)
        else json.dumps(claims, separators=(",", ":"))
    )
    payload = pagination._b64encode(raw.encode("utf-8"))
    signature = pagination._b64encode(pagination._sign(secret, version, payload))
    return f"{version}.{payload}.{signature}"


# --- Response shape, fields, and sources ---------------------------------------


def test_success_body_has_exactly_the_four_page_members(client):
    _world(client)
    body = _list(client).json()
    assert set(body) == PAGE_KEYS
    assert body["actor_id"] == "org-1"


def test_item_and_source_have_exactly_the_public_fields(client):
    att1, _ = _world(client)
    body = _list(client).json()
    assert body["count"] == 1
    item = body["items"][0]
    assert set(item) == ITEM_KEYS
    assert item["public_key"] == _key_b64(SEED_A)
    assert len(item["sources"]) == 1
    source = item["sources"][0]
    assert set(source) == SOURCE_KEYS
    assert source["source_type"] == "attestation"
    assert source["source_id"] == att1["id"]
    assert source["created_at"] == att1["created_at"]
    # Secret/internal material never appears, even under different spellings.
    serialized = json.dumps(body)
    for leak in ("seq", "private", "signature", "payload", "X-PA", "X-PT",
                 "X-PS"):
        assert leak not in serialized


def test_public_key_is_canonical_standard_base64_of_32_bytes(client):
    import base64

    _world(client)
    _post_rotation(client, _rotation_body())
    item = next(
        i for i in _list(client).json()["items"]
        if i["sources"][0]["source_type"] == "rotation"
    )
    raw = base64.b64decode(item["public_key"], validate=True)
    assert len(raw) == 32
    assert base64.b64encode(raw).decode("ascii") == item["public_key"]


def test_source_timestamps_are_utc(client):
    _world(client)
    _post_rotation(client, _rotation_body())
    for item in _list(client).json()["items"]:
        for source in item["sources"]:
            assert datetime.fromisoformat(
                source["created_at"]
            ).utcoffset().total_seconds() == 0


def test_rotation_key_appears_with_a_rotation_source(client):
    _world(client)
    created = _post_rotation(client, _rotation_body()).json()
    items = _list(client).json()["items"]
    assert len(items) == 2
    rotation_item = next(
        i for i in items if i["public_key"] == _key_b64(SEED_R1)
    )
    assert rotation_item["sources"] == [
        {
            "source_type": "rotation",
            "source_id": created["id"],
            "created_at": created["created_at"],
        }
    ]


def test_items_follow_first_valid_source_order(client):
    _world(client)
    # SEED_A attestation exists first; then two rotations; then a second
    # attestation carrying a new key. Items follow that introduction order.
    r1 = _post_rotation(client, _rotation_body(seed=SEED_R1)).json()
    r2 = _post_rotation(
        client, _rotation_body(seed=SEED_R2), seed=SEED_R1
    ).json()
    att_c = _make_attestation(
        client, "org-1", SEED_C,
        _make_claim(client, "org-1", digest=DIGEST_KEYS_C)["id"],
    )
    items = _list(client).json()["items"]
    assert [i["public_key"] for i in items] == [
        _key_b64(SEED_A),
        r1["new_public_key"],
        r2["new_public_key"],
        _key_b64(SEED_C),
    ]
    assert items[3]["sources"][0]["source_id"] == att_c["id"]


def test_same_key_from_attestation_and_rotation_merges_into_one_item(client):
    _world(client)
    # Rotate in the very key the existing attestation already carries.
    created = _post_rotation(client, _rotation_body(seed=SEED_A)).json()
    body = _list(client).json()
    assert body["count"] == 1
    item = body["items"][0]
    assert item["public_key"] == _key_b64(SEED_A)
    # Both valid sources are listed, attestation first (it was created
    # first), and the key appears exactly once.
    assert [s["source_type"] for s in item["sources"]] == [
        "attestation",
        "rotation",
    ]
    assert item["sources"][1]["source_id"] == created["id"]


def test_multiple_attestations_with_one_key_merge_in_creation_order(client):
    _world(client)
    second = _make_attestation(
        client, "org-1", SEED_A,
        _make_claim(client, "org-1", digest=DIGEST_KEYS_D)["id"],
    )
    body = _list(client).json()
    assert body["count"] == 1
    sources = body["items"][0]["sources"]
    assert [s["source_type"] for s in sources] == ["attestation"] * 2
    assert sources[1]["source_id"] == second["id"]
    assert sources[0]["created_at"] <= sources[1]["created_at"]


def test_listing_is_scoped_to_the_path_subject(client):
    _world(client)
    _create_rotations(client, 2, actor="org-1")
    _create_rotations(client, 3, actor="org-2")
    org1 = _list(client, "org-1").json()
    org2 = _list(client, "org-2").json()
    assert org1["count"] == 3  # one attestation key + two rotation keys
    assert org2["count"] == 4  # one attestation key + three rotation keys
    assert org1["actor_id"] == "org-1"
    assert org2["actor_id"] == "org-2"
    # The same key bytes under both subjects stay independent per subject.
    shared = _key_b64(_seed_for(0))
    assert shared in {i["public_key"] for i in org1["items"]}
    assert shared in {i["public_key"] for i in org2["items"]}


def test_existing_subject_without_any_keys_is_an_empty_collection(client):
    _world(client)
    create_actor(client, actor_id="org-9", name="No Keys", type="device")
    body = _list(client, "org-9").json()
    assert body == {
        "actor_id": "org-9",
        "items": [],
        "count": 0,
        "next_cursor": None,
    }


def test_reading_requires_no_authentication_headers(client):
    _world(client)
    resp = client.get(URL)
    assert resp.status_code == 200
    assert "X-PA" not in resp.request.headers


def test_repeated_reads_are_identical(client):
    _world(client)
    _create_rotations(client, 3)
    assert _list(client).json() == _list(client).json()


# --- Revocation and retirement remove keys -------------------------------------


def test_revoked_attestation_key_disappears(client):
    att1, _ = _world(client)
    assert _list(client).json()["count"] == 1
    _revoke(client, att1["id"])
    body = _list(client).json()
    assert body == {
        "actor_id": "org-1",
        "items": [],
        "count": 0,
        "next_cursor": None,
    }


def test_retired_rotation_key_disappears(client):
    _world(client)
    created = _post_rotation(client, _rotation_body()).json()
    assert _list(client).json()["count"] == 2
    assert _post_retire(client, created["id"], seed=SEED_R1).status_code == 200
    body = _list(client).json()
    assert body["count"] == 1
    assert [i["public_key"] for i in body["items"]] == [_key_b64(SEED_A)]


def test_key_with_one_revoked_source_keeps_its_remaining_source(client):
    att1, _ = _world(client)
    # The attestation key is also carried by an active rotation.
    _post_rotation(client, _rotation_body(seed=SEED_A))
    _revoke(client, att1["id"])
    body = _list(client).json()
    assert body["count"] == 1
    item = body["items"][0]
    assert item["public_key"] == _key_b64(SEED_A)
    assert [s["source_type"] for s in item["sources"]] == ["rotation"]


def test_revoked_and_retired_keys_never_resurrect(client):
    att1, _ = _world(client)
    created = _post_rotation(client, _rotation_body()).json()
    _revoke(client, att1["id"])
    assert _post_retire(client, created["id"], seed=SEED_R1).status_code == 200
    for _ in range(3):
        body = _list(client).json()
        assert body["count"] == 0
        assert body["items"] == []


def test_listing_matches_the_keys_that_authenticate(client):
    att1, _ = _world(client)
    created = _post_rotation(client, _rotation_body(seed=SEED_R1)).json()
    dead = _post_rotation(
        client, _rotation_body(seed=SEED_R2), seed=SEED_R1
    ).json()
    assert _post_retire(client, dead["id"], seed=SEED_R2).status_code == 200

    listed = {i["public_key"] for i in _list(client).json()["items"]}
    assert listed == {_key_b64(SEED_A), created["new_public_key"]}
    # The listed keys authenticate the protected read; the retired key and
    # a revoked attestation key do not.
    assert _protected_get(client, att1["id"], actor="org-1",
                          seed=SEED_A).status_code == 200
    assert _protected_get(client, att1["id"], actor="org-1",
                          seed=SEED_R1).status_code == 200
    assert _protected_get(client, att1["id"], actor="org-1",
                          seed=SEED_R2).status_code == 404
    _revoke(client, att1["id"])
    assert _protected_get(client, att1["id"], actor="org-1",
                          seed=SEED_A).status_code == 404
    assert {i["public_key"] for i in _list(client).json()["items"]} == {
        created["new_public_key"]
    }


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
    _create_rotations(client, 4)
    all_items, pages, count = _walk_pages(client, limit=2)
    # 1 attestation key + 4 rotation keys -> pages of 2, 2, 1.
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    keys = [i["public_key"] for i in all_items]
    assert len(keys) == len(set(keys)) == 5
    assert all_items == _list(client).json()["items"]


def test_count_is_the_deduplicated_total_on_every_page(client):
    _world(client)
    # Two rotations re-carrying the attestation key add sources, not items.
    _post_rotation(client, _rotation_body(seed=SEED_A))
    _create_rotations(client, 3)
    _, pages, _ = _walk_pages(client, limit=2)
    all_items, _, count = _walk_pages(client, limit=3)
    assert count == 4
    assert [len(page) for page in pages] == [2, 2]
    assert len(all_items) == 4


def test_last_page_cursor_is_null_on_exact_division(client):
    _world(client)
    _create_rotations(client, 3)
    first = _list(client, limit=2).json()
    assert first["next_cursor"] is not None
    second = _list(client, limit=2, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 2
    assert second["count"] == 4
    assert second["next_cursor"] is None


def test_default_limit_is_fifty(client):
    _world(client)
    _create_rotations(client, 54)
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
    for value in (1, 100):
        assert _list(client, limit=value).status_code == 200


def test_reusing_a_cursor_replays_the_same_page(client):
    _world(client)
    _create_rotations(client, 3)
    cursor = _list(client, limit=2).json()["next_cursor"]
    replay_one = _list(client, limit=2, cursor=cursor).json()
    replay_two = _list(client, limit=2, cursor=cursor).json()
    assert replay_one == replay_two


def test_issued_cursors_use_the_dedicated_opaque_family_marker(client):
    _world(client)
    _create_rotations(client, 1)
    token = _list(client, limit=1).json()["next_cursor"]
    assert token.startswith(
        pagination.AUTHENTICATION_KEYS_CURSOR_VERSION + "."
    )
    assert len(token.split(".")) == 3


def test_cursor_past_end_returns_empty_page_with_original_count(client, app):
    _world(client)
    _create_rotations(client, 4)
    token = pagination.encode_typed_cursor(
        app.state.authentication_keys_cursor_secret,
        pagination.AUTHENTICATION_KEYS_CURSOR,
        {"actor_id": "org-1", "limit": 50, "offset": 99},
    )
    body = _list(client, cursor=token).json()
    assert body["items"] == []
    assert body["count"] == 5
    assert body["next_cursor"] is None


def test_tail_cursor_returns_empty_items_and_same_count(client, app):
    _world(client)
    _create_rotations(client, 3)
    # A cursor pointing exactly at the tail (offset == total) is validly
    # signed but yields an empty page; the response still reports count 4.
    token = pagination.encode_typed_cursor(
        app.state.authentication_keys_cursor_secret,
        pagination.AUTHENTICATION_KEYS_CURSOR,
        {"actor_id": "org-1", "limit": 2, "offset": 4},
    )
    body = _list(client, limit=2, cursor=token).json()
    assert body["items"] == []
    assert body["count"] == 4
    assert body["next_cursor"] is None


# --- Cursor integrity ------------------------------------------------------------


def test_blank_malformed_or_tampered_cursors_are_validation_errors(client, app):
    _world(client)
    _create_rotations(client, 1)
    good = _list(client, limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    foreign = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.AUTHENTICATION_KEYS_CURSOR,
        {"actor_id": "org-1", "limit": 1, "offset": 1},
    )
    secret = app.state.authentication_keys_cursor_secret
    for token in (
        "",
        "   ",
        "\t",
        "not-a-cursor",
        "au1.onlytwoparts",
        "au1.too.many.parts",
        "au0.x.y",
        "au2.x.y",
        # Every other family marker, even signed with this family's secret,
        # is a version mismatch rather than a trusted token.
        _hand_cursor(secret, "v1", {"x": 1}),
        _hand_cursor(secret, "ak1", {"x": 1}),
        _hand_cursor(secret, "ce1", {"x": 1}),
        _hand_cursor(secret, "ae1", {"x": 1}),
        _hand_cursor(secret, "cl1", {"x": 1}),
        _hand_cursor(secret, "eb1", {"x": 1}),
        # Correctly signed, but structurally invalid claim payloads.
        _hand_cursor(secret, "au1", "not-json"),
        _hand_cursor(
            secret, "au1",
            {"actor_id": "org-1", "limit": 1, "offset": 1, "extra": 2},
        ),
        _hand_cursor(secret, "au1", {"actor_id": "org-1", "limit": 1}),
        _hand_cursor(secret, "au1", {"actor_id": "", "limit": 1, "offset": 1}),
        _hand_cursor(secret, "au1", {"actor_id": "org-1", "limit": 101, "offset": 1}),
        _hand_cursor(secret, "au1", {"actor_id": "org-1", "limit": 0, "offset": 1}),
        _hand_cursor(secret, "au1", {"actor_id": "org-1", "limit": 1, "offset": 0}),
        _hand_cursor(secret, "au1", {"actor_id": 7, "limit": 1, "offset": 1}),
        _hand_cursor(
            secret, "au1", {"actor_id": "org-1", "limit": "1", "offset": 1}
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
    _create_rotations(client, 1)
    secret = app.state.authentication_keys_cursor_secret
    # Tokens minted by other families (under this process's own secrets) can
    # never resume an authentication-keys page -- including the sibling
    # rotation-history family, which shares the claim shape.
    others = [
        pagination.encode_typed_cursor(
            app.state.authentication_key_rotations_cursor_secret,
            pagination.AUTHENTICATION_KEY_ROTATIONS_CURSOR,
            {"actor_id": "org-1", "limit": 1, "offset": 1},
        ),
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
    ]
    for token in others:
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


def test_authentication_keys_cursor_is_rejected_by_other_endpoints(client):
    _world(client)
    _create_rotations(client, 1)
    cursor = _list(client, limit=1).json()["next_cursor"]
    for path in (
        "/v1/actors/org-1/authentication-key-rotations",
        "/v1/claims",
        "/v1/evidence-bundles",
        "/v1/audit-events",
    ):
        resp = client.get(path, params={"limit": 1, "cursor": cursor})
        assert resp.status_code == 422, path
        assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_is_bound_to_its_subject(client, app):
    _world(client)
    _create_rotations(client, 2, actor="org-1")
    _create_rotations(client, 2, actor="org-2")
    cursor = _list(client, "org-1", limit=2).json()["next_cursor"]

    # Same cursor on another existing subject is a mismatch, not a page.
    resp = _list(client, "org-2", limit=2, cursor=cursor)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    # A correctly-signed cursor naming org-1 cannot be presented on the
    # org-2 path.
    secret = app.state.authentication_keys_cursor_secret
    org1_token = pagination.encode_typed_cursor(
        secret,
        pagination.AUTHENTICATION_KEYS_CURSOR,
        {"actor_id": "org-1", "limit": 2, "offset": 2},
    )
    assert _list(client, "org-2", limit=2, cursor=org1_token).status_code == 422


def test_cursor_is_bound_to_its_effective_limit(client):
    _world(client)
    _create_rotations(client, 3)
    cursor = _list(client, limit=2).json()["next_cursor"]
    for params in ({"limit": 3}, {"limit": 1}, {}):
        # Presenting the limit-2 cursor with another limit (or with the
        # default 50) is a mismatch rather than a re-paging.
        resp = _list(client, cursor=cursor, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_mismatch_is_rejected_before_actor_lookup(client):
    _world(client)
    _create_rotations(client, 1)
    cursor = _list(client, "org-1", limit=1).json()["next_cursor"]
    # The subject in the path does not exist; the bound cursor still
    # mismatches and validation wins over the unknown_actor lookup.
    resp = _list(client, "ghost", limit=1, cursor=cursor)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_secret_rotation_invalidates_outstanding_cursors(client):
    _world(client)
    _create_rotations(client, 1)
    cursor = _list(client, limit=1).json()["next_cursor"]
    client.app.state.authentication_keys_cursor_secret = secrets.token_bytes(32)
    resp = _list(client, limit=1, cursor=cursor)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    # A fresh cursor under the new secret pages normally.
    fresh = _list(client, limit=1).json()
    assert len(fresh["items"]) == 1


# --- Parameter and body validation ----------------------------------------------


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
        "limit=1&source_type=rotation",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_empty_body_is_a_validation_error(client):
    _world(client)
    for body in (b"{}", b" ", b"\n", b"not json", b'{"limit": 1}'):
        resp = client.request("GET", URL, content=body)
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"


def test_validation_failures_are_decided_before_actor_lookup(client):
    # Even with an unknown subject, every malformed request is a 422 rather
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
            "/v1/actors/ghost/authentication-keys?" + suffix
        )
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"
    resp = client.request(
        "GET", "/v1/actors/ghost/authentication-keys", content=b"{}"
    )
    assert resp.status_code == 422
    # A structurally valid request for the same unknown subject is the 404.
    plain = client.get("/v1/actors/ghost/authentication-keys")
    assert plain.status_code == 404


def test_validation_failure_locates_the_query_field(client):
    _world(client)
    resp = _list(client, limit="nope")
    issue = resp.json()["error"]["details"]["issues"][0]
    assert issue["loc"] == ["query", "limit"]
    resp = _list(client, cursor="garbage")
    issue = resp.json()["error"]["details"]["issues"][0]
    assert issue["loc"] == ["query", "cursor"]


def test_non_get_methods_are_405(client):
    _world(client)
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        resp = client.request(method, URL)
        assert resp.status_code == 405, method
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Read-only guarantee ---------------------------------------------------------


def test_reads_and_failures_write_nothing(client, db_session):
    _world(client)
    _create_rotations(client, 4)

    def counts():
        return (
            db_session.scalar(
                select(func.count()).select_from(Attestation)
            ),
            db_session.scalar(
                select(func.count()).select_from(AuthenticationKeyRotation)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    before = counts()

    # Successful reads: unfiltered, paginated through the tail, and the
    # empty-collection case for another subject.
    _walk_pages(client, limit=2)
    _list(client)
    _list(client, "org-2")

    # Failed reads: non-empty body, malformed limit/cursor, repeats, unknown
    # params, unknown subject, and a bound-cursor mismatch.
    client.request("GET", URL, content=b"{}")
    _list(client, limit=0)
    _list(client, limit=101)
    _list(client, cursor="tampered")
    _list(client, "ghost")
    client.get(f"{URL}?limit=1&limit=2")
    client.get(f"{URL}?unknown=1")
    cursor = _list(client, "org-1", limit=1).json()["next_cursor"]
    _list(client, "org-2", limit=1, cursor=cursor)

    db_session.expire_all()
    assert counts() == before


# --- Compatibility with the existing routes -------------------------------------


def test_existing_rotation_and_protected_routes_remain_unchanged(client):
    att1, _ = _world(client)
    created = _post_rotation(client, _rotation_body())
    assert created.status_code == 201
    # A repeat submission is still the idempotent 200 with no new record.
    repeat = _post_rotation(client, _rotation_body())
    assert repeat.status_code == 200
    assert repeat.json() == created.json()
    # The rotation history still lists active and retired records together.
    history = client.get(
        "/v1/actors/org-1/authentication-key-rotations"
    ).json()
    assert history["count"] == 1
    assert history["items"] == [created.json()]
    # The protected read still authenticates with the current key union.
    assert _protected_get(client, att1["id"], actor="org-1",
                          seed=SEED_R1).status_code == 200
