"""Offline deterministic tests for the public authentication-key listing.

Covers ``GET /v1/actors/{actor_id}/authentication-keys``:

* the success body carries exactly ``{"actor_id", "items", "count",
  "next_cursor"}``; each item is exactly ``{"public_key", "sources"}`` with
  the standard-Base64 32-byte key spelling, and each source is exactly
  ``{"source_type", "source_id", "created_at"}`` with ``source_type`` in
  ``{"attestation", "rotation"}`` -- never a private key, a raw signature,
  an authentication header, a payload, or an internal ordering surrogate;
* the result reflects exactly the existing authentication rules: the union
  of non-revoked attestation keys and active rotation keys, deduplicated by
  public-key bytes with every valid source merged under one item, items in
  first-valid-source order and sources in the same order;
* a revoked attestation key and a retired rotation key disappear and never
  resurrect, while a key still carried by another valid source remains with
  only its valid sources;
* ``limit`` is strictly a decimal integer in 1..100 (default 50); pages
  concatenate without gaps or duplicates, ``count`` is the deduplicated
  total on every page, the final cursor is null, and a cursor at/past the
  tail returns an empty page with the original count;
* blank, malformed, tampered, wrongly-signed, wrong-family, wrong-claim-set,
  cross-endpoint, and subject/limit-mismatching cursors are all
  ``422 validation_error``, and every query validation failure (including a
  non-empty body and repeated or undeclared parameters) is decided before
  the subject lookup; an unknown subject is the existing ``unknown_actor``
  404 and a non-GET method is ``405 method_not_allowed``;
* the route is strictly read-only: no resource or audit row is written by a
  success or a failure, repeated reads are identical, and the existing
  rotation routes and the authentication key union are unchanged.

All fixtures are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures); only fixed seed-derived public keys are used.
"""

from __future__ import annotations

import base64
import json
import secrets
from datetime import datetime

from sqlalchemy import func, select

from provenance import pagination
from provenance.models import (
    Attestation,
    AttestationRevocation,
    AuditEvent,
    AuthenticationKeyRotation,
)
from tests.helpers import DIGEST_B, DIGEST_C, SEED_B, create_actor
from tests.test_authentication_key_rotations import (
    SEED_A,
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


# --- World setup ----------------------------------------------------------------


def _seed_for(i: int) -> bytes:
    # Fixed, distinct, 32-byte seeds; each derives a distinct public key.
    return (b"test-ed25519-authkeys-%06d" % i).ljust(32, b"x")[:32]


def _create_rotations(client, count: int, *, actor: str = "org-1"):
    """Create ``count`` distinct rotations for one subject (see the rotations
    list tests: every request is signed by the subject's original non-revoked
    attestation key)."""
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
            "reason": "key material retired",
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


# --- Response shape, fields, and key spelling -----------------------------------


def test_success_body_has_exactly_the_four_page_members(client):
    _world(client)
    body = _list(client).json()
    assert set(body) == PAGE_KEYS
    assert body["actor_id"] == "org-1"


def test_item_and_source_have_exactly_the_public_fields(client):
    _world(client)
    _post_rotation(client, _rotation_body())
    body = _list(client).json()
    assert body["count"] == 2
    for item in body["items"]:
        assert set(item) == ITEM_KEYS
        for source in item["sources"]:
            assert set(source) == SOURCE_KEYS
            assert source["source_type"] in ("attestation", "rotation")
    # Internal/secret material never appears, even under different spellings.
    serialized = json.dumps(body)
    for leak in ("seq", "private", "signature", "payload", "X-PA", "X-PT",
                 "X-PS"):
        assert leak not in serialized


def test_public_key_is_the_standard_base64_32_byte_spelling(client):
    _world(client)
    _post_rotation(client, _rotation_body())
    body = _list(client).json()
    spellings = {item["public_key"] for item in body["items"]}
    assert spellings == {_key_b64(SEED_A), _key_b64(SEED_R1)}
    for spelling in spellings:
        raw = base64.b64decode(spelling, validate=True)
        assert len(raw) == 32
        # The served spelling is the canonical standard-Base64 one.
        assert base64.b64encode(raw).decode("ascii") == spelling


def test_source_created_at_is_utc(client):
    _world(client)
    item = _list(client).json()["items"][0]
    created_at = datetime.fromisoformat(item["sources"][0]["created_at"])
    assert created_at.utcoffset().total_seconds() == 0


def test_attestation_source_matches_the_existing_attestation(client):
    _world(client)
    attestation = client.get("/v1/attestations").json()["items"][0]
    body = _list(client).json()
    assert body["count"] == 1
    item = body["items"][0]
    assert item["public_key"] == attestation["public_key"]
    assert item["sources"] == [
        {
            "source_type": "attestation",
            "source_id": attestation["id"],
            "created_at": attestation["created_at"],
        }
    ]


def test_rotation_source_matches_the_existing_rotation(client):
    _world(client)
    created = _post_rotation(client, _rotation_body()).json()
    body = _list(client).json()
    rotated = next(
        item for item in body["items"]
        if item["public_key"] == created["new_public_key"]
    )
    assert rotated["sources"] == [
        {
            "source_type": "rotation",
            "source_id": created["id"],
            "created_at": created["created_at"],
        }
    ]


def test_reading_requires_no_authentication_headers(client):
    _world(client)
    resp = client.get(URL)
    assert resp.status_code == 200
    assert "X-PA" not in resp.request.headers


# --- Ordering, deduplication, and multi-source merging ---------------------------


def test_items_follow_first_valid_source_order(client):
    _world(client)
    rotations = _create_rotations(client, 3)
    body = _list(client).json()
    # The bootstrap attestation key is first; the rotation keys follow in
    # rotation creation order.
    assert [item["public_key"] for item in body["items"]] == [
        _key_b64(SEED_A),
        *[r["new_public_key"] for r in rotations],
    ]
    assert body["count"] == 4


def test_repeated_reads_are_byte_identical(client):
    _world(client)
    _create_rotations(client, 3)
    first = _list(client)
    second = _list(client)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_same_key_from_attestation_and_rotation_merges_into_one_item(client):
    _world(client)
    attestation = client.get("/v1/attestations").json()["items"][0]
    # Rotate in the very key the bootstrap attestation already carries.
    resp = _post_rotation(
        client, {"actor_id": "org-1", "new_public_key": _key_b64(SEED_A)}
    )
    assert resp.status_code == 201, resp.text
    rotation = resp.json()

    body = _list(client).json()
    assert body["count"] == 1
    (item,) = body["items"]
    assert item["public_key"] == _key_b64(SEED_A)
    # Both valid sources are listed once, in source creation order.
    assert item["sources"] == [
        {
            "source_type": "attestation",
            "source_id": attestation["id"],
            "created_at": attestation["created_at"],
        },
        {
            "source_type": "rotation",
            "source_id": rotation["id"],
            "created_at": rotation["created_at"],
        },
    ]


def test_same_key_from_two_attestations_merges_into_one_item(client):
    _world(client)
    second = _make_attestation(
        client, "org-1", SEED_A,
        _make_claim(client, "org-1", digest=DIGEST_C)["id"],
    )
    body = _list(client).json()
    assert body["count"] == 1
    (item,) = body["items"]
    assert [s["source_type"] for s in item["sources"]] == [
        "attestation",
        "attestation",
    ]
    assert item["sources"][1]["source_id"] == second["id"]
    assert item["sources"][1]["created_at"] == second["created_at"]


def test_listing_is_scoped_to_the_path_subject(client):
    _world(client)
    _create_rotations(client, 2, actor="org-1")
    _create_rotations(client, 1, actor="org-2")
    org1 = _list(client, "org-1").json()
    org2 = _list(client, "org-2").json()
    assert org1["count"] == 3
    assert org2["count"] == 2
    assert org1["actor_id"] == "org-1"
    assert org2["actor_id"] == "org-2"
    # No source record ever leaks across the subject boundary.
    org1_sources = {s["source_id"] for i in org1["items"] for s in i["sources"]}
    org2_sources = {s["source_id"] for i in org2["items"] for s in i["sources"]}
    assert not org1_sources & org2_sources


def test_existing_subject_without_any_valid_key_is_an_empty_collection(client):
    _world(client)
    create_actor(client, actor_id="org-9", name="No Keys", type="device")
    body = _list(client, "org-9").json()
    assert body == {
        "actor_id": "org-9",
        "items": [],
        "count": 0,
        "next_cursor": None,
    }


# --- Revocation and retirement remove keys ---------------------------------------


def test_revoking_the_only_attestation_removes_its_key(client):
    _world(client)
    attestation = client.get("/v1/attestations").json()["items"][0]
    assert _list(client).json()["count"] == 1

    _revoke(client, attestation["id"])
    body = _list(client).json()
    assert body == {
        "actor_id": "org-1",
        "items": [],
        "count": 0,
        "next_cursor": None,
    }


def test_retiring_the_only_rotation_removes_its_key(client):
    _world(client)
    created = _post_rotation(client, _rotation_body()).json()
    assert _list(client).json()["count"] == 2

    assert _post_retire(client, created["id"], seed=SEED_R1).status_code == 200
    body = _list(client).json()
    assert body["count"] == 1
    assert [item["public_key"] for item in body["items"]] == [_key_b64(SEED_A)]


def test_revoked_source_drops_out_but_key_survives_via_rotation(client):
    _world(client)
    attestation = client.get("/v1/attestations").json()["items"][0]
    rotation = _post_rotation(
        client, {"actor_id": "org-1", "new_public_key": _key_b64(SEED_A)}
    ).json()

    _revoke(client, attestation["id"])
    body = _list(client).json()
    assert body["count"] == 1
    (item,) = body["items"]
    assert item["public_key"] == _key_b64(SEED_A)
    # Only the still-valid rotation source remains.
    assert item["sources"] == [
        {
            "source_type": "rotation",
            "source_id": rotation["id"],
            "created_at": rotation["created_at"],
        }
    ]


def test_retired_source_drops_out_but_key_survives_via_attestation(client):
    _world(client)
    rotation = _post_rotation(
        client, {"actor_id": "org-1", "new_public_key": _key_b64(SEED_A)}
    ).json()
    assert _post_retire(client, rotation["id"]).status_code == 200

    body = _list(client).json()
    assert body["count"] == 1
    (item,) = body["items"]
    assert [s["source_type"] for s in item["sources"]] == ["attestation"]


def test_revoked_and_retired_keys_never_resurrect(client):
    _world(client)
    attestation = client.get("/v1/attestations").json()["items"][0]
    rotation = _post_rotation(client, _rotation_body()).json()
    _post_rotation(client, _rotation_body(seed=SEED_R2), seed=SEED_R1)
    assert _post_retire(client, rotation["id"], seed=SEED_R1).status_code == 200
    assert {i["public_key"] for i in _list(client).json()["items"]} == {
        _key_b64(SEED_A),
        _key_b64(SEED_R2),
    }

    # A post-retirement retry (authenticated by the still-live attestation
    # key) returns the original retired record but resurrects nothing.
    retry = _post_rotation(client, _rotation_body())
    assert retry.status_code == 200
    assert retry.json()["active"] is False
    assert {i["public_key"] for i in _list(client).json()["items"]} == {
        _key_b64(SEED_A),
        _key_b64(SEED_R2),
    }

    # Revoking the attestation removes its key as well; the retired key
    # stays dead throughout.
    _revoke(client, attestation["id"])
    body = _list(client).json()
    assert body["count"] == 1
    assert [i["public_key"] for i in body["items"]] == [_key_b64(SEED_R2)]


def test_listing_matches_the_authentication_key_union(client):
    att1, _ = _world(client)
    _post_rotation(client, _rotation_body(seed=SEED_R1))
    _post_rotation(client, _rotation_body(seed=SEED_R2), seed=SEED_R1)

    # Every listed key authenticates the protected read route.
    for seed in (SEED_A, SEED_R1, SEED_R2):
        assert _protected_get(client, att1["id"], actor="org-1",
                              seed=seed).status_code == 200
    assert {item["public_key"] for item in _list(client).json()["items"]} == {
        _key_b64(SEED_A),
        _key_b64(SEED_R1),
        _key_b64(SEED_R2),
    }

    # Retire R1 and revoke the attestation: both keys leave the listing and
    # stop authenticating at the same time; R2 keeps working.
    rotations = client.get(
        "/v1/actors/org-1/authentication-key-rotations"
    ).json()["items"]
    r1 = next(r for r in rotations if r["new_public_key"] == _key_b64(SEED_R1))
    assert _post_retire(client, r1["id"], seed=SEED_R1).status_code == 200
    _revoke(client, att1["id"])

    assert {item["public_key"] for item in _list(client).json()["items"]} == {
        _key_b64(SEED_R2)
    }
    assert _protected_get(client, att1["id"], actor="org-1",
                          seed=SEED_R1).status_code == 404
    assert _protected_get(client, att1["id"], actor="org-1",
                        seed=SEED_A).status_code == 404
    assert _protected_get(client, att1["id"], actor="org-1",
                        seed=SEED_R2).status_code == 200


# --- Unknown subject and wrong methods -------------------------------------------


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


def test_non_get_methods_are_405_method_not_allowed(client):
    _world(client)
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        resp = client.request(method, URL)
        assert resp.status_code == 405, method
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Pagination ------------------------------------------------------------------


def test_pagination_concatenates_without_gaps_or_duplicates(client):
    _world(client)
    rotations = _create_rotations(client, 4)
    all_items, pages, count = _walk_pages(client, limit=2)
    # 1 attestation key + 4 rotation keys -> pages of 2, 2, 1.
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    keys = [item["public_key"] for item in all_items]
    assert len(keys) == len(set(keys)) == 5
    assert keys == [_key_b64(SEED_A), *[r["new_public_key"] for r in rotations]]
    assert all_items == _list(client).json()["items"]


def test_count_is_the_deduplicated_total_on_every_page(client):
    _world(client)
    _create_rotations(client, 4)
    # A merged key (attestation + rotation) still counts once.
    _post_rotation(
        client, {"actor_id": "org-1", "new_public_key": _key_b64(SEED_A)}
    )
    _, pages, _ = _walk_pages(client, limit=2)
    all_items, _, count = _walk_pages(client, limit=3)
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    assert len(all_items) == 5


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
    _create_rotations(client, 52)
    first = _list(client).json()
    assert len(first["items"]) == 50
    assert first["count"] == 53
    assert first["next_cursor"] is not None
    second = _list(client, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 3
    assert second["count"] == 53
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
    assert token.startswith(pagination.AUTHENTICATION_KEYS_CURSOR_VERSION + ".")
    assert len(token.split(".")) == 3


def test_cursor_past_end_returns_empty_page_with_original_count(client, app):
    _world(client)
    _create_rotations(client, 2)
    token = pagination.encode_typed_cursor(
        app.state.authentication_keys_cursor_secret,
        pagination.AUTHENTICATION_KEYS_CURSOR,
        {"actor_id": "org-1", "limit": 50, "offset": 99},
    )
    body = _list(client, cursor=token).json()
    assert body["items"] == []
    assert body["count"] == 3
    assert body["next_cursor"] is None


def test_tail_cursor_returns_empty_items_and_same_count(client, app):
    _world(client)
    _create_rotations(client, 1)
    # A cursor pointing exactly at the tail (offset == total) is validly
    # signed but yields an empty page; the response still reports count 2.
    token = pagination.encode_typed_cursor(
        app.state.authentication_keys_cursor_secret,
        pagination.AUTHENTICATION_KEYS_CURSOR,
        {"actor_id": "org-1", "limit": 2, "offset": 2},
    )
    body = _list(client, limit=2, cursor=token).json()
    assert body["items"] == []
    assert body["count"] == 2
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
        _hand_cursor(secret, "ce1", {"x": 1}),
        _hand_cursor(secret, "ae1", {"x": 1}),
        _hand_cursor(secret, "ak1", {"x": 1}),
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
    _create_rotations(client, 2)
    secret = app.state.authentication_keys_cursor_secret
    # Tokens minted by other families (under this process's own secrets) can
    # never resume an authentication-keys page -- including the sibling
    # rotation-history family bound to the same subject and limit.
    rotation_cursor = client.get(
        "/v1/actors/org-1/authentication-key-rotations", params={"limit": 1}
    ).json()["next_cursor"]
    others = [
        rotation_cursor,
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
    _create_rotations(client, 3, actor="org-1")
    _create_rotations(client, 3, actor="org-2")
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


# --- Parameter and body validation ------------------------------------------------


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
    for body in (b"{}", b" ", b"{not json", b"\x00"):
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
        resp = client.get("/v1/actors/ghost/authentication-keys?" + suffix)
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


# --- Read-only guarantee ---------------------------------------------------------


def test_reads_and_failures_write_nothing(client, db_session):
    _world(client)
    _create_rotations(client, 3)
    create_actor(client, actor_id="org-9", name="No Keys", type="device")

    def counts():
        return (
            db_session.scalar(select(func.count()).select_from(Attestation)),
            db_session.scalar(
                select(func.count()).select_from(AttestationRevocation)
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
    _list(client, "org-9")

    # Failed reads: malformed limit/cursor, repeats, unknown params, a
    # non-empty body, an unknown subject, and a bound-cursor mismatch.
    _list(client, limit=0)
    _list(client, limit=101)
    _list(client, cursor="tampered")
    _list(client, "ghost")
    client.get(f"{URL}?limit=1&limit=2")
    client.get(f"{URL}?unknown=1")
    client.request("GET", URL, content=b"{}")
    cursor = _list(client, "org-1", limit=1).json()["next_cursor"]
    _list(client, "org-2", limit=1, cursor=cursor)

    db_session.expire_all()
    assert counts() == before


# --- Compatibility with the existing rotation routes -----------------------------


def test_existing_rotation_routes_remain_unchanged(client):
    _world(client)
    created = _post_rotation(client, _rotation_body())
    assert created.status_code == 201
    # A repeat submission is still the idempotent 200 with no new record.
    repeat = _post_rotation(client, _rotation_body())
    assert repeat.status_code == 200
    assert repeat.json() == created.json()
    # The rotation history still lists active and retired records together.
    assert _post_retire(client, created.json()["id"], seed=SEED_R1).status_code == 200
    history = client.get("/v1/actors/org-1/authentication-key-rotations").json()
    assert history["count"] == 1
    assert history["items"][0]["active"] is False
    # ...while the authentication-keys view no longer carries the dead key.
    assert {i["public_key"] for i in _list(client).json()["items"]} == {
        _key_b64(SEED_A)
    }
