"""Offline deterministic tests for the historical authentication-key query.

Covers ``GET /v1/actors/{actor_id}/authentication-keys/at``:

* the success body carries exactly ``{"actor_id", "at", "items", "count",
  "next_cursor"}``; each item is exactly ``{"public_key", "sources"}`` with
  the standard-Base64 32-byte key spelling, and each source is exactly
  ``{"source_type", "source_id", "created_at"}`` with ``source_type`` in
  ``{"attestation", "rotation"}`` -- never a private key, a raw signature,
  an authentication header, a payload, or an internal ordering surrogate;
* ``at`` is a closed-interval instant: a rotation valid at ``at`` has
  ``created_at <= at`` and ``retired_at`` null or strictly later than
  ``at``; an attestation valid at ``at`` has ``created_at <= at`` and no
  revocation with ``created_at <= at`` -- a source created exactly at
  ``at`` is included, a source retired or revoked exactly at ``at`` is
  excluded, and a key with another still-valid source survives;
* the historical view is stable: later revocations, retirements, and
  rotations never rewrite what an earlier ``at`` returns;
* keys are deduplicated by public-key bytes with every valid source merged
  under one item, items in first-valid-source order and sources in stable
  creation order, and ``count`` is the deduplicated total at ``at``;
* ``limit`` is strictly a decimal integer in 1..100 (default 50); pages
  concatenate without gaps or duplicates, the final cursor is null, and a
  cursor at/past the tail returns an empty page with the original count;
* ``cursor`` is an opaque HMAC-signed token in its own family, bound to the
  endpoint, the path subject, the resolved ``at``, and the effective limit;
  tampered, foreign, wrong-family, wrong-claim, cross-endpoint,
  cross-subject, and condition-mismatching cursors are all
  ``422 validation_error``;
* a missing/blank/non-RFC-3339-UTC ``at``, an illegal ``limit``, a repeated
  or undeclared parameter, and a non-empty GET body are all
  ``422 validation_error`` decided before the subject lookup; an unknown
  subject is the existing ``unknown_actor`` 404 and a non-GET method is
  ``405 method_not_allowed``;
* the route is strictly read-only: no resource or audit row is written by a
  success or a failure, and the existing current-keys, rotation, and
  revocation behaviour is unchanged.

All fixtures are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures); only fixed seed-derived public keys are used.
"""

from __future__ import annotations

import base64
import json
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from provenance import pagination
from provenance.models import (
    Attestation,
    AttestationRevocation,
    AuditEvent,
    AuthenticationKeyRotation,
)
from tests.helpers import DIGEST_C, create_actor
from tests.test_authentication_key_rotations import (
    SEED_A,
    SEED_B,
    SEED_R1,
    _key_b64,
    _make_attestation,
    _make_claim,
    _post_retire,
    _post_rotation,
    _rotation_body,
    _world,
)

URL = "/v1/actors/org-1/authentication-keys/at"

PAGE_KEYS = {"actor_id", "at", "items", "count", "next_cursor"}
ITEM_KEYS = {"public_key", "sources"}
SOURCE_KEYS = {"source_type", "source_id", "created_at"}

#: Fixed instants safely before/after anything the tests create.
EPOCH = "2020-01-01T00:00:00Z"
FAR_FUTURE = "2999-01-01T00:00:00Z"


# --- World setup ----------------------------------------------------------------


def _seed_for(i: int) -> bytes:
    # Fixed, distinct, 32-byte seeds; each derives a distinct public key.
    return (b"test-ed25519-authkeysat-%04d" % i).ljust(32, b"x")[:32]


def _create_rotations(client, count: int, *, actor: str = "org-1"):
    """Create ``count`` distinct rotations for one subject."""
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


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _list_at(client, actor_id="org-1", **params):
    return client.get(
        f"/v1/actors/{actor_id}/authentication-keys/at", params=params
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
        resp = _list_at(client, actor_id, **query)
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


# --- Response shape, fields, and the echoed instant -------------------------------


def test_success_body_has_exactly_the_five_page_members(client):
    _world(client)
    body = _list_at(client, at=FAR_FUTURE).json()
    assert set(body) == PAGE_KEYS
    assert body["actor_id"] == "org-1"


def test_at_is_echoed_as_the_resolved_utc_instant(client):
    _world(client)
    body = _list_at(client, at="2026-05-04T03:02:01Z").json()
    # The instant canonicalizes to the UTC "Z" spelling on the wire.
    assert body["at"] == "2026-05-04T03:02:01Z"
    # Equivalent spellings resolve to the identical response.
    again = _list_at(client, at="2026-05-04T03:02:01+00:00").json()
    assert again == body


def test_item_and_source_have_exactly_the_public_fields(client):
    _world(client)
    _post_rotation(client, _rotation_body())
    body = _list_at(client, at=FAR_FUTURE).json()
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
    body = _list_at(client, at=FAR_FUTURE).json()
    spellings = {item["public_key"] for item in body["items"]}
    assert spellings == {_key_b64(SEED_A), _key_b64(SEED_R1)}
    for spelling in spellings:
        raw = base64.b64decode(spelling, validate=True)
        assert len(raw) == 32
        assert base64.b64encode(raw).decode("ascii") == spelling


def test_reading_requires_no_authentication_headers(client):
    _world(client)
    resp = _list_at(client, at=FAR_FUTURE)
    assert resp.status_code == 200
    assert "X-PA" not in resp.request.headers


# --- The closed-interval historical judgment --------------------------------------


def test_at_before_any_source_is_an_empty_collection(client):
    _world(client)
    _create_rotations(client, 2)
    body = _list_at(client, at=EPOCH).json()
    assert body == {
        "actor_id": "org-1",
        "at": "2020-01-01T00:00:00Z",
        "items": [],
        "count": 0,
        "next_cursor": None,
    }


def test_source_created_exactly_at_is_included(client):
    _world(client)
    attestation = client.get("/v1/attestations").json()["items"][0]
    rotation = _post_rotation(client, _rotation_body()).json()

    body = _list_at(client, at=rotation["created_at"]).json()
    assert body["count"] == 2
    assert {item["public_key"] for item in body["items"]} == {
        _key_b64(SEED_A),
        _key_b64(SEED_R1),
    }

    at_creation = _list_at(client, at=attestation["created_at"]).json()
    assert at_creation["count"] == 1
    (item,) = at_creation["items"]
    assert item["sources"] == [
        {
            "source_type": "attestation",
            "source_id": attestation["id"],
            "created_at": attestation["created_at"],
        }
    ]


def test_one_instant_before_creation_is_excluded(client):
    _world(client)
    attestation = client.get("/v1/attestations").json()["items"][0]
    just_before = _iso(
        _parse(attestation["created_at"]) - timedelta(microseconds=1)
    )
    body = _list_at(client, at=just_before).json()
    assert body["items"] == []
    assert body["count"] == 0


def test_rotation_retired_exactly_at_is_excluded(client):
    _world(client)
    rotation = _post_rotation(client, _rotation_body()).json()
    retired = _post_retire(client, rotation["id"], seed=SEED_R1).json()
    assert retired["retired_at"] is not None

    # At the retirement instant itself the rotation no longer counts.
    body = _list_at(client, at=retired["retired_at"]).json()
    assert body["count"] == 1
    assert [item["public_key"] for item in body["items"]] == [_key_b64(SEED_A)]

    # One instant before retirement it still did.
    just_before = _iso(
        _parse(retired["retired_at"]) - timedelta(microseconds=1)
    )
    before = _list_at(client, at=just_before).json()
    assert before["count"] == 2


def test_attestation_revoked_exactly_at_is_excluded(client):
    _world(client)
    attestation = client.get("/v1/attestations").json()["items"][0]
    revocation = _revoke(client, attestation["id"])

    body = _list_at(client, at=revocation["created_at"]).json()
    assert body["items"] == []
    assert body["count"] == 0

    just_before = _iso(
        _parse(revocation["created_at"]) - timedelta(microseconds=1)
    )
    before = _list_at(client, at=just_before).json()
    assert before["count"] == 1
    assert [item["public_key"] for item in before["items"]] == [
        _key_b64(SEED_A)
    ]


def test_later_changes_never_rewrite_an_earlier_at(client):
    _world(client)
    attestation = client.get("/v1/attestations").json()["items"][0]
    rotation = _post_rotation(client, _rotation_body()).json()
    midpoint = _iso(
        _parse(rotation["created_at"]) + timedelta(microseconds=1)
    )
    snapshot = _list_at(client, at=midpoint)
    assert snapshot.json()["count"] == 2

    # Rotate in new keys, retire the rotation, and revoke the attestation:
    # the earlier instant's answer is byte-identical afterwards.
    _create_rotations(client, 2)
    _post_retire(client, rotation["id"], seed=SEED_R1)
    _revoke(client, attestation["id"])
    later = _list_at(client, at=midpoint)
    assert later.status_code == 200
    assert later.content == snapshot.content


def test_at_after_everything_matches_the_current_listing(client):
    _world(client)
    _create_rotations(client, 2)
    historical = _list_at(client, at=FAR_FUTURE).json()
    current = client.get("/v1/actors/org-1/authentication-keys").json()
    assert historical["items"] == current["items"]
    assert historical["count"] == current["count"]


def test_key_survives_at_via_another_still_valid_source(client):
    _world(client)
    attestation = client.get("/v1/attestations").json()["items"][0]
    # Rotate in the very key the bootstrap attestation already carries.
    rotation = _post_rotation(
        client, {"actor_id": "org-1", "new_public_key": _key_b64(SEED_A)}
    ).json()
    retired = _post_retire(client, rotation["id"]).json()

    # At the retirement instant the rotation source drops out but the key
    # remains through the still-valid attestation source.
    body = _list_at(client, at=retired["retired_at"]).json()
    assert body["count"] == 1
    (item,) = body["items"]
    assert item["public_key"] == _key_b64(SEED_A)
    assert item["sources"] == [
        {
            "source_type": "attestation",
            "source_id": attestation["id"],
            "created_at": attestation["created_at"],
        }
    ]


def test_revoked_attestation_source_drops_out_but_key_survives_at(client):
    _world(client)
    attestation = client.get("/v1/attestations").json()["items"][0]
    rotation = _post_rotation(
        client, {"actor_id": "org-1", "new_public_key": _key_b64(SEED_A)}
    ).json()
    revocation = _revoke(client, attestation["id"])

    body = _list_at(client, at=revocation["created_at"]).json()
    assert body["count"] == 1
    (item,) = body["items"]
    assert item["sources"] == [
        {
            "source_type": "rotation",
            "source_id": rotation["id"],
            "created_at": rotation["created_at"],
        }
    ]


# --- Ordering, deduplication, and multi-source merging at the instant -------------


def test_items_follow_first_valid_source_order_at(client):
    _world(client)
    rotations = _create_rotations(client, 3)
    body = _list_at(client, at=FAR_FUTURE).json()
    assert [item["public_key"] for item in body["items"]] == [
        _key_b64(SEED_A),
        *[r["new_public_key"] for r in rotations],
    ]
    assert body["count"] == 4


def test_same_key_from_attestation_and_rotation_merges_at(client):
    _world(client)
    attestation = client.get("/v1/attestations").json()["items"][0]
    rotation = _post_rotation(
        client, {"actor_id": "org-1", "new_public_key": _key_b64(SEED_A)}
    ).json()

    body = _list_at(client, at=FAR_FUTURE).json()
    assert body["count"] == 1
    (item,) = body["items"]
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


def test_same_key_from_two_attestations_merges_at(client):
    _world(client)
    second = _make_attestation(
        client, "org-1", SEED_A,
        _make_claim(client, "org-1", digest=DIGEST_C)["id"],
    )
    body = _list_at(client, at=FAR_FUTURE).json()
    assert body["count"] == 1
    (item,) = body["items"]
    assert [s["source_type"] for s in item["sources"]] == [
        "attestation",
        "attestation",
    ]
    assert item["sources"][1]["source_id"] == second["id"]


def test_only_sources_valid_at_are_merged(client):
    _world(client)
    attestation = client.get("/v1/attestations").json()["items"][0]
    # A second attestation of the same key, created later: an instant
    # between the two creations merges only the first source.
    second = _make_attestation(
        client, "org-1", SEED_A,
        _make_claim(client, "org-1", digest=DIGEST_C)["id"],
    )
    body = _list_at(client, at=attestation["created_at"]).json()
    assert body["count"] == 1
    (item,) = body["items"]
    assert [s["source_id"] for s in item["sources"]] == [attestation["id"]]
    both = _list_at(client, at=second["created_at"]).json()
    assert [s["source_id"] for s in both["items"][0]["sources"]] == [
        attestation["id"],
        second["id"],
    ]


def test_listing_is_scoped_to_the_path_subject(client):
    _world(client)
    _create_rotations(client, 2, actor="org-1")
    _create_rotations(client, 1, actor="org-2")
    org1 = _list_at(client, "org-1", at=FAR_FUTURE).json()
    org2 = _list_at(client, "org-2", at=FAR_FUTURE).json()
    assert org1["count"] == 3
    assert org2["count"] == 2
    org1_sources = {s["source_id"] for i in org1["items"] for s in i["sources"]}
    org2_sources = {s["source_id"] for i in org2["items"] for s in i["sources"]}
    assert not org1_sources & org2_sources


# --- Unknown subject and wrong methods -------------------------------------------


def test_unknown_subject_is_404_unknown_actor(client):
    _world(client)
    resp = _list_at(client, "ghost", at=FAR_FUTURE)
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "unknown_actor"
    assert error["details"]["actor_id"] == "ghost"


def test_non_get_methods_are_405_method_not_allowed(client):
    _world(client)
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        resp = client.request(method, URL, params={"at": FAR_FUTURE})
        assert resp.status_code == 405, method
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Pagination ------------------------------------------------------------------


def test_pagination_concatenates_without_gaps_or_duplicates(client):
    _world(client)
    rotations = _create_rotations(client, 4)
    all_items, pages, count = _walk_pages(client, at=FAR_FUTURE, limit=2)
    # 1 attestation key + 4 rotation keys -> pages of 2, 2, 1.
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    keys = [item["public_key"] for item in all_items]
    assert len(keys) == len(set(keys)) == 5
    assert keys == [_key_b64(SEED_A), *[r["new_public_key"] for r in rotations]]
    assert all_items == _list_at(client, at=FAR_FUTURE).json()["items"]


def test_count_is_the_deduplicated_total_at_on_every_page(client):
    _world(client)
    _create_rotations(client, 4)
    # A merged key (attestation + rotation) still counts once.
    _post_rotation(
        client, {"actor_id": "org-1", "new_public_key": _key_b64(SEED_A)}
    )
    _, pages, _ = _walk_pages(client, at=FAR_FUTURE, limit=2)
    all_items, _, count = _walk_pages(client, at=FAR_FUTURE, limit=3)
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    assert len(all_items) == 5


def test_last_page_cursor_is_null_on_exact_division(client):
    _world(client)
    _create_rotations(client, 3)
    first = _list_at(client, at=FAR_FUTURE, limit=2).json()
    assert first["next_cursor"] is not None
    second = _list_at(
        client, at=FAR_FUTURE, limit=2, cursor=first["next_cursor"]
    ).json()
    assert len(second["items"]) == 2
    assert second["count"] == 4
    assert second["next_cursor"] is None


def test_default_limit_is_fifty(client):
    _world(client)
    _create_rotations(client, 52)
    first = _list_at(client, at=FAR_FUTURE).json()
    assert len(first["items"]) == 50
    assert first["count"] == 53
    assert first["next_cursor"] is not None
    second = _list_at(
        client, at=FAR_FUTURE, cursor=first["next_cursor"]
    ).json()
    assert len(second["items"]) == 3
    assert second["count"] == 53
    assert second["next_cursor"] is None


def test_limit_boundaries_accepted(client):
    _world(client)
    for value in (1, 100):
        assert _list_at(client, at=FAR_FUTURE, limit=value).status_code == 200


def test_cursor_past_end_returns_empty_page_with_original_count(client, app):
    _world(client)
    _create_rotations(client, 2)
    token = pagination.encode_typed_cursor(
        app.state.authentication_keys_at_cursor_secret,
        pagination.AUTHENTICATION_KEYS_AT_CURSOR,
        {
            "actor_id": "org-1",
            "at": _iso(_parse(FAR_FUTURE)),
            "limit": 50,
            "offset": 99,
        },
    )
    body = _list_at(client, at=FAR_FUTURE, cursor=token).json()
    assert body["items"] == []
    assert body["count"] == 3
    assert body["next_cursor"] is None


def test_issued_cursors_use_the_dedicated_opaque_family_marker(client):
    _world(client)
    _create_rotations(client, 1)
    token = _list_at(client, at=FAR_FUTURE, limit=1).json()["next_cursor"]
    assert token.startswith(
        pagination.AUTHENTICATION_KEYS_AT_CURSOR_VERSION + "."
    )
    assert len(token.split(".")) == 3


# --- Cursor integrity --------------------------------------------------------------


def test_blank_malformed_or_tampered_cursors_are_validation_errors(client, app):
    _world(client)
    _create_rotations(client, 1)
    at_claim = _iso(_parse(FAR_FUTURE))
    good = _list_at(client, at=FAR_FUTURE, limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    foreign = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.AUTHENTICATION_KEYS_AT_CURSOR,
        {"actor_id": "org-1", "at": at_claim, "limit": 1, "offset": 1},
    )
    secret = app.state.authentication_keys_at_cursor_secret
    for token in (
        "",
        "   ",
        "\t",
        "not-a-cursor",
        "ah1.onlytwoparts",
        "ah1.too.many.parts",
        "ah0.x.y",
        "ah2.x.y",
        # Every other family marker, even signed with this family's secret,
        # is a version mismatch rather than a trusted token.
        _hand_cursor(secret, "v1", {"x": 1}),
        _hand_cursor(secret, "au1", {"x": 1}),
        _hand_cursor(secret, "ak1", {"x": 1}),
        _hand_cursor(secret, "ae1", {"x": 1}),
        # Correctly signed, but structurally invalid claim payloads.
        _hand_cursor(secret, "ah1", "not-json"),
        _hand_cursor(
            secret, "ah1",
            {"actor_id": "org-1", "at": at_claim, "limit": 1, "offset": 1,
             "extra": 2},
        ),
        _hand_cursor(secret, "ah1", {"actor_id": "org-1", "limit": 1,
                                     "offset": 1}),
        _hand_cursor(secret, "ah1", {"actor_id": "", "at": at_claim,
                                     "limit": 1, "offset": 1}),
        _hand_cursor(secret, "ah1", {"actor_id": "org-1", "at": "not-a-time",
                                     "limit": 1, "offset": 1}),
        _hand_cursor(secret, "ah1", {"actor_id": "org-1", "at": 7,
                                     "limit": 1, "offset": 1}),
        _hand_cursor(secret, "ah1", {"actor_id": "org-1", "at": at_claim,
                                     "limit": 101, "offset": 1}),
        _hand_cursor(secret, "ah1", {"actor_id": "org-1", "at": at_claim,
                                     "limit": 0, "offset": 1}),
        _hand_cursor(secret, "ah1", {"actor_id": "org-1", "at": at_claim,
                                     "limit": 1, "offset": 0}),
        _hand_cursor(secret, "ah1", {"actor_id": "org-1", "at": at_claim,
                                     "limit": "1", "offset": 1}),
        tampered,
        foreign,
    ):
        resp = _list_at(client, at=FAR_FUTURE, limit=1, cursor=token)
        assert resp.status_code == 422, repr(token)
        assert resp.json()["error"]["code"] == "validation_error", repr(token)
        assert "items" not in resp.json()


def test_cursor_from_other_endpoints_is_rejected(client):
    _world(client)
    _create_rotations(client, 2)
    # Tokens minted by other families (under this process's own secrets) can
    # never resume an at-query page -- including the sibling current-keys
    # and rotation-history families bound to the same subject and limit.
    current_cursor = client.get(
        "/v1/actors/org-1/authentication-keys", params={"limit": 1}
    ).json()["next_cursor"]
    rotation_cursor = client.get(
        "/v1/actors/org-1/authentication-key-rotations", params={"limit": 1}
    ).json()["next_cursor"]
    for token in (current_cursor, rotation_cursor):
        resp = _list_at(client, at=FAR_FUTURE, limit=1, cursor=token)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


def test_at_cursor_is_rejected_by_other_endpoints(client):
    _world(client)
    _create_rotations(client, 1)
    cursor = _list_at(client, at=FAR_FUTURE, limit=1).json()["next_cursor"]
    for path in (
        "/v1/actors/org-1/authentication-keys",
        "/v1/actors/org-1/authentication-key-rotations",
        "/v1/audit-events",
    ):
        resp = client.get(path, params={"limit": 1, "cursor": cursor})
        assert resp.status_code == 422, path
        assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_is_bound_to_its_subject(client, app):
    _world(client)
    _create_rotations(client, 3, actor="org-1")
    _create_rotations(client, 3, actor="org-2")
    cursor = _list_at(client, "org-1", at=FAR_FUTURE, limit=2).json()[
        "next_cursor"
    ]

    # Same cursor on another existing subject is a mismatch, not a page.
    resp = _list_at(client, "org-2", at=FAR_FUTURE, limit=2, cursor=cursor)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    # A correctly-signed cursor naming org-1 cannot be presented on the
    # org-2 path.
    secret = app.state.authentication_keys_at_cursor_secret
    org1_token = pagination.encode_typed_cursor(
        secret,
        pagination.AUTHENTICATION_KEYS_AT_CURSOR,
        {
            "actor_id": "org-1",
            "at": _iso(_parse(FAR_FUTURE)),
            "limit": 2,
            "offset": 2,
        },
    )
    assert _list_at(
        client, "org-2", at=FAR_FUTURE, limit=2, cursor=org1_token
    ).status_code == 422


def test_cursor_is_bound_to_the_resolved_at(client):
    _world(client)
    _create_rotations(client, 3)
    cursor = _list_at(client, at=FAR_FUTURE, limit=2).json()["next_cursor"]
    # The same cursor with any other instant is a mismatch, not a re-paging.
    for other in (EPOCH, "2026-05-04T03:02:01Z"):
        resp = _list_at(client, at=other, limit=2, cursor=cursor)
        assert resp.status_code == 422, other
        assert resp.json()["error"]["code"] == "validation_error"
    # An equivalent spelling of the bound instant resolves to the same
    # claim and resumes normally.
    z_cursor = _list_at(client, at="2999-01-01T00:00:00Z", limit=2).json()[
        "next_cursor"
    ]
    assert z_cursor is not None
    resp = _list_at(client, at="2999-01-01T00:00:00+00:00", limit=2,
                    cursor=z_cursor)
    assert resp.status_code == 200


def test_cursor_is_bound_to_its_effective_limit(client):
    _world(client)
    _create_rotations(client, 3)
    cursor = _list_at(client, at=FAR_FUTURE, limit=2).json()["next_cursor"]
    for params in ({"limit": 3}, {"limit": 1}, {}):
        resp = _list_at(client, at=FAR_FUTURE, cursor=cursor, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_mismatch_is_rejected_before_actor_lookup(client):
    _world(client)
    _create_rotations(client, 1)
    cursor = _list_at(client, "org-1", at=FAR_FUTURE, limit=1).json()[
        "next_cursor"
    ]
    # The subject in the path does not exist; the bound cursor still
    # mismatches and validation wins over the unknown_actor lookup.
    resp = _list_at(client, "ghost", at=FAR_FUTURE, limit=1, cursor=cursor)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_secret_rotation_invalidates_outstanding_cursors(client):
    _world(client)
    _create_rotations(client, 1)
    cursor = _list_at(client, at=FAR_FUTURE, limit=1).json()["next_cursor"]
    client.app.state.authentication_keys_at_cursor_secret = (
        secrets.token_bytes(32)
    )
    resp = _list_at(client, at=FAR_FUTURE, limit=1, cursor=cursor)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    # A fresh cursor under the new secret pages normally.
    fresh = _list_at(client, at=FAR_FUTURE, limit=1).json()
    assert len(fresh["items"]) == 1


# --- Parameter and body validation --------------------------------------------------


def test_missing_or_blank_at_is_a_validation_error(client):
    _world(client)
    for params in ({}, {"at": ""}, {"at": "   "}):
        resp = _list_at(client, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_rfc3339_utc_at_values_are_validation_errors(client):
    _world(client)
    for value in (
        "2026-01-01",
        "2026-01-01 00:00:00Z",
        "2026-01-01T00:00:00",          # no UTC designator
        "2026-01-01T00:00:00+01:00",    # non-UTC offset
        "2026-01-01T00:00:00z",         # lowercase designator
        "2026-01-01T00:00Z",            # missing seconds
        "2026-13-01T00:00:00Z",         # calendar-invalid
        "2026-01-01T25:00:00Z",
        "now",
        "0",
    ):
        resp = _list_at(client, at=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_fractional_seconds_at_is_accepted(client):
    _world(client)
    resp = _list_at(client, at="2026-05-04T03:02:01.123456Z")
    assert resp.status_code == 200
    assert _parse(resp.json()["at"]) == datetime(
        2026, 5, 4, 3, 2, 1, 123456, tzinfo=timezone.utc
    )


def test_illegal_limit_values_are_validation_errors(client):
    _world(client)
    for value in (
        "0", "101", "-1", "1.5", "8.0", "abc", "", "  2", "2  ",
        "+1", "1e2", "０１", "0x1",
    ):
        resp = _list_at(client, at=FAR_FUTURE, limit=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_validation_errors(client):
    _world(client)
    for suffix in (
        "at=2026-01-01T00:00:00Z&at=2026-01-02T00:00:00Z",
        "at=2026-01-01T00:00:00Z&limit=1&limit=2",
        "at=2026-01-01T00:00:00Z&cursor=x&cursor=y",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameters_are_validation_errors(client):
    _world(client)
    for suffix in (
        "active=true",
        "actor_id=org-2",
        "offset=1",
        "At=2026-01-01T00:00:00Z",
        "at=2026-01-01T00:00:00Z&source_type=rotation",
    ):
        resp = client.get(f"{URL}?at=2026-01-01T00:00:00Z&{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_empty_body_is_a_validation_error(client):
    _world(client)
    for body in (b"{}", b" ", b"{not json", b"\x00"):
        resp = client.request(
            "GET", URL, params={"at": FAR_FUTURE}, content=body
        )
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"


def test_validation_failures_are_decided_before_actor_lookup(client):
    # Even with an unknown subject, every malformed request is a 422 rather
    # than the unknown_actor 404: validation precedes the lookup.
    _world(client)
    for suffix in (
        "",
        "at=not-a-time",
        "at=2026-01-01T00:00:00Z&limit=0",
        "at=2026-01-01T00:00:00Z&limit=101",
        "at=2026-01-01T00:00:00Z&limit=abc",
        "at=2026-01-01T00:00:00Z&limit=1&limit=2",
        "at=2026-01-01T00:00:00Z&cursor=garbage",
        "at=2026-01-01T00:00:00Z&bogus=1",
    ):
        resp = client.get(
            "/v1/actors/ghost/authentication-keys/at?" + suffix
        )
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"
    resp = client.request(
        "GET",
        "/v1/actors/ghost/authentication-keys/at?at=2026-01-01T00:00:00Z",
        content=b"{}",
    )
    assert resp.status_code == 422
    # A structurally valid request for the same unknown subject is the 404.
    plain = client.get(
        "/v1/actors/ghost/authentication-keys/at",
        params={"at": "2026-01-01T00:00:00Z"},
    )
    assert plain.status_code == 404


def test_validation_failure_locates_the_query_field(client):
    _world(client)
    resp = _list_at(client, at="nope")
    issue = resp.json()["error"]["details"]["issues"][0]
    assert issue["loc"] == ["query", "at"]
    resp = _list_at(client, at=FAR_FUTURE, limit="nope")
    issue = resp.json()["error"]["details"]["issues"][0]
    assert issue["loc"] == ["query", "limit"]
    resp = _list_at(client, at=FAR_FUTURE, cursor="garbage")
    issue = resp.json()["error"]["details"]["issues"][0]
    assert issue["loc"] == ["query", "cursor"]


# --- Read-only guarantee -------------------------------------------------------------


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

    # Successful reads: paginated through the tail, the empty-collection
    # cases (an instant before any source; a subject without keys).
    _walk_pages(client, at=FAR_FUTURE, limit=2)
    _list_at(client, at=EPOCH)
    _list_at(client, "org-9", at=FAR_FUTURE)

    # Failed reads: missing/malformed at, malformed limit/cursor, repeats,
    # unknown params, a non-empty body, an unknown subject, and a
    # bound-cursor mismatch.
    _list_at(client)
    _list_at(client, at="nope")
    _list_at(client, at=FAR_FUTURE, limit=0)
    _list_at(client, at=FAR_FUTURE, cursor="tampered")
    _list_at(client, "ghost", at=FAR_FUTURE)
    client.get(f"{URL}?at={FAR_FUTURE}&limit=1&limit=2")
    client.get(f"{URL}?at={FAR_FUTURE}&unknown=1")
    client.request("GET", URL, params={"at": FAR_FUTURE}, content=b"{}")
    cursor = _list_at(client, "org-1", at=FAR_FUTURE, limit=1).json()[
        "next_cursor"
    ]
    _list_at(client, "org-2", at=FAR_FUTURE, limit=1, cursor=cursor)

    db_session.expire_all()
    assert counts() == before


# --- Compatibility with the existing routes ------------------------------------------


def test_existing_current_keys_and_rotation_routes_remain_unchanged(client):
    _world(client)
    created = _post_rotation(client, _rotation_body())
    assert created.status_code == 201
    # A repeat submission is still the idempotent 200 with no new record.
    repeat = _post_rotation(client, _rotation_body())
    assert repeat.status_code == 200
    assert repeat.json() == created.json()
    # The rotation history still lists active and retired records together.
    assert _post_retire(
        client, created.json()["id"], seed=SEED_R1
    ).status_code == 200
    history = client.get(
        "/v1/actors/org-1/authentication-key-rotations"
    ).json()
    assert history["count"] == 1
    assert history["items"][0]["active"] is False
    # ...and the current-keys view no longer carries the dead key, while
    # the historical view still shows it before the retirement instant.
    current = client.get("/v1/actors/org-1/authentication-keys").json()
    assert {i["public_key"] for i in current["items"]} == {_key_b64(SEED_A)}
    historical = _list_at(
        client, at=created.json()["created_at"]
    ).json()
    assert {i["public_key"] for i in historical["items"]} == {
        _key_b64(SEED_A),
        _key_b64(SEED_R1),
    }
