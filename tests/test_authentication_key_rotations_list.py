"""Tests for the reviewer read-only authentication-key rotation listing.

Covers ``GET /v1/actors/{actor_id}/authentication-key-rotations``:

* the success body is exactly ``{"items", "count", "next_cursor"}`` and
  each item exposes exactly the existing rotation public fields
  (``id``, ``actor_id``, ``new_public_key``, ``active``, ``created_at``,
  ``retired_at``) -- never the internal seq, a private key, a raw
  signature, or an authentication header -- with UTC timestamps;
* records follow stable creation order, are scoped to the one named
  existing subject, and include active and retired records alike;
* ``limit`` is a strict 1..100 decimal integer defaulting to 50, the
  opaque HMAC cursor binds the subject and the effective limit, pages
  concatenate without gaps or duplicates, the final cursor is null, and
  a (validly signed) cursor at or past the end returns empty items with
  the unchanged total count;
* blank/illegal/repeated/undeclared parameters and blank/malformed/
  tampered/foreign-family/interface- or subject-/limit-mismatching
  cursors are all ``422 validation_error`` and are rejected *before* the
  subject lookup, while a structurally valid request for an unknown
  subject is the existing ``404 unknown_actor``;
* a subject without rotations is an empty collection;
* the route is strictly read-only (no rotation, resource, or audit
  event is created), requires no authentication headers, and the
  existing rotation creation/retirement routes stay compatible.

All fixtures are deterministic and offline (the stdlib test signer
produces the Ed25519 signatures); only fixed seed-derived keys are used.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance import pagination
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import AuditEvent, AuthenticationKeyRotation
from tests.helpers import create_actor
from tests.helpers import SEED_A, SEED_B
from tests.test_authentication_key_rotations import (
    _post_retire,
    _post_rotation,
    _rotation_body,
    _world,
    SEED_R1,
    SEED_R2,
)

ROTATION_KEYS = {
    "id",
    "actor_id",
    "new_public_key",
    "active",
    "created_at",
    "retired_at",
}


def _seed(index: int) -> bytes:
    # Mirrors SEED_R1/SEED_R2 for indices 1 and 2, then extends the series;
    # sliced to the 32 bytes Ed25519 seeds use.
    return f"test-ed25519-rotate-r{index}-000000000000"[:32].encode()


def _path(actor_id: str) -> str:
    return f"/v1/actors/{actor_id}/authentication-key-rotations"


def _list(client, actor_id="org-1", **params):
    return client.get(_path(actor_id), params=params)


def _rotate(client, index: int, *, actor_id="org-1", signer_seed=None):
    """Introduce the deterministic key ``index`` for one subject.

    For org-1 the chain starts with the bootstrap attestation key (SEED_A);
    each subsequent rotation is signed by the subject's previous rotated
    key. org-2 signs with its bootstrap key (SEED_B) unless overridden.
    """
    body = _rotation_body(actor_id=actor_id, seed=_seed(index))
    kwargs = {}
    if actor_id != "org-1":
        kwargs["actor"] = actor_id
        kwargs["seed"] = signer_seed or SEED_B
    elif signer_seed is not None:
        kwargs["seed"] = signer_seed
    elif index == 1:
        kwargs["seed"] = SEED_A
    else:
        kwargs["seed"] = _seed(index - 1)
    resp = _post_rotation(client, body, **kwargs)
    assert resp.status_code in (200, 201), resp.text
    return resp.json()


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


# --- Response shape, fields, and ordering -------------------------------------


def test_existing_subject_without_rotations_is_an_empty_collection(client):
    create_actor(client)
    assert _list(client).json() == {
        "items": [],
        "count": 0,
        "next_cursor": None,
    }


def test_unknown_subject_is_404_unknown_actor(client):
    create_actor(client)
    resp = _list(client, "org-ghost")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "unknown_actor"
    assert body["error"]["details"]["actor_id"] == "org-ghost"


def test_response_shape_and_item_fields(client):
    _world(client)
    active = _rotate(client, 1)
    second = _rotate(client, 2)
    retired = _post_retire(client, second["id"], seed=SEED_R2)
    assert retired.status_code == 200, retired.text

    resp = _list(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"items", "count", "next_cursor"}
    assert body["count"] == 2
    assert body["next_cursor"] is None
    assert len(body["items"]) == 2
    for item in body["items"]:
        assert set(item) == ROTATION_KEYS
        assert item["id"].startswith("akr_")
        assert len(item["id"]) == len("akr_") + 64
        assert item["actor_id"] == "org-1"
        # The internal insertion surrogate and any credential material are
        # absent from the wire view.
        assert "seq" not in item
        assert "private_key" not in item
        assert "signature" not in item
        assert isinstance(item["new_public_key"], str)
        assert isinstance(item["active"], bool)


def test_items_are_the_existing_public_rotation_views(client):
    _world(client)
    first = _rotate(client, 1)
    second = _rotate(client, 2)
    assert _post_retire(client, second["id"], seed=SEED_R2).status_code == 200

    items = _list(client).json()["items"]
    assert items[0] == first
    assert items[0]["active"] is True
    assert items[0]["retired_at"] is None
    assert items[1]["id"] == second["id"]
    assert items[1]["new_public_key"] == second["new_public_key"]
    assert items[1]["active"] is False
    assert items[1]["retired_at"] is not None
    # Timestamps are UTC RFC 3339.
    for item in items:
        created = datetime.fromisoformat(item["created_at"])
        assert created.utcoffset() == timedelta(0)
        if item["retired_at"] is not None:
            retired_at = datetime.fromisoformat(item["retired_at"])
            assert retired_at.utcoffset() == timedelta(0)


def test_records_follow_stable_creation_order(client):
    _world(client)
    created = [_rotate(client, i) for i in range(1, 7)]
    # Retire a middle record: order is unchanged.
    assert _post_retire(client, created[2]["id"], seed=_seed(3)).status_code == 200
    items = _list(client).json()["items"]
    assert [item["id"] for item in items] == [r["id"] for r in created]


def test_repeated_submission_does_not_add_a_listing_position(client):
    _world(client)
    first = _rotate(client, 1)
    again = _rotate(client, 1)
    assert again["id"] == first["id"]
    body = _list(client).json()
    assert body["count"] == 1
    assert [item["id"] for item in body["items"]] == [first["id"]]


def test_listing_is_scoped_to_the_named_subject(client):
    _world(client)
    one = _rotate(client, 1)
    # org-2 introduces the same key index bytes under its own identity.
    other = _rotate(client, 1, actor_id="org-2")
    assert one["id"] != other["id"]

    org1 = _list(client, "org-1").json()
    org2 = _list(client, "org-2").json()
    assert [i["id"] for i in org1["items"]] == [one["id"]]
    assert [i["id"] for i in org2["items"]] == [other["id"]]
    assert all(i["actor_id"] == "org-1" for i in org1["items"])
    assert all(i["actor_id"] == "org-2" for i in org2["items"])


def test_listing_requires_no_authentication_headers(client):
    _world(client)
    _rotate(client, 1)
    # A bare GET with no X-PA/X-PT/X-PS credentials succeeds for reviewers.
    resp = client.get(_path("org-1"))
    assert resp.status_code == 200, resp.text
    assert resp.json()["count"] == 1


# --- Pagination ----------------------------------------------------------------


def test_pagination_concatenates_without_gaps_or_duplicates(client):
    _world(client)
    created = [_rotate(client, i) for i in range(1, 6)]
    all_items, pages, count = _walk_pages(client, limit=2)
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    ids = [item["id"] for item in all_items]
    assert len(ids) == len(set(ids)) == 5
    assert ids == [r["id"] for r in created]


def test_count_is_the_subject_total_on_every_page(client):
    _world(client)
    created = [_rotate(client, i) for i in range(1, 6)]
    all_items, pages, count = _walk_pages(client, limit=3)
    assert count == 5
    assert [len(page) for page in pages] == [3, 2]
    assert [i["id"] for i in all_items] == [r["id"] for r in created]


def test_last_page_cursor_is_null_on_exact_division(client):
    _world(client)
    [_rotate(client, i) for i in range(1, 5)]
    first = _list(client, limit=2).json()
    assert first["next_cursor"] is not None
    second = _list(client, limit=2, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 2
    assert second["count"] == 4
    assert second["next_cursor"] is None


def test_default_limit_is_fifty(client):
    _world(client)
    created = [_rotate(client, i) for i in range(1, 56)]
    first = _list(client).json()
    assert len(first["items"]) == 50
    assert first["count"] == 55
    assert first["next_cursor"] is not None
    second = _list(client, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 5
    assert second["count"] == 55
    assert second["next_cursor"] is None
    expected = [r["id"] for r in created]
    assert [i["id"] for i in first["items"]] == expected[:50]
    assert [i["id"] for i in second["items"]] == expected[50:]


def test_limit_boundaries_accepted(client):
    _world(client)
    [_rotate(client, i) for i in range(1, 3)]
    for value in (1, 100):
        assert _list(client, limit=value).status_code == 200


def test_reusing_a_cursor_replays_the_same_page(client):
    _world(client)
    [_rotate(client, i) for i in range(1, 6)]
    cursor = _list(client, limit=2).json()["next_cursor"]
    replay_one = _list(client, limit=2, cursor=cursor).json()
    replay_two = _list(client, limit=2, cursor=cursor).json()
    assert replay_one == replay_two
    assert replay_one["count"] == 5


def test_cursor_at_or_past_end_returns_empty_items_with_total_count(
    client, app
):
    _world(client)
    created = [_rotate(client, i) for i in range(1, 4)]
    for offset in (3, 99):
        token = pagination.encode_typed_cursor(
            app.state.authentication_key_rotations_cursor_secret,
            pagination.AUTHENTICATION_KEY_ROTATIONS_CURSOR,
            {"actor_id": "org-1", "limit": 50, "offset": offset},
        )
        body = _list(client, cursor=token).json()
        assert body["items"] == []
        assert body["count"] == len(created)
        assert body["next_cursor"] is None


def test_pagination_is_deterministic_across_walks(client):
    _world(client)
    [_rotate(client, i) for i in range(1, 8)]

    def cursors():
        seen = []
        cursor = None
        for _ in range(100):
            params = {"limit": 2}
            if cursor is not None:
                params["cursor"] = cursor
            body = _list(client, **params).json()
            cursor = body["next_cursor"]
            if cursor is None:
                break
            seen.append(cursor)
        return seen

    assert cursors() == cursors()


# --- Cursor integrity ----------------------------------------------------------


def test_tampered_or_malformed_cursors_are_validation_errors(client):
    _world(client)
    [_rotate(client, i) for i in range(1, 3)]
    good = _list(client, limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    foreign = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.AUTHENTICATION_KEY_ROTATIONS_CURSOR,
        {"actor_id": "org-1", "limit": 1, "offset": 1},
    )
    for token in (
        "",
        "   ",
        "not-a-cursor",
        "ak1.onlytwoparts",
        "ak1.too.many.parts",
        "ak0.x.y",
        "ak2.x.y",
        "v1.x.y",
        "cl1.x.y",
        "ce1.x.y",
        "ae1.x.y",
        tampered,
        foreign,
    ):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422, repr(token)
        assert resp.json()["error"]["code"] == "validation_error"
        assert "items" not in resp.json()


def test_other_family_cursor_signed_with_this_secret_is_still_rejected(
    client, app
):
    # A valid HMAC does not help a cursor from another interface: the family
    # marker/claim set must also match.
    _world(client)
    [_rotate(client, i) for i in range(1, 3)]
    secret = app.state.authentication_key_rotations_cursor_secret
    claims_cursor = pagination.encode_typed_cursor(
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
    )
    evidence_cursor = pagination.encode_typed_cursor(
        secret,
        pagination.CONTENT_EVIDENCE_CURSOR,
        {
            "content_id": "cnt_x",
            "evidence_type": None,
            "media_type": None,
            "limit": 1,
            "offset": 1,
        },
    )
    audit_cursor = pagination.encode_typed_cursor(
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
    )
    for token in (claims_cursor, evidence_cursor, audit_cursor):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


def test_rotations_cursor_is_rejected_by_other_endpoints(client, app):
    _world(client)
    [_rotate(client, i) for i in range(1, 3)]
    token = _list(client, limit=1).json()["next_cursor"]

    # The claims search uses its own marker and claim set.
    resp = client.get("/v1/claims", params={"limit": 1, "cursor": token})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    # As does the audit-event search.
    resp = client.get("/v1/audit-events", params={"limit": 1, "cursor": token})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_is_bound_to_the_subject(client, app):
    _world(client)
    [_rotate(client, i) for i in range(1, 3)]
    _rotate(client, 1, actor_id="org-2")
    cursor = _list(client, "org-1", limit=1).json()["next_cursor"]

    # Resuming under a different existing subject mismatches.
    resp = _list(client, "org-2", limit=1, cursor=cursor)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    # A correctly signed cursor that merely names a different subject (or a
    # subject that does not exist) is likewise rejected rather than trusted.
    for bound_actor in ("org-2", "org-ghost"):
        foreign_actor = pagination.encode_typed_cursor(
            app.state.authentication_key_rotations_cursor_secret,
            pagination.AUTHENTICATION_KEY_ROTATIONS_CURSOR,
            {"actor_id": bound_actor, "limit": 1, "offset": 1},
        )
        resp = _list(client, "org-1", limit=1, cursor=foreign_actor)
        assert resp.status_code == 422, bound_actor
        assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_is_bound_to_the_effective_limit(client):
    _world(client)
    [_rotate(client, i) for i in range(1, 6)]
    cursor = _list(client, limit=2).json()["next_cursor"]
    for params in ({"limit": 3}, {"limit": 1}, {}):
        # No ``limit`` in the resume request means the default 50, which
        # still mismatches the cursor's bound limit of 2.
        resp = _list(client, cursor=cursor, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_secret_rotation_invalidates_outstanding_cursors(
    client, tmp_db_url, file_client
):
    # An in-process secret swap refuses the outstanding cursor immediately.
    _world(client)
    [_rotate(client, i) for i in range(1, 3)]
    cursor = _list(client, limit=1).json()["next_cursor"]
    client.app.state.authentication_key_rotations_cursor_secret = (
        secrets.token_bytes(32)
    )
    assert _list(client, limit=1, cursor=cursor).status_code == 422

    # After a real restart the stored rotations survive even though a
    # cursor minted by the old process is rejected (a restart rotates the
    # per-process HMAC secret).
    _world(file_client)
    created = [_rotate(file_client, i) for i in range(1, 3)]
    stale = _list(file_client, limit=1).json()["next_cursor"]
    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as second_client:
        resp = _list(second_client, limit=1, cursor=stale)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"
        fresh = _list(second_client).json()
        assert [i["id"] for i in fresh["items"]] == [
            r["id"] for r in created
        ]


# --- Parameter validation ------------------------------------------------------


def test_illegal_limit_values_are_validation_errors(client):
    _world(client)
    [_rotate(client, i) for i in range(1, 3)]
    for value in ("0", "101", "-1", "1.5", "abc", "8.0", "  2", "+1", ""):
        resp = _list(client, limit=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_validation_errors(client):
    _world(client)
    [_rotate(client, i) for i in range(1, 3)]
    for suffix in ("limit=1&limit=2", "cursor=x&cursor=y"):
        resp = client.get(f"{_path('org-1')}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameters_are_validation_errors(client):
    _world(client)
    for suffix in (
        "active=true",
        "offset=1",
        "actor_id=org-2",
        "public_key=x",
        "retired=true",
        "limit=1&foo=2",
        "CURSOR=x",
    ):
        resp = client.get(f"{_path('org-1')}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_blank_cursor_is_a_validation_error(client):
    _world(client)
    for value in ("", "   ", "\t"):
        resp = _list(client, cursor=value)
        assert resp.status_code == 422, repr(value)
        assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_cursor_with_otherwise_valid_params_is_422(client):
    _world(client)
    resp = _list(client, limit=10, cursor="garbage")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_validation_failure_locates_the_query_field(client):
    _world(client)
    resp = _list(client, limit="nope")
    assert resp.status_code == 422
    issue = resp.json()["error"]["details"]["issues"][0]
    assert issue["loc"] == ["query", "limit"]


# --- 422 before the subject lookup ---------------------------------------------


def test_malformed_request_is_422_even_when_subject_is_missing(client):
    # No actors exist at all.
    for suffix in (
        "limit=0",
        "limit=abc",
        "limit=1&limit=2",
        "unknown=1",
        "cursor=garbage",
        "cursor=",
        "cursor=ak1.x.y",
    ):
        resp = client.get(f"{_path('org-ghost')}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_mismatch_is_422_before_the_subject_lookup(client, app):
    _world(client)
    [_rotate(client, i) for i in range(1, 3)]
    cursor = _list(client, "org-1", limit=1).json()["next_cursor"]
    # A structurally valid cursor bound to org-1/limit=1 carried on a path
    # for a missing subject must be the cursor mismatch 422, not a 404.
    resp = _list(client, "org-ghost", limit=1, cursor=cursor)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    # And a cursor minted for another existing subject is the same 422 when
    # presented against the missing subject (decode succeeds, subject claim
    # mismatches before any actor row is read).
    _rotate(client, 1, actor_id="org-2")
    other_cursor = _list(client, "org-2", limit=1).json()["next_cursor"]
    resp = _list(client, "org-ghost", limit=1, cursor=other_cursor)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only guarantee -------------------------------------------------------


def test_reads_and_failures_write_nothing(client, db_session):
    _world(client)
    created = [_rotate(client, i) for i in range(1, 5)]

    def counts():
        return (
            db_session.scalar(
                select(func.count()).select_from(AuthenticationKeyRotation)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    rotations_before, events_before = counts()
    assert rotations_before == len(created)

    # Successful reads: first page, pages to exhaustion, default page.
    cursor = None
    for _ in range(6):
        params = {"limit": 1}
        if cursor is not None:
            params["cursor"] = cursor
        resp = _list(client, **params)
        assert resp.status_code == 200, resp.text
        cursor = resp.json()["next_cursor"]
        if cursor is None:
            break
    _list(client)
    _list(client, "org-2")  # existing subject, no rotations
    _list(client, limit=100)

    # Failed reads must not write anything either.
    _list(client, limit=0)
    _list(client, limit=101)
    _list(client, limit="nope")
    _list(client, cursor="tampered")
    _list(client, cursor="ak1.x.y")
    client.get(f"{_path('org-1')}?limit=1&limit=2")
    client.get(f"{_path('org-1')}?unknown=1")
    _list(client, "org-ghost")
    _list(client, "org-ghost", limit=1)

    db_session.expire_all()
    assert counts() == (rotations_before, events_before)


# --- Compatibility with existing rotation routes -------------------------------


def test_existing_rotation_routes_remain_compatible(client):
    _world(client)
    created = _rotate(client, 1)
    # Introduce the second rotation while r1 is still active (it signs).
    second = _rotate(client, 2)

    # The create route still serves its exact single-resource view.
    assert set(created) == ROTATION_KEYS
    # The retire route still flips the record, and the listing reflects it.
    retired = _post_retire(client, created["id"], seed=SEED_R1)
    assert retired.status_code == 200
    listed = _list(client).json()["items"][0]
    assert listed["id"] == created["id"]
    assert listed["active"] is False
    assert listed["retired_at"] is not None
    assert listed["new_public_key"] == created["new_public_key"]

    # Nothing about the write-side identity or audit behavior changes: the
    # listing carries both records in creation order.
    body = _list(client).json()
    assert [i["id"] for i in body["items"]] == [created["id"], second["id"]]
    assert body["count"] == 2
