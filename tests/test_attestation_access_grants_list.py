"""Offline deterministic tests for the signer-only protected grant listing.

Covers ``GET /v1/attestations/{attestation_id}/access-grants``:

* the ``X-PA``/``X-PT``/``X-PS`` signed-header contract is exactly the one
  shared with the grant-write and single-proof protected-read routes
  (uppercase GET, path without query string, zero-byte body digest);
* only the proof's ``signer_actor_id`` may read the list -- a grantee (who
  may read the proof itself), any other authenticated actor, a missing
  target, and a missing/unauthenticated caller all get one opaque
  ``404 not_found``; malformed timestamps/signatures stay ``422``;
* the success body is strictly ``{"items", "count", "next_cursor"}`` and
  each item carries exactly ``id``, ``attestation_id``,
  ``grantee_actor_id``, and a UTC ``created_at``, in stable creation order;
* ``limit`` is a pure decimal integer in 1..100 (default 50); cursors are
  opaque, HMAC-signed (``ag1`` family), and bound to the proof, the caller,
  and the effective limit -- pages concatenate without gaps or duplicates,
  the final cursor is null, and a cursor at/past the tail returns an empty
  page with the original count;
* undeclared, repeated, blank, or illegal parameters, and a malformed,
  tampered, foreign-family, wrong-claim-set, or proof/caller/limit
  mismatching cursor are all ``422 validation_error``, decided before the
  proof lookup, so they never render as the opaque 404;
* a proof with no grants is an empty collection; revoked grants remain
  listed (grant rows are immutable) while revocation data is never echoed;
* the route is strictly read-only and compatible with the existing
  grant/grant-revocation/protected-read routes.

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures).
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from sqlalchemy import select

from provenance import pagination
from provenance.models import (
    Attestation,
    AttestationAccessGrant,
    AuditEvent,
)
from tests.helpers import SEED_A, SEED_B, create_actor
from tests.test_attestation_access_grants import (
    SEED_C,
    _grant_body,
    _make_attestation,
    _make_claim,
    _post_grant,
    _signed_headers,
    _world,
)

PAGE_KEYS = {"items", "count", "next_cursor"}
ITEM_KEYS = {"id", "attestation_id", "grantee_actor_id", "created_at"}


# --- Helpers ------------------------------------------------------------------


def _path(attestation_id: str) -> str:
    return f"/v1/attestations/{attestation_id}/access-grants"


def _list(
    client,
    attestation_id,
    *,
    actor="org-1",
    seed=SEED_A,
    path=None,
    params=None,
    raw_suffix=None,
    sign=None,
):
    """GET the grant list, signing the path without the query string.

    ``params`` carries real query parameters; ``sign`` carries overrides for
    the signed-header helper (timestamps/signature bindings), which never
    appear on the query string.
    """
    path = path or _path(attestation_id)
    headers = _signed_headers(
        "GET", path, b"", actor=actor, seed=seed, **(sign or {})
    )
    if raw_suffix is not None:
        return client.get(f"{path}?{raw_suffix}", headers=headers)
    return client.get(path, params=params, headers=headers)


def _unsigned(client, attestation_id, *, raw_suffix=None):
    path = _path(attestation_id)
    if raw_suffix:
        return client.get(f"{path}?{raw_suffix}")
    return client.get(path)


def _grant_many(client, attestation_id, count, *, start=0):
    """Create ``count`` grants to fresh, distinct grantee actors, in order."""
    created = []
    for i in range(start, start + count):
        grantee = f"grantee-{i:03d}"
        create_actor(
            client, actor_id=grantee, name=f"Grantee {i}", type="organization"
        )
        resp = _post_grant(client, _grant_body(attestation_id, grantee))
        assert resp.status_code == 201, resp.text
        created.append(resp.json())
    return created


def _walk_pages(client, attestation_id, *, initial_params=None):
    """Follow next_cursor to the end; return (all_items, pages, counts)."""
    pages = []
    counts = []
    all_items = []
    seen_cursors = set()
    cursor = None
    while True:
        request_params = dict(initial_params or {})
        if cursor is not None:
            request_params["cursor"] = cursor
        resp = _list(client, attestation_id, params=request_params)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert set(body) == PAGE_KEYS
        pages.append(body["items"])
        counts.append(body["count"])
        all_items.extend(body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            break
        assert cursor not in seen_cursors
        seen_cursors.add(cursor)
    return all_items, pages, counts


def _second_org1_attestation(client, digest_seed=b"second-proof"):
    """A second proof signed by org-1 (same current key, distinct claim)."""
    claim = _make_claim(
        client, "org-1", digest=hashlib.sha256(digest_seed).hexdigest()
    )
    return _make_attestation(client, "org-1", SEED_A, claim["id"])


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


# --- Success body, fields, order ----------------------------------------------


def test_signer_lists_grants_with_exact_envelope_and_item_fields(client):
    attestation = _world(client)
    _grant_many(client, attestation["id"], 3)

    resp = _list(client, attestation["id"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == PAGE_KEYS
    assert body["count"] == 3
    assert body["next_cursor"] is None
    assert len(body["items"]) == 3
    for item in body["items"]:
        assert set(item) == ITEM_KEYS
        assert item["id"].startswith("aag_")
        assert item["attestation_id"] == attestation["id"]
        created_at = datetime.fromisoformat(item["created_at"])
        assert created_at.utcoffset().total_seconds() == 0
        assert item["created_at"].endswith(("Z", "+00:00"))


def test_items_follow_stable_creation_order(client):
    attestation = _world(client)
    created = _grant_many(client, attestation["id"], 6)

    body = _list(client, attestation["id"], params={"limit": 100}).json()
    assert [item["id"] for item in body["items"]] == [g["id"] for g in created]
    assert [item["grantee_actor_id"] for item in body["items"]] == [
        g["grantee_actor_id"] for g in created
    ]
    # Every item is the exact public view returned at creation.
    assert body["items"] == created


def test_proof_without_grants_is_an_empty_collection(client):
    attestation = _world(client)
    resp = _list(client, attestation["id"])
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_grants_to_other_proofs_never_appear(client):
    first = _world(client)
    second = _second_org1_attestation(client)
    _grant_many(client, first["id"], 2)
    _grant_many(client, second["id"], 3, start=10)

    first_body = _list(client, first["id"], params={"limit": 100}).json()
    second_body = _list(client, second["id"], params={"limit": 100}).json()
    assert first_body["count"] == 2
    assert second_body["count"] == 3
    assert {i["attestation_id"] for i in first_body["items"]} == {first["id"]}
    assert {i["attestation_id"] for i in second_body["items"]} == {second["id"]}
    assert {i["id"] for i in first_body["items"]}.isdisjoint(
        {i["id"] for i in second_body["items"]}
    )


# --- Pagination ---------------------------------------------------------------


def test_pagination_concatenates_without_gaps_or_duplicates(client):
    attestation = _world(client)
    created = _grant_many(client, attestation["id"], 5)

    all_items, pages, counts = _walk_pages(
        client, attestation["id"], initial_params={"limit": 2}
    )
    assert set(counts) == {5}
    assert [len(page) for page in pages] == [2, 2, 1]
    assert [item["id"] for item in all_items] == [g["id"] for g in created]
    # No duplicates across pages.
    assert len({item["id"] for item in all_items}) == 5


def test_count_is_the_total_on_every_page(client):
    attestation = _world(client)
    _grant_many(client, attestation["id"], 5)
    _, pages, counts = _walk_pages(
        client, attestation["id"], initial_params={"limit": 2}
    )
    assert counts == [5, 5, 5]
    assert [len(page) for page in pages] == [2, 2, 1]


def test_default_limit_is_50(client):
    attestation = _world(client)
    _grant_many(client, attestation["id"], 55)

    first = _list(client, attestation["id"])
    assert first.status_code == 200, first.text
    first_body = first.json()
    assert len(first_body["items"]) == 50
    assert first_body["count"] == 55
    assert first_body["next_cursor"] is not None

    second = _list(
        client,
        attestation["id"],
        params={"cursor": first_body["next_cursor"]},
    )
    assert second.status_code == 200, second.text
    second_body = second.json()
    assert len(second_body["items"]) == 5
    assert second_body["count"] == 55
    assert second_body["next_cursor"] is None


def test_limit_boundaries_1_and_100_are_accepted(client):
    attestation = _world(client)
    _grant_many(client, attestation["id"], 2)
    for value in (1, 100):
        resp = _list(client, attestation["id"], params={"limit": value})
        assert resp.status_code == 200, value


def test_last_page_cursor_is_null_on_exact_division(client):
    attestation = _world(client)
    _grant_many(client, attestation["id"], 4)
    first = _list(client, attestation["id"], params={"limit": 2}).json()
    assert first["next_cursor"] is not None
    second = _list(
        client,
        attestation["id"],
        params={"limit": 2, "cursor": first["next_cursor"]},
    ).json()
    assert len(second["items"]) == 2
    assert second["count"] == 4
    assert second["next_cursor"] is None


def test_reusing_a_cursor_replays_the_same_page(client):
    attestation = _world(client)
    _grant_many(client, attestation["id"], 4)
    cursor = _list(client, attestation["id"], params={"limit": 2}).json()[
        "next_cursor"
    ]
    one = _list(
        client, attestation["id"], params={"limit": 2, "cursor": cursor}
    ).json()
    two = _list(
        client, attestation["id"], params={"limit": 2, "cursor": cursor}
    ).json()
    assert one == two


def test_cursor_at_or_past_the_tail_returns_empty_page_with_count(client, app):
    attestation = _world(client)
    _grant_many(client, attestation["id"], 4)

    def token_at(offset):
        return pagination.encode_typed_cursor(
            app.state.attestation_access_grants_cursor_secret,
            pagination.ATTESTATION_ACCESS_GRANTS_CURSOR,
            {
                "attestation_id": attestation["id"],
                "caller_actor_id": "org-1",
                "limit": 2,
                "offset": offset,
            },
        )

    # A validly signed offset==total token behaves like the natural end.
    body = _list(
        client, attestation["id"], params={"limit": 2, "cursor": token_at(4)}
    ).json()
    assert body["items"] == []
    assert body["count"] == 4
    assert body["next_cursor"] is None
    # Past the tail is the same empty page with the original count.
    body = _list(
        client, attestation["id"], params={"limit": 2, "cursor": token_at(40)}
    ).json()
    assert body["items"] == []
    assert body["count"] == 4
    assert body["next_cursor"] is None


def test_issued_cursors_use_the_dedicated_opaque_family_marker(client):
    attestation = _world(client)
    _grant_many(client, attestation["id"], 2)
    token = _list(client, attestation["id"], params={"limit": 1}).json()[
        "next_cursor"
    ]
    assert token.startswith(
        pagination.ATTESTATION_ACCESS_GRANTS_CURSOR_VERSION + "."
    )
    assert len(token.split(".")) == 3


# --- limit / parameter validation ---------------------------------------------


def test_illegal_limit_values_are_422(client):
    attestation = _world(client)
    for value in (
        "0", "101", "-1", "1.5", "8.0", "abc", "", "  2", "2  ",
        "+1", "1e2", "０１", "0x1",
    ):
        resp = _list(client, attestation["id"], params={"limit": value})
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error", value


def test_blank_limit_is_422_even_with_a_cursor(client):
    attestation = _world(client)
    _grant_many(client, attestation["id"], 2)
    cursor = _list(client, attestation["id"], params={"limit": 1}).json()[
        "next_cursor"
    ]
    resp = _list(
        client,
        attestation["id"],
        raw_suffix="limit=&cursor=" + cursor,
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_422(client):
    attestation = _world(client)
    for suffix in ("limit=1&limit=2", "cursor=x&cursor=y"):
        resp = _list(client, attestation["id"], raw_suffix=suffix)
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameters_are_422(client):
    attestation = _world(client)
    for suffix in (
        "offset=1",
        "page=1",
        "Limit=1",
        "limit=1&include_revoked=1",
        "grantee_actor_id=org-2",
    ):
        resp = _list(client, attestation["id"], raw_suffix=suffix)
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_blank_cursor_is_422(client):
    attestation = _world(client)
    resp = _list(client, attestation["id"], params={"cursor": ""})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_validation_error_locates_the_query_field(client):
    attestation = _world(client)
    resp = _list(client, attestation["id"], params={"limit": "nope"})
    issue = resp.json()["error"]["details"]["issues"][0]
    assert issue["loc"] == ["query", "limit"]
    resp = _list(client, attestation["id"], params={"cursor": "garbage"})
    issue = resp.json()["error"]["details"]["issues"][0]
    assert issue["loc"] == ["query", "cursor"]


# --- cursor integrity ----------------------------------------------------------


def test_blank_malformed_or_tampered_cursors_are_422(client, app):
    attestation = _world(client)
    _grant_many(client, attestation["id"], 2)
    good = _list(client, attestation["id"], params={"limit": 1}).json()[
        "next_cursor"
    ]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    foreign_secret = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.ATTESTATION_ACCESS_GRANTS_CURSOR,
        {
            "attestation_id": attestation["id"],
            "caller_actor_id": "org-1",
            "limit": 1,
            "offset": 1,
        },
    )
    for token in (
        "garbage",
        "ag1",
        "ag1.abc",
        "ag1.abc.def.ghi",
        "xx1." + good.split(".")[1] + "." + good.split(".")[2],
        tampered,
        foreign_secret,
    ):
        resp = _list(
            client, attestation["id"], params={"limit": 1, "cursor": token}
        )
        assert resp.status_code == 422, repr(token)
        assert resp.json()["error"]["code"] == "validation_error", repr(token)
        assert "items" not in resp.json()


def test_foreign_family_and_wrong_claim_set_cursors_are_422(client, app):
    attestation = _world(client)
    _grant_many(client, attestation["id"], 2)
    secret = app.state.attestation_access_grants_cursor_secret
    # A cursor minted by another paginated endpoint family never resumes
    # this one, even when it carries a plausible id.
    foreign_family = pagination.encode_typed_cursor(
        secret,
        pagination.AUTHENTICATION_KEY_ROTATIONS_CURSOR,
        {"actor_id": "org-1", "limit": 1, "offset": 1},
    )
    # Right marker/secret but the wrong claim set (missing caller binding).
    payload = pagination._b64encode(  # noqa: SLF001
        json.dumps(
            {
                "attestation_id": attestation["id"],
                "limit": 1,
                "offset": 1,
            },
            separators=(",", ":"),
        ).encode("utf-8")
    )
    wrong_claims = (
        "ag1."
        + payload
        + "."
        + pagination._b64encode(  # noqa: SLF001
            pagination._sign(secret, "ag1", payload)  # noqa: SLF001
        )
    )
    # Structurally valid claims (limit 50 within range) presented with
    # limit 1: a limit-binding mismatch, not a re-paging.
    bad_limit = pagination.encode_typed_cursor(
        secret,
        pagination.ATTESTATION_ACCESS_GRANTS_CURSOR,
        {
            "attestation_id": attestation["id"],
            "caller_actor_id": "org-1",
            "limit": 50,
            "offset": 1,
        },
    )
    for token in (foreign_family, wrong_claims, bad_limit):
        resp = _list(
            client, attestation["id"], params={"limit": 1, "cursor": token}
        )
        assert resp.status_code == 422, repr(token)
        assert resp.json()["error"]["code"] == "validation_error"


def test_ag1_cursor_is_rejected_by_other_paginated_endpoints(client):
    attestation = _world(client)
    _grant_many(client, attestation["id"], 2)
    cursor = _list(client, attestation["id"], params={"limit": 1}).json()[
        "next_cursor"
    ]
    for path in (
        "/v1/claims",
        "/v1/evidence-bundles",
        "/v1/audit-events",
        "/v1/actors/org-1/authentication-key-rotations",
    ):
        resp = client.get(path, params={"limit": 1, "cursor": cursor})
        assert resp.status_code == 422, path


def test_cursor_is_bound_to_its_proof(client):
    first = _world(client)
    second = _second_org1_attestation(client)
    _grant_many(client, first["id"], 2)
    cursor = _list(client, first["id"], params={"limit": 1}).json()[
        "next_cursor"
    ]
    # Same signer, another existing proof: the bound proof mismatches and
    # validation wins over any lookup.
    resp = _list(
        client, second["id"], params={"limit": 1, "cursor": cursor}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_is_bound_to_its_caller(client):
    attestation = _world(client)
    _post_grant(client, _grant_body(attestation["id"], "org-2"))
    _grant_many(client, attestation["id"], 2)
    cursor = _list(client, attestation["id"], params={"limit": 1}).json()[
        "next_cursor"
    ]
    # A grantee (org-2) presenting the signer's cursor is a caller mismatch
    # -- a 422, even though org-2 authenticates fine and would otherwise
    # merely get the opaque 404.
    resp = _list(
        client,
        attestation["id"],
        actor="org-2",
        seed=SEED_B,
        params={"limit": 1, "cursor": cursor},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_is_bound_to_its_effective_limit(client):
    attestation = _world(client)
    _grant_many(client, attestation["id"], 4)
    cursor = _list(client, attestation["id"], params={"limit": 2}).json()[
        "next_cursor"
    ]
    for request_params in ({"limit": 3}, {"limit": 1}, {"cursor": cursor}):
        if "cursor" not in request_params:
            request_params = {**request_params, "cursor": cursor}
        resp = _list(client, attestation["id"], params=request_params)
        assert resp.status_code == 422, request_params
        assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_mismatch_is_rejected_before_proof_lookup(client, app):
    attestation = _world(client)
    _grant_many(client, attestation["id"], 2)
    cursor = _list(client, attestation["id"], params={"limit": 1}).json()[
        "next_cursor"
    ]
    # The path names a proof that does not exist; the bound cursor still
    # mismatches and validation wins over the opaque not_found.
    resp = _list(
        client, "att_ghost", params={"limit": 1, "cursor": cursor}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    # A structurally valid cursor naming the ghost proof and this signer
    # passes validation, then collapses into the opaque 404.
    ghost_token = pagination.encode_typed_cursor(
        app.state.attestation_access_grants_cursor_secret,
        pagination.ATTESTATION_ACCESS_GRANTS_CURSOR,
        {
            "attestation_id": "att_ghost",
            "caller_actor_id": "org-1",
            "limit": 1,
            "offset": 1,
        },
    )
    resp = _list(
        client, "att_ghost", params={"limit": 1, "cursor": ghost_token}
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_cursor_secret_rotation_invalidates_outstanding_cursors(client, app):
    attestation = _world(client)
    _grant_many(client, attestation["id"], 2)
    cursor = _list(client, attestation["id"], params={"limit": 1}).json()[
        "next_cursor"
    ]
    app.state.attestation_access_grants_cursor_secret = secrets.token_bytes(32)
    resp = _list(
        client, attestation["id"], params={"limit": 1, "cursor": cursor}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    # A fresh cursor under the new secret pages normally.
    fresh = _list(client, attestation["id"], params={"limit": 1}).json()
    assert len(fresh["items"]) == 1


# --- The opaque 404 authorization boundary ------------------------------------


def test_grantee_and_other_actors_cannot_list_grants(client):
    attestation = _world(client)
    _post_grant(client, _grant_body(attestation["id"], "org-2"))
    path = _path(attestation["id"])
    # The grantee can read the proof itself but never its grant list.
    assert _list(
        client, attestation["id"], actor="org-2", seed=SEED_B
    ).status_code == 404
    # An authenticated stranger is indistinguishable.
    assert _list(
        client, attestation["id"], actor="org-3", seed=SEED_C
    ).status_code == 404
    # No headers at all.
    assert client.get(path).status_code == 404
    # Blank actor header.
    assert client.get(
        path,
        headers={"X-PA": "  ", "X-PT": "2026-09-20T00:00:00Z", "X-PS": "x"},
    ).status_code == 404
    # A valid signature under a key the claimed actor does not currently hold.
    assert _list(
        client, attestation["id"], actor="org-2", seed=SEED_A
    ).status_code == 404
    # A nonexistent claimed actor with a valid signature.
    assert _list(
        client, attestation["id"], actor="ghost", seed=SEED_A
    ).status_code == 404


def test_missing_target_is_the_same_opaque_404(client):
    _world(client)
    resp = _list(client, "att_ghost")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"
    assert "details" not in resp.json()["error"]


def test_opaque_404_bodies_are_identical(client):
    attestation = _world(client)
    _post_grant(client, _grant_body(attestation["id"], "org-2"))
    path = _path(attestation["id"])
    missing = _list(client, "att_ghost").json()
    no_headers = client.get(path).json()
    grantee = _list(
        client, attestation["id"], actor="org-2", seed=SEED_B
    ).json()
    stranger = _list(
        client, attestation["id"], actor="org-3", seed=SEED_C
    ).json()
    assert missing == no_headers == grantee == stranger
    assert missing == {
        "error": {
            "code": "not_found",
            "message": "The requested resource does not exist.",
        }
    }


def test_revoked_signer_key_collapses_the_listing_into_the_opaque_404(client):
    # Once org-1's only current key is revoked, the signer can no longer
    # authenticate: the read boundary is identical to the protected read.
    attestation = _world(client)
    _post_grant(client, _grant_body(attestation["id"], "org-2"))
    revoke = client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": attestation["id"],
            "revoker_actor_id": "org-1",
            "reason": "rotated",
        },
    )
    assert revoke.status_code == 201
    assert _list(client, attestation["id"]).status_code == 404


# --- Malformed credentials remain 422 -----------------------------------------


def test_malformed_timestamp_and_signature_are_422(client):
    attestation = _world(client)
    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    future = (
        datetime.now(timezone.utc) + timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    for raw, reason in (
        (stale, "timestamp_out_of_window"),
        (future, "timestamp_out_of_window"),
        ("12:00 o'clock", "invalid_timestamp"),
        ("2026-09-20T12:00:00", "invalid_timestamp"),
    ):
        resp = _list(client, attestation["id"], sign={"timestamp": raw})
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["details"]["reason"] == reason, raw

    path = _path(attestation["id"])
    for raw in ("@@@", "aGVsbG8", base64.b64encode(b"x" * 63).decode()):
        headers = _signed_headers("GET", path, b"")
        headers["X-PS"] = raw
        resp = client.get(path, headers=headers)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["code"] == "validation_error"


def test_malformed_credentials_are_422_even_for_a_missing_proof(client):
    _world(client)
    path = _path("att_ghost")
    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    headers = _signed_headers("GET", path, b"", timestamp=stale)
    resp = client.get(path, headers=headers)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_signature_binds_method_path_timestamp_and_empty_body(client):
    attestation = _world(client)
    # A signature committed to another path/method never authorizes this
    # route: well-formed but unverifiable -> the opaque 404.
    assert _list(
        client,
        attestation["id"],
        sign={"signed_path": f"/v1/protected/attestations/{attestation['id']}"},
    ).status_code == 404
    assert _list(
        client,
        attestation["id"],
        sign={"signed_path": _path(attestation["id"]) + "/"},
    ).status_code == 404
    assert _list(
        client, attestation["id"], sign={"signed_method": "POST"}
    ).status_code == 404
    sent = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    signed = (
        datetime.now(timezone.utc) - timedelta(seconds=5)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert _list(
        client,
        attestation["id"],
        sign={"timestamp": sent, "signed_timestamp": signed},
    ).status_code == 404
    # A non-empty body digest cannot authorize the bodyless GET.
    assert _list(
        client, attestation["id"], sign={"signed_body": b"x"}
    ).status_code == 404


def test_query_string_is_not_part_of_the_signed_path(client):
    attestation = _world(client)
    _grant_many(client, attestation["id"], 3)
    # The signature commits to the bare path; a declared query string still
    # succeeds (the zero-byte body digest is unchanged).
    resp = _list(client, attestation["id"], params={"limit": 1})
    assert resp.status_code == 200
    assert len(resp.json()["items"]) == 1
    # An undeclared parameter is the ordinary 422, never an authentication
    # failure, even though it is absent from the signed path.
    resp = _list(
        client, attestation["id"], raw_suffix="tracking=1&limit=1"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Validation precedence -----------------------------------------------------


def test_query_validation_precedes_authorization_for_nonsigners(client):
    attestation = _world(client)
    # An authenticated grantee sending malformed query input gets the 422;
    # the malformed request never collapses into the opaque 404.
    for suffix in (
        "limit=0",
        "limit=abc",
        "limit=1&limit=2",
        "bogus=1",
        "cursor=garbage",
    ):
        resp = _list(
            client,
            attestation["id"],
            actor="org-2",
            seed=SEED_B,
            raw_suffix=suffix,
        )
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error", suffix


def test_garbage_cursor_is_422_without_any_credentials(client):
    attestation = _world(client)
    # Cursor decoding precedes authentication: a forged token is rejected as
    # input validation rather than hidden behind the opaque 404.
    resp = _unsigned(
        client, attestation["id"], raw_suffix="cursor=garbage"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Revocations never leak and revoked grants remain listed ------------------


def test_revoked_grants_remain_listed_without_any_revocation_fields(client):
    attestation = _world(client)
    created = _grant_many(client, attestation["id"], 3)
    body = json.dumps(
        {"grant_id": created[1]["id"], "reason": "no longer needed"}
    ).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **_signed_headers(
            "POST", "/v1/attestation-access-grant-revocations", body
        ),
    }
    revoke = client.post(
        "/v1/attestation-access-grant-revocations",
        content=body,
        headers=headers,
    )
    assert revoke.status_code == 201, revoke.text

    listing = _list(
        client, attestation["id"], params={"limit": 100}
    ).json()
    assert listing["count"] == 3
    assert [item["id"] for item in listing["items"]] == [
        g["id"] for g in created
    ]
    for item in listing["items"]:
        assert set(item) == ITEM_KEYS
        assert "revoked" not in item
        assert "revocation" not in item
        assert "reason" not in item


# --- Read-only guarantees ------------------------------------------------------


def test_non_get_methods_are_not_allowed(client):
    attestation = _world(client)
    url = _path(attestation["id"])
    for method, kwargs in (
        ("post", {"json": {}}),
        ("put", {"json": {}}),
        ("patch", {"json": {}}),
        ("delete", {}),
    ):
        resp = getattr(client, method)(url, **kwargs)
        assert resp.status_code == 405, method
        assert resp.json()["error"]["code"] == "method_not_allowed"


def test_listing_writes_no_resources_or_audit(client, db_session):
    attestation = _world(client)
    created = _grant_many(client, attestation["id"], 3)
    cursor = _list(client, attestation["id"], params={"limit": 1}).json()[
        "next_cursor"
    ]
    events_before = _audit_count(db_session)
    grants_before = len(
        db_session.execute(select(AttestationAccessGrant)).scalars().all()
    )
    attestations_before = len(
        db_session.execute(select(Attestation)).scalars().all()
    )

    for resp in (
        _list(client, attestation["id"]),
        _list(
            client, attestation["id"], params={"limit": 1, "cursor": cursor}
        ),
        _list(client, attestation["id"], actor="org-2", seed=SEED_B),
        _list(client, attestation["id"], actor="org-3", seed=SEED_C),
        _list(client, "att_ghost"),
        _list(client, attestation["id"], params={"limit": 0}),
        _list(client, attestation["id"], params={"cursor": "garbage"}),
    ):
        assert resp.status_code in (200, 404, 422)

    db_session.expire_all()
    assert _audit_count(db_session) == events_before
    rows = db_session.execute(select(AttestationAccessGrant)).scalars().all()
    assert len(rows) == grants_before == 3
    assert {r.id for r in rows} == {g["id"] for g in created}
    assert len(
        db_session.execute(select(Attestation)).scalars().all()
    ) == attestations_before


def test_successful_read_is_byte_for_byte_stable(client):
    attestation = _world(client)
    _grant_many(client, attestation["id"], 3)
    first = _list(client, attestation["id"], params={"limit": 2})
    second = _list(client, attestation["id"], params={"limit": 2})
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_listing_persists_across_app_restart(tmp_db_url, file_client):
    from provenance.app import create_app
    from provenance.config import Settings

    attestation = _world(file_client)
    created = _grant_many(file_client, attestation["id"], 3)
    old_cursor = _list(
        file_client, attestation["id"], params={"limit": 1}
    ).json()["next_cursor"]

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        # Outstanding cursors are invalidated by the restart's secret
        # rotation; a fresh listing returns the same immutable records.
        resp = _list(
            client,
            attestation["id"],
            params={"limit": 1, "cursor": old_cursor},
        )
        assert resp.status_code == 422
        body = _list(
            client, attestation["id"], params={"limit": 100}
        ).json()
        assert body["count"] == 3
        assert body["items"] == created
        assert body["next_cursor"] is None
