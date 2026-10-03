"""Tests for the signer's protected access-grant-state view.

Covers ``GET /v1/attestations/{attestation_id}/access-grant-states``,
including:

* the ``X-PA``/``X-PT``/``X-PS`` signed-header contract with an empty GET
  body (any body bytes are a 422) and a signed path carrying no query
  string;
* signer-only reads: a missing target, missing/unauthenticated credentials,
  and a non-signer (including a grantee that may read the proof itself) all
  collapse into one opaque 404, while malformed credentials stay 422;
* the success envelope ``{"items", "count", "next_cursor", "checked_at"}``
  in that member order, with items carrying exactly the four grant public
  fields plus ``expires_at`` and ``state`` in stable creation order;
* the four derived states at ``checked_at``: ``revoked`` (any revocation
  record, winning over any expiry), ``expired`` (``expires_at`` at or
  before ``checked_at``), ``scheduled`` (a future ``expires_at``), and
  ``active`` (no expiry record at all);
* the ``state`` filter (omitted means all; only the four literals are
  accepted) with ``count`` always naming the proof's total grant count,
  never the filtered or paged window;
* strict ``limit`` (1-100, default 50) and ``cursor`` parsing: undeclared,
  repeated, blank, and malformed parameters are 422;
* opaque, tamper-evident cursors bound to the proof, the caller, the state
  filter, the limit, and the first page's ``checked_at``: continuation
  pages repeat and omit nothing, the final cursor is null, and paging past
  the end returns empty items with the original count;
* strictly read-only: no resource or audit writes from any read or failed
  request, and the existing protected routes keep working unchanged.

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from provenance import pagination
from provenance.models import (
    AttestationAccessGrant,
    AttestationAccessGrantExpiry,
    AuditEvent,
)
from provenance.time_utils import utc_now
from tests.helpers import (
    DIGEST_B,
    SEED_A,
    SEED_B,
    create_actor,
)
from tests.test_attestation_access_grants import (
    _grant_body,
    _make_attestation,
    _make_claim,
    _post_grant,
    _signed_headers,
    _world,
)

SEED_C = b"test-ed25519-seed-c-00000000000000"[:32]
SEED_D = b"test-ed25519-seed-d-00000000000000"[:32]

REVOCATIONS_PATH = "/v1/attestation-access-grant-revocations"


# --- Fixture-style helpers ------------------------------------------------------


def _states_path(attestation_id):
    return f"/v1/attestations/{attestation_id}/access-grant-states"


def _get_states(
    client,
    attestation_id,
    *,
    actor="org-1",
    seed=SEED_A,
    params=None,
    path=None,
    **sign_kwargs,
):
    path = path or _states_path(attestation_id)
    signed = _signed_headers(
        "GET", path, b"", actor=actor, seed=seed, **sign_kwargs
    )
    return client.get(path, params=params, headers=signed)


def _grant(client, attestation_id, grantee):
    resp = _post_grant(client, _grant_body(attestation_id, grantee))
    assert resp.status_code == 201, resp.text
    return resp.json()


def _expiry_path(grant_id):
    return f"/v1/attestation-access-grants/{grant_id}/expiry"


def _post_expiry(client, grant_id, expires_at, *, actor="org-1", seed=SEED_A):
    raw = json.dumps({"expires_at": expires_at}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **_signed_headers(
            "POST", _expiry_path(grant_id), raw, actor=actor, seed=seed
        ),
    }
    resp = client.post(_expiry_path(grant_id), content=raw, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _future(seconds=3600):
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _rewind_expiry(client, grant_id, seconds=5):
    # The expiry row is immutable through the API; rewind it directly to
    # exercise the at-or-before-checked_at boundary without sleeping.
    session = client.app.state.session_factory()
    try:
        row = session.execute(
            select(AttestationAccessGrantExpiry).where(
                AttestationAccessGrantExpiry.grant_id == grant_id
            )
        ).scalar_one()
        row.expires_at = utc_now() - timedelta(seconds=seconds)
        session.commit()
    finally:
        session.close()


def _post_revocation(client, grant_id, reason="no longer needed"):
    raw = json.dumps({"grant_id": grant_id, "reason": reason}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", REVOCATIONS_PATH, raw),
    }
    resp = client.post(REVOCATIONS_PATH, content=raw, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


# --- Success envelope and item shape ------------------------------------------


def test_empty_attestation_returns_empty_page_with_checked_at(client):
    attestation = _world(client)
    resp = _get_states(client, attestation["id"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert list(body) == ["items", "count", "next_cursor", "checked_at"]
    assert body["items"] == []
    assert body["count"] == 0
    assert body["next_cursor"] is None
    checked_at = datetime.fromisoformat(body["checked_at"])
    assert checked_at.utcoffset().total_seconds() == 0
    assert body["checked_at"].endswith(("Z", "+00:00"))


def test_active_item_carries_grant_fields_plus_expiry_and_state(client):
    attestation = _world(client)
    first = _grant(client, attestation["id"], "org-2")
    second = _grant(client, attestation["id"], "org-3")

    resp = _get_states(client, attestation["id"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 2
    assert body["next_cursor"] is None
    assert [item["id"] for item in body["items"]] == [
        first["id"],
        second["id"],
    ]
    for item, grant in zip(body["items"], (first, second)):
        assert list(item) == [
            "id",
            "attestation_id",
            "grantee_actor_id",
            "created_at",
            "expires_at",
            "state",
        ]
        # The existing grant public fields are carried over unchanged.
        assert item["id"] == grant["id"]
        assert item["attestation_id"] == grant["attestation_id"]
        assert item["grantee_actor_id"] == grant["grantee_actor_id"]
        assert item["created_at"] == grant["created_at"]
        assert item["expires_at"] is None
        assert item["state"] == "active"


# --- The four states ------------------------------------------------------------


def test_scheduled_state_echoes_the_future_expiry(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    expiry = _post_expiry(client, grant["id"], _future())

    body = _get_states(client, attestation["id"]).json()
    (item,) = body["items"]
    assert item["state"] == "scheduled"
    assert item["expires_at"] == expiry["expires_at"]


def test_expired_state_when_expiry_is_at_or_before_checked_at(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    _post_expiry(client, grant["id"], _future(3))
    _rewind_expiry(client, grant["id"], seconds=5)

    body = _get_states(client, attestation["id"]).json()
    (item,) = body["items"]
    assert item["state"] == "expired"
    assert datetime.fromisoformat(item["expires_at"]) <= datetime.fromisoformat(
        body["checked_at"]
    )


def test_revoked_state_and_revocation_wins_over_expiry(client):
    attestation = _world(client)
    # Revoked with no expiry at all.
    plain = _grant(client, attestation["id"], "org-2")
    _post_revocation(client, plain["id"])
    # Revoked with a future expiry: revocation still wins.
    scheduled = _grant(client, attestation["id"], "org-3")
    _post_expiry(client, scheduled["id"], _future())
    _post_revocation(client, scheduled["id"], reason="compromised")

    body = _get_states(client, attestation["id"]).json()
    assert [item["state"] for item in body["items"]] == ["revoked", "revoked"]
    assert body["items"][0]["expires_at"] is None
    assert body["items"][1]["expires_at"] is not None


def test_revocation_wins_over_an_elapsed_expiry(client, db_session):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    _post_expiry(client, grant["id"], _future(3))
    _post_revocation(client, grant["id"])
    _rewind_expiry(client, grant["id"], seconds=5)

    (item,) = _get_states(client, attestation["id"]).json()["items"]
    assert item["state"] == "revoked"


def test_all_four_states_in_one_listing(client):
    attestation = _world(client)
    create_actor(client, actor_id="org-4", name="org-4", type="organization")
    active = _grant(client, attestation["id"], "org-2")
    scheduled = _grant(client, attestation["id"], "org-3")
    _post_expiry(client, scheduled["id"], _future())
    expired = _grant(client, attestation["id"], "org-4")
    _post_expiry(client, expired["id"], _future(3))
    _rewind_expiry(client, expired["id"], seconds=5)
    create_actor(client, actor_id="org-5", name="org-5", type="organization")
    revoked = _grant(client, attestation["id"], "org-5")
    _post_revocation(client, revoked["id"])

    body = _get_states(client, attestation["id"]).json()
    assert body["count"] == 4
    assert [item["state"] for item in body["items"]] == [
        "active",
        "scheduled",
        "expired",
        "revoked",
    ]


# --- The state filter -----------------------------------------------------------


def _four_state_world(client):
    attestation = _world(client)
    create_actor(client, actor_id="org-4", name="org-4", type="organization")
    create_actor(client, actor_id="org-5", name="org-5", type="organization")
    active = _grant(client, attestation["id"], "org-2")
    scheduled = _grant(client, attestation["id"], "org-3")
    _post_expiry(client, scheduled["id"], _future())
    expired = _grant(client, attestation["id"], "org-4")
    _post_expiry(client, expired["id"], _future(3))
    _rewind_expiry(client, expired["id"], seconds=5)
    revoked = _grant(client, attestation["id"], "org-5")
    _post_revocation(client, revoked["id"])
    return attestation, {
        "active": active,
        "scheduled": scheduled,
        "expired": expired,
        "revoked": revoked,
    }


def test_state_filter_selects_exactly_one_literal(client):
    attestation, grants = _four_state_world(client)
    for state in ("active", "scheduled", "expired", "revoked"):
        resp = _get_states(
            client, attestation["id"], params={"state": state}
        )
        assert resp.status_code == 200, (state, resp.text)
        body = resp.json()
        # count is the proof's total grant count, never the filtered window.
        assert body["count"] == 4
        assert [item["id"] for item in body["items"]] == [grants[state]["id"]]
        assert all(item["state"] == state for item in body["items"])
        assert body["next_cursor"] is None


def test_state_filter_with_no_match_is_an_empty_page(client):
    attestation = _world(client)
    _grant(client, attestation["id"], "org-2")

    body = _get_states(
        client, attestation["id"], params={"state": "revoked"}
    ).json()
    assert body["items"] == []
    assert body["count"] == 1
    assert body["next_cursor"] is None
    assert body["checked_at"]


def test_illegal_blank_and_repeated_state_are_422(client):
    attestation = _world(client)
    path = _states_path(attestation["id"])

    def signed_get(params):
        return client.get(
            path, params=params, headers=_signed_headers("GET", path, b"")
        )

    cases = [
        ({"state": ""}, "state"),
        ({"state": "   "}, "state"),
        ({"state": "Active"}, "state"),
        ({"state": "ACTIVE"}, "state"),
        ({"state": "pending"}, "state"),
        ({"state": "active "}, "state"),
        ([("state", "active"), ("state", "revoked")], "state"),
    ]
    for params, field in cases:
        resp = signed_get(params)
        assert resp.status_code == 422, (params, resp.text)
        assert resp.json()["error"]["code"] == "validation_error"
        assert resp.json()["error"]["details"]["issues"][0]["loc"][-1] == field


# --- Pagination -----------------------------------------------------------------


def test_pagination_walks_every_grant_once_with_limit_one(client):
    attestation = _world(client)
    created = [
        _grant(client, attestation["id"], g) for g in ("org-2", "org-3")
    ]
    seen = []
    checked_at = None
    cursor = None
    pages = 0
    while True:
        params = {"limit": "1"}
        if cursor is not None:
            params["cursor"] = cursor
        page = _get_states(client, attestation["id"], params=params).json()
        pages += 1
        assert page["count"] == 2
        # The snapshot instant is fixed by the first page and carried by the
        # cursor chain unchanged.
        if checked_at is None:
            checked_at = page["checked_at"]
        assert page["checked_at"] == checked_at
        seen.extend(item["id"] for item in page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
        assert pages <= 2
    assert pages == 2
    assert seen == [grant["id"] for grant in created]


def test_pagination_boundaries_and_stable_replay(client):
    attestation = _world(client)
    created = [
        _grant(client, attestation["id"], g) for g in ("org-2", "org-3")
    ]

    first = _get_states(client, attestation["id"], params={"limit": "1"})
    assert first.status_code == 200, first.text
    first_body = first.json()
    assert len(first_body["items"]) == 1
    assert first_body["count"] == 2
    assert first_body["next_cursor"]
    assert first_body["items"][0]["id"] == created[0]["id"]

    second = _get_states(
        client,
        attestation["id"],
        params={"limit": "1", "cursor": first_body["next_cursor"]},
    )
    assert second.status_code == 200, second.text
    second_body = second.json()
    assert len(second_body["items"]) == 1
    assert second_body["count"] == 2
    assert second_body["next_cursor"] is None
    assert second_body["items"][0]["id"] == created[1]["id"]
    assert second_body["checked_at"] == first_body["checked_at"]

    # Replaying a still-valid cursor replays the identical page: no gap, no
    # duplication, no new cursor chain.
    replay = _get_states(
        client,
        attestation["id"],
        params={"limit": "1", "cursor": first_body["next_cursor"]},
    )
    assert replay.json() == second_body


def test_filtered_pagination_walks_the_matching_grants_only(client):
    attestation, grants = _four_state_world(client)
    # A second scheduled grant so the filtered set spans two pages.
    create_actor(client, actor_id="org-6", name="org-6", type="organization")
    extra = _grant(client, attestation["id"], "org-6")
    _post_expiry(client, extra["id"], _future())

    first = _get_states(
        client,
        attestation["id"],
        params={"state": "scheduled", "limit": "1"},
    ).json()
    assert first["count"] == 5
    assert [item["id"] for item in first["items"]] == [
        grants["scheduled"]["id"]
    ]
    assert first["next_cursor"]

    second = _get_states(
        client,
        attestation["id"],
        params={
            "state": "scheduled",
            "limit": "1",
            "cursor": first["next_cursor"],
        },
    ).json()
    assert second["count"] == 5
    assert [item["id"] for item in second["items"]] == [extra["id"]]
    assert second["next_cursor"] is None
    assert second["checked_at"] == first["checked_at"]


def test_cursor_past_the_end_returns_empty_items_and_original_count(
    client, app
):
    attestation = _world(client)
    _grant(client, attestation["id"], "org-2")
    # A server-signed token positioned past the tail: same proof, caller,
    # filter, limit, and snapshot instant, with an offset beyond the
    # single-row result set.
    token = pagination.encode_typed_cursor(
        app.state.attestation_access_grant_states_cursor_secret,
        pagination.ATTESTATION_ACCESS_GRANT_STATES_CURSOR,
        {
            "attestation_id": attestation["id"],
            "actor_id": "org-1",
            "state": None,
            "limit": 50,
            "checked_at": utc_now().isoformat(),
            "offset": 99,
        },
    )
    resp = _get_states(client, attestation["id"], params={"cursor": token})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["items"] == []
    assert body["count"] == 1
    assert body["next_cursor"] is None
    assert body["checked_at"]


def test_limit_boundaries_one_and_one_hundred_are_accepted(client):
    attestation = _world(client)
    for value in ("1", "100"):
        resp = _get_states(client, attestation["id"], params={"limit": value})
        assert resp.status_code == 200, (value, resp.text)
        assert resp.json()["count"] == 0
        assert resp.json()["items"] == []


def test_default_limit_is_fifty(client):
    attestation = _world(client)
    # 51 distinct new grantees (org-4..org-54); each grant needs an existing
    # actor, so register them first.
    grantees = [f"org-{i}" for i in range(4, 55)]
    for grantee in grantees:
        create_actor(
            client, actor_id=grantee, name=grantee, type="organization"
        )
    for grantee in grantees:
        _grant(client, attestation["id"], grantee)

    first = _get_states(client, attestation["id"])
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["count"] == 51
    assert len(body["items"]) == 50
    assert body["next_cursor"]

    second = _get_states(
        client, attestation["id"], params={"cursor": body["next_cursor"]}
    )
    assert second.status_code == 200, second.text
    tail = second.json()
    assert tail["count"] == 51
    assert len(tail["items"]) == 1
    assert tail["next_cursor"] is None
    assert tail["items"][0]["grantee_actor_id"] == "org-54"
    assert tail["checked_at"] == body["checked_at"]


# --- Query parameter and body validation ----------------------------------------


def test_undeclared_repeated_blank_and_illegal_params_are_422(client):
    attestation = _world(client)
    path = _states_path(attestation["id"])

    def signed_get(params):
        return client.get(
            path, params=params, headers=_signed_headers("GET", path, b"")
        )

    cases = [
        # Undeclared parameter, on its own or beside a valid one.
        ({"offset": "0"}, "offset"),
        ({"limit": "1", "track": "1"}, "track"),
        ({"checked_at": "2026-01-01T00:00:00Z"}, "checked_at"),
        # Repeated scalar parameters.
        ([("limit", "1"), ("limit", "2")], "limit"),
        ([("cursor", "a"), ("cursor", "b")], "cursor"),
        # Blank / whitespace limit is never coerced to the default.
        ({"limit": ""}, "limit"),
        ({"limit": "   "}, "limit"),
        # Out of range, non-decimal, and non-integer spellings.
        ({"limit": "0"}, "limit"),
        ({"limit": "101"}, "limit"),
        ({"limit": "-1"}, "limit"),
        ({"limit": "1.0"}, "limit"),
        ({"limit": "abc"}, "limit"),
        ({"limit": "0x1"}, "limit"),
        ({"limit": "+1"}, "limit"),
        # A blank cursor is an invalid token.
        ({"cursor": ""}, "cursor"),
        ({"cursor": "   "}, "cursor"),
    ]
    for params, field in cases:
        resp = signed_get(params)
        assert resp.status_code == 422, (params, resp.text)
        assert resp.json()["error"]["code"] == "validation_error"
        assert resp.json()["error"]["details"]["issues"][0]["loc"][-1] == field


def test_get_body_with_any_bytes_is_422(client):
    attestation = _world(client)
    path = _states_path(attestation["id"])
    for body in (b"{}", b" ", b"x", b"\x00"):
        headers = _signed_headers("GET", path, body)
        resp = client.request("GET", path, content=body, headers=headers)
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"


def test_malformed_query_is_422_even_without_credentials(client):
    attestation = _world(client)
    path = _states_path(attestation["id"])
    # Parameter validation precedes authentication, so a malformed request
    # never collapses into the opaque 404.
    resp = client.get(path, params={"limit": "0"})
    assert resp.status_code == 422
    resp = client.get(path, params={"cursor": "garbage"})
    assert resp.status_code == 422
    resp = client.get(path, params={"state": "pending"})
    assert resp.status_code == 422


# --- Cursor integrity and binding -----------------------------------------------


def test_tampered_and_foreign_family_cursors_are_422(client, app):
    attestation = _world(client)
    _grant(client, attestation["id"], "org-2")
    _grant(client, attestation["id"], "org-3")
    cursor = _get_states(
        client, attestation["id"], params={"limit": "1"}
    ).json()["next_cursor"]
    assert cursor

    tampered = cursor[:-1] + ("A" if cursor[-1] != "A" else "B")
    resp = _get_states(
        client, attestation["id"], params={"limit": "1", "cursor": tampered}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    for garbage in ("not-a-cursor", "xx.ab.cd", "v1.ab.cd", "../etc", "a b"):
        resp = _get_states(
            client, attestation["id"], params={"cursor": garbage}
        )
        assert resp.status_code == 422, garbage

    # A cursor minted by the sibling access-grants listing never resumes
    # this view, even for the same proof, caller, and limit.
    foreign = pagination.encode_typed_cursor(
        app.state.attestation_access_grants_cursor_secret,
        pagination.ATTESTATION_ACCESS_GRANTS_CURSOR,
        {
            "attestation_id": attestation["id"],
            "actor_id": "org-1",
            "limit": 1,
            "offset": 1,
        },
    )
    resp = _get_states(
        client, attestation["id"], params={"limit": "1", "cursor": foreign}
    )
    assert resp.status_code == 422


def test_cursor_binds_proof_caller_state_and_limit(client):
    attestation = _world(client)
    other_att = _make_attestation(
        client,
        "org-1",
        SEED_D,
        _make_claim(
            client, "org-1", digest=hashlib.sha256(b"other-proof").hexdigest()
        )["id"],
    )
    _grant(client, attestation["id"], "org-2")
    _grant(client, attestation["id"], "org-3")
    cursor = _get_states(
        client, attestation["id"], params={"limit": "1"}
    ).json()["next_cursor"]

    # A different effective limit invalidates the cursor.
    resp = _get_states(
        client, attestation["id"], params={"limit": "2", "cursor": cursor}
    )
    assert resp.status_code == 422

    # A different effective state filter invalidates the cursor.
    resp = _get_states(
        client,
        attestation["id"],
        params={"limit": "1", "state": "active", "cursor": cursor},
    )
    assert resp.status_code == 422

    # A cursor minted for one proof never pages another proof, even for the
    # same signer.
    resp = _get_states(
        client, other_att["id"], params={"limit": "1", "cursor": cursor}
    )
    assert resp.status_code == 422

    # A cursor bound to the signer cannot be replayed by an authenticated
    # different caller: it does not bind to that caller.
    resp = _get_states(
        client,
        attestation["id"],
        params={"limit": "1", "cursor": cursor},
        actor="org-2",
        seed=SEED_B,
    )
    assert resp.status_code == 422


def test_filtered_cursor_resumes_only_the_same_filter(client):
    attestation, _ = _four_state_world(client)
    create_actor(client, actor_id="org-6", name="org-6", type="organization")
    extra = _grant(client, attestation["id"], "org-6")
    _post_expiry(client, extra["id"], _future())
    cursor = _get_states(
        client,
        attestation["id"],
        params={"state": "scheduled", "limit": "1"},
    ).json()["next_cursor"]
    assert cursor

    # Dropping the filter while carrying the filtered cursor is a mismatch.
    resp = _get_states(
        client, attestation["id"], params={"limit": "1", "cursor": cursor}
    )
    assert resp.status_code == 422
    # Replaying with the same filter resumes exactly where it left off.
    resp = _get_states(
        client,
        attestation["id"],
        params={"state": "scheduled", "limit": "1", "cursor": cursor},
    )
    assert resp.status_code == 200
    assert [item["id"] for item in resp.json()["items"]] == [extra["id"]]


# --- The opaque 404 boundary ------------------------------------------------------


def test_missing_target_unauthenticated_and_non_signer_are_one_404(client):
    attestation = _world(client)
    path = _states_path(attestation["id"])
    _grant(client, attestation["id"], "org-2")

    # No credentials at all.
    no_headers = client.get(path)
    assert no_headers.status_code == 404
    assert no_headers.json()["error"]["code"] == "not_found"

    # Blank actor header (missing credential material), even with grants.
    blank = client.get(
        path,
        headers={"X-PA": "  ", "X-PT": "2026-09-20T00:00:00Z", "X-PS": "x"},
    )
    assert blank.status_code == 404

    # A well-formed signature that does not bind to the claimed identity.
    assert (
        _get_states(
            client, attestation["id"], actor="org-2", seed=SEED_A
        ).status_code
        == 404
    )

    # An authenticated stranger is not the signer: opaque 404.
    assert (
        _get_states(
            client, attestation["id"], actor="org-3", seed=SEED_C
        ).status_code
        == 404
    )

    # A grantee may read the protected proof but may not read its grant
    # states.
    assert (
        _get_states(
            client, attestation["id"], actor="org-2", seed=SEED_B
        ).status_code
        == 404
    )

    # A missing target renders identically for the signer.
    ghost = _get_states(client, "att_ghost")
    assert ghost.status_code == 404
    assert ghost.json() == no_headers.json() == blank.json()


def test_malformed_credentials_remain_422(client):
    attestation = _world(client)
    stale = (datetime.now(timezone.utc) - timedelta(seconds=301)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    assert (
        _get_states(client, attestation["id"], timestamp=stale).status_code
        == 422
    )
    assert (
        _get_states(
            client, attestation["id"], timestamp="12:00 o'clock"
        ).status_code
        == 422
    )

    import base64

    path = _states_path(attestation["id"])
    for raw in ("@@@", "aGVsbG8", base64.b64encode(b"x" * 63).decode()):
        headers = _signed_headers("GET", path, b"")
        headers["X-PS"] = raw
        resp = client.get(path, headers=headers)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["code"] == "validation_error"


def test_signature_binds_method_path_empty_body_and_timestamp(client):
    attestation = _world(client)
    # A signature minted for the write route never authorizes the read.
    assert _get_states(client, attestation["id"], signed_method="POST").status_code == 404
    path = _states_path(attestation["id"])
    assert (
        _get_states(
            client, attestation["id"], signed_path=path + "/"
        ).status_code
        == 404
    )
    # A non-empty body digest cannot authorize a bodyless GET.
    assert (
        _get_states(client, attestation["id"], signed_body=b"x").status_code
        == 404
    )
    sent = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    signed = (datetime.now(timezone.utc) - timedelta(seconds=5)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    assert (
        _get_states(
            client,
            attestation["id"],
            timestamp=sent,
            signed_timestamp=signed,
        ).status_code
        == 404
    )


def test_signed_path_excludes_the_query_string(client):
    attestation = _world(client)
    path = _states_path(attestation["id"])
    headers = _signed_headers("GET", path, b"")
    # The signature covers the path without a query string; attaching one
    # still succeeds, and the same signature validates both spellings.
    assert (
        client.get(path, params={"limit": "1"}, headers=headers).status_code
        == 200
    )
    assert client.get(path + "?limit=1", headers=headers).status_code == 200


# --- Read-only guarantees ---------------------------------------------------------


def test_listing_writes_no_resources_or_audit(client, db_session):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    _post_expiry(client, grant["id"], _future())
    events_before = _audit_count(db_session)
    grants_before = len(
        db_session.execute(select(AttestationAccessGrant)).scalars().all()
    )
    expiries_before = len(
        db_session.execute(select(AttestationAccessGrantExpiry))
        .scalars()
        .all()
    )

    path = _states_path(attestation["id"])
    responses = [
        _get_states(client, attestation["id"]),
        _get_states(client, attestation["id"], params={"limit": "1"}),
        _get_states(client, attestation["id"], params={"state": "active"}),
        _get_states(client, attestation["id"], actor="org-3", seed=SEED_C),
        _get_states(client, "att_ghost"),
        client.get(path),
        client.get(path, params={"limit": "0"}),
    ]
    for resp in responses:
        assert resp.status_code in (200, 404, 422), resp.text

    db_session.expire_all()
    assert _audit_count(db_session) == events_before
    assert (
        len(db_session.execute(select(AttestationAccessGrant)).scalars().all())
        == grants_before
    )
    assert (
        len(
            db_session.execute(select(AttestationAccessGrantExpiry))
            .scalars()
            .all()
        )
        == expiries_before
    )


def test_existing_protected_routes_keep_working(client):
    attestation = _world(client)
    _grant(client, attestation["id"], "org-2")
    # The signer's grant listing and the protected proof read still serve
    # their existing shapes unchanged.
    grants_path = f"/v1/attestations/{attestation['id']}/access-grants"
    listing = client.get(
        grants_path, headers=_signed_headers("GET", grants_path, b"")
    )
    assert listing.status_code == 200
    assert set(listing.json()) == {"items", "count", "next_cursor"}
    assert set(listing.json()["items"][0]) == {
        "id",
        "attestation_id",
        "grantee_actor_id",
        "created_at",
    }
    proof_path = f"/v1/protected/attestations/{attestation['id']}"
    signer = client.get(
        proof_path, headers=_signed_headers("GET", proof_path, b"")
    )
    assert signer.status_code == 200
    grantee = client.get(
        proof_path,
        headers=_signed_headers(
            "GET", proof_path, b"", actor="org-2", seed=SEED_B
        ),
    )
    assert grantee.status_code == 200
