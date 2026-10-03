"""Tests for the signer's effective access-grant-state listing.

Covers ``GET /v1/attestations/{attestation_id}/access-grant-states``:

* the ``X-PA``/``X-PT``/``X-PS`` signed-header contract with an empty GET
  body and a signed path carrying no query string;
* signer-only reads: a missing target, missing/unauthenticated credentials,
  and a non-signer (including a grantee that may read the proof itself) all
  collapse into one opaque 404, while malformed credentials stay 422;
* the success envelope ``{"items", "count", "next_cursor", "checked_at"}``
  in exactly that member order, each item carrying the existing grant
  public fields plus ``expires_at`` and ``state``;
* the four effective states at the server's ``checked_at`` UTC instant:
  ``active`` (no expiry), ``scheduled`` (future expiry), ``expired``
  (expiry at or before checked_at), and ``revoked`` (a recorded revocation,
  which wins even when the expiry is also due);
* the optional ``state`` filter and stable-creation-order pagination with a
  cursor signed and bound to the proof, caller, state, limit, and
  checked_at; ``count`` is always the total grant count;
* strict validation: any GET body byte, undeclared/repeated/blank params,
  an illegal state, an out-of-range/non-decimal/repeated limit, and an
  empty/tampered/cross-family/non-binding cursor are all 422;
* strictly read-only: no resource or audit writes from any read or failed
  request, and the existing protected routes keep working unchanged.

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures).
"""

from __future__ import annotations

import base64
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
from tests.helpers import create_actor, SEED_A, SEED_B
from tests.test_attestation_access_grants import (
    _make_attestation,
    _make_claim,
    _post_grant,
    _signed_headers,
    _world,
    _grant_body,
)

REVOCATIONS_PATH = "/v1/attestation-access-grant-revocations"
SEED_C = b"test-ed25519-seed-c-00000000000000"[:32]
SEED_D = b"test-ed25519-seed-d-00000000000000"[:32]


# --- Helpers ------------------------------------------------------------------


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
    content=b"",
    **sign_kwargs,
):
    path = path or _states_path(attestation_id)
    signed = _signed_headers(
        "GET", path, b"", actor=actor, seed=seed, **sign_kwargs
    )
    return client.request(
        "GET", path, params=params, headers=signed, content=content
    )


def _grant(client, attestation_id, grantee="org-2"):
    resp = _post_grant(client, _grant_body(attestation_id, grantee))
    assert resp.status_code in (200, 201), resp.text
    return resp.json()


def _revoke(client, grant_id, reason="no longer needed"):
    body = json.dumps({"grant_id": grant_id, "reason": reason}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", REVOCATIONS_PATH, body),
    }
    resp = client.post(REVOCATIONS_PATH, content=body, headers=headers)
    assert resp.status_code in (200, 201), resp.text
    return resp.json()


def _schedule_expiry(client, grant_id, *, seconds=3600):
    expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=seconds)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    path = f"/v1/attestation-access-grants/{grant_id}/expiry"
    body = json.dumps({"expires_at": expires_at}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", path, body),
    }
    resp = client.post(path, content=body, headers=headers)
    assert resp.status_code in (200, 201), resp.text
    return resp.json()


def _rewind_expiry(client, grant_id, *, seconds=1):
    """Move a scheduled expiry into the past directly in the database.

    The expiry row is immutable through the API; the test rewinds it to
    exercise the at-or-before-now boundary without sleeping.
    """
    factory = client.app.state.session_factory
    session = factory()
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


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


def _by_grant(body):
    return {item["id"]: item for item in body["items"]}


# --- Empty collection and envelope --------------------------------------------


def test_empty_attestation_returns_empty_page_with_null_cursor(client):
    attestation = _world(client)
    resp = _get_states(client, attestation["id"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["items"] == []
    assert body["count"] == 0
    assert body["next_cursor"] is None
    checked_at = datetime.fromisoformat(body["checked_at"])
    assert checked_at.utcoffset().total_seconds() == 0
    # Members appear in exactly the declared order in the raw body.
    raw = resp.text
    assert (
        raw.index('"items"')
        < raw.index('"count"')
        < raw.index('"next_cursor"')
        < raw.index('"checked_at"')
    )
    assert raw.endswith("\n") and not raw.endswith("\n\n")


# --- Item shape and the four states -------------------------------------------


def test_item_fields_and_active_state_for_a_grant_without_expiry(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"])
    resp = _get_states(client, attestation["id"])
    assert resp.status_code == 200, resp.text
    item = resp.json()["items"][0]
    assert set(item) == {
        "id",
        "attestation_id",
        "grantee_actor_id",
        "created_at",
        "expires_at",
        "state",
    }
    assert item["id"] == grant["id"]
    assert item["attestation_id"] == attestation["id"]
    assert item["grantee_actor_id"] == "org-2"
    assert item["created_at"] == grant["created_at"]
    assert item["expires_at"] is None
    assert item["state"] == "active"


def test_future_expiry_is_scheduled_and_past_expiry_is_expired(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"])
    expiry = _schedule_expiry(client, grant["id"], seconds=3600)

    scheduled = _get_states(client, attestation["id"]).json()
    item = scheduled["items"][0]
    assert item["state"] == "scheduled"
    assert item["expires_at"] == expiry["expires_at"]

    _rewind_expiry(client, grant["id"])
    expired = _get_states(client, attestation["id"]).json()
    item = expired["items"][0]
    assert item["state"] == "expired"
    # The scheduled instant stays visible (now in the past) once it is due.
    assert item["expires_at"] is not None
    expired_at = datetime.fromisoformat(item["expires_at"])
    assert expired_at <= datetime.fromisoformat(expired["checked_at"])


def test_revocation_is_revoked_even_when_the_expiry_is_also_due(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"])
    expiry = _schedule_expiry(client, grant["id"], seconds=3600)
    _revoke(client, grant["id"])

    item = _get_states(client, attestation["id"]).json()["items"][0]
    assert item["state"] == "revoked"
    # Revocation wins, but the expiry record is still part of the view.
    assert item["expires_at"] == expiry["expires_at"]

    # Revocation precedence holds with the expiry due as well.
    _rewind_expiry(client, grant["id"])
    item = _get_states(client, attestation["id"]).json()["items"][0]
    assert item["state"] == "revoked"


def test_all_four_states_coexist_in_creation_order(client):
    attestation = _world(client)
    create_actor(
        client, actor_id="org-4", name="Fourth Org", type="organization"
    )
    active_grant = _grant(client, attestation["id"], "org-2")
    scheduled_grant = _grant(client, attestation["id"], "org-3")
    expired_grant = _grant(client, attestation["id"], "org-4")
    # A fifth grantee carrying a revoked grant.
    create_actor(
        client, actor_id="org-5", name="Fifth Org", type="organization"
    )
    revoked_grant = _grant(client, attestation["id"], "org-5")

    _schedule_expiry(client, scheduled_grant["id"], seconds=3600)
    _schedule_expiry(client, expired_grant["id"], seconds=3600)
    _rewind_expiry(client, expired_grant["id"])
    _schedule_expiry(client, revoked_grant["id"], seconds=7200)
    _revoke(client, revoked_grant["id"])

    body = _get_states(client, attestation["id"]).json()
    assert body["count"] == 4
    assert [item["id"] for item in body["items"]] == [
        active_grant["id"],
        scheduled_grant["id"],
        expired_grant["id"],
        revoked_grant["id"],
    ]
    states = _by_grant(body)
    assert states[active_grant["id"]]["state"] == "active"
    assert states[active_grant["id"]]["expires_at"] is None
    assert states[scheduled_grant["id"]]["state"] == "scheduled"
    assert states[expired_grant["id"]]["state"] == "expired"
    expired_expires_at = datetime.fromisoformat(
        states[expired_grant["id"]]["expires_at"]
    )
    assert expired_expires_at <= datetime.fromisoformat(body["checked_at"])
    assert states[revoked_grant["id"]]["state"] == "revoked"


# --- state filter --------------------------------------------------------------


def test_state_filter_returns_only_matching_grants_but_count_is_total(client):
    attestation = _world(client)
    create_actor(
        client, actor_id="org-4", name="Fourth Org", type="organization"
    )
    create_actor(
        client, actor_id="org-5", name="Fifth Org", type="organization"
    )
    active_grant = _grant(client, attestation["id"], "org-2")
    scheduled_grant = _grant(client, attestation["id"], "org-3")
    expired_grant = _grant(client, attestation["id"], "org-4")
    revoked_grant = _grant(client, attestation["id"], "org-5")

    _schedule_expiry(client, scheduled_grant["id"], seconds=3600)
    _schedule_expiry(client, expired_grant["id"], seconds=3600)
    _rewind_expiry(client, expired_grant["id"])
    _revoke(client, revoked_grant["id"])

    for literal, expected in (
        ("active", {active_grant["id"]}),
        ("scheduled", {scheduled_grant["id"]}),
        ("expired", {expired_grant["id"]}),
        ("revoked", {revoked_grant["id"]}),
    ):
        body = _get_states(
            client, attestation["id"], params={"state": literal}
        ).json()
        assert {item["id"] for item in body["items"]} == expected
        assert all(item["state"] == literal for item in body["items"])
        # count never follows the filter.
        assert body["count"] == 4
        assert body["next_cursor"] is None

    # No filter returns everything.
    unfiltered = _get_states(client, attestation["id"]).json()
    assert len(unfiltered["items"]) == 4
    assert unfiltered["count"] == 4


def test_state_filter_with_a_matching_empty_set_keeps_count(client):
    attestation = _world(client)
    _grant(client, attestation["id"])
    body = _get_states(
        client, attestation["id"], params={"state": "revoked"}
    ).json()
    assert body == {
        "items": [],
        "count": 1,
        "next_cursor": None,
        "checked_at": body["checked_at"],
    }


# --- Pagination ---------------------------------------------------------------


def test_pagination_walks_every_grant_once_with_limit_one(client):
    attestation = _world(client)
    grantees = ["org-2", "org-3"]
    for grantee in ("org-4", "org-5"):
        create_actor(
            client, actor_id=grantee, name=grantee, type="organization"
        )
        grantees.append(grantee)
    created = [_grant(client, attestation["id"], g)["id"] for g in grantees]
    seen = []
    cursor = None
    checked_at = None
    pages = 0
    while True:
        params = {"limit": "1"}
        if cursor is not None:
            params["cursor"] = cursor
        page = _get_states(client, attestation["id"], params=params).json()
        pages += 1
        assert page["count"] == 4
        if checked_at is None:
            checked_at = page["checked_at"]
        # The checked instant is fixed for the whole cursor walk.
        assert page["checked_at"] == checked_at
        seen.extend(item["id"] for item in page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
        assert pages <= 4
    assert pages == 4
    assert seen == created


def test_filtered_pagination_walks_only_matches_without_gaps(client):
    attestation = _world(client)
    for grantee in ("org-4", "org-5"):
        create_actor(
            client, actor_id=grantee, name=grantee, type="organization"
        )
    active_one = _grant(client, attestation["id"], "org-2")
    scheduled_one = _grant(client, attestation["id"], "org-3")
    revoked_one = _grant(client, attestation["id"], "org-4")
    active_two = _grant(client, attestation["id"], "org-5")
    _schedule_expiry(client, scheduled_one["id"], seconds=3600)
    _revoke(client, revoked_one["id"])

    seen = []
    cursor = None
    while True:
        params = {"state": "active", "limit": "1"}
        if cursor is not None:
            params["cursor"] = cursor
        page = _get_states(client, attestation["id"], params=params).json()
        assert page["count"] == 4
        seen.extend(item["id"] for item in page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    # Filtered paging walks the two active rows in creation order and never
    # surfaces the intervening scheduled/revoked rows.
    assert seen == [active_one["id"], active_two["id"]]


def test_cursor_past_the_end_returns_empty_items_and_original_count(
    client, app
):
    attestation = _world(client)
    _grant(client, attestation["id"])
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


def test_limit_boundaries_and_default_of_fifty(client):
    attestation = _world(client)
    for value in ("1", "100"):
        resp = _get_states(
            client, attestation["id"], params={"limit": value}
        )
        assert resp.status_code == 200, (value, resp.text)
        assert resp.json()["items"] == []
        assert resp.json()["count"] == 0

    # 49 distinct new grantees (org-4..org-52), 51 grants total alongside
    # the world's org-2 and org-3; each grant needs an existing actor.
    new_grantees = [f"org-{i}" for i in range(4, 53)]
    for grantee in new_grantees:
        create_actor(
            client, actor_id=grantee, name=grantee, type="organization"
        )
    grantees = ["org-2", "org-3", *new_grantees]
    for grantee in grantees:
        _grant(client, attestation["id"], grantee)

    first = _get_states(client, attestation["id"]).json()
    assert first["count"] == 51
    assert len(first["items"]) == 50
    assert first["next_cursor"]
    tail = _get_states(
        client, attestation["id"],
        params={"cursor": first["next_cursor"]},
    ).json()
    assert tail["count"] == 51
    assert len(tail["items"]) == 1
    assert tail["next_cursor"] is None
    assert tail["items"][0]["grantee_actor_id"] == "org-52"


# --- Query parameter validation -----------------------------------------------


def test_illegal_repeated_undeclared_params_are_422(client):
    attestation = _world(client)
    path = _states_path(attestation["id"])

    cases = [
        # Undeclared parameter, on its own or beside a valid one.
        ({"offset": "0"}, "offset"),
        ({"limit": "1", "track": "1"}, "track"),
        ({"checked_at": "2026-01-01T00:00:00Z"}, "checked_at"),
        # Illegal state literals (case sensitive, no blanks/aliases).
        ({"state": ""}, "state"),
        ({"state": "   "}, "state"),
        ({"state": "REVOKED"}, "state"),
        ({"state": "revoke"}, "state"),
        ({"state": "expiry"}, "state"),
        # Repeated parameters.
        ([("state", "active"), ("state", "revoked")], "state"),
        ([("limit", "1"), ("limit", "2")], "limit"),
        ([("cursor", "a"), ("cursor", "b")], "cursor"),
        # limit boundaries and format.
        ({"limit": ""}, "limit"),
        ({"limit": "   "}, "limit"),
        ({"limit": "0"}, "limit"),
        ({"limit": "101"}, "limit"),
        ({"limit": "-1"}, "limit"),
        ({"limit": "1.0"}, "limit"),
        ({"limit": "abc"}, "limit"),
        ({"limit": "0x1"}, "limit"),
        ({"limit": " 1 "}, "limit"),
        # Blank cursor.
        ({"cursor": ""}, "cursor"),
        ({"cursor": "   "}, "cursor"),
    ]
    for params, field in cases:
        resp = _get_states(client, attestation["id"], params=params)
        assert resp.status_code == 422, (params, resp.text)
        body = resp.json()
        assert body["error"]["code"] == "validation_error"
        assert body["error"]["details"]["issues"][0]["loc"][-1] == field


def test_malformed_query_is_422_even_without_credentials(client):
    attestation = _world(client)
    path = _states_path(attestation["id"])
    # Parameter validation precedes authentication.
    assert client.get(path, params={"limit": "0"}).status_code == 422
    assert client.get(path, params={"state": "nope"}).status_code == 422
    assert client.get(path, params={"cursor": "garbage"}).status_code == 422


def test_any_get_body_byte_is_422_before_anything_else(client):
    attestation = _world(client)
    path = _states_path(attestation["id"])
    for raw in (b"{}", b" ", b"\x00", b"null"):
        # With credentials ...
        resp = _get_states(client, attestation["id"], content=raw)
        assert resp.status_code == 422, (raw, resp.text)
        assert resp.json()["error"]["details"]["issues"][0]["loc"][-1] == "body"
        # ... and with none at all: the body is rejected first.
        resp = client.request("GET", path, content=raw)
        assert resp.status_code == 422, raw


# --- Cursor integrity and binding ---------------------------------------------


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
        client, attestation["id"],
        params={"limit": "1", "cursor": tampered},
    )
    assert resp.status_code == 422

    for garbage in ("not-a-cursor", "xx.ab.cd", "v1.ab.cd", "../etc", "a b"):
        assert _get_states(
            client, attestation["id"], params={"cursor": garbage}
        ).status_code == 422, garbage

    # A cursor from the access-grants listing family (ag1) cannot resume
    # this entry even though the marker and claims look similar.
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
    assert _get_states(
        client, attestation["id"],
        params={"limit": "1", "cursor": foreign},
    ).status_code == 422


def test_cursor_binds_proof_caller_state_limit_and_checked_at(
    client, app
):
    attestation = _world(client)
    _grant(client, attestation["id"], "org-2")
    _grant(client, attestation["id"], "org-3")
    other_att = _make_attestation(
        client, "org-1", SEED_D,
        _make_claim(
            client, "org-1", digest=hashlib.sha256(b"other-proof").hexdigest()
        )["id"],
    )

    def mint(*, state=None, limit=1, attestation=attestation["id"],
             actor="org-1", checked_at=None, offset=1):
        return pagination.encode_typed_cursor(
            app.state.attestation_access_grant_states_cursor_secret,
            pagination.ATTESTATION_ACCESS_GRANT_STATES_CURSOR,
            {
                "attestation_id": attestation,
                "actor_id": actor,
                "state": state,
                "limit": limit,
                "checked_at": (checked_at or utc_now()).isoformat(),
                "offset": offset,
            },
        )

    valid = mint()
    # Different effective limit.
    assert _get_states(
        client, attestation["id"],
        params={"limit": "2", "cursor": valid},
    ).status_code == 422
    # Cursor minted unfiltered cannot carry a filter on replay.
    assert _get_states(
        client, attestation["id"],
        params={"limit": "1", "state": "active", "cursor": valid},
    ).status_code == 422
    # A filter-minted cursor cannot be replayed unfiltered (or under another
    # filter).
    filtered = mint(state="active")
    assert _get_states(
        client, attestation["id"],
        params={"limit": "1", "cursor": filtered},
    ).status_code == 422
    assert _get_states(
        client, attestation["id"],
        params={"limit": "1", "state": "revoked", "cursor": filtered},
    ).status_code == 422
    # Bound to another proof, even by the same signer.
    other_proof = mint(attestation=other_att["id"])
    assert _get_states(
        client, attestation["id"],
        params={"limit": "1", "cursor": other_proof},
    ).status_code == 422
    # Bound to another caller: a token naming a different actor does not
    # match the authenticated replay, even with a valid signature.
    other_caller = mint(actor="org-2")
    assert _get_states(
        client, attestation["id"],
        params={"limit": "1", "cursor": other_caller},
        actor="org-3", seed=SEED_C,
    ).status_code == 422
    # A structurally invalid checked_at claim is rejected at decode: the
    # walk's instant must be a canonical RFC 3339 UTC string. (A correctly
    # signed token carrying any other instant is simply the start of its own
    # walk -- the continuation request has no checked_at parameter to
    # mismatch, and the pagination-walk test asserts the instant stays fixed
    # across a real walk.)
    bad_instant = pagination.encode_typed_cursor(
        app.state.attestation_access_grant_states_cursor_secret,
        pagination.ATTESTATION_ACCESS_GRANT_STATES_CURSOR,
        {
            "attestation_id": attestation["id"],
            "actor_id": "org-1",
            "state": None,
            "limit": 1,
            "checked_at": "2026-01-01 12:00",
            "offset": 1,
        },
    )
    assert _get_states(
        client, attestation["id"],
        params={"limit": "1", "cursor": bad_instant},
    ).status_code == 422


# --- The opaque 404 boundary ---------------------------------------------------


def test_missing_target_unauthenticated_and_non_signer_are_one_404(client):
    attestation = _world(client)
    path = _states_path(attestation["id"])
    _grant(client, attestation["id"])

    no_headers = client.get(path)
    assert no_headers.status_code == 404
    assert no_headers.json()["error"]["code"] == "not_found"

    blank = client.get(
        path, headers={"X-PA": "  ", "X-PT": "2026-09-20T00:00:00Z",
                       "X-PS": "x"}
    )
    assert blank.status_code == 404

    # A well-formed signature that does not bind to the claimed identity.
    assert _get_states(
        client, attestation["id"],
        actor="org-2", seed=SEED_A,
    ).status_code == 404
    # An authenticated stranger.
    assert _get_states(
        client, attestation["id"], actor="org-3", seed=SEED_C
    ).status_code == 404
    # A grantee may read the protected proof but may not read states.
    assert _get_states(
        client, attestation["id"],
        actor="org-2", seed=SEED_B,
    ).status_code == 404
    # A missing target renders identically for the signer.
    ghost = _get_states(client, "att_ghost")
    assert ghost.status_code == 404
    assert ghost.json() == no_headers.json() == blank.json()


def test_malformed_credentials_remain_422(client):
    attestation = _world(client)
    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert _get_states(
        client, attestation["id"], timestamp=stale
    ).status_code == 422
    assert _get_states(
        client, attestation["id"], timestamp="12:00 o'clock"
    ).status_code == 422

    path = _states_path(attestation["id"])
    for raw in ("@@@", "aGVsbG8", base64.b64encode(b"x" * 63).decode()):
        headers = _signed_headers("GET", path, b"")
        headers["X-PS"] = raw
        resp = client.get(path, headers=headers)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["code"] == "validation_error"


def test_signature_binds_method_path_empty_body_and_timestamp(client):
    attestation = _world(client)
    assert _get_states(
        client, attestation["id"], signed_method="POST"
    ).status_code == 404
    path = _states_path(attestation["id"])
    assert _get_states(
        client, attestation["id"], signed_path=path + "/"
    ).status_code == 404
    assert _get_states(
        client, attestation["id"], signed_body=b"x"
    ).status_code == 404
    sent = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    signed = (
        datetime.now(timezone.utc) - timedelta(seconds=5)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert _get_states(
        client, attestation["id"],
        timestamp=sent, signed_timestamp=signed,
    ).status_code == 404


def test_signed_path_excludes_the_query_string(client):
    attestation = _world(client)
    path = _states_path(attestation["id"])
    headers = _signed_headers("GET", path, b"")
    assert client.get(
        path, params={"limit": "1", "state": "active"}, headers=headers
    ).status_code == 200
    assert client.get(path + "?limit=1", headers=headers).status_code == 200


# --- Read-only guarantees ------------------------------------------------------


def test_listing_writes_no_resources_or_audit(client, db_session):
    attestation = _world(client)
    grant = _grant(client, attestation["id"])
    _schedule_expiry(client, grant["id"], seconds=3600)
    events_before = _audit_count(db_session)
    grants_before = len(
        db_session.execute(select(AttestationAccessGrant)).scalars().all()
    )

    path = _states_path(attestation["id"])
    responses = [
        _get_states(client, attestation["id"]),
        _get_states(client, attestation["id"], params={"state": "active"}),
        _get_states(client, attestation["id"], params={"limit": "1"}),
        _get_states(
            client, attestation["id"], actor="org-3", seed=SEED_C
        ),
        _get_states(client, "att_ghost"),
        client.get(path),
        client.get(path, params={"limit": "0"}),
        client.request("GET", path, content=b"{}"),
    ]
    for resp in responses:
        assert resp.status_code in (200, 404, 422), resp.text

    db_session.expire_all()
    assert _audit_count(db_session) == events_before
    assert len(
        db_session.execute(select(AttestationAccessGrant)).scalars().all()
    ) == grants_before


def test_existing_protected_routes_keep_working(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"])

    states = _get_states(client, attestation["id"])
    assert states.status_code == 200
    grants = client.get(
        f"/v1/attestations/{attestation['id']}/access-grants",
        headers=_signed_headers(
            "GET",
            f"/v1/attestations/{attestation['id']}/access-grants",
            b"",
        ),
    )
    assert grants.status_code == 200
    assert grants.json()["items"][0]["id"] == grant["id"]

    proof_path = f"/v1/protected/attestations/{attestation['id']}"
    assert client.get(
        proof_path, headers=_signed_headers("GET", proof_path, b"")
    ).status_code == 200
