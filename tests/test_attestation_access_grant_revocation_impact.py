"""Tests for the read-only access-grant-revocation impact endpoint.

Covers ``GET /v1/attestation-access-grant-revocations/{revocation_id}/impact``:
the 200 impact view (compact UTF-8 JSON terminated by exactly one newline,
members in revocation, grant, attestation_id, grantee_actor_id, expires_at,
before_state, after_state, changed, effective_at order; the revocation and
grant members reuse their exact existing public views), the before/after
state judgment at the revocation's own instant (only revocations persisted
strictly earlier -- in ``created_at`` then ``seq`` order -- and an expiry
record that already existed at that instant count; later records never
rewrite the historical judgment), ``changed`` true exactly when a readable
before state (``active`` or ``scheduled``) becomes unreadable, the 404
``attestation_access_grant_revocation_not_found`` echoing the requested id,
the 422 ``validation_error`` for any request body or any query parameter
(validated before the lookup), the framework's 405 ``method_not_allowed``
for non-GET methods, the strict no-write guarantee, and restart
consistency.

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance.app import create_app
from provenance.config import Settings
from provenance.ids import (
    attestation_access_grant_expiry_id,
    attestation_access_grant_revocation_id,
)
from provenance.models import (
    AttestationAccessGrant,
    AttestationAccessGrantExpiry,
    AttestationAccessGrantRevocation,
    AuditEvent,
)
from tests.test_attestation_access_grant_expiries import _future, _post_expiry
from tests.test_attestation_access_grant_revocations import (
    REASON_A,
    REASON_B,
    REVOCATIONS_PATH,
    _grant,
    _post_revocation,
    _revocation_body,
    _world,
)

IMPACT_KEYS = [
    "revocation",
    "grant",
    "attestation_id",
    "grantee_actor_id",
    "expires_at",
    "before_state",
    "after_state",
    "changed",
    "effective_at",
]
REVOCATION_KEYS = ["id", "grant_id", "revoker_actor_id", "reason", "created_at"]
GRANT_KEYS = ["id", "attestation_id", "grantee_actor_id", "created_at"]


def _impact_path(revocation_id: str) -> str:
    return f"{REVOCATIONS_PATH}/{revocation_id}/impact"


def _impact(client, revocation_id: str, **kwargs):
    return client.get(_impact_path(revocation_id), **kwargs)


def _revoke(client, grant, reason=REASON_A):
    resp = _post_revocation(client, _revocation_body(grant["id"], reason))
    assert resp.status_code == 201, resp.text
    return resp.json()


def _setup(client, grantee="org-2"):
    """A world with one grant revoked once; returns ``(attestation, grant, revocation)``."""
    attestation = _world(client)
    grant = _grant(client, attestation["id"], grantee)
    revocation = _revoke(client, grant)
    return attestation, grant, revocation


# --- Happy path: exact view, order, and judgment -------------------------------


def test_impact_of_revoked_active_grant(client):
    attestation, grant, revocation = _setup(client)

    resp = _impact(client, revocation["id"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert list(body) == IMPACT_KEYS
    assert list(body["revocation"]) == REVOCATION_KEYS
    assert list(body["grant"]) == GRANT_KEYS
    assert body["revocation"] == revocation
    assert body["grant"] == grant
    assert body["attestation_id"] == attestation["id"] == grant["attestation_id"]
    assert body["grantee_actor_id"] == "org-2" == grant["grantee_actor_id"]
    assert body["expires_at"] is None
    assert body["before_state"] == "active"
    assert body["after_state"] == "revoked"
    assert body["changed"] is True
    assert body["effective_at"] == revocation["created_at"]
    created_at = datetime.fromisoformat(body["effective_at"])
    assert created_at.utcoffset().total_seconds() == 0


def test_impact_response_is_compact_utf8_json_with_one_newline(client):
    _, _, revocation = _setup(client)
    resp = _impact(client, revocation["id"])
    assert resp.status_code == 200
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    for earlier, later in zip(IMPACT_KEYS, IMPACT_KEYS[1:]):
        assert raw.index(f'"{earlier}"'.encode()) < raw.index(
            f'"{later}"'.encode()
        )
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    assert raw == expected


def test_impact_of_revoked_scheduled_grant(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    expiry = _post_expiry(client, grant["id"], _future())
    assert expiry.status_code == 201, expiry.text
    revocation = _revoke(client, grant)

    body = _impact(client, revocation["id"]).json()
    assert body["before_state"] == "scheduled"
    assert body["after_state"] == "revoked"
    assert body["changed"] is True
    # The scheduled expiry is echoed; both render the same UTC instant.
    assert datetime.fromisoformat(body["expires_at"]) == datetime.fromisoformat(
        expiry.json()["expires_at"]
    )


def test_impact_when_expiry_already_passed(client, db_session):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    # The API only schedules future expiries, so an already-reached expiry is
    # inserted directly with a deterministic past instant.
    past = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    db_session.add(
        AttestationAccessGrantExpiry(
            id=attestation_access_grant_expiry_id(grant["id"], past),
            grant_id=grant["id"],
            expires_at=past,
        )
    )
    db_session.commit()
    revocation = _revoke(client, grant)

    body = _impact(client, revocation["id"]).json()
    assert datetime.fromisoformat(body["expires_at"]) == past
    assert body["before_state"] == "expired"
    assert body["after_state"] == "revoked"
    # The grantee could no longer read even before this revocation.
    assert body["changed"] is False


def test_impact_when_earlier_revocation_exists(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    first = _revoke(client, grant, REASON_A)
    second = _revoke(client, grant, REASON_B)

    # The second revocation changes nothing: the grant was already revoked.
    later = _impact(client, second["id"]).json()
    assert later["before_state"] == "revoked"
    assert later["after_state"] == "revoked"
    assert later["changed"] is False
    assert later["effective_at"] == second["created_at"]

    # The later record never rewrites the first revocation's judgment.
    earlier = _impact(client, first["id"]).json()
    assert earlier["before_state"] == "active"
    assert earlier["after_state"] == "revoked"
    assert earlier["changed"] is True


def test_same_instant_revocations_judged_by_persistence_order(client, db_session):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    tie = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)

    def insert(reason: str) -> AttestationAccessGrantRevocation:
        row = AttestationAccessGrantRevocation(
            id=attestation_access_grant_revocation_id(
                grant["id"], "org-1", reason
            ),
            grant_id=grant["id"],
            revoker_actor_id="org-1",
            reason=reason,
            created_at=tie,
        )
        db_session.add(row)
        return row

    first = insert(REASON_A)
    second = insert(REASON_B)
    db_session.commit()

    # Same instant: the monotonic seq (insertion order) decides which record
    # is "earlier"; the first-persisted revocation is the one that changed
    # the grantee's access.
    first_view = _impact(client, first.id).json()
    assert first_view["before_state"] == "active"
    assert first_view["changed"] is True
    second_view = _impact(client, second.id).json()
    assert second_view["before_state"] == "revoked"
    assert second_view["after_state"] == "revoked"
    assert second_view["changed"] is False
    assert first_view["effective_at"] == second_view["effective_at"]


def test_expiry_scheduled_after_the_revocation_does_not_rewrite_it(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    revocation = _revoke(client, grant)
    before = _impact(client, revocation["id"]).json()
    assert before["before_state"] == "active"
    assert before["expires_at"] is None

    # An expiry scheduled only after the revocation did not exist at the
    # revocation's instant: the historical judgment stands, while the
    # grant's now-current scheduled expiry is still reported.
    expiry = _post_expiry(client, grant["id"], _future())
    assert expiry.status_code == 201, expiry.text
    after = _impact(client, revocation["id"]).json()
    assert after["before_state"] == "active"
    assert after["after_state"] == "revoked"
    assert after["changed"] is True
    assert datetime.fromisoformat(after["expires_at"]) == datetime.fromisoformat(
        expiry.json()["expires_at"]
    )


def test_impact_is_isolated_to_the_named_revocations_grant(client):
    attestation = _world(client)
    grant_org2 = _grant(client, attestation["id"], "org-2")
    grant_org3 = _grant(client, attestation["id"], "org-3")
    revocation_org3 = _revoke(client, grant_org3)

    body = _impact(client, revocation_org3["id"]).json()
    assert body["grant"] == grant_org3
    assert body["grantee_actor_id"] == "org-3"
    # The other grantee's unrevoked grant is irrelevant to this judgment.
    assert body["before_state"] == "active"
    assert body["changed"] is True
    assert grant_org2["id"] != body["grant"]["id"]


# --- Unknown revocation: explicit 404, never a reverse lookup -------------------


def test_unknown_revocation_is_a_specific_404(client):
    _, _, revocation = _setup(client)
    ghost = "agr_" + "0" * 64
    resp = _impact(client, ghost)
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "attestation_access_grant_revocation_not_found"
    assert error["details"]["revocation_id"] == ghost

    # The identifier match is verbatim: a real id with different casing or
    # surrounding whitespace is equally unknown.
    for variant in (revocation["id"].upper(), " " + revocation["id"]):
        resp = _impact(client, variant)
        assert resp.status_code == 404
        assert resp.json()["error"]["details"]["revocation_id"] == variant


def test_impact_never_reverse_looks_up_by_grant_or_attestation_id(client):
    attestation, grant, _ = _setup(client)
    for identifier in (grant["id"], attestation["id"]):
        resp = _impact(client, identifier)
        assert resp.status_code == 404, identifier
        assert (
            resp.json()["error"]["code"]
            == "attestation_access_grant_revocation_not_found"
        )
        assert resp.json()["error"]["details"]["revocation_id"] == identifier


# --- Request-shape validation, before the lookup --------------------------------


def test_any_request_body_is_422_before_the_lookup(client):
    _, _, revocation = _setup(client)
    for body in (b"{}", b" ", b"null", b"\x00"):
        for identifier in (revocation["id"], "agr_ghost"):
            resp = client.request(
                "GET", _impact_path(identifier), content=body
            )
            assert resp.status_code == 422, (body, identifier)
            assert resp.json()["error"]["code"] == "validation_error"


def test_any_query_parameter_is_422_before_the_lookup(client):
    _, _, revocation = _setup(client)
    for query in ("?unknown=1", "?revocation_id=x", "?blank=", "?x=1&x=2"):
        for identifier in (revocation["id"], "agr_ghost"):
            resp = client.get(_impact_path(identifier) + query)
            assert resp.status_code == 422, (query, identifier)
            assert resp.json()["error"]["code"] == "validation_error"

    # Sanity: the same resources answer 200/404 with no query string.
    assert _impact(client, revocation["id"]).status_code == 200
    assert _impact(client, "agr_ghost").status_code == 404


def test_non_get_methods_are_405(client):
    _, _, revocation = _setup(client)
    path = _impact_path(revocation["id"])
    for method in ("post", "put", "patch", "delete"):
        resp = getattr(client, method)(path)
        assert resp.status_code == 405, method
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Read-only guarantee, stability, and persistence ----------------------------


def test_impact_reads_never_create_or_modify_anything(client, db_session):
    _, grant, revocation = _setup(client)

    def _counts():
        db_session.expire_all()
        return (
            db_session.scalar(
                select(func.count()).select_from(AttestationAccessGrant)
            ),
            db_session.scalar(
                select(func.count()).select_from(AttestationAccessGrantRevocation)
            ),
            db_session.scalar(
                select(func.count()).select_from(AttestationAccessGrantExpiry)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    before = _counts()
    responses = (
        _impact(client, revocation["id"]),
        _impact(client, revocation["id"]),
        _impact(client, "agr_ghost"),
        client.request("GET", _impact_path(revocation["id"]), content=b"{}"),
        client.get(_impact_path(revocation["id"]) + "?x=1"),
        client.get(_impact_path("agr_ghost") + "?x=1"),
    )
    assert [r.status_code for r in responses] == [200, 200, 404, 422, 422, 422]
    assert _counts() == before

    # The existing rows are byte-for-byte unchanged.
    db_session.expire_all()
    row = db_session.execute(
        select(AttestationAccessGrantRevocation).where(
            AttestationAccessGrantRevocation.id == revocation["id"]
        )
    ).scalar_one()
    assert (row.id, row.grant_id, row.revoker_actor_id, row.reason) == (
        revocation["id"],
        grant["id"],
        "org-1",
        REASON_A,
    )


def test_impact_is_byte_for_byte_stable_across_reads(client):
    _, _, revocation = _setup(client)
    first = _impact(client, revocation["id"])
    assert first.status_code == 200
    for _ in range(3):
        repeat = _impact(client, revocation["id"])
        assert repeat.status_code == 200
        assert repeat.content == first.content


def test_impact_view_carries_no_sensitive_material(client):
    _, _, revocation = _setup(client)
    body = _impact(client, revocation["id"]).json()
    # Exactly the nine public members; the nested views carry exactly their
    # existing public fields -- no private key, raw signature,
    # authentication header, claim payload, content, or evidence byte.
    assert set(body) == set(IMPACT_KEYS)
    assert set(body["revocation"]) == set(REVOCATION_KEYS)
    assert set(body["grant"]) == set(GRANT_KEYS)
    assert body["before_state"] in {"active", "scheduled", "expired", "revoked"}
    assert body["after_state"] in {"active", "scheduled", "expired", "revoked"}


def test_impact_survives_restart(tmp_db_url):
    app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app) as client:
        _, _, revocation = _setup(client)
        first = _impact(client, revocation["id"])
        assert first.status_code == 200

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        again = _impact(client, revocation["id"])
        assert again.status_code == 200
        assert again.content == first.content
        assert again.json()["changed"] is True
