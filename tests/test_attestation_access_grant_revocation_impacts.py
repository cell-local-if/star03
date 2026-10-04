"""Tests for the read-only access-grant-revocation impact query.

Covers ``GET /v1/attestation-access-grant-revocations/{revocation_id}/impact``:

* the success body carries exactly ``revocation``, ``grant``,
  ``attestation_id``, ``grantee_actor_id``, ``expires_at``, ``before_state``,
  ``after_state``, ``changed``, ``effective_at`` in that fixed member order,
  where ``revocation`` and ``grant`` are exactly the existing public views,
  ``expires_at`` is the scheduled expiry as it stood when the revocation was
  persisted (null when none had been scheduled by then), and ``effective_at``
  is the revocation's own ``created_at``;
* the state classification at the instants immediately before and after the
  revocation: ``active`` (no expiry), ``scheduled`` (future expiry),
  ``expired`` (expiry reached), ``revoked`` (an earlier revocation exists);
  ``active``/``scheduled`` still authorize the read, and ``changed`` is true
  only when the revocation flipped the grant from readable to unreadable;
* history is pinned to the revocation's persistence point: an earlier
  revocation or an already-reached expiry keeps ``changed`` false, a later
  revocation or a later-scheduled expiry never rewrites the determination,
  and records sharing the instant are ordered by persistence (``seq``);
* an unknown revocation id is an explicit 404
  ``attestation_access_grant_revocation_not_found`` echoing the requested id;
  any request body bytes or any query parameter is a 422 validated *before*
  the lookup, and non-GET methods are 405;
* successful, repeated, and failed reads write no resource or audit event,
  and the view never carries a private key, raw signature, authentication
  header, claim payload, content, or evidence byte.

All tests are deterministic and offline.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from sqlalchemy import func, select

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
from tests.test_attestation_access_grant_expiries import (
    _future,
    _post_expiry,
)
from tests.test_attestation_access_grant_revocations import (
    REASON_A,
    REASON_B,
    REVOCATIONS_PATH,
    _grant,
    _post_revocation,
    _revocation_body,
    _world,
)

IMPACT_FIELDS = [
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

REVOCATION_PUBLIC_FIELDS = {
    "id",
    "grant_id",
    "revoker_actor_id",
    "reason",
    "created_at",
}
GRANT_PUBLIC_FIELDS = {"id", "attestation_id", "grantee_actor_id", "created_at"}


def _impact_url(revocation_id: str) -> str:
    return f"{REVOCATIONS_PATH}/{revocation_id}/impact"


def _revoke(client, grant, reason=REASON_A):
    resp = _post_revocation(client, _revocation_body(grant["id"], reason))
    assert resp.status_code == 201, resp.text
    return resp.json()


def _get_impact(client, revocation_id):
    resp = client.get(_impact_url(revocation_id))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Exactly the nine declared members, in the fixed declared order.
    assert list(body) == IMPACT_FIELDS
    assert set(body["revocation"]) == REVOCATION_PUBLIC_FIELDS
    assert set(body["grant"]) == GRANT_PUBLIC_FIELDS
    return body


# --- Success shape: exact members, fixed order, reused public views -----------


def test_impact_of_plain_revocation_is_active_to_revoked(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _revoke(client, grant)

    body = _get_impact(client, created["id"])
    assert body["revocation"] == created
    assert body["grant"] == grant
    assert body["attestation_id"] == attestation["id"]
    assert body["grantee_actor_id"] == "org-2"
    assert body["expires_at"] is None
    assert body["before_state"] == "active"
    assert body["after_state"] == "revoked"
    assert body["changed"] is True
    assert body["effective_at"] == created["created_at"]


def test_impact_requires_no_authentication_headers(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _revoke(client, grant)
    # Reviewer reads are public: no X-PA/X-PT/X-PS headers are sent.
    assert client.get(_impact_url(created["id"])).status_code == 200


def test_impact_is_byte_for_byte_stable_across_reads(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _revoke(client, grant)
    first = client.get(_impact_url(created["id"])).text
    for _ in range(3):
        assert client.get(_impact_url(created["id"])).text == first


def test_impact_views_match_the_existing_public_read_views(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _revoke(client, grant)

    body = _get_impact(client, created["id"])
    # The nested views are byte-identical to the existing read routes'.
    assert body["revocation"] == client.get(
        f"{REVOCATIONS_PATH}/{created['id']}"
    ).json()
    assert body["grant"] == grant


# --- State classification ------------------------------------------------------


def test_scheduled_expiry_makes_before_state_scheduled_and_still_changed(
    client,
):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    expiry = _post_expiry(client, grant["id"], _future())
    assert expiry.status_code == 201, expiry.text
    created = _revoke(client, grant)

    body = _get_impact(client, created["id"])
    assert body["expires_at"] == expiry.json()["expires_at"]
    assert body["before_state"] == "scheduled"
    assert body["after_state"] == "revoked"
    assert body["changed"] is True


def test_expiry_already_reached_makes_changed_false(client, db_session):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    # The write route only schedules future expiries, so the reached expiry
    # is inserted directly: persisted yesterday, elapsed an hour ago.
    now = datetime.now(timezone.utc)
    expires_at = now - timedelta(hours=1)
    db_session.add(
        AttestationAccessGrantExpiry(
            id=attestation_access_grant_expiry_id(grant["id"], expires_at),
            grant_id=grant["id"],
            expires_at=expires_at,
            created_at=now - timedelta(days=1),
        )
    )
    db_session.commit()
    created = _revoke(client, grant)

    body = _get_impact(client, created["id"])
    assert body["expires_at"] is not None
    assert datetime.fromisoformat(body["expires_at"]) == expires_at
    assert body["before_state"] == "expired"
    assert body["after_state"] == "revoked"
    assert body["changed"] is False


def test_earlier_revocation_makes_changed_false_but_record_is_kept(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    first = _revoke(client, grant, REASON_A)
    second = _revoke(client, grant, REASON_B)

    # The second revocation changed nothing: the grant was already revoked.
    body = _get_impact(client, second["id"])
    assert body["revocation"] == second
    assert body["before_state"] == "revoked"
    assert body["after_state"] == "revoked"
    assert body["changed"] is False

    # The later record never rewrites the first revocation's determination.
    first_body = _get_impact(client, first["id"])
    assert first_body["before_state"] == "active"
    assert first_body["after_state"] == "revoked"
    assert first_body["changed"] is True


def test_later_scheduled_expiry_never_rewrites_the_historical_impact(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _revoke(client, grant)
    # The expiry is scheduled only after the revocation was persisted: it is
    # invisible to the revocation's historical determination.
    expiry = _post_expiry(client, grant["id"], _future())
    assert expiry.status_code == 201, expiry.text

    body = _get_impact(client, created["id"])
    assert body["expires_at"] is None
    assert body["before_state"] == "active"
    assert body["after_state"] == "revoked"
    assert body["changed"] is True


def test_same_instant_revocations_are_ordered_by_persistence(client, db_session):
    # Two revocations of one grant share a created_at instant: the lower seq
    # (persisted first) flipped the grant, the higher seq found it revoked.
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    instant = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)

    def insert(reason: str):
        row = AttestationAccessGrantRevocation(
            id=attestation_access_grant_revocation_id(
                grant["id"], "org-1", reason
            ),
            grant_id=grant["id"],
            revoker_actor_id="org-1",
            reason=reason,
            created_at=instant,
        )
        db_session.add(row)
        return row

    first = insert("tie-first")
    second = insert("tie-second")
    db_session.commit()

    first_body = _get_impact(client, first.id)
    assert first_body["before_state"] == "active"
    assert first_body["after_state"] == "revoked"
    assert first_body["changed"] is True
    assert datetime.fromisoformat(first_body["effective_at"]) == instant

    second_body = _get_impact(client, second.id)
    assert second_body["before_state"] == "revoked"
    assert second_body["after_state"] == "revoked"
    assert second_body["changed"] is False


def test_impacts_are_isolated_between_grants(client):
    attestation = _world(client)
    grant_org2 = _grant(client, attestation["id"], "org-2")
    grant_org3 = _grant(client, attestation["id"], "org-3")
    created = _revoke(client, grant_org2)

    body = _get_impact(client, created["id"])
    assert body["grant"] == grant_org2
    assert body["grantee_actor_id"] == "org-2"
    # The other grant is untouched by this revocation.
    assert grant_org3["id"] != body["grant"]["id"]


# --- Unknown id: explicit 404, echoing the request value -----------------------


def test_unknown_revocation_is_an_explicit_404(client):
    resp = client.get(_impact_url("agr_does_not_exist"))
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "attestation_access_grant_revocation_not_found"
    assert error["details"]["revocation_id"] == "agr_does_not_exist"


def test_impact_never_reverse_looks_up_other_identifiers(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    _revoke(client, grant)
    # Existing grant and attestation ids are unknown *revocation* ids.
    for identifier in (grant["id"], attestation["id"]):
        resp = client.get(_impact_url(identifier))
        assert resp.status_code == 404, identifier
        assert (
            resp.json()["error"]["code"]
            == "attestation_access_grant_revocation_not_found"
        )
        assert resp.json()["error"]["details"]["revocation_id"] == identifier


# --- Body and query-parameter boundary, validated before the lookup -----------


def test_any_request_body_bytes_are_422_before_the_lookup(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _revoke(client, grant)

    for body in (b"{}", b" ", b"\x00\xff", b"not json"):
        known = client.request("GET", _impact_url(created["id"]), content=body)
        assert known.status_code == 422, body
        assert known.json()["error"]["code"] == "validation_error"
        # Body validation precedes the lookup: an unknown id plus any body
        # bytes is 422, never the 404 the unknown id alone would yield.
        unknown = client.request("GET", _impact_url("agr_ghost"), content=body)
        assert unknown.status_code == 422, body
        assert unknown.json()["error"]["code"] == "validation_error"

    assert client.get(_impact_url(created["id"])).status_code == 200
    assert client.get(_impact_url("agr_ghost")).status_code == 404


_QUERY_CASES = (
    "?x=1",
    "?unknown=",
    "?grant_id=aag_x",
    "?revocation_id=agr_x",
    "?a=1&b=2",
    "?a=1&a=2",
)


def test_any_query_parameter_is_422_before_the_lookup(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _revoke(client, grant)

    for query in _QUERY_CASES:
        known = client.get(_impact_url(created["id"]) + query)
        assert known.status_code == 422, query
        assert known.json()["error"]["code"] == "validation_error"
        # Parameter validation precedes the lookup: an unknown id plus a bad
        # parameter is 422, never the 404 the unknown id alone would yield.
        unknown = client.get(_impact_url("agr_ghost") + query)
        assert unknown.status_code == 422, query
        assert unknown.json()["error"]["code"] == "validation_error"

    assert client.get(_impact_url(created["id"])).status_code == 200
    assert client.get(_impact_url("agr_ghost")).status_code == 404


# --- Only GET is served ---------------------------------------------------------


def test_impact_route_rejects_non_get_methods(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _revoke(client, grant)
    url = _impact_url(created["id"])
    for method, kwargs in (
        ("put", {"json": {}}),
        ("patch", {"json": {}}),
        ("delete", {}),
    ):
        resp = getattr(client, method)(url, **kwargs)
        assert resp.status_code == 405, method
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Strict zero-write guarantees ------------------------------------------------


def test_reads_never_create_or_modify_anything(client, db_session):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _revoke(client, grant)

    db_session.expire_all()
    grants_before = db_session.scalar(
        select(func.count()).select_from(AttestationAccessGrant)
    )
    revocations_before = db_session.scalar(
        select(func.count()).select_from(AttestationAccessGrantRevocation)
    )
    expiries_before = db_session.scalar(
        select(func.count()).select_from(AttestationAccessGrantExpiry)
    )
    audits_before = db_session.scalar(select(func.count()).select_from(AuditEvent))

    responses = (
        # Successful reads, repeated.
        client.get(_impact_url(created["id"])),
        client.get(_impact_url(created["id"])),
        # Missing resource.
        client.get(_impact_url("agr_ghost")),
        # Body and parameter failures, known and unknown ids.
        client.request("GET", _impact_url(created["id"]), content=b"{}"),
        client.request("GET", _impact_url("agr_ghost"), content=b" "),
        client.get(_impact_url(created["id"]) + "?x=1"),
        client.get(_impact_url("agr_ghost") + "?x=1&x=2"),
    )
    assert [r.status_code for r in responses] == [
        200,
        200,
        404,
        422,
        422,
        422,
        422,
    ]

    db_session.expire_all()
    assert db_session.scalar(
        select(func.count()).select_from(AttestationAccessGrant)
    ) == grants_before
    assert db_session.scalar(
        select(func.count()).select_from(AttestationAccessGrantRevocation)
    ) == revocations_before
    assert db_session.scalar(
        select(func.count()).select_from(AttestationAccessGrantExpiry)
    ) == expiries_before
    assert db_session.scalar(
        select(func.count()).select_from(AuditEvent)
    ) == audits_before

    # The existing rows are unchanged.
    row = db_session.execute(
        select(AttestationAccessGrantRevocation).where(
            AttestationAccessGrantRevocation.id == created["id"]
        )
    ).scalar_one()
    assert (row.id, row.grant_id, row.revoker_actor_id, row.reason) == (
        created["id"],
        grant["id"],
        "org-1",
        REASON_A,
    )


# --- Persistence across restarts -------------------------------------------------


def test_impact_view_persists_unchanged_across_a_restart(tmp_db_url, file_client):
    attestation = _world(file_client)
    grant = _grant(file_client, attestation["id"], "org-2")
    created = _revoke(file_client, grant)
    before = file_client.get(_impact_url(created["id"])).json()

    from provenance.app import create_app
    from provenance.config import Settings

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        resp = client.get(_impact_url(created["id"]))
        assert resp.status_code == 200
        assert resp.json() == before
        assert client.get(_impact_url("agr_ghost")).status_code == 404

        import sqlite3

        con = sqlite3.connect(tmp_db_url.removeprefix("sqlite:///"))
        audits = con.execute(
            "SELECT COUNT(*) FROM audit_events "
            "WHERE event_type = 'attestation.access_grant_revoked'"
        ).fetchone()[0]
        con.close()
        # The reads added no audit rows: exactly the one create event.
        assert audits == 1
