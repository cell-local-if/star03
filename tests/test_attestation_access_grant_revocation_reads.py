"""Read-only tracking tests for attestation access-grant revocations.

Covers the two reviewer read routes added on top of the existing immutable
write route:

* ``GET /v1/attestation-access-grant-revocations/{revocation_id}`` -- one
  existing revocation's public view, or an explicit 404 for an unknown id;
* ``GET /v1/attestation-access-grants/{grant_id}/revocations`` -- only that
  existing grant's revocation records, in stable creation order as
  ``{"items", "count"}``; an unknown grant is 422 ``grant_not_found`` and an
  existing grant without revocations is an empty collection.

The tests pin down, deterministically and offline:

* the exact five public fields per record (``id``, ``grant_id``,
  ``revoker_actor_id``, ``reason``, UTC ``created_at``) and the strict
  ``{"items", "count"}`` collection envelope -- no private key, raw
  signature, authentication header, claim payload, content, or evidence byte
  can appear;
* lookup isolation: the individual route keys solely on the revocation id
  (a grant/attestation id is never reverse-resolved) and the collection
  routes solely on the grant id (an attestation id, a revocation id, or any
  other field never resolves it);
* stable creation ordering (``created_at`` first, monotonic ``seq`` as the
  tiebreaker), including rows inserted with identical or out-of-insertion
  timestamps;
* parameter precedence: any, blank, or repeated query parameter is a 422
  validated *before* the resource lookup, so an unknown id plus a bad
  parameter is 422 rather than 404/empty;
* strict zero-write behavior for successful, empty, missing, and
  parameter-invalid reads -- no grant, revocation, other resource, or audit
  event is created or modified;
* no credentials are required to read, only GET is served, and the views
  persist unchanged across an app restart.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance.ids import attestation_access_grant_revocation_id
from provenance.models import (
    AttestationAccessGrant,
    AttestationAccessGrantRevocation,
    AuditEvent,
)
from tests.test_attestation_access_grant_revocations import (
    GRANTS_PATH,
    REASON_A,
    REASON_B,
    REVOCATIONS_PATH,
    _grant,
    _post_revocation,
    _revocation_body,
    _world,
)

PUBLIC_FIELDS = {"id", "grant_id", "revoker_actor_id", "reason", "created_at"}


def _revocation_url(revocation_id: str) -> str:
    return f"{REVOCATIONS_PATH}/{revocation_id}"


def _grant_revocations_url(grant_id: str) -> str:
    return f"{GRANTS_PATH}/{grant_id}/revocations"


def _assert_public_item(item: dict) -> None:
    assert set(item) == PUBLIC_FIELDS
    assert item["id"].startswith("agr_")
    created_at = datetime.fromisoformat(item["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert item["created_at"].endswith(("Z", "+00:00"))


def _revoke(client, grant, reason=REASON_A):
    resp = _post_revocation(client, _revocation_body(grant["id"], reason))
    assert resp.status_code == 201, resp.text
    return resp.json()


# --- Individual resource: exact public view -----------------------------------


def test_get_revocation_returns_the_existing_public_view(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _revoke(client, grant)

    resp = client.get(_revocation_url(created["id"]))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == created
    _assert_public_item(body)
    assert body["grant_id"] == grant["id"]
    assert body["revoker_actor_id"] == "org-1"
    assert body["reason"] == REASON_A


def test_get_revocation_requires_no_authentication_headers(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _revoke(client, grant)
    # Reviewer reads are public: no X-PA/X-PT/X-PS headers are sent.
    resp = client.get(_revocation_url(created["id"]))
    assert resp.status_code == 200
    assert resp.json() == created


def test_get_revocation_is_byte_for_byte_stable_across_reads(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _revoke(client, grant)
    first = client.get(_revocation_url(created["id"])).json()
    for _ in range(3):
        assert client.get(_revocation_url(created["id"])).json() == first


# --- Individual resource: explicit 404, never a reverse lookup ----------------


def test_get_unknown_revocation_is_an_explicit_404(client):
    resp = client.get(_revocation_url("agr_does_not_exist"))
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "attestation_access_grant_revocation_not_found"
    assert error["details"]["revocation_id"] == "agr_does_not_exist"


def test_get_never_reverse_looks_up_by_grant_or_attestation_id(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    _revoke(client, grant)
    # Existing identifiers of other resource types are unknown *revocation*
    # ids: the route keys on the revocation id alone and never resolves a
    # grant id or attestation id back to the revocation record.
    for identifier in (grant["id"], attestation["id"]):
        resp = client.get(_revocation_url(identifier))
        assert resp.status_code == 404, identifier
        assert (
            resp.json()["error"]["code"]
            == "attestation_access_grant_revocation_not_found"
        )
        assert resp.json()["error"]["details"]["revocation_id"] == identifier


# --- Collection: empty, exact envelope, and full views ------------------------


def test_list_for_existing_grant_without_revocations_is_empty(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    resp = client.get(_grant_revocations_url(grant["id"]))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0}


def test_list_requires_no_authentication_headers(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    assert client.get(_grant_revocations_url(grant["id"])).status_code == 200


def test_list_returns_the_grants_records_with_exact_envelope_and_fields(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    first = _revoke(client, grant, REASON_A)
    second = _revoke(client, grant, REASON_B)

    resp = client.get(_grant_revocations_url(grant["id"]))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"items", "count"}
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [first["id"], second["id"]]
    for item in body["items"]:
        _assert_public_item(item)
        assert item["grant_id"] == grant["id"]
    assert body["items"] == [first, second]


# --- Collection: isolation across grants --------------------------------------


def test_listing_is_isolated_to_the_named_grant(client):
    attestation = _world(client)
    # Two distinct grantees on the same proof => two independent grants.
    grant_to_org2 = _grant(client, attestation["id"], "org-2")
    grant_to_org3 = _grant(client, attestation["id"], "org-3")
    rev_org2 = _revoke(client, grant_to_org2, REASON_A)
    rev_org3_a = _revoke(client, grant_to_org3, REASON_A)
    rev_org3_b = _revoke(client, grant_to_org3, REASON_B)

    org2 = client.get(_grant_revocations_url(grant_to_org2["id"])).json()
    org3 = client.get(_grant_revocations_url(grant_to_org3["id"])).json()
    assert [i["id"] for i in org2["items"]] == [rev_org2["id"]]
    assert org2["count"] == 1
    assert [i["id"] for i in org3["items"]] == [rev_org3_a["id"], rev_org3_b["id"]]
    assert org3["count"] == 2

    # Every listed record is individually readable and carries its own
    # grant_id; no record ever appears under the wrong grant.
    for item in org2["items"]:
        assert item["grant_id"] == grant_to_org2["id"]
        assert client.get(_revocation_url(item["id"])).json() == item
    for item in org3["items"]:
        assert item["grant_id"] == grant_to_org3["id"]
        assert client.get(_revocation_url(item["id"])).json() == item


def test_idempotent_revocation_retry_adds_nothing_to_the_listing(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _revoke(client, grant)
    for _ in range(2):
        assert _post_revocation(client, _revocation_body(grant["id"])).status_code == 200
    body = client.get(_grant_revocations_url(grant["id"])).json()
    assert body["count"] == 1
    assert [i["id"] for i in body["items"]] == [created["id"]]


# --- Collection: unknown grant is 422, never an empty set or reverse lookup ---


def test_list_for_unknown_grant_is_422_validation_error(client):
    _world(client)
    resp = client.get(_grant_revocations_url("aag_ghost"))
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"]["reason"] == "grant_not_found"


def test_listing_does_not_resolve_attestation_or_revocation_ids(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _revoke(client, grant)
    # An existing attestation id and an existing revocation id are both
    # unknown *grant* ids: they must 422, never resolve through another
    # field (e.g. the revocation's grant_id) to a collection.
    for identifier in (attestation["id"], created["id"], "att_ghost"):
        resp = client.get(_grant_revocations_url(identifier))
        assert resp.status_code == 422, identifier
        assert resp.json()["error"]["code"] == "validation_error"
        assert resp.json()["error"]["details"]["reason"] == "grant_not_found"


# --- Query-parameter boundary and lookup precedence ---------------------------


_QUERY_CASES = (
    "?x=1",
    "?unknown=",
    "?grant_id=aag_x",
    "?revoker_actor_id=org-1",
    "?reason=x",
    "?a=1&b=2",
    "?a=1&a=2",
)


def test_individual_read_rejects_any_query_parameter_before_lookup(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _revoke(client, grant)

    for query in _QUERY_CASES:
        known = client.get(_revocation_url(created["id"]) + query)
        assert known.status_code == 422, query
        assert known.json()["error"]["code"] == "validation_error"
        # Parameter validation precedes the lookup: an unknown id with a bad
        # parameter is 422, never the 404 the unknown id alone would yield.
        unknown = client.get(_revocation_url("agr_ghost") + query)
        assert unknown.status_code == 422, query
        assert unknown.json()["error"]["code"] == "validation_error"

    # Sanity: the same resources answer 200/404 with no query string.
    assert client.get(_revocation_url(created["id"])).status_code == 200
    assert client.get(_revocation_url("agr_ghost")).status_code == 404


def test_collection_read_rejects_any_query_parameter_before_lookup(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    _revoke(client, grant)

    for query in _QUERY_CASES:
        existing = client.get(_grant_revocations_url(grant["id"]) + query)
        assert existing.status_code == 422, query
        assert existing.json()["error"]["code"] == "validation_error"
        # Validation precedes the grant lookup: an unknown grant plus a bad
        # parameter is 422, never the grant_not_found 422 it yields bare.
        unknown = client.get(_grant_revocations_url("aag_ghost") + query)
        assert unknown.status_code == 422, query
        assert unknown.json()["error"]["code"] == "validation_error"

    assert client.get(_grant_revocations_url(grant["id"])).status_code == 200
    assert client.get(_grant_revocations_url("aag_ghost")).status_code == 422


# --- Only GET is served -------------------------------------------------------


def test_read_routes_reject_non_get_methods(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _revoke(client, grant)
    single = _revocation_url(created["id"])
    collection = _grant_revocations_url(grant["id"])
    for url in (single, collection):
        for method, kwargs in (
            ("put", {"json": {}}),
            ("patch", {"json": {}}),
            ("delete", {}),
        ):
            resp = getattr(client, method)(url, **kwargs)
            assert resp.status_code == 405, (method, url)
            assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Stable creation ordering, including the seq tiebreaker -------------------


def test_listing_orders_by_created_at_with_seq_tiebreaker(client, db_session):
    # Rows are inserted directly so timestamps are fully deterministic: two
    # rows share one instant (the monotonic seq is the tiebreaker) and a
    # third row carries an *earlier* timestamp despite being inserted last
    # (higher seq), proving created_at is the primary key and seq only breaks
    # ties -- the ordering is stable creation order, not insertion accident.
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")

    earlier = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    tie = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)

    def insert(reason: str, created_at: datetime):
        row = AttestationAccessGrantRevocation(
            id=attestation_access_grant_revocation_id(
                grant["id"], "org-1", reason
            ),
            grant_id=grant["id"],
            revoker_actor_id="org-1",
            reason=reason,
            created_at=created_at,
        )
        db_session.add(row)
        return row

    tie_a = insert("tie-a", tie)
    tie_b = insert("tie-b", tie)
    early = insert("earlier-instant", earlier)
    db_session.commit()

    body = client.get(_grant_revocations_url(grant["id"])).json()
    # Earlier instant first despite the highest seq; the two tied instants
    # follow in seq (insertion) order.
    assert [i["id"] for i in body["items"]] == [
        early.id,
        tie_a.id,
        tie_b.id,
    ]
    assert body["count"] == 3
    assert [i["reason"] for i in body["items"]] == [
        "earlier-instant",
        "tie-a",
        "tie-b",
    ]

    # Each deterministically-identified row is individually readable.
    for item in body["items"]:
        assert client.get(_revocation_url(item["id"])).json() == item


# --- Strict zero-write guarantees ----------------------------------------------


def test_reads_never_create_or_modify_anything(client, db_session):
    attestation = _world(client)
    grant_with = _grant(client, attestation["id"], "org-2")
    grant_without = _grant(client, attestation["id"], "org-3")
    created = _revoke(client, grant_with)
    original_reason = created["reason"]

    db_session.expire_all()
    grants_before = len(
        db_session.execute(select(AttestationAccessGrant)).scalars().all()
    )
    revocations_before = db_session.scalar(
        select(func.count()).select_from(AttestationAccessGrantRevocation)
    )
    audits_before = db_session.scalar(select(func.count()).select_from(AuditEvent))

    responses = (
        # Successful individual and collection reads.
        client.get(_revocation_url(created["id"])),
        client.get(_grant_revocations_url(grant_with["id"])),
        # Empty collection for an existing, never-revoked grant.
        client.get(_grant_revocations_url(grant_without["id"])),
        # Missing resources.
        client.get(_revocation_url("agr_ghost")),
        client.get(_grant_revocations_url("aag_ghost")),
        # Parameter failures (single and collection, known and unknown).
        client.get(_revocation_url(created["id"]) + "?x=1"),
        client.get(_revocation_url("agr_ghost") + "?x=1&x=2"),
        client.get(_grant_revocations_url(grant_with["id"]) + "?x=1"),
        client.get(_grant_revocations_url("aag_ghost") + "?x=1"),
    )
    assert [r.status_code for r in responses] == [
        200,
        200,
        200,
        404,
        422,
        422,
        422,
        422,
        422,
    ]

    db_session.expire_all()
    assert len(
        db_session.execute(select(AttestationAccessGrant)).scalars().all()
    ) == grants_before
    assert db_session.scalar(
        select(func.count()).select_from(AttestationAccessGrantRevocation)
    ) == revocations_before
    assert db_session.scalar(
        select(func.count()).select_from(AuditEvent)
    ) == audits_before

    # The existing rows are byte-for-byte unchanged.
    row = db_session.execute(
        select(AttestationAccessGrantRevocation).where(
            AttestationAccessGrantRevocation.id == created["id"]
        )
    ).scalar_one()
    assert (row.id, row.grant_id, row.revoker_actor_id, row.reason) == (
        created["id"],
        grant_with["id"],
        "org-1",
        original_reason,
    )
    grant_row = db_session.execute(
        select(AttestationAccessGrant).where(
            AttestationAccessGrant.id == grant_with["id"]
        )
    ).scalar_one()
    assert grant_row.grantee_actor_id == "org-2"


# --- Persistence across restarts ----------------------------------------------


def test_read_views_persist_unchanged_across_a_restart(tmp_db_url, file_client):
    attestation = _world(file_client)
    grant = _grant(file_client, attestation["id"], "org-2")
    created = _revoke(file_client, grant, REASON_A)
    second = _revoke(file_client, grant, REASON_B)

    from provenance.app import create_app
    from provenance.config import Settings

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        single = client.get(_revocation_url(created["id"]))
        assert single.status_code == 200
        assert single.json() == created

        listing = client.get(_grant_revocations_url(grant["id"]))
        assert listing.status_code == 200
        assert listing.json() == {
            "items": [created, second],
            "count": 2,
        }
        # An unknown grant stays 422 after the restart.
        assert client.get(_grant_revocations_url("aag_ghost")).status_code == 422

        import sqlite3

        con = sqlite3.connect(tmp_db_url.removeprefix("sqlite:///"))
        rows = con.execute(
            "SELECT id, grant_id, revoker_actor_id, reason "
            "FROM attestation_access_grant_revocations "
            "ORDER BY seq ASC"
        ).fetchall()
        grant_revoked_audits = con.execute(
            "SELECT COUNT(*) FROM audit_events "
            "WHERE event_type = 'attestation.access_grant_revoked'"
        ).fetchone()[0]
        con.close()
        assert rows == [
            (created["id"], grant["id"], "org-1", REASON_A),
            (second["id"], grant["id"], "org-1", REASON_B),
        ]
        # The reads added no audit rows: exactly the two create events.
        assert grant_revoked_audits == 2
