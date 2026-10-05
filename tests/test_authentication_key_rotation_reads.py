"""Read-only single-record tests for authentication key rotations.

Covers ``GET /v1/authentication-key-rotations/{rotation_id}`` -- the public
single-record read added on top of the existing frozen rotation write,
retire, and per-subject list routes. The tests pin down, deterministically
and offline:

* the exact six public fields (``id``, ``actor_id``, ``new_public_key``,
  ``active``, ``created_at``, ``retired_at``) in that order -- no private
  key, raw signature, authentication header, payload, or internal sequence
  can appear;
* lookup isolation: the route keys solely on the verbatim rotation id (an
  actor id or public key is never reverse-resolved, and no case, whitespace,
  or Base64 normalization is applied);
* lifecycle truthfulness: an active record reads ``active=true`` with a null
  ``retired_at``; after retirement the same id reads ``active=false`` with a
  non-null UTC ``retired_at``;
* request boundary: any query parameter (unknown, blank, repeated, or one
  the collection routes declare) and any non-empty body is a 422 validated
  *before* the record lookup, so a malformed request never renders as a 404;
* the explicit 404 ``authentication_key_rotation_not_found`` echoing the
  requested id verbatim, with no other error leaking existence;
* strict zero-write behavior for successful, missing, and invalid reads --
  no rotation, other resource, or audit event is created or modified;
* no credentials are required to read, only GET is served, and the view
  persists unchanged across an app restart.
"""

from __future__ import annotations

import json
from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance.models import AuditEvent, AuthenticationKeyRotation
from tests.helpers import SEED_B, ed25519_public_key
from tests.test_authentication_key_rotations import (
    ROTATIONS_PATH,
    SEED_R1,
    SEED_R2,
    _key_b64,
    _post_retire,
    _post_rotation,
    _rotation_body,
    _world,
)

PUBLIC_FIELDS = [
    "id",
    "actor_id",
    "new_public_key",
    "active",
    "created_at",
    "retired_at",
]


def _rotation_url(rotation_id: str) -> str:
    return f"{ROTATIONS_PATH}/{rotation_id}"


def _create(client, key_seed=SEED_R1, actor_id="org-1", **sign_kwargs) -> dict:
    resp = _post_rotation(
        client, _rotation_body(actor_id=actor_id, seed=key_seed), **sign_kwargs
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _assert_public_view(body: dict) -> None:
    # Exactly the six public members, in the declared wire order.
    assert list(body) == PUBLIC_FIELDS
    assert body["id"].startswith("akr_")
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))
    if body["active"]:
        assert body["retired_at"] is None
    else:
        assert body["retired_at"] is not None
        retired_at = datetime.fromisoformat(body["retired_at"])
        assert retired_at.utcoffset().total_seconds() == 0
        assert retired_at >= created_at


# --- Exact public view ----------------------------------------------------------


def test_get_rotation_returns_the_existing_public_view(client):
    _world(client)
    created = _create(client)

    resp = client.get(_rotation_url(created["id"]))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == created
    _assert_public_view(body)
    assert body["actor_id"] == "org-1"
    assert body["new_public_key"] == _key_b64(SEED_R1)
    assert body["active"] is True
    assert body["retired_at"] is None


def test_get_rotation_requires_no_authentication_headers(client):
    _world(client)
    created = _create(client)
    # The read is public: no X-PA/X-PT/X-PS headers are sent or consulted.
    resp = client.get(_rotation_url(created["id"]))
    assert resp.status_code == 200
    assert resp.json() == created


def test_get_rotation_is_byte_for_byte_stable_across_reads(client):
    _world(client)
    created = _create(client)
    first = client.get(_rotation_url(created["id"]))
    assert first.status_code == 200
    for _ in range(3):
        again = client.get(_rotation_url(created["id"]))
        assert again.status_code == 200
        assert again.content == first.content


def test_retired_record_reads_inactive_with_a_utc_retired_at(client):
    _world(client)
    created = _create(client)
    before = client.get(_rotation_url(created["id"])).json()
    assert before["active"] is True and before["retired_at"] is None

    assert _post_retire(client, created["id"], seed=SEED_R1).status_code == 200

    resp = client.get(_rotation_url(created["id"]))
    assert resp.status_code == 200
    body = resp.json()
    _assert_public_view(body)
    assert body["active"] is False
    assert body["retired_at"] is not None
    # Only the lifecycle fields change; identity and key material are stable.
    for field in ("id", "actor_id", "new_public_key", "created_at"):
        assert body[field] == before[field]


def test_each_subjects_record_is_read_under_its_own_id(client):
    _world(client)
    one = _create(client)
    two = _create(client, key_seed=SEED_R2, seed=SEED_R1)
    other_subject = _create(
        client, actor_id="org-2", actor="org-2", seed=SEED_B
    )

    assert client.get(_rotation_url(one["id"])).json() == one
    assert client.get(_rotation_url(two["id"])).json() == two
    assert client.get(_rotation_url(other_subject["id"])).json() == other_subject
    # Same key bytes under a different subject: a distinct record, read only
    # under its own id.
    assert one["new_public_key"] == other_subject["new_public_key"]
    assert one["id"] != other_subject["id"]


# --- Explicit 404, never a reverse lookup or normalization ----------------------


def test_get_unknown_rotation_is_an_explicit_404(client):
    _world(client)
    resp = client.get(_rotation_url("akr_does_not_exist"))
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "authentication_key_rotation_not_found"
    assert error["details"]["rotation_id"] == "akr_does_not_exist"


def test_get_never_reverse_looks_up_by_actor_or_key(client):
    _world(client)
    created = _create(client)
    # An existing actor id and the record's own public key material (hex
    # spelling; the standard Base64 spelling carries "/" and so can never be
    # a single path segment) are both unknown *rotation* ids: the route keys
    # on the rotation id alone and never resolves another field back to the
    # record.
    key_hex = ed25519_public_key(SEED_R1).hex()
    for identifier in ("org-1", key_hex):
        resp = client.get(_rotation_url(identifier))
        assert resp.status_code == 404, identifier
        error = resp.json()["error"]
        assert error["code"] == "authentication_key_rotation_not_found"
        assert error["details"]["rotation_id"] == identifier


def test_get_matches_the_id_verbatim_without_normalization(client):
    _world(client)
    created = _create(client)
    rid = created["id"]
    # Case- and whitespace-sensitive exact match: a differently-cased or
    # padded spelling of an existing id is simply an unknown id, echoed
    # verbatim in the 404 details. The first letter of the "akr_" prefix is
    # always a letter, so the case swap always changes the spelling.
    swapped = rid[0].swapcase() + rid[1:]
    variants = (
        rid.upper(),
        swapped,
        f" {rid}",
        f"{rid} ",
    )
    for variant in variants:
        assert variant != rid
        resp = client.get(_rotation_url(variant))
        assert resp.status_code == 404, variant
        error = resp.json()["error"]
        assert error["code"] == "authentication_key_rotation_not_found"
        assert error["details"]["rotation_id"] == variant
    # The exact spelling still resolves.
    assert client.get(_rotation_url(rid)).status_code == 200


# --- Query-parameter and body boundary, validated before the lookup -------------


_QUERY_CASES = (
    "?x=1",
    "?unknown=",
    "?limit=1",
    "?cursor=abc",
    "?actor_id=org-1",
    "?rotation_id=akr_x",
    "?a=1&b=2",
    "?a=1&a=2",
)


def test_get_rejects_any_query_parameter_before_lookup(client):
    _world(client)
    created = _create(client)

    for query in _QUERY_CASES:
        known = client.get(_rotation_url(created["id"]) + query)
        assert known.status_code == 422, query
        assert known.json()["error"]["code"] == "validation_error"
        # Parameter validation precedes the lookup: an unknown id with a bad
        # parameter is 422, never the 404 the unknown id alone would yield.
        unknown = client.get(_rotation_url("akr_ghost") + query)
        assert unknown.status_code == 422, query
        assert unknown.json()["error"]["code"] == "validation_error"

    # Sanity: the same ids answer 200/404 with no query string.
    assert client.get(_rotation_url(created["id"])).status_code == 200
    assert client.get(_rotation_url("akr_ghost")).status_code == 404


def test_get_rejects_a_non_empty_body_before_lookup(client):
    _world(client)
    created = _create(client)

    for body in (b"{}", b" ", b"{not json", b"\x00"):
        known = client.request(
            "GET", _rotation_url(created["id"]), content=body
        )
        assert known.status_code == 422, body
        assert known.json()["error"]["code"] == "validation_error"
        # Body validation precedes the lookup as well.
        unknown = client.request(
            "GET", _rotation_url("akr_ghost"), content=body
        )
        assert unknown.status_code == 422, body
        assert unknown.json()["error"]["code"] == "validation_error"

    assert client.get(_rotation_url(created["id"])).status_code == 200
    assert client.get(_rotation_url("akr_ghost")).status_code == 404


# --- Only GET is served ----------------------------------------------------------


def test_read_route_rejects_non_get_methods(client):
    _world(client)
    created = _create(client)
    url = _rotation_url(created["id"])
    for method, kwargs in (
        ("post", {"json": {}}),
        ("put", {"json": {}}),
        ("patch", {"json": {}}),
        ("delete", {}),
    ):
        resp = getattr(client, method)(url, **kwargs)
        assert resp.status_code == 405, method
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Strict zero-write guarantees -------------------------------------------------


def test_reads_never_create_or_modify_anything(client, db_session):
    _world(client)
    created = _create(client)
    retired = _create(client, key_seed=SEED_R2, seed=SEED_R1)
    assert _post_retire(client, retired["id"], seed=SEED_R2).status_code == 200

    db_session.expire_all()
    rotations_before = db_session.scalar(
        select(func.count()).select_from(AuthenticationKeyRotation)
    )
    audits_before = db_session.scalar(select(func.count()).select_from(AuditEvent))

    responses = (
        # Successful reads of an active and a retired record.
        client.get(_rotation_url(created["id"])),
        client.get(_rotation_url(retired["id"])),
        # A missing record.
        client.get(_rotation_url("akr_ghost")),
        # Parameter and body failures, on known and unknown ids.
        client.get(_rotation_url(created["id"]) + "?x=1"),
        client.get(_rotation_url("akr_ghost") + "?x=1&x=2"),
        client.request("GET", _rotation_url(created["id"]), content=b"{}"),
        client.request("GET", _rotation_url("akr_ghost"), content=b"{}"),
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
        select(func.count()).select_from(AuthenticationKeyRotation)
    ) == rotations_before
    assert db_session.scalar(
        select(func.count()).select_from(AuditEvent)
    ) == audits_before

    # The existing rows are byte-for-byte unchanged.
    active_row = db_session.execute(
        select(AuthenticationKeyRotation).where(
            AuthenticationKeyRotation.id == created["id"]
        )
    ).scalar_one()
    assert (active_row.actor_id, active_row.active, active_row.retired_at) == (
        "org-1",
        True,
        None,
    )
    retired_row = db_session.execute(
        select(AuthenticationKeyRotation).where(
            AuthenticationKeyRotation.id == retired["id"]
        )
    ).scalar_one()
    assert retired_row.active is False
    assert retired_row.retired_at is not None


# --- Persistence across restarts ----------------------------------------------------


def test_read_view_persists_unchanged_across_a_restart(tmp_db_url, file_client):
    _world(file_client)
    created = _create(file_client)
    retired = _create(file_client, key_seed=SEED_R2, seed=SEED_R1)
    assert (
        _post_retire(file_client, retired["id"], seed=SEED_R2).status_code == 200
    )
    retired_view = file_client.get(_rotation_url(retired["id"])).json()

    import sqlite3

    db_path = tmp_db_url.removeprefix("sqlite:///")
    con = sqlite3.connect(db_path)
    try:
        audits_before = con.execute(
            "SELECT COUNT(*) FROM audit_events"
        ).fetchone()[0]
    finally:
        con.close()

    from provenance.app import create_app
    from provenance.config import Settings

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        active = client.get(_rotation_url(created["id"]))
        assert active.status_code == 200
        assert active.json() == created

        retired_again = client.get(_rotation_url(retired["id"]))
        assert retired_again.status_code == 200
        assert retired_again.json() == retired_view
        assert retired_again.json()["active"] is False
        assert retired_again.json()["retired_at"] is not None

        # An unknown id stays the explicit 404 after the restart.
        missing = client.get(_rotation_url("akr_ghost"))
        assert missing.status_code == 404
        assert (
            missing.json()["error"]["code"]
            == "authentication_key_rotation_not_found"
        )

        con = sqlite3.connect(db_path)
        try:
            rows = con.execute(
                "SELECT id, actor_id, active FROM authentication_key_rotations "
                "ORDER BY seq ASC"
            ).fetchall()
            audits_after = con.execute(
                "SELECT COUNT(*) FROM audit_events"
            ).fetchone()[0]
        finally:
            con.close()
        assert rows == [
            (created["id"], "org-1", 1),
            (retired["id"], "org-1", 0),
        ]
        # The reads added no audit rows.
        assert audits_after == audits_before


def test_success_body_is_compact_json_with_the_six_members_in_order(client):
    _world(client)
    created = _create(client)
    resp = client.get(_rotation_url(created["id"]))
    assert resp.status_code == 200
    assert list(json.loads(resp.content)) == PUBLIC_FIELDS
