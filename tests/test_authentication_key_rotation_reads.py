"""Offline deterministic tests for the public single-rotation read.

Covers ``GET /v1/authentication-key-rotations/{rotation_id}``:

* the success body is exactly the existing rotation public view -- ``id``,
  ``actor_id``, ``new_public_key``, ``active``, ``created_at``,
  ``retired_at``, in that order -- with the stable persisted identifiers,
  the standard-Base64 32-byte Ed25519 public key, and timezone-aware UTC
  timestamps (``retired_at`` null while active, non-null once retired);
* the route is public: no ``X-PA``/``X-PT``/``X-PS`` credentials are
  required, and repeated reads before and after retirement are strictly
  read-only (no record or audit event is created, modified, or deleted);
* the path id is matched exactly as persisted (case- and
  whitespace-sensitive, no Base64 or format normalization, no reverse
  lookup by subject); an unknown id is the
  ``authentication_key_rotation_not_found`` 404 with the requested value
  echoed verbatim in ``error.details.rotation_id``;
* any query parameter (unknown, blank, or repeated) and any non-empty
  body is a 422 ``validation_error`` decided before the record is read;
* POST/PUT/PATCH/DELETE on the same path are 405 ``method_not_allowed``;
* the record's fields survive a restart unchanged, and only the existing
  retirement lifecycle later flips ``active``/``retired_at``.

All fixtures are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures); only fixed seed-derived public keys are used.
"""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import func, select

from provenance.models import AuditEvent, AuthenticationKeyRotation
from tests.test_authentication_key_rotations import (
    SEED_R1,
    SEED_R2,
    _key_b64,
    _post_retire,
    _post_rotation,
    _rotation_body,
    _world,
)

ROTATIONS_PATH = "/v1/authentication-key-rotations"
FIELD_ORDER = [
    "id",
    "actor_id",
    "new_public_key",
    "active",
    "created_at",
    "retired_at",
]


def _read_path(rotation_id: str) -> str:
    return f"{ROTATIONS_PATH}/{rotation_id}"


def _create_rotation(client, seed=SEED_R1):
    resp = _post_rotation(client, _rotation_body(seed=seed))
    assert resp.status_code == 201, resp.text
    return resp.json()


# --- Success view ----------------------------------------------------------------


def test_read_returns_exact_public_view_in_order(client):
    _world(client)
    created = _create_rotation(client)

    resp = client.get(_read_path(created["id"]))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Exactly the six public fields, in the declared order -- never the
    # internal ordering surrogate, a private key, a signature, or headers.
    assert list(body) == FIELD_ORDER
    assert body == created
    assert body["id"] == created["id"]
    assert body["actor_id"] == "org-1"
    assert body["new_public_key"] == _key_b64(SEED_R1)
    assert body["active"] is True
    assert body["retired_at"] is None
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0


def test_read_is_public_and_needs_no_credentials(client):
    _world(client)
    created = _create_rotation(client)
    # No X-PA/X-PT/X-PS headers at all; the default test client sends none.
    resp = client.get(_read_path(created["id"]))
    assert resp.status_code == 200
    assert resp.json() == created


def test_repeated_reads_are_identical_and_write_nothing(client, db_session):
    _world(client)
    created = _create_rotation(client)
    events_before = db_session.execute(
        select(func.count()).select_from(AuditEvent)
    ).scalar_one()

    first = client.get(_read_path(created["id"]))
    for _ in range(3):
        again = client.get(_read_path(created["id"]))
        assert again.status_code == 200
        assert again.json() == first.json()

    assert db_session.execute(
        select(func.count()).select_from(AuditEvent)
    ).scalar_one() == events_before
    rows = db_session.execute(select(AuthenticationKeyRotation)).scalars().all()
    assert [r.id for r in rows] == [created["id"]]


def test_read_after_retirement_reflects_only_the_lifecycle_flip(client):
    _world(client)
    created = _create_rotation(client)
    before = client.get(_read_path(created["id"])).json()
    assert before["active"] is True and before["retired_at"] is None

    retired = _post_retire(client, created["id"])
    assert retired.status_code == 200, retired.text

    after = client.get(_read_path(created["id"]))
    assert after.status_code == 200
    body = after.json()
    assert list(body) == FIELD_ORDER
    assert body["active"] is False
    assert body["retired_at"] is not None
    retired_at = datetime.fromisoformat(body["retired_at"])
    assert retired_at.utcoffset().total_seconds() == 0
    # Retirement changes only the lifecycle fields; identity and key material
    # are exactly as first persisted.
    for field in ("id", "actor_id", "new_public_key", "created_at"):
        assert body[field] == before[field]


def test_read_does_not_reverse_lookup_by_actor(client):
    _world(client)
    created = _create_rotation(client)
    # The subject's actor id is not a rotation id: no inverse resolution.
    resp = client.get(_read_path("org-1"))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == (
        "authentication_key_rotation_not_found"
    )
    assert resp.json()["error"]["details"]["rotation_id"] == "org-1"
    # The real record is still readable by its own id.
    assert client.get(_read_path(created["id"])).status_code == 200


# --- Unknown and near-miss ids -----------------------------------------------------


def test_unknown_id_is_404_and_echoes_the_requested_value(client, db_session):
    _world(client)
    events_before = db_session.execute(
        select(func.count()).select_from(AuditEvent)
    ).scalar_one()

    resp = client.get(_read_path("akr_does_not_exist"))
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "authentication_key_rotation_not_found"
    assert error["details"]["rotation_id"] == "akr_does_not_exist"
    # A failed read writes no record and no audit event.
    assert db_session.execute(
        select(func.count()).select_from(AuditEvent)
    ).scalar_one() == events_before


def test_id_matching_is_exact_case_and_whitespace_sensitive(client):
    _world(client)
    created = _create_rotation(client)
    rid = created["id"]

    variants = {
        "upper": rid.upper(),
        "trailing-space": rid + " ",
        "leading-space": " " + rid,
        "prefix-only": rid[:-1],
    }
    for name, value in variants.items():
        resp = client.get(_read_path(value))
        assert resp.status_code == 404, name
        error = resp.json()["error"]
        assert error["code"] == "authentication_key_rotation_not_found", name
        # The requested value is echoed verbatim, never normalized.
        assert error["details"]["rotation_id"] == value, name
    # The exact persisted id still resolves.
    assert client.get(_read_path(rid)).status_code == 200


def test_404_shape_does_not_distinguish_unknown_from_existing_records(client):
    _world(client)
    created = _create_rotation(client)
    ghost = client.get(_read_path("akr_ghost"))
    near_miss = client.get(_read_path(created["id"].upper()))
    assert ghost.status_code == near_miss.status_code == 404
    assert ghost.json()["error"]["code"] == near_miss.json()["error"]["code"]
    assert ghost.json()["error"]["message"] == near_miss.json()["error"]["message"]


# --- Query-parameter and body validation --------------------------------------------


def test_any_query_parameter_is_422_before_the_lookup(client):
    _world(client)
    created = _create_rotation(client)
    cases = (
        {"limit": "1"},          # a parameter declared by other routes
        {"cursor": "abc"},
        {"actor_id": "org-1"},
        {"unknown": "x"},
        {"blank": ""},
        {"limit": "1", "cursor": "c"},
    )
    for params in cases:
        # 422 for an existing id...
        resp = client.get(_read_path(created["id"]), params=params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"
        # ...and identically for an unknown id: validation precedes the read.
        ghost = client.get(_read_path("akr_ghost"), params=params)
        assert ghost.status_code == 422, params
        assert ghost.json() == resp.json(), params


def test_repeated_query_parameter_is_422(client):
    _world(client)
    created = _create_rotation(client)
    resp = client.get(_read_path(created["id"]) + "?limit=1&limit=2")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_non_empty_body_is_422_before_the_lookup(client, db_session):
    _world(client)
    created = _create_rotation(client)
    events_before = db_session.execute(
        select(func.count()).select_from(AuditEvent)
    ).scalar_one()

    for raw in (b"{}", b" ", b"{not valid json", b"null"):
        resp = client.request(
            "GET",
            _read_path(created["id"]),
            content=raw,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["code"] == "validation_error"
        ghost = client.request(
            "GET",
            _read_path("akr_ghost"),
            content=raw,
            headers={"Content-Type": "application/json"},
        )
        assert ghost.status_code == 422, raw

    assert db_session.execute(
        select(func.count()).select_from(AuditEvent)
    ).scalar_one() == events_before
    # The record itself is untouched and still readable.
    assert client.get(_read_path(created["id"])).json() == created


# --- Method handling ----------------------------------------------------------------


def test_non_get_methods_are_405(client):
    _world(client)
    created = _create_rotation(client)
    path = _read_path(created["id"])
    for method in ("post", "put", "patch", "delete"):
        resp = getattr(client, method)(path)
        assert resp.status_code == 405, method
        assert resp.json()["error"]["code"] == "method_not_allowed", method
    # The record is unaffected by the rejected methods.
    assert client.get(path).json() == created


# --- Persistence ---------------------------------------------------------------------


def test_record_fields_survive_restart(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    _world(file_client)
    created = _create_rotation(file_client)
    before = file_client.get(_read_path(created["id"])).json()
    assert before == created

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        resp = client.get(_read_path(created["id"]))
        assert resp.status_code == 200
        assert resp.json() == before

    # A retirement after the restart is the only field change, and the
    # retired view persists across a further restart.
    restarted_again = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted_again) as client:
        assert _post_retire(client, created["id"]).status_code == 200
    restarted_third = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted_third) as client:
        body = client.get(_read_path(created["id"])).json()
        assert body["active"] is False
        assert body["retired_at"] is not None
        for field in ("id", "actor_id", "new_public_key", "created_at"):
            assert body[field] == before[field]


def test_multiple_records_read_back_independently(client):
    _world(client)
    one = _create_rotation(client, seed=SEED_R1)
    two = _create_rotation(client, seed=SEED_R2)
    assert one["id"] != two["id"]

    read_one = client.get(_read_path(one["id"]))
    read_two = client.get(_read_path(two["id"]))
    assert read_one.status_code == read_two.status_code == 200
    assert read_one.json() == one
    assert read_two.json() == two
    assert read_one.json()["new_public_key"] == _key_b64(SEED_R1)
    assert read_two.json()["new_public_key"] == _key_b64(SEED_R2)
