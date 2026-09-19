"""Tests for content lineage relations.

Covers POST /v1/content-relations, GET /v1/content-relations/{relation_id},
and GET /v1/contents/{content_id}/relations: stable ids, idempotent retry,
cycle/self-loop rejection, missing-resource precedence, audit atomicity, and
stable inbound/outbound listing order.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import select

from provenance.app import create_app
from provenance.config import Settings
from provenance.models import (
    EVENT_CONTENT_RELATION_CREATED,
    ContentRelation,
    AuditEvent,
)
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    DIGEST_C,
    content_payload,
    create_actor,
)

DIGEST_D = hashlib.sha256(b"content-d").hexdigest()
DIGEST_E = hashlib.sha256(b"content-e").hexdigest()


def _create_content(client, digest, **overrides):
    resp = client.post("/v1/contents", json=content_payload(digest=digest, **overrides))
    assert resp.status_code == 201, resp.text
    return resp.json()


def _contents_ab(client):
    a = _create_content(client, DIGEST_A)["id"]
    b = _create_content(client, DIGEST_B)["id"]
    return a, b


def _relation_payload(child, parent, relation_type="version_of"):
    return {
        "content_id": child,
        "parent_content_id": parent,
        "relation_type": relation_type,
    }


def _create_relation(client, child, parent, relation_type="version_of"):
    resp = client.post(
        "/v1/content-relations",
        json=_relation_payload(child, parent, relation_type),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# --- Creation ---------------------------------------------------------------


def test_create_relation_returns_full_public_fields(client):
    create_actor(client)
    a, b = _contents_ab(client)
    body = _create_relation(client, b, a, "version_of")
    assert set(body) == {
        "id",
        "content_id",
        "parent_content_id",
        "relation_type",
        "created_at",
    }
    assert body["id"].startswith("rel_")
    assert body["content_id"] == b
    assert body["parent_content_id"] == a
    assert body["relation_type"] == "version_of"
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.tzinfo is not None
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))


def test_relation_id_is_stable_and_deterministic(client):
    create_actor(client)
    a, b = _contents_ab(client)
    first = _create_relation(client, b, a, "derived_from")
    # The stable id is derived solely from the identity triple.
    from provenance.ids import content_relation_id

    expected = content_relation_id(b, a, "derived_from")
    assert first["id"] == expected


def test_both_relation_types_accepted(client):
    create_actor(client)
    a = _create_content(client, DIGEST_A)["id"]
    b = _create_content(client, DIGEST_B)["id"]
    c = _create_content(client, DIGEST_C)["id"]
    _create_relation(client, b, a, "version_of")
    derived = _create_relation(client, c, a, "derived_from")
    assert derived["relation_type"] == "derived_from"


def test_same_endpoints_different_types_are_independent_relations(client):
    create_actor(client)
    a, b = _contents_ab(client)
    v = _create_relation(client, b, a, "version_of")
    d = _create_relation(client, b, a, "derived_from")
    assert v["id"] != d["id"]
    fetched = client.get(f"/v1/contents/{b}/relations").json()
    assert fetched["count"] == 2


# --- Idempotency ------------------------------------------------------------


def test_repeat_submission_returns_200_existing_relation(client):
    create_actor(client)
    a, b = _contents_ab(client)
    first = _create_relation(client, b, a)
    resp = client.post(
        "/v1/content-relations", json=_relation_payload(b, a, "version_of")
    )
    assert resp.status_code == 200
    assert resp.json() == first


def test_repeat_submission_writes_no_audit_event(client, db_session):
    create_actor(client)
    a, b = _contents_ab(client)
    _create_relation(client, b, a)
    events_after_create = len(_audit_events(db_session))

    for _ in range(3):
        resp = client.post(
            "/v1/content-relations", json=_relation_payload(b, a)
        )
        assert resp.status_code == 200

    assert len(_audit_events(db_session)) == events_after_create
    relations = db_session.execute(select(ContentRelation)).scalars().all()
    assert len(relations) == 1


def test_retry_after_other_relation_still_idempotent(client):
    create_actor(client)
    a = _create_content(client, DIGEST_A)["id"]
    b = _create_content(client, DIGEST_B)["id"]
    c = _create_content(client, DIGEST_C)["id"]
    first = _create_relation(client, b, a, "version_of")
    # An unrelated edge in between.
    _create_relation(client, c, a, "derived_from")
    resp = client.post(
        "/v1/content-relations", json=_relation_payload(b, a, "version_of")
    )
    assert resp.status_code == 200
    assert resp.json()["id"] == first["id"]


# --- GET by id --------------------------------------------------------------


def test_get_relation_returns_full_fields(client):
    create_actor(client)
    a, b = _contents_ab(client)
    created = _create_relation(client, b, a)
    resp = client.get(f"/v1/content-relations/{created['id']}")
    assert resp.status_code == 200
    assert resp.json() == created


def test_get_unknown_relation_is_404(client):
    resp = client.get("/v1/content-relations/rel_does_not_exist")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "content_relation_not_found"
    assert error["details"]["relation_id"] == "rel_does_not_exist"


# --- Missing resources ------------------------------------------------------


def test_unknown_child_content_is_content_not_found(client):
    create_actor(client)
    a = _create_content(client, DIGEST_A)["id"]
    resp = client.post(
        "/v1/content-relations",
        json=_relation_payload("cnt_missing", a),
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "content_not_found"


def test_unknown_parent_content_is_content_not_found(client):
    create_actor(client)
    a = _create_content(client, DIGEST_A)["id"]
    resp = client.post(
        "/v1/content-relations",
        json=_relation_payload(a, "cnt_missing"),
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "content_not_found"
    assert error["details"]["content_id"] == "cnt_missing"


def test_both_endpoints_unknown_reports_content_not_found(client):
    create_actor(client)
    resp = client.post(
        "/v1/content-relations",
        json=_relation_payload("cnt_one", "cnt_two"),
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "content_not_found"


def test_list_relations_unknown_content_is_404(client):
    resp = client.get("/v1/contents/cnt_missing/relations")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "content_not_found"


# --- Validation -------------------------------------------------------------


def test_self_loop_is_422_and_writes_nothing(client, db_session):
    create_actor(client)
    a = _create_content(client, DIGEST_A)["id"]
    # Caught at the schema boundary.
    resp = client.post(
        "/v1/content-relations", json=_relation_payload(a, a)
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert db_session.execute(select(ContentRelation)).scalars().all() == []
    assert [
        e for e in _audit_events(db_session)
        if e[0] == EVENT_CONTENT_RELATION_CREATED
    ] == []


def test_unknown_relation_type_is_422(client):
    create_actor(client)
    a, b = _contents_ab(client)
    resp = client.post(
        "/v1/content-relations",
        json=_relation_payload(b, a, "copy_of"),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_blank_identifiers_are_422(client):
    create_actor(client)
    a, b = _contents_ab(client)
    for field_name in ("content_id", "parent_content_id"):
        payload = _relation_payload(b, a)
        payload[field_name] = "   "
        resp = client.post("/v1/content-relations", json=payload)
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "validation_error"


def test_missing_required_fields_are_422(client):
    resp = client.post("/v1/content-relations", json={})
    assert resp.status_code == 422
    issue_fields = {
        ".".join(part for part in issue["loc"] if part != "body")
        for issue in resp.json()["error"]["details"]["issues"]
    }
    assert {"content_id", "parent_content_id", "relation_type"}.issubset(
        issue_fields
    )


def test_undeclared_field_is_ignored_or_rejected_but_never_stored(client):
    # The schema does not forbid extras; FastAPI ignores unknown fields by
    # default, so the relation is created from the declared triple only and
    # no extra data is persisted.
    create_actor(client)
    a, b = _contents_ab(client)
    payload = _relation_payload(b, a)
    payload["note"] = "unexpected"
    resp = client.post("/v1/content-relations", json=payload)
    assert resp.status_code == 201
    assert set(resp.json()) == {
        "id",
        "content_id",
        "parent_content_id",
        "relation_type",
        "created_at",
    }


def test_cycle_is_422_and_writes_nothing(client, db_session):
    # Existing graph: C -> B -> A. Proposing A -> C closes a cycle.
    create_actor(client)
    a = _create_content(client, DIGEST_A)["id"]
    b = _create_content(client, DIGEST_B)["id"]
    c = _create_content(client, DIGEST_C)["id"]
    _create_relation(client, b, a)
    _create_relation(client, c, b)

    before = len(db_session.execute(select(ContentRelation)).scalars().all())
    resp = client.post(
        "/v1/content-relations", json=_relation_payload(a, c, "derived_from")
    )
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"]["reason"] == "cycle_detected"

    assert (
        len(db_session.execute(select(ContentRelation)).scalars().all())
        == before
    )
    # No audit event for the rejected edge: only the two accepted edges
    # have relation audit rows.
    relation_audit = [
        e for e in _audit_events(db_session)
        if e[0] == EVENT_CONTENT_RELATION_CREATED
    ]
    assert len(relation_audit) == before


def test_cycle_via_longer_path_is_422(client):
    # E -> D -> C -> B -> A; proposing A -> E must fail even though no
    # direct A->E edge exists.
    create_actor(client)
    a = _create_content(client, DIGEST_A)["id"]
    b = _create_content(client, DIGEST_B)["id"]
    c = _create_content(client, DIGEST_C)["id"]
    d = _create_content(client, DIGEST_D)["id"]
    e = _create_content(client, DIGEST_E)["id"]
    _create_relation(client, b, a)
    _create_relation(client, c, b)
    _create_relation(client, d, c)
    _create_relation(client, e, d)

    resp = client.post(
        "/v1/content-relations", json=_relation_payload(a, e)
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_non_cyclic_diamond_shape_is_allowed(client):
    # A shared ancestor reached via two branches is not a cycle.
    create_actor(client)
    a = _create_content(client, DIGEST_A)["id"]
    b = _create_content(client, DIGEST_B)["id"]
    c = _create_content(client, DIGEST_C)["id"]
    d = _create_content(client, DIGEST_D)["id"]
    _create_relation(client, b, a)
    _create_relation(client, c, a)
    merged = _create_relation(client, d, b)
    _create_relation(client, d, c)
    assert merged["parent_content_id"] == b


# --- Inbound/outbound listing -----------------------------------------------


def test_list_includes_outbound_and_inbound_edges(client):
    create_actor(client)
    a = _create_content(client, DIGEST_A)["id"]
    b = _create_content(client, DIGEST_B)["id"]
    c = _create_content(client, DIGEST_C)["id"]
    edge_ba = _create_relation(client, b, a)
    edge_ca = _create_relation(client, c, a)

    # A is the parent (direct source) of both: two inbound edges.
    resp = client.get(f"/v1/contents/{a}/relations")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [
        edge_ba["id"],
        edge_ca["id"],
    ]

    # B has one outbound edge only.
    resp = client.get(f"/v1/contents/{b}/relations")
    body = resp.json()
    assert body["count"] == 1
    assert body["items"][0]["id"] == edge_ba["id"]


def test_list_returns_stable_creation_order(client):
    create_actor(client)
    a = _create_content(client, DIGEST_A)["id"]
    b = _create_content(client, DIGEST_B)["id"]
    c = _create_content(client, DIGEST_C)["id"]
    d = _create_content(client, DIGEST_D)["id"]
    e1 = _create_relation(client, b, a)
    e2 = _create_relation(client, a, c, "derived_from")
    e3 = _create_relation(client, d, a)
    # All three touch A (e1/e3 inbound, e2 outbound), interleaved in creation.
    body = client.get(f"/v1/contents/{a}/relations").json()
    assert [item["id"] for item in body["items"]] == [
        e1["id"],
        e2["id"],
        e3["id"],
    ]


def test_list_for_content_without_relations_is_empty(client):
    create_actor(client)
    a = _create_content(client, DIGEST_A)["id"]
    resp = client.get(f"/v1/contents/{a}/relations")
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 0}


# --- Audit atomicity --------------------------------------------------------


def _audit_events(session):
    rows = session.execute(
        select(AuditEvent).order_by(AuditEvent.seq.asc())
    ).scalars().all()
    return [(r.event_type, r.resource_id) for r in rows]


def test_first_creation_writes_audit_event_in_same_commit(client, db_session):
    create_actor(client)
    a, b = _contents_ab(client)
    created = _create_relation(client, b, a)
    events = _audit_events(db_session)
    relation_events = [
        e for e in events if e[0] == EVENT_CONTENT_RELATION_CREATED
    ]
    assert len(relation_events) == 1
    assert relation_events[0][1] == created["id"]


def test_audit_event_persists_across_restart(tmp_db_url):
    application = create_app(Settings(database_url=tmp_db_url))
    with TestClient(application) as client:
        create_actor(client)
        a, b = _contents_ab(client)
        _create_relation(client, b, a)

    import sqlite3

    path = tmp_db_url.removeprefix("sqlite:///")
    con = sqlite3.connect(path)
    rows = con.execute(
        "SELECT event_type FROM audit_events WHERE event_type = ?",
        (EVENT_CONTENT_RELATION_CREATED,),
    ).fetchall()
    tables = {
        row[0]
        for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    con.close()
    assert len(rows) == 1
    assert "content_relations" in tables
