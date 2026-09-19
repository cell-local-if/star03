"""Tests for immutable content lineage relations.

Covers POST /v1/content-relations, GET /v1/content-relations/{relation_id},
and GET /v1/contents/{content_id}/relations.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from provenance.models import (
    EVENT_CONTENT_RELATION_CREATED,
    AuditEvent,
    ContentRelation,
)
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    DIGEST_C,
    content_payload,
    create_actor,
)


def _create_content(client, digest, actor_id="org-1"):
    resp = client.post(
        "/v1/contents", json=content_payload(actor_id=actor_id, digest=digest)
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _relation_payload(content_id, parent_content_id, relation_type="version_of"):
    return {
        "content_id": content_id,
        "parent_content_id": parent_content_id,
        "relation_type": relation_type,
    }


def _create_relation(client, **overrides):
    resp = client.post("/v1/content-relations", json=_relation_payload(**overrides))
    assert resp.status_code == 201, resp.text
    return resp.json()


def _setup_contents(client, count=2):
    create_actor(client)
    digests = [DIGEST_A, DIGEST_B, DIGEST_C][:count]
    return [_create_content(client, digest=d) for d in digests]


def test_create_relation_returns_public_fields(client):
    child, parent = _setup_contents(client)
    body = _create_relation(
        client, content_id=child["id"], parent_content_id=parent["id"]
    )
    assert set(body) == {
        "id",
        "content_id",
        "parent_content_id",
        "relation_type",
        "created_at",
    }
    assert body["id"].startswith("rel_")
    assert body["content_id"] == child["id"]
    assert body["parent_content_id"] == parent["id"]
    assert body["relation_type"] == "version_of"
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))


def test_relation_id_is_deterministic_and_stable(client):
    child, parent = _setup_contents(client)
    first = _create_relation(
        client, content_id=child["id"], parent_content_id=parent["id"]
    )
    second = client.post(
        "/v1/content-relations",
        json=_relation_payload(child["id"], parent["id"]),
    )
    assert second.status_code == 200
    assert second.json() == first


def test_repeat_submission_returns_existing_and_adds_no_audit_event(
    client, db_session
):
    child, parent = _setup_contents(client)
    first = _create_relation(
        client, content_id=child["id"], parent_content_id=parent["id"]
    )
    events_after_create = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    for _ in range(3):
        repeat = client.post(
            "/v1/content-relations",
            json=_relation_payload(child["id"], parent["id"]),
        )
        assert repeat.status_code == 200
        assert repeat.json() == first

    events = db_session.execute(
        select(AuditEvent).order_by(AuditEvent.seq.asc())
    ).scalars().all()
    assert len(events) == events_after_create
    relation_events = [
        e for e in events if e.event_type == EVENT_CONTENT_RELATION_CREATED
    ]
    assert len(relation_events) == 1
    assert relation_events[0].resource_id == first["id"]


def test_distinct_relation_types_create_distinct_relations(client):
    child, parent = _setup_contents(client)
    version = _create_relation(
        client,
        content_id=child["id"],
        parent_content_id=parent["id"],
        relation_type="version_of",
    )
    derived = _create_relation(
        client,
        content_id=child["id"],
        parent_content_id=parent["id"],
        relation_type="derived_from",
    )
    assert version["id"] != derived["id"]
    assert version["relation_type"] == "version_of"
    assert derived["relation_type"] == "derived_from"


def test_existing_relation_is_never_updated(client, db_session):
    child, parent = _setup_contents(client)
    first = _create_relation(
        client, content_id=child["id"], parent_content_id=parent["id"]
    )
    repeat = client.post(
        "/v1/content-relations",
        json=_relation_payload(child["id"], parent["id"]),
    )
    assert repeat.status_code == 200
    assert repeat.json() == first
    again = client.get(f"/v1/content-relations/{first['id']}")
    assert again.json() == first
    rows = db_session.execute(select(ContentRelation)).scalars().all()
    assert [r.id for r in rows] == [first["id"]]


def test_get_relation_returns_full_public_fields(client):
    child, parent = _setup_contents(client)
    created = _create_relation(
        client, content_id=child["id"], parent_content_id=parent["id"]
    )
    resp = client.get(f"/v1/content-relations/{created['id']}")
    assert resp.status_code == 200
    assert resp.json() == created


def test_get_unknown_relation_is_distinct_not_found(client):
    resp = client.get("/v1/content-relations/rel_does_not_exist")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "content_relation_not_found"
    assert error["details"]["relation_id"] == "rel_does_not_exist"


def test_create_relation_unknown_content_is_not_found(client):
    child, parent = _setup_contents(client)
    for missing, present in (
        ("cnt_ghost", parent["id"]),
        (child["id"], "cnt_ghost"),
    ):
        resp = client.post(
            "/v1/content-relations",
            json=_relation_payload(missing, present),
        )
        assert resp.status_code == 404
        error = resp.json()["error"]
        assert error["code"] == "content_not_found"
        assert error["details"]["content_id"] == "cnt_ghost"


def test_reject_blank_identifiers(client):
    child, parent = _setup_contents(client)
    for payload in (
        _relation_payload("   ", parent["id"]),
        _relation_payload(child["id"], " "),
    ):
        resp = client.post("/v1/content-relations", json=payload)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_unknown_relation_type(client):
    child, parent = _setup_contents(client)
    for bad_type in ("copied_from", "VERSION_OF", "", 42):
        resp = client.post(
            "/v1/content-relations",
            json=_relation_payload(child["id"], parent["id"], bad_type),
        )
        assert resp.status_code == 422, bad_type
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_self_relation(client, db_session):
    (content,) = _setup_contents(client, count=1)
    resp = client.post(
        "/v1/content-relations",
        json=_relation_payload(content["id"], content["id"]),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert db_session.execute(select(ContentRelation)).scalars().all() == []


def test_reject_relation_cycle(client, db_session):
    a, b, c = _setup_contents(client, count=3)
    _create_relation(client, content_id=b["id"], parent_content_id=a["id"])
    _create_relation(client, content_id=c["id"], parent_content_id=b["id"])

    # Closing the loop a -> c (a is already reachable from c) is invalid.
    resp = client.post(
        "/v1/content-relations",
        json=_relation_payload(a["id"], c["id"]),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    # The direct reverse edge is a cycle too.
    resp = client.post(
        "/v1/content-relations",
        json=_relation_payload(a["id"], b["id"]),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    rows = db_session.execute(select(ContentRelation)).scalars().all()
    assert len(rows) == 2


def test_failed_relation_requests_write_no_rows_or_audit_events(
    client, db_session
):
    child, parent = _setup_contents(client)
    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())

    attempts = [
        client.post(
            "/v1/content-relations",
            json=_relation_payload("cnt_ghost", parent["id"]),
        ),
        client.post(
            "/v1/content-relations",
            json=_relation_payload(child["id"], "cnt_ghost"),
        ),
        client.post(
            "/v1/content-relations",
            json=_relation_payload(child["id"], child["id"]),
        ),
        client.post(
            "/v1/content-relations",
            json=_relation_payload(child["id"], parent["id"], "bogus"),
        ),
        client.post(
            "/v1/content-relations",
            json=_relation_payload(" ", parent["id"]),
        ),
    ]
    assert [r.status_code for r in attempts] == [404, 404, 422, 422, 422]
    assert db_session.execute(select(ContentRelation)).scalars().all() == []
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )


def test_relation_and_audit_event_commit_atomically(client, db_session):
    child, parent = _setup_contents(client)
    created = _create_relation(
        client, content_id=child["id"], parent_content_id=parent["id"]
    )

    relations = db_session.execute(select(ContentRelation)).scalars().all()
    events = db_session.execute(
        select(AuditEvent)
        .where(AuditEvent.event_type == EVENT_CONTENT_RELATION_CREATED)
        .order_by(AuditEvent.seq.asc())
    ).scalars().all()
    # The relation row and its audit event are both visible after one
    # request: they committed in the same transaction.
    assert [r.id for r in relations] == [created["id"]]
    assert [e.resource_id for e in events] == [created["id"]]
    created_at = events[0].created_at
    assert created_at.tzinfo.utcoffset(created_at).total_seconds() == 0


def test_list_relations_returns_in_and_out_edges_in_creation_order(client):
    a, b, c = _setup_contents(client, count=3)
    r1 = _create_relation(client, content_id=b["id"], parent_content_id=a["id"])
    r2 = _create_relation(
        client,
        content_id=c["id"],
        parent_content_id=a["id"],
        relation_type="derived_from",
    )
    r3 = _create_relation(client, content_id=c["id"], parent_content_id=b["id"])

    # a: two in-edges (b, c derive from it), no out-edges.
    resp = client.get(f"/v1/contents/{a['id']}/relations")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [r1["id"], r2["id"]]

    # b: one in-edge (from c), one out-edge (to a).
    resp = client.get(f"/v1/contents/{b['id']}/relations")
    body = resp.json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [r1["id"], r3["id"]]

    # c: two out-edges, no in-edges.
    resp = client.get(f"/v1/contents/{c['id']}/relations")
    body = resp.json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [r2["id"], r3["id"]]


def test_list_relations_empty_for_content_without_relations(client):
    (content,) = _setup_contents(client, count=1)
    resp = client.get(f"/v1/contents/{content['id']}/relations")
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 0}


def test_list_relations_unknown_content_is_not_found(client):
    resp = client.get("/v1/contents/cnt_missing/relations")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "content_not_found"
