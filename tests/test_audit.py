"""Audit-event tests: exactly one event per successful first creation."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from provenance.models import (
    EVENT_ACTOR_CREATED,
    EVENT_CONTENT_CREATED,
    AuditEvent,
)
from tests.helpers import DIGEST_A, content_payload, create_actor


def _events(session):
    rows = session.execute(
        select(AuditEvent).order_by(AuditEvent.seq.asc())
    ).scalars().all()
    return [(r.event_type, r.resource_id, r.created_at) for r in rows]


def test_actor_creation_writes_audit_event(client, db_session):
    create_actor(client, actor_id="org-1")
    events = _events(db_session)
    assert len(events) == 1
    event_type, resource_id, created_at = events[0]
    assert event_type == EVENT_ACTOR_CREATED
    assert resource_id == "org-1"
    assert created_at.tzinfo is not None
    assert datetime.fromisoformat(created_at.isoformat()).utcoffset() is not None


def test_first_content_creation_writes_audit_event(client, db_session):
    create_actor(client)
    created = client.post("/v1/contents", json=content_payload()).json()

    events = _events(db_session)
    assert [e[0] for e in events] == [
        EVENT_ACTOR_CREATED,
        EVENT_CONTENT_CREATED,
    ]
    assert events[1][1] == created["id"]


def test_duplicate_content_adds_no_audit_event(client, db_session):
    create_actor(client)
    first = client.post("/v1/contents", json=content_payload())
    assert first.status_code == 201
    count_after_create = len(_events(db_session))

    for _ in range(3):
        repeat = client.post("/v1/contents", json=content_payload())
        assert repeat.status_code == 200

    assert len(_events(db_session)) == count_after_create
    assert len(_events(db_session)) == 2  # actor + first content only


def test_failed_requests_write_no_audit_events(client, db_session):
    # Malformed content (validation) -> no event.
    bad = client.post(
        "/v1/contents",
        json=content_payload(actor_id="ghost"),
    )
    assert bad.status_code == 404
    assert _events(db_session) == []

    create_actor(client)
    # Duplicate actor conflict -> no additional event beyond first success.
    dup = client.post(
        "/v1/actors", json={"id": "org-1", "name": "X", "type": "person"}
    )
    assert dup.status_code == 409
    events = _events(db_session)
    assert len(events) == 1
    assert events[0][0] == EVENT_ACTOR_CREATED


def test_audit_timestamps_are_utc(client, db_session):
    create_actor(client)
    _, _, created_at = _events(db_session)[0]
    assert created_at.tzinfo.utcoffset(created_at).total_seconds() == 0
