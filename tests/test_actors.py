"""Actor registration: happy path, duplicates, and validation."""

from __future__ import annotations

from datetime import datetime, timezone

from conftest import audit_events


def test_create_actor_returns_public_fields_with_utc_timestamp(client):
    response = client.post(
        "/v1/actors",
        json={"actor_id": "actor-1", "name": "Alice", "actor_type": "person"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body == {
        "actor_id": "actor-1",
        "name": "Alice",
        "actor_type": "person",
        "created_at": body["created_at"],
    }
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.tzinfo is not None
    assert created_at.utcoffset() == timezone.utc.utcoffset(None)


def test_create_actor_persists_audit_event(client, app):
    client.post(
        "/v1/actors",
        json={"actor_id": "actor-1", "name": "Alice", "actor_type": "person"},
    )
    events = audit_events(app)
    assert len(events) == 1
    event = events[0]
    assert event.event_type == "actor.created"
    assert event.resource_id == "actor-1"
    assert event.occurred_at.tzinfo is not None


def test_duplicate_actor_id_is_rejected_with_distinct_error(client):
    payload = {"actor_id": "actor-1", "name": "Alice", "actor_type": "person"}
    assert client.post("/v1/actors", json=payload).status_code == 201
    response = client.post("/v1/actors", json=payload)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "actor_already_exists"


def test_actor_validation_errors_are_distinguishable(client):
    for payload in (
        {"actor_id": "", "name": "Alice", "actor_type": "person"},
        {"actor_id": "actor-1", "name": "", "actor_type": "person"},
        {"actor_id": "actor-1", "name": "Alice", "actor_type": ""},
        {"actor_id": "actor-1", "name": "Alice"},
    ):
        response = client.post("/v1/actors", json=payload)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"
