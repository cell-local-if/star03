"""Tests for POST /v1/actors."""

from __future__ import annotations

from datetime import datetime

from tests.helpers import actor_payload, create_actor


def test_create_actor_returns_full_fields(client):
    resp = client.post("/v1/actors", json=actor_payload())
    assert resp.status_code == 201
    body = resp.json()
    assert body["id"] == "org-1"
    assert body["name"] == "Example Org"
    assert body["type"] == "organization"
    # Timestamp is present and timezone-aware UTC ("Z" or "+00:00" both
    # denote a zero UTC offset).
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.tzinfo is not None
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))


def test_create_actor_persists_and_is_gettable(client):
    create_actor(client)
    # A second actor to ensure rows are independent.
    create_actor(client, actor_id="org-2", name="Other", type="person")
    # Re-query indirectly: creating content referencing it should succeed.
    from tests.helpers import DIGEST_A, content_payload

    resp = client.post(
        "/v1/contents", json=content_payload(actor_id="org-2", digest=DIGEST_A)
    )
    assert resp.status_code == 201
    assert resp.json()["actor_id"] == "org-2"


def test_duplicate_actor_id_is_distinct_conflict(client):
    create_actor(client, name="First")
    resp = client.post("/v1/actors", json=actor_payload(name="Second"))
    assert resp.status_code == 409
    error = resp.json()["error"]
    assert error["code"] == "actor_already_exists"
    assert error["details"]["actor_id"] == "org-1"
    # Original resource is untouched.
    from tests.helpers import DIGEST_B, content_payload

    follow = client.post(
        "/v1/contents", json=content_payload(digest=DIGEST_B)
    )
    assert follow.status_code == 201


def test_duplicate_actor_error_differs_from_validation_and_unknown(client):
    # Same actor id -> 409 conflict code.
    create_actor(client)
    conflict = client.post("/v1/actors", json=actor_payload()).json()["error"]

    # Unknown actor reference on content -> 404 unknown_actor code.
    from tests.helpers import content_payload

    unknown = client.post(
        "/v1/contents", json=content_payload(actor_id="ghost")
    ).json()["error"]

    # Malformed actor payload -> 422 validation_error code.
    malformed = client.post(
        "/v1/actors", json={"id": "x", "name": "", "type": "person"}
    ).json()["error"]

    assert conflict["code"] == "actor_already_exists"
    assert unknown["code"] == "unknown_actor"
    assert malformed["code"] == "validation_error"
    assert len({conflict["code"], unknown["code"], malformed["code"]}) == 3
