"""Content registration: validation, idempotency, and audit behavior."""

from __future__ import annotations

from datetime import datetime, timezone

from conftest import DIGEST_A, DIGEST_B, audit_events, create_content


def test_create_content_returns_full_public_fields(client, actor):
    response = create_content(client, title="Launch photo")
    assert response.status_code == 201
    body = response.json()
    assert body["content_id"]
    assert body["digest_algorithm"] == "sha256"
    assert body["digest_hex"] == DIGEST_A
    assert body["media_type"] == "image/png"
    assert body["title"] == "Launch photo"
    assert body["actor_id"] == "actor-1"
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.tzinfo is not None
    assert created_at.utcoffset() == timezone.utc.utcoffset(None)


def test_title_is_optional(client, actor):
    response = create_content(client)
    assert response.status_code == 201
    assert response.json()["title"] is None


def test_first_content_creation_persists_audit_event(client, app, actor):
    content_id = create_content(client).json()["content_id"]
    events = audit_events(app)
    content_events = [e for e in events if e.event_type == "content.created"]
    assert len(content_events) == 1
    assert content_events[0].resource_id == content_id
    assert content_events[0].occurred_at.tzinfo is not None


def test_duplicate_digest_returns_existing_resource_without_new_audit_event(
    client, app, actor
):
    first = create_content(client, title="first")
    assert first.status_code == 201
    replay = create_content(client, title="second submission")
    assert replay.status_code == 200
    assert replay.json()["content_id"] == first.json()["content_id"]
    # The stored record is unchanged, including the original title.
    assert replay.json()["title"] == "first"

    events = audit_events(app)
    content_events = [e for e in events if e.event_type == "content.created"]
    assert len(content_events) == 1


def test_digest_hex_is_case_insensitive_for_dedup(client, actor):
    first = create_content(client, digest=DIGEST_A.upper())
    assert first.status_code == 201
    replay = create_content(client, digest=DIGEST_A)
    assert replay.status_code == 200
    assert replay.json()["content_id"] == first.json()["content_id"]
    assert replay.json()["digest_hex"] == DIGEST_A


def test_unknown_actor_is_rejected_with_distinct_error(client, actor):
    response = create_content(client, actor_id="no-such-actor")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unknown_actor"


def test_unsupported_digest_algorithm_is_rejected(client, actor):
    response = create_content(client, digest_algorithm="md5")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_digest_algorithm"


def test_malformed_digest_hex_is_rejected(client, actor):
    for bad in (
        "abc",  # too short
        "g" * 64,  # non-hex characters
        DIGEST_A + "0",  # 65 characters
        DIGEST_A[:-1],  # 63 characters
    ):
        response = create_content(client, digest=bad)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_digest_hex"


def test_empty_media_type_and_actor_id_are_rejected(client, actor):
    for overrides in ({"media_type": ""}, {"actor_id": ""}):
        response = create_content(client, **overrides)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"


def test_failed_content_creation_persists_nothing(client, app, actor):
    create_content(client, digest_algorithm="md5")
    create_content(client, actor_id="no-such-actor")
    assert client.get("/v1/contents").json() == []
    assert [e.event_type for e in audit_events(app)] == ["actor.created"]
