"""Shared fixtures: an app backed by a throwaway SQLite file per test."""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from provenance.app import create_app
from provenance.config import Settings
from provenance.models import AuditEvent

DIGEST_A = hashlib.sha256(b"content-a").hexdigest()
DIGEST_B = hashlib.sha256(b"content-b").hexdigest()
DIGEST_C = hashlib.sha256(b"content-c").hexdigest()


@pytest.fixture()
def app(tmp_path):
    db_path = tmp_path / "test.db"
    settings = Settings(database_url=f"sqlite:///{db_path}")
    return create_app(settings)


@pytest.fixture()
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def actor(client):
    response = client.post(
        "/v1/actors",
        json={"actor_id": "actor-1", "name": "Alice", "actor_type": "person"},
    )
    assert response.status_code == 201
    return response.json()


def create_content(client, digest=DIGEST_A, actor_id="actor-1", **overrides):
    payload = {
        "digest_algorithm": "sha256",
        "digest_hex": digest,
        "media_type": "image/png",
        "actor_id": actor_id,
    }
    payload.update(overrides)
    return client.post("/v1/contents", json=payload)


def audit_events(app):
    with app.state.session_factory() as session:
        return list(session.execute(select(AuditEvent).order_by(AuditEvent.event_id)).scalars())
