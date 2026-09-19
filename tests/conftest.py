"""Shared deterministic test fixtures (offline, in-memory and temp-file DBs)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from provenance.app import create_app
from provenance.config import Settings


@pytest.fixture
def app():
    application = create_app(Settings(database_url="sqlite:///:memory:"))
    yield application


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def tmp_db_url(tmp_path):
    db_path = tmp_path / "provenance.db"
    return f"sqlite:///{db_path.as_posix()}"


@pytest.fixture
def file_app(tmp_db_url):
    # First startup: creates the file and all tables.
    application = create_app(Settings(database_url=tmp_db_url))
    yield application


@pytest.fixture
def file_client(file_app):
    with TestClient(file_app) as test_client:
        yield test_client


@pytest.fixture
def db_session(app):
    """A direct session for asserting on persisted rows (e.g. audit events)."""
    session = app.state.session_factory()
    try:
        yield session
    finally:
        session.close()
