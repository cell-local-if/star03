"""Persistence and startup tests: explicit DB location and auto table creation."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from provenance.app import create_app
from provenance.config import Settings
from tests.helpers import DIGEST_A, content_payload, create_actor


def test_first_startup_creates_database_file_and_tables(tmp_path):
    db_path = tmp_path / "nested" / "provenance.db"
    url = f"sqlite:///{db_path.as_posix()}"
    assert not db_path.exists()
    create_app(Settings(database_url=url))
    assert db_path.exists()
    # The file is a valid SQLite database with our tables.
    import sqlite3

    con = sqlite3.connect(db_path)
    tables = {
        row[0]
        for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    con.close()
    assert {"actors", "contents", "audit_events"}.issubset(tables)


def test_data_persists_across_app_restarts(tmp_db_url, file_client):
    create_actor(file_client)
    created = file_client.post(
        "/v1/contents", json=content_payload()
    ).json()
    content_id = created["id"]
    claim = file_client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": "org-1",
            "claim_type": "authorship",
            "payload": {"statement": "created by org-1"},
        },
    ).json()

    # Simulate a restart with a brand-new app/engine on the same database.
    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        resp = client.get(f"/v1/contents/{content_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["digest_hex"] == DIGEST_A
        assert body["actor_id"] == "org-1"

        # Restart does not clear or duplicate data.
        listing = client.get("/v1/contents").json()
        assert listing["count"] == 1

        # The claim and its per-content listing survived the restart.
        fetched = client.get(f"/v1/claims/{claim['id']}")
        assert fetched.status_code == 200
        assert fetched.json() == claim
        claims = client.get(f"/v1/contents/{content_id}/claims").json()
        assert claims["count"] == 1
        assert claims["items"][0]["id"] == claim["id"]

        # Audit history survived the restart.
        import sqlite3

        path = tmp_db_url.removeprefix("sqlite:///")
        con = sqlite3.connect(path)
        audit_count = con.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
        con.close()
        assert audit_count == 3


def test_environment_variable_selects_database(monkeypatch, tmp_path):
    db_path = tmp_path / "env.db"
    monkeypatch.setenv("PROVENANCE_DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    settings = Settings.from_env()
    assert Path(settings.database_url.removeprefix("sqlite:///")) == db_path.resolve()
