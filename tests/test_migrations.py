"""Tests for the versioned SQLite migration layer.

Covers startup migration from zero:

* a brand-new database is created at the latest schema, baselined in the
  ``schema_migrations`` ledger, and continues the explicit actor sequence via
  an insert trigger;
* a legacy database whose ``actors`` table predates the ledger gains
  ``display_seq`` backfilled from the original insertion order (stable even
  for same-created_at rows), the supporting index and trigger, and a single
  version-1 ledger row -- without changing any existing actor field;
* startup is idempotent: restarting never re-runs an applied version and
  ordering is identical;
* a failing migration rolls the whole batch back (no ledger, no new column,
  original rows intact) and a later successful startup migrates cleanly;
* the read-only actor list never writes a migration record.
"""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from provenance import migrations
from provenance.app import create_app
from provenance.config import Settings
from provenance.database import make_engine
from tests.helpers import create_actor

ACTORS_PATH = "/v1/actors"
TIE = "2026-03-01 00:00:00.000000"
LEGACY_ACTORS = [
    ("org-1", "Example Org", "organization"),
    ("p-1", "Alice", "person"),
    ("org-2", "Other Org", "organization"),
    ("dev-1", "Sensor", "device"),
]


def _connect(path):
    return sqlite3.connect(path)


def _create_legacy_database(path) -> None:
    """Create a pre-migration actors table with same-timestamp rows."""
    con = _connect(path)
    try:
        con.execute(
            "CREATE TABLE actors ("
            "id VARCHAR(255) PRIMARY KEY, "
            "name TEXT NOT NULL, "
            "type VARCHAR(64) NOT NULL, "
            "created_at DATETIME NOT NULL)"
        )
        for actor_id, name, actor_type in LEGACY_ACTORS:
            con.execute(
                "INSERT INTO actors (id, name, type, created_at) "
                "VALUES (?, ?, ?, ?)",
                (actor_id, name, actor_type, TIE),
            )
        con.commit()
    finally:
        con.close()


def _table_names(con):
    return {
        row[0]
        for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _columns(con, table):
    return [row[1] for row in con.execute(f"PRAGMA table_info({table})")]


# --- Fresh database -----------------------------------------------------------


def test_fresh_database_is_baselined_with_latest_schema(tmp_path):
    db_path = tmp_path / "fresh.db"
    url = f"sqlite:///{db_path.as_posix()}"
    with TestClient(create_app(Settings(database_url=url))):
        pass

    con = _connect(db_path)
    try:
        tables = _table_names(con)
        assert "schema_migrations" in tables
        assert [row[0] for row in con.execute("SELECT version FROM schema_migrations")] == [1]
        assert "display_seq" in _columns(con, "actors")
        triggers = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            )
        }
        assert migrations.ACTOR_SEQ_TRIGGER in triggers
        indexes = {
            row[1]
            for row in con.execute("PRAGMA index_list(actors)")
        }
        assert "ix_actors_created_order" in indexes
    finally:
        con.close()


def test_fresh_database_stamps_new_actors_with_dense_sequence(tmp_path):
    db_path = tmp_path / "fresh.db"
    url = f"sqlite:///{db_path.as_posix()}"
    app = create_app(Settings(database_url=url))
    with TestClient(app) as client:
        for index in range(3):
            create_actor(client, actor_id=f"a-{index}", name=f"N{index}")
    con = _connect(db_path)
    try:
        assert [
            row[0]
            for row in con.execute(
                "SELECT display_seq FROM actors ORDER BY display_seq"
            )
        ] == [1, 2, 3]
    finally:
        con.close()


# --- Legacy database ----------------------------------------------------------


def test_legacy_database_backfills_order_and_preserves_fields(tmp_path):
    db_path = tmp_path / "legacy.db"
    _create_legacy_database(db_path)
    url = f"sqlite:///{db_path.as_posix()}"

    with TestClient(create_app(Settings(database_url=url))) as client:
        body = client.get(ACTORS_PATH).json()

    # Same-created_at ties follow the original insertion order.
    assert [item["id"] for item in body["items"]] == [a[0] for a in LEGACY_ACTORS]
    assert body["count"] == 4
    for item, (actor_id, name, actor_type) in zip(body["items"], LEGACY_ACTORS):
        assert item["id"] == actor_id
        assert item["name"] == name
        assert item["type"] == actor_type
        assert item["created_at"].startswith("2026-03-01T00:00:00")
        assert "display_seq" not in item

    con = _connect(db_path)
    try:
        assert [row[0] for row in con.execute("SELECT version FROM schema_migrations")] == [1]
        assert [
            row[0]
            for row in con.execute(
                "SELECT id FROM actors ORDER BY created_at, display_seq"
            )
        ] == [a[0] for a in LEGACY_ACTORS]
        assert [
            row[0]
            for row in con.execute(
                "SELECT display_seq FROM actors ORDER BY display_seq"
            )
        ] == [1, 2, 3, 4]
        # No actor can ever be missed or reordered by the backfill.
        assert con.execute(
            "SELECT COUNT(*) FROM actors WHERE display_seq BETWEEN 1 AND 4"
        ).fetchone()[0] == 4
    finally:
        con.close()


def test_legacy_migration_continues_sequence_for_later_inserts(tmp_path):
    db_path = tmp_path / "legacy.db"
    _create_legacy_database(db_path)
    url = f"sqlite:///{db_path.as_posix()}"
    app = create_app(Settings(database_url=url))
    with TestClient(app) as client:
        create_actor(client, actor_id="new-1", name="New", type="device")
        body = client.get(ACTORS_PATH).json()

    # The legacy rows keep their order; the new actor sorts after them.
    assert [item["id"] for item in body["items"]] == [
        a[0] for a in LEGACY_ACTORS
    ] + ["new-1"]

    con = _connect(db_path)
    try:
        assert con.execute(
            "SELECT display_seq FROM actors WHERE id = 'new-1'"
        ).fetchone()[0] == 5
    finally:
        con.close()


def test_empty_legacy_table_migrates_and_starts_sequence_at_one(tmp_path):
    db_path = tmp_path / "empty-legacy.db"
    con = _connect(db_path)
    try:
        con.execute(
            "CREATE TABLE actors ("
            "id VARCHAR(255) PRIMARY KEY, "
            "name TEXT NOT NULL, "
            "type VARCHAR(64) NOT NULL, "
            "created_at DATETIME NOT NULL)"
        )
        con.commit()
    finally:
        con.close()
    url = f"sqlite:///{db_path.as_posix()}"
    app = create_app(Settings(database_url=url))
    with TestClient(app) as client:
        create_actor(client, actor_id="only-1", name="Only", type="device")
        body = client.get(ACTORS_PATH).json()
    assert [item["id"] for item in body["items"]] == ["only-1"]
    con = _connect(db_path)
    try:
        assert [row[0] for row in con.execute("SELECT version FROM schema_migrations")] == [1]
        assert con.execute(
            "SELECT display_seq FROM actors WHERE id = 'only-1'"
        ).fetchone()[0] == 1
    finally:
        con.close()


def test_migration_is_idempotent_across_restarts(tmp_path):
    db_path = tmp_path / "legacy.db"
    _create_legacy_database(db_path)
    url = f"sqlite:///{db_path.as_posix()}"

    app = create_app(Settings(database_url=url))
    with TestClient(app) as client:
        first = client.get(ACTORS_PATH)
    first_app = create_app(Settings(database_url=url))
    with TestClient(first_app) as client:
        second = client.get(ACTORS_PATH)
    third_app = create_app(Settings(database_url=url))
    with TestClient(third_app) as client:
        third = client.get(ACTORS_PATH)

    assert second.json() == first.json()
    assert third.json() == first.json()

    con = _connect(db_path)
    try:
        # Applied exactly once despite three startups.
        assert [row[0] for row in con.execute("SELECT version FROM schema_migrations")] == [1]
        assert [
            row[0]
            for row in con.execute(
                "SELECT display_seq FROM actors ORDER BY display_seq"
            )
        ] == [1, 2, 3, 4]
    finally:
        con.close()


# --- Failure rollback ---------------------------------------------------------


def test_failed_migration_rolls_back_and_a_later_startup_succeeds(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "fail.db"
    _create_legacy_database(db_path)
    url = f"sqlite:///{db_path.as_posix()}"

    def _failing_up(cursor, engine, metadata):
        cursor.execute(
            "ALTER TABLE actors ADD COLUMN display_seq BIGINT "
            "NOT NULL DEFAULT 0"
        )
        cursor.execute("UPDATE actors SET display_seq = 1")
        raise RuntimeError("simulated migration failure")

    engine = make_engine(url)
    monkeypatch.setattr(
        migrations,
        "MIGRATIONS",
        (migrations.Migration(version=1, up=_failing_up),),
    )

    with pytest.raises(RuntimeError, match="simulated migration failure"):
        migrations.run_migrations(engine)

    con = _connect(db_path)
    try:
        tables = _table_names(con)
        # No half-finished ledger and no partially added column.
        assert "schema_migrations" not in tables
        assert "display_seq" not in _columns(con, "actors")
        # Existing identity/fields are untouched.
        assert [row[0] for row in con.execute("SELECT id FROM actors ORDER BY id")] == [
            a[0] for a in sorted(LEGACY_ACTORS)
        ]
        assert con.execute("SELECT COUNT(*) FROM actors").fetchone()[0] == 4
    finally:
        con.close()

    # A later, healthy startup migrates cleanly (retry after rollback).
    monkeypatch.undo()
    with TestClient(create_app(Settings(database_url=url))) as client:
        body = client.get(ACTORS_PATH).json()
    assert [item["id"] for item in body["items"]] == [
        a[0] for a in LEGACY_ACTORS
    ]
    con = _connect(db_path)
    try:
        assert [row[0] for row in con.execute("SELECT version FROM schema_migrations")] == [1]
        assert "display_seq" in _columns(con, "actors")
    finally:
        con.close()


# --- Read-only migration ledger ----------------------------------------------


def test_actor_list_writes_no_migration_record(tmp_path):
    db_path = tmp_path / "ro.db"
    url = f"sqlite:///{db_path.as_posix()}"
    app = create_app(Settings(database_url=url))
    with TestClient(app) as client:
        client.get(ACTORS_PATH)
        client.get(ACTORS_PATH, params={"type": "unicorn"})
        client.get(ACTORS_PATH, params={"limit": "0"})
        client.get(ACTORS_PATH, params={"cursor": "bad"})
        client.request("GET", ACTORS_PATH, content=b"{}")
    con = _connect(db_path)
    try:
        # Still just the single baselined version; reads added nothing.
        assert [row[0] for row in con.execute("SELECT version FROM schema_migrations")] == [1]
        assert con.execute("SELECT COUNT(*) FROM actors").fetchone()[0] == 0
    finally:
        con.close()
