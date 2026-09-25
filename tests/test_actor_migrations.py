"""Tests for the versioned SQLite migration behind the actor list.

Covers the migration layer that introduces the actors' explicit persistent
ordering column:

* a brand-new database is created with the current schema and records the
  baseline version;
* a database already carrying the legacy ``actors`` table keeps every
  existing row and identifier and backfills the explicit sequence from the
  original insertion order (``created_at`` then ``rowid``), so same-instant
  actors stay stable;
* startup is repeatable: an applied version never runs twice;
* a failed migration rolls the whole startup back -- no ledger row, no new
  column, no partial index -- and a later healthy startup succeeds;
* after migration (and a restart) the list orders by ``created_at`` then
  the explicit sequence, and newly created actors continue the sequence.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from provenance.app import create_app
from provenance.config import Settings
from provenance.database import make_engine
from provenance.migrations import (
    ACTORS_SEQ_VERSION,
    MIGRATIONS,
    Migration,
    run_migrations,
)
from tests.helpers import create_actor

ACTORS_PATH = "/v1/actors"

_LEGACY_SCHEMA = (
    "CREATE TABLE actors ("
    "id VARCHAR(255) NOT NULL PRIMARY KEY, "
    "name TEXT NOT NULL, "
    "type VARCHAR(64) NOT NULL, "
    "created_at DATETIME NOT NULL)"
)


def _raw_connect(db_url: str) -> sqlite3.Connection:
    return sqlite3.connect(db_url.removeprefix("sqlite:///"))


# --- Fresh database ------------------------------------------------------------


def test_fresh_database_records_baseline_and_has_seq(tmp_db_url):
    create_app(Settings(database_url=tmp_db_url))
    con = _raw_connect(tmp_db_url)
    try:
        columns = [row[1] for row in con.execute("PRAGMA table_info(actors)")]
        assert "seq" in columns
        versions = [row[0] for row in con.execute(
            "SELECT version FROM schema_migrations"
        )]
        assert versions == [ACTORS_SEQ_VERSION]
        index_names = {row[1] for row in con.execute("PRAGMA index_list(actors)")}
        assert "ix_actors_created_order" in index_names
    finally:
        con.close()


def test_fresh_startup_is_idempotent(tmp_db_url):
    first = create_app(Settings(database_url=tmp_db_url))
    first.state.engine.dispose()
    second = create_app(Settings(database_url=tmp_db_url))
    assert run_migrations(second.state.engine) == ()
    con = _raw_connect(tmp_db_url)
    try:
        versions = con.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
        assert versions == 1
    finally:
        con.close()


# --- Legacy database -----------------------------------------------------------


def _seed_legacy_actors(tmp_db_url):
    """Create a legacy actors table with tied timestamps in insertion order."""
    tie = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc).replace(tzinfo=None)
    con = _raw_connect(tmp_db_url)
    try:
        con.execute(_LEGACY_SCHEMA)
        rows = [
            ("org-1", "Example Org", "organization", tie),
            ("p-1", "Alice", "person", tie),
            ("org-2", "Other Org", "organization", tie),
        ]
        con.executemany(
            "INSERT INTO actors (id, name, type, created_at) VALUES (?,?,?,?)",
            rows,
        )
        con.commit()
    finally:
        con.close()
    return [row[0] for row in rows]


def test_legacy_database_backfills_seq_from_insertion_order(tmp_db_url):
    ordered_ids = _seed_legacy_actors(tmp_db_url)

    create_app(Settings(database_url=tmp_db_url))

    con = _raw_connect(tmp_db_url)
    try:
        # Every existing identifier is retained unchanged.
        ids = [row[0] for row in con.execute("SELECT id FROM actors")]
        assert set(ids) == set(ordered_ids)
        backfilled = con.execute(
            "SELECT id, seq FROM actors ORDER BY created_at ASC, rowid ASC"
        ).fetchall()
        assert [row[0] for row in backfilled] == ordered_ids
        assert [row[1] for row in backfilled] == [1, 2, 3]
        assert con.execute("SELECT version FROM schema_migrations").fetchall() == [
            (ACTORS_SEQ_VERSION,)
        ]
    finally:
        con.close()


def test_legacy_list_keeps_same_timestamp_order_after_restart(tmp_db_url):
    ordered_ids = _seed_legacy_actors(tmp_db_url)

    app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app) as client:
        body = client.get(ACTORS_PATH).json()
        assert [item["id"] for item in body["items"]] == ordered_ids

        # A newly created actor continues the sequence past the backfill.
        created = create_actor(client, actor_id="dev-1", name="Sensor",
                               type="device")
        body = client.get(ACTORS_PATH).json()
        assert [item["id"] for item in body["items"]] == ordered_ids + ["dev-1"]
        assert body["items"][-1]["created_at"] == created["created_at"]

    # Restart on the migrated file: ordering and sequence survive intact.
    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        body = client.get(ACTORS_PATH).json()
        assert [item["id"] for item in body["items"]] == ordered_ids + ["dev-1"]
        con = _raw_connect(tmp_db_url)
        try:
            seqs = con.execute("SELECT seq FROM actors ORDER BY seq").fetchall()
            assert [row[0] for row in seqs] == [1, 2, 3, 4]
        finally:
            con.close()


def test_legacy_mixed_timestamps_backfill_in_created_order(tmp_db_url):
    con = _raw_connect(tmp_db_url)
    try:
        con.execute(_LEGACY_SCHEMA)
        # Deliberately non-monotonic ids, monotonic timestamps; the sequence
        # must follow creation order, not id ordering.
        con.executemany(
            "INSERT INTO actors (id, name, type, created_at) VALUES (?,?,?,?)",
            [
                ("zzz", "Z", "person", "2026-01-03 00:00:00.000000"),
                ("aaa", "A", "person", "2026-01-01 00:00:00.000000"),
                ("mmm", "M", "person", "2026-01-02 00:00:00.000000"),
            ],
        )
        con.commit()
    finally:
        con.close()

    create_app(Settings(database_url=tmp_db_url))
    con = _raw_connect(tmp_db_url)
    try:
        ordered = con.execute("SELECT id, seq FROM actors ORDER BY seq").fetchall()
        assert [row[0] for row in ordered] == ["aaa", "mmm", "zzz"]
        assert [row[1] for row in ordered] == [1, 2, 3]
    finally:
        con.close()


# --- Failure rollback ----------------------------------------------------------


def _always_fails():
    def _upgrade(conn, ctx):
        raise RuntimeError("simulated migration failure")

    return Migration(
        version="9999", description="intentionally failing migration",
        upgrade=_upgrade,
    )


def test_failed_migration_rolls_back_legacy_database(tmp_db_url):
    _seed_legacy_actors(tmp_db_url)
    engine = make_engine(tmp_db_url)

    with pytest.raises(RuntimeError):
        run_migrations(engine, MIGRATIONS + (_always_fails(),))

    con = _raw_connect(tmp_db_url)
    try:
        tables = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        # No half-applied ledger, no new column, no partial index.
        assert "schema_migrations" not in tables
        assert [row[1] for row in con.execute("PRAGMA table_info(actors)")] == [
            "id",
            "name",
            "type",
            "created_at",
        ]
        indexes = {row[1] for row in con.execute("PRAGMA index_list(actors)")}
        assert "ix_actors_created_order" not in indexes
        assert con.execute("SELECT COUNT(*) FROM actors").fetchone()[0] == 3
    finally:
        con.close()

    # A subsequent healthy startup applies the baseline cleanly.
    assert run_migrations(engine) == (ACTORS_SEQ_VERSION,)
    con = _raw_connect(tmp_db_url)
    try:
        assert con.execute("SELECT seq FROM actors ORDER BY seq").fetchall() == [
            (1,),
            (2,),
            (3,),
        ]
    finally:
        con.close()


def test_failed_fresh_migration_leaves_no_schema(tmp_path):
    db_url = f"sqlite:///{(tmp_path / 'fresh.db').as_posix()}"
    engine = make_engine(db_url)

    with pytest.raises(RuntimeError):
        run_migrations(engine, (_always_fails(),))

    con = _raw_connect(db_url)
    try:
        tables = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert tables == set()
    finally:
        con.close()


# --- SQL-level pagination relies on the migrated column ------------------------


def test_sql_pagination_walks_every_row_without_gap_or_repeat(tmp_db_url):
    ordered_ids = _seed_legacy_actors(tmp_db_url)
    app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app) as client:
        seen: list[str] = []
        cursor = None
        for _ in range(10):
            params = {"limit": "2"}
            if cursor is not None:
                params["cursor"] = cursor
            page = client.get(ACTORS_PATH, params=params).json()
            assert page["count"] == 3
            seen.extend(item["id"] for item in page["items"])
            cursor = page["next_cursor"]
            if cursor is None:
                break
        assert seen == ordered_ids
