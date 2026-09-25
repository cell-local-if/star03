"""Versioned SQLite migrations with an explicit persistence-order column.

The baseline shipped without a migration ledger or an explicit ordering
column: actors broke same-``created_at`` ties on SQLite's implicit
``rowid``. This module introduces both, applied at startup from zero:

* a ``schema_migrations`` ledger records every applied version exactly once;
* version 1 adds ``actors.display_seq`` and an index on
  ``(created_at, display_seq)``, backfilling the explicit order from the
  original insertion order (``rowid``) so same-timestamp rows stay stable,
  and installs an ``AFTER INSERT`` trigger that stamps every later actor with
  the next position.

The runner distinguishes a brand-new database (no ``actors`` table yet) from
an existing one (an ``actors`` table that predates the ledger). Both paths
are idempotent and never re-run an applied version, never mutate an existing
actor's public identity, and run the whole startup migration batch inside a
single transaction so any failure rolls back to the pre-startup state --
leaving no half-finished version row and no partially added column.

SQLite executes DDL transactionally, but the stock pysqlite driver's legacy
transaction control would autocommit a leading DDL statement even inside an
ORM/``engine.begin()`` transaction. The batch therefore runs on a raw DBAPI
connection with the driver's implicit transaction shimming disabled, under
one explicit ``BEGIN IMMEDIATE``/``COMMIT``/``ROLLBACK``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sqlalchemy.schema import CreateIndex, CreateTable

#: First schema generation: explicit actor ordering column, backfill, index,
#: and the insert trigger that continues the sequence for new actors.
SCHEMA_VERSION_1 = 1

#: Name of the AFTER INSERT trigger that assigns display_seq to new actors.
ACTOR_SEQ_TRIGGER = "trg_actors_display_seq"

#: Statements that continue the explicit actor sequence on every insert. The
#: trigger fires only when the writer did not already supply a positive
#: position (the application always leaves the column at its default 0); the
#: new row's own zero participates in the MAX, so a first-ever insert yields
#: 1 and later inserts resume at MAX + 1.
_CREATE_ACTOR_SEQ_TRIGGER = f"""
CREATE TRIGGER IF NOT EXISTS {ACTOR_SEQ_TRIGGER}
AFTER INSERT ON actors
FOR EACH ROW
WHEN NEW.display_seq = 0
BEGIN
    UPDATE actors
       SET display_seq = (
               SELECT COALESCE(MAX(display_seq), 0) + 1 FROM actors
           )
     WHERE rowid = NEW.rowid;
END
""".strip()


@dataclass(frozen=True)
class Migration:
    """One versioned change applied to a pre-existing (legacy) database."""

    version: int
    #: Runs inside the shared version transaction, receiving the raw DBAPI
    #: cursor, the engine (for DDL compilation), and the declarative metadata.
    up: Callable[[object, object, object], None]


def _existing_tables(cursor) -> set[str]:
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )
    return {row[0] for row in cursor.fetchall()}


def _table_has_column(cursor, table: str, column: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def _create_full_schema(cursor, engine, metadata) -> None:
    """Create every metadata table and index at its current (latest) shape."""
    for table in metadata.sorted_tables:
        cursor.execute(str(CreateTable(table).compile(engine)))
    for table in metadata.sorted_tables:
        for index in table.indexes:
            cursor.execute(str(CreateIndex(index).compile(engine)))


def _migration_1_up(cursor, engine, metadata) -> None:
    """Bring a legacy ``actors`` table to the explicit-order shape.

    Only a pre-existing database reaches this function; a fresh database is
    created directly at the latest shape. Each step is individually
    idempotent in addition to the surrounding transaction, so a re-run after
    external intervention neither fails nor double-backfills.
    """
    if not _table_has_column(cursor, "actors", "display_seq"):
        cursor.execute(
            "ALTER TABLE actors ADD COLUMN display_seq BIGINT "
            "NOT NULL DEFAULT 0"
        )
        # Existing rows keep their original insertion order: the number of
        # rows at or before each rowid is a dense 1..N ranking that is stable
        # even when several rows share a created_at.
        cursor.execute(
            "UPDATE actors SET display_seq = ("
            "SELECT COUNT(*) FROM actors AS a2 WHERE a2.rowid <= actors.rowid)"
        )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS ix_actors_created_order "
        "ON actors (created_at, display_seq)"
    )
    cursor.execute(_CREATE_ACTOR_SEQ_TRIGGER)


#: Known versions in application order. A fresh database is baselined past
#: all of them; a legacy database applies each pending one in turn.
MIGRATIONS: tuple[Migration, ...] = (
    Migration(version=SCHEMA_VERSION_1, up=_migration_1_up),
)

_CREATE_LEDGER = (
    "CREATE TABLE IF NOT EXISTS schema_migrations ("
    "version INTEGER NOT NULL PRIMARY KEY, "
    "applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)"
)


def _run_batch(engine, work: Callable[[object], set[int]]) -> None:
    """Run ``work`` on a raw connection under one explicit transaction.

    ``work`` receives the raw DBAPI cursor and returns the set of applied
    versions. Its DDL/DML plus the ledger reads/writes it performs share the
    single ``BEGIN IMMEDIATE`` transaction: any exception rolls everything
    back before it propagates, leaving no ledger row and no schema change.
    """
    fairy = engine.raw_connection()
    dbapi = fairy.dbapi_connection
    previous_isolation = dbapi.isolation_level
    cursor = dbapi.cursor()
    committed = False
    try:
        # Disable pysqlite's implicit transaction handling so the explicit
        # BEGIN below is the only transaction boundary (DDL included).
        dbapi.isolation_level = None
        cursor.execute("BEGIN IMMEDIATE")
        work(cursor)
        cursor.execute("COMMIT")
        committed = True
    except BaseException:
        if not committed:
            cursor.execute("ROLLBACK")
        raise
    finally:
        cursor.close()
        try:
            dbapi.isolation_level = previous_isolation
        finally:
            fairy.close()


def run_migrations(engine) -> None:
    """Bring the database up to the latest known schema version.

    Safe to call on every startup.

    * Brand-new database (no ``actors`` table): the full current schema is
      created -- including ``display_seq``, its index, and the insert
      trigger -- and every known version is recorded as baselined.
    * Legacy database (an ``actors`` table predating the ledger): pending
      versions are applied in order; version 1 adds the column, backfills the
      original insertion order, adds the index, and installs the trigger.

    Already-recorded versions are never re-executed and existing actor
    identifiers/fields are never changed. Any failure rolls the entire batch
    back to the pre-startup state.
    """
    from provenance import models  # noqa: F401  register tables on metadata
    from provenance.database import Base

    metadata = Base.metadata

    def _work(cursor) -> None:
        cursor.execute(_CREATE_LEDGER)
        cursor.execute("SELECT version FROM schema_migrations")
        applied = {int(row[0]) for row in cursor.fetchall()}

        fresh = "actors" not in _existing_tables(cursor)

        if fresh:
            # Create directly at the latest shape, then stamp every version.
            _create_full_schema(cursor, engine, metadata)
            cursor.execute(_CREATE_ACTOR_SEQ_TRIGGER)
            for migration in MIGRATIONS:
                if migration.version in applied:
                    continue
                cursor.execute(
                    "INSERT INTO schema_migrations (version) VALUES (?)",
                    (migration.version,),
                )
            return

        for migration in MIGRATIONS:
            if migration.version in applied:
                continue
            migration.up(cursor, engine, metadata)
            cursor.execute(
                "INSERT INTO schema_migrations (version) VALUES (?)",
                (migration.version,),
            )

    _run_batch(engine, _work)
