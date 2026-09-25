"""Versioned, restart-safe SQLite schema migrations.

Each migration is an ordered, named step that upgrades the schema exactly
once. Applied versions are recorded in the ``schema_migrations`` table; a
second startup sees the recorded versions and skips them, so an already
applied migration never runs twice and existing identifiers (actor ids and
every other stable resource id) are never changed.

Two starting states are recognized:

* a brand-new database with no ``actors`` table -- the current metadata is
  created wholesale, so the table is born with every current column and
  index and the baseline version is recorded without a backfill;
* a database already carrying the legacy ``actors`` table -- the missing
  explicit ordering column is added in place and existing rows are
  backfilled from their original insertion order (``created_at`` first,
  the stable SQLite ``rowid`` as the tiebreaker), so actors created at the
  same instant keep the order in which they were inserted.

All pending migrations for one startup run inside a single transaction.
SQLite participates DDL in transactions, so any failure rolls the whole
startup back to its pre-migration state: no half-filled version row, no
partially added column, no partial index survives a failed upgrade.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import event, text
from sqlalchemy.engine import Connection, Engine

#: Table recording exactly which migration versions have been applied.
MIGRATIONS_TABLE = "schema_migrations"

#: Baseline ordering version: actors gain an explicit persistent sequence.
ACTORS_SEQ_VERSION = "0001"

_CREATE_MIGRATIONS_TABLE = text(
    f"""
    CREATE TABLE IF NOT EXISTS {MIGRATIONS_TABLE} (
        version VARCHAR(255) NOT NULL PRIMARY KEY,
        applied_at DATETIME NOT NULL
    )
    """
)

_SELECT_APPLIED = text(f"SELECT version FROM {MIGRATIONS_TABLE}")

_INSERT_APPLIED = text(
    f"INSERT INTO {MIGRATIONS_TABLE} (version, applied_at) VALUES (:version, :applied_at)"
)

_TABLE_EXISTS = text(
    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = :name"
)


@dataclass(frozen=True)
class MigrationContext:
    """The pre-migration shape of the database visible to one migration."""

    #: Whether the legacy ``actors`` table already exists.
    actors_table_exists: bool
    #: Whether ``actors`` already carries the explicit ``seq`` column.
    actors_seq_exists: bool


@dataclass(frozen=True)
class Migration:
    """One ordered schema upgrade: a version label and its upgrade body."""

    version: str
    description: str
    upgrade: Callable[[Connection, MigrationContext], None]


def _table_exists(conn: Connection, name: str) -> bool:
    return conn.execute(_TABLE_EXISTS, {"name": name}).first() is not None


def _column_exists(conn: Connection, table: str, column: str) -> bool:
    rows = conn.execute(text(f"PRAGMA table_info({table})")).all()
    return any(row[1] == column for row in rows)


def _create_current_metadata(conn: Connection) -> None:
    """Create every current table on a brand-new database.

    Imported lazily so the migration module never imports the ORM models at
    module load time (the models import the database base).
    """
    # Import models so they are registered on ``Base.metadata`` first.
    from provenance import models  # noqa: F401
    from provenance.database import Base

    Base.metadata.create_all(bind=conn, checkfirst=True)


def _upgrade_actors_seq(conn: Connection, ctx: MigrationContext) -> None:
    """Add the explicit actor ordering column and backfill existing rows."""
    if not ctx.actors_table_exists:
        # Brand-new database: the current metadata creates ``actors`` with
        # the ``seq`` column and both indexes already in place; only the
        # other current tables still need creating.
        _create_current_metadata(conn)
        return

    if not ctx.actors_seq_exists:
        # Legacy database: add the column nullable (the only SQLite add-
        # column form), then fill it from the original insertion order.
        # The correlated scalar subquery numbers rows by created_at with
        # the stable rowid tiebreaker, so equal timestamps keep the order
        # in which the actors were originally inserted.
        conn.execute(text("ALTER TABLE actors ADD COLUMN seq BIGINT"))
        conn.execute(
            text(
                """
                UPDATE actors
                SET seq = (
                    SELECT new_seq
                    FROM (
                        SELECT id,
                               ROW_NUMBER() OVER (
                                   ORDER BY created_at ASC, rowid ASC
                               ) AS new_seq
                        FROM actors
                    ) AS ordered
                    WHERE ordered.id = actors.id
                )
                """
            )
        )

    # Match the indexes the current metadata declares on a fresh table;
    # IF NOT EXISTS covers a database stopped after the column/index steps
    # of an older deployment without a recorded version.
    conn.execute(
        text("CREATE UNIQUE INDEX IF NOT EXISTS uq_actors_seq ON actors (seq)")
    )
    conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_actors_created_order "
            "ON actors (created_at, seq)"
        )
    )

    # A genuine legacy database already carries every other table; this is
    # a no-op there. It also makes a database containing only the legacy
    # actors table whole without touching any existing identifier.
    _create_current_metadata(conn)


#: The ordered migration history. Append new versions here; never edit or
#: reorder a released version.
MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=ACTORS_SEQ_VERSION,
        description="add explicit persistent ordering column to actors",
        upgrade=_upgrade_actors_seq,
    ),
)


@contextlib.contextmanager
def _migration_connection(engine: Engine):
    """Yield a connection whose DDL participates in one transaction.

    The SQLite ``pysqlite`` driver normally COMMITs just before a DDL
    statement, which would leave a half-applied schema behind after a
    failed migration. Rather than alter the application engine's dialect
    behavior (which the request path relies on), the migration borrows the
    engine's underlying DBAPI connection -- the same single connection for
    an in-memory database, so the migrated schema is exactly the one the
    application then uses -- and drives it through a short-lived engine
    configured for explicit transactional DDL. The connection is returned
    to the original pool afterward with its isolation setting restored.

    Non-SQLite engines already provide transactional DDL, so they are used
    directly.
    """
    if engine.dialect.name != "sqlite":
        with engine.connect() as conn:
            yield conn
        return

    # Borrow the real DBAPI connection without opening an SQLAlchemy
    # transaction on the application engine.
    pooled = engine.raw_connection()
    dbapi_connection = pooled.dbapi_connection
    original_isolation = dbapi_connection.isolation_level

    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    migration_engine = create_engine(
        "sqlite://",
        creator=lambda: dbapi_connection,
        poolclass=StaticPool,
    )

    # Put the driver into explicit-transaction mode for this engine only.
    @event.listens_for(migration_engine, "begin")
    def _begin(conn):  # pragma: no cover - trivial listener
        conn.exec_driver_sql("BEGIN")

    migration_conn = migration_engine.connect()
    try:
        dbapi_connection.isolation_level = None
        yield migration_conn
    finally:
        # Restore the driver before handing the connection back so the
        # application engine sees its original implicit-transaction mode.
        # ``migration_conn.close()`` returns (never closes) the borrowed
        # connection to the StaticPool; deliberately do not dispose the
        # throwaway engine, since disposing would close that connection --
        # the sole in-memory database under StaticPool.
        dbapi_connection.isolation_level = original_isolation
        migration_conn.close()
        pooled.close()


def run_migrations(
    engine: Engine,
    migrations: tuple[Migration, ...] = MIGRATIONS,
) -> tuple[str, ...]:
    """Apply every not-yet-recorded migration and return the versions applied.

    Safe to call repeatedly: recorded versions are skipped and never run a
    second time. The full pending batch -- including the migration ledger
    -- commits in one transaction; a failure rolls back to the pre-startup
    state with no version marker, column, or index left behind.
    """
    # Imported here so the stored timestamp uses the same UTC convention as
    # the rest of the service without pulling the models in at import time.
    from provenance.time_utils import utc_now

    applied_this_run: list[str] = []
    with _migration_connection(engine) as conn:
        transaction = conn.begin()
        try:
            conn.execute(_CREATE_MIGRATIONS_TABLE)
            applied = {row[0] for row in conn.execute(_SELECT_APPLIED)}

            actors_exists = _table_exists(conn, "actors")
            ctx = MigrationContext(
                actors_table_exists=actors_exists,
                actors_seq_exists=(
                    _column_exists(conn, "actors", "seq")
                    if actors_exists
                    else False
                ),
            )

            for migration in migrations:
                if migration.version in applied:
                    continue
                migration.upgrade(conn, ctx)
                # A migration that creates the schema wholesale (fresh DB)
                # makes every later context observation current.
                ctx = MigrationContext(
                    actors_table_exists=True,
                    actors_seq_exists=True,
                )
                conn.execute(
                    _INSERT_APPLIED,
                    {
                        "version": migration.version,
                        "applied_at": utc_now().replace(tzinfo=None),
                    },
                )
                applied_this_run.append(migration.version)
            transaction.commit()
        except Exception:
            transaction.rollback()
            raise

    return tuple(applied_this_run)
