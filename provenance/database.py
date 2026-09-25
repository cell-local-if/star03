"""SQLAlchemy engine, session, and a timezone-aware UTC column type."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.types import TypeDecorator

# SQLite serializes writes; a single shared connection per in-memory URL is
# required for the test suite's ``StaticPool`` setup. File-backed databases use
# the normal pool. ``check_same_thread`` is needed because FastAPI handlers run
# in worker threads.


class Base(DeclarativeBase):
    """Declarative base for all provenance models."""


class UTCDateTime(TypeDecorator):
    """Stores naive UTC and returns timezone-aware UTC.

    SQLite has no native timestamp-with-timezone type, so values are written as
    naive UTC and converted back to timezone-aware UTC on the way out. Aware
    non-UTC values are normalized to UTC first; naive values are assumed UTC.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=timezone.utc)


def _ensure_sqlite_parent_dir(database_url: str) -> None:
    """Create the parent directory of a file-backed SQLite database URL."""
    if not database_url.startswith("sqlite:///"):
        return
    raw_path = database_url.removeprefix("sqlite:///")
    if not raw_path or raw_path == ":memory:":
        return
    parent = Path(raw_path).expanduser().parent
    parent.mkdir(parents=True, exist_ok=True)


def make_engine(database_url: str):
    """Create an engine tuned for the configured database URL."""
    connect_args: dict[str, object] = {}
    kwargs: dict[str, object] = {"future": True}
    if database_url.startswith("sqlite"):
        # Wait briefly on a busy database instead of failing immediately;
        # SQLite serializes concurrent writers.
        _ensure_sqlite_parent_dir(database_url)
        connect_args["check_same_thread"] = False
        connect_args["timeout"] = 30
        kwargs["connect_args"] = connect_args
        if ":memory:" in database_url:
            # Keep a single connection so an in-memory schema and data survive
            # across sessions (used by deterministic tests).
            from sqlalchemy.pool import StaticPool

            kwargs["poolclass"] = StaticPool
    return create_engine(database_url, **kwargs)


def make_session_factory(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db(engine) -> None:
    """Create all tables and record the current migration version.

    Safe and idempotent on a fresh or existing database. Delegates to the
    versioned migration runner so direct callers and application startup
    share the one schema-establishment path.
    """
    from provenance.migrations import run_migrations

    run_migrations(engine)
