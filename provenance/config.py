"""Explicit runtime configuration.

The database location is never guessed from an ambiguous environment: callers
name it through the constructor (:class:`Settings`), the ``PROVENANCE_DATABASE_URL``
environment variable, or the CLI flag. A relative SQLite path is resolved
against the current working directory and surfaced as an absolute path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DATABASE_URL_ENV = "PROVENANCE_DATABASE_URL"
DEFAULT_DATABASE_URL = "sqlite:///./provenance.db"


def _normalize_sqlite_url(url: str) -> str:
    """Return an absolute path-based SQLite URL for a relative ``sqlite:///`` one.

    In-memory databases (``sqlite://`` / ``sqlite:///:memory:``) and URLs that
    already carry an absolute path are returned unchanged.
    """
    if not url.startswith("sqlite:///"):
        return url
    raw_path = url.removeprefix("sqlite:///")
    if not raw_path or raw_path == ":memory:":
        return url
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return f"sqlite:///{path.as_posix()}"


@dataclass(frozen=True)
class Settings:
    """Runtime settings resolved in constructor > env > default order."""

    database_url: str = DEFAULT_DATABASE_URL

    @classmethod
    def from_env(cls, database_url: str | None = None) -> "Settings":
        resolved = database_url or os.environ.get(DATABASE_URL_ENV) or DEFAULT_DATABASE_URL
        return cls(database_url=_normalize_sqlite_url(resolved))
