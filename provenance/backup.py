"""Disaster-recovery entry points: transaction-consistent backup and restore.

Both operations work directly on SQLite database files through the stdlib
driver -- they never start the HTTP service, never run migrations, and never
write audit rows. The flow is identical in both directions:

* the source (the live database for ``backup``, the snapshot file for
  ``restore``) is opened read-only and is never modified;
* the snapshot is produced in a temporary file inside the destination
  directory -- for ``backup`` via SQLite's online backup API (a
  transaction-consistent copy even while the service is writing), for
  ``restore`` as a byte-exact copy of the validated input;
* the candidate must pass ``PRAGMA integrity_check`` and the migration-ledger
  structural check (a ``schema_migrations`` table with a ``version`` column
  and at least one positive integer version);
* only then is the destination atomically replaced via ``os.replace``.

Any failure removes the temporary file and leaves both the source and any
existing destination untouched: no half-finished output is ever visible at
the target path. An existing destination is never overwritten unless the
caller passes ``--force``.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

#: Error codes surfaced on the single-line stderr JSON envelope.
SOURCE_UNSUPPORTED = "source_unsupported"
SOURCE_NOT_FOUND = "source_not_found"
INPUT_NOT_FOUND = "input_not_found"
DESTINATION_EXISTS = "destination_exists"
INVALID_BACKUP = "invalid_backup"
OPERATION_FAILED = "operation_failed"


@dataclass
class BackupError(Exception):
    """A classified backup/restore failure rendered as the stderr envelope."""

    code: str
    message: str
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.code}: {self.message}"

    def envelope(self) -> str:
        """The single compact stderr line for a failed operation."""
        return json.dumps(
            {
                "error": {
                    "code": self.code,
                    "message": self.message,
                    "details": self.details,
                }
            },
            separators=(",", ":"),
        )


def _sqlite_file_path(database_url: str) -> Path:
    """Extract the file path from a SQLite file URL, rejecting everything else.

    In-memory URLs (``sqlite://`` / ``sqlite:///:memory:``) and non-SQLite
    URLs have no file to snapshot and are refused as unsupported sources.
    """
    if not database_url.startswith("sqlite:///"):
        raise BackupError(
            SOURCE_UNSUPPORTED,
            "only file-backed SQLite database URLs are supported",
            {"database_url_scheme": database_url.split(":", 1)[0]},
        )
    raw_path = database_url.removeprefix("sqlite:///")
    if not raw_path or raw_path == ":memory:":
        raise BackupError(
            SOURCE_UNSUPPORTED,
            "in-memory SQLite databases cannot be backed up or restored",
            {},
        )
    return Path(raw_path)


def _read_only_connect(path: Path) -> sqlite3.Connection:
    """Open a database read-only so the source bytes are never altered."""
    return sqlite3.connect(
        f"file:{quote(str(path))}?mode=ro", uri=True, timeout=30
    )


def _validate_snapshot(path: Path) -> int:
    """Check integrity and ledger structure; return the schema version.

    The schema version is the maximum version recorded in the
    ``schema_migrations`` ledger. A file that is not a SQLite database, fails
    ``PRAGMA integrity_check``, or lacks a well-formed non-empty ledger is
    rejected as an invalid backup.
    """
    try:
        connection = _read_only_connect(path)
    except sqlite3.Error as exc:
        raise BackupError(
            INVALID_BACKUP,
            "snapshot is not a readable SQLite database",
            {"path": str(path), "reason": str(exc)},
        ) from exc
    try:
        try:
            rows = connection.execute("PRAGMA integrity_check").fetchall()
        except sqlite3.Error as exc:
            raise BackupError(
                INVALID_BACKUP,
                "snapshot is not a readable SQLite database",
                {"path": str(path), "reason": str(exc)},
            ) from exc
        if rows != [("ok",)]:
            raise BackupError(
                INVALID_BACKUP,
                "snapshot failed the SQLite integrity check",
                {
                    "path": str(path),
                    "integrity_check": [row[0] for row in rows][:5],
                },
            )
        try:
            table_info = connection.execute(
                "PRAGMA table_info(schema_migrations)"
            ).fetchall()
            versions = [
                row[0]
                for row in connection.execute(
                    "SELECT version FROM schema_migrations"
                ).fetchall()
            ]
        except sqlite3.Error as exc:
            raise BackupError(
                INVALID_BACKUP,
                "snapshot is missing the schema_migrations ledger",
                {"path": str(path), "reason": str(exc)},
            ) from exc
        columns = {row[1] for row in table_info}
        if not table_info or "version" not in columns:
            raise BackupError(
                INVALID_BACKUP,
                "snapshot is missing the schema_migrations ledger",
                {"path": str(path)},
            )
        if not versions or any(
            not isinstance(version, int) or version <= 0 for version in versions
        ):
            raise BackupError(
                INVALID_BACKUP,
                "snapshot has a malformed schema_migrations ledger",
                {"path": str(path)},
            )
        return max(versions)
    finally:
        connection.close()


def _temp_sibling(destination: Path) -> Path:
    """Create an empty temporary file next to the destination."""
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    except OSError as exc:
        raise BackupError(
            OPERATION_FAILED,
            "cannot create a temporary file in the destination directory",
            {"directory": str(destination.parent), "reason": str(exc)},
        ) from exc
    os.close(fd)
    return Path(tmp_name)


def _atomic_replace(candidate: Path, destination: Path) -> None:
    """Atomically move the validated candidate onto the destination."""
    try:
        os.replace(candidate, destination)
    except OSError as exc:
        raise BackupError(
            OPERATION_FAILED,
            "failed to replace the destination atomically",
            {"destination": str(destination), "reason": str(exc)},
        ) from exc


def _resolve_destination(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _check_destination(destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        raise BackupError(
            DESTINATION_EXISTS,
            "destination already exists; pass --force to replace it",
            {"destination": str(destination)},
        )


def perform_backup(database_url: str, output: str, force: bool) -> dict:
    """Snapshot the configured database to ``output`` and describe the result."""
    source = _sqlite_file_path(database_url)
    if not source.is_file():
        raise BackupError(
            SOURCE_NOT_FOUND,
            "database file does not exist",
            {"source": str(source)},
        )
    destination = _resolve_destination(output)
    _check_destination(destination, force)

    candidate = _temp_sibling(destination)
    try:
        try:
            source_connection = _read_only_connect(source)
        except sqlite3.Error as exc:
            raise BackupError(
                OPERATION_FAILED,
                "cannot open the source database",
                {"source": str(source), "reason": str(exc)},
            ) from exc
        try:
            target_connection = sqlite3.connect(str(candidate))
            try:
                # Online backup: a transaction-consistent snapshot even while
                # the service keeps writing to the source.
                try:
                    source_connection.backup(target_connection)
                except sqlite3.OperationalError as exc:
                    raise BackupError(
                        OPERATION_FAILED,
                        "failed to snapshot the source database",
                        {"source": str(source), "reason": str(exc)},
                    ) from exc
                except sqlite3.DatabaseError as exc:
                    raise BackupError(
                        INVALID_BACKUP,
                        "source is corrupt or not a SQLite database",
                        {"source": str(source), "reason": str(exc)},
                    ) from exc
            finally:
                target_connection.close()
        finally:
            source_connection.close()
        schema_version = _validate_snapshot(candidate)
        _atomic_replace(candidate, destination)
    except BaseException:
        candidate.unlink(missing_ok=True)
        raise
    return {
        "operation": "backup",
        "path": str(destination),
        "schema_version": schema_version,
    }


def perform_restore(database_url: str, input_path: str, force: bool) -> dict:
    """Replace the configured database with the validated ``input`` snapshot."""
    source = _sqlite_file_path(database_url)
    snapshot = _resolve_destination(input_path)
    if not snapshot.is_file():
        raise BackupError(
            INPUT_NOT_FOUND,
            "backup input file does not exist",
            {"input": str(snapshot)},
        )
    schema_version = _validate_snapshot(snapshot)
    _check_destination(source, force)

    candidate = _temp_sibling(source)
    try:
        try:
            shutil.copyfile(snapshot, candidate)
        except OSError as exc:
            raise BackupError(
                OPERATION_FAILED,
                "failed to stage the backup snapshot",
                {"input": str(snapshot), "reason": str(exc)},
            ) from exc
        _atomic_replace(candidate, source)
    except BaseException:
        candidate.unlink(missing_ok=True)
        raise
    return {
        "operation": "restore",
        "path": str(source),
        "schema_version": schema_version,
    }
