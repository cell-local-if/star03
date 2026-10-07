"""Offline disaster-recovery snapshots: ``backup`` and ``restore``.

Both operations work on quiescent files only -- they never start the HTTP
service, never run migrations, and never write audit rows:

* ``backup`` copies the configured SQLite database (CLI flag >
  ``PROVENANCE_DATABASE_URL`` > default, exactly as the service resolves it)
  into a transactionally consistent snapshot using SQLite's online backup
  API. The snapshot is written to a temporary file in the destination
  directory, verified (``PRAGMA integrity_check`` plus a migration-ledger
  structure check), and only then atomically moved over ``--output``.
* ``restore`` validates ``--input`` the same way and atomically moves a
  verified copy over the file the configured database URL points at. The
  restored file is byte-for-byte the backup: no migration is applied and no
  row is added, so an older ``schema_migrations`` ledger stays older.
* ``verify`` exposes the same read-only validation that the other two
  commands perform internally: it opens ``--input`` read-only, runs
  ``PRAGMA integrity_check`` and the migration-ledger structure check, and
  reports the ledger version without copying or replacing anything. It
  takes no database URL, never starts the service, migrations, or audit
  writes, and creates no file beside or journal file next to the input.

A failure at any step removes the temporary file and leaves both the input
and any pre-existing destination untouched -- no half-written database is
ever visible at the target path. Existing destinations are refused unless
``--force`` is given. Both commands accept file-backed SQLite URLs only:
in-memory databases and non-SQLite URLs are rejected, as is any source that
is not a regular file.

The commands emit a single compact JSON line: on success
``{"operation","path","schema_version"}`` on stdout, on failure
``{"error":{"code","message","details"}}`` on stderr. Neither payload ever
carries keys, signatures, content, or evidence bytes.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

from provenance.config import Settings

#: Stable machine-readable failure codes (the ``error.code`` field).
SOURCE_UNSUPPORTED = "source_unsupported"
SOURCE_NOT_FOUND = "source_not_found"
INPUT_NOT_FOUND = "input_not_found"
DESTINATION_EXISTS = "destination_exists"
INVALID_BACKUP = "invalid_backup"
OPERATION_FAILED = "operation_failed"

_SQLITE_FILE_PREFIX = "sqlite:///"
_LEDGER_TABLE = "schema_migrations"


class BackupError(Exception):
    """A disaster-recovery failure rendered as one JSON error line."""

    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def _sqlite_file_path(database_url: str, *, what: str) -> Path:
    """Resolve a file-backed SQLite URL to an absolute path.

    In-memory databases and non-SQLite URLs are not snapshot-able files and
    are refused as ``source_unsupported``.
    """
    if not database_url.startswith(_SQLITE_FILE_PREFIX):
        raise BackupError(
            SOURCE_UNSUPPORTED,
            f"The {what} must be a file-backed SQLite database URL.",
            {"database_url_scheme": database_url.split(":", 1)[0]},
        )
    raw_path = database_url.removeprefix(_SQLITE_FILE_PREFIX)
    if not raw_path or raw_path == ":memory:":
        raise BackupError(
            SOURCE_UNSUPPORTED,
            f"The {what} must be a file-backed SQLite database URL.",
            {"database_url_scheme": "sqlite"},
        )
    return Path(os.path.abspath(raw_path))


def _absolute_file_argument(raw_path: str, *, missing_code: str, what: str) -> Path:
    """Resolve a CLI file argument, refusing missing and non-file paths."""
    path = Path(os.path.abspath(raw_path))
    if not path.exists():
        raise BackupError(
            missing_code,
            f"The {what} does not exist.",
            {"path": str(path)},
        )
    if not path.is_file():
        raise BackupError(
            SOURCE_UNSUPPORTED,
            f"The {what} is not a regular file.",
            {"path": str(path)},
        )
    return path


def _refuse_existing_destination(path: Path) -> None:
    if path.exists():
        raise BackupError(
            DESTINATION_EXISTS,
            "The destination already exists; pass --force to replace it.",
            {"path": str(path)},
        )


def _validate_snapshot(path: Path) -> int:
    """Verify a snapshot file and return its ledger schema version.

    The file must be a SQLite database that passes ``PRAGMA
    integrity_check`` and carries a well-formed ``schema_migrations``
    ledger: the table exists with an integer ``version`` column and holds at
    least one positive version. Anything else is ``invalid_backup``; the
    file itself is never modified (it is opened read-only).
    """
    try:
        connection = sqlite3.connect(
            f"file:{path}?mode=ro", uri=True, timeout=30
        )
    except sqlite3.Error as exc:
        raise BackupError(
            OPERATION_FAILED,
            "The snapshot could not be opened for verification.",
            {"reason": str(exc)},
        ) from exc
    try:
        try:
            rows = connection.execute("PRAGMA integrity_check").fetchall()
        except sqlite3.DatabaseError as exc:
            raise BackupError(
                INVALID_BACKUP,
                "The snapshot is not a valid SQLite database.",
                {"reason": "integrity_check_unreadable"},
            ) from exc
        if rows != [("ok",)]:
            raise BackupError(
                INVALID_BACKUP,
                "The snapshot failed the SQLite integrity check.",
                {"reason": "integrity_check_failed"},
            )
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if _LEDGER_TABLE not in tables:
            raise BackupError(
                INVALID_BACKUP,
                "The snapshot is missing the schema_migrations ledger.",
                {"reason": "ledger_missing"},
            )
        columns = {
            row[1]
            for row in connection.execute(
                f"PRAGMA table_info({_LEDGER_TABLE})"
            )
        }
        if "version" not in columns:
            raise BackupError(
                INVALID_BACKUP,
                "The schema_migrations ledger has an unexpected structure.",
                {"reason": "ledger_structure_invalid"},
            )
        try:
            versions = [
                row[0]
                for row in connection.execute(
                    f"SELECT version FROM {_LEDGER_TABLE}"
                )
            ]
        except sqlite3.DatabaseError as exc:
            raise BackupError(
                INVALID_BACKUP,
                "The schema_migrations ledger could not be read.",
                {"reason": "ledger_unreadable"},
            ) from exc
        if not versions or any(
            not isinstance(version, int) or version < 1 for version in versions
        ):
            raise BackupError(
                INVALID_BACKUP,
                "The schema_migrations ledger holds no valid version.",
                {"reason": "ledger_version_invalid"},
            )
        return max(versions)
    finally:
        connection.close()


def _snapshot_into(source: Path, staging: Path) -> None:
    """Copy ``source`` into ``staging`` via SQLite's online backup API.

    The backup API reads a transactionally consistent view of the source --
    concurrent writers neither corrupt the snapshot nor are blocked into
    failure. The source is opened read-only so the command can never modify
    it; a source that is not a readable SQLite database surfaces as
    ``invalid_backup`` here or in the staging-file validation that follows.
    """
    try:
        source_connection = sqlite3.connect(
            f"file:{source}?mode=ro", uri=True, timeout=30
        )
    except sqlite3.Error as exc:
        raise BackupError(
            OPERATION_FAILED,
            "The source database could not be opened.",
            {"reason": str(exc)},
        ) from exc
    try:
        try:
            staging_connection = sqlite3.connect(str(staging), timeout=30)
        except sqlite3.Error as exc:
            raise BackupError(
                OPERATION_FAILED,
                "The staging snapshot file could not be created.",
                {"reason": str(exc)},
            ) from exc
        try:
            try:
                source_connection.backup(staging_connection)
            except sqlite3.OperationalError as exc:
                raise BackupError(
                    OPERATION_FAILED,
                    "The snapshot copy failed.",
                    {"reason": str(exc)},
                ) from exc
            except sqlite3.DatabaseError as exc:
                raise BackupError(
                    INVALID_BACKUP,
                    "The source is not a valid SQLite database.",
                    {"reason": str(exc)},
                ) from exc
        finally:
            staging_connection.close()
    finally:
        source_connection.close()


def _replace_atomically(
    copy_source: Path, destination: Path, *, force: bool
) -> int:
    """Stage a verified copy of ``copy_source`` next to ``destination`` and
    move it into place, returning the snapshot's schema version.

    The copy lands in a temporary file inside the destination directory (so
    the final ``os.replace`` is atomic on the same filesystem), is validated
    there, and only then replaces the destination. Any failure removes the
    temporary file and leaves the existing destination untouched.
    """
    if not force:
        _refuse_existing_destination(destination)
    parent = destination.parent
    try:
        # Same convention as the service: the configured database location's
        # parent directories are created on demand.
        parent.mkdir(parents=True, exist_ok=True)
        fd, staging_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=str(parent)
        )
    except OSError as exc:
        raise BackupError(
            OPERATION_FAILED,
            "The destination directory is not writable.",
            {"path": str(parent), "reason": str(exc)},
        ) from exc
    os.close(fd)
    staging = Path(staging_name)
    try:
        _snapshot_into(copy_source, staging)
        schema_version = _validate_snapshot(staging)
        try:
            os.replace(staging, destination)
        except OSError as exc:
            raise BackupError(
                OPERATION_FAILED,
                "The destination could not be replaced.",
                {"path": str(destination), "reason": str(exc)},
            ) from exc
        return schema_version
    except BaseException:
        # Never leave a half-finished snapshot behind; the pre-existing
        # destination (if any) is still in place because the atomic replace
        # either happened fully or not at all.
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def run_backup(settings: Settings, output: str, *, force: bool) -> dict:
    """Snapshot the configured database to ``output``; return the report."""
    source = _sqlite_file_path(settings.database_url, what="backup source")
    source = _absolute_file_argument(
        str(source), missing_code=SOURCE_NOT_FOUND, what="backup source"
    )
    destination = Path(os.path.abspath(output))
    schema_version = _replace_atomically(source, destination, force=force)
    return {
        "operation": "backup",
        "path": str(destination),
        "schema_version": schema_version,
    }


def run_restore(settings: Settings, input_path: str, *, force: bool) -> dict:
    """Replace the configured database with a verified copy of ``input_path``."""
    target = _sqlite_file_path(settings.database_url, what="restore target")
    source = _absolute_file_argument(
        input_path, missing_code=INPUT_NOT_FOUND, what="restore input"
    )
    schema_version = _replace_atomically(source, target, force=force)
    return {
        "operation": "restore",
        "path": str(target),
        "schema_version": schema_version,
    }


def run_verify(input_path: str) -> dict:
    """Verify a snapshot file read-only and report its ledger version.

    No database URL is consulted and nothing is copied, replaced, or
    created: the input is resolved exactly like a restore ``--input`` and
    passed through the same validation as every internally staged backup.
    Repeated runs over an unchanged file return an identical report.
    """
    source = _absolute_file_argument(
        input_path, missing_code=INPUT_NOT_FOUND, what="verify input"
    )
    schema_version = _validate_snapshot(source)
    return {
        "operation": "verify",
        "path": str(source),
        "schema_version": schema_version,
    }
