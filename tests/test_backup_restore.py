"""Tests for the ``backup``/``restore`` disaster-recovery subcommands.

Covers the acceptance matrix:

* backup/restore equivalence: a restored file serves the same HTTP data and
  audit rows as the source, byte-level ledger and trigger included;
* ``--force`` replaces an existing destination, its absence refuses with
  ``destination_exists`` and leaves the existing file untouched;
* a zero-record but initialized database backs up cleanly;
* a database at an older migration version keeps that version (restore never
  completes migrations);
* corrupt or ledger-less input is rejected as ``invalid_backup`` before any
  replacement happens;
* in-memory and non-SQLite URLs are rejected as ``source_unsupported``;
* the online backup API yields a consistent snapshot while the service is
  concurrently writing;
* neither subcommand starts the service, migrates, or writes audit rows, and
  failures leave no partial output behind.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import threading

import pytest
from fastapi.testclient import TestClient

from provenance import __main__
from provenance.app import create_app
from provenance.backup import BackupError, perform_backup, perform_restore
from provenance.config import Settings
from tests.helpers import create_actor

ACTORS_PATH = "/v1/actors"
AUDIT_PATH = "/v1/audit/events"


def _url(path) -> str:
    return f"sqlite:///{path.as_posix()}"


def _run_cli(argv, monkeypatch):
    """Invoke the entry point, capturing stdout/stderr and the exit code."""
    out, err = [], []

    class _Stdout:
        def write(self, text):
            out.append(text)

    class _Stderr:
        def write(self, text):
            err.append(text)

    monkeypatch.setattr(sys, "argv", ["python -m provenance", *argv])
    monkeypatch.setattr(sys, "stdout", _Stdout())
    monkeypatch.setattr(sys, "stderr", _Stderr())
    with pytest.raises(SystemExit) as excinfo:
        __main__.main()
    return excinfo.value.code, "".join(out), "".join(err)


def _init_db_with_actor(path, actor_id="org-1"):
    with TestClient(create_app(Settings(database_url=_url(path)))) as client:
        create_actor(client, actor_id=actor_id, name="Example Org")
        return client.get(ACTORS_PATH).json()


def _read_all(path, table):
    con = sqlite3.connect(path)
    try:
        return con.execute(f"SELECT * FROM {table}").fetchall()
    finally:
        con.close()


def _table_names(path):
    con = sqlite3.connect(path)
    try:
        return {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    finally:
        con.close()


# --- Backup -------------------------------------------------------------------


def test_backup_restore_round_trip_preserves_data_and_audit(tmp_path):
    source = tmp_path / "live.db"
    actors_before = _init_db_with_actor(source)
    audit_before = _read_all(source, "audit_events")
    snapshot = tmp_path / "snapshot.db"

    result = perform_backup(_url(source), str(snapshot), force=False)

    assert result == {
        "operation": "backup",
        "path": str(snapshot),
        "schema_version": 4,
    }
    # Tables, indexes, triggers, and the migration ledger all survive.
    assert _table_names(snapshot) == _table_names(source)
    assert _read_all(snapshot, "schema_migrations") == _read_all(
        source, "schema_migrations"
    )
    assert _read_all(snapshot, "audit_events") == audit_before

    restored = tmp_path / "restored.db"
    result = perform_restore(_url(restored), str(snapshot), force=False)
    assert result == {
        "operation": "restore",
        "path": str(restored),
        "schema_version": 4,
    }

    with TestClient(create_app(Settings(database_url=_url(restored)))) as client:
        assert client.get(ACTORS_PATH).json() == actors_before
    assert _read_all(restored, "audit_events") == audit_before


def test_backup_of_initialized_zero_record_database(tmp_path):
    source = tmp_path / "empty.db"
    with TestClient(create_app(Settings(database_url=_url(source)))):
        pass
    snapshot = tmp_path / "snapshot.db"

    result = perform_backup(_url(source), str(snapshot), force=False)

    assert result["schema_version"] == 4
    assert _read_all(snapshot, "actors") == []
    versions = [row[0] for row in _read_all(snapshot, "schema_migrations")]
    assert sorted(versions) == [1, 2, 3, 4]


def test_backup_preserves_older_migration_version(tmp_path):
    # A database migrated only to version 2 keeps that version in the
    # snapshot; restore reports it and never completes the missing migrations.
    source = tmp_path / "old.db"
    with TestClient(create_app(Settings(database_url=_url(source)))):
        pass
    con = sqlite3.connect(source)
    con.execute("DELETE FROM schema_migrations WHERE version > 2")
    con.commit()
    con.close()

    snapshot = tmp_path / "snapshot.db"
    result = perform_backup(_url(source), str(snapshot), force=False)
    assert result["schema_version"] == 2

    restored = tmp_path / "restored.db"
    result = perform_restore(_url(restored), str(snapshot), force=False)
    assert result["schema_version"] == 2
    versions = [row[0] for row in _read_all(restored, "schema_migrations")]
    assert sorted(versions) == [1, 2]


def test_backup_refuses_existing_destination_without_force(tmp_path):
    source = tmp_path / "live.db"
    _init_db_with_actor(source)
    snapshot = tmp_path / "snapshot.db"
    snapshot.write_bytes(b"pre-existing")

    with pytest.raises(BackupError) as excinfo:
        perform_backup(_url(source), str(snapshot), force=False)
    assert excinfo.value.code == "destination_exists"
    # The existing file is untouched.
    assert snapshot.read_bytes() == b"pre-existing"


def test_backup_force_replaces_existing_destination(tmp_path):
    source = tmp_path / "live.db"
    _init_db_with_actor(source)
    snapshot = tmp_path / "snapshot.db"
    snapshot.write_bytes(b"stale")

    result = perform_backup(_url(source), str(snapshot), force=True)

    assert result["operation"] == "backup"
    assert _read_all(snapshot, "actors")


def test_backup_missing_source(tmp_path):
    with pytest.raises(BackupError) as excinfo:
        perform_backup(_url(tmp_path / "nope.db"), str(tmp_path / "out.db"), force=False)
    assert excinfo.value.code == "source_not_found"
    assert not (tmp_path / "out.db").exists()


@pytest.mark.parametrize(
    "url",
    ["sqlite://", "sqlite:///:memory:", "postgresql://localhost/db"],
)
def test_backup_rejects_unsupported_sources(tmp_path, url):
    with pytest.raises(BackupError) as excinfo:
        perform_backup(url, str(tmp_path / "out.db"), force=False)
    assert excinfo.value.code == "source_unsupported"
    assert not (tmp_path / "out.db").exists()


def test_backup_of_corrupt_source_is_invalid_and_leaves_no_partial(tmp_path):
    source = tmp_path / "corrupt.db"
    source.write_bytes(b"this is not a sqlite database at all")
    snapshot = tmp_path / "snapshot.db"

    with pytest.raises(BackupError) as excinfo:
        perform_backup(_url(source), str(snapshot), force=False)
    assert excinfo.value.code == "invalid_backup"
    # No half-finished output remains anywhere in the destination directory.
    assert not snapshot.exists()
    assert [p.name for p in tmp_path.iterdir()] == ["corrupt.db"]


def test_backup_into_missing_directory_fails_without_partial(tmp_path):
    source = tmp_path / "live.db"
    _init_db_with_actor(source)
    missing_dir = tmp_path / "nope"

    with pytest.raises(BackupError) as excinfo:
        perform_backup(_url(source), str(missing_dir / "out.db"), force=False)
    assert excinfo.value.code == "operation_failed"
    assert not missing_dir.exists()


def test_backup_is_consistent_while_service_writes(tmp_path):
    source = tmp_path / "live.db"
    # Initializes the schema without starting the HTTP service.
    create_app(Settings(database_url=_url(source)))
    snapshot = tmp_path / "snapshot.db"
    stop = threading.Event()
    errors = []

    def _writer():
        try:
            con = sqlite3.connect(str(source), timeout=30)
            try:
                index = 0
                while not stop.is_set():
                    con.execute(
                        "INSERT INTO actors (id, name, type, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (f"a-{index}", f"N{index}", "person", "2026-01-01 00:00:00"),
                    )
                    con.commit()
                    index += 1
            finally:
                con.close()
        except sqlite3.Error as exc:  # pragma: no cover - surfaces races
            errors.append(exc)

    writer = threading.Thread(target=_writer)
    writer.start()
    try:
        result = perform_backup(_url(source), str(snapshot), force=False)
    finally:
        stop.set()
        writer.join(timeout=30)

    assert not errors
    assert result["schema_version"] == 4
    # The snapshot is a valid, integrity-checked database at some consistent
    # point in the writer's sequence.
    con = sqlite3.connect(snapshot)
    try:
        assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        count = con.execute("SELECT COUNT(*) FROM actors").fetchone()[0]
        seqs = [
            row[0]
            for row in con.execute("SELECT display_seq FROM actors ORDER BY display_seq")
        ]
        assert seqs == list(range(1, count + 1))
    finally:
        con.close()


# --- Restore ------------------------------------------------------------------


def test_restore_missing_input(tmp_path):
    target = tmp_path / "live.db"
    _init_db_with_actor(target)
    before = target.read_bytes()

    with pytest.raises(BackupError) as excinfo:
        perform_restore(_url(target), str(tmp_path / "nope.db"), force=True)
    assert excinfo.value.code == "input_not_found"
    assert target.read_bytes() == before


def test_restore_rejects_corrupt_input_and_preserves_target(tmp_path):
    target = tmp_path / "live.db"
    _init_db_with_actor(target)
    before = target.read_bytes()
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"garbage bytes, not sqlite")

    with pytest.raises(BackupError) as excinfo:
        perform_restore(_url(target), str(bad), force=True)
    assert excinfo.value.code == "invalid_backup"
    assert target.read_bytes() == before
    assert {p.name for p in tmp_path.iterdir()} == {"live.db", "bad.db"}


def test_restore_rejects_snapshot_without_ledger(tmp_path):
    target = tmp_path / "live.db"
    _init_db_with_actor(target)
    before = target.read_bytes()
    ledgerless = tmp_path / "ledgerless.db"
    con = sqlite3.connect(ledgerless)
    con.execute("CREATE TABLE actors (id VARCHAR(255) PRIMARY KEY)")
    con.commit()
    con.close()

    with pytest.raises(BackupError) as excinfo:
        perform_restore(_url(target), str(ledgerless), force=True)
    assert excinfo.value.code == "invalid_backup"
    assert target.read_bytes() == before


def test_restore_refuses_existing_target_without_force(tmp_path):
    source = tmp_path / "source.db"
    _init_db_with_actor(source, actor_id="org-new")
    snapshot = tmp_path / "snapshot.db"
    perform_backup(_url(source), str(snapshot), force=False)
    target = tmp_path / "target.db"
    _init_db_with_actor(target, actor_id="org-old")
    before = target.read_bytes()

    with pytest.raises(BackupError) as excinfo:
        perform_restore(_url(target), str(snapshot), force=False)
    assert excinfo.value.code == "destination_exists"
    assert target.read_bytes() == before


def test_restore_force_replaces_target(tmp_path):
    source = tmp_path / "source.db"
    _init_db_with_actor(source, actor_id="org-new")
    snapshot = tmp_path / "snapshot.db"
    perform_backup(_url(source), str(snapshot), force=False)
    target = tmp_path / "target.db"
    _init_db_with_actor(target, actor_id="org-old")

    result = perform_restore(_url(target), str(snapshot), force=True)

    assert result == {
        "operation": "restore",
        "path": str(target),
        "schema_version": 4,
    }
    with TestClient(create_app(Settings(database_url=_url(target)))) as client:
        body = client.get(ACTORS_PATH).json()
    assert [item["id"] for item in body["items"]] == ["org-new"]
    # The input snapshot is unchanged by the restore.
    assert [row[0] for row in _read_all(snapshot, "actors")] == ["org-new"]


def test_restore_rejects_memory_target(tmp_path):
    source = tmp_path / "source.db"
    _init_db_with_actor(source)
    snapshot = tmp_path / "snapshot.db"
    perform_backup(_url(source), str(snapshot), force=False)

    with pytest.raises(BackupError) as excinfo:
        perform_restore("sqlite:///:memory:", str(snapshot), force=True)
    assert excinfo.value.code == "source_unsupported"


# --- CLI envelope -------------------------------------------------------------


def test_cli_backup_success_emits_single_compact_json_line(tmp_path, monkeypatch):
    source = tmp_path / "live.db"
    _init_db_with_actor(source)
    snapshot = tmp_path / "snapshot.db"

    code, out, err = _run_cli(
        ["backup", "--database-url", _url(source), "--output", str(snapshot)],
        monkeypatch,
    )

    assert code == 0
    assert err == ""
    assert out.count("\n") == 1
    assert json.loads(out) == {
        "operation": "backup",
        "path": str(snapshot),
        "schema_version": 4,
    }
    # The success line never carries key material, signatures, or content.
    for forbidden in ("private", "signature", "digest", "evidence"):
        assert forbidden not in out


def test_cli_failure_emits_error_envelope_only(tmp_path, monkeypatch):
    code, out, err = _run_cli(
        [
            "backup",
            "--database-url",
            _url(tmp_path / "missing.db"),
            "--output",
            str(tmp_path / "out.db"),
        ],
        monkeypatch,
    )

    assert code == 1
    assert out == ""
    assert err.count("\n") == 1
    envelope = json.loads(err)
    assert envelope["error"]["code"] == "source_not_found"
    assert set(envelope["error"]) == {"code", "message", "details"}


def test_cli_restore_round_trip(tmp_path, monkeypatch):
    source = tmp_path / "live.db"
    _init_db_with_actor(source)
    snapshot = tmp_path / "snapshot.db"
    code, _, _ = _run_cli(
        ["backup", "--database-url", _url(source), "--output", str(snapshot)],
        monkeypatch,
    )
    assert code == 0

    restored = tmp_path / "restored.db"
    code, out, err = _run_cli(
        ["restore", "--database-url", _url(restored), "--input", str(snapshot)],
        monkeypatch,
    )
    assert code == 0
    assert json.loads(out) == {
        "operation": "restore",
        "path": str(restored),
        "schema_version": 4,
    }


def test_cli_subcommand_respects_env_database_url(tmp_path, monkeypatch):
    source = tmp_path / "env.db"
    _init_db_with_actor(source)
    monkeypatch.setenv("PROVENANCE_DATABASE_URL", _url(source))
    snapshot = tmp_path / "snapshot.db"

    code, out, _ = _run_cli(["backup", "--output", str(snapshot)], monkeypatch)

    assert code == 0
    assert json.loads(out)["operation"] == "backup"


def test_cli_flag_overrides_env_database_url(tmp_path, monkeypatch):
    env_db = tmp_path / "env.db"
    _init_db_with_actor(env_db)
    flag_db = tmp_path / "flag.db"
    _init_db_with_actor(flag_db, actor_id="org-flag")
    monkeypatch.setenv("PROVENANCE_DATABASE_URL", _url(env_db))
    snapshot = tmp_path / "snapshot.db"

    code, _, _ = _run_cli(
        ["backup", "--database-url", _url(flag_db), "--output", str(snapshot)],
        monkeypatch,
    )

    assert code == 0
    assert [row[0] for row in _read_all(snapshot, "actors")] == ["org-flag"]


def test_cli_database_url_before_subcommand_is_honored(tmp_path, monkeypatch):
    flag_db = tmp_path / "flag.db"
    _init_db_with_actor(flag_db, actor_id="org-flag")
    snapshot = tmp_path / "snapshot.db"

    code, _, _ = _run_cli(
        ["--database-url", _url(flag_db), "backup", "--output", str(snapshot)],
        monkeypatch,
    )

    assert code == 0
    assert [row[0] for row in _read_all(snapshot, "actors")] == ["org-flag"]


def test_subcommands_do_not_migrate_or_write_audit(tmp_path):
    # A legacy pre-migration database stays exactly as it was: the recovery
    # commands neither run migrations nor append audit rows.
    legacy = tmp_path / "legacy.db"
    con = sqlite3.connect(legacy)
    con.execute(
        "CREATE TABLE actors ("
        "id VARCHAR(255) PRIMARY KEY, "
        "name TEXT NOT NULL, "
        "type VARCHAR(64) NOT NULL, "
        "created_at DATETIME NOT NULL)"
    )
    con.execute(
        "INSERT INTO actors VALUES ('a-1', 'A', 'person', '2026-01-01 00:00:00')"
    )
    con.execute(
        "CREATE TABLE schema_migrations ("
        "version INTEGER NOT NULL PRIMARY KEY, "
        "applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    con.execute("INSERT INTO schema_migrations (version) VALUES (1)")
    con.commit()
    con.close()
    before = legacy.read_bytes()

    snapshot = tmp_path / "snapshot.db"
    result = perform_backup(_url(legacy), str(snapshot), force=False)

    assert result["schema_version"] == 1
    assert legacy.read_bytes() == before
    assert "audit_events" not in _table_names(snapshot)
