"""Acceptance tests for the ``backup`` and ``restore`` CLI subcommands.

The commands are exercised through ``provenance.__main__.main`` exactly as
the ``python -m provenance`` entry point dispatches them: success is a
single compact JSON line on stdout, failure a single error JSON line on
stderr with a non-zero exit and no success payload.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient

from provenance.__main__ import main
from provenance.app import create_app
from provenance.config import Settings
from provenance.migrations import MIGRATIONS
from tests.helpers import DIGEST_A, content_payload, create_actor

LATEST_VERSION = max(m.version for m in MIGRATIONS)


def _run_cli(argv):
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    return excinfo.value.code


def _read_json_lines(capsys):
    captured = capsys.readouterr()
    out_lines = [line for line in captured.out.splitlines() if line.strip()]
    err_lines = [line for line in captured.err.splitlines() if line.strip()]
    return out_lines, err_lines


def _run_ok(capsys, argv):
    code = _run_cli(argv)
    out_lines, err_lines = _read_json_lines(capsys)
    assert code == 0
    assert err_lines == []
    assert len(out_lines) == 1
    return json.loads(out_lines[0])


def _run_err(capsys, argv):
    code = _run_cli(argv)
    out_lines, err_lines = _read_json_lines(capsys)
    assert code == 1
    assert out_lines == []  # no success JSON on failure
    assert len(err_lines) == 1
    return json.loads(err_lines[0])["error"]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_db_with_data(tmp_path, name="provenance.db"):
    """Create a file-backed database through the service and add records."""
    db_path = tmp_path / name
    app = create_app(Settings(database_url=f"sqlite:///{db_path.as_posix()}"))
    with TestClient(app) as client:
        actor = create_actor(client, actor_id="org-1")
        created = client.post("/v1/contents", json=content_payload())
        assert created.status_code == 201, created.text
        content = created.json()
    return db_path, actor, content


def _table_rows(db_path, table):
    connection = sqlite3.connect(str(db_path))
    try:
        return connection.execute(
            f"SELECT * FROM {table} ORDER BY 1"
        ).fetchall()
    finally:
        connection.close()


def _schema_objects(db_path):
    connection = sqlite3.connect(str(db_path))
    try:
        return connection.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
    finally:
        connection.close()


def _ledger_versions(db_path):
    return {row[0] for row in _table_rows(db_path, "schema_migrations")}


# --- backup -----------------------------------------------------------------


def test_backup_restore_round_trip_preserves_everything(tmp_path, capsys):
    db_path, actor, content = _make_db_with_data(tmp_path)
    backup_path = tmp_path / "snapshots" / "backup.db"

    report = _run_ok(
        capsys,
        [
            "--database-url", f"sqlite:///{db_path.as_posix()}",
            "backup", "--output", str(backup_path),
        ],
    )
    assert report == {
        "operation": "backup",
        "path": str(backup_path),
        "schema_version": LATEST_VERSION,
    }

    restored_path = tmp_path / "restored.db"
    report = _run_ok(
        capsys,
        [
            "--database-url", f"sqlite:///{restored_path.as_posix()}",
            "restore", "--input", str(backup_path),
        ],
    )
    assert report == {
        "operation": "restore",
        "path": str(restored_path),
        "schema_version": LATEST_VERSION,
    }

    # Tables, indexes, triggers, and the migration ledger are identical.
    assert _schema_objects(restored_path) == _schema_objects(db_path)
    for table in ("actors", "contents", "audit_events", "schema_migrations"):
        assert _table_rows(restored_path, table) == _table_rows(db_path, table)


def test_backup_leaves_source_untouched(tmp_path, capsys):
    db_path, _, _ = _make_db_with_data(tmp_path)
    before = _sha256(db_path)
    _run_ok(
        capsys,
        [
            "--database-url", f"sqlite:///{db_path.as_posix()}",
            "backup", "--output", str(tmp_path / "out.db"),
        ],
    )
    assert _sha256(db_path) == before


def test_backup_refuses_existing_destination_without_force(tmp_path, capsys):
    db_path, _, _ = _make_db_with_data(tmp_path)
    output = tmp_path / "existing.db"
    output.write_bytes(b"keep-me")

    error = _run_err(
        capsys,
        [
            "--database-url", f"sqlite:///{db_path.as_posix()}",
            "backup", "--output", str(output),
        ],
    )
    assert error["code"] == "destination_exists"
    assert output.read_bytes() == b"keep-me"


def test_backup_force_replaces_existing_destination(tmp_path, capsys):
    db_path, _, _ = _make_db_with_data(tmp_path)
    output = tmp_path / "existing.db"
    output.write_bytes(b"stale")

    report = _run_ok(
        capsys,
        [
            "--database-url", f"sqlite:///{db_path.as_posix()}",
            "backup", "--output", str(output), "--force",
        ],
    )
    assert report["operation"] == "backup"
    assert _table_rows(output, "actors") == _table_rows(db_path, "actors")


def test_backup_zero_record_initialized_database(tmp_path, capsys):
    db_path = tmp_path / "empty.db"
    app = create_app(Settings(database_url=f"sqlite:///{db_path.as_posix()}"))
    with TestClient(app):
        pass  # startup initializes the schema without any business records

    output = tmp_path / "empty-backup.db"
    report = _run_ok(
        capsys,
        [
            "--database-url", f"sqlite:///{db_path.as_posix()}",
            "backup", "--output", str(output),
        ],
    )
    assert report["schema_version"] == LATEST_VERSION
    assert _ledger_versions(output) == set(range(1, LATEST_VERSION + 1))
    assert _table_rows(output, "actors") == []


def test_backup_preserves_old_schema_version(tmp_path, capsys):
    # A database initialized by an older build: ledger stops at version 1
    # and the later migration tables do not exist.
    db_path, _, _ = _make_db_with_data(tmp_path)
    connection = sqlite3.connect(str(db_path))
    connection.execute("DELETE FROM schema_migrations WHERE version > 1")
    connection.execute("DROP TABLE attestation_access_grant_expiries")
    connection.execute("DROP TABLE actor_trust_policy_revocations")
    connection.execute("DROP TABLE evidence_bundle_revocations")
    connection.commit()
    connection.close()

    backup_path = tmp_path / "old-backup.db"
    report = _run_ok(
        capsys,
        [
            "--database-url", f"sqlite:///{db_path.as_posix()}",
            "backup", "--output", str(backup_path),
        ],
    )
    assert report["schema_version"] == 1

    # Restore copies the old ledger verbatim: it does not apply migrations.
    restored_path = tmp_path / "old-restored.db"
    report = _run_ok(
        capsys,
        [
            "--database-url", f"sqlite:///{restored_path.as_posix()}",
            "restore", "--input", str(backup_path),
        ],
    )
    assert report["schema_version"] == 1
    assert _ledger_versions(restored_path) == {1}
    object_names = {name for _, name, _ in _schema_objects(restored_path)}
    assert "attestation_access_grant_expiries" not in object_names


def test_backup_missing_source(tmp_path, capsys):
    error = _run_err(
        capsys,
        [
            "--database-url", f"sqlite:///{(tmp_path / 'nope.db').as_posix()}",
            "backup", "--output", str(tmp_path / "out.db"),
        ],
    )
    assert error["code"] == "source_not_found"
    assert not (tmp_path / "out.db").exists()


@pytest.mark.parametrize(
    "url",
    ["sqlite:///:memory:", "sqlite://", "postgresql://localhost/db"],
)
def test_backup_rejects_non_file_or_non_sqlite_source(tmp_path, capsys, url):
    error = _run_err(
        capsys,
        ["--database-url", url, "backup", "--output", str(tmp_path / "o.db")],
    )
    assert error["code"] == "source_unsupported"
    assert not (tmp_path / "o.db").exists()


def test_backup_rejects_source_missing_ledger(tmp_path, capsys):
    # A valid SQLite file that was never initialized by the service.
    source = tmp_path / "foreign.db"
    connection = sqlite3.connect(str(source))
    connection.execute("CREATE TABLE stuff (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()

    output = tmp_path / "out.db"
    error = _run_err(
        capsys,
        [
            "--database-url", f"sqlite:///{source.as_posix()}",
            "backup", "--output", str(output),
        ],
    )
    assert error["code"] == "invalid_backup"
    assert not output.exists()


def test_backup_is_consistent_under_concurrent_writes(tmp_path, capsys):
    db_path = tmp_path / "live.db"
    app = create_app(Settings(database_url=f"sqlite:///{db_path.as_posix()}"))
    with TestClient(app):
        pass

    stop = threading.Event()

    def writer():
        connection = sqlite3.connect(str(db_path), timeout=30)
        seq = 0
        while not stop.is_set():
            seq += 1
            connection.execute(
                "INSERT INTO audit_events (event_type, resource_id, created_at)"
                " VALUES ('test_event', ?, '2026-01-01 00:00:00')",
                (f"resource-{seq}",),
            )
            connection.commit()
        connection.close()

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        output = tmp_path / "live-backup.db"
        report = _run_ok(
            capsys,
            [
                "--database-url", f"sqlite:///{db_path.as_posix()}",
                "backup", "--output", str(output),
            ],
        )
    finally:
        stop.set()
        thread.join()

    assert report["operation"] == "backup"
    # The snapshot is a transactionally consistent view: integrity passes
    # (enforced by the command) and the row count is a single point-in-time
    # count, i.e. not larger than the source's final count.
    final_count = _table_rows(db_path, "audit_events")
    snapshot_count = _table_rows(output, "audit_events")
    assert 0 <= len(snapshot_count) <= len(final_count)


# --- restore ----------------------------------------------------------------


def test_restore_missing_input(tmp_path, capsys):
    error = _run_err(
        capsys,
        [
            "--database-url", f"sqlite:///{(tmp_path / 'db.db').as_posix()}",
            "restore", "--input", str(tmp_path / "missing.db"),
        ],
    )
    assert error["code"] == "input_not_found"
    assert not (tmp_path / "db.db").exists()


def test_restore_rejects_corrupted_input_and_keeps_target(tmp_path, capsys):
    target = tmp_path / "target.db"
    target.write_bytes(b"original-target")
    corrupted = tmp_path / "corrupted.db"
    corrupted.write_bytes(b"this is not a sqlite database" * 64)

    error = _run_err(
        capsys,
        [
            "--database-url", f"sqlite:///{target.as_posix()}",
            "restore", "--input", str(corrupted), "--force",
        ],
    )
    assert error["code"] == "invalid_backup"
    # The existing target is untouched and no staging file is left behind.
    assert target.read_bytes() == b"original-target"
    assert [p.name for p in tmp_path.iterdir()] == ["target.db", "corrupted.db"] or (
        not list(tmp_path.glob("*.tmp"))
    )


def test_restore_rejects_input_missing_ledger(tmp_path, capsys):
    bad = tmp_path / "bad.db"
    connection = sqlite3.connect(str(bad))
    connection.execute("CREATE TABLE stuff (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()

    target = tmp_path / "target.db"
    error = _run_err(
        capsys,
        [
            "--database-url", f"sqlite:///{target.as_posix()}",
            "restore", "--input", str(bad),
        ],
    )
    assert error["code"] == "invalid_backup"
    assert not target.exists()


def test_restore_refuses_existing_target_without_force(tmp_path, capsys):
    db_path, _, _ = _make_db_with_data(tmp_path)
    backup_path = tmp_path / "backup.db"
    _run_ok(
        capsys,
        [
            "--database-url", f"sqlite:///{db_path.as_posix()}",
            "backup", "--output", str(backup_path),
        ],
    )

    db_path.write_bytes(b"keep-me")
    error = _run_err(
        capsys,
        [
            "--database-url", f"sqlite:///{db_path.as_posix()}",
            "restore", "--input", str(backup_path),
        ],
    )
    assert error["code"] == "destination_exists"
    assert db_path.read_bytes() == b"keep-me"


def test_restore_rejects_non_file_input(tmp_path, capsys):
    error = _run_err(
        capsys,
        [
            "--database-url", f"sqlite:///{(tmp_path / 'db.db').as_posix()}",
            "restore", "--input", str(tmp_path),
        ],
    )
    assert error["code"] == "source_unsupported"


@pytest.mark.parametrize(
    "url",
    ["sqlite:///:memory:", "postgresql://localhost/db"],
)
def test_restore_rejects_non_file_target_url(tmp_path, capsys, url):
    db_path, _, _ = _make_db_with_data(tmp_path)
    error = _run_err(
        capsys,
        ["--database-url", url, "restore", "--input", str(db_path)],
    )
    assert error["code"] == "source_unsupported"


# --- URL resolution and service behaviour -----------------------------------


def test_env_var_and_flag_priority(tmp_path, capsys, monkeypatch):
    env_db, _, _ = _make_db_with_data(tmp_path, "env.db")
    flag_db, _, _ = _make_db_with_data(tmp_path, "flag.db")
    monkeypatch.setenv(
        "PROVENANCE_DATABASE_URL", f"sqlite:///{env_db.as_posix()}"
    )

    # No flag: the environment variable supplies the source.
    report = _run_ok(
        capsys, ["backup", "--output", str(tmp_path / "from-env.db")]
    )
    assert report["operation"] == "backup"
    assert _table_rows(tmp_path / "from-env.db", "actors") == _table_rows(
        env_db, "actors"
    )

    # The flag wins over the environment variable.
    report = _run_ok(
        capsys,
        [
            "--database-url", f"sqlite:///{flag_db.as_posix()}",
            "backup", "--output", str(tmp_path / "from-flag.db"),
        ],
    )
    assert _table_rows(tmp_path / "from-flag.db", "actors") == _table_rows(
        flag_db, "actors"
    )


def test_default_database_location(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("PROVENANCE_DATABASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    db_path, _, _ = _make_db_with_data(tmp_path)  # tmp_path/provenance.db

    report = _run_ok(capsys, ["backup", "--output", "out.db"])
    assert report["path"] == str(tmp_path / "out.db")
    assert _table_rows(tmp_path / "out.db", "actors") == _table_rows(
        db_path, "actors"
    )


def test_restored_database_serves_identical_data_and_audit(tmp_path, capsys):
    db_path, actor, content = _make_db_with_data(tmp_path)
    backup_path = tmp_path / "backup.db"
    _run_ok(
        capsys,
        [
            "--database-url", f"sqlite:///{db_path.as_posix()}",
            "backup", "--output", str(backup_path),
        ],
    )
    restored_path = tmp_path / "restored.db"
    _run_ok(
        capsys,
        [
            "--database-url", f"sqlite:///{restored_path.as_posix()}",
            "restore", "--input", str(backup_path),
        ],
    )

    app = create_app(
        Settings(database_url=f"sqlite:///{restored_path.as_posix()}")
    )
    with TestClient(app) as client:
        actors = client.get("/v1/actors").json()
        assert actors["items"] == [actor]
        assert client.get(f"/v1/contents/{content['id']}").json() == content

    # The restore wrote no audit rows of its own: the restored ledger is
    # exactly the backed-up one.
    assert _table_rows(restored_path, "audit_events") == _table_rows(
        db_path, "audit_events"
    )


def test_success_output_is_a_single_compact_json_line(tmp_path, capsys):
    db_path, _, _ = _make_db_with_data(tmp_path)
    output = tmp_path / "out.db"
    code = _run_cli(
        [
            "--database-url", f"sqlite:///{db_path.as_posix()}",
            "backup", "--output", str(output),
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    line = captured.out.strip()
    assert "\n" not in line
    assert line == json.dumps(json.loads(line), separators=(",", ":"))
    assert list(json.loads(line)) == ["operation", "path", "schema_version"]
