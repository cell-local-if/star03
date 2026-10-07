"""Acceptance tests for the ``verify`` CLI subcommand.

``verify`` is the read-only counterpart of ``backup``/``restore``: it
validates a snapshot file in place (integrity check plus migration-ledger
structure) and reports the ledger schema version without resolving a
database URL, starting the service, or modifying any file.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from provenance.__main__ import main
from provenance.app import create_app
from provenance.config import Settings
from provenance.migrations import MIGRATIONS
from tests.helpers import content_payload, create_actor

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


def _make_db(tmp_path, name="provenance.db"):
    """Create an initialized file-backed database through the service."""
    db_path = tmp_path / name
    app = create_app(Settings(database_url=f"sqlite:///{db_path.as_posix()}"))
    with TestClient(app) as client:
        create_actor(client, actor_id="org-1")
        created = client.post("/v1/contents", json=content_payload())
        assert created.status_code == 201, created.text
    return db_path


def _make_backup(tmp_path, capsys, name="backup.db"):
    db_path = _make_db(tmp_path)
    backup_path = tmp_path / name
    _run_ok(
        capsys,
        [
            "--database-url", f"sqlite:///{db_path.as_posix()}",
            "backup", "--output", str(backup_path),
        ],
    )
    return backup_path


# --- success ----------------------------------------------------------------


def test_verify_reports_schema_version(tmp_path, capsys):
    backup_path = _make_backup(tmp_path, capsys)
    report = _run_ok(capsys, ["verify", "--input", str(backup_path)])
    assert report == {
        "operation": "verify",
        "path": str(backup_path),
        "schema_version": LATEST_VERSION,
    }


def test_verify_output_is_a_single_compact_json_line(tmp_path, capsys):
    backup_path = _make_backup(tmp_path, capsys)
    code = _run_cli(["verify", "--input", str(backup_path)])
    assert code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.endswith("\n")
    line = captured.out.strip()
    assert "\n" not in line
    assert line == json.dumps(json.loads(line), separators=(",", ":"))
    assert list(json.loads(line)) == ["operation", "path", "schema_version"]


def test_verify_is_repeatable_and_leaves_input_untouched(tmp_path, capsys):
    backup_path = _make_backup(tmp_path, capsys)
    before = _sha256(backup_path)
    first = _run_ok(capsys, ["verify", "--input", str(backup_path)])
    second = _run_ok(capsys, ["verify", "--input", str(backup_path)])
    assert first == second
    assert _sha256(backup_path) == before
    # No sidecar or temporary files appear next to the input.
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "backup.db",
        "provenance.db",
    ]


def test_verify_preserves_old_schema_version(tmp_path, capsys):
    backup_path = _make_backup(tmp_path, capsys)
    connection = sqlite3.connect(str(backup_path))
    connection.execute("DELETE FROM schema_migrations WHERE version > 1")
    connection.commit()
    connection.close()

    report = _run_ok(capsys, ["verify", "--input", str(backup_path)])
    assert report["schema_version"] == 1


def test_verify_resolves_relative_input_to_absolute_path(
    tmp_path, capsys, monkeypatch
):
    backup_path = _make_backup(tmp_path, capsys)
    monkeypatch.chdir(tmp_path)
    report = _run_ok(capsys, ["verify", "--input", "backup.db"])
    assert report["path"] == str(backup_path)


def test_verify_ignores_database_url_configuration(
    tmp_path, capsys, monkeypatch
):
    backup_path = _make_backup(tmp_path, capsys)
    # A configured database URL must not be read, resolved, or created.
    missing = tmp_path / "configured.db"
    monkeypatch.setenv(
        "PROVENANCE_DATABASE_URL", f"sqlite:///{missing.as_posix()}"
    )
    report = _run_ok(capsys, ["verify", "--input", str(backup_path)])
    assert report["operation"] == "verify"
    assert not missing.exists()


# --- failures ---------------------------------------------------------------


def test_verify_missing_input(tmp_path, capsys):
    missing = tmp_path / "missing.db"
    error = _run_err(capsys, ["verify", "--input", str(missing)])
    assert error["code"] == "input_not_found"
    assert error["details"]["path"] == str(missing)


def test_verify_rejects_directory_input(tmp_path, capsys):
    error = _run_err(capsys, ["verify", "--input", str(tmp_path)])
    assert error["code"] == "source_unsupported"
    assert error["details"]["path"] == str(tmp_path)


def test_verify_rejects_corrupted_input(tmp_path, capsys):
    corrupted = tmp_path / "corrupted.db"
    corrupted.write_bytes(b"this is not a sqlite database" * 64)
    before = _sha256(corrupted)

    error = _run_err(capsys, ["verify", "--input", str(corrupted)])
    assert error["code"] == "invalid_backup"
    assert error["details"]["reason"] in (
        "integrity_check_unreadable",
        "integrity_check_failed",
    )
    assert _sha256(corrupted) == before


def test_verify_rejects_input_missing_ledger(tmp_path, capsys):
    bad = tmp_path / "bad.db"
    connection = sqlite3.connect(str(bad))
    connection.execute("CREATE TABLE stuff (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()

    error = _run_err(capsys, ["verify", "--input", str(bad)])
    assert error["code"] == "invalid_backup"
    assert error["details"]["reason"] == "ledger_missing"


def test_verify_rejects_ledger_without_version_column(tmp_path, capsys):
    bad = tmp_path / "bad.db"
    connection = sqlite3.connect(str(bad))
    connection.execute("CREATE TABLE schema_migrations (applied TEXT)")
    connection.commit()
    connection.close()

    error = _run_err(capsys, ["verify", "--input", str(bad)])
    assert error["code"] == "invalid_backup"
    assert error["details"]["reason"] == "ledger_structure_invalid"


@pytest.mark.parametrize("version", [0, -1, "abc"])
def test_verify_rejects_ledger_without_positive_integer_version(
    tmp_path, capsys, version
):
    bad = tmp_path / "bad.db"
    connection = sqlite3.connect(str(bad))
    connection.execute("CREATE TABLE schema_migrations (version INTEGER)")
    connection.execute("INSERT INTO schema_migrations (version) VALUES (?)", (version,))
    connection.commit()
    connection.close()

    error = _run_err(capsys, ["verify", "--input", str(bad)])
    assert error["code"] == "invalid_backup"
    assert error["details"]["reason"] == "ledger_version_invalid"


def test_verify_rejects_empty_ledger(tmp_path, capsys):
    bad = tmp_path / "bad.db"
    connection = sqlite3.connect(str(bad))
    connection.execute("CREATE TABLE schema_migrations (version INTEGER)")
    connection.commit()
    connection.close()

    error = _run_err(capsys, ["verify", "--input", str(bad)])
    assert error["code"] == "invalid_backup"
    assert error["details"]["reason"] == "ledger_version_invalid"


def test_verify_failure_emits_no_traceback(tmp_path, capsys):
    corrupted = tmp_path / "corrupted.db"
    corrupted.write_bytes(b"junk" * 256)
    _run_err(capsys, ["verify", "--input", str(corrupted)])
    assert "Traceback" not in capsys.readouterr().err


def test_verify_argument_error_exits_2_without_json(tmp_path, capsys):
    code = _run_cli(["verify"])  # missing required --input
    assert code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    err_lines = [line for line in captured.err.splitlines() if line.strip()]
    assert err_lines  # argparse usage message
    for line in err_lines:
        with pytest.raises(json.JSONDecodeError):
            json.loads(line)


def test_verify_failure_leaves_other_databases_untouched(tmp_path, capsys):
    db_path = _make_db(tmp_path)
    before = _sha256(db_path)
    corrupted = tmp_path / "corrupted.db"
    corrupted.write_bytes(b"junk" * 256)

    _run_err(capsys, ["verify", "--input", str(corrupted)])
    assert _sha256(db_path) == before
