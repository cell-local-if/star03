"""Acceptance tests for the read-only ``verify`` CLI subcommand.

``verify`` opens ``--input`` with SQLite read-only mode and applies the
same ``PRAGMA integrity_check`` plus ``schema_migrations`` ledger checks
that ``backup``/``restore`` run internally, without consulting any
database URL, starting the service, or writing anything. Success is one
compact JSON line on stdout -- exactly
``{"operation":"verify","path":...,"schema_version":N}`` -- and failure
is one ``{"error":...}`` line on stderr with exit code 1 and no stdout;
argument errors exit 2 with argparse's own (non-JSON) message.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat

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
    return json.loads(out_lines[0]), out_lines[0]


def _run_err(capsys, argv):
    code = _run_cli(argv)
    out_lines, err_lines = _read_json_lines(capsys)
    assert code == 1
    assert out_lines == []  # no success JSON on failure
    assert len(err_lines) == 1
    return json.loads(err_lines[0])["error"]


def _fails_json(line):
    try:
        json.loads(line)
    except ValueError:
        return True
    return False


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _service_db(tmp_path, name="provenance.db"):
    """A service-initialized database carrying one actor and one content."""
    db_path = tmp_path / name
    app = create_app(Settings(database_url=f"sqlite:///{db_path.as_posix()}"))
    with TestClient(app) as client:
        actor = create_actor(client, actor_id="org-1")
        created = client.post("/v1/contents", json=content_payload())
        assert created.status_code == 201, created.text
        content = created.json()
    return db_path, actor, content


def _ledger_db(path, versions, *, extra_table=True):
    """Create a plain SQLite file with only a schema_migrations ledger."""
    connection = sqlite3.connect(str(path))
    connection.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY)")
    connection.executemany(
        "INSERT INTO schema_migrations VALUES (?)", [(v,) for v in versions]
    )
    if extra_table:
        connection.execute("CREATE TABLE filler (id INTEGER PRIMARY KEY, x TEXT)")
        connection.executemany(
            "INSERT INTO filler (x) VALUES (?)", [("y" * 300,)] * 40
        )
    connection.commit()
    connection.close()
    return path


def _open_ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)


# --- success ----------------------------------------------------------------


def test_verify_backup_snapshot_reports_latest_version(tmp_path, capsys):
    db_path, _, _ = _service_db(tmp_path)
    backup_path = tmp_path / "backup.db"
    _run_ok(
        capsys,
        [
            "--database-url", f"sqlite:///{db_path.as_posix()}",
            "backup", "--output", str(backup_path),
        ],
    )

    report, raw = _run_ok(capsys, ["verify", "--input", str(backup_path)])
    assert report == {
        "operation": "verify",
        "path": str(backup_path),
        "schema_version": LATEST_VERSION,
    }
    # Exactly the documented compact line with a trailing newline.
    assert raw == (
        f'{{"operation":"verify","path":"{backup_path}","schema_version":{LATEST_VERSION}}}'
    )


def test_verify_reports_maximum_positive_ledger_version(tmp_path, capsys):
    snapshot = _ledger_db(tmp_path / "s.db", [1, 2, 3])
    report, _ = _run_ok(capsys, ["verify", "--input", str(snapshot)])
    assert report == {
        "operation": "verify",
        "path": str(snapshot),
        "schema_version": 3,
    }


def test_verify_reports_old_ledger_version_without_migrating(tmp_path, capsys):
    # A ledger stuck at version 1 verifies at 1 -- no schema changes applied.
    snapshot = _ledger_db(tmp_path / "old.db", [1])
    report, _ = _run_ok(capsys, ["verify", "--input", str(snapshot)])
    assert report["schema_version"] == 1


def test_verify_relative_input_path_reports_absolute_path(tmp_path, capsys, monkeypatch):
    snapshot = _ledger_db(tmp_path / "s.db", [1, 2])
    monkeypatch.chdir(tmp_path)
    report, _ = _run_ok(capsys, ["verify", "--input", "s.db"])
    assert report["path"] == os.path.abspath("s.db")
    assert os.path.isabs(report["path"])


def test_verify_success_stdout_is_exactly_one_json_line_plus_newline(tmp_path, capsys):
    snapshot = _ledger_db(tmp_path / "s.db", [1])
    code = _run_cli(["verify", "--input", str(snapshot)])
    assert code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.endswith("\n")
    assert captured.out.count("\n") == 1
    payload = json.loads(captured.out)
    assert list(payload) == ["operation", "path", "schema_version"]
    assert captured.out.strip() == json.dumps(payload, separators=(",", ":"))


def test_verify_is_repeatable(tmp_path, capsys):
    snapshot = _ledger_db(tmp_path / "s.db", [1, 2, 3])
    before = _sha256(snapshot)
    first, raw1 = _run_ok(capsys, ["verify", "--input", str(snapshot)])
    second, raw2 = _run_ok(capsys, ["verify", "--input", str(snapshot)])
    assert first == second
    assert raw1 == raw2
    assert _sha256(snapshot) == before


# --- read-only behaviour ----------------------------------------------------


def test_verify_never_modifies_input_or_creates_sidecars(tmp_path, capsys):
    snapshot = _ledger_db(tmp_path / "s.db", [1, 2])
    before = _sha256(snapshot)
    siblings_before = sorted(p.name for p in tmp_path.iterdir())

    _run_ok(capsys, ["verify", "--input", str(snapshot)])

    assert _sha256(snapshot) == before
    # No -journal, -wal, -shm, or any other file appears next to the input.
    assert sorted(p.name for p in tmp_path.iterdir()) == siblings_before


def test_verify_failure_also_leaves_input_untouched(tmp_path, capsys):
    corrupted = tmp_path / "corrupted.db"
    corrupted.write_bytes(b"this is not a sqlite database" * 64)
    before = _sha256(corrupted)
    siblings_before = sorted(p.name for p in tmp_path.iterdir())

    error = _run_err(capsys, ["verify", "--input", str(corrupted)])

    assert error["code"] == "invalid_backup"
    assert _sha256(corrupted) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == siblings_before


def test_verify_ignores_database_url_flag_and_environment(tmp_path, capsys, monkeypatch):
    snapshot = _ledger_db(tmp_path / "s.db", [1, 2])
    # A bogus URL on the global flag and in the environment must not be read.
    monkeypatch.setenv("PROVENANCE_DATABASE_URL", "postgresql://does-not-exist/db")
    report, _ = _run_ok(
        capsys,
        [
            "--database-url", "postgresql://also-does-not-exist/db",
            "verify", "--input", str(snapshot),
        ],
    )
    assert report["operation"] == "verify"
    assert report["path"] == str(snapshot)


def test_verify_does_not_accept_database_url_option(tmp_path, capsys):
    # verify takes no --database-url of its own; argparse rejects it (exit 2,
    # no JSON) so the URL-resolution surface cannot widen by accident.
    snapshot = _ledger_db(tmp_path / "s.db", [1])
    with pytest.raises(SystemExit) as excinfo:
        main(["verify", "--input", str(snapshot), "--database-url", "sqlite:///x"])
    assert excinfo.value.code == 2
    out_lines, err_lines = _read_json_lines(capsys)
    assert out_lines == []
    # argparse's own usage/error text, never a JSON error document.
    assert all(_fails_json(line) for line in err_lines)


# --- input resolution failures ---------------------------------------------


def test_verify_missing_input(tmp_path, capsys):
    missing = tmp_path / "missing.db"
    error = _run_err(capsys, ["verify", "--input", str(missing)])
    assert error["code"] == "input_not_found"
    assert error["details"]["path"] == str(missing)


def test_verify_missing_relative_input_path_is_reported_absolute(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    error = _run_err(capsys, ["verify", "--input", "nope.db"])
    assert error["code"] == "input_not_found"
    assert error["details"]["path"] == os.path.abspath("nope.db")


def test_verify_rejects_directory(tmp_path, capsys):
    error = _run_err(capsys, ["verify", "--input", str(tmp_path)])
    assert error["code"] == "source_unsupported"
    assert error["details"]["path"] == str(tmp_path)


@pytest.mark.skipif(
    os.geteuid() == 0, reason="root bypasses file permissions"
)
def test_verify_unreadable_file_is_operation_failed(tmp_path, capsys):
    snapshot = _ledger_db(tmp_path / "locked.db", [1])
    snapshot.chmod(0)
    try:
        error = _run_err(capsys, ["verify", "--input", str(snapshot)])
    finally:
        snapshot.chmod(stat.S_IRUSR | stat.S_IWUSR)
    assert error["code"] == "operation_failed"
    assert error["details"]["reason"]  # exception text is preserved


# --- invalid backup failures ------------------------------------------------


def _integrity_failed_snapshot(path):
    """Build a DB whose PRAGMA integrity_check returns error rows.

    A flipped cell-pointer on a ledger b-tree page makes SQLite report the
    mismatch as result rows (rather than raising), exercising the
    ``rows != [("ok",)]`` branch. The exact sensitive byte varies between
    SQLite builds, so candidate (page, offset) pairs are probed and the
    first one that yields error rows is used.
    """
    _ledger_db(path, [1, 2])
    page_count = path.stat().st_size // 4096
    candidates = [
        (page, offset)
        for page in range(1, page_count)
        for offset in (8, 100, 1000, 2000, 4000)
    ]
    original = path.read_bytes()
    for page, offset in candidates:
        path.write_bytes(original)
        with open(path, "r+b") as handle:
            handle.seek(page * 4096 + offset)
            handle.write(b"\xff" * 30)
        connection = _open_ro(path)
        try:
            try:
                rows = connection.execute("PRAGMA integrity_check").fetchall()
            except sqlite3.DatabaseError:
                continue  # this build raises here -> integrity_check_unreadable
        finally:
            connection.close()
        if rows != [("ok",)]:
            return path
    pytest.fail("could not construct an integrity_check-rows corruption fixture")


def test_verify_non_sqlite_file_is_integrity_check_unreadable(tmp_path, capsys):
    bogus = tmp_path / "bogus.db"
    bogus.write_bytes(b"this is not a sqlite database" * 64)
    error = _run_err(capsys, ["verify", "--input", str(bogus)])
    assert error["code"] == "invalid_backup"
    assert error["details"]["reason"] == "integrity_check_unreadable"


def test_verify_integrity_check_returning_errors_is_invalid_backup(tmp_path, capsys):
    snapshot = _integrity_failed_snapshot(tmp_path / "bad.db")
    error = _run_err(capsys, ["verify", "--input", str(snapshot)])
    assert error["code"] == "invalid_backup"
    assert error["details"]["reason"] == "integrity_check_failed"


def test_verify_database_missing_ledger(tmp_path, capsys):
    foreign = tmp_path / "foreign.db"
    connection = sqlite3.connect(str(foreign))
    connection.execute("CREATE TABLE stuff (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()
    error = _run_err(capsys, ["verify", "--input", str(foreign)])
    assert error["code"] == "invalid_backup"
    assert error["details"]["reason"] == "ledger_missing"


def test_verify_ledger_without_version_column_is_structure_invalid(tmp_path, capsys):
    snapshot = tmp_path / "wrong.db"
    connection = sqlite3.connect(str(snapshot))
    connection.execute("CREATE TABLE schema_migrations (revision INTEGER)")
    connection.execute("INSERT INTO schema_migrations VALUES (1)")
    connection.commit()
    connection.close()
    error = _run_err(capsys, ["verify", "--input", str(snapshot)])
    assert error["code"] == "invalid_backup"
    assert error["details"]["reason"] == "ledger_structure_invalid"


def test_verify_empty_ledger_is_version_invalid(tmp_path, capsys):
    snapshot = _ledger_db(tmp_path / "empty.db", [])
    error = _run_err(capsys, ["verify", "--input", str(snapshot)])
    assert error["code"] == "invalid_backup"
    assert error["details"]["reason"] == "ledger_version_invalid"


@pytest.mark.parametrize("bad_version", [0, -1, -99])
def test_verify_ledger_without_positive_version_is_invalid(tmp_path, capsys, bad_version):
    snapshot = tmp_path / f"v{bad_version}.db"
    connection = sqlite3.connect(str(snapshot))
    # TEXT affinity still passes the "version column exists" structure check
    # but lets non-integer values through to the version validation.
    connection.execute("CREATE TABLE schema_migrations (version TEXT)")
    connection.execute(
        "INSERT INTO schema_migrations VALUES (?)", (str(bad_version),)
    )
    connection.commit()
    connection.close()
    error = _run_err(capsys, ["verify", "--input", str(snapshot)])
    assert error["code"] == "invalid_backup"
    assert error["details"]["reason"] == "ledger_version_invalid"


def test_verify_ledger_with_non_integer_version_is_invalid(tmp_path, capsys):
    snapshot = tmp_path / "text.db"
    connection = sqlite3.connect(str(snapshot))
    connection.execute("CREATE TABLE schema_migrations (version TEXT)")
    connection.executemany(
        "INSERT INTO schema_migrations VALUES (?)", [("1",), ("oops",)]
    )
    connection.commit()
    connection.close()
    error = _run_err(capsys, ["verify", "--input", str(snapshot)])
    assert error["code"] == "invalid_backup"
    assert error["details"]["reason"] == "ledger_version_invalid"


def test_verify_failure_outputs_no_traceback(tmp_path, capsys):
    bogus = tmp_path / "bogus.db"
    bogus.write_bytes(b"definitely not a database" * 64)
    code = _run_cli(["verify", "--input", str(bogus)])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "Traceback" not in captured.err
    error = json.loads(captured.err.strip())["error"]
    assert error["code"] == "invalid_backup"


# --- argument errors --------------------------------------------------------


def test_verify_without_input_exits_2_without_json(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["verify"])
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert all(_fails_json(line) for line in captured.err.splitlines() if line.strip())


def test_unknown_command_exits_2(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["frobnicate", "--input", "x"])
    assert excinfo.value.code == 2


def test_verify_extra_unexpected_argument_exits_2(tmp_path, capsys):
    snapshot = _ledger_db(tmp_path / "s.db", [1])
    with pytest.raises(SystemExit) as excinfo:
        main(["verify", "--input", str(snapshot), "--force"])
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert all(_fails_json(line) for line in captured.err.splitlines() if line.strip())
