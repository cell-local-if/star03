"""End-to-end tests for the logical SQLite backup export/restore CLI.

The CLI is exercised through real subprocesses (``python -m provenance
export-backup`` / ``restore-backup``) against temporary file databases,
exactly as an operator would run them.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from provenance.app import create_app
from provenance.canonical import canonical_json_bytes
from provenance.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]

# Fixed 32-byte Ed25519 public key material for attestation fixtures.
_PUBLIC_KEY_B64 = "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE="


def _run_cli(args, *, stdin: bytes | None = None, env_extra: dict | None = None):
    env = dict(os.environ)
    env.pop("PROVENANCE_DATABASE_URL", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "provenance", *args],
        input=stdin,
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
    )


def _db_url(path: Path) -> str:
    return f"sqlite:///{path.as_posix()}"


@pytest.fixture
def populated_db(tmp_path):
    """A migrated file database holding one of several resource families."""
    db_path = tmp_path / "source.db"
    app = create_app(Settings(database_url=_db_url(db_path)))
    with TestClient(app) as client:
        assert client.post(
            "/v1/actors",
            json={"id": "org-1", "name": "Example Org", "type": "organization"},
        ).status_code == 201
        assert client.post(
            "/v1/actors",
            json={"id": "org-2", "name": "Other Org", "type": "organization"},
        ).status_code == 201
        digest = hashlib.sha256(b"backup-content").hexdigest()
        content = client.post(
            "/v1/contents",
            json={
                "digest_algorithm": "sha256",
                "digest_hex": digest,
                "media_type": "image/png",
                "title": "Backup subject",
                "actor_id": "org-1",
            },
        )
        assert content.status_code == 201, content.text
        content_id = content.json()["id"]
        parent = client.post(
            "/v1/contents",
            json={
                "digest_algorithm": "sha256",
                "digest_hex": hashlib.sha256(b"backup-parent").hexdigest(),
                "media_type": "image/png",
                "title": None,
                "actor_id": "org-2",
            },
        )
        assert parent.status_code == 201, parent.text
        parent_id = parent.json()["id"]
        relation = client.post(
            "/v1/content-relations",
            json={
                "content_id": content_id,
                "parent_content_id": parent_id,
                "relation_type": "version_of",
            },
        )
        assert relation.status_code == 201, relation.text
        claim = client.post(
            "/v1/claims",
            json={
                "content_id": content_id,
                "actor_id": "org-1",
                "claim_type": "ownership",
                "payload": {"statement": "owned"},
            },
        )
        assert claim.status_code == 201, claim.text
        claim_id = claim.json()["id"]
        bundle = client.post(
            "/v1/evidence-bundles",
            json={
                "claim_id": claim_id,
                "evidence_type": "receipt",
                "digest_algorithm": "sha256",
                "digest_hex": hashlib.sha256(b"evidence").hexdigest(),
                "media_type": "application/json",
                "metadata": {"origin": "test"},
            },
        )
        assert bundle.status_code == 201, bundle.text
    return db_path


def _export(db_path: Path):
    result = _run_cli(["export-backup", "--database-url", _db_url(db_path)])
    assert result.returncode == 0, result.stderr
    assert result.stderr == b""
    return result


def _restore(db_path: Path, payload: bytes):
    return _run_cli(
        ["restore-backup", "--database-url", _db_url(db_path)], stdin=payload
    )


def _load_single_object(stdout: bytes):
    assert stdout.endswith(b"\n")
    assert not stdout.endswith(b"\n\n")
    return json.loads(stdout)


def test_export_envelope_and_digest(populated_db):
    package = _load_single_object(_export(populated_db).stdout)
    assert package["backup_version"] == 1
    assert package["schema_version"] == 2
    assert package["digest_algorithm"] == "sha256"
    expected = hashlib.sha256(
        canonical_json_bytes(package["resources"])
    ).hexdigest()
    assert package["digest_hex"] == expected
    assert set(package) == {
        "backup_version",
        "schema_version",
        "digest_algorithm",
        "digest_hex",
        "resources",
    }


def test_export_covers_every_family(populated_db):
    package = _load_single_object(_export(populated_db).stdout)
    resources = package["resources"]
    expected_families = {
        "actors",
        "contents",
        "claims",
        "claim_supersessions",
        "evidence_bundles",
        "attestations",
        "attestation_revocations",
        "attestation_access_grants",
        "attestation_access_grant_revocations",
        "attestation_access_grant_expiries",
        "authentication_key_rotations",
        "actor_trust_policies",
        "content_relations",
        "audit_events",
        "evidence_bundle_exchange_imports",
        "audit_checkpoint_imports",
        "csp_checkpoint_imports",
        "audit_exchange_imports",
        "audit_recon_exchange_imports",
        "revocation_impact_imports",
        "impact_recon_exchange_imports",
        "content_export_jobs",
        "audit_checkpoint_jobs",
        "schema_migrations",
    }
    assert set(resources) == expected_families
    assert [row["id"] for row in resources["actors"]] == ["org-1", "org-2"]
    assert resources["actors"][0]["display_seq"] == 1
    assert resources["contents"][0]["title"] == "Backup subject"
    assert resources["claims"][0]["claim_type"] == "ownership"
    assert resources["evidence_bundles"][0]["metadata"] == {"origin": "test"}
    assert [row["version"] for row in resources["schema_migrations"]] == [1, 2]
    # The audit trail recorded the creations, in stable order.
    event_types = [row["event_type"] for row in resources["audit_events"]]
    assert event_types[0] == "actor.created"
    assert "claim.created" in event_types


def test_export_never_carries_raw_material(populated_db):
    package = _load_single_object(_export(populated_db).stdout)
    blob = json.dumps(package)
    for forbidden in (
        "private_key",
        "signature\"",
        "payload\"",
        "content_bytes",
        "evidence_bytes",
        "credentials",
    ):
        assert forbidden not in blob
    claim = package["resources"]["claims"][0]
    assert set(claim) == {
        "seq",
        "id",
        "content_id",
        "actor_id",
        "claim_type",
        "payload_digest_algorithm",
        "payload_digest_hex",
        "created_at",
    }


def test_export_is_deterministic(populated_db):
    first = _export(populated_db).stdout
    second = _export(populated_db).stdout
    assert first == second


def test_round_trip_restores_identical_state(populated_db, tmp_path):
    backup = _export(populated_db).stdout
    restored_db = tmp_path / "restored.db"
    result = _restore(restored_db, backup)
    assert result.returncode == 0, result.stderr
    assert result.stderr == b""
    # A re-export of the restored database is byte-identical.
    assert _export(restored_db).stdout == backup


def test_restored_database_serves_identical_api(populated_db, tmp_path):
    backup = _export(populated_db).stdout
    restored_db = tmp_path / "restored.db"
    assert _restore(restored_db, backup).returncode == 0

    source_app = create_app(Settings(database_url=_db_url(populated_db)))
    restored_app = create_app(Settings(database_url=_db_url(restored_db)))
    with TestClient(source_app) as source, TestClient(restored_app) as restored:
        for path in (
            "/v1/actors",
            "/v1/contents",
            "/v1/claims",
            "/v1/evidence-bundles",
            "/v1/audit-events",
        ):
            source_response = source.get(path)
            restored_response = restored.get(path)
            assert source_response.status_code == 200
            assert restored_response.status_code == 200
            assert restored_response.json() == source_response.json()


def test_restore_into_identical_target_is_a_noop(populated_db):
    backup = _export(populated_db).stdout
    # The source database already contains exactly this backup.
    result = _restore(populated_db, backup)
    assert result.returncode == 0, result.stderr
    # And a restored database accepts the same backup again without rewriting.
    assert _export(populated_db).stdout == backup


def test_restore_rejects_non_empty_target(populated_db, tmp_path):
    backup = _export(populated_db).stdout
    other_db = tmp_path / "other.db"
    other_app = create_app(Settings(database_url=_db_url(other_db)))
    with TestClient(other_app) as client:
        assert client.post(
            "/v1/actors",
            json={"id": "someone-else", "name": "Other", "type": "person"},
        ).status_code == 201
    result = _restore(other_db, backup)
    assert result.returncode != 0
    assert b"restore_target_not_empty" in result.stderr


def test_restore_rejects_empty_input(tmp_path):
    target = tmp_path / "target.db"
    for payload in (b"", b"  \n\t "):
        result = _restore(target, payload)
        assert result.returncode != 0
        assert b"backup_validation_error" in result.stderr


def test_restore_rejects_invalid_json(tmp_path):
    target = tmp_path / "target.db"
    for payload in (b"{not json", b"[1, 2, 3]", b'"text"', b"null"):
        result = _restore(target, payload)
        assert result.returncode != 0
        assert b"backup_validation_error" in result.stderr


def test_restore_rejects_missing_extra_and_duplicate_fields(tmp_path):
    # Build a valid backup first.
    source = tmp_path / "source.db"
    create_app(Settings(database_url=_db_url(source)))
    package = json.loads(_export(source).stdout)

    target = tmp_path / "target.db"

    missing = dict(package)
    del missing["resources"]
    result = _restore(target, json.dumps(missing).encode())
    assert result.returncode != 0
    assert b"backup_validation_error" in result.stderr

    extra = dict(package)
    extra["unexpected"] = True
    result = _restore(target, json.dumps(extra).encode())
    assert result.returncode != 0
    assert b"backup_validation_error" in result.stderr

    raw = _export(source).stdout.decode("utf-8")
    duplicated = raw.replace(
        '"backup_version":1', '"backup_version":1,"backup_version":1', 1
    )
    result = _restore(target, duplicated.encode("utf-8"))
    assert result.returncode != 0
    assert b"backup_validation_error" in result.stderr


def test_restore_rejects_unsupported_versions(tmp_path):
    source = tmp_path / "source.db"
    create_app(Settings(database_url=_db_url(source)))
    package = json.loads(_export(source).stdout)
    target = tmp_path / "target.db"
    for field, value in (
        ("backup_version", 2),
        ("backup_version", 0),
        ("schema_version", 1),
        ("schema_version", 3),
    ):
        mutated = json.loads(json.dumps(package))
        mutated[field] = value
        result = _restore(target, json.dumps(mutated).encode())
        assert result.returncode != 0
        assert b"backup_version_unsupported" in result.stderr


def test_restore_rejects_integrity_mismatch(tmp_path):
    source = tmp_path / "source.db"
    app = create_app(Settings(database_url=_db_url(source)))
    with TestClient(app) as client:
        assert client.post(
            "/v1/actors",
            json={"id": "org-1", "name": "Example Org", "type": "organization"},
        ).status_code == 201
    package = json.loads(_export(source).stdout)
    target = tmp_path / "target.db"

    # A tampered resource no longer matches the carried digest.
    tampered = json.loads(json.dumps(package))
    tampered["resources"]["actors"][0]["name"] = "Forged Org"
    result = _restore(target, json.dumps(tampered).encode())
    assert result.returncode != 0
    assert b"backup_integrity_mismatch" in result.stderr

    # A replaced digest does not match the resources either.
    forged = json.loads(json.dumps(package))
    forged["digest_hex"] = "0" * 64
    result = _restore(target, json.dumps(forged).encode())
    assert result.returncode != 0
    assert b"backup_integrity_mismatch" in result.stderr


def test_failed_restore_leaves_target_untouched(tmp_path):
    source = tmp_path / "source.db"
    app = create_app(Settings(database_url=_db_url(source)))
    with TestClient(app) as client:
        assert client.post(
            "/v1/actors",
            json={"id": "org-1", "name": "Example Org", "type": "organization"},
        ).status_code == 201
    package = json.loads(_export(source).stdout)

    # Corrupt one family so the transactional write must fail: an exact
    # duplicate row violates the actors primary key on insert.
    broken = json.loads(json.dumps(package))
    broken["resources"]["actors"].append(
        dict(broken["resources"]["actors"][0])
    )
    broken["digest_hex"] = hashlib.sha256(
        canonical_json_bytes(broken["resources"])
    ).hexdigest()

    target = tmp_path / "target.db"
    result = _restore(target, json.dumps(broken).encode())
    assert result.returncode != 0

    # The target holds no half-restored rows: a valid backup still restores
    # cleanly into it afterwards.
    good = _export(source).stdout
    assert _restore(target, good).returncode == 0
    assert _export(target).stdout == good


def test_export_reports_database_unavailable(tmp_path):
    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"this is not a sqlite database")
    result = _run_cli(["export-backup", "--database-url", _db_url(corrupt)])
    assert result.returncode != 0
    assert b"database_unavailable" in result.stderr
    assert result.stdout == b""

    missing = tmp_path / "missing.db"
    result = _run_cli(["export-backup", "--database-url", _db_url(missing)])
    assert result.returncode != 0
    assert b"database_unavailable" in result.stderr
    assert result.stdout == b""


def test_database_url_resolution_order(tmp_path):
    # The environment variable is used when no flag is given.
    env_db = tmp_path / "env.db"
    create_app(Settings(database_url=_db_url(env_db)))
    result = _run_cli(
        ["export-backup"],
        env_extra={"PROVENANCE_DATABASE_URL": _db_url(env_db)},
    )
    assert result.returncode == 0, result.stderr
    package = json.loads(result.stdout)
    assert package["backup_version"] == 1

    # The flag wins over the environment variable.
    flag_db = tmp_path / "flag.db"
    app = create_app(Settings(database_url=_db_url(flag_db)))
    with TestClient(app) as client:
        assert client.post(
            "/v1/actors",
            json={"id": "flag-actor", "name": "Flag", "type": "person"},
        ).status_code == 201
    result = _run_cli(
        ["export-backup", "--database-url", _db_url(flag_db)],
        env_extra={"PROVENANCE_DATABASE_URL": _db_url(env_db)},
    )
    assert result.returncode == 0, result.stderr
    package = json.loads(result.stdout)
    assert [row["id"] for row in package["resources"]["actors"]] == [
        "flag-actor"
    ]

    # The flag also works when placed before the subcommand.
    result = _run_cli(
        ["--database-url", _db_url(flag_db), "export-backup"],
        env_extra={"PROVENANCE_DATABASE_URL": _db_url(env_db)},
    )
    assert result.returncode == 0, result.stderr
    package = json.loads(result.stdout)
    assert [row["id"] for row in package["resources"]["actors"]] == [
        "flag-actor"
    ]


def test_server_entrypoint_unchanged():
    # The default (serve) command line keeps its existing flags.
    result = _run_cli(["--help"])
    assert result.returncode == 0
    assert b"--host" in result.stdout
    assert b"--port" in result.stdout
    assert b"--database-url" in result.stdout
