"""Tests for the logical SQLite backup/restore commands.

Covers ``python -m provenance export-backup`` and ``... restore-backup``:
the deterministic package shape (``backup_version`` 1, ``schema_version``
2, canonical JSON, whole-resources SHA-256 digest), the full round trip of
every resource family with identifiers, UTC timestamps, stable order, and
references preserved, the single-transaction restore semantics (empty
target only, exact-same-backup no-op, full rollback on failure), and every
stable failure code (``database_unavailable``, ``backup_validation_error``,
``backup_integrity_mismatch``, ``backup_version_unsupported``,
``restore_target_not_empty``).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from provenance import backup, models
from provenance.app import create_app
from provenance.config import Settings
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    SEED_A,
    SEED_B,
    actor_payload,
    content_payload,
    create_actor,
    ed25519_public_key,
    ed25519_sign,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

TS = datetime(2026, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc)
TS_LATER = datetime(2026, 6, 7, 8, 9, 10, 654321, tzinfo=timezone.utc)


def _fixed_id(prefix: str, fill: str) -> str:
    return prefix + (fill * 64)[:64]


def _digest(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


# --- Population helpers -----------------------------------------------------


def _claim_payload(content_id: str, actor_id: str, statement: str) -> dict:
    return {
        "content_id": content_id,
        "actor_id": actor_id,
        "claim_type": "authorship",
        "payload": {"statement": statement},
    }


def _attestation_payload(target_type: str, target_id: str, seed: bytes) -> dict:
    from provenance.signing import attestation_message_bytes

    import base64

    message = attestation_message_bytes(target_type, target_id, "org-1")
    return {
        "target_type": target_type,
        "target_id": target_id,
        "signer_actor_id": "org-1",
        "public_key": base64.b64encode(ed25519_public_key(seed)).decode(),
        "signature": base64.b64encode(ed25519_sign(seed, message)).decode(),
    }


def _populate_via_api(client: TestClient) -> dict:
    """Create a spread of resources through the public HTTP API."""
    create_actor(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="person")
    content_a = client.post(
        "/v1/contents", json=content_payload(actor_id="org-1", digest=DIGEST_A)
    ).json()
    content_b = client.post(
        "/v1/contents",
        json=content_payload(
            actor_id="org-1", digest=DIGEST_B, title="Second", media_type="text/plain"
        ),
    ).json()
    relation = client.post(
        "/v1/content-relations",
        json={
            "content_id": content_b["id"],
            "parent_content_id": content_a["id"],
            "relation_type": "derived_from",
        },
    )
    assert relation.status_code == 201, relation.text
    claim_1 = client.post(
        "/v1/claims", json=_claim_payload(content_a["id"], "org-1", "one")
    ).json()
    claim_2 = client.post(
        "/v1/claims", json=_claim_payload(content_a["id"], "org-1", "two")
    ).json()
    supersession = client.post(
        "/v1/claim-supersessions",
        json={
            "superseded_claim_id": claim_1["id"],
            "replacement_claim_id": claim_2["id"],
            "reason": "corrected statement",
        },
    )
    assert supersession.status_code == 201, supersession.text
    bundle = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim_1["id"],
            "evidence_type": "raw_capture",
            "digest_algorithm": "sha256",
            "digest_hex": _digest("evidence"),
            "media_type": "image/jpeg",
            "metadata": {"source": "camera-1"},
        },
    ).json()
    attestation = client.post(
        "/v1/attestations",
        json=_attestation_payload("claim", claim_1["id"], SEED_A),
    )
    assert attestation.status_code == 201, attestation.text
    job = client.post(
        "/v1/content-export-jobs",
        json={"content_id": content_a["id"], "request_id": "req-export-1"},
    )
    assert job.status_code == 201, job.text
    run = client.post(f"/v1/content-export-jobs/{job.json()['id']}/run")
    assert run.status_code == 200, run.text
    assert run.json()["status"] == "succeeded"
    checkpoint_job = client.post(
        "/v1/audit-checkpoint-jobs", json={"request_id": "req-checkpoint-1"}
    )
    assert checkpoint_job.status_code == 201, checkpoint_job.text
    return {
        "content_a": content_a,
        "content_b": content_b,
        "claim_1": claim_1,
        "claim_2": claim_2,
        "bundle": bundle,
        "attestation": attestation.json(),
    }


def _populate_via_orm(app, ids: dict) -> None:
    """Insert rows for the remaining families with fixed deterministic data."""
    session = app.state.session_factory()
    try:
        key_a = ed25519_public_key(SEED_A)
        key_b = ed25519_public_key(SEED_B)
        attestation_id = ids["attestation"]["id"]
        grant = models.AttestationAccessGrant(
            id=_fixed_id("aag_", "1"),
            attestation_id=attestation_id,
            grantee_actor_id="org-2",
            created_at=TS,
        )
        session.add(grant)
        session.flush()
        session.add_all(
            [
                models.AttestationRevocation(
                    id=_fixed_id("rev_", "2"),
                    attestation_id=attestation_id,
                    revoker_actor_id="org-1",
                    reason="key compromised",
                    created_at=TS,
                ),
                models.AttestationAccessGrantRevocation(
                    id=_fixed_id("agr_", "3"),
                    grant_id=grant.id,
                    revoker_actor_id="org-1",
                    reason="access no longer needed",
                    created_at=TS,
                ),
                models.AttestationAccessGrantExpiry(
                    id=_fixed_id("aage_", "4"),
                    grant_id=grant.id,
                    expires_at=TS_LATER,
                    created_at=TS,
                ),
                models.AuthenticationKeyRotation(
                    id=_fixed_id("akr_", "5"),
                    actor_id="org-1",
                    public_key=key_b,
                    active=True,
                    created_at=TS,
                    retired_at=None,
                ),
                models.AuthenticationKeyRotation(
                    id=_fixed_id("akr_", "6"),
                    actor_id="org-1",
                    public_key=key_a,
                    active=False,
                    created_at=TS,
                    retired_at=TS_LATER,
                ),
                models.ActorTrustPolicy(
                    id=_fixed_id("atp_", "7"),
                    actor_id="org-1",
                    threshold=2,
                    enabled=True,
                    created_at=TS,
                ),
                models.ExchangeImportRecord(
                    id=_fixed_id("eir_", "8"),
                    manifest_version="evidence-bundle-exchange/v1",
                    evidence_bundle_id=ids["bundle"]["id"],
                    manifest_digest_hex=_digest("manifest"),
                    created_at=TS,
                ),
                models.CheckpointImportRecord(
                    id=_fixed_id("aci_", "9"),
                    checkpoint_version="audit-checkpoint/v1",
                    event_count=3,
                    events_digest_hex=_digest("events"),
                    created_at=TS,
                ),
                models.CspImportRecord(
                    id=_fixed_id("csi_", "a"),
                    checkpoint_version="csp-checkpoint/v1",
                    correction_count=1,
                    corrections_digest_hex=_digest("corrections"),
                    created_at=TS,
                ),
                models.AuditExchangeImportRecord(
                    id=_fixed_id("acx_", "b"),
                    signature_version="audit-exchange/v1",
                    signer_subject="org-9",
                    public_key=key_a,
                    package_digest_algorithm="sha256",
                    package_digest_hex=_digest("audit-package"),
                    signature_digest_algorithm="sha256",
                    signature_digest_hex=_digest("audit-signature"),
                    created_at=TS,
                ),
                models.AuditReconExchangeImportRecord(
                    id=_fixed_id("arx_", "c"),
                    signature_version="audit-recon-exchange/v1",
                    signer_subject="org-9",
                    public_key=key_b,
                    package_digest_algorithm="sha256",
                    package_digest_hex=_digest("recon-package"),
                    signature_digest_algorithm="sha256",
                    signature_digest_hex=_digest("recon-signature"),
                    created_at=TS,
                ),
                models.ImpactImportRecord(
                    id=_fixed_id("rii_", "d"),
                    checkpoint_version="impact-checkpoint/v1",
                    impact_count=2,
                    impacts_digest_hex=_digest("impacts"),
                    created_at=TS,
                ),
                models.ImpactReconExchangeImportRecord(
                    id=_fixed_id("irx_", "e"),
                    signature_version="impact-recon-exchange/v1",
                    signer_subject="org-9",
                    public_key=key_a,
                    package_digest_algorithm="sha256",
                    package_digest_hex=_digest("impact-package"),
                    signature_digest_algorithm="sha256",
                    signature_digest_hex=_digest("impact-signature"),
                    created_at=TS,
                ),
            ]
        )
        session.commit()
    finally:
        session.close()


@pytest.fixture
def source(tmp_path):
    """A populated file-backed database plus its URL and created ids."""
    url = f"sqlite:///{(tmp_path / 'source.db').as_posix()}"
    app = create_app(Settings(database_url=url))
    with TestClient(app) as client:
        ids = _populate_via_api(client)
    _populate_via_orm(app, ids)
    return {"url": url, "ids": ids}


def _target_url(tmp_path, name="target.db") -> str:
    return f"sqlite:///{(tmp_path / name).as_posix()}"


# --- Export shape ------------------------------------------------------------


def test_export_empty_database_has_stable_shape(tmp_path):
    url = _target_url(tmp_path, "empty.db")
    app = create_app(Settings(database_url=url))
    with TestClient(app):
        pass
    text = backup.export_backup(url)
    assert text.endswith("\n") and not text.endswith("\n\n")
    # The package is one canonical JSON object: sorted keys, compact
    # separators, no insignificant whitespace before the final newline.
    assert text == json.dumps(json.loads(text), sort_keys=True,
                              separators=(",", ":"), ensure_ascii=False) + "\n"
    package = json.loads(text)
    assert package["backup_version"] == 1
    assert package["schema_version"] == 2
    assert set(package) == {
        "backup_version",
        "schema_version",
        "resources",
        "digest",
    }
    assert package["digest"]["algorithm"] == "sha256"
    resources = package["resources"]
    assert set(resources) == set(backup._ALL_FAMILIES)
    for family in backup.RESOURCE_FAMILIES:
        assert resources[family.name] == []
    assert [row["version"] for row in resources["schema_migrations"]] == [1, 2]
    expected = hashlib.sha256(
        backup.canonical_json_bytes(resources)
    ).hexdigest()
    assert package["digest"]["value_hex"] == expected


def test_export_missing_database_is_database_unavailable(tmp_path):
    url = _target_url(tmp_path, "absent.db")
    with pytest.raises(backup.BackupError) as excinfo:
        backup.export_backup(url)
    assert excinfo.value.code == "database_unavailable"
    # The failed export did not create a database file as a side effect.
    assert not (tmp_path / "absent.db").exists()


def test_export_unmigrated_database_is_database_unavailable(tmp_path):
    # A file that exists but holds no provenance schema cannot be read.
    path = tmp_path / "junk.db"
    path.write_bytes(b"not a sqlite database at all")
    with pytest.raises(backup.BackupError) as excinfo:
        backup.export_backup(f"sqlite:///{path.as_posix()}")
    assert excinfo.value.code == "database_unavailable"


def test_backup_contains_no_secret_material(source):
    package = json.loads(backup.export_backup(source["url"]))
    resources = package["resources"]
    # Claims commit to their payload by digest only; the raw payload and the
    # content bytes are never in the backup.
    for claim in resources["claims"]:
        assert "payload" not in claim
        assert "content" not in claim
    # Attestations and signed receipts carry public keys and signature
    # *digests* only -- never a raw signature, private key, or credential.
    for family in (
        "attestations",
        "audit_exchange_imports",
        "audit_recon_exchange_imports",
        "impact_recon_exchange_imports",
    ):
        for row in resources[family]:
            assert "signature" not in row
            assert "private_key" not in row
            assert "public_key" in row
    text = json.dumps(package)
    assert "private_key" not in text
    assert "signature_digest_hex" in text


# --- Round trip --------------------------------------------------------------


def test_roundtrip_restores_everything_byte_identical(source, tmp_path):
    original = backup.export_backup(source["url"])
    target = _target_url(tmp_path)
    backup.restore_backup(target, original)
    assert backup.export_backup(target) == original


def test_restored_database_preserves_ids_times_order_and_references(
    source, tmp_path
):
    original = json.loads(backup.export_backup(source["url"]))
    target = _target_url(tmp_path)
    backup.restore_backup(target, backup.export_backup(source["url"]))
    restored = json.loads(backup.export_backup(target))
    assert restored == original
    resources = restored["resources"]
    # Stable order is the persisted surrogate order.
    for family in backup.RESOURCE_FAMILIES:
        rows = resources[family.name]
        if family.name == "actors":
            keys = [row["display_seq"] for row in rows]
        elif rows and "seq" in rows[0]:
            keys = [row["seq"] for row in rows]
        else:
            continue
        assert keys == sorted(keys)
    # UTC timestamps survive with their exact ISO spelling.
    actor = resources["actors"][0]
    assert actor["created_at"].endswith("+00:00")
    # References still point at the restored rows.
    claim = resources["claims"][0]
    content_ids = {row["id"] for row in resources["contents"]}
    actor_ids = {row["id"] for row in resources["actors"]}
    assert claim["content_id"] in content_ids
    assert claim["actor_id"] in actor_ids
    relation = resources["content_relations"][0]
    assert relation["parent_content_id"] in content_ids
    # The migration ledger is restored, not re-baselined.
    assert [row["version"] for row in resources["schema_migrations"]] == [1, 2]


def test_restored_database_serves_the_same_api(source, tmp_path):
    backup_text = backup.export_backup(source["url"])
    target = _target_url(tmp_path)
    backup.restore_backup(target, backup_text)
    source_app = create_app(Settings(database_url=source["url"]))
    restored_app = create_app(Settings(database_url=target))
    with TestClient(source_app) as source_client, TestClient(
        restored_app
    ) as restored_client:
        for path in (
            "/v1/actors",
            "/v1/contents",
            "/v1/claims",
            "/v1/audit-events",
            "/v1/content-relations",
            "/v1/content-export-jobs",
            "/v1/audit-checkpoint-jobs",
        ):
            assert restored_client.get(path).json() == source_client.get(
                path
            ).json()
        # New writes keep working on the restored database and continue the
        # restored sequences rather than colliding with restored rows.
        created = restored_client.post("/v1/actors", json=actor_payload(
            actor_id="org-3", name="Third", type="software"
        ))
        assert created.status_code == 201, created.text
        new_content = restored_client.post(
            "/v1/contents",
            json=content_payload(actor_id="org-3", digest=_digest("new")),
        )
        assert new_content.status_code == 201, new_content.text


def test_restore_same_backup_twice_is_a_noop(source, tmp_path):
    backup_text = backup.export_backup(source["url"])
    target = _target_url(tmp_path)
    backup.restore_backup(target, backup_text)
    first = backup.export_backup(target)
    # A second restore of the identical backup succeeds without rewriting.
    backup.restore_backup(target, backup_text)
    assert backup.export_backup(target) == first == backup_text


def test_restore_into_nonempty_target_is_refused(source, tmp_path):
    backup_text = backup.export_backup(source["url"])
    target = _target_url(tmp_path)
    other = create_app(Settings(database_url=target))
    with TestClient(other) as client:
        create_actor(client, actor_id="someone-else")
    before = backup.export_backup(target)
    with pytest.raises(backup.BackupError) as excinfo:
        backup.restore_backup(target, backup_text)
    assert excinfo.value.code == "restore_target_not_empty"
    # The refused restore changed nothing.
    assert backup.export_backup(target) == before


def test_restore_failure_leaves_no_partial_state(source, tmp_path):
    package = json.loads(backup.export_backup(source["url"]))
    # Corrupt one row so a mid-transaction insert fails (duplicate primary
    # key inside the actors family), then fix the digest so the failure
    # happens during the write phase, not validation.
    package["resources"]["actors"].append(
        dict(package["resources"]["actors"][0])
    )
    package["digest"]["value_hex"] = hashlib.sha256(
        backup.canonical_json_bytes(package["resources"])
    ).hexdigest()
    raw = json.dumps(package)
    target = _target_url(tmp_path)
    with pytest.raises(backup.BackupError):
        backup.restore_backup(target, raw)
    # The target is left without any half-restored resource rows.
    app = create_app(Settings(database_url=target))
    session = app.state.session_factory()
    try:
        assert not backup._has_resource_rows(session)
    finally:
        session.close()


# --- Restore input validation -------------------------------------------------


def _valid_backup_text(source) -> str:
    return backup.export_backup(source["url"])


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   \n  ",
        "not json",
        "[1, 2]",
        '"text"',
        "null",
        '{"backup_version": 1,} ',
    ],
)
def test_restore_rejects_unparseable_input(tmp_path, raw):
    with pytest.raises(backup.BackupError) as excinfo:
        backup.restore_backup(_target_url(tmp_path), raw)
    assert excinfo.value.code == "backup_validation_error"


def _mutated(source, mutate) -> str:
    package = json.loads(_valid_backup_text(source))
    mutate(package)
    return json.dumps(package)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.pop("backup_version"),
        lambda p: p.pop("schema_version"),
        lambda p: p.pop("resources"),
        lambda p: p.pop("digest"),
        lambda p: p.update(extra_field=True),
        lambda p: p.update(backup_version="1"),
        lambda p: p.update(resources=[]),
        lambda p: p.update(digest={"algorithm": "sha256"}),
        lambda p: p["digest"].update(algorithm="md5"),
        lambda p: p["digest"].update(value_hex="zz" * 32),
        lambda p: p["resources"].pop("actors"),
        lambda p: p["resources"].update(unknown_family=[]),
        lambda p: p["resources"]["actors"].append({"id": "x"}),
        lambda p: p["resources"]["actors"][0].pop("created_at"),
        lambda p: p["resources"]["actors"][0].update(surprise=1),
        lambda p: p["resources"]["actors"][0].update(display_seq="1"),
        lambda p: p["resources"]["actors"][0].update(
            created_at="2026-01-01 00:00:00"
        ),
        lambda p: p["resources"]["attestations"][0].update(public_key="!!!"),
    ],
)
def test_restore_rejects_invalid_structure(source, tmp_path, mutate):
    raw = _mutated(source, mutate)
    with pytest.raises(backup.BackupError) as excinfo:
        backup.restore_backup(_target_url(tmp_path), raw)
    assert excinfo.value.code == "backup_validation_error"


def test_restore_rejects_duplicate_fields(source, tmp_path):
    package = json.loads(_valid_backup_text(source))
    raw = json.dumps(package)
    # Inject a duplicated top-level member into the serialized object.
    injected = raw.replace(
        '"backup_version": 1', '"backup_version": 1, "backup_version": 1', 1
    )
    with pytest.raises(backup.BackupError) as excinfo:
        backup.restore_backup(_target_url(tmp_path), injected)
    assert excinfo.value.code == "backup_validation_error"


def test_restore_rejects_integrity_mismatch(source, tmp_path):
    def mutate(package):
        package["resources"]["actors"][0]["name"] = "tampered"

    with pytest.raises(backup.BackupError) as excinfo:
        backup.restore_backup(_target_url(tmp_path), _mutated(source, mutate))
    assert excinfo.value.code == "backup_integrity_mismatch"


@pytest.mark.parametrize(
    "field,value",
    [
        ("backup_version", 2),
        ("backup_version", 0),
        ("schema_version", 1),
        ("schema_version", 3),
    ],
)
def test_restore_rejects_unsupported_versions(source, tmp_path, field, value):
    raw = _mutated(source, lambda p: p.update({field: value}))
    with pytest.raises(backup.BackupError) as excinfo:
        backup.restore_backup(_target_url(tmp_path), raw)
    assert excinfo.value.code == "backup_version_unsupported"


# --- CLI ----------------------------------------------------------------------


def _run_cli(args, *, input_text=None, env=None):
    return subprocess.run(
        [sys.executable, "-m", "provenance", *args],
        input=input_text,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )


def test_cli_export_writes_one_newline_terminated_object(source):
    result = _run_cli(
        ["export-backup", "--database-url", source["url"]]
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == backup.export_backup(source["url"])
    assert result.stdout.endswith("\n") and not result.stdout.endswith("\n\n")
    json.loads(result.stdout)


def test_cli_roundtrip_through_stdin_stdout(source, tmp_path):
    exported = _run_cli(["export-backup", "--database-url", source["url"]])
    assert exported.returncode == 0, exported.stderr
    target = _target_url(tmp_path)
    restored = _run_cli(
        ["restore-backup", "--database-url", target],
        input_text=exported.stdout,
    )
    assert restored.returncode == 0, restored.stderr
    assert restored.stdout == ""
    assert restored.stderr == ""
    assert backup.export_backup(target) == exported.stdout


def test_cli_export_missing_database_fails_without_output(tmp_path):
    result = _run_cli(
        ["export-backup", "--database-url", _target_url(tmp_path, "none.db")]
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert "database_unavailable" in result.stderr


def test_cli_restore_empty_input_fails(tmp_path):
    result = _run_cli(
        ["restore-backup", "--database-url", _target_url(tmp_path)],
        input_text="",
    )
    assert result.returncode != 0
    assert "backup_validation_error" in result.stderr


def test_cli_restore_integrity_mismatch_fails(source, tmp_path):
    package = json.loads(backup.export_backup(source["url"]))
    package["resources"]["actors"][0]["name"] = "tampered"
    result = _run_cli(
        ["restore-backup", "--database-url", _target_url(tmp_path)],
        input_text=json.dumps(package),
    )
    assert result.returncode != 0
    assert "backup_integrity_mismatch" in result.stderr


def test_cli_restore_version_unsupported_fails(source, tmp_path):
    package = json.loads(backup.export_backup(source["url"]))
    package["backup_version"] = 99
    result = _run_cli(
        ["restore-backup", "--database-url", _target_url(tmp_path)],
        input_text=json.dumps(package),
    )
    assert result.returncode != 0
    assert "backup_version_unsupported" in result.stderr


def test_cli_restore_into_nonempty_target_fails(source, tmp_path):
    target = _target_url(tmp_path)
    other = create_app(Settings(database_url=target))
    with TestClient(other) as client:
        create_actor(client, actor_id="someone-else")
    result = _run_cli(
        ["restore-backup", "--database-url", target],
        input_text=backup.export_backup(source["url"]),
    )
    assert result.returncode != 0
    assert "restore_target_not_empty" in result.stderr


def test_cli_database_url_resolution_order(source, tmp_path):
    # The environment variable is used when the flag is absent...
    env = dict(os.environ, PROVENANCE_DATABASE_URL=source["url"])
    result = _run_cli(["export-backup"], env=env)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["resources"]["actors"]
    # ...and the flag wins over the environment variable.
    empty = _target_url(tmp_path, "empty.db")
    app = create_app(Settings(database_url=empty))
    with TestClient(app):
        pass
    result = _run_cli(
        ["export-backup", "--database-url", empty], env=env
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["resources"]["actors"] == []
