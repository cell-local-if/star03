"""Tests for the versioned SQLite migration layer.

Covers startup migration from zero:

* a brand-new database is created at the latest schema, baselined in the
  ``schema_migrations`` ledger, and continues the explicit actor sequence via
  an insert trigger;
* a legacy database whose ``actors`` table predates the ledger gains
  ``display_seq`` backfilled from the original insertion order (stable even
  for same-created_at rows), the supporting index and trigger, and a single
  version-1 ledger row -- without changing any existing actor field;
* a legacy database that already shipped versions 1-4 gains every remaining
  current table and index as version 5, ends up structurally identical to a
  fresh database, keeps every pre-existing row, and immediately serves the
  current features (key rotations, trust policies, relations, export jobs,
  import receipts) through the public API;
* startup is idempotent: restarting never re-runs an applied version and
  ordering is identical;
* a failing migration rolls the whole batch back (no ledger, no new column,
  original rows intact) and a later successful startup migrates cleanly;
* the read-only actor list never writes a migration record.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from provenance import migrations
from provenance.app import create_app
from provenance.access_signing import access_message_bytes
from provenance.config import Settings
from provenance.database import Base, make_engine
from provenance.signing import attestation_message_bytes
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    SEED_A,
    content_payload,
    create_actor,
    ed25519_public_key,
    ed25519_sign,
)

ACTORS_PATH = "/v1/actors"
TIE = "2026-03-01 00:00:00.000000"
LEGACY_ACTORS = [
    ("org-1", "Example Org", "organization"),
    ("p-1", "Alice", "person"),
    ("org-2", "Other Org", "organization"),
    ("dev-1", "Sensor", "device"),
]


def _connect(path):
    return sqlite3.connect(path)


def _create_legacy_database(path) -> None:
    """Create a pre-migration actors table with same-timestamp rows."""
    con = _connect(path)
    try:
        con.execute(
            "CREATE TABLE actors ("
            "id VARCHAR(255) PRIMARY KEY, "
            "name TEXT NOT NULL, "
            "type VARCHAR(64) NOT NULL, "
            "created_at DATETIME NOT NULL)"
        )
        for actor_id, name, actor_type in LEGACY_ACTORS:
            con.execute(
                "INSERT INTO actors (id, name, type, created_at) "
                "VALUES (?, ?, ?, ?)",
                (actor_id, name, actor_type, TIE),
            )
        con.commit()
    finally:
        con.close()


def _table_names(con):
    return {
        row[0]
        for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _columns(con, table):
    return [row[1] for row in con.execute(f"PRAGMA table_info({table})")]


# --- Fresh database -----------------------------------------------------------


def test_fresh_database_is_baselined_with_latest_schema(tmp_path):
    db_path = tmp_path / "fresh.db"
    url = f"sqlite:///{db_path.as_posix()}"
    with TestClient(create_app(Settings(database_url=url))):
        pass

    con = _connect(db_path)
    try:
        tables = _table_names(con)
        assert "schema_migrations" in tables
        assert "actor_trust_policy_revocations" in tables
        assert "evidence_bundle_revocations" in tables
        assert [row[0] for row in con.execute("SELECT version FROM schema_migrations")] == [1, 2, 3, 4, 5]
        assert "display_seq" in _columns(con, "actors")
        triggers = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            )
        }
        assert migrations.ACTOR_SEQ_TRIGGER in triggers
        indexes = {
            row[1]
            for row in con.execute("PRAGMA index_list(actors)")
        }
        assert "ix_actors_created_order" in indexes
    finally:
        con.close()


def test_fresh_database_stamps_new_actors_with_dense_sequence(tmp_path):
    db_path = tmp_path / "fresh.db"
    url = f"sqlite:///{db_path.as_posix()}"
    app = create_app(Settings(database_url=url))
    with TestClient(app) as client:
        for index in range(3):
            create_actor(client, actor_id=f"a-{index}", name=f"N{index}")
    con = _connect(db_path)
    try:
        assert [
            row[0]
            for row in con.execute(
                "SELECT display_seq FROM actors ORDER BY display_seq"
            )
        ] == [1, 2, 3]
    finally:
        con.close()


# --- Legacy database ----------------------------------------------------------


def test_legacy_database_backfills_order_and_preserves_fields(tmp_path):
    db_path = tmp_path / "legacy.db"
    _create_legacy_database(db_path)
    url = f"sqlite:///{db_path.as_posix()}"

    with TestClient(create_app(Settings(database_url=url))) as client:
        body = client.get(ACTORS_PATH).json()

    # Same-created_at ties follow the original insertion order.
    assert [item["id"] for item in body["items"]] == [a[0] for a in LEGACY_ACTORS]
    assert body["count"] == 4
    for item, (actor_id, name, actor_type) in zip(body["items"], LEGACY_ACTORS):
        assert item["id"] == actor_id
        assert item["name"] == name
        assert item["type"] == actor_type
        assert item["created_at"].startswith("2026-03-01T00:00:00")
        assert "display_seq" not in item

    con = _connect(db_path)
    try:
        assert [row[0] for row in con.execute("SELECT version FROM schema_migrations")] == [1, 2, 3, 4, 5]
        assert [
            row[0]
            for row in con.execute(
                "SELECT id FROM actors ORDER BY created_at, display_seq"
            )
        ] == [a[0] for a in LEGACY_ACTORS]
        assert [
            row[0]
            for row in con.execute(
                "SELECT display_seq FROM actors ORDER BY display_seq"
            )
        ] == [1, 2, 3, 4]
        # No actor can ever be missed or reordered by the backfill.
        assert con.execute(
            "SELECT COUNT(*) FROM actors WHERE display_seq BETWEEN 1 AND 4"
        ).fetchone()[0] == 4
    finally:
        con.close()


def test_legacy_migration_continues_sequence_for_later_inserts(tmp_path):
    db_path = tmp_path / "legacy.db"
    _create_legacy_database(db_path)
    url = f"sqlite:///{db_path.as_posix()}"
    app = create_app(Settings(database_url=url))
    with TestClient(app) as client:
        create_actor(client, actor_id="new-1", name="New", type="device")
        body = client.get(ACTORS_PATH).json()

    # The legacy rows keep their order; the new actor sorts after them.
    assert [item["id"] for item in body["items"]] == [
        a[0] for a in LEGACY_ACTORS
    ] + ["new-1"]

    con = _connect(db_path)
    try:
        assert con.execute(
            "SELECT display_seq FROM actors WHERE id = 'new-1'"
        ).fetchone()[0] == 5
    finally:
        con.close()


def test_empty_legacy_table_migrates_and_starts_sequence_at_one(tmp_path):
    db_path = tmp_path / "empty-legacy.db"
    con = _connect(db_path)
    try:
        con.execute(
            "CREATE TABLE actors ("
            "id VARCHAR(255) PRIMARY KEY, "
            "name TEXT NOT NULL, "
            "type VARCHAR(64) NOT NULL, "
            "created_at DATETIME NOT NULL)"
        )
        con.commit()
    finally:
        con.close()
    url = f"sqlite:///{db_path.as_posix()}"
    app = create_app(Settings(database_url=url))
    with TestClient(app) as client:
        create_actor(client, actor_id="only-1", name="Only", type="device")
        body = client.get(ACTORS_PATH).json()
    assert [item["id"] for item in body["items"]] == ["only-1"]
    con = _connect(db_path)
    try:
        assert [row[0] for row in con.execute("SELECT version FROM schema_migrations")] == [1, 2, 3, 4, 5]
        assert con.execute(
            "SELECT display_seq FROM actors WHERE id = 'only-1'"
        ).fetchone()[0] == 1
    finally:
        con.close()


def test_migration_is_idempotent_across_restarts(tmp_path):
    db_path = tmp_path / "legacy.db"
    _create_legacy_database(db_path)
    url = f"sqlite:///{db_path.as_posix()}"

    app = create_app(Settings(database_url=url))
    with TestClient(app) as client:
        first = client.get(ACTORS_PATH)
    first_app = create_app(Settings(database_url=url))
    with TestClient(first_app) as client:
        second = client.get(ACTORS_PATH)
    third_app = create_app(Settings(database_url=url))
    with TestClient(third_app) as client:
        third = client.get(ACTORS_PATH)

    assert second.json() == first.json()
    assert third.json() == first.json()

    con = _connect(db_path)
    try:
        # Applied exactly once despite three startups.
        assert [row[0] for row in con.execute("SELECT version FROM schema_migrations")] == [1, 2, 3, 4, 5]
        assert [
            row[0]
            for row in con.execute(
                "SELECT display_seq FROM actors ORDER BY display_seq"
            )
        ] == [1, 2, 3, 4]
    finally:
        con.close()


# --- Failure rollback ---------------------------------------------------------


def test_failed_migration_rolls_back_and_a_later_startup_succeeds(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "fail.db"
    _create_legacy_database(db_path)
    url = f"sqlite:///{db_path.as_posix()}"

    def _failing_up(cursor, engine, metadata):
        cursor.execute(
            "ALTER TABLE actors ADD COLUMN display_seq BIGINT "
            "NOT NULL DEFAULT 0"
        )
        cursor.execute("UPDATE actors SET display_seq = 1")
        raise RuntimeError("simulated migration failure")

    engine = make_engine(url)
    monkeypatch.setattr(
        migrations,
        "MIGRATIONS",
        (migrations.Migration(version=1, up=_failing_up),),
    )

    with pytest.raises(RuntimeError, match="simulated migration failure"):
        migrations.run_migrations(engine)

    con = _connect(db_path)
    try:
        tables = _table_names(con)
        # No half-finished ledger and no partially added column.
        assert "schema_migrations" not in tables
        assert "display_seq" not in _columns(con, "actors")
        # Existing identity/fields are untouched.
        assert [row[0] for row in con.execute("SELECT id FROM actors ORDER BY id")] == [
            a[0] for a in sorted(LEGACY_ACTORS)
        ]
        assert con.execute("SELECT COUNT(*) FROM actors").fetchone()[0] == 4
    finally:
        con.close()

    # A later, healthy startup migrates cleanly (retry after rollback).
    monkeypatch.undo()
    with TestClient(create_app(Settings(database_url=url))) as client:
        body = client.get(ACTORS_PATH).json()
    assert [item["id"] for item in body["items"]] == [
        a[0] for a in LEGACY_ACTORS
    ]
    con = _connect(db_path)
    try:
        assert [row[0] for row in con.execute("SELECT version FROM schema_migrations")] == [1, 2, 3, 4, 5]
        assert "display_seq" in _columns(con, "actors")
    finally:
        con.close()


# --- Read-only migration ledger ----------------------------------------------


def test_actor_list_writes_no_migration_record(tmp_path):
    db_path = tmp_path / "ro.db"
    url = f"sqlite:///{db_path.as_posix()}"
    app = create_app(Settings(database_url=url))
    with TestClient(app) as client:
        client.get(ACTORS_PATH)
        client.get(ACTORS_PATH, params={"type": "unicorn"})
        client.get(ACTORS_PATH, params={"limit": "0"})
        client.get(ACTORS_PATH, params={"cursor": "bad"})
        client.request("GET", ACTORS_PATH, content=b"{}")
    con = _connect(db_path)
    try:
        # Still just the single baselined version; reads added nothing.
        assert [row[0] for row in con.execute("SELECT version FROM schema_migrations")] == [1, 2, 3, 4, 5]
        assert con.execute("SELECT COUNT(*) FROM actors").fetchone()[0] == 0
    finally:
        con.close()


# --- Version-5 legacy upgrade -------------------------------------------------

#: Tables a database that shipped versions 1-4 already carries. Everything
#: else the current schema defines is exactly what version 5 must add.
V4_TABLES = frozenset(
    {
        "actors",
        "contents",
        "claims",
        "claim_supersessions",
        "evidence_bundles",
        "evidence_bundle_revocations",
        "attestations",
        "attestation_revocations",
        "attestation_access_grants",
        "attestation_access_grant_revocations",
        "attestation_access_grant_expiries",
        "actor_trust_policy_revocations",
        "audit_events",
        "schema_migrations",
    }
)

#: V4-era business tables whose rows an upgrade must never touch. The
#: ledger itself is excluded: advancing it is the point of the upgrade.
V4_DATA_TABLES = frozenset(V4_TABLES - {"schema_migrations"})

#: Deterministic seed for the rotated key introduced after the upgrade.
SEED_R1 = b"test-ed25519-rotate-r1-000000000000"[:32]


def _signed_headers(method, path, body, *, actor="org-1", seed=SEED_A):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    message = access_message_bytes(
        method, path, ts, hashlib.sha256(body).hexdigest()
    )
    signature = base64.b64encode(ed25519_sign(seed, message)).decode("ascii")
    return {"X-PA": actor, "X-PT": ts, "X-PS": signature}


def _signed_post(client, path, payload, *, actor="org-1", seed=SEED_A):
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", path, body, actor=actor, seed=seed),
    }
    return client.post(path, content=body, headers=headers)


def _legacy_world(client):
    """Populate v4-era business data through the public API."""
    create_actor(client, actor_id="org-1", name="Example Org")
    create_actor(client, actor_id="org-2", name="Other Org")
    content = client.post(
        "/v1/contents", json=content_payload(actor_id="org-1", digest=DIGEST_A)
    ).json()
    derived = client.post(
        "/v1/contents", json=content_payload(actor_id="org-1", digest=DIGEST_B)
    ).json()
    claim = client.post(
        "/v1/claims",
        json={
            "content_id": content["id"],
            "actor_id": "org-1",
            "claim_type": "authorship",
            "payload": {"statement": "captured"},
        },
    ).json()
    # org-1's attestation key authenticates the protected routes later.
    signature = ed25519_sign(
        SEED_A, attestation_message_bytes("claim", claim["id"], "org-1")
    )
    attestation = client.post(
        "/v1/attestations",
        json={
            "target_type": "claim",
            "target_id": claim["id"],
            "signer_actor_id": "org-1",
            "public_key": base64.b64encode(ed25519_public_key(SEED_A)).decode(),
            "signature": base64.b64encode(signature).decode(),
        },
    ).json()
    bundle = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim["id"],
            "evidence_type": "raw_capture",
            "digest_algorithm": "sha256",
            "digest_hex": hashlib.sha256(b"evidence-legacy").hexdigest(),
            "media_type": "image/jpeg",
            "metadata": {"source": "camera-1"},
        },
    ).json()
    revocation = client.post(
        "/v1/evidence-bundle-revocations",
        json={
            "evidence_bundle_id": bundle["id"],
            "revoker_actor_id": "org-2",
            "reason": "superseded by a newer capture",
        },
    ).json()
    return {
        "content": content,
        "derived": derived,
        "claim": claim,
        "attestation": attestation,
        "bundle": bundle,
        "revocation": revocation,
    }


def _downgrade_to_v4(db_path):
    """Shape a populated current database as a legacy version-4 database."""
    con = _connect(db_path)
    try:
        tables = _table_names(con)
        for table in sorted(tables - V4_TABLES):
            con.execute(f'DROP TABLE "{table}"')
        con.execute("DELETE FROM schema_migrations WHERE version > 4")
        con.commit()
    finally:
        con.close()


def _create_v4_legacy_database(db_path):
    """Build a populated database exactly as the version-4 service left it."""
    url = f"sqlite:///{db_path.as_posix()}"
    with TestClient(create_app(Settings(database_url=url))) as client:
        world = _legacy_world(client)
    _downgrade_to_v4(db_path)
    return world


def _table_rows(db_path, table):
    con = _connect(db_path)
    try:
        return con.execute(f'SELECT * FROM "{table}"').fetchall()
    finally:
        con.close()


def _schema_signature(db_path):
    """Structural signature: every table's columns, every index's columns,
    and every trigger's statement -- independent of creation SQL spelling."""
    con = _connect(db_path)
    try:
        objects = con.execute(
            "SELECT type, name FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' "
            "AND type IN ('table', 'index', 'trigger') "
            "ORDER BY type, name"
        ).fetchall()
        signature = []
        for object_type, name in objects:
            if object_type == "table":
                detail = con.execute(
                    f'PRAGMA table_info("{name}")'
                ).fetchall()
            elif object_type == "index":
                detail = con.execute(
                    f'PRAGMA index_info("{name}")'
                ).fetchall()
            else:
                detail = con.execute(
                    "SELECT sql FROM sqlite_master WHERE name = ?", (name,)
                ).fetchall()
            signature.append((object_type, name, tuple(detail)))
        return signature
    finally:
        con.close()


def _ledger_versions(db_path):
    con = _connect(db_path)
    try:
        return [
            row[0]
            for row in con.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            )
        ]
    finally:
        con.close()


def test_v4_legacy_upgrade_reaches_current_schema_and_preserves_rows(tmp_path):
    db_path = tmp_path / "legacy.db"
    _create_v4_legacy_database(db_path)
    url = f"sqlite:///{db_path.as_posix()}"

    assert _ledger_versions(db_path) == [1, 2, 3, 4]
    rows_before = {
        table: _table_rows(db_path, table) for table in sorted(V4_DATA_TABLES)
    }

    with TestClient(create_app(Settings(database_url=url))):
        pass

    # The ledger advanced to the new determined version, exactly once.
    assert _ledger_versions(db_path) == [1, 2, 3, 4, 5]
    # Every pre-existing row -- actors, content, claims, evidence,
    # attestations, revocations, grants, and audit events -- is untouched.
    for table in sorted(V4_DATA_TABLES):
        assert _table_rows(db_path, table) == rows_before[table]
    # Every current metadata table now exists.
    con = _connect(db_path)
    try:
        assert set(Base.metadata.tables) <= _table_names(con)
    finally:
        con.close()
    # The upgraded database is structurally identical to a fresh one.
    fresh_path = tmp_path / "fresh.db"
    fresh_url = f"sqlite:///{fresh_path.as_posix()}"
    with TestClient(create_app(Settings(database_url=fresh_url))):
        pass
    assert _schema_signature(db_path) == _schema_signature(fresh_path)


def test_v4_legacy_upgrade_is_idempotent_across_restarts(tmp_path):
    db_path = tmp_path / "legacy.db"
    _create_v4_legacy_database(db_path)
    url = f"sqlite:///{db_path.as_posix()}"

    with TestClient(create_app(Settings(database_url=url))):
        pass
    signature = _schema_signature(db_path)
    rows_before = {
        table: _table_rows(db_path, table) for table in sorted(V4_DATA_TABLES)
    }

    for _ in range(2):
        with TestClient(create_app(Settings(database_url=url))):
            pass
    # No repeated ledger writes, no schema drift, no data change.
    assert _ledger_versions(db_path) == [1, 2, 3, 4, 5]
    assert _schema_signature(db_path) == signature
    for table in sorted(V4_DATA_TABLES):
        assert _table_rows(db_path, table) == rows_before[table]


def test_upgraded_v4_database_serves_current_features(tmp_path):
    db_path = tmp_path / "legacy.db"
    world = _create_v4_legacy_database(db_path)
    url = f"sqlite:///{db_path.as_posix()}"

    app = create_app(Settings(database_url=url))
    with TestClient(app) as client:
        # The legacy rows are served in their original order.
        actors = client.get(ACTORS_PATH).json()
        assert [item["id"] for item in actors["items"]] == ["org-1", "org-2"]
        assert (
            client.get(f"/v1/contents/{world['content']['id']}").json()
            == world["content"]
        )
        assert (
            client.get(f"/v1/claims/{world['claim']['id']}").json()
            == world["claim"]
        )

        # Authentication key rotation: create and read back.
        rotation = _signed_post(
            client,
            "/v1/authentication-key-rotations",
            {
                "actor_id": "org-1",
                "new_public_key": base64.b64encode(
                    ed25519_public_key(SEED_R1)
                ).decode(),
            },
        )
        assert rotation.status_code == 201, rotation.text
        assert (
            client.get(
                f"/v1/authentication-key-rotations/{rotation.json()['id']}"
            ).json()
            == rotation.json()
        )

        # Subject trust policy and a trust decision that uses it.
        policy = _signed_post(
            client, "/v1/trust-policies", {"actor_id": "org-1", "threshold": 1}
        )
        assert policy.status_code == 201, policy.text
        decision = client.get(
            "/v1/trust-decisions",
            params={"target_type": "claim", "target_id": world["claim"]["id"]},
            headers=_signed_headers("GET", "/v1/trust-decisions", b""),
        )
        assert decision.status_code == 200, decision.text
        assert decision.json()["policy_id"] == policy.json()["id"]

        # Content relation and lineage over the legacy contents.
        relation = client.post(
            "/v1/content-relations",
            json={
                "content_id": world["derived"]["id"],
                "parent_content_id": world["content"]["id"],
                "relation_type": "version_of",
            },
        )
        assert relation.status_code == 201, relation.text
        lineage = client.get(
            f"/v1/contents/{world['derived']['id']}/lineage",
            params={"direction": "ancestors"},
        ).json()
        assert [item["id"] for item in lineage["items"]] == [
            world["content"]["id"]
        ]

        # The three asynchronous job queues: create and run one job each.
        export_job = client.post(
            "/v1/content-export-jobs",
            json={
                "content_id": world["content"]["id"],
                "request_id": "legacy-export-1",
            },
        )
        assert export_job.status_code == 201, export_job.text
        export_run = client.post(
            f"/v1/content-export-jobs/{export_job.json()['id']}/run"
        )
        assert export_run.json()["status"] == "succeeded"
        assert export_run.json()["result"]["content"] == world["content"]

        checkpoint_job = client.post(
            "/v1/audit-checkpoint-jobs", json={"request_id": "legacy-acp-1"}
        )
        assert checkpoint_job.status_code == 201, checkpoint_job.text
        checkpoint_run = client.post(
            f"/v1/audit-checkpoint-jobs/{checkpoint_job.json()['id']}/run"
        )
        assert checkpoint_run.json()["status"] == "succeeded"

        bundle_job = client.post(
            "/v1/evidence-bundle-export-jobs",
            json={
                "evidence_bundle_id": world["bundle"]["id"],
                "request_id": "legacy-ebx-1",
            },
        )
        assert bundle_job.status_code == 201, bundle_job.text
        bundle_run = client.post(
            f"/v1/evidence-bundle-export-jobs/{bundle_job.json()['id']}/run"
        )
        assert bundle_run.json()["status"] == "succeeded"

        # The import receipts and reconciliation views are readable.
        for path in (
            "/v1/csp-imports",
            "/v1/audit-events/checkpoint-imports",
            "/v1/evidence-bundle-exchange-imports",
            "/v1/audit-exchanges",
            "/v1/audit-recon-exchanges",
            "/v1/impact-imports",
            "/v1/impact-recon-exchange-imports",
        ):
            page = client.get(path)
            assert page.status_code == 200, path
            assert page.json()["count"] == 0

        # The pre-existing audit rows are intact and new events appended.
        audits = client.get("/v1/audit-events").json()
        assert audits["count"] > 0


def test_failed_v5_migration_rolls_back_and_a_later_startup_succeeds(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "fail.db"
    _create_v4_legacy_database(db_path)
    url = f"sqlite:///{db_path.as_posix()}"
    rows_before = {
        table: _table_rows(db_path, table) for table in sorted(V4_DATA_TABLES)
    }

    def _failing_up(cursor, engine, metadata):
        cursor.execute(
            "CREATE TABLE authentication_key_rotations (seq INTEGER)"
        )
        raise RuntimeError("simulated migration failure")

    monkeypatch.setattr(
        migrations,
        "MIGRATIONS",
        migrations.MIGRATIONS[:4]
        + (migrations.Migration(version=5, up=_failing_up),),
    )
    engine = make_engine(url)
    with pytest.raises(RuntimeError, match="simulated migration failure"):
        migrations.run_migrations(engine)

    con = _connect(db_path)
    try:
        # No half-created table, no new ledger row, old data intact.
        assert "authentication_key_rotations" not in _table_names(con)
    finally:
        con.close()
    assert _ledger_versions(db_path) == [1, 2, 3, 4]
    for table in sorted(V4_DATA_TABLES):
        assert _table_rows(db_path, table) == rows_before[table]

    # A later, healthy startup completes the upgrade.
    monkeypatch.undo()
    with TestClient(create_app(Settings(database_url=url))):
        pass
    assert _ledger_versions(db_path) == [1, 2, 3, 4, 5]
    con = _connect(db_path)
    try:
        assert "authentication_key_rotations" in _table_names(con)
    finally:
        con.close()
    for table in sorted(V4_DATA_TABLES):
        assert _table_rows(db_path, table) == rows_before[table]
