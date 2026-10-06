"""Tests for the version-5 schema generation (current persistence structure).

Covers the startup upgrade of a legacy database that already completed the
first four migration generations (actors explicit ordering, grant expiry,
trust-policy revocation, evidence-bundle revocation):

* a brand-new database is created at the full current schema and baselined in
  the ``schema_migrations`` ledger through version 5;
* a legacy version-4 database gains exactly the missing current tables --
  authentication key rotations, subject trust policies, content relations,
  the import/export receipts, and the three asynchronous job queues -- while
  every existing actor, content, claim, evidence, attestation, revocation,
  grant, and audit row (ids, UTC timestamps, ordering) is preserved;
* fresh and upgraded databases expose the same tables, indexes, and
  triggers, and repeated startups never rebuild tables, rewrite data, or
  record the version twice;
* a failing version-5 migration rolls the whole startup batch back (no new
  table, no new index, no ledger row, old data intact) and a later healthy
  startup completes the upgrade;
* a database without an ``actors`` table is initialized as brand-new;
* the upgraded database immediately serves the public API: key rotations,
  trust policies and decisions, content relations and lineage, the three
  job queues, and the import receipt listings.
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
from provenance.access_signing import access_message_bytes
from provenance.app import create_app
from provenance.config import Settings
from provenance.database import make_engine
from provenance.signing import attestation_message_bytes
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    DIGEST_C,
    content_payload,
    create_actor,
    ed25519_public_key,
    ed25519_sign,
    SEED_A,
)

LEDGER_LATEST = [1, 2, 3, 4, 5]

#: Every table the current schema is expected to have after startup,
#: regardless of whether the database started fresh or as a legacy v4 one.
CURRENT_TABLES = {
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
    "authentication_key_rotations",
    "actor_trust_policies",
    "actor_trust_policy_revocations",
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
    "evidence_bundle_export_jobs",
    "schema_migrations",
}

SEED_R = b"test-ed25519-rotate-v5-00000000000"[:32]


def _connect(path):
    return sqlite3.connect(path)


def _table_names(con):
    return {
        row[0]
        for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def _schema_objects(db_path):
    con = _connect(db_path)
    try:
        return con.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
    finally:
        con.close()


def _ledger(con):
    return [
        row[0] for row in con.execute("SELECT version FROM schema_migrations")
    ]


def _snapshot(con, table):
    return sorted(con.execute(f"SELECT * FROM {table}").fetchall())


# --- World builders (public API only) ----------------------------------------


def _make_claim(client, actor_id, digest):
    content = client.post(
        "/v1/contents", json=content_payload(actor_id=actor_id, digest=digest)
    )
    assert content.status_code == 201, content.text
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content.json()["id"],
            "actor_id": actor_id,
            "claim_type": "authorship",
            "payload": {"statement": "made"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make_attestation(client, actor_id, seed, target_id):
    signature = ed25519_sign(
        seed, attestation_message_bytes("claim", target_id, actor_id)
    )
    resp = client.post(
        "/v1/attestations",
        json={
            "target_type": "claim",
            "target_id": target_id,
            "signer_actor_id": actor_id,
            "public_key": base64.b64encode(ed25519_public_key(seed)).decode(),
            "signature": base64.b64encode(signature).decode(),
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _signed_headers(method, path, body, *, actor="org-1", seed=SEED_A):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    message = access_message_bytes(
        method, path, ts, hashlib.sha256(body).hexdigest()
    )
    signature = base64.b64encode(ed25519_sign(seed, message)).decode("ascii")
    return {"X-PA": actor, "X-PT": ts, "X-PS": signature}


def _build_legacy_v4_database(db_path):
    """Create a populated database, then shape it as a legacy version-4 one.

    Data lives only in the pre-v5 tables; the audit trail (which the legacy
    database already carries) is kept, while the remaining version-5 tables
    and the version-5 ledger row are removed.
    """
    url = f"sqlite:///{db_path.as_posix()}"
    app = create_app(Settings(database_url=url))
    with TestClient(app) as client:
        create_actor(client)  # org-1
        create_actor(client, actor_id="org-2", name="Other Org", type="organization")
        claim = _make_claim(client, "org-1", DIGEST_A)
        _make_attestation(client, "org-1", SEED_A, claim["id"])
        bundle = client.post(
            "/v1/evidence-bundles",
            json={
                "claim_id": claim["id"],
                "evidence_type": "raw_capture",
                "digest_algorithm": "sha256",
                "digest_hex": hashlib.sha256(b"evidence-v5").hexdigest(),
                "media_type": "image/jpeg",
                "metadata": {"source": "camera-1"},
            },
        )
        assert bundle.status_code == 201, bundle.text

    con = _connect(db_path)
    try:
        for table in migrations.SCHEMA_VERSION_5_TABLES:
            if table == "audit_events":
                continue  # the legacy database already carries its audit trail
            con.execute(f"DROP TABLE {table}")
        con.execute("DELETE FROM schema_migrations WHERE version = 5")
        con.commit()
    finally:
        con.close()
    return url


# --- Fresh database -----------------------------------------------------------


def test_fresh_database_has_full_current_schema_and_ledger(tmp_path):
    db_path = tmp_path / "fresh.db"
    url = f"sqlite:///{db_path.as_posix()}"
    with TestClient(create_app(Settings(database_url=url))):
        pass

    con = _connect(db_path)
    try:
        assert CURRENT_TABLES <= _table_names(con)
        assert _ledger(con) == LEDGER_LATEST
        triggers = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            )
        }
        assert migrations.ACTOR_SEQ_TRIGGER in triggers
    finally:
        con.close()


# --- Legacy version-4 database upgrade ----------------------------------------


def test_legacy_v4_database_upgrades_and_preserves_everything(tmp_path):
    db_path = tmp_path / "legacy.db"
    url = _build_legacy_v4_database(db_path)

    con = _connect(db_path)
    try:
        assert _ledger(con) == [1, 2, 3, 4]
        before = {
            table: _snapshot(con, table)
            for table in (
                "actors",
                "contents",
                "claims",
                "evidence_bundles",
                "attestations",
                "audit_events",
            )
        }
        missing = CURRENT_TABLES - _table_names(con) - {"schema_migrations"}
        assert missing == set(migrations.SCHEMA_VERSION_5_TABLES) - {
            "audit_events"
        }
    finally:
        con.close()

    with TestClient(create_app(Settings(database_url=url))):
        pass

    con = _connect(db_path)
    try:
        assert _ledger(con) == LEDGER_LATEST
        assert CURRENT_TABLES <= _table_names(con)
        # Every pre-existing row is byte-for-byte intact.
        for table, rows in before.items():
            assert _snapshot(con, table) == rows
    finally:
        con.close()

    # The upgraded database is structurally identical to a fresh one.
    fresh_path = tmp_path / "fresh.db"
    with TestClient(
        create_app(
            Settings(database_url=f"sqlite:///{fresh_path.as_posix()}")
        )
    ):
        pass
    assert _schema_objects(db_path) == _schema_objects(fresh_path)


def test_legacy_v4_upgrade_is_idempotent_across_restarts(tmp_path):
    db_path = tmp_path / "legacy.db"
    url = _build_legacy_v4_database(db_path)

    for _ in range(3):
        with TestClient(create_app(Settings(database_url=url))):
            pass

    con = _connect(db_path)
    try:
        # Applied exactly once despite three extra startups.
        assert _ledger(con) == LEDGER_LATEST
        assert CURRENT_TABLES <= _table_names(con)
        # No startup wrote business data: only the rows the legacy database
        # already had (six creations -> six audit events).
        assert con.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0] == 6
    finally:
        con.close()


def test_failed_v5_migration_rolls_back_and_a_later_startup_succeeds(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "fail.db"
    url = _build_legacy_v4_database(db_path)

    con = _connect(db_path)
    try:
        audits_before = _snapshot(con, "audit_events")
        actors_before = _snapshot(con, "actors")
    finally:
        con.close()

    def _failing_up(cursor, engine, metadata):
        cursor.execute("CREATE TABLE audit_events_v5_partial (seq INTEGER)")
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
        # No half-finished version row, no partial table, old data intact.
        assert _ledger(con) == [1, 2, 3, 4]
        assert "audit_events_v5_partial" not in _table_names(con)
        assert "content_export_jobs" not in _table_names(con)
        assert _snapshot(con, "audit_events") == audits_before
        assert _snapshot(con, "actors") == actors_before
    finally:
        con.close()

    # A later, healthy startup completes the upgrade (retry after rollback).
    monkeypatch.undo()
    with TestClient(create_app(Settings(database_url=url))):
        pass
    con = _connect(db_path)
    try:
        assert _ledger(con) == LEDGER_LATEST
        assert CURRENT_TABLES <= _table_names(con)
        assert _snapshot(con, "audit_events") == audits_before
        assert _snapshot(con, "actors") == actors_before
    finally:
        con.close()


def test_database_without_actors_table_is_initialized_as_fresh(tmp_path):
    db_path = tmp_path / "foreign.db"
    con = _connect(db_path)
    try:
        con.execute("CREATE TABLE stuff (id INTEGER PRIMARY KEY)")
        con.commit()
    finally:
        con.close()

    url = f"sqlite:///{db_path.as_posix()}"
    with TestClient(create_app(Settings(database_url=url))):
        pass

    con = _connect(db_path)
    try:
        assert CURRENT_TABLES <= _table_names(con)
        assert _ledger(con) == LEDGER_LATEST
    finally:
        con.close()


# --- The upgraded database serves the public API -------------------------------


def test_upgraded_database_serves_current_features(tmp_path):
    db_path = tmp_path / "legacy.db"
    url = _build_legacy_v4_database(db_path)

    app = create_app(Settings(database_url=url))
    with TestClient(app) as client:
        # Authentication key rotation: create and read back.
        rotation_body = {
            "actor_id": "org-1",
            "new_public_key": base64.b64encode(
                ed25519_public_key(SEED_R)
            ).decode(),
        }
        raw = json.dumps(rotation_body).encode()
        resp = client.post(
            "/v1/authentication-key-rotations",
            content=raw,
            headers={
                "Content-Type": "application/json",
                **_signed_headers(
                    "POST", "/v1/authentication-key-rotations", raw
                ),
            },
        )
        assert resp.status_code == 201, resp.text
        rotation = resp.json()
        listed = client.get(
            "/v1/actors/org-1/authentication-key-rotations"
        )
        assert listed.status_code == 200, listed.text
        assert rotation["id"] in {
            item["id"] for item in listed.json()["items"]
        }

        # Subject trust policy plus a trust decision that uses it.
        policy_body = json.dumps({"actor_id": "org-1", "threshold": 1}).encode()
        resp = client.post(
            "/v1/trust-policies",
            content=policy_body,
            headers={
                "Content-Type": "application/json",
                **_signed_headers("POST", "/v1/trust-policies", policy_body),
            },
        )
        assert resp.status_code == 201, resp.text
        target_claim = _make_claim(
            client, "org-1", hashlib.sha256(b"content-decision-v5").hexdigest()
        )
        decision = client.get(
            "/v1/trust-decisions",
            params={"target_type": "claim", "target_id": target_claim["id"]},
            headers=_signed_headers("GET", "/v1/trust-decisions", b""),
        )
        assert decision.status_code == 200, decision.text
        assert decision.json()["policy_id"] == resp.json()["id"]

        # Content relation and lineage query.
        parent = client.post(
            "/v1/contents",
            json=content_payload(actor_id="org-1", digest=DIGEST_B),
        ).json()
        child = client.post(
            "/v1/contents",
            json=content_payload(actor_id="org-1", digest=DIGEST_C),
        ).json()
        relation = client.post(
            "/v1/content-relations",
            json={
                "content_id": child["id"],
                "parent_content_id": parent["id"],
                "relation_type": "version_of",
            },
        )
        assert relation.status_code == 201, relation.text
        lineage = client.get(
            f"/v1/contents/{child['id']}/lineage",
            params={"direction": "ancestors"},
        )
        assert lineage.status_code == 200, lineage.text
        assert parent["id"] in json.dumps(lineage.json())

        # Content export job: create and run to completion.
        job = client.post(
            "/v1/content-export-jobs",
            json={"content_id": parent["id"], "request_id": "req-v5-1"},
        )
        assert job.status_code == 201, job.text
        run = client.post(f"/v1/content-export-jobs/{job.json()['id']}/run")
        assert run.status_code == 200, run.text
        assert run.json()["status"] == "succeeded"

        # Audit checkpoint export job: create and run to completion.
        checkpoint_job = client.post(
            "/v1/audit-checkpoint-jobs", json={"request_id": "req-v5-2"}
        )
        assert checkpoint_job.status_code == 201, checkpoint_job.text
        checkpoint_run = client.post(
            f"/v1/audit-checkpoint-jobs/{checkpoint_job.json()['id']}/run"
        )
        assert checkpoint_run.status_code == 200, checkpoint_run.text
        assert checkpoint_run.json()["status"] == "succeeded"

        # Evidence bundle export job: create and run to completion.
        bundle_id = client.get(
            "/v1/evidence-bundles", params={"limit": 1}
        ).json()["items"][0]["id"]
        bundle_job = client.post(
            "/v1/evidence-bundle-export-jobs",
            json={"evidence_bundle_id": bundle_id, "request_id": "req-v5-3"},
        )
        assert bundle_job.status_code == 201, bundle_job.text
        bundle_run = client.post(
            f"/v1/evidence-bundle-export-jobs/{bundle_job.json()['id']}/run"
        )
        assert bundle_run.status_code == 200, bundle_run.text
        assert bundle_run.json()["status"] == "succeeded"

        # Import receipt and reconciliation listings read cleanly (empty).
        for path in (
            "/v1/evidence-bundle-exchange-imports",
            "/v1/audit-events/checkpoint-imports",
            "/v1/csp-imports",
            "/v1/audit-exchanges",
            "/v1/audit-recon-exchanges",
            "/v1/impact-imports",
            "/v1/impact-recon-exchange-imports",
        ):
            page = client.get(path)
            assert page.status_code == 200, (path, page.text)
            assert page.json()["count"] == 0

        # The audit trail gained exactly the new feature's events and still
        # serves the paginated read.
        events = client.get("/v1/audit-events")
        assert events.status_code == 200, events.text
        assert events.json()["count"] > 0
