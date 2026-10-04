"""Tests for subject trust-policy revocation.

Covers ``POST /v1/trust-policy-revocations``: the protected signed-header
contract, first-revocation 201 with the exact public view (compact UTF-8 JSON
terminated by exactly one newline, members in id, policy_id, actor_id,
reason, created_at order), the stable ``tpr_`` identifier, the
single-transaction ``actor_trust_policy.revoked`` audit commit, same-reason
retry idempotency (200, no new row/audit/id), the different-reason 409
``trust_policy_revocation_conflict``, the opaque 404 shared by an unknown
policy and a non-subject caller, strict body validation (missing/wrong/extra/
duplicate fields, blank or out-of-range verbatim reason, non-object JSON),
the no-write guarantee on every failure, preservation of the policy and its
history, the post-revocation ``policy_missing`` decision that never reads the
target, unchanged semantics for unrevoked subjects, and the version-3 SQLite
migration (legacy upgrade preserves policies and audits, failure rolls back,
restart consistency).

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures).
"""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from provenance import migrations
from provenance.access_signing import access_message_bytes
from provenance.app import create_app
from provenance.config import Settings
from provenance.database import make_engine
from provenance.ids import actor_trust_policy_revocation_id
from provenance.models import (
    EVENT_ACTOR_TRUST_POLICY_CREATED,
    EVENT_ACTOR_TRUST_POLICY_REVOKED,
    ActorTrustPolicy,
    ActorTrustPolicyRevocation,
    AuditEvent,
)
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
    SEED_B,
)

POLICIES_PATH = "/v1/trust-policies"
REVOCATIONS_PATH = "/v1/trust-policy-revocations"
DECISIONS_PATH = "/v1/trust-decisions"
EVALUATIONS_PATH = "/v1/trust-evaluations"


# --- Fixture-style setup ------------------------------------------------------


def _make_claim(client, actor_id, digest=DIGEST_A):
    content = client.post(
        "/v1/contents",
        json=content_payload(actor_id=actor_id, digest=digest),
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


def _attest(client, target_id, actor_id, seed):
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


def _world(client):
    """Two actors, each holding a current authentication key."""
    create_actor(client)  # org-1
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    _attest(client, _make_claim(client, "org-1")["id"], "org-1", SEED_A)
    _attest(
        client, _make_claim(client, "org-2", digest=DIGEST_B)["id"], "org-2", SEED_B
    )


# --- Signed-request helpers ---------------------------------------------------


def _signed_headers(
    method,
    path,
    body,
    *,
    actor="org-1",
    seed=SEED_A,
    timestamp=None,
):
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    message = access_message_bytes(
        method, path, ts, hashlib.sha256(body).hexdigest()
    )
    signature = base64.b64encode(ed25519_sign(seed, message)).decode("ascii")
    return {"X-PA": actor, "X-PT": ts, "X-PS": signature}


def _create_policy(client, actor="org-1", seed=SEED_A, threshold=2):
    body = json.dumps({"actor_id": actor, "threshold": threshold}).encode()
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", POLICIES_PATH, body, actor=actor, seed=seed),
    }
    resp = client.post(POLICIES_PATH, content=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _post_revocation(client, body, *, actor="org-1", seed=SEED_A, headers=None):
    if isinstance(body, (dict, list)):
        body = json.dumps(body).encode("utf-8")
    elif isinstance(body, str):
        body = body.encode("utf-8")
    if headers is None:
        headers = _signed_headers(
            "POST", REVOCATIONS_PATH, body, actor=actor, seed=seed
        )
    return client.post(
        REVOCATIONS_PATH,
        content=body,
        headers={"Content-Type": "application/json", **headers},
    )


def _revoke(client, policy_id, reason="no longer relied upon", **kwargs):
    resp = _post_revocation(
        client, {"policy_id": policy_id, "reason": reason}, **kwargs
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _get_decision(client, target_type, target_id, *, actor="org-1", seed=SEED_A):
    headers = _signed_headers("GET", DECISIONS_PATH, b"", actor=actor, seed=seed)
    return client.get(
        DECISIONS_PATH,
        params={"target_type": target_type, "target_id": target_id},
        headers=headers,
    )


def _revocation_rows(session):
    return session.execute(select(ActorTrustPolicyRevocation)).scalars().all()


def _revoked_audit_events(session):
    return session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ACTOR_TRUST_POLICY_REVOKED
        )
    ).scalars().all()


# --- First revocation ---------------------------------------------------------


def test_first_revocation_returns_201_with_exact_public_view(client, db_session):
    _world(client)
    policy = _create_policy(client, threshold=2)

    resp = _post_revocation(
        client, {"policy_id": policy["id"], "reason": "key material retired"}
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert list(body) == ["id", "policy_id", "actor_id", "reason", "created_at"]
    assert body["id"] == actor_trust_policy_revocation_id(policy["id"])
    assert body["id"].startswith("tpr_")
    assert len(body["id"]) == len("tpr_") + 64
    assert body["policy_id"] == policy["id"]
    assert body["actor_id"] == "org-1"
    assert body["reason"] == "key material retired"
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))


def test_revocation_response_is_compact_utf8_json_with_one_newline(client):
    _world(client)
    policy = _create_policy(client)
    resp = _post_revocation(
        client, {"policy_id": policy["id"], "reason": "done"}
    )
    assert resp.status_code == 201
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    # Members appear in the declared public order.
    assert raw.index(b'"id"') < raw.index(b'"policy_id"')
    assert raw.index(b'"policy_id"') < raw.index(b'"actor_id"')
    assert raw.index(b'"actor_id"') < raw.index(b'"reason"')
    assert raw.index(b'"reason"') < raw.index(b'"created_at"')
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    assert raw == expected


def test_revocation_and_audit_commit_in_one_transaction(client, db_session):
    _world(client)
    policy = _create_policy(client, threshold=3)
    before_policy = db_session.execute(
        select(ActorTrustPolicy).where(ActorTrustPolicy.id == policy["id"])
    ).scalar_one()

    body = _revoke(client, policy["id"], reason="subject withdrew the rule")

    rows = _revocation_rows(db_session)
    assert [(r.id, r.policy_id, r.actor_id, r.reason) for r in rows] == [
        (body["id"], policy["id"], "org-1", "subject withdrew the rule")
    ]
    events = _revoked_audit_events(db_session)
    assert [e.resource_id for e in events] == [body["id"]]
    assert events[0].created_at.tzinfo.utcoffset(
        events[0].created_at
    ).total_seconds() == 0

    # The policy itself is preserved untouched: same row, threshold, enabled
    # flag, and creation timestamp, and its creation audit event remains.
    after_policy = db_session.execute(
        select(ActorTrustPolicy).where(ActorTrustPolicy.id == policy["id"])
    ).scalar_one()
    assert after_policy.threshold == before_policy.threshold == 3
    assert after_policy.enabled is True
    assert after_policy.created_at == before_policy.created_at
    created_events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ACTOR_TRUST_POLICY_CREATED
        )
    ).scalars().all()
    assert [e.resource_id for e in created_events] == [policy["id"]]


def test_reason_is_stored_verbatim_never_trimmed_or_rewritten(client):
    _world(client)
    policy = _create_policy(client)
    reason = "  撤回原因:policy moved to a new root 🔒\n"
    body = _revoke(client, policy["id"], reason=reason)
    assert body["reason"] == reason

    # Boundary length: exactly 1 Unicode character, by another subject.
    other = _create_policy(client, actor="org-2", seed=SEED_B, threshold=1)
    assert (
        _revoke(client, other["id"], reason="x", actor="org-2", seed=SEED_B)[
            "reason"
        ]
        == "x"
    )


def test_reason_of_one_thousand_characters_is_accepted(client):
    _world(client)
    policy = _create_policy(client)
    reason = "撤" * 1000
    body = _revoke(client, policy["id"], reason=reason)
    assert body["reason"] == reason


# --- Idempotency and conflict ---------------------------------------------------


def test_same_reason_retry_returns_200_original_and_no_new_writes(
    client, db_session
):
    _world(client)
    policy = _create_policy(client)
    first = _post_revocation(
        client, {"policy_id": policy["id"], "reason": "superseded externally"}
    )
    assert first.status_code == 201

    for _ in range(3):
        retry = _post_revocation(
            client, {"policy_id": policy["id"], "reason": "superseded externally"}
        )
        assert retry.status_code == 200
        assert retry.content == first.content

    assert len(_revocation_rows(db_session)) == 1
    assert len(_revoked_audit_events(db_session)) == 1


def test_different_reason_for_same_policy_is_409_and_writes_nothing(
    client, db_session
):
    _world(client)
    policy = _create_policy(client)
    first = _revoke(client, policy["id"], reason="first rationale")

    conflict = _post_revocation(
        client, {"policy_id": policy["id"], "reason": "another rationale"}
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "trust_policy_revocation_conflict"
    assert conflict.json()["error"]["details"]["policy_id"] == policy["id"]

    rows = _revocation_rows(db_session)
    assert [r.id for r in rows] == [first["id"]]
    assert len(_revoked_audit_events(db_session)) == 1


def test_concurrent_identical_revocations_yield_one_record_and_audit(
    tmp_db_url, file_app, file_client
):
    _world(file_client)
    policy = _create_policy(file_client)

    from provenance import service
    from provenance.schemas import ActorTrustPolicyRevocationCreate

    factory = file_app.state.session_factory
    payload = ActorTrustPolicyRevocationCreate(
        policy_id=policy["id"], reason="raced"
    )
    results: list[tuple[str, bool]] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        session = factory()
        try:
            barrier.wait()
            record, created = service.create_actor_trust_policy_revocation(
                session, payload, "org-1"
            )
            results.append((record.id, created))
        except Exception as exc:  # pragma: no cover - fails the test below
            errors.append(exc)
        finally:
            session.close()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(results) == 4
    # Exactly one 201: one submission created the record, the rest idempotent.
    assert sum(1 for _, created in results if created) == 1
    assert {rid for rid, _ in results} == {results[0][0]}

    audit_session = factory()
    try:
        assert len(_revocation_rows(audit_session)) == 1
        assert len(_revoked_audit_events(audit_session)) == 1
    finally:
        audit_session.close()


# --- Opaque 404 -----------------------------------------------------------------


def test_unknown_policy_and_non_subject_caller_share_the_same_404(
    client, db_session
):
    _world(client)
    policy = _create_policy(client)

    unknown = _post_revocation(
        client, {"policy_id": "atp_" + "0" * 64, "reason": "ghost"}
    )
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "not_found"

    # org-2 is authenticated but is not the policy's subject: identical 404.
    stranger = _post_revocation(
        client,
        {"policy_id": policy["id"], "reason": "not mine"},
        actor="org-2",
        seed=SEED_B,
    )
    assert stranger.status_code == 404
    assert stranger.json() == unknown.json()

    assert _revocation_rows(db_session) == []
    assert _revoked_audit_events(db_session) == []


# --- Validation -----------------------------------------------------------------


def test_field_type_and_shape_failures_are_422_and_write_nothing(
    client, db_session
):
    _world(client)
    policy = _create_policy(client)
    cases = [
        {"reason": "no policy id"},  # missing policy_id
        {"policy_id": policy["id"]},  # missing reason
        {"policy_id": 5, "reason": "x"},  # wrong policy_id type
        {"policy_id": policy["id"], "reason": 5},  # wrong reason type
        {"policy_id": policy["id"], "reason": True},
        {"policy_id": policy["id"], "reason": None},
        {"policy_id": "", "reason": "x"},  # empty policy_id
        {"policy_id": "   ", "reason": "x"},  # blank policy_id
        {"policy_id": policy["id"], "reason": ""},  # empty reason
        {"policy_id": policy["id"], "reason": "   \n\t "},  # blank reason
        {"policy_id": policy["id"], "reason": "y" * 1001},  # out of range
        {"policy_id": policy["id"], "reason": "x", "actor_id": "org-1"},  # extra
    ]
    for payload in cases:
        resp = _post_revocation(client, payload)
        assert resp.status_code == 422, payload
        assert resp.json()["error"]["code"] == "validation_error", payload
    assert _revocation_rows(db_session) == []
    assert _revoked_audit_events(db_session) == []


def test_duplicate_field_is_422_and_writes_nothing(client, db_session):
    _world(client)
    policy = _create_policy(client)
    raw = (
        '{"policy_id": "%s", "reason": "one", "reason": "two"}' % policy["id"]
    )
    resp = _post_revocation(client, raw)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert _revocation_rows(db_session) == []


def test_non_object_and_malformed_json_are_422_and_write_nothing(
    client, db_session
):
    _world(client)
    for raw in (
        b"[1, 2]",
        b'"just a string"',
        b"42",
        b"null",
        b"{not valid json",
        b"",
    ):
        resp = _post_revocation(client, raw)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["code"] == "validation_error", raw
    assert _revocation_rows(db_session) == []
    assert _revoked_audit_events(db_session) == []


def test_authentication_failures_are_422_and_write_nothing(client, db_session):
    _world(client)
    policy = _create_policy(client)
    body = json.dumps({"policy_id": policy["id"], "reason": "x"}).encode()

    # Missing credentials entirely.
    missing = client.post(
        REVOCATIONS_PATH,
        content=body,
        headers={"Content-Type": "application/json"},
    )
    assert missing.status_code == 422
    assert missing.json()["error"]["code"] == "validation_error"

    # Expired timestamp.
    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    expired = _post_revocation(
        client,
        body,
        headers=_signed_headers("POST", REVOCATIONS_PATH, body, timestamp=stale),
    )
    assert expired.status_code == 422

    # Malformed signature encoding.
    malformed_headers = _signed_headers("POST", REVOCATIONS_PATH, body)
    malformed_headers["X-PS"] = "@@@"
    malformed = _post_revocation(client, body, headers=malformed_headers)
    assert malformed.status_code == 422

    # A well-formed signature under a key that is not the caller's.
    wrong = _post_revocation(client, body, actor="org-1", seed=SEED_B)
    assert wrong.status_code == 422

    assert _revocation_rows(db_session) == []
    assert _revoked_audit_events(db_session) == []


# --- Decisions after revocation -------------------------------------------------


def test_revoked_subject_decision_is_policy_missing_without_target_lookup(
    client,
):
    _world(client)
    policy = _create_policy(client, threshold=1)
    target = _make_claim(client, "org-1", digest=DIGEST_C)
    _attest(client, target["id"], "org-2", SEED_B)

    # Before revocation the threshold is met.
    before = _get_decision(client, "claim", target["id"])
    assert before.status_code == 200
    assert before.json()["decision"] == "trusted"
    assert before.json()["reason"] == "threshold_met"

    _revoke(client, policy["id"], reason="withdrawn")

    expected = {
        "policy_id": None,
        "threshold": None,
        "qualified_signer_count": 0,
        "decision": "untrusted",
        "reason": "policy_missing",
    }
    after = _get_decision(client, "claim", target["id"])
    assert after.status_code == 200
    assert after.json() == {"target_type": "claim", "target_id": target["id"], **expected}

    # The target is never read: even a nonexistent target renders the same
    # policy_missing decision rather than any 404.
    ghost = _get_decision(client, "claim", "clm_ghost")
    assert ghost.status_code == 200
    assert ghost.json() == {"target_type": "claim", "target_id": "clm_ghost", **expected}


def test_unrevoked_subject_decision_and_evaluation_semantics_unchanged(client):
    _world(client)
    revoked_policy = _create_policy(client, actor="org-1", seed=SEED_A, threshold=1)
    kept_policy = _create_policy(client, actor="org-2", seed=SEED_B, threshold=1)
    target = _make_claim(client, "org-1", digest=DIGEST_C)
    _attest(client, target["id"], "org-2", SEED_B)

    _revoke(client, revoked_policy["id"], reason="org-1 withdrew")

    # The unrevoked subject still decides under its own policy.
    kept = _get_decision(client, "claim", target["id"], actor="org-2", seed=SEED_B)
    assert kept.status_code == 200
    assert kept.json() == {
        "target_type": "claim",
        "target_id": target["id"],
        "policy_id": kept_policy["id"],
        "threshold": 1,
        "qualified_signer_count": 1,
        "decision": "trusted",
        "reason": "threshold_met",
    }

    # Anonymous trust evaluation is independent of policies and unchanged.
    evaluation = client.get(
        EVALUATIONS_PATH,
        params={"target_type": "claim", "target_id": target["id"]},
    )
    assert evaluation.status_code == 200
    assert evaluation.json()["qualified_signer_count"] == 1
    assert evaluation.json()["decision"] == "trusted"


def test_policy_list_and_creation_idempotency_survive_revocation(client):
    _world(client)
    policy = _create_policy(client, threshold=2)
    _revoke(client, policy["id"], reason="withdrawn")

    # The paginated view still lists the revoked policy, unchanged.
    page = client.get(POLICIES_PATH, params={"actor_id": "org-1"})
    assert page.status_code == 200
    assert page.json()["count"] == 1
    assert page.json()["items"][0] == policy

    # Creation idempotency is unchanged: same threshold -> 200 original,
    # a different threshold -> 409, and no second policy ever appears.
    body = json.dumps({"actor_id": "org-1", "threshold": 2}).encode()
    retry = client.post(
        POLICIES_PATH,
        content=body,
        headers={
            "Content-Type": "application/json",
            **_signed_headers("POST", POLICIES_PATH, body),
        },
    )
    assert retry.status_code == 200
    assert retry.json() == policy

    other = json.dumps({"actor_id": "org-1", "threshold": 5}).encode()
    conflict = client.post(
        POLICIES_PATH,
        content=other,
        headers={
            "Content-Type": "application/json",
            **_signed_headers("POST", POLICIES_PATH, other),
        },
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "actor_trust_policy_conflict"


# --- Persistence and migration ---------------------------------------------------


def test_revocation_and_decision_survive_restart(tmp_db_url):
    app = create_app(Settings(database_url=tmp_db_url))
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        _world(client)
        policy = _create_policy(client, threshold=1)
        created = _revoke(client, policy["id"], reason="durable")

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        # The revocation record persists and a retry is the idempotent 200.
        retry = _post_revocation(
            client, {"policy_id": policy["id"], "reason": "durable"}
        )
        assert retry.status_code == 200
        assert retry.json() == created
        decision = _get_decision(client, "claim", "clm_anything")
        assert decision.status_code == 200
        assert decision.json()["reason"] == "policy_missing"
        assert decision.json()["policy_id"] is None


def _downgrade_to_v2(db_path):
    """Shape a current database as a legacy version-2 database."""
    con = sqlite3.connect(db_path)
    try:
        con.execute("DROP TABLE actor_trust_policy_revocations")
        con.execute("DROP TABLE evidence_bundle_export_jobs")
        con.execute("DELETE FROM schema_migrations WHERE version >= 3")
        con.commit()
    finally:
        con.close()


def _snapshot(con, table):
    return sorted(
        con.execute(f"SELECT * FROM {table}").fetchall()
    )


def test_legacy_v2_upgrade_preserves_policies_and_audits(tmp_path):
    db_path = tmp_path / "legacy.db"
    url = f"sqlite:///{db_path.as_posix()}"
    from fastapi.testclient import TestClient

    app = create_app(Settings(database_url=url))
    with TestClient(app) as client:
        _world(client)
        policy = _create_policy(client, threshold=2)
    _downgrade_to_v2(db_path)

    con = sqlite3.connect(db_path)
    try:
        policies_before = _snapshot(con, "actor_trust_policies")
        audits_before = _snapshot(con, "audit_events")
        assert [row[0] for row in con.execute("SELECT version FROM schema_migrations")] == [1, 2]
    finally:
        con.close()

    # The upgrade adds only the revocation table; existing rows are untouched.
    upgraded = create_app(Settings(database_url=url))
    with TestClient(upgraded) as client:
        con = sqlite3.connect(db_path)
        try:
            assert _snapshot(con, "actor_trust_policies") == policies_before
            assert _snapshot(con, "audit_events") == audits_before
            assert [row[0] for row in con.execute("SELECT version FROM schema_migrations")] == [1, 2, 3, 4]
            assert "actor_trust_policy_revocations" in {
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        finally:
            con.close()

        # The upgraded database serves revocations and decisions consistently.
        created = _revoke(client, policy["id"], reason="post-upgrade")
        assert created["policy_id"] == policy["id"]
        decision = _get_decision(client, "claim", "clm_anything")
        assert decision.json()["reason"] == "policy_missing"


def test_failed_v3_migration_rolls_back_and_keeps_old_data(tmp_path, monkeypatch):
    db_path = tmp_path / "fail.db"
    url = f"sqlite:///{db_path.as_posix()}"
    from fastapi.testclient import TestClient

    app = create_app(Settings(database_url=url))
    with TestClient(app) as client:
        _world(client)
        _create_policy(client, threshold=2)
    _downgrade_to_v2(db_path)

    con = sqlite3.connect(db_path)
    try:
        policies_before = _snapshot(con, "actor_trust_policies")
        audits_before = _snapshot(con, "audit_events")
    finally:
        con.close()

    def _failing_up(cursor, engine, metadata):
        cursor.execute(
            "CREATE TABLE actor_trust_policy_revocations (seq INTEGER)"
        )
        raise RuntimeError("simulated migration failure")

    monkeypatch.setattr(
        migrations,
        "MIGRATIONS",
        migrations.MIGRATIONS[:2]
        + (migrations.Migration(version=3, up=_failing_up),),
    )
    engine = make_engine(url)
    with pytest.raises(RuntimeError, match="simulated migration failure"):
        migrations.run_migrations(engine)

    con = sqlite3.connect(db_path)
    try:
        # No half-finished version row and no partial table; the old data is
        # byte-for-byte intact.
        assert [row[0] for row in con.execute("SELECT version FROM schema_migrations")] == [1, 2]
        assert "actor_trust_policy_revocations" not in {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert _snapshot(con, "actor_trust_policies") == policies_before
        assert _snapshot(con, "audit_events") == audits_before
    finally:
        con.close()

    # A later, healthy startup migrates cleanly.
    monkeypatch.undo()
    recovered = create_app(Settings(database_url=url))
    with TestClient(recovered):
        pass
    con = sqlite3.connect(db_path)
    try:
        assert [row[0] for row in con.execute("SELECT version FROM schema_migrations")] == [1, 2, 3, 4]
        assert _snapshot(con, "actor_trust_policies") == policies_before
        assert _snapshot(con, "audit_events") == audits_before
    finally:
        con.close()
