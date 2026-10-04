"""Tests for attestation access-grant expiries.

Covers the two new protected routes and the protected-read boundary change:

* ``POST /v1/attestation-access-grants/{grant_id}/expiry`` -- the grant's
  attestation signer schedules one immutable UTC expiry; first creation is
  201 with the exact four public fields and a same-transaction
  ``attestation.access_grant_expiry_scheduled`` audit event; a retry for the
  same grant and instant is 200 with the original record and no new audit; a
  different instant is 409 ``attestation_access_grant_expiry_conflict``;
* ``GET /v1/attestation-access-grants/{grant_id}/expiry`` -- signer-only
  read of ``{grant_id, expires_at}`` where ``expires_at`` is null before an
  expiry is ever scheduled;
* unknown grant / non-signer / unauthenticated callers collapse into one
  opaque 404 on both routes, while malformed credentials stay 422;
* structurally bad bodies (malformed JSON, missing/extra/non-string field)
  are 422 ``grant_expiry_invalid`` and a non-future UTC instant is 422
  ``grant_expiry_not_future``; no failure writes anything;
* ``GET /v1/protected/attestations/{attestation_id}`` admits a grantee only
  while the grant is unrevoked and carries no expiry at or before now; the
  signer is never affected by a grant expiry;
* grants created before the feature (no expiry row) keep working unchanged;
* concurrent scheduling produces one row and one audit event, different
  instants race to a single write with the rest 409, and everything
  survives a restart.

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from sqlalchemy import select

from provenance.ids import attestation_access_grant_expiry_id
from provenance.models import (
    EVENT_ATTESTATION_ACCESS_GRANT_EXPIRY_SCHEDULED,
    AttestationAccessGrantExpiry,
    AuditEvent,
)
from tests.helpers import SEED_A, SEED_B
from tests.test_attestation_access_grants import (
    _get_protected,
    _grant_body,
    _post_grant,
    _signed_headers,
    _world,
)

SEED_C = b"test-ed25519-seed-c-00000000000000"[:32]


def _expiry_path(grant_id: str) -> str:
    return f"/v1/attestation-access-grants/{grant_id}/expiry"


def _instant(seconds: int) -> str:
    return (
        datetime.now(timezone.utc) + timedelta(seconds=seconds)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


def _future(seconds: int = 3600) -> str:
    return _instant(seconds)


def _past(seconds: int = 60) -> str:
    return _instant(-seconds)


def _expiry_body(expires_at: str) -> bytes:
    return json.dumps({"expires_at": expires_at}).encode("utf-8")


def _grant(client):
    attestation = _world(client)
    created = _post_grant(client, _grant_body(attestation["id"], "org-2"))
    assert created.status_code == 201, created.text
    return attestation, created.json()


def _post_expiry(
    client,
    grant_id,
    expires_at=None,
    *,
    body=None,
    actor="org-1",
    seed=SEED_A,
    **sign_kwargs,
):
    raw = body if body is not None else _expiry_body(
        expires_at if expires_at is not None else _future()
    )
    headers = {
        "Content-Type": "application/json",
        **_signed_headers(
            "POST", _expiry_path(grant_id), raw,
            actor=actor, seed=seed, **sign_kwargs,
        ),
    }
    return client.post(_expiry_path(grant_id), content=raw, headers=headers)


def _get_expiry(client, grant_id, *, actor="org-1", seed=SEED_A, **sign_kwargs):
    headers = _signed_headers(
        "GET", _expiry_path(grant_id), b"", actor=actor, seed=seed, **sign_kwargs
    )
    return client.get(_expiry_path(grant_id), headers=headers)


def _audit_count(session) -> int:
    return len(session.execute(select(AuditEvent)).scalars().all())


# --- First scheduling ----------------------------------------------------------


def test_first_expiry_returns_201_with_exact_public_fields(client):
    _, grant = _grant(client)
    expires_at = _future()
    resp = _post_expiry(client, grant["id"], expires_at)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == {"id", "grant_id", "expires_at", "created_at"}
    assert body["id"].startswith("aage_")
    assert len(body["id"]) == len("aage_") + 64
    assert body["grant_id"] == grant["id"]
    # The instant is echoed as strict UTC and names the scheduled instant.
    assert body["expires_at"].endswith(("Z", "+00:00"))
    assert datetime.fromisoformat(
        body["expires_at"].replace("Z", "+00:00")
    ) == datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    created = datetime.fromisoformat(body["created_at"])
    assert created.utcoffset().total_seconds() == 0


def test_expiry_id_is_the_stable_pair_derived_identifier(client):
    _, grant = _grant(client)
    expires_at = _future()
    body = _post_expiry(client, grant["id"], expires_at).json()
    parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    assert body["id"] == attestation_access_grant_expiry_id(grant["id"], parsed)


def test_expiry_and_audit_event_commit_in_one_transaction(client, db_session):
    _, grant = _grant(client)
    expires_at = _future()
    created = _post_expiry(client, grant["id"], expires_at).json()

    rows = db_session.execute(
        select(AttestationAccessGrantExpiry)
    ).scalars().all()
    assert [(r.id, r.grant_id) for r in rows] == [(created["id"], grant["id"])]
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type
            == EVENT_ATTESTATION_ACCESS_GRANT_EXPIRY_SCHEDULED
        )
    ).scalars().all()
    assert [e.resource_id for e in events] == [created["id"]]
    assert len(events) == 1


# --- Idempotency and the 409 conflict ------------------------------------------


def test_same_grant_and_instant_retry_returns_200_original_and_no_audit(
    client, db_session
):
    _, grant = _grant(client)
    expires_at = _future()
    first = _post_expiry(client, grant["id"], expires_at)
    assert first.status_code == 201
    events_after_first = _audit_count(db_session)

    for _ in range(3):
        retry = _post_expiry(client, grant["id"], expires_at)
        assert retry.status_code == 200, retry.text
        assert retry.json() == first.json()

    rows = db_session.execute(select(AttestationAccessGrantExpiry)).scalars().all()
    assert [r.id for r in rows] == [first.json()["id"]]
    assert _audit_count(db_session) == events_after_first


def test_same_instant_accepts_z_and_plus0000_spellings_as_one_record(client):
    _, grant = _grant(client)
    instant = datetime.now(timezone.utc) + timedelta(hours=1)
    zed = instant.strftime("%Y-%m-%dT%H:%M:%SZ")
    offset = instant.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    first = _post_expiry(client, grant["id"], zed)
    assert first.status_code == 201
    # The two spellings name the same UTC instant: this is an idempotent
    # retry of the original record, not a conflict.
    retry = _post_expiry(client, grant["id"], offset)
    assert retry.status_code == 200, retry.text
    assert retry.json()["id"] == first.json()["id"]


def test_different_instant_is_409_conflict_and_writes_nothing(client, db_session):
    attestation, grant = _grant(client)
    first_at = _future(3600)
    assert _post_expiry(client, grant["id"], first_at).status_code == 201
    events_after = _audit_count(db_session)

    other = _post_expiry(client, grant["id"], _future(7200))
    assert other.status_code == 409
    error = other.json()["error"]
    assert error["code"] == "attestation_access_grant_expiry_conflict"
    assert error["details"]["grant_id"] == grant["id"]

    # The original record is untouched and the loser added nothing.
    rows = db_session.execute(select(AttestationAccessGrantExpiry)).scalars().all()
    assert len(rows) == 1
    assert rows[0].id == attestation_access_grant_expiry_id(
        grant["id"],
        datetime.fromisoformat(first_at.replace("Z", "+00:00")),
    )
    assert _audit_count(db_session) == events_after


def test_conflict_does_not_change_the_get_state(client):
    _, grant = _grant(client)
    first_at = _future(3600)
    _post_expiry(client, grant["id"], first_at)
    assert _post_expiry(client, grant["id"], _future(7200)).status_code == 409
    state = _get_expiry(client, grant["id"])
    assert state.status_code == 200
    assert state.json()["expires_at"] == first_at


# --- GET state -----------------------------------------------------------------


def test_get_state_is_null_before_an_expiry_is_scheduled(client):
    _, grant = _grant(client)
    resp = _get_expiry(client, grant["id"])
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"grant_id": grant["id"], "expires_at": None}


def test_get_state_returns_the_scheduled_instant(client):
    _, grant = _grant(client)
    expires_at = _future()
    _post_expiry(client, grant["id"], expires_at)
    resp = _get_expiry(client, grant["id"])
    assert resp.status_code == 200
    assert resp.json() == {"grant_id": grant["id"], "expires_at": expires_at}


def test_get_state_writes_nothing(client, db_session):
    _, grant = _grant(client)
    before = _audit_count(db_session)
    assert _get_expiry(client, grant["id"]).status_code == 200
    assert _audit_count(db_session) == before
    assert (
        db_session.execute(select(AttestationAccessGrantExpiry)).scalars().all()
        == []
    )


# --- Opaque 404 boundary -------------------------------------------------------


def test_unknown_grant_is_404_on_post_and_get(client, db_session):
    _world(client)
    before = _audit_count(db_session)
    post = _post_expiry(client, "aag_ghost", _future())
    assert post.status_code == 404
    assert post.json()["error"]["code"] == "not_found"
    got = _get_expiry(client, "aag_ghost")
    assert got.status_code == 404
    assert got.json() == post.json()
    assert _audit_count(db_session) == before
    assert (
        db_session.execute(select(AttestationAccessGrantExpiry)).scalars().all()
        == []
    )


def test_non_signer_is_indistinguishable_from_unknown_grant(client, db_session):
    _, grant = _grant(client)
    before = _audit_count(db_session)
    # org-2 is the grantee and holds a current key; it is not the signer.
    post = _post_expiry(
        client, grant["id"], _future(), actor="org-2", seed=SEED_B
    )
    assert post.status_code == 404
    assert post.json()["error"]["code"] == "not_found"
    got = _get_expiry(client, grant["id"], actor="org-2", seed=SEED_B)
    assert got.status_code == 404
    # An unrelated actor (org-3) collapses into the same opaque 404.
    other = _get_expiry(client, grant["id"], actor="org-3", seed=SEED_C)
    assert other.status_code == 404
    assert other.json() == got.json()
    assert _audit_count(db_session) == before


def test_post_and_get_share_one_opaque_404_for_every_unauthorized_caller(client):
    _, grant = _grant(client)
    unknown_post = _post_expiry(client, "aag_ghost", _future())
    unauth_get = client.get(_expiry_path(grant["id"]))  # no credentials at all
    assert unknown_post.status_code == 404
    assert unauth_get.status_code == 404
    assert unknown_post.json() == unauth_get.json()


def test_get_with_malformed_credentials_is_422_not_404(client):
    _, grant = _grant(client)
    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    resp = _get_expiry(client, grant["id"], timestamp=stale)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert resp.json()["error"]["details"]["reason"] == (
        "timestamp_out_of_window"
    )


def test_post_with_malformed_credentials_is_422(client):
    _, grant = _grant(client)
    resp = _post_expiry(
        client, grant["id"], _future(), actor="org-1", seed=SEED_B
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )


# --- Body validation ------------------------------------------------------------


def test_post_rejects_malformed_and_ill_typed_bodies_as_invalid(client, db_session):
    _, grant = _grant(client)
    before = _audit_count(db_session)
    path = _expiry_path(grant["id"])

    def post_raw(raw):
        headers = {
            "Content-Type": "application/json",
            **_signed_headers("POST", path, raw),
        }
        return client.post(path, content=raw, headers=headers)

    cases = [
        b"{not json",
        b"[]",
        b"null",
        b'"2030-01-01T00:00:00Z"',
        b"{}",
        b'{"expires_at": null}',
        b'{"expires_at": 1234567890}',
        b'{"expires_at": true}',
        b'{"expires_at": "2030-01-01T00:00:00Z", "extra": 1}',
        b'{"other": "2030-01-01T00:00:00Z"}',
        b'{"expires_at": "2030-01-01 12:00:00Z"}',
        b'{"expires_at": "2030-01-01T12:00:00"}',
        b'{"expires_at": "2030-01-01T12:00:00+01:00"}',
        b'{"expires_at": "not-a-timestamp"}',
        b'{"expires_at": "2030-13-01T12:00:00Z"}',
    ]
    for raw in cases:
        resp = post_raw(raw)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["code"] == "validation_error", raw
        assert resp.json()["error"]["details"]["reason"] == (
            "grant_expiry_invalid"
        ), raw

    assert _audit_count(db_session) == before
    assert (
        db_session.execute(select(AttestationAccessGrantExpiry)).scalars().all()
        == []
    )


def test_post_rejects_a_non_future_instant_as_not_future(client, db_session):
    _, grant = _grant(client)
    before = _audit_count(db_session)
    for raw_value in (_past(60), _past(1), datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")):
        resp = _post_expiry(client, grant["id"], raw_value)
        assert resp.status_code == 422, raw_value
        assert resp.json()["error"]["details"]["reason"] == (
            "grant_expiry_not_future"
        ), raw_value
    assert _audit_count(db_session) == before
    assert (
        db_session.execute(select(AttestationAccessGrantExpiry)).scalars().all()
        == []
    )


def test_post_accepts_a_near_future_instant(client):
    _, grant = _grant(client)
    resp = _post_expiry(client, grant["id"], _future(5))
    assert resp.status_code == 201, resp.text


def test_post_failure_binds_signature_to_exact_body(client):
    _, grant = _grant(client)
    # Signature over one body, wire carries another -> unauthenticated 422,
    # never a 201/409/404.
    resp = _post_expiry(
        client, grant["id"], _future(),
        signed_body=b'{"expires_at":"2030-01-01T00:00:00Z"}',
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )


# --- Protected read honors the expiry -------------------------------------------


def test_grantee_read_allowed_before_expiry_and_denied_after(client):
    attestation, grant = _grant(client)
    # Schedule an expiry a few seconds out: the grant is live now.
    expires_at = _future(3)
    assert _post_expiry(client, grant["id"], expires_at).status_code == 201

    allowed = _get_protected(
        client, attestation["id"], actor="org-2", seed=SEED_B
    )
    assert allowed.status_code == 200, allowed.text

    # Move the scheduled instant into the past (the expiry row is immutable
    # through the API; the test rewinds it directly to exercise the
    # at-or-before-now boundary without sleeping).
    from provenance.models import AttestationAccessGrantExpiry as Expiry
    from provenance.time_utils import utc_now

    factory = client.app.state.session_factory
    session = factory()
    try:
        row = session.execute(
            select(Expiry).where(Expiry.grant_id == grant["id"])
        ).scalar_one()
        row.expires_at = utc_now() - timedelta(seconds=1)
        session.commit()
    finally:
        session.close()

    denied = _get_protected(
        client, attestation["id"], actor="org-2", seed=SEED_B
    )
    assert denied.status_code == 404
    assert denied.json()["error"]["code"] == "not_found"


def test_signer_read_is_unaffected_by_a_grant_expiry(client):
    attestation, grant = _grant(client)
    # A future expiry through the API, then rewind it into the past
    # directly (the expiry row is immutable through the API).
    assert _post_expiry(client, grant["id"], _future(3600)).status_code in (
        200, 201
    )
    from provenance.models import AttestationAccessGrantExpiry as Expiry
    from provenance.time_utils import utc_now

    factory = client.app.state.session_factory
    session = factory()
    try:
        row = session.execute(
            select(Expiry).where(Expiry.grant_id == grant["id"])
        ).scalar_one()
        row.expires_at = utc_now() - timedelta(seconds=5)
        session.commit()
    finally:
        session.close()

    # The grantee is now cut off ...
    assert _get_protected(
        client, attestation["id"], actor="org-2", seed=SEED_B
    ).status_code == 404
    # ... but the signer still reads: expiry never constrains the signer.
    signer = _get_protected(client, attestation["id"])
    assert signer.status_code == 200
    assert signer.json()["id"] == attestation["id"]


def test_a_grant_without_an_expiry_keeps_working_indefinitely(client):
    # No expiry row at all -- the state of every older grant.
    attestation, _grant_row = _grant(client)
    assert _get_protected(
        client, attestation["id"], actor="org-2", seed=SEED_B
    ).status_code == 200


def test_revocation_still_denies_even_with_a_later_expiry(client):
    attestation, grant = _grant(client)
    _post_expiry(client, grant["id"], _future(3600))
    headers, body = _revocation_headers(grant["id"])
    revoked = client.post(
        "/v1/attestation-access-grant-revocations",
        content=body,
        headers=headers,
    )
    assert revoked.status_code == 201, revoked.text
    assert _get_protected(
        client, attestation["id"], actor="org-2", seed=SEED_B
    ).status_code == 404
    # Signer still reads.
    assert _get_protected(client, attestation["id"]).status_code == 200


def _revocation_headers(grant_id):
    import base64
    import hashlib

    from provenance.access_signing import access_message_bytes
    from tests.helpers import ed25519_sign

    body = json.dumps({"grant_id": grant_id, "reason": "done"}).encode()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sig = base64.b64encode(
        ed25519_sign(
            SEED_A,
            access_message_bytes(
                "POST",
                "/v1/attestation-access-grant-revocations",
                ts,
                hashlib.sha256(body).hexdigest(),
            ),
        )
    ).decode()
    headers = {
        "Content-Type": "application/json",
        "X-PA": "org-1",
        "X-PT": ts,
        "X-PS": sig,
    }
    return headers, body


# --- Persistence and concurrency ------------------------------------------------


def test_expiry_persists_across_restart(tmp_db_url, file_client):
    attestation = _world(file_client)
    grant = _post_grant(
        file_client, _grant_body(attestation["id"], "org-2")
    ).json()
    expires_at = _future(3600)
    created = _post_expiry(file_client, grant["id"], expires_at).json()

    from provenance.app import create_app
    from provenance.config import Settings

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        state = _get_expiry(client, grant["id"])
        assert state.status_code == 200
        assert state.json() == {"grant_id": grant["id"], "expires_at": expires_at}

        con = sqlite3.connect(tmp_db_url.removeprefix("sqlite:///"))
        rows = con.execute(
            "SELECT id, grant_id, expires_at FROM attestation_access_grant_expiries"
        ).fetchall()
        audits = con.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = ?",
            (EVENT_ATTESTATION_ACCESS_GRANT_EXPIRY_SCHEDULED,),
        ).fetchone()[0]
        versions = con.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        con.close()
        assert [r[0] for r in rows] == [created["id"]]
        assert [r[1] for r in rows] == [grant["id"]]
        assert audits == 1
        assert [v[0] for v in versions] == [1, 2, 3, 4]


def test_concurrent_identical_expiries_yield_one_record_and_audit(
    tmp_db_url, file_app, file_client
):
    attestation = _world(file_client)
    grant = _post_grant(
        file_client, _grant_body(attestation["id"], "org-2")
    ).json()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

    from provenance import service

    factory = file_app.state.session_factory
    results: list[tuple[str, bool]] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        session = factory()
        try:
            barrier.wait()
            record, created = service.set_attestation_access_grant_expiry(
                session, grant["id"], expires_at, "org-1"
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
    assert {rid for rid, _ in results} == {results[0][0]}
    assert sum(1 for _, created in results if created) == 1

    audit_session = factory()
    try:
        rows = audit_session.execute(
            select(AttestationAccessGrantExpiry)
        ).scalars().all()
        events = audit_session.execute(
            select(AuditEvent).where(
                AuditEvent.event_type
                == EVENT_ATTESTATION_ACCESS_GRANT_EXPIRY_SCHEDULED
            )
        ).scalars().all()
        assert [r.id for r in rows] == [results[0][0]]
        assert [e.resource_id for e in events] == [results[0][0]]
    finally:
        audit_session.close()


def test_concurrent_different_expiries_one_writer_rest_conflict(
    tmp_db_url, file_app, file_client
):
    attestation = _world(file_client)
    grant = _post_grant(
        file_client, _grant_body(attestation["id"], "org-2")
    ).json()

    from provenance.errors import AttestationAccessGrantExpiryConflictError
    from provenance import service

    factory = file_app.state.session_factory
    outcomes: list[str] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    def worker(seconds: int) -> None:
        session = factory()
        try:
            barrier.wait()
            instant = datetime.now(timezone.utc) + timedelta(seconds=seconds)
            _record, created = service.set_attestation_access_grant_expiry(
                session, grant["id"], instant, "org-1"
            )
            outcomes.append("created" if created else "retry")
        except AttestationAccessGrantExpiryConflictError:
            outcomes.append("conflict")
        except Exception as exc:  # pragma: no cover - fails the test below
            errors.append(exc)
        finally:
            session.close()

    threads = [
        threading.Thread(target=worker, args=(3600 + i * 100,))
        for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert sorted(outcomes).count("created") == 1
    assert outcomes.count("conflict") == 3

    audit_session = factory()
    try:
        rows = audit_session.execute(
            select(AttestationAccessGrantExpiry)
        ).scalars().all()
        events = audit_session.execute(
            select(AuditEvent).where(
                AuditEvent.event_type
                == EVENT_ATTESTATION_ACCESS_GRANT_EXPIRY_SCHEDULED
            )
        ).scalars().all()
        assert len(rows) == 1
        assert len(events) == 1
        assert events[0].resource_id == rows[0].id
    finally:
        audit_session.close()


# --- Legacy migration ------------------------------------------------------------


def test_legacy_v1_database_gains_the_expiry_table(tmp_path):
    import sqlite3 as _sqlite3
    from provenance.app import create_app
    from provenance.config import Settings

    # A minimal legacy database: build the full current schema at an older
    # app version by creating the app once, then removing the expiry table
    # and its v2 ledger row to simulate a pre-v2 database.
    db_path = tmp_path / "legacy.db"
    url = f"sqlite:///{db_path.as_posix()}"
    with TestClient(create_app(Settings(database_url=url))):
        pass
    con = _sqlite3.connect(db_path)
    try:
        con.execute("DROP TABLE attestation_access_grant_expiries")
        con.execute("DELETE FROM schema_migrations WHERE version = 2")
        con.commit()
    finally:
        con.close()

    with TestClient(create_app(Settings(database_url=url))):
        # Startup migrates the simulated pre-v2 database to version 2.
        pass
    con = _sqlite3.connect(db_path)
    try:
        tables = {
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "attestation_access_grant_expiries" in tables
        assert [
            r[0]
            for r in con.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            )
        ] == [1, 2, 3, 4]
    finally:
        con.close()
