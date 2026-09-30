"""Tests for attestation access-grant expiry scheduling and enforcement.

Covers the two routes added on top of the existing immutable grants:

* ``POST /v1/attestation-access-grants/{grant_id}/expiry`` -- signer-only
  scheduling of one future ``expires_at`` (strict RFC 3339 UTC), returning
  201 on first creation (with an ``aage_`` id and an
  ``attestation.access_grant_expiry_scheduled`` audit event in the same
  transaction), 200 with the original record on a same-instant retry, and
  409 ``attestation_access_grant_expiry_conflict`` on a different instant;
* ``GET /v1/attestation-access-grants/{grant_id}/expiry`` -- signer-only
  read returning ``{"grant_id", "expires_at"}`` (null when unset), with an
  opaque 404 for an unknown grant or a non-signer and 422 for malformed
  credentials;

plus the enforcement boundary on
``GET /v1/protected/attestations/{attestation_id}``: a grantee keeps access
while the grant is unrevoked and carries no expiry or a still-future
expiry, and gets the same opaque 404 once the instant arrives; the signer
is unaffected by expiry, and revocation still denies. An unknown grant and
a non-signer caller are always the same opaque 404 on the new routes.

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures).
"""

from __future__ import annotations

import base64
import hashlib
import json
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
from provenance.time_utils import utc_now
from tests.helpers import SEED_A
from tests.test_attestation_access_grants import (
    GRANTS_PATH,
    SEED_B,
    SEED_C,
    _get_protected,
    _grant_body,
    _post_grant,
    _signed_headers,
    _world,
)

FUTURE_A = "2030-01-01T00:00:00Z"
FUTURE_B = "2031-06-15T12:30:00+00:00"
FUTURE_C = "2032-09-30T23:59:59.5Z"


def _expiry_path(grant_id: str) -> str:
    return f"{GRANTS_PATH}/{grant_id}/expiry"


def _grant(client):
    """Create org-1's attestation world and a grant to org-2."""
    attestation = _world(client)
    resp = _post_grant(client, _grant_body(attestation["id"], "org-2"))
    assert resp.status_code == 201, resp.text
    return attestation, resp.json()


def _post_expiry(
    client,
    grant_id,
    expires_at,
    *,
    actor="org-1",
    seed=None,
    raw_body=None,
    **sign_kwargs,
):
    path = _expiry_path(grant_id)
    body = raw_body if raw_body is not None else json.dumps(
        {"expires_at": expires_at}
    ).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **_signed_headers(
            "POST", path, body, actor=actor, seed=seed or SEED_A, **sign_kwargs
        ),
    }
    return client.post(path, content=body, headers=headers)


def _get_expiry(client, grant_id, *, actor="org-1", seed=None, **sign_kwargs):
    path = _expiry_path(grant_id)
    headers = _signed_headers(
        "GET", path, b"", actor=actor, seed=seed or SEED_A, **sign_kwargs
    )
    return client.get(path, headers=headers)


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


REVOCATIONS_PATH = "/v1/attestation-access-grant-revocations"


def _post_grant_revocation(client, grant_id, reason="offboarding"):
    body = json.dumps({"grant_id": grant_id, "reason": reason}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", REVOCATIONS_PATH, body),
    }
    return client.post(REVOCATIONS_PATH, content=body, headers=headers)


def _expiry_rows(session):
    return session.execute(
        select(AttestationAccessGrantExpiry)
    ).scalars().all()


def _force_expiry_instant(session, grant_id, when):
    """Move the grant's scheduled expiry to ``when`` directly (simulate time)."""
    updated = (
        session.query(AttestationAccessGrantExpiry)
        .filter(AttestationAccessGrantExpiry.grant_id == grant_id)
        .update({"expires_at": when}, synchronize_session=False)
    )
    assert updated == 1
    session.commit()


# --- First scheduling ----------------------------------------------------------


def test_first_expiry_returns_201_with_exact_public_fields(client, db_session):
    attestation, grant = _grant(client)
    resp = _post_expiry(client, grant["id"], FUTURE_A)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == {"id", "grant_id", "expires_at", "created_at"}
    assert body["id"].startswith("aage_")
    assert len(body["id"]) == len("aage_") + 64
    assert body["grant_id"] == grant["id"]
    assert body["expires_at"] == FUTURE_A
    expires = datetime.fromisoformat(body["expires_at"])
    assert expires.utcoffset().total_seconds() == 0
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))


def test_expiry_id_is_the_stable_pair_derived_identifier(client):
    _, grant = _grant(client)
    body = _post_expiry(client, grant["id"], FUTURE_A).json()
    assert body["id"] == attestation_access_grant_expiry_id(grant["id"], FUTURE_A)


def test_expiry_and_audit_event_commit_in_one_transaction(client, db_session):
    _, grant = _grant(client)
    created = _post_expiry(client, grant["id"], FUTURE_A).json()

    rows = _expiry_rows(db_session)
    assert [(r.id, r.grant_id) for r in rows] == [(created["id"], grant["id"])]
    assert rows[0].expires_at == datetime(2030, 1, 1, tzinfo=timezone.utc)
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type
            == EVENT_ATTESTATION_ACCESS_GRANT_EXPIRY_SCHEDULED
        )
    ).scalars().all()
    assert [e.resource_id for e in events] == [created["id"]]
    assert len(events) == 1


def test_fractional_and_offset_spellings_are_accepted_as_utc(client):
    _, grant = _grant(client)
    resp = _post_expiry(client, grant["id"], FUTURE_C)
    assert resp.status_code == 201, resp.text
    assert resp.json()["expires_at"].startswith("2032-09-30T23:59:59.5")


# --- Idempotency and conflict --------------------------------------------------


def test_same_instant_retry_returns_200_original_and_no_audit(client, db_session):
    _, grant = _grant(client)
    first = _post_expiry(client, grant["id"], FUTURE_A)
    assert first.status_code == 201
    events_after_first = _audit_count(db_session)

    for _ in range(3):
        retry = _post_expiry(client, grant["id"], FUTURE_A)
        assert retry.status_code == 200
        assert retry.json() == first.json()

    assert [r.id for r in _expiry_rows(db_session)] == [first.json()["id"]]
    assert _audit_count(db_session) == events_after_first
    scheduled = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type
            == EVENT_ATTESTATION_ACCESS_GRANT_EXPIRY_SCHEDULED
        )
    ).scalars().all()
    assert len(scheduled) == 1


def test_equivalent_utc_spelling_of_same_instant_returns_the_original(client):
    # "Z" and "+00:00" name the same instant; the retry resolves to the
    # stored record instead of a conflict.
    _, grant = _grant(client)
    first = _post_expiry(client, grant["id"], "2030-01-01T00:00:00Z")
    assert first.status_code == 201
    retry = _post_expiry(client, grant["id"], "2030-01-01T00:00:00+00:00")
    assert retry.status_code == 200
    assert retry.json()["id"] == first.json()["id"]


def test_different_instant_is_409_and_writes_nothing(client, db_session):
    _, grant = _grant(client)
    assert _post_expiry(client, grant["id"], FUTURE_A).status_code == 201
    events_before = _audit_count(db_session)

    conflict = _post_expiry(client, grant["id"], FUTURE_B)
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == (
        "attestation_access_grant_expiry_conflict"
    )
    assert conflict.json()["error"]["details"]["grant_id"] == grant["id"]
    assert [r.id for r in _expiry_rows(db_session)] == [
        attestation_access_grant_expiry_id(grant["id"], FUTURE_A)
    ]
    assert _audit_count(db_session) == events_before

    # The conflict did not replace or lock the original: the stored instant
    # still retries to its original record.
    again = _post_expiry(client, grant["id"], FUTURE_A)
    assert again.status_code == 200
    assert again.json()["expires_at"] == FUTURE_A


# --- Unknown grant / authorization on the write route --------------------------


def test_unknown_grant_is_opaque_404_for_the_signer(client, db_session):
    _world(client)
    events_before = _audit_count(db_session)
    resp = _post_expiry(client, "aag_ghost", FUTURE_A)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"
    assert _expiry_rows(db_session) == []
    assert _audit_count(db_session) == events_before


def test_non_signer_caller_gets_opaque_404_on_known_grant(client, db_session):
    _, grant = _grant(client)
    events_before = _audit_count(db_session)
    # The grantee itself is authenticated but is not the grant's signer.
    resp = _post_expiry(
        client, grant["id"], FUTURE_A, actor="org-2", seed=SEED_B
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"
    # A different, unrelated actor collapses into the identical response.
    other = _post_expiry(
        client, grant["id"], FUTURE_A, actor="org-3", seed=SEED_C
    )
    assert other.status_code == 404
    assert other.json() == resp.json()
    assert _expiry_rows(db_session) == []
    assert _audit_count(db_session) == events_before


def test_non_signer_on_unknown_grant_gets_the_same_opaque_404(client):
    _world(client)
    by_signer = _post_expiry(client, "aag_ghost", FUTURE_A)
    by_grantee = _post_expiry(
        client, "aag_ghost", FUTURE_A, actor="org-2", seed=SEED_B
    )
    assert by_signer.status_code == by_grantee.status_code == 404
    assert by_grantee.json() == by_signer.json()


def test_expiry_may_be_scheduled_on_a_revoked_grant(client):
    # Revocation does not delete the grant or strip the signer's ability to
    # schedule its expiry.
    attestation = _world(client)
    grant = _post_grant(client, _grant_body(attestation["id"], "org-2")).json()
    revoked = _post_grant_revocation(client, grant["id"])
    assert revoked.status_code == 201
    resp = _post_expiry(client, grant["id"], FUTURE_A)
    assert resp.status_code == 201, resp.text


# --- Credential and body validation on the write route -------------------------


def test_post_expiry_without_credentials_is_422(client, db_session):
    _, grant = _grant(client)
    events_before = _audit_count(db_session)
    body = json.dumps({"expires_at": FUTURE_A}).encode()
    resp = client.post(
        _expiry_path(grant["id"]),
        content=body,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "missing_credentials"
    assert _audit_count(db_session) == events_before


def test_post_expiry_wrong_signature_is_422(client):
    _, grant = _grant(client)
    resp = _post_expiry(
        client, grant["id"], FUTURE_A, actor="org-1", seed=SEED_B
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )


def test_post_expiry_malformed_credentials_are_422(client):
    _, grant = _grant(client)
    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert _post_expiry(client, grant["id"], FUTURE_A, timestamp=stale) \
        .status_code == 422
    raw = base64.b64encode(b"x" * 63).decode()
    body = json.dumps({"expires_at": FUTURE_A}).encode()
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", _expiry_path(grant["id"]), body),
    }
    headers["X-PS"] = raw
    resp = client.post(_expiry_path(grant["id"]), content=body, headers=headers)
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "invalid_signature"


def test_post_expiry_signature_binds_to_body_bytes(client, db_session):
    _, grant = _grant(client)
    events_before = _audit_count(db_session)
    resp = _post_expiry(
        client, grant["id"], FUTURE_A, signed_body=b'{"expires_at":"2031-01-01Z"}'
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )
    assert _audit_count(db_session) == events_before


def test_post_expiry_malformed_bodies_are_422_grant_expiry_invalid(client, db_session):
    _, grant = _grant(client)
    events_before = _audit_count(db_session)
    path = _expiry_path(grant["id"])

    def post_raw(raw):
        headers = {
            "Content-Type": "application/json",
            **_signed_headers("POST", path, raw),
        }
        return client.post(path, content=raw, headers=headers)

    cases = [
        b"",
        b"{not valid json",
        b"[]",
        b'"2030-01-01T00:00:00Z"',
        b"null",
        b"{}",
        b'{"expires_at": null}',
        b'{"expires_at": 1234567890}',
        b'{"expires_at": true}',
        b'{"expires_at": {"when": "2030-01-01T00:00:00Z"}}',
        b'{"expires_at": "   "}',
        b'{"expires_at": "2030-01-01"}',
        b'{"expires_at": "2030-01-01 00:00:00Z"}',
        b'{"expires_at": "2030-01-01T00:00:00"}',
        b'{"expires_at": "2030-01-01T00:00:00+01:00"}',
        b'{"expires_at": "2030-01-01t00:00:00z"}',
        b'{"expires_at": "not-a-timestamp"}',
        b'{"expires_at": "2030-13-01T00:00:00Z"}',
        b'{"expires_at": "2030-01-01T00:00:00Z", "nope": true}',
    ]
    for raw in cases:
        resp = post_raw(raw)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["code"] == "validation_error", raw
        assert resp.json()["error"]["details"]["reason"] == (
            "grant_expiry_invalid"
        ), raw

    assert _expiry_rows(db_session) == []
    assert _audit_count(db_session) == events_before


def test_post_expiry_past_and_present_instants_are_not_future(client, db_session):
    _, grant = _grant(client)
    events_before = _audit_count(db_session)
    past = (
        datetime.now(timezone.utc) - timedelta(seconds=10)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for raw in (past, now, "2020-01-01T00:00:00Z"):
        resp = _post_expiry(client, grant["id"], raw)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["details"]["reason"] == (
            "grant_expiry_not_future"
        ), raw
    assert _expiry_rows(db_session) == []
    assert _audit_count(db_session) == events_before


def test_post_expiry_a_few_seconds_in_the_future_is_accepted(client):
    _, grant = _grant(client)
    soon = (
        datetime.now(timezone.utc) + timedelta(seconds=10)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    resp = _post_expiry(client, grant["id"], soon)
    assert resp.status_code == 201, resp.text


def test_post_expiry_body_shape_is_validated_before_credentials(client, db_session):
    # Same precedence as the other protected write routes (FastAPI parses
    # the body before the handler authenticates): a malformed body reports
    # grant_expiry_invalid even when the request carries no credentials, and
    # writes nothing.
    _, grant = _grant(client)
    events_before = _audit_count(db_session)
    for raw in (b"{not json", b'{"nope": "2030-01-01T00:00:00Z"}', b"{}"):
        resp = client.post(
            _expiry_path(grant["id"]),
            content=raw,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["details"]["reason"] == (
            "grant_expiry_invalid"
        ), raw
    # A well-formed but non-future instant is only a semantic error, so
    # missing credentials take precedence there.
    resp = client.post(
        _expiry_path(grant["id"]),
        content=json.dumps({"expires_at": "2020-01-01T00:00:00Z"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "missing_credentials"
    assert _expiry_rows(db_session) == []
    assert _audit_count(db_session) == events_before


# --- GET -----------------------------------------------------------------------


def test_get_expiry_unset_returns_null(client, db_session):
    _, grant = _grant(client)
    events_before = _audit_count(db_session)
    resp = _get_expiry(client, grant["id"])
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"grant_id": grant["id"], "expires_at": None}
    # The read writes nothing.
    assert _audit_count(db_session) == events_before
    assert _expiry_rows(db_session) == []


def test_get_expiry_after_scheduling_returns_the_instant(client):
    _, grant = _grant(client)
    _post_expiry(client, grant["id"], FUTURE_B)
    resp = _get_expiry(client, grant["id"])
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"grant_id", "expires_at"}
    assert body["grant_id"] == grant["id"]
    parsed = datetime.fromisoformat(body["expires_at"])
    assert parsed == datetime(2031, 6, 15, 12, 30, tzinfo=timezone.utc)
    assert parsed.utcoffset().total_seconds() == 0


def test_get_expiry_unknown_grant_and_non_signer_are_opaque_404(client, db_session):
    attestation, grant = _grant(client)
    _post_expiry(client, grant["id"], FUTURE_A)
    events_before = _audit_count(db_session)

    unknown = _get_expiry(client, "aag_ghost")
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "not_found"

    # The grantee and an unrelated actor may not read the signer-only view.
    grantee = _get_expiry(client, grant["id"], actor="org-2", seed=SEED_B)
    stranger = _get_expiry(client, grant["id"], actor="org-3", seed=SEED_C)
    for resp in (grantee, stranger):
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "not_found"

    # Unknown-grant and non-signer are indistinguishable even after a set.
    assert grantee.json() == unknown.json()

    # No credentials at all is the same opaque 404 on a read.
    no_headers = client.get(_expiry_path(grant["id"]))
    assert no_headers.status_code == 404
    assert no_headers.json() == unknown.json()
    assert _audit_count(db_session) == events_before


def test_get_expiry_malformed_credentials_are_422(client):
    _, grant = _grant(client)
    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert _get_expiry(client, grant["id"], timestamp=stale).status_code == 422
    path = _expiry_path(grant["id"])
    headers = _signed_headers("GET", path, b"")
    headers["X-PS"] = base64.b64encode(b"x" * 63).decode()
    resp = client.get(path, headers=headers)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_get_expiry_well_formed_but_unverifiable_signature_is_404(client):
    _, grant = _grant(client)
    # SEED_B is not a current key of org-1: well-formed signature, but the
    # caller cannot be authenticated, which collapses into the opaque 404.
    resp = _get_expiry(client, grant["id"], actor="org-1", seed=SEED_B)
    assert resp.status_code == 404


# --- Enforcement on the protected read -----------------------------------------


def test_grantee_keeps_reading_with_a_still_future_expiry(client):
    attestation, grant = _grant(client)
    assert _post_expiry(client, grant["id"], FUTURE_A).status_code == 201
    resp = _get_protected(
        client, attestation["id"], actor="org-2", seed=SEED_B
    )
    assert resp.status_code == 200
    assert resp.json() == attestation


def test_grantee_is_denied_once_the_expiry_instant_arrives(client, db_session):
    attestation, grant = _grant(client)
    _post_expiry(client, grant["id"], FUTURE_A)

    _force_expiry_instant(
        db_session, grant["id"],
        datetime(2020, 1, 1, tzinfo=timezone.utc),
    )

    expired = _get_protected(
        client, attestation["id"], actor="org-2", seed=SEED_B
    )
    assert expired.status_code == 404
    assert expired.json()["error"]["code"] == "not_found"

    # The denial is the exact opaque 404 used for a never-granted stranger.
    never_granted = _get_protected(
        client, attestation["id"], actor="org-3", seed=SEED_C
    )
    assert never_granted.status_code == 404
    assert expired.json() == never_granted.json()

    # The signer is unaffected by the expiry.
    signer = _get_protected(client, attestation["id"])
    assert signer.status_code == 200
    assert signer.json() == attestation


def test_grant_expiring_exactly_now_is_denied(client, db_session):
    attestation, grant = _grant(client)
    _post_expiry(client, grant["id"], FUTURE_A)
    now = utc_now()
    _force_expiry_instant(db_session, grant["id"], now)
    resp = _get_protected(
        client, attestation["id"], actor="org-2", seed=SEED_B
    )
    assert resp.status_code == 404


def test_expiry_is_per_grant_other_grantees_and_proofs_are_unaffected(client):
    import hashlib

    from tests.test_attestation_access_grants import (
        _make_attestation,
        _make_claim,
    )

    attestation_a = _world(client)
    grant_a = _post_grant(
        client, _grant_body(attestation_a["id"], "org-2")
    ).json()
    # A second grantee on the same proof and a second proof for org-2.
    grant_to_org3 = _post_grant(
        client, _grant_body(attestation_a["id"], "org-3")
    )
    assert grant_to_org3.status_code == 201
    second_claim = _make_claim(
        client, "org-1", digest=hashlib.sha256(b"second-proof").hexdigest()
    )
    # org-1's SEED_A is reused on a distinct target (allowed: key discovery
    # is per-actor); build the second proof with the same seed.
    attestation_b = _make_attestation(
        client, "org-1",
        b"test-ed25519-seed-a-00000000000000"[:32],
        second_claim["id"],
    )
    grant_b = _post_grant(
        client, _grant_body(attestation_b["id"], "org-2")
    ).json()

    _post_expiry(client, grant_a["id"], FUTURE_A)
    # Simulate grant A's expiry arriving.
    app_session = client.app.state.session_factory()
    try:
        _force_expiry_instant(
            app_session, grant_a["id"],
            datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
    finally:
        app_session.close()

    # org-2 loses proof A ...
    assert _get_protected(
        client, attestation_a["id"], actor="org-2", seed=SEED_B
    ).status_code == 404
    # ... but keeps proof B, and org-3's independent grant on proof A still
    # authorizes.
    assert _get_protected(
        client, attestation_b["id"], actor="org-2", seed=SEED_B
    ).status_code == 200
    assert _get_protected(
        client, attestation_a["id"], actor="org-3", seed=SEED_C
    ).status_code == 200
    # The signer keeps both.
    assert _get_protected(client, attestation_a["id"]).status_code == 200
    assert _get_protected(client, attestation_b["id"]).status_code == 200


def test_revocation_still_denies_with_a_future_expiry_present(client):
    attestation, grant = _grant(client)
    _post_expiry(client, grant["id"], FUTURE_A)
    revoked = _post_grant_revocation(client, grant["id"])
    assert revoked.status_code == 201
    # Revocation denies even though the scheduled instant has not arrived.
    assert _get_protected(
        client, attestation["id"], actor="org-2", seed=SEED_B
    ).status_code == 404
    # Signer access is independent of both grant state and expiry.
    assert _get_protected(client, attestation["id"]).status_code == 200


def test_expired_grant_reads_write_nothing(client, db_session):
    attestation, grant = _grant(client)
    _post_expiry(client, grant["id"], FUTURE_A)
    db_session.commit()
    _force_expiry_instant(
        db_session, grant["id"],
        datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    events_before = _audit_count(db_session)
    for _ in range(3):
        resp = _get_protected(
            client, attestation["id"], actor="org-2", seed=SEED_B
        )
        assert resp.status_code == 404
    db_session.expire_all()
    assert _audit_count(db_session) == events_before
    assert len(_expiry_rows(db_session)) == 1


# --- Concurrency ---------------------------------------------------------------


def test_concurrent_different_expiries_yield_one_write_rest_are_409(
    tmp_db_url, file_app, file_client
):
    from provenance import service
    from provenance.errors import AttestationAccessGrantExpiryConflictError

    attestation = _world(file_client)
    grant = _post_grant(
        file_client, _grant_body(attestation["id"], "org-2")
    ).json()

    factory = file_app.state.session_factory
    instants = [
        (datetime(2030, 1, 1, tzinfo=timezone.utc), "2030-01-01T00:00:00Z"),
        (datetime(2031, 1, 1, tzinfo=timezone.utc), "2031-01-01T00:00:00Z"),
        (datetime(2032, 1, 1, tzinfo=timezone.utc), "2032-01-01T00:00:00Z"),
        (datetime(2033, 1, 1, tzinfo=timezone.utc), "2033-01-01T00:00:00Z"),
    ]
    outcomes: list[str] = []
    barrier = threading.Barrier(4)

    def worker(when, raw):
        session = factory()
        try:
            barrier.wait()
            try:
                _record, created = (
                    service.schedule_attestation_access_grant_expiry(
                        session, grant["id"], when, raw, "org-1"
                    )
                )
                outcomes.append("created" if created else "retry")
            except AttestationAccessGrantExpiryConflictError:
                outcomes.append("conflict")
        finally:
            session.close()

    threads = [
        threading.Thread(target=worker, args=(when, raw))
        for when, raw in instants
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(outcomes) == ["conflict", "conflict", "conflict", "created"]

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
        winner_raw = next(
            raw for when, raw in instants if when == rows[0].expires_at
        )
        assert rows[0].id == attestation_access_grant_expiry_id(
            grant["id"], winner_raw
        )
    finally:
        audit_session.close()


def test_concurrent_same_instant_yields_one_record_one_audit(
    tmp_db_url, file_app, file_client
):
    from provenance import service

    attestation = _world(file_client)
    grant = _post_grant(
        file_client, _grant_body(attestation["id"], "org-2")
    ).json()

    factory = file_app.state.session_factory
    when = datetime(2030, 1, 1, tzinfo=timezone.utc)
    raw = "2030-01-01T00:00:00Z"
    results: list[tuple[str, bool]] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    def worker():
        session = factory()
        try:
            barrier.wait()
            record, created = service.schedule_attestation_access_grant_expiry(
                session, grant["id"], when, raw, "org-1"
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


# --- Persistence across restart ------------------------------------------------


def test_expiry_persists_across_app_restart(tmp_db_url, file_client):
    import sqlite3

    attestation = _world(file_client)
    grant = _post_grant(
        file_client, _grant_body(attestation["id"], "org-2")
    ).json()
    created = _post_expiry(file_client, grant["id"], FUTURE_A).json()

    from provenance.app import create_app
    from provenance.config import Settings

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        view = _get_expiry(client, grant["id"])
        assert view.status_code == 200
        assert view.json() == {"grant_id": grant["id"], "expires_at": FUTURE_A}
        # Still-future expiry: the grantee keeps reading after the restart.
        resp = _get_protected(
            client, attestation["id"], actor="org-2", seed=SEED_B
        )
        assert resp.status_code == 200
        assert resp.json() == attestation
        # The signer still reads as well.
        assert _get_protected(client, attestation["id"]).status_code == 200

    con = sqlite3.connect(tmp_db_url.removeprefix("sqlite:///"))
    try:
        rows = con.execute(
            "SELECT id, grant_id, expires_at FROM attestation_access_grant_expiries"
        ).fetchall()
        audit = con.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = ?",
            (EVENT_ATTESTATION_ACCESS_GRANT_EXPIRY_SCHEDULED,),
        ).fetchone()[0]
    finally:
        con.close()
    assert len(rows) == 1
    assert rows[0][0] == created["id"]
    assert rows[0][1] == grant["id"]
    assert audit == 1


def test_expired_grant_stays_denied_across_restart(tmp_db_url, file_client):
    from provenance.app import create_app
    from provenance.config import Settings

    attestation = _world(file_client)
    grant = _post_grant(
        file_client, _grant_body(attestation["id"], "org-2")
    ).json()
    assert _post_expiry(file_client, grant["id"], FUTURE_A).status_code == 201

    # Move the stored instant into the past before restarting.
    con_path = tmp_db_url.removeprefix("sqlite:///")
    import sqlite3

    con = sqlite3.connect(con_path)
    try:
        con.execute(
            "UPDATE attestation_access_grant_expiries SET expires_at = ?",
            ("2020-01-01 00:00:00.000000",),
        )
        con.commit()
    finally:
        con.close()

    with TestClient(create_app(Settings(database_url=tmp_db_url))) as client:
        denied = _get_protected(
            client, attestation["id"], actor="org-2", seed=SEED_B
        )
        assert denied.status_code == 404
        assert _get_protected(client, attestation["id"]).status_code == 200
        view = _get_expiry(client, grant["id"])
        assert view.status_code == 200
        assert view.json()["expires_at"].startswith("2020-01-01T00:00:00")


# --- Legacy data: a grant without an expiry row never expires ------------------


def test_existing_unexpiring_grant_behavior_is_unchanged(client, db_session):
    # Baseline grants carry no expiry row: the grantee reads indefinitely,
    # and no expiry table data appears for them.
    attestation, grant = _grant(client)
    assert _expiry_rows(db_session) == []
    for _ in range(2):
        assert _get_protected(
            client, attestation["id"], actor="org-2", seed=SEED_B
        ).status_code == 200
    # Unset read remains null even after the grantee has read.
    assert _get_expiry(client, grant["id"]).json()["expires_at"] is None
