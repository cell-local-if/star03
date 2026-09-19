"""Tests for read-only attestation access grants and protected attestation reads.

Covers POST /v1/attestation-access-grants and
GET /v1/protected/attestations/{attestation_id}: the signed-access header
contract (X-PA / X-PT / X-PS), the compact UTF-8 access message with the
actual-body SHA-256, the 300-second timestamp window, authentication by
any non-revoked attestation key of the actor, signer-only grant creation,
201/200 pair idempotency with no audit on retry, independent immutable
pairs, single-transaction audit commit, the read-side 404 collapse
(missing / unauthenticated / unauthorized are indistinguishable), no
writes from reads or failed writes, a concurrent-identical race, and
persistence across restarts. All tests are deterministic and offline
(access signatures are produced by the stdlib test signer).
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from provenance.models import (
    EVENT_ATTESTATION_ACCESS_GRANTED,
    AttestationAccessGrant,
    AuditEvent,
)
from provenance.signing import access_message_bytes
from tests.helpers import (
    content_payload,
    create_actor,
    ed25519_public_key,
    ed25519_sign,
    SEED_A,
    SEED_B,
)

SEED_C = b"test-ed25519-seed-c-00000000000000"[:32]
SEED_D = b"test-ed25519-seed-d-00000000000000"[:32]
SEED_E = b"test-ed25519-seed-e-00000000000000"[:32]

GRANT_PATH = "/v1/attestation-access-grants"


# --- Setup helpers ----------------------------------------------------------


def _content(client, actor_id, tag):
    digest = hashlib.sha256(f"content-{actor_id}-{tag}".encode()).hexdigest()
    resp = client.post(
        "/v1/contents", json=content_payload(actor_id=actor_id, digest=digest)
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _claim(client, actor_id, tag):
    content = _content(client, actor_id, tag)
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content["id"],
            "actor_id": actor_id,
            "claim_type": "authorship",
            "payload": {"tag": tag},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _attest(client, actor_id, seed, claim_id):
    from provenance.signing import attestation_message_bytes

    signature = base64.b64encode(
        ed25519_sign(
            seed, attestation_message_bytes("claim", claim_id, actor_id)
        )
    ).decode("ascii")
    resp = client.post(
        "/v1/attestations",
        json={
            "target_type": "claim",
            "target_id": claim_id,
            "signer_actor_id": actor_id,
            "public_key": base64.b64encode(ed25519_public_key(seed)).decode(),
            "signature": signature,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _provision(client, actor_id, seed, tag="a", **actor_overrides):
    """Create an actor that holds one non-revoked attestation (can auth)."""
    create_actor(client, actor_id=actor_id, **actor_overrides)
    claim = _claim(client, actor_id, tag)
    return _attest(client, actor_id, seed, claim["id"])


def _revoke(client, attestation_id, revoker_actor_id="org-1", reason="rotated"):
    resp = client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": attestation_id,
            "revoker_actor_id": revoker_actor_id,
            "reason": reason,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _access_headers(
    seed,
    actor_id,
    method,
    path,
    body: bytes,
    *,
    timestamp: str | None = None,
    signature_message: bytes | None = None,
    raw_signature: bytes | None = None,
):
    """Build the three signed-access headers for the exact request bytes."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if signature_message is not None:
        message = signature_message
    else:
        message = access_message_bytes(
            method, path, timestamp, hashlib.sha256(body).hexdigest()
        )
    if raw_signature is not None:
        signature = raw_signature
    else:
        signature = ed25519_sign(seed, message)
    return {
        "X-PA": actor_id,
        "X-PT": timestamp,
        "X-PS": base64.b64encode(signature).decode("ascii"),
    }


def _grant_body(attestation_id, grantee_actor_id):
    payload = {
        "attestation_id": attestation_id,
        "grantee_actor_id": grantee_actor_id,
    }
    return payload, json.dumps(payload).encode("utf-8")


def _post_grant(
    client,
    attestation_id,
    grantee_actor_id,
    *,
    seed=SEED_A,
    actor_id="org-1",
    headers=None,
    body: bytes | None = None,
):
    _, canonical_body = _grant_body(attestation_id, grantee_actor_id)
    raw = body if body is not None else canonical_body
    hdrs = headers if headers is not None else _access_headers(
        seed, actor_id, "POST", GRANT_PATH, raw
    )
    hdrs = {**hdrs, "content-type": "application/json"}
    return client.post(GRANT_PATH, content=raw, headers=hdrs)


def _protected_path(attestation_id):
    return f"/v1/protected/attestations/{attestation_id}"


def _get_protected(
    client, attestation_id, *, seed=SEED_A, actor_id="org-1", headers=None
):
    path = _protected_path(attestation_id)
    hdrs = headers if headers is not None else _access_headers(
        seed, actor_id, "GET", path, b""
    )
    return client.get(path, headers=hdrs)


def _setup_world(client):
    """org-1 signs the protected proof; org-2 is a grantee; org-3 a stranger."""
    protected = _provision(client, "org-1", SEED_A, tag="protected")
    _provision(
        client, "org-2", SEED_B, tag="other",
        name="Grantee Org", type="organization",
    )
    _provision(
        client, "org-3", SEED_C, tag="stranger",
        name="Stranger Org", type="organization",
    )
    return protected


# --- Canonical message contract ---------------------------------------------


def test_access_message_is_compact_utf8_array_with_body_sha256():
    ts = "2026-09-20T12:00:00Z"
    body = json.dumps({"attestation_id": "att_x", "grantee_actor_id": "org-2"})
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    msg = access_message_bytes("POST", GRANT_PATH, ts, digest)
    assert msg == (
        f'["provenance-access-v1","POST","{GRANT_PATH}","{ts}","{digest}"]'
    ).encode("utf-8")
    assert b" " not in msg and b"provenance-access-v0" not in msg
    # Non-ASCII path elements are emitted unescaped as raw UTF-8.
    unicode_msg = access_message_bytes("GET", "/v1/x/证据", ts, digest)
    assert unicode_msg.endswith(
        f'"{ts}","{digest}"]'.encode("utf-8")
    )
    assert "证据".encode("utf-8") in unicode_msg
    assert b"\\u" not in unicode_msg


def test_empty_body_uses_sha256_of_zero_bytes():
    msg = access_message_bytes("GET", "/v1/p", "t", hashlib.sha256(b"").hexdigest())
    assert (
        hashlib.sha256(b"").hexdigest()
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert b"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" in msg


# --- Grant creation -----------------------------------------------------------


def test_signer_creates_grant_returns_201_public_fields(client):
    attestation = _setup_world(client)
    resp = _post_grant(client, attestation["id"], "org-2")
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == {
        "id",
        "attestation_id",
        "grantee_actor_id",
        "created_at",
    }
    assert body["id"].startswith("aag_")
    assert len(body["id"]) == len("aag_") + 64
    assert body["attestation_id"] == attestation["id"]
    assert body["grantee_actor_id"] == "org-2"
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))
    # The grant view carries no key or signature material.
    assert "public_key" not in body and "signature" not in body


def test_grant_id_is_deterministic_for_the_pair(client):
    attestation = _setup_world(client)
    created = _post_grant(client, attestation["id"], "org-2").json()
    material = json.dumps(
        ["attestation_access_grant", attestation["id"], "org-2"],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert created["id"] == "aag_" + hashlib.sha256(material).hexdigest()


def test_grant_audit_event_commits_with_the_row(client, db_session):
    attestation = _setup_world(client)
    created = _post_grant(client, attestation["id"], "org-2").json()

    rows = db_session.execute(select(AttestationAccessGrant)).scalars().all()
    assert [r.id for r in rows] == [created["id"]]
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ATTESTATION_ACCESS_GRANTED
        )
    ).scalars().all()
    assert [e.resource_id for e in events] == [created["id"]]
    assert events[0].created_at.utcoffset().total_seconds() == 0


def test_pretty_printed_body_verifies_via_actual_bytes_hash(client):
    # The body digest is of the actual transmitted bytes, not a
    # re-serialization: a pretty-printed body with a matching signature works.
    attestation = _setup_world(client)
    payload, _ = _grant_body(attestation["id"], "org-2")
    pretty = json.dumps(payload, indent=2).encode("utf-8")
    headers = _access_headers(SEED_A, "org-1", "POST", GRANT_PATH, pretty)
    resp = _post_grant(
        client, attestation["id"], "org-2", headers=headers, body=pretty
    )
    assert resp.status_code == 201, resp.text


# --- Idempotency and independence ---------------------------------------------


def test_same_pair_retry_returns_200_unchanged_and_no_new_audit(
    client, db_session
):
    attestation = _setup_world(client)
    first = _post_grant(client, attestation["id"], "org-2")
    assert first.status_code == 201
    first_body = first.json()
    events_after_create = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    for _ in range(3):
        repeat = _post_grant(client, attestation["id"], "org-2")
        assert repeat.status_code == 200
        assert repeat.json() == first_body

    rows = db_session.execute(select(AttestationAccessGrant)).scalars().all()
    assert [r.id for r in rows] == [first_body["id"]]
    grants = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ATTESTATION_ACCESS_GRANTED
        )
    ).scalars().all()
    assert len(grants) == 1
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_after_create
    )


def test_distinct_grantees_form_independent_immutable_grants(client):
    attestation = _setup_world(client)
    to_org2 = _post_grant(client, attestation["id"], "org-2")
    to_org3 = _post_grant(client, attestation["id"], "org-3")
    assert to_org2.status_code == to_org3.status_code == 201
    assert to_org2.json()["id"] != to_org3.json()["id"]

    # Retries never mutate or replace either record.
    assert _post_grant(client, attestation["id"], "org-2").json() == to_org2.json()
    assert _post_grant(client, attestation["id"], "org-3").json() == to_org3.json()


# --- Write authorization boundary ---------------------------------------------


def test_non_signer_cannot_create_grant_even_when_authenticated(client):
    attestation = _setup_world(client)
    # org-2 authenticates with its own valid, non-revoked key but does not
    # own the attestation.
    resp = _post_grant(
        client, attestation["id"], "org-3",
        seed=SEED_B, actor_id="org-2",
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert resp.json()["error"]["details"]["reason"] == "not_signer"


def test_grant_for_unknown_attestation_is_422(client):
    _setup_world(client)
    resp = _post_grant(client, "att_ghost", "org-2")
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "unknown_attestation"


def test_grant_for_unknown_grantee_actor_is_422(client):
    attestation = _setup_world(client)
    resp = _post_grant(client, attestation["id"], "ghost")
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "unknown_grantee_actor"


def test_failed_writes_persist_nothing(client, db_session):
    attestation = _setup_world(client)
    events_before = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    no_headers = client.post(
        GRANT_PATH,
        content=json.dumps(_grant_body(attestation["id"], "org-2")[0]),
        headers={"content-type": "application/json"},
    )
    not_signer = _post_grant(
        client, attestation["id"], "org-3", seed=SEED_B, actor_id="org-2"
    )
    unknown_att = _post_grant(client, "att_ghost", "org-2")
    unknown_actor = _post_grant(client, attestation["id"], "ghost")

    assert [r.status_code for r in (no_headers, not_signer, unknown_att,
                                    unknown_actor)] == [422, 422, 422, 422]
    assert db_session.execute(select(AttestationAccessGrant)).scalars().all() == []
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )


def test_reject_malformed_and_undeclared_body_fields(client):
    attestation = _setup_world(client)
    base, raw = _grant_body(attestation["id"], "org-2")

    # Blank identifiers.
    for field in ("attestation_id", "grantee_actor_id"):
        payload = dict(base)
        payload[field] = "   "
        body = json.dumps(payload).encode()
        resp = _post_grant(
            client, attestation["id"], "org-2", body=body,
            headers=_access_headers(SEED_A, "org-1", "POST", GRANT_PATH, body),
        )
        assert resp.status_code == 422, field

    # Undeclared field.
    payload = dict(base)
    payload["unexpected"] = "value"
    body = json.dumps(payload).encode()
    resp = _post_grant(
        client, attestation["id"], "org-2", body=body,
        headers=_access_headers(SEED_A, "org-1", "POST", GRANT_PATH, body),
    )
    assert resp.status_code == 422
    assert any(
        "unexpected" in issue["loc"]
        for issue in resp.json()["error"]["details"]["issues"]
    )

    # Missing field.
    body = json.dumps({"attestation_id": attestation["id"]}).encode()
    resp = _post_grant(
        client, attestation["id"], "org-2", body=body,
        headers=_access_headers(SEED_A, "org-1", "POST", GRANT_PATH, body),
    )
    assert resp.status_code == 422


# --- Access-header validation on the write path (all 422) ----------------------


def test_write_requires_all_three_headers(client):
    attestation = _setup_world(client)
    full = _access_headers(
        SEED_A, "org-1", "POST", GRANT_PATH,
        _grant_body(attestation["id"], "org-2")[1],
    )
    for missing in ("X-PA", "X-PT", "X-PS"):
        partial = {k: v for k, v in full.items() if k != missing}
        resp = _post_grant(
            client, attestation["id"], "org-2", headers=partial
        )
        assert resp.status_code == 422, missing
        assert resp.json()["error"]["details"]["reason"] == "missing_access_headers"

    # Blank values count as missing.
    for name in ("X-PA", "X-PT", "X-PS"):
        blank = dict(full)
        blank[name] = "   "
        resp = _post_grant(
            client, attestation["id"], "org-2", headers=blank
        )
        assert resp.status_code == 422, name


def test_repeated_access_headers_are_rejected_on_write(client):
    attestation = _setup_world(client)
    body = _grant_body(attestation["id"], "org-2")[1]
    base = _access_headers(SEED_A, "org-1", "POST", GRANT_PATH, body)
    headers = [
        ("X-PA", base["X-PA"]), ("X-PA", base["X-PA"]),
        ("X-PT", base["X-PT"]),
        ("X-PS", base["X-PS"]),
        ("content-type", "application/json"),
    ]
    resp = client.post(GRANT_PATH, content=body, headers=headers)
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "missing_access_headers"


def test_timestamp_window_is_enforced_on_write(client):
    attestation = _setup_world(client)
    # Distinct grantee actors so the in-window cases are genuine first
    # creations (201); a grantee only needs an actor row, not a key.
    create_actor(client, actor_id="org-in-past", name="P", type="organization")
    create_actor(client, actor_id="org-in-future", name="F", type="organization")

    def at(offset):
        return (
            datetime.now(timezone.utc) + timedelta(seconds=offset)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

    cases = (
        (-295, "org-in-past", 201),   # just inside the window, past
        (-301, "org-2", 422),         # just outside, past
        (295, "org-in-future", 201),  # just inside, future
        (301, "org-2", 422),          # just outside, future
    )
    for offset, grantee, expected in cases:
        body = _grant_body(attestation["id"], grantee)[1]
        headers = _access_headers(
            SEED_A, "org-1", "POST", GRANT_PATH, body, timestamp=at(offset)
        )
        resp = _post_grant(
            client, attestation["id"], grantee, headers=headers, body=body
        )
        assert resp.status_code == expected, offset


def test_malformed_timestamps_are_422_on_write(client):
    attestation = _setup_world(client)
    bad_timestamps = [
        "not-a-timestamp",
        "2026-09-20",                       # date only
        "2026-09-20 12:00:00",              # space separator
        "2026-09-20T12:00:00",              # naive, no UTC designator
        "2026-09-20T12:00:00+05:00",        # non-UTC offset
        "2026-02-30T12:00:00Z",             # impossible calendar date
        "2026-9-20T12:00:00Z",              # non-padded digits
    ]
    for ts in bad_timestamps:
        body = _grant_body(attestation["id"], "org-2")[1]
        headers = _access_headers(
            SEED_A, "org-1", "POST", GRANT_PATH, body, timestamp=ts
        )
        resp = _post_grant(
            client, attestation["id"], "org-2", headers=headers, body=body
        )
        assert resp.status_code == 422, ts
        assert resp.json()["error"]["details"]["reason"] == "invalid_timestamp"


def test_rfc3339_offset_and_fractional_forms_are_accepted(client):
    attestation = _setup_world(client)
    for ts, grantee in (
        (datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"), "org-2"),
        (datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"), "org-3"),
    ):
        body = _grant_body(attestation["id"], grantee)[1]
        headers = _access_headers(
            SEED_A, "org-1", "POST", GRANT_PATH, body, timestamp=ts
        )
        resp = _post_grant(
            client, attestation["id"], grantee, headers=headers, body=body
        )
        assert resp.status_code == 201, ts


def test_signature_must_cover_the_exact_request_on_write(client):
    attestation = _setup_world(client)
    body = _grant_body(attestation["id"], "org-2")[1]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    wrong_messages = [
        access_message_bytes("GET", GRANT_PATH, ts, hashlib.sha256(body).hexdigest()),
        access_message_bytes("POST", "/v1/attestation-access-grant", ts,
                             hashlib.sha256(body).hexdigest()),
        access_message_bytes("POST", GRANT_PATH, "2026-09-20T12:00:00Z",
                             hashlib.sha256(body).hexdigest()),
        access_message_bytes("POST", GRANT_PATH, ts, hashlib.sha256(b"").hexdigest()),
        json.dumps(
            ["provenance-access-v0", "POST", GRANT_PATH, ts,
             hashlib.sha256(body).hexdigest()],
            separators=(",", ":"),
        ).encode("utf-8"),
    ]
    for message in wrong_messages:
        headers = _access_headers(
            SEED_A, "org-1", "POST", GRANT_PATH, body,
            timestamp=ts, signature_message=message,
        )
        resp = _post_grant(
            client, attestation["id"], "org-2", headers=headers, body=body
        )
        assert resp.status_code == 422, message
        assert (
            resp.json()["error"]["details"]["reason"]
            == "signature_verification_failed"
        )

    # The signed hash is of the actual body: sending different bytes fails.
    headers = _access_headers(
        SEED_A, "org-1", "POST", GRANT_PATH,
        _grant_body(attestation["id"], "org-3")[1],
    )
    tampered = _post_grant(
        client, attestation["id"], "org-2", headers=headers
    )
    assert tampered.status_code == 422


def test_wrong_key_and_actor_mismatch_fail_on_write(client):
    attestation = _setup_world(client)

    # Signature by org-2's key, claimed as org-1.
    resp = _post_grant(
        client, attestation["id"], "org-2", seed=SEED_B, actor_id="org-1"
    )
    assert resp.status_code == 422

    # org-2's own key under its own name authenticates but is not signer.
    resp = _post_grant(
        client, attestation["id"], "org-2", seed=SEED_B, actor_id="org-2"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "not_signer"


def test_malformed_signature_encoding_is_422_on_write(client):
    attestation = _setup_world(client)
    body = _grant_body(attestation["id"], "org-2")[1]
    headers = _access_headers(SEED_A, "org-1", "POST", GRANT_PATH, body)
    for bad in (
        "@@@ not base64",
        base64.b64encode(b"s" * 63).decode(),
        base64.b64encode(b"s" * 65).decode(),
        base64.urlsafe_b64encode(b"\xff" * 64).decode(),
    ):
        headers["X-PS"] = bad
        resp = _post_grant(
            client, attestation["id"], "org-2", headers=dict(headers), body=body
        )
        assert resp.status_code == 422, bad
        assert resp.json()["error"]["details"]["reason"] == "invalid_signature"


def test_actor_with_no_attestation_cannot_authenticate_write(client):
    create_actor(client, actor_id="org-nokeys", name="Keyless", type="organization")
    attestation = _setup_world(client)
    body = _grant_body(attestation["id"], "org-2")[1]
    headers = _access_headers(SEED_D, "org-nokeys", "POST", GRANT_PATH, body)
    resp = _post_grant(
        client, attestation["id"], "org-2", headers=headers, body=body
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "signature_verification_failed"


# --- Revocation semantics ------------------------------------------------------


def test_revoked_key_no_longer_authenticates_write(client, db_session):
    attestation = _setup_world(client)
    # Revoking the proof revokes the key bound to it: org-1 now has no
    # non-revoked attestation key.
    _revoke(client, attestation["id"])
    resp = _post_grant(client, attestation["id"], "org-2")
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "signature_verification_failed"

    # A second, non-revoked attestation under a new key restores access.
    second = _attest(client, "org-1", SEED_D, _claim(client, "org-1", "second")["id"])
    assert _post_grant(
        client, attestation["id"], "org-2", seed=SEED_D, actor_id="org-1"
    ).status_code == 201
    # The revoked key still fails even though another key works.
    assert _post_grant(client, attestation["id"], "org-3").status_code == 422
    # Revoking the second key removes the remaining credential.
    _revoke(client, second["id"])
    assert _post_grant(
        client, attestation["id"], "org-3", seed=SEED_D
    ).status_code == 422


def test_one_revoked_key_among_several_does_not_block_the_other(client):
    attestation = _setup_world(client)
    # SEED_A stays live on the protected proof; SEED_D is attested and then
    # revoked. Authentication under SEED_A must keep working.
    revoked_att = _attest(
        client, "org-1", SEED_D, _claim(client, "org-1", "rotated")["id"]
    )
    _revoke(client, revoked_att["id"])
    assert _post_grant(client, attestation["id"], "org-2").status_code == 201


# --- Protected read ------------------------------------------------------------


def test_signer_reads_protected_attestation_public_view(client):
    attestation = _setup_world(client)
    _post_grant(client, attestation["id"], "org-2")

    resp = _get_protected(client, attestation["id"])
    assert resp.status_code == 200
    # The protected view is exactly the existing attestation public view.
    public = client.get(f"/v1/attestations/{attestation['id']}")
    assert resp.json() == public.json() == attestation
    assert "signature" not in resp.json()


def test_grantee_reads_protected_attestation(client):
    attestation = _setup_world(client)
    _post_grant(client, attestation["id"], "org-2")

    resp = _get_protected(
        client, attestation["id"], seed=SEED_B, actor_id="org-2"
    )
    assert resp.status_code == 200
    assert resp.json()["id"] == attestation["id"]


def test_read_is_404_for_missing_unauthenticated_and_unauthorized(client):
    attestation = _setup_world(client)
    _post_grant(client, attestation["id"], "org-2")
    path = _protected_path(attestation["id"])

    cases: list[tuple[str, object]] = [
        ("no headers at all", client.get(path)),
        (
            "stranger actor with valid auth",
            _get_protected(client, attestation["id"], seed=SEED_C, actor_id="org-3"),
        ),
        (
            "unknown actor claimed in X-PA",
            _get_protected(client, attestation["id"], seed=SEED_A, actor_id="ghost"),
        ),
        (
            "bad signature",
            _get_protected(
                client, attestation["id"], seed=SEED_B, actor_id="org-1"
            ),
        ),
        (
            "stale timestamp",
            _get_protected(
                client,
                attestation["id"],
                headers=_access_headers(
                    SEED_A, "org-1", "GET", path, b"",
                    timestamp=(
                        datetime.now(timezone.utc) - timedelta(seconds=301)
                    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                ),
            ),
        ),
        (
            "future timestamp",
            _get_protected(
                client,
                attestation["id"],
                headers=_access_headers(
                    SEED_A, "org-1", "GET", path, b"",
                    timestamp=(
                        datetime.now(timezone.utc) + timedelta(seconds=301)
                    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                ),
            ),
        ),
        (
            "malformed timestamp",
            _get_protected(
                client,
                attestation["id"],
                headers=_access_headers(
                    SEED_A, "org-1", "GET", path, b"", timestamp="yesterday"
                ),
            ),
        ),
        (
            "malformed signature encoding",
            _get_protected(
                client,
                attestation["id"],
                headers={
                    **_access_headers(SEED_A, "org-1", "GET", path, b""),
                    "X-PS": "not-base64!",
                },
            ),
        ),
        (
            "signature over non-empty body on a GET",
            _get_protected(
                client,
                attestation["id"],
                headers=_access_headers(
                    SEED_A, "org-1", "GET", path, b"unexpected"
                ),
            ),
        ),
        (
            "repeated header",
            client.get(
                path,
                headers=[
                    ("X-PA", "org-1"), ("X-PA", "org-1"),
                    (
                        "X-PT",
                        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    ),
                    ("X-PS", base64.b64encode(b"s" * 64).decode()),
                ],
            ),
        ),
    ]
    for label, resp in cases:
        assert resp.status_code == 404, label
        assert resp.json()["error"]["code"] == "attestation_not_found", label
        assert resp.json()["error"]["details"] == {
            "attestation_id": attestation["id"]
        }, label

    # A well-authenticated signer reading a missing attestation also 404s,
    # with the unknown id in the details.
    missing = _get_protected(client, "att_ghost")
    assert missing.status_code == 404
    assert missing.json()["error"]["details"] == {"attestation_id": "att_ghost"}


def test_revoked_grantee_key_loses_read_access(client):
    attestation = _setup_world(client)
    _post_grant(client, attestation["id"], "org-2")
    assert _get_protected(
        client, attestation["id"], seed=SEED_B, actor_id="org-2"
    ).status_code == 200

    # Revoke org-2's provisioned (and only) attestation: its key no longer
    # authenticates, so the protected read collapses to 404.
    org2_att = next(
        a
        for a in client.get("/v1/attestations").json()["items"]
        if a["signer_actor_id"] == "org-2"
    )
    _revoke(client, org2_att["id"])
    assert _get_protected(
        client, attestation["id"], seed=SEED_B, actor_id="org-2"
    ).status_code == 404

    # The grant itself is untouched: a new non-revoked key for the same
    # grantee restores the read.
    _attest(client, "org-2", SEED_E, _claim(client, "org-2", "newkey")["id"])
    assert _get_protected(
        client, attestation["id"], seed=SEED_E, actor_id="org-2"
    ).status_code == 200


def test_grant_to_a_third_party_does_not_authorize_others(client):
    attestation = _setup_world(client)
    _post_grant(client, attestation["id"], "org-2")
    # org-3 authenticates fine but holds no grant.
    assert _get_protected(
        client, attestation["id"], seed=SEED_C, actor_id="org-3"
    ).status_code == 404
    # Granting org-2 does not let org-2 act as a delegate granter.
    assert _post_grant(
        client, attestation["id"], "org-3", seed=SEED_B, actor_id="org-2"
    ).status_code == 422


# --- Read-only guarantees ------------------------------------------------------


def test_protected_reads_write_no_resource_or_audit_rows(client, db_session):
    attestation = _setup_world(client)
    _post_grant(client, attestation["id"], "org-2")
    grants_before = len(
        db_session.execute(select(AttestationAccessGrant)).scalars().all()
    )
    events_before = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    _get_protected(client, attestation["id"])
    _get_protected(client, attestation["id"], seed=SEED_B, actor_id="org-2")
    _get_protected(client, attestation["id"], seed=SEED_C, actor_id="org-3")
    _get_protected(client, "att_ghost")
    client.get(_protected_path(attestation["id"]))

    assert len(
        db_session.execute(select(AttestationAccessGrant)).scalars().all()
    ) == grants_before
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )


# --- Concurrent race -----------------------------------------------------------


def test_concurrent_identical_grants_yield_one_row_and_audit(
    tmp_db_url, file_app, file_client
):
    attestation = _setup_world(file_client)

    from provenance.schemas import AttestationAccessGrantCreate
    from provenance import service

    factory = file_app.state.session_factory
    request = AttestationAccessGrantCreate(
        attestation_id=attestation["id"], grantee_actor_id="org-2"
    )
    results: list[tuple[str, bool]] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        session = factory()
        try:
            barrier.wait()
            grant, created = service.create_attestation_access_grant(
                session, request, "org-1"
            )
            results.append((grant.id, created))
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
    assert {gid for gid, _ in results} == {results[0][0]}
    assert sum(1 for _, created in results if created) == 1

    audit_session = factory()
    try:
        rows = audit_session.execute(
            select(AttestationAccessGrant)
        ).scalars().all()
        events = audit_session.execute(
            select(AuditEvent).where(
                AuditEvent.event_type == EVENT_ATTESTATION_ACCESS_GRANTED
            )
        ).scalars().all()
        assert [r.id for r in rows] == [results[0][0]]
        assert [e.resource_id for e in events] == [results[0][0]]
    finally:
        audit_session.close()


# --- Persistence across restarts ------------------------------------------------


def test_grants_and_protected_reads_persist_across_restart(
    tmp_db_url, file_client
):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    attestation = _setup_world(file_client)
    grant_payload = json.dumps(
        {"attestation_id": attestation["id"], "grantee_actor_id": "org-2"}
    ).encode()
    created = file_client.post(
        GRANT_PATH,
        content=grant_payload,
        headers={
            **_access_headers(
                SEED_A, "org-1", "POST", GRANT_PATH, grant_payload
            ),
            "content-type": "application/json",
        },
    ).json()

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        # Signer and grantee both still read after restart.
        signer = _get_protected(client, attestation["id"])
        grantee = _get_protected(
            client, attestation["id"], seed=SEED_B, actor_id="org-2"
        )
        assert signer.status_code == grantee.status_code == 200
        assert signer.json()["id"] == attestation["id"]

        import sqlite3

        path = tmp_db_url.removeprefix("sqlite:///")
        con = sqlite3.connect(path)
        audit_count = con.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = ?",
            (EVENT_ATTESTATION_ACCESS_GRANTED,),
        ).fetchone()[0]
        grant_ids = {
            row[0]
            for row in con.execute("SELECT id FROM attestation_access_grants")
        }
        con.close()
        assert audit_count == 1
        assert grant_ids == {created["id"]}
