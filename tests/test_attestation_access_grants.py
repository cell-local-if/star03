"""Tests for read-only proof access grants and protected attestation reads.

Covers ``POST /v1/attestation-access-grants`` and
``GET /v1/protected/attestations/{attestation_id}``, including:

* the ``X-PA``/``X-PT``/``X-PS`` signed-header contract (compact UTF-8
  array, RFC 3339 UTC timestamps, 300-second window, SHA-256 of the actual
  body with zero bytes for an empty body);
* authentication against any non-revoked attestation public key of the
  actor, including the revocation boundary and multiple keys;
* signer-only grant creation, 201/200 idempotency for the same
  ``(attestation_id, grantee_actor_id)`` pair, independent immutable
  records for distinct pairs, and single-transaction audit commit;
* the read authorization boundary (signer or grantee) with a single
  opaque 404 for a missing target, an unauthenticated caller, and an
  unauthorized caller -- while malformed credentials stay 422;
* no resource or audit writes from any read or failed request.

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures).
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from provenance.access_signing import access_message_bytes
from provenance.ids import attestation_access_grant_id
from provenance.models import (
    EVENT_ATTESTATION_ACCESS_GRANTED,
    Attestation,
    AttestationAccessGrant,
    AuditEvent,
)
from provenance.signing import attestation_message_bytes
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    content_payload,
    create_actor,
    ed25519_public_key,
    ed25519_sign,
    SEED_A,
    SEED_B,
)

GRANTS_PATH = "/v1/attestation-access-grants"
SEED_C = b"test-ed25519-seed-c-00000000000000"[:32]


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


def _world(client):
    """Create three actors, each holding a usable non-revoked attestation."""
    create_actor(client)  # org-1, the signer of the protected attestation
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    create_actor(client, actor_id="org-3", name="Third Org", type="organization")

    attestation = _make_attestation(
        client, "org-1", SEED_A, _make_claim(client, "org-1")["id"]
    )
    # org-2 and org-3 need their own current attestations, hence current
    # public keys, before their signatures can authenticate anything.
    _make_attestation(
        client, "org-2", SEED_B,
        _make_claim(client, "org-2", digest=DIGEST_B)["id"],
    )
    _make_attestation(
        client, "org-3", SEED_C,
        _make_claim(client, "org-3", digest=hashlib.sha256(b"content-c3").hexdigest())["id"],
    )
    return attestation


# --- Signed-header helper ----------------------------------------------------

def _signed_headers(
    method,
    path,
    body,
    *,
    actor="org-1",
    seed=SEED_A,
    timestamp=None,
    signed_timestamp=None,
    signed_method=None,
    signed_path=None,
    signed_body=None,
):
    """Build X-PA/X-PT/X-PS headers over the exact canonical array.

    The ``signed_*`` overrides let a test commit the signature to a value
    that differs from the one actually sent (method/path/body/timestamp
    binding tests).
    """
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    signed_ts = signed_timestamp if signed_timestamp is not None else ts
    body_digest = hashlib.sha256(
        signed_body if signed_body is not None else body
    ).hexdigest()
    message = access_message_bytes(
        signed_method or method,
        signed_path if signed_path is not None else path,
        signed_ts,
        body_digest,
    )
    signature = base64.b64encode(ed25519_sign(seed, message)).decode("ascii")
    return {"X-PA": actor, "X-PT": ts, "X-PS": signature}


def _grant_body(attestation_id, grantee_actor_id="org-2"):
    return {"attestation_id": attestation_id, "grantee_actor_id": grantee_actor_id}


def _post_grant(client, body_obj, *, actor="org-1", seed=SEED_A, **sign_kwargs):
    body = json.dumps(body_obj).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", GRANTS_PATH, body, actor=actor, seed=seed,
                          **sign_kwargs),
    }
    return client.post(GRANTS_PATH, content=body, headers=headers)


def _get_protected(client, attestation_id, *, actor="org-1", seed=SEED_A,
                   path=None, **sign_kwargs):
    path = path or f"/v1/protected/attestations/{attestation_id}"
    headers = _signed_headers(
        "GET", path, b"", actor=actor, seed=seed, **sign_kwargs
    )
    return client.get(path, headers=headers)


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


# --- Canonical message contract ------------------------------------------------


def test_access_message_is_compact_utf8_array_with_lowercase_hex():
    empty = access_message_bytes(
        "GET", "/v1/protected/attestations/att_x", "2026-09-20T00:00:00Z",
        hashlib.sha256(b"").hexdigest(),
    )
    assert empty == (
        b'["provenance-access-v1","GET",'
        b'"/v1/protected/attestations/att_x",'
        b'"2026-09-20T00:00:00Z",'
        b'"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"]'
    )
    unicode_body = access_message_bytes(
        "POST", "/v1/attestation-access-grants", "2026-09-20T00:00:00Z",
        hashlib.sha256('{"grantee_actor_id":"org-证据"}'.encode("utf-8")).hexdigest(),
    )
    # Non-ASCII is emitted as raw UTF-8, never \u escapes; the digest is
    # lowercase hex.
    assert b"\\u" not in unicode_body
    assert unicode_body.startswith(b'["provenance-access-v1","POST",')


# --- First creation -----------------------------------------------------------


def test_first_grant_returns_201_with_exact_public_fields(client):
    attestation = _world(client)
    resp = _post_grant(client, _grant_body(attestation["id"]))
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == {"id", "attestation_id", "grantee_actor_id", "created_at"}
    assert body["id"].startswith("aag_")
    assert len(body["id"]) == len("aag_") + 64
    assert body["attestation_id"] == attestation["id"]
    assert body["grantee_actor_id"] == "org-2"
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))


def test_grant_id_is_the_stable_pair_derived_identifier(client):
    attestation = _world(client)
    body = _post_grant(client, _grant_body(attestation["id"])).json()
    assert body["id"] == attestation_access_grant_id(attestation["id"], "org-2")


def test_grant_and_audit_event_commit_in_one_transaction(client, db_session):
    attestation = _world(client)
    created = _post_grant(client, _grant_body(attestation["id"])).json()

    rows = db_session.execute(select(AttestationAccessGrant)).scalars().all()
    assert [(r.id, r.attestation_id, r.grantee_actor_id) for r in rows] == [
        (created["id"], attestation["id"], "org-2")
    ]
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ATTESTATION_ACCESS_GRANTED
        )
    ).scalars().all()
    assert [e.resource_id for e in events] == [created["id"]]
    assert events[0].created_at.tzinfo.utcoffset(
        events[0].created_at
    ).total_seconds() == 0


# --- Idempotency and immutability ---------------------------------------------


def test_same_pair_retry_returns_200_original_record_and_no_audit(
    client, db_session
):
    attestation = _world(client)
    first = _post_grant(client, _grant_body(attestation["id"]))
    assert first.status_code == 201
    events_after_first = _audit_count(db_session)

    for _ in range(3):
        retry = _post_grant(client, _grant_body(attestation["id"]))
        assert retry.status_code == 200
        assert retry.json() == first.json()

    rows = db_session.execute(select(AttestationAccessGrant)).scalars().all()
    assert [r.id for r in rows] == [first.json()["id"]]
    assert _audit_count(db_session) == events_after_first
    grants_audit = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ATTESTATION_ACCESS_GRANTED
        )
    ).scalars().all()
    assert len(grants_audit) == 1


def test_different_grantees_form_independent_immutable_records(client):
    attestation = _world(client)
    to_org2 = _post_grant(client, _grant_body(attestation["id"], "org-2"))
    to_org3 = _post_grant(client, _grant_body(attestation["id"], "org-3"))
    assert to_org2.status_code == 201
    assert to_org3.status_code == 201
    assert to_org2.json()["id"] != to_org3.json()["id"]

    # Retrying each pair returns its own original record.
    assert (
        _post_grant(client, _grant_body(attestation["id"], "org-2")).json()["id"]
        == to_org2.json()["id"]
    )
    assert (
        _post_grant(client, _grant_body(attestation["id"], "org-3")).json()["id"]
        == to_org3.json()["id"]
    )


# --- Grant authorization boundary ---------------------------------------------


def test_non_signer_cannot_create_a_grant_even_with_a_valid_key(client, db_session):
    attestation = _world(client)
    events_before = _audit_count(db_session)
    resp = _post_grant(
        client, _grant_body(attestation["id"], "org-3"),
        actor="org-2", seed=SEED_B,
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert resp.json()["error"]["details"]["reason"] == "caller_not_signer"
    assert db_session.execute(select(AttestationAccessGrant)).scalars().all() == []
    assert _audit_count(db_session) == events_before


def test_grant_for_missing_attestation_is_422_and_writes_nothing(
    client, db_session
):
    _world(client)
    events_before = _audit_count(db_session)
    resp = _post_grant(client, _grant_body("att_ghost", "org-2"))
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "attestation_not_found"
    assert db_session.execute(select(AttestationAccessGrant)).scalars().all() == []
    assert _audit_count(db_session) == events_before


def test_grant_for_missing_grantee_actor_is_422_and_writes_nothing(
    client, db_session
):
    attestation = _world(client)
    events_before = _audit_count(db_session)
    resp = _post_grant(client, _grant_body(attestation["id"], "ghost"))
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "unknown_grantee_actor"
    assert db_session.execute(select(AttestationAccessGrant)).scalars().all() == []
    assert _audit_count(db_session) == events_before


def test_signer_cannot_grant_to_themselves_is_still_an_independent_record(client):
    # A grant to the signer itself is not forbidden: it is simply a distinct
    # pair the signer may create; reading was already permitted regardless.
    attestation = _world(client)
    resp = _post_grant(client, _grant_body(attestation["id"], "org-1"))
    assert resp.status_code == 201, resp.text


# --- Signed-header contract on the write route --------------------------------


def test_grant_without_headers_is_422(client, db_session):
    attestation = _world(client)
    body = json.dumps(_grant_body(attestation["id"])).encode()
    events_before = _audit_count(db_session)
    resp = client.post(
        GRANTS_PATH, content=body, headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "missing_credentials"
    assert _audit_count(db_session) == events_before


def test_grant_with_wrong_signature_is_422(client, db_session):
    attestation = _world(client)
    resp = _post_grant(
        client, _grant_body(attestation["id"]), actor="org-1", seed=SEED_B
    )
    # SEED_B is not a current key of org-1, so nothing verifies.
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )


def test_grant_rejects_stale_and_malformed_timestamps(client):
    attestation = _world(client)
    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    future = (
        datetime.now(timezone.utc) + timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    cases = {
        stale: "timestamp_out_of_window",
        future: "timestamp_out_of_window",
        "2026-09-20T12:00:00": "invalid_timestamp",
        "2026-09-20 12:00:00Z": "invalid_timestamp",
        "2026-09-20T12:00:00+01:00": "invalid_timestamp",
        "not-a-timestamp": "invalid_timestamp",
    }
    for raw, reason in cases.items():
        resp = _post_grant(client, _grant_body(attestation["id"]), timestamp=raw)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["details"]["reason"] == reason, raw


def test_grant_timestamp_inside_the_300_second_window_is_accepted(client):
    attestation = _world(client)
    within = (
        datetime.now(timezone.utc) - timedelta(seconds=290)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    resp = _post_grant(client, _grant_body(attestation["id"]), timestamp=within)
    assert resp.status_code == 201, resp.text


def test_grant_rejects_malformed_signature_encoding(client):
    attestation = _world(client)
    for raw in ("@@@", "aGVsbG8", base64.b64encode(b"s" * 63).decode()):
        body = json.dumps(_grant_body(attestation["id"])).encode()
        headers = {
            "Content-Type": "application/json",
            **_signed_headers("POST", GRANTS_PATH, body),
        }
        headers["X-PS"] = raw
        resp = client.post(GRANTS_PATH, content=body, headers=headers)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["details"]["reason"] == "invalid_signature"


def test_grant_signature_binds_to_body_bytes(client, db_session):
    attestation = _world(client)
    events_before = _audit_count(db_session)
    # Signature commits to one body; the wire carries another.
    resp = _post_grant(
        client, _grant_body(attestation["id"]),
        signed_body=b'{"attestation_id":"att_other","grantee_actor_id":"x"}',
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )
    assert _audit_count(db_session) == events_before


def test_grant_signature_binds_to_method_and_path(client):
    attestation = _world(client)
    wrong_method = _post_grant(
        client, _grant_body(attestation["id"]), signed_method="GET"
    )
    assert wrong_method.status_code == 422
    wrong_path = _post_grant(
        client, _grant_body(attestation["id"]),
        signed_path="/v1/attestation-access-grants/",
    )
    assert wrong_path.status_code == 422
    other_route = _post_grant(
        client, _grant_body(attestation["id"]),
        signed_path="/v1/attestations",
    )
    assert other_route.status_code == 422


def test_grant_signature_timestamp_must_match_header(client):
    attestation = _world(client)
    sent = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    signed = (
        datetime.now(timezone.utc) - timedelta(seconds=10)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    resp = _post_grant(
        client, _grant_body(attestation["id"]),
        timestamp=sent, signed_timestamp=signed,
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )


def test_grant_body_validation_failures_are_422_and_write_nothing(client, db_session):
    _world(client)
    events_before = _audit_count(db_session)

    def post_obj(obj):
        body = json.dumps(obj).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            **_signed_headers("POST", GRANTS_PATH, body),
        }
        return client.post(GRANTS_PATH, content=body, headers=headers)

    blank_att = post_obj({"attestation_id": "  ", "grantee_actor_id": "org-2"})
    blank_grantee = post_obj({"attestation_id": "att_x", "grantee_actor_id": ""})
    missing = post_obj({"attestation_id": "att_x"})
    extra = post_obj(
        {"attestation_id": "att_x", "grantee_actor_id": "org-2", "nope": True}
    )
    for resp in (blank_att, blank_grantee, missing, extra):
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "validation_error"

    # Malformed JSON is rejected at the boundary regardless of a signature
    # that commits to those exact (malformed) bytes.
    raw_body = b"{not valid json"
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", GRANTS_PATH, raw_body),
    }
    malformed = client.post(GRANTS_PATH, content=raw_body, headers=headers)
    assert malformed.status_code == 422

    assert _audit_count(db_session) == events_before
    assert db_session.execute(select(AttestationAccessGrant)).scalars().all() == []


# --- Key discovery and the revocation boundary --------------------------------


def test_any_current_key_of_the_actor_authenticates(client):
    # org-1 holds two attestations under two different keys; either
    # signature authenticates grant creation.
    create_actor(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    first = _make_attestation(
        client, "org-1", SEED_A, _make_claim(client, "org-1")["id"]
    )
    # A second, still-current attestation/key for the same actor.
    second = _make_attestation(
        client, "org-1", SEED_B,
        _make_claim(
            client, "org-1",
            digest=hashlib.sha256(b"second-content").hexdigest(),
        )["id"],
    )
    assert first["id"] != second["id"]

    # The SEED_B key authenticates even when the grant targets the other
    # attestation: key discovery is per-actor, not per-target.
    resp = _post_grant(
        client, _grant_body(first["id"], "org-2"),
        actor="org-1", seed=SEED_B,
    )
    assert resp.status_code == 201, resp.text


def test_revoked_key_no_longer_authenticates_but_another_current_key_does(client):
    attestation = _world(client)  # org-1 currently holds only the SEED_A key.
    revoke = client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": attestation["id"],
            "revoker_actor_id": "org-1",
            "reason": "rotated",
        },
    )
    assert revoke.status_code == 201

    resp = _post_grant(client, _grant_body(attestation["id"], "org-2"))
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )

    # A second, still-current attestation/key restores authentication.
    second = _make_attestation(
        client, "org-1", SEED_B,
        _make_claim(
            client, "org-1",
            digest=hashlib.sha256(b"after-revoke").hexdigest(),
        )["id"],
    )
    resp = _post_grant(
        client, _grant_body(second["id"], "org-2"),
        actor="org-1", seed=SEED_B,
    )
    assert resp.status_code == 201, resp.text


# --- Protected read: success ---------------------------------------------------


def test_signer_reads_the_attestation_public_view(client):
    attestation = _world(client)
    public = client.get(f"/v1/attestations/{attestation['id']}")
    assert public.status_code == 200

    resp = _get_protected(client, attestation["id"])
    assert resp.status_code == 200
    assert resp.json() == public.json() == attestation


def test_grantee_can_read_after_a_grant(client):
    attestation = _world(client)
    # Before the grant, org-2 gets the opaque 404.
    denied = _get_protected(
        client, attestation["id"], actor="org-2", seed=SEED_B
    )
    assert denied.status_code == 404

    granted = _post_grant(client, _grant_body(attestation["id"], "org-2"))
    assert granted.status_code == 201

    resp = _get_protected(
        client, attestation["id"], actor="org-2", seed=SEED_B
    )
    assert resp.status_code == 200
    assert resp.json() == attestation


def test_protected_read_signs_the_zero_byte_body_digest(client):
    attestation = _world(client)
    path = f"/v1/protected/attestations/{attestation['id']}"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    message = access_message_bytes(
        "GET", path, ts, hashlib.sha256(b"").hexdigest()
    )
    headers = {
        "X-PA": "org-1",
        "X-PT": ts,
        "X-PS": base64.b64encode(ed25519_sign(SEED_A, message)).decode(),
    }
    resp = client.get(path, headers=headers)
    assert resp.status_code == 200


def test_protected_read_query_string_is_not_part_of_the_signed_path(client):
    attestation = _world(client)
    path = f"/v1/protected/attestations/{attestation['id']}"
    # Signature covers the path without the query string; sending one still
    # succeeds.
    resp = _get_protected(client, attestation["id"], path=path)
    assert resp.status_code == 200
    ts_headers = _signed_headers("GET", path, b"")
    assert client.get(path + "?tracking=1", headers=ts_headers).status_code == 200


# --- Protected read: the opaque 404 boundary ----------------------------------


def test_protected_read_missing_target_is_404(client):
    _world(client)
    resp = _get_protected(client, "att_ghost")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_protected_read_unauthenticated_and_unauthorized_are_404(client, db_session):
    attestation = _world(client)
    events_before = _audit_count(db_session)
    path = f"/v1/protected/attestations/{attestation['id']}"

    # No headers at all.
    assert client.get(path).status_code == 404
    # Blank actor header.
    assert client.get(
        path, headers={"X-PA": "  ", "X-PT": "2026-09-20T00:00:00Z", "X-PS": "x"}
    ).status_code == 404
    # A well-formed signature by an existing actor without a current key
    # binding to the claimed identity (org-1 claiming to be org-2).
    assert _get_protected(
        client, attestation["id"], actor="org-2", seed=SEED_A
    ).status_code == 404
    # An authenticated stranger (org-3) without a grant is unauthorized.
    assert _get_protected(
        client, attestation["id"], actor="org-3", seed=SEED_C
    ).status_code == 404
    # An existing but non-existent actor id with a valid signature.
    assert _get_protected(
        client, attestation["id"], actor="ghost", seed=SEED_A
    ).status_code == 404

    # Nothing distinguishes the missing-target response from these.
    assert _get_protected(client, "att_ghost").json() == client.get(
        path
    ).json()
    assert _audit_count(db_session) == events_before


def test_protected_read_signature_binds_method_path_and_timestamp(client):
    attestation = _world(client)
    # A signature minted for the write route never authorizes the read.
    path = f"/v1/protected/attestations/{attestation['id']}"
    assert _get_protected(
        client, attestation["id"], signed_method="POST"
    ).status_code == 404
    assert _get_protected(
        client, attestation["id"], signed_path=path + "/"
    ).status_code == 404
    sent = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    signed = (
        datetime.now(timezone.utc) - timedelta(seconds=5)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert _get_protected(
        client, attestation["id"], timestamp=sent, signed_timestamp=signed
    ).status_code == 404
    # A non-empty body digest cannot authorize a bodyless GET.
    assert _get_protected(
        client, attestation["id"], signed_body=b"x"
    ).status_code == 404


# --- Protected read: malformed credentials remain 422 ---------------------------


def test_protected_read_malformed_timestamp_and_signature_are_422(client):
    attestation = _world(client)
    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert _get_protected(
        client, attestation["id"], timestamp=stale
    ).status_code == 422
    assert _get_protected(
        client, attestation["id"], timestamp="12:00 o'clock"
    ).status_code == 422

    path = f"/v1/protected/attestations/{attestation['id']}"
    # Malformed encoding or a wrong decoded length is invalid input (422);
    # a well-formed but unverifiable signature is merely unauthenticated
    # (404) and is covered by the opaque-404 tests.
    for raw in ("@@@", "aGVsbG8", base64.b64encode(b"x" * 63).decode()):
        headers = _signed_headers("GET", path, b"")
        headers["X-PS"] = raw
        resp = client.get(path, headers=headers)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["code"] == "validation_error"


def test_protected_read_after_grant_is_denied_once_grantee_key_revoked(client):
    attestation = _world(client)
    assert _post_grant(
        client, _grant_body(attestation["id"], "org-2")
    ).status_code == 201
    # org-2 currently holds a SEED_B attestation on its own claim; revoke it.
    org2_att = client.get(
        "/v1/attestations", params={"target_type": "claim"}
    ).json()["items"]
    org2_att = next(a for a in org2_att if a["signer_actor_id"] == "org-2")
    revoked = client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": org2_att["id"],
            "revoker_actor_id": "org-2",
            "reason": "key rotation",
        },
    )
    assert revoked.status_code == 201
    # The grant still exists, but org-2 no longer has a current key: 404.
    assert _get_protected(
        client, attestation["id"], actor="org-2", seed=SEED_B
    ).status_code == 404


# --- Read-only guarantees -------------------------------------------------------


def test_protected_reads_write_no_resources_or_audit(client, db_session):
    attestation = _world(client)
    _post_grant(client, _grant_body(attestation["id"], "org-2"))
    events_before = _audit_count(db_session)
    grants_before = len(
        db_session.execute(select(AttestationAccessGrant)).scalars().all()
    )
    attestations_before = len(
        db_session.execute(select(Attestation)).scalars().all()
    )

    for resp in (
        _get_protected(client, attestation["id"]),
        _get_protected(client, attestation["id"], actor="org-2", seed=SEED_B),
        _get_protected(client, attestation["id"], actor="org-3", seed=SEED_C),
        _get_protected(client, "att_ghost"),
        client.get(f"/v1/protected/attestations/{attestation['id']}"),
    ):
        assert resp.status_code in (200, 404)

    db_session.expire_all()
    assert _audit_count(db_session) == events_before
    assert len(
        db_session.execute(select(AttestationAccessGrant)).scalars().all()
    ) == grants_before
    assert len(
        db_session.execute(select(Attestation)).scalars().all()
    ) == attestations_before


# --- Concurrent race and persistence --------------------------------------------


def test_concurrent_identical_grants_yield_one_record_and_audit(
    tmp_db_url, file_app, file_client
):
    # Build the world through the file-backed client.
    create_actor(file_client)
    content = file_client.post(
        "/v1/contents", json=content_payload()
    ).json()
    claim = file_client.post(
        "/v1/claims",
        json={
            "content_id": content["id"],
            "actor_id": "org-1",
            "claim_type": "authorship",
            "payload": {"s": 1},
        },
    ).json()
    attestation = _make_attestation(file_client, "org-1", SEED_A, claim["id"])
    create_actor(file_client, actor_id="org-2", name="Other", type="organization")

    from provenance import service
    from provenance.schemas import AttestationAccessGrantCreate

    factory = file_app.state.session_factory
    payload = AttestationAccessGrantCreate(
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
                session, payload, "org-1"
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


def test_grants_persist_across_app_restarts(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    attestation = _world(file_client)
    body = json.dumps(_grant_body(attestation["id"], "org-2")).encode()
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", GRANTS_PATH, body),
    }
    created = file_client.post(GRANTS_PATH, content=body, headers=headers).json()

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        path = f"/v1/protected/attestations/{attestation['id']}"
        # Signer access survives the restart.
        headers = _signed_headers("GET", path, b"")
        fetched = client.get(path, headers=headers)
        assert fetched.status_code == 200
        assert fetched.json() == attestation

        import sqlite3

        con = sqlite3.connect(tmp_db_url.removeprefix("sqlite:///"))
        rows = con.execute(
            "SELECT id, attestation_id, grantee_actor_id "
            "FROM attestation_access_grants"
        ).fetchall()
        audit = con.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = ?",
            (EVENT_ATTESTATION_ACCESS_GRANTED,),
        ).fetchone()[0]
        con.close()
        assert rows == [(created["id"], attestation["id"], "org-2")]
        assert audit == 1
