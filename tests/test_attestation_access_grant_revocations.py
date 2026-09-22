"""Tests for immutable attestation access-grant revocations.

Covers ``POST /v1/attestation-access-grant-revocations``, its effect on
``GET /v1/protected/attestations/{attestation_id}``, and the two reviewer
read-only routes ``GET
/v1/attestation-access-grant-revocations/{revocation_id}`` and ``GET
/v1/attestation-access-grants/{grant_id}/revocations``, including:

* the same ``X-PA``/``X-PT``/``X-PS`` signed-header contract as the other
  protected write route (compact UTF-8 array, RFC 3339 UTC timestamps, the
  300-second window, and the SHA-256 of the actual body);
* signer-only revocation: only the signer of the grant's attestation may
  revoke it; an unknown grant and a non-signer caller are 422
  ``validation_error`` and write nothing;
* first-creation 201 with the stable ``agr_`` id, ``grant_id``,
  ``revoker_actor_id``, trimmed reason, and UTC ``created_at``, committed in
  the same transaction as the ``attestation.access_grant_revoked`` audit
  event;
* idempotent 200 retries for the same grant, revoking signer, and reason
  (whitespace-padded variants collapse to the trimmed record) with no new
  row or audit event, while a different reason is an independent immutable
  record;
* the post-revocation read boundary: the signer keeps reading, the revoked
  grant's grantee is denied with the same opaque 404, and other unrevoked
  grants -- another grantee on the same proof and the same grantee on
  another proof -- are unaffected;
* the reviewer read boundary: the single-record view returns exactly the
  public fields (or an explicit 404 for an unknown id), and the per-grant
  collection returns that grant's records only (an unknown grant is 422
  ``validation_error``, an existing unrevoked grant is an empty collection),
  in stable creation order, isolated per grant, with any or repeated query
  parameter rejected as 422 before the lookup, and zero writes on every
  read, empty result, miss, and parameter failure;
* the 422 boundary for credentials, blank/missing/extra fields, and
  malformed JSON, none of which write;
* append-only immutability (no update or delete path), the absence of any
  private-key/signature/payload/byte column, read-only guarantees, a
  concurrent-identical race, and persistence across restarts.

All tests are deterministic and offline (the stdlib test signer produces the
Ed25519 signatures).
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from provenance.access_signing import access_message_bytes
from provenance.ids import attestation_access_grant_revocation_id
from provenance.models import (
    EVENT_ATTESTATION_ACCESS_GRANT_REVOKED,
    AttestationAccessGrant,
    AttestationAccessGrantRevocation,
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

REVOCATIONS_PATH = "/v1/attestation-access-grant-revocations"
GRANTS_PATH = "/v1/attestation-access-grants"
SEED_C = b"test-ed25519-seed-c-00000000000000"[:32]

REASON_A = "access withdrawn after offboarding"
REASON_B = "grant issued in error"


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
    """Three actors, each with a current key; org-1 signs one attestation."""
    create_actor(client)  # org-1, the signer
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    create_actor(client, actor_id="org-3", name="Third Org", type="organization")

    attestation = _make_attestation(
        client, "org-1", SEED_A, _make_claim(client, "org-1")["id"]
    )
    _make_attestation(
        client, "org-2", SEED_B,
        _make_claim(client, "org-2", digest=DIGEST_B)["id"],
    )
    _make_attestation(
        client, "org-3", SEED_C,
        _make_claim(
            client, "org-3", digest=hashlib.sha256(b"content-c3").hexdigest()
        )["id"],
    )
    return attestation


def _grant(client, attestation_id, grantee_actor_id):
    body = json.dumps(
        {"attestation_id": attestation_id, "grantee_actor_id": grantee_actor_id}
    ).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", GRANTS_PATH, body),
    }
    resp = client.post(GRANTS_PATH, content=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


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


def _revocation_body(grant_id, reason=REASON_A):
    return {"grant_id": grant_id, "reason": reason}


def _post_revocation(client, body_obj, *, actor="org-1", seed=SEED_A, **sign_kwargs):
    body = json.dumps(body_obj).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **_signed_headers(
            "POST", REVOCATIONS_PATH, body, actor=actor, seed=seed, **sign_kwargs
        ),
    }
    return client.post(REVOCATIONS_PATH, content=body, headers=headers)


def _get_protected(client, attestation_id, *, actor="org-1", seed=SEED_A):
    path = f"/v1/protected/attestations/{attestation_id}"
    return client.get(
        path, headers=_signed_headers("GET", path, b"", actor=actor, seed=seed)
    )


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


def _revocation_count(session):
    return session.scalar(
        select(func.count()).select_from(AttestationAccessGrantRevocation)
    )


# --- First creation -----------------------------------------------------------


def test_first_revocation_returns_201_with_exact_public_fields(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    resp = _post_revocation(client, _revocation_body(grant["id"]))
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == {
        "id",
        "grant_id",
        "revoker_actor_id",
        "reason",
        "created_at",
    }
    assert body["id"].startswith("agr_")
    assert len(body["id"]) == len("agr_") + 64
    assert body["grant_id"] == grant["id"]
    assert body["revoker_actor_id"] == "org-1"
    assert body["reason"] == REASON_A
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))


def test_revocation_id_is_the_stable_triple_derived_identifier(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    body = _post_revocation(client, _revocation_body(grant["id"])).json()
    assert body["id"] == attestation_access_grant_revocation_id(
        grant["id"], "org-1", REASON_A
    )


def test_reason_surrounding_whitespace_is_trimmed(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    resp = _post_revocation(client, _revocation_body(grant["id"], f"  {REASON_A}  "))
    assert resp.status_code == 201
    assert resp.json()["reason"] == REASON_A


def test_revocation_and_audit_event_commit_in_one_transaction(client, db_session):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _post_revocation(client, _revocation_body(grant["id"])).json()

    rows = db_session.execute(
        select(AttestationAccessGrantRevocation)
    ).scalars().all()
    assert [(r.id, r.grant_id, r.revoker_actor_id, r.reason) for r in rows] == [
        (created["id"], grant["id"], "org-1", REASON_A)
    ]
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ATTESTATION_ACCESS_GRANT_REVOKED
        )
    ).scalars().all()
    assert [e.resource_id for e in events] == [created["id"]]
    assert events[0].created_at.tzinfo.utcoffset(
        events[0].created_at
    ).total_seconds() == 0

    # The grant itself is neither mutated nor deleted.
    grant_row = db_session.execute(
        select(AttestationAccessGrant).where(
            AttestationAccessGrant.id == grant["id"]
        )
    ).scalar_one()
    assert grant_row.grantee_actor_id == "org-2"


# --- Idempotency ---------------------------------------------------------------


def test_same_grant_revoker_reason_retry_returns_200_and_no_audit(
    client, db_session
):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    first = _post_revocation(client, _revocation_body(grant["id"]))
    assert first.status_code == 201
    events_after_first = _audit_count(db_session)

    for _ in range(3):
        retry = _post_revocation(client, _revocation_body(grant["id"]))
        assert retry.status_code == 200
        assert retry.json() == first.json()

    assert _revocation_count(db_session) == 1
    assert _audit_count(db_session) == events_after_first
    grant_revocation_events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ATTESTATION_ACCESS_GRANT_REVOKED
        )
    ).scalars().all()
    assert len(grant_revocation_events) == 1


def test_whitespace_padded_reason_retry_matches_the_trimmed_record(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    first = _post_revocation(client, _revocation_body(grant["id"], REASON_A))
    assert first.status_code == 201
    padded = _post_revocation(
        client, _revocation_body(grant["id"], f"   {REASON_A}\n")
    )
    assert padded.status_code == 200
    assert padded.json() == first.json()


def test_different_reason_forms_an_independent_record_and_keeps_the_first(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    first = _post_revocation(client, _revocation_body(grant["id"], REASON_A))
    second = _post_revocation(client, _revocation_body(grant["id"], REASON_B))
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] != second.json()["id"]

    # Retrying each triple returns its own original record.
    assert (
        _post_revocation(client, _revocation_body(grant["id"], REASON_A)).json()
        == first.json()
    )
    assert (
        _post_revocation(client, _revocation_body(grant["id"], REASON_B)).json()
        == second.json()
    )
    # The first record's reason is never overwritten.
    assert first.json()["reason"] == REASON_A


# --- Signer-only permission isolation ------------------------------------------


def test_the_grantee_cannot_revoke_the_grant(client, db_session):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    events_before = _audit_count(db_session)
    resp = _post_revocation(
        client, _revocation_body(grant["id"]), actor="org-2", seed=SEED_B
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert resp.json()["error"]["details"]["reason"] == "caller_not_signer"
    assert _revocation_count(db_session) == 0
    assert _audit_count(db_session) == events_before


def test_an_unrelated_actor_cannot_revoke_the_grant(client, db_session):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    resp = _post_revocation(
        client, _revocation_body(grant["id"]), actor="org-3", seed=SEED_C
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "caller_not_signer"
    assert _revocation_count(db_session) == 0


def test_unknown_grant_is_422_and_writes_nothing(client, db_session):
    _world(client)
    events_before = _audit_count(db_session)
    resp = _post_revocation(client, _revocation_body("aag_ghost"))
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert resp.json()["error"]["details"]["reason"] == "grant_not_found"
    assert _revocation_count(db_session) == 0
    assert _audit_count(db_session) == events_before


# --- Post-revocation read isolation ---------------------------------------------


def test_signer_still_reads_after_revocation(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    assert _post_revocation(client, _revocation_body(grant["id"])).status_code == 201
    resp = _get_protected(client, attestation["id"])
    assert resp.status_code == 200
    assert resp.json() == attestation


def test_revoked_grants_grantee_is_denied_the_protected_read(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    # The grantee can read while the grant is live.
    assert (
        _get_protected(client, attestation["id"], actor="org-2", seed=SEED_B).status_code
        == 200
    )
    assert _post_revocation(client, _revocation_body(grant["id"])).status_code == 201
    denied = _get_protected(client, attestation["id"], actor="org-2", seed=SEED_B)
    assert denied.status_code == 404
    assert denied.json()["error"]["code"] == "not_found"


def test_another_grantee_on_the_same_proof_is_unaffected(client):
    attestation = _world(client)
    grant_to_org2 = _grant(client, attestation["id"], "org-2")
    _grant(client, attestation["id"], "org-3")
    assert _post_revocation(
        client, _revocation_body(grant_to_org2["id"])
    ).status_code == 201

    assert (
        _get_protected(client, attestation["id"], actor="org-2", seed=SEED_B).status_code
        == 404
    )
    assert (
        _get_protected(client, attestation["id"], actor="org-3", seed=SEED_C).status_code
        == 200
    )


def test_the_same_grantees_other_unrevoked_grant_is_unaffected(client):
    first_attestation = _world(client)
    # A second proof signed by the same signer, with its own grant to org-2.
    second_attestation = _make_attestation(
        client, "org-1", SEED_A,
        _make_claim(
            client, "org-1",
            digest=hashlib.sha256(b"second-content").hexdigest(),
        )["id"],
    )
    first_grant = _grant(client, first_attestation["id"], "org-2")
    _grant(client, second_attestation["id"], "org-2")

    assert _post_revocation(
        client, _revocation_body(first_grant["id"])
    ).status_code == 201

    # The revoked grant's proof is denied; the other proof remains readable.
    assert (
        _get_protected(
            client, first_attestation["id"], actor="org-2", seed=SEED_B
        ).status_code
        == 404
    )
    assert (
        _get_protected(
            client, second_attestation["id"], actor="org-2", seed=SEED_B
        ).status_code
        == 200
    )
    # And the signer retains both.
    assert _get_protected(client, first_attestation["id"]).status_code == 200
    assert _get_protected(client, second_attestation["id"]).status_code == 200


def test_protected_reads_after_revocation_write_nothing(client, db_session):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    _post_revocation(client, _revocation_body(grant["id"]))
    events_before = _audit_count(db_session)
    revocations_before = _revocation_count(db_session)
    grants_before = len(
        db_session.execute(select(AttestationAccessGrant)).scalars().all()
    )

    for resp in (
        _get_protected(client, attestation["id"]),
        _get_protected(client, attestation["id"], actor="org-2", seed=SEED_B),
        _get_protected(client, attestation["id"], actor="org-3", seed=SEED_C),
    ):
        assert resp.status_code in (200, 404)

    db_session.expire_all()
    assert _audit_count(db_session) == events_before
    assert _revocation_count(db_session) == revocations_before
    assert len(
        db_session.execute(select(AttestationAccessGrant)).scalars().all()
    ) == grants_before


# --- Reviewer read routes --------------------------------------------------------


def _get_revocation(client, revocation_id, *, query=""):
    return client.get(f"{REVOCATIONS_PATH}/{revocation_id}{query}")


def _list_revocations(client, grant_id, *, query=""):
    return client.get(f"{GRANTS_PATH}/{grant_id}/revocations{query}")


def _two_revoked_grants(client):
    """Two grants on the same proof, each carrying its own revocation(s)."""
    attestation = _world(client)
    grant_to_org2 = _grant(client, attestation["id"], "org-2")
    grant_to_org3 = _grant(client, attestation["id"], "org-3")
    first_a = _post_revocation(
        client, _revocation_body(grant_to_org2["id"], REASON_A)
    ).json()
    second_b = _post_revocation(
        client, _revocation_body(grant_to_org2["id"], REASON_B)
    ).json()
    other = _post_revocation(
        client, _revocation_body(grant_to_org3["id"], REASON_A)
    ).json()
    return attestation, grant_to_org2, grant_to_org3, first_a, second_b, other


# --- Single-record detail view ---------------------------------------------------


def test_get_revocation_returns_the_created_record_without_credentials(client):
    # Reviewer access: no X-PA/X-PT/X-PS headers are required.
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _post_revocation(client, _revocation_body(grant["id"])).json()
    resp = _get_revocation(client, created["id"])
    assert resp.status_code == 200, resp.text
    assert resp.json() == created


def test_get_revocation_view_carries_exactly_the_public_fields(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _post_revocation(client, _revocation_body(grant["id"])).json()
    body = _get_revocation(client, created["id"]).json()
    assert set(body) == {
        "id",
        "grant_id",
        "revoker_actor_id",
        "reason",
        "created_at",
    }
    assert body["id"] == created["id"]
    assert body["grant_id"] == grant["id"]
    assert body["revoker_actor_id"] == "org-1"
    assert body["reason"] == REASON_A
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))
    # No private key, raw signature, authentication header, claim payload,
    # content/evidence bytes, or internal surrogate is ever rendered.
    for forbidden in (
        "signature",
        "public_key",
        "signature_digest_hex",
        "payload",
        "content",
        "evidence",
        "headers",
        "seq",
        "attestation_id",
    ):
        assert forbidden not in body


def test_get_revocation_id_is_never_reverse_looked_up_by_other_fields(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _post_revocation(client, _revocation_body(grant["id"])).json()
    # The id of the grant itself and of its attestation are not revocation
    # ids: each is a plain id lookup that misses, never a field-based search.
    for foreign_id in (grant["id"], attestation["id"]):
        resp = _get_revocation(client, foreign_id)
        assert resp.status_code == 404, foreign_id


def test_get_unknown_revocation_is_an_explicit_404(client):
    _world(client)
    for unknown_id in ("agr_ghost", "agr_" + "0" * 64):
        resp = _get_revocation(client, unknown_id)
        assert resp.status_code == 404, unknown_id
        error = resp.json()["error"]
        assert error["code"] == "attestation_access_grant_revocation_not_found"
        assert error["details"]["revocation_id"] == unknown_id


# --- Per-grant collection ---------------------------------------------------------


def test_list_for_grant_returns_exactly_items_and_count_envelope(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    _post_revocation(client, _revocation_body(grant["id"]))
    body = _list_revocations(client, grant["id"]).json()
    assert set(body) == {"items", "count"}
    assert body["count"] == 1
    assert set(body["items"][0]) == {
        "id",
        "grant_id",
        "revoker_actor_id",
        "reason",
        "created_at",
    }


def test_list_for_grant_returns_its_records_in_stable_creation_order(client):
    _att, grant, _other, first_a, second_b, _other_rev = _two_revoked_grants(
        client
    )
    resp = _list_revocations(client, grant["id"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [
        first_a["id"],
        second_b["id"],
    ]
    assert [item["reason"] for item in body["items"]] == [REASON_A, REASON_B]
    for item in body["items"]:
        assert item["grant_id"] == grant["id"]


def test_listing_is_isolated_per_grant_even_with_shared_reason_text(client):
    _att, grant_a, grant_b, first_a, second_b, other = _two_revoked_grants(
        client
    )
    list_a = _list_revocations(client, grant_a["id"]).json()
    list_b = _list_revocations(client, grant_b["id"]).json()
    assert [i["id"] for i in list_a["items"]] == [
        first_a["id"],
        second_b["id"],
    ]
    assert list_a["count"] == 2
    # The same reason text on a different grant never bleeds across.
    assert [i["id"] for i in list_b["items"]] == [other["id"]]
    assert list_b["items"][0]["reason"] == REASON_A
    assert list_b["count"] == 1


def test_listing_does_not_reverse_lookup_an_attestation_or_revocation_id(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _post_revocation(client, _revocation_body(grant["id"])).json()
    # Only a grant id scopes the collection: an attestation id or the
    # revocation's own id in the path is an unknown *grant*, not a hit.
    for foreign_id in (attestation["id"], created["id"]):
        resp = _list_revocations(client, foreign_id)
        assert resp.status_code == 422, foreign_id
        error = resp.json()["error"]
        assert error["code"] == "validation_error"
        assert error["details"]["reason"] == "grant_not_found"


def test_existing_grant_without_revocations_is_empty_collection(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    resp = _list_revocations(client, grant["id"])
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0}


def test_unknown_grant_is_422_validation_error_not_an_empty_collection(client):
    _world(client)
    resp = _list_revocations(client, "aag_ghost")
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"]["reason"] == "grant_not_found"


def test_per_grant_collection_requires_no_credentials(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    _post_revocation(client, _revocation_body(grant["id"]))
    # A bare GET with no signed headers is a valid reviewer read.
    resp = client.get(f"{GRANTS_PATH}/{grant['id']}/revocations")
    assert resp.status_code == 200, resp.text


# --- Query-parameter boundary and its priority over lookups -----------------------


def test_get_revocation_rejects_any_or_repeated_query_parameter(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _post_revocation(client, _revocation_body(grant["id"])).json()
    base = f"{REVOCATIONS_PATH}/{created['id']}"
    # (query string, rejected parameter name -- the lexicographically first
    # distinct name, which is all of them when only one is present).
    cases = (
        ("?a=1", "a"),
        ("?unknown=", "unknown"),
        ("?limit=1", "limit"),
        ("?a=1&b=2", "a"),
        ("?a=1&a=2", "a"),
    )
    for query, field in cases:
        resp = client.get(base + query)
        assert resp.status_code == 422, query
        error = resp.json()["error"]
        assert error["code"] == "validation_error"
        issue = error["details"]["issues"][0]
        assert issue["loc"] == ["query", field]
        assert "unknown query parameter" in issue["msg"]


def test_get_revocation_param_failure_precedes_the_404_lookup(client):
    # With an unknown id the read would be 404; any parameter must instead
    # produce 422 before the lookup runs.
    resp = client.get(f"{REVOCATIONS_PATH}/agr_ghost?anything=1")
    assert resp.status_code == 422, resp.text
    issue = resp.json()["error"]["details"]["issues"][0]
    assert issue["loc"] == ["query", "anything"]
    # A repeated parameter is rejected the same way, not collapsed.
    repeated = client.get(f"{REVOCATIONS_PATH}/agr_ghost?x=1&x=2")
    assert repeated.status_code == 422
    assert repeated.json()["error"]["details"]["issues"][0]["loc"] == [
        "query",
        "x",
    ]


def test_list_rejects_any_or_repeated_query_parameter_before_grant_lookup(client):
    # An unknown grant is itself 422; distinguish parameter failure from
    # grant_not_found by the issue-shaped details, proving params win.
    resp = client.get(f"{GRANTS_PATH}/aag_ghost/revocations?anything=1")
    assert resp.status_code == 422, resp.text
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"]["issues"][0]["loc"] == ["query", "anything"]

    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    _post_revocation(client, _revocation_body(grant["id"]))
    base = f"{GRANTS_PATH}/{grant['id']}/revocations"
    for query in ("?a=1", "?cursor=", "?a=1&b=2", "?a=1&a=2"):
        rejected = client.get(base + query)
        assert rejected.status_code == 422, query
        assert rejected.json()["error"]["code"] == "validation_error"


# --- Strictly read-only -------------------------------------------------------------


def test_reviewer_reads_never_write_grants_revocations_or_audit(client, db_session):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _post_revocation(client, _revocation_body(grant["id"])).json()
    unrevoked = _grant(client, attestation["id"], "org-3")

    grants_before = len(
        db_session.execute(select(AttestationAccessGrant)).scalars().all()
    )
    revocations_before = _revocation_count(db_session)
    events_before = _audit_count(db_session)

    responses = (
        _get_revocation(client, created["id"]),                       # 200
        _get_revocation(client, "agr_ghost"),                         # 404 miss
        _list_revocations(client, grant["id"]),                       # 200
        _list_revocations(client, unrevoked["id"]),                   # empty
        _list_revocations(client, "aag_ghost"),                       # 422
        client.get(f"{REVOCATIONS_PATH}/{created['id']}?bad=1"),      # 422
        client.get(f"{GRANTS_PATH}/{grant['id']}/revocations?bad=1"),# 422
    )
    assert [r.status_code for r in responses] == [200, 404, 200, 200, 422, 422, 422]

    # Repeat the successful reads; determinism plus still zero writes.
    assert _get_revocation(client, created["id"]).json() == created
    assert [
        i["id"] for i in _list_revocations(client, grant["id"]).json()["items"]
    ] == [created["id"]]

    db_session.expire_all()
    assert len(
        db_session.execute(select(AttestationAccessGrant)).scalars().all()
    ) == grants_before
    assert _revocation_count(db_session) == revocations_before
    assert _audit_count(db_session) == events_before
    # The unrevoked grant is untouched and still authorizes its grantee.
    assert (
        _get_protected(client, attestation["id"], actor="org-3", seed=SEED_C).status_code
        == 200
    )
    assert (
        _get_protected(client, attestation["id"], actor="org-2", seed=SEED_B).status_code
        == 404
    )


def test_reviewer_reads_persist_unchanged_across_a_restart(
    tmp_db_url, file_client
):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    attestation = _world(file_client)
    grant = _grant(file_client, attestation["id"], "org-2")
    first = _post_revocation(
        file_client, _revocation_body(grant["id"], REASON_A)
    ).json()
    second = _post_revocation(
        file_client, _revocation_body(grant["id"], REASON_B)
    ).json()

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        assert _get_revocation(client, first["id"]).json() == first
        body = _list_revocations(client, grant["id"]).json()
        assert body["count"] == 2
        assert [i["id"] for i in body["items"]] == [first["id"], second["id"]]
        assert _get_revocation(client, "agr_ghost").status_code == 404


# --- Signed-header contract on the write route ---------------------------------


def test_revocation_without_headers_is_422(client, db_session):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    body = json.dumps(_revocation_body(grant["id"])).encode()
    events_before = _audit_count(db_session)
    resp = client.post(
        REVOCATIONS_PATH, content=body, headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == "missing_credentials"
    assert _audit_count(db_session) == events_before


def test_revocation_with_wrong_signature_is_422(client, db_session):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    events_before = _audit_count(db_session)
    resp = _post_revocation(
        client, _revocation_body(grant["id"]), actor="org-1", seed=SEED_B
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )
    assert _audit_count(db_session) == events_before


def test_revocation_rejects_stale_and_malformed_timestamps(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
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
        resp = _post_revocation(client, _revocation_body(grant["id"]), timestamp=raw)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["details"]["reason"] == reason, raw


def test_revocation_timestamp_inside_the_window_is_accepted(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    within = (
        datetime.now(timezone.utc) - timedelta(seconds=290)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    resp = _post_revocation(client, _revocation_body(grant["id"]), timestamp=within)
    assert resp.status_code == 201, resp.text


def test_revocation_rejects_malformed_signature_encoding(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    for raw in ("@@@", "aGVsbG8", base64.b64encode(b"s" * 63).decode()):
        body = json.dumps(_revocation_body(grant["id"])).encode()
        headers = {
            "Content-Type": "application/json",
            **_signed_headers("POST", REVOCATIONS_PATH, body),
        }
        headers["X-PS"] = raw
        resp = client.post(REVOCATIONS_PATH, content=body, headers=headers)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["details"]["reason"] == "invalid_signature"


def test_revocation_signature_binds_to_body_method_path_and_timestamp(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")

    wrong_body = _post_revocation(
        client, _revocation_body(grant["id"]),
        signed_body=b'{"grant_id":"aag_other","reason":"x"}',
    )
    assert wrong_body.status_code == 422
    assert wrong_body.json()["error"]["details"]["reason"] == (
        "signature_verification_failed"
    )

    assert _post_revocation(
        client, _revocation_body(grant["id"]), signed_method="GET"
    ).status_code == 422
    assert _post_revocation(
        client, _revocation_body(grant["id"]),
        signed_path=REVOCATIONS_PATH + "/",
    ).status_code == 422
    # A signature minted for the grants route never authorizes a revocation.
    assert _post_revocation(
        client, _revocation_body(grant["id"]), signed_path=GRANTS_PATH
    ).status_code == 422

    sent = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    signed = (
        datetime.now(timezone.utc) - timedelta(seconds=10)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert _post_revocation(
        client, _revocation_body(grant["id"]),
        timestamp=sent, signed_timestamp=signed,
    ).status_code == 422


# --- Body validation ------------------------------------------------------------


def test_revocation_body_validation_failures_are_422_and_write_nothing(
    client, db_session
):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    events_before = _audit_count(db_session)

    def post_obj(obj):
        body = json.dumps(obj).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            **_signed_headers("POST", REVOCATIONS_PATH, body),
        }
        return client.post(REVOCATIONS_PATH, content=body, headers=headers)

    blank_grant = post_obj({"grant_id": "  ", "reason": REASON_A})
    blank_reason = post_obj({"grant_id": grant["id"], "reason": "   "})
    empty_reason = post_obj({"grant_id": grant["id"], "reason": ""})
    missing_grant = post_obj({"reason": REASON_A})
    missing_reason = post_obj({"grant_id": grant["id"]})
    empty = post_obj({})
    extra = post_obj(
        {"grant_id": grant["id"], "reason": REASON_A, "nope": True}
    )
    non_string_grant = post_obj({"grant_id": 123, "reason": REASON_A})
    for resp in (
        blank_grant,
        blank_reason,
        empty_reason,
        missing_grant,
        missing_reason,
        empty,
        extra,
        non_string_grant,
    ):
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "validation_error"

    # Malformed JSON is rejected at the boundary even with a signature over
    # those exact bytes.
    raw_body = b"{not valid json"
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", REVOCATIONS_PATH, raw_body),
    }
    malformed = client.post(REVOCATIONS_PATH, content=raw_body, headers=headers)
    assert malformed.status_code == 422

    assert _audit_count(db_session) == events_before
    assert _revocation_count(db_session) == 0


# --- Append-only immutability ---------------------------------------------------


def test_revocation_collection_rejects_non_post_methods(client):
    _world(client)
    for method in ("get", "put", "patch", "delete"):
        resp = getattr(client, method)(REVOCATIONS_PATH)
        assert resp.status_code == 405, method
        assert resp.json()["error"]["code"] == "method_not_allowed"


def test_individual_revocation_resource_cannot_be_updated_or_deleted(client):
    attestation = _world(client)
    grant = _grant(client, attestation["id"], "org-2")
    created = _post_revocation(client, _revocation_body(grant["id"])).json()
    url = f"{REVOCATIONS_PATH}/{created['id']}"
    # The individual resource has a read-only reviewer GET but no update or
    # delete path: every mutation attempt is 405 method_not_allowed.
    for method, kwargs in (
        ("put", {"json": {"reason": "changed"}}),
        ("patch", {"json": {"reason": "changed"}}),
        ("delete", {}),
    ):
        resp = getattr(client, method)(url, **kwargs)
        assert resp.status_code == 405, method
        assert resp.json()["error"]["code"] == "method_not_allowed"

    # The record is unchanged and still governs the read boundary.
    assert client.get(url).json() == created
    assert (
        _get_protected(client, attestation["id"], actor="org-2", seed=SEED_B).status_code
        == 404
    )


def test_revocation_table_carries_no_signature_or_material_columns(client):
    columns = {c.name for c in AttestationAccessGrantRevocation.__table__.columns}
    assert columns == {
        "seq",
        "id",
        "grant_id",
        "revoker_actor_id",
        "reason",
        "created_at",
    }
    assert "signature" not in columns
    assert "signature_digest_hex" not in columns
    assert "attestation_id" not in columns


# --- Concurrent race -------------------------------------------------------------


def test_concurrent_identical_revocations_yield_one_record_and_audit(
    tmp_db_url, file_app, file_client
):
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
    create_actor(
        file_client, actor_id="org-2", name="Other", type="organization"
    )
    grant = _grant(file_client, attestation["id"], "org-2")

    from provenance import service
    from provenance.schemas import AttestationAccessGrantRevocationCreate

    factory = file_app.state.session_factory
    payload = AttestationAccessGrantRevocationCreate(
        grant_id=grant["id"], reason=REASON_A
    )
    results: list[tuple[str, bool]] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        session = factory()
        try:
            barrier.wait()
            record, created = service.create_attestation_access_grant_revocation(
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
    assert {rid for rid, _ in results} == {results[0][0]}
    assert sum(1 for _, created in results if created) == 1

    audit_session = factory()
    try:
        rows = audit_session.execute(
            select(AttestationAccessGrantRevocation)
        ).scalars().all()
        events = audit_session.execute(
            select(AuditEvent).where(
                AuditEvent.event_type
                == EVENT_ATTESTATION_ACCESS_GRANT_REVOKED
            )
        ).scalars().all()
        assert [r.id for r in rows] == [results[0][0]]
        assert [e.resource_id for e in events] == [results[0][0]]
    finally:
        audit_session.close()


# --- Persistence across restarts --------------------------------------------------


def test_revocation_persists_and_keeps_denying_across_restarts(
    tmp_db_url, file_client
):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    attestation = _world(file_client)
    grant = _grant(file_client, attestation["id"], "org-2")
    created = _post_revocation(
        file_client, _revocation_body(grant["id"])
    ).json()

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        # Signer access survives; the grantee stays denied.
        assert _get_protected(client, attestation["id"]).status_code == 200
        assert (
            _get_protected(
                client, attestation["id"], actor="org-2", seed=SEED_B
            ).status_code
            == 404
        )

        import sqlite3

        con = sqlite3.connect(tmp_db_url.removeprefix("sqlite:///"))
        rows = con.execute(
            "SELECT id, grant_id, revoker_actor_id, reason "
            "FROM attestation_access_grant_revocations"
        ).fetchall()
        audit = con.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = ?",
            (EVENT_ATTESTATION_ACCESS_GRANT_REVOKED,),
        ).fetchone()[0]
        con.close()
        assert rows == [(created["id"], grant["id"], "org-1", REASON_A)]
        assert audit == 1
