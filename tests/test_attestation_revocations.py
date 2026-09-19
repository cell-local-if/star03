"""Tests for immutable attestation revocation records.

Covers POST /v1/attestation-revocations, GET /v1/attestation-revocations/{id},
GET /v1/attestations/{id}/revocations, and the extended trust evaluation:
creation and response shape, idempotent dedup by (attestation, revoker,
reason), independent records for distinct combinations, single-transaction
audit commit, the 404/422 boundary with a strict no-write guarantee,
immutability (no update/delete path), exclusion of revoked attestations from
trust evaluations, and persistence across restarts. All tests are
deterministic and offline (signatures are produced by the stdlib test
signer).
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime

from sqlalchemy import select

from provenance.models import (
    EVENT_ATTESTATION_CREATED,
    EVENT_ATTESTATION_REVOKED,
    Attestation,
    AttestationRevocation,
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


# --- Setup helpers ----------------------------------------------------------


def _create_content(client, actor_id="org-1", digest=DIGEST_A):
    resp = client.post(
        "/v1/contents", json=content_payload(actor_id=actor_id, digest=digest)
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, actor_id="org-1"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": "authorship",
            "payload": {"statement": "revocable"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _attestation_body(target_type, target_id, *, seed=SEED_A, signer_actor_id="org-1"):
    signature = ed25519_sign(
        seed, attestation_message_bytes(target_type, target_id, signer_actor_id)
    )
    return {
        "target_type": target_type,
        "target_id": target_id,
        "signer_actor_id": signer_actor_id,
        "public_key": base64.b64encode(ed25519_public_key(seed)).decode("ascii"),
        "signature": base64.b64encode(signature).decode("ascii"),
    }


def _attest(client, target_type, target_id, **kwargs):
    resp = client.post(
        "/v1/attestations",
        json=_attestation_body(target_type, target_id, **kwargs),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _setup_attestation(client, *, seed=SEED_A, signer_actor_id="org-1"):
    """Create actor, content, claim, and one attestation of the claim."""
    create_actor(client)
    claim = _create_claim(client, _create_content(client)["id"])
    return _attest(client, "claim", claim["id"], seed=seed,
                   signer_actor_id=signer_actor_id)


def _revocation_payload(attestation_id, revoker_actor_id="org-1",
                        reason="key compromised"):
    return {
        "attestation_id": attestation_id,
        "revoker_actor_id": revoker_actor_id,
        "reason": reason,
    }


def _revoke(client, attestation_id, **kwargs):
    resp = client.post(
        "/v1/attestation-revocations",
        json=_revocation_payload(attestation_id, **kwargs),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _evaluate(client, target_type, target_id, **params):
    return client.get(
        "/v1/trust-evaluations",
        params={"target_type": target_type, "target_id": target_id, **params},
    )


# --- Normal creation ---------------------------------------------------------


def test_create_revocation_returns_full_public_fields(client):
    attestation = _setup_attestation(client)
    body = _revoke(client, attestation["id"])

    assert set(body) == {
        "id",
        "attestation_id",
        "revoker_actor_id",
        "reason",
        "created_at",
    }
    assert body["id"].startswith("rev_")
    assert len(body["id"]) == len("rev_") + 64
    assert body["attestation_id"] == attestation["id"]
    assert body["revoker_actor_id"] == "org-1"
    assert body["reason"] == "key compromised"
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))


def test_revocation_id_is_stable_and_deterministic(client):
    attestation = _setup_attestation(client)
    body = _revoke(client, attestation["id"])
    expected_material = json.dumps(
        ["attestation_revocation", attestation["id"], "org-1", "key compromised"],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert body["id"] == "rev_" + hashlib.sha256(expected_material).hexdigest()


def test_get_revocation_returns_the_created_record(client):
    attestation = _setup_attestation(client)
    created = _revoke(client, attestation["id"])
    resp = client.get(f"/v1/attestation-revocations/{created['id']}")
    assert resp.status_code == 200
    assert resp.json() == created


def test_revocation_does_not_modify_the_attestation(client):
    attestation = _setup_attestation(client)
    _revoke(client, attestation["id"])
    # The attestation resource is untouched and still retrievable.
    resp = client.get(f"/v1/attestations/{attestation['id']}")
    assert resp.status_code == 200
    assert resp.json() == attestation


# --- Idempotency and identity ------------------------------------------------


def test_repeat_submission_is_idempotent_200_and_adds_no_audit(
    client, db_session
):
    attestation = _setup_attestation(client)
    first = _revoke(client, attestation["id"])
    events_after_create = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    for _ in range(3):
        repeat = client.post(
            "/v1/attestation-revocations",
            json=_revocation_payload(attestation["id"]),
        )
        assert repeat.status_code == 200
        assert repeat.json() == first

    rows = db_session.execute(select(AttestationRevocation)).scalars().all()
    assert [r.id for r in rows] == [first["id"]]
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ATTESTATION_REVOKED
        )
    ).scalars().all()
    assert len(events) == 1
    assert events[0].resource_id == first["id"]
    # No audit event from the repeat submissions.
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_after_create
    )


def test_distinct_combinations_form_independent_records(client):
    attestation = _setup_attestation(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")

    by_reason = _revoke(client, attestation["id"], reason="key compromised")
    # Same attestation and revoker, different reason: independent record.
    other_reason = _revoke(client, attestation["id"], reason="superseded")
    # Same attestation and reason, different revoker: independent record.
    other_revoker = _revoke(
        client, attestation["id"], revoker_actor_id="org-2"
    )

    ids = {by_reason["id"], other_reason["id"], other_revoker["id"]}
    assert len(ids) == 3
    assert other_reason["reason"] == "superseded"
    assert other_revoker["revoker_actor_id"] == "org-2"


def test_same_reason_on_different_attestations_is_independent(client):
    first_att = _setup_attestation(client)
    content_two = _create_content(client, digest=DIGEST_B)
    claim_two = _create_claim(client, content_two["id"])
    second_att = _attest(client, "claim", claim_two["id"], seed=SEED_B)

    rev_one = _revoke(client, first_att["id"])
    rev_two = _revoke(client, second_att["id"])
    assert rev_one["id"] != rev_two["id"]
    assert rev_one["attestation_id"] == first_att["id"]
    assert rev_two["attestation_id"] == second_att["id"]


# --- Per-attestation listing ---------------------------------------------------


def test_list_returns_revocations_in_creation_order(client):
    attestation = _setup_attestation(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")

    first = _revoke(client, attestation["id"], reason="first")
    second = _revoke(client, attestation["id"], reason="second")
    third = _revoke(
        client, attestation["id"], revoker_actor_id="org-2", reason="first"
    )

    resp = client.get(f"/v1/attestations/{attestation['id']}/revocations")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    assert [item["id"] for item in body["items"]] == [
        first["id"],
        second["id"],
        third["id"],
    ]


def test_list_is_scoped_to_the_attestation(client):
    first_att = _setup_attestation(client)
    content_two = _create_content(client, digest=DIGEST_B)
    claim_two = _create_claim(client, content_two["id"])
    second_att = _attest(client, "claim", claim_two["id"], seed=SEED_B)

    rev = _revoke(client, first_att["id"])
    _revoke(client, second_att["id"])

    body = client.get(
        f"/v1/attestations/{first_att['id']}/revocations"
    ).json()
    assert body["count"] == 1
    assert body["items"][0]["id"] == rev["id"]

    empty = client.get(f"/v1/attestations/{second_att['id']}/revocations")
    # second_att has its own revocation; check an attestation with none.
    third_att = _attest(
        client, "claim", claim_two["id"], seed=SEED_A, signer_actor_id="org-1"
    )
    empty = client.get(f"/v1/attestations/{third_att['id']}/revocations")
    assert empty.status_code == 200
    assert empty.json() == {"items": [], "count": 0}


# --- Missing-resource boundary (distinct from validation) ---------------------


def test_get_unknown_revocation_is_404(client):
    resp = client.get("/v1/attestation-revocations/rev_does_not_exist")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "attestation_revocation_not_found"
    assert error["details"]["revocation_id"] == "rev_does_not_exist"


def test_create_revocation_unknown_attestation_is_404(client):
    create_actor(client)
    resp = client.post(
        "/v1/attestation-revocations",
        json=_revocation_payload("att_ghost"),
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "attestation_not_found"
    assert error["details"]["attestation_id"] == "att_ghost"


def test_create_revocation_unknown_revoker_is_404(client):
    attestation = _setup_attestation(client)
    resp = client.post(
        "/v1/attestation-revocations",
        json=_revocation_payload(attestation["id"], revoker_actor_id="ghost"),
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "unknown_actor"


def test_list_revocations_of_unknown_attestation_is_404(client):
    resp = client.get("/v1/attestations/att_ghost/revocations")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "attestation_not_found"
    assert error["details"]["attestation_id"] == "att_ghost"


# --- Validation boundary -------------------------------------------------------


def test_reject_blank_fields(client):
    attestation = _setup_attestation(client)
    for field in ("attestation_id", "revoker_actor_id", "reason"):
        payload = _revocation_payload(attestation["id"])
        payload[field] = "   "
        resp = client.post("/v1/attestation-revocations", json=payload)
        assert resp.status_code == 422, field
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_missing_required_fields(client):
    resp = client.post("/v1/attestation-revocations", json={})
    assert resp.status_code == 422
    issue_fields = {
        ".".join(part for part in issue["loc"] if part != "body")
        for issue in resp.json()["error"]["details"]["issues"]
    }
    assert {"attestation_id", "revoker_actor_id", "reason"}.issubset(
        issue_fields
    )


def test_reject_undeclared_fields(client):
    attestation = _setup_attestation(client)
    payload = _revocation_payload(attestation["id"])
    payload["unexpected"] = "value"
    resp = client.post("/v1/attestation-revocations", json=payload)
    assert resp.status_code == 422
    issues = resp.json()["error"]["details"]["issues"]
    assert any("unexpected" in issue["loc"] for issue in issues)


def test_reject_malformed_json_body(client):
    create_actor(client)
    resp = client.post(
        "/v1/attestation-revocations",
        content="{not valid json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_failed_creations_write_no_rows_or_audit(client, db_session):
    attestation = _setup_attestation(client)
    events_before = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    attempts = [
        # Unknown attestation.
        client.post(
            "/v1/attestation-revocations",
            json=_revocation_payload("att_ghost"),
        ),
        # Unknown revoker.
        client.post(
            "/v1/attestation-revocations",
            json=_revocation_payload(attestation["id"], revoker_actor_id="ghost"),
        ),
        # Blank reason.
        client.post(
            "/v1/attestation-revocations",
            json=_revocation_payload(attestation["id"], reason="  "),
        ),
        # Undeclared field.
        client.post(
            "/v1/attestation-revocations",
            json=_revocation_payload(attestation["id"]) | {"extra": 1},
        ),
        # Malformed JSON.
        client.post(
            "/v1/attestation-revocations",
            content="{broken",
            headers={"content-type": "application/json"},
        ),
    ]
    assert [r.status_code for r in attempts] == [404, 404, 422, 422, 422]
    assert db_session.execute(select(AttestationRevocation)).scalars().all() == []
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )


# --- Immutability ----------------------------------------------------------------


def test_revocation_records_cannot_be_updated_or_deleted(client):
    attestation = _setup_attestation(client)
    created = _revoke(client, attestation["id"])

    for method in ("put", "patch", "delete"):
        resp = getattr(client, method)(
            f"/v1/attestation-revocations/{created['id']}"
        )
        assert resp.status_code == 405, method
    resp = client.delete(f"/v1/attestations/{attestation['id']}/revocations")
    assert resp.status_code == 405

    # The record is unchanged.
    assert (
        client.get(f"/v1/attestation-revocations/{created['id']}").json()
        == created
    )


# --- Transactional guarantees -----------------------------------------------------


def test_revocation_and_audit_event_commit_atomically(client, db_session):
    attestation = _setup_attestation(client)
    created = _revoke(client, attestation["id"])

    rows = db_session.execute(select(AttestationRevocation)).scalars().all()
    events = db_session.execute(
        select(AuditEvent)
        .where(AuditEvent.event_type == EVENT_ATTESTATION_REVOKED)
        .order_by(AuditEvent.seq.asc())
    ).scalars().all()
    assert [r.id for r in rows] == [created["id"]]
    assert [e.resource_id for e in events] == [created["id"]]
    created_at = events[0].created_at
    assert created_at.tzinfo.utcoffset(created_at).total_seconds() == 0


def test_no_signature_material_is_stored_or_echoed(client, db_session):
    attestation = _setup_attestation(client)
    raw_sig = ed25519_sign(
        SEED_A,
        attestation_message_bytes(
            "claim", attestation["target_id"], "org-1"
        ),
    )
    created = _revoke(client, attestation["id"])

    columns = [c.name for c in AttestationRevocation.__table__.columns]
    assert "signature" not in columns
    assert "public_key" not in columns
    body = json.dumps(created)
    assert base64.b64encode(raw_sig).decode() not in body
    assert "signature" not in created
    # The revocation id derives from association fields, not key material.
    assert raw_sig.hex() not in created["id"]


# --- Trust evaluation interaction ---------------------------------------------------


def test_revoked_attestation_no_longer_counts_toward_trust(client):
    attestation = _setup_attestation(client)
    claim_id = attestation["target_id"]

    before = _evaluate(client, "claim", claim_id).json()
    assert before["qualified_signer_count"] == 1
    assert before["decision"] == "trusted"

    _revoke(client, attestation["id"])

    after = _evaluate(client, "claim", claim_id).json()
    assert after["qualified_signer_count"] == 0
    assert after["decision"] == "untrusted"
    # Other response semantics are unchanged.
    assert after["target_type"] == "claim"
    assert after["target_id"] == claim_id
    assert after["min_signers"] == before["min_signers"]


def test_revoking_one_of_two_signers_flips_threshold_decision(client):
    create_actor(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    claim = _create_claim(client, _create_content(client)["id"])
    att_one = _attest(client, "claim", claim["id"], seed=SEED_A)
    _attest(
        client, "claim", claim["id"], seed=SEED_B, signer_actor_id="org-2"
    )

    trusted = _evaluate(client, "claim", claim["id"], min_signers=2).json()
    assert trusted["qualified_signer_count"] == 2
    assert trusted["decision"] == "trusted"

    _revoke(client, att_one["id"], revoker_actor_id="org-2")

    after = _evaluate(client, "claim", claim["id"], min_signers=2).json()
    assert after["qualified_signer_count"] == 1
    assert after["decision"] == "untrusted"
    # With the default threshold the remaining signer still qualifies.
    default = _evaluate(client, "claim", claim["id"]).json()
    assert default["qualified_signer_count"] == 1
    assert default["decision"] == "trusted"


def test_revocation_of_other_target_does_not_affect_trust(client):
    first_att = _setup_attestation(client)
    content_two = _create_content(client, digest=DIGEST_B)
    claim_two = _create_claim(client, content_two["id"])
    second_att = _attest(client, "claim", claim_two["id"], seed=SEED_B)

    _revoke(client, first_att["id"])

    unaffected = _evaluate(client, "claim", second_att["target_id"]).json()
    assert unaffected["qualified_signer_count"] == 1
    assert unaffected["decision"] == "trusted"


def test_trust_evaluation_still_writes_nothing_after_revocation(
    client, db_session
):
    attestation = _setup_attestation(client)
    _revoke(client, attestation["id"])
    events_before = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    resp = _evaluate(client, "claim", attestation["target_id"])
    assert resp.status_code == 200
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )
    # The attestation rows themselves are untouched by the evaluation.
    assert len(db_session.execute(select(Attestation)).scalars().all()) == 1


# --- Persistence across restarts --------------------------------------------------


def test_revocations_persist_across_app_restarts(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    attestation = _setup_attestation(file_client)
    created = _revoke(file_client, attestation["id"])

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        fetched = client.get(f"/v1/attestation-revocations/{created['id']}")
        assert fetched.status_code == 200
        assert fetched.json() == created

        listing = client.get(
            f"/v1/attestations/{attestation['id']}/revocations"
        ).json()
        assert listing["count"] == 1
        assert listing["items"][0]["id"] == created["id"]

        # The revocation still excludes the attestation from trust.
        evaluation = client.get(
            "/v1/trust-evaluations",
            params={"target_type": "claim",
                    "target_id": attestation["target_id"]},
        ).json()
        assert evaluation["qualified_signer_count"] == 0
        assert evaluation["decision"] == "untrusted"

        import sqlite3

        path = tmp_db_url.removeprefix("sqlite:///")
        con = sqlite3.connect(path)
        revoked_audit = con.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = ?",
            (EVENT_ATTESTATION_REVOKED,),
        ).fetchone()[0]
        created_audit = con.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = ?",
            (EVENT_ATTESTATION_CREATED,),
        ).fetchone()[0]
        con.close()
        assert revoked_audit == 1
        assert created_audit == 1
