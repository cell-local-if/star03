"""Tests for immutable attestation revocation records.

Covers POST /v1/attestation-revocations, GET by id, and the per-attestation
list, including: first-creation 201 response fields and UTC timestamps,
reason trimming, idempotent 200 retries that add no record or audit,
independent records for different combinations, per-attestation isolation
and stable ordering, the 404 boundary (unknown attestation, revocation,
revoker), the 422 boundary (blank/missing/undeclared fields, malformed JSON)
with no writes, append-only immutability (no update/delete), single-
transaction audit commit, a concurrent-identical race, persistence across
restarts, and the trust-evaluation change whereby revoked attestations no
longer qualify. All tests are deterministic and offline (signatures are
produced by the stdlib test signer).
"""

from __future__ import annotations

import json
import threading
from datetime import datetime

from sqlalchemy import func, select

from provenance.models import (
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

REASON_A = "key compromised during incident"
REASON_B = "signer requested withdrawal"


# --- Setup helpers ----------------------------------------------------------


def _create_content(client, actor_id="org-1", digest=DIGEST_A):
    resp = client.post(
        "/v1/contents", json=content_payload(actor_id=actor_id, digest=digest)
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, actor_id="org-1", claim_type="authorship"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": claim_type,
            "payload": {"statement": "attested"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _setup_claim(client):
    create_actor(client)
    return _create_claim(client, _create_content(client)["id"])


def _attestation_body(attestation_id_target, *, seed=SEED_A, signer_actor_id="org-1"):
    import base64

    target_type, target_id = attestation_id_target
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


def _attest(client, target_type, target_id, *, seed=SEED_A, signer_actor_id="org-1"):
    resp = client.post(
        "/v1/attestations",
        json=_attestation_body(
            (target_type, target_id), seed=seed, signer_actor_id=signer_actor_id
        ),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _setup_attestation(client, *, seed=SEED_A, signer_actor_id="org-1"):
    claim = _setup_claim(client)
    att = _attest(
        client, "claim", claim["id"], seed=seed, signer_actor_id=signer_actor_id
    )
    return claim, att


def _revocation_payload(attestation_id, *, revoker_actor_id="org-1", reason=REASON_A):
    return {
        "attestation_id": attestation_id,
        "revoker_actor_id": revoker_actor_id,
        "reason": reason,
    }


def _revoke(client, *args, **kwargs):
    resp = client.post(
        "/v1/attestation-revocations", json=_revocation_payload(*args, **kwargs)
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _evaluate(client, target_type, target_id, **params):
    return client.get(
        "/v1/trust-evaluations",
        params={"target_type": target_type, "target_id": target_id, **params},
    )


# --- Creation ---------------------------------------------------------------


def test_create_revocation_returns_full_public_fields(client):
    _claim, att = _setup_attestation(client)
    body = _revoke(client, att["id"])

    assert set(body) == {
        "id",
        "attestation_id",
        "revoker_actor_id",
        "reason",
        "created_at",
    }
    assert body["id"].startswith("rev_")
    assert len(body["id"]) == len("rev_") + 64
    assert body["attestation_id"] == att["id"]
    assert body["revoker_actor_id"] == "org-1"
    assert body["reason"] == REASON_A
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))


def test_revocation_id_is_stable_and_deterministic(client):
    _claim, att = _setup_attestation(client)
    created = _revoke(client, att["id"])
    expected_material = json.dumps(
        ["attestation_revocation", att["id"], "org-1", REASON_A],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    import hashlib

    assert created["id"] == "rev_" + hashlib.sha256(expected_material).hexdigest()


def test_reason_surrounding_whitespace_is_trimmed(client):
    _claim, att = _setup_attestation(client)
    resp = client.post(
        "/v1/attestation-revocations",
        json=_revocation_payload(att["id"], reason=f"   {REASON_A}\t "),
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["reason"] == REASON_A


def test_revoker_need_not_be_the_attestation_signer(client):
    _claim, att = _setup_attestation(client)
    create_actor(client, actor_id="org-2", name="Reviewer Org", type="organization")
    body = _revoke(client, att["id"], revoker_actor_id="org-2", reason=REASON_B)
    assert body["revoker_actor_id"] == "org-2"
    assert body["reason"] == REASON_B


def test_get_revocation_returns_the_created_record(client):
    _claim, att = _setup_attestation(client)
    created = _revoke(client, att["id"])
    resp = client.get(f"/v1/attestation-revocations/{created['id']}")
    assert resp.status_code == 200
    assert resp.json() == created


# --- Idempotency and distinct combinations ----------------------------------


def test_repeat_submission_is_idempotent_200_and_adds_no_record_or_audit(
    client, db_session
):
    _claim, att = _setup_attestation(client)
    first = _revoke(client, att["id"])
    events_after_create = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    for _ in range(3):
        repeat = client.post(
            "/v1/attestation-revocations",
            json=_revocation_payload(att["id"]),
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
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_after_create
    )


def test_retry_with_whitespace_padded_reason_matches_trimmed_record(client):
    _claim, att = _setup_attestation(client)
    first = _revoke(client, att["id"], reason=REASON_A)
    repeat = client.post(
        "/v1/attestation-revocations",
        json=_revocation_payload(att["id"], reason=f"  {REASON_A}  "),
    )
    assert repeat.status_code == 200
    assert repeat.json()["id"] == first["id"]


def test_different_revoker_and_different_reason_form_independent_records(
    client, db_session
):
    _claim, att = _setup_attestation(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")

    first = _revoke(client, att["id"], revoker_actor_id="org-1", reason=REASON_A)
    # Same attestation and revoker, different reason: independent record.
    second = _revoke(client, att["id"], revoker_actor_id="org-1", reason=REASON_B)
    # Same reason, different revoker: independent record.
    third = _revoke(client, att["id"], revoker_actor_id="org-2", reason=REASON_A)

    ids = {first["id"], second["id"], third["id"]}
    assert len(ids) == 3
    rows = db_session.execute(select(AttestationRevocation)).scalars().all()
    assert len(rows) == 3


# --- Per-attestation listing, ordering, and isolation -----------------------


def test_list_for_attestation_returns_its_records_in_creation_order(client):
    _claim, att = _setup_attestation(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    first = _revoke(client, att["id"], revoker_actor_id="org-1", reason=REASON_A)
    second = _revoke(client, att["id"], revoker_actor_id="org-2", reason=REASON_B)

    resp = client.get(f"/v1/attestations/{att['id']}/revocations")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [first["id"], second["id"]]


def test_listing_is_isolated_per_attestation_and_empty_for_none(client):
    claim_one = _setup_claim(client)
    content_two = _create_content(client, digest=DIGEST_B)
    claim_two = _create_claim(client, content_two["id"])
    att_one = _attest(client, "claim", claim_one["id"])
    att_two = _attest(client, "claim", claim_two["id"], seed=SEED_B)

    revocation = _revoke(client, att_one["id"])

    only_one = client.get(f"/v1/attestations/{att_one['id']}/revocations").json()
    assert [item["id"] for item in only_one["items"]] == [revocation["id"]]
    assert only_one["count"] == 1

    # The other attestation has no revocations: an empty collection, not 404.
    none = client.get(f"/v1/attestations/{att_two['id']}/revocations")
    assert none.status_code == 200
    assert none.json() == {"items": [], "count": 0}


def test_revocation_preserves_the_attestation_unchanged(client):
    _claim, att = _setup_attestation(client)
    _revoke(client, att["id"])

    # The original proof is still fetchable and listed, byte-for-byte.
    assert client.get(f"/v1/attestations/{att['id']}").json() == att
    listed = client.get(
        "/v1/attestations",
        params={"target_type": "claim", "target_id": att["target_id"]},
    ).json()
    assert [item["id"] for item in listed["items"]] == [att["id"]]


# --- Trust evaluation change ------------------------------------------------


def test_revoked_attestation_no_longer_qualifies_trust(client):
    claim, att = _setup_attestation(client)
    assert _evaluate(client, "claim", claim["id"]).json()["decision"] == "trusted"

    _revoke(client, att["id"])
    body = _evaluate(client, "claim", claim["id"]).json()
    assert body["qualified_signer_count"] == 0
    assert body["decision"] == "untrusted"
    # The rest of the evaluation semantics are unchanged.
    assert body["target_type"] == "claim"
    assert body["target_id"] == claim["id"]
    assert body["min_signers"] == 1


def test_revoking_one_key_keeps_a_signer_with_another_live_key(client):
    # Distinct-signer counting is per actor across keys; an actor still
    # qualifies while any non-revoked attestation of theirs remains.
    claim = _setup_claim(client)
    key_a = _attest(client, "claim", claim["id"], seed=SEED_A)
    _attest(client, "claim", claim["id"], seed=SEED_B)

    assert _evaluate(client, "claim", claim["id"]).json()[
        "qualified_signer_count"
    ] == 1
    _revoke(client, key_a["id"])
    after_one = _evaluate(client, "claim", claim["id"]).json()
    assert after_one["qualified_signer_count"] == 1
    assert after_one["decision"] == "trusted"


def test_revoking_one_of_two_distinct_signers_drops_only_that_signer(client):
    claim = _setup_claim(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    att_one = _attest(client, "claim", claim["id"], seed=SEED_A,
                      signer_actor_id="org-1")
    _attest(client, "claim", claim["id"], seed=SEED_B, signer_actor_id="org-2")
    assert _evaluate(client, "claim", claim["id"]).json()[
        "qualified_signer_count"
    ] == 2

    _revoke(client, att_one["id"])
    body = _evaluate(client, "claim", claim["id"]).json()
    assert body["qualified_signer_count"] == 1
    assert body["decision"] == "trusted"


def test_revocation_of_one_target_does_not_change_another_target(client):
    claim_one = _setup_claim(client)
    content_two = _create_content(client, digest=DIGEST_B)
    claim_two = _create_claim(client, content_two["id"])
    att_one = _attest(client, "claim", claim_one["id"])
    _attest(client, "claim", claim_two["id"], seed=SEED_B)

    _revoke(client, att_one["id"])
    assert _evaluate(client, "claim", claim_one["id"]).json()["decision"] == (
        "untrusted"
    )
    assert _evaluate(client, "claim", claim_two["id"]).json()["decision"] == (
        "trusted"
    )


def test_revocation_affects_evidence_bundle_trust_evaluation(client):
    import hashlib

    evidence_digest = hashlib.sha256(b"evidence-rev").hexdigest()
    claim = _setup_claim(client)
    bundle = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim["id"],
            "evidence_type": "raw_capture",
            "digest_algorithm": "sha256",
            "digest_hex": evidence_digest,
            "media_type": "image/jpeg",
            "metadata": {"source": "camera-1"},
        },
    ).json()
    att = _attest(client, "evidence_bundle", bundle["id"])
    assert _evaluate(client, "evidence_bundle", bundle["id"]).json()[
        "decision"
    ] == "trusted"

    _revoke(client, att["id"])
    assert _evaluate(client, "evidence_bundle", bundle["id"]).json() == {
        "target_type": "evidence_bundle",
        "target_id": bundle["id"],
        "min_signers": 1,
        "qualified_signer_count": 0,
        "decision": "untrusted",
    }


# --- Missing-resource boundary ----------------------------------------------


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
    _claim, att = _setup_attestation(client)
    resp = client.post(
        "/v1/attestation-revocations",
        json=_revocation_payload(att["id"], revoker_actor_id="ghost"),
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "unknown_actor"
    assert resp.json()["error"]["details"]["actor_id"] == "ghost"


def test_get_unknown_revocation_is_404(client):
    resp = client.get("/v1/attestation-revocations/rev_does_not_exist")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "attestation_revocation_not_found"
    assert error["details"]["revocation_id"] == "rev_does_not_exist"


def test_list_for_unknown_attestation_is_404(client):
    resp = client.get("/v1/attestations/att_ghost/revocations")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "attestation_not_found"
    assert error["details"]["attestation_id"] == "att_ghost"


# --- Validation boundary -----------------------------------------------------


def test_reject_missing_required_fields(client):
    resp = client.post("/v1/attestation-revocations", json={})
    assert resp.status_code == 422
    issue_fields = {
        ".".join(part for part in issue["loc"] if part != "body")
        for issue in resp.json()["error"]["details"]["issues"]
    }
    assert {"attestation_id", "revoker_actor_id", "reason"}.issubset(issue_fields)


def test_reject_blank_fields(client):
    _claim, att = _setup_attestation(client)
    for field in ("attestation_id", "revoker_actor_id", "reason"):
        payload = _revocation_payload(att["id"])
        payload[field] = "   "
        resp = client.post("/v1/attestation-revocations", json=payload)
        assert resp.status_code == 422, field
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_undeclared_fields(client):
    _claim, att = _setup_attestation(client)
    payload = _revocation_payload(att["id"])
    payload["unexpected"] = "value"
    resp = client.post("/v1/attestation-revocations", json=payload)
    assert resp.status_code == 422
    issues = resp.json()["error"]["details"]["issues"]
    assert any("unexpected" in issue["loc"] for issue in issues)


def test_reject_malformed_json_body(client):
    resp = client.post(
        "/v1/attestation-revocations",
        content="{not valid json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_validation_failures_write_no_records_or_audit(client, db_session):
    create_actor(client)
    revocations_before = db_session.scalar(
        select(func.count()).select_from(AttestationRevocation)
    )
    audit_before = db_session.scalar(select(func.count()).select_from(AuditEvent))

    attempts = [
        client.post("/v1/attestation-revocations", json={}),
        client.post(
            "/v1/attestation-revocations",
            json=_revocation_payload("att_ghost"),
        ),  # 404, also writes nothing
        client.post(
            "/v1/attestation-revocations",
            content="{bad json",
            headers={"content-type": "application/json"},
        ),
    ]
    assert [r.status_code for r in attempts] == [422, 404, 422]
    assert (
        db_session.scalar(select(func.count()).select_from(AttestationRevocation))
        == revocations_before
    )
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audit_before
    )


# --- Append-only immutability ------------------------------------------------


def test_revocation_has_no_update_or_delete_path(client):
    _claim, att = _setup_attestation(client)
    created = _revoke(client, att["id"])
    url = f"/v1/attestation-revocations/{created['id']}"

    for method, kwargs in (
        ("put", {"json": {"reason": "changed"}}),
        ("patch", {"json": {"reason": "changed"}}),
        ("delete", {}),
    ):
        resp = getattr(client, method)(url, **kwargs)
        assert resp.status_code == 405, method
        assert resp.json()["error"]["code"] == "method_not_allowed"

    # The record is unchanged and remains fetchable.
    assert client.get(url).json() == created


def test_re_post_with_a_new_reason_does_not_overwrite(client):
    _claim, att = _setup_attestation(client)
    first = _revoke(client, att["id"], reason=REASON_A)
    second = _revoke(client, att["id"], reason=REASON_B)

    assert first["id"] != second["id"]
    # The first record keeps its original reason.
    assert client.get(
        f"/v1/attestation-revocations/{first['id']}"
    ).json()["reason"] == REASON_A
    listing = client.get(f"/v1/attestations/{att['id']}/revocations").json()
    assert [item["reason"] for item in listing["items"]] == [REASON_A, REASON_B]


# --- Transactional guarantees ------------------------------------------------


def test_revocation_and_audit_event_commit_atomically(client, db_session):
    _claim, att = _setup_attestation(client)
    created = _revoke(client, att["id"])

    rows = db_session.execute(select(AttestationRevocation)).scalars().all()
    events = db_session.execute(
        select(AuditEvent)
        .where(AuditEvent.event_type == EVENT_ATTESTATION_REVOKED)
        .order_by(AuditEvent.seq.asc())
    ).scalars().all()
    assert [r.id for r in rows] == [created["id"]]
    assert [e.resource_id for e in events] == [created["id"]]
    assert events[0].created_at.tzinfo.utcoffset(
        events[0].created_at
    ).total_seconds() == 0


def test_revocation_never_persists_a_signature(client, db_session):
    # The revocation table carries no signature material of any kind.
    columns = {c.name for c in AttestationRevocation.__table__.columns}
    assert columns == {
        "seq",
        "id",
        "attestation_id",
        "revoker_actor_id",
        "reason",
        "created_at",
    }
    assert "signature" not in columns
    assert "signature_digest_hex" not in columns


def test_read_endpoints_and_trust_evaluation_write_nothing(client, db_session):
    _claim, att = _setup_attestation(client)
    created = _revoke(client, att["id"])

    revocations_before = db_session.scalar(
        select(func.count()).select_from(AttestationRevocation)
    )
    attestations_before = db_session.scalar(
        select(func.count()).select_from(Attestation)
    )
    audit_before = db_session.scalar(select(func.count()).select_from(AuditEvent))

    client.get(f"/v1/attestation-revocations/{created['id']}")
    client.get(f"/v1/attestations/{att['id']}/revocations")
    _evaluate(client, "claim", att["target_id"])
    _evaluate(client, "claim", "clm_ghost")  # missing target, still read-only

    assert (
        db_session.scalar(select(func.count()).select_from(AttestationRevocation))
        == revocations_before
    )
    assert (
        db_session.scalar(select(func.count()).select_from(Attestation))
        == attestations_before
    )
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audit_before
    )


# --- Concurrent race boundary ------------------------------------------------


def test_concurrent_identical_revocation_requests_yield_one_record_and_audit(
    tmp_db_url, file_app, file_client
):
    claim = _setup_claim(file_client)
    att = _attest(file_client, "claim", claim["id"])

    from provenance.schemas import AttestationRevocationCreate
    from provenance import service

    factory = file_app.state.session_factory
    request = AttestationRevocationCreate(**_revocation_payload(att["id"]))
    results: list[tuple[str, bool]] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        session = factory()
        try:
            barrier.wait()
            record, created = service.create_attestation_revocation(
                session, request
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
    assert {record_id for record_id, _ in results} == {results[0][0]}
    assert sum(1 for _, created in results if created) == 1

    audit_session = factory()
    try:
        rows = audit_session.execute(
            select(AttestationRevocation)
        ).scalars().all()
        events = audit_session.execute(
            select(AuditEvent).where(
                AuditEvent.event_type == EVENT_ATTESTATION_REVOKED
            )
        ).scalars().all()
        assert [r.id for r in rows] == [results[0][0]]
        assert [e.resource_id for e in events] == [results[0][0]]
    finally:
        audit_session.close()


# --- Persistence across restarts ---------------------------------------------


def test_revocations_persist_across_app_restarts(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    claim = _setup_claim(file_client)
    att = file_client.post(
        "/v1/attestations",
        json=_attestation_body(("claim", claim["id"])),
    ).json()
    revocation = file_client.post(
        "/v1/attestation-revocations",
        json=_revocation_payload(att["id"]),
    ).json()

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        fetched = client.get(
            f"/v1/attestation-revocations/{revocation['id']}"
        )
        assert fetched.status_code == 200
        assert fetched.json() == revocation

        listing = client.get(
            f"/v1/attestations/{att['id']}/revocations"
        ).json()
        assert listing["count"] == 1
        assert [i["id"] for i in listing["items"]] == [revocation["id"]]

        # The revocation continues to suppress trust after a restart.
        assert _evaluate(client, "claim", claim["id"]).json()["decision"] == (
            "untrusted"
        )

        import sqlite3

        path = tmp_db_url.removeprefix("sqlite:///")
        con = sqlite3.connect(path)
        audit_count = con.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = ?",
            (EVENT_ATTESTATION_REVOKED,),
        ).fetchone()[0]
        table_cols = {
            row[1]
            for row in con.execute("PRAGMA table_info(attestation_revocations)")
        }
        con.close()
        assert audit_count == 1
        assert "signature" not in table_cols
