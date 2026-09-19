"""Tests for evidence bundles: POST/GET /v1/evidence-bundles and per-claim lists."""

from __future__ import annotations

import hashlib
from datetime import datetime

from sqlalchemy import select

from provenance.models import (
    EVENT_EVIDENCE_BUNDLE_CREATED,
    AuditEvent,
    EvidenceBundle,
)
from tests.helpers import actor_payload, content_payload

METADATA_1 = {"source": "scanner-1", "score": 0.98, "labels": ["a", "b"]}
METADATA_2 = {"source": "scanner-2", "nested": {"depth": 2}}

EVIDENCE_DIGEST_A = hashlib.sha256(b"evidence-a").hexdigest()
EVIDENCE_DIGEST_B = hashlib.sha256(b"evidence-b").hexdigest()


def _ensure_actor(client, actor_id="org-1"):
    resp = client.post("/v1/actors", json=actor_payload(actor_id=actor_id))
    assert resp.status_code in (201, 409), resp.text


def _setup_claim(client, claim_type="authorship"):
    _ensure_actor(client)
    content = client.post("/v1/contents", json=content_payload()).json()
    claim = client.post(
        "/v1/claims",
        json={
            "content_id": content["id"],
            "actor_id": "org-1",
            "claim_type": claim_type,
            "payload": {"statement": "created by org-1"},
        },
    ).json()
    return claim


_UNSET = object()


def _bundle_payload(
    claim_id,
    evidence_type="signature",
    digest=EVIDENCE_DIGEST_A,
    media_type="application/json",
    metadata=_UNSET,
    algorithm="sha256",
):
    return {
        "claim_id": claim_id,
        "evidence_type": evidence_type,
        "digest_algorithm": algorithm,
        "digest_hex": digest,
        "media_type": media_type,
        "metadata": METADATA_1 if metadata is _UNSET else metadata,
    }


def _create_bundle(client, **overrides):
    resp = client.post("/v1/evidence-bundles", json=_bundle_payload(**overrides))
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_create_bundle_returns_public_fields(client):
    claim = _setup_claim(client)
    body = _create_bundle(client, claim_id=claim["id"])
    assert set(body) == {
        "id",
        "claim_id",
        "evidence_type",
        "digest_algorithm",
        "digest_hex",
        "media_type",
        "metadata",
        "created_at",
    }
    assert body["id"].startswith("evb_")
    assert body["claim_id"] == claim["id"]
    assert body["evidence_type"] == "signature"
    assert body["digest_algorithm"] == "sha256"
    assert body["digest_hex"] == EVIDENCE_DIGEST_A
    assert body["media_type"] == "application/json"
    assert body["metadata"] == METADATA_1
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))


def test_bundle_id_is_deterministic_and_stable(client):
    claim = _setup_claim(client)
    first = _create_bundle(client, claim_id=claim["id"])
    second = client.post(
        "/v1/evidence-bundles", json=_bundle_payload(claim_id=claim["id"])
    )
    assert second.status_code == 200
    assert second.json()["id"] == first["id"]


def test_repeat_submission_keeps_first_metadata_and_adds_no_audit_event(
    client, db_session
):
    claim = _setup_claim(client)
    first = _create_bundle(client, claim_id=claim["id"])
    events_after_create = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    for _ in range(3):
        repeat = client.post(
            "/v1/evidence-bundles",
            json=_bundle_payload(claim_id=claim["id"], metadata=METADATA_2),
        )
        assert repeat.status_code == 200
        # The existing resource is returned with the first metadata intact.
        assert repeat.json() == first
        assert repeat.json()["metadata"] == METADATA_1

    events = db_session.execute(
        select(AuditEvent).order_by(AuditEvent.seq.asc())
    ).scalars().all()
    assert len(events) == events_after_create
    bundle_events = [
        e for e in events if e.event_type == EVENT_EVIDENCE_BUNDLE_CREATED
    ]
    assert len(bundle_events) == 1
    assert bundle_events[0].resource_id == first["id"]


def test_distinct_field_combinations_create_distinct_bundles(client):
    claim = _setup_claim(client)
    other_claim = _setup_claim(client, claim_type="endorsement")
    base = _create_bundle(client, claim_id=claim["id"])
    by_type = _create_bundle(client, claim_id=claim["id"], evidence_type="hash-match")
    by_digest = _create_bundle(client, claim_id=claim["id"], digest=EVIDENCE_DIGEST_B)
    by_claim = _create_bundle(client, claim_id=other_claim["id"])

    ids = {base["id"], by_type["id"], by_digest["id"], by_claim["id"]}
    assert len(ids) == 4
    assert by_type["evidence_type"] == "hash-match"
    assert by_digest["digest_hex"] == EVIDENCE_DIGEST_B
    assert by_claim["claim_id"] == other_claim["id"]


def test_get_bundle_returns_full_public_fields(client):
    claim = _setup_claim(client)
    created = _create_bundle(client, claim_id=claim["id"])
    resp = client.get(f"/v1/evidence-bundles/{created['id']}")
    assert resp.status_code == 200
    assert resp.json() == created


def test_get_unknown_bundle_is_distinct_not_found(client):
    resp = client.get("/v1/evidence-bundles/evb_does_not_exist")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "evidence_bundle_not_found"
    assert error["details"]["evidence_bundle_id"] == "evb_does_not_exist"


def test_list_bundles_for_claim_is_isolated_and_stably_ordered(client):
    claim_a = _setup_claim(client)
    claim_b = _setup_claim(client, claim_type="endorsement")
    a1 = _create_bundle(client, claim_id=claim_a["id"])
    b1 = _create_bundle(client, claim_id=claim_b["id"], evidence_type="other")
    a2 = _create_bundle(
        client, claim_id=claim_a["id"], digest=EVIDENCE_DIGEST_B, metadata=METADATA_2
    )

    resp = client.get(f"/v1/claims/{claim_a['id']}/evidence-bundles")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [a1["id"], a2["id"]]
    assert all(item["claim_id"] == claim_a["id"] for item in body["items"])
    assert body["items"][1]["metadata"] == METADATA_2

    resp_b = client.get(f"/v1/claims/{claim_b['id']}/evidence-bundles")
    assert [item["id"] for item in resp_b.json()["items"]] == [b1["id"]]


def test_list_bundles_empty_for_claim_without_bundles(client):
    claim = _setup_claim(client)
    resp = client.get(f"/v1/claims/{claim['id']}/evidence-bundles")
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 0}


def test_list_bundles_unknown_claim_is_not_found(client):
    resp = client.get("/v1/claims/clm_missing/evidence-bundles")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "claim_not_found"


def test_create_bundle_unknown_claim_is_not_found(client):
    resp = client.post("/v1/evidence-bundles", json=_bundle_payload("clm_ghost"))
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "claim_not_found"
    assert error["details"]["claim_id"] == "clm_ghost"


def test_reject_blank_text_fields(client):
    claim = _setup_claim(client)
    for field in ("claim_id", "evidence_type", "media_type"):
        payload = _bundle_payload(claim_id=claim["id"])
        payload[field] = "   "
        resp = client.post("/v1/evidence-bundles", json=payload)
        assert resp.status_code == 422, field
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_unsupported_digest_algorithm(client):
    claim = _setup_claim(client)
    for bad in ("sha1", "md5", "SHA-512", ""):
        resp = client.post(
            "/v1/evidence-bundles",
            json=_bundle_payload(claim_id=claim["id"], algorithm=bad),
        )
        assert resp.status_code == 422, bad
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_malformed_digest_hex(client):
    claim = _setup_claim(client)
    for bad in ("abc", "z" * 64, "0" * 63, "0" * 65, ""):
        resp = client.post(
            "/v1/evidence-bundles",
            json=_bundle_payload(claim_id=claim["id"], digest=bad),
        )
        assert resp.status_code == 422, bad
        assert resp.json()["error"]["code"] == "validation_error"


def test_uppercase_digest_normalizes_and_deduplicates(client):
    claim = _setup_claim(client)
    first = _create_bundle(client, claim_id=claim["id"])
    repeat = client.post(
        "/v1/evidence-bundles",
        json=_bundle_payload(claim_id=claim["id"], digest=EVIDENCE_DIGEST_A.upper()),
    )
    assert repeat.status_code == 200
    assert repeat.json()["id"] == first["id"]
    assert repeat.json()["digest_hex"] == EVIDENCE_DIGEST_A


def test_reject_non_object_metadata(client):
    claim = _setup_claim(client)
    for bad in ([1, 2], "text", 42, 3.14, True, None):
        resp = client.post(
            "/v1/evidence-bundles",
            json=_bundle_payload(claim_id=claim["id"], metadata=bad),
        )
        assert resp.status_code == 422, bad
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_missing_required_fields(client):
    resp = client.post("/v1/evidence-bundles", json={})
    assert resp.status_code == 422
    issue_fields = {
        ".".join(part for part in issue["loc"] if part != "body")
        for issue in resp.json()["error"]["details"]["issues"]
    }
    assert {
        "claim_id",
        "evidence_type",
        "digest_algorithm",
        "digest_hex",
        "media_type",
        "metadata",
    }.issubset(issue_fields)


def test_reject_malformed_json_body(client):
    resp = client.post(
        "/v1/evidence-bundles",
        content="{not valid json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_failed_bundle_requests_write_no_rows_or_audit_events(client, db_session):
    claim = _setup_claim(client)
    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())

    attempts = [
        client.post("/v1/evidence-bundles", json=_bundle_payload("clm_ghost")),
        client.post(
            "/v1/evidence-bundles",
            json=_bundle_payload(claim_id=claim["id"], evidence_type=" "),
        ),
        client.post(
            "/v1/evidence-bundles",
            json=_bundle_payload(claim_id=claim["id"], algorithm="sha1"),
        ),
        client.post(
            "/v1/evidence-bundles",
            json=_bundle_payload(claim_id=claim["id"], metadata=[]),
        ),
    ]
    assert [r.status_code for r in attempts] == [404, 422, 422, 422]
    assert db_session.execute(select(EvidenceBundle)).scalars().all() == []
    assert len(db_session.execute(select(AuditEvent)).scalars().all()) == events_before


def test_bundle_and_audit_event_commit_atomically(client, db_session):
    claim = _setup_claim(client)
    created = _create_bundle(client, claim_id=claim["id"])

    bundles = db_session.execute(select(EvidenceBundle)).scalars().all()
    events = db_session.execute(
        select(AuditEvent)
        .where(AuditEvent.event_type == EVENT_EVIDENCE_BUNDLE_CREATED)
        .order_by(AuditEvent.seq.asc())
    ).scalars().all()
    # The bundle row and its audit event are both visible after one request:
    # they committed in the same transaction.
    assert [b.id for b in bundles] == [created["id"]]
    assert [e.resource_id for e in events] == [created["id"]]
    created_at = events[0].created_at
    assert created_at.tzinfo.utcoffset(created_at).total_seconds() == 0


def test_raw_evidence_bytes_are_not_accepted_or_persisted(client, db_session):
    claim = _setup_claim(client)
    secret_marker = "unique-evidence-bytes-marker-7f3a"
    payload = _bundle_payload(claim_id=claim["id"])
    # There is no field capable of carrying evidence bytes; an extra one is
    # not accepted into the model, persisted, or echoed back.
    payload["evidence_bytes"] = secret_marker
    resp = client.post("/v1/evidence-bundles", json=payload)
    assert resp.status_code == 201, resp.text
    assert secret_marker not in resp.text

    row = db_session.execute(select(EvidenceBundle)).scalars().one()
    persisted = " ".join(
        str(getattr(row, column))
        for column in (
            "id",
            "claim_id",
            "evidence_type",
            "digest_algorithm",
            "digest_hex",
            "media_type",
            "metadata_json",
        )
    )
    assert secret_marker not in persisted


def test_metadata_round_trips_nested_json(client):
    claim = _setup_claim(client)
    metadata = {"nested": {"list": [1, "two", None], "flag": True}, "n": 3}
    created = _create_bundle(client, claim_id=claim["id"], metadata=metadata)
    assert created["metadata"] == metadata
    fetched = client.get(f"/v1/evidence-bundles/{created['id']}")
    assert fetched.json()["metadata"] == metadata
