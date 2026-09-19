"""Tests for immutable provenance claims: POST/GET /v1/claims and per-content lists."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

from sqlalchemy import select

from provenance.models import (
    EVENT_CLAIM_CREATED,
    AuditEvent,
    Claim,
)
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    content_payload,
    create_actor,
)

PAYLOAD_1 = {"statement": "created by org-1", "confidence": 0.9}
PAYLOAD_2 = {"statement": "reviewed", "approved": True}


def _canonical_hex(payload) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _create_content(client, digest=DIGEST_A, actor_id="org-1"):
    resp = client.post(
        "/v1/contents", json=content_payload(actor_id=actor_id, digest=digest)
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


_UNSET = object()


def _claim_payload(content_id, actor_id="org-1", claim_type="authorship", payload=_UNSET):
    return {
        "content_id": content_id,
        "actor_id": actor_id,
        "claim_type": claim_type,
        "payload": PAYLOAD_1 if payload is _UNSET else payload,
    }


def _create_claim(client, **overrides):
    resp = client.post("/v1/claims", json=_claim_payload(**overrides))
    assert resp.status_code == 201, resp.text
    return resp.json()


def _setup_content(client, digest=DIGEST_A):
    create_actor(client)
    return _create_content(client, digest=digest)


def test_create_claim_returns_public_fields_without_payload(client):
    content = _setup_content(client)
    body = _create_claim(client, content_id=content["id"])
    assert set(body) == {
        "id",
        "content_id",
        "actor_id",
        "claim_type",
        "payload_digest_algorithm",
        "payload_digest_hex",
        "created_at",
    }
    assert body["id"].startswith("clm_")
    assert body["content_id"] == content["id"]
    assert body["actor_id"] == "org-1"
    assert body["claim_type"] == "authorship"
    assert body["payload_digest_algorithm"] == "sha256"
    assert body["payload_digest_hex"] == _canonical_hex(PAYLOAD_1)
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))


def test_payload_digest_uses_canonical_json(client):
    content = _setup_content(client)
    # Same object, different key order and whitespace: one canonical digest.
    reordered = {"confidence": 0.9, "statement": "created by org-1"}
    assert list(reordered) != list(PAYLOAD_1)
    body = _create_claim(client, content_id=content["id"], payload=reordered)
    assert body["payload_digest_hex"] == _canonical_hex(PAYLOAD_1)


def test_claim_id_is_deterministic_and_stable(client):
    content = _setup_content(client)
    first = _create_claim(client, content_id=content["id"])
    second = client.post("/v1/claims", json=_claim_payload(content_id=content["id"]))
    assert second.status_code == 200
    assert second.json()["id"] == first["id"]


def test_claim_actor_need_not_be_content_actor(client):
    content = _setup_content(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    body = _create_claim(client, content_id=content["id"], actor_id="org-2")
    assert body["actor_id"] == "org-2"
    assert body["content_id"] == content["id"]


def test_repeat_submission_returns_existing_and_adds_no_audit_event(
    client, db_session
):
    content = _setup_content(client)
    first = _create_claim(client, content_id=content["id"])
    events_after_create = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    for _ in range(3):
        repeat = client.post(
            "/v1/claims", json=_claim_payload(content_id=content["id"])
        )
        assert repeat.status_code == 200
        assert repeat.json() == first

    events = db_session.execute(
        select(AuditEvent).order_by(AuditEvent.seq.asc())
    ).scalars().all()
    assert len(events) == events_after_create
    claim_events = [e for e in events if e.event_type == EVENT_CLAIM_CREATED]
    assert len(claim_events) == 1
    assert claim_events[0].resource_id == first["id"]


def test_distinct_field_combinations_create_distinct_claims(client):
    content = _setup_content(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    base = _create_claim(client, content_id=content["id"])
    by_type = _create_claim(
        client, content_id=content["id"], claim_type="endorsement"
    )
    by_payload = _create_claim(client, content_id=content["id"], payload=PAYLOAD_2)
    by_actor = _create_claim(client, content_id=content["id"], actor_id="org-2")

    ids = {base["id"], by_type["id"], by_payload["id"], by_actor["id"]}
    assert len(ids) == 4
    assert by_type["claim_type"] == "endorsement"
    assert by_payload["payload_digest_hex"] == _canonical_hex(PAYLOAD_2)
    assert by_actor["actor_id"] == "org-2"


def test_existing_claim_is_never_updated(client):
    content = _setup_content(client)
    first = _create_claim(client, content_id=content["id"], payload=PAYLOAD_1)
    # A resubmission with the same identity returns the stored claim as-is.
    repeat = client.post("/v1/claims", json=_claim_payload(content_id=content["id"]))
    assert repeat.status_code == 200
    assert repeat.json() == first
    # A different payload is a new claim, not a mutation of the old one.
    other = _create_claim(client, content_id=content["id"], payload=PAYLOAD_2)
    again = client.get(f"/v1/claims/{first['id']}")
    assert again.json() == first
    assert again.json()["payload_digest_hex"] != other["payload_digest_hex"]


def test_get_claim_returns_full_public_fields(client):
    content = _setup_content(client)
    created = _create_claim(client, content_id=content["id"])
    resp = client.get(f"/v1/claims/{created['id']}")
    assert resp.status_code == 200
    assert resp.json() == created


def test_get_unknown_claim_is_distinct_not_found(client):
    resp = client.get("/v1/claims/clm_does_not_exist")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "claim_not_found"
    assert error["details"]["claim_id"] == "clm_does_not_exist"


def test_list_claims_for_content_is_isolated_and_stably_ordered(client):
    create_actor(client)
    content_a = _create_content(client, digest=DIGEST_A)
    content_b = _create_content(client, digest=DIGEST_B)
    a1 = _create_claim(client, content_id=content_a["id"])
    b1 = _create_claim(client, content_id=content_b["id"], claim_type="other")
    a2 = _create_claim(client, content_id=content_a["id"], payload=PAYLOAD_2)

    resp = client.get(f"/v1/contents/{content_a['id']}/claims")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [a1["id"], a2["id"]]
    assert all(item["content_id"] == content_a["id"] for item in body["items"])

    resp_b = client.get(f"/v1/contents/{content_b['id']}/claims")
    assert [item["id"] for item in resp_b.json()["items"]] == [b1["id"]]


def test_list_claims_empty_for_content_without_claims(client):
    content = _setup_content(client)
    resp = client.get(f"/v1/contents/{content['id']}/claims")
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 0}


def test_list_claims_unknown_content_is_not_found(client):
    resp = client.get("/v1/contents/cnt_missing/claims")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "content_not_found"


def test_create_claim_unknown_content_is_not_found(client):
    create_actor(client)
    resp = client.post("/v1/claims", json=_claim_payload(content_id="cnt_ghost"))
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "content_not_found"
    assert error["details"]["content_id"] == "cnt_ghost"


def test_create_claim_unknown_actor_is_not_found(client):
    content = _setup_content(client)
    resp = client.post(
        "/v1/claims", json=_claim_payload(content_id=content["id"], actor_id="ghost")
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "unknown_actor"
    assert error["details"]["actor_id"] == "ghost"


def test_reject_blank_claim_type(client):
    content = _setup_content(client)
    resp = client.post(
        "/v1/claims",
        json=_claim_payload(content_id=content["id"], claim_type="   "),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_reject_missing_claim_type(client):
    content = _setup_content(client)
    payload = _claim_payload(content_id=content["id"])
    del payload["claim_type"]
    resp = client.post("/v1/claims", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_reject_non_object_payloads(client):
    content = _setup_content(client)
    for bad in ([1, 2], "text", 42, 3.14, True, None):
        resp = client.post(
            "/v1/claims", json=_claim_payload(content_id=content["id"], payload=bad)
        )
        assert resp.status_code == 422, bad
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_missing_payload_and_ids(client):
    resp = client.post("/v1/claims", json={})
    assert resp.status_code == 422
    issue_fields = {
        ".".join(part for part in issue["loc"] if part != "body")
        for issue in resp.json()["error"]["details"]["issues"]
    }
    assert {"content_id", "actor_id", "claim_type", "payload"}.issubset(issue_fields)


def test_reject_malformed_json_body(client):
    resp = client.post(
        "/v1/claims",
        content="{not valid json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_failed_claim_requests_write_no_rows_or_audit_events(client, db_session):
    create_actor(client)
    content = _create_content(client)
    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())

    attempts = [
        client.post("/v1/claims", json=_claim_payload(content_id="cnt_ghost")),
        client.post(
            "/v1/claims",
            json=_claim_payload(content_id=content["id"], actor_id="ghost"),
        ),
        client.post(
            "/v1/claims",
            json=_claim_payload(content_id=content["id"], claim_type=" "),
        ),
        client.post(
            "/v1/claims", json=_claim_payload(content_id=content["id"], payload=[])
        ),
    ]
    assert [r.status_code for r in attempts] == [404, 404, 422, 422]
    assert db_session.execute(select(Claim)).scalars().all() == []
    assert len(db_session.execute(select(AuditEvent)).scalars().all()) == events_before


def test_claim_and_audit_event_commit_atomically(client, db_session):
    content = _setup_content(client)
    created = _create_claim(client, content_id=content["id"])

    claims = db_session.execute(select(Claim)).scalars().all()
    events = db_session.execute(
        select(AuditEvent)
        .where(AuditEvent.event_type == EVENT_CLAIM_CREATED)
        .order_by(AuditEvent.seq.asc())
    ).scalars().all()
    # The claim row and its audit event are both visible after one request:
    # they committed in the same transaction.
    assert [c.id for c in claims] == [created["id"]]
    assert [e.resource_id for e in events] == [created["id"]]
    created_at = events[0].created_at
    assert created_at.tzinfo.utcoffset(created_at).total_seconds() == 0


def test_raw_payload_is_not_persisted(client, db_session):
    content = _setup_content(client)
    secret_marker = "unique-payload-marker-7f3a"
    _create_claim(
        client,
        content_id=content["id"],
        payload={"note": secret_marker},
    )
    row = db_session.execute(select(Claim)).scalars().one()
    persisted = " ".join(
        str(getattr(row, column))
        for column in (
            "id",
            "content_id",
            "actor_id",
            "claim_type",
            "payload_digest_algorithm",
            "payload_digest_hex",
        )
    )
    assert secret_marker not in persisted
    assert row.payload_digest_hex == _canonical_hex({"note": secret_marker})
