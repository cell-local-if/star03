"""Tests for immutable claim supersessions.

Covers POST /v1/claim-supersessions,
GET /v1/claim-supersessions/{supersession_id}, and
GET /v1/claims/{claim_id}/supersessions.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from provenance.models import (
    EVENT_CLAIM_SUPERSEDED,
    AuditEvent,
    ClaimSupersession,
)
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    content_payload,
    create_actor,
)

PAYLOAD_1 = {"statement": "first version"}
PAYLOAD_2 = {"statement": "corrected version"}
PAYLOAD_3 = {"statement": "third version"}

SUPERSESSIONS_PATH = "/v1/claim-supersessions"


def _create_content(client, digest=DIGEST_A, actor_id="org-1"):
    resp = client.post(
        "/v1/contents",
        json=content_payload(actor_id=actor_id, digest=digest),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, payload, claim_type="authorship"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": "org-1",
            "claim_type": claim_type,
            "payload": payload,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _supersession_payload(
    superseded_claim_id, replacement_claim_id, reason="corrected source claim"
):
    return {
        "superseded_claim_id": superseded_claim_id,
        "replacement_claim_id": replacement_claim_id,
        "reason": reason,
    }


def _create_supersession(client, **overrides):
    resp = client.post(SUPERSESSIONS_PATH, json=_supersession_payload(**overrides))
    assert resp.status_code == 201, resp.text
    return resp.json()


def _setup_content_with_claims(client, count=2, digest=DIGEST_A):
    create_actor(client)
    content = _create_content(client, digest=digest)
    payloads = [PAYLOAD_1, PAYLOAD_2, PAYLOAD_3][:count]
    claims = [_create_claim(client, content["id"], p) for p in payloads]
    return content, claims


# --- Creation ---------------------------------------------------------------


def test_create_supersession_returns_public_fields(client):
    _, (old, new) = _setup_content_with_claims(client)
    body = _create_supersession(
        client,
        superseded_claim_id=old["id"],
        replacement_claim_id=new["id"],
    )
    assert set(body) == {
        "id",
        "superseded_claim_id",
        "replacement_claim_id",
        "reason",
        "created_at",
    }
    assert body["id"].startswith("csp_")
    assert body["superseded_claim_id"] == old["id"]
    assert body["replacement_claim_id"] == new["id"]
    assert body["reason"] == "corrected source claim"
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))


def test_supersession_id_is_deterministic_and_stable(client):
    _, (old, new) = _setup_content_with_claims(client)
    first = _create_supersession(
        client,
        superseded_claim_id=old["id"],
        replacement_claim_id=new["id"],
    )
    second = client.post(
        SUPERSESSIONS_PATH,
        json=_supersession_payload(old["id"], new["id"]),
    )
    assert second.status_code == 200
    assert second.json() == first


def test_reason_is_trimmed_and_part_of_the_identity(client):
    _, (old, new) = _setup_content_with_claims(client)
    padded = _create_supersession(
        client,
        superseded_claim_id=old["id"],
        replacement_claim_id=new["id"],
        reason="  corrected source claim  ",
    )
    assert padded["reason"] == "corrected source claim"
    # The trimmed spelling is the same identity: a retry returns the record.
    retry = client.post(
        SUPERSESSIONS_PATH,
        json=_supersession_payload(old["id"], new["id"]),
    )
    assert retry.status_code == 200
    assert retry.json() == padded


def test_repeat_submission_returns_existing_and_adds_no_audit_event(
    client, db_session
):
    _, (old, new) = _setup_content_with_claims(client)
    first = _create_supersession(
        client,
        superseded_claim_id=old["id"],
        replacement_claim_id=new["id"],
    )
    events_after_create = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    for _ in range(3):
        repeat = client.post(
            SUPERSESSIONS_PATH,
            json=_supersession_payload(old["id"], new["id"]),
        )
        assert repeat.status_code == 200
        assert repeat.json() == first

    events = db_session.execute(
        select(AuditEvent).order_by(AuditEvent.seq.asc())
    ).scalars().all()
    assert len(events) == events_after_create
    supersession_events = [
        e for e in events if e.event_type == EVENT_CLAIM_SUPERSEDED
    ]
    assert len(supersession_events) == 1
    assert supersession_events[0].resource_id == first["id"]


def test_different_reason_is_an_independent_record(client, db_session):
    _, (old, new) = _setup_content_with_claims(client)
    first = _create_supersession(
        client,
        superseded_claim_id=old["id"],
        replacement_claim_id=new["id"],
        reason="first rationale",
    )
    second = _create_supersession(
        client,
        superseded_claim_id=old["id"],
        replacement_claim_id=new["id"],
        reason="second rationale",
    )
    assert second["id"] != first["id"]
    assert second["reason"] == "second rationale"

    # Retrying either identity returns its own record, never the other.
    for body, expected in (
        (_supersession_payload(old["id"], new["id"], "first rationale"), first),
        (
            _supersession_payload(old["id"], new["id"], "second rationale"),
            second,
        ),
    ):
        retry = client.post(SUPERSESSIONS_PATH, json=body)
        assert retry.status_code == 200
        assert retry.json() == expected

    rows = db_session.execute(
        select(ClaimSupersession).order_by(ClaimSupersession.seq.asc())
    ).scalars().all()
    assert [r.id for r in rows] == [first["id"], second["id"]]
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_CLAIM_SUPERSEDED
        )
    ).scalars().all()
    assert sorted(e.resource_id for e in events) == sorted(
        [first["id"], second["id"]]
    )


def test_existing_supersession_is_never_updated(client, db_session):
    _, (old, new) = _setup_content_with_claims(client)
    first = _create_supersession(
        client,
        superseded_claim_id=old["id"],
        replacement_claim_id=new["id"],
    )
    repeat = client.post(
        SUPERSESSIONS_PATH,
        json=_supersession_payload(old["id"], new["id"]),
    )
    assert repeat.status_code == 200
    assert repeat.json() == first
    again = client.get(f"{SUPERSESSIONS_PATH}/{first['id']}")
    assert again.json() == first
    rows = db_session.execute(select(ClaimSupersession)).scalars().all()
    assert [r.id for r in rows] == [first["id"]]
    assert rows[0].reason == "corrected source claim"


# --- Detail read ------------------------------------------------------------


def test_get_supersession_returns_full_public_fields(client):
    _, (old, new) = _setup_content_with_claims(client)
    created = _create_supersession(
        client,
        superseded_claim_id=old["id"],
        replacement_claim_id=new["id"],
    )
    resp = client.get(f"{SUPERSESSIONS_PATH}/{created['id']}")
    assert resp.status_code == 200
    assert resp.json() == created


def test_get_unknown_supersession_is_distinct_not_found(client):
    resp = client.get(f"{SUPERSESSIONS_PATH}/csp_does_not_exist")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "claim_supersession_not_found"
    assert error["details"]["supersession_id"] == "csp_does_not_exist"


# --- Creation validation ----------------------------------------------------


def test_create_supersession_unknown_claim_is_claim_not_found(client):
    _, (old, new) = _setup_content_with_claims(client)
    for missing, present in (
        ("clm_ghost", new["id"]),
        (old["id"], "clm_ghost"),
    ):
        resp = client.post(
            SUPERSESSIONS_PATH,
            json=_supersession_payload(missing, present),
        )
        assert resp.status_code == 404
        error = resp.json()["error"]
        assert error["code"] == "claim_not_found"
        assert error["details"]["claim_id"] == "clm_ghost"


def test_reject_blank_and_missing_fields(client):
    _, (old, new) = _setup_content_with_claims(client)
    invalid_bodies = [
        {"replacement_claim_id": new["id"], "reason": "r"},
        {"superseded_claim_id": old["id"], "reason": "r"},
        {"superseded_claim_id": old["id"], "replacement_claim_id": new["id"]},
        _supersession_payload("   ", new["id"]),
        _supersession_payload(old["id"], " "),
        _supersession_payload(old["id"], new["id"], reason="   "),
    ]
    for body in invalid_bodies:
        resp = client.post(SUPERSESSIONS_PATH, json=body)
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_extra_fields(client):
    _, (old, new) = _setup_content_with_claims(client)
    body = _supersession_payload(old["id"], new["id"])
    body["unexpected"] = "value"
    resp = client.post(SUPERSESSIONS_PATH, json=body)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_reject_wrong_field_types(client):
    _, (old, new) = _setup_content_with_claims(client)
    invalid_bodies = [
        _supersession_payload(123, new["id"]),
        _supersession_payload(old["id"], None),
        _supersession_payload(old["id"], new["id"], reason=42),
        [old["id"], new["id"], "r"],
        "not an object",
    ]
    for body in invalid_bodies:
        resp = client.post(SUPERSESSIONS_PATH, json=body)
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_malformed_json_body(client):
    resp = client.post(
        SUPERSESSIONS_PATH,
        content="{not valid json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_reject_cross_content_supersession(client, db_session):
    create_actor(client)
    content_a = _create_content(client, digest=DIGEST_A)
    content_b = _create_content(client, digest=DIGEST_B)
    claim_a = _create_claim(client, content_a["id"], PAYLOAD_1)
    claim_b = _create_claim(client, content_b["id"], PAYLOAD_2)

    for body in (
        _supersession_payload(claim_a["id"], claim_b["id"]),
        _supersession_payload(claim_b["id"], claim_a["id"]),
    ):
        resp = client.post(SUPERSESSIONS_PATH, json=body)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"

    assert db_session.execute(select(ClaimSupersession)).scalars().all() == []


def test_reject_self_supersession(client, db_session):
    _, (old, _new) = _setup_content_with_claims(client)
    resp = client.post(
        SUPERSESSIONS_PATH,
        json=_supersession_payload(old["id"], old["id"]),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert db_session.execute(select(ClaimSupersession)).scalars().all() == []


def test_reject_supersession_cycle(client, db_session):
    _, (a, b, c) = _setup_content_with_claims(client, count=3)
    s1 = _create_supersession(
        client, superseded_claim_id=a["id"], replacement_claim_id=b["id"]
    )
    s2 = _create_supersession(
        client, superseded_claim_id=b["id"], replacement_claim_id=c["id"]
    )

    # Closing the loop (a replaces c while c already replaces b replaces a)
    # is invalid.
    resp = client.post(
        SUPERSESSIONS_PATH,
        json=_supersession_payload(c["id"], a["id"]),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    # The direct reverse edge is a cycle too.
    resp = client.post(
        SUPERSESSIONS_PATH,
        json=_supersession_payload(b["id"], a["id"]),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    rows = db_session.execute(select(ClaimSupersession)).scalars().all()
    assert [r.id for r in rows] == [s1["id"], s2["id"]]


def test_failed_requests_write_no_rows_or_audit_events(client, db_session):
    _, (old, new) = _setup_content_with_claims(client)
    # A cross-content failure needs a claim on a second content; create it
    # before counting so its legitimate audit events are excluded.
    content_b = _create_content(client, digest=DIGEST_B)
    other = _create_claim(client, content_b["id"], PAYLOAD_3)
    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())

    attempts = [
        client.post(
            SUPERSESSIONS_PATH,
            json=_supersession_payload("clm_ghost", new["id"]),
        ),
        client.post(
            SUPERSESSIONS_PATH,
            json=_supersession_payload(old["id"], "clm_ghost"),
        ),
        client.post(
            SUPERSESSIONS_PATH,
            json=_supersession_payload(old["id"], old["id"]),
        ),
        client.post(
            SUPERSESSIONS_PATH,
            json=_supersession_payload(old["id"], new["id"], reason=" "),
        ),
        client.post(
            SUPERSESSIONS_PATH,
            json=_supersession_payload(old["id"], other["id"]),
        ),
    ]
    assert [r.status_code for r in attempts] == [404, 404, 422, 422, 422]

    assert db_session.execute(select(ClaimSupersession)).scalars().all() == []
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )


def test_supersession_and_audit_event_commit_atomically(client, db_session):
    _, (old, new) = _setup_content_with_claims(client)
    created = _create_supersession(
        client,
        superseded_claim_id=old["id"],
        replacement_claim_id=new["id"],
    )

    rows = db_session.execute(select(ClaimSupersession)).scalars().all()
    events = db_session.execute(
        select(AuditEvent)
        .where(AuditEvent.event_type == EVENT_CLAIM_SUPERSEDED)
        .order_by(AuditEvent.seq.asc())
    ).scalars().all()
    # The record and its audit event are both visible after one request:
    # they committed in the same transaction.
    assert [r.id for r in rows] == [created["id"]]
    assert [e.resource_id for e in events] == [created["id"]]
    created_at = events[0].created_at
    assert created_at.tzinfo.utcoffset(created_at).total_seconds() == 0


# --- Per-claim listing ------------------------------------------------------


def test_list_supersessions_returns_either_endpoint_in_creation_order(client):
    _, (a, b, c) = _setup_content_with_claims(client, count=3)
    s1 = _create_supersession(
        client, superseded_claim_id=a["id"], replacement_claim_id=b["id"]
    )
    s2 = _create_supersession(
        client, superseded_claim_id=b["id"], replacement_claim_id=c["id"]
    )
    # A converging edge with another rationale; not a cycle (no path back).
    s3 = _create_supersession(
        client,
        superseded_claim_id=a["id"],
        replacement_claim_id=c["id"],
        reason="additional correction",
    )

    # a: superseded twice, never a replacement.
    resp = client.get(f"/v1/claims/{a['id']}/supersessions")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"items", "count"}
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [s1["id"], s3["id"]]

    # b: one in-edge (s2 supersedes it) and one out-edge (s1 replaces a).
    resp = client.get(f"/v1/claims/{b['id']}/supersessions")
    body = resp.json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [s1["id"], s2["id"]]

    # c: replacement in both records.
    resp = client.get(f"/v1/claims/{c['id']}/supersessions")
    body = resp.json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [s2["id"], s3["id"]]

    # Every item is the full public view.
    for item in body["items"]:
        assert set(item) == {
            "id",
            "superseded_claim_id",
            "replacement_claim_id",
            "reason",
            "created_at",
        }


def test_list_supersessions_empty_for_claim_without_records(client):
    _, (old, _new) = _setup_content_with_claims(client)
    resp = client.get(f"/v1/claims/{old['id']}/supersessions")
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 0}


def test_list_supersessions_unknown_claim_is_not_found(client):
    resp = client.get("/v1/claims/clm_missing/supersessions")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "claim_not_found"
    assert error["details"]["claim_id"] == "clm_missing"


def test_reads_write_no_audit_events(client, db_session):
    _, (old, new) = _setup_content_with_claims(client)
    created = _create_supersession(
        client,
        superseded_claim_id=old["id"],
        replacement_claim_id=new["id"],
    )
    events_after_create = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    assert client.get(f"{SUPERSESSIONS_PATH}/{created['id']}").status_code == 200
    assert (
        client.get(f"/v1/claims/{old['id']}/supersessions").status_code == 200
    )
    assert (
        client.get(f"/v1/claims/{new['id']}/supersessions").status_code == 200
    )
    assert client.get(f"{SUPERSESSIONS_PATH}/csp_ghost").status_code == 404
    assert (
        client.get("/v1/claims/clm_ghost/supersessions").status_code == 404
    )

    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_after_create
    )


def test_supersession_never_echoes_claim_or_content_material(client):
    _, (old, new) = _setup_content_with_claims(client)
    created = _create_supersession(
        client,
        superseded_claim_id=old["id"],
        replacement_claim_id=new["id"],
    )
    detail = client.get(f"{SUPERSESSIONS_PATH}/{created['id']}").json()
    listing = client.get(f"/v1/claims/{old['id']}/supersessions").json()
    # Only the declared public fields appear: no payload, digest, content,
    # actor, signature, or evidence material is rendered anywhere.
    expected = {
        "id",
        "superseded_claim_id",
        "replacement_claim_id",
        "reason",
        "created_at",
    }
    assert set(detail) == expected
    assert set(listing["items"][0]) == expected
