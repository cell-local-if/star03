"""Tests for immutable claim supersessions (source-statement corrections).

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

PAYLOAD_1 = {"statement": "created by org-1", "confidence": 0.9}
PAYLOAD_2 = {"statement": "corrected attribution", "confidence": 0.8}


def _create_content(client, digest=DIGEST_A, actor_id="org-1"):
    resp = client.post(
        "/v1/contents", json=content_payload(actor_id=actor_id, digest=digest)
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, actor_id="org-1", claim_type="authorship",
                  payload=PAYLOAD_1):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": claim_type,
            "payload": payload,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _supersession_payload(
    superseded_claim_id, replacement_claim_id, reason="source corrected"
):
    return {
        "superseded_claim_id": superseded_claim_id,
        "replacement_claim_id": replacement_claim_id,
        "reason": reason,
    }


def _create_supersession(client, **overrides):
    resp = client.post(
        "/v1/claim-supersessions", json=_supersession_payload(**overrides)
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _setup_two_claims_same_content(client):
    create_actor(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    content = _create_content(client)
    old = _create_claim(client, content["id"], actor_id="org-1")
    new = _create_claim(client, content["id"], actor_id="org-2")
    return content, old, new


def test_create_supersession_returns_public_fields(client):
    _, old, new = _setup_two_claims_same_content(client)
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
    assert body["reason"] == "source corrected"
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))


def test_supersession_id_is_deterministic_and_stable(client):
    _, old, new = _setup_two_claims_same_content(client)
    first = _create_supersession(
        client, superseded_claim_id=old["id"], replacement_claim_id=new["id"]
    )
    second = client.post(
        "/v1/claim-supersessions",
        json=_supersession_payload(old["id"], new["id"]),
    )
    assert second.status_code == 200
    assert second.json() == first


def test_reason_is_trimmed_and_part_of_the_identity(client):
    _, old, new = _setup_two_claims_same_content(client)
    first = _create_supersession(
        client, superseded_claim_id=old["id"], replacement_claim_id=new["id"]
    )
    # Surrounding whitespace is trimmed, so this is the same reason and the
    # same record.
    repeat = client.post(
        "/v1/claim-supersessions",
        json=_supersession_payload(old["id"], new["id"], "  source corrected  "),
    )
    assert repeat.status_code == 200
    assert repeat.json() == first


def test_repeat_submission_returns_existing_and_adds_no_audit_event(
    client, db_session
):
    _, old, new = _setup_two_claims_same_content(client)
    first = _create_supersession(
        client, superseded_claim_id=old["id"], replacement_claim_id=new["id"]
    )
    events_after_create = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    for _ in range(3):
        repeat = client.post(
            "/v1/claim-supersessions",
            json=_supersession_payload(old["id"], new["id"]),
        )
        assert repeat.status_code == 200
        assert repeat.json() == first

    events = db_session.execute(
        select(AuditEvent).order_by(AuditEvent.seq.asc())
    ).scalars().all()
    assert len(events) == events_after_create
    supersede_events = [
        e for e in events if e.event_type == EVENT_CLAIM_SUPERSEDED
    ]
    assert len(supersede_events) == 1
    assert supersede_events[0].resource_id == first["id"]


def test_different_reason_forms_independent_record(client, db_session):
    _, old, new = _setup_two_claims_same_content(client)
    first = _create_supersession(
        client,
        superseded_claim_id=old["id"],
        replacement_claim_id=new["id"],
        reason="first correction",
    )
    second_resp = client.post(
        "/v1/claim-supersessions",
        json=_supersession_payload(old["id"], new["id"], "second correction"),
    )
    assert second_resp.status_code == 201, second_resp.text
    second = second_resp.json()
    assert second["id"] != first["id"]
    assert second["reason"] == "second correction"

    rows = db_session.execute(select(ClaimSupersession)).scalars().all()
    assert sorted(r.id for r in rows) == sorted([first["id"], second["id"]])
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_CLAIM_SUPERSEDED
        )
    ).scalars().all()
    assert sorted(e.resource_id for e in events) == sorted(
        [first["id"], second["id"]]
    )


def test_existing_record_is_never_updated(client, db_session):
    _, old, new = _setup_two_claims_same_content(client)
    first = _create_supersession(
        client, superseded_claim_id=old["id"], replacement_claim_id=new["id"]
    )
    repeat = client.post(
        "/v1/claim-supersessions",
        json=_supersession_payload(old["id"], new["id"], "  source corrected "),
    )
    assert repeat.status_code == 200
    assert repeat.json() == first
    again = client.get(f"/v1/claim-supersessions/{first['id']}")
    assert again.json() == first
    rows = db_session.execute(select(ClaimSupersession)).scalars().all()
    assert [r.id for r in rows] == [first["id"]]


def test_get_supersession_returns_full_public_fields(client):
    _, old, new = _setup_two_claims_same_content(client)
    created = _create_supersession(
        client, superseded_claim_id=old["id"], replacement_claim_id=new["id"]
    )
    resp = client.get(f"/v1/claim-supersessions/{created['id']}")
    assert resp.status_code == 200
    assert resp.json() == created


def test_get_unknown_supersession_is_distinct_not_found(client):
    resp = client.get("/v1/claim-supersessions/csp_does_not_exist")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "claim_supersession_not_found"
    assert error["details"]["supersession_id"] == "csp_does_not_exist"


def test_create_supersession_unknown_claim_is_not_found(client):
    _, old, new = _setup_two_claims_same_content(client)
    for missing, present in (
        ("clm_ghost", new["id"]),
        (old["id"], "clm_ghost"),
    ):
        resp = client.post(
            "/v1/claim-supersessions",
            json=_supersession_payload(missing, present),
        )
        assert resp.status_code == 404
        error = resp.json()["error"]
        assert error["code"] == "claim_not_found"
        assert error["details"]["claim_id"] == "clm_ghost"


def test_reject_blank_fields(client):
    _, old, new = _setup_two_claims_same_content(client)
    for payload in (
        _supersession_payload("   ", new["id"]),
        _supersession_payload(old["id"], " "),
        _supersession_payload(old["id"], new["id"], "   "),
    ):
        resp = client.post("/v1/claim-supersessions", json=payload)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_missing_and_extra_fields(client):
    _, old, new = _setup_two_claims_same_content(client)
    base = _supersession_payload(old["id"], new["id"])
    for field in ("superseded_claim_id", "replacement_claim_id", "reason"):
        missing = {k: v for k, v in base.items() if k != field}
        resp = client.post("/v1/claim-supersessions", json=missing)
        assert resp.status_code == 422, field
        assert resp.json()["error"]["code"] == "validation_error"

    extra = dict(base)
    extra["unexpected"] = "value"
    resp = client.post("/v1/claim-supersessions", json=extra)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_reject_malformed_json(client, db_session):
    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())
    resp = client.post(
        "/v1/claim-supersessions",
        content="{not valid json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert db_session.execute(select(ClaimSupersession)).scalars().all() == []
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )


def test_reject_self_supersession(client, db_session):
    _, old, _ = _setup_two_claims_same_content(client)
    resp = client.post(
        "/v1/claim-supersessions",
        json=_supersession_payload(old["id"], old["id"]),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert db_session.execute(select(ClaimSupersession)).scalars().all() == []


def test_reject_cross_content_claims(client, db_session):
    create_actor(client)
    content_a = _create_content(client, digest=DIGEST_A)
    content_b = _create_content(client, digest=DIGEST_B)
    claim_a = _create_claim(client, content_a["id"])
    claim_b = _create_claim(client, content_b["id"])

    resp = client.post(
        "/v1/claim-supersessions",
        json=_supersession_payload(claim_a["id"], claim_b["id"]),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert db_session.execute(select(ClaimSupersession)).scalars().all() == []


def test_reject_supersession_cycle(client, db_session):
    create_actor(client)
    content = _create_content(client)
    a = _create_claim(client, content["id"], claim_type="authorship",
                      payload={"v": 1})
    b = _create_claim(client, content["id"], claim_type="review",
                      payload={"v": 2})
    c = _create_claim(client, content["id"], claim_type="licensing",
                      payload={"v": 3})

    _create_supersession(client, superseded_claim_id=a["id"],
                         replacement_claim_id=b["id"], reason="r1")
    _create_supersession(client, superseded_claim_id=b["id"],
                         replacement_claim_id=c["id"], reason="r2")

    # Closing the loop c -> a (c already replaces ... reaching a) is invalid.
    resp = client.post(
        "/v1/claim-supersessions",
        json=_supersession_payload(c["id"], a["id"], "r3"),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    # The direct reverse edge b -> a is a cycle too.
    resp = client.post(
        "/v1/claim-supersessions",
        json=_supersession_payload(b["id"], a["id"], "r4"),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    rows = db_session.execute(select(ClaimSupersession)).scalars().all()
    assert len(rows) == 2


def test_failed_requests_write_no_rows_or_audit_events(client, db_session):
    _, old, new = _setup_two_claims_same_content(client)
    create_actor(client, actor_id="org-3", name="Third", type="organization")
    other_content = _create_content(client, digest=DIGEST_B)
    other_claim = _create_claim(client, other_content["id"])
    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())

    attempts = [
        client.post(
            "/v1/claim-supersessions",
            json=_supersession_payload("clm_ghost", new["id"]),
        ),
        client.post(
            "/v1/claim-supersessions",
            json=_supersession_payload(old["id"], "clm_ghost"),
        ),
        client.post(
            "/v1/claim-supersessions",
            json=_supersession_payload(old["id"], old["id"]),
        ),
        client.post(
            "/v1/claim-supersessions",
            json=_supersession_payload(old["id"], other_claim["id"]),
        ),
        client.post(
            "/v1/claim-supersessions",
            json=_supersession_payload(" ", new["id"]),
        ),
    ]
    assert [r.status_code for r in attempts] == [404, 404, 422, 422, 422]
    assert db_session.execute(select(ClaimSupersession)).scalars().all() == []
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )


def test_supersession_and_audit_event_commit_atomically(client, db_session):
    _, old, new = _setup_two_claims_same_content(client)
    created = _create_supersession(
        client, superseded_claim_id=old["id"], replacement_claim_id=new["id"]
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


def test_list_supersessions_includes_claim_as_either_end_in_creation_order(
    client,
):
    create_actor(client)
    content = _create_content(client)
    a = _create_claim(client, content["id"], claim_type="authorship",
                      payload={"v": 1})
    b = _create_claim(client, content["id"], claim_type="review",
                      payload={"v": 2})
    c = _create_claim(client, content["id"], claim_type="licensing",
                      payload={"v": 3})

    r1 = _create_supersession(client, superseded_claim_id=a["id"],
                              replacement_claim_id=b["id"], reason="r1")
    r2 = _create_supersession(client, superseded_claim_id=b["id"],
                              replacement_claim_id=c["id"], reason="r2")
    r3 = _create_supersession(client, superseded_claim_id=a["id"],
                              replacement_claim_id=c["id"], reason="r3")

    # a: two records where it is the superseded end.
    resp = client.get(f"/v1/claims/{a['id']}/supersessions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [r1["id"], r3["id"]]

    # b: one as the superseded end, one as the replacement end.
    resp = client.get(f"/v1/claims/{b['id']}/supersessions")
    body = resp.json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [r1["id"], r2["id"]]

    # c: two records where it is the replacement end.
    resp = client.get(f"/v1/claims/{c['id']}/supersessions")
    body = resp.json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [r2["id"], r3["id"]]


def test_list_supersessions_empty_for_claim_without_any(client):
    _, old, _ = _setup_two_claims_same_content(client)
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
    _, old, new = _setup_two_claims_same_content(client)
    created = _create_supersession(
        client, superseded_claim_id=old["id"], replacement_claim_id=new["id"]
    )
    events_after_create = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    assert client.get(f"/v1/claim-supersessions/{created['id']}").status_code == 200
    assert client.get(f"/v1/claims/{old['id']}/supersessions").status_code == 200
    assert client.get(
        "/v1/claim-supersessions/csp_absent"
    ).status_code == 404

    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_after_create
    )
