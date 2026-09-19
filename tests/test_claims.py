"""Tests for immutable content claims: POST/GET /v1/claims and per-content listing."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import select

from provenance.app import create_app
from provenance.config import Settings
from provenance.models import EVENT_CLAIM_CREATED, AuditEvent, Claim
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    claim_payload,
    create_actor,
    create_content,
)

CLAIM_FIELDS = {
    "id",
    "content_id",
    "actor_id",
    "claim_type",
    "payload_digest_algorithm",
    "payload_digest",
    "created_at",
}

PAYLOAD_A = {"statement": "captured by the actor", "confidence": 0.9}
PAYLOAD_B = {"statement": "different statement"}


def _setup(client, actors=("org-1", "org-2"), digest=DIGEST_A):
    for index, actor_id in enumerate(actors):
        create_actor(
            client,
            actor_id=actor_id,
            name=f"Actor {index}",
            type="person" if index else "organization",
        )
    content = create_content(client, actor_id=actors[0], digest=digest)
    return content


def _create_claim(client, expected=201, **overrides):
    resp = client.post("/v1/claims", json=claim_payload(**overrides))
    assert resp.status_code == expected, resp.text
    return resp.json()


# --------------------------------------------------------------------------- #
# Normal behavior
# --------------------------------------------------------------------------- #


def test_create_claim_returns_full_public_fields(client):
    content = _setup(client)
    body = _create_claim(
        client,
        content_id=content["id"],
        actor_id="org-2",
        claim_type="attribution",
        payload=PAYLOAD_A,
    )
    assert set(body) == CLAIM_FIELDS
    assert body["id"].startswith("clm_") and len(body["id"]) == 68
    assert body["content_id"] == content["id"]
    assert body["actor_id"] == "org-2"
    assert body["claim_type"] == "attribution"
    assert body["payload_digest_algorithm"] == "sha256"
    # The digest is sha256 of the canonical payload, not of any echoed input.
    canonical = b'{"confidence":0.9,"statement":"captured by the actor"}'
    assert body["payload_digest"] == hashlib.sha256(canonical).hexdigest()
    # The raw payload is never echoed.
    assert "payload" not in body
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.tzinfo is not None
    assert created_at.utcoffset().total_seconds() == 0


def test_claiming_actor_need_not_equal_registering_actor(client):
    content = _setup(client)
    body = _create_claim(
        client,
        content_id=content["id"],
        actor_id="org-2",
        payload=PAYLOAD_A,
    )
    assert body["content_id"] == content["id"]
    assert body["actor_id"] == "org-2"
    assert content["actor_id"] == "org-1"


def test_registering_actor_may_also_claim(client):
    content = _setup(client)
    body = _create_claim(
        client, content_id=content["id"], actor_id="org-1", payload=PAYLOAD_A
    )
    assert body["actor_id"] == "org-1"


def test_empty_object_payload_is_allowed(client):
    content = _setup(client)
    body = _create_claim(
        client, content_id=content["id"], actor_id="org-1", payload={}
    )
    assert body["payload_digest"] == hashlib.sha256(b"{}").hexdigest()


def test_get_claim_returns_full_public_fields(client):
    content = _setup(client)
    created = _create_claim(
        client, content_id=content["id"], actor_id="org-2", payload=PAYLOAD_A
    )
    resp = client.get(f"/v1/claims/{created['id']}")
    assert resp.status_code == 200
    assert resp.json() == created
    assert set(resp.json()) == CLAIM_FIELDS


def test_list_claims_for_content_in_stable_creation_order(client):
    content = _setup(client)
    first = _create_claim(
        client, content_id=content["id"], actor_id="org-1",
        claim_type="attribution", payload=PAYLOAD_A,
    )
    second = _create_claim(
        client, content_id=content["id"], actor_id="org-2",
        claim_type="copyright", payload=PAYLOAD_B,
    )
    third = _create_claim(
        client, content_id=content["id"], actor_id="org-2",
        claim_type="attribution", payload=PAYLOAD_B,
    )
    resp = client.get(f"/v1/contents/{content['id']}/claims")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    assert [item["id"] for item in body["items"]] == [
        first["id"], second["id"], third["id"]
    ]
    assert all(set(item) == CLAIM_FIELDS for item in body["items"])
    assert all(item["content_id"] == content["id"] for item in body["items"])


# --------------------------------------------------------------------------- #
# Idempotency
# --------------------------------------------------------------------------- #


def test_identical_resubmission_returns_existing_claim(client):
    content = _setup(client)
    first = _create_claim(
        client, content_id=content["id"], actor_id="org-2",
        claim_type="attribution", payload=PAYLOAD_A,
    )
    resp = client.post(
        "/v1/claims",
        json=claim_payload(
            content_id=content["id"], actor_id="org-2",
            claim_type="attribution", payload=dict(reversed(list(PAYLOAD_A.items()))),
        ),
    )
    assert resp.status_code == 200
    assert resp.json() == first


def test_repeated_resubmissions_stay_idempotent(client, db_session):
    content = _setup(client)
    _create_claim(client, content_id=content["id"], payload=PAYLOAD_A)
    for _ in range(3):
        body = _create_claim(client, expected=200, content_id=content["id"], payload=PAYLOAD_A)
    rows = db_session.execute(select(Claim)).scalars().all()
    assert len(rows) == 1
    assert body["payload_digest"] == rows[0].payload_digest


def test_different_payloads_form_independent_claims(client):
    content = _setup(client)
    first = _create_claim(
        client, content_id=content["id"], actor_id="org-2",
        claim_type="attribution", payload=PAYLOAD_A,
    )
    second = _create_claim(
        client, content_id=content["id"], actor_id="org-2",
        claim_type="attribution", payload=PAYLOAD_B,
    )
    assert first["id"] != second["id"]
    assert first["payload_digest"] != second["payload_digest"]
    assert client.get(f"/v1/contents/{content['id']}/claims").json()["count"] == 2


def test_different_claim_types_form_independent_claims(client):
    content = _setup(client)
    first = _create_claim(
        client, content_id=content["id"], actor_id="org-2",
        claim_type="attribution", payload=PAYLOAD_A,
    )
    second = _create_claim(
        client, content_id=content["id"], actor_id="org-2",
        claim_type="copyright", payload=PAYLOAD_A,
    )
    assert first["id"] != second["id"]


def test_different_actors_form_independent_claims(client):
    content = _setup(client)
    first = _create_claim(
        client, content_id=content["id"], actor_id="org-1", payload=PAYLOAD_A
    )
    second = _create_claim(
        client, content_id=content["id"], actor_id="org-2", payload=PAYLOAD_A
    )
    assert first["id"] != second["id"]
    assert first["actor_id"] != second["actor_id"]


def test_different_content_forms_independent_claims(client):
    first_content = _setup(client, digest=DIGEST_A)
    second_content = create_content(client, actor_id="org-1", digest=DIGEST_B)
    first = _create_claim(
        client, content_id=first_content["id"], actor_id="org-2", payload=PAYLOAD_A
    )
    second = _create_claim(
        client, content_id=second_content["id"], actor_id="org-2", payload=PAYLOAD_A
    )
    assert first["id"] != second["id"]


def test_canonically_equivalent_payload_is_idempotent_across_wire_variations(client):
    content = _setup(client)
    first = _create_claim(
        client, content_id=content["id"], actor_id="org-2", payload={"v": 1}
    )
    # Whitespace, key order, and 1 vs 1.0 do not change the canonical payload.
    resp = client.post(
        "/v1/claims",
        content='{"content_id": "%s", "actor_id": "org-2", '
        '"claim_type": "attribution", "payload": {"v": 1.0}}' % content["id"],
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 200
    assert resp.json()["id"] == first["id"]


# --------------------------------------------------------------------------- #
# Isolation
# --------------------------------------------------------------------------- #


def test_listing_is_scoped_to_the_content(client):
    first_content = _setup(client, digest=DIGEST_A)
    second_content = create_content(client, actor_id="org-1", digest=DIGEST_B)
    c1_claim = _create_claim(
        client, content_id=first_content["id"], actor_id="org-2", payload=PAYLOAD_A
    )
    _create_claim(
        client, content_id=second_content["id"], actor_id="org-2", payload=PAYLOAD_A
    )
    resp = client.get(f"/v1/contents/{first_content['id']}/claims")
    body = resp.json()
    assert body["count"] == 1
    assert body["items"][0]["id"] == c1_claim["id"]


def test_content_without_claims_returns_empty_list(client):
    content = _setup(client)
    resp = client.get(f"/v1/contents/{content['id']}/claims")
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 0}


def test_claims_are_immutable_no_update_or_delete_routes(client):
    content = _setup(client)
    created = _create_claim(client, content_id=content["id"], payload=PAYLOAD_A)

    # There is no update or delete verb on the resource; existing claims must
    # survive resubmission attempts with different payloads unchanged.
    changed = client.post(
        "/v1/claims",
        json=claim_payload(
            content_id=content["id"], actor_id="org-1", payload=PAYLOAD_B
        ),
    )
    assert changed.status_code == 201
    assert changed.json()["id"] != created["id"]

    refetched = client.get(f"/v1/claims/{created['id']}")
    assert refetched.status_code == 200
    assert refetched.json() == created

    for method in ("put", "patch", "delete"):
        resp = getattr(client, method)(f"/v1/claims/{created['id']}")
        assert resp.status_code == 405, resp.text


# --------------------------------------------------------------------------- #
# Validation and missing-resource boundaries
# --------------------------------------------------------------------------- #


def test_unknown_content_on_create_is_content_not_found(client):
    create_actor(client)
    resp = client.post(
        "/v1/claims",
        json=claim_payload(content_id="cnt_does_not_exist", payload=PAYLOAD_A),
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "content_not_found"
    assert error["details"]["content_id"] == "cnt_does_not_exist"


def test_unknown_actor_on_create_is_unknown_actor(client):
    content = _setup(client)
    resp = client.post(
        "/v1/claims",
        json=claim_payload(
            content_id=content["id"], actor_id="ghost", payload=PAYLOAD_A
        ),
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "unknown_actor"
    assert error["details"]["actor_id"] == "ghost"


def test_unknown_claim_get_is_claim_not_found(client):
    resp = client.get("/v1/claims/clm_does_not_exist")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "claim_not_found"
    assert error["details"]["claim_id"] == "clm_does_not_exist"


def test_listing_claims_for_unknown_content_is_not_found(client):
    resp = client.get("/v1/contents/cnt_does_not_exist/claims")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "content_not_found"


def test_reject_non_object_payloads(client):
    content = _setup(client)
    for bad_payload in ([1, 2, 3], "a string", 42, 3.14, True, False, None):
        # Build the body directly: the helper treats ``payload=None`` as
        # "use the default", which would mask an explicit JSON null here.
        resp = client.post(
            "/v1/claims",
            json={
                "content_id": content["id"],
                "actor_id": "org-1",
                "claim_type": "attribution",
                "payload": bad_payload,
            },
        )
        assert resp.status_code == 422, bad_payload
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_blank_claim_type(client):
    content = _setup(client)
    for blank in ("", "   ", "\t\n"):
        resp = client.post(
            "/v1/claims",
            json=claim_payload(
                content_id=content["id"], claim_type=blank, payload=PAYLOAD_A
            ),
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_blank_content_and_actor_ids(client):
    content = _setup(client)
    resp = client.post(
        "/v1/claims",
        json=claim_payload(content_id="   ", payload=PAYLOAD_A),
    )
    assert resp.status_code == 422
    resp = client.post(
        "/v1/claims",
        json=claim_payload(
            content_id=content["id"], actor_id="  ", payload=PAYLOAD_A
        ),
    )
    assert resp.status_code == 422


def test_reject_missing_required_fields(client):
    resp = client.post("/v1/claims", json={})
    assert resp.status_code == 422
    issue_fields = {
        ".".join(part for part in issue["loc"] if part != "body")
        for issue in resp.json()["error"]["details"]["issues"]
    }
    assert {"content_id", "actor_id", "claim_type", "payload"}.issubset(
        issue_fields
    )


def test_reject_malformed_json_body(client):
    content = _setup(client)
    resp = client.post(
        "/v1/claims",
        content='{"content_id": "%s", payload: }' % content["id"],
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_reject_non_finite_numbers_in_payload(client):
    content = _setup(client)
    for token in ("NaN", "Infinity", "-Infinity"):
        resp = client.post(
            "/v1/claims",
            content=(
                '{"content_id": "%s", "actor_id": "org-1", '
                '"claim_type": "attribution", "payload": {"v": %s}}'
                % (content["id"], token)
            ),
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422, token
        assert resp.json()["error"]["code"] == "validation_error"


# --------------------------------------------------------------------------- #
# Audit semantics
# --------------------------------------------------------------------------- #


def _audit_rows(session):
    rows = session.execute(
        select(AuditEvent).order_by(AuditEvent.seq.asc())
    ).scalars().all()
    return [(r.event_type, r.resource_id, r.created_at) for r in rows]


def test_first_claim_creation_writes_claim_created_event(client, db_session):
    content = _setup(client)
    created = _create_claim(
        client, content_id=content["id"], actor_id="org-2", payload=PAYLOAD_A
    )
    events = _audit_rows(db_session)
    # actor.created, content.created, then exactly one claim.created.
    assert [event[0] for event in events].count(EVENT_CLAIM_CREATED) == 1
    event = events[-1]
    assert event[0] == EVENT_CLAIM_CREATED
    assert event[1] == created["id"]
    assert event[2].tzinfo is not None
    assert event[2].utcoffset().total_seconds() == 0
    # The audit row and the claim share the same UTC creation instant.
    claim = db_session.execute(
        select(Claim).where(Claim.id == created["id"])
    ).scalar_one()
    assert event[2] == claim.created_at


def test_duplicate_claim_adds_no_audit_event(client, db_session):
    content = _setup(client)
    _create_claim(client, content_id=content["id"], payload=PAYLOAD_A)
    events_after_create = len(_audit_rows(db_session))
    for _ in range(2):
        repeat = client.post(
            "/v1/claims",
            json=claim_payload(content_id=content["id"], payload=PAYLOAD_A),
        )
        assert repeat.status_code == 200
    assert len(_audit_rows(db_session)) == events_after_create


def test_failed_claim_requests_write_no_audit_events(client, db_session):
    content = _setup(client)
    base = len(_audit_rows(db_session))

    client.post(
        "/v1/claims",
        json=claim_payload(content_id="cnt_ghost", payload=PAYLOAD_A),
    )
    client.post(
        "/v1/claims",
        json=claim_payload(
            content_id=content["id"], actor_id="ghost", payload=PAYLOAD_A
        ),
    )
    client.post(
        "/v1/claims",
        json=claim_payload(content_id=content["id"], claim_type=" ", payload={}),
    )
    assert len(_audit_rows(db_session)) == base


def test_claim_and_audit_event_commit_together(tmp_db_url):
    # File-backed DB: after the request completes, claim + audit event are
    # both visible from a brand-new app/engine (a single committed txn).
    app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app) as local_client:
        content = _setup(local_client)
        created = _create_claim(
            local_client, content_id=content["id"], payload=PAYLOAD_A
        )

    path = tmp_db_url.removeprefix("sqlite:///")
    con = sqlite3.connect(path)
    try:
        claim_count = con.execute(
            "SELECT COUNT(*) FROM claims WHERE id = ?", (created["id"],)
        ).fetchone()[0]
        audit_count = con.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = ? AND resource_id = ?",
            (EVENT_CLAIM_CREATED, created["id"]),
        ).fetchone()[0]
        columns = {
            row[1]
            for row in con.execute("PRAGMA table_info(claims)")
        }
    finally:
        con.close()
    assert claim_count == 1
    assert audit_count == 1
    # The stored columns describe the digest, never a raw payload.
    assert "payload" not in columns
    assert {"payload_digest_algorithm", "payload_digest"}.issubset(columns)


def test_raw_payload_is_not_persisted(tmp_db_url):
    # A fresh file-backed database: if the raw payload were stored, its bytes
    # would appear somewhere in the SQLite file (no encryption is in use).
    app = create_app(Settings(database_url=tmp_db_url))
    secret = "super-secret-statement-value-xyz"
    with TestClient(app) as local_client:
        content = _setup(local_client)
        _create_claim(
            local_client,
            content_id=content["id"],
            payload={"statement": secret, "nested": {"also_secret": secret}},
        )
    path = tmp_db_url.removeprefix("sqlite:///")
    # Checkpoint/flush is complete once every connection is closed.
    app.state.engine.dispose()
    with open(path, "rb") as handle:
        db_bytes = handle.read()
    assert secret.encode("utf-8") not in db_bytes


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


def test_concurrent_identical_claims_create_exactly_one(tmp_db_url):
    # Two simultaneous first-submissions of the same identity must end with a
    # single claim: one wins the insert, the other loses the unique-constraint
    # race and returns the existing claim, still 200 at the HTTP layer.
    from threading import Barrier, Thread

    from provenance import service
    from provenance.database import make_engine, make_session_factory
    from provenance.schemas import ClaimCreate

    app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app) as local_client:
        content = _setup(local_client)

    engine = make_engine(tmp_db_url)
    factory = make_session_factory(engine)
    barrier = Barrier(2)
    results: list[tuple[str, str]] = []

    def submit() -> None:
        session = factory()
        try:
            barrier.wait()
            claim, created = service.create_claim(
                session,
                ClaimCreate(
                    content_id=content["id"],
                    actor_id="org-2",
                    claim_type="attribution",
                    payload=PAYLOAD_A,
                ),
            )
            results.append((claim.id, "created" if created else "existing"))
        finally:
            session.close()

    threads = [Thread(target=submit) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert len(results) == 2
    assert {result[0] for result in results} == {results[0][0]}
    assert sorted(result[1] for result in results) == ["created", "existing"]

    with TestClient(create_app(Settings(database_url=tmp_db_url))) as client:
        listing = client.get(f"/v1/contents/{content['id']}/claims").json()
        assert listing["count"] == 1

    audit = engine.connect()
    try:
        from sqlalchemy import text as sql_text

        count = audit.execute(
            sql_text(
                "SELECT COUNT(*) FROM audit_events "
                "WHERE event_type = 'claim.created'"
            )
        ).scalar_one()
    finally:
        audit.close()
        engine.dispose()
    assert count == 1


# --------------------------------------------------------------------------- #
# Persistence across restarts
# --------------------------------------------------------------------------- #

def test_claims_persist_across_app_restarts(tmp_db_url, file_client):
    content = _setup(file_client)
    created = _create_claim(
        file_client, content_id=content["id"], actor_id="org-2", payload=PAYLOAD_A
    )

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        detail = client.get(f"/v1/claims/{created['id']}")
        assert detail.status_code == 200
        assert detail.json() == created

        listing = client.get(f"/v1/contents/{content['id']}/claims")
        assert listing.json()["count"] == 1
        assert listing.json()["items"][0]["id"] == created["id"]

        # Idempotency survives the restart: no duplicate claim or event.
        repeat = client.post(
            "/v1/claims",
            json=claim_payload(
                content_id=content["id"], actor_id="org-2", payload=PAYLOAD_A
            ),
        )
        assert repeat.status_code == 200
        assert repeat.json()["id"] == created["id"]
