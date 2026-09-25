"""Tests for read-only trust-policy retrieval.

Covers ``GET /v1/trust-policies``: the empty-body contract, the exact
``items``/``count``/``next_cursor`` member order in compact UTF-8 JSON
terminated by one newline, the optional exact ``actor_id`` filter (an
unknown subject is an empty collection, never a 404), stable creation
ordering, pure-decimal ``limit`` validation, the opaque HMAC-signed
``tp1`` cursor family (binding the effective filter and limit, resuming
without duplication or omission, a null cursor on the final page, and an
empty page with the original count at or past the tail), every 422
validation boundary, and the strictly read-only guarantee (no policy,
resource, or audit writes on success, empty results, repeated reads, or
failures).

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures used to register the policies).
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone

from sqlalchemy import select

from provenance import pagination
from provenance.access_signing import access_message_bytes
from provenance.models import ActorTrustPolicy, AuditEvent
from provenance.signing import attestation_message_bytes
from tests.helpers import (
    content_payload,
    create_actor,
    ed25519_public_key,
    ed25519_sign,
    SEED_A,
    SEED_B,
)

POLICIES_PATH = "/v1/trust-policies"

SEED_C = b"test-ed25519-seed-c-00000000000000"[:32]


# --- Fixture-style setup ------------------------------------------------------


def _make_claim(client, actor_id, digest):
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


def _signed_headers(method, path, body, *, actor, seed):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    body_digest = hashlib.sha256(body).hexdigest()
    message = access_message_bytes(method, path, ts, body_digest)
    signature = base64.b64encode(ed25519_sign(seed, message)).decode("ascii")
    return {"X-PA": actor, "X-PT": ts, "X-PS": signature}


def _register_policy(client, actor_id, threshold, seed):
    """Create the actor (with a current key) and register its policy."""
    create_actor(client, actor_id=actor_id, name=f"Org {actor_id}")
    digest = hashlib.sha256(f"content-{actor_id}".encode()).hexdigest()
    _make_attestation(
        client, actor_id, seed, _make_claim(client, actor_id, digest)["id"]
    )
    body = json.dumps({"actor_id": actor_id, "threshold": threshold}).encode()
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", POLICIES_PATH, body, actor=actor_id, seed=seed),
    }
    resp = client.post(POLICIES_PATH, content=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _world(client):
    """Register three policies for three distinct subjects, in order."""
    one = _register_policy(client, "org-1", 2, SEED_A)
    two = _register_policy(client, "org-2", 3, SEED_B)
    three = _register_policy(client, "org-3", 1, SEED_C)
    return [one, two, three]


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


def _policy_rows(session):
    return session.execute(select(ActorTrustPolicy)).scalars().all()


# --- Empty collection and response shape ---------------------------------------


def test_empty_registry_is_an_empty_collection(client):
    resp = client.get(POLICIES_PATH)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_response_is_compact_json_with_ordered_members_and_one_newline(client):
    _world(client)
    resp = client.get(POLICIES_PATH)
    assert resp.status_code == 200
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    # The top-level members appear in exactly this order.
    assert raw.startswith(b'{"items":[')
    assert b'],"count":3,"next_cursor":null}' in raw
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    assert raw == expected


def test_items_carry_exactly_the_policy_public_view(client):
    policies = _world(client)
    resp = client.get(POLICIES_PATH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    assert body["next_cursor"] is None
    # Stable creation order, and each item is exactly the public view.
    assert [item["id"] for item in body["items"]] == [
        policy["id"] for policy in policies
    ]
    for item, policy in zip(body["items"], policies):
        assert list(item) == ["id", "actor_id", "threshold", "enabled", "created_at"]
        assert item == policy
        assert item["id"].startswith("atp_")
        assert item["enabled"] is True
        created_at = datetime.fromisoformat(item["created_at"])
        assert created_at.utcoffset().total_seconds() == 0
        assert item["created_at"].endswith(("Z", "+00:00"))


# --- actor_id filtering ---------------------------------------------------------


def test_actor_id_filter_is_an_exact_match(client):
    policies = _world(client)
    resp = client.get(POLICIES_PATH, params={"actor_id": "org-2"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["next_cursor"] is None
    assert [item["id"] for item in body["items"]] == [policies[1]["id"]]


def test_actor_id_filter_is_case_and_whitespace_sensitive(client):
    _world(client)
    for spelling in ("Org-1", "org-1 ", " org-1"):
        resp = client.get(POLICIES_PATH, params={"actor_id": spelling})
        assert resp.status_code == 200, spelling
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_unknown_subject_is_an_empty_collection_not_a_404(client):
    _world(client)
    resp = client.get(POLICIES_PATH, params={"actor_id": "ghost"})
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_blank_actor_id_is_422(client):
    _world(client)
    for blank in ("", "   "):
        resp = client.get(POLICIES_PATH, params={"actor_id": blank})
        assert resp.status_code == 422, repr(blank)
        assert resp.json()["error"]["code"] == "validation_error"


# --- limit validation -----------------------------------------------------------


def test_limit_boundaries_one_and_one_hundred_are_accepted(client):
    _world(client)
    assert client.get(POLICIES_PATH, params={"limit": "1"}).status_code == 200
    assert client.get(POLICIES_PATH, params={"limit": "100"}).status_code == 200


def test_limit_must_be_a_pure_decimal_integer_in_range(client):
    _world(client)
    for bad in ("0", "101", "-1", "5.0", "5e0", " 5", "5 ", "five", "+5"):
        resp = client.get(POLICIES_PATH, params={"limit": bad})
        assert resp.status_code == 422, bad
        assert resp.json()["error"]["code"] == "validation_error", bad


def test_default_limit_is_fifty(client, app):
    # 51 policies exceed the default page: the first page carries 50 items.
    for index in range(51):
        actor_id = f"org-bulk-{index:03d}"
        seed = hashlib.sha256(actor_id.encode()).digest()
        _register_policy(client, actor_id, 1, seed)
    resp = client.get(POLICIES_PATH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 51
    assert len(body["items"]) == 50
    assert body["next_cursor"] is not None


# --- Parameter strictness -------------------------------------------------------


def test_repeated_and_undeclared_parameters_are_422(client):
    _world(client)
    repeated = client.get(
        POLICIES_PATH, params=[("actor_id", "org-1"), ("actor_id", "org-2")]
    )
    assert repeated.status_code == 422
    repeated_limit = client.get(
        POLICIES_PATH, params=[("limit", "1"), ("limit", "2")]
    )
    assert repeated_limit.status_code == 422
    for params in ({"actor_ids": "org-1"}, {"threshold": "2"}, {"offset": "1"}):
        resp = client.get(POLICIES_PATH, params=params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_empty_body_is_422_rejected_before_any_read(client, db_session):
    _world(client)
    events_before = _audit_count(db_session)
    for body in (b"{}", b" ", b"{not valid json", b"[]"):
        resp = client.request(
            "GET",
            POLICIES_PATH,
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"
    assert _audit_count(db_session) == events_before


# --- Pagination -----------------------------------------------------------------


def test_pages_resume_without_duplication_or_omission(client):
    policies = _world(client)
    first = client.get(POLICIES_PATH, params={"limit": "2"})
    assert first.status_code == 200
    page_one = first.json()
    assert page_one["count"] == 3
    assert [item["id"] for item in page_one["items"]] == [
        policies[0]["id"],
        policies[1]["id"],
    ]
    cursor = page_one["next_cursor"]
    assert cursor is not None

    second = client.get(POLICIES_PATH, params={"limit": "2", "cursor": cursor})
    assert second.status_code == 200
    page_two = second.json()
    # The filtered total covers every page, including the final one.
    assert page_two["count"] == 3
    assert [item["id"] for item in page_two["items"]] == [policies[2]["id"]]
    assert page_two["next_cursor"] is None

    seen = [item["id"] for item in page_one["items"] + page_two["items"]]
    assert seen == [policy["id"] for policy in policies]


def test_filtered_pages_bind_the_actor_filter(client):
    _world(client)
    # A second policy for the same subject is impossible, so filter paging
    # is exercised with a single matching policy and a tiny limit.
    resp = client.get(POLICIES_PATH, params={"actor_id": "org-1", "limit": "1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert len(body["items"]) == 1
    assert body["next_cursor"] is None


def test_replayed_cursor_returns_the_same_page(client):
    _world(client)
    first = client.get(POLICIES_PATH, params={"limit": "1"}).json()
    cursor = first["next_cursor"]
    assert cursor is not None
    page = client.get(POLICIES_PATH, params={"limit": "1", "cursor": cursor})
    replay = client.get(POLICIES_PATH, params={"limit": "1", "cursor": cursor})
    assert page.status_code == 200
    assert replay.status_code == 200
    assert replay.content == page.content


def test_cursor_at_or_past_the_tail_returns_empty_page_with_count(client, app):
    _world(client)
    cursor = pagination.encode_typed_cursor(
        app.state.trust_policies_cursor_secret,
        pagination.TRUST_POLICIES_CURSOR,
        {"actor_id": None, "limit": 50, "offset": 3},
    )
    resp = client.get(POLICIES_PATH, params={"cursor": cursor})
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 3, "next_cursor": None}

    past = pagination.encode_typed_cursor(
        app.state.trust_policies_cursor_secret,
        pagination.TRUST_POLICIES_CURSOR,
        {"actor_id": None, "limit": 50, "offset": 99},
    )
    resp = client.get(POLICIES_PATH, params={"cursor": past})
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 3, "next_cursor": None}


def test_cursor_binds_the_effective_filter_and_limit(client):
    _world(client)
    cursor = client.get(POLICIES_PATH, params={"limit": "2"}).json()["next_cursor"]
    assert cursor is not None
    # A different limit, a new filter, or a dropped filter all mismatch.
    for params in (
        {"limit": "3", "cursor": cursor},
        {"actor_id": "org-1", "limit": "2", "cursor": cursor},
        {"cursor": cursor},
    ):
        resp = client.get(POLICIES_PATH, params=params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_blank_malformed_and_tampered_cursors_are_422(client):
    _world(client)
    valid = client.get(POLICIES_PATH, params={"limit": "1"}).json()["next_cursor"]
    assert valid is not None
    tampered = valid[:-1] + ("A" if valid[-1] != "A" else "B")
    for bad in ("", "   ", "not-a-cursor", tampered):
        resp = client.get(POLICIES_PATH, params={"cursor": bad})
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_foreign_family_cursor_is_422(client, app):
    _world(client)
    # A well-formed cursor minted by another endpoint family never resumes
    # this retrieval.
    foreign = pagination.encode_typed_cursor(
        app.state.claims_cursor_secret,
        pagination.CLAIMS_CURSOR,
        {
            "content_id": None,
            "actor_id": None,
            "claim_type": None,
            "payload_digest_hex": None,
            "limit": 50,
            "offset": 1,
        },
    )
    resp = client.get(POLICIES_PATH, params={"cursor": foreign})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only guarantee ---------------------------------------------------------


def test_queries_and_failures_write_nothing(client, db_session):
    _world(client)
    events_before = _audit_count(db_session)
    rows_before = [r.id for r in _policy_rows(db_session)]

    assert client.get(POLICIES_PATH).status_code == 200
    assert client.get(POLICIES_PATH, params={"actor_id": "ghost"}).status_code == 200
    assert client.get(POLICIES_PATH, params={"limit": "1"}).status_code == 200
    assert client.get(POLICIES_PATH, params={"limit": "0"}).status_code == 422
    assert client.get(POLICIES_PATH, params={"cursor": "bad"}).status_code == 422

    assert [r.id for r in _policy_rows(db_session)] == rows_before
    assert _audit_count(db_session) == events_before


def test_policies_list_identically_across_an_app_restart(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    _world(file_client)
    first = file_client.get(POLICIES_PATH)
    assert first.status_code == 200

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as restarted_client:
        second = restarted_client.get(POLICIES_PATH)
        assert second.status_code == 200
        assert second.json() == first.json()
