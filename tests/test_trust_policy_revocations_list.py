"""Tests for read-only trust-policy revocation retrieval.

Covers ``GET /v1/trust-policy-revocations``: the empty-body contract, the
exact ``items``/``count``/``next_cursor`` member order in compact UTF-8 JSON
terminated by one newline, each item's exact public view in id, policy_id,
actor_id, reason, created_at order (the stable ``tpr_`` id and the reason
byte-for-byte as stored), the optional exact ``policy_id``/``actor_id``
filters (case- and whitespace-sensitive, combined as AND; an unknown policy
or subject is an empty collection, never a 404), stable created_at plus
insertion ordering, pure-decimal ``limit`` validation (1..100, default 50),
the opaque HMAC-signed ``tpr1`` cursor family (binding the effective
filters -- null included -- and limit, resuming without duplication or
omission, a null cursor on the final page, and an empty page with the
original count at or past the tail), every 422 validation boundary
(blank/null/repeated/undeclared parameters, non-empty bodies,
blank/tampered/foreign-family/mismatched cursors), the 405 rejection of
PUT/PATCH/DELETE while POST stays the revocation creation route, and the
strictly read-only guarantee (no revocation, policy, resource, or audit
writes on success, empty results, repeated and concurrent reads, failures,
or across a restart).

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures used to register and revoke the policies).
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
from datetime import datetime, timezone

from sqlalchemy import select

from provenance import pagination
from provenance.access_signing import access_message_bytes
from provenance.models import ActorTrustPolicyRevocation, AuditEvent
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
REVOCATIONS_PATH = "/v1/trust-policy-revocations"

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


def _revoke(client, policy_id, *, actor, seed, reason):
    body = json.dumps({"policy_id": policy_id, "reason": reason}).encode()
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", REVOCATIONS_PATH, body, actor=actor, seed=seed),
    }
    resp = client.post(REVOCATIONS_PATH, content=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _register_revocation(client, actor_id, seed, reason, threshold=1):
    policy = _register_policy(client, actor_id, threshold, seed)
    return _revoke(
        client, policy["id"], actor=actor_id, seed=seed, reason=reason
    )


def _world(client):
    """Create three revocations for three distinct subjects, in order."""
    one = _register_revocation(client, "org-1", SEED_A, "first rationale")
    two = _register_revocation(client, "org-2", SEED_B, "  spaced reason  ")
    three = _register_revocation(client, "org-3", SEED_C, "撤回:third 🔒")
    return [one, two, three]


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


def _revocation_rows(session):
    return session.execute(
        select(ActorTrustPolicyRevocation)
    ).scalars().all()


# --- Empty collection and response shape ---------------------------------------


def test_empty_registry_is_an_empty_collection(client):
    resp = client.get(REVOCATIONS_PATH)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_response_is_compact_json_with_ordered_members_and_one_newline(client):
    _world(client)
    resp = client.get(REVOCATIONS_PATH)
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


def test_items_carry_exactly_the_revocation_public_view_verbatim(client):
    revocations = _world(client)
    resp = client.get(REVOCATIONS_PATH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    assert body["next_cursor"] is None
    # Stable created_at + insertion order, each item exactly the public view.
    assert [item["id"] for item in body["items"]] == [
        revocation["id"] for revocation in revocations
    ]
    for item, revocation in zip(body["items"], revocations):
        assert list(item) == [
            "id",
            "policy_id",
            "actor_id",
            "reason",
            "created_at",
        ]
        assert item == revocation
        assert item["id"].startswith("tpr_")
        assert len(item["id"]) == len("tpr_") + 64
        created_at = datetime.fromisoformat(item["created_at"])
        assert created_at.utcoffset().total_seconds() == 0
        assert item["created_at"].endswith(("Z", "+00:00"))

    # The reason is returned byte-for-byte as stored: surrounding whitespace
    # and non-ASCII content are neither trimmed, escaped beyond JSON, nor
    # rewritten.
    assert body["items"][1]["reason"] == "  spaced reason  "
    assert body["items"][2]["reason"] == "撤回:third 🔒"


# --- Exact filtering ------------------------------------------------------------


def test_policy_id_filter_is_an_exact_match(client):
    revocations = _world(client)
    resp = client.get(
        REVOCATIONS_PATH, params={"policy_id": revocations[1]["policy_id"]}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["next_cursor"] is None
    assert [item["id"] for item in body["items"]] == [revocations[1]["id"]]


def test_actor_id_filter_is_an_exact_match(client):
    revocations = _world(client)
    resp = client.get(REVOCATIONS_PATH, params={"actor_id": "org-2"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["next_cursor"] is None
    assert [item["id"] for item in body["items"]] == [revocations[1]["id"]]


def test_filters_are_case_and_whitespace_sensitive(client):
    revocations = _world(client)
    for params in (
        {"policy_id": revocations[0]["policy_id"].upper()},
        {"policy_id": revocations[0]["policy_id"] + " "},
        {"actor_id": "Org-1"},
        {"actor_id": "org-1 "},
        {"actor_id": " org-1"},
    ):
        resp = client.get(REVOCATIONS_PATH, params=params)
        assert resp.status_code == 200, params
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_policy_and_actor_filters_combine_as_logical_and(client):
    revocations = _world(client)
    # Matching pair.
    matched = client.get(
        REVOCATIONS_PATH,
        params={
            "policy_id": revocations[0]["policy_id"],
            "actor_id": "org-1",
        },
    )
    assert matched.status_code == 200
    assert [item["id"] for item in matched.json()["items"]] == [
        revocations[0]["id"]
    ]
    # An existing policy paired with another subject matches nothing.
    mismatched = client.get(
        REVOCATIONS_PATH,
        params={
            "policy_id": revocations[0]["policy_id"],
            "actor_id": "org-2",
        },
    )
    assert mismatched.status_code == 200
    assert mismatched.json() == {
        "items": [],
        "count": 0,
        "next_cursor": None,
    }


def test_unknown_policy_or_subject_is_an_empty_collection_not_a_404(client):
    _world(client)
    for params in (
        {"policy_id": "atp_" + "0" * 64},
        {"policy_id": "tpr_" + "0" * 64},
        {"actor_id": "ghost"},
    ):
        resp = client.get(REVOCATIONS_PATH, params=params)
        assert resp.status_code == 200, params
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_blank_filters_are_422(client):
    _world(client)
    for field in ("policy_id", "actor_id"):
        for blank in ("", "   ", "\n\t "):
            resp = client.get(REVOCATIONS_PATH, params={field: blank})
            assert resp.status_code == 422, (field, repr(blank))
            assert resp.json()["error"]["code"] == "validation_error"


# --- limit validation -----------------------------------------------------------


def test_limit_boundaries_one_and_one_hundred_are_accepted(client):
    _world(client)
    assert client.get(REVOCATIONS_PATH, params={"limit": "1"}).status_code == 200
    assert (
        client.get(REVOCATIONS_PATH, params={"limit": "100"}).status_code == 200
    )


def test_limit_must_be_a_pure_decimal_integer_in_range(client):
    _world(client)
    for bad in ("0", "101", "-1", "5.0", "5e0", " 5", "5 ", "five", "+5"):
        resp = client.get(REVOCATIONS_PATH, params={"limit": bad})
        assert resp.status_code == 422, bad
        assert resp.json()["error"]["code"] == "validation_error", bad


def test_default_limit_is_fifty(client):
    # 51 revocations exceed the default page: the first page carries 50.
    for index in range(51):
        actor_id = f"org-bulk-{index:03d}"
        seed = hashlib.sha256(actor_id.encode()).digest()
        _register_revocation(client, actor_id, seed, f"reason-{index}")
    resp = client.get(REVOCATIONS_PATH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 51
    assert len(body["items"]) == 50
    assert body["next_cursor"] is not None


# --- Parameter strictness -------------------------------------------------------


def test_repeated_and_undeclared_parameters_are_422(client):
    _world(client)
    for repeated in (
        [("policy_id", "atp_a"), ("policy_id", "atp_b")],
        [("actor_id", "org-1"), ("actor_id", "org-2")],
        [("limit", "1"), ("limit", "2")],
        [("cursor", "a"), ("cursor", "b")],
    ):
        resp = client.get(REVOCATIONS_PATH, params=repeated)
        assert resp.status_code == 422, repeated
        assert resp.json()["error"]["code"] == "validation_error"
    for params in (
        {"policy_ids": "atp_x"},
        {"reason": "first rationale"},
        {"offset": "1"},
        {"revoker_actor_id": "org-1"},
    ):
        resp = client.get(REVOCATIONS_PATH, params=params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_empty_body_is_422_rejected_before_any_read(client, db_session):
    _world(client)
    events_before = _audit_count(db_session)
    rows_before = [r.id for r in _revocation_rows(db_session)]
    for body in (b"{}", b" ", b"{not valid json", b"[]", b"\x00\xff"):
        resp = client.request(
            "GET",
            REVOCATIONS_PATH,
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"
    assert _audit_count(db_session) == events_before
    assert [r.id for r in _revocation_rows(db_session)] == rows_before


# --- Pagination -----------------------------------------------------------------


def test_pages_resume_without_duplication_or_omission(client):
    revocations = _world(client)
    first = client.get(REVOCATIONS_PATH, params={"limit": "2"})
    assert first.status_code == 200
    page_one = first.json()
    assert page_one["count"] == 3
    assert [item["id"] for item in page_one["items"]] == [
        revocations[0]["id"],
        revocations[1]["id"],
    ]
    cursor = page_one["next_cursor"]
    assert cursor is not None
    assert cursor.startswith("tpr1.")

    second = client.get(REVOCATIONS_PATH, params={"limit": "2", "cursor": cursor})
    assert second.status_code == 200
    page_two = second.json()
    # The filtered total covers every page, including the final one.
    assert page_two["count"] == 3
    assert [item["id"] for item in page_two["items"]] == [revocations[2]["id"]]
    assert page_two["next_cursor"] is None

    seen = [item["id"] for item in page_one["items"] + page_two["items"]]
    assert seen == [revocation["id"] for revocation in revocations]


def test_filtered_pages_bind_both_filters(client):
    revocations = _world(client)
    resp = client.get(
        REVOCATIONS_PATH,
        params={
            "policy_id": revocations[1]["policy_id"],
            "actor_id": "org-2",
            "limit": "1",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["id"] == revocations[1]["id"]
    assert body["next_cursor"] is None


def test_replayed_cursor_returns_the_same_page(client):
    _world(client)
    first = client.get(REVOCATIONS_PATH, params={"limit": "1"}).json()
    cursor = first["next_cursor"]
    assert cursor is not None
    page = client.get(REVOCATIONS_PATH, params={"limit": "1", "cursor": cursor})
    replay = client.get(
        REVOCATIONS_PATH, params={"limit": "1", "cursor": cursor}
    )
    assert page.status_code == 200
    assert replay.status_code == 200
    assert replay.content == page.content


def test_cursor_at_or_past_the_tail_returns_empty_page_with_count(client, app):
    _world(client)
    cursor = pagination.encode_typed_cursor(
        app.state.trust_policy_revocations_cursor_secret,
        pagination.TRUST_POLICY_REVOCATIONS_CURSOR,
        {"policy_id": None, "actor_id": None, "limit": 50, "offset": 3},
    )
    resp = client.get(REVOCATIONS_PATH, params={"cursor": cursor})
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 3, "next_cursor": None}

    past = pagination.encode_typed_cursor(
        app.state.trust_policy_revocations_cursor_secret,
        pagination.TRUST_POLICY_REVOCATIONS_CURSOR,
        {"policy_id": None, "actor_id": None, "limit": 50, "offset": 99},
    )
    resp = client.get(REVOCATIONS_PATH, params={"cursor": past})
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 3, "next_cursor": None}


def test_cursor_binds_the_effective_filters_and_limit(client):
    _world(client)
    cursor = client.get(REVOCATIONS_PATH, params={"limit": "2"}).json()[
        "next_cursor"
    ]
    assert cursor is not None
    # A different limit, a new filter, a dropped implicit filter, and a
    # swapped filter all mismatch.
    for params in (
        {"limit": "3", "cursor": cursor},
        {"actor_id": "org-1", "limit": "2", "cursor": cursor},
        {"policy_id": "atp_x", "limit": "2", "cursor": cursor},
        {"cursor": cursor},
    ):
        resp = client.get(REVOCATIONS_PATH, params=params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"

    # A cursor minted under the actor filter does not resume the same page
    # when the filter value changes (only one revocation matches, so no
    # cursor is ever issued for that tiny page; mint one directly).
    app = client.app
    filtered_cursor = pagination.encode_typed_cursor(
        app.state.trust_policy_revocations_cursor_secret,
        pagination.TRUST_POLICY_REVOCATIONS_CURSOR,
        {"policy_id": None, "actor_id": "org-1", "limit": 1, "offset": 1},
    )
    for params in (
        {"actor_id": "org-2", "limit": "1", "cursor": filtered_cursor},
        {"limit": "1", "cursor": filtered_cursor},
    ):
        resp = client.get(REVOCATIONS_PATH, params=params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_blank_malformed_and_tampered_cursors_are_422(client):
    _world(client)
    valid = client.get(REVOCATIONS_PATH, params={"limit": "1"}).json()[
        "next_cursor"
    ]
    assert valid is not None
    tampered = valid[:-1] + ("A" if valid[-1] != "A" else "B")
    for bad in ("", "   ", "not-a-cursor", tampered):
        resp = client.get(REVOCATIONS_PATH, params={"cursor": bad})
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_foreign_family_cursors_are_422(client, app):
    _world(client)
    # The trust-policy retrieval family (tp1) and an unrelated family never
    # resume this retrieval, even though all families share the server.
    trust_policies_cursor = pagination.encode_typed_cursor(
        app.state.trust_policies_cursor_secret,
        pagination.TRUST_POLICIES_CURSOR,
        {"actor_id": None, "limit": 50, "offset": 1},
    )
    claims_cursor = pagination.encode_typed_cursor(
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
    for foreign in (trust_policies_cursor, claims_cursor):
        resp = client.get(REVOCATIONS_PATH, params={"cursor": foreign})
        assert resp.status_code == 422, foreign
        assert resp.json()["error"]["code"] == "validation_error"


# --- Method boundary ------------------------------------------------------------


def test_put_patch_and_delete_are_405_method_not_allowed(client):
    _world(client)
    for method in (client.put, client.patch, client.delete):
        resp = method(REVOCATIONS_PATH)
        assert resp.status_code == 405
        assert resp.json()["error"]["code"] == "method_not_allowed"


def test_post_remains_the_revocation_creation_route(client, app):
    # The new read entry does not shadow or alter revocation creation.
    created = _register_revocation(
        client, "org-1", SEED_A, "created after reads began"
    )
    resp = client.get(REVOCATIONS_PATH)
    assert resp.status_code == 200
    assert [item["id"] for item in resp.json()["items"]] == [created["id"]]

    # Creation still wrote exactly one revocation row and one revoked audit
    # event (checked from a fresh session, since the creation commits in
    # another connection).
    from provenance.models import EVENT_ACTOR_TRUST_POLICY_REVOKED

    session = app.state.session_factory()
    try:
        assert [r.id for r in _revocation_rows(session)] == [created["id"]]
        assert (
            len(
                session.execute(
                    select(AuditEvent).where(
                        AuditEvent.event_type
                        == EVENT_ACTOR_TRUST_POLICY_REVOKED
                    )
                ).scalars().all()
            )
            == 1
        )
    finally:
        session.close()


# --- Read-only guarantee ---------------------------------------------------------


def test_queries_and_failures_write_nothing(client, db_session):
    revocations = _world(client)
    events_before = _audit_count(db_session)
    rows_before = [r.id for r in _revocation_rows(db_session)]

    assert client.get(REVOCATIONS_PATH).status_code == 200
    assert (
        client.get(REVOCATIONS_PATH, params={"actor_id": "ghost"}).status_code
        == 200
    )
    assert (
        client.get(
            REVOCATIONS_PATH,
            params={"policy_id": revocations[0]["policy_id"]},
        ).status_code
        == 200
    )
    assert client.get(REVOCATIONS_PATH, params={"limit": "1"}).status_code == 200
    assert client.get(REVOCATIONS_PATH, params={"limit": "0"}).status_code == 422
    assert (
        client.get(REVOCATIONS_PATH, params={"cursor": "bad"}).status_code
        == 422
    )

    assert [r.id for r in _revocation_rows(db_session)] == rows_before
    assert _audit_count(db_session) == events_before


def test_concurrent_reads_return_identical_pages_and_cursors(client):
    _world(client)
    seen: list[bytes] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(6)

    def worker() -> None:
        try:
            barrier.wait()
            resp = client.get(REVOCATIONS_PATH, params={"limit": "2"})
            assert resp.status_code == 200
            seen.append(resp.content)
        except Exception as exc:  # pragma: no cover - fails the test below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(seen) == 6
    assert all(payload == seen[0] for payload in seen)


def test_revocation_reads_are_identical_across_an_app_restart(
    tmp_db_url, file_client
):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    _world(file_client)
    first = file_client.get(REVOCATIONS_PATH, params={"limit": "2"})
    assert first.status_code == 200
    first_page = first.json()
    assert first_page["count"] == 3
    assert len(first_page["items"]) == 2

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as restarted_client:
        second = restarted_client.get(REVOCATIONS_PATH, params={"limit": "2"})
        assert second.status_code == 200
        second_page = second.json()
        # The items and the filtered total are restart-stable; the cursor
        # token itself is process-local and need not survive the restart.
        assert second_page["items"] == first_page["items"]
        assert second_page["count"] == first_page["count"]
        assert second_page["next_cursor"] is not None
