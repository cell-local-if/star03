"""Tests for read-only trust-policy-revocation retrieval.

Covers ``GET /v1/trust-policy-revocations``: the empty-body contract, the
exact ``items``/``count``/``next_cursor`` member order in compact UTF-8 JSON
terminated by one newline, the per-item member order (``id``, ``policy_id``,
``actor_id``, ``reason``, ``created_at``) with the reason returned verbatim,
the optional exact ``policy_id``/``actor_id`` filters (combined as logical
AND; unknown values are an empty collection, never a 404), stable creation
ordering, pure-decimal ``limit`` validation, the opaque HMAC-signed ``tr1``
cursor family (binding the effective filters and limit, resuming without
duplication or omission, a null cursor on the final page, and an empty page
with the original count at or past the tail), every 422 validation boundary,
the 405 rejection of PUT/PATCH/DELETE (POST remains the creation route),
and the strictly read-only guarantee (no revocation, policy, resource, or
audit writes on success, empty results, repeated reads, or failures).

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures used to register the policies and revocations).
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
from datetime import datetime, timezone

from sqlalchemy import func, select

from provenance import pagination
from provenance.access_signing import access_message_bytes
from provenance.models import (
    ActorTrustPolicy,
    ActorTrustPolicyRevocation,
    AuditEvent,
)
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

REASON_A = "key material retired"
#: Verbatim evidence: surrounding whitespace and non-ASCII text are stored
#: and returned byte-exact, never trimmed or rewritten.
REASON_B = "  rotated\tsigning keys — ünïcode  "
REASON_C = "policy no longer relied upon"


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


def _revoke(client, policy_id, reason, *, actor_id, seed):
    body = json.dumps({"policy_id": policy_id, "reason": reason}).encode()
    headers = {
        "Content-Type": "application/json",
        **_signed_headers(
            "POST", REVOCATIONS_PATH, body, actor=actor_id, seed=seed
        ),
    }
    resp = client.post(REVOCATIONS_PATH, content=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _world(client):
    """Three policies for three subjects, each revoked in order."""
    one = _register_policy(client, "org-1", 2, SEED_A)
    two = _register_policy(client, "org-2", 3, SEED_B)
    three = _register_policy(client, "org-3", 1, SEED_C)
    revocations = [
        _revoke(client, one["id"], REASON_A, actor_id="org-1", seed=SEED_A),
        _revoke(client, two["id"], REASON_B, actor_id="org-2", seed=SEED_B),
        _revoke(client, three["id"], REASON_C, actor_id="org-3", seed=SEED_C),
    ]
    return [one, two, three], revocations


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


def _revocation_rows(session):
    return session.execute(select(ActorTrustPolicyRevocation)).scalars().all()


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


def test_items_carry_exactly_the_revocation_public_view(client):
    _, revocations = _world(client)
    resp = client.get(REVOCATIONS_PATH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    assert body["next_cursor"] is None
    # Stable creation order, and each item is exactly the public view.
    assert [item["id"] for item in body["items"]] == [
        revocation["id"] for revocation in revocations
    ]
    for item, revocation in zip(body["items"], revocations):
        assert list(item) == ["id", "policy_id", "actor_id", "reason", "created_at"]
        assert item == revocation
        assert item["id"].startswith("tpr_")
        created_at = datetime.fromisoformat(item["created_at"])
        assert created_at.utcoffset().total_seconds() == 0
        assert item["created_at"].endswith(("Z", "+00:00"))


def test_reason_is_returned_verbatim(client):
    _, revocations = _world(client)
    resp = client.get(REVOCATIONS_PATH)
    assert resp.status_code == 200
    reasons = [item["reason"] for item in resp.json()["items"]]
    # The padded, tabbed, non-ASCII reason comes back byte-exact.
    assert reasons == [REASON_A, REASON_B, REASON_C]
    assert revocations[1]["reason"] == REASON_B


# --- Filtering ------------------------------------------------------------------


def test_policy_id_filter_is_an_exact_match(client):
    policies, revocations = _world(client)
    resp = client.get(REVOCATIONS_PATH, params={"policy_id": policies[1]["id"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["next_cursor"] is None
    assert [item["id"] for item in body["items"]] == [revocations[1]["id"]]


def test_actor_id_filter_is_an_exact_match(client):
    _, revocations = _world(client)
    resp = client.get(REVOCATIONS_PATH, params={"actor_id": "org-3"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert [item["id"] for item in body["items"]] == [revocations[2]["id"]]


def test_filters_combine_as_logical_and(client):
    policies, revocations = _world(client)
    both = client.get(
        REVOCATIONS_PATH,
        params={"policy_id": policies[0]["id"], "actor_id": "org-1"},
    )
    assert both.status_code == 200
    assert [item["id"] for item in both.json()["items"]] == [revocations[0]["id"]]
    # Each filter matches a different record: the intersection is empty.
    crossed = client.get(
        REVOCATIONS_PATH,
        params={"policy_id": policies[0]["id"], "actor_id": "org-2"},
    )
    assert crossed.status_code == 200
    assert crossed.json() == {"items": [], "count": 0, "next_cursor": None}


def test_filters_are_case_and_whitespace_sensitive(client):
    policies, _ = _world(client)
    for params in (
        {"actor_id": "Org-1"},
        {"actor_id": "org-1 "},
        {"actor_id": " org-1"},
        {"policy_id": policies[0]["id"].upper()},
        {"policy_id": policies[0]["id"] + " "},
    ):
        resp = client.get(REVOCATIONS_PATH, params=params)
        assert resp.status_code == 200, params
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_unknown_filter_values_are_an_empty_collection_not_a_404(client):
    _world(client)
    for params in (
        {"policy_id": "atp_" + "0" * 64},
        {"actor_id": "ghost"},
        {"policy_id": "atp_" + "0" * 64, "actor_id": "ghost"},
    ):
        resp = client.get(REVOCATIONS_PATH, params=params)
        assert resp.status_code == 200, params
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_blank_filters_are_422(client):
    _world(client)
    for field in ("policy_id", "actor_id"):
        for blank in ("", "   "):
            resp = client.get(REVOCATIONS_PATH, params={field: blank})
            assert resp.status_code == 422, (field, repr(blank))
            assert resp.json()["error"]["code"] == "validation_error"


# --- limit validation -----------------------------------------------------------


def test_limit_boundaries_one_and_one_hundred_are_accepted(client):
    _world(client)
    assert client.get(REVOCATIONS_PATH, params={"limit": "1"}).status_code == 200
    assert client.get(REVOCATIONS_PATH, params={"limit": "100"}).status_code == 200


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
        policy = _register_policy(client, actor_id, 1, seed)
        _revoke(client, policy["id"], "bulk", actor_id=actor_id, seed=seed)
    resp = client.get(REVOCATIONS_PATH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 51
    assert len(body["items"]) == 50
    assert body["next_cursor"] is not None


# --- Parameter strictness -------------------------------------------------------


def test_repeated_and_undeclared_parameters_are_422(client):
    _world(client)
    repeated = client.get(
        REVOCATIONS_PATH, params=[("actor_id", "org-1"), ("actor_id", "org-2")]
    )
    assert repeated.status_code == 422
    repeated_policy = client.get(
        REVOCATIONS_PATH, params=[("policy_id", "a"), ("policy_id", "b")]
    )
    assert repeated_policy.status_code == 422
    repeated_limit = client.get(
        REVOCATIONS_PATH, params=[("limit", "1"), ("limit", "2")]
    )
    assert repeated_limit.status_code == 422
    for params in (
        {"policy_ids": "org-1"},
        {"reason": "bulk"},
        {"offset": "1"},
    ):
        resp = client.get(REVOCATIONS_PATH, params=params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_empty_body_is_422_rejected_before_any_read(client, db_session):
    _world(client)
    events_before = _audit_count(db_session)
    for body in (b"{}", b" ", b"{not valid json", b"[]"):
        resp = client.request(
            "GET",
            REVOCATIONS_PATH,
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"
    assert _audit_count(db_session) == events_before


# --- Pagination -----------------------------------------------------------------


def test_pages_resume_without_duplication_or_omission(client):
    _, revocations = _world(client)
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

    second = client.get(REVOCATIONS_PATH, params={"limit": "2", "cursor": cursor})
    assert second.status_code == 200
    page_two = second.json()
    # The filtered total covers every page, including the final one.
    assert page_two["count"] == 3
    assert [item["id"] for item in page_two["items"]] == [revocations[2]["id"]]
    assert page_two["next_cursor"] is None

    seen = [item["id"] for item in page_one["items"] + page_two["items"]]
    assert seen == [revocation["id"] for revocation in revocations]


def test_filtered_pages_bind_the_filters(client):
    policies, revocations = _world(client)
    # A filtered page carries the filtered total and no cursor past the end.
    resp = client.get(
        REVOCATIONS_PATH,
        params={"policy_id": policies[0]["id"], "actor_id": "org-1", "limit": "1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert [item["id"] for item in body["items"]] == [revocations[0]["id"]]
    assert body["next_cursor"] is None


def test_replayed_cursor_returns_the_same_page(client):
    _world(client)
    first = client.get(REVOCATIONS_PATH, params={"limit": "1"}).json()
    cursor = first["next_cursor"]
    assert cursor is not None
    page = client.get(REVOCATIONS_PATH, params={"limit": "1", "cursor": cursor})
    replay = client.get(REVOCATIONS_PATH, params={"limit": "1", "cursor": cursor})
    assert page.status_code == 200
    assert replay.status_code == 200
    assert replay.content == page.content


def test_cursor_at_or_past_the_tail_returns_empty_page_with_count(client, app):
    _world(client)
    claims = {"policy_id": None, "actor_id": None, "limit": 50}
    cursor = pagination.encode_typed_cursor(
        app.state.trust_policy_revocations_cursor_secret,
        pagination.TRUST_POLICY_REVOCATIONS_CURSOR,
        {**claims, "offset": 3},
    )
    resp = client.get(REVOCATIONS_PATH, params={"cursor": cursor})
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 3, "next_cursor": None}

    past = pagination.encode_typed_cursor(
        app.state.trust_policy_revocations_cursor_secret,
        pagination.TRUST_POLICY_REVOCATIONS_CURSOR,
        {**claims, "offset": 99},
    )
    resp = client.get(REVOCATIONS_PATH, params={"cursor": past})
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 3, "next_cursor": None}


def test_cursor_binds_the_effective_filters_and_limit(client):
    policies, _ = _world(client)
    cursor = client.get(REVOCATIONS_PATH, params={"limit": "2"}).json()[
        "next_cursor"
    ]
    assert cursor is not None
    # A different limit, a new filter, or a dropped filter all mismatch.
    for params in (
        {"limit": "3", "cursor": cursor},
        {"actor_id": "org-1", "limit": "2", "cursor": cursor},
        {"policy_id": policies[0]["id"], "limit": "2", "cursor": cursor},
        {"cursor": cursor},
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


def test_foreign_family_cursor_is_422(client, app):
    _world(client)
    # Well-formed cursors minted by other endpoint families never resume
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
    resp = client.get(REVOCATIONS_PATH, params={"cursor": foreign})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    # The sibling trust-policy family (tp1) is likewise foreign here.
    policies_cursor = pagination.encode_typed_cursor(
        app.state.trust_policies_cursor_secret,
        pagination.TRUST_POLICIES_CURSOR,
        {"actor_id": None, "limit": 50, "offset": 1},
    )
    resp = client.get(REVOCATIONS_PATH, params={"cursor": policies_cursor})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Method boundary ------------------------------------------------------------


def test_put_patch_and_delete_are_405_method_not_allowed(client, db_session):
    _world(client)
    rows_before = [row.id for row in _revocation_rows(db_session)]
    events_before = _audit_count(db_session)
    for method in (client.put, client.patch, client.delete):
        resp = method(REVOCATIONS_PATH)
        assert resp.status_code == 405
        assert resp.json()["error"]["code"] == "method_not_allowed"
    assert [row.id for row in _revocation_rows(db_session)] == rows_before
    assert _audit_count(db_session) == events_before


def test_post_remains_the_revocation_creation_route(client):
    # The new read entry does not shadow or alter revocation creation.
    policy = _register_policy(client, "org-1", 2, SEED_A)
    body = json.dumps(
        {"policy_id": policy["id"], "reason": "done"}
    ).encode()
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", REVOCATIONS_PATH, body, actor="org-1", seed=SEED_A),
    }
    first = client.post(REVOCATIONS_PATH, content=body, headers=headers)
    assert first.status_code == 201, first.text
    repeat = client.post(REVOCATIONS_PATH, content=body, headers=headers)
    assert repeat.status_code == 200
    assert repeat.json() == first.json()


# --- Read-only guarantee ---------------------------------------------------------


def test_queries_and_failures_write_nothing(client, db_session):
    _world(client)
    events_before = _audit_count(db_session)
    rows_before = [row.id for row in _revocation_rows(db_session)]
    policies_before = [
        row.id for row in db_session.execute(select(ActorTrustPolicy)).scalars()
    ]

    assert client.get(REVOCATIONS_PATH).status_code == 200
    assert client.get(REVOCATIONS_PATH, params={"actor_id": "ghost"}).status_code == 200
    assert client.get(REVOCATIONS_PATH, params={"limit": "1"}).status_code == 200
    assert client.get(REVOCATIONS_PATH, params={"limit": "0"}).status_code == 422
    assert client.get(REVOCATIONS_PATH, params={"cursor": "bad"}).status_code == 422

    assert [row.id for row in _revocation_rows(db_session)] == rows_before
    assert [
        row.id for row in db_session.execute(select(ActorTrustPolicy)).scalars()
    ] == policies_before
    assert _audit_count(db_session) == events_before


def test_concurrent_reads_return_identical_pages(client):
    _world(client)
    expected = client.get(REVOCATIONS_PATH, params={"limit": "2"})
    assert expected.status_code == 200
    cursor = expected.json()["next_cursor"]
    results = []
    errors = []

    def _read():
        try:
            first = client.get(REVOCATIONS_PATH, params={"limit": "2"})
            second = client.get(
                REVOCATIONS_PATH, params={"limit": "2", "cursor": cursor}
            )
            results.append((first.content, second.content))
        except Exception as exc:  # pragma: no cover - failure reporting
            errors.append(exc)

    threads = [threading.Thread(target=_read) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    follow_up = client.get(REVOCATIONS_PATH, params={"limit": "2", "cursor": cursor})
    assert follow_up.status_code == 200
    for first_content, second_content in results:
        assert first_content == expected.content
        assert second_content == follow_up.content


def test_revocations_list_identically_across_an_app_restart(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    _world(file_client)
    first = file_client.get(REVOCATIONS_PATH)
    assert first.status_code == 200

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as restarted_client:
        second = restarted_client.get(REVOCATIONS_PATH)
        assert second.status_code == 200
        assert second.json() == first.json()
