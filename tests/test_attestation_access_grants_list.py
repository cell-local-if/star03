"""Tests for the signer's protected access-grant listing.

Covers ``GET /v1/attestations/{attestation_id}/access-grants``, including:

* the ``X-PA``/``X-PT``/``X-PS`` signed-header contract with an empty GET
  body and a signed path carrying no query string;
* signer-only reads: a missing target, missing/unauthenticated credentials,
  and a non-signer (including a grantee that may read the proof itself) all
  collapse into one opaque 404, while malformed credentials stay 422;
* the success envelope ``{"items", "count", "next_cursor"}`` with items
  carrying exactly ``id``/``attestation_id``/``grantee_actor_id``/
  ``created_at`` in stable creation order with UTC timestamps;
* strict ``limit`` (1-100, default 50) and ``cursor`` parsing: undeclared,
  repeated, blank, and malformed parameters are 422;
* opaque, tamper-evident cursors bound to the proof, the caller, and the
  effective limit: continuation pages repeat and omit nothing, the final
  cursor is null, and paging past the end returns empty items with the
  original count;
* revoked grants remain retained rows in the listing but no revocation,
  private key, signature, header, payload, or byte is ever echoed;
* strictly read-only: no resource or audit writes from any read or failed
  request, and the existing protected routes keep working unchanged.

All tests are deterministic and offline (the stdlib test signer produces
the Ed25519 signatures).
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from provenance.access_signing import access_message_bytes
from provenance import pagination
from provenance.models import AttestationAccessGrant, AuditEvent
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

GRANTS_PATH = "/v1/attestation-access-grants"
SEED_C = b"test-ed25519-seed-c-00000000000000"[:32]
SEED_D = b"test-ed25519-seed-d-00000000000000"[:32]


# --- Fixture-style setup ------------------------------------------------------


def _make_claim(client, actor_id, digest=DIGEST_A):
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


def _world(client):
    """Three actors, each holding a usable non-revoked attestation."""
    create_actor(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    create_actor(client, actor_id="org-3", name="Third Org", type="organization")
    attestation = _make_attestation(
        client, "org-1", SEED_A, _make_claim(client, "org-1")["id"]
    )
    _make_attestation(
        client, "org-2", SEED_B,
        _make_claim(client, "org-2", digest=DIGEST_B)["id"],
    )
    _make_attestation(
        client, "org-3", SEED_C,
        _make_claim(
            client, "org-3",
            digest=hashlib.sha256(b"content-c3").hexdigest(),
        )["id"],
    )
    return attestation


def _signed_headers(
    method,
    path,
    body,
    *,
    actor="org-1",
    seed=SEED_A,
    timestamp=None,
    signed_timestamp=None,
    signed_method=None,
    signed_path=None,
    signed_body=None,
):
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    signed_ts = signed_timestamp if signed_timestamp is not None else ts
    body_digest = hashlib.sha256(
        signed_body if signed_body is not None else body
    ).hexdigest()
    message = access_message_bytes(
        signed_method or method,
        signed_path if signed_path is not None else path,
        signed_ts,
        body_digest,
    )
    signature = base64.b64encode(ed25519_sign(seed, message)).decode("ascii")
    return {"X-PA": actor, "X-PT": ts, "X-PS": signature}


def _grant_body(attestation_id, grantee_actor_id):
    return {
        "attestation_id": attestation_id,
        "grantee_actor_id": grantee_actor_id,
    }


def _post_grant(client, attestation_id, grantee_actor_id, *, actor="org-1",
                seed=SEED_A):
    body = json.dumps(_grant_body(attestation_id, grantee_actor_id)).encode()
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", GRANTS_PATH, body, actor=actor, seed=seed),
    }
    resp = client.post(GRANTS_PATH, content=body, headers=headers)
    assert resp.status_code in (200, 201), resp.text
    return resp.json()


def _list_path(attestation_id):
    return f"/v1/attestations/{attestation_id}/access-grants"


def _get_grants(
    client,
    attestation_id,
    *,
    actor="org-1",
    seed=SEED_A,
    params=None,
    path=None,
    **sign_kwargs,
):
    path = path or _list_path(attestation_id)
    signed = _signed_headers("GET", path, b"", actor=actor, seed=seed,
                             **sign_kwargs)
    return client.get(path, params=params, headers=signed)


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


# --- Success envelope and item shape ------------------------------------------


def test_empty_attestation_returns_empty_page_with_null_cursor(client):
    attestation = _world(client)
    resp = _get_grants(client, attestation["id"])
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_items_have_exactly_the_four_public_fields_in_creation_order(client):
    attestation = _world(client)
    first = _post_grant(client, attestation["id"], "org-2")
    second = _post_grant(client, attestation["id"], "org-3")

    resp = _get_grants(client, attestation["id"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"items", "count", "next_cursor"}
    assert body["count"] == 2
    assert body["next_cursor"] is None
    assert [item["id"] for item in body["items"]] == [first["id"], second["id"]]
    for item in body["items"]:
        assert set(item) == {
            "id",
            "attestation_id",
            "grantee_actor_id",
            "created_at",
        }
        assert item["attestation_id"] == attestation["id"]
        created_at = datetime.fromisoformat(item["created_at"])
        assert created_at.utcoffset().total_seconds() == 0
        assert item["created_at"].endswith(("Z", "+00:00"))
    assert body["items"] == [first, second]


def test_grants_to_distinct_attestations_are_listed_separately(client):
    attestation = _world(client)
    # A second proof signed by the same actor under a second current key.
    second_att = _make_attestation(
        client, "org-1", SEED_D,
        _make_claim(
            client, "org-1",
            digest=hashlib.sha256(b"second-proof").hexdigest(),
        )["id"],
    )
    _post_grant(client, attestation["id"], "org-2")
    _post_grant(client, second_att["id"], "org-3")

    first_page = _get_grants(client, attestation["id"]).json()
    second_page = _get_grants(client, second_att["id"]).json()
    assert [i["grantee_actor_id"] for i in first_page["items"]] == ["org-2"]
    assert [i["grantee_actor_id"] for i in second_page["items"]] == ["org-3"]


# --- Pagination ---------------------------------------------------------------


def test_pagination_walks_every_grant_once_with_limit_one(client):
    attestation = _world(client)
    created = [
        _post_grant(client, attestation["id"], g)
        for g in ("org-2", "org-3")
    ]
    seen = []
    cursor = None
    pages = 0
    while True:
        params = {"limit": "1"}
        if cursor is not None:
            params["cursor"] = cursor
        page = _get_grants(client, attestation["id"], params=params).json()
        pages += 1
        assert page["count"] == 2
        seen.extend(item["id"] for item in page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
        assert pages <= 2
    assert pages == 2
    assert seen == [item["id"] for item in created]


def test_pagination_boundaries_and_stable_replay(client):
    attestation = _world(client)
    created = [
        _post_grant(client, attestation["id"], g) for g in ("org-2", "org-3")
    ]

    first = _get_grants(client, attestation["id"], params={"limit": "1"})
    assert first.status_code == 200, first.text
    first_body = first.json()
    assert len(first_body["items"]) == 1
    assert first_body["count"] == 2
    assert first_body["next_cursor"]
    assert first_body["items"][0]["id"] == created[0]["id"]

    second = _get_grants(
        client, attestation["id"],
        params={"limit": "1", "cursor": first_body["next_cursor"]},
    )
    assert second.status_code == 200, second.text
    second_body = second.json()
    assert len(second_body["items"]) == 1
    assert second_body["count"] == 2
    assert second_body["next_cursor"] is None
    assert second_body["items"][0]["id"] == created[1]["id"]

    # Replaying a still-valid cursor replays the identical page: no gap, no
    # duplication, no new cursor chain.
    replay = _get_grants(
        client, attestation["id"],
        params={"limit": "1", "cursor": first_body["next_cursor"]},
    )
    assert replay.json() == second_body


def test_cursor_past_the_end_returns_empty_items_and_original_count(
    client, app
):
    attestation = _world(client)
    _post_grant(client, attestation["id"], "org-2")
    # A server-signed token positioned past the tail: same proof, caller,
    # and limit, with an offset beyond the single-row result set.
    token = pagination.encode_typed_cursor(
        app.state.attestation_access_grants_cursor_secret,
        pagination.ATTESTATION_ACCESS_GRANTS_CURSOR,
        {
            "attestation_id": attestation["id"],
            "actor_id": "org-1",
            "limit": 50,
            "offset": 99,
        },
    )
    resp = _get_grants(client, attestation["id"], params={"cursor": token})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 1, "next_cursor": None}


def test_limit_boundaries_one_and_one_hundred_are_accepted(client):
    attestation = _world(client)
    for value in ("1", "100"):
        resp = _get_grants(client, attestation["id"], params={"limit": value})
        assert resp.status_code == 200, (value, resp.text)
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_default_limit_is_fifty(client):
    attestation = _world(client)
    # 51 distinct new grantees (org-4..org-54); each grant needs an existing
    # actor, so register them first.
    grantees = [f"org-{i}" for i in range(4, 55)]
    for grantee in grantees:
        create_actor(client, actor_id=grantee, name=grantee, type="organization")
    for grantee in grantees:
        _post_grant(client, attestation["id"], grantee)

    first = _get_grants(client, attestation["id"])
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["count"] == 51
    assert len(body["items"]) == 50
    assert body["next_cursor"]

    second = _get_grants(
        client, attestation["id"], params={"cursor": body["next_cursor"]}
    )
    assert second.status_code == 200, second.text
    tail = second.json()
    assert tail["count"] == 51
    assert len(tail["items"]) == 1
    assert tail["next_cursor"] is None
    assert tail["items"][0]["grantee_actor_id"] == "org-54"


# --- Query parameter validation -----------------------------------------------


def test_undeclared_repeated_blank_and_illegal_params_are_422(client):
    attestation = _world(client)
    path = _list_path(attestation["id"])

    def signed_get(params):
        return client.get(
            path, params=params,
            headers=_signed_headers("GET", path, b""),
        )

    cases = [
        # Undeclared parameter, on its own or beside a valid one.
        ({"offset": "0"}, "offset"),
        ({"limit": "1", "track": "1"}, "track"),
        # Repeated scalar parameters.
        ([("limit", "1"), ("limit", "2")], "limit"),
        ([("cursor", "a"), ("cursor", "b")], "cursor"),
        # Blank / whitespace limit is never coerced to the default.
        ({"limit": ""}, "limit"),
        ({"limit": "   "}, "limit"),
        # Out of range, non-decimal, and non-integer spellings.
        ({"limit": "0"}, "limit"),
        ({"limit": "101"}, "limit"),
        ({"limit": "-1"}, "limit"),
        ({"limit": "1.0"}, "limit"),
        ({"limit": "abc"}, "limit"),
        ({"limit": "0x1"}, "limit"),
        # A blank cursor is an invalid token.
        ({"cursor": ""}, "cursor"),
        ({"cursor": "   "}, "cursor"),
    ]
    for params, field in cases:
        resp = signed_get(params)
        assert resp.status_code == 422, (params, resp.text)
        assert resp.json()["error"]["code"] == "validation_error"
        assert resp.json()["error"]["details"]["issues"][0]["loc"][-1] == field


def test_malformed_query_is_422_even_without_credentials(client):
    attestation = _world(client)
    path = _list_path(attestation["id"])
    # Parameter validation precedes authentication, so a malformed request
    # never collapses into the opaque 404.
    resp = client.get(path, params={"limit": "0"})
    assert resp.status_code == 422
    resp = client.get(path, params={"cursor": "garbage"})
    assert resp.status_code == 422


# --- Cursor integrity and binding ---------------------------------------------


def test_tampered_and_foreign_family_cursors_are_422(client):
    attestation = _world(client)
    _post_grant(client, attestation["id"], "org-2")
    _post_grant(client, attestation["id"], "org-3")
    cursor = _get_grants(
        client, attestation["id"], params={"limit": "1"}
    ).json()["next_cursor"]
    assert cursor

    tampered = cursor[:-1] + ("A" if cursor[-1] != "A" else "B")
    resp = _get_grants(
        client, attestation["id"], params={"limit": "1", "cursor": tampered}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    for garbage in ("not-a-cursor", "xx.ab.cd", "v1.ab.cd", "../etc", "a b"):
        resp = _get_grants(
            client, attestation["id"], params={"cursor": garbage}
        )
        assert resp.status_code == 422, garbage


def test_cursor_binds_proof_caller_and_limit(client):
    attestation = _world(client)
    other_att = _make_attestation(
        client, "org-1", SEED_D,
        _make_claim(
            client, "org-1",
            digest=hashlib.sha256(b"other-proof").hexdigest(),
        )["id"],
    )
    _post_grant(client, attestation["id"], "org-2")
    _post_grant(client, attestation["id"], "org-3")
    cursor = _get_grants(
        client, attestation["id"], params={"limit": "1"}
    ).json()["next_cursor"]

    # A different effective limit invalidates the cursor.
    resp = _get_grants(
        client, attestation["id"], params={"limit": "2", "cursor": cursor}
    )
    assert resp.status_code == 422

    # A cursor minted for one proof never pages another proof, even for the
    # same signer.
    resp = _get_grants(
        client, other_att["id"], params={"limit": "1", "cursor": cursor}
    )
    assert resp.status_code == 422

    # A cursor bound to the signer cannot be replayed by an authenticated
    # different caller: it does not bind to that caller.
    resp = _get_grants(
        client, attestation["id"],
        params={"limit": "1", "cursor": cursor},
        actor="org-2", seed=SEED_B,
    )
    assert resp.status_code == 422


# --- The opaque 404 boundary --------------------------------------------------


def test_missing_target_unauthenticated_and_non_signer_are_one_404(client):
    attestation = _world(client)
    path = _list_path(attestation["id"])
    _post_grant(client, attestation["id"], "org-2")

    # No credentials at all.
    no_headers = client.get(path)
    assert no_headers.status_code == 404
    assert no_headers.json()["error"]["code"] == "not_found"

    # Blank actor header (missing credential material), even with grants.
    blank = client.get(
        path, headers={"X-PA": "  ", "X-PT": "2026-09-20T00:00:00Z",
                       "X-PS": "x"}
    )
    assert blank.status_code == 404

    # A well-formed signature that does not bind to the claimed identity.
    assert _get_grants(
        client, attestation["id"], actor="org-2", seed=SEED_A
    ).status_code == 404

    # An authenticated stranger is not the signer: opaque 404.
    assert _get_grants(
        client, attestation["id"], actor="org-3", seed=SEED_C
    ).status_code == 404

    # A grantee may read the protected proof but may not list its grants.
    assert _get_grants(
        client, attestation["id"], actor="org-2", seed=SEED_B
    ).status_code == 404

    # A missing target renders identically for the signer.
    ghost = _get_grants(client, "att_ghost")
    assert ghost.status_code == 404
    assert ghost.json() == no_headers.json() == blank.json()


def test_malformed_credentials_remain_422(client):
    attestation = _world(client)
    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert _get_grants(
        client, attestation["id"], timestamp=stale
    ).status_code == 422
    assert _get_grants(
        client, attestation["id"], timestamp="12:00 o'clock"
    ).status_code == 422

    path = _list_path(attestation["id"])
    for raw in ("@@@", "aGVsbG8", base64.b64encode(b"x" * 63).decode()):
        headers = _signed_headers("GET", path, b"")
        headers["X-PS"] = raw
        resp = client.get(path, headers=headers)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["code"] == "validation_error"


def test_signature_binds_method_path_empty_body_and_timestamp(client):
    attestation = _world(client)
    # A signature minted for the write route never authorizes the read.
    assert _get_grants(
        client, attestation["id"], signed_method="POST"
    ).status_code == 404
    path = _list_path(attestation["id"])
    assert _get_grants(
        client, attestation["id"], signed_path=path + "/"
    ).status_code == 404
    # A non-empty body digest cannot authorize a bodyless GET.
    assert _get_grants(
        client, attestation["id"], signed_body=b"x"
    ).status_code == 404
    sent = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    signed = (
        datetime.now(timezone.utc) - timedelta(seconds=5)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert _get_grants(
        client, attestation["id"],
        timestamp=sent, signed_timestamp=signed,
    ).status_code == 404


def test_signed_path_excludes_the_query_string(client):
    attestation = _world(client)
    path = _list_path(attestation["id"])
    headers = _signed_headers("GET", path, b"")
    # The signature covers the path without a query string; attaching one
    # still succeeds, and the same signature validates both spellings.
    assert client.get(path, params={"limit": "1"}, headers=headers).status_code == 200
    assert client.get(path + "?limit=1", headers=headers).status_code == 200


# --- Revocations are not echoed ------------------------------------------------


def test_revoked_grant_stays_listed_without_any_revocation_fields(client):
    attestation = _world(client)
    grant = _post_grant(client, attestation["id"], "org-2")
    revocation_body = json.dumps(
        {"grant_id": grant["id"], "reason": "no longer needed"}
    ).encode()
    revoked = client.post(
        "/v1/attestation-access-grant-revocations",
        content=revocation_body,
        headers={
            "Content-Type": "application/json",
            **_signed_headers(
                "POST",
                "/v1/attestation-access-grant-revocations",
                revocation_body,
            ),
        },
    )
    assert revoked.status_code == 201, revoked.text

    resp = _get_grants(client, attestation["id"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 1
    item = body["items"][0]
    assert set(item) == {
        "id", "attestation_id", "grantee_actor_id", "created_at"
    }
    assert item == grant
    text = resp.text
    for absent in ("revocation", "revoked", "reason", "private", "signature"):
        assert absent not in text


# --- Read-only guarantees ------------------------------------------------------


def test_listing_writes_no_resources_or_audit(client, db_session):
    attestation = _world(client)
    _post_grant(client, attestation["id"], "org-2")
    events_before = _audit_count(db_session)
    grants_before = len(
        db_session.execute(select(AttestationAccessGrant)).scalars().all()
    )

    path = _list_path(attestation["id"])
    responses = [
        _get_grants(client, attestation["id"]),
        _get_grants(client, attestation["id"], params={"limit": "1"}),
        _get_grants(client, attestation["id"], actor="org-3", seed=SEED_C),
        _get_grants(client, "att_ghost"),
        client.get(path),
        client.get(path, params={"limit": "0"}),
    ]
    for resp in responses:
        assert resp.status_code in (200, 404, 422), resp.text

    db_session.expire_all()
    assert _audit_count(db_session) == events_before
    assert len(
        db_session.execute(select(AttestationAccessGrant)).scalars().all()
    ) == grants_before


def test_existing_protected_routes_keep_working(client):
    attestation = _world(client)
    _post_grant(client, attestation["id"], "org-2")
    # The protected proof read still serves the signer and the grantee.
    assert _get_grants(client, attestation["id"]).status_code == 200
    proof_path = f"/v1/protected/attestations/{attestation['id']}"
    signer = client.get(
        proof_path, headers=_signed_headers("GET", proof_path, b"")
    )
    assert signer.status_code == 200
    grantee = client.get(
        proof_path,
        headers=_signed_headers(
            "GET", proof_path, b"", actor="org-2", seed=SEED_B
        ),
    )
    assert grantee.status_code == 200
