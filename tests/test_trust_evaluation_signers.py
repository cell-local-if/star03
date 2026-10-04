"""Tests for the read-only trust-evaluation signer trace endpoint.

Covers GET /v1/trust-evaluation-signers: the qualified-signer detail behind
the trust-evaluation count (distinct signing actors, their qualifying
attestation ids in stable creation order, and the earliest/latest UTC
creation times), the exact ``target_type``/``target_id``/``items``/
``count``/``next_cursor`` member order in compact UTF-8 JSON terminated by
one newline, signer ordering by earliest qualifying proof with the
``signer_actor_id`` tiebreak, revocation exclusion, both target types, the
opaque HMAC-signed ``ts1`` cursor family (binding the exact target and the
limit), every 422/404/405 boundary (validation precedes existence), and the
strictly read-only guarantee. All tests are deterministic and offline
(signatures are produced by the stdlib test signer).
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone

from sqlalchemy import func, select

from provenance import pagination
from provenance.models import Attestation, AuditEvent
from provenance.signing import attestation_message_bytes
from tests.helpers import (
    DIGEST_B,
    DIGEST_C,
    content_payload,
    create_actor,
    ed25519_public_key,
    ed25519_sign,
    SEED_A,
    SEED_B,
)

SIGNERS_PATH = "/v1/trust-evaluation-signers"

SEED_C = b"test-ed25519-seed-c-00000000000000"[:32]
SEED_D = b"test-ed25519-seed-d-00000000000000"[:32]

EVIDENCE_DIGEST = hashlib.sha256(b"evidence-signers").hexdigest()


# --- Setup helpers ----------------------------------------------------------


def _create_content(client, actor_id="org-1", digest=DIGEST_B):
    resp = client.post(
        "/v1/contents", json=content_payload(actor_id=actor_id, digest=digest)
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, actor_id="org-1"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": "authorship",
            "payload": {"statement": "trust me"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_bundle(client, claim_id):
    resp = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim_id,
            "evidence_type": "raw_capture",
            "digest_algorithm": "sha256",
            "digest_hex": EVIDENCE_DIGEST,
            "media_type": "image/jpeg",
            "metadata": {"source": "camera-1"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _setup_claim(client):
    create_actor(client)
    return _create_claim(client, _create_content(client)["id"])


def _attest(client, target_type, target_id, *, seed=SEED_A, signer_actor_id="org-1"):
    signature = ed25519_sign(
        seed, attestation_message_bytes(target_type, target_id, signer_actor_id)
    )
    resp = client.post(
        "/v1/attestations",
        json={
            "target_type": target_type,
            "target_id": target_id,
            "signer_actor_id": signer_actor_id,
            "public_key": base64.b64encode(ed25519_public_key(seed)).decode("ascii"),
            "signature": base64.b64encode(signature).decode("ascii"),
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _revoke(client, attestation_id, *, revoker_actor_id="org-1"):
    resp = client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": attestation_id,
            "revoker_actor_id": revoker_actor_id,
            "reason": "no longer relied upon",
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _signers(client, target_type, target_id, **params):
    return client.get(
        SIGNERS_PATH,
        params={"target_type": target_type, "target_id": target_id, **params},
    )


def _audit_count(session):
    return session.scalar(select(func.count()).select_from(AuditEvent))


def _attestation_count(session):
    return session.scalar(select(func.count()).select_from(Attestation))


# --- Response shape and empty collection -------------------------------------


def test_existing_target_without_attestations_is_an_empty_collection(client):
    claim = _setup_claim(client)
    resp = _signers(client, "claim", claim["id"])
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "target_type": "claim",
        "target_id": claim["id"],
        "items": [],
        "count": 0,
        "next_cursor": None,
    }


def test_response_is_compact_json_with_ordered_members_and_one_newline(client):
    claim = _setup_claim(client)
    _attest(client, "claim", claim["id"])
    resp = _signers(client, "claim", claim["id"])
    assert resp.status_code == 200
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    # The top-level members appear in exactly this order.
    assert raw.startswith(b'{"target_type":"claim","target_id":"')
    assert b'"items":[{' in raw
    assert b'}],"count":1,"next_cursor":null}' in raw
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    assert raw == expected


def test_item_carries_exactly_the_signer_trace_fields(client):
    claim = _setup_claim(client)
    att = _attest(client, "claim", claim["id"])
    resp = _signers(client, "claim", claim["id"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    (item,) = body["items"]
    assert list(item) == [
        "signer_actor_id",
        "attestation_ids",
        "attestation_count",
        "first_attested_at",
        "latest_attested_at",
    ]
    assert item["signer_actor_id"] == "org-1"
    assert item["attestation_ids"] == [att["id"]]
    assert item["attestation_count"] == 1
    assert item["first_attested_at"] == att["created_at"]
    assert item["latest_attested_at"] == att["created_at"]
    assert item["first_attested_at"].endswith(("Z", "+00:00"))
    # No key, signature, digest, or payload material leaks into the view.
    assert "public_key" not in item
    assert "signature" not in item
    assert "signature_digest_hex" not in item


# --- Qualified-signer semantics ------------------------------------------------


def test_same_signer_with_distinct_keys_produces_one_item(client):
    claim = _setup_claim(client)
    first = _attest(client, "claim", claim["id"], seed=SEED_A)
    second = _attest(client, "claim", claim["id"], seed=SEED_B)

    resp = _signers(client, "claim", claim["id"])
    body = resp.json()
    assert body["count"] == 1
    (item,) = body["items"]
    assert item["signer_actor_id"] == "org-1"
    # Both qualifying proofs are listed in stable creation order.
    assert item["attestation_ids"] == [first["id"], second["id"]]
    assert item["attestation_count"] == 2
    assert item["first_attested_at"] == first["created_at"]
    assert item["latest_attested_at"] == second["created_at"]


def test_signers_are_ordered_by_their_earliest_qualifying_attestation(client):
    claim = _setup_claim(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    create_actor(client, actor_id="org-3", name="Third Org", type="organization")
    # org-2 attests first, then org-3, then org-1; org-2 also attests again.
    _attest(client, "claim", claim["id"], seed=SEED_A, signer_actor_id="org-2")
    _attest(client, "claim", claim["id"], seed=SEED_B, signer_actor_id="org-3")
    _attest(client, "claim", claim["id"], seed=SEED_C, signer_actor_id="org-1")
    _attest(client, "claim", claim["id"], seed=SEED_D, signer_actor_id="org-2")

    resp = _signers(client, "claim", claim["id"])
    body = resp.json()
    assert body["count"] == 3
    assert [item["signer_actor_id"] for item in body["items"]] == [
        "org-2",
        "org-3",
        "org-1",
    ]
    counts = {item["signer_actor_id"]: item["attestation_count"] for item in body["items"]}
    assert counts == {"org-1": 1, "org-2": 2, "org-3": 1}


def test_signers_tied_on_time_sort_by_actor_id(client, db_session):
    claim = _setup_claim(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    instant = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    # Insert two qualifying proofs at the exact same creation instant, the
    # lexicographically later actor first.
    for index, actor_id in enumerate(("org-2", "org-1")):
        db_session.add(
            Attestation(
                id=f"att_tie_{index}",
                target_type="claim",
                target_id=claim["id"],
                signer_actor_id=actor_id,
                public_key=b"\x01" * 32,
                signature_digest_algorithm="sha256",
                signature_digest_hex=hashlib.sha256(f"sig-{index}".encode()).hexdigest(),
                created_at=instant,
            )
        )
    db_session.commit()

    resp = _signers(client, "claim", claim["id"])
    assert resp.status_code == 200
    body = resp.json()
    assert [item["signer_actor_id"] for item in body["items"]] == ["org-1", "org-2"]
    for item in body["items"]:
        assert item["first_attested_at"] == item["latest_attested_at"]


def test_only_attestations_of_the_exact_target_are_listed(client):
    claim_one = _setup_claim(client)
    content_two = _create_content(client, digest=DIGEST_C)
    claim_two = _create_claim(client, content_two["id"])
    bundle = _create_bundle(client, claim_one["id"])

    _attest(client, "claim", claim_two["id"], seed=SEED_B)
    _attest(client, "evidence_bundle", bundle["id"], seed=SEED_C)

    resp = _signers(client, "claim", claim_one["id"])
    assert resp.json()["count"] == 0
    assert resp.json()["items"] == []

    resp = _signers(client, "claim", claim_two["id"])
    assert resp.json()["count"] == 1
    resp = _signers(client, "evidence_bundle", bundle["id"])
    assert resp.json()["count"] == 1


def test_evidence_bundle_target_is_traced(client):
    claim = _setup_claim(client)
    bundle = _create_bundle(client, claim["id"])
    att = _attest(client, "evidence_bundle", bundle["id"])

    resp = _signers(client, "evidence_bundle", bundle["id"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["target_type"] == "evidence_bundle"
    assert body["target_id"] == bundle["id"]
    assert body["count"] == 1
    assert body["items"][0]["attestation_ids"] == [att["id"]]


def test_revoked_attestations_never_qualify(client):
    claim = _setup_claim(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    kept = _attest(client, "claim", claim["id"], seed=SEED_A, signer_actor_id="org-1")
    dropped = _attest(client, "claim", claim["id"], seed=SEED_B, signer_actor_id="org-1")
    other = _attest(client, "claim", claim["id"], seed=SEED_C, signer_actor_id="org-2")

    # Revoking one of org-1's two proofs keeps the signer with the other.
    _revoke(client, dropped["id"])
    body = _signers(client, "claim", claim["id"]).json()
    assert body["count"] == 2
    by_actor = {item["signer_actor_id"]: item for item in body["items"]}
    assert by_actor["org-1"]["attestation_ids"] == [kept["id"]]
    assert by_actor["org-1"]["first_attested_at"] == kept["created_at"]
    assert by_actor["org-1"]["latest_attested_at"] == kept["created_at"]
    assert by_actor["org-2"]["attestation_ids"] == [other["id"]]

    # Revoking org-2's only proof drops that signer entirely; the revoked
    # rows themselves remain stored (immutable revocation history).
    _revoke(client, other["id"], revoker_actor_id="org-2")
    body = _signers(client, "claim", claim["id"]).json()
    assert body["count"] == 1
    assert [item["signer_actor_id"] for item in body["items"]] == ["org-1"]


def test_detail_is_computed_live_from_persisted_state(client):
    claim = _setup_claim(client)
    assert _signers(client, "claim", claim["id"]).json()["count"] == 0
    att = _attest(client, "claim", claim["id"])
    body = _signers(client, "claim", claim["id"]).json()
    assert body["count"] == 1
    assert body["items"][0]["attestation_ids"] == [att["id"]]
    _revoke(client, att["id"])
    assert _signers(client, "claim", claim["id"]).json()["count"] == 0


# --- Missing-resource boundary ------------------------------------------------


def test_unknown_claim_target_is_404(client):
    _setup_claim(client)
    resp = _signers(client, "claim", "clm_ghost")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "claim_not_found"
    assert error["details"]["claim_id"] == "clm_ghost"


def test_unknown_evidence_bundle_target_is_404(client):
    _setup_claim(client)
    resp = _signers(client, "evidence_bundle", "evb_ghost")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "evidence_bundle_not_found"
    assert error["details"]["evidence_bundle_id"] == "evb_ghost"


def test_wrong_target_type_for_existing_resource_is_404(client):
    claim = _setup_claim(client)
    bundle = _create_bundle(client, claim["id"])
    r1 = _signers(client, "claim", bundle["id"])
    r2 = _signers(client, "evidence_bundle", claim["id"])
    assert r1.status_code == 404
    assert r1.json()["error"]["code"] == "claim_not_found"
    assert r2.status_code == 404
    assert r2.json()["error"]["code"] == "evidence_bundle_not_found"


# --- Validation boundary -------------------------------------------------------


def test_missing_target_type_or_target_id_is_422(client):
    _setup_claim(client)
    no_type = client.get(SIGNERS_PATH, params={"target_id": "clm_x"})
    assert no_type.status_code == 422
    assert no_type.json()["error"]["code"] == "validation_error"

    no_id = client.get(SIGNERS_PATH, params={"target_type": "claim"})
    assert no_id.status_code == 422
    assert no_id.json()["error"]["code"] == "validation_error"

    neither = client.get(SIGNERS_PATH)
    assert neither.status_code == 422
    assert neither.json()["error"]["code"] == "validation_error"


def test_blank_or_unknown_target_type_is_422(client):
    claim = _setup_claim(client)
    for value in ("", "   ", "Claim", "claim ", " content", "evidence", "attestation"):
        resp = _signers(client, value, claim["id"])
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_blank_target_id_is_422(client):
    for value in ("", "   ", "\t"):
        resp = _signers(client, "claim", value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_limit_must_be_a_pure_decimal_integer_in_range(client):
    claim = _setup_claim(client)
    for bad in ("0", "101", "-1", "5.0", "5e0", " 5", "5 ", "five", "+5", "", "1.0"):
        resp = _signers(client, "claim", claim["id"], limit=bad)
        assert resp.status_code == 422, bad
        assert resp.json()["error"]["code"] == "validation_error", bad


def test_limit_boundaries_one_and_one_hundred_are_accepted(client):
    claim = _setup_claim(client)
    assert _signers(client, "claim", claim["id"], limit="1").status_code == 200
    assert _signers(client, "claim", claim["id"], limit="100").status_code == 200


def test_repeated_and_undeclared_parameters_are_422(client):
    claim = _setup_claim(client)
    base = f"{SIGNERS_PATH}?target_type=claim&target_id={claim['id']}"
    for url in (
        f"{base}&target_type=claim",
        f"{base}&target_id={claim['id']}",
        f"{base}&limit=1&limit=2",
        f"{base}&cursor=a&cursor=b",
        f"{base}&min_signers=1",
        f"{base}&foo=bar",
    ):
        resp = client.get(url)
        assert resp.status_code == 422, url
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_empty_body_is_422_rejected_before_any_read(client, db_session):
    claim = _setup_claim(client)
    events_before = _audit_count(db_session)
    for body in (b"{}", b" ", b"{not valid json", b"[]"):
        resp = client.request(
            "GET",
            SIGNERS_PATH,
            params={"target_type": "claim", "target_id": claim["id"]},
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"
    assert _audit_count(db_session) == events_before


def test_validation_errors_take_precedence_over_missing_target(client):
    # A non-existent target with an otherwise malformed request is 422, not
    # 404: parameters (including the cursor) are validated before any
    # existence lookup.
    resp = _signers(client, "claim", "clm_ghost", limit="abc")
    assert resp.status_code == 422
    resp = _signers(client, "nope", "clm_ghost")
    assert resp.status_code == 422
    resp = _signers(client, "claim", "   ")
    assert resp.status_code == 422
    resp = _signers(client, "claim", "clm_ghost", cursor="not-a-cursor")
    assert resp.status_code == 422


def test_non_get_methods_are_405(client):
    claim = _setup_claim(client)
    for method in ("post", "put", "patch", "delete"):
        resp = getattr(client, method)(SIGNERS_PATH)
        assert resp.status_code == 405, method
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Pagination -----------------------------------------------------------------


def _world(client):
    """One claim with three distinct qualified signers, in attestation order."""
    claim = _setup_claim(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    create_actor(client, actor_id="org-3", name="Third Org", type="organization")
    _attest(client, "claim", claim["id"], seed=SEED_A, signer_actor_id="org-1")
    _attest(client, "claim", claim["id"], seed=SEED_B, signer_actor_id="org-2")
    _attest(client, "claim", claim["id"], seed=SEED_C, signer_actor_id="org-3")
    return claim


def test_pages_resume_without_duplication_or_omission(client):
    claim = _world(client)
    first = _signers(client, "claim", claim["id"], limit="2")
    assert first.status_code == 200
    page_one = first.json()
    # The total covers every page, including the first and the final one.
    assert page_one["count"] == 3
    assert [item["signer_actor_id"] for item in page_one["items"]] == ["org-1", "org-2"]
    cursor = page_one["next_cursor"]
    assert cursor is not None

    second = _signers(client, "claim", claim["id"], limit="2", cursor=cursor)
    assert second.status_code == 200
    page_two = second.json()
    assert page_two["count"] == 3
    assert [item["signer_actor_id"] for item in page_two["items"]] == ["org-3"]
    assert page_two["next_cursor"] is None

    seen = [
        item["signer_actor_id"]
        for item in page_one["items"] + page_two["items"]
    ]
    assert seen == ["org-1", "org-2", "org-3"]


def test_replayed_cursor_returns_the_same_page(client):
    claim = _world(client)
    cursor = _signers(client, "claim", claim["id"], limit="1").json()["next_cursor"]
    assert cursor is not None
    page = _signers(client, "claim", claim["id"], limit="1", cursor=cursor)
    replay = _signers(client, "claim", claim["id"], limit="1", cursor=cursor)
    assert page.status_code == 200
    assert replay.status_code == 200
    assert replay.content == page.content


def test_cursor_at_or_past_the_tail_returns_empty_page_with_count(client, app):
    claim = _world(client)
    for offset in (3, 99):
        cursor = pagination.encode_typed_cursor(
            app.state.trust_evaluation_signers_cursor_secret,
            pagination.TRUST_EVALUATION_SIGNERS_CURSOR,
            {
                "target_type": "claim",
                "target_id": claim["id"],
                "limit": 50,
                "offset": offset,
            },
        )
        resp = _signers(client, "claim", claim["id"], cursor=cursor)
        assert resp.status_code == 200
        assert resp.json() == {
            "target_type": "claim",
            "target_id": claim["id"],
            "items": [],
            "count": 3,
            "next_cursor": None,
        }


def test_cursor_binds_the_exact_target_and_limit(client):
    claim = _world(client)
    other = _create_claim(client, _create_content(client, digest=DIGEST_C)["id"])
    cursor = _signers(client, "claim", claim["id"], limit="2").json()["next_cursor"]
    assert cursor is not None
    # A different limit, a different target id, or a different target type
    # all mismatch the cursor's bound query.
    for params in (
        {"target_type": "claim", "target_id": claim["id"], "limit": "3"},
        {"target_type": "claim", "target_id": other["id"], "limit": "2"},
        {"target_type": "evidence_bundle", "target_id": claim["id"], "limit": "2"},
        {"target_type": "claim", "target_id": claim["id"]},
    ):
        resp = client.get(SIGNERS_PATH, params={**params, "cursor": cursor})
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_blank_malformed_and_tampered_cursors_are_422(client):
    claim = _world(client)
    valid = _signers(client, "claim", claim["id"], limit="1").json()["next_cursor"]
    assert valid is not None
    tampered = valid[:-1] + ("A" if valid[-1] != "A" else "B")
    for bad in ("", "   ", "not-a-cursor", tampered):
        resp = _signers(client, "claim", claim["id"], cursor=bad)
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_foreign_family_cursor_is_422(client, app):
    claim = _world(client)
    # A well-formed cursor minted by another endpoint family never resumes
    # this retrieval.
    foreign = pagination.encode_typed_cursor(
        app.state.trust_policies_cursor_secret,
        pagination.TRUST_POLICIES_CURSOR,
        {"actor_id": None, "limit": 50, "offset": 1},
    )
    resp = _signers(client, "claim", claim["id"], cursor=foreign)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_default_limit_is_fifty(client):
    claim = _setup_claim(client)
    for index in range(51):
        actor_id = f"org-bulk-{index:03d}"
        create_actor(client, actor_id=actor_id, name=f"Org {index}")
        seed = hashlib.sha256(f"seed-{actor_id}".encode()).digest()
        _attest(client, "claim", claim["id"], seed=seed, signer_actor_id=actor_id)
    resp = _signers(client, "claim", claim["id"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 51
    assert len(body["items"]) == 50
    assert body["next_cursor"] is not None

    last = _signers(
        client, "claim", claim["id"], cursor=body["next_cursor"]
    )
    assert last.status_code == 200
    tail = last.json()
    assert tail["count"] == 51
    assert len(tail["items"]) == 1
    assert tail["next_cursor"] is None


# --- Read-only guarantee ---------------------------------------------------------


def test_queries_and_failures_write_nothing(client, db_session):
    claim = _world(client)
    events_before = _audit_count(db_session)
    attestations_before = _attestation_count(db_session)

    ok = _signers(client, "claim", claim["id"])
    assert ok.status_code == 200
    assert _signers(client, "claim", claim["id"], limit="1").status_code == 200
    assert _signers(client, "claim", "clm_ghost").status_code == 404
    assert _signers(client, "claim", claim["id"], limit="0").status_code == 422
    assert _signers(client, "claim", claim["id"], cursor="bad").status_code == 422

    assert _attestation_count(db_session) == attestations_before
    assert _audit_count(db_session) == events_before


def test_signers_list_identically_across_an_app_restart(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    claim = _world(file_client)
    first = _signers(file_client, "claim", claim["id"])
    assert first.status_code == 200

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as restarted_client:
        second = _signers(restarted_client, "claim", claim["id"])
        assert second.status_code == 200
        assert second.json() == first.json()
