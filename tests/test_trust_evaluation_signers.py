"""Tests for the read-only trust-evaluation signer trace endpoint.

Covers GET /v1/trust-evaluation-signers: the qualified-signer detail view
behind the trust-evaluation count (same verified, non-revoked, exact-target,
distinct-signer definition), one item per subject however many keys or
proofs it qualified with, attestation ids in stable creation order, signer
ordering by earliest proof with the actor-id tiebreak, the exact response
member order on the compact one-newline wire, cursor pagination bound to
the target and limit, the 422/404/405 boundary (validation precedes
existence), live recomputation as proofs are added or revoked, and the
no-write guarantee. All tests are deterministic and offline (signatures
are produced by the stdlib test signer).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from sqlalchemy import func, select

from provenance import pagination
from provenance.models import Attestation, AttestationRevocation, AuditEvent
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

PATH = "/v1/trust-evaluation-signers"

EVIDENCE_DIGEST = hashlib.sha256(b"evidence-signers").hexdigest()
SEED_C = hashlib.sha256(b"test-ed25519-seed-c").digest()
SEED_D = hashlib.sha256(b"test-ed25519-seed-d").digest()


# --- Setup helpers ----------------------------------------------------------


def _create_content(client, actor_id="org-1", digest=None):
    resp = client.post(
        "/v1/contents",
        json=content_payload(actor_id=actor_id, digest=digest or DIGEST_B),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, actor_id="org-1", claim_type="authorship"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": claim_type,
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


def _setup_claim_and_bundle(client):
    claim = _setup_claim(client)
    return claim, _create_bundle(client, claim["id"])


def _attest(client, target_type, target_id, *, seed=SEED_A, signer_actor_id="org-1"):
    import base64

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


def _get(client, target_type, target_id, **params):
    return client.get(
        PATH,
        params={"target_type": target_type, "target_id": target_id, **params},
    )


def _make_actor(client, actor_id):
    return create_actor(
        client, actor_id=actor_id, name=f"Org {actor_id}", type="organization"
    )


def _three_signer_world(client):
    """One claim attested by org-1, org-2, org-3 in that order."""
    claim = _setup_claim(client)
    _make_actor(client, "org-2")
    _make_actor(client, "org-3")
    _attest(client, "claim", claim["id"], seed=SEED_A, signer_actor_id="org-1")
    _attest(client, "claim", claim["id"], seed=SEED_B, signer_actor_id="org-2")
    _attest(client, "claim", claim["id"], seed=SEED_C, signer_actor_id="org-3")
    return claim


# --- Successful listings ----------------------------------------------------


def test_existing_target_without_attestations_returns_empty_page(client):
    claim = _setup_claim(client)
    resp = _get(client, "claim", claim["id"])
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
    resp = _get(client, "claim", claim["id"])
    assert resp.status_code == 200
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    # The top-level members appear in exactly this order.
    assert raw.startswith(b'{"target_type":"claim","target_id":"')
    assert b'"items":[{"signer_actor_id":"' in raw
    assert raw.endswith(b'"count":1,"next_cursor":null}\n')
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    assert raw == expected


def test_single_signer_with_multiple_keys_appears_once(client):
    claim = _setup_claim(client)
    first = _attest(client, "claim", claim["id"], seed=SEED_A)
    second = _attest(client, "claim", claim["id"], seed=SEED_B)
    third = _attest(client, "claim", claim["id"], seed=SEED_C)

    resp = _get(client, "claim", claim["id"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["next_cursor"] is None
    assert len(body["items"]) == 1
    item = body["items"][0]
    # Exactly the five item members, in this order.
    assert list(item) == [
        "signer_actor_id",
        "attestation_ids",
        "attestation_count",
        "first_attested_at",
        "latest_attested_at",
    ]
    assert item["signer_actor_id"] == "org-1"
    # The qualified proofs in stable creation order; the count equals the
    # identifier list length.
    assert item["attestation_ids"] == [first["id"], second["id"], third["id"]]
    assert item["attestation_count"] == 3 == len(item["attestation_ids"])
    # Earliest and latest UTC creation times across those proofs.
    assert item["first_attested_at"] == first["created_at"]
    assert item["latest_attested_at"] == third["created_at"]
    for key in ("first_attested_at", "latest_attested_at"):
        parsed = datetime.fromisoformat(item[key])
        assert parsed.utcoffset().total_seconds() == 0
        assert item[key].endswith(("Z", "+00:00"))


def test_signers_are_ordered_by_their_earliest_qualified_attestation(client):
    claim = _three_signer_world(client)
    resp = _get(client, "claim", claim["id"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    assert [item["signer_actor_id"] for item in body["items"]] == [
        "org-1",
        "org-2",
        "org-3",
    ]
    for item in body["items"]:
        assert item["attestation_count"] == 1
        assert item["first_attested_at"] == item["latest_attested_at"]


def test_same_instant_signers_are_ordered_by_actor_id(client, db_session):
    claim = _setup_claim(client)
    _make_actor(client, "org-tie-a")
    _make_actor(client, "org-tie-b")
    tie = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def insert(attestation_id, signer_actor_id):
        row = Attestation(
            id=attestation_id,
            target_type="claim",
            target_id=claim["id"],
            signer_actor_id=signer_actor_id,
            public_key=ed25519_public_key(SEED_A),
            signature_digest_algorithm="sha256",
            signature_digest_hex=hashlib.sha256(
                attestation_id.encode()
            ).hexdigest(),
            created_at=tie,
        )
        db_session.add(row)
        return row

    # Insertion order is the reverse of the lexicographic order: the
    # actor-id tiebreak, not the insertion sequence, decides.
    insert("att_" + "b" * 64, "org-tie-b")
    insert("att_" + "a" * 64, "org-tie-a")
    db_session.commit()

    body = _get(client, "claim", claim["id"]).json()
    assert [item["signer_actor_id"] for item in body["items"]] == [
        "org-tie-a",
        "org-tie-b",
    ]


def test_evidence_bundle_target_is_listed(client):
    _claim, bundle = _setup_claim_and_bundle(client)
    _attest(client, "evidence_bundle", bundle["id"])

    resp = _get(client, "evidence_bundle", bundle["id"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["target_type"] == "evidence_bundle"
    assert body["target_id"] == bundle["id"]
    assert body["count"] == 1
    assert [item["signer_actor_id"] for item in body["items"]] == ["org-1"]


def test_only_attestations_of_the_exact_target_qualify(client):
    claim_one = _setup_claim(client)
    content_two = _create_content(client, digest=DIGEST_C)
    claim_two = _create_claim(client, content_two["id"])
    bundle = _create_bundle(client, claim_one["id"])
    _make_actor(client, "org-2")

    _attest(client, "claim", claim_two["id"], seed=SEED_B, signer_actor_id="org-2")
    _attest(client, "evidence_bundle", bundle["id"])

    # Neither the other claim's signer nor the bundle's signer attests
    # claim one.
    assert _get(client, "claim", claim_one["id"]).json()["count"] == 0
    assert _get(client, "claim", claim_two["id"]).json()["count"] == 1
    assert _get(client, "evidence_bundle", bundle["id"]).json()["count"] == 1


def test_revoked_attestations_never_qualify(client):
    claim = _setup_claim(client)
    _make_actor(client, "org-2")
    first = _attest(client, "claim", claim["id"], seed=SEED_A)
    second = _attest(client, "claim", claim["id"], seed=SEED_B)
    other = _attest(
        client, "claim", claim["id"], seed=SEED_C, signer_actor_id="org-2"
    )

    # Revoking one of a signer's two proofs keeps the signer with the
    # remaining proof only.
    _revoke(client, first["id"])
    body = _get(client, "claim", claim["id"]).json()
    assert body["count"] == 2
    by_signer = {item["signer_actor_id"]: item for item in body["items"]}
    assert by_signer["org-1"]["attestation_ids"] == [second["id"]]
    assert by_signer["org-1"]["first_attested_at"] == second["created_at"]
    assert by_signer["org-2"]["attestation_ids"] == [other["id"]]

    # A signer whose last qualified proof is revoked drops out entirely.
    _revoke(client, second["id"])
    body = _get(client, "claim", claim["id"]).json()
    assert body["count"] == 1
    assert [item["signer_actor_id"] for item in body["items"]] == ["org-2"]

    # A target whose proofs are all revoked lists like one with none.
    _revoke(client, other["id"], revoker_actor_id="org-2")
    assert _get(client, "claim", claim["id"]).json() == {
        "target_type": "claim",
        "target_id": claim["id"],
        "items": [],
        "count": 0,
        "next_cursor": None,
    }


def test_listing_is_computed_live(client):
    claim = _setup_claim(client)
    assert _get(client, "claim", claim["id"]).json()["count"] == 0

    att = _attest(client, "claim", claim["id"])
    assert _get(client, "claim", claim["id"]).json()["count"] == 1

    _revoke(client, att["id"])
    assert _get(client, "claim", claim["id"]).json()["count"] == 0


def test_count_matches_the_trust_evaluation_signer_count(client):
    claim = _three_signer_world(client)
    evaluation = client.get(
        "/v1/trust-evaluations",
        params={"target_type": "claim", "target_id": claim["id"]},
    ).json()
    listing = _get(client, "claim", claim["id"]).json()
    assert listing["count"] == evaluation["qualified_signer_count"] == 3


# --- Missing-resource boundary ----------------------------------------------


def test_unknown_claim_target_is_404(client):
    _setup_claim(client)
    resp = _get(client, "claim", "clm_ghost")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "claim_not_found"
    assert error["details"]["claim_id"] == "clm_ghost"


def test_unknown_evidence_bundle_target_is_404(client):
    _setup_claim(client)
    resp = _get(client, "evidence_bundle", "evb_ghost")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "evidence_bundle_not_found"
    assert error["details"]["evidence_bundle_id"] == "evb_ghost"


def test_wrong_target_type_for_existing_resource_is_404(client):
    claim, bundle = _setup_claim_and_bundle(client)
    r1 = _get(client, "claim", bundle["id"])
    r2 = _get(client, "evidence_bundle", claim["id"])
    assert r1.status_code == 404
    assert r1.json()["error"]["code"] == "claim_not_found"
    assert r2.status_code == 404
    assert r2.json()["error"]["code"] == "evidence_bundle_not_found"


# --- Validation boundary -----------------------------------------------------


def test_missing_target_type_or_target_id_is_422(client):
    _setup_claim(client)
    no_type = client.get(PATH, params={"target_id": "clm_x"})
    assert no_type.status_code == 422
    assert no_type.json()["error"]["code"] == "validation_error"

    no_id = client.get(PATH, params={"target_type": "claim"})
    assert no_id.status_code == 422
    assert no_id.json()["error"]["code"] == "validation_error"

    neither = client.get(PATH)
    assert neither.status_code == 422
    assert neither.json()["error"]["code"] == "validation_error"


def test_blank_or_unknown_target_type_is_422(client):
    claim = _setup_claim(client)
    for value in ("", "   ", "Claim", "claim ", " content", "evidence"):
        resp = _get(client, value, claim["id"])
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_blank_target_id_is_422(client):
    for value in ("", "   ", "\t"):
        resp = _get(client, "claim", value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_limit_boundaries_one_and_one_hundred_are_accepted(client):
    claim = _setup_claim(client)
    assert _get(client, "claim", claim["id"], limit="1").status_code == 200
    assert _get(client, "claim", claim["id"], limit="100").status_code == 200


def test_limit_must_be_a_plain_integer_in_range(client):
    claim = _setup_claim(client)
    for value in (
        "",
        "   ",
        "abc",
        "1.5",
        "0",
        "-1",
        "101",
        "+1",
        " 1",
        "1 ",
        "1.0",
        "0x1",
    ):
        resp = _get(client, "claim", claim["id"], limit=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_422(client):
    claim = _setup_claim(client)
    base = f"{PATH}?target_id={claim['id']}"
    for url in (
        f"{base}&target_type=claim&target_type=evidence_bundle",
        f"{base}&target_type=claim&target_type=claim",
        f"{base}&target_type=claim&target_id={claim['id']}",
        f"{base}&target_type=claim&limit=1&limit=2",
        f"{base}&target_type=claim&cursor=a&cursor=b",
    ):
        resp = client.get(url)
        assert resp.status_code == 422, url
        assert resp.json()["error"]["code"] == "validation_error"


def test_unknown_parameters_are_422(client):
    claim = _setup_claim(client)
    for url in (
        f"{PATH}?target_type=claim&target_id={claim['id']}&min_signers=1",
        f"{PATH}?target_type=claim&target_id={claim['id']}&foo=bar",
    ):
        resp = client.get(url)
        assert resp.status_code == 422, url
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_empty_body_is_422_rejected_before_any_read(client, db_session):
    claim = _setup_claim(client)
    events_before = db_session.scalar(select(func.count()).select_from(AuditEvent))
    for body in (b"{}", b" ", b"{not valid json", b"[]"):
        resp = client.request(
            "GET",
            PATH,
            params={"target_type": "claim", "target_id": claim["id"]},
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == events_before
    )


def test_validation_errors_take_precedence_over_missing_target(client):
    # A non-existent target with an otherwise malformed request is 422, not
    # 404: parameters are validated before any existence lookup.
    resp = client.get(
        PATH,
        params={"target_type": "claim", "target_id": "clm_ghost", "limit": "abc"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    resp = client.get(
        PATH, params={"target_type": "nope", "target_id": "clm_ghost"}
    )
    assert resp.status_code == 422

    resp = client.get(PATH, params={"target_type": "claim", "target_id": "   "})
    assert resp.status_code == 422

    resp = client.get(
        PATH,
        params={"target_type": "claim", "target_id": "clm_ghost", "cursor": "bad"},
    )
    assert resp.status_code == 422


# --- Pagination --------------------------------------------------------------


def test_pages_resume_without_duplication_or_omission(client):
    claim = _three_signer_world(client)
    first = _get(client, "claim", claim["id"], limit="2")
    assert first.status_code == 200
    page_one = first.json()
    # The total covers every page, including the first.
    assert page_one["count"] == 3
    assert [i["signer_actor_id"] for i in page_one["items"]] == ["org-1", "org-2"]
    cursor = page_one["next_cursor"]
    assert cursor is not None

    second = _get(client, "claim", claim["id"], limit="2", cursor=cursor)
    assert second.status_code == 200
    page_two = second.json()
    assert page_two["count"] == 3
    assert [i["signer_actor_id"] for i in page_two["items"]] == ["org-3"]
    assert page_two["next_cursor"] is None

    seen = [i["signer_actor_id"] for i in page_one["items"] + page_two["items"]]
    assert seen == ["org-1", "org-2", "org-3"]


def test_default_limit_is_fifty(client):
    claim = _setup_claim(client)
    # 51 distinct signers exceed the default page: the first page carries 50.
    for index in range(51):
        actor_id = f"org-bulk-{index:03d}"
        _make_actor(client, actor_id)
        _attest(
            client,
            "claim",
            claim["id"],
            seed=hashlib.sha256(actor_id.encode()).digest(),
            signer_actor_id=actor_id,
        )
    body = _get(client, "claim", claim["id"]).json()
    assert body["count"] == 51
    assert len(body["items"]) == 50
    assert body["next_cursor"] is not None


def test_replayed_cursor_returns_the_same_page(client):
    claim = _three_signer_world(client)
    cursor = _get(client, "claim", claim["id"], limit="1").json()["next_cursor"]
    assert cursor is not None
    page = _get(client, "claim", claim["id"], limit="1", cursor=cursor)
    replay = _get(client, "claim", claim["id"], limit="1", cursor=cursor)
    assert page.status_code == 200
    assert replay.status_code == 200
    assert replay.content == page.content


def test_cursor_at_or_past_the_tail_returns_empty_page_with_count(client, app):
    claim = _three_signer_world(client)

    def mint(offset):
        return pagination.encode_typed_cursor(
            app.state.trust_evaluation_signers_cursor_secret,
            pagination.TRUST_EVALUATION_SIGNERS_CURSOR,
            {
                "target_type": "claim",
                "target_id": claim["id"],
                "limit": 50,
                "offset": offset,
            },
        )

    for offset in (3, 99):
        resp = _get(client, "claim", claim["id"], cursor=mint(offset))
        assert resp.status_code == 200
        assert resp.json() == {
            "target_type": "claim",
            "target_id": claim["id"],
            "items": [],
            "count": 3,
            "next_cursor": None,
        }


def test_cursor_binds_the_target_and_limit(client):
    claim = _three_signer_world(client)
    cursor = _get(client, "claim", claim["id"], limit="2").json()["next_cursor"]
    assert cursor is not None
    other_claim = _create_claim(client, _create_content(client, digest=DIGEST_C)["id"])
    # A different limit, a different target id, a different target type, or
    # a dropped target parameter all mismatch the cursor's query.
    for params in (
        {"target_type": "claim", "target_id": claim["id"], "limit": "3"},
        {"target_type": "claim", "target_id": other_claim["id"], "limit": "2"},
        {"target_type": "evidence_bundle", "target_id": claim["id"], "limit": "2"},
        {"limit": "2"},
        {},
    ):
        resp = client.get(PATH, params={**params, "cursor": cursor})
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_blank_malformed_and_tampered_cursors_are_422(client):
    claim = _three_signer_world(client)
    valid = _get(client, "claim", claim["id"], limit="1").json()["next_cursor"]
    assert valid is not None
    tampered = valid[:-1] + ("A" if valid[-1] != "A" else "B")
    for bad in ("", "   ", "not-a-cursor", tampered):
        resp = _get(client, "claim", claim["id"], cursor=bad)
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_foreign_family_cursor_is_422(client, app):
    claim = _three_signer_world(client)
    # A well-formed cursor minted by another endpoint family never resumes
    # this retrieval.
    foreign = pagination.encode_typed_cursor(
        app.state.trust_policies_cursor_secret,
        pagination.TRUST_POLICIES_CURSOR,
        {"actor_id": None, "limit": 50, "offset": 1},
    )
    resp = _get(client, "claim", claim["id"], cursor=foreign)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Method boundary ----------------------------------------------------------


def test_non_get_methods_are_405(client):
    claim = _setup_claim(client)
    query = f"?target_type=claim&target_id={claim['id']}"
    for method in ("post", "put", "patch", "delete"):
        resp = getattr(client, method)(PATH + query)
        assert resp.status_code == 405, method
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Read-only guarantee -------------------------------------------------------


def test_queries_and_failures_write_nothing(client, db_session):
    claim = _three_signer_world(client)
    attestations_before = db_session.scalar(
        select(func.count()).select_from(Attestation)
    )
    revocations_before = db_session.scalar(
        select(func.count()).select_from(AttestationRevocation)
    )
    audit_before = db_session.scalar(select(func.count()).select_from(AuditEvent))

    assert _get(client, "claim", claim["id"]).status_code == 200
    assert _get(client, "claim", claim["id"], limit="1").status_code == 200
    assert _get(client, "claim", "clm_ghost").status_code == 404
    assert _get(client, "claim", claim["id"], limit="0").status_code == 422
    assert _get(client, "claim", claim["id"], cursor="bad").status_code == 422

    assert (
        db_session.scalar(select(func.count()).select_from(Attestation))
        == attestations_before
    )
    assert (
        db_session.scalar(select(func.count()).select_from(AttestationRevocation))
        == revocations_before
    )
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audit_before
    )


def test_signers_list_identically_across_an_app_restart(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    claim = _three_signer_world(file_client)
    first = file_client.get(
        PATH, params={"target_type": "claim", "target_id": claim["id"]}
    )
    assert first.status_code == 200
    assert first.json()["next_cursor"] is None

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as restarted_client:
        second = restarted_client.get(
            PATH, params={"target_type": "claim", "target_id": claim["id"]}
        )
        assert second.status_code == 200
        assert second.json() == first.json()
