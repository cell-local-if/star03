"""Tests for the read-only reviewer batch trust evaluation endpoint.

Covers POST /v1/trust-evaluation-batches: in-order results without
deduplication, per-item min_signers default and thresholds, the same
qualified-signer semantics as the single-target GET (verified,
non-revoked, distinct signer_actor_id), the 404 batch failure reporting
the first missing target in request order, full structural validation
before any target lookup (422 validation_error), query-parameter
rejection, and the no-write/no-auth guarantee. All tests are
deterministic and offline (signatures are produced by the stdlib test
signer).
"""

from __future__ import annotations

import base64
import hashlib

from sqlalchemy import func, select

from provenance.models import Attestation, AuditEvent
from provenance.signing import attestation_message_bytes
from tests.helpers import (
    DIGEST_B,
    content_payload,
    create_actor,
    ed25519_public_key,
    ed25519_sign,
    SEED_A,
    SEED_B,
)

EVIDENCE_DIGEST = hashlib.sha256(b"evidence-trust-batch").hexdigest()


# --- Setup helpers ----------------------------------------------------------


def _create_content(client, actor_id="org-1", digest=None):
    resp = client.post(
        "/v1/contents",
        json=content_payload(actor_id=actor_id, digest=digest or DIGEST_B),
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


def _setup_claim_and_bundle(client):
    claim = _setup_claim(client)
    return claim, _create_bundle(client, claim["id"])


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


def _revoke(client, attestation_id, revoker_actor_id="org-1"):
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


def _batch(client, items, **kwargs):
    return client.post("/v1/trust-evaluation-batches", json={"items": items}, **kwargs)


def _item(target_type, target_id, **extra):
    return {"target_type": target_type, "target_id": target_id, **extra}


# --- Successful batches -----------------------------------------------------


def test_batch_evaluates_mixed_targets_in_request_order(client):
    claim, bundle = _setup_claim_and_bundle(client)
    _attest(client, "claim", claim["id"])

    resp = _batch(
        client,
        [
            _item("claim", claim["id"]),
            _item("evidence_bundle", bundle["id"]),
            _item("claim", claim["id"], min_signers=2),
        ],
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "items": [
            {
                "target_type": "claim",
                "target_id": claim["id"],
                "min_signers": 1,
                "qualified_signer_count": 1,
                "decision": "trusted",
            },
            {
                "target_type": "evidence_bundle",
                "target_id": bundle["id"],
                "min_signers": 1,
                "qualified_signer_count": 0,
                "decision": "untrusted",
            },
            {
                "target_type": "claim",
                "target_id": claim["id"],
                "min_signers": 2,
                "qualified_signer_count": 1,
                "decision": "untrusted",
            },
        ],
        "count": 3,
    }


def test_single_item_batch_matches_the_single_target_evaluation(client):
    claim = _setup_claim(client)
    _attest(client, "claim", claim["id"])

    single = client.get(
        "/v1/trust-evaluations",
        params={"target_type": "claim", "target_id": claim["id"], "min_signers": 1},
    )
    batch = _batch(client, [_item("claim", claim["id"], min_signers=1)])
    assert batch.status_code == 200
    assert batch.json()["items"] == [single.json()]
    assert batch.json()["count"] == 1


def test_min_signers_defaults_to_one_and_boundary_values_are_accepted(client):
    claim = _setup_claim(client)
    resp = _batch(
        client,
        [
            _item("claim", claim["id"]),
            _item("claim", claim["id"], min_signers=1),
            _item("claim", claim["id"], min_signers=100),
        ],
    )
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert [item["min_signers"] for item in items] == [1, 1, 100]
    assert [item["decision"] for item in items] == [
        "untrusted",
        "untrusted",
        "untrusted",
    ]


def test_duplicate_items_are_evaluated_not_merged(client):
    claim = _setup_claim(client)
    _attest(client, "claim", claim["id"])

    resp = _batch(
        client,
        [
            _item("claim", claim["id"]),
            _item("claim", claim["id"]),
            _item("claim", claim["id"]),
        ],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    assert len(body["items"]) == 3
    assert body["items"][0] == body["items"][1] == body["items"][2]


def test_distinct_signers_and_revocations_match_single_target_semantics(client):
    claim = _setup_claim(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    # Same actor under two keys qualifies once; a second actor qualifies once.
    _attest(client, "claim", claim["id"], seed=SEED_A, signer_actor_id="org-1")
    _attest(client, "claim", claim["id"], seed=SEED_B, signer_actor_id="org-1")
    att_two = _attest(client, "claim", claim["id"], seed=SEED_A, signer_actor_id="org-2")

    resp = _batch(client, [_item("claim", claim["id"], min_signers=2)])
    assert resp.json()["items"][0]["qualified_signer_count"] == 2
    assert resp.json()["items"][0]["decision"] == "trusted"

    # Revoking org-2's only attestation drops that actor from the count.
    _revoke(client, att_two["id"])
    resp = _batch(client, [_item("claim", claim["id"], min_signers=2)])
    item = resp.json()["items"][0]
    assert item["qualified_signer_count"] == 1
    assert item["decision"] == "untrusted"


def test_repeated_calls_return_identical_results(client):
    claim = _setup_claim(client)
    _attest(client, "claim", claim["id"])
    payload = [_item("claim", claim["id"], min_signers=1)]
    first = _batch(client, payload)
    second = _batch(client, payload)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


# --- Missing-target boundary -------------------------------------------------


def test_first_missing_target_in_request_order_fails_the_whole_batch(client):
    claim, bundle = _setup_claim_and_bundle(client)
    resp = _batch(
        client,
        [
            _item("claim", claim["id"]),
            _item("evidence_bundle", "evb_ghost"),
            _item("claim", "clm_ghost"),
        ],
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "evidence_bundle_not_found"
    assert error["details"] == {
        "target_type": "evidence_bundle",
        "target_id": "evb_ghost",
    }


def test_missing_claim_reports_claim_not_found(client):
    _setup_claim(client)
    resp = _batch(client, [_item("claim", "clm_ghost")])
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "claim_not_found"
    assert error["details"] == {"target_type": "claim", "target_id": "clm_ghost"}


def test_wrong_target_type_for_existing_resource_is_404(client):
    claim, bundle = _setup_claim_and_bundle(client)
    r1 = _batch(client, [_item("claim", bundle["id"])])
    assert r1.status_code == 404
    assert r1.json()["error"]["code"] == "claim_not_found"
    assert r1.json()["error"]["details"] == {
        "target_type": "claim",
        "target_id": bundle["id"],
    }

    r2 = _batch(client, [_item("evidence_bundle", claim["id"])])
    assert r2.status_code == 404
    assert r2.json()["error"]["code"] == "evidence_bundle_not_found"


# --- Validation boundary -----------------------------------------------------


def _assert_validation_error(resp):
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def test_empty_missing_or_oversized_items_list_is_422(client):
    _setup_claim(client)
    _assert_validation_error(_batch(client, []))
    _assert_validation_error(
        client.post("/v1/trust-evaluation-batches", json={})
    )
    _assert_validation_error(
        client.post("/v1/trust-evaluation-batches", json=[])
    )
    _assert_validation_error(
        _batch(client, [_item("claim", "clm_x")] * 101)
    )


def test_exactly_one_hundred_items_are_accepted(client):
    claim = _setup_claim(client)
    resp = _batch(client, [_item("claim", claim["id"])] * 100)
    assert resp.status_code == 200
    assert resp.json()["count"] == 100
    assert len(resp.json()["items"]) == 100


def test_undeclared_top_level_or_item_fields_are_422(client):
    claim = _setup_claim(client)
    _assert_validation_error(
        client.post(
            "/v1/trust-evaluation-batches",
            json={"items": [_item("claim", claim["id"])], "limit": 1},
        )
    )
    _assert_validation_error(
        _batch(client, [_item("claim", claim["id"], decision="trusted")])
    )


def test_missing_item_fields_are_422(client):
    claim = _setup_claim(client)
    _assert_validation_error(_batch(client, [{"target_id": claim["id"]}]))
    _assert_validation_error(_batch(client, [{"target_type": "claim"}]))
    _assert_validation_error(_batch(client, [{}]))


def test_invalid_target_type_or_blank_target_id_is_422(client):
    claim = _setup_claim(client)
    for value in ("Claim", "claim ", " content", "evidence", "", "   "):
        _assert_validation_error(_batch(client, [_item(value, claim["id"])]))
    for value in ("", "   ", "\t"):
        _assert_validation_error(_batch(client, [_item("claim", value)]))
    # Non-string identifiers are never coerced.
    _assert_validation_error(_batch(client, [_item("claim", 123)]))
    _assert_validation_error(_batch(client, [_item("claim", None)]))


def test_min_signers_must_be_a_plain_integer_in_range(client):
    claim = _setup_claim(client)
    for value in (True, False, "1", " 1", "1 ", "abc", 1.5, 1.0, 8.0, 0, -1, 101, None):
        _assert_validation_error(
            _batch(client, [_item("claim", claim["id"], min_signers=value)])
        )


def test_malformed_json_body_is_422(client):
    resp = client.post(
        "/v1/trust-evaluation-batches",
        content=b'{"items": [',
        headers={"Content-Type": "application/json"},
    )
    _assert_validation_error(resp)


def test_any_query_parameter_is_422(client):
    claim = _setup_claim(client)
    resp = client.post(
        f"/v1/trust-evaluation-batches?target_id={claim['id']}",
        json={"items": [_item("claim", claim["id"])]},
    )
    _assert_validation_error(resp)


def test_validation_precedes_any_target_lookup(client):
    # A structurally invalid batch naming a non-existent target is a 422,
    # never a 404: the whole body is validated before any target is read.
    _setup_claim(client)
    resp = _batch(client, [_item("claim", "clm_ghost", min_signers="abc")])
    _assert_validation_error(resp)

    resp = _batch(client, [_item("nope", "clm_ghost")])
    _assert_validation_error(resp)

    resp = client.post(
        "/v1/trust-evaluation-batches?foo=bar",
        json={"items": [_item("claim", "clm_ghost")]},
    )
    _assert_validation_error(resp)


# --- Read-only guarantee -----------------------------------------------------


def test_batch_writes_no_resources_or_audit_events(client, db_session):
    claim = _setup_claim(client)
    _attest(client, "claim", claim["id"])

    attestations_before = db_session.scalar(
        select(func.count()).select_from(Attestation)
    )
    audit_before = db_session.scalar(select(func.count()).select_from(AuditEvent))

    resp = _batch(
        client,
        [
            _item("claim", claim["id"]),
            _item("claim", claim["id"], min_signers=2),
        ],
    )
    assert resp.status_code == 200
    # Including failed batches, which also write nothing.
    assert _batch(client, [_item("claim", "clm_ghost")]).status_code == 404
    _assert_validation_error(_batch(client, []))

    assert (
        db_session.scalar(select(func.count()).select_from(Attestation))
        == attestations_before
    )
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audit_before
    )
