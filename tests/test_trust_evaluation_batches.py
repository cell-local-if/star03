"""Tests for the read-only reviewer batch trust evaluation endpoint.

Covers POST /v1/trust-evaluation-batches: live per-target decisions under
the exact single-target counting (verified, non-revoked attestations,
deduplicated by signing actor), input-order preservation, unmerged
duplicates, per-item thresholds and the count field, both target types,
the 404 boundary (the first missing/type-mismatched target fails the whole
batch, validation precedes existence), the 422 boundary (malformed JSON,
structure, cardinality, strict types, strict min_signers, query
parameters), compact JSON rendering, and the no-write/no-auth/idempotence
guarantees. All tests are deterministic and offline (signatures are
produced by the stdlib test signer).
"""

from __future__ import annotations

import base64
import hashlib
import json

from sqlalchemy import func, select

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

EVIDENCE_DIGEST = hashlib.sha256(b"evidence-batch-trust").hexdigest()
BATCH_PATH = "/v1/trust-evaluation-batches"


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
            "payload": {"statement": "batch trust me"},
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
            "public_key": base64.b64encode(ed25519_public_key(seed)).decode(
                "ascii"
            ),
            "signature": base64.b64encode(signature).decode("ascii"),
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _revoke(
    client, attestation_id, *, revoker_actor_id="org-1", reason="key compromised"
):
    resp = client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": attestation_id,
            "revoker_actor_id": revoker_actor_id,
            "reason": reason,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _batch(client, items, **kwargs):
    return client.post(BATCH_PATH, json={"items": items}, **kwargs)


def _item(target_type, target_id, min_signers=None):
    entry = {"target_type": target_type, "target_id": target_id}
    if min_signers is not None:
        entry["min_signers"] = min_signers
    return entry


# --- Successful batches ------------------------------------------------------


def test_batch_mixed_targets_preserve_order_defaults_and_count(client):
    claim, bundle = _setup_claim_and_bundle(client)
    _attest(client, "evidence_bundle", bundle["id"])

    resp = _batch(
        client,
        [
            _item("claim", claim["id"]),
            _item("evidence_bundle", bundle["id"]),
            _item("claim", claim["id"], min_signers=2),
        ],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 3
    assert body["items"] == [
        {
            "target_type": "claim",
            "target_id": claim["id"],
            "min_signers": 1,
            "qualified_signer_count": 0,
            "decision": "untrusted",
        },
        {
            "target_type": "evidence_bundle",
            "target_id": bundle["id"],
            "min_signers": 1,
            "qualified_signer_count": 1,
            "decision": "trusted",
        },
        {
            "target_type": "claim",
            "target_id": claim["id"],
            "min_signers": 2,
            "qualified_signer_count": 0,
            "decision": "untrusted",
        },
    ]


def test_batch_threshold_boundary_equality_is_trusted(client):
    claim = _setup_claim(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    _attest(client, "claim", claim["id"], seed=SEED_A, signer_actor_id="org-1")
    _attest(client, "claim", claim["id"], seed=SEED_B, signer_actor_id="org-2")

    resp = _batch(
        client,
        [
            _item("claim", claim["id"], min_signers=2),
            _item("claim", claim["id"], min_signers=3),
        ],
    )
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert items[0]["qualified_signer_count"] == 2
    assert items[0]["decision"] == "trusted"
    assert items[1]["qualified_signer_count"] == 2
    assert items[1]["decision"] == "untrusted"


def test_batch_same_signer_distinct_keys_counts_once(client):
    claim = _setup_claim(client)
    _attest(client, "claim", claim["id"], seed=SEED_A)
    _attest(client, "claim", claim["id"], seed=SEED_B)

    resp = _batch(client, [_item("claim", claim["id"])])
    assert resp.json()["items"][0]["qualified_signer_count"] == 1

    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    _attest(client, "claim", claim["id"], seed=SEED_A, signer_actor_id="org-2")
    resp = _batch(client, [_item("claim", claim["id"])])
    assert resp.json()["items"][0]["qualified_signer_count"] == 2


def test_batch_duplicate_items_are_not_merged(client):
    claim = _setup_claim(client)
    _attest(client, "claim", claim["id"])

    resp = _batch(
        client,
        [
            _item("claim", claim["id"], min_signers=1),
            _item("claim", claim["id"], min_signers=2),
            _item("claim", claim["id"]),
        ],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    assert [item["min_signers"] for item in body["items"]] == [1, 2, 1]
    assert [item["decision"] for item in body["items"]] == [
        "trusted",
        "untrusted",
        "trusted",
    ]
    assert all(item["target_id"] == claim["id"] for item in body["items"])


def test_batch_exact_target_scoping_and_revocation(client):
    claim_one = _setup_claim(client)
    content_two = _create_content(client, digest=DIGEST_C)
    claim_two = _create_claim(client, content_two["id"])
    bundle = _create_bundle(client, claim_one["id"])
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    other_att = _attest(
        client, "claim", claim_two["id"], seed=SEED_B, signer_actor_id="org-2"
    )
    own_att = _attest(client, "evidence_bundle", bundle["id"])

    resp = _batch(
        client,
        [
            _item("claim", claim_one["id"]),
            _item("evidence_bundle", bundle["id"]),
            _item("claim", claim_two["id"]),
        ],
    )
    counts = [item["qualified_signer_count"] for item in resp.json()["items"]]
    assert counts == [0, 1, 1]

    # Revoking the bundle's only qualifying proof drops that entry to zero;
    # the other target's attestation is untouched.
    _revoke(client, own_att["id"])
    resp = _batch(
        client,
        [_item("evidence_bundle", bundle["id"]), _item("claim", claim_two["id"])],
    )
    counts = [item["qualified_signer_count"] for item in resp.json()["items"]]
    assert counts == [0, 1]

    # Unrelated revocation never restores the revoked entry.
    _revoke(client, other_att["id"], revoker_actor_id="org-2")
    resp = _batch(client, [_item("evidence_bundle", bundle["id"])])
    assert resp.json()["items"][0]["qualified_signer_count"] == 0
    assert resp.json()["items"][0]["decision"] == "untrusted"


def test_batch_of_100_items_is_accepted(client):
    claim = _setup_claim(client)
    resp = _batch(client, [_item("claim", claim["id"])] * 100)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 100
    assert len(body["items"]) == 100


def test_batch_verbatim_target_id_with_surrounding_whitespace_is_literal(client):
    claim = _setup_claim(client)
    # Whitespace on a non-blank id is part of the identifier: it is not
    # trimmed, so no claim matches and the batch is a type-matched 404.
    resp = _batch(
        client,
        [
            _item("claim", claim["id"]),
            _item("claim", f"  {claim['id']}  "),
        ],
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "claim_not_found"
    assert error["details"] == {
        "target_type": "claim",
        "target_id": f"  {claim['id']}  ",
    }


def test_batch_is_computed_live(client):
    claim = _setup_claim(client)
    first = _batch(client, [_item("claim", claim["id"])]).json()["items"][0]
    assert first["qualified_signer_count"] == 0

    _attest(client, "claim", claim["id"])
    second = _batch(client, [_item("claim", claim["id"])]).json()["items"][0]
    assert second["qualified_signer_count"] == 1
    assert second["decision"] == "trusted"


# --- Missing-resource boundary ----------------------------------------------


def test_batch_unknown_claim_is_404_with_first_target_details(client):
    claim = _setup_claim(client)
    resp = _batch(
        client,
        [
            _item("claim", claim["id"]),
            _item("claim", "clm_ghost"),
        ],
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "claim_not_found"
    assert error["details"] == {"target_type": "claim", "target_id": "clm_ghost"}
    # A failed batch never returns a partial items array.
    assert "items" not in resp.json()


def test_batch_unknown_evidence_bundle_is_404(client):
    _setup_claim(client)
    resp = _batch(client, [_item("evidence_bundle", "evb_ghost")])
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "evidence_bundle_not_found"
    assert error["details"] == {
        "target_type": "evidence_bundle",
        "target_id": "evb_ghost",
    }


def test_batch_reports_the_first_missing_target_in_request_order(client):
    claim, bundle = _setup_claim_and_bundle(client)
    resp = _batch(
        client,
        [
            _item("claim", claim["id"]),
            _item("claim", "clm_first_ghost"),
            _item("evidence_bundle", "evb_second_ghost"),
        ],
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "claim_not_found"
    assert error["details"]["target_id"] == "clm_first_ghost"

    # When the first missing target is a bundle, its type drives the code.
    resp = _batch(
        client,
        [
            _item("claim", claim["id"]),
            _item("evidence_bundle", "evb_first_ghost"),
            _item("claim", "clm_second_ghost"),
        ],
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "evidence_bundle_not_found"
    assert resp.json()["error"]["details"]["target_id"] == "evb_first_ghost"

    # An existing bundle/claim id declared with the opposite type is itself
    # the first (and only) mismatch.
    resp = _batch(
        client,
        [
            _item("claim", bundle["id"]),
            _item("evidence_bundle", claim["id"]),
        ],
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "claim_not_found"
    assert resp.json()["error"]["details"]["target_id"] == bundle["id"]


# --- Validation boundary -----------------------------------------------------


def _assert_422(resp):
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"
    return resp.json()["error"]["details"]["issues"]


def test_batch_requires_valid_json_object(client):
    for raw in (
        b"",
        b"{not json",
        b"[]",
        b'"items"',
        b"42",
        b"true",
        b"null",
        b"{",
        b'{"items": [',
    ):
        resp = client.post(
            BATCH_PATH,
            content=raw,
            headers={"content-type": "application/json"},
        )
        issues = _assert_422(resp)
        assert issues, raw


def test_batch_top_level_structure_is_validated(client):
    claim = _setup_claim(client)

    # Missing items / items not an array / empty array.
    for payload in ({}, {"items": None}, {"items": {}}, {"items": []}):
        _assert_422(client.post(BATCH_PATH, json=payload))

    # An undeclared top-level field is rejected rather than discarded.
    issues = _assert_422(
        client.post(
            BATCH_PATH,
            json={
                "items": [_item("claim", claim["id"])],
                "extra": 1,
            },
        )
    )
    assert any(issue["loc"][-1] == "extra" for issue in issues)

    # items as a JSON object is not an array.
    object_items = {"items": {"0": _item("claim", claim["id"])}}
    _assert_422(client.post(BATCH_PATH, json=object_items))


def test_batch_cardinality_above_100_is_422(client):
    claim = _setup_claim(client)
    resp = _batch(client, [_item("claim", claim["id"])] * 101)
    _assert_422(resp)


def test_batch_items_must_be_objects_with_exactly_three_fields(client):
    claim = _setup_claim(client)

    for non_object in ("claim", 42, True, None, ["claim", claim["id"]]):
        _assert_422(_batch(client, [non_object]))

    valid = _item("claim", claim["id"])

    # Missing required fields.
    _assert_422(_batch(client, [{"target_id": claim["id"]}]))
    _assert_422(_batch(client, [{"target_type": "claim"}]))
    _assert_422(_batch(client, [{"min_signers": 1}]))

    # Undeclared item field.
    issues = _assert_422(_batch(client, [{**valid, "bogus": 1}]))
    assert any(issue["loc"][-1] == "bogus" for issue in issues)


def test_batch_target_type_must_be_one_of_the_two_literals(client):
    claim = _setup_claim(client)
    for bad in (
        "Claim",
        "claim ",
        " claim",
        "EVIDENCE_BUNDLE",
        "evidence",
        "attestation",
        "",
        "   ",
        1,
        True,
        None,
        ["claim"],
    ):
        _assert_422(_batch(client, [{"target_type": bad, "target_id": claim["id"]}]))


def test_batch_target_id_must_be_a_nonblank_string(client):
    for bad in ("", "   ", "\t", 42, True, None, ["clm_x"], {"id": "x"}):
        _assert_422(_batch(client, [{"target_type": "claim", "target_id": bad}]))


def test_batch_min_signers_must_be_a_plain_integer_in_range(client):
    claim = _setup_claim(client)

    def with_threshold(value):
        return _batch(
            client,
            [
                {
                    "target_type": "claim",
                    "target_id": claim["id"],
                    "min_signers": value,
                }
            ],
        )

    # Booleans, strings, fractions, exponents, null, and out-of-range values
    # are never coerced or defaulted.
    for bad in (
        True,
        False,
        "1",
        " 1",
        "1 ",
        1.5,
        1.0,
        0.0,
        1e2,
        2e0,
        0,
        -1,
        101,
        None,
        [],
        {},
    ):
        _assert_422(with_threshold(bad))

    # Boundary values 1 and 100 are accepted.
    for value in (1, 100):
        resp = with_threshold(value)
        assert resp.status_code == 200, (value, resp.text)
        assert resp.json()["items"][0]["min_signers"] == value


def test_batch_validation_precedes_any_existence_lookup(client):
    # Every malformed batch is a 422 even when a referenced target (early or
    # late in the batch) does not exist: validation completes first.
    _setup_claim(client)

    cases = [
        {"items": []},
        {"items": [_item("claim", "clm_ghost")], "extra": True},
        {"items": [{"target_type": "nope", "target_id": "clm_ghost"}]},
        {"items": [{"target_type": "claim", "target_id": "   "}]},
        {
            "items": [
                {
                    "target_type": "claim",
                    "target_id": "clm_ghost",
                    "min_signers": 0,
                }
            ]
        },
        {
            "items": [
                {
                    "target_type": "claim",
                    "target_id": "clm_ghost",
                    "min_signers": True,
                }
            ]
        },
        {"items": [_item("claim", "clm_ghost"), "not-an-object"]},
        {"items": [_item("claim", "clm_ghost")] * 101},
    ]
    for payload in cases:
        resp = client.post(BATCH_PATH, json=payload)
        _assert_422(resp)


def test_batch_any_query_parameter_is_422(client):
    claim = _setup_claim(client)
    for url in (
        f"{BATCH_PATH}?foo=bar",
        f"{BATCH_PATH}?min_signers=1",
        f"{BATCH_PATH}?target_type=claim",
    ):
        resp = client.post(url, json={"items": [_item("claim", claim["id"])]})
        _assert_422(resp)


# --- Wire format, read-only and idempotence guarantees ------------------------


def test_batch_success_body_is_compact_json_with_one_newline(client):
    claim, bundle = _setup_claim_and_bundle(client)
    resp = _batch(
        client,
        [_item("claim", claim["id"]), _item("evidence_bundle", bundle["id"])],
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert not raw.endswith(b"\n\n")
    # Compact separators: no whitespace after commas or colons.
    assert b": " not in raw
    assert b", " not in raw
    # Top-level member order: items then count.
    assert raw.index(b'"items"') < raw.index(b'"count"')
    # Re-decoding yields the same compact object with integral counts.
    decoded = json.loads(raw)
    assert set(decoded) == {"items", "count"}
    for entry in decoded["items"]:
        assert set(entry) == {
            "target_type",
            "target_id",
            "min_signers",
            "qualified_signer_count",
            "decision",
        }
        assert isinstance(entry["min_signers"], int)
        assert isinstance(entry["qualified_signer_count"], int)


def test_batch_requires_no_authentication_and_writes_nothing(client, db_session):
    claim = _setup_claim(client)
    _attest(client, "claim", claim["id"])

    attestations_before = db_session.scalar(
        select(func.count()).select_from(Attestation)
    )
    audit_before = db_session.scalar(select(func.count()).select_from(AuditEvent))

    payload = {
        "items": [
            _item("claim", claim["id"], min_signers=1),
            _item("claim", claim["id"], min_signers=2),
        ]
    }
    # No Authorization/Signature headers of any kind are sent.
    first = client.post(BATCH_PATH, json=payload)
    assert first.status_code == 200
    second = client.post(BATCH_PATH, json=payload)
    assert second.status_code == 200
    # Repeated calls are byte-for-byte identical for the same state.
    assert first.content == second.content

    # Successful and failing (422/404) batches all write nothing.
    assert client.post(BATCH_PATH, json={"items": []}).status_code == 422
    assert _batch(client, [_item("claim", "clm_ghost")]).status_code == 404

    assert (
        db_session.scalar(select(func.count()).select_from(Attestation))
        == attestations_before
    )
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audit_before
    )


def test_batch_response_never_carries_payloads_or_bytes(client):
    claim, bundle = _setup_claim_and_bundle(client)
    _attest(client, "claim", claim["id"])
    _attest(client, "evidence_bundle", bundle["id"])

    resp = _batch(
        client,
        [_item("claim", claim["id"]), _item("evidence_bundle", bundle["id"])],
    )
    raw = resp.text
    # No signature/public-key material and no content/payload fields.
    for absent in (
        "signature",
        "public_key",
        "payload",
        "digest_hex",
        "metadata",
        "media_type",
        "content",
        "revocation",
    ):
        assert absent not in raw, absent
