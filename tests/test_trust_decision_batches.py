"""Tests for protected batch authorization decisions.

Covers ``POST /v1/trust-decision-batches``: 1..100 targets decided in one
call under the calling subject's current immutable policy, the
``trusted``/``untrusted`` outcomes with
``threshold_met``/``below_threshold``/``policy_missing`` reasons,
qualified-signer counting (verified, non-revoked, deduplicated by signer),
input-order preservation and unmerged duplicates, the no-policy and
revoked-policy paths that never look up or leak a target, the single
type-matched 404 for the first missing target once a policy exists, the
opaque-404/422 credential boundary, strict body and query-parameter
validation, the compact one-newline JSON wire format, and the strict
read-only guarantee.

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
from provenance.models import ActorTrustPolicy, Attestation, AuditEvent
from provenance.signing import attestation_message_bytes
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    DIGEST_C,
    content_payload,
    create_actor,
    ed25519_public_key,
    ed25519_sign,
    SEED_A,
    SEED_B,
)

BATCH_PATH = "/v1/trust-decision-batches"
POLICIES_PATH = "/v1/trust-policies"
REVOCATIONS_PATH = "/v1/trust-policy-revocations"
SEED_C = b"test-ed25519-seed-c-00000000000000"[:32]
SEED_D = b"test-ed25519-seed-d-00000000000000"[:32]
EVIDENCE_DIGEST = hashlib.sha256(b"evidence-decision-batch").hexdigest()


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


def _make_bundle(client, claim_id):
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


def _attest(client, target_type, target_id, actor_id, seed):
    signature = ed25519_sign(
        seed, attestation_message_bytes(target_type, target_id, actor_id)
    )
    resp = client.post(
        "/v1/attestations",
        json={
            "target_type": target_type,
            "target_id": target_id,
            "signer_actor_id": actor_id,
            "public_key": base64.b64encode(ed25519_public_key(seed)).decode(),
            "signature": base64.b64encode(signature).decode(),
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _world(client):
    """Three actors; org-1 (the usual caller) holds a current key."""
    create_actor(client)  # org-1
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    create_actor(client, actor_id="org-3", name="Third Org", type="organization")
    # org-1's own attestation gives org-1 a current authentication key.
    _attest(client, "claim", _make_claim(client, "org-1")["id"], "org-1", SEED_A)


def _target_claim(client):
    """A second claim (the decision target), distinct from org-1's key claim."""
    return _make_claim(client, "org-1", digest=DIGEST_C)


# --- Signed-request helpers ----------------------------------------------------


def _signed_headers(
    method,
    path,
    body,
    *,
    actor="org-1",
    seed=SEED_A,
    timestamp=None,
):
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    message = access_message_bytes(
        method, path, ts, hashlib.sha256(body).hexdigest()
    )
    signature = base64.b64encode(ed25519_sign(seed, message)).decode("ascii")
    return {"X-PA": actor, "X-PT": ts, "X-PS": signature}


def _create_policy(client, actor, seed, threshold):
    body = json.dumps({"actor_id": actor, "threshold": threshold}).encode()
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", POLICIES_PATH, body, actor=actor, seed=seed),
    }
    resp = client.post(POLICIES_PATH, content=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _revoke_policy(client, policy_id, actor, seed, reason="no longer used"):
    body = json.dumps({"policy_id": policy_id, "reason": reason}).encode()
    headers = {
        "Content-Type": "application/json",
        **_signed_headers("POST", REVOCATIONS_PATH, body, actor=actor, seed=seed),
    }
    resp = client.post(REVOCATIONS_PATH, content=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _item(target_type, target_id):
    return {"target_type": target_type, "target_id": target_id}


def _post_batch(client, items, *, actor="org-1", seed=SEED_A, headers=None):
    body = json.dumps({"items": items}).encode()
    if headers is None:
        headers = _signed_headers("POST", BATCH_PATH, body, actor=actor, seed=seed)
    return client.post(
        BATCH_PATH,
        content=body,
        headers={"Content-Type": "application/json", **headers},
    )


def _post_raw(client, body, *, actor="org-1", seed=SEED_A, url=BATCH_PATH):
    headers = _signed_headers("POST", BATCH_PATH, body, actor=actor, seed=seed)
    return client.post(
        url,
        content=body,
        headers={"Content-Type": "application/json", **headers},
    )


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


# --- No policy: policy_missing without any target lookup ------------------------


def test_no_policy_is_policy_missing_per_item_and_never_looks_up_targets(client):
    _world(client)
    # Neither target exists: without a policy there is no lookup, so the
    # response is 200 policy_missing for every item rather than any 404.
    resp = _post_batch(
        client,
        [
            _item("claim", "clm_ghost"),
            _item("evidence_bundle", "evb_ghost"),
            _item("claim", "clm_ghost"),
        ],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 3
    assert body["items"] == [
        {
            "target_type": "claim",
            "target_id": "clm_ghost",
            "policy_id": None,
            "threshold": None,
            "qualified_signer_count": 0,
            "decision": "untrusted",
            "reason": "policy_missing",
        },
        {
            "target_type": "evidence_bundle",
            "target_id": "evb_ghost",
            "policy_id": None,
            "threshold": None,
            "qualified_signer_count": 0,
            "decision": "untrusted",
            "reason": "policy_missing",
        },
        {
            "target_type": "claim",
            "target_id": "clm_ghost",
            "policy_id": None,
            "threshold": None,
            "qualified_signer_count": 0,
            "decision": "untrusted",
            "reason": "policy_missing",
        },
    ]


def test_no_policy_reports_zero_signers_even_when_proofs_exist(client):
    _world(client)
    target = _target_claim(client)
    _attest(client, "claim", target["id"], "org-2", SEED_B)
    _attest(client, "claim", target["id"], "org-3", SEED_C)
    resp = _post_batch(client, [_item("claim", target["id"])])
    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["reason"] == "policy_missing"
    assert item["qualified_signer_count"] == 0
    assert item["decision"] == "untrusted"


def test_revoked_policy_renders_exactly_like_no_policy(client):
    _world(client)
    policy = _create_policy(client, "org-1", SEED_A, 1)
    target = _target_claim(client)
    _attest(client, "claim", target["id"], "org-2", SEED_B)
    _revoke_policy(client, policy["id"], "org-1", SEED_A)

    resp = _post_batch(
        client,
        [_item("claim", target["id"]), _item("claim", "clm_ghost")],
    )
    assert resp.status_code == 200
    for item in resp.json()["items"]:
        assert item["policy_id"] is None
        assert item["threshold"] is None
        assert item["qualified_signer_count"] == 0
        assert item["decision"] == "untrusted"
        assert item["reason"] == "policy_missing"


# --- Decisions under a policy ----------------------------------------------------


def test_batch_mixed_targets_preserve_order_duplicates_and_count(client):
    _world(client)
    policy = _create_policy(client, "org-1", SEED_A, 2)
    target = _target_claim(client)
    bundle = _make_bundle(client, target["id"])
    _attest(client, "claim", target["id"], "org-2", SEED_B)
    _attest(client, "claim", target["id"], "org-3", SEED_C)
    _attest(client, "evidence_bundle", bundle["id"], "org-2", SEED_B)

    resp = _post_batch(
        client,
        [
            _item("claim", target["id"]),
            _item("evidence_bundle", bundle["id"]),
            _item("claim", target["id"]),
        ],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 3
    assert body["items"] == [
        {
            "target_type": "claim",
            "target_id": target["id"],
            "policy_id": policy["id"],
            "threshold": 2,
            "qualified_signer_count": 2,
            "decision": "trusted",
            "reason": "threshold_met",
        },
        {
            "target_type": "evidence_bundle",
            "target_id": bundle["id"],
            "policy_id": policy["id"],
            "threshold": 2,
            "qualified_signer_count": 1,
            "decision": "untrusted",
            "reason": "below_threshold",
        },
        {
            "target_type": "claim",
            "target_id": target["id"],
            "policy_id": policy["id"],
            "threshold": 2,
            "qualified_signer_count": 2,
            "decision": "trusted",
            "reason": "threshold_met",
        },
    ]


def test_batch_threshold_boundary_equality_is_trusted(client):
    _world(client)
    _create_policy(client, "org-1", SEED_A, 2)
    target = _target_claim(client)
    _attest(client, "claim", target["id"], "org-2", SEED_B)
    _attest(client, "claim", target["id"], "org-3", SEED_C)

    resp = _post_batch(client, [_item("claim", target["id"])])
    item = resp.json()["items"][0]
    assert item["qualified_signer_count"] == 2
    assert item["decision"] == "trusted"
    assert item["reason"] == "threshold_met"


def test_batch_same_signer_under_two_keys_counts_once(client):
    _world(client)
    _create_policy(client, "org-1", SEED_A, 2)
    target = _target_claim(client)
    # org-2 attests the target under two different keys: one qualified signer.
    _attest(client, "claim", target["id"], "org-2", SEED_B)
    _attest(client, "claim", target["id"], "org-2", SEED_D)

    resp = _post_batch(client, [_item("claim", target["id"])])
    item = resp.json()["items"][0]
    assert item["qualified_signer_count"] == 1
    assert item["decision"] == "untrusted"
    assert item["reason"] == "below_threshold"


def test_batch_revoked_attestation_no_longer_counts(client):
    _world(client)
    _create_policy(client, "org-1", SEED_A, 2)
    target = _target_claim(client)
    att_b = _attest(client, "claim", target["id"], "org-2", SEED_B)
    _attest(client, "claim", target["id"], "org-3", SEED_C)
    revoked = client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": att_b["id"],
            "revoker_actor_id": "org-2",
            "reason": "key rotation",
        },
    )
    assert revoked.status_code == 201

    resp = _post_batch(client, [_item("claim", target["id"])])
    item = resp.json()["items"][0]
    assert item["qualified_signer_count"] == 1
    assert item["decision"] == "untrusted"
    assert item["reason"] == "below_threshold"


def test_batch_uses_only_the_callers_own_policy(client):
    _world(client)
    # org-1 requires two signers; org-2 requires one. org-2 attests its own
    # claim so it holds a current key; org-3 alone attests the target.
    _create_policy(client, "org-1", SEED_A, 2)
    _attest(client, "claim", _make_claim(client, "org-2", digest=DIGEST_B)["id"],
            "org-2", SEED_B)
    _create_policy(client, "org-2", SEED_B, 1)
    target = _target_claim(client)
    _attest(client, "claim", target["id"], "org-3", SEED_C)

    as_org1 = _post_batch(client, [_item("claim", target["id"])])
    assert as_org1.json()["items"][0]["decision"] == "untrusted"
    assert as_org1.json()["items"][0]["reason"] == "below_threshold"

    as_org2 = _post_batch(
        client, [_item("claim", target["id"])], actor="org-2", seed=SEED_B
    )
    assert as_org2.json()["items"][0]["decision"] == "trusted"
    assert as_org2.json()["items"][0]["reason"] == "threshold_met"
    assert (
        as_org1.json()["items"][0]["policy_id"]
        != as_org2.json()["items"][0]["policy_id"]
    )


def test_batch_of_100_items_is_accepted(client):
    _world(client)
    _create_policy(client, "org-1", SEED_A, 1)
    target = _target_claim(client)
    resp = _post_batch(client, [_item("claim", target["id"])] * 100)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 100
    assert len(body["items"]) == 100


# --- Missing targets once a policy exists ----------------------------------------


def test_unknown_claim_is_single_404_with_first_target_details(client):
    _world(client)
    _create_policy(client, "org-1", SEED_A, 1)
    target = _target_claim(client)
    resp = _post_batch(
        client,
        [_item("claim", target["id"]), _item("claim", "clm_ghost")],
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "claim_not_found"
    assert error["details"] == {"target_type": "claim", "target_id": "clm_ghost"}
    # A failed batch never returns a partial items array.
    assert "items" not in resp.json()


def test_unknown_evidence_bundle_is_404_with_policy(client):
    _world(client)
    _create_policy(client, "org-1", SEED_A, 1)
    resp = _post_batch(client, [_item("evidence_bundle", "evb_ghost")])
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "evidence_bundle_not_found"
    assert error["details"] == {
        "target_type": "evidence_bundle",
        "target_id": "evb_ghost",
    }


def test_batch_reports_the_first_missing_target_in_request_order(client):
    _world(client)
    _create_policy(client, "org-1", SEED_A, 1)
    target = _target_claim(client)

    resp = _post_batch(
        client,
        [
            _item("claim", target["id"]),
            _item("claim", "clm_first_ghost"),
            _item("evidence_bundle", "evb_second_ghost"),
        ],
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "claim_not_found"
    assert resp.json()["error"]["details"]["target_id"] == "clm_first_ghost"

    # When the first missing target is a bundle, its type drives the code.
    resp = _post_batch(
        client,
        [
            _item("evidence_bundle", "evb_first_ghost"),
            _item("claim", "clm_second_ghost"),
        ],
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "evidence_bundle_not_found"
    assert resp.json()["error"]["details"]["target_id"] == "evb_first_ghost"


def test_batch_target_types_never_cross_match(client):
    _world(client)
    _create_policy(client, "org-1", SEED_A, 1)
    claim = _target_claim(client)
    bundle = _make_bundle(client, claim["id"])

    # An existing bundle id declared as a claim, and vice versa, is a 404.
    resp = _post_batch(client, [_item("claim", bundle["id"])])
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "claim_not_found"
    assert resp.json()["error"]["details"] == {
        "target_type": "claim",
        "target_id": bundle["id"],
    }
    resp = _post_batch(client, [_item("evidence_bundle", claim["id"])])
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "evidence_bundle_not_found"


def test_batch_verbatim_target_id_with_surrounding_whitespace_is_literal(client):
    _world(client)
    _create_policy(client, "org-1", SEED_A, 1)
    target = _target_claim(client)
    # Whitespace on a non-blank id is part of the identifier: it is not
    # trimmed, so no claim matches and the batch is a type-matched 404.
    resp = _post_batch(
        client,
        [_item("claim", target["id"]), _item("claim", f"  {target['id']}  ")],
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "claim_not_found"
    assert error["details"] == {
        "target_type": "claim",
        "target_id": f"  {target['id']}  ",
    }


# --- Credential boundary ------------------------------------------------------------


def test_missing_or_unverifiable_credentials_are_the_opaque_404(client):
    _world(client)
    _create_policy(client, "org-1", SEED_A, 1)
    target = _target_claim(client)
    body = json.dumps({"items": [_item("claim", target["id"])]}).encode()

    # No headers at all.
    assert client.post(
        BATCH_PATH, content=body, headers={"Content-Type": "application/json"}
    ).status_code == 404
    # A well-formed signature no current key of the claimed actor verifies.
    assert _post_batch(
        client, [_item("claim", target["id"])], actor="org-1", seed=SEED_B
    ).status_code == 404
    # A ghost actor id with a well-formed signature.
    resp = _post_batch(
        client, [_item("claim", target["id"])], actor="ghost", seed=SEED_A
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_malformed_credentials_are_422(client):
    _world(client)
    target = _target_claim(client)
    items = [_item("claim", target["id"])]
    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    for ts in (stale, "not-a-timestamp", "2026-09-20T12:00:00+01:00"):
        body = json.dumps({"items": items}).encode()
        headers = _signed_headers("POST", BATCH_PATH, body, timestamp=ts)
        resp = _post_batch(client, items, headers=headers)
        assert resp.status_code == 422, ts
        assert resp.json()["error"]["code"] == "validation_error"

    for raw in ("@@@", "aGVsbG8", base64.b64encode(b"x" * 63).decode()):
        body = json.dumps({"items": items}).encode()
        headers = _signed_headers("POST", BATCH_PATH, body)
        headers["X-PS"] = raw
        resp = _post_batch(client, items, headers=headers)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["code"] == "validation_error"


# --- Body and parameter validation ----------------------------------------------------


def _assert_422(resp):
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"
    return resp.json()["error"]["details"]["issues"]


def test_batch_requires_valid_json_object(client):
    _world(client)
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
        resp = _post_raw(client, raw)
        issues = _assert_422(resp)
        assert issues, raw


def test_batch_top_level_structure_is_validated(client):
    _world(client)
    target = _target_claim(client)

    # Missing items / items not an array / empty array.
    for payload in ({}, {"items": None}, {"items": {}}, {"items": []}):
        body = json.dumps(payload).encode()
        _assert_422(_post_raw(client, body))

    # An undeclared top-level field is rejected rather than discarded.
    body = json.dumps(
        {"items": [_item("claim", target["id"])], "extra": 1}
    ).encode()
    issues = _assert_422(_post_raw(client, body))
    assert any(issue["loc"][-1] == "extra" for issue in issues)

    # items as a JSON object is not an array.
    body = json.dumps({"items": {"0": _item("claim", target["id"])}}).encode()
    _assert_422(_post_raw(client, body))


def test_batch_cardinality_above_100_is_422(client):
    _world(client)
    target = _target_claim(client)
    _assert_422(_post_batch(client, [_item("claim", target["id"])] * 101))


def test_batch_items_must_be_objects_with_exactly_two_fields(client):
    _world(client)
    target = _target_claim(client)

    for non_object in ("claim", 42, True, None, ["claim", target["id"]]):
        _assert_422(_post_batch(client, [non_object]))

    valid = _item("claim", target["id"])

    # Missing required fields.
    _assert_422(_post_batch(client, [{"target_id": target["id"]}]))
    _assert_422(_post_batch(client, [{"target_type": "claim"}]))

    # Undeclared item fields, including the evaluation batch's min_signers.
    issues = _assert_422(_post_batch(client, [{**valid, "bogus": 1}]))
    assert any(issue["loc"][-1] == "bogus" for issue in issues)
    issues = _assert_422(_post_batch(client, [{**valid, "min_signers": 1}]))
    assert any(issue["loc"][-1] == "min_signers" for issue in issues)


def test_batch_target_type_must_be_one_of_the_two_literals(client):
    _world(client)
    target = _target_claim(client)
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
        _assert_422(
            _post_batch(
                client, [{"target_type": bad, "target_id": target["id"]}]
            )
        )


def test_batch_target_id_must_be_a_nonblank_string(client):
    _world(client)
    for bad in ("", "   ", "\t", 42, True, None, ["clm_x"], {"id": "x"}):
        _assert_422(
            _post_batch(client, [{"target_type": "claim", "target_id": bad}])
        )


def test_batch_validation_precedes_authentication_and_existence(client):
    _world(client)
    # Every malformed batch is a 422 even with no credentials at all and
    # even when a referenced target does not exist: validation completes
    # before authentication and before any lookup.
    cases = [
        {"items": []},
        {"items": [_item("claim", "clm_ghost")], "extra": True},
        {"items": [{"target_type": "nope", "target_id": "clm_ghost"}]},
        {"items": [{"target_type": "claim", "target_id": "   "}]},
        {"items": [_item("claim", "clm_ghost"), "not-an-object"]},
        {"items": [_item("claim", "clm_ghost")] * 101},
    ]
    for payload in cases:
        resp = client.post(BATCH_PATH, json=payload)
        _assert_422(resp)


def test_batch_any_query_parameter_is_422(client):
    _world(client)
    target = _target_claim(client)
    for url in (
        f"{BATCH_PATH}?foo=bar",
        f"{BATCH_PATH}?target_type=claim",
        f"{BATCH_PATH}?target_id={target['id']}",
    ):
        resp = _post_raw(
            client,
            json.dumps({"items": [_item("claim", target["id"])]}).encode(),
            url=url,
        )
        _assert_422(resp)


# --- Wire format and read-only guarantee -----------------------------------------------


def test_batch_success_body_is_compact_json_with_one_newline(client):
    _world(client)
    _create_policy(client, "org-1", SEED_A, 1)
    target = _target_claim(client)
    bundle = _make_bundle(client, target["id"])
    resp = _post_batch(
        client,
        [_item("claim", target["id"]), _item("evidence_bundle", bundle["id"])],
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
    assert isinstance(decoded["count"], int)
    for entry in decoded["items"]:
        assert set(entry) == {
            "target_type",
            "target_id",
            "policy_id",
            "threshold",
            "qualified_signer_count",
            "decision",
            "reason",
        }
        assert isinstance(entry["threshold"], int)
        assert isinstance(entry["qualified_signer_count"], int)
    expected = (
        json.dumps(decoded, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    assert raw == expected


def test_batch_response_never_carries_payloads_or_bytes(client):
    _world(client)
    _create_policy(client, "org-1", SEED_A, 1)
    target = _target_claim(client)
    bundle = _make_bundle(client, target["id"])
    _attest(client, "claim", target["id"], "org-2", SEED_B)
    _attest(client, "evidence_bundle", bundle["id"], "org-2", SEED_B)

    resp = _post_batch(
        client,
        [_item("claim", target["id"]), _item("evidence_bundle", bundle["id"])],
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


def test_batches_write_no_resources_or_audit(client, db_session):
    _world(client)
    _create_policy(client, "org-1", SEED_A, 1)
    target = _target_claim(client)
    _attest(client, "claim", target["id"], "org-2", SEED_B)
    # org-3 holds a current key (its own attestation) but no policy.
    _attest(client, "claim", _make_claim(client, "org-3", digest=DIGEST_B)["id"],
            "org-3", SEED_C)

    events_before = _audit_count(db_session)
    policies_before = len(
        db_session.execute(select(ActorTrustPolicy)).scalars().all()
    )
    attestations_before = len(
        db_session.execute(select(Attestation)).scalars().all()
    )

    responses = [
        _post_batch(client, [_item("claim", target["id"])]),  # trusted
        _post_batch(client, [_item("claim", "clm_ghost")]),  # 404
        _post_batch(client, [_item("evidence_bundle", "evb_ghost")]),  # 404
        _post_batch(client, []),  # 422 empty batch
        # A no-policy caller batch (org-3 has a key but no policy).
        _post_batch(
            client, [_item("claim", target["id"])], actor="org-3", seed=SEED_C
        ),
    ]
    assert [r.status_code for r in responses] == [200, 404, 404, 422, 200]

    # Repeated calls are byte-for-byte identical for the same state.
    again = _post_batch(client, [_item("claim", target["id"])])
    assert again.content == responses[0].content

    db_session.expire_all()
    assert _audit_count(db_session) == events_before
    assert len(
        db_session.execute(select(ActorTrustPolicy)).scalars().all()
    ) == policies_before
    assert len(
        db_session.execute(select(Attestation)).scalars().all()
    ) == attestations_before
