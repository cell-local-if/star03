"""Tests for verifiable attestations.

Covers POST/GET /v1/attestations and the filtered list, including:
verification against the exact canonical message, claim and evidence-bundle
targets, idempotent dedup by (target, signer, key, signature digest), the
404/422 error boundary, single-transaction audit commit, non-persistence of
the raw signature, stable ordering and filtering, a concurrent-identical
race, and persistence across restarts. All tests are deterministic and
offline (signatures are produced by the stdlib test signer).
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
from datetime import datetime

from sqlalchemy import select

from provenance.models import (
    EVENT_ATTESTATION_CREATED,
    Attestation,
    AuditEvent,
)
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

EVIDENCE_DIGEST = hashlib.sha256(b"evidence-att").hexdigest()


# --- Setup helpers ----------------------------------------------------------


def _create_content(client, actor_id="org-1", digest=DIGEST_A):
    resp = client.post(
        "/v1/contents", json=content_payload(actor_id=actor_id, digest=digest)
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
            "payload": {"statement": "attested"},
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


def _canonical_message(target_type, target_id, signer_actor_id):
    """Independently reproduce the specified canonical signing bytes."""
    return json.dumps(
        ["provenance-attestation-v1", target_type, target_id, signer_actor_id],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _public_b64(seed=SEED_A) -> str:
    return base64.b64encode(ed25519_public_key(seed)).decode("ascii")


def _signature_b64(seed, target_type, target_id, signer_actor_id, *,
                   message=None) -> str:
    msg = message if message is not None else attestation_message_bytes(
        target_type, target_id, signer_actor_id
    )
    return base64.b64encode(ed25519_sign(seed, msg)).decode("ascii")


def _attestation_payload(
    target_type,
    target_id,
    *,
    seed=SEED_A,
    signer_actor_id="org-1",
    signature_message=None,
    raw_signature=None,
):
    if raw_signature is not None:
        signature = base64.b64encode(raw_signature).decode("ascii")
    else:
        signature = _signature_b64(
            seed,
            target_type,
            target_id,
            signer_actor_id,
            message=signature_message,
        )
    return {
        "target_type": target_type,
        "target_id": target_id,
        "signer_actor_id": signer_actor_id,
        "public_key": _public_b64(seed),
        "signature": signature,
    }


def _create_attestation(client, *args, **kwargs):
    resp = client.post("/v1/attestations", json=_attestation_payload(*args, **kwargs))
    assert resp.status_code == 201, resp.text
    return resp.json()


# --- Canonical message contract ---------------------------------------------


def test_canonical_message_format_is_compact_utf8_unescaped():
    msg = _canonical_message("claim", "clm_x", "org-1")
    assert msg == b'["provenance-attestation-v1","claim","clm_x","org-1"]'
    unicode_msg = _canonical_message("evidence_bundle", "evb_x", "org-证据")
    # Non-ASCII is emitted as raw UTF-8, never \\u escapes.
    assert unicode_msg == (
        '["provenance-attestation-v1","evidence_bundle","evb_x","org-证据"]'
    ).encode("utf-8")
    assert b"\\u" not in unicode_msg
    assert attestation_message_bytes(
        "evidence_bundle", "evb_x", "org-证据"
    ) == unicode_msg


# --- Normal creation ---------------------------------------------------------


def test_create_claim_attestation_returns_full_public_fields(client):
    claim = _setup_claim(client)
    raw_sig = ed25519_sign(
        SEED_A, attestation_message_bytes("claim", claim["id"], "org-1")
    )
    body = _create_attestation(client, "claim", claim["id"])

    assert set(body) == {
        "id",
        "target_type",
        "target_id",
        "signer_actor_id",
        "public_key",
        "signature_digest_algorithm",
        "signature_digest_hex",
        "verified",
        "created_at",
    }
    assert body["id"].startswith("att_")
    assert len(body["id"]) == len("att_") + 64
    assert body["target_type"] == "claim"
    assert body["target_id"] == claim["id"]
    assert body["signer_actor_id"] == "org-1"
    assert base64.b64decode(body["public_key"]) == ed25519_public_key(SEED_A)
    assert body["signature_digest_algorithm"] == "sha256"
    # Digest is of the raw 64 signature bytes, not the Base64 string.
    assert body["signature_digest_hex"] == hashlib.sha256(raw_sig).hexdigest()
    assert body["verified"] is True
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))
    # The raw signature is never echoed in any field or header.
    assert "signature" not in body
    assert base64.b64encode(raw_sig).decode() not in json.dumps(body)


def test_create_evidence_bundle_attestation(client):
    _claim, bundle = _setup_claim_and_bundle(client)
    body = _create_attestation(client, "evidence_bundle", bundle["id"])
    assert body["target_type"] == "evidence_bundle"
    assert body["target_id"] == bundle["id"]


def test_attestation_id_is_stable_and_deterministic(client):
    claim = _setup_claim(client)
    first = _create_attestation(client, "claim", claim["id"])
    expected_material = json.dumps(
        [
            "attestation",
            "claim",
            claim["id"],
            "org-1",
            ed25519_public_key(SEED_A).hex(),
            hashlib.sha256(
                ed25519_sign(
                    SEED_A,
                    attestation_message_bytes("claim", claim["id"], "org-1"),
                )
            ).hexdigest(),
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert first["id"] == "att_" + hashlib.sha256(expected_material).hexdigest()


def test_get_attestation_returns_the_created_record(client):
    claim = _setup_claim(client)
    created = _create_attestation(client, "claim", claim["id"])
    resp = client.get(f"/v1/attestations/{created['id']}")
    assert resp.status_code == 200
    assert resp.json() == created


# --- Idempotency and identity ------------------------------------------------


def test_repeat_submission_is_idempotent_200_and_adds_no_audit(
    client, db_session
):
    claim = _setup_claim(client)
    first = _create_attestation(client, "claim", claim["id"])
    events_after_create = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    for _ in range(3):
        repeat = client.post(
            "/v1/attestations",
            json=_attestation_payload("claim", claim["id"]),
        )
        assert repeat.status_code == 200
        assert repeat.json() == first

    rows = db_session.execute(select(Attestation)).scalars().all()
    assert [r.id for r in rows] == [first["id"]]
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_ATTESTATION_CREATED
        )
    ).scalars().all()
    assert len(events) == 1
    assert events[0].resource_id == first["id"]
    # No audit event from the repeat submissions.
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_after_create
    )


def test_distinct_keys_and_digests_form_independent_attestations(client):
    claim = _setup_claim(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")

    by_key_a = _create_attestation(client, "claim", claim["id"], seed=SEED_A)
    # Same target and actor, different key/signature: independent record.
    by_key_b = _create_attestation(client, "claim", claim["id"], seed=SEED_B)
    # Same key, different actor: independent record.
    by_actor = _create_attestation(
        client, "claim", claim["id"], seed=SEED_A, signer_actor_id="org-2"
    )

    ids = {by_key_a["id"], by_key_b["id"], by_actor["id"]}
    assert len(ids) == 3
    assert by_key_b["public_key"] != by_key_a["public_key"]
    assert by_actor["signer_actor_id"] == "org-2"


def test_same_identity_with_a_tampered_signature_still_fails_422(client):
    claim = _setup_claim(client)
    first = _create_attestation(client, "claim", claim["id"])

    good = ed25519_sign(
        SEED_A, attestation_message_bytes("claim", claim["id"], "org-1")
    )
    tampered = good[:-1] + bytes([good[-1] ^ 1])
    resp = client.post(
        "/v1/attestations",
        json=_attestation_payload(
            "claim", claim["id"], raw_signature=tampered
        ),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "attestation_verification_failed"
    # The existing attestation is untouched.
    assert client.get(f"/v1/attestations/{first['id']}").json() == first


# --- Signature verification boundary -----------------------------------------


def test_tampered_signature_is_422_and_writes_nothing(client, db_session):
    claim = _setup_claim(client)
    events_before = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )
    good = ed25519_sign(
        SEED_A, attestation_message_bytes("claim", claim["id"], "org-1")
    )
    bad = bytearray(good)
    bad[0] ^= 0x01
    resp = client.post(
        "/v1/attestations",
        json=_attestation_payload("claim", claim["id"], raw_signature=bytes(bad)),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "attestation_verification_failed"
    assert db_session.execute(select(Attestation)).scalars().all() == []
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )


def test_signature_must_be_over_the_exact_canonical_message(client):
    claim = _setup_claim(client)
    target = ("claim", claim["id"], "org-1")

    wrong_actor = attestation_message_bytes("claim", claim["id"], "org-2")
    reordered = json.dumps(
        ["provenance-attestation-v1", claim["id"], "claim", "org-1"],
        separators=(",", ":"),
    ).encode("utf-8")
    pretty = json.dumps(
        ["provenance-attestation-v1", *target],
        separators=(", ", ": "),
    ).encode("utf-8")
    wrong_prefix = json.dumps(
        ["provenance-attestation-v0", *target],
        separators=(",", ":"),
    ).encode("utf-8")

    for wrong in (wrong_actor, reordered, pretty, wrong_prefix):
        resp = client.post(
            "/v1/attestations",
            json=_attestation_payload(
                "claim", claim["id"], signature_message=wrong
            ),
        )
        assert resp.status_code == 422, wrong
        assert resp.json()["error"]["code"] == "attestation_verification_failed"


def test_unicode_signer_id_verifies_against_unescaped_utf8_message(client):
    unicode_actor = "org-证据"
    create_actor(client, actor_id=unicode_actor, name="Unicode Org",
                 type="organization")
    content = _create_content(client, actor_id=unicode_actor)
    claim = _create_claim(client, content["id"], actor_id=unicode_actor)

    # Signing the compact, non-ASCII-escaped UTF-8 message succeeds.
    body = _create_attestation(
        client, "claim", claim["id"], signer_actor_id=unicode_actor
    )
    assert body["signer_actor_id"] == unicode_actor

    # The same signature produced over an ASCII-escaped serialization fails.
    escaped = json.dumps(
        ["provenance-attestation-v1", "claim", claim["id"], unicode_actor],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    resp = client.post(
        "/v1/attestations",
        json=_attestation_payload(
            "claim",
            claim["id"],
            signer_actor_id=unicode_actor,
            signature_message=escaped,
        ),
    )
    assert resp.status_code == 422


def test_public_key_not_matching_signature_fails(client):
    claim = _setup_claim(client)
    payload = _attestation_payload("claim", claim["id"], seed=SEED_A)
    payload["public_key"] = _public_b64(SEED_B)  # signature stays from SEED_A
    resp = client.post("/v1/attestations", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "attestation_verification_failed"


# --- Missing-resource boundary (distinct from validation) --------------------


def test_get_unknown_attestation_is_404(client):
    resp = client.get("/v1/attestations/att_does_not_exist")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "attestation_not_found"
    assert error["details"]["attestation_id"] == "att_does_not_exist"


def test_create_attestation_unknown_claim_target_is_404(client):
    create_actor(client)
    resp = client.post(
        "/v1/attestations",
        json=_attestation_payload("claim", "clm_ghost"),
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "claim_not_found"
    assert error["details"]["claim_id"] == "clm_ghost"


def test_create_attestation_unknown_bundle_target_is_404(client):
    create_actor(client)
    resp = client.post(
        "/v1/attestations",
        json=_attestation_payload("evidence_bundle", "evb_ghost"),
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "evidence_bundle_not_found"
    assert error["details"]["evidence_bundle_id"] == "evb_ghost"


def test_wrong_target_type_for_existing_resource_is_404(client):
    claim, bundle = _setup_claim_and_bundle(client)
    # A bundle id declared as a claim, and a claim id declared as a bundle.
    r1 = client.post(
        "/v1/attestations",
        json=_attestation_payload("claim", bundle["id"]),
    )
    r2 = client.post(
        "/v1/attestations",
        json=_attestation_payload("evidence_bundle", claim["id"]),
    )
    assert r1.status_code == 404
    assert r1.json()["error"]["code"] == "claim_not_found"
    assert r2.status_code == 404
    assert r2.json()["error"]["code"] == "evidence_bundle_not_found"


def test_unknown_signer_actor_is_404_even_with_valid_signature(client):
    claim = _setup_claim(client)
    # Signature itself is cryptographically valid; only the actor is missing.
    resp = client.post(
        "/v1/attestations",
        json=_attestation_payload("claim", claim["id"], signer_actor_id="ghost"),
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "unknown_actor"


# --- Validation boundary ------------------------------------------------------


def test_reject_unknown_target_type(client):
    create_actor(client)
    resp = client.post(
        "/v1/attestations",
        json=_attestation_payload("content", "cnt_x")
        | {"target_type": "content"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_reject_blank_target_and_actor_ids(client):
    create_actor(client)
    for field in ("target_id", "signer_actor_id"):
        payload = _attestation_payload("claim", "clm_x")
        payload[field] = "   "
        resp = client.post("/v1/attestations", json=payload)
        assert resp.status_code == 422, field
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_public_keys_and_signatures_of_wrong_length(client):
    create_actor(client)
    base = _attestation_payload("claim", "clm_ghost")
    bad_values = [
        ("public_key", base64.b64encode(b"k" * 31).decode()),
        ("public_key", base64.b64encode(b"k" * 33).decode()),
        ("public_key", ""),
        ("signature", base64.b64encode(b"s" * 63).decode()),
        ("signature", base64.b64encode(b"s" * 65).decode()),
    ]
    for field, value in bad_values:
        payload = dict(base)
        payload[field] = value
        resp = client.post("/v1/attestations", json=payload)
        assert resp.status_code == 422, (field, value)
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_non_base64_key_and_signature(client):
    create_actor(client)
    base = _attestation_payload("claim", "clm_ghost")
    bad_values = [
        ("public_key", "@@@ not base64"),
        ("public_key", "aGVsbG8"),  # missing padding
        ("public_key", _public_b64(SEED_A)[:-1] + "$"),
        # URL-safe alphabet uses '-'/'_', which standard Base64 rejects.
        ("public_key", base64.urlsafe_b64encode(b"\xff" * 32).decode()),
        ("signature", 12345),
        ("signature", None),
    ]
    for field, value in bad_values:
        payload = dict(base)
        payload[field] = value
        resp = client.post("/v1/attestations", json=payload)
        assert resp.status_code == 422, (field, value)
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_missing_required_fields(client):
    resp = client.post("/v1/attestations", json={})
    assert resp.status_code == 422
    issue_fields = {
        ".".join(part for part in issue["loc"] if part != "body")
        for issue in resp.json()["error"]["details"]["issues"]
    }
    assert {
        "target_type",
        "target_id",
        "signer_actor_id",
        "public_key",
        "signature",
    }.issubset(issue_fields)


def test_reject_undeclared_fields(client):
    claim = _setup_claim(client)
    payload = _attestation_payload("claim", claim["id"])
    payload["unexpected"] = "value"
    resp = client.post("/v1/attestations", json=payload)
    assert resp.status_code == 422
    issues = resp.json()["error"]["details"]["issues"]
    assert any("unexpected" in issue["loc"] for issue in issues)


def test_reject_malformed_json_body(client):
    create_actor(client)
    resp = client.post(
        "/v1/attestations",
        content="{not valid json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- List filtering and stable ordering --------------------------------------


def test_list_returns_all_in_stable_creation_order(client):
    claim_one = _setup_claim(client)
    create_actor(client, actor_id="org-2", name="Other", type="organization")
    content_two = _create_content(client, actor_id="org-2", digest=DIGEST_B)
    claim_two = _create_claim(client, content_two["id"], actor_id="org-2")
    bundle = _create_bundle(client, claim_one["id"])

    first = _create_attestation(client, "claim", claim_one["id"])
    second = _create_attestation(client, "claim", claim_two["id"],
                                 seed=SEED_B, signer_actor_id="org-2")
    third = _create_attestation(client, "evidence_bundle", bundle["id"])

    resp = client.get("/v1/attestations")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    assert [item["id"] for item in body["items"]] == [
        first["id"],
        second["id"],
        third["id"],
    ]


def test_list_filters_by_target_type(client):
    claim, bundle = _setup_claim_and_bundle(client)
    claim_att = _create_attestation(client, "claim", claim["id"])
    _create_attestation(client, "evidence_bundle", bundle["id"], seed=SEED_B)

    claims = client.get(
        "/v1/attestations", params={"target_type": "claim"}
    ).json()
    assert claims["count"] == 1
    assert claims["items"][0]["id"] == claim_att["id"]

    bundles = client.get(
        "/v1/attestations", params={"target_type": "evidence_bundle"}
    ).json()
    assert bundles["count"] == 1
    assert bundles["items"][0]["target_id"] == bundle["id"]


def test_list_filters_by_target_id(client):
    claim_one = _setup_claim(client)
    content_two = _create_content(client, digest=DIGEST_B)
    claim_two = _create_claim(client, content_two["id"])
    a1 = _create_attestation(client, "claim", claim_one["id"])
    _create_attestation(client, "claim", claim_two["id"], seed=SEED_B)

    body = client.get(
        "/v1/attestations", params={"target_id": claim_one["id"]}
    ).json()
    assert body["count"] == 1
    assert body["items"][0]["id"] == a1["id"]


def test_list_combined_filters_and_empty_results_are_not_404(client):
    claim = _setup_claim(client)
    _create_attestation(client, "claim", claim["id"])

    empty = client.get(
        "/v1/attestations",
        params={"target_type": "claim", "target_id": "clm_unknown"},
    )
    assert empty.status_code == 200
    assert empty.json() == {"items": [], "count": 0}

    no_bundles = client.get(
        "/v1/attestations", params={"target_type": "evidence_bundle"}
    )
    assert no_bundles.status_code == 200
    assert no_bundles.json() == {"items": [], "count": 0}


def test_list_with_bad_target_type_filter_is_422(client):
    resp = client.get("/v1/attestations", params={"target_type": "bogus"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Transactional guarantees -------------------------------------------------


def test_attestation_and_audit_event_commit_atomically(client, db_session):
    claim = _setup_claim(client)
    created = _create_attestation(client, "claim", claim["id"])

    rows = db_session.execute(select(Attestation)).scalars().all()
    events = db_session.execute(
        select(AuditEvent)
        .where(AuditEvent.event_type == EVENT_ATTESTATION_CREATED)
        .order_by(AuditEvent.seq.asc())
    ).scalars().all()
    assert [r.id for r in rows] == [created["id"]]
    assert [e.resource_id for e in events] == [created["id"]]
    created_at = events[0].created_at
    assert created_at.tzinfo.utcoffset(created_at).total_seconds() == 0


def test_failed_creations_write_no_rows_or_audit(client, db_session):
    create_actor(client)
    content = _create_content(client)
    good_target_claim = _create_claim(client, content["id"])
    events_before = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    valid_sig = ed25519_sign(
        SEED_A, attestation_message_bytes("claim", good_target_claim["id"], "org-1")
    )
    attempts = [
        # Missing target.
        client.post(
            "/v1/attestations",
            json=_attestation_payload("claim", "clm_ghost"),
        ),
        # Valid target but invalid signature.
        client.post(
            "/v1/attestations",
            json=_attestation_payload(
                "claim",
                good_target_claim["id"],
                raw_signature=valid_sig[:-1] + bytes([valid_sig[-1] ^ 1]),
            ),
        ),
        # Valid signature but missing actor.
        client.post(
            "/v1/attestations",
            json=_attestation_payload(
                "claim", good_target_claim["id"], signer_actor_id="ghost"
            ),
        ),
        # Malformed key.
        client.post(
            "/v1/attestations",
            json={**_attestation_payload("claim", "clm_ghost"), "public_key": "x"},
        ),
    ]
    assert [r.status_code for r in attempts] == [404, 422, 404, 422]
    assert db_session.execute(select(Attestation)).scalars().all() == []
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )


def test_raw_signature_is_never_persisted(client, db_session):
    claim = _setup_claim(client)
    raw_sig = ed25519_sign(
        SEED_A, attestation_message_bytes("claim", claim["id"], "org-1")
    )
    created = _create_attestation(client, "claim", claim["id"])

    columns = [c.name for c in Attestation.__table__.columns]
    assert "signature" not in columns
    assert {"public_key", "signature_digest_hex"}.issubset(columns)

    row = db_session.execute(select(Attestation)).scalars().one()
    assert row.public_key == ed25519_public_key(SEED_A)
    assert row.signature_digest_hex == hashlib.sha256(raw_sig).hexdigest()
    assert raw_sig not in bytes(row.id, "ascii")  # id derives from digest
    assert created["id"] == row.id


# --- Concurrent race boundary -------------------------------------------------


def test_concurrent_identical_requests_yield_one_attestation_and_audit(
    tmp_db_url, file_app, file_client
):
    create_actor(file_client)
    content = file_client.post(
        "/v1/contents", json=content_payload()
    ).json()
    claim = file_client.post(
        "/v1/claims",
        json={
            "content_id": content["id"],
            "actor_id": "org-1",
            "claim_type": "authorship",
            "payload": {"statement": "x"},
        },
    ).json()

    from provenance.schemas import AttestationCreate
    from provenance import service

    factory = file_app.state.session_factory
    request = AttestationCreate(
        **_attestation_payload("claim", claim["id"])
    )
    results: list[tuple[str, bool]] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        session = factory()
        try:
            barrier.wait()
            attestation, created = service.create_attestation(session, request)
            results.append((attestation.id, created))
        except Exception as exc:  # pragma: no cover - fails the test below
            errors.append(exc)
        finally:
            session.close()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(results) == 4
    assert {att_id for att_id, _ in results} == {results[0][0]}
    assert sum(1 for _, created in results if created) == 1

    audit_session = factory()
    try:
        rows = audit_session.execute(select(Attestation)).scalars().all()
        events = audit_session.execute(
            select(AuditEvent).where(
                AuditEvent.event_type == EVENT_ATTESTATION_CREATED
            )
        ).scalars().all()
        assert [r.id for r in rows] == [results[0][0]]
        assert [e.resource_id for e in events] == [results[0][0]]
    finally:
        audit_session.close()


# --- Persistence across restarts ----------------------------------------------


def test_attestations_persist_across_app_restarts(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    claim, bundle = _setup_claim_and_bundle(file_client)
    claim_att = file_client.post(
        "/v1/attestations",
        json=_attestation_payload("claim", claim["id"]),
    ).json()
    bundle_att = file_client.post(
        "/v1/attestations",
        json=_attestation_payload(
            "evidence_bundle", bundle["id"], seed=SEED_B
        ),
    ).json()

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        fetched = client.get(f"/v1/attestations/{claim_att['id']}")
        assert fetched.status_code == 200
        assert fetched.json() == claim_att

        listing = client.get("/v1/attestations").json()
        assert listing["count"] == 2
        assert [i["id"] for i in listing["items"]] == [
            claim_att["id"],
            bundle_att["id"],
        ]

        import sqlite3

        path = tmp_db_url.removeprefix("sqlite:///")
        con = sqlite3.connect(path)
        attestation_audit = con.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = ?",
            (EVENT_ATTESTATION_CREATED,),
        ).fetchone()[0]
        # The signature digest and public key survived; no signature column.
        table_cols = {
            row[1]
            for row in con.execute("PRAGMA table_info(attestations)")
        }
        con.close()
        assert attestation_audit == 2
        assert "signature" not in table_cols
