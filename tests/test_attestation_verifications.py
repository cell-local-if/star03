"""Tests for the stateless attestation verification endpoint.

Covers POST /v1/attestation-verifications: exact ``{"valid": ...}``
verdicts, consistency with stateful attestation creation, no lookup of
target/signer identifiers (unknown ids never 404), the 422 validation
boundary (missing/extra fields, bad target type, blank or overlong
identifiers, non-canonical Base64, non-object JSON), tampering, swapped
identifiers, non-ASCII identifiers, repeat stability, and the guarantee
that verification writes nothing and echoes no key or signature material.
All tests are deterministic and offline (signatures are produced by the
stdlib test signer).
"""

from __future__ import annotations

import base64
import json

from sqlalchemy import func, select

from provenance.models import Attestation, AuditEvent
from provenance.signing import attestation_message_bytes
from tests.helpers import (
    DIGEST_A,
    content_payload,
    create_actor,
    ed25519_public_key,
    ed25519_sign,
    SEED_A,
    SEED_B,
)

URL = "/v1/attestation-verifications"


# --- Helpers -----------------------------------------------------------------


def _public_b64(seed=SEED_A) -> str:
    return base64.b64encode(ed25519_public_key(seed)).decode("ascii")


def _payload(
    target_type="claim",
    target_id="clm_target-1",
    signer_actor_id="org-1",
    *,
    seed=SEED_A,
    message=None,
    raw_signature=None,
):
    if raw_signature is not None:
        signature = base64.b64encode(raw_signature).decode("ascii")
    else:
        msg = (
            message
            if message is not None
            else attestation_message_bytes(
                target_type, target_id, signer_actor_id
            )
        )
        signature = base64.b64encode(ed25519_sign(seed, msg)).decode("ascii")
    return {
        "target_type": target_type,
        "target_id": target_id,
        "signer_actor_id": signer_actor_id,
        "public_key": _public_b64(seed),
        "signature": signature,
    }


def _verify(client, payload):
    return client.post(URL, json=payload)


def _assert_valid(resp, expected):
    assert resp.status_code == 200, resp.text
    # Exactly one member, the boolean verdict, and nothing else.
    assert resp.json() == {"valid": expected}
    assert resp.text == json.dumps({"valid": expected}, separators=(",", ":")) + "\n"


def _assert_validation_error(resp):
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["message"] == "Request payload failed validation."


def _create_claim(client, actor_id="org-1"):
    create_actor(client, actor_id=actor_id)
    resp = client.post(
        "/v1/contents", json=content_payload(actor_id=actor_id, digest=DIGEST_A)
    )
    assert resp.status_code == 201, resp.text
    content = resp.json()
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content["id"],
            "actor_id": actor_id,
            "claim_type": "authorship",
            "payload": {"statement": "attested"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _db_counts(session):
    attestations = session.execute(
        select(func.count()).select_from(Attestation)
    ).scalar_one()
    audit_events = session.execute(
        select(func.count()).select_from(AuditEvent)
    ).scalar_one()
    return attestations, audit_events


# --- Verdicts -----------------------------------------------------------------


def test_valid_signature_returns_exactly_valid_true(client):
    resp = _verify(client, _payload())
    _assert_valid(resp, True)


def test_valid_signature_for_evidence_bundle_target(client):
    resp = _verify(client, _payload("evidence_bundle", "evb_target-9"))
    _assert_valid(resp, True)


def test_unknown_target_and_signer_are_not_looked_up(client):
    # Nothing exists in the service at all: no actor, claim, or bundle.
    # Stateless verification never resolves identifiers, so there is no 404.
    resp = _verify(
        client,
        _payload("claim", "clm_never_registered", "org-never-registered"),
    )
    _assert_valid(resp, True)


def test_non_ascii_identifiers_verify(client):
    resp = _verify(
        client,
        _payload("claim", "clm_证据-①", "org-签署者-€"),
    )
    _assert_valid(resp, True)


def test_tampered_signature_byte_is_false_not_error(client):
    raw = bytearray(
        ed25519_sign(
            SEED_A, attestation_message_bytes("claim", "clm_target-1", "org-1")
        )
    )
    for index in (0, 31, 32, 63):
        tampered = bytearray(raw)
        tampered[index] ^= 0x01
        resp = _verify(
            client,
            _payload(raw_signature=bytes(tampered)),
        )
        _assert_valid(resp, False)


def test_tampered_public_key_byte_is_false(client):
    payload = _payload()
    key = bytearray(base64.b64decode(payload["public_key"]))
    key[7] ^= 0x01
    payload["public_key"] = base64.b64encode(bytes(key)).decode("ascii")
    _assert_valid(_verify(client, payload), False)


def test_signature_over_different_message_is_false(client):
    # Signed over one target, submitted for another.
    message = attestation_message_bytes("claim", "clm_other", "org-1")
    resp = _verify(client, _payload(message=message))
    _assert_valid(resp, False)


def test_swapped_target_and_signer_is_false(client):
    # The signature binds field order: a signature over (target, signer)
    # does not verify for (signer, target).
    message = attestation_message_bytes("claim", "clm_target-1", "org-1")
    payload = _payload(
        target_id="org-1", signer_actor_id="clm_target-1", message=message
    )
    _assert_valid(_verify(client, payload), False)


def test_wrong_seed_signature_is_false(client):
    payload = _payload(seed=SEED_A)
    payload["signature"] = base64.b64encode(
        ed25519_sign(
            SEED_B, attestation_message_bytes("claim", "clm_target-1", "org-1")
        )
    ).decode("ascii")
    _assert_valid(_verify(client, payload), False)


def test_response_never_echoes_key_or_signature_material(client):
    payload = _payload()
    resp = _verify(client, payload)
    _assert_valid(resp, True)
    assert payload["public_key"] not in resp.text
    assert payload["signature"] not in resp.text
    assert set(resp.json()) == {"valid"}


# --- Consistency with stateful creation ---------------------------------------


def test_created_attestation_material_verifies_true(client):
    claim = _create_claim(client)
    payload = _payload("claim", claim["id"], "org-1")
    created = client.post("/v1/attestations", json=payload)
    assert created.status_code == 201, created.text
    _assert_valid(_verify(client, payload), True)


def test_creation_rejected_material_verifies_false(client):
    claim = _create_claim(client)
    payload = _payload("claim", claim["id"], "org-1", seed=SEED_A)
    # A signature from a different key than the declared public key.
    payload["signature"] = base64.b64encode(
        ed25519_sign(
            SEED_B,
            attestation_message_bytes("claim", claim["id"], "org-1"),
        )
    ).decode("ascii")
    created = client.post("/v1/attestations", json=payload)
    assert created.status_code == 422, created.text
    assert created.json()["error"]["code"] == "attestation_verification_failed"
    _assert_valid(_verify(client, payload), False)


# --- Stateless guarantees ------------------------------------------------------


def test_repeated_requests_are_stable_and_write_nothing(client, db_session):
    _create_claim(client)
    before = _db_counts(db_session)
    good = _payload()
    bad = _payload(message=attestation_message_bytes("claim", "clm_x", "org-9"))
    first = _verify(client, good)
    second = _verify(client, good)
    _assert_valid(first, True)
    assert second.text == first.text
    _assert_valid(_verify(client, bad), False)
    _assert_valid(_verify(client, bad), False)
    # No attestation row and no audit event: persistent state is untouched.
    assert _db_counts(db_session) == before


def test_any_query_parameter_is_rejected(client):
    resp = client.post(URL + "?unexpected=1", json=_payload())
    _assert_validation_error(resp)


# --- Validation boundary --------------------------------------------------------


def test_missing_fields_are_422(client):
    payload = _payload()
    for field in payload:
        incomplete = {k: v for k, v in payload.items() if k != field}
        _assert_validation_error(_verify(client, incomplete))
    _assert_validation_error(_verify(client, {}))


def test_extra_field_is_422(client):
    payload = _payload()
    payload["unexpected"] = "value"
    _assert_validation_error(_verify(client, payload))


def test_invalid_target_type_is_422(client):
    for bad in ("actor", "Claim", "claims", "", "evidence-bundle"):
        _assert_validation_error(_verify(client, _payload(target_type=bad)))


def test_blank_identifiers_are_422(client):
    _assert_validation_error(_verify(client, _payload(target_id="")))
    _assert_validation_error(_verify(client, _payload(target_id="   ")))
    _assert_validation_error(_verify(client, _payload(signer_actor_id="")))
    _assert_validation_error(_verify(client, _payload(signer_actor_id=" \t ")))


def test_identifier_length_limits(client):
    # 80/255 characters are accepted; 81/256 are rejected.
    _assert_valid(_verify(client, _payload(target_id="t" * 80)), True)
    _assert_validation_error(_verify(client, _payload(target_id="t" * 81)))
    _assert_valid(_verify(client, _payload(signer_actor_id="s" * 255)), True)
    _assert_validation_error(_verify(client, _payload(signer_actor_id="s" * 256)))


def test_non_object_json_is_422(client):
    for body in ("[1,2,3]", '"text"', "42", "null", "true"):
        resp = client.post(
            URL, content=body, headers={"content-type": "application/json"}
        )
        _assert_validation_error(resp)


def test_non_string_base64_fields_are_422(client):
    payload = _payload()
    payload["public_key"] = 12345
    _assert_validation_error(_verify(client, payload))
    payload = _payload()
    payload["signature"] = ["not", "a", "string"]
    _assert_validation_error(_verify(client, payload))


def test_url_safe_base64_is_422(client):
    # Bytes whose standard encoding contains '+'/'/': the URL-safe alphabet
    # is a different, rejected spelling.
    raw_key = bytes([251]) * 32
    assert b"+" in base64.b64encode(raw_key)
    payload = _payload()
    payload["public_key"] = base64.urlsafe_b64encode(raw_key).decode("ascii")
    _assert_validation_error(_verify(client, payload))

    raw_sig = bytes([254]) * 64
    assert b"/" in base64.b64encode(raw_sig)
    payload = _payload()
    payload["signature"] = base64.urlsafe_b64encode(raw_sig).decode("ascii")
    _assert_validation_error(_verify(client, payload))


def test_missing_or_extra_padding_is_422(client):
    payload = _payload()
    assert payload["public_key"].endswith("=")
    payload["public_key"] = payload["public_key"].rstrip("=")
    _assert_validation_error(_verify(client, payload))

    payload = _payload()
    payload["signature"] = payload["signature"] + "="
    _assert_validation_error(_verify(client, payload))


def test_whitespace_in_base64_is_422(client):
    payload = _payload()
    payload["public_key"] = " " + payload["public_key"]
    _assert_validation_error(_verify(client, payload))

    payload = _payload()
    sig = payload["signature"]
    payload["signature"] = sig[:8] + "\n" + sig[8:]
    _assert_validation_error(_verify(client, payload))


def test_wrong_decoded_length_is_422(client):
    payload = _payload()
    payload["public_key"] = base64.b64encode(b"k" * 31).decode("ascii")
    _assert_validation_error(_verify(client, payload))

    payload = _payload()
    payload["public_key"] = base64.b64encode(b"k" * 33).decode("ascii")
    _assert_validation_error(_verify(client, payload))

    payload = _payload()
    payload["signature"] = base64.b64encode(b"s" * 63).decode("ascii")
    _assert_validation_error(_verify(client, payload))

    payload = _payload()
    payload["signature"] = base64.b64encode(b"s" * 65).decode("ascii")
    _assert_validation_error(_verify(client, payload))
