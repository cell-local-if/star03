"""Tests for the stateless attestation verification endpoint.

Covers POST /v1/attestation-verifications: a fully stateless Ed25519
verification over the exact canonical attestation message
``["provenance-attestation-v1", target_type, target_id, signer_actor_id]``.

Neither identifier is resolved against local state (no 404, no query),
nothing is created or audited, and a failed signature is a normal
``200 {"valid": false}`` verdict rather than a service error. Structural
failures are ``422 validation_error`` under the existing error JSON.
All fixtures are deterministic and offline.
"""

from __future__ import annotations

import base64
import json
import sqlite3

import pytest
from sqlalchemy import select

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

UNICODE_ACTOR = "org-证据人"


# --- Payload helpers --------------------------------------------------------


def _public_b64(seed=SEED_A) -> str:
    return base64.b64encode(ed25519_public_key(seed)).decode("ascii")


def _signature_b64(
    seed, target_type, target_id, signer_actor_id, *, message=None
) -> str:
    msg = message if message is not None else attestation_message_bytes(
        target_type, target_id, signer_actor_id
    )
    return base64.b64encode(ed25519_sign(seed, msg)).decode("ascii")


def _payload(
    target_type="claim",
    target_id="clm_unknown",
    signer_actor_id="ghost-actor",
    *,
    seed=SEED_A,
    signature_message=None,
    raw_signature=None,
    public_key=None,
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
        "public_key": public_key or _public_b64(seed),
        "signature": signature,
    }


def _assert_validation_error(resp) -> dict:
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["details"]["issues"]
    return body


def _create_content(client, actor_id="org-1", digest=DIGEST_A):
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
            "payload": {"statement": "attested"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _setup_claim(client):
    create_actor(client)
    return _create_claim(client, _create_content(client)["id"])


# --- Valid verdicts ---------------------------------------------------------


def test_valid_signature_for_unknown_identifiers_is_200_valid_true(client):
    # No actor, claim, or bundle exists: nothing is ever looked up.
    resp = client.post(URL, json=_payload("claim", "clm_ghost", "ghost"))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}
    assert set(resp.json()) == {"valid"}
    # Compact JSON body, exactly the one member.
    assert resp.text == '{"valid":true}'


def test_valid_evidence_bundle_signature_for_unknown_id(client):
    resp = client.post(
        URL, json=_payload("evidence_bundle", "evb_ghost", "ghost")
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_non_ascii_identifiers_verify_against_unescaped_utf8_message(client):
    unicode_target = "clm-证据"
    resp = client.post(
        URL,
        json=_payload(
            "evidence_bundle", unicode_target, UNICODE_ACTOR
        ),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}

    # A signature over the ASCII-escaped serialization must not verify.
    escaped = json.dumps(
        ["provenance-attestation-v1", "evidence_bundle",
         unicode_target, UNICODE_ACTOR],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    resp = client.post(
        URL,
        json=_payload(
            "evidence_bundle",
            unicode_target,
            UNICODE_ACTOR,
            signature_message=escaped,
        ),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": False}


def test_identifiers_at_max_length_boundary_verify(client):
    target_id = "c" * 80
    actor_id = "a" * 255
    resp = client.post(
        URL, json=_payload("claim", target_id, actor_id)
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_surrounding_whitespace_is_trimmed_like_creation(client):
    # The independent entry shares the exact identifier rule with creation:
    # surrounding whitespace is stripped, so the signature binds the
    # trimmed identifier.
    payload = _payload("claim", "clm_ghost", "ghost-actor")
    payload["target_id"] = "  clm_ghost\t"
    payload["signer_actor_id"] = " ghost-actor \n"
    resp = client.post(URL, json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


# --- Invalid signatures: 200 {"valid": false}, never a service error --------


def test_tampering_any_signature_byte_returns_valid_false(client):
    good = ed25519_sign(
        SEED_A, attestation_message_bytes("claim", "clm_ghost", "ghost")
    )
    for index in range(len(good)):
        tampered = bytearray(good)
        tampered[index] ^= 0x01
        resp = client.post(
            URL, json=_payload(raw_signature=bytes(tampered))
        )
        assert resp.status_code == 200, (index, resp.text)
        assert resp.json() == {"valid": False}, index


def test_tampered_public_key_byte_returns_valid_false(client):
    key = bytearray(ed25519_public_key(SEED_A))
    key[0] ^= 0x01
    resp = client.post(
        URL,
        json=_payload(public_key=base64.b64encode(bytes(key)).decode()),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": False}


def test_swapping_target_and_actor_identifiers_returns_valid_false(client):
    target_id, actor_id = "clm_ghost", "ghost-actor"
    # The signature commits to (target_id, actor_id) in that order. Sign
    # the original tuple, then submit the id fields exchanged without
    # regenerating the signature.
    payload = _payload("claim", target_id, actor_id)
    payload["target_id"], payload["signer_actor_id"] = actor_id, target_id
    resp = client.post(URL, json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": False}

    # A signature genuinely produced over the exchanged ordering does
    # verify: the failure above is field-order binding, not the
    # identifiers themselves.
    resp = client.post(URL, json=_payload("claim", actor_id, target_id))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_signature_over_other_type_returns_valid_false(client):
    # The signature commits to the literal target_type verbatim.
    resp = client.post(
        URL,
        json=_payload(
            "claim",
            "clm_ghost",
            "ghost",
            signature_message=attestation_message_bytes(
                "evidence_bundle", "clm_ghost", "ghost"
            ),
        ),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": False}


def test_wrong_public_key_returns_valid_false(client):
    resp = client.post(URL, json=_payload(seed=SEED_A, public_key=_public_b64(SEED_B)))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": False}


def test_other_serializations_and_prefixes_return_valid_false(client):
    target = ("claim", "clm_ghost", "ghost")
    wrong_prefix = json.dumps(
        ["provenance-attestation-v0", *target],
        separators=(",", ":"),
    ).encode("utf-8")
    reordered = json.dumps(
        ["provenance-attestation-v1", "ghost", "clm_ghost", "claim"],
        separators=(",", ":"),
    ).encode("utf-8")
    pretty = json.dumps(
        ["provenance-attestation-v1", *target], separators=(", ", ": ")
    ).encode("utf-8")
    for message in (wrong_prefix, reordered, pretty):
        resp = client.post(
            URL, json=_payload(signature_message=message)
        )
        assert resp.status_code == 200, message
        assert resp.json() == {"valid": False}


def test_failed_verdict_does_not_echo_submitted_material(client):
    payload = _payload(seed=SEED_A, public_key=_public_b64(SEED_B))
    resp = client.post(URL, json=payload)
    assert resp.status_code == 200
    assert resp.text == '{"valid":false}'
    assert payload["public_key"] not in resp.text
    assert payload["signature"] not in resp.text


# --- Stability and agreement with creation ----------------------------------


def test_repeated_verifications_are_stable_and_stateless(client):
    valid = _payload("claim", "clm_ghost", "ghost")
    invalid = _payload(seed=SEED_A, public_key=_public_b64(SEED_B))
    for _ in range(3):
        assert client.post(URL, json=valid).json() == {"valid": True}
        assert client.post(URL, json=invalid).json() == {"valid": False}


def test_verdict_agrees_with_attestation_creation(client, db_session):
    claim = _setup_claim(client)
    good = _payload("claim", claim["id"], "org-1")

    # Materials the creation endpoint accepts verify independently as true.
    created = client.post("/v1/attestations", json=good)
    assert created.status_code == 201, created.text
    assert client.post(URL, json=good).json() == {"valid": True}

    # The idempotent repeat creation (200) still verifies true.
    repeat = client.post("/v1/attestations", json=good)
    assert repeat.status_code == 200
    assert client.post(URL, json=good).json() == {"valid": True}

    # Materials creation rejects for signature failure verify as false,
    # without being upgraded to an error here.
    good_sig = ed25519_sign(
        SEED_A, attestation_message_bytes("claim", claim["id"], "org-1")
    )
    bad = _payload(
        "claim",
        claim["id"],
        "org-1",
        raw_signature=good_sig[:-1] + bytes([good_sig[-1] ^ 1]),
    )
    refused = client.post("/v1/attestations", json=bad)
    assert refused.status_code == 422
    assert refused.json()["error"]["code"] == "attestation_verification_failed"
    verdict = client.post(URL, json=bad)
    assert verdict.status_code == 200
    assert verdict.json() == {"valid": False}

    # One attestation, one creation audit event: verification wrote nothing.
    assert len(db_session.execute(select(Attestation)).scalars().all()) == 1


def test_unknown_local_identifiers_never_404(client):
    # No setup whatsoever; a structurally valid request is always 200.
    for target_type in ("claim", "evidence_bundle"):
        resp = client.post(
            URL, json=_payload(target_type, "totally-absent", "nobody")
        )
        assert resp.status_code == 200
        assert resp.json() == {"valid": True}


# --- Persistence is untouched ------------------------------------------------


def test_verifications_write_no_rows_or_audit_events(client, db_session):
    _setup_claim(client)
    attestations_before = len(
        db_session.execute(select(Attestation)).scalars().all()
    )
    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())

    requests = [
        _payload("claim", "clm_ghost", "ghost"),  # valid, unknown ids
        _payload("evidence_bundle", "evb_ghost", "ghost"),  # valid
        _payload(seed=SEED_A, public_key=_public_b64(SEED_B)),  # invalid
        _payload("claim", "clm_x", "ghost", raw_signature=b"\x00" * 64),
    ]
    for body in requests:
        resp = client.post(URL, json=body)
        assert resp.status_code == 200, resp.text

    assert (
        len(db_session.execute(select(Attestation)).scalars().all())
        == attestations_before
    )
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )


def test_entire_database_file_is_unchanged_by_verifications(
    tmp_db_url, file_client
):
    create_actor(file_client)
    claim = _create_claim(file_client, _create_content(file_client)["id"])
    # A real attestation exists before the verification calls.
    created = file_client.post(
        "/v1/attestations", json=_payload("claim", claim["id"], "org-1")
    )
    assert created.status_code == 201

    path = tmp_db_url.removeprefix("sqlite:///")

    def table_counts():
        con = sqlite3.connect(path)
        try:
            tables = [
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            ]
            return {
                table: con.execute(
                    f"SELECT COUNT(*) FROM {table}"  # noqa: S608 - fixed names
                ).fetchone()[0]
                for table in sorted(tables)
            }
        finally:
            con.close()

    before = table_counts()

    bodies = [
        _payload("claim", "clm_ghost", "ghost"),
        _payload("evidence_bundle", "evb_ghost", "ghost"),
        _payload(seed=SEED_A, public_key=_public_b64(SEED_B)),
        {**_payload("claim", claim["id"], "org-1"), "extra": 1},
    ]
    statuses = [file_client.post(URL, json=body).status_code for body in bodies]
    assert statuses == [200, 200, 200, 422]

    assert table_counts() == before


# --- 422 validation boundary -------------------------------------------------


def test_missing_fields_are_422(client):
    body = _assert_validation_error(client.post(URL, json={}))
    issue_fields = {
        ".".join(part for part in issue["loc"] if part != "body")
        for issue in body["error"]["details"]["issues"]
    }
    assert {
        "target_type",
        "target_id",
        "signer_actor_id",
        "public_key",
        "signature",
    }.issubset(issue_fields)


def test_extra_fields_are_422_and_never_recorded(client):
    payload = _payload()
    payload["unexpected"] = "value"
    resp = client.post(URL, json=payload)
    _assert_validation_error(resp)


@pytest.mark.parametrize("raw", ["[]", "123", "null", '"a string"', "4.5"])
def test_non_object_json_body_is_422(client, raw):
    resp = client.post(
        URL, content=raw, headers={"content-type": "application/json"}
    )
    _assert_validation_error(resp)


def test_malformed_json_body_is_422(client):
    resp = client.post(
        URL,
        content="{not valid json",
        headers={"content-type": "application/json"},
    )
    _assert_validation_error(resp)


def test_illegal_target_type_is_422(client):
    payload = _payload("claim", "clm_ghost", "ghost")
    payload["target_type"] = "content"
    _assert_validation_error(client.post(URL, json=payload))


@pytest.mark.parametrize("field", ["target_id", "signer_actor_id"])
def test_empty_or_whitespace_identifiers_are_422(client, field):
    for value in ("", "   ", "\t\n "):
        payload = _payload()
        payload[field] = value
        _assert_validation_error(client.post(URL, json=payload))


def test_overlength_identifiers_are_422(client):
    payload = _payload()
    payload["target_id"] = "c" * 81
    _assert_validation_error(client.post(URL, json=payload))

    payload = _payload()
    payload["signer_actor_id"] = "a" * 256
    _assert_validation_error(client.post(URL, json=payload))

    # Length is measured in characters, so non-ASCII counts too.
    payload = _payload()
    payload["target_id"] = "证" * 81
    _assert_validation_error(client.post(URL, json=payload))


def test_non_string_identifiers_are_422(client):
    for field in ("target_id", "signer_actor_id"):
        for value in (123, None, True, ["x"]):
            payload = _payload()
            payload[field] = value
            _assert_validation_error(client.post(URL, json=payload))


@pytest.mark.parametrize("field", ["public_key", "signature"])
def test_blank_material_is_422(client, field):
    for value in ("", "   ", "\t\n"):
        payload = _payload()
        payload[field] = value
        _assert_validation_error(client.post(URL, json=payload))


@pytest.mark.parametrize("field", ["public_key", "signature"])
def test_non_canonical_base64_material_is_422(client, field):
    canonical = _payload()[field]
    bad_values = [
        canonical[:-1] + "$",  # non-alphabet character
        canonical.rstrip("="),  # missing padding
        canonical + "=",  # excess padding
        " " + canonical,  # leading whitespace
        canonical + "\n",  # trailing newline
        "-" + canonical[1:],  # URL-safe-style character
        "_" + canonical[1:],  # URL-safe-style character
        12345,
        None,
        ["not", "a", "string"],
    ]
    for value in bad_values:
        payload = _payload()
        payload[field] = value
        _assert_validation_error(client.post(URL, json=payload))


def test_lowercase_base64_is_a_different_value_not_a_422(client):
    # Lowercase Base64 letters carry different six-bit values, so they
    # decode canonically to *different* bytes: structurally acceptable, and
    # therefore a verdict (a key mismatch -> false), never a 422.
    payload = _payload()
    payload["public_key"] = payload["public_key"].lower()
    resp = client.post(URL, json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": False}


def test_urlsafe_base64_alphabet_is_422(client):
    # Bytes 0xff force both '+' and '/' into standard Base64, which become
    # '-' and '_' in the URL-safe variant.
    payload = _payload(
        public_key=base64.urlsafe_b64encode(b"\xff" * 32).decode("ascii")
    )
    _assert_validation_error(client.post(URL, json=payload))

    payload = _payload(
        raw_signature=b"\xff" * 64
    )
    payload["signature"] = base64.urlsafe_b64encode(b"\xff" * 64).decode()
    _assert_validation_error(client.post(URL, json=payload))


@pytest.mark.parametrize("field", ["public_key", "signature"])
def test_base64_with_wrong_decoded_length_is_422(client, field):
    encodings = {
        "public_key": [
            base64.b64encode(b"k" * n).decode() for n in (0, 31, 33)
        ],
        "signature": [
            base64.b64encode(b"s" * n).decode() for n in (0, 32, 63, 65)
        ],
    }
    for value in encodings[field]:
        payload = _payload()
        payload[field] = value
        _assert_validation_error(client.post(URL, json=payload))


def test_validation_error_body_never_echoes_material(client):
    payload = _payload()
    payload["signature"] = "not base64!"
    resp = client.post(URL, json=payload)
    assert resp.status_code == 422
    assert payload["public_key"] not in resp.text
    assert "not base64!" not in resp.text


def test_any_query_parameter_is_422(client):
    resp = client.post(URL + "?x=1", json=_payload())
    _assert_validation_error(resp)
