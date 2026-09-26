"""Tests for signed audit checkpoint exchange controlled imports.

Covers POST /v1/audit-exchanges and the GET detail route:

* the request body is exactly ``{"package", "signature_metadata"}``; the
  package validates under the existing audit checkpoint verification
  rules (structure, event count, events digest), and the metadata carries
  the fixed signature version, a non-empty signing subject,
  standard-Base64 Ed25519 key/signature (exactly 32/64 decoded bytes),
  and the 64-lowercase-hex whole-package digest;
* the events digest is recomputed over the received events array (its
  order participating unchanged) and the full package digest is
  recomputed under the package canonical rules (root/array order kept,
  nested keys sorted); the Ed25519 signature is checked over the exact
  UTF-8 compact JSON array
  ``["provenance-audit-exchange-v1", subject, "sha256", digest]``;
* a first registration returns 201 with a stable ``acx_`` receipt; an
  exact retry returns 200 with the original receipt; a different signing
  identity or signature for the same package is a 422 validation_error
  that leaves the original record untouched; a failed signature check is
  the distinct 422 audit_signature_invalid;
* no described event is ever queried or created locally (a package
  produced by another system imports identically), and the receipt row
  commits together with the ``audit.exchange_imported`` audit event,
  never storing the package, raw signature, or private key;
* GET detail reads the receipt, unknown ids are a 404
  audit_exchange_import_not_found, any query parameter is a 422, and
  other methods are 405 method_not_allowed. Responses are compact UTF-8
  JSON terminated by exactly one newline.
"""

from __future__ import annotations

import base64
import hashlib
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, inspect, select

from provenance import canonical, ids
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import AuditEvent, AuditExchangeImportRecord
from provenance.signing import audit_exchange_message_bytes
from tests.helpers import (
    SEED_A,
    SEED_B,
    ed25519_public_key,
    ed25519_sign,
)

POST_PATH = "/v1/audit-exchanges"
SIGNATURE_VERSION = "provenance-audit-exchange-v1"
CHECKPOINT_VERSION = "provenance-audit-checkpoint-v1"
DIGEST_ALGORITHM = "sha256"
EVENT_TYPE = "audit.exchange_imported"

RECEIPT_FIELDS = {
    "id",
    "signature_version",
    "signer_subject",
    "public_key",
    "package_digest_hex",
    "signature_digest_hex",
    "received_at",
}
RECEIPT_ORDER = [
    "id",
    "signature_version",
    "signer_subject",
    "public_key",
    "package_digest_hex",
    "signature_digest_hex",
    "received_at",
]


# --- Payload construction ---------------------------------------------------------


def _events(n: int = 0) -> list[dict]:
    """Deterministic, self-contained audit-event arrays for offline tests."""
    return [
        {
            "event_type": f"event.type.{i % 3}",
            "resource_id": f"resource-{i}",
            "created_at": f"2026-01-{(i % 28) + 1:02d}T12:30:0{i % 10}Z",
        }
        for i in range(n)
    ]


def _package(events: list[dict]) -> dict:
    """Build a structurally valid audit checkpoint package for ``events``."""
    return {
        "checkpoint": {
            "checkpoint_version": CHECKPOINT_VERSION,
            "digest_algorithm": DIGEST_ALGORITHM,
            "event_count": len(events),
            "events_digest_hex": canonical.audit_events_digest_hex(events),
        },
        "events": events,
    }


def _signed_metadata(subject: str, package_digest_hex: str, seed: bytes = SEED_A):
    """Return (metadata dict, raw signature bytes) for a package digest."""
    public_key = ed25519_public_key(seed)
    message = audit_exchange_message_bytes(
        subject, DIGEST_ALGORITHM, package_digest_hex
    )
    signature = ed25519_sign(seed, message)
    return (
        {
            "signature_version": SIGNATURE_VERSION,
            "signer_subject": subject,
            "public_key": base64.b64encode(public_key).decode("ascii"),
            "signature": base64.b64encode(signature).decode("ascii"),
            "package_digest_algorithm": DIGEST_ALGORITHM,
            "package_digest_hex": package_digest_hex,
        },
        signature,
    )


def _request(package: dict, *, subject="remote-system", seed=SEED_A):
    """Build a valid signed import request for one package."""
    digest_hex = canonical.audit_checkpoint_package_digest_hex(package)
    metadata, _ = _signed_metadata(subject, digest_hex, seed)
    return {"package": package, "signature_metadata": metadata}, metadata


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def _audit_count(db_session) -> int:
    return db_session.scalar(
        select(func.count())
        .select_from(AuditEvent)
        .where(AuditEvent.event_type == EVENT_TYPE)
    )


def _receipt_count(db_session) -> int:
    return db_session.scalar(
        select(func.count()).select_from(AuditExchangeImportRecord)
    )


# --- First creation ---------------------------------------------------------------


def test_first_import_returns_201_stable_receipt(client, db_session):
    package = _package(_events(2))
    request, metadata = _request(package)
    digest_hex = metadata["package_digest_hex"]

    response = client.post(POST_PATH, json=request)
    assert response.status_code == 201, response.text
    receipt = response.json()
    assert set(receipt) == RECEIPT_FIELDS
    assert list(receipt) == RECEIPT_ORDER
    assert receipt["id"].startswith("acx_")
    assert len(receipt["id"]) == len("acx_") + 64
    # The id is determined by the exchange version and the whole-package
    # digest alone.
    assert receipt["id"] == ids.audit_exchange_import_id(
        SIGNATURE_VERSION, digest_hex
    )
    assert receipt["signature_version"] == SIGNATURE_VERSION
    assert receipt["signer_subject"] == "remote-system"
    assert receipt["public_key"] == metadata["public_key"]
    assert receipt["package_digest_hex"] == digest_hex
    # The signature digest is the SHA-256 of the verified signature; the
    # raw signature itself is not echoed.
    raw_signature = base64.b64decode(metadata["signature"])
    assert receipt["signature_digest_hex"] == hashlib.sha256(
        raw_signature
    ).hexdigest()
    assert receipt["received_at"].endswith(("Z", "+00:00"))


def test_empty_event_array_package_imports(client):
    package = _package([])
    assert (
        canonical.audit_events_digest_hex([])
        == hashlib.sha256(b"[]").hexdigest()
    )
    request, _ = _request(package)
    response = client.post(POST_PATH, json=request)
    assert response.status_code == 201, response.text
    assert response.json()["id"].startswith("acx_")


def test_response_is_compact_utf8_json_with_one_newline(client):
    request, _ = _request(_package(_events(1)))
    raw = client.post(POST_PATH, json=request).content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    assert raw.startswith(b'{"id":"acx_')


def test_receipt_id_is_stable_for_the_same_package(client, tmp_db_url):
    request, metadata = _request(_package(_events(1)))
    first = client.post(POST_PATH, json=request)
    assert first.status_code == 201
    import_id = first.json()["id"]

    # The id is a deterministic function of the package identity alone:
    # re-presenting the same package to a restarted service mints the same
    # id, so it survives a restart.
    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as other:
        resp = other.post(POST_PATH, json=request)
        assert resp.status_code == 201, resp.text
        assert resp.json()["id"] == import_id


def test_first_creation_writes_receipt_and_audit_event_in_one_commit(
    client, db_session
):
    assert _audit_count(db_session) == 0
    assert _receipt_count(db_session) == 0
    request, _ = _request(_package(_events(1)))
    assert client.post(POST_PATH, json=request).status_code == 201
    db_session.expire_all()
    assert _receipt_count(db_session) == 1
    assert _audit_count(db_session) == 1
    event = db_session.execute(
        select(AuditEvent).where(AuditEvent.event_type == EVENT_TYPE)
    ).scalar_one()
    record = db_session.execute(
        select(AuditExchangeImportRecord)
    ).scalar_one()
    assert event.resource_id == record.id


def test_package_raw_signature_and_private_key_are_never_persisted(
    client, db_session
):
    request, metadata = _request(_package(_events(1)))
    assert client.post(POST_PATH, json=request).status_code == 201
    columns = {
        column["name"]
        for column in inspect(db_session.bind).get_columns(
            "audit_exchange_imports"
        )
    }
    assert columns == {
        "seq",
        "id",
        "signature_version",
        "signer_subject",
        "public_key",
        "package_digest_algorithm",
        "package_digest_hex",
        "signature_digest_algorithm",
        "signature_digest_hex",
        "created_at",
    }
    # The receipt view never echoes the package or the raw signature.
    rendered = client.post(POST_PATH, json=request).text
    assert "events" not in rendered
    assert metadata["signature"] not in rendered
    assert '"signature"' not in rendered


# --- Idempotency and conflicts ----------------------------------------------------


def test_exact_retry_returns_200_original_receipt_without_new_write(
    client, db_session
):
    request, _ = _request(_package(_events(1)))
    first = client.post(POST_PATH, json=request)
    assert first.status_code == 201
    receipt = first.json()

    retry = client.post(POST_PATH, json=request)
    assert retry.status_code == 200
    assert retry.json() == receipt
    db_session.expire_all()
    assert _receipt_count(db_session) == 1
    assert _audit_count(db_session) == 1


def test_different_signing_subject_is_422_and_leaves_record_untouched(client):
    package = _package(_events(1))
    request, _ = _request(package, subject="system-alpha")
    first = client.post(POST_PATH, json=request)
    assert first.status_code == 201
    receipt = first.json()

    # A valid signature for the same package under a different subject: a
    # conflict against the already-registered package, not a new receipt.
    conflicting, _ = _request(package, subject="system-beta")
    response = client.post(POST_PATH, json=conflicting)
    _assert_validation_error(response)
    assert (
        response.json()["error"]["details"]["reason"]
        == "exchange_import_identity_conflict"
    )
    detail = client.get(f"{POST_PATH}/{receipt['id']}")
    assert detail.status_code == 200
    assert detail.json() == receipt


def test_different_public_key_is_422_and_leaves_record_untouched(client):
    package = _package(_events(1))
    request, _ = _request(package, subject="system-alpha", seed=SEED_A)
    receipt = client.post(POST_PATH, json=request).json()

    conflicting, _ = _request(package, subject="system-alpha", seed=SEED_B)
    response = client.post(POST_PATH, json=conflicting)
    _assert_validation_error(response)
    assert client.get(f"{POST_PATH}/{receipt['id']}").json() == receipt


def test_different_signature_for_same_identity_is_422(client, db_session):
    package = _package(_events(1))
    request, _ = _request(package, subject="system-alpha")
    receipt = client.post(POST_PATH, json=request).json()

    # Same key and same claimed package digest, but a signature over a
    # different (valid) tuple. It verifies for its own tuple but the
    # package is already registered under another identity -> conflict.
    altered = json.loads(json.dumps(request))
    other_message = audit_exchange_message_bytes(
        "system-beta", DIGEST_ALGORITHM, receipt["package_digest_hex"]
    )
    altered["signature_metadata"]["signer_subject"] = "system-beta"
    altered["signature_metadata"]["signature"] = base64.b64encode(
        ed25519_sign(SEED_A, other_message)
    ).decode("ascii")
    response = client.post(POST_PATH, json=altered)
    _assert_validation_error(response)
    assert _receipt_count(db_session) == 1
    assert _audit_count(db_session) == 1


def test_distinct_package_is_an_independent_201(client):
    first, _ = _request(_package(_events(1)))
    first_receipt = client.post(POST_PATH, json=first).json()

    second, _ = _request(_package(_events(2)))
    response = client.post(POST_PATH, json=second)
    assert response.status_code == 201
    assert response.json()["id"] != first_receipt["id"]
    assert (
        response.json()["package_digest_hex"]
        != first_receipt["package_digest_hex"]
    )


def test_event_array_order_participates_in_the_digest(client, db_session):
    events = _events(3)
    reordered = list(reversed(events))
    package_a = _package(events)
    package_b = _package(reordered)
    # The reordered array has a different events digest and a different
    # whole-package digest.
    assert (
        package_a["checkpoint"]["events_digest_hex"]
        != package_b["checkpoint"]["events_digest_hex"]
    )
    request_a, _ = _request(package_a)
    request_b, _ = _request(package_b)
    ra = client.post(POST_PATH, json=request_a)
    rb = client.post(POST_PATH, json=request_b)
    assert ra.status_code == 201 and rb.status_code == 201
    assert ra.json()["id"] != rb.json()["id"]
    db_session.expire_all()
    assert _receipt_count(db_session) == 2


# --- Signature verification -------------------------------------------------------


def test_invalid_signature_is_distinct_422_code(client, db_session):
    request, _ = _request(_package(_events(1)))
    raw = base64.b64decode(request["signature_metadata"]["signature"])
    tampered = bytearray(raw)
    tampered[0] ^= 0x01
    request["signature_metadata"]["signature"] = base64.b64encode(
        bytes(tampered)
    ).decode("ascii")
    response = client.post(POST_PATH, json=request)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "audit_signature_invalid"
    db_session.expire_all()
    assert _receipt_count(db_session) == 0
    assert _audit_count(db_session) == 0


def test_signature_over_other_serialization_fails(client, db_session):
    package = _package(_events(1))
    digest_hex = canonical.audit_checkpoint_package_digest_hex(package)
    public_key = ed25519_public_key(SEED_A)
    # Sign the same values in a different serialization (reordered /
    # whitespace): verification must fail.
    other = json.dumps(
        ["remote-system", SIGNATURE_VERSION, digest_hex, DIGEST_ALGORITHM],
        separators=(", ", ": "),
    ).encode("utf-8")
    signature = ed25519_sign(SEED_A, other)
    metadata = {
        "signature_version": SIGNATURE_VERSION,
        "signer_subject": "remote-system",
        "public_key": base64.b64encode(public_key).decode("ascii"),
        "signature": base64.b64encode(signature).decode("ascii"),
        "package_digest_algorithm": DIGEST_ALGORITHM,
        "package_digest_hex": digest_hex,
    }
    response = client.post(
        POST_PATH, json={"package": package, "signature_metadata": metadata}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "audit_signature_invalid"
    assert _receipt_count(db_session) == 0


def test_signature_over_another_domain_prefix_fails(client, db_session):
    # A signature minted under the impact-recon exchange prefix over
    # otherwise identical values must not verify here.
    package = _package(_events(1))
    digest_hex = canonical.audit_checkpoint_package_digest_hex(package)
    public_key = ed25519_public_key(SEED_A)
    other = json.dumps(
        ["provenance-impact-recon-exchange-v1", "remote-system", "sha256", digest_hex],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    metadata = {
        "signature_version": SIGNATURE_VERSION,
        "signer_subject": "remote-system",
        "public_key": base64.b64encode(public_key).decode("ascii"),
        "signature": base64.b64encode(ed25519_sign(SEED_A, other)).decode("ascii"),
        "package_digest_algorithm": DIGEST_ALGORITHM,
        "package_digest_hex": digest_hex,
    }
    response = client.post(
        POST_PATH, json={"package": package, "signature_metadata": metadata}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "audit_signature_invalid"
    assert _receipt_count(db_session) == 0


# --- Package and digest validation ------------------------------------------------


def test_events_digest_mismatch_is_validation_error(client, db_session):
    request, _ = _request(_package(_events(1)))
    request["package"]["checkpoint"]["events_digest_hex"] = "00" * 32
    response = client.post(POST_PATH, json=request)
    _assert_validation_error(response)
    details = response.json()["error"]["details"]
    assert details["reason"] == "events_digest_mismatch"
    # The independent recomputation is carried alongside the reason.
    assert details["computed_digest_hex"] == canonical.audit_events_digest_hex(
        request["package"]["events"]
    )
    assert _receipt_count(db_session) == 0


def test_package_digest_mismatch_is_validation_error(client, db_session):
    request, _ = _request(_package(_events(1)))
    # Claim a different package digest while keeping the signature over the
    # claimed (wrong) digest; the recomputed package digest mismatch is
    # caught before signature verification.
    claimed = "11" * 32
    public_key = ed25519_public_key(SEED_A)
    message = audit_exchange_message_bytes("remote-system", "sha256", claimed)
    request["signature_metadata"]["package_digest_hex"] = claimed
    request["signature_metadata"]["signature"] = base64.b64encode(
        ed25519_sign(SEED_A, message)
    ).decode("ascii")
    response = client.post(POST_PATH, json=request)
    _assert_validation_error(response)
    details = response.json()["error"]["details"]
    assert details["reason"] == "package_digest_mismatch"
    assert (
        details["computed_digest_hex"]
        == canonical.audit_checkpoint_package_digest_hex(request["package"])
    )
    assert _receipt_count(db_session) == 0


def test_event_count_mismatch_is_validation_error(client, db_session):
    request, _ = _request(_package(_events(2)))
    request["package"]["checkpoint"]["event_count"] = 3
    _assert_validation_error(client.post(POST_PATH, json=request))
    assert _receipt_count(db_session) == 0


def test_malformed_package_structure_is_validation_error(client):
    request, _ = _request(_package(_events(1)))
    request["package"]["checkpoint"]["checkpoint_version"] = "other-v1"
    _assert_validation_error(client.post(POST_PATH, json=request))


@pytest.mark.parametrize(
    "mutation",
    (
        lambda md: md.update({"signature_version": "other-v1"}),
        lambda md: md.update({"package_digest_algorithm": "sha512"}),
        lambda md: md.update({"package_digest_hex": "00" * 32}),
        lambda md: md.update({"package_digest_hex": ("A" + "00" * 32)[:64]}),
        lambda md: md.update({"package_digest_hex": "0" * 63}),
        lambda md: md.update({"signer_subject": "   "}),
        lambda md: md.update({"signer_subject": ""}),
        lambda md: md.update({"extra": 1}),
    ),
)
def test_bad_metadata_fields_are_validation_errors(client, mutation):
    request, _ = _request(_package(_events(1)))
    mutation(request["signature_metadata"])
    _assert_validation_error(client.post(POST_PATH, json=request))


@pytest.mark.parametrize(
    "field,factory",
    (
        ("public_key", b"\x00" * 31),
        ("public_key", b"\x00" * 33),
        ("signature", b"\x00" * 63),
        ("signature", b"\x00" * 65),
    ),
)
def test_wrong_decoded_lengths_are_validation_errors(client, field, factory):
    request, _ = _request(_package(_events(1)))
    request["signature_metadata"][field] = base64.b64encode(factory).decode()
    _assert_validation_error(client.post(POST_PATH, json=request))


@pytest.mark.parametrize(
    "value",
    (
        "not base64",
        "AAAA",  # urlsafe/valid alphabet but wrong length
        "AAAA/",  # standard alphabet but bad length (padded)
    ),
)
def test_noncanonical_base64_is_validation_error(client, value):
    request, _ = _request(_package(_events(1)))
    request["signature_metadata"]["public_key"] = value
    _assert_validation_error(client.post(POST_PATH, json=request))


def test_urlsafe_base64_alphabet_is_rejected(client):
    request, _ = _request(_package(_events(1)))
    # 32 bytes whose standard encoding uses the '+'/'/' alphabet; the
    # URL-safe spelling ('-'/'_') must not be accepted.
    raw = bytes((0xFB, 0xFF, 0xBF) * 10 + (0xFB, 0xFF))
    standard = base64.b64encode(raw).decode("ascii")
    assert "+" in standard or "/" in standard
    request["signature_metadata"]["public_key"] = standard.replace(
        "+", "-"
    ).replace("/", "_")
    _assert_validation_error(client.post(POST_PATH, json=request))


@pytest.mark.parametrize(
    "value",
    (
        # Missing padding for 32-byte (standard spelling ends in '=').
        lambda: base64.b64encode(b"\x00" * 32).decode("ascii").rstrip("="),
        # Extra padding.
        lambda: base64.b64encode(b"\x00" * 30).decode("ascii") + "==",
    ),
)
def test_missing_or_extra_base64_padding_is_rejected(client, value):
    request, _ = _request(_package(_events(1)))
    request["signature_metadata"]["public_key"] = value()
    _assert_validation_error(client.post(POST_PATH, json=request))


@pytest.mark.parametrize(
    "body",
    (
        {},
        {"package": None},
        {"signature_metadata": None},
        {"package": {}},
        {"package": {}, "signature_metadata": {}},
        [],
        "package-only",
        123,
    ),
)
def test_malformed_top_level_body_is_validation_error(client, body):
    _assert_validation_error(client.post(POST_PATH, json=body))


def test_extra_top_level_field_is_validation_error(client):
    request, _ = _request(_package(_events(1)))
    request["unexpected"] = 1
    _assert_validation_error(client.post(POST_PATH, json=request))


def test_missing_top_level_field_is_validation_error(client):
    request, _ = _request(_package(_events(1)))
    _assert_validation_error(
        client.post(POST_PATH, json={"package": request["package"]})
    )
    _assert_validation_error(
        client.post(
            POST_PATH,
            json={"signature_metadata": request["signature_metadata"]},
        )
    )


def test_extra_nested_fields_are_validation_errors(client):
    request, _ = _request(_package(_events(1)))
    request["package"]["events"][0]["unexpected"] = 1
    _assert_validation_error(client.post(POST_PATH, json=request))
    request2, _ = _request(_package(_events(1)))
    request2["package"]["checkpoint"]["extra"] = 1
    _assert_validation_error(client.post(POST_PATH, json=request2))


def test_malformed_json_is_validation_error(client):
    response = client.post(
        POST_PATH,
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    _assert_validation_error(response)


def test_query_parameters_on_post_are_validation_errors(client):
    request, _ = _request(_package(_events(1)))
    _assert_validation_error(
        client.post(f"{POST_PATH}?x=1", json=request)
    )


def test_non_ascii_subject_is_signed_and_verified(client):
    # Non-ASCII subjects participate verbatim in the compact UTF-8 signed
    # array (never \u-escaped), so a correctly signed request verifies.
    request, _ = _request(_package(_events(1)), subject="系统-✓")
    response = client.post(POST_PATH, json=request)
    assert response.status_code == 201, response.text
    assert response.json()["signer_subject"] == "系统-✓"


# --- Cross-system independence ----------------------------------------------------


def test_described_events_are_never_materialized_locally(client, db_session):
    # The package describes audit events that need not exist locally;
    # importing never queries, creates, or modifies them. The only audit
    # row is the import's own receipt event.
    request, _ = _request(_package(_events(3)))
    assert client.post(POST_PATH, json=request).status_code == 201
    db_session.expire_all()
    event_types = (
        db_session.execute(select(AuditEvent.event_type)).scalars().all()
    )
    assert event_types == [EVENT_TYPE]
    assert _receipt_count(db_session) == 1


def test_external_package_imports_on_an_empty_receiver(tmp_db_url):
    # Build and sign a package on one system, import it on a separate empty
    # receiver: the verdict depends on the body alone.
    producer = create_app(Settings(database_url=tmp_db_url))
    with TestClient(producer):
        package = _package(_events(2))
    request, _ = _request(package, subject="external-system")

    receiver = create_app(Settings(database_url="sqlite:///:memory:"))
    with TestClient(receiver) as receiver_client:
        assert (
            receiver_client.post(POST_PATH, json=request).status_code == 201
        )


# --- Detail GET -------------------------------------------------------------------


def test_get_detail_returns_the_receipt(client):
    request, _ = _request(_package(_events(1)))
    receipt = client.post(POST_PATH, json=request).json()
    detail = client.get(f"{POST_PATH}/{receipt['id']}")
    assert detail.status_code == 200
    assert detail.json() == receipt
    raw = detail.content
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1


def test_get_unknown_import_is_404(client):
    import_id = "acx_" + "00" * 32
    response = client.get(f"{POST_PATH}/{import_id}")
    assert response.status_code == 404, response.text
    body = response.json()
    assert body["error"]["code"] == "audit_exchange_import_not_found"
    assert body["error"]["details"]["import_id"] == import_id


@pytest.mark.parametrize("suffix", ("?x=1", "?import_id=x", "?limit=1"))
def test_get_detail_rejects_query_parameters(client, suffix):
    request, _ = _request(_package(_events(1)))
    import_id = client.post(POST_PATH, json=request).json()["id"]
    response = client.get(f"{POST_PATH}/{import_id}{suffix}")
    _assert_validation_error(response)


def test_get_detail_query_parameter_validated_before_lookup(client):
    # An unknown id together with a query parameter is the 422, since the
    # parameters are checked first.
    response = client.get(f"{POST_PATH}/acx_x?x=1")
    _assert_validation_error(response)


@pytest.mark.parametrize("method", ("put", "patch", "delete", "post"))
def test_non_get_methods_on_detail_are_405(client, method):
    request, _ = _request(_package(_events(1)))
    import_id = client.post(POST_PATH, json=request).json()["id"]
    response = getattr(client, method)(f"{POST_PATH}/{import_id}")
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"


def test_non_get_methods_on_collection_are_405(client):
    for method in ("put", "patch", "delete"):
        response = getattr(client, method)(POST_PATH)
        assert response.status_code == 405
        assert response.json()["error"]["code"] == "method_not_allowed"


# --- Failure atomicity ------------------------------------------------------------


def test_failures_write_nothing(client, db_session):
    request, _ = _request(_package(_events(1)))

    def send(mutated):
        return client.post(POST_PATH, json=mutated)

    # Structural, digest, and signature failures.
    bad_digest = json.loads(json.dumps(request))
    bad_digest["signature_metadata"]["package_digest_hex"] = "22" * 32
    assert send(bad_digest).status_code == 422

    bad_events = json.loads(json.dumps(request))
    bad_events["package"]["checkpoint"]["events_digest_hex"] = "33" * 32
    assert send(bad_events).status_code == 422

    bad_sig = json.loads(json.dumps(request))
    raw = bytearray(
        base64.b64decode(bad_sig["signature_metadata"]["signature"])
    )
    raw[-1] ^= 0xFF
    bad_sig["signature_metadata"]["signature"] = base64.b64encode(
        bytes(raw)
    ).decode()
    assert send(bad_sig).status_code == 422

    bad_extra = json.loads(json.dumps(request))
    bad_extra["signature_metadata"]["private_key"] = "never-logged"
    assert send(bad_extra).status_code == 422

    assert send({}).status_code == 422

    db_session.expire_all()
    assert _receipt_count(db_session) == 0
    assert _audit_count(db_session) == 0


def test_conflict_failure_writes_nothing_and_preserves_receipt(
    client, db_session
):
    package = _package(_events(1))
    request, _ = _request(package, subject="system-alpha")
    receipt = client.post(POST_PATH, json=request).json()
    conflicting, _ = _request(package, subject="system-beta")
    assert client.post(POST_PATH, json=conflicting).status_code == 422
    db_session.expire_all()
    assert _receipt_count(db_session) == 1
    assert _audit_count(db_session) == 1
    assert client.get(f"{POST_PATH}/{receipt['id']}").json() == receipt
