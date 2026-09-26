"""Tests for signed audit checkpoint exchange controlled imports.

Covers POST /v1/audit-exchanges, the GET list route, and the GET detail
route:

* the request body is exactly ``{"package", "signature_metadata"}``; the
  package validates under the existing audit-checkpoint verification rules
  (structure, event count, events digest), and the metadata carries the
  fixed signature version, a non-empty signing subject, standard-Base64
  Ed25519 key/signature (exactly 32/64 decoded bytes), and the
  64-lowercase-hex package digest;
* the full package digest is recomputed under the existing package
  canonical rules (root/array order kept, nested keys sorted), and the
  Ed25519 signature is checked over the exact UTF-8 compact JSON array
  ``["provenance-audit-exchange-v1", subject, "sha256", digest]``;
* a first registration returns 201 with a stable ``acx_`` receipt; an
  exact retry returns 200 with the original receipt; a different signing
  identity or signature for the same package is a 422 validation_error
  that leaves the original record untouched; a failed signature check is
  the distinct 422 audit_signature_invalid;
* no referenced resource is ever queried or created locally (a package
  produced by another system imports identically), and the receipt row
  commits together with the ``audit.exchange_imported`` audit event,
  never storing the package, raw signature, or private key;
* GET list and detail read receipts, unknown ids are a 404
  audit_exchange_not_found, any query parameter on the detail is a 422,
  and other methods are 405 method_not_allowed. Responses are compact
  UTF-8 JSON terminated by exactly one newline.
"""

from __future__ import annotations

import base64
import hashlib
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, inspect, select

from provenance import canonical
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import (
    AuditEvent,
    AuditExchangeImportRecord,
)
from provenance.signing import audit_exchange_message_bytes
from tests.helpers import (
    SEED_A,
    SEED_B,
    create_actor,
    ed25519_public_key,
    ed25519_sign,
)

POST_PATH = "/v1/audit-exchanges"
SIGNATURE_VERSION = "provenance-audit-exchange-v1"
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
METADATA_FIELDS = {
    "signature_version",
    "signer_subject",
    "public_key",
    "signature",
    "package_digest_algorithm",
    "package_digest_hex",
}


# --- Payload construction ---------------------------------------------------------


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
    digest_hex = canonical.audit_exchange_package_digest_hex(package)
    metadata, _ = _signed_metadata(subject, digest_hex, seed)
    return {"package": package, "signature_metadata": metadata}, metadata


def _empty_package(client) -> dict:
    resp = client.get("/v1/audit-events/checkpoint/package")
    assert resp.status_code == 200
    return resp.json()


def _populated_package(client) -> dict:
    """A package with at least one event, over one created actor."""
    create_actor(client)
    package = client.get("/v1/audit-events/checkpoint/package").json()
    assert package["checkpoint"]["event_count"] >= 1
    return package


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
    package = _empty_package(client)
    request, metadata = _request(package)
    digest_hex = metadata["package_digest_hex"]

    response = client.post(POST_PATH, json=request)
    assert response.status_code == 201, response.text
    receipt = response.json()
    assert set(receipt) == RECEIPT_FIELDS
    assert list(receipt) == RECEIPT_ORDER
    assert receipt["id"].startswith("acx_")
    assert len(receipt["id"]) == len("acx_") + 64
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
    assert receipt["received_at"].endswith("Z") or receipt["received_at"].endswith(
        "+00:00"
    )


def test_response_is_compact_utf8_json_with_one_newline(client):
    request, _ = _request(_empty_package(client))
    raw = client.post(POST_PATH, json=request).content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    assert raw.startswith(b'{"id":"acx_')


def test_receipt_id_is_stable_for_the_same_package(client, tmp_db_url):
    request, metadata = _request(_empty_package(client))
    first = client.post(POST_PATH, json=request)
    assert first.status_code == 201
    import_id = first.json()["id"]

    # The id is a deterministic function of the package identity alone:
    # re-presenting the same package to a restarted service mints the same
    # id (it is an idempotent 200 there), so it survives a restart.
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
    request, _ = _request(_empty_package(client))
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
    request, metadata = _request(_empty_package(client))
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
    request, _ = _request(_empty_package(client))
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
    package = _empty_package(client)
    request, _ = _request(package, subject="system-alpha")
    first = client.post(POST_PATH, json=request)
    assert first.status_code == 201
    receipt = first.json()

    # A valid signature by the same key for a different subject: a
    # conflict against the already-registered package, not a new receipt.
    conflicting, _ = _request(package, subject="system-beta")
    response = client.post(POST_PATH, json=conflicting)
    _assert_validation_error(response)
    detail = client.get(f"{POST_PATH}/{receipt['id']}")
    assert detail.status_code == 200
    assert detail.json() == receipt


def test_different_public_key_is_422_and_leaves_record_untouched(client):
    package = _empty_package(client)
    request, _ = _request(package, subject="system-alpha", seed=SEED_A)
    receipt = client.post(POST_PATH, json=request).json()

    conflicting, _ = _request(package, subject="system-alpha", seed=SEED_B)
    response = client.post(POST_PATH, json=conflicting)
    _assert_validation_error(response)
    assert client.get(f"{POST_PATH}/{receipt['id']}").json() == receipt


def test_different_signature_for_same_identity_is_422(client, db_session):
    package = _empty_package(client)
    request, _ = _request(package, subject="system-alpha")
    receipt = client.post(POST_PATH, json=request).json()

    # Same subject/key and same claimed digest, but a different (still
    # valid) signature (e.g. a different message re-signed by the key).
    altered = json.loads(json.dumps(request))
    other_message = audit_exchange_message_bytes(
        "system-beta", DIGEST_ALGORITHM, receipt["package_digest_hex"]
    )
    altered["signature_metadata"]["signer_subject"] = "system-beta"
    altered["signature_metadata"]["signature"] = base64.b64encode(
        ed25519_sign(SEED_A, other_message)
    ).decode("ascii")
    # This signature is valid for its own (version, subject, algorithm,
    # digest) tuple but does not match the stored receipt's subject; the
    # signature verifies, yet the package is already registered by another
    # identity -> validation conflict.
    response = client.post(POST_PATH, json=altered)
    _assert_validation_error(response)
    assert _receipt_count(db_session) == 1
    assert _audit_count(db_session) == 1


def test_distinct_package_is_an_independent_201(client):
    first_package = _empty_package(client)
    first, _ = _request(first_package)
    first_receipt = client.post(POST_PATH, json=first).json()

    second_package = _populated_package(client)
    second, _ = _request(second_package)
    response = client.post(POST_PATH, json=second)
    assert response.status_code == 201
    assert response.json()["id"] != first_receipt["id"]
    assert response.json()["package_digest_hex"] != first_receipt["package_digest_hex"]


# --- Signature verification -------------------------------------------------------


def test_invalid_signature_is_distinct_422_code(client, db_session):
    package = _empty_package(client)
    request, _ = _request(package)
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
    package = _empty_package(client)
    digest_hex = canonical.audit_exchange_package_digest_hex(package)
    public_key = ed25519_public_key(SEED_A)
    # Sign the same four values in a different serialization (reordered /
    # with whitespace): verification must fail.
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
    assert (
        response.status_code == 422
        and response.json()["error"]["code"] == "audit_signature_invalid"
    )
    assert _receipt_count(db_session) == 0


# --- Package and digest validation ------------------------------------------------


def test_events_digest_mismatch_is_validation_error(client, db_session):
    package = _empty_package(client)
    request, _ = _request(package)
    request["package"]["checkpoint"]["events_digest_hex"] = "00" * 32
    response = client.post(POST_PATH, json=request)
    _assert_validation_error(response)
    assert (
        response.json()["error"]["details"]["reason"]
        == "events_digest_mismatch"
    )
    assert "computed_digest_hex" in response.json()["error"]["details"]
    assert _receipt_count(db_session) == 0


def test_event_count_mismatch_is_validation_error(client, db_session):
    package = _populated_package(client)
    request, _ = _request(package)
    request["package"]["checkpoint"]["event_count"] = (
        package["checkpoint"]["event_count"] + 1
    )
    _assert_validation_error(client.post(POST_PATH, json=request))
    assert _receipt_count(db_session) == 0


def test_package_digest_mismatch_is_validation_error(client, db_session):
    package = _empty_package(client)
    request, _ = _request(package)
    # Claim a different package digest; the signature is still over the
    # claimed (wrong) digest, so the recomputed digest mismatch is caught
    # first as a validation error.
    request["signature_metadata"]["package_digest_hex"] = "11" * 32
    response = client.post(POST_PATH, json=request)
    _assert_validation_error(response)
    assert (
        response.json()["error"]["details"]["reason"]
        == "package_digest_mismatch"
    )
    assert "computed_digest_hex" in response.json()["error"]["details"]
    assert _receipt_count(db_session) == 0


def test_malformed_package_structure_is_validation_error(client):
    package = _empty_package(client)
    request, _ = _request(package)
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
    package = _empty_package(client)
    request, _ = _request(package)
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
    package = _empty_package(client)
    request, _ = _request(package)
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
    package = _empty_package(client)
    request, _ = _request(package)
    request["signature_metadata"]["public_key"] = value
    _assert_validation_error(client.post(POST_PATH, json=request))


def test_urlsafe_base64_alphabet_is_rejected(client):
    package = _empty_package(client)
    request, _ = _request(package)
    # 32 bytes whose standard encoding uses the '+'/'/' alphabet; the
    # URL-safe spelling ('-'/'_') must not be accepted.
    raw = bytes((0xFB, 0xFF, 0xBF) * 10 + (0xFB, 0xFF))
    standard = base64.b64encode(raw).decode("ascii")
    assert "+" in standard or "/" in standard
    request["signature_metadata"]["public_key"] = standard.replace(
        "+", "-"
    ).replace("/", "_")
    _assert_validation_error(client.post(POST_PATH, json=request))


def test_missing_base64_padding_is_rejected(client):
    package = _empty_package(client)
    request, _ = _request(package)
    padded = request["signature_metadata"]["public_key"]
    assert padded.endswith("=")
    request["signature_metadata"]["public_key"] = padded.rstrip("=")
    _assert_validation_error(client.post(POST_PATH, json=request))


def test_extra_base64_padding_is_rejected(client):
    package = _empty_package(client)
    request, _ = _request(package)
    request["signature_metadata"]["public_key"] = (
        request["signature_metadata"]["public_key"] + "="
    )
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
    request, _ = _request(_empty_package(client))
    request["unexpected"] = 1
    _assert_validation_error(client.post(POST_PATH, json=request))


def test_missing_top_level_field_is_validation_error(client):
    package = _empty_package(client)
    request, _ = _request(package)
    _assert_validation_error(
        client.post(POST_PATH, json={"package": request["package"]})
    )
    _assert_validation_error(
        client.post(
            POST_PATH,
            json={"signature_metadata": request["signature_metadata"]},
        )
    )


def test_malformed_json_is_validation_error(client):
    response = client.post(
        POST_PATH,
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    _assert_validation_error(response)


def test_query_parameters_on_post_are_validation_errors(client):
    request, _ = _request(_empty_package(client))
    _assert_validation_error(client.post(f"{POST_PATH}?x=1", json=request))


def test_nonempty_populated_package_round_trips(client):
    # A package with events (the full existing verification structure)
    # validates and imports exactly like the empty package.
    request, _ = _request(_populated_package(client))
    assert client.post(POST_PATH, json=request).status_code == 201


# --- Cross-system independence ----------------------------------------------------


def test_unknown_local_resources_do_not_change_the_verdict(
    client, db_session, tmp_db_url
):
    # Build the package on one system (``producer``) and import it on a
    # separate, empty system (``receiver``): the receiver never queries or
    # creates the events the package describes, yet the signed package
    # imports successfully.
    producer = create_app(Settings(database_url=tmp_db_url))
    with TestClient(producer) as producer_client:
        package = _populated_package(producer_client)
    request, metadata = _request(package, subject="external-system")

    receiver = create_app(Settings(database_url="sqlite:///:memory:"))
    with TestClient(receiver) as receiver_client:
        assert receiver_client.post(POST_PATH, json=request).status_code == 201
        receiver_session = receiver.state.session_factory()
        try:
            # No local resource was materialized: only the exchange receipt
            # and its audit event exist.
            events = receiver_session.execute(
                select(AuditEvent.event_type)
            ).scalars().all()
            assert events == [EVENT_TYPE]
        finally:
            receiver_session.close()


# --- Detail GET -------------------------------------------------------------------


def test_get_detail_returns_the_receipt(client):
    request, _ = _request(_empty_package(client))
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
    assert body["error"]["code"] == "audit_exchange_not_found"
    assert body["error"]["details"]["import_id"] == import_id


@pytest.mark.parametrize("suffix", ("?x=1", "?import_id=x", "?limit=1"))
def test_get_detail_rejects_query_parameters(client, suffix):
    request, _ = _request(_empty_package(client))
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
    request, _ = _request(_empty_package(client))
    import_id = client.post(POST_PATH, json=request).json()["id"]
    response = getattr(client, method)(f"{POST_PATH}/{import_id}")
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"


def test_non_get_methods_on_collection_are_405(client):
    # GET is the receipt listing route on the same collection path; the
    # remaining non-POST/non-GET methods stay rejected.
    for method in ("put", "patch", "delete"):
        response = getattr(client, method)(POST_PATH)
        assert response.status_code == 405
        assert response.json()["error"]["code"] == "method_not_allowed"


# --- List GET ---------------------------------------------------------------------


def test_list_returns_receipts_in_creation_order(client):
    first, _ = _request(_empty_package(client), subject="system-alpha")
    first_receipt = client.post(POST_PATH, json=first).json()
    second, _ = _request(_populated_package(client), subject="system-beta")
    second_receipt = client.post(POST_PATH, json=second).json()

    response = client.get(POST_PATH)
    assert response.status_code == 200
    page = response.json()
    assert page["count"] == 2
    assert page["next_cursor"] is None
    assert [item["id"] for item in page["items"]] == [
        first_receipt["id"],
        second_receipt["id"],
    ]
    assert page["items"][0] == first_receipt
    assert page["items"][1] == second_receipt
    raw = response.content
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1


def test_list_filters_by_signer_subject(client):
    first, _ = _request(_empty_package(client), subject="system-alpha")
    first_receipt = client.post(POST_PATH, json=first).json()
    second, _ = _request(_populated_package(client), subject="system-beta")
    client.post(POST_PATH, json=second)

    page = client.get(POST_PATH, params={"signer_subject": "system-alpha"}).json()
    assert page["count"] == 1
    assert [item["id"] for item in page["items"]] == [first_receipt["id"]]


def test_list_filters_by_public_key_exact_spelling(client):
    request, metadata = _request(_empty_package(client))
    receipt = client.post(POST_PATH, json=request).json()

    page = client.get(
        POST_PATH, params={"public_key": metadata["public_key"]}
    ).json()
    assert page["count"] == 1
    assert [item["id"] for item in page["items"]] == [receipt["id"]]

    # A non-canonical spelling simply matches nothing.
    page = client.get(
        POST_PATH,
        params={"public_key": metadata["public_key"].rstrip("=")},
    ).json()
    assert page["count"] == 0
    assert page["items"] == []


def test_list_paginates_with_cursor(client):
    packages = [_empty_package(client), _populated_package(client)]
    create_actor(client, actor_id="org-2")
    packages.append(client.get("/v1/audit-events/checkpoint/package").json())
    for package in packages:
        request, _ = _request(package)
        assert client.post(POST_PATH, json=request).status_code == 201

    first_page = client.get(POST_PATH, params={"limit": 2}).json()
    assert first_page["count"] == 3
    assert len(first_page["items"]) == 2
    assert first_page["next_cursor"] is not None

    second_page = client.get(
        POST_PATH,
        params={"limit": 2, "cursor": first_page["next_cursor"]},
    ).json()
    assert second_page["count"] == 3
    assert len(second_page["items"]) == 1
    assert second_page["next_cursor"] is None


def test_list_rejects_unknown_and_repeated_parameters(client):
    _assert_validation_error(client.get(f"{POST_PATH}?unknown=1"))
    _assert_validation_error(
        client.get(f"{POST_PATH}?signer_subject=a&signer_subject=b")
    )


def test_list_rejects_nonempty_body(client):
    response = client.request(
        "GET",
        POST_PATH,
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    _assert_validation_error(response)


def test_list_cursor_from_another_family_is_rejected(client):
    request, _ = _request(_empty_package(client))
    client.post(POST_PATH, json=request)
    page = client.get(
        "/v1/audit-events/checkpoint-imports", params={"limit": 1}
    )
    # Mint a foreign cursor only if that family can issue one here; a
    # hand-built wrong-version token is rejected regardless.
    _assert_validation_error(
        client.get(f"{POST_PATH}?cursor=ci1.e30=.bogus")
    )


# --- Failure atomicity ------------------------------------------------------------


def test_failures_write_nothing(client, db_session):
    package = _empty_package(client)
    request, _ = _request(package)

    def send(mutated):
        return client.post(POST_PATH, json=mutated)

    # Structural, digest, and signature failures.
    bad_digest = json.loads(json.dumps(request))
    bad_digest["signature_metadata"]["package_digest_hex"] = "22" * 32
    assert send(bad_digest).status_code == 422

    bad_sig = json.loads(json.dumps(request))
    raw = bytearray(base64.b64decode(bad_sig["signature_metadata"]["signature"]))
    raw[-1] ^= 0xFF
    bad_sig["signature_metadata"]["signature"] = base64.b64encode(bytes(raw)).decode()
    assert send(bad_sig).status_code == 422

    bad_extra = json.loads(json.dumps(request))
    bad_extra["signature_metadata"]["private_key"] = "never-logged"
    assert send(bad_extra).status_code == 422

    assert send({}).status_code == 422

    db_session.expire_all()
    assert _receipt_count(db_session) == 0
    assert _audit_count(db_session) == 0


def test_conflict_failure_writes_nothing_and_preserves_receipt(client, db_session):
    package = _empty_package(client)
    request, _ = _request(package, subject="system-alpha")
    receipt = client.post(POST_PATH, json=request).json()
    conflicting, _ = _request(package, subject="system-beta")
    assert client.post(POST_PATH, json=conflicting).status_code == 422
    db_session.expire_all()
    assert _receipt_count(db_session) == 1
    assert _audit_count(db_session) == 1
    assert client.get(f"{POST_PATH}/{receipt['id']}").json() == receipt
