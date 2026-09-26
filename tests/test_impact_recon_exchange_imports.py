"""Tests for the signed impact-recon exchange import endpoints.

Covers POST /v1/impact-recon-exchange-imports and
GET /v1/impact-recon-exchange-imports/{import_id}. The request body is
exactly ``{"package", "signature_metadata"}``: ``package`` is exactly the
stateless impact-recon verification structure (``checkpoint`` pinned to
"pir-checkpoint-v1"/"sha256" with a non-negative ``entry_count`` and 64
lowercase hex ``entries_digest_hex``, plus exactly ``entry_count`` entries
of exactly the exported 8-field reconciliation view), and
``signature_metadata`` is exactly {"signature_version", "subject",
"public_key", "signature", "digest_algorithm", "package_digest_hex"} with
the version pinned to "provenance-impact-recon-exchange-v1", the algorithm
to "sha256", the digest 64 lowercase hex characters, and the key/signature
canonical standard Base64 decoding to exactly 32/64 bytes.

The entries digest is the SHA-256 of the canonical entries array and the
package digest the SHA-256 of the canonical whole-package object (object
keys sorted by Unicode code point, array order preserved, compact
separators, non-ASCII unescaped, UTF-8), both recomputed over the package
exactly as received. The Ed25519 signature must verify over the UTF-8
compact JSON array ["provenance-impact-recon-exchange-v1", subject,
digest_algorithm, package_digest_hex].

Registration is decided by the request body alone -- no id named by the
package is ever resolved against local state. Every structural, Base64,
digest, or association failure is a 422 validation_error and writes
nothing; an Ed25519 verification failure is a 422
impact_recon_signature_verification_failed and writes nothing. On success
the receiving identity is anchored on the package digest: the first
submission returns 201 with a stable deterministic ``irx_`` receipt id,
the version, subject, public key, both digests, and a UTC ``received_at``,
and writes the receipt row together with the
``revocation_impact.exchange_imported`` audit event in a single
transaction. A retry carrying the identical signature metadata returns 200
with the original receipt and no new audit; a resubmission of the same
package digest under a different signer identity or signature is a 422
and leaves the original record unchanged. The package, the raw signature,
and any private key are never persisted. GET reads only the receipt
fields, rejects every query parameter, and answers an unknown id with an
explicit 404; other methods are 405. All fixtures are deterministic and
offline.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re

from sqlalchemy import select

from provenance.models import (
    AuditEvent,
    ImpactReconExchangeImportRecord,
)
from provenance.models import EVENT_REVOCATION_IMPACT_EXCHANGE_IMPORTED
from tests.helpers import SEED_A, SEED_B, ed25519_public_key, ed25519_sign

URL = "/v1/impact-recon-exchange-imports"
SIGNATURE_VERSION = "provenance-impact-recon-exchange-v1"
PACKAGE_VERSION = "pir-checkpoint-v1"
LOCAL_CHECKPOINT_VERSION = "provenance-revocation-impact-checkpoint-v1"

RECEIPT_KEYS = {
    "id",
    "signature_version",
    "subject",
    "public_key",
    "entries_digest_hex",
    "package_digest_hex",
    "received_at",
}
METADATA_KEYS = {
    "signature_version",
    "subject",
    "public_key",
    "signature",
    "digest_algorithm",
    "package_digest_hex",
}
_IRX_ID = re.compile(r"^irx_[0-9a-f]{64}$")
_RFC3339_UTC = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(Z|\+00:00)$"
)


def _canonical(value) -> bytes:
    """Independently canonicalize a JSON value and return its bytes."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _digest_of(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _entry(**overrides) -> dict:
    entry = {
        "id": "rii_" + hashlib.sha256(b"receipt-1").hexdigest(),
        "checkpoint_version": LOCAL_CHECKPOINT_VERSION,
        "impact_count": 2,
        "impacts_digest_hex": hashlib.sha256(b"impacts").hexdigest(),
        "received_at": "2026-01-02T03:04:05Z",
        "local_available": True,
        "local_checkpoint": {
            "checkpoint_version": LOCAL_CHECKPOINT_VERSION,
            "digest_algorithm": "sha256",
            "impact_count": 2,
            "impacts_digest_hex": hashlib.sha256(b"impacts").hexdigest(),
        },
        "matches": True,
    }
    entry.update(overrides)
    return entry


def _package(entries=None) -> dict:
    """A self-contained recon package fabricated without any service state."""
    if entries is None:
        entries = [_entry()]
    return {
        "checkpoint": {
            "checkpoint_version": PACKAGE_VERSION,
            "digest_algorithm": "sha256",
            "entry_count": len(entries),
            "entries_digest_hex": _digest_of(entries),
        },
        "entries": entries,
    }


def _message_bytes(subject: str, package_digest_hex: str) -> bytes:
    return json.dumps(
        [SIGNATURE_VERSION, subject, "sha256", package_digest_hex],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _metadata(
    package: dict, *, seed: bytes = SEED_A, subject: str = "org-remote-émoji-✓"
) -> dict:
    """Signature metadata validly signed over ``package`` under ``seed``."""
    package_digest_hex = _digest_of(package)
    public_key = ed25519_public_key(seed)
    signature = ed25519_sign(seed, _message_bytes(subject, package_digest_hex))
    return {
        "signature_version": SIGNATURE_VERSION,
        "subject": subject,
        "public_key": base64.b64encode(public_key).decode("ascii"),
        "signature": base64.b64encode(signature).decode("ascii"),
        "digest_algorithm": "sha256",
        "package_digest_hex": package_digest_hex,
    }


def _body(package=None, metadata=None) -> dict:
    if package is None:
        package = _package()
    if metadata is None:
        metadata = _metadata(package)
    return {"package": package, "signature_metadata": metadata}


def _post(client, body) -> object:
    return client.post(URL, json=body)


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def _exchange_events(db_session):
    return (
        db_session.execute(
            select(AuditEvent)
            .where(
                AuditEvent.event_type
                == EVENT_REVOCATION_IMPACT_EXCHANGE_IMPORTED
            )
            .order_by(AuditEvent.seq.asc())
        )
        .scalars()
        .all()
    )


def _records(db_session):
    return (
        db_session.execute(select(ImpactReconExchangeImportRecord))
        .scalars()
        .all()
    )


# --- First registration: success shape -----------------------------------------


def test_first_import_returns_201_compact_receipt(client, db_session):
    # The database is empty and every identifier is unknown locally: the
    # receipt is decided by the body alone.
    body = _body()
    resp = _post(client, body)
    assert resp.status_code == 201, resp.text
    # Compact UTF-8 JSON terminated by exactly one newline.
    assert resp.content.endswith(b"\n")
    assert b"\n" not in resp.content[:-1]
    assert b": " not in resp.content
    receipt = resp.json()
    assert set(receipt) == RECEIPT_KEYS
    assert _IRX_ID.fullmatch(receipt["id"])
    metadata = body["signature_metadata"]
    assert receipt["signature_version"] == SIGNATURE_VERSION
    assert receipt["subject"] == metadata["subject"]
    assert receipt["public_key"] == metadata["public_key"]
    assert (
        receipt["entries_digest_hex"]
        == body["package"]["checkpoint"]["entries_digest_hex"]
    )
    assert receipt["package_digest_hex"] == metadata["package_digest_hex"]
    assert _RFC3339_UTC.fullmatch(receipt["received_at"])
    # The raw signature is never echoed.
    assert metadata["signature"] not in resp.text


def test_first_import_writes_receipt_and_audit_in_one_transaction(
    client, db_session
):
    resp = _post(client, _body())
    assert resp.status_code == 201, resp.text
    receipt_id = resp.json()["id"]
    records = _records(db_session)
    assert len(records) == 1
    assert records[0].id == receipt_id
    events = _exchange_events(db_session)
    assert len(events) == 1
    assert events[0].resource_id == receipt_id


def test_package_and_raw_signature_are_never_persisted(client, db_session):
    body = _body()
    resp = _post(client, body)
    assert resp.status_code == 201, resp.text
    record = _records(db_session)[0]
    persisted = json.dumps(
        {c.name: getattr(record, c.name) for c in record.__table__.columns},
        default=str,
    )
    # No entries array, no raw signature, no key material beyond the public
    # key: only the receiving identity and the signature's digest remain.
    assert body["signature_metadata"]["signature"] not in persisted
    assert body["package"]["entries"][0]["id"] not in persisted
    assert record.signature_digest_hex == hashlib.sha256(
        base64.b64decode(body["signature_metadata"]["signature"])
    ).hexdigest()


def test_receipt_id_is_deterministic_across_instances(app, client):
    body = _body()
    first = _post(client, body)
    assert first.status_code == 201, first.text
    # A second, independent application instance derives the same receipt
    # id from the same signed package.
    from provenance.app import create_app
    from provenance.config import Settings
    from fastapi.testclient import TestClient

    other_app = create_app(Settings(database_url="sqlite:///:memory:"))
    with TestClient(other_app) as other:
        second = _post(other, body)
    assert second.status_code == 201, second.text
    assert second.json() == first.json() | {"received_at": second.json()["received_at"]}
    assert second.json()["id"] == first.json()["id"]


# --- Idempotent retry and identity conflicts ------------------------------------


def test_identical_retry_returns_200_original_receipt(client, db_session):
    body = _body()
    first = _post(client, body)
    assert first.status_code == 201, first.text
    retry = _post(client, body)
    assert retry.status_code == 200, retry.text
    assert retry.json() == first.json()
    # No second row and no second audit event.
    assert len(_records(db_session)) == 1
    assert len(_exchange_events(db_session)) == 1


def test_resubmission_with_different_subject_is_422_and_keeps_original(
    client, db_session
):
    package = _package()
    first = _post(client, _body(package))
    assert first.status_code == 201, first.text
    # The same package digest signed naming a different subject.
    other = _body(package, _metadata(package, subject="org-other"))
    conflict = _post(client, other)
    _assert_validation_error(conflict)
    # The original record is unchanged and still reads back identically.
    assert len(_records(db_session)) == 1
    assert len(_exchange_events(db_session)) == 1
    reread = client.get(f"{URL}/{first.json()['id']}")
    assert reread.status_code == 200
    assert reread.json() == first.json()


def test_resubmission_with_different_key_is_422(client, db_session):
    package = _package()
    assert _post(client, _body(package)).status_code == 201
    other = _body(package, _metadata(package, seed=SEED_B))
    _assert_validation_error(_post(client, other))
    assert len(_records(db_session)) == 1
    assert len(_exchange_events(db_session)) == 1


def test_resubmission_with_different_signature_is_422(client, db_session):
    package = _package()
    body = _body(package)
    assert _post(client, body).status_code == 201
    # Same subject and key, but a foreign valid-looking signature over the
    # same package (here: the other seed's signature for this message).
    metadata = copy.deepcopy(body["signature_metadata"])
    metadata["signature"] = base64.b64encode(
        ed25519_sign(SEED_B, _message_bytes(metadata["subject"], metadata["package_digest_hex"]))
    ).decode("ascii")
    _assert_validation_error(
        _post(client, {"package": package, "signature_metadata": metadata})
    )
    assert len(_records(db_session)) == 1
    assert len(_exchange_events(db_session)) == 1


def test_different_package_registers_its_own_receipt(client, db_session):
    first = _post(client, _body(_package([_entry()])))
    assert first.status_code == 201, first.text
    other_package = _package([_entry(), _entry(id="rii_" + hashlib.sha256(b"receipt-2").hexdigest())])
    second = _post(client, _body(other_package))
    assert second.status_code == 201, second.text
    assert second.json()["id"] != first.json()["id"]
    assert len(_records(db_session)) == 2
    assert len(_exchange_events(db_session)) == 2


# --- Signature verification ------------------------------------------------------


def test_unverifiable_signature_is_422_specific_code(client, db_session):
    package = _package()
    metadata = _metadata(package)
    # A signature over a different message cannot verify.
    metadata["signature"] = base64.b64encode(
        ed25519_sign(SEED_A, b"an unrelated message")
    ).decode("ascii")
    resp = _post(client, {"package": package, "signature_metadata": metadata})
    assert resp.status_code == 422, resp.text
    assert (
        resp.json()["error"]["code"]
        == "impact_recon_signature_verification_failed"
    )
    # Zero writes: no receipt row and no audit event.
    assert _records(db_session) == []
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_signature_must_bind_the_exact_subject(client, db_session):
    package = _package()
    metadata = _metadata(package, subject="org-a")
    # Signed as org-a but submitted naming org-b.
    metadata["subject"] = "org-b"
    resp = _post(client, {"package": package, "signature_metadata": metadata})
    assert resp.status_code == 422
    assert (
        resp.json()["error"]["code"]
        == "impact_recon_signature_verification_failed"
    )
    assert _records(db_session) == []


def test_signature_must_bind_the_exact_package_digest(client, db_session):
    package = _package()
    metadata = _metadata(package)
    # Signed over a different digest than the metadata claims.
    other_digest = _digest_of(_package([]))
    metadata["signature"] = base64.b64encode(
        ed25519_sign(SEED_A, _message_bytes(metadata["subject"], other_digest))
    ).decode("ascii")
    resp = _post(client, {"package": package, "signature_metadata": metadata})
    assert resp.status_code == 422
    assert (
        resp.json()["error"]["code"]
        == "impact_recon_signature_verification_failed"
    )
    assert _records(db_session) == []


# --- Digest enforcement ------------------------------------------------------------


def test_entries_digest_mismatch_is_422_and_writes_nothing(client, db_session):
    body = _body()
    package = body["package"]
    package["checkpoint"]["entries_digest_hex"] = hashlib.sha256(
        b"not-the-entries"
    ).hexdigest()
    # Re-sign so the package digest matches the mutated package: only the
    # entries digest claim is wrong.
    body["signature_metadata"] = _metadata(package)
    resp = _post(client, body)
    _assert_validation_error(resp)
    assert resp.json()["error"]["details"]["reason"] == "entries_digest_mismatch"
    assert _records(db_session) == []
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_package_digest_mismatch_is_422_and_writes_nothing(client, db_session):
    body = _body()
    body["signature_metadata"]["package_digest_hex"] = hashlib.sha256(
        b"not-the-package"
    ).hexdigest()
    resp = _post(client, body)
    _assert_validation_error(resp)
    assert resp.json()["error"]["details"]["reason"] == "package_digest_mismatch"
    assert _records(db_session) == []
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_package_digest_commits_to_member_spelling(client, db_session):
    # The digest is recomputed over the package exactly as received: a body
    # whose checkpoint members arrive in a different order still verifies,
    # because the canonical form sorts object members.
    body = _body()
    checkpoint = body["package"]["checkpoint"]
    body["package"]["checkpoint"] = dict(
        sorted(checkpoint.items(), reverse=True)
    )
    resp = _post(client, body)
    assert resp.status_code == 201, resp.text


# --- Structural validation ----------------------------------------------------------


def test_malformed_json_is_422(client):
    resp = client.post(
        URL, content=b"{not json", headers={"content-type": "application/json"}
    )
    _assert_validation_error(resp)


def test_non_object_body_is_422(client):
    _assert_validation_error(_post(client, ["not", "an", "object"]))


def test_missing_members_are_422(client):
    _assert_validation_error(_post(client, {"package": _package()}))
    _assert_validation_error(
        _post(client, {"signature_metadata": _metadata(_package())})
    )
    _assert_validation_error(_post(client, {}))


def test_extra_root_member_is_422(client):
    body = _body()
    body["signature"] = "smuggled"
    _assert_validation_error(_post(client, body))


def test_extra_package_member_is_422(client):
    body = _body()
    body["package"]["raw"] = []
    _assert_validation_error(_post(client, body))


def test_extra_metadata_member_is_422(client):
    body = _body()
    body["signature_metadata"]["private_key"] = "must-not-be-accepted"
    _assert_validation_error(_post(client, body))


def test_missing_metadata_members_are_422(client):
    for key in sorted(METADATA_KEYS):
        body = _body()
        del body["signature_metadata"][key]
        _assert_validation_error(_post(client, body))


def test_wrong_signature_version_is_422(client):
    body = _body()
    body["signature_metadata"]["signature_version"] = "pir-checkpoint-v1"
    _assert_validation_error(_post(client, body))


def test_wrong_digest_algorithm_is_422(client):
    body = _body()
    body["signature_metadata"]["digest_algorithm"] = "sha512"
    _assert_validation_error(_post(client, body))


def test_blank_and_missing_subject_are_422(client):
    for subject in ("", "   "):
        body = _body()
        body["signature_metadata"]["subject"] = subject
        _assert_validation_error(_post(client, body))


def test_package_digest_hex_spelling_is_enforced(client):
    body = _body()
    good = body["signature_metadata"]["package_digest_hex"]
    for bad in (good.upper(), good[:-1], "z" * 64, 64):
        body["signature_metadata"]["package_digest_hex"] = bad
        _assert_validation_error(_post(client, body))


def test_public_key_must_be_canonical_base64_of_32_bytes(client):
    body = _body()
    good = body["signature_metadata"]["public_key"]
    raw = base64.b64decode(good)
    variants = [
        good[:-1],  # broken padding
        good[:2] + "-" + good[3:],  # URL-safe-only alphabet character
        base64.b64encode(raw + b"\x00").decode("ascii"),  # 33 bytes
        base64.b64encode(raw[:-1]).decode("ascii"),  # 31 bytes
        12345,
    ]
    for bad in variants:
        body["signature_metadata"]["public_key"] = bad
        _assert_validation_error(_post(client, body))


def test_signature_must_be_canonical_base64_of_64_bytes(client):
    body = _body()
    good = body["signature_metadata"]["signature"]
    raw = base64.b64decode(good)
    variants = [
        good[:-1],
        good[:2] + "_" + good[3:],  # URL-safe-only alphabet character
        base64.b64encode(raw + b"\x00").decode("ascii"),
        base64.b64encode(raw[:-1]).decode("ascii"),
        True,
    ]
    for bad in variants:
        body["signature_metadata"]["signature"] = bad
        _assert_validation_error(_post(client, body))


def test_package_structure_follows_verification_rules(client):
    # Entry count must match the array length.
    body = _body()
    body["package"]["checkpoint"]["entry_count"] = 99
    _assert_validation_error(_post(client, body))
    # Checkpoint version is pinned.
    body = _body()
    body["package"]["checkpoint"]["checkpoint_version"] = "other"
    _assert_validation_error(_post(client, body))
    # Entries digest must be 64 lowercase hex.
    body = _body()
    body["package"]["checkpoint"]["entries_digest_hex"] = "ABC"
    _assert_validation_error(_post(client, body))
    # An entry with an undeclared member is rejected.
    body = _body()
    body["package"]["entries"][0]["impacts"] = []
    _assert_validation_error(_post(client, body))


# --- Local state never participates -------------------------------------------------


def test_unknown_resources_do_not_change_the_outcome(client, db_session):
    # Every id named by the package is fabricated and unknown locally; the
    # import succeeds and creates none of them.
    body = _body()
    resp = _post(client, body)
    assert resp.status_code == 201, resp.text
    assert _records(db_session)[0].id == resp.json()["id"]
    # Only the receipt row and its audit event exist.
    assert len(_records(db_session)) == 1
    events = db_session.execute(select(AuditEvent)).scalars().all()
    assert len(events) == 1
    assert events[0].event_type == EVENT_REVOCATION_IMPACT_EXCHANGE_IMPORTED


# --- Receipt read ---------------------------------------------------------------------


def test_get_returns_the_original_receipt(client):
    body = _body()
    created = _post(client, body)
    assert created.status_code == 201, created.text
    resp = client.get(f"{URL}/{created.json()['id']}")
    assert resp.status_code == 200, resp.text
    assert resp.content.endswith(b"\n")
    assert resp.json() == created.json()


def test_get_unknown_id_is_404_specific_code(client):
    resp = client.get(f"{URL}/irx_{'0' * 64}")
    assert resp.status_code == 404, resp.text
    assert (
        resp.json()["error"]["code"]
        == "impact_recon_exchange_import_not_found"
    )


def test_get_rejects_any_query_parameter(client):
    created = _post(client, _body())
    import_id = created.json()["id"]
    _assert_validation_error(client.get(f"{URL}/{import_id}?limit=1"))
    _assert_validation_error(client.get(f"{URL}/{import_id}?a=1&a=2"))


def test_unsupported_methods_are_405(client):
    created = _post(client, _body())
    import_id = created.json()["id"]
    for method in ("put", "patch", "delete"):
        resp = getattr(client, method)(f"{URL}/{import_id}")
        assert resp.status_code == 405, (method, resp.text)
        assert resp.json()["error"]["code"] == "method_not_allowed"
    # No collection read or mutation exists beyond POST.
    assert client.get(URL).status_code == 405
    assert client.delete(URL).status_code == 405


def test_get_never_echoes_package_or_signature(client):
    body = _body()
    created = _post(client, body)
    resp = client.get(f"{URL}/{created.json()['id']}")
    assert resp.status_code == 200
    assert set(resp.json()) == RECEIPT_KEYS
    assert body["signature_metadata"]["signature"] not in resp.text
