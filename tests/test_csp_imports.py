"""Tests for controlled correction-checkpoint imports (``POST /v1/csp-imports``).

Covers the success, failure, and retry branches, including:

* the first valid submission returns 201 with the stable deterministic
  ``csi_`` receipt, the three receiving-identity fields, and a UTC
  ``received_at``; the success body is compact UTF-8 JSON terminated by
  exactly one newline;
* a retried submission for the same identity returns 200 with the original
  receipt (same id and same original ``received_at``) and writes no second
  row and no second audit event;
* a digest mismatch is 422 with ``details.reason``
  ``corrections_digest_mismatch`` and the computed digest, and writes no
  receipt and no audit event;
* structural failures (malformed JSON, missing/extra fields, wrong types,
  bad version/algorithm/digest/timestamp, count mismatch) are 422 and write
  nothing;
* the corrections array is never persisted: no claim supersession or other
  resource is created -- only the receipt row and the single
  ``csp.checkpoint_imported`` audit event commit in one transaction;
* the receipt search and detail routes and the existing error JSON are
  unchanged.

All fixtures are deterministic and offline (in-memory and temporary-file
SQLite, no network).
"""

from __future__ import annotations

import hashlib
import json

import pytest
from sqlalchemy import func, select

from provenance.models import (
    AuditEvent,
    ClaimSupersession,
    CspImportRecord,
)

URL = "/v1/csp-imports"
CHECKPOINT_VERSION = "provenance-csp-checkpoint-v1"


def _digest_of(corrections: list) -> str:
    canonical = json.dumps(
        corrections,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _correction(
    *,
    id="csp_" + "0" * 64,
    superseded_claim_id="clm_" + "1" * 64,
    replacement_claim_id="clm_" + "2" * 64,
    reason="corrected source claim",
    created_at="2026-01-02T03:04:05Z",
):
    return {
        "id": id,
        "superseded_claim_id": superseded_claim_id,
        "replacement_claim_id": replacement_claim_id,
        "reason": reason,
        "created_at": created_at,
    }


def _request(corrections: list | None = None, *, digest=None) -> dict:
    corrections = [_correction()] if corrections is None else corrections
    return {
        "checkpoint": {
            "checkpoint_version": CHECKPOINT_VERSION,
            "digest_algorithm": "sha256",
            "correction_count": len(corrections),
            "corrections_digest_hex": _digest_of(corrections)
            if digest is None
            else digest,
        },
        "corrections": corrections,
    }


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def _audit_rows(db_session):
    return db_session.execute(select(AuditEvent)).scalars().all()


# --- Success branch ----------------------------------------------------------


def test_first_import_returns_201_stable_receipt_and_one_newline(
    client, db_session
):
    payload = _request()
    resp = client.post(URL, json=payload)
    assert resp.status_code == 201, resp.text

    # Compact UTF-8 JSON terminated by exactly one newline.
    raw = resp.content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert b" " not in raw

    body = resp.json()
    assert list(body) == [
        "id",
        "checkpoint_version",
        "corrections_digest_hex",
        "correction_count",
        "received_at",
    ]
    assert body["id"].startswith("csi_")
    assert len(body["id"]) == len("csi_") + 64
    assert body["checkpoint_version"] == CHECKPOINT_VERSION
    assert body["corrections_digest_hex"] == _digest_of(
        payload["corrections"]
    )
    assert body["correction_count"] == 1
    # A timezone-aware UTC timestamp on the wire.
    assert body["received_at"].endswith(("Z", "+00:00"))

    # Exactly one receipt row and one csp.checkpoint_imported audit event.
    receipts = db_session.execute(
        select(CspImportRecord)
    ).scalars().all()
    assert len(receipts) == 1
    assert receipts[0].id == body["id"]
    events = _audit_rows(db_session)
    assert len(events) == 1
    assert events[0].event_type == "csp.checkpoint_imported"
    assert events[0].resource_id == body["id"]


def test_import_creates_no_corrections_or_other_resources(
    client, db_session
):
    # The described corrections reference fabricated, nonexistent claims:
    # import is a pure function of the request body and never materializes
    # them.
    resp = client.post(URL, json=_request())
    assert resp.status_code == 201, resp.text
    assert (
        db_session.scalar(
            select(func.count()).select_from(ClaimSupersession)
        )
        == 0
    )
    # The receipt stores no raw correction data: only identity fields.
    record = db_session.execute(select(CspImportRecord)).scalar_one()
    assert not hasattr(record, "corrections")
    assert record.correction_count == 1
    assert record.checkpoint_version == CHECKPOINT_VERSION


def test_empty_corrections_array_is_a_valid_receipt(client, db_session):
    payload = _request(corrections=[])
    resp = client.post(URL, json=payload)
    assert resp.status_code == 201, resp.text
    assert resp.json()["correction_count"] == 0
    assert resp.content.endswith(b"\n")
    assert len(_audit_rows(db_session)) == 1


# --- Retry branch ------------------------------------------------------------


def test_retry_returns_200_original_receipt_and_no_new_audit(
    client, db_session
):
    payload = _request()
    first = client.post(URL, json=payload)
    assert first.status_code == 201, first.text
    first_body = first.json()

    retry = client.post(URL, json=payload)
    assert retry.status_code == 200, retry.text
    retry_body = retry.json()
    # The original receipt, including its original received_at, is served.
    assert retry_body == first_body
    assert retry.content.endswith(b"\n")

    assert (
        db_session.scalar(
            select(func.count()).select_from(CspImportRecord)
        )
        == 1
    )
    events = _audit_rows(db_session)
    assert len(events) == 1
    assert events[0].resource_id == first_body["id"]


def test_different_package_is_an_independent_receipt(client, db_session):
    first = client.post(URL, json=_request())
    assert first.status_code == 201
    other_corrections = [
        _correction(
            id="csp_" + "a" * 64,
            superseded_claim_id="clm_" + "3" * 64,
            replacement_claim_id="clm_" + "4" * 64,
            reason="a different correction",
        )
    ]
    second = client.post(URL, json=_request(other_corrections))
    assert second.status_code == 201, second.text
    assert second.json()["id"] != first.json()["id"]
    assert (
        db_session.scalar(
            select(func.count()).select_from(CspImportRecord)
        )
        == 2
    )
    assert len(_audit_rows(db_session)) == 2


# --- Failure branches --------------------------------------------------------


def test_digest_mismatch_is_422_with_reason_and_writes_nothing(
    client, db_session
):
    resp = client.post(URL, json=_request(digest="a" * 64))
    _assert_validation_error(resp)
    details = resp.json()["error"]["details"]
    assert details["reason"] == "corrections_digest_mismatch"
    assert details["computed_digest_hex"] == _digest_of(
        _request()["corrections"]
    )
    # The mismatch never echoes the corrections content.
    assert b"superseded_claim_id" not in resp.content
    assert (
        db_session.scalar(
            select(func.count()).select_from(CspImportRecord)
        )
        == 0
    )
    assert _audit_rows(db_session) == []


def test_malformed_json_is_422_and_writes_nothing(client, db_session):
    resp = client.post(
        URL,
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    _assert_validation_error(resp)
    assert _audit_rows(db_session) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.pop("checkpoint"),
        lambda p: p.pop("corrections"),
        lambda p: p.update({"extra": 1}),
        lambda p: p["checkpoint"].pop("checkpoint_version"),
        lambda p: p["checkpoint"].update({"extra": 1}),
        lambda p: p["checkpoint"].update(
            {"checkpoint_version": "other-version"}
        ),
        lambda p: p["checkpoint"].update({"digest_algorithm": "sha512"}),
        lambda p: p["checkpoint"].update(
            {"corrections_digest_hex": "A" * 64}
        ),
        lambda p: p["checkpoint"].update({"correction_count": 2}),
        lambda p: p["corrections"][0].pop("reason"),
        lambda p: p["corrections"][0].update({"extra": 1}),
        lambda p: p["corrections"][0].update(
            {"created_at": "2026-01-02 03:04:05"}
        ),
        lambda p: p["corrections"][0].update({"id": ""}),
    ],
)
def test_structural_violations_are_422_and_write_nothing(
    client, db_session, mutate
):
    payload = _request()
    mutate(payload)
    resp = client.post(URL, json=payload)
    _assert_validation_error(resp)
    assert (
        db_session.scalar(
            select(func.count()).select_from(CspImportRecord)
        )
        == 0
    )
    assert _audit_rows(db_session) == []


def test_retry_with_tampered_corrections_is_422_and_keeps_original(
    client, db_session
):
    payload = _request()
    first = client.post(URL, json=payload)
    assert first.status_code == 201
    original = first.json()

    # Same claimed identity envelope, but the corrections no longer match
    # their digest: the retry is refused and leaves the original untouched.
    tampered = json.loads(json.dumps(payload))
    tampered["corrections"][0]["reason"] = "changed after import"
    resp = client.post(URL, json=tampered)
    _assert_validation_error(resp)
    assert (
        resp.json()["error"]["details"]["reason"]
        == "corrections_digest_mismatch"
    )

    record = db_session.execute(select(CspImportRecord)).scalar_one()
    assert record.id == original["id"]
    assert len(_audit_rows(db_session)) == 1


def test_mutation_methods_on_collection_are_405(client):
    # GET on this path is the existing read-only receipt search and POST is
    # the import; the mutating verbs are not served.
    for method in ("put", "patch", "delete"):
        resp = getattr(client, method)(URL)
        assert resp.status_code == 405, (method, resp.text)
        assert resp.json()["error"]["code"] == "method_not_allowed"
