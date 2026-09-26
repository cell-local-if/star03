"""Tests for controlled correction-checkpoint imports (``POST /v1/csp-imports``).

The baseline serves this route through the framework renderer, which did
not terminate the receipt with a newline; these tests pin the compact
single-newline wire contract and re-confirm the route's unchanged
guarantees:

* strict structure validation (exact top-level/nested fields, fixed
  version/algorithm, 64-lowercase-hex digest, strict RFC 3339 UTC
  timestamps, count/array length agreement);
* a canonical digest mismatch is a 422 carrying
  ``details.reason="corrections_digest_mismatch"`` and the computed
  digest, and writes nothing;
* the first submission returns 201 with the stable ``csi_`` receipt and a
  UTC ``received_at``, and the body is compact UTF-8 JSON terminated by
  exactly one newline;
* an exact retry returns 200 with the original receipt (same
  ``received_at``), no second row, and no second audit event;
* the receipt and audit commit transactionally, the corrections array is
  never persisted, and verification is purely a function of the body.

All fixtures are deterministic and offline (in-memory and temporary-file
SQLite, no network).
"""

from __future__ import annotations

import hashlib
import json

from sqlalchemy import select

from provenance.models import (
    EVENT_CSP_CHECKPOINT_IMPORTED,
    AuditEvent,
    ClaimSupersession,
    CspImportRecord,
)

URL = "/v1/csp-imports"
VERSION = "provenance-csp-checkpoint-v1"


def _correction(
    *,
    id="csp_" + "a" * 64,
    superseded="clm_" + "1" * 64,
    replacement="clm_" + "2" * 64,
    reason="corrected provenance",
    created_at="2026-01-02T03:04:05Z",
):
    return {
        "id": id,
        "superseded_claim_id": superseded,
        "replacement_claim_id": replacement,
        "reason": reason,
        "created_at": created_at,
    }


def _digest_of(corrections: list) -> str:
    return hashlib.sha256(
        json.dumps(
            corrections,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _body(corrections: list | None = None, *, digest=None, count=None):
    corrections = corrections if corrections is not None else []
    return {
        "checkpoint": {
            "checkpoint_version": VERSION,
            "digest_algorithm": "sha256",
            "correction_count": (
                len(corrections) if count is None else count
            ),
            "corrections_digest_hex": (
                _digest_of(corrections) if digest is None else digest
            ),
        },
        "corrections": corrections,
    }


# --- Success wire format -----------------------------------------------------


def test_first_import_201_compact_json_single_newline(client, db_session):
    payload = _body([_correction()])
    resp = client.post(URL, json=payload)
    assert resp.status_code == 201, resp.text
    raw = resp.content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    # Compact separators: no incidental whitespace.
    assert b" " not in raw
    assert b": " not in raw

    body = json.loads(raw.decode("utf-8"))
    assert list(body) == [
        "id",
        "checkpoint_version",
        "corrections_digest_hex",
        "correction_count",
        "received_at",
    ]
    assert body["id"].startswith("csi_")
    assert body["checkpoint_version"] == VERSION
    assert body["correction_count"] == 1
    assert body["corrections_digest_hex"] == _digest_of([_correction()])
    assert body["received_at"].endswith(("Z", "+00:00"))

    # The receipt row and the audit event commit together.
    records = db_session.execute(select(CspImportRecord)).scalars().all()
    assert len(records) == 1
    events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_CSP_CHECKPOINT_IMPORTED
        )
    ).scalars().all()
    assert len(events) == 1
    assert events[0].resource_id == records[0].id
    # The corrections array is never persisted.
    assert db_session.execute(select(ClaimSupersession)).scalars().all() == []
    assert not hasattr(records[0], "corrections")


def test_empty_corrections_set_is_a_valid_first_import(client, db_session):
    resp = client.post(URL, json=_body([]))
    assert resp.status_code == 201, resp.text
    raw = resp.content
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    body = json.loads(raw)
    assert body["correction_count"] == 0
    assert body["corrections_digest_hex"] == _digest_of([])


# --- Idempotent retry --------------------------------------------------------


def test_retry_returns_200_original_receipt_and_no_new_audit(
    client, db_session
):
    payload = _body([_correction()])
    first = client.post(URL, json=payload)
    assert first.status_code == 201
    first_body = first.json()
    events_after_first = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    retry = client.post(URL, json=payload)
    assert retry.status_code == 200, retry.text
    # The retry body obeys the same compact single-newline contract.
    assert retry.content.endswith(b"\n")
    assert not retry.content.endswith(b"\n\n")
    retry_body = retry.json()
    assert retry_body == first_body

    assert len(db_session.execute(select(CspImportRecord)).scalars().all()) == 1
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_after_first
    )


def test_retry_with_non_matching_corrections_is_422_and_keeps_receipt(
    client, db_session
):
    payload = _body([_correction(reason="first reason")])
    assert client.post(URL, json=payload).status_code == 201

    # Same receiving identity fields are recomputed from the retried body;
    # a body whose corrections no longer match the claimed digest is a 422
    # and leaves the original record untouched.
    tampered = _body(
        [_correction(reason="first reason")],
        digest="0" * 64,
    )
    resp = client.post(URL, json=tampered)
    assert resp.status_code == 422, resp.text
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"]["reason"] == "corrections_digest_mismatch"
    assert error["details"]["computed_digest_hex"] == _digest_of(
        [_correction(reason="first reason")]
    )
    assert len(db_session.execute(select(CspImportRecord)).scalars().all()) == 1


# --- Structural validation ---------------------------------------------------


def _post_raw(client, raw: bytes):
    return client.post(
        URL,
        content=raw,
        headers={"content-type": "application/json"},
    )


def test_structure_violations_are_422_and_write_nothing(client, db_session):
    good = _body([_correction()])
    cases: list[bytes] = []
    # Malformed JSON.
    cases.append(b"{not json")
    # Extra top-level field.
    extra = json.loads(json.dumps(good))
    extra["unexpected"] = True
    cases.append(json.dumps(extra).encode())
    # Missing member.
    missing = json.loads(json.dumps(good))
    del missing["corrections"]
    cases.append(json.dumps(missing).encode())
    # Wrong checkpoint version/algorithm.
    wrong_version = json.loads(json.dumps(good))
    wrong_version["checkpoint"]["checkpoint_version"] = "other"
    cases.append(json.dumps(wrong_version).encode())
    wrong_algorithm = json.loads(json.dumps(good))
    wrong_algorithm["checkpoint"]["digest_algorithm"] = "sha512"
    cases.append(json.dumps(wrong_algorithm).encode())
    # Malformed digest spelling (never normalized).
    upper_digest = json.loads(json.dumps(good))
    upper_digest["checkpoint"]["corrections_digest_hex"] = "A" * 64
    cases.append(json.dumps(upper_digest).encode())
    # Count disagrees with the array length.
    bad_count = json.loads(json.dumps(good))
    bad_count["checkpoint"]["correction_count"] = 2
    cases.append(json.dumps(bad_count).encode())
    # Extra field on a correction.
    extra_item = json.loads(json.dumps(good))
    extra_item["corrections"][0]["payload"] = {}
    cases.append(json.dumps(extra_item).encode())
    # Missing correction field.
    missing_field = json.loads(json.dumps(good))
    del missing_field["corrections"][0]["reason"]
    cases.append(json.dumps(missing_field).encode())
    # Naive/non-UTC timestamp.
    naive_time = json.loads(json.dumps(good))
    naive_time["corrections"][0]["created_at"] = "2026-01-02T03:04:05"
    cases.append(json.dumps(naive_time).encode())
    # Blank identifier and reason.
    blank_id = json.loads(json.dumps(good))
    blank_id["corrections"][0]["id"] = ""
    cases.append(json.dumps(blank_id).encode())
    blank_reason = json.loads(json.dumps(good))
    blank_reason["corrections"][0]["reason"] = "   "
    cases.append(json.dumps(blank_reason).encode())

    for raw in cases:
        resp = _post_raw(client, raw)
        assert resp.status_code == 422, raw
        assert resp.json()["error"]["code"] == "validation_error"

    # Every structural failure wrote nothing and no raw material landed.
    assert db_session.execute(select(CspImportRecord)).scalars().all() == []
    assert db_session.execute(select(AuditEvent)).scalars().all() == []
    assert db_session.execute(select(ClaimSupersession)).scalars().all() == []


def test_structural_failure_never_resolves_local_resources(client, db_session):
    # No actors/claims/supersessions exist, yet a structurally valid import
    # of fabricated identifiers still succeeds: the decision is body-only.
    payload = _body([_correction()])
    resp = client.post(URL, json=payload)
    assert resp.status_code == 201, resp.text
    assert (
        db_session.execute(select(ClaimSupersession)).scalars().all() == []
    )
