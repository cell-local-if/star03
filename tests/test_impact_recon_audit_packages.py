"""Tests for the read-only impact-recon audit checkpoint package export.

Covers GET /v1/impact-recon-audit-packages: the success body is exactly
{"checkpoint", "entries"} in that member order; ``checkpoint`` is exactly
{"checkpoint_version", "digest_algorithm", "entry_count",
"entries_digest_hex"} with the version pinned to "pir-audit-checkpoint-v1"
and the algorithm to "sha256"; ``entries`` lists the signed
exchange-import receipts in their stable creation order, each with
exactly the nine fields id/signer_subject/public_key/package_digest_hex/
signature_digest_hex/received_at plus the read-time reconciliation triple
local_available/local_package_digest_hex/matches (the imported package,
the raw signature, and every private key are never carried). The
composable filters signer_subject, public_key (the exact Base64
spelling), package_digest_hex, from/to (strict RFC 3339 UTC, inclusive),
and matches (only lowercase true/false) combine as logical AND; a blank,
repeated, or undeclared parameter (including limit/cursor), a malformed
timestamp or boolean, a from later than to, and any non-empty request
body are all 422 validation_error raised before any receipt or local
state is read. The checkpoint digest is independently recomputed from
the entries member of the same response under the checkpoint canonical
rules (array order kept, nested object keys sorted by Unicode code
point, compact separators, unescaped non-ASCII, UTF-8); an empty match
set returns "entries": [] with entry_count 0 and the deterministic
digest of the empty array; repeated reads and reads across a restart
return the same package for the same persisted state; and neither
successful nor failed exports write any receipt, resource, or audit
row. Non-GET methods are 405 method_not_allowed. All fixtures are
deterministic and offline.
"""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance import canonical
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import AuditEvent, ImpactReconExchangeImportRecord
from tests.test_impact_recon_exchange_imports import (
    POST_PATH,
    _empty_package,
    _request,
)
from tests.test_impact_recon_exchange_imports_list import (
    _stamp,
    _three_receipts,
)

PACKAGE_PATH = "/v1/impact-recon-audit-packages"
VERIFY_PATH = "/v1/impact-recon-audit-verifications"
RECON_PACKAGE_PATH = "/v1/impact-recon-package"

PACKAGE_VERSION = "pir-audit-checkpoint-v1"
DIGEST_ALGORITHM = "sha256"

PACKAGE_FIELDS = ["checkpoint", "entries"]
CHECKPOINT_FIELDS = [
    "checkpoint_version",
    "digest_algorithm",
    "entry_count",
    "entries_digest_hex",
]
ENTRY_FIELDS = [
    "id",
    "signer_subject",
    "public_key",
    "package_digest_hex",
    "signature_digest_hex",
    "received_at",
    "local_available",
    "local_package_digest_hex",
    "matches",
]
EMPTY_ARRAY_DIGEST = hashlib.sha256(b"[]").hexdigest()


def _package(client, **params):
    return client.get(PACKAGE_PATH, params=params)


def _local_package_digest(client) -> str:
    """Independently digest the currently served unfiltered recon package."""
    resp = client.get(RECON_PACKAGE_PATH)
    assert resp.status_code == 200, resp.text
    return canonical.impact_recon_package_digest_hex(resp.json())


def _receipt_count(db_session) -> int:
    return db_session.scalar(
        select(func.count()).select_from(ImpactReconExchangeImportRecord)
    )


def _audit_count(db_session) -> int:
    return db_session.scalar(select(func.count()).select_from(AuditEvent))


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- Baseline export ------------------------------------------------------------


def test_empty_state_returns_empty_entries_and_empty_array_digest(client):
    resp = _package(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert list(body) == PACKAGE_FIELDS
    checkpoint = body["checkpoint"]
    assert list(checkpoint) == CHECKPOINT_FIELDS
    assert checkpoint["checkpoint_version"] == PACKAGE_VERSION
    assert checkpoint["digest_algorithm"] == DIGEST_ALGORITHM
    assert checkpoint["entry_count"] == 0
    assert checkpoint["entries_digest_hex"] == EMPTY_ARRAY_DIGEST
    assert body["entries"] == []


def test_response_is_compact_json_with_exactly_one_trailing_newline(client):
    _three_receipts(client)
    resp = _package(client)
    assert resp.status_code == 200, resp.text
    raw = resp.content
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b", " not in raw
    assert b": " not in raw
    assert raw.startswith(b'{"checkpoint":{')


def test_entries_carry_the_nine_fields_in_order(client):
    r1, r2, r3 = _three_receipts(client)
    body = _package(client).json()
    assert body["checkpoint"]["entry_count"] == 3
    entries = body["entries"]
    # Stable creation order, one audit entry per receipt.
    assert [entry["id"] for entry in entries] == [r1["id"], r2["id"], r3["id"]]
    local_digest = _local_package_digest(client)
    for entry, receipt in zip(entries, (r1, r2, r3)):
        assert list(entry) == ENTRY_FIELDS
        assert entry["signer_subject"] == receipt["signer_subject"]
        assert entry["public_key"] == receipt["public_key"]
        assert (
            entry["package_digest_hex"] == receipt["package_digest_hex"]
        )
        assert (
            entry["signature_digest_hex"] == receipt["signature_digest_hex"]
        )
        assert entry["received_at"] == receipt["received_at"]
        # The reconciliation triple is computed at this read and is
        # identical in its first two members for every entry.
        assert entry["local_available"] is True
        assert entry["local_package_digest_hex"] == local_digest
        assert (
            entry["matches"]
            is (entry["package_digest_hex"] == local_digest)
        )
    # Only the last imported receipt matches the current local package.
    assert [entry["matches"] for entry in entries] == [False, False, True]
    # The imported package, the raw signature, and any private key are
    # never carried.
    for entry in entries:
        assert "package" not in entry
        assert "signature" not in entry
        assert "private_key" not in entry


def test_empty_local_state_still_yields_deterministic_entry_state(client):
    # A receipt for the empty local package: unavailable, yet matching.
    request, _ = _request(_empty_package(client))
    resp = client.post(POST_PATH, json=request)
    assert resp.status_code == 201, resp.text
    body = _package(client).json()
    (entry,) = body["entries"]
    assert entry["local_available"] is False
    assert entry["local_package_digest_hex"] == _local_package_digest(client)
    assert entry["matches"] is True


def test_checkpoint_digest_binds_exactly_the_served_entries(client):
    _three_receipts(client)
    body = _package(client).json()
    recomputed = canonical.impact_recon_audit_entries_digest_hex(
        body["entries"]
    )
    assert body["checkpoint"]["entries_digest_hex"] == recomputed
    assert body["checkpoint"]["entry_count"] == len(body["entries"])


def test_export_verifies_offline(client):
    _three_receipts(client)
    package = _package(client).json()
    resp = client.post(VERIFY_PATH, json=package)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_repeated_reads_return_the_same_package(client):
    _three_receipts(client)
    assert _package(client).content == _package(client).content


def test_restart_returns_the_same_export(client, tmp_db_url):
    producer = create_app(Settings(database_url=tmp_db_url))
    with TestClient(producer) as first:
        _three_receipts(first)
        before = first.get(PACKAGE_PATH)
        assert before.status_code == 200, before.text
    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as second:
        after = second.get(PACKAGE_PATH)
        assert after.status_code == 200, after.text
        assert after.content == before.content


# --- Filters ---------------------------------------------------------------------


def test_filter_by_signer_subject(client):
    _, _, r3 = _three_receipts(client)
    body = _package(client, signer_subject="other-subject").json()
    assert [entry["id"] for entry in body["entries"]] == [r3["id"]]
    assert body["checkpoint"]["entry_count"] == 1
    body = _package(client, signer_subject="remote-system").json()
    assert body["checkpoint"]["entry_count"] == 2
    # Exact, case-sensitive match: a different spelling names nothing.
    body = _package(client, signer_subject="Remote-System").json()
    assert body["entries"] == []
    assert body["checkpoint"]["entries_digest_hex"] == EMPTY_ARRAY_DIGEST


def test_filter_by_public_key_exact_spelling(client):
    r1, r2, r3 = _three_receipts(client)
    # r1 and r2 share the signing key; r3 was signed under a second key.
    body = _package(client, public_key=r1["public_key"]).json()
    assert [entry["id"] for entry in body["entries"]] == [r1["id"], r2["id"]]
    body = _package(client, public_key=r3["public_key"]).json()
    assert [entry["id"] for entry in body["entries"]] == [r3["id"]]
    # The filter value is compared by exact spelling, never decoded or
    # normalized: non-canonical or malformed spellings match nothing.
    assert (
        _package(client, public_key=r1["public_key"].rstrip("="))
        .json()["entries"]
        == []
    )
    assert (
        _package(client, public_key=r1["public_key"].swapcase())
        .json()["entries"]
        == []
    )
    assert _package(client, public_key="not-base64!!").json()["entries"] == []


def test_filter_by_package_digest_hex(client):
    r1, _, _ = _three_receipts(client)
    body = _package(client, package_digest_hex=r1["package_digest_hex"]).json()
    assert [entry["id"] for entry in body["entries"]] == [r1["id"]]
    body = _package(client, package_digest_hex="0" * 64).json()
    assert body["entries"] == []


def test_time_filters_are_inclusive(client, db_session):
    from datetime import datetime, timezone

    r1, r2, r3 = _three_receipts(client)
    _stamp(
        db_session,
        {
            r1["id"]: datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc),
            r2["id"]: datetime(2026, 2, 1, 11, 0, 0, tzinfo=timezone.utc),
            r3["id"]: datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc),
        },
    )
    body = _package(
        client, **{"from": "2026-02-01T11:00:00Z", "to": "2026-02-01T11:00:00Z"}
    ).json()
    assert [entry["id"] for entry in body["entries"]] == [r2["id"]]
    body = _package(client, **{"from": "2026-02-01T11:00:00Z"}).json()
    assert [entry["id"] for entry in body["entries"]] == [r2["id"], r3["id"]]
    body = _package(client, to="2026-02-01T11:00:00Z").json()
    assert [entry["id"] for entry in body["entries"]] == [r1["id"], r2["id"]]
    # The "+00:00" spelling is the same instant as "Z".
    body = _package(client, **{"from": "2026-02-01T11:00:00+00:00"}).json()
    assert [entry["id"] for entry in body["entries"]] == [r2["id"], r3["id"]]
    body = _package(
        client, **{"from": "2027-01-01T00:00:00Z", "to": "2027-01-02T00:00:00Z"}
    ).json()
    assert body["entries"] == []
    assert body["checkpoint"]["entry_count"] == 0
    assert body["checkpoint"]["entries_digest_hex"] == EMPTY_ARRAY_DIGEST


def test_from_later_than_to_is_422(client):
    resp = _package(
        client, **{"from": "2026-02-02T00:00:00Z", "to": "2026-02-01T00:00:00Z"}
    )
    _assert_validation_error(resp)


def test_filter_by_matches(client):
    r1, r2, r3 = _three_receipts(client)
    body = _package(client, matches="true").json()
    assert [entry["id"] for entry in body["entries"]] == [r3["id"]]
    assert body["entries"][0]["matches"] is True
    body = _package(client, matches="false").json()
    assert [entry["id"] for entry in body["entries"]] == [r1["id"], r2["id"]]
    assert all(entry["matches"] is False for entry in body["entries"])


def test_filters_combine_as_logical_and(client):
    r1, _, r3 = _three_receipts(client)
    body = _package(
        client, signer_subject="other-subject", matches="true"
    ).json()
    assert [entry["id"] for entry in body["entries"]] == [r3["id"]]
    body = _package(
        client,
        signer_subject="remote-system",
        package_digest_hex=r1["package_digest_hex"],
        matches="false",
    ).json()
    assert [entry["id"] for entry in body["entries"]] == [r1["id"]]
    # One contradictory conjunct empties the set.
    body = _package(
        client,
        signer_subject="remote-system",
        package_digest_hex=r3["package_digest_hex"],
    ).json()
    assert body["entries"] == []
    assert body["checkpoint"]["entries_digest_hex"] == EMPTY_ARRAY_DIGEST


# --- Validation ------------------------------------------------------------------


@pytest.mark.parametrize("body", [b"{}", b" ", b"x", b"\x00"])
def test_non_empty_body_is_422(client, body):
    resp = client.request("GET", PACKAGE_PATH, content=body)
    _assert_validation_error(resp)


def test_repeated_parameter_is_422(client):
    resp = client.get(
        f"{PACKAGE_PATH}?signer_subject=a&signer_subject=b"
    )
    _assert_validation_error(resp)
    resp = client.get(f"{PACKAGE_PATH}?matches=true&matches=false")
    _assert_validation_error(resp)


@pytest.mark.parametrize(
    "suffix",
    [
        "unknown=1",
        "limit=10",
        "cursor=abc",
        "signature_version=provenance-impact-recon-exchange-v1",
        "local_available=true",
    ],
)
def test_undeclared_parameter_is_422(client, suffix):
    _assert_validation_error(client.get(f"{PACKAGE_PATH}?{suffix}"))


@pytest.mark.parametrize(
    "param", ["signer_subject", "public_key", "package_digest_hex", "from", "to", "matches"]
)
def test_blank_parameter_is_422(client, param):
    _assert_validation_error(_package(client, **{param: ""}))
    _assert_validation_error(_package(client, **{param: "  "}))


@pytest.mark.parametrize(
    "value",
    [
        "2026-01-01",
        "2026-01-01T00:00:00",
        "2026-01-01 00:00:00Z",
        "2026-01-01T00:00Z",
        "2026-01-01T00:00:00z",
        "2026-01-01T00:00:00+01:00",
        "2026-13-01T00:00:00Z",
    ],
)
def test_malformed_time_is_422(client, value):
    _assert_validation_error(_package(client, **{"from": value}))
    _assert_validation_error(_package(client, to=value))


@pytest.mark.parametrize("value", ["True", "FALSE", "1", "0", "yes", "true "])
def test_bad_boolean_is_422(client, value):
    _assert_validation_error(_package(client, matches=value))


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_non_get_methods_are_405(client, method):
    resp = getattr(client, method)(PACKAGE_PATH)
    assert resp.status_code == 405, resp.text
    assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Read-only guarantees ---------------------------------------------------------


def test_successful_and_failed_exports_write_nothing(client, db_session):
    _three_receipts(client)
    db_session.expire_all()
    receipts_before = _receipt_count(db_session)
    audits_before = _audit_count(db_session)
    assert _package(client).status_code == 200
    assert _package(client, matches="true").status_code == 200
    _assert_validation_error(_package(client, matches="yes"))
    _assert_validation_error(client.get(f"{PACKAGE_PATH}?limit=1"))
    _assert_validation_error(client.request("GET", PACKAGE_PATH, content=b"{}"))
    db_session.expire_all()
    assert _receipt_count(db_session) == receipts_before
    assert _audit_count(db_session) == audits_before
