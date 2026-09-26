"""Tests for the signed exchange-import receipt reconciliation endpoint.

Covers GET /v1/impact-recon-exchange-imports/{import_id}/recon: the
success body is strictly ``{"import_id", "local_available",
"local_package_digest_hex", "matches"}`` in that member order, computed at
read time. ``local_available`` is true only when the current, complete,
unfiltered local impact-recon package carries at least one reconciliation
entry; ``local_package_digest_hex`` is the SHA-256 package digest of that
unfiltered package under the existing package canonical rules (the empty
state still yields a deterministic digest); ``matches`` is true only when
that current digest equals the receipt's ``package_digest_hex`` character
for character. An unknown id is always the
``404 impact_recon_exchange_import_not_found`` and changes nothing; any
(or repeated) query parameter and any non-empty body are
``422 validation_error`` before the lookup; non-GET methods are
``405 method_not_allowed``. The route is strictly read-only and never
echoes the package, the raw signature, or any private key. All fixtures
are deterministic and offline.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance import canonical
from provenance.models import (
    AuditEvent,
    ImpactImportRecord,
    ImpactReconExchangeImportRecord,
)
from tests.test_impact_recon_exchange_imports import (
    POST_PATH,
    _empty_package,
    _populated_package,
    _request,
)

RECON_PACKAGE_PATH = "/v1/impact-recon-package"
RECON_KEYS = [
    "import_id",
    "local_available",
    "local_package_digest_hex",
    "matches",
]
UNKNOWN_ID = "irx_" + "0" * 64


def _import(client, package: dict, **kwargs) -> dict:
    """Exchange-import ``package`` and return the receipt JSON."""
    request, _ = _request(package, **kwargs)
    resp = client.post(POST_PATH, json=request)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _recon(client, import_id: str):
    return client.get(f"{POST_PATH}/{import_id}/recon")


def _recon_url(import_id: str) -> str:
    return f"{POST_PATH}/{import_id}/recon"


def _current_package_digest(client) -> str:
    """Independently digest the currently served unfiltered package."""
    resp = client.get(RECON_PACKAGE_PATH)
    assert resp.status_code == 200, resp.text
    return canonical.impact_recon_package_digest_hex(resp.json())


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- Verdicts -------------------------------------------------------------------


def test_empty_state_recon_matches(client):
    receipt = _import(client, _empty_package(client))
    resp = _recon(client, receipt["id"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert list(body) == RECON_KEYS
    assert body["import_id"] == receipt["id"]
    # No local reconciliation entries exist: unavailable, but the empty
    # package still determines a deterministic digest.
    assert body["local_available"] is False
    assert re.fullmatch(r"[0-9a-f]{64}", body["local_package_digest_hex"])
    assert body["local_package_digest_hex"] == _current_package_digest(client)
    assert body["local_package_digest_hex"] == receipt["package_digest_hex"]
    assert body["matches"] is True


def test_populated_state_recon_matches(client):
    receipt = _import(client, _populated_package(client))
    body = _recon(client, receipt["id"]).json()
    assert body["local_available"] is True
    assert body["local_package_digest_hex"] == _current_package_digest(client)
    assert body["matches"] is True


def test_stale_receipt_does_not_match(client):
    receipt = _import(client, _empty_package(client))
    # Local recon state moves on after the receipt was registered.
    _populated_package(client)
    body = _recon(client, receipt["id"]).json()
    assert body["local_available"] is True
    assert body["local_package_digest_hex"] == _current_package_digest(client)
    assert body["local_package_digest_hex"] != receipt["package_digest_hex"]
    assert body["matches"] is False


def test_verdict_is_computed_at_read_time(client):
    stale = _import(client, _empty_package(client))
    current = _import(client, _populated_package(client))
    stale_body = _recon(client, stale["id"]).json()
    current_body = _recon(client, current["id"]).json()
    # Both reads see the same current digest; only the verdicts differ.
    assert stale_body["matches"] is False
    assert current_body["matches"] is True
    assert (
        stale_body["local_package_digest_hex"]
        == current_body["local_package_digest_hex"]
        == current["package_digest_hex"]
    )
    # Repeated reads of the same state are identical.
    assert _recon(client, stale["id"]).json() == stale_body


def test_response_is_compact_utf8_json_with_one_newline(client):
    receipt = _import(client, _empty_package(client))
    resp = _recon(client, receipt["id"])
    raw = resp.content
    assert resp.headers["content-type"].startswith("application/json")
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    text = raw.decode("utf-8")[:-1]
    assert '", "' not in text and '": ' not in text
    assert list(resp.json()) == RECON_KEYS


def test_package_signature_and_keys_are_never_echoed(client):
    receipt = _import(client, _populated_package(client))
    resp = _recon(client, receipt["id"])
    text = resp.text
    assert set(resp.json()) == set(RECON_KEYS)
    assert receipt["public_key"] not in text
    assert receipt["signature_digest_hex"] not in text


# --- Missing receipts -------------------------------------------------------------


def test_unknown_import_id_is_404(client, db_session):
    resp = _recon(client, UNKNOWN_ID)
    assert resp.status_code == 404
    assert (
        resp.json()["error"]["code"] == "impact_recon_exchange_import_not_found"
    )
    # A missing receipt changes nothing.
    assert db_session.scalar(select(func.count()).select_from(AuditEvent)) == 0
    assert (
        db_session.scalar(
            select(func.count()).select_from(ImpactReconExchangeImportRecord)
        )
        == 0
    )


def test_unknown_import_id_is_404_even_with_local_state(client):
    _populated_package(client)
    resp = _recon(client, UNKNOWN_ID)
    assert resp.status_code == 404
    assert (
        resp.json()["error"]["code"] == "impact_recon_exchange_import_not_found"
    )


# --- Request boundary ---------------------------------------------------------------


@pytest.mark.parametrize("query", ["x=1", "matches=true", "limit=1", "x=1&x=2"])
def test_query_parameters_are_422_before_lookup(client, query):
    receipt = _import(client, _empty_package(client))
    _assert_validation_error(client.get(f"{_recon_url(receipt['id'])}?{query}"))
    # Parameter validation precedes the receipt lookup: an unknown id with
    # a parameter is a 422, never a 404.
    _assert_validation_error(client.get(f"{_recon_url(UNKNOWN_ID)}?{query}"))


@pytest.mark.parametrize(
    "body", [b"{}", b"null", b"[1]", b"not json", b"{", b" ", b"\n"]
)
def test_non_empty_body_is_422(client, body):
    receipt = _import(client, _empty_package(client))
    _assert_validation_error(
        client.request("GET", _recon_url(receipt["id"]), content=body)
    )
    # The body check precedes the receipt lookup as well.
    _assert_validation_error(
        client.request("GET", _recon_url(UNKNOWN_ID), content=body)
    )


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_non_get_methods_are_405(client, method):
    receipt = _import(client, _empty_package(client))
    resp = getattr(client, method)(_recon_url(receipt["id"]))
    assert resp.status_code == 405
    assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Read-only guarantees ------------------------------------------------------------


def test_recon_writes_nothing(client, db_session):
    receipt = _import(client, _empty_package(client))
    audits_before = db_session.scalar(select(func.count()).select_from(AuditEvent))
    receipts_before = db_session.scalar(
        select(func.count()).select_from(ImpactReconExchangeImportRecord)
    )
    impacts_before = db_session.scalar(
        select(func.count()).select_from(ImpactImportRecord)
    )
    assert _recon(client, receipt["id"]).status_code == 200
    assert _recon(client, receipt["id"]).status_code == 200
    assert _recon(client, UNKNOWN_ID).status_code == 404
    client.get(f"{_recon_url(receipt['id'])}?x=1")
    client.request("GET", _recon_url(receipt["id"]), content=b"{}")
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent))
        == audits_before
    )
    assert (
        db_session.scalar(
            select(func.count()).select_from(ImpactReconExchangeImportRecord)
        )
        == receipts_before
    )
    assert (
        db_session.scalar(select(func.count()).select_from(ImpactImportRecord))
        == impacts_before
    )
