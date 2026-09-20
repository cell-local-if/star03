"""Tests for the read-only exchange-import reconciliation endpoint.

Covers GET /v1/evidence-bundle-exchange-imports/{import_id}/reconciliation.
The receipt is read by ``import_id``; an unknown id is the existing 404
evidence_bundle_exchange_import_not_found. When the receipt exists, its
``evidence_bundle_id`` is the only lookup key for the local bundle (never a
digest or other reverse association). The success body is exactly
{"import_id", "local_available", "local_manifest_digest_hex", "matches"}:
with no local bundle under that id it is the receipt id, false, null, false
(still 200); with a local bundle, ``local_available`` is true and
``local_manifest_digest_hex`` is the bundle's current digest computed
exactly as on GET /v1/evidence-bundles/{evidence_bundle_id}/exchange/
manifest, and ``matches`` is true only on character-for-character equality
with the receipt's ``manifest_digest_hex``. Any or repeated query parameter
is 422 validation_error before the receipt lookup. The route echoes no
snapshot, raw signature, payload, or bytes, and is strictly read-only: it
writes no resource, record, or audit event. All fixtures are deterministic
and offline.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance.app import create_app
from provenance.config import Settings
from provenance.models import (
    Actor,
    Attestation,
    AttestationRevocation,
    AuditEvent,
    Claim,
    Content,
    ContentRelation,
    EvidenceBundle,
    ExchangeImportRecord,
)
from tests.helpers import SEED_A
from tests.test_evidence_bundle_exchange import (
    _create_attestation,
    _setup_bundle,
)
from tests.test_evidence_bundle_exchange_imports import (
    _import_request,
    _served_import_request,
)
from tests.test_exchange_manifest_verifications import _digest_of

URL = "/v1/evidence-bundle-exchange-imports"
RECONCILIATION_KEYS = {
    "import_id",
    "local_available",
    "local_manifest_digest_hex",
    "matches",
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_DOMAIN_MODELS = (
    Actor,
    Content,
    Claim,
    EvidenceBundle,
    Attestation,
    AttestationRevocation,
    ContentRelation,
    ExchangeImportRecord,
    AuditEvent,
)


def _reconcile_url(import_id: str) -> str:
    return f"{URL}/{import_id}/reconciliation"


def _manifest_url(bundle) -> str:
    return f"/v1/evidence-bundles/{bundle['id']}/exchange/manifest"


def _import(client, request: dict) -> dict:
    resp = client.post(URL, json=request)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _reconcile(client, import_id: str) -> dict:
    resp = client.get(_reconcile_url(import_id))
    assert resp.status_code == 200, resp.text
    return resp.json()


def _counts(db_session) -> dict:
    return {
        model: db_session.execute(
            select(func.count()).select_from(model)
        ).scalar_one()
        for model in _DOMAIN_MODELS
    }


# --- Missing receipt -----------------------------------------------------------


def test_unknown_import_id_is_an_explicit_specific_404(client, db_session):
    unknown = "eir_" + "0" * 64
    resp = client.get(_reconcile_url(unknown))
    assert resp.status_code == 404, resp.text
    error = resp.json()["error"]
    assert error["code"] == "evidence_bundle_exchange_import_not_found"
    assert error["details"]["import_id"] == unknown
    # The failed read wrote nothing.
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_unshaped_and_other_prefix_ids_are_404(client):
    for raw_id in ("eir_doesnotexist", "evb_" + "a" * 64, "not-an-id"):
        resp = client.get(_reconcile_url(raw_id))
        assert resp.status_code == 404, raw_id
        assert (
            resp.json()["error"]["code"]
            == "evidence_bundle_exchange_import_not_found"
        )


# --- Query parameter boundary ---------------------------------------------------


def test_rejects_any_query_parameter(client):
    receipt = _import(client, _import_request())
    base = _reconcile_url(receipt["id"])

    for url in (
        f"{base}?limit=10",
        f"{base}?cursor=abc",
        f"{base}?unknown=",
        f"{base}?a=1&b=2",
        # The same parameter repeated is also rejected rather than collapsed.
        f"{base}?a=1&a=2",
    ):
        resp = client.get(url)
        assert resp.status_code == 422, (url, resp.text)
        assert resp.json()["error"]["code"] == "validation_error"


def test_query_parameter_is_422_before_the_receipt_lookup(client):
    # Parameters are validated before any existence lookup: a malformed
    # request is a 422 even when the receipt id is also unknown.
    resp = client.get(f"{_reconcile_url('eir_' + '0' * 64)}?anything=1")
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- No local bundle -------------------------------------------------------------


def test_no_local_bundle_is_200_unavailable(client):
    # The offline package's identifiers are all unknown locally.
    receipt = _import(client, _import_request())

    body = _reconcile(client, receipt["id"])
    assert set(body) == RECONCILIATION_KEYS
    assert body == {
        "import_id": receipt["id"],
        "local_available": False,
        "local_manifest_digest_hex": None,
        "matches": False,
    }


def test_local_lookup_uses_only_the_receipt_bundle_id(client):
    # An offline receipt stays unavailable even after unrelated local
    # resources appear: no digest or other reverse association is consulted.
    receipt = _import(client, _import_request())
    _setup_bundle(client)

    body = _reconcile(client, receipt["id"])
    assert body == {
        "import_id": receipt["id"],
        "local_available": False,
        "local_manifest_digest_hex": None,
        "matches": False,
    }


# --- Local bundle present ---------------------------------------------------------


def test_local_bundle_with_matching_digest_matches(client):
    _, _, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    receipt = _import(client, _served_import_request(client, bundle))

    body = _reconcile(client, receipt["id"])
    assert set(body) == RECONCILIATION_KEYS
    assert body["import_id"] == receipt["id"]
    assert body["local_available"] is True
    # The current local digest is exactly what the manifest route serves.
    current = client.get(_manifest_url(bundle)).json()["manifest_digest_hex"]
    assert _HEX64.fullmatch(body["local_manifest_digest_hex"])
    assert body["local_manifest_digest_hex"] == current
    assert body["local_manifest_digest_hex"] == receipt["manifest_digest_hex"]
    assert body["matches"] is True


def test_local_bundle_with_diverged_digest_does_not_match(client):
    _, _, bundle = _setup_bundle(client)
    # Import a package for this exact bundle id whose snapshot (and hence
    # manifest digest) differs from the local one.
    request = _served_import_request(client, bundle)
    request["snapshot"]["content"]["title"] = "diverged 快照"
    request["manifest"]["manifest_digest_hex"] = _digest_of(request["snapshot"])
    receipt = _import(client, request)
    assert receipt["evidence_bundle_id"] == bundle["id"]

    body = _reconcile(client, receipt["id"])
    assert body["local_available"] is True
    current = client.get(_manifest_url(bundle)).json()["manifest_digest_hex"]
    assert body["local_manifest_digest_hex"] == current
    assert body["local_manifest_digest_hex"] != receipt["manifest_digest_hex"]
    assert body["matches"] is False


def test_local_change_after_import_flips_matches(client):
    _, _, bundle = _setup_bundle(client)
    receipt = _import(client, _served_import_request(client, bundle))
    assert _reconcile(client, receipt["id"])["matches"] is True

    # The digest is the current one: a new attestation on the bundle changes
    # the snapshot, so the receipt no longer matches.
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    body = _reconcile(client, receipt["id"])
    current = client.get(_manifest_url(bundle)).json()["manifest_digest_hex"]
    assert body["local_available"] is True
    assert body["local_manifest_digest_hex"] == current
    assert body["local_manifest_digest_hex"] != receipt["manifest_digest_hex"]
    assert body["matches"] is False


def test_repeated_reconciliation_is_stable(client):
    _, _, bundle = _setup_bundle(client)
    receipt = _import(client, _served_import_request(client, bundle))

    first = _reconcile(client, receipt["id"])
    second = _reconcile(client, receipt["id"])
    assert first == second


# --- Response boundary and read-only guarantees ------------------------------------


def test_response_carries_no_snapshot_or_raw_material(client):
    _, _, bundle = _setup_bundle(client)
    receipt = _import(client, _served_import_request(client, bundle))

    body = _reconcile(client, receipt["id"])
    # Exactly the four reconciliation members: no snapshot, manifest object,
    # raw signature, claim payload, content bytes, or evidence bytes.
    assert set(body) == RECONCILIATION_KEYS
    for forbidden in ("snapshot", "manifest", "signature", "payload", "data"):
        assert forbidden not in body


def test_reconciliation_writes_nothing(client, db_session):
    _, _, bundle = _setup_bundle(client)
    matching = _import(client, _served_import_request(client, bundle))
    offline = _import(client, _import_request())

    before = _counts(db_session)

    # A matching read, an unavailable read, a 404, and a 422 all leave every
    # table untouched.
    assert client.get(_reconcile_url(matching["id"])).status_code == 200
    assert client.get(_reconcile_url(offline["id"])).status_code == 200
    assert client.get(_reconcile_url("eir_" + "0" * 64)).status_code == 404
    assert (
        client.get(f"{_reconcile_url(matching['id'])}?x=1").status_code == 422
    )

    db_session.expire_all()
    assert _counts(db_session) == before


def test_reconciliation_is_deterministic_across_restart(file_client, tmp_db_url):
    _, _, bundle = _setup_bundle(file_client)
    receipt = _import(file_client, _served_import_request(file_client, bundle))
    expected = _reconcile(file_client, receipt["id"])
    assert expected["matches"] is True

    # A brand-new app/engine over the same file reproduces the
    # reconciliation byte-for-byte.
    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        assert _reconcile(client, receipt["id"]) == expected
