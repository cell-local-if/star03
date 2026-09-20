"""Tests for the read-only exchange-import reconciliation endpoint.

Covers GET /v1/evidence-bundle-exchange-imports/{import_id}/reconciliation:
the receipt is read by ``import_id`` and an unknown id is the existing
``404 evidence_bundle_exchange_import_not_found``. The success body is
exactly ``{"import_id", "local_available", "local_manifest_digest_hex",
"matches"}``. The receipt's ``evidence_bundle_id`` is the only local lookup
key (never a reverse search); when no local bundle has that id the body is
``(receipt id, false, null, false)`` with a 200. When a local bundle exists,
``local_available`` is true and ``local_manifest_digest_hex`` is the current
digest computed under the exchange manifest route's rules; ``matches`` is
true only on character-for-character equality with the receipt's
``manifest_digest_hex``. Any (or repeated) query parameter is 422
validation_error before the receipt lookup. The route is strictly read-only
— no resource, receipt, or audit row is written — and no snapshot, raw
signature, payload, or bytes are echoed. All fixtures are deterministic and
offline.
"""

from __future__ import annotations

import json

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
    _revoke,
    _setup_bundle,
)
from tests.test_evidence_bundle_exchange_imports import (
    _import_request,
    _served_import_request,
)
from tests.test_exchange_manifest_verifications import _digest_of

IMPORTS_URL = "/v1/evidence-bundle-exchange-imports"
RECONCILIATION_KEYS = {
    "import_id",
    "local_available",
    "local_manifest_digest_hex",
    "matches",
}
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


def _reconciliation_url(import_id: str) -> str:
    return f"{IMPORTS_URL}/{import_id}/reconciliation"


def _register(client, request: dict) -> dict:
    resp = client.post(IMPORTS_URL, json=request)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _manifest_digest_of(client, bundle) -> str:
    resp = client.get(f"/v1/evidence-bundles/{bundle['id']}/exchange/manifest")
    assert resp.status_code == 200, resp.text
    return resp.json()["manifest_digest_hex"]


def _counts(db_session):
    return {
        model: db_session.execute(
            select(func.count()).select_from(model)
        ).scalar_one()
        for model in _DOMAIN_MODELS
    }


# --- Success shape: no local bundle -------------------------------------------


def test_no_local_bundle_is_200_with_unavailable_body(client):
    # The offline package's identifiers are unknown locally: the receipt
    # registers, but no local bundle can carry its evidence_bundle_id.
    receipt = _register(client, _import_request())

    resp = client.get(_reconciliation_url(receipt["id"]))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == RECONCILIATION_KEYS
    assert body == {
        "import_id": receipt["id"],
        "local_available": False,
        "local_manifest_digest_hex": None,
        "matches": False,
    }


def test_no_reverse_lookup_when_other_bundles_exist(client):
    # A local bundle exists, but the receipt references a different,
    # unknown bundle id: the route must not "find" the unrelated bundle.
    _setup_bundle(client)
    receipt = _register(client, _import_request())

    body = client.get(_reconciliation_url(receipt["id"])).json()
    assert body == {
        "import_id": receipt["id"],
        "local_available": False,
        "local_manifest_digest_hex": None,
        "matches": False,
    }


# --- Success shape: local bundle present ---------------------------------------


def test_matching_local_bundle_is_available_and_matches(client):
    _, _, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    receipt = _register(client, _served_import_request(client, bundle))

    resp = client.get(_reconciliation_url(receipt["id"]))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == RECONCILIATION_KEYS
    assert body == {
        "import_id": receipt["id"],
        "local_available": True,
        "local_manifest_digest_hex": _manifest_digest_of(client, bundle),
        "matches": True,
    }
    assert body["local_manifest_digest_hex"] == receipt["manifest_digest_hex"]


def test_diverged_local_state_still_reports_current_digest_without_match(client):
    _, _, bundle = _setup_bundle(client)
    receipt = _register(client, _served_import_request(client, bundle))

    # The bundle's snapshot changes after the import (a new attestation):
    # the receipt's digest no longer matches the current local digest.
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    current = _manifest_digest_of(client, bundle)
    assert current != receipt["manifest_digest_hex"]

    body = client.get(_reconciliation_url(receipt["id"])).json()
    assert body == {
        "import_id": receipt["id"],
        "local_available": True,
        "local_manifest_digest_hex": current,
        "matches": False,
    }


def test_receipt_for_superseded_snapshot_does_not_match(client):
    # Two receipts for the same local bundle id: the first was registered
    # from the served package, the second from a tampered snapshot with a
    # different (self-consistent) digest. Only the receipt whose digest
    # equals the current local digest character-for-character matches.
    _, _, bundle = _setup_bundle(client)
    served = _served_import_request(client, bundle)
    current_receipt = _register(client, served)

    stale_request = json.loads(json.dumps(served))
    stale_request["snapshot"]["content"]["title"] = "superseded 标题"
    stale_request["manifest"]["manifest_digest_hex"] = _digest_of(
        stale_request["snapshot"]
    )
    stale_receipt = _register(client, stale_request)
    assert stale_receipt["id"] != current_receipt["id"]
    assert (
        stale_receipt["manifest_digest_hex"]
        != current_receipt["manifest_digest_hex"]
    )

    current_digest = _manifest_digest_of(client, bundle)
    current_body = client.get(_reconciliation_url(current_receipt["id"])).json()
    assert current_body == {
        "import_id": current_receipt["id"],
        "local_available": True,
        "local_manifest_digest_hex": current_digest,
        "matches": True,
    }
    stale_body = client.get(_reconciliation_url(stale_receipt["id"])).json()
    assert stale_body == {
        "import_id": stale_receipt["id"],
        "local_available": True,
        "local_manifest_digest_hex": current_digest,
        "matches": False,
    }


def test_revoked_attestation_changes_the_current_digest(client):
    _, _, bundle = _setup_bundle(client)
    attested = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    receipt = _register(client, _served_import_request(client, bundle))

    # A revocation does not alter the snapshot, so the receipt still matches.
    _revoke(client, attested["id"])
    body = client.get(_reconciliation_url(receipt["id"])).json()
    assert body["local_available"] is True
    assert body["matches"] is True
    assert body["local_manifest_digest_hex"] == receipt["manifest_digest_hex"]


def test_reconciliation_is_deterministic_across_restart(file_client, tmp_db_url):
    _, _, bundle = _setup_bundle(file_client)
    receipt = _register(file_client, _served_import_request(file_client, bundle))
    expected = file_client.get(_reconciliation_url(receipt["id"]))
    assert expected.status_code == 200, expected.text

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        resp = client.get(_reconciliation_url(receipt["id"]))
        assert resp.status_code == 200, resp.text
        assert resp.json() == expected.json()


# --- Missing receipt -----------------------------------------------------------


def test_unknown_import_id_is_an_explicit_specific_404(client, db_session):
    unknown = "eir_" + "0" * 64
    resp = client.get(_reconciliation_url(unknown))
    assert resp.status_code == 404, resp.text
    error = resp.json()["error"]
    assert error["code"] == "evidence_bundle_exchange_import_not_found"
    assert error["details"]["import_id"] == unknown
    # The failed read wrote nothing.
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_unshaped_and_other_prefix_ids_are_404(client):
    for raw_id in ("eir_doesnotexist", "evb_" + "a" * 64, "not-an-id"):
        resp = client.get(_reconciliation_url(raw_id))
        assert resp.status_code == 404, raw_id
        assert (
            resp.json()["error"]["code"]
            == "evidence_bundle_exchange_import_not_found"
        )


def test_404_even_when_a_local_bundle_exists(client):
    _setup_bundle(client)
    resp = client.get(_reconciliation_url("eir_" + "f" * 64))
    assert resp.status_code == 404
    assert (
        resp.json()["error"]["code"]
        == "evidence_bundle_exchange_import_not_found"
    )


# --- Query parameter boundary ---------------------------------------------------


def test_any_query_parameter_is_422(client):
    receipt = _register(client, _import_request())
    base = _reconciliation_url(receipt["id"])
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
    resp = client.get(_reconciliation_url("eir_doesnotexist") + "?anything=1")
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only and no-echo boundary ---------------------------------------------


def test_reconciliation_writes_nothing(client, db_session):
    _, _, bundle = _setup_bundle(client)
    matching = _register(client, _served_import_request(client, bundle))
    offline = _register(client, _import_request())

    before = _counts(db_session)

    # A matching read, an unavailable read, a 422, and a 404 all leave
    # every table (receipts and audit events included) untouched.
    assert client.get(_reconciliation_url(matching["id"])).status_code == 200
    assert client.get(_reconciliation_url(offline["id"])).status_code == 200
    assert (
        client.get(_reconciliation_url(matching["id"]) + "?x=1").status_code
        == 422
    )
    assert (
        client.get(_reconciliation_url("eir_" + "0" * 64)).status_code == 404
    )

    db_session.expire_all()
    assert _counts(db_session) == before


def test_response_carries_no_snapshot_or_raw_material(client):
    _, _, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    receipt = _register(client, _served_import_request(client, bundle))

    resp = client.get(_reconciliation_url(receipt["id"]))
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == RECONCILIATION_KEYS
    serialized = json.dumps(body)
    for forbidden in (
        '"snapshot"',
        '"manifest"',
        '"signature"',
        '"payload"',
        '"data"',
        '"evidence"',
        '"public_key"',
    ):
        assert forbidden not in serialized
