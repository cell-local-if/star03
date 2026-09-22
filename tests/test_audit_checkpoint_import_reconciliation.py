"""Tests for the read-only checkpoint-import reconciliation endpoint.

Covers GET /v1/audit-events/checkpoint-imports/{import_id}/reconciliation.
The receipt is read by ``import_id``; an unknown id is the existing
``404 audit_checkpoint_import_not_found``. Any (or repeated) query
parameter is a 422 validation_error before the receipt lookup. The success
body is exactly ``{"import_id", "local_checkpoint", "matches"}``:
``local_checkpoint`` is the four-field checkpoint computed under the
existing unfiltered ``GET /v1/audit-events/checkpoint`` rules (same fixed
version, algorithm, stable order, UTC representation, and canonical digest
semantics), and ``matches`` is true only when the receipt's
``checkpoint_version``, ``event_count``, and ``events_digest_hex`` each
equal the corresponding local checkpoint field. The imported event array
is never read back or echoed, and no unpersisted material is consulted.
The route is strictly read-only -- success, empty-library, parameter
failures, and missing receipts all write no resource, receipt, or audit
event. All fixtures are deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance import ids
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import (
    Actor,
    Attestation,
    AttestationRevocation,
    AuditEvent,
    CheckpointImportRecord,
    Claim,
    Content,
    ContentRelation,
    EvidenceBundle,
)
from tests.helpers import create_actor
from tests.test_audit_checkpoint_imports import (
    CHECKPOINT_VERSION,
    URL as IMPORTS_URL,
    _digest_of,
    _offline_request,
    _served_request,
    _setup_events,
)

CHECKPOINT_URL = "/v1/audit-events/checkpoint"
RECONCILIATION_KEYS = {"import_id", "local_checkpoint", "matches"}
CHECKPOINT_KEYS = {
    "checkpoint_version",
    "digest_algorithm",
    "event_count",
    "events_digest_hex",
}
EMPTY_DIGEST = hashlib.sha256(b"[]").hexdigest()
_DOMAIN_MODELS = (
    Actor,
    Content,
    Claim,
    EvidenceBundle,
    Attestation,
    AttestationRevocation,
    ContentRelation,
    CheckpointImportRecord,
    AuditEvent,
)


def _reconciliation_url(import_id: str) -> str:
    return f"{IMPORTS_URL}/{import_id}/reconciliation"


def _register(client, request: dict) -> dict:
    resp = client.post(IMPORTS_URL, json=request)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _served_checkpoint(client) -> dict:
    resp = client.get(CHECKPOINT_URL)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _insert_receipt(
    db_session,
    checkpoint_version: str,
    event_count: int,
    events_digest_hex: str,
) -> str:
    """Persist a receipt row directly, exactly as the import route stores it.

    The import route appends its own audit event, so a receipt matching the
    current local checkpoint can only be staged by writing the same
    identity triple the route would have persisted.
    """
    import_id = ids.checkpoint_import_id(
        checkpoint_version, event_count, events_digest_hex
    )
    db_session.add(
        CheckpointImportRecord(
            id=import_id,
            checkpoint_version=checkpoint_version,
            event_count=event_count,
            events_digest_hex=events_digest_hex,
        )
    )
    db_session.commit()
    return import_id


def _counts(db_session):
    return {
        model: db_session.execute(
            select(func.count()).select_from(model)
        ).scalar_one()
        for model in _DOMAIN_MODELS
    }


# --- Match: receipt identity equals the current local checkpoint ----------------


def test_matching_receipt_is_true_with_the_served_local_checkpoint(
    client, db_session
):
    _setup_events(client)
    checkpoint = _served_checkpoint(client)
    import_id = _insert_receipt(
        db_session,
        checkpoint["checkpoint_version"],
        checkpoint["event_count"],
        checkpoint["events_digest_hex"],
    )

    resp = client.get(_reconciliation_url(import_id))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == RECONCILIATION_KEYS
    assert set(body["local_checkpoint"]) == CHECKPOINT_KEYS
    assert body == {
        "import_id": import_id,
        "local_checkpoint": checkpoint,
        "matches": True,
    }


def test_empty_library_receipt_for_the_empty_sequence_matches(
    client, db_session
):
    # The empty local sequence checkpoints to (version, 0, sha256("[]")); a
    # receipt with exactly that identity matches in an otherwise empty
    # database.
    import_id = _insert_receipt(db_session, CHECKPOINT_VERSION, 0, EMPTY_DIGEST)

    body = client.get(_reconciliation_url(import_id)).json()
    assert body == {
        "import_id": import_id,
        "local_checkpoint": {
            "checkpoint_version": CHECKPOINT_VERSION,
            "digest_algorithm": "sha256",
            "event_count": 0,
            "events_digest_hex": EMPTY_DIGEST,
        },
        "matches": True,
    }


def test_local_checkpoint_matches_the_checkpoint_route_exactly(
    client, db_session
):
    _setup_events(client)
    import_id = _insert_receipt(db_session, CHECKPOINT_VERSION, 0, EMPTY_DIGEST)

    body = client.get(_reconciliation_url(import_id)).json()
    assert body["local_checkpoint"] == _served_checkpoint(client)
    assert body["matches"] is False


# --- Change: local state moves on after the receipt's checkpoint ----------------


def test_new_local_events_turn_a_match_into_a_mismatch(client, db_session):
    checkpoint = _served_checkpoint(client)
    import_id = _insert_receipt(
        db_session,
        checkpoint["checkpoint_version"],
        checkpoint["event_count"],
        checkpoint["events_digest_hex"],
    )
    assert client.get(_reconciliation_url(import_id)).json()["matches"] is True

    # A new audit event changes the local sequence; the receipt no longer
    # matches, and the response reports the new current checkpoint.
    create_actor(client, actor_id="org-later")
    body = client.get(_reconciliation_url(import_id)).json()
    assert body["matches"] is False
    assert body["local_checkpoint"] == _served_checkpoint(client)
    assert body["local_checkpoint"]["event_count"] == checkpoint["event_count"] + 1
    assert (
        body["local_checkpoint"]["events_digest_hex"]
        != checkpoint["events_digest_hex"]
    )


def test_imported_served_checkpoint_no_longer_matches_after_its_own_import(
    client,
):
    # Importing appends the import's own audit event, so the receipt's
    # checkpoint is already superseded by the current local sequence.
    _setup_events(client)
    receipt = _register(client, _served_request(client))

    body = client.get(_reconciliation_url(receipt["id"])).json()
    assert body["import_id"] == receipt["id"]
    assert body["matches"] is False
    assert body["local_checkpoint"] == _served_checkpoint(client)
    assert body["local_checkpoint"]["event_count"] == receipt["event_count"] + 1


def test_offline_receipt_never_matches_local_state(client):
    # A wholly fabricated checkpoint against unknown ids cannot match the
    # local sequence, before or after local events exist.
    receipt = _register(client, _offline_request())
    body = client.get(_reconciliation_url(receipt["id"])).json()
    assert body["matches"] is False
    assert body["local_checkpoint"] == _served_checkpoint(client)


# --- Matches requires all three identity fields ---------------------------------


def test_each_identity_field_must_equal_the_local_checkpoint(
    client, db_session
):
    checkpoint = _served_checkpoint(client)
    other_digest = ("0" if EMPTY_DIGEST[0] != "0" else "1") + EMPTY_DIGEST[1:]
    variants = {
        "version": ("other-checkpoint-version", 0, EMPTY_DIGEST),
        "count": (CHECKPOINT_VERSION, 1, EMPTY_DIGEST),
        "digest": (CHECKPOINT_VERSION, 0, other_digest),
    }
    for name, (version, count, digest) in variants.items():
        import_id = _insert_receipt(db_session, version, count, digest)
        body = client.get(_reconciliation_url(import_id)).json()
        assert body["matches"] is False, name
        assert body["local_checkpoint"] == checkpoint


# --- Determinism -----------------------------------------------------------------


def test_reconciliation_is_deterministic_across_restart(file_client, tmp_db_url):
    receipt = _register(file_client, _offline_request())
    expected = file_client.get(_reconciliation_url(receipt["id"]))
    assert expected.status_code == 200, expected.text

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        resp = client.get(_reconciliation_url(receipt["id"]))
        assert resp.status_code == 200, resp.text
        assert resp.json() == expected.json()


# --- Missing receipt --------------------------------------------------------------


def test_unknown_import_id_is_an_explicit_specific_404(client, db_session):
    unknown = "aci_" + "0" * 64
    resp = client.get(_reconciliation_url(unknown))
    assert resp.status_code == 404, resp.text
    error = resp.json()["error"]
    assert error["code"] == "audit_checkpoint_import_not_found"
    assert error["details"]["import_id"] == unknown
    # The failed read wrote nothing.
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_unshaped_and_other_prefix_ids_are_404(client):
    for raw_id in ("aci_doesnotexist", "eir_" + "a" * 64, "not-an-id"):
        resp = client.get(_reconciliation_url(raw_id))
        assert resp.status_code == 404, raw_id
        assert (
            resp.json()["error"]["code"] == "audit_checkpoint_import_not_found"
        )


def test_404_even_when_receipts_and_events_exist(client):
    _register(client, _offline_request())
    resp = client.get(_reconciliation_url("aci_" + "f" * 64))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "audit_checkpoint_import_not_found"


# --- Query parameter boundary -------------------------------------------------------


def test_any_query_parameter_is_422(client):
    receipt = _register(client, _offline_request())
    base = _reconciliation_url(receipt["id"])
    for url in (
        f"{base}?limit=10",
        f"{base}?cursor=abc",
        f"{base}?event_type=actor.created",
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
    resp = client.get(_reconciliation_url("aci_doesnotexist") + "?anything=1")
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only and no-echo boundary -------------------------------------------------


def test_reconciliation_writes_nothing(client, db_session):
    _setup_events(client)
    checkpoint = _served_checkpoint(client)
    matching = _insert_receipt(
        db_session,
        checkpoint["checkpoint_version"],
        checkpoint["event_count"],
        checkpoint["events_digest_hex"],
    )
    offline = _register(client, _offline_request())

    before = _counts(db_session)

    # A matching read, a mismatching read, a 422, and a 404 all leave every
    # table (receipts and audit events included) untouched.
    assert client.get(_reconciliation_url(matching)).status_code == 200
    assert client.get(_reconciliation_url(offline["id"])).status_code == 200
    assert client.get(_reconciliation_url(matching) + "?x=1").status_code == 422
    assert client.get(_reconciliation_url("aci_" + "0" * 64)).status_code == 404

    db_session.expire_all()
    assert _counts(db_session) == before


def test_response_carries_no_events_or_raw_material(client, db_session):
    _setup_events(client)
    checkpoint = _served_checkpoint(client)
    import_id = _insert_receipt(
        db_session,
        checkpoint["checkpoint_version"],
        checkpoint["event_count"],
        checkpoint["events_digest_hex"],
    )

    resp = client.get(_reconciliation_url(import_id))
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == RECONCILIATION_KEYS
    assert set(body["local_checkpoint"]) == CHECKPOINT_KEYS
    serialized = json.dumps(body)
    for forbidden in (
        '"events"',
        '"event_type"',
        '"resource_id"',
        '"created_at"',
        '"received_at"',
        '"digest"',
    ):
        assert forbidden not in serialized
