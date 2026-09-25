"""Tests for the read-only impact-import reconciliation endpoint.

Covers GET /v1/impact-imports/{import_id}/recon: the receipt is read by
``import_id`` and an unknown id is the ``404 impact_import_not_found``. The
success body is strictly ``{"import_id", "local_checkpoint", "matches"}``;
``local_checkpoint`` is the four-field checkpoint computed over the
complete, unfiltered local revocation-impact set under the existing
``GET /v1/revocation-impact-package`` rules (fixed version/algorithm,
stable creation order, UTC representation, canonical SHA-256 digest), and
``matches`` is true only when the receipt's ``checkpoint_version``,
``impact_count``, and ``impacts_digest_hex`` all equal the local
checkpoint fields. Any (or repeated) query parameter is
``422 validation_error`` before the receipt lookup. The imported impacts
array is never read or echoed and no unpersisted material participates in
the verdict. The route is strictly read-only -- no resource, receipt, or
audit row is written -- on success, on an empty database, on parameter
failure, and for a missing receipt. All fixtures are deterministic and
offline.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

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
    ExchangeImportRecord,
    ImpactImportRecord,
)
from provenance.models import EVENT_REVOCATION_IMPACT_IMPORTED
from tests.helpers import create_actor
from tests.test_impact_imports import _offline_request
from tests.test_revocation_impacts import _world

IMPORTS_URL = "/v1/impact-imports"
PACKAGE_URL = "/v1/revocation-impact-package"
CHECKPOINT_VERSION = "provenance-revocation-impact-checkpoint-v1"
EMPTY_DIGEST = hashlib.sha256(b"[]").hexdigest()
RECON_KEYS = {"import_id", "local_checkpoint", "matches"}
CHECKPOINT_KEYS = {
    "checkpoint_version",
    "digest_algorithm",
    "impact_count",
    "impacts_digest_hex",
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
    CheckpointImportRecord,
    ImpactImportRecord,
    AuditEvent,
)


def _recon_url(import_id: str) -> str:
    return f"{IMPORTS_URL}/{import_id}/recon"


def _digest_of(impacts: list) -> str:
    """Independently canonicalize the impact wire array and digest it."""
    canonical = json.dumps(
        impacts,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _served_checkpoint(client) -> dict:
    resp = client.get(PACKAGE_URL)
    assert resp.status_code == 200, resp.text
    return resp.json()["checkpoint"]


def _insert_receipt(db_session, checkpoint_version, impact_count, digest_hex):
    """Persist a receipt directly, without writing an import audit event.

    The reconciliation reads persisted state alone; direct insertion keeps
    the surrounding state exactly controlled.
    """
    receipt_id = ids.impact_import_id(
        checkpoint_version, impact_count, digest_hex
    )
    db_session.add(
        ImpactImportRecord(
            id=receipt_id,
            checkpoint_version=checkpoint_version,
            impact_count=impact_count,
            impacts_digest_hex=digest_hex,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
    )
    db_session.commit()
    return receipt_id


def _matching_receipt(db_session, client) -> tuple[str, dict]:
    """A receipt whose identity equals the current local checkpoint."""
    checkpoint = _served_checkpoint(client)
    receipt_id = _insert_receipt(
        db_session,
        checkpoint["checkpoint_version"],
        checkpoint["impact_count"],
        checkpoint["impacts_digest_hex"],
    )
    return receipt_id, checkpoint


def _counts(db_session):
    return {
        model: db_session.execute(
            select(func.count()).select_from(model)
        ).scalar_one()
        for model in _DOMAIN_MODELS
    }


# --- Success shape and semantics ------------------------------------------------


def test_empty_library_match(client, db_session):
    # Empty local impact set + receipt for the empty checkpoint: matches
    # true, and the local checkpoint is exactly the served empty checkpoint.
    receipt_id = _insert_receipt(db_session, CHECKPOINT_VERSION, 0, EMPTY_DIGEST)

    resp = client.get(_recon_url(receipt_id))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == RECON_KEYS
    assert set(body["local_checkpoint"]) == CHECKPOINT_KEYS
    assert body == {
        "import_id": receipt_id,
        "local_checkpoint": {
            "checkpoint_version": CHECKPOINT_VERSION,
            "digest_algorithm": "sha256",
            "impact_count": 0,
            "impacts_digest_hex": EMPTY_DIGEST,
        },
        "matches": True,
    }


def test_matching_receipt_reports_current_checkpoint_and_matches(
    client, db_session
):
    _world(client)
    receipt_id, checkpoint = _matching_receipt(db_session, client)

    resp = client.get(_recon_url(receipt_id))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == RECON_KEYS
    assert body["import_id"] == receipt_id
    assert body["matches"] is True
    # The local checkpoint is exactly the unfiltered served checkpoint.
    assert body["local_checkpoint"] == checkpoint
    assert body["local_checkpoint"] == _served_checkpoint(client)
    # The digest is independently reproducible from the package impacts.
    served = client.get(PACKAGE_URL).json()["impacts"]
    assert body["local_checkpoint"]["impact_count"] == len(served)
    assert body["local_checkpoint"]["impacts_digest_hex"] == _digest_of(served)


def test_imported_served_package_matches_immediately(client):
    # Unlike an audit-trail import, registering an impact checkpoint writes
    # no revocation: the receipt of the just-served package still matches.
    _world(client)
    package = client.get(PACKAGE_URL).json()
    resp = client.post(IMPORTS_URL, json=package)
    assert resp.status_code == 201, resp.text
    receipt = resp.json()

    body = client.get(_recon_url(receipt["id"])).json()
    assert body == {
        "import_id": receipt["id"],
        "local_checkpoint": package["checkpoint"],
        "matches": True,
    }


def test_local_impacts_growing_after_registration_flips_match_to_false(
    client, db_session
):
    _world(client)
    receipt_id, checkpoint = _matching_receipt(db_session, client)
    assert client.get(_recon_url(receipt_id)).json()["matches"] is True

    # The local impact set changes after registration: the receipt no longer
    # matches, but the current checkpoint is still reported in full.
    create_actor(client, actor_id="org-late", name="Late", type="organization")
    resp = client.post(
        "/v1/contents",
        json={
            "digest_algorithm": "sha256",
            "digest_hex": hashlib.sha256(b"late-content").hexdigest(),
            "media_type": "image/png",
            "actor_id": "org-late",
        },
    )
    assert resp.status_code == 201, resp.text

    current = _served_checkpoint(client)
    assert current["impact_count"] == checkpoint["impact_count"]
    # No new revocation yet: still matching. Now revoke to grow the set.
    assert client.get(_recon_url(receipt_id)).json()["matches"] is True

    impacts = client.get(PACKAGE_URL).json()["impacts"]
    # Revoke one of the world's remaining non-revoked attestations.
    from tests.test_revocation_impacts import _revoke

    _revoke(client, impacts[1]["attestation_id"], reason="late revocation")
    current = _served_checkpoint(client)
    assert current["impact_count"] == checkpoint["impact_count"] + 1
    assert current["impacts_digest_hex"] != checkpoint["impacts_digest_hex"]

    body = client.get(_recon_url(receipt_id)).json()
    assert body["local_checkpoint"] == current
    assert body["matches"] is False


def test_each_identity_field_is_required_for_a_match(client, db_session):
    _world(client)
    current = _served_checkpoint(client)

    # Same digest and count, different version -> no match.
    version_receipt = _insert_receipt(
        db_session,
        "provenance-revocation-impact-checkpoint-v2",
        current["impact_count"],
        current["impacts_digest_hex"],
    )
    body = client.get(_recon_url(version_receipt)).json()
    assert body["local_checkpoint"] == current
    assert body["matches"] is False

    # Same version and digest, different count -> no match.
    count_receipt = _insert_receipt(
        db_session,
        current["checkpoint_version"],
        current["impact_count"] + 1,
        current["impacts_digest_hex"],
    )
    body = client.get(_recon_url(count_receipt)).json()
    assert body["local_checkpoint"] == current
    assert body["matches"] is False

    # Same version and count, different digest -> no match.
    other_digest = "0" * 64
    assert other_digest != current["impacts_digest_hex"]
    digest_receipt = _insert_receipt(
        db_session,
        current["checkpoint_version"],
        current["impact_count"],
        other_digest,
    )
    body = client.get(_recon_url(digest_receipt)).json()
    assert body["local_checkpoint"] == current
    assert body["matches"] is False


def test_offline_receipt_against_unrelated_empty_library_does_not_match(
    client, db_session
):
    # A wholly fabricated, self-consistent receipt registered while local
    # state is empty: the described impacts are not materialized, so the
    # verdict cannot be inferred from them.
    resp = client.post(IMPORTS_URL, json=_offline_request())
    assert resp.status_code == 201, resp.text
    receipt = resp.json()

    # The import itself wrote no revocation; the local impact set is empty
    # while the receipt claims the two-impact offline sequence.
    local = _served_checkpoint(client)
    assert local["impact_count"] == 0
    assert receipt["impact_count"] == 2

    body = client.get(_recon_url(receipt["id"])).json()
    assert body == {
        "import_id": receipt["id"],
        "local_checkpoint": local,
        "matches": False,
    }


def test_reconciliation_is_deterministic_across_restart(file_client, tmp_db_url):
    # Direct, event-free fixture state so the verdict survives identically.
    session = file_client.app.state.session_factory()
    try:
        receipt_id = _insert_receipt(
            session, CHECKPOINT_VERSION, 0, EMPTY_DIGEST
        )
    finally:
        session.close()

    expected = file_client.get(_recon_url(receipt_id))
    assert expected.status_code == 200, expected.text

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        resp = client.get(_recon_url(receipt_id))
        assert resp.status_code == 200, resp.text
        assert resp.json() == expected.json()
        assert resp.json()["matches"] is True


# --- Missing receipt -------------------------------------------------------------


def test_unknown_import_id_is_an_explicit_specific_404(client, db_session):
    unknown = "rii_" + "0" * 64
    resp = client.get(_recon_url(unknown))
    assert resp.status_code == 404, resp.text
    error = resp.json()["error"]
    assert error["code"] == "impact_import_not_found"
    assert error["details"]["import_id"] == unknown
    # The failed read wrote nothing.
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_unshaped_and_other_prefix_ids_are_404(client):
    for raw_id in ("rii_doesnotexist", "aci_" + "a" * 64, "not-an-id"):
        resp = client.get(_recon_url(raw_id))
        assert resp.status_code == 404, raw_id
        assert resp.json()["error"]["code"] == "impact_import_not_found"


def test_404_even_when_local_impacts_exist(client, db_session):
    _world(client)
    resp = client.get(_recon_url("rii_" + "f" * 64))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "impact_import_not_found"


# --- Query parameter boundary ----------------------------------------------------


def test_any_query_parameter_is_422(client, db_session):
    receipt_id, _ = _matching_receipt(db_session, client)
    base = _recon_url(receipt_id)
    for url in (
        f"{base}?limit=10",
        f"{base}?cursor=abc",
        f"{base}?attestation_id=att_1",
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
    resp = client.get(_recon_url("rii_doesnotexist") + "?anything=1")
    assert resp.status_code == 422, resp.text
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"]["issues"][0]["loc"] == ["query", "anything"]


def test_repeated_parameter_is_422_before_the_receipt_lookup(client):
    resp = client.get(_recon_url("rii_" + "0" * 64) + "?x=1&x=2")
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only and no-echo boundary ----------------------------------------------


def test_reconciliation_writes_nothing(client, db_session):
    _world(client)
    matching, _ = _matching_receipt(db_session, client)
    # A second receipt for a state that does not exist locally.
    non_matching = _insert_receipt(
        db_session, CHECKPOINT_VERSION, 99, "a" * 64
    )

    before = _counts(db_session)

    # A matching read, a non-matching read, a 422, and a 404 all leave every
    # table (receipts and audit events included) untouched.
    assert client.get(_recon_url(matching)).status_code == 200
    assert client.get(_recon_url(non_matching)).status_code == 200
    assert client.get(_recon_url(matching) + "?x=1").status_code == 422
    assert client.get(_recon_url("rii_" + "0" * 64)).status_code == 404

    db_session.expire_all()
    assert _counts(db_session) == before


def test_empty_library_success_and_failures_write_nothing(client, db_session):
    receipt_id = _insert_receipt(db_session, CHECKPOINT_VERSION, 0, EMPTY_DIGEST)
    db_session.expire_all()
    before = _counts(db_session)

    assert client.get(_recon_url(receipt_id)).status_code == 200
    assert client.get(_recon_url(receipt_id) + "?z=1").status_code == 422
    assert client.get(_recon_url("rii_" + "9" * 64)).status_code == 404

    db_session.expire_all()
    after = _counts(db_session)
    assert after == before
    # Nothing besides the directly-inserted receipt exists: no audit events.
    assert after[AuditEvent] == 0
    assert after[ImpactImportRecord] == 1


def test_response_never_echoes_the_imported_impacts(client, db_session):
    marker = "recon-no-echo-marker-3f9c"
    request = _offline_request()
    request["impacts"][0]["reason"] = marker
    request["checkpoint"]["impacts_digest_hex"] = _digest_of(request["impacts"])
    receipt = client.post(IMPORTS_URL, json=request).json()

    resp = client.get(_recon_url(receipt["id"]))
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == RECON_KEYS
    assert set(body["local_checkpoint"]) == CHECKPOINT_KEYS
    serialized = json.dumps(body, ensure_ascii=False)
    # The imported impacts array and its values are absent in every form.
    assert '"impacts"' not in serialized
    assert marker not in serialized
    assert '"received_at"' not in serialized
    for impact in request["impacts"]:
        assert impact["id"] not in serialized
        assert impact["created_at"] not in serialized


def test_import_audit_event_does_not_enter_the_local_checkpoint(client):
    # The import's own revocation_impact.imported audit event is not a
    # revocation: the local impact checkpoint is unaffected by it.
    assert _served_checkpoint(client)["impact_count"] == 0
    resp = client.post(IMPORTS_URL, json=_offline_request())
    assert resp.status_code == 201
    events = client.get("/v1/audit-events").json()["items"]
    assert [e["event_type"] for e in events] == [EVENT_REVOCATION_IMPACT_IMPORTED]
    assert _served_checkpoint(client)["impact_count"] == 0
