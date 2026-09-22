"""Tests for the read-only checkpoint-import reconciliation endpoint.

Covers GET
/v1/audit-events/checkpoint-imports/{import_id}/reconciliation: the receipt
is read by ``import_id`` and an unknown id is the existing
``404 audit_checkpoint_import_not_found``. The success body is strictly
``{"import_id", "local_checkpoint", "matches"}``; ``local_checkpoint`` is the
four-field checkpoint computed over the complete, unfiltered local audit
sequence under the existing ``GET /v1/audit-events/checkpoint`` rules (fixed
version/algorithm, stable creation order, UTC representation, canonical
SHA-256 digest), and ``matches`` is true only when the receipt's
``checkpoint_version``, ``event_count``, and ``events_digest_hex`` all equal
the local checkpoint fields. Any (or repeated) query parameter is
``422 validation_error`` before the receipt lookup. The imported event array
is never read or echoed and no unpersisted material participates in the
verdict. The route is strictly read-only -- no resource, receipt, or audit
row is written -- on success, on an empty database, on parameter failure,
and for a missing receipt. All fixtures are deterministic and offline.
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
)
from provenance.models import (
    EVENT_AUDIT_CHECKPOINT_IMPORTED,
    EVENT_CONTENT_CREATED,
)

IMPORTS_URL = "/v1/audit-events/checkpoint-imports"
CHECKPOINT_URL = "/v1/audit-events/checkpoint"
EVENTS_URL = "/v1/audit-events"
CHECKPOINT_VERSION = "provenance-audit-checkpoint-v1"
EMPTY_DIGEST = hashlib.sha256(b"[]").hexdigest()
RECONCILIATION_KEYS = {"import_id", "local_checkpoint", "matches"}
CHECKPOINT_KEYS = {
    "checkpoint_version",
    "digest_algorithm",
    "event_count",
    "events_digest_hex",
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
    AuditEvent,
)


def _reconciliation_url(import_id: str) -> str:
    return f"{IMPORTS_URL}/{import_id}/reconciliation"


def _digest_of(events: list) -> str:
    """Independently canonicalize the event wire array and digest it."""
    canonical = json.dumps(
        events,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _served_checkpoint(client) -> dict:
    resp = client.get(CHECKPOINT_URL)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _served_events(client) -> list:
    """The full unpaginated audit-event search wire view."""
    items = []
    cursor = None
    for _ in range(100):
        params = {} if cursor is None else {"cursor": cursor}
        resp = client.get(EVENTS_URL, params=params)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        items.extend(body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            return items
    raise AssertionError("pagination did not terminate")


def _offline_events() -> list:
    """A self-contained event sequence fabricated without service state."""
    return [
        {
            "event_type": "actor.created",
            "resource_id": "org-émoji-✓",
            "created_at": "2026-01-02T03:04:05Z",
        },
        {
            "event_type": "content.created",
            "resource_id": "cnt_" + hashlib.sha256(b"offline").hexdigest(),
            "created_at": "2026-01-02T03:04:06.500Z",
        },
    ]


def _offline_request(events: list | None = None) -> dict:
    events = _offline_events() if events is None else events
    return {
        "checkpoint": {
            "checkpoint_version": CHECKPOINT_VERSION,
            "digest_algorithm": "sha256",
            "event_count": len(events),
            "events_digest_hex": _digest_of(events),
        },
        "events": events,
    }


def _insert_event(db_session, event_type, resource_id, created_at):
    """Append one audit row at a fixed instant (deterministic state)."""
    db_session.add(
        AuditEvent(
            event_type=event_type,
            resource_id=resource_id,
            created_at=created_at,
        )
    )
    db_session.commit()


def _insert_receipt(db_session, checkpoint_version, event_count, digest_hex):
    """Persist a receipt directly, without writing an import audit event.

    The reconciliation reads persisted state alone; direct insertion keeps
    the surrounding audit sequence exactly controlled.
    """
    receipt_id = ids.checkpoint_import_id(
        checkpoint_version, event_count, digest_hex
    )
    db_session.add(
        CheckpointImportRecord(
            id=receipt_id,
            checkpoint_version=checkpoint_version,
            event_count=event_count,
            events_digest_hex=digest_hex,
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
        checkpoint["event_count"],
        checkpoint["events_digest_hex"],
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
    # Empty local trail + receipt for the empty checkpoint: matches true,
    # and the local checkpoint is exactly the served empty checkpoint.
    receipt_id = _insert_receipt(db_session, CHECKPOINT_VERSION, 0, EMPTY_DIGEST)

    resp = client.get(_reconciliation_url(receipt_id))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == RECONCILIATION_KEYS
    assert set(body["local_checkpoint"]) == CHECKPOINT_KEYS
    assert body == {
        "import_id": receipt_id,
        "local_checkpoint": {
            "checkpoint_version": CHECKPOINT_VERSION,
            "digest_algorithm": "sha256",
            "event_count": 0,
            "events_digest_hex": EMPTY_DIGEST,
        },
        "matches": True,
    }


def test_matching_receipt_reports_current_checkpoint_and_matches(
    client, db_session
):
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    _insert_event(
        db_session,
        EVENT_CONTENT_CREATED,
        "cnt_t2",
        datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    receipt_id, checkpoint = _matching_receipt(db_session, client)

    resp = client.get(_reconciliation_url(receipt_id))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == RECONCILIATION_KEYS
    assert body["import_id"] == receipt_id
    assert body["matches"] is True
    # The local checkpoint is exactly the unfiltered served checkpoint.
    assert body["local_checkpoint"] == checkpoint
    assert body["local_checkpoint"] == _served_checkpoint(client)
    # The digest is independently reproducible from the audit-event listing.
    served = _served_events(client)
    assert body["local_checkpoint"]["event_count"] == len(served)
    assert body["local_checkpoint"]["events_digest_hex"] == _digest_of(served)


def test_local_checkpoint_is_unfiltered_when_filters_would_exclude_events(
    client, db_session
):
    # A filter on the checkpoint route would exclude events; reconciliation
    # must always hash the complete local sequence regardless.
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    _insert_event(
        db_session,
        EVENT_CONTENT_CREATED,
        "cnt_x",
        datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    receipt_id, checkpoint = _matching_receipt(db_session, client)
    assert checkpoint["event_count"] == 2

    filtered = client.get(CHECKPOINT_URL, params={"event_type": "actor.created"})
    assert filtered.json()["event_count"] == 1

    body = client.get(_reconciliation_url(receipt_id)).json()
    assert body["local_checkpoint"]["event_count"] == 2
    assert body["local_checkpoint"] == checkpoint
    assert body["matches"] is True


def test_local_trail_growing_after_registration_flips_match_to_false(
    client, db_session
):
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    receipt_id, checkpoint = _matching_receipt(db_session, client)
    assert client.get(_reconciliation_url(receipt_id)).json()["matches"] is True

    # The local sequence changes after registration: the receipt no longer
    # matches, but the current checkpoint is still reported in full.
    _insert_event(
        db_session,
        EVENT_CONTENT_CREATED,
        "cnt_later",
        datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    current = _served_checkpoint(client)
    assert current["event_count"] == checkpoint["event_count"] + 1
    assert current["events_digest_hex"] != checkpoint["events_digest_hex"]

    body = client.get(_reconciliation_url(receipt_id)).json()
    assert body["local_checkpoint"] == current
    assert body["matches"] is False


def test_each_identity_field_is_required_for_a_match(client, db_session):
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    current = _served_checkpoint(client)

    # Same digest and count, different version -> no match.
    version_receipt = _insert_receipt(
        db_session,
        "provenance-audit-checkpoint-v2",
        current["event_count"],
        current["events_digest_hex"],
    )
    body = client.get(_reconciliation_url(version_receipt)).json()
    assert body["local_checkpoint"] == current
    assert body["matches"] is False

    # Same version and digest, different count -> no match.
    count_receipt = _insert_receipt(
        db_session,
        current["checkpoint_version"],
        current["event_count"] + 1,
        current["events_digest_hex"],
    )
    body = client.get(_reconciliation_url(count_receipt)).json()
    assert body["local_checkpoint"] == current
    assert body["matches"] is False

    # Same version and count, different digest -> no match.
    other_digest = "0" * 64
    assert other_digest != current["events_digest_hex"]
    digest_receipt = _insert_receipt(
        db_session,
        current["checkpoint_version"],
        current["event_count"],
        other_digest,
    )
    body = client.get(_reconciliation_url(digest_receipt)).json()
    assert body["local_checkpoint"] == current
    assert body["matches"] is False


def test_offline_receipt_against_unrelated_empty_trail_does_not_match(
    client, db_session
):
    # A wholly fabricated, self-consistent receipt registered while local
    # state is empty: the described events are not materialized, so the
    # verdict cannot be inferred from them.
    resp = client.post(IMPORTS_URL, json=_offline_request())
    assert resp.status_code == 201, resp.text
    receipt = resp.json()

    # The import itself wrote exactly one local audit event; the receipt
    # claims the two-event offline sequence.
    local = _served_checkpoint(client)
    assert local["event_count"] == 1
    assert receipt["event_count"] == 2

    body = client.get(_reconciliation_url(receipt["id"])).json()
    assert body == {
        "import_id": receipt["id"],
        "local_checkpoint": local,
        "matches": False,
    }


def test_api_import_of_a_served_sequence_diverges_by_its_own_audit_event(
    client, db_session
):
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    before = _served_checkpoint(client)
    assert before["event_count"] == 1

    # Registering the checkpoint appends its own audit.checkpoint_imported
    # event, so the receipt commits to the pre-registration sequence and
    # therefore cannot match the grown local sequence.
    resp = client.post(
        IMPORTS_URL,
        json={"checkpoint": before, "events": _served_events(client)},
    )
    assert resp.status_code == 201, resp.text
    receipt = resp.json()

    after = _served_checkpoint(client)
    assert after["event_count"] == 2
    assert after["events_digest_hex"] != before["events_digest_hex"]

    body = client.get(_reconciliation_url(receipt["id"])).json()
    assert body["import_id"] == receipt["id"]
    assert body["local_checkpoint"] == after
    assert body["matches"] is False

    # The extra local event is exactly this import's audit event.
    items = _served_events(client)
    assert [(i["event_type"], i["resource_id"]) for i in items][-1] == (
        EVENT_AUDIT_CHECKPOINT_IMPORTED,
        receipt["id"],
    )


def test_reconciliation_is_deterministic_across_restart(file_client, tmp_db_url):
    # Direct, event-free fixture state so the verdict survives identically.
    session = file_client.app.state.session_factory()
    try:
        receipt_id = _insert_receipt(
            session, CHECKPOINT_VERSION, 0, EMPTY_DIGEST
        )
    finally:
        session.close()

    expected = file_client.get(_reconciliation_url(receipt_id))
    assert expected.status_code == 200, expected.text

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        resp = client.get(_reconciliation_url(receipt_id))
        assert resp.status_code == 200, resp.text
        assert resp.json() == expected.json()
        assert resp.json()["matches"] is True


# --- Missing receipt -------------------------------------------------------------


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


def test_404_even_when_local_events_exist(client, db_session):
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    resp = client.get(_reconciliation_url("aci_" + "f" * 64))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "audit_checkpoint_import_not_found"


# --- Query parameter boundary ----------------------------------------------------


def test_any_query_parameter_is_422(client, db_session):
    receipt_id, _ = _matching_receipt(db_session, client)
    base = _reconciliation_url(receipt_id)
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
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"]["issues"][0]["loc"] == ["query", "anything"]


def test_repeated_parameter_is_422_before_the_receipt_lookup(client):
    resp = client.get(
        _reconciliation_url("aci_" + "0" * 64) + "?x=1&x=2"
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only and no-echo boundary ----------------------------------------------


def test_reconciliation_writes_nothing(client, db_session):
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    matching, _ = _matching_receipt(db_session, client)
    # A second receipt for a state that does not exist locally.
    non_matching = _insert_receipt(
        db_session, CHECKPOINT_VERSION, 99, "a" * 64
    )

    before = _counts(db_session)

    # A matching read, a non-matching read, a 422, and a 404 all leave every
    # table (receipts and audit events included) untouched.
    assert client.get(_reconciliation_url(matching)).status_code == 200
    assert client.get(_reconciliation_url(non_matching)).status_code == 200
    assert (
        client.get(_reconciliation_url(matching) + "?x=1").status_code == 422
    )
    assert (
        client.get(_reconciliation_url("aci_" + "0" * 64)).status_code == 404
    )

    db_session.expire_all()
    assert _counts(db_session) == before


def test_empty_library_success_and_failures_write_nothing(client, db_session):
    receipt_id = _insert_receipt(db_session, CHECKPOINT_VERSION, 0, EMPTY_DIGEST)
    db_session.expire_all()
    before = _counts(db_session)

    assert client.get(_reconciliation_url(receipt_id)).status_code == 200
    assert (
        client.get(_reconciliation_url(receipt_id) + "?z=1").status_code == 422
    )
    assert (
        client.get(_reconciliation_url("aci_" + "9" * 64)).status_code == 404
    )

    db_session.expire_all()
    after = _counts(db_session)
    assert after == before
    # Nothing besides the directly-inserted receipt exists: no audit events.
    assert after[AuditEvent] == 0
    assert after[CheckpointImportRecord] == 1


def test_response_never_echoes_the_imported_events(client, db_session):
    marker = "reconciliation-no-echo-marker-3f9c"
    request = _offline_request()
    request["events"][0]["resource_id"] = marker
    request["checkpoint"]["events_digest_hex"] = _digest_of(request["events"])
    receipt = client.post(IMPORTS_URL, json=request).json()

    resp = client.get(_reconciliation_url(receipt["id"]))
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == RECONCILIATION_KEYS
    assert set(body["local_checkpoint"]) == CHECKPOINT_KEYS
    serialized = json.dumps(body, ensure_ascii=False)
    # The imported event array and its values are absent in every form.
    assert '"events"' not in serialized
    assert marker not in serialized
    assert '"received_at"' not in serialized
    for event in request["events"]:
        assert event["resource_id"] not in serialized
        assert event["created_at"] not in serialized
