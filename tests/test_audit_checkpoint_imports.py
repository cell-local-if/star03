"""Tests for the controlled audit-checkpoint import endpoint.

Covers POST /v1/audit-events/checkpoint-imports and GET
/v1/audit-events/checkpoint-imports/{import_id}. The request body is
exactly ``{"checkpoint", "events"}``: ``checkpoint`` is the existing
four-field audit checkpoint and ``events`` is the existing three-field
event sequence, validated exactly as on the stateless verification route
(fixed ``provenance-audit-checkpoint-v1`` version, ``sha256`` algorithm,
non-negative strict-integer count equal to the array length, 64 lowercase
hex digest, non-empty strings, and strict RFC 3339 UTC timestamps). The
digest is recomputed over the events in their received array order under
the existing canonical SHA-256 rules; a structural, field, count, or
digest mismatch is a 422 validation_error and writes nothing.

Verification is decided by the request body alone: the described events
are never created, modified, or queried and no id is resolved against
local state. The receipt stores only the receiving identity (checkpoint
version, events digest, event count), never the events. First
registration returns 201 with a stable ``aci_`` id and a UTC
``received_at``, and writes the ``audit.checkpoint_imported`` audit event
in the same transaction; a retry for the same identity returns the
original receipt with 200 and no new audit. GET reads the public receipt;
an unknown id is an explicit 404. All fixtures are deterministic and
offline.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import func, select

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
from provenance.models import EVENT_AUDIT_CHECKPOINT_IMPORTED
from tests.helpers import DIGEST_A, create_actor

URL = "/v1/audit-events/checkpoint-imports"
CHECKPOINT_VERSION = "provenance-audit-checkpoint-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
RECEIPT_KEYS = {
    "id",
    "checkpoint_version",
    "event_count",
    "events_digest_hex",
    "received_at",
}
_DOMAIN_MODELS = (
    Actor,
    Content,
    Claim,
    EvidenceBundle,
    Attestation,
    AttestationRevocation,
    ContentRelation,
)


def _digest_of(events: list) -> str:
    """Independently canonicalize the event array and digest it."""
    canonical = json.dumps(
        events,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _offline_events() -> list:
    """A self-contained event sequence fabricated without any service state."""
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


def _import_request(events: list | None = None) -> dict:
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


def _served_import_request(client) -> dict:
    """A ``{checkpoint, events}`` body built from the service's own audit views."""
    checkpoint = client.get("/v1/audit-events/checkpoint").json()
    events = client.get("/v1/audit-events").json()["items"]
    return {"checkpoint": checkpoint, "events": events}


def _setup_events(client):
    create_actor(client, actor_id="org-1")
    resp = client.post(
        "/v1/contents",
        json={
            "digest_algorithm": "sha256",
            "digest_hex": DIGEST_A,
            "media_type": "image/png",
            "actor_id": "org-1",
        },
    )
    assert resp.status_code == 201, resp.text


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def _import_events(db_session):
    return (
        db_session.execute(
            select(AuditEvent)
            .where(AuditEvent.event_type == EVENT_AUDIT_CHECKPOINT_IMPORTED)
            .order_by(AuditEvent.seq.asc())
        )
        .scalars()
        .all()
    )


# --- First registration: success shape -----------------------------------------


def test_offline_checkpoint_first_import_returns_201_receipt(client, db_session):
    # The database is empty: every identifier in the checkpoint is unknown
    # locally, and registration must succeed on the body alone.
    request = _import_request()
    resp = client.post(URL, json=request)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == RECEIPT_KEYS
    assert body["id"].startswith("aci_")
    assert len(body["id"]) == len("aci_") + 64
    assert body["checkpoint_version"] == CHECKPOINT_VERSION
    assert body["event_count"] == len(request["events"])
    assert body["events_digest_hex"] == request["checkpoint"]["events_digest_hex"]
    # The events are never echoed back on the receipt.
    assert "events" not in body
    assert "checkpoint" not in body
    assert "digest_algorithm" not in body

    received_at = datetime.fromisoformat(body["received_at"])
    assert received_at.utcoffset().total_seconds() == 0
    assert body["received_at"].endswith(("Z", "+00:00"))

    # Exactly one receipt row exists with the same identity.
    records = db_session.execute(select(CheckpointImportRecord)).scalars().all()
    assert len(records) == 1
    (record,) = records
    assert record.id == body["id"]
    assert record.checkpoint_version == CHECKPOINT_VERSION
    assert record.event_count == body["event_count"]
    assert record.events_digest_hex == body["events_digest_hex"]


def test_served_checkpoint_imports_without_local_lookups(client):
    _setup_events(client)
    resp = client.post(URL, json=_served_import_request(client))
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["id"].startswith("aci_")
    assert body["event_count"] == 2


def test_receipt_id_is_the_deterministic_identity_id(client):
    from provenance import ids

    request = _import_request()
    body = client.post(URL, json=request).json()
    expected = ids.checkpoint_import_id(
        CHECKPOINT_VERSION,
        len(request["events"]),
        request["checkpoint"]["events_digest_hex"],
    )
    assert body["id"] == expected


def test_distinct_checkpoint_identities_get_distinct_stable_ids(client):
    first = client.post(URL, json=_import_request()).json()

    other_events = _offline_events()
    other_events.append(
        {
            "event_type": "claim.created",
            "resource_id": "clm_" + hashlib.sha256(b"offline-claim").hexdigest(),
            "created_at": "2026-01-02T03:04:07Z",
        }
    )
    second = client.post(URL, json=_import_request(other_events)).json()

    # A changed sequence changes both the count and the digest identity.
    assert second["event_count"] != first["event_count"]
    assert second["events_digest_hex"] != first["events_digest_hex"]
    assert second["id"] != first["id"]


def test_empty_sequence_imports(client):
    request = _import_request(events=[])
    assert request["checkpoint"]["events_digest_hex"] == hashlib.sha256(
        b"[]"
    ).hexdigest()
    resp = client.post(URL, json=request)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["event_count"] == 0
    assert body["events_digest_hex"] == hashlib.sha256(b"[]").hexdigest()


# --- Idempotency ---------------------------------------------------------------


def test_same_identity_retry_returns_200_unchanged_record(client):
    request = _import_request()
    first = client.post(URL, json=request)
    assert first.status_code == 201
    original = first.json()

    for _ in range(2):
        retry = client.post(URL, json=json.loads(json.dumps(request)))
        assert retry.status_code == 200, retry.text
        assert retry.json() == original


def test_retry_returns_original_received_at(client, db_session):
    request = _import_request()
    original = client.post(URL, json=request).json()
    retry = client.post(URL, json=request).json()
    assert retry["id"] == original["id"]
    assert retry["received_at"] == original["received_at"]

    records = db_session.execute(select(CheckpointImportRecord)).scalars().all()
    assert [r.id for r in records] == [original["id"]]


def test_retry_writes_no_second_row_or_audit_event(client, db_session):
    request = _import_request()
    assert client.post(URL, json=request).status_code == 201
    assert len(_import_events(db_session)) == 1
    total_after_first = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    assert client.post(URL, json=request).status_code == 200
    assert client.post(URL, json=request).status_code == 200

    assert len(_import_events(db_session)) == 1
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == total_after_first
    )
    assert (
        len(db_session.execute(select(CheckpointImportRecord)).scalars().all())
        == 1
    )


def test_event_order_participates_in_the_identity(client):
    # The array keeps its order: the same events in another order are a
    # different sequence, hence a different receiving identity.
    request = _import_request()
    reordered = json.loads(json.dumps(request))
    reordered["events"] = list(reversed(reordered["events"]))
    reordered["checkpoint"]["events_digest_hex"] = _digest_of(reordered["events"])
    assert reordered["checkpoint"]["events_digest_hex"] != request["checkpoint"][
        "events_digest_hex"
    ]

    first = client.post(URL, json=request)
    assert first.status_code == 201
    second = client.post(URL, json=reordered)
    assert second.status_code == 201, second.text
    assert second.json()["id"] != first.json()["id"]


def test_import_is_idempotent_across_app_restart(file_client, tmp_db_url):
    request = _import_request()
    original = file_client.post(URL, json=request)
    assert original.status_code == 201
    expected = original.json()

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        retry = client.post(URL, json=request)
        assert retry.status_code == 200, retry.text
        assert retry.json() == expected


# --- Read endpoint -------------------------------------------------------------


def test_get_returns_the_public_receipt(client):
    created = client.post(URL, json=_import_request()).json()
    resp = client.get(f"{URL}/{created['id']}")
    assert resp.status_code == 200, resp.text
    assert resp.json() == created
    assert set(resp.json()) == RECEIPT_KEYS


def test_get_unknown_id_is_an_explicit_specific_404(client, db_session):
    unknown = "aci_" + "0" * 64
    resp = client.get(f"{URL}/{unknown}")
    assert resp.status_code == 404, resp.text
    error = resp.json()["error"]
    assert error["code"] == "audit_checkpoint_import_not_found"
    assert error["details"]["import_id"] == unknown
    # The failed read wrote nothing.
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_get_unshaped_and_other_prefix_ids_are_404(client):
    for raw_id in ("aci_doesnotexist", "eir_" + "a" * 64, "not-an-id"):
        resp = client.get(f"{URL}/{raw_id}")
        assert resp.status_code == 404, raw_id
        assert resp.json()["error"]["code"] == "audit_checkpoint_import_not_found"


def test_get_is_read_only(client, db_session):
    created = client.post(URL, json=_import_request()).json()
    events_after_import = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )
    assert client.get(f"{URL}/{created['id']}").status_code == 200
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_after_import
    )


# --- Audit transaction ---------------------------------------------------------


def test_first_import_writes_one_checkpoint_imported_audit_event(client, db_session):
    body = client.post(URL, json=_import_request()).json()
    events = _import_events(db_session)
    assert len(events) == 1
    event = events[0]
    assert event.event_type == EVENT_AUDIT_CHECKPOINT_IMPORTED
    assert event.resource_id == body["id"]
    assert event.created_at.tzinfo.utcoffset(event.created_at).total_seconds() == 0


def test_record_and_audit_event_commit_together(client, db_session):
    resp = client.post(URL, json=_import_request())
    assert resp.status_code == 201
    receipt_id = resp.json()["id"]

    # Both the receipt and its audit event are visible to another session
    # immediately: they committed in the same transaction.
    db_session.expire_all()
    record = db_session.execute(
        select(CheckpointImportRecord).where(
            CheckpointImportRecord.id == receipt_id
        )
    ).scalar_one()
    event = db_session.execute(
        select(AuditEvent).where(AuditEvent.resource_id == receipt_id)
    ).scalar_one()
    assert record is not None
    assert event.event_type == EVENT_AUDIT_CHECKPOINT_IMPORTED


# --- Offline independence and no described-event side effects ------------------


def test_registration_never_depends_on_local_state(client, db_session):
    # A wholly fabricated, self-consistent checkpoint against unknown ids is
    # accepted into an empty database.
    resp = client.post(URL, json=_import_request())
    assert resp.status_code == 201, resp.text

    # No domain resource was materialized.
    for model in _DOMAIN_MODELS:
        count = db_session.execute(
            select(func.count()).select_from(model)
        ).scalar_one()
        assert count == 0, model


def test_described_events_are_never_created_or_listed(client, db_session):
    request = _import_request()
    described = {(e["event_type"], e["resource_id"]) for e in request["events"]}
    receipt = client.post(URL, json=request)
    assert receipt.status_code == 201
    receipt_id = receipt.json()["id"]

    # The only audit event in the store is the import receipt's own; the
    # described events were not created and therefore never appear in the
    # audit-event listing or checkpoint.
    stored = [
        (event.event_type, event.resource_id)
        for event in db_session.execute(select(AuditEvent)).scalars().all()
    ]
    assert stored == [(EVENT_AUDIT_CHECKPOINT_IMPORTED, receipt_id)]
    assert described.isdisjoint(stored)

    listing = client.get("/v1/audit-events").json()
    assert [
        (item["event_type"], item["resource_id"]) for item in listing["items"]
    ] == [(EVENT_AUDIT_CHECKPOINT_IMPORTED, receipt_id)]


def test_described_events_matching_local_resources_are_not_queried(client, db_session):
    # A real local audit event exists with the same type and a real actor;
    # importing an offline checkpoint that merely resembles local state
    # neither queries nor mutates that state.
    create_actor(client, actor_id="org-1")
    local_events_before = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    events = [
        {
            "event_type": "actor.created",
            "resource_id": "org-1",
            "created_at": "2020-01-01T00:00:00Z",
        }
    ]
    resp = client.post(URL, json=_import_request(events))
    assert resp.status_code == 201, resp.text

    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == local_events_before + 1
    )
    assert db_session.execute(select(func.count()).select_from(Actor)).scalar_one() == 1


def test_receipt_table_persists_no_events_column(client, db_session):
    # The stored record has exactly the identity and bookkeeping columns;
    # there is nowhere an event id, type, timestamp, or array could be
    # persisted.
    marker = "events-persistence-marker-7e4a"
    request = _import_request()
    request["events"][0]["resource_id"] = marker
    request["checkpoint"]["events_digest_hex"] = _digest_of(request["events"])
    assert client.post(URL, json=request).status_code == 201

    columns = set(CheckpointImportRecord.__table__.columns.keys())
    assert columns == {
        "seq",
        "id",
        "checkpoint_version",
        "event_count",
        "events_digest_hex",
        "created_at",
    }

    record = db_session.execute(select(CheckpointImportRecord)).scalar_one()
    assert marker not in repr(vars(record))


# --- Digest mismatch -----------------------------------------------------------


def test_events_digest_mismatch_is_422_and_writes_nothing(client, db_session):
    request = _import_request()
    claimed = request["checkpoint"]["events_digest_hex"]
    request["checkpoint"]["events_digest_hex"] = claimed[:-1] + (
        "0" if claimed[-1] != "0" else "1"
    )

    resp = client.post(URL, json=request)
    assert resp.status_code == 422, resp.text
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"]["reason"] == "events_digest_mismatch"
    assert error["details"]["computed_digest_hex"] == _digest_of(
        request["events"]
    )

    assert db_session.execute(select(CheckpointImportRecord)).scalars().all() == []
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_tampered_event_is_422_and_writes_nothing(client, db_session):
    request = _import_request()
    # The checkpoint still commits to the untouched sequence.
    request["events"][0]["resource_id"] = "org-forged"

    _assert_validation_error(client.post(URL, json=request))
    assert db_session.execute(select(CheckpointImportRecord)).scalars().all() == []
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_reordered_array_against_original_checkpoint_is_422(client):
    request = _import_request()
    request["events"] = list(reversed(request["events"]))
    _assert_validation_error(client.post(URL, json=request))


def test_tampered_retry_under_existing_identity_is_422_and_keeps_record(
    client, db_session
):
    request = _import_request()
    original = client.post(URL, json=request)
    assert original.status_code == 201

    # Same identity fields, but the presented events no longer match:
    # validation beats the idempotent lookup and the original survives.
    tampered = json.loads(json.dumps(request))
    tampered["events"][0]["resource_id"] = "forged"
    _assert_validation_error(client.post(URL, json=tampered))

    assert client.get(
        f"{URL}/{original.json()['id']}"
    ).json() == original.json()
    assert (
        len(db_session.execute(select(CheckpointImportRecord)).scalars().all())
        == 1
    )
    assert len(_import_events(db_session)) == 1


def test_mismatch_does_not_query_local_state(client):
    # A digest mismatch is rejected on the body alone in an empty database;
    # no id resolution could change it.
    request = _import_request()
    request["checkpoint"]["events_digest_hex"] = "0" * 64
    _assert_validation_error(client.post(URL, json=request))


def test_unescaped_non_ascii_utf8_digest_is_recomputed_offline(client):
    request = _import_request()
    assert any(
        ord(ch) > 127 for ch in json.dumps(request["events"], ensure_ascii=False)
    )
    resp = client.post(URL, json=request)
    assert resp.status_code == 201, resp.text
    assert resp.json()["events_digest_hex"] == _digest_of(request["events"])

    # A digest computed over ASCII-escaped bytes would not match.
    escaped = json.dumps(
        request["events"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    assert hashlib.sha256(escaped).hexdigest() != _digest_of(request["events"])


def test_equivalent_utc_spellings_are_distinct_identities(client):
    # Both explicit UTC designators are strict RFC 3339 UTC; each spelling
    # participates in the digest exactly as received.
    events_z = _offline_events()
    events_offset = json.loads(json.dumps(events_z))
    events_offset[0]["created_at"] = "2026-01-02T03:04:05+00:00"
    assert _digest_of(events_z) != _digest_of(events_offset)

    first = client.post(URL, json=_import_request(events_z))
    assert first.status_code == 201
    second = client.post(URL, json=_import_request(events_offset))
    assert second.status_code == 201, second.text
    assert second.json()["id"] != first.json()["id"]


# --- Structural / field boundary -----------------------------------------------


def test_body_requires_exactly_checkpoint_and_events(client):
    request = _import_request()

    for missing in ("checkpoint", "events"):
        incomplete = {k: v for k, v in request.items() if k != missing}
        _assert_validation_error(client.post(URL, json=incomplete))

    augmented = dict(request)
    augmented["batch_id"] = "b-1"
    _assert_validation_error(client.post(URL, json=augmented))


def test_non_object_bodies_are_422(client):
    for body in (None, [], "checkpoint", 42):
        _assert_validation_error(client.post(URL, json=body))
    _assert_validation_error(client.post(URL, json={}))


def test_members_have_correct_types(client):
    request = _import_request()
    for member, wrong in (
        ("checkpoint", []),
        ("checkpoint", None),
        ("events", {}),
        ("events", "events"),
    ):
        bad = json.loads(json.dumps(request))
        bad[member] = wrong
        _assert_validation_error(client.post(URL, json=bad))


def test_checkpoint_has_exactly_the_four_members(client):
    request = _import_request()

    for member in (
        "checkpoint_version",
        "digest_algorithm",
        "event_count",
        "events_digest_hex",
    ):
        incomplete = json.loads(json.dumps(request))
        del incomplete["checkpoint"][member]
        _assert_validation_error(client.post(URL, json=incomplete))

    augmented = json.loads(json.dumps(request))
    augmented["checkpoint"]["limit"] = 50
    _assert_validation_error(client.post(URL, json=augmented))


def test_requires_fixed_checkpoint_version(client):
    request = _import_request()
    for version in (
        "provenance-audit-checkpoint-v2",
        "Provenance-Audit-Checkpoint-V1",
        "",
    ):
        bad = json.loads(json.dumps(request))
        bad["checkpoint"]["checkpoint_version"] = version
        _assert_validation_error(client.post(URL, json=bad))


def test_requires_sha256_digest_algorithm(client):
    request = _import_request()
    for algorithm in ("sha512", "SHA256", "sha-256", ""):
        bad = json.loads(json.dumps(request))
        bad["checkpoint"]["digest_algorithm"] = algorithm
        _assert_validation_error(client.post(URL, json=bad))


def test_requires_64_lowercase_hex_events_digest(client):
    request = _import_request()
    valid = request["checkpoint"]["events_digest_hex"]
    for digest in (
        valid.upper(),
        valid[:-1],
        valid + "0",
        "g" + valid[1:],
        " " + valid,
        123,
        None,
    ):
        bad = json.loads(json.dumps(request))
        bad["checkpoint"]["events_digest_hex"] = digest
        _assert_validation_error(client.post(URL, json=bad))


def test_requires_non_negative_integer_event_count(client):
    request = _import_request()
    for count in (-1, 2.0, 2.5, "2", "two", True, None, [2]):
        bad = json.loads(json.dumps(request))
        bad["checkpoint"]["event_count"] = count
        _assert_validation_error(client.post(URL, json=bad))


def test_events_length_must_equal_event_count(client):
    request = _import_request()

    too_small = json.loads(json.dumps(request))
    too_small["checkpoint"]["event_count"] = len(request["events"]) - 1
    _assert_validation_error(client.post(URL, json=too_small))

    too_large = json.loads(json.dumps(request))
    too_large["checkpoint"]["event_count"] = len(request["events"]) + 1
    _assert_validation_error(client.post(URL, json=too_large))


def test_events_have_exactly_the_three_fields(client):
    request = _import_request()

    for field in ("event_type", "resource_id", "created_at"):
        incomplete = json.loads(json.dumps(request))
        del incomplete["events"][0][field]
        _assert_validation_error(client.post(URL, json=incomplete))

    augmented = json.loads(json.dumps(request))
    augmented["events"][0]["id"] = "evt_1"
    _assert_validation_error(client.post(URL, json=augmented))


def test_event_items_must_be_objects(client):
    request = _import_request()
    for wrong in (None, "event", 42, ["actor.created", "org-1"]):
        bad = json.loads(json.dumps(request))
        bad["events"][0] = wrong
        _assert_validation_error(client.post(URL, json=bad))


def test_event_type_and_resource_id_must_be_non_empty(client):
    request = _import_request()
    for field in ("event_type", "resource_id"):
        for value in ("", "   ", "\t", None, 42):
            bad = json.loads(json.dumps(request))
            bad["events"][0][field] = value
            _assert_validation_error(client.post(URL, json=bad))


def test_created_at_must_be_strict_rfc3339_utc(client):
    request = _import_request()
    for value in (
        "",
        "2026-01-02",
        "2026-01-02T03:04:05",
        "2026-01-02 03:04:05Z",
        "2026-01-02T03:04Z",
        "2026-01-02T03:04:05+01:00",
        "2026-13-02T03:04:05Z",
        "2026-01-02t03:04:05z",
        "not-a-time",
        None,
        42,
    ):
        bad = json.loads(json.dumps(request))
        bad["events"][0]["created_at"] = value
        _assert_validation_error(client.post(URL, json=bad))


def test_malformed_json_body_is_422(client, db_session):
    resp = client.post(
        URL,
        content='{"checkpoint":',
        headers={"content-type": "application/json"},
    )
    _assert_validation_error(resp)
    assert db_session.execute(select(CheckpointImportRecord)).scalars().all() == []
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


# --- Aggregate no-write boundary ------------------------------------------------


def test_every_failed_attempt_writes_nothing(client, db_session):
    good = _import_request()

    mismatch = _import_request()
    mismatch["checkpoint"]["events_digest_hex"] = "0" * 64

    attempts = [
        client.post(URL, json={}),
        client.post(URL, json={"checkpoint": good["checkpoint"]}),
        client.post(
            URL,
            content="{not json",
            headers={"content-type": "application/json"},
        ),
        client.post(URL, json=mismatch),
    ]
    assert [r.status_code for r in attempts] == [422, 422, 422, 422]

    assert db_session.execute(select(CheckpointImportRecord)).scalars().all() == []
    assert db_session.execute(select(AuditEvent)).scalars().all() == []
    for model in _DOMAIN_MODELS:
        assert (
            db_session.execute(
                select(func.count()).select_from(model)
            ).scalar_one()
            == 0
        )

    # After the failures, the valid checkpoint still registers for the
    # first time with exactly one audit event.
    assert client.post(URL, json=good).status_code == 201
    assert len(_import_events(db_session)) == 1


# --- Compatibility with the existing checkpoint interfaces ----------------------


def test_imported_checkpoint_still_verifies_offline(client):
    body = _import_request()

    # The same body is accepted by the import route and still verifies
    # through the stateless verification route, in either order.
    verification = client.post(
        "/v1/audit-events/checkpoint-verifications", json=body
    )
    assert verification.status_code == 200
    assert verification.json() == {"valid": True}

    imported = client.post(URL, json=body)
    assert imported.status_code == 201, imported.text

    # The verification route remains strictly stateless after an import.
    again = client.post(
        "/v1/audit-events/checkpoint-verifications", json=body
    )
    assert again.json() == {"valid": True}


def test_existing_checkpoint_routes_remain_read_only_after_import(client):
    request = _import_request()
    assert client.post(URL, json=request).status_code == 201

    # The read-only checkpoint over the (still empty of described events)
    # audit log is unaffected: importing does not add described events.
    checkpoint = client.get("/v1/audit-events/checkpoint").json()
    listing = client.get("/v1/audit-events").json()
    # Only the import's own audit event is visible to the read routes.
    assert listing["count"] == 1
    assert listing["items"][0]["event_type"] == EVENT_AUDIT_CHECKPOINT_IMPORTED
    assert checkpoint["event_count"] == 1
    assert checkpoint["checkpoint_version"] == CHECKPOINT_VERSION
