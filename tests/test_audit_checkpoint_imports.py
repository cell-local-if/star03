"""Tests for the controlled audit-checkpoint import endpoint.

Covers POST /v1/audit-events/checkpoint-imports and GET
/v1/audit-events/checkpoint-imports/{import_id}. The request body is
exactly ``{"checkpoint", "events"}`` with the same structure as the
stateless verification route: ``checkpoint`` is exactly
{"checkpoint_version", "digest_algorithm", "event_count",
"events_digest_hex"} with the version pinned to
"provenance-audit-checkpoint-v1", the algorithm to "sha256", the count a
non-negative integer, and the digest 64 lowercase hex characters; the
events array has exactly ``event_count`` elements, each exactly
{"event_type", "resource_id", "created_at"} with non-empty strings and a
strict RFC 3339 UTC timestamp. The digest is the SHA-256 of the canonical
event array (array order preserved, object keys sorted by Unicode code
point, compact separators, non-ASCII unescaped, UTF-8 encoded) recomputed
over the events exactly as received.

Verification is decided by the request body alone -- the described events
are never created, modified, or queried, and no local resource needs to
exist. Every structural, field, count, or digest failure is a 422
validation_error and writes nothing. On success the receiving identity is
exactly ``(checkpoint_version, event_count, events_digest_hex)``; the
first submission returns 201 with a stable deterministic ``aci_`` receipt
id, the three identity fields, and a UTC ``received_at``, and writes the
receipt row together with the ``audit.checkpoint_imported`` audit event in
a single transaction. A retry for the same identity returns 200 with the
original receipt and no new audit. GET reads only the receipt fields; an
unknown id is an explicit 404. The full events and raw data are never
persisted or echoed. All fixtures are deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json
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
VERIFY_URL = "/v1/audit-events/checkpoint-verifications"
CHECKPOINT_VERSION = "provenance-audit-checkpoint-v1"
RECEIPT_KEYS = {
    "id",
    "checkpoint_version",
    "events_digest_hex",
    "event_count",
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


def _served_request(client) -> dict:
    """An import body built from the service's own checkpoint and audit views."""
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
    # The database is empty and every identifier is unknown locally: the
    # receipt is decided by the body alone.
    request = _offline_request()
    resp = client.post(URL, json=request)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == RECEIPT_KEYS
    assert body["id"].startswith("aci_")
    assert len(body["id"]) == len("aci_") + 64
    assert body["checkpoint_version"] == CHECKPOINT_VERSION
    assert body["events_digest_hex"] == request["checkpoint"]["events_digest_hex"]
    assert body["event_count"] == len(request["events"])
    # The event array is never echoed back on the receipt.
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


def test_served_checkpoint_imports_when_local_events_exist(client):
    _setup_events(client)
    resp = client.post(URL, json=_served_request(client))
    assert resp.status_code == 201, resp.text
    assert resp.json()["id"].startswith("aci_")
    assert resp.json()["event_count"] == 2


def test_receipt_id_is_the_deterministic_identity_id(client):
    from provenance import ids

    request = _offline_request()
    body = client.post(URL, json=request).json()
    expected = ids.checkpoint_import_id(
        CHECKPOINT_VERSION,
        request["checkpoint"]["event_count"],
        request["checkpoint"]["events_digest_hex"],
    )
    assert body["id"] == expected


def test_distinct_checkpoint_identities_get_distinct_stable_ids(client):
    first = client.post(URL, json=_offline_request()).json()

    other_events = _offline_events()
    other_events.append(
        {
            "event_type": "claim.created",
            "resource_id": "clm-" + hashlib.sha256(b"more").hexdigest(),
            "created_at": "2026-01-02T03:04:07Z",
        }
    )
    second = client.post(URL, json=_offline_request(other_events)).json()

    # A changed sequence changes count and digest, hence the identity.
    assert second["event_count"] != first["event_count"]
    assert second["events_digest_hex"] != first["events_digest_hex"]
    assert second["id"] != first["id"]


def test_empty_sequence_imports(client):
    request = _offline_request(events=[])
    assert request["checkpoint"]["events_digest_hex"] == hashlib.sha256(
        b"[]"
    ).hexdigest()
    resp = client.post(URL, json=request)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["event_count"] == 0
    assert body["events_digest_hex"] == hashlib.sha256(b"[]").hexdigest()


def test_unescaped_non_ascii_participates_in_the_receipt(client):
    request = _offline_request()
    assert any(
        ord(ch) > 127 for ch in json.dumps(request["events"], ensure_ascii=False)
    )
    body = client.post(URL, json=request).json()
    assert body["events_digest_hex"] == _digest_of(request["events"])
    escaped = json.dumps(
        request["events"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    assert body["events_digest_hex"] != hashlib.sha256(escaped).hexdigest()


# --- Idempotency ---------------------------------------------------------------


def test_same_identity_retry_returns_200_unchanged_receipt(client):
    request = _offline_request()
    first = client.post(URL, json=request)
    assert first.status_code == 201
    original = first.json()

    for _ in range(2):
        retry = client.post(URL, json=json.loads(json.dumps(request)))
        assert retry.status_code == 200, retry.text
        assert retry.json() == original


def test_retry_returns_original_received_at(client, db_session):
    request = _offline_request()
    original = client.post(URL, json=request).json()
    retry = client.post(URL, json=request).json()
    assert retry["id"] == original["id"]
    assert retry["received_at"] == original["received_at"]

    records = db_session.execute(select(CheckpointImportRecord)).scalars().all()
    assert [r.id for r in records] == [original["id"]]


def test_retry_writes_no_second_row_or_audit_event(client, db_session):
    request = _offline_request()
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
        len(
            db_session.execute(select(CheckpointImportRecord)).scalars().all()
        )
        == 1
    )


def test_event_order_participates_in_identity(client):
    # The array keeps its order: the same events in another order are a
    # different sequence with a different digest, hence a different receipt.
    request = _offline_request()
    first = client.post(URL, json=request).json()

    reordered = json.loads(json.dumps(request))
    reordered["events"] = list(reversed(reordered["events"]))
    # Recompute the checkpoint over the reordered array: the retried body is
    # internally consistent but commits to a different sequence.
    reordered["checkpoint"]["events_digest_hex"] = _digest_of(reordered["events"])

    second_resp = client.post(URL, json=reordered)
    assert second_resp.status_code == 201, second_resp.text
    second = second_resp.json()
    assert second["id"] != first["id"]
    assert second["events_digest_hex"] != first["events_digest_hex"]

    # Reversed twice is the original sequence again: an idempotent retry.
    assert client.post(URL, json=request).status_code == 200


def test_equivalent_utc_spellings_are_distinct_identities(client):
    # Both UTC designators verify independently, but the spelling participates
    # in the digest exactly as received.
    events_z = _offline_events()
    events_offset = _offline_events()
    events_offset[0]["created_at"] = "2026-01-02T03:04:05+00:00"
    assert _digest_of(events_z) != _digest_of(events_offset)

    first = client.post(URL, json=_offline_request(events_z)).json()
    second = client.post(URL, json=_offline_request(events_offset)).json()
    assert first["id"] != second["id"]


def test_import_is_idempotent_across_app_restart(file_client, tmp_db_url):
    request = _offline_request()
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
    created = client.post(URL, json=_offline_request()).json()
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
    created = client.post(URL, json=_offline_request()).json()
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
    body = client.post(URL, json=_offline_request()).json()
    events = _import_events(db_session)
    assert len(events) == 1
    event = events[0]
    assert event.event_type == EVENT_AUDIT_CHECKPOINT_IMPORTED
    assert event.resource_id == body["id"]
    assert event.created_at.tzinfo.utcoffset(event.created_at).total_seconds() == 0


def test_record_and_audit_event_commit_together(client, db_session):
    resp = client.post(URL, json=_offline_request())
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


# --- Offline independence: described events are never materialized -------------


def test_import_never_depends_on_local_state(client, db_session):
    # A wholly fabricated, self-consistent checkpoint against unknown ids is
    # accepted into an empty database.
    resp = client.post(URL, json=_offline_request())
    assert resp.status_code == 201, resp.text

    # No domain resource was materialized.
    for model in _DOMAIN_MODELS:
        count = db_session.execute(
            select(func.count()).select_from(model)
        ).scalar_one()
        assert count == 0, model


def test_importing_creates_no_described_events(client, db_session):
    request = _offline_request()
    described = {(e["event_type"], e["resource_id"]) for e in request["events"]}

    resp = client.post(URL, json=request)
    assert resp.status_code == 201

    # Exactly one receipt plus one audit event (the import's own); neither
    # described event exists in the local audit trail.
    all_events = db_session.execute(select(AuditEvent)).scalars().all()
    assert len(all_events) == 1
    assert (all_events[0].event_type, all_events[0].resource_id) not in described
    assert all_events[0].event_type == EVENT_AUDIT_CHECKPOINT_IMPORTED
    assert (
        db_session.execute(
            select(func.count()).select_from(CheckpointImportRecord)
        ).scalar_one()
        == 1
    )


def test_receipt_table_persists_no_events_column(client, db_session):
    # The stored record has exactly the identity and bookkeeping columns;
    # there is nowhere an event array or raw event value could be persisted.
    marker = "events-persistence-marker-7e4a"
    request = _offline_request()
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


def test_import_does_not_change_served_checkpoint_of_described_events(client):
    # Registering a checkpoint never creates the described events: a served
    # checkpoint over local state stays empty after importing an offline one.
    empty_digest = client.get("/v1/audit-events/checkpoint").json()[
        "events_digest_hex"
    ]
    assert client.post(URL, json=_offline_request()).status_code == 201

    # The only new local event is the import's own audit event; the events
    # described by the imported checkpoint are absent from the local trail.
    items = client.get("/v1/audit-events").json()["items"]
    assert [i["event_type"] for i in items] == [EVENT_AUDIT_CHECKPOINT_IMPORTED]
    assert empty_digest != client.get("/v1/audit-events/checkpoint").json()[
        "events_digest_hex"
    ]


# --- Digest mismatch -----------------------------------------------------------


def test_digest_mismatch_is_422_and_writes_nothing(client, db_session):
    request = _offline_request()
    claimed = request["checkpoint"]["events_digest_hex"]
    request["checkpoint"]["events_digest_hex"] = (
        claimed[:-1] + ("0" if claimed[-1] != "0" else "1")
    )

    resp = client.post(URL, json=request)
    assert resp.status_code == 422, resp.text
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"]["reason"] == "events_digest_mismatch"
    assert error["details"]["computed_digest_hex"] == _digest_of(
        request["events"]
    )

    assert (
        db_session.execute(select(CheckpointImportRecord)).scalars().all() == []
    )
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_tampered_event_is_422_and_writes_nothing(client, db_session):
    request = _offline_request()
    # The checkpoint digest still commits to the untouched sequence.
    request["events"][0]["resource_id"] = "org-forged"

    _assert_validation_error(client.post(URL, json=request))
    assert (
        db_session.execute(select(CheckpointImportRecord)).scalars().all() == []
    )
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_reordered_events_without_recomputed_digest_is_422(client):
    request = _offline_request()
    reordered = json.loads(json.dumps(request))
    reordered["events"] = list(reversed(reordered["events"]))
    _assert_validation_error(client.post(URL, json=reordered))


def test_tampered_retry_under_existing_identity_is_422_and_keeps_record(
    client, db_session
):
    request = _offline_request()
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
        len(
            db_session.execute(select(CheckpointImportRecord)).scalars().all()
        )
        == 1
    )
    assert len(_import_events(db_session)) == 1


def test_mismatch_does_not_query_local_state(client):
    # A digest mismatch is rejected on the body alone in an empty database;
    # no audit-event lookup could change it.
    request = _offline_request()
    request["checkpoint"]["events_digest_hex"] = "0" * 64
    _assert_validation_error(client.post(URL, json=request))


# --- Structural / field / count boundary --------------------------------------


def test_body_requires_exactly_checkpoint_and_events(client):
    request = _offline_request()

    for field in ("checkpoint", "events"):
        incomplete = {k: v for k, v in request.items() if k != field}
        _assert_validation_error(client.post(URL, json=incomplete))

    augmented = dict(request)
    augmented["computed_digest_hex"] = request["checkpoint"]["events_digest_hex"]
    _assert_validation_error(client.post(URL, json=augmented))


def test_non_object_bodies_are_422(client):
    for body in (None, [], "checkpoint", 42):
        _assert_validation_error(client.post(URL, json=body))
    _assert_validation_error(client.post(URL, json={}))


def test_members_have_correct_types(client):
    request = _offline_request()
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
    request = _offline_request()

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
    request = _offline_request()
    for version in (
        "provenance-audit-checkpoint-v2",
        "Provenance-Audit-Checkpoint-V1",
        "",
    ):
        bad = json.loads(json.dumps(request))
        bad["checkpoint"]["checkpoint_version"] = version
        _assert_validation_error(client.post(URL, json=bad))


def test_requires_sha256_digest_algorithm(client):
    request = _offline_request()
    for algorithm in ("sha512", "SHA256", "sha-256", ""):
        bad = json.loads(json.dumps(request))
        bad["checkpoint"]["digest_algorithm"] = algorithm
        _assert_validation_error(client.post(URL, json=bad))


def test_requires_64_lowercase_hex_events_digest(client):
    request = _offline_request()
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
    request = _offline_request()
    for count in (-1, 2.0, 2.5, "2", "two", True, None, [2]):
        bad = json.loads(json.dumps(request))
        bad["checkpoint"]["event_count"] = count
        _assert_validation_error(client.post(URL, json=bad))


def test_events_length_must_equal_event_count(client):
    request = _offline_request()

    too_small = json.loads(json.dumps(request))
    too_small["checkpoint"]["event_count"] = len(request["events"]) - 1
    _assert_validation_error(client.post(URL, json=too_small))

    too_large = json.loads(json.dumps(request))
    too_large["checkpoint"]["event_count"] = len(request["events"]) + 1
    _assert_validation_error(client.post(URL, json=too_large))


def test_events_have_exactly_the_three_fields(client):
    request = _offline_request()

    for field in ("event_type", "resource_id", "created_at"):
        incomplete = json.loads(json.dumps(request))
        del incomplete["events"][0][field]
        _assert_validation_error(client.post(URL, json=incomplete))

    augmented = json.loads(json.dumps(request))
    augmented["events"][0]["id"] = "evt_1"
    _assert_validation_error(client.post(URL, json=augmented))


def test_event_items_must_be_objects(client):
    request = _offline_request()
    for wrong in (None, "event", 42, ["actor.created", "org-1"]):
        bad = json.loads(json.dumps(request))
        bad["events"][0] = wrong
        _assert_validation_error(client.post(URL, json=bad))


def test_event_type_and_resource_id_must_be_non_empty(client):
    request = _offline_request()
    for field in ("event_type", "resource_id"):
        for value in ("", "   ", "\t", None, 42):
            bad = json.loads(json.dumps(request))
            bad["events"][0][field] = value
            _assert_validation_error(client.post(URL, json=bad))


def test_created_at_must_be_strict_rfc3339_utc(client):
    request = _offline_request()
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
    assert (
        db_session.execute(select(CheckpointImportRecord)).scalars().all() == []
    )
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_every_failed_attempt_writes_nothing(client, db_session):
    good = _offline_request()

    mismatch = _offline_request()
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

    assert (
        db_session.execute(select(CheckpointImportRecord)).scalars().all() == []
    )
    assert db_session.execute(select(AuditEvent)).scalars().all() == []
    for model in _DOMAIN_MODELS:
        assert (
            db_session.execute(
                select(func.count()).select_from(model)
            ).scalar_one()
            == 0
        )

    # After the failures, the valid checkpoint still registers for the first
    # time with exactly one audit event.
    assert client.post(URL, json=good).status_code == 201
    assert len(_import_events(db_session)) == 1


# --- Compatibility with the existing checkpoint routes -------------------------


def test_imported_checkpoint_still_verifies_offline(client):
    request = _offline_request()

    # The same checkpoint is accepted by the import route and still verifies
    # through the stateless verification route, in either order.
    verification = client.post(VERIFY_URL, json=request)
    assert verification.status_code == 200
    assert verification.json() == {"valid": True}

    imported = client.post(URL, json=request)
    assert imported.status_code == 201, imported.text

    again = client.post(VERIFY_URL, json=request)
    assert again.json() == {"valid": True}


def test_served_checkpoint_round_trips_through_import_without_new_events(client):
    _setup_events(client)

    # The locally served checkpoint + event listing verifies, imports once
    # (201; the import's own audit event joins the trail), and the same
    # request re-presented from the pre-import view is an idempotent 200.
    request = _served_request(client)
    assert client.post(VERIFY_URL, json=request).json() == {"valid": True}
    first = client.post(URL, json=request)
    assert first.status_code == 201, first.text
    assert first.json()["event_count"] == 2
    assert client.post(URL, json=request).status_code == 200

    # The trail gained exactly the import audit event; re-importing the
    # checkpoint that now includes that event is a fresh identity.
    grown = _served_request(client)
    assert grown["checkpoint"]["event_count"] == 3
    assert grown["checkpoint"]["events_digest_hex"] != request["checkpoint"][
        "events_digest_hex"
    ]
    grown_import = client.post(URL, json=grown)
    assert grown_import.status_code == 201
    assert grown_import.json()["id"] != first.json()["id"]
