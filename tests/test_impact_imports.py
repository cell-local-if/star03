"""Tests for the controlled revocation-impact import endpoint.

Covers POST /v1/impact-imports and GET /v1/impact-imports/{import_id}. The
request body is exactly ``{"checkpoint", "impacts"}`` with the same
structure as the stateless verification route: ``checkpoint`` is exactly
{"checkpoint_version", "digest_algorithm", "impact_count",
"impacts_digest_hex"} with the version pinned to
"provenance-revocation-impact-checkpoint-v1", the algorithm to "sha256",
the count a non-negative integer, and the digest 64 lowercase hex
characters; the impacts array has exactly ``impact_count`` elements, each
exactly the exported 13-field impact public view with internally
consistent before/after coverage fields. The digest is the SHA-256 of the
canonical impacts array (array order preserved, nested object keys sorted
by Unicode code point, compact separators, non-ASCII unescaped, UTF-8
encoded) recomputed over the impacts exactly as received.

Verification is decided by the request body alone -- the described impacts
are never created, modified, or queried, and no local resource needs to
exist. Every structural, field, count, or digest failure is a 422
validation_error and writes nothing. On success the receiving identity is
exactly ``(checkpoint_version, impact_count, impacts_digest_hex)``; the
first submission returns 201 with a stable deterministic ``rii_`` receipt
id, the three identity fields, and a UTC ``received_at``, and writes the
receipt row together with the ``revocation_impact.imported`` audit event
in a single transaction. A retry for the same identity returns 200 with
the original receipt and no new audit. GET reads only the receipt fields;
an unknown id is an explicit 404 and any query parameter is a 422. The
full impacts and raw data are never persisted or echoed. All fixtures are
deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance import ids
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import (
    EVENT_REVOCATION_IMPACT_IMPORTED,
    Actor,
    Attestation,
    AttestationRevocation,
    AuditEvent,
    Claim,
    Content,
    ContentRelation,
    EvidenceBundle,
    RevocationImpactImportRecord,
)
from tests.test_revocation_impact_verifications import (
    _digest_of,
    _impact,
    _offline_impacts,
    _request,
)
from tests.test_revocation_impacts import _world

URL = "/v1/impact-imports"
VERIFY_URL = "/v1/impact-verifications"
PACKAGE_PATH = "/v1/revocation-impact-package"
CHECKPOINT_VERSION = "provenance-revocation-impact-checkpoint-v1"
RECEIPT_KEYS = {
    "id",
    "checkpoint_version",
    "impact_count",
    "impacts_digest_hex",
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


def _served_request(client) -> dict:
    """An import body built from the service's own exported package."""
    package = client.get(PACKAGE_PATH).json()
    return {"checkpoint": package["checkpoint"], "impacts": package["impacts"]}


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def _import_events(db_session):
    return (
        db_session.execute(
            select(AuditEvent)
            .where(AuditEvent.event_type == EVENT_REVOCATION_IMPACT_IMPORTED)
            .order_by(AuditEvent.seq.asc())
        )
        .scalars()
        .all()
    )


# --- First registration: success shape -----------------------------------------


def test_offline_checkpoint_first_import_returns_201_receipt(client, db_session):
    # The database is empty and every identifier is unknown locally: the
    # receipt is decided by the body alone.
    request = _request()
    resp = client.post(URL, json=request)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == RECEIPT_KEYS
    assert body["id"].startswith("rii_")
    assert len(body["id"]) == len("rii_") + 64
    assert body["checkpoint_version"] == CHECKPOINT_VERSION
    assert body["impacts_digest_hex"] == request["checkpoint"][
        "impacts_digest_hex"
    ]
    assert body["impact_count"] == len(request["impacts"])
    # The impacts array is never echoed back on the receipt.
    assert "impacts" not in body
    assert "checkpoint" not in body
    assert "digest_algorithm" not in body

    received_at = datetime.fromisoformat(body["received_at"])
    assert received_at.utcoffset().total_seconds() == 0
    assert body["received_at"].endswith(("Z", "+00:00"))

    # Exactly one receipt row exists with the same identity.
    records = (
        db_session.execute(select(RevocationImpactImportRecord))
        .scalars()
        .all()
    )
    assert len(records) == 1
    (record,) = records
    assert record.id == body["id"]
    assert record.checkpoint_version == CHECKPOINT_VERSION
    assert record.impact_count == body["impact_count"]
    assert record.impacts_digest_hex == body["impacts_digest_hex"]


def test_served_package_imports_when_local_impacts_exist(client):
    _world(client)
    resp = client.post(URL, json=_served_request(client))
    assert resp.status_code == 201, resp.text
    assert resp.json()["id"].startswith("rii_")
    assert resp.json()["impact_count"] == 6


def test_receipt_id_is_the_deterministic_identity_id(client):
    request = _request()
    body = client.post(URL, json=request).json()
    expected = ids.revocation_impact_import_id(
        CHECKPOINT_VERSION,
        request["checkpoint"]["impact_count"],
        request["checkpoint"]["impacts_digest_hex"],
    )
    assert body["id"] == expected


def test_distinct_checkpoint_identities_get_distinct_stable_ids(client):
    first = client.post(URL, json=_request()).json()

    other_impacts = _offline_impacts()
    other_impacts.append(
        _impact(
            id="rev_01hztest000000000000000004",
            attestation_id="att_01hztest00000000000000004",
            content_id="cnt_01hztest00000000000000004",
            created_at="2026-01-03T03:04:05Z",
        )
    )
    second = client.post(URL, json=_request(other_impacts)).json()

    # A changed sequence changes count and digest, hence the identity.
    assert second["impact_count"] != first["impact_count"]
    assert second["impacts_digest_hex"] != first["impacts_digest_hex"]
    assert second["id"] != first["id"]


def test_empty_sequence_imports(client):
    request = _request(impacts=[])
    assert request["checkpoint"]["impacts_digest_hex"] == hashlib.sha256(
        b"[]"
    ).hexdigest()
    resp = client.post(URL, json=request)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["impact_count"] == 0
    assert body["impacts_digest_hex"] == hashlib.sha256(b"[]").hexdigest()


def test_unescaped_non_ascii_participates_in_the_receipt(client):
    request = _request()
    assert any(
        ord(ch) > 127
        for ch in json.dumps(request["impacts"], ensure_ascii=False)
    )
    body = client.post(URL, json=request).json()
    assert body["impacts_digest_hex"] == _digest_of(request["impacts"])
    escaped = json.dumps(
        request["impacts"], sort_keys=True, separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    assert body["impacts_digest_hex"] != hashlib.sha256(escaped).hexdigest()


# --- Idempotency ---------------------------------------------------------------


def test_same_identity_retry_returns_200_unchanged_receipt(client):
    request = _request()
    first = client.post(URL, json=request)
    assert first.status_code == 201
    original = first.json()

    for _ in range(2):
        retry = client.post(URL, json=json.loads(json.dumps(request)))
        assert retry.status_code == 200, retry.text
        assert retry.json() == original


def test_retry_returns_original_received_at(client, db_session):
    request = _request()
    original = client.post(URL, json=request).json()
    retry = client.post(URL, json=request).json()
    assert retry["id"] == original["id"]
    assert retry["received_at"] == original["received_at"]

    records = (
        db_session.execute(select(RevocationImpactImportRecord))
        .scalars()
        .all()
    )
    assert [r.id for r in records] == [original["id"]]


def test_retry_writes_no_second_row_or_audit_event(client, db_session):
    request = _request()
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
            db_session.execute(select(RevocationImpactImportRecord))
            .scalars()
            .all()
        )
        == 1
    )


def test_impact_order_participates_in_identity(client):
    # The array keeps its order: the same impacts in another order are a
    # different sequence with a different digest, hence a different receipt.
    request = _request()
    first = client.post(URL, json=request).json()

    reordered = json.loads(json.dumps(request))
    reordered["impacts"] = list(reversed(reordered["impacts"]))
    # Recompute the checkpoint over the reordered array: the retried body is
    # internally consistent but commits to a different sequence.
    reordered["checkpoint"]["impacts_digest_hex"] = _digest_of(
        reordered["impacts"]
    )

    second_resp = client.post(URL, json=reordered)
    assert second_resp.status_code == 201, second_resp.text
    second = second_resp.json()
    assert second["id"] != first["id"]
    assert second["impacts_digest_hex"] != first["impacts_digest_hex"]

    # Reversed twice is the original sequence again: an idempotent retry.
    assert client.post(URL, json=request).status_code == 200


def test_equivalent_utc_spellings_are_distinct_identities(client):
    # Both UTC designators verify independently, but the spelling participates
    # in the digest exactly as received.
    impacts_z = _offline_impacts()
    impacts_offset = _offline_impacts()
    impacts_offset[0]["created_at"] = "2026-01-02T03:04:05+00:00"
    assert _digest_of(impacts_z) != _digest_of(impacts_offset)

    first = client.post(URL, json=_request(impacts_z)).json()
    second = client.post(URL, json=_request(impacts_offset)).json()
    assert first["id"] != second["id"]


def test_import_is_idempotent_across_app_restart(file_client, tmp_db_url):
    request = _request()
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
    created = client.post(URL, json=_request()).json()
    resp = client.get(f"{URL}/{created['id']}")
    assert resp.status_code == 200, resp.text
    assert resp.json() == created
    assert set(resp.json()) == RECEIPT_KEYS


def test_get_unknown_id_is_an_explicit_specific_404(client, db_session):
    unknown = "rii_" + "0" * 64
    resp = client.get(f"{URL}/{unknown}")
    assert resp.status_code == 404, resp.text
    error = resp.json()["error"]
    assert error["code"] == "impact_import_not_found"
    assert error["details"]["import_id"] == unknown
    # The failed read wrote nothing.
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_get_unshaped_and_other_prefix_ids_are_404(client):
    for raw_id in ("rii_doesnotexist", "aci_" + "a" * 64, "not-an-id"):
        resp = client.get(f"{URL}/{raw_id}")
        assert resp.status_code == 404, raw_id
        assert resp.json()["error"]["code"] == "impact_import_not_found"


def test_get_any_query_parameter_is_422(client):
    created = client.post(URL, json=_request()).json()
    base = f"{URL}/{created['id']}"
    for url in (
        f"{base}?limit=10",
        f"{base}?cursor=abc",
        f"{base}?unknown=",
        f"{base}?a=1&b=2",
        f"{base}?a=1&a=2",
    ):
        resp = client.get(url)
        assert resp.status_code == 422, (url, resp.text)
        assert resp.json()["error"]["code"] == "validation_error"


def test_get_query_parameter_is_422_before_the_receipt_lookup(client):
    # Parameters are validated before any existence lookup: a malformed
    # request is a 422 even when the receipt id is also unknown.
    resp = client.get(f"{URL}/rii_doesnotexist?anything=1")
    assert resp.status_code == 422, resp.text
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"]["issues"][0]["loc"] == ["query", "anything"]


def test_get_is_read_only(client, db_session):
    created = client.post(URL, json=_request()).json()
    events_after_import = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )
    assert client.get(f"{URL}/{created['id']}").status_code == 200
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_after_import
    )


# --- Audit transaction ---------------------------------------------------------


def test_first_import_writes_one_impact_imported_audit_event(
    client, db_session
):
    body = client.post(URL, json=_request()).json()
    events = _import_events(db_session)
    assert len(events) == 1
    event = events[0]
    assert event.event_type == EVENT_REVOCATION_IMPACT_IMPORTED
    assert event.resource_id == body["id"]
    assert event.created_at.tzinfo.utcoffset(event.created_at).total_seconds() == 0


def test_record_and_audit_event_commit_together(client, db_session):
    resp = client.post(URL, json=_request())
    assert resp.status_code == 201
    receipt_id = resp.json()["id"]

    # Both the receipt and its audit event are visible to another session
    # immediately: they committed in the same transaction.
    db_session.expire_all()
    record = db_session.execute(
        select(RevocationImpactImportRecord).where(
            RevocationImpactImportRecord.id == receipt_id
        )
    ).scalar_one()
    event = db_session.execute(
        select(AuditEvent).where(AuditEvent.resource_id == receipt_id)
    ).scalar_one()
    assert record is not None
    assert event.event_type == EVENT_REVOCATION_IMPACT_IMPORTED


# --- Offline independence: described impacts are never materialized ------------


def test_import_never_depends_on_local_state(client, db_session):
    # A wholly fabricated, self-consistent checkpoint against unknown ids is
    # accepted into an empty database.
    resp = client.post(URL, json=_request())
    assert resp.status_code == 201, resp.text

    # No domain resource was materialized.
    for model in _DOMAIN_MODELS:
        count = db_session.execute(
            select(func.count()).select_from(model)
        ).scalar_one()
        assert count == 0, model


def test_importing_creates_no_described_revocations(client, db_session):
    request = _request()
    described = {impact["id"] for impact in request["impacts"]}

    resp = client.post(URL, json=request)
    assert resp.status_code == 201

    # Exactly one receipt plus one audit event (the import's own); no
    # revocation described by the imported impacts exists locally.
    all_events = db_session.execute(select(AuditEvent)).scalars().all()
    assert len(all_events) == 1
    assert all_events[0].event_type == EVENT_REVOCATION_IMPACT_IMPORTED
    assert all_events[0].resource_id not in described
    assert (
        db_session.execute(
            select(func.count()).select_from(AttestationRevocation)
        ).scalar_one()
        == 0
    )
    assert (
        db_session.execute(
            select(func.count()).select_from(RevocationImpactImportRecord)
        ).scalar_one()
        == 1
    )


def test_receipt_table_persists_no_impacts_column(client, db_session):
    # The stored record has exactly the identity and bookkeeping columns;
    # there is nowhere an impacts array or raw impact value could persist.
    marker = "impacts-persistence-marker-7e4a"
    request = _request()
    request["impacts"][0]["reason"] = marker
    request["checkpoint"]["impacts_digest_hex"] = _digest_of(
        request["impacts"]
    )
    assert client.post(URL, json=request).status_code == 201

    columns = set(RevocationImpactImportRecord.__table__.columns.keys())
    assert columns == {
        "seq",
        "id",
        "checkpoint_version",
        "impact_count",
        "impacts_digest_hex",
        "created_at",
    }

    record = db_session.execute(
        select(RevocationImpactImportRecord)
    ).scalar_one()
    assert marker not in repr(vars(record))


def test_import_does_not_change_served_package(client):
    # Registering a checkpoint never creates the described revocations: the
    # exported package over local state stays the empty snapshot after
    # importing an offline one.
    empty_package = client.get(PACKAGE_PATH).json()
    assert empty_package["checkpoint"]["impact_count"] == 0
    assert client.post(URL, json=_request()).status_code == 201

    grown = client.get(PACKAGE_PATH).json()
    assert grown == empty_package


# --- Digest mismatch -----------------------------------------------------------


def test_digest_mismatch_is_422_and_writes_nothing(client, db_session):
    request = _request()
    claimed = request["checkpoint"]["impacts_digest_hex"]
    request["checkpoint"]["impacts_digest_hex"] = claimed[:-1] + (
        "0" if claimed[-1] != "0" else "1"
    )

    resp = client.post(URL, json=request)
    assert resp.status_code == 422, resp.text
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"]["reason"] == "impacts_digest_mismatch"
    assert error["details"]["computed_digest_hex"] == _digest_of(
        request["impacts"]
    )

    assert (
        db_session.execute(select(RevocationImpactImportRecord))
        .scalars()
        .all()
        == []
    )
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_tampered_impact_is_422_and_writes_nothing(client, db_session):
    request = _request()
    # The checkpoint digest still commits to the untouched sequence.
    request["impacts"][0]["reason"] = "forged reason"

    _assert_validation_error(client.post(URL, json=request))
    assert (
        db_session.execute(select(RevocationImpactImportRecord))
        .scalars()
        .all()
        == []
    )
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_reordered_impacts_without_recomputed_digest_is_422(client):
    request = _request()
    reordered = json.loads(json.dumps(request))
    reordered["impacts"] = list(reversed(reordered["impacts"]))
    _assert_validation_error(client.post(URL, json=reordered))


def test_tampered_retry_under_existing_identity_is_422_and_keeps_record(
    client, db_session
):
    request = _request()
    original = client.post(URL, json=request)
    assert original.status_code == 201

    # Same identity fields, but the presented impacts no longer match:
    # validation beats the idempotent lookup and the original survives.
    tampered = json.loads(json.dumps(request))
    tampered["impacts"][0]["reason"] = "forged"
    _assert_validation_error(client.post(URL, json=tampered))

    assert (
        client.get(f"{URL}/{original.json()['id']}").json() == original.json()
    )
    assert (
        len(
            db_session.execute(select(RevocationImpactImportRecord))
            .scalars()
            .all()
        )
        == 1
    )
    assert len(_import_events(db_session)) == 1


def test_mismatch_does_not_query_local_state(client):
    # A digest mismatch is rejected on the body alone in an empty database;
    # no revocation lookup could change it.
    request = _request()
    request["checkpoint"]["impacts_digest_hex"] = "0" * 64
    _assert_validation_error(client.post(URL, json=request))


# --- Structural / field / count boundary ---------------------------------------


def test_body_requires_exactly_checkpoint_and_impacts(client):
    request = _request()

    for field in ("checkpoint", "impacts"):
        incomplete = {k: v for k, v in request.items() if k != field}
        _assert_validation_error(client.post(URL, json=incomplete))

    augmented = dict(request)
    augmented["computed_digest_hex"] = request["checkpoint"][
        "impacts_digest_hex"
    ]
    _assert_validation_error(client.post(URL, json=augmented))


def test_non_object_bodies_are_422(client):
    for body in (None, [], "checkpoint", 42):
        _assert_validation_error(client.post(URL, json=body))
    _assert_validation_error(client.post(URL, json={}))


def test_members_have_correct_types(client):
    request = _request()
    for member, wrong in (
        ("checkpoint", []),
        ("checkpoint", None),
        ("impacts", {}),
        ("impacts", "impacts"),
    ):
        bad = json.loads(json.dumps(request))
        bad[member] = wrong
        _assert_validation_error(client.post(URL, json=bad))


def test_checkpoint_has_exactly_the_four_members(client):
    request = _request()

    for member in (
        "checkpoint_version",
        "digest_algorithm",
        "impact_count",
        "impacts_digest_hex",
    ):
        incomplete = json.loads(json.dumps(request))
        del incomplete["checkpoint"][member]
        _assert_validation_error(client.post(URL, json=incomplete))

    augmented = json.loads(json.dumps(request))
    augmented["checkpoint"]["limit"] = 50
    _assert_validation_error(client.post(URL, json=augmented))


def test_requires_fixed_checkpoint_version(client):
    request = _request()
    for version in (
        "provenance-revocation-impact-checkpoint-v2",
        "Provenance-Revocation-Impact-Checkpoint-V1",
        "",
    ):
        bad = json.loads(json.dumps(request))
        bad["checkpoint"]["checkpoint_version"] = version
        _assert_validation_error(client.post(URL, json=bad))


def test_requires_sha256_digest_algorithm(client):
    request = _request()
    for algorithm in ("sha512", "SHA256", "sha-256", ""):
        bad = json.loads(json.dumps(request))
        bad["checkpoint"]["digest_algorithm"] = algorithm
        _assert_validation_error(client.post(URL, json=bad))


def test_requires_64_lowercase_hex_impacts_digest(client):
    request = _request()
    valid = request["checkpoint"]["impacts_digest_hex"]
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
        bad["checkpoint"]["impacts_digest_hex"] = digest
        _assert_validation_error(client.post(URL, json=bad))


def test_requires_non_negative_integer_impact_count(client):
    request = _request()
    for count in (-1, 3.0, 3.5, "3", "three", True, None, [3]):
        bad = json.loads(json.dumps(request))
        bad["checkpoint"]["impact_count"] = count
        _assert_validation_error(client.post(URL, json=bad))


def test_impacts_length_must_equal_impact_count(client):
    request = _request()

    too_small = json.loads(json.dumps(request))
    too_small["checkpoint"]["impact_count"] = len(request["impacts"]) - 1
    _assert_validation_error(client.post(URL, json=too_small))

    too_large = json.loads(json.dumps(request))
    too_large["checkpoint"]["impact_count"] = len(request["impacts"]) + 1
    _assert_validation_error(client.post(URL, json=too_large))


def test_impacts_have_exactly_the_thirteen_fields(client):
    request = _request()

    for field in (
        "id",
        "attestation_id",
        "revoker_actor_id",
        "reason",
        "created_at",
        "content_id",
        "target_type",
        "signer_actor_id",
        "qualified_signer_count_after",
        "coverage_status_after",
        "qualified_signer_count_before",
        "coverage_status_before",
        "qualified_signer_count_delta",
    ):
        incomplete = json.loads(json.dumps(request))
        del incomplete["impacts"][0][field]
        _assert_validation_error(client.post(URL, json=incomplete))

    augmented = json.loads(json.dumps(request))
    augmented["impacts"][0]["payload"] = {}
    _assert_validation_error(client.post(URL, json=augmented))


def test_impact_items_must_be_objects(client):
    request = _request()
    for wrong in (None, "impact", 42, ["rev_1"]):
        bad = json.loads(json.dumps(request))
        bad["impacts"][0] = wrong
        _assert_validation_error(client.post(URL, json=bad))


def test_impact_identifiers_and_reason_must_be_non_empty(client):
    request = _request()
    for field in (
        "id",
        "attestation_id",
        "revoker_actor_id",
        "content_id",
        "signer_actor_id",
        "reason",
    ):
        for value in ("", "   ", None, 42):
            bad = json.loads(json.dumps(request))
            bad["impacts"][0][field] = value
            _assert_validation_error(client.post(URL, json=bad))


def test_impact_created_at_must_be_strict_rfc3339_utc(client):
    request = _request()
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
        bad["impacts"][0]["created_at"] = value
        _assert_validation_error(client.post(URL, json=bad))


def test_impact_coverage_fields_must_be_internally_consistent(client):
    request = _request()

    # Delta must equal before - after.
    bad = json.loads(json.dumps(request))
    bad["impacts"][0]["qualified_signer_count_delta"] = 0
    _assert_validation_error(client.post(URL, json=bad))

    # After must not exceed before.
    bad = json.loads(json.dumps(request))
    bad["impacts"][0]["qualified_signer_count_after"] = (
        bad["impacts"][0]["qualified_signer_count_before"] + 1
    )
    _assert_validation_error(client.post(URL, json=bad))

    # A positive count must pair with "covered".
    bad = json.loads(json.dumps(request))
    bad["impacts"][0]["coverage_status_before"] = "partial"
    _assert_validation_error(client.post(URL, json=bad))

    # A zero count must not pair with "covered".
    bad = json.loads(json.dumps(request))
    bad["impacts"][0]["coverage_status_after"] = "covered"
    _assert_validation_error(client.post(URL, json=bad))


def test_malformed_json_body_is_422(client, db_session):
    resp = client.post(
        URL,
        content='{"checkpoint":',
        headers={"content-type": "application/json"},
    )
    _assert_validation_error(resp)
    assert (
        db_session.execute(select(RevocationImpactImportRecord))
        .scalars()
        .all()
        == []
    )
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_every_failed_attempt_writes_nothing(client, db_session):
    good = _request()

    mismatch = _request()
    mismatch["checkpoint"]["impacts_digest_hex"] = "0" * 64

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
        db_session.execute(select(RevocationImpactImportRecord))
        .scalars()
        .all()
        == []
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


# --- Compatibility with the existing impact routes -----------------------------


def test_imported_checkpoint_still_verifies_offline(client):
    request = _request()

    # The same checkpoint is accepted by the import route and still verifies
    # through the stateless verification route, in either order.
    verification = client.post(VERIFY_URL, json=request)
    assert verification.status_code == 200
    assert verification.json() == {"valid": True}

    imported = client.post(URL, json=request)
    assert imported.status_code == 201, imported.text

    again = client.post(VERIFY_URL, json=request)
    assert again.json() == {"valid": True}


def test_served_package_round_trips_through_import(client):
    _world(client)

    # The locally exported package verifies, imports once (201), and the
    # same request re-presented is an idempotent 200; unlike the audit
    # checkpoint, the import's own audit event is not an impact, so the
    # exported package is unchanged by the import.
    request = _served_request(client)
    assert client.post(VERIFY_URL, json=request).json() == {"valid": True}
    first = client.post(URL, json=request)
    assert first.status_code == 201, first.text
    assert first.json()["impact_count"] == 6
    assert client.post(URL, json=request).status_code == 200
    assert client.get(PACKAGE_PATH).json()["checkpoint"] == request[
        "checkpoint"
    ]
