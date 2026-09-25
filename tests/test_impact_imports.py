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
canonical impacts array (array order preserved, object keys sorted by
Unicode code point, compact separators, non-ASCII unescaped, UTF-8
encoded) recomputed over the impacts exactly as received.

Verification is decided by the request body alone -- the described impacts
are never created, modified, or queried, and no local resource needs to
exist. Every structural, field, association, count, or digest failure is a
422 validation_error and writes nothing. On success the receiving identity
is exactly ``(checkpoint_version, impact_count, impacts_digest_hex)``; the
first submission returns 201 with a stable deterministic ``rii_`` receipt
id, the three identity fields, and a UTC ``received_at``, and writes the
receipt row together with the ``revocation_impact.imported`` audit event
in a single transaction. A retry for the same identity returns 200 with
the original receipt and no new audit. GET reads only the receipt fields;
an unknown id is an explicit 404. The full impacts and raw data are never
persisted or echoed. All fixtures are deterministic and offline.
"""

from __future__ import annotations

import concurrent.futures
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
    Claim,
    Content,
    ContentRelation,
    EvidenceBundle,
    ImpactImportRecord,
)
from provenance.models import EVENT_REVOCATION_IMPACT_IMPORTED
from tests.test_revocation_impacts import _world

URL = "/v1/impact-imports"
VERIFY_URL = "/v1/impact-verifications"
PACKAGE_URL = "/v1/revocation-impact-package"
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


def _digest_of(impacts: list) -> str:
    """Independently canonicalize the impact array and digest it."""
    canonical = json.dumps(
        impacts,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _offline_impacts() -> list:
    """A self-contained impact sequence fabricated without any service state."""
    return [
        {
            "id": "rev_" + hashlib.sha256(b"offline-rev-1").hexdigest(),
            "attestation_id": "att_"
            + hashlib.sha256(b"offline-att-1").hexdigest(),
            "revoker_actor_id": "org-émoji-✓",
            "reason": "superseded by a rotated key",
            "created_at": "2026-01-02T03:04:05Z",
            "content_id": "cnt_" + hashlib.sha256(b"offline-c1").hexdigest(),
            "target_type": "claim",
            "signer_actor_id": "org-signer-1",
            "qualified_signer_count_after": 0,
            "coverage_status_after": "partial",
            "qualified_signer_count_before": 1,
            "coverage_status_before": "covered",
            "qualified_signer_count_delta": 1,
        },
        {
            "id": "rev_" + hashlib.sha256(b"offline-rev-2").hexdigest(),
            "attestation_id": "att_"
            + hashlib.sha256(b"offline-att-2").hexdigest(),
            "revoker_actor_id": "org-émoji-✓",
            "reason": "duplicate proof",
            "created_at": "2026-01-02T03:04:06.500Z",
            "content_id": "cnt_" + hashlib.sha256(b"offline-c2").hexdigest(),
            "target_type": "evidence_bundle",
            "signer_actor_id": "org-signer-2",
            "qualified_signer_count_after": 1,
            "coverage_status_after": "covered",
            "qualified_signer_count_before": 1,
            "coverage_status_before": "covered",
            "qualified_signer_count_delta": 0,
        },
    ]


def _offline_request(impacts: list | None = None) -> dict:
    impacts = _offline_impacts() if impacts is None else impacts
    return {
        "checkpoint": {
            "checkpoint_version": CHECKPOINT_VERSION,
            "digest_algorithm": "sha256",
            "impact_count": len(impacts),
            "impacts_digest_hex": _digest_of(impacts),
        },
        "impacts": impacts,
    }


def _served_request(client) -> dict:
    """An import body built from the service's own package export."""
    resp = client.get(PACKAGE_URL)
    assert resp.status_code == 200, resp.text
    return resp.json()


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
    request = _offline_request()
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
    records = db_session.execute(select(ImpactImportRecord)).scalars().all()
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
    from provenance import ids

    request = _offline_request()
    body = client.post(URL, json=request).json()
    expected = ids.impact_import_id(
        CHECKPOINT_VERSION,
        request["checkpoint"]["impact_count"],
        request["checkpoint"]["impacts_digest_hex"],
    )
    assert body["id"] == expected


def test_distinct_checkpoint_identities_get_distinct_stable_ids(client):
    first = client.post(URL, json=_offline_request()).json()

    other_impacts = _offline_impacts()
    extra = json.loads(json.dumps(other_impacts[0]))
    extra["id"] = "rev_" + hashlib.sha256(b"offline-rev-3").hexdigest()
    other_impacts.append(extra)
    second = client.post(URL, json=_offline_request(other_impacts)).json()

    # A changed sequence changes count and digest, hence the identity.
    assert second["impact_count"] != first["impact_count"]
    assert second["impacts_digest_hex"] != first["impacts_digest_hex"]
    assert second["id"] != first["id"]


def test_empty_sequence_imports(client):
    request = _offline_request(impacts=[])
    assert request["checkpoint"]["impacts_digest_hex"] == hashlib.sha256(
        b"[]"
    ).hexdigest()
    resp = client.post(URL, json=request)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["impact_count"] == 0
    assert body["impacts_digest_hex"] == hashlib.sha256(b"[]").hexdigest()


def test_unescaped_non_ascii_participates_in_the_receipt(client):
    request = _offline_request()
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

    records = db_session.execute(select(ImpactImportRecord)).scalars().all()
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
        len(db_session.execute(select(ImpactImportRecord)).scalars().all())
        == 1
    )


def test_impact_order_participates_in_identity(client):
    # The array keeps its order: the same impacts in another order are a
    # different sequence with a different digest, hence a different receipt.
    request = _offline_request()
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

    first = client.post(URL, json=_offline_request(impacts_z)).json()
    second = client.post(URL, json=_offline_request(impacts_offset)).json()
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


def test_concurrent_identical_imports_register_exactly_once(file_client):
    # The real service serves requests concurrently: a burst of identical
    # imports must not wedge on connection checkout or fail with service
    # errors -- exactly one submission registers (201) and every retry
    # receives the original receipt (200), with one row and one audit event.
    request = _offline_request()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        responses = list(
            pool.map(lambda _: file_client.post(URL, json=request), range(40))
        )

    statuses = [resp.status_code for resp in responses]
    assert statuses.count(201) == 1, statuses
    assert statuses.count(200) == len(responses) - 1, statuses
    bodies = {
        json.dumps(resp.json(), sort_keys=True) for resp in responses
    }
    assert len(bodies) == 1

    session = file_client.app.state.session_factory()
    try:
        rows = session.execute(select(ImpactImportRecord)).scalars().all()
        assert len(rows) == 1
        assert len(_import_events(session)) == 1
    finally:
        session.close()


# --- Read endpoint -------------------------------------------------------------


def test_get_returns_the_public_receipt(client):
    created = client.post(URL, json=_offline_request()).json()
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
    created = client.post(URL, json=_offline_request()).json()
    base = f"{URL}/{created['id']}"
    for url in (
        f"{base}?limit=10",
        f"{base}?unknown=",
        f"{base}?a=1&b=2",
        f"{base}?a=1&a=2",
    ):
        resp = client.get(url)
        assert resp.status_code == 422, (url, resp.text)
        assert resp.json()["error"]["code"] == "validation_error"


def test_get_query_parameter_is_422_before_the_receipt_lookup(client):
    resp = client.get(f"{URL}/rii_doesnotexist?anything=1")
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


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


def test_first_import_writes_one_impact_imported_audit_event(client, db_session):
    body = client.post(URL, json=_offline_request()).json()
    events = _import_events(db_session)
    assert len(events) == 1
    event = events[0]
    assert event.event_type == EVENT_REVOCATION_IMPACT_IMPORTED
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
        select(ImpactImportRecord).where(ImpactImportRecord.id == receipt_id)
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
    resp = client.post(URL, json=_offline_request())
    assert resp.status_code == 201, resp.text

    # No domain resource was materialized.
    for model in _DOMAIN_MODELS:
        count = db_session.execute(
            select(func.count()).select_from(model)
        ).scalar_one()
        assert count == 0, model


def test_importing_creates_no_described_resources(client, db_session):
    request = _offline_request()
    described = {impact["id"] for impact in request["impacts"]}

    resp = client.post(URL, json=request)
    assert resp.status_code == 201

    # Exactly one receipt plus one audit event (the import's own); nothing
    # described by the imported impacts exists in the local state.
    all_events = db_session.execute(select(AuditEvent)).scalars().all()
    assert len(all_events) == 1
    assert all_events[0].resource_id not in described
    assert all_events[0].event_type == EVENT_REVOCATION_IMPACT_IMPORTED
    assert (
        db_session.execute(
            select(func.count()).select_from(ImpactImportRecord)
        ).scalar_one()
        == 1
    )


def test_receipt_table_persists_no_impacts_column(client, db_session):
    # The stored record has exactly the identity and bookkeeping columns;
    # there is nowhere an impacts array or raw impact value could persist.
    marker = "impacts-persistence-marker-7e4a"
    request = _offline_request()
    request["impacts"][0]["reason"] = marker
    request["checkpoint"]["impacts_digest_hex"] = _digest_of(request["impacts"])
    assert client.post(URL, json=request).status_code == 201

    columns = set(ImpactImportRecord.__table__.columns.keys())
    assert columns == {
        "seq",
        "id",
        "checkpoint_version",
        "impact_count",
        "impacts_digest_hex",
        "created_at",
    }

    record = db_session.execute(select(ImpactImportRecord)).scalar_one()
    assert marker not in repr(vars(record))


def test_import_does_not_change_served_package(client):
    # Registering a checkpoint never creates the described revocations: the
    # served package over local state stays empty after importing offline.
    empty_digest = client.get(PACKAGE_URL).json()["checkpoint"][
        "impacts_digest_hex"
    ]
    assert client.post(URL, json=_offline_request()).status_code == 201

    package = client.get(PACKAGE_URL).json()
    assert package["impacts"] == []
    assert package["checkpoint"]["impact_count"] == 0
    assert package["checkpoint"]["impacts_digest_hex"] == empty_digest


# --- Digest mismatch -----------------------------------------------------------


def test_digest_mismatch_is_422_and_writes_nothing(client, db_session):
    request = _offline_request()
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

    assert db_session.execute(select(ImpactImportRecord)).scalars().all() == []
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_tampered_impact_is_422_and_writes_nothing(client, db_session):
    request = _offline_request()
    # The checkpoint digest still commits to the untouched sequence.
    request["impacts"][0]["signer_actor_id"] = "org-forged"

    _assert_validation_error(client.post(URL, json=request))
    assert db_session.execute(select(ImpactImportRecord)).scalars().all() == []
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_reordered_impacts_without_recomputed_digest_is_422(client):
    request = _offline_request()
    reordered = json.loads(json.dumps(request))
    reordered["impacts"] = list(reversed(reordered["impacts"]))
    _assert_validation_error(client.post(URL, json=reordered))


def test_tampered_retry_under_existing_identity_is_422_and_keeps_record(
    client, db_session
):
    request = _offline_request()
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
        len(db_session.execute(select(ImpactImportRecord)).scalars().all())
        == 1
    )
    assert len(_import_events(db_session)) == 1


def test_mismatch_does_not_query_local_state(client):
    # A digest mismatch is rejected on the body alone in an empty database;
    # no revocation lookup could change it.
    request = _offline_request()
    request["checkpoint"]["impacts_digest_hex"] = "0" * 64
    _assert_validation_error(client.post(URL, json=request))


# --- Structural / field / count boundary --------------------------------------


def test_body_requires_exactly_checkpoint_and_impacts(client):
    request = _offline_request()

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
    request = _offline_request()
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
    request = _offline_request()

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
    request = _offline_request()
    for version in (
        "provenance-revocation-impact-checkpoint-v2",
        "Provenance-Revocation-Impact-Checkpoint-V1",
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


def test_requires_64_lowercase_hex_impacts_digest(client):
    request = _offline_request()
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
    request = _offline_request()
    for count in (-1, 2.0, 2.5, "2", "two", True, None, [2]):
        bad = json.loads(json.dumps(request))
        bad["checkpoint"]["impact_count"] = count
        _assert_validation_error(client.post(URL, json=bad))


def test_impacts_length_must_equal_impact_count(client):
    request = _offline_request()

    too_small = json.loads(json.dumps(request))
    too_small["checkpoint"]["impact_count"] = len(request["impacts"]) - 1
    _assert_validation_error(client.post(URL, json=too_small))

    too_large = json.loads(json.dumps(request))
    too_large["checkpoint"]["impact_count"] = len(request["impacts"]) + 1
    _assert_validation_error(client.post(URL, json=too_large))


def test_impacts_have_exactly_the_thirteen_fields(client):
    request = _offline_request()

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
    augmented["impacts"][0]["signature"] = "deadbeef"
    _assert_validation_error(client.post(URL, json=augmented))


def test_impact_items_must_be_objects(client):
    request = _offline_request()
    for wrong in (None, "impact", 42, ["rev_1", "org-1"]):
        bad = json.loads(json.dumps(request))
        bad["impacts"][0] = wrong
        _assert_validation_error(client.post(URL, json=bad))


def test_impact_identifiers_and_reason_must_be_non_empty(client):
    request = _offline_request()
    for field in (
        "id",
        "attestation_id",
        "revoker_actor_id",
        "content_id",
        "signer_actor_id",
        "reason",
    ):
        for value in ("", "   ", "\t", None, 42):
            bad = json.loads(json.dumps(request))
            bad["impacts"][0][field] = value
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
        bad["impacts"][0]["created_at"] = value
        _assert_validation_error(client.post(URL, json=bad))


def test_coverage_fields_must_be_internally_consistent(client):
    request = _offline_request()

    # Delta must equal before - after.
    bad = json.loads(json.dumps(request))
    bad["impacts"][0]["qualified_signer_count_delta"] = 0
    _assert_validation_error(client.post(URL, json=bad))

    # After must not exceed before.
    bad = json.loads(json.dumps(request))
    bad["impacts"][0]["qualified_signer_count_after"] = 2
    bad["impacts"][0]["qualified_signer_count_delta"] = 0
    bad["impacts"][0]["coverage_status_after"] = "covered"
    _assert_validation_error(client.post(URL, json=bad))

    # A positive count is always covered.
    bad = json.loads(json.dumps(request))
    bad["impacts"][0]["coverage_status_before"] = "partial"
    _assert_validation_error(client.post(URL, json=bad))

    # A zero count is never covered.
    bad = json.loads(json.dumps(request))
    bad["impacts"][0]["coverage_status_after"] = "covered"
    _assert_validation_error(client.post(URL, json=bad))

    # Unknown literals are rejected.
    for field in ("coverage_status_after", "coverage_status_before"):
        bad = json.loads(json.dumps(request))
        bad["impacts"][0][field] = "unknown"
        _assert_validation_error(client.post(URL, json=bad))

    bad = json.loads(json.dumps(request))
    bad["impacts"][0]["target_type"] = "content"
    _assert_validation_error(client.post(URL, json=bad))


def test_malformed_json_body_is_422(client, db_session):
    resp = client.post(
        URL,
        content='{"checkpoint":',
        headers={"content-type": "application/json"},
    )
    _assert_validation_error(resp)
    assert db_session.execute(select(ImpactImportRecord)).scalars().all() == []
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_every_failed_attempt_writes_nothing(client, db_session):
    good = _offline_request()

    mismatch = _offline_request()
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

    assert db_session.execute(select(ImpactImportRecord)).scalars().all() == []
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


def test_served_package_round_trips_through_import(client):
    _world(client)

    # The locally served package verifies, imports once (201), and the same
    # request re-presented is an idempotent 200; the local impact set is
    # untouched by the import.
    request = _served_request(client)
    assert client.post(VERIFY_URL, json=request).json() == {"valid": True}
    first = client.post(URL, json=request)
    assert first.status_code == 201, first.text
    assert first.json()["impact_count"] == 6
    assert client.post(URL, json=request).status_code == 200
    assert client.get(PACKAGE_URL).json()["checkpoint"] == request["checkpoint"]
