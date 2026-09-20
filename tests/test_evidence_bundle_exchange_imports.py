"""Tests for the controlled evidence-bundle exchange-package import endpoint.

Covers POST /v1/evidence-bundle-exchange-imports and GET
/v1/evidence-bundle-exchange-imports/{import_id}. The request body is
exactly ``{"manifest", "snapshot"}``: ``manifest`` is the existing
four-field exchange manifest and ``snapshot`` is the existing four-member
exchange snapshot, validated exactly as on the stateless verification
route (fixed version/algorithm, 64 lowercase hex digest, public-view fields
only, and consistent internal associations). The digest match is enforced
over the raw received snapshot under the existing canonical SHA-256 rules;
a mismatch is 422 validation_error and writes nothing.

Verification is decided by the request body alone -- no local resource
needs to exist and none is created -- and the request can carry no raw
signature, claim payload, content, or evidence bytes. The receipt stores
only the receiving identity (manifest version, evidence bundle id,
manifest digest), never the snapshot. First registration returns 201 with
a stable ``eir_`` id and a UTC ``received_at``, and writes the
``evidence_bundle.exchange_imported`` audit event in the same transaction;
a retry for the same identity returns the existing record with 200 and no
new audit. GET reads the public receipt; an unknown id is an explicit 404.
All fixtures are deterministic and offline.
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
    Claim,
    Content,
    ContentRelation,
    EvidenceBundle,
    ExchangeImportRecord,
)
from provenance.models import EVENT_EVIDENCE_BUNDLE_EXCHANGE_IMPORTED
from tests.test_evidence_bundle_exchange import (
    _create_attestation,
    _revoke,
    _setup_bundle,
)
from tests.test_exchange_manifest_verifications import (
    MANIFEST_VERSION,
    _digest_of,
    _offline_snapshot,
)
from tests.helpers import SEED_A

URL = "/v1/evidence-bundle-exchange-imports"
RECEIPT_KEYS = {
    "id",
    "manifest_version",
    "evidence_bundle_id",
    "manifest_digest_hex",
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


def _import_request(snapshot: dict | None = None) -> dict:
    """A controlled-import body built around one self-consistent snapshot."""
    snapshot = _offline_snapshot() if snapshot is None else snapshot
    return {
        "manifest": {
            "manifest_version": MANIFEST_VERSION,
            "evidence_bundle_id": snapshot["evidence_bundle"]["id"],
            "digest_algorithm": "sha256",
            "manifest_digest_hex": _digest_of(snapshot),
        },
        "snapshot": snapshot,
    }


def _served_import_request(client, bundle) -> dict:
    """A ``{manifest, snapshot}`` body built from the service's own package."""
    package = client.get(
        f"/v1/evidence-bundles/{bundle['id']}/exchange/package"
    ).json()
    return {"manifest": package["manifest"], "snapshot": package["snapshot"]}


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def _import_events(db_session):
    return (
        db_session.execute(
            select(AuditEvent)
            .where(
                AuditEvent.event_type
                == EVENT_EVIDENCE_BUNDLE_EXCHANGE_IMPORTED
            )
            .order_by(AuditEvent.seq.asc())
        )
        .scalars()
        .all()
    )


# --- First registration: success shape -----------------------------------------


def test_offline_package_first_import_returns_201_receipt(client, db_session):
    # The database is empty: every identifier in the package is unknown
    # locally, and verification must succeed on the body alone.
    request = _import_request()
    resp = client.post(URL, json=request)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == RECEIPT_KEYS
    assert body["id"].startswith("eir_")
    assert len(body["id"]) == len("eir_") + 64
    assert body["manifest_version"] == MANIFEST_VERSION
    assert body["evidence_bundle_id"] == request["manifest"]["evidence_bundle_id"]
    assert body["manifest_digest_hex"] == request["manifest"]["manifest_digest_hex"]
    # The snapshot is never echoed back on the receipt.
    assert "snapshot" not in body
    assert "manifest" not in body

    received_at = datetime.fromisoformat(body["received_at"])
    assert received_at.utcoffset().total_seconds() == 0
    assert body["received_at"].endswith(("Z", "+00:00"))

    # Exactly one receipt row exists with the same identity.
    records = db_session.execute(select(ExchangeImportRecord)).scalars().all()
    assert len(records) == 1
    (record,) = records
    assert record.id == body["id"]
    assert record.manifest_version == MANIFEST_VERSION
    assert record.evidence_bundle_id == body["evidence_bundle_id"]
    assert record.manifest_digest_hex == body["manifest_digest_hex"]


def test_served_package_imports_when_local_resources_exist(client):
    _, _, bundle = _setup_bundle(client)
    attested = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    _revoke(client, attested["id"])

    resp = client.post(URL, json=_served_import_request(client, bundle))
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["evidence_bundle_id"] == bundle["id"]
    assert body["id"].startswith("eir_")


def test_receipt_id_is_the_deterministic_identity_id(client):
    from provenance import ids

    request = _import_request()
    body = client.post(URL, json=request).json()
    expected = ids.exchange_import_id(
        MANIFEST_VERSION,
        request["manifest"]["evidence_bundle_id"],
        request["manifest"]["manifest_digest_hex"],
    )
    assert body["id"] == expected


def test_distinct_package_identities_get_distinct_stable_ids(client):
    first = client.post(URL, json=_import_request()).json()

    other_snapshot = _offline_snapshot()
    other_snapshot["content"]["title"] = "different offline 快照"
    second = client.post(URL, json=_import_request(other_snapshot)).json()

    # A changed snapshot changes the manifest digest, hence the identity.
    assert second["manifest_digest_hex"] != first["manifest_digest_hex"]
    assert second["id"] != first["id"]
    assert second["evidence_bundle_id"] == first["evidence_bundle_id"]


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

    records = db_session.execute(select(ExchangeImportRecord)).scalars().all()
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
        len(db_session.execute(select(ExchangeImportRecord)).scalars().all())
        == 1
    )


def test_reordered_root_is_a_different_package_identity(client):
    # The canonical digest commits to root member order: a package whose
    # manifest was computed over a differently ordered root is a different
    # receiving identity, even though it describes the same bundle.
    snapshot = _offline_snapshot()
    reordered = {
        key: snapshot[key]
        for key in ("attestations", "evidence_bundle", "claim", "content")
    }
    request = _import_request(reordered)
    # The manifest matches the reordered snapshot (both are valid offline).
    assert request["manifest"]["manifest_digest_hex"] != _digest_of(snapshot)

    first = client.post(URL, json=_import_request(snapshot)).json()
    second_resp = client.post(URL, json=request)
    assert second_resp.status_code == 201, second_resp.text
    second = second_resp.json()
    assert second["id"] != first["id"]
    assert second["manifest_digest_hex"] != first["manifest_digest_hex"]


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
    unknown = "eir_" + "0" * 64
    resp = client.get(f"{URL}/{unknown}")
    assert resp.status_code == 404, resp.text
    error = resp.json()["error"]
    assert error["code"] == "evidence_bundle_exchange_import_not_found"
    assert error["details"]["import_id"] == unknown
    # The failed read wrote nothing.
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_get_unshaped_and_other_prefix_ids_are_404(client):
    for raw_id in ("eir_doesnotexist", "evb_" + "a" * 64, "not-an-id"):
        resp = client.get(f"{URL}/{raw_id}")
        assert resp.status_code == 404, raw_id
        assert (
            resp.json()["error"]["code"]
            == "evidence_bundle_exchange_import_not_found"
        )


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


def test_first_import_writes_one_exchange_imported_audit_event(client, db_session):
    body = client.post(URL, json=_import_request()).json()
    events = _import_events(db_session)
    assert len(events) == 1
    event = events[0]
    assert event.event_type == EVENT_EVIDENCE_BUNDLE_EXCHANGE_IMPORTED
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
        select(ExchangeImportRecord).where(ExchangeImportRecord.id == receipt_id)
    ).scalar_one()
    event = db_session.execute(
        select(AuditEvent).where(AuditEvent.resource_id == receipt_id)
    ).scalar_one()
    assert record is not None
    assert event.event_type == EVENT_EVIDENCE_BUNDLE_EXCHANGE_IMPORTED


# --- Offline independence and no local-resource side effects -------------------


def test_verification_never_depends_on_local_resources(client, db_session):
    # A wholly fabricated, self-consistent package against unknown ids is
    # accepted into an empty database.
    resp = client.post(URL, json=_import_request())
    assert resp.status_code == 201, resp.text

    # No domain resource was materialized: the receipt is not a registration
    # of the snapshot's content, claim, bundle, or attestations.
    for model in _DOMAIN_MODELS:
        count = db_session.execute(
            select(func.count()).select_from(model)
        ).scalar_one()
        assert count == 0, model


def test_importing_creates_no_domain_resources_for_unknown_ids(client, db_session):
    request = _import_request()
    resp = client.post(URL, json=request)
    assert resp.status_code == 201

    # Exactly one receipt plus one audit event; nothing else exists.
    assert (
        db_session.execute(select(func.count()).select_from(Content)).scalar_one()
        == 0
    )
    assert (
        db_session.execute(select(func.count()).select_from(Claim)).scalar_one()
        == 0
    )
    assert (
        db_session.execute(
            select(func.count()).select_from(EvidenceBundle)
        ).scalar_one()
        == 0
    )
    assert (
        db_session.execute(
            select(func.count()).select_from(Attestation)
        ).scalar_one()
        == 0
    )
    assert (
        db_session.execute(
            select(func.count()).select_from(ExchangeImportRecord)
        ).scalar_one()
        == 1
    )
    assert (
        db_session.execute(select(func.count()).select_from(AuditEvent)).scalar_one()
        == 1
    )


def test_receipt_table_persists_no_snapshot_column(client, db_session):
    # The stored record has exactly the identity and bookkeeping columns;
    # there is nowhere a snapshot, raw signature, payload, or bytes could be
    # persisted.
    marker = "snapshot-persistence-marker-c21d"
    request = _import_request()
    request["snapshot"]["evidence_bundle"]["metadata"]["marker"] = marker
    request["manifest"]["manifest_digest_hex"] = _digest_of(request["snapshot"])
    assert client.post(URL, json=request).status_code == 201

    columns = set(ExchangeImportRecord.__table__.columns.keys())
    assert columns == {
        "seq",
        "id",
        "manifest_version",
        "evidence_bundle_id",
        "manifest_digest_hex",
        "created_at",
    }

    record = db_session.execute(select(ExchangeImportRecord)).scalar_one()
    assert marker not in repr(vars(record))


# --- Digest mismatch -----------------------------------------------------------


def test_manifest_digest_mismatch_is_422_and_writes_nothing(client, db_session):
    request = _import_request()
    claimed = request["manifest"]["manifest_digest_hex"]
    request["manifest"]["manifest_digest_hex"] = (
        claimed[:-1] + ("0" if claimed[-1] != "0" else "1")
    )

    resp = client.post(URL, json=request)
    assert resp.status_code == 422, resp.text
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"]["reason"] == "manifest_digest_mismatch"
    assert error["details"]["computed_digest_hex"] == _digest_of(
        request["snapshot"]
    )

    assert db_session.execute(select(ExchangeImportRecord)).scalars().all() == []
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_tampered_snapshot_is_422_and_writes_nothing(client, db_session):
    request = _import_request()
    # The manifest still commits to the untouched snapshot.
    request["snapshot"]["evidence_bundle"]["media_type"] = "video/mp4"

    _assert_validation_error(client.post(URL, json=request))
    assert db_session.execute(select(ExchangeImportRecord)).scalars().all() == []
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_tampered_retry_under_existing_identity_is_422_and_keeps_record(
    client, db_session
):
    request = _import_request()
    original = client.post(URL, json=request)
    assert original.status_code == 201

    # Same identity fields, but the presented snapshot no longer matches:
    # validation beats the idempotent lookup and the original survives.
    tampered = json.loads(json.dumps(request))
    tampered["snapshot"]["content"]["title"] = "forged"
    _assert_validation_error(client.post(URL, json=tampered))

    assert client.get(
        f"{URL}/{original.json()['id']}"
    ).json() == original.json()
    assert (
        len(db_session.execute(select(ExchangeImportRecord)).scalars().all())
        == 1
    )
    assert len(_import_events(db_session)) == 1


def test_mismatch_does_not_query_local_resources(client):
    # A digest mismatch is rejected on the body alone in an empty database;
    # no id resolution could change it.
    request = _import_request()
    request["manifest"]["manifest_digest_hex"] = "0" * 64
    _assert_validation_error(client.post(URL, json=request))


# --- Structural / field boundary -----------------------------------------------


def test_body_requires_exactly_manifest_and_snapshot(client):
    request = _import_request()

    for missing in ("manifest", "snapshot"):
        incomplete = {k: v for k, v in request.items() if k != missing}
        _assert_validation_error(client.post(URL, json=incomplete))

    augmented = dict(request)
    augmented["batch_id"] = "b-1"
    _assert_validation_error(client.post(URL, json=augmented))


def test_rejects_flat_verification_request_shape(client):
    # The five-field shape used by the stateless verification route is not
    # the nested {manifest, snapshot} shape accepted here.
    snapshot = _offline_snapshot()
    flat = {
        "manifest_version": MANIFEST_VERSION,
        "evidence_bundle_id": snapshot["evidence_bundle"]["id"],
        "digest_algorithm": "sha256",
        "manifest_digest_hex": _digest_of(snapshot),
        "snapshot": snapshot,
    }
    _assert_validation_error(client.post(URL, json=flat))


def test_non_object_bodies_are_422(client):
    for body in (None, [], "package", 42):
        _assert_validation_error(client.post(URL, json=body))
    _assert_validation_error(client.post(URL, json={}))


def test_manifest_requires_exactly_its_four_fields(client):
    request = _import_request()
    for field in (
        "manifest_version",
        "evidence_bundle_id",
        "digest_algorithm",
        "manifest_digest_hex",
    ):
        incomplete = json.loads(json.dumps(request))
        del incomplete["manifest"][field]
        _assert_validation_error(client.post(URL, json=incomplete))

    augmented = json.loads(json.dumps(request))
    augmented["manifest"]["computed_digest_hex"] = (
        request["manifest"]["manifest_digest_hex"]
    )
    _assert_validation_error(client.post(URL, json=augmented))


def test_manifest_fixed_version_and_algorithm(client):
    for field, values in (
        ("manifest_version", ("provenance-exchange-manifest-v2", "")),
        ("digest_algorithm", ("sha512", "SHA256", "sha-256")),
    ):
        for value in values:
            bad = _import_request()
            bad["manifest"][field] = value
            _assert_validation_error(client.post(URL, json=bad))


def test_manifest_digest_must_be_64_lowercase_hex(client):
    valid = _import_request()["manifest"]["manifest_digest_hex"]
    for digest in (valid.upper(), valid[:-1], valid + "0", "g" + valid[1:], 123):
        bad = _import_request()
        bad["manifest"]["manifest_digest_hex"] = digest
        _assert_validation_error(client.post(URL, json=bad))


def test_manifest_and_snapshot_members_have_correct_types(client):
    cases = []
    for member, wrong in (("manifest", []), ("manifest", None),
                          ("snapshot", {}), ("snapshot", "s")):
        bad = _import_request()
        bad[member] = wrong
        cases.append(bad)
    for case in cases:
        _assert_validation_error(client.post(URL, json=case))


def test_snapshot_requires_exactly_its_four_members(client):
    request = _import_request()
    for member in ("content", "claim", "evidence_bundle", "attestations"):
        incomplete = json.loads(json.dumps(request))
        del incomplete["snapshot"][member]
        # The missing member is itself a structural 422, regardless of the
        # manifest digest.
        _assert_validation_error(client.post(URL, json=incomplete))

    augmented = json.loads(json.dumps(request))
    augmented["snapshot"]["lineage"] = []
    _assert_validation_error(client.post(URL, json=augmented))


def test_raw_material_fields_are_rejected_and_never_echoed(client, db_session):
    secret = "raw-material-marker-9b3f"
    cases = []
    for path in (
        ("snapshot", "content", "data"),
        ("snapshot", "claim", "payload"),
        ("snapshot", "evidence_bundle", "evidence"),
        ("snapshot", "attestations", 0, "signature"),
    ):
        bad = _import_request()
        target = bad
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = secret
        cases.append(bad)

    for case in cases:
        resp = client.post(URL, json=case)
        _assert_validation_error(resp)
        assert secret not in resp.text

    assert db_session.execute(select(ExchangeImportRecord)).scalars().all() == []
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_non_finite_metadata_is_422(client):
    for non_finite in ("NaN", "Infinity", "-Infinity"):
        request = _import_request()
        raw = json.dumps(request).replace(
            '"origin": "offline"', f'"origin": {non_finite}'
        )
        resp = client.post(
            URL, content=raw, headers={"content-type": "application/json"}
        )
        _assert_validation_error(resp)


def test_malformed_json_body_is_422(client, db_session):
    resp = client.post(
        URL,
        content='{"manifest":',
        headers={"content-type": "application/json"},
    )
    _assert_validation_error(resp)
    assert db_session.execute(select(ExchangeImportRecord)).scalars().all() == []
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


# --- Snapshot internal associations --------------------------------------------


def test_snapshot_bundle_id_must_match_manifest(client):
    request = _import_request()
    other = "evb_" + hashlib.sha256(b"other-bundle").hexdigest()
    request["manifest"]["evidence_bundle_id"] = other
    _assert_validation_error(client.post(URL, json=request))


def test_snapshot_claim_must_match_bundle(client):
    request = _import_request()
    request["snapshot"]["claim"]["id"] = (
        "clm_" + hashlib.sha256(b"other-claim").hexdigest()
    )
    _assert_validation_error(client.post(URL, json=request))


def test_snapshot_content_must_match_claim(client):
    request = _import_request()
    request["snapshot"]["content"]["id"] = (
        "cnt_" + hashlib.sha256(b"other-content").hexdigest()
    )
    _assert_validation_error(client.post(URL, json=request))


def test_snapshot_attestations_must_target_the_bundle(client):
    wrong_type = _import_request()
    wrong_type["snapshot"]["attestations"][0]["target_type"] = "claim"
    wrong_type["snapshot"]["attestations"][0]["target_id"] = (
        wrong_type["snapshot"]["claim"]["id"]
    )
    _assert_validation_error(client.post(URL, json=wrong_type))

    wrong_id = _import_request()
    wrong_id["snapshot"]["attestations"][0]["target_id"] = (
        "evb_" + hashlib.sha256(b"other-bundle").hexdigest()
    )
    _assert_validation_error(client.post(URL, json=wrong_id))


def test_empty_attestation_list_imports(client):
    request = _import_request()
    request["snapshot"]["attestations"] = []
    request["manifest"]["manifest_digest_hex"] = _digest_of(request["snapshot"])
    resp = client.post(URL, json=request)
    assert resp.status_code == 201, resp.text
    assert resp.json()["id"].startswith("eir_")


# --- Aggregate no-write boundary ------------------------------------------------


def test_every_failed_attempt_writes_nothing(client, db_session):
    good = _import_request()

    mismatch = _import_request()
    mismatch["manifest"]["manifest_digest_hex"] = "0" * 64

    attempts = [
        client.post(URL, json={}),
        client.post(URL, json={"manifest": good["manifest"]}),
        client.post(URL, content="{not json",
                    headers={"content-type": "application/json"}),
        client.post(URL, json=mismatch),
    ]
    assert [r.status_code for r in attempts] == [422, 422, 422, 422]

    assert db_session.execute(select(ExchangeImportRecord)).scalars().all() == []
    assert db_session.execute(select(AuditEvent)).scalars().all() == []
    for model in _DOMAIN_MODELS:
        assert (
            db_session.execute(
                select(func.count()).select_from(model)
            ).scalar_one()
            == 0
        )

    # After the failures, the valid package still registers for the first
    # time with exactly one audit event.
    assert client.post(URL, json=good).status_code == 201
    assert len(_import_events(db_session)) == 1


# --- Compatibility with the read-only exchange/verification routes --------------


def test_imported_package_still_verifies_offline(client):
    _, _, bundle = _setup_bundle(client)
    body = _served_import_request(client, bundle)

    # The same package is accepted by the import route and still verifies
    # through the stateless verification route, in either order.
    flat = {**body["manifest"], "snapshot": body["snapshot"]}
    verification = client.post("/v1/exchange-manifest-verifications", json=flat)
    assert verification.status_code == 200
    assert verification.json() == {"valid": True}

    imported = client.post(URL, json=body)
    assert imported.status_code == 201, imported.text

    # The exchange routes remain strictly read-only after an import: the
    # package can still be read back unchanged.
    again = client.post(
        "/v1/exchange-manifest-verifications",
        json={**body["manifest"], "snapshot": body["snapshot"]},
    )
    assert again.json() == {"valid": True}
