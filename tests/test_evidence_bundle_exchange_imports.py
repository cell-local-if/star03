"""Tests for the controlled received-package import endpoint.

Covers POST /v1/evidence-bundle-exchange-imports and
GET /v1/evidence-bundle-exchange-imports/{import_id}: the request body is
exactly {"manifest", "snapshot"}, each reusing the existing four-field
exchange manifest and four-member exchange snapshot public structures;
the server re-verifies offline that the claimed manifest digest equals the
SHA-256 of the snapshot canonicalized exactly as received, and the schema
enforces the snapshot's internal content/claim/bundle/attestation
associations and the absence of raw material. The verdict never depends on
whether the named resources exist locally. Structural failures, missing or
extra fields, malformed JSON, and digest mismatches are 422
validation_error and write nothing. Success returns 201 with a stable
``eir_`` id, the identity fields, and a UTC ``received_at``, writing the
receipt row and the ``evidence_bundle.exchange_imported`` audit event in a
single transaction; retrying the same identity returns 200 with the
existing record and adds no audit. The GET route reads the public receipt
(unknown id is an explicit 404) and the full snapshot is never persisted.
All fixtures are deterministic and offline.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance.app import create_app
from provenance.config import Settings
from provenance.ids import evidence_bundle_exchange_import_id
from provenance.models import (
    Actor,
    Attestation,
    AttestationRevocation,
    AuditEvent,
    Claim,
    Content,
    EvidenceBundle,
    EvidenceBundleExchangeImport,
    EVENT_EVIDENCE_BUNDLE_EXCHANGE_IMPORTED,
)
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    DIGEST_C,
    SEED_A,
    SEED_B,
    ed25519_public_key,
)
from tests.test_evidence_bundle_exchange import (
    _create_attestation,
    _create_bundle,
    _create_claim,
    _create_content,
    _revoke,
    _setup_bundle,
)

URL = "/v1/evidence-bundle-exchange-imports"
MANIFEST_VERSION = "provenance-exchange-manifest-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_EIR = re.compile(r"^eir_[0-9a-f]{64}$")

RECEIPT_FIELDS = {
    "id",
    "manifest_version",
    "evidence_bundle_id",
    "digest_algorithm",
    "manifest_digest_hex",
    "received_at",
}

_ALL_MODELS = (
    Actor,
    Content,
    Claim,
    EvidenceBundle,
    Attestation,
    AttestationRevocation,
    EvidenceBundleExchangeImport,
    AuditEvent,
)


def _item_url(import_id: str) -> str:
    return f"{URL}/{import_id}"


def _canonical_snapshot_bytes(snapshot: dict) -> bytes:
    """Independently apply the manifest canonicalization to a parsed snapshot."""

    def normalize(value, sort_keys: bool):
        if isinstance(value, dict):
            items = sorted(value.items()) if sort_keys else value.items()
            return {key: normalize(member, True) for key, member in items}
        if isinstance(value, list):
            return [normalize(item, True) for item in value]
        return value

    import json as _json

    return _json.dumps(
        normalize(snapshot, False),
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest_of(snapshot: dict) -> str:
    return hashlib.sha256(_canonical_snapshot_bytes(snapshot)).hexdigest()


def _package_url(bundle) -> str:
    return f"/v1/evidence-bundles/{bundle['id']}/exchange/package"


def _served_package(client, bundle) -> dict:
    """A valid import body: exactly what the package route serves."""
    resp = client.get(_package_url(bundle))
    assert resp.status_code == 200, resp.text
    return resp.json()


def _offline_snapshot() -> dict:
    """A fully self-consistent snapshot fabricated without service state."""
    content = {
        "id": "cnt_" + hashlib.sha256(b"offline-content-import").hexdigest(),
        "digest_algorithm": "sha256",
        "digest_hex": DIGEST_A,
        "media_type": "image/png",
        "title": "offline 快照",
        "actor_id": "org-offline",
        "created_at": "2026-01-02T03:04:05Z",
    }
    claim = {
        "id": "clm_" + hashlib.sha256(b"offline-claim-import").hexdigest(),
        "content_id": content["id"],
        "actor_id": "org-offline",
        "claim_type": "authorship",
        "payload_digest_algorithm": "sha256",
        "payload_digest_hex": DIGEST_B,
        "created_at": "2026-01-02T03:04:06Z",
    }
    bundle_id = "evb_" + hashlib.sha256(b"offline-bundle-import").hexdigest()
    bundle = {
        "id": bundle_id,
        "claim_id": claim["id"],
        "evidence_type": "raw_capture",
        "digest_algorithm": "sha256",
        "digest_hex": DIGEST_C,
        "media_type": "image/jpeg",
        "metadata": {"origin": "offline", "标签": {"中": True}},
        "created_at": "2026-01-02T03:04:07Z",
    }
    attestation = {
        "id": "att_" + hashlib.sha256(b"offline-att-import").hexdigest(),
        "target_type": "evidence_bundle",
        "target_id": bundle_id,
        "signer_actor_id": "org-offline",
        "public_key": base64.b64encode(ed25519_public_key(SEED_A)).decode("ascii"),
        "signature_digest_algorithm": "sha256",
        "signature_digest_hex": hashlib.sha256(
            b"offline-signature-import"
        ).hexdigest(),
        "verified": True,
        "created_at": "2026-01-02T03:04:08Z",
    }
    return {
        "content": content,
        "claim": claim,
        "evidence_bundle": bundle,
        "attestations": [attestation],
    }


def _offline_package(snapshot: dict | None = None) -> dict:
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


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def _counts(db_session) -> dict:
    return {
        model: db_session.execute(
            select(func.count()).select_from(model)
        ).scalar_one()
        for model in _ALL_MODELS
    }


# --- Success shape and identity ----------------------------------------------


def test_import_served_package_returns_201_exact_receipt_shape(client):
    _, _, bundle = _setup_bundle(client)
    package = _served_package(client, bundle)

    resp = client.post(URL, json=package)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == RECEIPT_FIELDS
    assert _EIR.fullmatch(body["id"])
    assert body["manifest_version"] == MANIFEST_VERSION
    assert body["evidence_bundle_id"] == bundle["id"]
    assert body["digest_algorithm"] == "sha256"
    assert body["manifest_digest_hex"] == package["manifest"]["manifest_digest_hex"]
    assert _HEX64.fullmatch(body["manifest_digest_hex"])
    # The eir id is the stable digest of the receipt identity.
    assert body["id"] == evidence_bundle_exchange_import_id(
        MANIFEST_VERSION, bundle["id"], body["manifest_digest_hex"]
    )
    # received_at is a UTC timestamp rendered with a zero offset.
    received_at = datetime.fromisoformat(body["received_at"])
    assert received_at.utcoffset().total_seconds() == 0
    assert body["received_at"].endswith(("Z", "+00:00"))
    # The snapshot is not echoed back in any form.
    assert "snapshot" not in body
    assert "manifest" not in body


def test_import_succeeds_with_no_local_resources_at_all(client, db_session):
    # Empty database and wholly unknown identifiers: the verdict is decided
    # by the package alone and must not depend on local resource existence.
    before = _counts(db_session)
    resp = client.post(URL, json=_offline_package())
    assert resp.status_code == 201, resp.text
    # Only the receipt row and its audit event were written.
    after = _counts(db_session)
    assert after[EvidenceBundleExchangeImport] == 1
    assert after[AuditEvent] == 1
    for model in (Actor, Content, Claim, EvidenceBundle, Attestation):
        assert after[model] == before[model] == 0


def test_import_verdict_is_identical_known_and_unknown_bundle(client):
    # The same offline package is accepted whether or not a local bundle with
    # that id exists: local state never enters verification.
    _, _, bundle = _setup_bundle(client)
    served = _served_package(client, bundle)
    served_snapshot = served["snapshot"]

    # A structurally identical, self-consistent package for a bundle id the
    # service has never seen is accepted on the same terms.
    unknown = _offline_package()
    assert (
        unknown["manifest"]["evidence_bundle_id"]
        != served["manifest"]["evidence_bundle_id"]
    )

    first = client.post(URL, json=served)
    second = client.post(URL, json=unknown)
    assert first.status_code == second.status_code == 201
    assert first.json()["evidence_bundle_id"] == served_snapshot["evidence_bundle"]["id"]
    assert second.json()["evidence_bundle_id"] == unknown["manifest"]["evidence_bundle_id"]


def test_import_accepts_non_ascii_and_reordered_root(client):
    # Canonicalization follows the offline verification rules: root member
    # order is preserved and non-ASCII is hashed as raw UTF-8. A manifest
    # computed over a reordered, non-ASCII root verifies in that same order.
    snapshot = _offline_snapshot()
    reordered = {
        key: snapshot[key]
        for key in ("attestations", "evidence_bundle", "claim", "content")
    }
    resp = client.post(URL, json=_offline_package(reordered))
    assert resp.status_code == 201, resp.text
    # The digest differs from the standard-order root, proving raw order binds.
    assert _digest_of(reordered) != _digest_of(snapshot)
    assert resp.json()["manifest_digest_hex"] == _digest_of(reordered)


# --- Idempotency --------------------------------------------------------------


def test_retry_same_identity_returns_200_existing_record_and_no_new_audit(
    client, db_session
):
    package = _offline_package()
    first = client.post(URL, json=package)
    assert first.status_code == 201, first.text
    first_body = first.json()

    events_after_first = db_session.execute(
        select(func.count()).select_from(AuditEvent)
    ).scalar_one()
    receipts_after_first = db_session.execute(
        select(func.count()).select_from(EvidenceBundleExchangeImport)
    ).scalar_one()

    # Retry with the same identity, even though snapshot whitespace/formatting
    # and JSON key presentation differ (bytes parse to the same package).
    retry = client.post(
        URL,
        content=json.dumps(package, indent=2),
        headers={"content-type": "application/json"},
    )
    assert retry.status_code == 200, retry.text
    assert retry.json() == first_body

    assert (
        db_session.execute(
            select(func.count()).select_from(EvidenceBundleExchangeImport)
        ).scalar_one()
        == receipts_after_first
    )
    events = db_session.execute(select(AuditEvent)).scalars().all()
    assert len(events) == events_after_first == 1
    assert events[0].event_type == EVENT_EVIDENCE_BUNDLE_EXCHANGE_IMPORTED
    assert events[0].resource_id == first_body["id"]


def test_distinct_identities_create_distinct_receipts(client):
    package_one = _offline_package()
    package_two = json.loads(json.dumps(package_one))
    # A different claimed snapshot digest (empty attestations) is a distinct
    # receipt identity even for the same bundle id.
    package_two["snapshot"]["attestations"] = []
    package_two["manifest"]["manifest_digest_hex"] = _digest_of(
        package_two["snapshot"]
    )

    first = client.post(URL, json=package_one)
    second = client.post(URL, json=package_two)
    assert first.status_code == second.status_code == 201
    assert first.json()["id"] != second.json()["id"]
    assert (
        first.json()["manifest_digest_hex"]
        != second.json()["manifest_digest_hex"]
    )

    # Retrying each independently is 200 and returns its own record.
    assert client.post(URL, json=package_one).status_code == 200
    assert client.post(URL, json=package_two).status_code == 200


def test_retry_is_stable_across_app_restart(file_client, tmp_db_url):
    package = _offline_package()
    first = file_client.post(URL, json=package)
    assert first.status_code == 201, first.text
    expected = first.json()

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        retry = client.post(URL, json=package)
        assert retry.status_code == 200, retry.text
        assert retry.json() == expected
        fetched = client.get(_item_url(expected["id"]))
        assert fetched.status_code == 200
        assert fetched.json() == expected


# --- Audit transaction --------------------------------------------------------


def test_receipt_and_audit_event_commit_together(client, db_session):
    package = _offline_package()
    resp = client.post(URL, json=package)
    assert resp.status_code == 201
    receipt_id = resp.json()["id"]

    receipts = db_session.execute(
        select(EvidenceBundleExchangeImport)
    ).scalars().all()
    assert [r.id for r in receipts] == [receipt_id]

    events = (
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
    assert len(events) == 1
    event = events[0]
    assert event.resource_id == receipt_id
    assert event.created_at.tzinfo.utcoffset(event.created_at).total_seconds() == 0
    # The receipt time and the audit time come from the same transaction.
    record = receipts[0]
    assert record.received_at == event.created_at


def test_audit_event_is_listable_with_its_exact_type(client):
    body = client.post(URL, json=_offline_package()).json()
    page = client.get(
        "/v1/audit-events",
        params={"event_type": EVENT_EVIDENCE_BUNDLE_EXCHANGE_IMPORTED},
    )
    assert page.status_code == 200, page.text
    items = page.json()["items"]
    assert [(i["event_type"], i["resource_id"]) for i in items] == [
        (EVENT_EVIDENCE_BUNDLE_EXCHANGE_IMPORTED, body["id"])
    ]


# --- GET receipt --------------------------------------------------------------


def test_get_receipt_returns_the_public_record(client):
    created = client.post(URL, json=_offline_package()).json()
    resp = client.get(_item_url(created["id"]))
    assert resp.status_code == 200, resp.text
    assert resp.json() == created
    assert set(resp.json()) == RECEIPT_FIELDS


def test_get_unknown_import_is_explicit_404(client):
    resp = client.get(_item_url("eir_doesnotexist000000000000000000000000000000000000000000000000000"))
    assert resp.status_code == 404, resp.text
    error = resp.json()["error"]
    assert error["code"] == "evidence_bundle_exchange_import_not_found"
    assert error["details"]["import_id"].startswith("eir_")


def test_get_receipt_rejects_any_query_parameter(client):
    created = client.post(URL, json=_offline_package()).json()
    base = _item_url(created["id"])
    for url in (
        f"{base}?limit=10",
        f"{base}?unknown=",
        f"{base}?a=1&b=2",
        f"{base}?a=1&a=2",
    ):
        resp = client.get(url)
        _assert_validation_error(resp)


def test_snapshot_is_not_persisted(client, db_session):
    package = _offline_package()
    marker = json.dumps(package["snapshot"], ensure_ascii=False)
    created = client.post(URL, json=package).json()

    record = db_session.execute(
        select(EvidenceBundleExchangeImport).where(
            EvidenceBundleExchangeImport.id == created["id"]
        )
    ).scalar_one()
    # The ORM row exposes only identity and receipt time; the snapshot marker
    # is nowhere in the persisted row.
    assert marker not in str(record.__dict__)
    assert not hasattr(record, "snapshot")
    assert not hasattr(record, "manifest")
    assert record.manifest_version == MANIFEST_VERSION
    assert record.evidence_bundle_id == package["manifest"]["evidence_bundle_id"]
    assert record.manifest_digest_hex == package["manifest"]["manifest_digest_hex"]


# --- Verification failures: digest mismatch -----------------------------------


def test_digest_mismatch_is_422_and_writes_nothing(client, db_session):
    package = _offline_package()
    claimed = package["manifest"]["manifest_digest_hex"]
    package["manifest"]["manifest_digest_hex"] = claimed[:-1] + (
        "0" if claimed[-1] != "0" else "1"
    )
    before = _counts(db_session)
    resp = client.post(URL, json=package)
    _assert_validation_error(resp)
    assert db_session.execute(
        select(EvidenceBundleExchangeImport)
    ).scalars().all() == []
    assert _counts(db_session) == before


def test_tampered_snapshot_is_422_and_writes_nothing(client, db_session):
    package = _offline_package()
    package["snapshot"]["content"]["title"] = "forged title"
    before = _counts(db_session)
    resp = client.post(URL, json=package)
    _assert_validation_error(resp)
    assert _counts(db_session) == before


def test_attestation_reorder_against_manifest_is_422(client):
    _, _, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_B, signer="org-2"
    )
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    package = _served_package(client, bundle)
    package["snapshot"]["attestations"] = list(
        reversed(package["snapshot"]["attestations"])
    )
    _assert_validation_error(client.post(URL, json=package))


def test_revoked_attestation_still_verifies_and_imports(client):
    _, _, bundle = _setup_bundle(client)
    attested = _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    _revoke(client, attested["id"])
    resp = client.post(URL, json=_served_package(client, bundle))
    assert resp.status_code == 201, resp.text


# --- Structural, field, and association validation ----------------------------


def test_requires_exactly_manifest_and_snapshot(client):
    package = _offline_package()

    for missing in ("manifest", "snapshot"):
        _assert_validation_error(
            client.post(URL, json={k: v for k, v in package.items() if k != missing})
        )

    augmented = dict(package)
    augmented["computed_digest_hex"] = package["manifest"]["manifest_digest_hex"]
    _assert_validation_error(client.post(URL, json=augmented))


def test_rejects_non_object_and_empty_bodies(client):
    for body in (None, [], "package", 42):
        _assert_validation_error(client.post(URL, json=body))
    _assert_validation_error(client.post(URL, json={}))


def test_manifest_rejects_extra_and_missing_fields(client):
    package = _offline_package()
    manifest = package["manifest"]

    for field in (
        "manifest_version",
        "evidence_bundle_id",
        "digest_algorithm",
        "manifest_digest_hex",
    ):
        bad = json.loads(json.dumps(package))
        del bad["manifest"][field]
        _assert_validation_error(client.post(URL, json=bad))

    bad = json.loads(json.dumps(package))
    bad["manifest"]["computed_digest_hex"] = manifest["manifest_digest_hex"]
    _assert_validation_error(client.post(URL, json=bad))


def test_manifest_rejects_bad_version_algorithm_and_digest(client):
    package = _offline_package()
    valid_digest = package["manifest"]["manifest_digest_hex"]

    for version in (
        "provenance-exchange-manifest-v2",
        "Provenance-Exchange-Manifest-V1",
        "",
    ):
        bad = json.loads(json.dumps(package))
        bad["manifest"]["manifest_version"] = version
        _assert_validation_error(client.post(URL, json=bad))

    for algorithm in ("sha512", "SHA256", "sha-256", ""):
        bad = json.loads(json.dumps(package))
        bad["manifest"]["digest_algorithm"] = algorithm
        _assert_validation_error(client.post(URL, json=bad))

    for digest in (
        valid_digest.upper(),
        valid_digest[:-1],
        valid_digest + "0",
        "g" + valid_digest[1:],
        123,
        None,
    ):
        bad = json.loads(json.dumps(package))
        bad["manifest"]["manifest_digest_hex"] = digest
        _assert_validation_error(client.post(URL, json=bad))


def test_snapshot_rejects_missing_and_extra_members(client):
    package = _offline_package()
    for member in ("content", "claim", "evidence_bundle", "attestations"):
        bad = json.loads(json.dumps(package))
        del bad["snapshot"][member]
        _assert_validation_error(client.post(URL, json=bad))

    bad = json.loads(json.dumps(package))
    bad["snapshot"]["lineage"] = []
    _assert_validation_error(client.post(URL, json=bad))


def test_snapshot_member_types_are_enforced(client):
    package = _offline_package()
    for member, wrong in (
        ("content", []),
        ("claim", None),
        ("evidence_bundle", "evb"),
        ("attestations", {}),
    ):
        bad = json.loads(json.dumps(package))
        bad["snapshot"][member] = wrong
        _assert_validation_error(client.post(URL, json=bad))


def test_nested_views_reject_raw_material_fields(client):
    package = _offline_package()
    secret_marker = "raw-exchange-import-marker-9c2e"

    cases = []
    for path, value in (
        (("snapshot", "content", "data"), secret_marker),
        (("snapshot", "claim", "payload"), {"raw": True}),
        (("snapshot", "evidence_bundle", "evidence"), secret_marker),
        (("snapshot", "attestations", 0, "signature"), secret_marker),
    ):
        bad = json.loads(json.dumps(package))
        target = bad
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        cases.append(bad)

    for bad in cases:
        resp = client.post(URL, json=bad)
        _assert_validation_error(resp)
        assert secret_marker not in resp.text


def test_nested_views_require_every_public_field(client):
    package = _offline_package()
    for member, field in (
        ("content", "created_at"),
        ("claim", "payload_digest_hex"),
        ("evidence_bundle", "metadata"),
        ("attestations", 0),
    ):
        bad = json.loads(json.dumps(package))
        if member == "attestations":
            del bad["snapshot"]["attestations"][0]["signature_digest_hex"]
        else:
            del bad["snapshot"][member][field]
        _assert_validation_error(client.post(URL, json=bad))


def test_nested_field_types_are_enforced(client):
    package = _offline_package()
    cases = []
    bad = json.loads(json.dumps(package))
    bad["snapshot"]["content"]["created_at"] = "not-a-timestamp"
    cases.append(bad)
    bad = json.loads(json.dumps(package))
    bad["snapshot"]["claim"]["payload_digest_hex"] = 123
    cases.append(bad)
    bad = json.loads(json.dumps(package))
    bad["snapshot"]["evidence_bundle"]["metadata"] = ["not", "an", "object"]
    cases.append(bad)
    bad = json.loads(json.dumps(package))
    bad["snapshot"]["attestations"][0]["verified"] = "not-a-bool"
    cases.append(bad)
    for case in cases:
        _assert_validation_error(client.post(URL, json=case))


def test_metadata_with_non_finite_numbers_is_422(client):
    for non_finite in ("NaN", "Infinity", "-Infinity"):
        package = _offline_package()
        raw = json.dumps(package).replace(
            '"origin": "offline"', f'"origin": {non_finite}'
        )
        resp = client.post(
            URL, content=raw, headers={"content-type": "application/json"}
        )
        _assert_validation_error(resp)


def test_manifest_bundle_id_must_match_snapshot(client):
    package = _offline_package()
    package["manifest"]["evidence_bundle_id"] = (
        "evb_" + hashlib.sha256(b"other-bundle").hexdigest()
    )
    _assert_validation_error(client.post(URL, json=package))


def test_claim_and_content_associations_must_be_consistent(client):
    bad_claim = _offline_package()
    bad_claim["snapshot"]["claim"]["id"] = (
        "clm_" + hashlib.sha256(b"x").hexdigest()
    )
    _assert_validation_error(client.post(URL, json=bad_claim))

    bad_content = _offline_package()
    bad_content["snapshot"]["content"]["id"] = (
        "cnt_" + hashlib.sha256(b"x").hexdigest()
    )
    _assert_validation_error(client.post(URL, json=bad_content))


def test_attestations_must_target_the_bundle(client):
    wrong_type = _offline_package()
    wrong_type["snapshot"]["attestations"][0]["target_type"] = "claim"
    wrong_type["snapshot"]["attestations"][0]["target_id"] = (
        wrong_type["snapshot"]["claim"]["id"]
    )
    _assert_validation_error(client.post(URL, json=wrong_type))

    wrong_id = _offline_package()
    wrong_id["snapshot"]["attestations"][0]["target_id"] = (
        "evb_" + hashlib.sha256(b"other").hexdigest()
    )
    _assert_validation_error(client.post(URL, json=wrong_id))


# --- Malformed JSON and no-write boundary -------------------------------------


def test_malformed_json_is_422_and_writes_nothing(client, db_session):
    before = _counts(db_session)
    resp = client.post(
        URL,
        content='{"manifest":',
        headers={"content-type": "application/json"},
    )
    _assert_validation_error(resp)
    assert _counts(db_session) == before


def test_all_failures_write_nothing(client, db_session):
    _, _, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    valid_package = _served_package(client, bundle)

    mismatch = json.loads(json.dumps(valid_package))
    mismatch["manifest"]["manifest_digest_hex"] = "0" * 64
    tampered = json.loads(json.dumps(valid_package))
    tampered["snapshot"]["evidence_bundle"]["evidence"] = "AAAA"
    offline = _offline_package()
    del offline["snapshot"]["claim"]

    before = _counts(db_session)
    _assert_validation_error(client.post(URL, json=mismatch))
    _assert_validation_error(client.post(URL, json=tampered))
    _assert_validation_error(client.post(URL, json=offline))
    _assert_validation_error(
        client.post(URL, content="not json", headers={"content-type": "application/json"})
    )
    assert _counts(db_session) == before

    # The well-formed package still imports exactly once afterwards.
    ok = client.post(URL, json=valid_package)
    assert ok.status_code == 201
    assert (
        db_session.execute(
            select(func.count()).select_from(EvidenceBundleExchangeImport)
        ).scalar_one()
        == 1
    )


def test_failed_import_adds_no_audit_even_when_local_bundle_exists(
    client, db_session
):
    _, _, bundle = _setup_bundle(client)
    package = _served_package(client, bundle)
    package["manifest"]["manifest_digest_hex"] = "0" * 64

    events_before = db_session.execute(
        select(AuditEvent)
    ).scalars().all()
    _assert_validation_error(client.post(URL, json=package))
    events_after = db_session.execute(select(AuditEvent)).scalars().all()
    assert events_after == events_before
    assert all(
        e.event_type != EVENT_EVIDENCE_BUNDLE_EXCHANGE_IMPORTED
        for e in events_after
    )
