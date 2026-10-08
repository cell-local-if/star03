"""Tests for the read-only signed audit exchange-import reconciliation.

Covers GET
/v1/audit-exchanges/{import_id}/reconciliation:

* the receipt is matched by ``import_id`` exactly against
  ``audit_exchange_imports``; an unknown id is the existing
  ``404 audit_exchange_import_not_found``;
* only an empty GET is accepted -- any body bytes (even whitespace or
  malformed JSON) and any blank, unknown, illegal, or repeated query
  parameter is a ``422 validation_error`` rejected before the receipt
  lookup and before any local event is read;
* the complete, unfiltered local audit sequence is read once in stable
  creation order and rebuilt through exactly the public
  ``GET /v1/audit-events/checkpoint/package`` views; the response carries,
  in order, ``import_id``, the receipt's ``package_digest_hex``,
  ``received_at``, the four-field ``local_checkpoint``, the rebuilt
  whole-package ``local_package_digest_hex``, and ``matches`` -- true only
  when the rebuilt digest equals the receipt digest character for
  character; the empty sequence uses the zero-event count and empty-array
  digest and matches only an empty imported package;
* an unreadable database or a rebuild/digest failure is the
  existing-structure ``503 service_unavailable`` with a machine-readable
  reason, never a partial body;
* the route is strictly read-only -- no receipt, audit event, task, or
  resource is written on success, mismatch, 404, 422, or 503 -- repeated
  reads are identical, and the original package, the signature, private
  keys, and event bytes are never echoed.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance import canonical, ids
from provenance.api import get_db
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import (
    Actor,
    Attestation,
    AttestationRevocation,
    AuditEvent,
    AuditExchangeImportRecord,
    CheckpointImportRecord,
    Claim,
    Content,
    ContentRelation,
    EvidenceBundle,
    ExchangeImportRecord,
)
from tests.helpers import SEED_A, ed25519_public_key
from tests.test_audit_exchange_imports import (
    CHECKPOINT_VERSION,
    DIGEST_ALGORITHM,
    POST_PATH,
    SIGNATURE_VERSION,
    _events,
    _package,
    _request,
)

IMPORTS_URL = POST_PATH
CHECKPOINT_URL = "/v1/audit-events/checkpoint"
PACKAGE_URL = "/v1/audit-events/checkpoint/package"
EMPTY_EVENTS_DIGEST = hashlib.sha256(b"[]").hexdigest()
RECEIVED_AT = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

RESPONSE_KEYS = [
    "import_id",
    "package_digest_hex",
    "received_at",
    "local_checkpoint",
    "local_package_digest_hex",
    "matches",
]
CHECKPOINT_KEYS = [
    "checkpoint_version",
    "digest_algorithm",
    "event_count",
    "events_digest_hex",
]

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
    AuditExchangeImportRecord,
    AuditEvent,
)


def _reconciliation_url(import_id: str) -> str:
    return f"{IMPORTS_URL}/{import_id}/reconciliation"


def _empty_package() -> dict:
    return _package([])


def _package_digest(package: dict) -> str:
    return canonical.audit_checkpoint_package_digest_hex(package)


def _served_checkpoint(client) -> dict:
    resp = client.get(CHECKPOINT_URL)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _served_package(client) -> dict:
    resp = client.get(PACKAGE_URL)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _served_package_digest(client) -> str:
    # The package canonical rules keep the root order (checkpoint, events)
    # and sort nested keys; the served view already has that shape, so the
    # canonical digest reproduces directly from the wire body.
    return _package_digest(_served_package(client))


def _insert_event(db_session, event_type, resource_id, created_at):
    db_session.add(
        AuditEvent(
            event_type=event_type,
            resource_id=resource_id,
            created_at=created_at,
        )
    )
    db_session.commit()


def _insert_receipt(db_session, package_digest_hex: str) -> str:
    """Persist an exchange-import receipt directly, without an audit event.

    The reconciliation reads persisted state alone; direct insertion keeps
    the surrounding audit sequence exactly controlled.
    """
    receipt_id = ids.audit_exchange_import_id(
        SIGNATURE_VERSION, package_digest_hex
    )
    db_session.add(
        AuditExchangeImportRecord(
            id=receipt_id,
            signature_version=SIGNATURE_VERSION,
            signer_subject="remote-system",
            public_key=ed25519_public_key(SEED_A),
            package_digest_algorithm=DIGEST_ALGORITHM,
            package_digest_hex=package_digest_hex,
            signature_digest_algorithm=DIGEST_ALGORITHM,
            signature_digest_hex=hashlib.sha256(b"signature").hexdigest(),
            created_at=RECEIVED_AT,
        )
    )
    db_session.commit()
    return receipt_id


def _matching_receipt(db_session, client) -> tuple[str, dict, str]:
    """A receipt whose whole-package digest equals the current local state."""
    checkpoint = _served_checkpoint(client)
    digest_hex = _served_package_digest(client)
    receipt_id = _insert_receipt(db_session, digest_hex)
    return receipt_id, checkpoint, digest_hex


def _counts(db_session):
    return {
        model: db_session.execute(
            select(func.count()).select_from(model)
        ).scalar_one()
        for model in _DOMAIN_MODELS
    }


# --- Success shape ---------------------------------------------------------------


def test_empty_local_sequence_matches_empty_package_receipt(client, db_session):
    digest_hex = _package_digest(_empty_package())
    assert digest_hex == _package_digest(
        {
            "checkpoint": {
                "checkpoint_version": CHECKPOINT_VERSION,
                "digest_algorithm": DIGEST_ALGORITHM,
                "event_count": 0,
                "events_digest_hex": EMPTY_EVENTS_DIGEST,
            },
            "events": [],
        }
    )
    receipt_id = _insert_receipt(db_session, digest_hex)

    resp = client.get(_reconciliation_url(receipt_id))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert list(body) == RESPONSE_KEYS
    assert list(body["local_checkpoint"]) == CHECKPOINT_KEYS
    assert body == {
        "import_id": receipt_id,
        "package_digest_hex": digest_hex,
        "received_at": RECEIVED_AT.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "local_checkpoint": {
            "checkpoint_version": CHECKPOINT_VERSION,
            "digest_algorithm": DIGEST_ALGORITHM,
            "event_count": 0,
            "events_digest_hex": EMPTY_EVENTS_DIGEST,
        },
        "local_package_digest_hex": digest_hex,
        "matches": True,
    }
    assert body["received_at"].endswith(("Z", "+00:00"))


def test_matching_receipt_reports_local_views_and_matches(client, db_session):
    _insert_event(
        db_session,
        "actor.created",
        "org-émoji-✓",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    _insert_event(
        db_session,
        "content.created",
        "cnt_t2",
        datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    receipt_id, checkpoint, digest_hex = _matching_receipt(db_session, client)

    resp = client.get(_reconciliation_url(receipt_id))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert list(body) == RESPONSE_KEYS
    assert body["import_id"] == receipt_id
    assert body["package_digest_hex"] == digest_hex
    assert body["local_checkpoint"] == checkpoint
    assert body["local_checkpoint"] == _served_checkpoint(client)
    assert body["local_package_digest_hex"] == digest_hex
    assert body["local_package_digest_hex"] == _served_package_digest(client)
    assert body["matches"] is True


def test_response_is_compact_utf8_json_with_one_newline(client, db_session):
    receipt_id, _, _ = _matching_receipt(db_session, client)
    raw = client.get(_reconciliation_url(receipt_id)).content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    assert raw.startswith(b'{"import_id":"acx_')


def test_non_matching_digest_reports_local_state_with_matches_false(
    client, db_session
):
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    local_checkpoint = _served_checkpoint(client)
    local_digest = _served_package_digest(client)
    other_digest = "a" * 64
    assert other_digest != local_digest
    receipt_id = _insert_receipt(db_session, other_digest)

    body = client.get(_reconciliation_url(receipt_id)).json()
    assert body["package_digest_hex"] == other_digest
    assert body["local_checkpoint"] == local_checkpoint
    assert body["local_package_digest_hex"] == local_digest
    assert body["matches"] is False


def test_empty_sequence_against_non_empty_package_receipt_does_not_match(
    client, db_session
):
    # A receipt for a two-event package on an empty local trail.
    digest_hex = _package_digest(_package(_events(2)))
    receipt_id = _insert_receipt(db_session, digest_hex)

    body = client.get(_reconciliation_url(receipt_id)).json()
    assert body["local_checkpoint"]["event_count"] == 0
    assert body["local_checkpoint"]["events_digest_hex"] == EMPTY_EVENTS_DIGEST
    assert body["local_package_digest_hex"] == _package_digest(_empty_package())
    assert body["local_package_digest_hex"] != digest_hex
    assert body["matches"] is False


def test_grown_trail_against_empty_package_receipt_does_not_match(
    client, db_session
):
    # Start matching (empty local sequence, empty package receipt).
    receipt_id = _insert_receipt(db_session, _package_digest(_empty_package()))
    assert client.get(_reconciliation_url(receipt_id)).json()["matches"] is True

    # One local event appears after registration: the rebuilt package no
    # longer equals the receipt's empty package.
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    current_checkpoint = _served_checkpoint(client)
    current_digest = _served_package_digest(client)
    assert current_checkpoint["event_count"] == 1
    assert current_digest != _package_digest(_empty_package())

    body = client.get(_reconciliation_url(receipt_id)).json()
    assert body["local_checkpoint"] == current_checkpoint
    assert body["local_package_digest_hex"] == current_digest
    assert body["matches"] is False


def test_reconciliation_after_a_real_import_diverges_by_its_own_audit_event(
    client, db_session
):
    # Import an empty package through the real route. The import appends its
    # own audit.exchange_imported event, so the registered receipt (an empty
    # package) cannot match the grown one-event local sequence.
    response = client.post(IMPORTS_URL, json=_request(_empty_package())[0])
    assert response.status_code == 201, response.text
    receipt = response.json()

    body = client.get(_reconciliation_url(receipt["id"])).json()
    assert body["import_id"] == receipt["id"]
    assert body["package_digest_hex"] == receipt["package_digest_hex"]
    assert body["received_at"] == receipt["received_at"]
    assert body["local_checkpoint"]["event_count"] == 1
    assert body["local_package_digest_hex"] == _served_package_digest(client)
    assert body["local_package_digest_hex"] != receipt["package_digest_hex"]
    assert body["matches"] is False


def test_local_package_rebuild_uses_stable_creation_order(client, db_session):
    # Two events with an out-of-order insertion sequence but ordered
    # timestamps; the rebuilt digest must follow the stable creation order
    # served by the package route, not insertion order.
    _insert_event(
        db_session,
        "content.created",
        "cnt_later",
        datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    _insert_event(
        db_session,
        "actor.created",
        "org_earlier",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    receipt_id = _matching_receipt(db_session, client)[0]
    body = client.get(_reconciliation_url(receipt_id)).json()
    assert body["local_checkpoint"]["event_count"] == 2
    assert body["local_package_digest_hex"] == _served_package_digest(client)
    assert body["matches"] is True


def test_repeated_reads_are_identical(client, db_session):
    receipt_id, _, _ = _matching_receipt(db_session, client)
    first = client.get(_reconciliation_url(receipt_id))
    second = client.get(_reconciliation_url(receipt_id))
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_reconciliation_is_deterministic_across_restart(file_client, tmp_db_url):
    session = file_client.app.state.session_factory()
    try:
        session.add(
            AuditEvent(
                event_type="actor.created",
                resource_id="org-1",
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        )
        digest = canonical.audit_checkpoint_package_digest_hex(
            {
                "checkpoint": {
                    "checkpoint_version": CHECKPOINT_VERSION,
                    "digest_algorithm": DIGEST_ALGORITHM,
                    "event_count": 1,
                    "events_digest_hex": canonical.audit_events_digest_hex(
                        [
                            {
                                "event_type": "actor.created",
                                "resource_id": "org-1",
                                "created_at": "2026-01-01T00:00:00Z",
                            }
                        ]
                    ),
                },
                "events": [
                    {
                        "event_type": "actor.created",
                        "resource_id": "org-1",
                        "created_at": "2026-01-01T00:00:00Z",
                    }
                ],
            }
        )
        receipt_id = ids.audit_exchange_import_id(SIGNATURE_VERSION, digest)
        session.add(
            AuditExchangeImportRecord(
                id=receipt_id,
                signature_version=SIGNATURE_VERSION,
                signer_subject="remote-system",
                public_key=ed25519_public_key(SEED_A),
                package_digest_algorithm=DIGEST_ALGORITHM,
                package_digest_hex=digest,
                signature_digest_algorithm=DIGEST_ALGORITHM,
                signature_digest_hex=hashlib.sha256(b"signature").hexdigest(),
                created_at=RECEIVED_AT,
            )
        )
        session.commit()
    finally:
        session.close()

    expected = file_client.get(_reconciliation_url(receipt_id))
    assert expected.status_code == 200, expected.text
    assert expected.json()["matches"] is True

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as restarted_client:
        resp = restarted_client.get(_reconciliation_url(receipt_id))
        assert resp.status_code == 200, resp.text
        assert resp.json() == expected.json()


# --- Missing receipt -------------------------------------------------------------


def test_unknown_import_id_is_404(client, db_session):
    unknown = "acx_" + "0" * 64
    resp = client.get(_reconciliation_url(unknown))
    assert resp.status_code == 404, resp.text
    error = resp.json()["error"]
    assert error["code"] == "audit_exchange_import_not_found"
    assert error["details"]["import_id"] == unknown
    assert db_session.execute(select(AuditEvent)).scalars().all() == []


def test_unshaped_and_other_prefix_ids_are_404(client):
    for raw_id in (
        "acx_doesnotexist",
        "aci_" + "a" * 64,
        "arx_" + "a" * 64,
        "not-an-id",
    ):
        resp = client.get(_reconciliation_url(raw_id))
        assert resp.status_code == 404, raw_id
        assert (
            resp.json()["error"]["code"]
            == "audit_exchange_import_not_found"
        )


def test_404_even_when_local_events_exist(client, db_session):
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    resp = client.get(_reconciliation_url("acx_" + "f" * 64))
    assert resp.status_code == 404
    assert (
        resp.json()["error"]["code"] == "audit_exchange_import_not_found"
    )


# --- Empty-GET boundary ----------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    (
        "?limit=10",
        "?cursor=abc",
        "?unknown=",
        "?a=1&b=2",
        "?a=1&a=2",
    ),
)
def test_any_query_parameter_is_422(client, db_session, url):
    receipt_id, _, _ = _matching_receipt(db_session, client)
    resp = client.get(_reconciliation_url(receipt_id) + url)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def test_query_parameter_is_422_before_the_receipt_lookup(client):
    resp = client.get(_reconciliation_url("acx_doesnotexist") + "?anything=1")
    assert resp.status_code == 422, resp.text
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"]["issues"][0]["loc"] == ["query", "anything"]


def test_repeated_parameter_is_422_before_the_receipt_lookup(client):
    resp = client.get(
        _reconciliation_url("acx_" + "0" * 64) + "?x=1&x=2"
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize(
    "content",
    (b"  ", b"\t\n", b"{}", b"[]", b"not json", b"\x00\x01"),
)
def test_non_empty_body_is_422(client, db_session, content):
    receipt_id, _, _ = _matching_receipt(db_session, client)
    resp = client.request(
        "GET",
        _reconciliation_url(receipt_id),
        content=content,
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422, resp.text
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"]["issues"][0]["loc"] == ["query", "body"]


def test_body_is_422_before_the_receipt_lookup(client):
    resp = client.request(
        "GET",
        _reconciliation_url("acx_doesnotexist"),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def test_body_is_422_before_query_parameters_and_lookup(client):
    # Both invalid together: the body is rejected first, and nothing is read.
    resp = client.request(
        "GET",
        _reconciliation_url("acx_doesnotexist") + "?x=1",
        content=b"  ",
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize("method", ("put", "patch", "delete", "post"))
def test_non_get_methods_are_405(client, db_session, method):
    receipt_id, _, _ = _matching_receipt(db_session, client)
    resp = getattr(client, method)(_reconciliation_url(receipt_id))
    assert resp.status_code == 405
    assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Read-only and no-echo boundary ----------------------------------------------


def test_reconciliation_writes_nothing(client, db_session):
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    matching, _, _ = _matching_receipt(db_session, client)
    non_matching = _insert_receipt(db_session, "b" * 64)

    before = _counts(db_session)

    assert client.get(_reconciliation_url(matching)).status_code == 200
    assert client.get(_reconciliation_url(non_matching)).status_code == 200
    assert (
        client.get(_reconciliation_url(matching) + "?x=1").status_code == 422
    )
    assert client.request(
        "GET", _reconciliation_url(matching), content=b"{}"
    ).status_code == 422
    assert (
        client.get(_reconciliation_url("acx_" + "0" * 64)).status_code == 404
    )

    db_session.expire_all()
    assert _counts(db_session) == before


def test_response_never_echoes_package_signature_or_event_bytes(
    client, db_session
):
    marker = "reconciliation-no-echo-marker-7e21"
    package = _package(
        [
            {
                "event_type": "actor.created",
                "resource_id": marker,
                "created_at": "2026-01-02T03:04:05Z",
            }
        ]
    )
    request, metadata = _request(package)
    receipt = client.post(IMPORTS_URL, json=request).json()

    resp = client.get(_reconciliation_url(receipt["id"]))
    assert resp.status_code == 200
    body = resp.json()
    assert list(body) == RESPONSE_KEYS
    assert list(body["local_checkpoint"]) == CHECKPOINT_KEYS
    raw = resp.text

    # The imported event array and its bytes are absent in every form.
    assert '"events"' not in raw
    assert marker not in raw
    assert "2026-01-02T03:04:05Z" not in raw
    # The original signature, a private key, and the package envelope are
    # never carried; only the receipt's public digests and counts appear.
    assert metadata["signature"] not in raw
    assert '"signature"' not in raw
    assert '"public_key"' not in raw
    assert '"signer_subject"' not in raw
    assert '"private_key"' not in raw


# --- 503 service_unavailable -----------------------------------------------------


def _client_without_table(file_app, table: str):
    # A TestClient created without entering its context never runs the
    # lifespan startup, so dropping a table leaves the schema unreadable
    # instead of being silently recreated.
    from sqlalchemy import text

    with file_app.state.engine.begin() as conn:
        conn.execute(text(f"DROP TABLE {table}"))
    return TestClient(file_app)


def test_unreadable_events_table_returns_503_with_reason(file_app):
    digest = _package_digest(_empty_package())
    session = file_app.state.session_factory()
    try:
        receipt_id = _insert_receipt(session, digest)
    finally:
        session.close()

    client = _client_without_table(file_app, "audit_events")
    resp = client.get(_reconciliation_url(receipt_id))
    assert resp.status_code == 503, resp.text
    error = resp.json()["error"]
    assert error["code"] == "service_unavailable"
    assert "message" in error
    assert error["details"]["reason"] == "database_unavailable"
    # No partial body shape is served on failure.
    assert "local_package_digest_hex" not in resp.text


def test_unreadable_receipts_table_returns_503(file_app):
    client = _client_without_table(file_app, "audit_exchange_imports")
    resp = client.get(_reconciliation_url("acx_" + "0" * 64))
    assert resp.status_code == 503, resp.text
    error = resp.json()["error"]
    assert error["code"] == "service_unavailable"
    assert error["details"]["reason"] == "database_unavailable"


def test_validation_runs_before_the_unreadable_database(file_app):
    # Empty or illegal input is rejected first; the database is never read,
    # so an unreadable events table still answers these with 422. The valid
    # empty request resolves the existing receipt, reaches the failed event
    # read, and gets the 503 rather than a partial body.
    digest = _package_digest(_empty_package())
    session = file_app.state.session_factory()
    try:
        receipt_id = _insert_receipt(session, digest)
    finally:
        session.close()

    client = _client_without_table(file_app, "audit_events")
    url = _reconciliation_url(receipt_id)

    whitespace = client.request("GET", url, content=b"  ")
    assert whitespace.status_code == 422, whitespace.text
    unknown_param = client.get(url + "?x=1")
    assert unknown_param.status_code == 422, unknown_param.text
    repeated = client.get(url + "?x=1&x=2")
    assert repeated.status_code == 422, repeated.text

    valid = client.get(url)
    assert valid.status_code == 503, valid.text


class _BrokenReconciliationSession:
    """A session whose reads fail with a non-database internal error."""

    def execute(self, *_args, **_kwargs):
        raise RuntimeError("reconciliation machinery broken")

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_internal_failure_is_503_with_reason(app, client):
    app.dependency_overrides[get_db] = lambda: _BrokenReconciliationSession()
    try:
        resp = client.get(_reconciliation_url("acx_" + "0" * 64))
    finally:
        app.dependency_overrides.pop(get_db, None)
    assert resp.status_code == 503, resp.text
    error = resp.json()["error"]
    assert error["code"] == "service_unavailable"
    assert error["details"]["reason"] == "internal_error"
