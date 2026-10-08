"""Tests for the read-only signed audit exchange reconciliation endpoint.

Covers GET
``/v1/audit-exchanges/{import_id}/reconciliation``:

* the endpoint accepts only an empty GET -- any request body (including
  whitespace, arbitrary bytes, or malformed JSON) and any empty,
  illegal, repeated, or undeclared query parameter is a 422
  ``validation_error`` rejected before the receipt is looked up or any
  local audit event is read;
* ``import_id`` matches ``audit_exchange_imports`` exactly and an
  unknown id is the existing ``404 audit_exchange_import_not_found``;
* the success body is exactly, in order, ``import_id``,
  ``package_digest_hex`` (the receipt digest), ``received_at``,
  ``local_checkpoint`` (the four-field checkpoint rebuilt over the
  complete unfiltered local sequence under the existing
  ``GET /v1/audit-events/checkpoint`` rules),
  ``local_package_digest_hex`` (the whole-package digest rebuilt under
  the ``GET /v1/audit-events/checkpoint/package`` rules), and
  ``matches`` -- true only when the rebuilt package digest equals the
  receipt digest;
* the empty local sequence yields a zero-event checkpoint, the digest
  of ``[]``, and the empty-package digest, matching only a receipt for
  the empty package;
* the read is strictly read-only (no receipt, audit event, task, or
  resource is written), deterministic on repeat and across restart,
  echoes neither the original package events nor signature/key bytes,
  and an unreadable database or rebuild/digest failure is a 503
  ``service_unavailable`` (existing error response) with no partial
  result.

All fixtures are deterministic and offline.
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text

from provenance import canonical, ids
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import AuditEvent, AuditExchangeImportRecord
from provenance.signing import audit_exchange_message_bytes
from tests.helpers import SEED_A, ed25519_public_key, ed25519_sign

POST_PATH = "/v1/audit-exchanges"
CHECKPOINT_URL = "/v1/audit-events/checkpoint"
PACKAGE_URL = "/v1/audit-events/checkpoint/package"
EVENTS_URL = "/v1/audit-events"
SIGNATURE_VERSION = "provenance-audit-exchange-v1"
CHECKPOINT_VERSION = "provenance-audit-checkpoint-v1"
DIGEST_ALGORITHM = "sha256"
EVENT_TYPE = "audit.exchange_imported"

RECONCILIATION_KEYS = {
    "import_id",
    "package_digest_hex",
    "received_at",
    "local_checkpoint",
    "local_package_digest_hex",
    "matches",
}
RECONCILIATION_ORDER = [
    "import_id",
    "package_digest_hex",
    "received_at",
    "local_checkpoint",
    "local_package_digest_hex",
    "matches",
]
CHECKPOINT_KEYS = {
    "checkpoint_version",
    "digest_algorithm",
    "event_count",
    "events_digest_hex",
}
EMPTY_EVENTS_DIGEST = hashlib.sha256(b"[]").hexdigest()
EMPTY_PACKAGE_DIGEST = canonical.audit_checkpoint_package_digest_hex(
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


# --- Payload / state construction --------------------------------------------------


def _events(n: int = 0) -> list[dict]:
    """Deterministic, self-contained audit-event arrays for offline tests."""
    return [
        {
            "event_type": f"event.type.{i % 3}",
            "resource_id": f"resource-{i}",
            "created_at": f"2026-01-{(i % 28) + 1:02d}T12:30:0{i % 10}Z",
        }
        for i in range(n)
    ]


def _package(events: list[dict]) -> dict:
    """Build a structurally valid audit checkpoint package for ``events``."""
    return {
        "checkpoint": {
            "checkpoint_version": CHECKPOINT_VERSION,
            "digest_algorithm": DIGEST_ALGORITHM,
            "event_count": len(events),
            "events_digest_hex": canonical.audit_events_digest_hex(events),
        },
        "events": events,
    }


def _signed_request(package: dict, *, subject="remote-system", seed=SEED_A):
    """Build a valid signed import request for one package."""
    digest_hex = canonical.audit_checkpoint_package_digest_hex(package)
    public_key = ed25519_public_key(seed)
    message = audit_exchange_message_bytes(subject, DIGEST_ALGORITHM, digest_hex)
    signature = ed25519_sign(seed, message)
    metadata = {
        "signature_version": SIGNATURE_VERSION,
        "signer_subject": subject,
        "public_key": base64.b64encode(public_key).decode("ascii"),
        "signature": base64.b64encode(signature).decode("ascii"),
        "package_digest_algorithm": DIGEST_ALGORITHM,
        "package_digest_hex": digest_hex,
    }
    return {"package": package, "signature_metadata": metadata}, metadata


def _reconciliation_url(import_id: str) -> str:
    return f"{POST_PATH}/{import_id}/reconciliation"


def _served_checkpoint(client) -> dict:
    resp = client.get(CHECKPOINT_URL)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _served_package(client) -> dict:
    resp = client.get(PACKAGE_URL)
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


def _insert_receipt(
    db_session,
    package_digest_hex: str,
    *,
    created_at=None,
    signature_version: str = SIGNATURE_VERSION,
):
    """Persist an exchange receipt directly, without an import audit event.

    The reconciliation reads persisted state alone; direct insertion keeps
    the surrounding audit sequence exactly controlled.
    """
    receipt_id = ids.audit_exchange_import_id(signature_version, package_digest_hex)
    db_session.add(
        AuditExchangeImportRecord(
            id=receipt_id,
            signature_version=signature_version,
            signer_subject="remote-system",
            public_key=b"\x01" * 32,
            package_digest_algorithm=DIGEST_ALGORITHM,
            package_digest_hex=package_digest_hex,
            signature_digest_algorithm=DIGEST_ALGORITHM,
            signature_digest_hex="a" * 64,
            created_at=created_at,
        )
    )
    db_session.commit()
    return receipt_id


def _matching_receipt(db_session, client) -> tuple[str, dict, str]:
    """A receipt whose package digest equals the current local package."""
    package = _served_package(client)
    digest_hex = canonical.audit_checkpoint_package_digest_hex(package)
    receipt_id = _insert_receipt(db_session, digest_hex)
    return receipt_id, package["checkpoint"], digest_hex


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def _receipt_count(db_session) -> int:
    return db_session.scalar(
        select(func.count()).select_from(AuditExchangeImportRecord)
    )


def _event_count(db_session) -> int:
    return db_session.scalar(select(func.count()).select_from(AuditEvent))


# --- Success shape and semantics ---------------------------------------------------


def test_empty_library_matches_only_the_empty_package(client, db_session):
    # Empty local sequence: zero-event checkpoint, digest of [], and the
    # empty-package digest; matches true only for an empty-package receipt.
    receipt_id = _insert_receipt(db_session, EMPTY_PACKAGE_DIGEST)

    resp = client.get(_reconciliation_url(receipt_id))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == RECONCILIATION_KEYS
    assert list(body) == RECONCILIATION_ORDER
    assert body["import_id"] == receipt_id
    assert body["package_digest_hex"] == EMPTY_PACKAGE_DIGEST
    assert body["received_at"].endswith(("Z", "+00:00"))
    assert set(body["local_checkpoint"]) == CHECKPOINT_KEYS
    assert body["local_checkpoint"] == {
        "checkpoint_version": CHECKPOINT_VERSION,
        "digest_algorithm": DIGEST_ALGORITHM,
        "event_count": 0,
        "events_digest_hex": EMPTY_EVENTS_DIGEST,
    }
    assert body["local_package_digest_hex"] == EMPTY_PACKAGE_DIGEST
    assert body["matches"] is True


def test_non_empty_receipt_against_empty_library_does_not_match(
    client, db_session
):
    package = _package(_events(2))
    digest_hex = canonical.audit_checkpoint_package_digest_hex(package)
    receipt_id = _insert_receipt(db_session, digest_hex)

    body = client.get(_reconciliation_url(receipt_id)).json()
    assert body["local_checkpoint"]["event_count"] == 0
    assert body["local_checkpoint"]["events_digest_hex"] == EMPTY_EVENTS_DIGEST
    assert body["local_package_digest_hex"] == EMPTY_PACKAGE_DIGEST
    assert body["package_digest_hex"] == digest_hex
    assert body["matches"] is False


def test_matching_receipt_rebuilds_served_checkpoint_and_package(
    client, db_session
):
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
    assert body["matches"] is True
    assert body["package_digest_hex"] == digest_hex
    assert body["local_checkpoint"] == checkpoint
    assert body["local_checkpoint"] == _served_checkpoint(client)
    # The local package digest is independently reproducible from the
    # served checkpoint/package body alone.
    served_package = _served_package(client)
    assert (
        body["local_package_digest_hex"]
        == canonical.audit_checkpoint_package_digest_hex(served_package)
    )
    # The checkpoint's events digest is that of the served events array.
    served = _served_events(client)
    assert body["local_checkpoint"]["event_count"] == len(served)
    assert (
        body["local_checkpoint"]["events_digest_hex"]
        == canonical.audit_events_digest_hex(served)
    )


def test_received_at_and_digest_come_from_the_stored_receipt(
    client, db_session
):
    registered_at = datetime(2026, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
    receipt_id = _insert_receipt(
        db_session, EMPTY_PACKAGE_DIGEST, created_at=registered_at
    )

    detail = client.get(f"{POST_PATH}/{receipt_id}")
    assert detail.status_code == 200
    receipt = detail.json()

    body = client.get(_reconciliation_url(receipt_id)).json()
    assert body["import_id"] == receipt["id"]
    assert body["package_digest_hex"] == receipt["package_digest_hex"]
    assert body["received_at"] == receipt["received_at"]


def test_local_sequence_growing_after_registration_flips_match_to_false(
    client, db_session
):
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    receipt_id, _, _ = _matching_receipt(db_session, client)
    assert client.get(_reconciliation_url(receipt_id)).json()["matches"] is True

    # The local sequence changes after registration: the rebuilt package
    # differs, so matches flips to false while the current local package is
    # still reported in full.
    _insert_event(
        db_session,
        "content.created",
        "cnt_later",
        datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    body = client.get(_reconciliation_url(receipt_id)).json()
    current_package = _served_package(client)
    assert body["local_checkpoint"]["event_count"] == 2
    assert body["local_checkpoint"] == current_package["checkpoint"]
    assert (
        body["local_package_digest_hex"]
        == canonical.audit_checkpoint_package_digest_hex(current_package)
    )
    assert body["package_digest_hex"] != body["local_package_digest_hex"]
    assert body["matches"] is False


def test_any_digest_inequality_is_a_mismatch(client, db_session):
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    # A well-formed 64-hex package digest that is simply not the current
    # one: matches false, local state still fully reported.
    receipt_id = _insert_receipt(db_session, "f" * 64)
    body = client.get(_reconciliation_url(receipt_id)).json()
    assert body["matches"] is False
    assert body["local_checkpoint"]["event_count"] == 1
    assert (
        body["local_package_digest_hex"]
        == canonical.audit_checkpoint_package_digest_hex(_served_package(client))
    )


def test_same_timestamp_events_follow_stable_creation_order(client, db_session):
    instant = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _insert_event(db_session, "actor.created", "org-first", instant)
    _insert_event(db_session, "actor.created", "org-second", instant)
    receipt_id, _, _ = _matching_receipt(db_session, client)

    body = client.get(_reconciliation_url(receipt_id)).json()
    assert body["matches"] is True
    served = _served_events(client)
    assert [item["resource_id"] for item in served] == [
        "org-first",
        "org-second",
    ]


# --- End-to-end import flow --------------------------------------------------------


def test_api_imported_empty_package_diverges_by_its_own_audit_event(client):
    request, metadata = _signed_request(_package([]))
    posted = client.post(POST_PATH, json=request)
    assert posted.status_code == 201, posted.text
    receipt = posted.json()
    assert receipt["package_digest_hex"] == EMPTY_PACKAGE_DIGEST
    assert metadata["package_digest_hex"] == EMPTY_PACKAGE_DIGEST

    # The import appended its own audit.exchange_imported event, so the
    # rebuilt local package is no longer the empty package: matches false.
    body = client.get(_reconciliation_url(receipt["id"])).json()
    assert body["import_id"] == receipt["id"]
    assert body["package_digest_hex"] == EMPTY_PACKAGE_DIGEST
    assert body["local_checkpoint"]["event_count"] == 1
    assert body["matches"] is False
    served_package = _served_package(client)
    assert (
        body["local_package_digest_hex"]
        == canonical.audit_checkpoint_package_digest_hex(served_package)
    )
    assert served_package["events"][0]["event_type"] == EVENT_TYPE
    assert served_package["events"][0]["resource_id"] == receipt["id"]


def test_offline_package_never_materialized_locally_does_not_match(client):
    # A package describing events from another system: the events are not
    # created locally, so the rebuilt package cannot equal the receipt's.
    request, metadata = _signed_request(_package(_events(3)))
    receipt = client.post(POST_PATH, json=request).json()

    body = client.get(_reconciliation_url(receipt["id"])).json()
    assert body["package_digest_hex"] == metadata["package_digest_hex"]
    # Only the import's own audit event exists locally.
    assert body["local_checkpoint"]["event_count"] == 1
    assert body["matches"] is False


# --- Response framing --------------------------------------------------------------


def test_response_is_compact_utf8_json_with_one_newline(client, db_session):
    receipt_id = _insert_receipt(db_session, EMPTY_PACKAGE_DIGEST)
    raw = client.get(_reconciliation_url(receipt_id)).content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    assert raw.startswith(b'{"import_id":"acx_')


# --- Missing receipt ---------------------------------------------------------------


def test_unknown_import_id_is_an_explicit_specific_404(client, db_session):
    unknown = "acx_" + "0" * 64
    resp = client.get(_reconciliation_url(unknown))
    assert resp.status_code == 404, resp.text
    error = resp.json()["error"]
    assert error["code"] == "audit_exchange_import_not_found"
    assert error["details"]["import_id"] == unknown
    # The failed read wrote nothing.
    assert _event_count(db_session) == 0


def test_unshaped_and_other_prefix_ids_are_404(client):
    for raw_id in ("acx_doesnotexist", "aci_" + "a" * 64, "not-an-id"):
        resp = client.get(_reconciliation_url(raw_id))
        assert resp.status_code == 404, raw_id
        assert (
            resp.json()["error"]["code"]
            == "audit_exchange_import_not_found"
        )


# --- Empty-GET boundary ------------------------------------------------------------


@pytest.mark.parametrize(
    "body,headers",
    [
        (b'{"a":1}', {"content-type": "application/json"}),
        (b" ", {"content-type": "application/json"}),
        (b"{}", {"content-type": "application/json"}),
        (b"{not json", {"content-type": "application/json"}),
        (b"x=1", {"content-type": "application/x-www-form-urlencoded"}),
        (b"\x00", {}),
    ],
)
def test_any_get_body_is_422(client, db_session, body, headers):
    receipt_id = _insert_receipt(db_session, EMPTY_PACKAGE_DIGEST)
    resp = client.request(
        "GET", _reconciliation_url(receipt_id), content=body, headers=headers
    )
    _assert_validation_error(resp)


@pytest.mark.parametrize(
    "suffix",
    (
        "?x=1",
        "?import_id=x",
        "?limit=1",
        "?cursor=abc",
        "?unknown=",
        "?a=1&b=2",
        "?x=1&x=2",
    ),
)
def test_any_query_parameter_is_422(client, db_session, suffix):
    receipt_id = _insert_receipt(db_session, EMPTY_PACKAGE_DIGEST)
    _assert_validation_error(
        client.get(_reconciliation_url(receipt_id) + suffix)
    )


def test_body_and_query_are_422_before_the_receipt_lookup(client):
    # Validation precedes the existence check: body or parameters on an
    # unknown id are still 422, not 404.
    _assert_validation_error(
        client.get(_reconciliation_url("acx_doesnotexist") + "?x=1")
    )
    _assert_validation_error(
        client.request(
            "GET",
            _reconciliation_url("acx_doesnotexist"),
            content=b"{}",
            headers={"content-type": "application/json"},
        )
    )


@pytest.mark.parametrize("method", ("put", "patch", "delete", "post"))
def test_non_get_methods_are_405(client, db_session, method):
    receipt_id = _insert_receipt(db_session, EMPTY_PACKAGE_DIGEST)
    response = getattr(client, method)(_reconciliation_url(receipt_id))
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"


# --- Read-only and no-echo boundary ------------------------------------------------


def test_reconciliation_writes_nothing_on_every_outcome(client, db_session):
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    matching, _, _ = _matching_receipt(db_session, client)
    non_matching = _insert_receipt(db_session, "a" * 64)

    receipts_before = _receipt_count(db_session)
    events_before = _event_count(db_session)

    assert client.get(_reconciliation_url(matching)).status_code == 200
    assert client.get(_reconciliation_url(non_matching)).status_code == 200
    assert (
        client.get(_reconciliation_url(matching) + "?x=1").status_code == 422
    )
    assert (
        client.request(
            "GET",
            _reconciliation_url(matching),
            content=b" ",
            headers={"content-type": "application/json"},
        ).status_code
        == 422
    )
    assert (
        client.get(_reconciliation_url("acx_" + "0" * 64)).status_code == 404
    )

    db_session.expire_all()
    assert _receipt_count(db_session) == receipts_before
    assert _event_count(db_session) == events_before


def test_repeated_reads_are_identical(client, db_session):
    _insert_event(
        db_session,
        "actor.created",
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    matching, _, _ = _matching_receipt(db_session, client)

    first = client.get(_reconciliation_url(matching))
    second = client.get(_reconciliation_url(matching))
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.content == second.content


def test_response_never_echoes_package_signature_or_key_bytes(client):
    marker = "exchange-reconciliation-no-echo-marker-9d2c"
    events = _events(2)
    events[0]["resource_id"] = marker
    package = _package(events)
    request, metadata = _signed_request(package)
    receipt = client.post(POST_PATH, json=request).json()

    resp = client.get(_reconciliation_url(receipt["id"]))
    assert resp.status_code == 200
    serialized = resp.text
    assert set(resp.json()) == RECONCILIATION_KEYS
    # The imported event array and its raw values are absent in every form.
    assert '"events"' not in serialized
    assert marker not in serialized
    assert events[0]["created_at"] not in serialized
    # No signature, public key, signing subject, or private material.
    assert metadata["signature"] not in serialized
    assert metadata["public_key"] not in serialized
    assert "remote-system" not in serialized
    assert '"signature"' not in serialized
    assert '"public_key"' not in serialized
    assert '"private_key"' not in serialized


def test_reconciliation_is_deterministic_across_restart(
    file_client, tmp_db_url
):
    session = file_client.app.state.session_factory()
    try:
        receipt_id = _insert_receipt(session, EMPTY_PACKAGE_DIGEST)
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


# --- 503 service_unavailable -------------------------------------------------------


def _client_with_dropped_table(file_app, table: str):
    # A TestClient created without entering its context never runs the
    # lifespan startup, so dropping a table here leaves the schema
    # unreadable instead of being silently recreated.
    with file_app.state.engine.begin() as conn:
        conn.execute(text(f"DROP TABLE {table}"))
    return TestClient(file_app)


def test_unreadable_receipt_table_is_503_without_partial_result(file_app):
    client = _client_with_dropped_table(file_app, "audit_exchange_imports")
    resp = client.get(_reconciliation_url("acx_" + "0" * 64))
    assert resp.status_code == 503, resp.text
    error = resp.json()["error"]
    assert error["code"] == "service_unavailable"
    assert "message" in error
    assert error["details"]["reason"] == "database_unavailable"


def test_unreadable_events_table_is_503_without_partial_result(file_app):
    # Seed a receipt directly (the import route also writes an audit
    # event, which is impossible after the audit table is dropped).
    receipt_id = ids.audit_exchange_import_id(
        SIGNATURE_VERSION, EMPTY_PACKAGE_DIGEST
    )
    with file_app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO audit_exchange_imports "
                "(id, signature_version, signer_subject, public_key, "
                "package_digest_algorithm, package_digest_hex, "
                "signature_digest_algorithm, signature_digest_hex, "
                "created_at) VALUES "
                "(:id, :sv, :sub, :pk, :da, :dh, :sa, :sh, :ts)"
            ),
            {
                "id": receipt_id,
                "sv": SIGNATURE_VERSION,
                "sub": "remote-system",
                "pk": b"\x01" * 32,
                "da": DIGEST_ALGORITHM,
                "dh": EMPTY_PACKAGE_DIGEST,
                "sa": DIGEST_ALGORITHM,
                "sh": "a" * 64,
                "ts": "2026-01-01 00:00:00",
            },
        )
    client = _client_with_dropped_table(file_app, "audit_events")
    resp = client.get(_reconciliation_url(receipt_id))
    assert resp.status_code == 503, resp.text
    error = resp.json()["error"]
    assert error["code"] == "service_unavailable"
    assert error["details"]["reason"] == "database_unavailable"


def test_validation_runs_before_the_unreadable_database(file_app):
    # Body/parameter rejection happens before any database read: an
    # unreadable database still answers these with 422 rather than 503.
    client = _client_with_dropped_table(file_app, "audit_exchange_imports")

    whitespace = client.request(
        "GET",
        _reconciliation_url("acx_" + "0" * 64),
        content=b"  ",
    )
    assert whitespace.status_code == 422, whitespace.text
    unknown_param = client.get(
        _reconciliation_url("acx_" + "0" * 64) + "?x=1"
    )
    assert unknown_param.status_code == 422, unknown_param.text
    repeated = client.get(
        _reconciliation_url("acx_" + "0" * 64) + "?x=1&x=2"
    )
    assert repeated.status_code == 422, repeated.text


def test_rebuild_digest_failure_is_503_internal_error(client, db_session, monkeypatch):
    receipt_id = _insert_receipt(db_session, EMPTY_PACKAGE_DIGEST)

    def _boom(_events):
        raise RuntimeError("digest machinery unavailable")

    monkeypatch.setattr(
        "provenance.service.canonical.audit_events_digest_hex", _boom
    )
    resp = client.get(_reconciliation_url(receipt_id))
    assert resp.status_code == 503, resp.text
    error = resp.json()["error"]
    assert error["code"] == "service_unavailable"
    assert error["details"]["reason"] == "internal_error"
    # A failed read wrote nothing and a later valid read works again.
    db_session.expire_all()
    assert _receipt_count(db_session) == 1
    assert _event_count(db_session) == 0
