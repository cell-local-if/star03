"""Tests for the read-only impact-recon package export.

Covers GET /v1/impact-recon-package: the success body is exactly
{"checkpoint", "entries"}; ``checkpoint`` is exactly
{"checkpoint_version", "digest_algorithm", "entry_count",
"entries_digest_hex"} with the version pinned to "pir-checkpoint-v1"
and the algorithm to "sha256", and ``entries`` lists the same current
impact-import reconciliation views the global reconciliation listing
(GET /v1/impact-import-reconciliations) serves for the same effective
local_available/matches filter, in the receipts' stable creation order,
each with exactly the existing single-receipt public view
(id/checkpoint_version/impact_count/impacts_digest_hex/received_at) plus
local_available, the current unfiltered local checkpoint, and matches.
The checkpoint digest is independently recomputed from the entries
member of the same response under the checkpoint canonical rules (array
order kept, nested object keys sorted recursively by Unicode code point,
compact separators, unescaped non-ASCII, UTF-8), so both halves are
bound to one read state; the pair verifies offline via the stateless
POST /v1/impact-recon-verifications route. The local_available/matches
filters accept only lowercase true/false, are each provided at most
once, and combine as logical AND exactly as on the global reconciliation
listing; limit/cursor and any other undeclared, blank, repeated, or
malformed parameter is a 422 validation_error, as is any non-empty
request body; an empty match set returns "entries": [] together with the
deterministic digest of the empty array; and neither successful nor
failed package reads write any resource or audit rows. All fixtures are
deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance.app import create_app
from provenance.config import Settings
from provenance.models import (
    Actor,
    AttestationRevocation,
    AuditEvent,
    Claim,
    Content,
    EvidenceBundle,
    ImpactImportRecord,
)
from tests.test_impact_import_recon import (
    CHECKPOINT_KEYS,
    CHECKPOINT_VERSION,
    EMPTY_DIGEST,
    IMPORTS_URL,
    PACKAGE_URL,
    _digest_of,
    _insert_receipt,
    _matching_receipt,
    _served_checkpoint,
)
from tests.test_impact_imports import RECEIPT_KEYS, _offline_request
from tests.test_revocation_impacts import _world

PACKAGE_PATH = "/v1/impact-recon-package"
RECON_LIST_PATH = "/v1/impact-import-reconciliations"
VERIFY_PATH = "/v1/impact-recon-verifications"

PACKAGE_VERSION = "pir-checkpoint-v1"
DIGEST_ALGORITHM = "sha256"

PACKAGE_FIELDS = {"checkpoint", "entries"}
CHECKPOINT_FIELDS = {
    "checkpoint_version",
    "digest_algorithm",
    "entry_count",
    "entries_digest_hex",
}
ENTRY_FIELDS = RECEIPT_KEYS | {
    "local_available",
    "local_checkpoint",
    "matches",
}
EMPTY_ARRAY_DIGEST = hashlib.sha256(b"[]").hexdigest()


def _package(client, **params):
    return client.get(PACKAGE_PATH, params=params)


def _register(client, request: dict) -> dict:
    resp = client.post(IMPORTS_URL, json=request)
    assert resp.status_code in (200, 201), resp.text
    return resp.json()


def _import(client, label: str) -> dict:
    """Register one receipt for a fabricated offline checkpoint."""
    request = _offline_request()
    marker = f"offline-{label}"
    request["impacts"][0]["reason"] = marker
    request["checkpoint"]["impacts_digest_hex"] = _digest_of(request["impacts"])
    return _register(client, request)


def _import_many(client, count: int):
    return [_import(client, f"bulk-{i:03d}") for i in range(count)]


def _recon_list_items(client, **params):
    """The full unpaginated global reconciliation list for ``params``."""
    items = []
    cursor = None
    for _ in range(100):
        query = {**params}
        if cursor is not None:
            query["cursor"] = cursor
        resp = client.get(RECON_LIST_PATH, params=query)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        items.extend(body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            return items
    raise AssertionError("pagination did not terminate")


def _expected_digest(entries):
    """Independently canonicalize the served entries and digest them."""
    canonical = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- Response shape ---------------------------------------------------------------


def test_empty_database_yields_empty_array_and_checkpoint(client):
    resp = _package(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == PACKAGE_FIELDS
    assert list(body) == ["checkpoint", "entries"]
    assert set(body["checkpoint"]) == CHECKPOINT_FIELDS
    assert list(body["checkpoint"]) == [
        "checkpoint_version",
        "digest_algorithm",
        "entry_count",
        "entries_digest_hex",
    ]
    assert body["checkpoint"]["checkpoint_version"] == PACKAGE_VERSION
    assert body["checkpoint"]["digest_algorithm"] == DIGEST_ALGORITHM
    assert body["checkpoint"]["entry_count"] == 0
    assert body["checkpoint"]["entries_digest_hex"] == EMPTY_ARRAY_DIGEST
    assert body["entries"] == []


def test_success_body_has_exact_shape(client):
    _import_many(client, 3)
    body = _package(client).json()
    assert set(body) == PACKAGE_FIELDS
    assert set(body["checkpoint"]) == CHECKPOINT_FIELDS
    assert body["checkpoint"]["checkpoint_version"] == PACKAGE_VERSION
    assert body["checkpoint"]["digest_algorithm"] == DIGEST_ALGORITHM
    assert body["checkpoint"]["entry_count"] == 3
    assert len(body["entries"]) == 3
    # Every item carries exactly the receipt view plus the three added
    # fields, in the public-view member order.
    for entry in body["entries"]:
        assert set(entry) == ENTRY_FIELDS
        assert list(entry) == [
            "id",
            "checkpoint_version",
            "impact_count",
            "impacts_digest_hex",
            "received_at",
            "local_available",
            "local_checkpoint",
            "matches",
        ]
        assert set(entry["local_checkpoint"]) == CHECKPOINT_KEYS
        assert entry["local_checkpoint"]["checkpoint_version"] == CHECKPOINT_VERSION
        assert entry["local_checkpoint"]["digest_algorithm"] == "sha256"


def test_no_raw_material_or_internal_fields_are_echoed(client):
    _import(client, "secrets")
    rendered = _package(client).content.decode()
    # No proof bytes, public key, claim payload, ordering surrogate, raw
    # evidence field, or the imported impacts array can ever appear.
    for forbidden in (
        "signature",
        "public_key",
        "payload",
        '"seq"',
        '"evidence"',
        "content_bytes",
        '"impacts"',
    ):
        assert forbidden not in rendered


def test_response_is_compact_json_with_one_newline(client):
    _import_many(client, 2)
    raw = _package(client).content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    assert raw.startswith(b'{"checkpoint":{"checkpoint_version":"')
    expected = (
        json.dumps(
            json.loads(raw),
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    assert raw == expected


def test_non_ascii_is_emitted_unescaped(client, db_session):
    # A receipt's checkpoint_version is an arbitrary string; a non-ASCII
    # value must reach the entry (and digest) unescaped.
    _insert_receipt(db_session, "v-快照-é", 0, EMPTY_DIGEST)
    raw = _package(client).content.decode("utf-8")
    assert "v-快照-é" in raw
    assert "\\u" not in raw


# --- Consistency with the global reconciliation listing ---------------------------


def test_entries_equal_the_global_reconciliation_view(client, db_session):
    _world(client)
    current = _served_checkpoint(client)
    _import_many(client, 3)
    # One matching receipt on top of the live local impact set.
    _insert_receipt(
        db_session,
        current["checkpoint_version"],
        current["impact_count"],
        current["impacts_digest_hex"],
    )

    for params in (
        {},
        {"local_available": "true"},
        {"local_available": "false"},
        {"matches": "true"},
        {"matches": "false"},
        {"local_available": "true", "matches": "false"},
        {"local_available": "false", "matches": "true"},
    ):
        package = _package(client, **params).json()
        listed = _recon_list_items(client, **params)
        assert package["entries"] == listed, params
        assert package["checkpoint"]["entry_count"] == len(listed), params
        assert package["checkpoint"]["entries_digest_hex"] == _expected_digest(
            package["entries"]
        ), params


def test_entry_carries_the_existing_receipt_public_view(client):
    receipts = _import_many(client, 2)
    entries = {e["id"]: e for e in _package(client).json()["entries"]}
    assert set(entries) == {r["id"] for r in receipts}
    for receipt in receipts:
        # The receipt half of the entry is exactly the existing
        # single-receipt view, including its original received_at.
        assert {key: entries[receipt["id"]][key] for key in RECEIPT_KEYS} == receipt
        assert (
            {key: entries[receipt["id"]][key] for key in RECEIPT_KEYS}
            == client.get(f"{IMPORTS_URL}/{receipt['id']}").json()
        )


def test_entries_follow_stable_creation_order(client):
    receipts = _import_many(client, 5)
    entries = _package(client).json()["entries"]
    assert [e["id"] for e in entries] == [r["id"] for r in receipts]


def test_local_checkpoint_is_the_unfiltered_served_checkpoint(client, db_session):
    _world(client)
    _insert_receipt(db_session, "other-version", 99, "0" * 64)
    served = _served_checkpoint(client)
    entries = _package(client).json()["entries"]
    assert len(entries) == 1
    assert entries[0]["local_checkpoint"] == served
    # Every filter keeps the same unfiltered local checkpoint.
    for params in (
        {"matches": "true"},
        {"local_available": "false"},
        {"local_available": "true", "matches": "false"},
    ):
        for entry in _package(client, **params).json()["entries"]:
            assert entry["local_checkpoint"] == served, params


def test_local_available_and_matches_match_the_global_listing_semantics(
    client, db_session
):
    _world(client)
    matching_id, checkpoint = _matching_receipt(db_session, client)
    wrong_version = _insert_receipt(
        db_session,
        "provenance-revocation-impact-checkpoint-v2",
        checkpoint["impact_count"],
        checkpoint["impacts_digest_hex"],
    )
    wrong_state = _insert_receipt(db_session, CHECKPOINT_VERSION, 99, "a" * 64)

    body = _package(client).json()
    entries = {e["id"]: e for e in body["entries"]}
    assert set(entries) == {matching_id, wrong_version, wrong_state}
    assert entries[matching_id]["matches"] is True
    assert entries[wrong_version]["matches"] is False
    assert entries[wrong_state]["matches"] is False
    assert all(e["local_available"] is True for e in entries.values())

    assert [
        e["id"] for e in _package(client, matches="true").json()["entries"]
    ] == [matching_id]
    assert [
        e["id"] for e in _package(client, matches="false").json()["entries"]
    ] == [wrong_version, wrong_state]
    assert _package(client, local_available="false").json()["entries"] == []


def test_empty_local_set_entry_is_unavailable_but_can_match(client, db_session):
    receipt_id = _insert_receipt(db_session, CHECKPOINT_VERSION, 0, EMPTY_DIGEST)
    entries = _package(client).json()["entries"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["id"] == receipt_id
    assert entry["local_available"] is False
    assert entry["matches"] is True
    assert entry["local_checkpoint"] == {
        "checkpoint_version": CHECKPOINT_VERSION,
        "digest_algorithm": "sha256",
        "impact_count": 0,
        "impacts_digest_hex": EMPTY_DIGEST,
    }
    # It is the sole hit under the conjunction unavailable+matches.
    assert [
        e["id"]
        for e in _package(
            client, local_available="false", matches="true"
        ).json()["entries"]
    ] == [receipt_id]


# --- Digest binding ---------------------------------------------------------------


def test_checkpoint_digest_binds_to_returned_entries(client):
    _import_many(client, 3)
    body = _package(client).json()
    assert body["checkpoint"]["entries_digest_hex"] == _expected_digest(
        body["entries"]
    )

    tampered = json.loads(json.dumps(body["entries"]))
    tampered[0], tampered[1] = tampered[1], tampered[0]
    assert body["checkpoint"]["entries_digest_hex"] != _expected_digest(tampered)

    augmented = list(body["entries"])
    augmented.append(json.loads(json.dumps(body["entries"][0])))
    assert body["checkpoint"]["entries_digest_hex"] != _expected_digest(augmented)

    dropped = body["entries"][:-1]
    assert body["checkpoint"]["entries_digest_hex"] != _expected_digest(dropped)


def test_nested_object_key_order_does_not_change_digest(client):
    # The digest sorts nested object keys recursively by Unicode code
    # point (local_checkpoint included), so a verifier reproducing the
    # canonical form gets the same value though the wire view keeps its
    # public-view field order.
    _import_many(client, 2)
    body = _package(client).json()
    reordered = []
    for entry in body["entries"]:
        reordered.append({k: entry[k] for k in sorted(entry, reverse=True)})
    assert body["checkpoint"]["entries_digest_hex"] == _expected_digest(reordered)


def test_package_pair_verifies_offline(client):
    _world(client)
    _import(client, "roundtrip")
    body = _package(client, matches="false").json()
    resp = client.post(
        VERIFY_PATH,
        json={"checkpoint": body["checkpoint"], "entries": body["entries"]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_empty_pair_verifies_offline(client):
    body = _package(client, matches="true").json()
    assert body["entries"] == []
    resp = client.post(
        VERIFY_PATH,
        json={"checkpoint": body["checkpoint"], "entries": body["entries"]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


# --- Parameter and body validation ------------------------------------------------


def test_limit_and_cursor_are_undeclared_validation_errors(client):
    _import_many(client, 2)
    for suffix in ("limit=1", "cursor=x", "limit=50&cursor=abc"):
        resp = client.get(f"{PACKAGE_PATH}?{suffix}")
        _assert_validation_error(resp)


def test_undeclared_parameters_are_validation_errors(client):
    _import_many(client, 2)
    for suffix in (
        "import_id=rii_x",
        "checkpoint_version=" + PACKAGE_VERSION,
        "entries_digest_hex=" + "0" * 64,
        "entry_count=2",
        "offset=2",
        "LOCAL_AVAILABLE=true",
        "Matches=false",
    ):
        resp = client.get(f"{PACKAGE_PATH}?{suffix}")
        _assert_validation_error(resp)


def test_repeated_parameters_are_validation_errors(client):
    _import_many(client, 2)
    for suffix in (
        "local_available=true&local_available=false",
        "matches=true&matches=false",
        "matches=true&matches=true",
    ):
        resp = client.get(f"{PACKAGE_PATH}?{suffix}")
        _assert_validation_error(resp)


def test_blank_and_non_lowercase_boolean_filters_are_validation_errors(client):
    _import_many(client, 2)
    for field in ("local_available", "matches"):
        for value in ("", " ", "True", "FALSE", "1", "0", "yes", "true ", " true"):
            resp = _package(client, **{field: value})
            _assert_validation_error(resp)


def test_any_request_body_is_a_validation_error(client):
    _import_many(client, 2)
    for body_bytes in (b" ", b"\t\n", b"{}", b"not json", b"\xff\xfe"):
        resp = client.request("GET", PACKAGE_PATH, content=body_bytes)
        _assert_validation_error(resp)


def test_validation_boundaries_match_the_global_listing_route(client):
    # The package route shares the listing's local_available/matches
    # filter contract: the invalid queries they have in common must
    # render the same 422 validation_error JSON. (limit/cursor are valid
    # on the listing and rejected here, so they are covered separately.)
    _import_many(client, 2)
    cases = (
        "unknown=1",
        "local_available=true&local_available=false",
        "matches=",
        "local_available=TRUE",
        "matches=1",
    )
    for suffix in cases:
        package_resp = client.get(f"{PACKAGE_PATH}?{suffix}")
        list_resp = client.get(f"{RECON_LIST_PATH}?{suffix}")
        assert package_resp.status_code == list_resp.status_code == 422
        assert package_resp.json() == list_resp.json()


def test_non_get_methods_are_allowed_then_405(client):
    # The export is a GET; POST (despite the sibling verification route)
    # and the mutating verbs are not part of its contract.
    for method in ("post", "put", "patch", "delete"):
        resp = getattr(client, method)(PACKAGE_PATH)
        assert resp.status_code == 405, (method, resp.text)
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Read-only guarantee ----------------------------------------------------------


def test_reads_and_failures_write_no_rows_or_audit_events(client, db_session):
    _world(client)
    _import_many(client, 3)
    models = (
        Actor,
        Content,
        Claim,
        EvidenceBundle,
        AttestationRevocation,
        ImpactImportRecord,
        AuditEvent,
    )

    def counts():
        return tuple(
            db_session.scalar(select(func.count()).select_from(model))
            for model in models
        )

    before = counts()

    # Successful unfiltered, filtered, and empty reads.
    _package(client)
    _package(client, local_available="true")
    _package(client, local_available="false")
    _package(client, matches="true")
    _package(client, matches="false")
    _package(client, local_available="true", matches="false")
    _package(client, local_available="false", matches="true")

    # Failures must not write anything either.
    _package(client, limit=1)
    _package(client, cursor="tampered")
    _package(client, matches="TRUE")
    _package(client, local_available="")
    client.get(f"{PACKAGE_PATH}?matches=true&matches=false")
    client.get(f"{PACKAGE_PATH}?unknown=1")
    client.request("GET", PACKAGE_PATH, content=b"{}")

    db_session.expire_all()
    assert counts() == before


def test_package_is_deterministic_across_repeated_reads(client):
    _world(client)
    _import_many(client, 3)
    first = _package(client).content
    second = _package(client).content
    assert first == second


def test_package_is_deterministic_across_restart(file_client, tmp_db_url):
    # Direct, event-free fixture state so the package survives identically
    # across a process restart.
    session = file_client.app.state.session_factory()
    try:
        # Empty local set: one matching empty-checkpoint receipt and one
        # non-matching receipt, both event-free at fixed instants.
        _insert_receipt(session, CHECKPOINT_VERSION, 0, EMPTY_DIGEST)
        _insert_receipt(session, CHECKPOINT_VERSION, 7, "a" * 64)
    finally:
        session.close()

    expected = _package(file_client)
    assert expected.status_code == 200, expected.text
    expected_body = expected.content

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        resp = _package(client)
        assert resp.status_code == 200, resp.text
        assert resp.content == expected_body
        assert _package(client).content == expected_body
