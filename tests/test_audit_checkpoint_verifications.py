"""Tests for the stateless audit-checkpoint verification endpoint.

Covers POST /v1/audit-events/checkpoint-verifications: the request body is
exactly {"checkpoint", "events"}; the checkpoint is exactly
{"checkpoint_version", "digest_algorithm", "event_count",
"events_digest_hex"} with the version pinned to
"provenance-audit-checkpoint-v1", the algorithm to "sha256", the count a
non-negative integer, and the digest 64 lowercase hex characters. The
events array must contain exactly ``event_count`` elements, each exactly
{"event_type", "resource_id", "created_at"} with non-empty strings and a
strict RFC 3339 UTC timestamp. The digest is the SHA-256 of the canonical
event array (array order preserved, object keys sorted by Unicode code
point, compact separators, non-ASCII unescaped, UTF-8 encoded) computed
over the events exactly as received. Any structural, field, or count
violation is a 422 validation_error; a structurally valid request whose
digest differs is 200 {"valid": false, "computed_digest_hex": ...}; a
match is exactly {"valid": true}. The route is fully stateless: it needs
no persisted events and creates, modifies, and queries nothing. All
fixtures are deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json
import re

from sqlalchemy import func, select

from provenance.models import (
    Actor,
    Attestation,
    AttestationRevocation,
    AuditEvent,
    Claim,
    Content,
    EvidenceBundle,
)
from tests.helpers import DIGEST_A, create_actor

URL = "/v1/audit-events/checkpoint-verifications"
CHECKPOINT_VERSION = "provenance-audit-checkpoint-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


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
    """A verification request built from the service's own audit views."""
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


# --- Verdicts -----------------------------------------------------------------


def test_verification_of_served_checkpoint_is_valid(client):
    _setup_events(client)
    resp = client.post(URL, json=_served_request(client))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_digest_mismatch_returns_computed_digest(client):
    request = _offline_request()
    events = request["events"]

    # Flip the last hex character of the claimed digest.
    claimed = request["checkpoint"]["events_digest_hex"]
    request["checkpoint"]["events_digest_hex"] = claimed[:-1] + (
        "0" if claimed[-1] != "0" else "1"
    )

    resp = client.post(URL, json=request)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"valid", "computed_digest_hex"}
    assert body["valid"] is False
    assert body["computed_digest_hex"] == _digest_of(events)
    assert body["computed_digest_hex"] == claimed


def test_verification_tampered_event_is_invalid(client):
    request = _offline_request()
    # The checkpoint digest still commits to the untouched sequence.
    request["events"][0]["resource_id"] = "org-forged"

    resp = client.post(URL, json=request)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert body["computed_digest_hex"] == _digest_of(request["events"])
    assert body["computed_digest_hex"] != request["checkpoint"]["events_digest_hex"]


def test_verification_event_order_participates(client):
    request = _offline_request()

    # The array keeps its order: the same events in another order are a
    # different sequence and no longer match the checkpoint.
    reordered = json.loads(json.dumps(request))
    reordered["events"] = list(reversed(reordered["events"]))
    resp = client.post(URL, json=reordered)
    assert resp.status_code == 200, resp.text
    assert resp.json()["valid"] is False
    assert resp.json()["computed_digest_hex"] == _digest_of(reordered["events"])


def test_verification_uses_unescaped_non_ascii_utf8(client):
    request = _offline_request()
    assert any(
        ord(ch) > 127 for ch in json.dumps(request["events"], ensure_ascii=False)
    )
    resp = client.post(URL, json=request)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}

    # A digest computed over ASCII-escaped bytes would not match.
    escaped = json.dumps(
        request["events"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    assert hashlib.sha256(escaped).hexdigest() != _digest_of(request["events"])


def test_verification_empty_sequence_matches_empty_checkpoint(client):
    request = _offline_request(events=[])
    assert request["checkpoint"]["events_digest_hex"] == hashlib.sha256(
        b"[]"
    ).hexdigest()
    resp = client.post(URL, json=request)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_is_deterministic_across_repeated_requests(client):
    request = _offline_request()
    first = client.post(URL, json=request)
    second = client.post(URL, json=request)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json() == {"valid": True}


def test_verification_recomputed_digest_matches_served_checkpoint(client):
    # End-to-end agreement: the digest this endpoint computes for the served
    # event listing equals the digest the checkpoint endpoint issued for it.
    _setup_events(client)
    request = _served_request(client)
    request["checkpoint"]["events_digest_hex"] = "0" * 64

    resp = client.post(URL, json=request)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    assert _HEX64.fullmatch(body["computed_digest_hex"])
    assert body["computed_digest_hex"] == client.get(
        "/v1/audit-events/checkpoint"
    ).json()["events_digest_hex"]


# --- Statelessness -------------------------------------------------------------


def test_verification_needs_no_persisted_events(client):
    # The database is empty and every identifier is unknown to the service:
    # verification is decided by the request body alone.
    resp = client.post(URL, json=_offline_request())
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_writes_no_resources_or_audit_events(client, db_session):
    _setup_events(client)
    valid_request = _served_request(client)
    invalid_request = json.loads(json.dumps(valid_request))
    invalid_request["events"][0]["resource_id"] = "tampered"
    malformed_request = _offline_request()
    del malformed_request["checkpoint"]["event_count"]

    models = (
        Actor,
        Content,
        Claim,
        EvidenceBundle,
        Attestation,
        AttestationRevocation,
        AuditEvent,
    )

    def _counts():
        return {
            model: db_session.execute(
                select(func.count()).select_from(model)
            ).scalar_one()
            for model in models
        }

    before = _counts()

    assert client.post(URL, json=valid_request).status_code == 200
    assert client.post(URL, json=invalid_request).status_code == 200
    _assert_validation_error(client.post(URL, json=malformed_request))
    # A wholly offline request against unknown ids also writes nothing.
    assert client.post(URL, json=_offline_request()).status_code == 200

    db_session.expire_all()
    assert _counts() == before


# --- Top-level structure ---------------------------------------------------------


def test_verification_requires_exactly_the_two_top_level_fields(client):
    request = _offline_request()

    for field in ("checkpoint", "events"):
        incomplete = {k: v for k, v in request.items() if k != field}
        _assert_validation_error(client.post(URL, json=incomplete))

    augmented = dict(request)
    augmented["computed_digest_hex"] = request["checkpoint"]["events_digest_hex"]
    _assert_validation_error(client.post(URL, json=augmented))


def test_verification_rejects_non_object_and_empty_bodies(client):
    for body in (None, [], "checkpoint", 42):
        _assert_validation_error(client.post(URL, json=body))
    _assert_validation_error(client.post(URL, json={}))


def test_verification_members_have_correct_types(client):
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


# --- Checkpoint fields ------------------------------------------------------------


def test_verification_checkpoint_has_exactly_the_four_members(client):
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


def test_verification_requires_fixed_checkpoint_version(client):
    request = _offline_request()
    for version in (
        "provenance-audit-checkpoint-v2",
        "Provenance-Audit-Checkpoint-V1",
        "",
    ):
        bad = json.loads(json.dumps(request))
        bad["checkpoint"]["checkpoint_version"] = version
        _assert_validation_error(client.post(URL, json=bad))


def test_verification_requires_sha256_digest_algorithm(client):
    request = _offline_request()
    for algorithm in ("sha512", "SHA256", "sha-256", ""):
        bad = json.loads(json.dumps(request))
        bad["checkpoint"]["digest_algorithm"] = algorithm
        _assert_validation_error(client.post(URL, json=bad))


def test_verification_requires_64_lowercase_hex_events_digest(client):
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


def test_verification_requires_non_negative_integer_event_count(client):
    request = _offline_request()
    for count in (-1, 2.0, 2.5, "2", "two", True, None, [2]):
        bad = json.loads(json.dumps(request))
        bad["checkpoint"]["event_count"] = count
        _assert_validation_error(client.post(URL, json=bad))


def test_verification_events_length_must_equal_event_count(client):
    request = _offline_request()

    too_small = json.loads(json.dumps(request))
    too_small["checkpoint"]["event_count"] = len(request["events"]) - 1
    _assert_validation_error(client.post(URL, json=too_small))

    too_large = json.loads(json.dumps(request))
    too_large["checkpoint"]["event_count"] = len(request["events"]) + 1
    _assert_validation_error(client.post(URL, json=too_large))


# --- Event fields ------------------------------------------------------------------


def test_verification_events_have_exactly_the_three_fields(client):
    request = _offline_request()

    for field in ("event_type", "resource_id", "created_at"):
        incomplete = json.loads(json.dumps(request))
        del incomplete["events"][0][field]
        _assert_validation_error(client.post(URL, json=incomplete))

    augmented = json.loads(json.dumps(request))
    augmented["events"][0]["id"] = "evt_1"
    _assert_validation_error(client.post(URL, json=augmented))


def test_verification_event_items_must_be_objects(client):
    request = _offline_request()
    for wrong in (None, "event", 42, ["actor.created", "org-1"]):
        bad = json.loads(json.dumps(request))
        bad["events"][0] = wrong
        _assert_validation_error(client.post(URL, json=bad))


def test_verification_event_type_and_resource_id_must_be_non_empty(client):
    request = _offline_request()
    for field in ("event_type", "resource_id"):
        for value in ("", "   ", "\t", None, 42):
            bad = json.loads(json.dumps(request))
            bad["events"][0][field] = value
            _assert_validation_error(client.post(URL, json=bad))


def test_verification_created_at_must_be_strict_rfc3339_utc(client):
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


def test_verification_created_at_equivalent_utc_spellings_accepted(client):
    # Both explicit UTC designators are strict RFC 3339 UTC; each spelling
    # participates in the digest exactly as received.
    for spelling in ("2026-01-02T03:04:05Z", "2026-01-02T03:04:05+00:00"):
        events = _offline_events()
        events[0]["created_at"] = spelling
        request = _offline_request(events)
        resp = client.post(URL, json=request)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"valid": True}

    # The spellings are distinct byte sequences: a digest computed over one
    # does not verify a sequence spelled the other way.
    events_z = _offline_events()
    events_offset = _offline_events()
    events_offset[0]["created_at"] = "2026-01-02T03:04:05+00:00"
    assert _digest_of(events_z) != _digest_of(events_offset)
