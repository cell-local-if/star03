"""Tests for the stateless audit-checkpoint verification endpoint.

Covers POST /v1/audit-events/checkpoint-verifications: the request body is
exactly {"checkpoint", "events"}; ``checkpoint`` is exactly
{"checkpoint_version", "digest_algorithm", "event_count",
"events_digest_hex"} with the version pinned to
"provenance-audit-checkpoint-v1", the algorithm to "sha256", the digest to
64 lowercase hexadecimal characters, and the count a non-negative integer;
``events`` is an array whose length equals the count, each item carrying
exactly a non-empty ``event_type``, a non-empty ``resource_id``, and a
strict RFC 3339 UTC ``created_at``. The digest is the SHA-256 of the
canonical JSON array of the events exactly as received (array order kept,
object keys sorted by Unicode code point, compact separators, non-ASCII
unescaped, UTF-8 encoded). Any structural or field violation is a 422
validation_error; a structurally valid request whose computed digest
differs is 200 {"valid": false, "computed_digest_hex": ...}; a match is
exactly {"valid": true}. The route is fully stateless: events need not
exist locally, and the route queries, creates, and modifies nothing.
All fixtures are deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json

from sqlalchemy import func, select

from provenance.models import AuditEvent, Content

URL = "/v1/audit-events/checkpoint-verifications"
CHECKPOINT_VERSION = "provenance-audit-checkpoint-v1"
DIGEST_ALGORITHM = "sha256"


def _digest_of(events) -> str:
    """Independently canonicalize the events array and digest it."""
    canonical = json.dumps(
        events,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _event(
    event_type="content.created",
    resource_id="cnt_1",
    created_at="2026-01-02T03:04:05Z",
):
    return {
        "event_type": event_type,
        "resource_id": resource_id,
        "created_at": created_at,
    }


def _checkpoint(events, **overrides):
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "digest_algorithm": DIGEST_ALGORITHM,
        "event_count": len(events),
        "events_digest_hex": _digest_of(events),
    }
    checkpoint.update(overrides)
    return checkpoint


def _body(events=None, checkpoint=None):
    if events is None:
        events = [_event()]
    if checkpoint is None:
        checkpoint = _checkpoint(events)
    return {"checkpoint": checkpoint, "events": events}


def _verify(client, body):
    return client.post(URL, json=body)


def _assert_validation_error(resp):
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- Verdicts ----------------------------------------------------------------


def test_matching_checkpoint_returns_exactly_valid_true(client):
    events = [
        _event(),
        _event("actor.created", "org-1", "2026-01-02T03:04:06Z"),
    ]
    resp = _verify(client, _body(events))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_mismatched_digest_returns_valid_false_with_computed(client):
    events = [_event()]
    checkpoint = _checkpoint(events, events_digest_hex=_digest_of([]))
    resp = _verify(client, _body(events, checkpoint))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "valid": False,
        "computed_digest_hex": _digest_of(events),
    }


def test_empty_sequence_matches_empty_array_digest(client):
    resp = _verify(client, _body(events=[]))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}

    # A nonzero digest of the empty array cannot match a nonempty claim.
    checkpoint = _checkpoint([], events_digest_hex=_digest_of([_event()]))
    resp = _verify(client, _body([], checkpoint))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "valid": False,
        "computed_digest_hex": hashlib.sha256(b"[]").hexdigest(),
    }


def test_array_order_is_committed(client):
    events = [
        _event("actor.created", "org-1", "2026-01-02T03:04:05Z"),
        _event("content.created", "cnt_1", "2026-01-02T03:04:06Z"),
    ]
    # The digest of the reversed sequence does not verify the given order.
    checkpoint = _checkpoint(list(reversed(events)))
    resp = _verify(client, _body(events, checkpoint))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "valid": False,
        "computed_digest_hex": _digest_of(events),
    }


def test_created_at_spelling_is_committed(client):
    # Equivalent instants spelled differently digest differently: the digest
    # commits to the received spelling, not a normalized form.
    z_spelling = [_event(created_at="2026-01-02T03:04:05Z")]
    offset_spelling = [_event(created_at="2026-01-02T03:04:05+00:00")]
    assert _digest_of(z_spelling) != _digest_of(offset_spelling)

    resp = _verify(client, _body(offset_spelling))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}

    checkpoint = _checkpoint(z_spelling)
    resp = _verify(client, _body(offset_spelling, checkpoint))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "valid": False,
        "computed_digest_hex": _digest_of(offset_spelling),
    }


def test_non_ascii_values_are_not_escaped(client):
    events = [_event("actor.created", "org-émoji-✓")]
    resp = _verify(client, _body(events))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}

    # The ASCII-escaped canonicalization yields a different digest.
    escaped = json.dumps(
        events, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    escaped_digest = hashlib.sha256(escaped).hexdigest()
    assert escaped_digest != _digest_of(events)
    checkpoint = _checkpoint(events, events_digest_hex=escaped_digest)
    resp = _verify(client, _body(events, checkpoint))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": False, "computed_digest_hex": _digest_of(events)}


# --- Statelessness -------------------------------------------------------------


def test_events_need_not_exist_locally(client, db_session):
    # Fabricated events referencing nothing stored still verify.
    events = [
        _event("content.created", "cnt_does_not_exist"),
        _event("actor.created", "org-also-absent", "2026-01-03T00:00:00Z"),
    ]
    resp = _verify(client, _body(events))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}
    assert (
        db_session.scalar(select(func.count()).select_from(AuditEvent)) == 0
    )


def test_verifications_write_no_rows_or_audit_events(client, db_session):
    def counts():
        return (
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
            db_session.scalar(select(func.count()).select_from(Content)),
        )

    before = counts()

    events = [_event()]
    # A matching verification.
    _verify(client, _body(events))
    # A digest mismatch.
    _verify(client, _body(events, _checkpoint(events, event_count=1,
                                                 events_digest_hex=_digest_of([]))))
    # Validation failures.
    _verify(client, {"checkpoint": _checkpoint(events)})
    _verify(client, _body(events, _checkpoint(events, event_count=2)))
    _verify(client, _body([_event(created_at="not-a-time")]))

    db_session.expire_all()
    assert counts() == before


# --- Request structure ---------------------------------------------------------


def test_top_level_members_are_exactly_checkpoint_and_events(client):
    events = [_event()]
    good = _body(events)

    for key in ("checkpoint", "events"):
        incomplete = {k: v for k, v in good.items() if k != key}
        _assert_validation_error(_verify(client, incomplete))

    for extra in ("snapshot", "digest", "events_digest_hex"):
        _assert_validation_error(_verify(client, {**good, extra: None}))

    _assert_validation_error(_verify(client, []))
    _assert_validation_error(_verify(client, {"checkpoint": None, "events": events}))
    _assert_validation_error(
        _verify(client, {"checkpoint": _checkpoint(events), "events": {}})
    )


def test_checkpoint_members_are_exactly_the_four_fields(client):
    events = [_event()]
    good = _checkpoint(events)

    for key in (
        "checkpoint_version",
        "digest_algorithm",
        "event_count",
        "events_digest_hex",
    ):
        incomplete = {k: v for k, v in good.items() if k != key}
        _assert_validation_error(_verify(client, _body(events, incomplete)))

    for extra in ("manifest_version", "limit", "cursor"):
        _assert_validation_error(
            _verify(client, _body(events, {**good, extra: None}))
        )

    _assert_validation_error(_verify(client, _body(events, "not-an-object")))


def test_event_members_are_exactly_the_three_fields(client):
    good = _event()

    for key in ("event_type", "resource_id", "created_at"):
        incomplete = {k: v for k, v in good.items() if k != key}
        _assert_validation_error(_verify(client, _body([incomplete])))

    for extra in ("id", "actor_id", "payload"):
        _assert_validation_error(_verify(client, _body([{**good, extra: None}])))

    _assert_validation_error(_verify(client, _body(["not-an-object"])))
    _assert_validation_error(_verify(client, _body([None])))


# --- Field validation ----------------------------------------------------------


def test_fixed_version_and_algorithm_are_enforced(client):
    events = [_event()]
    for field, values in (
        (
            "checkpoint_version",
            [
                "provenance-audit-checkpoint-v2",
                "provenance-exchange-manifest-v1",
                "",
                None,
            ],
        ),
        ("digest_algorithm", ["sha512", "SHA256", "", None]),
    ):
        for value in values:
            checkpoint = _checkpoint(events, **{field: value})
            _assert_validation_error(_verify(client, _body(events, checkpoint)))


def test_events_digest_hex_must_be_64_lowercase_hex(client):
    events = [_event()]
    valid = _digest_of(events)
    for value in (
        valid.upper(),
        valid[:-1],
        valid + "0",
        "g" * 64,
        "",
        " " * 64,
        None,
        123,
    ):
        checkpoint = _checkpoint(events, events_digest_hex=value)
        _assert_validation_error(_verify(client, _body(events, checkpoint)))


def test_event_count_must_be_a_non_negative_integer(client):
    events = [_event()]
    for value in (-1, -100, 1.5, 1.0, "1", "1.0", True, None, [1]):
        checkpoint = _checkpoint(events, event_count=value)
        _assert_validation_error(_verify(client, _body(events, checkpoint)))


def test_event_count_must_equal_events_length(client):
    events = [_event(), _event("actor.created", "org-1")]
    for count in (0, 1, 3, 100):
        checkpoint = _checkpoint(events, event_count=count)
        _assert_validation_error(_verify(client, _body(events, checkpoint)))


def test_event_type_and_resource_id_must_be_non_empty(client):
    for field in ("event_type", "resource_id"):
        for value in ("", "   ", "\t", None, 7):
            _assert_validation_error(_verify(client, _body([_event(**{field: value})])))


def test_created_at_must_be_strict_rfc3339_utc(client):
    for value in (
        "",
        "   ",
        "2026-01-02",
        "2026-01-02T03:04:05",
        "2026-01-02 03:04:05Z",
        "2026-01-02T03:04Z",
        "2026-01-02T03:04:05+01:00",
        "2026-01-02T03:04:05-00:00",
        "2026-13-02T03:04:05Z",
        "2026-01-02t03:04:05z",
        "not-a-time",
        None,
        0,
    ):
        _assert_validation_error(
            _verify(client, _body([_event(created_at=value)]))
        )


def test_created_at_accepts_z_offset_and_fractional_seconds(client):
    events = [
        _event(created_at="2026-01-02T03:04:05Z"),
        _event(created_at="2026-01-02T03:04:05+00:00"),
        _event(created_at="2026-01-02T03:04:05.123456Z"),
    ]
    resp = _verify(client, _body(events))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_malformed_json_is_a_validation_error(client):
    resp = client.post(
        URL, content=b"{not json", headers={"Content-Type": "application/json"}
    )
    _assert_validation_error(resp)
