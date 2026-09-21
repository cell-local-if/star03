"""Tests for the read-only audit-event checkpoint endpoint.

Covers GET /v1/audit-events/checkpoint: the response shape carries exactly
{"checkpoint_version", "digest_algorithm", "event_count",
"events_digest_hex"} with the fixed version "provenance-audit-checkpoint-v1"
and algorithm "sha256"; the digest is the SHA-256 of the canonical
serialization of the filtered events in stable creation order (array order
preserved, object keys sorted by Unicode code point, compact separators,
non-ASCII unescaped, UTF-8 encoded) and is reproducible offline from the
audit-event listing; event_type/resource_id are non-empty exact combinable
filters and from/to are strict RFC 3339 UTC inclusive bounds (from must not
exceed to), identical to the audit search; limit/cursor and any other
undeclared, blank, or repeated parameter is a 422 validation_error; an
empty result still returns the digest of the empty array; and neither
queries nor failures write any resource or audit rows. All fixtures are
deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from sqlalchemy import func, select

from provenance.models import (
    EVENT_ACTOR_CREATED,
    EVENT_CONTENT_CREATED,
    AuditEvent,
    Content,
)
from tests.helpers import DIGEST_A, DIGEST_B, DIGEST_C, create_actor

CHECKPOINT_VERSION = "provenance-audit-checkpoint-v1"
DIGEST_ALGORITHM = "sha256"

#: The exact checkpoint field set.
CHECKPOINT_FIELDS = {
    "checkpoint_version",
    "digest_algorithm",
    "event_count",
    "events_digest_hex",
}

#: SHA-256 of the canonical empty array ``[]``.
EMPTY_DIGEST = hashlib.sha256(b"[]").hexdigest()


def _content_payload(digest, actor_id="org-1"):
    return {
        "digest_algorithm": "sha256",
        "digest_hex": digest,
        "media_type": "image/png",
        "actor_id": actor_id,
    }


def _create_content(client, digest, actor_id="org-1"):
    resp = client.post("/v1/contents", json=_content_payload(digest, actor_id))
    assert resp.status_code == 201, resp.text
    return resp.json()


def _checkpoint(client, **params):
    return client.get("/v1/audit-events/checkpoint", params=params)


def _setup_events(client):
    """Interleaved actor/content creations -> a known audit sequence."""
    create_actor(client, actor_id="org-1")
    c1 = _create_content(client, DIGEST_A)
    create_actor(client, actor_id="org-2", name="Other", type="person")
    c2 = _create_content(client, DIGEST_B, actor_id="org-2")
    c3 = _create_content(client, DIGEST_C)
    return [c1, c2, c3]


def _insert_event(db_session, event_type, resource_id, created_at):
    """Append one audit row at a fixed instant (deterministic time tests)."""
    db_session.add(
        AuditEvent(
            event_type=event_type,
            resource_id=resource_id,
            created_at=created_at,
        )
    )
    db_session.commit()


def _canonical_digest(items):
    """Offline recomputation of the checkpoint digest from wire items."""
    canonical = json.dumps(
        items,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _listed_events(client, **params):
    """All audit-event wire items for the filters, via the search route."""
    body = client.get("/v1/audit-events", params=params).json()
    assert body["next_cursor"] is None
    return body["items"]


# --- Response shape and offline reproducibility -------------------------------


def test_response_shape_and_fixed_values(client):
    _setup_events(client)
    resp = _checkpoint(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == CHECKPOINT_FIELDS
    assert body["checkpoint_version"] == CHECKPOINT_VERSION
    assert body["digest_algorithm"] == DIGEST_ALGORITHM
    assert body["event_count"] == 5
    assert isinstance(body["events_digest_hex"], str)
    assert len(body["events_digest_hex"]) == 64
    assert body["events_digest_hex"] == body["events_digest_hex"].lower()


def test_digest_reproducible_offline_from_listing(client):
    _setup_events(client)
    items = _listed_events(client)
    body = _checkpoint(client).json()
    assert body["event_count"] == len(items)
    assert body["events_digest_hex"] == _canonical_digest(items)


def test_checkpoint_is_stable_across_repeated_reads(client):
    _setup_events(client)
    first = _checkpoint(client).json()
    second = _checkpoint(client).json()
    assert first == second


def test_canonical_form_sorts_keys_and_keeps_non_ascii(client, db_session):
    # A non-ASCII resource id proves the canonical form does not escape
    # non-ASCII characters; key order is fixed by code point, not insertion.
    _insert_event(
        db_session,
        EVENT_ACTOR_CREATED,
        "org-ünïcodé",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    items = _listed_events(client)
    assert items[0]["resource_id"] == "org-ünïcodé"
    expected_bytes = (
        '[{"created_at":"'
        + items[0]["created_at"]
        + '","event_type":"'
        + EVENT_ACTOR_CREATED
        + '","resource_id":"org-ünïcodé"}]'
    ).encode("utf-8")
    body = _checkpoint(client).json()
    assert body["event_count"] == 1
    assert body["events_digest_hex"] == hashlib.sha256(expected_bytes).hexdigest()
    assert body["events_digest_hex"] == _canonical_digest(items)


# --- Filtering (same semantics as the audit search) ----------------------------


def test_event_type_filter_matches_filtered_listing(client):
    _setup_events(client)
    params = {"event_type": EVENT_CONTENT_CREATED}
    items = _listed_events(client, **params)
    body = _checkpoint(client, **params).json()
    assert body["event_count"] == 3
    assert body["events_digest_hex"] == _canonical_digest(items)


def test_resource_id_filter_matches_filtered_listing(client):
    c1, _, _ = _setup_events(client)
    params = {"resource_id": c1["id"]}
    items = _listed_events(client, **params)
    body = _checkpoint(client, **params).json()
    assert body["event_count"] == 1
    assert body["events_digest_hex"] == _canonical_digest(items)


def test_filters_combine_as_logical_and(client):
    _, c2, _ = _setup_events(client)
    params = {"event_type": EVENT_CONTENT_CREATED, "resource_id": c2["id"]}
    items = _listed_events(client, **params)
    body = _checkpoint(client, **params).json()
    assert body["event_count"] == 1
    assert body["events_digest_hex"] == _canonical_digest(items)


def test_exact_match_is_case_and_whitespace_sensitive(client):
    _setup_events(client)
    for params in (
        {"event_type": "ACTOR.CREATED"},
        {"event_type": "actor.created "},
        {"resource_id": "ORG-1"},
        {"resource_id": "org-1 "},
    ):
        body = _checkpoint(client, **params).json()
        assert body["event_count"] == 0, params
        assert body["events_digest_hex"] == EMPTY_DIGEST


T1 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc)
T3 = datetime(2026, 1, 3, 23, 59, 59, tzinfo=timezone.utc)


def _setup_timed_events(db_session):
    _insert_event(db_session, EVENT_ACTOR_CREATED, "org-1", T1)
    _insert_event(db_session, EVENT_CONTENT_CREATED, "cnt_t2", T2)
    _insert_event(db_session, EVENT_CONTENT_CREATED, "cnt_t3", T3)


def test_time_bounds_are_inclusive_and_filter_the_digest(client, db_session):
    _setup_timed_events(db_session)
    params = {"from": "2026-01-02T12:30:00Z", "to": "2026-01-03T23:59:59Z"}
    items = _listed_events(client, **params)
    assert [i["resource_id"] for i in items] == ["cnt_t2", "cnt_t3"]
    body = _checkpoint(client, **params).json()
    assert body["event_count"] == 2
    assert body["events_digest_hex"] == _canonical_digest(items)


def test_from_equal_to_matches_exact_instant(client, db_session):
    _setup_timed_events(db_session)
    params = {"from": "2026-01-02T12:30:00Z", "to": "2026-01-02T12:30:00Z"}
    items = _listed_events(client, **params)
    body = _checkpoint(client, **params).json()
    assert body["event_count"] == 1
    assert body["events_digest_hex"] == _canonical_digest(items)


def test_utc_offset_notation_and_fractional_seconds_accepted(client, db_session):
    _setup_timed_events(db_session)
    params = {
        "from": "2026-01-02T12:30:00+00:00",
        "to": "2026-01-03T23:59:59.000Z",
    }
    items = _listed_events(client, **params)
    body = _checkpoint(client, **params).json()
    assert body["event_count"] == 2
    assert body["events_digest_hex"] == _canonical_digest(items)


# --- Empty results ---------------------------------------------------------------


def test_empty_database_returns_empty_array_digest(client):
    body = _checkpoint(client).json()
    assert set(body) == CHECKPOINT_FIELDS
    assert body["checkpoint_version"] == CHECKPOINT_VERSION
    assert body["digest_algorithm"] == DIGEST_ALGORITHM
    assert body["event_count"] == 0
    assert body["events_digest_hex"] == EMPTY_DIGEST


def test_no_matching_events_returns_empty_array_digest(client):
    _setup_events(client)
    body = _checkpoint(client, event_type="claim.created").json()
    assert body["event_count"] == 0
    assert body["events_digest_hex"] == EMPTY_DIGEST


# --- Parameter validation ----------------------------------------------------------


def test_blank_filters_are_validation_errors(client):
    _setup_events(client)
    for field in ("event_type", "resource_id"):
        for value in ("", "   ", "\t"):
            resp = _checkpoint(client, **{field: value})
            assert resp.status_code == 422, (field, value)
            assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_time_values_are_validation_errors(client):
    for field in ("from", "to"):
        for value in (
            "",
            "   ",
            "2026-01-02",
            "2026-01-02T12:30:00",
            "2026-01-02 12:30:00Z",
            "2026-01-02T12:30Z",
            "2026-01-02T12:30:00+01:00",
            "2026-01-02T12:30:00-00:01",
            "2026-13-02T12:30:00Z",
            "2026-01-02T25:30:00Z",
            "2026-01-02t12:30:00z",
            "not-a-time",
        ):
            resp = _checkpoint(client, **{field: value})
            assert resp.status_code == 422, (field, value)
            assert resp.json()["error"]["code"] == "validation_error"


def test_from_later_than_to_is_validation_error(client):
    resp = _checkpoint(
        client,
        **{"from": "2026-01-03T00:00:00Z", "to": "2026-01-02T00:00:00Z"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_validation_errors(client):
    _setup_events(client)
    for suffix in (
        "event_type=actor.created&event_type=content.created",
        "resource_id=a&resource_id=b",
        "from=2026-01-01T00:00:00Z&from=2026-01-02T00:00:00Z",
        "to=2026-01-01T00:00:00Z&to=2026-01-02T00:00:00Z",
    ):
        resp = client.get(f"/v1/audit-events/checkpoint?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_limit_and_cursor_are_not_accepted(client):
    _setup_events(client)
    for suffix in (
        "limit=1",
        "limit=50",
        "cursor=abc",
        "limit=1&cursor=abc",
    ):
        resp = client.get(f"/v1/audit-events/checkpoint?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameters_are_validation_errors(client):
    _setup_events(client)
    for suffix in (
        "actor_id=org-1",
        "event_types=actor.created",
        "offset=2",
        "FROM=2026-01-01T00:00:00Z",
    ):
        resp = client.get(f"/v1/audit-events/checkpoint?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only guarantee ----------------------------------------------------------


def test_queries_and_failures_write_no_rows_or_audit_events(client, db_session):
    _setup_events(client)

    def counts():
        return (
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
            db_session.scalar(select(func.count()).select_from(Content)),
        )

    before = counts()

    # Successful unfiltered, filtered, and empty-result reads.
    resp = _checkpoint(client)
    assert resp.status_code == 200, resp.text
    _checkpoint(client, event_type=EVENT_CONTENT_CREATED)
    _checkpoint(client, event_type="no.such.event")
    _checkpoint(client, **{"from": "2026-01-01T00:00:00Z"})

    # Failed reads must not write anything either.
    _checkpoint(client, event_type=" ")
    _checkpoint(client, **{"from": "not-a-time"})
    _checkpoint(client, **{"from": "2027-01-01T00:00:00Z",
                           "to": "2026-01-01T00:00:00Z"})
    client.get("/v1/audit-events/checkpoint?limit=1")
    client.get("/v1/audit-events/checkpoint?cursor=x")
    client.get("/v1/audit-events/checkpoint?event_type=a&event_type=b")
    client.get("/v1/audit-events/checkpoint?unknown=1")

    db_session.expire_all()
    assert counts() == before
