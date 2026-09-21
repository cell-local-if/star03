"""Tests for the read-only audit-event checkpoint endpoint.

Covers GET /v1/audit-events/checkpoint: the response shape carries exactly
{"checkpoint_version", "digest_algorithm", "event_count",
"events_digest_hex"} with the version pinned to
"provenance-audit-checkpoint-v1" and the algorithm to "sha256"; the digest
is the SHA-256 of the canonical JSON array of the filtered events (stable
creation order, each item reduced to event_type/resource_id/UTC created_at,
keys sorted by Unicode code point, compact separators, non-ASCII
unescaped, UTF-8 encoded) and is recomputed independently here from the
audit-event search wire view; event_type/resource_id are non-empty exact
combinable filters and from/to are strict RFC 3339 UTC inclusive bounds
exactly as on the search route; limit/cursor and any other undeclared,
blank, repeated, or malformed parameter is a 422 validation_error; an
empty match set still returns the digest of the empty array; and neither
successful nor failed checkpoint reads write any resource or audit rows.
All fixtures are deterministic and offline.
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

RESPONSE_FIELDS = {
    "checkpoint_version",
    "digest_algorithm",
    "event_count",
    "events_digest_hex",
}


def _create_content(client, digest, actor_id="org-1"):
    resp = client.post(
        "/v1/contents",
        json={
            "digest_algorithm": "sha256",
            "digest_hex": digest,
            "media_type": "image/png",
            "actor_id": actor_id,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _checkpoint(client, **params):
    return client.get("/v1/audit-events/checkpoint", params=params)


def _search_items(client, **params):
    """The full unpaginated audit-event search wire view for ``params``."""
    items = []
    cursor = None
    for _ in range(100):
        query = {**params}
        if cursor is not None:
            query["cursor"] = cursor
        resp = client.get("/v1/audit-events", params=query)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        items.extend(body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            return items
    raise AssertionError("pagination did not terminate")


def _expected_digest(items):
    """Independently canonicalize the search wire items and digest them."""
    canonical = json.dumps(
        items,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


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


# --- Response shape and digest recomputation -----------------------------------


def test_response_shape_and_fixed_version_and_algorithm(client):
    _setup_events(client)
    resp = _checkpoint(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == RESPONSE_FIELDS
    assert body["checkpoint_version"] == CHECKPOINT_VERSION
    assert body["digest_algorithm"] == DIGEST_ALGORITHM
    assert body["event_count"] == 5
    assert isinstance(body["events_digest_hex"], str)
    assert len(body["events_digest_hex"]) == 64
    assert body["events_digest_hex"] == body["events_digest_hex"].lower()


def test_digest_matches_independent_canonicalization(client):
    _setup_events(client)
    items = _search_items(client)
    body = _checkpoint(client).json()
    assert body["event_count"] == len(items)
    assert body["events_digest_hex"] == _expected_digest(items)


def test_digest_is_stable_across_repeated_reads(client):
    _setup_events(client)
    first = _checkpoint(client).json()
    second = _checkpoint(client).json()
    assert first == second


def test_checkpoint_covers_every_filtered_event_in_order(client, db_session):
    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 2, tzinfo=timezone.utc)
    t3 = datetime(2026, 1, 3, tzinfo=timezone.utc)
    _insert_event(db_session, EVENT_ACTOR_CREATED, "org-1", t1)
    _insert_event(db_session, EVENT_CONTENT_CREATED, "cnt_t2", t2)
    _insert_event(db_session, EVENT_CONTENT_CREATED, "cnt_t3", t3)

    items = _search_items(client)
    assert [i["resource_id"] for i in items] == ["org-1", "cnt_t2", "cnt_t3"]
    body = _checkpoint(client).json()
    assert body["event_count"] == 3
    assert body["events_digest_hex"] == _expected_digest(items)

    # Reordering the same events changes the digest: order is committed.
    reordered = [items[1], items[0], items[2]]
    assert body["events_digest_hex"] != _expected_digest(reordered)


def test_non_ascii_values_are_not_escaped(client, db_session):
    _insert_event(
        db_session,
        EVENT_ACTOR_CREATED,
        "org-émoji-✓",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    items = _search_items(client)
    body = _checkpoint(client).json()
    assert body["events_digest_hex"] == _expected_digest(items)
    # And it differs from the ASCII-escaped canonicalization.
    escaped = json.dumps(
        items, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    assert body["events_digest_hex"] != hashlib.sha256(escaped).hexdigest()


# --- Filtering -------------------------------------------------------------------


def test_event_type_filter_matches_search(client):
    _setup_events(client)
    items = _search_items(client, event_type=EVENT_CONTENT_CREATED)
    body = _checkpoint(client, event_type=EVENT_CONTENT_CREATED).json()
    assert body["event_count"] == 3
    assert body["events_digest_hex"] == _expected_digest(items)


def test_resource_id_filter_matches_search(client):
    c1, _, _ = _setup_events(client)
    items = _search_items(client, resource_id=c1["id"])
    body = _checkpoint(client, resource_id=c1["id"]).json()
    assert body["event_count"] == 1
    assert body["events_digest_hex"] == _expected_digest(items)


def test_combined_filters_match_search(client):
    _, c2, _ = _setup_events(client)
    params = {"event_type": EVENT_CONTENT_CREATED, "resource_id": c2["id"]}
    items = _search_items(client, **params)
    body = _checkpoint(client, **params).json()
    assert body["event_count"] == 1
    assert body["events_digest_hex"] == _expected_digest(items)


def test_exact_match_is_case_and_whitespace_sensitive(client):
    _setup_events(client)
    empty_digest = _expected_digest([])
    for params in (
        {"event_type": "ACTOR.CREATED"},
        {"event_type": "actor.created "},
        {"resource_id": "ORG-1"},
        {"resource_id": "org-1 "},
    ):
        body = _checkpoint(client, **params).json()
        assert body["event_count"] == 0, params
        assert body["events_digest_hex"] == empty_digest


def test_time_bounds_match_search(client, db_session):
    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 2, 12, 30, tzinfo=timezone.utc)
    t3 = datetime(2026, 1, 3, 23, 59, 59, tzinfo=timezone.utc)
    _insert_event(db_session, EVENT_ACTOR_CREATED, "org-1", t1)
    _insert_event(db_session, EVENT_CONTENT_CREATED, "cnt_t2", t2)
    _insert_event(db_session, EVENT_CONTENT_CREATED, "cnt_t3", t3)

    params = {"from": "2026-01-02T12:30:00Z"}
    items = _search_items(client, **params)
    body = _checkpoint(client, **params).json()
    assert body["event_count"] == 2
    assert body["events_digest_hex"] == _expected_digest(items)

    params = {"to": "2026-01-02T12:30:00+00:00"}
    items = _search_items(client, **params)
    body = _checkpoint(client, **params).json()
    assert body["event_count"] == 2
    assert body["events_digest_hex"] == _expected_digest(items)

    params = {"from": "2026-01-02T12:30:00Z", "to": "2026-01-02T12:30:00.000Z"}
    items = _search_items(client, **params)
    body = _checkpoint(client, **params).json()
    assert body["event_count"] == 1
    assert body["events_digest_hex"] == _expected_digest(items)


# --- Empty results -----------------------------------------------------------------


def test_empty_database_yields_empty_array_digest(client):
    body = _checkpoint(client).json()
    assert set(body) == RESPONSE_FIELDS
    assert body["checkpoint_version"] == CHECKPOINT_VERSION
    assert body["digest_algorithm"] == DIGEST_ALGORITHM
    assert body["event_count"] == 0
    assert body["events_digest_hex"] == hashlib.sha256(b"[]").hexdigest()


def test_no_matching_events_yields_empty_array_digest(client):
    _setup_events(client)
    body = _checkpoint(client, event_type="claim.created").json()
    assert body["event_count"] == 0
    assert body["events_digest_hex"] == hashlib.sha256(b"[]").hexdigest()


# --- Parameter validation ----------------------------------------------------------


def test_limit_and_cursor_are_undeclared_validation_errors(client):
    _setup_events(client)
    for suffix in ("limit=1", "cursor=x", "limit=50&cursor=abc"):
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
            "2026-13-02T12:30:00Z",
            "2026-01-02t12:30:00z",
            "not-a-time",
        ):
            resp = _checkpoint(client, **{field: value})
            assert resp.status_code == 422, (field, value)
            assert resp.json()["error"]["code"] == "validation_error"


def test_from_later_than_to_is_validation_error(client):
    _setup_events(client)
    resp = _checkpoint(
        client,
        **{"from": "2026-01-03T00:00:00Z", "to": "2026-01-02T00:00:00Z"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only guarantee -----------------------------------------------------------


def test_reads_and_failures_write_no_rows_or_audit_events(client, db_session):
    _setup_events(client)

    def counts():
        return (
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
            db_session.scalar(select(func.count()).select_from(Content)),
        )

    before = counts()

    # Successful filtered and unfiltered checkpoint reads.
    _checkpoint(client)
    _checkpoint(client, event_type=EVENT_CONTENT_CREATED)
    _checkpoint(client, event_type="no.such.event")
    _checkpoint(client, **{"from": "2026-01-01T00:00:00Z"})

    # Failed reads must not write anything either.
    _checkpoint(client, event_type=" ")
    _checkpoint(client, limit=1)
    _checkpoint(client, cursor="tampered")
    _checkpoint(client, **{"from": "not-a-time"})
    _checkpoint(
        client,
        **{"from": "2027-01-01T00:00:00Z", "to": "2026-01-01T00:00:00Z"},
    )
    client.get("/v1/audit-events/checkpoint?event_type=a&event_type=b")
    client.get("/v1/audit-events/checkpoint?unknown=1")

    db_session.expire_all()
    assert counts() == before
