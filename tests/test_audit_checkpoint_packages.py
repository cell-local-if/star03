"""Tests for the read-only audit-event checkpoint package export.

Covers GET /v1/audit-events/checkpoint/package: the success body is exactly
{"checkpoint", "events"}; ``checkpoint`` is byte-for-byte the body of the
existing GET /v1/audit-events/checkpoint route under the same filters (four
fields, version "provenance-audit-checkpoint-v1", algorithm "sha256");
``events`` is the filtered audit-event sequence in stable creation order,
each item carrying exactly event_type/resource_id/UTC created_at, equal to
the unpaginated GET /v1/audit-events view under the same filters; the
checkpoint digest is independently recomputed from the ``events`` member of
this same response under the existing canonical rules (array order kept,
object keys sorted by Unicode code point, compact separators, non-ASCII
unescaped, UTF-8, SHA-256) and verifies offline via the stateless
checkpoint-verification route, so both halves are bound to one read state;
only event_type/resource_id/from/to are accepted (exact-match and strict RFC
3339 UTC semantics identical to the existing checkpoint route) and
limit/cursor plus any other undeclared, blank, or repeated parameter is a
422 validation_error; an empty match set returns an empty array and a valid
empty-array checkpoint; and successful, empty, and failed reads write no
resource or audit rows. All fixtures are deterministic and offline.
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
from provenance.time_utils import parse_rfc3339_utc
from tests.helpers import DIGEST_A, DIGEST_B, DIGEST_C, create_actor

CHECKPOINT_VERSION = "provenance-audit-checkpoint-v1"
DIGEST_ALGORITHM = "sha256"

PACKAGE_FIELDS = {"checkpoint", "events"}
CHECKPOINT_FIELDS = {
    "checkpoint_version",
    "digest_algorithm",
    "event_count",
    "events_digest_hex",
}
EVENT_FIELDS = {"event_type", "resource_id", "created_at"}


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


def _package(client, **params):
    return client.get("/v1/audit-events/checkpoint/package", params=params)


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
    """Independently canonicalize the events wire items and digest them."""
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


# --- Response shape ---------------------------------------------------------------


def test_success_body_has_exactly_checkpoint_and_events(client):
    _setup_events(client)
    resp = _package(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == PACKAGE_FIELDS
    assert set(body["checkpoint"]) == CHECKPOINT_FIELDS
    assert isinstance(body["events"], list)
    assert body["checkpoint"]["event_count"] == len(body["events"])


def test_every_event_item_has_exactly_the_public_fields(client):
    _setup_events(client)
    for item in _package(client).json()["events"]:
        assert set(item) == EVENT_FIELDS
        assert isinstance(item["event_type"], str)
        assert isinstance(item["resource_id"], str)
        assert isinstance(item["created_at"], str)


# --- Consistency with the existing routes -----------------------------------------


def test_checkpoint_equals_existing_checkpoint_route_for_each_filter(client):
    _setup_events(client)
    filter_sets = [
        {},
        {"event_type": EVENT_CONTENT_CREATED},
        {"event_type": EVENT_ACTOR_CREATED, "resource_id": "org-1"},
        {"resource_id": "org-2"},
        {"from": "2026-01-01T00:00:00Z", "to": "2030-01-01T00:00:00Z"},
        {"event_type": "no.such.event"},
    ]
    for params in filter_sets:
        package = _package(client, **params).json()
        checkpoint = _checkpoint(client, **params).json()
        assert package["checkpoint"] == checkpoint, params
        assert package["checkpoint"]["checkpoint_version"] == CHECKPOINT_VERSION
        assert package["checkpoint"]["digest_algorithm"] == DIGEST_ALGORITHM


def test_events_equal_unpaginated_search_view_under_same_filters(client):
    c1, c2, _ = _setup_events(client)
    filter_sets = [
        {},
        {"event_type": EVENT_CONTENT_CREATED},
        {"event_type": EVENT_CONTENT_CREATED, "resource_id": c2["id"]},
        {"resource_id": c1["id"]},
        {"resource_id": "missing-resource"},
    ]
    for params in filter_sets:
        package = _package(client, **params).json()
        search = _search_items(client, **params)
        assert package["events"] == search, params
        assert package["checkpoint"]["event_count"] == len(search)


def test_events_follow_stable_creation_order(client, db_session):
    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 2, tzinfo=timezone.utc)
    t3 = datetime(2026, 1, 3, tzinfo=timezone.utc)
    _insert_event(db_session, EVENT_ACTOR_CREATED, "org-a", t1)
    _insert_event(db_session, EVENT_CONTENT_CREATED, "cnt-b", t2)
    _insert_event(db_session, EVENT_CONTENT_CREATED, "cnt-c", t3)

    events = _package(client).json()["events"]
    assert [(e["event_type"], e["resource_id"]) for e in events] == [
        (EVENT_ACTOR_CREATED, "org-a"),
        (EVENT_CONTENT_CREATED, "cnt-b"),
        (EVENT_CONTENT_CREATED, "cnt-c"),
    ]
    # Timestamps render as strict RFC 3339 UTC at the exact inserted instants;
    # the UTC designator spelling ("Z" vs "+00:00") is storage-dependent.
    assert [parse_rfc3339_utc(e["created_at"]) for e in events] == [t1, t2, t3]


# --- Digest binding to the same response's events ---------------------------------


def test_checkpoint_digest_is_independently_recomputed_from_response_events(client):
    _setup_events(client)
    body = _package(client).json()
    assert body["checkpoint"]["events_digest_hex"] == _expected_digest(
        body["events"]
    )


def test_checkpoint_digest_binds_under_every_filter(client, db_session):
    _setup_events(client)
    t = datetime(2026, 2, 1, tzinfo=timezone.utc)
    _insert_event(db_session, EVENT_ACTOR_CREATED, "org-later", t)
    for params in (
        {},
        {"event_type": EVENT_ACTOR_CREATED},
        {"resource_id": "org-later"},
        {"from": "2026-01-15T00:00:00Z"},
        {"to": "2025-12-31T00:00:00Z"},
        {
            "event_type": EVENT_CONTENT_CREATED,
            "from": "2026-01-01T00:00:00Z",
            "to": "2026-01-31T00:00:00Z",
        },
    ):
        body = _package(client, **params).json()
        assert (
            body["checkpoint"]["events_digest_hex"]
            == _expected_digest(body["events"])
        ), params
        assert body["checkpoint"]["event_count"] == len(body["events"])


def test_digest_commits_to_array_order(client, db_session):
    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 2, tzinfo=timezone.utc)
    _insert_event(db_session, EVENT_ACTOR_CREATED, "org-a", t1)
    _insert_event(db_session, EVENT_CONTENT_CREATED, "cnt-b", t2)

    body = _package(client).json()
    assert body["checkpoint"]["events_digest_hex"] == _expected_digest(
        body["events"]
    )
    reordered = [body["events"][1], body["events"][0]]
    assert body["checkpoint"]["events_digest_hex"] != _expected_digest(reordered)


def test_tampering_with_returned_events_breaks_binding(client):
    _setup_events(client)
    body = _package(client).json()

    dropped = list(body["events"])
    dropped.pop()
    assert body["checkpoint"]["events_digest_hex"] != _expected_digest(dropped)
    assert body["checkpoint"]["event_count"] != len(dropped)

    tampered = [dict(body["events"][0])]
    tampered[0]["resource_id"] = "different-resource"
    assert body["checkpoint"]["events_digest_hex"] != _expected_digest(tampered)


def test_non_ascii_values_are_not_escaped_in_binding(client, db_session):
    _insert_event(
        db_session,
        EVENT_ACTOR_CREATED,
        "org-émoji-✓",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    body = _package(client).json()
    assert body["checkpoint"]["events_digest_hex"] == _expected_digest(
        body["events"]
    )
    escaped = json.dumps(
        body["events"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    assert body["checkpoint"]["events_digest_hex"] != hashlib.sha256(
        escaped
    ).hexdigest()


def test_package_verifies_offline_via_stateless_verification_route(client):
    _setup_events(client)
    for params in (
        {},
        {"event_type": EVENT_CONTENT_CREATED},
        {"event_type": "no.such.event"},
    ):
        body = _package(client, **params).json()
        resp = client.post(
            "/v1/audit-events/checkpoint-verifications",
            json={"checkpoint": body["checkpoint"], "events": body["events"]},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"valid": True}, params


# --- Filtering semantics (determined offline against the existing routes) ---------


def test_exact_match_is_case_and_whitespace_sensitive(client):
    _setup_events(client)
    for params in (
        {"event_type": "ACTOR.CREATED"},
        {"event_type": "actor.created "},
        {"resource_id": "ORG-1"},
        {"resource_id": "org-1 "},
    ):
        package = _package(client, **params).json()
        assert package["events"] == [], params
        assert package["checkpoint"]["event_count"] == 0


def test_time_bounds_are_strict_rfc3339_utc_and_inclusive(client, db_session):
    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 2, 12, 30, tzinfo=timezone.utc)
    t3 = datetime(2026, 1, 3, 23, 59, 59, tzinfo=timezone.utc)
    _insert_event(db_session, EVENT_ACTOR_CREATED, "org-1", t1)
    _insert_event(db_session, EVENT_CONTENT_CREATED, "cnt_t2", t2)
    _insert_event(db_session, EVENT_CONTENT_CREATED, "cnt_t3", t3)

    # "Z" and "+00:00" spellings of the same instant are equivalent.
    params = {"from": "2026-01-02T12:30:00Z"}
    body = _package(client, **params).json()
    assert [e["resource_id"] for e in body["events"]] == ["cnt_t2", "cnt_t3"]
    assert body["events"] == _search_items(client, **params)

    params = {"to": "2026-01-02T12:30:00+00:00"}
    body = _package(client, **params).json()
    assert [e["resource_id"] for e in body["events"]] == ["org-1", "cnt_t2"]
    assert body["events"] == _search_items(client, **params)

    # Bounds are inclusive on both ends.
    params = {"from": "2026-01-02T12:30:00Z", "to": "2026-01-02T12:30:00.000Z"}
    body = _package(client, **params).json()
    assert [e["resource_id"] for e in body["events"]] == ["cnt_t2"]
    assert body["checkpoint"]["events_digest_hex"] == _expected_digest(
        body["events"]
    )


# --- Empty results ----------------------------------------------------------------


def test_empty_database_yields_empty_array_and_empty_checkpoint(client):
    body = _package(client).json()
    assert set(body) == PACKAGE_FIELDS
    assert body["events"] == []
    checkpoint = body["checkpoint"]
    assert checkpoint["checkpoint_version"] == CHECKPOINT_VERSION
    assert checkpoint["digest_algorithm"] == DIGEST_ALGORITHM
    assert checkpoint["event_count"] == 0
    assert checkpoint["events_digest_hex"] == hashlib.sha256(b"[]").hexdigest()


def test_no_matching_events_yields_empty_array_and_empty_checkpoint(client):
    _setup_events(client)
    body = _package(client, event_type="claim.created").json()
    assert body["events"] == []
    checkpoint = body["checkpoint"]
    assert checkpoint["event_count"] == 0
    assert checkpoint["events_digest_hex"] == hashlib.sha256(b"[]").hexdigest()


def test_empty_package_verifies_offline(client):
    body = _package(client, event_type="nothing.matches").json()
    assert body["events"] == []
    resp = client.post(
        "/v1/audit-events/checkpoint-verifications",
        json={"checkpoint": body["checkpoint"], "events": body["events"]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


# --- Parameter boundaries ---------------------------------------------------------


def test_limit_and_cursor_are_not_accepted(client):
    _setup_events(client)
    for suffix in (
        "limit=1",
        "cursor=x",
        "limit=50&cursor=abc",
        "event_type=actor.created&limit=1",
    ):
        resp = client.get(f"/v1/audit-events/checkpoint/package?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameters_are_validation_errors(client):
    _setup_events(client)
    for suffix in (
        "actor_id=org-1",
        "event_types=actor.created",
        "offset=2",
        "FROM=2026-01-01T00:00:00Z",
        "checkpoint_version=x",
    ):
        resp = client.get(f"/v1/audit-events/checkpoint/package?{suffix}")
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
        resp = client.get(f"/v1/audit-events/checkpoint/package?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_blank_filters_are_validation_errors(client):
    _setup_events(client)
    for field in ("event_type", "resource_id"):
        for value in ("", "   ", "\t"):
            resp = _package(client, **{field: value})
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
            resp = _package(client, **{field: value})
            assert resp.status_code == 422, (field, value)
            assert resp.json()["error"]["code"] == "validation_error"


def test_from_later_than_to_is_validation_error(client):
    _setup_events(client)
    resp = _package(
        client,
        **{"from": "2027-01-01T00:00:00Z", "to": "2026-01-01T00:00:00Z"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_parameter_errors_match_existing_checkpoint_route(client):
    # The 422 semantics are identical: same issue loc/type/msg for a
    # representative malformed request on both routes.
    def issue(url):
        resp = client.get(url)
        assert resp.status_code == 422
        details = resp.json()["error"]["details"]["issues"]
        return [(i["loc"], i["type"], i["msg"]) for i in details]

    for suffix in (
        "limit=1",
        "event_type=a&event_type=b",
        "from=not-a-time",
        "event_type=",
        "bogus=1",
    ):
        assert issue(
            f"/v1/audit-events/checkpoint/package?{suffix}"
        ) == issue(f"/v1/audit-events/checkpoint?{suffix}")


# --- Read-only guarantee and determinism ------------------------------------------


def test_reads_and_failures_write_no_rows_or_audit_events(client, db_session):
    _setup_events(client)

    def counts():
        return (
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
            db_session.scalar(select(func.count()).select_from(Content)),
        )

    before = counts()

    # Successful filtered, unfiltered, and empty package reads.
    _package(client)
    _package(client, event_type=EVENT_CONTENT_CREATED)
    _package(client, event_type="no.such.event")
    _package(client, **{"from": "2026-01-01T00:00:00Z"})

    # Failed reads must not write anything either.
    _package(client, event_type=" ")
    _package(client, limit=1)
    _package(client, cursor="tampered")
    _package(client, **{"from": "not-a-time"})
    _package(
        client,
        **{"from": "2027-01-01T00:00:00Z", "to": "2026-01-01T00:00:00Z"},
    )
    client.get("/v1/audit-events/checkpoint/package?event_type=a&event_type=b")
    client.get("/v1/audit-events/checkpoint/package?unknown=1")

    db_session.expire_all()
    assert counts() == before


def test_package_is_deterministic_across_repeated_reads(client):
    _setup_events(client)
    first = _package(client)
    second = _package(client)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
