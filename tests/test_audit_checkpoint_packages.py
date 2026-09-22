"""Tests for the read-only audit-event checkpoint package export.

Covers GET /v1/audit-events/checkpoint/package: the success body is exactly
{"checkpoint", "events"}; ``checkpoint`` is byte-for-byte the body of the
existing GET /v1/audit-events/checkpoint route under the same effective
filter (exactly checkpoint_version/digest_algorithm/event_count/
events_digest_hex, version "provenance-audit-checkpoint-v1", sha256) and
``events`` lists the same filtered events the audit-event search serves, in
stable creation order, each reduced to event_type/resource_id/UTC
created_at. The checkpoint digest is independently recomputed from the
events member of the same response under the existing checkpoint canonical
rules (array order kept, object keys sorted by Unicode code point, compact
separators, unescaped non-ASCII, UTF-8), so both halves are bound to one
read state; the pair even verifies offline via the stateless
checkpoint-verification route. event_type/resource_id are non-empty exact
combinable filters and from/to are strict RFC 3339 UTC inclusive bounds
exactly as on the checkpoint route; limit/cursor and any other undeclared,
blank, repeated, or malformed parameter is a 422 validation_error; an empty
match set returns "events": [] together with the digest of the empty array;
and neither successful nor failed package reads write any resource or audit
rows. All fixtures are deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance.app import create_app
from provenance.config import Settings
from provenance.models import (
    EVENT_ACTOR_CREATED,
    EVENT_CONTENT_CREATED,
    Actor,
    AuditEvent,
    Claim,
    Content,
    EvidenceBundle,
)
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
    """Independently canonicalize the served event items and digest them."""
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


def test_success_body_has_exact_shape(client):
    _setup_events(client)
    resp = _package(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == PACKAGE_FIELDS
    assert set(body["checkpoint"]) == CHECKPOINT_FIELDS
    assert body["checkpoint"]["checkpoint_version"] == CHECKPOINT_VERSION
    assert body["checkpoint"]["digest_algorithm"] == DIGEST_ALGORITHM
    assert body["checkpoint"]["event_count"] == 5
    assert len(body["events"]) == 5
    for event in body["events"]:
        assert set(event) == EVENT_FIELDS


def test_event_items_carry_only_public_fields(client):
    _setup_events(client)
    body = _package(client).json()
    serialized = json.dumps(body)
    # No internal sequence/primary-key bookkeeping leaks into any event.
    for forbidden in ('"seq"', '"sequence"', '"rowid"', '"id"'):
        assert forbidden not in serialized
    for event in body["events"]:
        assert set(event) == EVENT_FIELDS


# --- Consistency with the existing checkpoint and search routes -------------------


def test_checkpoint_member_equals_existing_checkpoint_route(client):
    _setup_events(client)
    for params in (
        {},
        {"event_type": EVENT_CONTENT_CREATED},
        {"event_type": EVENT_ACTOR_CREATED, "resource_id": "org-1"},
        {"from": "2026-01-01T00:00:00Z"},
    ):
        package = _package(client, **params).json()
        checkpoint = _checkpoint(client, **params).json()
        # The checkpoint member is byte-for-byte the existing checkpoint body.
        assert package["checkpoint"] == checkpoint, params
        assert package["checkpoint"]["event_count"] == len(package["events"])


def test_events_member_equals_audit_search_view(client):
    _setup_events(client)
    params = {"event_type": EVENT_CONTENT_CREATED}
    package = _package(client, **params).json()
    searched = _search_items(client, **params)
    # The events member is exactly the filtered search wire view, in order.
    assert package["events"] == searched
    assert [e["event_type"] for e in package["events"]] == [
        EVENT_CONTENT_CREATED
    ] * len(searched)


def test_events_follow_stable_creation_order(client, db_session):
    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 2, tzinfo=timezone.utc)
    t3 = datetime(2026, 1, 3, tzinfo=timezone.utc)
    _insert_event(db_session, EVENT_ACTOR_CREATED, "org-1", t1)
    _insert_event(db_session, EVENT_CONTENT_CREATED, "cnt_t2", t2)
    _insert_event(db_session, EVENT_CONTENT_CREATED, "cnt_t3", t3)

    body = _package(client).json()
    assert [e["resource_id"] for e in body["events"]] == [
        "org-1",
        "cnt_t2",
        "cnt_t3",
    ]
    assert [e["created_at"] for e in body["events"]] == [
        "2026-01-01T00:00:00Z",
        "2026-01-02T00:00:00Z",
        "2026-01-03T00:00:00Z",
    ]


# --- Digest binding to the response's events --------------------------------------


def test_checkpoint_digest_binds_to_returned_events(client):
    _setup_events(client)
    body = _package(client).json()

    # Independently canonicalize the events member of THIS response: the
    # checkpoint in the same response must equal it.
    assert body["checkpoint"]["events_digest_hex"] == _expected_digest(
        body["events"]
    )
    assert body["checkpoint"]["event_count"] == len(body["events"])

    # Mutating the returned events breaks the binding, proving the digest
    # commits to exactly the served array.
    tampered = json.loads(json.dumps(body["events"]))
    tampered[0], tampered[1] = tampered[1], tampered[0]
    assert body["checkpoint"]["events_digest_hex"] != _expected_digest(tampered)

    augmented = list(body["events"])
    augmented.append(dict(body["events"][0]))
    assert body["checkpoint"]["events_digest_hex"] != _expected_digest(augmented)

    dropped = body["events"][:-1]
    assert body["checkpoint"]["events_digest_hex"] != _expected_digest(dropped)


def test_array_order_is_committed_by_digest(client, db_session):
    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 2, tzinfo=timezone.utc)
    t3 = datetime(2026, 1, 3, tzinfo=timezone.utc)
    _insert_event(db_session, EVENT_ACTOR_CREATED, "org-1", t1)
    _insert_event(db_session, EVENT_CONTENT_CREATED, "cnt_t2", t2)
    _insert_event(db_session, EVENT_CONTENT_CREATED, "cnt_t3", t3)

    body = _package(client).json()
    assert body["checkpoint"]["events_digest_hex"] == _expected_digest(
        body["events"]
    )
    reordered = [body["events"][1], body["events"][0], body["events"][2]]
    assert body["checkpoint"]["events_digest_hex"] != _expected_digest(reordered)


def test_non_ascii_values_are_not_escaped(client, db_session):
    _insert_event(
        db_session,
        EVENT_ACTOR_CREATED,
        "org-émoji-✓",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    body = _package(client).json()
    assert body["events"][0]["resource_id"] == "org-émoji-✓"
    assert body["checkpoint"]["events_digest_hex"] == _expected_digest(
        body["events"]
    )
    # And it differs from the ASCII-escaped canonicalization.
    escaped = json.dumps(
        body["events"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    assert body["checkpoint"]["events_digest_hex"] != hashlib.sha256(
        escaped
    ).hexdigest()


def test_package_pair_verifies_offline(client):
    _setup_events(client)
    body = _package(
        client, event_type=EVENT_CONTENT_CREATED
    ).json()
    # The package is self-contained: its checkpoint verifies against its own
    # events via the existing stateless verification route.
    resp = client.post(
        "/v1/audit-events/checkpoint-verifications",
        json={"checkpoint": body["checkpoint"], "events": body["events"]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


# --- Filtering --------------------------------------------------------------------


def test_event_type_filter_matches_routes(client):
    _setup_events(client)
    params = {"event_type": EVENT_CONTENT_CREATED}
    package = _package(client, **params).json()
    assert package["events"] == _search_items(client, **params)
    assert package["checkpoint"] == _checkpoint(client, **params).json()
    assert package["checkpoint"]["event_count"] == 3


def test_resource_id_filter_matches_routes(client):
    c1, _, _ = _setup_events(client)
    params = {"resource_id": c1["id"]}
    package = _package(client, **params).json()
    assert package["events"] == _search_items(client, **params)
    assert package["checkpoint"] == _checkpoint(client, **params).json()
    assert package["checkpoint"]["event_count"] == 1


def test_combined_filters_match_routes(client):
    _, c2, _ = _setup_events(client)
    params = {"event_type": EVENT_CONTENT_CREATED, "resource_id": c2["id"]}
    package = _package(client, **params).json()
    assert package["events"] == _search_items(client, **params)
    assert package["checkpoint"] == _checkpoint(client, **params).json()
    assert package["checkpoint"]["event_count"] == 1


def test_exact_match_is_case_and_whitespace_sensitive(client):
    _setup_events(client)
    empty_digest = hashlib.sha256(b"[]").hexdigest()
    for params in (
        {"event_type": "ACTOR.CREATED"},
        {"event_type": "actor.created "},
        {"resource_id": "ORG-1"},
        {"resource_id": "org-1 "},
    ):
        body = _package(client, **params).json()
        assert body["events"] == [], params
        assert body["checkpoint"]["event_count"] == 0, params
        assert body["checkpoint"]["events_digest_hex"] == empty_digest


def test_time_bounds_match_routes(client, db_session):
    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 2, 12, 30, tzinfo=timezone.utc)
    t3 = datetime(2026, 1, 3, 23, 59, 59, tzinfo=timezone.utc)
    _insert_event(db_session, EVENT_ACTOR_CREATED, "org-1", t1)
    _insert_event(db_session, EVENT_CONTENT_CREATED, "cnt_t2", t2)
    _insert_event(db_session, EVENT_CONTENT_CREATED, "cnt_t3", t3)

    for params in (
        {"from": "2026-01-02T12:30:00Z"},
        {"to": "2026-01-02T12:30:00+00:00"},
        {"from": "2026-01-02T12:30:00Z", "to": "2026-01-02T12:30:00.000Z"},
    ):
        package = _package(client, **params).json()
        assert package["events"] == _search_items(client, **params), params
        assert package["checkpoint"] == _checkpoint(client, **params).json()
        assert package["checkpoint"]["events_digest_hex"] == _expected_digest(
            package["events"]
        )


def test_equivalent_utc_spellings_agree(client, db_session):
    _insert_event(
        db_session,
        EVENT_ACTOR_CREATED,
        "org-1",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    zed = _package(client, **{"from": "2026-01-01T00:00:00Z"}).json()
    offset = _package(client, **{"from": "2026-01-01T00:00:00+00:00"}).json()
    assert zed["events"] == offset["events"]
    assert zed["checkpoint"] == offset["checkpoint"]


# --- Empty results ----------------------------------------------------------------


def test_empty_database_yields_empty_array_and_checkpoint(client):
    body = _package(client).json()
    assert set(body) == PACKAGE_FIELDS
    assert body["events"] == []
    assert set(body["checkpoint"]) == CHECKPOINT_FIELDS
    assert body["checkpoint"]["checkpoint_version"] == CHECKPOINT_VERSION
    assert body["checkpoint"]["digest_algorithm"] == DIGEST_ALGORITHM
    assert body["checkpoint"]["event_count"] == 0
    assert body["checkpoint"]["events_digest_hex"] == hashlib.sha256(
        b"[]"
    ).hexdigest()


def test_no_matching_events_yields_empty_array_and_checkpoint(client):
    _setup_events(client)
    body = _package(client, event_type="claim.created").json()
    assert body["events"] == []
    assert body["checkpoint"]["event_count"] == 0
    assert body["checkpoint"]["events_digest_hex"] == hashlib.sha256(
        b"[]"
    ).hexdigest()
    # The empty pair still verifies offline.
    resp = client.post(
        "/v1/audit-events/checkpoint-verifications",
        json={"checkpoint": body["checkpoint"], "events": body["events"]},
    )
    assert resp.json() == {"valid": True}


# --- Parameter validation ----------------------------------------------------------


def test_limit_and_cursor_are_undeclared_validation_errors(client):
    _setup_events(client)
    for suffix in ("limit=1", "cursor=x", "limit=50&cursor=abc"):
        resp = client.get(
            f"/v1/audit-events/checkpoint/package?{suffix}"
        )
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
        resp = client.get(
            f"/v1/audit-events/checkpoint/package?{suffix}"
        )
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
        resp = client.get(
            f"/v1/audit-events/checkpoint/package?{suffix}"
        )
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
        **{"from": "2026-01-03T00:00:00Z", "to": "2026-01-02T00:00:00Z"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_validation_errors_are_identical_to_checkpoint_route(client):
    # The package route must share the checkpoint route's exact 422 boundary.
    cases = (
        "limit=1",
        "cursor=x",
        "unknown=1",
        "event_type=a&event_type=b",
        "event_type=",
        "from=not-a-time",
    )
    for suffix in cases:
        package_resp = client.get(
            f"/v1/audit-events/checkpoint/package?{suffix}"
        )
        checkpoint_resp = client.get(
            f"/v1/audit-events/checkpoint?{suffix}"
        )
        assert package_resp.status_code == checkpoint_resp.status_code == 422
        assert package_resp.json() == checkpoint_resp.json()


# --- Read-only guarantee -----------------------------------------------------------


def test_reads_and_failures_write_no_rows_or_audit_events(client, db_session):
    _setup_events(client)

    models = (Actor, Content, Claim, EvidenceBundle, AuditEvent)

    def counts():
        return tuple(
            db_session.scalar(select(func.count()).select_from(model))
            for model in models
        )

    before = counts()

    # Successful unfiltered, filtered, and empty package reads.
    _package(client)
    _package(client, event_type=EVENT_CONTENT_CREATED)
    _package(client, event_type="no.such.event")
    _package(client, **{"from": "2026-01-01T00:00:00Z"})
    _package(
        client,
        **{
            "event_type": EVENT_CONTENT_CREATED,
            "resource_id": "missing",
            "from": "2026-01-01T00:00:00Z",
            "to": "2027-01-01T00:00:00Z",
        },
    )

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
    first = _package(client).json()
    second = _package(client).json()
    assert first == second


def test_package_is_deterministic_across_restart(file_client, tmp_db_url):
    _setup_events(file_client)
    expected = _package(file_client)
    assert expected.status_code == 200, expected.text
    expected_body = expected.json()

    # A brand-new app/engine over the same file reproduces the package
    # byte-for-byte (stable ids, UTC timestamps, and creation order).
    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        resp = _package(client)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body == expected_body
        assert body["checkpoint"]["events_digest_hex"] == _expected_digest(
            body["events"]
        )
        # The read after restart created nothing and stays stable.
        assert _package(client).json() == expected_body
