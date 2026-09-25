"""Tests for the global read-only attestation-revocation search.

Covers ``GET /v1/attestation-revocations``: the empty-body contract, the
exact ``items``/``count``/``next_cursor`` member order in compact UTF-8 JSON
terminated by one newline, each item reusing the revocation public view
(id/attestation_id/revoker_actor_id/reason/created_at only -- never proof
text, signatures, authentication headers, private keys, or content bytes),
stable creation ordering (``created_at`` with the persistent insertion-order
tiebreaker that survives an app restart), the exact
``attestation_id``/``revoker_actor_id``/``reason`` filters (case- and
whitespace-sensitive, an unknown value is an empty collection), strict
RFC 3339 UTC ``from``/``to`` inclusive bounds with ``from`` not later than
``to``, pure-decimal ``limit`` 1..100 defaulting to 50, the opaque HMAC-signed
``ar1`` cursor family (binding every effective filter and the limit,
resuming without duplication or omission, a null cursor on the final page,
and an empty page with the original count at or past the tail), every 422
validation boundary, the 405 rejection of PUT/PATCH/DELETE, and the strictly
read-only guarantee (no revocation, grant, or audit writes on success,
empty results, repeated reads, or failures).
"""

from __future__ import annotations

import base64
import hashlib
import hmac as hmac_mod
import json
from datetime import datetime, timezone

from sqlalchemy import func, select

from provenance import pagination
from provenance.models import (
    AttestationRevocation,
    AuditEvent,
)
from provenance.signing import attestation_message_bytes
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    content_payload,
    create_actor,
    ed25519_public_key,
    ed25519_sign,
    SEED_A,
    SEED_B,
)

REVOCATIONS_PATH = "/v1/attestation-revocations"
REASON_A = "key compromised during incident"
REASON_B = "signer requested withdrawal"


# --- Fixture-style setup ------------------------------------------------------


def _create_content(client, actor_id="org-1", digest=DIGEST_A):
    resp = client.post(
        "/v1/contents", json=content_payload(actor_id=actor_id, digest=digest)
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, actor_id="org-1", claim_type="authorship"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": claim_type,
            "payload": {"statement": "attested"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _attest(client, target_type, target_id, *, seed=SEED_A, signer_actor_id="org-1"):
    signature = ed25519_sign(
        seed, attestation_message_bytes(target_type, target_id, signer_actor_id)
    )
    resp = client.post(
        "/v1/attestations",
        json={
            "target_type": target_type,
            "target_id": target_id,
            "signer_actor_id": signer_actor_id,
            "public_key": base64.b64encode(ed25519_public_key(seed)).decode("ascii"),
            "signature": base64.b64encode(signature).decode("ascii"),
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _revoke(client, attestation_id, *, revoker_actor_id="org-1", reason=REASON_A):
    resp = client.post(
        REVOCATIONS_PATH,
        json={
            "attestation_id": attestation_id,
            "revoker_actor_id": revoker_actor_id,
            "reason": reason,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _world(client):
    """Create four revocation records in a stable, known order.

    Returns ``(records, attestations)`` where records/attestations are in
    creation order: ``r_a`` (att_a, org-1, REASON_A), ``r_b`` (att_a,
    org-2, REASON_B), ``r_c`` (att_b, org-1, REASON_A), ``r_d`` (att_b,
    org-2, REASON_B).
    """
    create_actor(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")

    att_a = _attest(
        client, "claim", _create_claim(client, _create_content(client)["id"])["id"]
    )
    att_b = _attest(
        client,
        "claim",
        _create_claim(client, _create_content(client, digest=DIGEST_B)["id"])["id"],
        seed=SEED_B,
    )
    records = [
        _revoke(client, att_a["id"], revoker_actor_id="org-1", reason=REASON_A),
        _revoke(client, att_a["id"], revoker_actor_id="org-2", reason=REASON_B),
        _revoke(client, att_b["id"], revoker_actor_id="org-1", reason=REASON_A),
        _revoke(client, att_b["id"], revoker_actor_id="org-2", reason=REASON_B),
    ]
    return records, [att_a, att_b]


def _list(client, **params):
    return client.get(REVOCATIONS_PATH, params=params)


def _walk_pages(client, **params):
    """Follow next_cursor until exhausted; return (all_items, pages, count)."""
    pages = []
    all_items = []
    count = None
    cursor = None
    for _ in range(100):
        query = {**params}
        if cursor is not None:
            query["cursor"] = cursor
        resp = _list(client, **query)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        count = body["count"]
        pages.append(body["items"])
        all_items.extend(body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            break
    return all_items, pages, count


def _audit_count(session):
    return session.scalar(select(func.count()).select_from(AuditEvent))


def _revocation_count(session):
    return session.scalar(
        select(func.count()).select_from(AttestationRevocation)
    )


# --- Empty collection and response shape --------------------------------------


def test_empty_database_is_an_empty_collection(client):
    resp = _list(client)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_response_is_compact_json_with_ordered_members_and_one_newline(client):
    records, _ = _world(client)
    resp = _list(client)
    assert resp.status_code == 200
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    # The top-level members appear in exactly this order.
    assert raw.startswith(b'{"items":[')
    assert b'],"count":4,"next_cursor":null}' in raw
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        .encode("utf-8")
        + b"\n"
    )
    assert raw == expected


def test_items_carry_exactly_the_revocation_public_view(client):
    records, _ = _world(client)
    body = _list(client).json()
    assert body["count"] == 4
    assert body["next_cursor"] is None
    assert [item["id"] for item in body["items"]] == [r["id"] for r in records]
    for item, record in zip(body["items"], records):
        assert list(item) == [
            "id",
            "attestation_id",
            "revoker_actor_id",
            "reason",
            "created_at",
        ]
        assert item == record
        created_at = datetime.fromisoformat(item["created_at"])
        assert created_at.utcoffset().total_seconds() == 0
        assert item["created_at"].endswith(("Z", "+00:00"))
    # Proof text, signatures, authentication headers, private keys, content
    # bytes, and the ordering column can never appear on the wire.
    rendered = client.get(REVOCATIONS_PATH).content.decode()
    for forbidden in (
        "signature",
        "public_key",
        "private",
        "authorization",
        "x-pa",
        "x-pt",
        "x-ps",
        "payload",
        "digest",
        "seq",
    ):
        assert forbidden not in rendered


# --- Stable ordering -----------------------------------------------------------


def test_items_follow_stable_creation_order(client):
    records, _ = _world(client)
    body = _list(client).json()
    assert [item["id"] for item in body["items"]] == [r["id"] for r in records]
    created = [datetime.fromisoformat(item["created_at"]) for item in body["items"]]
    assert created == sorted(created)


def test_same_timestamp_ties_break_by_persistence_order(client, db_session):
    # Force every record onto one instant: the persistent insertion order
    # (seq) must still yield a stable, restart-durable order.
    records, _ = _world(client)
    tie = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    for row in db_session.execute(select(AttestationRevocation)).scalars():
        row.created_at = tie
    db_session.commit()

    body = _list(client).json()
    assert [item["id"] for item in body["items"]] == [r["id"] for r in records]


def test_ordering_is_identical_across_an_app_restart(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    records, _ = _world(file_client)
    first = file_client.get(REVOCATIONS_PATH)
    assert first.status_code == 200

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as restarted_client:
        second = restarted_client.get(REVOCATIONS_PATH)
        assert second.status_code == 200
        assert second.json() == first.json()
        assert [item["id"] for item in second.json()["items"]] == [
            r["id"] for r in records
        ]


# --- Exact-match filtering -----------------------------------------------------


def test_attestation_id_filter_is_an_exact_match(client):
    records, attestations = _world(client)
    body = _list(client, attestation_id=attestations[0]["id"]).json()
    assert body["count"] == 2
    assert body["next_cursor"] is None
    assert [item["id"] for item in body["items"]] == [
        records[0]["id"],
        records[1]["id"],
    ]


def test_revoker_actor_id_and_reason_filters_match_exactly(client):
    records, _ = _world(client)
    by_revoker = _list(client, revoker_actor_id="org-2").json()
    assert [item["id"] for item in by_revoker["items"]] == [
        records[1]["id"],
        records[3]["id"],
    ]
    assert by_revoker["count"] == 2

    by_reason = _list(client, reason=REASON_B).json()
    assert [item["id"] for item in by_reason["items"]] == [
        records[1]["id"],
        records[3]["id"],
    ]
    assert by_reason["count"] == 2


def test_filters_combine_as_logical_and(client):
    records, attestations = _world(client)
    body = _list(
        client,
        attestation_id=attestations[1]["id"],
        revoker_actor_id="org-1",
        reason=REASON_A,
    ).json()
    assert body["count"] == 1
    assert [item["id"] for item in body["items"]] == [records[2]["id"]]

    # A combination no record satisfies is an empty set, not an error.
    miss = _list(
        client,
        attestation_id=attestations[0]["id"],
        revoker_actor_id="org-1",
        reason=REASON_B,
    ).json()
    assert miss == {"items": [], "count": 0, "next_cursor": None}


def test_filters_are_case_and_whitespace_sensitive(client):
    _world(client)
    for params in (
        {"revoker_actor_id": "ORG-1"},
        {"revoker_actor_id": "org-1 "},
        {"revoker_actor_id": " org-1"},
        {"reason": REASON_A.upper()},
        {"reason": f" {REASON_A}"},
        {"reason": f"{REASON_A}\t"},
    ):
        resp = _list(client, **params)
        assert resp.status_code == 200, params
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_unknown_or_nonexistent_filter_value_is_an_empty_collection(client):
    records, attestations = _world(client)
    for params in (
        {"attestation_id": "att_ghost"},
        {"revoker_actor_id": "ghost"},
        {"reason": "no such reason was ever recorded"},
        {"attestation_id": attestations[0]["id"], "revoker_actor_id": "ghost"},
    ):
        resp = _list(client, **params)
        assert resp.status_code == 200, params
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_blank_filters_are_422(client):
    _world(client)
    for field in ("attestation_id", "revoker_actor_id", "reason"):
        for blank in ("", "   ", "\t"):
            resp = _list(client, **{field: blank})
            assert resp.status_code == 422, (field, repr(blank))
            assert resp.json()["error"]["code"] == "validation_error"


# --- Time-bound filtering -------------------------------------------------------

T1 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc)
T3 = datetime(2026, 1, 3, 23, 59, 59, tzinfo=timezone.utc)


def _fix_times(db_session, records):
    times = [T1, T2, T3, T3]
    by_id = {record["id"]: instant for record, instant in zip(records, times)}
    for row in db_session.execute(select(AttestationRevocation)).scalars():
        row.created_at = by_id[row.id]
    db_session.commit()


def test_from_bound_is_inclusive(client, db_session):
    records, _ = _world(client)
    _fix_times(db_session, records)
    body = _list(client, **{"from": "2026-01-02T12:30:00Z"}).json()
    assert [item["id"] for item in body["items"]] == [
        records[1]["id"],
        records[2]["id"],
        records[3]["id"],
    ]
    assert body["count"] == 3


def test_to_bound_is_inclusive(client, db_session):
    records, _ = _world(client)
    _fix_times(db_session, records)
    body = _list(client, to="2026-01-02T12:30:00Z").json()
    assert [item["id"] for item in body["items"]] == [
        records[0]["id"],
        records[1]["id"],
    ]
    assert body["count"] == 2


def test_from_equal_to_matches_exact_instant(client, db_session):
    records, _ = _world(client)
    _fix_times(db_session, records)
    body = _list(
        client,
        **{"from": "2026-01-03T23:59:59Z", "to": "2026-01-03T23:59:59Z"},
    ).json()
    # Same-instant ties keep persistent insertion order.
    assert [item["id"] for item in body["items"]] == [
        records[2]["id"],
        records[3]["id"],
    ]
    assert body["count"] == 2


def test_time_window_between_instants(client, db_session):
    records, _ = _world(client)
    _fix_times(db_session, records)
    body = _list(
        client,
        **{"from": "2026-01-01T00:00:01Z", "to": "2026-01-03T00:00:00Z"},
    ).json()
    assert [item["id"] for item in body["items"]] == [records[1]["id"]]
    assert body["count"] == 1


def test_time_filters_combine_with_exact_filters(client, db_session):
    records, attestations = _world(client)
    _fix_times(db_session, records)
    body = _list(
        client,
        attestation_id=attestations[1]["id"],
        **{"from": "2026-01-03T00:00:00Z"},
    ).json()
    assert [item["id"] for item in body["items"]] == [
        records[2]["id"],
        records[3]["id"],
    ]

    body = _list(
        client,
        revoker_actor_id="org-1",
        to="2026-01-01T00:00:00Z",
    ).json()
    assert [item["id"] for item in body["items"]] == [records[0]["id"]]


def test_utc_offset_notation_and_fractional_seconds_accepted(client, db_session):
    records, _ = _world(client)
    _fix_times(db_session, records)
    body = _list(
        client,
        **{
            "from": "2026-01-02T12:30:00+00:00",
            "to": "2026-01-03T23:59:59.000Z",
        },
    ).json()
    assert [item["id"] for item in body["items"]] == [
        records[1]["id"],
        records[2]["id"],
        records[3]["id"],
    ]


def test_from_later_than_to_is_validation_error(client, db_session):
    records, _ = _world(client)
    _fix_times(db_session, records)
    resp = _list(
        client,
        **{"from": "2026-01-03T00:00:00Z", "to": "2026-01-02T00:00:00Z"},
    )
    assert resp.status_code == 422
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
        ):
            resp = _list(client, **{field: value})
            assert resp.status_code == 422, (field, repr(value))
            assert resp.json()["error"]["code"] == "validation_error"


# --- limit validation -----------------------------------------------------------


def test_limit_boundaries_one_and_one_hundred_are_accepted(client):
    _world(client)
    assert _list(client, limit="1").status_code == 200
    assert _list(client, limit="100").status_code == 200


def test_limit_must_be_a_pure_decimal_integer_in_range(client):
    _world(client)
    for bad in ("0", "101", "-1", "5.0", "5e0", " 5", "5 ", "five", "+5", ""):
        resp = _list(client, limit=bad)
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_default_limit_is_fifty(client):
    records, _ = _world(client)
    # Bulk additional records on fresh attestations to cross 50.
    for index in range(50):
        content = _create_content(client, digest=hashlib.sha256(
            f"bulk-{index}".encode()
        ).hexdigest())
        claim = _create_claim(client, content["id"])
        att = _attest(client, "claim", claim["id"])
        _revoke(client, att["id"], reason=f"bulk reason {index}")
    resp = _list(client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 54
    assert len(body["items"]) == 50
    assert body["next_cursor"] is not None


# --- Parameter strictness -------------------------------------------------------


def test_repeated_and_undeclared_parameters_are_422(client):
    _world(client)
    assert (
        client.get(
            REVOCATIONS_PATH,
            params=[("revoker_actor_id", "org-1"), ("revoker_actor_id", "org-2")],
        ).status_code
        == 422
    )
    assert (
        client.get(
            REVOCATIONS_PATH, params=[("reason", "a"), ("reason", "b")]
        ).status_code
        == 422
    )
    assert (
        client.get(
            REVOCATIONS_PATH, params=[("limit", "1"), ("limit", "2")]
        ).status_code
        == 422
    )
    assert (
        client.get(
            REVOCATIONS_PATH,
            params=[("from", "2026-01-01T00:00:00Z"),
                    ("from", "2026-01-02T00:00:00Z")],
        ).status_code
        == 422
    )
    for params in (
        {"revoker": "org-1"},
        {"attestation": "att_x"},
        {"offset": "1"},
        {"q": "x"},
    ):
        resp = _list(client, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_empty_body_is_422_rejected_before_any_read(client, db_session):
    _world(client)
    audit_before = _audit_count(db_session)
    for body in (b"{}", b" ", b"{not valid json", b"[]", b"\n"):
        resp = client.request(
            "GET",
            REVOCATIONS_PATH,
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"
    assert _audit_count(db_session) == audit_before


# --- Pagination -----------------------------------------------------------------


def test_pages_resume_without_duplication_or_omission(client):
    records, _ = _world(client)
    all_items, pages, count = _walk_pages(client, limit="2")
    assert count == 4
    assert [len(page) for page in pages] == [2, 2]
    assert [item["id"] for item in all_items] == [r["id"] for r in records]
    assert pages[0][0]["id"] == records[0]["id"]
    assert client.get(
        REVOCATIONS_PATH, params={"limit": "2"}
    ).json()["next_cursor"].startswith("ar1.")


def test_count_covers_every_page_including_a_filtered_one(client):
    records, attestations = _world(client)
    all_items, pages, count = _walk_pages(
        client, attestation_id=attestations[1]["id"], limit="1"
    )
    assert count == 2
    assert [len(page) for page in pages] == [1, 1]
    assert [item["id"] for item in all_items] == [
        records[2]["id"],
        records[3]["id"],
    ]


def test_replayed_cursor_returns_the_same_page(client):
    _world(client)
    cursor = _list(client, limit="1").json()["next_cursor"]
    assert cursor is not None
    page = _list(client, limit="1", cursor=cursor)
    replay = _list(client, limit="1", cursor=cursor)
    assert page.status_code == 200
    assert replay.status_code == 200
    assert replay.content == page.content


def test_cursor_at_or_past_the_tail_returns_empty_page_with_count(client, app):
    _world(client)
    claims = {
        "attestation_id": None,
        "revoker_actor_id": None,
        "reason": None,
        "from": None,
        "to": None,
        "limit": 50,
    }
    cursor = pagination.encode_typed_cursor(
        app.state.attestation_revocations_cursor_secret,
        pagination.ATTESTATION_REVOCATIONS_CURSOR,
        {**claims, "offset": 4},
    )
    resp = _list(client, cursor=cursor)
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 4, "next_cursor": None}

    past = pagination.encode_typed_cursor(
        app.state.attestation_revocations_cursor_secret,
        pagination.ATTESTATION_REVOCATIONS_CURSOR,
        {**claims, "offset": 99},
    )
    resp = _list(client, cursor=past)
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 4, "next_cursor": None}


def test_cursor_binds_every_effective_filter_and_limit(client):
    _world(client)
    cursor = _list(client, limit="2").json()["next_cursor"]
    assert cursor is not None
    # A different limit, a new filter, or a dropped limit all mismatch.
    for params in (
        {"limit": "3", "cursor": cursor},
        {"revoker_actor_id": "org-1", "limit": "2", "cursor": cursor},
        {"reason": REASON_A, "limit": "2", "cursor": cursor},
        {"cursor": cursor},
    ):
        resp = _list(client, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"

    # A cursor minted under a filter cannot resume a different filter.
    mid = _list(
        client, revoker_actor_id="org-2", limit="1"
    ).json()["next_cursor"]
    assert mid is not None
    mismatch = _list(client, revoker_actor_id="org-1", limit="1", cursor=mid)
    assert mismatch.status_code == 422
    changed_attestation = _list(
        client, attestation_id="att_ghost", limit="1", cursor=mid
    )
    assert changed_attestation.status_code == 422

    # Time-bound cursors bind their bounds; equivalent notations still match.
    timed_cursor = _list(
        client, **{"from": "2026-01-01T00:00:00Z", "limit": "2"}
    ).json()["next_cursor"]
    assert timed_cursor is not None
    assert (
        _list(
            client, **{"from": "2026-01-02T00:00:00Z", "limit": "2"},
            cursor=timed_cursor,
        ).status_code
        == 422
    )
    same_instant = _list(
        client, **{"from": "2026-01-01T00:00:00+00:00", "limit": "2"},
        cursor=timed_cursor,
    )
    assert same_instant.status_code == 200


def test_blank_malformed_and_tampered_cursors_are_422(client):
    _world(client)
    valid = _list(client, limit="1").json()["next_cursor"]
    assert valid is not None
    tampered = valid[:-1] + ("A" if valid[-1] != "A" else "B")
    for bad in ("", "   ", "not-a-cursor", "ar1", "ar1.abc", tampered):
        resp = _list(client, cursor=bad)
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_foreign_family_cursor_is_422(client, app):
    _world(client)
    # A well-formed cursor minted by another endpoint family never resumes
    # this retrieval.
    foreign = pagination.encode_typed_cursor(
        app.state.audit_events_cursor_secret,
        pagination.AUDIT_EVENTS_CURSOR,
        {
            "event_type": None,
            "resource_id": None,
            "from": None,
            "to": None,
            "limit": 50,
            "offset": 1,
        },
    )
    resp = _list(client, cursor=foreign)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    claims_cursor = pagination.encode_typed_cursor(
        app.state.claims_cursor_secret,
        pagination.CLAIMS_CURSOR,
        {
            "content_id": None,
            "actor_id": None,
            "claim_type": None,
            "payload_digest_hex": None,
            "limit": 50,
            "offset": 1,
        },
    )
    assert _list(client, cursor=claims_cursor).status_code == 422


def test_cursor_with_wrong_claim_set_is_422(client, app):
    _world(client)
    # Hand-built ar1 token with a valid HMAC but a payload that drops a
    # bound claim.
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "attestation_id": None,
                "revoker_actor_id": None,
                "reason": None,
                "limit": 50,
                "offset": 1,
            },
            separators=(",", ":"),
        ).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(
        hmac_mod.new(
            app.state.attestation_revocations_cursor_secret,
            f"ar1.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    resp = _list(client, cursor=f"ar1.{payload}.{sig}")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Method boundary and POST coexistence --------------------------------------


def test_put_patch_and_delete_are_405_method_not_allowed(client):
    _world(client)
    for method in (client.put, client.patch, client.delete):
        resp = method(REVOCATIONS_PATH)
        assert resp.status_code == 405
        assert resp.json()["error"]["code"] == "method_not_allowed"


def test_post_remains_the_revocation_creation_route(client):
    # The new read entry neither shadows nor alters creation semantics.
    create_actor(client)
    claim = _create_claim(client, _create_content(client)["id"])
    att = _attest(client, "claim", claim["id"])
    first = _revoke(client, att["id"])
    repeat = client.post(
        REVOCATIONS_PATH,
        json={
            "attestation_id": att["id"],
            "revoker_actor_id": "org-1",
            "reason": REASON_A,
        },
    )
    assert repeat.status_code == 200
    assert repeat.json() == first


# --- Read-only guarantee ---------------------------------------------------------


def test_queries_and_failures_write_nothing(client, db_session):
    records, attestations = _world(client)
    revocations_before = _revocation_count(db_session)
    audit_before = _audit_count(db_session)

    assert _list(client).status_code == 200
    assert _list(client, attestation_id="att_ghost").status_code == 200
    assert _list(client, revoker_actor_id="ghost").status_code == 200
    assert _list(client, reason="nothing").status_code == 200
    assert _list(client, limit="2").status_code == 200
    assert _list(client, limit="0").status_code == 422
    assert _list(client, revoker_actor_id=" ").status_code == 422
    assert _list(
        client, **{"from": "2026-01-02T12:30:00Z", "to": "2026-01-01T00:00:00Z"}
    ).status_code == 422
    assert _list(client, cursor="bad").status_code == 422
    assert (
        client.request("GET", REVOCATIONS_PATH, content=b"{}").status_code == 422
    )

    assert _revocation_count(db_session) == revocations_before == 4
    assert _audit_count(db_session) == audit_before
