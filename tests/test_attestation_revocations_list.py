"""Tests for the global read-only attestation-revocation search.

Covers ``GET /v1/attestation-revocations``: the exact
``items``/``count``/``next_cursor`` member order in compact UTF-8 JSON
terminated by one newline, each item reusing the revocation public view
(no proof bytes, signature, authentication header, private key, or content
byte), stable creation ordering (``created_at`` with the persistent
insertion-order tiebreaker that survives an app restart), the exact
``attestation_id``/``revoker_actor_id``/``reason`` filters and the strict
inclusive RFC 3339 UTC ``from``/``to`` bounds (from must not be later than
to), unknown values yielding an empty collection, pure-decimal
``limit`` validation (1..100, default 50), the opaque HMAC-signed ``ar1``
cursor family binding every effective filter and the limit, paging without
duplication or omission, a null cursor on the final page and an empty page
with the original count at or past the tail, every 422 validation boundary
(blank/bad-type/bad-time/out-of-range/repeated/undeclared parameters,
non-empty bodies, blank/tampered/foreign-family/mismatched cursors), the
405 rejection of PUT/PATCH/DELETE, POST remaining the creation route, and
the strictly read-only guarantee on success, empty results, and failures.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timezone

from sqlalchemy import func, select

from provenance import pagination
from provenance.models import AttestationRevocation, AuditEvent
from provenance.signing import attestation_message_bytes
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    create_actor,
    content_payload,
    ed25519_public_key,
    ed25519_sign,
    SEED_A,
    SEED_B,
)

REVOCATIONS_PATH = "/v1/attestation-revocations"

REASON_A = "key compromise during incident"
REASON_B = "signer requested withdrawal"
REASON_C = "rotated signing material"


# --- Fixture-style setup ------------------------------------------------------


def _create_content(client, *, actor_id="org-1", digest=DIGEST_A):
    resp = client.post(
        "/v1/contents",
        json=content_payload(actor_id=actor_id, digest=digest),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, *, actor_id="org-1", claim_type="authorship"):
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


def _attest(client, target_id, *, seed=SEED_A, signer_actor_id="org-1"):
    signature = ed25519_sign(
        seed, attestation_message_bytes("claim", target_id, signer_actor_id)
    )
    resp = client.post(
        "/v1/attestations",
        json={
            "target_type": "claim",
            "target_id": target_id,
            "signer_actor_id": signer_actor_id,
            "public_key": base64.b64encode(
                ed25519_public_key(seed)
            ).decode("ascii"),
            "signature": base64.b64encode(signature).decode("ascii"),
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _revoke(client, attestation_id, *, revoker_actor_id, reason):
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
    """Two attestations and four interleaved revocations, in creation order."""
    create_actor(client, actor_id="org-1")
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")

    claim_one = _create_claim(client, _create_content(client)["id"])
    att_one = _attest(client, claim_one["id"], seed=SEED_A, signer_actor_id="org-1")
    claim_two = _create_claim(
        client, _create_content(client, actor_id="org-2", digest=DIGEST_B)["id"],
        actor_id="org-2",
    )
    att_two = _attest(client, claim_two["id"], seed=SEED_B, signer_actor_id="org-2")

    r1 = _revoke(client, att_one["id"], revoker_actor_id="org-1", reason=REASON_A)
    r2 = _revoke(client, att_one["id"], revoker_actor_id="org-2", reason=REASON_B)
    r3 = _revoke(client, att_two["id"], revoker_actor_id="org-2", reason=REASON_A)
    r4 = _revoke(client, att_two["id"], revoker_actor_id="org-1", reason=REASON_C)
    return att_one, att_two, [r1, r2, r3, r4]


def _pin_times(db_session, revocations, instants):
    """Overwrite created_at per revocation id with a fixed UTC instant."""
    by_id = {r["id"]: instant for r, instant in zip(revocations, instants)}
    for row in db_session.execute(select(AttestationRevocation)).scalars():
        row.created_at = by_id[row.id]
    db_session.commit()


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
    return len(session.execute(select(AuditEvent)).scalars().all())


# --- Empty collection and response shape --------------------------------------


def test_empty_store_is_an_empty_collection(client):
    resp = _list(client)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_response_is_compact_json_with_ordered_members_and_one_newline(client):
    _, _, revocations = _world(client)
    resp = _list(client)
    assert resp.status_code == 200
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    assert raw.startswith(b'{"items":[')
    assert b'],"count":4,"next_cursor":null}' in raw
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        .encode("utf-8")
        + b"\n"
    )
    assert raw == expected


def test_items_carry_exactly_the_revocation_public_view(client):
    _, _, revocations = _world(client)
    resp = _list(client)
    body = resp.json()
    assert body["count"] == 4
    assert body["next_cursor"] is None
    assert [item["id"] for item in body["items"]] == [r["id"] for r in revocations]
    for item, revocation in zip(body["items"], revocations):
        assert list(item) == [
            "id",
            "attestation_id",
            "revoker_actor_id",
            "reason",
            "created_at",
        ]
        assert item == revocation
        created_at = datetime.fromisoformat(item["created_at"])
        assert created_at.utcoffset().total_seconds() == 0
        assert item["created_at"].endswith(("Z", "+00:00"))
    # No proof, signature, authentication, key, or byte material is echoed.
    rendered = resp.content.decode()
    for forbidden in (
        "signature",
        "public_key",
        "private_key",
        "authorization",
        "payload",
        "content_bytes",
        "seq",
    ):
        assert forbidden not in rendered


# --- Stable ordering -----------------------------------------------------------


def test_items_follow_stable_creation_order(client):
    _, _, revocations = _world(client)
    body = _list(client).json()
    assert [item["id"] for item in body["items"]] == [
        r["id"] for r in revocations
    ]
    created = [datetime.fromisoformat(item["created_at"]) for item in body["items"]]
    assert created == sorted(created)


def test_same_timestamp_ties_break_by_persistence_order(client, db_session):
    _, _, revocations = _world(client)
    tie = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    _pin_times(db_session, revocations, [tie] * 4)

    body = _list(client).json()
    assert [item["id"] for item in body["items"]] == [
        r["id"] for r in revocations
    ]


def test_ordering_is_identical_across_an_app_restart(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    _, _, revocations = _world(file_client)
    first = _list(file_client)
    assert first.status_code == 200

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as restarted_client:
        second = _list(restarted_client)
        assert second.status_code == 200
        assert second.json() == first.json()
        assert [item["id"] for item in second.json()["items"]] == [
            r["id"] for r in revocations
        ]


# --- Exact-match filters --------------------------------------------------------


def test_attestation_id_filter_is_an_exact_match(client):
    att_one, _att_two, revocations = _world(client)
    body = _list(client, attestation_id=att_one["id"]).json()
    assert body["count"] == 2
    assert body["next_cursor"] is None
    assert [item["id"] for item in body["items"]] == [
        revocations[0]["id"],
        revocations[1]["id"],
    ]


def test_revoker_actor_id_filter_is_an_exact_match(client):
    _, _, revocations = _world(client)
    body = _list(client, revoker_actor_id="org-1").json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [
        revocations[0]["id"],
        revocations[3]["id"],
    ]


def test_reason_filter_is_an_exact_match(client):
    _, _, revocations = _world(client)
    body = _list(client, reason=REASON_A).json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [
        revocations[0]["id"],
        revocations[2]["id"],
    ]


def test_filters_combine_as_logical_and(client):
    att_one, _, revocations = _world(client)
    body = _list(
        client,
        attestation_id=att_one["id"],
        revoker_actor_id="org-2",
        reason=REASON_B,
    ).json()
    assert body["count"] == 1
    assert [item["id"] for item in body["items"]] == [revocations[1]["id"]]

    miss = _list(
        client,
        attestation_id=att_one["id"],
        revoker_actor_id="org-2",
        reason=REASON_A,
    ).json()
    assert miss == {"items": [], "count": 0, "next_cursor": None}


def test_filters_are_case_and_whitespace_sensitive(client):
    _world(client)
    for params in (
        {"attestation_id": "ATT_GHOST"},
        {"revoker_actor_id": "Org-1"},
        {"revoker_actor_id": " org-1"},
        {"revoker_actor_id": "org-1 "},
        {"reason": "KEY COMPROMISE DURING INCIDENT"},
        {"reason": f" {REASON_A}"},
        {"reason": f"{REASON_A}\t"},
    ):
        resp = _list(client, **params)
        assert resp.status_code == 200, params
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_unknown_or_nonexistent_filter_value_is_an_empty_collection(client):
    _world(client)
    for params in (
        {"attestation_id": "att_ghost"},
        {"revoker_actor_id": "ghost"},
        {"reason": "no such reason was ever recorded"},
        {
            "attestation_id": "att_ghost",
            "revoker_actor_id": "ghost",
            "reason": REASON_A,
        },
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


# --- Time bounds ----------------------------------------------------------------


def test_from_bound_is_inclusive(client, db_session):
    _, _, revocations = _world(client)
    _pin_times(
        db_session,
        revocations,
        [
            datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 3, 9, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 4, 0, 0, 0, tzinfo=timezone.utc),
        ],
    )
    body = _list(client, **{"from": "2026-01-02T12:30:00Z"}).json()
    assert [item["id"] for item in body["items"]] == [
        r["id"] for r in revocations[1:]
    ]
    assert body["count"] == 3


def test_to_bound_is_inclusive(client, db_session):
    _, _, revocations = _world(client)
    _pin_times(
        db_session,
        revocations,
        [
            datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 3, 9, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 4, 0, 0, 0, tzinfo=timezone.utc),
        ],
    )
    body = _list(client, to="2026-01-02T12:30:00Z").json()
    assert [item["id"] for item in body["items"]] == [
        r["id"] for r in revocations[:2]
    ]
    assert body["count"] == 2


def test_equal_from_and_to_selects_exactly_that_instant(client, db_session):
    _, _, revocations = _world(client)
    _pin_times(
        db_session,
        revocations,
        [
            datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 4, 0, 0, 0, tzinfo=timezone.utc),
        ],
    )
    body = _list(
        client, **{"from": "2026-01-02T12:30:00Z", "to": "2026-01-02T12:30:00Z"}
    ).json()
    assert [item["id"] for item in body["items"]] == [
        revocations[1]["id"],
        revocations[2]["id"],
    ]
    assert body["count"] == 2


def test_equivalent_utc_notations_are_the_same_bound(client, db_session):
    _, _, revocations = _world(client)
    instant = datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc)
    _pin_times(
        db_session,
        revocations,
        [
            datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            instant,
            datetime(2026, 1, 3, 9, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 4, 0, 0, 0, tzinfo=timezone.utc),
        ],
    )
    zed = _list(client, **{"from": "2026-01-02T12:30:00Z"}).json()
    offset = _list(client, **{"from": "2026-01-02T12:30:00+00:00"}).json()
    assert zed == offset
    assert [item["id"] for item in zed["items"]] == [
        r["id"] for r in revocations[1:]
    ]


def test_bad_time_bounds_and_inverted_range_are_422(client):
    _world(client)
    for bad in (
        "2026-01-02",
        "2026-01-02T12:30:00",
        "2026-01-02 12:30:00Z",
        "2026-01-02T12:30Z",
        "2026-01-02T12:30:00+01:00",
        "2026-13-02T12:30:00Z",
        "2026-01-02T25:30:00Z",
        "2026-01-02t12:30:00z",
        "soon",
    ):
        resp = _list(client, **{"from": bad})
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error"
        resp = _list(client, to=bad)
        assert resp.status_code == 422, repr(bad)
    inverted = _list(
        client, **{"from": "2026-01-03T00:00:00Z", "to": "2026-01-02T00:00:00Z"}
    )
    assert inverted.status_code == 422
    assert inverted.json()["error"]["code"] == "validation_error"
    for blank in ("", "   "):
        assert _list(client, **{"from": blank}).status_code == 422
        assert _list(client, to=blank).status_code == 422


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
    create_actor(client, actor_id="org-1")
    claim = _create_claim(client, _create_content(client)["id"])
    att = _attest(client, claim["id"])
    for index in range(51):
        _revoke(
            client, att["id"], revoker_actor_id="org-1", reason=f"reason-{index:03d}"
        )
    resp = _list(client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 51
    assert len(body["items"]) == 50
    assert body["next_cursor"] is not None


# --- Parameter strictness -------------------------------------------------------


def test_repeated_and_undeclared_parameters_are_422(client):
    _world(client)
    repeated = [
        ("attestation_id", "att_a"),
        ("attestation_id", "att_b"),
    ]
    assert client.get(REVOCATIONS_PATH, params=repeated).status_code == 422
    assert (
        client.get(
            REVOCATIONS_PATH,
            params=[("reason", REASON_A), ("reason", REASON_B)],
        ).status_code
        == 422
    )
    assert (
        client.get(
            REVOCATIONS_PATH,
            params=[("limit", "1"), ("limit", "2")],
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
        {"id": "rev_1"},
        {"revoker": "org-1"},
        {"offset": "1"},
        {"q": "x"},
        {"cursor": "x", "bogus": "y"},
    ):
        resp = _list(client, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_empty_body_is_422_rejected_before_any_read(client, db_session):
    _world(client)
    events_before = _audit_count(db_session)
    for body in (b"{}", b" ", b"{not valid json", b"[]", b"\n", b"\x00\xff"):
        resp = client.request(
            "GET",
            REVOCATIONS_PATH,
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"
    assert _audit_count(db_session) == events_before


# --- Pagination -----------------------------------------------------------------


def test_pages_resume_without_duplication_or_omission(client):
    _, _, revocations = _world(client)
    all_items, pages, count = _walk_pages(client, limit="2")
    assert count == 4
    assert [len(page) for page in pages] == [2, 2]
    assert [item["id"] for item in all_items] == [r["id"] for r in revocations]

    first = _list(client, limit="2").json()
    assert first["next_cursor"] is not None
    assert first["next_cursor"].startswith("ar1.")
    second = _list(client, limit="2", cursor=first["next_cursor"]).json()
    assert second["next_cursor"] is None


def test_count_covers_every_filtered_page(client):
    att_one, _, _ = _world(client)
    all_items, _pages, count = _walk_pages(
        client, attestation_id=att_one["id"], limit="1"
    )
    assert count == 2
    assert len(all_items) == 2
    assert {item["attestation_id"] for item in all_items} == {att_one["id"]}


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
    tail = pagination.encode_typed_cursor(
        app.state.attestation_revocations_cursor_secret,
        pagination.ATTESTATION_REVOCATIONS_CURSOR,
        {**claims, "offset": 4},
    )
    resp = _list(client, cursor=tail)
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
    att_one, att_two, _ = _world(client)
    cursor = _list(client, limit="2").json()["next_cursor"]
    assert cursor is not None
    # A different limit, a new filter, or a dropped filter all mismatch.
    for params in (
        {"limit": "3", "cursor": cursor},
        {"cursor": cursor},
        {"revoker_actor_id": "org-1", "limit": "2", "cursor": cursor},
        {"reason": REASON_A, "limit": "2", "cursor": cursor},
        {
            "attestation_id": att_one["id"],
            "limit": "2",
            "cursor": cursor,
        },
        {
            "from": "2026-01-01T00:00:00Z",
            "limit": "2",
            "cursor": cursor,
        },
        {
            "to": "2030-01-01T00:00:00Z",
            "limit": "2",
            "cursor": cursor,
        },
    ):
        resp = _list(client, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"

    # A cursor minted under one exact filter cannot resume another.
    att_one_cursor = _list(
        client, attestation_id=att_one["id"], limit="1"
    ).json()["next_cursor"]
    assert att_one_cursor is not None
    switched = _list(
        client, attestation_id=att_two["id"], limit="1", cursor=att_one_cursor
    )
    assert switched.status_code == 422
    revoker_cursor = _list(
        client, revoker_actor_id="org-1", limit="1"
    ).json()["next_cursor"]
    assert _list(
        client, revoker_actor_id="org-2", limit="1", cursor=revoker_cursor
    ).status_code == 422

    # A time-bound cursor is bound to the exact instant; equivalent UTC
    # notation resumes, a different second does not.
    time_cursor = _list(
        client, **{"from": "2000-01-01T00:00:00Z", "limit": "1"}
    ).json()["next_cursor"]
    assert time_cursor is not None
    same_bound = _list(
        client,
        **{
            "from": "2000-01-01T00:00:00+00:00",
            "limit": "1",
            "cursor": time_cursor,
        },
    )
    assert same_bound.status_code == 200
    changed_bound = _list(
        client,
        **{
            "from": "2000-01-01T00:00:01Z",
            "limit": "1",
            "cursor": time_cursor,
        },
    )
    assert changed_bound.status_code == 422


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
        app.state.actors_cursor_secret,
        pagination.ACTORS_CURSOR,
        {"actor_id": None, "name": None, "actor_type": None, "limit": 50,
         "offset": 1},
    )
    resp = _list(client, cursor=foreign)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    audit_cursor = pagination.encode_typed_cursor(
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
    assert _list(client, cursor=audit_cursor).status_code == 422


def test_cursor_with_wrong_claim_set_is_422(client, app):
    _world(client)
    # Correct family marker, HMAC, and base64, but a payload missing a bound
    # claim is rejected rather than trusted.
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
        hmac.new(
            app.state.attestation_revocations_cursor_secret,
            f"ar1.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    resp = _list(client, cursor=f"ar1.{payload}.{sig}")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_secret_rotates_on_restart_invalidating_old_cursors(
    tmp_db_url, file_client
):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    _world(file_client)
    cursor = _list(file_client, limit="1").json()["next_cursor"]
    assert cursor is not None

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as restarted_client:
        resp = _list(restarted_client, limit="1", cursor=cursor)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


# --- Method boundary ------------------------------------------------------------


def test_put_patch_and_delete_are_405_method_not_allowed(client):
    _world(client)
    for method in (client.put, client.patch, client.delete):
        resp = method(REVOCATIONS_PATH)
        assert resp.status_code == 405
        assert resp.json()["error"]["code"] == "method_not_allowed"


def test_post_remains_the_revocation_creation_route(client):
    # The new read entry does not shadow or alter revocation creation.
    create_actor(client, actor_id="org-1")
    claim = _create_claim(client, _create_content(client)["id"])
    att = _attest(client, claim["id"])
    payload = {
        "attestation_id": att["id"],
        "revoker_actor_id": "org-1",
        "reason": REASON_A,
    }
    first = client.post(REVOCATIONS_PATH, json=payload)
    assert first.status_code == 201
    repeat = client.post(REVOCATIONS_PATH, json=payload)
    assert repeat.status_code == 200
    assert repeat.json() == first.json()


# --- Read-only guarantee ---------------------------------------------------------


def test_queries_and_failures_write_nothing(client, db_session):
    _, _, revocations = _world(client)
    events_before = _audit_count(db_session)
    revocations_before = db_session.scalar(
        select(func.count()).select_from(AttestationRevocation)
    )

    assert _list(client).status_code == 200
    assert _list(client, revoker_actor_id="ghost").status_code == 200
    assert _list(client, reason="nobody said this").status_code == 200
    assert (
        _list(
            client, limit="2", **{"from": "2026-01-01T00:00:00Z"}
        ).status_code
        == 200
    )
    all_items, _pages, count = _walk_pages(client, limit="1")
    assert count == len(revocations) == 4
    assert len(all_items) == 4

    # Failures are read-only as well.
    assert _list(client, limit="0").status_code == 422
    assert _list(client, attestation_id=" ").status_code == 422
    assert _list(client, **{"from": "not-a-time"}).status_code == 422
    assert _list(client, cursor="bad").status_code == 422
    assert (
        client.request("GET", REVOCATIONS_PATH, content=b"{}").status_code == 422
    )

    assert (
        db_session.scalar(select(func.count()).select_from(AttestationRevocation))
        == revocations_before
    )
    assert _audit_count(db_session) == events_before
