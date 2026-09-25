"""Tests for read-only content retrieval.

Covers ``GET /v1/contents``: the empty-body contract, the exact
``items``/``count``/``next_cursor`` member order in compact UTF-8 JSON
terminated by one newline, the optional exact ``actor_id``/
``digest_algorithm``/``digest_hex``/``media_type`` filters (an unknown or
nonexistent value is an empty collection, never a 404), the strict
64-lowercase-hex ``digest_hex`` rule, stable creation ordering
(``created_at`` with the persistent insertion-order tiebreaker that survives
an app restart), pure-decimal ``limit`` validation, the opaque HMAC-signed
``ct1`` cursor family (binding every effective filter and the limit,
resuming without duplication or omission, a null cursor on the final page,
and an empty page with the original count at or past the tail), every 422
validation boundary, and the strictly read-only guarantee (no content,
resource, or audit writes on success, empty results, repeated reads, or
failures).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from sqlalchemy import select

from provenance import pagination
from provenance.models import AuditEvent, Content
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    DIGEST_C,
    content_payload,
    create_actor,
)

CONTENTS_PATH = "/v1/contents"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


DIGEST_D = _digest("content-d")


# --- Fixture-style setup ------------------------------------------------------


def _world(client):
    """Register four contents under two actors, in stable creation order."""
    create_actor(client)
    create_actor(client, actor_id="org-2", name="Other", type="person")
    one = client.post(
        "/v1/contents",
        json=content_payload(digest=DIGEST_A, media_type="image/png"),
    ).json()
    two = client.post(
        "/v1/contents",
        json=content_payload(
            actor_id="org-2", digest=DIGEST_B, media_type="image/jpeg"
        ),
    ).json()
    three = client.post(
        "/v1/contents",
        json=content_payload(digest=DIGEST_C, media_type="image/png"),
    ).json()
    four = client.post(
        "/v1/contents",
        json=content_payload(
            actor_id="org-2", digest=DIGEST_D, media_type="image/png"
        ),
    ).json()
    return [one, two, three, four]


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


def _content_ids(session):
    return {row.id for row in session.execute(select(Content.id)).all()}


# --- Empty collection and response shape --------------------------------------


def test_empty_registry_is_an_empty_collection(client):
    resp = client.get(CONTENTS_PATH)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_response_is_compact_json_with_ordered_members_and_one_newline(client):
    _world(client)
    resp = client.get(CONTENTS_PATH)
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


def test_items_carry_exactly_the_content_public_view(client):
    contents = _world(client)
    resp = client.get(CONTENTS_PATH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 4
    assert body["next_cursor"] is None
    # Stable creation order, and each item is exactly the public view.
    assert [item["id"] for item in body["items"]] == [c["id"] for c in contents]
    for item, content in zip(body["items"], contents):
        assert list(item) == [
            "id",
            "digest_algorithm",
            "digest_hex",
            "media_type",
            "title",
            "actor_id",
            "created_at",
        ]
        assert item == content
        created_at = datetime.fromisoformat(item["created_at"])
        assert created_at.tzinfo is not None
        assert created_at.utcoffset().total_seconds() == 0
        assert item["created_at"].endswith(("Z", "+00:00"))
    # No raw material, payload, or ordering column can ever be echoed.
    rendered = resp.content.decode()
    for forbidden in ("seq", "payload", "raw", "private", "signature"):
        assert forbidden not in rendered


# --- Stable ordering -----------------------------------------------------------


def test_items_follow_stable_creation_order(client):
    contents = _world(client)
    body = client.get(CONTENTS_PATH).json()
    assert [item["id"] for item in body["items"]] == [c["id"] for c in contents]
    created = [datetime.fromisoformat(item["created_at"]) for item in body["items"]]
    assert created == sorted(created)


def test_same_timestamp_ties_break_by_persistence_order(client, db_session):
    # Force every content onto one instant: the persistent insertion order
    # (seq) must still yield a stable, restart-durable order.
    contents = _world(client)
    tie = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    for row in db_session.execute(select(Content)).scalars():
        row.created_at = tie
    db_session.commit()

    body = client.get(CONTENTS_PATH).json()
    assert [item["id"] for item in body["items"]] == [c["id"] for c in contents]


def test_ordering_is_identical_across_an_app_restart(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    contents = _world(file_client)
    first = file_client.get(CONTENTS_PATH)
    assert first.status_code == 200

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as restarted_client:
        second = restarted_client.get(CONTENTS_PATH)
        assert second.status_code == 200
        assert second.json() == first.json()
        assert [item["id"] for item in second.json()["items"]] == [
            c["id"] for c in contents
        ]


# --- Filtering -----------------------------------------------------------------


def test_actor_filter_is_an_exact_match(client):
    contents = _world(client)
    resp = client.get(CONTENTS_PATH, params={"actor_id": "org-1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [
        contents[0]["id"],
        contents[2]["id"],
    ]
    assert all(item["actor_id"] == "org-1" for item in body["items"])


def test_algorithm_digest_and_media_type_filters_match_exactly(client):
    contents = _world(client)
    by_algorithm = client.get(
        CONTENTS_PATH, params={"digest_algorithm": "sha256"}
    ).json()
    assert [item["id"] for item in by_algorithm["items"]] == [
        c["id"] for c in contents
    ]

    by_digest = client.get(
        CONTENTS_PATH, params={"digest_hex": DIGEST_B}
    ).json()
    assert by_digest["count"] == 1
    assert [item["id"] for item in by_digest["items"]] == [contents[1]["id"]]

    by_media = client.get(
        CONTENTS_PATH, params={"media_type": "image/jpeg"}
    ).json()
    assert by_media["count"] == 1
    assert [item["id"] for item in by_media["items"]] == [contents[1]["id"]]


def test_filters_combine_as_logical_and(client):
    _world(client)
    resp = client.get(
        CONTENTS_PATH,
        params={"actor_id": "org-2", "media_type": "image/png"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["items"][0]["digest_hex"] == DIGEST_D

    full = client.get(
        CONTENTS_PATH,
        params={
            "actor_id": "org-1",
            "digest_algorithm": "sha256",
            "digest_hex": DIGEST_C,
            "media_type": "image/png",
        },
    ).json()
    assert full["count"] == 1
    assert full["items"][0]["digest_hex"] == DIGEST_C

    # Every filter valid, but their intersection is empty.
    miss = client.get(
        CONTENTS_PATH,
        params={"actor_id": "org-1", "media_type": "image/jpeg"},
    ).json()
    assert miss == {"items": [], "count": 0, "next_cursor": None}


def test_string_filters_are_case_and_whitespace_sensitive(client):
    _world(client)
    for params in (
        {"actor_id": "Org-1"},
        {"actor_id": "org-1 "},
        {"actor_id": " org-1"},
        {"digest_algorithm": "SHA256"},
        {"digest_algorithm": "sha256 "},
        {"media_type": "Image/PNG"},
        {"media_type": "image/png "},
    ):
        resp = client.get(CONTENTS_PATH, params=params)
        assert resp.status_code == 200, params
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_unknown_or_nonexistent_filter_value_is_an_empty_collection(client):
    _world(client)
    for params in (
        {"actor_id": "ghost"},
        {"digest_algorithm": "sha512"},
        {"digest_hex": "a" * 64},
        {"media_type": "application/x-unknown"},
        {
            "actor_id": "ghost",
            "digest_algorithm": "sha256",
            "digest_hex": DIGEST_A,
            "media_type": "image/png",
        },
    ):
        resp = client.get(CONTENTS_PATH, params=params)
        assert resp.status_code == 200, params
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_blank_filters_are_422(client):
    _world(client)
    for field in (
        "actor_id",
        "digest_algorithm",
        "digest_hex",
        "media_type",
    ):
        for blank in ("", "   ", "\t"):
            resp = client.get(CONTENTS_PATH, params={field: blank})
            assert resp.status_code == 422, (field, repr(blank))
            assert resp.json()["error"]["code"] == "validation_error"


def test_digest_filter_requires_64_lowercase_hex(client):
    _world(client)
    for bad in (
        DIGEST_A.upper(),
        "A" * 64,
        "a" * 63,
        "a" * 65,
        DIGEST_A[:-1] + "z",
        " " + DIGEST_A,
        DIGEST_A + " ",
        "g" * 64,
    ):
        resp = client.get(CONTENTS_PATH, params={"digest_hex": bad})
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


# --- limit validation -----------------------------------------------------------


def test_limit_boundaries_one_and_one_hundred_are_accepted(client):
    _world(client)
    assert client.get(CONTENTS_PATH, params={"limit": "1"}).status_code == 200
    assert client.get(CONTENTS_PATH, params={"limit": "100"}).status_code == 200


def test_limit_must_be_a_pure_decimal_integer_in_range(client):
    _world(client)
    for bad in ("0", "101", "-1", "5.0", "5e0", " 5", "5 ", "five", "+5", ""):
        resp = client.get(CONTENTS_PATH, params={"limit": bad})
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_default_limit_is_fifty(client):
    create_actor(client)
    for index in range(51):
        resp = client.post(
            "/v1/contents",
            json=content_payload(digest=_digest(f"bulk-{index}")),
        )
        assert resp.status_code == 201, resp.text
    resp = client.get(CONTENTS_PATH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 51
    assert len(body["items"]) == 50
    assert body["next_cursor"] is not None


# --- Parameter strictness -------------------------------------------------------


def test_repeated_and_undeclared_parameters_are_422(client):
    _world(client)
    assert (
        client.get(
            CONTENTS_PATH,
            params=[("actor_id", "org-1"), ("actor_id", "org-2")],
        ).status_code
        == 422
    )
    assert (
        client.get(
            CONTENTS_PATH, params=[("limit", "1"), ("limit", "2")]
        ).status_code
        == 422
    )
    assert (
        client.get(
            CONTENTS_PATH,
            params=[("digest_hex", DIGEST_A), ("digest_hex", DIGEST_B)],
        ).status_code
        == 422
    )
    for params in (
        {"algorithm": "sha256"},
        {"actor": "org-1"},
        {"offset": "1"},
        {"q": "x"},
    ):
        resp = client.get(CONTENTS_PATH, params=params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_empty_body_is_422_rejected_before_any_read(client, db_session):
    _world(client)
    events_before = _audit_count(db_session)
    for body in (b"{}", b" ", b"{not valid json", b"[]", b"\n"):
        resp = client.request(
            "GET",
            CONTENTS_PATH,
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"
    assert _audit_count(db_session) == events_before


# --- Pagination -----------------------------------------------------------------


def test_pages_resume_without_duplication_or_omission(client):
    contents = _world(client)
    first = client.get(CONTENTS_PATH, params={"limit": "2"})
    assert first.status_code == 200
    page_one = first.json()
    assert page_one["count"] == 4
    assert [item["id"] for item in page_one["items"]] == [
        contents[0]["id"],
        contents[1]["id"],
    ]
    cursor = page_one["next_cursor"]
    assert cursor is not None
    assert cursor.startswith("ct1.")

    second = client.get(
        CONTENTS_PATH, params={"limit": "2", "cursor": cursor}
    )
    assert second.status_code == 200
    page_two = second.json()
    assert page_two["count"] == 4
    assert [item["id"] for item in page_two["items"]] == [
        contents[2]["id"],
        contents[3]["id"],
    ]
    assert page_two["next_cursor"] is None

    seen = [item["id"] for item in page_one["items"] + page_two["items"]]
    assert seen == [c["id"] for c in contents]


def test_count_covers_every_page_including_a_filtered_one(client):
    _world(client)
    page = client.get(
        CONTENTS_PATH,
        params={"media_type": "image/png", "limit": "1"},
    ).json()
    assert page["count"] == 3
    assert len(page["items"]) == 1
    seen = [page["items"][0]["id"]]
    cursor = page["next_cursor"]
    while cursor is not None:
        nxt = client.get(
            CONTENTS_PATH,
            params={"media_type": "image/png", "limit": "1", "cursor": cursor},
        ).json()
        assert nxt["count"] == 3
        seen.extend(item["id"] for item in nxt["items"])
        cursor = nxt["next_cursor"]
    assert len(seen) == 3
    assert len(set(seen)) == 3


def test_replayed_cursor_returns_the_same_page(client):
    _world(client)
    first = client.get(CONTENTS_PATH, params={"limit": "1"}).json()
    cursor = first["next_cursor"]
    assert cursor is not None
    page = client.get(CONTENTS_PATH, params={"limit": "1", "cursor": cursor})
    replay = client.get(
        CONTENTS_PATH, params={"limit": "1", "cursor": cursor}
    )
    assert page.status_code == 200
    assert replay.status_code == 200
    assert replay.content == page.content


def test_cursor_at_or_past_the_tail_returns_empty_page_with_count(client, app):
    _world(client)
    cursor = pagination.encode_typed_cursor(
        app.state.contents_cursor_secret,
        pagination.CONTENTS_CURSOR,
        {
            "actor_id": None,
            "digest_algorithm": None,
            "digest_hex": None,
            "media_type": None,
            "limit": 50,
            "offset": 4,
        },
    )
    resp = client.get(CONTENTS_PATH, params={"cursor": cursor})
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 4, "next_cursor": None}

    past = pagination.encode_typed_cursor(
        app.state.contents_cursor_secret,
        pagination.CONTENTS_CURSOR,
        {
            "actor_id": None,
            "digest_algorithm": None,
            "digest_hex": None,
            "media_type": None,
            "limit": 50,
            "offset": 99,
        },
    )
    resp = client.get(CONTENTS_PATH, params={"cursor": past})
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 4, "next_cursor": None}


def test_cursor_binds_every_effective_filter_and_limit(client):
    _world(client)
    cursor = client.get(CONTENTS_PATH, params={"limit": "2"}).json()["next_cursor"]
    assert cursor is not None
    # A different limit, a new filter, or a dropped limit all mismatch.
    for params in (
        {"limit": "3", "cursor": cursor},
        {"actor_id": "org-1", "limit": "2", "cursor": cursor},
        {"cursor": cursor},
    ):
        resp = client.get(CONTENTS_PATH, params=params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"

    # A cursor minted under one filter cannot resume a different filter.
    mid = client.get(
        CONTENTS_PATH, params={"media_type": "image/png", "limit": "1"}
    ).json()["next_cursor"]
    assert mid is not None
    changed_media = client.get(
        CONTENTS_PATH,
        params={"media_type": "image/jpeg", "limit": "1", "cursor": mid},
    )
    assert changed_media.status_code == 422
    changed_actor = client.get(
        CONTENTS_PATH,
        params={"actor_id": "org-1", "limit": "1", "cursor": mid},
    )
    assert changed_actor.status_code == 422
    changed_digest = client.get(
        CONTENTS_PATH,
        params={"digest_hex": DIGEST_A, "limit": "1", "cursor": mid},
    )
    assert changed_digest.status_code == 422


def test_blank_malformed_and_tampered_cursors_are_422(client):
    _world(client)
    valid = client.get(CONTENTS_PATH, params={"limit": "1"}).json()["next_cursor"]
    assert valid is not None
    tampered = valid[:-1] + ("A" if valid[-1] != "A" else "B")
    for bad in ("", "   ", "not-a-cursor", "ct1", "ct1.abc", tampered):
        resp = client.get(CONTENTS_PATH, params={"cursor": bad})
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_foreign_family_cursor_is_422(client, app):
    _world(client)
    # A well-formed cursor minted by another endpoint family never resumes
    # this retrieval.
    foreign = pagination.encode_typed_cursor(
        app.state.actors_cursor_secret,
        pagination.ACTORS_CURSOR,
        {
            "actor_id": None,
            "name": None,
            "actor_type": None,
            "limit": 50,
            "offset": 1,
        },
    )
    resp = client.get(CONTENTS_PATH, params={"cursor": foreign})
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
    resp = client.get(CONTENTS_PATH, params={"cursor": claims_cursor})
    assert resp.status_code == 422


def test_cursor_with_wrong_claim_set_is_422(client, app):
    _world(client)
    # Correct family marker, HMAC, and base64, but a foreign claim payload.
    token = pagination.encode_typed_cursor(
        app.state.contents_cursor_secret,
        pagination.CONTENTS_CURSOR,
        {
            "actor_id": None,
            "digest_algorithm": None,
            "digest_hex": None,
            "media_type": None,
            "limit": 50,
            "offset": 1,
        },
    )
    assert client.get(CONTENTS_PATH, params={"cursor": token}).status_code == 200
    import base64
    import hashlib
    import hmac
    import json

    payload = base64.urlsafe_b64encode(
        json.dumps(
            {"actor_id": None, "limit": 50, "offset": 1},
            separators=(",", ":"),
        ).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(
        hmac.new(
            app.state.contents_cursor_secret,
            f"ct1.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    resp = client.get(CONTENTS_PATH, params={"cursor": f"ct1.{payload}.{sig}"})
    assert resp.status_code == 422


# --- Method boundary ------------------------------------------------------------


def test_put_patch_and_delete_are_405_method_not_allowed(client):
    _world(client)
    for method in (client.put, client.patch, client.delete):
        resp = method(CONTENTS_PATH)
        assert resp.status_code == 405
        assert resp.json()["error"]["code"] == "method_not_allowed"


def test_post_remains_the_content_registration_route(client):
    # The retrieval entry does not shadow or alter registration semantics:
    # first submission is 201, an idempotent repeat returns 200 unchanged.
    create_actor(client)
    first = client.post("/v1/contents", json=content_payload())
    assert first.status_code == 201
    repeat = client.post("/v1/contents", json=content_payload())
    assert repeat.status_code == 200
    assert repeat.json()["id"] == first.json()["id"]


# --- Read-only guarantee ---------------------------------------------------------


def test_queries_and_failures_write_nothing(client, db_session):
    contents = _world(client)
    events_before = _audit_count(db_session)
    ids_before = _content_ids(db_session)

    assert client.get(CONTENTS_PATH).status_code == 200
    assert (
        client.get(CONTENTS_PATH, params={"actor_id": "ghost"}).status_code
        == 200
    )
    assert (
        client.get(
            CONTENTS_PATH,
            params={"media_type": "image/png", "limit": "2"},
        ).status_code
        == 200
    )
    # Walk every page.
    cursor = None
    for _ in range(10):
        params = {"limit": "2"}
        if cursor is not None:
            params["cursor"] = cursor
        body = client.get(CONTENTS_PATH, params=params).json()
        cursor = body["next_cursor"]
        if cursor is None:
            break
    assert cursor is None

    assert client.get(CONTENTS_PATH, params={"limit": "0"}).status_code == 422
    assert (
        client.get(CONTENTS_PATH, params={"actor_id": " "}).status_code == 422
    )
    assert (
        client.get(
            CONTENTS_PATH, params={"digest_hex": DIGEST_A.upper()}
        ).status_code
        == 422
    )
    assert client.get(CONTENTS_PATH, params={"cursor": "bad"}).status_code == 422
    assert (
        client.request("GET", CONTENTS_PATH, content=b"{}").status_code == 422
    )

    assert _content_ids(db_session) == ids_before == {c["id"] for c in contents}
    assert _audit_count(db_session) == events_before
