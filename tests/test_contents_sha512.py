"""Tests for SHA-512 content identities alongside SHA-256.

Covers ``POST /v1/contents`` with ``digest_algorithm=sha512`` (exact
lowercase name, exactly 128 lowercase hex characters), the stable
algorithm+digest-derived identity, idempotent repeat submission across both
algorithms, the detail read, and the ``GET /v1/contents`` digest filter
pairs (a bare ``digest_hex`` keeps its SHA-256 interpretation; an explicit
algorithm binds the digest length; every other combination is a 422).
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from sqlalchemy import select

from provenance import ids
from provenance.models import AuditEvent, Content
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    DIGEST_512_A,
    DIGEST_512_B,
    content_payload,
    create_actor,
)

CONTENTS_PATH = "/v1/contents"


def _create_content(client, **overrides):
    resp = client.post(CONTENTS_PATH, json=content_payload(**overrides))
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_sha512(client, digest=DIGEST_512_A, **overrides):
    return _create_content(client, algorithm="sha512", digest=digest, **overrides)


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


def _content_count(session):
    return len(session.execute(select(Content)).scalars().all())


# --- Registration and public view ----------------------------------------------


def test_register_sha512_content_returns_full_public_view(client):
    create_actor(client)
    body = _create_sha512(client, title="A sha512 picture")
    assert set(body) == {
        "id",
        "digest_algorithm",
        "digest_hex",
        "media_type",
        "title",
        "actor_id",
        "created_at",
    }
    assert body["digest_algorithm"] == "sha512"
    assert body["digest_hex"] == DIGEST_512_A
    assert len(body["digest_hex"]) == 128
    assert body["title"] == "A sha512 picture"
    assert body["actor_id"] == "org-1"
    assert body["id"].startswith("cnt_")
    assert body["id"] == ids.content_id("sha512", DIGEST_512_A)
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.tzinfo is not None
    assert created_at.utcoffset().total_seconds() == 0


def test_algorithms_form_distinct_identities(client):
    create_actor(client)
    sha256_content = _create_content(client, digest=DIGEST_A)
    sha512_content = _create_sha512(client)
    assert sha256_content["id"] != sha512_content["id"]
    assert sha256_content["id"] == ids.content_id("sha256", DIGEST_A)
    assert sha512_content["id"] == ids.content_id("sha512", DIGEST_512_A)


def test_get_sha512_content_returns_registered_fields(client):
    create_actor(client)
    created = _create_sha512(client, media_type="application/pdf", title="T")
    resp = client.get(f"{CONTENTS_PATH}/{created['id']}")
    assert resp.status_code == 200
    assert resp.json() == created


# --- Idempotent repeat submission ------------------------------------------------


def test_duplicate_sha512_registration_returns_existing(client, db_session):
    create_actor(client)
    create_actor(client, actor_id="org-2", name="Other", type="person")
    first = _create_sha512(client, media_type="image/png", title="Original")

    # A repeat with conflicting metadata creates nothing and overwrites
    # nothing: the first-submission record is returned unchanged.
    resp = client.post(
        CONTENTS_PATH,
        json=content_payload(
            algorithm="sha512",
            digest=DIGEST_512_A,
            media_type="image/jpeg",
            title="Different title",
            actor_id="org-2",
        ),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == first["id"]
    assert body["media_type"] == "image/png"
    assert body["title"] == "Original"
    assert body["actor_id"] == "org-1"
    assert _content_count(db_session) == 1
    # Exactly one content.created audit event exists (from the actor events
    # aside): the repeat added none.
    content_events = [
        event
        for event in db_session.execute(select(AuditEvent)).scalars()
        if event.event_type == "content.created"
    ]
    assert len(content_events) == 1


def test_sha256_idempotency_is_unaffected_by_sha512(client):
    create_actor(client)
    first = _create_content(client, digest=DIGEST_A)
    _create_sha512(client)
    resp = client.post(CONTENTS_PATH, json=content_payload(digest=DIGEST_A))
    assert resp.status_code == 200
    assert resp.json()["id"] == first["id"]


# --- Validation boundaries --------------------------------------------------------


def test_sha512_requires_exactly_128_lowercase_hex(client):
    create_actor(client)
    for bad in (
        DIGEST_512_A[:127],
        DIGEST_512_A + "a",
        DIGEST_512_A.upper(),
        DIGEST_512_A[:-1] + "z",
        " " + DIGEST_512_A,
        DIGEST_A,  # a 64-char digest is a SHA-256 spelling, not SHA-512
    ):
        resp = client.post(
            CONTENTS_PATH,
            json=content_payload(algorithm="sha512", digest=bad),
        )
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_sha256_rejects_128_char_digest(client):
    create_actor(client)
    resp = client.post(
        CONTENTS_PATH,
        json=content_payload(algorithm="sha256", digest=DIGEST_512_A),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_failed_validation_writes_nothing(client, db_session):
    create_actor(client)
    events_before = _audit_count(db_session)
    contents_before = _content_count(db_session)
    for payload in (
        content_payload(algorithm="sha512", digest=DIGEST_A),
        content_payload(algorithm="sha256", digest=DIGEST_512_A),
        content_payload(algorithm="SHA512", digest=DIGEST_512_A),
        content_payload(algorithm="sha512", digest=DIGEST_512_A.upper()),
    ):
        resp = client.post(CONTENTS_PATH, json=payload)
        assert resp.status_code == 422
    assert _audit_count(db_session) == events_before
    assert _content_count(db_session) == contents_before


# --- Search filters ---------------------------------------------------------------


def _mixed_world(client):
    """One sha256 and two sha512 contents, in creation order."""
    create_actor(client)
    one = _create_content(client, digest=DIGEST_A)
    two = _create_sha512(client, digest=DIGEST_512_A)
    three = _create_sha512(client, digest=DIGEST_512_B)
    four = _create_content(client, digest=DIGEST_B)
    return [one, two, three, four]


def test_digest_algorithm_filter_matches_each_algorithm(client):
    contents = _mixed_world(client)
    sha256 = client.get(CONTENTS_PATH, params={"digest_algorithm": "sha256"})
    assert [item["id"] for item in sha256.json()["items"]] == [
        contents[0]["id"],
        contents[3]["id"],
    ]
    assert sha256.json()["count"] == 2

    sha512 = client.get(CONTENTS_PATH, params={"digest_algorithm": "sha512"})
    assert [item["id"] for item in sha512.json()["items"]] == [
        contents[1]["id"],
        contents[2]["id"],
    ]
    assert sha512.json()["count"] == 2


def test_bare_digest_hex_keeps_sha256_interpretation(client):
    contents = _mixed_world(client)
    resp = client.get(CONTENTS_PATH, params={"digest_hex": DIGEST_B})
    assert resp.status_code == 200
    assert [item["id"] for item in resp.json()["items"]] == [contents[3]["id"]]
    # A 128-char digest without an explicit algorithm is a 422, not a match.
    resp = client.get(CONTENTS_PATH, params={"digest_hex": DIGEST_512_A})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_explicit_algorithm_binds_digest_length(client):
    contents = _mixed_world(client)
    resp = client.get(
        CONTENTS_PATH,
        params={"digest_algorithm": "sha512", "digest_hex": DIGEST_512_A},
    )
    assert resp.status_code == 200
    assert [item["id"] for item in resp.json()["items"]] == [contents[1]["id"]]

    resp = client.get(
        CONTENTS_PATH,
        params={"digest_algorithm": "sha256", "digest_hex": DIGEST_A},
    )
    assert resp.status_code == 200
    assert [item["id"] for item in resp.json()["items"]] == [contents[0]["id"]]


def test_mismatched_digest_pairs_are_422(client):
    _mixed_world(client)
    for params in (
        # SHA-256 spelled with a 128-char digest and vice versa.
        {"digest_algorithm": "sha256", "digest_hex": DIGEST_512_A},
        {"digest_algorithm": "sha512", "digest_hex": DIGEST_A},
        # An unrecognized algorithm spelling paired with any digest.
        {"digest_algorithm": "md5", "digest_hex": DIGEST_A},
        {"digest_algorithm": "SHA256", "digest_hex": DIGEST_A},
        {"digest_algorithm": "SHA512", "digest_hex": DIGEST_512_A},
        # Case/padding is never normalized on the digest either.
        {"digest_algorithm": "sha512", "digest_hex": DIGEST_512_A.upper()},
        {"digest_algorithm": "sha512", "digest_hex": " " + DIGEST_512_A},
    ):
        resp = client.get(CONTENTS_PATH, params=params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error", params


def test_digest_pair_filter_combines_as_logical_and(client):
    contents = _mixed_world(client)
    body = client.get(
        CONTENTS_PATH,
        params={
            "actor_id": "org-1",
            "digest_algorithm": "sha512",
            "digest_hex": DIGEST_512_B,
            "media_type": "image/png",
        },
    ).json()
    assert [item["id"] for item in body["items"]] == [contents[2]["id"]]
    assert body["count"] == 1

    # A valid pair whose digest belongs to another actor matches nothing.
    miss = client.get(
        CONTENTS_PATH,
        params={
            "actor_id": "ghost",
            "digest_algorithm": "sha512",
            "digest_hex": DIGEST_512_B,
        },
    ).json()
    assert miss == {"items": [], "count": 0, "next_cursor": None}


def test_sha512_filtered_pagination_resumes_with_cursor(client):
    create_actor(client)
    digests = [
        hashlib.sha512(f"bulk-{index}".encode()).hexdigest()
        for index in range(3)
    ]
    created = [
        _create_sha512(client, digest=digest, media_type="image/png")
        for digest in digests
    ]
    first = client.get(
        CONTENTS_PATH, params={"digest_algorithm": "sha512", "limit": "2"}
    )
    assert first.status_code == 200
    page_one = first.json()
    assert page_one["count"] == 3
    assert [item["id"] for item in page_one["items"]] == [
        created[0]["id"],
        created[1]["id"],
    ]
    cursor = page_one["next_cursor"]
    assert cursor is not None

    second = client.get(
        CONTENTS_PATH,
        params={"digest_algorithm": "sha512", "limit": "2", "cursor": cursor},
    )
    assert second.status_code == 200
    page_two = second.json()
    assert [item["id"] for item in page_two["items"]] == [created[2]["id"]]
    assert page_two["next_cursor"] is None

    # The cursor is bound to its conditions: another algorithm or digest
    # cannot resume it.
    for params in (
        {"digest_algorithm": "sha256", "limit": "2", "cursor": cursor},
        {"limit": "2", "cursor": cursor},
        {
            "digest_algorithm": "sha512",
            "digest_hex": digests[0],
            "limit": "2",
            "cursor": cursor,
        },
    ):
        resp = client.get(CONTENTS_PATH, params=params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error", params
