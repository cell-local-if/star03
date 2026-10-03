"""Tests for POST /v1/contents and GET endpoints."""

from __future__ import annotations

import hashlib
from datetime import datetime

from sqlalchemy import select

from provenance.models import AuditEvent, Content
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    DIGEST_C,
    content_payload,
    create_actor,
)

# sha512 of fixed, tiny inputs: stable and offline.
DIGEST_512_A = hashlib.sha512(b"content-512-a").hexdigest()
DIGEST_512_B = hashlib.sha512(b"content-512-b").hexdigest()


def _create_content(client, **overrides):
    resp = client.post("/v1/contents", json=content_payload(**overrides))
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_create_content_returns_full_public_fields(client):
    create_actor(client)
    body = _create_content(client, title="A picture")
    assert set(body) == {
        "id",
        "digest_algorithm",
        "digest_hex",
        "media_type",
        "title",
        "actor_id",
        "created_at",
    }
    assert body["digest_algorithm"] == "sha256"
    assert body["digest_hex"] == DIGEST_A
    assert body["media_type"] == "image/png"
    assert body["title"] == "A picture"
    assert body["actor_id"] == "org-1"
    assert body["id"].startswith("cnt_")
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.tzinfo is not None
    assert created_at.utcoffset().total_seconds() == 0
    assert body["created_at"].endswith(("Z", "+00:00"))


def test_title_is_optional_and_can_be_null(client):
    create_actor(client)
    body = _create_content(client)
    assert body["title"] is None


def test_content_id_is_deterministic_and_stable(client):
    create_actor(client)
    first = _create_content(client)
    # A different actor, same digest: identity still resolves to the same id.
    create_actor(client, actor_id="org-2", name="Other", type="person")
    resp = client.post(
        "/v1/contents", json=content_payload(actor_id="org-2")
    )
    assert resp.status_code == 200
    assert resp.json()["id"] == first["id"]


def test_duplicate_content_returns_existing_resource_not_new_id(client):
    create_actor(client)
    first = _create_content(client, media_type="image/png", title="Original")

    # Repeat with conflicting metadata: must NOT create a new identity.
    repeat_payload = content_payload(
        media_type="image/jpeg", title="Different title"
    )
    resp = client.post("/v1/contents", json=repeat_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == first["id"]
    # The stored resource retains the first-submission metadata.
    assert body["media_type"] == "image/png"
    assert body["title"] == "Original"
    assert body["actor_id"] == "org-1"


def test_uppercase_hex_digest_is_rejected(client):
    # The digest spelling is already final: uppercase hex is a 422, never
    # lowercased into an idempotent repeat.
    create_actor(client)
    _create_content(client)
    resp = client.post(
        "/v1/contents", json=content_payload(digest=DIGEST_A.upper())
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_get_content_returns_full_fields(client):
    create_actor(client)
    created = _create_content(client, title="T")
    resp = client.get(f"/v1/contents/{created['id']}")
    assert resp.status_code == 200
    assert resp.json() == created


def test_get_unknown_content_is_distinct_not_found(client):
    resp = client.get("/v1/contents/cnt_does_not_exist")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "content_not_found"
    assert error["details"]["content_id"] == "cnt_does_not_exist"


def test_unknown_actor_on_create_is_distinct_error(client):
    resp = client.post("/v1/contents", json=content_payload(actor_id="ghost"))
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "unknown_actor"
    assert error["details"]["actor_id"] == "ghost"


def test_reject_unsupported_algorithm(client):
    create_actor(client)
    # Unknown names, wrong case, and padded spellings are all 422: only the
    # exact lowercase names sha256 and sha512 are accepted.
    for algorithm in ("md5", "sha1", "sha-256", "SHA256", "Sha512", " sha256", "sha512 "):
        resp = client.post(
            "/v1/contents", json=content_payload(algorithm=algorithm)
        )
        assert resp.status_code == 422, repr(algorithm)
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_short_digest(client):
    create_actor(client)
    resp = client.post(
        "/v1/contents", json=content_payload(digest="a" * 63)
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_reject_long_digest(client):
    create_actor(client)
    resp = client.post(
        "/v1/contents", json=content_payload(digest="a" * 65)
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_reject_non_hex_digest(client):
    create_actor(client)
    bad = DIGEST_A[:-1] + "z"
    resp = client.post("/v1/contents", json=content_payload(digest=bad))
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_reject_empty_media_type(client):
    create_actor(client)
    resp = client.post(
        "/v1/contents", json=content_payload(media_type="   ")
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_reject_empty_actor_id_field(client):
    create_actor(client)
    resp = client.post("/v1/contents", json=content_payload(actor_id="  "))
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_reject_missing_required_fields(client):
    resp = client.post("/v1/contents", json={})
    assert resp.status_code == 422
    issue_fields = {
        ".".join(part for part in issue["loc"] if part != "body")
        for issue in resp.json()["error"]["details"]["issues"]
    }
    assert {
        "digest_algorithm",
        "digest_hex",
        "media_type",
        "actor_id",
    }.issubset(issue_fields)


def test_reject_malformed_json_body(client):
    create_actor(client)
    resp = client.post(
        "/v1/contents",
        content="{not valid json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_list_returns_stable_creation_order(client):
    create_actor(client)
    ids = [_create_content(client, digest=d)["id"] for d in (DIGEST_A, DIGEST_B, DIGEST_C)]
    resp = client.get("/v1/contents")
    assert resp.status_code == 200
    listed = resp.json()
    assert listed["count"] == 3
    assert [item["id"] for item in listed["items"]] == ids


def test_list_filter_by_actor(client):
    create_actor(client)
    create_actor(client, actor_id="org-2", name="Other", type="person")
    a1 = _create_content(client, digest=DIGEST_A, actor_id="org-1")
    a2 = _create_content(client, digest=DIGEST_B, actor_id="org-2")
    a3 = _create_content(client, digest=DIGEST_C, actor_id="org-1")

    resp = client.get("/v1/contents", params={"actor_id": "org-1"})
    body = resp.json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [a1["id"], a3["id"]]
    # Filtering preserves the global stable creation order.
    assert all(item["actor_id"] == "org-1" for item in body["items"])

    resp2 = client.get("/v1/contents", params={"actor_id": "org-2"})
    assert [item["id"] for item in resp2.json()["items"]] == [a2["id"]]


def test_list_filter_unknown_actor_returns_empty_list(client):
    # An unknown filter value is an empty collection, not a 404.
    resp = client.get("/v1/contents", params={"actor_id": "ghost"})
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


# --- SHA-512 content identities -------------------------------------------------


def _create_sha512_content(client, **overrides):
    overrides.setdefault("algorithm", "sha512")
    overrides.setdefault("digest", DIGEST_512_A)
    resp = client.post("/v1/contents", json=content_payload(**overrides))
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_create_sha512_content_returns_full_public_fields(client):
    create_actor(client)
    body = _create_sha512_content(client, title="A document")
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
    assert body["title"] == "A document"
    assert body["actor_id"] == "org-1"
    assert body["id"].startswith("cnt_")
    created_at = datetime.fromisoformat(body["created_at"])
    assert created_at.tzinfo is not None
    assert created_at.utcoffset().total_seconds() == 0


def test_sha512_content_id_is_deterministic_and_stable(client):
    create_actor(client)
    first = _create_sha512_content(client)
    create_actor(client, actor_id="org-2", name="Other", type="person")
    resp = client.post(
        "/v1/contents",
        json=content_payload(
            algorithm="sha512", digest=DIGEST_512_A, actor_id="org-2"
        ),
    )
    assert resp.status_code == 200
    assert resp.json()["id"] == first["id"]


def test_sha512_and_sha256_identities_are_disjoint(client):
    # The algorithm is part of the identity: the same actor can register a
    # sha256 and a sha512 identity, and each derives its own stable id.
    create_actor(client)
    sha256_content = _create_content(client, digest=DIGEST_A)
    sha512_content = _create_sha512_content(client)
    assert sha256_content["id"] != sha512_content["id"]
    assert sha256_content["digest_algorithm"] == "sha256"
    assert sha512_content["digest_algorithm"] == "sha512"


def test_duplicate_sha512_content_returns_existing_resource(client, db_session):
    create_actor(client)
    first = _create_sha512_content(client, media_type="image/png", title="Original")
    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())
    rows_before = len(db_session.execute(select(Content)).scalars().all())

    # Repeat with conflicting metadata: must NOT create a new identity, row,
    # or audit event, and must not overwrite the first-submission metadata.
    resp = client.post(
        "/v1/contents",
        json=content_payload(
            algorithm="sha512",
            digest=DIGEST_512_A,
            media_type="image/jpeg",
            title="Different title",
        ),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == first["id"]
    assert body["media_type"] == "image/png"
    assert body["title"] == "Original"
    assert len(db_session.execute(select(AuditEvent)).scalars().all()) == events_before
    assert len(db_session.execute(select(Content)).scalars().all()) == rows_before


def test_get_sha512_content_returns_registered_fields(client):
    create_actor(client)
    created = _create_sha512_content(client, title="T")
    resp = client.get(f"/v1/contents/{created['id']}")
    assert resp.status_code == 200
    assert resp.json() == created
    assert resp.json()["digest_algorithm"] == "sha512"
    assert resp.json()["digest_hex"] == DIGEST_512_A


def test_unknown_actor_on_sha512_create_is_distinct_error(client):
    resp = client.post(
        "/v1/contents",
        json=content_payload(
            algorithm="sha512", digest=DIGEST_512_A, actor_id="ghost"
        ),
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "unknown_actor"
    assert error["details"]["actor_id"] == "ghost"


def test_reject_digest_length_mismatched_to_algorithm(client):
    create_actor(client)
    # sha256 accepts exactly 64 hex chars; sha512 exactly 128.
    for algorithm, digest in (
        ("sha256", DIGEST_512_A),
        ("sha512", DIGEST_A),
        ("sha512", "a" * 127),
        ("sha512", "a" * 129),
        ("sha256", "a" * 63),
        ("sha256", "a" * 65),
    ):
        resp = client.post(
            "/v1/contents",
            json=content_payload(algorithm=algorithm, digest=digest),
        )
        assert resp.status_code == 422, (algorithm, len(digest))
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_non_hex_and_uppercase_sha512_digest(client):
    create_actor(client)
    for bad in (
        DIGEST_512_A[:-1] + "z",
        DIGEST_512_A.upper(),
        " " + DIGEST_512_A,
        DIGEST_512_A + " ",
    ):
        resp = client.post(
            "/v1/contents",
            json=content_payload(algorithm="sha512", digest=bad),
        )
        assert resp.status_code == 422, repr(bad[:24])
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_extra_field_on_create(client):
    create_actor(client)
    payload = content_payload()
    payload["content_bytes"] = "not-allowed"
    resp = client.post("/v1/contents", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_sha512_requests_write_nothing(client, db_session):
    create_actor(client)
    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())
    ids_before = {row.id for row in db_session.execute(select(Content)).scalars()}
    for payload in (
        content_payload(algorithm="sha512", digest=DIGEST_A),
        content_payload(algorithm="sha512", digest=DIGEST_512_A.upper()),
        content_payload(algorithm="SHA512", digest=DIGEST_512_A),
        content_payload(algorithm="md5", digest=DIGEST_512_A),
    ):
        resp = client.post("/v1/contents", json=payload)
        assert resp.status_code == 422
    assert len(db_session.execute(select(AuditEvent)).scalars().all()) == events_before
    assert {row.id for row in db_session.execute(select(Content)).scalars()} == ids_before
