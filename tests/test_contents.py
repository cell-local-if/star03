"""Tests for POST /v1/contents and GET endpoints."""

from __future__ import annotations

from datetime import datetime

from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    DIGEST_C,
    content_payload,
    create_actor,
)


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


def test_duplicate_content_with_uppercase_hex_is_idempotent(client):
    create_actor(client)
    first = _create_content(client)
    resp = client.post(
        "/v1/contents", json=content_payload(digest=DIGEST_A.upper())
    )
    assert resp.status_code == 200
    assert resp.json()["id"] == first["id"]


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


def test_reject_non_sha256_algorithm(client):
    create_actor(client)
    resp = client.post(
        "/v1/contents", json=content_payload(algorithm="sha512")
    )
    assert resp.status_code == 422
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
