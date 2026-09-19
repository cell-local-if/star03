"""Content retrieval: detail, listing, filtering, and stable ordering."""

from __future__ import annotations

from conftest import DIGEST_A, DIGEST_B, DIGEST_C, create_content


def test_get_content_returns_full_public_fields(client, actor):
    created = create_content(client, title="Launch photo").json()
    response = client.get(f"/v1/contents/{created['content_id']}")
    assert response.status_code == 200
    assert response.json() == created


def test_get_unknown_content_returns_distinct_error(client):
    response = client.get("/v1/contents/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "content_not_found"


def test_list_contents_returns_stable_creation_order(client, actor):
    ids = [
        create_content(client, digest=d).json()["content_id"]
        for d in (DIGEST_A, DIGEST_B, DIGEST_C)
    ]
    response = client.get("/v1/contents")
    assert response.status_code == 200
    assert [c["content_id"] for c in response.json()] == ids


def test_list_contents_filters_by_actor(client, actor):
    client.post(
        "/v1/actors",
        json={"actor_id": "actor-2", "name": "Bob", "actor_type": "organization"},
    )
    own = create_content(client, digest=DIGEST_A).json()["content_id"]
    other = create_content(client, digest=DIGEST_B, actor_id="actor-2").json()[
        "content_id"
    ]
    own2 = create_content(client, digest=DIGEST_C).json()["content_id"]

    all_ids = [c["content_id"] for c in client.get("/v1/contents").json()]
    assert all_ids == [own, other, own2]

    filtered = client.get("/v1/contents", params={"actor_id": "actor-1"}).json()
    assert [c["content_id"] for c in filtered] == [own, own2]

    filtered = client.get("/v1/contents", params={"actor_id": "actor-2"}).json()
    assert [c["content_id"] for c in filtered] == [other]

    assert client.get("/v1/contents", params={"actor_id": "nobody"}).json() == []
