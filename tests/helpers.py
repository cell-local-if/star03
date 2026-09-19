"""Deterministic payload helpers for tests."""

from __future__ import annotations

import hashlib

# sha256 of a fixed, tiny input: stable and offline.
DIGEST_A = hashlib.sha256(b"content-a").hexdigest()
DIGEST_B = hashlib.sha256(b"content-b").hexdigest()
DIGEST_C = hashlib.sha256(b"content-c").hexdigest()


def actor_payload(actor_id="org-1", name="Example Org", type="organization"):
    return {"id": actor_id, "name": name, "type": type}


def create_actor(client, **overrides):
    payload = actor_payload(**overrides)
    response = client.post("/v1/actors", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def content_payload(
    actor_id="org-1",
    digest=DIGEST_A,
    media_type="image/png",
    title=None,
    algorithm="sha256",
):
    return {
        "digest_algorithm": algorithm,
        "digest_hex": digest,
        "media_type": media_type,
        "title": title,
        "actor_id": actor_id,
    }


def create_content(client, **overrides):
    resp = client.post("/v1/contents", json=content_payload(**overrides))
    assert resp.status_code == 201, resp.text
    return resp.json()


def claim_payload(
    content_id=None,
    actor_id="org-1",
    claim_type="attribution",
    payload=None,
):
    return {
        "content_id": content_id,
        "actor_id": actor_id,
        "claim_type": claim_type,
        "payload": payload if payload is not None else {"statement": "created by actor"},
    }
