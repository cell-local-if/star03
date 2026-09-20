"""Tests for the read-only content provenance-evidence export.

Covers GET /v1/contents/{content_id}/export: the success body is exactly
{"content", "claims"}; content is the full public content view; claims are
limited to those directly asserting this content (no lineage traversal, no
foreign claims/bundles), in stable creation order, each carrying its
evidence bundles in stable creation order as the existing public views;
empty claim/bundle collections serialize as empty arrays; any query
parameter is 422 validation_error; an unknown content is 404
content_not_found; and the route writes no resource or audit rows.
All fixtures are deterministic and offline.
"""

from __future__ import annotations

import hashlib

from sqlalchemy import func, select

from provenance.models import AuditEvent, Claim, Content, EvidenceBundle
from tests.helpers import create_actor


def _digest(name: str) -> str:
    return hashlib.sha256(f"export-{name}".encode()).hexdigest()


def _create_content(client, name, actor_id="org-1"):
    resp = client.post(
        "/v1/contents",
        json={
            "digest_algorithm": "sha256",
            "digest_hex": _digest(f"content-{name}"),
            "media_type": "image/png",
            "title": name,
            "actor_id": actor_id,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, marker, actor_id="org-1", claim_type="authorship"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": claim_type,
            "payload": {"marker": marker},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_bundle(client, claim_id, name, evidence_type="raw_capture"):
    resp = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim_id,
            "evidence_type": evidence_type,
            "digest_algorithm": "sha256",
            "digest_hex": _digest(f"evidence-{name}"),
            "media_type": "image/jpeg",
            "metadata": {"name": name},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_export_empty_claims(client):
    create_actor(client)
    content = _create_content(client, "lonely")

    resp = client.get(f"/v1/contents/{content['id']}/export")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"content", "claims"}
    assert body["content"] == content
    assert body["claims"] == []


def test_export_claims_and_bundles_in_stable_order(client):
    create_actor(client)
    content = _create_content(client, "main")
    other = _create_content(client, "other")

    claim_a = _create_claim(client, content["id"], "a")
    claim_b = _create_claim(client, content["id"], "b", claim_type="review")
    # A claim on a different content must never leak into this export.
    foreign_claim = _create_claim(client, other["id"], "foreign")

    bundle_a2 = _create_bundle(client, claim_a["id"], "a2")
    bundle_a1 = _create_bundle(client, claim_a["id"], "a1")
    _create_bundle(client, foreign_claim["id"], "foreign")

    resp = client.get(f"/v1/contents/{content['id']}/export")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"content", "claims"}
    assert body["content"] == content

    claims = body["claims"]
    assert [c["id"] for c in claims] == [claim_a["id"], claim_b["id"]]

    expected_claim_keys = {
        "id",
        "content_id",
        "actor_id",
        "claim_type",
        "payload_digest_algorithm",
        "payload_digest_hex",
        "created_at",
        "evidence_bundles",
    }
    assert all(set(c) == expected_claim_keys for c in claims)

    first = claims[0]
    assert [b["id"] for b in first["evidence_bundles"]] == [
        bundle_a2["id"],
        bundle_a1["id"],
    ]
    expected_bundle_keys = {
        "id",
        "claim_id",
        "evidence_type",
        "digest_algorithm",
        "digest_hex",
        "media_type",
        "metadata",
        "created_at",
    }
    assert all(set(b) == expected_bundle_keys for b in first["evidence_bundles"])
    # A claim without evidence bundles exports an empty array.
    assert claims[1]["evidence_bundles"] == []


def test_export_does_not_traverse_lineage(client):
    create_actor(client)
    parent = _create_content(client, "parent")
    child = _create_content(client, "child")
    resp = client.post(
        "/v1/content-relations",
        json={
            "content_id": child["id"],
            "parent_content_id": parent["id"],
            "relation_type": "derived_from",
        },
    )
    assert resp.status_code == 201, resp.text

    parent_claim = _create_claim(client, parent["id"], "parent-claim")
    _create_bundle(client, parent_claim["id"], "parent-evidence")
    _create_claim(client, child["id"], "child-claim")

    resp = client.get(f"/v1/contents/{child['id']}/export")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["content"]["id"] == child["id"]
    assert [c["content_id"] for c in body["claims"]] == [child["id"]]


def test_export_unknown_content_is_404(client):
    resp = client.get("/v1/contents/cnt_doesnotexist/export")
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "content_not_found"


def test_export_rejects_any_query_parameter(client):
    create_actor(client)
    content = _create_content(client, "params")

    for url in (
        f"/v1/contents/{content['id']}/export?limit=10",
        f"/v1/contents/{content['id']}/export?cursor=abc",
        f"/v1/contents/{content['id']}/export?unknown=",
        f"/v1/contents/{content['id']}/export?a=1&b=2",
    ):
        resp = client.get(url)
        assert resp.status_code == 422, (url, resp.text)
        assert resp.json()["error"]["code"] == "validation_error"


def test_export_writes_no_resources_or_audit_events(client, db_session):
    create_actor(client)
    content = _create_content(client, "readonly")
    claim = _create_claim(client, content["id"], "ro")
    _create_bundle(client, claim["id"], "ro")

    counts_before = {
        model: db_session.execute(select(func.count()).select_from(model)).scalar_one()
        for model in (Content, Claim, EvidenceBundle, AuditEvent)
    }

    resp = client.get(f"/v1/contents/{content['id']}/export")
    assert resp.status_code == 200, resp.text
    # Failures are read-only too.
    assert client.get(f"/v1/contents/{content['id']}/export?x=1").status_code == 422
    assert client.get("/v1/contents/cnt_missing/export").status_code == 404

    for model, before in counts_before.items():
        after = db_session.execute(
            select(func.count()).select_from(model)
        ).scalar_one()
        assert after == before, model.__name__
