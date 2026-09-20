"""Tests for the read-only content export snapshot.

Covers GET /v1/contents/{content_id}/export: the success body is exactly
{"content", "claims"}; ``content`` is the existing full public content
view; ``claims`` follows the claims' stable creation order, each item
carrying the existing full public claim view plus ``evidence_bundles`` in
the bundles' stable creation order, each the existing public bundle view.
Only claims directly asserting the content are included (no lineage
traversal, no other contents' claims or bundles); a claim-less content
yields ``claims: []`` and a bundle-less claim yields
``evidence_bundles: []``. Any query parameter is 422 validation_error, an
unknown content is 404 content_not_found, and reads, empty results, and
failures write no resource or audit rows. All fixtures are deterministic
and offline.
"""

from __future__ import annotations

import hashlib

from sqlalchemy import func, select

from provenance.models import AuditEvent, Claim, Content, EvidenceBundle
from tests.helpers import create_actor


def _content_digest(name: str) -> str:
    return hashlib.sha256(f"ex-content-{name}".encode()).hexdigest()


def _evidence_digest(name: str) -> str:
    return hashlib.sha256(f"ex-evidence-{name}".encode()).hexdigest()


def _create_content(client, name, actor_id="org-1"):
    resp = client.post(
        "/v1/contents",
        json={
            "digest_algorithm": "sha256",
            "digest_hex": _content_digest(name),
            "media_type": "image/png",
            "title": name,
            "actor_id": actor_id,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, payload, actor_id="org-1", claim_type="authorship"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": claim_type,
            "payload": payload,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_bundle(client, claim_id, name, evidence_type="raw_capture", media_type="image/jpeg"):
    resp = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim_id,
            "evidence_type": evidence_type,
            "digest_algorithm": "sha256",
            "digest_hex": _evidence_digest(name),
            "media_type": media_type,
            "metadata": {"name": name},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _edge(client, child, parent, relation_type="derived_from"):
    resp = client.post(
        "/v1/content-relations",
        json={
            "content_id": child,
            "parent_content_id": parent,
            "relation_type": relation_type,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _export(client, content_id, **params):
    return client.get(f"/v1/contents/{content_id}/export", params=params)


#: The exact public field sets; nothing byte-bearing and no internals.
CONTENT_FIELDS = {
    "id",
    "digest_algorithm",
    "digest_hex",
    "media_type",
    "title",
    "actor_id",
    "created_at",
}
CLAIM_FIELDS = {
    "id",
    "content_id",
    "actor_id",
    "claim_type",
    "payload_digest_algorithm",
    "payload_digest_hex",
    "created_at",
}
BUNDLE_FIELDS = {
    "id",
    "claim_id",
    "evidence_type",
    "digest_algorithm",
    "digest_hex",
    "media_type",
    "metadata",
    "created_at",
}


def _setup_graph(client):
    """Three contents (c3 derived from c1), interleaved claims and bundles."""
    create_actor(client)
    create_actor(client, actor_id="org-2", name="Other", type="person")
    c1 = _create_content(client, "c1")
    c2 = _create_content(client, "c2")
    c3 = _create_content(client, "c3")
    _edge(client, c3["id"], c1["id"])

    cl1a = _create_claim(client, c1["id"], {"statement": "c1-a"})
    cl1b = _create_claim(
        client, c1["id"], {"statement": "c1-b"}, actor_id="org-2",
        claim_type="review",
    )
    cl2 = _create_claim(client, c2["id"], {"statement": "c2"})
    cl3 = _create_claim(client, c3["id"], {"statement": "c3"})

    # Deliberately interleaved across contents and claims.
    b1 = _create_bundle(client, cl1a["id"], "b1")
    b2 = _create_bundle(client, cl2["id"], "b2")
    b3 = _create_bundle(client, cl1b["id"], "b3", media_type="image/png")
    b4 = _create_bundle(client, cl1a["id"], "b4", evidence_type="signature")
    b5 = _create_bundle(client, cl2["id"], "b5")
    b6 = _create_bundle(client, cl3["id"], "b6")
    return {
        "c1": c1, "c2": c2, "c3": c3,
        "cl1a": cl1a, "cl1b": cl1b, "cl2": cl2, "cl3": cl3,
        "b1": b1, "b2": b2, "b3": b3, "b4": b4, "b5": b5, "b6": b6,
    }


# --- Response shape and public views -------------------------------------------


def test_response_shape_is_exactly_content_and_claims(client):
    g = _setup_graph(client)
    resp = _export(client, g["c1"]["id"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"content", "claims"}
    assert set(body["content"]) == CONTENT_FIELDS
    for claim in body["claims"]:
        assert set(claim) == CLAIM_FIELDS | {"evidence_bundles"}
        for bundle in claim["evidence_bundles"]:
            assert set(bundle) == BUNDLE_FIELDS
            assert "data" not in bundle and "evidence" not in bundle
        # No claim payload or internal sequencing leaks.
        assert "payload" not in claim
        assert "seq" not in claim


def test_nested_views_equal_the_existing_detail_responses(client):
    g = _setup_graph(client)
    body = _export(client, g["c1"]["id"]).json()

    content = client.get(f"/v1/contents/{g['c1']['id']}").json()
    assert body["content"] == content

    for exported_claim in body["claims"]:
        claim = client.get(f"/v1/claims/{exported_claim['id']}").json()
        assert {k: v for k, v in exported_claim.items() if k != "evidence_bundles"} == claim
        for exported_bundle in exported_claim["evidence_bundles"]:
            bundle = client.get(
                f"/v1/evidence-bundles/{exported_bundle['id']}"
            ).json()
            assert exported_bundle == bundle


def test_claims_and_bundles_follow_stable_creation_order(client):
    g = _setup_graph(client)
    body = _export(client, g["c1"]["id"]).json()
    assert [claim["id"] for claim in body["claims"]] == [
        g["cl1a"]["id"], g["cl1b"]["id"]
    ]
    by_id = {claim["id"]: claim for claim in body["claims"]}
    # Per-claim bundle order is the bundles' own creation order, not the
    # global interleaving across claims.
    assert [b["id"] for b in by_id[g["cl1a"]["id"]]["evidence_bundles"]] == [
        g["b1"]["id"], g["b4"]["id"]
    ]
    assert [b["id"] for b in by_id[g["cl1b"]["id"]]["evidence_bundles"]] == [
        g["b3"]["id"]
    ]


# --- Isolation ------------------------------------------------------------------


def test_only_direct_claims_and_their_bundles_are_included(client):
    g = _setup_graph(client)
    body = _export(client, g["c1"]["id"]).json()
    claim_ids = {claim["id"] for claim in body["claims"]}
    assert claim_ids == {g["cl1a"]["id"], g["cl1b"]["id"]}
    bundle_ids = {
        bundle["id"]
        for claim in body["claims"]
        for bundle in claim["evidence_bundles"]
    }
    assert bundle_ids == {g["b1"]["id"], g["b3"]["id"], g["b4"]["id"]}


def test_does_not_traverse_lineage_in_either_direction(client):
    g = _setup_graph(client)
    # c3 derives from c1: c3's export must not include ancestor c1's claims
    # or bundles, and c1's must not include descendant c3's.
    c3_body = _export(client, g["c3"]["id"]).json()
    assert [claim["id"] for claim in c3_body["claims"]] == [g["cl3"]["id"]]
    assert [b["id"] for b in c3_body["claims"][0]["evidence_bundles"]] == [
        g["b6"]["id"]
    ]

    c1_ids = {claim["id"] for claim in _export(client, g["c1"]["id"]).json()["claims"]}
    assert g["cl3"]["id"] not in c1_ids


# --- Empty results ----------------------------------------------------------------


def test_content_without_claims_exports_empty_claims_array(client):
    create_actor(client)
    content = _create_content(client, "lonely")
    resp = _export(client, content["id"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["claims"] == []
    assert body["content"]["id"] == content["id"]


def test_claim_without_bundles_exports_empty_bundle_array(client):
    create_actor(client)
    content = _create_content(client, "bare")
    claim = _create_claim(client, content["id"], {"statement": "z"})
    body = _export(client, content["id"]).json()
    assert [c["id"] for c in body["claims"]] == [claim["id"]]
    assert body["claims"][0]["evidence_bundles"] == []


# --- Query parameters --------------------------------------------------------------


def test_any_query_parameter_is_a_validation_error(client):
    g = _setup_graph(client)
    base = f"/v1/contents/{g['c1']['id']}/export"
    for suffix in (
        "limit=1",
        "cursor=abc",
        "direction=ancestors",
        "evidence_type=raw_capture",
        "include=claims",
        "foo=",
        "foo",
        "limit=1&limit=2",
    ):
        resp = client.get(f"{base}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"
        assert "content" not in resp.json()


def test_query_parameter_on_unknown_content_is_still_a_validation_error(client):
    create_actor(client)
    resp = _export(client, "cnt_ghost", limit=1)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Missing content -----------------------------------------------------------------


def test_unknown_content_is_not_found(client):
    create_actor(client)
    resp = _export(client, "cnt_ghost")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "content_not_found"
    assert error["details"]["content_id"] == "cnt_ghost"


# --- Read-only guarantee ---------------------------------------------------------------


def test_reads_and_failures_write_no_rows_or_audit_events(client, db_session):
    g = _setup_graph(client)
    create_actor(client, actor_id="org-3", name="Third", type="person")
    empty = _create_content(client, "empty", actor_id="org-3")

    def counts():
        return (
            db_session.scalar(select(func.count()).select_from(Content)),
            db_session.scalar(select(func.count()).select_from(Claim)),
            db_session.scalar(
                select(func.count()).select_from(EvidenceBundle)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    before = counts()

    # Successful, repeated, and empty reads.
    for content_id in (g["c1"]["id"], g["c2"]["id"], g["c3"]["id"], empty["id"]):
        resp = _export(client, content_id)
        assert resp.status_code == 200, resp.text
        resp = _export(client, content_id)
        assert resp.status_code == 200, resp.text

    # Rejected and missing requests must not write anything either.
    _export(client, g["c1"]["id"], limit=1)
    _export(client, g["c1"]["id"], foo="bar")
    _export(client, "cnt_ghost")
    _export(client, "cnt_ghost", limit=1)

    db_session.expire_all()
    assert counts() == before
