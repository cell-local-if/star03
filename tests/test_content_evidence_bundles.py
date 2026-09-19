"""Tests for the per-content evidence-bundle listing.

Covers GET /v1/contents/{content_id}/evidence-bundles: only bundles on the
content's own claims are returned (lineage-reachable contents and raw bytes
are never included), evidence_type/media_type are non-empty exact-match
filters that combine conjunctively and default to unfiltered, results are in
stable bundle creation order and paginate with an opaque HMAC cursor (no
duplicates or omissions, ``count`` is the filtered total, the final cursor is
null), unknown contents stay 404 while blank/illegal/repeated parameters and
blank/tampered/malformed/mismatched/family-foreign cursors are 422 without
silent defaults, and the query (including its error paths) writes no
resource or audit rows. All fixtures are deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json
import secrets

from sqlalchemy import func, select

from provenance import pagination
from provenance.models import (
    AuditEvent,
    Claim,
    Content,
    ContentRelation,
    EvidenceBundle,
)
from tests.helpers import content_payload, create_actor


def _content_digest(name: str) -> str:
    return hashlib.sha256(f"ceb-content-{name}".encode()).hexdigest()


def _evidence_digest(name: str) -> str:
    return hashlib.sha256(f"ceb-evidence-{name}".encode()).hexdigest()


def _create_content(client, name, actor_id="org-1"):
    resp = client.post(
        "/v1/contents",
        json=content_payload(
            actor_id=actor_id, digest=_content_digest(name), title=name
        ),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(
    client, content_id, actor_id="org-1", claim_type="authorship", payload=None
):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": claim_type,
            "payload": payload if payload is not None else {"statement": "x"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_bundle(
    client,
    claim_id,
    evidence_type="raw_capture",
    media_type="image/jpeg",
    digest_name=None,
    metadata=None,
):
    digest = (
        _evidence_digest(digest_name)
        if digest_name is not None
        else hashlib.sha256(
            f"{claim_id}.{evidence_type}.{media_type}".encode()
        ).hexdigest()
    )
    resp = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim_id,
            "evidence_type": evidence_type,
            "digest_algorithm": "sha256",
            "digest_hex": digest,
            "media_type": media_type,
            "metadata": metadata if metadata is not None else {"k": "v"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _edge(client, child, parent, relation_type="version_of"):
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


def _list(client, content_id, **params):
    return client.get(f"/v1/contents/{content_id}/evidence-bundles", params=params)


def _walk_pages(client, content_id, **params):
    """Follow next_cursor to exhaustion; return (all_items, pages, count)."""
    pages = []
    all_items = []
    cursor = None
    count = None
    for _ in range(100):
        query = dict(params)
        if cursor is not None:
            query["cursor"] = cursor
        resp = _list(client, content_id, **query)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        count = body["count"]
        pages.append(body["items"])
        all_items.extend(body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            break
    return all_items, pages, count


# --- Response shape and public fields ----------------------------------------


def test_response_shape_items_count_next_cursor(client):
    create_actor(client)
    content = _create_content(client, "a")
    body = _list(client, content["id"]).json()
    assert set(body) == {"items", "count", "next_cursor"}
    assert body == {"items": [], "count": 0, "next_cursor": None}


def test_items_are_existing_bundle_public_fields_without_bytes(client):
    create_actor(client)
    content = _create_content(client, "a")
    claim = _create_claim(client, content["id"])
    created = _create_bundle(client, claim["id"], digest_name="a1")

    body = _list(client, content["id"]).json()
    assert body["count"] == 1
    item = body["items"][0]
    assert item == created
    assert set(item) == {
        "id",
        "claim_id",
        "evidence_type",
        "digest_algorithm",
        "digest_hex",
        "media_type",
        "metadata",
        "created_at",
    }
    # Nothing capable of carrying raw evidence is ever projected.
    assert "data" not in item and "evidence" not in item


# --- Lineage isolation --------------------------------------------------------


def test_only_bundles_on_the_contents_own_claims_are_returned(client):
    # B is a version of A; A carries evidence. B's listing must not include
    # A's bundles merely because A is lineage-reachable.
    create_actor(client)
    content_a = _create_content(client, "a", )
    content_b = _create_content(client, "b")
    claim_a = _create_claim(client, content_a["id"])
    claim_b = _create_claim(client, content_b["id"])
    bundle_a = _create_bundle(client, claim_a["id"], digest_name="on-a")
    bundle_b = _create_bundle(client, claim_b["id"], digest_name="on-b")
    _edge(client, content_b["id"], content_a["id"], "version_of")

    body_b = _list(client, content_b["id"]).json()
    assert [item["id"] for item in body_b["items"]] == [bundle_b["id"]]
    assert bundle_a["id"] not in {item["id"] for item in body_b["items"]}

    body_a = _list(client, content_a["id"]).json()
    assert [item["id"] for item in body_a["items"]] == [bundle_a["id"]]


def test_descendant_evidence_is_not_returned_for_an_ancestor(client):
    create_actor(client)
    parent = _create_content(client, "parent")
    child = _create_content(client, "child")
    claim_child = _create_claim(client, child["id"])
    bundle_child = _create_bundle(client, claim_child["id"], digest_name="c")
    _edge(client, child["id"], parent["id"], "derived_from")

    # The parent has no claims of its own: an empty collection, never the
    # descendant's evidence and never a 404 (the parent exists).
    body = _list(client, parent["id"])
    assert body.status_code == 200
    assert body.json() == {"items": [], "count": 0, "next_cursor": None}
    assert bundle_child["id"] not in {
        item["id"] for item in body.json()["items"]
    }


def test_bundles_aggregate_across_the_contents_claims_only(client):
    # Two contents each with two claims; bundles on all of content X's claims
    # aggregate, while content Y's interleaved bundles never leak in.
    create_actor(client)
    x = _create_content(client, "x")
    y = _create_content(client, "y")
    x_claims = [
        _create_claim(client, x["id"], claim_type="authorship"),
        _create_claim(client, x["id"], claim_type="integrity"),
    ]
    y_claim = _create_claim(client, y["id"], claim_type="authorship")

    x1 = _create_bundle(client, x_claims[0]["id"], digest_name="x1")
    y1 = _create_bundle(client, y_claim["id"], digest_name="y1")
    x2 = _create_bundle(client, x_claims[1]["id"], digest_name="x2")

    body = _list(client, x["id"]).json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [x1["id"], x2["id"]]
    assert {item["claim_id"] for item in body["items"]} == {
        c["id"] for c in x_claims
    }
    assert y1["id"] not in {item["id"] for item in body["items"]}


# --- Combined exact-match filtering ------------------------------------------


def _four_bundles(client, claim_id):
    # 2x2 matrix of evidence_type x media_type, created in a fixed order.
    specs = [
        ("raw_capture", "image/jpeg", "b1"),
        ("raw_capture", "application/pdf", "b2"),
        ("signature", "image/jpeg", "b3"),
        ("signature", "application/pdf", "b4"),
    ]
    return [
        _create_bundle(client, claim_id, evidence_type=t, media_type=m, digest_name=d)
        for t, m, d in specs
    ]


def test_unfiltered_returns_all_and_each_filter_is_exact(client):
    create_actor(client)
    content = _create_content(client, "a")
    claim = _create_claim(client, content["id"])
    b1, b2, b3, b4 = _four_bundles(client, claim["id"])

    assert [i["id"] for i in _list(client, content["id"]).json()["items"]] == [
        b1["id"], b2["id"], b3["id"], b4["id"]
    ]
    by_type = _list(client, content["id"], evidence_type="raw_capture").json()
    assert [i["id"] for i in by_type["items"]] == [b1["id"], b2["id"]]
    by_media = _list(client, content["id"], media_type="image/jpeg").json()
    assert [i["id"] for i in by_media["items"]] == [b1["id"], b3["id"]]


def test_filters_combine_conjunctively(client):
    create_actor(client)
    content = _create_content(client, "a")
    claim = _create_claim(client, content["id"])
    b1, _, _, _ = _four_bundles(client, claim["id"])

    both = _list(
        client,
        content["id"],
        evidence_type="raw_capture",
        media_type="image/jpeg",
    ).json()
    assert [i["id"] for i in both["items"]] == [b1["id"]]
    assert both["count"] == 1

    # Combined filters with no intersection: empty collection, count zero.
    none_match = _list(
        client,
        content["id"],
        evidence_type="raw_capture",
        media_type="image/png",
    )
    assert none_match.status_code == 200
    assert none_match.json() == {"items": [], "count": 0, "next_cursor": None}


def test_filters_are_case_and_suffix_sensitive_exact_matches(client):
    create_actor(client)
    content = _create_content(client, "a")
    claim = _create_claim(client, content["id"])
    _create_bundle(client, claim["id"], evidence_type="raw_capture", digest_name="x")

    # No normalization: casing variants, surrounding whitespace, and partial
    # values match nothing (and are distinct from blank-input 422s).
    for params in (
        {"evidence_type": "RAW_CAPTURE"},
        {"evidence_type": "raw_capture "},
        {"evidence_type": "raw"},
        {"media_type": "IMAGE/JPEG"},
        {"media_type": "image"},
    ):
        resp = _list(client, content["id"], **params)
        assert resp.status_code == 200, params
        assert resp.json()["count"] == 0, params


# --- Stable order and pagination continuity -----------------------------------


def test_stable_order_interleaved_across_claims(client):
    create_actor(client)
    content = _create_content(client, "x")
    claims = [
        _create_claim(client, content["id"], claim_type=f"t{i}")
        for i in range(3)
    ]
    b1 = _create_bundle(client, claims[0]["id"], digest_name="1")
    b2 = _create_bundle(client, claims[1]["id"], digest_name="2")
    b3 = _create_bundle(client, claims[0]["id"], digest_name="3")
    b4 = _create_bundle(client, claims[2]["id"], digest_name="4")

    body = _list(client, content["id"]).json()
    # Global bundle creation order (timestamp, monotonic seq), not grouped by
    # claim.
    assert [i["id"] for i in body["items"]] == [
        b1["id"], b2["id"], b3["id"], b4["id"]
    ]


def test_pages_concatenate_without_gaps_or_duplicates(client):
    create_actor(client)
    content = _create_content(client, "x")
    claim = _create_claim(client, content["id"])
    created = [
        _create_bundle(
            client,
            claim["id"],
            evidence_type=f"type{i:02d}",
            digest_name=f"d{i:02d}",
        )
        for i in range(7)
    ]
    all_items, pages, count = _walk_pages(client, content["id"], limit=3)
    assert [len(page) for page in pages] == [3, 3, 1]
    assert count == 7
    ids = [item["id"] for item in all_items]
    assert len(ids) == len(set(ids)) == 7
    assert ids == [b["id"] for b in created]


def test_count_is_the_filtered_total_on_every_page(client):
    create_actor(client)
    content = _create_content(client, "x")
    claim = _create_claim(client, content["id"])
    keep = []
    for i in range(5):
        keep.append(
            _create_bundle(
                client, claim["id"], evidence_type="keep",
                digest_name=f"k{i}",
            )
        )
    for i in range(3):
        _create_bundle(
            client, claim["id"], evidence_type="other",
            digest_name=f"o{i}",
        )

    all_items, pages, count = _walk_pages(
        client, content["id"], evidence_type="keep", limit=2
    )
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    assert [i["id"] for i in all_items] == [b["id"] for b in keep]
    assert all(
        item["evidence_type"] == "keep"
        for page in pages
        for item in page
    )


def test_final_page_cursor_is_null_when_pages_divide_exactly(client):
    create_actor(client)
    content = _create_content(client, "x")
    claim = _create_claim(client, content["id"])
    for i in range(4):
        _create_bundle(client, claim["id"], digest_name=f"d{i}")

    first = _list(client, content["id"], limit=2).json()
    assert first["count"] == 4
    assert first["next_cursor"] is not None
    second = _list(
        client, content["id"], limit=2, cursor=first["next_cursor"]
    ).json()
    assert len(second["items"]) == 2
    assert second["count"] == 4
    assert second["next_cursor"] is None


def test_partial_final_page_cursor_is_null(client):
    create_actor(client)
    content = _create_content(client, "x")
    claim = _create_claim(client, content["id"])
    bundles = [
        _create_bundle(client, claim["id"], digest_name=f"d{i}")
        for i in range(4)
    ]
    first = _list(client, content["id"], limit=3).json()
    second = _list(
        client, content["id"], limit=3, cursor=first["next_cursor"]
    ).json()
    assert [i["id"] for i in second["items"]] == [bundles[3]["id"]]
    assert second["next_cursor"] is None


def test_default_limit_is_fifty(client):
    create_actor(client)
    content = _create_content(client, "x")
    claim = _create_claim(client, content["id"])
    bundles = [
        _create_bundle(
            client, claim["id"], evidence_type=f"t{i:02d}",
            digest_name=f"d{i:02d}",
        )
        for i in range(55)
    ]
    first = _list(client, content["id"]).json()
    assert len(first["items"]) == 50
    assert first["count"] == 55
    assert first["next_cursor"] is not None
    second = _list(
        client, content["id"], cursor=first["next_cursor"]
    ).json()
    assert len(second["items"]) == 5
    assert second["count"] == 55
    assert second["next_cursor"] is None
    full = [i["id"] for i in first["items"]] + [
        i["id"] for i in second["items"]
    ]
    assert full == [b["id"] for b in bundles]


def test_limit_one_pages_every_item(client):
    create_actor(client)
    content = _create_content(client, "x")
    claim = _create_claim(client, content["id"])
    bundles = [
        _create_bundle(client, claim["id"], digest_name=f"d{i}")
        for i in range(3)
    ]
    all_items, pages, count = _walk_pages(client, content["id"], limit=1)
    assert [len(page) for page in pages] == [1, 1, 1]
    assert count == 3
    assert [i["id"] for i in all_items] == [b["id"] for b in bundles]


def test_pagination_is_deterministic_across_walks(client):
    create_actor(client)
    content = _create_content(client, "x")
    claim = _create_claim(client, content["id"])
    for i in range(6):
        _create_bundle(
            client, claim["id"], evidence_type="keep", digest_name=f"d{i}"
        )

    def cursors_only():
        seen = []
        cursor = None
        for _ in range(20):
            params = {"evidence_type": "keep", "limit": 2}
            if cursor is not None:
                params["cursor"] = cursor
            body = _list(client, content["id"], **params).json()
            cursor = body["next_cursor"]
            if cursor is None:
                break
            seen.append(cursor)
        return seen

    assert cursors_only() == cursors_only()


def test_reusing_a_cursor_replays_the_same_page(client):
    create_actor(client)
    content = _create_content(client, "x")
    claim = _create_claim(client, content["id"])
    bundles = [
        _create_bundle(client, claim["id"], digest_name=f"d{i}")
        for i in range(4)
    ]
    cursor = _list(client, content["id"], limit=2).json()["next_cursor"]
    r1 = _list(client, content["id"], limit=2, cursor=cursor).json()
    r2 = _list(client, content["id"], limit=2, cursor=cursor).json()
    assert r1 == r2
    assert [i["id"] for i in r1["items"]] == [
        bundles[2]["id"], bundles[3]["id"]
    ]


def test_explicit_params_equal_to_effective_first_page_resume(client):
    create_actor(client)
    content = _create_content(client, "x")
    claim = _create_claim(client, content["id"])
    bundles = [
        _create_bundle(client, claim["id"], evidence_type="keep",
                       digest_name=f"d{i}")
        for i in range(4)
    ]
    first = _list(
        client, content["id"], evidence_type="keep", limit=2
    ).json()
    # Repeating the same effective parameters explicitly resumes normally.
    second = _list(
        client,
        content["id"],
        evidence_type="keep",
        limit=2,
        cursor=first["next_cursor"],
    )
    assert second.status_code == 200, second.text
    assert [i["id"] for i in second.json()["items"]] == [
        bundles[2]["id"], bundles[3]["id"]
    ]


# --- Cursor integrity ---------------------------------------------------------


def test_blank_malformed_tampered_and_foreign_cursors_are_422(client, app):
    create_actor(client)
    content = _create_content(client, "x")
    claim = _create_claim(client, content["id"])
    for i in range(3):
        _create_bundle(client, claim["id"], digest_name=f"d{i}")
    good = _list(client, content["id"], limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    foreign = pagination.encode_cursor(
        secrets.token_bytes(32),
        {
            "content_id": content["id"],
            "evidence_type": None,
            "media_type": None,
            "limit": 1,
            "offset": 1,
        },
        kind="evidence_bundle",
    )
    for token in (
        "",
        "   ",
        "not-a-cursor",
        "v1eb.onlytwoparts",
        "v1eb.too.many.parts",
        "v0.x.y",
        "v2.x.y",
        tampered,
        foreign,
    ):
        resp = _list(client, content["id"], limit=1, cursor=token)
        assert resp.status_code == 422, repr(token)
        assert resp.json()["error"]["code"] == "validation_error"
        assert "items" not in resp.json()


def test_cursor_from_lineage_family_does_not_resume_this_endpoint(
    client, app
):
    # A lineage cursor (marker "v1") signed by the lineage secret is a
    # cross-family token here and must be rejected, and vice versa.
    create_actor(client)
    content = _create_content(client, "x")
    claim = _create_claim(client, content["id"])
    for i in range(3):
        _create_bundle(client, claim["id"], digest_name=f"d{i}")
    lineage_token = pagination.encode_cursor(
        app.state.lineage_cursor_secret,
        {
            "content_id": content["id"],
            "direction": "ancestors",
            "max_depth": 8,
            "min_depth": 1,
            "relation_type": None,
            "limit": 1,
            "offset": 1,
        },
        kind="lineage",
    )
    resp = _list(client, content["id"], cursor=lineage_token)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    bundle_cursor = _list(
        client, content["id"], limit=1
    ).json()["next_cursor"]
    lineage_resp = client.get(
        f"/v1/contents/{content['id']}/lineage",
        params={"direction": "ancestors", "cursor": bundle_cursor},
    )
    assert lineage_resp.status_code == 422
    assert lineage_resp.json()["error"]["code"] == "validation_error"


def test_cursor_with_expired_format_marker_is_rejected(client, app):
    create_actor(client)
    content = _create_content(client, "x")
    # Hand-mint a structurally valid token carrying an obsolete marker.
    payload = pagination._b64encode(  # type: ignore[attr-defined]
        json.dumps(
            {
                "content_id": content["id"],
                "evidence_type": None,
                "media_type": None,
                "limit": 50,
                "offset": 1,
            },
            separators=(",", ":"),
        ).encode("utf-8")
    )
    sig = pagination._b64encode(  # type: ignore[attr-defined]
        pagination._sign(  # type: ignore[attr-defined]
            app.state.evidence_bundle_cursor_secret, "v0", payload
        )
    )
    resp = _list(client, content["id"], cursor=f"v0.{payload}.{sig}")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_is_bound_to_every_effective_parameter(client):
    create_actor(client)
    content = _create_content(client, "x")
    other = _create_content(client, "y")
    claim = _create_claim(client, content["id"])
    for i in range(6):
        _create_bundle(
            client, claim["id"], evidence_type="keep",
            media_type="image/jpeg", digest_name=f"d{i}",
        )

    # Cursor minted with evidence_type=keep, limit=2.
    cursor = _list(
        client, content["id"], evidence_type="keep", limit=2
    ).json()["next_cursor"]

    # Any divergence in filter or effective limit rejects the continuation.
    drifts = [
        {},  # evidence_type dropped -> cursor binds keep, request is unfiltered
        {"evidence_type": "other"},
        {"media_type": "image/jpeg"},
        {"evidence_type": "keep", "limit": 3},
        {"evidence_type": "keep", "limit": 50},
    ]
    for params in drifts:
        resp = _list(client, content["id"], cursor=cursor, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"

    # A different path origin is a mismatch even with identical query params.
    resp = _list(
        client, other["id"], evidence_type="keep", limit=2, cursor=cursor
    )
    assert resp.status_code == 422

    # A cursor minted without filters cannot be resumed with a filter added.
    unfiltered_cursor = _list(
        client, content["id"], limit=2
    ).json()["next_cursor"]
    resp = _list(
        client,
        content["id"],
        evidence_type="keep",
        limit=2,
        cursor=unfiltered_cursor,
    )
    assert resp.status_code == 422


# --- Query parameter validation ----------------------------------------------


def test_blank_filters_are_422_not_silent_absent(client):
    create_actor(client)
    content = _create_content(client, "x")
    for field in ("evidence_type", "media_type"):
        for value in ("", "   ", "\t"):
            resp = _list(client, content["id"], **{field: value})
            assert resp.status_code == 422, (field, value)
            assert resp.json()["error"]["code"] == "validation_error"


def test_limit_boundaries(client):
    create_actor(client)
    content = _create_content(client, "x")
    for value in ("0", "101", "-1", "1.0", "8.0", "abc", ""):
        resp = _list(client, content["id"], limit=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"
    for value in (1, 100):
        resp = _list(client, content["id"], limit=value)
        assert resp.status_code == 200, value


def test_repeated_parameters_are_422(client):
    create_actor(client)
    content = _create_content(client, "x")
    base = f"/v1/contents/{content['id']}/evidence-bundles"
    for suffix in (
        "evidence_type=a&evidence_type=b",
        "media_type=a&media_type=b",
        "limit=1&limit=2",
        "cursor=x&cursor=y",
    ):
        resp = client.get(f"{base}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_cursor_with_otherwise_valid_params_is_422(client):
    create_actor(client)
    content = _create_content(client, "x")
    resp = _list(
        client,
        content["id"],
        evidence_type="keep",
        media_type="image/jpeg",
        limit=10,
        cursor="garbage",
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Missing resource ---------------------------------------------------------


def test_unknown_content_is_404_even_with_filters_and_limit(client):
    create_actor(client)
    resp = _list(
        client,
        "cnt_ghost",
        evidence_type="keep",
        media_type="image/jpeg",
        limit=10,
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "content_not_found"
    assert error["details"]["content_id"] == "cnt_ghost"


def test_signed_cursor_for_unknown_content_still_404(client, app):
    # A structurally valid, correctly-signed cursor does not conjure a
    # resource: unknown content is a 404 after cursor validation.
    create_actor(client)
    token = pagination.encode_cursor(
        app.state.evidence_bundle_cursor_secret,
        {
            "content_id": "cnt_ghost",
            "evidence_type": None,
            "media_type": None,
            "limit": 50,
            "offset": 1,
        },
        kind="evidence_bundle",
    )
    resp = _list(client, "cnt_ghost", cursor=token)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "content_not_found"


def test_existing_content_without_claims_is_empty_not_404(client):
    create_actor(client)
    content = _create_content(client, "lonely")
    resp = _list(client, content["id"])
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


# --- Read-only guarantee ------------------------------------------------------


def test_queries_and_their_error_paths_write_nothing(client, db_session):
    create_actor(client)
    content = _create_content(client, "x")
    claim = _create_claim(client, content["id"])
    for i in range(5):
        _create_bundle(
            client, claim["id"], evidence_type="keep",
            digest_name=f"d{i}",
        )

    def counts():
        return (
            db_session.scalar(select(func.count()).select_from(Content)),
            db_session.scalar(select(func.count()).select_from(Claim)),
            db_session.scalar(
                select(func.count()).select_from(EvidenceBundle)
            ),
            db_session.scalar(
                select(func.count()).select_from(ContentRelation)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    before = counts()

    # Walk every page, filtered and unfiltered.
    for params in (
        {"limit": 2},
        {"evidence_type": "keep", "media_type": "image/jpeg", "limit": 2},
        {"evidence_type": "none-such"},
    ):
        cursor = None
        for _ in range(10):
            query = dict(params)
            if cursor is not None:
                query["cursor"] = cursor
            resp = _list(client, content["id"], **query)
            assert resp.status_code == 200, resp.text
            cursor = resp.json()["next_cursor"]
            if cursor is None:
                break

    # Missing-resource and validation paths must also write nothing.
    assert _list(client, "cnt_ghost").status_code == 404
    for attempt in (
        _list(client, content["id"], evidence_type="  "),
        _list(client, content["id"], media_type=""),
        _list(client, content["id"], limit="0"),
        _list(client, content["id"], cursor="tampered"),
        client.get(
            f"/v1/contents/{content['id']}/evidence-bundles?limit=1&limit=2"
        ),
    ):
        assert attempt.status_code == 422, attempt.text

    db_session.expire_all()
    assert counts() == before
