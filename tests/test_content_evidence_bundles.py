"""Tests for the content-scoped, read-only evidence-bundle listing.

Covers GET /v1/contents/{content_id}/evidence-bundles: only bundles linked
through claims that directly assert the content are returned (no lineage
traversal to related contents, no raw bytes), bundles across the content's
several claims follow the stable bundle creation order, evidence_type and
media_type are non-empty exact, combinable filters, pages concatenate
without gaps or duplicates while count stays the filtered total and the
final cursor is null, cursors are opaque/HMAC-bound to the origin and
every effective parameter (tampered, foreign-family, or mismatched
cursors are 422), unknown content stays 404, blank/illegal/repeated
parameters are 422, and the endpoint writes no resource or audit rows.
All fixtures are deterministic and offline.
"""

from __future__ import annotations

import base64
import hashlib
import hmac as hmac_mod
import json
import secrets

from sqlalchemy import func, select

from provenance import pagination
from provenance.models import AuditEvent, Claim, Content, EvidenceBundle
from tests.helpers import create_actor


def _content_digest(name: str) -> str:
    return hashlib.sha256(f"ce-content-{name}".encode()).hexdigest()


def _evidence_digest(name: str) -> str:
    return hashlib.sha256(f"ce-evidence-{name}".encode()).hexdigest()


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


def _list(client, content_id, **params):
    return client.get(
        f"/v1/contents/{content_id}/evidence-bundles", params=params
    )


def _walk_pages(client, content_id, **params):
    """Follow next_cursor until exhausted; return (all_items, pages, count)."""
    pages = []
    all_items = []
    count = None
    cursor = None
    seen_cursors = []
    for _ in range(100):
        query = {**params}
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
        seen_cursors.append(cursor)
    return all_items, pages, count, seen_cursors


#: The exact public evidence-bundle field set; nothing byte-bearing.
PUBLIC_FIELDS = {
    "id",
    "claim_id",
    "evidence_type",
    "digest_algorithm",
    "digest_hex",
    "media_type",
    "metadata",
    "created_at",
}


def _setup_isolation_graph(client):
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
    b2 = _create_bundle(
        client, cl2["id"], "b2", evidence_type="signature",
        media_type="application/json",
    )
    b3 = _create_bundle(client, cl1b["id"], "b3", media_type="image/png")
    b4 = _create_bundle(
        client, cl1a["id"], "b4", evidence_type="signature"
    )
    b5 = _create_bundle(client, cl2["id"], "b5")
    b6 = _create_bundle(client, cl3["id"], "b6")
    return {
        "c1": c1, "c2": c2, "c3": c3,
        "cl1a": cl1a, "cl1b": cl1b, "cl2": cl2, "cl3": cl3,
        "b1": b1, "b2": b2, "b3": b3, "b4": b4, "b5": b5, "b6": b6,
    }


# --- Isolation: only this content's own claims' bundles ----------------------


def test_response_shape_and_public_fields_only(client):
    g = _setup_isolation_graph(client)
    body = _list(client, g["c1"]["id"]).json()
    assert set(body) == {"items", "count", "next_cursor"}
    assert body["count"] == 3
    assert body["next_cursor"] is None
    for item in body["items"]:
        assert set(item) == PUBLIC_FIELDS
        assert "data" not in item and "evidence" not in item


def test_items_equal_single_bundle_get_responses(client):
    g = _setup_isolation_graph(client)
    items = _list(client, g["c1"]["id"]).json()["items"]
    for item in items:
        fetched = client.get(f"/v1/evidence-bundles/{item['id']}")
        assert fetched.status_code == 200
        assert fetched.json() == item


def test_lists_only_bundles_of_that_contents_claims_in_creation_order(client):
    g = _setup_isolation_graph(client)
    body = _list(client, g["c1"]["id"]).json()
    # b1, b3, b4 are attached to c1's two claims; b2/b5 belong to c2 and
    # b6 to c3 -- global stable creation order is preserved across claims.
    assert [item["id"] for item in body["items"]] == [
        g["b1"]["id"], g["b3"]["id"], g["b4"]["id"]
    ]
    assert {item["claim_id"] for item in body["items"]} == {
        g["cl1a"]["id"], g["cl1b"]["id"]
    }

    body2 = _list(client, g["c2"]["id"]).json()
    assert [item["id"] for item in body2["items"]] == [
        g["b2"]["id"], g["b5"]["id"]
    ]


def test_does_not_traverse_lineage_in_either_direction(client):
    g = _setup_isolation_graph(client)
    # c3 derives from c1: c3's listing must not include ancestor c1's
    # bundles, and c1's must not include descendant c3's bundle.
    c3_items = _list(client, g["c3"]["id"]).json()["items"]
    assert [item["id"] for item in c3_items] == [g["b6"]["id"]]

    c1_ids = {item["id"] for item in _list(client, g["c1"]["id"]).json()["items"]}
    assert g["b6"]["id"] not in c1_ids

    # Even with the lineage endpoint reachable, no depth field appears.
    assert "depth" not in c3_items[0]


def test_same_type_and_digest_on_another_content_is_excluded(client):
    g = _setup_isolation_graph(client)
    # b1 (c1) and b5 (c2) share evidence_type/media_type but differ by
    # claim/content: filtering c1 by that type returns b1 alone.
    body = _list(
        client, g["c1"]["id"], evidence_type="raw_capture",
        media_type="image/jpeg",
    ).json()
    assert [item["id"] for item in body["items"]] == [g["b1"]["id"]]
    assert body["count"] == 1


def test_existing_content_without_bundles_is_empty_not_missing(client):
    create_actor(client)
    content = _create_content(client, "lonely")
    resp = _list(client, content["id"])
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}

    # A content with a claim that itself has no bundles is also empty.
    _create_claim(client, content["id"], {"statement": "z"})
    resp = _list(client, content["id"])
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


# --- Combined exact-match filtering ------------------------------------------


def test_evidence_type_exact_match(client):
    g = _setup_isolation_graph(client)
    body = _list(client, g["c1"]["id"], evidence_type="signature").json()
    assert [item["id"] for item in body["items"]] == [g["b4"]["id"]]
    assert body["count"] == 1
    assert body["next_cursor"] is None


def test_media_type_exact_match(client):
    g = _setup_isolation_graph(client)
    body = _list(client, g["c1"]["id"], media_type="image/jpeg").json()
    assert [item["id"] for item in body["items"]] == [
        g["b1"]["id"], g["b4"]["id"]
    ]
    assert body["count"] == 2


def test_filters_combine_as_logical_and(client):
    g = _setup_isolation_graph(client)
    body = _list(
        client, g["c1"]["id"], evidence_type="raw_capture",
        media_type="image/png",
    ).json()
    assert [item["id"] for item in body["items"]] == [g["b3"]["id"]]
    assert body["count"] == 1


def test_exact_match_is_case_and_whitespace_sensitive(client):
    g = _setup_isolation_graph(client)
    # Non-empty but non-matching values are an empty result, not a 422:
    # the filter is an exact string match with no normalization.
    for params in (
        {"evidence_type": "RAW_CAPTURE"},
        {"evidence_type": "raw_capture "},
        {"media_type": "Image/JPEG"},
        {"media_type": "image/jpeg "},
        {"evidence_type": "signature", "media_type": "image/png"},
    ):
        body = _list(client, g["c1"]["id"], **params).json()
        assert body == {"items": [], "count": 0, "next_cursor": None}, params


def test_filter_with_pagination_count_is_filtered_total(client):
    g = _setup_isolation_graph(client)
    all_items, pages, count, _ = _walk_pages(
        client, g["c1"]["id"], evidence_type="raw_capture", limit=1
    )
    # b1, b3 match; b4 is a signature.
    assert count == 2
    assert [len(page) for page in pages] == [1, 1]
    assert [item["id"] for item in all_items] == [
        g["b1"]["id"], g["b3"]["id"]
    ]


# --- Pagination continuity ----------------------------------------------------


def _setup_bundle_sequence(client, count=12):
    """``count`` bundles on one content, alternated across two claims."""
    create_actor(client)
    content = _create_content(client, "seq")
    claim_a = _create_claim(client, content["id"], {"statement": "a"})
    claim_b = _create_claim(client, content["id"], {"statement": "b"})
    bundles = []
    for i in range(count):
        claim_id = claim_a["id"] if i % 2 == 0 else claim_b["id"]
        bundles.append(_create_bundle(client, claim_id, f"e{i:02d}"))
    return content, bundles


def test_pagination_concatenates_without_gaps_or_duplicates(client):
    content, bundles = _setup_bundle_sequence(client, 12)
    all_items, pages, count, _ = _walk_pages(
        client, content["id"], limit=3
    )
    assert [len(page) for page in pages] == [3, 3, 3, 3]
    assert count == 12
    ids = [item["id"] for item in all_items]
    assert len(ids) == len(set(ids)) == 12
    assert ids == [b["id"] for b in bundles]


def test_count_is_total_on_every_page_with_filter(client):
    content, bundles = _setup_bundle_sequence(client, 12)
    # Every bundle carries the uniform media_type: count stays 12 on each
    # page, independent of the page size.
    all_items, pages, count, _ = _walk_pages(
        client, content["id"], media_type="image/jpeg", limit=4
    )
    assert count == 12
    assert [len(page) for page in pages] == [4, 4, 4]
    assert [item["id"] for item in all_items] == [b["id"] for b in bundles]


def test_last_page_cursor_null_on_exact_division(client):
    content, bundles = _setup_bundle_sequence(client, 4)
    first = _list(client, content["id"], limit=2).json()
    assert first["next_cursor"] is not None
    second = _list(
        client, content["id"], limit=2, cursor=first["next_cursor"]
    ).json()
    assert len(second["items"]) == 2
    assert second["count"] == 4
    assert second["next_cursor"] is None


def test_partial_final_page_cursor_is_null(client):
    content, bundles = _setup_bundle_sequence(client, 5)
    first = _list(client, content["id"], limit=3).json()
    second = _list(
        client, content["id"], limit=3, cursor=first["next_cursor"]
    ).json()
    assert [item["id"] for item in second["items"]] == [
        bundles[3]["id"], bundles[4]["id"]
    ]
    assert second["next_cursor"] is None


def test_default_limit_is_fifty(client):
    create_actor(client)
    content = _create_content(client, "many")
    claim = _create_claim(client, content["id"], {"statement": "x"})
    names = [f"m{i:02d}" for i in range(60)]
    for name in names:
        _create_bundle(client, claim["id"], name)

    first = _list(client, content["id"]).json()
    assert len(first["items"]) == 50
    assert first["count"] == 60
    assert first["next_cursor"] is not None
    second = _list(
        client, content["id"], cursor=first["next_cursor"]
    ).json()
    assert len(second["items"]) == 10
    assert second["count"] == 60
    assert second["next_cursor"] is None
    assert [item["metadata"]["name"] for item in first["items"]] == names[:50]
    assert [item["metadata"]["name"] for item in second["items"]] == names[50:]


def test_limit_boundaries_accepted(client):
    content, _ = _setup_bundle_sequence(client, 2)
    for value in (1, 100):
        assert _list(client, content["id"], limit=value).status_code == 200


def test_pagination_is_stable_and_deterministic(client):
    content, _ = _setup_bundle_sequence(client, 12)
    first_items, _, _, first_cursors = _walk_pages(
        client, content["id"], limit=4
    )
    second_items, _, _, second_cursors = _walk_pages(
        client, content["id"], limit=4
    )
    assert first_cursors == second_cursors
    assert first_items == second_items


def test_reusing_a_cursor_replays_the_same_page(client):
    content, bundles = _setup_bundle_sequence(client, 6)
    cursor = _list(client, content["id"], limit=2).json()["next_cursor"]
    replay_one = _list(client, content["id"], limit=2, cursor=cursor).json()
    replay_two = _list(client, content["id"], limit=2, cursor=cursor).json()
    assert replay_one == replay_two
    assert [item["id"] for item in replay_one["items"]] == [
        bundles[2]["id"], bundles[3]["id"]
    ]


def test_empty_filtered_result_has_no_cursor(client):
    content, _ = _setup_bundle_sequence(client, 3)
    body = _list(
        client, content["id"], evidence_type="no-such-type", limit=1
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


def test_cursor_past_end_returns_empty_page_with_total_count(client, app):
    content, _ = _setup_bundle_sequence(client, 3)
    token = pagination.encode_typed_cursor(
        app.state.content_evidence_cursor_secret,
        pagination.CONTENT_EVIDENCE_CURSOR,
        {
            "content_id": content["id"],
            "evidence_type": None,
            "media_type": None,
            "limit": 50,
            "offset": 99,
        },
    )
    body = _list(client, content["id"], cursor=token).json()
    assert body["items"] == []
    assert body["count"] == 3
    assert body["next_cursor"] is None


# --- Cursor integrity ---------------------------------------------------------


def test_tampered_or_malformed_cursors_are_validation_errors(client, app):
    content, _ = _setup_bundle_sequence(client, 3)
    good = _list(client, content["id"], limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    foreign = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.CONTENT_EVIDENCE_CURSOR,
        {
            "content_id": content["id"],
            "evidence_type": None,
            "media_type": None,
            "limit": 1,
            "offset": 1,
        },
    )
    for token in (
        "",
        "   ",
        "not-a-cursor",
        "ce1.onlytwoparts",
        "ce1.too.many.parts",
        "ce0.x.y",
        "ce2.x.y",
        "v1.x.y",
        tampered,
        foreign,
    ):
        resp = _list(client, content["id"], limit=1, cursor=token)
        assert resp.status_code == 422, repr(token)
        assert resp.json()["error"]["code"] == "validation_error"
        assert "items" not in resp.json()


def test_cursor_signed_with_old_format_marker_is_rejected(client, app):
    content, _ = _setup_bundle_sequence(client, 2)
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "content_id": content["id"],
                "evidence_type": None,
                "media_type": None,
                "limit": 50,
                "offset": 1,
            }
        ).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(
        hmac_mod.new(
            app.state.content_evidence_cursor_secret,
            f"ce0.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    resp = _list(client, content["id"], cursor=f"ce0.{payload}.{sig}")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_from_lineage_family_is_rejected_here(client):
    g = _setup_isolation_graph(client)
    # A structurally valid lineage cursor (v1 marker) must not resume this
    # endpoint's listing.
    lineage_cursor = pagination.encode_cursor(
        # encode_cursor uses the lineage family; secret value is irrelevant
        # because the family marker alone must fail decoding.
        secrets.token_bytes(32),
        {
            "content_id": g["c1"]["id"],
            "direction": "ancestors",
            "max_depth": 8,
            "min_depth": 1,
            "relation_type": None,
            "limit": 1,
            "offset": 1,
        },
    )
    resp = _list(client, g["c1"]["id"], limit=1, cursor=lineage_cursor)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_content_cursor_is_rejected_by_lineage_endpoint(client):
    content, _ = _setup_bundle_sequence(client, 3)
    content_cursor = _list(
        client, content["id"], limit=1
    ).json()["next_cursor"]
    resp = client.get(
        f"/v1/contents/{content['id']}/lineage",
        params={"direction": "ancestors", "limit": 1, "cursor": content_cursor},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_bound_to_every_effective_query_parameter(client):
    content, _ = _setup_bundle_sequence(client, 12)
    other = _create_content(client, "other-content")
    cursor = _list(client, content["id"], limit=2).json()["next_cursor"]

    # A filter present in the request but absent from the cursor mismatches.
    mismatches = [
        {"limit": 3},
        {"evidence_type": "raw_capture"},
        {"media_type": "image/png"},
    ]
    for params in mismatches:
        resp = _list(client, content["id"], cursor=cursor, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"

    # Different origin path mismatches even with the same limit.
    resp = _list(client, other["id"], limit=2, cursor=cursor)
    assert resp.status_code == 422

    # A filtered cursor resumed without its filter also mismatches.
    filtered_cursor = _list(
        client, content["id"], evidence_type="raw_capture", limit=2
    ).json()["next_cursor"]
    resp = _list(client, content["id"], limit=2, cursor=filtered_cursor)
    assert resp.status_code == 422

    # Same applies to a media_type filter and a changed filter value.
    media_cursor = _list(
        client, content["id"], media_type="image/jpeg", limit=2
    ).json()["next_cursor"]
    assert _list(
        client, content["id"], limit=2, cursor=media_cursor
    ).status_code == 422
    assert _list(
        client, content["id"], media_type="image/png", limit=2,
        cursor=media_cursor,
    ).status_code == 422


def test_cursor_accepts_explicit_params_equal_to_defaults(client):
    content, bundles = _setup_bundle_sequence(client, 6)
    cursor = _list(client, content["id"], limit=2).json()["next_cursor"]
    resp = _list(
        client, content["id"], limit=2, cursor=cursor,
    )
    assert resp.status_code == 200
    assert [item["id"] for item in resp.json()["items"]] == [
        bundles[2]["id"], bundles[3]["id"]
    ]


# --- Parameter validation -----------------------------------------------------


def test_blank_filters_are_validation_errors(client):
    content, _ = _setup_bundle_sequence(client, 2)
    for field in ("evidence_type", "media_type"):
        for value in ("", "   ", "\t"):
            resp = _list(client, content["id"], **{field: value})
            assert resp.status_code == 422, (field, value)
            assert resp.json()["error"]["code"] == "validation_error"


def test_illegal_limit_values_are_validation_errors(client):
    content, _ = _setup_bundle_sequence(client, 2)
    for value in ("0", "101", "-1", "1.5", "abc", "8.0", "  2", ""):
        resp = _list(client, content["id"], limit=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_validation_errors(client):
    content, _ = _setup_bundle_sequence(client, 2)
    base = f"/v1/contents/{content['id']}/evidence-bundles"
    for suffix in (
        "evidence_type=raw_capture&evidence_type=signature",
        "media_type=image/png&media_type=image/jpeg",
        "limit=1&limit=2",
        "cursor=x&cursor=y",
    ):
        resp = client.get(f"{base}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_cursor_with_otherwise_valid_params_is_422(client):
    content, _ = _setup_bundle_sequence(client, 2)
    resp = _list(
        client, content["id"], evidence_type="raw_capture",
        media_type="image/jpeg", limit=10, cursor="garbage",
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Missing content ----------------------------------------------------------


def test_unknown_content_is_not_found_even_with_filters(client):
    create_actor(client)
    for params in (
        {},
        {"evidence_type": "raw_capture"},
        {"media_type": "image/png"},
        {"limit": 10},
        {"evidence_type": "raw_capture", "media_type": "image/png",
         "limit": 10},
    ):
        resp = _list(client, "cnt_ghost", **params)
        assert resp.status_code == 404, params
        error = resp.json()["error"]
        assert error["code"] == "content_not_found"
        assert error["details"]["content_id"] == "cnt_ghost"


def test_unknown_content_with_malformed_cursor_is_validation_error(client):
    resp = _list(client, "cnt_ghost", limit=10, cursor="tampered")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only guarantee -------------------------------------------------------


def test_queries_write_no_rows_or_audit_events(client, db_session):
    g = _setup_isolation_graph(client)

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
    cursor = None
    for _ in range(5):
        params = {"evidence_type": "raw_capture", "limit": 1}
        if cursor is not None:
            params["cursor"] = cursor
        resp = _list(client, g["c1"]["id"], **params)
        assert resp.status_code == 200, resp.text
        cursor = resp.json()["next_cursor"]
        if cursor is None:
            break

    # Successful and empty reads on several contents.
    _list(client, g["c2"]["id"])
    _list(client, g["c3"]["id"], media_type="application/json")

    # Invalid and missing requests must not write anything either.
    _list(client, g["c1"]["id"], evidence_type=" ")
    _list(client, g["c1"]["id"], limit=0)
    _list(client, g["c1"]["id"], cursor="tampered")
    _list(client, "cnt_ghost")
    client.get(
        f"/v1/contents/{g['c1']['id']}/evidence-bundles?limit=1&limit=2"
    )

    db_session.expire_all()
    assert counts() == before
