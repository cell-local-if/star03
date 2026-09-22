"""Tests for the reviewer read-only evidence-bundle search.

Covers GET /v1/evidence-bundles: the response shape carries exactly
``{"items", "count", "next_cursor"}`` with each item exposing only the
existing public evidence-bundle fields (never raw evidence bytes), bundles
follow stable creation order, and ``created_at`` stays UTC. The optional
``claim_id``, ``evidence_type``, and ``media_type`` filters are non-empty,
case- and whitespace-sensitive exact matches that combine as logical AND
(absent means unfiltered); ``digest_hex`` must be exactly 64 lowercase
hexadecimal characters. ``limit`` is 1..100 defaulting to 50; the opaque
HMAC cursor binds every effective filter and the limit so pages
concatenate without gaps or duplicates and the final cursor is null.
Blank/illegal/repeated/undeclared parameters and blank/malformed/
tampered/foreign-family/query-mismatching cursors are 422
validation_error; no match is an empty collection. Neither queries nor
failures write any bundle, resource, or audit row, and the existing bundle
creation/detail/per-claim/per-content routes stay compatible. All fixtures
are deterministic and offline.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance import pagination
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import AuditEvent, EvidenceBundle
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    content_payload,
    create_actor,
)

URL = "/v1/evidence-bundles"

BUNDLE_KEYS = {
    "id",
    "claim_id",
    "evidence_type",
    "digest_algorithm",
    "digest_hex",
    "media_type",
    "metadata",
    "created_at",
}

EVIDENCE_DIGEST_1 = hashlib.sha256(b"list-evidence-a").hexdigest()
EVIDENCE_DIGEST_2 = hashlib.sha256(b"list-evidence-b").hexdigest()
EVIDENCE_DIGEST_3 = hashlib.sha256(b"list-evidence-c").hexdigest()
EVIDENCE_DIGEST_4 = hashlib.sha256(b"list-evidence-d").hexdigest()
EVIDENCE_DIGEST_5 = hashlib.sha256(b"list-evidence-e").hexdigest()


def _list(client, **params):
    return client.get(URL, params=params)


def _walk_pages(client, **params):
    """Follow next_cursor until exhausted; return (all_items, pages, count)."""
    pages = []
    all_items = []
    count = None
    cursor = None
    for _ in range(100):
        query = {**params}
        if cursor is not None:
            query["cursor"] = cursor
        resp = _list(client, **query)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        count = body["count"]
        pages.append(body["items"])
        all_items.extend(body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            break
    return all_items, pages, count


def _create_content(client, digest, actor_id="org-1"):
    resp = client.post(
        "/v1/contents", json=content_payload(actor_id=actor_id, digest=digest)
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, actor_id="org-1", claim_type="authorship"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": claim_type,
            "payload": {"statement": claim_type},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_bundle(
    client,
    claim_id,
    name,
    evidence_type="raw_capture",
    media_type="image/jpeg",
    digest=None,
):
    resp = client.post(
        URL,
        json={
            "claim_id": claim_id,
            "evidence_type": evidence_type,
            "digest_algorithm": "sha256",
            "digest_hex": digest
            if digest is not None
            else hashlib.sha256(name.encode()).hexdigest(),
            "media_type": media_type,
            "metadata": {"name": name},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _setup_grouped(client):
    """Five bundles across two contents/claims, types, media, digests.

    Returns ``(content_a, content_b, claim_a, claim_b,
    bundles_in_creation_order)``.
    """
    create_actor(client)
    content_a = _create_content(client, DIGEST_A)
    content_b = _create_content(client, DIGEST_B)
    claim_a = _create_claim(client, content_id=content_a["id"])
    claim_b = _create_claim(client, content_id=content_b["id"])
    bundles = [
        _create_bundle(
            client,
            claim_a["id"],
            "b1",
            evidence_type="raw_capture",
            media_type="image/jpeg",
            digest=EVIDENCE_DIGEST_1,
        ),
        _create_bundle(
            client,
            claim_a["id"],
            "b2",
            evidence_type="document",
            media_type="application/pdf",
            digest=EVIDENCE_DIGEST_2,
        ),
        _create_bundle(
            client,
            claim_a["id"],
            "b3",
            evidence_type="raw_capture",
            media_type="image/png",
            digest=EVIDENCE_DIGEST_3,
        ),
        _create_bundle(
            client,
            claim_b["id"],
            "b4",
            evidence_type="document",
            media_type="application/pdf",
            digest=EVIDENCE_DIGEST_4,
        ),
        _create_bundle(
            client,
            claim_b["id"],
            "b5",
            evidence_type="audio",
            media_type="audio/wav",
            digest=EVIDENCE_DIGEST_5,
        ),
    ]
    return content_a, content_b, claim_a, claim_b, bundles


def _create_many_bundles(client, count: int):
    """Create ``count`` distinct bundles on one claim with distinct digests."""
    create_actor(client)
    content = _create_content(client, DIGEST_A)
    claim = _create_claim(client, content_id=content["id"])
    created = []
    for i in range(count):
        created.append(
            _create_bundle(
                client,
                claim["id"],
                f"many-{i}",
                digest=hashlib.sha256(f"many-{i}".encode()).hexdigest(),
            )
        )
    return created


# --- Response shape, fields, and ordering -------------------------------------


def test_empty_database_is_an_empty_collection(client):
    assert _list(client).json() == {
        "items": [],
        "count": 0,
        "next_cursor": None,
    }


def test_response_shape_and_item_fields(client):
    _, _, _, _, bundles = _setup_grouped(client)
    resp = _list(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"items", "count", "next_cursor"}
    assert body["count"] == 5
    assert body["next_cursor"] is None
    assert len(body["items"]) == 5
    for item in body["items"]:
        assert set(item) == BUNDLE_KEYS
        assert item["id"].startswith("evb_")
        # Raw evidence bytes are never accepted or stored and must never be
        # echoed: no "data"/"evidence"/"bytes" member exists.
        assert "data" not in item
        assert "evidence" not in item
        assert "bytes" not in item


def test_item_is_the_public_bundle_and_matches_the_single_read(client):
    _, _, _, _, bundles = _setup_grouped(client)
    items = _list(client).json()["items"]
    for listed, created in zip(items, bundles):
        assert listed == created
        assert listed == client.get(f"{URL}/{created['id']}").json()


def test_created_at_is_utc(client):
    _setup_grouped(client)
    for item in _list(client).json()["items"]:
        parsed = datetime.fromisoformat(item["created_at"])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)
        assert item["created_at"].endswith(("Z", "+00:00"))


def test_bundles_follow_stable_creation_order(client):
    _, _, _, _, bundles = _setup_grouped(client)
    items = _list(client).json()["items"]
    assert [i["id"] for i in items] == [b["id"] for b in bundles]
    assert items == bundles


def test_listing_order_matches_the_per_claim_listing(client):
    _, _, claim_a, _, _ = _setup_grouped(client)
    by_search = [
        i["id"]
        for i in _list(client, claim_id=claim_a["id"]).json()["items"]
    ]
    by_claim = [
        i["id"]
        for i in client.get(f"/v1/claims/{claim_a['id']}/evidence-bundles")
        .json()["items"]
    ]
    assert by_search == by_claim


def test_repeat_bundle_submission_adds_no_search_position(client):
    create_actor(client)
    content = _create_content(client, DIGEST_A)
    claim = _create_claim(client, content_id=content["id"])
    first = _create_bundle(
        client, claim["id"], "r1", digest=EVIDENCE_DIGEST_1
    )
    # An idempotent repeat of the same (claim, type, algorithm, digest)
    # identity is 200 and introduces no second search position.
    repeat = client.post(
        URL,
        json={
            "claim_id": claim["id"],
            "evidence_type": "raw_capture",
            "digest_algorithm": "sha256",
            "digest_hex": EVIDENCE_DIGEST_1,
            "media_type": "image/jpeg",
            "metadata": {"name": "r1"},
        },
    )
    assert repeat.status_code == 200
    later = _create_bundle(
        client, claim["id"], "r2", digest=EVIDENCE_DIGEST_2
    )
    items = _list(client).json()["items"]
    assert [i["id"] for i in items] == [first["id"], later["id"]]


# --- Exact-match filtering -----------------------------------------------------


def test_claim_id_exact_match(client):
    _, _, claim_a, claim_b, _ = _setup_grouped(client)
    body = _list(client, claim_id=claim_a["id"]).json()
    assert body["count"] == 3
    assert {i["claim_id"] for i in body["items"]} == {claim_a["id"]}
    body = _list(client, claim_id=claim_b["id"]).json()
    assert body["count"] == 2
    assert {i["claim_id"] for i in body["items"]} == {claim_b["id"]}


def test_evidence_type_exact_match(client):
    _setup_grouped(client)
    body = _list(client, evidence_type="document").json()
    assert body["count"] == 2
    assert {i["evidence_type"] for i in body["items"]} == {"document"}
    body = _list(client, evidence_type="audio").json()
    assert body["count"] == 1
    assert {i["evidence_type"] for i in body["items"]} == {"audio"}


def test_media_type_exact_match(client):
    _setup_grouped(client)
    body = _list(client, media_type="application/pdf").json()
    assert body["count"] == 2
    assert {i["media_type"] for i in body["items"]} == {"application/pdf"}
    body = _list(client, media_type="audio/wav").json()
    assert body["count"] == 1


def test_digest_hex_exact_match(client):
    _setup_grouped(client)
    body = _list(client, digest_hex=EVIDENCE_DIGEST_1).json()
    assert body["count"] == 1
    assert body["items"][0]["digest_hex"] == EVIDENCE_DIGEST_1
    body = _list(client, digest_hex=EVIDENCE_DIGEST_4).json()
    assert body["count"] == 1


def test_filters_combine_as_logical_and(client):
    _, _, claim_a, claim_b, _ = _setup_grouped(client)
    body = _list(
        client,
        claim_id=claim_a["id"],
        evidence_type="raw_capture",
        media_type="image/jpeg",
        digest_hex=EVIDENCE_DIGEST_1,
    ).json()
    assert body["count"] == 1
    assert body["items"][0]["digest_hex"] == EVIDENCE_DIGEST_1

    # claim_b with a digest that only exists under claim_a satisfies
    # nothing: the filters are ANDed, not coerced.
    body = _list(
        client, claim_id=claim_b["id"], digest_hex=EVIDENCE_DIGEST_1
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}

    # The same evidence type across both claims narrows to the claim.
    body = _list(
        client, claim_id=claim_a["id"], evidence_type="document"
    ).json()
    assert body["count"] == 1
    assert body["items"][0]["digest_hex"] == EVIDENCE_DIGEST_2


def test_exact_match_is_case_and_whitespace_sensitive(client):
    _, _, claim_a, _, _ = _setup_grouped(client)
    for params in (
        {"evidence_type": "Raw_Capture"},
        {"evidence_type": " raw_capture"},
        {"media_type": "IMAGE/JPEG"},
        {"media_type": "image/jpeg "},
        {"claim_id": claim_a["id"].swapcase()},
        {"claim_id": f" {claim_a['id']}"},
    ):
        body = _list(client, **params).json()
        assert body == {"items": [], "count": 0, "next_cursor": None}, params
    # The digest filter rejects rather than normalizes the uppercase form;
    # a well-formed unrelated lowercase digest simply matches nothing.
    assert _list(client, digest_hex="0" * 64).json() == {
        "items": [],
        "count": 0,
        "next_cursor": None,
    }


def test_unknown_filter_values_are_empty_collection_not_not_found(client):
    _setup_grouped(client)
    for params in (
        {"claim_id": "clm_" + "0" * 64},
        {"evidence_type": "never-captured"},
        {"media_type": "application/x-nonexistent"},
    ):
        assert _list(client, **params).json() == {
            "items": [],
            "count": 0,
            "next_cursor": None,
        }, params


# --- Pagination -----------------------------------------------------------------


def test_pagination_concatenates_without_gaps_or_duplicates(client):
    _, _, _, _, bundles = _setup_grouped(client)
    all_items, pages, count = _walk_pages(client, limit=2)
    # 5 bundles -> pages of 2, 2, 1.
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    ids = [i["id"] for i in all_items]
    assert len(ids) == len(set(ids)) == 5
    assert ids == [b["id"] for b in bundles]
    assert all_items == _list(client).json()["items"]


def test_count_is_filtered_total_on_every_page(client):
    _, _, claim_a, _, _ = _setup_grouped(client)
    all_items, pages, count = _walk_pages(
        client, claim_id=claim_a["id"], limit=2
    )
    assert count == 3
    assert [len(page) for page in pages] == [2, 1]
    assert {i["claim_id"] for i in all_items} == {claim_a["id"]}


def test_last_page_cursor_null_on_exact_division(client):
    _create_many_bundles(client, 4)
    first = _list(client, limit=2).json()
    assert first["next_cursor"] is not None
    second = _list(client, limit=2, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 2
    assert second["count"] == 4
    assert second["next_cursor"] is None


def test_default_limit_is_fifty(client):
    _create_many_bundles(client, 55)
    first = _list(client).json()
    assert len(first["items"]) == 50
    assert first["count"] == 55
    assert first["next_cursor"] is not None
    second = _list(client, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 5
    assert second["count"] == 55
    assert second["next_cursor"] is None


def test_limit_boundaries_accepted(client):
    _create_many_bundles(client, 2)
    for value in (1, 100):
        assert _list(client, limit=value).status_code == 200


def test_reusing_a_cursor_replays_the_same_page(client):
    _setup_grouped(client)
    cursor = _list(client, limit=2).json()["next_cursor"]
    replay_one = _list(client, limit=2, cursor=cursor).json()
    replay_two = _list(client, limit=2, cursor=cursor).json()
    assert replay_one == replay_two


def test_cursor_past_end_returns_empty_page_with_total_count(client, app):
    _setup_grouped(client)
    token = pagination.encode_typed_cursor(
        app.state.evidence_bundles_cursor_secret,
        pagination.EVIDENCE_BUNDLES_CURSOR,
        {
            "claim_id": None,
            "evidence_type": None,
            "media_type": None,
            "digest_hex": None,
            "limit": 50,
            "offset": 99,
        },
    )
    body = _list(client, cursor=token).json()
    assert body["items"] == []
    assert body["count"] == 5
    assert body["next_cursor"] is None


# --- Cursor isolation ------------------------------------------------------------


def test_tampered_or_malformed_cursors_are_validation_errors(client):
    _setup_grouped(client)
    good = _list(client, limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    foreign = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.EVIDENCE_BUNDLES_CURSOR,
        {
            "claim_id": None,
            "evidence_type": None,
            "media_type": None,
            "digest_hex": None,
            "limit": 1,
            "offset": 1,
        },
    )
    for token in (
        "",
        "   ",
        "not-a-cursor",
        "eb1.onlytwoparts",
        "eb1.too.many.parts",
        "eb0.x.y",
        "eb2.x.y",
        "v1.x.y",
        "ce1.x.y",
        "cl1.x.y",
        "ae1.x.y",
        tampered,
        foreign,
    ):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422, repr(token)
        assert resp.json()["error"]["code"] == "validation_error"
        assert "items" not in resp.json()


def test_cursor_from_other_families_is_rejected(client):
    _setup_grouped(client)
    lineage_cursor = pagination.encode_cursor(
        secrets.token_bytes(32),
        {
            "content_id": "cnt_x",
            "direction": "ancestors",
            "max_depth": 8,
            "min_depth": 1,
            "relation_type": None,
            "limit": 1,
            "offset": 1,
        },
    )
    content_evidence_cursor = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.CONTENT_EVIDENCE_CURSOR,
        {
            "content_id": "cnt_x",
            "evidence_type": None,
            "media_type": None,
            "limit": 1,
            "offset": 1,
        },
    )
    claims_cursor = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.CLAIMS_CURSOR,
        {
            "content_id": None,
            "actor_id": None,
            "claim_type": None,
            "payload_digest_hex": None,
            "limit": 1,
            "offset": 1,
        },
    )
    audit_cursor = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.AUDIT_EVENTS_CURSOR,
        {
            "event_type": None,
            "resource_id": None,
            "from": None,
            "to": None,
            "limit": 1,
            "offset": 1,
        },
    )
    for token in (
        lineage_cursor,
        content_evidence_cursor,
        claims_cursor,
        audit_cursor,
    ):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


def test_evidence_bundles_cursor_is_rejected_by_other_endpoints(client):
    _, _, _, claim_a, _ = _setup_grouped(client)
    cursor = _list(client, limit=1).json()["next_cursor"]

    # The reviewer claim search uses its own marker and claim set.
    resp = client.get("/v1/claims", params={"limit": 1, "cursor": cursor})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    # As does the per-content evidence-bundle listing.
    resp = client.get(
        f"/v1/contents/{'cnt_' + '0' * 64}/evidence-bundles",
        params={"limit": 1, "cursor": cursor},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    # As does the audit-events listing.
    resp = client.get(
        "/v1/audit-events", params={"limit": 1, "cursor": cursor}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    # As does the lineage route.
    resp = client.get(
        f"/v1/contents/{'cnt_' + '0' * 64}/lineage",
        params={"direction": "ancestors", "limit": 1, "cursor": cursor},
    )
    assert resp.status_code == 422


def test_cursor_bound_to_every_effective_query_parameter(client):
    _, _, claim_a, claim_b, _ = _setup_grouped(client)

    cursor = _list(client, limit=2).json()["next_cursor"]
    # A filter/limit present in the resume request but absent from the
    # cursor mismatches.
    for params in (
        {"limit": 3},
        {"claim_id": claim_a["id"]},
        {"evidence_type": "raw_capture"},
        {"media_type": "image/jpeg"},
        {"digest_hex": EVIDENCE_DIGEST_1},
    ):
        resp = _list(client, cursor=cursor, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"

    # A cursor minted under a filter cannot be resumed without it or with a
    # different value: no filter may be dropped or substituted.
    filtered = _list(client, claim_id=claim_a["id"], limit=2).json()[
        "next_cursor"
    ]
    assert _list(client, limit=2, cursor=filtered).status_code == 422
    assert (
        _list(
            client, claim_id=claim_b["id"], limit=2, cursor=filtered
        ).status_code
        == 422
    )

    typed = _list(client, evidence_type="document", limit=2).json()[
        "next_cursor"
    ]
    assert _list(client, limit=2, cursor=typed).status_code == 422
    assert (
        _list(
            client, evidence_type="audio", limit=2, cursor=typed
        ).status_code
        == 422
    )

    mediaed = _list(client, media_type="image/png", limit=2).json()[
        "next_cursor"
    ]
    assert _list(client, limit=2, cursor=mediaed).status_code == 422

    digested = _list(client, digest_hex=EVIDENCE_DIGEST_1, limit=2).json()[
        "next_cursor"
    ]
    assert _list(client, limit=2, cursor=digested).status_code == 422
    assert (
        _list(
            client,
            digest_hex=EVIDENCE_DIGEST_2,
            limit=2,
            cursor=digested,
        ).status_code
        == 422
    )


def test_cursor_secret_rotation_invalidates_outstanding_cursors(
    client, tmp_db_url
):
    # A restart rotates the per-process HMAC secret, so a cursor minted by
    # the old process is rejected rather than trusted; the stored bundles
    # themselves survive.
    created = _create_many_bundles(client, 2)
    cursor = _list(client, limit=1).json()["next_cursor"]
    client.app.state.evidence_bundles_cursor_secret = secrets.token_bytes(32)
    assert _list(client, limit=1, cursor=cursor).status_code == 422

    first_app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(first_app) as first_client:
        made = _create_many_bundles(first_client, 1)
        token = _list(first_client, limit=1).json()["next_cursor"]
    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as second_client:
        stale = _list(second_client, limit=1, cursor=token)
        assert stale.status_code == 422
        # The bundles themselves survive the restart and still list.
        fresh = _list(second_client).json()
        assert [i["id"] for i in fresh["items"]] == [made[0]["id"]]


# --- Parameter validation --------------------------------------------------------


def test_blank_filters_are_validation_errors(client):
    _setup_grouped(client)
    for field in ("claim_id", "evidence_type", "media_type", "digest_hex"):
        for value in ("", "   ", "\t"):
            resp = _list(client, **{field: value})
            assert resp.status_code == 422, (field, value)
            assert resp.json()["error"]["code"] == "validation_error"


def test_digest_hex_spelling_is_strict(client):
    good = "0" * 64
    for value in (
        "A" * 64,  # uppercase rejected, never lowercased
        "a" * 63,  # too short
        "a" * 65,  # too long
        ("0" * 63) + "g",  # non-hex
        f" {good}",  # whitespace padding never trimmed
        f"{good} ",
        "0x" + "0" * 62,
    ):
        resp = _list(client, digest_hex=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_illegal_limit_values_are_validation_errors(client):
    _setup_grouped(client)
    for value in ("0", "101", "-1", "1.5", "abc", "8.0", "  2", "+1", ""):
        resp = _list(client, limit=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_validation_errors(client):
    _setup_grouped(client)
    for suffix in (
        "claim_id=a&claim_id=b",
        "evidence_type=a&evidence_type=b",
        "media_type=a&media_type=b",
        "digest_hex=" + "0" * 64 + "&digest_hex=" + "1" * 64,
        "limit=1&limit=2",
        "cursor=x&cursor=y",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameters_are_validation_errors(client):
    _setup_grouped(client)
    for suffix in (
        "evidence_types=raw_capture",
        "claim=clm_x",
        "content_id=cnt_x",
        "evidence=" + "0" * 64,
        "digest=" + "0" * 64,
        "payload_digest_hex=" + "0" * 64,
        "limit=1&offset=2",
        "CURSOR=x",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_cursor_with_otherwise_valid_params_is_422(client):
    _setup_grouped(client)
    resp = _list(
        client, evidence_type="document", limit=10, cursor="garbage"
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_validation_failure_locates_the_query_field(client):
    _setup_grouped(client)
    resp = _list(client, limit="nope")
    assert resp.status_code == 422
    issue = resp.json()["error"]["details"]["issues"][0]
    assert issue["loc"] == ["query", "limit"]


# --- Read-only guarantee ----------------------------------------------------------


def test_queries_and_failures_write_no_bundles_or_audit_events(
    client, db_session
):
    _, _, _, _, bundles = _setup_grouped(client)

    def counts():
        return (
            db_session.scalar(
                select(func.count()).select_from(EvidenceBundle)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    bundles_before, events_before = counts()
    assert bundles_before == 5

    # Successful unfiltered, filtered, and paginated reads.
    cursor = None
    for _ in range(6):
        params = {"limit": 1}
        if cursor is not None:
            params["cursor"] = cursor
        resp = _list(client, **params)
        assert resp.status_code == 200, resp.text
        cursor = resp.json()["next_cursor"]
        if cursor is None:
            break
    _list(client)
    _list(client, evidence_type="audio")
    _list(client, media_type="never/existed")
    _list(client, digest_hex="0" * 64)
    _list(
        client,
        claim_id=bundles[0]["claim_id"],
        evidence_type="raw_capture",
    )

    # Failed reads must not write anything either.
    _list(client, claim_id=" ")
    _list(client, evidence_type=" ")
    _list(client, media_type="\t")
    _list(client, digest_hex="A" * 64)
    _list(client, limit=0)
    _list(client, limit=101)
    _list(client, cursor="tampered")
    _list(client, limit=1, cursor="cl1.x.y")
    client.get(f"{URL}?limit=1&limit=2")
    client.get(f"{URL}?unknown=1")
    _list(client, evidence_type="a", cursor="garbage")

    db_session.expire_all()
    assert counts() == (bundles_before, events_before)


# --- Compatibility with existing evidence-bundle routes --------------------------


def test_existing_evidence_bundle_routes_remain_unchanged(client):
    create_actor(client)
    content = _create_content(client, DIGEST_A)
    claim = _create_claim(client, content_id=content["id"])
    created = _create_bundle(
        client, claim["id"], "compat", digest=EVIDENCE_DIGEST_1
    )

    single = client.get(f"{URL}/{created['id']}")
    assert single.status_code == 200
    assert single.json() == created

    per_claim = client.get(f"/v1/claims/{claim['id']}/evidence-bundles")
    assert per_claim.status_code == 200
    assert per_claim.json() == {"items": [created], "count": 1}

    per_content = client.get(
        f"/v1/contents/{content['id']}/evidence-bundles"
    )
    assert per_content.status_code == 200
    assert per_content.json() == {
        "items": [created],
        "count": 1,
        "next_cursor": None,
    }

    # A repeat submission is still the idempotent 200, not a new bundle.
    repeat = client.post(
        URL,
        json={
            "claim_id": claim["id"],
            "evidence_type": "raw_capture",
            "digest_algorithm": "sha256",
            "digest_hex": EVIDENCE_DIGEST_1,
            "media_type": "image/jpeg",
            "metadata": {"name": "compat"},
        },
    )
    assert repeat.status_code == 200
    assert repeat.json() == created

    missing = client.get(f"{URL}/evb_{'0' * 64}")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "evidence_bundle_not_found"

    # Search results are unaffected by idempotent repeats: still one row.
    body = _list(client).json()
    assert body["count"] == 1
    assert body["items"] == [created]


def test_search_does_not_require_claim_existence(client):
    # Filtering is a pure value match: ids are never resolved, so a
    # non-existent claim is an empty collection (200), never a 404.
    assert _list(client, claim_id="clm_missing").status_code == 200
    assert _list(client, claim_id="clm_missing").json() == {
        "items": [],
        "count": 0,
        "next_cursor": None,
    }
