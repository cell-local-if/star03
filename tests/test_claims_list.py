"""Tests for the reviewer read-only claim search.

Covers GET /v1/claims: the response shape carries exactly
``{"items", "count", "next_cursor"}`` with each item exposing only the
existing public claim fields (never the raw payload), claims follow stable
creation order, and ``created_at`` stays UTC. The optional ``content_id``,
``actor_id``, and ``claim_type`` filters are non-empty, case- and
whitespace-sensitive exact matches that combine as logical AND (absent
means unfiltered); ``payload_digest_hex`` must be exactly 64 lowercase
hexadecimal characters. ``limit`` is 1..100 defaulting to 50; the opaque
HMAC cursor binds every effective filter and the limit so pages
concatenate without gaps or duplicates and the final cursor is null.
Blank/illegal/repeated/undeclared parameters and blank/malformed/
tampered/foreign-family/query-mismatching cursors are 422
validation_error; no match is an empty collection. Neither queries nor
failures write any claim, resource, or audit row, and the existing claim
creation/detail/per-content routes stay compatible. All fixtures are
deterministic and offline.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance import pagination
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import AuditEvent, Claim
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    create_actor,
)
from tests.test_claims import (
    PAYLOAD_1,
    PAYLOAD_2,
    _canonical_hex,
    _claim_payload,
    _create_claim,
    _create_content,
    _setup_content,
)

URL = "/v1/claims"

CLAIM_KEYS = {
    "id",
    "content_id",
    "actor_id",
    "claim_type",
    "payload_digest_algorithm",
    "payload_digest_hex",
    "created_at",
}

HEX1 = _canonical_hex(PAYLOAD_1)
HEX2 = _canonical_hex(PAYLOAD_2)


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


def _setup_grouped(client):
    """Five claims across two contents, two actors, two types, two payloads.

    Returns ``(content_a, content_b, claims_in_creation_order)``.
    """
    create_actor(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    content_a = _create_content(client, digest=DIGEST_A)
    content_b = _create_content(client, digest=DIGEST_B)
    claims = [
        _create_claim(
            client,
            content_id=content_a["id"],
            actor_id="org-1",
            claim_type="authorship",
            payload=PAYLOAD_1,
        ),
        _create_claim(
            client,
            content_id=content_a["id"],
            actor_id="org-2",
            claim_type="authorship",
            payload=PAYLOAD_1,
        ),
        _create_claim(
            client,
            content_id=content_a["id"],
            actor_id="org-1",
            claim_type="endorsement",
            payload=PAYLOAD_1,
        ),
        _create_claim(
            client,
            content_id=content_b["id"],
            actor_id="org-1",
            claim_type="authorship",
            payload=PAYLOAD_2,
        ),
        _create_claim(
            client,
            content_id=content_b["id"],
            actor_id="org-2",
            claim_type="endorsement",
            payload=PAYLOAD_2,
        ),
    ]
    return content_a, content_b, claims


def _create_many_claims(client, count: int):
    """Create ``count`` distinct claims on one content with distinct payloads."""
    content = _setup_content(client)
    created = []
    for i in range(count):
        created.append(
            _create_claim(
                client, content_id=content["id"], payload={"i": i, "k": "v"}
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
    _, _, claims = _setup_grouped(client)
    resp = _list(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"items", "count", "next_cursor"}
    assert body["count"] == 5
    assert body["next_cursor"] is None
    assert len(body["items"]) == 5
    for item in body["items"]:
        assert set(item) == CLAIM_KEYS
        assert item["id"].startswith("clm_")
        # The raw payload is never stored and must never be echoed.
        assert "payload" not in item


def test_item_is_the_public_claim_and_matches_the_single_read(client):
    _, _, claims = _setup_grouped(client)
    items = _list(client).json()["items"]
    for listed, created in zip(items, claims):
        assert listed == created
        assert listed == client.get(f"{URL}/{created['id']}").json()


def test_created_at_is_utc(client):
    _setup_grouped(client)
    for item in _list(client).json()["items"]:
        parsed = datetime.fromisoformat(item["created_at"])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)
        assert item["created_at"].endswith(("Z", "+00:00"))


def test_claims_follow_stable_creation_order(client):
    _, _, claims = _setup_grouped(client)
    items = _list(client).json()["items"]
    assert [i["id"] for i in items] == [c["id"] for c in claims]
    assert items == claims


def test_listing_order_matches_the_per_content_listing(client):
    content_a, _, _ = _setup_grouped(client)
    by_search = [
        i["id"] for i in _list(client, content_id=content_a["id"]).json()["items"]
    ]
    by_content = [
        i["id"]
        for i in client.get(f"/v1/contents/{content_a['id']}/claims")
        .json()["items"]
    ]
    assert by_search == by_content


def test_repeat_claim_submission_adds_no_search_position(client):
    content = _setup_content(client)
    first = _create_claim(client, content_id=content["id"])
    repeat = client.post("/v1/claims", json=_claim_payload(content_id=content["id"]))
    assert repeat.status_code == 200
    later = _create_claim(client, content_id=content["id"], payload=PAYLOAD_2)
    items = _list(client).json()["items"]
    assert [i["id"] for i in items] == [first["id"], later["id"]]


# --- Exact-match filtering -----------------------------------------------------


def test_content_id_exact_match(client):
    content_a, content_b, _ = _setup_grouped(client)
    body = _list(client, content_id=content_a["id"]).json()
    assert body["count"] == 3
    assert {i["content_id"] for i in body["items"]} == {content_a["id"]}
    body = _list(client, content_id=content_b["id"]).json()
    assert body["count"] == 2
    assert {i["content_id"] for i in body["items"]} == {content_b["id"]}


def test_actor_id_exact_match(client):
    _setup_grouped(client)
    body = _list(client, actor_id="org-2").json()
    assert body["count"] == 2
    assert {i["actor_id"] for i in body["items"]} == {"org-2"}


def test_claim_type_exact_match(client):
    _setup_grouped(client)
    body = _list(client, claim_type="endorsement").json()
    assert body["count"] == 2
    assert {i["claim_type"] for i in body["items"]} == {"endorsement"}


def test_payload_digest_hex_exact_match(client):
    _setup_grouped(client)
    body = _list(client, payload_digest_hex=HEX1).json()
    assert body["count"] == 3
    assert {i["payload_digest_hex"] for i in body["items"]} == {HEX1}
    body = _list(client, payload_digest_hex=HEX2).json()
    assert body["count"] == 2


def test_filters_combine_as_logical_and(client):
    content_a, content_b, claims = _setup_grouped(client)
    body = _list(
        client,
        content_id=content_a["id"],
        actor_id="org-1",
        claim_type="authorship",
        payload_digest_hex=HEX1,
    ).json()
    assert [i["id"] for i in body["items"]] == [claims[0]["id"]]
    assert body["count"] == 1

    # A content paired with a digest asserted only on the other content
    # satisfies nothing: the filters are ANDed, not coerced.
    body = _list(
        client,
        content_id=content_b["id"],
        payload_digest_hex=HEX1,
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


def _content_id_for_digest(client, digest):
    # Content ids are deterministic per digest; resolve via the per-actor
    # content listing rather than hard-coding the id scheme.
    for item in client.get("/v1/contents").json()["items"]:
        if item["digest_hex"] == digest:
            return item["id"]
    raise AssertionError(digest)


def test_exact_match_is_case_and_whitespace_sensitive(client):
    content_a, _, claims = _setup_grouped(client)
    digest = claims[0]["payload_digest_hex"]
    for params in (
        {"claim_type": "Authorship"},
        {"claim_type": " authorship"},
        {"actor_id": "ORG-1"},
        {"actor_id": "org-1 "},
        {"content_id": content_a["id"].swapcase()},
        {"content_id": f" {content_a['id']}"},
    ):
        body = _list(client, **params).json()
        assert body == {"items": [], "count": 0, "next_cursor": None}, params
    # The digest filter rejects rather than normalizes the uppercase form;
    # but a well-formed unrelated lowercase digest simply matches nothing.
    assert _list(client, payload_digest_hex="0" * 64).json() == {
        "items": [],
        "count": 0,
        "next_cursor": None,
    }


def test_unknown_filter_values_are_empty_collection_not_not_found(client):
    _setup_grouped(client)
    for params in (
        {"content_id": "cnt_" + "0" * 64},
        {"actor_id": "ghost-actor"},
        {"claim_type": "never-asserted"},
    ):
        assert _list(client, **params).json() == {
            "items": [],
            "count": 0,
            "next_cursor": None,
        }, params


# --- Pagination -----------------------------------------------------------------


def test_pagination_concatenates_without_gaps_or_duplicates(client):
    _, _, claims = _setup_grouped(client)
    all_items, pages, count = _walk_pages(client, limit=2)
    # 5 claims -> pages of 2, 2, 1.
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    ids = [i["id"] for i in all_items]
    assert len(ids) == len(set(ids)) == 5
    assert ids == [c["id"] for c in claims]
    assert all_items == _list(client).json()["items"]


def test_count_is_filtered_total_on_every_page(client):
    content_a, _, _ = _setup_grouped(client)
    all_items, pages, count = _walk_pages(
        client, content_id=content_a["id"], limit=2
    )
    assert count == 3
    assert [len(page) for page in pages] == [2, 1]
    assert {i["content_id"] for i in all_items} == {content_a["id"]}


def test_last_page_cursor_null_on_exact_division(client):
    _create_many_claims(client, 4)
    first = _list(client, limit=2).json()
    assert first["next_cursor"] is not None
    second = _list(client, limit=2, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 2
    assert second["count"] == 4
    assert second["next_cursor"] is None


def test_default_limit_is_fifty(client):
    _create_many_claims(client, 55)
    first = _list(client).json()
    assert len(first["items"]) == 50
    assert first["count"] == 55
    assert first["next_cursor"] is not None
    second = _list(client, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 5
    assert second["count"] == 55
    assert second["next_cursor"] is None


def test_limit_boundaries_accepted(client):
    _create_many_claims(client, 2)
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
        app.state.claims_cursor_secret,
        pagination.CLAIMS_CURSOR,
        {
            "content_id": None,
            "actor_id": None,
            "claim_type": None,
            "payload_digest_hex": None,
            "limit": 50,
            "offset": 99,
        },
    )
    body = _list(client, cursor=token).json()
    assert body["items"] == []
    assert body["count"] == 5
    assert body["next_cursor"] is None


# --- Cursor integrity ------------------------------------------------------------


def test_tampered_or_malformed_cursors_are_validation_errors(client):
    _setup_grouped(client)
    good = _list(client, limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    foreign = pagination.encode_typed_cursor(
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
    for token in (
        "",
        "   ",
        "not-a-cursor",
        "cl1.onlytwoparts",
        "cl1.too.many.parts",
        "cl0.x.y",
        "cl2.x.y",
        "v1.x.y",
        "ce1.x.y",
        "ae1.x.y",
        "ei1.x.y",
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
    evidence_cursor = pagination.encode_typed_cursor(
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
    exchange_cursor = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.EXCHANGE_IMPORTS_CURSOR,
        {
            "manifest_version": None,
            "evidence_bundle_id": None,
            "manifest_digest_hex": None,
            "limit": 1,
            "offset": 1,
        },
    )
    for token in (lineage_cursor, evidence_cursor, audit_cursor, exchange_cursor):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


def test_claims_cursor_is_rejected_by_other_endpoints(client):
    _setup_grouped(client)
    cursor = _list(client, limit=1).json()["next_cursor"]

    # The audit-events family uses its own marker and claim set.
    resp = client.get(
        "/v1/audit-events", params={"limit": 1, "cursor": cursor}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    # As does the exchange-import receipt listing.
    resp = client.get(
        "/v1/evidence-bundle-exchange-imports",
        params={"limit": 1, "cursor": cursor},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    # And the lineage route.
    resp = client.get(
        f"/v1/contents/{'cnt_' + '0' * 64}/lineage",
        params={"direction": "ancestors", "limit": 1, "cursor": cursor},
    )
    assert resp.status_code == 422


def test_cursor_bound_to_every_effective_query_parameter(client):
    content_a, _, claims = _setup_grouped(client)
    digest = claims[0]["payload_digest_hex"]

    cursor = _list(client, limit=2).json()["next_cursor"]
    # A filter/limit present in the resume request but absent from the
    # cursor mismatches.
    for params in (
        {"limit": 3},
        {"content_id": content_a["id"]},
        {"actor_id": "org-1"},
        {"claim_type": "authorship"},
        {"payload_digest_hex": digest},
    ):
        resp = _list(client, cursor=cursor, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"

    # A cursor minted under a filter cannot be resumed without it or with a
    # different value: no filter may be dropped or substituted.
    filtered = _list(client, content_id=content_a["id"], limit=2).json()[
        "next_cursor"
    ]
    assert _list(client, limit=2, cursor=filtered).status_code == 422
    content_b_id = _content_id_for_digest(client, DIGEST_B)
    assert (
        _list(
            client, content_id=content_b_id, limit=2, cursor=filtered
        ).status_code
        == 422
    )

    typed = _list(client, claim_type="endorsement", limit=2).json()["next_cursor"]
    assert _list(client, limit=2, cursor=typed).status_code == 422

    acted = _list(client, actor_id="org-2", limit=2).json()["next_cursor"]
    assert _list(client, limit=2, cursor=acted).status_code == 422

    digested = _list(
        client, payload_digest_hex=digest, limit=2
    ).json()["next_cursor"]
    assert _list(client, limit=2, cursor=digested).status_code == 422
    other_digest = "1" * 64
    assert (
        _list(
            client, payload_digest_hex=other_digest, limit=2, cursor=digested
        ).status_code
        == 422
    )


def test_cursor_secret_rotation_invalidates_outstanding_cursors(
    client, tmp_db_url
):
    # A restart rotates the per-process HMAC secret, so a cursor minted by
    # the old process is rejected rather than trusted; the stored claims
    # themselves survive.
    content = _setup_content(client)
    _create_claim(client, content_id=content["id"])
    cursor = _list(client, limit=1).json()["next_cursor"]
    client.app.state.claims_cursor_secret = secrets.token_bytes(32)
    assert _list(client, limit=1, cursor=cursor).status_code == 422

    first_app = create_app(Settings(database_url=tmp_db_url))
    with TestClient(first_app) as first_client:
        create_actor(first_client)
        content = _create_content(first_client)
        created = _create_claim(first_client, content_id=content["id"])
        token = _list(first_client, limit=1).json()["next_cursor"]
    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as second_client:
        stale = _list(second_client, limit=1, cursor=token)
        assert stale.status_code == 422
        # The claims themselves survive the restart and still list.
        fresh = _list(second_client).json()
        assert [i["id"] for i in fresh["items"]] == [created["id"]]


# --- Parameter validation --------------------------------------------------------


def test_blank_filters_are_validation_errors(client):
    _setup_grouped(client)
    for field in ("content_id", "actor_id", "claim_type", "payload_digest_hex"):
        for value in ("", "   ", "\t"):
            resp = _list(client, **{field: value})
            assert resp.status_code == 422, (field, value)
            assert resp.json()["error"]["code"] == "validation_error"


def test_payload_digest_hex_spelling_is_strict(client):
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
        resp = _list(client, payload_digest_hex=value)
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
        "content_id=a&content_id=b",
        "actor_id=a&actor_id=b",
        "claim_type=a&claim_type=b",
        "payload_digest_hex=" + "0" * 64 + "&payload_digest_hex=" + "1" * 64,
        "limit=1&limit=2",
        "cursor=x&cursor=y",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameters_are_validation_errors(client):
    _setup_grouped(client)
    for suffix in (
        "claim_id=clm_x",
        "claim_types=authorship",
        "actor=org-1",
        "digest_hex=" + "0" * 64,
        "payload_digest=" + "0" * 64,
        "limit=1&offset=2",
        "CURSOR=x",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_cursor_with_otherwise_valid_params_is_422(client):
    _setup_grouped(client)
    resp = _list(client, actor_id="org-1", limit=10, cursor="garbage")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_validation_failure_locates_the_query_field(client):
    _setup_grouped(client)
    resp = _list(client, limit="nope")
    assert resp.status_code == 422
    issue = resp.json()["error"]["details"]["issues"][0]
    assert issue["loc"] == ["query", "limit"]


# --- Read-only guarantee ----------------------------------------------------------


def test_queries_and_failures_write_no_claims_or_audit_events(
    client, db_session
):
    _, _, claims = _setup_grouped(client)

    def counts():
        return (
            db_session.scalar(select(func.count()).select_from(Claim)),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    claims_before, events_before = counts()
    assert claims_before == 5

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
    _list(client, actor_id="org-2")
    _list(client, claim_type="never-asserted")
    _list(client, payload_digest_hex="0" * 64)
    _list(client, content_id=claims[0]["content_id"], claim_type="endorsement")

    # Failed reads must not write anything either.
    _list(client, actor_id=" ")
    _list(client, payload_digest_hex="A" * 64)
    _list(client, limit=0)
    _list(client, limit=101)
    _list(client, cursor="tampered")
    _list(client, limit=1, cursor="ae1.x.y")
    client.get(f"{URL}?limit=1&limit=2")
    client.get(f"{URL}?unknown=1")
    _list(client, actor_id="a", cursor="garbage")

    db_session.expire_all()
    assert counts() == (claims_before, events_before)


# --- Compatibility with existing claim routes ------------------------------------


def test_existing_claim_routes_remain_unchanged(client):
    content = _setup_content(client)
    created = _create_claim(client, content_id=content["id"])

    single = client.get(f"{URL}/{created['id']}")
    assert single.status_code == 200
    assert single.json() == created

    per_content = client.get(f"/v1/contents/{content['id']}/claims")
    assert per_content.status_code == 200
    assert per_content.json() == {"items": [created], "count": 1}

    # A repeat submission is still the idempotent 200, not a new claim.
    repeat = client.post("/v1/claims", json=_claim_payload(content_id=content["id"]))
    assert repeat.status_code == 200
    assert repeat.json() == created

    missing = client.get(f"{URL}/clm_{'0' * 64}")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "claim_not_found"

    # Search results are unaffected by idempotent repeats: still one row.
    body = _list(client).json()
    assert body["count"] == 1
    assert body["items"] == [created]


def test_search_does_not_require_content_or_actor_existence(client):
    # Filtering is a pure value match: ids are never resolved, so a
    # non-existent reference is an empty collection (200), never a 404.
    assert _list(client, content_id="cnt_missing").status_code == 200
    assert _list(client, actor_id="ghost").status_code == 200
