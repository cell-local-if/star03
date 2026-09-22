"""Tests for the read-only claim search endpoint.

Covers GET /v1/claims: the response shape carries exactly
{"items", "count", "next_cursor"} with each item exposing only the existing
claim public fields (never the raw payload), claims follow the global stable
creation order, timestamps stay UTC, content_id/actor_id/claim_type are
non-empty case- and whitespace-sensitive exact filters that combine as
logical AND, payload_digest_hex must be exactly 64 lowercase hexadecimal
characters, limit is 1..100 defaulting to 50, opaque HMAC cursors bind every
effective filter and the limit so pages concatenate without gaps or
duplicates and the final cursor is null, blank/illegal/repeated/undeclared
parameters and blank/tampered/malformed/foreign-family/query-mismatching
cursors are 422 validation_error, no match is an empty collection (an
unknown filter value is never a 404), and neither queries nor failures
write any resource, claim, or audit row. All fixtures are deterministic and
offline.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timedelta

from sqlalchemy import func, select

from provenance import pagination
from provenance.models import AuditEvent, Claim
from tests.helpers import DIGEST_A, DIGEST_B, content_payload, create_actor

PAYLOAD_1 = {"statement": "created by org-1", "confidence": 0.9}
PAYLOAD_2 = {"statement": "reviewed", "approved": True}
PAYLOAD_3 = {"statement": "third distinct payload"}


def _canonical_hex(payload) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


DIGEST_1 = _canonical_hex(PAYLOAD_1)
DIGEST_2 = _canonical_hex(PAYLOAD_2)
DIGEST_3 = _canonical_hex(PAYLOAD_3)


def _create_content(client, digest=DIGEST_A, actor_id="org-1"):
    resp = client.post(
        "/v1/contents", json=content_payload(actor_id=actor_id, digest=digest)
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


_UNSET = object()


def _create_claim(
    client,
    content_id,
    actor_id="org-1",
    claim_type="authorship",
    payload=PAYLOAD_1,
):
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


def _list(client, **params):
    return client.get("/v1/claims", params=params)


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


#: The exact public claim field set; there is deliberately no payload field.
PUBLIC_FIELDS = {
    "id",
    "content_id",
    "actor_id",
    "claim_type",
    "payload_digest_algorithm",
    "payload_digest_hex",
    "created_at",
}


def _setup_claims(client):
    """Create a known interleaved set of claims across contents and actors."""
    create_actor(client)  # org-1
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    content_a = _create_content(client, digest=DIGEST_A)
    content_b = _create_content(client, digest=DIGEST_B, actor_id="org-2")
    c1 = _create_claim(client, content_a["id"])
    c2 = _create_claim(
        client, content_a["id"], claim_type="endorsement", payload=PAYLOAD_2
    )
    c3 = _create_claim(
        client, content_b["id"], actor_id="org-2", claim_type="other",
        payload=PAYLOAD_3,
    )
    c4 = _create_claim(client, content_b["id"], actor_id="org-2")
    return content_a, content_b, [c1, c2, c3, c4]


# --- Response shape, fields, and ordering -------------------------------------


def test_response_shape_and_item_fields(client):
    _, _, created = _setup_claims(client)
    resp = _list(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"items", "count", "next_cursor"}
    assert body["count"] == 4
    assert body["next_cursor"] is None
    assert len(body["items"]) == 4
    for item in body["items"]:
        assert set(item) == PUBLIC_FIELDS
        # The raw payload can never be echoed: only its stored digest.
        assert "payload" not in item
    assert [item["id"] for item in body["items"]] == [c["id"] for c in created]


def test_items_match_existing_claim_public_view(client):
    _, _, created = _setup_claims(client)
    items = _list(client).json()["items"]
    for listed, made in zip(items, created):
        assert listed == made
        detail = client.get(f"/v1/claims/{made['id']}").json()
        assert listed == detail


def test_created_at_preserves_utc_timezone(client):
    _setup_claims(client)
    for item in _list(client).json()["items"]:
        parsed = datetime.fromisoformat(item["created_at"])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)
        assert item["created_at"].endswith(("Z", "+00:00"))


def test_global_stable_creation_order_across_contents(client):
    _, _, created = _setup_claims(client)
    items = _list(client).json()["items"]
    assert [item["id"] for item in items] == [c["id"] for c in created]
    created_ats = [datetime.fromisoformat(i["created_at"]) for i in items]
    assert created_ats == sorted(created_ats)


def test_empty_database_is_an_empty_collection(client):
    assert _list(client).json() == {"items": [], "count": 0, "next_cursor": None}


# --- Exact-match filtering -----------------------------------------------------


def test_filter_by_content_id(client):
    content_a, content_b, _ = _setup_claims(client)
    body = _list(client, content_id=content_a["id"]).json()
    assert body["count"] == 2
    assert {i["content_id"] for i in body["items"]} == {content_a["id"]}
    body_b = _list(client, content_id=content_b["id"]).json()
    assert body_b["count"] == 2
    assert {i["content_id"] for i in body_b["items"]} == {content_b["id"]}


def test_filter_by_actor_id(client):
    _, _, _ = _setup_claims(client)
    body = _list(client, actor_id="org-2").json()
    assert body["count"] == 2
    assert {i["actor_id"] for i in body["items"]} == {"org-2"}


def test_filter_by_claim_type(client):
    _, _, _ = _setup_claims(client)
    body = _list(client, claim_type="authorship").json()
    assert body["count"] == 2
    assert {i["claim_type"] for i in body["items"]} == {"authorship"}


def test_filter_by_payload_digest_hex(client):
    _, _, _ = _setup_claims(client)
    body = _list(client, payload_digest_hex=DIGEST_2).json()
    assert body["count"] == 1
    item = body["items"][0]
    assert item["payload_digest_hex"] == DIGEST_2
    assert item["claim_type"] == "endorsement"


def test_filters_combine_as_logical_and(client):
    content_a, _, _ = _setup_claims(client)
    body = _list(
        client,
        content_id=content_a["id"],
        actor_id="org-1",
        claim_type="endorsement",
        payload_digest_hex=DIGEST_2,
    ).json()
    assert [i["payload_digest_hex"] for i in body["items"]] == [DIGEST_2]
    assert body["count"] == 1

    # A combination no claim satisfies is an empty collection, not an error.
    body = _list(
        client,
        content_id=content_a["id"],
        claim_type="other",
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


def test_unknown_filter_value_is_empty_collection_not_not_found(client):
    _setup_claims(client)
    # Unlike GET /v1/contents/{id}/claims, a search never resolves a filter
    # value against resource existence: nothing matches -> empty 200 page.
    assert _list(client, content_id="cnt_ghost").json() == {
        "items": [],
        "count": 0,
        "next_cursor": None,
    }
    assert _list(client, actor_id="ghost-actor").json() == {
        "items": [],
        "count": 0,
        "next_cursor": None,
    }
    assert _list(
        client, payload_digest_hex="0" * 64
    ).json() == {"items": [], "count": 0, "next_cursor": None}


def test_exact_match_is_case_and_whitespace_sensitive(client):
    _setup_claims(client)
    for params in (
        {"actor_id": "ORG-1"},
        {"actor_id": "org-1 "},
        {"actor_id": " org-1"},
        {"claim_type": "AUTHORSHIP"},
        {"claim_type": "authorship\t"},
        {"content_id": "cnt_x "},
    ):
        body = _list(client, **params).json()
        assert body == {"items": [], "count": 0, "next_cursor": None}, params


def test_payload_digest_filter_is_case_sensitive(client):
    _setup_claims(client)
    # The stored digest is lowercase; the uppercase spelling is a 422, never
    # a normalized match.
    resp = _list(client, payload_digest_hex=DIGEST_1.upper())
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_payload_digest_malformed_values_are_validation_errors(client):
    _setup_claims(client)
    bad_values = [
        DIGEST_1.upper(),
        DIGEST_1[:-1],          # 63 chars
        DIGEST_1 + "ab",        # 66 chars
        "g" * 64,               # non-hex letters
        " " + DIGEST_1,         # leading whitespace
        DIGEST_1 + " ",         # trailing whitespace
        "0x" + "0" * 62,
        "",
        "   ",
        "\t",
    ]
    for value in bad_values:
        resp = _list(client, payload_digest_hex=value)
        assert resp.status_code == 422, repr(value)
        assert resp.json()["error"]["code"] == "validation_error"


# --- Pagination -----------------------------------------------------------------


def _setup_many_claims(client, count=7):
    """``count`` distinct claims on one content/actor in a fixed order."""
    create_actor(client)
    content = _create_content(client)
    made = []
    for i in range(count):
        payload = {"index": i, "marker": f"claim-{i}"}
        made.append(_create_claim(client, content["id"], payload=payload))
    return content, made


def test_pagination_concatenates_without_gaps_or_duplicates(client):
    _, made = _setup_many_claims(client, 7)
    all_items, pages, count = _walk_pages(client, limit=3)
    assert count == 7
    assert [len(page) for page in pages] == [3, 3, 1]
    ids = [i["id"] for i in all_items]
    assert len(ids) == len(set(ids)) == 7
    # Concatenated pages equal the unpaginated listing exactly.
    assert ids == [c["id"] for c in made]
    assert all_items == _list(client).json()["items"]


def test_count_is_filtered_total_on_every_page(client):
    _, made = _setup_many_claims(client, 7)
    all_items, pages, count = _walk_pages(
        client, payload_digest_hex=made[0]["payload_digest_hex"], limit=1
    )
    assert count == 1
    assert [len(page) for page in pages] == [1]
    assert [i["id"] for i in all_items] == [made[0]["id"]]


def test_last_page_cursor_null_on_exact_division(client):
    _setup_many_claims(client, 4)
    first = _list(client, limit=2).json()
    assert first["next_cursor"] is not None
    second = _list(client, limit=2, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 2
    assert second["count"] == 4
    assert second["next_cursor"] is None


def test_default_limit_is_fifty(client):
    _setup_many_claims(client, 55)
    first = _list(client).json()
    assert len(first["items"]) == 50
    assert first["count"] == 55
    assert first["next_cursor"] is not None
    second = _list(client, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 5
    assert second["count"] == 55
    assert second["next_cursor"] is None


def test_limit_boundaries_accepted(client):
    _setup_many_claims(client, 2)
    for value in (1, 100):
        assert _list(client, limit=value).status_code == 200


def test_reusing_a_cursor_replays_the_same_page(client):
    _setup_many_claims(client, 5)
    cursor = _list(client, limit=2).json()["next_cursor"]
    one = _list(client, limit=2, cursor=cursor).json()
    two = _list(client, limit=2, cursor=cursor).json()
    assert one == two


def test_cursor_past_end_returns_empty_page_with_total_count(client, app):
    _setup_many_claims(client, 3)
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
    assert body["count"] == 3
    assert body["next_cursor"] is None


# --- Cursor integrity ------------------------------------------------------------


def test_tampered_or_malformed_cursors_are_validation_errors(client):
    _setup_many_claims(client, 3)
    good = _list(client, limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    for token in (
        "",
        "   ",
        "not-a-cursor",
        "cl1.onlytwoparts",
        "cl1.too.many.parts",
        "cl0.x.y",
        "cl2.x.y",
        "v1.x.y",
        "ae1.x.y",
        "ce1.x.y",
        tampered,
    ):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422, repr(token)
        assert resp.json()["error"]["code"] == "validation_error"
        assert "items" not in resp.json()


def test_cursor_signed_with_foreign_secret_is_rejected(client, app):
    _setup_many_claims(client, 3)
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
    resp = _list(client, limit=1, cursor=foreign)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_from_other_families_is_rejected(client, app):
    _setup_many_claims(client, 3)
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
    for token in (audit_cursor, evidence_cursor):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


def test_claims_cursor_is_rejected_by_other_endpoints(client):
    _setup_many_claims(client, 3)
    claims_cursor = _list(client, limit=1).json()["next_cursor"]
    # A cross-interface cursor must not be accepted by another collection.
    resp = client.get(
        "/v1/audit-events", params={"limit": 1, "cursor": claims_cursor}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    resp = client.get(
        "/v1/evidence-bundle-exchange-imports",
        params={"limit": 1, "cursor": claims_cursor},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_bound_to_every_effective_query_parameter(client):
    _, made = _setup_many_claims(client, 5)
    cursor = _list(client, limit=2).json()["next_cursor"]

    # The cursor resumes only the exact query that minted it.
    mismatches = [
        {"limit": 3},
        {"actor_id": "org-1"},
        {"claim_type": "authorship"},
        {"content_id": "cnt_other"},
        {"payload_digest_hex": "0" * 64},
    ]
    for params in mismatches:
        resp = _list(client, cursor=cursor, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"

    # A filtered cursor resumed without (or with a changed) filter mismatches.
    filtered = _list(
        client, payload_digest_hex=made[0]["payload_digest_hex"], limit=2
    ).json()["next_cursor"]
    assert _list(client, limit=2, cursor=filtered).status_code == 422
    assert (
        _list(
            client,
            payload_digest_hex=made[1]["payload_digest_hex"],
            limit=2,
            cursor=filtered,
        ).status_code
        == 422
    )


# --- Parameter validation --------------------------------------------------------


def test_blank_filters_are_validation_errors(client):
    _setup_many_claims(client, 1)
    for field in ("content_id", "actor_id", "claim_type"):
        for value in ("", "   ", "\t"):
            resp = _list(client, **{field: value})
            assert resp.status_code == 422, (field, value)
            assert resp.json()["error"]["code"] == "validation_error"


def test_illegal_limit_values_are_validation_errors(client):
    _setup_many_claims(client, 1)
    for value in ("0", "101", "-1", "1.5", "abc", "8.0", "  2", ""):
        resp = _list(client, limit=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_validation_errors(client):
    _setup_many_claims(client, 1)
    digest = "a" * 64
    for suffix in (
        "actor_id=org-1&actor_id=org-2",
        "claim_type=a&claim_type=b",
        "content_id=x&content_id=y",
        f"payload_digest_hex={digest}&payload_digest_hex={'b' * 64}",
        "limit=1&limit=2",
        "cursor=x&cursor=y",
    ):
        resp = client.get(f"/v1/claims?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameters_are_validation_errors(client):
    _setup_many_claims(client, 1)
    for suffix in (
        "claim_types=authorship",
        "payload_digest=" + "a" * 64,
        "digest_hex=" + "a" * 64,
        "limit=1&offset=2",
        "CURSOR=x",
    ):
        resp = client.get(f"/v1/claims?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_invalid_cursor_with_otherwise_valid_params_is_422(client):
    _setup_many_claims(client, 1)
    resp = _list(client, actor_id="org-1", limit=10, cursor="garbage")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only guarantee ----------------------------------------------------------


def test_queries_and_failures_write_no_rows_or_audit_events(client, db_session):
    _, made = _setup_many_claims(client, 4)

    def counts():
        return (
            db_session.scalar(select(func.count()).select_from(Claim)),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    before = counts()

    # Successful filtered and paginated reads.
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
    _list(client, actor_id="nobody")
    _list(client, payload_digest_hex=made[0]["payload_digest_hex"])
    _list(client, content_id="cnt_missing")

    # Failed reads must not write anything either.
    _list(client, claim_type=" ")
    _list(client, limit=0)
    _list(client, cursor="tampered")
    _list(client, payload_digest_hex="ABCDEF")
    client.get("/v1/claims?limit=1&limit=2")
    client.get("/v1/claims?unknown=1")

    db_session.expire_all()
    assert counts() == before
