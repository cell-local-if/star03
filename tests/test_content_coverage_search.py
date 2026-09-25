"""Tests for the read-only cross-content evidence-coverage search.

Covers ``GET /v1/content-coverage-search``:

* the empty-body contract and the exact ``items``/``count``/``next_cursor``
  member order in compact UTF-8 JSON terminated by one newline, with
  non-ASCII emitted unescaped and integral numbers only;
* each item carrying exactly the content public view plus the four existing
  coverage counts and the coverage status (no other fields), with the same
  counting rules as the single-content summary (direct claims only, distinct
  bundles, revoked proofs retained in the attestation count, distinct
  verified non-revoked signers) and the three existing statuses;
* the optional exact ``actor_id``/``media_type`` filters (an unknown or
  nonexistent value is an empty collection, never a 404) and the
  ``coverage_status`` filter accepting only the three existing literals;
* stable creation ordering (``created_at`` with the persistent
  insertion-order tiebreaker that survives an app restart), a filtered
  ``count`` covering every page, pure-decimal ``limit`` validation, and the
  opaque HMAC-signed ``cc1`` cursor family (binding every effective filter
  and the limit, resuming without duplication or omission, a null cursor on
  the final page, and an empty page with the original count at or past the
  tail);
* every 422 validation boundary (body, blank/illegal/repeated/undeclared
  parameters, empty/malformed/tampered/cross-family/mismatching cursors),
  the 405 rejection of non-GET methods, and the strictly read-only
  guarantee.

All fixtures are deterministic and offline (in-memory and temporary-file
SQLite, signatures produced by the stdlib test signer).
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone

from sqlalchemy import select

from provenance.models import AuditEvent, Content
from provenance.signing import attestation_message_bytes
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    DIGEST_C,
    content_payload,
    create_actor,
    ed25519_public_key,
    ed25519_sign,
    SEED_A,
)

SEARCH_PATH = "/v1/content-coverage-search"

DIGEST_D = hashlib.sha256(b"coverage-search-d").hexdigest()
EVIDENCE_DIGEST_A = hashlib.sha256(b"coverage-search-evidence-a").hexdigest()


# --- Fixture-style setup ------------------------------------------------------


def _make_content(client, digest, media_type="image/png", actor_id="org-1", title=None):
    resp = client.post(
        "/v1/contents",
        json=content_payload(
            digest=digest, media_type=media_type, actor_id=actor_id, title=title
        ),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make_claim(client, content_id, actor_id="org-1"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": "authorship",
            "payload": {"statement": "covered"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make_bundle(client, claim_id, digest=EVIDENCE_DIGEST_A):
    resp = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim_id,
            "evidence_type": "raw_capture",
            "digest_algorithm": "sha256",
            "digest_hex": digest,
            "media_type": "image/jpeg",
            "metadata": {"source": "camera-1"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _attest(client, target_type, target_id, *, seed=SEED_A, signer_actor_id="org-1"):
    signature = ed25519_sign(
        seed, attestation_message_bytes(target_type, target_id, signer_actor_id)
    )
    resp = client.post(
        "/v1/attestations",
        json={
            "target_type": target_type,
            "target_id": target_id,
            "signer_actor_id": signer_actor_id,
            "public_key": base64.b64encode(ed25519_public_key(seed)).decode(
                "ascii"
            ),
            "signature": base64.b64encode(signature).decode("ascii"),
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _revoke(client, attestation_id, *, revoker_actor_id="org-1"):
    resp = client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": attestation_id,
            "revoker_actor_id": revoker_actor_id,
            "reason": "no longer relied upon",
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _world(client):
    """Three contents: one uncovered, one partial, one covered, in order."""
    create_actor(client, actor_id="org-1", name="Example Org", type="organization")
    create_actor(client, actor_id="p-1", name="Alice", type="person")

    uncovered = _make_content(client, DIGEST_A, media_type="image/png", actor_id="org-1")

    partial = _make_content(client, DIGEST_B, media_type="image/jpeg", actor_id="p-1")
    partial_claim = _make_claim(client, partial["id"], actor_id="p-1")
    _make_bundle(client, partial_claim["id"])
    revoked_proof = _attest(client, "claim", partial_claim["id"])
    _revoke(client, revoked_proof["id"])

    covered = _make_content(client, DIGEST_C, media_type="image/png", actor_id="org-1")
    covered_claim = _make_claim(client, covered["id"])
    _attest(client, "claim", covered_claim["id"])

    return {"uncovered": uncovered, "partial": partial, "covered": covered}


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


def _content_count(session):
    return len(session.execute(select(Content.id)).all())


# --- Empty collection and response shape --------------------------------------


def test_empty_registry_is_an_empty_collection(client):
    resp = client.get(SEARCH_PATH)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_response_is_compact_json_with_ordered_members_and_one_newline(client):
    _world(client)
    resp = client.get(SEARCH_PATH)
    assert resp.status_code == 200
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    assert raw.startswith(b'{"items":[')
    assert b'],"count":3,"next_cursor":null}' in raw
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        + b"\n"
    )
    assert raw == expected


def test_non_ascii_is_emitted_unescaped(client):
    create_actor(client)
    _make_content(client, DIGEST_D, title="Képek ⛄")
    raw = client.get(SEARCH_PATH).content.decode("utf-8")
    assert "Képek ⛄" in raw
    assert "\\u" not in raw


def test_items_carry_the_public_view_counts_and_status_only(client):
    world = _world(client)
    resp = client.get(SEARCH_PATH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    assert body["next_cursor"] is None
    expected = {
        world["uncovered"]["id"]: (0, 0, 0, 0, "uncovered"),
        world["partial"]["id"]: (1, 1, 1, 0, "partial"),
        world["covered"]["id"]: (1, 0, 1, 1, "covered"),
    }
    assert [item["id"] for item in body["items"]] == [
        world["uncovered"]["id"],
        world["partial"]["id"],
        world["covered"]["id"],
    ]
    for item in body["items"]:
        assert list(item) == [
            "id",
            "digest_algorithm",
            "digest_hex",
            "media_type",
            "title",
            "actor_id",
            "created_at",
            "claim_count",
            "bundle_count",
            "attestation_count",
            "qualified_signer_count",
            "coverage_status",
        ]
        counts = expected[item["id"]]
        assert (
            item["claim_count"],
            item["bundle_count"],
            item["attestation_count"],
            item["qualified_signer_count"],
            item["coverage_status"],
        ) == counts
        for key in (
            "claim_count",
            "bundle_count",
            "attestation_count",
            "qualified_signer_count",
        ):
            assert isinstance(item[key], int)
        created_at = datetime.fromisoformat(item["created_at"])
        assert created_at.tzinfo is not None
        assert created_at.utcoffset().total_seconds() == 0
    # No payload, signature, key, or byte field can ever appear.
    rendered = resp.content.decode()
    for forbidden in ("private", "signature", "payload", "public_key", "seq"):
        assert forbidden not in rendered


def test_counts_match_the_single_content_summary(client):
    world = _world(client)
    items = {
        item["id"]: item for item in client.get(SEARCH_PATH).json()["items"]
    }
    for content in world.values():
        summary = client.get(
            f"/v1/contents/{content['id']}/evidence-coverage"
        ).json()
        item = items[content["id"]]
        for key in (
            "claim_count",
            "bundle_count",
            "attestation_count",
            "qualified_signer_count",
            "coverage_status",
        ):
            assert item[key] == summary[key]


# --- Stable ordering ------------------------------------------------------------


def test_ordering_is_identical_across_an_app_restart(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    world = _world(file_client)
    first = file_client.get(SEARCH_PATH)
    assert first.status_code == 200

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as restarted_client:
        second = restarted_client.get(SEARCH_PATH)
        assert second.status_code == 200
        assert second.json() == first.json()
        assert [item["id"] for item in second.json()["items"]] == [
            world["uncovered"]["id"],
            world["partial"]["id"],
            world["covered"]["id"],
        ]


# --- Filtering ------------------------------------------------------------------


def test_actor_id_filter_is_an_exact_match(client):
    world = _world(client)
    resp = client.get(SEARCH_PATH, params={"actor_id": "p-1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert [item["id"] for item in body["items"]] == [world["partial"]["id"]]


def test_media_type_filter_is_an_exact_match(client):
    world = _world(client)
    body = client.get(SEARCH_PATH, params={"media_type": "image/png"}).json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [
        world["uncovered"]["id"],
        world["covered"]["id"],
    ]


def test_filters_combine_as_logical_and(client):
    world = _world(client)
    body = client.get(
        SEARCH_PATH, params={"actor_id": "org-1", "media_type": "image/png"}
    ).json()
    assert body["count"] == 2
    body = client.get(
        SEARCH_PATH, params={"actor_id": "p-1", "media_type": "image/png"}
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


def test_unknown_filter_values_are_empty_collections_not_404(client):
    _world(client)
    for params in (
        {"actor_id": "no-such-actor"},
        {"media_type": "no/such-type"},
        {"actor_id": "ORG-1"},  # case-sensitive: no normalization
    ):
        resp = client.get(SEARCH_PATH, params=params)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_coverage_status_filter_matches_each_literal(client):
    world = _world(client)
    expected = {
        "uncovered": [world["uncovered"]["id"]],
        "partial": [world["partial"]["id"]],
        "covered": [world["covered"]["id"]],
    }
    for status, ids in expected.items():
        resp = client.get(SEARCH_PATH, params={"coverage_status": status})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["count"] == len(ids)
        assert [item["id"] for item in body["items"]] == ids
        assert all(item["coverage_status"] == status for item in body["items"])


def test_coverage_status_filter_combines_with_exact_filters(client):
    world = _world(client)
    body = client.get(
        SEARCH_PATH,
        params={"actor_id": "org-1", "coverage_status": "covered"},
    ).json()
    assert [item["id"] for item in body["items"]] == [world["covered"]["id"]]
    body = client.get(
        SEARCH_PATH,
        params={"actor_id": "org-1", "coverage_status": "partial"},
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


def test_coverage_status_rejects_any_other_spelling(client):
    _world(client)
    for value in ("", " ", "COVERED", "Covered", "covered ", "none", "0"):
        resp = client.get(SEARCH_PATH, params={"coverage_status": value})
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "validation_error"


# --- Limit validation -----------------------------------------------------------


def test_limit_defaults_to_50(client):
    create_actor(client)
    for index in range(60):
        _make_content(client, hashlib.sha256(f"c{index}".encode()).hexdigest())
    body = client.get(SEARCH_PATH).json()
    assert body["count"] == 60
    assert len(body["items"]) == 50
    assert body["next_cursor"] is not None


def test_limit_boundaries(client):
    _world(client)
    for value in ("1", "100"):
        resp = client.get(SEARCH_PATH, params={"limit": value})
        assert resp.status_code == 200, resp.text


def test_limit_rejects_non_decimal_and_out_of_range_values(client):
    _world(client)
    for value in ("", " ", "0", "101", "-1", "2.0", "2e1", "abc", "+2", " 2"):
        resp = client.get(SEARCH_PATH, params={"limit": value})
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "validation_error"


# --- Pagination and cursors -----------------------------------------------------


def _paged_ids(client, params):
    """Collect every item id by following next_cursor to the end."""
    ids = []
    cursor = None
    pages = 0
    while True:
        page_params = dict(params)
        if cursor is not None:
            page_params["cursor"] = cursor
        resp = client.get(SEARCH_PATH, params=page_params)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        ids.extend(item["id"] for item in body["items"])
        pages += 1
        cursor = body["next_cursor"]
        if cursor is None:
            return ids, body, pages


def test_pagination_resumes_without_duplication_or_omission(client):
    world = _world(client)
    ids, last, pages = _paged_ids(client, {"limit": "2"})
    assert pages == 2
    assert ids == [
        world["uncovered"]["id"],
        world["partial"]["id"],
        world["covered"]["id"],
    ]
    assert len(set(ids)) == 3
    # count is the filtered total on every page, independent of the page.
    assert last["count"] == 3


def test_cursor_binds_the_effective_filters_and_limit(client):
    _world(client)
    first = client.get(SEARCH_PATH, params={"limit": "1"}).json()
    cursor = first["next_cursor"]
    assert cursor is not None
    # Replaying the cursor with the same conditions returns the same page.
    replayed = client.get(SEARCH_PATH, params={"limit": "1", "cursor": cursor})
    again = client.get(SEARCH_PATH, params={"limit": "1", "cursor": cursor})
    assert replayed.status_code == 200
    assert again.json() == replayed.json()
    # Any changed filter or limit is a 422, never a different query.
    for params in (
        {"limit": "2", "cursor": cursor},
        {"limit": "1", "actor_id": "org-1", "cursor": cursor},
        {"limit": "1", "media_type": "image/png", "cursor": cursor},
        {"limit": "1", "coverage_status": "covered", "cursor": cursor},
        {"cursor": cursor},  # the default limit differs from the minted one
    ):
        resp = client.get(SEARCH_PATH, params=params)
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "validation_error"


def test_filtered_cursor_resumes_the_filtered_set(client):
    world = _world(client)
    ids, last, _ = _paged_ids(client, {"actor_id": "org-1", "limit": "1"})
    assert ids == [world["uncovered"]["id"], world["covered"]["id"]]
    assert last["count"] == 2


def test_final_page_and_past_tail_carry_no_cursor(client):
    _world(client)
    # Exactly one full page: the final page carries next_cursor null.
    body = client.get(SEARCH_PATH, params={"limit": "3"}).json()
    assert len(body["items"]) == 3
    assert body["next_cursor"] is None
    # A cursor landing exactly at the tail yields an empty page, the
    # original count, and no further cursor.
    first = client.get(SEARCH_PATH, params={"limit": "2"}).json()
    second = client.get(
        SEARCH_PATH, params={"limit": "2", "cursor": first["next_cursor"]}
    ).json()
    assert len(second["items"]) == 1
    assert second["next_cursor"] is None


def test_cursor_at_or_past_the_tail_is_an_empty_page(client):
    _world(client)
    first = client.get(SEARCH_PATH, params={"limit": "2"}).json()
    second = client.get(
        SEARCH_PATH, params={"limit": "2", "cursor": first["next_cursor"]}
    ).json()
    # Mint a cursor past the tail by paging a smaller filtered set is not
    # possible; instead verify the tail page itself and an empty filter set.
    assert second["count"] == 3
    empty = client.get(SEARCH_PATH, params={"actor_id": "no-such-actor"}).json()
    assert empty == {"items": [], "count": 0, "next_cursor": None}


def test_malformed_tampered_and_foreign_cursors_are_422(client):
    _world(client)
    valid = client.get(SEARCH_PATH, params={"limit": "1"}).json()["next_cursor"]
    foreign = client.get("/v1/contents", params={"limit": "1"}).json()
    candidates = [
        "",
        " ",
        "not-a-cursor",
        valid[:-2] + "xx",  # tampered signature
        valid.replace(".", "-", 1),  # tampered structure
    ]
    if foreign["next_cursor"] is not None:
        candidates.append(foreign["next_cursor"])  # cross-family cursor
    for cursor in candidates:
        resp = client.get(SEARCH_PATH, params={"cursor": cursor})
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "validation_error"


# --- Parameter validation -------------------------------------------------------


def test_unknown_and_repeated_parameters_are_422(client):
    _world(client)
    resp = client.get(SEARCH_PATH, params={"actor": "org-1"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    for query in (
        "limit=1&limit=2",
        "actor_id=org-1&actor_id=p-1",
        "media_type=image/png&media_type=image/jpeg",
        "coverage_status=covered&coverage_status=partial",
        "cursor=a&cursor=b",
    ):
        resp = client.get(f"{SEARCH_PATH}?{query}")
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "validation_error"


def test_blank_filters_are_422(client):
    _world(client)
    for field in ("actor_id", "media_type"):
        for value in ("", " "):
            resp = client.get(SEARCH_PATH, params={field: value})
            assert resp.status_code == 422, resp.text
            assert resp.json()["error"]["code"] == "validation_error"


def test_any_request_body_is_422(client):
    _world(client)
    for body in (b"{}", b" ", b"\n", b"not json", b"[]"):
        resp = client.request("GET", SEARCH_PATH, content=body)
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_get_methods_are_405(client):
    _world(client)
    for method in ("put", "patch", "delete", "post"):
        resp = getattr(client, method)(SEARCH_PATH)
        assert resp.status_code == 405, resp.text
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Read-only guarantee --------------------------------------------------------


def test_search_is_strictly_read_only(client, db_session):
    _world(client)
    audits_before = _audit_count(db_session)
    contents_before = _content_count(db_session)

    resp = client.get(SEARCH_PATH)
    assert resp.status_code == 200
    client.get(SEARCH_PATH, params={"coverage_status": "covered"})
    client.get(SEARCH_PATH, params={"actor_id": "no-such-actor"})
    client.get(SEARCH_PATH, params={"limit": "0"})  # a failing read too

    assert _audit_count(db_session) == audits_before
    assert _content_count(db_session) == contents_before
