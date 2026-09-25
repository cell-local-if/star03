"""Tests for the read-only cross-content evidence-coverage search.

Covers ``GET /v1/content-coverage-search``:

* each item is the content public view plus the existing four coverage
  counts and the coverage status (identical to the per-content coverage
  read), and no other field;
* the exact ``items``/``count``/``next_cursor`` member order in compact
  UTF-8 JSON terminated by one newline, with unescaped non-ASCII;
* the optional exact ``actor_id``/``media_type`` filters (unknown or
  nonexistent values are an empty collection, never a 404) and the
  ``coverage_status`` filter accepting only the three existing literals;
* stable creation ordering (``created_at`` with the persistent
  insertion-order tiebreaker that survives an app restart);
* pure-decimal ``limit`` validation (1-100, default 50), the opaque
  HMAC-signed ``cc1`` cursor family binding every effective filter and the
  limit, resuming without duplication or omission, a null cursor on the
  final page, and an empty page with the original count at or past the
  tail;
* every 422 validation boundary (body, blank/illegal/repeated/undeclared
  parameters, blank/malformed/tampered/foreign/mismatching cursors) and
  the 405 rejection of PUT/PATCH/DELETE;
* the strictly read-only guarantee (no resource or audit writes).

All fixtures are deterministic and offline (in-memory and temporary-file
SQLite, signatures produced by the stdlib test signer).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from sqlalchemy import select

from provenance import pagination
from provenance.models import AuditEvent, Content
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    DIGEST_C,
    content_payload,
    create_actor,
)

SEARCH_PATH = "/v1/content-coverage-search"

DIGEST_D = hashlib.sha256(b"content-d").hexdigest()
EVIDENCE_DIGEST = hashlib.sha256(b"coverage-search-evidence").hexdigest()

ITEM_FIELDS = [
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


# --- Fixture-style setup ------------------------------------------------------


def _create_content(client, digest, media_type, actor_id, title=None):
    resp = client.post(
        "/v1/contents",
        json=content_payload(
            digest=digest, media_type=media_type, actor_id=actor_id, title=title
        ),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, actor_id="org-1"):
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


def _create_bundle(client, claim_id):
    resp = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim_id,
            "evidence_type": "raw_capture",
            "digest_algorithm": "sha256",
            "digest_hex": EVIDENCE_DIGEST,
            "media_type": "image/jpeg",
            "metadata": {"source": "camera-1"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _attest(client, target_type, target_id, signer_actor_id="org-1"):
    import base64

    from provenance.signing import attestation_message_bytes
    from tests.helpers import SEED_A, ed25519_public_key, ed25519_sign

    signature = ed25519_sign(
        SEED_A, attestation_message_bytes(target_type, target_id, signer_actor_id)
    )
    resp = client.post(
        "/v1/attestations",
        json={
            "target_type": target_type,
            "target_id": target_id,
            "signer_actor_id": signer_actor_id,
            "public_key": base64.b64encode(ed25519_public_key(SEED_A)).decode(
                "ascii"
            ),
            "signature": base64.b64encode(signature).decode("ascii"),
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _world(client):
    """Register one uncovered, one partial, and two covered contents, in order."""
    create_actor(client, actor_id="org-1", name="Example Org", type="organization")
    create_actor(client, actor_id="p-1", name="Alice", type="person")

    uncovered = _create_content(client, DIGEST_A, "image/png", "org-1")

    partial = _create_content(client, DIGEST_B, "image/jpeg", "p-1")
    _create_claim(client, partial["id"], actor_id="p-1")

    covered = _create_content(client, DIGEST_C, "image/png", "org-1")
    claim = _create_claim(client, covered["id"])
    _attest(client, "claim", claim["id"])

    covered_with_bundle = _create_content(client, DIGEST_D, "application/pdf", "p-1")
    claim = _create_claim(client, covered_with_bundle["id"], actor_id="p-1")
    bundle = _create_bundle(client, claim["id"])
    _attest(client, "evidence_bundle", bundle["id"])

    return [uncovered, partial, covered, covered_with_bundle]


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


def _content_ids(session):
    return {row.id for row in session.execute(select(Content.id)).all()}


# --- Empty collection and response shape --------------------------------------


def test_empty_registry_is_an_empty_collection(client):
    resp = client.get(SEARCH_PATH)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_response_is_compact_json_with_ordered_members_and_one_newline(client):
    contents = _world(client)
    resp = client.get(SEARCH_PATH)
    assert resp.status_code == 200
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    # The top-level members appear in exactly this order.
    assert raw.startswith(b'{"items":[')
    assert b'],"count":4,"next_cursor":null}' in raw
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        .encode("utf-8")
        + b"\n"
    )
    assert raw == expected
    assert [item["id"] for item in resp.json()["items"]] == [
        c["id"] for c in contents
    ]


def test_non_ascii_is_emitted_unescaped(client):
    create_actor(client)
    _create_content(client, DIGEST_D, "image/png", "org-1", title="Képek ⛄")
    raw = client.get(SEARCH_PATH).content.decode("utf-8")
    assert "Képek ⛄" in raw
    assert "\\u" not in raw


def test_items_carry_the_content_view_plus_the_coverage_summary(client):
    contents = _world(client)
    resp = client.get(SEARCH_PATH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 4
    assert body["next_cursor"] is None
    expected_statuses = ["uncovered", "partial", "covered", "covered"]
    for item, content, status_literal in zip(
        body["items"], contents, expected_statuses
    ):
        assert list(item) == ITEM_FIELDS
        assert item["coverage_status"] == status_literal
        for key in (
            "claim_count",
            "bundle_count",
            "attestation_count",
            "qualified_signer_count",
        ):
            assert isinstance(item[key], int)
        # The content public view is echoed exactly.
        for key in ITEM_FIELDS[:7]:
            assert item[key] == content[key]
        created_at = datetime.fromisoformat(item["created_at"])
        assert created_at.tzinfo is not None
        assert created_at.utcoffset().total_seconds() == 0
    # The counts and status match the per-content coverage read exactly.
    for item in body["items"]:
        per_content = client.get(
            f"/v1/contents/{item['id']}/evidence-coverage"
        ).json()
        assert item["claim_count"] == per_content["claim_count"]
        assert item["bundle_count"] == per_content["bundle_count"]
        assert item["attestation_count"] == per_content["attestation_count"]
        assert (
            item["qualified_signer_count"]
            == per_content["qualified_signer_count"]
        )
        assert item["coverage_status"] == per_content["coverage_status"]
    # No private, payload, signature, ordering, or byte field can appear.
    rendered = resp.content.decode()
    for forbidden in ("private", "signature", "payload", "public_key", "seq"):
        assert forbidden not in rendered


def test_counts_follow_the_existing_coverage_rules(client):
    create_actor(client)
    content = _create_content(client, DIGEST_A, "image/png", "org-1")
    claim = _create_claim(client, content["id"])
    bundle = _create_bundle(client, claim["id"])
    _attest(client, "evidence_bundle", bundle["id"])

    (item,) = client.get(SEARCH_PATH).json()["items"]
    assert item["claim_count"] == 1
    assert item["bundle_count"] == 1
    assert item["attestation_count"] == 1
    assert item["qualified_signer_count"] == 1
    assert item["coverage_status"] == "covered"


# --- Stable ordering -----------------------------------------------------------


def test_items_follow_stable_creation_order(client):
    contents = _world(client)
    body = client.get(SEARCH_PATH).json()
    assert [item["id"] for item in body["items"]] == [c["id"] for c in contents]
    created = [datetime.fromisoformat(item["created_at"]) for item in body["items"]]
    assert created == sorted(created)


def test_same_timestamp_ties_break_by_persistence_order(client, db_session):
    # Force every content onto one instant: the persistent insertion order
    # (seq) must still yield a stable, restart-durable order.
    contents = _world(client)
    tie = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    for row in db_session.execute(select(Content)).scalars():
        row.created_at = tie
    db_session.commit()

    body = client.get(SEARCH_PATH).json()
    assert [item["id"] for item in body["items"]] == [c["id"] for c in contents]


def test_ordering_and_results_are_identical_across_an_app_restart(
    tmp_db_url, file_client
):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    contents = _world(file_client)
    first = file_client.get(SEARCH_PATH)
    assert first.status_code == 200

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as restarted_client:
        second = restarted_client.get(SEARCH_PATH)
        assert second.status_code == 200
        assert second.json() == first.json()
        assert [item["id"] for item in second.json()["items"]] == [
            c["id"] for c in contents
        ]


# --- Filtering -----------------------------------------------------------------


def test_actor_and_media_type_filters_are_exact_and_combine(client):
    contents = _world(client)
    by_actor = client.get(SEARCH_PATH, params={"actor_id": "org-1"}).json()
    assert [item["id"] for item in by_actor["items"]] == [
        contents[0]["id"],
        contents[2]["id"],
    ]
    assert by_actor["count"] == 2

    by_media = client.get(SEARCH_PATH, params={"media_type": "image/png"}).json()
    assert [item["id"] for item in by_media["items"]] == [
        contents[0]["id"],
        contents[2]["id"],
    ]
    assert by_media["count"] == 2

    both = client.get(
        SEARCH_PATH, params={"actor_id": "org-1", "media_type": "image/png"}
    ).json()
    assert [item["id"] for item in both["items"]] == [
        contents[0]["id"],
        contents[2]["id"],
    ]
    assert both["count"] == 2

    # A valid actor paired with a media type it never registered matches nothing.
    miss = client.get(
        SEARCH_PATH, params={"actor_id": "org-1", "media_type": "image/jpeg"}
    ).json()
    assert miss == {"items": [], "count": 0, "next_cursor": None}


def test_filters_are_case_and_whitespace_sensitive(client):
    _world(client)
    for params in (
        {"actor_id": "Org-1"},
        {"actor_id": "org-1 "},
        {"media_type": "Image/png"},
        {"media_type": " image/png"},
    ):
        resp = client.get(SEARCH_PATH, params=params)
        assert resp.status_code == 200, params
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_unknown_or_nonexistent_filter_value_is_an_empty_collection(client):
    _world(client)
    for params in (
        {"actor_id": "ghost"},
        {"media_type": "video/mp4"},
        {"actor_id": "ghost", "media_type": "image/png"},
        {"actor_id": "ghost", "coverage_status": "covered"},
    ):
        resp = client.get(SEARCH_PATH, params=params)
        assert resp.status_code == 200, params
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_coverage_status_filter_matches_the_three_literals(client):
    contents = _world(client)
    uncovered = client.get(
        SEARCH_PATH, params={"coverage_status": "uncovered"}
    ).json()
    assert [item["id"] for item in uncovered["items"]] == [contents[0]["id"]]
    assert uncovered["count"] == 1

    partial = client.get(SEARCH_PATH, params={"coverage_status": "partial"}).json()
    assert [item["id"] for item in partial["items"]] == [contents[1]["id"]]
    assert partial["count"] == 1

    covered = client.get(SEARCH_PATH, params={"coverage_status": "covered"}).json()
    assert [item["id"] for item in covered["items"]] == [
        contents[2]["id"],
        contents[3]["id"],
    ]
    assert covered["count"] == 2

    # The status filter combines with the exact-match filters as logical AND.
    combined = client.get(
        SEARCH_PATH,
        params={"actor_id": "p-1", "coverage_status": "covered"},
    ).json()
    assert [item["id"] for item in combined["items"]] == [contents[3]["id"]]
    assert combined["count"] == 1


def test_coverage_status_rejects_any_other_spelling(client):
    _world(client)
    for bad in ("Covered", "COVERED", " covered", "covered ", "unknown", "", "   "):
        resp = client.get(SEARCH_PATH, params={"coverage_status": bad})
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_blank_filters_are_422(client):
    _world(client)
    for field in ("actor_id", "media_type"):
        for blank in ("", "   ", "\t"):
            resp = client.get(SEARCH_PATH, params={field: blank})
            assert resp.status_code == 422, (field, repr(blank))
            assert resp.json()["error"]["code"] == "validation_error"


# --- limit validation -----------------------------------------------------------


def test_limit_boundaries_one_and_one_hundred_are_accepted(client):
    _world(client)
    assert client.get(SEARCH_PATH, params={"limit": "1"}).status_code == 200
    assert client.get(SEARCH_PATH, params={"limit": "100"}).status_code == 200


def test_limit_must_be_a_pure_decimal_integer_in_range(client):
    _world(client)
    for bad in ("0", "101", "-1", "5.0", "5e0", " 5", "5 ", "five", "+5", ""):
        resp = client.get(SEARCH_PATH, params={"limit": bad})
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_default_limit_is_fifty(client):
    create_actor(client)
    for index in range(51):
        digest = hashlib.sha256(f"bulk-{index}".encode()).hexdigest()
        _create_content(client, digest, "image/png", "org-1")
    resp = client.get(SEARCH_PATH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 51
    assert len(body["items"]) == 50
    assert body["next_cursor"] is not None


# --- Parameter strictness -------------------------------------------------------


def test_repeated_and_undeclared_parameters_are_422(client):
    _world(client)
    assert (
        client.get(
            SEARCH_PATH,
            params=[("actor_id", "org-1"), ("actor_id", "p-1")],
        ).status_code
        == 422
    )
    assert (
        client.get(
            SEARCH_PATH, params=[("limit", "1"), ("limit", "2")]
        ).status_code
        == 422
    )
    assert (
        client.get(
            SEARCH_PATH,
            params=[("coverage_status", "covered"), ("coverage_status", "partial")],
        ).status_code
        == 422
    )
    for params in (
        {"actor": "org-1"},
        {"media_types": "image/png"},
        {"offset": "1"},
        {"content_id": "x"},
        {"digest_hex": "0" * 64},
        {"q": "x"},
    ):
        resp = client.get(SEARCH_PATH, params=params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_empty_body_is_422_rejected_before_any_read(client, db_session):
    _world(client)
    events_before = _audit_count(db_session)
    for body in (b"{}", b" ", b"{not valid json", b"[]", b"\n"):
        resp = client.request(
            "GET",
            SEARCH_PATH,
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"
    assert _audit_count(db_session) == events_before


# --- Pagination -----------------------------------------------------------------


def test_pages_resume_without_duplication_or_omission(client):
    contents = _world(client)
    first = client.get(SEARCH_PATH, params={"limit": "2"})
    assert first.status_code == 200
    page_one = first.json()
    assert page_one["count"] == 4
    assert [item["id"] for item in page_one["items"]] == [
        contents[0]["id"],
        contents[1]["id"],
    ]
    cursor = page_one["next_cursor"]
    assert cursor is not None
    assert cursor.startswith("cc1.")

    second = client.get(SEARCH_PATH, params={"limit": "2", "cursor": cursor})
    assert second.status_code == 200
    page_two = second.json()
    assert page_two["count"] == 4
    assert [item["id"] for item in page_two["items"]] == [
        contents[2]["id"],
        contents[3]["id"],
    ]
    assert page_two["next_cursor"] is None

    seen = [item["id"] for item in page_one["items"] + page_two["items"]]
    assert seen == [c["id"] for c in contents]


def test_count_covers_every_page_including_a_filtered_one(client):
    _world(client)
    page = client.get(
        SEARCH_PATH, params={"coverage_status": "covered", "limit": "1"}
    ).json()
    assert page["count"] == 2
    assert len(page["items"]) == 1
    cursor = page["next_cursor"]
    assert cursor is not None
    final = client.get(
        SEARCH_PATH,
        params={"coverage_status": "covered", "limit": "1", "cursor": cursor},
    ).json()
    assert final["count"] == 2
    assert len(final["items"]) == 1
    assert final["next_cursor"] is None


def test_replayed_cursor_returns_the_same_page(client):
    _world(client)
    first = client.get(SEARCH_PATH, params={"limit": "1"}).json()
    cursor = first["next_cursor"]
    assert cursor is not None
    page = client.get(SEARCH_PATH, params={"limit": "1", "cursor": cursor})
    replay = client.get(SEARCH_PATH, params={"limit": "1", "cursor": cursor})
    assert page.status_code == 200
    assert replay.status_code == 200
    assert replay.content == page.content


def test_cursor_at_or_past_the_tail_returns_empty_page_with_count(client, app):
    _world(client)

    def _cursor(offset):
        return pagination.encode_typed_cursor(
            app.state.content_coverage_search_cursor_secret,
            pagination.CONTENT_COVERAGE_SEARCH_CURSOR,
            {
                "actor_id": None,
                "media_type": None,
                "coverage_status": None,
                "limit": 50,
                "offset": offset,
            },
        )

    resp = client.get(SEARCH_PATH, params={"cursor": _cursor(4)})
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 4, "next_cursor": None}

    resp = client.get(SEARCH_PATH, params={"cursor": _cursor(99)})
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 4, "next_cursor": None}


def test_cursor_binds_every_effective_filter_and_limit(client):
    _world(client)
    cursor = client.get(SEARCH_PATH, params={"limit": "2"}).json()["next_cursor"]
    assert cursor is not None
    # A different limit, a new filter, or a dropped limit all mismatch.
    for params in (
        {"limit": "3", "cursor": cursor},
        {"media_type": "image/png", "limit": "2", "cursor": cursor},
        {"coverage_status": "covered", "limit": "2", "cursor": cursor},
        {"cursor": cursor},
    ):
        resp = client.get(SEARCH_PATH, params=params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"

    # A cursor minted under one filter cannot resume a different filter.
    png_cursor = client.get(
        SEARCH_PATH, params={"media_type": "image/png", "limit": "1"}
    ).json()["next_cursor"]
    assert png_cursor is not None
    for params in (
        {"media_type": "application/pdf", "limit": "1", "cursor": png_cursor},
        {"actor_id": "org-1", "limit": "1", "cursor": png_cursor},
        {"coverage_status": "covered", "limit": "1", "cursor": png_cursor},
    ):
        resp = client.get(SEARCH_PATH, params=params)
        assert resp.status_code == 422, params


def test_blank_malformed_and_tampered_cursors_are_422(client):
    _world(client)
    valid = client.get(SEARCH_PATH, params={"limit": "1"}).json()["next_cursor"]
    assert valid is not None
    tampered = valid[:-1] + ("A" if valid[-1] != "A" else "B")
    for bad in ("", "   ", "not-a-cursor", "cc1", "cc1.abc", tampered):
        resp = client.get(SEARCH_PATH, params={"cursor": bad})
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_foreign_family_cursor_is_422(client, app):
    _world(client)
    # A well-formed cursor minted by another endpoint family never resumes
    # this search.
    foreign = pagination.encode_typed_cursor(
        app.state.contents_cursor_secret,
        pagination.CONTENTS_CURSOR,
        {
            "actor_id": None,
            "digest_algorithm": None,
            "digest_hex": None,
            "media_type": None,
            "limit": 50,
            "offset": 1,
        },
    )
    resp = client.get(SEARCH_PATH, params={"cursor": foreign})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    claims_cursor = pagination.encode_typed_cursor(
        app.state.claims_cursor_secret,
        pagination.CLAIMS_CURSOR,
        {
            "content_id": None,
            "actor_id": None,
            "claim_type": None,
            "payload_digest_hex": None,
            "limit": 50,
            "offset": 1,
        },
    )
    resp = client.get(SEARCH_PATH, params={"cursor": claims_cursor})
    assert resp.status_code == 422


def test_cursor_with_wrong_claim_set_is_422(client, app):
    _world(client)
    # Correct family marker, HMAC, and base64, but a foreign claim payload.
    import base64
    import hmac

    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "actor_id": None,
                "media_type": None,
                "limit": 50,
                "offset": 1,
            },
            separators=(",", ":"),
        ).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(
        hmac.new(
            app.state.content_coverage_search_cursor_secret,
            f"cc1.{payload}".encode(),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode()
    resp = client.get(SEARCH_PATH, params={"cursor": f"cc1.{payload}.{sig}"})
    assert resp.status_code == 422


# --- Method boundary ------------------------------------------------------------


def test_put_patch_and_delete_are_405_method_not_allowed(client):
    _world(client)
    for method in (client.put, client.patch, client.delete):
        resp = method(SEARCH_PATH)
        assert resp.status_code == 405
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Read-only guarantee ---------------------------------------------------------


def test_queries_and_failures_write_nothing(client, db_session):
    contents = _world(client)
    events_before = _audit_count(db_session)
    ids_before = _content_ids(db_session)

    assert client.get(SEARCH_PATH).status_code == 200
    assert (
        client.get(SEARCH_PATH, params={"actor_id": "ghost"}).status_code == 200
    )
    assert (
        client.get(
            SEARCH_PATH, params={"coverage_status": "covered", "limit": "1"}
        ).status_code
        == 200
    )
    assert client.get(SEARCH_PATH, params={"limit": "0"}).status_code == 422
    assert client.get(SEARCH_PATH, params={"actor_id": " "}).status_code == 422
    assert (
        client.get(SEARCH_PATH, params={"coverage_status": "bogus"}).status_code
        == 422
    )
    assert client.get(SEARCH_PATH, params={"cursor": "bad"}).status_code == 422
    assert client.request("GET", SEARCH_PATH, content=b"{}").status_code == 422

    assert _content_ids(db_session) == ids_before == {c["id"] for c in contents}
    assert _audit_count(db_session) == events_before
