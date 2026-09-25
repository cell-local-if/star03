"""Tests for the read-only cross-content revocation-impact search.

Covers ``GET /v1/revocation-impacts``:

* the empty-body contract and the exact ``items``/``count``/``next_cursor``
  member order in compact UTF-8 JSON terminated by one newline, with
  non-ASCII emitted unescaped and integral numbers only (no negative zero);
* each item reusing the existing revocation public view and adding exactly
  the content association, target type, signing subject, and the
  before/after qualified-signer counts, delta, and coverage statuses,
  computed under the content evidence-coverage summary caliber -- the
  "before" counts ignore only the item's own revocation (every other
  revocation still applies), so the delta is always zero or one;
* the optional exact ``attestation_id``/``revoker_actor_id``/``reason``
  filters and the strict inclusive RFC 3339 UTC ``from``/``to`` bounds
  (an unknown or nonexistent value is an empty collection, never a 404;
  a ``from`` later than ``to`` is a 422);
* stable revocation creation ordering (``created_at`` with the persistent
  insertion-order tiebreaker that survives an app restart), a filtered
  ``count`` covering every page, pure-decimal ``limit`` validation, and the
  opaque HMAC-signed ``ri1`` cursor family;
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

from provenance.models import AuditEvent
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
    SEED_B,
)

SEARCH_PATH = "/v1/revocation-impacts"

DIGEST_D = hashlib.sha256(b"revocation-impact-d").hexdigest()
DIGEST_E = hashlib.sha256(b"revocation-impact-e").hexdigest()
EVIDENCE_DIGEST_1 = hashlib.sha256(b"impact-evidence-1").hexdigest()
EVIDENCE_DIGEST_2 = hashlib.sha256(b"impact-evidence-2").hexdigest()


# --- Fixture-style setup ------------------------------------------------------


def _make_content(client, digest, media_type="image/png", actor_id="org-1"):
    resp = client.post(
        "/v1/contents",
        json=content_payload(
            digest=digest, media_type=media_type, actor_id=actor_id
        ),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make_claim(client, content_id, actor_id="org-1", statement="s"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": "authorship",
            "payload": {"statement": statement},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make_bundle(client, claim_id, digest):
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


def _revoke(client, attestation_id, *, revoker_actor_id="org-1", reason="no longer relied upon"):
    resp = client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": attestation_id,
            "revoker_actor_id": revoker_actor_id,
            "reason": reason,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _world(client):
    """Five contents with distinct revocation-impact shapes.

    * c1: one claim, one proof (org-1), revoked -> covered/partial change,
      delta 1;
    * c2: one claim with two distinct signers (org-1, p-1), org-1's proof
      revoked -> after 1 / before 2, delta 1, stays covered;
    * c3: org-1 signs both the claim and a bundle, the claim proof revoked
      -> the bundle proof keeps the signer qualified, delta 0;
    * c4: a bundle-only proof by p-1 revoked (bundle -> claim -> content
      association), delta 1;
    * c5: one proof carrying two independent revocation records; each
      record's before-view still sees the other revocation, delta 0.
    """
    create_actor(client, actor_id="org-1", name="Example Org", type="organization")
    create_actor(client, actor_id="p-1", name="Alice", type="person")

    c1 = _make_content(client, DIGEST_A)
    cl1 = _make_claim(client, c1["id"], statement="one")
    a1 = _attest(client, "claim", cl1["id"])
    rev1 = _revoke(client, a1["id"], reason="first")

    c2 = _make_content(client, DIGEST_B, media_type="image/jpeg")
    cl2 = _make_claim(client, c2["id"], statement="two")
    a2a = _attest(client, "claim", cl2["id"])
    a2b = _attest(
        client, "claim", cl2["id"], seed=SEED_B, signer_actor_id="p-1"
    )
    rev2 = _revoke(client, a2a["id"], reason="second")

    c3 = _make_content(client, DIGEST_C)
    cl3 = _make_claim(client, c3["id"], statement="three")
    b3 = _make_bundle(client, cl3["id"], EVIDENCE_DIGEST_1)
    a3a = _attest(client, "claim", cl3["id"])
    a3b = _attest(client, "evidence_bundle", b3["id"])
    rev3 = _revoke(client, a3a["id"], reason="third")

    c4 = _make_content(client, DIGEST_D, media_type="image/jpeg")
    cl4 = _make_claim(client, c4["id"], statement="four", actor_id="p-1")
    b4 = _make_bundle(client, cl4["id"], EVIDENCE_DIGEST_2)
    a4 = _attest(
        client, "evidence_bundle", b4["id"], seed=SEED_B, signer_actor_id="p-1"
    )
    rev4 = _revoke(client, a4["id"], revoker_actor_id="p-1", reason="fourth")

    c5 = _make_content(client, DIGEST_E)
    cl5 = _make_claim(client, c5["id"], statement="five")
    a5 = _attest(client, "claim", cl5["id"])
    rev5a = _revoke(client, a5["id"], reason="fifth-a")
    rev5b = _revoke(client, a5["id"], revoker_actor_id="p-1", reason="fifth-b")

    return {
        "contents": [c1, c2, c3, c4, c5],
        "attestations": {
            "a1": a1,
            "a2a": a2a,
            "a2b": a2b,
            "a3a": a3a,
            "a3b": a3b,
            "a4": a4,
            "a5": a5,
        },
        "revocations": [rev1, rev2, rev3, rev4, rev5a, rev5b],
    }


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


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
    assert b'],"count":6,"next_cursor":null}' in raw
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        + b"\n"
    )
    assert raw == expected


def test_non_ascii_is_emitted_unescaped(client):
    create_actor(client)
    c = _make_content(client, DIGEST_D)
    cl = _make_claim(client, c["id"])
    att = _attest(client, "claim", cl["id"])
    _revoke(client, att["id"], reason="Képek ⛄")
    raw = client.get(SEARCH_PATH).content.decode("utf-8")
    assert "Képek ⛄" in raw
    assert "\\u" not in raw


def test_items_reuse_the_revocation_public_view(client):
    world = _world(client)
    items = client.get(SEARCH_PATH).json()["items"]
    for item, revocation in zip(items, world["revocations"], strict=True):
        detail = client.get(
            f"/v1/attestation-revocations/{revocation['id']}"
        ).json()
        for key in ("id", "attestation_id", "revoker_actor_id", "reason"):
            assert item[key] == detail[key]
        assert item["created_at"] == detail["created_at"]
        created_at = datetime.fromisoformat(item["created_at"])
        assert created_at.tzinfo is not None
        assert created_at.utcoffset().total_seconds() == 0


def test_items_carry_exactly_the_declared_fields(client):
    _world(client)
    item = client.get(SEARCH_PATH).json()["items"][0]
    assert list(item) == [
        "id",
        "attestation_id",
        "revoker_actor_id",
        "reason",
        "created_at",
        "content_id",
        "target_type",
        "signer_actor_id",
        "qualified_signer_count_after",
        "coverage_status_after",
        "qualified_signer_count_before",
        "coverage_status_before",
        "qualified_signer_count_delta",
    ]
    # No raw signature, public key, payload, ordering surrogate, or byte can
    # ever appear.
    rendered = client.get(SEARCH_PATH).content.decode()
    for forbidden in ("signature", "public_key", "payload", "seq", "digest_hex"):
        assert forbidden not in rendered


# --- Impact semantics ----------------------------------------------------------


def test_before_after_counts_delta_and_statuses(client):
    world = _world(client)
    c1, c2, c3, c4, c5 = world["contents"]
    items = {item["id"]: item for item in client.get(SEARCH_PATH).json()["items"]}

    rev1, rev2, rev3, rev4, rev5a, rev5b = world["revocations"]
    expected = {
        rev1["id"]: (c1["id"], "claim", "org-1", 0, "partial", 1, "covered", 1),
        rev2["id"]: (c2["id"], "claim", "org-1", 1, "covered", 2, "covered", 1),
        rev3["id"]: (c3["id"], "claim", "org-1", 1, "covered", 1, "covered", 0),
        rev4["id"]: (c4["id"], "evidence_bundle", "p-1", 0, "partial", 1, "covered", 1),
        rev5a["id"]: (c5["id"], "claim", "org-1", 0, "partial", 0, "partial", 0),
        rev5b["id"]: (c5["id"], "claim", "org-1", 0, "partial", 0, "partial", 0),
    }
    for revocation_id, (
        content_id,
        target_type,
        signer,
        after,
        status_after,
        before,
        status_before,
        delta,
    ) in expected.items():
        item = items[revocation_id]
        assert item["content_id"] == content_id
        assert item["target_type"] == target_type
        assert item["signer_actor_id"] == signer
        assert item["qualified_signer_count_after"] == after
        assert item["coverage_status_after"] == status_after
        assert item["qualified_signer_count_before"] == before
        assert item["coverage_status_before"] == status_before
        assert item["qualified_signer_count_delta"] == delta
        assert delta in (0, 1)
        assert (
            item["qualified_signer_count_before"]
            - item["qualified_signer_count_after"]
            == item["qualified_signer_count_delta"]
        )
        for key in (
            "qualified_signer_count_after",
            "qualified_signer_count_before",
            "qualified_signer_count_delta",
        ):
            assert isinstance(item[key], int)


def test_after_counts_match_the_single_content_coverage_summary(client):
    world = _world(client)
    items = {item["id"]: item for item in client.get(SEARCH_PATH).json()["items"]}
    seen = set()
    for item in items.values():
        if item["id"] in seen:
            continue
        summary = client.get(
            f"/v1/contents/{item['content_id']}/evidence-coverage"
        ).json()
        assert item["qualified_signer_count_after"] == summary[
            "qualified_signer_count"
        ]
        assert item["coverage_status_after"] == summary["coverage_status"]
        seen.add(item["id"])
    # Every world content appears at least once.
    assert {item["content_id"] for item in items.values()} == {
        content["id"] for content in world["contents"]
    }


def test_results_follow_stable_revocation_creation_order(client):
    world = _world(client)
    ids = [item["id"] for item in client.get(SEARCH_PATH).json()["items"]]
    assert ids == [rev["id"] for rev in world["revocations"]]


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
            rev["id"] for rev in world["revocations"]
        ]


# --- Filtering ------------------------------------------------------------------


def test_attestation_id_filter_is_an_exact_match(client):
    world = _world(client)
    body = client.get(
        SEARCH_PATH,
        params={"attestation_id": world["attestations"]["a5"]["id"]},
    ).json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [
        rev["id"] for rev in world["revocations"][-2:]
    ]


def test_revoker_actor_id_and_reason_filters(client):
    world = _world(client)
    body = client.get(
        SEARCH_PATH, params={"revoker_actor_id": "p-1"}
    ).json()
    assert [item["id"] for item in body["items"]] == [
        world["revocations"][3]["id"],
        world["revocations"][5]["id"],
    ]
    body = client.get(SEARCH_PATH, params={"reason": "first"}).json()
    assert [item["id"] for item in body["items"]] == [
        world["revocations"][0]["id"]
    ]


def test_filters_combine_as_logical_and(client):
    world = _world(client)
    body = client.get(
        SEARCH_PATH,
        params={"revoker_actor_id": "p-1", "reason": "fourth"},
    ).json()
    assert [item["id"] for item in body["items"]] == [
        world["revocations"][3]["id"]
    ]
    body = client.get(
        SEARCH_PATH,
        params={"revoker_actor_id": "p-1", "reason": "first"},
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}


def test_unknown_filter_values_are_empty_collections_not_404(client):
    _world(client)
    for params in (
        {"attestation_id": "att-does-not-exist"},
        {"revoker_actor_id": "no-such-actor"},
        {"reason": "no such reason"},
        {"revoker_actor_id": "P-1"},  # case-sensitive: no normalization
    ):
        resp = client.get(SEARCH_PATH, params=params)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_from_and_to_are_inclusive_rfc3339_utc_bounds(client):
    world = _world(client)
    first_at = world["revocations"][0]["created_at"]
    last_at = world["revocations"][-1]["created_at"]
    # Exact stored instants (inclusive) match; the "+00:00" and "Z" spellings
    # of the same instant are equivalent.
    z_spelling = datetime.fromisoformat(first_at).astimezone(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    body = client.get(SEARCH_PATH, params={"from": z_spelling}).json()
    assert body["count"] == 6
    body = client.get(SEARCH_PATH, params={"to": first_at}).json()
    assert body["count"] == 1
    assert body["items"][0]["id"] == world["revocations"][0]["id"]
    # Both bounds pinned to the first instant keep only that record.
    body = client.get(
        SEARCH_PATH, params={"from": first_at, "to": z_spelling}
    ).json()
    assert body["count"] == 1
    # A window ending before the first record and one starting after the
    # last are empty collections.
    body = client.get(
        SEARCH_PATH,
        params={"to": "2000-01-01T00:00:00Z"},
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}
    body = client.get(
        SEARCH_PATH,
        params={"from": "2999-01-01T00:00:00+00:00"},
    ).json()
    assert body == {"items": [], "count": 0, "next_cursor": None}
    assert last_at  # the bounds use the stored UTC instants


def test_from_later_than_to_is_422(client):
    _world(client)
    resp = client.get(
        SEARCH_PATH,
        params={
            "from": "2026-01-02T00:00:00Z",
            "to": "2026-01-01T00:00:00Z",
        },
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_malformed_time_bounds_are_422(client):
    _world(client)
    for field, value in (
        ("from", ""),
        ("from", " "),
        ("from", "2026-01-01"),  # date only
        ("from", "2026-01-01T00:00Z"),  # missing seconds
        ("from", "2026-01-01T00:00:00"),  # naive
        ("from", "2026-01-01T00:00:00z"),  # lowercase z
        ("from", "2026-01-01T01:00:00+01:00"),  # non-UTC offset
        ("from", "not-a-timestamp"),
        ("to", "2026-13-01T00:00:00Z"),  # calendar-invalid
    ):
        resp = client.get(SEARCH_PATH, params={field: value})
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "validation_error"


# --- Limit validation -----------------------------------------------------------


def test_limit_defaults_to_50(client):
    create_actor(client)
    # 60 independent content/claim/proof/revocation triples.
    for index in range(60):
        digest = hashlib.sha256(f"bulk-{index}".encode()).hexdigest()
        c = _make_content(client, digest)
        cl = _make_claim(client, c["id"], statement=f"bulk-{index}")
        att = _attest(client, "claim", cl["id"])
        _revoke(client, att["id"], reason=f"bulk-{index}")
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
    assert pages == 3
    assert ids == [rev["id"] for rev in world["revocations"]]
    assert len(set(ids)) == 6
    # count is the filtered total on every page, independent of the page.
    assert last["count"] == 6


def test_cursor_binds_the_effective_filters_and_limit(client):
    world = _world(client)
    first = client.get(SEARCH_PATH, params={"limit": "2"}).json()
    cursor = first["next_cursor"]
    assert cursor is not None
    # Replaying the cursor with the same conditions returns the same page.
    replayed = client.get(SEARCH_PATH, params={"limit": "2", "cursor": cursor})
    again = client.get(SEARCH_PATH, params={"limit": "2", "cursor": cursor})
    assert replayed.status_code == 200
    assert again.json() == replayed.json()
    # Any changed filter or limit is a 422, never a different query.
    for params in (
        {"limit": "3", "cursor": cursor},
        {"limit": "2", "attestation_id": world["attestations"]["a1"]["id"], "cursor": cursor},
        {"limit": "2", "revoker_actor_id": "org-1", "cursor": cursor},
        {"limit": "2", "reason": "first", "cursor": cursor},
        {
            "limit": "2",
            "from": "2000-01-01T00:00:00Z",
            "cursor": cursor,
        },
        {"limit": "2", "to": "2999-01-01T00:00:00Z", "cursor": cursor},
        {"cursor": cursor},  # the default limit differs from the minted one
    ):
        resp = client.get(SEARCH_PATH, params=params)
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "validation_error"


def test_filtered_cursor_resumes_the_filtered_set(client):
    world = _world(client)
    # org-1 recorded rev1, rev2, rev3, and rev5a (rev4 and rev5b are p-1's).
    expected_indices = [0, 1, 2, 4]
    ids, last, pages = _paged_ids(
        client, {"revoker_actor_id": "org-1", "limit": "2"}
    )
    assert pages == 2
    assert ids == [world["revocations"][i]["id"] for i in expected_indices]
    assert last["count"] == 4


def test_final_page_carries_no_cursor_and_impacts_stay_consistent(client):
    world = _world(client)
    first = client.get(SEARCH_PATH, params={"limit": "4"}).json()
    assert len(first["items"]) == 4
    second = client.get(
        SEARCH_PATH, params={"limit": "4", "cursor": first["next_cursor"]}
    ).json()
    assert len(second["items"]) == 2
    assert second["next_cursor"] is None
    assert [item["id"] for item in second["items"]] == [
        rev["id"] for rev in world["revocations"][4:]
    ]


def test_cursor_past_the_tail_is_an_empty_page_with_the_original_count(client):
    _world(client)
    first = client.get(SEARCH_PATH, params={"limit": "6"}).json()
    assert first["next_cursor"] is None
    # An exactly-full single page issues no cursor; walk a smaller limit and
    # then replay the final cursor to land at the tail, and once more past it.
    page1 = client.get(SEARCH_PATH, params={"limit": "3"}).json()
    page2 = client.get(
        SEARCH_PATH, params={"limit": "3", "cursor": page1["next_cursor"]}
    ).json()
    assert page2["next_cursor"] is None
    # A filtered set shorter than a page has no cursor at all.
    empty = client.get(
        SEARCH_PATH, params={"attestation_id": "att-nonexistent"}
    ).json()
    assert empty == {"items": [], "count": 0, "next_cursor": None}


def test_malformed_tampered_and_foreign_cursors_are_422(client):
    _world(client)
    valid = client.get(SEARCH_PATH, params={"limit": "2"}).json()["next_cursor"]
    # Cursor minted by the sibling revocation-list family (ar1).
    foreign_ar = client.get(
        "/v1/attestation-revocations", params={"limit": "2"}
    ).json()["next_cursor"]
    # Cursor minted by the coverage-search family (cc1).
    foreign_cc = client.get(
        "/v1/content-coverage-search", params={"limit": "2"}
    ).json()["next_cursor"]
    candidates = [
        "",
        " ",
        "not-a-cursor",
        "ri1.payload.signature",
        valid[:-2] + "xx",  # tampered signature
        valid.replace(".", "-", 1),  # tampered structure
        foreign_ar,  # cross-family cursor
        foreign_cc,  # another cross-family cursor
    ]
    for cursor in candidates:
        resp = client.get(SEARCH_PATH, params={"cursor": cursor})
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_time_claim_accepts_equivalent_utc_spellings(client):
    # The cursor binds the effective instant: a next page minted with "Z"
    # resumes with "+00:00" (and vice versa) because the claims canonicalize
    # to the same instant.
    _world(client)
    first = client.get(
        SEARCH_PATH,
        params={"limit": "2", "from": "2000-01-01T00:00:00Z"},
    ).json()
    cursor = first["next_cursor"]
    resp = client.get(
        SEARCH_PATH,
        params={
            "limit": "2",
            "from": "2000-01-01T00:00:00+00:00",
            "cursor": cursor,
        },
    )
    assert resp.status_code == 200, resp.text


# --- Parameter validation -------------------------------------------------------


def test_unknown_and_repeated_parameters_are_422(client):
    _world(client)
    resp = client.get(SEARCH_PATH, params={"actor_id": "org-1"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    for query in (
        "limit=1&limit=2",
        "attestation_id=a&attestation_id=b",
        "revoker_actor_id=org-1&revoker_actor_id=p-1",
        "reason=a&reason=b",
        "from=2026-01-01T00:00:00Z&from=2026-01-02T00:00:00Z",
        "to=2026-01-01T00:00:00Z&to=2026-01-02T00:00:00Z",
        "cursor=a&cursor=b",
    ):
        resp = client.get(f"{SEARCH_PATH}?{query}")
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "validation_error"


def test_blank_filters_are_422(client):
    _world(client)
    for field in ("attestation_id", "revoker_actor_id", "reason"):
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

    client.get(SEARCH_PATH)
    client.get(SEARCH_PATH, params={"revoker_actor_id": "org-1"})
    client.get(SEARCH_PATH, params={"attestation_id": "att-nonexistent"})
    client.get(SEARCH_PATH, params={"limit": "0"})  # a failing read too
    client.request("GET", SEARCH_PATH, content=b"{}")  # a rejected read

    assert _audit_count(db_session) == audits_before
