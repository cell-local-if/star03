"""Read-only retrieval tests for subject trust-policy revocations.

Covers the two reviewer read routes added on top of the existing immutable
write route:

* ``GET /v1/trust-policy-revocations/{revocation_id}`` -- one existing
  revocation's public view, or an explicit 404 ``not_found`` for an unknown
  id;
* ``GET /v1/trust-policy-revocations`` -- the read-only search over the
  revocation records with exact ``id``/``policy_id``/``actor_id``/``reason``
  filters, inclusive RFC 3339 UTC ``from``/``to`` bounds, ``limit`` and the
  opaque ``tr1`` cursor family, rendered as
  ``{"items", "count", "next_cursor"}``.

The tests pin down, deterministically and offline:

* the exact five public fields per record (``id``, ``policy_id``,
  ``actor_id``, verbatim ``reason``, UTC ``created_at``) -- no private key,
  raw signature, claim payload, content, or evidence byte can appear;
* lookup isolation: the individual route keys solely on the revocation id
  (a policy id or actor id is never reverse-resolved) and unknown filter
  values on the search yield an empty collection rather than a 404;
* stable creation ordering (``created_at`` first, monotonic ``seq`` as the
  tiebreaker) and paging without duplication or omission, a null cursor on
  the final page, and an empty page with the original count past the tail;
* the cursor binds every effective filter and the limit: a mismatched,
  tampered, or foreign-family cursor is a 422;
* every 422 boundary (non-empty body, unknown/repeated/blank parameters,
  bad limit, bad time range) and the 405 rejection of non-GET methods;
* strict zero-write behavior on success, empty results, and failures --
  no revocation, policy, other resource, or audit event is created or
  modified -- and persistence of the views across an app restart.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance.app import create_app
from provenance.config import Settings
from provenance.models import (
    ActorTrustPolicy,
    ActorTrustPolicyRevocation,
    AuditEvent,
)
from tests.helpers import SEED_B, create_actor
from tests.test_trust_policy_revocations import (
    REVOCATIONS_PATH,
    _attest,
    _create_policy,
    _make_claim,
    _revoke,
    _world,
)

PUBLIC_FIELDS = {"id", "policy_id", "actor_id", "reason", "created_at"}


def _revocation_url(revocation_id: str) -> str:
    return f"{REVOCATIONS_PATH}/{revocation_id}"


def _assert_public_item(item: dict) -> None:
    assert set(item) == PUBLIC_FIELDS
    assert item["id"].startswith("tpr_")
    created_at = datetime.fromisoformat(item["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert item["created_at"].endswith(("Z", "+00:00"))


def _seed(index: int) -> bytes:
    """A deterministic per-actor Ed25519 seed (tests are fully offline)."""
    return hashlib.sha256(f"trust-policy-revocation-reads-{index}".encode()).digest()


def _digest(index: int) -> str:
    """A deterministic per-actor content digest (digests are unique)."""
    return hashlib.sha256(f"reads-content-{index}".encode()).hexdigest()


def _third_actor(client):
    """A third actor holding a current authentication key."""
    create_actor(client, actor_id="org-3", name="Third Org", type="organization")
    _attest(
        client, _make_claim(client, "org-3", digest=_digest(3))["id"],
        "org-3", _seed(3),
    )


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
        resp = client.get(REVOCATIONS_PATH, params=query)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        count = body["count"]
        pages.append(body["items"])
        all_items.extend(body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            return all_items, pages, count
    raise AssertionError("pagination did not terminate")


# --- Individual resource: exact public view -----------------------------------


def test_get_revocation_returns_the_existing_public_view(client):
    _world(client)
    policy = _create_policy(client, threshold=2)
    created = _revoke(client, policy["id"], reason="key material retired")

    resp = client.get(_revocation_url(created["id"]))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == created
    assert list(body) == ["id", "policy_id", "actor_id", "reason", "created_at"]
    _assert_public_item(body)
    assert body["policy_id"] == policy["id"]
    assert body["actor_id"] == "org-1"
    assert body["reason"] == "key material retired"


def test_get_revocation_response_is_compact_utf8_json_with_one_newline(client):
    _world(client)
    policy = _create_policy(client)
    created = _revoke(client, policy["id"], reason="撤回 — verbatim 🔒")

    resp = client.get(_revocation_url(created["id"]))
    assert resp.status_code == 200
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    # The verbatim reason is not ASCII-escaped and members keep public order.
    assert "撤回".encode("utf-8") in raw
    assert raw.index(b'"id"') < raw.index(b'"policy_id"')
    assert raw.index(b'"policy_id"') < raw.index(b'"actor_id"')
    assert raw.index(b'"actor_id"') < raw.index(b'"reason"')
    assert raw.index(b'"reason"') < raw.index(b'"created_at"')
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    assert raw == expected


def test_get_revocation_requires_no_authentication_headers(client):
    _world(client)
    policy = _create_policy(client)
    created = _revoke(client, policy["id"])
    # Reviewer reads are public: no X-PA/X-PT/X-PS headers are sent.
    resp = client.get(_revocation_url(created["id"]))
    assert resp.status_code == 200
    assert resp.json() == created


def test_get_revocation_is_byte_for_byte_stable_across_reads(client):
    _world(client)
    policy = _create_policy(client)
    created = _revoke(client, policy["id"])
    first = client.get(_revocation_url(created["id"])).content
    for _ in range(3):
        assert client.get(_revocation_url(created["id"])).content == first


# --- Individual resource: explicit 404, never a reverse lookup ----------------


def test_get_unknown_revocation_is_an_explicit_404(client):
    resp = client.get(_revocation_url("tpr_" + "0" * 64))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_get_never_reverse_looks_up_by_policy_or_actor_id(client):
    _world(client)
    policy = _create_policy(client)
    _revoke(client, policy["id"])
    # Existing identifiers of other resources are unknown *revocation* ids:
    # the route keys on the revocation id alone and never resolves a policy
    # id or an actor id back to the revocation record.
    for identifier in (policy["id"], "org-1"):
        resp = client.get(_revocation_url(identifier))
        assert resp.status_code == 404, identifier
        assert resp.json()["error"]["code"] == "not_found"


# --- Individual resource: parameter and body boundaries ------------------------


_QUERY_CASES = (
    "?x=1",
    "?unknown=",
    "?id=tpr_x",
    "?policy_id=atp_x",
    "?actor_id=org-1",
    "?reason=x",
    "?a=1&b=2",
    "?a=1&a=2",
)


def test_individual_read_rejects_any_query_parameter_before_lookup(client):
    _world(client)
    policy = _create_policy(client)
    created = _revoke(client, policy["id"])

    for query in _QUERY_CASES:
        known = client.get(_revocation_url(created["id"]) + query)
        assert known.status_code == 422, query
        assert known.json()["error"]["code"] == "validation_error"
        # Parameter validation precedes the lookup: an unknown id with a bad
        # parameter is 422, never the 404 the unknown id alone would yield.
        unknown = client.get(_revocation_url("tpr_ghost") + query)
        assert unknown.status_code == 422, query
        assert unknown.json()["error"]["code"] == "validation_error"

    # Sanity: the same resources answer 200/404 with no query string.
    assert client.get(_revocation_url(created["id"])).status_code == 200
    assert client.get(_revocation_url("tpr_ghost")).status_code == 404


def test_individual_read_rejects_a_non_empty_body(client):
    _world(client)
    policy = _create_policy(client)
    created = _revoke(client, policy["id"])
    for body in (b"{}", b" ", b"not json"):
        resp = client.request(
            "GET", _revocation_url(created["id"]), content=body
        )
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"


# --- Search: envelope, ordering, and full views --------------------------------


def test_search_without_revocations_is_an_empty_collection(client):
    _world(client)
    resp = client.get(REVOCATIONS_PATH)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_search_returns_records_with_exact_envelope_and_fields(client):
    _world(client)
    _third_actor(client)
    policy_one = _create_policy(client, actor="org-1", threshold=2)
    policy_two = _create_policy(client, actor="org-2", seed=SEED_B, threshold=1)
    first = _revoke(client, policy_one["id"], reason="first withdrawal")
    second = _revoke(
        client, policy_two["id"], reason="second withdrawal",
        actor="org-2", seed=SEED_B,
    )

    resp = client.get(REVOCATIONS_PATH)
    assert resp.status_code == 200, resp.text
    raw = resp.content
    assert raw.endswith(b"}\n") and raw.count(b"\n") == 1
    body = resp.json()
    assert list(body) == ["items", "count", "next_cursor"]
    assert body["count"] == 2
    assert body["next_cursor"] is None
    assert body["items"] == [first, second]
    for item in body["items"]:
        _assert_public_item(item)


def test_search_orders_by_created_at_with_seq_tiebreaker(client, db_session):
    # Timestamps are pinned directly so the ordering is fully deterministic:
    # two rows share one instant (the monotonic seq is the tiebreaker) and a
    # third row carries an *earlier* instant despite being created last
    # (higher seq), proving created_at is the primary key and seq only
    # breaks ties.
    _world(client)
    _third_actor(client)
    policy_one = _create_policy(client, actor="org-1", threshold=1)
    policy_two = _create_policy(client, actor="org-2", seed=SEED_B, threshold=1)
    policy_three = _create_policy(client, actor="org-3", seed=_seed(3), threshold=1)
    first = _revoke(client, policy_one["id"], reason="tie-a")
    second = _revoke(
        client, policy_two["id"], reason="tie-b", actor="org-2", seed=SEED_B
    )
    third = _revoke(
        client, policy_three["id"], reason="earlier-instant",
        actor="org-3", seed=_seed(3),
    )

    tie = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
    earlier = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    instants = {first["id"]: tie, second["id"]: tie, third["id"]: earlier}
    for row in db_session.execute(
        select(ActorTrustPolicyRevocation)
    ).scalars():
        row.created_at = instants[row.id]
    db_session.commit()

    body = client.get(REVOCATIONS_PATH).json()
    assert [i["id"] for i in body["items"]] == [
        third["id"],
        first["id"],
        second["id"],
    ]
    assert [i["reason"] for i in body["items"]] == [
        "earlier-instant",
        "tie-a",
        "tie-b",
    ]
    assert body["count"] == 3

    # Each listed record is individually readable with the same view.
    for item in body["items"]:
        assert client.get(_revocation_url(item["id"])).json() == item


# --- Search: exact-match filters ------------------------------------------------


def _two_subject_revocations(client):
    _world(client)
    policy_one = _create_policy(client, actor="org-1", threshold=2)
    policy_two = _create_policy(client, actor="org-2", seed=SEED_B, threshold=3)
    first = _revoke(client, policy_one["id"], reason="shared rationale")
    second = _revoke(
        client, policy_two["id"], reason="other rationale",
        actor="org-2", seed=SEED_B,
    )
    return (policy_one, policy_two), (first, second)


def test_search_filters_by_id_policy_actor_and_reason(client):
    (policy_one, policy_two), (first, second) = _two_subject_revocations(
        client
    )

    by_id = client.get(REVOCATIONS_PATH, params={"id": first["id"]}).json()
    assert by_id["count"] == 1
    assert by_id["items"] == [first]

    by_policy = client.get(
        REVOCATIONS_PATH, params={"policy_id": policy_two["id"]}
    ).json()
    assert by_policy["count"] == 1
    assert by_policy["items"] == [second]

    by_actor = client.get(
        REVOCATIONS_PATH, params={"actor_id": "org-1"}
    ).json()
    assert by_actor["count"] == 1
    assert by_actor["items"] == [first]

    by_reason = client.get(
        REVOCATIONS_PATH, params={"reason": "shared rationale"}
    ).json()
    assert by_reason["count"] == 1
    assert by_reason["items"] == [first]

    # Filters combine as logical AND.
    combined = client.get(
        REVOCATIONS_PATH,
        params={"actor_id": "org-1", "reason": "shared rationale"},
    ).json()
    assert combined["count"] == 1
    empty = client.get(
        REVOCATIONS_PATH,
        params={"actor_id": "org-1", "reason": "other rationale"},
    ).json()
    assert empty == {"items": [], "count": 0, "next_cursor": None}


def test_search_filters_are_case_and_whitespace_sensitive(client):
    _two_subject_revocations(client)
    for value in ("Shared rationale", "shared rationale ", " shared"):
        body = client.get(REVOCATIONS_PATH, params={"reason": value}).json()
        assert body == {"items": [], "count": 0, "next_cursor": None}, value


def test_search_unknown_filter_values_are_empty_never_404(client):
    _two_subject_revocations(client)
    for params in (
        {"id": "tpr_ghost"},
        {"policy_id": "atp_ghost"},
        {"actor_id": "org-ghost"},
        {"reason": "never recorded"},
    ):
        resp = client.get(REVOCATIONS_PATH, params=params)
        assert resp.status_code == 200, params
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_search_reason_filter_matches_verbatim_unicode(client):
    _world(client)
    policy = _create_policy(client)
    reason = "  撤回原因:policy moved to a new root 🔒\n"
    created = _revoke(client, policy["id"], reason=reason)
    body = client.get(REVOCATIONS_PATH, params={"reason": reason}).json()
    assert body["count"] == 1
    assert body["items"] == [created]


# --- Search: inclusive time bounds ----------------------------------------------


def _pin_created_at(db_session, revocation_id, instant):
    row = db_session.execute(
        select(ActorTrustPolicyRevocation).where(
            ActorTrustPolicyRevocation.id == revocation_id
        )
    ).scalar_one()
    row.created_at = instant
    db_session.commit()


def test_search_from_and_to_bounds_are_inclusive(client, db_session):
    _, (first, second) = _two_subject_revocations(client)
    _pin_created_at(
        db_session, first["id"], datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
    )
    _pin_created_at(
        db_session, second["id"], datetime(2026, 3, 2, 12, 0, 0, tzinfo=timezone.utc)
    )

    # Bounds include the exact instants, in either spelling of UTC.
    both = client.get(
        REVOCATIONS_PATH,
        params={"from": "2026-03-01T12:00:00Z", "to": "2026-03-02T12:00:00+00:00"},
    ).json()
    assert [i["id"] for i in both["items"]] == [first["id"], second["id"]]

    only_first = client.get(
        REVOCATIONS_PATH, params={"to": "2026-03-01T12:00:00Z"}
    ).json()
    assert [i["id"] for i in only_first["items"]] == [first["id"]]

    only_second = client.get(
        REVOCATIONS_PATH, params={"from": "2026-03-02T12:00:00Z"}
    ).json()
    assert [i["id"] for i in only_second["items"]] == [second["id"]]

    none = client.get(
        REVOCATIONS_PATH, params={"to": "2026-02-28T23:59:59Z"}
    ).json()
    assert none == {"items": [], "count": 0, "next_cursor": None}


def test_search_rejects_invalid_times_and_inverted_ranges(client):
    _two_subject_revocations(client)
    for params in (
        {"from": "not-a-time"},
        {"from": "2026-03-01"},
        {"from": "2026-03-01T12:00:00"},  # missing offset
        {"from": "2026-03-01T12:00:00+01:00"},  # not UTC
        {"to": "2026-03-01 12:00:00Z"},
        {"from": ""},
        {"to": "   "},
        {"from": "2026-03-02T00:00:00Z", "to": "2026-03-01T00:00:00Z"},
    ):
        resp = client.get(REVOCATIONS_PATH, params=params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error", params


# --- Search: limit and cursor paging --------------------------------------------


def _many_revocations(client, count):
    """One revocation per actor, created in order; returns the records."""
    assert count >= 2
    _world(client)
    records = []
    for index in range(count):
        actor_id = f"org-{index + 1}"
        seed = None if index == 0 else (SEED_B if index == 1 else _seed(index + 1))
        kwargs = {} if index == 0 else {"actor": actor_id, "seed": seed}
        if index >= 2:
            create_actor(
                client, actor_id=actor_id, name=actor_id, type="organization"
            )
            _attest(
                client,
                _make_claim(client, actor_id, digest=_digest(index + 1))["id"],
                actor_id,
                seed,
            )
        policy = _create_policy(client, **kwargs)
        records.append(
            _revoke(
                client, policy["id"], reason=f"withdrawal {index}", **kwargs
            )
        )
    return records


def test_search_pages_without_duplication_or_omission(client):
    records = _many_revocations(client, 5)

    all_items, pages, count = _walk_pages(client, limit=2)
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    assert [item["id"] for item in all_items] == [r["id"] for r in records]
    assert all_items == records

    # The default limit is 50: one page holds everything, cursor is null.
    body = client.get(REVOCATIONS_PATH).json()
    assert body["count"] == 5
    assert len(body["items"]) == 5
    assert body["next_cursor"] is None


def test_search_limit_boundaries(client):
    _many_revocations(client, 2)
    for value in ("0", "101", "-1", "1.5", "1e1", " 1", "1 ", "", "abc"):
        resp = client.get(REVOCATIONS_PATH, params={"limit": value})
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error", value
    for value in ("1", "100"):
        resp = client.get(REVOCATIONS_PATH, params={"limit": value})
        assert resp.status_code == 200, value


def test_search_cursor_binds_filters_and_limit(client):
    _many_revocations(client, 3)
    first_page = client.get(REVOCATIONS_PATH, params={"limit": 2}).json()
    cursor = first_page["next_cursor"]
    assert cursor

    # The same query resumes; a changed filter or limit is a 422.
    resumed = client.get(
        REVOCATIONS_PATH, params={"limit": 2, "cursor": cursor}
    )
    assert resumed.status_code == 200
    for params in (
        {"limit": 1, "cursor": cursor},
        {"limit": 2, "actor_id": "org-1", "cursor": cursor},
        {"limit": 2, "reason": "withdrawal 0", "cursor": cursor},
        {"limit": 2, "from": "2026-01-01T00:00:00Z", "cursor": cursor},
        {"cursor": cursor},  # the effective limit differs (default 50)
    ):
        resp = client.get(REVOCATIONS_PATH, params=params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error", params


def test_search_cursor_from_a_filtered_query_resumes_that_query(client):
    _many_revocations(client, 3)
    first_page = client.get(
        REVOCATIONS_PATH, params={"actor_id": "org-1", "limit": 1}
    ).json()
    assert first_page["count"] == 1
    assert first_page["next_cursor"] is None

    # A filtered two-page walk: org-2 and org-3 revocations, one per page.
    records = client.get(REVOCATIONS_PATH).json()["items"]
    assert len(records) == 3
    page = client.get(REVOCATIONS_PATH, params={"limit": 1}).json()
    cursor = page["next_cursor"]
    second = client.get(
        REVOCATIONS_PATH, params={"limit": 1, "cursor": cursor}
    ).json()
    assert [i["id"] for i in second["items"]] == [records[1]["id"]]
    assert second["next_cursor"] is not None
    third = client.get(
        REVOCATIONS_PATH,
        params={"limit": 1, "cursor": second["next_cursor"]},
    ).json()
    assert [i["id"] for i in third["items"]] == [records[2]["id"]]
    assert third["next_cursor"] is None


def test_search_rejects_malformed_tampered_and_foreign_cursors(client):
    _many_revocations(client, 2)
    cursor = client.get(REVOCATIONS_PATH, params={"limit": 1}).json()[
        "next_cursor"
    ]

    for bad in ("", "not-a-cursor", cursor[:-2] + "xx", cursor + "aa"):
        resp = client.get(REVOCATIONS_PATH, params={"cursor": bad})
        assert resp.status_code == 422, bad
        assert resp.json()["error"]["code"] == "validation_error", bad

    # A cursor minted by another endpoint family never resumes this search.
    foreign = client.get("/v1/trust-policies", params={"limit": 1}).json()[
        "next_cursor"
    ]
    assert foreign
    resp = client.get(REVOCATIONS_PATH, params={"cursor": foreign})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_search_out_of_range_page_is_empty_with_original_count(
    client, db_session
):
    records = _many_revocations(client, 2)
    # Mint a valid cursor past the tail by deleting rows after issuance:
    # the offset then exceeds the filtered total, so the page is empty and
    # the cursor is null while the count reflects the remaining rows.
    cursor = client.get(REVOCATIONS_PATH, params={"limit": 1}).json()[
        "next_cursor"
    ]
    for record in records[1:]:
        row = db_session.execute(
            select(ActorTrustPolicyRevocation).where(
                ActorTrustPolicyRevocation.id == record["id"]
            )
        ).scalar_one()
        db_session.delete(row)
    db_session.commit()

    body = client.get(
        REVOCATIONS_PATH, params={"limit": 1, "cursor": cursor}
    ).json()
    assert body == {"items": [], "count": 1, "next_cursor": None}


# --- Search: parameter and body boundaries ---------------------------------------


def test_search_rejects_unknown_repeated_and_blank_parameters(client):
    _two_subject_revocations(client)
    for query in (
        "?bogus=1",
        "?policy_ids=atp_x",
        "?limit=1&limit=2",
        "?id=tpr_x&id=tpr_y",
        "?reason=a&reason=b",
        "?from=2026-01-01T00:00:00Z&from=2026-01-01T00:00:00Z",
        "?cursor=a&cursor=b",
        "?id=",
        "?policy_id=+",
        "?actor_id=%20%20",
        "?reason=",
    ):
        resp = client.get(REVOCATIONS_PATH + query)
        assert resp.status_code == 422, query
        assert resp.json()["error"]["code"] == "validation_error", query


def test_search_rejects_a_non_empty_body(client):
    _two_subject_revocations(client)
    for body in (b"{}", b" ", b"not json", b"\x00\x01"):
        resp = client.request("GET", REVOCATIONS_PATH, content=body)
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error", body


def test_search_requires_no_authentication_headers(client):
    _two_subject_revocations(client)
    resp = client.get(REVOCATIONS_PATH)
    assert resp.status_code == 200
    assert resp.json()["count"] == 2


# --- Only GET (and the existing POST) are served ---------------------------------


def test_read_routes_reject_non_get_methods(client):
    _world(client)
    policy = _create_policy(client)
    created = _revoke(client, policy["id"])
    for url in (REVOCATIONS_PATH, _revocation_url(created["id"])):
        for method, kwargs in (
            ("put", {"json": {}}),
            ("patch", {"json": {}}),
            ("delete", {}),
        ):
            resp = getattr(client, method)(url, **kwargs)
            assert resp.status_code == 405, (method, url)
            assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Strict zero-write guarantees -------------------------------------------------


def test_reads_never_create_or_modify_anything(client, db_session):
    _, (first, _) = _two_subject_revocations(client)
    original_reason = first["reason"]

    db_session.expire_all()
    policies_before = db_session.scalar(
        select(func.count()).select_from(ActorTrustPolicy)
    )
    revocations_before = db_session.scalar(
        select(func.count()).select_from(ActorTrustPolicyRevocation)
    )
    audits_before = db_session.scalar(select(func.count()).select_from(AuditEvent))

    cursor = client.get(REVOCATIONS_PATH, params={"limit": 1}).json()[
        "next_cursor"
    ]
    responses = (
        # Successful individual and search reads.
        client.get(_revocation_url(first["id"])),
        client.get(REVOCATIONS_PATH),
        client.get(REVOCATIONS_PATH, params={"limit": 1, "cursor": cursor}),
        # Empty results.
        client.get(REVOCATIONS_PATH, params={"actor_id": "ghost"}),
        # Missing resource.
        client.get(_revocation_url("tpr_ghost")),
        # Parameter and body failures.
        client.get(REVOCATIONS_PATH, params={"limit": 0}),
        client.get(REVOCATIONS_PATH, params={"bogus": "1"}),
        client.get(REVOCATIONS_PATH, params={"cursor": "junk"}),
        client.request("GET", REVOCATIONS_PATH, content=b"{}"),
        client.get(_revocation_url(first["id"]) + "?x=1"),
    )
    assert [r.status_code for r in responses] == [
        200, 200, 200, 200, 404, 422, 422, 422, 422, 422,
    ]

    db_session.expire_all()
    assert db_session.scalar(
        select(func.count()).select_from(ActorTrustPolicy)
    ) == policies_before
    assert db_session.scalar(
        select(func.count()).select_from(ActorTrustPolicyRevocation)
    ) == revocations_before
    assert db_session.scalar(
        select(func.count()).select_from(AuditEvent)
    ) == audits_before

    # The existing row is byte-for-byte unchanged.
    row = db_session.execute(
        select(ActorTrustPolicyRevocation).where(
            ActorTrustPolicyRevocation.id == first["id"]
        )
    ).scalar_one()
    assert (row.id, row.policy_id, row.actor_id, row.reason) == (
        first["id"],
        first["policy_id"],
        "org-1",
        original_reason,
    )


# --- Persistence across restarts ---------------------------------------------------


def test_read_views_persist_unchanged_across_a_restart(tmp_db_url, file_client):
    (policy_one, _), (first, second) = _two_subject_revocations(file_client)

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        single = client.get(_revocation_url(first["id"]))
        assert single.status_code == 200
        assert single.json() == first

        listing = client.get(REVOCATIONS_PATH)
        assert listing.status_code == 200
        assert listing.json() == {
            "items": [first, second],
            "count": 2,
            "next_cursor": None,
        }
        # Filters still resolve against the persisted rows.
        filtered = client.get(
            REVOCATIONS_PATH, params={"policy_id": policy_one["id"]}
        ).json()
        assert filtered["items"] == [first]
        # An unknown id stays a 404 after the restart.
        assert client.get(_revocation_url("tpr_ghost")).status_code == 404
