"""Tests for the optional reconciliation filters on the exchange-import
reconciliation listing.

Covers the optional, at-most-once ``local_available`` and ``matches`` query
parameters of
``GET /v1/evidence-bundle-exchange-import-reconciliations``. Each accepts
only the exact lowercase literals ``true``/``false``; absent means
unfiltered and the two combine as logical AND. Filtering applies to the
reconciliation computed at that read — every receipt still resolves its
local bundle solely by its own ``evidence_bundle_id`` under the existing
single-receipt rules — and never via any reverse lookup. The filtered
sequence keeps stable creation order, ``count`` is the filtered total, and
the opaque ``ir1`` cursor binds both effective filters (null when absent)
and the effective limit, so pages resume without duplication or omission
and the final/overrun cursor is null. Blank, repeated, capitalized,
illegal, or undeclared parameters and blank, malformed, tampered,
foreign-family, wrongly-claimed, or filter/limit-mismatching cursors are
``422 validation_error`` before any receipt is read; an empty match set is a
200 empty collection. Success and failure alike write no resource, receipt,
or audit rows and never echo a snapshot, signature, payload, or bytes. All
fixtures are deterministic and offline.
"""

from __future__ import annotations

import base64
import hashlib
import hmac as hmac_mod
import json
import secrets

from fastapi.testclient import TestClient

from provenance import pagination
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import AuditEvent, ExchangeImportRecord
from sqlalchemy import func, select
from tests.helpers import SEED_A
from tests.test_evidence_bundle_exchange import (
    _create_attestation,
    _create_bundle,
    _create_claim,
    _create_content,
    _setup_bundle,
)
from tests.test_evidence_bundle_exchange_import_reconciliations import (
    URL,
    _list,
    _offline_receipt,
    _register,
    _walk_pages,
)
from tests.test_evidence_bundle_exchange_import_reconciliation import (
    IMPORTS_URL,
)
from tests.test_evidence_bundle_exchange_imports import (
    _served_import_request,
)


# --- Fixture worlds --------------------------------------------------------------


def _mixed_world(client) -> dict[str, str]:
    """One receipt in each of the three reconciliation states.

    Creation order is offline, stale, matching:

    * ``offline``  — bundle id unknown locally: false/null/false.
    * ``stale``    — local bundle exists, attestation later changed the
      current snapshot digest: true/<current>/false.
    * ``matching``  — receipt taken from the now-current snapshot:
      true/<current>/true.
    """
    offline = _offline_receipt(client, "offline-filter")
    _, _, bundle = _setup_bundle(client)
    stale = _register(client, _served_import_request(client, bundle))
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    matching = _register(client, _served_import_request(client, bundle))
    return {"offline": offline["id"], "stale": stale["id"],
            "matching": matching["id"]}


def _paged_world(client) -> dict[str, list[str]]:
    """Four unavailable receipts plus one stale and one matching receipt."""
    offline = [
        _offline_receipt(client, f"offline-page-{i}")["id"] for i in range(4)
    ]
    _, _, bundle = _setup_bundle(client)
    stale = _register(client, _served_import_request(client, bundle))
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    matching = _register(client, _served_import_request(client, bundle))
    return {
        "offline": offline,
        "stale": [stale["id"]],
        "matching": [matching["id"]],
    }


def _ids(items) -> list[str]:
    return [item["id"] for item in items]


def _two_stale_world(client) -> list[str]:
    """Two different bundles whose served receipts both become stale.

    Both receipts are registered while their bundles carry no attestation;
    attesting each bundle afterwards changes its current snapshot digest, so
    the ``local_available=true&matches=false`` pair matches both receipts.
    """
    _, _, bundle_one = _setup_bundle(client)
    stale_one = _register(client, _served_import_request(client, bundle_one))
    content_two = _create_content(client, "second")
    claim_two = _create_claim(client, content_two["id"], "second-claim")
    bundle_two = _create_bundle(client, claim_two["id"], "second-evidence")
    stale_two = _register(client, _served_import_request(client, bundle_two))
    _create_attestation(
        client, "evidence_bundle", bundle_one["id"], seed=SEED_A,
        signer="org-1",
    )
    _create_attestation(
        client, "evidence_bundle", bundle_two["id"], seed=SEED_A,
        signer="org-1",
    )
    return [stale_one["id"], stale_two["id"]]


# --- Filter semantics -------------------------------------------------------------


def test_local_available_true_keeps_only_available(client):
    world = _mixed_world(client)
    body = _list(client, local_available="true").json()
    assert _ids(body["items"]) == [world["stale"], world["matching"]]
    assert body["count"] == 2
    assert all(item["local_available"] is True for item in body["items"])


def test_local_available_false_keeps_only_unavailable(client):
    world = _mixed_world(client)
    body = _list(client, local_available="false").json()
    assert _ids(body["items"]) == [world["offline"]]
    assert body["count"] == 1
    item = body["items"][0]
    assert item["local_available"] is False
    assert item["local_manifest_digest_hex"] is None
    assert item["matches"] is False


def test_matches_true_keeps_only_matching(client):
    world = _mixed_world(client)
    body = _list(client, matches="true").json()
    assert _ids(body["items"]) == [world["matching"]]
    assert body["count"] == 1
    assert body["items"][0]["matches"] is True


def test_matches_false_includes_unavailable_and_stale(client):
    # matches=false is purely the computed ``matches`` flag: an unavailable
    # receipt (whose matches is false) is included even though it has no
    # local bundle.
    world = _mixed_world(client)
    body = _list(client, matches="false").json()
    assert _ids(body["items"]) == [world["offline"], world["stale"]]
    assert body["count"] == 2
    assert all(item["matches"] is False for item in body["items"])


def test_filters_combine_as_logical_and(client):
    world = _mixed_world(client)
    cases = {
        ("true", "true"): [world["matching"]],
        ("true", "false"): [world["stale"]],
        ("false", "false"): [world["offline"]],
        # Unavailable receipts never match: this combination is well-formed
        # and simply empty.
        ("false", "true"): [],
    }
    for (available, matched), expected in cases.items():
        body = _list(
            client, local_available=available, matches=matched
        ).json()
        assert _ids(body["items"]) == expected, (available, matched)
        assert body["count"] == len(expected)
        assert body["next_cursor"] is None


def test_filtered_items_agree_with_single_reconciliation_route(client):
    world = _mixed_world(client)
    for kwargs in (
        {"local_available": "true"},
        {"local_available": "false"},
        {"matches": "true"},
        {"matches": "false"},
        {"local_available": "true", "matches": "false"},
    ):
        items = _list(client, **kwargs).json()["items"]
        for item in items:
            single = client.get(
                f"{IMPORTS_URL}/{item['id']}/reconciliation"
            ).json()
            assert (
                item["local_available"],
                item["local_manifest_digest_hex"],
                item["matches"],
            ) == (
                single["local_available"],
                single["local_manifest_digest_hex"],
                single["matches"]
            )


def test_filters_apply_to_results_computed_at_read_time(client):
    # The current digest is computed at read time: adding an attestation
    # after the import moves the receipt out of matches=true and into
    # matches=false without it ever leaving local_available=true.
    _, _, bundle = _setup_bundle(client)
    receipt = _register(client, _served_import_request(client, bundle))
    receipt_id = receipt["id"]

    assert _ids(_list(client, matches="true").json()["items"]) == [receipt_id]
    assert _list(client, matches="false").json()["count"] == 0

    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )

    assert _list(client, matches="true").json()["count"] == 0
    false_body = _list(client, matches="false").json()
    assert _ids(false_body["items"]) == [receipt_id]
    available_body = _list(client, local_available="true").json()
    assert _ids(available_body["items"]) == [receipt_id]
    assert _list(client, local_available="false").json()["count"] == 0


def test_filtered_sequence_preserves_stable_creation_order(client):
    world = _paged_world(client)
    # matches=false yields the four offline receipts then the stale one, in
    # creation order.
    body = _list(client, matches="false", limit=50).json()
    assert _ids(body["items"]) == world["offline"] + world["stale"]


def test_unfiltered_call_returns_every_receipt(client):
    world = _mixed_world(client)
    body = _list(client).json()
    assert _ids(body["items"]) == [
        world["offline"],
        world["stale"],
        world["matching"],
    ]
    assert body["count"] == 3


def test_filtered_response_shape_still_exactly_three_members(client):
    _mixed_world(client)
    body = _list(client, local_available="true").json()
    assert set(body) == {"items", "count", "next_cursor"}


# --- Pagination over the filtered set ---------------------------------------------


def test_filtered_pagination_concatenates_without_gaps_or_duplicates(client):
    world = _paged_world(client)
    all_items, pages, count = _walk_pages(client, matches="false", limit=2)
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    ids = _ids(all_items)
    assert len(ids) == len(set(ids)) == 5
    assert ids == world["offline"] + world["stale"]
    assert all(item["matches"] is False for item in all_items)


def test_available_filtered_pagination(client):
    world = _paged_world(client)
    all_items, pages, count = _walk_pages(
        client, local_available="true", limit=1
    )
    assert count == 2
    assert [len(page) for page in pages] == [1, 1]
    assert _ids(all_items) == world["stale"] + world["matching"]


def test_filtered_final_page_cursor_is_null(client):
    _paged_world(client)
    first = _list(client, matches="false", limit=2).json()
    assert first["count"] == 5
    assert first["next_cursor"] is not None
    second = _list(
        client, matches="false", limit=2, cursor=first["next_cursor"]
    ).json()
    assert second["count"] == 5
    assert len(second["items"]) == 2
    third = _list(
        client, matches="false", limit=2, cursor=second["next_cursor"]
    ).json()
    assert len(third["items"]) == 1
    assert third["next_cursor"] is None


def test_filtered_cursor_past_end_is_empty_with_filtered_count(client, app):
    _paged_world(client)
    token = pagination.encode_typed_cursor(
        app.state.exchange_import_reconciliations_cursor_secret,
        pagination.EXCHANGE_IMPORT_RECONCILIATIONS_CURSOR,
        {
            "local_available": None,
            "matches": False,
            "limit": 50,
            "offset": 99,
        },
    )
    body = _list(client, matches="false", cursor=token).json()
    assert body == {"items": [], "count": 5, "next_cursor": None}


def test_filtered_default_limit_is_fifty(client):
    _paged_world(client)
    # Fifty-five unavailable receipts, all matches=false; the default limit
    # pages 50 then 5 with the filter bound.
    for i in range(51):
        _offline_receipt(client, f"mass-{i:03d}")
    first = _list(client, local_available="false").json()
    assert len(first["items"]) == 50
    assert first["count"] == 55
    assert first["next_cursor"] is not None
    second = _list(
        client, local_available="false", cursor=first["next_cursor"]
    ).json()
    assert len(second["items"]) == 5
    assert second["count"] == 55
    assert second["next_cursor"] is None


def test_filtered_cursor_replays_same_page(client):
    _paged_world(client)
    cursor = _list(client, matches="false", limit=2).json()["next_cursor"]
    one = _list(client, matches="false", limit=2, cursor=cursor).json()
    two = _list(client, matches="false", limit=2, cursor=cursor).json()
    assert one == two


# --- Cursor binding to filters and limit -------------------------------------------


def test_cursor_binds_local_available_filter(client):
    _paged_world(client)
    cursor = _list(client, local_available="true", limit=1).json()["next_cursor"]
    # Same filter and limit resumes; dropping or flipping the filter (even
    # with the same limit) is a mismatch.
    assert (
        _list(client, local_available="true", limit=1, cursor=cursor).status_code
        == 200
    )
    assert _list(client, limit=1, cursor=cursor).status_code == 422
    assert (
        _list(
            client, local_available="false", limit=1, cursor=cursor
        ).status_code
        == 422
    )


def test_cursor_binds_matches_filter(client):
    _paged_world(client)
    cursor = _list(client, matches="false", limit=2).json()["next_cursor"]
    assert (
        _list(client, matches="false", limit=2, cursor=cursor).status_code
        == 200
    )
    assert _list(client, limit=2, cursor=cursor).status_code == 422
    assert (
        _list(client, matches="true", limit=2, cursor=cursor).status_code
        == 422
    )


def test_cursor_binds_both_filters(client):
    stale_ids = _two_stale_world(client)
    cursor = _list(
        client, local_available="true", matches="false", limit=1
    ).json()["next_cursor"]
    assert cursor is not None
    # The exact pair resumes and walks both stale receipts.
    resumed = _list(
        client,
        local_available="true",
        matches="false",
        limit=1,
        cursor=cursor,
    )
    assert resumed.status_code == 200
    assert _ids(resumed.json()["items"]) == [stale_ids[1]]
    # Dropping either bound filter or flipping the other dimension is a
    # mismatch.
    for params in (
        {"local_available": "true", "limit": 1},
        {"matches": "false", "limit": 1},
        {"local_available": "true", "matches": "true", "limit": 1},
        {"local_available": "false", "matches": "false", "limit": 1},
    ):
        resp = _list(client, cursor=cursor, **params)
        assert resp.status_code == 422, params


def test_unfiltered_cursor_rejected_when_filter_added(client):
    _paged_world(client)
    cursor = _list(client, limit=2).json()["next_cursor"]
    resp = _list(
        client, local_available="true", limit=2, cursor=cursor
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_filtered_cursor_rejected_with_limit_mismatch(client):
    _paged_world(client)
    cursor = _list(client, matches="false", limit=2).json()["next_cursor"]
    resp = _list(client, matches="false", limit=3, cursor=cursor)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    # Dropping the explicit limit changes the effective limit too.
    assert _list(client, matches="false", cursor=cursor).status_code == 422


def test_tampered_filtered_cursor_is_rejected(client):
    _paged_world(client)
    good = _list(client, matches="false", limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    resp = _list(client, matches="false", limit=1, cursor=tampered)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_foreign_family_cursor_rejected_alongside_filters(client):
    _paged_world(client)
    imports_cursor = client.get(
        IMPORTS_URL, params={"limit": 1}
    ).json()["next_cursor"]
    resp = _list(
        client,
        local_available="true",
        matches="true",
        limit=1,
        cursor=imports_cursor,
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def _hand_signed_ir1(app, payload_obj: dict) -> str:
    payload = base64.urlsafe_b64encode(
        json.dumps(payload_obj, separators=(",", ":")).encode("utf-8")
    ).rstrip(b"=").decode("ascii")
    sig = base64.urlsafe_b64encode(
        hmac_mod.new(
            app.state.exchange_import_reconciliations_cursor_secret,
            f"ir1.{payload}".encode("utf-8"),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode("ascii")
    return f"ir1.{payload}.{sig}"


def test_legacy_claim_set_cursor_is_rejected(client, app):
    _paged_world(client)
    # A correctly signed ir1 token carrying only the old limit/offset claim
    # set must not resume the now-filter-aware family.
    token = _hand_signed_ir1(app, {"limit": 1, "offset": 1})
    resp = _list(client, limit=1, cursor=token)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_wrongly_typed_filter_claims_are_rejected(client, app):
    _paged_world(client)
    for payload in (
        {"local_available": 1, "matches": None, "limit": 1, "offset": 1},
        {"local_available": "true", "matches": None, "limit": 1,
         "offset": 1},
        {"local_available": None, "matches": 0, "limit": 1, "offset": 1},
        {"local_available": True, "matches": None, "limit": 0, "offset": 1},
    ):
        token = _hand_signed_ir1(app, payload)
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422, payload


def test_cursor_signed_with_foreign_secret_is_rejected(client):
    _paged_world(client)
    token = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.EXCHANGE_IMPORT_RECONCILIATIONS_CURSOR,
        {
            "local_available": True,
            "matches": None,
            "limit": 1,
            "offset": 1,
        },
    )
    resp = _list(client, local_available="true", limit=1, cursor=token)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Parameter validation -----------------------------------------------------------


def test_illegal_bool_values_are_validation_errors(client):
    _paged_world(client)
    for value in ("True", "FALSE", "False", "1", "0", "yes", "t", "TRUE"):
        for field in ("local_available", "matches"):
            resp = _list(client, **{field: value})
            assert resp.status_code == 422, (field, value)
            assert resp.json()["error"]["code"] == "validation_error"


def test_blank_bool_values_are_validation_errors(client):
    _paged_world(client)
    for suffix in (
        "local_available=",
        "matches=",
        "local_available=%20",
        "matches=%09",
        "local_available=true%20",
        "%20local_available=true",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_filters_are_validation_errors(client):
    _paged_world(client)
    for suffix in (
        "local_available=true&local_available=false",
        "matches=true&matches=true",
        "local_available=true&local_available=true&matches=false",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameter_alongside_filter_is_validation_error(client):
    _paged_world(client)
    for suffix in (
        "local_available=true&unknown=1",
        "matches=false&manifest_digest_hex=" + "0" * 64,
        "local_available=true&offset=1",
        "Matches=true",
        "LOCAL_AVAILABLE=TRUE",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_illegal_filter_is_422_even_on_empty_database(client):
    # Parameter validation happens before any receipt is read: a bad filter
    # on an empty database is still 422, not an empty collection.
    for suffix in (
        "local_available=yes",
        "matches=",
        "local_available=true&matches=1",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix


def test_invalid_filter_and_illegal_limit_are_validation_errors(client):
    resp = _list(client, local_available="maybe", limit=0)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Empty results ------------------------------------------------------------------


def test_empty_database_with_filter_is_empty_collection(client):
    assert _list(client, local_available="true").json() == {
        "items": [],
        "count": 0,
        "next_cursor": None,
    }
    assert _list(client, matches="true").json() == {
        "items": [],
        "count": 0,
        "next_cursor": None,
    }
    assert _list(
        client, local_available="false", matches="true"
    ).json() == {"items": [], "count": 0, "next_cursor": None}


def test_impossible_combination_is_empty_but_total_counts_all(client):
    world = _mixed_world(client)
    body = _list(client, local_available="false", matches="true").json()
    assert body == {"items": [], "count": 0, "next_cursor": None}
    # The unfiltered collection is unaffected.
    assert _list(client).json()["count"] == 3
    assert world  # fixture sanity


# --- Read-only guarantee -------------------------------------------------------------


def test_filtered_reads_and_failures_write_nothing(client, db_session):
    _paged_world(client)

    def counts():
        return (
            db_session.scalar(
                select(func.count()).select_from(ExchangeImportRecord)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    receipts_before, events_before = counts()
    assert receipts_before == 6

    # Successful filtered reads, including full page walks.
    _walk_pages(client, matches="false", limit=2)
    _walk_pages(client, local_available="true", limit=1)
    _list(client, local_available="true", matches="false")
    _list(client, local_available="false", matches="true")

    # Failed reads must not write anything either.
    _list(client, local_available="yes")
    _list(client, matches="")
    client.get(f"{URL}?matches=true&matches=false")
    _list(client, local_available="true", limit=0)
    _list(client, matches="false", cursor="tampered")
    _list(client, local_available="true", cursor="ei1.x.y")
    client.get(f"{URL}?local_available=true&unknown=1")

    db_session.expire_all()
    assert counts() == (receipts_before, events_before)


def test_filtered_response_carries_no_snapshot_or_raw_material(client):
    _mixed_world(client)
    for params in (
        {"local_available": "true"},
        {"matches": "false"},
        {"local_available": "true", "matches": "true"},
    ):
        serialized = json.dumps(_list(client, **params).json())
        for forbidden in (
            '"snapshot"',
            '"manifest"',
            '"signature"',
            '"payload"',
            '"data"',
            '"evidence"',
            '"public_key"',
        ):
            assert forbidden not in serialized


# --- Determinism and compatibility ---------------------------------------------------


def test_filtered_listing_is_deterministic_across_restart(
    file_client, tmp_db_url
):
    _mixed_world(file_client)
    expected = {
        key: _list(file_client, **params).json()
        for key, params in (
            ("unfiltered", {}),
            ("available", {"local_available": "true"}),
            ("unavailable", {"local_available": "false"}),
            ("matching", {"matches": "true"}),
            ("non_matching", {"matches": "false"}),
            (
                "available_non_matching",
                {"local_available": "true", "matches": "false"},
            ),
        )
    }
    for body in expected.values():
        assert set(body) == {"items", "count", "next_cursor"}

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        for key, params in (
            ("unfiltered", {}),
            ("available", {"local_available": "true"}),
            ("unavailable", {"local_available": "false"}),
            ("matching", {"matches": "true"}),
            ("non_matching", {"matches": "false"}),
            (
                "available_non_matching",
                {"local_available": "true", "matches": "false"},
            ),
        ):
            resp = _list(client, **params)
            assert resp.status_code == 200, resp.text
            assert resp.json() == expected[key], key


def test_single_and_collection_routes_remain_compatible(client):
    world = _mixed_world(client)
    # The single-receipt reconciliation route is unchanged and stays the
    # reference for the filtered collection.
    for receipt_id in world.values():
        single = client.get(
            f"{IMPORTS_URL}/{receipt_id}/reconciliation"
        ).json()
        assert set(single) == {
            "import_id",
            "local_available",
            "local_manifest_digest_hex",
            "matches",
        }
    # The plain imports receipt list ignores the new reconciliation filters.
    imports = client.get(IMPORTS_URL, params={"local_available": "true"})
    assert imports.status_code == 422
    plain = client.get(IMPORTS_URL)
    assert plain.status_code == 200
    assert plain.json()["count"] == 3
