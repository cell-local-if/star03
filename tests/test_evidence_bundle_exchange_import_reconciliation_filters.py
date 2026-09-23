"""Offline deterministic tests for the reconciliation listing filters.

Covers the optional, at-most-once ``local_available`` and ``matches`` query
filters on
``GET /v1/evidence-bundle-exchange-import-reconciliations``. Each accepts
only the lowercase literals ``true``/``false``; absent means unfiltered and
the two combine as logical AND against the reconciliation computed at read
time -- every receipt is looked up locally by its own
``evidence_bundle_id`` only and reconciled under the existing single-receipt
rules, never by a reverse digest/resource search. The success body stays
strictly ``{"items", "count", "next_cursor"}``: items are the filtered
receipts in the same stable creation order, ``count`` is the filtered total,
and the opaque ``ir1`` cursor binds both effective filters and the
effective limit so pages never repeat or omit a row. Unknown, repeated,
blank, illegal, or undeclared parameters and blank, tampered,
foreign-family, or filter/limit-mismatching cursors are
``422 validation_error`` before any receipt is read; an impossible
combination is a 200 empty collection. Reads and failures write no resource,
receipt, or audit row and never echo a snapshot, signature, payload, or
bytes. All fixtures are deterministic and offline.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance import pagination
from provenance.app import create_app
from provenance.config import Settings
from provenance.models import AuditEvent, ExchangeImportRecord
from tests.helpers import SEED_A
from tests.test_evidence_bundle_exchange import _create_attestation, _setup_bundle
from tests.test_evidence_bundle_exchange_import_reconciliations import (
    URL,
    _list,
    _offline_many,
    _offline_receipt,
    _register,
    _single_reconciliation,
    _walk_pages,
)
from tests.test_evidence_bundle_exchange_import_reconciliation import (
    IMPORTS_URL,
)
from tests.test_evidence_bundle_exchange_imports import (
    _served_import_request,
)
from tests.test_exchange_manifest_verifications import _digest_of


# --- Fixtures -------------------------------------------------------------------


def _stale_request(served: dict, title: str) -> dict:
    """A self-consistent import request for the served bundle's id with a
    different snapshot (and therefore a different, valid digest)."""
    request = json.loads(json.dumps(served))
    request["snapshot"]["content"]["title"] = title
    request["manifest"]["manifest_digest_hex"] = _digest_of(
        request["snapshot"]
    )
    return request


def _mixed(client) -> SimpleNamespace:
    """Five receipts in stable creation order:

    offline1 (unavailable / no match), fresh (available / matches), stale1
    (available / no match), offline2 (unavailable / no match), stale2
    (available / no match).
    """
    offline1 = _offline_receipt(client, "offline-1")
    _, _, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    fresh = _register(client, _served_import_request(client, bundle))
    served = _served_import_request(client, bundle)
    stale1 = _register(client, _stale_request(served, "superseded 一"))
    offline2 = _offline_receipt(client, "offline-2")
    stale2 = _register(client, _stale_request(served, "superseded 二"))
    return SimpleNamespace(
        offline=[offline1, offline2],
        fresh=fresh,
        stale=[stale1, stale2],
        bundle=bundle,
    )


def _ids(receipts) -> list[str]:
    return [receipt["id"] for receipt in receipts]


# --- Filtering ------------------------------------------------------------------


def test_empty_database_with_each_filter_is_empty_collection(client):
    for params in (
        {"local_available": "true"},
        {"local_available": "false"},
        {"matches": "true"},
        {"matches": "false"},
        {"local_available": "false", "matches": "true"},
    ):
        assert _list(client, **params).json() == {
            "items": [],
            "count": 0,
            "next_cursor": None,
        }, params


def test_local_available_true_returns_only_locally_available(client):
    mixed = _mixed(client)
    body = _list(client, local_available="true").json()
    assert body["count"] == 3
    assert body["next_cursor"] is None
    assert [item["id"] for item in body["items"]] == _ids(
        [mixed.fresh, *mixed.stale]
    )
    assert all(item["local_available"] is True for item in body["items"])


def test_local_available_false_returns_only_unavailable(client):
    mixed = _mixed(client)
    body = _list(client, local_available="false").json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == _ids(mixed.offline)
    for item in body["items"]:
        assert item["local_available"] is False
        assert item["local_manifest_digest_hex"] is None
        assert item["matches"] is False


def test_matches_true_returns_only_currently_matching(client):
    mixed = _mixed(client)
    body = _list(client, matches="true").json()
    assert body["count"] == 1
    assert [item["id"] for item in body["items"]] == [mixed.fresh["id"]]
    assert body["items"][0]["matches"] is True


def test_matches_false_includes_unavailable_and_diverged(client):
    mixed = _mixed(client)
    # matches=false is not the negation of local_available: it keeps both
    # unavailable receipts and available-but-diverged receipts, in stable
    # creation order.
    body = _list(client, matches="false").json()
    assert body["count"] == 4
    assert [item["id"] for item in body["items"]] == _ids(
        [mixed.offline[0], mixed.stale[0], mixed.offline[1], mixed.stale[1]]
    )
    assert all(item["matches"] is False for item in body["items"])


def test_combined_filters_intersect(client):
    mixed = _mixed(client)
    available_matching = _list(
        client, local_available="true", matches="true"
    ).json()
    assert [item["id"] for item in available_matching["items"]] == [
        mixed.fresh["id"]
    ]
    assert available_matching["count"] == 1

    available_diverged = _list(
        client, local_available="true", matches="false"
    ).json()
    assert [item["id"] for item in available_diverged["items"]] == _ids(
        mixed.stale
    )
    assert available_diverged["count"] == 2

    unavailable = _list(
        client, local_available="false", matches="false"
    ).json()
    assert [item["id"] for item in unavailable["items"]] == _ids(
        mixed.offline
    )
    assert unavailable["count"] == 2

    # matches=true implies local_available=true: the combination with
    # local_available=false can never match.
    impossible = _list(
        client, local_available="false", matches="true"
    ).json()
    assert impossible == {"items": [], "count": 0, "next_cursor": None}


def test_filtered_items_keep_stable_creation_order(client):
    mixed = _mixed(client)
    unfiltered = _list(client).json()["items"]
    by_id = {item["id"]: item for item in unfiltered}

    for params, expected in (
        ({"local_available": "true"}, [mixed.fresh, *mixed.stale]),
        ({"local_available": "false"}, mixed.offline),
        ({"matches": "true"}, [mixed.fresh]),
        (
            {"matches": "false"},
            [
                mixed.offline[0],
                mixed.stale[0],
                mixed.offline[1],
                mixed.stale[1],
            ],
        ),
    ):
        ids = [item["id"] for item in _list(client, **params).json()["items"]]
        assert ids == _ids(expected)
        # The filtered order is exactly the unfiltered stable order with
        # non-matching receipts removed.
        assert ids == [
            item["id"]
            for item in unfiltered
            if item["id"] in set(ids)
        ]
        for receipt in expected:
            assert by_id[receipt["id"]] is not None


def test_filtered_items_agree_with_unfiltered_and_single_route(client):
    mixed = _mixed(client)
    unfiltered = {
        item["id"]: item for item in _list(client).json()["items"]
    }
    for params in (
        {"local_available": "true"},
        {"local_available": "false"},
        {"matches": "true"},
        {"matches": "false"},
        {"local_available": "true", "matches": "false"},
    ):
        for item in _list(client, **params).json()["items"]:
            # Exactly the same item payload the unfiltered collection serves.
            assert item == unfiltered[item["id"]]
            single = _single_reconciliation(client, item["id"])
            assert (
                item["local_available"],
                item["local_manifest_digest_hex"],
                item["matches"],
            ) == (
                single["local_available"],
                single["local_manifest_digest_hex"],
                single["matches"],
            )


def test_filter_targets_the_read_time_reconciliation(client):
    # A receipt that matches when imported flips to non-matching after the
    # local snapshot changes; the filters follow the read-time verdict.
    _, _, bundle = _setup_bundle(client)
    receipt = _register(client, _served_import_request(client, bundle))

    assert [
        item["id"]
        for item in _list(client, matches="true").json()["items"]
    ] == [receipt["id"]]

    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    assert _list(client, matches="true").json()["items"] == []
    diverged = _list(
        client, local_available="true", matches="false"
    ).json()
    assert [item["id"] for item in diverged["items"]] == [receipt["id"]]
    assert diverged["count"] == 1
    assert _list(client, local_available="false").json()["items"] == []


def test_filter_never_resolves_an_unrelated_local_bundle(client):
    # The offline receipt references an unknown bundle id; the existence of
    # an unrelated local bundle must not surface it under local_available=true.
    offline = _offline_receipt(client, "offline")
    _, _, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    served = _register(client, _served_import_request(client, bundle))

    available = {
        item["id"]
        for item in _list(client, local_available="true").json()["items"]
    }
    assert available == {served["id"]}
    unavailable = {
        item["id"]
        for item in _list(client, local_available="false").json()["items"]
    }
    assert unavailable == {offline["id"]}


# --- Pagination over the filtered set -------------------------------------------


def test_filtered_pages_concatenate_without_gaps_or_duplicates(client):
    receipts = _offline_many(client, 7)
    all_items, pages, count = _walk_pages(client, matches="false", limit=3)
    assert count == 7
    assert [len(page) for page in pages] == [3, 3, 1]
    ids = [item["id"] for item in all_items]
    assert len(ids) == len(set(ids)) == 7
    assert ids == _ids(receipts)
    # Unavailable receipts satisfy both sides of this combined filter.
    all_items, pages, count = _walk_pages(
        client, local_available="false", matches="false", limit=3
    )
    assert count == 7
    assert [len(page) for page in pages] == [3, 3, 1]
    assert [item["id"] for item in all_items] == _ids(receipts)


def test_filtered_walk_excludes_every_non_matching_receipt(client):
    receipts = _offline_many(client, 6)
    _, _, bundle = _setup_bundle(client)
    _create_attestation(
        client, "evidence_bundle", bundle["id"], seed=SEED_A, signer="org-1"
    )
    matching = _register(client, _served_import_request(client, bundle))

    all_items, _, count = _walk_pages(client, matches="false", limit=2)
    assert count == 6
    walked = {item["id"] for item in all_items}
    assert walked == set(_ids(receipts))
    assert matching["id"] not in walked

    matched = _list(client, matches="true").json()
    assert [item["id"] for item in matched["items"]] == [matching["id"]]


def test_filtered_count_is_the_filtered_total_on_every_page(client):
    _offline_many(client, 7)
    pages = []
    cursor = None
    for _ in range(10):
        params = {"local_available": "false", "limit": 3}
        if cursor is not None:
            params["cursor"] = cursor
        body = _list(client, **params).json()
        pages.append(body["items"])
        assert body["count"] == 7
        cursor = body["next_cursor"]
        if cursor is None:
            break
    assert [len(page) for page in pages] == [3, 3, 1]


def test_filtered_last_page_cursor_null_on_exact_division(client):
    _offline_many(client, 4)
    first = _list(client, matches="false", limit=2).json()
    assert first["count"] == 4
    assert first["next_cursor"] is not None
    second = _list(
        client, matches="false", limit=2, cursor=first["next_cursor"]
    ).json()
    assert len(second["items"]) == 2
    assert second["count"] == 4
    assert second["next_cursor"] is None


def test_filtered_default_limit_is_fifty(client):
    receipts = _offline_many(client, 55)
    first = _list(client, matches="false").json()
    assert len(first["items"]) == 50
    assert first["count"] == 55
    second = _list(client, matches="false", cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 5
    assert second["count"] == 55
    assert second["next_cursor"] is None
    assert [item["id"] for item in first["items"] + second["items"]] == _ids(
        receipts
    )


def test_filtered_cursor_past_end_is_empty_with_filtered_count(client, app):
    _offline_many(client, 5)
    _, _, bundle = _setup_bundle(client)
    _register(client, _served_import_request(client, bundle))
    token = pagination.encode_typed_cursor(
        app.state.exchange_import_reconciliations_cursor_secret,
        pagination.EXCHANGE_IMPORT_RECONCILIATIONS_CURSOR,
        {
            "local_available": False,
            "matches": False,
            "limit": 50,
            "offset": 99,
        },
    )
    body = _list(
        client,
        local_available="false",
        matches="false",
        cursor=token,
    ).json()
    # count is the filtered total (5), never the unfiltered total (6).
    assert body == {"items": [], "count": 5, "next_cursor": None}


# --- Cursor binding --------------------------------------------------------------


def test_filter_cursor_is_bound_to_the_filter_and_limit(client):
    _offline_many(client, 3)
    cursor = _list(client, local_available="false", limit=2).json()[
        "next_cursor"
    ]
    assert cursor is not None

    # Dropping the filter, changing its value, adding another filter,
    # changing the limit, or dropping an explicit limit onto the default all
    # mismatch the bound claims.
    assert _list(client, cursor=cursor).status_code == 422
    assert _list(
        client, local_available="true", limit=2, cursor=cursor
    ).status_code == 422
    assert _list(
        client,
        local_available="false",
        matches="false",
        limit=2,
        cursor=cursor,
    ).status_code == 422
    assert _list(
        client, local_available="false", limit=3, cursor=cursor
    ).status_code == 422
    assert _list(client, local_available="false", cursor=cursor).status_code == 422

    # Replaying with exactly the bound filter and limit resumes the page.
    replay = _list(
        client, local_available="false", limit=2, cursor=cursor
    ).json()
    assert len(replay["items"]) == 1
    assert replay["count"] == 3
    assert replay["next_cursor"] is None


def test_unfiltered_cursor_is_rejected_when_a_filter_is_added(client):
    _offline_many(client, 3)
    cursor = _list(client, limit=2).json()["next_cursor"]
    assert _list(
        client, local_available="false", limit=2, cursor=cursor
    ).status_code == 422
    assert _list(client, matches="false", limit=2, cursor=cursor).status_code == 422
    # The same cursor still resumes the unfiltered collection unchanged.
    assert _list(client, limit=2, cursor=cursor).status_code == 200


def test_cursor_is_bound_to_both_filters(client):
    _mixed(client)
    cursor = _list(
        client, local_available="true", matches="false", limit=1
    ).json()["next_cursor"]
    assert cursor is not None
    assert (
        _list(
            client,
            local_available="true",
            matches="false",
            limit=1,
            cursor=cursor,
        ).status_code
        == 200
    )
    # Changing either filter, or dropping one, is a mismatch.
    assert (
        _list(
            client, local_available="true", matches="true", limit=1,
            cursor=cursor,
        ).status_code
        == 422
    )
    assert (
        _list(
            client, local_available="false", matches="false", limit=1,
            cursor=cursor,
        ).status_code
        == 422
    )
    assert (
        _list(client, local_available="true", limit=1, cursor=cursor).status_code
        == 422
    )
    assert (
        _list(client, matches="false", limit=1, cursor=cursor).status_code
        == 422
    )


def test_cursor_with_different_bound_filter_claims_is_422(client, app):
    _mixed(client)
    # A structurally valid, correctly signed ir1 cursor whose claims do not
    # describe this request must be rejected rather than re-filtered.
    token = pagination.encode_typed_cursor(
        app.state.exchange_import_reconciliations_cursor_secret,
        pagination.EXCHANGE_IMPORT_RECONCILIATIONS_CURSOR,
        {
            "local_available": True,
            "matches": None,
            "limit": 2,
            "offset": 1,
        },
    )
    resp = _list(
        client, local_available="false", limit=2, cursor=token
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_tampered_and_foreign_cursors_with_filters_are_422(client):
    _mixed(client)
    good = _list(client, matches="false", limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")

    imports_cursor = client.get(
        IMPORTS_URL, params={"limit": 1}
    ).json()["next_cursor"]

    for token in (
        "",
        "   ",
        "not-a-cursor",
        "ir0.abc.def",
        "ei1.x.y",
        "cr1.x.y",
        tampered,
        imports_cursor,
    ):
        resp = _list(client, matches="false", limit=1, cursor=token)
        assert resp.status_code == 422, repr(token)
        assert resp.json()["error"]["code"] == "validation_error"
        assert "items" not in resp.json()


# --- Parameter validation --------------------------------------------------------


def test_illegal_bool_literals_are_validation_errors(client):
    _mixed(client)
    illegal = (
        "",
        " ",
        "  ",
        "true ",
        " true",
        "True",
        "TRUE",
        "trUe",
        "false ",
        "False",
        "1",
        "0",
        "yes",
        "t",
        "null",
    )
    for field in ("local_available", "matches"):
        for value in illegal:
            resp = _list(client, **{field: value})
            assert resp.status_code == 422, (field, repr(value))
            assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_bool_parameters_are_validation_errors(client):
    _mixed(client)
    for suffix in (
        "local_available=true&local_available=false",
        "matches=true&matches=true",
        "local_available=true&matches=false&matches=true",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameter_alongside_a_filter_is_422(client):
    _mixed(client)
    for suffix in (
        "local_available=true&unknown=1",
        "matches=false&limit=2&offset=0",
        "local_available=true&evidence_bundle_id=evb_x",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_filter_parameters_validated_before_any_receipt_read(client):
    # An illegal filter is a 422 on an empty database: no receipt (and no
    # local bundle) is read before validation completes.
    for suffix in (
        "local_available=yes",
        "matches=1",
        "local_available=",
        "matches=%20%20",
        "local_available=true&local_available=false",
        "local_available=true&unknown=1",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_bad_filter_with_cursor_is_still_422(client):
    _mixed(client)
    good_cursor = _list(client, matches="false", limit=2).json()["next_cursor"]
    resp = _list(
        client, matches="not-a-bool", limit=2, cursor=good_cursor
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- Read-only, no-echo, determinism ----------------------------------------------


def test_filtered_reads_and_failures_write_no_rows_or_audit_events(
    client, db_session
):
    mixed = _mixed(client)

    def counts():
        return (
            db_session.scalar(
                select(func.count()).select_from(ExchangeImportRecord)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    receipts_before, events_before = counts()
    assert receipts_before == 5

    # Successful filtered reads, including full page walks under each filter.
    _walk_pages(client, local_available="true", limit=2)
    _walk_pages(client, local_available="false", limit=2)
    _walk_pages(client, matches="true", limit=2)
    _walk_pages(client, matches="false", limit=2)
    _walk_pages(
        client, local_available="true", matches="false", limit=2
    )
    _list(client, local_available="false", matches="true")
    _list(client, local_available="true", limit=100)

    # Failed filtered reads write nothing either.
    _list(client, local_available="yes")
    _list(client, matches="")
    client.get(f"{URL}?matches=true&matches=false")
    _list(client, local_available="true", cursor="tampered")
    _list(client, matches="false", cursor="ei1.x.y")
    _list(client, local_available="true", limit=0)
    cursor = _list(client, local_available="false", limit=2).json()[
        "next_cursor"
    ]
    _list(client, local_available="true", limit=2, cursor=cursor)

    db_session.expire_all()
    assert counts() == (receipts_before, events_before)
    # The five fixtures themselves are untouched and still list unfiltered.
    assert len(_list(client).json()["items"]) == 5
    assert {item["id"] for item in _list(client).json()["items"]} == set(
        _ids([*mixed.offline, mixed.fresh, *mixed.stale])
    )


def test_filtered_response_carries_no_snapshot_or_raw_material(client):
    _mixed(client)
    for params in (
        {"local_available": "true"},
        {"matches": "false"},
        {"local_available": "true", "matches": "false"},
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


def test_filtered_listing_is_deterministic_across_restart(
    file_client, tmp_db_url
):
    _mixed(file_client)
    expected = {
        key: _list(file_client, **params).json()
        for key, params in (
            ("available", {"local_available": "true"}),
            ("unavailable", {"local_available": "false"}),
            ("matching", {"matches": "true"}),
            ("diverged", {"local_available": "true", "matches": "false"}),
        )
    }

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as client:
        for key, params in (
            ("available", {"local_available": "true"}),
            ("unavailable", {"local_available": "false"}),
            ("matching", {"matches": "true"}),
            ("diverged", {"local_available": "true", "matches": "false"}),
        ):
            resp = _list(client, **params)
            assert resp.status_code == 200, resp.text
            assert resp.json() == expected[key]
