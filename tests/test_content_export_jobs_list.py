"""Tests for the read-only content export job search.

Covers ``GET /v1/content-export-jobs``:

* the success body is exactly ``{"items", "count", "next_cursor"}`` with each
  item the existing single-job public view, in stable creation order;
* ``content_id``/``request_id``/``status`` exact, case/whitespace-sensitive
  filters (absent means unfiltered; unknown values are an empty set, not an
  error), plus inclusive strict RFC 3339 UTC ``from``/``to`` bounds;
* ``limit`` (1..100, default 50, strict decimal) and an opaque HMAC cursor
  bound to every effective filter and the limit, with no duplication/omission
  across pages and ``null`` on the final page;
* blank/illegal/repeated/undeclared parameters, an inverted time range, an
  illegal status, and a blank/malformed/tampered/foreign-family/mismatching
  cursor are all ``422 validation_error`` rejected before any job is read;
* a non-empty GET body is a 422;
* the body is compact UTF-8 JSON ending in exactly one newline;
* the query is strictly read-only (no job, state change, or audit write).
"""

from __future__ import annotations

import hashlib

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from provenance import pagination
from provenance.models import AuditEvent, ContentExportJob
from provenance.time_utils import parse_rfc3339_utc
from tests.helpers import create_actor


def _digest(name: str) -> str:
    return hashlib.sha256(f"joblist-{name}".encode()).hexdigest()


def _create_content(client, name, actor_id="org-1"):
    resp = client.post(
        "/v1/contents",
        json={
            "digest_algorithm": "sha256",
            "digest_hex": _digest(f"content-{name}"),
            "media_type": "image/png",
            "title": name,
            "actor_id": actor_id,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_job(client, content_id, request_id):
    resp = client.post(
        "/v1/content-export-jobs",
        json={"content_id": content_id, "request_id": request_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


_PATH = "/v1/content-export-jobs"


# --- wire format ------------------------------------------------------------


def test_empty_collection_body_is_compact_json_with_single_newline(client):
    resp = client.get(_PATH)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].split(";")[0] == "application/json"
    # The body is UTF-8 JSON.
    assert resp.content.decode("utf-8") == (
        '{"items":[],"count":0,"next_cursor":null}\n'
    )
    # Exactly the compact body plus one trailing newline.
    assert resp.content == (
        b'{"items":[],"count":0,"next_cursor":null}\n'
    )
    body = resp.json()
    assert set(body) == {"items", "count", "next_cursor"}
    assert body == {"items": [], "count": 0, "next_cursor": None}


# --- ordering and the public view -------------------------------------------


def test_items_follow_stable_creation_order_and_reuse_detail_view(client):
    create_actor(client)
    content = _create_content(client, "main")
    created = [
        _create_job(client, content["id"], f"req-{i}") for i in range(5)
    ]

    resp = client.get(_PATH)
    body = resp.json()
    assert [item["id"] for item in body["items"]] == [j["id"] for j in created]
    assert body["count"] == 5
    assert body["next_cursor"] is None

    # Each list item is exactly the single-detail public view.
    for item in body["items"]:
        detail = client.get(f"{_PATH}/{item['id']}").json()
        assert item == detail
        assert set(item) == {
            "id",
            "content_id",
            "request_id",
            "status",
            "created_at",
            "started_at",
            "finished_at",
            "result",
            "error",
        }


def test_default_limit_is_fifty(client):
    create_actor(client)
    content = _create_content(client, "many")
    for i in range(55):
        _create_job(client, content["id"], f"req-{i:03d}")

    first = client.get(_PATH).json()
    assert len(first["items"]) == 50
    assert first["count"] == 55
    assert first["next_cursor"] is not None


# --- pagination -------------------------------------------------------------


def _fetch_all(client, page_limit):
    seen = []
    url = f"{_PATH}?limit={page_limit}"
    pages = 0
    while True:
        body = client.get(url).json()
        pages += 1
        seen.extend(item["id"] for item in body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            return seen, body["count"], pages
        url = f"{_PATH}?limit={page_limit}&cursor={cursor}"


def test_pagination_resumes_without_duplication_or_omission(client):
    create_actor(client)
    content = _create_content(client, "paging")
    created = [
        _create_job(client, content["id"], f"req-{i:02d}") for i in range(7)
    ]
    expected = [j["id"] for j in created]

    seen, total, pages = _fetch_all(client, 3)
    assert pages == 3
    assert total == 7
    assert seen == expected
    assert len(seen) == len(set(seen)) == 7


def test_resending_last_cursor_and_a_past_tail_cursor(client, app):
    create_actor(client)
    content = _create_content(client, "tail")
    for i in range(2):
        _create_job(client, content["id"], f"req-{i}")

    # Walk to the final page; it carries the last item and a null cursor.
    first = client.get(f"{_PATH}?limit=1").json()
    last_cursor = first["next_cursor"]
    page_two = client.get(f"{_PATH}?limit=1&cursor={last_cursor}").json()
    assert len(page_two["items"]) == 1
    assert page_two["count"] == 2
    assert page_two["next_cursor"] is None
    # Re-sending the final in-range cursor returns the same final page
    # deterministically (no duplication).
    again = client.get(f"{_PATH}?limit=1&cursor={last_cursor}").json()
    assert [i["id"] for i in again["items"]] == [
        i["id"] for i in page_two["items"]
    ]
    assert again["count"] == 2

    # A validly-signed cursor positioned past the tail returns an empty page,
    # the total count, and a null cursor.
    past = pagination.encode_typed_cursor(
        app.state.content_export_jobs_cursor_secret,
        pagination.CONTENT_EXPORT_JOBS_CURSOR,
        {
            "content_id": None,
            "request_id": None,
            "status": None,
            "from": None,
            "to": None,
            "limit": 1,
            "offset": 5,
        },
    )
    empty = client.get(f"{_PATH}?limit=1&cursor={past}").json()
    assert empty["items"] == []
    assert empty["count"] == 2
    assert empty["next_cursor"] is None


# --- filters ----------------------------------------------------------------


def test_filter_by_content_id_unknown_id_is_empty_not_error(client):
    create_actor(client)
    content = _create_content(client, "known")
    _create_job(client, content["id"], "req-1")

    body = client.get(f"{_PATH}?content_id={content['id']}").json()
    assert body["count"] == 1

    missing = client.get(f"{_PATH}?content_id=cnt_unknown").json()
    assert missing == {"items": [], "count": 0, "next_cursor": None}


def test_filter_by_request_id_is_exact_and_unknown_is_empty(client):
    create_actor(client)
    content = _create_content(client, "byreq")
    _create_job(client, content["id"], "alpha")
    _create_job(client, content["id"], "beta")

    assert client.get(f"{_PATH}?request_id=alpha").json()["count"] == 1
    assert client.get(f"{_PATH}?request_id=gamma").json()["count"] == 0


def test_string_filters_are_case_and_whitespace_sensitive(client):
    create_actor(client)
    content = _create_content(client, "case")
    _create_job(client, content["id"], "Req-X")

    # The id/request filters match exactly; padding or casing never matches.
    assert client.get(f"{_PATH}?request_id=Req-X").json()["count"] == 1
    assert client.get(f"{_PATH}?request_id=req-x").json()["count"] == 0
    assert client.get(f"{_PATH}?request_id=Req-X%20").json()["count"] == 0
    assert client.get(f"{_PATH}?content_id={content['id']}%20").json()["count"] == 0


def test_status_filter_accepts_only_the_four_literals(client):
    create_actor(client)
    content = _create_content(client, "states")
    job = _create_job(client, content["id"], "req-run")
    client.post(f"{_PATH}/{job['id']}/run")

    assert client.get(f"{_PATH}?status=succeeded").json()["count"] == 1
    assert client.get(f"{_PATH}?status=pending").json()["count"] == 0
    # Absent status imposes no filter.
    assert client.get(_PATH).json()["count"] == 1


def test_status_filter_reflects_state_changes_but_keeps_order(client):
    create_actor(client)
    content = _create_content(client, "changing")
    first = _create_job(client, content["id"], "req-a")
    second = _create_job(client, content["id"], "req-b")

    pending = client.get(f"{_PATH}?status=pending").json()
    assert [i["id"] for i in pending["items"]] == [first["id"], second["id"]]

    client.post(f"{_PATH}/{second['id']}/run")
    still_pending = client.get(f"{_PATH}?status=pending").json()
    assert [i["id"] for i in still_pending["items"]] == [first["id"]]
    succeeded = client.get(f"{_PATH}?status=succeeded").json()
    assert [i["id"] for i in succeeded["items"]] == [second["id"]]
    # Creation order across the unfiltered collection is unchanged.
    all_jobs = client.get(_PATH).json()
    assert [i["id"] for i in all_jobs["items"]] == [first["id"], second["id"]]


def test_filters_combine_as_logical_and(client):
    create_actor(client)
    content_a = _create_content(client, "a")
    content_b = _create_content(client, "b")
    job = _create_job(client, content_a["id"], "shared-a")
    _create_job(client, content_b["id"], "shared-b")
    client.post(f"{_PATH}/{job['id']}/run")

    body = client.get(
        f"{_PATH}?content_id={content_a['id']}"
        "&request_id=shared-a&status=succeeded"
    ).json()
    assert body["count"] == 1
    assert body["items"][0]["id"] == job["id"]

    # A succeeded job exists only on content_a; combining content_b with
    # status=succeeded excludes it.
    assert (
        client.get(
            f"{_PATH}?content_id={content_b['id']}&status=succeeded"
        ).json()["count"]
        == 0
    )


def test_from_to_are_inclusive_utc_bounds(client):
    create_actor(client)
    content = _create_content(client, "bounds")
    job = _create_job(client, content["id"], "req-only")
    created_at = job["created_at"]

    # A bound exactly at created_at includes the job (inclusive).
    assert client.get(f"{_PATH}?from={created_at}").json()["count"] == 1
    assert client.get(f"{_PATH}?to={created_at}").json()["count"] == 1
    assert (
        client.get(f"{_PATH}?from={created_at}&to={created_at}").json()["count"]
        == 1
    )
    # A lower bound strictly after it excludes the job.
    assert client.get(f"{_PATH}?from=9999-01-01T00:00:00Z").json()["count"] == 0
    assert client.get(f"{_PATH}?to=2000-01-01T00:00:00Z").json()["count"] == 0
    # Both notations parse; the timestamp itself is a valid UTC instant.
    assert parse_rfc3339_utc(created_at) is not None


def test_from_and_to_accept_plus_utc_notation(client):
    create_actor(client)
    content = _create_content(client, "plus")
    job = _create_job(client, content["id"], "req-plus")
    resp = client.get(
        _PATH, params={"from": job["created_at"].replace("Z", "+00:00")}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["count"] == 1


# --- parameter validation ---------------------------------------------------


def test_illegal_status_values_are_422(client):
    for value in ["Pending", "PENDING", "queued", "done", "succeded"]:
        resp = client.get(_PATH, params={"status": value})
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_blank_status_is_422_not_unfiltered(client):
    resp = client.get(_PATH, params={"status": ""})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_blank_string_filters_are_422(client):
    for field in ("content_id", "request_id"):
        resp = client.get(_PATH, params={field: "   "})
        assert resp.status_code == 422, field
        assert resp.json()["error"]["code"] == "validation_error"


def test_limit_validation(client):
    for ok_value in ("1", "100"):
        assert client.get(f"{_PATH}?limit={ok_value}").status_code == 200
    for bad in ("0", "101", "-1", "2.0", "1e1", "abc", "", " 5", "5 "):
        resp = client.get(_PATH, params={"limit": bad})
        assert resp.status_code == 422, bad
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_422(client):
    resp = client.get(f"{_PATH}?status=pending&status=failed")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert client.get(f"{_PATH}?limit=1&limit=2").status_code == 422


def test_undeclared_parameter_is_422(client):
    resp = client.get(f"{_PATH}?unknown=1")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_malformed_and_inverted_time_ranges_are_422(client):
    for bad in (
        "2026-01-02",
        "2026-01-02T00:00Z",  # missing seconds
        "2026-13-02T00:00:00Z",  # impossible month
        "2026-01-02T00:00:00",  # no UTC designator
        "2026-01-02T00:00:00+01:00",  # non-UTC offset
        "2026-01-02T00:00:00z",  # lowercase designator
    ):
        resp = client.get(_PATH, params={"from": bad})
        assert resp.status_code == 422, bad
    inverted = client.get(
        _PATH,
        params={"from": "2026-01-03T00:00:00Z", "to": "2026-01-02T00:00:00Z"},
    )
    assert inverted.status_code == 422
    # Equal bounds are valid (an inclusive zero-width window).
    equal = client.get(
        _PATH,
        params={"from": "2026-01-02T00:00:00Z", "to": "2026-01-02T00:00:00Z"},
    )
    assert equal.status_code == 200


# --- cursor validation ------------------------------------------------------


def test_blank_cursor_is_422(client):
    resp = client.get(_PATH, params={"cursor": ""})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_malformed_and_tampered_cursor_are_422(client):
    create_actor(client)
    content = _create_content(client, "cur")
    _create_job(client, content["id"], "req-1")
    _create_job(client, content["id"], "req-2")
    valid = client.get(f"{_PATH}?limit=1").json()["next_cursor"]
    assert valid is not None

    tampered = valid[:-2] + ("AA" if valid[-2:] != "AA" else "BB")
    for bad in ("not-a-cursor", "cx1.abc.sig", tampered):
        resp = client.get(_PATH, params={"limit": "1", "cursor": bad})
        assert resp.status_code == 422, bad


def test_foreign_family_cursor_is_422(client, app):
    create_actor(client)
    content = _create_content(client, "fam")
    _create_job(client, content["id"], "req-1")
    # A cursor minted by another endpoint's family (the claims search), even
    # though it shares the server secret, must not resume this query.
    foreign = pagination.encode_typed_cursor(
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
    resp = client.get(_PATH, params={"cursor": foreign})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_bound_to_other_filters_or_limit_is_422(client):
    create_actor(client)
    content = _create_content(client, "bind")
    _create_job(client, content["id"], "req-1")
    _create_job(client, content["id"], "req-2")
    cursor = client.get(f"{_PATH}?limit=1").json()["next_cursor"]

    # Same cursor, different effective limit -> 422.
    assert client.get(f"{_PATH}?limit=2&cursor={cursor}").status_code == 422
    # Different effective filter (a status now present) -> 422.
    assert (
        client.get(f"{_PATH}?limit=1&status=pending&cursor={cursor}").status_code
        == 422
    )
    # A different content filter changes the bound claim set -> 422.
    assert (
        client.get(
            f"{_PATH}?limit=1&content_id=cnt_other&cursor={cursor}"
        ).status_code
        == 422
    )


# --- request body -----------------------------------------------------------


def test_get_with_a_body_is_422(client):
    resp = client.request("GET", _PATH, content=b'{"x":1}')
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_get_with_empty_body_is_accepted(client):
    resp = client.request("GET", _PATH, content=b"")
    assert resp.status_code == 200


# --- read-only semantics ----------------------------------------------------


def test_search_is_read_only(client, db_session):
    create_actor(client)
    content = _create_content(client, "ro")
    job = _create_job(client, content["id"], "req-ro")
    client.post(f"{_PATH}/{job['id']}/run")

    def audit_count():
        return db_session.execute(
            select(func.count()).select_from(AuditEvent)
        ).scalar_one()

    def job_count():
        return db_session.execute(
            select(func.count()).select_from(ContentExportJob)
        ).scalar_one()

    before_jobs = job_count()
    before_events = audit_count()

    for query in (
        _PATH,
        f"{_PATH}?status=succeeded",
        f"{_PATH}?content_id={content['id']}",
        f"{_PATH}?from=2000-01-01T00:00:00Z&to=2100-01-01T00:00:00Z",
        f"{_PATH}?content_id=cnt_unknown",
        f"{_PATH}?status=bogus",
    ):
        client.get(query)

    # Neither a successful nor a rejected read creates a job or audit row.
    assert job_count() == before_jobs
    assert audit_count() == before_events
    # The read leaves the settled job's lifecycle state untouched.
    detail = client.get(f"{_PATH}/{job['id']}").json()
    assert detail["status"] == "succeeded"
