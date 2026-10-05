"""Tests for the read-only claim payload verification endpoint.

Covers POST /v1/claim-verifications: a third party submits exactly
``claim_id`` and a candidate ``payload``; the service canonicalizes the
payload under the same deterministic JSON rules as claim creation
(Unicode-code-point key order, minimal separators, UTF-8), hashes it with
SHA-256, and compares it against the claim's persisted
``payload_digest_algorithm``/``payload_digest_hex``.

A match and an ordinary mismatch on an existing claim are both plain
``200 {"valid": ...}`` verdicts; an unknown claim is ``404 claim_not_found``.
Structural failures (non-object body, missing fields, non-string/blank
claim_id, non-object or non-canonicalizable payload, undeclared fields, any
query parameter) are ``422 validation_error`` and never read the target
claim. The endpoint is strictly read-only: it never creates, updates, or
deletes any claim, evidence, task, or audit event, and never echoes,
persists, or logs the submitted payload.
"""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import select

from provenance.models import AuditEvent, Claim
from tests.helpers import DIGEST_A, content_payload, create_actor

URL = "/v1/claim-verifications"

PAYLOAD_1 = {"statement": "created by org-1", "confidence": 0.9}
PAYLOAD_2 = {"statement": "reviewed", "approved": True}
# A non-ASCII key sorts among ASCII keys purely by Unicode code point.
UNICODE_PAYLOAD = {"名称": "证据", "z-key": 1, "a-key": [1, 2, {"nested": True}]}

SECRET_MARKER = "unique-verification-payload-marker-9b2c"


def _assert_validation_error(resp) -> dict:
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["details"]["issues"]
    return body


def _create_content(client, actor_id="org-1", digest=DIGEST_A):
    resp = client.post(
        "/v1/contents", json=content_payload(actor_id=actor_id, digest=digest)
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, actor_id="org-1", payload=PAYLOAD_1):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": "authorship",
            "payload": payload,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _setup_claim(client, payload=PAYLOAD_1):
    create_actor(client)
    return _create_claim(client, _create_content(client)["id"], payload=payload)


# --- Matching and mismatching verdicts --------------------------------------


def test_matching_payload_is_200_valid_true(client):
    claim = _setup_claim(client)
    resp = client.post(URL, json={"claim_id": claim["id"], "payload": PAYLOAD_1})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}
    assert set(resp.json()) == {"valid"}
    # Compact JSON body, exactly the one member.
    assert resp.text == '{"valid":true}'


def test_mismatching_payload_on_existing_claim_is_200_valid_false(client):
    claim = _setup_claim(client)
    resp = client.post(URL, json={"claim_id": claim["id"], "payload": PAYLOAD_2})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": False}
    assert resp.text == '{"valid":false}'


def test_reordered_keys_and_insignificant_serialization_verify_true(client):
    claim = _setup_claim(client)
    # Same semantics, different root and nested key order.
    reordered = {"confidence": 0.9, "statement": "created by org-1"}
    assert list(reordered) != list(PAYLOAD_1)
    resp = client.post(
        URL, json={"claim_id": claim["id"], "payload": reordered}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}

    # Pretty/escaped raw JSON with the same semantics also verifies.
    raw = json.dumps(PAYLOAD_1, indent=2, ensure_ascii=True)
    resp = client.post(
        URL,
        content=json.dumps({"claim_id": claim["id"]})[:-1]
        + ',"payload":'
        + raw
        + "}",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_unicode_keys_sort_by_code_point_and_use_unescaped_utf8(client):
    claim = _setup_claim(client, payload=UNICODE_PAYLOAD)
    resp = client.post(
        URL, json={"claim_id": claim["id"], "payload": dict(reversed(list(UNICODE_PAYLOAD.items())))}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}

    # The same object over ASCII-escaped UTF-8 as an actual semantic change
    # cannot be submitted through JSON parsing (escapes decode away); but a
    # genuinely different value must not verify.
    changed = {**UNICODE_PAYLOAD, "z-key": 2}
    resp = client.post(URL, json={"claim_id": claim["id"], "payload": changed})
    assert resp.json() == {"valid": False}


def test_semantically_different_payloads_are_not_merged(client):
    claim = _setup_claim(client, payload={"a": 1, "b": 2})
    # Key order alone never matters; value/type differences always do.
    for different in (
        {"a": 2, "b": 1},
        {"a": 1, "b": "2"},
        {"a": 1, "b": 2, "c": None},
        {"a": 1},
        {"a": 1, "b": 2.0},
        {"a": [1, 2], "b": 2},
        {"a": [2, 1], "b": 2},  # array order is significant
    ):
        resp = client.post(
            URL, json={"claim_id": claim["id"], "payload": different}
        )
        assert resp.status_code == 200, different
        assert resp.json() == {"valid": False}, different


def test_empty_object_payload_round_trips_through_verification(client):
    claim = _setup_claim(client, payload={})
    resp = client.post(URL, json={"claim_id": claim["id"], "payload": {}})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}
    resp = client.post(
        URL, json={"claim_id": claim["id"], "payload": {"x": 0}}
    )
    assert resp.json() == {"valid": False}


def test_surrounding_whitespace_in_claim_id_is_trimmed_like_other_ids(client):
    claim = _setup_claim(client)
    resp = client.post(
        URL,
        json={"claim_id": f"  {claim['id']}\t", "payload": PAYLOAD_1},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verdict_agrees_with_claim_creation_digest(client):
    # The digest served at creation is exactly what verification recomputes.
    claim = _setup_claim(client, payload=PAYLOAD_2)
    assert claim["payload_digest_algorithm"] == "sha256"
    resp = client.post(URL, json={"claim_id": claim["id"], "payload": PAYLOAD_2})
    assert resp.json() == {"valid": True}


def test_repeated_verifications_are_stable(client):
    claim = _setup_claim(client)
    for _ in range(3):
        assert client.post(
            URL, json={"claim_id": claim["id"], "payload": PAYLOAD_1}
        ).json() == {"valid": True}
        assert client.post(
            URL, json={"claim_id": claim["id"], "payload": PAYLOAD_2}
        ).json() == {"valid": False}


# --- Unknown claim ------------------------------------------------------------


def test_unknown_claim_id_is_404_claim_not_found(client):
    # No setup whatsoever: structurally valid request, missing resource.
    resp = client.post(
        URL, json={"claim_id": "clm_does_not_exist", "payload": {}}
    )
    assert resp.status_code == 404, resp.text
    error = resp.json()["error"]
    assert error["code"] == "claim_not_found"
    assert error["details"]["claim_id"] == "clm_does_not_exist"


def test_unknown_claim_id_never_echoes_payload(client):
    secret = {"secret": SECRET_MARKER}
    resp = client.post(
        URL, json={"claim_id": "clm_does_not_exist", "payload": secret}
    )
    assert resp.status_code == 404
    assert SECRET_MARKER not in resp.text


def test_concurrent_reads_agree_and_change_nothing(file_client, tmp_db_url):
    # A file-backed database has a normal connection pool, so concurrent
    # worker threads each get their own SQLite connection.
    create_actor(file_client)
    claim = _create_claim(file_client, _create_content(file_client)["id"])
    bodies = [
        {"claim_id": claim["id"], "payload": PAYLOAD_1},
        {"claim_id": claim["id"], "payload": PAYLOAD_2},
        {"claim_id": "clm_does_not_exist", "payload": {}},
    ] * 8

    def call(body):
        return file_client.post(URL, json=body)

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(call, bodies))

    assert {r.status_code for r in responses} == {200, 404}
    true_count = sum(
        r.json() == {"valid": True} for r in responses if r.status_code == 200
    )
    false_count = sum(
        r.json() == {"valid": False} for r in responses if r.status_code == 200
    )
    not_found_count = sum(r.status_code == 404 for r in responses)
    assert (true_count, false_count, not_found_count) == (8, 8, 8)

    # Repeated and concurrent reads left exactly the one setup claim in place.
    con = sqlite3.connect(tmp_db_url.removeprefix("sqlite:///"))
    try:
        assert con.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 1
        assert con.execute(
            "SELECT payload_digest_hex FROM claims"
        ).fetchone()[0] == claim["payload_digest_hex"]
    finally:
        con.close()


# --- Read-only / no echo ------------------------------------------------------


def test_verifications_write_no_rows_or_audit_events(client, db_session):
    claim = _setup_claim(client)
    claims_before = db_session.execute(select(Claim)).scalars().all()
    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())

    bodies = [
        {"claim_id": claim["id"], "payload": PAYLOAD_1},  # valid
        {"claim_id": claim["id"], "payload": PAYLOAD_2},  # mismatch
        {"claim_id": "clm_does_not_exist", "payload": {}},  # not found
        {"claim_id": claim["id"], "payload": {"marker": SECRET_MARKER}},
    ]
    statuses = [client.post(URL, json=b).status_code for b in bodies]
    assert statuses == [200, 200, 404, 200]

    assert [c.id for c in db_session.execute(select(Claim)).scalars()] == [
        c.id for c in claims_before
    ]
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )
    # The stored claim keeps its original digest: verification never mutates.
    stored = db_session.execute(
        select(Claim).where(Claim.id == claim["id"])
    ).scalar_one()
    assert stored.payload_digest_hex == claim["payload_digest_hex"]


def test_entire_database_file_is_unchanged_by_verifications(
    tmp_db_url, file_client
):
    create_actor(file_client)
    claim = _create_claim(file_client, _create_content(file_client)["id"])

    path = tmp_db_url.removeprefix("sqlite:///")

    def table_counts():
        con = sqlite3.connect(path)
        try:
            tables = [
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            ]
            return {
                table: con.execute(
                    f"SELECT COUNT(*) FROM {table}"  # noqa: S608 - fixed names
                ).fetchone()[0]
                for table in sorted(tables)
            }
        finally:
            con.close()

    before = table_counts()

    bodies = [
        {"claim_id": claim["id"], "payload": PAYLOAD_1},
        {"claim_id": claim["id"], "payload": PAYLOAD_2},
        {"claim_id": "clm_does_not_exist", "payload": {}},
        {"payload": PAYLOAD_1},  # structurally invalid
    ]
    statuses = [file_client.post(URL, json=b).status_code for b in bodies]
    assert statuses == [200, 200, 404, 422]

    assert table_counts() == before


def test_verdict_never_echoes_payload_fields_or_raw_text(client):
    claim = _setup_claim(client)
    secret_payload = {"marker": SECRET_MARKER, "nested": {"k": ["v"]}}
    for verdict_payload in (PAYLOAD_1, secret_payload):
        resp = client.post(
            URL, json={"claim_id": claim["id"], "payload": verdict_payload}
        )
        assert resp.status_code == 200
        assert SECRET_MARKER not in resp.text
        assert '"marker"' not in resp.text
        assert resp.text in ('{"valid":true}', '{"valid":false}')


# --- 422 validation boundary ---------------------------------------------------


def test_missing_fields_are_422(client):
    body = _assert_validation_error(client.post(URL, json={}))
    issue_fields = {
        ".".join(part for part in issue["loc"] if part != "body")
        for issue in body["error"]["details"]["issues"]
    }
    assert {"claim_id", "payload"}.issubset(issue_fields)


def test_missing_each_field_is_422(client):
    _assert_validation_error(client.post(URL, json={"claim_id": "clm_x"}))
    _assert_validation_error(client.post(URL, json={"payload": {}}))


def test_extra_fields_are_422(client):
    body = {"claim_id": "clm_x", "payload": {}, "unexpected": "value"}
    _assert_validation_error(client.post(URL, json=body))


@pytest.mark.parametrize("raw", ["[]", "123", "null", '"a string"', "4.5", "true"])
def test_non_object_json_body_is_422(client, raw):
    resp = client.post(
        URL, content=raw, headers={"content-type": "application/json"}
    )
    _assert_validation_error(resp)


def test_malformed_json_body_is_422(client):
    resp = client.post(
        URL,
        content="{not valid json",
        headers={"content-type": "application/json"},
    )
    _assert_validation_error(resp)


@pytest.mark.parametrize("value", ["", "   ", "\t\n "])
def test_blank_claim_id_is_422(client, value):
    _assert_validation_error(
        client.post(URL, json={"claim_id": value, "payload": {}})
    )


@pytest.mark.parametrize("value", [123, None, True, ["x"], {"nested": "id"}])
def test_non_string_claim_id_is_422(client, value):
    _assert_validation_error(
        client.post(URL, json={"claim_id": value, "payload": {}})
    )


@pytest.mark.parametrize("value", [[], "text", 42, 3.14, True, None])
def test_non_object_payload_is_422(client, value):
    _assert_validation_error(
        client.post(URL, json={"claim_id": "clm_x", "payload": value})
    )


@pytest.mark.parametrize("non_finite", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_in_payload_are_422(client, non_finite):
    # Python's json emits NaN/Infinity literals; they have no canonical form.
    raw = f'{{"claim_id":"clm_x","payload":{{"v":{non_finite}}}}}'
    resp = client.post(
        URL, content=raw, headers={"content-type": "application/json"}
    )
    _assert_validation_error(resp)


def test_nested_non_finite_number_is_422(client):
    raw = '{"claim_id":"clm_x","payload":{"a":[{"b":NaN}]}}'
    resp = client.post(
        URL, content=raw, headers={"content-type": "application/json"}
    )
    _assert_validation_error(resp)


def test_any_query_parameter_is_422(client):
    resp = client.post(
        URL + "?x=1", json={"claim_id": "clm_x", "payload": {}}
    )
    _assert_validation_error(resp)


def test_validation_error_body_never_echoes_payload(client):
    raw = (
        '{"claim_id":"clm_x","payload":{"marker":"' + SECRET_MARKER + '"},"extra":1}'
    )
    resp = client.post(
        URL, content=raw, headers={"content-type": "application/json"}
    )
    assert resp.status_code == 422
    assert SECRET_MARKER not in resp.text


def test_structural_validation_precedes_claim_lookup(client, db_session):
    # Every structurally invalid request is a 422 even though no claim with
    # the referenced id exists -- and it must not collapse into the 404 path.
    missing_claim = "clm_does_not_exist"
    attempts = [
        client.post(URL, json={"payload": {}}),
        client.post(URL, json={"claim_id": missing_claim}),
        client.post(URL, json={"claim_id": "", "payload": {}}),
        client.post(URL, json={"claim_id": 123, "payload": {}}),
        client.post(URL, json={"claim_id": missing_claim, "payload": []}),
        client.post(
            URL + "?x=1",
            json={"claim_id": missing_claim, "payload": {}},
        ),
    ]
    assert all(r.status_code == 422 for r in attempts), [
        r.status_code for r in attempts
    ]
    # Nothing was read into existence or written; no claim rows at all.
    assert db_session.execute(select(Claim)).scalars().all() == []


def test_non_finite_payload_against_existing_claim_is_422_not_false(client):
    claim = _setup_claim(client)
    raw = f'{{"claim_id":{json.dumps(claim["id"])},"payload":{{"v":NaN}}}}'
    resp = client.post(
        URL, content=raw, headers={"content-type": "application/json"}
    )
    _assert_validation_error(resp)
