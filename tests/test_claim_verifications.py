"""Tests for the read-only third-party claim payload verification endpoint.

Covers POST /v1/claim-verifications: the offered ``payload`` is
canonicalized under the exact same deterministic rules as claim creation
(object keys sorted by Unicode code point, minimal separators, unescaped
non-ASCII, UTF-8), hashed with SHA-256, and compared to the existing
claim's persisted digest algorithm and value.

A digest match returns ``200 {"valid": true}``; a claim that exists but
whose digest differs returns ``200 {"valid": false}`` (a mismatch is
never a service error); an unknown ``claim_id`` is ``404
claim_not_found``. Structural failures are ``422 validation_error``
under the existing error JSON and are rejected before the claim is read.
The endpoint is strictly read-only: no claim, evidence, task, or audit
row is created, updated, or deleted, and the offered payload is never
echoed, persisted, or logged.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading

from sqlalchemy import select

from provenance.models import AuditEvent, Claim
from tests.helpers import DIGEST_A, content_payload, create_actor

URL = "/v1/claim-verifications"


# --- Payload and setup helpers ----------------------------------------------


PAYLOAD = {
    "statement": "created by org-1",
    "confidence": 0.9,
    "nested": {"b": 1, "a": [True, None, "文本"], "c": {"z": 0, "中": 2}},
    "list": [{"x": 1}, {"y": 2}],
    "unicode": "证据",
}

DIFFERENT_PAYLOADS = [
    {"statement": "created by org-1", "confidence": 0.91},  # changed value
    {"statement": "created by org-1", "extra": True},  # added member
    {"statement": "created by org-1"},  # removed member
    {"statement": "created by org-1", "confidence": "0.9"},  # number -> string
    {"statement": "created by org-1", "confidence": [0.9]},  # scalar -> array
    {"statement": "created by org-1", "confidence": {"v": 0.9}},  # -> object
    {
        "statement": "created by org-1",
        "confidence": 0.9,
        "nested": {"b": 1, "a": [True, None, "文本"], "c": {"z": 1, "中": 2}},
    },  # deep value changed
    {
        "statement": "created by org-1",
        "confidence": 0.9,
        "list": [{"y": 2}, {"x": 1}],  # array element order is significant
    },
    {"statement": "证据", "confidence": 0.9},  # different unicode value
]


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


def _create_claim(client, content_id, actor_id="org-1", claim_type="authorship",
                  payload=PAYLOAD):
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


def _setup_claim(client, payload=PAYLOAD):
    create_actor(client)
    return _create_claim(client, _create_content(client)["id"], payload=payload)


def _verification_body(claim_id, payload, **dump_opts) -> bytes:
    return json.dumps(
        {"claim_id": claim_id, "payload": payload}, **dump_opts
    ).encode("utf-8")


def _verify_raw(client, claim_id, payload, **dump_opts):
    return client.post(
        URL,
        content=_verification_body(claim_id, payload, **dump_opts),
        headers={"content-type": "application/json"},
    )


# --- Matching verdicts -------------------------------------------------------


def test_exact_payload_verifies_valid_true(client):
    claim = _setup_claim(client)
    resp = client.post(URL, json={"claim_id": claim["id"], "payload": PAYLOAD})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}
    assert set(resp.json()) == {"valid"}
    # Compact JSON body, exactly the one member.
    assert resp.text == '{"valid":true}'


def test_reordered_object_keys_verify_identically(client):
    claim = _setup_claim(client)
    reordered = {
        "unicode": "证据",
        "list": [{"x": 1}, {"y": 2}],
        "nested": {"c": {"中": 2, "z": 0}, "a": [True, None, "文本"], "b": 1},
        "confidence": 0.9,
        "statement": "created by org-1",
    }
    resp = client.post(
        URL, json={"claim_id": claim["id"], "payload": reordered}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_whitespace_and_ascii_escaping_serializations_verify(client):
    claim = _setup_claim(client)
    # Pretty-printed, ASCII-escaped bytes parse to the same JSON object and
    # must canonicalize to the same digest as the compact UTF-8 submission.
    resp = _verify_raw(
        client, claim["id"], PAYLOAD, indent=2, ensure_ascii=True
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_empty_object_payload_verifies(client):
    claim = _setup_claim(client, payload={})
    resp = client.post(
        URL, json={"claim_id": claim["id"], "payload": {}}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_unicode_keys_sorted_by_code_point(client):
    payload = {"é": 1, "a": 2, "中": 3, "A": 4}
    claim = _setup_claim(client, payload=payload)
    resp = client.post(
        URL,
        json={"claim_id": claim["id"], "payload": {"中": 3, "a": 2, "A": 4, "é": 1}},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_semantically_identical_integer_float_spelling_is_not_merged(client):
    # Canonical JSON preserves the lexical number distinction: 1 and 1.0
    # are different documents, so their digests differ.
    claim = _setup_claim(client, payload={"n": 1})
    resp = client.post(
        URL, json={"claim_id": claim["id"], "payload": {"n": 1.0}}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": False}


def test_repeated_verifications_are_stable(client):
    claim = _setup_claim(client)
    body = {"claim_id": claim["id"], "payload": PAYLOAD}
    for _ in range(3):
        resp = client.post(URL, json=body)
        assert resp.status_code == 200
        assert resp.json() == {"valid": True}


# --- Mismatch verdicts -------------------------------------------------------


def test_existing_claim_with_different_payload_is_valid_false(client):
    claim = _setup_claim(client)
    for different in DIFFERENT_PAYLOADS:
        resp = client.post(
            URL, json={"claim_id": claim["id"], "payload": different}
        )
        assert resp.status_code == 200, different
        assert resp.json() == {"valid": False}, different
        assert resp.text == '{"valid":false}', different


def test_payload_of_one_claim_does_not_verify_against_another(client):
    claim_a = _setup_claim(client, payload=PAYLOAD)
    # A second, distinct claim about the same content/actor/type commits to
    # a different payload; the two digests must never cross-verify.
    other = {"statement": "something else entirely"}
    claim_b = _create_claim(client, claim_a["content_id"], payload=other)
    assert claim_a["id"] != claim_b["id"]
    assert (
        claim_a["payload_digest_hex"] != claim_b["payload_digest_hex"]
    )

    resp = client.post(
        URL, json={"claim_id": claim_b["id"], "payload": PAYLOAD}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": False}

    resp = client.post(
        URL, json={"claim_id": claim_a["id"], "payload": other}
    )
    assert resp.json() == {"valid": False}

    # Each payload verifies only against its own claim.
    assert client.post(
        URL, json={"claim_id": claim_a["id"], "payload": PAYLOAD}
    ).json() == {"valid": True}
    assert client.post(
        URL, json={"claim_id": claim_b["id"], "payload": other}
    ).json() == {"valid": True}


def test_failed_verdict_does_not_echo_payload(client):
    claim = _setup_claim(client)
    secret = "supercalifragilistic-payload-value"
    offered = {"statement": secret}
    resp = client.post(
        URL, json={"claim_id": claim["id"], "payload": offered}
    )
    assert resp.status_code == 200
    assert resp.text == '{"valid":false}'
    assert secret not in resp.text


def test_verdict_agrees_with_creation_digest(client):
    claim = _setup_claim(client)
    resp = client.post(
        URL, json={"claim_id": claim["id"], "payload": PAYLOAD}
    )
    assert resp.json() == {"valid": True}
    # The recomputed digest is exactly the one exposed on creation.
    canonical = json.dumps(
        PAYLOAD, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    import hashlib

    assert claim["payload_digest_algorithm"] == "sha256"
    assert claim["payload_digest_hex"] == hashlib.sha256(canonical).hexdigest()


# --- Unknown claim -----------------------------------------------------------


def test_unknown_claim_id_is_404_claim_not_found(client):
    resp = client.post(
        URL, json={"claim_id": "clm_doesnotexist0000000000000000000000000000",
                   "payload": PAYLOAD}
    )
    assert resp.status_code == 404, resp.text
    body = resp.json()
    assert body["error"]["code"] == "claim_not_found"
    assert (
        body["error"]["details"]["claim_id"]
        == "clm_doesnotexist0000000000000000000000000000"
    )
    # The offered payload is not echoed.
    assert "created by org-1" not in resp.text


def test_404_repeated_and_stable(client):
    body = {
        "claim_id": "clm_doesnotexist0000000000000000000000000000",
        "payload": {},
    }
    for _ in range(2):
        resp = client.post(URL, json=body)
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "claim_not_found"


# --- 422 validation boundary -------------------------------------------------


def test_missing_fields_are_422(client):
    body = _assert_validation_error(client.post(URL, json={}))
    issue_fields = {
        ".".join(part for part in issue["loc"] if part != "body")
        for issue in body["error"]["details"]["issues"]
    }
    assert {"claim_id", "payload"}.issubset(issue_fields)


def test_missing_claim_id_is_422(client):
    _assert_validation_error(client.post(URL, json={"payload": {}}))


def test_missing_payload_is_422(client):
    _assert_validation_error(client.post(
        URL, json={"claim_id": "clm_doesnotexist0000000000000000000000000000"}
    ))


def test_extra_fields_are_422(client):
    body = {"claim_id": "clm_x", "payload": {}, "unexpected": "value"}
    _assert_validation_error(client.post(URL, json=body))


def test_non_object_json_body_is_422(client):
    for raw in ["[]", "123", "null", '"a string"', "4.5", "true"]:
        resp = client.post(
            URL, content=raw, headers={"content-type": "application/json"}
        )
        _assert_validation_error(resp)


def test_malformed_or_empty_json_body_is_422(client):
    for raw in ["{not valid json", "", "   "]:
        resp = client.post(
            URL, content=raw, headers={"content-type": "application/json"}
        )
        _assert_validation_error(resp)


def test_claim_id_empty_or_whitespace_is_422(client):
    for value in ("", "   ", "\t\n "):
        _assert_validation_error(
            client.post(URL, json={"claim_id": value, "payload": {}})
        )


def test_claim_id_non_string_is_422(client):
    for value in (123, None, True, ["x"], {}, 1.5):
        _assert_validation_error(
            client.post(URL, json={"claim_id": value, "payload": {}})
        )


def test_claim_id_overlength_is_422(client):
    _assert_validation_error(
        client.post(
            URL, json={"claim_id": "c" * 81, "payload": {}}
        )
    )


def test_payload_non_object_is_422(client):
    claim_id = "clm_doesnotexist0000000000000000000000000000"
    for value in ([], "a string", 123, None, True, 4.5):
        _assert_validation_error(
            client.post(URL, json={"claim_id": claim_id, "payload": value})
        )


def test_non_finite_numbers_are_422_at_any_depth(client):
    claim_id = "clm_doesnotexist0000000000000000000000000000"
    for raw_payload in (
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":-Infinity}',
        '{"x":[1,{"y":NaN}]}',
        '{"a":{"b":{"c":-Infinity}}}',
    ):
        raw = (
            '{"claim_id":'
            + json.dumps(claim_id)
            + ',"payload":'
            + raw_payload
            + "}"
        )
        resp = client.post(
            URL,
            content=raw,
            headers={"content-type": "application/json"},
        )
        _assert_validation_error(resp)


def test_any_query_parameter_is_422(client):
    claim = _setup_claim(client)
    for suffix in ("?x=1", "?claim_id=clm_x", "?x=1&x=2", "?="):
        resp = client.post(
            URL + suffix, json={"claim_id": claim["id"], "payload": PAYLOAD}
        )
        _assert_validation_error(resp)


def test_structural_validation_precedes_claim_lookup(client):
    # Even with an existing, readable claim, a structurally bad body is a
    # 422 -- never a 200 verdict, never partial output.
    claim = _setup_claim(client)
    _assert_validation_error(
        client.post(URL, json={"claim_id": claim["id"], "payload": []})
    )
    _assert_validation_error(
        client.post(URL, json={"claim_id": claim["id"]})
    )
    _assert_validation_error(
        client.post(
            URL + "?x=1",
            json={"claim_id": claim["id"], "payload": PAYLOAD},
        )
    )
    # And with an unknown claim id the structural failure still wins over
    # the 404 lookup, so no claim read affects the response.
    _assert_validation_error(
        client.post(
            URL,
            json={
                "claim_id": "clm_doesnotexist0000000000000000000000000000",
                "payload": None,
            },
        )
    )


def test_validation_error_body_never_echoes_payload(client):
    claim = _setup_claim(client)
    secret = "another-secret-payload-value"
    resp = client.post(
        URL, json={"claim_id": claim["id"], "payload": secret}
    )
    assert resp.status_code == 422
    assert secret not in resp.text


# --- Persistence is untouched ------------------------------------------------


def test_verifications_write_no_rows_or_audit_events(client, db_session):
    claim = _setup_claim(client)
    claims_before = len(db_session.execute(select(Claim)).scalars().all())
    events_before = len(db_session.execute(select(AuditEvent)).scalars().all())

    bodies = [
        {"claim_id": claim["id"], "payload": PAYLOAD},  # valid true
        {"claim_id": claim["id"], "payload": {"nope": True}},  # valid false
        {
            "claim_id": "clm_doesnotexist0000000000000000000000000000",
            "payload": PAYLOAD,
        },  # 404
        {"claim_id": claim["id"], "payload": []},  # 422
        {"claim_id": claim["id"]},  # 422
        {"claim_id": claim["id"], "payload": {}, "extra": 1},  # 422
    ]
    statuses = [client.post(URL, json=body).status_code for body in bodies]
    assert statuses == [200, 200, 404, 422, 422, 422]

    assert (
        len(db_session.execute(select(Claim)).scalars().all())
        == claims_before
    )
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )
    # The verified claim is returned byte-for-byte unchanged afterwards.
    after = client.get(f"/v1/claims/{claim['id']}")
    assert after.status_code == 200
    assert after.json() == claim


def test_entire_database_file_is_unchanged_by_verifications(
    tmp_db_url, file_client
):
    claim = _setup_claim(file_client)

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
        {"claim_id": claim["id"], "payload": PAYLOAD},
        {"claim_id": claim["id"], "payload": {"different": True}},
        {
            "claim_id": "clm_doesnotexist0000000000000000000000000000",
            "payload": {},
        },
        {"claim_id": claim["id"], "payload": None},
    ]
    statuses = [file_client.post(URL, json=body).status_code for body in bodies]
    assert statuses == [200, 200, 404, 422]

    assert table_counts() == before


def test_concurrent_verifications_are_consistent_and_read_only(file_client):
    claim = _setup_claim(file_client)
    barrier = threading.Barrier(8)
    results: list[int] = []
    errors: list[Exception] = []

    def worker() -> None:
        barrier.wait()
        try:
            for _ in range(5):
                r1 = file_client.post(
                    URL, json={"claim_id": claim["id"], "payload": PAYLOAD}
                )
                r2 = file_client.post(
                    URL,
                    json={"claim_id": claim["id"], "payload": {"other": 1}},
                )
                results.append((r1.status_code, r1.json()["valid"]))
                results.append((r2.status_code, r2.json()["valid"]))
        except Exception as exc:  # pragma: no cover - failure reporting
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 80
    assert set(results) == {(200, True), (200, False)}

    # Repeat and concurrent reads changed nothing: the claim still verifies.
    follow_up = file_client.post(
        URL, json={"claim_id": claim["id"], "payload": PAYLOAD}
    )
    assert follow_up.status_code == 200
    assert follow_up.json() == {"valid": True}
