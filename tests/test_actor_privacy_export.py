"""Tests for the read-only minimal-disclosure actor privacy export.

Covers ``GET /v1/actors/{actor_id}/privacy-export``:

* an actor without contents or claims -> both counts 0 and both arrays
  empty;
* the actor view carries exactly ``id``, ``name``, ``type``, ``created_at``;
* contents in stable creation order, each carrying exactly ``content_id``,
  ``digest_algorithm``, ``digest_hex``, ``media_type``, and ``created_at``
  -- never the title, the actor id, or any content bytes;
* claims in stable creation order, each carrying exactly ``claim_id``,
  ``content_id``, ``claim_type``, ``payload_digest_algorithm``,
  ``payload_digest_hex``, and ``created_at`` -- never the actor id, the raw
  payload, or any signature or key material;
* only records directly attributed to the actor appear: no version,
  derivation, supersession, or evidence expansion, and no other actor's
  records;
* the success body is compact UTF-8 JSON terminated by exactly one newline
  with members in the exact documented order;
* any body (whitespace, malformed JSON, an object), any query parameter
  (unknown, blank, or repeated), and a blank actor id are 422
  ``validation_error``; an unknown actor id is 404 ``actor_not_found``;
  a non-GET method is 405 ``method_not_allowed``;
* success and failure are strictly read-only (no resource, task, or audit
  writes) and repeated reads, concurrent reads, and reads across an app
  restart return byte-identical bodies;
* the existing actor, content export, content privacy export, claim, and
  evidence-bundle reads are unaffected by the new view.

All fixtures are deterministic and offline (in-memory and temporary-file
SQLite).
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from provenance.app import create_app
from provenance.config import Settings
from provenance.models import AuditEvent
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    DIGEST_C,
    content_payload,
    create_actor,
)


def _url(actor_id: str) -> str:
    return f"/v1/actors/{actor_id}/privacy-export"


# --- Setup helpers ----------------------------------------------------------


def _create_content(client, actor_id="org-1", digest=DIGEST_A, title=None):
    resp = client.post(
        "/v1/contents",
        json=content_payload(actor_id=actor_id, digest=digest, title=title),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, actor_id="org-1", claim_type="authorship"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": claim_type,
            "payload": {"statement": "private detail"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _audit_count(db_session) -> int:
    return len(db_session.execute(select(AuditEvent)).scalars().all())


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# --- empty and minimal views -------------------------------------------------


def test_actor_privacy_export_without_records_is_empty(client, db_session):
    actor = create_actor(client)
    events_before = _audit_count(db_session)

    resp = client.get(_url(actor["id"]))
    assert resp.status_code == 200, resp.text
    # Compact JSON terminated by exactly one newline, exact member order.
    assert resp.content == (
        b'{"actor":{"id":"org-1","name":"Example Org",'
        b'"type":"organization","created_at":"'
        + actor["created_at"].encode("ascii")
        + b'"},"content_count":0,"contents":[],"claim_count":0,"claims":[]}\n'
    )
    body = resp.json()
    assert list(body) == [
        "actor",
        "content_count",
        "contents",
        "claim_count",
        "claims",
    ]
    assert list(body["actor"]) == ["id", "name", "type", "created_at"]
    assert body["actor"] == {
        "id": actor["id"],
        "name": "Example Org",
        "type": "organization",
        "created_at": actor["created_at"],
    }
    assert body["content_count"] == 0
    assert body["contents"] == []
    assert body["claim_count"] == 0
    assert body["claims"] == []
    # Strictly read-only: no audit event was added.
    assert _audit_count(db_session) == events_before


def test_actor_privacy_export_contents_and_claims_in_creation_order(client):
    create_actor(client)
    content_a = _create_content(client, digest=DIGEST_A, title="secret title")
    content_b = _create_content(client, digest=DIGEST_B)
    claim_a = _create_claim(client, content_a["id"])
    claim_b = _create_claim(client, content_b["id"], claim_type="review")

    body = client.get(_url("org-1")).json()
    assert body["content_count"] == 2
    assert [c["content_id"] for c in body["contents"]] == [
        content_a["id"],
        content_b["id"],
    ]
    first = body["contents"][0]
    assert list(first) == [
        "content_id",
        "digest_algorithm",
        "digest_hex",
        "media_type",
        "created_at",
    ]
    assert first == {
        "content_id": content_a["id"],
        "digest_algorithm": "sha256",
        "digest_hex": DIGEST_A,
        "media_type": "image/png",
        "created_at": content_a["created_at"],
    }
    assert body["claim_count"] == 2
    assert [c["claim_id"] for c in body["claims"]] == [
        claim_a["id"],
        claim_b["id"],
    ]
    first_claim = body["claims"][0]
    assert list(first_claim) == [
        "claim_id",
        "content_id",
        "claim_type",
        "payload_digest_algorithm",
        "payload_digest_hex",
        "created_at",
    ]
    assert first_claim == {
        "claim_id": claim_a["id"],
        "content_id": content_a["id"],
        "claim_type": "authorship",
        "payload_digest_algorithm": "sha256",
        "payload_digest_hex": claim_a["payload_digest_hex"],
        "created_at": claim_a["created_at"],
    }
    assert body["claims"][1]["claim_type"] == "review"
    # Timestamps are strict RFC 3339 UTC.
    for item in (*body["contents"], *body["claims"]):
        assert item["created_at"].endswith(("Z", "+00:00"))
        _parse_ts(item["created_at"])


def test_actor_privacy_export_counts_match_array_lengths(client):
    create_actor(client)
    content = _create_content(client)
    _create_claim(client, content["id"])
    _create_claim(client, content["id"], claim_type="review")

    body = client.get(_url("org-1")).json()
    assert body["content_count"] == len(body["contents"]) == 1
    assert body["claim_count"] == len(body["claims"]) == 2


# --- minimal disclosure and direct attribution --------------------------------


def test_actor_privacy_export_never_discloses_private_fields(client):
    create_actor(client)
    content = _create_content(client, title="secret title")
    claim = _create_claim(client, content["id"])
    # Evidence attached to the claim is never expanded into the export.
    resp = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim["id"],
            "evidence_type": "raw_capture",
            "digest_algorithm": "sha256",
            "digest_hex": DIGEST_C,
            "media_type": "image/jpeg",
            "metadata": {"source": "camera-1", "note": "sensitive"},
        },
    )
    assert resp.status_code == 201, resp.text

    raw = client.get(_url("org-1")).content
    body = json.loads(raw.decode("utf-8"))
    assert set(body["actor"]) == {"id", "name", "type", "created_at"}
    assert set(body["contents"][0]) == {
        "content_id",
        "digest_algorithm",
        "digest_hex",
        "media_type",
        "created_at",
    }
    assert set(body["claims"][0]) == {
        "claim_id",
        "content_id",
        "claim_type",
        "payload_digest_algorithm",
        "payload_digest_hex",
        "created_at",
    }
    # No actor reference inside the items, no title, payload, metadata,
    # evidence, signature, or key material anywhere in the document.
    for forbidden in (
        b"actor_id",
        b"title",
        b"secret title",
        b"payload\"",
        b"metadata",
        b"evidence",
        b"signature",
        b"private",
        b"camera-1",
        b"sensitive",
        b"private detail",
    ):
        assert forbidden not in raw


def test_actor_privacy_export_excludes_other_actors_records(client):
    create_actor(client)
    create_actor(client, actor_id="org-2", name="Other Org")
    own_content = _create_content(client, actor_id="org-1", digest=DIGEST_A)
    other_content = _create_content(client, actor_id="org-2", digest=DIGEST_B)
    own_claim = _create_claim(client, own_content["id"], actor_id="org-1")
    _create_claim(client, other_content["id"], actor_id="org-2")
    # A claim by the other actor about this actor's content is not exported.
    _create_claim(client, own_content["id"], actor_id="org-2",
                  claim_type="review")

    body = client.get(_url("org-1")).json()
    assert body["content_count"] == 1
    assert [c["content_id"] for c in body["contents"]] == [own_content["id"]]
    assert body["claim_count"] == 1
    assert [c["claim_id"] for c in body["claims"]] == [own_claim["id"]]

    other_body = client.get(_url("org-2")).json()
    assert other_body["content_count"] == 1
    assert other_body["contents"][0]["content_id"] == other_content["id"]
    # Both claims org-2 made (on its own and on org-1's content) are its
    # direct records; org-1's claim never appears here.
    assert other_body["claim_count"] == 2
    assert own_claim["id"] not in [c["claim_id"] for c in other_body["claims"]]


def test_actor_privacy_export_does_not_expand_supersessions(client):
    create_actor(client)
    content = _create_content(client)
    claim_a = _create_claim(client, content["id"])
    claim_b = _create_claim(client, content["id"], claim_type="review")
    resp = client.post(
        "/v1/claim-supersessions",
        json={
            "superseded_claim_id": claim_a["id"],
            "replacement_claim_id": claim_b["id"],
            "reason": "correction",
        },
    )
    assert resp.status_code == 201, resp.text

    body = client.get(_url("org-1")).json()
    # Both claims remain direct records; the supersession adds nothing.
    assert body["claim_count"] == 2
    assert [c["claim_id"] for c in body["claims"]] == [
        claim_a["id"],
        claim_b["id"],
    ]
    assert b"supersession" not in client.get(_url("org-1")).content


# --- wire format ---------------------------------------------------------------


def test_actor_privacy_export_body_is_compact_json_with_single_trailing_newline(
    client,
):
    create_actor(client)
    content = _create_content(client)
    _create_claim(client, content["id"])

    resp = client.get(_url("org-1"))
    assert resp.status_code == 200, resp.text
    raw = resp.content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    # Compact separators: no incidental whitespace after colons or commas
    # (the actor name itself legitimately contains a space).
    assert b": " not in raw
    assert b", " not in raw
    body = json.loads(raw.decode("utf-8"))
    assert list(body) == [
        "actor",
        "content_count",
        "contents",
        "claim_count",
        "claims",
    ]
    assert isinstance(body["content_count"], int)
    assert isinstance(body["claim_count"], int)


# --- request validation --------------------------------------------------------


@pytest.mark.parametrize(
    "send",
    [
        # Any non-empty body is rejected, even whitespace.
        lambda c, url: c.request("GET", url, content=b"   "),
        # Malformed JSON is a validation error, not a parser crash.
        lambda c, url: c.request(
            "GET",
            url,
            content=b"{not json",
            headers={"content-type": "application/json"},
        ),
        # An empty JSON object is still a non-empty body.
        lambda c, url: c.request("GET", url, content=b"{}"),
        # No query parameters are accepted in any shape.
        lambda c, url: c.get(url + "?x=1"),
        lambda c, url: c.get(url + "?limit=10"),
        lambda c, url: c.get(url + "?x="),
        lambda c, url: c.get(url + "?x=1&x=2"),
        # A body together with a query parameter is still a 422.
        lambda c, url: c.request("GET", url + "?x=1", content=b"{}"),
    ],
)
def test_actor_privacy_export_validation_errors_422_write_nothing(
    client, db_session, send
):
    create_actor(client)
    events_before = _audit_count(db_session)

    resp = send(client, _url("org-1"))
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"

    # A rejected request reads and writes nothing.
    assert _audit_count(db_session) == events_before
    ok = client.get(_url("org-1"))
    assert ok.status_code == 200
    assert ok.json()["content_count"] == 0


def test_actor_privacy_export_blank_actor_id_is_422(client):
    create_actor(client)

    resp = client.get("/v1/actors/%20/privacy-export")
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def test_actor_privacy_export_unknown_actor_id_is_404(client):
    create_actor(client)

    resp = client.get(_url("org-missing"))
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "actor_not_found"


def test_actor_privacy_export_query_param_beats_unknown_actor_404(client):
    # Parameters are validated before the actor lookup.
    resp = client.get(_url("org-missing") + "?x=1")
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_actor_privacy_export_non_get_method_is_405(client, method):
    create_actor(client)

    resp = client.request(method, _url("org-1"))
    assert resp.status_code == 405, resp.text
    assert resp.json()["error"]["code"] == "method_not_allowed"


# --- determinism, concurrency, and restart stability ----------------------------


def test_actor_privacy_export_repeated_reads_are_identical(client, db_session):
    create_actor(client)
    content = _create_content(client)
    _create_claim(client, content["id"])
    events_before = _audit_count(db_session)

    first = client.get(_url("org-1"))
    second = client.get(_url("org-1"))
    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    assert _audit_count(db_session) == events_before


def test_actor_privacy_export_concurrent_reads_are_identical(client):
    from concurrent.futures import ThreadPoolExecutor

    create_actor(client)
    content = _create_content(client)
    _create_claim(client, content["id"])
    expected = client.get(_url("org-1")).content

    with ThreadPoolExecutor(max_workers=8) as pool:
        bodies = list(
            pool.map(lambda _: client.get(_url("org-1")).content, range(16))
        )
    assert all(body == expected for body in bodies)


def test_actor_privacy_export_is_stable_across_restart(tmp_db_url):
    app1 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app1) as first:
        create_actor(first)
        content_a = _create_content(first, digest=DIGEST_A)
        content_b = _create_content(first, digest=DIGEST_B)
        claim_a = _create_claim(first, content_a["id"])
        claim_b = _create_claim(first, content_b["id"], claim_type="review")
        before_restart = first.get(_url("org-1")).content

    # A brand-new process/app over the same database file.
    app2 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app2) as second:
        after_restart = second.get(_url("org-1")).content

    assert after_restart == before_restart
    body = json.loads(after_restart.decode("utf-8"))
    assert body["actor"]["id"] == "org-1"
    assert body["content_count"] == 2
    assert [c["content_id"] for c in body["contents"]] == [
        content_a["id"],
        content_b["id"],
    ]
    assert body["claim_count"] == 2
    assert [c["claim_id"] for c in body["claims"]] == [
        claim_a["id"],
        claim_b["id"],
    ]


# --- the existing surfaces are unaffected ---------------------------------------


def test_actor_privacy_export_does_not_change_existing_reads(client):
    create_actor(client)
    content = _create_content(client)
    claim = _create_claim(client, content["id"])

    actors_before = client.get("/v1/actors")
    export_before = client.get(f"/v1/contents/{content['id']}/export")
    privacy_before = client.get(f"/v1/contents/{content['id']}/privacy-export")
    claim_before = client.get(f"/v1/claims/{claim['id']}")

    resp = client.get(_url("org-1"))
    assert resp.status_code == 200, resp.text

    assert client.get("/v1/actors").content == actors_before.content
    assert (
        client.get(f"/v1/contents/{content['id']}/export").content
        == export_before.content
    )
    assert (
        client.get(f"/v1/contents/{content['id']}/privacy-export").content
        == privacy_before.content
    )
    assert client.get(f"/v1/claims/{claim['id']}").content == claim_before.content
    # The full export still carries the fields the privacy view withholds.
    export_body = client.get(f"/v1/contents/{content['id']}/export").json()
    assert export_body["claims"][0]["actor_id"] == "org-1"
