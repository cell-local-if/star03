"""Tests for the read-only minimal-disclosure content privacy export.

Covers ``GET /v1/contents/{content_id}/privacy-export``:

* a content without claims -> ``claim_count`` 0 and an empty ``claims``
  array;
* claims in stable creation order, each carrying exactly ``claim_id``,
  ``claim_type``, ``payload_digest_algorithm``, ``payload_digest_hex``,
  ``created_at``, and ``evidence_bundles`` -- never the actor id, the raw
  payload, or any signature or key material;
* evidence bundles in stable creation order, each carrying exactly
  ``evidence_bundle_id``, ``evidence_type``, ``digest_algorithm``,
  ``digest_hex``, and ``media_type`` -- never the metadata object, the
  claim id, or any content bytes;
* the success body is compact UTF-8 JSON terminated by exactly one newline
  with members in the exact documented order;
* any body (whitespace, malformed JSON, an object), any query parameter
  (unknown, blank, or repeated), and a blank content id are 422
  ``validation_error``; an unknown content id is 404 ``content_not_found``;
  a non-GET method is 405 ``method_not_allowed``;
* success and failure are strictly read-only (no resource, task, or audit
  writes) and repeated reads, concurrent reads, and reads across an app
  restart return byte-identical bodies;
* the existing full export, claim, and evidence-bundle reads are
  unaffected by the new view.

All fixtures are deterministic and offline (in-memory and temporary-file
SQLite).
"""

from __future__ import annotations

import hashlib
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

EVIDENCE_DIGEST_A = hashlib.sha256(b"privacy-export-evidence-a").hexdigest()
EVIDENCE_DIGEST_B = hashlib.sha256(b"privacy-export-evidence-b").hexdigest()
EVIDENCE_DIGEST_C = hashlib.sha256(b"privacy-export-evidence-c").hexdigest()


def _url(content_id: str) -> str:
    return f"/v1/contents/{content_id}/privacy-export"


# --- Setup helpers ----------------------------------------------------------


def _create_content(client, actor_id="org-1", digest=DIGEST_A):
    resp = client.post(
        "/v1/contents", json=content_payload(actor_id=actor_id, digest=digest)
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


def _create_bundle(client, claim_id, digest=EVIDENCE_DIGEST_A,
                   media_type="image/jpeg"):
    resp = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim_id,
            "evidence_type": "raw_capture",
            "digest_algorithm": "sha256",
            "digest_hex": digest,
            "media_type": media_type,
            "metadata": {"source": "camera-1", "note": "sensitive"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _audit_count(db_session) -> int:
    return len(db_session.execute(select(AuditEvent)).scalars().all())


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# --- empty and minimal views -------------------------------------------------


def test_privacy_export_without_claims_is_empty(client, db_session):
    create_actor(client)
    content = _create_content(client)
    events_before = _audit_count(db_session)

    resp = client.get(_url(content["id"]))
    assert resp.status_code == 200, resp.text
    # Compact JSON terminated by exactly one newline, exact member order.
    assert resp.content == (
        b'{"content_id":"' + content["id"].encode("ascii") + b'",'
        b'"claim_count":0,"claims":[]}\n'
    )
    body = resp.json()
    assert list(body) == ["content_id", "claim_count", "claims"]
    assert body == {
        "content_id": content["id"],
        "claim_count": 0,
        "claims": [],
    }
    # Strictly read-only: no audit event was added.
    assert _audit_count(db_session) == events_before


def test_privacy_export_claim_without_bundles_has_empty_bundle_list(client):
    create_actor(client)
    content = _create_content(client)
    claim = _create_claim(client, content["id"])

    body = client.get(_url(content["id"])).json()
    assert body["claim_count"] == 1
    assert len(body["claims"]) == 1
    item = body["claims"][0]
    assert list(item) == [
        "claim_id",
        "claim_type",
        "payload_digest_algorithm",
        "payload_digest_hex",
        "created_at",
        "evidence_bundles",
    ]
    assert item["claim_id"] == claim["id"]
    assert item["claim_type"] == "authorship"
    assert item["payload_digest_algorithm"] == "sha256"
    assert item["payload_digest_hex"] == claim["payload_digest_hex"]
    assert _parse_ts(item["created_at"]) == _parse_ts(claim["created_at"])
    assert item["evidence_bundles"] == []


def test_privacy_export_claims_and_bundles_in_creation_order(client):
    create_actor(client)
    content = _create_content(client)
    claim_a = _create_claim(client, content["id"])
    claim_b = _create_claim(client, content["id"], claim_type="review")
    bundle_a1 = _create_bundle(client, claim_a["id"])
    bundle_a2 = _create_bundle(
        client, claim_a["id"], digest=EVIDENCE_DIGEST_B, media_type="video/mp4"
    )
    bundle_b1 = _create_bundle(client, claim_b["id"], digest=EVIDENCE_DIGEST_C)

    body = client.get(_url(content["id"])).json()
    assert body["claim_count"] == 2
    assert [c["claim_id"] for c in body["claims"]] == [
        claim_a["id"],
        claim_b["id"],
    ]
    first, second = body["claims"]
    assert first["claim_type"] == "authorship"
    assert second["claim_type"] == "review"
    assert [b["evidence_bundle_id"] for b in first["evidence_bundles"]] == [
        bundle_a1["id"],
        bundle_a2["id"],
    ]
    assert [b["evidence_bundle_id"] for b in second["evidence_bundles"]] == [
        bundle_b1["id"]
    ]
    bundle = first["evidence_bundles"][0]
    assert list(bundle) == [
        "evidence_bundle_id",
        "evidence_type",
        "digest_algorithm",
        "digest_hex",
        "media_type",
    ]
    assert bundle == {
        "evidence_bundle_id": bundle_a1["id"],
        "evidence_type": "raw_capture",
        "digest_algorithm": "sha256",
        "digest_hex": EVIDENCE_DIGEST_A,
        "media_type": "image/jpeg",
    }
    assert first["evidence_bundles"][1]["media_type"] == "video/mp4"


# --- minimal disclosure -------------------------------------------------------


def test_privacy_export_never_discloses_private_fields(client):
    create_actor(client)
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    _create_bundle(client, claim["id"])

    raw = client.get(_url(content["id"])).content
    body = json.loads(raw.decode("utf-8"))
    claim_item = body["claims"][0]
    bundle_item = claim_item["evidence_bundles"][0]
    # Exactly the documented fields, nothing else.
    assert set(claim_item) == {
        "claim_id",
        "claim_type",
        "payload_digest_algorithm",
        "payload_digest_hex",
        "created_at",
        "evidence_bundles",
    }
    assert set(bundle_item) == {
        "evidence_bundle_id",
        "evidence_type",
        "digest_algorithm",
        "digest_hex",
        "media_type",
    }
    # No actor reference, payload, metadata, signature, or key material can
    # appear anywhere in the document.
    for forbidden in (
        b"actor_id",
        b"payload\"",
        b"metadata",
        b"signature",
        b"private",
        b"camera-1",
        b"sensitive",
        b"private detail",
    ):
        assert forbidden not in raw


def test_privacy_export_excludes_other_contents(client):
    create_actor(client)
    content = _create_content(client)
    other = _create_content(client, digest=DIGEST_B)
    other_claim = _create_claim(client, other["id"])
    _create_bundle(client, other_claim["id"])

    body = client.get(_url(content["id"])).json()
    assert body["claim_count"] == 0
    assert body["claims"] == []

    other_body = client.get(_url(other["id"])).json()
    assert other_body["claim_count"] == 1
    assert other_body["claims"][0]["claim_id"] == other_claim["id"]


# --- wire format ---------------------------------------------------------------


def test_privacy_export_body_is_compact_json_with_single_trailing_newline(
    client,
):
    create_actor(client)
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    _create_bundle(client, claim["id"])

    resp = client.get(_url(content["id"]))
    assert resp.status_code == 200, resp.text
    raw = resp.content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    # Compact separators: no incidental whitespace inside the document.
    assert b" " not in raw
    assert b": " not in raw
    body = json.loads(raw.decode("utf-8"))
    assert list(body) == ["content_id", "claim_count", "claims"]
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
def test_privacy_export_validation_errors_422_write_nothing(
    client, db_session, send
):
    create_actor(client)
    content = _create_content(client)
    events_before = _audit_count(db_session)

    resp = send(client, _url(content["id"]))
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"

    # A rejected request reads and writes nothing.
    assert _audit_count(db_session) == events_before
    ok = client.get(_url(content["id"]))
    assert ok.status_code == 200
    assert ok.json()["claim_count"] == 0


def test_privacy_export_blank_content_id_is_422(client):
    create_actor(client)
    _create_content(client)

    resp = client.get("/v1/contents/%20/privacy-export")
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def test_privacy_export_unknown_content_id_is_404(client):
    create_actor(client)
    _create_content(client)

    resp = client.get(_url("ctn_" + "0" * 64))
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "content_not_found"


def test_privacy_export_query_param_beats_unknown_content_404(client):
    # Parameters are validated before the content lookup.
    resp = client.get(_url("ctn_" + "0" * 64) + "?x=1")
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_privacy_export_non_get_method_is_405(client, method):
    create_actor(client)
    content = _create_content(client)

    resp = client.request(method, _url(content["id"]))
    assert resp.status_code == 405, resp.text
    assert resp.json()["error"]["code"] == "method_not_allowed"


# --- determinism, concurrency, and restart stability ----------------------------


def test_privacy_export_repeated_reads_are_identical(client, db_session):
    create_actor(client)
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    _create_bundle(client, claim["id"])
    events_before = _audit_count(db_session)

    first = client.get(_url(content["id"]))
    second = client.get(_url(content["id"]))
    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    assert _audit_count(db_session) == events_before


def test_privacy_export_concurrent_reads_are_identical(client):
    from concurrent.futures import ThreadPoolExecutor

    create_actor(client)
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    _create_bundle(client, claim["id"])
    expected = client.get(_url(content["id"])).content

    with ThreadPoolExecutor(max_workers=8) as pool:
        bodies = list(
            pool.map(lambda _: client.get(_url(content["id"])).content, range(16))
        )
    assert all(body == expected for body in bodies)


def test_privacy_export_is_stable_across_restart(tmp_db_url):
    app1 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app1) as first:
        create_actor(first)
        content = _create_content(first)
        claim_a = _create_claim(first, content["id"])
        claim_b = _create_claim(first, content["id"], claim_type="review")
        bundle_a1 = _create_bundle(first, claim_a["id"])
        bundle_a2 = _create_bundle(
            first, claim_a["id"], digest=EVIDENCE_DIGEST_B
        )
        bundle_b1 = _create_bundle(first, claim_b["id"], digest=EVIDENCE_DIGEST_C)
        before_restart = first.get(_url(content["id"])).content

    # A brand-new process/app over the same database file.
    app2 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app2) as second:
        after_restart = second.get(_url(content["id"])).content

    assert after_restart == before_restart
    body = json.loads(after_restart.decode("utf-8"))
    assert body["content_id"] == content["id"]
    assert body["claim_count"] == 2
    assert [c["claim_id"] for c in body["claims"]] == [
        claim_a["id"],
        claim_b["id"],
    ]
    assert [
        b["evidence_bundle_id"] for b in body["claims"][0]["evidence_bundles"]
    ] == [bundle_a1["id"], bundle_a2["id"]]
    assert [
        b["evidence_bundle_id"] for b in body["claims"][1]["evidence_bundles"]
    ] == [bundle_b1["id"]]


# --- the existing surfaces are unaffected ---------------------------------------


def test_privacy_export_does_not_change_existing_export_or_reads(client):
    create_actor(client)
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    bundle = _create_bundle(client, claim["id"])

    export_before = client.get(f"/v1/contents/{content['id']}/export")
    claim_before = client.get(f"/v1/claims/{claim['id']}")
    bundles_before = client.get(f"/v1/claims/{claim['id']}/evidence-bundles")

    resp = client.get(_url(content["id"]))
    assert resp.status_code == 200, resp.text

    export_after = client.get(f"/v1/contents/{content['id']}/export")
    claim_after = client.get(f"/v1/claims/{claim['id']}")
    bundles_after = client.get(f"/v1/claims/{claim['id']}/evidence-bundles")

    assert export_after.content == export_before.content
    assert claim_after.content == claim_before.content
    assert bundles_after.content == bundles_before.content
    # The full export still carries the fields the privacy view withholds.
    export_body = export_after.json()
    assert export_body["claims"][0]["actor_id"] == "org-1"
    assert export_body["claims"][0]["evidence_bundles"][0]["metadata"] == {
        "source": "camera-1",
        "note": "sensitive",
    }
    assert export_body["claims"][0]["evidence_bundles"][0]["id"] == bundle["id"]
