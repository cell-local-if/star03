"""Tests for the read-only minimal-disclosure content privacy export.

Covers ``GET /v1/contents/{content_id}/privacy-export``:

* a content without claims -> ``claim_count`` 0 and an empty ``claims``
  array;
* claims and evidence bundles appear in stable creation order, each claim
  carrying exactly ``claim_id``, ``claim_type``,
  ``payload_digest_algorithm``, ``payload_digest_hex``, ``created_at``, and
  ``evidence_bundles``, and each bundle exactly ``evidence_bundle_id``,
  ``evidence_type``, ``digest_algorithm``, ``digest_hex``, ``media_type``;
* no evidence metadata, actor id, raw payload, signature, key material, or
  content bytes ever appear;
* the success body is compact UTF-8 JSON terminated by exactly one newline
  with members in the exact documented order;
* any body (whitespace, malformed JSON, an object), any query parameter
  (unknown, blank, or repeated), and a blank content id are 422
  ``validation_error``; an unknown content id is 404 ``content_not_found``;
  non-GET methods are 405 ``method_not_allowed``;
* success and failure are strictly read-only (no resource, task, or audit
  writes) and repeated reads, including across an app restart, return the
  same fields and order;
* the existing full export and claim/bundle reads are unchanged.

All fixtures are deterministic and offline (in-memory and temporary-file
SQLite).
"""

from __future__ import annotations

import hashlib
import json

import pytest
from sqlalchemy import select

from provenance.app import create_app
from provenance.config import Settings
from provenance.models import AuditEvent
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    content_payload,
    create_actor,
)
from fastapi.testclient import TestClient

EVIDENCE_DIGEST_A = hashlib.sha256(b"privacy-export-evidence-a").hexdigest()
EVIDENCE_DIGEST_B = hashlib.sha256(b"privacy-export-evidence-b").hexdigest()


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
            "payload": {"statement": "private"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_bundle(client, claim_id, digest=EVIDENCE_DIGEST_A):
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


def _audit_count(db_session) -> int:
    return len(db_session.execute(select(AuditEvent)).scalars().all())


# --- success shape -----------------------------------------------------------


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


def test_privacy_export_claims_and_bundles_minimal_fields_in_order(client):
    create_actor(client)
    content = _create_content(client)
    claim_a = _create_claim(client, content["id"])
    claim_b = _create_claim(client, content["id"], claim_type="review")
    bundle_a1 = _create_bundle(client, claim_a["id"])
    bundle_a2 = _create_bundle(client, claim_a["id"], digest=EVIDENCE_DIGEST_B)

    resp = client.get(_url(content["id"]))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert list(body) == ["content_id", "claim_count", "claims"]
    assert body["content_id"] == content["id"]
    assert body["claim_count"] == 2

    claims = body["claims"]
    # Claims follow their stable creation order.
    assert [c["claim_id"] for c in claims] == [claim_a["id"], claim_b["id"]]
    for claim, raw in zip(claims, (claim_a, claim_b)):
        assert list(claim) == [
            "claim_id",
            "claim_type",
            "payload_digest_algorithm",
            "payload_digest_hex",
            "created_at",
            "evidence_bundles",
        ]
        assert claim["claim_type"] == raw["claim_type"]
        assert (
            claim["payload_digest_algorithm"] == raw["payload_digest_algorithm"]
        )
        assert claim["payload_digest_hex"] == raw["payload_digest_hex"]
        assert claim["created_at"] == raw["created_at"]

    # Bundles follow their stable creation order with minimal fields only.
    bundles = claims[0]["evidence_bundles"]
    assert [b["evidence_bundle_id"] for b in bundles] == [
        bundle_a1["id"],
        bundle_a2["id"],
    ]
    for bundle, raw in zip(bundles, (bundle_a1, bundle_a2)):
        assert list(bundle) == [
            "evidence_bundle_id",
            "evidence_type",
            "digest_algorithm",
            "digest_hex",
            "media_type",
        ]
        assert bundle["evidence_type"] == raw["evidence_type"]
        assert bundle["digest_algorithm"] == raw["digest_algorithm"]
        assert bundle["digest_hex"] == raw["digest_hex"]
        assert bundle["media_type"] == raw["media_type"]
    # A claim without bundles exports an empty array.
    assert claims[1]["evidence_bundles"] == []


def test_privacy_export_never_discloses_private_fields(client):
    create_actor(client)
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    _create_bundle(client, claim["id"])

    raw = client.get(_url(content["id"])).content
    text = raw.decode("utf-8")
    # No actor, metadata, raw payload, signature, key, or content-byte
    # material: neither the field names nor the stored values appear.
    for forbidden in (
        "actor_id",
        "metadata",
        "signature",
        "public_key",
        "private_key",
        # The raw claim payload's members and values.
        "statement",
        "private",
        # The evidence metadata's members and values.
        "source",
        "camera-1",
        # The content's own digest never appears (only claim/bundle
        # digests do).
        content["digest_hex"],
    ):
        assert forbidden not in text
    body = json.loads(text)
    claim_item = body["claims"][0]
    assert set(claim_item) == {
        "claim_id",
        "claim_type",
        "payload_digest_algorithm",
        "payload_digest_hex",
        "created_at",
        "evidence_bundles",
    }
    assert set(claim_item["evidence_bundles"][0]) == {
        "evidence_bundle_id",
        "evidence_type",
        "digest_algorithm",
        "digest_hex",
        "media_type",
    }


def test_privacy_export_excludes_other_contents_and_never_traverses_lineage(
    client,
):
    create_actor(client)
    content = _create_content(client)
    other = _create_content(client, digest=DIGEST_B)
    other_claim = _create_claim(client, other["id"])
    _create_bundle(client, other_claim["id"])
    # A lineage relation does not pull the other content's claims in.
    resp = client.post(
        "/v1/content-relations",
        json={
            "content_id": content["id"],
            "parent_content_id": other["id"],
            "relation_type": "derived_from",
        },
    )
    assert resp.status_code == 201, resp.text

    body = client.get(_url(content["id"])).json()
    assert body["claim_count"] == 0
    assert body["claims"] == []

    # The other content's own export is unaffected.
    other_body = client.get(_url(other["id"])).json()
    assert other_body["claim_count"] == 1
    assert [c["claim_id"] for c in other_body["claims"]] == [other_claim["id"]]


# --- wire format -------------------------------------------------------------


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


# --- request validation ------------------------------------------------------


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
def test_privacy_export_non_get_methods_are_405(client, method):
    create_actor(client)
    content = _create_content(client)

    resp = client.request(method, _url(content["id"]))
    assert resp.status_code == 405, resp.text
    assert resp.json()["error"]["code"] == "method_not_allowed"


# --- determinism and restart stability ---------------------------------------


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


def test_privacy_export_is_stable_across_restart(tmp_db_url):
    app1 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app1) as first:
        create_actor(first)
        content = _create_content(first)
        claim = _create_claim(first, content["id"])
        _create_bundle(first, claim["id"])
        before_restart = first.get(_url(content["id"])).content

    # A brand-new process/app over the same database file.
    app2 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app2) as second:
        after_restart = second.get(_url(content["id"])).content

    assert after_restart == before_restart
    body = json.loads(after_restart.decode("utf-8"))
    assert body["content_id"] == content["id"]
    assert body["claim_count"] == 1
    assert [c["claim_id"] for c in body["claims"]] == [claim["id"]]


# --- existing interfaces unchanged -------------------------------------------


def test_full_export_and_claim_reads_are_unchanged(client):
    create_actor(client)
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    bundle = _create_bundle(client, claim["id"])

    # The full export still carries the complete public views, including
    # the evidence metadata the privacy export withholds.
    full = client.get(f"/v1/contents/{content['id']}/export").json()
    assert set(full) == {"content", "claims"}
    assert full["content"] == content
    assert full["claims"][0]["actor_id"] == "org-1"
    assert full["claims"][0]["evidence_bundles"][0]["metadata"] == {
        "source": "camera-1"
    }

    # The claim and bundle reads still return their full public views.
    claim_read = client.get(f"/v1/claims/{claim['id']}").json()
    assert claim_read["actor_id"] == "org-1"
    bundle_read = client.get(f"/v1/evidence-bundles/{bundle['id']}").json()
    assert bundle_read["metadata"] == {"source": "camera-1"}
