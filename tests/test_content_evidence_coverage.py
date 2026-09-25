"""Tests for the read-only content evidence-coverage summary.

Covers ``GET /v1/contents/{content_id}/evidence-coverage``:

* a content without claims -> all counts zero and ``uncovered``;
* claims but no qualified signer -> ``partial`` (including when every
  attestation is revoked);
* one qualified signer -> ``covered``;
* claim/bundle/attestation counting rules: only claims directly asserting
  the content, only their bundles (deduplicated), attestations targeting
  those claims or bundles with revoked proofs retained in the count, and
  distinct verified non-revoked signing actors for the qualified count;
* no lineage traversal and no cross-content leakage;
* the success body is compact UTF-8 JSON terminated by exactly one newline
  with members in the exact documented order;
* any body (whitespace, malformed JSON, an object), any query parameter,
  and a blank content id are 422 ``validation_error``; an unknown content
  id is 404 ``content_not_found``;
* success and failure are strictly read-only (no resource, audit, or log
  writes) and repeated reads, including across an app restart, return the
  same counts and status.

All fixtures are deterministic and offline (in-memory and temporary-file
SQLite, signatures produced by the stdlib test signer).
"""

from __future__ import annotations

import base64
import hashlib
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from provenance.app import create_app
from provenance.config import Settings
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

EVIDENCE_DIGEST_A = hashlib.sha256(b"evidence-coverage-a").hexdigest()
EVIDENCE_DIGEST_B = hashlib.sha256(b"evidence-coverage-b").hexdigest()


def _url(content_id: str) -> str:
    return f"/v1/contents/{content_id}/evidence-coverage"


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
            "payload": {"statement": "covered"},
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


def _revoke(client, attestation_id, *, revoker_actor_id="org-1"):
    resp = client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": attestation_id,
            "revoker_actor_id": revoker_actor_id,
            "reason": "no longer relied upon",
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _audit_count(db_session) -> int:
    return len(db_session.execute(select(AuditEvent)).scalars().all())


# --- uncovered ---------------------------------------------------------------


def test_coverage_without_claims_is_all_zero_and_uncovered(client, db_session):
    create_actor(client)
    content = _create_content(client)
    events_before = _audit_count(db_session)

    resp = client.get(_url(content["id"]))
    assert resp.status_code == 200, resp.text
    # Compact JSON terminated by exactly one newline, exact member order.
    assert resp.content == (
        b'{"content_id":"' + content["id"].encode("ascii") + b'",'
        b'"claim_count":0,"bundle_count":0,"attestation_count":0,'
        b'"qualified_signer_count":0,"coverage_status":"uncovered"}\n'
    )
    body = resp.json()
    assert list(body) == [
        "content_id",
        "claim_count",
        "bundle_count",
        "attestation_count",
        "qualified_signer_count",
        "coverage_status",
    ]
    assert body == {
        "content_id": content["id"],
        "claim_count": 0,
        "bundle_count": 0,
        "attestation_count": 0,
        "qualified_signer_count": 0,
        "coverage_status": "uncovered",
    }
    # Strictly read-only: no audit event was added.
    assert _audit_count(db_session) == events_before


# --- partial -----------------------------------------------------------------


def test_coverage_with_claims_and_bundles_but_no_attestations_is_partial(
    client,
):
    create_actor(client)
    content = _create_content(client)
    claim_a = _create_claim(client, content["id"])
    claim_b = _create_claim(client, content["id"], claim_type="review")
    _create_bundle(client, claim_a["id"])
    _create_bundle(client, claim_a["id"], digest=EVIDENCE_DIGEST_B)
    _create_bundle(client, claim_b["id"], digest=DIGEST_B)

    body = client.get(_url(content["id"])).json()
    assert body["claim_count"] == 2
    assert body["bundle_count"] == 3
    assert body["attestation_count"] == 0
    assert body["qualified_signer_count"] == 0
    assert body["coverage_status"] == "partial"


def test_coverage_with_all_attestations_revoked_is_partial(client):
    create_actor(client)
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    att = _attest(client, "claim", claim["id"])
    _revoke(client, att["id"])

    body = client.get(_url(content["id"])).json()
    # The revoked proof is retained and still counted as an attestation, but
    # it no longer qualifies its signer.
    assert body["attestation_count"] == 1
    assert body["qualified_signer_count"] == 0
    assert body["coverage_status"] == "partial"


# --- covered -----------------------------------------------------------------


def test_coverage_with_one_qualified_signer_is_covered(client):
    create_actor(client)
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    bundle = _create_bundle(client, claim["id"])
    _attest(client, "claim", claim["id"])
    _attest(client, "evidence_bundle", bundle["id"], seed=SEED_B)

    body = client.get(_url(content["id"])).json()
    assert body["claim_count"] == 1
    assert body["bundle_count"] == 1
    assert body["attestation_count"] == 2
    # Both proofs are by the same actor (under different keys): one signer.
    assert body["qualified_signer_count"] == 1
    assert body["coverage_status"] == "covered"


def test_coverage_counts_distinct_signers_once(client):
    create_actor(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    _attest(client, "claim", claim["id"], seed=SEED_A)
    _attest(client, "claim", claim["id"], seed=SEED_B)
    _attest(client, "claim", claim["id"], seed=SEED_A, signer_actor_id="org-2")

    body = client.get(_url(content["id"])).json()
    assert body["attestation_count"] == 3
    assert body["qualified_signer_count"] == 2
    assert body["coverage_status"] == "covered"


def test_coverage_revoked_signer_drops_out_but_proof_is_retained(client):
    create_actor(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    revoked = _attest(client, "claim", claim["id"], seed=SEED_A)
    _attest(client, "claim", claim["id"], seed=SEED_A, signer_actor_id="org-2")
    _revoke(client, revoked["id"])

    body = client.get(_url(content["id"])).json()
    assert body["attestation_count"] == 2
    assert body["qualified_signer_count"] == 1
    assert body["coverage_status"] == "covered"


# --- scoping ------------------------------------------------------------------


def test_coverage_excludes_other_contents_and_never_traverses_lineage(client):
    create_actor(client)
    content = _create_content(client)
    other = _create_content(client, digest=DIGEST_B)
    # The other content has its own claims, bundles, and attestations.
    other_claim = _create_claim(client, other["id"])
    other_bundle = _create_bundle(client, other_claim["id"])
    _attest(client, "claim", other_claim["id"])
    _attest(client, "evidence_bundle", other_bundle["id"], seed=SEED_B)
    # A lineage relation does not pull the other content's evidence in.
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
    assert body["bundle_count"] == 0
    assert body["attestation_count"] == 0
    assert body["qualified_signer_count"] == 0
    assert body["coverage_status"] == "uncovered"

    # The other content's own summary is unaffected.
    other_body = client.get(_url(other["id"])).json()
    assert other_body["claim_count"] == 1
    assert other_body["bundle_count"] == 1
    assert other_body["attestation_count"] == 2
    assert other_body["qualified_signer_count"] == 1
    assert other_body["coverage_status"] == "covered"


def test_coverage_ignores_attestations_of_unrelated_targets(client):
    create_actor(client)
    content = _create_content(client, digest=DIGEST_C)
    claim = _create_claim(client, content["id"])
    unrelated = _create_content(client, digest=DIGEST_B)
    unrelated_claim = _create_claim(client, unrelated["id"])
    _attest(client, "claim", unrelated_claim["id"])

    body = client.get(_url(content["id"])).json()
    assert body["claim_count"] == 1
    assert body["attestation_count"] == 0
    assert body["qualified_signer_count"] == 0
    assert body["coverage_status"] == "partial"


# --- wire format ---------------------------------------------------------------


def test_coverage_body_is_compact_json_with_single_trailing_newline(client):
    create_actor(client)
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    _attest(client, "claim", claim["id"])

    resp = client.get(_url(content["id"]))
    assert resp.status_code == 200, resp.text
    raw = resp.content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    # Compact separators: no incidental whitespace inside the document.
    assert b" " not in raw
    assert b": " not in raw
    body = json.loads(raw.decode("utf-8"))
    assert list(body) == [
        "content_id",
        "claim_count",
        "bundle_count",
        "attestation_count",
        "qualified_signer_count",
        "coverage_status",
    ]
    for key in (
        "claim_count",
        "bundle_count",
        "attestation_count",
        "qualified_signer_count",
    ):
        assert isinstance(body[key], int)
    assert body["coverage_status"] == "covered"


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
def test_coverage_validation_errors_422_write_nothing(client, db_session, send):
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
    assert ok.json()["coverage_status"] == "uncovered"


def test_coverage_blank_content_id_is_422(client):
    create_actor(client)
    _create_content(client)

    resp = client.get("/v1/contents/%20/evidence-coverage")
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


def test_coverage_unknown_content_id_is_404(client):
    create_actor(client)
    _create_content(client)

    resp = client.get(_url("ctn_" + "0" * 64))
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "content_not_found"


def test_coverage_query_param_beats_unknown_content_404(client):
    # Parameters are validated before the content lookup.
    resp = client.get(_url("ctn_" + "0" * 64) + "?x=1")
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- determinism and restart stability -----------------------------------------


def test_coverage_repeated_reads_are_identical(client, db_session):
    create_actor(client)
    content = _create_content(client)
    claim = _create_claim(client, content["id"])
    bundle = _create_bundle(client, claim["id"])
    _attest(client, "evidence_bundle", bundle["id"])
    events_before = _audit_count(db_session)

    first = client.get(_url(content["id"]))
    second = client.get(_url(content["id"]))
    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    assert _audit_count(db_session) == events_before


def test_coverage_is_stable_across_restart(tmp_db_url):
    app1 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app1) as first:
        create_actor(first)
        content = _create_content(first)
        claim = _create_claim(first, content["id"])
        bundle = _create_bundle(first, claim["id"])
        att = _attest(first, "evidence_bundle", bundle["id"])
        _revoke(first, att["id"])
        _attest(first, "claim", claim["id"], seed=SEED_B)
        before_restart = first.get(_url(content["id"])).content

    # A brand-new process/app over the same database file.
    app2 = create_app(Settings(database_url=tmp_db_url))
    with TestClient(app2) as second:
        after_restart = second.get(_url(content["id"])).content

    assert after_restart == before_restart
    body = json.loads(after_restart.decode("utf-8"))
    assert body == {
        "content_id": content["id"],
        "claim_count": 1,
        "bundle_count": 1,
        "attestation_count": 2,
        "qualified_signer_count": 1,
        "coverage_status": "covered",
    }
