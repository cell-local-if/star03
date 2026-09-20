"""Tests for the atomic evidence-bundle batch import endpoint.

Covers POST /v1/evidence-bundle-imports: batch validation identical to the
single-create boundary, all-claims-must-exist atomicity (a missing claim
writes nothing), in-batch identity dedup (first occurrence wins), existing
identities returned with their original public view, first-occurrence
ordering, the 201/200 status boundary, single-transaction commit of all new
bundles and their audit events, and compatibility with the single-create
and read endpoints. All tests are deterministic and offline.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from sqlalchemy import select

from provenance.models import (
    EVENT_EVIDENCE_BUNDLE_CREATED,
    AuditEvent,
    EvidenceBundle,
)
from tests.helpers import (
    DIGEST_A,
    DIGEST_B,
    content_payload,
    create_actor,
)

EVIDENCE_DIGEST_1 = hashlib.sha256(b"import-evidence-a").hexdigest()
EVIDENCE_DIGEST_2 = hashlib.sha256(b"import-evidence-b").hexdigest()
EVIDENCE_DIGEST_3 = hashlib.sha256(b"import-evidence-c").hexdigest()
EVIDENCE_DIGEST_4 = hashlib.sha256(b"import-evidence-d").hexdigest()

METADATA_1 = {"source": "camera-1", "captured_at": "2026-01-01T00:00:00Z"}
METADATA_2 = {"source": "scanner-2", "pages": 3}
METADATA_3 = {"nested": {"a": [1, 2, {"b": True}], "n": None}, "unicode": "证据"}


def _create_content(client, digest=DIGEST_A, actor_id="org-1"):
    resp = client.post(
        "/v1/contents", json=content_payload(actor_id=actor_id, digest=digest)
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, actor_id="org-1"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": "authorship",
            "payload": {"statement": "x"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _setup_claim(client, digest=DIGEST_A):
    create_actor(client)
    content = _create_content(client, digest=digest)
    return _create_claim(client, content_id=content["id"])


_UNSET = object()


def _item(
    claim_id,
    evidence_type="raw_capture",
    digest=EVIDENCE_DIGEST_1,
    algorithm="sha256",
    media_type="image/jpeg",
    metadata=_UNSET,
):
    return {
        "claim_id": claim_id,
        "evidence_type": evidence_type,
        "digest_algorithm": algorithm,
        "digest_hex": digest,
        "media_type": media_type,
        "metadata": METADATA_1 if metadata is _UNSET else metadata,
    }


def _import(client, items, **extra):
    body = {"items": items}
    body.update(extra)
    return client.post("/v1/evidence-bundle-imports", json=body)


def _bundle_events(db_session):
    return (
        db_session.execute(
            select(AuditEvent)
            .where(AuditEvent.event_type == EVENT_EVIDENCE_BUNDLE_CREATED)
            .order_by(AuditEvent.seq.asc())
        )
        .scalars()
        .all()
    )


# --- Normal batch creation ---------------------------------------------------


def test_import_all_new_returns_201_in_first_seen_order(client):
    claim = _setup_claim(client)
    items = [
        _item(claim["id"], digest=EVIDENCE_DIGEST_2, metadata=METADATA_2),
        _item(claim["id"], digest=EVIDENCE_DIGEST_1, metadata=METADATA_1),
        _item(claim["id"], digest=EVIDENCE_DIGEST_3, metadata=METADATA_3),
    ]
    resp = _import(client, items)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == {"items", "count"}
    assert body["count"] == 3
    assert len(body["items"]) == 3
    # Response order is the request's first-occurrence order, not sorted.
    assert [i["digest_hex"] for i in body["items"]] == [
        EVIDENCE_DIGEST_2,
        EVIDENCE_DIGEST_1,
        EVIDENCE_DIGEST_3,
    ]
    assert [i["metadata"] for i in body["items"]] == [
        METADATA_2,
        METADATA_1,
        METADATA_3,
    ]


def test_import_response_items_have_full_public_shape_and_utc(client):
    claim = _setup_claim(client)
    resp = _import(client, [_item(claim["id"], metadata=METADATA_3)])
    assert resp.status_code == 201, resp.text
    (item,) = resp.json()["items"]
    assert set(item) == {
        "id",
        "claim_id",
        "evidence_type",
        "digest_algorithm",
        "digest_hex",
        "media_type",
        "metadata",
        "created_at",
    }
    assert item["id"].startswith("evb_")
    assert item["claim_id"] == claim["id"]
    assert item["digest_algorithm"] == "sha256"
    assert item["digest_hex"] == EVIDENCE_DIGEST_1
    assert item["metadata"] == METADATA_3
    created_at = datetime.fromisoformat(item["created_at"])
    assert created_at.utcoffset().total_seconds() == 0
    assert item["created_at"].endswith(("Z", "+00:00"))


def test_import_spans_multiple_claims(client):
    claim_one = _setup_claim(client, digest=DIGEST_A)
    content_two = _create_content(client, digest=DIGEST_B)
    claim_two = _create_claim(client, content_id=content_two["id"])

    resp = _import(
        client,
        [
            _item(claim_two["id"], digest=EVIDENCE_DIGEST_1),
            _item(claim_one["id"], digest=EVIDENCE_DIGEST_1),
        ],
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["count"] == 2
    # Same type and digest on different claims are independent identities.
    assert [i["claim_id"] for i in body["items"]] == [
        claim_two["id"],
        claim_one["id"],
    ]
    assert body["items"][0]["id"] != body["items"][1]["id"]


def test_import_ids_are_stable_across_requests(client):
    claim = _setup_claim(client)
    first = _import(client, [_item(claim["id"])]).json()
    repeat = _import(client, [_item(claim["id"])])
    assert repeat.status_code == 200
    assert repeat.json()["items"][0]["id"] == first["items"][0]["id"]


# --- In-batch dedup ------------------------------------------------------------


def test_duplicate_identity_within_batch_keeps_first_occurrence(
    client, db_session
):
    claim = _setup_claim(client)
    resp = _import(
        client,
        [
            _item(claim["id"], metadata=METADATA_1),
            _item(
                claim["id"],
                metadata=METADATA_2,
                media_type="application/pdf",
            ),
            _item(claim["id"], metadata=METADATA_3),
        ],
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    # Only the first occurrence survives; later duplicates never appear.
    assert body["count"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["metadata"] == METADATA_1
    assert body["items"][0]["media_type"] == "image/jpeg"

    # Exactly one bundle row and one audit event were written.
    bundles = db_session.execute(select(EvidenceBundle)).scalars().all()
    assert [b.id for b in bundles] == [body["items"][0]["id"]]
    events = _bundle_events(db_session)
    assert [e.resource_id for e in events] == [body["items"][0]["id"]]


def test_duplicate_with_uppercase_hex_dedups_within_batch(client):
    claim = _setup_claim(client)
    resp = _import(
        client,
        [
            _item(claim["id"], digest=EVIDENCE_DIGEST_1),
            _item(claim["id"], digest=EVIDENCE_DIGEST_1.upper()),
        ],
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["count"] == 1
    # The stored digest keeps the normalized lowercase form.
    assert body["items"][0]["digest_hex"] == EVIDENCE_DIGEST_1


def test_distinct_identities_in_batch_are_all_created(client):
    claim = _setup_claim(client)
    resp = _import(
        client,
        [
            _item(claim["id"]),
            _item(claim["id"], evidence_type="signature"),
            _item(claim["id"], digest=EVIDENCE_DIGEST_2),
        ],
    )
    assert resp.status_code == 201, resp.text
    ids = {item["id"] for item in resp.json()["items"]}
    assert len(ids) == 3


# --- Existing identities -------------------------------------------------------


def test_existing_identity_returns_original_public_view(client, db_session):
    claim = _setup_claim(client)
    created = client.post(
        "/v1/evidence-bundles", json=_item(claim["id"], metadata=METADATA_1)
    )
    assert created.status_code == 201
    original = created.json()
    events_after_create = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    # Same identity via the batch, with different media type and metadata:
    # the existing resource wins, unchanged, and nothing is written.
    resp = _import(
        client,
        [
            _item(
                claim["id"],
                metadata=METADATA_2,
                media_type="application/pdf",
            )
        ],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 1
    assert body["items"][0] == original
    assert body["items"][0]["metadata"] == METADATA_1
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_after_create
    )


def test_all_existing_identities_return_200_and_write_nothing(
    client, db_session
):
    claim = _setup_claim(client)
    first = _import(
        client,
        [
            _item(claim["id"], digest=EVIDENCE_DIGEST_1),
            _item(claim["id"], digest=EVIDENCE_DIGEST_2),
        ],
    )
    assert first.status_code == 201
    events_after_create = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    repeat = _import(
        client,
        [
            _item(claim["id"], digest=EVIDENCE_DIGEST_2, metadata=METADATA_2),
            _item(claim["id"], digest=EVIDENCE_DIGEST_1, metadata=METADATA_2),
        ],
    )
    assert repeat.status_code == 200, repeat.text
    body = repeat.json()
    assert body["count"] == 2
    # Original views are returned (first-submission metadata retained), in
    # this batch's first-occurrence order.
    assert [i["digest_hex"] for i in body["items"]] == [
        EVIDENCE_DIGEST_2,
        EVIDENCE_DIGEST_1,
    ]
    assert all(i["metadata"] == METADATA_1 for i in body["items"])
    assert body["items"] == [
        first.json()["items"][1],
        first.json()["items"][0],
    ]
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_after_create
    )


def test_mixed_new_and_existing_returns_201_and_creates_only_new(
    client, db_session
):
    claim = _setup_claim(client)
    existing = client.post(
        "/v1/evidence-bundles", json=_item(claim["id"], metadata=METADATA_1)
    ).json()

    resp = _import(
        client,
        [
            _item(claim["id"], digest=EVIDENCE_DIGEST_2, metadata=METADATA_2),
            _item(claim["id"], metadata=METADATA_3),  # existing identity
            _item(claim["id"], digest=EVIDENCE_DIGEST_3, metadata=METADATA_3),
        ],
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["count"] == 3
    assert [i["digest_hex"] for i in body["items"]] == [
        EVIDENCE_DIGEST_2,
        EVIDENCE_DIGEST_1,
        EVIDENCE_DIGEST_3,
    ]
    # The existing identity keeps its original public view.
    assert body["items"][1] == existing

    # Only the two genuinely new identities wrote rows and audit events.
    bundles = db_session.execute(select(EvidenceBundle)).scalars().all()
    assert len(bundles) == 3
    events = _bundle_events(db_session)
    assert len(events) == 3
    new_ids = {body["items"][0]["id"], body["items"][2]["id"]}
    assert {e.resource_id for e in events} == new_ids | {existing["id"]}


# --- Atomicity across the batch -------------------------------------------------


def test_missing_claim_fails_whole_batch_and_writes_nothing(client, db_session):
    claim = _setup_claim(client)
    events_before = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )

    resp = _import(
        client,
        [
            _item(claim["id"], digest=EVIDENCE_DIGEST_1),
            _item("clm_ghost", digest=EVIDENCE_DIGEST_2),
            _item(claim["id"], digest=EVIDENCE_DIGEST_3),
        ],
    )
    assert resp.status_code == 404, resp.text
    error = resp.json()["error"]
    assert error["code"] == "claim_not_found"
    assert error["details"]["claim_id"] == "clm_ghost"

    # Atomic: not even the valid items were created, and no audit events.
    assert db_session.execute(select(EvidenceBundle)).scalars().all() == []
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )


def test_all_claims_are_checked_before_any_write(client, db_session):
    claim = _setup_claim(client)
    events_before = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )
    # The missing claim sits behind valid new items in request order.
    resp = _import(
        client,
        [
            _item(claim["id"], digest=EVIDENCE_DIGEST_1),
            _item(claim["id"], digest=EVIDENCE_DIGEST_2),
            _item("clm_ghost", digest=EVIDENCE_DIGEST_3),
        ],
    )
    assert resp.status_code == 404
    assert db_session.execute(select(EvidenceBundle)).scalars().all() == []
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )


def test_validation_error_beats_missing_claim(client, db_session):
    # Malformed items are 422 before any claim-existence lookup, and nothing
    # is written even though the batch also references an unknown claim.
    events_before = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )
    resp = _import(
        client,
        [
            _item("clm_ghost", digest=EVIDENCE_DIGEST_1),
            _item("clm_ghost", digest="not-hex"),
        ],
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )


# --- Validation boundary (same rules as single create) --------------------------


def test_reject_empty_and_oversized_batches(client):
    claim = _setup_claim(client)
    empty = _import(client, [])
    assert empty.status_code == 422
    assert empty.json()["error"]["code"] == "validation_error"

    ok_items = [
        _item(claim["id"], digest=hashlib.sha256(f"ev-{i}".encode()).hexdigest())
        for i in range(100)
    ]
    ok = _import(client, ok_items)
    assert ok.status_code == 201, ok.text
    assert ok.json()["count"] == 100

    too_many = _import(client, ok_items + [_item(claim["id"])])
    assert too_many.status_code == 422
    assert too_many.json()["error"]["code"] == "validation_error"


def test_reject_missing_items_and_undeclared_top_level_fields(client):
    claim = _setup_claim(client)
    missing = client.post("/v1/evidence-bundle-imports", json={})
    assert missing.status_code == 422
    assert missing.json()["error"]["code"] == "validation_error"

    extra = _import(client, [_item(claim["id"])], batch_id="b-1")
    assert extra.status_code == 422
    assert extra.json()["error"]["code"] == "validation_error"
    issue_fields = {
        ".".join(part for part in issue["loc"] if part != "body")
        for issue in extra.json()["error"]["details"]["issues"]
    }
    assert "batch_id" in issue_fields


def test_reject_non_array_items(client):
    _setup_claim(client)
    for bad in ({}, "text", 42, None):
        resp = client.post("/v1/evidence-bundle-imports", json={"items": bad})
        assert resp.status_code == 422, bad
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_invalid_items_within_batch(client):
    claim = _setup_claim(client)
    valid = _item(claim["id"], digest=EVIDENCE_DIGEST_2)
    bad_items = [
        _item(claim["id"], evidence_type="   "),
        _item("  ", digest=EVIDENCE_DIGEST_1),
        _item(claim["id"], media_type=" "),
        _item(claim["id"], algorithm="sha512"),
        _item(claim["id"], digest="a" * 63),
        _item(claim["id"], digest="a" * 65),
        _item(claim["id"], digest=EVIDENCE_DIGEST_1[:-1] + "z"),
        _item(claim["id"], metadata=[1, 2]),
        _item(claim["id"], metadata="text"),
        _item(claim["id"], metadata=None),
    ]
    for bad in bad_items:
        # The invalid item sits behind a valid one: the whole batch is 422.
        resp = _import(client, [valid, bad])
        assert resp.status_code == 422, bad
        assert resp.json()["error"]["code"] == "validation_error"


def test_reject_item_with_missing_required_fields(client):
    _setup_claim(client)
    resp = _import(client, [{}])
    assert resp.status_code == 422
    issue_fields = {
        ".".join(part for part in issue["loc"] if part != "body")
        for issue in resp.json()["error"]["details"]["issues"]
    }
    for field in (
        "claim_id",
        "evidence_type",
        "digest_algorithm",
        "digest_hex",
        "media_type",
        "metadata",
    ):
        assert any(loc.endswith(field) for loc in issue_fields), field


def test_reject_raw_evidence_fields_within_batch_item(client, db_session):
    claim = _setup_claim(client)
    events_before = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )
    secret_marker = "raw-evidence-byte-marker-7f1a"
    bad = _item(claim["id"])
    bad["data"] = secret_marker
    bad["evidence"] = secret_marker.encode().hex()
    resp = _import(client, [bad])
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"
    assert secret_marker not in resp.text
    assert db_session.execute(select(EvidenceBundle)).scalars().all() == []
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )


def test_reject_malformed_json_body(client):
    _setup_claim(client)
    resp = client.post(
        "/v1/evidence-bundle-imports",
        content='{"items": [',
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_failed_imports_write_no_rows_or_audit_events(client, db_session):
    claim = _setup_claim(client)
    events_before = len(
        db_session.execute(select(AuditEvent)).scalars().all()
    )
    attempts = [
        _import(client, []),
        _import(client, [_item(claim["id"], metadata=[])]),
        _import(client, [_item(claim["id"], evidence_type=" ")]),
        _import(client, [_item("clm_ghost")]),
        _import(client, [_item(claim["id"]), _item("clm_ghost")]),
    ]
    assert [r.status_code for r in attempts] == [422, 422, 422, 404, 404]
    assert db_session.execute(select(EvidenceBundle)).scalars().all() == []
    assert (
        len(db_session.execute(select(AuditEvent)).scalars().all())
        == events_before
    )


# --- Transactional commit -------------------------------------------------------


def test_new_bundles_and_audit_events_commit_atomically(client, db_session):
    claim = _setup_claim(client)
    resp = _import(
        client,
        [
            _item(claim["id"], digest=EVIDENCE_DIGEST_1),
            _item(claim["id"], digest=EVIDENCE_DIGEST_2),
            _item(claim["id"], digest=EVIDENCE_DIGEST_3),
        ],
    )
    assert resp.status_code == 201, resp.text
    ids = [item["id"] for item in resp.json()["items"]]

    # All three bundle rows and all three audit events are visible after the
    # single request: they committed in the same transaction.
    bundles = db_session.execute(select(EvidenceBundle)).scalars().all()
    assert {b.id for b in bundles} == set(ids)
    events = _bundle_events(db_session)
    assert [e.resource_id for e in events] == ids
    for event in events:
        assert event.created_at.tzinfo.utcoffset(
            event.created_at
        ).total_seconds() == 0


# --- Compatibility with the single-create and read endpoints --------------------


def test_batch_created_bundles_are_visible_to_single_endpoints(client):
    claim = _setup_claim(client)
    imported = _import(
        client,
        [
            _item(claim["id"], digest=EVIDENCE_DIGEST_1, metadata=METADATA_1),
            _item(claim["id"], digest=EVIDENCE_DIGEST_2, metadata=METADATA_2),
        ],
    ).json()

    for item in imported["items"]:
        fetched = client.get(f"/v1/evidence-bundles/{item['id']}")
        assert fetched.status_code == 200
        assert fetched.json() == item

    listing = client.get(f"/v1/claims/{claim['id']}/evidence-bundles").json()
    assert listing["count"] == 2
    assert [i["id"] for i in listing["items"]] == [
        item["id"] for item in imported["items"]
    ]


def test_single_create_after_batch_is_idempotent(client):
    claim = _setup_claim(client)
    imported = _import(client, [_item(claim["id"])]).json()["items"][0]

    # The same identity through the single-create endpoint dedups against
    # the batch-created bundle and returns its original view.
    repeat = client.post("/v1/evidence-bundles", json=_item(claim["id"]))
    assert repeat.status_code == 200
    assert repeat.json() == imported


def test_batch_after_single_create_is_idempotent(client):
    claim = _setup_claim(client)
    created = client.post(
        "/v1/evidence-bundles", json=_item(claim["id"], metadata=METADATA_1)
    ).json()

    resp = _import(
        client,
        [
            _item(claim["id"], metadata=METADATA_2),
            _item(claim["id"], digest=EVIDENCE_DIGEST_4, metadata=METADATA_2),
        ],
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["count"] == 2
    assert body["items"][0] == created
    assert body["items"][1]["digest_hex"] == EVIDENCE_DIGEST_4
