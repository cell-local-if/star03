"""Tests for the atomic evidence-bundle batch import.

Covers POST /v1/evidence-bundle-imports: per-item validation identical to
the single-create route, first-wins in-batch dedup by
(claim, evidence_type, digest), pre-existing identities returning their
original public view, first-appearance response ordering, 201 when at least
one bundle is created versus 200 when all identities already exist, the
all-items/all-references validation that turns one missing claim into a
404 claim_not_found creating nothing, and the single-transaction commit of
every new bundle with its ``evidence_bundle.created`` audit event. Also the
raw-byte boundary, stable ids/UTC times, and compatibility with the single
endpoint. All fixtures are deterministic and offline.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

import pytest
from sqlalchemy import func, select

from provenance.models import (
    EVENT_EVIDENCE_BUNDLE_CREATED,
    AuditEvent,
    EvidenceBundle,
)
from tests.helpers import DIGEST_A, content_payload, create_actor

PATH = "/v1/evidence-bundle-imports"

METADATA_1 = {"source": "forensics-1", "captured_at": "2026-01-01T00:00:00Z"}
METADATA_2 = {"source": "forensics-2", "pages": 7}


def _evidence_digest(name: str) -> str:
    return hashlib.sha256(f"imp-evidence-{name}".encode()).hexdigest()


D1 = _evidence_digest("d1")
D2 = _evidence_digest("d2")
D3 = _evidence_digest("d3")


_UNSET = object()


def _item(
    claim_id,
    evidence_type="raw_capture",
    digest=D1,
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


def _import(client, items):
    return client.post(PATH, json={"items": items})


def _create_content(client, digest=DIGEST_A, actor_id="org-1"):
    resp = client.post(
        "/v1/contents",
        json=content_payload(actor_id=actor_id, digest=digest),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_claim(client, content_id, name="x", actor_id="org-1", claim_type="authorship"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": claim_type,
            "payload": {"statement": name},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _setup_claim(client):
    create_actor(client)
    content = _create_content(client)
    return _create_claim(client, content["id"])


def _setup_two_claims(client):
    create_actor(client)
    content = _create_content(client)
    claim_one = _create_claim(client, content["id"], name="one")
    # A second claim on the same content (different payload digest).
    claim_two = _create_claim(client, content["id"], name="two")
    return claim_one, claim_two


def _audit_count(db_session) -> int:
    return db_session.scalar(
        select(func.count()).select_from(AuditEvent)
    )


def _bundle_count(db_session) -> int:
    return db_session.scalar(
        select(func.count()).select_from(EvidenceBundle)
    )


PUBLIC_FIELDS = {
    "id",
    "claim_id",
    "evidence_type",
    "digest_algorithm",
    "digest_hex",
    "media_type",
    "metadata",
    "created_at",
}


# --- Response shape and normal creation --------------------------------------


def test_import_all_new_returns_201_with_items_and_count(client):
    claim = _setup_claim(client)
    resp = _import(
        client,
        [
            _item(claim["id"], digest=D1),
            _item(claim["id"], evidence_type="signature", digest=D2),
            _item(claim["id"], evidence_type="report", digest=D3,
                  media_type="application/pdf", metadata=METADATA_2),
        ],
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body) == {"items", "count"}
    assert body["count"] == 3
    assert len(body["items"]) == 3
    for item in body["items"]:
        assert set(item) == PUBLIC_FIELDS
        assert item["id"].startswith("evb_")
        assert "data" not in item and "evidence" not in item
        created_at = datetime.fromisoformat(item["created_at"])
        assert created_at.utcoffset().total_seconds() == 0
        assert item["created_at"].endswith(("Z", "+00:00"))
    assert [item["digest_hex"] for item in body["items"]] == [D1, D2, D3]


def test_single_item_batch_is_accepted(client):
    claim = _setup_claim(client)
    resp = _import(client, [_item(claim["id"])])
    assert resp.status_code == 201
    body = resp.json()
    assert body["count"] == 1
    assert body["items"][0]["digest_hex"] == D1


def test_exactly_one_hundred_items_accepted(client):
    claim = _setup_claim(client)
    digests = [_evidence_digest(f"bulk-{i:03d}") for i in range(100)]
    resp = _import(client, [_item(claim["id"], digest=d) for d in digests])
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["count"] == 100
    assert [item["digest_hex"] for item in body["items"]] == digests


def test_rich_metadata_round_trips(client):
    claim = _setup_claim(client)
    rich = {
        "nested": {"a": [1, 2, {"b": True}], "n": None},
        "unicode": "证据",
        "float": 1.5,
    }
    resp = _import(client, [_item(claim["id"], metadata=rich)])
    assert resp.status_code == 201
    assert resp.json()["items"][0]["metadata"] == rich


# --- In-batch dedup -----------------------------------------------------------


def test_duplicate_identity_in_batch_keeps_first_and_drops_later(client):
    claim = _setup_claim(client)
    resp = _import(
        client,
        [
            _item(claim["id"], digest=D1, media_type="image/jpeg",
                  metadata=METADATA_1),
            _item(claim["id"], digest=D2),
            # Same identity as the first item, but different media/metadata:
            # the first occurrence wins; this entry must not appear again.
            _item(claim["id"], digest=D1, media_type="application/pdf",
                  metadata=METADATA_2),
        ],
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["count"] == 2
    assert [item["digest_hex"] for item in body["items"]] == [D1, D2]
    first = body["items"][0]
    assert first["media_type"] == "image/jpeg"
    assert first["metadata"] == METADATA_1


def test_dedup_is_case_insensitive_on_normalized_hex(client):
    claim = _setup_claim(client)
    resp = _import(
        client,
        [
            _item(claim["id"], digest=D1),
            _item(claim["id"], digest=D1.upper()),
        ],
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["count"] == 1
    assert body["items"][0]["digest_hex"] == D1


def test_same_type_and_digest_on_different_claims_are_distinct(client):
    claim_one, claim_two = _setup_two_claims(client)
    resp = _import(
        client,
        [_item(claim_one["id"], digest=D1), _item(claim_two["id"], digest=D1)],
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["count"] == 2
    assert body["items"][0]["id"] != body["items"][1]["id"]
    assert {item["claim_id"] for item in body["items"]} == {
        claim_one["id"], claim_two["id"]
    }


def test_different_evidence_types_or_digests_are_distinct(client):
    claim = _setup_claim(client)
    resp = _import(
        client,
        [
            _item(claim["id"], evidence_type="raw_capture", digest=D1),
            _item(claim["id"], evidence_type="signature", digest=D1),
            _item(claim["id"], evidence_type="raw_capture", digest=D2),
        ],
    )
    assert resp.status_code == 201
    assert resp.json()["count"] == 3


# --- Pre-existing identities and mixed batches --------------------------------


def test_all_existing_identities_return_200_original_views(client, db_session):
    claim = _setup_claim(client)
    first = _import(
        client, [_item(claim["id"], digest=D1, metadata=METADATA_1)]
    ).json()["items"][0]
    events_before = _audit_count(db_session)

    resp = _import(
        client,
        [
            # Same identity, conflicting media/metadata: original retained.
            _item(claim["id"], digest=D1, media_type="application/pdf",
                  metadata=METADATA_2),
        ],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["items"] == [first]
    assert body["items"][0]["media_type"] == "image/jpeg"
    assert body["items"][0]["metadata"] == METADATA_1
    # An all-existing batch writes nothing.
    db_session.expire_all()
    assert _audit_count(db_session) == events_before


def test_mixed_batch_returns_201_and_preserves_originals(client, db_session):
    claim = _setup_claim(client)
    existing = _import(
        client, [_item(claim["id"], digest=D1, metadata=METADATA_1)]
    ).json()["items"][0]
    events_before = _audit_count(db_session)

    resp = _import(
        client,
        [
            _item(claim["id"], digest=D2, metadata=METADATA_2),
            # Existing identity with a conflicting view: original returned.
            _item(claim["id"], digest=D1, media_type="image/png",
                  metadata={"later": True}),
            _item(claim["id"], digest=D3),
        ],
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["count"] == 3
    assert [item["digest_hex"] for item in body["items"]] == [D2, D1, D3]
    middle = body["items"][1]
    assert middle["id"] == existing["id"] == body["items"][1]["id"]
    assert middle == existing
    assert middle["metadata"] == METADATA_1

    # Exactly one audit event per newly created bundle (two), none for the
    # pre-existing identity.
    db_session.expire_all()
    assert _audit_count(db_session) == events_before + 2
    new_ids = {body["items"][0]["id"], body["items"][2]["id"]}
    created_events = db_session.execute(
        select(AuditEvent).where(
            AuditEvent.event_type == EVENT_EVIDENCE_BUNDLE_CREATED,
            AuditEvent.resource_id.in_(new_ids),
        )
    ).scalars().all()
    assert {e.resource_id for e in created_events} == new_ids


# --- Ordering ------------------------------------------------------------------


def test_items_follow_first_appearance_order(client):
    claim_one, claim_two = _setup_two_claims(client)
    resp = _import(
        client,
        [
            _item(claim_two["id"], digest=D3),
            _item(claim_one["id"], digest=D1),
            _item(claim_two["id"], digest=D2),
            # Repeats out of order must not reshuffle or reappear.
            _item(claim_one["id"], digest=D1),
            _item(claim_two["id"], digest=D3),
        ],
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["count"] == 3
    ordered = [
        (item["claim_id"], item["digest_hex"]) for item in body["items"]
    ]
    assert ordered == [
        (claim_two["id"], D3),
        (claim_one["id"], D1),
        (claim_two["id"], D2),
    ]


def test_order_is_deterministic_across_retries(client):
    claim = _setup_claim(client)
    items = [_item(claim["id"], digest=d) for d in (D1, D2, D3)]
    first = _import(client, items).json()
    # Every identity now exists: 200 and byte-identical order and views.
    second = _import(client, items)
    assert second.status_code == 200
    assert second.json() == first


# --- Missing claim: 404 and whole-batch atomicity ------------------------------


def test_unknown_claim_is_404_with_claim_id(client):
    create_actor(client)
    resp = _import(client, [_item("clm_ghost")])
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "claim_not_found"
    assert error["details"]["claim_id"] == "clm_ghost"


def test_one_missing_claim_aborts_entire_batch(client, db_session):
    claim, _other = _setup_two_claims(client)
    bundles_before = _bundle_count(db_session)
    events_before = _audit_count(db_session)

    resp = _import(
        client,
        [
            # Valid item referencing an existing claim...
            _item(claim["id"], digest=D1),
            # ...and an item whose claim does not exist.
            _item("clm_ghost", digest=D2),
        ],
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "claim_not_found"

    # Neither bundle exists: the valid item never created a row or event.
    db_session.expire_all()
    assert _bundle_count(db_session) == bundles_before
    assert _audit_count(db_session) == events_before
    assert db_session.execute(
        select(EvidenceBundle).where(EvidenceBundle.digest_hex == D1)
    ).scalars().all() == []


def test_missing_claim_is_reported_in_first_reference_order(client):
    create_actor(client)
    resp = _import(
        client,
        [_item("clm_aaa"), _item("clm_bbb")],
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["details"]["claim_id"] == "clm_aaa"

    resp = _import(
        client,
        [_item("clm_bbb"), _item("clm_aaa")],
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["details"]["claim_id"] == "clm_bbb"


def test_missing_claim_behind_repeated_and_existing_items(client, db_session):
    claim, _ = _setup_two_claims(client)
    bundles_before = _bundle_count(db_session)
    resp = _import(
        client,
        [
            _item(claim["id"], digest=D1),
            _item("clm_ghost", digest=D2),
            # A repeat of the missing claim and another existing-claim item:
            # the batch must still fail wholesale on the missing reference.
            _item("clm_ghost", digest=D2),
            _item(claim["id"], digest=D3),
        ],
    )
    assert resp.status_code == 404
    db_session.expire_all()
    assert _bundle_count(db_session) == bundles_before
    assert db_session.execute(select(EvidenceBundle)).scalars().all() == []


# --- Validation boundary (identical rules to single create) --------------------


def test_malformed_json_body_is_422(client):
    _setup_claim(client)
    resp = client.post(
        PATH,
        content="{not valid json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_body_must_be_an_object_with_items(client):
    _setup_claim(client)
    for body in ([], ["x"], "x", 42, None):
        resp = client.post(PATH, json=body)
        assert resp.status_code == 422, body


def test_missing_items_field_is_422(client):
    resp = client.post(PATH, json={})
    assert resp.status_code == 422
    fields = {
        ".".join(part for part in issue["loc"] if part != "body")
        for issue in resp.json()["error"]["details"]["issues"]
    }
    assert "items" in fields


def test_unknown_top_level_field_is_422(client, db_session):
    claim = _setup_claim(client)
    events_before = _audit_count(db_session)
    resp = client.post(
        PATH, json={"items": [_item(claim["id"])], "data": "raw-bytes-marker"})
    assert resp.status_code == 422
    fields = {
        ".".join(part for part in issue["loc"] if part != "body")
        for issue in resp.json()["error"]["details"]["issues"]
    }
    assert "data" in fields
    assert "raw-bytes-marker" not in resp.text
    db_session.expire_all()
    assert _audit_count(db_session) == events_before


def test_items_must_be_array_and_bounded(client):
    claim = _setup_claim(client)
    for body in ({"items": {}}, {"items": "x"}, {"items": _item(claim["id"])}):
        resp = client.post(PATH, json=body)
        assert resp.status_code == 422, body

    resp = client.post(PATH, json={"items": []})
    assert resp.status_code == 422

    too_many = [_item(claim["id"], digest=_evidence_digest(f"x{i}")) for i in range(101)]
    resp = client.post(PATH, json={"items": too_many})
    assert resp.status_code == 422


def test_non_object_item_is_422(client):
    _setup_claim(client)
    for bad in ("x", 42, 3.14, True, None, ["a"]):
        resp = client.post(PATH, json={"items": [bad]})
        assert resp.status_code == 422, bad


def test_blank_item_fields_are_422_with_indexed_locations(client):
    claim = _setup_claim(client)
    for field, value in (
        ("claim_id", "  "),
        ("evidence_type", "\t"),
        ("media_type", "   "),
    ):
        item = _item(claim["id"])
        item[field] = value
        resp = client.post(PATH, json={"items": [item]})
        assert resp.status_code == 422, field
        locs = [
            tuple(part for part in issue["loc"] if part != "body")
            for issue in resp.json()["error"]["details"]["issues"]
        ]
        assert ("items", "0", field) in locs


def test_item_digest_rules_match_single_create(client):
    claim = _setup_claim(client)
    bad_items = [
        _item(claim["id"], algorithm="sha512"),
        _item(claim["id"], digest="a" * 63),
        _item(claim["id"], digest="a" * 65),
        _item(claim["id"], digest=D1[:-1] + "z"),
    ]
    for item in bad_items:
        resp = client.post(PATH, json={"items": [item]})
        assert resp.status_code == 422, item


def test_non_object_item_metadata_is_422(client):
    claim = _setup_claim(client)
    for bad in ([1, 2], "text", 42, 3.14, True, None):
        resp = client.post(PATH, json={"items": [_item(claim["id"], metadata=bad)]})
        assert resp.status_code == 422, bad


def test_raw_evidence_fields_on_item_are_422_and_not_echoed(client, db_session):
    claim = _setup_claim(client)
    events_before = _audit_count(db_session)
    marker = "batch-raw-byte-marker-77aa"
    item = _item(claim["id"], metadata={"k": "v"})
    item["data"] = marker
    item["evidence"] = marker.encode().hex()
    resp = client.post(PATH, json={"items": [item, _item(claim["id"], digest=D2)]})
    assert resp.status_code == 422
    fields = {
        ".".join(part for part in issue["loc"] if part != "body")
        for issue in resp.json()["error"]["details"]["issues"]
    }
    assert {"items.0.data", "items.0.evidence"}.issubset(fields)
    assert marker not in resp.text

    # The structural failure rejects the whole request: the valid second
    # item was never written either.
    db_session.expire_all()
    assert _bundle_count(db_session) == 0
    assert _audit_count(db_session) == events_before


def test_malformed_item_is_422_even_when_claim_also_missing(client):
    # Structural validation of every item precedes reference existence: a
    # blank field is a 422 regardless of whether the claim exists.
    resp = client.post(
        PATH, json={"items": [_item("clm_ghost", evidence_type=" ")]}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_error_body_has_no_items(client):
    resp = client.post(PATH, json={"items": []})
    assert resp.status_code == 422
    assert "items" not in resp.json()


# --- Atomicity: one transaction for bundles and audit events -------------------


def test_new_bundles_and_audit_events_commit_together(client, db_session):
    claim = _setup_claim(client)
    resp = _import(
        client,
        [_item(claim["id"], digest=D1), _item(claim["id"], digest=D2)],
    )
    assert resp.status_code == 201
    ids = [item["id"] for item in resp.json()["items"]]

    bundles = db_session.execute(
        select(EvidenceBundle).order_by(EvidenceBundle.seq.asc())
    ).scalars().all()
    events = db_session.execute(
        select(AuditEvent)
        .where(AuditEvent.event_type == EVENT_EVIDENCE_BUNDLE_CREATED)
        .order_by(AuditEvent.seq.asc())
    ).scalars().all()
    # One row and exactly one creation event per new bundle, visible after a
    # single request: they committed in the same transaction.
    assert [b.id for b in bundles] == ids
    assert [e.resource_id for e in events] == ids
    assert len(events) == 2
    stamped = events[0].created_at
    assert stamped.tzinfo.utcoffset(stamped).total_seconds() == 0


def test_batch_uses_a_single_commit(client, db_session, monkeypatch):
    claim = _setup_claim(client)
    commits = 0
    original_commit = type(db_session).commit

    def counting_commit(self):
        nonlocal commits
        commits += 1
        return original_commit(self)

    monkeypatch.setattr(
        "sqlalchemy.orm.Session.commit", counting_commit
    )
    resp = _import(
        client,
        [_item(claim["id"], digest=D1), _item(claim["id"], digest=D2)],
    )
    assert resp.status_code == 201
    # All new bundles plus their audit events go out in one COMMIT.
    assert commits == 1


def test_failed_commit_rolls_back_everything(client, db_session, monkeypatch):
    from provenance import service
    from provenance.schemas import EvidenceBundleCreate

    claim = _setup_claim(client)
    db_session.expire_all()
    bundles_before = _bundle_count(db_session)
    events_before = _audit_count(db_session)

    payloads = [
        EvidenceBundleCreate(**_item(claim["id"], digest=D1)),
        EvidenceBundleCreate(**_item(claim["id"], digest=D2)),
    ]

    def failing_commit(self):
        # Flush every staged row/event inside the open transaction, then
        # fail immediately before COMMIT: a real partial write that must be
        # rolled back wholesale.
        self.flush()
        raise RuntimeError("simulated storage failure before COMMIT")

    monkeypatch.setattr("sqlalchemy.orm.Session.commit", failing_commit)
    with pytest.raises(RuntimeError):
        service.import_evidence_bundles(db_session, payloads)

    # The open transaction is rolled back; nothing reached storage.
    db_session.rollback()
    db_session.expire_all()
    assert _bundle_count(db_session) == bundles_before
    assert _audit_count(db_session) == events_before


# --- Single-endpoint compatibility ---------------------------------------------


def test_batch_views_match_get_and_single_create(client):
    claim = _setup_claim(client)
    body = _import(client, [_item(claim["id"], digest=D1)]).json()
    bundle = body["items"][0]

    fetched = client.get(f"/v1/evidence-bundles/{bundle['id']}")
    assert fetched.status_code == 200
    assert fetched.json() == bundle

    # A repeat via the single-item endpoint is idempotent on the batch row.
    single = client.post("/v1/evidence-bundles", json=_item(claim["id"], digest=D1))
    assert single.status_code == 200
    assert single.json() == bundle

    # The batch row participates in the existing per-claim listing.
    listing = client.get(f"/v1/claims/{claim['id']}/evidence-bundles")
    assert listing.status_code == 200
    assert listing.json()["items"] == [bundle]


def test_single_created_bundle_is_returned_by_batch(client):
    claim = _setup_claim(client)
    single = client.post(
        "/v1/evidence-bundles",
        json=_item(claim["id"], digest=D1, metadata=METADATA_1),
    )
    assert single.status_code == 201

    resp = _import(
        client,
        [
            _item(claim["id"], digest=D1, metadata=METADATA_2,
                  media_type="application/pdf"),
        ],
    )
    assert resp.status_code == 200
    assert resp.json()["items"] == [single.json()]


def test_batch_id_is_the_stable_identity_id(client):
    claim = _setup_claim(client)
    from provenance import ids

    expected = ids.evidence_bundle_id(
        claim["id"], "raw_capture", "sha256", D1
    )
    resp = _import(client, [_item(claim["id"], digest=D1)])
    assert resp.json()["items"][0]["id"] == expected


# --- Read/no-write failure guarantees ------------------------------------------


def test_failed_batches_write_no_rows_or_audit_events(client, db_session):
    claim, _ = _setup_two_claims(client)
    bundles_before = _bundle_count(db_session)
    events_before = _audit_count(db_session)

    attempts = [
        client.post(PATH, json={"items": []}),
        client.post(PATH, json={"items": [_item(claim["id"], evidence_type=" ")]}),
        client.post(PATH, json={"items": [_item("clm_ghost")]}),
        client.post(PATH, json={"items": [_item(claim["id"], metadata=[])]}),
        client.post(PATH, json={"items": [_item(claim["id"], algorithm="sha1")]}),
        client.post(
            PATH,
            json={"items": [_item(claim["id"])], "extra": 1},
        ),
    ]
    assert [r.status_code for r in attempts] == [422, 422, 404, 422, 422, 422]

    db_session.expire_all()
    assert _bundle_count(db_session) == bundles_before
    assert _audit_count(db_session) == events_before
