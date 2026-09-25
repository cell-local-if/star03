"""Tests for the read-only cross-content revocation-impact retrieval.

Covers ``GET /v1/revocation-impacts``:

* the empty-body contract and the exact ``items``/``count``/``next_cursor``
  member order in compact UTF-8 JSON terminated by one newline, with
  non-ASCII emitted unescaped and integral numbers only (no negative zero);
* each item reusing the revocation public view and appending exactly the
  content association (``content_id``, ``target_type``), the signing
  subject (``signer_actor_id``), and the before/after qualified-signer
  counts, their delta, and the before/after coverage statuses -- no raw
  signature, signature digest, public key, claim payload, content, or
  evidence byte ever appears;
* the counting rules: the after counts use the content evidence-coverage
  summary rules (distinct subjects with a verified non-revoked proof of the
  content's direct claims or their bundles); the before counts ignore only
  that one revocation while every other revocation stays in effect; the
  delta is always 0 or 1; the three existing coverage literals reflect the
  change;
* the ``attestation_id``/``revoker_actor_id``/``reason``/``from``/``to``
  filters (non-empty exact matches combined as logical AND; strict
  inclusive RFC 3339 UTC bounds; an unknown or nonexistent value is an
  empty collection, never a 404), stable revocation creation ordering
  (``created_at`` with the persistent insertion-order tiebreaker that
  survives an app restart), a filtered ``count`` covering every page,
  pure-decimal ``limit`` validation (1..100, default 50), and the opaque
  HMAC-signed ``ri1`` cursor family (binding every effective condition and
  the limit, resuming without duplication or omission, a null cursor on
  the final page, and an empty page with the original count at or past the
  tail);
* every 422 validation boundary (body, blank/illegal/repeated/undeclared
  parameters, empty/malformed/tampered/cross-family/mismatching cursors),
  the 405 rejection of non-GET methods, and the strictly read-only
  guarantee.

All fixtures are deterministic and offline (in-memory and temporary-file
SQLite, signatures produced by the stdlib test signer).
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone

from sqlalchemy import func, select

from provenance import pagination
from provenance.models import AttestationRevocation, AuditEvent
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

IMPACTS_PATH = "/v1/revocation-impacts"
REVOCATIONS_PATH = "/v1/attestation-revocations"

EVIDENCE_DIGEST_A = hashlib.sha256(b"revocation-impact-evidence-a").hexdigest()

REASON_A = "key compromise during incident"
REASON_B = "signer requested withdrawal"
REASON_C = "rotated signing material"
REASON_D = "Révocation ⛄ doublon"


# --- Fixture-style setup ------------------------------------------------------


def _make_content(client, *, digest=DIGEST_A, actor_id="org-1", media_type="image/png"):
    resp = client.post(
        "/v1/contents",
        json=content_payload(
            actor_id=actor_id, digest=digest, media_type=media_type
        ),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make_claim(client, content_id, *, actor_id="org-1"):
    resp = client.post(
        "/v1/claims",
        json={
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": "authorship",
            "payload": {"statement": "attested"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make_bundle(client, claim_id):
    resp = client.post(
        "/v1/evidence-bundles",
        json={
            "claim_id": claim_id,
            "evidence_type": "raw_capture",
            "digest_algorithm": "sha256",
            "digest_hex": EVIDENCE_DIGEST_A,
            "media_type": "image/jpeg",
            "metadata": {"source": "camera-1"},
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _attest(client, target_type, target_id, *, seed=SEED_A, signer_actor_id="org-1"):
    signature = ed25519_sign(
        seed,
        attestation_message_bytes(target_type, target_id, signer_actor_id),
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


def _revoke(client, attestation_id, *, revoker_actor_id="org-1", reason=REASON_A):
    resp = client.post(
        REVOCATIONS_PATH,
        json={
            "attestation_id": attestation_id,
            "revoker_actor_id": revoker_actor_id,
            "reason": reason,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _world(client):
    """Three contents and five revocations, in stable creation order.

    Content C1 (claim L1 + bundle B1) has: two org-1 proofs (``att_bundle``
    on the bundle and ``att_claim`` on the claim) plus one org-2 proof
    (``att_other`` on the claim). Content C2 (claim L2) has one org-1 proof
    (``att_lonely``). Content C3 (claim L3) has one org-2 proof
    (``att_double``) carrying two revocation records. Revocations:

    1. ``r_redundant`` revokes org-1's bundle proof -- org-1 keeps its
       non-revoked claim proof, so delta 0 with coverage covered throughout;
    2. ``r_other`` revokes org-2's only C1 proof while org-1 remains, so
       delta 1 with covered before and covered after;
    3. ``r_lonely`` revokes C2's sole proof -- delta 1, covered -> partial;
    4. ``r_double_first`` is the first revocation of C3's sole proof;
    5. ``r_double_second`` is the second revocation of that same proof.

    For either C3 record the "before" count still treats the *other*
    revocation of that proof as in effect, so both show delta 0 and
    partial/partial even though one of them was created first.
    """
    create_actor(client, actor_id="org-1")
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")

    c1 = _make_content(client, digest=DIGEST_A, actor_id="org-1")
    l1 = _make_claim(client, c1["id"], actor_id="org-1")
    b1 = _make_bundle(client, l1["id"])
    att_bundle = _attest(
        client, "evidence_bundle", b1["id"], seed=SEED_A, signer_actor_id="org-1"
    )
    att_claim = _attest(
        client, "claim", l1["id"], seed=SEED_A, signer_actor_id="org-1"
    )
    att_other = _attest(
        client, "claim", l1["id"], seed=SEED_B, signer_actor_id="org-2"
    )

    c2 = _make_content(client, digest=DIGEST_B, actor_id="org-2")
    l2 = _make_claim(client, c2["id"], actor_id="org-2")
    att_lonely = _attest(
        client, "claim", l2["id"], seed=SEED_A, signer_actor_id="org-1"
    )

    c3 = _make_content(
        client, digest=DIGEST_C, actor_id="org-1", media_type="image/jpeg"
    )
    l3 = _make_claim(client, c3["id"], actor_id="org-1")
    att_double = _attest(
        client, "claim", l3["id"], seed=SEED_B, signer_actor_id="org-2"
    )

    r_redundant = _revoke(
        client, att_bundle["id"], revoker_actor_id="org-1", reason=REASON_B
    )
    r_other = _revoke(
        client, att_other["id"], revoker_actor_id="org-2", reason=REASON_A
    )
    r_lonely = _revoke(
        client, att_lonely["id"], revoker_actor_id="org-2", reason=REASON_D
    )
    r_double_first = _revoke(
        client, att_double["id"], revoker_actor_id="org-2", reason=REASON_A
    )
    r_double_second = _revoke(
        client, att_double["id"], revoker_actor_id="org-1", reason=REASON_C
    )
    return {
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "l1": l1,
        "b1": b1,
        "att_bundle": att_bundle,
        "att_claim": att_claim,
        "att_other": att_other,
        "att_lonely": att_lonely,
        "att_double": att_double,
        "revocations": [
            r_redundant,
            r_other,
            r_lonely,
            r_double_first,
            r_double_second,
        ],
    }


def _pin_times(db_session, revocations, instants):
    """Overwrite created_at per revocation id with a fixed UTC instant."""
    by_id = {r["id"]: instant for r, instant in zip(revocations, instants)}
    for row in db_session.execute(select(AttestationRevocation)).scalars():
        row.created_at = by_id[row.id]
    db_session.commit()


def _list(client, **params):
    return client.get(IMPACTS_PATH, params=params)


def _walk_pages(client, **params):
    """Follow next_cursor until exhausted; return (all_items, pages, count)."""
    pages = []
    all_items = []
    count = None
    cursor = None
    for _ in range(100):
        query = {**params}
        if cursor is not None:
            query["cursor"] = cursor
        resp = _list(client, **query)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        count = body["count"]
        pages.append(body["items"])
        all_items.extend(body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            break
    return all_items, pages, count


def _audit_count(session):
    return len(session.execute(select(AuditEvent)).scalars().all())


# --- Empty collection and response shape --------------------------------------


def test_empty_store_is_an_empty_collection(client):
    resp = _list(client)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_response_is_compact_json_with_ordered_members_and_one_newline(client):
    _world(client)
    resp = _list(client)
    assert resp.status_code == 200
    raw = resp.content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    assert raw.startswith(b'{"items":[')
    assert b'],"count":5,"next_cursor":null}' in raw
    expected = (
        json.dumps(resp.json(), separators=(",", ":"), ensure_ascii=False)
        .encode("utf-8")
        + b"\n"
    )
    assert raw == expected


def test_non_ascii_reason_is_emitted_unescaped(client):
    _world(client)
    raw = _list(client).content.decode("utf-8")
    assert REASON_D in raw
    assert "\\u" not in raw


def test_items_carry_exactly_the_public_view_plus_impact_fields(client):
    world = _world(client)
    body = _list(client).json()
    assert body["count"] == 5
    assert body["next_cursor"] is None
    assert [item["id"] for item in body["items"]] == [
        r["id"] for r in world["revocations"]
    ]
    expected_fields = [
        "id",
        "attestation_id",
        "revoker_actor_id",
        "reason",
        "created_at",
        "content_id",
        "target_type",
        "signer_actor_id",
        "before_qualified_signer_count",
        "after_qualified_signer_count",
        "qualified_signer_count_delta",
        "before_coverage_status",
        "after_coverage_status",
    ]
    for item in body["items"]:
        assert list(item) == expected_fields
        for key in (
            "before_qualified_signer_count",
            "after_qualified_signer_count",
            "qualified_signer_count_delta",
        ):
            assert isinstance(item[key], int)
            assert item[key] >= 0
        created_at = datetime.fromisoformat(item["created_at"])
        assert created_at.tzinfo is not None
        assert created_at.utcoffset().total_seconds() == 0
    # The revocation public-view prefix matches the plain revocation record.
    plain = {
        item["id"]: item
        for item in client.get(REVOCATIONS_PATH).json()["items"]
    }
    for item in body["items"]:
        for key in (
            "id",
            "attestation_id",
            "revoker_actor_id",
            "reason",
            "created_at",
        ):
            assert item[key] == plain[item["id"]][key]
    # No raw signature, key, payload, or byte material is echoed.
    rendered = client.get(IMPACTS_PATH).content.decode()
    for forbidden in (
        "signature",
        "public_key",
        "private_key",
        "payload",
        "content_bytes",
        "seq",
    ):
        assert forbidden not in rendered
    # Counts are plain non-negative integers: never a float, never -0.
    for item in client.get(IMPACTS_PATH).json()["items"]:
        for key in (
            "before_qualified_signer_count",
            "after_qualified_signer_count",
            "qualified_signer_count_delta",
        ):
            assert type(item[key]) is int
            assert item[key] >= 0


# --- Impact associations --------------------------------------------------------


def test_items_resolve_the_target_content_type_and_signing_subject(client):
    world = _world(client)
    items = {item["id"]: item for item in _list(client).json()["items"]}

    redundant = items[world["revocations"][0]["id"]]
    # An evidence-bundle proof resolves through its claim to the content.
    assert redundant["content_id"] == world["c1"]["id"]
    assert redundant["target_type"] == "evidence_bundle"
    assert redundant["signer_actor_id"] == "org-1"

    other = items[world["revocations"][1]["id"]]
    assert other["content_id"] == world["c1"]["id"]
    assert other["target_type"] == "claim"
    assert other["signer_actor_id"] == "org-2"

    lonely = items[world["revocations"][2]["id"]]
    assert lonely["content_id"] == world["c2"]["id"]
    assert lonely["target_type"] == "claim"
    assert lonely["signer_actor_id"] == "org-1"

    for index in (3, 4):
        doubled = items[world["revocations"][index]["id"]]
        assert doubled["content_id"] == world["c3"]["id"]
        assert doubled["target_type"] == "claim"
        assert doubled["signer_actor_id"] == "org-2"


# --- Before/after counts, delta, and coverage statuses ---------------------------


def test_before_after_counts_delta_and_statuses(client):
    world = _world(client)
    items = {item["id"]: item for item in _list(client).json()["items"]}

    # 1. The signer keeps a second non-revoked proof of the same content:
    #    dropping this revocation requalifies the same subject, not a new
    #    one, so the distinct-signer count does not move.
    redundant = items[world["revocations"][0]["id"]]
    assert (
        redundant["before_qualified_signer_count"],
        redundant["after_qualified_signer_count"],
        redundant["qualified_signer_count_delta"],
    ) == (1, 1, 0)
    assert (
        redundant["before_coverage_status"],
        redundant["after_coverage_status"],
    ) == ("covered", "covered")

    # 2. The signer's only proof is revoked while another subject remains.
    other = items[world["revocations"][1]["id"]]
    assert (
        other["before_qualified_signer_count"],
        other["after_qualified_signer_count"],
        other["qualified_signer_count_delta"],
    ) == (2, 1, 1)
    assert (
        other["before_coverage_status"],
        other["after_coverage_status"],
    ) == ("covered", "covered")

    # 3. The content's sole qualified signer drops out: covered -> partial.
    lonely = items[world["revocations"][2]["id"]]
    assert (
        lonely["before_qualified_signer_count"],
        lonely["after_qualified_signer_count"],
        lonely["qualified_signer_count_delta"],
    ) == (1, 0, 1)
    assert (
        lonely["before_coverage_status"],
        lonely["after_coverage_status"],
    ) == ("covered", "partial")

    # 4./5. A proof carrying two revocation records: ignoring either record
    # alone leaves the other one in effect, so the proof stays revoked and
    # both records show delta 0 with partial throughout.
    for index in (3, 4):
        doubled = items[world["revocations"][index]["id"]]
        assert (
            doubled["before_qualified_signer_count"],
            doubled["after_qualified_signer_count"],
            doubled["qualified_signer_count_delta"],
        ) == (0, 0, 0)
        assert (
            doubled["before_coverage_status"],
            doubled["after_coverage_status"],
        ) == ("partial", "partial")

    # Invariants for every item.
    for item in items.values():
        assert item["qualified_signer_count_delta"] in (0, 1)
        assert (
            item["before_qualified_signer_count"]
            == item["after_qualified_signer_count"]
            + item["qualified_signer_count_delta"]
        )
        assert item["before_coverage_status"] in (
            "uncovered",
            "partial",
            "covered",
        )
        assert item["after_coverage_status"] in (
            "uncovered",
            "partial",
            "covered",
        )


def test_after_counts_match_the_content_evidence_coverage_summary(client):
    world = _world(client)
    items = _list(client).json()["items"]
    summaries = {
        content_id: client.get(
            f"/v1/contents/{content_id}/evidence-coverage"
        ).json()
        for content_id in (
            world["c1"]["id"],
            world["c2"]["id"],
            world["c3"]["id"],
        )
    }
    for item in items:
        summary = summaries[item["content_id"]]
        assert item["after_qualified_signer_count"] == (
            summary["qualified_signer_count"]
        )
        assert item["after_coverage_status"] == summary["coverage_status"]


def test_revocation_of_one_content_never_counts_the_other_content(client):
    world = _world(client)
    items = {item["id"]: item for item in _list(client).json()["items"]}
    # org-1 signs on both C1 and C2; revoking C2's proof reports C2's own
    # counts only (C1's org-1 claim proof is irrelevant to C2).
    lonely = items[world["revocations"][2]["id"]]
    assert lonely["content_id"] == world["c2"]["id"]
    assert lonely["before_qualified_signer_count"] == 1
    assert lonely["after_qualified_signer_count"] == 0


def test_before_count_keeps_every_other_revocation_in_effect(client):
    """Counterfactual semantics: later-created revocations still apply.

    Two distinct signers each lose their only proof of one content. For the
    first revocation record the "before" count is not the historical state
    at its creation (two signers): the later revocation of the other
    signer's proof stays in effect, so only this record's signer returns.
    """
    create_actor(client, actor_id="org-1")
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    content = _make_content(client, digest=DIGEST_C)
    claim = _make_claim(client, content["id"])
    att_a = _attest(
        client, "claim", claim["id"], seed=SEED_A, signer_actor_id="org-1"
    )
    att_b = _attest(
        client, "claim", claim["id"], seed=SEED_B, signer_actor_id="org-2"
    )
    r_a = _revoke(client, att_a["id"], revoker_actor_id="org-1", reason=REASON_A)
    r_b = _revoke(client, att_b["id"], revoker_actor_id="org-2", reason=REASON_B)

    items = {item["id"]: item for item in _list(client).json()["items"]}
    # Ignoring only r_a still leaves r_b in effect: B does not return.
    assert (
        items[r_a["id"]]["before_qualified_signer_count"],
        items[r_a["id"]]["after_qualified_signer_count"],
        items[r_a["id"]]["qualified_signer_count_delta"],
    ) == (1, 0, 1)
    # Ignoring only r_b still leaves r_a in effect: A does not return.
    assert (
        items[r_b["id"]]["before_qualified_signer_count"],
        items[r_b["id"]]["after_qualified_signer_count"],
        items[r_b["id"]]["qualified_signer_count_delta"],
    ) == (1, 0, 1)
    for record in (r_a, r_b):
        assert items[record["id"]]["before_coverage_status"] == "covered"
        assert items[record["id"]]["after_coverage_status"] == "partial"


# --- Stable ordering ------------------------------------------------------------


def test_items_follow_stable_revocation_creation_order(client):
    world = _world(client)
    body = _list(client).json()
    assert [item["id"] for item in body["items"]] == [
        r["id"] for r in world["revocations"]
    ]
    created = [datetime.fromisoformat(item["created_at"]) for item in body["items"]]
    assert created == sorted(created)


def test_same_timestamp_ties_break_by_persistence_order(client, db_session):
    world = _world(client)
    tie = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    _pin_times(db_session, world["revocations"], [tie] * 5)
    body = _list(client).json()
    assert [item["id"] for item in body["items"]] == [
        r["id"] for r in world["revocations"]
    ]


def test_ordering_is_identical_across_an_app_restart(tmp_db_url, file_client):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    world = _world(file_client)
    first = _list(file_client)
    assert first.status_code == 200

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as restarted_client:
        second = _list(restarted_client)
        assert second.status_code == 200
        assert second.json() == first.json()
        assert [item["id"] for item in second.json()["items"]] == [
            r["id"] for r in world["revocations"]
        ]


# --- Exact-match filters --------------------------------------------------------


def test_attestation_id_filter_is_an_exact_match(client):
    world = _world(client)
    body = _list(client, attestation_id=world["att_double"]["id"]).json()
    assert body["count"] == 2
    assert [item["id"] for item in body["items"]] == [
        world["revocations"][3]["id"],
        world["revocations"][4]["id"],
    ]
    assert {item["attestation_id"] for item in body["items"]} == {
        world["att_double"]["id"]
    }
    body = _list(client, attestation_id=world["att_other"]["id"]).json()
    assert [item["id"] for item in body["items"]] == [
        world["revocations"][1]["id"]
    ]


def test_revoker_and_reason_filters_are_exact_matches(client):
    world = _world(client)
    body = _list(client, revoker_actor_id="org-2").json()
    assert [item["id"] for item in body["items"]] == [
        world["revocations"][1]["id"],
        world["revocations"][2]["id"],
        world["revocations"][3]["id"],
    ]
    body = _list(client, reason=REASON_C).json()
    assert [item["id"] for item in body["items"]] == [
        world["revocations"][4]["id"]
    ]


def test_filters_combine_as_logical_and(client):
    world = _world(client)
    body = _list(
        client,
        attestation_id=world["att_double"]["id"],
        revoker_actor_id="org-1",
    ).json()
    assert [item["id"] for item in body["items"]] == [
        world["revocations"][4]["id"]
    ]
    miss = _list(
        client,
        attestation_id=world["att_other"]["id"],
        revoker_actor_id="org-1",
    ).json()
    assert miss == {"items": [], "count": 0, "next_cursor": None}


def test_unknown_or_nonexistent_filter_value_is_an_empty_collection(client):
    _world(client)
    for params in (
        {"attestation_id": "att_ghost"},
        {"revoker_actor_id": "ghost"},
        {"reason": "no such reason was ever recorded"},
        {"revoker_actor_id": "Org-1"},
        {"revoker_actor_id": " org-1"},
        {
            "attestation_id": "att_ghost",
            "revoker_actor_id": "ghost",
            "reason": REASON_A,
        },
    ):
        resp = _list(client, **params)
        assert resp.status_code == 200, params
        assert resp.json() == {"items": [], "count": 0, "next_cursor": None}


def test_blank_filters_are_422(client):
    _world(client)
    for field in ("attestation_id", "revoker_actor_id", "reason"):
        for blank in ("", "   ", "\t"):
            resp = _list(client, **{field: blank})
            assert resp.status_code == 422, (field, repr(blank))
            assert resp.json()["error"]["code"] == "validation_error"


# --- Time bounds ----------------------------------------------------------------


def test_from_and_to_bounds_are_inclusive(client, db_session):
    world = _world(client)
    instants = [
        datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 2, 12, 30, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 3, 9, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 4, 0, 0, 0, tzinfo=timezone.utc),
    ]
    _pin_times(db_session, world["revocations"], instants)

    body = _list(client, **{"from": "2026-01-02T12:30:00Z"}).json()
    assert [item["id"] for item in body["items"]] == [
        r["id"] for r in world["revocations"][1:]
    ]
    assert body["count"] == 4

    body = _list(client, to="2026-01-02T12:30:00Z").json()
    assert [item["id"] for item in body["items"]] == [
        r["id"] for r in world["revocations"][:3]
    ]
    assert body["count"] == 3

    body = _list(
        client, **{"from": "2026-01-02T12:30:00Z", "to": "2026-01-02T12:30:00Z"}
    ).json()
    assert [item["id"] for item in body["items"]] == [
        world["revocations"][1]["id"],
        world["revocations"][2]["id"],
    ]


def test_equivalent_utc_notations_are_the_same_bound(client, db_session):
    world = _world(client)
    _pin_times(
        db_session,
        world["revocations"],
        [datetime(2026, 1, index + 1, 0, 0, 0, tzinfo=timezone.utc)
         for index in range(5)],
    )
    zed = _list(client, **{"from": "2026-01-03T00:00:00Z"}).json()
    offset = _list(client, **{"from": "2026-01-03T00:00:00+00:00"}).json()
    assert zed == offset
    assert [item["id"] for item in zed["items"]] == [
        world["revocations"][2]["id"],
        world["revocations"][3]["id"],
        world["revocations"][4]["id"],
    ]


def test_bad_time_bounds_and_inverted_range_are_422(client):
    _world(client)
    for bad in (
        "2026-01-02",
        "2026-01-02T12:30:00",
        "2026-01-02 12:30:00Z",
        "2026-01-02T12:30Z",
        "2026-01-02T12:30:00+01:00",
        "2026-13-02T12:30:00Z",
        "2026-01-02t12:30:00z",
        "soon",
        "",
        "   ",
    ):
        assert _list(client, **{"from": bad}).status_code == 422, repr(bad)
        assert _list(client, to=bad).status_code == 422, repr(bad)
    inverted = _list(
        client, **{"from": "2026-01-03T00:00:00Z", "to": "2026-01-02T00:00:00Z"}
    )
    assert inverted.status_code == 422
    assert inverted.json()["error"]["code"] == "validation_error"
    # Equal bounds are valid; a one-second inversion is still rejected.
    equal = _list(
        client, **{"from": "2026-01-02T00:00:00Z", "to": "2026-01-02T00:00:00Z"}
    )
    assert equal.status_code == 200


# --- limit validation -----------------------------------------------------------


def test_limit_boundaries_one_and_one_hundred_are_accepted(client):
    _world(client)
    assert _list(client, limit="1").status_code == 200
    assert _list(client, limit="100").status_code == 200


def test_limit_must_be_a_pure_decimal_integer_in_range(client):
    _world(client)
    for bad in ("0", "101", "-1", "5.0", "5e0", " 5", "5 ", "five", "+5", ""):
        resp = _list(client, limit=bad)
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_default_limit_is_fifty(client):
    create_actor(client, actor_id="org-1")
    content = _make_content(client, digest=DIGEST_C)
    claim = _make_claim(client, content["id"])
    att = _attest(client, "claim", claim["id"])
    # Distinct (revoker, reason) pairs keep each revocation as its own row.
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")
    for index in range(51):
        _revoke(
            client,
            att["id"],
            revoker_actor_id="org-1" if index % 2 == 0 else "org-2",
            reason=f"reason-{index:03d}",
        )
    body = _list(client).json()
    assert body["count"] == 51
    assert len(body["items"]) == 50
    assert body["next_cursor"] is not None


# --- Parameter strictness -------------------------------------------------------


def test_repeated_and_undeclared_parameters_are_422(client):
    _world(client)
    assert (
        client.get(
            IMPACTS_PATH,
            params=[("attestation_id", "att_a"), ("attestation_id", "att_b")],
        ).status_code
        == 422
    )
    for query in (
        "limit=1&limit=2",
        "revoker_actor_id=org-1&revoker_actor_id=org-2",
        "reason=a&reason=b",
        "from=2026-01-01T00:00:00Z&from=2026-01-02T00:00:00Z",
        "to=2026-01-01T00:00:00Z&to=2026-01-02T00:00:00Z",
        "cursor=a&cursor=b",
    ):
        resp = client.get(f"{IMPACTS_PATH}?{query}")
        assert resp.status_code == 422, query
        assert resp.json()["error"]["code"] == "validation_error"
    for params in (
        {"id": "rev_1"},
        {"revoker": "org-1"},
        {"offset": "1"},
        {"content_id": "cnt_x"},
        {"q": "x"},
    ):
        resp = _list(client, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_empty_body_is_422_rejected_before_any_read(client, db_session):
    _world(client)
    events_before = _audit_count(db_session)
    for body in (b"{}", b" ", b"{not valid json", b"[]", b"\n", b"\x00\xff"):
        resp = client.request(
            "GET",
            IMPACTS_PATH,
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"
    assert _audit_count(db_session) == events_before


# --- Pagination -----------------------------------------------------------------


def test_pages_resume_without_duplication_or_omission(client):
    world = _world(client)
    all_items, pages, count = _walk_pages(client, limit="2")
    assert count == 5
    assert [len(page) for page in pages] == [2, 2, 1]
    assert [item["id"] for item in all_items] == [
        r["id"] for r in world["revocations"]
    ]

    first = _list(client, limit="2").json()
    assert first["next_cursor"] is not None
    assert first["next_cursor"].startswith("ri1.")
    second = _list(client, limit="2", cursor=first["next_cursor"]).json()
    assert second["next_cursor"] is not None
    third = _list(client, limit="2", cursor=second["next_cursor"]).json()
    assert third["next_cursor"] is None


def test_count_covers_every_filtered_page(client):
    world = _world(client)
    all_items, _pages, count = _walk_pages(
        client, attestation_id=world["att_double"]["id"], limit="1"
    )
    assert count == 2
    assert len(all_items) == 2


def test_replayed_cursor_returns_the_same_page(client):
    _world(client)
    cursor = _list(client, limit="1").json()["next_cursor"]
    assert cursor is not None
    page = _list(client, limit="1", cursor=cursor)
    replay = _list(client, limit="1", cursor=cursor)
    assert page.status_code == 200
    assert replay.status_code == 200
    assert replay.content == page.content


def test_cursor_at_or_past_the_tail_returns_empty_page_with_count(client, app):
    _world(client)
    claims = {
        "attestation_id": None,
        "revoker_actor_id": None,
        "reason": None,
        "from": None,
        "to": None,
        "limit": 50,
    }
    tail = pagination.encode_typed_cursor(
        app.state.revocation_impacts_cursor_secret,
        pagination.REVOCATION_IMPACTS_CURSOR,
        {**claims, "offset": 5},
    )
    resp = _list(client, cursor=tail)
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 5, "next_cursor": None}

    past = pagination.encode_typed_cursor(
        app.state.revocation_impacts_cursor_secret,
        pagination.REVOCATION_IMPACTS_CURSOR,
        {**claims, "offset": 99},
    )
    resp = _list(client, cursor=past)
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 5, "next_cursor": None}


def test_cursor_binds_every_effective_condition_and_limit(client):
    world = _world(client)
    cursor = _list(client, limit="2").json()["next_cursor"]
    assert cursor is not None
    for params in (
        {"limit": "3", "cursor": cursor},
        {"cursor": cursor},  # default limit differs from the minted one
        {"attestation_id": world["att_other"]["id"], "limit": "2", "cursor": cursor},
        {"revoker_actor_id": "org-1", "limit": "2", "cursor": cursor},
        {"reason": REASON_A, "limit": "2", "cursor": cursor},
        {"from": "2000-01-01T00:00:00Z", "limit": "2", "cursor": cursor},
        {"to": "2030-01-01T00:00:00Z", "limit": "2", "cursor": cursor},
    ):
        resp = _list(client, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"

    # A cursor minted under one exact filter cannot resume another.
    double_cursor = _list(
        client, attestation_id=world["att_double"]["id"], limit="1"
    ).json()["next_cursor"]
    assert double_cursor is not None
    switched = _list(
        client,
        attestation_id=world["att_other"]["id"],
        limit="1",
        cursor=double_cursor,
    )
    assert switched.status_code == 422

    # Equivalent UTC notation resumes; a different second does not.
    time_cursor = _list(
        client, **{"from": "2000-01-01T00:00:00Z", "limit": "1"}
    ).json()["next_cursor"]
    assert time_cursor is not None
    same_bound = _list(
        client,
        **{
            "from": "2000-01-01T00:00:00+00:00",
            "limit": "1",
            "cursor": time_cursor,
        },
    )
    assert same_bound.status_code == 200
    changed_bound = _list(
        client,
        **{
            "from": "2000-01-01T00:00:01Z",
            "limit": "1",
            "cursor": time_cursor,
        },
    )
    assert changed_bound.status_code == 422


def test_blank_malformed_and_tampered_cursors_are_422(client):
    _world(client)
    valid = _list(client, limit="1").json()["next_cursor"]
    assert valid is not None
    tampered = valid[:-1] + ("A" if valid[-1] != "A" else "B")
    for bad in ("", "   ", "not-a-cursor", "ri1", "ri1.abc", tampered):
        resp = _list(client, cursor=bad)
        assert resp.status_code == 422, repr(bad)
        assert resp.json()["error"]["code"] == "validation_error", repr(bad)


def test_foreign_family_cursor_is_422(client, app):
    _world(client)
    # The sibling revocation-search family (ar1) and an unrelated family
    # never resume this retrieval.
    foreign_ar1 = pagination.encode_typed_cursor(
        app.state.attestation_revocations_cursor_secret,
        pagination.ATTESTATION_REVOCATIONS_CURSOR,
        {
            "attestation_id": None,
            "revoker_actor_id": None,
            "reason": None,
            "from": None,
            "to": None,
            "limit": 50,
            "offset": 1,
        },
    )
    resp = _list(client, cursor=foreign_ar1)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    foreign_cc1 = pagination.encode_typed_cursor(
        app.state.content_coverage_search_cursor_secret,
        pagination.CONTENT_COVERAGE_SEARCH_CURSOR,
        {
            "actor_id": None,
            "media_type": None,
            "coverage_status": None,
            "limit": 50,
            "offset": 1,
        },
    )
    resp = _list(client, cursor=foreign_cc1)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_secret_rotates_on_restart_invalidating_old_cursors(
    tmp_db_url, file_client
):
    from fastapi.testclient import TestClient

    from provenance.app import create_app
    from provenance.config import Settings

    _world(file_client)
    cursor = _list(file_client, limit="1").json()["next_cursor"]
    assert cursor is not None

    restarted = create_app(Settings(database_url=tmp_db_url))
    with TestClient(restarted) as restarted_client:
        resp = _list(restarted_client, limit="1", cursor=cursor)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


# --- Method boundary ------------------------------------------------------------


def test_non_get_methods_are_405_method_not_allowed(client):
    _world(client)
    for method in ("post", "put", "patch", "delete"):
        resp = getattr(client, method)(IMPACTS_PATH)
        assert resp.status_code == 405, (method, resp.text)
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Read-only guarantee ---------------------------------------------------------


def test_queries_and_failures_write_nothing(client, db_session):
    world = _world(client)
    events_before = _audit_count(db_session)
    revocations_before = db_session.scalar(
        select(func.count()).select_from(AttestationRevocation)
    )

    assert _list(client).status_code == 200
    assert _list(client, revoker_actor_id="ghost").status_code == 200
    assert _list(client, reason="nobody said this").status_code == 200
    assert (
        _list(client, limit="2", **{"from": "2026-01-01T00:00:00Z"}).status_code
        == 200
    )
    all_items, _pages, count = _walk_pages(client, limit="1")
    assert count == len(world["revocations"]) == 5
    assert len(all_items) == 5

    # Failures are read-only as well.
    assert _list(client, limit="0").status_code == 422
    assert _list(client, attestation_id=" ").status_code == 422
    assert _list(client, **{"from": "not-a-time"}).status_code == 422
    assert _list(client, cursor="bad").status_code == 422
    assert (
        client.request("GET", IMPACTS_PATH, content=b"{}").status_code == 422
    )

    assert (
        db_session.scalar(select(func.count()).select_from(AttestationRevocation))
        == revocations_before
    )
    assert _audit_count(db_session) == events_before
