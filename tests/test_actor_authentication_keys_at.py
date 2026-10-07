"""Offline deterministic tests for the historical authentication-key query.

Covers ``GET /v1/actors/{actor_id}/authentication-keys/at``:

* the success body carries exactly ``{"actor_id", "at", "items", "count",
  "next_cursor"}``; each item is exactly ``{"public_key", "sources"}`` and
  each source exactly ``{"source_type", "source_id", "created_at"}`` with
  ``source_type`` in ``{"attestation", "rotation"}`` -- never a private key,
  a raw signature, an authentication header, a payload, or an internal
  ordering surrogate;
* ``at`` is a closed judgment point: a rotation created no later than ``at``
  and unretired (or retired strictly later) is valid, and an attestation
  created no later than ``at`` with no revocation created no later than
  ``at`` is valid -- sources created exactly at ``at`` are included and
  sources retired/revoked exactly at ``at`` are excluded, while a key still
  carried by another valid source stays;
* keys are deduplicated by public-key bytes with every valid source merged,
  items follow the earliest valid source and sources stable creation order,
  and a point earlier than every source returns zero items;
* ``limit`` is a pure decimal integer in 1..100 (default 50); pages
  concatenate without gaps or duplicates, ``count`` is the deduplicated
  point-in-time total on every page (never the page count), the final cursor
  is null, and a cursor at/ past the tail returns an empty page with count;
* the opaque HMAC cursor is its own family bound to endpoint, position,
  subject, the parsed ``at``, and the effective limit: tampered, reused on a
  changed condition, cross-subject, wrong-family, and wrongly-signed tokens
  are all ``422 validation_error``;
* missing/blank/malformed ``at``, bad or repeated parameters, undeclared
  parameters, malformed cursors, and a non-empty body are all 422 decided
  before the subject lookup; an unknown subject is ``unknown_actor`` 404 and
  a non-GET method is ``405 method_not_allowed``;
* the route is strictly read-only (no resource, task, or audit write) and
  the existing current-key, rotation, revocation, signing, and protected
  behavior is unchanged.

Rows are inserted directly with fixed timestamps so every closed-interval
boundary is deterministic; only fixed seed-derived public keys are used.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from provenance import ids, pagination
from provenance.models import (
    Attestation,
    AttestationRevocation,
    AuditEvent,
    AuthenticationKeyRotation,
)
from provenance.time_utils import parse_rfc3339_utc
from tests.helpers import SEED_A, SEED_B, create_actor, ed25519_public_key
from tests.test_authentication_key_rotations import (
    SEED_R1,
    SEED_R2,
    _key_b64,
    _post_retire,
    _post_rotation,
    _rotation_body,
    _world,
)

URL = "/v1/actors/org-1/authentication-keys/at"

PAGE_KEYS = {"actor_id", "at", "items", "count", "next_cursor"}
ITEM_KEYS = {"public_key", "sources"}
SOURCE_KEYS = {"source_type", "source_id", "created_at"}

# A fifth distinct key, only ever inserted directly below.
SEED_K4 = b"test-ed25519-hist-k4-0000000000000"[:32]
SEED_EXTRA = b"test-ed25519-hist-extra-%06d"

# Fixed timeline (2026-01-01, UTC). Closed-interval boundaries land exactly
# on these instants.
T_A4 = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
T_R4 = datetime(2026, 1, 1, 8, 30, 0, tzinfo=timezone.utc)
T_V4 = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
T_A0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
T_R1 = datetime(2026, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
T_R1_RET = datetime(2026, 1, 1, 11, 30, 0, tzinfo=timezone.utc)
T_A2 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
T_R0 = datetime(2026, 1, 1, 12, 30, 0, tzinfo=timezone.utc)
T_R0_RET = datetime(2026, 1, 1, 12, 45, 0, tzinfo=timezone.utc)
T_V2 = datetime(2026, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
T_R3 = datetime(2026, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
# Genuinely after every row, including rows stamped at the wall clock by the
# API-backed worlds (the suite itself runs later than 2026).
FUTURE = "2099-01-01T00:00:00Z"


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _extra_seed(i: int) -> bytes:
    return (SEED_EXTRA % i).ljust(32, b"z")[:32]


# --- Deterministic world -----------------------------------------------------------


def _setup(client, db_session, *, extra_rotations=0):
    """Create org-1/org-2 and insert the fixed-timeline sources directly.

    K4: attestation 08:00, rotation 08:30, attestation revoked 09:00.
    K0: attestation 10:00 (never revoked), rotation 12:30 retired 12:45.
    K1: rotation 11:00 retired 11:30.
    K2: attestation 12:00 revoked 13:00.
    K3: rotation 14:00, never retired.
    Plus ``extra_rotations`` distinct active rotations after 14:00.
    """
    create_actor(client)
    create_actor(client, actor_id="org-2", name="Other Org", type="organization")

    keys = {
        "K0": ed25519_public_key(SEED_A),
        "K1": ed25519_public_key(SEED_R1),
        "K2": ed25519_public_key(SEED_R2),
        "K3": ed25519_public_key(SEED_B),
        "K4": ed25519_public_key(SEED_K4),
    }
    made = {}

    def add_att(name, key, when, target):
        sig_digest = hashlib.sha256(b"att" + key + target.encode()).hexdigest()
        aid = ids.attestation_id(
            "claim", target, "org-1", key.hex(), sig_digest
        )
        db_session.add(
            Attestation(
                id=aid,
                target_type="claim",
                target_id=target,
                signer_actor_id="org-1",
                public_key=key,
                signature_digest_algorithm="sha256",
                signature_digest_hex=sig_digest,
                created_at=when,
            )
        )
        made[name] = aid
        return aid

    def add_rev(name, aid, when):
        rid = ids.attestation_revocation_id(aid, "org-1", f"reason-{name}")
        db_session.add(
            AttestationRevocation(
                id=rid,
                attestation_id=aid,
                revoker_actor_id="org-1",
                reason=f"reason-{name}",
                created_at=when,
            )
        )
        made[name] = rid
        return rid

    def add_rot(name, key, when, retired):
        rid = ids.authentication_key_rotation_id("org-1", key.hex())
        db_session.add(
            AuthenticationKeyRotation(
                id=rid,
                actor_id="org-1",
                public_key=key,
                active=retired is None,
                created_at=when,
                retired_at=retired,
            )
        )
        made[name] = rid
        return rid

    add_att("A4", keys["K4"], T_A4, "claim-k4")
    add_rot("R4", keys["K4"], T_R4, None)
    add_rev("V4", made["A4"], T_V4)

    add_att("A0", keys["K0"], T_A0, "claim-k0")

    add_rot("R1", keys["K1"], T_R1, T_R1_RET)

    add_att("A2", keys["K2"], T_A2, "claim-k2")
    add_rev("V2", made["A2"], T_V2)

    add_rot("R0", keys["K0"], T_R0, T_R0_RET)
    add_rot("R3", keys["K3"], T_R3, None)

    for i in range(extra_rotations):
        add_rot(
            f"X{i}",
            ed25519_public_key(_extra_seed(i)),
            T_R3 + timedelta(minutes=i + 1),
            None,
        )

    db_session.commit()
    return keys, made


def _list(client, actor_id="org-1", at=FUTURE, **params):
    query = {"at": at, **params}
    return client.get(
        f"/v1/actors/{actor_id}/authentication-keys/at", params=query
    )


def _walk_pages(client, actor_id="org-1", at=FUTURE, **params):
    """Follow next_cursor until exhausted; return (all_items, pages, count)."""
    pages = []
    all_items = []
    count = None
    cursor = None
    for _ in range(100):
        query = {**params}
        if cursor is not None:
            query["cursor"] = cursor
        resp = _list(client, actor_id, at, **query)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        count = body["count"]
        pages.append(body["items"])
        all_items.extend(body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            break
    return all_items, pages, count


def _hand_cursor(secret: bytes, version: str, claims) -> str:
    """Sign a hand-built token so claim-set/format failures are exercised."""
    raw = (
        claims
        if isinstance(claims, str)
        else json.dumps(claims, separators=(",", ":"))
    )
    payload = pagination._b64encode(raw.encode("utf-8"))
    signature = pagination._b64encode(pagination._sign(secret, version, payload))
    return f"{version}.{payload}.{signature}"


def _public_keys(body) -> list[str]:
    return [item["public_key"] for item in body["items"]]


# --- Response shape, fields, and key spelling --------------------------------------


def test_success_body_has_exactly_the_five_page_members(client, db_session):
    keys, _ = _setup(client, db_session)
    body = _list(client).json()
    assert set(body) == PAGE_KEYS
    assert body["actor_id"] == "org-1"
    # The judgment point is echoed as strict RFC 3339 UTC.
    assert body["at"] == FUTURE
    assert parse_rfc3339_utc(body["at"]) is not None


def test_item_and_source_have_exactly_the_public_fields(client, db_session):
    _setup(client, db_session)
    body = _list(client, at=_iso(T_R1)).json()
    assert body["count"] == 3
    for item in body["items"]:
        assert set(item) == ITEM_KEYS
        for source in item["sources"]:
            assert set(source) == SOURCE_KEYS
            assert source["source_type"] in ("attestation", "rotation")
    # Internal/secret material never appears, even under different spellings.
    serialized = json.dumps(body)
    for leak in ("seq", "private", "signature", "payload", "active",
                 "retired", "X-PA", "X-PT", "X-PS"):
        assert leak not in serialized


def test_public_key_is_the_standard_base64_32_byte_spelling(client, db_session):
    keys, _ = _setup(client, db_session)
    spellings = set(_public_keys(_list(client).json()))
    assert spellings == {
        _key_b64(SEED_A),
        _key_b64(SEED_B),
        _key_b64(SEED_K4),
    }
    for spelling in spellings:
        raw = base64.b64decode(spelling, validate=True)
        assert len(raw) == 32
        assert base64.b64encode(raw).decode("ascii") == spelling


def test_source_created_at_is_utc(client, db_session):
    _setup(client, db_session)
    item = _list(client, at=_iso(T_A0)).json()["items"][0]
    created_at = datetime.fromisoformat(item["sources"][0]["created_at"])
    assert created_at.utcoffset().total_seconds() == 0


def test_at_is_echoed_canonically_for_equivalent_zone_spellings(client, db_session):
    _setup(client, db_session)
    z = _list(client, at="2026-01-01T10:00:00Z").json()
    offset = _list(client, at="2026-01-01T10:00:00+00:00").json()
    assert z["at"] == offset["at"] == "2026-01-01T10:00:00Z"
    assert z["items"] == offset["items"]


def test_reading_requires_no_authentication_headers(client, db_session):
    _setup(client, db_session)
    resp = _list(client)
    assert resp.status_code == 200
    assert "X-PA" not in resp.request.headers


# --- Closed-interval judgment semantics --------------------------------------------


def test_at_before_any_source_returns_zero_items(client, db_session):
    _setup(client, db_session)
    body = _list(client, at="2026-01-01T07:59:59Z").json()
    assert body == {
        "actor_id": "org-1",
        "at": "2026-01-01T07:59:59Z",
        "items": [],
        "count": 0,
        "next_cursor": None,
    }


def test_attestation_created_exactly_at_is_included(client, db_session):
    keys, made = _setup(client, db_session)
    body = _list(client, at=_iso(T_A4)).json()
    assert body["count"] == 1
    (item,) = body["items"]
    assert item["public_key"] == _key_b64(SEED_K4)
    assert item["sources"] == [
        {
            "source_type": "attestation",
            "source_id": made["A4"],
            "created_at": _iso(T_A4),
        }
    ]


def test_rotation_created_exactly_at_is_included(client, db_session):
    keys, made = _setup(client, db_session)
    body = _list(client, at=_iso(T_R1)).json()
    assert _public_keys(body) == [
        _key_b64(SEED_K4),
        _key_b64(SEED_A),
        _key_b64(SEED_R1),
    ]
    k1 = next(i for i in body["items"] if i["public_key"] == _key_b64(SEED_R1))
    assert k1["sources"] == [
        {
            "source_type": "rotation",
            "source_id": made["R1"],
            "created_at": _iso(T_R1),
        }
    ]


def test_rotation_retired_exactly_at_is_excluded(client, db_session):
    _setup(client, db_session)
    # One second before retirement K1 is still present; exactly at the
    # retirement instant it is gone.
    before = _list(client, at=_iso(T_R1_RET - timedelta(seconds=1))).json()
    assert _key_b64(SEED_R1) in _public_keys(before)
    at = _list(client, at=_iso(T_R1_RET)).json()
    assert _key_b64(SEED_R1) not in _public_keys(at)
    assert _public_keys(at) == [_key_b64(SEED_K4), _key_b64(SEED_A)]


def test_attestation_revoked_exactly_at_is_excluded(client, db_session):
    _setup(client, db_session)
    before = _list(client, at=_iso(T_V2 - timedelta(seconds=1))).json()
    assert _key_b64(SEED_R2) in _public_keys(before)
    at = _list(client, at=_iso(T_V2)).json()
    assert _key_b64(SEED_R2) not in _public_keys(at)


def test_revoked_attestation_source_excluded_but_key_survives_via_rotation(
    client, db_session
):
    keys, made = _setup(client, db_session)
    # Exactly at the revocation instant K4 stays, carried only by R4.
    body = _list(client, at=_iso(T_V4)).json()
    assert body["count"] == 1
    (item,) = body["items"]
    assert item["public_key"] == _key_b64(SEED_K4)
    assert item["sources"] == [
        {
            "source_type": "rotation",
            "source_id": made["R4"],
            "created_at": _iso(T_R4),
        }
    ]


def test_retired_rotation_source_excluded_but_key_survives_via_attestation(
    client, db_session
):
    keys, made = _setup(client, db_session)
    # Exactly at R0's retirement instant K0 stays, carried only by A0.
    body = _list(client, at=_iso(T_R0_RET)).json()
    k0 = next(i for i in body["items"] if i["public_key"] == _key_b64(SEED_A))
    assert k0["sources"] == [
        {
            "source_type": "attestation",
            "source_id": made["A0"],
            "created_at": _iso(T_A0),
        }
    ]


def test_sources_of_one_key_merge_while_both_valid(client, db_session):
    keys, made = _setup(client, db_session)
    # Between R0's creation and retirement K0 is carried by A0 and R0; the
    # sources appear once each in stable source-creation order.
    body = _list(client, at=_iso(T_R0)).json()
    k0 = next(i for i in body["items"] if i["public_key"] == _key_b64(SEED_A))
    assert [s["source_type"] for s in k0["sources"]] == [
        "attestation",
        "rotation",
    ]
    assert [s["source_id"] for s in k0["sources"]] == [made["A0"], made["R0"]]
    assert k0["sources"][0]["created_at"] == _iso(T_A0)
    assert k0["sources"][1]["created_at"] == _iso(T_R0)


def test_full_timeline(client, db_session):
    _setup(client, db_session)
    cases = [
        ("2026-01-01T07:30:00Z", []),
        (_iso(T_A4), [SEED_K4]),
        (_iso(T_R4), [SEED_K4]),
        (_iso(T_V4), [SEED_K4]),
        ("2026-01-01T09:30:00Z", [SEED_K4]),
        (_iso(T_A0), [SEED_K4, SEED_A]),
        (_iso(T_R1), [SEED_K4, SEED_A, SEED_R1]),
        (_iso(T_R1_RET), [SEED_K4, SEED_A]),
        (_iso(T_A2), [SEED_K4, SEED_A, SEED_R2]),
        (_iso(T_R0), [SEED_K4, SEED_A, SEED_R2]),
        (_iso(T_R0_RET), [SEED_K4, SEED_A, SEED_R2]),
        (_iso(T_V2), [SEED_K4, SEED_A]),
        (_iso(T_R3), [SEED_K4, SEED_A, SEED_B]),
        (FUTURE, [SEED_K4, SEED_A, SEED_B]),
    ]
    for at, seeds in cases:
        body = _list(client, at=at).json()
        assert _public_keys(body) == [_key_b64(s) for s in seeds], at
        assert body["count"] == len(seeds), at


def test_historical_ordering_follows_first_valid_source(client, db_session):
    _setup(client, db_session)
    body = _list(client, at=_iso(T_R3)).json()
    # K4 (08:00) before K0 (10:00) before K3 (14:00); K1/K2 have lapsed.
    assert _public_keys(body) == [
        _key_b64(SEED_K4),
        _key_b64(SEED_A),
        _key_b64(SEED_B),
    ]


def test_repeated_reads_are_byte_identical(client, db_session):
    _setup(client, db_session)
    first = _list(client)
    second = _list(client)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_existing_subject_without_keys_is_an_empty_collection(client, db_session):
    _setup(client, db_session)
    create_actor(client, actor_id="org-9", name="No Keys", type="device")
    body = _list(client, "org-9").json()
    assert body == {
        "actor_id": "org-9",
        "at": FUTURE,
        "items": [],
        "count": 0,
        "next_cursor": None,
    }


def test_listing_is_scoped_to_the_path_subject(client, db_session):
    _setup(client, db_session)
    # org-2 has no directly-inserted sources at any point.
    body = _list(client, "org-2").json()
    assert body["actor_id"] == "org-2"
    assert body["items"] == []
    assert body["count"] == 0


# --- Pagination --------------------------------------------------------------------


def test_pagination_concatenates_without_gaps_or_duplicates(client, db_session):
    _setup(client, db_session, extra_rotations=4)
    all_items, pages, count = _walk_pages(client, limit=2)
    # K4, K0, K3 + 4 extra rotations -> 7 keys, pages of 2,2,2,1.
    assert count == 7
    assert [len(page) for page in pages] == [2, 2, 2, 1]
    keys = [item["public_key"] for item in all_items]
    assert len(keys) == len(set(keys)) == 7
    assert keys == [
        _key_b64(SEED_K4),
        _key_b64(SEED_A),
        _key_b64(SEED_B),
        *[_key_b64(_extra_seed(i)) for i in range(4)],
    ]
    assert all_items == _list(client).json()["items"]


def test_count_is_the_dedup_total_on_every_page_not_the_page_count(
    client, db_session
):
    _setup(client, db_session, extra_rotations=4)
    all_items, pages, count = _walk_pages(client, limit=3)
    assert count == 7
    assert [len(page) for page in pages] == [3, 3, 1]
    # Every page independently reports the point-in-time dedup total.
    cursor = None
    for _ in range(3):
        params = {"limit": 3}
        if cursor is not None:
            params["cursor"] = cursor
        body = _list(client, **params).json()
        assert body["count"] == 7
        cursor = body["next_cursor"]
    assert len(all_items) == 7


def test_last_page_cursor_is_null_on_exact_division(client, db_session):
    _setup(client, db_session)
    # Three future-valid keys; a limit of 3 ends exactly on the tail.
    first = _list(client, limit=3).json()
    assert first["next_cursor"] is None
    assert first["count"] == 3


def test_default_limit_is_fifty(client, db_session):
    _setup(client, db_session, extra_rotations=52)
    first = _list(client).json()
    assert len(first["items"]) == 50
    assert first["count"] == 55
    assert first["next_cursor"] is not None
    second = _list(client, cursor=first["next_cursor"]).json()
    assert len(second["items"]) == 5
    assert second["count"] == 55
    assert second["next_cursor"] is None


def test_limit_boundaries_accepted(client, db_session):
    _setup(client, db_session)
    for value in (1, 100):
        assert _list(client, limit=value).status_code == 200


def test_reusing_a_cursor_replays_the_same_page(client, db_session):
    _setup(client, db_session, extra_rotations=3)
    cursor = _list(client, limit=2).json()["next_cursor"]
    one = _list(client, limit=2, cursor=cursor).json()
    two = _list(client, limit=2, cursor=cursor).json()
    assert one == two


def test_issued_cursors_use_the_dedicated_opaque_family_marker(
    client, db_session
):
    _setup(client, db_session, extra_rotations=1)
    token = _list(client, limit=1).json()["next_cursor"]
    assert token.startswith(
        pagination.AUTHENTICATION_KEYS_AT_CURSOR_VERSION + "."
    )
    assert len(token.split(".")) == 3


def test_cursor_past_end_returns_empty_page_with_original_count(client, app, db_session):
    _setup(client, db_session, extra_rotations=2)
    token = pagination.encode_typed_cursor(
        app.state.authentication_keys_at_cursor_secret,
        pagination.AUTHENTICATION_KEYS_AT_CURSOR,
        {"actor_id": "org-1", "at": "2099-01-01T00:00:00+00:00",
         "limit": 50, "offset": 99},
    )
    body = _list(client, cursor=token).json()
    assert body["items"] == []
    assert body["count"] == 5
    assert body["next_cursor"] is None


def test_tail_cursor_returns_empty_items_and_same_count(client, app, db_session):
    _setup(client, db_session)
    token = pagination.encode_typed_cursor(
        app.state.authentication_keys_at_cursor_secret,
        pagination.AUTHENTICATION_KEYS_AT_CURSOR,
        {"actor_id": "org-1", "at": "2099-01-01T00:00:00+00:00",
         "limit": 3, "offset": 3},
    )
    body = _list(client, limit=3, cursor=token).json()
    assert body["items"] == []
    assert body["count"] == 3
    assert body["next_cursor"] is None


# --- Cursor integrity --------------------------------------------------------------


def test_blank_malformed_or_tampered_cursors_are_validation_errors(
    client, app, db_session
):
    _setup(client, db_session, extra_rotations=1)
    good = _list(client, limit=1).json()["next_cursor"]
    tampered = good[:-2] + ("aa" if good[-2:] != "aa" else "bb")
    foreign = pagination.encode_typed_cursor(
        secrets.token_bytes(32),
        pagination.AUTHENTICATION_KEYS_AT_CURSOR,
        {"actor_id": "org-1", "at": "2099-01-01T00:00:00+00:00",
         "limit": 1, "offset": 1},
    )
    secret = app.state.authentication_keys_at_cursor_secret
    at_claim = "2099-01-01T00:00:00+00:00"
    for token in (
        "",
        "   ",
        "\t",
        "not-a-cursor",
        "at1.onlytwoparts",
        "at1.too.many.parts",
        "at0.x.y",
        "at2.x.y",
        # Every other family marker, even signed with this family's secret,
        # is a version mismatch rather than a trusted token.
        _hand_cursor(secret, "au1", {"x": 1}),
        _hand_cursor(secret, "ak1", {"x": 1}),
        _hand_cursor(secret, "v1", {"x": 1}),
        _hand_cursor(secret, "ae1", {"x": 1}),
        # Correctly signed, but structurally invalid claim payloads.
        _hand_cursor(secret, "at1", "not-json"),
        _hand_cursor(
            secret, "at1",
            {"actor_id": "org-1", "at": at_claim, "limit": 1,
             "offset": 1, "extra": 2},
        ),
        _hand_cursor(secret, "at1",
                     {"actor_id": "org-1", "at": at_claim, "limit": 1}),
        _hand_cursor(secret, "at1",
                     {"actor_id": "", "at": at_claim, "limit": 1, "offset": 1}),
        _hand_cursor(secret, "at1",
                     {"actor_id": "org-1", "at": "not-a-time",
                      "limit": 1, "offset": 1}),
        _hand_cursor(secret, "at1",
                     {"actor_id": "org-1", "at": at_claim,
                      "limit": 101, "offset": 1}),
        _hand_cursor(secret, "at1",
                     {"actor_id": "org-1", "at": at_claim,
                      "limit": 0, "offset": 1}),
        _hand_cursor(secret, "at1",
                     {"actor_id": "org-1", "at": at_claim,
                      "limit": 1, "offset": 0}),
        _hand_cursor(secret, "at1",
                     {"actor_id": 7, "at": at_claim, "limit": 1, "offset": 1}),
        _hand_cursor(secret, "at1",
                     {"actor_id": "org-1", "at": at_claim,
                      "limit": "1", "offset": 1}),
        tampered,
        foreign,
    ):
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422, repr(token)
        assert resp.json()["error"]["code"] == "validation_error", repr(token)
        assert "items" not in resp.json()


def test_cursor_from_other_endpoints_is_rejected(client, app, db_session):
    _setup(client, db_session, extra_rotations=2)
    secret = app.state.authentication_keys_at_cursor_secret
    current_cursor = client.get(
        "/v1/actors/org-1/authentication-keys", params={"limit": 1}
    ).json()["next_cursor"]
    rotation_cursor = client.get(
        "/v1/actors/org-1/authentication-key-rotations", params={"limit": 1}
    ).json()["next_cursor"]
    others = [
        current_cursor,
        rotation_cursor,
        pagination.encode_typed_cursor(
            secret,
            pagination.CLAIMS_CURSOR,
            {
                "content_id": None,
                "actor_id": None,
                "claim_type": None,
                "payload_digest_hex": None,
                "limit": 1,
                "offset": 1,
            },
        ),
    ]
    for token in others:
        resp = _list(client, limit=1, cursor=token)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


def test_historical_cursor_is_rejected_by_other_endpoints(client, db_session):
    _setup(client, db_session, extra_rotations=1)
    cursor = _list(client, limit=1).json()["next_cursor"]
    for path in (
        "/v1/actors/org-1/authentication-keys",
        "/v1/actors/org-1/authentication-key-rotations",
        "/v1/claims",
        "/v1/audit-events",
    ):
        resp = client.get(path, params={"limit": 1, "cursor": cursor})
        assert resp.status_code == 422, path
        assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_is_bound_to_its_subject(client, app, db_session):
    _setup(client, db_session, extra_rotations=3)
    cursor = _list(client, "org-1", limit=2).json()["next_cursor"]
    # org-2 exists but has no sources; the org-1 cursor still mismatches.
    resp = _list(client, "org-2", limit=2, cursor=cursor)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

    secret = app.state.authentication_keys_at_cursor_secret
    org1_token = pagination.encode_typed_cursor(
        secret,
        pagination.AUTHENTICATION_KEYS_AT_CURSOR,
        {"actor_id": "org-1", "at": "2099-01-01T00:00:00+00:00",
         "limit": 2, "offset": 2},
    )
    assert _list(
        client, "org-2", limit=2, cursor=org1_token
    ).status_code == 422


def test_cursor_is_bound_to_the_parsed_at(client, db_session):
    _setup(client, db_session, extra_rotations=3)
    cursor = _list(client, at=_iso(T_R1), limit=2).json()["next_cursor"]
    # The same cursor at a different historical condition cannot splice in
    # that other point's result set.
    for other_at in (_iso(T_R3), FUTURE, _iso(T_A0)):
        resp = _list(client, at=other_at, limit=2, cursor=cursor)
        assert resp.status_code == 422, other_at
        assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_binds_the_parsed_instant_not_its_spelling(client, db_session):
    _setup(client, db_session, extra_rotations=3)
    cursor = _list(
        client, at="2099-01-01T00:00:00Z", limit=2
    ).json()["next_cursor"]
    # The equivalent +00:00 spelling is the same parsed instant and resumes.
    resp = _list(
        client, at="2099-01-01T00:00:00+00:00", limit=2, cursor=cursor
    )
    assert resp.status_code == 200, resp.text


def test_cursor_is_bound_to_its_effective_limit(client, db_session):
    _setup(client, db_session, extra_rotations=3)
    cursor = _list(client, limit=2).json()["next_cursor"]
    for params in ({"limit": 3}, {"limit": 1}, {}):
        resp = _list(client, cursor=cursor, **params)
        assert resp.status_code == 422, params
        assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_mismatch_is_rejected_before_actor_lookup(client, db_session):
    _setup(client, db_session, extra_rotations=1)
    cursor = _list(client, "org-1", limit=1).json()["next_cursor"]
    resp = _list(client, "ghost", limit=1, cursor=cursor)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_cursor_secret_rotation_invalidates_outstanding_cursors(
    client, db_session
):
    _setup(client, db_session, extra_rotations=1)
    cursor = _list(client, limit=1).json()["next_cursor"]
    client.app.state.authentication_keys_at_cursor_secret = (
        secrets.token_bytes(32)
    )
    resp = _list(client, limit=1, cursor=cursor)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    fresh = _list(client, limit=1).json()
    assert len(fresh["items"]) == 1


# --- Parameter and body validation -------------------------------------------------


def test_missing_at_is_a_validation_error(client, db_session):
    _setup(client, db_session)
    resp = client.get(URL)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    issue = resp.json()["error"]["details"]["issues"][0]
    assert issue["loc"] == ["query", "at"]


def test_illegal_at_values_are_validation_errors(client, db_session):
    _setup(client, db_session)
    for value in (
        "",
        "   ",
        "2026-01-01",
        "2026-01-01T10:00:00",
        "2026-01-01T10:00Z",
        "2026-01-01T10:00:00z",
        "2026-01-01T10:00:00+01:00",
        "2026-01-01T10:00:00-00:00",
        "2026-13-01T10:00:00Z",
        "2026-01-01 10:00:00Z",
        "not-a-time",
        "now",
    ):
        resp = _list(client, at=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_strict_rfc3339_utc_at_values_are_accepted(client, db_session):
    _setup(client, db_session)
    for value in (
        "2026-01-01T08:00:00Z",
        "2026-01-01T08:00:00+00:00",
        "2026-01-01T08:00:00.500Z",
    ):
        assert _list(client, at=value).status_code == 200, value


def test_illegal_limit_values_are_validation_errors(client, db_session):
    _setup(client, db_session)
    for value in (
        "0", "101", "-1", "1.5", "8.0", "abc", "", "  2", "2  ",
        "+1", "1e2", "０１", "0x1",
    ):
        resp = _list(client, limit=value)
        assert resp.status_code == 422, value
        assert resp.json()["error"]["code"] == "validation_error"


def test_repeated_parameters_are_validation_errors(client, db_session):
    _setup(client, db_session)
    for suffix in (
        f"at={FUTURE}&at=2026-06-02T00:00:00Z",
        "limit=1&limit=2",
        "cursor=x&cursor=y",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_undeclared_parameters_are_validation_errors(client, db_session):
    _setup(client, db_session)
    for suffix in (
        "from=2026-01-01T00:00:00Z",
        "active=true",
        "actor_id=org-2",
        "offset=1",
        "page=1",
        "At=2099-01-01T00:00:00Z",
        f"at={FUTURE}&source_type=rotation",
    ):
        resp = client.get(f"{URL}?{suffix}")
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"


def test_non_empty_body_is_a_validation_error(client, db_session):
    _setup(client, db_session)
    for body in (b"{}", b" ", b"{not json", b"\x00"):
        resp = client.request(
            "GET",
            URL,
            params={"at": FUTURE},
            content=body,
        )
        assert resp.status_code == 422, body
        assert resp.json()["error"]["code"] == "validation_error"


def test_validation_failures_are_decided_before_actor_lookup(client, db_session):
    _setup(client, db_session)
    base = "/v1/actors/ghost/authentication-keys/at"
    for suffix in (
        "",  # missing at
        "at=not-a-time",
        "at=",
        "limit=0",
        "limit=101",
        "limit=abc",
        "limit=1&limit=2",
        "cursor=garbage",
        "cursor=x&cursor=y",
        "bogus=1",
        "active=false",
        f"at={FUTURE}&unknown=1",
    ):
        resp = client.get(f"{base}?{suffix}" if suffix else base)
        assert resp.status_code == 422, suffix
        assert resp.json()["error"]["code"] == "validation_error"
    resp = client.request("GET", base, params={"at": FUTURE}, content=b"{}")
    assert resp.status_code == 422
    # A structurally valid request for the same unknown subject is the 404.
    plain = client.get(base, params={"at": FUTURE})
    assert plain.status_code == 404


def test_validation_failure_locates_the_query_field(client, db_session):
    _setup(client, db_session)
    assert (
        _list(client, at="nope").json()["error"]["details"]["issues"][0]["loc"]
        == ["query", "at"]
    )
    assert (
        _list(client, limit="nope").json()["error"]["details"]["issues"][0][
            "loc"
        ]
        == ["query", "limit"]
    )
    assert (
        _list(client, cursor="garbage").json()["error"]["details"]["issues"][0][
            "loc"
        ]
        == ["query", "cursor"]
    )


# --- Unknown subject and wrong methods ---------------------------------------------


def test_unknown_subject_is_404_unknown_actor(client, db_session):
    _setup(client, db_session)
    resp = _list(client, "ghost")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "unknown_actor"
    assert error["details"]["actor_id"] == "ghost"


def test_unknown_subject_with_limit_is_still_404(client, db_session):
    _setup(client, db_session)
    assert _list(client, "ghost", limit=10).status_code == 404


def test_non_get_methods_are_405_method_not_allowed(client, db_session):
    _setup(client, db_session)
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        resp = client.request(method, URL)
        assert resp.status_code == 405, method
        assert resp.json()["error"]["code"] == "method_not_allowed"


# --- Read-only guarantee -----------------------------------------------------------


def test_reads_and_failures_write_nothing(client, db_session):
    _setup(client, db_session, extra_rotations=2)
    create_actor(client, actor_id="org-9", name="No Keys", type="device")

    def counts():
        return (
            db_session.scalar(select(func.count()).select_from(Attestation)),
            db_session.scalar(
                select(func.count()).select_from(AttestationRevocation)
            ),
            db_session.scalar(
                select(func.count()).select_from(AuthenticationKeyRotation)
            ),
            db_session.scalar(select(func.count()).select_from(AuditEvent)),
        )

    before = counts()

    _walk_pages(client, limit=2)
    _list(client)
    _list(client, at=_iso(T_R1))
    _list(client, "org-9")
    # Failed reads: malformed at/limit/cursor, repeats, unknown params, a
    # non-empty body, an unknown subject, and a bound-cursor mismatch.
    _list(client, at="nope")
    _list(client, limit=0)
    _list(client, cursor="tampered")
    _list(client, "ghost")
    client.get(f"{URL}?limit=1&limit=2")
    client.get(f"{URL}?unknown=1")
    client.request("GET", URL, content=b"{}")
    cursor = _list(client, "org-1", limit=1).json()["next_cursor"]
    _list(client, "org-2", limit=1, cursor=cursor)

    db_session.expire_all()
    assert counts() == before


# --- Compatibility: existing behavior is unchanged ----------------------------------


def test_far_future_historical_matches_the_current_view(client):
    att1, _ = _world(client)
    _post_rotation(client, _rotation_body(seed=SEED_R1), seed=SEED_A)
    _post_rotation(client, _rotation_body(seed=SEED_R2), seed=SEED_R1)

    current = client.get(
        "/v1/actors/org-1/authentication-keys"
    ).json()
    historical = _list(client, at=FUTURE).json()
    assert _public_keys(historical) == [
        item["public_key"] for item in current["items"]
    ]
    assert historical["count"] == current["count"] == 3

    # Retire R1 and revoke the bootstrap attestation: the historical future
    # view tracks the current union, and the current route is unchanged.
    rotations = client.get(
        "/v1/actors/org-1/authentication-key-rotations"
    ).json()["items"]
    r1 = next(r for r in rotations if r["new_public_key"] == _key_b64(SEED_R1))
    assert _post_retire(client, r1["id"], seed=SEED_R1).status_code == 200
    client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": att1["id"],
            "revoker_actor_id": "org-1",
            "reason": "key material retired",
        },
    )
    current = client.get(
        "/v1/actors/org-1/authentication-keys"
    ).json()
    historical = _list(client, at=FUTURE).json()
    assert _public_keys(historical) == [_key_b64(SEED_R2)]
    assert [i["public_key"] for i in current["items"]] == [_key_b64(SEED_R2)]


def test_historical_route_does_not_affect_rotation_or_current_routes(client):
    att1, _ = _world(client)
    created = _post_rotation(client, _rotation_body())
    assert created.status_code == 201
    # Repeated historical reads never create or modify a rotation.
    for _ in range(3):
        assert _list(client, at=FUTURE).status_code == 200
    repeat = _post_rotation(client, _rotation_body())
    assert repeat.status_code == 200
    assert repeat.json() == created.json()
