"""Tests for the stateless revocation-impact checkpoint verification route.

Covers POST /v1/impact-verifications: the request body is exactly
{"checkpoint", "impacts"}; the checkpoint is exactly
{"checkpoint_version", "digest_algorithm", "impact_count",
"impacts_digest_hex"} with the version pinned to
"provenance-revocation-impact-checkpoint-v1", the algorithm to "sha256",
the count a non-negative integer, and the digest 64 lowercase hex
characters. The impacts array must contain exactly ``impact_count``
elements, each exactly the exported 13-field revocation-impact public
view (strict RFC 3339 UTC ``created_at``; non-empty identifiers and
reason; ``target_type`` in {"claim", "evidence_bundle"}; the two
coverage statuses in {"uncovered", "partial", "covered"}; non-negative
integral counts; no undeclared member) with internally consistent
before/after coverage fields (``delta == before - after`` in {0, 1};
the after count never exceeds the before count; a positive signer count
always pairs with "covered" and a zero count never does). The digest
is the SHA-256 of the canonical impacts array (array order preserved,
nested object keys sorted by Unicode code point, compact separators,
non-ASCII unescaped, UTF-8) computed over the impacts exactly as
received. Any structural, field, type, association, count, timestamp,
digest-input, or malformed-JSON violation is a 422 validation_error; a
structurally valid request whose digest differs is 200 {"valid":
false, "computed_digest_hex": ...}; a match is exactly {"valid": true}.
The route accepts no query parameters and is fully stateless: it needs
no persisted revocation and creates, modifies, logs, and queries
nothing, so unknown resources, repeated requests, and differing local
state verify identically. All fixtures are deterministic and offline.
"""

from __future__ import annotations

import copy
import hashlib
import json

from sqlalchemy import func, select

from provenance.models import (
    Actor,
    AttestationRevocation,
    AuditEvent,
    Claim,
    Content,
    EvidenceBundle,
)
from tests.test_revocation_impacts import _world

URL = "/v1/impact-verifications"
PACKAGE_PATH = "/v1/revocation-impact-package"
CHECKPOINT_VERSION = "provenance-revocation-impact-checkpoint-v1"

CHECKPOINT_FIELDS = {
    "checkpoint_version",
    "digest_algorithm",
    "impact_count",
    "impacts_digest_hex",
}


def _digest_of(impacts: list) -> str:
    """Independently canonicalize the impact array and digest it."""
    canonical = json.dumps(
        impacts,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _impact(
    *,
    id="rev_01hztest000000000000000001",
    attestation_id="att_01hztest00000000000000001",
    revoker_actor_id="org-1",
    reason="no longer relied upon",
    created_at="2026-01-02T03:04:05Z",
    content_id="cnt_01hztest00000000000000001",
    target_type="claim",
    signer_actor_id="org-1",
    after=0,
    status_after="partial",
    before=1,
    status_before="covered",
):
    return {
        "id": id,
        "attestation_id": attestation_id,
        "revoker_actor_id": revoker_actor_id,
        "reason": reason,
        "created_at": created_at,
        "content_id": content_id,
        "target_type": target_type,
        "signer_actor_id": signer_actor_id,
        "qualified_signer_count_after": after,
        "coverage_status_after": status_after,
        "qualified_signer_count_before": before,
        "coverage_status_before": status_before,
        "qualified_signer_count_delta": before - after,
    }


def _offline_impacts() -> list:
    """A self-contained impact sequence fabricated without service state."""
    return [
        _impact(
            id="rev_01hztest000000000000000001",
            attestation_id="att_01hztest00000000000000001",
            content_id="cnt_01hztest00000000000000001",
            created_at="2026-01-02T03:04:05Z",
            after=0,
            status_after="partial",
            before=1,
            status_before="covered",
        ),
        _impact(
            id="rev_01hztest000000000000000002",
            attestation_id="att_01hztest00000000000000002",
            revoker_actor_id="p-1",
            reason="Képek ⛄",
            created_at="2026-01-02T03:04:06.500+00:00",
            content_id="cnt_01hztest00000000000000002",
            target_type="evidence_bundle",
            signer_actor_id="p-1",
            after=1,
            status_after="covered",
            before=2,
            status_before="covered",
        ),
        _impact(
            id="rev_01hztest000000000000000003",
            attestation_id="att_01hztest00000000000000003",
            content_id="cnt_01hztest00000000000000003",
            created_at="2026-01-03T00:00:00Z",
            after=0,
            status_after="uncovered",
            before=0,
            status_before="uncovered",
        ),
    ]


def _request(impacts: list | None = None, *, digest=None) -> dict:
    impacts = _offline_impacts() if impacts is None else impacts
    return {
        "checkpoint": {
            "checkpoint_version": CHECKPOINT_VERSION,
            "digest_algorithm": "sha256",
            "impact_count": len(impacts),
            "impacts_digest_hex": _digest_of(impacts)
            if digest is None
            else digest,
        },
        "impacts": impacts,
    }


def _served_request(client, **params) -> dict:
    package = client.get(PACKAGE_PATH, params=params).json()
    return {"checkpoint": package["checkpoint"], "impacts": package["impacts"]}


def _assert_validation_error(resp) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"


# --- Verdicts ---------------------------------------------------------------------


def test_verification_of_offline_package_is_valid_on_empty_database(client):
    # No resource exists locally and every identifier is unknown: the
    # verdict is decided by the request body alone.
    resp = client.post(URL, json=_request())
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_of_served_package_is_valid(client):
    _world(client)
    resp = client.post(URL, json=_served_request(client))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_verification_of_empty_snapshot_is_valid(client):
    for payload in (
        _request(impacts=[]),
        _served_request(client, revoker_actor_id="nobody"),
    ):
        assert payload["checkpoint"]["impact_count"] == 0
        assert payload["checkpoint"]["impacts_digest_hex"] == hashlib.sha256(
            b"[]"
        ).hexdigest()
        resp = client.post(URL, json=payload)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"valid": True}


def test_digest_mismatch_returns_exactly_valid_false_and_computed_digest(client):
    request = _request()
    claimed = request["checkpoint"]["impacts_digest_hex"]
    request["checkpoint"]["impacts_digest_hex"] = claimed[:-1] + (
        "0" if claimed[-1] != "0" else "1"
    )
    resp = client.post(URL, json=request)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"valid", "computed_digest_hex"}
    assert body["valid"] is False
    assert body["computed_digest_hex"] == _digest_of(_offline_impacts())
    assert body["computed_digest_hex"] == claimed


def test_mismatch_response_is_compact_json_with_one_newline(client):
    request = _request()
    request["checkpoint"]["impacts_digest_hex"] = "0" * 64
    raw = client.post(URL, json=request).content
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw
    assert b": " not in raw
    body = json.loads(raw)
    assert body["valid"] is False
    assert len(body["computed_digest_hex"]) == 64

    matched = client.post(URL, json=_request()).content
    assert matched == b'{"valid":true}\n'


def test_tampered_impact_is_invalid(client):
    request = _request()
    request["impacts"][0]["reason"] = "forged"
    resp = client.post(URL, json=request)
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert body["computed_digest_hex"] == _digest_of(request["impacts"])


def test_array_order_participates_in_the_digest(client):
    request = _request()
    reordered = copy.deepcopy(request)
    reordered["impacts"] = list(reversed(reordered["impacts"]))
    resp = client.post(URL, json=reordered)
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert body["computed_digest_hex"] == _digest_of(reordered["impacts"])


def test_nested_object_key_spelling_is_irrelevant_to_the_digest(client):
    # The canonical form sorts nested object keys by Unicode code point:
    # reserializing each impact with keys in another order must still
    # match, while the array order itself stays significant.
    request = _request()
    reshuffled = copy.deepcopy(request)
    reshuffled["impacts"] = [
        {k: impact[k] for k in sorted(impact, reverse=True)}
        for impact in reshuffled["impacts"]
    ]
    resp = client.post(URL, json=reshuffled)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


def test_non_ascii_is_canonicalized_unescaped(client):
    request = _request()
    assert any(
        ord(ch) > 127
        for ch in json.dumps(request["impacts"], ensure_ascii=False)
    )
    assert client.post(URL, json=request).json() == {"valid": True}
    escaped = json.dumps(
        request["impacts"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    assert hashlib.sha256(escaped).hexdigest() != _digest_of(request["impacts"])


def test_equivalent_utc_spellings_are_accepted(client):
    # Both Z and +00:00 are strict RFC 3339 UTC spellings; the digest
    # commits to the exact received spelling (the route normalizes
    # nothing), but each is a legal digest input and verifies.
    for spelling in (
        "2026-01-02T03:04:05Z",
        "2026-01-02T03:04:05+00:00",
        "2026-01-02T03:04:05.000Z",
    ):
        impacts = [_impact(created_at=spelling)]
        assert client.post(URL, json=_request(impacts)).json() == {"valid": True}


# --- Statelessness ----------------------------------------------------------------


def test_verification_is_deterministic_across_repeated_requests(client):
    _world(client)
    request = _served_request(client)
    first = client.post(URL, json=request)
    second = client.post(URL, json=request)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content == b'{"valid":true}\n'


def test_local_state_changes_do_not_change_the_verdict(client):
    # First verify a served package, then add more revocations: the
    # earlier package must still verify byte-for-byte because the route
    # never queries local state.
    _world(client)
    package = _served_request(client)
    assert client.post(URL, json=package).json() == {"valid": True}

    # Creating new local resources changes the current impact listing but
    # cannot influence a verdict computed from the request body.
    import base64

    from provenance.signing import attestation_message_bytes
    from tests.helpers import (
        content_payload,
        create_actor,
        ed25519_public_key,
        ed25519_sign,
        SEED_A,
    )

    create_actor(client, actor_id="org-9", name="Nine", type="person")
    content = client.post(
        "/v1/contents",
        json=content_payload(
            digest=hashlib.sha256(b"state-change").hexdigest(),
            actor_id="org-9",
        ),
    ).json()
    claim = client.post(
        "/v1/claims",
        json={
            "content_id": content["id"],
            "actor_id": "org-9",
            "claim_type": "authorship",
            "payload": {"s": "nine"},
        },
    ).json()
    signature = ed25519_sign(
        SEED_A,
        attestation_message_bytes("claim", claim["id"], "org-9"),
    )
    attestation = client.post(
        "/v1/attestations",
        json={
            "target_type": "claim",
            "target_id": claim["id"],
            "signer_actor_id": "org-9",
            "public_key": base64.b64encode(ed25519_public_key(SEED_A)).decode(),
            "signature": base64.b64encode(signature).decode(),
        },
    ).json()
    client.post(
        "/v1/attestation-revocations",
        json={
            "attestation_id": attestation["id"],
            "revoker_actor_id": "org-9",
            "reason": "later change",
        },
    )
    assert client.post(URL, json=package).json() == {"valid": True}
    # And a package carrying unknown/fabricated ids verifies regardless.
    assert client.post(URL, json=_request()).json() == {"valid": True}


def test_verification_writes_no_resources_or_audit_events(client, db_session):
    _world(client)
    valid_request = _served_request(client)
    tampered = copy.deepcopy(valid_request)
    tampered["impacts"][0]["reason"] = "forged"
    offline = _request()

    models = (
        Actor,
        Content,
        Claim,
        EvidenceBundle,
        AttestationRevocation,
        AuditEvent,
    )

    def _counts():
        return {
            model: db_session.scalar(select(func.count()).select_from(model))
            for model in models
        }

    before = _counts()
    assert client.post(URL, json=valid_request).status_code == 200
    assert client.post(URL, json=tampered).status_code == 200
    assert client.post(URL, json=offline).status_code == 200
    _assert_validation_error(client.post(URL, json={}))
    _assert_validation_error(
        client.post(
            URL,
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
    )
    db_session.expire_all()
    assert _counts() == before


# --- Top-level structure -----------------------------------------------------------


def test_requires_exactly_checkpoint_and_impacts(client):
    request = _request()
    for field in ("checkpoint", "impacts"):
        incomplete = {k: v for k, v in request.items() if k != field}
        _assert_validation_error(client.post(URL, json=incomplete))

    augmented = dict(request)
    augmented["extra"] = "nope"
    _assert_validation_error(client.post(URL, json=augmented))
    _assert_validation_error(client.post(URL, json=[]))
    _assert_validation_error(client.post(URL, json="checkpoint"))
    _assert_validation_error(client.post(URL, json=None))
    _assert_validation_error(client.post(URL, json=42))
    _assert_validation_error(client.post(URL, json={}))


def test_members_have_correct_types(client):
    request = _request()
    for member, wrong in (
        ("checkpoint", []),
        ("checkpoint", None),
        ("checkpoint", "x"),
        ("impacts", {}),
        ("impacts", None),
        ("impacts", "impacts"),
    ):
        bad = copy.deepcopy(request)
        bad[member] = wrong
        _assert_validation_error(client.post(URL, json=bad))


def test_malformed_and_empty_bodies_are_validation_errors(client):
    for body_bytes in (
        b"",
        b"   ",
        b"\n\t",
        b"{not json",
        b'{"checkpoint":',
        b"[]",
        b"null",
        b"\xff\xfe",
    ):
        resp = client.post(
            URL,
            content=body_bytes,
            headers={"Content-Type": "application/json"},
        )
        _assert_validation_error(resp)


def test_query_parameters_are_rejected(client):
    request = _request()
    for suffix in ("?x=1", "?limit=1", "?cursor=x", "?valid=true"):
        resp = client.post(URL + suffix, json=request)
        _assert_validation_error(resp)


# --- Checkpoint fields -------------------------------------------------------------


def test_checkpoint_requires_exactly_its_four_fields(client):
    request = _request()
    for field in CHECKPOINT_FIELDS:
        bad = copy.deepcopy(request)
        del bad["checkpoint"][field]
        _assert_validation_error(client.post(URL, json=bad))
    bad = copy.deepcopy(request)
    bad["checkpoint"]["extra"] = 1
    _assert_validation_error(client.post(URL, json=bad))


def test_checkpoint_version_and_algorithm_are_pinned(client):
    request = _request()
    for field, wrong in (
        ("checkpoint_version", "provenance-revocation-impact-checkpoint-v2"),
        ("checkpoint_version", ""),
        ("checkpoint_version", None),
        ("checkpoint_version", 1),
        ("digest_algorithm", "sha512"),
        ("digest_algorithm", "SHA256"),
        ("digest_algorithm", None),
    ):
        bad = copy.deepcopy(request)
        bad["checkpoint"][field] = wrong
        _assert_validation_error(client.post(URL, json=bad))


def test_impact_count_must_be_a_non_negative_integer(client):
    request = _request()
    for wrong in (-1, 1.0, "3", None, True, False):
        bad = copy.deepcopy(request)
        bad["checkpoint"]["impact_count"] = wrong
        _assert_validation_error(client.post(URL, json=bad))


def test_impact_count_must_match_the_array_length(client):
    request = _request()
    for claimed in (0, 1, 2, 4, 99):
        bad = copy.deepcopy(request)
        bad["checkpoint"]["impact_count"] = claimed
        _assert_validation_error(client.post(URL, json=bad))
    # The exact length is accepted (regardless of the digest, which is
    # checked independently of the count).
    okay = copy.deepcopy(request)
    okay["checkpoint"]["impacts_digest_hex"] = "0" * 64
    resp = client.post(URL, json=okay)
    assert resp.status_code == 200
    assert resp.json()["valid"] is False


def test_digest_must_be_64_lowercase_hex(client):
    request = _request()
    good = request["checkpoint"]["impacts_digest_hex"]
    for wrong in (
        good.upper(),
        good[:-1],
        good + "0",
        "g" * 64,
        " " + good,
        good + " ",
        "",
        None,
        64,
        True,
    ):
        bad = copy.deepcopy(request)
        bad["checkpoint"]["impacts_digest_hex"] = wrong
        _assert_validation_error(client.post(URL, json=bad))


# --- Impact item structure ---------------------------------------------------------


def test_impact_requires_exactly_its_thirteen_fields(client):
    impacts = _offline_impacts()
    required = set(impacts[0])
    assert len(required) == 13
    for field in required:
        bad_impacts = copy.deepcopy(impacts)
        del bad_impacts[0][field]
        _assert_validation_error(client.post(URL, json=_request(bad_impacts)))
    bad_impacts = copy.deepcopy(impacts)
    bad_impacts[0]["signature"] = "abcd"
    _assert_validation_error(client.post(URL, json=_request(bad_impacts)))
    bad_impacts = copy.deepcopy(impacts)
    bad_impacts[0]["public_key"] = "abcd"
    _assert_validation_error(client.post(URL, json=_request(bad_impacts)))
    bad_impacts = copy.deepcopy(impacts)
    bad_impacts[0]["payload"] = {}
    _assert_validation_error(client.post(URL, json=_request(bad_impacts)))
    bad_impacts = copy.deepcopy(impacts)
    bad_impacts[0]["seq"] = 1
    _assert_validation_error(client.post(URL, json=_request(bad_impacts)))


def test_impact_string_fields_must_be_nonempty_strings(client):
    impacts = _offline_impacts()
    for field in (
        "id",
        "attestation_id",
        "revoker_actor_id",
        "reason",
        "content_id",
        "signer_actor_id",
    ):
        for wrong in ("", "   ", "\t", None, 7, [], {}):
            bad_impacts = copy.deepcopy(impacts)
            bad_impacts[0][field] = wrong
            _assert_validation_error(client.post(URL, json=_request(bad_impacts)))


def test_target_type_and_coverage_statuses_are_pinned_literals(client):
    impacts = _offline_impacts()
    for field, wrong in (
        ("target_type", "Claim"),
        ("target_type", "claim "),
        ("target_type", "content"),
        ("target_type", None),
        ("target_type", 1),
        ("coverage_status_after", "COVERED"),
        ("coverage_status_after", "trusted"),
        ("coverage_status_after", None),
        ("coverage_status_before", "Covered"),
        ("coverage_status_before", "untrusted"),
    ):
        bad_impacts = copy.deepcopy(impacts)
        bad_impacts[0][field] = wrong
        _assert_validation_error(client.post(URL, json=_request(bad_impacts)))


def test_count_fields_must_be_non_negative_integers(client):
    impacts = _offline_impacts()
    for field in (
        "qualified_signer_count_after",
        "qualified_signer_count_before",
        "qualified_signer_count_delta",
    ):
        for wrong in (-1, 1.5, "1", None, True, False):
            bad_impacts = copy.deepcopy(impacts)
            bad_impacts[0][field] = wrong
            _assert_validation_error(client.post(URL, json=_request(bad_impacts)))


def test_created_at_must_be_strict_rfc3339_utc(client):
    impacts = _offline_impacts()
    for wrong in (
        "",
        "   ",
        "2026-01-02",
        "2026-01-02T03:04:05",
        "2026-01-02 03:04:05Z",
        "2026-01-02T03:04Z",
        "2026-01-02T03:04:05+01:00",
        "2026-13-02T03:04:05Z",
        "2026-01-02t03:04:05z",
        None,
        1735786000,
    ):
        bad_impacts = copy.deepcopy(impacts)
        bad_impacts[0]["created_at"] = wrong
        _assert_validation_error(client.post(URL, json=_request(bad_impacts)))


# --- Internal association rules ----------------------------------------------------


def test_delta_must_equal_before_minus_after(client):
    impacts = _offline_impacts()
    for wrong_delta in (0, 2, -1):
        bad_impacts = copy.deepcopy(impacts)
        bad_impacts[0]["qualified_signer_count_delta"] = wrong_delta
        _assert_validation_error(client.post(URL, json=_request(bad_impacts)))
    # The valid delta verifies.
    assert client.post(URL, json=_request(impacts)).json() == {"valid": True}


def test_after_count_must_not_exceed_before_count(client):
    bad_impacts = [
        _impact(after=2, status_after="covered", before=1, status_before="covered")
    ]
    # delta = -1 independently violates the zero/one rule too, but the
    # after>before association is itself rejected regardless of delta.
    bad_impacts[0]["qualified_signer_count_delta"] = 1
    _assert_validation_error(client.post(URL, json=_request(bad_impacts)))


def test_delta_must_be_zero_or_one(client):
    # Construct counts whose raw difference is 2, then claim it: after=0,
    # before=2, delta=2.
    bad_impacts = [
        _impact(after=0, status_after="partial", before=2, status_before="covered")
    ]
    bad_impacts[0]["qualified_signer_count_delta"] = 2
    _assert_validation_error(client.post(URL, json=_request(bad_impacts)))


def test_coverage_status_must_match_its_signer_count(client):
    # A positive count cannot be uncovered/partial; a zero count cannot be
    # covered.
    cases = [
        _impact(after=1, status_after="partial", before=2, status_before="covered"),
        _impact(after=1, status_after="uncovered", before=2, status_before="covered"),
        _impact(after=0, status_after="partial", before=1, status_before="partial"),
        _impact(after=0, status_after="partial", before=1, status_before="uncovered"),
    ]
    for bad_impact in cases:
        bad_impacts = [bad_impact]
        _assert_validation_error(client.post(URL, json=_request(bad_impacts)))

    # Zero counts with uncovered/partial and positive counts with covered
    # are all internally consistent.
    for impact in (
        _impact(after=0, status_after="uncovered", before=0, status_before="uncovered"),
        _impact(after=0, status_after="partial", before=0, status_before="partial"),
        _impact(after=1, status_after="covered", before=2, status_before="covered"),
        _impact(after=0, status_after="partial", before=1, status_before="covered"),
    ):
        resp = client.post(URL, json=_request([impact]))
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"valid": True}


def test_structurally_valid_but_fabricated_associations_verify(client):
    # The route never resolves ids: cross-references that do not exist
    # locally (or that could not be produced by the live search, such as
    # an unrelated signer/content pairing) are not 422s -- they verify as
    # long as the structure and digest hold. Only the record's own
    # before/after invariants are checkable associations.
    impacts = [
        _impact(
            attestation_id="att_fabricated",
            content_id="cnt_fabricated",
            signer_actor_id="actor_fabricated",
            revoker_actor_id="revoker_fabricated",
        )
    ]
    resp = client.post(URL, json=_request(impacts))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True}


# --- End-to-end with the export ----------------------------------------------------


def test_every_served_package_round_trips_through_verification(client):
    _world(client)
    for params in (
        {},
        {"revoker_actor_id": "p-1"},
        {"reason": "fourth"},
        {"attestation_id": "att_nonexistent"},
        {"from": "2026-01-01T00:00:00Z"},
        {
            "from": "2026-01-01T00:00:00Z",
            "to": "2027-01-01T00:00:00Z",
        },
    ):
        package = client.get(PACKAGE_PATH, params=params).json()
        assert package["checkpoint"]["impact_count"] == len(package["impacts"])
        resp = client.post(
            URL,
            json={
                "checkpoint": package["checkpoint"],
                "impacts": package["impacts"],
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"valid": True}, params
