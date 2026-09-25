"""Versioned, signed, opaque pagination cursors.

A cursor captures the exact position within one filtered result set and
binds to every effective query parameter of the request that produced it.
Cursors are opaque to clients: the payload is JSON wrapped in URL-safe
Base64 and authenticated with HMAC-SHA256 using a server-held secret, so a
cursor cannot be forged or tampered with. Any malformed, unsigned,
wrong-version/kind, or structurally invalid token is rejected by the
caller as a ``422 validation_error`` rather than trusted.

Each cursor family (lineage traversal, a content's evidence-bundle list,
...) defines its own :class:`CursorKind`: a distinct format marker, claim
set, and claim validation. A cursor minted for one family can never resume
another family's query, even though all families share the server secret.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from provenance.time_utils import parse_rfc3339_utc

#: Exactly 64 lowercase hexadecimal characters (a SHA-256 digest spelling).
_HEX64_LOWER = re.compile(r"[0-9a-f]{64}")

#: Legacy lineage cursor format generation. Bump only when the lineage claim
#: set/encoding changes; cursors carrying any other marker are rejected as
#: expired/unknown.
LINEAGE_CURSOR_VERSION = "v1"

#: Marker for cursors that page through one content's evidence bundles.
CONTENT_EVIDENCE_CURSOR_VERSION = "ce1"

#: Marker for cursors that page through the audit-event search.
AUDIT_EVENTS_CURSOR_VERSION = "ae1"

#: Marker for cursors that page through exchange-import receipts.
EXCHANGE_IMPORTS_CURSOR_VERSION = "ei1"

#: Marker for cursors that page through exchange-import reconciliations.
EXCHANGE_IMPORT_RECONCILIATIONS_CURSOR_VERSION = "ir1"

#: Marker for cursors that page through checkpoint-import receipts.
CHECKPOINT_IMPORTS_CURSOR_VERSION = "ci1"

#: Marker for cursors that page through impact-import receipts.
IMPACT_IMPORTS_CURSOR_VERSION = "ii1"

#: Marker for cursors that page through checkpoint-import reconciliations.
CHECKPOINT_IMPORT_RECONCILIATIONS_CURSOR_VERSION = "cr1"

#: Marker for cursors that page through the reviewer claim search.
CLAIMS_CURSOR_VERSION = "cl1"

#: Marker for cursors that page through the reviewer evidence-bundle search.
EVIDENCE_BUNDLES_CURSOR_VERSION = "eb1"

#: Marker for cursors that page through one actor's authentication-key
#: rotations.
AUTHENTICATION_KEY_ROTATIONS_CURSOR_VERSION = "ak1"

#: Marker for cursors that page through one attestation's access grants.
ATTESTATION_ACCESS_GRANTS_CURSOR_VERSION = "ag1"

#: Marker for cursors that page through the content export job search.
CONTENT_EXPORT_JOBS_CURSOR_VERSION = "cx1"

#: Marker for cursors that page through a claim's supersession-lineage
#: traversal.
CLAIM_SUPERSESSION_LINEAGE_CURSOR_VERSION = "sl1"

#: Marker for cursors that page through the trust-policy retrieval.
TRUST_POLICIES_CURSOR_VERSION = "tp1"

#: Marker for cursors that page through the reviewer actor retrieval.
ACTORS_CURSOR_VERSION = "ac1"

#: Marker for cursors that page through the reviewer content search.
CONTENTS_CURSOR_VERSION = "ct1"

#: Marker for cursors that page through the global attestation-revocation
#: search.
ATTESTATION_REVOCATIONS_CURSOR_VERSION = "ar1"

#: Marker for cursors that page through the global content-relation search.
CONTENT_RELATIONS_CURSOR_VERSION = "rl1"

#: Marker for cursors that page through the cross-content coverage search.
CONTENT_COVERAGE_SEARCH_CURSOR_VERSION = "cc1"

#: Marker for cursors that page through the cross-content revocation-impact
#: search.
REVOCATION_IMPACTS_CURSOR_VERSION = "ri1"


class InvalidCursorError(ValueError):
    """The cursor token is absent, malformed, expired, or unverifiable."""


@dataclass(frozen=True)
class CursorKind:
    """One cursor family: its wire marker, claim fields, and validation."""

    #: First token segment; namespaces the family and its format generation.
    version: str
    #: Claims whose exact values must match the request carrying the cursor,
    #: in stable serialized order. ``offset`` is the position state itself.
    claim_fields: tuple[str, ...]
    #: Raises :class:`InvalidCursorError` for a wrongly-typed/out-of-range
    #: claim of a structurally decoded payload.
    validate: Callable[[dict[str, Any]], None]


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    try:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode(value + padding)
    except (binascii.Error, ValueError) as exc:
        raise InvalidCursorError("cursor is not valid base64") from exc


def _sign(secret: bytes, version: str, payload: str) -> bytes:
    return hmac.new(
        secret, f"{version}.{payload}".encode("utf-8"), hashlib.sha256
    ).digest()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _optional_nonempty_str(claims: dict[str, Any], field: str) -> None:
    value = claims[field]
    if value is not None and (not isinstance(value, str) or not value):
        raise InvalidCursorError(f"cursor {field} is invalid")


def _validate_lineage_claims(claims: dict[str, Any]) -> None:
    if not isinstance(claims["content_id"], str) or not claims["content_id"]:
        raise InvalidCursorError("cursor content_id is invalid")
    if claims["direction"] not in ("ancestors", "descendants"):
        raise InvalidCursorError("cursor direction is invalid")
    if not isinstance(claims["relation_type"], str | None) or (
        isinstance(claims["relation_type"], str)
        and claims["relation_type"] not in ("version_of", "derived_from")
    ):
        raise InvalidCursorError("cursor relation_type is invalid")
    for field in ("max_depth", "min_depth", "limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["max_depth"] <= 32):
        raise InvalidCursorError("cursor max_depth is out of range")
    if not (1 <= claims["min_depth"] <= claims["max_depth"]):
        raise InvalidCursorError("cursor min_depth is out of range")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    # An offset of 0 is never issued (a first page carries no cursor); a
    # negative offset is structurally invalid. No tight upper bound is
    # imposed: a wide graph can legitimately page past any small constant,
    # and a client cannot forge a large value without the server secret.
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Lineage traversal cursor family (kept under the original ``v1`` marker).
LINEAGE_CURSOR = CursorKind(
    version=LINEAGE_CURSOR_VERSION,
    claim_fields=(
        "content_id",
        "direction",
        "max_depth",
        "min_depth",
        "relation_type",
        "limit",
        "offset",
    ),
    validate=_validate_lineage_claims,
)


def _validate_content_evidence_claims(claims: dict[str, Any]) -> None:
    if not isinstance(claims["content_id"], str) or not claims["content_id"]:
        raise InvalidCursorError("cursor content_id is invalid")
    # The exact-match filters are either absent (null) or non-empty strings.
    _optional_nonempty_str(claims, "evidence_type")
    _optional_nonempty_str(claims, "media_type")
    for field in ("limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Cursor family for ``GET /v1/contents/{id}/evidence-bundles``.
CONTENT_EVIDENCE_CURSOR = CursorKind(
    version=CONTENT_EVIDENCE_CURSOR_VERSION,
    claim_fields=(
        "content_id",
        "evidence_type",
        "media_type",
        "limit",
        "offset",
    ),
    validate=_validate_content_evidence_claims,
)


def _validate_audit_events_claims(claims: dict[str, Any]) -> None:
    # The exact-match filters are either absent (null) or non-empty strings.
    _optional_nonempty_str(claims, "event_type")
    _optional_nonempty_str(claims, "resource_id")
    # The time bounds are absent (null) or canonical RFC 3339 UTC strings.
    for field in ("from", "to"):
        value = claims[field]
        if value is not None and (
            not isinstance(value, str) or parse_rfc3339_utc(value) is None
        ):
            raise InvalidCursorError(f"cursor {field} is invalid")
    for field in ("limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Cursor family for ``GET /v1/audit-events``.
AUDIT_EVENTS_CURSOR = CursorKind(
    version=AUDIT_EVENTS_CURSOR_VERSION,
    claim_fields=(
        "event_type",
        "resource_id",
        "from",
        "to",
        "limit",
        "offset",
    ),
    validate=_validate_audit_events_claims,
)


def _validate_exchange_imports_claims(claims: dict[str, Any]) -> None:
    # The exact-match filters are either absent (null) or non-empty strings;
    # matching is case- and whitespace-sensitive, so the raw value is bound.
    _optional_nonempty_str(claims, "manifest_version")
    _optional_nonempty_str(claims, "evidence_bundle_id")
    _optional_nonempty_str(claims, "manifest_digest_hex")
    for field in ("limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Cursor family for ``GET /v1/evidence-bundle-exchange-imports``.
EXCHANGE_IMPORTS_CURSOR = CursorKind(
    version=EXCHANGE_IMPORTS_CURSOR_VERSION,
    claim_fields=(
        "manifest_version",
        "evidence_bundle_id",
        "manifest_digest_hex",
        "limit",
        "offset",
    ),
    validate=_validate_exchange_imports_claims,
)


def _validate_exchange_import_reconciliations_claims(
    claims: dict[str, Any],
) -> None:
    # The two read-time reconciliation filters are either absent (null,
    # meaning unfiltered) or strict JSON booleans bound from the request's
    # lowercase ``true``/``false`` literals.
    for field in ("local_available", "matches"):
        if claims[field] is not None and not isinstance(claims[field], bool):
            raise InvalidCursorError(f"cursor {field} is invalid")
    for field in ("limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Cursor family for
#: ``GET /v1/evidence-bundle-exchange-import-reconciliations``.
EXCHANGE_IMPORT_RECONCILIATIONS_CURSOR = CursorKind(
    version=EXCHANGE_IMPORT_RECONCILIATIONS_CURSOR_VERSION,
    claim_fields=("local_available", "matches", "limit", "offset"),
    validate=_validate_exchange_import_reconciliations_claims,
)


def _validate_checkpoint_imports_claims(claims: dict[str, Any]) -> None:
    # The exact-match string filters are either absent (null) or non-empty
    # strings; matching is case- and whitespace-sensitive, so the raw value
    # is bound. The event-count filter is absent (null) or a non-negative
    # integer.
    _optional_nonempty_str(claims, "checkpoint_version")
    _optional_nonempty_str(claims, "events_digest_hex")
    event_count = claims["event_count"]
    if event_count is not None and (
        not _is_int(event_count) or event_count < 0
    ):
        raise InvalidCursorError("cursor event_count is invalid")
    for field in ("limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Cursor family for ``GET /v1/audit-events/checkpoint-imports``.
CHECKPOINT_IMPORTS_CURSOR = CursorKind(
    version=CHECKPOINT_IMPORTS_CURSOR_VERSION,
    claim_fields=(
        "checkpoint_version",
        "events_digest_hex",
        "event_count",
        "limit",
        "offset",
    ),
    validate=_validate_checkpoint_imports_claims,
)


def _validate_impact_imports_claims(claims: dict[str, Any]) -> None:
    # The exact-match string filters are either absent (null) or non-empty
    # strings; matching is case- and whitespace-sensitive, so the raw value
    # is bound. The impact-count filter is absent (null) or a non-negative
    # integer.
    _optional_nonempty_str(claims, "checkpoint_version")
    _optional_nonempty_str(claims, "impacts_digest_hex")
    impact_count = claims["impact_count"]
    if impact_count is not None and (
        not _is_int(impact_count) or impact_count < 0
    ):
        raise InvalidCursorError("cursor impact_count is invalid")
    for field in ("limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Cursor family for ``GET /v1/impact-imports``.
IMPACT_IMPORTS_CURSOR = CursorKind(
    version=IMPACT_IMPORTS_CURSOR_VERSION,
    claim_fields=(
        "checkpoint_version",
        "impacts_digest_hex",
        "impact_count",
        "limit",
        "offset",
    ),
    validate=_validate_impact_imports_claims,
)


def _validate_checkpoint_import_reconciliations_claims(
    claims: dict[str, Any],
) -> None:
    # The collection takes no filters: the cursor binds only limit/offset.
    for field in ("limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Cursor family for
#: ``GET /v1/audit-events/checkpoint-import-reconciliations``.
CHECKPOINT_IMPORT_RECONCILIATIONS_CURSOR = CursorKind(
    version=CHECKPOINT_IMPORT_RECONCILIATIONS_CURSOR_VERSION,
    claim_fields=("limit", "offset"),
    validate=_validate_checkpoint_import_reconciliations_claims,
)


def _validate_claims_claims(claims: dict[str, Any]) -> None:
    # The exact-match filters are either absent (null) or non-empty strings;
    # matching is case- and whitespace-sensitive, so the raw value is bound.
    _optional_nonempty_str(claims, "content_id")
    _optional_nonempty_str(claims, "actor_id")
    _optional_nonempty_str(claims, "claim_type")
    # The digest filter is absent (null) or the strict 64 lowercase hex
    # spelling; an uppercase or trimmed spelling is a different value and is
    # never normalized.
    digest = claims["payload_digest_hex"]
    if digest is not None and (
        not isinstance(digest, str) or not _HEX64_LOWER.fullmatch(digest)
    ):
        raise InvalidCursorError("cursor payload_digest_hex is invalid")
    for field in ("limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Cursor family for ``GET /v1/claims``.
CLAIMS_CURSOR = CursorKind(
    version=CLAIMS_CURSOR_VERSION,
    claim_fields=(
        "content_id",
        "actor_id",
        "claim_type",
        "payload_digest_hex",
        "limit",
        "offset",
    ),
    validate=_validate_claims_claims,
)


def _validate_evidence_bundles_claims(claims: dict[str, Any]) -> None:
    # The exact-match filters are either absent (null) or non-empty strings;
    # matching is case- and whitespace-sensitive, so the raw value is bound.
    _optional_nonempty_str(claims, "claim_id")
    _optional_nonempty_str(claims, "evidence_type")
    _optional_nonempty_str(claims, "media_type")
    # The digest filter is absent (null) or the strict 64 lowercase hex
    # spelling; an uppercase or trimmed spelling is a different value and is
    # never normalized.
    digest = claims["digest_hex"]
    if digest is not None and (
        not isinstance(digest, str) or not _HEX64_LOWER.fullmatch(digest)
    ):
        raise InvalidCursorError("cursor digest_hex is invalid")
    for field in ("limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Cursor family for ``GET /v1/evidence-bundles``.
EVIDENCE_BUNDLES_CURSOR = CursorKind(
    version=EVIDENCE_BUNDLES_CURSOR_VERSION,
    claim_fields=(
        "claim_id",
        "evidence_type",
        "media_type",
        "digest_hex",
        "limit",
        "offset",
    ),
    validate=_validate_evidence_bundles_claims,
)


def _validate_authentication_key_rotations_claims(claims: dict[str, Any]) -> None:
    # The collection belongs to exactly one subject: a non-empty actor_id
    # bound claim. Matching is case- and whitespace-sensitive, so the raw
    # path value is bound.
    if not isinstance(claims["actor_id"], str) or not claims["actor_id"]:
        raise InvalidCursorError("cursor actor_id is invalid")
    for field in ("limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Cursor family for
#: ``GET /v1/actors/{actor_id}/authentication-key-rotations``.
AUTHENTICATION_KEY_ROTATIONS_CURSOR = CursorKind(
    version=AUTHENTICATION_KEY_ROTATIONS_CURSOR_VERSION,
    claim_fields=("actor_id", "limit", "offset"),
    validate=_validate_authentication_key_rotations_claims,
)


def _validate_attestation_access_grants_claims(claims: dict[str, Any]) -> None:
    # The page belongs to exactly one proof and one authenticated caller:
    # non-empty attestation_id and actor_id bound claims. Matching is case-
    # and whitespace-sensitive, so the raw path/header values are bound.
    if not isinstance(claims["attestation_id"], str) or not claims[
        "attestation_id"
    ]:
        raise InvalidCursorError("cursor attestation_id is invalid")
    if not isinstance(claims["actor_id"], str) or not claims["actor_id"]:
        raise InvalidCursorError("cursor actor_id is invalid")
    for field in ("limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Cursor family for
#: ``GET /v1/attestations/{attestation_id}/access-grants``.
ATTESTATION_ACCESS_GRANTS_CURSOR = CursorKind(
    version=ATTESTATION_ACCESS_GRANTS_CURSOR_VERSION,
    claim_fields=("attestation_id", "actor_id", "limit", "offset"),
    validate=_validate_attestation_access_grants_claims,
)


def _validate_content_export_jobs_claims(claims: dict[str, Any]) -> None:
    # The exact-match filters are either absent (null) or non-empty strings;
    # matching is case- and whitespace-sensitive, so the raw value is bound.
    _optional_nonempty_str(claims, "content_id")
    _optional_nonempty_str(claims, "request_id")
    # The status filter is absent (null, meaning unfiltered) or one of the
    # four existing lifecycle literals; no other spelling is valid.
    status = claims["status"]
    if status is not None and status not in (
        "pending",
        "running",
        "succeeded",
        "failed",
    ):
        raise InvalidCursorError("cursor status is invalid")
    # The time bounds are absent (null) or canonical RFC 3339 UTC strings.
    for field in ("from", "to"):
        value = claims[field]
        if value is not None and (
            not isinstance(value, str) or parse_rfc3339_utc(value) is None
        ):
            raise InvalidCursorError(f"cursor {field} is invalid")
    for field in ("limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Cursor family for ``GET /v1/content-export-jobs``.
CONTENT_EXPORT_JOBS_CURSOR = CursorKind(
    version=CONTENT_EXPORT_JOBS_CURSOR_VERSION,
    claim_fields=(
        "content_id",
        "request_id",
        "status",
        "from",
        "to",
        "limit",
        "offset",
    ),
    validate=_validate_content_export_jobs_claims,
)


def _validate_claim_supersession_lineage_claims(claims: dict[str, Any]) -> None:
    # The page belongs to exactly one traversal: a non-empty origin claim id
    # bound claim. Matching is case- and whitespace-sensitive, so the raw
    # path value is bound.
    if not isinstance(claims["claim_id"], str) or not claims["claim_id"]:
        raise InvalidCursorError("cursor claim_id is invalid")
    if claims["direction"] not in ("newer", "older"):
        raise InvalidCursorError("cursor direction is invalid")
    for field in ("max_depth", "min_depth", "limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["max_depth"] <= 32):
        raise InvalidCursorError("cursor max_depth is out of range")
    if not (1 <= claims["min_depth"] <= claims["max_depth"]):
        raise InvalidCursorError("cursor min_depth is out of range")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    # An offset of 0 is never issued (a first page carries no cursor); a
    # negative offset is structurally invalid. No tight upper bound is
    # imposed: a wide graph can legitimately page past any small constant,
    # and a client cannot forge a large value without the server secret.
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Cursor family for
#: ``GET /v1/claims/{claim_id}/supersession-lineage``.
CLAIM_SUPERSESSION_LINEAGE_CURSOR = CursorKind(
    version=CLAIM_SUPERSESSION_LINEAGE_CURSOR_VERSION,
    claim_fields=(
        "claim_id",
        "direction",
        "max_depth",
        "min_depth",
        "limit",
        "offset",
    ),
    validate=_validate_claim_supersession_lineage_claims,
)


def _validate_trust_policies_claims(claims: dict[str, Any]) -> None:
    # The exact-match subject filter is either absent (null, meaning
    # unfiltered) or a non-empty string; matching is case- and
    # whitespace-sensitive, so the raw value is bound.
    _optional_nonempty_str(claims, "actor_id")
    for field in ("limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Cursor family for ``GET /v1/trust-policies``.
TRUST_POLICIES_CURSOR = CursorKind(
    version=TRUST_POLICIES_CURSOR_VERSION,
    claim_fields=("actor_id", "limit", "offset"),
    validate=_validate_trust_policies_claims,
)


def _validate_actors_claims(claims: dict[str, Any]) -> None:
    # The exact-match filters are either absent (null, meaning unfiltered) or
    # non-empty strings; matching is case- and whitespace-sensitive, so the
    # raw value is bound.
    _optional_nonempty_str(claims, "actor_id")
    _optional_nonempty_str(claims, "name")
    _optional_nonempty_str(claims, "actor_type")
    for field in ("limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Cursor family for ``GET /v1/actors``.
ACTORS_CURSOR = CursorKind(
    version=ACTORS_CURSOR_VERSION,
    claim_fields=("actor_id", "name", "actor_type", "limit", "offset"),
    validate=_validate_actors_claims,
)


def _validate_contents_claims(claims: dict[str, Any]) -> None:
    # The exact-match filters are either absent (null, meaning unfiltered) or
    # non-empty strings; matching is case- and whitespace-sensitive, so the
    # raw value is bound.
    _optional_nonempty_str(claims, "actor_id")
    _optional_nonempty_str(claims, "digest_algorithm")
    _optional_nonempty_str(claims, "media_type")
    # The digest filter is absent (null) or the strict 64 lowercase hex
    # spelling; an uppercase or trimmed spelling is a different value and is
    # never normalized.
    digest = claims["digest_hex"]
    if digest is not None and (
        not isinstance(digest, str) or not _HEX64_LOWER.fullmatch(digest)
    ):
        raise InvalidCursorError("cursor digest_hex is invalid")
    for field in ("limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Cursor family for ``GET /v1/contents``.
CONTENTS_CURSOR = CursorKind(
    version=CONTENTS_CURSOR_VERSION,
    claim_fields=(
        "actor_id",
        "digest_algorithm",
        "digest_hex",
        "media_type",
        "limit",
        "offset",
    ),
    validate=_validate_contents_claims,
)


def _validate_attestation_revocations_claims(claims: dict[str, Any]) -> None:
    # The exact-match filters are either absent (null) or non-empty strings;
    # matching is case- and whitespace-sensitive, so the raw value is bound.
    _optional_nonempty_str(claims, "attestation_id")
    _optional_nonempty_str(claims, "revoker_actor_id")
    _optional_nonempty_str(claims, "reason")
    # The time bounds are absent (null) or canonical RFC 3339 UTC strings.
    for field in ("from", "to"):
        value = claims[field]
        if value is not None and (
            not isinstance(value, str) or parse_rfc3339_utc(value) is None
        ):
            raise InvalidCursorError(f"cursor {field} is invalid")
    for field in ("limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Cursor family for ``GET /v1/attestation-revocations``.
ATTESTATION_REVOCATIONS_CURSOR = CursorKind(
    version=ATTESTATION_REVOCATIONS_CURSOR_VERSION,
    claim_fields=(
        "attestation_id",
        "revoker_actor_id",
        "reason",
        "from",
        "to",
        "limit",
        "offset",
    ),
    validate=_validate_attestation_revocations_claims,
)


def _validate_content_relations_claims(claims: dict[str, Any]) -> None:
    # The exact-match filters are either absent (null) or non-empty strings;
    # matching is case- and whitespace-sensitive, so the raw value is bound.
    _optional_nonempty_str(claims, "relation_id")
    _optional_nonempty_str(claims, "content_id")
    _optional_nonempty_str(claims, "parent_content_id")
    # The type filter is absent (null) or one of the two README relation
    # literals; no other spelling is valid.
    relation_type = claims["relation_type"]
    if relation_type is not None and relation_type not in (
        "version_of",
        "derived_from",
    ):
        raise InvalidCursorError("cursor relation_type is invalid")
    # The time bounds are absent (null) or canonical RFC 3339 UTC strings.
    for field in ("from", "to"):
        value = claims[field]
        if value is not None and (
            not isinstance(value, str) or parse_rfc3339_utc(value) is None
        ):
            raise InvalidCursorError(f"cursor {field} is invalid")
    for field in ("limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Cursor family for ``GET /v1/content-relations``.
CONTENT_RELATIONS_CURSOR = CursorKind(
    version=CONTENT_RELATIONS_CURSOR_VERSION,
    claim_fields=(
        "relation_id",
        "content_id",
        "parent_content_id",
        "relation_type",
        "from",
        "to",
        "limit",
        "offset",
    ),
    validate=_validate_content_relations_claims,
)


def _validate_content_coverage_search_claims(claims: dict[str, Any]) -> None:
    # The exact-match filters are either absent (null, meaning unfiltered) or
    # non-empty strings; matching is case- and whitespace-sensitive, so the
    # raw value is bound.
    _optional_nonempty_str(claims, "actor_id")
    _optional_nonempty_str(claims, "media_type")
    # The status filter is absent (null) or one of the three existing
    # coverage literals; no other spelling is valid.
    coverage_status = claims["coverage_status"]
    if coverage_status is not None and coverage_status not in (
        "uncovered",
        "partial",
        "covered",
    ):
        raise InvalidCursorError("cursor coverage_status is invalid")
    for field in ("limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Cursor family for ``GET /v1/content-coverage-search``.
CONTENT_COVERAGE_SEARCH_CURSOR = CursorKind(
    version=CONTENT_COVERAGE_SEARCH_CURSOR_VERSION,
    claim_fields=(
        "actor_id",
        "media_type",
        "coverage_status",
        "limit",
        "offset",
    ),
    validate=_validate_content_coverage_search_claims,
)


def _validate_revocation_impacts_claims(claims: dict[str, Any]) -> None:
    # The exact-match filters are either absent (null) or non-empty strings;
    # matching is case- and whitespace-sensitive, so the raw value is bound.
    _optional_nonempty_str(claims, "attestation_id")
    _optional_nonempty_str(claims, "revoker_actor_id")
    _optional_nonempty_str(claims, "reason")
    # The time bounds are absent (null) or canonical RFC 3339 UTC strings.
    for field in ("from", "to"):
        value = claims[field]
        if value is not None and (
            not isinstance(value, str) or parse_rfc3339_utc(value) is None
        ):
            raise InvalidCursorError(f"cursor {field} is invalid")
    for field in ("limit", "offset"):
        if not _is_int(claims[field]):
            raise InvalidCursorError(f"cursor {field} must be an integer")
    if not (1 <= claims["limit"] <= 100):
        raise InvalidCursorError("cursor limit is out of range")
    if claims["offset"] < 1:
        raise InvalidCursorError("cursor offset must be a positive integer")


#: Cursor family for ``GET /v1/revocation-impacts``.
REVOCATION_IMPACTS_CURSOR = CursorKind(
    version=REVOCATION_IMPACTS_CURSOR_VERSION,
    claim_fields=(
        "attestation_id",
        "revoker_actor_id",
        "reason",
        "from",
        "to",
        "limit",
        "offset",
    ),
    validate=_validate_revocation_impacts_claims,
)


def encode_typed_cursor(
    secret: bytes, kind: CursorKind, claims: dict[str, Any]
) -> str:
    """Return an opaque, signed token for normalized pagination ``claims``."""
    payload = _b64encode(
        json.dumps(
            {field: claims[field] for field in kind.claim_fields},
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    signature = _b64encode(_sign(secret, kind.version, payload))
    return f"{kind.version}.{payload}.{signature}"


def decode_typed_cursor(
    secret: bytes, kind: CursorKind, token: str
) -> dict[str, Any]:
    """Verify and decode ``token``, returning its normalized claims dict.

    Raises :class:`InvalidCursorError` for an empty token, wrong number of
    segments, an unknown family/version marker, bad Base64/JSON, a missing-
    or wrongly-typed claim, an out-of-range offset/limit, or a signature
    that does not authenticate the payload.
    """
    if not isinstance(token, str) or not token:
        raise InvalidCursorError("cursor must not be empty")
    parts = token.split(".")
    if len(parts) != 3:
        raise InvalidCursorError("cursor has an invalid structure")
    version, payload, signature = parts
    if version != kind.version:
        raise InvalidCursorError("cursor uses an unknown or expired version")

    expected_signature = _b64encode(_sign(secret, version, payload))
    if not hmac.compare_digest(signature, expected_signature):
        raise InvalidCursorError("cursor signature is invalid")

    try:
        raw = _b64decode(payload)
        claims = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidCursorError("cursor payload is not valid JSON") from exc

    if not isinstance(claims, dict) or set(claims) != set(kind.claim_fields):
        raise InvalidCursorError("cursor payload has an invalid claim set")
    kind.validate(claims)
    return claims


def encode_cursor(secret: bytes, claims: dict[str, Any]) -> str:
    """Encode a lineage cursor (backward-compatible wrapper)."""
    return encode_typed_cursor(secret, LINEAGE_CURSOR, claims)


def decode_cursor(secret: bytes, token: str) -> dict[str, Any]:
    """Decode a lineage cursor (backward-compatible wrapper)."""
    return decode_typed_cursor(secret, LINEAGE_CURSOR, token)
