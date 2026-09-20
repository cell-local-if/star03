"""Request and response schemas for the versioned JSON API.

There is deliberately no field capable of carrying content bytes: only digest
and metadata are accepted, so raw content cannot be persisted or logged.
"""

from __future__ import annotations

import base64
import binascii
import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from provenance.canonical import canonical_json_bytes
from provenance.ed25519 import PUBLIC_KEY_LENGTH, SIGNATURE_LENGTH
from provenance.models import SUPPORTED_DIGEST_ALGORITHMS

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _required_nonempty(value: str, field: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field} must not be empty")
    return cleaned


def _decode_base64(value: Any, field: str, expected_length: int) -> bytes:
    """Strictly decode standard Base64 (RFC 4648) and enforce a byte length.

    The URL/filename-safe alphabet, missing/incorrect padding, non-string
    input, and any decoded length other than ``expected_length`` are
    rejected.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a Base64-encoded string")
    try:
        # validate=True rejects non-alphabet characters (including '-'/'_')
        # and bad padding; then re-encoding enforces canonical padding.
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError(
            f"{field} must be standard Base64 (RFC 4648) with correct padding"
        ) from None
    if base64.b64encode(raw).decode("ascii") != value:
        raise ValueError(f"{field} must use canonical standard Base64 padding")
    if len(raw) != expected_length:
        raise ValueError(
            f"{field} must decode to exactly {expected_length} bytes"
            f" (got {len(raw)})"
        )
    return raw


class ActorCreate(BaseModel):
    id: str = Field(..., min_length=1, max_length=255)
    name: str = Field(..., min_length=1, max_length=4096)
    type: str = Field(..., min_length=1, max_length=64)

    @field_validator("id")
    @classmethod
    def _id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "id")

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "name")

    @field_validator("type")
    @classmethod
    def _type_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "type")


class ActorResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    type: str
    created_at: datetime


class ContentCreate(BaseModel):
    digest_algorithm: str = Field(..., min_length=1, max_length=32)
    digest_hex: str = Field(..., min_length=1, max_length=128)
    media_type: str = Field(..., min_length=1, max_length=255)
    title: str | None = Field(default=None, max_length=4096)
    actor_id: str = Field(..., min_length=1, max_length=255)

    @field_validator("digest_algorithm")
    @classmethod
    def _algorithm_supported(cls, v: str) -> str:
        normalized = v.strip().lower()
        if normalized not in SUPPORTED_DIGEST_ALGORITHMS:
            supported = ", ".join(sorted(SUPPORTED_DIGEST_ALGORITHMS))
            raise ValueError(f"unsupported digest algorithm; supported: {supported}")
        return normalized

    @field_validator("digest_hex")
    @classmethod
    def _digest_is_sha256_hex(cls, v: str) -> str:
        # Normalize case so the same digest cannot be registered twice via
        # different casing; then enforce exactly 64 lowercase hex chars.
        normalized = v.strip().lower()
        if not _HEX64.fullmatch(normalized):
            raise ValueError("digest_hex must be exactly 64 hexadecimal characters")
        return normalized

    @field_validator("media_type")
    @classmethod
    def _media_type_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "media_type")

    @field_validator("actor_id")
    @classmethod
    def _actor_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "actor_id")

    @field_validator("title")
    @classmethod
    def _title_optional(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = v.strip()
        return cleaned or None


class ContentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    digest_algorithm: str
    digest_hex: str
    media_type: str
    title: str | None
    actor_id: str
    created_at: datetime


class ContentListResponse(BaseModel):
    items: list[ContentResponse]
    count: int


class ContentLineageItem(ContentResponse):
    """A content reached by lineage traversal: the full public view plus depth.

    ``depth`` is the shortest number of lineage edges from the traversal
    origin; the origin itself is never included.
    """

    depth: int


class ContentLineageResponse(BaseModel):
    items: list[ContentLineageItem]
    #: Total number of items after relation/depth filtering, independent of
    #: pagination; not merely the size of the returned page.
    count: int
    #: Opaque server cursor for the next page, or null when the final page
    #: has been returned.
    next_cursor: str | None = None


class ClaimCreate(BaseModel):
    content_id: str = Field(..., min_length=1, max_length=80)
    actor_id: str = Field(..., min_length=1, max_length=255)
    claim_type: str = Field(..., min_length=1, max_length=128)
    #: Must be a JSON object; arrays, scalars, and null are rejected.
    payload: dict[str, Any]

    @field_validator("content_id")
    @classmethod
    def _content_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "content_id")

    @field_validator("actor_id")
    @classmethod
    def _actor_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "actor_id")

    @field_validator("claim_type")
    @classmethod
    def _claim_type_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "claim_type")

    @field_validator("payload")
    @classmethod
    def _payload_canonicalizable(cls, v: dict[str, Any]) -> dict[str, Any]:
        try:
            canonical_json_bytes(v)
        except (TypeError, ValueError):
            # Non-finite numbers (NaN/Infinity) have no canonical JSON form.
            raise ValueError(
                "payload must be a JSON object with finite numbers"
            ) from None
        return v


class ClaimResponse(BaseModel):
    """Public claim view: associations, digest, and timestamps — no payload."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    content_id: str
    actor_id: str
    claim_type: str
    payload_digest_algorithm: str
    payload_digest_hex: str
    created_at: datetime


class ClaimListResponse(BaseModel):
    items: list[ClaimResponse]
    count: int


class EvidenceBundleCreate(BaseModel):
    # Evidence bytes must never reach the service: any field not declared
    # here (e.g. "data" or "evidence") is a client error, not silently
    # dropped, so raw-byte smuggling is rejected at the boundary.
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(..., min_length=1, max_length=80)
    evidence_type: str = Field(..., min_length=1, max_length=128)
    digest_algorithm: str = Field(..., min_length=1, max_length=32)
    digest_hex: str = Field(..., min_length=1, max_length=128)
    media_type: str = Field(..., min_length=1, max_length=255)
    #: Must be a JSON object; arrays, scalars, and null are rejected.
    metadata: dict[str, Any]

    @field_validator("claim_id")
    @classmethod
    def _claim_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "claim_id")

    @field_validator("evidence_type")
    @classmethod
    def _evidence_type_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "evidence_type")

    @field_validator("digest_algorithm")
    @classmethod
    def _algorithm_supported(cls, v: str) -> str:
        normalized = v.strip().lower()
        if normalized not in SUPPORTED_DIGEST_ALGORITHMS:
            supported = ", ".join(sorted(SUPPORTED_DIGEST_ALGORITHMS))
            raise ValueError(f"unsupported digest algorithm; supported: {supported}")
        return normalized

    @field_validator("digest_hex")
    @classmethod
    def _digest_is_sha256_hex(cls, v: str) -> str:
        # Normalize case so the same digest cannot create two bundles via
        # different casing; then enforce exactly 64 lowercase hex chars.
        normalized = v.strip().lower()
        if not _HEX64.fullmatch(normalized):
            raise ValueError("digest_hex must be exactly 64 hexadecimal characters")
        return normalized

    @field_validator("media_type")
    @classmethod
    def _media_type_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "media_type")

    @field_validator("metadata")
    @classmethod
    def _metadata_json_object(cls, v: dict[str, Any]) -> dict[str, Any]:
        try:
            canonical_json_bytes(v)
        except (TypeError, ValueError):
            # Non-finite numbers (NaN/Infinity) have no valid JSON form.
            raise ValueError(
                "metadata must be a JSON object with finite values"
            ) from None
        return v


class EvidenceBundleResponse(BaseModel):
    """Public evidence bundle view: associations, digest, metadata, time."""

    id: str
    claim_id: str
    evidence_type: str
    digest_algorithm: str
    digest_hex: str
    media_type: str
    metadata: dict[str, Any]
    created_at: datetime


class EvidenceBundleImportCreate(BaseModel):
    # The batch carries exactly an ``items`` array; undeclared fields are
    # rejected rather than silently discarded. Each item is validated exactly
    # as a single bundle create, so raw-byte smuggling and non-object
    # metadata are rejected at the boundary for every element.
    model_config = ConfigDict(extra="forbid")

    #: 1 to 100 bundle requests per atomic import.
    items: list[EvidenceBundleCreate] = Field(..., min_length=1, max_length=100)


class EvidenceBundleImportResponse(BaseModel):
    """Batch result: one public view per unique identity.

    ``items`` follows the first-occurrence order of each unique
    (claim, evidence type, digest) identity in the request; in-batch
    duplicates never appear.
    """

    items: list[EvidenceBundleResponse]
    #: Number of unique identities in the batch (== len(items)).
    count: int


class EvidenceBundleListResponse(BaseModel):
    items: list[EvidenceBundleResponse]
    count: int


class EvidenceBundlePageResponse(BaseModel):
    """A cursor-paginated page of evidence bundle public views."""

    items: list[EvidenceBundleResponse]
    #: Total number of items after filtering, independent of pagination.
    count: int
    #: Opaque server cursor for the next page, or null on the final page.
    next_cursor: str | None = None


class ClaimExportItem(ClaimResponse):
    """A claim in a content export: the full public view plus its evidence.

    ``evidence_bundles`` lists the claim's evidence bundles in their stable
    creation order, each rendered as the existing public bundle view.
    """

    evidence_bundles: list[EvidenceBundleResponse]


class ContentExportResponse(BaseModel):
    """A transferable provenance-evidence snapshot of one content.

    Exactly ``content`` (the full public content view) and ``claims`` (the
    claims directly asserting this content, in stable creation order, each
    with its evidence bundles). No lineage traversal and no raw bytes.
    """

    content: ContentResponse
    claims: list[ClaimExportItem]


class ContentRelationCreate(BaseModel):
    content_id: str = Field(..., min_length=1, max_length=80)
    parent_content_id: str = Field(..., min_length=1, max_length=80)
    #: "version_of" (a new version of the source) or "derived_from" (derived
    #: from the source); any other type is a validation error.
    relation_type: Literal["version_of", "derived_from"]

    @field_validator("content_id")
    @classmethod
    def _content_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "content_id")

    @field_validator("parent_content_id")
    @classmethod
    def _parent_content_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "parent_content_id")


class ContentRelationResponse(BaseModel):
    """Public relation view: the two endpoints, the type, and the timestamp."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    content_id: str
    parent_content_id: str
    relation_type: Literal["version_of", "derived_from"]
    created_at: datetime


class ContentRelationListResponse(BaseModel):
    items: list[ContentRelationResponse]
    count: int


class AttestationCreate(BaseModel):
    # Attestation requests carry exactly the declared verification material;
    # undeclared fields are rejected rather than silently discarded.
    model_config = ConfigDict(extra="forbid")

    #: Polymorphic target: an existing claim or evidence bundle.
    target_type: Literal["claim", "evidence_bundle"]
    target_id: str = Field(..., min_length=1, max_length=80)
    #: An already-registered actor making the attestation.
    signer_actor_id: str = Field(..., min_length=1, max_length=255)
    #: Base64 Ed25519 public key; must decode to exactly 32 bytes.
    public_key: bytes
    #: Base64 Ed25519 signature; must decode to exactly 64 bytes. The raw
    #: signature is verified and discarded -- it is never persisted.
    signature: bytes

    @field_validator("target_id")
    @classmethod
    def _target_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "target_id")

    @field_validator("signer_actor_id")
    @classmethod
    def _signer_actor_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "signer_actor_id")

    @field_validator("public_key", mode="before")
    @classmethod
    def _public_key_is_32_bytes(cls, v: Any) -> bytes:
        return _decode_base64(v, "public_key", PUBLIC_KEY_LENGTH)

    @field_validator("signature", mode="before")
    @classmethod
    def _signature_is_64_bytes(cls, v: Any) -> bytes:
        return _decode_base64(v, "signature", SIGNATURE_LENGTH)


class AttestationResponse(BaseModel):
    """Public attestation view: no raw signature, only its SHA-256 digest."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    target_type: Literal["claim", "evidence_bundle"]
    target_id: str
    signer_actor_id: str
    #: Base64 of the 32-byte Ed25519 public key.
    public_key: str
    signature_digest_algorithm: str
    signature_digest_hex: str
    #: Always true for stored attestations: only verified rows are written.
    verified: bool
    created_at: datetime


class AttestationListResponse(BaseModel):
    items: list[AttestationResponse]
    count: int


class AttestationRevocationCreate(BaseModel):
    # A revocation carries exactly its declared fields; undeclared fields
    # are rejected rather than silently discarded.
    model_config = ConfigDict(extra="forbid")

    #: The existing attestation being revoked.
    attestation_id: str = Field(..., min_length=1, max_length=80)
    #: An already-registered actor recording the revocation.
    revoker_actor_id: str = Field(..., min_length=1, max_length=255)
    #: Non-empty rationale; surrounding whitespace is trimmed.
    reason: str = Field(..., min_length=1, max_length=4096)

    @field_validator("attestation_id")
    @classmethod
    def _attestation_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "attestation_id")

    @field_validator("revoker_actor_id")
    @classmethod
    def _revoker_actor_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "revoker_actor_id")

    @field_validator("reason")
    @classmethod
    def _reason_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "reason")


class AttestationRevocationResponse(BaseModel):
    """Public revocation view: associations, reason text, and timestamp."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    attestation_id: str
    revoker_actor_id: str
    reason: str
    created_at: datetime


class AttestationRevocationListResponse(BaseModel):
    items: list[AttestationRevocationResponse]
    count: int


class AttestationAccessGrantCreate(BaseModel):
    # A grant carries exactly its declared fields; undeclared fields are
    # rejected rather than silently discarded.
    model_config = ConfigDict(extra="forbid")

    #: The existing attestation whose read access is being granted.
    attestation_id: str = Field(..., min_length=1, max_length=80)
    #: An already-registered actor receiving read-only access.
    grantee_actor_id: str = Field(..., min_length=1, max_length=255)

    @field_validator("attestation_id")
    @classmethod
    def _attestation_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "attestation_id")

    @field_validator("grantee_actor_id")
    @classmethod
    def _grantee_actor_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "grantee_actor_id")


class AttestationAccessGrantResponse(BaseModel):
    """Public grant view: the proof, the grantee, and the UTC timestamp."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    attestation_id: str
    grantee_actor_id: str
    created_at: datetime


class AuditEventItem(BaseModel):
    """Public audit-event view: type, resource, and UTC timestamp only."""

    model_config = ConfigDict(from_attributes=True)

    event_type: str
    resource_id: str
    created_at: datetime


class AuditEventPageResponse(BaseModel):
    """A cursor-paginated page of audit-event public views."""

    items: list[AuditEventItem]
    #: Total number of events after filtering, independent of pagination.
    count: int
    #: Opaque server cursor for the next page, or null on the final page.
    next_cursor: str | None = None


class TrustEvaluationResponse(BaseModel):
    """Read-only trust assessment of a claim or evidence bundle.

    Computed on demand from the attestations already stored for the target:
    no resource or audit event is created.
    """

    target_type: Literal["claim", "evidence_bundle"]
    target_id: str
    min_signers: int
    #: Number of distinct signing actors with a verified attestation of the
    #: target; multiple attestations by the same actor count once.
    qualified_signer_count: int
    #: ``"trusted"`` when ``qualified_signer_count >= min_signers``.
    decision: Literal["trusted", "untrusted"]


class AuthenticationKeyRotationCreate(BaseModel):
    # A rotation carries exactly its declared verification material. There
    # is deliberately no field capable of carrying a private key or a raw
    # signature: undeclared fields are rejected rather than silently dropped.
    model_config = ConfigDict(extra="forbid")

    #: The subject introducing the key. The authenticated caller must be
    #: this same actor.
    actor_id: str = Field(..., min_length=1, max_length=255)
    #: Base64 Ed25519 public key; must decode to exactly 32 bytes.
    new_public_key: bytes

    @field_validator("actor_id")
    @classmethod
    def _actor_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "actor_id")

    @field_validator("new_public_key", mode="before")
    @classmethod
    def _new_public_key_is_32_bytes(cls, v: Any) -> bytes:
        return _decode_base64(v, "new_public_key", PUBLIC_KEY_LENGTH)


class AuthenticationKeyRotationResponse(BaseModel):
    """Public rotation view: subject, public key, lifecycle flags, and times."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    #: The subject who owns the key.
    actor_id: str
    #: Base64 of the 32-byte Ed25519 public key.
    new_public_key: str
    active: bool
    created_at: datetime
    #: UTC retirement time, or null while the key is active.
    retired_at: datetime | None
