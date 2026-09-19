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


def _base64_exact_length(value: str, field: str, length: int) -> str:
    """Validate standard Base64 decoding to exactly ``length`` bytes.

    Returns the canonical re-encoding so equivalent-but-noncanonical input
    cannot create distinct identities.
    """
    try:
        raw = base64.b64decode(value.strip(), validate=True)
    except (binascii.Error, ValueError):
        raise ValueError(f"{field} must be valid Base64") from None
    if len(raw) != length:
        raise ValueError(f"{field} must decode to exactly {length} bytes")
    return base64.b64encode(raw).decode("ascii")


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
    # Undeclared fields are rejected: no field may smuggle raw evidence
    # bytes into the service, even transiently.
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


class EvidenceBundleListResponse(BaseModel):
    items: list[EvidenceBundleResponse]
    count: int


AttestationTargetType = Literal["claim", "evidence_bundle"]


class AttestationCreate(BaseModel):
    # Undeclared fields are rejected, consistent with evidence bundles.
    model_config = ConfigDict(extra="forbid")

    target_type: AttestationTargetType
    target_id: str = Field(..., min_length=1, max_length=80)
    signer_actor_id: str = Field(..., min_length=1, max_length=255)
    #: Base64-encoded Ed25519 public key; must decode to exactly 32 bytes.
    public_key: str = Field(..., min_length=1, max_length=128)
    #: Base64-encoded Ed25519 signature; must decode to exactly 64 bytes.
    signature: str = Field(..., min_length=1, max_length=256)

    @field_validator("target_id")
    @classmethod
    def _target_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "target_id")

    @field_validator("signer_actor_id")
    @classmethod
    def _signer_actor_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "signer_actor_id")

    @field_validator("public_key")
    @classmethod
    def _public_key_is_base64_32_bytes(cls, v: str) -> str:
        return _base64_exact_length(v, "public_key", PUBLIC_KEY_LENGTH)

    @field_validator("signature")
    @classmethod
    def _signature_is_base64_64_bytes(cls, v: str) -> str:
        return _base64_exact_length(v, "signature", SIGNATURE_LENGTH)


class AttestationResponse(BaseModel):
    """Public attestation view: target, signer, key, signature digest.

    The raw signature is never echoed; only its SHA-256 digest is exposed.
    """

    id: str
    target_type: str
    target_id: str
    signer_actor_id: str
    public_key: str
    signature_digest_algorithm: str
    signature_digest_hex: str
    verified: bool
    created_at: datetime


class AttestationListResponse(BaseModel):
    items: list[AttestationResponse]
    count: int
