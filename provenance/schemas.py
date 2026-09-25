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

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from provenance.canonical import canonical_json_bytes
from provenance.ed25519 import PUBLIC_KEY_LENGTH, SIGNATURE_LENGTH
from provenance.models import SUPPORTED_DIGEST_ALGORITHMS
from provenance.time_utils import parse_rfc3339_utc

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


class ClaimPageResponse(BaseModel):
    """A cursor-paginated page of claim public views (never the payload)."""

    items: list[ClaimResponse]
    #: Total number of claims after filtering, independent of pagination.
    count: int
    #: Opaque server cursor for the next page, or null on the final page.
    next_cursor: str | None = None


class ClaimSupersessionCreate(BaseModel):
    # A supersession carries exactly its declared fields; undeclared fields
    # are rejected rather than silently discarded.
    model_config = ConfigDict(extra="forbid")

    #: The earlier existing claim being superseded.
    superseded_claim_id: str = Field(..., min_length=1, max_length=80)
    #: The newer existing claim that replaces it.
    replacement_claim_id: str = Field(..., min_length=1, max_length=80)
    #: Non-empty rationale; surrounding whitespace is trimmed.
    reason: str = Field(..., min_length=1, max_length=4096)

    @field_validator("superseded_claim_id")
    @classmethod
    def _superseded_claim_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "superseded_claim_id")

    @field_validator("replacement_claim_id")
    @classmethod
    def _replacement_claim_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "replacement_claim_id")

    @field_validator("reason")
    @classmethod
    def _reason_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "reason")


class ClaimSupersessionResponse(BaseModel):
    """Public supersession view: the two endpoint ids, reason, and timestamp."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    superseded_claim_id: str
    replacement_claim_id: str
    reason: str
    created_at: datetime


class ClaimSupersessionListResponse(BaseModel):
    items: list[ClaimSupersessionResponse]
    count: int


class ClaimSupersessionLineageItem(ClaimResponse):
    """A claim reached by supersession traversal: the full public view plus depth.

    ``depth`` is the shortest number of supersession edges from the
    traversal origin; the origin claim itself is never included. The raw
    payload is never stored and so is never part of this view.
    """

    depth: int


class ClaimSupersessionLineageResponse(BaseModel):
    items: list[ClaimSupersessionLineageItem]
    #: Total number of items after the minimum-depth filter, independent of
    #: pagination; not merely the size of the returned page.
    count: int
    #: Opaque server cursor for the next page, or null when the final page
    #: has been returned.
    next_cursor: str | None = None


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


class ContentExportJobCreate(BaseModel):
    """A request to register one asynchronous content export job.

    Exactly two members: an existing ``content_id`` and a non-empty
    client-supplied ``request_id`` idempotency key. Undeclared fields are
    rejected rather than silently discarded.
    """

    model_config = ConfigDict(extra="forbid")

    content_id: str = Field(..., min_length=1, max_length=80)
    request_id: str = Field(..., min_length=1, max_length=255)

    @field_validator("content_id")
    @classmethod
    def _content_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "content_id")

    @field_validator("request_id")
    @classmethod
    def _request_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "request_id")


class ContentExportJobResponse(BaseModel):
    """Public view of one content export job and its lifecycle state.

    Exactly the stable ``cxj_`` id, the content/request association, the
    current status, the UTC lifecycle timestamps, and the settled outcome:
    ``result`` is the existing content export snapshot on success (otherwise
    null) and ``error`` is the stable failure code on a failed run
    (otherwise null). A freshly created job is ``pending`` with
    ``started_at``/``finished_at``/``result``/``error`` all null.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    content_id: str
    request_id: str
    status: Literal["pending", "running", "succeeded", "failed"]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    result: ContentExportResponse | None
    error: str | None


class ContentExportJobPageResponse(BaseModel):
    """A cursor-paginated page of content export job public views.

    Each item is exactly the existing single-job public view (including its
    settled ``result`` snapshot, which itself carries only existing public
    views -- never raw content, claim payloads, or evidence bytes).
    """

    items: list[ContentExportJobResponse]
    #: Total number of jobs after filtering, independent of pagination.
    count: int
    #: Opaque server cursor for the next page, or null on the final page.
    next_cursor: str | None = None


class ContentExportJobSummaryResponse(BaseModel):
    """Read-only queue summary over all existing content export jobs.

    Exactly six members: the four per-status counts (covering every existing
    job, settled ones included), plus the stable id of the oldest ``pending``
    job and the whole seconds it has waited so far. On an empty queue every
    count is zero and both oldest-pending members are null. All numbers are
    integers.
    """

    pending: int
    running: int
    succeeded: int
    failed: int
    #: Stable id of the oldest pending job, or null when none is pending.
    oldest_pending_id: str | None
    #: Whole seconds the oldest pending job has waited, or null when none.
    oldest_pending_wait_seconds: int | None


class EvidenceBundleExchangeResponse(BaseModel):
    """An interoperability snapshot of one evidence bundle for external verifiers.

    Exactly four members: ``content`` (the content the bundle's directly
    associated claim asserts), ``claim`` (the claim the bundle is directly
    attached to), ``evidence_bundle`` (the bundle's existing full public
    view), and ``attestations`` (the existing attestations whose target is
    this exact evidence bundle, in stable creation order, each rendered with
    exactly the existing attestation detail fields). No lineage traversal is
    performed and no other content, claim, or bundle is included; revoked
    attestations are retained for historical auditability.
    """

    content: ContentResponse
    claim: ClaimResponse
    evidence_bundle: EvidenceBundleResponse
    attestations: list[AttestationResponse]


#: Fixed manifest format version emitted by this service.
EXCHANGE_MANIFEST_VERSION = "provenance-exchange-manifest-v1"
#: The sole digest algorithm used for exchange manifests.
EXCHANGE_MANIFEST_DIGEST_ALGORITHM = "sha256"


class EvidenceBundleExchangeManifestResponse(BaseModel):
    """A read-only integrity manifest over one exchange snapshot.

    Exactly four members: the fixed manifest ``manifest_version`` and
    ``digest_algorithm``, the referenced bundle id, and the SHA-256 hex
    digest of the bundle's canonical exchange snapshot. The manifest is
    derived purely from the existing snapshot: it introduces no resource,
    record, or audit event, and its value is stable for unchanged
    persisted state.
    """

    manifest_version: Literal[EXCHANGE_MANIFEST_VERSION]
    evidence_bundle_id: str
    digest_algorithm: Literal[EXCHANGE_MANIFEST_DIGEST_ALGORITHM]
    manifest_digest_hex: str


class EvidenceBundleExchangePackageResponse(BaseModel):
    """One read returning an exchange snapshot together with its manifest.

    Exactly two members: ``snapshot`` is the bundle's existing exchange
    snapshot view and ``manifest`` is its existing four-field integrity
    manifest. The manifest digest is computed over exactly the ``snapshot``
    member returned in this same response under the existing canonical
    rules; both derive from a single read-only snapshot read, so the package
    introduces no resource, record, or audit event.
    """

    snapshot: EvidenceBundleExchangeResponse
    manifest: EvidenceBundleExchangeManifestResponse


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


class ExchangeManifestVerificationSnapshotContent(ContentResponse):
    """The ``content`` member of a snapshot under offline verification.

    Exactly the existing public content view's fields: undeclared members
    (including any raw-material field) are rejected rather than ignored.
    """

    model_config = ConfigDict(extra="forbid")


class ExchangeManifestVerificationSnapshotClaim(ClaimResponse):
    """The ``claim`` member: exactly the existing public claim view's fields."""

    model_config = ConfigDict(extra="forbid")


class ExchangeManifestVerificationSnapshotEvidenceBundle(EvidenceBundleResponse):
    """The ``evidence_bundle`` member: exactly the existing public bundle view.

    ``metadata`` must be canonicalizable JSON with finite numbers, exactly as
    at bundle creation.
    """

    model_config = ConfigDict(extra="forbid")

    @field_validator("metadata")
    @classmethod
    def _metadata_canonicalizable(cls, v: dict[str, Any]) -> dict[str, Any]:
        try:
            canonical_json_bytes(v)
        except (TypeError, ValueError):
            # Non-finite numbers (NaN/Infinity) have no canonical JSON form.
            raise ValueError(
                "metadata must be a JSON object with finite values"
            ) from None
        return v


class ExchangeManifestVerificationSnapshotAttestation(AttestationResponse):
    """An ``attestations`` entry: exactly the existing public attestation view."""

    model_config = ConfigDict(extra="forbid")


class ExchangeManifestVerificationSnapshot(BaseModel):
    """The exchange snapshot under verification: exactly the four members."""

    model_config = ConfigDict(extra="forbid")

    content: ExchangeManifestVerificationSnapshotContent
    claim: ExchangeManifestVerificationSnapshotClaim
    evidence_bundle: ExchangeManifestVerificationSnapshotEvidenceBundle
    attestations: list[ExchangeManifestVerificationSnapshotAttestation]


def _validate_exchange_snapshot_associations(
    snapshot: ExchangeManifestVerificationSnapshot, evidence_bundle_id: str
) -> None:
    """Validate the internal references of one exchange snapshot.

    The bundle id must match the enclosing manifest reference, the claim
    must match the bundle's direct association, the content must match the
    claim's, and every attestation must target this exact evidence bundle.
    These checks are purely structural and never resolve any id against
    local state, so they apply identically to offline verification and to
    controlled imports.
    """
    if snapshot.evidence_bundle.id != evidence_bundle_id:
        raise ValueError(
            "snapshot.evidence_bundle.id must equal evidence_bundle_id"
        )
    if snapshot.claim.id != snapshot.evidence_bundle.claim_id:
        raise ValueError(
            "snapshot.claim.id must equal snapshot.evidence_bundle.claim_id"
        )
    if snapshot.content.id != snapshot.claim.content_id:
        raise ValueError(
            "snapshot.content.id must equal snapshot.claim.content_id"
        )
    for attestation in snapshot.attestations:
        if (
            attestation.target_type != "evidence_bundle"
            or attestation.target_id != evidence_bundle_id
        ):
            raise ValueError(
                "every snapshot attestation must target this evidence bundle"
            )


class ExchangeManifestVerificationCreate(BaseModel):
    """An offline exchange-manifest verification request.

    Exactly five members: the fixed ``manifest_version`` and
    ``digest_algorithm``, the referenced bundle id, the claimed
    ``manifest_digest_hex``, and the exchange ``snapshot`` the manifest
    commits to. Verification is pure: it reads and writes no server state,
    so the bundle id is an opaque reference, never a lookup key.
    """

    model_config = ConfigDict(extra="forbid")

    manifest_version: Literal[EXCHANGE_MANIFEST_VERSION]
    evidence_bundle_id: str = Field(..., min_length=1, max_length=80)
    digest_algorithm: Literal[EXCHANGE_MANIFEST_DIGEST_ALGORITHM]
    manifest_digest_hex: str
    snapshot: ExchangeManifestVerificationSnapshot

    @field_validator("evidence_bundle_id")
    @classmethod
    def _evidence_bundle_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "evidence_bundle_id")

    @field_validator("manifest_digest_hex")
    @classmethod
    def _manifest_digest_is_sha256_hex(cls, v: str) -> str:
        # No normalization: the claimed digest must already be exactly 64
        # lowercase hexadecimal characters.
        if not _HEX64.fullmatch(v):
            raise ValueError(
                "manifest_digest_hex must be exactly 64 lowercase"
                " hexadecimal characters"
            )
        return v

    @model_validator(mode="after")
    def _snapshot_associations_consistent(self):
        _validate_exchange_snapshot_associations(
            self.snapshot, self.evidence_bundle_id
        )
        return self


class ExchangeManifestVerificationResponse(BaseModel):
    """The offline verification verdict.

    ``computed_digest_hex`` is present only on a mismatch, so a matching
    manifest renders exactly ``{"valid": true}``.
    """

    valid: bool
    #: The digest computed from the submitted snapshot under the manifest
    #: canonical rules; omitted when it matches the claimed digest.
    computed_digest_hex: str | None = None


#: The sole digest algorithm accepted for offline content export verification.
CONTENT_EXPORT_DIGEST_ALGORITHM = "sha256"


class ContentExportVerificationContent(ContentResponse):
    """The ``content`` member of a content export snapshot under verification.

    Exactly the existing public content view's fields: undeclared members
    (including any raw-material field such as content bytes) are rejected
    rather than ignored.
    """

    model_config = ConfigDict(extra="forbid")


class ContentExportVerificationEvidenceBundle(EvidenceBundleResponse):
    """An evidence bundle nested in a content export snapshot under verification.

    Exactly the existing public bundle view. ``metadata`` must be
    canonicalizable JSON with finite numbers, exactly as at bundle creation.
    """

    model_config = ConfigDict(extra="forbid")

    @field_validator("metadata")
    @classmethod
    def _metadata_canonicalizable(cls, v: dict[str, Any]) -> dict[str, Any]:
        try:
            canonical_json_bytes(v)
        except (TypeError, ValueError):
            # Non-finite numbers (NaN/Infinity) have no canonical JSON form.
            raise ValueError(
                "metadata must be a JSON object with finite values"
            ) from None
        return v


class ContentExportVerificationClaim(ClaimExportItem):
    """A claim in a content export snapshot under verification.

    The full public claim view plus its ``evidence_bundles`` array, each
    bundle exactly the existing strict public bundle view. Undeclared
    members (including a raw claim ``payload`` or a signature field) are
    rejected on the claim and on every bundle.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_bundles: list[ContentExportVerificationEvidenceBundle]


class ContentExportVerificationSnapshot(BaseModel):
    """The content export snapshot under verification: exactly two members.

    ``content`` is the full public content view and ``claims`` is the array
    of directly asserting claims (an empty array is a legal snapshot), each
    with its evidence bundles. Undeclared members are rejected rather than
    ignored, so no raw content, payload, evidence, or signature field can
    enter the snapshot.
    """

    model_config = ConfigDict(extra="forbid")

    content: ContentExportVerificationContent
    claims: list[ContentExportVerificationClaim]


def _validate_content_export_snapshot_associations(
    snapshot: ContentExportVerificationSnapshot,
) -> None:
    """Validate the internal references of one content export snapshot.

    Every claim must directly assert the snapshot's exact content, and each
    claim's evidence bundle must be attached to that exact claim. The checks
    are purely structural and never resolve any id against local state, so
    they apply identically whether any referenced resource exists locally.
    """
    for claim in snapshot.claims:
        if claim.content_id != snapshot.content.id:
            raise ValueError(
                "every snapshot claim must directly assert snapshot.content.id"
            )
        for bundle in claim.evidence_bundles:
            if bundle.claim_id != claim.id:
                raise ValueError(
                    "every snapshot evidence bundle must attach to its"
                    " enclosing claim"
                )


class ContentExportVerificationCreate(BaseModel):
    """An offline content export snapshot verification request.

    Exactly three members: the public content export ``snapshot`` (only the
    direct assertions of its content and their evidence bundles), the fixed
    ``digest_algorithm`` (``sha256``), and the claimed ``digest_hex`` (exactly
    64 lowercase hexadecimal characters). Verification is pure: it reads and
    writes no server state, so every identifier is an opaque reference.
    """

    model_config = ConfigDict(extra="forbid")

    snapshot: ContentExportVerificationSnapshot
    digest_algorithm: Literal[CONTENT_EXPORT_DIGEST_ALGORITHM]
    digest_hex: str

    @field_validator("digest_hex")
    @classmethod
    def _digest_is_sha256_hex(cls, v: str) -> str:
        # No normalization: the claimed digest must already be exactly 64
        # lowercase hexadecimal characters.
        if not _HEX64.fullmatch(v):
            raise ValueError(
                "digest_hex must be exactly 64 lowercase hexadecimal characters"
            )
        return v

    @model_validator(mode="after")
    def _snapshot_associations_consistent(self):
        _validate_content_export_snapshot_associations(self.snapshot)
        return self


class ContentExportVerificationResponse(BaseModel):
    """The offline content export verification verdict.

    ``computed_digest_hex`` is present only on a mismatch, so a matching
    digest renders exactly ``{"valid": true}``.
    """

    valid: bool
    #: The digest recomputed from the submitted snapshot; omitted on match.
    computed_digest_hex: str | None = None


class EvidenceBundleExchangeImportManifest(BaseModel):
    """The manifest half of a received exchange package under controlled import.

    Exactly the existing four-field exchange manifest structure: the fixed
    ``manifest_version`` and ``digest_algorithm``, the referenced bundle id,
    and the claimed snapshot digest. Undeclared members are rejected rather
    than ignored, exactly as on the read and verification routes.
    """

    model_config = ConfigDict(extra="forbid")

    manifest_version: Literal[EXCHANGE_MANIFEST_VERSION]
    evidence_bundle_id: str = Field(..., min_length=1, max_length=80)
    digest_algorithm: Literal[EXCHANGE_MANIFEST_DIGEST_ALGORITHM]
    manifest_digest_hex: str

    @field_validator("evidence_bundle_id")
    @classmethod
    def _evidence_bundle_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "evidence_bundle_id")

    @field_validator("manifest_digest_hex")
    @classmethod
    def _manifest_digest_is_sha256_hex(cls, v: str) -> str:
        # No normalization: the claimed digest must already be exactly 64
        # lowercase hexadecimal characters.
        if not _HEX64.fullmatch(v):
            raise ValueError(
                "manifest_digest_hex must be exactly 64 lowercase"
                " hexadecimal characters"
            )
        return v


class EvidenceBundleExchangeImportCreate(BaseModel):
    """A controlled import of one already-received exchange package.

    Exactly two members: ``manifest`` carries the existing four-field
    exchange manifest and ``snapshot`` carries the existing four-member
    exchange snapshot. Both are validated exactly as on the stateless
    verification route (fixed version/algorithm, strict digest spelling,
    existing public-view fields only, and consistent internal
    associations); the digest match itself is enforced at the route, over
    the raw received JSON so root member and array order participate.

    Verification is a pure function of the request body and never resolves
    any identifier against local resources: the package need not describe
    any locally stored content, claim, bundle, or attestation. The request
    has no field capable of carrying a raw signature, claim payload,
    content bytes, or evidence bytes.
    """

    model_config = ConfigDict(extra="forbid")

    manifest: EvidenceBundleExchangeImportManifest
    snapshot: ExchangeManifestVerificationSnapshot

    @model_validator(mode="after")
    def _snapshot_associations_consistent(self):
        _validate_exchange_snapshot_associations(
            self.snapshot, self.manifest.evidence_bundle_id
        )
        return self


class EvidenceBundleExchangeImportResponse(BaseModel):
    """The public immutable receipt for one registered exchange import.

    Exactly the stable ``eir_`` receipt id, the three receiving-identity
    fields (manifest version, evidence bundle id, manifest digest), and the
    UTC ``received_at`` instant. The snapshot itself is deliberately not
    part of the receipt: it is neither copied into the record nor echoed
    here.
    """

    id: str
    manifest_version: str
    evidence_bundle_id: str
    manifest_digest_hex: str
    received_at: datetime


class EvidenceBundleExchangeImportPageResponse(BaseModel):
    """A cursor-paginated page of exchange-import receipt public views."""

    items: list[EvidenceBundleExchangeImportResponse]
    #: Total number of receipts after filtering, independent of pagination.
    count: int
    #: Opaque server cursor for the next page, or null on the final page.
    next_cursor: str | None = None


class EvidenceBundleExchangeImportReconciliationResponse(BaseModel):
    """A read-only reconciliation of one exchange-import receipt with local state.

    Exactly four members: the receipt's ``import_id``, whether the receipt's
    ``evidence_bundle_id`` resolves to a local bundle (``local_available``),
    the bundle's current exchange manifest digest when it does
    (``local_manifest_digest_hex``, else null), and ``matches`` — true only
    when the local digest equals the receipt's ``manifest_digest_hex``
    character for character. The receipt's snapshot, raw signatures,
    payloads, and bytes are never echoed.
    """

    import_id: str
    local_available: bool
    local_manifest_digest_hex: str | None
    matches: bool


class EvidenceBundleExchangeImportReconciliationItem(BaseModel):
    """One receipt's current local reconciliation in a list page.

    Carries the existing single-receipt public view exactly (``id``,
    ``manifest_version``, ``evidence_bundle_id``, ``manifest_digest_hex``,
    ``received_at``) plus the three reconciliation fields computed against
    current local state: ``local_available`` is false,
    ``local_manifest_digest_hex`` null, and ``matches`` false when no local
    bundle carries the receipt's ``evidence_bundle_id``; when one does, the
    current exchange manifest digest is reported under the existing manifest
    rules and ``matches`` is true only on character-for-character equality
    with the receipt digest. The snapshot and every raw material are absent.
    """

    id: str
    manifest_version: str
    evidence_bundle_id: str
    manifest_digest_hex: str
    received_at: datetime
    local_available: bool
    local_manifest_digest_hex: str | None
    matches: bool


class EvidenceBundleExchangeImportReconciliationPageResponse(BaseModel):
    """A cursor-paginated page of receipt local reconciliations."""

    items: list[EvidenceBundleExchangeImportReconciliationItem]
    #: Total number of receipts, independent of pagination.
    count: int
    #: Opaque server cursor for the next page, or null on the final page.
    next_cursor: str | None = None


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


class AttestationAccessGrantPageResponse(BaseModel):
    """A cursor-paginated page of access-grant public views."""

    items: list[AttestationAccessGrantResponse]
    #: Total number of grants for the attestation, independent of pagination.
    count: int
    #: Opaque continuation token; null on the final (or past-the-end) page.
    next_cursor: str | None


class AttestationAccessGrantRevocationCreate(BaseModel):
    # A grant revocation carries exactly its declared fields; undeclared
    # fields are rejected rather than silently discarded.
    model_config = ConfigDict(extra="forbid")

    #: The existing access grant being revoked.
    grant_id: str = Field(..., min_length=1, max_length=80)
    #: Non-empty rationale; surrounding whitespace is trimmed.
    reason: str = Field(..., min_length=1, max_length=4096)

    @field_validator("grant_id")
    @classmethod
    def _grant_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "grant_id")

    @field_validator("reason")
    @classmethod
    def _reason_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "reason")


class AttestationAccessGrantRevocationResponse(BaseModel):
    """Public grant-revocation view: the grant, the revoker, reason, and time."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    grant_id: str
    revoker_actor_id: str
    reason: str
    created_at: datetime


class AttestationAccessGrantRevocationListResponse(BaseModel):
    items: list[AttestationAccessGrantRevocationResponse]
    count: int


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


#: Fixed checkpoint format version emitted by this service.
AUDIT_CHECKPOINT_VERSION = "provenance-audit-checkpoint-v1"
#: The sole digest algorithm used for audit-event checkpoints.
AUDIT_CHECKPOINT_DIGEST_ALGORITHM = "sha256"


class AuditEventCheckpointResponse(BaseModel):
    """A read-only integrity checkpoint over a filtered audit-event sequence.

    Exactly four members: the fixed ``checkpoint_version`` and
    ``digest_algorithm``, the number of events in the filtered sequence,
    and the SHA-256 hex digest of the canonical event array (each event
    reduced to ``event_type``/``resource_id``/UTC ``created_at``). The
    checkpoint is derived purely from the existing events: it introduces
    no resource, record, or audit event, and its value is stable for
    unchanged persisted state -- including the empty sequence.
    """

    checkpoint_version: Literal[AUDIT_CHECKPOINT_VERSION]
    digest_algorithm: Literal[AUDIT_CHECKPOINT_DIGEST_ALGORITHM]
    event_count: int
    events_digest_hex: str


class AuditEventCheckpointPackageResponse(BaseModel):
    """One read returning a filtered audit-event sequence with its checkpoint.

    Exactly two members: ``checkpoint`` is the existing four-field audit
    checkpoint for the effective filter and ``events`` lists, in stable
    creation order, the existing audit-event public views (each exactly
    ``event_type``/``resource_id``/UTC ``created_at``) matched by that same
    filter. The checkpoint digest is computed over exactly the ``events``
    array returned in this response under the existing checkpoint canonical
    rules; both derive from a single read-only state read, so they can never
    disagree. The package introduces no resource, record, or audit event; an
    empty match yields ``"events": []`` together with the digest of the empty
    array.
    """

    checkpoint: AuditEventCheckpointResponse
    events: list[AuditEventItem]


class AuditCheckpointVerificationEvent(BaseModel):
    """An event under checkpoint verification: exactly the three public fields.

    ``created_at`` is kept as the raw string so the digest commits to the
    timestamp exactly as spelled on the wire; it must be a strict RFC 3339
    UTC timestamp, exactly as the audit-event time filters require.
    """

    model_config = ConfigDict(extra="forbid")

    event_type: str = Field(..., min_length=1)
    resource_id: str = Field(..., min_length=1)
    created_at: str

    @field_validator("event_type")
    @classmethod
    def _event_type_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "event_type")

    @field_validator("resource_id")
    @classmethod
    def _resource_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "resource_id")

    @field_validator("created_at")
    @classmethod
    def _created_at_strict_rfc3339_utc(cls, v: str) -> str:
        if parse_rfc3339_utc(v) is None:
            raise ValueError(
                "created_at must be a strict RFC 3339 UTC timestamp"
            )
        return v


class AuditCheckpointVerificationCheckpoint(BaseModel):
    """The claimed checkpoint under stateless verification.

    Exactly the existing four-field audit checkpoint structure: the fixed
    ``checkpoint_version`` and ``digest_algorithm``, the claimed event
    count, and the claimed events digest. Undeclared members are rejected
    rather than ignored, exactly as on the checkpoint read route.
    """

    model_config = ConfigDict(extra="forbid")

    checkpoint_version: Literal[AUDIT_CHECKPOINT_VERSION]
    digest_algorithm: Literal[AUDIT_CHECKPOINT_DIGEST_ALGORITHM]
    #: Non-negative integer count the submitted event array must match.
    event_count: StrictInt = Field(..., ge=0)
    events_digest_hex: str

    @field_validator("events_digest_hex")
    @classmethod
    def _events_digest_is_sha256_hex(cls, v: str) -> str:
        # No normalization: the claimed digest must already be exactly 64
        # lowercase hexadecimal characters.
        if not _HEX64.fullmatch(v):
            raise ValueError(
                "events_digest_hex must be exactly 64 lowercase"
                " hexadecimal characters"
            )
        return v


class AuditCheckpointVerificationCreate(BaseModel):
    """A stateless audit-checkpoint verification request.

    Exactly two members: ``checkpoint`` carries the claimed four-field
    audit checkpoint and ``events`` carries the event sequence it commits
    to, with exactly ``checkpoint.event_count`` elements. Verification is
    pure: it reads and writes no server state, so no event or resource id
    is ever resolved against local state.
    """

    model_config = ConfigDict(extra="forbid")

    checkpoint: AuditCheckpointVerificationCheckpoint
    events: list[AuditCheckpointVerificationEvent]

    @model_validator(mode="after")
    def _events_match_claimed_count(self):
        if len(self.events) != self.checkpoint.event_count:
            raise ValueError(
                "events must contain exactly checkpoint.event_count elements"
            )
        return self


class AuditCheckpointVerificationResponse(BaseModel):
    """The stateless checkpoint verification verdict.

    ``computed_digest_hex`` is present only on a mismatch, so a matching
    checkpoint renders exactly ``{"valid": true}``.
    """

    valid: bool
    #: The digest computed from the submitted event array under the
    #: checkpoint canonical rules; omitted when it matches the claim.
    computed_digest_hex: str | None = None


class AuditCheckpointImportCreate(AuditCheckpointVerificationCreate):
    """A controlled import of one offline-verified audit checkpoint.

    Exactly the existing two-member verification structure: ``checkpoint``
    carries the claimed four-field audit checkpoint and ``events`` carries
    the event sequence it commits to, with exactly
    ``checkpoint.event_count`` elements. The digest match itself is
    enforced at the route, over the raw received JSON so array order and
    datetime spellings participate exactly as on the stateless
    verification route; a mismatch is a 422 and writes nothing.

    Registration is a pure function of the request body that never
    resolves any event or resource id against local state: the described
    events are not created, modified, or queried, and they need not exist
    locally.
    """


class AuditCheckpointImportResponse(BaseModel):
    """The public immutable receipt for one registered checkpoint import.

    Exactly the stable ``aci_`` receipt id, the three receiving-identity
    fields (checkpoint version, events digest, event count), and the UTC
    ``received_at`` instant. The event array itself is deliberately not
    part of the receipt: it is neither copied into the record nor echoed
    here.
    """

    id: str
    checkpoint_version: str
    events_digest_hex: str
    event_count: int
    received_at: datetime


class AuditCheckpointImportPageResponse(BaseModel):
    """A cursor-paginated page of checkpoint-import receipt public views."""

    items: list[AuditCheckpointImportResponse]
    #: Total number of receipts after filtering, independent of pagination.
    count: int
    #: Opaque server cursor for the next page, or null on the final page.
    next_cursor: str | None = None


class AuditCheckpointImportReconciliationResponse(BaseModel):
    """A read-only reconciliation of one checkpoint-import receipt with local state.

    Exactly three members: the receipt's ``import_id``, the four-field
    checkpoint computed over the current, unfiltered local audit-event
    sequence under the existing checkpoint rules
    (``local_checkpoint``), and ``matches`` -- true only when the receipt's
    ``checkpoint_version``, ``event_count``, and ``events_digest_hex`` all
    equal the corresponding local checkpoint fields. The imported event
    array is never read (it is not persisted) or echoed; no material beyond
    the receipt identity participates in the verdict.
    """

    import_id: str
    local_checkpoint: AuditEventCheckpointResponse
    matches: bool


class AuditCheckpointImportReconciliationItem(BaseModel):
    """One receipt's current local reconciliation in a list page.

    Carries the existing single-receipt public view exactly (``id``,
    ``checkpoint_version``, ``events_digest_hex``, ``event_count``,
    ``received_at``) plus ``local_checkpoint`` -- the four-field checkpoint
    computed over the complete, unfiltered local audit sequence under the
    existing checkpoint rules -- and ``matches``, true only when the
    receipt's ``checkpoint_version``, ``event_count``, and
    ``events_digest_hex`` all equal that local checkpoint. The imported
    event array is never echoed.
    """

    id: str
    checkpoint_version: str
    events_digest_hex: str
    event_count: int
    received_at: datetime
    local_checkpoint: AuditEventCheckpointResponse
    matches: bool


class AuditCheckpointImportReconciliationPageResponse(BaseModel):
    """A cursor-paginated page of receipt local reconciliations."""

    items: list[AuditCheckpointImportReconciliationItem]
    #: Total number of receipts, independent of pagination.
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


class ActorTrustPolicyCreate(BaseModel):
    # A policy carries exactly its declared fields; undeclared fields are
    # rejected rather than silently discarded.
    model_config = ConfigDict(extra="forbid")

    #: The subject the policy is registered for. The authenticated caller
    #: must be this same actor.
    actor_id: str = Field(..., min_length=1, max_length=255)
    #: Distinct qualified signers required for a ``trusted`` decision. A
    #: pure decimal integer only: booleans, strings, and fractional or
    #: exponent-notation numbers are rejected, never coerced.
    threshold: StrictInt = Field(..., ge=1, le=100)

    @field_validator("actor_id")
    @classmethod
    def _actor_id_nonempty(cls, v: str) -> str:
        return _required_nonempty(v, "actor_id")


class ActorTrustPolicyResponse(BaseModel):
    """Public policy view: subject, threshold, enabled flag, UTC timestamp."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    actor_id: str
    threshold: int
    #: Always true: policies are immutable and never deactivated.
    enabled: bool
    created_at: datetime


class ActorTrustPolicyPageResponse(BaseModel):
    """A cursor-paginated page of trust-policy public views."""

    items: list[ActorTrustPolicyResponse]
    #: Total number of policies after filtering, independent of pagination.
    count: int
    #: Opaque server cursor for the next page, or null on the final page.
    next_cursor: str | None = None


class TrustDecisionResponse(BaseModel):
    """Read-only authorization decision under the caller's current policy.

    Computed on demand from the caller's stored policy and the attestations
    already stored for the target: no resource or audit event is created.
    """

    target_type: Literal["claim", "evidence_bundle"]
    target_id: str
    #: The deciding policy; null when the caller has no policy.
    policy_id: str | None
    #: The policy's signer threshold; null when the caller has no policy.
    threshold: int | None
    #: Number of distinct signing actors with a verified, non-revoked
    #: attestation of the target; zero when no policy exists (the target is
    #: never looked up in that case).
    qualified_signer_count: int
    #: ``"trusted"`` only when a policy exists and its threshold is met.
    decision: Literal["trusted", "untrusted"]
    #: Why the decision came out: the threshold was met, the count fell
    #: below it, or the caller has no policy at all.
    reason: Literal["threshold_met", "below_threshold", "policy_missing"]


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


class AuthenticationKeyRotationPageResponse(BaseModel):
    """A cursor-paginated page of rotation public views for one subject."""

    items: list[AuthenticationKeyRotationResponse]
    #: Total number of the subject's rotations, independent of pagination.
    count: int
    #: Opaque server cursor for the next page, or null on the final page.
    next_cursor: str | None = None
