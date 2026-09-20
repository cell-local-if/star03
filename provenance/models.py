"""ORM models for actors, content identities, and the audit trail.

Content bytes are intentionally not modeled: only the digest and metadata are
persisted, so raw content can never reach the database or logs through this
service.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    ForeignKey,
    Index,
    Integer,
    JSON,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from provenance.database import Base, UTCDateTime
from provenance.time_utils import utc_now

# Allowed content digest algorithms. Only SHA-256 is accepted in this version.
SUPPORTED_DIGEST_ALGORITHMS = frozenset({"sha256"})

# Allowed content relation (lineage edge) types.
RELATION_VERSION_OF = "version_of"
RELATION_DERIVED_FROM = "derived_from"
SUPPORTED_RELATION_TYPES = frozenset({RELATION_VERSION_OF, RELATION_DERIVED_FROM})

# Audit event types.
EVENT_ACTOR_CREATED = "actor.created"
EVENT_CONTENT_CREATED = "content.created"
EVENT_CLAIM_CREATED = "claim.created"
EVENT_EVIDENCE_BUNDLE_CREATED = "evidence_bundle.created"
EVENT_ATTESTATION_CREATED = "attestation.created"
EVENT_ATTESTATION_REVOKED = "attestation.revoked"
EVENT_ATTESTATION_ACCESS_GRANTED = "attestation.access_granted"
EVENT_CONTENT_RELATION_CREATED = "content_relation.created"
EVENT_AUTHENTICATION_KEY_ROTATED = "authentication_key.rotated"
EVENT_AUTHENTICATION_KEY_RETIRED = "authentication_key.retired"

# Renders as INTEGER on SQLite (required for AUTOINCREMENT) and BIGINT elsewhere.
_surrogate_key = BigInteger().with_variant(Integer, "sqlite")


class Actor(Base):
    """A provenance subject (person, organization, device, software, ...)."""

    __tablename__ = "actors"

    #: Client-supplied stable identifier.
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now
    )

    contents: Mapped[list["Content"]] = relationship(
        back_populates="actor", passive_deletes=True
    )


class Content(Base):
    """A content identity: a digest plus metadata attributed to one actor."""

    __tablename__ = "contents"
    __table_args__ = (
        UniqueConstraint(
            "digest_algorithm",
            "digest_hex",
            name="uq_contents_algorithm_digest",
        ),
        Index("ix_contents_actor_id", "actor_id"),
        Index("ix_contents_created_order", "created_at", "seq"),
    )

    #: Monotonic insertion surrogate; the primary key for stable ordering.
    seq: Mapped[int] = mapped_column(
        _surrogate_key, primary_key=True, autoincrement=True
    )
    #: Server-generated stable resource identifier ("cnt_" + 64 hex chars).
    id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    digest_algorithm: Mapped[str] = mapped_column(String(32), nullable=False)
    digest_hex: Mapped[str] = mapped_column(String(128), nullable=False)
    media_type: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    actor_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("actors.id", ondelete="RESTRICT"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now
    )

    actor: Mapped[Actor] = relationship(back_populates="contents")


class Claim(Base):
    """An immutable provenance claim about a content identity by an actor.

    The raw payload is never persisted: only its deterministic canonical-JSON
    digest is stored, so a claim commits to its payload without retaining it.
    Claims are append-only; there is deliberately no update or delete path.
    """

    __tablename__ = "claims"
    __table_args__ = (
        UniqueConstraint(
            "content_id",
            "actor_id",
            "claim_type",
            "payload_digest_hex",
            name="uq_claims_identity",
        ),
        Index("ix_claims_content_order", "content_id", "created_at", "seq"),
        Index("ix_claims_actor_id", "actor_id"),
    )

    #: Monotonic insertion surrogate; the primary key for stable ordering.
    seq: Mapped[int] = mapped_column(
        _surrogate_key, primary_key=True, autoincrement=True
    )
    #: Server-generated stable resource identifier ("clm_" + 64 hex chars).
    id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    content_id: Mapped[str] = mapped_column(
        String(80), ForeignKey("contents.id", ondelete="RESTRICT"), nullable=False
    )
    #: The claiming actor; need not be the content's registering actor.
    actor_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("actors.id", ondelete="RESTRICT"), nullable=False
    )
    claim_type: Mapped[str] = mapped_column(String(128), nullable=False)
    #: Digest algorithm of the canonical payload digest ("sha256").
    payload_digest_algorithm: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_digest_hex: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now
    )

    content: Mapped[Content] = relationship()
    actor: Mapped[Actor] = relationship()


class EvidenceBundle(Base):
    """A verifiable evidence bundle attached to an immutable claim.

    The evidence itself is never received or persisted: only its digest
    algorithm/value, media type, and a metadata object are stored, so the
    bundle can be independently verified against externally held bytes
    without those bytes ever reaching this service. Bundles are
    append-only; there is deliberately no update or delete path.
    """

    __tablename__ = "evidence_bundles"
    __table_args__ = (
        UniqueConstraint(
            "claim_id",
            "evidence_type",
            "digest_algorithm",
            "digest_hex",
            name="uq_evidence_bundles_identity",
        ),
        Index(
            "ix_evidence_bundles_claim_order", "claim_id", "created_at", "seq"
        ),
    )

    #: Monotonic insertion surrogate; the primary key for stable ordering.
    seq: Mapped[int] = mapped_column(
        _surrogate_key, primary_key=True, autoincrement=True
    )
    #: Server-generated stable resource identifier ("evb_" + 64 hex chars).
    id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    claim_id: Mapped[str] = mapped_column(
        String(80), ForeignKey("claims.id", ondelete="RESTRICT"), nullable=False
    )
    evidence_type: Mapped[str] = mapped_column(String(128), nullable=False)
    digest_algorithm: Mapped[str] = mapped_column(String(32), nullable=False)
    digest_hex: Mapped[str] = mapped_column(String(128), nullable=False)
    media_type: Mapped[str] = mapped_column(String(255), nullable=False)
    #: Structured client metadata; always a JSON object. The Python attribute
    #: avoids the declarative ``metadata`` reserved name; the column stays
    #: ``metadata`` on the wire and in storage.
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSON, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now
    )

    claim: Mapped[Claim] = relationship()


class Attestation(Base):
    """A verified Ed25519 attestation of a claim or evidence bundle.

    Only records that passed signature verification are ever written. The
    raw signature is never persisted: only its SHA-256 digest is stored,
    alongside the 32-byte Ed25519 public key and the association to the
    attested target and the existing signing actor. Attestations are
    append-only; there is deliberately no update or delete path.
    """

    __tablename__ = "attestations"
    __table_args__ = (
        UniqueConstraint(
            "target_type",
            "target_id",
            "signer_actor_id",
            "public_key",
            "signature_digest_hex",
            name="uq_attestations_identity",
        ),
        Index(
            "ix_attestations_target",
            "target_type",
            "target_id",
            "created_at",
            "seq",
        ),
        Index("ix_attestations_created_order", "created_at", "seq"),
    )

    #: Monotonic insertion surrogate; the primary key for stable ordering.
    seq: Mapped[int] = mapped_column(
        _surrogate_key, primary_key=True, autoincrement=True
    )
    #: Server-generated stable resource identifier ("att_" + 64 hex chars).
    id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    #: "claim" or "evidence_bundle"; target_id is polymorphic and has no FK.
    target_type: Mapped[str] = mapped_column(String(32), nullable=False)
    target_id: Mapped[str] = mapped_column(String(80), nullable=False)
    signer_actor_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("actors.id", ondelete="RESTRICT"), nullable=False
    )
    #: The 32 raw Ed25519 public-key bytes (Base64 on the wire).
    public_key: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)
    #: Digest algorithm of the signature digest ("sha256").
    signature_digest_algorithm: Mapped[str] = mapped_column(
        String(32), nullable=False
    )
    #: SHA-256 hex digest of the submitted signature; the signature itself
    #: is never stored or echoed back.
    signature_digest_hex: Mapped[str] = mapped_column(
        String(128), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now
    )

    signer_actor: Mapped[Actor] = relationship()


class AttestationRevocation(Base):
    """An immutable revocation record for an existing attestation.

    A revocation is an append-only statement that an attestation is no
    longer relied upon. It neither mutates nor deletes the attestation:
    the original proof and its ``attestation.created`` audit relationship
    are preserved. Revocations are append-only; there is deliberately no
    update or delete path.
    """

    __tablename__ = "attestation_revocations"
    __table_args__ = (
        UniqueConstraint(
            "attestation_id",
            "revoker_actor_id",
            "reason",
            name="uq_attestation_revocations_identity",
        ),
        Index(
            "ix_attestation_revocations_attestation_order",
            "attestation_id",
            "created_at",
            "seq",
        ),
        Index("ix_attestation_revocations_created_order", "created_at", "seq"),
    )

    #: Monotonic insertion surrogate; the primary key for stable ordering.
    seq: Mapped[int] = mapped_column(
        _surrogate_key, primary_key=True, autoincrement=True
    )
    #: Server-generated stable resource identifier ("rev_" + 64 hex chars).
    id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    attestation_id: Mapped[str] = mapped_column(
        String(80), ForeignKey("attestations.id", ondelete="RESTRICT"),
        nullable=False,
    )
    #: The actor recording the revocation; need not be the signer.
    revoker_actor_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("actors.id", ondelete="RESTRICT"), nullable=False
    )
    #: Non-empty, human/audit rationale; stored verbatim (trimmed) text.
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now
    )

    attestation: Mapped[Attestation] = relationship()
    revoker_actor: Mapped[Actor] = relationship()


class AttestationAccessGrant(Base):
    """An immutable read-only proof-access grant for one attestation.

    A grant authorizes a grantee actor -- distinct from the attestation's
    signer -- to read the attestation through the protected endpoint. Only
    the attestation's ``signer_actor_id`` may create a grant. Grants are
    append-only and immutable: there is deliberately no update or delete
    path. The ``(attestation_id, grantee_actor_id)`` pair is unique, so a
    retried submission for the same pair returns the original record.
    """

    __tablename__ = "attestation_access_grants"
    __table_args__ = (
        UniqueConstraint(
            "attestation_id",
            "grantee_actor_id",
            name="uq_attestation_access_grants_identity",
        ),
        Index(
            "ix_attestation_access_grants_attestation_order",
            "attestation_id",
            "created_at",
            "seq",
        ),
        Index(
            "ix_attestation_access_grants_created_order", "created_at", "seq"
        ),
    )

    #: Monotonic insertion surrogate; the primary key for stable ordering.
    seq: Mapped[int] = mapped_column(
        _surrogate_key, primary_key=True, autoincrement=True
    )
    #: Server-generated stable resource identifier ("aag_" + 64 hex chars).
    id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    attestation_id: Mapped[str] = mapped_column(
        String(80), ForeignKey("attestations.id", ondelete="RESTRICT"),
        nullable=False,
    )
    #: The actor granted read access; must already exist and need not be the
    #: signer. Only the attestation's signer may create the grant.
    grantee_actor_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("actors.id", ondelete="RESTRICT"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now
    )

    attestation: Mapped[Attestation] = relationship()
    grantee_actor: Mapped[Actor] = relationship()


class AuthenticationKeyRotation(Base):
    """An authentication public-key rotation record for one actor.

    Only the 32-byte Ed25519 public key is ever stored -- never a private
    key or any signature. An active record's key authenticates the actor on
    the protected routes alongside the keys of its non-revoked attestations.
    Retiring a record flips ``active`` and stamps ``retired_at`` in place --
    the record and its audit relationship are preserved, and the key stops
    authenticating immediately. There is deliberately no delete path.
    """

    __tablename__ = "authentication_key_rotations"
    __table_args__ = (
        UniqueConstraint(
            "actor_id",
            "public_key",
            name="uq_authentication_key_rotations_identity",
        ),
        Index(
            "ix_authentication_key_rotations_actor_order",
            "actor_id",
            "created_at",
            "seq",
        ),
    )

    #: Monotonic insertion surrogate; the primary key for stable ordering.
    seq: Mapped[int] = mapped_column(
        _surrogate_key, primary_key=True, autoincrement=True
    )
    #: Server-generated stable resource identifier ("akr_" + 64 hex chars).
    id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    #: The subject actor whose authentication key set holds this key.
    actor_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("actors.id", ondelete="RESTRICT"), nullable=False
    )
    #: The 32 raw Ed25519 public-key bytes (Base64 on the wire).
    public_key: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)
    #: True while the key authenticates; False once retired.
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now
    )
    #: UTC retirement instant; null while the record is active.
    retired_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime, nullable=True
    )

    actor: Mapped[Actor] = relationship()


class ContentRelation(Base):
    """An immutable lineage edge between two existing content identities.

    ``content_id`` is the newer version or derived content;
    ``parent_content_id`` is its direct source. Edges are append-only; there
    is deliberately no update or delete path.
    """

    __tablename__ = "content_relations"
    __table_args__ = (
        UniqueConstraint(
            "content_id",
            "parent_content_id",
            "relation_type",
            name="uq_content_relations_identity",
        ),
        Index(
            "ix_content_relations_content_order",
            "content_id",
            "created_at",
            "seq",
        ),
        Index(
            "ix_content_relations_parent_order",
            "parent_content_id",
            "created_at",
            "seq",
        ),
    )

    #: Monotonic insertion surrogate; the primary key for stable ordering.
    seq: Mapped[int] = mapped_column(
        _surrogate_key, primary_key=True, autoincrement=True
    )
    #: Server-generated stable resource identifier ("rel_" + 64 hex chars).
    id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    #: The newer version or derived content (out-edge endpoint).
    content_id: Mapped[str] = mapped_column(
        String(80), ForeignKey("contents.id", ondelete="RESTRICT"), nullable=False
    )
    #: The direct source content (in-edge endpoint).
    parent_content_id: Mapped[str] = mapped_column(
        String(80), ForeignKey("contents.id", ondelete="RESTRICT"), nullable=False
    )
    #: "version_of" or "derived_from".
    relation_type: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now
    )

    content: Mapped[Content] = relationship(foreign_keys=[content_id])
    parent_content: Mapped[Content] = relationship(
        foreign_keys=[parent_content_id]
    )


class AuditEvent(Base):
    """Append-only audit record. One row per successful resource creation."""

    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_events_created_order", "created_at", "seq"),)

    seq: Mapped[int] = mapped_column(
        _surrogate_key, primary_key=True, autoincrement=True
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now
    )
