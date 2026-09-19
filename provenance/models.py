"""ORM models for actors, content identities, and the audit trail.

Content bytes are intentionally not modeled: only the digest and metadata are
persisted, so raw content can never reach the database or logs through this
service.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from provenance.database import Base, UTCDateTime
from provenance.time_utils import utc_now

# Allowed content digest algorithms. Only SHA-256 is accepted in this version.
SUPPORTED_DIGEST_ALGORITHMS = frozenset({"sha256"})

# Audit event types.
EVENT_ACTOR_CREATED = "actor.created"
EVENT_CONTENT_CREATED = "content.created"
EVENT_CLAIM_CREATED = "claim.created"
EVENT_EVIDENCE_BUNDLE_CREATED = "evidence_bundle.created"
EVENT_ATTESTATION_CREATED = "attestation.created"

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
    """A verified Ed25519 attestation over an existing claim or bundle.

    Only attestations whose signature verified at creation time are
    persisted, so every stored row carries ``verified=True`` semantics. The
    raw signature is never stored: only its SHA-256 digest is kept, which is
    enough to recognize a repeat submission without retaining the signature
    itself. Attestations are append-only; there is deliberately no update or
    delete path.
    """

    __tablename__ = "attestations"
    __table_args__ = (
        UniqueConstraint(
            "target_type",
            "target_id",
            "signer_actor_id",
            "public_key_b64",
            "signature_digest_hex",
            name="uq_attestations_identity",
        ),
        Index(
            "ix_attestations_target_order",
            "target_type",
            "target_id",
            "created_at",
            "seq",
        ),
        Index("ix_attestations_signer_actor_id", "signer_actor_id"),
    )

    #: Monotonic insertion surrogate; the primary key for stable ordering.
    seq: Mapped[int] = mapped_column(
        _surrogate_key, primary_key=True, autoincrement=True
    )
    #: Server-generated stable resource identifier ("att_" + 64 hex chars).
    id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    #: "claim" or "evidence_bundle".
    target_type: Mapped[str] = mapped_column(String(32), nullable=False)
    #: Id of the attested claim or evidence bundle. Not a database-level
    #: foreign key: the referenced table depends on ``target_type`` and is
    #: enforced at the service layer.
    target_id: Mapped[str] = mapped_column(String(80), nullable=False)
    signer_actor_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("actors.id", ondelete="RESTRICT"), nullable=False
    )
    #: Canonical Base64 of the 32-byte Ed25519 public key.
    public_key_b64: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Digest algorithm of the stored signature digest ("sha256").
    signature_digest_algorithm: Mapped[str] = mapped_column(
        String(32), nullable=False
    )
    signature_digest_hex: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now
    )

    signer: Mapped[Actor] = relationship()


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
