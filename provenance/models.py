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
    text,
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
EVENT_CLAIM_SUPERSEDED = "claim.superseded"
EVENT_EVIDENCE_BUNDLE_CREATED = "evidence_bundle.created"
EVENT_ATTESTATION_CREATED = "attestation.created"
EVENT_ATTESTATION_REVOKED = "attestation.revoked"
EVENT_ATTESTATION_ACCESS_GRANTED = "attestation.access_granted"
EVENT_ATTESTATION_ACCESS_GRANT_REVOKED = "attestation.access_grant_revoked"
EVENT_CONTENT_RELATION_CREATED = "content_relation.created"
EVENT_AUTHENTICATION_KEY_ROTATED = "authentication_key.rotated"
EVENT_AUTHENTICATION_KEY_RETIRED = "authentication_key.retired"
EVENT_EVIDENCE_BUNDLE_EXCHANGE_IMPORTED = "evidence_bundle.exchange_imported"
EVENT_AUDIT_CHECKPOINT_IMPORTED = "audit.checkpoint_imported"
EVENT_REVOCATION_IMPACT_IMPORTED = "revocation_impact.imported"
EVENT_REVOCATION_IMPACT_EXCHANGE_IMPORTED = (
    "revocation_impact.exchange_imported"
)
EVENT_CONTENT_EXPORT_JOB_CREATED = "content_export_job.created"
EVENT_CONTENT_EXPORT_JOB_RUN = "content_export_job.run"
EVENT_ACTOR_TRUST_POLICY_CREATED = "actor_trust_policy.created"

# Content export job lifecycle states. A job is created ``pending``; a run
# atomically claims it into ``running`` and then settles it as
# ``succeeded`` or ``failed``. There is deliberately no path back to an
# earlier state: a non-pending job can never be claimed or re-run.
CONTENT_EXPORT_JOB_PENDING = "pending"
CONTENT_EXPORT_JOB_RUNNING = "running"
CONTENT_EXPORT_JOB_SUCCEEDED = "succeeded"
CONTENT_EXPORT_JOB_FAILED = "failed"
CONTENT_EXPORT_JOB_STATES = frozenset(
    {
        CONTENT_EXPORT_JOB_PENDING,
        CONTENT_EXPORT_JOB_RUNNING,
        CONTENT_EXPORT_JOB_SUCCEEDED,
        CONTENT_EXPORT_JOB_FAILED,
    }
)
# Stable error recorded on a failed export run.
CONTENT_EXPORT_JOB_FAILED_ERROR = "content_export_failed"

# Renders as INTEGER on SQLite (required for AUTOINCREMENT) and BIGINT elsewhere.
_surrogate_key = BigInteger().with_variant(Integer, "sqlite")


class Actor(Base):
    """A provenance subject (person, organization, device, software, ...)."""

    __tablename__ = "actors"
    __table_args__ = (
        Index("ix_actors_created_order", "created_at", "display_seq"),
    )

    #: Client-supplied stable identifier.
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now
    )
    #: Explicit persistence order, dense from 1 in original insertion order.
    #: Backfilled from SQLite rowid for pre-existing actors and stamped by an
    #: AFTER INSERT trigger for new ones; breaks same-created_at ties in SQL.
    #: Never part of the public view.
    display_seq: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0")
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


class ClaimSupersession(Base):
    """An immutable statement that one claim supersedes another.

    Both endpoints must be existing claims about the same content; the
    superseded claim and the replacement must differ, and a supersession may
    never close a cycle in the supersession graph. Neither endpoint claim is
    ever mutated or deleted: the original claim, its ``claim.created`` audit
    relationship, and its evidence remain intact. Supersessions are
    append-only; there is deliberately no update or delete path.

    The ``(superseded_claim_id, replacement_claim_id, reason)`` triple is
    unique, so a retried submission of the same three fields returns the
    original record; a different reason is an independent archival record.
    """

    __tablename__ = "claim_supersessions"
    __table_args__ = (
        UniqueConstraint(
            "superseded_claim_id",
            "replacement_claim_id",
            "reason",
            name="uq_claim_supersessions_identity",
        ),
        Index(
            "ix_claim_supersessions_superseded_order",
            "superseded_claim_id",
            "created_at",
            "seq",
        ),
        Index(
            "ix_claim_supersessions_replacement_order",
            "replacement_claim_id",
            "created_at",
            "seq",
        ),
        Index("ix_claim_supersessions_created_order", "created_at", "seq"),
    )

    #: Monotonic insertion surrogate; the primary key for stable ordering.
    seq: Mapped[int] = mapped_column(
        _surrogate_key, primary_key=True, autoincrement=True
    )
    #: Server-generated stable resource identifier ("csp_" + 64 hex chars).
    id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    #: The earlier claim being superseded (in-edge endpoint).
    superseded_claim_id: Mapped[str] = mapped_column(
        String(80), ForeignKey("claims.id", ondelete="RESTRICT"), nullable=False
    )
    #: The newer claim that replaces it (out-edge endpoint).
    replacement_claim_id: Mapped[str] = mapped_column(
        String(80), ForeignKey("claims.id", ondelete="RESTRICT"), nullable=False
    )
    #: Non-empty, human/audit rationale; stored verbatim (trimmed) text.
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now
    )

    superseded_claim: Mapped[Claim] = relationship(
        foreign_keys=[superseded_claim_id]
    )
    replacement_claim: Mapped[Claim] = relationship(
        foreign_keys=[replacement_claim_id]
    )


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


class AttestationAccessGrantRevocation(Base):
    """An immutable revocation of one existing read-only proof-access grant.

    A revocation is an append-only statement by the attestation's signer that
    a previously issued :class:`AttestationAccessGrant` no longer authorizes
    its grantee to read the proof through the protected endpoint. It neither
    mutates nor deletes the grant or the attestation: the original grant, its
    ``attestation.access_granted`` audit relationship, and the attestation
    itself are preserved. The signer retains their own read access, and other
    unrevoked grants held by the same grantee are unaffected. Revocations are
    append-only and immutable: there is deliberately no update or delete
    path.

    Only the signer of the grant's attestation may create a revocation. The
    ``(grant_id, revoker_actor_id, reason)`` triple is unique, so a retried
    submission for the same three fields returns the original record; a
    different reason is an independent archival record. The record stores no
    private key, raw signature, claim payload, content, or evidence bytes.
    """

    __tablename__ = "attestation_access_grant_revocations"
    __table_args__ = (
        UniqueConstraint(
            "grant_id",
            "revoker_actor_id",
            "reason",
            name="uq_attestation_access_grant_revocations_identity",
        ),
        Index(
            "ix_attestation_access_grant_revocations_grant_order",
            "grant_id",
            "created_at",
            "seq",
        ),
        Index(
            "ix_attestation_access_grant_revocations_created_order",
            "created_at",
            "seq",
        ),
    )

    #: Monotonic insertion surrogate; the primary key for stable ordering.
    seq: Mapped[int] = mapped_column(
        _surrogate_key, primary_key=True, autoincrement=True
    )
    #: Server-generated stable resource identifier ("agr_" + 64 hex chars).
    id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    grant_id: Mapped[str] = mapped_column(
        String(80),
        ForeignKey("attestation_access_grants.id", ondelete="RESTRICT"),
        nullable=False,
    )
    #: The signer recording the revocation; must be the signer of the
    #: attestation the grant concerns.
    revoker_actor_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("actors.id", ondelete="RESTRICT"), nullable=False
    )
    #: Non-empty, human/audit rationale; stored verbatim (trimmed) text.
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now
    )

    grant: Mapped[AttestationAccessGrant] = relationship()
    revoker_actor: Mapped[Actor] = relationship()


class AuthenticationKeyRotation(Base):
    """A subject's rotated authentication public key.

    A rotation introduces a new 32-byte Ed25519 public key that immediately
    joins the subject's non-revoked authentication key set for the protected
    routes -- no attestation is required, and the key does not replace the
    public keys still carried by the subject's existing non-revoked
    attestations. Only the public key is ever received: there is no field
    capable of carrying a private key or a raw signature.

    A rotation is retired, never deleted: ``active`` flips to false and a
    UTC ``retired_at`` is stamped, preserving the original record and its
    ``authentication_key.rotated`` audit relationship. A retired key stops
    authenticating immediately. The ``(actor_id, public_key)`` pair is
    unique, so a retried submission for the same subject and key returns the
    original record (whether still active or already retired).
    """

    __tablename__ = "authentication_key_rotations"
    __table_args__ = (
        UniqueConstraint(
            "actor_id",
            "public_key",
            name="uq_authentication_key_rotations_identity",
        ),
        Index("ix_authentication_key_rotations_actor_order", "actor_id", "seq"),
        Index(
            "ix_authentication_key_rotations_active_actor",
            "actor_id",
            "active",
        ),
    )

    #: Monotonic insertion surrogate; the primary key for stable ordering.
    seq: Mapped[int] = mapped_column(
        _surrogate_key, primary_key=True, autoincrement=True
    )
    #: Server-generated stable resource identifier ("akr_" + 64 hex chars).
    id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    #: The subject who owns the key and whose signature authenticated the
    #: request; the caller is always this actor.
    actor_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("actors.id", ondelete="RESTRICT"), nullable=False
    )
    #: The 32 raw Ed25519 public-key bytes (Base64 on the wire).
    public_key: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)
    #: True while the key authenticates; flips to false exactly once on retire.
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now
    )
    #: Set in the same transaction as the ``authentication_key.retired``
    #: audit event; null while the key is active.
    retired_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime, nullable=True
    )

    actor: Mapped[Actor] = relationship()


class ActorTrustPolicy(Base):
    """A subject's immutable signer-threshold trust policy.

    A policy registers the number of distinct qualified signers (1..100) the
    subject requires before a claim or evidence bundle is trusted in an
    authorization decision. Each subject has at most one policy: the
    ``actor_id`` is unique, so a retried registration of the same threshold
    returns the original record and a different threshold is a conflict
    rather than a replacement. Policies are append-only and immutable --
    there is deliberately no update, delete, deactivate, or replace path;
    ``enabled`` is true from creation and never flips.
    """

    __tablename__ = "actor_trust_policies"
    __table_args__ = (
        UniqueConstraint("actor_id", name="uq_actor_trust_policies_actor"),
        Index("ix_actor_trust_policies_created_order", "created_at", "seq"),
    )

    #: Monotonic insertion surrogate; the primary key for stable ordering.
    seq: Mapped[int] = mapped_column(
        _surrogate_key, primary_key=True, autoincrement=True
    )
    #: Server-generated stable policy identifier ("atp_" + 64 hex chars).
    id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    #: The subject the policy belongs to; exactly one policy per subject.
    actor_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("actors.id", ondelete="RESTRICT"), nullable=False
    )
    #: Distinct qualified signers required for a ``trusted`` decision.
    threshold: Mapped[int] = mapped_column(Integer, nullable=False)
    #: True from creation; policies are never deactivated.
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now
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


class ExchangeImportRecord(Base):
    """An immutable receipt for one offline-verified exchange-bundle package.

    The record is the controlled-import counterpart of the stateless
    verification route: a package whose manifest matches its snapshot under
    the existing canonical SHA-256 rules, and whose snapshot internal
    associations are consistent, is registered exactly once. The receipt
    stores only the receiving identity -- the manifest version, referenced
    evidence bundle id, and manifest digest -- never the snapshot itself and
    never any raw signature, claim payload, content, or evidence bytes; the
    full package must be re-presented on a retry. Verification never depends
    on whether the referenced resources exist locally, so no foreign keys and
    no local resource lookup participate in the decision.

    Records are append-only and immutable; there is deliberately no update
    or delete path. The identity triple is unique, so a retried submission
    for the same package returns the original record without another audit
    event.
    """

    __tablename__ = "evidence_bundle_exchange_imports"
    __table_args__ = (
        UniqueConstraint(
            "manifest_version",
            "evidence_bundle_id",
            "manifest_digest_hex",
            name="uq_exchange_imports_identity",
        ),
        Index(
            "ix_exchange_imports_created_order",
            "created_at",
            "seq",
        ),
    )

    #: Monotonic insertion surrogate; the primary key for stable ordering.
    seq: Mapped[int] = mapped_column(
        _surrogate_key, primary_key=True, autoincrement=True
    )
    #: Server-generated stable receipt identifier ("eir_" + 64 hex chars).
    id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    #: Fixed exchange manifest format version committed to by the package.
    manifest_version: Mapped[str] = mapped_column(String(64), nullable=False)
    #: The bundle id named by the manifest (an opaque reference, never a
    #: local lookup key at import time).
    evidence_bundle_id: Mapped[str] = mapped_column(String(80), nullable=False)
    #: The SHA-256 manifest digest the received snapshot was verified against.
    manifest_digest_hex: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now
    )


class CheckpointImportRecord(Base):
    """An immutable receipt for one offline-verified audit-event checkpoint.

    The record is the controlled-import counterpart of the stateless
    checkpoint-verification route: a checkpoint whose digest matches the
    canonical SHA-256 of the received event array under the existing
    checkpoint rules, with exactly the claimed event count, is registered
    exactly once. The receipt stores only the receiving identity -- the
    fixed checkpoint version, the event count, and the events digest --
    never the event array itself and never any raw event data; the full
    events must be re-presented on a retry. Verification never depends on
    whether the described events exist locally (the described events are
    never created, modified, or queried), so no foreign keys and no local
    audit lookup participate in the decision.

    Records are append-only and immutable; there is deliberately no update
    or delete path. The identity triple is unique, so a retried submission
    for the same checkpoint returns the original record without another
    audit event.
    """

    __tablename__ = "audit_checkpoint_imports"
    __table_args__ = (
        UniqueConstraint(
            "checkpoint_version",
            "event_count",
            "events_digest_hex",
            name="uq_checkpoint_imports_identity",
        ),
        Index(
            "ix_checkpoint_imports_created_order",
            "created_at",
            "seq",
        ),
    )

    #: Monotonic insertion surrogate; the primary key for stable ordering.
    seq: Mapped[int] = mapped_column(
        _surrogate_key, primary_key=True, autoincrement=True
    )
    #: Server-generated stable receipt identifier ("aci_" + 64 hex chars).
    id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    #: Fixed checkpoint format version committed to by the request.
    checkpoint_version: Mapped[str] = mapped_column(String(64), nullable=False)
    #: The number of events the checkpoint commits to.
    event_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    #: The SHA-256 events digest the received event array was verified against.
    events_digest_hex: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now
    )


class ImpactImportRecord(Base):
    """An immutable receipt for one offline-verified revocation-impact checkpoint.

    The record is the controlled-import counterpart of the stateless
    impact-verification route: a checkpoint whose digest matches the
    canonical SHA-256 of the received impacts array under the existing
    checkpoint rules, with exactly the claimed impact count, is registered
    exactly once. The receipt stores only the receiving identity -- the
    fixed checkpoint version, the impact count, and the impacts digest --
    never the impacts array itself and never any raw impact, signature,
    key, payload, content, or evidence data; the full impacts must be
    re-presented on a retry. Verification never depends on whether the
    described revocations exist locally (they are never created, modified,
    or queried), so no foreign keys and no local revocation lookup
    participate in the decision.

    Records are append-only and immutable; there is deliberately no update
    or delete path. The identity triple is unique, so a retried submission
    for the same checkpoint returns the original record without another
    audit event.
    """

    __tablename__ = "revocation_impact_imports"
    __table_args__ = (
        UniqueConstraint(
            "checkpoint_version",
            "impact_count",
            "impacts_digest_hex",
            name="uq_impact_imports_identity",
        ),
        Index(
            "ix_impact_imports_created_order",
            "created_at",
            "seq",
        ),
    )

    #: Monotonic insertion surrogate; the primary key for stable ordering.
    seq: Mapped[int] = mapped_column(
        _surrogate_key, primary_key=True, autoincrement=True
    )
    #: Server-generated stable receipt identifier ("rii_" + 64 hex chars).
    id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    #: Fixed checkpoint format version committed to by the request.
    checkpoint_version: Mapped[str] = mapped_column(String(64), nullable=False)
    #: The number of impacts the checkpoint commits to.
    impact_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    #: The SHA-256 impacts digest the received impacts array was verified against.
    impacts_digest_hex: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now
    )


class ImpactReconExchangeImportRecord(Base):
    """An immutable receipt for one signed, offline-verified recon package.

    The record is the controlled cross-system counterpart of the stateless
    impact-recon verification route: a package whose structure, counts, and
    array digest all pass the existing verification rules, and whose
    ``signature_metadata`` carries an Ed25519 signature verified over the
    canonical array binding the fixed exchange version, the signing
    subject, the package digest algorithm, and the package digest, is
    registered exactly once. The receipt stores only the package identity
    and the accepted signature's claims -- the signature version, the
    signing subject, the signer's 32-byte public key, both digest
    algorithms/values, and the SHA-256 digest of the verified signature --
    never the package itself, the raw signature, or any private key; the
    full package and signature must be re-presented on a retry. The
    signature is checked but its raw bytes are never persisted.
    Verification never depends on whether the referenced resources exist
    locally (they are never queried, created, or modified), so no foreign
    keys and no local resource lookup participate in the decision.

    Records are append-only and immutable; there is deliberately no update
    or delete path. The package identity tuple is unique, so a verified
    package is registered exactly once: a retried submission with the same
    identity and signature returns the original record without another
    audit event, while a different subject, key, or signature for the same
    package is refused with the original record untouched.
    """

    __tablename__ = "impact_recon_exchange_imports"
    __table_args__ = (
        UniqueConstraint(
            "signature_version",
            "package_digest_algorithm",
            "package_digest_hex",
            name="uq_impact_recon_exchange_imports_identity",
        ),
        Index(
            "ix_impact_recon_exchange_imports_created_order",
            "created_at",
            "seq",
        ),
    )

    #: Monotonic insertion surrogate; the primary key for stable ordering.
    seq: Mapped[int] = mapped_column(
        _surrogate_key, primary_key=True, autoincrement=True
    )
    #: Server-generated stable receipt identifier ("irx_" + 64 hex chars).
    id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    #: Fixed signature-exchange format version committed to by the metadata.
    signature_version: Mapped[str] = mapped_column(String(64), nullable=False)
    #: The signing subject named by the accepted signature metadata (an
    #: opaque reference, never a local actor lookup key at import time).
    signer_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    #: The 32 raw Ed25519 public-key bytes of the accepted signing subject
    #: (standard Base64 on the wire).
    public_key: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)
    #: The package digest algorithm named by the signature metadata.
    package_digest_algorithm: Mapped[str] = mapped_column(
        String(32), nullable=False
    )
    #: The SHA-256 package digest the received package was verified against
    #: and the signature was checked over.
    package_digest_hex: Mapped[str] = mapped_column(String(128), nullable=False)
    #: Digest algorithm of the signature digest ("sha256").
    signature_digest_algorithm: Mapped[str] = mapped_column(
        String(32), nullable=False
    )
    #: SHA-256 hex digest of the submitted signature; the raw signature is
    #: never stored or echoed back.
    signature_digest_hex: Mapped[str] = mapped_column(
        String(128), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now
    )


class ContentExportJob(Base):
    """An asynchronous job that exports one content's provenance-evidence snapshot.

    A job is registered with a client-changed ``request_id`` idempotency key
    and an existing ``content_id``. It starts ``pending`` with no run
    timestamps, result, or error. A run atomically claims a pending job into
    ``running`` (stamping UTC ``started_at``) and then settles it exactly once
    as ``succeeded`` -- stamping UTC ``finished_at`` and storing the existing
    read-only content export (``{"content", "claims"}``) as its ``result`` --
    or ``failed`` -- stamping ``finished_at``, leaving ``result`` null and
    recording the stable ``content_export_failed`` error. The state machine is
    monotonic: only a ``pending`` job can be claimed, so a concurrent or
    repeated run is a conflict rather than a second execution.

    ``request_id`` is unique: a retried submission for the same key returns
    the original job; the same key reused for a different content is a
    conflict. The result is the only field that carries export data; no raw
    content, claim payload, or evidence bytes are otherwise stored by a job.
    """

    __tablename__ = "content_export_jobs"
    __table_args__ = (
        Index("ix_content_export_jobs_created_order", "created_at", "seq"),
    )

    #: Monotonic insertion surrogate; the primary key for stable ordering.
    seq: Mapped[int] = mapped_column(
        _surrogate_key, primary_key=True, autoincrement=True
    )
    #: Server-generated stable job identifier ("cxj_" + 64 hex chars).
    id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    #: The content this job exports; must exist at creation.
    content_id: Mapped[str] = mapped_column(
        String(80), ForeignKey("contents.id", ondelete="RESTRICT"), nullable=False
    )
    #: Client-supplied idempotency key; unique across all jobs.
    request_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    #: "pending", "running", "succeeded", or "failed".
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utc_now
    )
    #: Set when a run claims the job; null while pending.
    started_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime, nullable=True
    )
    #: Set when the run settles (succeeded or failed); null beforehand.
    finished_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime, nullable=True
    )
    #: The export snapshot ({"content", "claims"}) on success; null otherwise.
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    #: Stable failure code on a failed run; null otherwise.
    error: Mapped[str | None] = mapped_column(String(64), nullable=True)

    content: Mapped[Content] = relationship()
