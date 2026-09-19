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
