"""SQLAlchemy models for actors, content identities, and audit events.

Content bytes are never stored; only digests and metadata are persisted.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, TypeDecorator, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class UTCDateTime(TypeDecorator):
    """Store datetimes as UTC and always return timezone-aware values."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("only timezone-aware datetimes may be persisted")
        return value.astimezone(timezone.utc)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_content_id() -> str:
    return str(uuid.uuid4())


class Actor(Base):
    __tablename__ = "actors"

    actor_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    actor_type: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)


class Content(Base):
    __tablename__ = "contents"
    __table_args__ = (
        UniqueConstraint("digest_algorithm", "digest_hex", name="uq_contents_digest"),
    )

    # Surrogate row id gives a stable creation order for list queries.
    row_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True, default=new_content_id)
    digest_algorithm: Mapped[str] = mapped_column(String(32), nullable=False)
    digest_hex: Mapped[str] = mapped_column(String(64), nullable=False)
    media_type: Mapped[str] = mapped_column(String(256), nullable=False)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    actor_id: Mapped[str] = mapped_column(ForeignKey("actors.actor_id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(128), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)


EVENT_ACTOR_CREATED = "actor.created"
EVENT_CONTENT_CREATED = "content.created"
