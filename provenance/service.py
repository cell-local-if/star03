"""Domain operations. Each public function commits exactly one transaction."""

from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import errors, models
from .schemas import ActorCreate, ContentCreate

SUPPORTED_DIGEST_ALGORITHM = "sha256"
_DIGEST_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def create_actor(session: Session, payload: ActorCreate) -> models.Actor:
    existing = session.get(models.Actor, payload.actor_id)
    if existing is not None:
        raise errors.actor_already_exists(payload.actor_id)
    actor = models.Actor(
        actor_id=payload.actor_id,
        name=payload.name,
        actor_type=payload.actor_type,
    )
    session.add(actor)
    session.add(
        models.AuditEvent(
            event_type=models.EVENT_ACTOR_CREATED,
            resource_id=actor.actor_id,
        )
    )
    session.commit()
    return actor


def _validate_digest(algorithm: str, digest_hex: str) -> str:
    if algorithm != SUPPORTED_DIGEST_ALGORITHM:
        raise errors.unsupported_digest_algorithm(algorithm)
    if not _DIGEST_HEX_RE.fullmatch(digest_hex):
        raise errors.invalid_digest_hex()
    return digest_hex.lower()


def _find_content_by_digest(
    session: Session, algorithm: str, digest_hex: str
) -> models.Content | None:
    stmt = select(models.Content).where(
        models.Content.digest_algorithm == algorithm,
        models.Content.digest_hex == digest_hex,
    )
    return session.execute(stmt).scalar_one_or_none()


def create_content(
    session: Session, payload: ContentCreate
) -> tuple[models.Content, bool]:
    """Register a content identity.

    Returns ``(content, created)``. Submitting an already-registered
    (algorithm, digest) pair returns the existing row without creating a
    new identifier or audit event.
    """
    digest_hex = _validate_digest(payload.digest_algorithm, payload.digest_hex)

    actor = session.get(models.Actor, payload.actor_id)
    if actor is None:
        raise errors.unknown_actor(payload.actor_id)

    existing = _find_content_by_digest(session, payload.digest_algorithm, digest_hex)
    if existing is not None:
        return existing, False

    content = models.Content(
        content_id=models.new_content_id(),
        digest_algorithm=payload.digest_algorithm,
        digest_hex=digest_hex,
        media_type=payload.media_type,
        title=payload.title,
        actor_id=payload.actor_id,
    )
    session.add(content)
    session.add(
        models.AuditEvent(
            event_type=models.EVENT_CONTENT_CREATED,
            resource_id=content.content_id,
        )
    )
    try:
        session.commit()
    except IntegrityError:
        # A concurrent request registered the same digest first; the unique
        # constraint on (algorithm, digest_hex) guarantees idempotency.
        session.rollback()
        existing = _find_content_by_digest(session, payload.digest_algorithm, digest_hex)
        if existing is None:  # pragma: no cover - constraint was on something else
            raise
        return existing, False
    return content, True


def get_content(session: Session, content_id: str) -> models.Content:
    stmt = select(models.Content).where(models.Content.content_id == content_id)
    content = session.execute(stmt).scalar_one_or_none()
    if content is None:
        raise errors.content_not_found(content_id)
    return content


def list_contents(session: Session, actor_id: str | None = None) -> list[models.Content]:
    stmt = select(models.Content).order_by(models.Content.row_id)
    if actor_id is not None:
        stmt = stmt.where(models.Content.actor_id == actor_id)
    return list(session.execute(stmt).scalars().all())
