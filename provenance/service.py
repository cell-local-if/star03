"""Domain services: transactional creation, idempotent dedup, and lookups.

Every successful creation writes the resource row and its audit event in a
single transaction/COMMIT. A repeat content or claim submission performs no
writes and adds no audit event.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from provenance import canonical, ids
from provenance.errors import (
    ActorAlreadyExistsError,
    ClaimNotFoundError,
    ContentNotFoundError,
    UnknownActorError,
)
from provenance.models import (
    EVENT_ACTOR_CREATED,
    EVENT_CLAIM_CREATED,
    EVENT_CONTENT_CREATED,
    Actor,
    AuditEvent,
    Claim,
    Content,
)
from provenance.schemas import ActorCreate, ClaimCreate, ContentCreate
from provenance.time_utils import utc_now

# Stable creation order: timestamp first, with the monotonic sequence as a
# deterministic tiebreaker.
_CONTENT_ORDER = (Content.created_at.asc(), Content.seq.asc())
_CLAIM_ORDER = (Claim.created_at.asc(), Claim.seq.asc())


def _content_by_id(session: Session, content_id: str) -> Content | None:
    return session.execute(
        select(Content).where(Content.id == content_id)
    ).scalar_one_or_none()


def _claim_by_id(session: Session, claim_id: str) -> Claim | None:
    return session.execute(
        select(Claim).where(Claim.id == claim_id)
    ).scalar_one_or_none()


def create_actor(session: Session, payload: ActorCreate) -> Actor:
    """Create an actor and its audit event atomically.

    Raises :class:`ActorAlreadyExistsError` on a duplicate identifier.
    """
    existing = session.get(Actor, payload.id)
    if existing is not None:
        raise ActorAlreadyExistsError(payload.id)

    actor = Actor(id=payload.id, name=payload.name, type=payload.type)
    session.add(actor)
    session.add(
        AuditEvent(event_type=EVENT_ACTOR_CREATED, resource_id=actor.id)
    )
    try:
        session.commit()
    except IntegrityError:
        # Lost a concurrent insert race for the same identifier.
        session.rollback()
        raise ActorAlreadyExistsError(payload.id) from None
    session.refresh(actor)
    return actor


def create_content(
    session: Session, payload: ContentCreate
) -> tuple[Content, bool]:
    """Register content, returning ``(content, created)``.

    The referenced actor must exist. If content with the same algorithm and
    digest already exists, the existing resource is returned with
    ``created=False`` and no row or audit event is written.
    """
    actor = session.get(Actor, payload.actor_id)
    if actor is None:
        raise UnknownActorError(payload.actor_id)

    existing = session.execute(
        select(Content).where(
            Content.digest_algorithm == payload.digest_algorithm,
            Content.digest_hex == payload.digest_hex,
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing, False

    content = Content(
        id=ids.content_id(payload.digest_algorithm, payload.digest_hex),
        digest_algorithm=payload.digest_algorithm,
        digest_hex=payload.digest_hex,
        media_type=payload.media_type,
        title=payload.title,
        actor_id=payload.actor_id,
    )
    session.add(content)
    session.add(
        AuditEvent(event_type=EVENT_CONTENT_CREATED, resource_id=content.id)
    )
    try:
        session.commit()
    except IntegrityError:
        # Concurrent identical registration won the race: return its resource.
        session.rollback()
        raced = session.execute(
            select(Content).where(
                Content.digest_algorithm == payload.digest_algorithm,
                Content.digest_hex == payload.digest_hex,
            )
        ).scalar_one_or_none()
        if raced is None:  # pragma: no cover - defensive
            raise
        return raced, False
    session.refresh(content)
    return content, True


def get_content(session: Session, content_id: str) -> Content:
    """Return content by its stable id or raise :class:`ContentNotFoundError`."""
    content = session.execute(
        select(Content).where(Content.id == content_id)
    ).scalar_one_or_none()
    if content is None:
        raise ContentNotFoundError(content_id)
    return content


def list_contents(session: Session, actor_id: str | None = None) -> list[Content]:
    """Return contents in stable creation order, optionally by source actor."""
    stmt = select(Content)
    if actor_id is not None:
        stmt = stmt.where(Content.actor_id == actor_id)
    stmt = stmt.order_by(*_CONTENT_ORDER)
    return list(session.execute(stmt).scalars().all())


def create_claim(session: Session, payload: ClaimCreate) -> tuple[Claim, bool]:
    """Create an immutable claim, returning ``(claim, created)``.

    Both the content and the actor must already exist; the claiming actor may
    differ from the content's registering actor. The raw payload is never
    stored: only the SHA-256 digest of its canonical form is. A resubmission
    with the same content, actor, claim type, and canonical payload returns
    the existing claim with ``created=False`` and writes nothing.

    On first creation the claim row and its ``claim.created`` audit event
    share one timestamp and commit in a single transaction.
    """
    content = _content_by_id(session, payload.content_id)
    if content is None:
        raise ContentNotFoundError(payload.content_id)
    actor = session.get(Actor, payload.actor_id)
    if actor is None:
        raise UnknownActorError(payload.actor_id)

    algorithm, digest_hex = canonical.canonical_digest(payload.payload)
    claim_id = ids.claim_id(content.id, actor.id, payload.claim_type, digest_hex)

    existing = _claim_by_id(session, claim_id)
    if existing is not None:
        return existing, False

    created_at = utc_now()
    claim = Claim(
        id=claim_id,
        content_id=content.id,
        actor_id=actor.id,
        claim_type=payload.claim_type,
        payload_digest_algorithm=algorithm,
        payload_digest=digest_hex,
        created_at=created_at,
    )
    session.add(claim)
    session.add(
        AuditEvent(
            event_type=EVENT_CLAIM_CREATED,
            resource_id=claim.id,
            created_at=created_at,
        )
    )
    try:
        session.commit()
    except IntegrityError:
        # A concurrent identical submission won the race: return its claim.
        session.rollback()
        raced = _claim_by_id(session, claim_id)
        if raced is None:  # pragma: no cover - defensive
            raise
        return raced, False
    session.refresh(claim)
    return claim, True


def get_claim(session: Session, claim_id: str) -> Claim:
    """Return a claim by its stable id or raise :class:`ClaimNotFoundError`."""
    claim = _claim_by_id(session, claim_id)
    if claim is None:
        raise ClaimNotFoundError(claim_id)
    return claim


def list_claims_for_content(session: Session, content_id: str) -> list[Claim]:
    """Return a content's claims in stable creation order.

    Raises :class:`ContentNotFoundError` when the parent content is unknown.
    """
    content = _content_by_id(session, content_id)
    if content is None:
        raise ContentNotFoundError(content_id)
    stmt = (
        select(Claim)
        .where(Claim.content_id == content.id)
        .order_by(*_CLAIM_ORDER)
    )
    return list(session.execute(stmt).scalars().all())
