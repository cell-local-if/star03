"""Domain services: transactional creation, idempotent dedup, and lookups.

Every successful creation writes the resource row and its audit event in a
single transaction/COMMIT. A repeat content submission performs no writes and
adds no audit event.
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
    EvidenceBundleNotFoundError,
    UnknownActorError,
)
from provenance.models import (
    EVENT_ACTOR_CREATED,
    EVENT_CLAIM_CREATED,
    EVENT_CONTENT_CREATED,
    EVENT_EVIDENCE_BUNDLE_CREATED,
    Actor,
    AuditEvent,
    Claim,
    Content,
    EvidenceBundle,
)
from provenance.schemas import (
    ActorCreate,
    ClaimCreate,
    ContentCreate,
    EvidenceBundleCreate,
)

# Stable creation order: timestamp first, with the monotonic sequence as a
# deterministic tiebreaker.
_CONTENT_ORDER = (Content.created_at.asc(), Content.seq.asc())
_CLAIM_ORDER = (Claim.created_at.asc(), Claim.seq.asc())
_EVIDENCE_BUNDLE_ORDER = (
    EvidenceBundle.created_at.asc(),
    EvidenceBundle.seq.asc(),
)


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


def _claim_identity_select(payload: ClaimCreate, digest_hex: str):
    return select(Claim).where(
        Claim.content_id == payload.content_id,
        Claim.actor_id == payload.actor_id,
        Claim.claim_type == payload.claim_type,
        Claim.payload_digest_hex == digest_hex,
    )


def create_claim(session: Session, payload: ClaimCreate) -> tuple[Claim, bool]:
    """Create an immutable claim, returning ``(claim, created)``.

    Both the content and the claiming actor must already exist; the actor
    need not be the content's registering actor. The payload is committed to
    via its canonical-JSON SHA-256 digest and is never stored. A repeat
    submission of the same content, actor, claim type, and canonical payload
    returns the existing claim with ``created=False`` and writes no row or
    audit event. The claim row and its ``claim.created`` audit event commit
    in a single transaction.
    """
    content = session.execute(
        select(Content).where(Content.id == payload.content_id)
    ).scalar_one_or_none()
    if content is None:
        raise ContentNotFoundError(payload.content_id)
    actor = session.get(Actor, payload.actor_id)
    if actor is None:
        raise UnknownActorError(payload.actor_id)

    digest_hex = canonical.payload_digest_hex(payload.payload)
    existing = session.execute(
        _claim_identity_select(payload, digest_hex)
    ).scalar_one_or_none()
    if existing is not None:
        return existing, False

    claim = Claim(
        id=ids.claim_id(
            payload.content_id, payload.actor_id, payload.claim_type, digest_hex
        ),
        content_id=payload.content_id,
        actor_id=payload.actor_id,
        claim_type=payload.claim_type,
        payload_digest_algorithm=canonical.CANONICAL_DIGEST_ALGORITHM,
        payload_digest_hex=digest_hex,
    )
    session.add(claim)
    session.add(AuditEvent(event_type=EVENT_CLAIM_CREATED, resource_id=claim.id))
    try:
        session.commit()
    except IntegrityError:
        # Concurrent identical claim won the race: return its resource.
        session.rollback()
        raced = session.execute(
            _claim_identity_select(payload, digest_hex)
        ).scalar_one_or_none()
        if raced is None:  # pragma: no cover - defensive
            raise
        return raced, False
    session.refresh(claim)
    return claim, True


def get_claim(session: Session, claim_id: str) -> Claim:
    """Return a claim by its stable id or raise :class:`ClaimNotFoundError`."""
    claim = session.execute(
        select(Claim).where(Claim.id == claim_id)
    ).scalar_one_or_none()
    if claim is None:
        raise ClaimNotFoundError(claim_id)
    return claim


def list_claims_for_content(session: Session, content_id: str) -> list[Claim]:
    """Return claims for one content in stable creation order.

    The content must exist; an unknown content id is a missing resource, not
    an empty collection.
    """
    content = session.execute(
        select(Content).where(Content.id == content_id)
    ).scalar_one_or_none()
    if content is None:
        raise ContentNotFoundError(content_id)
    stmt = (
        select(Claim)
        .where(Claim.content_id == content_id)
        .order_by(*_CLAIM_ORDER)
    )
    return list(session.execute(stmt).scalars().all())


def _evidence_bundle_identity_select(payload: EvidenceBundleCreate):
    return select(EvidenceBundle).where(
        EvidenceBundle.claim_id == payload.claim_id,
        EvidenceBundle.evidence_type == payload.evidence_type,
        EvidenceBundle.digest_hex == payload.digest_hex,
    )


def create_evidence_bundle(
    session: Session, payload: EvidenceBundleCreate
) -> tuple[EvidenceBundle, bool]:
    """Create an evidence bundle for a claim, returning ``(bundle, created)``.

    The referenced claim must already exist. Only the digest, media type, and
    JSON metadata are persisted — raw evidence bytes are never accepted or
    stored. A repeat submission with the same claim, evidence type, and digest
    returns the existing bundle (keeping the first metadata) with
    ``created=False`` and writes no row or audit event; any other field
    combination forms an independent bundle. The bundle row and its
    ``evidence_bundle.created`` audit event commit in a single transaction.
    """
    claim = session.execute(
        select(Claim).where(Claim.id == payload.claim_id)
    ).scalar_one_or_none()
    if claim is None:
        raise ClaimNotFoundError(payload.claim_id)

    existing = session.execute(
        _evidence_bundle_identity_select(payload)
    ).scalar_one_or_none()
    if existing is not None:
        return existing, False

    bundle = EvidenceBundle(
        id=ids.evidence_bundle_id(
            payload.claim_id, payload.evidence_type, payload.digest_hex
        ),
        claim_id=payload.claim_id,
        evidence_type=payload.evidence_type,
        digest_algorithm=payload.digest_algorithm,
        digest_hex=payload.digest_hex,
        media_type=payload.media_type,
        metadata_json=payload.metadata,
    )
    session.add(bundle)
    session.add(
        AuditEvent(
            event_type=EVENT_EVIDENCE_BUNDLE_CREATED, resource_id=bundle.id
        )
    )
    try:
        session.commit()
    except IntegrityError:
        # Concurrent identical bundle won the race: return its resource.
        session.rollback()
        raced = session.execute(
            _evidence_bundle_identity_select(payload)
        ).scalar_one_or_none()
        if raced is None:  # pragma: no cover - defensive
            raise
        return raced, False
    session.refresh(bundle)
    return bundle, True


def get_evidence_bundle(session: Session, evidence_bundle_id: str) -> EvidenceBundle:
    """Return a bundle by id or raise :class:`EvidenceBundleNotFoundError`."""
    bundle = session.execute(
        select(EvidenceBundle).where(EvidenceBundle.id == evidence_bundle_id)
    ).scalar_one_or_none()
    if bundle is None:
        raise EvidenceBundleNotFoundError(evidence_bundle_id)
    return bundle


def list_evidence_bundles_for_claim(
    session: Session, claim_id: str
) -> list[EvidenceBundle]:
    """Return evidence bundles for one claim in stable creation order.

    The claim must exist; an unknown claim id is a missing resource, not an
    empty collection.
    """
    claim = session.execute(
        select(Claim).where(Claim.id == claim_id)
    ).scalar_one_or_none()
    if claim is None:
        raise ClaimNotFoundError(claim_id)
    stmt = (
        select(EvidenceBundle)
        .where(EvidenceBundle.claim_id == claim_id)
        .order_by(*_EVIDENCE_BUNDLE_ORDER)
    )
    return list(session.execute(stmt).scalars().all())
