"""Domain services: transactional creation, idempotent dedup, and lookups.

Every successful creation writes the resource row and its audit event in a
single transaction/COMMIT. A repeat content submission performs no writes and
adds no audit event.
"""

from __future__ import annotations

import hashlib
from typing import Callable

from sqlalchemy import exists, func, or_, select, update as sa_update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from provenance import canonical, ed25519, ids, signing
from provenance.errors import (
    ActorAlreadyExistsError,
    ActorTrustPolicyConflictError,
    AuditCheckpointImportNotFoundError,
    AttestationAccessGrantRevocationNotFoundError,
    AttestationNotFoundError,
    AttestationRevocationNotFoundError,
    AttestationVerificationError,
    ClaimNotFoundError,
    ClaimSupersessionNotFoundError,
    ClaimSupersessionValidationError,
    ContentExportJobConflictError,
    ContentExportJobNotFoundError,
    ContentExportRequestConflictError,
    ContentNotFoundError,
    ContentRelationNotFoundError,
    ContentRelationValidationError,
    EvidenceBundleExchangeImportNotFoundError,
    EvidenceBundleNotFoundError,
    ProtectedAccessValidationError,
    UnknownActorError,
)
from provenance.models import (
    CONTENT_EXPORT_JOB_FAILED,
    CONTENT_EXPORT_JOB_FAILED_ERROR,
    CONTENT_EXPORT_JOB_PENDING,
    CONTENT_EXPORT_JOB_RUNNING,
    CONTENT_EXPORT_JOB_STATES,
    CONTENT_EXPORT_JOB_SUCCEEDED,
    EVENT_ACTOR_CREATED,
    EVENT_ACTOR_TRUST_POLICY_CREATED,
    EVENT_ATTESTATION_ACCESS_GRANTED,
    EVENT_ATTESTATION_ACCESS_GRANT_REVOKED,
    EVENT_ATTESTATION_CREATED,
    EVENT_ATTESTATION_REVOKED,
    EVENT_AUDIT_CHECKPOINT_IMPORTED,
    EVENT_AUTHENTICATION_KEY_RETIRED,
    EVENT_AUTHENTICATION_KEY_ROTATED,
    EVENT_CLAIM_CREATED,
    EVENT_CLAIM_SUPERSEDED,
    EVENT_CONTENT_CREATED,
    EVENT_CONTENT_EXPORT_JOB_CREATED,
    EVENT_CONTENT_EXPORT_JOB_RUN,
    EVENT_CONTENT_RELATION_CREATED,
    EVENT_EVIDENCE_BUNDLE_CREATED,
    EVENT_EVIDENCE_BUNDLE_EXCHANGE_IMPORTED,
    Actor,
    ActorTrustPolicy,
    Attestation,
    AttestationAccessGrant,
    AttestationAccessGrantRevocation,
    AttestationRevocation,
    AuthenticationKeyRotation,
    AuditEvent,
    CheckpointImportRecord,
    Claim,
    ClaimSupersession,
    Content,
    ContentExportJob,
    ContentRelation,
    EvidenceBundle,
    ExchangeImportRecord,
)
from provenance.schemas import (
    ActorCreate,
    ActorTrustPolicyCreate,
    AttestationAccessGrantCreate,
    AttestationAccessGrantRevocationCreate,
    AttestationCreate,
    AttestationRevocationCreate,
    AuditCheckpointImportCreate,
    AuthenticationKeyRotationCreate,
    ClaimCreate,
    ClaimSupersessionCreate,
    ContentCreate,
    ContentExportJobCreate,
    ContentRelationCreate,
    EvidenceBundleExchangeImportCreate,
    EvidenceBundleCreate,
    EvidenceBundleImportCreate,
)
from provenance.time_utils import utc_now

# Stable creation order: timestamp first, with the monotonic sequence as a
# deterministic tiebreaker.
_ACTOR_ORDER = (Actor.created_at.asc(), Actor.seq.asc())
_CONTENT_ORDER = (Content.created_at.asc(), Content.seq.asc())
_CLAIM_ORDER = (Claim.created_at.asc(), Claim.seq.asc())
_CLAIM_SUPERSESSION_ORDER = (
    ClaimSupersession.created_at.asc(),
    ClaimSupersession.seq.asc(),
)
_EVIDENCE_BUNDLE_ORDER = (
    EvidenceBundle.created_at.asc(),
    EvidenceBundle.seq.asc(),
)
_ATTESTATION_ORDER = (Attestation.created_at.asc(), Attestation.seq.asc())
_ATTESTATION_REVOCATION_ORDER = (
    AttestationRevocation.created_at.asc(),
    AttestationRevocation.seq.asc(),
)
_ATTESTATION_ACCESS_GRANT_ORDER = (
    AttestationAccessGrant.created_at.asc(),
    AttestationAccessGrant.seq.asc(),
)
_ATTESTATION_ACCESS_GRANT_REVOCATION_ORDER = (
    AttestationAccessGrantRevocation.created_at.asc(),
    AttestationAccessGrantRevocation.seq.asc(),
)
_CONTENT_RELATION_ORDER = (
    ContentRelation.created_at.asc(),
    ContentRelation.seq.asc(),
)
_AUDIT_EVENT_ORDER = (AuditEvent.created_at.asc(), AuditEvent.seq.asc())
_CONTENT_EXPORT_JOB_ORDER = (
    ContentExportJob.created_at.asc(),
    ContentExportJob.seq.asc(),
)
_ACTOR_TRUST_POLICY_ORDER = (
    ActorTrustPolicy.created_at.asc(),
    ActorTrustPolicy.seq.asc(),
)


def get_actor_by_id(session: Session, actor_id: str) -> Actor | None:
    """Return the actor with the client-supplied id, or ``None``.

    The actor's stable business identifier is unique but is not the ORM
    primary key (the monotonic ``seq`` is), so lookups go through a select
    rather than ``Session.get``.
    """
    return session.execute(
        select(Actor).where(Actor.id == actor_id)
    ).scalar_one_or_none()


def create_actor(session: Session, payload: ActorCreate) -> Actor:
    """Create an actor and its audit event atomically.

    Raises :class:`ActorAlreadyExistsError` on a duplicate identifier.
    """
    existing = get_actor_by_id(session, payload.id)
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


# Actor search paging bounds.
DEFAULT_ACTORS_LIMIT = 50
MIN_ACTORS_LIMIT = 1
MAX_ACTORS_LIMIT = 100


def list_actors(
    session: Session,
    actor_id: str | None = None,
    name: str | None = None,
    actor_type: str | None = None,
) -> list[Actor]:
    """Return existing actors in stable creation order, optionally filtered.

    ``actor_id``, ``name``, and ``actor_type`` are exact, case- and
    whitespace-sensitive matches that combine as logical AND; ``None`` means
    unfiltered. The values are never resolved for existence, so an unknown
    identifier, name, or type is an empty result rather than a missing
    resource. Results follow the actors' stable creation order
    (``created_at`` with the monotonic ``seq`` tiebreaker), which stays fixed
    across an app restart. The search is strictly read-only: it writes no
    actor, resource, or audit event.
    """
    stmt = select(Actor)
    if actor_id is not None:
        stmt = stmt.where(Actor.id == actor_id)
    if name is not None:
        stmt = stmt.where(Actor.name == name)
    if actor_type is not None:
        stmt = stmt.where(Actor.type == actor_type)
    stmt = stmt.order_by(*_ACTOR_ORDER)
    return list(session.execute(stmt).scalars().all())


def create_content(
    session: Session, payload: ContentCreate
) -> tuple[Content, bool]:
    """Register content, returning ``(content, created)``.

    The referenced actor must exist. If content with the same algorithm and
    digest already exists, the existing resource is returned with
    ``created=False`` and no row or audit event is written.
    """
    actor = get_actor_by_id(session, payload.actor_id)
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
    actor = get_actor_by_id(session, payload.actor_id)
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


def _supersession_identity_select(payload: ClaimSupersessionCreate):
    return select(ClaimSupersession).where(
        ClaimSupersession.superseded_claim_id
        == payload.superseded_claim_id,
        ClaimSupersession.replacement_claim_id
        == payload.replacement_claim_id,
        ClaimSupersession.reason == payload.reason,
    )


def _supersession_would_create_cycle(
    session: Session,
    superseded_claim_id: str,
    replacement_claim_id: str,
) -> bool:
    """True if edge ``replacement -> superseded`` would close a cycle.

    Edges point from a replacement to the claim it supersedes, so a cycle
    forms exactly when ``replacement_claim_id`` is already reachable from
    ``superseded_claim_id`` by following existing supersession edges.
    """
    visited: set[str] = set()
    frontier = [superseded_claim_id]
    while frontier:
        current = frontier.pop()
        if current == replacement_claim_id:
            return True
        if current in visited:
            continue
        visited.add(current)
        older = session.execute(
            select(ClaimSupersession.superseded_claim_id).where(
                ClaimSupersession.replacement_claim_id == current
            )
        ).scalars().all()
        frontier.extend(older)
    return False


def create_claim_supersession(
    session: Session, payload: ClaimSupersessionCreate
) -> tuple[ClaimSupersession, bool]:
    """Create an immutable claim supersession, returning ``(record, created)``.

    Both endpoints must be existing claims. A self-supersession, endpoints on
    different contents, and an edge that would close a cycle are validation
    errors and write nothing. A repeat submission of the same superseded
    claim, replacement claim, and (trimmed) reason returns the existing
    record with ``created=False`` and writes no row or audit event; a
    different reason forms an independent record. On first creation the row
    and its ``claim.superseded`` audit event commit in a single transaction.
    """
    # A self-supersession is invalid input, checked before any lookup.
    if payload.superseded_claim_id == payload.replacement_claim_id:
        raise ClaimSupersessionValidationError("self_supersession")

    superseded = session.execute(
        select(Claim).where(Claim.id == payload.superseded_claim_id)
    ).scalar_one_or_none()
    if superseded is None:
        raise ClaimNotFoundError(payload.superseded_claim_id)
    replacement = session.execute(
        select(Claim).where(Claim.id == payload.replacement_claim_id)
    ).scalar_one_or_none()
    if replacement is None:
        raise ClaimNotFoundError(payload.replacement_claim_id)

    # The correction must concern one and the same content identity.
    if superseded.content_id != replacement.content_id:
        raise ClaimSupersessionValidationError("different_content")

    # An existing record with this exact identity is returned unchanged; a
    # repeat submission is idempotent and needs no row or audit event.
    existing = session.execute(
        _supersession_identity_select(payload)
    ).scalar_one_or_none()
    if existing is not None:
        return existing, False

    if _supersession_would_create_cycle(
        session,
        payload.superseded_claim_id,
        payload.replacement_claim_id,
    ):
        raise ClaimSupersessionValidationError("supersession_cycle")

    record = ClaimSupersession(
        id=ids.claim_supersession_id(
            payload.superseded_claim_id,
            payload.replacement_claim_id,
            payload.reason,
        ),
        superseded_claim_id=payload.superseded_claim_id,
        replacement_claim_id=payload.replacement_claim_id,
        reason=payload.reason,
    )
    session.add(record)
    session.add(
        AuditEvent(event_type=EVENT_CLAIM_SUPERSEDED, resource_id=record.id)
    )
    try:
        session.commit()
    except IntegrityError:
        # Concurrent identical supersession won the race: return its record.
        session.rollback()
        raced = session.execute(
            _supersession_identity_select(payload)
        ).scalar_one_or_none()
        if raced is None:  # pragma: no cover - defensive
            raise
        return raced, False
    session.refresh(record)
    return record, True


def get_claim_supersession(
    session: Session, supersession_id: str
) -> ClaimSupersession:
    """Return a supersession by id or raise
    :class:`ClaimSupersessionNotFoundError`."""
    record = session.execute(
        select(ClaimSupersession).where(
            ClaimSupersession.id == supersession_id
        )
    ).scalar_one_or_none()
    if record is None:
        raise ClaimSupersessionNotFoundError(supersession_id)
    return record


def list_supersessions_for_claim(
    session: Session, claim_id: str
) -> list[ClaimSupersession]:
    """Return supersessions where the claim is either endpoint, creation order.

    The claim must exist; an unknown claim id is the existing
    ``claim_not_found`` missing resource, not an empty collection.
    """
    claim = session.execute(
        select(Claim).where(Claim.id == claim_id)
    ).scalar_one_or_none()
    if claim is None:
        raise ClaimNotFoundError(claim_id)
    stmt = (
        select(ClaimSupersession)
        .where(
            or_(
                ClaimSupersession.superseded_claim_id == claim_id,
                ClaimSupersession.replacement_claim_id == claim_id,
            )
        )
        .order_by(*_CLAIM_SUPERSESSION_ORDER)
    )
    return list(session.execute(stmt).scalars().all())


# Reviewer claim search paging bounds.
DEFAULT_CLAIMS_LIMIT = 50
MIN_CLAIMS_LIMIT = 1
MAX_CLAIMS_LIMIT = 100


# Supersession-lineage traversal directions.
SUPERSESSION_LINEAGE_NEWER = "newer"
SUPERSESSION_LINEAGE_OLDER = "older"
SUPERSESSION_LINEAGE_DIRECTIONS = frozenset(
    {SUPERSESSION_LINEAGE_NEWER, SUPERSESSION_LINEAGE_OLDER}
)

DEFAULT_SUPERSESSION_LINEAGE_MAX_DEPTH = 8
MIN_SUPERSESSION_LINEAGE_MAX_DEPTH = 1
MAX_SUPERSESSION_LINEAGE_MAX_DEPTH = 32

DEFAULT_SUPERSESSION_LINEAGE_MIN_DEPTH = 1
MIN_SUPERSESSION_LINEAGE_MIN_DEPTH = 1
MAX_SUPERSESSION_LINEAGE_MIN_DEPTH = 32

DEFAULT_SUPERSESSION_LINEAGE_LIMIT = 50
MIN_SUPERSESSION_LINEAGE_LIMIT = 1
MAX_SUPERSESSION_LINEAGE_LIMIT = 100


def get_claim_supersession_lineage(
    session: Session,
    claim_id: str,
    direction: str,
    max_depth: int = DEFAULT_SUPERSESSION_LINEAGE_MAX_DEPTH,
) -> list[tuple[Claim, int]]:
    """Return claims reachable through supersessions as ``(claim, depth)`` pairs.

    ``newer`` traversal follows an edge from the superseded claim to its
    ``replacement_claim_id``; ``older`` traversal follows it in reverse to
    the ``superseded_claim_id``. The origin claim is never included. Only
    claims reachable within ``max_depth`` edges are returned; each claim
    appears once at its shortest depth. Pairs are ordered by depth
    ascending; within one depth, claims are ordered by the stable creation
    order of the supersession record through which they were first reached.

    The traversal is read-only and tracks a visited set, so even anomalous
    history containing a cycle terminates and the walk is bounded by
    ``max_depth``. The origin must exist or :class:`ClaimNotFoundError` is
    raised.
    """
    # A missing origin is a missing resource, checked before any traversal.
    get_claim(session, claim_id)

    if direction == SUPERSESSION_LINEAGE_NEWER:
        source_col = ClaimSupersession.superseded_claim_id
        neighbor_col = ClaimSupersession.replacement_claim_id
    else:
        source_col = ClaimSupersession.replacement_claim_id
        neighbor_col = ClaimSupersession.superseded_claim_id

    visited: set[str] = {claim_id}
    frontier: list[str] = [claim_id]
    # id -> shortest depth; discovery order is the dictionary insertion
    # order itself.
    reached: dict[str, int] = {}
    for depth in range(1, max_depth + 1):
        if not frontier:
            break
        # Ordering every edge leaving the current frontier by its stable
        # creation order fixes each level's first-discovery order, including
        # converging paths and parallel records between the same two claims.
        rows = session.execute(
            select(neighbor_col)
            .where(source_col.in_(frontier))
            .order_by(*_CLAIM_SUPERSESSION_ORDER)
        ).scalars().all()
        next_frontier: list[str] = []
        for neighbor_id in rows:
            if neighbor_id in visited:
                # Already reached at an equal or shorter depth; also what
                # makes an anomalous cycle terminate.
                continue
            visited.add(neighbor_id)
            next_frontier.append(neighbor_id)
            reached[neighbor_id] = depth
        frontier = next_frontier

    if not reached:
        return []

    claims = (
        session.execute(select(Claim).where(Claim.id.in_(list(reached))))
        .scalars()
        .all()
    )
    by_id = {claim.id: claim for claim in claims}
    return [(by_id[reached_id], reached_depth) for reached_id, reached_depth in reached.items()]


def list_claims(
    session: Session,
    content_id: str | None = None,
    actor_id: str | None = None,
    claim_type: str | None = None,
    payload_digest_hex: str | None = None,
) -> list[Claim]:
    """Return existing immutable claims in stable creation order, filtered.

    Every filter is an exact, case- and whitespace-sensitive string match
    applied as logical AND; ``None`` means unfiltered.
    ``payload_digest_hex`` is the strict 64-lowercase-hex spelling already
    enforced at the boundary. Results follow the claims' stable creation
    order (``created_at`` with the monotonic ``seq`` tiebreaker). The search
    is strictly read-only: it writes no claim, resource, or audit event, and
    no referenced content/actor existence is required, so a filter that
    matches nothing is an empty result rather than a missing resource. The
    raw payload is never stored and therefore never appears here.
    """
    stmt = select(Claim)
    if content_id is not None:
        stmt = stmt.where(Claim.content_id == content_id)
    if actor_id is not None:
        stmt = stmt.where(Claim.actor_id == actor_id)
    if claim_type is not None:
        stmt = stmt.where(Claim.claim_type == claim_type)
    if payload_digest_hex is not None:
        stmt = stmt.where(Claim.payload_digest_hex == payload_digest_hex)
    stmt = stmt.order_by(*_CLAIM_ORDER)
    return list(session.execute(stmt).scalars().all())


def _evidence_bundle_identity(payload: EvidenceBundleCreate) -> tuple[str, str, str, str]:
    """The dedup identity of a bundle: claim, type, algorithm, and digest."""
    return (
        payload.claim_id,
        payload.evidence_type,
        payload.digest_algorithm,
        payload.digest_hex,
    )


def _evidence_bundle_identity_select(payload: EvidenceBundleCreate):
    return select(EvidenceBundle).where(
        EvidenceBundle.claim_id == payload.claim_id,
        EvidenceBundle.evidence_type == payload.evidence_type,
        EvidenceBundle.digest_algorithm == payload.digest_algorithm,
        EvidenceBundle.digest_hex == payload.digest_hex,
    )


def create_evidence_bundle(
    session: Session, payload: EvidenceBundleCreate
) -> tuple[EvidenceBundle, bool]:
    """Create an evidence bundle, returning ``(bundle, created)``.

    The referenced claim must already exist. Evidence bytes are never seen
    by this service: only the digest, media type, and metadata are stored.
    A repeat submission of the same claim, evidence type, digest algorithm,
    and digest value returns the existing bundle with ``created=False`` --
    its first-submission metadata is retained -- and writes no row or audit
    event. Any other field combination forms an independent bundle. The
    bundle row and its ``evidence_bundle.created`` audit event commit in a
    single transaction.
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
            payload.claim_id,
            payload.evidence_type,
            payload.digest_algorithm,
            payload.digest_hex,
        ),
        claim_id=payload.claim_id,
        evidence_type=payload.evidence_type,
        digest_algorithm=payload.digest_algorithm,
        digest_hex=payload.digest_hex,
        media_type=payload.media_type,
        metadata_=payload.metadata,
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


def create_evidence_bundle_imports(
    session: Session, payload: EvidenceBundleImportCreate
) -> tuple[list[EvidenceBundle], bool]:
    """Import a batch of evidence bundles atomically.

    Returns ``(bundles, created_any)``: one bundle per unique
    (claim, evidence type, digest algorithm, digest) identity, in the
    identity's first-occurrence order within the batch. ``created_any`` is
    true iff at least one identity was newly created by this batch.

    Every item is validated by the request schema exactly as a single
    create, and every referenced claim must exist: the first missing claim
    (in request order) raises :class:`ClaimNotFoundError` and the whole
    batch writes nothing -- no bundle rows and no audit events.

    In-batch duplicates of an identity collapse to their first occurrence;
    an identity matching an existing bundle returns that bundle with its
    original first-submission metadata, unchanged. Only genuinely new
    identities are inserted, each with its own ``evidence_bundle.created``
    audit event, and all new rows and events commit in a single
    transaction.
    """
    # Validate every referenced claim before any write: a single missing
    # claim rejects the entire batch.
    claim_ids = list(dict.fromkeys(item.claim_id for item in payload.items))
    existing_claim_ids = set(
        session.execute(
            select(Claim.id).where(Claim.id.in_(claim_ids))
        ).scalars().all()
    )
    for claim_id in claim_ids:
        if claim_id not in existing_claim_ids:
            raise ClaimNotFoundError(claim_id)

    # In-batch identity dedup: the first occurrence of each identity wins;
    # later duplicates are dropped from both the writes and the response.
    unique_items: list[EvidenceBundleCreate] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in payload.items:
        identity = _evidence_bundle_identity(item)
        if identity in seen:
            continue
        seen.add(identity)
        unique_items.append(item)

    while True:
        existing: dict[tuple[str, str, str, str], EvidenceBundle] = {}
        rows = session.execute(
            select(EvidenceBundle).where(EvidenceBundle.claim_id.in_(claim_ids))
        ).scalars().all()
        for row in rows:
            existing[
                (row.claim_id, row.evidence_type, row.digest_algorithm, row.digest_hex)
            ] = row

        results: list[EvidenceBundle] = []
        new_bundles: list[EvidenceBundle] = []
        for item in unique_items:
            bundle = existing.get(_evidence_bundle_identity(item))
            if bundle is None:
                bundle = EvidenceBundle(
                    id=ids.evidence_bundle_id(
                        item.claim_id,
                        item.evidence_type,
                        item.digest_algorithm,
                        item.digest_hex,
                    ),
                    claim_id=item.claim_id,
                    evidence_type=item.evidence_type,
                    digest_algorithm=item.digest_algorithm,
                    digest_hex=item.digest_hex,
                    media_type=item.media_type,
                    metadata_=item.metadata,
                )
                new_bundles.append(bundle)
            results.append(bundle)

        if not new_bundles:
            # Every identity already existed: the batch writes nothing and
            # adds no audit event.
            return results, False

        for bundle in new_bundles:
            session.add(bundle)
            session.add(
                AuditEvent(
                    event_type=EVENT_EVIDENCE_BUNDLE_CREATED,
                    resource_id=bundle.id,
                )
            )
        try:
            session.commit()
        except IntegrityError:
            # A concurrent import created at least one of these identities:
            # roll back and re-resolve; the raced identities now exist and
            # only the still-missing ones are retried.
            session.rollback()
            continue
        for bundle in new_bundles:
            session.refresh(bundle)
        return results, True


def get_evidence_bundle(
    session: Session, evidence_bundle_id: str
) -> EvidenceBundle:
    """Return an evidence bundle by id or raise :class:`EvidenceBundleNotFoundError`."""
    bundle = session.execute(
        select(EvidenceBundle).where(EvidenceBundle.id == evidence_bundle_id)
    ).scalar_one_or_none()
    if bundle is None:
        raise EvidenceBundleNotFoundError(evidence_bundle_id)
    return bundle


def find_evidence_bundle(
    session: Session, evidence_bundle_id: str
) -> EvidenceBundle | None:
    """Return the evidence bundle with this id, or ``None`` when none exists.

    Strictly read-only: it writes no resource and no audit event.
    """
    return session.execute(
        select(EvidenceBundle).where(EvidenceBundle.id == evidence_bundle_id)
    ).scalar_one_or_none()


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


DEFAULT_CONTENT_EVIDENCE_LIMIT = 50
MIN_CONTENT_EVIDENCE_LIMIT = 1
MAX_CONTENT_EVIDENCE_LIMIT = 100


def list_evidence_bundles_for_content(
    session: Session,
    content_id: str,
    evidence_type: str | None = None,
    media_type: str | None = None,
) -> list[EvidenceBundle]:
    """Return evidence bundles attached to a content's claims.

    Only bundles linked through claims that directly assert this exact
    content are returned: evidence attached to a different content
    reachable through lineage relations is not -- the listing performs no
    graph traversal. Optional ``evidence_type``/``media_type`` are exact,
    combinable string matches. Results follow the bundles' stable creation
    order (``created_at`` with the monotonic ``seq`` tiebreaker).

    The content must exist; an unknown content id is a missing resource,
    not an empty collection.
    """
    _require_content(session, content_id)
    stmt = (
        select(EvidenceBundle)
        .join(Claim, EvidenceBundle.claim_id == Claim.id)
        .where(Claim.content_id == content_id)
    )
    if evidence_type is not None:
        stmt = stmt.where(EvidenceBundle.evidence_type == evidence_type)
    if media_type is not None:
        stmt = stmt.where(EvidenceBundle.media_type == media_type)
    stmt = stmt.order_by(*_EVIDENCE_BUNDLE_ORDER)
    return list(session.execute(stmt).scalars().all())


# Reviewer evidence-bundle search paging bounds.
DEFAULT_EVIDENCE_BUNDLES_LIMIT = 50
MIN_EVIDENCE_BUNDLES_LIMIT = 1
MAX_EVIDENCE_BUNDLES_LIMIT = 100


def list_evidence_bundles(
    session: Session,
    claim_id: str | None = None,
    evidence_type: str | None = None,
    media_type: str | None = None,
    digest_hex: str | None = None,
) -> list[EvidenceBundle]:
    """Return all existing evidence bundles in stable creation order, filtered.

    Every filter is an exact, case- and whitespace-sensitive string match
    applied as logical AND; ``None`` means unfiltered. ``digest_hex`` is the
    strict 64-lowercase-hex spelling already enforced at the boundary.
    Results follow the bundles' stable creation order (``created_at`` with
    the monotonic ``seq`` tiebreaker). The search is strictly read-only: it
    writes no bundle, resource, or audit event, and no referenced claim
    existence is required, so a filter that matches nothing is an empty
    result rather than a missing resource. Evidence bytes are never stored
    and therefore never appear here.
    """
    stmt = select(EvidenceBundle)
    if claim_id is not None:
        stmt = stmt.where(EvidenceBundle.claim_id == claim_id)
    if evidence_type is not None:
        stmt = stmt.where(EvidenceBundle.evidence_type == evidence_type)
    if media_type is not None:
        stmt = stmt.where(EvidenceBundle.media_type == media_type)
    if digest_hex is not None:
        stmt = stmt.where(EvidenceBundle.digest_hex == digest_hex)
    stmt = stmt.order_by(*_EVIDENCE_BUNDLE_ORDER)
    return list(session.execute(stmt).scalars().all())


def _attestation_identity_select(
    payload: AttestationCreate, signature_digest_hex: str
):
    return select(Attestation).where(
        Attestation.target_type == payload.target_type,
        Attestation.target_id == payload.target_id,
        Attestation.signer_actor_id == payload.signer_actor_id,
        Attestation.public_key == payload.public_key,
        Attestation.signature_digest_hex == signature_digest_hex,
    )


def create_attestation(
    session: Session, payload: AttestationCreate
) -> tuple[Attestation, bool]:
    """Create a verified attestation, returning ``(attestation, created)``.

    The attested target (an existing claim or evidence bundle) and the
    signing actor must already exist. The Ed25519 signature is verified
    against the canonical message
    ``["provenance-attestation-v1", target_type, target_id,
    signer_actor_id]``; a signature that cannot be verified is rejected and
    nothing is written. Only the SHA-256 digest of the signature is stored
    -- never the raw signature.

    A repeat submission for the same target, signing actor, public key, and
    signature digest returns the existing attestation with ``created=False``
    and writes no row or audit event. On first creation, the attestation row
    and its ``attestation.created`` audit event commit in a single
    transaction.
    """
    # Missing target is a missing resource, distinct from validation errors;
    # check the polymorphic target first, then the signing actor.
    if payload.target_type == signing.TARGET_CLAIM:
        target = session.execute(
            select(Claim).where(Claim.id == payload.target_id)
        ).scalar_one_or_none()
        if target is None:
            raise ClaimNotFoundError(payload.target_id)
    else:
        target = session.execute(
            select(EvidenceBundle).where(
                EvidenceBundle.id == payload.target_id
            )
        ).scalar_one_or_none()
        if target is None:
            raise EvidenceBundleNotFoundError(payload.target_id)

    actor = get_actor_by_id(session, payload.signer_actor_id)
    if actor is None:
        raise UnknownActorError(payload.signer_actor_id)

    signature_digest_hex = hashlib.sha256(payload.signature).hexdigest()

    # An existing record with this exact identity was verified when it was
    # first written; a repeat submission is idempotent and needs no row,
    # audit event, or re-verification.
    existing = session.execute(
        _attestation_identity_select(payload, signature_digest_hex)
    ).scalar_one_or_none()
    if existing is not None:
        return existing, False

    message = signing.attestation_message_bytes(
        payload.target_type, payload.target_id, payload.signer_actor_id
    )
    if not ed25519.verify(payload.public_key, message, payload.signature):
        raise AttestationVerificationError(
            details={"reason": "signature_verification_failed"}
        )

    attestation = Attestation(
        id=ids.attestation_id(
            payload.target_type,
            payload.target_id,
            payload.signer_actor_id,
            payload.public_key.hex(),
            signature_digest_hex,
        ),
        target_type=payload.target_type,
        target_id=payload.target_id,
        signer_actor_id=payload.signer_actor_id,
        public_key=payload.public_key,
        signature_digest_algorithm=canonical.CANONICAL_DIGEST_ALGORITHM,
        signature_digest_hex=signature_digest_hex,
    )
    session.add(attestation)
    session.add(
        AuditEvent(
            event_type=EVENT_ATTESTATION_CREATED, resource_id=attestation.id
        )
    )
    try:
        session.commit()
    except IntegrityError:
        # Concurrent identical attestation won the race: return its resource.
        session.rollback()
        raced = session.execute(
            _attestation_identity_select(payload, signature_digest_hex)
        ).scalar_one_or_none()
        if raced is None:  # pragma: no cover - defensive
            raise
        return raced, False
    session.refresh(attestation)
    return attestation, True


def get_attestation(session: Session, attestation_id: str) -> Attestation:
    """Return an attestation by id or raise :class:`AttestationNotFoundError`."""
    attestation = session.execute(
        select(Attestation).where(Attestation.id == attestation_id)
    ).scalar_one_or_none()
    if attestation is None:
        raise AttestationNotFoundError(attestation_id)
    return attestation


def list_attestations(
    session: Session,
    target_type: str | None = None,
    target_id: str | None = None,
) -> list[Attestation]:
    """Return attestations in stable creation order.

    Optionally filtered by ``target_type`` (``"claim"`` /
    ``"evidence_bundle"``) and/or ``target_id``. Filtering by a target with
    no attestations is an empty collection, not a missing resource.
    """
    stmt = select(Attestation)
    if target_type is not None:
        stmt = stmt.where(Attestation.target_type == target_type)
    if target_id is not None:
        stmt = stmt.where(Attestation.target_id == target_id)
    stmt = stmt.order_by(*_ATTESTATION_ORDER)
    return list(session.execute(stmt).scalars().all())


def _revocation_identity_select(payload: AttestationRevocationCreate):
    return select(AttestationRevocation).where(
        AttestationRevocation.attestation_id == payload.attestation_id,
        AttestationRevocation.revoker_actor_id == payload.revoker_actor_id,
        AttestationRevocation.reason == payload.reason,
    )


def create_attestation_revocation(
    session: Session, payload: AttestationRevocationCreate
) -> tuple[AttestationRevocation, bool]:
    """Create an immutable attestation revocation, returning ``(record, created)``.

    The revoked attestation and the revoking actor must already exist; the
    revoking actor need not be the attestation's signer. The attestation is
    never mutated or deleted -- only an append-only revocation record is
    added. A repeat submission of the same attestation, revoking actor, and
    (trimmed) reason returns the existing record with ``created=False`` and
    writes no row or audit event. Any other field combination forms an
    independent record. On first creation the revocation row and its
    ``attestation.revoked`` audit event commit in a single transaction.
    """
    # Missing references are missing resources, distinct from validation
    # errors; check the attestation first, then the revoking actor.
    attestation = session.execute(
        select(Attestation).where(Attestation.id == payload.attestation_id)
    ).scalar_one_or_none()
    if attestation is None:
        raise AttestationNotFoundError(payload.attestation_id)

    actor = get_actor_by_id(session, payload.revoker_actor_id)
    if actor is None:
        raise UnknownActorError(payload.revoker_actor_id)

    existing = session.execute(
        _revocation_identity_select(payload)
    ).scalar_one_or_none()
    if existing is not None:
        return existing, False

    revocation = AttestationRevocation(
        id=ids.attestation_revocation_id(
            payload.attestation_id,
            payload.revoker_actor_id,
            payload.reason,
        ),
        attestation_id=payload.attestation_id,
        revoker_actor_id=payload.revoker_actor_id,
        reason=payload.reason,
    )
    session.add(revocation)
    session.add(
        AuditEvent(
            event_type=EVENT_ATTESTATION_REVOKED, resource_id=revocation.id
        )
    )
    try:
        session.commit()
    except IntegrityError:
        # Concurrent identical revocation won the race: return its record.
        session.rollback()
        raced = session.execute(
            _revocation_identity_select(payload)
        ).scalar_one_or_none()
        if raced is None:  # pragma: no cover - defensive
            raise
        return raced, False
    session.refresh(revocation)
    return revocation, True


def get_attestation_revocation(
    session: Session, revocation_id: str
) -> AttestationRevocation:
    """Return a revocation by id or raise
    :class:`AttestationRevocationNotFoundError`."""
    revocation = session.execute(
        select(AttestationRevocation).where(
            AttestationRevocation.id == revocation_id
        )
    ).scalar_one_or_none()
    if revocation is None:
        raise AttestationRevocationNotFoundError(revocation_id)
    return revocation


def list_revocations_for_attestation(
    session: Session, attestation_id: str
) -> list[AttestationRevocation]:
    """Return revocation records for one attestation in stable creation order.

    The attestation must exist; an unknown attestation id is a missing
    resource, not an empty collection.
    """
    attestation = session.execute(
        select(Attestation).where(Attestation.id == attestation_id)
    ).scalar_one_or_none()
    if attestation is None:
        raise AttestationNotFoundError(attestation_id)
    stmt = (
        select(AttestationRevocation)
        .where(AttestationRevocation.attestation_id == attestation_id)
        .order_by(*_ATTESTATION_REVOCATION_ORDER)
    )
    return list(session.execute(stmt).scalars().all())


def _grant_identity_select(payload: AttestationAccessGrantCreate):
    return select(AttestationAccessGrant).where(
        AttestationAccessGrant.attestation_id == payload.attestation_id,
        AttestationAccessGrant.grantee_actor_id == payload.grantee_actor_id,
    )


def create_attestation_access_grant(
    session: Session,
    payload: AttestationAccessGrantCreate,
    caller_actor_id: str,
) -> tuple[AttestationAccessGrant, bool]:
    """Create a read-only proof-access grant, returning ``(grant, created)``.

    The authenticated caller must be the ``signer_actor_id`` of the existing
    attestation; the grantee must be an existing actor. Every rejected
    request -- a missing attestation, a missing grantee, or a caller who is
    not the signer -- is a ``422`` validation error and writes nothing. A
    repeat submission for the same ``(attestation_id, grantee_actor_id)``
    pair returns the original grant with ``created=False`` and no audit
    event; a different grantee forms an independent, immutable grant. On
    first creation the grant row and its ``attestation.access_granted``
    audit event commit in a single transaction.
    """
    attestation = session.execute(
        select(Attestation).where(Attestation.id == payload.attestation_id)
    ).scalar_one_or_none()
    if attestation is None:
        raise ProtectedAccessValidationError("attestation_not_found")

    if attestation.signer_actor_id != caller_actor_id:
        raise ProtectedAccessValidationError("caller_not_signer")

    grantee = get_actor_by_id(session, payload.grantee_actor_id)
    if grantee is None:
        raise ProtectedAccessValidationError("unknown_grantee_actor")

    existing = session.execute(
        _grant_identity_select(payload)
    ).scalar_one_or_none()
    if existing is not None:
        return existing, False

    grant = AttestationAccessGrant(
        id=ids.attestation_access_grant_id(
            payload.attestation_id, payload.grantee_actor_id
        ),
        attestation_id=payload.attestation_id,
        grantee_actor_id=payload.grantee_actor_id,
    )
    session.add(grant)
    session.add(
        AuditEvent(
            event_type=EVENT_ATTESTATION_ACCESS_GRANTED, resource_id=grant.id
        )
    )
    try:
        session.commit()
    except IntegrityError:
        # Concurrent identical grant won the race: return its record.
        session.rollback()
        raced = session.execute(
            _grant_identity_select(payload)
        ).scalar_one_or_none()
        if raced is None:  # pragma: no cover - defensive
            raise
        return raced, False
    session.refresh(grant)
    return grant, True


def create_attestation_access_grant_revocation(
    session: Session,
    payload: AttestationAccessGrantRevocationCreate,
    caller_actor_id: str,
) -> tuple[AttestationAccessGrantRevocation, bool]:
    """Revoke an existing read-access grant, returning ``(record, created)``.

    The authenticated caller must be the ``signer_actor_id`` of the
    attestation the grant concerns; the grant must already exist. Every
    rejected request -- an unknown grant or a caller who is not the signer --
    is a ``422`` validation error and writes nothing. The grant and the
    attestation are never mutated or deleted: only an append-only revocation
    record is added.

    A repeat submission for the same grant, revoking actor, and (trimmed)
    reason returns the existing record with ``created=False`` and writes no
    row or audit event; a different reason forms an independent, immutable
    record. On first creation the revocation row and its
    ``attestation.access_grant_revoked`` audit event commit in a single
    transaction. No private key, raw signature, claim payload, content, or
    evidence byte is ever stored.
    """
    grant = session.execute(
        select(AttestationAccessGrant).where(
            AttestationAccessGrant.id == payload.grant_id
        )
    ).scalar_one_or_none()
    if grant is None:
        raise ProtectedAccessValidationError("grant_not_found")

    attestation = session.execute(
        select(Attestation).where(Attestation.id == grant.attestation_id)
    ).scalar_one_or_none()
    # A stored grant always references a stored attestation; the guard
    # preserves the 422 contract even against anomalous history.
    if attestation is None:  # pragma: no cover - defensive
        raise ProtectedAccessValidationError("grant_not_found")

    if attestation.signer_actor_id != caller_actor_id:
        raise ProtectedAccessValidationError("caller_not_signer")

    existing = session.execute(
        select(AttestationAccessGrantRevocation).where(
            AttestationAccessGrantRevocation.grant_id == payload.grant_id,
            AttestationAccessGrantRevocation.revoker_actor_id == caller_actor_id,
            AttestationAccessGrantRevocation.reason == payload.reason,
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing, False

    revocation = AttestationAccessGrantRevocation(
        id=ids.attestation_access_grant_revocation_id(
            payload.grant_id, caller_actor_id, payload.reason
        ),
        grant_id=payload.grant_id,
        revoker_actor_id=caller_actor_id,
        reason=payload.reason,
    )
    session.add(revocation)
    session.add(
        AuditEvent(
            event_type=EVENT_ATTESTATION_ACCESS_GRANT_REVOKED,
            resource_id=revocation.id,
        )
    )
    try:
        session.commit()
    except IntegrityError:
        # Concurrent identical revocation won the race: return its record.
        session.rollback()
        raced = session.execute(
            select(AttestationAccessGrantRevocation).where(
                AttestationAccessGrantRevocation.grant_id == payload.grant_id,
                AttestationAccessGrantRevocation.revoker_actor_id
                == caller_actor_id,
                AttestationAccessGrantRevocation.reason == payload.reason,
            )
        ).scalar_one_or_none()
        if raced is None:  # pragma: no cover - defensive
            raise
        return raced, False
    session.refresh(revocation)
    return revocation, True


def get_attestation_access_grant_revocation(
    session: Session, revocation_id: str
) -> AttestationAccessGrantRevocation:
    """Return one existing grant revocation by its own stable id.

    The revocation id is the only lookup key: the record is never resolved
    by grant, revoker, or reason. An unknown id is an explicit, specific
    404 carrying the requested id. The function is strictly read-only: it
    writes no revocation, grant, resource, or audit event.
    """
    revocation = session.execute(
        select(AttestationAccessGrantRevocation).where(
            AttestationAccessGrantRevocation.id == revocation_id
        )
    ).scalar_one_or_none()
    if revocation is None:
        raise AttestationAccessGrantRevocationNotFoundError(revocation_id)
    return revocation


def list_revocations_for_access_grant(
    session: Session, grant_id: str
) -> list[AttestationAccessGrantRevocation]:
    """Return one existing grant's revocation records in creation order.

    The grant id is the only lookup key: records are selected by their
    ``grant_id`` column alone, never by revoker, reason, or any attestation
    field. The grant must already exist; an unknown grant id is a
    ``422`` validation error (``grant_not_found``), the same boundary as the
    revocation write route -- never an empty collection and never a reverse
    lookup. An existing grant without revocations yields an empty list.
    Results follow stable creation order (``created_at`` with the monotonic
    ``seq`` tiebreaker). The function is strictly read-only: it writes no
    revocation, grant, resource, or audit event.
    """
    grant = session.execute(
        select(AttestationAccessGrant.id).where(
            AttestationAccessGrant.id == grant_id
        )
    ).first()
    if grant is None:
        raise ProtectedAccessValidationError("grant_not_found")
    stmt = (
        select(AttestationAccessGrantRevocation)
        .where(AttestationAccessGrantRevocation.grant_id == grant_id)
        .order_by(*_ATTESTATION_ACCESS_GRANT_REVOCATION_ORDER)
    )
    return list(session.execute(stmt).scalars().all())


def get_accessible_attestation(
    session: Session, attestation_id: str, actor_id: str
) -> Attestation | None:
    """Return the attestation iff ``actor_id`` may read it, else ``None``.

    The attestation's signer and any grantee holding an access grant for
    this exact attestation that carries no revocation may read it. A grant
    with any recorded revocation no longer authorizes its grantee, even
    though the grant row is retained. The function is strictly read-only: it
    performs no resource or audit writes.
    """
    attestation = session.execute(
        select(Attestation).where(Attestation.id == attestation_id)
    ).scalar_one_or_none()
    if attestation is None:
        return None
    if attestation.signer_actor_id == actor_id:
        return attestation
    revoked_grant = exists().where(
        AttestationAccessGrantRevocation.grant_id == AttestationAccessGrant.id
    )
    grant_exists = session.execute(
        select(AttestationAccessGrant.id).where(
            AttestationAccessGrant.attestation_id == attestation_id,
            AttestationAccessGrant.grantee_actor_id == actor_id,
            ~revoked_grant,
        )
    ).first()
    return attestation if grant_exists is not None else None


DEFAULT_ATTESTATION_ACCESS_GRANTS_LIMIT = 50
MIN_ATTESTATION_ACCESS_GRANTS_LIMIT = 1
MAX_ATTESTATION_ACCESS_GRANTS_LIMIT = 100


def list_access_grants_for_attestation(
    session: Session, attestation_id: str, actor_id: str
) -> list[AttestationAccessGrant] | None:
    """Return one attestation's access grants for its signer, in creation order.

    The caller must be the authenticated ``signer_actor_id`` of an existing
    attestation; a missing attestation and a caller who is not its signer are
    indistinguishable to the caller and both return ``None``, so the route
    renders one opaque 404 and never reveals existence. On success every
    grant row of the attestation is returned -- grants are append-only and
    retained even after a revocation -- in stable creation order
    (``created_at`` with the monotonic ``seq`` tiebreaker). An attestation
    without grants yields an empty list. The function is strictly read-only:
    it writes no grant, revocation, resource, or audit event.
    """
    attestation = session.execute(
        select(Attestation).where(Attestation.id == attestation_id)
    ).scalar_one_or_none()
    if attestation is None or attestation.signer_actor_id != actor_id:
        # A missing proof and a caller who is not its signer are
        # indistinguishable to the caller: both collapse into the route's
        # single opaque 404.
        return None

    stmt = (
        select(AttestationAccessGrant)
        .where(AttestationAccessGrant.attestation_id == attestation_id)
        .order_by(*_ATTESTATION_ACCESS_GRANT_ORDER)
    )
    return list(session.execute(stmt).scalars().all())


DEFAULT_AUTHENTICATION_KEY_ROTATIONS_LIMIT = 50
MIN_AUTHENTICATION_KEY_ROTATIONS_LIMIT = 1
MAX_AUTHENTICATION_KEY_ROTATIONS_LIMIT = 100

_AUTHENTICATION_KEY_ROTATION_ORDER = (
    AuthenticationKeyRotation.created_at.asc(),
    AuthenticationKeyRotation.seq.asc(),
)


def list_authentication_key_rotations_for_actor(
    session: Session, actor_id: str
) -> list[AuthenticationKeyRotation]:
    """Return one existing subject's rotations in stable creation order.

    Results follow the rotations' stable creation order (``created_at`` with
    the monotonic ``seq`` tiebreaker). The subject must exist; an unknown
    actor is a missing resource (``unknown_actor`` 404), not an empty
    collection. The function is strictly read-only: it writes no rotation,
    resource, or audit event.
    """
    if get_actor_by_id(session, actor_id) is None:
        raise UnknownActorError(actor_id)
    stmt = (
        select(AuthenticationKeyRotation)
        .where(AuthenticationKeyRotation.actor_id == actor_id)
        .order_by(*_AUTHENTICATION_KEY_ROTATION_ORDER)
    )
    return list(session.execute(stmt).scalars().all())


def create_authentication_key_rotation(
    session: Session,
    payload: AuthenticationKeyRotationCreate,
    caller_actor_id: str,
) -> tuple[AuthenticationKeyRotation, bool]:
    """Rotate in a new authentication public key, returning ``(record, created)``.

    The authenticated caller is the subject: ``payload.actor_id`` must be the
    caller and the subject must already exist. The new key immediately
    joins the subject's non-revoked authentication key set for the
    protected routes without any attestation. A repeat submission for the
    same subject and public key returns the original record with
    ``created=False`` and writes no row or audit event -- including a record
    that has since been retired, which is returned unchanged. A different
    public key forms an independent rotation. On first creation the row and
    its ``authentication_key.rotated`` audit event commit in a single
    transaction. Every rejected request is a ``422`` validation error and
    writes nothing.
    """
    # Validate the subject before the credential-derived identity lookup:
    # an unknown subject and a caller/body mismatch are 422s, not 404s.
    actor = get_actor_by_id(session, payload.actor_id)
    if actor is None:
        raise ProtectedAccessValidationError("unknown_actor")

    if payload.actor_id != caller_actor_id:
        raise ProtectedAccessValidationError("actor_mismatch")

    existing = session.execute(
        select(AuthenticationKeyRotation).where(
            AuthenticationKeyRotation.actor_id == payload.actor_id,
            AuthenticationKeyRotation.public_key == payload.new_public_key,
        )
    ).scalar_one_or_none()
    if existing is not None:
        # Idempotent even after retirement: the original (possibly retired)
        # record is returned and no audit event is written.
        return existing, False

    rotation = AuthenticationKeyRotation(
        id=ids.authentication_key_rotation_id(
            payload.actor_id, payload.new_public_key.hex()
        ),
        actor_id=payload.actor_id,
        public_key=payload.new_public_key,
        active=True,
    )
    session.add(rotation)
    session.add(
        AuditEvent(
            event_type=EVENT_AUTHENTICATION_KEY_ROTATED,
            resource_id=rotation.id,
        )
    )
    try:
        session.commit()
    except IntegrityError:
        # Concurrent identical rotation won the race: return its record.
        session.rollback()
        raced = session.execute(
            select(AuthenticationKeyRotation).where(
                AuthenticationKeyRotation.actor_id == payload.actor_id,
                AuthenticationKeyRotation.public_key == payload.new_public_key,
            )
        ).scalar_one_or_none()
        if raced is None:  # pragma: no cover - defensive
            raise
        return raced, False
    session.refresh(rotation)
    return rotation, True


def retire_authentication_key_rotation(
    session: Session,
    rotation_id: str,
    caller_actor_id: str,
) -> AuthenticationKeyRotation:
    """Retire an active key rotation owned by the caller.

    Only the owning subject may retire the record, and only while it is
    active. An unknown record, an owner mismatch, or a repeat retirement of
    an already-retired record is a ``422`` validation error and writes
    nothing -- existence is never revealed to a non-owner. On success the
    record's ``active`` flips to false and a UTC ``retired_at`` is stamped
    in the same transaction as the ``authentication_key.retired`` audit
    event; the key then stops authenticating immediately.

    The active check and the state flip are a single conditional UPDATE,
    so two concurrent retire calls cannot both succeed and write two
    audit events: exactly one flips the row, the other is rejected.
    """
    rotation = session.execute(
        select(AuthenticationKeyRotation).where(
            AuthenticationKeyRotation.id == rotation_id
        )
    ).scalar_one_or_none()
    if rotation is None:
        raise ProtectedAccessValidationError("rotation_not_found")

    if rotation.actor_id != caller_actor_id:
        # Collapse owner mismatch into the same opaque validation error as
        # an unknown record, without revealing that the record exists.
        raise ProtectedAccessValidationError("rotation_not_found")

    retired_at = utc_now()
    result = session.execute(
        sa_update(AuthenticationKeyRotation)
        .where(
            AuthenticationKeyRotation.id == rotation_id,
            AuthenticationKeyRotation.active.is_(True),
        )
        .values(active=False, retired_at=retired_at)
    )
    if result.rowcount != 1:
        # The row was active at load time but a concurrent transaction
        # retired it first: this request is a repeat retirement, not a
        # success, so it writes no state and no audit event.
        session.rollback()
        raise ProtectedAccessValidationError("rotation_already_retired")

    session.add(
        AuditEvent(
            event_type=EVENT_AUTHENTICATION_KEY_RETIRED,
            resource_id=rotation.id,
        )
    )
    session.commit()
    # The core UPDATE bypassed the ORM unit of work; reload the final state
    # onto the identity-map object before returning it.
    session.refresh(rotation)
    return rotation


def get_content_export(
    session: Session, content_id: str
) -> tuple[Content, list[Claim], dict[str, list[EvidenceBundle]]]:
    """Return a content's transferable provenance-evidence snapshot.

    The result is ``(content, claims, bundles_by_claim_id)``: the content
    itself, the claims that directly assert this exact content in stable
    creation order, and each claim's evidence bundles in their own stable
    creation order keyed by claim id (a claim without bundles maps to an
    empty list). No lineage traversal is performed: claims or bundles of
    related contents are never included. The function is strictly read-only:
    it writes no resource and no audit event.

    The content must exist; an unknown content id is a missing resource, not
    an empty export.
    """
    content = _require_content(session, content_id)
    claims = list(
        session.execute(
            select(Claim)
            .where(Claim.content_id == content_id)
            .order_by(*_CLAIM_ORDER)
        )
        .scalars()
        .all()
    )
    bundles_by_claim: dict[str, list[EvidenceBundle]] = {
        claim.id: [] for claim in claims
    }
    if claims:
        bundles = session.execute(
            select(EvidenceBundle)
            .join(Claim, EvidenceBundle.claim_id == Claim.id)
            .where(Claim.content_id == content_id)
            .order_by(*_EVIDENCE_BUNDLE_ORDER)
        ).scalars().all()
        for bundle in bundles:
            bundles_by_claim[bundle.claim_id].append(bundle)
    return content, claims, bundles_by_claim


#: Builds the stored export result for one content: the wire-shaped
#: ``{"content", "claims"}`` snapshot served by the read-only export route.
#: Injected by the API layer so the service stays free of response schemas
#: and the failure path remains independently exercisable in tests.
ExportResultBuilder = Callable[[Session, str], dict]


def create_content_export_job(
    session: Session, payload: ContentExportJobCreate
) -> tuple[ContentExportJob, bool]:
    """Register one asynchronous content export job, returning ``(job, created)``.

    The referenced content must exist; an unknown content is a missing
    resource. ``request_id`` is the client idempotency key and is unique. A
    repeat submission for the same ``request_id`` and ``content_id`` returns
    the existing job with ``created=False`` and writes no row or audit
    event, regardless of the job's current lifecycle state. The same
    ``request_id`` reused for a *different* ``content_id`` is a
    :class:`ContentExportRequestConflictError` and writes nothing.

    A first submission creates the job ``pending`` with null
    ``started_at``/``finished_at``/``result``/``error``; the job row and its
    ``content_export_job.created`` audit event commit in a single
    transaction.
    """
    content = session.execute(
        select(Content).where(Content.id == payload.content_id)
    ).scalar_one_or_none()
    if content is None:
        raise ContentNotFoundError(payload.content_id)

    existing = session.execute(
        select(ContentExportJob).where(
            ContentExportJob.request_id == payload.request_id
        )
    ).scalar_one_or_none()
    if existing is not None:
        if existing.content_id != payload.content_id:
            raise ContentExportRequestConflictError(payload.request_id)
        return existing, False

    job = ContentExportJob(
        id=ids.content_export_job_id(payload.content_id, payload.request_id),
        content_id=payload.content_id,
        request_id=payload.request_id,
        status=CONTENT_EXPORT_JOB_PENDING,
    )
    session.add(job)
    session.add(
        AuditEvent(
            event_type=EVENT_CONTENT_EXPORT_JOB_CREATED, resource_id=job.id
        )
    )
    try:
        session.commit()
    except IntegrityError:
        # A concurrent request registered this request_id first: roll back
        # and reconcile against that job instead of duplicating it or
        # writing a second audit event.
        session.rollback()
        raced = session.execute(
            select(ContentExportJob).where(
                ContentExportJob.request_id == payload.request_id
            )
        ).scalar_one_or_none()
        if raced is None:  # pragma: no cover - defensive
            raise
        if raced.content_id != payload.content_id:
            raise ContentExportRequestConflictError(
                payload.request_id
            ) from None
        return raced, False
    session.refresh(job)
    return job, True


def get_content_export_job(session: Session, job_id: str) -> ContentExportJob:
    """Return a content export job by id or raise the 404 domain error.

    Strictly read-only: the read writes no job and no audit event.
    """
    job = session.execute(
        select(ContentExportJob).where(ContentExportJob.id == job_id)
    ).scalar_one_or_none()
    if job is None:
        raise ContentExportJobNotFoundError(job_id)
    return job


def run_content_export_job(
    session: Session,
    job_id: str,
    build_result: ExportResultBuilder,
) -> ContentExportJob:
    """Atomically claim a pending export job (looked up by id) and run it.

    The job must exist or :class:`ContentExportJobNotFoundError` is raised.
    Claim and settlement follow :func:`_claim_and_run_content_export_job`:
    only a ``pending`` job can be claimed, so a concurrent or repeated run is
    a :class:`ContentExportJobConflictError` rather than a second execution.
    """
    job = session.execute(
        select(ContentExportJob).where(ContentExportJob.id == job_id)
    ).scalar_one_or_none()
    if job is None:
        raise ContentExportJobNotFoundError(job_id)
    return _claim_and_run_content_export_job(session, job, build_result)


def run_next_content_export_job(
    session: Session,
    build_result: ExportResultBuilder,
) -> ContentExportJob:
    """Claim the oldest pending export job and run it to completion.

    The queue candidate is the single oldest job still ``pending`` under the
    stable creation order (``created_at`` with the monotonic ``seq``
    tiebreaker, which survives restarts); other pending jobs are left
    untouched in the queue. When no job is pending,
    :class:`ContentExportJobNotFoundError` is raised with zero writes: no job
    is claimed or created and no audit event is recorded.

    The selected job is claimed and settled through exactly the same atomic
    path as :func:`run_content_export_job`. If a concurrent caller claims the
    same oldest job (or it otherwise stops being pending) between selection
    and the claim, the conditional update matches zero rows and this call
    raises :class:`ContentExportJobConflictError` without modifying any job,
    creating any resource, or writing an audit event; this loser never falls
    through to a different (younger) pending job.
    """
    job = session.execute(
        select(ContentExportJob)
        .where(ContentExportJob.status == CONTENT_EXPORT_JOB_PENDING)
        .order_by(*_CONTENT_EXPORT_JOB_ORDER)
        .limit(1)
    ).scalar_one_or_none()
    if job is None:
        # An empty queue is a missing runnable resource, not an execution:
        # nothing is claimed, created, or written and no audit event is
        # recorded.
        raise ContentExportJobNotFoundError()
    return _claim_and_run_content_export_job(session, job, build_result)


def _claim_and_run_content_export_job(
    session: Session,
    job: ContentExportJob,
    build_result: ExportResultBuilder,
) -> ContentExportJob:
    """Atomically flip one already-selected job pending->running and settle it.

    The pending-to-running transition is a single conditional
    ``UPDATE ... WHERE status = 'pending'``, so exactly one concurrent run
    wins and any other request -- a concurrent loser, or a repeat run after
    the job has settled -- observes zero updated rows and raises
    :class:`ContentExportJobConflictError` without writing a state change or
    audit event.

    The winning run stamps UTC ``started_at`` while claiming, builds the
    existing read-only content export, and settles the job in the SAME
    transaction: on success it stamps UTC ``finished_at`` and stores the
    ``{"content", "claims"}`` snapshot as ``result`` with status
    ``succeeded``; if the export raises, it stamps UTC ``finished_at``,
    leaves ``result`` null, records the stable ``content_export_failed``
    error, and sets status ``failed``. Either settlement writes the
    ``content_export_job.run`` audit event in that same transaction.
    """
    job_id = job.id
    # Atomic claim: the status predicate makes the pending->running flip a
    # compare-and-set. Row locking serializes concurrent writers; the loser
    # (the row is no longer pending) updates zero rows.
    started_at = utc_now()
    claimed = session.execute(
        sa_update(ContentExportJob)
        .where(
            ContentExportJob.id == job_id,
            ContentExportJob.status == CONTENT_EXPORT_JOB_PENDING,
        )
        .values(status=CONTENT_EXPORT_JOB_RUNNING, started_at=started_at)
    )
    if claimed.rowcount != 1:
        # A concurrent run claimed it first, or it already settled: this
        # request is a conflict, not an execution, so it writes no state and
        # no audit event. Reload to report the job's current status.
        session.rollback()
        current = session.execute(
            select(ContentExportJob).where(ContentExportJob.id == job_id)
        ).scalar_one_or_none()
        current_status = (
            current.status if current is not None else CONTENT_EXPORT_JOB_PENDING
        )
        raise ContentExportJobConflictError(job_id, current_status)

    try:
        result = build_result(session, job.content_id)
    except Exception:
        # The export failed: settle as failed in the SAME transaction that
        # holds the claim, so the job is never left stuck in ``running``.
        finished_at = utc_now()
        session.execute(
            sa_update(ContentExportJob)
            .where(ContentExportJob.id == job_id)
            .values(
                status=CONTENT_EXPORT_JOB_FAILED,
                finished_at=finished_at,
                result=None,
                error=CONTENT_EXPORT_JOB_FAILED_ERROR,
            )
        )
        session.add(
            AuditEvent(
                event_type=EVENT_CONTENT_EXPORT_JOB_RUN, resource_id=job_id
            )
        )
        session.commit()
        session.refresh(job)
        return job

    finished_at = utc_now()
    session.execute(
        sa_update(ContentExportJob)
        .where(ContentExportJob.id == job_id)
        .values(
            status=CONTENT_EXPORT_JOB_SUCCEEDED,
            finished_at=finished_at,
            result=result,
        )
    )
    session.add(
        AuditEvent(event_type=EVENT_CONTENT_EXPORT_JOB_RUN, resource_id=job_id)
    )
    session.commit()
    # The Core UPDATEs bypassed the ORM unit of work; reload the final state
    # onto the identity-map object before returning it.
    session.refresh(job)
    return job


# Reviewer content export job search paging bounds.
DEFAULT_CONTENT_EXPORT_JOBS_LIMIT = 50
MIN_CONTENT_EXPORT_JOBS_LIMIT = 1
MAX_CONTENT_EXPORT_JOBS_LIMIT = 100


def list_content_export_jobs(
    session: Session,
    content_id: str | None = None,
    request_id: str | None = None,
    status: str | None = None,
    from_dt=None,
    to_dt=None,
) -> list[ContentExportJob]:
    """Return content export jobs in stable creation order, optionally filtered.

    ``content_id``, ``request_id``, and ``status`` are exact, combinable
    string matches (case- and whitespace-sensitive); ``status`` is one of
    the four lifecycle literals when supplied. ``from_dt``/``to_dt`` are
    timezone-aware UTC instants applied as inclusive ``created_at`` bounds.
    Filter values are never resolved for existence, so an unknown content or
    request id is simply an empty match set rather than an error. Results
    follow the jobs' stable creation order (``created_at`` with the monotonic
    ``seq`` tiebreaker). The function is strictly read-only: it writes no
    job and no audit event.
    """
    stmt = select(ContentExportJob)
    if content_id is not None:
        stmt = stmt.where(ContentExportJob.content_id == content_id)
    if request_id is not None:
        stmt = stmt.where(ContentExportJob.request_id == request_id)
    if status is not None:
        stmt = stmt.where(ContentExportJob.status == status)
    if from_dt is not None:
        stmt = stmt.where(ContentExportJob.created_at >= from_dt)
    if to_dt is not None:
        stmt = stmt.where(ContentExportJob.created_at <= to_dt)
    stmt = stmt.order_by(*_CONTENT_EXPORT_JOB_ORDER)
    return list(session.execute(stmt).scalars().all())


def summarize_content_export_jobs(
    session: Session,
) -> tuple[dict[str, int], ContentExportJob | None]:
    """Return per-status counts over all jobs and the oldest pending job.

    The counts cover every existing job keyed by the four lifecycle states
    (``pending``/``running``/``succeeded``/``failed``), each defaulting to
    zero; settled jobs remain in their ``succeeded``/``failed`` counts. The
    second element is the single oldest ``pending`` job under the stable
    creation order (``created_at`` with the monotonic ``seq`` tiebreaker,
    which survives restarts), or ``None`` when no job is pending. Both are
    computed from the current persisted state at call time -- nothing is
    frozen, claimed, or settled. The function is strictly read-only: it
    writes no job and no audit event.
    """
    rows = session.execute(
        select(ContentExportJob.status, func.count()).group_by(
            ContentExportJob.status
        )
    ).all()
    counts = {state: 0 for state in CONTENT_EXPORT_JOB_STATES}
    for status_value, total in rows:
        counts[status_value] = total
    oldest_pending = session.execute(
        select(ContentExportJob)
        .where(ContentExportJob.status == CONTENT_EXPORT_JOB_PENDING)
        .order_by(*_CONTENT_EXPORT_JOB_ORDER)
        .limit(1)
    ).scalar_one_or_none()
    return counts, oldest_pending


def get_evidence_bundle_exchange(
    session: Session, evidence_bundle_id: str
) -> tuple[EvidenceBundle, Claim, Content, list[Attestation]]:
    """Return an evidence bundle's interoperability snapshot.

    The result is ``(bundle, claim, content, attestations)``: the bundle
    itself, the single claim the bundle is directly attached to, the single
    content that claim directly asserts, and the attestations whose target is
    exactly this evidence bundle in stable creation order. No content lineage
    is traversed and no other content, claim, bundle, or attestation is
    returned. Attestations targeting claims (including this bundle's claim)
    or other bundles are excluded; revoked attestations of this bundle are
    retained so the snapshot stays historically auditable. The function is
    strictly read-only: it writes no resource and no audit event.

    The bundle must exist; an unknown evidence bundle id is a missing
    resource, not an empty snapshot.
    """
    bundle = session.execute(
        select(EvidenceBundle).where(EvidenceBundle.id == evidence_bundle_id)
    ).scalar_one_or_none()
    if bundle is None:
        raise EvidenceBundleNotFoundError(evidence_bundle_id)

    claim = session.execute(
        select(Claim).where(Claim.id == bundle.claim_id)
    ).scalar_one_or_none()
    # A stored bundle always references a stored claim; the guard preserves
    # the 404 contract even against anomalous history.
    if claim is None:  # pragma: no cover - defensive
        raise EvidenceBundleNotFoundError(evidence_bundle_id)

    content = session.execute(
        select(Content).where(Content.id == claim.content_id)
    ).scalar_one_or_none()
    if content is None:  # pragma: no cover - defensive
        raise EvidenceBundleNotFoundError(evidence_bundle_id)

    attestations = list(
        session.execute(
            select(Attestation)
            .where(
                Attestation.target_type == signing.TARGET_EVIDENCE_BUNDLE,
                Attestation.target_id == evidence_bundle_id,
            )
            .order_by(*_ATTESTATION_ORDER)
        )
        .scalars()
        .all()
    )
    return bundle, claim, content, attestations


def _exchange_import_identity_select(
    manifest_version: str, evidence_bundle_id: str, manifest_digest_hex: str
):
    return select(ExchangeImportRecord).where(
        ExchangeImportRecord.manifest_version == manifest_version,
        ExchangeImportRecord.evidence_bundle_id == evidence_bundle_id,
        ExchangeImportRecord.manifest_digest_hex == manifest_digest_hex,
    )


def create_evidence_bundle_exchange_import(
    session: Session,
    payload: EvidenceBundleExchangeImportCreate,
) -> tuple[ExchangeImportRecord, bool]:
    """Register one offline-verified exchange package, returning ``(record, created)``.

    The caller has already verified the package: the request parses under
    the existing exchange manifest and snapshot structures, its internal
    associations are consistent, and the manifest digest matches the
    canonical SHA-256 of the snapshot exactly as received. This function
    performs no such verification and no local-resource lookup: the bundle
    id in the manifest is an opaque reference, so whether the referenced
    resources exist locally never changes the outcome.

    The receiving identity is ``(manifest_version, evidence_bundle_id,
    manifest_digest_hex)``; the snapshot itself is never stored. A first
    submission writes the receipt row and its
    ``evidence_bundle.exchange_imported`` audit event in a single
    transaction. A retried submission for the same identity returns the
    existing record with ``created=False`` and writes nothing -- no second
    row and no second audit event.
    """
    manifest = payload.manifest
    manifest_version = manifest.manifest_version
    evidence_bundle_id = manifest.evidence_bundle_id
    manifest_digest_hex = manifest.manifest_digest_hex

    existing = session.execute(
        _exchange_import_identity_select(
            manifest_version, evidence_bundle_id, manifest_digest_hex
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing, False

    while True:
        record = ExchangeImportRecord(
            id=ids.exchange_import_id(
                manifest_version, evidence_bundle_id, manifest_digest_hex
            ),
            manifest_version=manifest_version,
            evidence_bundle_id=evidence_bundle_id,
            manifest_digest_hex=manifest_digest_hex,
        )
        session.add(record)
        session.add(
            AuditEvent(
                event_type=EVENT_EVIDENCE_BUNDLE_EXCHANGE_IMPORTED,
                resource_id=record.id,
            )
        )
        try:
            session.commit()
        except IntegrityError:
            # A concurrent import registered this identity first: roll back
            # and return that record instead of duplicating it or writing a
            # second audit event.
            session.rollback()
            raced = session.execute(
                _exchange_import_identity_select(
                    manifest_version, evidence_bundle_id, manifest_digest_hex
                )
            ).scalar_one_or_none()
            if raced is None:  # pragma: no cover - defensive
                raise
            return raced, False
        session.refresh(record)
        return record, True


def get_evidence_bundle_exchange_import(
    session: Session, import_id: str
) -> ExchangeImportRecord:
    """Return an exchange-import receipt by id or raise the 404 domain error."""
    record = session.execute(
        select(ExchangeImportRecord).where(ExchangeImportRecord.id == import_id)
    ).scalar_one_or_none()
    if record is None:
        raise EvidenceBundleExchangeImportNotFoundError(import_id)
    return record


def _checkpoint_import_identity_select(
    checkpoint_version: str, event_count: int, events_digest_hex: str
):
    return select(CheckpointImportRecord).where(
        CheckpointImportRecord.checkpoint_version == checkpoint_version,
        CheckpointImportRecord.event_count == event_count,
        CheckpointImportRecord.events_digest_hex == events_digest_hex,
    )


def create_audit_checkpoint_import(
    session: Session,
    payload: AuditCheckpointImportCreate,
) -> tuple[CheckpointImportRecord, bool]:
    """Register one offline-verified audit checkpoint, returning ``(record, created)``.

    The caller has already verified the checkpoint: the request parses under
    the existing checkpoint structure, its event count matches the array,
    and the events digest matches the canonical SHA-256 of the received
    event array exactly as received. This function performs no such
    verification and no local-event lookup: the described events are never
    queried, created, or modified, so whether they exist locally never
    changes the outcome.

    The receiving identity is ``(checkpoint_version, event_count,
    events_digest_hex)``; the event array itself is never stored. A first
    submission writes the receipt row and its
    ``audit.checkpoint_imported`` audit event in a single transaction. A
    retried submission for the same identity returns the existing record
    with ``created=False`` and writes nothing -- no second row and no
    second audit event.
    """
    checkpoint = payload.checkpoint
    checkpoint_version = checkpoint.checkpoint_version
    event_count = checkpoint.event_count
    events_digest_hex = checkpoint.events_digest_hex

    existing = session.execute(
        _checkpoint_import_identity_select(
            checkpoint_version, event_count, events_digest_hex
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing, False

    while True:
        record = CheckpointImportRecord(
            id=ids.checkpoint_import_id(
                checkpoint_version, event_count, events_digest_hex
            ),
            checkpoint_version=checkpoint_version,
            event_count=event_count,
            events_digest_hex=events_digest_hex,
        )
        session.add(record)
        session.add(
            AuditEvent(
                event_type=EVENT_AUDIT_CHECKPOINT_IMPORTED,
                resource_id=record.id,
            )
        )
        try:
            session.commit()
        except IntegrityError:
            # A concurrent import registered this identity first: roll back
            # and return that record instead of duplicating it or writing a
            # second audit event.
            session.rollback()
            raced = session.execute(
                _checkpoint_import_identity_select(
                    checkpoint_version, event_count, events_digest_hex
                )
            ).scalar_one_or_none()
            if raced is None:  # pragma: no cover - defensive
                raise
            return raced, False
        session.refresh(record)
        return record, True


def get_audit_checkpoint_import(
    session: Session, import_id: str
) -> CheckpointImportRecord:
    """Return a checkpoint-import receipt by id or raise the 404 domain error."""
    record = session.execute(
        select(CheckpointImportRecord).where(CheckpointImportRecord.id == import_id)
    ).scalar_one_or_none()
    if record is None:
        raise AuditCheckpointImportNotFoundError(import_id)
    return record


# Checkpoint-import receipt search paging bounds.
DEFAULT_CHECKPOINT_IMPORTS_LIMIT = 50
MIN_CHECKPOINT_IMPORTS_LIMIT = 1
MAX_CHECKPOINT_IMPORTS_LIMIT = 100

_CHECKPOINT_IMPORT_ORDER = (
    CheckpointImportRecord.created_at.asc(),
    CheckpointImportRecord.seq.asc(),
)


def list_audit_checkpoint_imports(
    session: Session,
    checkpoint_version: str | None = None,
    events_digest_hex: str | None = None,
    event_count: int | None = None,
) -> list[CheckpointImportRecord]:
    """Return checkpoint-import receipts in stable creation order, optionally filtered.

    ``checkpoint_version`` and ``events_digest_hex`` are exact, case- and
    whitespace-sensitive string matches; ``event_count`` is an exact
    non-negative integer match. The three combine as logical AND; an absent
    filter imposes no restriction. Results follow the receipts' stable
    creation order (``created_at`` with the monotonic ``seq`` tiebreaker).
    The function is strictly read-only: it writes no resource and no audit
    event.
    """
    stmt = select(CheckpointImportRecord)
    if checkpoint_version is not None:
        stmt = stmt.where(
            CheckpointImportRecord.checkpoint_version == checkpoint_version
        )
    if events_digest_hex is not None:
        stmt = stmt.where(
            CheckpointImportRecord.events_digest_hex == events_digest_hex
        )
    if event_count is not None:
        stmt = stmt.where(CheckpointImportRecord.event_count == event_count)
    stmt = stmt.order_by(*_CHECKPOINT_IMPORT_ORDER)
    return list(session.execute(stmt).scalars().all())


# Checkpoint-import reconciliation listing paging bounds.
DEFAULT_CHECKPOINT_IMPORT_RECONCILIATIONS_LIMIT = 50
MIN_CHECKPOINT_IMPORT_RECONCILIATIONS_LIMIT = 1
MAX_CHECKPOINT_IMPORT_RECONCILIATIONS_LIMIT = 100


def list_audit_checkpoint_import_reconciliations(
    session: Session,
) -> list[CheckpointImportRecord]:
    """Return every checkpoint-import receipt in stable creation order.

    The reconciliation collection is unfiltered; each receipt's current
    local checkpoint reconciliation is computed by the caller against the
    complete local audit sequence. Results follow the receipts' stable
    creation order (``created_at`` with the monotonic ``seq`` tiebreaker).
    The function is strictly read-only: it writes no resource and no audit
    event.
    """
    stmt = select(CheckpointImportRecord).order_by(*_CHECKPOINT_IMPORT_ORDER)
    return list(session.execute(stmt).scalars().all())


# Exchange-import receipt search paging bounds.
DEFAULT_EXCHANGE_IMPORTS_LIMIT = 50
MIN_EXCHANGE_IMPORTS_LIMIT = 1
MAX_EXCHANGE_IMPORTS_LIMIT = 100

_EXCHANGE_IMPORT_ORDER = (
    ExchangeImportRecord.created_at.asc(),
    ExchangeImportRecord.seq.asc(),
)

# Exchange-import reconciliation listing paging bounds.
DEFAULT_EXCHANGE_IMPORT_RECONCILIATIONS_LIMIT = 50
MIN_EXCHANGE_IMPORT_RECONCILIATIONS_LIMIT = 1
MAX_EXCHANGE_IMPORT_RECONCILIATIONS_LIMIT = 100


def list_evidence_bundle_exchange_import_reconciliations(
    session: Session,
) -> list[ExchangeImportRecord]:
    """Return every exchange-import receipt in stable creation order.

    The reconciliation collection is unfiltered; each receipt's local
    reconciliation is computed by the caller from the receipt's own
    ``evidence_bundle_id``. Results follow the receipts' stable creation
    order (``created_at`` with the monotonic ``seq`` tiebreaker). The
    function is strictly read-only: it writes no resource and no audit
    event.
    """
    stmt = select(ExchangeImportRecord).order_by(*_EXCHANGE_IMPORT_ORDER)
    return list(session.execute(stmt).scalars().all())


def list_evidence_bundle_exchange_imports(
    session: Session,
    manifest_version: str | None = None,
    evidence_bundle_id: str | None = None,
    manifest_digest_hex: str | None = None,
) -> list[ExchangeImportRecord]:
    """Return exchange-import receipts in stable creation order, optionally filtered.

    The three receiving-identity fields are exact, case- and
    whitespace-sensitive string matches that combine as logical AND; an
    absent filter imposes no restriction. Results follow the receipts'
    stable creation order (``created_at`` with the monotonic ``seq``
    tiebreaker). The function is strictly read-only: it writes no resource
    and no audit event.
    """
    stmt = select(ExchangeImportRecord)
    if manifest_version is not None:
        stmt = stmt.where(
            ExchangeImportRecord.manifest_version == manifest_version
        )
    if evidence_bundle_id is not None:
        stmt = stmt.where(
            ExchangeImportRecord.evidence_bundle_id == evidence_bundle_id
        )
    if manifest_digest_hex is not None:
        stmt = stmt.where(
            ExchangeImportRecord.manifest_digest_hex == manifest_digest_hex
        )
    stmt = stmt.order_by(*_EXCHANGE_IMPORT_ORDER)
    return list(session.execute(stmt).scalars().all())


# Trust evaluation threshold bounds.
DEFAULT_TRUST_MIN_SIGNERS = 1
MIN_TRUST_MIN_SIGNERS = 1
MAX_TRUST_MIN_SIGNERS = 100

TRUST_DECISION_TRUSTED = "trusted"
TRUST_DECISION_UNTRUSTED = "untrusted"

# Authorization-decision reasons.
TRUST_REASON_THRESHOLD_MET = "threshold_met"
TRUST_REASON_BELOW_THRESHOLD = "below_threshold"
TRUST_REASON_POLICY_MISSING = "policy_missing"


def _require_trust_target(session: Session, target_type: str, target_id: str):
    """Return the existing claim or evidence bundle, or raise its 404.

    The declared ``target_type`` alone selects the lookup: a claim id is
    never matched against bundles nor a bundle id against claims.
    """
    if target_type == signing.TARGET_CLAIM:
        target = session.execute(
            select(Claim).where(Claim.id == target_id)
        ).scalar_one_or_none()
        if target is None:
            raise ClaimNotFoundError(target_id)
        return target
    target = session.execute(
        select(EvidenceBundle).where(EvidenceBundle.id == target_id)
    ).scalar_one_or_none()
    if target is None:
        raise EvidenceBundleNotFoundError(target_id)
    return target


def _qualified_signer_count(
    session: Session, target_type: str, target_id: str
) -> int:
    """Distinct actors with a verified, non-revoked attestation of the target.

    Only attestations of the exact target count, and only distinct signing
    actors count: multiple attestations by the same ``signer_actor_id`` --
    e.g. under different keys -- qualify once. An attestation carrying any
    recorded revocation does not count: its proof is retained but no longer
    relied upon. Every stored attestation is a verified one (unverifiable
    signatures are rejected before any row is written), so every non-revoked
    matching row qualifies.
    """
    revoked = exists().where(
        AttestationRevocation.attestation_id == Attestation.id
    )
    signer_ids = session.execute(
        select(Attestation.signer_actor_id)
        .where(
            Attestation.target_type == target_type,
            Attestation.target_id == target_id,
            ~revoked,
        )
        .distinct()
    ).scalars().all()
    return len(set(signer_ids))


def evaluate_trust(
    session: Session,
    target_type: str,
    target_id: str,
    min_signers: int = DEFAULT_TRUST_MIN_SIGNERS,
) -> dict:
    """Evaluate trust in one claim or evidence bundle from existing proofs.

    Only attestations of the exact target count, and only distinct signing
    actors count: multiple attestations by the same ``signer_actor_id`` --
    e.g. under different keys -- qualify once. An attestation with any
    recorded revocation does not count: its proof is retained but is no
    longer relied upon, even though the row is never removed. Every stored
    attestation is a verified attestation (unverifiable signatures are
    rejected before any row is written), so every non-revoked matching row
    qualifies. The result is computed live and the function is strictly
    read-only: it writes no resource and no audit event.

    The target must exist: a missing claim raises
    :class:`ClaimNotFoundError` and a missing evidence bundle raises
    :class:`EvidenceBundleNotFoundError`, matched by the declared
    ``target_type``. Returns the response-shaped evaluation dict; the
    decision is ``"trusted"`` when the qualified signer count reaches
    ``min_signers`` and ``"untrusted"`` otherwise (including zero).
    """
    _require_trust_target(session, target_type, target_id)
    qualified = _qualified_signer_count(session, target_type, target_id)
    return {
        "target_type": target_type,
        "target_id": target_id,
        "min_signers": min_signers,
        "qualified_signer_count": qualified,
        "decision": (
            TRUST_DECISION_TRUSTED
            if qualified >= min_signers
            else TRUST_DECISION_UNTRUSTED
        ),
    }


def create_actor_trust_policy(
    session: Session,
    payload: ActorTrustPolicyCreate,
    caller_actor_id: str,
) -> tuple[ActorTrustPolicy, bool]:
    """Register a subject's immutable trust policy, returning ``(policy, created)``.

    The authenticated caller is the subject: ``payload.actor_id`` must be
    the caller and the subject must already exist; an unknown subject or a
    caller/body mismatch is a ``422`` validation error and writes nothing.
    Each subject has at most one policy: a repeat registration of the same
    threshold returns the original record with ``created=False`` and writes
    no row, audit event, or new identifier, while a different threshold for
    the same subject is a ``409 actor_trust_policy_conflict`` and likewise
    writes nothing. Policies are immutable: there is no update, delete,
    deactivate, or replace path. On first creation the policy row and its
    ``actor_trust_policy.created`` audit event commit in a single
    transaction.
    """
    # Validate the subject before the identity lookup: an unknown subject
    # and a caller/body mismatch are 422s, not 404s.
    actor = get_actor_by_id(session, payload.actor_id)
    if actor is None:
        raise ProtectedAccessValidationError("unknown_actor")

    if payload.actor_id != caller_actor_id:
        raise ProtectedAccessValidationError("actor_mismatch")

    existing = session.execute(
        select(ActorTrustPolicy).where(
            ActorTrustPolicy.actor_id == payload.actor_id
        )
    ).scalar_one_or_none()
    if existing is not None:
        if existing.threshold == payload.threshold:
            # Idempotent retry: the original record, no new write or audit.
            return existing, False
        # A policy is never updated, replaced, or superseded.
        raise ActorTrustPolicyConflictError(payload.actor_id)

    policy = ActorTrustPolicy(
        id=ids.actor_trust_policy_id(payload.actor_id),
        actor_id=payload.actor_id,
        threshold=payload.threshold,
        enabled=True,
    )
    session.add(policy)
    session.add(
        AuditEvent(
            event_type=EVENT_ACTOR_TRUST_POLICY_CREATED,
            resource_id=policy.id,
        )
    )
    try:
        session.commit()
    except IntegrityError:
        # A concurrent registration for the same subject won the race.
        session.rollback()
        raced = session.execute(
            select(ActorTrustPolicy).where(
                ActorTrustPolicy.actor_id == payload.actor_id
            )
        ).scalar_one_or_none()
        if raced is None:  # pragma: no cover - defensive
            raise
        if raced.threshold == payload.threshold:
            return raced, False
        raise ActorTrustPolicyConflictError(payload.actor_id)
    session.refresh(policy)
    return policy, True


def decide_trust(
    session: Session,
    actor_id: str,
    target_type: str,
    target_id: str,
) -> dict:
    """Decide trust in one target under the calling subject's current policy.

    Only the caller's own stored policy participates. When the caller has no
    policy the target is never looked up -- its existence is neither checked
    nor revealed -- and the decision is ``untrusted``/``policy_missing``
    with a null policy id and threshold. With a policy, the target must
    exist (a missing claim raises :class:`ClaimNotFoundError`, a missing
    evidence bundle :class:`EvidenceBundleNotFoundError`, matched by the
    declared ``target_type`` with no cross-type matching) and the qualified
    signer count is computed exactly as in :func:`evaluate_trust`: verified,
    non-revoked attestations of the exact target, deduplicated by signing
    actor. The function is strictly read-only: it writes no resource and no
    audit event.
    """
    policy = session.execute(
        select(ActorTrustPolicy).where(ActorTrustPolicy.actor_id == actor_id)
    ).scalar_one_or_none()
    if policy is None:
        return {
            "target_type": target_type,
            "target_id": target_id,
            "policy_id": None,
            "threshold": None,
            "qualified_signer_count": 0,
            "decision": TRUST_DECISION_UNTRUSTED,
            "reason": TRUST_REASON_POLICY_MISSING,
        }

    _require_trust_target(session, target_type, target_id)
    qualified = _qualified_signer_count(session, target_type, target_id)
    met = qualified >= policy.threshold
    return {
        "target_type": target_type,
        "target_id": target_id,
        "policy_id": policy.id,
        "threshold": policy.threshold,
        "qualified_signer_count": qualified,
        "decision": (
            TRUST_DECISION_TRUSTED if met else TRUST_DECISION_UNTRUSTED
        ),
        "reason": (
            TRUST_REASON_THRESHOLD_MET if met else TRUST_REASON_BELOW_THRESHOLD
        ),
    }


# Trust-policy retrieval paging bounds.
DEFAULT_TRUST_POLICIES_LIMIT = 50
MIN_TRUST_POLICIES_LIMIT = 1
MAX_TRUST_POLICIES_LIMIT = 100


def list_actor_trust_policies(
    session: Session, actor_id: str | None = None
) -> list[ActorTrustPolicy]:
    """Return existing immutable trust policies in stable creation order.

    ``actor_id`` is an exact, case- and whitespace-sensitive match on the
    policy subject; ``None`` means unfiltered. The filter value is never
    resolved for existence, so an unknown subject is an empty result rather
    than a missing resource. Results follow the policies' stable creation
    order (``created_at`` with the monotonic ``seq`` tiebreaker). The search
    is strictly read-only: it writes no policy, resource, or audit event.
    """
    stmt = select(ActorTrustPolicy)
    if actor_id is not None:
        stmt = stmt.where(ActorTrustPolicy.actor_id == actor_id)
    stmt = stmt.order_by(*_ACTOR_TRUST_POLICY_ORDER)
    return list(session.execute(stmt).scalars().all())


# Audit-event search paging bounds.
DEFAULT_AUDIT_EVENTS_LIMIT = 50
MIN_AUDIT_EVENTS_LIMIT = 1
MAX_AUDIT_EVENTS_LIMIT = 100


def list_audit_events(
    session: Session,
    event_type: str | None = None,
    resource_id: str | None = None,
    from_dt=None,
    to_dt=None,
) -> list[AuditEvent]:
    """Return audit events in stable creation order, optionally filtered.

    ``event_type`` and ``resource_id`` are exact, combinable string matches.
    ``from_dt``/``to_dt`` are timezone-aware UTC instants applied as
    inclusive ``created_at`` bounds. Results follow the events' stable
    creation order (``created_at`` with the monotonic ``seq`` tiebreaker).
    The function is strictly read-only: it writes no resource and no audit
    event.
    """
    stmt = select(AuditEvent)
    if event_type is not None:
        stmt = stmt.where(AuditEvent.event_type == event_type)
    if resource_id is not None:
        stmt = stmt.where(AuditEvent.resource_id == resource_id)
    if from_dt is not None:
        stmt = stmt.where(AuditEvent.created_at >= from_dt)
    if to_dt is not None:
        stmt = stmt.where(AuditEvent.created_at <= to_dt)
    stmt = stmt.order_by(*_AUDIT_EVENT_ORDER)
    return list(session.execute(stmt).scalars().all())


def _require_content(session: Session, content_id: str) -> Content:
    """Return content by id or raise :class:`ContentNotFoundError`."""
    content = session.execute(
        select(Content).where(Content.id == content_id)
    ).scalar_one_or_none()
    if content is None:
        raise ContentNotFoundError(content_id)
    return content


def _content_relation_identity_select(payload: ContentRelationCreate):
    return select(ContentRelation).where(
        ContentRelation.content_id == payload.content_id,
        ContentRelation.parent_content_id == payload.parent_content_id,
        ContentRelation.relation_type == payload.relation_type,
    )


def _would_create_cycle(
    session: Session, content_id: str, parent_content_id: str
) -> bool:
    """True if an edge ``content_id -> parent_content_id`` would close a cycle.

    Edges point from a content to its direct source, so a cycle forms exactly
    when ``content_id`` is already reachable from ``parent_content_id`` by
    following existing edges.
    """
    visited: set[str] = set()
    frontier = [parent_content_id]
    while frontier:
        current = frontier.pop()
        if current == content_id:
            return True
        if current in visited:
            continue
        visited.add(current)
        parents = session.execute(
            select(ContentRelation.parent_content_id).where(
                ContentRelation.content_id == current
            )
        ).scalars().all()
        frontier.extend(parents)
    return False


def create_content_relation(
    session: Session, payload: ContentRelationCreate
) -> tuple[ContentRelation, bool]:
    """Create an immutable lineage edge, returning ``(relation, created)``.

    Both endpoints must be existing contents. A self-loop or an edge that
    would close a cycle is a validation error and writes nothing. A repeat
    submission of the same two endpoints and relation type returns the
    existing relation with ``created=False`` and writes no row or audit
    event. On first creation, the relation row and its
    ``content_relation.created`` audit event commit in a single transaction.
    """
    # A self-loop is invalid input, checked before any existence lookup.
    if payload.content_id == payload.parent_content_id:
        raise ContentRelationValidationError("self_relation")

    _require_content(session, payload.content_id)
    _require_content(session, payload.parent_content_id)

    # An existing edge with this exact identity is returned unchanged; a
    # repeat submission is idempotent and needs no row or audit event.
    existing = session.execute(
        _content_relation_identity_select(payload)
    ).scalar_one_or_none()
    if existing is not None:
        return existing, False

    if _would_create_cycle(session, payload.content_id, payload.parent_content_id):
        raise ContentRelationValidationError("relation_cycle")

    relation = ContentRelation(
        id=ids.content_relation_id(
            payload.content_id, payload.parent_content_id, payload.relation_type
        ),
        content_id=payload.content_id,
        parent_content_id=payload.parent_content_id,
        relation_type=payload.relation_type,
    )
    session.add(relation)
    session.add(
        AuditEvent(
            event_type=EVENT_CONTENT_RELATION_CREATED, resource_id=relation.id
        )
    )
    try:
        session.commit()
    except IntegrityError:
        # Concurrent identical relation won the race: return its resource.
        session.rollback()
        raced = session.execute(
            _content_relation_identity_select(payload)
        ).scalar_one_or_none()
        if raced is None:  # pragma: no cover - defensive
            raise
        return raced, False
    session.refresh(relation)
    return relation, True


def get_content_relation(session: Session, relation_id: str) -> ContentRelation:
    """Return a relation by id or raise :class:`ContentRelationNotFoundError`."""
    relation = session.execute(
        select(ContentRelation).where(ContentRelation.id == relation_id)
    ).scalar_one_or_none()
    if relation is None:
        raise ContentRelationNotFoundError(relation_id)
    return relation


def list_content_relations(
    session: Session, content_id: str
) -> list[ContentRelation]:
    """Return the in- and out-edges of one content in stable creation order.

    The content must exist; an unknown content id is a missing resource, not
    an empty collection.
    """
    _require_content(session, content_id)
    stmt = (
        select(ContentRelation)
        .where(
            or_(
                ContentRelation.content_id == content_id,
                ContentRelation.parent_content_id == content_id,
            )
        )
        .order_by(*_CONTENT_RELATION_ORDER)
    )
    return list(session.execute(stmt).scalars().all())


# Lineage traversal directions.
LINEAGE_ANCESTORS = "ancestors"
LINEAGE_DESCENDANTS = "descendants"
LINEAGE_DIRECTIONS = frozenset({LINEAGE_ANCESTORS, LINEAGE_DESCENDANTS})

DEFAULT_LINEAGE_MAX_DEPTH = 8
MIN_LINEAGE_MAX_DEPTH = 1
MAX_LINEAGE_MAX_DEPTH = 32

DEFAULT_LINEAGE_MIN_DEPTH = 1
MIN_LINEAGE_MIN_DEPTH = 1
MAX_LINEAGE_MIN_DEPTH = 32

DEFAULT_LINEAGE_LIMIT = 50
MIN_LINEAGE_LIMIT = 1
MAX_LINEAGE_LIMIT = 100


def get_content_lineage(
    session: Session,
    content_id: str,
    direction: str,
    max_depth: int = DEFAULT_LINEAGE_MAX_DEPTH,
) -> list[tuple[Content, int, str]]:
    """Return reachable contents as ``(content, depth, relation_type)`` triples.

    Ancestor traversal follows edges from a content to its direct source
    (``content_id -> parent_content_id``); descendant traversal follows them
    in reverse. The origin is never included. Only contents reachable within
    ``max_depth`` edges are returned; each content appears once at its
    shortest depth. Triples are ordered by depth ascending; within one depth,
    contents are ordered by the stable creation order of the edge through
    which they were first reached. The third element is the
    ``relation_type`` of that first-discovery edge; it is recorded for
    filtering but never changes the visited set, reachability, shortest-depth
    computation, or ordering.

    The traversal is read-only and tracks a visited set, so even anomalous
    history containing a cycle terminates and the walk is bounded by
    ``max_depth``. The origin must exist or :class:`ContentNotFoundError` is
    raised.
    """
    # A missing origin is a missing resource, checked before any traversal.
    _require_content(session, content_id)

    if direction == LINEAGE_ANCESTORS:
        source_col = ContentRelation.content_id
        neighbor_col = ContentRelation.parent_content_id
    else:
        source_col = ContentRelation.parent_content_id
        neighbor_col = ContentRelation.content_id

    visited: set[str] = {content_id}
    frontier: list[str] = [content_id]
    # id -> (shortest depth, first-discovery edge type); discovery order is
    # the dictionary insertion order itself.
    reached: dict[str, tuple[int, str]] = {}
    for depth in range(1, max_depth + 1):
        if not frontier:
            break
        # Ordering every edge leaving the current frontier by its stable
        # creation order fixes each level's first-discovery order, including
        # converging paths whose discovering edges differ.
        rows = session.execute(
            select(neighbor_col, ContentRelation.relation_type)
            .where(source_col.in_(frontier))
            .order_by(*_CONTENT_RELATION_ORDER)
        ).all()
        next_frontier: list[str] = []
        for neighbor_id, relation_type in rows:
            if neighbor_id in visited:
                # Already reached at an equal or shorter depth; also what
                # makes an anomalous cycle terminate.
                continue
            visited.add(neighbor_id)
            next_frontier.append(neighbor_id)
            reached[neighbor_id] = (depth, relation_type)
        frontier = next_frontier

    if not reached:
        return []

    contents = (
        session.execute(select(Content).where(Content.id.in_(list(reached))))
        .scalars()
        .all()
    )
    by_id = {content.id: content for content in contents}
    return [
        (by_id[reached_id], reached_depth, relation_type)
        for reached_id, (reached_depth, relation_type) in reached.items()
    ]
