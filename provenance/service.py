"""Domain services: transactional creation, idempotent dedup, and lookups.

Every successful creation writes the resource row and its audit event in a
single transaction/COMMIT. A repeat content submission performs no writes and
adds no audit event.
"""

from __future__ import annotations

import hashlib

from sqlalchemy import exists, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from provenance import canonical, ed25519, ids, signing
from provenance.errors import (
    ActorAlreadyExistsError,
    AttestationNotFoundError,
    AttestationRevocationNotFoundError,
    AttestationVerificationError,
    ClaimNotFoundError,
    ContentNotFoundError,
    ContentRelationNotFoundError,
    ContentRelationValidationError,
    EvidenceBundleNotFoundError,
    ProtectedAccessValidationError,
    UnknownActorError,
)
from provenance.models import (
    EVENT_ACTOR_CREATED,
    EVENT_ATTESTATION_ACCESS_GRANTED,
    EVENT_ATTESTATION_CREATED,
    EVENT_ATTESTATION_REVOKED,
    EVENT_AUTHENTICATION_KEY_RETIRED,
    EVENT_AUTHENTICATION_KEY_ROTATED,
    EVENT_CLAIM_CREATED,
    EVENT_CONTENT_CREATED,
    EVENT_CONTENT_RELATION_CREATED,
    EVENT_EVIDENCE_BUNDLE_CREATED,
    Actor,
    Attestation,
    AttestationAccessGrant,
    AttestationRevocation,
    AuditEvent,
    AuthenticationKeyRotation,
    Claim,
    Content,
    ContentRelation,
    EvidenceBundle,
)
from provenance.schemas import (
    ActorCreate,
    AttestationAccessGrantCreate,
    AttestationCreate,
    AttestationRevocationCreate,
    AuthenticationKeyRotationCreate,
    ClaimCreate,
    ContentCreate,
    ContentRelationCreate,
    EvidenceBundleCreate,
)
from provenance.time_utils import utc_now

# Stable creation order: timestamp first, with the monotonic sequence as a
# deterministic tiebreaker.
_CONTENT_ORDER = (Content.created_at.asc(), Content.seq.asc())
_CLAIM_ORDER = (Claim.created_at.asc(), Claim.seq.asc())
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
_CONTENT_RELATION_ORDER = (
    ContentRelation.created_at.asc(),
    ContentRelation.seq.asc(),
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

    actor = session.get(Actor, payload.signer_actor_id)
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

    actor = session.get(Actor, payload.revoker_actor_id)
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

    grantee = session.get(Actor, payload.grantee_actor_id)
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


def get_accessible_attestation(
    session: Session, attestation_id: str, actor_id: str
) -> Attestation | None:
    """Return the attestation iff ``actor_id`` may read it, else ``None``.

    The attestation's signer and any grantee holding an access grant for
    this exact attestation may read it. The function is strictly read-only:
    it performs no resource or audit writes.
    """
    attestation = session.execute(
        select(Attestation).where(Attestation.id == attestation_id)
    ).scalar_one_or_none()
    if attestation is None:
        return None
    if attestation.signer_actor_id == actor_id:
        return attestation
    grant_exists = session.execute(
        select(AttestationAccessGrant.id).where(
            AttestationAccessGrant.attestation_id == attestation_id,
            AttestationAccessGrant.grantee_actor_id == actor_id,
        )
    ).first()
    return attestation if grant_exists is not None else None


def _authentication_key_rotation_identity_select(
    actor_id: str, public_key: bytes
):
    return select(AuthenticationKeyRotation).where(
        AuthenticationKeyRotation.actor_id == actor_id,
        AuthenticationKeyRotation.public_key == public_key,
    )


def create_authentication_key_rotation(
    session: Session,
    payload: AuthenticationKeyRotationCreate,
    caller_actor_id: str,
) -> tuple[AuthenticationKeyRotation, bool]:
    """Create an authentication key rotation, returning ``(record, created)``.

    The authenticated caller must be the subject actor: a rotation is only
    ever created for oneself. Only the 32-byte public key is stored -- never
    a private key or any signature. A repeat submission of the same actor
    and public key returns the original record with ``created=False`` and
    writes no row or audit event (even if that record has since been
    retired); any different key forms an independent record. On first
    creation the rotation row and its ``authentication_key.rotated`` audit
    event commit in a single transaction, and the key immediately joins the
    actor's authentication set for the protected routes.
    """
    # The caller must be the subject of the rotation, checked before any
    # existence lookup so no information about other actors leaks.
    if payload.actor_id != caller_actor_id:
        raise ProtectedAccessValidationError("caller_not_subject")
    actor = session.get(Actor, payload.actor_id)
    if actor is None:
        raise ProtectedAccessValidationError("unknown_actor")

    existing = session.execute(
        _authentication_key_rotation_identity_select(
            payload.actor_id, payload.new_public_key
        )
    ).scalar_one_or_none()
    if existing is not None:
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
            _authentication_key_rotation_identity_select(
                payload.actor_id, payload.new_public_key
            )
        ).scalar_one_or_none()
        if raced is None:  # pragma: no cover - defensive
            raise
        return raced, False
    session.refresh(rotation)
    return rotation, True


def retire_authentication_key_rotation(
    session: Session, rotation_id: str, caller_actor_id: str
) -> AuthenticationKeyRotation:
    """Retire an active rotation record, deactivating its key immediately.

    Only the subject actor may retire its own record, and only while it is
    active. An unknown record, a caller who is not the subject, and a
    repeated retirement are all ``422`` validation errors that write
    nothing. On success the record's ``active``/``retired_at`` update and
    its ``authentication_key.retired`` audit event commit in a single
    transaction; the record itself is preserved, never deleted.
    """
    rotation = session.execute(
        select(AuthenticationKeyRotation).where(
            AuthenticationKeyRotation.id == rotation_id
        )
    ).scalar_one_or_none()
    if rotation is None:
        raise ProtectedAccessValidationError(
            "authentication_key_rotation_not_found"
        )
    if rotation.actor_id != caller_actor_id:
        raise ProtectedAccessValidationError("caller_not_subject")
    if not rotation.active:
        raise ProtectedAccessValidationError("rotation_already_retired")

    rotation.active = False
    rotation.retired_at = utc_now()
    session.add(
        AuditEvent(
            event_type=EVENT_AUTHENTICATION_KEY_RETIRED,
            resource_id=rotation.id,
        )
    )
    session.commit()
    session.refresh(rotation)
    return rotation


# Trust evaluation threshold bounds.
DEFAULT_TRUST_MIN_SIGNERS = 1
MIN_TRUST_MIN_SIGNERS = 1
MAX_TRUST_MIN_SIGNERS = 100

TRUST_DECISION_TRUSTED = "trusted"
TRUST_DECISION_UNTRUSTED = "untrusted"


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
    if target_type == signing.TARGET_CLAIM:
        target = session.execute(
            select(Claim).where(Claim.id == target_id)
        ).scalar_one_or_none()
        if target is None:
            raise ClaimNotFoundError(target_id)
    else:
        target = session.execute(
            select(EvidenceBundle).where(EvidenceBundle.id == target_id)
        ).scalar_one_or_none()
        if target is None:
            raise EvidenceBundleNotFoundError(target_id)

    # Exclude any attestation carrying at least one revocation record: its
    # proof is preserved but no longer qualifies the target for trust.
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
    qualified = len(set(signer_ids))
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
