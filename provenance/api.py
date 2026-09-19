"""Versioned HTTP routes (``/v1``)."""

from __future__ import annotations

import base64
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request, Response, status
from sqlalchemy.orm import Session

from provenance import service
from provenance.schemas import (
    ActorCreate,
    ActorResponse,
    AttestationCreate,
    AttestationListResponse,
    AttestationResponse,
    ClaimCreate,
    ClaimListResponse,
    ClaimResponse,
    ContentCreate,
    ContentListResponse,
    ContentRelationCreate,
    ContentRelationListResponse,
    ContentRelationResponse,
    ContentResponse,
    EvidenceBundleCreate,
    EvidenceBundleListResponse,
    EvidenceBundleResponse,
)

router = APIRouter(prefix="/v1")


def get_db(request: Request) -> Session:
    """Yield a session from the factory bound to the application."""
    factory = request.app.state.session_factory
    session = factory()
    try:
        yield session
    finally:
        session.close()


DbSession = Annotated[Session, Depends(get_db)]


@router.post("/actors", response_model=ActorResponse, status_code=status.HTTP_201_CREATED)
def create_actor(payload: ActorCreate, session: DbSession) -> ActorResponse:
    actor = service.create_actor(session, payload)
    return ActorResponse.model_validate(actor)


@router.post("/contents", response_model=ContentResponse)
def create_content(
    payload: ContentCreate, session: DbSession, response: Response
) -> ContentResponse:
    content, created = service.create_content(session, payload)
    # First registration -> 201; an idempotent repeat submission -> 200, and
    # the existing resource is returned unchanged.
    response.status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return ContentResponse.model_validate(content)


@router.get("/contents/{content_id}", response_model=ContentResponse)
def get_content(content_id: str, session: DbSession) -> ContentResponse:
    content = service.get_content(session, content_id)
    return ContentResponse.model_validate(content)


@router.get("/contents", response_model=ContentListResponse)
def list_contents(
    session: DbSession, actor_id: str | None = None
) -> ContentListResponse:
    items = service.list_contents(session, actor_id)
    return ContentListResponse(
        items=[ContentResponse.model_validate(item) for item in items],
        count=len(items),
    )


@router.post(
    "/content-relations",
    response_model=ContentRelationResponse,
)
def create_content_relation(
    payload: ContentRelationCreate, session: DbSession, response: Response
) -> ContentRelationResponse:
    relation, created = service.create_content_relation(session, payload)
    # First creation -> 201; an idempotent repeat submission -> 200, and the
    # existing immutable relation is returned unchanged.
    response.status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return ContentRelationResponse.model_validate(relation)


@router.get(
    "/content-relations/{relation_id}",
    response_model=ContentRelationResponse,
)
def get_content_relation(
    relation_id: str, session: DbSession
) -> ContentRelationResponse:
    relation = service.get_content_relation(session, relation_id)
    return ContentRelationResponse.model_validate(relation)


@router.get(
    "/contents/{content_id}/relations",
    response_model=ContentRelationListResponse,
)
def list_content_relations(
    content_id: str, session: DbSession
) -> ContentRelationListResponse:
    items = service.list_relations_for_content(session, content_id)
    return ContentRelationListResponse(
        items=[ContentRelationResponse.model_validate(item) for item in items],
        count=len(items),
    )


@router.post("/claims", response_model=ClaimResponse)
def create_claim(
    payload: ClaimCreate, session: DbSession, response: Response
) -> ClaimResponse:
    claim, created = service.create_claim(session, payload)
    # First creation -> 201; an idempotent repeat submission -> 200, and the
    # existing immutable claim is returned unchanged.
    response.status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return ClaimResponse.model_validate(claim)


@router.get("/claims/{claim_id}", response_model=ClaimResponse)
def get_claim(claim_id: str, session: DbSession) -> ClaimResponse:
    claim = service.get_claim(session, claim_id)
    return ClaimResponse.model_validate(claim)


@router.get("/contents/{content_id}/claims", response_model=ClaimListResponse)
def list_content_claims(content_id: str, session: DbSession) -> ClaimListResponse:
    items = service.list_claims_for_content(session, content_id)
    return ClaimListResponse(
        items=[ClaimResponse.model_validate(item) for item in items],
        count=len(items),
    )


def _bundle_response(bundle) -> EvidenceBundleResponse:
    # Map the ORM's reserved-name-safe ``metadata_`` attribute to the
    # wire-level ``metadata`` object explicitly.
    return EvidenceBundleResponse(
        id=bundle.id,
        claim_id=bundle.claim_id,
        evidence_type=bundle.evidence_type,
        digest_algorithm=bundle.digest_algorithm,
        digest_hex=bundle.digest_hex,
        media_type=bundle.media_type,
        metadata=bundle.metadata_,
        created_at=bundle.created_at,
    )


@router.post("/evidence-bundles", response_model=EvidenceBundleResponse)
def create_evidence_bundle(
    payload: EvidenceBundleCreate, session: DbSession, response: Response
) -> EvidenceBundleResponse:
    bundle, created = service.create_evidence_bundle(session, payload)
    # First creation -> 201; an idempotent repeat submission -> 200, and the
    # existing bundle (with its original metadata) is returned unchanged.
    response.status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return _bundle_response(bundle)


@router.get(
    "/evidence-bundles/{evidence_bundle_id}",
    response_model=EvidenceBundleResponse,
)
def get_evidence_bundle(
    evidence_bundle_id: str, session: DbSession
) -> EvidenceBundleResponse:
    bundle = service.get_evidence_bundle(session, evidence_bundle_id)
    return _bundle_response(bundle)


@router.get(
    "/claims/{claim_id}/evidence-bundles",
    response_model=EvidenceBundleListResponse,
)
def list_claim_evidence_bundles(
    claim_id: str, session: DbSession
) -> EvidenceBundleListResponse:
    items = service.list_evidence_bundles_for_claim(session, claim_id)
    return EvidenceBundleListResponse(
        items=[_bundle_response(item) for item in items],
        count=len(items),
    )


def _attestation_response(attestation) -> AttestationResponse:
    # The raw signature is never available here: only the stored public key
    # bytes and signature digest leave the service.
    return AttestationResponse(
        id=attestation.id,
        target_type=attestation.target_type,
        target_id=attestation.target_id,
        signer_actor_id=attestation.signer_actor_id,
        public_key=base64.b64encode(attestation.public_key).decode("ascii"),
        signature_digest_algorithm=attestation.signature_digest_algorithm,
        signature_digest_hex=attestation.signature_digest_hex,
        verified=True,
        created_at=attestation.created_at,
    )


@router.post("/attestations", response_model=AttestationResponse)
def create_attestation(
    payload: AttestationCreate, session: DbSession, response: Response
) -> AttestationResponse:
    attestation, created = service.create_attestation(session, payload)
    # First creation -> 201; an idempotent repeat submission -> 200, and the
    # existing attestation is returned unchanged.
    response.status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return _attestation_response(attestation)


@router.get(
    "/attestations/{attestation_id}",
    response_model=AttestationResponse,
)
def get_attestation(
    attestation_id: str, session: DbSession
) -> AttestationResponse:
    attestation = service.get_attestation(session, attestation_id)
    return _attestation_response(attestation)


@router.get("/attestations", response_model=AttestationListResponse)
def list_attestations(
    session: DbSession,
    target_type: Literal["claim", "evidence_bundle"] | None = None,
    target_id: str | None = None,
) -> AttestationListResponse:
    items = service.list_attestations(session, target_type, target_id)
    return AttestationListResponse(
        items=[_attestation_response(item) for item in items],
        count=len(items),
    )
