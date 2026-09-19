"""Versioned HTTP routes (``/v1``)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status
from sqlalchemy.orm import Session

from provenance import service
from provenance.schemas import (
    ActorCreate,
    ActorResponse,
    ClaimCreate,
    ClaimListResponse,
    ClaimResponse,
    ContentCreate,
    ContentListResponse,
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


@router.post("/evidence-bundles", response_model=EvidenceBundleResponse)
def create_evidence_bundle(
    payload: EvidenceBundleCreate, session: DbSession, response: Response
) -> EvidenceBundleResponse:
    bundle, created = service.create_evidence_bundle(session, payload)
    # First creation -> 201; an idempotent repeat submission -> 200, and the
    # existing bundle (with its first metadata) is returned unchanged.
    response.status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return EvidenceBundleResponse.model_validate(bundle)


@router.get(
    "/evidence-bundles/{evidence_bundle_id}", response_model=EvidenceBundleResponse
)
def get_evidence_bundle(
    evidence_bundle_id: str, session: DbSession
) -> EvidenceBundleResponse:
    bundle = service.get_evidence_bundle(session, evidence_bundle_id)
    return EvidenceBundleResponse.model_validate(bundle)


@router.get(
    "/claims/{claim_id}/evidence-bundles",
    response_model=EvidenceBundleListResponse,
)
def list_claim_evidence_bundles(
    claim_id: str, session: DbSession
) -> EvidenceBundleListResponse:
    items = service.list_evidence_bundles_for_claim(session, claim_id)
    return EvidenceBundleListResponse(
        items=[EvidenceBundleResponse.model_validate(item) for item in items],
        count=len(items),
    )
