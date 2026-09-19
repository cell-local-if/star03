"""Versioned HTTP routes (``/v1``)."""

from __future__ import annotations

import base64
import re
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request, Response, status
from sqlalchemy.orm import Session

from provenance import pagination, service
from provenance.errors import LineageValidationError
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
    ContentLineageItem,
    ContentLineageResponse,
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


@router.post("/content-relations", response_model=ContentRelationResponse)
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
    items = service.list_content_relations(session, content_id)
    return ContentRelationListResponse(
        items=[ContentRelationResponse.model_validate(item) for item in items],
        count=len(items),
    )


def _lineage_item(content, depth: int):
    # The full public content view, with the traversal depth added.
    fields = ContentResponse.model_validate(content).model_dump()
    return ContentLineageItem(**fields, depth=depth)


_INTEGER_RE = re.compile(r"-?[0-9]+")

#: Every query parameter the lineage endpoint understands, in validation
#: order. Anything else is an ordinary FastAPI unknown-field rejection.
_LINEAGE_FIELDS = (
    "direction",
    "max_depth",
    "min_depth",
    "relation_type",
    "limit",
    "cursor",
)


def _lineage_query_error(field: str, msg: str, error_type: str):
    return LineageValidationError(
        [{"loc": ["query", field], "msg": msg, "type": error_type}]
    )


def _parse_lineage_int(
    raw: str | None,
    field: str,
    *,
    default: int,
    lower: int,
    upper: int,
) -> int:
    """Parse a strictly-integer bounded query parameter, mirroring max_depth.

    A missing value yields ``default``; blank/whitespace, non-integer text
    (including "8.0"), and out-of-range values are 422 validation errors.
    """
    if raw is None:
        return default
    if not _INTEGER_RE.fullmatch(raw):
        raise _lineage_query_error(
            field,
            f"{field} must be an integer",
            "value_error.integer",
        )
    value = int(raw)
    if not lower <= value <= upper:
        raise _lineage_query_error(
            field,
            f"{field} must be between {lower} and {upper}",
            "value_error.range",
        )
    return value


@router.get(
    "/contents/{content_id}/lineage",
    response_model=ContentLineageResponse,
)
def get_content_lineage(
    content_id: str,
    request: Request,
    session: DbSession,
    direction: str | None = Query(default=None),
    max_depth: str | None = Query(default=None),
    min_depth: str | None = Query(default=None),
    relation_type: str | None = Query(default=None),
    limit: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
) -> ContentLineageResponse:
    # Raw multi-values are inspected deliberately: FastAPI otherwise keeps
    # only the last value of a repeated scalar parameter and Pydantic coerces
    # "8.0" to 8, both of which must be rejected rather than silently
    # defaulted or normalized.
    raw = request.query_params
    for field in _LINEAGE_FIELDS:
        if len(raw.getlist(field)) > 1:
            raise _lineage_query_error(
                field,
                f"query parameter {field} must be provided exactly once",
                "value_error.repeated",
            )

    if direction is None:
        raise _lineage_query_error(
            "direction", "Field required", "value_error.missing"
        )
    if direction not in service.LINEAGE_DIRECTIONS:
        # Covers missing values, whitespace/blank strings, and any value
        # other than the two literal traversal directions.
        raise _lineage_query_error(
            "direction",
            "direction must be 'ancestors' or 'descendants'",
            "value_error",
        )

    max_depth_value = _parse_lineage_int(
        max_depth,
        "max_depth",
        default=service.DEFAULT_LINEAGE_MAX_DEPTH,
        lower=service.MIN_LINEAGE_MAX_DEPTH,
        upper=service.MAX_LINEAGE_MAX_DEPTH,
    )
    min_depth_value = _parse_lineage_int(
        min_depth,
        "min_depth",
        default=service.DEFAULT_LINEAGE_MIN_DEPTH,
        lower=service.MIN_LINEAGE_MIN_DEPTH,
        upper=service.MAX_LINEAGE_MIN_DEPTH,
    )
    if min_depth_value > max_depth_value:
        raise _lineage_query_error(
            "min_depth",
            "min_depth must not be greater than max_depth",
            "value_error.range",
        )

    if relation_type is not None and (
        not relation_type
        or relation_type not in service.LINEAGE_RELATION_TYPES
    ):
        # A blank value is an explicit, invalid filter rather than "no
        # filter"; only omission disables relation-type filtering.
        raise _lineage_query_error(
            "relation_type",
            "relation_type must be 'version_of' or 'derived_from'",
            "value_error",
        )

    limit_value = _parse_lineage_int(
        limit,
        "limit",
        default=service.DEFAULT_LINEAGE_LIMIT,
        lower=service.MIN_LINEAGE_LIMIT,
        upper=service.MAX_LINEAGE_LIMIT,
    )

    offset = 0
    if cursor is not None:
        # An empty/whitespace cursor is malformed input, not "first page".
        if not cursor.strip():
            raise _lineage_query_error(
                "cursor",
                "cursor is malformed or invalid",
                "value_error.cursor",
            )
        secret = request.app.state.settings.lineage_cursor_secret
        payload = pagination.decode_lineage_cursor(secret, cursor)
        if payload is None:
            raise _lineage_query_error(
                "cursor",
                "cursor is malformed or invalid",
                "value_error.cursor",
            )
        # The cursor only resumes the exact query that minted it. A changed
        # filter or pagination parameter is a client error, not a silently
        # re-anchored walk.
        expected = {
            "o": content_id,
            "d": direction,
            "x": max_depth_value,
            "m": min_depth_value,
            "r": relation_type or "",
            "l": limit_value,
        }
        if any(payload[key] != value for key, value in expected.items()):
            raise _lineage_query_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = payload["f"]

    # Traversal reachability and shortest-depth computation ignore the
    # filters; filtering and pagination only shape the returned items.
    items = service.get_content_lineage(
        session,
        content_id,
        direction,
        max_depth_value,
        relation_type=relation_type,
        min_depth=min_depth_value,
    )
    total = len(items)
    page = items[offset : offset + limit_value] if offset <= total else []

    next_cursor = None
    next_offset = offset + len(page)
    if next_offset < total:
        next_cursor = pagination.encode_lineage_cursor(
            request.app.state.settings.lineage_cursor_secret,
            origin_id=content_id,
            direction=direction,
            max_depth=max_depth_value,
            min_depth=min_depth_value,
            relation_type=relation_type,
            limit=limit_value,
            offset=next_offset,
        )

    return ContentLineageResponse(
        items=[_lineage_item(content, d) for content, d in page],
        count=total,
        next_cursor=next_cursor,
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
