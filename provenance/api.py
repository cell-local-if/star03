"""Versioned HTTP routes (``/v1``)."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request, Response, status
from sqlalchemy.orm import Session

from provenance import (
    access_signing,
    canonical,
    ed25519,
    pagination,
    service,
    signing,
)
from provenance.access_signing import AccessAuthError
from provenance.errors import (
    AuditCheckpointImportValidationError,
    AuditExchangeImportValidationError,
    AuditSignatureVerificationError,
    EvidenceBundleExchangeImportValidationError,
    ImpactImportValidationError,
    ImpactReconExchangeImportValidationError,
    ImpactReconSignatureVerificationError,
    LineageValidationError,
    ProtectedAccessValidationError,
    ProtectedResourceNotFoundError,
)
from provenance.models import (
    CONTENT_EXPORT_JOB_FAILED,
    CONTENT_EXPORT_JOB_PENDING,
    CONTENT_EXPORT_JOB_RUNNING,
    CONTENT_EXPORT_JOB_STATES,
    CONTENT_EXPORT_JOB_SUCCEEDED,
    RELATION_DERIVED_FROM,
    RELATION_VERSION_OF,
)
from provenance.pagination import InvalidCursorError
from provenance.signing import ATTESTATION_TARGET_TYPES
from provenance.time_utils import parse_rfc3339_utc, utc_now
from provenance.schemas import (
    AUDIT_CHECKPOINT_DIGEST_ALGORITHM,
    AUDIT_CHECKPOINT_VERSION,
    CSP_CHECKPOINT_VERSION,
    CSP_DIGEST_ALGORITHM,
    EXCHANGE_MANIFEST_DIGEST_ALGORITHM,
    EXCHANGE_MANIFEST_VERSION,
    IMPACT_RECON_AUDIT_CHECKPOINT_VERSION,
    IMPACT_RECON_AUDIT_DIGEST_ALGORITHM,
    IMPACT_RECON_CHECKPOINT_VERSION,
    IMPACT_RECON_DIGEST_ALGORITHM,
    REVOCATION_IMPACT_CHECKPOINT_VERSION,
    REVOCATION_IMPACT_DIGEST_ALGORITHM,
    ActorCreate,
    ActorPageResponse,
    ActorResponse,
    ActorTrustPolicyCreate,
    ActorTrustPolicyPageResponse,
    ActorTrustPolicyResponse,
    AttestationAccessGrantCreate,
    AttestationAccessGrantPageResponse,
    AttestationAccessGrantResponse,
    AttestationAccessGrantRevocationCreate,
    AttestationAccessGrantRevocationListResponse,
    AttestationAccessGrantRevocationPageResponse,
    AttestationAccessGrantRevocationResponse,
    AttestationCreate,
    AttestationListResponse,
    AttestationResponse,
    AttestationRevocationCreate,
    AttestationRevocationListResponse,
    AttestationRevocationPageResponse,
    AttestationRevocationResponse,
    RevocationImpactPageResponse,
    RevocationImpactResponse,
    RevocationImpactCheckpointResponse,
    RevocationImpactImportCreate,
    RevocationImpactImportPageResponse,
    RevocationImpactImportReconciliationItem,
    RevocationImpactImportReconciliationPageResponse,
    RevocationImpactImportReconciliationResponse,
    RevocationImpactImportResponse,
    RevocationImpactPackageResponse,
    RevocationImpactVerificationCreate,
    RevocationImpactVerificationResponse,
    ImpactReconAuditCheckpointResponse,
    ImpactReconAuditEntryResponse,
    ImpactReconAuditPackageResponse,
    ImpactReconAuditVerificationCreate,
    ImpactReconAuditVerificationResponse,
    ImpactReconCheckpointResponse,
    ImpactReconEntryResponse,
    ImpactReconExchangeImportCreate,
    ImpactReconExchangeImportPageResponse,
    ImpactReconExchangeImportReconResponse,
    ImpactReconExchangeImportResponse,
    ImpactReconPackageResponse,
    ImpactReconVerificationCreate,
    ImpactReconVerificationResponse,
    AuditCheckpointImportCreate,
    AuditCheckpointImportPageResponse,
    AuditCheckpointImportReconciliationItem,
    AuditCheckpointImportReconciliationPageResponse,
    AuditCheckpointImportReconciliationResponse,
    AuditCheckpointImportResponse,
    AuditCheckpointVerificationCreate,
    AuditCheckpointVerificationResponse,
    AuditEventCheckpointPackageResponse,
    AuditEventCheckpointResponse,
    AuditEventItem,
    AuditEventPageResponse,
    AuditExchangeImportCreate,
    AuditExchangeImportPageResponse,
    AuditExchangeImportResponse,
    AuthenticationKeyRotationCreate,
    AuthenticationKeyRotationPageResponse,
    AuthenticationKeyRotationResponse,
    ClaimCreate,
    ClaimExportItem,
    ClaimListResponse,
    ClaimPageResponse,
    ClaimResponse,
    ClaimSupersessionCreate,
    ClaimSupersessionListResponse,
    ClaimSupersessionPageResponse,
    ClaimSupersessionResponse,
    ClaimSupersessionLineageItem,
    ClaimSupersessionLineageResponse,
    ContentCreate,
    ContentCoverageSearchItem,
    ContentCoverageSearchPageResponse,
    ContentEvidenceCoverageResponse,
    ContentExportJobCreate,
    ContentExportJobPageResponse,
    ContentExportJobResponse,
    ContentExportJobSummaryResponse,
    ContentExportResponse,
    ContentExportVerificationCreate,
    ContentExportVerificationResponse,
    ContentLineageItem,
    ContentLineageResponse,
    ContentPageResponse,
    ContentRelationCreate,
    ContentRelationListResponse,
    ContentRelationPageResponse,
    ContentRelationResponse,
    ContentResponse,
    CorrectionCheckpointResponse,
    CorrectionPackageResponse,
    CorrectionVerificationCreate,
    CorrectionVerificationResponse,
    EvidenceBundleCreate,
    EvidenceBundleExchangeImportCreate,
    EvidenceBundleExchangeImportPageResponse,
    EvidenceBundleExchangeImportReconciliationItem,
    EvidenceBundleExchangeImportReconciliationPageResponse,
    EvidenceBundleExchangeImportReconciliationResponse,
    EvidenceBundleExchangeImportResponse,
    EvidenceBundleExchangeManifestResponse,
    EvidenceBundleExchangePackageResponse,
    EvidenceBundleExchangeResponse,
    EvidenceBundleImportCreate,
    EvidenceBundleImportResponse,
    EvidenceBundleListResponse,
    EvidenceBundlePageResponse,
    EvidenceBundleResponse,
    ExchangeManifestVerificationCreate,
    ExchangeManifestVerificationResponse,
    TrustDecisionResponse,
    TrustEvaluationResponse,
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


_ACTORS_PARAMS = frozenset({"id", "name", "type", "limit", "cursor"})


@router.get("/actors")
async def list_actors(request: Request, session: DbSession) -> Response:
    # Read-only reviewer retrieval of existing source actors. The GET request
    # body must be empty: carrying any bytes (even whitespace or malformed
    # JSON) is a 422 validated before any parameter or actor is read. Raw
    # multi-values are inspected deliberately: a repeated scalar is rejected
    # instead of silently taking the last value, undeclared parameters are
    # rejected rather than ignored, and a blank filter/limit is never coerced
    # to a default.
    raw_body = await request.body()
    if raw_body:
        raise _query_validation_error(
            "body",
            "request body must be empty",
            "value_error.body",
        )

    raw = request.query_params

    unknown = set(raw) - _ACTORS_PARAMS
    if unknown:
        # A typo (e.g. ``actor_id``) never silently changes the retrieval.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    actor_id = _parse_nonempty_filter(raw, "id")
    name = _parse_nonempty_filter(raw, "name")
    actor_type = _parse_nonempty_filter(raw, "type")

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_ACTORS_LIMIT,
        service.MIN_ACTORS_LIMIT,
        service.MAX_ACTORS_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.actors_cursor_secret,
                pagination.ACTORS_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: every effective
        # filter (null when unfiltered) and the effective limit must match
        # exactly. A cursor minted by another endpoint family already fails
        # decoding above.
        expected = {
            "actor_id": actor_id,
            "name": name,
            "actor_type": actor_type,
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Strictly read-only: the retrieval writes no actor, migration record,
    # resource, or audit event. The total is a SQL COUNT over the filtered
    # set and the page is a SQL LIMIT/OFFSET window ordered in SQL by
    # created_at then the explicit persistence-order column, so neither value
    # depends on in-memory sorting. Filter values are never resolved for
    # existence, so an unknown or nonexistent id/name/type is an empty
    # collection rather than a 404. The actor public view carries only
    # id/name/type/created_at: no private key, raw signature, payload,
    # content, ordering column, or byte can ever be echoed.
    page, total = service.list_actors_page(
        session, actor_id, name, actor_type, page_limit, offset
    )

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.actors_cursor_secret,
                pagination.ACTORS_CURSOR,
                {
                    "actor_id": actor_id,
                    "name": name,
                    "actor_type": actor_type,
                    "limit": page_limit,
                    "offset": next_offset,
                },
            )

    result = ActorPageResponse(
        items=[ActorResponse.model_validate(item) for item in page],
        count=total,
        next_cursor=next_cursor,
    )
    # Compact UTF-8 JSON, null literal, integral numbers only, terminated by
    # exactly one newline; members appear as items, count, next_cursor.
    body = (
        json.dumps(
            result.model_dump(mode="json"),
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return Response(content=body, media_type="application/json")


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


_CONTENTS_PARAMS = frozenset(
    {
        "actor_id",
        "digest_algorithm",
        "digest_hex",
        "media_type",
        "limit",
        "cursor",
    }
)


@router.get("/contents")
async def list_contents(request: Request, session: DbSession) -> Response:
    # Read-only reviewer search over existing content identities. The GET
    # request body must be empty: carrying any bytes (even whitespace or
    # malformed JSON) is a 422 validated before any parameter or content is
    # read. Raw multi-values are inspected deliberately: a repeated scalar is
    # rejected instead of silently taking the last value, undeclared
    # parameters are rejected rather than ignored, and a blank filter/limit is
    # never coerced to a default.
    raw_body = await request.body()
    if raw_body:
        raise _query_validation_error(
            "body",
            "request body must be empty",
            "value_error.body",
        )

    raw = request.query_params

    unknown = set(raw) - _CONTENTS_PARAMS
    if unknown:
        # A typo (e.g. ``actor``) never silently changes the search.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    actor_id = _parse_nonempty_filter(raw, "actor_id")
    digest_algorithm = _parse_nonempty_filter(raw, "digest_algorithm")
    digest_hex = _parse_digest_hex_param(raw, "digest_hex")
    media_type = _parse_nonempty_filter(raw, "media_type")

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_CONTENTS_LIMIT,
        service.MIN_CONTENTS_LIMIT,
        service.MAX_CONTENTS_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.contents_cursor_secret,
                pagination.CONTENTS_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: every effective
        # filter (null when unfiltered) and the effective limit must match
        # exactly. A cursor minted by another endpoint family already fails
        # decoding above.
        expected = {
            "actor_id": actor_id,
            "digest_algorithm": digest_algorithm,
            "digest_hex": digest_hex,
            "media_type": media_type,
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Strictly read-only: the search writes no content, migration record,
    # resource, or audit event. The total is a SQL COUNT over the filtered
    # set and the page is a SQL LIMIT/OFFSET window ordered in SQL by
    # created_at then the monotonic insertion sequence, so neither value
    # depends on in-memory sorting and the order survives a restart. Filter
    # values are never resolved for existence, so an unknown or nonexistent
    # value is an empty collection rather than a 404. The content public
    # view carries only the existing fields (no ordering column or byte).
    page, total = service.list_contents_page(
        session,
        actor_id,
        digest_algorithm,
        digest_hex,
        media_type,
        page_limit,
        offset,
    )

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.contents_cursor_secret,
                pagination.CONTENTS_CURSOR,
                {
                    "actor_id": actor_id,
                    "digest_algorithm": digest_algorithm,
                    "digest_hex": digest_hex,
                    "media_type": media_type,
                    "limit": page_limit,
                    "offset": next_offset,
                },
            )

    result = ContentPageResponse(
        items=[ContentResponse.model_validate(item) for item in page],
        count=total,
        next_cursor=next_cursor,
    )
    # Compact UTF-8 JSON, null literal, integral numbers only, terminated by
    # exactly one newline; members appear as items, count, next_cursor.
    body = (
        json.dumps(
            result.model_dump(mode="json"),
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return Response(content=body, media_type="application/json")


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


_CONTENT_RELATIONS_PARAMS = frozenset(
    {
        "id",
        "content_id",
        "parent_content_id",
        "relation_type",
        "from",
        "to",
        "limit",
        "cursor",
    }
)


@router.get("/content-relations")
async def list_content_relations_global(
    request: Request, session: DbSession
) -> Response:
    # Global read-only search over the immutable lineage relations. The GET
    # request body must be empty: carrying any bytes (even whitespace,
    # arbitrary bytes, or malformed JSON) is a 422 validated before any query
    # parameter, cursor, or relation record is read. Raw multi-values are
    # inspected deliberately: a repeated scalar is rejected instead of
    # silently taking the last value, undeclared parameters are rejected
    # rather than ignored, and a blank filter/limit is never coerced to a
    # default.
    raw_body = await request.body()
    if raw_body:
        raise _query_validation_error(
            "body",
            "request body must be empty",
            "value_error.body",
        )

    raw = request.query_params

    unknown = set(raw) - _CONTENT_RELATIONS_PARAMS
    if unknown:
        # A typo (e.g. ``relation_id``) never silently changes the search.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    relation_id = _parse_nonempty_filter(raw, "id")
    content_id = _parse_nonempty_filter(raw, "content_id")
    parent_content_id = _parse_nonempty_filter(raw, "parent_content_id")

    relation_type = _parse_once(raw, "relation_type")
    if relation_type is not None and relation_type not in (
        RELATION_VERSION_OF,
        RELATION_DERIVED_FROM,
    ):
        # Covers blank/whitespace values and any spelling other than the two
        # README relation literals; no value is ever normalized.
        raise _query_validation_error(
            "relation_type",
            "relation_type must be 'version_of' or 'derived_from'",
            "value_error",
        )

    from_dt = _parse_rfc3339_param(raw, "from")
    to_dt = _parse_rfc3339_param(raw, "to")
    if from_dt is not None and to_dt is not None and from_dt > to_dt:
        raise _query_validation_error(
            "from",
            "from must not be later than to",
            "value_error.range",
        )

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_CONTENT_RELATIONS_LIMIT,
        service.MIN_CONTENT_RELATIONS_LIMIT,
        service.MAX_CONTENT_RELATIONS_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.content_relations_cursor_secret,
                pagination.CONTENT_RELATIONS_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: every effective
        # filter (null when unfiltered; the time claim canonicalizes "Z" and
        # "+00:00" to the same instant) and the effective limit must match
        # exactly. A cursor minted by another endpoint family already fails
        # decoding above.
        expected = {
            "relation_id": relation_id,
            "content_id": content_id,
            "parent_content_id": parent_content_id,
            "relation_type": relation_type,
            "from": _audit_time_claim(from_dt),
            "to": _audit_time_claim(to_dt),
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Strictly read-only: the search writes no relation, content, resource,
    # or audit event. The total is a SQL COUNT over the filtered set
    # (covering every page) and the page is a SQL LIMIT/OFFSET window
    # ordered in SQL by created_at then the monotonic insertion sequence,
    # so ordering and paging survive a restart. Filter values are never
    # resolved for existence, so an unknown relation id, content id, or
    # parent id is an empty collection rather than a 404. Each item is
    # exactly the relation public view: stable id, both endpoints, the
    # relation type, and its UTC created_at.
    page, total = service.list_content_relations_page(
        session,
        relation_id,
        content_id,
        parent_content_id,
        relation_type,
        from_dt,
        to_dt,
        page_limit,
        offset,
    )

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.content_relations_cursor_secret,
                pagination.CONTENT_RELATIONS_CURSOR,
                {
                    "relation_id": relation_id,
                    "content_id": content_id,
                    "parent_content_id": parent_content_id,
                    "relation_type": relation_type,
                    "from": _audit_time_claim(from_dt),
                    "to": _audit_time_claim(to_dt),
                    "limit": page_limit,
                    "offset": next_offset,
                },
            )

    result = ContentRelationPageResponse(
        items=[
            ContentRelationResponse.model_validate(item) for item in page
        ],
        count=total,
        next_cursor=next_cursor,
    )
    # Compact UTF-8 JSON, null literal, integral numbers only, terminated by
    # exactly one newline; members appear as items, count, next_cursor.
    body = (
        json.dumps(
            result.model_dump(mode="json"),
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return Response(content=body, media_type="application/json")


def _lineage_item(content, depth: int):
    # The full public content view, with the traversal depth added.
    fields = ContentResponse.model_validate(content).model_dump()
    return ContentLineageItem(**fields, depth=depth)


_INTEGER_RE = re.compile(r"-?[0-9]+")


def _query_validation_error(field: str, msg: str, error_type: str):
    return LineageValidationError(
        [{"loc": ["query", field], "msg": msg, "type": error_type}]
    )


def _parse_once(raw, field: str) -> str | None:
    """Return a query parameter provided at most once, else raise 422."""
    values = raw.getlist(field)
    if len(values) > 1:
        raise _query_validation_error(
            field,
            f"query parameter {field} must be provided exactly once",
            "value_error.repeated",
        )
    return values[0] if values else None


def _parse_int_param(raw, field: str, default: int, minimum: int, maximum: int):
    """Parse a strictly-formatted integer query parameter in [minimum, maximum]."""
    value = _parse_once(raw, field)
    if value is None:
        return default
    if not _INTEGER_RE.fullmatch(value):
        raise _query_validation_error(
            field,
            f"{field} must be an integer",
            "value_error.integer",
        )
    parsed = int(value)
    if not (minimum <= parsed <= maximum):
        raise _query_validation_error(
            field,
            f"{field} must be between {minimum} and {maximum}",
            "value_error.range",
        )
    return parsed


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
    relation_type: str | None = Query(default=None),
    min_depth: str | None = Query(default=None),
    limit: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
) -> ContentLineageResponse:
    # Raw multi-values are inspected deliberately: FastAPI otherwise keeps
    # only the last value of a repeated scalar parameter and Pydantic coerces
    # "8.0" to 8, both of which must be rejected rather than silently
    # defaulted or normalized.
    raw = request.query_params

    direction = _parse_once(raw, "direction")
    if direction is None:
        raise _query_validation_error(
            "direction", "Field required", "value_error.missing"
        )
    if direction not in service.LINEAGE_DIRECTIONS:
        # Covers missing values, whitespace/blank strings, and any value
        # other than the two literal traversal directions.
        raise _query_validation_error(
            "direction",
            "direction must be 'ancestors' or 'descendants'",
            "value_error",
        )

    depth = _parse_int_param(
        raw,
        "max_depth",
        service.DEFAULT_LINEAGE_MAX_DEPTH,
        service.MIN_LINEAGE_MAX_DEPTH,
        service.MAX_LINEAGE_MAX_DEPTH,
    )
    min_level = _parse_int_param(
        raw,
        "min_depth",
        service.DEFAULT_LINEAGE_MIN_DEPTH,
        service.MIN_LINEAGE_MIN_DEPTH,
        service.MAX_LINEAGE_MIN_DEPTH,
    )
    if min_level > depth:
        raise _query_validation_error(
            "min_depth",
            "min_depth must not be greater than max_depth",
            "value_error.range",
        )

    relation_type = _parse_once(raw, "relation_type")
    if relation_type is not None and relation_type not in (
        RELATION_VERSION_OF,
        RELATION_DERIVED_FROM,
    ):
        # Only the two literal edge types are accepted; blanks and casing
        # variants are rejected rather than normalized.
        raise _query_validation_error(
            "relation_type",
            "relation_type must be 'version_of' or 'derived_from'",
            "value_error",
        )

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_LINEAGE_LIMIT,
        service.MIN_LINEAGE_LIMIT,
        service.MAX_LINEAGE_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_cursor(
                request.app.state.lineage_cursor_secret, cursor
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: the effective
        # parameters (including their defaults) and the origin must match
        # exactly. A mismatch is a client validation error, not a new query.
        expected = {
            "content_id": content_id,
            "direction": direction,
            "max_depth": depth,
            "min_depth": min_level,
            "relation_type": relation_type,
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Traversal reachability, shortest depths, and first-discovery order are
    # computed without any filter; filtering only removes returned rows.
    rows = service.get_content_lineage(session, content_id, direction, depth)
    filtered = [
        (content, reached_depth)
        for content, reached_depth, edge_type in rows
        if reached_depth >= min_level
        and (relation_type is None or edge_type == relation_type)
    ]
    total = len(filtered)

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        page = filtered[offset : offset + page_limit]
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_cursor(
                request.app.state.lineage_cursor_secret,
                {
                    "content_id": content_id,
                    "direction": direction,
                    "max_depth": depth,
                    "min_depth": min_level,
                    "relation_type": relation_type,
                    "limit": page_limit,
                    "offset": next_offset,
                },
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


_CLAIMS_PARAMS = frozenset(
    {
        "content_id",
        "actor_id",
        "claim_type",
        "payload_digest_hex",
        "limit",
        "cursor",
    }
)

_HEX64_LOWER_RE = re.compile(r"[0-9a-f]{64}")

_BOOL_LITERALS = {"true": True, "false": False}


def _parse_bool_literal_param(raw, field: str) -> bool | None:
    """Parse an optional filter accepting only lowercase ``true``/``false``.

    Provided at most once; an absent parameter is ``None`` (unfiltered).
    Blank, whitespace-padded, capitalized (``True``/``FALSE``), numeric
    (``1``/``0``), or any other spelling is a 422 rather than coerced.
    """
    value = _parse_once(raw, field)
    if value is None:
        return None
    if value not in _BOOL_LITERALS:
        raise _query_validation_error(
            field,
            f"{field} must be exactly 'true' or 'false'",
            "value_error",
        )
    return _BOOL_LITERALS[value]


def _parse_digest_hex_param(raw, field: str) -> str | None:
    """Parse an optional strict 64-lowercase-hex digest filter, at most once."""
    value = _parse_once(raw, field)
    if value is None:
        return None
    if not value.strip():
        raise _query_validation_error(
            field, f"{field} must not be empty", "value_error"
        )
    if not _HEX64_LOWER_RE.fullmatch(value):
        # Uppercase, whitespace-padded, wrong-length, or non-hex spellings
        # are rejected rather than lowercased or trimmed.
        raise _query_validation_error(
            field,
            f"{field} must be exactly 64 lowercase hexadecimal characters",
            "value_error",
        )
    return value


@router.get("/claims", response_model=ClaimPageResponse)
def list_claims(
    request: Request,
    session: DbSession,
    content_id: str | None = Query(default=None),
    actor_id: str | None = Query(default=None),
    claim_type: str | None = Query(default=None),
    payload_digest_hex: str | None = Query(default=None),
    limit: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
) -> ClaimPageResponse:
    # Read-only reviewer search over existing immutable claims. Raw
    # multi-values are inspected deliberately: a repeated scalar is rejected
    # instead of silently taking the last value, undeclared parameters are
    # rejected rather than ignored, and a blank filter/limit is never coerced
    # to a default.
    raw = request.query_params

    unknown = set(raw) - _CLAIMS_PARAMS
    if unknown:
        # A typo (e.g. ``claim_types``) never silently changes the search.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    content_id = _parse_nonempty_filter(raw, "content_id")
    actor_id = _parse_nonempty_filter(raw, "actor_id")
    claim_type = _parse_nonempty_filter(raw, "claim_type")
    payload_digest_hex = _parse_digest_hex_param(raw, "payload_digest_hex")

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_CLAIMS_LIMIT,
        service.MIN_CLAIMS_LIMIT,
        service.MAX_CLAIMS_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.claims_cursor_secret,
                pagination.CLAIMS_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: every effective
        # filter and the effective limit must match exactly.
        expected = {
            "content_id": content_id,
            "actor_id": actor_id,
            "claim_type": claim_type,
            "payload_digest_hex": payload_digest_hex,
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Strictly read-only: the search writes no resource, claim, or audit
    # event. No filter value is resolved for existence, so no match is an
    # empty collection rather than a 404.
    items = service.list_claims(
        session, content_id, actor_id, claim_type, payload_digest_hex
    )
    total = len(items)

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        page = items[offset : offset + page_limit]
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.claims_cursor_secret,
                pagination.CLAIMS_CURSOR,
                {
                    "content_id": content_id,
                    "actor_id": actor_id,
                    "claim_type": claim_type,
                    "payload_digest_hex": payload_digest_hex,
                    "limit": page_limit,
                    "offset": next_offset,
                },
            )

    # Each item is the existing claim public view: associations, digest, and
    # UTC timestamp. The raw payload is never stored and so can never be
    # echoed.
    return ClaimPageResponse(
        items=[ClaimResponse.model_validate(item) for item in page],
        count=total,
        next_cursor=next_cursor,
    )


@router.get("/contents/{content_id}/claims", response_model=ClaimListResponse)
def list_content_claims(content_id: str, session: DbSession) -> ClaimListResponse:
    items = service.list_claims_for_content(session, content_id)
    return ClaimListResponse(
        items=[ClaimResponse.model_validate(item) for item in items],
        count=len(items),
    )


def _claim_supersession_response(record) -> ClaimSupersessionResponse:
    return ClaimSupersessionResponse.model_validate(record)


@router.post(
    "/claim-supersessions", response_model=ClaimSupersessionResponse
)
def create_claim_supersession(
    payload: ClaimSupersessionCreate, session: DbSession, response: Response
) -> ClaimSupersessionResponse:
    record, created = service.create_claim_supersession(session, payload)
    # First creation -> 201; a retried submission of the same three fields
    # -> 200 with the original record and no new audit event. A different
    # reason is an independent record with its own 201.
    response.status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return _claim_supersession_response(record)


@router.get(
    "/claim-supersessions/{supersession_id}",
    response_model=ClaimSupersessionResponse,
)
def get_claim_supersession(
    supersession_id: str, session: DbSession
) -> ClaimSupersessionResponse:
    # Strictly read-only: the read writes no resource and no audit event. An
    # unknown id is an explicit, specific 404 carrying the requested id.
    record = service.get_claim_supersession(session, supersession_id)
    return _claim_supersession_response(record)


_CLAIM_SUPERSESSIONS_PARAMS = frozenset(
    {
        "id",
        "superseded_claim_id",
        "replacement_claim_id",
        "reason",
        "from",
        "to",
        "limit",
        "cursor",
    }
)


@router.get("/claim-supersessions")
async def list_claim_supersessions(
    request: Request, session: DbSession
) -> Response:
    # Global read-only search over the immutable claim supersessions. The
    # GET request body must be empty: carrying any bytes (even whitespace,
    # arbitrary bytes, or malformed JSON) is a 422 validated before any
    # query parameter, cursor, or supersession record is read. Raw
    # multi-values are inspected deliberately: a repeated scalar is
    # rejected instead of silently taking the last value, undeclared
    # parameters are rejected rather than ignored, and a blank filter/limit
    # is never coerced to a default.
    raw_body = await request.body()
    if raw_body:
        raise _query_validation_error(
            "body",
            "request body must be empty",
            "value_error.body",
        )

    raw = request.query_params

    unknown = set(raw) - _CLAIM_SUPERSESSIONS_PARAMS
    if unknown:
        # A typo (e.g. ``supersession_id``) never silently changes the
        # search.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    supersession_id = _parse_nonempty_filter(raw, "id")
    superseded_claim_id = _parse_nonempty_filter(raw, "superseded_claim_id")
    replacement_claim_id = _parse_nonempty_filter(
        raw, "replacement_claim_id"
    )
    reason = _parse_nonempty_filter(raw, "reason")

    from_dt = _parse_rfc3339_param(raw, "from")
    to_dt = _parse_rfc3339_param(raw, "to")
    if from_dt is not None and to_dt is not None and from_dt > to_dt:
        raise _query_validation_error(
            "from",
            "from must not be later than to",
            "value_error.range",
        )

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_CLAIM_SUPERSESSIONS_LIMIT,
        service.MIN_CLAIM_SUPERSESSIONS_LIMIT,
        service.MAX_CLAIM_SUPERSESSIONS_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.claim_supersessions_cursor_secret,
                pagination.CLAIM_SUPERSESSIONS_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: every effective
        # filter (null when unfiltered; the time claim canonicalizes "Z" and
        # "+00:00" to the same instant) and the effective limit must match
        # exactly. A cursor minted by another endpoint family (including the
        # supersession-lineage family) already fails decoding above.
        expected = {
            "supersession_id": supersession_id,
            "superseded_claim_id": superseded_claim_id,
            "replacement_claim_id": replacement_claim_id,
            "reason": reason,
            "from": _audit_time_claim(from_dt),
            "to": _audit_time_claim(to_dt),
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Strictly read-only: the search writes no supersession, claim,
    # resource, or audit event. The total is a SQL COUNT over the filtered
    # set (covering every page) and the page is a SQL LIMIT/OFFSET window
    # ordered in SQL by created_at then the monotonic insertion sequence,
    # so ordering and paging survive a restart. Filter values are never
    # resolved for existence, so an unknown id, endpoint, or reason is an
    # empty collection rather than a 404. Each item is exactly the
    # supersession public view: stable id, both endpoint ids, the reason,
    # and its UTC created_at — never claim payloads, signatures, content,
    # or evidence bytes.
    page, total = service.list_claim_supersessions_page(
        session,
        supersession_id,
        superseded_claim_id,
        replacement_claim_id,
        reason,
        from_dt,
        to_dt,
        page_limit,
        offset,
    )

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.claim_supersessions_cursor_secret,
                pagination.CLAIM_SUPERSESSIONS_CURSOR,
                {
                    "supersession_id": supersession_id,
                    "superseded_claim_id": superseded_claim_id,
                    "replacement_claim_id": replacement_claim_id,
                    "reason": reason,
                    "from": _audit_time_claim(from_dt),
                    "to": _audit_time_claim(to_dt),
                    "limit": page_limit,
                    "offset": next_offset,
                },
            )

    result = ClaimSupersessionPageResponse(
        items=[
            _claim_supersession_response(item) for item in page
        ],
        count=total,
        next_cursor=next_cursor,
    )
    # Compact UTF-8 JSON, null literal, integral numbers only, terminated by
    # exactly one newline; members appear as items, count, next_cursor.
    body = (
        json.dumps(
            result.model_dump(mode="json"),
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return Response(content=body, media_type="application/json")


@router.get(
    "/claims/{claim_id}/supersessions",
    response_model=ClaimSupersessionListResponse,
)
def list_claim_supersessions(
    claim_id: str, session: DbSession
) -> ClaimSupersessionListResponse:
    # Records where the claim is either endpoint, in stable creation order.
    # An unknown claim is the existing claim_not_found 404, not an empty
    # collection. The read is strictly read-only and never renders claim
    # payloads, signatures, content, or evidence bytes.
    items = service.list_supersessions_for_claim(session, claim_id)
    return ClaimSupersessionListResponse(
        items=[_claim_supersession_response(item) for item in items],
        count=len(items),
    )


_CSP_PACKAGE_PARAMS = frozenset(
    {"id", "superseded_claim_id", "replacement_claim_id", "reason", "from", "to"}
)


def _csp_package_filters(request: Request):
    """Parse and validate the correction-package query filters.

    The package route shares the claim-supersession search's filter
    contract minus pagination: only id/superseded_claim_id/
    replacement_claim_id/reason/from/to are accepted, so limit/cursor and
    every other undeclared parameter are rejected rather than ignored, a
    repeated scalar is rejected instead of silently taking the last value,
    and blank or malformed values are never coerced. Returns
    ``(supersession_id, superseded_claim_id, replacement_claim_id, reason,
    from_dt, to_dt)``.
    """
    raw = request.query_params

    unknown = set(raw) - _CSP_PACKAGE_PARAMS
    if unknown:
        # A typo or a pagination parameter never silently changes the
        # snapshot.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    supersession_id = _parse_nonempty_filter(raw, "id")
    superseded_claim_id = _parse_nonempty_filter(raw, "superseded_claim_id")
    replacement_claim_id = _parse_nonempty_filter(
        raw, "replacement_claim_id"
    )
    reason = _parse_nonempty_filter(raw, "reason")

    from_dt = _parse_rfc3339_param(raw, "from")
    to_dt = _parse_rfc3339_param(raw, "to")
    if from_dt is not None and to_dt is not None and from_dt > to_dt:
        raise _query_validation_error(
            "from",
            "from must not be later than to",
            "value_error.range",
        )

    return (
        supersession_id,
        superseded_claim_id,
        replacement_claim_id,
        reason,
        from_dt,
        to_dt,
    )


@router.get("/csp/package")
async def export_correction_package(
    request: Request, session: DbSession
) -> Response:
    # Read-only export of the whole filtered correction (claim
    # supersession) snapshot together with its auditable checkpoint. The
    # GET request body must be empty, exactly as on the supersession
    # search: carrying any bytes (even whitespace, arbitrary bytes, or
    # malformed JSON) is a 422 validated before any query parameter or
    # supersession record is read.
    raw_body = await request.body()
    if raw_body:
        raise _query_validation_error(
            "body",
            "request body must be empty",
            "value_error.body",
        )

    # Same filter contract as GET /v1/claim-supersessions, but no limit
    # and no cursor: the package always snapshots every matching
    # correction in one read. Undeclared (including limit/cursor), blank,
    # repeated, or malformed parameters are all 422 before any
    # supersession is read.
    (
        supersession_id,
        superseded_claim_id,
        replacement_claim_id,
        reason,
        from_dt,
        to_dt,
    ) = _csp_package_filters(request)

    # Strictly read-only, and both halves come from the same state read:
    # every matching supersession is read once in the records' stable
    # creation order, the corrections member renders those existing public
    # views, and the checkpoint digests exactly that same array, so the
    # two can never disagree. No resource, record, or audit event is
    # written; an empty match set yields "corrections": [] with
    # correction_count 0 and the digest of the empty array.
    records = service.list_claim_supersessions(
        session,
        supersession_id,
        superseded_claim_id,
        replacement_claim_id,
        reason,
        from_dt,
        to_dt,
    )
    correction_views = [_claim_supersession_response(item) for item in records]
    # mode="json" yields exactly the wire view served in this response's
    # corrections member (UTC datetimes as RFC 3339 strings), so the
    # checkpoint binds to exactly those corrections and is reproducible by
    # an external verifier from the package body alone.
    correction_payloads = [
        view.model_dump(mode="json") for view in correction_views
    ]
    package = CorrectionPackageResponse(
        checkpoint=CorrectionCheckpointResponse(
            checkpoint_version=CSP_CHECKPOINT_VERSION,
            digest_algorithm=CSP_DIGEST_ALGORITHM,
            correction_count=len(correction_payloads),
            corrections_digest_hex=canonical.corrections_digest_hex(
                correction_payloads
            ),
        ),
        corrections=correction_views,
    )
    # Compact UTF-8 JSON, null/boolean literals, integral numbers only,
    # terminated by exactly one newline; root members appear as
    # checkpoint, corrections.
    body = (
        json.dumps(
            package.model_dump(mode="json"),
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return Response(content=body, media_type="application/json")


@router.post(
    "/csp/verify",
    response_model=CorrectionVerificationResponse,
    response_model_exclude_none=True,
)
async def verify_correction_package(
    payload: CorrectionVerificationCreate, request: Request
) -> Response:
    # The route accepts no query parameters: any (or repeated) parameter is
    # a 422 validation_error.
    _reject_any_query_param(request)
    # Strictly stateless: no session is injected, so nothing is queried,
    # created, or modified, and no resource, audit row, or log is written.
    # Verification uses the request body alone; no supersession, claim,
    # content, or other resource id is ever resolved against local state,
    # so unknown resources, repeated requests, and differing local state
    # verify identically. Malformed JSON and every structural, field,
    # type, count, timestamp, or digest-input failure are rejected by the
    # request model as a 422 before this body runs.
    body = await request.json()
    # The digest commits to the corrections array exactly as received: the
    # raw JSON values (array order, datetime spellings), not any parsed or
    # re-serialized form. The array keeps its order; nested object keys
    # sort by Unicode code point under the checkpoint canonical rules.
    computed_digest_hex = canonical.corrections_digest_hex(
        body["corrections"]
    )
    if computed_digest_hex == payload.checkpoint.corrections_digest_hex:
        result = {"valid": True}
    else:
        result = {
            "valid": False,
            "computed_digest_hex": computed_digest_hex,
        }
    # Compact UTF-8 JSON, boolean literals, terminated by exactly one
    # newline; a match carries no field besides valid.
    data = (
        json.dumps(
            result,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return Response(content=data, media_type="application/json")


def _claim_supersession_lineage_item(
    claim, depth: int
) -> ClaimSupersessionLineageItem:
    # The full public claim view, with the traversal depth added.
    fields = ClaimResponse.model_validate(claim).model_dump()
    return ClaimSupersessionLineageItem(**fields, depth=depth)


_CLAIM_SUPERSESSION_LINEAGE_PARAMS = frozenset(
    {"direction", "max_depth", "min_depth", "limit", "cursor"}
)


@router.get(
    "/claims/{claim_id}/supersession-lineage",
    response_model=ClaimSupersessionLineageResponse,
)
def get_claim_supersession_lineage(
    claim_id: str,
    request: Request,
    session: DbSession,
    direction: str | None = Query(default=None),
    max_depth: str | None = Query(default=None),
    min_depth: str | None = Query(default=None),
    limit: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
) -> ClaimSupersessionLineageResponse:
    # Read-only reviewer multi-hop traversal over the immutable supersession
    # graph with a minimum-depth filter and stable pagination. Raw
    # multi-values are inspected deliberately: FastAPI otherwise keeps only
    # the last value of a repeated scalar parameter and Pydantic coerces
    # "8.0" to 8, both of which must be rejected rather than silently
    # defaulted or normalized. Undeclared parameters are rejected rather
    # than ignored.
    raw = request.query_params

    unknown = set(raw) - _CLAIM_SUPERSESSION_LINEAGE_PARAMS
    if unknown:
        # A typo never silently changes the traversal.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    direction = _parse_once(raw, "direction")
    if direction is None:
        raise _query_validation_error(
            "direction", "Field required", "value_error.missing"
        )
    if direction not in service.SUPERSESSION_LINEAGE_DIRECTIONS:
        # Covers missing values, whitespace/blank strings, and any value
        # other than the two literal traversal directions.
        raise _query_validation_error(
            "direction",
            "direction must be 'newer' or 'older'",
            "value_error",
        )

    depth = _parse_int_param(
        raw,
        "max_depth",
        service.DEFAULT_SUPERSESSION_LINEAGE_MAX_DEPTH,
        service.MIN_SUPERSESSION_LINEAGE_MAX_DEPTH,
        service.MAX_SUPERSESSION_LINEAGE_MAX_DEPTH,
    )
    min_level = _parse_int_param(
        raw,
        "min_depth",
        service.DEFAULT_SUPERSESSION_LINEAGE_MIN_DEPTH,
        service.MIN_SUPERSESSION_LINEAGE_MIN_DEPTH,
        service.MAX_SUPERSESSION_LINEAGE_MIN_DEPTH,
    )
    if min_level > depth:
        # The minimum depth is bounded by the effective maximum; it is never
        # silently clamped or otherwise rewritten.
        raise _query_validation_error(
            "min_depth",
            "min_depth must not be greater than max_depth",
            "value_error.range",
        )

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_SUPERSESSION_LINEAGE_LIMIT,
        service.MIN_SUPERSESSION_LINEAGE_LIMIT,
        service.MAX_SUPERSESSION_LINEAGE_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.claim_supersession_lineage_cursor_secret,
                pagination.CLAIM_SUPERSESSION_LINEAGE_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: the origin,
        # direction, and every effective depth/limit value (including their
        # defaults) must match exactly. A cursor minted by another endpoint
        # family already fails decoding above.
        expected = {
            "claim_id": claim_id,
            "direction": direction,
            "max_depth": depth,
            "min_depth": min_level,
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Parameters are validated first; a structurally valid request for an
    # unknown origin claim is the existing claim_not_found 404, not an empty
    # collection. Traversal reachability, shortest depths, and first-
    # discovery order are computed without the minimum-depth filter:
    # min_depth only removes returned rows, never pruning the walk.
    rows = service.get_claim_supersession_lineage(
        session, claim_id, direction, depth
    )
    filtered = [
        (claim, reached_depth)
        for claim, reached_depth in rows
        if reached_depth >= min_level
    ]
    total = len(filtered)

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        page = filtered[offset : offset + page_limit]
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.claim_supersession_lineage_cursor_secret,
                pagination.CLAIM_SUPERSESSION_LINEAGE_CURSOR,
                {
                    "claim_id": claim_id,
                    "direction": direction,
                    "max_depth": depth,
                    "min_depth": min_level,
                    "limit": page_limit,
                    "offset": next_offset,
                },
            )

    # The traversal is read-only, dedups at shortest depth, and terminates
    # over anomalous cyclic history; it renders only existing claim public
    # views plus depth, never the payload.
    return ClaimSupersessionLineageResponse(
        items=[_claim_supersession_lineage_item(claim, d) for claim, d in page],
        count=total,
        next_cursor=next_cursor,
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


@router.post(
    "/evidence-bundle-imports",
    response_model=EvidenceBundleImportResponse,
)
def import_evidence_bundles(
    payload: EvidenceBundleImportCreate, session: DbSession, response: Response
) -> EvidenceBundleImportResponse:
    items, created = service.create_evidence_bundle_imports(session, payload)
    # At least one new bundle -> 201; a batch of only existing identities
    # -> 200, with each identity's original public view and no new writes.
    response.status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return EvidenceBundleImportResponse(
        items=[_bundle_response(item) for item in items],
        count=len(items),
    )


@router.get(
    "/evidence-bundles/{evidence_bundle_id}",
    response_model=EvidenceBundleResponse,
)
def get_evidence_bundle(
    evidence_bundle_id: str, session: DbSession
) -> EvidenceBundleResponse:
    bundle = service.get_evidence_bundle(session, evidence_bundle_id)
    return _bundle_response(bundle)


_EVIDENCE_BUNDLES_PARAMS = frozenset(
    {
        "claim_id",
        "evidence_type",
        "media_type",
        "digest_hex",
        "limit",
        "cursor",
    }
)


@router.get("/evidence-bundles", response_model=EvidenceBundlePageResponse)
def list_evidence_bundles(
    request: Request,
    session: DbSession,
    claim_id: str | None = Query(default=None),
    evidence_type: str | None = Query(default=None),
    media_type: str | None = Query(default=None),
    digest_hex: str | None = Query(default=None),
    limit: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
) -> EvidenceBundlePageResponse:
    # Read-only reviewer search over all existing evidence bundles. Raw
    # multi-values are inspected deliberately: a repeated scalar is rejected
    # instead of silently taking the last value, undeclared parameters are
    # rejected rather than ignored, and a blank filter/limit is never coerced
    # to a default.
    raw = request.query_params

    unknown = set(raw) - _EVIDENCE_BUNDLES_PARAMS
    if unknown:
        # A typo (e.g. ``evidence_types``) never silently changes the search.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    claim_id = _parse_nonempty_filter(raw, "claim_id")
    evidence_type = _parse_nonempty_filter(raw, "evidence_type")
    media_type = _parse_nonempty_filter(raw, "media_type")
    digest_hex = _parse_digest_hex_param(raw, "digest_hex")

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_EVIDENCE_BUNDLES_LIMIT,
        service.MIN_EVIDENCE_BUNDLES_LIMIT,
        service.MAX_EVIDENCE_BUNDLES_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.evidence_bundles_cursor_secret,
                pagination.EVIDENCE_BUNDLES_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: every effective
        # filter and the effective limit must match exactly.
        expected = {
            "claim_id": claim_id,
            "evidence_type": evidence_type,
            "media_type": media_type,
            "digest_hex": digest_hex,
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Strictly read-only: the search writes no resource, bundle, or audit
    # event. No filter value is resolved for existence, so no match is an
    # empty collection rather than a 404. Evidence bytes are never stored
    # and so can never be echoed.
    items = service.list_evidence_bundles(
        session, claim_id, evidence_type, media_type, digest_hex
    )
    total = len(items)

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        page = items[offset : offset + page_limit]
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.evidence_bundles_cursor_secret,
                pagination.EVIDENCE_BUNDLES_CURSOR,
                {
                    "claim_id": claim_id,
                    "evidence_type": evidence_type,
                    "media_type": media_type,
                    "digest_hex": digest_hex,
                    "limit": page_limit,
                    "offset": next_offset,
                },
            )

    return EvidenceBundlePageResponse(
        items=[_bundle_response(item) for item in page],
        count=total,
        next_cursor=next_cursor,
    )


def _reject_any_query_param(request: Request) -> None:
    """422 on any query parameter (known, unknown, blank, or repeated).

    The exchange-style read routes take no query parameters: any parameter
    at all is a 422 rather than silently ignored.
    """
    raw = request.query_params
    if raw:
        field = sorted(set(raw))[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )


def _exchange_snapshot_response(
    evidence_bundle_id: str, session: Session
) -> EvidenceBundleExchangeResponse:
    # Strictly read-only: the snapshot writes no resource and no audit event.
    bundle, claim, content, attestations = service.get_evidence_bundle_exchange(
        session, evidence_bundle_id
    )
    return EvidenceBundleExchangeResponse(
        content=ContentResponse.model_validate(content),
        claim=ClaimResponse.model_validate(claim),
        evidence_bundle=_bundle_response(bundle),
        attestations=[_attestation_response(item) for item in attestations],
    )


@router.get(
    "/evidence-bundles/{evidence_bundle_id}/exchange",
    response_model=EvidenceBundleExchangeResponse,
)
def get_evidence_bundle_exchange(
    evidence_bundle_id: str, request: Request, session: DbSession
) -> EvidenceBundleExchangeResponse:
    _reject_any_query_param(request)
    return _exchange_snapshot_response(evidence_bundle_id, session)


def _exchange_manifest_response(
    evidence_bundle_id: str, snapshot: EvidenceBundleExchangeResponse
) -> EvidenceBundleExchangeManifestResponse:
    """Build the four-field manifest for one already-read exchange snapshot.

    mode="json" yields exactly the wire view the snapshot serves (UTC
    datetimes as RFC 3339 strings, Base64 keys, plain booleans), so the
    digest is reproducible by an external verifier from the exchange JSON.
    """
    snapshot_json = snapshot.model_dump(mode="json")
    manifest_digest_hex = canonical.exchange_manifest_digest_hex(snapshot_json)
    return EvidenceBundleExchangeManifestResponse(
        manifest_version=EXCHANGE_MANIFEST_VERSION,
        evidence_bundle_id=evidence_bundle_id,
        digest_algorithm=EXCHANGE_MANIFEST_DIGEST_ALGORITHM,
        manifest_digest_hex=manifest_digest_hex,
    )


@router.get(
    "/evidence-bundles/{evidence_bundle_id}/exchange/manifest",
    response_model=EvidenceBundleExchangeManifestResponse,
)
def get_evidence_bundle_exchange_manifest(
    evidence_bundle_id: str, request: Request, session: DbSession
) -> EvidenceBundleExchangeManifestResponse:
    # Same boundary as the exchange snapshot: any (or repeated) query
    # parameter is a 422 before the bundle lookup.
    _reject_any_query_param(request)
    # Strictly read-only: the manifest only re-reads the existing snapshot
    # and hashes it; it writes no resource, record, or audit event. An
    # unknown bundle is the existing evidence_bundle_not_found 404.
    snapshot = _exchange_snapshot_response(evidence_bundle_id, session)
    return _exchange_manifest_response(evidence_bundle_id, snapshot)


@router.get(
    "/evidence-bundles/{evidence_bundle_id}/exchange/package",
    response_model=EvidenceBundleExchangePackageResponse,
)
def get_evidence_bundle_exchange_package(
    evidence_bundle_id: str, request: Request, session: DbSession
) -> EvidenceBundleExchangePackageResponse:
    # Same boundary as the exchange and manifest routes: any (or repeated)
    # query parameter is a 422 before the bundle lookup.
    _reject_any_query_param(request)
    # Strictly read-only, and both halves come from the same read state:
    # the snapshot is read exactly once, and the manifest digests exactly
    # that snapshot object, so the two can never disagree. No lineage is
    # traversed and no resource, record, or audit event is written; an
    # unknown bundle is the existing evidence_bundle_not_found 404.
    snapshot = _exchange_snapshot_response(evidence_bundle_id, session)
    manifest = _exchange_manifest_response(evidence_bundle_id, snapshot)
    return EvidenceBundleExchangePackageResponse(
        snapshot=snapshot, manifest=manifest
    )


@router.post(
    "/exchange-manifest-verifications",
    response_model=ExchangeManifestVerificationResponse,
    response_model_exclude_none=True,
)
async def verify_exchange_manifest(
    payload: ExchangeManifestVerificationCreate, request: Request
) -> ExchangeManifestVerificationResponse:
    # Strictly stateless: no session is injected, so nothing is queried,
    # created, or modified, and no audit event is written. Verification
    # uses the request body alone; the bundle id is never resolved against
    # local state.
    body = await request.json()
    # The digest commits to the snapshot exactly as received: the raw JSON
    # values (root member order, array order, datetime spellings), not any
    # parsed or re-serialized form. Field validation has already guaranteed
    # every member is canonicalizable.
    computed_digest_hex = canonical.exchange_manifest_digest_hex(body["snapshot"])
    if computed_digest_hex == payload.manifest_digest_hex:
        return ExchangeManifestVerificationResponse(valid=True)
    return ExchangeManifestVerificationResponse(
        valid=False, computed_digest_hex=computed_digest_hex
    )


def _exchange_import_response(record) -> EvidenceBundleExchangeImportResponse:
    return EvidenceBundleExchangeImportResponse(
        id=record.id,
        manifest_version=record.manifest_version,
        evidence_bundle_id=record.evidence_bundle_id,
        manifest_digest_hex=record.manifest_digest_hex,
        received_at=record.created_at,
    )


@router.post(
    "/evidence-bundle-exchange-imports",
    response_model=EvidenceBundleExchangeImportResponse,
)
async def import_evidence_bundle_exchange(
    payload: EvidenceBundleExchangeImportCreate,
    request: Request,
    session: DbSession,
    response: Response,
) -> EvidenceBundleExchangeImportResponse:
    # Structure, fixed version/algorithm, strict digest spelling, the
    # existing public-view fields, and the snapshot's internal associations
    # have all passed request validation. The digest match is enforced here
    # over the raw received JSON, so root member order, array order, and
    # datetime spellings participate exactly as on the stateless
    # verification route; a mismatch is a 422 and writes nothing.
    body = await request.json()
    computed_digest_hex = canonical.exchange_manifest_digest_hex(body["snapshot"])
    if computed_digest_hex != payload.manifest.manifest_digest_hex:
        raise EvidenceBundleExchangeImportValidationError(
            "manifest_digest_mismatch",
            details={"computed_digest_hex": computed_digest_hex},
        )

    # Verification is decided entirely by the request body: the service
    # never resolves the bundle id (or any other id) against local
    # resources, so whether they exist locally cannot change the receipt.
    record, created = service.create_evidence_bundle_exchange_import(
        session, payload
    )
    # First registration of this receiving identity -> 201; a retried
    # submission -> 200 with the original record and no new audit event.
    response.status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return _exchange_import_response(record)


_EXCHANGE_IMPORTS_PARAMS = frozenset(
    {
        "manifest_version",
        "evidence_bundle_id",
        "manifest_digest_hex",
        "limit",
        "cursor",
    }
)


@router.get(
    "/evidence-bundle-exchange-imports",
    response_model=EvidenceBundleExchangeImportPageResponse,
)
def list_evidence_bundle_exchange_imports(
    request: Request,
    session: DbSession,
    manifest_version: str | None = Query(default=None),
    evidence_bundle_id: str | None = Query(default=None),
    manifest_digest_hex: str | None = Query(default=None),
    limit: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
) -> EvidenceBundleExchangeImportPageResponse:
    # Raw multi-values are inspected deliberately: a repeated scalar is
    # rejected instead of silently taking the last value, undeclared
    # parameters are rejected rather than ignored, and a blank filter/limit
    # is never coerced to a default.
    raw = request.query_params

    unknown = set(raw) - _EXCHANGE_IMPORTS_PARAMS
    if unknown:
        # A typo (e.g. ``manifest_versions``) never silently changes the
        # search.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    manifest_version = _parse_nonempty_filter(raw, "manifest_version")
    evidence_bundle_id = _parse_nonempty_filter(raw, "evidence_bundle_id")
    manifest_digest_hex = _parse_nonempty_filter(
        raw, "manifest_digest_hex"
    )

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_EXCHANGE_IMPORTS_LIMIT,
        service.MIN_EXCHANGE_IMPORTS_LIMIT,
        service.MAX_EXCHANGE_IMPORTS_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.exchange_imports_cursor_secret,
                pagination.EXCHANGE_IMPORTS_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: every effective
        # filter and the effective limit must match exactly.
        expected = {
            "manifest_version": manifest_version,
            "evidence_bundle_id": evidence_bundle_id,
            "manifest_digest_hex": manifest_digest_hex,
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Strictly read-only: the search writes no resource and no audit event.
    items = service.list_evidence_bundle_exchange_imports(
        session, manifest_version, evidence_bundle_id, manifest_digest_hex
    )
    total = len(items)

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        page = items[offset : offset + page_limit]
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.exchange_imports_cursor_secret,
                pagination.EXCHANGE_IMPORTS_CURSOR,
                {
                    "manifest_version": manifest_version,
                    "evidence_bundle_id": evidence_bundle_id,
                    "manifest_digest_hex": manifest_digest_hex,
                    "limit": page_limit,
                    "offset": next_offset,
                },
            )

    return EvidenceBundleExchangeImportPageResponse(
        items=[_exchange_import_response(item) for item in page],
        count=total,
        next_cursor=next_cursor,
    )


@router.get(
    "/evidence-bundle-exchange-imports/{import_id}",
    response_model=EvidenceBundleExchangeImportResponse,
)
def get_evidence_bundle_exchange_import(
    import_id: str, session: DbSession
) -> EvidenceBundleExchangeImportResponse:
    # Strictly read-only: a receipt read writes no resource and no audit
    # event. An unknown id is an explicit, specific 404.
    record = service.get_evidence_bundle_exchange_import(session, import_id)
    return _exchange_import_response(record)


def _reconcile_exchange_import_record(
    session: Session, record
) -> tuple[bool, str | None, bool]:
    """Compute one receipt's current local reconciliation.

    Returns ``(local_available, local_manifest_digest_hex, matches)``. The
    receipt's ``evidence_bundle_id`` is the only lookup key: the local
    bundle is resolved by that id alone, never in reverse (no digest or
    receipt search). A missing local bundle yields ``(False, None, False)``.
    When the bundle exists, the current local digest is computed under
    exactly the exchange manifest route's rules, from a single read-only
    snapshot read, and matches only character for character.
    """
    bundle = service.find_evidence_bundle(session, record.evidence_bundle_id)
    if bundle is None:
        return False, None, False
    snapshot = _exchange_snapshot_response(record.evidence_bundle_id, session)
    manifest = _exchange_manifest_response(record.evidence_bundle_id, snapshot)
    local_digest_hex = manifest.manifest_digest_hex
    return (
        True,
        local_digest_hex,
        # Exact character-for-character equality; anything else is false.
        local_digest_hex == record.manifest_digest_hex,
    )


@router.get(
    "/evidence-bundle-exchange-imports/{import_id}/reconciliation",
    response_model=EvidenceBundleExchangeImportReconciliationResponse,
)
def reconcile_evidence_bundle_exchange_import(
    import_id: str, request: Request, session: DbSession
) -> EvidenceBundleExchangeImportReconciliationResponse:
    # Same boundary as the other exchange read routes: any (or repeated)
    # query parameter is a 422 before the receipt lookup.
    _reject_any_query_param(request)
    # Strictly read-only: the reconciliation writes no resource, receipt, or
    # audit event. An unknown receipt id is the existing
    # evidence_bundle_exchange_import_not_found 404.
    record = service.get_evidence_bundle_exchange_import(session, import_id)
    local_available, local_digest_hex, matches = (
        _reconcile_exchange_import_record(session, record)
    )
    return EvidenceBundleExchangeImportReconciliationResponse(
        import_id=record.id,
        local_available=local_available,
        local_manifest_digest_hex=local_digest_hex,
        matches=matches,
    )


_EXCHANGE_IMPORT_RECONCILIATIONS_PARAMS = frozenset(
    {"local_available", "matches", "limit", "cursor"}
)


def _exchange_import_reconciliation_item(
    record,
    local_available: bool,
    local_digest_hex: str | None,
    matches: bool,
) -> EvidenceBundleExchangeImportReconciliationItem:
    # The existing single-receipt public view, with the three current local
    # reconciliation fields added; the snapshot and raw materials are absent.
    return EvidenceBundleExchangeImportReconciliationItem(
        id=record.id,
        manifest_version=record.manifest_version,
        evidence_bundle_id=record.evidence_bundle_id,
        manifest_digest_hex=record.manifest_digest_hex,
        received_at=record.created_at,
        local_available=local_available,
        local_manifest_digest_hex=local_digest_hex,
        matches=matches,
    )


@router.get(
    "/evidence-bundle-exchange-import-reconciliations",
    response_model=EvidenceBundleExchangeImportReconciliationPageResponse,
)
def list_evidence_bundle_exchange_import_reconciliations(
    request: Request,
    session: DbSession,
    local_available: str | None = Query(default=None),
    matches: str | None = Query(default=None),
    limit: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
) -> EvidenceBundleExchangeImportReconciliationPageResponse:
    # Raw multi-values are inspected deliberately: a repeated scalar is
    # rejected instead of silently taking the last value, undeclared
    # parameters are rejected rather than ignored, a blank filter or limit
    # is never coerced to a default, and the two boolean filters accept only
    # the lowercase literals ``true``/``false``. Every parameter and the
    # cursor is validated before any receipt (or local bundle) is read.
    raw = request.query_params

    unknown = set(raw) - _EXCHANGE_IMPORT_RECONCILIATIONS_PARAMS
    if unknown:
        # The collection accepts only the two filters plus limit/cursor; a
        # typo never silently changes the page.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    local_available = _parse_bool_literal_param(raw, "local_available")
    matches = _parse_bool_literal_param(raw, "matches")

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_EXCHANGE_IMPORT_RECONCILIATIONS_LIMIT,
        service.MIN_EXCHANGE_IMPORT_RECONCILIATIONS_LIMIT,
        service.MAX_EXCHANGE_IMPORT_RECONCILIATIONS_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.exchange_import_reconciliations_cursor_secret,
                pagination.EXCHANGE_IMPORT_RECONCILIATIONS_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: both effective
        # filters (null when unfiltered) and the effective limit are bound.
        expected = {
            "local_available": local_available,
            "matches": matches,
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    def _encode_cursor(next_offset: int) -> str:
        return pagination.encode_typed_cursor(
            request.app.state.exchange_import_reconciliations_cursor_secret,
            pagination.EXCHANGE_IMPORT_RECONCILIATIONS_CURSOR,
            {
                "local_available": local_available,
                "matches": matches,
                "limit": page_limit,
                "offset": next_offset,
            },
        )

    # Strictly read-only: the listing writes no resource, receipt, or audit
    # event. Receipts follow stable creation order.
    records = service.list_evidence_bundle_exchange_import_reconciliations(
        session
    )

    next_cursor: str | None = None
    items: list[EvidenceBundleExchangeImportReconciliationItem] = []
    if local_available is None and matches is None:
        # Unfiltered request: preserve the existing behavior exactly -- page
        # the receipts first and reconcile only the page's own receipts.
        total = len(records)
        if offset < total:
            page = records[offset : offset + page_limit]
            next_offset = offset + len(page)
            if next_offset < total:
                next_cursor = _encode_cursor(next_offset)
            for record in page:
                available, local_digest_hex, record_matches = (
                    _reconcile_exchange_import_record(session, record)
                )
                items.append(
                    _exchange_import_reconciliation_item(
                        record, available, local_digest_hex, record_matches
                    )
                )
    else:
        # A filter is active: each receipt is reconciled at this read using
        # only its own evidence_bundle_id, and the filters apply to those
        # read-time results (logical AND). There is no reverse lookup by
        # digest or any other resource.
        reconciled = []
        for record in records:
            available, local_digest_hex, record_matches = (
                _reconcile_exchange_import_record(session, record)
            )
            if local_available is not None and available != local_available:
                continue
            if matches is not None and record_matches != matches:
                continue
            reconciled.append(
                (record, available, local_digest_hex, record_matches)
            )
        total = len(reconciled)
        if offset < total:
            page = reconciled[offset : offset + page_limit]
            for record, available, local_digest_hex, record_matches in page:
                items.append(
                    _exchange_import_reconciliation_item(
                        record, available, local_digest_hex, record_matches
                    )
                )
            next_offset = offset + len(page)
            if next_offset < total:
                next_cursor = _encode_cursor(next_offset)

    return EvidenceBundleExchangeImportReconciliationPageResponse(
        items=items,
        count=total,
        next_cursor=next_cursor,
    )


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


@router.get(
    "/contents/{content_id}/export",
    response_model=ContentExportResponse,
)
def get_content_export(
    content_id: str, request: Request, session: DbSession
) -> ContentExportResponse:
    # The export takes no query parameters: any parameter at all (known or
    # unknown, blank or repeated) is a 422 rather than silently ignored.
    _reject_any_query_param(request)
    # Strictly read-only: the export writes no resource and no audit event.
    content, claims, bundles_by_claim = service.get_content_export(
        session, content_id
    )
    return ContentExportResponse(
        content=ContentResponse.model_validate(content),
        claims=[
            ClaimExportItem(
                **ClaimResponse.model_validate(claim).model_dump(),
                evidence_bundles=[
                    _bundle_response(bundle)
                    for bundle in bundles_by_claim[claim.id]
                ],
            )
            for claim in claims
        ],
    )


@router.get("/contents/{content_id}/evidence-coverage")
async def get_content_evidence_coverage(
    content_id: str, request: Request, session: DbSession
) -> Response:
    # Read-only evidence-coverage summary of one content. The request takes
    # no body and no query parameters: any non-empty body (including
    # whitespace or malformed JSON) and any parameter (unknown, blank, or
    # repeated) is a 422 validation_error, both rejected before anything is
    # read. A blank content identifier is a 422 as well; an unknown one is
    # the existing content_not_found 404, decided only after validation.
    raw_body = await request.body()
    if raw_body:
        raise _query_validation_error(
            "body",
            "request body must be empty",
            "value_error.body",
        )
    _reject_any_query_param(request)
    if not content_id.strip():
        raise _query_validation_error(
            "content_id", "content_id must not be empty", "value_error"
        )
    # Strictly read-only: the summary is computed from the persisted claims,
    # bundles, attestations, and revocations at read time and writes no
    # resource, snapshot, audit event, or log. Only counts and the status
    # are returned -- never raw content, claim payloads, evidence bytes,
    # raw signatures, or key material.
    result = service.get_content_evidence_coverage(session, content_id)
    # Compact UTF-8 JSON, integral counts, terminated by exactly one
    # newline; members appear as content_id, claim_count, bundle_count,
    # attestation_count, qualified_signer_count, coverage_status.
    return _compact_json_response(
        ContentEvidenceCoverageResponse(**result), status.HTTP_200_OK
    )


_CONTENT_COVERAGE_SEARCH_PARAMS = frozenset(
    {"actor_id", "media_type", "coverage_status", "limit", "cursor"}
)


def _content_coverage_search_item(content, summary: dict):
    # The full public content view, with the four coverage counts and the
    # coverage status added.
    fields = ContentResponse.model_validate(content).model_dump()
    return ContentCoverageSearchItem(**fields, **summary)


@router.get("/content-coverage-search")
async def search_content_coverage(request: Request, session: DbSession) -> Response:
    # Read-only reviewer cross-content evidence-coverage search. The GET
    # request body must be empty: carrying any bytes (even whitespace or
    # malformed JSON) is a 422 validated before any parameter or content is
    # read. Raw multi-values are inspected deliberately: a repeated scalar is
    # rejected instead of silently taking the last value, undeclared
    # parameters are rejected rather than ignored, and a blank filter/limit
    # is never coerced to a default.
    raw_body = await request.body()
    if raw_body:
        raise _query_validation_error(
            "body",
            "request body must be empty",
            "value_error.body",
        )

    raw = request.query_params

    unknown = set(raw) - _CONTENT_COVERAGE_SEARCH_PARAMS
    if unknown:
        # A typo (e.g. ``actor``) never silently changes the search.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    actor_id = _parse_nonempty_filter(raw, "actor_id")
    media_type = _parse_nonempty_filter(raw, "media_type")

    coverage_status = _parse_once(raw, "coverage_status")
    if coverage_status is not None and coverage_status not in (
        service.COVERAGE_UNCOVERED,
        service.COVERAGE_PARTIAL,
        service.COVERAGE_COVERED,
    ):
        # Only the three existing coverage literals are accepted; blanks and
        # casing variants are rejected rather than normalized.
        raise _query_validation_error(
            "coverage_status",
            "coverage_status must be 'uncovered', 'partial', or 'covered'",
            "value_error",
        )

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_CONTENT_COVERAGE_SEARCH_LIMIT,
        service.MIN_CONTENT_COVERAGE_SEARCH_LIMIT,
        service.MAX_CONTENT_COVERAGE_SEARCH_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.content_coverage_search_cursor_secret,
                pagination.CONTENT_COVERAGE_SEARCH_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: every effective
        # filter (null when unfiltered) and the effective limit must match
        # exactly. A cursor minted by another endpoint family already fails
        # decoding above.
        expected = {
            "actor_id": actor_id,
            "media_type": media_type,
            "coverage_status": coverage_status,
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Strictly read-only: the search writes no content, resource, or audit
    # event. Items follow the contents' stable creation order (created_at
    # with the monotonic insertion sequence breaking same-timestamp ties), so
    # ordering and paging survive a restart. Filter values are never resolved
    # for existence, so an unknown value is an empty collection rather than a
    # 404. Each item is exactly the content public view plus the existing
    # four coverage counts and coverage status: no claim payload, raw
    # signature, private key, content, or evidence byte is ever echoed.
    items = service.list_content_coverage(
        session, actor_id, media_type, coverage_status
    )
    total = len(items)

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        page = items[offset : offset + page_limit]
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.content_coverage_search_cursor_secret,
                pagination.CONTENT_COVERAGE_SEARCH_CURSOR,
                {
                    "actor_id": actor_id,
                    "media_type": media_type,
                    "coverage_status": coverage_status,
                    "limit": page_limit,
                    "offset": next_offset,
                },
            )

    result = ContentCoverageSearchPageResponse(
        items=[
            _content_coverage_search_item(content, summary)
            for content, summary in page
        ],
        count=total,
        next_cursor=next_cursor,
    )
    # Compact UTF-8 JSON, null literal, integral numbers only, terminated by
    # exactly one newline; members appear as items, count, next_cursor.
    body = (
        json.dumps(
            result.model_dump(mode="json"),
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return Response(content=body, media_type="application/json")


def _content_export_snapshot(session: Session, content_id: str) -> ContentExportResponse:
    """Read-only content export snapshot, built exactly as the export route."""
    content, claims, bundles_by_claim = service.get_content_export(
        session, content_id
    )
    return ContentExportResponse(
        content=ContentResponse.model_validate(content),
        claims=[
            ClaimExportItem(
                **ClaimResponse.model_validate(claim).model_dump(),
                evidence_bundles=[
                    _bundle_response(bundle)
                    for bundle in bundles_by_claim[claim.id]
                ],
            )
            for claim in claims
        ],
    )


def _content_export_result_payload(session: Session, content_id: str) -> dict:
    """Wire-shaped ``{"content", "claims"}`` snapshot persisted as a job result.

    ``model_dump(mode="json")`` yields exactly the body served by
    ``GET /v1/contents/{content_id}/export`` (UTC datetimes as RFC 3339
    strings), so a successful job's result is byte-for-byte that export.
    """
    return _content_export_snapshot(session, content_id).model_dump(mode="json")


def _export_job_response(job) -> ContentExportJobResponse:
    return ContentExportJobResponse(
        id=job.id,
        content_id=job.content_id,
        request_id=job.request_id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        result=job.result,
        error=job.error,
    )


_CONTENT_EXPORT_JOBS_PARAMS = frozenset(
    {"content_id", "request_id", "status", "from", "to", "limit", "cursor"}
)


def _parse_literal_filter(raw, field: str, allowed) -> str | None:
    """Parse an optional filter accepting only exact allowed literals.

    Provided at most once; an absent parameter is ``None`` (unfiltered).
    Blank, whitespace-padded, and differently-cased spellings are a 422
    rather than coerced or normalized.
    """
    value = _parse_once(raw, field)
    if value is None:
        return None
    if value not in allowed:
        raise _query_validation_error(
            field,
            f"{field} must be one of: {', '.join(sorted(allowed))}",
            "value_error",
        )
    return value


@router.get("/content-export-jobs")
async def list_content_export_jobs(
    request: Request,
    session: DbSession,
) -> Response:
    # Read-only reviewer search over content export jobs. The GET request
    # body must be empty: carrying any bytes (even whitespace) is a 422
    # validated before any parameter or job is read.
    raw_body = await request.body()
    if raw_body:
        raise _query_validation_error(
            "body",
            "request body must be empty",
            "value_error.body",
        )

    # Raw multi-values are inspected deliberately: a repeated scalar is
    # rejected instead of silently taking the last value, undeclared
    # parameters are rejected rather than ignored, and a blank filter/limit
    # is never coerced to a default.
    raw = request.query_params

    unknown = set(raw) - _CONTENT_EXPORT_JOBS_PARAMS
    if unknown:
        # A typo (e.g. ``content_ids``) never silently changes the search.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    content_id = _parse_nonempty_filter(raw, "content_id")
    request_id = _parse_nonempty_filter(raw, "request_id")
    status = _parse_literal_filter(raw, "status", CONTENT_EXPORT_JOB_STATES)

    from_dt = _parse_rfc3339_param(raw, "from")
    to_dt = _parse_rfc3339_param(raw, "to")
    if from_dt is not None and to_dt is not None and from_dt > to_dt:
        raise _query_validation_error(
            "from",
            "from must not be later than to",
            "value_error.range",
        )

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_CONTENT_EXPORT_JOBS_LIMIT,
        service.MIN_CONTENT_EXPORT_JOBS_LIMIT,
        service.MAX_CONTENT_EXPORT_JOBS_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.content_export_jobs_cursor_secret,
                pagination.CONTENT_EXPORT_JOBS_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: every effective
        # filter (null when unfiltered) and the effective limit must match.
        expected = {
            "content_id": content_id,
            "request_id": request_id,
            "status": status,
            "from": _audit_time_claim(from_dt),
            "to": _audit_time_claim(to_dt),
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Strictly read-only: the search writes no job and no audit event. No
    # filter value is resolved for existence, so an unknown content or
    # request id is an empty collection rather than a 404. Items reuse the
    # single-job public view; result snapshots carry only existing public
    # views, never raw content, payloads, or evidence bytes.
    items = service.list_content_export_jobs(
        session, content_id, request_id, status, from_dt, to_dt
    )
    total = len(items)

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        page = items[offset : offset + page_limit]
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.content_export_jobs_cursor_secret,
                pagination.CONTENT_EXPORT_JOBS_CURSOR,
                {
                    "content_id": content_id,
                    "request_id": request_id,
                    "status": status,
                    "from": _audit_time_claim(from_dt),
                    "to": _audit_time_claim(to_dt),
                    "limit": page_limit,
                    "offset": next_offset,
                },
            )

    result = ContentExportJobPageResponse(
        items=[_export_job_response(item) for item in page],
        count=total,
        next_cursor=next_cursor,
    )
    # Compact UTF-8 JSON, booleans/null literal, integral numbers only,
    # terminated by exactly one newline.
    body = (
        json.dumps(
            result.model_dump(mode="json"),
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return Response(content=body, media_type="application/json")


@router.get("/content-export-jobs/summary")
async def get_content_export_jobs_summary(
    request: Request, session: DbSession
) -> Response:
    # Registered before "/content-export-jobs/{job_id}" so the literal
    # "summary" segment is never captured as a job id. The summary takes no
    # body and no query parameters: any non-empty body (including whitespace
    # or malformed JSON) and any parameter (unknown, blank, or repeated) is
    # a 422 validation_error, both rejected before any job is read -- an
    # invalid request never produces a partial summary.
    raw_body = await request.body()
    if raw_body:
        raise _query_validation_error(
            "body",
            "request body must be empty",
            "value_error.body",
        )
    _reject_any_query_param(request)
    # Strictly read-only: the four counts and the oldest pending job are
    # computed from the current persisted state at read time; nothing is
    # frozen, claimed, settled, created, or audited, and the job lifecycle
    # is unchanged.
    counts, oldest = service.summarize_content_export_jobs(session)
    wait_seconds: int | None = None
    if oldest is not None:
        # Whole seconds waited so far: the current UTC instant minus the
        # job's creation time, floored to an integer.
        wait_seconds = math.floor(
            (utc_now() - oldest.created_at).total_seconds()
        )
    result = ContentExportJobSummaryResponse(
        pending=counts[CONTENT_EXPORT_JOB_PENDING],
        running=counts[CONTENT_EXPORT_JOB_RUNNING],
        succeeded=counts[CONTENT_EXPORT_JOB_SUCCEEDED],
        failed=counts[CONTENT_EXPORT_JOB_FAILED],
        oldest_pending_id=oldest.id if oldest is not None else None,
        oldest_pending_wait_seconds=wait_seconds,
    )
    # Compact UTF-8 JSON, null literal, integral numbers only, terminated by
    # exactly one newline.
    body = (
        json.dumps(
            result.model_dump(mode="json"),
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return Response(content=body, media_type="application/json")


@router.post(
    "/content-export-jobs", response_model=ContentExportJobResponse
)
def create_content_export_job(
    payload: ContentExportJobCreate, session: DbSession, response: Response
) -> ContentExportJobResponse:
    job, created = service.create_content_export_job(session, payload)
    # First registration -> 201; a retried submission for the same
    # (request_id, content_id) -> 200 with the original job and no new audit
    # event. The same request_id for a different content is a 409.
    response.status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return _export_job_response(job)


@router.get(
    "/content-export-jobs/{job_id}",
    response_model=ContentExportJobResponse,
)
def get_content_export_job(
    job_id: str, session: DbSession
) -> ContentExportJobResponse:
    # Strictly read-only: the read returns one job's public view and writes
    # no job and no audit event. An unknown id is an explicit, specific 404.
    job = service.get_content_export_job(session, job_id)
    return _export_job_response(job)


@router.post(
    "/content-export-jobs/{job_id}/run",
    response_model=ContentExportJobResponse,
)
def run_content_export_job(
    job_id: str, session: DbSession
) -> ContentExportJobResponse:
    # Atomically claim the pending job (running + UTC started_at), then build
    # the existing read-only export and settle it as succeeded (result + UTC
    # finished_at) or failed (null result, content_export_failed) in one
    # transaction with the run audit event. A job that is not pending is a
    # 409 conflict; an unknown id is a 404 content_export_job_not_found.
    job = service.run_content_export_job(
        session, job_id, _content_export_result_payload
    )
    return _export_job_response(job)


@router.post(
    "/content-export-jobs/run-next",
    response_model=ContentExportJobResponse,
)
async def run_next_content_export_job(
    request: Request, session: DbSession
) -> ContentExportJobResponse:
    # The queue claim takes no body and no query parameters: any non-empty
    # body (including whitespace, malformed JSON, a JSON object, or any
    # extra field) and any parameter (unknown, blank, or repeated) is a 422
    # validation_error, rejected before any job is read -- so a malformed
    # request never claims, creates, modifies, or audits anything.
    raw_body = await request.body()
    if raw_body:
        raise _query_validation_error(
            "body",
            "request body must be empty",
            "value_error.body",
        )
    _reject_any_query_param(request)
    # Server-side queue claim: the single oldest pending job in stable
    # creation order is atomically claimed and settled exactly as one
    # single-job run. An empty queue is a 404 content_export_job_not_found
    # with zero writes; losing the claim to a concurrent caller (the chosen
    # job was claimed first or is no longer pending) is a 409 conflict that
    # touches no job, creates no resource, and writes no audit event.
    job = service.run_next_content_export_job(
        session, _content_export_result_payload
    )
    return _export_job_response(job)


@router.post(
    "/content-export-verifications",
    response_model=ContentExportVerificationResponse,
    response_model_exclude_none=True,
)
async def verify_content_export(
    payload: ContentExportVerificationCreate, request: Request
) -> Response:
    # The route accepts no query parameters: any (or repeated) parameter is a
    # 422 validation_error.
    _reject_any_query_param(request)
    # Strictly stateless: no session is injected, so nothing is queried,
    # created, or modified, and no audit event is written. Verification uses
    # the request body alone; no content, claim, or bundle id is ever
    # resolved against local state, and no raw material is ever echoed.
    body = await request.json()
    # The digest commits to the snapshot exactly as received: the raw JSON
    # values (root member order, array order, datetime spellings), not any
    # parsed or re-serialized form. Field validation has already guaranteed
    # every member is canonicalizable, the digest is 64 lowercase hex, and
    # the snapshot's claim/bundle associations are internally consistent.
    computed_digest_hex = canonical.content_export_snapshot_digest_hex(
        body["snapshot"]
    )
    if computed_digest_hex == payload.digest_hex:
        result = {"valid": True}
    else:
        result = {
            "valid": False,
            "computed_digest_hex": computed_digest_hex,
        }
    # Compact UTF-8 JSON, boolean literals, terminated by exactly one newline.
    data = (
        json.dumps(
            result,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return Response(content=data, media_type="application/json")


def _parse_nonempty_filter(raw, field: str) -> str | None:
    """Parse a non-empty, exact-match string filter provided at most once."""
    value = _parse_once(raw, field)
    if value is None:
        return None
    if not value.strip():
        # Blank/whitespace filters are invalid rather than matched against
        # nothing or silently dropped.
        raise _query_validation_error(
            field, f"{field} must not be empty", "value_error"
        )
    return value


_NONNEGATIVE_INTEGER_RE = re.compile(r"[0-9]+")


def _parse_nonnegative_int_param(raw, field: str) -> int | None:
    """Parse an optional non-negative decimal integer filter, at most once."""
    value = _parse_once(raw, field)
    if value is None:
        return None
    if not value.strip():
        raise _query_validation_error(
            field, f"{field} must not be empty", "value_error"
        )
    if not _NONNEGATIVE_INTEGER_RE.fullmatch(value):
        # Signs, decimals, whitespace padding, and non-numeric values are
        # rejected rather than coerced.
        raise _query_validation_error(
            field,
            f"{field} must be a non-negative integer",
            "value_error.integer",
        )
    return int(value)


@router.get(
    "/contents/{content_id}/evidence-bundles",
    response_model=EvidenceBundlePageResponse,
)
def list_content_evidence_bundles(
    content_id: str,
    request: Request,
    session: DbSession,
    evidence_type: str | None = Query(default=None),
    media_type: str | None = Query(default=None),
    limit: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
) -> EvidenceBundlePageResponse:
    # Raw multi-values are inspected deliberately: a repeated scalar is
    # rejected instead of silently taking the last value, and a blank
    # filter/limit is never coerced to a default.
    raw = request.query_params

    evidence_type = _parse_nonempty_filter(raw, "evidence_type")
    media_type = _parse_nonempty_filter(raw, "media_type")
    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_CONTENT_EVIDENCE_LIMIT,
        service.MIN_CONTENT_EVIDENCE_LIMIT,
        service.MAX_CONTENT_EVIDENCE_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.content_evidence_cursor_secret,
                pagination.CONTENT_EVIDENCE_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: the origin, every
        # effective filter, and the effective limit must match exactly.
        expected = {
            "content_id": content_id,
            "evidence_type": evidence_type,
            "media_type": media_type,
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Parameters and cursor are validated first; a malformed request is a
    # 422 regardless of the origin. A structurally valid request for an
    # unknown content is a missing resource (404), not an empty collection.
    items = service.list_evidence_bundles_for_content(
        session, content_id, evidence_type, media_type
    )
    total = len(items)

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        page = items[offset : offset + page_limit]
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.content_evidence_cursor_secret,
                pagination.CONTENT_EVIDENCE_CURSOR,
                {
                    "content_id": content_id,
                    "evidence_type": evidence_type,
                    "media_type": media_type,
                    "limit": page_limit,
                    "offset": next_offset,
                },
            )

    return EvidenceBundlePageResponse(
        items=[_bundle_response(item) for item in page],
        count=total,
        next_cursor=next_cursor,
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


@router.post(
    "/attestation-revocations", response_model=AttestationRevocationResponse
)
def create_attestation_revocation(
    payload: AttestationRevocationCreate, session: DbSession, response: Response
) -> AttestationRevocationResponse:
    revocation, created = service.create_attestation_revocation(session, payload)
    # First creation -> 201; an idempotent repeat submission -> 200, and the
    # existing immutable record is returned unchanged.
    response.status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return AttestationRevocationResponse.model_validate(revocation)


@router.get(
    "/attestation-revocations/{revocation_id}",
    response_model=AttestationRevocationResponse,
)
def get_attestation_revocation(
    revocation_id: str, session: DbSession
) -> AttestationRevocationResponse:
    revocation = service.get_attestation_revocation(session, revocation_id)
    return AttestationRevocationResponse.model_validate(revocation)


@router.get(
    "/attestations/{attestation_id}/revocations",
    response_model=AttestationRevocationListResponse,
)
def list_attestation_revocations(
    attestation_id: str, session: DbSession
) -> AttestationRevocationListResponse:
    items = service.list_revocations_for_attestation(session, attestation_id)
    return AttestationRevocationListResponse(
        items=[AttestationRevocationResponse.model_validate(item) for item in items],
        count=len(items),
    )


_ATTESTATION_REVOCATIONS_PARAMS = frozenset(
    {
        "attestation_id",
        "revoker_actor_id",
        "reason",
        "from",
        "to",
        "limit",
        "cursor",
    }
)


@router.get("/attestation-revocations")
async def list_attestation_revocations_global(
    request: Request, session: DbSession
) -> Response:
    # Global read-only search over the immutable revocation records. The GET
    # request body must be empty: carrying any bytes (even whitespace,
    # arbitrary bytes, or malformed JSON) is a 422 validated before any query
    # parameter, cursor, or revocation record is read. Raw multi-values are
    # inspected deliberately: a repeated scalar is rejected instead of
    # silently taking the last value, undeclared parameters are rejected
    # rather than ignored, and a blank filter/limit is never coerced to a
    # default.
    raw_body = await request.body()
    if raw_body:
        raise _query_validation_error(
            "body",
            "request body must be empty",
            "value_error.body",
        )

    raw = request.query_params

    unknown = set(raw) - _ATTESTATION_REVOCATIONS_PARAMS
    if unknown:
        # A typo (e.g. ``revoker``) never silently changes the search.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    attestation_id = _parse_nonempty_filter(raw, "attestation_id")
    revoker_actor_id = _parse_nonempty_filter(raw, "revoker_actor_id")
    reason = _parse_nonempty_filter(raw, "reason")

    from_dt = _parse_rfc3339_param(raw, "from")
    to_dt = _parse_rfc3339_param(raw, "to")
    if from_dt is not None and to_dt is not None and from_dt > to_dt:
        raise _query_validation_error(
            "from",
            "from must not be later than to",
            "value_error.range",
        )

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_ATTESTATION_REVOCATIONS_LIMIT,
        service.MIN_ATTESTATION_REVOCATIONS_LIMIT,
        service.MAX_ATTESTATION_REVOCATIONS_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.attestation_revocations_cursor_secret,
                pagination.ATTESTATION_REVOCATIONS_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: every effective
        # filter (null when unfiltered; the time claim canonicalizes "Z" and
        # "+00:00" to the same instant) and the effective limit must match
        # exactly. A cursor minted by another endpoint family already fails
        # decoding above.
        expected = {
            "attestation_id": attestation_id,
            "revoker_actor_id": revoker_actor_id,
            "reason": reason,
            "from": _audit_time_claim(from_dt),
            "to": _audit_time_claim(to_dt),
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Strictly read-only: the search writes no revocation, attestation,
    # grant, resource, or audit event. The total is a SQL COUNT over the
    # filtered set (covering every page) and the page is a SQL
    # LIMIT/OFFSET window ordered in SQL by created_at then the monotonic
    # insertion sequence, so ordering and paging survive a restart. Filter
    # values are never resolved for existence, so an unknown attestation id,
    # revoker id, or reason is an empty collection rather than a 404. Each
    # item is exactly the revocation public view: no proof bytes, signature,
    # authentication header, private key, or content byte is ever echoed.
    page, total = service.list_attestation_revocations_page(
        session,
        attestation_id,
        revoker_actor_id,
        reason,
        from_dt,
        to_dt,
        page_limit,
        offset,
    )

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.attestation_revocations_cursor_secret,
                pagination.ATTESTATION_REVOCATIONS_CURSOR,
                {
                    "attestation_id": attestation_id,
                    "revoker_actor_id": revoker_actor_id,
                    "reason": reason,
                    "from": _audit_time_claim(from_dt),
                    "to": _audit_time_claim(to_dt),
                    "limit": page_limit,
                    "offset": next_offset,
                },
            )

    result = AttestationRevocationPageResponse(
        items=[
            AttestationRevocationResponse.model_validate(item) for item in page
        ],
        count=total,
        next_cursor=next_cursor,
    )
    # Compact UTF-8 JSON, null literal, integral numbers only, terminated by
    # exactly one newline; members appear as items, count, next_cursor.
    body = (
        json.dumps(
            result.model_dump(mode="json"),
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return Response(content=body, media_type="application/json")


_REVOCATION_IMPACTS_PARAMS = frozenset(
    {
        "attestation_id",
        "revoker_actor_id",
        "reason",
        "from",
        "to",
        "limit",
        "cursor",
    }
)


def _revocation_impact_item(impact: dict) -> RevocationImpactResponse:
    # The existing revocation public view plus the content association,
    # signing subject, and the before/after coverage fields.
    revocation = impact["revocation"]
    return RevocationImpactResponse(
        id=revocation.id,
        attestation_id=revocation.attestation_id,
        revoker_actor_id=revocation.revoker_actor_id,
        reason=revocation.reason,
        created_at=revocation.created_at,
        content_id=impact["content_id"],
        target_type=impact["target_type"],
        signer_actor_id=impact["signer_actor_id"],
        qualified_signer_count_after=impact[
            "qualified_signer_count_after"
        ],
        coverage_status_after=impact["coverage_status_after"],
        qualified_signer_count_before=impact[
            "qualified_signer_count_before"
        ],
        coverage_status_before=impact["coverage_status_before"],
        qualified_signer_count_delta=impact[
            "qualified_signer_count_delta"
        ],
    )


@router.get("/revocation-impacts")
async def list_revocation_impacts(
    request: Request, session: DbSession
) -> Response:
    # Read-only cross-content impact search over the existing revocation
    # records. The GET request body must be empty: carrying any bytes (even
    # whitespace, arbitrary bytes, or malformed JSON) is a 422 validated
    # before any query parameter, cursor, or revocation record is read. Raw
    # multi-values are inspected deliberately: a repeated scalar is rejected
    # instead of silently taking the last value, undeclared parameters are
    # rejected rather than ignored, and a blank filter/limit is never coerced
    # to a default.
    raw_body = await request.body()
    if raw_body:
        raise _query_validation_error(
            "body",
            "request body must be empty",
            "value_error.body",
        )

    raw = request.query_params

    unknown = set(raw) - _REVOCATION_IMPACTS_PARAMS
    if unknown:
        # A typo (e.g. ``revoker``) never silently changes the search.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    attestation_id = _parse_nonempty_filter(raw, "attestation_id")
    revoker_actor_id = _parse_nonempty_filter(raw, "revoker_actor_id")
    reason = _parse_nonempty_filter(raw, "reason")

    from_dt = _parse_rfc3339_param(raw, "from")
    to_dt = _parse_rfc3339_param(raw, "to")
    if from_dt is not None and to_dt is not None and from_dt > to_dt:
        raise _query_validation_error(
            "from",
            "from must not be later than to",
            "value_error.range",
        )

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_REVOCATION_IMPACTS_LIMIT,
        service.MIN_REVOCATION_IMPACTS_LIMIT,
        service.MAX_REVOCATION_IMPACTS_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.revocation_impacts_cursor_secret,
                pagination.REVOCATION_IMPACTS_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: every effective
        # filter (null when unfiltered; the time claim canonicalizes "Z" and
        # "+00:00" to the same instant) and the effective limit must match
        # exactly. A cursor minted by another endpoint family already fails
        # decoding above.
        expected = {
            "attestation_id": attestation_id,
            "revoker_actor_id": revoker_actor_id,
            "reason": reason,
            "from": _audit_time_claim(from_dt),
            "to": _audit_time_claim(to_dt),
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Strictly read-only: the search writes no revocation, attestation,
    # grant, resource, or audit event. Filter values are never resolved for
    # existence, so an unknown attestation id, revoker id, or reason is an
    # empty collection rather than a 404. Each item reuses the revocation
    # public view and adds only the content association, signing subject,
    # and the before/after counts and statuses: no proof bytes, signature,
    # authentication header, private key, payload, or content byte is ever
    # echoed.
    page, total = service.list_revocation_impacts_page(
        session,
        attestation_id,
        revoker_actor_id,
        reason,
        from_dt,
        to_dt,
        page_limit,
        offset,
    )

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.revocation_impacts_cursor_secret,
                pagination.REVOCATION_IMPACTS_CURSOR,
                {
                    "attestation_id": attestation_id,
                    "revoker_actor_id": revoker_actor_id,
                    "reason": reason,
                    "from": _audit_time_claim(from_dt),
                    "to": _audit_time_claim(to_dt),
                    "limit": page_limit,
                    "offset": next_offset,
                },
            )

    result = RevocationImpactPageResponse(
        items=[_revocation_impact_item(impact) for impact in page],
        count=total,
        next_cursor=next_cursor,
    )
    # Compact UTF-8 JSON, null literal, integral numbers only (never a
    # negative zero), terminated by exactly one newline; members appear as
    # items, count, next_cursor.
    body = (
        json.dumps(
            result.model_dump(mode="json"),
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return Response(content=body, media_type="application/json")


_REVOCATION_IMPACT_PACKAGE_PARAMS = frozenset(
    {"attestation_id", "revoker_actor_id", "reason", "from", "to"}
)


def _revocation_impact_package_filters(request: Request):
    """Parse and validate the impact-package query filters.

    The package route shares the impact search's parameter contract minus
    pagination: only attestation_id/revoker_actor_id/reason/from/to are
    accepted, so limit/cursor and every other undeclared parameter are
    rejected rather than ignored, a repeated scalar is rejected instead
    of silently taking the last value, and blank or malformed values are
    never coerced. Returns
    ``(attestation_id, revoker_actor_id, reason, from_dt, to_dt)``.
    """
    raw = request.query_params

    unknown = set(raw) - _REVOCATION_IMPACT_PACKAGE_PARAMS
    if unknown:
        # A typo (e.g. ``revoker``) or a pagination parameter never
        # silently changes the snapshot.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    attestation_id = _parse_nonempty_filter(raw, "attestation_id")
    revoker_actor_id = _parse_nonempty_filter(raw, "revoker_actor_id")
    reason = _parse_nonempty_filter(raw, "reason")

    from_dt = _parse_rfc3339_param(raw, "from")
    to_dt = _parse_rfc3339_param(raw, "to")
    if from_dt is not None and to_dt is not None and from_dt > to_dt:
        raise _query_validation_error(
            "from",
            "from must not be later than to",
            "value_error.range",
        )

    return attestation_id, revoker_actor_id, reason, from_dt, to_dt


@router.get("/revocation-impact-package")
async def export_revocation_impact_package(
    request: Request, session: DbSession
) -> Response:
    # Read-only export of the whole filtered revocation-impact snapshot
    # together with its auditable checkpoint. The GET request body must be
    # empty, exactly as on the impact search: carrying any bytes (even
    # whitespace, arbitrary bytes, or malformed JSON) is a 422 validated
    # before any query parameter or revocation record is read.
    raw_body = await request.body()
    if raw_body:
        raise _query_validation_error(
            "body",
            "request body must be empty",
            "value_error.body",
        )

    # Same filter contract as GET /v1/revocation-impacts, but no limit and
    # no cursor: the package always snapshots every matching impact in one
    # read. Undeclared (including limit/cursor), blank, repeated, or
    # malformed parameters are all 422 before any revocation is read.
    (
        attestation_id,
        revoker_actor_id,
        reason,
        from_dt,
        to_dt,
    ) = _revocation_impact_package_filters(request)

    # Strictly read-only, and both halves come from the same state read:
    # every matching impact is read once in the revocations' stable
    # creation order, the impacts member renders those existing public
    # views, and the checkpoint digests exactly that same array, so the
    # two can never disagree. No resource, record, or audit event is
    # written; an empty match set yields "impacts": [] with impact_count 0
    # and the digest of the empty array.
    impacts = service.list_revocation_impacts(
        session,
        attestation_id,
        revoker_actor_id,
        reason,
        from_dt,
        to_dt,
    )
    impact_views = [_revocation_impact_item(impact) for impact in impacts]
    # mode="json" yields exactly the wire view served in this response's
    # impacts member (UTC datetimes as RFC 3339 strings), so the
    # checkpoint binds to exactly those impacts and is reproducible by an
    # external verifier from the package body alone.
    impact_payloads = [view.model_dump(mode="json") for view in impact_views]
    package = RevocationImpactPackageResponse(
        checkpoint=RevocationImpactCheckpointResponse(
            checkpoint_version=REVOCATION_IMPACT_CHECKPOINT_VERSION,
            digest_algorithm=REVOCATION_IMPACT_DIGEST_ALGORITHM,
            impact_count=len(impact_payloads),
            impacts_digest_hex=canonical.revocation_impacts_digest_hex(
                impact_payloads
            ),
        ),
        impacts=impact_views,
    )
    # Compact UTF-8 JSON, null/boolean literals, integral numbers only,
    # terminated by exactly one newline; root members appear as
    # checkpoint, impacts.
    body = (
        json.dumps(
            package.model_dump(mode="json"),
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return Response(content=body, media_type="application/json")


@router.post(
    "/impact-verifications",
    response_model=RevocationImpactVerificationResponse,
    response_model_exclude_none=True,
)
async def verify_revocation_impacts(
    payload: RevocationImpactVerificationCreate, request: Request
) -> Response:
    # The route accepts no query parameters: any (or repeated) parameter is
    # a 422 validation_error.
    _reject_any_query_param(request)
    # Strictly stateless: no session is injected, so nothing is queried,
    # created, or modified, and no resource, audit row, or log is written.
    # Verification uses the request body alone; no revocation,
    # attestation, actor, content, or bundle id is ever resolved against
    # local state, so unknown resources, repeated requests, and differing
    # local state verify identically. Malformed JSON and every
    # structural, field, type, association, or digest-input failure are
    # rejected by the request model as a 422 before this body runs.
    body = await request.json()
    # The digest commits to the impacts array exactly as received: the raw
    # JSON values (array order, datetime spellings), not any parsed or
    # re-serialized form. The array keeps its order; nested object keys
    # sort by Unicode code point under the checkpoint canonical rules.
    computed_digest_hex = canonical.revocation_impacts_digest_hex(
        body["impacts"]
    )
    if computed_digest_hex == payload.checkpoint.impacts_digest_hex:
        result = {"valid": True}
    else:
        result = {
            "valid": False,
            "computed_digest_hex": computed_digest_hex,
        }
    # Compact UTF-8 JSON, boolean literals, terminated by exactly one
    # newline; a match carries no field besides valid.
    data = (
        json.dumps(
            result,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return Response(content=data, media_type="application/json")


def _impact_import_response(record) -> RevocationImpactImportResponse:
    return RevocationImpactImportResponse(
        id=record.id,
        checkpoint_version=record.checkpoint_version,
        impact_count=record.impact_count,
        impacts_digest_hex=record.impacts_digest_hex,
        received_at=record.created_at,
    )


@router.post(
    "/impact-imports",
    response_model=RevocationImpactImportResponse,
)
async def import_revocation_impacts(
    payload: RevocationImpactImportCreate,
    request: Request,
    session: DbSession,
    response: Response,
) -> RevocationImpactImportResponse:
    # Structure, fixed version/algorithm, strict digest spelling, the
    # impact-item fields and timestamps, the internal before/after coverage
    # associations, and the claimed impact count have all passed request
    # validation exactly as on the stateless verification route. The digest
    # match is enforced here over the raw received JSON, so array order and
    # datetime spellings participate exactly as received; a mismatch is a
    # 422 and writes nothing.
    body = await request.json()
    computed_digest_hex = canonical.revocation_impacts_digest_hex(
        body["impacts"]
    )
    if computed_digest_hex != payload.checkpoint.impacts_digest_hex:
        raise ImpactImportValidationError(
            "impacts_digest_mismatch",
            details={"computed_digest_hex": computed_digest_hex},
        )

    # Verification is decided entirely by the request body: the described
    # impacts are never resolved against local state, so whether they exist
    # locally cannot change the receipt, and no revocation or other resource
    # is created or modified.
    record, created = service.create_revocation_impact_import(session, payload)
    # First registration of this receiving identity -> 201; a retried
    # submission -> 200 with the original record and no new audit event.
    response.status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return _impact_import_response(record)


_IMPACT_IMPORTS_PARAMS = frozenset(
    {
        "checkpoint_version",
        "impacts_digest_hex",
        "impact_count",
        "limit",
        "cursor",
    }
)


@router.get("/impact-imports")
async def list_revocation_impact_imports(
    request: Request, session: DbSession
) -> Response:
    # Read-only reviewer search over the registered impact-import receipts.
    # The GET request body must be empty: carrying any bytes (even
    # whitespace or malformed JSON) is a 422 validated before any parameter
    # or receipt is read. Raw multi-values are inspected deliberately: a
    # repeated scalar is rejected instead of silently taking the last value,
    # undeclared parameters are rejected rather than ignored, and a blank
    # filter/limit is never coerced to a default.
    raw_body = await request.body()
    if raw_body:
        raise _query_validation_error(
            "body",
            "request body must be empty",
            "value_error.body",
        )

    raw = request.query_params

    unknown = set(raw) - _IMPACT_IMPORTS_PARAMS
    if unknown:
        # A typo (e.g. ``checkpoint_versions``) never silently changes the
        # search.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    checkpoint_version = _parse_nonempty_filter(raw, "checkpoint_version")
    impacts_digest_hex = _parse_nonempty_filter(raw, "impacts_digest_hex")
    impact_count = _parse_nonnegative_int_param(raw, "impact_count")

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_IMPACT_IMPORTS_LIMIT,
        service.MIN_IMPACT_IMPORTS_LIMIT,
        service.MAX_IMPACT_IMPORTS_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.impact_imports_cursor_secret,
                pagination.IMPACT_IMPORTS_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: every effective
        # filter (null when unfiltered) and the effective limit must match
        # exactly. A cursor minted by another endpoint family already fails
        # decoding above.
        expected = {
            "checkpoint_version": checkpoint_version,
            "impacts_digest_hex": impacts_digest_hex,
            "impact_count": impact_count,
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Strictly read-only: the search writes no receipt, resource, or audit
    # event. The total is the filtered count covering every page. Filter
    # values are never resolved for existence, so an unknown or nonexistent
    # value is an empty collection rather than a 404. Each item is exactly
    # the single-receipt public view; the impacts array and every raw
    # impact, signature, payload, content, or evidence datum are never
    # persisted and so can never be echoed.
    items = service.list_revocation_impact_imports(
        session, checkpoint_version, impacts_digest_hex, impact_count
    )
    total = len(items)

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        page = items[offset : offset + page_limit]
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.impact_imports_cursor_secret,
                pagination.IMPACT_IMPORTS_CURSOR,
                {
                    "checkpoint_version": checkpoint_version,
                    "impacts_digest_hex": impacts_digest_hex,
                    "impact_count": impact_count,
                    "limit": page_limit,
                    "offset": next_offset,
                },
            )

    result = RevocationImpactImportPageResponse(
        items=[_impact_import_response(item) for item in page],
        count=total,
        next_cursor=next_cursor,
    )
    # Compact UTF-8 JSON, null literal, integral numbers only, terminated by
    # exactly one newline; members appear as items, count, next_cursor.
    body = (
        json.dumps(
            result.model_dump(mode="json"),
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return Response(content=body, media_type="application/json")


@router.get(
    "/impact-imports/{import_id}",
    response_model=RevocationImpactImportResponse,
)
def get_revocation_impact_import(
    import_id: str, request: Request, session: DbSession
) -> RevocationImpactImportResponse:
    # The receipt read takes no query parameters: any (or repeated)
    # parameter is a 422 before the receipt lookup.
    _reject_any_query_param(request)
    # Strictly read-only: a receipt read writes no resource and no audit
    # event. An unknown id is an explicit, specific 404.
    record = service.get_revocation_impact_import(session, import_id)
    return _impact_import_response(record)


def _local_revocation_impact_checkpoint(
    session: Session,
) -> RevocationImpactCheckpointResponse:
    """Compute the current unfiltered local revocation-impact checkpoint.

    Exactly the ``GET /v1/revocation-impact-package`` rules with no
    filters: every local impact is read once in the revocations' stable
    creation order and rendered through the same public wire view, so the
    fixed version, digest algorithm, impact count, canonical digest, and
    UTC representation are identical to that route. Strictly read-only.
    """
    impacts = service.list_revocation_impacts(session)
    impact_views = [_revocation_impact_item(impact) for impact in impacts]
    # mode="json" yields exactly the wire view the package export serves
    # (UTC datetimes as RFC 3339 strings), so the digest is reproducible by
    # an external verifier from the package body alone.
    impact_payloads = [view.model_dump(mode="json") for view in impact_views]
    return RevocationImpactCheckpointResponse(
        checkpoint_version=REVOCATION_IMPACT_CHECKPOINT_VERSION,
        digest_algorithm=REVOCATION_IMPACT_DIGEST_ALGORITHM,
        impact_count=len(impact_payloads),
        impacts_digest_hex=canonical.revocation_impacts_digest_hex(
            impact_payloads
        ),
    )


@router.get(
    "/impact-imports/{import_id}/recon",
    response_model=RevocationImpactImportReconciliationResponse,
)
def reconcile_revocation_impact_import(
    import_id: str, request: Request, session: DbSession
) -> RevocationImpactImportReconciliationResponse:
    # Same boundary as the receipt read route: any (or repeated) query
    # parameter is a 422 before the receipt lookup.
    _reject_any_query_param(request)
    # Strictly read-only: the reconciliation writes no resource, receipt, or
    # audit event. An unknown receipt id is the impact_import_not_found 404.
    record = service.get_revocation_impact_import(session, import_id)

    # The current local impact set is always read complete and unfiltered;
    # the receipt's imported impacts array is not persisted and is never
    # read or echoed.
    local_checkpoint = _local_revocation_impact_checkpoint(session)
    matches = (
        record.checkpoint_version == local_checkpoint.checkpoint_version
        and record.impact_count == local_checkpoint.impact_count
        and record.impacts_digest_hex == local_checkpoint.impacts_digest_hex
    )
    return RevocationImpactImportReconciliationResponse(
        import_id=record.id,
        local_checkpoint=local_checkpoint,
        matches=matches,
    )


_IMPACT_IMPORT_RECONCILIATIONS_PARAMS = frozenset(
    {"local_available", "matches", "limit", "cursor"}
)


def _impact_import_reconciliation_item(
    record,
    local_available: bool,
    local_checkpoint: RevocationImpactCheckpointResponse,
    matches: bool,
) -> RevocationImpactImportReconciliationItem:
    # The existing single-receipt public view, with the local availability
    # flag, the current unfiltered local checkpoint, and the match verdict
    # added; the imported impacts array is absent.
    return RevocationImpactImportReconciliationItem(
        id=record.id,
        checkpoint_version=record.checkpoint_version,
        impact_count=record.impact_count,
        impacts_digest_hex=record.impacts_digest_hex,
        received_at=record.created_at,
        local_available=local_available,
        local_checkpoint=local_checkpoint,
        matches=matches,
    )


@router.get("/impact-import-reconciliations")
async def list_revocation_impact_import_reconciliations(
    request: Request, session: DbSession
) -> Response:
    # Read-only, paginated reconciliation of every registered impact-import
    # receipt against the current local revocation-impact set. The GET
    # request body must be empty: carrying any bytes (even whitespace or
    # malformed JSON) is a 422 validated before any parameter, receipt, or
    # local impact is read. Raw multi-values are inspected deliberately: a
    # repeated scalar is rejected instead of silently taking the last value,
    # undeclared parameters are rejected rather than ignored, a blank filter
    # or limit is never coerced to a default, and the two boolean filters
    # accept only the lowercase literals ``true``/``false``.
    raw_body = await request.body()
    if raw_body:
        raise _query_validation_error(
            "body",
            "request body must be empty",
            "value_error.body",
        )

    raw = request.query_params

    unknown = set(raw) - _IMPACT_IMPORT_RECONCILIATIONS_PARAMS
    if unknown:
        # The collection accepts only the two filters plus limit/cursor; a
        # typo never silently changes the page.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    local_available = _parse_bool_literal_param(raw, "local_available")
    matches = _parse_bool_literal_param(raw, "matches")

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_IMPACT_IMPORT_RECONCILIATIONS_LIMIT,
        service.MIN_IMPACT_IMPORT_RECONCILIATIONS_LIMIT,
        service.MAX_IMPACT_IMPORT_RECONCILIATIONS_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.impact_import_reconciliations_cursor_secret,
                pagination.IMPACT_IMPORT_RECONCILIATIONS_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: both effective
        # filters (null when unfiltered) and the effective limit are bound.
        expected = {
            "local_available": local_available,
            "matches": matches,
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    def _encode_cursor(next_offset: int) -> str:
        return pagination.encode_typed_cursor(
            request.app.state.impact_import_reconciliations_cursor_secret,
            pagination.IMPACT_IMPORT_RECONCILIATIONS_CURSOR,
            {
                "local_available": local_available,
                "matches": matches,
                "limit": page_limit,
                "offset": next_offset,
            },
        )

    # Strictly read-only: the listing writes no resource, receipt, or audit
    # event. Receipts follow stable creation order. The current local impact
    # set is read once, complete and unfiltered, under exactly the
    # single-receipt reconciliation rules; it is identical for every receipt
    # on the page. The imported impacts arrays are not persisted and are
    # never read or echoed.
    records = service.list_revocation_impact_import_reconciliations(session)
    local_checkpoint = _local_revocation_impact_checkpoint(session)
    available = local_checkpoint.impact_count > 0

    reconciled: list[RevocationImpactImportReconciliationItem] = []
    for record in records:
        record_matches = (
            record.checkpoint_version == local_checkpoint.checkpoint_version
            and record.impact_count == local_checkpoint.impact_count
            and record.impacts_digest_hex == local_checkpoint.impacts_digest_hex
        )
        # The filters apply to the read-time reconciliation results and
        # combine as logical AND; an absent filter imposes no restriction.
        if local_available is not None and available != local_available:
            continue
        if matches is not None and record_matches != matches:
            continue
        reconciled.append(
            _impact_import_reconciliation_item(
                record, available, local_checkpoint, record_matches
            )
        )
    total = len(reconciled)

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) filtered set: the page is empty
        # and no further cursor can be issued.
        page: list[RevocationImpactImportReconciliationItem] = []
    else:
        page = reconciled[offset : offset + page_limit]
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = _encode_cursor(next_offset)

    result = RevocationImpactImportReconciliationPageResponse(
        items=page,
        count=total,
        next_cursor=next_cursor,
    )
    # Compact UTF-8 JSON, null literal, integral numbers only, terminated by
    # exactly one newline; members appear as items, count, next_cursor.
    body = (
        json.dumps(
            result.model_dump(mode="json"),
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return Response(content=body, media_type="application/json")


_IMPACT_RECON_PACKAGE_PARAMS = frozenset({"local_available", "matches"})


def _impact_recon_package_item(
    record,
    local_available: bool,
    local_checkpoint: RevocationImpactCheckpointResponse,
    matches: bool,
) -> ImpactReconEntryResponse:
    # Exactly the existing global-reconciliation list item: the receipt
    # public view with the local availability flag, current local
    # checkpoint, and match verdict; the imported impacts array is absent.
    return ImpactReconEntryResponse(
        id=record.id,
        checkpoint_version=record.checkpoint_version,
        impact_count=record.impact_count,
        impacts_digest_hex=record.impacts_digest_hex,
        received_at=record.created_at,
        local_available=local_available,
        local_checkpoint=local_checkpoint,
        matches=matches,
    )


def _unfiltered_impact_recon_entry_views(
    session: Session,
) -> list[ImpactReconEntryResponse]:
    """Every current reconciliation entry view, complete and unfiltered.

    Exactly the global reconciliation listing's read: receipts in stable
    creation order, the current local impact set read once complete and
    unfiltered (identical for every entry), and each entry's match verdict
    computed at this read. Strictly read-only.
    """
    records = service.list_revocation_impact_import_reconciliations(session)
    local_checkpoint = _local_revocation_impact_checkpoint(session)
    available = local_checkpoint.impact_count > 0
    entry_views: list[ImpactReconEntryResponse] = []
    for record in records:
        record_matches = (
            record.checkpoint_version == local_checkpoint.checkpoint_version
            and record.impact_count == local_checkpoint.impact_count
            and record.impacts_digest_hex == local_checkpoint.impacts_digest_hex
        )
        entry_views.append(
            _impact_recon_package_item(
                record, available, local_checkpoint, record_matches
            )
        )
    return entry_views


def _impact_recon_package_from_views(
    entry_views: list[ImpactReconEntryResponse],
) -> ImpactReconPackageResponse:
    """Assemble the wire package (checkpoint + entries) over entry views."""
    # mode="json" yields exactly the wire view served in a package
    # response's entries member (UTC datetimes as RFC 3339 strings), so
    # the checkpoint binds to exactly those entries and is reproducible by
    # an external verifier from the package body alone.
    entry_payloads = [view.model_dump(mode="json") for view in entry_views]
    return ImpactReconPackageResponse(
        checkpoint=ImpactReconCheckpointResponse(
            checkpoint_version=IMPACT_RECON_CHECKPOINT_VERSION,
            digest_algorithm=IMPACT_RECON_DIGEST_ALGORITHM,
            entry_count=len(entry_payloads),
            entries_digest_hex=canonical.impact_recon_entries_digest_hex(
                entry_payloads
            ),
        ),
        entries=entry_views,
    )


@router.get("/impact-recon-package")
async def export_impact_recon_package(
    request: Request, session: DbSession
) -> Response:
    # Read-only export of every currently-hit impact-import reconciliation
    # entry together with an offline-recomputable checkpoint. The GET
    # request body must be empty: carrying any bytes (even whitespace,
    # arbitrary bytes, or malformed JSON) is a 422 validated before any
    # query parameter, receipt, or local impact is read.
    raw_body = await request.body()
    if raw_body:
        raise _query_validation_error(
            "body",
            "request body must be empty",
            "value_error.body",
        )

    # The filter contract is exactly the global reconciliation listing's
    # minus pagination: only local_available/matches are accepted, so
    # limit/cursor and every other undeclared parameter are rejected rather
    # than ignored, a repeated scalar is rejected instead of silently
    # taking the last value, and only the lowercase literals true/false
    # are accepted.
    raw = request.query_params

    unknown = set(raw) - _IMPACT_RECON_PACKAGE_PARAMS
    if unknown:
        # A typo or a pagination parameter never silently changes the
        # snapshot.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    local_available_filter = _parse_bool_literal_param(raw, "local_available")
    matches_filter = _parse_bool_literal_param(raw, "matches")

    # Strictly read-only, and the package derives from one state read under
    # exactly the global reconciliation rules: receipts follow stable
    # creation order, the current local impact set is read once complete
    # and unfiltered (identical for every entry), and the two filters apply
    # to the read-time reconciliation and combine as logical AND. No
    # resource, receipt, or audit event is written; an empty match set
    # yields "entries": [] with entry_count 0 and the digest of the empty
    # array.
    entry_views = [
        view
        for view in _unfiltered_impact_recon_entry_views(session)
        if (
            local_available_filter is None
            or view.local_available == local_available_filter
        )
        and (matches_filter is None or view.matches == matches_filter)
    ]
    package = _impact_recon_package_from_views(entry_views)
    # Compact UTF-8 JSON, null/boolean literals, integral numbers only
    # (never a negative zero or non-finite number), terminated by exactly
    # one newline; root members appear as checkpoint, entries.
    body = (
        json.dumps(
            package.model_dump(mode="json"),
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return Response(content=body, media_type="application/json")


@router.post(
    "/impact-recon-verifications",
    response_model=ImpactReconVerificationResponse,
    response_model_exclude_none=True,
)
async def verify_impact_recon_package(
    payload: ImpactReconVerificationCreate, request: Request
) -> Response:
    # The route accepts no query parameters: any (or repeated) parameter is
    # a 422 validation_error.
    _reject_any_query_param(request)
    # Strictly stateless: no session is injected, so nothing is queried,
    # created, or modified, and no resource, audit row, or log is written.
    # Verification uses the request body alone; no receipt, impact,
    # attestation, actor, content, or bundle id is ever resolved against
    # local state, so unknown resources, repeated requests, and differing
    # local state verify identically. Malformed JSON and every
    # structural, field, type, count, or digest-format failure are
    # rejected by the request model as a 422 before this body runs.
    body = await request.json()
    # The digest commits to the entries array exactly as received: the raw
    # JSON values (array order, timestamp spellings), not any parsed or
    # re-serialized form. The array keeps its order; nested object keys
    # (including local_checkpoint's) sort by Unicode code point under the
    # checkpoint canonical rules.
    computed_digest_hex = canonical.impact_recon_entries_digest_hex(
        body["entries"]
    )
    if computed_digest_hex == payload.checkpoint.entries_digest_hex:
        result = {"valid": True}
    else:
        result = {
            "valid": False,
            "computed_digest_hex": computed_digest_hex,
        }
    # Compact UTF-8 JSON, boolean literals, terminated by exactly one
    # newline; a match carries no field besides valid.
    data = (
        json.dumps(
            result,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return Response(content=data, media_type="application/json")


_TRUST_EVALUATION_PARAMS = frozenset(
    {"target_type", "target_id", "min_signers"}
)


def _impact_recon_exchange_import_response(record) -> dict:
    """Build the compact public receipt dict for one exchange import."""
    return ImpactReconExchangeImportResponse(
        id=record.id,
        signature_version=record.signature_version,
        signer_subject=record.signer_subject,
        public_key=base64.b64encode(record.public_key).decode("ascii"),
        package_digest_hex=record.package_digest_hex,
        signature_digest_hex=record.signature_digest_hex,
        received_at=record.created_at,
    ).model_dump(mode="json")


def _render_compact_json(payload: dict, status_code: int = 200) -> Response:
    # Compact UTF-8 JSON, null/boolean literals, integral numbers only,
    # terminated by exactly one newline. The status code is set on the
    # returned Response itself (a manually built Response does not inherit
    # the injected response's status).
    data = (
        json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return Response(
        content=data, media_type="application/json", status_code=status_code
    )


@router.post("/impact-recon-exchange-imports")
async def import_impact_recon_exchange(
    payload: ImpactReconExchangeImportCreate,
    request: Request,
    session: DbSession,
) -> Response:
    # The import route accepts no query parameters: any (or repeated)
    # parameter is a 422 validation_error before the package is read.
    _reject_any_query_param(request)

    # Structure, the fixed package/version algorithm, the claimed entry
    # count, the strict entry fields and timestamps, the signature
    # metadata fields, standard Base64 key/signature lengths, and the
    # 64-lowercase-hex digest spellings have all passed request
    # validation. The digest bindings are enforced here over the raw
    # received JSON, so root/array order and datetime spellings
    # participate exactly as on the stateless verification route; every
    # mismatch is a 422 validation_error and writes nothing.
    body = await request.json()
    raw_package = body["package"]
    metadata = payload.signature_metadata

    # 1. The entries array digest under the existing checkpoint rules.
    computed_entries_digest_hex = canonical.impact_recon_entries_digest_hex(
        raw_package["entries"]
    )
    if computed_entries_digest_hex != payload.package.checkpoint.entries_digest_hex:
        raise ImpactReconExchangeImportValidationError(
            "entries_digest_mismatch",
            details={"computed_digest_hex": computed_entries_digest_hex},
        )

    # 2. The full package digest: root members (checkpoint, entries) and
    # arrays keep their received order; nested object keys sort by Unicode
    # code point under the existing package/snapshot canonical rules.
    computed_package_digest_hex = canonical.impact_recon_package_digest_hex(
        raw_package
    )
    if computed_package_digest_hex != metadata.package_digest_hex:
        raise ImpactReconExchangeImportValidationError(
            "package_digest_mismatch",
            details={"computed_digest_hex": computed_package_digest_hex},
        )

    # 3. The Ed25519 signature binds, in order, the fixed version, the
    #    signing subject, the digest algorithm, and the package digest --
    #    over the exact UTF-8 compact JSON array. Only now, after every
    #    structural and digest check, is verification attempted; a failure
    #    is its own distinct 422 code and writes nothing.
    message = signing.impact_recon_exchange_message_bytes(
        metadata.signer_subject,
        metadata.package_digest_algorithm,
        metadata.package_digest_hex,
    )
    if not ed25519.verify(
        metadata.public_key, message, metadata.signature
    ):
        raise ImpactReconSignatureVerificationError(
            details={"reason": "signature_verification_failed"}
        )

    # Only the SHA-256 digest of the signature is ever passed on for
    # storage; the raw signature and the package are not persisted.
    signature_digest_hex = hashlib.sha256(metadata.signature).hexdigest()

    # Verification is decided entirely by the request body: no resource id
    # named by the package is ever resolved against local state, so
    # whether the referenced receipts or impacts exist locally cannot
    # change the receipt, and no local resource is queried or created.
    record, created = service.create_impact_recon_exchange_import(
        session, payload, signature_digest_hex
    )
    # First registration of this package identity -> 201; an exact retried
    # submission (same subject, key, and signature) -> 200 with the
    # original receipt and no new audit event. A different subject, key,
    # or signature for the same package is a 422 from the service with the
    # original record untouched.
    status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return _render_compact_json(
        _impact_recon_exchange_import_response(record),
        status_code=status_code,
    )


@router.get("/impact-recon-exchange-imports/{import_id}")
def get_impact_recon_exchange_import(
    import_id: str, request: Request, session: DbSession
) -> Response:
    # The receipt read takes no query parameters: any (or repeated)
    # parameter is a 422 before the receipt lookup. PUT/PATCH/DELETE and
    # every other non-GET method on this path is the framework's 405
    # method_not_allowed.
    _reject_any_query_param(request)
    # Strictly read-only: a receipt read writes no resource and no audit
    # event. An unknown id is an explicit, specific 404.
    record = service.get_impact_recon_exchange_import(session, import_id)
    return _render_compact_json(
        _impact_recon_exchange_import_response(record)
    )


_IMPACT_RECON_EXCHANGE_IMPORTS_PARAMS = frozenset(
    {
        "signature_version",
        "signer_subject",
        "public_key",
        "package_digest_hex",
        "from",
        "to",
        "limit",
        "cursor",
    }
)


@router.get("/impact-recon-exchange-imports")
async def list_impact_recon_exchange_imports(
    request: Request, session: DbSession
) -> Response:
    # The GET request body must be empty: carrying any bytes (even
    # whitespace, arbitrary bytes, or malformed JSON) is a 422 validated
    # before any query parameter or receipt is read.
    raw_body = await request.body()
    if raw_body:
        raise _query_validation_error(
            "body",
            "request body must be empty",
            "value_error.body",
        )

    # Raw multi-values are inspected deliberately: a repeated scalar is
    # rejected instead of silently taking the last value, undeclared
    # parameters are rejected rather than ignored, and a blank filter/limit
    # is never coerced to a default.
    raw = request.query_params

    unknown = set(raw) - _IMPACT_RECON_EXCHANGE_IMPORTS_PARAMS
    if unknown:
        # A typo never silently changes the search.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    signature_version = _parse_nonempty_filter(raw, "signature_version")
    signer_subject = _parse_nonempty_filter(raw, "signer_subject")
    public_key = _parse_nonempty_filter(raw, "public_key")
    package_digest_hex = _parse_nonempty_filter(raw, "package_digest_hex")

    from_dt = _parse_rfc3339_param(raw, "from")
    to_dt = _parse_rfc3339_param(raw, "to")
    if from_dt is not None and to_dt is not None and from_dt > to_dt:
        raise _query_validation_error(
            "from",
            "from must not be later than to",
            "value_error.range",
        )

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_IMPACT_RECON_EXCHANGE_IMPORTS_LIMIT,
        service.MIN_IMPACT_RECON_EXCHANGE_IMPORTS_LIMIT,
        service.MAX_IMPACT_RECON_EXCHANGE_IMPORTS_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.impact_recon_exchange_imports_cursor_secret,
                pagination.IMPACT_RECON_EXCHANGE_IMPORTS_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: every effective
        # filter (null when unfiltered; the time claims canonicalize "Z"
        # and "+00:00" to the same instant) and the effective limit must
        # match exactly. A cursor minted by another endpoint family already
        # fails decoding above.
        expected = {
            "signature_version": signature_version,
            "signer_subject": signer_subject,
            "public_key": public_key,
            "package_digest_hex": package_digest_hex,
            "from": _audit_time_claim(from_dt),
            "to": _audit_time_claim(to_dt),
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Strictly read-only: the search writes no receipt, resource, or audit
    # event. The total is the filtered count covering every page. Filter
    # values are never resolved for existence, so an unknown value is an
    # empty collection rather than a 404. Each item is exactly the
    # single-receipt public view; the package, the raw signature, and every
    # private key are never persisted and so can never be echoed.
    items = service.list_impact_recon_exchange_imports(
        session,
        signature_version,
        signer_subject,
        public_key,
        package_digest_hex,
        from_dt,
        to_dt,
    )
    total = len(items)

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        page = items[offset : offset + page_limit]
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.impact_recon_exchange_imports_cursor_secret,
                pagination.IMPACT_RECON_EXCHANGE_IMPORTS_CURSOR,
                {
                    "signature_version": signature_version,
                    "signer_subject": signer_subject,
                    "public_key": public_key,
                    "package_digest_hex": package_digest_hex,
                    "from": _audit_time_claim(from_dt),
                    "to": _audit_time_claim(to_dt),
                    "limit": page_limit,
                    "offset": next_offset,
                },
            )

    result = ImpactReconExchangeImportPageResponse(
        items=[_impact_recon_exchange_import_response(item) for item in page],
        count=total,
        next_cursor=next_cursor,
    )
    # Compact UTF-8 JSON, null literal, integral numbers only, terminated
    # by exactly one newline; members appear as items, count, next_cursor.
    return _render_compact_json(result.model_dump(mode="json"))


@router.get("/impact-recon-exchange-imports/{import_id}/recon")
async def reconcile_impact_recon_exchange_import(
    import_id: str, request: Request, session: DbSession
) -> Response:
    # The GET request body must be empty: any bytes are a 422 validated
    # before any query parameter, receipt, or local state is read.
    raw_body = await request.body()
    if raw_body:
        raise _query_validation_error(
            "body",
            "request body must be empty",
            "value_error.body",
        )
    # Same boundary as the receipt read route: any (or repeated) query
    # parameter is a 422 before the receipt lookup.
    _reject_any_query_param(request)
    # Strictly read-only: the reconciliation writes no resource, receipt,
    # or audit event. An unknown receipt id is the existing
    # impact_recon_exchange_import_not_found 404 and changes nothing.
    record = service.get_impact_recon_exchange_import(session, import_id)

    # The current local recon package is always read complete and
    # unfiltered under exactly the package export's rules, so the digest
    # is reproducible by an external verifier from a fresh
    # GET /v1/impact-recon-package body alone -- including the empty
    # state, which still yields a deterministic digest. The verdict is
    # computed at this read and compares character for character; no
    # digest is ever resolved in reverse to a resource, and the package,
    # the raw signature, and any private key are never read back or
    # echoed.
    entry_views = _unfiltered_impact_recon_entry_views(session)
    package = _impact_recon_package_from_views(entry_views)
    local_package_digest_hex = canonical.impact_recon_package_digest_hex(
        package.model_dump(mode="json")
    )
    result = ImpactReconExchangeImportReconResponse(
        import_id=record.id,
        local_available=len(entry_views) > 0,
        local_package_digest_hex=local_package_digest_hex,
        matches=local_package_digest_hex == record.package_digest_hex,
    )
    return _render_compact_json(result.model_dump(mode="json"))


_IMPACT_RECON_AUDIT_PACKAGES_PARAMS = frozenset(
    {
        "signer_subject",
        "public_key",
        "package_digest_hex",
        "from",
        "to",
        "matches",
    }
)


def _local_impact_recon_package_state(session: Session) -> tuple[bool, str]:
    """The current unfiltered local recon package's availability and digest.

    Exactly the single-receipt reconciliation's local read: the current
    local impact-recon package is read once, complete and unfiltered,
    under the package export's rules, so the digest is deterministic even
    for the empty state. Strictly read-only.
    """
    entry_views = _unfiltered_impact_recon_entry_views(session)
    package = _impact_recon_package_from_views(entry_views)
    local_package_digest_hex = canonical.impact_recon_package_digest_hex(
        package.model_dump(mode="json")
    )
    return len(entry_views) > 0, local_package_digest_hex


def _impact_recon_audit_entry(
    record,
    local_available: bool,
    local_package_digest_hex: str,
) -> ImpactReconAuditEntryResponse:
    # The stable receipt fields with the read-time reconciliation triple
    # appended; the imported package, the raw signature, and every
    # private key are never carried.
    return ImpactReconAuditEntryResponse(
        id=record.id,
        signer_subject=record.signer_subject,
        public_key=base64.b64encode(record.public_key).decode("ascii"),
        package_digest_hex=record.package_digest_hex,
        signature_digest_hex=record.signature_digest_hex,
        received_at=record.created_at,
        local_available=local_available,
        local_package_digest_hex=local_package_digest_hex,
        matches=record.package_digest_hex == local_package_digest_hex,
    )


@router.get("/impact-recon-audit-packages")
async def export_impact_recon_audit_package(
    request: Request, session: DbSession
) -> Response:
    # Read-only audit checkpoint export over the signed exchange-import
    # receipts. The GET request body must be empty: carrying any bytes
    # (even whitespace, arbitrary bytes, or malformed JSON) is a 422
    # validated before any query parameter, receipt, or local state is
    # read.
    raw_body = await request.body()
    if raw_body:
        raise _query_validation_error(
            "body",
            "request body must be empty",
            "value_error.body",
        )

    # Raw multi-values are inspected deliberately: a repeated scalar is
    # rejected instead of silently taking the last value, undeclared
    # parameters (including the limit/cursor pagination pair) are
    # rejected rather than ignored, a blank filter is never coerced, and
    # only the lowercase literals true/false are accepted for matches.
    raw = request.query_params

    unknown = set(raw) - _IMPACT_RECON_AUDIT_PACKAGES_PARAMS
    if unknown:
        # A typo or a pagination parameter never silently changes the
        # snapshot.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    signer_subject = _parse_nonempty_filter(raw, "signer_subject")
    public_key = _parse_nonempty_filter(raw, "public_key")
    package_digest_hex = _parse_nonempty_filter(raw, "package_digest_hex")

    from_dt = _parse_rfc3339_param(raw, "from")
    to_dt = _parse_rfc3339_param(raw, "to")
    if from_dt is not None and to_dt is not None and from_dt > to_dt:
        raise _query_validation_error(
            "from",
            "from must not be later than to",
            "value_error.range",
        )

    matches_filter = _parse_bool_literal_param(raw, "matches")

    # Strictly read-only: every validation failure above is raised before
    # any receipt or local state is read, and the export itself writes no
    # receipt, resource, audit event, or log. The receipt-field filters
    # (including the exact-spelling public key) are pushed down to the
    # database query; the current local recon package is then read once,
    # complete and unfiltered, so its availability flag and digest are
    # identical for every entry; the matches filter applies to the
    # read-time verdict and combines with the rest as logical AND. An
    # empty match set yields "entries": [] with entry_count 0 and the
    # deterministic digest of the empty array.
    records = service.list_impact_recon_audit_entries(
        session,
        signer_subject,
        public_key,
        package_digest_hex,
        from_dt,
        to_dt,
    )
    local_available, local_package_digest_hex = (
        _local_impact_recon_package_state(session)
    )
    entry_views = [
        _impact_recon_audit_entry(
            record, local_available, local_package_digest_hex
        )
        for record in records
    ]
    if matches_filter is not None:
        entry_views = [
            view for view in entry_views if view.matches == matches_filter
        ]
    # mode="json" yields exactly the wire view served in this response's
    # entries member (UTC datetimes as RFC 3339 strings), so the
    # checkpoint binds to exactly those entries and is reproducible by an
    # external verifier from the package body alone.
    entry_payloads = [view.model_dump(mode="json") for view in entry_views]
    package = ImpactReconAuditPackageResponse(
        checkpoint=ImpactReconAuditCheckpointResponse(
            checkpoint_version=IMPACT_RECON_AUDIT_CHECKPOINT_VERSION,
            digest_algorithm=IMPACT_RECON_AUDIT_DIGEST_ALGORITHM,
            entry_count=len(entry_payloads),
            entries_digest_hex=canonical.impact_recon_audit_entries_digest_hex(
                entry_payloads
            ),
        ),
        entries=entry_views,
    )
    # Compact UTF-8 JSON, null/boolean literals, integral numbers only,
    # terminated by exactly one newline; root members appear as
    # checkpoint, entries.
    return _render_compact_json(package.model_dump(mode="json"))


@router.post(
    "/impact-recon-audit-verifications",
    response_model=ImpactReconAuditVerificationResponse,
    response_model_exclude_none=True,
)
async def verify_impact_recon_audit_package(
    payload: ImpactReconAuditVerificationCreate, request: Request
) -> Response:
    # The route accepts no query parameters: any (or repeated) parameter
    # is a 422 validation_error.
    _reject_any_query_param(request)
    # Strictly stateless: no session is injected, so nothing is queried,
    # created, or modified, and no receipt, resource, audit row, or log
    # is written. Verification uses the request body alone; no receipt or
    # resource id is ever resolved against local state, so unknown
    # resources, repeated requests, and differing local state verify
    # identically. Malformed JSON and every structural, field, type,
    # count, timestamp, or digest-format failure are rejected by the
    # request model as a 422 before this body runs.
    body = await request.json()
    # The digest commits to the entries array exactly as received: the
    # raw JSON values (array order, timestamp spellings), not any parsed
    # or re-serialized form. The array keeps its order; nested object
    # keys sort by Unicode code point under the checkpoint canonical
    # rules.
    computed_digest_hex = canonical.impact_recon_audit_entries_digest_hex(
        body["entries"]
    )
    if computed_digest_hex == payload.checkpoint.entries_digest_hex:
        result = {"valid": True}
    else:
        result = {
            "valid": False,
            "computed_digest_hex": computed_digest_hex,
        }
    # Compact UTF-8 JSON, boolean literals, terminated by exactly one
    # newline; a match carries no field besides valid.
    return _render_compact_json(result)


_TRUST_EVALUATION_PARAMS = frozenset(
    {"target_type", "target_id", "min_signers"}
)


async def _authenticate_protected(
    request: Request, session: Session, body: bytes, *, read: bool
) -> str:
    """Authenticate an X-PA/X-PT/X-PS protected request.

    On the write route every authentication failure is a ``422``. On the
    read route only malformed credentials are ``422``; a missing or
    unverifiable identity collapses into the same opaque ``404`` as a
    missing target or an unauthorized caller.
    """
    try:
        return access_signing.authenticate(session, request, body)
    except AccessAuthError as exc:
        if read and not exc.malformed:
            raise ProtectedResourceNotFoundError() from exc
        raise ProtectedAccessValidationError(exc.reason) from exc


@router.post(
    "/attestation-access-grants",
    response_model=AttestationAccessGrantResponse,
)
async def create_attestation_access_grant(
    request: Request,
    payload: AttestationAccessGrantCreate,
    session: DbSession,
    response: Response,
) -> AttestationAccessGrantResponse:
    # The signature covers the exact bytes on the wire; FastAPI's parsed
    # model is built from the same cached body, so body_sha256 matches what
    # the client signed.
    raw_body = await request.body()
    caller = await _authenticate_protected(
        request, session, raw_body, read=False
    )
    grant, created = service.create_attestation_access_grant(
        session, payload, caller
    )
    # First creation -> 201; a retried submission for the same
    # (attestation_id, grantee_actor_id) pair -> 200 with the original
    # record and no new audit event.
    response.status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return AttestationAccessGrantResponse.model_validate(grant)


@router.post(
    "/attestation-access-grant-revocations",
    response_model=AttestationAccessGrantRevocationResponse,
)
async def create_attestation_access_grant_revocation(
    request: Request,
    payload: AttestationAccessGrantRevocationCreate,
    session: DbSession,
    response: Response,
) -> AttestationAccessGrantRevocationResponse:
    # The signature covers the exact bytes on the wire; FastAPI's parsed
    # model is built from the same cached body, so body_sha256 matches what
    # the client signed.
    raw_body = await request.body()
    caller = await _authenticate_protected(
        request, session, raw_body, read=False
    )
    revocation, created = service.create_attestation_access_grant_revocation(
        session, payload, caller
    )
    # First creation -> 201; a retried submission for the same grant,
    # revoking signer, and reason -> 200 with the original record and no new
    # audit event.
    response.status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return AttestationAccessGrantRevocationResponse.model_validate(revocation)


_ATTESTATION_ACCESS_GRANT_REVOCATIONS_PARAMS = frozenset(
    {
        "grant_id",
        "revoker_actor_id",
        "reason",
        "from",
        "to",
        "limit",
        "cursor",
    }
)


@router.get("/attestation-access-grant-revocations")
async def list_attestation_access_grant_revocations_global(
    request: Request, session: DbSession
) -> Response:
    # Global read-only search across grants over the immutable grant-
    # revocation records. The GET request body must be empty: carrying any
    # bytes (even whitespace, arbitrary bytes, or malformed JSON) is a 422
    # validated before any query parameter, cursor, or revocation record is
    # read. No authentication headers are required. Raw multi-values are
    # inspected deliberately: a repeated scalar is rejected instead of
    # silently taking the last value, undeclared parameters are rejected
    # rather than ignored, and a blank filter/limit is never coerced to a
    # default.
    raw_body = await request.body()
    if raw_body:
        raise _query_validation_error(
            "body",
            "request body must be empty",
            "value_error.body",
        )

    raw = request.query_params

    unknown = set(raw) - _ATTESTATION_ACCESS_GRANT_REVOCATIONS_PARAMS
    if unknown:
        # A typo (e.g. ``grant``) never silently changes the search.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    grant_id = _parse_nonempty_filter(raw, "grant_id")
    revoker_actor_id = _parse_nonempty_filter(raw, "revoker_actor_id")
    reason = _parse_nonempty_filter(raw, "reason")

    from_dt = _parse_rfc3339_param(raw, "from")
    to_dt = _parse_rfc3339_param(raw, "to")
    if from_dt is not None and to_dt is not None and from_dt > to_dt:
        raise _query_validation_error(
            "from",
            "from must not be later than to",
            "value_error.range",
        )

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_ATTESTATION_ACCESS_GRANT_REVOCATIONS_LIMIT,
        service.MIN_ATTESTATION_ACCESS_GRANT_REVOCATIONS_LIMIT,
        service.MAX_ATTESTATION_ACCESS_GRANT_REVOCATIONS_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.attestation_access_grant_revocations_cursor_secret,
                pagination.ATTESTATION_ACCESS_GRANT_REVOCATIONS_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: every effective
        # filter (null when unfiltered; the time claim canonicalizes "Z" and
        # "+00:00" to the same instant) and the effective limit must match
        # exactly. A cursor minted by another endpoint family already fails
        # decoding above.
        expected = {
            "grant_id": grant_id,
            "revoker_actor_id": revoker_actor_id,
            "reason": reason,
            "from": _audit_time_claim(from_dt),
            "to": _audit_time_claim(to_dt),
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Strictly read-only: the search writes no revocation, grant, resource,
    # or audit event. The total is a SQL COUNT over the filtered set
    # (covering every page) and the page is a SQL LIMIT/OFFSET window
    # ordered in SQL by created_at then the monotonic insertion sequence, so
    # ordering and paging survive a restart. Filter values are never resolved
    # for existence, so an unknown grant id, revoker id, or reason is an
    # empty collection rather than a 404. Each item is exactly the single-
    # revocation public view: no raw signature, private key, payload,
    # content, or evidence byte is ever echoed.
    page, total = service.list_attestation_access_grant_revocations_page(
        session,
        grant_id,
        revoker_actor_id,
        reason,
        from_dt,
        to_dt,
        page_limit,
        offset,
    )

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.attestation_access_grant_revocations_cursor_secret,
                pagination.ATTESTATION_ACCESS_GRANT_REVOCATIONS_CURSOR,
                {
                    "grant_id": grant_id,
                    "revoker_actor_id": revoker_actor_id,
                    "reason": reason,
                    "from": _audit_time_claim(from_dt),
                    "to": _audit_time_claim(to_dt),
                    "limit": page_limit,
                    "offset": next_offset,
                },
            )

    result = AttestationAccessGrantRevocationPageResponse(
        items=[
            AttestationAccessGrantRevocationResponse.model_validate(item)
            for item in page
        ],
        count=total,
        next_cursor=next_cursor,
    )
    # Compact UTF-8 JSON, null literal, integral numbers only, terminated by
    # exactly one newline; members appear as items, count, next_cursor.
    body = (
        json.dumps(
            result.model_dump(mode="json"),
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return Response(content=body, media_type="application/json")


@router.get(
    "/attestation-access-grant-revocations/{revocation_id}",
    response_model=AttestationAccessGrantRevocationResponse,
)
def get_attestation_access_grant_revocation(
    revocation_id: str, request: Request, session: DbSession
) -> AttestationAccessGrantRevocationResponse:
    # The read takes no query parameters: any parameter at all (known or
    # unknown, blank or repeated) is a 422 before the revocation lookup, so a
    # malformed request never renders as a 404.
    _reject_any_query_param(request)
    # Strictly read-only: the read returns one existing revocation's public
    # view and writes no revocation, grant, resource, or audit event. The id
    # is the only lookup key (never a reverse lookup by other fields); an
    # unknown id is an explicit, specific 404.
    revocation = service.get_attestation_access_grant_revocation(
        session, revocation_id
    )
    return AttestationAccessGrantRevocationResponse.model_validate(revocation)


@router.get(
    "/attestation-access-grants/{grant_id}/revocations",
    response_model=AttestationAccessGrantRevocationListResponse,
)
def list_attestation_access_grant_revocations(
    grant_id: str, request: Request, session: DbSession
) -> AttestationAccessGrantRevocationListResponse:
    # The read takes no query parameters: any parameter at all (known or
    # unknown, blank or repeated) is a 422 before the grant lookup, so a
    # malformed request never renders as an unknown-grant error or an empty
    # collection.
    _reject_any_query_param(request)
    # Strictly read-only: only that existing grant's revocation records are
    # returned in stable creation order; the grant id is the sole lookup key.
    # An unknown grant is the same 422 validation_error (grant_not_found) as
    # on the write route -- never an empty collection and never a reverse
    # lookup. An existing grant without revocations returns an empty array
    # and zero count. Nothing is created or modified and no audit event is
    # written.
    items = service.list_revocations_for_access_grant(session, grant_id)
    return AttestationAccessGrantRevocationListResponse(
        items=[
            AttestationAccessGrantRevocationResponse.model_validate(item)
            for item in items
        ],
        count=len(items),
    )


@router.get(
    "/protected/attestations/{attestation_id}",
    response_model=AttestationResponse,
)
async def get_protected_attestation(
    attestation_id: str, request: Request, session: DbSession
) -> AttestationResponse:
    # A GET carries no body; the signed body_sha256 is therefore the digest
    # of zero bytes.
    raw_body = await request.body()
    actor = await _authenticate_protected(
        request, session, raw_body, read=True
    )
    # A missing target and an unauthorized caller are indistinguishable to
    # the caller: both are the same opaque 404.
    attestation = service.get_accessible_attestation(
        session, attestation_id, actor
    )
    if attestation is None:
        raise ProtectedResourceNotFoundError()
    return _attestation_response(attestation)


_ATTESTATION_ACCESS_GRANTS_PARAMS = frozenset({"limit", "cursor"})


@router.get(
    "/attestations/{attestation_id}/access-grants",
    response_model=AttestationAccessGrantPageResponse,
)
async def list_attestation_access_grants(
    attestation_id: str,
    request: Request,
    session: DbSession,
    limit: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
) -> AttestationAccessGrantPageResponse:
    # The signer's protected listing of one proof's access grants. Raw
    # multi-values are inspected deliberately: a repeated scalar is rejected
    # instead of silently taking the last value, any parameter other than
    # limit/cursor is rejected rather than ignored, and a blank or
    # non-integer limit is never coerced to the default. Every such
    # validation failure is a 422, so a malformed request never renders as
    # the opaque 404 below.
    raw = request.query_params

    unknown = set(raw) - _ATTESTATION_ACCESS_GRANTS_PARAMS
    if unknown:
        # The collection accepts only limit/cursor; a typo never silently
        # changes the page.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_ATTESTATION_ACCESS_GRANTS_LIMIT,
        service.MIN_ATTESTATION_ACCESS_GRANTS_LIMIT,
        service.MAX_ATTESTATION_ACCESS_GRANTS_LIMIT,
    )

    # The cursor token itself is verified and structurally decoded before
    # authentication: a forged, malformed, or other-family token is a client
    # validation error regardless of who presents it. Binding the claims to
    # this proof and caller happens once the caller is authenticated.
    cursor = _parse_once(raw, "cursor")
    claims = None
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.attestation_access_grants_cursor_secret,
                pagination.ATTESTATION_ACCESS_GRANTS_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc

    # A GET carries no body; the signed body_sha256 is therefore the digest
    # of zero bytes, exactly as on the other protected read route. Missing
    # or unauthenticated credentials collapse into the same opaque 404 as a
    # missing target or a non-signer; malformed credentials stay 422.
    raw_body = await request.body()
    actor = await _authenticate_protected(
        request, session, raw_body, read=True
    )

    if claims is not None:
        # The cursor only resumes the query that issued it: the origin proof,
        # the authenticated caller, and the effective limit must all match
        # exactly. A cursor minted for another proof, another signer, or a
        # different limit is a client validation error, not a new query.
        expected = {
            "attestation_id": attestation_id,
            "actor_id": actor,
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
    offset = claims["offset"] if claims is not None else 0

    # A missing proof and a caller who is not its signer are indistinguishable
    # to the caller: both render as the same opaque 404.
    items = service.list_access_grants_for_attestation(
        session, attestation_id, actor
    )
    if items is None:
        raise ProtectedResourceNotFoundError()
    total = len(items)

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty,
        # count is unchanged, and no further cursor can be issued.
        page = []
    else:
        page = items[offset : offset + page_limit]
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.attestation_access_grants_cursor_secret,
                pagination.ATTESTATION_ACCESS_GRANTS_CURSOR,
                {
                    "attestation_id": attestation_id,
                    "actor_id": actor,
                    "limit": page_limit,
                    "offset": next_offset,
                },
            )

    # Each item is the existing grant public view only: id, the proof and
    # grantee associations, and the UTC timestamp. Revocations, private
    # keys, raw signatures, authentication headers, payloads, and bytes are
    # never part of that view and so can never be echoed.
    return AttestationAccessGrantPageResponse(
        items=[
            AttestationAccessGrantResponse.model_validate(item) for item in page
        ],
        count=total,
        next_cursor=next_cursor,
    )


def _rotation_response(rotation) -> AuthenticationKeyRotationResponse:
    # Only the 32 public-key bytes leave the service, Base64 on the wire;
    # there is never a private key or raw signature to render.
    return AuthenticationKeyRotationResponse(
        id=rotation.id,
        actor_id=rotation.actor_id,
        new_public_key=base64.b64encode(rotation.public_key).decode("ascii"),
        active=rotation.active,
        created_at=rotation.created_at,
        retired_at=rotation.retired_at,
    )


@router.post(
    "/authentication-key-rotations",
    response_model=AuthenticationKeyRotationResponse,
)
async def create_authentication_key_rotation(
    request: Request,
    payload: AuthenticationKeyRotationCreate,
    session: DbSession,
    response: Response,
) -> AuthenticationKeyRotationResponse:
    # The signature covers the exact bytes on the wire; FastAPI's parsed
    # model is built from the same cached body, so body_sha256 matches what
    # the client signed. The caller authenticated by the signature is the
    # rotation's subject.
    raw_body = await request.body()
    caller = await _authenticate_protected(
        request, session, raw_body, read=False
    )
    rotation, created = service.create_authentication_key_rotation(
        session, payload, caller
    )
    # First creation -> 201; a retried submission for the same subject and
    # public key -> 200 with the original record (retired or not) and no
    # audit event.
    response.status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return _rotation_response(rotation)


@router.post(
    "/authentication-key-rotations/{rotation_id}/retire",
    response_model=AuthenticationKeyRotationResponse,
)
async def retire_authentication_key_rotation(
    rotation_id: str, request: Request, session: DbSession
) -> AuthenticationKeyRotationResponse:
    raw_body = await request.body()
    # The request body is empty, so the signed body_sha256 is the digest of
    # zero bytes, exactly as for a bodyless request. A non-empty body is
    # malformed input rejected before any state change.
    if raw_body:
        raise ProtectedAccessValidationError("body_must_be_empty")
    caller = await _authenticate_protected(
        request, session, raw_body, read=False
    )
    rotation = service.retire_authentication_key_rotation(
        session, rotation_id, caller
    )
    return _rotation_response(rotation)


_ACTOR_ROTATIONS_PARAMS = frozenset({"limit", "cursor"})


@router.get(
    "/actors/{actor_id}/authentication-key-rotations",
    response_model=AuthenticationKeyRotationPageResponse,
)
def list_actor_authentication_key_rotations(
    actor_id: str,
    request: Request,
    session: DbSession,
    limit: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
) -> AuthenticationKeyRotationPageResponse:
    # Read-only reviewer retrieval of one existing subject's key-rotation
    # history. Raw multi-values are inspected deliberately: a repeated
    # scalar is rejected instead of silently taking the last value, any
    # parameter other than limit/cursor is rejected rather than ignored, and
    # a blank or non-integer limit is never coerced to the default. Every
    # such validation failure is a 422 before the subject lookup, so a
    # malformed request never renders as an unknown_actor 404.
    raw = request.query_params

    unknown = set(raw) - _ACTOR_ROTATIONS_PARAMS
    if unknown:
        # The collection accepts only limit/cursor; a typo never silently
        # changes the page.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_AUTHENTICATION_KEY_ROTATIONS_LIMIT,
        service.MIN_AUTHENTICATION_KEY_ROTATIONS_LIMIT,
        service.MAX_AUTHENTICATION_KEY_ROTATIONS_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.authentication_key_rotations_cursor_secret,
                pagination.AUTHENTICATION_KEY_ROTATIONS_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: the origin
        # subject and the effective limit must match exactly. A cursor from
        # another subject, another endpoint's family, or a different limit
        # is a client validation error, not a new query.
        if claims["actor_id"] != actor_id or claims["limit"] != page_limit:
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Parameters and cursor are validated first; a structurally valid
    # request for an unknown subject is a missing resource (the existing
    # unknown_actor 404), not an empty collection.
    items = service.list_authentication_key_rotations_for_actor(
        session, actor_id
    )
    total = len(items)

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        page = items[offset : offset + page_limit]
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.authentication_key_rotations_cursor_secret,
                pagination.AUTHENTICATION_KEY_ROTATIONS_CURSOR,
                {
                    "actor_id": actor_id,
                    "limit": page_limit,
                    "offset": next_offset,
                },
            )

    # Each item is the existing rotation public view: the 32 public-key
    # bytes, lifecycle flags, and UTC timestamps. Private keys, raw
    # signatures, authentication headers, and the internal ordering
    # surrogate are never part of that view and so can never be echoed.
    return AuthenticationKeyRotationPageResponse(
        items=[_rotation_response(item) for item in page],
        count=total,
        next_cursor=next_cursor,
    )


@router.get(
    "/trust-evaluations",
    response_model=TrustEvaluationResponse,
)
def evaluate_trust(
    request: Request,
    session: DbSession,
    target_type: str | None = Query(default=None),
    target_id: str | None = Query(default=None),
    min_signers: str | None = Query(default=None),
) -> TrustEvaluationResponse:
    # Raw multi-values are inspected deliberately: a repeated scalar is
    # rejected instead of silently taking the last value, and blank or
    # non-numeric values are never coerced to defaults.
    raw = request.query_params

    unknown = set(raw) - _TRUST_EVALUATION_PARAMS
    if unknown:
        # Undeclared parameters are rejected rather than ignored, so a typo
        # (e.g. ``min_signer``) never silently changes the evaluation.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    target_type = _parse_once(raw, "target_type")
    if target_type is None:
        raise _query_validation_error(
            "target_type", "Field required", "value_error.missing"
        )
    if target_type not in ATTESTATION_TARGET_TYPES:
        # Covers missing values, whitespace/blank strings, casing variants,
        # and anything other than the two literal target types.
        raise _query_validation_error(
            "target_type",
            "target_type must be 'claim' or 'evidence_bundle'",
            "value_error",
        )

    target_id = _parse_once(raw, "target_id")
    if target_id is None:
        raise _query_validation_error(
            "target_id", "Field required", "value_error.missing"
        )
    if not target_id.strip():
        # A blank identifier is invalid rather than a lookup of the empty
        # string (which would merely 404).
        raise _query_validation_error(
            "target_id", "target_id must not be empty", "value_error"
        )

    threshold = _parse_int_param(
        raw,
        "min_signers",
        service.DEFAULT_TRUST_MIN_SIGNERS,
        service.MIN_TRUST_MIN_SIGNERS,
        service.MAX_TRUST_MIN_SIGNERS,
    )

    # All parameters are validated before any existence lookup: a malformed
    # request is a 422 even when the target also happens to be missing.
    result = service.evaluate_trust(
        session, target_type, target_id, threshold
    )
    return TrustEvaluationResponse(**result)


def _compact_json_response(model, status_code: int) -> Response:
    """Serialize a response model as compact UTF-8 JSON + one newline.

    Numbers render as integers (never ``-0`` or a non-finite value), exactly
    as on the other compact-wire routes.
    """
    body = (
        json.dumps(
            model.model_dump(mode="json"),
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return Response(
        content=body, status_code=status_code, media_type="application/json"
    )


@router.post("/trust-policies")
async def create_actor_trust_policy(
    request: Request,
    payload: ActorTrustPolicyCreate,
    session: DbSession,
) -> Response:
    # The signature covers the exact bytes on the wire; FastAPI's parsed
    # model is built from the same cached body, so body_sha256 matches what
    # the client signed. The caller authenticated by the signature is the
    # policy's subject; every authentication or field failure is a 422 and
    # writes nothing.
    raw_body = await request.body()
    caller = await _authenticate_protected(
        request, session, raw_body, read=False
    )
    policy, created = service.create_actor_trust_policy(
        session, payload, caller
    )
    # First registration -> 201; a retried submission of the same threshold
    # for the same subject -> 200 with the original policy and no new row,
    # audit event, or identifier. A different threshold is a 409.
    return _compact_json_response(
        ActorTrustPolicyResponse.model_validate(policy),
        status.HTTP_201_CREATED if created else status.HTTP_200_OK,
    )


_TRUST_POLICIES_PARAMS = frozenset({"actor_id", "limit", "cursor"})


@router.get("/trust-policies")
async def list_actor_trust_policies(request: Request, session: DbSession) -> Response:
    # Read-only retrieval of the existing immutable trust policies. The GET
    # request body must be empty: carrying any bytes (even whitespace or
    # malformed JSON) is a 422 validated before any parameter or policy is
    # read. Raw multi-values are inspected deliberately: a repeated scalar
    # is rejected instead of silently taking the last value, undeclared
    # parameters are rejected rather than ignored, and a blank filter/limit
    # is never coerced to a default.
    raw_body = await request.body()
    if raw_body:
        raise _query_validation_error(
            "body",
            "request body must be empty",
            "value_error.body",
        )

    raw = request.query_params

    unknown = set(raw) - _TRUST_POLICIES_PARAMS
    if unknown:
        # A typo (e.g. ``actor_ids``) never silently changes the retrieval.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    actor_id = _parse_nonempty_filter(raw, "actor_id")

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_TRUST_POLICIES_LIMIT,
        service.MIN_TRUST_POLICIES_LIMIT,
        service.MAX_TRUST_POLICIES_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.trust_policies_cursor_secret,
                pagination.TRUST_POLICIES_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: the effective
        # subject filter (null when unfiltered) and the effective limit must
        # match exactly. A cursor minted by another endpoint family already
        # fails decoding above.
        expected = {
            "actor_id": actor_id,
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Strictly read-only: the retrieval writes no policy, resource, or audit
    # event. The subject filter is never resolved for existence, so an
    # unknown actor is an empty collection rather than a 404.
    items = service.list_actor_trust_policies(session, actor_id)
    total = len(items)

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        page = items[offset : offset + page_limit]
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.trust_policies_cursor_secret,
                pagination.TRUST_POLICIES_CURSOR,
                {
                    "actor_id": actor_id,
                    "limit": page_limit,
                    "offset": next_offset,
                },
            )

    # Each item is the existing policy public view only: the stable id,
    # subject, threshold, enabled flag, and UTC timestamp. No private key,
    # raw signature, claim payload, content, or evidence byte is ever part
    # of that view.
    result = ActorTrustPolicyPageResponse(
        items=[ActorTrustPolicyResponse.model_validate(item) for item in page],
        count=total,
        next_cursor=next_cursor,
    )
    return _compact_json_response(result, status.HTTP_200_OK)


_TRUST_DECISION_PARAMS = frozenset({"target_type", "target_id"})


@router.get("/trust-decisions")
async def get_trust_decision(
    request: Request,
    session: DbSession,
    target_type: str | None = Query(default=None),
    target_id: str | None = Query(default=None),
) -> Response:
    # The decision takes no body: carrying any bytes (even whitespace or
    # malformed JSON) is a 422 validated before any parameter is read, so an
    # invalid request never produces a partial decision.
    raw_body = await request.body()
    if raw_body:
        raise ProtectedAccessValidationError("body_must_be_empty")

    # Raw multi-values are inspected deliberately: a repeated scalar is
    # rejected instead of silently taking the last value, undeclared
    # parameters are rejected rather than ignored, and blank values are
    # never coerced. Every such failure is a 422, so a malformed request
    # never renders as the opaque 404 below.
    raw = request.query_params

    unknown = set(raw) - _TRUST_DECISION_PARAMS
    if unknown:
        # A typo (e.g. ``target_types``) never silently changes the decision.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    target_type = _parse_once(raw, "target_type")
    if target_type is None:
        raise _query_validation_error(
            "target_type", "Field required", "value_error.missing"
        )
    if target_type not in ATTESTATION_TARGET_TYPES:
        # Covers missing values, whitespace/blank strings, casing variants,
        # and anything other than the two literal target types.
        raise _query_validation_error(
            "target_type",
            "target_type must be 'claim' or 'evidence_bundle'",
            "value_error",
        )

    target_id = _parse_once(raw, "target_id")
    if target_id is None:
        raise _query_validation_error(
            "target_id", "Field required", "value_error.missing"
        )
    if not target_id.strip():
        # A blank identifier is invalid rather than a lookup of the empty
        # string (which would merely 404).
        raise _query_validation_error(
            "target_id", "target_id must not be empty", "value_error"
        )

    # A GET carries no body; the signed body_sha256 is therefore the digest
    # of zero bytes, exactly as on the other protected read routes. Missing
    # or unauthenticatable credentials collapse into the same opaque 404 as
    # on those routes; malformed credentials stay 422.
    actor = await _authenticate_protected(
        request, session, raw_body, read=True
    )

    # Strictly read-only: the decision uses only the caller's current
    # policy, writes no resource and no audit event. Without a policy the
    # target is never looked up; with one, an unknown target is the
    # type-matched 404 and never a partial decision.
    result = service.decide_trust(session, actor, target_type, target_id)
    return _compact_json_response(
        TrustDecisionResponse(**result), status.HTTP_200_OK
    )


_AUDIT_EVENTS_PARAMS = frozenset(
    {"event_type", "resource_id", "from", "to", "limit", "cursor"}
)


def _parse_rfc3339_param(raw, field: str):
    """Parse an optional strict RFC 3339 UTC timestamp query parameter."""
    value = _parse_once(raw, field)
    if value is None:
        return None
    if not value.strip():
        raise _query_validation_error(
            field, f"{field} must not be empty", "value_error"
        )
    parsed = parse_rfc3339_utc(value)
    if parsed is None:
        raise _query_validation_error(
            field,
            f"{field} must be an RFC 3339 UTC timestamp",
            "value_error.datetime",
        )
    return parsed


def _audit_time_claim(dt) -> str | None:
    # The cursor binds the effective instant, not its spelling: equivalent
    # notations ("Z" vs "+00:00") canonicalize to the same claim.
    return dt.isoformat() if dt is not None else None


@router.get("/audit-events", response_model=AuditEventPageResponse)
def list_audit_events(request: Request, session: DbSession) -> AuditEventPageResponse:
    # Raw multi-values are inspected deliberately: a repeated scalar is
    # rejected instead of silently taking the last value, and blank or
    # malformed values are never coerced to defaults.
    raw = request.query_params

    unknown = set(raw) - _AUDIT_EVENTS_PARAMS
    if unknown:
        # Undeclared parameters are rejected rather than ignored, so a typo
        # never silently changes the search.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    event_type = _parse_nonempty_filter(raw, "event_type")
    resource_id = _parse_nonempty_filter(raw, "resource_id")

    from_dt = _parse_rfc3339_param(raw, "from")
    to_dt = _parse_rfc3339_param(raw, "to")
    if from_dt is not None and to_dt is not None and from_dt > to_dt:
        raise _query_validation_error(
            "from",
            "from must not be later than to",
            "value_error.range",
        )

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_AUDIT_EVENTS_LIMIT,
        service.MIN_AUDIT_EVENTS_LIMIT,
        service.MAX_AUDIT_EVENTS_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.audit_events_cursor_secret,
                pagination.AUDIT_EVENTS_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: every effective
        # filter and the effective limit must match exactly.
        expected = {
            "event_type": event_type,
            "resource_id": resource_id,
            "from": _audit_time_claim(from_dt),
            "to": _audit_time_claim(to_dt),
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Strictly read-only: the search writes no resource and no audit event.
    items = service.list_audit_events(
        session, event_type, resource_id, from_dt, to_dt
    )
    total = len(items)

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        page = items[offset : offset + page_limit]
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.audit_events_cursor_secret,
                pagination.AUDIT_EVENTS_CURSOR,
                {
                    "event_type": event_type,
                    "resource_id": resource_id,
                    "from": _audit_time_claim(from_dt),
                    "to": _audit_time_claim(to_dt),
                    "limit": page_limit,
                    "offset": next_offset,
                },
            )

    return AuditEventPageResponse(
        items=[AuditEventItem.model_validate(item) for item in page],
        count=total,
        next_cursor=next_cursor,
    )


_AUDIT_CHECKPOINT_PARAMS = frozenset({"event_type", "resource_id", "from", "to"})


def _audit_checkpoint_filters(request: Request):
    """Parse and validate the checkpoint/package query filters.

    The package route shares the checkpoint route's parameter contract
    exactly: a repeated scalar is rejected instead of silently taking the
    last value, undeclared parameters (including limit/cursor) are rejected
    rather than ignored, and blank or malformed values are never coerced.
    Returns ``(event_type, resource_id, from_dt, to_dt)``.
    """
    raw = request.query_params

    unknown = set(raw) - _AUDIT_CHECKPOINT_PARAMS
    if unknown:
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    event_type = _parse_nonempty_filter(raw, "event_type")
    resource_id = _parse_nonempty_filter(raw, "resource_id")

    from_dt = _parse_rfc3339_param(raw, "from")
    to_dt = _parse_rfc3339_param(raw, "to")
    if from_dt is not None and to_dt is not None and from_dt > to_dt:
        raise _query_validation_error(
            "from",
            "from must not be later than to",
            "value_error.range",
        )

    return event_type, resource_id, from_dt, to_dt


def _audit_checkpoint_response(
    events: list[dict],
) -> AuditEventCheckpointResponse:
    """Build the four-field checkpoint for an already-read canonical event array.

    ``events`` must be the ``mode="json"`` wire view the audit-event search
    serves (UTC datetimes as RFC 3339 strings), so the digest is reproducible
    by an external verifier from the audit-event listing alone.
    """
    return AuditEventCheckpointResponse(
        checkpoint_version=AUDIT_CHECKPOINT_VERSION,
        digest_algorithm=AUDIT_CHECKPOINT_DIGEST_ALGORITHM,
        event_count=len(events),
        events_digest_hex=canonical.audit_events_digest_hex(events),
    )


@router.get(
    "/audit-events/checkpoint",
    response_model=AuditEventCheckpointResponse,
)
def get_audit_events_checkpoint(
    request: Request, session: DbSession
) -> AuditEventCheckpointResponse:
    # Same parameter contract as the audit-event search, minus pagination:
    # a repeated scalar is rejected instead of silently taking the last
    # value, undeclared parameters (including limit/cursor) are rejected
    # rather than ignored, and blank or malformed values are never coerced.
    event_type, resource_id, from_dt, to_dt = _audit_checkpoint_filters(request)

    # Strictly read-only: the checkpoint only re-reads the filtered events
    # in their stable creation order and hashes them; it writes no resource
    # and no audit event, and an empty match set still yields a digest.
    items = service.list_audit_events(
        session, event_type, resource_id, from_dt, to_dt
    )
    # mode="json" yields exactly the wire view the search serves (UTC
    # datetimes as RFC 3339 strings), so the digest is reproducible by an
    # external verifier from the audit-event listing alone.
    events = [
        AuditEventItem.model_validate(item).model_dump(mode="json")
        for item in items
    ]
    return _audit_checkpoint_response(events)


@router.get(
    "/audit-events/checkpoint/package",
    response_model=AuditEventCheckpointPackageResponse,
)
def get_audit_events_checkpoint_package(
    request: Request, session: DbSession
) -> AuditEventCheckpointPackageResponse:
    # Same parameter contract as the checkpoint route: only
    # event_type/resource_id/from/to are accepted, so limit/cursor and every
    # other undeclared, blank, repeated, or malformed parameter is a 422
    # validation_error before any event is read.
    event_type, resource_id, from_dt, to_dt = _audit_checkpoint_filters(request)

    # Strictly read-only, and both halves come from the same read state: the
    # filtered events are read exactly once in their stable creation order,
    # the events member renders those existing public views, and the
    # checkpoint digests exactly that same array under the existing
    # checkpoint canonical rules, so the two can never disagree. No
    # resource, record, or audit event is written; an empty match set yields
    # "events": [] together with the digest of the empty array.
    items = service.list_audit_events(
        session, event_type, resource_id, from_dt, to_dt
    )
    event_views = [AuditEventItem.model_validate(item) for item in items]
    # mode="json" yields exactly the wire view served in this response's
    # events member, so the checkpoint binds to exactly those events.
    events = [view.model_dump(mode="json") for view in event_views]
    return AuditEventCheckpointPackageResponse(
        checkpoint=_audit_checkpoint_response(events),
        events=event_views,
    )


@router.post(
    "/audit-events/checkpoint-verifications",
    response_model=AuditCheckpointVerificationResponse,
    response_model_exclude_none=True,
)
async def verify_audit_events_checkpoint(
    payload: AuditCheckpointVerificationCreate, request: Request
) -> AuditCheckpointVerificationResponse:
    # Strictly stateless: no session is injected, so nothing is queried,
    # created, or modified, and no audit event is written. Verification
    # uses the request body alone; no event or resource id is ever
    # resolved against local state.
    body = await request.json()
    # The digest commits to the event array exactly as received: the raw
    # JSON values (array order, datetime spellings), not any parsed or
    # re-serialized form. Field validation has already guaranteed every
    # element is canonicalizable.
    computed_digest_hex = canonical.audit_events_digest_hex(body["events"])
    if computed_digest_hex == payload.checkpoint.events_digest_hex:
        return AuditCheckpointVerificationResponse(valid=True)
    return AuditCheckpointVerificationResponse(
        valid=False, computed_digest_hex=computed_digest_hex
    )


def _checkpoint_import_response(record) -> AuditCheckpointImportResponse:
    return AuditCheckpointImportResponse(
        id=record.id,
        checkpoint_version=record.checkpoint_version,
        events_digest_hex=record.events_digest_hex,
        event_count=record.event_count,
        received_at=record.created_at,
    )


@router.post(
    "/audit-events/checkpoint-imports",
    response_model=AuditCheckpointImportResponse,
)
async def import_audit_events_checkpoint(
    payload: AuditCheckpointImportCreate,
    request: Request,
    session: DbSession,
    response: Response,
) -> AuditCheckpointImportResponse:
    # Structure, fixed version/algorithm, strict digest spelling, the
    # event-item fields and timestamps, and the claimed event count have all
    # passed request validation exactly as on the stateless verification
    # route. The digest match is enforced here over the raw received JSON,
    # so array order and datetime spellings participate exactly as
    # received; a mismatch is a 422 and writes nothing.
    body = await request.json()
    computed_digest_hex = canonical.audit_events_digest_hex(body["events"])
    if computed_digest_hex != payload.checkpoint.events_digest_hex:
        raise AuditCheckpointImportValidationError(
            "events_digest_mismatch",
            details={"computed_digest_hex": computed_digest_hex},
        )

    # Verification is decided entirely by the request body: the described
    # events are never resolved against local audit state, so whether they
    # exist locally cannot change the receipt, and no event is created or
    # modified.
    record, created = service.create_audit_checkpoint_import(session, payload)
    # First registration of this receiving identity -> 201; a retried
    # submission -> 200 with the original record and no new audit event.
    response.status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return _checkpoint_import_response(record)


_CHECKPOINT_IMPORTS_PARAMS = frozenset(
    {
        "checkpoint_version",
        "events_digest_hex",
        "event_count",
        "limit",
        "cursor",
    }
)


@router.get(
    "/audit-events/checkpoint-imports",
    response_model=AuditCheckpointImportPageResponse,
)
def list_audit_events_checkpoint_imports(
    request: Request,
    session: DbSession,
    checkpoint_version: str | None = Query(default=None),
    events_digest_hex: str | None = Query(default=None),
    event_count: str | None = Query(default=None),
    limit: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
) -> AuditCheckpointImportPageResponse:
    # Raw multi-values are inspected deliberately: a repeated scalar is
    # rejected instead of silently taking the last value, undeclared
    # parameters are rejected rather than ignored, and a blank filter/limit
    # is never coerced to a default.
    raw = request.query_params

    unknown = set(raw) - _CHECKPOINT_IMPORTS_PARAMS
    if unknown:
        # A typo (e.g. ``checkpoint_versions``) never silently changes the
        # search.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    checkpoint_version = _parse_nonempty_filter(raw, "checkpoint_version")
    events_digest_hex = _parse_nonempty_filter(raw, "events_digest_hex")
    event_count = _parse_nonnegative_int_param(raw, "event_count")

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_CHECKPOINT_IMPORTS_LIMIT,
        service.MIN_CHECKPOINT_IMPORTS_LIMIT,
        service.MAX_CHECKPOINT_IMPORTS_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.checkpoint_imports_cursor_secret,
                pagination.CHECKPOINT_IMPORTS_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: every effective
        # filter and the effective limit must match exactly.
        expected = {
            "checkpoint_version": checkpoint_version,
            "events_digest_hex": events_digest_hex,
            "event_count": event_count,
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Strictly read-only: the search writes no resource and no audit event.
    items = service.list_audit_checkpoint_imports(
        session, checkpoint_version, events_digest_hex, event_count
    )
    total = len(items)

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        page = items[offset : offset + page_limit]
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.checkpoint_imports_cursor_secret,
                pagination.CHECKPOINT_IMPORTS_CURSOR,
                {
                    "checkpoint_version": checkpoint_version,
                    "events_digest_hex": events_digest_hex,
                    "event_count": event_count,
                    "limit": page_limit,
                    "offset": next_offset,
                },
            )

    return AuditCheckpointImportPageResponse(
        items=[_checkpoint_import_response(item) for item in page],
        count=total,
        next_cursor=next_cursor,
    )


@router.get(
    "/audit-events/checkpoint-imports/{import_id}",
    response_model=AuditCheckpointImportResponse,
)
def get_audit_events_checkpoint_import(
    import_id: str, session: DbSession
) -> AuditCheckpointImportResponse:
    # Strictly read-only: a receipt read writes no resource and no audit
    # event. An unknown id is an explicit, specific 404.
    record = service.get_audit_checkpoint_import(session, import_id)
    return _checkpoint_import_response(record)


def _local_audit_checkpoint(session: Session) -> AuditEventCheckpointResponse:
    """Compute the current unfiltered local audit-event checkpoint.

    Exactly the existing ``GET /v1/audit-events/checkpoint`` rules with no
    filters: every local event is read once in stable creation order and
    rendered through the same public wire view, so the fixed version,
    digest algorithm, event count, canonical digest, and UTC representation
    are identical to that route. Strictly read-only.
    """
    items = service.list_audit_events(session)
    # mode="json" yields exactly the wire view the search serves (UTC
    # datetimes as RFC 3339 strings), so the digest is reproducible by an
    # external verifier from the audit-event listing alone.
    events = [
        AuditEventItem.model_validate(item).model_dump(mode="json")
        for item in items
    ]
    return _audit_checkpoint_response(events)


@router.get(
    "/audit-events/checkpoint-imports/{import_id}/reconciliation",
    response_model=AuditCheckpointImportReconciliationResponse,
)
def reconcile_audit_events_checkpoint_import(
    import_id: str, request: Request, session: DbSession
) -> AuditCheckpointImportReconciliationResponse:
    # Same boundary as the other import read routes: any (or repeated)
    # query parameter is a 422 before the receipt lookup.
    _reject_any_query_param(request)
    # Strictly read-only: the reconciliation writes no resource, receipt, or
    # audit event. An unknown receipt id is the existing
    # audit_checkpoint_import_not_found 404.
    record = service.get_audit_checkpoint_import(session, import_id)

    # The current local sequence is always read unfiltered; the receipt's
    # imported event array is not persisted and is never read or echoed.
    local_checkpoint = _local_audit_checkpoint(session)
    matches = (
        record.checkpoint_version == local_checkpoint.checkpoint_version
        and record.event_count == local_checkpoint.event_count
        and record.events_digest_hex == local_checkpoint.events_digest_hex
    )
    return AuditCheckpointImportReconciliationResponse(
        import_id=record.id,
        local_checkpoint=local_checkpoint,
        matches=matches,
    )


_CHECKPOINT_IMPORT_RECONCILIATIONS_PARAMS = frozenset({"limit", "cursor"})


def _checkpoint_import_reconciliation_item(
    record, local_checkpoint: AuditEventCheckpointResponse, matches: bool
) -> AuditCheckpointImportReconciliationItem:
    # The existing single-receipt public view, with the current unfiltered
    # local checkpoint and its match verdict added; the imported event array
    # is absent.
    return AuditCheckpointImportReconciliationItem(
        id=record.id,
        checkpoint_version=record.checkpoint_version,
        events_digest_hex=record.events_digest_hex,
        event_count=record.event_count,
        received_at=record.created_at,
        local_checkpoint=local_checkpoint,
        matches=matches,
    )


@router.get(
    "/audit-events/checkpoint-import-reconciliations",
    response_model=AuditCheckpointImportReconciliationPageResponse,
)
def list_audit_events_checkpoint_import_reconciliations(
    request: Request,
    session: DbSession,
    limit: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
) -> AuditCheckpointImportReconciliationPageResponse:
    # Raw multi-values are inspected deliberately: a repeated scalar is
    # rejected instead of silently taking the last value, undeclared
    # parameters are rejected rather than ignored, and a blank limit is
    # never coerced to the default.
    raw = request.query_params

    unknown = set(raw) - _CHECKPOINT_IMPORT_RECONCILIATIONS_PARAMS
    if unknown:
        # The collection accepts only limit/cursor; a typo never silently
        # changes the page.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_CHECKPOINT_IMPORT_RECONCILIATIONS_LIMIT,
        service.MIN_CHECKPOINT_IMPORT_RECONCILIATIONS_LIMIT,
        service.MAX_CHECKPOINT_IMPORT_RECONCILIATIONS_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.checkpoint_import_reconciliations_cursor_secret,
                pagination.CHECKPOINT_IMPORT_RECONCILIATIONS_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: the collection is
        # unfiltered, so the effective limit is the only bound claim.
        if claims["limit"] != page_limit:
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Strictly read-only: the listing writes no resource, receipt, or audit
    # event. Receipts follow stable creation order.
    records = service.list_audit_checkpoint_import_reconciliations(session)
    total = len(records)

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        page = records[offset : offset + page_limit]
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.checkpoint_import_reconciliations_cursor_secret,
                pagination.CHECKPOINT_IMPORT_RECONCILIATIONS_CURSOR,
                {"limit": page_limit, "offset": next_offset},
            )

    # The current local sequence is unfiltered and identical for every
    # receipt on the page; read it once and reconcile each receipt's stored
    # identity against it independently. The imported event arrays are not
    # persisted and are never read or echoed.
    local_checkpoint = _local_audit_checkpoint(session)
    items = []
    for record in page:
        matches = (
            record.checkpoint_version == local_checkpoint.checkpoint_version
            and record.event_count == local_checkpoint.event_count
            and record.events_digest_hex == local_checkpoint.events_digest_hex
        )
        items.append(
            _checkpoint_import_reconciliation_item(
                record, local_checkpoint, matches
            )
        )

    return AuditCheckpointImportReconciliationPageResponse(
        items=items,
        count=total,
        next_cursor=next_cursor,
    )


# --- Signed audit checkpoint exchange imports -------------------------------------


_AUDIT_EXCHANGE_POST_PATH = "/audit-exchanges"


def _audit_exchange_import_response(record) -> dict:
    """Build the compact public receipt dict for one audit exchange import."""
    return AuditExchangeImportResponse(
        id=record.id,
        signature_version=record.signature_version,
        signer_subject=record.signer_subject,
        public_key=base64.b64encode(record.public_key).decode("ascii"),
        package_digest_hex=record.package_digest_hex,
        signature_digest_hex=record.signature_digest_hex,
        received_at=record.created_at,
    ).model_dump(mode="json")


@router.post(_AUDIT_EXCHANGE_POST_PATH)
async def import_audit_exchange(
    payload: AuditExchangeImportCreate,
    request: Request,
    session: DbSession,
) -> Response:
    # The import route accepts no query parameters: any (or repeated)
    # parameter is a 422 validation_error before the package is read.
    _reject_any_query_param(request)

    # Structure, the fixed checkpoint version/algorithm, the claimed event
    # count, the strict event fields and timestamps, the signature metadata
    # fields, standard Base64 key/signature lengths, and the
    # 64-lowercase-hex digest spelling have all passed request validation.
    # The digest bindings are enforced here over the raw received JSON, so
    # root/array order and datetime spellings participate exactly as
    # received; every mismatch is a 422 validation_error and writes
    # nothing.
    body = await request.json()
    raw_package = body["package"]
    metadata = payload.signature_metadata

    # 1. The events array digest under the existing checkpoint rules: the
    #    received array order participates in the recomputation unchanged.
    computed_events_digest_hex = canonical.audit_events_digest_hex(
        raw_package["events"]
    )
    if computed_events_digest_hex != payload.package.checkpoint.events_digest_hex:
        raise AuditExchangeImportValidationError(
            "events_digest_mismatch",
            details={"computed_digest_hex": computed_events_digest_hex},
        )

    # 2. The full package digest: root members (checkpoint, events) and the
    #    events array keep their received order; nested object keys sort by
    #    Unicode code point under the package canonical rules.
    computed_package_digest_hex = canonical.audit_checkpoint_package_digest_hex(
        raw_package
    )
    if computed_package_digest_hex != metadata.package_digest_hex:
        raise AuditExchangeImportValidationError(
            "package_digest_mismatch",
            details={"computed_digest_hex": computed_package_digest_hex},
        )

    # 3. The Ed25519 signature binds, in order, the fixed exchange version,
    #    the signing subject, the package digest algorithm, and the
    #    whole-package digest -- over the exact UTF-8 compact JSON array.
    #    Only now, after every structural and digest check, is verification
    #    attempted; a failure is its own distinct 422 code and writes
    #    nothing.
    message = signing.audit_exchange_message_bytes(
        metadata.signer_subject,
        metadata.package_digest_algorithm,
        metadata.package_digest_hex,
    )
    if not ed25519.verify(
        metadata.public_key, message, metadata.signature
    ):
        raise AuditSignatureVerificationError(
            details={"reason": "signature_verification_failed"}
        )

    # Only the SHA-256 digest of the signature is ever passed on for
    # storage; the raw signature and the package are not persisted.
    signature_digest_hex = hashlib.sha256(metadata.signature).hexdigest()

    # Verification is decided entirely by the request body: no described
    # event is ever resolved against local state, so whether the events
    # exist locally cannot change the receipt, and no local resource is
    # queried or created.
    record, created = service.create_audit_exchange_import(
        session, payload, signature_digest_hex
    )
    # First registration of this package identity -> 201; an exact retried
    # submission (same subject, key, and signature) -> 200 with the
    # original receipt and no new audit event. A different subject, key, or
    # signature for the same package is a 422 from the service with the
    # original record untouched.
    status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return _render_compact_json(
        _audit_exchange_import_response(record),
        status_code=status_code,
    )


_AUDIT_EXCHANGES_PARAMS = frozenset(
    {
        "signature_version",
        "signer_subject",
        "public_key",
        "package_digest_hex",
        "limit",
        "cursor",
    }
)


@router.get(_AUDIT_EXCHANGE_POST_PATH)
async def list_audit_exchanges(
    request: Request, session: DbSession
) -> Response:
    # The GET request body must be empty: carrying any bytes (even
    # whitespace, arbitrary bytes, or malformed JSON) is a 422 validated
    # before any query parameter or receipt is read.
    raw_body = await request.body()
    if raw_body:
        raise _query_validation_error(
            "body",
            "request body must be empty",
            "value_error.body",
        )

    # Raw multi-values are inspected deliberately: a repeated scalar is
    # rejected instead of silently taking the last value, undeclared
    # parameters are rejected rather than ignored, and a blank filter/limit
    # is never coerced to a default.
    raw = request.query_params

    unknown = set(raw) - _AUDIT_EXCHANGES_PARAMS
    if unknown:
        # A typo never silently changes the search.
        field = sorted(unknown)[0]
        raise _query_validation_error(
            field, f"unknown query parameter: {field}", "value_error.unknown"
        )

    signature_version = _parse_nonempty_filter(raw, "signature_version")
    signer_subject = _parse_nonempty_filter(raw, "signer_subject")
    public_key = _parse_nonempty_filter(raw, "public_key")
    package_digest_hex = _parse_nonempty_filter(raw, "package_digest_hex")

    page_limit = _parse_int_param(
        raw,
        "limit",
        service.DEFAULT_AUDIT_EXCHANGE_IMPORTS_LIMIT,
        service.MIN_AUDIT_EXCHANGE_IMPORTS_LIMIT,
        service.MAX_AUDIT_EXCHANGE_IMPORTS_LIMIT,
    )

    cursor = _parse_once(raw, "cursor")
    offset = 0
    if cursor is not None:
        try:
            claims = pagination.decode_typed_cursor(
                request.app.state.audit_exchange_imports_cursor_secret,
                pagination.AUDIT_EXCHANGE_IMPORTS_CURSOR,
                cursor,
            )
        except InvalidCursorError as exc:
            raise _query_validation_error(
                "cursor",
                "cursor is malformed, expired, or invalid",
                "value_error.cursor",
            ) from exc
        # The cursor only resumes the query that issued it: every effective
        # filter and the effective limit must match exactly.
        expected = {
            "signature_version": signature_version,
            "signer_subject": signer_subject,
            "public_key": public_key,
            "package_digest_hex": package_digest_hex,
            "limit": page_limit,
        }
        if any(claims[key] != value for key, value in expected.items()):
            raise _query_validation_error(
                "cursor",
                "cursor does not match the query parameters",
                "value_error.cursor",
            )
        offset = claims["offset"]

    # Strictly read-only: the search writes no receipt, resource, or audit
    # event. Filter values are never resolved for existence, so an unknown
    # value is an empty collection rather than a 404. Each item is exactly
    # the single-receipt public view; the package, the raw signature, and
    # every private key are never persisted and so can never be echoed.
    items = service.list_audit_exchange_imports(
        session,
        signature_version,
        signer_subject,
        public_key,
        package_digest_hex,
    )
    total = len(items)

    next_cursor: str | None = None
    if offset >= total:
        # At or past the end of the (stable) result set: the page is empty
        # and no further cursor can be issued.
        page = []
    else:
        page = items[offset : offset + page_limit]
        next_offset = offset + len(page)
        if next_offset < total:
            next_cursor = pagination.encode_typed_cursor(
                request.app.state.audit_exchange_imports_cursor_secret,
                pagination.AUDIT_EXCHANGE_IMPORTS_CURSOR,
                {
                    "signature_version": signature_version,
                    "signer_subject": signer_subject,
                    "public_key": public_key,
                    "package_digest_hex": package_digest_hex,
                    "limit": page_limit,
                    "offset": next_offset,
                },
            )

    result = AuditExchangeImportPageResponse(
        items=[_audit_exchange_import_response(item) for item in page],
        count=total,
        next_cursor=next_cursor,
    )
    # Compact UTF-8 JSON, null literal, integral numbers only, terminated
    # by exactly one newline; members appear as items, count, next_cursor.
    return _render_compact_json(result.model_dump(mode="json"))


@router.get("/audit-exchanges/{import_id}")
def get_audit_exchange(
    import_id: str, request: Request, session: DbSession
) -> Response:
    # The receipt read takes no query parameters: any (or repeated)
    # parameter is a 422 before the receipt lookup. PUT/PATCH/DELETE and
    # every other non-GET method on this path is the framework's 405
    # method_not_allowed.
    _reject_any_query_param(request)
    # Strictly read-only: a receipt read writes no resource and no audit
    # event. An unknown id is an explicit, specific 404.
    record = service.get_audit_exchange_import(session, import_id)
    return _render_compact_json(_audit_exchange_import_response(record))
