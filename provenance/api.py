"""Versioned HTTP routes (``/v1``)."""

from __future__ import annotations

import base64
import re
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request, Response, status
from sqlalchemy.orm import Session

from provenance import access_signing, canonical, pagination, service
from provenance.access_signing import AccessAuthError
from provenance.errors import (
    LineageValidationError,
    ProtectedAccessValidationError,
    ProtectedResourceNotFoundError,
)
from provenance.models import RELATION_DERIVED_FROM, RELATION_VERSION_OF
from provenance.pagination import InvalidCursorError
from provenance.signing import ATTESTATION_TARGET_TYPES
from provenance.time_utils import parse_rfc3339_utc
from provenance.schemas import (
    EXCHANGE_MANIFEST_DIGEST_ALGORITHM,
    EXCHANGE_MANIFEST_VERSION,
    ActorCreate,
    ActorResponse,
    AttestationAccessGrantCreate,
    AttestationAccessGrantResponse,
    AttestationCreate,
    AttestationListResponse,
    AttestationResponse,
    AttestationRevocationCreate,
    AttestationRevocationListResponse,
    AttestationRevocationResponse,
    AuditEventItem,
    AuditEventPageResponse,
    AuthenticationKeyRotationCreate,
    AuthenticationKeyRotationResponse,
    ClaimCreate,
    ClaimExportItem,
    ClaimListResponse,
    ClaimResponse,
    ContentCreate,
    ContentExportResponse,
    ContentLineageItem,
    ContentLineageResponse,
    ContentListResponse,
    ContentRelationCreate,
    ContentRelationListResponse,
    ContentRelationResponse,
    ContentResponse,
    EvidenceBundleCreate,
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
    # mode="json" yields exactly the wire view the snapshot serves (UTC
    # datetimes as RFC 3339 strings, Base64 keys, plain booleans), so the
    # digest is reproducible by an external verifier from the exchange JSON.
    snapshot_json = snapshot.model_dump(mode="json")
    manifest_digest_hex = canonical.exchange_manifest_digest_hex(snapshot_json)
    return EvidenceBundleExchangeManifestResponse(
        manifest_version=EXCHANGE_MANIFEST_VERSION,
        evidence_bundle_id=evidence_bundle_id,
        digest_algorithm=EXCHANGE_MANIFEST_DIGEST_ALGORITHM,
        manifest_digest_hex=manifest_digest_hex,
    )


@router.get(
    "/evidence-bundles/{evidence_bundle_id}/exchange/package",
    response_model=EvidenceBundleExchangePackageResponse,
)
def get_evidence_bundle_exchange_package(
    evidence_bundle_id: str, request: Request, session: DbSession
) -> EvidenceBundleExchangePackageResponse:
    # Same boundary as the exchange snapshot and manifest routes: any (or
    # repeated) query parameter is a 422 before the bundle lookup.
    _reject_any_query_param(request)
    # Strictly read-only and single-read: the snapshot is assembled once
    # from one session read state, and the manifest is derived from that
    # exact same snapshot object (the snapshot member of this response),
    # so the digest always binds the snapshot served alongside it. No
    # resource, record, or audit event is written; an unknown bundle is the
    # existing evidence_bundle_not_found 404.
    snapshot = _exchange_snapshot_response(evidence_bundle_id, session)
    snapshot_json = snapshot.model_dump(mode="json")
    manifest_digest_hex = canonical.exchange_manifest_digest_hex(snapshot_json)
    manifest = EvidenceBundleExchangeManifestResponse(
        manifest_version=EXCHANGE_MANIFEST_VERSION,
        evidence_bundle_id=evidence_bundle_id,
        digest_algorithm=EXCHANGE_MANIFEST_DIGEST_ALGORITHM,
        manifest_digest_hex=manifest_digest_hex,
    )
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
