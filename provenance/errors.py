"""Domain errors and their JSON rendering.

Every failure class maps to a stable error code and HTTP status so callers can
distinguish conflicts, unknown references, malformed input, and missing
resources programmatically::

    {"error": {"code": "actor_already_exists", "message": "...", "details": {...}}}
"""

from __future__ import annotations

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class DomainError(Exception):
    """Base class for expected, client-facing domain failures."""

    status_code: int = 400
    code: str = "domain_error"
    message: str = "Request failed."

    def __init__(self, message: str | None = None, details: dict | None = None):
        super().__init__(message or self.message)
        if message is not None:
            self.message = message
        self.details = details or {}

    def to_response(self) -> JSONResponse:
        body: dict = {"code": self.code, "message": self.message}
        if self.details:
            body["details"] = self.details
        return JSONResponse(
            status_code=self.status_code,
            content={"error": body},
        )


class ActorAlreadyExistsError(DomainError):
    status_code = 409
    code = "actor_already_exists"
    message = "An actor with the given identifier already exists."

    def __init__(self, actor_id: str):
        super().__init__(details={"actor_id": actor_id})


class UnknownActorError(DomainError):
    status_code = 404
    code = "unknown_actor"
    message = "The referenced source actor does not exist."

    def __init__(self, actor_id: str):
        super().__init__(details={"actor_id": actor_id})


class ContentNotFoundError(DomainError):
    status_code = 404
    code = "content_not_found"
    message = "The requested content does not exist."

    def __init__(self, content_id: str):
        super().__init__(details={"content_id": content_id})


class ClaimNotFoundError(DomainError):
    status_code = 404
    code = "claim_not_found"
    message = "The requested claim does not exist."

    def __init__(self, claim_id: str):
        super().__init__(details={"claim_id": claim_id})


class EvidenceBundleNotFoundError(DomainError):
    status_code = 404
    code = "evidence_bundle_not_found"
    message = "The requested evidence bundle does not exist."

    def __init__(self, evidence_bundle_id: str):
        super().__init__(details={"evidence_bundle_id": evidence_bundle_id})


class AttestationNotFoundError(DomainError):
    status_code = 404
    code = "attestation_not_found"
    message = "The requested attestation does not exist."

    def __init__(self, attestation_id: str):
        super().__init__(details={"attestation_id": attestation_id})


class AttestationRevocationNotFoundError(DomainError):
    status_code = 404
    code = "attestation_revocation_not_found"
    message = "The requested attestation revocation does not exist."

    def __init__(self, revocation_id: str):
        super().__init__(details={"revocation_id": revocation_id})


class EvidenceBundleExchangeImportNotFoundError(DomainError):
    status_code = 404
    code = "evidence_bundle_exchange_import_not_found"
    message = "The requested evidence bundle exchange import does not exist."

    def __init__(self, import_id: str):
        super().__init__(details={"import_id": import_id})


class EvidenceBundleExchangeImportValidationError(DomainError):
    """A structurally valid exchange-import request that fails verification.

    A manifest digest that does not match the canonical SHA-256 of the
    received snapshot renders with the same ``validation_error`` code and
    422 status as a malformed payload: the package is refused before any
    record or audit row is written.
    """

    status_code = 422
    code = "validation_error"
    message = "Request payload failed validation."

    def __init__(self, reason: str, details: dict | None = None):
        merged = {"reason": reason}
        if details:
            merged.update(details)
        super().__init__(details=merged)


class AuditCheckpointImportNotFoundError(DomainError):
    status_code = 404
    code = "audit_checkpoint_import_not_found"
    message = "The requested audit checkpoint import does not exist."

    def __init__(self, import_id: str):
        super().__init__(details={"import_id": import_id})


class AuditCheckpointImportValidationError(DomainError):
    """A structurally valid checkpoint-import request that fails verification.

    An events digest that does not match the canonical SHA-256 of the
    received event array renders with the same ``validation_error`` code
    and 422 status as a malformed payload: the checkpoint is refused before
    any receipt or audit row is written.
    """

    status_code = 422
    code = "validation_error"
    message = "Request payload failed validation."

    def __init__(self, reason: str, details: dict | None = None):
        merged = {"reason": reason}
        if details:
            merged.update(details)
        super().__init__(details=merged)


class ContentRelationNotFoundError(DomainError):
    status_code = 404
    code = "content_relation_not_found"
    message = "The requested content relation does not exist."

    def __init__(self, relation_id: str):
        super().__init__(details={"relation_id": relation_id})


class ContentRelationValidationError(DomainError):
    """A structurally valid relation request that the graph rejects.

    Self-loops and edges that would close a cycle render as the same
    ``validation_error`` code as malformed payloads: both are client input
    the service refuses to persist.
    """

    status_code = 422
    code = "validation_error"
    message = "Request payload failed validation."

    def __init__(self, reason: str):
        super().__init__(details={"reason": reason})


class ClaimSupersessionNotFoundError(DomainError):
    status_code = 404
    code = "claim_supersession_not_found"
    message = "The requested claim supersession does not exist."

    def __init__(self, supersession_id: str):
        super().__init__(details={"supersession_id": supersession_id})


class ClaimSupersessionValidationError(DomainError):
    """A structurally valid supersession request the graph rejects.

    Self-supersessions, endpoints asserting different content, and edges
    that would close a cycle render as the same ``validation_error`` code
    as malformed payloads: both are client input the service refuses to
    persist.
    """

    status_code = 422
    code = "validation_error"
    message = "Request payload failed validation."

    def __init__(self, reason: str):
        super().__init__(details={"reason": reason})


class AttestationVerificationError(DomainError):
    status_code = 422
    code = "attestation_verification_failed"
    message = "The attestation signature could not be verified."

    def __init__(self, details: dict | None = None):
        super().__init__(details=details)


class ProtectedAccessValidationError(DomainError):
    """A malformed protected request (credentials or grant fields).

    Renders with the same ``validation_error`` code as request-body
    validation failures on both protected routes.
    """

    status_code = 422
    code = "validation_error"
    message = "Request payload failed validation."

    def __init__(self, reason: str):
        super().__init__(details={"reason": reason})


class ProtectedResourceNotFoundError(DomainError):
    """Opaque 404 for the protected read route.

    A missing attestation, an unauthenticated caller, and an unauthorized
    caller all render identically so resource existence is never revealed
    to a caller without access.
    """

    status_code = 404
    code = "not_found"
    message = "The requested resource does not exist."

    def __init__(self) -> None:
        super().__init__()


class LineageValidationError(DomainError):
    """A malformed lineage query (``direction`` / ``max_depth``).

    Renders with the same ``validation_error`` code and issue-shaped details
    as request-body validation failures.
    """

    status_code = 422
    code = "validation_error"
    message = "Request payload failed validation."

    def __init__(self, issues: list[dict]):
        super().__init__(details={"issues": issues})


def _validation_body(exc: RequestValidationError) -> dict:
    # Keep the payload small, stable, and JSON-safe: loc/msg/type per error.
    issues = [
        {
            "loc": [str(part) for part in error.get("loc", [])],
            "msg": error.get("msg", ""),
            "type": error.get("type", ""),
        }
        for error in exc.errors()
    ]
    return {
        "error": {
            "code": "validation_error",
            "message": "Request payload failed validation.",
            "details": {"issues": issues},
        }
    }


def register_exception_handlers(app) -> None:
    @app.exception_handler(DomainError)
    async def _handle_domain_error(_: Request, exc: DomainError) -> JSONResponse:
        return exc.to_response()

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content=_validation_body(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_error(
        _: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": {
                        404: "not_found",
                        405: "method_not_allowed",
                        409: "conflict",
                    }.get(exc.status_code, "http_error"),
                    "message": str(exc.detail),
                }
            },
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # Never leak internals (or content-bearing payloads) to clients.
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Internal error."}},
        )
