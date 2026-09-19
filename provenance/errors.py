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
