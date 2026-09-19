"""Application errors mapped to distinguishable JSON error responses."""

from __future__ import annotations


class ApiError(Exception):
    """An error that becomes a structured JSON response of the form
    ``{"error": {"code": ..., "message": ...}}``."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def actor_already_exists(actor_id: str) -> ApiError:
    return ApiError(409, "actor_already_exists", f"actor {actor_id!r} already exists")


def unknown_actor(actor_id: str) -> ApiError:
    return ApiError(422, "unknown_actor", f"actor {actor_id!r} does not exist")


def unsupported_digest_algorithm(algorithm: str) -> ApiError:
    return ApiError(
        422,
        "unsupported_digest_algorithm",
        f"digest algorithm {algorithm!r} is not supported; expected 'sha256'",
    )


def invalid_digest_hex() -> ApiError:
    return ApiError(
        422,
        "invalid_digest_hex",
        "digest_hex must be exactly 64 hexadecimal characters",
    )


def content_not_found(content_id: str) -> ApiError:
    return ApiError(404, "content_not_found", f"content {content_id!r} does not exist")
