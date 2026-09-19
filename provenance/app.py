"""FastAPI application factory exposing the versioned JSON API."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, sessionmaker

from . import service
from .config import Settings, load_settings
from .db import init_schema, make_engine, make_session_factory
from .errors import ApiError
from .schemas import (
    ActorCreate,
    ActorResponse,
    ContentCreate,
    ContentResponse,
    ErrorResponse,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    engine = make_engine(settings.database_url)
    init_schema(engine)
    session_factory = make_session_factory(engine)

    app = FastAPI(title="Digital Content Provenance", version="1")
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory

    def get_session() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    @app.exception_handler(ApiError)
    def handle_api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(RequestValidationError)
    def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": "request body failed validation",
                    "details": exc.errors(),
                }
            },
        )

    @app.post(
        "/v1/actors",
        status_code=201,
        response_model=ActorResponse,
        responses={409: {"model": ErrorResponse}},
    )
    def post_actor(
        payload: ActorCreate, session: Session = Depends(get_session)
    ) -> ActorResponse:
        actor = service.create_actor(session, payload)
        return ActorResponse.model_validate(actor)

    @app.post(
        "/v1/contents",
        status_code=201,
        response_model=ContentResponse,
        responses={
            200: {"model": ContentResponse},
            409: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
        },
    )
    def post_content(
        payload: ContentCreate,
        response: Response,
        session: Session = Depends(get_session),
    ) -> ContentResponse:
        content, created = service.create_content(session, payload)
        if not created:
            # Idempotent replay: return the existing resource with 200.
            response.status_code = 200
        return ContentResponse.model_validate(content)

    @app.get(
        "/v1/contents/{content_id}",
        response_model=ContentResponse,
        responses={404: {"model": ErrorResponse}},
    )
    def get_content(
        content_id: str, session: Session = Depends(get_session)
    ) -> ContentResponse:
        content = service.get_content(session, content_id)
        return ContentResponse.model_validate(content)

    @app.get("/v1/contents", response_model=list[ContentResponse])
    def get_contents(
        actor_id: str | None = Query(default=None),
        session: Session = Depends(get_session),
    ) -> list[ContentResponse]:
        contents = service.list_contents(session, actor_id=actor_id)
        return [ContentResponse.model_validate(c) for c in contents]

    return app
