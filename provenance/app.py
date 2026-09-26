"""FastAPI application factory.

The database location comes from explicit :class:`~provenance.config.Settings`
(constructor, ``PROVENANCE_DATABASE_URL``, or the CLI flag). Tables are created
automatically on first startup.
"""

from __future__ import annotations

import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI

from provenance import __version__
from provenance.api import router as v1_router
from provenance.config import Settings
from provenance.database import (
    init_db,
    make_engine,
    make_session_factory,
)
from provenance.errors import register_exception_handlers


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    engine = make_engine(settings.database_url)
    # Create tables eagerly so the database is ready as soon as the app exists.
    init_db(engine)
    session_factory = make_session_factory(engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Re-ensure the schema on every startup, not just at factory time:
        # shutdown disposes the engine, and for an in-memory database that
        # drops the whole schema, so a restarted lifespan must recreate the
        # tables before the reused connection serves requests again. The
        # call is idempotent on an already-migrated database.
        init_db(engine)
        try:
            yield
        finally:
            engine.dispose()

    app = FastAPI(
        title="Digital Content Provenance API",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    # Per-process HMAC secrets for opaque pagination cursors. Cursors are
    # stateless and only valid within the process that minted them; a
    # restart rotates the secret and renders outstanding cursors invalid
    # (reported as 422 validation_error) rather than guessable.
    app.state.lineage_cursor_secret = secrets.token_bytes(32)
    app.state.content_evidence_cursor_secret = secrets.token_bytes(32)
    app.state.audit_events_cursor_secret = secrets.token_bytes(32)
    app.state.exchange_imports_cursor_secret = secrets.token_bytes(32)
    app.state.exchange_import_reconciliations_cursor_secret = (
        secrets.token_bytes(32)
    )
    app.state.checkpoint_imports_cursor_secret = secrets.token_bytes(32)
    app.state.checkpoint_import_reconciliations_cursor_secret = (
        secrets.token_bytes(32)
    )
    app.state.claims_cursor_secret = secrets.token_bytes(32)
    app.state.evidence_bundles_cursor_secret = secrets.token_bytes(32)
    app.state.authentication_key_rotations_cursor_secret = (
        secrets.token_bytes(32)
    )
    app.state.attestation_access_grants_cursor_secret = secrets.token_bytes(32)
    app.state.attestation_access_grant_revocations_cursor_secret = (
        secrets.token_bytes(32)
    )
    app.state.content_export_jobs_cursor_secret = secrets.token_bytes(32)
    app.state.claim_supersession_lineage_cursor_secret = (
        secrets.token_bytes(32)
    )
    app.state.trust_policies_cursor_secret = secrets.token_bytes(32)
    app.state.actors_cursor_secret = secrets.token_bytes(32)
    app.state.contents_cursor_secret = secrets.token_bytes(32)
    app.state.attestation_revocations_cursor_secret = secrets.token_bytes(32)
    app.state.content_relations_cursor_secret = secrets.token_bytes(32)
    app.state.content_coverage_search_cursor_secret = secrets.token_bytes(32)
    app.state.revocation_impacts_cursor_secret = secrets.token_bytes(32)
    app.state.impact_imports_cursor_secret = secrets.token_bytes(32)
    app.state.impact_import_reconciliations_cursor_secret = (
        secrets.token_bytes(32)
    )
    app.state.impact_recon_exchange_imports_cursor_secret = (
        secrets.token_bytes(32)
    )
    app.state.audit_exchange_imports_cursor_secret = secrets.token_bytes(32)

    register_exception_handlers(app)
    app.include_router(v1_router)

    @app.get("/healthz", tags=["meta"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app
