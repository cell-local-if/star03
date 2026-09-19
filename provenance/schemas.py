"""Pydantic request/response schemas for the versioned JSON API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ActorCreate(BaseModel):
    actor_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=256)
    actor_type: str = Field(min_length=1, max_length=64)


class ActorResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    actor_id: str
    name: str
    actor_type: str
    created_at: datetime


class ContentCreate(BaseModel):
    digest_algorithm: str = Field(min_length=1, max_length=32)
    # Length/charset rules are enforced by the service layer so malformed
    # digests yield the dedicated invalid_digest_hex error code.
    digest_hex: str = Field(min_length=1, max_length=256)
    media_type: str = Field(min_length=1, max_length=256)
    title: str | None = Field(default=None, max_length=512)
    actor_id: str = Field(min_length=1, max_length=128)


class ContentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    content_id: str
    digest_algorithm: str
    digest_hex: str
    media_type: str
    title: str | None
    actor_id: str
    created_at: datetime


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
