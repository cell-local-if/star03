"""Runtime configuration for the provenance service.

All settings are resolved from explicit environment variables so the
database location and bind address are never implicit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

DEFAULT_DATABASE_URL = "sqlite:///./provenance.db"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


@dataclass(frozen=True)
class Settings:
    database_url: str = DEFAULT_DATABASE_URL
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    source = os.environ if env is None else env
    return Settings(
        database_url=source.get("PROVENANCE_DATABASE_URL", DEFAULT_DATABASE_URL),
        host=source.get("PROVENANCE_HOST", DEFAULT_HOST),
        port=int(source.get("PROVENANCE_PORT", str(DEFAULT_PORT))),
    )
