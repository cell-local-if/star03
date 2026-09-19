"""UTC time helpers.

All API-visible timestamps are timezone-aware UTC. Stored values use a fixed
``+00:00`` offset so ordering and JSON rendering are unambiguous.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)
