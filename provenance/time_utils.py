"""UTC time helpers.

All API-visible timestamps are timezone-aware UTC. Stored values use a fixed
``+00:00`` offset so ordering and JSON rendering are unambiguous.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

# Strict RFC 3339 timestamp expressed in UTC: full date, "T", full time with
# seconds, optional fractional seconds, and an explicit UTC designator ("Z"
# or "+00:00"). Lowercase designators, missing seconds, naive timestamps, and
# non-UTC offsets are not accepted -- nothing is silently normalized.
_RFC3339_UTC_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)"
)


def parse_rfc3339_utc(value: str) -> datetime | None:
    """Parse a strict RFC 3339 UTC timestamp, or return ``None`` if invalid.

    Only an explicit UTC form is accepted; the result is always a
    timezone-aware UTC datetime. Calendar-invalid values (e.g. month 13)
    are rejected along with structurally malformed ones.
    """
    if not _RFC3339_UTC_RE.fullmatch(value):
        return None
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc)


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)
