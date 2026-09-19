"""Entry point: ``python -m provenance``.

Examples::

    python -m provenance
    python -m provenance --database-url sqlite:////tmp/provenance.db
    PROVENANCE_DATABASE_URL=sqlite:///./x.db python -m provenance --port 8080
"""

from __future__ import annotations

import argparse

import uvicorn

from provenance.app import create_app
from provenance.config import DATABASE_URL_ENV, Settings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m provenance",
        description="Run the digital content provenance HTTP service.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help=(
            "SQLAlchemy database URL (e.g. sqlite:///./provenance.db). "
            f"Overrides the {DATABASE_URL_ENV} environment variable."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = Settings.from_env(database_url=args.database_url)
    app = create_app(settings)
    # Passing the app object directly keeps the already-configured engine and
    # avoids importing a settings-bearing module string.
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
