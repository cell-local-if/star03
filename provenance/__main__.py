"""Entry point: ``python -m provenance``.

Examples::

    python -m provenance
    python -m provenance --database-url sqlite:////tmp/provenance.db
    PROVENANCE_DATABASE_URL=sqlite:///./x.db python -m provenance --port 8080

Logical backup commands (the database URL resolves exactly as for the
server: ``--database-url``, then ``PROVENANCE_DATABASE_URL``, then
``sqlite:///./provenance.db``)::

    python -m provenance export-backup --database-url sqlite:///./provenance.db
    python -m provenance restore-backup --database-url sqlite:///./restored.db
"""

from __future__ import annotations

import argparse

import uvicorn

from provenance.app import create_app
from provenance.config import DATABASE_URL_ENV, Settings

_DATABASE_URL_HELP = (
    "SQLAlchemy database URL (e.g. sqlite:///./provenance.db). "
    f"Overrides the {DATABASE_URL_ENV} environment variable."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m provenance",
        description="Run the digital content provenance HTTP service.",
    )
    parser.add_argument("--database-url", default=None, help=_DATABASE_URL_HELP)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    subparsers = parser.add_subparsers(dest="command")
    export_parser = subparsers.add_parser(
        "export-backup",
        help="Write a logical backup of the database to standard output.",
    )
    # SUPPRESS keeps a ``--database-url`` given before the subcommand (or the
    # environment/default resolution) instead of clobbering it with None.
    export_parser.add_argument(
        "--database-url", default=argparse.SUPPRESS, help=_DATABASE_URL_HELP
    )
    restore_parser = subparsers.add_parser(
        "restore-backup",
        help="Restore a logical backup read from standard input.",
    )
    restore_parser.add_argument(
        "--database-url", default=argparse.SUPPRESS, help=_DATABASE_URL_HELP
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command in ("export-backup", "restore-backup"):
        from provenance.backup import (
            export_backup_command,
            restore_backup_command,
        )

        handler = {
            "export-backup": export_backup_command,
            "restore-backup": restore_backup_command,
        }[args.command]
        raise SystemExit(handler(args.database_url))
    settings = Settings.from_env(database_url=args.database_url)
    app = create_app(settings)
    # Passing the app object directly keeps the already-configured engine and
    # avoids importing a settings-bearing module string.
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
