"""Entry point: ``python -m provenance``.

Examples::

    python -m provenance
    python -m provenance --database-url sqlite:////tmp/provenance.db
    PROVENANCE_DATABASE_URL=sqlite:///./x.db python -m provenance --port 8080

Disaster recovery (no service is started, no migrations run, no audit rows
are written)::

    python -m provenance backup --database-url sqlite:///./provenance.db --output /safe/snapshot.db
    python -m provenance restore --database-url sqlite:///./provenance.db --input /safe/snapshot.db --force
"""

from __future__ import annotations

import argparse
import json
import sys

import uvicorn

from provenance.app import create_app
from provenance.backup import BackupError, perform_backup, perform_restore
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

    subparsers = parser.add_subparsers(dest="command")

    backup = subparsers.add_parser(
        "backup",
        help="Write a transaction-consistent snapshot of the database.",
    )
    backup.add_argument(
        "--database-url",
        # SUPPRESS keeps a top-level ``--database-url`` given before the
        # subcommand from being clobbered by the subparser default.
        default=argparse.SUPPRESS,
        help="SQLite file URL of the database to snapshot.",
    )
    backup.add_argument(
        "--output",
        required=True,
        help="Destination file for the snapshot.",
    )
    backup.add_argument(
        "--force",
        action="store_true",
        help="Replace the output file if it already exists.",
    )

    restore = subparsers.add_parser(
        "restore",
        help="Replace the database with a validated backup snapshot.",
    )
    restore.add_argument(
        "--database-url",
        default=argparse.SUPPRESS,
        help="SQLite file URL of the database to replace.",
    )
    restore.add_argument(
        "--input",
        required=True,
        help="Backup snapshot to restore from.",
    )
    restore.add_argument(
        "--force",
        action="store_true",
        help="Replace the database file if it already exists.",
    )

    return parser.parse_args()


def _run_recovery(args: argparse.Namespace) -> int:
    """Run a backup/restore subcommand; return the process exit code."""
    # Same resolution order as the service: flag > environment > default.
    settings = Settings.from_env(database_url=args.database_url)
    try:
        if args.command == "backup":
            result = perform_backup(settings.database_url, args.output, args.force)
        else:
            result = perform_restore(settings.database_url, args.input, args.force)
    except BackupError as exc:
        sys.stderr.write(exc.envelope() + "\n")
        return 1
    sys.stdout.write(json.dumps(result, separators=(",", ":")) + "\n")
    return 0


def main() -> None:
    args = _parse_args()
    if args.command in ("backup", "restore"):
        sys.exit(_run_recovery(args))
    settings = Settings.from_env(database_url=args.database_url)
    app = create_app(settings)
    # Passing the app object directly keeps the already-configured engine and
    # avoids importing a settings-bearing module string.
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
