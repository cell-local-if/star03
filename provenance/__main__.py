"""Entry point: ``python -m provenance``.

Examples::

    python -m provenance
    python -m provenance --database-url sqlite:////tmp/provenance.db
    PROVENANCE_DATABASE_URL=sqlite:///./x.db python -m provenance --port 8080

Disaster recovery (offline; the service is not started)::

    python -m provenance backup --output /backups/provenance.db
    python -m provenance restore --input /backups/provenance.db --force
    python -m provenance verify --input /backups/provenance.db
"""

from __future__ import annotations

import argparse
import json
import sys

from provenance.config import DATABASE_URL_ENV, Settings


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        help="Write a verified snapshot of the database to --output.",
    )
    backup.add_argument(
        "--database-url",
        dest="subcommand_database_url",
        default=None,
        help="Database to snapshot (same resolution as the service).",
    )
    backup.add_argument(
        "--output",
        required=True,
        help="Destination file for the snapshot.",
    )
    backup.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing --output file.",
    )

    restore = subparsers.add_parser(
        "restore",
        help="Replace the database with a verified copy of --input.",
    )
    restore.add_argument(
        "--database-url",
        dest="subcommand_database_url",
        default=None,
        help="Database to replace (same resolution as the service).",
    )
    restore.add_argument(
        "--input",
        required=True,
        help="Snapshot file to restore from.",
    )
    restore.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing database file.",
    )

    verify = subparsers.add_parser(
        "verify",
        help="Verify --input read-only and report its schema version.",
    )
    verify.add_argument(
        "--input",
        required=True,
        help="Snapshot file to verify; it is never modified.",
    )
    return parser.parse_args(argv)


def _emit_error(exc: "BackupError") -> None:
    line = json.dumps(
        {
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            }
        },
        separators=(",", ":"),
    )
    print(line, file=sys.stderr)


def _run_disaster_recovery(args: argparse.Namespace) -> int:
    # Imported lazily so the serve path and its imports stay untouched.
    from provenance.backup import BackupError, run_backup, run_restore

    settings = Settings.from_env(
        database_url=args.subcommand_database_url or args.database_url
    )
    try:
        if args.command == "backup":
            report = run_backup(settings, args.output, force=args.force)
        else:
            report = run_restore(settings, args.input, force=args.force)
    except BackupError as exc:
        _emit_error(exc)
        return 1
    except Exception as exc:  # never leak a traceback to stderr
        _emit_error(
            BackupError(
                "operation_failed",
                "The operation failed unexpectedly.",
                {"reason": str(exc)},
            )
        )
        return 1
    print(json.dumps(report, separators=(",", ":")))
    return 0


def _run_verify(args: argparse.Namespace) -> int:
    # Read-only and URL-free: no Settings are constructed, so neither the
    # CLI flag nor PROVENANCE_DATABASE_URL can influence the verdict.
    from provenance.backup import BackupError, run_verify

    try:
        report = run_verify(args.input)
    except BackupError as exc:
        _emit_error(exc)
        return 1
    except Exception as exc:  # never leak a traceback to stderr
        _emit_error(
            BackupError(
                "operation_failed",
                "The operation failed unexpectedly.",
                {"reason": str(exc)},
            )
        )
        return 1
    print(json.dumps(report, separators=(",", ":")))
    return 0


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "verify":
        raise SystemExit(_run_verify(args))
    if args.command in ("backup", "restore"):
        raise SystemExit(_run_disaster_recovery(args))

    import uvicorn

    from provenance.app import create_app

    settings = Settings.from_env(database_url=args.database_url)
    app = create_app(settings)
    # Passing the app object directly keeps the already-configured engine and
    # avoids importing a settings-bearing module string.
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
