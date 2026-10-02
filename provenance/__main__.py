"""Entry point: ``python -m provenance``.

Examples::

    python -m provenance
    python -m provenance --database-url sqlite:////tmp/provenance.db
    PROVENANCE_DATABASE_URL=sqlite:///./x.db python -m provenance --port 8080
    python -m provenance export-backup --database-url sqlite:///./provenance.db
    python -m provenance restore-backup --database-url sqlite:///./restored.db

The database URL always resolves in the same order: the ``--database-url``
flag, then the ``PROVENANCE_DATABASE_URL`` environment variable, then the
``sqlite:///./provenance.db`` default. ``export-backup`` writes the logical
backup object to standard output (terminated by a single newline);
``restore-backup`` reads one backup object from standard input. Both report
a stable machine-readable code on standard error and exit non-zero on any
failure.
"""

from __future__ import annotations

import argparse
import sys

import uvicorn

from provenance import backup
from provenance.app import create_app
from provenance.config import DATABASE_URL_ENV, Settings

_COMMAND_EXPORT = "export-backup"
_COMMAND_RESTORE = "restore-backup"


def _add_database_url(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--database-url",
        default=None,
        help=(
            "SQLAlchemy database URL (e.g. sqlite:///./provenance.db). "
            f"Overrides the {DATABASE_URL_ENV} environment variable."
        ),
    )


def _parse_args(argv: list[str]) -> tuple[str | None, argparse.Namespace]:
    if argv and argv[0] in (_COMMAND_EXPORT, _COMMAND_RESTORE):
        command = argv[0]
        parser = argparse.ArgumentParser(
            prog=f"python -m provenance {command}",
            description=(
                "Write the logical database backup to standard output."
                if command == _COMMAND_EXPORT
                else "Restore a logical database backup read from standard input."
            ),
        )
        _add_database_url(parser)
        return command, parser.parse_args(argv[1:])

    parser = argparse.ArgumentParser(
        prog="python -m provenance",
        description="Run the digital content provenance HTTP service.",
    )
    _add_database_url(parser)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return None, parser.parse_args(argv)


def main() -> None:
    command, args = _parse_args(sys.argv[1:])
    settings = Settings.from_env(database_url=args.database_url)
    if command is not None:
        try:
            if command == _COMMAND_EXPORT:
                sys.stdout.write(backup.export_backup(settings.database_url))
            else:
                backup.restore_backup(settings.database_url, sys.stdin.read())
        except backup.BackupError as exc:
            sys.stderr.write(exc.code + "\n")
            raise SystemExit(1) from exc
        return
    # Passing the app object directly keeps the already-configured engine and
    # avoids importing a settings-bearing module string.
    app = create_app(settings)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
