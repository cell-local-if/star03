"""Logical SQLite backup export and restore for the provenance database.

The backup is a single deterministic JSON object identified by
``backup_version`` 1 and ``schema_version`` 2. Its ``resources`` member
carries every resource family's persisted fields -- actors, contents,
claims, supersessions, evidence bundles, attestations, revocations, access
grants, grant revocations and expiries, authentication key rotations, trust
policies, content relations, audit events, exchange/checkpoint import
receipts, and both asynchronous job families -- in each family's stable
order, plus the ``schema_migrations`` migration ledger. The whole package
carries a SHA-256 digest computed over exactly the normalized ``resources``
member under the existing canonical-JSON rules (sorted keys, compact
separators, unescaped non-ASCII, UTF-8).

Timestamps render as strict RFC 3339 UTC (the same spelling the API
serves), public keys as canonical standard Base64, and digests as
lowercase hexadecimal. The backup never contains private keys,
credentials, raw signatures, claim payloads, or content bytes: those are
never persisted, so no persisted field can carry them.

Export reads an already-migrated database and never writes to it; any
failure to read surfaces as ``database_unavailable`` on standard error
with a non-zero exit and no partial backup on standard output. Restore
validates the envelope (``backup_validation_error``), the format versions
(``backup_version_unsupported``), and the whole-package digest
(``backup_integrity_mismatch``) before touching the target, writes only
into an empty database (``restore_target_not_empty``), is a successful
no-op when the target already contains exactly the same backup, and
performs the full write -- resources, relations, audit, tasks, and the
migration ledger -- in a single transaction that rolls back completely on
any failure.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select, text

from provenance import models  # noqa: F401  register tables on Base.metadata
from provenance.canonical import canonical_json_bytes
from provenance.config import Settings
from provenance.database import Base, init_db, make_engine
from provenance.migrations import SCHEMA_VERSION_2
from provenance.time_utils import parse_rfc3339_utc

#: Backup format version emitted and accepted by this build.
BACKUP_VERSION = 1
#: Schema generation the backup describes; tracks the migration ledger.
BACKUP_SCHEMA_VERSION = SCHEMA_VERSION_2
#: The sole digest algorithm used for the whole-package digest.
BACKUP_DIGEST_ALGORITHM = "sha256"

#: Resources member that carries the migration ledger rows.
LEDGER_FAMILY = "schema_migrations"

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

_TOP_LEVEL_FIELDS = frozenset(
    {"backup_version", "schema_version", "resources", "digest_algorithm", "digest_hex"}
)


class BackupError(Exception):
    """Base class for expected, machine-readable backup CLI failures."""

    code = "backup_error"
    message = "The backup operation failed."

    def __init__(self, message: str | None = None):
        super().__init__(message or self.message)
        if message is not None:
            self.message = message


class BackupValidationError(BackupError):
    """Empty input, invalid JSON, or a missing/duplicate/extra field."""

    code = "backup_validation_error"
    message = "The backup payload failed validation."


class BackupIntegrityMismatchError(BackupError):
    """The carried digest does not match the normalized resources."""

    code = "backup_integrity_mismatch"
    message = "The backup digest does not match the backup contents."


class BackupVersionUnsupportedError(BackupError):
    """A backup/schema version this build does not understand."""

    code = "backup_version_unsupported"
    message = "The backup version is not supported."


class RestoreTargetNotEmptyError(BackupError):
    """The target database holds resource rows and is not the same backup."""

    code = "restore_target_not_empty"
    message = "The restore target database is not empty."


class DatabaseUnavailableError(BackupError):
    """The database cannot be read or written."""

    code = "database_unavailable"
    message = "The database is unavailable."


@dataclass(frozen=True)
class _FamilySpec:
    """One resource family's backup shape.

    ``key`` is both the ``resources`` member name and the persisted table
    name; ``fields`` lists every persisted column with its encoding kind;
    ``order_by`` names the columns of the family's stable order.
    """

    key: str
    fields: tuple[tuple[str, str], ...]
    order_by: tuple[str, ...]


# Field encoding kinds:
#   str / str?        plain (nullable) text
#   int               integer (never a boolean)
#   bool              JSON boolean
#   datetime / datetime?   strict RFC 3339 UTC timestamp (nullable variant)
#   base64_32         canonical standard Base64 of exactly 32 bytes
#   hex64             exactly 64 lowercase hexadecimal characters
#   json / json?      a JSON object (nullable variant)
_FAMILY_SPECS: tuple[_FamilySpec, ...] = (
    _FamilySpec(
        "actors",
        (
            ("id", "str"),
            ("name", "str"),
            ("type", "str"),
            ("created_at", "datetime"),
            ("display_seq", "int"),
        ),
        ("display_seq", "id"),
    ),
    _FamilySpec(
        "contents",
        (
            ("seq", "int"),
            ("id", "str"),
            ("digest_algorithm", "str"),
            ("digest_hex", "hex64"),
            ("media_type", "str"),
            ("title", "str?"),
            ("actor_id", "str"),
            ("created_at", "datetime"),
        ),
        ("seq",),
    ),
    _FamilySpec(
        "claims",
        (
            ("seq", "int"),
            ("id", "str"),
            ("content_id", "str"),
            ("actor_id", "str"),
            ("claim_type", "str"),
            ("payload_digest_algorithm", "str"),
            ("payload_digest_hex", "hex64"),
            ("created_at", "datetime"),
        ),
        ("seq",),
    ),
    _FamilySpec(
        "claim_supersessions",
        (
            ("seq", "int"),
            ("id", "str"),
            ("superseded_claim_id", "str"),
            ("replacement_claim_id", "str"),
            ("reason", "str"),
            ("created_at", "datetime"),
        ),
        ("seq",),
    ),
    _FamilySpec(
        "evidence_bundles",
        (
            ("seq", "int"),
            ("id", "str"),
            ("claim_id", "str"),
            ("evidence_type", "str"),
            ("digest_algorithm", "str"),
            ("digest_hex", "hex64"),
            ("media_type", "str"),
            ("metadata", "json"),
            ("created_at", "datetime"),
        ),
        ("seq",),
    ),
    _FamilySpec(
        "attestations",
        (
            ("seq", "int"),
            ("id", "str"),
            ("target_type", "str"),
            ("target_id", "str"),
            ("signer_actor_id", "str"),
            ("public_key", "base64_32"),
            ("signature_digest_algorithm", "str"),
            ("signature_digest_hex", "hex64"),
            ("created_at", "datetime"),
        ),
        ("seq",),
    ),
    _FamilySpec(
        "attestation_revocations",
        (
            ("seq", "int"),
            ("id", "str"),
            ("attestation_id", "str"),
            ("revoker_actor_id", "str"),
            ("reason", "str"),
            ("created_at", "datetime"),
        ),
        ("seq",),
    ),
    _FamilySpec(
        "attestation_access_grants",
        (
            ("seq", "int"),
            ("id", "str"),
            ("attestation_id", "str"),
            ("grantee_actor_id", "str"),
            ("created_at", "datetime"),
        ),
        ("seq",),
    ),
    _FamilySpec(
        "attestation_access_grant_revocations",
        (
            ("seq", "int"),
            ("id", "str"),
            ("grant_id", "str"),
            ("revoker_actor_id", "str"),
            ("reason", "str"),
            ("created_at", "datetime"),
        ),
        ("seq",),
    ),
    _FamilySpec(
        "attestation_access_grant_expiries",
        (
            ("seq", "int"),
            ("id", "str"),
            ("grant_id", "str"),
            ("expires_at", "datetime"),
            ("created_at", "datetime"),
        ),
        ("seq",),
    ),
    _FamilySpec(
        "authentication_key_rotations",
        (
            ("seq", "int"),
            ("id", "str"),
            ("actor_id", "str"),
            ("public_key", "base64_32"),
            ("active", "bool"),
            ("created_at", "datetime"),
            ("retired_at", "datetime?"),
        ),
        ("seq",),
    ),
    _FamilySpec(
        "actor_trust_policies",
        (
            ("seq", "int"),
            ("id", "str"),
            ("actor_id", "str"),
            ("threshold", "int"),
            ("enabled", "bool"),
            ("created_at", "datetime"),
        ),
        ("seq",),
    ),
    _FamilySpec(
        "content_relations",
        (
            ("seq", "int"),
            ("id", "str"),
            ("content_id", "str"),
            ("parent_content_id", "str"),
            ("relation_type", "str"),
            ("created_at", "datetime"),
        ),
        ("seq",),
    ),
    _FamilySpec(
        "audit_events",
        (
            ("seq", "int"),
            ("event_type", "str"),
            ("resource_id", "str"),
            ("created_at", "datetime"),
        ),
        ("seq",),
    ),
    _FamilySpec(
        "evidence_bundle_exchange_imports",
        (
            ("seq", "int"),
            ("id", "str"),
            ("manifest_version", "str"),
            ("evidence_bundle_id", "str"),
            ("manifest_digest_hex", "hex64"),
            ("created_at", "datetime"),
        ),
        ("seq",),
    ),
    _FamilySpec(
        "audit_checkpoint_imports",
        (
            ("seq", "int"),
            ("id", "str"),
            ("checkpoint_version", "str"),
            ("event_count", "int"),
            ("events_digest_hex", "hex64"),
            ("created_at", "datetime"),
        ),
        ("seq",),
    ),
    _FamilySpec(
        "csp_checkpoint_imports",
        (
            ("seq", "int"),
            ("id", "str"),
            ("checkpoint_version", "str"),
            ("correction_count", "int"),
            ("corrections_digest_hex", "hex64"),
            ("created_at", "datetime"),
        ),
        ("seq",),
    ),
    _FamilySpec(
        "audit_exchange_imports",
        (
            ("seq", "int"),
            ("id", "str"),
            ("signature_version", "str"),
            ("signer_subject", "str"),
            ("public_key", "base64_32"),
            ("package_digest_algorithm", "str"),
            ("package_digest_hex", "hex64"),
            ("signature_digest_algorithm", "str"),
            ("signature_digest_hex", "hex64"),
            ("created_at", "datetime"),
        ),
        ("seq",),
    ),
    _FamilySpec(
        "audit_recon_exchange_imports",
        (
            ("seq", "int"),
            ("id", "str"),
            ("signature_version", "str"),
            ("signer_subject", "str"),
            ("public_key", "base64_32"),
            ("package_digest_algorithm", "str"),
            ("package_digest_hex", "hex64"),
            ("signature_digest_algorithm", "str"),
            ("signature_digest_hex", "hex64"),
            ("created_at", "datetime"),
        ),
        ("seq",),
    ),
    _FamilySpec(
        "revocation_impact_imports",
        (
            ("seq", "int"),
            ("id", "str"),
            ("checkpoint_version", "str"),
            ("impact_count", "int"),
            ("impacts_digest_hex", "hex64"),
            ("created_at", "datetime"),
        ),
        ("seq",),
    ),
    _FamilySpec(
        "impact_recon_exchange_imports",
        (
            ("seq", "int"),
            ("id", "str"),
            ("signature_version", "str"),
            ("signer_subject", "str"),
            ("public_key", "base64_32"),
            ("package_digest_algorithm", "str"),
            ("package_digest_hex", "hex64"),
            ("signature_digest_algorithm", "str"),
            ("signature_digest_hex", "hex64"),
            ("created_at", "datetime"),
        ),
        ("seq",),
    ),
    _FamilySpec(
        "content_export_jobs",
        (
            ("seq", "int"),
            ("id", "str"),
            ("content_id", "str"),
            ("request_id", "str"),
            ("status", "str"),
            ("created_at", "datetime"),
            ("started_at", "datetime?"),
            ("finished_at", "datetime?"),
            ("result", "json?"),
            ("error", "str?"),
        ),
        ("seq",),
    ),
    _FamilySpec(
        "audit_checkpoint_jobs",
        (
            ("seq", "int"),
            ("id", "str"),
            ("request_id", "str"),
            ("event_type", "str?"),
            ("resource_id", "str?"),
            ("from_dt", "datetime?"),
            ("to_dt", "datetime?"),
            ("status", "str"),
            ("created_at", "datetime"),
            ("started_at", "datetime?"),
            ("finished_at", "datetime?"),
            ("result", "json?"),
            ("error", "str?"),
        ),
        ("seq",),
    ),
)

#: The migration ledger family's persisted fields, in backup order.
_LEDGER_FIELDS: tuple[tuple[str, str], ...] = (
    ("version", "int"),
    ("applied_at", "datetime"),
)


# --- Encoding helpers (database -> backup JSON) ---------------------------


def _coerce_stored_datetime(value: Any) -> datetime:
    """Normalize a stored UTC timestamp to a timezone-aware UTC datetime.

    ORM-read columns already arrive as aware UTC datetimes; the migration
    ledger's ``applied_at`` arrives as stored text (either the SQLite
    ``CURRENT_TIMESTAMP`` spelling or a restored RFC 3339 spelling).
    """
    if isinstance(value, datetime):
        candidate = value
    elif isinstance(value, str):
        candidate = parse_rfc3339_utc(value)
        if candidate is None:
            try:
                candidate = datetime.fromisoformat(value)
            except ValueError:
                candidate = None
        if candidate is None:
            raise DatabaseUnavailableError(
                "A stored timestamp cannot be interpreted as UTC."
            )
    else:
        raise DatabaseUnavailableError(
            "A stored timestamp cannot be interpreted as UTC."
        )
    if candidate.tzinfo is None:
        candidate = candidate.replace(tzinfo=timezone.utc)
    return candidate.astimezone(timezone.utc)


def _encode_datetime(value: Any) -> str:
    """Render a stored UTC instant exactly as the API serves it."""
    return _coerce_stored_datetime(value).isoformat()


def _encode_value(kind: str, value: Any) -> Any:
    if value is None:
        return None
    if kind.startswith("datetime"):
        return _encode_datetime(value)
    if kind == "base64_32":
        return base64.b64encode(value).decode("ascii")
    return value


# --- Decoding/validation helpers (backup JSON -> database) ----------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_base64_32(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return False
    return len(raw) == 32 and base64.b64encode(raw).decode("ascii") == value


def _is_rfc3339_utc(value: Any) -> bool:
    return isinstance(value, str) and parse_rfc3339_utc(value) is not None


def _value_matches(kind: str, value: Any) -> bool:
    if kind == "str":
        return isinstance(value, str)
    if kind == "str?":
        return value is None or isinstance(value, str)
    if kind == "int":
        return _is_int(value)
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "datetime":
        return _is_rfc3339_utc(value)
    if kind == "datetime?":
        return value is None or _is_rfc3339_utc(value)
    if kind == "base64_32":
        return _is_base64_32(value)
    if kind == "hex64":
        return isinstance(value, str) and _HEX64_RE.fullmatch(value) is not None
    if kind == "json":
        return isinstance(value, dict)
    if kind == "json?":
        return value is None or isinstance(value, dict)
    raise AssertionError(f"unknown backup field kind: {kind}")


def _decode_value(kind: str, value: Any) -> Any:
    if value is None:
        return None
    if kind.startswith("datetime"):
        parsed = parse_rfc3339_utc(value)
        if parsed is None:  # pragma: no cover - validated before restore
            raise BackupValidationError(
                "Timestamps must be strict RFC 3339 UTC."
            )
        return parsed
    if kind == "base64_32":
        return base64.b64decode(value, validate=True)
    return value


# --- Export ----------------------------------------------------------------


def _export_ledger(conn) -> list[dict[str, Any]]:
    rows = conn.execute(
        text(
            "SELECT version, applied_at FROM schema_migrations"
            " ORDER BY version"
        )
    ).mappings()
    return [
        {
            "version": int(row["version"]),
            "applied_at": _encode_datetime(row["applied_at"]),
        }
        for row in rows
    ]


def build_backup_resources(engine) -> dict[str, Any]:
    """Read every resource family and the migration ledger, normalized.

    Each family's rows appear in its stable order with every persisted
    field encoded under the backup rules (UTC timestamps, Base64 public
    keys, lowercase hex digests). The read is strictly read-only.
    """
    metadata = Base.metadata
    resources: dict[str, Any] = {}
    with engine.connect() as conn:
        for spec in _FAMILY_SPECS:
            table = metadata.tables[spec.key]
            order = [table.c[name] for name in spec.order_by]
            rows = conn.execute(table.select().order_by(*order)).mappings()
            resources[spec.key] = [
                {
                    name: _encode_value(kind, row[name])
                    for name, kind in spec.fields
                }
                for row in rows
            ]
        resources[LEDGER_FAMILY] = _export_ledger(conn)
    return resources


def build_backup_package(resources: dict[str, Any]) -> dict[str, Any]:
    """Wrap normalized resources in the versioned, digested envelope."""
    return {
        "backup_version": BACKUP_VERSION,
        "schema_version": BACKUP_SCHEMA_VERSION,
        "digest_algorithm": BACKUP_DIGEST_ALGORITHM,
        "digest_hex": hashlib.sha256(canonical_json_bytes(resources)).hexdigest(),
        "resources": resources,
    }


def export_backup(engine) -> dict[str, Any]:
    """Build the full backup package for an already-migrated database."""
    return build_backup_package(build_backup_resources(engine))


# --- Restore validation -----------------------------------------------------


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object, rejecting duplicate member names."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BackupValidationError(
                f"The backup contains a duplicate field: {key!r}."
            )
        result[key] = value
    return result


def _validate_envelope(package: dict[str, Any]) -> None:
    keys = set(package)
    if keys != _TOP_LEVEL_FIELDS:
        missing = sorted(_TOP_LEVEL_FIELDS - keys)
        extra = sorted(keys - _TOP_LEVEL_FIELDS)
        parts = []
        if missing:
            parts.append("missing fields: " + ", ".join(missing))
        if extra:
            parts.append("unexpected fields: " + ", ".join(extra))
        raise BackupValidationError(
            "The backup envelope is invalid (" + "; ".join(parts) + ")."
        )
    if not _is_int(package["backup_version"]):
        raise BackupValidationError("backup_version must be an integer.")
    if not _is_int(package["schema_version"]):
        raise BackupValidationError("schema_version must be an integer.")
    if not isinstance(package["resources"], dict):
        raise BackupValidationError("resources must be a JSON object.")
    if package["digest_algorithm"] != BACKUP_DIGEST_ALGORITHM:
        raise BackupValidationError("digest_algorithm must be 'sha256'.")
    digest_hex = package["digest_hex"]
    if not isinstance(digest_hex, str) or _HEX64_RE.fullmatch(digest_hex) is None:
        raise BackupValidationError(
            "digest_hex must be exactly 64 lowercase hexadecimal characters."
        )


def _validate_versions(package: dict[str, Any]) -> None:
    if (
        package["backup_version"] != BACKUP_VERSION
        or package["schema_version"] != BACKUP_SCHEMA_VERSION
    ):
        raise BackupVersionUnsupportedError()


def _validate_row(family: str, fields: tuple[tuple[str, str], ...], row: Any) -> None:
    if not isinstance(row, dict):
        raise BackupValidationError(f"Every {family} row must be a JSON object.")
    expected = {name for name, _ in fields}
    if set(row) != expected:
        raise BackupValidationError(
            f"A {family} row does not carry exactly the persisted fields."
        )
    for name, kind in fields:
        if not _value_matches(kind, row[name]):
            raise BackupValidationError(
                f"A {family} row carries an invalid {name} value."
            )


def _validate_resources(resources: dict[str, Any]) -> None:
    expected = {spec.key for spec in _FAMILY_SPECS} | {LEDGER_FAMILY}
    if set(resources) != expected:
        raise BackupValidationError(
            "resources must carry exactly the known resource families."
        )
    for spec in _FAMILY_SPECS:
        rows = resources[spec.key]
        if not isinstance(rows, list):
            raise BackupValidationError(f"resources.{spec.key} must be an array.")
        for row in rows:
            _validate_row(spec.key, spec.fields, row)
    ledger = resources[LEDGER_FAMILY]
    if not isinstance(ledger, list):
        raise BackupValidationError(f"resources.{LEDGER_FAMILY} must be an array.")
    for row in ledger:
        _validate_row(LEDGER_FAMILY, _LEDGER_FIELDS, row)


def _verify_digest(package: dict[str, Any]) -> None:
    computed = hashlib.sha256(
        canonical_json_bytes(package["resources"])
    ).hexdigest()
    if computed != package["digest_hex"]:
        raise BackupIntegrityMismatchError()


def parse_backup_package(raw: bytes) -> dict[str, Any]:
    """Parse and fully validate one backup object from standard input.

    The checks run in a fixed order -- well-formed single JSON object,
    envelope fields, supported versions, normalized resource structure,
    and finally the whole-package digest -- so every rejection carries its
    stable error code before the target database is touched.
    """
    if not raw or not raw.strip():
        raise BackupValidationError("The backup input is empty.")
    try:
        text_input = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise BackupValidationError(
            "The backup input is not valid UTF-8."
        ) from None
    try:
        package = json.loads(text_input, object_pairs_hook=_unique_object)
    except BackupValidationError:
        raise
    except ValueError:
        raise BackupValidationError(
            "The backup input is not a single JSON object."
        ) from None
    if not isinstance(package, dict):
        raise BackupValidationError("The backup must be a single JSON object.")
    _validate_envelope(package)
    _validate_versions(package)
    _validate_resources(package["resources"])
    _verify_digest(package)
    return package


# --- Restore ----------------------------------------------------------------


def _target_has_rows(engine) -> bool:
    with engine.connect() as conn:
        for spec in _FAMILY_SPECS:
            table = Base.metadata.tables[spec.key]
            count = conn.execute(
                select(func.count()).select_from(table)
            ).scalar_one()
            if count:
                return True
    return False


def _write_all(engine, resources: dict[str, Any]) -> None:
    """Write the whole backup in one transaction.

    Every resource family, the audit trail, the task rows, and the
    migration ledger are inserted with their original identifiers, UTC
    timestamps, stable-order values, and references. Any failure rolls the
    transaction back completely, leaving no half-restored state.
    """
    metadata = Base.metadata
    with engine.begin() as conn:
        # Re-check emptiness inside the write transaction: only an empty
        # database may be written.
        for spec in _FAMILY_SPECS:
            table = metadata.tables[spec.key]
            count = conn.execute(
                select(func.count()).select_from(table)
            ).scalar_one()
            if count:
                raise RestoreTargetNotEmptyError()
        for spec in _FAMILY_SPECS:
            rows = resources[spec.key]
            if not rows:
                continue
            table = metadata.tables[spec.key]
            conn.execute(
                table.insert(),
                [
                    {
                        name: _decode_value(kind, row[name])
                        for name, kind in spec.fields
                    }
                    for row in rows
                ],
            )
        # Replace the migration ledger with the backup's exact records.
        conn.execute(text("DELETE FROM schema_migrations"))
        for row in resources[LEDGER_FAMILY]:
            conn.execute(
                text(
                    "INSERT INTO schema_migrations (version, applied_at)"
                    " VALUES (:version, :applied_at)"
                ),
                {"version": row["version"], "applied_at": row["applied_at"]},
            )


def restore_backup(engine, resources: dict[str, Any]) -> str:
    """Restore validated resources into an empty (or identical) database.

    Returns ``"restored"`` when the backup was written and
    ``"already_present"`` when the target already contains exactly this
    backup (a successful no-op that rewrites nothing). A target holding
    any other resource rows raises :class:`RestoreTargetNotEmptyError`.
    """
    if _target_has_rows(engine):
        current = build_backup_resources(engine)
        if current == resources:
            return "already_present"
        raise RestoreTargetNotEmptyError()
    _write_all(engine, resources)
    return "restored"


# --- CLI entry points --------------------------------------------------------


def _write_error(code: str, message: str) -> None:
    payload = json.dumps(
        {"error": {"code": code, "message": message}},
        separators=(",", ":"),
        ensure_ascii=False,
    )
    sys.stderr.write(payload + "\n")


def export_backup_command(database_url: str | None) -> int:
    """``python -m provenance export-backup`` entry point.

    Writes the backup object to standard output terminated by exactly one
    newline. Any failure to read the database reports
    ``database_unavailable`` on standard error with a non-zero exit and
    produces no partial backup.
    """
    settings = Settings.from_env(database_url=database_url)
    try:
        engine = make_engine(settings.database_url)
        try:
            payload = canonical_json_bytes(export_backup(engine))
        finally:
            engine.dispose()
    except Exception:
        _write_error(
            DatabaseUnavailableError.code, DatabaseUnavailableError.message
        )
        return 1
    # The package is fully assembled before anything is written, so a
    # failed export never leaves half a backup on standard output.
    sys.stdout.buffer.write(payload + b"\n")
    sys.stdout.buffer.flush()
    return 0


def restore_backup_command(database_url: str | None) -> int:
    """``python -m provenance restore-backup`` entry point.

    Reads one JSON backup object from standard input and restores it into
    an empty (or identically populated) database, reporting the stable
    error codes on standard error with a non-zero exit on any failure.
    """
    raw = sys.stdin.buffer.read()
    try:
        package = parse_backup_package(raw)
    except BackupError as exc:
        _write_error(exc.code, exc.message)
        return 1
    settings = Settings.from_env(database_url=database_url)
    try:
        engine = make_engine(settings.database_url)
        try:
            # Bring a brand-new target to the current schema (idempotent on
            # an existing one) before the emptiness check and the restore
            # write.
            init_db(engine)
            restore_backup(engine, package["resources"])
        finally:
            engine.dispose()
    except BackupError as exc:
        _write_error(exc.code, exc.message)
        return 1
    except Exception:
        _write_error(
            DatabaseUnavailableError.code, DatabaseUnavailableError.message
        )
        return 1
    return 0
