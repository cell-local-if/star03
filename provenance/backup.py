"""Logical SQLite backup and restore for the provenance evidence network.

The backup is a single deterministic JSON object identified by
``backup_version`` 1 and ``schema_version`` 2. Its ``resources`` member
carries every resource family's persisted fields -- actors, contents,
claims, supersessions, evidence bundles, attestations, revocations, access
grants, grant revocations, grant expiries, authentication key rotations,
trust policies, content relations, audit events, exchange/checkpoint import
receipts, the asynchronous job state, and the migration ledger -- each row
in its stable persisted order (the surrogate ``seq`` / actor ``display_seq``
order, the ledger in version order). Values are encoded under the existing
rules: UTC ISO-8601 timestamps, standard Base64 for public keys, lowercase
hex for digests, and deterministic canonical JSON for the whole package.

The package carries a SHA-256 ``digest`` computed over the canonicalized
``resources`` member only, so the format and schema versions travel outside
the digest while every persisted byte is integrity-protected. A backup
never contains private keys, credentials, raw signatures, claim payloads,
or content bytes: those are never persisted in the first place, and the
export only ever reads the persisted fields.

Export reads an already-migrated database strictly read-only and either
emits the complete backup or fails with ``database_unavailable`` without
producing a partial package. Restore reads exactly one JSON object,
validates its structure (``backup_validation_error``), its supported
versions (``backup_version_unsupported``), and its digest
(``backup_integrity_mismatch``), then refuses a target that already holds
resource rows (``restore_target_not_empty``) -- unless the target already
contains exactly this backup, which is a success without any rewrite. A
real restore writes every resource, relation, audit, job, and migration
record in a single transaction, preserving the original identifiers, UTC
timestamps, stable order, and references; any failure rolls the whole
restore back, leaving no half-restored state.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select, text

from provenance import models
from provenance.canonical import canonical_json_bytes
from provenance.database import init_db, make_engine, make_session_factory
from provenance.time_utils import parse_rfc3339_utc

#: Backup format generation emitted and accepted by this build.
BACKUP_VERSION = 1
#: Schema generation the backup describes (matches the migration ledger).
SCHEMA_VERSION = 2

#: Digest algorithm of the whole-package integrity digest.
DIGEST_ALGORITHM = "sha256"

#: Stable machine-readable failure codes reported on standard error.
DATABASE_UNAVAILABLE = "database_unavailable"
BACKUP_VALIDATION_ERROR = "backup_validation_error"
BACKUP_INTEGRITY_MISMATCH = "backup_integrity_mismatch"
BACKUP_VERSION_UNSUPPORTED = "backup_version_unsupported"
RESTORE_TARGET_NOT_EMPTY = "restore_target_not_empty"

#: Field kinds used by the per-family specifications below. ``ts`` is a
#: timezone-aware UTC timestamp rendered by ``isoformat``; ``b64_key`` is a
#: 32-byte Ed25519 public key in standard Base64; ``json_obj``/``json_opt``
#: are structured JSON values carried verbatim.
_STR = "str"
_STR_OPT = "str_opt"
_INT = "int"
_BOOL = "bool"
_TS = "ts"
_TS_OPT = "ts_opt"
_B64_KEY = "b64_key"
_JSON_OBJ = "json_obj"
_JSON_OPT = "json_opt"


@dataclass(frozen=True)
class _Family:
    """One resource family: its model, stable order, and persisted fields."""

    name: str
    model: type
    #: Column names giving the family's stable persisted order.
    order_by: tuple[str, ...]
    #: ``(key, kind)`` pairs in the family's backup row objects.
    fields: tuple[tuple[str, str], ...]


#: The ORM attribute backing a backup key when the two differ (the
#: declarative ``metadata`` reserved name is avoided on the model only).
_ATTRIBUTE_OVERRIDES = {("evidence_bundles", "metadata"): "metadata_"}


def _attr(family: str, key: str) -> str:
    return _ATTRIBUTE_OVERRIDES.get((family, key), key)


#: Every resource family, in an insertion order that respects the foreign
#: key graph (actors before contents before claims, and so on). The order
#: of families in the backup object itself is irrelevant: canonicalization
#: sorts object members.
RESOURCE_FAMILIES: tuple[_Family, ...] = (
    _Family(
        "actors",
        models.Actor,
        ("display_seq", "id"),
        (
            ("id", _STR),
            ("name", _STR),
            ("type", _STR),
            ("created_at", _TS),
            ("display_seq", _INT),
        ),
    ),
    _Family(
        "contents",
        models.Content,
        ("seq",),
        (
            ("seq", _INT),
            ("id", _STR),
            ("digest_algorithm", _STR),
            ("digest_hex", _STR),
            ("media_type", _STR),
            ("title", _STR_OPT),
            ("actor_id", _STR),
            ("created_at", _TS),
        ),
    ),
    _Family(
        "claims",
        models.Claim,
        ("seq",),
        (
            ("seq", _INT),
            ("id", _STR),
            ("content_id", _STR),
            ("actor_id", _STR),
            ("claim_type", _STR),
            ("payload_digest_algorithm", _STR),
            ("payload_digest_hex", _STR),
            ("created_at", _TS),
        ),
    ),
    _Family(
        "claim_supersessions",
        models.ClaimSupersession,
        ("seq",),
        (
            ("seq", _INT),
            ("id", _STR),
            ("superseded_claim_id", _STR),
            ("replacement_claim_id", _STR),
            ("reason", _STR),
            ("created_at", _TS),
        ),
    ),
    _Family(
        "evidence_bundles",
        models.EvidenceBundle,
        ("seq",),
        (
            ("seq", _INT),
            ("id", _STR),
            ("claim_id", _STR),
            ("evidence_type", _STR),
            ("digest_algorithm", _STR),
            ("digest_hex", _STR),
            ("media_type", _STR),
            ("metadata", _JSON_OBJ),
            ("created_at", _TS),
        ),
    ),
    _Family(
        "attestations",
        models.Attestation,
        ("seq",),
        (
            ("seq", _INT),
            ("id", _STR),
            ("target_type", _STR),
            ("target_id", _STR),
            ("signer_actor_id", _STR),
            ("public_key", _B64_KEY),
            ("signature_digest_algorithm", _STR),
            ("signature_digest_hex", _STR),
            ("created_at", _TS),
        ),
    ),
    _Family(
        "attestation_revocations",
        models.AttestationRevocation,
        ("seq",),
        (
            ("seq", _INT),
            ("id", _STR),
            ("attestation_id", _STR),
            ("revoker_actor_id", _STR),
            ("reason", _STR),
            ("created_at", _TS),
        ),
    ),
    _Family(
        "attestation_access_grants",
        models.AttestationAccessGrant,
        ("seq",),
        (
            ("seq", _INT),
            ("id", _STR),
            ("attestation_id", _STR),
            ("grantee_actor_id", _STR),
            ("created_at", _TS),
        ),
    ),
    _Family(
        "attestation_access_grant_revocations",
        models.AttestationAccessGrantRevocation,
        ("seq",),
        (
            ("seq", _INT),
            ("id", _STR),
            ("grant_id", _STR),
            ("revoker_actor_id", _STR),
            ("reason", _STR),
            ("created_at", _TS),
        ),
    ),
    _Family(
        "attestation_access_grant_expiries",
        models.AttestationAccessGrantExpiry,
        ("seq",),
        (
            ("seq", _INT),
            ("id", _STR),
            ("grant_id", _STR),
            ("expires_at", _TS),
            ("created_at", _TS),
        ),
    ),
    _Family(
        "authentication_key_rotations",
        models.AuthenticationKeyRotation,
        ("seq",),
        (
            ("seq", _INT),
            ("id", _STR),
            ("actor_id", _STR),
            ("public_key", _B64_KEY),
            ("active", _BOOL),
            ("created_at", _TS),
            ("retired_at", _TS_OPT),
        ),
    ),
    _Family(
        "actor_trust_policies",
        models.ActorTrustPolicy,
        ("seq",),
        (
            ("seq", _INT),
            ("id", _STR),
            ("actor_id", _STR),
            ("threshold", _INT),
            ("enabled", _BOOL),
            ("created_at", _TS),
        ),
    ),
    _Family(
        "content_relations",
        models.ContentRelation,
        ("seq",),
        (
            ("seq", _INT),
            ("id", _STR),
            ("content_id", _STR),
            ("parent_content_id", _STR),
            ("relation_type", _STR),
            ("created_at", _TS),
        ),
    ),
    _Family(
        "audit_events",
        models.AuditEvent,
        ("seq",),
        (
            ("seq", _INT),
            ("event_type", _STR),
            ("resource_id", _STR),
            ("created_at", _TS),
        ),
    ),
    _Family(
        "evidence_bundle_exchange_imports",
        models.ExchangeImportRecord,
        ("seq",),
        (
            ("seq", _INT),
            ("id", _STR),
            ("manifest_version", _STR),
            ("evidence_bundle_id", _STR),
            ("manifest_digest_hex", _STR),
            ("created_at", _TS),
        ),
    ),
    _Family(
        "audit_checkpoint_imports",
        models.CheckpointImportRecord,
        ("seq",),
        (
            ("seq", _INT),
            ("id", _STR),
            ("checkpoint_version", _STR),
            ("event_count", _INT),
            ("events_digest_hex", _STR),
            ("created_at", _TS),
        ),
    ),
    _Family(
        "csp_checkpoint_imports",
        models.CspImportRecord,
        ("seq",),
        (
            ("seq", _INT),
            ("id", _STR),
            ("checkpoint_version", _STR),
            ("correction_count", _INT),
            ("corrections_digest_hex", _STR),
            ("created_at", _TS),
        ),
    ),
    _Family(
        "audit_exchange_imports",
        models.AuditExchangeImportRecord,
        ("seq",),
        (
            ("seq", _INT),
            ("id", _STR),
            ("signature_version", _STR),
            ("signer_subject", _STR),
            ("public_key", _B64_KEY),
            ("package_digest_algorithm", _STR),
            ("package_digest_hex", _STR),
            ("signature_digest_algorithm", _STR),
            ("signature_digest_hex", _STR),
            ("created_at", _TS),
        ),
    ),
    _Family(
        "audit_recon_exchange_imports",
        models.AuditReconExchangeImportRecord,
        ("seq",),
        (
            ("seq", _INT),
            ("id", _STR),
            ("signature_version", _STR),
            ("signer_subject", _STR),
            ("public_key", _B64_KEY),
            ("package_digest_algorithm", _STR),
            ("package_digest_hex", _STR),
            ("signature_digest_algorithm", _STR),
            ("signature_digest_hex", _STR),
            ("created_at", _TS),
        ),
    ),
    _Family(
        "revocation_impact_imports",
        models.ImpactImportRecord,
        ("seq",),
        (
            ("seq", _INT),
            ("id", _STR),
            ("checkpoint_version", _STR),
            ("impact_count", _INT),
            ("impacts_digest_hex", _STR),
            ("created_at", _TS),
        ),
    ),
    _Family(
        "impact_recon_exchange_imports",
        models.ImpactReconExchangeImportRecord,
        ("seq",),
        (
            ("seq", _INT),
            ("id", _STR),
            ("signature_version", _STR),
            ("signer_subject", _STR),
            ("public_key", _B64_KEY),
            ("package_digest_algorithm", _STR),
            ("package_digest_hex", _STR),
            ("signature_digest_algorithm", _STR),
            ("signature_digest_hex", _STR),
            ("created_at", _TS),
        ),
    ),
    _Family(
        "content_export_jobs",
        models.ContentExportJob,
        ("seq",),
        (
            ("seq", _INT),
            ("id", _STR),
            ("content_id", _STR),
            ("request_id", _STR),
            ("status", _STR),
            ("created_at", _TS),
            ("started_at", _TS_OPT),
            ("finished_at", _TS_OPT),
            ("result", _JSON_OPT),
            ("error", _STR_OPT),
        ),
    ),
    _Family(
        "audit_checkpoint_jobs",
        models.AuditCheckpointJob,
        ("seq",),
        (
            ("seq", _INT),
            ("id", _STR),
            ("request_id", _STR),
            ("event_type", _STR_OPT),
            ("resource_id", _STR_OPT),
            ("from_dt", _TS_OPT),
            ("to_dt", _TS_OPT),
            ("status", _STR),
            ("created_at", _TS),
            ("started_at", _TS_OPT),
            ("finished_at", _TS_OPT),
            ("result", _JSON_OPT),
            ("error", _STR_OPT),
        ),
    ),
)

#: The migration ledger family; read and written through raw SQL because it
#: has no ORM model. ``applied_at`` is carried verbatim as stored.
LEDGER_FAMILY = "schema_migrations"
LEDGER_FIELDS = (("version", _INT), ("applied_at", _STR))

_ALL_FAMILIES = tuple(family.name for family in RESOURCE_FAMILIES) + (
    LEDGER_FAMILY,
)


class BackupError(Exception):
    """A backup/restore failure carrying a stable machine-readable code."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


# --- Value encoding (export) and decoding (restore validation) ------------


def _encode_value(kind: str, value: Any) -> Any:
    """Render a persisted Python value under the backup encoding rules."""
    if value is None:
        return None
    if kind in (_TS, _TS_OPT):
        # Aware UTC datetime to its ISO-8601 UTC spelling, as on the wire.
        return value.isoformat()
    if kind == _B64_KEY:
        return base64.b64encode(value).decode("ascii")
    return value


def _decode_value(kind: str, raw: Any) -> Any:
    """Validate and decode one backup field, or raise a validation error."""
    invalid = BackupError(BACKUP_VALIDATION_ERROR)
    if raw is None:
        if kind in (_STR_OPT, _TS_OPT, _JSON_OPT):
            return None
        raise invalid
    if kind in (_STR, _STR_OPT):
        if not isinstance(raw, str):
            raise invalid
        return raw
    if kind == _INT:
        # ``bool`` is an ``int`` subclass; it is never a valid integer field.
        if type(raw) is not int:
            raise invalid
        return raw
    if kind == _BOOL:
        if type(raw) is not bool:
            raise invalid
        return raw
    if kind in (_TS, _TS_OPT):
        if not isinstance(raw, str):
            raise invalid
        parsed = parse_rfc3339_utc(raw)
        if parsed is None:
            raise invalid
        return parsed
    if kind == _B64_KEY:
        if not isinstance(raw, str):
            raise invalid
        try:
            decoded = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError):
            raise invalid from None
        if len(decoded) != 32:
            raise invalid
        return decoded
    if kind == _JSON_OBJ:
        if not isinstance(raw, dict):
            raise invalid
        return raw
    if kind == _JSON_OPT:
        if not isinstance(raw, dict):
            raise invalid
        return raw
    raise invalid  # pragma: no cover - unknown kind is a programming error


# --- Export -----------------------------------------------------------------


def _read_resources(session) -> dict[str, Any]:
    """Read every family's persisted rows in stable order (decoded values)."""
    resources: dict[str, Any] = {}
    for family in RESOURCE_FAMILIES:
        order = [getattr(family.model, column) for column in family.order_by]
        rows = session.execute(
            select(family.model).order_by(*order)
        ).scalars().all()
        resources[family.name] = [
            {
                key: getattr(row, _attr(family.name, key))
                for key, _kind in family.fields
            }
            for row in rows
        ]
    ledger_rows = session.execute(
        text(
            "SELECT version, applied_at FROM schema_migrations "
            "ORDER BY version"
        )
    ).all()
    resources[LEDGER_FAMILY] = [
        {
            "version": int(version),
            "applied_at": (
                applied_at
                if isinstance(applied_at, str)
                else str(applied_at)
            ),
        }
        for version, applied_at in ledger_rows
    ]
    return resources


def _encode_resources(resources: dict[str, Any]) -> dict[str, Any]:
    """Render decoded resources into their JSON-safe backup encoding."""
    encoded: dict[str, Any] = {}
    for family in RESOURCE_FAMILIES:
        encoded[family.name] = [
            {
                key: _encode_value(kind, row[key])
                for key, kind in family.fields
            }
            for row in resources[family.name]
        ]
    encoded[LEDGER_FAMILY] = [
        {key: row[key] for key, _kind in LEDGER_FIELDS}
        for row in resources[LEDGER_FAMILY]
    ]
    return encoded


def _resources_digest_hex(encoded_resources: dict[str, Any]) -> str:
    """Return the SHA-256 hex digest of the canonicalized resources."""
    return hashlib.sha256(canonical_json_bytes(encoded_resources)).hexdigest()


def _build_backup_object(encoded_resources: dict[str, Any]) -> dict[str, Any]:
    return {
        "backup_version": BACKUP_VERSION,
        "schema_version": SCHEMA_VERSION,
        "resources": encoded_resources,
        "digest": {
            "algorithm": DIGEST_ALGORITHM,
            "value_hex": _resources_digest_hex(encoded_resources),
        },
    }


def _sqlite_file_path(database_url: str) -> Path | None:
    """Return the file path of a file-backed SQLite URL, else ``None``."""
    if not database_url.startswith("sqlite:///"):
        return None
    raw_path = database_url.removeprefix("sqlite:///")
    if not raw_path or raw_path == ":memory:":
        return None
    return Path(raw_path)


def export_backup(database_url: str) -> str:
    """Read the migrated database and return the backup object plus newline.

    The read is strictly read-only and the whole package is assembled in
    memory before anything is written, so a failure never produces half a
    backup. A missing, unreadable, or unmigrated database raises
    :class:`BackupError` with the ``database_unavailable`` code.
    """
    path = _sqlite_file_path(database_url)
    if path is not None and not path.exists():
        raise BackupError(DATABASE_UNAVAILABLE)
    try:
        engine = make_engine(database_url)
        try:
            session_factory = make_session_factory(engine)
            session = session_factory()
            try:
                resources = _read_resources(session)
                session.rollback()
            finally:
                session.close()
        finally:
            engine.dispose()
    except BackupError:
        raise
    except Exception:
        raise BackupError(DATABASE_UNAVAILABLE) from None
    backup_object = _build_backup_object(_encode_resources(resources))
    return canonical_json_bytes(backup_object).decode("utf-8") + "\n"


# --- Restore ----------------------------------------------------------------


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict:
    """``object_pairs_hook`` that refuses any duplicated object member."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BackupError(BACKUP_VALIDATION_ERROR)
        result[key] = value
    return result


def _parse_backup_object(raw: str) -> dict[str, Any]:
    """Parse the single input JSON object and check its top-level shape."""
    invalid = BackupError(BACKUP_VALIDATION_ERROR)
    if not raw or not raw.strip():
        raise invalid
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_fields)
    except BackupError:
        raise
    except (json.JSONDecodeError, ValueError):
        raise invalid from None
    if not isinstance(value, dict):
        raise invalid
    if set(value) != {"backup_version", "schema_version", "resources", "digest"}:
        raise invalid
    backup_version = value["backup_version"]
    schema_version = value["schema_version"]
    if type(backup_version) is not int or type(schema_version) is not int:
        raise invalid
    if not isinstance(value["resources"], dict):
        raise invalid
    digest = value["digest"]
    if not isinstance(digest, dict) or set(digest) != {"algorithm", "value_hex"}:
        raise invalid
    if not isinstance(digest["algorithm"], str) or not isinstance(
        digest["value_hex"], str
    ):
        raise invalid
    return value


def _decode_resources(raw_resources: dict[str, Any]) -> dict[str, Any]:
    """Validate the resources structure and decode every field value."""
    invalid = BackupError(BACKUP_VALIDATION_ERROR)
    if set(raw_resources) != set(_ALL_FAMILIES):
        raise invalid
    decoded: dict[str, Any] = {}
    for family in RESOURCE_FAMILIES:
        rows = raw_resources[family.name]
        if not isinstance(rows, list):
            raise invalid
        expected_keys = {key for key, _kind in family.fields}
        decoded_rows = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != expected_keys:
                raise invalid
            decoded_rows.append(
                {
                    key: _decode_value(kind, row[key])
                    for key, kind in family.fields
                }
            )
        decoded[family.name] = decoded_rows
    ledger_rows = raw_resources[LEDGER_FAMILY]
    if not isinstance(ledger_rows, list):
        raise invalid
    expected_ledger_keys = {key for key, _kind in LEDGER_FIELDS}
    decoded_ledger = []
    for row in ledger_rows:
        if not isinstance(row, dict) or set(row) != expected_ledger_keys:
            raise invalid
        decoded_ledger.append(
            {key: _decode_value(kind, row[key]) for key, kind in LEDGER_FIELDS}
        )
    decoded[LEDGER_FAMILY] = decoded_ledger
    return decoded


def _validate_backup(raw: str) -> dict[str, Any]:
    """Parse, structurally validate, and integrity-check the backup object.

    Returns the decoded resources on success. The checks run in a fixed
    order: structure (``backup_validation_error``), version support
    (``backup_version_unsupported``), then the whole-resources digest
    (``backup_integrity_mismatch``).
    """
    value = _parse_backup_object(raw)
    if (
        value["backup_version"] != BACKUP_VERSION
        or value["schema_version"] != SCHEMA_VERSION
    ):
        raise BackupError(BACKUP_VERSION_UNSUPPORTED)
    digest = value["digest"]
    if digest["algorithm"] != DIGEST_ALGORITHM:
        raise BackupError(BACKUP_VALIDATION_ERROR)
    value_hex = digest["value_hex"]
    if (
        len(value_hex) != 64
        or any(character not in "0123456789abcdef" for character in value_hex)
    ):
        raise BackupError(BACKUP_VALIDATION_ERROR)
    decoded = _decode_resources(value["resources"])
    encoded = _encode_resources(decoded)
    if _resources_digest_hex(encoded) != value_hex:
        raise BackupError(BACKUP_INTEGRITY_MISMATCH)
    return decoded


def _has_resource_rows(session) -> bool:
    """True when any resource table (not the migration ledger) has a row."""
    for family in RESOURCE_FAMILIES:
        if session.execute(
            select(family.model).limit(1)
        ).first() is not None:
            return True
    return False


def _write_resources(session, resources: dict[str, Any]) -> None:
    """Insert every family and the ledger inside the session's transaction."""
    for family in RESOURCE_FAMILIES:
        for row in resources[family.name]:
            session.add(
                family.model(
                    **{
                        _attr(family.name, key): row[key]
                        for key, _kind in family.fields
                    }
                )
            )
    # Replace the startup-baselined ledger with the backed-up one so the
    # restored ledger is byte-identical to the source database's.
    session.execute(text("DELETE FROM schema_migrations"))
    session.execute(
        text(
            "INSERT INTO schema_migrations (version, applied_at) "
            "VALUES (:version, :applied_at)"
        ),
        resources[LEDGER_FAMILY],
    )


def restore_backup(database_url: str, raw: str) -> None:
    """Restore one backup object read from standard input.

    The target database is created and migrated on first use exactly like
    service startup. A target that already holds exactly this backup is a
    success without any write; any other non-empty target is refused with
    ``restore_target_not_empty``. The restore itself is a single
    transaction: every resource, relation, audit event, job, and migration
    record is written with its original identifiers, UTC timestamps, stable
    order, and references, and any failure rolls the whole restore back.
    """
    decoded = _validate_backup(raw)
    encoded = _encode_resources(decoded)
    try:
        engine = make_engine(database_url)
        try:
            # Startup migrations are unchanged: a fresh target is created at
            # the latest schema and baselined before any restore write.
            init_db(engine)
            session_factory = make_session_factory(engine)
            session = session_factory()
            try:
                current = _encode_resources(_read_resources(session))
                if canonical_json_bytes(current) == canonical_json_bytes(
                    encoded
                ):
                    # The target already contains exactly this backup:
                    # success without rewriting anything.
                    session.rollback()
                    return
                if _has_resource_rows(session):
                    raise BackupError(RESTORE_TARGET_NOT_EMPTY)
                _write_resources(session, decoded)
                session.commit()
            except BackupError:
                session.rollback()
                raise
            except Exception:
                session.rollback()
                raise BackupError(DATABASE_UNAVAILABLE) from None
            finally:
                session.close()
        finally:
            engine.dispose()
    except BackupError:
        raise
    except Exception:
        raise BackupError(DATABASE_UNAVAILABLE) from None
