# Digital Content Provenance Evidence Network

This repository is the backend for a digital content provenance evidence network. Its long-term purpose is to let systems register digital content, identify its source, attach verifiable claims and evidence, connect revisions and derivatives, and independently verify the resulting provenance graph.

The production roadmap must stay within backend engineering. It should grow through independently testable capabilities covering content identities and digests, source actors, claims, evidence bundles, attestations, signatures, version lineage, derivation edges, verification, revocation, trust policy, authorization, audit history, search, import and export, asynchronous processing, idempotency, concurrency, migrations, privacy, observability, recovery, and interoperability.

## Public engineering contract

- Python 3.12 is the runtime. The initial persistence target is SQLite. New dependencies must be declared and locked.
- The stable command-line entry point is `python -m provenance`. HTTP interfaces, when present, use JSON and expose explicit versioned routes.
- Persisted identifiers are stable. Timestamps use timezone-aware UTC. Hashes, signatures, and canonical encodings declare their algorithms and reject malformed input.
- Mutating operations are transactional and idempotent where clients can retry. Validation errors are distinct from missing resources, conflicts, authorization failures, and internal failures.
- Provenance history is append-oriented. Corrections, supersession, expiry, and revocation preserve the earlier evidence and their audit relationship.
- Tests must be deterministic and run without external network access. Security-sensitive behavior requires negative and boundary coverage.
- Secrets, credentials, private keys, and raw copyrighted content do not enter the repository or logs. Content bytes are referenced by digest and metadata unless a later public requirement explicitly defines controlled storage.
- Every change must preserve documented APIs and stored data unless it includes an explicit compatibility and migration path.

The baseline intentionally contains only this public contract. The first production task must create a complete runnable backend capability that does not already exist here; later tasks must build on reviewed, selected commits without repeating earlier work.

## Running the service

```bash
python -m provenance                      # uses sqlite:///./provenance.db
python -m provenance --database-url sqlite:////var/lib/provenance.db --port 8080
PROVENANCE_DATABASE_URL=sqlite:///./x.db python -m provenance
```

The database location is taken from `--database-url`, then the `PROVENANCE_DATABASE_URL` environment variable, then the default. Tables are created automatically on first startup (parent directories included).

## Initial capability: content identity and source actors

Versioned JSON routes under `/v1`:

- `POST /v1/actors` — register a source actor (`id`, `name`, `type`); returns `201`.
- `POST /v1/contents` — register content identity by digest. Requires an existing actor; validates `digest_algorithm=sha256`, exactly 64 hexadecimal characters for `digest_hex`, and non-empty `media_type`/`actor_id`. The first registration returns `201`; a repeat submission for the same algorithm+digest returns `200` with the existing resource and no new id or audit event. Content bytes are never accepted, stored, or logged — only the digest and metadata.
- `GET /v1/contents/{content_id}` — full public fields, or `404 content_not_found`.
- `GET /v1/contents?actor_id=...` — contents in stable creation order, optionally filtered by source actor.

## Immutable content claims

Claims attach a typed, digest-committed statement by an actor to a registered content identity. Claims are immutable and append-only: there is no update or delete path.

- `POST /v1/claims` — create a claim with `content_id`, `actor_id`, non-empty `claim_type`, and a JSON-object `payload`. Both the content and the actor must already exist; the claiming actor need not be the content's registering actor. The server computes a SHA-256 digest over a deterministic canonical JSON serialization of the payload (sorted keys, minimal separators, UTF-8) and stores only the digest — the raw payload is never persisted or echoed. The first creation returns `201` with the stable claim id, associations, digest algorithm and value, and UTC `created_at`; a repeat submission with the same content, actor, claim type, and canonical payload returns `200` with the existing claim and adds no audit event. Any different field combination forms an independent claim.
- `GET /v1/claims/{claim_id}` — full public fields, or `404 claim_not_found`.
- `GET /v1/contents/{content_id}/claims` — that content's claims only, in stable creation order as `{"items", "count"}`; an unknown content id is `404 content_not_found`.

The first claim creation and its `claim.created` audit event commit in a single transaction. Non-object payloads, blank claim types, and malformed JSON are `422 validation_error`; unknown contents and actors remain `404 content_not_found` / `unknown_actor`.

## Verifiable evidence bundles

Evidence bundles attach verifiable, digest-committed evidence to an existing immutable claim. Raw evidence bytes are never accepted, persisted, or logged — only the digest, media type, and JSON metadata.

- `POST /v1/evidence-bundles` — create a bundle with an existing `claim_id`, non-empty `evidence_type`, `digest_algorithm=sha256`, exactly 64 hexadecimal characters for `digest_hex`, non-empty `media_type`, and a JSON-object `metadata`. The first creation returns `201` with the stable bundle id, claim association, evidence type, digest algorithm and value, media type, metadata, and UTC `created_at`; a repeat submission with the same claim, evidence type, and digest returns `200` with the existing bundle (first metadata preserved) and adds no audit event. Any other field combination forms an independent bundle.
- `GET /v1/evidence-bundles/{evidence_bundle_id}` — full public fields, or `404 evidence_bundle_not_found`.
- `GET /v1/claims/{claim_id}/evidence-bundles` — that claim's bundles only, in stable creation order as `{"items", "count"}`; an unknown claim id is `404 claim_not_found`.

The first bundle creation and its `evidence_bundle.created` audit event commit in a single transaction. Blank text fields, non-object metadata, unsupported digest algorithms, malformed digests, and malformed JSON are `422 validation_error`; unknown claims and bundles remain `404 claim_not_found` / `evidence_bundle_not_found`.

Errors are distinct JSON bodies under `{"error": {"code", ...}}`: `actor_already_exists` (409), `unknown_actor` (404), `content_not_found` (404), `claim_not_found` (404), `evidence_bundle_not_found` (404), and `validation_error` (422). Every successful actor creation and first content creation appends one audit row (`event_type`, `resource_id`, UTC `created_at`) in the same transaction as the resource write. All returned timestamps are timezone-aware UTC.

## Tests

```bash
pip install -e ".[test]"   # or: pip install -r requirements.lock
pytest
```

Tests are deterministic and fully offline (in-memory and temporary-file SQLite, no network).

