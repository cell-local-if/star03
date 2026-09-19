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

Evidence bundles attach externally verifiable evidence to an existing claim by digest alone. The raw evidence bytes are never accepted, persisted, or logged — only the digest algorithm/value, media type, and a metadata object are stored. Bundles are immutable and append-only: there is no update or delete path.

- `POST /v1/evidence-bundles` — create a bundle with an existing `claim_id`, a non-empty `evidence_type`, `digest_algorithm`, `digest_hex`, a non-empty `media_type`, and a JSON-object `metadata`. Only `digest_algorithm=sha256` with exactly 64 hexadecimal characters is accepted (hex is case-normalized, as for contents). The first creation returns `201` with the stable bundle id, claim association, evidence type, digest algorithm and value, media type, metadata, and UTC `created_at`; a repeat submission with the same claim, evidence type, and digest returns `200` with the existing bundle — its first-submission metadata and media type are retained — and adds no audit event. Any different field combination forms an independent bundle.
- `GET /v1/evidence-bundles/{evidence_bundle_id}` — full public fields, or `404 evidence_bundle_not_found`.
- `GET /v1/claims/{claim_id}/evidence-bundles` — that claim's bundles only, in stable creation order as `{"items", "count"}`; an unknown claim id is `404 claim_not_found` (a missing claim is never an empty collection).

The first bundle creation and its `evidence_bundle.created` audit event commit in a single transaction. Blank text fields, non-object metadata, unsupported algorithms or malformed digests, malformed JSON, and **any undeclared request field** are `422 validation_error`; an unknown claim on creation or listing remains `404 claim_not_found`. In particular, fields that could carry raw evidence (for example `data` or `evidence`) are not silently ignored: they are rejected with `422` and write neither a resource nor an audit event.

## Verifiable attestations

Attestations cryptographically bind an existing signing actor to an existing claim or evidence bundle. They only associate targets that already exist; they never create claims, bundles, or actors. The service verifies Ed25519 signatures but never holds a private key, and only verified attestations are ever written.

- `POST /v1/attestations` — body contains `target_type` (`"claim"` or `"evidence_bundle"`), the existing `target_id`, an existing `signer_actor_id`, a Base64 Ed25519 `public_key` that decodes to exactly **32 bytes**, and a Base64 `signature` that decodes to exactly **64 bytes**. The signature is verified against the exact UTF-8 canonical JSON array (compact separators, non-ASCII emitted unescaped):

  ```
  ["provenance-attestation-v1",target_type,target_id,signer_actor_id]
  ```

  A signature over any other serialization (reordered elements, whitespace, ASCII `\u` escapes, a different prefix/actor) fails verification. The first creation returns `201` with the stable `att_` id, `target_type`/`target_id`, `signer_actor_id`, the Base64 public key, the `sha256` hex digest of the signature, `verified: true`, and a UTC `created_at`. The raw signature is never persisted or echoed — only its SHA-256 digest is stored. A repeat submission with the same target, signing actor, public key, and signature digest returns `200` with the existing attestation and writes no row or audit event.
- `GET /v1/attestations/{attestation_id}` — full public fields, or `404 attestation_not_found`.
- `GET /v1/attestations?target_type=...&target_id=...` — all attestations in stable creation order as `{"items", "count"}`, optionally filtered by `target_type` (`claim` / `evidence_bundle`; any other value is `422 validation_error`) and/or `target_id`. Filtering is an empty collection (`{"items": [], "count": 0}`), never a 404.

The first attestation write and its `attestation.created` audit event commit in a single transaction. An unknown target is `404 claim_not_found` / `evidence_bundle_not_found` (matched by the declared `target_type`), and an unknown signing actor is `404 unknown_actor` even when the signature itself is valid. Bad Base64, wrong decoded lengths (32/64 bytes), blank identifiers, unknown `target_type`, undeclared fields, and malformed JSON are `422 validation_error`; a well-formed request whose signature fails verification is `422 attestation_verification_failed`, with no resource or audit write. Signature verification uses an RFC 8032 Ed25519 implementation in the Python standard library (no external crypto dependency).

## Content lineage relations

Relations record how content identities descend from one another: `content_id` is the newer version or derived content, `parent_content_id` is its direct source, and `relation_type` is `version_of` or `derived_from`. Relations are immutable and append-only: there is no update or delete path.

- `POST /v1/content-relations` — create a relation between two existing contents. The first creation returns `201` with the stable `rel_` id, both content ids, the relation type, and a UTC `created_at`; a repeat submission of the same three fields returns `200` with the existing relation and writes no row or audit event. An unknown content on either side is `404 content_not_found`; blank identifiers, an unknown relation type, a self-loop, or an edge that would close a lineage cycle are `422 validation_error` and write nothing.
- `GET /v1/content-relations/{relation_id}` — full public fields, or `404 content_relation_not_found`.
- `GET /v1/contents/{content_id}/relations` — the in- and out-edges of one content in stable creation order as `{"items", "count"}`; an unknown content id is `404 content_not_found`.
- `GET /v1/contents/{content_id}/lineage` — read-only multi-hop traversal. The required `direction` is exactly `ancestors` or `descendants`: ancestors follow existing edges from each content to its parent, descendants traverse the reverse edges. The origin is never included. `max_depth` is optional and defaults to `8`; when present it must be a plain integer from `1` to `32` (`01`, `+1`, whitespace, floats, and empty values are rejected). Repeating either query parameter is `422 validation_error`; a missing, blank, or illegal `direction`, and an empty, non-integer, or out-of-range `max_depth` are also `422 validation_error` — nothing is defaulted or silently coerced. A missing origin is the existing `404 content_not_found`.

  Success returns `{"items", "count"}`; each item carries every full public content field plus an integer `depth` (1-based hops from the origin). Only contents reachable within the depth bound are returned; a content reached by multiple paths appears once at its shortest depth. Items are ordered by `depth` ascending; within one depth, by the stable creation order (`created_at`, then `seq`) of the relation edge on which each content was first discovered. No reachable contents yields `{"items": [], "count": 0}`. The endpoint only reads contents and relations — it writes no resources and no audit events — and traversal is visited-set bounded, so it terminates even if anomalous history contains a cycle.

The first relation creation and its `content_relation.created` audit event commit in a single transaction.

Errors are distinct JSON bodies under `{"error": {"code", ...}}`: `actor_already_exists` (409), `unknown_actor` (404), `content_not_found` (404), `claim_not_found` (404), `evidence_bundle_not_found` (404), `attestation_not_found` (404), `content_relation_not_found` (404), `attestation_verification_failed` (422), and `validation_error` (422). Every successful actor, content, claim, evidence bundle, attestation, and content relation creation appends one audit row (`event_type`, `resource_id`, UTC `created_at`) in the same transaction as the resource write. All returned timestamps are timezone-aware UTC.

## Tests

```bash
pip install -e ".[test]"   # or: pip install -r requirements.lock
pytest
```

Tests are deterministic and fully offline (in-memory and temporary-file SQLite, no network).

