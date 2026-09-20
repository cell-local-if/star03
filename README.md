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
- `GET /v1/contents/{content_id}/evidence-bundles?evidence_type=...&media_type=...&limit=...&cursor=...` — read-only listing for reviewers of the evidence attached to one content. It returns only bundles linked through claims that directly assert this exact content; it does not traverse lineage to ancestor/descendant contents and never returns raw bytes. Optional `evidence_type` and `media_type` are non-empty, case/whitespace-sensitive exact matches that combine as logical AND; absent means unfiltered. Results follow the bundles' stable creation order (across all of the content's claims) as `{"items", "count", "next_cursor"}` where each item is the existing evidence-bundle public view and `count` is the filtered total, independent of the page. `limit` is an integer from 1 to 100 defaulting to 50; `cursor` is an opaque HMAC-signed server token that binds to the origin and every effective filter/limit, so pages resume without duplication or omission and the final page carries `next_cursor: null`. Blank/illegal/repeated/mismatching parameters and an empty, malformed, tampered, foreign-family, or query-mismatching `cursor` are `422 validation_error` (nothing is silently defaulted); an unknown content id is `404 content_not_found`. The query writes no resource or audit rows.

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

## Trust evaluation for reviewers

Reviewers can ask the service to evaluate how many independent signing subjects have verified a given claim or evidence bundle. The evaluation is purely read-only: it computes its answer from the attestations already stored and creates neither a resource nor an audit event.

- `GET /v1/trust-evaluations?target_type=claim|evidence_bundle&target_id=...&min_signers=...` — `target_type` is required and must be exactly `claim` or `evidence_bundle`; `target_id` is required and non-blank. `min_signers` is optional and must be a plain decimal integer from 1 to 100, defaulting to 1 (no sign, decimal point, or leading/trailing whitespace). Returns exactly `{"target_type", "target_id", "min_signers", "qualified_signer_count", "decision"}`. Only attestations of the exact target with `verified=true` and **no recorded revocation** are counted (every stored attestation is a verified one, but a revoked attestation is retained for history yet no longer qualifies), and the count is the number of **distinct** `signer_actor_id` values: several attestations by the same actor — for example under different keys — qualify once. If a signer's only attestation of the target is revoked, that actor drops out of the count; a target whose attestations are all revoked evaluates like one with no attestations. `decision` is `trusted` when `qualified_signer_count >= min_signers` and `untrusted` otherwise (an existing target with no qualifying attestations evaluates to zero signers and `untrusted`, not to a 404). The result is computed live on every request. Parameters are fully validated before any existence lookup: missing/blank/unknown/repeated `target_type` or `target_id`, a blank identifier, an undeclared parameter, and a non-integer, out-of-range, signed, or whitespace-bearing `min_signers` are all `422 validation_error` — nothing is silently defaulted or normalized. Once the parameters are valid, a missing target is `404 claim_not_found` or `404 evidence_bundle_not_found` according to the declared `target_type` (declaring an existing bundle's id as a claim, or vice versa, is likewise a 404).

## Immutable attestation revocations

A revocation is an immutable, append-only record that an existing attestation is no longer relied upon. It neither updates nor deletes the attestation: the original proof and its `attestation.created` audit relationship are preserved. There is deliberately no update or delete path for a revocation.

- `POST /v1/attestation-revocations` — body contains the existing `attestation_id`, an existing `revoker_actor_id` (which need not be the attestation's signer), and a non-empty `reason` (surrounding whitespace is trimmed). The first creation returns `201` with the stable `rev_` id, the `attestation_id`/`revoker_actor_id` associations, the reason, and a UTC `created_at`. A repeat submission of the same attestation, revoking actor, and reason returns `200` with the existing record and adds no row or audit event; any different combination (a different revoker or a different reason) forms an independent record that is separately retained. Revocations never accept, store, or echo the original signature.
- `GET /v1/attestation-revocations/{revocation_id}` — full public fields, or `404 attestation_revocation_not_found`.
- `GET /v1/attestations/{attestation_id}/revocations` — that attestation's revocation records only, in stable creation order as `{"items", "count"}`; an unknown attestation id is `404 attestation_not_found` (a missing attestation is never an empty collection).

The first revocation write and its `attestation.revoked` audit event commit in a single transaction. An unknown attestation is `404 attestation_not_found` and an unknown revoking actor is `404 unknown_actor`. Blank or whitespace-only `attestation_id`, `revoker_actor_id`, or `reason`, malformed JSON, and any undeclared request field are `422 validation_error` and write neither a record nor an audit event. Because records are append-only, any attempt to update or delete a revocation is met with `405 method_not_allowed`; no such mutation path is provided.

## Read-only proof access grants

A read-only access grant authorizes another actor to read one existing attestation through a protected endpoint. Grants are immutable and append-only: there is no update or delete path, and a grant never mutates the attestation it references.

Both protected routes authenticate every request with three headers:

- `X-PA` — the calling actor id;
- `X-PT` — the request timestamp as an RFC 3339 UTC instant (`Z` or an explicit `+00:00`; no naive or non-UTC value), no more than **300 seconds** from the server's current time;
- `X-PS` — a standard-Base64 Ed25519 signature (64 raw bytes) over the exact UTF-8 compact JSON array (compact separators, non-ASCII emitted unescaped):

  ```
  ["provenance-access-v1",method,path,timestamp,body_sha256]
  ```

  `method` is the uppercase HTTP method, `path` is the request path without any query string, `timestamp` is the exact `X-PT` header value, and `body_sha256` is the lowercase-hex SHA-256 of the actual request body bytes — the digest of zero bytes for an empty (GET) body. The signature is accepted if it verifies under **any** public key of a non-revoked attestation created by the calling actor; a revoked attestation's key never authenticates.

- `POST /v1/attestation-access-grants` — JSON body contains exactly `attestation_id` and `grantee_actor_id`; any undeclared field is rejected. Only the attestation's `signer_actor_id` (authenticated as above) may create a grant. The first creation returns `201` with exactly `{id, attestation_id, grantee_actor_id, created_at}` — a stable `aag_` id and a UTC `created_at`. A retried submission for the same `(attestation_id, grantee_actor_id)` pair returns `200` with the original record and writes no new row or audit event; a different grantee forms an independent, immutable record. The first grant write and its `attestation.access_granted` audit event commit in a single transaction. Missing credentials, an unparseable or out-of-window timestamp, malformed signature encoding, a signature no current key of the actor verifies, malformed fields or JSON, a missing attestation, a missing grantee actor, and a caller who is not the attestation's signer are all `422 validation_error`; no failure writes a resource or an audit event.
- `GET /v1/protected/attestations/{attestation_id}` — read-only. Only the attestation's `signer_actor_id` or an actor holding an access grant for that exact attestation may call it; success returns the existing attestation's public view (the same fields as `GET /v1/attestations/{id}`). A missing target, an unauthenticated caller, and an unauthorized caller all return the **same opaque `404 not_found`**, so existence is never revealed to a caller without access. An unparseable/out-of-window timestamp or a signature that is not canonical Base64 of exactly 64 bytes is `422 validation_error`. The read writes no resource and no audit event.

## Content lineage relations

Relations record how content identities descend from one another: `content_id` is the newer version or derived content, `parent_content_id` is its direct source, and `relation_type` is `version_of` or `derived_from`. Relations are immutable and append-only: there is no update or delete path.

- `POST /v1/content-relations` — create a relation between two existing contents. The first creation returns `201` with the stable `rel_` id, both content ids, the relation type, and a UTC `created_at`; a repeat submission of the same three fields returns `200` with the existing relation and writes no row or audit event. An unknown content on either side is `404 content_not_found`; blank identifiers, an unknown relation type, a self-loop, or an edge that would close a lineage cycle are `422 validation_error` and write nothing.
- `GET /v1/content-relations/{relation_id}` — full public fields, or `404 content_relation_not_found`.
- `GET /v1/contents/{content_id}/relations` — the in- and out-edges of one content in stable creation order as `{"items", "count"}`; an unknown content id is `404 content_not_found`.
- `GET /v1/contents/{content_id}/lineage?direction=ancestors|descendants&max_depth=...&relation_type=...&min_depth=...&limit=...&cursor=...` — read-only multi-hop traversal. `direction` is required and must be exactly `ancestors` (content to direct source, following edges onward) or `descendants` (the reverse); `max_depth` is an optional integer from 1 to 32 defaulting to 8. Optional filters and paging: `relation_type` is exactly `version_of` or `derived_from` (absent means no edge-type filtering); `min_depth` is an integer from 1 to 32 defaulting to 1 and must not exceed the effective `max_depth`; `limit` is an integer from 1 to 100 defaulting to 50; `cursor` is an opaque server token returned by a previous page. Filtering only removes returned items — it never prunes traversal reachability, changes shortest-depth dedup, or changes depth/discovery ordering: an item matches `relation_type` when the edge through which it was *first* reached (at its shortest depth) has that type. Missing/blank/illegal `direction`, an empty/non-integer/out-of-range/repeated `max_depth`, `min_depth`, or `limit`, an unknown `relation_type`, a `min_depth` greater than `max_depth`, and an empty, malformed, tampered, expired-format, or query-parameter-mismatching `cursor` are all `422 validation_error` — nothing is defaulted or silently normalized, and no resource or audit row is written. The origin is never included; only contents reachable within the depth limit are returned, each deduplicated at its shortest depth, ordered by depth ascending and within a level by the discovering edge's stable creation order, as `{"items", "count", "next_cursor"}` where each item is the full public content view plus an integer `depth`. `count` is the total number of items after filtering, independent of the page returned; `next_cursor` is an opaque token for the next page or `null` on the final (and only, for an empty result) page. Re-sending a cursor with the same effective parameters resumes exactly where the previous page ended, with no duplicates or omissions; cursors are HMAC-signed and bind to the origin, direction, and every effective filter/limit value. No reachable contents yields an empty set. An unknown origin is `404 content_not_found`. The query reads only contents and relations (no resource or audit writes), deduplicates visited nodes, and terminates even if anomalous history contains a cycle.

The first relation creation and its `content_relation.created` audit event commit in a single transaction.

## Authentication public-key rotation

A subject can rotate the public key that authenticates its protected-route requests without minting a new attestation. A rotation introduces a new Ed25519 public key that immediately joins the subject's **non-revoked authentication set** — the union of the public keys on the subject's existing non-revoked attestations and the subject's active rotation keys. No new attestation or proof is required, and a public key still carried by an existing non-revoked attestation remains valid. The service never receives a private key or a raw signature. Both routes authenticate with the same `X-PA`/`X-PT`/`X-PS` header contract as the other protected routes, and the authenticated caller is always the rotation's subject.

- `POST /v1/authentication-key-rotations` — JSON body contains exactly a non-empty `actor_id` and a standard-Base64 `new_public_key` that decodes to exactly **32 bytes**; any undeclared field is rejected. The caller must authenticate as that same `actor_id`, which must already exist. The first creation returns `201` with exactly `{id, actor_id, new_public_key, active, created_at, retired_at}` — a stable `akr_` id, the subject, the Base64 public key, `active: true`, a timezone-aware UTC `created_at`, and `retired_at: null`. The first write and its `authentication_key.rotated` audit event commit in a single transaction. A retried submission for the same subject and public key returns `200` with the original record and writes no row or audit event — this stays idempotent even after the record has been retired, in which case the original (now retired) record is returned unchanged. A different public key for the same subject forms an independent rotation.
- `POST /v1/authentication-key-rotations/{rotation_id}/retire` — the request body is **empty** (the signed `body_sha256` is therefore the digest of zero bytes). Only the owning subject may retire a record, and only while it is active. Success returns the record with `active: false` and a timezone-aware UTC `retired_at` (the `akr_` id, subject, and public key are unchanged), and the state change and the `authentication_key.retired` audit event commit in a single transaction. The retired key stops authenticating immediately; the record and its `authentication_key.rotated` history are preserved.

An active rotation key authenticates `POST /v1/attestation-access-grants`, `GET /v1/protected/attestations/{attestation_id}`, and the two rotation routes exactly as an attestation key would; retiring it removes it from the set at once. The request body and the credentials are validated before any state change. An unknown subject, an unknown rotation id, a caller who is not the record's subject, and a repeat retirement of an already-retired record are all `422 validation_error` (existence is never revealed to a non-owner) and write neither a resource nor an audit event; credential failures use the same `validation_error` code as on the grant route.

Errors are distinct JSON bodies under `{"error": {"code", ...}}`: `actor_already_exists` (409), `unknown_actor` (404), `content_not_found` (404), `claim_not_found` (404), `evidence_bundle_not_found` (404), `attestation_not_found` (404), `attestation_revocation_not_found` (404), `content_relation_not_found` (404), `attestation_verification_failed` (422), and `validation_error` (422); the protected attestation read answers a missing target, an unauthenticated caller, and an unauthorized caller with one opaque `not_found` (404). Every successful actor, content, claim, evidence bundle, attestation, attestation revocation, attestation access grant, content relation, and authentication key rotation creation appends one audit row (`event_type`, `resource_id`, UTC `created_at`) in the same transaction as the resource write; a revocation records the `attestation.revoked` event, an access grant records the `attestation.access_granted` event, a rotation records the `authentication_key.rotated` event, and retiring a rotation records the `authentication_key.retired` event. All returned timestamps are timezone-aware UTC.

## Tests

```bash
pip install -e ".[test]"   # or: pip install -r requirements.lock
pytest
```

Tests are deterministic and fully offline (in-memory and temporary-file SQLite, no network).

