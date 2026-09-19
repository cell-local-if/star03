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
