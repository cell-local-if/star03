# Architecture direction

The system is expected to evolve as a modular backend whose domain graph separates content artifacts, actors, claims, evidence, attestations, lineage edges, trust decisions, and audit events. Storage, cryptographic verification, policy evaluation, API transport, background work, and observability should remain separable so later tasks can extend one boundary without rewriting unrelated behavior.

This file describes direction rather than implemented behavior. Only code and tests present in a selected commit count as existing capability.
