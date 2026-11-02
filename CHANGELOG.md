# Changelog

## 0.1.0 - 2026-11-03

First release.

- `Memory` / `ScopedMemory` API with the four Rs: retain, recall, reuse, refine.
- Four memory stores: working, episodic, semantic, procedural.
- `RuleRouter` and `CallableRouter` (plug in a fine-tuned model or LLM, with automatic fallback).
- Hybrid retrieval: vector, BM25, recency, importance, reuse; relevance gate and result dedupe.
- Context Cards with token budgets, markdown and XML rendering.
- `KnowledgeGraph` (entity relations, multi-hop) and `Glossary` (canonical definitions).
- Scope-enforced `SQLiteStore` and `InMemoryStore`, plus a storage contract test suite.
- Refiner: dedupe, decay to archive, promotion of recurring episodes to insights.
- Traces with JSONL export and a router training dataset built from human corrections.
