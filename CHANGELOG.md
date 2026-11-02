# Changelog

## 0.2.0 - 2026-11-03

Cost and speed release. See `docs/OPTIMIZATION.md` for measurements.

- Warm per-scope search index (vectors + BM25 postings), patched in place on writes and rebuilt
  when another process writes. Context latency at 20k memories: 748 ms -> 17 ms (numpy) / 72 ms (pure Python).
- Optional numpy acceleration (`pip install numpy`); falls back to `math.sumprod`.
- `add_many()` embeds a whole batch in one call and writes one transaction.
- Built-in LRU embedding cache (`embed_cache_size`) and `Memory.cache_info()`.
- Write-time dedupe (`dedupe_on_write=0.97`): near-copies reinforce the existing memory and count
  `occurrences`. Never merges texts with different numbers or negation (also applied in `refine()`).
- `CallableRouter(escalate_below=...)` cascade: only uncertain writes call the model.
- Pinned memories (`add(pin=True)`, `pin()`), a Profile section, and `ContextCard.render_split()`
  for provider prompt caching.
- `relative_cutoff` in retrieval and `ContextBuilder` (default 0.5): ~29% fewer card tokens at equal recall.
  Procedural memories are exempt.
- `MemoryTools`: remember/recall/forget function-calling schemas (Anthropic, OpenAI, JSON) and a dispatcher.
- `StorageBackend.generation()` (optional) lets custom backends opt into the warm index.
- Minimum Python is now 3.10 (3.9 is end-of-life and no longer on GitHub's runners). CI covers 3.10-3.14.

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
