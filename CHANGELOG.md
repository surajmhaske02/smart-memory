# Changelog

## 0.3.0 - 2026-11-11

### Fixed
- **Non-English text was unsearchable.** The tokenizer only kept the letters a to z, so Hindi and Marathi
  memories embedded to an empty vector and German or Spanish words were cut up ("München" became
  "m" + "nchen"). Words in any script are now kept whole, including Devanagari vowel signs.
- `refine()` could archive **pinned** memories through decay. Pinned memories never decay or expire now.
- Deleting another user's memory id answered "forbidden" (revealing that the id exists). It now answers
  "not found"; read-only global records still raise `PermissionDenied`.
- The embedding cache stored vectors as Python float lists: about 73 MB for 5,000 memories. It now stores
  float32 arrays: about 29 MB.

### Added
- Hindi and Marathi routing cues (failures, events, preferences, rules), Hindi and Marathi negations for the
  dedupe guard, and irregular English verbs ("bought" matches "buy"). Benchmark recall: 93% to 100%.
- Contradiction handling: "Alice moved to Berlin" replaces "Alice lives in Pune", with history kept.
  `Memory(contradictions="supersede" | "flag" | "off")` and `register_slot()` for custom facts.
- Time filters: `search(..., since="7d", until=...)` and the same for `context()`.
- Governance: `Memory(redact=True)` (email, phone, card, Aadhaar, PAN, SSN, IBAN, IP, secrets),
  `RetentionPolicy`, `apply_retention()`, and `export_audit()`.
- `AsyncMemory` (`smart_memory.aio`) for asyncio apps.
- `FastEmbedEmbedder`: small local neural embeddings (ONNX), including multilingual models.
- `smart-memory` command line: `mcp`, `serve`, `add`, `search`, `context`, `stats`, `export`, `audit`.
- MCP server over stdio for Claude Desktop, Claude Code and other MCP clients (tested with the official
  MCP Python SDK). A `context` tool was added to `MemoryTools`.
- REST API (`smart-memory serve`) with bearer-token auth; it refuses public binds without a key.
- `PostgresStore` (psycopg 3) with multi-process cache invalidation, JSONB metadata, full-text search and
  optional pgvector HNSW. `Memory(candidate_limit=...)` lets the database pick search candidates.
- Adapters: LangChain (retriever, chat history, context message), LlamaIndex (retriever), CrewAI (tools).
- Read-write lock: searches no longer block each other; `index_cache_max_memories` caps index RAM.
- Release workflow for PyPI trusted publishing, plus CI jobs for Postgres + pgvector, adapters and packaging.

### Changed
- The built-in embedder is now `hashing-v2`. Databases written by 0.2 are re-embedded automatically on
  first open (local, no API calls). Other embedder changes still require `on_embedder_change="reembed"`.
- Single `add()` throughput is about 6,000/s (was about 7,300/s) because of the contradiction and Unicode
  checks; `add_many()` bulk imports are unaffected.

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
