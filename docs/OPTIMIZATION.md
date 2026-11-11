# Cost and performance guide

Where agent-memory spend comes from, what smart-memory does about each cost, measured
results, and settings to use in production.

All numbers come from the scripts in [`benchmarks/`](../benchmarks) on an Apple Silicon Mac,
with the default hashing embedder and synthetic data. Re-run them on your own data before
quoting them to anyone.

## 1. Where the money goes

| Cost driver | Scales with | Share for a typical agent |
|---|---|---|
| **LLM input tokens** | memory in the prompt × turns | Usually the largest by far |
| **LLM calls for routing or summarizing** | writes, sessions | Medium. Avoidable for most writes |
| **Embedding API calls** | writes + queries | Small per call, but rate limits and per-request overhead add up |
| **Compute and latency** | memories per user × queries | Hidden cost: slower agents, bigger servers |
| **Storage and noise** | memory growth | Small in dollars, but noise lowers answer quality and raises tokens |

The biggest lever is the prompt. Whatever reaches the model's context window is billed on
every turn, so a memory layer is really a token-control layer.

## 2. What v0.2 changed, and what it measured

### 2.1 Fewer prompt tokens

| Change | How | Result |
|---|---|---|
| Context Cards (v0.1) | Send only memories relevant to this query, within a token budget | 4,271 → ~141 tokens per turn at 300 memories per user; 14,225 → ~146 at 1,000 (‑97% to ‑99%) |
| **Relative cutoff** | `ContextBuilder(relative_cutoff=0.5)` drops results scoring under half the best match. Procedures and rules are exempt. | **‑29% card tokens** (133 → 95) with **no recall loss** over 75 queries |
| **Write-time dedupe** | A near-copy reinforces the existing memory (`occurrences` + 1) instead of adding a new line | One line `Deploy failed (x3)` instead of three |
| **Pinned profile + `render_split()`** | Pinned memories render first, in a fixed order, as a separate stable block | The stable block can be cached by the provider (see 3.2) |

Cutoff sweep (75 queries, 500 memories per user, 300-token budget):

| `relative_cutoff` | Mean card tokens | Answer in card |
|---|---|---|
| 0.0 (v0.1 behaviour) | 133 | 97.3% |
| **0.5 (default)** | **95** | **97.3%** |
| 0.6 | 74 | 97.3% on this data, but it dropped a relevant rule in mixed queries |

### 2.2 Fewer model calls

| Change | How | Result |
|---|---|---|
| **Batched writes** | `add_many()` embeds all texts in **one** call and writes **one** transaction | 20,000 memories: 20,000 → **1** embedding call |
| **Embedding LRU cache** | Built in (`embed_cache_size=4096`). Repeated queries and re-added text are free | 15 → **5** query embedding calls in the benchmark loop (each question asked 3 times) |
| **Router cascade** | `CallableRouter(model, escalate_below=0.6)`: rules decide first; the model only sees uncertain text | **75% of writes** never reach the model on the benchmark data |

### 2.3 Faster search

v0.1 re-read, re-decoded and re-tokenized every memory in scope for every query. v0.2
keeps a warm in-process index per user and agent (vectors, BM25 postings, records). Writes
made through `Memory` patch it in place. Writes from other processes are detected through
the store's generation token (SQLite `PRAGMA data_version`) and trigger a rebuild.

Context-card latency, p50:

| Memories per user | v0.1 | v0.2 pure Python | v0.2 + numpy |
|---|---|---|---|
| 1,000 | 32 ms | 5 ms | **1.0 ms** |
| 5,000 | 175 ms | 20 ms | **4.2 ms** |
| 20,000 | 748 ms | 72 ms | **17 ms** (44× faster) |

Bulk ingest stays around 10,000 memories/s even with dedupe on. The dedupe check only
compares a new memory with stored ones that share its two rarest terms, so it doesn't
slow down as memory grows.

### 2.4 Memory use and concurrency (v0.3)

| | v0.2 | v0.3 |
|---|---|---|
| RAM for 5,000 memories (index + embedding cache) | ~73 MB | **~29 MB** (float32 cache) |
| RAM cap across warm indexes | 64 indexes, any size | `index_cache_max_memories` (default 500,000) |
| 4 threads searching while waiting on a slow embedding API (40 searches) | already parallel (embedding ran outside the lock) | **0.36 s** vs 1.38 s on 1 thread |
| 4 threads, pure CPU ranking | serialized by one lock | no lock contention, but limited by Python's GIL |

For CPU-heavy search traffic, run several worker processes (`PostgresStore` keeps them in
sync) or use free-threaded Python.

For very large users, `Memory(store=PostgresStore(...), candidate_limit=200)` asks pgvector
and Postgres full-text search for candidates instead of holding the whole scope in RAM.

### 2.5 Safer merging

Neural embedders put "Order 1234 failed" and "Order 1235 failed" very close together, and
the same goes for "likes tea" and "doesn't like tea". Dedupe, both on write and in
`refine()`, only merges texts that contain **the same numbers and the same negation**
(`smart_memory.text.same_specifics`). A cost optimization must never corrupt facts.

## 3. Production playbook

### 3.1 Configuration presets

```python
from smart_memory import CallableRouter, ContextBuilder, Memory, RetrievalConfig

# Cost-first: high-volume support or chat agents
mem = Memory(
    "memory.db",
    router=CallableRouter(my_small_classifier, escalate_below=0.6),  # model only for unsure writes
    context_builder=ContextBuilder(relative_cutoff=0.6),              # tighter cards
    dedupe_on_write=0.95,                                              # merge more aggressively
)
card = scoped.context(query, max_tokens=300)

# Quality-first: analyst or advisor agents where a missed fact is expensive
mem = Memory(
    "memory.db",
    embedder=my_neural_embedder,
    context_builder=ContextBuilder(relative_cutoff=0.3),
    retrieval=RetrievalConfig(min_relevance=0.08),
)
card = scoped.context(query, max_tokens=1500, k=25, graph_depth=2)
```

### 3.2 Use the provider's prompt cache

Most LLM providers bill cached input tokens at a fraction of the normal price. The cache
only hits when the start of the prompt is byte-for-byte identical. Lay out the prompt like this:

```python
stable, dynamic = scoped.context(user_message).render_split()
prompt = SYSTEM_PROMPT + "\n\n" + stable      # identical across turns: cacheable
prompt_tail = dynamic                            # changes every turn
```

Put long-lived facts that must always be present (account tier, language, hard rules) in
pinned memories: `scoped.add("...", pin=True)`.

### 3.3 Replace chat history with memory

Re-sending the full transcript every turn makes cost grow quadratically with conversation
length. Keep only the last few turns verbatim (`context(..., recent_turns=4)`), call
`end_session(summarizer=...)` when a thread ends, and let the Context Card bring back what
matters later.

### 3.4 Batch, then refine offline

- Ingest documents, CRM exports and logs with `add_many()`, not a loop of `add()`.
- Run `refine()` from a scheduled job, not on every turn. Its dedupe pass compares every
  pair of memories in scope.
- Install `numpy` (`pip install "smart-memory[fast]"` once published, or just `pip install numpy`).

### 3.5 Watch the counters

```python
mem.cache_info()
# {'index_builds': 3, 'warm_indexes': 3, 'embed_calls': 41, 'embed_hits': 220, 'embed_misses': 980}
```

A high `index_builds` count means another process keeps writing to the same store, or the
index cache is too small (`index_cache_size`). A low `embed_hits` count means queries are
rarely repeated, so the cache can be smaller.

### 3.6 Turn traces into cheaper models

`mem.traces.routing_dataset()` gives labelled `(text, label)` rows, and human
`reclassify()` corrections override the router's labels. Fine-tune a small classifier on
them, plug it into `CallableRouter`, and raise `escalate_below` as it gets better.

## 4. Illustrative monthly cost

Assumptions: 10,000 conversations a day, 8 turns each (80,000 turns a day), 1,000
memories per user, and an example price of $3 per million input tokens. Substitute your
own model's price and traffic.

| Approach | Memory tokens per turn | Tokens per day | Cost per month |
|---|---|---|---|
| Put every memory in the prompt | 14,225 | 1.14 B | ~$102,000 |
| v0.1 Context Card | 146 | 11.7 M | ~$1,050 |
| **v0.2 Context Card (cutoff 0.5)** | **~105** | **8.4 M** | **~$760** |
| v0.2 + cached pinned prefix | lower still, depending on how much is pinned and the provider's cache discount | | |

This counts only the memory part of the prompt, not system prompts, tool definitions or
outputs.

## 5. Known limits and next steps

| Limit | Plan | Good first contribution? |
|---|---|---|
| Vector scoring is a full scan of the scope (fast up to ~50k with numpy) | Optional ANN index (hnswlib or FAISS) behind the same `ScopeIndex` interface | Yes |
| One process holds the warm index | Postgres + pgvector backend for shared, server-side ranking | Yes |
| `refine()` dedupe compares every pair | Reuse the rare-term candidate lookup from write dedupe | Yes |
| Hashing embedder misses paraphrases ("buy" vs "bought") | Ship a small ONNX embedder as an optional extra | Yes |
| Token counts are estimates | Tokenizer adapters for exact `token_counter` | Yes |
| Sync API only | `AsyncMemory` wrapper | Yes |
| Reranking weights are hand-tuned | Learn weights from `retrieve` + `feedback` traces | Research |
