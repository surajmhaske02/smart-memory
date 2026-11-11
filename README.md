# smart-memory

**Long-term memory for AI agents that uses up to 99% fewer prompt tokens, runs offline, works in any language, and keeps every user's data separate.**

[![CI](https://github.com/surajmhaske02/smart-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/surajmhaske02/smart-memory/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Dependencies: 0](https://img.shields.io/badge/dependencies-0-brightgreen)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Everyone uses the same frontier models. Agents that stand out remember the right things,
forget the noise, and get better with use. smart-memory gives any agent that memory layer,
whatever model or framework you use, in plain Python with no required dependencies.

Use it as a Python library, as an **MCP server** for Claude Desktop and Claude Code (no code),
as a **REST API**, or through **LangChain, LlamaIndex and CrewAI** adapters.

```python
from smart_memory import Memory

memory = Memory("agent.db").scope(user_id="alice", agent_id="support")
memory.add("Alice prefers email, never phone calls")
prompt = f"You are a support agent.\n\n{memory.context('How should I contact Alice?')}"
```

## Why developers use it

Most agents do one of these, and each one costs you:

| Common approach | What goes wrong | What smart-memory does |
|---|---|---|
| Re-send the whole chat history | Cost grows with every turn; old facts fall out of the window | Sends a **Context Card**: only what this question needs, within a token budget |
| Plain vector RAG over notes | Pulls whatever looks similar, guesses at business terms, repeats near-duplicates | Combines vector, keyword, recency and importance ranking, plus an **entity graph** and a **glossary** of exact definitions |
| One memory store for everything | "Never share PINs" sits next to "user said hi", with no notion of failures or procedures | **Four stores** (working, episodic, semantic, procedural), with routing done for you |
| Access checks in the prompt | One prompt injection away from leaking another customer's data | **Isolation enforced in the database query** for every read and write |
| Static memory | Mistakes repeat; stale facts stay forever | **Learns**: feedback, procedure outcomes, dedupe, decay, and promotion of repeated patterns |
| Old and new facts both stay | "Lives in Pune" and "moved to Berlin" both reach the prompt | **Contradictions resolved**: the newer fact replaces the older one, with history kept |
| English-only text handling | Hindi or Marathi memories can't be found | **Any script**: Hindi, Marathi, German, Spanish, Russian and more |
| Personal data stored as-is | Emails, phone numbers and card numbers sit in the database | **Redaction** before storing, **retention** periods, and an **audit log** |

## Measured results

From [`benchmarks/`](benchmarks) (synthetic data, default offline embedder; see the [cost guide](docs/OPTIMIZATION.md)):

| | Everything in the prompt | smart-memory |
|---|---|---|
| Prompt tokens per turn (1,000 memories per user) | 14,225 | **~105 (‑99%)** |
| Correct memory in the prompt | 100% | **100%** on this benchmark (offline embedder) |
| Context latency at 20,000 memories | n/a | **17 ms** (44× faster than v0.1) |
| Embedding calls to ingest 20,000 memories | 20,000 | **1** |
| Writes that need an LLM router call (cascade) | 100% | **25%** |
| Cross-user leaks | n/a | **0** |

At 80,000 turns a day and $3 per million input tokens, that's about **$102k a month vs $760 a month**
for the memory part of the prompt. The [cost guide](docs/OPTIMIZATION.md) explains the math and the settings.

## Install

Not on PyPI yet. Install from GitHub:

```bash
pip install "git+https://github.com/surajmhaske02/smart-memory.git"
```

No other dependencies and no API keys needed. Optional extras:

| Install | Adds |
|---|---|
| `pip install numpy` | About 4× faster search |
| `pip install fastembed` | Small local neural embedder: matches meaning, works offline, multilingual models available |
| `pip install "psycopg[binary]"` | PostgreSQL + pgvector backend |
| `pip install langchain-core` / `llama-index-core` / `crewai` | Framework adapters |

## Quick start

```python
from smart_memory import Memory

mem = Memory("agent.db")                                    # SQLite file, or Memory() for in-memory
alice = mem.scope(user_id="alice", agent_id="support", thread_id="chat-001")

# Retain: the router picks the right store for each memory
alice.add("Alice prefers email, never phone calls")                    # semantic / preference
alice.add("Always confirm the order number before a refund")           # procedural / instruction
alice.record_event("Refund API timed out for order 4411", outcome="failure")
alice.add("Alice is a premium customer", pin=True)                     # always in context
alice.glossary.define("refund window", "30 days from delivery", aliases=["return window"])
alice.graph.relate("Alice", "customer_of", "Acme Corp")

# Reuse: a compact memory block for your prompt
card = alice.context("Alice wants a refund on order 4411. What's the return window?", max_tokens=400)
print(card)
```

```markdown
## Memory
### Profile
- Alice is a premium customer
### Definitions
- refund window (aka return window): 30 days from delivery
### Procedures & instructions
- Always confirm the order number before a refund
### Facts
- Alice customer of Acme Corp
### Past events
- (just now) [failure] Refund API timed out for order 4411
```

More: [`examples/quickstart.py`](examples/quickstart.py) (a full tour) and
[`examples/agent_loop.py`](examples/agent_loop.py) (a complete agent loop for any LLM).

## Two ways to wire it into an agent

**1. You control memory.** Build a card each turn and save what the user says:

```python
def chat(memory, user_text):
    card = memory.context(user_text, max_tokens=600)
    reply = call_llm(system=f"{SYSTEM_PROMPT}\n\n{card}", user=user_text)
    memory.add(user_text)                              # the router decides where it goes
    memory.feedback(card.memory_ids, helpful=True)     # memories that get used rank higher
    return reply
```

**2. The model controls memory** through function calling (Anthropic, OpenAI, Gemini and others):

```python
from smart_memory import MemoryTools

tools = MemoryTools(memory)                    # remember, recall, context (forget is opt-in)
tool_defs = tools.schemas("anthropic")         # or "openai", or "json"
# ... when the model calls a tool:
result_json = tools.handle(call.name, call.arguments)
```

**3. No code: an MCP server** for Claude Desktop, Claude Code, Cursor and other MCP clients:

```bash
# Claude Code
claude mcp add smart-memory -- smart-memory mcp --db ~/smart-memory.db --user me
```

```json
// Claude Desktop: claude_desktop_config.json
{
  "mcpServers": {
    "smart-memory": {
      "command": "/full/path/to/smart-memory",
      "args": ["mcp", "--db", "/Users/you/smart-memory.db", "--user", "me"]
    }
  }
}
```

The model gets `remember`, `recall` and `context` tools (`--allow-forget` adds `forget`). Each
server process is bound to one user, so the model can't read anyone else's memory. Use
`which smart-memory` to find the full path for Claude Desktop.

**4. Any language or service: the REST API and CLI**

```bash
smart-memory serve --db memory.db --api-key "$SMART_MEMORY_API_KEY"     # http://127.0.0.1:8765

curl -s localhost:8765/v1/context -H "Authorization: Bearer $SMART_MEMORY_API_KEY" \
  -d '{"user_id": "alice", "query": "How should I contact Alice?"}'

smart-memory add     --db memory.db --user alice "Alice prefers email"
smart-memory search  --db memory.db --user alice "contact Alice" --since 7d
smart-memory context --db memory.db --user alice "refund for Alice"
smart-memory audit   --db memory.db audit.jsonl --user alice
```

Endpoints: `POST /v1/memories`, `/v1/memories/batch`, `/v1/search`, `/v1/context`, `/v1/feedback`,
`/v1/forget_user`, `DELETE /v1/memories/<id>`, `GET /health`. The server refuses to listen on a
public address without an API key.

**5. Your framework**

```python
from smart_memory.integrations.langchain import SmartMemoryRetriever, SmartMemoryChatHistory, context_message
from smart_memory.integrations.llamaindex import SmartMemoryLlamaRetriever
from smart_memory.integrations.crewai import memory_tools

retriever = SmartMemoryRetriever(memory=alice, k=5)        # LangChain retriever
nodes = SmartMemoryLlamaRetriever(alice).retrieve("...")   # LlamaIndex retriever
agent = Agent(role="Support", goal="...", backstory="...", tools=memory_tools(alice))  # CrewAI
```

**6. Async code**

```python
from smart_memory.aio import AsyncMemory

mem = AsyncMemory("agent.db")
alice = mem.scope(user_id="alice")
await alice.add("Alice prefers email")
card = await alice.context("How do I contact Alice?")
```

## Core ideas

### Four stores

| Store | Holds | Example | Lifetime |
|---|---|---|---|
| **Working** | Current goal, plan, recent turns | `goal: refund order 4411` | One thread, in process |
| **Episodic** | Events, outcomes, failures | `Refund API timed out (x3)` | Fades unless important or reused |
| **Semantic** | Facts, preferences, definitions, relations | `Alice prefers email` | Long-lived, versioned by `key` |
| **Procedural** | Workflows, rules, tool playbooks | `issue_refund: 1) ... 2) ...` | Long-lived, reinforced by outcomes |

### Four operations

| | Method | What it does |
|---|---|---|
| **Retain** | `add()`, `add_many()` | Route, embed, deduplicate, version and store |
| **Recall** | `search()` | Hybrid ranking inside the caller's scope |
| **Reuse** | `context()` | Build a Context Card within a token budget |
| **Refine** | `feedback()`, `record_outcome()`, `refine()` | Learn from usage, merge duplicates, archive stale items, promote patterns |

### Architecture

```mermaid
flowchart LR
    A[Agent] -->|add| R[Router<br/>rules → model if unsure]
    R --> E[Embedder + LRU cache]
    E --> D{Near duplicate?}
    D -->|yes| X[Reinforce existing]
    D -->|no| S[(Storage<br/>scope enforced in SQL)]
    S <--> I[Warm index<br/>vectors + BM25]
    A -->|context| C[Context builder]
    C --> P[Pinned profile]
    C --> G[Glossary + entity graph]
    C --> I
    C --> W[Working memory]
    C -->|Context Card| A
    A -->|feedback / outcomes| F[Refiner]
    F --> S
    R & I & C & F -.-> T[(Traces)]
    T -.->|fine-tune| R
```

## Guide

<details open>
<summary><b>Scopes and access control</b></summary>

```python
alice_support = mem.scope(user_id="alice", agent_id="support")
alice_sales   = mem.scope(user_id="alice", agent_id="sales")
org           = mem.scope()                     # global: everyone can read, only this scope can write
```

- `user_id` and `agent_id` are hard boundaries, compiled into every storage query.
- `add(..., shared=True)` shares a memory with every agent acting for that user.
- `thread_id` records where a memory came from; it doesn't block access. Use `search(..., thread_only=True)` to stay inside one conversation.
- Changing a record you don't own raises `PermissionDenied`. `forget_user()` deletes everything a scope owns, for example to handle a GDPR erasure request.
</details>

<details>
<summary><b>Retain: storing memories</b></summary>

```python
alice.add("Alice is allergic to peanuts")                  # router picks type, kind and importance
alice.add("Met Alice at the Pune office", "episodic")      # set the type yourself
alice.add("Alice lives in Berlin", key="alice.city")       # versioned: a new value replaces the old one
alice.history("alice.city")                                # every version, newest first
alice.add("Alice is on the Enterprise plan", pin=True)     # always shown first in context
alice.add_many(lines_from_crm)                             # one embedding call, one transaction
alice.add_procedure("issue_refund", ["Verify order", "Check window", "Call refund API"],
                    when="customer asks for money back")
alice.record_event("Pricing API returned 503", outcome="failure")
```

A memory that nearly repeats an existing one doesn't create a new row. It reinforces the
existing one, so `metadata["occurrences"]` goes up and the card shows `(x3)`. Memories
with different numbers ("order 1234" vs "order 1235") or opposite meaning ("likes" vs
"doesn't like") are never merged.
</details>

<details>
<summary><b>Recall: searching</b></summary>

```python
for hit in alice.search("peanut allergy", k=5, memory_types=["semantic"]):
    print(hit.score, hit.content, hit.signals)   # semantic, keyword, recency, importance, reuse
```

A memory has to be actually relevant (`min_relevance`) before importance or recency can
lift it, so important but unrelated memories never reach the prompt.

**Time windows.** `since` and `until` take `"30m"`, `"24h"`, `"7d"`, `"2w"`, a date like
`"2026-11-01"`, a `datetime` or a `timedelta`, and work on `context()` too:

```python
alice.search("deploy failed", since="7d")
alice.context("what went wrong this week?", since="7d")
```

**Any language.** Words in any script are kept whole, so Hindi, Marathi, German, Spanish or
Russian memories are searchable. Hindi and Marathi phrases are also routed correctly
("कल पेमेंट API फेल हो गया" becomes an episodic failure). For questions in one language
matching memories in another, use a multilingual embedder such as
`FastEmbedEmbedder("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")`.
</details>

<details>
<summary><b>Contradictions: newer facts replace older ones</b></summary>

```python
alice.add("Alice lives in Pune")
alice.add("Alice moved to Berlin")          # replaces Pune automatically
alice.search("where does Alice live")       # -> "Alice moved to Berlin"
alice.history("slot:location:alice")        # both versions, newest first
```

Recognised out of the box: where someone lives, employer, email, phone, manager, plan, age
and timezone, in English plus common Hindi and Marathi forms. Add your own with
`smart_memory.contradictions.register_slot(name, regex)`. Choose the behaviour with
`Memory(contradictions="supersede" | "flag" | "off")`. `"flag"` keeps both facts and marks
them with `conflicts_with`.
</details>

<details>
<summary><b>Governance: redaction, retention, audit</b></summary>

```python
from smart_memory import Memory
from smart_memory.governance import Redactor, RetentionPolicy

mem = Memory(
    "agent.db",
    redact=True,   # or Redactor(kinds=["email", "phone"], custom={"EMPLOYEE_ID": r"\bEMP-\d{5}\b"})
    retention=RetentionPolicy({"episodic": 90, "kind:turn": 7}, action="delete"),
)
mem.apply_retention()                                   # e.g. from a nightly job, all users
mem.export_audit("audit.jsonl", user_id="alice", since="30d")
```

Redaction runs before anything is embedded or stored, so raw identifiers never reach the
database, the vectors, the traces or a prompt. Built-in detectors: email, phone (India,
international, US formats), payment card (Luhn-checked), Aadhaar (checksum-verified), PAN,
SSN, IBAN, IP address, and API keys. Pinned memories never expire. `forget_user()` deletes
everything a user owns.
</details>

<details>
<summary><b>Reuse: Context Cards and prompt caching</b></summary>

```python
card = alice.context(user_message, max_tokens=600, k=12, graph_depth=1, recent_turns=0)
card.render()            # markdown
card.render("xml")       # <memory>...</memory>
card.memory_ids          # what was included, for feedback
stable, dynamic = card.render_split()   # pinned profile first: put it in the provider's prompt cache
```

The budget is spent in priority order: profile, current state, definitions, procedures,
facts, past events, then recent conversation. Weak results are trimmed by
`ContextBuilder(relative_cutoff=0.5)`. Rules and procedures are never trimmed. Pass your
model's tokenizer with `ContextBuilder(token_counter=...)` for exact counts.
</details>

<details>
<summary><b>Entity graph and glossary</b></summary>

```python
alice.graph.relate("Acme", "supplier_of", "Globex")
alice.graph.relate("Alice", "reports_to", "Bob", exclusive=True)   # only one value at a time
alice.graph.neighbors("Acme", depth=2)
org.glossary.define("concentration", "Share of exposure held by the top 10 counterparties")
```

When a question mentions a known entity or term, its relations and exact definition go
straight into the card. The model doesn't have to guess what your business means by a word.
</details>

<details>
<summary><b>Refine: memory that improves</b></summary>

```python
alice.feedback(card.memory_ids, helpful=True)       # or a score from -1 to 1
alice.record_outcome("issue_refund", success=False)  # procedure success rate sets its importance
alice.refine(summarizer=my_llm_summarize)            # merge, archive, promote
# RefineReport(examined=120, merged=4, archived=17, promoted=1)
```

`refine()` merges near-duplicates, archives stale and unimportant episodes (they can be
recovered), and with a summarizer turns repeated episodes into a lesson, such as "the
pricing API times out under peak load; retry with backoff".
</details>

<details>
<summary><b>Traces: training data for continual learning</b></summary>

```python
mem.traces.export_jsonl("traces.jsonl")                  # routing, retrieval, feedback, refine events
mem.traces.routing_dataset("router_train.jsonl")         # {"text", "label", "source"}
alice.reclassify(record_id, "episodic")                  # human corrections become training labels
mem.cache_info()                                         # embedding calls saved, index rebuilds
```
</details>

## Use your own models and database

**Router.** Rules are free and deterministic. To use a small fine-tuned model or an LLM,
only for the writes the rules are unsure about:

```python
from smart_memory import CallableRouter
mem = Memory("agent.db", router=CallableRouter(my_classifier, escalate_below=0.6))
```

If the model fails, returns something invalid, or isn't confident, the rules decide, so a
write never fails because of the model.

**Embeddings.** The default embedder runs offline and matches wording. For matching meaning:

```python
from smart_memory import FunctionEmbedder, SentenceTransformerEmbedder
from smart_memory.embeddings import FastEmbedEmbedder

mem = Memory("agent.db", embedder=FastEmbedEmbedder())                # pip install fastembed (offline, ONNX)
mem = Memory("agent.db", embedder=SentenceTransformerEmbedder())     # pip install sentence-transformers
mem = Memory("agent.db", embedder=FunctionEmbedder(my_embed_fn, dim=1536, name="my-model"))
```

With `FastEmbedEmbedder`, "what did Priya buy?" finds "Priya purchased a new laptop", which a
word-matching embedder can't do.

The store records which embedder built it. Switching to a different one raises an error
unless you pass `on_embedder_change="reembed"`.

**Storage.** SQLite (default), in-memory and PostgreSQL are built in.

```python
from smart_memory.storage.postgres import PostgresStore   # pip install "psycopg[binary]"

store = PostgresStore("postgresql://user:pass@host/db")
mem = Memory(store=store)                         # many processes share one database
mem = Memory(store=store, candidate_limit=200)    # very large scopes: pgvector + full-text pick candidates
```

Every process notices the others' writes. With the pgvector extension, embeddings also get
an HNSW index. `candidate_limit` lets the database pre-select candidates for each search
instead of loading the whole scope into RAM.

For Oracle, MongoDB or Redis, subclass `StorageBackend` (about ten methods), apply `Scope`
checks in your queries, and run [`tests/test_storage_contract.py`](tests/test_storage_contract.py)
against it. Those tests check user isolation for you.

## Production checklist

- Use `add_many()` for bulk imports and run `refine()` from a scheduled job.
- `pip install numpy` for faster search.
- Pin always-needed facts and send `render_split()`'s stable part as a cached prefix.
- Keep 2 to 4 recent turns and let memory cover the rest of the history.
- Check `mem.cache_info()` and the routing traces from time to time.
- Turn on `redact=True` and a `RetentionPolicy` if users may share personal data.
- Several processes or servers: use `PostgresStore`. Very large users: add `candidate_limit`.

The [cost and performance guide](docs/OPTIMIZATION.md) has presets, measurements and the cost math.

## Limits

- With the in-process index, ranking scans every memory in the scope: fine up to roughly 50k memories per user and agent with numpy. Beyond that, use `PostgresStore` with `candidate_limit`.
- Searches don't block each other, but CPU-bound ranking in pure Python is limited by Python's GIL (global interpreter lock). For CPU parallelism, run several worker processes or use free-threaded Python. Waiting on an embedding API does run in parallel.
- The warm index lives in each process. Writes from other processes are detected and trigger a rebuild, but each process keeps its own copy.
- The default embedder matches words (with irregular verbs such as "bought" → "buy"), not meaning. Use `FastEmbedEmbedder` or another neural embedder when paraphrases matter.
- Contradiction detection covers common single-valued facts, not every possible conflict.
- Working memory is per process; persist it with `end_session()`.

## Contributing

This project is new, so a contribution here makes a visible difference. Each area below is
self-contained, with a clear interface and existing tests to follow:

| Area | Idea | Where |
|---|---|---|
| Storage | MongoDB, Redis or Oracle backend | `smart_memory/storage/`, contract tests included |
| Speed | Optional in-process ANN index (hnswlib or FAISS) | `smart_memory/index.py` |
| Quality | Learned ranking weights from traces; more contradiction slots and languages | `retrieval.py`, `contradictions.py`, `router.py` |
| Integrations | AutoGen, Haystack and Semantic Kernel adapters; a CrewAI storage backend | `smart_memory/integrations/` |
| Evaluation | Real-world memory benchmarks | `benchmarks/` |

```bash
git clone https://github.com/surajmhaske02/smart-memory.git && cd smart-memory
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" numpy
pytest            # 220+ tests in a few seconds (Postgres and framework tests run when those are installed)
ruff check smart_memory tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and discussions are welcome, including "I tried it and X was confusing".

## Credits

The design follows the agent-memory ideas in Richmond Alake's AI Engineer talk
*"Everyone Has the Same AI Models. Here's What Sets Agents Apart"*: four memory types, the
four operations, Context Cards, ontology and semantic layers, data-layer access control,
and memory traces for continual learning. This is an independent implementation, not
affiliated with the talk, its author, or any vendor.

## License

[MIT](LICENSE). Use it in commercial products, fork it, and build on it.
