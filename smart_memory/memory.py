"""The public API: ``Memory`` (the engine) and ``ScopedMemory`` (a user/agent/thread view of it).

The four operations every agent memory needs:

* **Retain** - :meth:`ScopedMemory.add` routes and stores new information.
* **Recall** - :meth:`ScopedMemory.search` finds relevant memories with hybrid ranking.
* **Reuse** - :meth:`ScopedMemory.context` compiles a compact Context Card for the prompt.
* **Refine** - :meth:`ScopedMemory.feedback` and :meth:`ScopedMemory.refine` improve memory over time.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .context import ContextBuilder, ContextCard
from .embeddings import Embedder, HashingEmbedder
from .ontology import Glossary, KnowledgeGraph, norm
from .refine import Refiner, RefineReport, Summarizer
from .retrieval import HybridRetriever, RetrievalConfig
from .router import Router, RuleRouter
from .storage.base import StorageBackend
from .storage.sqlite import SQLiteStore
from .traces import TraceLog
from .types import DURABLE_TYPES, MemoryRecord, MemoryType, RoutingDecision, Scope, SearchResult, Status, clamp
from .working import WorkingMemory

_DEFAULT_KIND = {MemoryType.EPISODIC: "event", MemoryType.SEMANTIC: "fact", MemoryType.PROCEDURAL: "instruction"}


class EmbedderMismatchError(RuntimeError):
    """The store was built with a different embedder. Vectors from two models are not comparable."""


class Memory:
    """The memory engine. Create one per process and hand out scoped views.

    >>> mem = Memory("agent.db")                       # or Memory() for in-memory
    >>> alice = mem.scope(user_id="alice", agent_id="support")
    >>> alice.add("Alice prefers email over phone calls")
    >>> print(alice.context("How should I contact Alice?"))

    Args:
        path: SQLite file path, or ``":memory:"``. Ignored when ``store`` is given.
        store: Any :class:`StorageBackend`.
        embedder: Any object with ``name``, ``dim``, ``embed(texts)``. Defaults to :class:`HashingEmbedder`.
        router: Decides which store new text belongs to. Defaults to :class:`RuleRouter`.
        retrieval: Ranking weights and thresholds.
        on_embedder_change: ``"error"`` (default) or ``"reembed"`` when the store was built with another embedder.
        trace: Record routing/retrieval/feedback traces for auditing and continual learning.
        clock: Time source; inject a fake clock in tests.
    """

    def __init__(self, path: Union[str, Path] = ":memory:", *, store: Optional[StorageBackend] = None,
                 embedder: Optional[Embedder] = None, router: Optional[Router] = None,
                 retrieval: Optional[RetrievalConfig] = None, context_builder: Optional[ContextBuilder] = None,
                 refiner: Optional[Refiner] = None, on_embedder_change: str = "error", trace: bool = True,
                 clock: Callable[[], float] = time.time, working_max_turns: int = 20) -> None:
        self.store = store if store is not None else SQLiteStore(str(path))
        self.embedder: Embedder = embedder or HashingEmbedder()
        self.router: Router = router or RuleRouter()
        self.retriever = HybridRetriever(retrieval)
        self.context_builder = context_builder or ContextBuilder()
        self.refiner = refiner or Refiner()
        self.clock = clock
        self.traces = TraceLog(self.store, enabled=trace, clock=clock)
        self._working: Dict[Tuple[Optional[str], Optional[str], Optional[str]], WorkingMemory] = {}
        self._working_max_turns = working_max_turns
        self._write_lock = threading.RLock()
        self._check_embedder(on_embedder_change)

    # --- setup --------------------------------------------------------------------
    @property
    def embedder_signature(self) -> str:
        return f"{self.embedder.name}:{self.embedder.dim}"

    def _check_embedder(self, on_change: str) -> None:
        if on_change not in ("error", "reembed"):
            raise ValueError("on_embedder_change must be 'error' or 'reembed'")
        stored = self.store.get_meta("embedder")
        if stored is None:
            self.store.set_meta("embedder", self.embedder_signature)
        elif stored != self.embedder_signature:
            if on_change == "error":
                raise EmbedderMismatchError(
                    f"Store was embedded with {stored!r} but this Memory uses {self.embedder_signature!r}. "
                    "Pass on_embedder_change='reembed' to re-embed every memory with the new model."
                )
            self.reembed()

    def scope(self, user_id: Optional[str] = None, agent_id: Optional[str] = None,
              thread_id: Optional[str] = None) -> ScopedMemory:
        """A view limited to one user/agent (and optionally a conversation thread).

        ``mem.scope()`` with no ids is the *global* scope: knowledge every user can read
        (e.g. a company glossary) but only the global scope can modify.
        """
        return ScopedMemory(self, Scope(user_id, agent_id, thread_id))

    # --- maintenance --------------------------------------------------------------
    def reembed(self, batch_size: int = 64) -> int:
        """Recompute every embedding with the current embedder. Returns the number of records."""
        records = list(self.store.iter_all())
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            vectors = self.embedder.embed([_index_text(r.content, r.metadata) for r in batch])
            for record, vec in zip(batch, vectors):
                record.embedding = vec
                self.store.update(record, record.scope)
        self.store.set_meta("embedder", self.embedder_signature)
        return len(records)

    def stats(self) -> Dict[str, Any]:
        by_type: Counter = Counter()
        by_status: Counter = Counter()
        users = set()
        for r in self.store.iter_all():
            by_status[r.status.value] += 1
            if r.status == Status.ACTIVE:
                by_type[r.memory_type.value] += 1
            users.add(r.user_id)
        return {"active_by_type": dict(by_type), "by_status": dict(by_status), "users": len(users),
                "embedder": self.embedder_signature}

    def export_jsonl(self, path: Union[str, Path], include_embeddings: bool = False) -> int:
        """Dump every memory (all scopes, all statuses) for backup or migration."""
        count = 0
        with open(path, "w", encoding="utf-8") as fh:
            for record in self.store.iter_all():
                fh.write(json.dumps(record.to_dict(include_embeddings), default=str) + "\n")
                count += 1
        return count

    def _working_for(self, scope: Scope) -> WorkingMemory:
        key = (scope.user_id, scope.agent_id, scope.thread_id)
        with self._write_lock:
            if key not in self._working:
                self._working[key] = WorkingMemory(self._working_max_turns)
            return self._working[key]

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _index_text(content: str, metadata: Mapping[str, Any]) -> str:
    extra = metadata.get("search_text")
    return f"{content} {extra}" if extra else content


IdsOrResults = Union[str, SearchResult, MemoryRecord, Iterable[Union[str, SearchResult, MemoryRecord]]]


def _ids(items: IdsOrResults) -> List[str]:
    if isinstance(items, (str, SearchResult, MemoryRecord)):
        items = [items]
    return [i if isinstance(i, str) else i.id for i in items]


class ScopedMemory:
    """Everything an agent does with memory, limited to one :class:`Scope`."""

    def __init__(self, mem: Memory, scope: Scope) -> None:
        self._mem = mem
        self.scope = scope
        self._store = mem.store
        self._traces = mem.traces
        self.graph = KnowledgeGraph(self)
        self.glossary = Glossary(self)

    def __repr__(self) -> str:
        return f"ScopedMemory({self.scope})"

    @property
    def working(self) -> WorkingMemory:
        return self._mem._working_for(self.scope)

    def with_thread(self, thread_id: Optional[str]) -> ScopedMemory:
        return ScopedMemory(self._mem, self.scope.with_thread(thread_id))

    def _now(self) -> float:
        return self._mem.clock()

    def _query(self, memory_types: Optional[Sequence[MemoryType]] = None, **filters: Any) -> List[MemoryRecord]:
        return self._store.query(self.scope, memory_types=list(memory_types or DURABLE_TYPES), **filters)

    # --- Retain -------------------------------------------------------------------
    def add(self, content: str, memory_type: Optional[Union[MemoryType, str]] = None, *,
            kind: Optional[str] = None, importance: Optional[float] = None,
            metadata: Optional[Dict[str, Any]] = None, key: Optional[str] = None,
            shared: bool = False) -> MemoryRecord:
        """Store a memory. Without ``memory_type`` the router decides.

        Args:
            key: A stable identity such as ``"alice.diet"``. Adding again with the same key
                supersedes the old version (same content, ignoring case, is a no-op), so facts stay current.
            shared: Store at user level (``agent_id=None``) so every agent of this user sees it.
        """
        content = content.strip()
        if not content:
            raise ValueError("Memory content must be non-empty")
        metadata = dict(metadata or {})
        decision = self._mem.router.route(content, metadata)
        if memory_type is not None:
            forced = MemoryType.coerce(memory_type)
            if forced != decision.memory_type:
                decision = RoutingDecision(forced, kind=_DEFAULT_KIND.get(forced, "note"), importance=0.5,
                                           reason="explicit")
            else:
                decision.router, decision.reason, decision.confidence = "manual", "explicit", 1.0
        if decision.memory_type == MemoryType.WORKING:
            raise ValueError("Working memory is not persisted; use scoped.working.set()/add_turn()")
        if kind:
            decision.kind = kind
        if importance is not None:
            decision.importance = clamp(importance)
        if self.scope.thread_id and decision.memory_type != MemoryType.EPISODIC:
            metadata.setdefault("source_thread", self.scope.thread_id)

        now = self._now()
        record = MemoryRecord(
            content=content, memory_type=decision.memory_type, kind=decision.kind, key=key,
            user_id=self.scope.user_id, agent_id=None if shared else self.scope.agent_id,
            thread_id=self.scope.thread_id if decision.memory_type == MemoryType.EPISODIC else None,
            metadata=metadata, importance=decision.importance, created_at=now,
            embedding=self._mem.embedder.embed([_index_text(content, metadata)])[0],
        )
        with self._mem._write_lock:
            if key is not None:
                for existing in self._store.query(self.scope, key=key, owned_only=True):
                    if norm(existing.content) == norm(content) and existing.memory_type == record.memory_type:
                        if existing.content == content:
                            existing.metadata.update(metadata)
                        else:  # case-only difference: keep the original spelling everywhere
                            for k, v in metadata.items():
                                existing.metadata.setdefault(k, v)
                        existing.importance = max(existing.importance, record.importance)
                        existing.updated_at = now
                        self._store.update(existing, self.scope)
                        return existing
                    existing.status, existing.superseded_by, existing.updated_at = Status.SUPERSEDED, record.id, now
                    self._store.update(existing, self.scope)
            self._store.add(record)
        self._traces.log("route", self.scope, memory_id=record.id, text=content[:2000],
                         memory_type=record.memory_type.value, kind=record.kind, confidence=decision.confidence,
                         router=decision.router, reason=decision.reason)
        return record

    retain = add

    def add_many(self, contents: Iterable[str], memory_type: Optional[Union[MemoryType, str]] = None,
                 **kwargs: Any) -> List[MemoryRecord]:
        return [self.add(c, memory_type, **kwargs) for c in contents]

    def record_event(self, content: str, *, outcome: Optional[str] = None, importance: Optional[float] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> MemoryRecord:
        """Store something that happened. ``outcome="failure"`` marks it so the agent avoids repeating it."""
        metadata = dict(metadata or {})
        if outcome:
            metadata["outcome"] = outcome
        failed = outcome in ("failure", "failed", "error")
        return self.add(content, MemoryType.EPISODIC, kind="failure" if failed else "event",
                        importance=importance if importance is not None else (0.75 if failed else 0.45),
                        metadata=metadata)

    def add_procedure(self, name: str, steps: Sequence[str], *, description: str = "", when: Optional[str] = None,
                      importance: float = 0.7, shared: bool = False) -> MemoryRecord:
        """Store (or replace, by name) a reusable workflow / tool playbook."""
        steps = [s.strip() for s in steps if s and s.strip()]
        if not name.strip() or not steps:
            raise ValueError("A procedure needs a name and at least one step")
        content = f"{name}: {description}".strip().rstrip(":") + " | " + " -> ".join(steps)
        return self.add(content, MemoryType.PROCEDURAL, kind="procedure", key=f"proc:{norm(name)}",
                        importance=importance, shared=shared,
                        metadata={"name": name, "steps": steps, "when": when, "search_text": when or ""})

    def procedure(self, name: str) -> Optional[MemoryRecord]:
        found = self._store.query(self.scope, key=f"proc:{norm(name)}", limit=1)
        return found[0] if found else None

    def record_outcome(self, name: str, success: bool) -> MemoryRecord:
        """Reinforce or weaken a procedure from real outcomes; success rate feeds its importance."""
        with self._mem._write_lock:
            record = self.procedure(name)
            if record is None:
                raise KeyError(f"No procedure named {name!r}")
            stats = record.metadata.setdefault("outcomes", {"success": 0, "failure": 0})
            stats["success" if success else "failure"] += 1
            total = stats["success"] + stats["failure"]
            rate = (stats["success"] + 1) / (total + 2)  # Laplace smoothing
            record.importance = clamp(0.3 + 0.7 * rate)
            record.updated_at = self._now()
            self._store.update(record, self.scope)
        self._traces.log("outcome", self.scope, memory_id=record.id, procedure=name, success=success)
        return record

    # --- Recall -------------------------------------------------------------------
    def search(self, query: str, k: int = 5, *, memory_types: Optional[Sequence[Union[MemoryType, str]]] = None,
               kinds: Optional[Sequence[str]] = None, thread_only: bool = False,
               min_relevance: Optional[float] = None) -> List[SearchResult]:
        """Most relevant memories for ``query``, ranked by relevance, recency, importance and reuse."""
        return self._search(query, k=k, memory_types=memory_types, kinds=kinds, thread_only=thread_only,
                            min_relevance=min_relevance, touch=True)

    recall = search

    def _search(self, query: str, *, k: int, memory_types: Optional[Sequence[Union[MemoryType, str]]] = None,
                kinds: Optional[Sequence[str]] = None, thread_only: bool = False,
                min_relevance: Optional[float] = None, touch: bool = True) -> List[SearchResult]:
        if thread_only and not self.scope.thread_id:
            raise ValueError("thread_only=True needs a scope with a thread_id")
        types = [MemoryType.coerce(t) for t in memory_types] if memory_types else list(DURABLE_TYPES)
        candidates = self._query(types, kinds=kinds, thread_id=self.scope.thread_id if thread_only else None)
        now = self._now()
        query_vec = self._mem.embedder.embed([query])[0] if candidates else None
        results = self._mem.retriever.rank(query, query_vec, candidates, now, top_k=k, min_relevance=min_relevance)
        if touch:
            self._store.touch([r.id for r in results], now)
            self._traces.log("retrieve", self.scope, query=query, k=k,
                             results=[{"id": r.id, "score": round(r.score, 4)} for r in results])
        return results

    # --- Reuse --------------------------------------------------------------------
    def context(self, query: str, max_tokens: int = 800, **kwargs: Any) -> ContextCard:
        """A compact Context Card for ``query``: put ``str(card)`` in your system prompt.

        Keyword args go to :meth:`ContextBuilder.build` (``k``, ``graph_depth``, ``recent_turns``,
        ``include_working``, ``memory_types``, ``min_relevance``).
        """
        return self._mem.context_builder.build(self, query, max_tokens=max_tokens, **kwargs)

    reuse = context

    # --- Refine -------------------------------------------------------------------
    def feedback(self, items: IdsOrResults, helpful: Union[bool, float] = True) -> int:
        """Tell memory whether recalled items helped. ``helpful`` may be a bool or a -1..1 score.

        Returns the number of records updated (records this scope can't modify are only traced).
        """
        delta = (0.1 if helpful else -0.15) if isinstance(helpful, bool) else 0.15 * max(-1.0, min(1.0, helpful))
        ids = _ids(items)
        updated = 0
        with self._mem._write_lock:
            for rid in ids:
                record = self._store.get(rid, self.scope)
                if record is None or not self.scope.can_write(record):
                    continue
                record.importance = clamp(record.importance + delta)
                self._store.update(record, self.scope)
                updated += 1
        self._traces.log("feedback", self.scope, memory_ids=ids, helpful=helpful, delta=delta)
        return updated

    def refine(self, summarizer: Optional[Summarizer] = None) -> RefineReport:
        """Dedupe, decay and (with a summarizer such as an LLM call) promote episodes to insights."""
        return self._mem.refiner.run(self, summarizer)

    def reclassify(self, record_id: str, memory_type: Union[MemoryType, str],
                   kind: Optional[str] = None) -> MemoryRecord:
        """Fix a routing mistake. Logged as a ``route_correction`` trace: free labels for router training."""
        record = self._require(record_id)
        old = record.memory_type
        record.memory_type = MemoryType.coerce(memory_type)
        if record.memory_type == MemoryType.WORKING:
            raise ValueError("Cannot reclassify into working memory")
        record.kind = kind or _DEFAULT_KIND.get(record.memory_type, record.kind)
        record.updated_at = self._now()
        self._store.update(record, self.scope)
        self._traces.log("route_correction", self.scope, memory_id=record_id, old_type=old.value,
                         new_type=record.memory_type.value)
        return record

    # --- CRUD ---------------------------------------------------------------------
    def get(self, record_id: str) -> Optional[MemoryRecord]:
        return self._store.get(record_id, self.scope)

    def _require(self, record_id: str) -> MemoryRecord:
        record = self.get(record_id)
        if record is None:
            raise KeyError(f"No memory {record_id!r} visible to {self.scope}")
        return record

    def update(self, record_id: str, *, content: Optional[str] = None, importance: Optional[float] = None,
               metadata: Optional[Dict[str, Any]] = None) -> MemoryRecord:
        """Edit in place (no version history). Use ``add(..., key=...)`` when you want history."""
        with self._mem._write_lock:
            record = self._require(record_id)
            if metadata is not None:
                record.metadata.update(metadata)
            if content is not None:
                content = content.strip()
                if not content:
                    raise ValueError("Memory content must be non-empty")
                record.content = content
            if content is not None or metadata is not None:
                record.embedding = self._mem.embedder.embed([_index_text(record.content, record.metadata)])[0]
            if importance is not None:
                record.importance = clamp(importance)
            record.updated_at = self._now()
            self._store.update(record, self.scope)
        return record

    def delete(self, record_id: str) -> bool:
        """Permanently delete one memory this scope owns."""
        deleted = self._store.delete(record_id, self.scope)
        if deleted:
            self._traces.log("delete", self.scope, memory_id=record_id)
        return deleted

    def all(self, memory_type: Optional[Union[MemoryType, str]] = None, *, kinds: Optional[Sequence[str]] = None,
            include_inactive: bool = False) -> List[MemoryRecord]:
        types = [MemoryType.coerce(memory_type)] if memory_type else None
        statuses = tuple(Status) if include_inactive else (Status.ACTIVE,)
        return self._query(types, kinds=kinds, statuses=statuses)

    def history(self, key: str) -> List[MemoryRecord]:
        """Every version stored under ``key``, newest first."""
        return self._store.query(self.scope, memory_types=list(DURABLE_TYPES), key=key, statuses=tuple(Status))

    def forget_user(self) -> int:
        """Delete every memory this scope owns (e.g. a GDPR erasure request). Returns the count."""
        count = 0
        for record in self._store.query(self.scope, memory_types=list(DURABLE_TYPES), statuses=tuple(Status),
                                        owned_only=True):
            count += self._store.delete(record.id, self.scope)
        self.working.clear()
        self._traces.log("forget", self.scope, count=count)
        return count

    # --- Sessions -----------------------------------------------------------------
    def add_turn(self, role: str, content: str) -> None:
        """Append a conversation turn to working memory."""
        self.working.add_turn(role, content, self._now())

    def end_session(self, summarizer: Optional[Callable[[str], str]] = None, clear: bool = True) -> List[MemoryRecord]:
        """Move this thread's working memory into episodic memory.

        With a ``summarizer(transcript) -> str`` (usually an LLM call) one summary episode is
        stored; without one, each user turn is stored as an episode.
        """
        turns = self.working.turns()
        stored: List[MemoryRecord] = []
        if turns:
            if summarizer is not None:
                summary = summarizer(self.working.transcript()).strip()
                if summary:
                    stored.append(self.add(summary, MemoryType.EPISODIC, kind="session_summary", importance=0.6))
            else:
                stored = [self.add(f"{t.role} said: {t.content}", MemoryType.EPISODIC, kind="turn", importance=0.3)
                          for t in turns if t.role == "user"]
        if clear:
            self.working.clear()
        return stored
