"""The public API: ``Memory`` (the engine) and ``ScopedMemory`` (a user/agent/thread view of it).

The four operations every agent memory needs:

* **Retain** - :meth:`ScopedMemory.add` routes and stores new information.
* **Recall** - :meth:`ScopedMemory.search` finds relevant memories with hybrid ranking.
* **Reuse** - :meth:`ScopedMemory.context` compiles a compact Context Card for the prompt.
* **Refine** - :meth:`ScopedMemory.feedback` and :meth:`ScopedMemory.refine` improve memory over time.
"""

from __future__ import annotations

import copy
import json
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .context import ContextBuilder, ContextCard
from .embeddings import CachedEmbedder, Embedder, HashingEmbedder
from .index import ScopeIndex
from .ontology import Glossary, KnowledgeGraph, norm
from .refine import Refiner, RefineReport, Summarizer
from .retrieval import HybridRetriever, RetrievalConfig
from .router import Router, RuleRouter
from .storage.base import StorageBackend
from .storage.sqlite import SQLiteStore
from .text import same_specifics
from .traces import TraceLog
from .types import DURABLE_TYPES, MemoryRecord, MemoryType, RoutingDecision, Scope, SearchResult, Status, clamp
from .working import WorkingMemory

_DEFAULT_KIND = {MemoryType.EPISODIC: "event", MemoryType.SEMANTIC: "fact", MemoryType.PROCEDURAL: "instruction"}

TypeArg = Optional[Union[MemoryType, str]]


class EmbedderMismatchError(RuntimeError):
    """The store was built with a different embedder. Vectors from two models are not comparable."""


def _clone(record: MemoryRecord) -> MemoryRecord:
    return replace(record, metadata=copy.deepcopy(record.metadata))


def _index_text(content: str, metadata: Mapping[str, Any]) -> str:
    extra = metadata.get("search_text")
    return f"{content} {extra}" if extra else content


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
        dedupe_on_write: Cosine similarity at which a new memory reinforces an existing one of the
            same type instead of being stored again (counted in ``metadata["occurrences"]``).
            ``None`` stores every write.
        embed_cache_size: LRU entries cached in front of the embedder (0 disables).
        index_cache_size: Number of user/agent search indexes kept warm in process (0 disables).
        clock: Time source; inject a fake clock in tests.
    """

    def __init__(self, path: Union[str, Path] = ":memory:", *, store: Optional[StorageBackend] = None,
                 embedder: Optional[Embedder] = None, router: Optional[Router] = None,
                 retrieval: Optional[RetrievalConfig] = None, context_builder: Optional[ContextBuilder] = None,
                 refiner: Optional[Refiner] = None, on_embedder_change: str = "error", trace: bool = True,
                 dedupe_on_write: Optional[float] = 0.97, embed_cache_size: int = 4096,
                 index_cache_size: int = 64, clock: Callable[[], float] = time.time,
                 working_max_turns: int = 20) -> None:
        self.store = store if store is not None else SQLiteStore(str(path))
        self.embedder: Embedder = embedder or HashingEmbedder()
        self._embedder: Embedder = (CachedEmbedder(self.embedder, embed_cache_size) if embed_cache_size > 0
                                    else self.embedder)
        self.router: Router = router or RuleRouter()
        self.retriever = HybridRetriever(retrieval)
        self.context_builder = context_builder or ContextBuilder()
        self.refiner = refiner or Refiner()
        self.dedupe_on_write = dedupe_on_write
        self.clock = clock
        self.traces = TraceLog(self.store, enabled=trace, clock=clock)
        self._working: Dict[Tuple[Optional[str], Optional[str], Optional[str]], WorkingMemory] = {}
        self._working_max_turns = working_max_turns
        self._lock = threading.RLock()
        self._indexes: OrderedDict[Tuple[Optional[str], Optional[str]], Tuple[Any, ScopeIndex]] = OrderedDict()
        self._index_cache_size = index_cache_size
        self._index_builds = 0
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

    # --- warm index cache ---------------------------------------------------------
    def _index_for(self, scope: Scope) -> ScopeIndex:
        """The scope's search index. Caller must hold ``self._lock``."""
        key = (scope.user_id, scope.agent_id)
        generation = self.store.generation()
        cached = self._indexes.get(key)
        if cached is not None and generation is not None and cached[0] == generation:
            self._indexes.move_to_end(key)
            return cached[1]
        records = self.store.query(Scope(scope.user_id, scope.agent_id), memory_types=list(DURABLE_TYPES))
        index = ScopeIndex.build(records, self.embedder.dim)
        self._index_builds += 1
        if generation is not None and self._index_cache_size > 0:
            self._indexes[key] = (generation, index)
            self._indexes.move_to_end(key)
            while len(self._indexes) > self._index_cache_size:
                self._indexes.popitem(last=False)
        return index

    def _apply(self, before: Any, upserts: Sequence[MemoryRecord] = (), removes: Sequence[str] = ()) -> None:
        """Patch warm indexes after this process wrote to the store. Caller holds ``self._lock``.

        Indexes that were current at ``before`` are patched and moved to the new generation;
        anything else (e.g. another process also wrote) is dropped and rebuilt on next use.
        """
        after = self.store.generation()
        if before is None or after is None:
            self._indexes.clear()
            return
        for key in list(self._indexes):
            generation, index = self._indexes[key]
            if generation != before:
                del self._indexes[key]
                continue
            view = Scope(*key)
            for rid in removes:
                index.remove(rid)
            for record in upserts:
                if not view.can_see(record) or record.memory_type not in DURABLE_TYPES:
                    continue
                if record.status == Status.ACTIVE:
                    index.upsert(_clone(record))
                else:
                    index.remove(record.id)
            self._indexes[key] = (after, index)

    def _touch(self, ids: Sequence[str], now: float) -> None:
        if not ids:
            return
        with self._lock:
            self.store.touch(ids, now)
            for _, index in self._indexes.values():
                index.touch(ids, now)

    def cache_info(self) -> Dict[str, Any]:
        """Counters that show what the caches saved (embedding calls, index rebuilds)."""
        info: Dict[str, Any] = {"index_builds": self._index_builds, "warm_indexes": len(self._indexes)}
        if isinstance(self._embedder, CachedEmbedder):
            info.update(embed_calls=self._embedder.calls, embed_hits=self._embedder.hits,
                        embed_misses=self._embedder.misses)
        return info

    # --- maintenance --------------------------------------------------------------
    def reembed(self, batch_size: int = 64) -> int:
        """Recompute every embedding with the current embedder. Returns the number of records."""
        records = list(self.store.iter_all())
        with self._lock:
            for start in range(0, len(records), batch_size):
                batch = records[start:start + batch_size]
                vectors = self._embedder.embed([_index_text(r.content, r.metadata) for r in batch])
                for record, vec in zip(batch, vectors):
                    record.embedding = vec
                    self.store.update(record, record.scope)
            self.store.set_meta("embedder", self.embedder_signature)
            self._indexes.clear()
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
        with self._lock:
            if key not in self._working:
                self._working[key] = WorkingMemory(self._working_max_turns)
            return self._working[key]

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


IdsOrResults = Union[str, SearchResult, MemoryRecord, Iterable[Union[str, SearchResult, MemoryRecord]]]


def _ids(items: IdsOrResults) -> List[str]:
    if isinstance(items, (str, SearchResult, MemoryRecord)):
        items = [items]
    return [i if isinstance(i, str) else i.id for i in items]


@dataclass
class _Pending:
    record: MemoryRecord
    decision: RoutingDecision


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

    def _active(self, kinds: Optional[Sequence[str]] = None,
                memory_types: Optional[Sequence[MemoryType]] = None) -> List[MemoryRecord]:
        """Active visible records from the warm index. Read-only: do not mutate the results."""
        with self._mem._lock:
            records = [r for _, r in self._mem._index_for(self.scope).items()]
        return [r for r in records if (kinds is None or r.kind in kinds)
                and (memory_types is None or r.memory_type in memory_types)]

    def _persist(self, *records: MemoryRecord) -> None:
        """Write changed records through to the store and the warm indexes."""
        with self._mem._lock:
            before = self._store.generation()
            for record in records:
                self._store.update(record, self.scope)
            self._mem._apply(before, upserts=records)

    # --- Retain -------------------------------------------------------------------
    def add(self, content: str, memory_type: TypeArg = None, *, kind: Optional[str] = None,
            importance: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None,
            key: Optional[str] = None, shared: bool = False, pin: bool = False) -> MemoryRecord:
        """Store a memory. Without ``memory_type`` the router decides.

        Args:
            key: A stable identity such as ``"alice.diet"``. Adding again with the same key
                supersedes the old version (same content, ignoring case, is a no-op), so facts stay current.
            shared: Store at user level (``agent_id=None``) so every agent of this user sees it.
            pin: Always include this memory at the top of every Context Card (a user profile,
                persona or hard rule). Pinned lines form a stable, prompt-cache-friendly prefix.
        """
        return self.add_many([content], memory_type, kind=kind, importance=importance, metadata=metadata,
                             key=key, shared=shared, pin=pin)[0]

    retain = add

    def add_many(self, contents: Iterable[str], memory_type: TypeArg = None, *, kind: Optional[str] = None,
                 importance: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None,
                 key: Optional[str] = None, shared: bool = False, pin: bool = False) -> List[MemoryRecord]:
        """Store many memories with one embedding call and one database transaction."""
        contents = [c.strip() for c in contents]
        if any(not c for c in contents):
            raise ValueError("Memory content must be non-empty")
        if not contents:
            return []
        if key is not None and len(contents) > 1:
            raise ValueError("key identifies a single memory; add keyed memories one at a time")
        pending = [self._prepare(c, memory_type, kind, importance, metadata, key, shared, pin) for c in contents]
        vectors = self._mem._embedder.embed([_index_text(p.record.content, p.record.metadata) for p in pending])
        for p, vec in zip(pending, vectors):
            p.record.embedding = vec

        out: List[MemoryRecord] = []
        inserts: List[_Pending] = []
        with self._mem._lock:
            before = self._store.generation()
            index = self._mem._index_for(self.scope)
            pending_by_id: Dict[str, MemoryRecord] = {}
            changed: List[MemoryRecord] = []
            try:
                for p in pending:
                    record = p.record
                    if key is not None:
                        existing, superseded = self._upsert_key(record, key)
                        changed.extend(superseded)
                        if existing is not None:
                            changed.append(existing)
                            out.append(existing)
                            continue
                    elif self._mem.dedupe_on_write is not None:
                        twin = self._find_twin(index, record, pending_by_id)
                        if twin is not None:
                            if twin.id not in pending_by_id:
                                changed.append(twin)
                            out.append(twin)
                            continue
                    inserts.append(p)
                    pending_by_id[record.id] = record
                    clone = _clone(record)
                    clone.embedding = list(record.embedding or [])
                    index.upsert(clone)  # visible to dedupe for the rest of this batch
                    out.append(record)
                if inserts:
                    self._store.add_many([p.record for p in inserts])
            except BaseException:
                self._mem._indexes.clear()  # the index may hold records that never reached the store
                raise
            self._mem._apply(before, upserts=[*changed, *(p.record for p in inserts)])
        for p in inserts:
            d = p.decision
            self._traces.log("route", self.scope, memory_id=p.record.id, text=p.record.content[:2000],
                             memory_type=p.record.memory_type.value, kind=p.record.kind, confidence=d.confidence,
                             router=d.router, reason=d.reason)
        return out

    def _prepare(self, content: str, memory_type: TypeArg, kind: Optional[str], importance: Optional[float],
                 metadata: Optional[Dict[str, Any]], key: Optional[str], shared: bool, pin: bool) -> _Pending:
        metadata = dict(metadata or {})
        if pin:
            metadata["pinned"] = True
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
        record = MemoryRecord(
            content=content, memory_type=decision.memory_type, kind=decision.kind, key=key,
            user_id=self.scope.user_id, agent_id=None if shared else self.scope.agent_id,
            thread_id=self.scope.thread_id if decision.memory_type == MemoryType.EPISODIC else None,
            metadata=metadata, importance=decision.importance, created_at=self._now(),
        )
        return _Pending(record, decision)

    def _upsert_key(self, record: MemoryRecord, key: str) -> Tuple[Optional[MemoryRecord], List[MemoryRecord]]:
        """(existing record if this re-add is a no-op, records superseded by the new version)."""
        superseded: List[MemoryRecord] = []
        now = record.created_at
        for existing in self._store.query(self.scope, key=key, owned_only=True):
            if norm(existing.content) == norm(record.content) and existing.memory_type == record.memory_type:
                if existing.content == record.content:
                    existing.metadata.update(record.metadata)
                else:  # case-only difference: keep the original spelling everywhere
                    for k, v in record.metadata.items():
                        existing.metadata.setdefault(k, v)
                existing.importance = max(existing.importance, record.importance)
                existing.updated_at = now
                self._store.update(existing, self.scope)
                return existing, superseded
            existing.status, existing.superseded_by, existing.updated_at = Status.SUPERSEDED, record.id, now
            self._store.update(existing, self.scope)
            superseded.append(existing)
        return None, superseded

    def _find_twin(self, index: ScopeIndex, record: MemoryRecord,
                   pending: Dict[str, MemoryRecord]) -> Optional[MemoryRecord]:
        """A near-identical memory (stored, or earlier in this batch) to reinforce instead of storing again."""
        threshold = self._mem.dedupe_on_write
        best: Optional[Tuple[float, MemoryRecord]] = None
        for slot in index.near_duplicate_candidates(_index_text(record.content, record.metadata)):
            candidate = index.record(slot)
            if candidate.memory_type != record.memory_type or candidate.key or candidate.agent_id != record.agent_id \
                    or not self.scope.can_write(candidate) or not same_specifics(candidate.content, record.content):
                continue
            sim = index.similarity(slot, record.embedding) if record.embedding else 0.0
            if sim >= threshold and (best is None or sim > best[0]):
                best = (sim, candidate)
        if best is None:
            return None
        candidate = best[1]
        in_batch = candidate.id in pending
        twin = pending[candidate.id] if in_batch else self._store.get(candidate.id, self.scope)
        if twin is None:
            return None
        now = record.created_at
        twin.metadata["occurrences"] = int(twin.metadata.get("occurrences", 1)) + 1
        twin.metadata["last_seen"] = now
        for k, v in record.metadata.items():
            twin.metadata.setdefault(k, v)
        twin.importance = max(twin.importance, record.importance)
        twin.updated_at = now
        if not in_batch:  # batch members are written by the batch insert
            self._store.update(twin, self.scope)
            self._traces.log("reinforce", self.scope, memory_id=twin.id, text=record.content[:2000],
                             occurrences=twin.metadata["occurrences"])
        return twin

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
        with self._mem._lock:
            record = self.procedure(name)
            if record is None:
                raise KeyError(f"No procedure named {name!r}")
            stats = record.metadata.setdefault("outcomes", {"success": 0, "failure": 0})
            stats["success" if success else "failure"] += 1
            total = stats["success"] + stats["failure"]
            rate = (stats["success"] + 1) / (total + 2)  # Laplace smoothing
            record.importance = clamp(0.3 + 0.7 * rate)
            record.updated_at = self._now()
            self._persist(record)
        self._traces.log("outcome", self.scope, memory_id=record.id, procedure=name, success=success)
        return record

    # --- Recall -------------------------------------------------------------------
    def search(self, query: str, k: int = 5, *, memory_types: Optional[Sequence[Union[MemoryType, str]]] = None,
               kinds: Optional[Sequence[str]] = None, thread_only: bool = False,
               min_relevance: Optional[float] = None, relative_cutoff: Optional[float] = None) -> List[SearchResult]:
        """Most relevant memories for ``query``, ranked by relevance, recency, importance and reuse."""
        return self._search(query, k=k, memory_types=memory_types, kinds=kinds, thread_only=thread_only,
                            min_relevance=min_relevance, relative_cutoff=relative_cutoff, touch=True)

    recall = search

    def _search(self, query: str, *, k: int, memory_types: Optional[Sequence[Union[MemoryType, str]]] = None,
                kinds: Optional[Sequence[str]] = None, thread_only: bool = False,
                min_relevance: Optional[float] = None, relative_cutoff: Optional[float] = None,
                touch: bool = True) -> List[SearchResult]:
        if thread_only and not self.scope.thread_id:
            raise ValueError("thread_only=True needs a scope with a thread_id")
        types = {MemoryType.coerce(t) for t in memory_types} if memory_types else set(DURABLE_TYPES)
        kind_set = set(kinds) if kinds is not None else None
        thread = self.scope.thread_id

        def allow(r: MemoryRecord) -> bool:
            return (r.memory_type in types and (kind_set is None or r.kind in kind_set)
                    and (not thread_only or r.thread_id == thread))

        now = self._now()
        query_vec = self._mem._embedder.embed([query])[0]
        with self._mem._lock:
            index = self._mem._index_for(self.scope)
            results = self._mem.retriever.rank_index(query, query_vec, index, now, top_k=k,
                                                     min_relevance=min_relevance,
                                                     relative_cutoff=relative_cutoff, allow=allow)
        if touch:
            self._mem._touch([r.id for r in results], now)
            self._traces.log("retrieve", self.scope, query=query, k=k,
                             results=[{"id": r.id, "score": round(r.score, 4)} for r in results])
        return results

    # --- Reuse --------------------------------------------------------------------
    def context(self, query: str, max_tokens: int = 800, **kwargs: Any) -> ContextCard:
        """A compact Context Card for ``query``: put ``str(card)`` in your system prompt.

        Keyword args go to :meth:`ContextBuilder.build` (``k``, ``graph_depth``, ``recent_turns``,
        ``include_working``, ``memory_types``, ``min_relevance``, ``relative_cutoff``).
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
        changed = []
        with self._mem._lock:
            for rid in ids:
                record = self._store.get(rid, self.scope)
                if record is None or not self.scope.can_write(record):
                    continue
                record.importance = clamp(record.importance + delta)
                changed.append(record)
            self._persist(*changed)
        self._traces.log("feedback", self.scope, memory_ids=ids, helpful=helpful, delta=delta)
        return len(changed)

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
        self._persist(record)
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
        with self._mem._lock:
            record = self._require(record_id)
            if metadata is not None:
                record.metadata.update(metadata)
            if content is not None:
                content = content.strip()
                if not content:
                    raise ValueError("Memory content must be non-empty")
                record.content = content
            if content is not None or metadata is not None:
                record.embedding = self._mem._embedder.embed([_index_text(record.content, record.metadata)])[0]
            if importance is not None:
                record.importance = clamp(importance)
            record.updated_at = self._now()
            self._persist(record)
        return record

    def pin(self, record_id: str, pinned: bool = True) -> MemoryRecord:
        """Pin (or unpin) a memory so it always opens the Context Card."""
        with self._mem._lock:
            record = self._require(record_id)
            if pinned:
                record.metadata["pinned"] = True
            else:
                record.metadata.pop("pinned", None)
            self._persist(record)
        return record

    def delete(self, record_id: str) -> bool:
        """Permanently delete one memory this scope owns."""
        with self._mem._lock:
            before = self._store.generation()
            deleted = self._store.delete(record_id, self.scope)
            self._mem._apply(before, removes=[record_id])
        if deleted:
            self._traces.log("delete", self.scope, memory_id=record_id)
        return deleted

    def all(self, memory_type: TypeArg = None, *, kinds: Optional[Sequence[str]] = None,
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
        with self._mem._lock:
            before = self._store.generation()
            removed = []
            for record in self._store.query(self.scope, memory_types=list(DURABLE_TYPES), statuses=tuple(Status),
                                            owned_only=True):
                if self._store.delete(record.id, self.scope):
                    count += 1
                    removed.append(record.id)
            self._mem._apply(before, removes=removed)
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
                user_turns = [f"{t.role} said: {t.content}" for t in turns if t.role == "user"]
                if user_turns:
                    stored = self.add_many(user_turns, MemoryType.EPISODIC, kind="turn", importance=0.3)
        if clear:
            self.working.clear()
        return stored
