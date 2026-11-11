"""Hybrid retrieval: vector similarity + BM25 keywords + recency + importance + reuse."""

from __future__ import annotations

import copy
import math
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .index import ScopeIndex
from .types import MemoryRecord, MemoryType, SearchResult

DAY = 86_400.0


@dataclass
class RetrievalConfig:
    """Weights for the final score. Relevance (semantic/keyword) gates; the rest re-ranks.

    ``min_relevance`` drops candidates whose best relevance signal is below it, so an
    important-but-unrelated memory never sneaks into a prompt. ``relative_cutoff`` trims
    the weak tail of a result list, which saves prompt tokens.
    """

    semantic_weight: float = 0.55
    keyword_weight: float = 0.25
    recency_weight: float = 0.08
    importance_weight: float = 0.10
    reuse_weight: float = 0.02
    min_relevance: float = 0.12
    relative_cutoff: float = 0.0  # e.g. 0.5 drops results scoring under half of the best one
    cutoff_exempt_types: Tuple[MemoryType, ...] = (MemoryType.PROCEDURAL,)
    dedupe_threshold: float = 0.97
    half_life_days: Dict[MemoryType, float] = field(default_factory=lambda: {
        MemoryType.EPISODIC: 14.0,
        MemoryType.SEMANTIC: 365.0,
        MemoryType.PROCEDURAL: 365.0,
    })

    def recency(self, record: MemoryRecord, now: float) -> float:
        half_life = self.half_life_days.get(record.memory_type, 90.0) * DAY
        last_seen = max(record.updated_at, record.last_accessed_at or 0.0)
        age = max(0.0, now - last_seen)
        return 0.5 ** (age / half_life)


def bm25_scores(query_tokens: Sequence[str], docs: Sequence[Sequence[str]], k1: float = 1.2,
                b: float = 0.75) -> List[float]:
    """BM25 over an in-memory candidate set, normalized to [0, 1] by the best match."""
    if not docs or not query_tokens:
        return [0.0] * len(docs)
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n or 1.0
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))
    query_terms = set(query_tokens)
    scores = []
    for d in docs:
        tf = Counter(d)
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            freq = tf[term]
            score += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * len(d) / avgdl))
        scores.append(score)
    best = max(scores)
    if best <= 0:
        return [0.0] * n
    # Fraction of query terms matched keeps scores comparable across queries.
    coverage = [len(query_terms & set(d)) / len(query_terms) for d in docs]
    return [0.5 * (s / best) + 0.5 * c for s, c in zip(scores, coverage)]


class HybridRetriever:
    def __init__(self, config: Optional[RetrievalConfig] = None) -> None:
        self.config = config or RetrievalConfig()

    def rank(self, query: str, query_vec: Optional[Sequence[float]], candidates: Sequence[MemoryRecord],
             now: float, top_k: int = 5, min_relevance: Optional[float] = None,
             relative_cutoff: Optional[float] = None,
             allow: Optional[Callable[[MemoryRecord], bool]] = None) -> List[SearchResult]:
        """Rank an ad-hoc list of records (builds a throwaway index)."""
        dim = len(query_vec) if query_vec else 0
        clones = [replace(c, metadata=copy.deepcopy(c.metadata)) for c in candidates]
        index = ScopeIndex.build(clones, dim, use_numpy=False)
        return self.rank_index(query, query_vec, index, now, top_k, min_relevance, relative_cutoff, allow)

    def rank_index(self, query: str, query_vec: Optional[Sequence[float]], index: ScopeIndex, now: float,
                   top_k: int = 5, min_relevance: Optional[float] = None,
                   relative_cutoff: Optional[float] = None,
                   allow: Optional[Callable[[MemoryRecord], bool]] = None) -> List[SearchResult]:
        """Score only the documents that can pass the relevance gate, then re-rank them.

        Returned records are copies (with embeddings), safe for callers to mutate.
        """
        cfg = self.config
        threshold = cfg.min_relevance if min_relevance is None else min_relevance
        cutoff = cfg.relative_cutoff if relative_cutoff is None else relative_cutoff
        if not len(index):
            return []
        semantic = index.similarities(query_vec, floor=threshold)
        keyword = index.bm25(query)
        slots = set(semantic) | {s for s, kw in keyword.items() if kw >= threshold}

        scored: List[Tuple[float, int, Dict[str, float]]] = []
        for slot in slots:
            record = index.record(slot)
            if allow is not None and not allow(record):
                continue
            sem = semantic.get(slot)
            if sem is None:
                sem = index.similarity(slot, query_vec) if query_vec else 0.0
            kw = keyword.get(slot, 0.0)
            if max(sem, kw) < threshold:
                continue
            signals = {
                "semantic": sem,
                "keyword": kw,
                "recency": cfg.recency(record, now),
                "importance": record.importance,
                "reuse": min(record.access_count, 20) / 20,
            }
            score = (cfg.semantic_weight * sem + cfg.keyword_weight * kw + cfg.recency_weight * signals["recency"]
                     + cfg.importance_weight * record.importance + cfg.reuse_weight * signals["reuse"])
            scored.append((score, slot, signals))

        scored.sort(key=lambda t: (-t[0], t[1]))
        kept: List[Tuple[float, int, Dict[str, float]]] = []
        best = scored[0][0] if scored else 0.0
        for item in scored:
            # Rules and procedures are rare and costly to miss, so the cutoff never drops them.
            if cutoff and item[0] < cutoff * best \
                    and index.record(item[1]).memory_type not in cfg.cutoff_exempt_types:
                continue
            # Skip near-copies of a higher-ranked result so k slots carry k distinct facts.
            if any(index.pair_similarity(item[1], k[1]) >= cfg.dedupe_threshold for k in kept):
                continue
            kept.append(item)
            if len(kept) >= top_k:
                break
        results = []
        for score, slot, signals in kept:
            record = index.record(slot)
            out = replace(record, metadata=copy.deepcopy(record.metadata), embedding=index.vector(slot))
            results.append(SearchResult(record=out, score=score, signals=signals))
        return results
