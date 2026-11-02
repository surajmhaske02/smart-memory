"""Hybrid retrieval: vector similarity + BM25 keywords + recency + importance + reuse."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .embeddings import cosine
from .text import tokenize
from .types import MemoryRecord, MemoryType, SearchResult

DAY = 86_400.0


@dataclass
class RetrievalConfig:
    """Weights for the final score. Relevance (semantic/keyword) gates; the rest re-ranks.

    ``min_relevance`` drops candidates whose best relevance signal is below it, so an
    important-but-unrelated memory never sneaks into a prompt.
    """

    semantic_weight: float = 0.55
    keyword_weight: float = 0.25
    recency_weight: float = 0.08
    importance_weight: float = 0.10
    reuse_weight: float = 0.02
    min_relevance: float = 0.12
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
             now: float, top_k: int = 5, min_relevance: Optional[float] = None) -> List[SearchResult]:
        cfg = self.config
        threshold = cfg.min_relevance if min_relevance is None else min_relevance
        if not candidates:
            return []
        keyword = bm25_scores(tokenize(query), [tokenize(self._index_text(c)) for c in candidates])

        results: List[SearchResult] = []
        for record, kw in zip(candidates, keyword):
            sem = max(0.0, cosine(query_vec, record.embedding)) if query_vec and record.embedding else 0.0
            relevance = max(sem, kw)
            if relevance < threshold:
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
            results.append(SearchResult(record=record, score=score, signals=signals))

        results.sort(key=lambda r: r.score, reverse=True)
        return self._dedupe(results, top_k)

    @staticmethod
    def _index_text(record: MemoryRecord) -> str:
        extra = record.metadata.get("search_text")
        return f"{record.content} {extra}" if extra else record.content

    def _dedupe(self, ranked: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Skip results nearly identical to a higher-ranked one, so k slots carry k distinct facts."""
        kept: List[SearchResult] = []
        for result in ranked:
            emb = result.record.embedding
            if emb and any(k.record.embedding and cosine(emb, k.record.embedding) >= self.config.dedupe_threshold
                           for k in kept):
                continue
            kept.append(result)
            if len(kept) >= top_k:
                break
        return kept
