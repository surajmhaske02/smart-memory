"""In-process search index for one scope: vectors, BM25 postings and records, kept warm between queries.

Building it costs one storage scan; after that, writes made through ``Memory`` patch it
in place, so search never re-reads, re-decodes or re-tokenizes the whole scope. Uses
numpy when installed (``pip install numpy``), otherwise C-level ``math.sumprod``.
"""

from __future__ import annotations

import math
import operator
from array import array
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .text import tokenize
from .types import MemoryRecord

try:  # optional acceleration
    import numpy as _np
except ImportError:  # pragma: no cover - exercised when numpy is absent
    _np = None

_sumprod = getattr(math, "sumprod", None)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return _sumprod(a, b) if _sumprod else sum(map(operator.mul, a, b))  # pragma: no branch


def index_text(record: MemoryRecord) -> str:
    extra = record.metadata.get("search_text")
    return f"{record.content} {extra}" if extra else record.content


class ScopeIndex:
    def __init__(self, dim: int, use_numpy: Optional[bool] = None) -> None:
        self.dim = dim
        self.use_numpy = (_np is not None) if use_numpy is None else (use_numpy and _np is not None)
        self.records: List[Optional[MemoryRecord]] = []
        self.pos: Dict[str, int] = {}
        self._tf: List[Optional[Counter]] = []
        self._len: List[int] = []
        self.postings: Dict[str, Set[int]] = {}
        self.df: Counter = Counter()
        self.total_len = 0
        self.live = 0
        if self.use_numpy:
            self._matrix = _np.zeros((16, dim), dtype=_np.float32)
            self._alive = _np.zeros(16, dtype=bool)
        else:
            self._vectors: List[Optional[array]] = []

    @classmethod
    def build(cls, records: Iterable[MemoryRecord], dim: int, use_numpy: Optional[bool] = None) -> ScopeIndex:
        index = cls(dim, use_numpy)
        for record in records:
            index.upsert(record)
        return index

    def __len__(self) -> int:
        return self.live

    def __contains__(self, record_id: str) -> bool:
        return record_id in self.pos

    # --- mutation -----------------------------------------------------------------
    def upsert(self, record: MemoryRecord) -> None:
        if record.id in self.pos:
            self.remove(record.id)
        slot = len(self.records)
        vec = record.embedding
        record.embedding = None  # the index owns the vector; avoids a second copy per record
        tokens = Counter(tokenize(index_text(record)))
        self.records.append(record)
        self._tf.append(tokens)
        self._len.append(sum(tokens.values()))
        self.pos[record.id] = slot
        for term in tokens:
            self.postings.setdefault(term, set()).add(slot)
            self.df[term] += 1
        self.total_len += self._len[slot]
        self.live += 1
        if self.use_numpy:
            if slot >= len(self._matrix):
                grow = max(16, len(self._matrix))
                self._matrix = _np.vstack([self._matrix, _np.zeros((grow, self.dim), dtype=_np.float32)])
                self._alive = _np.concatenate([self._alive, _np.zeros(grow, dtype=bool)])
            if vec is not None:
                self._matrix[slot] = vec
            self._alive[slot] = vec is not None
        else:
            self._vectors.append(array("f", vec) if vec is not None else None)

    def remove(self, record_id: str) -> None:
        slot = self.pos.pop(record_id, None)
        if slot is None:
            return
        for term in self._tf[slot] or ():
            self.postings[term].discard(slot)
            self.df[term] -= 1
            if not self.df[term]:
                del self.df[term], self.postings[term]
        self.total_len -= self._len[slot]
        self.live -= 1
        self.records[slot] = None
        self._tf[slot] = None
        if self.use_numpy:
            self._alive[slot] = False
        else:
            self._vectors[slot] = None

    def touch(self, record_ids: Sequence[str], timestamp: float) -> None:
        for rid in record_ids:
            slot = self.pos.get(rid)
            if slot is not None:
                record = self.records[slot]
                record.access_count += 1
                record.last_accessed_at = timestamp

    # --- lookup -------------------------------------------------------------------
    def live_slots(self) -> List[int]:
        return sorted(self.pos.values())

    def vector(self, slot: int) -> Optional[List[float]]:
        if self.use_numpy:
            return self._matrix[slot].tolist() if self._alive[slot] else None
        vec = self._vectors[slot]
        return vec.tolist() if vec is not None else None

    def similarities(self, query_vec: Optional[Sequence[float]], floor: float = 0.0) -> Dict[int, float]:
        """Cosine similarity (vectors are normalized) for live slots scoring at least ``floor`` (and > 0)."""
        if not query_vec or not self.live:
            return {}
        if self.use_numpy:
            n = len(self.records)
            # Apple's Accelerate BLAS (some numpy 2.x builds) raises spurious floating-point flags
            # in matmul even for finite inputs; the values are correct, and non-finite ones are dropped.
            with _np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                sims = self._matrix[:n] @ _np.asarray(query_vec, dtype=_np.float32)
            hits = _np.nonzero(self._alive[:n] & _np.isfinite(sims) & (sims >= max(floor, 1e-9)))[0]
            return {int(i): float(sims[i]) for i in hits}
        q = array("f", query_vec)
        out = {}
        for slot, vec in enumerate(self._vectors):
            if vec is not None:
                sim = _dot(vec, q)
                if sim > 0 and sim >= floor:
                    out[slot] = sim
        return out

    def similarity(self, slot: int, query_vec: Sequence[float]) -> float:
        if self.use_numpy:
            if not self._alive[slot]:
                return 0.0
            with _np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                return max(0.0, float(self._matrix[slot] @ _np.asarray(query_vec, dtype=_np.float32)))
        vec = self._vectors[slot]
        return max(0.0, _dot(vec, array("f", query_vec))) if vec is not None else 0.0

    def pair_similarity(self, a: int, b: int) -> float:
        if self.use_numpy:
            if not (self._alive[a] and self._alive[b]):
                return 0.0
            with _np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                return float(self._matrix[a] @ self._matrix[b])
        va, vb = self._vectors[a], self._vectors[b]
        return _dot(va, vb) if va is not None and vb is not None else 0.0

    def bm25(self, query: str, k1: float = 1.2, b: float = 0.75) -> Dict[int, float]:
        """BM25 over documents sharing a term with the query, normalized to [0, 1].

        Half best-match-relative score, half query-term coverage, so a document matching
        every query term outranks one matching a single rare term.
        """
        terms = set(tokenize(query))
        if not terms or not self.live:
            return {}
        n = self.live
        avgdl = self.total_len / n or 1.0
        raw: Dict[int, float] = {}
        matched: Dict[int, int] = {}
        for term in terms:
            slots = self.postings.get(term)
            if not slots:
                continue
            idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
            for slot in slots:
                freq = self._tf[slot][term]
                raw[slot] = raw.get(slot, 0.0) + idf * freq * (k1 + 1) / (
                    freq + k1 * (1 - b + b * self._len[slot] / avgdl))
                matched[slot] = matched.get(slot, 0) + 1
        best = max(raw.values(), default=0.0)
        if best <= 0:
            return {}
        return {slot: 0.5 * (s / best) + 0.5 * matched[slot] / len(terms) for slot, s in raw.items()}

    def near_duplicate_candidates(self, text: str, rare_terms: int = 2) -> Set[int]:
        """Slots sharing the text's rarest terms: the only places a near-duplicate can hide.

        Keeps write-time dedupe O(matches) instead of O(index size). A term the index has
        never seen means nothing stored can be a near-copy, so the result is empty.
        """
        terms = set(tokenize(text))
        if not terms or any(t not in self.df for t in terms):
            return set()
        rarest = sorted(terms, key=lambda t: (self.df[t], t))[:rare_terms]
        slots = set(self.postings[rarest[0]])
        for term in rarest[1:]:
            slots &= self.postings[term]
        return slots

    def record(self, slot: int) -> MemoryRecord:
        record = self.records[slot]
        assert record is not None
        return record

    def items(self) -> List[Tuple[int, MemoryRecord]]:
        return [(slot, self.records[slot]) for slot in self.live_slots()]  # type: ignore[misc]
