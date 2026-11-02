"""Refine: keep memory small, current and useful over time.

* **Dedupe** - near-identical memories merge; the newest wording wins, signals add up.
* **Decay** - stale, unimportant, never-reused episodes are archived (soft, recoverable).
* **Promote** - recurring episodes are summarized into a semantic insight (needs a summarizer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence

from .embeddings import cosine
from .retrieval import DAY
from .text import same_specifics
from .types import MemoryRecord, MemoryType, Status

if TYPE_CHECKING:  # pragma: no cover
    from .memory import ScopedMemory

Summarizer = Callable[[List[str]], str]

# Keyed records (relations, glossary terms, procedures) are maintained by upsert instead.
_SKIP_KINDS = {"relation"}


@dataclass
class RefineReport:
    examined: int = 0
    merged: List[str] = field(default_factory=list)
    archived: List[str] = field(default_factory=list)
    promoted: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (f"RefineReport(examined={self.examined}, merged={len(self.merged)}, "
                f"archived={len(self.archived)}, promoted={len(self.promoted)})")


class Refiner:
    """
    Args:
        dedupe_threshold: cosine similarity at which two memories of the same type merge.
        archive_below: decay strength (importance x recency + reuse) under which an old memory is archived.
        min_age_days: never archive anything younger than this.
        keep_importance: never archive memories at least this important (e.g. a production incident).
        decay_types: memory types that decay. Facts and procedures don't by default.
        cluster_threshold / min_cluster: how similar, and how many, episodes make a promotable pattern.
    """

    def __init__(self, dedupe_threshold: float = 0.93, archive_below: float = 0.08, min_age_days: float = 30.0,
                 keep_importance: float = 0.7, decay_types: Sequence[MemoryType] = (MemoryType.EPISODIC,),
                 cluster_threshold: float = 0.55, min_cluster: int = 3) -> None:
        self.dedupe_threshold = dedupe_threshold
        self.archive_below = archive_below
        self.keep_importance = keep_importance
        self.min_age_days = min_age_days
        self.decay_types = tuple(decay_types)
        self.cluster_threshold = cluster_threshold
        self.min_cluster = min_cluster

    def run(self, memory: ScopedMemory, summarizer: Optional[Summarizer] = None) -> RefineReport:
        report = RefineReport()
        records = [r for r in memory._query(owned_only=True) if r.kind not in _SKIP_KINDS and not r.key]
        report.examined = len(records)
        records = self._dedupe(memory, records, report)
        records = self._decay(memory, records, report)
        if summarizer is not None:
            self._promote(memory, records, summarizer, report)
        memory._traces.log("refine", memory.scope, merged=report.merged, archived=report.archived,
                           promoted=report.promoted, examined=report.examined)
        return report

    def _dedupe(self, memory: ScopedMemory, records: List[MemoryRecord], report: RefineReport) -> List[MemoryRecord]:
        survivors: List[MemoryRecord] = []
        for record in sorted(records, key=lambda r: r.created_at, reverse=True):  # newest first
            twin = next((s for s in survivors if s.memory_type == record.memory_type and s.embedding
                         and record.embedding and cosine(s.embedding, record.embedding) >= self.dedupe_threshold
                         and same_specifics(s.content, record.content)), None)
            if twin is None:
                survivors.append(record)
                continue
            twin.importance = max(twin.importance, record.importance)
            twin.access_count += record.access_count
            twin.metadata.setdefault("merged_from", []).append(record.id)
            record.status, record.superseded_by = Status.SUPERSEDED, twin.id
            memory._persist(twin, record)
            report.merged.append(record.id)
        return survivors

    def _decay(self, memory: ScopedMemory, records: List[MemoryRecord], report: RefineReport) -> List[MemoryRecord]:
        now = memory._now()
        recency = memory._mem.retriever.config.recency
        kept = []
        for record in records:
            age_days = (now - record.created_at) / DAY
            strength = record.importance * recency(record, now) + min(record.access_count, 10) * 0.05
            if (record.memory_type in self.decay_types and age_days >= self.min_age_days
                    and record.importance < self.keep_importance and strength < self.archive_below):
                record.status = Status.ARCHIVED
                memory._persist(record)
                report.archived.append(record.id)
            else:
                kept.append(record)
        return kept

    def _promote(self, memory: ScopedMemory, records: List[MemoryRecord], summarizer: Summarizer,
                 report: RefineReport) -> None:
        episodes = [r for r in records if r.memory_type == MemoryType.EPISODIC and r.embedding
                    and "promoted_to" not in r.metadata]
        clusters: List[List[MemoryRecord]] = []
        for episode in sorted(episodes, key=lambda r: r.created_at):
            home = next((c for c in clusters if cosine(c[0].embedding, episode.embedding) >= self.cluster_threshold),
                        None)
            if home is None:
                clusters.append([episode])
            else:
                home.append(episode)
        for cluster in clusters:
            if len(cluster) < self.min_cluster:
                continue
            summary = summarizer([r.content for r in cluster]).strip()
            if not summary:
                continue
            insight = memory.add(summary, MemoryType.SEMANTIC, kind="insight",
                                 importance=max(r.importance for r in cluster),
                                 metadata={"source_ids": [r.id for r in cluster]})
            for episode in cluster:
                episode.metadata["promoted_to"] = insight.id
            memory._persist(*cluster)
            report.promoted.append(insight.id)
