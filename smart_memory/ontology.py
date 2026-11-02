"""Ontology layer (entity graph) and semantic layer (glossary of canonical definitions).

Both are stored as ordinary semantic memories (``kind="relation"`` / ``kind="term"``),
so they inherit scoping, persistence, tracing and refinement for free.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Set

from .text import tokenize
from .types import MemoryRecord, MemoryType

if TYPE_CHECKING:  # pragma: no cover
    from .memory import ScopedMemory


def norm(text: str) -> str:
    return " ".join(text.casefold().split())


def _mentions(text: str, phrase: str, text_tokens: Sequence[str]) -> bool:
    if re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, re.IGNORECASE):
        return True
    phrase_tokens = tokenize(phrase)
    n = len(phrase_tokens)
    return n > 0 and any(list(text_tokens[i:i + n]) == phrase_tokens for i in range(len(text_tokens) - n + 1))


@dataclass(frozen=True)
class Relation:
    subject: str
    predicate: str
    object: str
    id: str
    confidence: float = 1.0

    def __str__(self) -> str:
        return f"{self.subject} --{self.predicate}--> {self.object}"

    @property
    def sentence(self) -> str:
        return f"{self.subject} {self.predicate.replace('_', ' ')} {self.object}"


class KnowledgeGraph:
    """Explicit entity relationships so the agent reasons over structure, not keyword overlap.

    >>> graph.relate("Acme Corp", "supplier_of", "Globex")
    >>> graph.relate("Alice", "lives_in", "Paris", exclusive=True)   # later "Berlin" replaces "Paris"
    >>> graph.neighbors("Acme Corp")
    """

    def __init__(self, memory: ScopedMemory) -> None:
        self._memory = memory

    def relate(self, subject: str, predicate: str, obj: str, *, exclusive: bool = False,
               confidence: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> Relation:
        """Add ``subject --predicate--> obj``. Idempotent.

        With ``exclusive=True`` the (subject, predicate) pair is single-valued: a new
        object supersedes the old one (e.g. ``lives_in``, ``reports_to``).
        """
        subject, predicate, obj = subject.strip(), predicate.strip(), obj.strip()
        if not (subject and predicate and obj):
            raise ValueError("subject, predicate and object must be non-empty")
        key = f"rel:{norm(subject)}|{norm(predicate)}" + ("" if exclusive else f"|{norm(obj)}")
        sentence = Relation(subject, predicate, obj, "").sentence
        record = self._memory.add(
            sentence, memory_type=MemoryType.SEMANTIC, kind="relation", key=key, importance=0.6,
            metadata={**(metadata or {}), "subject": subject, "predicate": predicate, "object": obj,
                      "confidence": confidence},
        )
        return self._to_relation(record)

    def unrelate(self, subject: str, predicate: str, obj: str) -> bool:
        removed = False
        for rel in self.relations(subject=subject, predicate=predicate):
            if norm(rel.object) == norm(obj):
                removed = self._memory.delete(rel.id) or removed
        return removed

    def relations(self, *, subject: Optional[str] = None, predicate: Optional[str] = None,
                  obj: Optional[str] = None, entity: Optional[str] = None) -> List[Relation]:
        out = []
        for record in self._memory._active(kinds=["relation"], memory_types=[MemoryType.SEMANTIC]):
            rel = self._to_relation(record)
            if subject is not None and norm(rel.subject) != norm(subject):
                continue
            if predicate is not None and norm(rel.predicate) != norm(predicate):
                continue
            if obj is not None and norm(rel.object) != norm(obj):
                continue
            if entity is not None and norm(entity) not in (norm(rel.subject), norm(rel.object)):
                continue
            out.append(rel)
        return out

    def entities(self) -> List[str]:
        seen: Dict[str, str] = {}
        for rel in self.relations():
            seen.setdefault(norm(rel.subject), rel.subject)
            seen.setdefault(norm(rel.object), rel.object)
        return sorted(seen.values(), key=str.casefold)

    def neighbors(self, entity: str, depth: int = 1) -> Set[str]:
        rels = self.facts_about(entity, depth)
        names = {r.subject for r in rels} | {r.object for r in rels}
        return {n for n in names if norm(n) != norm(entity)}

    def facts_about(self, entity: str, depth: int = 1, limit: int = 50) -> List[Relation]:
        """Relations reachable from ``entity`` within ``depth`` hops (breadth-first, nearest first)."""
        adjacency: Dict[str, List[Relation]] = {}
        for rel in self.relations():
            adjacency.setdefault(norm(rel.subject), []).append(rel)
            adjacency.setdefault(norm(rel.object), []).append(rel)
        seen_rel: Set[str] = set()
        out: List[Relation] = []
        visited = {norm(entity)}
        frontier = deque([(norm(entity), 0)])
        while frontier and len(out) < limit:
            node, dist = frontier.popleft()
            if dist >= depth:
                continue
            for rel in adjacency.get(node, []):
                if rel.id in seen_rel:
                    continue
                seen_rel.add(rel.id)
                out.append(rel)
                for nxt in (norm(rel.subject), norm(rel.object)):
                    if nxt not in visited:
                        visited.add(nxt)
                        frontier.append((nxt, dist + 1))
        return out[:limit]

    def find_entities(self, text: str) -> List[str]:
        """Known entities mentioned in ``text`` (longest names first)."""
        tokens = tokenize(text)
        found = [e for e in self.entities() if _mentions(text, e, tokens)]
        return sorted(found, key=len, reverse=True)

    @staticmethod
    def _to_relation(record: MemoryRecord) -> Relation:
        md = record.metadata
        return Relation(md["subject"], md["predicate"], md["object"], record.id, float(md.get("confidence", 1.0)))


@dataclass(frozen=True)
class Term:
    term: str
    definition: str
    aliases: tuple
    id: str

    def __str__(self) -> str:
        alias = f" (aka {', '.join(self.aliases)})" if self.aliases else ""
        return f"{self.term}{alias}: {self.definition}"


class Glossary:
    """Canonical business definitions, injected verbatim whenever a query mentions a term.

    >>> glossary.define("concentration", "Share of total exposure held by the top 10 counterparties.",
    ...                 aliases=["exposure concentration"])
    """

    def __init__(self, memory: ScopedMemory) -> None:
        self._memory = memory

    def define(self, term: str, definition: str, aliases: Iterable[str] = ()) -> Term:
        term, definition = term.strip(), definition.strip()
        if not term or not definition:
            raise ValueError("term and definition must be non-empty")
        aliases = tuple(a.strip() for a in aliases if a.strip())
        record = self._memory.add(
            f"{term}: {definition}", memory_type=MemoryType.SEMANTIC, kind="definition",
            key=f"term:{norm(term)}", importance=0.8,
            metadata={"term": term, "definition": definition, "aliases": list(aliases),
                      "search_text": " ".join(aliases), "glossary": True},
        )
        return self._to_term(record)

    def lookup(self, term: str) -> Optional[Term]:
        key = norm(term)
        for t in self.terms():
            if norm(t.term) == key or key in (norm(a) for a in t.aliases):
                return t
        return None

    def terms(self) -> List[Term]:
        records = self._memory._active(kinds=["definition"], memory_types=[MemoryType.SEMANTIC])
        return [self._to_term(r) for r in records if r.metadata.get("glossary")]

    def remove(self, term: str) -> bool:
        found = self.lookup(term)
        return bool(found) and self._memory.delete(found.id)

    def find_terms(self, text: str) -> List[Term]:
        tokens = tokenize(text)
        found = [t for t in self.terms() if any(_mentions(text, p, tokens) for p in (t.term, *t.aliases))]
        folded = text.casefold()

        def position(t: Term) -> int:
            hits = [folded.find(p.casefold()) for p in (t.term, *t.aliases)]
            return min((h for h in hits if h >= 0), default=len(folded))

        return sorted(found, key=position)

    @staticmethod
    def _to_term(record: MemoryRecord) -> Term:
        md = record.metadata
        return Term(md["term"], md["definition"], tuple(md.get("aliases", ())), record.id)
