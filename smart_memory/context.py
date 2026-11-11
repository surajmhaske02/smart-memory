"""Context Cards: compile the relevant slices of every memory store into a compact, token-budgeted block.

Instead of stuffing transcripts into the prompt, the agent gets only what this query
needs, in priority order: current state, definitions, procedures, facts, past events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

from .text import estimate_tokens, truncate
from .types import MemoryRecord, MemoryType, SearchResult

if TYPE_CHECKING:  # pragma: no cover
    from .memory import ScopedMemory


@dataclass
class CardSection:
    title: str
    lines: List[str] = field(default_factory=list)
    memory_ids: List[str] = field(default_factory=list)


@dataclass
class ContextCard:
    query: str
    sections: List[CardSection]
    token_estimate: int
    max_tokens: int
    entities: List[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def memory_ids(self) -> List[str]:
        return [mid for s in self.sections for mid in s.memory_ids]

    @property
    def is_empty(self) -> bool:
        return not any(s.lines for s in self.sections)

    def render(self, fmt: str = "markdown") -> str:
        stable, dynamic = self.render_split(fmt)
        return "\n".join(p for p in (stable, dynamic) if p)

    def render_split(self, fmt: str = "markdown") -> Tuple[str, str]:
        """``(stable, dynamic)`` halves of :meth:`render`, joined by a newline.

        ``stable`` holds the header and pinned Profile, which only change when pinned
        memories change. Put it right after your system prompt and before the per-query
        part so provider prompt caching (billed at a fraction of normal input) covers it.
        """
        if fmt not in ("markdown", "xml"):
            raise ValueError("fmt must be 'markdown' or 'xml'")
        sections = [s for s in self.sections if s.lines]
        if not sections:
            return "", ""
        head, tail = ("<memory>", "</memory>") if fmt == "xml" else ("## Memory", "")
        stable_parts, dynamic_parts = [head], []
        for s in sections:
            target = stable_parts if s.title == PROFILE else dynamic_parts
            if fmt == "xml":
                tag = s.title.lower().replace(" & ", "_").replace(" ", "_")
                target.extend([f"<{tag}>", *(escape(line, quote=False) for line in s.lines), f"</{tag}>"])
            else:
                target.extend([f"### {s.title}", *(f"- {line}" for line in s.lines)])
        if tail:
            dynamic_parts.append(tail)
        return "\n".join(stable_parts), "\n".join(dynamic_parts)

    def __str__(self) -> str:
        return self.render()

    def to_dict(self) -> Dict[str, object]:
        return {
            "query": self.query, "token_estimate": self.token_estimate, "max_tokens": self.max_tokens,
            "truncated": self.truncated, "entities": self.entities,
            "sections": [{"title": s.title, "lines": s.lines, "memory_ids": s.memory_ids} for s in self.sections],
        }


PROFILE = "Profile"


def _ago(seconds: float) -> str:
    seconds = max(0.0, seconds)
    for unit, size in (("d", 86_400), ("h", 3_600), ("m", 60)):
        if seconds >= size:
            return f"{int(seconds // size)}{unit} ago"
    return "just now"


class ContextBuilder:
    """Builds Context Cards.

    Args:
        token_counter: ``fn(text) -> int``; pass your model's tokenizer for exact budgets.
        max_line_chars: Long memories are truncated to this many characters.
        relative_cutoff: Drop retrieved memories scoring below this fraction of the best
            match. Trims weak, token-wasting tail results (0 disables). Procedural memories
            are exempt (see ``RetrievalConfig.cutoff_exempt_types``).
    """

    def __init__(self, token_counter: Callable[[str], int] = estimate_tokens, max_line_chars: int = 400,
                 relative_cutoff: float = 0.5) -> None:
        self.count = token_counter
        self.max_line_chars = max_line_chars
        self.relative_cutoff = relative_cutoff

    def build(self, memory: ScopedMemory, query: str, *, max_tokens: int = 800, k: int = 12,
              graph_depth: int = 1, recent_turns: int = 0, include_working: bool = True,
              memory_types: Optional[Sequence[MemoryType]] = None,
              min_relevance: Optional[float] = None, relative_cutoff: Optional[float] = None,
              since: Any = None, until: Any = None) -> ContextCard:
        now = memory._now()
        # Priority order: the budget is spent top to bottom.
        titles = [PROFILE, "Current state", "Definitions", "Procedures & instructions", "Facts", "Past events",
                  "Recent conversation"]
        items: Dict[str, List[Tuple[str, Optional[str]]]] = {t: [] for t in titles}
        profile, state, definitions, procedures, facts, events, conversation = titles
        seen: set = set()

        def put(section: str, line: str, record_id: Optional[str] = None) -> None:
            if record_id:
                if record_id in seen:
                    return
                seen.add(record_id)
            items[section].append((truncate(line, self.max_line_chars), record_id))

        pinned = memory._pinned()
        for record in sorted(pinned, key=lambda r: (r.created_at, r.id)):  # deterministic: cache-friendly
            put(profile, record.content, record.id)

        if include_working:
            for key, value in memory.working.scratch.items():
                put(state, f"{key}: {value}")
            for turn in memory.working.turns(recent_turns) if recent_turns else []:
                put(conversation, f"{turn.role}: {turn.content}")

        for term in memory.glossary.find_terms(query):
            put(definitions, str(term), term.id)

        entities = memory.graph.find_entities(query)
        for entity in entities:
            for rel in memory.graph.facts_about(entity, depth=graph_depth, limit=10):
                put(facts, rel.sentence, rel.id)

        cutoff = self.relative_cutoff if relative_cutoff is None else relative_cutoff
        results: List[SearchResult] = memory._search(query, k=k, memory_types=memory_types,
                                                     min_relevance=min_relevance, relative_cutoff=cutoff,
                                                     since=since, until=until, touch=False)
        for result in results:
            record = result.record
            if record.memory_type == MemoryType.PROCEDURAL:
                put(procedures, self._procedure_line(record), record.id)
            elif record.memory_type == MemoryType.EPISODIC:
                tag = f"[{record.kind}] " if record.kind == "failure" else ""
                seen_at = record.metadata.get("last_seen", record.created_at)
                times = int(record.metadata.get("occurrences", 1))
                repeat = f" (x{times})" if times > 1 else ""
                put(events, f"({_ago(now - seen_at)}) {tag}{record.content}{repeat}", record.id)
            elif record.kind == "definition" and record.metadata.get("glossary"):
                put(definitions, f"{record.content}", record.id)
            else:
                put(facts, record.content, record.id)

        card = self._fit(query, items, max_tokens, entities)
        memory._mem._touch(card.memory_ids, now)
        memory._traces.log("context", memory.scope, query=query, memory_ids=card.memory_ids,
                           token_estimate=card.token_estimate, truncated=card.truncated)
        return card

    def _procedure_line(self, record: MemoryRecord) -> str:
        steps = record.metadata.get("steps")
        if not steps:
            return record.content
        name = record.metadata.get("name", "procedure")
        when = record.metadata.get("when")
        head = f"{name}" + (f" (use when: {when})" if when else "")
        return head + ": " + " ".join(f"{i}) {s}" for i, s in enumerate(steps, 1))

    def _fit(self, query: str, items: Dict[str, List[Tuple[str, Optional[str]]]], max_tokens: int,
             entities: List[str]) -> ContextCard:
        """Greedy fill in priority order; a line that does not fit is skipped, smaller later ones may still fit."""
        used = self.count("## Memory")
        out: List[CardSection] = []
        truncated = False
        for title, lines in items.items():
            kept = CardSection(title)
            header_cost = self.count(f"### {title}")
            for line, record_id in lines:
                cost = self.count(f"- {line}") + (0 if kept.lines else header_cost)
                if used + cost > max_tokens:
                    truncated = True
                    continue
                used += cost
                kept.lines.append(line)
                if record_id:
                    kept.memory_ids.append(record_id)
            out.append(kept)
        return ContextCard(query=query, sections=out, token_estimate=used if any(s.lines for s in out) else 0,
                           max_tokens=max_tokens, entities=entities, truncated=truncated)
