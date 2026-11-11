"""Memory routing: decide which store an incoming piece of text belongs to.

The default :class:`RuleRouter` is fast, free and deterministic. For higher accuracy,
wrap a small fine-tuned classifier or an LLM call in :class:`CallableRouter`; the
traces smart_memory records (``route`` and ``route_correction`` events) are exactly
the labelled data you need to fine-tune that classifier.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple, Union, runtime_checkable

from .types import MemoryType, RoutingDecision


@runtime_checkable
class Router(Protocol):
    def route(self, text: str, metadata: Optional[Mapping[str, Any]] = None) -> RoutingDecision: ...


def _rx(*patterns: str) -> List[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


# Each rule: (pattern, weight, kind). Kinds become MemoryRecord.kind.
_PROCEDURAL: List[Tuple[re.Pattern, float, str]] = [
    *[(p, 2.0, "procedure") for p in _rx(
        r"^\s*(step\s*\d+|\d+[.)]\s)", r"\bstep\s*\d+\b", r"\b(first|1st)\b.*\b(then|next|after that|finally)\b",
        r"\bhow to\b", r"\b(workflow|playbook|runbook|procedure|checklist|recipe)\b",
        r"^\s*to\s+\w+.*,\s*(first\s+)?(call|use|run|open|send|check|create|go|click)\b",
    )],
    *[(p, 2.0, "instruction") for p in _rx(
        r"^\s*(always|never|do not|don't|make sure|ensure|remember to)\b",
        r"\b(you|the agent|the assistant) (should|must|must not|should not|shouldn't)\b",
        r"^\s*when(ever)?\b.*,\s*(always\s+|never\s+)?(call|use|run|respond|reply|ask|escalate|check|send)\b",
        r"\b(respond|reply|answer|write) (in|with|using)\b",
    )],
]

_EPISODIC: List[Tuple[re.Pattern, float, str]] = [
    *[(p, 2.5, "failure") for p in _rx(
        r"\b(failed|failure|error(ed)?|exception|timed? ?out|crash(ed)?|broke|rejected|denied|bug)\b",
    )],
    *[(p, 1.5, "event") for p in _rx(
        r"\b(yesterday|today|tonight|this morning|last (week|month|night|time|session)|earlier|ago)\b",
        r"\b(on|at) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b\d{4}-\d{2}-\d{2}\b", r"\b\d{1,2}:\d{2}\b",
        r"\b(asked|requested|called|ran|executed|tried|attempted|completed|succeeded|sent|received|booked|"
        r"ordered|returned|visited|met|said|told|decided|deployed|merged|opened|closed|resolved)\b",
        r"\b(user|customer|agent|we|i) (asked|said|wanted|tried|called|reported|mentioned)\b",
        r"\btool call\b",
    )],
]

_SEMANTIC: List[Tuple[re.Pattern, float, str]] = [
    *[(p, 2.0, "preference") for p in _rx(
        r"\b(prefers?|preferred|likes?|loves?|hates?|dislikes?|favou?rite|allergic|wants? (to|a|the))\b",
    )],
    *[(p, 2.0, "definition") for p in _rx(
        r"\b(means|is defined as|refers to|stands for|definition of)\b",
    )],
    *[(p, 1.0, "fact") for p in _rx(
        r"\b(is|are|has|have|was born|lives in|works (at|as|for)|owns|uses|belongs to)\b",
        r"\bmy (name|email|role|job|company|team|timezone|birthday) is\b",
    )],
]

# Hindi / Marathi cues. Python's \b is unreliable around Devanagari vowel signs, so words
# are delimited by whitespace or punctuation (including the danda "।") instead.
_D = r"(?:^|(?<=[\s,.!?।\"'(]))"
_E = r"(?=$|[\s,.!?।\"')])"


def _ind(*alternatives: str) -> re.Pattern:
    return re.compile(_D + "(?:" + "|".join(alternatives) + ")" + _E)


_PROCEDURAL += [
    (_ind("हमेशा", "कभी नहीं", "कभी भी नहीं", "ज़रूर", "जरूर", "नेहमी", "कधीही", "करना चाहिए", "करू नका",
          "मत करो", "मत करना", "करावे", "करावा"), 2.0, "instruction"),
    (_ind("सबसे पहले", "चरण", "स्टेप", "प्रक्रिया", "आधी", "नंतर", "फिर"), 1.0, "procedure"),
]
_EPISODIC += [
    (_ind("फेल", "असफल", "विफल", "त्रुटि", "एरर", "अयशस्वी", "अपयशी", "बंद पड़", "टाइमआउट"), 2.5, "failure"),
    (_ind("कल", "आज", "परसों", "पिछले हफ्ते", "पिछले महीने", "काल", "परवा", "गेल्या आठवड्यात", "पूछा", "कहा",
          "भेजा", "किया", "गया", "गई", "गए", "हुआ", "हुई", "केले", "केला", "पाठवले", "विचारले", "झाले"),
     1.5, "event"),
]
_SEMANTIC += [
    (_ind("पसंद", "नापसंद", "आवडते", "आवडतो", "आवडत", "प्रिय", "चाहिए", "हवे", "हवा"), 2.0, "preference"),
    (_ind("मतलब", "अर्थ", "म्हणजे"), 2.0, "definition"),
    (_ind("रहता", "रहती", "रहते", "राहतो", "राहते", "काम करता", "काम करती", "करतो", "करते"), 1.0, "fact"),
]

_IMPORTANCE_BOOST = [*_rx(r"\b(important|critical|remember|never forget|crucial|must|always|urgent)\b"),
                     _ind("ज़रूरी", "जरूरी", "महत्वपूर्ण", "याद रखें", "महत्त्वाचे", "लक्षात ठेवा", "हमेशा", "नेहमी")]

_BASE_IMPORTANCE = {
    "failure": 0.7, "instruction": 0.75, "procedure": 0.7, "preference": 0.7,
    "definition": 0.65, "fact": 0.5, "event": 0.4, "note": 0.5,
}


def _score(text: str, rules: List[Tuple[re.Pattern, float, str]]) -> Tuple[float, str]:
    total = 0.0
    kind_scores: Dict[str, float] = {}
    for pattern, weight, kind in rules:
        if pattern.search(text):
            total += weight
            kind_scores[kind] = kind_scores.get(kind, 0.0) + weight
    best_kind = max(kind_scores, key=kind_scores.__getitem__) if kind_scores else ""
    return total, best_kind


class RuleRouter:
    """Heuristic router. Ties and unmatched text default to semantic memory."""

    name = "rules"

    def route(self, text: str, metadata: Optional[Mapping[str, Any]] = None) -> RoutingDecision:
        candidates = {
            MemoryType.PROCEDURAL: _score(text, _PROCEDURAL),
            MemoryType.EPISODIC: _score(text, _EPISODIC),
            MemoryType.SEMANTIC: _score(text, _SEMANTIC),
        }
        # Multi-line numbered/bulleted lists are almost always procedures.
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) >= 2 and sum(bool(re.match(r"\s*(\d+[.)]|[-*])\s", ln)) for ln in lines) >= 2:
            score, _ = candidates[MemoryType.PROCEDURAL]
            candidates[MemoryType.PROCEDURAL] = (score + 3.0, "procedure")

        ranked = sorted(candidates.items(), key=lambda kv: kv[1][0], reverse=True)
        (best_type, (best_score, best_kind)), (_, (second_score, _)) = ranked[0], ranked[1]
        if best_score == 0 or best_score == second_score and best_type != MemoryType.SEMANTIC:
            semantic_kind = candidates[MemoryType.SEMANTIC][1] or "fact"
            best_type, best_kind = MemoryType.SEMANTIC, semantic_kind
            confidence = 0.4 if best_score == 0 else 0.5
        else:
            confidence = min(0.95, 0.5 + (best_score - second_score) / (2 * best_score))

        importance = _BASE_IMPORTANCE.get(best_kind, 0.5)
        if any(p.search(text) for p in _IMPORTANCE_BOOST):
            importance += 0.15
        return RoutingDecision(
            memory_type=best_type, kind=best_kind or "note", importance=importance,
            confidence=confidence, reason=f"scores={ {k.value: v[0] for k, v in candidates.items()} }",
            router=self.name,
        )


RouterOutput = Union[RoutingDecision, MemoryType, str, Tuple[Union[MemoryType, str], float]]


class CallableRouter:
    """Route with any function: a fine-tuned small model, an LLM call, a remote service.

    ``fn(text)`` may return a :class:`RoutingDecision`, a memory type (enum or string such
    as ``"episodic"``), or a ``(memory_type, confidence)`` tuple. When ``fn`` raises, returns
    something unparseable, or its confidence is below ``min_confidence``, the ``fallback``
    router decides instead, so a flaky model never blocks a write.

    ``escalate_below`` turns this into a cascade: the cheap fallback router runs first and
    ``fn`` is only called when the fallback's confidence is below that value. With an LLM
    router this typically removes most model calls on writes.
    """

    def __init__(self, fn: Callable[[str], RouterOutput], fallback: Optional[Router] = None,
                 min_confidence: float = 0.5, name: str = "callable",
                 escalate_below: Optional[float] = None) -> None:
        self._fn = fn
        self.fallback = fallback or RuleRouter()
        self.min_confidence = min_confidence
        self.name = name
        self.escalate_below = escalate_below
        self.model_calls = 0

    def route(self, text: str, metadata: Optional[Mapping[str, Any]] = None) -> RoutingDecision:
        fallback = self.fallback.route(text, metadata)
        if self.escalate_below is not None and fallback.confidence >= self.escalate_below:
            return fallback
        self.model_calls += 1
        try:
            out = self._fn(text)
            if isinstance(out, RoutingDecision):
                decision = out
            elif isinstance(out, tuple):
                decision = RoutingDecision(MemoryType.coerce(out[0]), confidence=float(out[1]))
            else:
                decision = RoutingDecision(MemoryType.coerce(out))
        except Exception as exc:  # noqa: BLE001 - any model failure falls back
            fallback.reason = f"fallback after error: {exc!r}"
            return fallback
        if decision.memory_type == MemoryType.WORKING or decision.confidence < self.min_confidence:
            fallback.reason = f"fallback: {self.name} returned {decision.memory_type.value}@{decision.confidence:.2f}"
            return fallback
        if decision.router == "manual":
            decision.router = self.name
        # Borrow kind/importance from the rules when the model only gave a type.
        if decision.kind == "note" and decision.memory_type == fallback.memory_type:
            decision.kind, decision.importance = fallback.kind, fallback.importance
        return decision
