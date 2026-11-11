"""Detect facts that replace earlier facts ("Alice lives in Pune" -> "Alice moved to Berlin").

Each pattern recognises a single-valued *slot* (where someone lives, who they work for,
their email...) and extracts the subject and the value. Two facts with the same slot and
subject but different values contradict each other. Add your own patterns with
:func:`register_slot`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Pattern, Tuple

_SLOTS: List[Tuple[str, Pattern[str]]] = []


def register_slot(slot: str, pattern: str) -> None:
    """Add a pattern with named groups ``subject`` and ``value`` (matched case-insensitively, full text)."""
    compiled = re.compile(pattern, re.IGNORECASE)
    if not {"subject", "value"} <= set(compiled.groupindex):
        raise ValueError("pattern needs named groups 'subject' and 'value'")
    _SLOTS.append((slot, compiled))


_S = r"(?P<subject>[^\s,.;:!?][^,.;:!?]*?)"
_V = r"(?P<value>[^,;!?]+?)"
_END = r"[\s.!।]*$"
for _slot, _pattern in [
    ("location", rf"^{_S}\s+(?:now\s+)?(?:lives|is living|resides|is based|stays|is staying)\s+in\s+{_V}{_END}"),
    ("location", rf"^{_S}\s+(?:has\s+)?(?:moved|relocated|shifted)\s+to\s+{_V}{_END}"),
    ("employer", rf"^{_S}\s+(?:now\s+)?(?:works|is working)\s+(?:at|for)\s+{_V}{_END}"),
    ("employer", rf"^{_S}\s+(?:has\s+)?(?:joined|now works at)\s+{_V}{_END}"),
    ("email", rf"^{_S}(?:'s|\s+has\s+an?)\s+(?:new\s+)?(?:email|e-mail)(?:\s+address)?(?:\s+is)?\s+{_V}{_END}"),
    ("phone", rf"^{_S}(?:'s|\s+has\s+an?)\s+(?:new\s+)?(?:phone|mobile)(?:\s+number)?(?:\s+is)?\s+{_V}{_END}"),
    ("manager", rf"^{_S}\s+(?:now\s+)?reports\s+to\s+{_V}{_END}"),
    ("plan", rf"^{_S}\s+is\s+(?:now\s+)?on\s+the\s+{_V}\s+plan{_END}"),
    ("age", rf"^{_S}\s+is\s+(?:now\s+)?(?P<value>\d{{1,3}})\s+years?\s+old{_END}"),
    ("timezone", rf"^{_S}(?:'s)?\s+(?:time\s*zone|timezone)\s+is\s+{_V}{_END}"),
    # Hindi: "<subject> <city> में रहता/रहती/रहते है/हैं"
    ("location", rf"^{_S}\s+(?P<value>\S+)\s+में\s+(?:रहता|रहती|रहते)\s+(?:है|हैं){_END}"),
    # Marathi: "<subject> <city>मध्ये/ला राहतो/राहते"
    ("location", rf"^{_S}\s+(?P<value>\S+?)(?:मध्ये)?\s+(?:राहतो|राहते|राहतात){_END}"),
]:
    register_slot(_slot, _pattern)


@dataclass(frozen=True)
class SlotFact:
    slot: str
    subject: str
    value: str

    @property
    def key(self) -> str:
        return f"slot:{self.slot}:{self.subject}"


def _norm(text: str) -> str:
    text = " ".join(text.casefold().split())
    text = re.sub(r"^(?:the|a|an)\s+", "", text)
    text = re.sub(r"\s+(?:now|अब|आता)$", "", text)
    return text.strip(" '\"")


def extract_slot(text: str) -> Optional[SlotFact]:
    """The single-valued fact ``text`` states, if it matches a known pattern."""
    text = " ".join(text.split())
    for slot, pattern in _SLOTS:
        m = pattern.match(text)
        if m:
            subject, value = _norm(m.group("subject")), _norm(m.group("value"))
            if subject and value and len(subject.split()) <= 6:
                return SlotFact(slot, subject, value)
    return None
