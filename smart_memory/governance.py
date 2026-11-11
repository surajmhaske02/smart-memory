"""Data governance: redact personal data before it is stored, and enforce retention periods.

    mem = Memory("agent.db", redact=True,
                 retention=RetentionPolicy({"episodic": 90, "kind:turn": 7}, action="delete"))

Redaction runs before embedding and storage, so raw identifiers never reach the database,
the vectors, the traces or a Context Card.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Pattern, Tuple, Union

from .types import MemoryRecord, MemoryType

DAY = 86_400.0


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_VERHOEFF_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
               [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
               [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
               [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_VERHOEFF_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
               [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
               [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]


def _aadhaar_ok(match: str) -> bool:
    """Aadhaar numbers carry a Verhoeff check digit; random 12-digit numbers rarely pass."""
    c = 0
    for i, ch in enumerate(reversed(re.sub(r"\D", "", match))):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def _card_ok(match: str) -> bool:
    digits = re.sub(r"\D", "", match)
    return 13 <= len(digits) <= 19 and _luhn_ok(digits)


# (label, pattern, extra validator). Order matters: more specific patterns run first.
_BUILTIN: List[Tuple[str, str, Optional[Callable[[str], bool]]]] = [
    ("EMAIL", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", None),
    ("SECRET", r"\b(?:sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|xox[abpr]-[A-Za-z0-9-]{10,})",
     None),
    ("CARD", r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)", _card_ok),
    ("AADHAAR", r"(?<!\d)(?<!\d[ -])[2-9]\d{3}([ -]?)\d{4}\1\d{4}(?![ -]?\d)", _aadhaar_ok),
    ("PAN", r"\b[A-Z]{5}\d{4}[A-Z]\b", None),
    ("SSN", r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)", None),
    ("IBAN", r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,4})?\b", None),
    # India: +91 98765 43210 / 9876543210; international: +44 20 7946 0958; US: (555) 123-4567
    ("PHONE", r"(?<![\w+])(?:\+91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?!\d)", None),
    ("PHONE", r"(?<![\w+])\+\d{1,3}(?:[ -]?\(?\d{1,5}\)?){2,5}(?!\d)", None),
    ("PHONE", r"(?<![\w+])\(?\d{3}\)?[ -.]\d{3}[ -.]\d{4}(?!\d)", None),
    ("IP", r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)", None),
]


@dataclass
class Redactor:
    """Replace personal data with typed placeholders such as ``[EMAIL]`` or ``[CARD]``.

    Args:
        kinds: Built-in detectors to use (default: all). Names: EMAIL, SECRET, CARD, AADHAAR,
            PAN, SSN, IBAN, PHONE, IP.
        custom: Extra ``{"LABEL": regex}`` detectors, applied before the built-ins.
    """

    kinds: Optional[Iterable[str]] = None
    custom: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        wanted = {k.upper() for k in self.kinds} if self.kinds is not None else None
        unknown = (wanted or set()) - {label for label, _, _ in _BUILTIN}
        if unknown:
            raise ValueError(f"Unknown redaction kinds: {sorted(unknown)}")
        self._rules: List[Tuple[str, Pattern[str], Optional[Callable[[str], bool]]]] = [
            (label.upper(), re.compile(rx), None) for label, rx in self.custom.items()
        ] + [(label, re.compile(rx), check) for label, rx, check in _BUILTIN if wanted is None or label in wanted]

    def redact(self, text: str) -> Tuple[str, List[str]]:
        """Returns ``(clean_text, labels_found)``."""
        found: List[str] = []
        for label, pattern, check in self._rules:
            def repl(m: re.Match[str], label: str = label, check=check) -> str:
                if check is not None and not check(m.group(0)):
                    return m.group(0)
                found.append(label)
                return f"[{label}]"
            text = pattern.sub(repl, text)
        return text, sorted(set(found))

    def __call__(self, text: str) -> str:
        return self.redact(text)[0]


RetentionKey = Union[str, MemoryType]


@dataclass
class RetentionPolicy:
    """How long memories live. Keys are memory types (``"episodic"``) or kinds (``"kind:turn"``).

    A kind rule beats a type rule. Pinned memories are never expired. ``action`` is
    ``"archive"`` (soft, recoverable) or ``"delete"`` (permanent).

    >>> RetentionPolicy({"episodic": 90, "kind:turn": 7, "kind:failure": 365}, action="archive")
    """

    days: Mapping[RetentionKey, float]
    action: str = "archive"

    def __post_init__(self) -> None:
        if self.action not in ("archive", "delete"):
            raise ValueError("action must be 'archive' or 'delete'")
        self._by_type: Dict[MemoryType, float] = {}
        self._by_kind: Dict[str, float] = {}
        for key, value in self.days.items():
            if value <= 0:
                raise ValueError(f"retention for {key!r} must be > 0 days")
            if isinstance(key, str) and key.startswith("kind:"):
                self._by_kind[key[5:]] = float(value)
            else:
                self._by_type[MemoryType.coerce(key)] = float(value)

    def limit_days(self, record: MemoryRecord) -> Optional[float]:
        if record.kind in self._by_kind:
            return self._by_kind[record.kind]
        return self._by_type.get(record.memory_type)

    def expired(self, record: MemoryRecord, now: float) -> bool:
        if record.metadata.get("pinned"):
            return False
        days = self.limit_days(record)
        if days is None:
            return False
        last = max(record.created_at, float(record.metadata.get("last_seen", 0) or 0))
        return now - last > days * DAY
