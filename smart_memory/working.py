"""Working memory: the short-lived state of one conversation thread (turns + scratchpad)."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional


@dataclass(frozen=True)
class Turn:
    role: str
    content: str
    timestamp: float


class WorkingMemory:
    """Bounded buffer of recent turns plus a key/value scratchpad (current goal, plan, variables).

    It is in-process and per thread. Call :meth:`ScopedMemory.end_session` to move what
    matters into durable episodic memory.
    """

    def __init__(self, max_turns: int = 20) -> None:
        self._turns: Deque[Turn] = deque(maxlen=max_turns)
        self._scratch: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def add_turn(self, role: str, content: str, timestamp: float = 0.0) -> None:
        with self._lock:
            self._turns.append(Turn(role, content, timestamp))

    def turns(self, last_n: Optional[int] = None) -> List[Turn]:
        with self._lock:
            items = list(self._turns)
        return items[-last_n:] if last_n else items

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._scratch[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._scratch.get(key, default)

    def pop(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._scratch.pop(key, default)

    @property
    def scratch(self) -> Dict[str, Any]:
        return dict(self._scratch)

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()
            self._scratch.clear()

    def transcript(self, last_n: Optional[int] = None) -> str:
        return "\n".join(f"{t.role}: {t.content}" for t in self.turns(last_n))

    def __len__(self) -> int:
        return len(self._turns)
