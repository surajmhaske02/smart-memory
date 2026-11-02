"""Core data types shared across smart_memory."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryType(str, Enum):
    """The four cognitive memory stores.

    WORKING memory lives in-process (per thread) and is never persisted by the
    storage backend. The other three are durable.
    """

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

    @classmethod
    def coerce(cls, value: MemoryType | str) -> MemoryType:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError:
            valid = ", ".join(m.value for m in cls)
            raise ValueError(f"Unknown memory type {value!r}. Expected one of: {valid}") from None


DURABLE_TYPES = (MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL)


class Status(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"  # replaced by a newer version (see superseded_by)
    ARCHIVED = "archived"  # forgotten by decay; recoverable, never returned by default


@dataclass(frozen=True)
class Scope:
    """Who a memory belongs to. Enforced by the storage layer on every read/write.

    Visibility rules (applied in the database query, not in prompts):

    * ``user_id`` is a hard boundary. A record is visible to a scope when its
      ``user_id`` equals the scope's, or when the record is global (``user_id=None``).
    * ``agent_id`` is a hard boundary too. Records with ``agent_id=None`` are
      shared by every agent acting for that user.
    * ``thread_id`` is *not* a boundary - it is provenance plus an optional filter,
      so an agent can recall what happened in earlier conversations.
    """

    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    thread_id: Optional[str] = None

    def with_thread(self, thread_id: Optional[str]) -> Scope:
        return replace(self, thread_id=thread_id)

    def can_see(self, record: MemoryRecord) -> bool:
        return (record.user_id is None or record.user_id == self.user_id) and (
            record.agent_id is None or record.agent_id == self.agent_id
        )

    def can_write(self, record: MemoryRecord) -> bool:
        """A scope may modify records it owns: same user, and same agent or user-shared."""
        return record.user_id == self.user_id and record.agent_id in (None, self.agent_id)


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class MemoryRecord:
    content: str
    memory_type: MemoryType
    kind: str = "note"
    id: str = field(default_factory=new_id)
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    thread_id: Optional[str] = None
    key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    embedding: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = 0.0
    last_accessed_at: Optional[float] = None
    access_count: int = 0
    status: Status = Status.ACTIVE
    superseded_by: Optional[str] = None

    def __post_init__(self) -> None:
        self.memory_type = MemoryType.coerce(self.memory_type)
        self.status = Status(self.status)
        self.importance = clamp(self.importance)
        if not self.updated_at:
            self.updated_at = self.created_at

    @property
    def scope(self) -> Scope:
        return Scope(self.user_id, self.agent_id, self.thread_id)

    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "kind": self.kind,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "thread_id": self.thread_id,
            "key": self.key,
            "metadata": dict(self.metadata),
            "importance": self.importance,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "status": self.status.value,
            "superseded_by": self.superseded_by,
        }
        if include_embedding:
            data["embedding"] = self.embedding
        return data


@dataclass
class SearchResult:
    record: MemoryRecord
    score: float
    signals: Dict[str, float] = field(default_factory=dict)

    @property
    def content(self) -> str:
        return self.record.content

    @property
    def id(self) -> str:
        return self.record.id

    def __repr__(self) -> str:
        preview = self.record.content if len(self.record.content) < 60 else self.record.content[:57] + "..."
        return f"SearchResult(score={self.score:.3f}, type={self.record.memory_type.value}, content={preview!r})"


@dataclass
class RoutingDecision:
    memory_type: MemoryType
    kind: str = "note"
    importance: float = 0.5
    confidence: float = 1.0
    reason: str = ""
    router: str = "manual"

    def __post_init__(self) -> None:
        self.memory_type = MemoryType.coerce(self.memory_type)
        self.importance = clamp(self.importance)
        self.confidence = clamp(self.confidence)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))
