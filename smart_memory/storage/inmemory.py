"""Pure-Python dict backend. Handy for tests and as a reference for writing new backends."""

from __future__ import annotations

import copy
import threading
from typing import Any, Dict, Iterator, List, Optional, Sequence

from ..types import MemoryRecord, MemoryType, Scope, Status
from .base import PermissionDenied, StorageBackend


class InMemoryStore(StorageBackend):
    def __init__(self) -> None:
        self._records: Dict[str, MemoryRecord] = {}
        self._traces: List[Dict[str, Any]] = []
        self._meta: Dict[str, str] = {}
        self._lock = threading.RLock()

    def add(self, record: MemoryRecord) -> None:
        with self._lock:
            if record.id in self._records:
                raise ValueError(f"Duplicate memory id {record.id}")
            self._records[record.id] = copy.deepcopy(record)

    def get(self, record_id: str, scope: Scope) -> Optional[MemoryRecord]:
        with self._lock:
            record = self._records.get(record_id)
            return copy.deepcopy(record) if record and scope.can_see(record) else None

    def update(self, record: MemoryRecord, scope: Scope) -> None:
        with self._lock:
            existing = self._records.get(record.id)
            if existing is None:
                raise KeyError(record.id)
            if not scope.can_write(existing):
                raise PermissionDenied(f"Scope {scope} cannot modify memory {record.id}")
            self._records[record.id] = copy.deepcopy(record)

    def delete(self, record_id: str, scope: Scope) -> bool:
        with self._lock:
            existing = self._records.get(record_id)
            if existing is None:
                return False
            if not scope.can_write(existing):
                raise PermissionDenied(f"Scope {scope} cannot delete memory {record_id}")
            del self._records[record_id]
            return True

    def query(self, scope: Scope, *, memory_types: Optional[Sequence[MemoryType]] = None,
              kinds: Optional[Sequence[str]] = None, thread_id: Optional[str] = None,
              key: Optional[str] = None, statuses: Sequence[Status] = (Status.ACTIVE,),
              owned_only: bool = False, limit: Optional[int] = None) -> List[MemoryRecord]:
        allowed = scope.can_write if owned_only else scope.can_see
        with self._lock:
            out = [
                copy.deepcopy(r) for r in self._records.values()
                if allowed(r)
                and (memory_types is None or r.memory_type in memory_types)
                and (kinds is None or r.kind in kinds)
                and (thread_id is None or r.thread_id == thread_id)
                and (key is None or r.key == key)
                and r.status in statuses
            ]
        out.reverse()  # stable sort below then breaks created_at ties newest-inserted first
        out.sort(key=lambda r: r.created_at, reverse=True)
        return out[:limit] if limit is not None else out

    def touch(self, record_ids: Sequence[str], timestamp: float) -> None:
        with self._lock:
            for rid in record_ids:
                record = self._records.get(rid)
                if record:
                    record.access_count += 1
                    record.last_accessed_at = timestamp

    def iter_all(self) -> Iterator[MemoryRecord]:
        with self._lock:
            records = [copy.deepcopy(r) for r in self._records.values()]
        return iter(records)

    def add_trace(self, event: str, timestamp: float, scope: Scope, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._traces.append({"ts": timestamp, "event": event, "user_id": scope.user_id,
                                 "agent_id": scope.agent_id, "thread_id": scope.thread_id,
                                 "payload": copy.deepcopy(payload)})

    def iter_traces(self, event: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        with self._lock:
            traces = [copy.deepcopy(t) for t in self._traces if event is None or t["event"] == event]
        return iter(traces)

    def get_meta(self, key: str) -> Optional[str]:
        return self._meta.get(key)

    def set_meta(self, key: str, value: str) -> None:
        self._meta[key] = value
