"""Storage backend interface.

To add a backend (Postgres + pgvector, Oracle, MongoDB, Redis...), subclass
:class:`StorageBackend` and make it pass ``tests/test_storage_contract.py``.
Every read must apply :meth:`Scope.can_see` *inside the query* so entitlements are
enforced by the data layer, never by prompt instructions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from ..types import MemoryRecord, MemoryType, Scope, Status


class PermissionDenied(PermissionError):
    """Raised when a scope tries to modify a record it does not own."""


class StorageBackend(ABC):
    # --- memories -----------------------------------------------------------------
    @abstractmethod
    def add(self, record: MemoryRecord) -> None:
        """Insert a new record. Raises ValueError if the id already exists."""

    def add_many(self, records: Iterable[MemoryRecord]) -> None:
        for record in records:
            self.add(record)

    @abstractmethod
    def get(self, record_id: str, scope: Scope) -> Optional[MemoryRecord]:
        """Return the record if it exists *and* is visible to ``scope``."""

    @abstractmethod
    def update(self, record: MemoryRecord, scope: Scope) -> None:
        """Persist all fields of ``record``. Raises PermissionDenied unless ``scope.can_write``."""

    @abstractmethod
    def delete(self, record_id: str, scope: Scope) -> bool:
        """Hard-delete a record. Returns False if missing; raises PermissionDenied if not owned."""

    @abstractmethod
    def query(
        self,
        scope: Scope,
        *,
        memory_types: Optional[Sequence[MemoryType]] = None,
        kinds: Optional[Sequence[str]] = None,
        thread_id: Optional[str] = None,
        key: Optional[str] = None,
        statuses: Sequence[Status] = (Status.ACTIVE,),
        owned_only: bool = False,
        limit: Optional[int] = None,
    ) -> List[MemoryRecord]:
        """Visible records matching every given filter, newest first.

        ``owned_only`` narrows to records ``scope`` may write (used for upserts so a
        user-level write never supersedes a global record).
        """

    @abstractmethod
    def touch(self, record_ids: Sequence[str], timestamp: float) -> None:
        """Bump access_count and last_accessed_at (the "reuse" signal)."""

    def find_by_metadata(self, scope: Scope, key: str, value: Any) -> List[MemoryRecord]:
        """Active visible records whose ``metadata[key] == value``. Override with an indexed query."""
        return [r for r in self.query(scope) if r.metadata.get(key) == value]

    def generation(self) -> Optional[Any]:
        """A token that changes whenever stored records change (``touch`` excluded).

        ``Memory`` uses it to keep a warm in-process search index. Return ``None`` (the
        default) to disable that cache, e.g. for a backend shared with writers you can't observe.
        """
        return None

    @abstractmethod
    def iter_all(self) -> Iterator[MemoryRecord]:
        """Every record regardless of scope. Admin/maintenance use only (re-embedding, export)."""

    # --- traces -------------------------------------------------------------------
    @abstractmethod
    def add_trace(self, event: str, timestamp: float, scope: Scope, payload: Dict[str, Any]) -> None: ...

    @abstractmethod
    def iter_traces(self, event: Optional[str] = None) -> Iterator[Dict[str, Any]]: ...

    # --- key/value metadata -------------------------------------------------------
    @abstractmethod
    def get_meta(self, key: str) -> Optional[str]: ...

    @abstractmethod
    def set_meta(self, key: str, value: str) -> None: ...

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release connections. Backends without resources can ignore it."""

    def __enter__(self) -> StorageBackend:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
