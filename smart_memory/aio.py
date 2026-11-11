"""Async API for FastAPI, async agent frameworks and anything else running on an event loop.

    from smart_memory.aio import AsyncMemory

    mem = AsyncMemory("agent.db")
    alice = mem.scope(user_id="alice", agent_id="support")
    await alice.add("Alice prefers email")
    card = await alice.context("How do I contact Alice?")

Every call runs the synchronous engine in a worker thread (``asyncio.to_thread``), so the
event loop never blocks on SQLite, embeddings or ranking. Searches from many coroutines
run concurrently; writes are serialized by the engine's lock.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable, Optional, TypeVar

from .memory import Memory, ScopedMemory

T = TypeVar("T")

# ScopedMemory methods exposed as coroutines.
_SCOPED_METHODS = (
    "add", "add_many", "record_event", "add_procedure", "procedure", "record_outcome",
    "search", "context", "feedback", "refine", "apply_retention", "reclassify",
    "get", "update", "pin", "delete", "all", "history", "forget_user", "add_turn", "end_session",
)
_MEMORY_METHODS = ("reembed", "stats", "export_jsonl", "export_audit", "apply_retention", "cache_info")


def _wrap(name: str, sync_cls: type) -> Callable[..., Any]:
    target = getattr(sync_cls, name)

    @functools.wraps(target)
    async def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(getattr(self.sync, name), *args, **kwargs)

    return method


class AsyncScopedMemory:
    """Coroutine version of :class:`ScopedMemory`. The synchronous object is ``.sync``."""

    def __init__(self, scoped: ScopedMemory) -> None:
        self.sync = scoped

    @property
    def scope(self):  # noqa: ANN201 - mirrors ScopedMemory.scope
        return self.sync.scope

    @property
    def working(self):  # noqa: ANN201 - in-process and cheap, so it stays synchronous
        return self.sync.working

    def with_thread(self, thread_id: Optional[str]) -> AsyncScopedMemory:
        return AsyncScopedMemory(self.sync.with_thread(thread_id))

    async def run(self, fn: Callable[[ScopedMemory], T]) -> T:
        """Run any synchronous operation off the event loop, e.g. graph or glossary calls:

        ``await alice.run(lambda m: m.graph.relate("Alice", "customer_of", "Acme"))``
        """
        return await asyncio.to_thread(fn, self.sync)

    def __repr__(self) -> str:
        return f"AsyncScopedMemory({self.sync.scope})"


for _name in _SCOPED_METHODS:
    setattr(AsyncScopedMemory, _name, _wrap(_name, ScopedMemory))


class AsyncMemory:
    """Coroutine version of :class:`Memory`. Takes the same arguments, or an existing ``memory=``."""

    def __init__(self, *args: Any, memory: Optional[Memory] = None, **kwargs: Any) -> None:
        self.sync = memory if memory is not None else Memory(*args, **kwargs)

    def scope(self, user_id: Optional[str] = None, agent_id: Optional[str] = None,
              thread_id: Optional[str] = None) -> AsyncScopedMemory:
        return AsyncScopedMemory(self.sync.scope(user_id, agent_id, thread_id))

    @property
    def traces(self):  # noqa: ANN201
        return self.sync.traces

    async def close(self) -> None:
        await asyncio.to_thread(self.sync.close)

    async def __aenter__(self) -> AsyncMemory:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


for _name in _MEMORY_METHODS:
    setattr(AsyncMemory, _name, _wrap(_name, Memory))
