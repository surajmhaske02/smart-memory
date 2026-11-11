import asyncio
import inspect
import os

import pytest

from smart_memory import Memory
from smart_memory.aio import AsyncMemory, AsyncScopedMemory


def test_async_api_round_trip(clock):
    async def main():
        async with AsyncMemory(clock=clock) as mem:
            alice = mem.scope(user_id="alice", agent_id="support", thread_id="t1")
            assert isinstance(alice, AsyncScopedMemory) and alice.scope.user_id == "alice"
            r = await alice.add("Alice prefers email over phone calls")
            await alice.add_many(["Alice lives in Pune", "Always confirm the order number"])
            await alice.record_event("Refund API timed out", outcome="failure")
            await alice.run(lambda m: m.graph.relate("Alice", "customer_of", "Acme"))
            hits = await alice.search("email phone")
            assert hits[0].id == r.id
            card = await alice.context("refund for Alice")
            assert "Refund API timed out" in card.render()
            assert await alice.feedback(r.id) == 1
            assert (await alice.get(r.id)).importance > 0.7
            await alice.add_turn("user", "hello")
            assert len(alice.working) == 1
            assert (await alice.end_session())[0].kind == "turn"
            assert (await mem.stats())["users"] == 1
            assert await alice.with_thread("t2").forget_user() >= 4
            assert mem.traces is mem.sync.traces

    asyncio.run(main())


def test_async_calls_run_concurrently_off_the_event_loop(clock):
    async def main():
        mem = AsyncMemory(memory=Memory(clock=clock))
        s = mem.scope("u")
        await s.add_many([f"fact {i} about topic {i % 5}" for i in range(200)])
        ticks = 0

        async def ticker():
            nonlocal ticks
            for _ in range(5):
                ticks += 1
                await asyncio.sleep(0)

        results = await asyncio.gather(*(s.search(f"topic {i % 5}") for i in range(20)), ticker())
        assert all(results[:20]) and ticks == 5  # the loop kept running while searches ran in threads
        await mem.close()

    asyncio.run(main())


def test_async_methods_keep_docs():
    assert "Store a memory" in AsyncScopedMemory.add.__doc__
    assert inspect.iscoroutinefunction(AsyncScopedMemory.search)


@pytest.mark.skipif(not os.environ.get("SMART_MEMORY_TEST_FASTEMBED"),
                    reason="downloads a model; set SMART_MEMORY_TEST_FASTEMBED=1 to run")
def test_fastembed_matches_meaning():  # pragma: no cover - optional dependency
    from smart_memory.embeddings import FastEmbedEmbedder

    mem = Memory(embedder=FastEmbedEmbedder())
    s = mem.scope("u")
    s.add_many(["Priya purchased a new laptop last month", "The board meeting is in March"])
    assert s.search("what did Priya buy?")[0].content.startswith("Priya purchased")
