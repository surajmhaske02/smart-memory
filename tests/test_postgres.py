"""PostgreSQL + pgvector backend at the Memory level. Skipped without a Postgres server.

Run with $SMART_MEMORY_TEST_PG_DSN set, or with the test-only `pgserver` package installed.
"""

import pytest
from pg_support import make_store, pg_dsn

from smart_memory import Memory, Status

pytestmark = pytest.mark.skipif(not pg_dsn(), reason="no PostgreSQL available")


@pytest.fixture
def pg_store():
    store = make_store()
    yield store
    store.drop_tables()
    store.close()


def test_memory_on_postgres_end_to_end(pg_store, clock):
    mem = Memory(store=pg_store, clock=clock)
    alice = mem.scope("alice", "support")
    alice.add("Alice prefers email, never phone calls")
    alice.add("Alice is a premium customer", pin=True)
    alice.record_event("Refund API timed out for order 4411", outcome="failure")
    alice.glossary.define("refund window", "30 days from delivery")
    alice.graph.relate("Alice", "customer_of", "Acme")
    card = alice.context("Alice wants a refund. What is the refund window?")
    text = card.render()
    for expected in ("premium customer", "30 days", "Refund API timed out", "Alice customer of Acme"):
        assert expected in text
    assert mem.scope("bob").search("Alice email") == []


def test_other_process_writes_invalidate_the_warm_index(pg_store, clock):
    from smart_memory.storage.postgres import PostgresStore

    second_connection = PostgresStore(pg_dsn(), table_prefix=pg_store._p)
    a, b = Memory(store=pg_store, clock=clock), Memory(store=second_connection, clock=clock)
    a.scope("alice").add("Alice likes jazz")
    assert b.scope("alice").search("jazz")  # b builds its warm index
    gen = second_connection.generation()
    a.scope("alice").add("Alice plays the violin")
    assert second_connection.generation() > gen
    assert b.scope("alice").search("violin")[0].content == "Alice plays the violin"
    second_connection.close()


def test_pgvector_candidate_mode(pg_store, clock):
    assert pg_store.has_pgvector
    mem = Memory(store=pg_store, clock=clock, candidate_limit=50)
    s = mem.scope("u", "a")
    s.add_many([f"Customer {i} asked about invoice {1000 + i}" for i in range(300)])
    s.add("Priya prefers WhatsApp over email")
    s.add("Priya is a premium customer", pin=True)
    hits = s.search("does Priya prefer WhatsApp or email?")
    assert hits[0].content == "Priya prefers WhatsApp over email"
    assert s.search("invoice 1042")[0].content == "Customer 42 asked about invoice 1042"
    card = s.context("message Priya")
    assert card.sections[0].title == "Profile" and "premium" in card.sections[0].lines[0]
    assert mem._indexes == {} or len(mem._indexes) == 0  # no whole-scope index was built
    # dedupe and contradictions still work without a warm index
    again = s.add("Priya prefers WhatsApp over email")
    assert again.metadata["occurrences"] == 2
    old = s.add("Priya lives in Pune")
    s.add("Priya moved to Mumbai")
    assert s.get(old.id).status == Status.SUPERSEDED


def test_find_by_metadata_and_candidates_respect_scope(pg_store):
    from smart_memory import HashingEmbedder, MemoryRecord, MemoryType, Scope

    emb = HashingEmbedder()
    texts = ["alice private note", "bob private note"]
    for text, user in zip(texts, ("alice", "bob")):
        pg_store.add(MemoryRecord(text, MemoryType.SEMANTIC, user_id=user, metadata={"pinned": True},
                                  embedding=emb.embed([text])[0]))
    alice = Scope("alice")
    assert [r.content for r in pg_store.find_by_metadata(alice, "pinned", True)] == ["alice private note"]
    found = pg_store.candidates(alice, "private note", emb.embed(["private note"])[0], limit=10)
    assert [r.content for r in found] == ["alice private note"]


def test_bad_table_prefix_rejected():
    from smart_memory.storage.postgres import PostgresStore

    with pytest.raises(ValueError):
        PostgresStore(pg_dsn(), table_prefix="x; DROP TABLE y")


def test_candidate_limit_requires_capable_backend():
    with pytest.raises(ValueError, match="candidate_limit"):
        Memory(candidate_limit=10)
