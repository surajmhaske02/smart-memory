"""Cost and speed features: warm index, embedding cache, batching, router cascade, pinning, cutoff, tools."""

import json

import pytest

from smart_memory import (
    CachedEmbedder,
    CallableRouter,
    ContextBuilder,
    HashingEmbedder,
    InMemoryStore,
    Memory,
    MemoryTools,
    MemoryType,
    RetrievalConfig,
    Status,
)
from smart_memory.index import ScopeIndex
from smart_memory.storage.base import StorageBackend

FACTS = ["Alice prefers email over phone calls", "Alice lives in Pune", "Alice's favourite food is Thai curry",
         "Bob manages the billing team", "The refund window is 30 days", "Deploys happen on Tuesdays"]


class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.texts = 0

    def embed(self, texts):
        self.calls += 1
        self.texts += len(texts)
        return super().embed(texts)


def ids(results):
    return [r.id for r in results]


# --- warm index -------------------------------------------------------------------

def test_index_is_reused_and_patched_not_rebuilt(mem):
    s = mem.scope("alice", "support")
    s.add_many(FACTS)
    s.search("Pune")
    builds = mem.cache_info()["index_builds"]
    r = s.add("Alice switched her phone number")
    s.feedback(r.id)
    s.update(r.id, content="Alice switched her phone number to a new one")
    s.delete(s.search("Thai curry")[0].id)
    s.search("phone")
    assert mem.cache_info()["index_builds"] == builds  # every write patched the warm index


@pytest.mark.parametrize("store_factory", [InMemoryStore, None])
def test_cached_results_match_a_cold_rebuild(clock, store_factory, tmp_path):
    store = store_factory() if store_factory else None
    m = Memory(tmp_path / "x.db" if store is None else ":memory:", store=store, clock=clock)
    s = m.scope("alice", "support")
    s.add_many(FACTS)
    for q in ("Pune", "email phone", "refund"):
        s.search(q)
    s.add("Alice now lives in Berlin", key="city")
    s.add("Alice now lives in Mumbai", key="city")
    s.feedback(s.search("email")[0].id, helpful=False)
    s.record_event("Refund API failed", outcome="failure")
    s.refine()
    warm = {q: [(r.id, round(r.score, 6)) for r in s.search(q)] for q in ("Alice lives", "refund", "email")}
    m._indexes.clear()
    cold = {q: [(r.id, round(r.score, 6)) for r in s.search(q)] for q in ("Alice lives", "refund", "email")}
    # access counts moved between the two passes, so compare ids and order, not reuse-sensitive scores
    assert {q: [i for i, _ in v] for q, v in warm.items()} == {q: [i for i, _ in v] for q, v in cold.items()}


def test_other_scopes_indexes_are_patched_for_shared_and_global_writes(mem):
    alice_support, alice_sales = mem.scope("alice", "support"), mem.scope("alice", "sales")
    bob = mem.scope("bob", "support")
    for s in (alice_support, alice_sales, bob):
        s.search("warm up")
    alice_support.add("Alice prefers morning meetings", shared=True)
    mem.scope().glossary.define("SLA", "4 hour response for premium customers")
    assert alice_sales.search("morning meetings")
    assert not bob.search("morning meetings")
    assert bob.glossary.lookup("sla") is not None


def test_external_writer_invalidates_cache(tmp_path, clock):
    path = tmp_path / "shared.db"
    a, b = Memory(path, clock=clock), Memory(path, clock=clock)
    a.scope("alice").add("Alice likes jazz")
    assert ids(b.scope("alice").search("jazz"))  # b builds its index
    a.scope("alice").add("Alice plays the violin")  # written by another connection
    assert b.scope("alice").search("violin")[0].content == "Alice plays the violin"
    a.close()
    b.close()


def test_backend_without_generation_disables_cache(clock):
    class NoGen(InMemoryStore):
        def generation(self):
            return StorageBackend.generation(self)

    m = Memory(store=NoGen(), clock=clock)
    s = m.scope("u")
    s.add("Alice likes jazz")
    s.search("jazz")
    s.search("jazz")
    assert m.cache_info()["warm_indexes"] == 0 and m.cache_info()["index_builds"] >= 2


def test_numpy_and_pure_python_paths_agree():
    np = pytest.importorskip("numpy")  # noqa: F841
    from smart_memory.types import MemoryRecord
    emb = HashingEmbedder()
    records = [MemoryRecord(t, MemoryType.SEMANTIC, embedding=v) for t, v in zip(FACTS, emb.embed(FACTS))]
    fast = ScopeIndex.build([MemoryRecord(**{**r.__dict__}) for r in records], emb.dim, use_numpy=True)
    slow = ScopeIndex.build([MemoryRecord(**{**r.__dict__}) for r in records], emb.dim, use_numpy=False)
    fast.remove(records[1].id)
    slow.remove(records[1].id)
    q = emb.embed(["Alice email phone"])[0]
    a, b = fast.similarities(q), slow.similarities(q)
    assert a.keys() == b.keys() and all(abs(a[k] - b[k]) < 1e-5 for k in a)
    assert abs(fast.pair_similarity(0, 2) - slow.pair_similarity(0, 2)) < 1e-5
    assert fast.vector(1) is None and len(fast) == len(slow) == len(FACTS) - 1


def test_index_grows_past_initial_capacity():
    emb = HashingEmbedder(dim=32)
    from smart_memory.types import MemoryRecord
    texts = [f"memory number {i}" for i in range(40)]
    index = ScopeIndex.build([MemoryRecord(t, MemoryType.SEMANTIC, embedding=v)
                              for t, v in zip(texts, emb.embed(texts))], 32)
    assert len(index) == 40 and index.bm25("number 39")


# --- fewer model calls ------------------------------------------------------------

def test_add_many_uses_one_embedding_call_and_cache_skips_repeats(clock):
    counting = CountingEmbedder()
    m = Memory(embedder=counting, clock=clock)
    s = m.scope("u")
    s.add_many(FACTS)
    assert counting.calls == 1 and counting.texts == len(FACTS)
    s.search("Pune")
    s.search("Pune")
    s.context("Pune")
    assert counting.calls == 2  # the query was embedded once
    info = m.cache_info()
    assert info["embed_hits"] >= 2 and info["embed_misses"] == len(FACTS) + 1


def test_cached_embedder_lru_and_batch_dedupe():
    inner = CountingEmbedder()
    cache = CachedEmbedder(inner, maxsize=2)
    out = cache.embed(["a b", "a b", "c d"])
    assert inner.texts == 2 and out[0] == out[1]
    out[0][0] = 99.0  # callers get copies
    assert cache.embed(["a b"])[0][0] != 99.0
    cache.embed(["e f"])  # evicts the least recently used entry ("c d")
    cache.embed(["c d"])
    assert inner.texts == 4


def test_embed_cache_can_be_disabled(clock):
    counting = CountingEmbedder()
    m = Memory(embedder=counting, embed_cache_size=0, clock=clock)
    m.scope("u").search("x")
    m.scope("u").search("x")
    assert counting.calls == 2 and "embed_calls" not in m.cache_info()


def test_router_cascade_only_calls_model_when_rules_are_unsure():
    calls = []

    def model(text):
        calls.append(text)
        return ("episodic", 0.9)

    router = CallableRouter(model, escalate_below=0.6)
    assert router.route("Alice prefers email over phone calls").memory_type == MemoryType.SEMANTIC
    assert calls == []  # confident rules, no model call
    assert router.route("blue").memory_type == MemoryType.EPISODIC
    assert calls == ["blue"] and router.model_calls == 1


# --- smaller prompts --------------------------------------------------------------

def test_pinned_memories_open_every_card_in_stable_order(alice, clock):
    first = alice.add("Alice is a premium customer", pin=True)
    clock.advance(seconds=1)
    alice.add("Alice's preferred language is Marathi", pin=True)
    alice.add("Alice ordered a laptop", "episodic")
    card = alice.context("laptop order")
    assert card.sections[0].title == "Profile"
    assert card.sections[0].lines == ["Alice is a premium customer", "Alice's preferred language is Marathi"]
    stable, dynamic = card.render_split()
    assert stable == "## Memory\n### Profile\n- Alice is a premium customer\n- Alice's preferred language is Marathi"
    assert dynamic.startswith("### Past events") and card.render() == stable + "\n" + dynamic
    assert alice.context("something else entirely").render_split()[0] == stable  # cacheable prefix
    alice.pin(first.id, False)
    assert "premium" not in alice.context("laptop order").render_split()[0]
    xml_stable, xml_dynamic = card.render_split("xml")
    assert xml_stable.startswith("<memory>\n<profile>") and xml_dynamic.endswith("</memory>")


def test_relative_cutoff_trims_weak_results(clock):
    m = Memory(clock=clock, retrieval=RetrievalConfig(relative_cutoff=0.0))
    s = m.scope("u")
    s.add_many(["Refund policy: refunds within 30 days of delivery", "Alice mentioned a refund once",
                "The office has a refund jar for coffee"])
    loose = s.search("refund policy 30 days delivery", k=5)
    tight = s.search("refund policy 30 days delivery", k=5, relative_cutoff=0.9)
    assert len(tight) < len(loose) and tight[0].id == loose[0].id
    assert len(s.context("refund policy 30 days delivery", relative_cutoff=0.9).memory_ids) == 1


def test_default_builder_cutoff_is_configurable(clock):
    m = Memory(clock=clock, context_builder=ContextBuilder(relative_cutoff=0.0))
    assert m.context_builder.relative_cutoff == 0.0


# --- tools ------------------------------------------------------------------------

def test_tool_schemas_formats(alice):
    tools = MemoryTools(alice)
    assert tools.names == ["remember", "recall"]
    assert [t["name"] for t in tools.schemas("anthropic")] == ["remember", "recall"]
    assert "input_schema" in tools.schemas("anthropic")[0]
    openai = tools.schemas("openai")[0]
    assert openai["type"] == "function" and openai["function"]["parameters"]["required"] == ["content"]
    assert MemoryTools(alice, allow_forget=True).names[-1] == "forget"
    with pytest.raises(ValueError):
        tools.schemas("xml")


def test_tool_calls_round_trip(alice):
    tools = MemoryTools(alice, allow_forget=True)
    saved = json.loads(tools.handle("remember", {"content": "Alice lives in Pune", "key": "alice.city"}))
    assert saved["memory_type"] == "semantic"
    found = json.loads(tools.handle("recall", '{"query": "where does Alice live", "k": 100}'))
    assert found["results"][0]["content"] == "Alice lives in Pune"
    assert json.loads(tools.handle("forget", {"memory_id": saved["id"]})) == {"deleted": True}
    assert alice.get(saved["id"]) is None


def test_tool_errors_are_returned_not_raised(alice):
    tools = MemoryTools(alice)
    assert "error" in json.loads(tools.handle("forget", {"memory_id": "x"}))  # disabled by default
    assert "error" in json.loads(tools.handle("remember", {"content": " "}))
    assert "error" in json.loads(tools.handle("recall", "not json"))
    assert "error" in json.loads(tools.handle("recall", {"q": "wrong arg"}))


def test_superseded_and_archived_leave_the_index(alice):
    v1 = alice.add("Alice lives in Pune", key="city")
    alice.add("Alice lives in Berlin", key="city")
    assert v1.id not in ids(alice.search("Alice lives in Pune", k=5))
    assert alice.get(v1.id).status == Status.SUPERSEDED


def test_dedupe_never_merges_different_numbers_or_negation(clock):
    """Worst case: an embedder that maps everything to one vector (neural models come close for these)."""
    from smart_memory import FunctionEmbedder
    flat = FunctionEmbedder(lambda texts: [[1.0, 0.0] for _ in texts], dim=2, name="flat")
    m = Memory(embedder=flat, clock=clock)
    s = m.scope("u")
    s.add_many(["Order 1234 failed", "Order 1235 failed", "Alice likes tea", "Alice doesn't like tea"], "episodic")
    assert len(s.all()) == 4
    s.add("Order 1234 failed", "episodic")
    assert len(s.all()) == 4  # same numbers, same polarity: reinforced
    m2 = Memory(embedder=flat, clock=clock, dedupe_on_write=None)
    s2 = m2.scope("u")
    s2.add_many(["Invoice 77 paid", "Invoice 78 paid", "Invoice 77 paid!"], "episodic")
    assert len(s2.refine().merged) == 1


def test_same_specifics():
    from smart_memory.text import same_specifics
    assert same_specifics("Meeting at 10:00", "meeting at 10:00!")
    assert not same_specifics("Meeting at 10:00", "Meeting at 11:00")
    assert not same_specifics("total was $500", "total was $5000")
    assert not same_specifics("Alice likes tea", "Alice does not like tea")
    assert same_specifics("never call Bob", "Never call Bob again")
