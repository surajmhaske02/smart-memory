import json

import pytest

from smart_memory import (
    EmbedderMismatchError,
    FunctionEmbedder,
    HashingEmbedder,
    InMemoryStore,
    Memory,
    MemoryType,
    PermissionDenied,
    Status,
)


def test_add_routes_automatically(alice):
    fact = alice.add("Alice prefers email over phone calls")
    event = alice.add("The refund API call failed with a timeout yesterday")
    rule = alice.add("Always confirm the order number before issuing a refund")
    assert (fact.memory_type, fact.kind) == (MemoryType.SEMANTIC, "preference")
    assert (event.memory_type, event.kind) == (MemoryType.EPISODIC, "failure")
    assert rule.memory_type == MemoryType.PROCEDURAL


def test_add_explicit_type_and_overrides(alice):
    r = alice.add("Some note", "episodic", kind="custom", importance=0.9, metadata={"src": "crm"})
    assert (r.memory_type, r.kind, r.importance, r.metadata["src"]) == (MemoryType.EPISODIC, "custom", 0.9, "crm")
    with pytest.raises(ValueError):
        alice.add("   ")
    with pytest.raises(ValueError):
        alice.add("x", "working")
    with pytest.raises(ValueError):
        alice.add("x", "nonsense")


def test_thread_provenance(alice):
    event = alice.add("User asked about invoices today", "episodic")
    fact = alice.add("Alice lives in Pune", "semantic")
    assert event.thread_id == "t1"
    assert fact.thread_id is None and fact.metadata["source_thread"] == "t1"


def test_semantic_memory_visible_across_threads(mem):
    t1 = mem.scope("alice", "support", "t1")
    t2 = t1.with_thread("t2")
    t1.add("Alice is allergic to peanuts")
    t1.add("We discussed a refund for order 42 today", "episodic")
    assert t2.search("peanuts allergy")[0].content == "Alice is allergic to peanuts"
    assert t2.search("refund order 42")  # episodes from other threads are recallable...
    assert t2.search("refund order 42", thread_only=True) == []  # ...unless restricted
    with pytest.raises(ValueError):
        mem.scope("alice").search("x", thread_only=True)


def test_search_ranks_relevant_first_and_filters_unrelated(alice):
    alice.add("Alice prefers email over phone calls")
    alice.add("Alice's favourite cuisine is Thai food")
    alice.add("The quarterly board meeting is in March")
    results = alice.search("Thai food cuisine")
    assert results[0].content == "Alice's favourite cuisine is Thai food"
    assert all("board" not in r.content for r in results)
    assert alice.search("zebra quantum xylophone") == []
    assert set(results[0].signals) == {"semantic", "keyword", "recency", "importance", "reuse"}


def test_search_filters_by_type_and_kind(alice):
    alice.add("Refund requests over $500 need manager approval", "procedural")
    alice.add("Refund for order 9 failed yesterday", "episodic")
    assert {r.record.memory_type for r in alice.search("refund", memory_types=["episodic"])} == {MemoryType.EPISODIC}
    assert [r.record.kind for r in alice.search("refund", kinds=["failure"])] == ["failure"]


def test_recency_breaks_ties(alice, clock):
    old = alice.add("Deploy to production failed", "episodic")
    clock.advance(days=60)
    new = alice.add("Deploy to production failed", "episodic", metadata={"n": 2})
    ids = [r.id for r in alice.search("deploy production failed", k=2)]
    assert ids[0] == new.id
    assert old.id not in ids  # identical text is deduplicated in results


def test_importance_breaks_ties(alice):
    low = alice.add("Server backup runs nightly", importance=0.1)
    high = alice.add("Server backup runs nightly at 2am", importance=1.0)
    assert alice.search("server backup nightly", k=2)[0].id == high.id != low.id


def test_search_touches_reuse_signal(alice):
    r = alice.add("Alice uses a Mac")
    alice.search("Mac")
    alice.search("Mac")
    assert alice.get(r.id).access_count == 2


def test_key_upsert_supersedes_and_keeps_history(alice):
    v1 = alice.add("Alice lives in Pune", key="alice.city")
    same = alice.add("Alice lives in Pune", key="alice.city")
    assert same.id == v1.id  # identical content is a no-op
    v2 = alice.add("Alice lives in Berlin", key="alice.city")
    assert [r.content for r in alice.search("where does Alice live")] == ["Alice lives in Berlin"]
    history = alice.history("alice.city")
    assert [r.id for r in history] == [v2.id, v1.id]
    assert history[1].status == Status.SUPERSEDED and history[1].superseded_by == v2.id


def test_scoping_end_to_end(mem):
    alice_support = mem.scope("alice", "support")
    alice_sales = mem.scope("alice", "sales")
    bob = mem.scope("bob", "support")
    org = mem.scope()
    secret = alice_support.add("Alice's account PIN hint is her dog's name")
    alice_support.add("Alice prefers morning meetings", shared=True)
    org.glossary.define("SLA", "Service level agreement: 4 hour response for premium customers")

    assert bob.search("Alice PIN dog") == []
    assert bob.get(secret.id) is None
    assert [r.content for r in alice_sales.search("morning meetings")] == ["Alice prefers morning meetings"]
    assert alice_sales.search("PIN hint dog") == []
    assert bob.glossary.lookup("sla").definition.startswith("Service level")
    with pytest.raises(PermissionDenied):
        bob.delete(secret.id)
    sla = org.glossary.lookup("SLA")
    with pytest.raises(PermissionDenied):
        bob.delete(sla.id)


def test_feedback_adjusts_importance(alice, mem):
    r = alice.add("Alice likes green tea", importance=0.5)
    assert alice.feedback(alice.search("green tea"), helpful=True) == 1
    assert alice.get(r.id).importance == pytest.approx(0.6)
    alice.feedback(r.id, helpful=False)
    assert alice.get(r.id).importance == pytest.approx(0.45)
    alice.feedback([r], helpful=-1.0)
    assert alice.get(r.id).importance == pytest.approx(0.30)
    glossary = mem.scope().glossary.define("ARR", "Annual recurring revenue")
    assert alice.feedback(glossary.id) == 0  # visible but not writable: traced only


def test_update_reembeds_and_delete(alice):
    r = alice.add("Alice drives a Honda")
    alice.update(r.id, content="Alice drives a Tesla", importance=0.9, metadata={"verified": True})
    assert alice.search("Tesla")[0].id == r.id
    assert alice.search("Honda") == []
    got = alice.get(r.id)
    assert got.importance == 0.9 and got.metadata["verified"] is True
    assert alice.delete(r.id) is True
    assert alice.get(r.id) is None
    with pytest.raises(KeyError):
        alice.update(r.id, content="x")


def test_reclassify_logs_correction(alice, mem):
    r = alice.add("The meeting is on Friday")
    alice.reclassify(r.id, "episodic")
    assert alice.get(r.id).memory_type == MemoryType.EPISODIC
    (row,) = [x for x in mem.traces.routing_dataset() if x["text"] == "The meeting is on Friday"]
    assert row == {"text": "The meeting is on Friday", "label": "episodic", "source": "correction"}


def test_procedures_and_outcomes(alice):
    alice.add_procedure("issue_refund", ["Verify the order", "Check refund window", "Call refund API"],
                        description="Refund a customer order", when="customer asks for money back")
    v2 = alice.add_procedure("issue_refund", ["Verify order", "Call refund API"])
    assert len(alice.all("procedural")) == 1 and alice.procedure("ISSUE_REFUND").id == v2.id
    assert alice.search("customer wants refund")[0].id == v2.id
    for ok in (True, True, False):
        rec = alice.record_outcome("issue_refund", ok)
    assert rec.metadata["outcomes"] == {"success": 2, "failure": 1}
    assert 0.3 < rec.importance < 1.0
    with pytest.raises(KeyError):
        alice.record_outcome("missing", True)
    with pytest.raises(ValueError):
        alice.add_procedure("empty", [])


def test_record_event_outcomes(alice):
    failure = alice.record_event("Called pricing API; got 503", outcome="failure")
    ok = alice.record_event("Sent invoice to finance", outcome="success")
    assert failure.kind == "failure" and failure.importance > ok.importance
    assert ok.metadata["outcome"] == "success"


def test_working_memory_and_end_session(alice, mem):
    alice.add_turn("user", "I want to change my delivery address to Pune")
    alice.add_turn("assistant", "Sure, updating it now.")
    alice.working.set("goal", "update address")
    assert mem.scope("alice", "support", "t1").working.get("goal") == "update address"
    assert mem.scope("alice", "support", "t2").working.get("goal") is None
    stored = alice.end_session()
    assert [r.kind for r in stored] == ["turn"] and len(alice.working) == 0

    alice.add_turn("user", "Cancel order 77")
    (summary,) = alice.end_session(summarizer=lambda transcript: f"Summary: {transcript}")
    assert summary.kind == "session_summary" and "Cancel order 77" in summary.content


def test_all_and_forget_user(mem):
    alice = mem.scope("alice", "support")
    bob = mem.scope("bob", "support")
    alice.add("Alice likes cats")
    alice.add("Alice failed login yesterday")
    bob.add("Bob likes dogs")
    assert len(alice.all()) == 2 and len(alice.all("semantic")) == 1
    assert alice.forget_user() == 2
    assert alice.all(include_inactive=True) == []
    assert len(bob.all()) == 1


def test_persistence_and_embedder_mismatch(tmp_path):
    path = str(tmp_path / "agent.db")
    with Memory(path) as m:
        m.scope("alice").add("Alice speaks Marathi and English")
    with Memory(path) as m:
        assert m.scope("alice").search("Marathi")[0].content == "Alice speaks Marathi and English"
    with pytest.raises(EmbedderMismatchError):
        Memory(path, embedder=HashingEmbedder(dim=64))
    with Memory(path, embedder=HashingEmbedder(dim=64), on_embedder_change="reembed") as m:
        hit = m.scope("alice").search("Marathi")[0]
        assert len(hit.record.embedding) == 64


def test_custom_embedder_and_store():
    fake = FunctionEmbedder(lambda texts: [[float(len(t)), 1.0] for t in texts], dim=2, name="len")
    with Memory(store=InMemoryStore(), embedder=fake) as m:
        r = m.scope("u").add("hello world")
        assert len(r.embedding) == 2


def test_stats_and_export(mem, tmp_path):
    mem.scope("alice").add("Alice likes jazz")
    mem.scope("bob").add("Bob failed the exam yesterday")
    stats = mem.stats()
    assert stats["active_by_type"] == {"semantic": 1, "episodic": 1} and stats["users"] == 2
    out = tmp_path / "dump.jsonl"
    assert mem.export_jsonl(out) == 2
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert {r["user_id"] for r in rows} == {"alice", "bob"} and "embedding" not in rows[0]


def test_four_rs_aliases(alice):
    assert alice.retain == alice.add and alice.recall == alice.search and alice.reuse == alice.context
