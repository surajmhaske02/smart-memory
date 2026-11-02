import json

from smart_memory import Memory, MemoryType, Refiner, Status


def test_dedupe_keeps_newest_and_merges_signals(alice, clock):
    old = alice.add("Alice prefers email over phone calls", importance=0.9)
    alice.search("email")  # access_count 1 on old
    clock.advance(days=1)
    new = alice.add("Alice prefers email over phone calls!", importance=0.4)
    other = alice.add("Alice lives in Pune")
    report = alice.refine()
    assert report.merged == [old.id]
    survivor = alice.get(new.id)
    assert survivor.importance == 0.9 and survivor.access_count == 1
    assert survivor.metadata["merged_from"] == [old.id]
    assert alice.get(old.id).status == Status.SUPERSEDED
    assert alice.get(other.id).status == Status.ACTIVE


def test_decay_archives_stale_unimportant_episodes_only(alice, clock):
    stale = alice.add("User said hello", "episodic", importance=0.1)
    important = alice.add("Production database was deleted by a script", "episodic", importance=0.9)
    fact = alice.add("Alice works at Acme", importance=0.1)
    clock.advance(days=120)
    fresh = alice.add("User said goodbye", "episodic", importance=0.1)
    report = alice.refine()
    assert report.archived == [stale.id]
    assert alice.get(stale.id).status == Status.ARCHIVED
    for r in (important, fact, fresh):
        assert alice.get(r.id).status == Status.ACTIVE
    assert stale.id not in [r.id for r in alice.search("user said hello")]
    assert stale.id in [r.id for r in alice.all(include_inactive=True)]


def test_promote_recurring_episodes_to_insight(alice):
    for day in ("Monday", "Tuesday", "Thursday"):
        alice.record_event(f"Pricing API call timed out on {day} during peak load", outcome="failure")
    alice.record_event("Alice ordered a laptop")
    calls = []

    def summarize(texts):
        calls.append(texts)
        return "The pricing API tends to time out under peak load; retry with backoff."

    report = alice.refine(summarizer=summarize)
    assert len(calls) == 1 and len(calls[0]) == 3
    (insight_id,) = report.promoted
    insight = alice.get(insight_id)
    assert insight.memory_type == MemoryType.SEMANTIC and insight.kind == "insight"
    assert len(insight.metadata["source_ids"]) == 3
    assert alice.refine(summarizer=summarize).promoted == []  # episodes are not promoted twice


def test_refine_ignores_keyed_and_global_records(mem):
    org = mem.scope()
    org.glossary.define("ARR", "Annual recurring revenue")
    alice = mem.scope("alice")
    alice.add("ARR: Annual recurring revenue")
    report = alice.refine()
    assert report.examined == 1 and report.merged == []


def test_refiner_is_configurable(clock):
    with Memory(clock=clock, refiner=Refiner(min_age_days=0, archive_below=0.2)) as m:
        s = m.scope("u")
        r = s.add("User said ok", "episodic", importance=0.1)
        assert s.refine().archived == [r.id]


def test_traces_cover_the_lifecycle(alice, mem, tmp_path):
    r = alice.add("Alice likes chess")
    alice.search("chess")
    alice.context("chess")
    alice.feedback(r.id)
    alice.refine()
    alice.delete(r.id)
    events = [t["event"] for t in mem.traces]
    assert events == ["route", "retrieve", "context", "feedback", "refine", "delete"]
    route = next(mem.traces.events("route"))
    assert route["payload"]["router"] == "rules" and route["user_id"] == "alice"
    out = tmp_path / "traces.jsonl"
    assert mem.traces.export_jsonl(out) == 6
    assert json.loads(out.read_text().splitlines()[0])["event"] == "route"


def test_routing_dataset_export(alice, mem, tmp_path):
    alice.add("Yesterday the export job failed")
    alice.add("Alice prefers dark mode")
    rows = mem.traces.routing_dataset(tmp_path / "routes.jsonl")
    assert {(r["label"], r["source"]) for r in rows} == {("episodic", "rules"), ("semantic", "rules")}
    assert len((tmp_path / "routes.jsonl").read_text().splitlines()) == 2


def test_tracing_can_be_disabled(clock):
    with Memory(clock=clock, trace=False) as m:
        m.scope("u").add("x is y")
        assert list(m.traces) == []
