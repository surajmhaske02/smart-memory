"""v0.3: non-English text, concurrency, memory use, time filters, contradictions, governance."""

import json
import threading
import time
from datetime import datetime, timedelta

import pytest

from smart_memory import CachedEmbedder, HashingEmbedder, Memory, MemoryType, RuleRouter, Status
from smart_memory._locks import RWLock
from smart_memory.contradictions import extract_slot, register_slot
from smart_memory.governance import Redactor, RetentionPolicy
from smart_memory.text import same_specifics, tokenize
from smart_memory.timeutil import to_timestamp

# --- Unicode text -------------------------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    ("प्रिया पुणे में रहती है", ["प्रिया", "पुणे", "रहती"]),
    ("मला चहा आवडतो", ["मला", "चहा", "आवडतो"]),
    ("Müller wohnt in München", ["müller", "wohnt", "münchen"]),
    ("José prefiere café", ["josé", "prefiere", "café"]),
    ("Москва большая", ["москва", "большая"]),
    ("He bought a laptop", ["buy", "laptop"]),
    ("Alice doesn’t like tea", ["alice", "doesnt", "like", "tea"]),
])
def test_tokenizer_keeps_words_whole_in_any_script(text, expected):
    assert tokenize(text) == expected


def test_hindi_and_marathi_memories_are_searchable(alice):
    alice.add("प्रिया पुणे में रहती है")
    alice.add("ग्राहक को ईमेल पसंद है")
    alice.add("मला चहा आवडतो")
    alice.add("Müller wohnt in München")
    assert alice.search("प्रिया कहाँ रहती है")[0].content == "प्रिया पुणे में रहती है"
    assert alice.search("ईमेल")[0].content == "ग्राहक को ईमेल पसंद है"
    assert alice.search("चहा")[0].content == "मला चहा आवडतो"
    assert alice.search("München")[0].content == "Müller wohnt in München"
    assert alice.search("munchen wohnt")  # still matches without the umlaut via shared trigrams/words


def test_irregular_verbs_match(alice):
    alice.add("Priya bought the router X200 last month", "episodic")
    assert alice.search("what did Priya buy")[0].content.startswith("Priya bought")


@pytest.mark.parametrize("text, memory_type, kind", [
    ("कल पेमेंट API फेल हो गया", MemoryType.EPISODIC, "failure"),
    ("काल रिफंड API अयशस्वी झाले", MemoryType.EPISODIC, "failure"),
    ("प्रिया को ईमेल पसंद है", MemoryType.SEMANTIC, "preference"),
    ("मला चहा आवडतो", MemoryType.SEMANTIC, "preference"),
    ("हमेशा ऑर्डर नंबर पहले कन्फर्म करें", MemoryType.PROCEDURAL, "instruction"),
    ("नेहमी ग्राहकाला ईमेल पाठवा", MemoryType.PROCEDURAL, "instruction"),
    ("प्रिया पुणे में रहती है", MemoryType.SEMANTIC, "fact"),
    ("Kalpana lives in Pune", MemoryType.SEMANTIC, "fact"),  # "kal" inside a word is not "कल"
])
def test_router_understands_hindi_and_marathi(text, memory_type, kind):
    decision = RuleRouter().route(text)
    assert (decision.memory_type, decision.kind) == (memory_type, kind)


def test_negation_guard_in_hindi():
    assert not same_specifics("मुझे चाय पसंद है", "मुझे चाय पसंद नहीं है")
    assert same_specifics("मुझे चाय पसंद है", "मुझे चाय पसंद है!")


# --- concurrency and memory -------------------------------------------------------------


def test_rwlock_readers_share_writers_exclude():
    lock, inside, peak, order = RWLock(), [0], [0], []
    gate = threading.Barrier(3)

    def reader():
        gate.wait()
        with lock.read():
            inside[0] += 1
            peak[0] = max(peak[0], inside[0])
            time.sleep(0.05)
            inside[0] -= 1

    threads = [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] >= 2  # readers overlapped

    with lock:
        with lock:  # reentrant
            with lock.read():  # a writer may read
                order.append("nested ok")
    assert order == ["nested ok"]

    def writer():
        with lock:
            order.append("writer")

    with lock.read():
        t = threading.Thread(target=writer)
        t.start()
        time.sleep(0.05)
        order.append("reader done")
    t.join()
    assert order[1:] == ["reader done", "writer"]  # writer waited for the reader


def test_concurrent_reads_and_writes_stay_consistent(mem):
    s = mem.scope("u", "a")
    s.add_many([f"fact number {i} about topic {i % 7}" for i in range(300)])
    errors = []

    def reader():
        try:
            for i in range(40):
                s.search(f"topic {i % 7}")
                s.context(f"fact number {i}")
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    def writer():
        try:
            for i in range(40):
                r = s.add(f"new event {i} happened today", "episodic")
                s.feedback(r.id)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(3)] + [threading.Thread(target=writer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(s.all("episodic")) == 40
    assert s.search("new event 39 happened")[0].content == "new event 39 happened today"


def test_embedding_cache_stores_compact_float32():
    cache = CachedEmbedder(HashingEmbedder(dim=64))
    out = cache.embed(["hello world"])
    stored = next(iter(cache._cache.values()))
    assert stored.typecode == "f" and len(stored) == 64
    assert isinstance(out[0], list) and abs(sum(x * x for x in out[0]) - 1) < 1e-5


def test_index_cache_is_bounded_by_total_memories(clock):
    m = Memory(clock=clock, index_cache_max_memories=50)
    for user in ("a", "b", "c"):
        s = m.scope(user)
        s.add_many([f"{user} fact {i}" for i in range(30)])
        s.search("fact")
    assert sum(len(i) for _, i in m._indexes.values()) <= 60  # evicted down to the cap (+ the newest)
    assert m.scope("a").search("a fact 3")  # an evicted index is rebuilt on demand


# --- time filters ----------------------------------------------------------------------


def test_time_parsing():
    now = 1_000_000.0
    assert to_timestamp("7d", now) == now - 7 * 86_400
    assert to_timestamp("24h", now) == now - 86_400
    assert to_timestamp("30m", now) == now - 1_800
    assert to_timestamp("2w", now) == now - 14 * 86_400
    assert to_timestamp(timedelta(hours=1), now) == now - 3_600
    assert to_timestamp(123.0, now) == 123.0
    assert to_timestamp(datetime(2026, 1, 1), now) == datetime(2026, 1, 1).timestamp()
    assert to_timestamp("2026-01-01", now) == datetime(2026, 1, 1).timestamp()
    assert to_timestamp(None, now) is None
    with pytest.raises(ValueError):
        to_timestamp("last tuesday", now)
    with pytest.raises(TypeError):
        to_timestamp(True, now)


def test_search_and_context_since_until(alice, clock):
    old = alice.record_event("Deploy to staging failed", outcome="failure")
    clock.advance(days=10)
    recent = alice.record_event("Deploy to production failed", outcome="failure")
    assert [r.id for r in alice.search("deploy failed", since="7d")] == [recent.id]
    assert [r.id for r in alice.search("deploy failed", until="7d")] == [old.id]
    assert len(alice.search("deploy failed", since="30d")) == 2
    card = alice.context("deploy failed", since="7d")
    assert recent.id in card.memory_ids and old.id not in card.memory_ids


def test_reinforced_memory_counts_as_recent(alice, clock):
    first = alice.add("Backup job failed", "episodic")
    clock.advance(days=20)
    alice.add("Backup job failed", "episodic")  # reinforces: last_seen is now
    assert [r.id for r in alice.search("backup failed", since="1d")] == [first.id]


# --- contradictions --------------------------------------------------------------------


@pytest.mark.parametrize("text, slot, subject, value", [
    ("Alice lives in Pune", "location", "alice", "pune"),
    ("Alice moved to Berlin.", "location", "alice", "berlin"),
    ("Alice now works at Acme Corp", "employer", "alice", "acme corp"),
    ("Bob joined Globex", "employer", "bob", "globex"),
    ("Alice's email is alice@new.com", "email", "alice", "alice@new.com"),
    ("Alice is 34 years old", "age", "alice", "34"),
    ("Alice reports to Bob", "manager", "alice", "bob"),
    ("Acme is on the Enterprise plan", "plan", "acme", "enterprise"),
    ("प्रिया पुणे में रहती है", "location", "प्रिया", "पुणे"),
    ("प्रिया अब बर्लिन में रहती है", "location", "प्रिया", "बर्लिन"),
])
def test_slot_extraction(text, slot, subject, value):
    fact = extract_slot(text)
    assert (fact.slot, fact.subject, fact.value) == (slot, subject, value)


def test_non_slot_text_is_ignored():
    assert extract_slot("Alice prefers email") is None
    assert extract_slot("The office lives in chaos, honestly") is None


def test_new_fact_supersedes_contradicting_one(alice, mem):
    pune = alice.add("Alice lives in Pune")
    berlin = alice.add("Alice moved to Berlin")
    assert alice.get(pune.id).status == Status.SUPERSEDED
    assert alice.get(pune.id).superseded_by == berlin.id
    assert [r.content for r in alice.search("where does Alice live")] == ["Alice moved to Berlin"]
    assert [r.id for r in alice.history("slot:location:alice")] == [berlin.id, pune.id]
    again = alice.add("Alice lives in Berlin.")  # same value: reinforce, not a new version
    assert again.id == berlin.id
    (trace,) = list(mem.traces.events("contradiction"))
    assert trace["payload"]["others"] == [pune.id]
    alice.add("Bob lives in Pune")  # different subject: no conflict
    assert alice.get(berlin.id).status == Status.ACTIVE


def test_contradiction_inside_one_batch(alice):
    first, second = alice.add_many(["Alice works at Acme", "Alice now works at Globex"])
    assert alice.get(first.id).status == Status.SUPERSEDED
    assert [r.content for r in alice.all("semantic")] == ["Alice now works at Globex"]


def test_flag_mode_keeps_both(clock):
    m = Memory(clock=clock, contradictions="flag")
    s = m.scope("u")
    a = s.add("Alice lives in Pune")
    b = s.add("Alice lives in Berlin")
    assert s.get(a.id).status == Status.ACTIVE and s.get(b.id).status == Status.ACTIVE
    assert s.get(b.id).metadata["conflicts_with"] == [a.id]
    assert s.get(a.id).metadata["conflicts_with"] == [b.id]


def test_contradictions_off_and_explicit_keys_win(clock):
    m = Memory(clock=clock, contradictions="off")
    s = m.scope("u")
    s.add("Alice lives in Pune")
    s.add("Alice lives in Berlin")
    assert len(s.all()) == 2
    with pytest.raises(ValueError):
        Memory(contradictions="maybe")


def test_custom_slot_pattern(alice):
    register_slot("shoe_size", r"^(?P<subject>.+?)\s+wears\s+size\s+(?P<value>\d+)\s+shoes$")
    old = alice.add("Alice wears size 7 shoes")
    alice.add("Alice wears size 8 shoes")
    assert alice.get(old.id).status == Status.SUPERSEDED
    with pytest.raises(ValueError):
        register_slot("bad", r"(?P<subject>.+)")


# --- governance -------------------------------------------------------------------------


def _valid_aadhaar() -> str:
    from smart_memory.governance import _aadhaar_ok
    base = "23456789012"
    return next(base + str(d) for d in range(10) if _aadhaar_ok(base + str(d)))


@pytest.mark.parametrize("text, label", [
    ("mail priya@example.com", "EMAIL"),
    ("call +91 98765 43210", "PHONE"),
    ("call 9876543210", "PHONE"),
    ("UK +44 20 7946 0958", "PHONE"),
    ("US (555) 123-4567", "PHONE"),
    ("card 4111 1111 1111 1111", "CARD"),
    ("PAN ABCDE1234F", "PAN"),
    ("SSN 123-45-6789", "SSN"),
    ("from 192.168.1.20", "IP"),
    ("key sk-abcdefghijklmnopqrstuv", "SECRET"),
])
def test_redactor_detects(text, label):
    clean, found = Redactor().redact(text)
    assert f"[{label}]" in clean and found == [label]


def test_redactor_aadhaar_checksum_and_false_positives():
    number = _valid_aadhaar()
    assert Redactor().redact(f"Aadhaar {number[:4]} {number[4:8]} {number[8:]}")[1] == ["AADHAAR"]
    for safe in ["Order 1234 failed at 14:05 on 2026-11-01", "Invoice total was 45000 rupees in 2026",
                 "Room 404, ext 1234", "Not a card 1234 5678 9012 3456", "version 4.2.10"]:
        assert Redactor().redact(safe) == (safe, [])


def test_redactor_options():
    only_email = Redactor(kinds=["email"])
    assert only_email.redact("a@b.co 9876543210") == ("[EMAIL] 9876543210", ["EMAIL"])
    custom = Redactor(custom={"EMPLOYEE_ID": r"\bEMP-\d{5}\b"})
    assert custom("badge EMP-12345") == "badge [EMPLOYEE_ID]"
    with pytest.raises(ValueError):
        Redactor(kinds=["nope"])


def test_memory_redacts_before_storing_embedding_and_tracing(clock, tmp_path):
    m = Memory(clock=clock, redact=True)
    s = m.scope("alice", "support", "t1")
    r = s.add("Alice's email is alice@example.com and card 4111 1111 1111 1111")
    assert "alice@example.com" not in r.content and "[EMAIL]" in r.content and "[CARD]" in r.content
    assert r.metadata["redacted"] == ["CARD", "EMAIL"]
    s.add_turn("user", "my number is 9876543210")
    assert s.working.turns()[0].content == "my number is [PHONE]"
    out = tmp_path / "audit.jsonl"
    m.export_audit(out)
    assert "alice@example.com" not in out.read_text() and "9876543210" not in out.read_text()


def test_retention_archive_and_delete(clock):
    policy = RetentionPolicy({"episodic": 30, "kind:turn": 1}, action="archive")
    m = Memory(clock=clock, retention=policy, dedupe_on_write=None)
    s = m.scope("u")
    old_event = s.add("Server rebooted", "episodic", importance=0.9)  # important: decay alone keeps it
    turn = s.add("user said hi", "episodic", kind="turn")
    pinned = s.add("VIP since 2020", "episodic", pin=True)
    fact = s.add("Alice likes jazz")
    clock.advance(days=2)
    assert s.apply_retention() == [turn.id]
    clock.advance(days=40)
    report = s.refine()
    assert report.expired == [old_event.id]
    assert s.get(old_event.id).status == Status.ARCHIVED
    assert s.get(pinned.id).status == Status.ACTIVE and s.get(fact.id).status == Status.ACTIVE

    deleting = Memory(clock=clock, retention=RetentionPolicy({"episodic": 1}, action="delete"))
    for user in ("a", "b"):
        deleting.scope(user).add("Server rebooted", "episodic")
    clock.advance(days=3)
    assert deleting.apply_retention() == 2
    assert deleting.scope("a").all(include_inactive=True) == []


def test_retention_validation():
    with pytest.raises(ValueError):
        RetentionPolicy({"episodic": 0})
    with pytest.raises(ValueError):
        RetentionPolicy({"episodic": 5}, action="shred")
    with pytest.raises(ValueError):
        Memory().scope("u").apply_retention()


def test_export_audit_filters(clock, tmp_path):
    m = Memory(clock=clock)
    m.scope("alice").add("Alice likes jazz")
    clock.advance(days=5)
    m.scope("bob").add("Bob likes rock")
    m.scope("bob").forget_user()
    out = tmp_path / "a.jsonl"
    assert m.export_audit(out, user_id="bob") == 2
    assert {json.loads(x)["event"] for x in out.read_text().splitlines()} == {"route", "forget"}
    assert m.export_audit(out, since="1d") == 2
    assert m.export_audit(out, events=["forget"]) == 1
