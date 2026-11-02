"""Behaviour every StorageBackend must have. New backends: add a factory to BACKENDS."""

import threading

import pytest

from smart_memory import InMemoryStore, MemoryRecord, MemoryType, PermissionDenied, Scope, SQLiteStore, Status

BACKENDS = {
    "inmemory": lambda tmp_path: InMemoryStore(),
    "sqlite-memory": lambda tmp_path: SQLiteStore(":memory:"),
    "sqlite-file": lambda tmp_path: SQLiteStore(str(tmp_path / "mem.db")),
}


@pytest.fixture(params=sorted(BACKENDS))
def store(request, tmp_path):
    s = BACKENDS[request.param](tmp_path)
    yield s
    s.close()


ALICE = Scope("alice", "support")
ALICE_OTHER_AGENT = Scope("alice", "sales")
BOB = Scope("bob", "support")
GLOBAL = Scope()


def rec(content, scope=ALICE, memory_type=MemoryType.SEMANTIC, **kw):
    kw.setdefault("thread_id", scope.thread_id)
    return MemoryRecord(content=content, memory_type=memory_type, user_id=scope.user_id,
                        agent_id=scope.agent_id, embedding=[1.0, 0.0, 0.5], **kw)


def test_roundtrip_preserves_all_fields(store):
    r = rec("hello", kind="fact", key="k1", metadata={"a": [1, 2], "b": {"c": "d"}}, importance=0.7,
            created_at=10.0, last_accessed_at=11.0, access_count=3)
    store.add(r)
    got = store.get(r.id, ALICE)
    assert got.to_dict(include_embedding=True) == r.to_dict(include_embedding=True)


def test_duplicate_id_rejected(store):
    r = rec("x")
    store.add(r)
    with pytest.raises(ValueError):
        store.add(r)
    with pytest.raises(ValueError):
        store.add_many([rec("y"), r])


def test_user_isolation(store):
    a, b = rec("alice secret"), rec("bob secret", BOB)
    store.add_many([a, b])
    assert store.get(b.id, ALICE) is None
    assert [r.content for r in store.query(ALICE)] == ["alice secret"]
    assert [r.content for r in store.query(BOB)] == ["bob secret"]


def test_agent_isolation_and_user_shared_records(store):
    private = rec("support only")
    shared = MemoryRecord("for every agent", MemoryType.SEMANTIC, user_id="alice", agent_id=None)
    store.add_many([private, shared])
    assert {r.content for r in store.query(ALICE_OTHER_AGENT)} == {"for every agent"}
    assert {r.content for r in store.query(ALICE)} == {"support only", "for every agent"}


def test_global_records_visible_to_all_but_writable_only_globally(store):
    g = MemoryRecord("company glossary", MemoryType.SEMANTIC)
    store.add(g)
    for scope in (ALICE, BOB, GLOBAL):
        assert store.get(g.id, scope) is not None
    with pytest.raises(PermissionDenied):
        store.delete(g.id, ALICE)
    g.content = "edited"
    with pytest.raises(PermissionDenied):
        store.update(g, ALICE)
    store.update(g, GLOBAL)
    assert store.get(g.id, BOB).content == "edited"


def test_global_scope_cannot_see_user_records(store):
    store.add(rec("private"))
    assert store.query(GLOBAL) == []


def test_cross_user_write_denied(store):
    r = rec("alice data")
    store.add(r)
    with pytest.raises(PermissionDenied):
        store.delete(r.id, BOB)
    with pytest.raises(PermissionDenied):
        store.update(r, BOB)
    assert store.get(r.id, ALICE).content == "alice data"


def test_delete_and_update(store):
    r = rec("v1")
    store.add(r)
    r.content, r.status = "v2", Status.SUPERSEDED
    store.update(r, ALICE)
    assert store.get(r.id, ALICE).content == "v2"
    assert store.delete(r.id, ALICE) is True
    assert store.delete(r.id, ALICE) is False
    with pytest.raises(KeyError):
        store.update(r, ALICE)


def test_query_filters(store):
    rs = [
        rec("e1", memory_type=MemoryType.EPISODIC, kind="event", thread_id="t1", created_at=1),
        rec("e2", memory_type=MemoryType.EPISODIC, kind="failure", thread_id="t2", created_at=2),
        rec("s1", kind="fact", key="alice.city", created_at=3),
        rec("p1", memory_type=MemoryType.PROCEDURAL, kind="procedure", created_at=4,
            status=Status.ARCHIVED),
    ]
    store.add_many(rs)
    assert [r.content for r in store.query(ALICE)] == ["s1", "e2", "e1"]  # newest first, active only
    assert [r.content for r in store.query(ALICE, memory_types=[MemoryType.EPISODIC])] == ["e2", "e1"]
    assert [r.content for r in store.query(ALICE, kinds=["failure"])] == ["e2"]
    assert [r.content for r in store.query(ALICE, thread_id="t1")] == ["e1"]
    assert [r.content for r in store.query(ALICE, key="alice.city")] == ["s1"]
    assert [r.content for r in store.query(ALICE, statuses=(Status.ARCHIVED,))] == ["p1"]
    assert len(store.query(ALICE, statuses=tuple(Status))) == 4
    assert [r.content for r in store.query(ALICE, limit=1)] == ["s1"]
    assert store.query(ALICE, memory_types=[]) == []


def test_owned_only_excludes_global(store):
    store.add_many([rec("mine"), MemoryRecord("global", MemoryType.SEMANTIC)])
    assert {r.content for r in store.query(ALICE)} == {"mine", "global"}
    assert {r.content for r in store.query(ALICE, owned_only=True)} == {"mine"}


def test_touch(store):
    r = rec("x")
    store.add(r)
    store.touch([r.id, r.id], 99.0)
    got = store.get(r.id, ALICE)
    assert got.access_count == 2 and got.last_accessed_at == 99.0


def test_traces_and_meta(store):
    store.add_trace("route", 1.0, ALICE, {"memory_id": "m1", "nested": {"x": 1}})
    store.add_trace("retrieve", 2.0, BOB, {"query": "q"})
    assert [t["event"] for t in store.iter_traces()] == ["route", "retrieve"]
    (t,) = list(store.iter_traces("route"))
    assert t["payload"] == {"memory_id": "m1", "nested": {"x": 1}} and t["user_id"] == "alice"
    assert store.get_meta("missing") is None
    store.set_meta("embedder", "a")
    store.set_meta("embedder", "b")
    assert store.get_meta("embedder") == "b"


def test_iter_all_ignores_scope(store):
    store.add_many([rec("a"), rec("b", BOB), MemoryRecord("g", MemoryType.SEMANTIC)])
    assert {r.content for r in store.iter_all()} == {"a", "b", "g"}


def test_concurrent_writes(store):
    def worker(n):
        for i in range(50):
            store.add(rec(f"{n}-{i}"))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store.query(ALICE)) == 200


def test_sqlite_persists_across_reopen(tmp_path):
    path = str(tmp_path / "p.db")
    s = SQLiteStore(path)
    r = rec("durable")
    s.add(r)
    s.close()
    s2 = SQLiteStore(path)
    assert s2.get(r.id, ALICE).embedding == [1.0, 0.0, 0.5]
    s2.close()
