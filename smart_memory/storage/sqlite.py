"""SQLite backend: zero-dependency, durable, safe to share across threads.

Scope visibility is compiled into every SQL ``WHERE`` clause. Embeddings are stored as
float32 blobs and scored in Python, which comfortably handles tens of thousands of
memories per scope. Beyond that, implement a backend on a vector database.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from array import array
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from ..types import MemoryRecord, MemoryType, Scope, Status
from .base import PermissionDenied, StorageBackend

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    agent_id TEXT,
    thread_id TEXT,
    memory_type TEXT NOT NULL,
    kind TEXT NOT NULL,
    key TEXT,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    importance REAL NOT NULL,
    embedding BLOB,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_accessed_at REAL,
    access_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active',
    superseded_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(user_id, agent_id, status, memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(user_id, agent_id, key);
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    event TEXT NOT NULL,
    user_id TEXT,
    agent_id TEXT,
    thread_id TEXT,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_event ON traces(event, ts);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
"""

_COLUMNS = ("id", "user_id", "agent_id", "thread_id", "memory_type", "kind", "key", "content",
            "metadata", "importance", "embedding", "created_at", "updated_at", "last_accessed_at",
            "access_count", "status", "superseded_by")


def _encode_embedding(vec: Optional[Sequence[float]]) -> Optional[bytes]:
    return array("f", vec).tobytes() if vec is not None else None


def _decode_embedding(blob: Optional[bytes]) -> Optional[List[float]]:
    if blob is None:
        return None
    arr = array("f")
    arr.frombytes(blob)
    return arr.tolist()


def _visible_sql(scope: Scope, owned_only: bool = False) -> Tuple[str, List[Any]]:
    # `IS` compares NULLs as equal in SQLite, so a global scope (user_id None) works too.
    if owned_only:
        return "user_id IS ? AND (agent_id IS NULL OR agent_id IS ?)", [scope.user_id, scope.agent_id]
    return ("(user_id IS NULL OR user_id IS ?) AND (agent_id IS NULL OR agent_id IS ?)",
            [scope.user_id, scope.agent_id])


class SQLiteStore(StorageBackend):
    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._lock = threading.RLock()
        self._writes = 0
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)

    # --- helpers ------------------------------------------------------------------
    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"], user_id=row["user_id"], agent_id=row["agent_id"], thread_id=row["thread_id"],
            memory_type=MemoryType(row["memory_type"]), kind=row["kind"], key=row["key"],
            content=row["content"], metadata=json.loads(row["metadata"]), importance=row["importance"],
            embedding=_decode_embedding(row["embedding"]), created_at=row["created_at"],
            updated_at=row["updated_at"], last_accessed_at=row["last_accessed_at"],
            access_count=row["access_count"], status=Status(row["status"]), superseded_by=row["superseded_by"],
        )

    @staticmethod
    def _record_values(record: MemoryRecord) -> Tuple[Any, ...]:
        return (record.id, record.user_id, record.agent_id, record.thread_id, record.memory_type.value,
                record.kind, record.key, record.content, json.dumps(record.metadata, default=str),
                record.importance, _encode_embedding(record.embedding), record.created_at, record.updated_at,
                record.last_accessed_at, record.access_count, record.status.value, record.superseded_by)

    # --- memories -----------------------------------------------------------------
    def add(self, record: MemoryRecord) -> None:
        placeholders = ",".join("?" * len(_COLUMNS))
        with self._lock:
            try:
                self._conn.execute(f"INSERT INTO memories ({','.join(_COLUMNS)}) VALUES ({placeholders})",
                                   self._record_values(record))
                self._writes += 1
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Duplicate memory id {record.id}") from exc

    def add_many(self, records) -> None:  # type: ignore[override]
        placeholders = ",".join("?" * len(_COLUMNS))
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.executemany(f"INSERT INTO memories ({','.join(_COLUMNS)}) VALUES ({placeholders})",
                                       [self._record_values(r) for r in records])
                self._conn.execute("COMMIT")
                self._writes += 1
            except sqlite3.IntegrityError as exc:
                self._conn.execute("ROLLBACK")
                raise ValueError("Duplicate memory id in batch") from exc
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def get(self, record_id: str, scope: Scope) -> Optional[MemoryRecord]:
        where, params = _visible_sql(scope)
        with self._lock:
            row = self._conn.execute(f"SELECT * FROM memories WHERE id = ? AND {where}",
                                     [record_id, *params]).fetchone()
        return self._row_to_record(row) if row else None

    def _check_owner(self, record_id: str, scope: Scope) -> bool:
        row = self._conn.execute("SELECT user_id, agent_id FROM memories WHERE id = ?", [record_id]).fetchone()
        if row is None:
            return False
        if row["user_id"] != scope.user_id or row["agent_id"] not in (None, scope.agent_id):
            raise PermissionDenied(f"Scope {scope} cannot modify memory {record_id}")
        return True

    def update(self, record: MemoryRecord, scope: Scope) -> None:
        assignments = ",".join(f"{c} = ?" for c in _COLUMNS[1:])
        values = self._record_values(record)
        with self._lock:
            if not self._check_owner(record.id, scope):
                raise KeyError(record.id)
            self._conn.execute(f"UPDATE memories SET {assignments} WHERE id = ?", [*values[1:], record.id])
            self._writes += 1

    def delete(self, record_id: str, scope: Scope) -> bool:
        with self._lock:
            if not self._check_owner(record_id, scope):
                return False
            self._conn.execute("DELETE FROM memories WHERE id = ?", [record_id])
            self._writes += 1
            return True

    def query(self, scope: Scope, *, memory_types: Optional[Sequence[MemoryType]] = None,
              kinds: Optional[Sequence[str]] = None, thread_id: Optional[str] = None,
              key: Optional[str] = None, statuses: Sequence[Status] = (Status.ACTIVE,),
              owned_only: bool = False, limit: Optional[int] = None) -> List[MemoryRecord]:
        where, params = _visible_sql(scope, owned_only)
        clauses = [where]

        def add_in(column: str, values: Sequence[Any]) -> None:
            clauses.append(f"{column} IN ({','.join('?' * len(values))})")
            params.extend(values)

        if memory_types is not None:
            if not memory_types:
                return []
            add_in("memory_type", [MemoryType.coerce(t).value for t in memory_types])
        if kinds is not None:
            if not kinds:
                return []
            add_in("kind", list(kinds))
        if not statuses:
            return []
        add_in("status", [Status(s).value for s in statuses])
        if thread_id is not None:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if key is not None:
            clauses.append("key = ?")
            params.append(key)
        sql = f"SELECT * FROM memories WHERE {' AND '.join(clauses)} ORDER BY created_at DESC, rowid DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def touch(self, record_ids: Sequence[str], timestamp: float) -> None:
        if not record_ids:
            return
        with self._lock:
            self._conn.executemany(
                "UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ?",
                [(timestamp, rid) for rid in record_ids],
            )

    def generation(self) -> tuple:
        # data_version changes when *another* connection (process) commits, so external
        # writers invalidate the in-process index too.
        with self._lock:
            return self._writes, self._conn.execute("PRAGMA data_version").fetchone()[0]

    def iter_all(self) -> Iterator[MemoryRecord]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM memories ORDER BY created_at, rowid").fetchall()
        return (self._row_to_record(r) for r in rows)

    # --- traces -------------------------------------------------------------------
    def add_trace(self, event: str, timestamp: float, scope: Scope, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO traces (ts, event, user_id, agent_id, thread_id, payload) VALUES (?,?,?,?,?,?)",
                [timestamp, event, scope.user_id, scope.agent_id, scope.thread_id, json.dumps(payload, default=str)],
            )

    def iter_traces(self, event: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        sql, params = "SELECT * FROM traces", []
        if event is not None:
            sql, params = sql + " WHERE event = ?", [event]
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY id", params).fetchall()
        return ({"ts": r["ts"], "event": r["event"], "user_id": r["user_id"], "agent_id": r["agent_id"],
                 "thread_id": r["thread_id"], "payload": json.loads(r["payload"])} for r in rows)

    # --- meta ---------------------------------------------------------------------
    def get_meta(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute("SELECT v FROM meta WHERE k = ?", [key]).fetchone()
        return row["v"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO meta (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
                               [key, value])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __del__(self) -> None:  # a forgotten close() shouldn't leak the connection
        conn = getattr(self, "_conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - interpreter shutdown
                pass
