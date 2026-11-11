"""PostgreSQL backend with optional pgvector (``pip install "psycopg[binary]"``).

    from smart_memory import Memory
    from smart_memory.storage.postgres import PostgresStore

    store = PostgresStore("postgresql://user:pass@host/db")
    mem = Memory(store=store)                         # many processes can share it
    mem = Memory(store=store, candidate_limit=200)    # huge scopes: pgvector + full-text pick candidates

* Scope visibility is compiled into every SQL query, exactly like :class:`SQLiteStore`.
* A write counter in the database (``generation``) lets every process notice other
  processes' writes and refresh its warm index.
* With pgvector, embeddings are also kept in a ``vector`` column with an HNSW index, and
  :meth:`candidates` returns the nearest memories plus full-text matches from the server.
"""

from __future__ import annotations

import json
import re
import threading
from array import array
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from ..text import STOPWORDS, words
from ..types import MemoryRecord, MemoryType, Scope, Status
from .base import PermissionDenied, StorageBackend

try:
    import psycopg
    from psycopg import errors as pg_errors
    from psycopg.rows import dict_row
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError('PostgresStore needs psycopg 3: pip install "psycopg[binary]"') from exc

_COLUMNS = ("id", "user_id", "agent_id", "thread_id", "memory_type", "kind", "key", "content", "metadata",
            "importance", "embedding", "created_at", "updated_at", "last_accessed_at", "access_count", "status",
            "superseded_by")


def _encode(vec: Optional[Sequence[float]]) -> Optional[bytes]:
    return array("f", vec).tobytes() if vec is not None else None


def _decode(blob: Any) -> Optional[List[float]]:
    if blob is None:
        return None
    arr = array("f")
    arr.frombytes(bytes(blob))
    return arr.tolist()


def _vector_literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"


class PostgresStore(StorageBackend):
    """Args:
        dsn: libpq connection string or URL.
        table_prefix: Table name prefix, so several apps (or tests) can share one database.
        use_pgvector: ``None`` auto-detects the ``vector`` extension; ``False`` never uses it.
    """

    def __init__(self, dsn: str, *, table_prefix: str = "smart_memory", use_pgvector: Optional[bool] = None) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,40}", table_prefix):
            raise ValueError("table_prefix must be a plain SQL identifier")
        self._p = table_prefix
        self._lock = threading.RLock()
        self._conn = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)
        self.has_pgvector = False
        self._dim: Optional[int] = None
        self._create_schema(use_pgvector)

    # --- schema -------------------------------------------------------------------
    def _create_schema(self, use_pgvector: Optional[bool]) -> None:
        p = self._p
        with self._lock, self._conn.transaction():
            self._conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {p}_memories (
                    id TEXT PRIMARY KEY, user_id TEXT, agent_id TEXT, thread_id TEXT,
                    memory_type TEXT NOT NULL, kind TEXT NOT NULL, key TEXT, content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}', importance DOUBLE PRECISION NOT NULL,
                    embedding BYTEA, created_at DOUBLE PRECISION NOT NULL, updated_at DOUBLE PRECISION NOT NULL,
                    last_accessed_at DOUBLE PRECISION, access_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active', superseded_by TEXT, seq BIGSERIAL)""")
            self._conn.execute(f"CREATE INDEX IF NOT EXISTS {p}_scope_idx "
                               f"ON {p}_memories (user_id, agent_id, status, memory_type)")
            self._conn.execute(f"CREATE INDEX IF NOT EXISTS {p}_key_idx ON {p}_memories (user_id, agent_id, key)")
            self._conn.execute(f"CREATE INDEX IF NOT EXISTS {p}_fts_idx ON {p}_memories "
                               f"USING gin (to_tsvector('simple', content))")
            self._conn.execute(f"CREATE INDEX IF NOT EXISTS {p}_meta_idx ON {p}_memories USING gin (metadata)")
            self._conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {p}_traces (
                    id BIGSERIAL PRIMARY KEY, ts DOUBLE PRECISION NOT NULL, event TEXT NOT NULL,
                    user_id TEXT, agent_id TEXT, thread_id TEXT, payload JSONB NOT NULL)""")
            self._conn.execute(f"CREATE TABLE IF NOT EXISTS {p}_meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
            self._conn.execute(f"INSERT INTO {p}_meta (k, v) VALUES ('generation', '0') ON CONFLICT (k) DO NOTHING")
        if use_pgvector is not False:
            try:
                with self._lock, self._conn.transaction():
                    self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    self._conn.execute(f"ALTER TABLE {p}_memories ADD COLUMN IF NOT EXISTS vec vector")
                self.has_pgvector = True
                dim = self.get_meta("vector_dim")
                self._dim = int(dim) if dim else None
            except psycopg.Error:
                if use_pgvector:
                    raise
        if use_pgvector and not self.has_pgvector:  # pragma: no cover - defensive
            raise RuntimeError("pgvector requested but the 'vector' extension is unavailable")

    def _ensure_vector_index(self, dim: int) -> None:
        """HNSW index on ``vec::vector(dim)``, created when the first embedding arrives."""
        if self._dim == dim:
            return
        if self._dim is not None:
            raise ValueError(f"Store holds {self._dim}-dim vectors; got {dim}. Re-embed into a new table_prefix.")
        self._conn.execute(f"CREATE INDEX IF NOT EXISTS {self._p}_vec_idx ON {self._p}_memories "
                           f"USING hnsw ((vec::vector({dim})) vector_cosine_ops)")
        self._set_meta("vector_dim", str(dim))
        self._dim = dim

    def drop_tables(self) -> None:
        """Delete this store's tables (tests and teardown)."""
        with self._lock:
            for t in ("memories", "traces", "meta"):
                self._conn.execute(f"DROP TABLE IF EXISTS {self._p}_{t}")

    # --- helpers ------------------------------------------------------------------
    @staticmethod
    def _visible(scope: Scope, owned_only: bool = False) -> Tuple[str, List[Any]]:
        if owned_only:
            return ("user_id IS NOT DISTINCT FROM %s AND (agent_id IS NULL OR agent_id IS NOT DISTINCT FROM %s)",
                    [scope.user_id, scope.agent_id])
        return ("(user_id IS NULL OR user_id IS NOT DISTINCT FROM %s) "
                "AND (agent_id IS NULL OR agent_id IS NOT DISTINCT FROM %s)", [scope.user_id, scope.agent_id])

    def _row(self, row: Dict[str, Any]) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"], user_id=row["user_id"], agent_id=row["agent_id"], thread_id=row["thread_id"],
            memory_type=MemoryType(row["memory_type"]), kind=row["kind"], key=row["key"], content=row["content"],
            metadata=row["metadata"] or {}, importance=row["importance"], embedding=_decode(row["embedding"]),
            created_at=row["created_at"], updated_at=row["updated_at"], last_accessed_at=row["last_accessed_at"],
            access_count=row["access_count"], status=Status(row["status"]), superseded_by=row["superseded_by"],
        )

    def _values(self, r: MemoryRecord) -> List[Any]:
        return [r.id, r.user_id, r.agent_id, r.thread_id, r.memory_type.value, r.kind, r.key, r.content,
                json.dumps(r.metadata, default=str), r.importance, _encode(r.embedding), r.created_at, r.updated_at,
                r.last_accessed_at, r.access_count, r.status.value, r.superseded_by]

    def _bump(self) -> None:
        self._conn.execute(f"UPDATE {self._p}_meta SET v = (v::bigint + 1)::text WHERE k = 'generation'")

    def _insert(self, records: Sequence[MemoryRecord]) -> None:
        cols = list(_COLUMNS)
        placeholders = ["%s"] * len(cols)
        if self.has_pgvector:
            cols.append("vec")
            placeholders.append("%s::vector")
        sql = f"INSERT INTO {self._p}_memories ({','.join(cols)}) VALUES ({','.join(placeholders)})"
        rows = []
        for r in records:
            values = self._values(r)
            if self.has_pgvector:
                if r.embedding:
                    self._ensure_vector_index(len(r.embedding))
                values.append(_vector_literal(r.embedding) if r.embedding else None)
            rows.append(values)
        with self._conn.cursor() as cur:
            cur.executemany(sql, rows)

    # --- memories -----------------------------------------------------------------
    def add(self, record: MemoryRecord) -> None:
        self.add_many([record])

    def add_many(self, records) -> None:  # type: ignore[override]
        records = list(records)
        if not records:
            return
        with self._lock:
            try:
                with self._conn.transaction():
                    self._insert(records)
                    self._bump()
            except pg_errors.UniqueViolation as exc:
                raise ValueError("Duplicate memory id") from exc

    def get(self, record_id: str, scope: Scope) -> Optional[MemoryRecord]:
        where, params = self._visible(scope)
        with self._lock:
            row = self._conn.execute(f"SELECT * FROM {self._p}_memories WHERE id = %s AND {where}",
                                     [record_id, *params]).fetchone()
        return self._row(row) if row else None

    def _check_owner(self, record_id: str, scope: Scope) -> bool:
        row = self._conn.execute(f"SELECT user_id, agent_id FROM {self._p}_memories WHERE id = %s",
                                 [record_id]).fetchone()
        if row is None:
            return False
        if row["user_id"] != scope.user_id or row["agent_id"] not in (None, scope.agent_id):
            raise PermissionDenied(f"Scope {scope} cannot modify memory {record_id}")
        return True

    def update(self, record: MemoryRecord, scope: Scope) -> None:
        cols = list(_COLUMNS[1:])
        values = self._values(record)[1:]
        assignments = [f"{c} = %s" for c in cols]
        if self.has_pgvector:
            if record.embedding:
                self._ensure_vector_index(len(record.embedding))
            assignments.append("vec = %s::vector")
            values.append(_vector_literal(record.embedding) if record.embedding else None)
        with self._lock:
            if not self._check_owner(record.id, scope):
                raise KeyError(record.id)
            with self._conn.transaction():
                self._conn.execute(f"UPDATE {self._p}_memories SET {', '.join(assignments)} WHERE id = %s",
                                   [*values, record.id])
                self._bump()

    def delete(self, record_id: str, scope: Scope) -> bool:
        with self._lock:
            if not self._check_owner(record_id, scope):
                return False
            with self._conn.transaction():
                self._conn.execute(f"DELETE FROM {self._p}_memories WHERE id = %s", [record_id])
                self._bump()
            return True

    def query(self, scope: Scope, *, memory_types: Optional[Sequence[MemoryType]] = None,
              kinds: Optional[Sequence[str]] = None, thread_id: Optional[str] = None,
              key: Optional[str] = None, statuses: Sequence[Status] = (Status.ACTIVE,),
              owned_only: bool = False, limit: Optional[int] = None) -> List[MemoryRecord]:
        where, params = self._visible(scope, owned_only)
        clauses = [where]
        if memory_types is not None:
            if not memory_types:
                return []
            clauses.append("memory_type = ANY(%s)")
            params.append([MemoryType.coerce(t).value for t in memory_types])
        if kinds is not None:
            if not kinds:
                return []
            clauses.append("kind = ANY(%s)")
            params.append(list(kinds))
        if not statuses:
            return []
        clauses.append("status = ANY(%s)")
        params.append([Status(s).value for s in statuses])
        if thread_id is not None:
            clauses.append("thread_id = %s")
            params.append(thread_id)
        if key is not None:
            clauses.append("key = %s")
            params.append(key)
        sql = f"SELECT * FROM {self._p}_memories WHERE {' AND '.join(clauses)} ORDER BY created_at DESC, seq DESC"
        if limit is not None:
            sql += " LIMIT %s"
            params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row(r) for r in rows]

    def find_by_metadata(self, scope: Scope, key: str, value: Any) -> List[MemoryRecord]:
        where, params = self._visible(scope)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM {self._p}_memories WHERE {where} AND status = 'active' AND metadata @> %s::jsonb "
                f"ORDER BY created_at DESC, seq DESC", [*params, json.dumps({key: value})]).fetchall()
        return [self._row(r) for r in rows]

    def candidates(self, scope: Scope, query: str, query_vec: Optional[Sequence[float]],
                   limit: int = 200) -> List[MemoryRecord]:
        """Server-side pre-selection: nearest vectors (pgvector HNSW) plus full-text matches."""
        where, params = self._visible(scope)
        found: Dict[str, MemoryRecord] = {}
        with self._lock:
            if self.has_pgvector and query_vec and self._dim == len(query_vec):
                rows = self._conn.execute(
                    f"SELECT * FROM {self._p}_memories WHERE {where} AND status = 'active' AND vec IS NOT NULL "
                    f"ORDER BY vec::vector({self._dim}) <=> %s::vector({self._dim}) LIMIT %s",
                    [*params, _vector_literal(query_vec), limit]).fetchall()
                found.update((r["id"], self._row(r)) for r in rows)
            # OR of the query's content words: natural questions rarely contain every word of a memory.
            terms = [w for w in dict.fromkeys(words(query)) if w not in STOPWORDS]
            if terms:
                tsquery = " | ".join("'" + w.replace("'", "''") + "'" for w in terms)
                rows = self._conn.execute(
                    f"SELECT * FROM {self._p}_memories WHERE {where} AND status = 'active' "
                    f"AND to_tsvector('simple', content) @@ to_tsquery('simple', %s) "
                    f"ORDER BY ts_rank(to_tsvector('simple', content), to_tsquery('simple', %s)) DESC "
                    f"LIMIT %s", [*params, tsquery, tsquery, limit]).fetchall()
                found.update((r["id"], self._row(r)) for r in rows if r["id"] not in found)
        return list(found.values())

    def touch(self, record_ids: Sequence[str], timestamp: float) -> None:
        if not record_ids:
            return
        with self._lock, self._conn.cursor() as cur:  # one row per id, so repeated ids count repeatedly
            cur.executemany(f"UPDATE {self._p}_memories SET access_count = access_count + 1, "
                            f"last_accessed_at = %s WHERE id = %s", [(timestamp, rid) for rid in record_ids])

    def generation(self) -> int:
        with self._lock:
            return int(self._conn.execute(f"SELECT v FROM {self._p}_meta WHERE k = 'generation'").fetchone()["v"])

    def iter_all(self) -> Iterator[MemoryRecord]:
        with self._lock:
            rows = self._conn.execute(f"SELECT * FROM {self._p}_memories ORDER BY created_at, seq").fetchall()
        return (self._row(r) for r in rows)

    # --- traces & meta ------------------------------------------------------------
    def add_trace(self, event: str, timestamp: float, scope: Scope, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                f"INSERT INTO {self._p}_traces (ts, event, user_id, agent_id, thread_id, payload) "
                f"VALUES (%s, %s, %s, %s, %s, %s::jsonb)",
                [timestamp, event, scope.user_id, scope.agent_id, scope.thread_id, json.dumps(payload, default=str)])

    def iter_traces(self, event: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        sql, params = f"SELECT * FROM {self._p}_traces", []
        if event is not None:
            sql, params = sql + " WHERE event = %s", [event]
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY id", params).fetchall()
        return ({"ts": r["ts"], "event": r["event"], "user_id": r["user_id"], "agent_id": r["agent_id"],
                 "thread_id": r["thread_id"], "payload": r["payload"]} for r in rows)

    def get_meta(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(f"SELECT v FROM {self._p}_meta WHERE k = %s", [key]).fetchone()
        return row["v"] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(f"INSERT INTO {self._p}_meta (k, v) VALUES (%s, %s) "
                           f"ON CONFLICT (k) DO UPDATE SET v = excluded.v", [key, value])

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._set_meta(key, value)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
