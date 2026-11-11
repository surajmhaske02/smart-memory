"""Find a PostgreSQL server for tests: $SMART_MEMORY_TEST_PG_DSN, else an embedded `pgserver` if installed."""

import functools
import os
import tempfile
import uuid


@functools.lru_cache(maxsize=1)
def pg_dsn():
    dsn = os.environ.get("SMART_MEMORY_TEST_PG_DSN")
    if dsn:
        return dsn
    try:
        import pgserver  # embedded Postgres + pgvector, test-only
        import psycopg  # noqa: F401
    except ImportError:
        return None
    server = pgserver.get_server(tempfile.mkdtemp(prefix="smart_memory_pg_"), cleanup_mode="stop")
    return server.get_uri()


def make_store(**kwargs):
    from smart_memory.storage.postgres import PostgresStore

    return PostgresStore(pg_dsn(), table_prefix=f"t_{uuid.uuid4().hex[:12]}", **kwargs)
