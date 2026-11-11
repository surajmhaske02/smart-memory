# Contributing

Thanks for helping. Bug reports, docs fixes, backends and adapters are all welcome.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check smart_memory tests
```

## Ground rules

- **Measure performance claims.** Use `benchmarks/benchmark.py` and update `docs/OPTIMIZATION.md`
  when a change moves cost, latency or recall.
- **No new required dependencies.** The core must stay `pip install` and go. Put integrations
  behind an optional extra (`[project.optional-dependencies]`) and import lazily.
- **Entitlements live in the data layer.** Any backend must apply `Scope` visibility inside
  its query. Add your backend to `BACKENDS` in `tests/test_storage_contract.py`; it must pass.
- **Tests for behaviour changes.** Use the `clock` fixture for anything time-dependent.
- **Keep the public API small.** New features should usually be a parameter or a pluggable
  component, not a new top-level concept.

## Adding a storage backend

1. Subclass `smart_memory.storage.StorageBackend` and implement its abstract methods.
2. Mirror `Scope.can_see` (reads) and `Scope.can_write` (updates and deletes) in your queries.
3. Add a factory to `tests/test_storage_contract.py` and run the suite. For Postgres, set
   `SMART_MEMORY_TEST_PG_DSN`, or `pip install pgserver` for an embedded test server.
4. Optionally implement `generation()` (a token that changes on every write) so `Memory` can keep
   a warm in-process index for your backend.

## Adding an embedder

Anything with `name`, `dim` and `embed(texts) -> list[list[float]]` works. Wrap a provider
with `FunctionEmbedder`, or add a lazily imported adapter next to `SentenceTransformerEmbedder`.

## Pull requests

Keep PRs focused, describe the behaviour change, and add a line to `CHANGELOG.md`.
