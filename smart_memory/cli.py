"""Command line: ``smart-memory <command>`` (or ``python -m smart_memory``).

    smart-memory mcp     --db memory.db --user me                 # MCP server for Claude & co.
    smart-memory serve   --db memory.db --api-key KEY             # REST API
    smart-memory add     --db memory.db --user me "Alice prefers email"
    smart-memory search  --db memory.db --user me "how to contact Alice"
    smart-memory context --db memory.db --user me "refund for Alice"
    smart-memory stats   --db memory.db
    smart-memory export  --db memory.db backup.jsonl
    smart-memory audit   --db memory.db audit.jsonl --user me --since 30d
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import __version__
from .memory import Memory


def _memory(args: argparse.Namespace) -> Memory:
    return Memory(os.path.expanduser(args.db), redact=getattr(args, "redact", False) or None)


def _scoped(args: argparse.Namespace):  # noqa: ANN202
    return _memory(args).scope(args.user, args.agent, getattr(args, "thread", None))


def _print(data: object) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp_server import MCPServer

    print(f"smart-memory {__version__} MCP server on stdio (db={args.db}, user={args.user})", file=sys.stderr)
    MCPServer(_scoped(args), allow_forget=args.allow_forget).serve()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .http_server import create_server

    api_key = args.api_key or os.environ.get("SMART_MEMORY_API_KEY")
    server = create_server(_memory(args), args.host, args.port, api_key, allow_insecure=args.insecure_no_auth)
    auth = "API key required" if api_key else "no auth (loopback only)"
    print(f"smart-memory {__version__} REST API on http://{args.host}:{args.port} ({auth})", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    r = _scoped(args).add(args.text, args.type, key=args.key, pin=args.pin)
    _print({"id": r.id, "memory_type": r.memory_type.value, "kind": r.kind, "content": r.content})
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    hits = _scoped(args).search(args.query, k=args.k, since=args.since)
    _print([{"id": h.id, "score": round(h.score, 3), "type": h.record.memory_type.value, "content": h.content}
            for h in hits])
    return 0


def cmd_context(args: argparse.Namespace) -> int:
    print(_scoped(args).context(args.query, max_tokens=args.max_tokens).render() or "(no relevant memories)")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    _print(_memory(args).stats())
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    print(f"wrote {_memory(args).export_jsonl(args.out)} memories to {args.out}", file=sys.stderr)
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    n = _memory(args).export_audit(args.out, user_id=args.user, since=args.since)
    print(f"wrote {n} audit events to {args.out}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smart-memory", description="Long-term memory for AI agents.")
    parser.add_argument("--version", action="version", version=f"smart-memory {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, scoped: bool = True) -> None:
        p.add_argument("--db", default=os.environ.get("SMART_MEMORY_DB", "smart_memory.db"),
                       help="SQLite file (default: $SMART_MEMORY_DB or smart_memory.db)")
        if scoped:
            p.add_argument("--user", default=os.environ.get("SMART_MEMORY_USER"), help="user_id")
            p.add_argument("--agent", default=os.environ.get("SMART_MEMORY_AGENT"), help="agent_id")
            p.add_argument("--thread", default=None, help="thread_id")
        p.add_argument("--redact", action="store_true", help="strip personal data before storing")

    p = sub.add_parser("mcp", help="run an MCP server on stdio")
    common(p)
    p.add_argument("--allow-forget", action="store_true", help="let the model delete memories")
    p.set_defaults(func=cmd_mcp)

    p = sub.add_parser("serve", help="run the REST API")
    common(p, scoped=False)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--api-key", default=None, help="required bearer token (or $SMART_MEMORY_API_KEY)")
    p.add_argument("--insecure-no-auth", action="store_true", help="allow a non-loopback host without a key")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("add", help="store a memory")
    common(p)
    p.add_argument("text")
    p.add_argument("--type", choices=["episodic", "semantic", "procedural"])
    p.add_argument("--key")
    p.add_argument("--pin", action="store_true")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("search", help="search memories")
    common(p)
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--since", help="e.g. 7d, 24h, 2026-11-01")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("context", help="print a Context Card")
    common(p)
    p.add_argument("query")
    p.add_argument("--max-tokens", type=int, default=800)
    p.set_defaults(func=cmd_context)

    p = sub.add_parser("stats", help="counts by type and status")
    common(p, scoped=False)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("export", help="dump all memories to JSON lines")
    common(p, scoped=False)
    p.add_argument("out")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("audit", help="export the audit log to JSON lines")
    common(p, scoped=False)
    p.add_argument("out")
    p.add_argument("--user", default=None)
    p.add_argument("--since", default=None)
    p.set_defaults(func=cmd_audit)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
