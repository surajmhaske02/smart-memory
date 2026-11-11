"""Model Context Protocol (MCP) server over stdio, standard library only.

Gives Claude Desktop, Claude Code and any other MCP client long-term memory with no code:

    smart-memory mcp --db ~/memory.db --user me

The server is bound to one user/agent scope chosen at start-up; the model cannot switch
to another user's memory. Tools: ``remember``, ``recall``, ``context`` (and ``forget``
with ``--allow-forget``).
"""

from __future__ import annotations

import json
import sys
from typing import IO, TYPE_CHECKING, Any, Dict, List, Optional

from .tools import MemoryTools

if TYPE_CHECKING:  # pragma: no cover
    from .memory import ScopedMemory

SUPPORTED_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

INSTRUCTIONS = (
    "Long-term memory for this user. Call `context` at the start of a task to load what is known. "
    "Call `remember` when you learn a durable fact, preference, decision, outcome or procedure. "
    "Call `recall` to look something specific up."
)

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603


class MCPServer:
    def __init__(self, memory: ScopedMemory, allow_forget: bool = False, name: str = "smart-memory") -> None:
        from . import __version__

        self.tools = MemoryTools(memory, allow_forget=allow_forget)
        self.server_info = {"name": name, "version": __version__}

    # --- protocol -----------------------------------------------------------------
    def handle(self, message: Any) -> Optional[Dict[str, Any]]:
        """Handle one JSON-RPC message. Returns the response, or ``None`` for notifications."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or "method" not in message:
            return _error(message.get("id") if isinstance(message, dict) else None, INVALID_REQUEST,
                          "Invalid JSON-RPC 2.0 request")
        method, params = message["method"], message.get("params") or {}
        if "id" not in message:
            return None  # notification (e.g. notifications/initialized): no reply
        req_id = message["id"]
        try:
            if method == "initialize":
                asked = params.get("protocolVersion")
                return _result(req_id, {
                    "protocolVersion": asked if asked in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0],
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": self.server_info,
                    "instructions": INSTRUCTIONS,
                })
            if method == "ping":
                return _result(req_id, {})
            if method == "tools/list":
                return _result(req_id, {"tools": [
                    {"name": s["name"], "description": s["description"], "inputSchema": s["parameters"]}
                    for s in self.tools.schemas("json")
                ]})
            if method == "tools/call":
                name = params.get("name")
                if name not in self.tools.names:
                    return _error(req_id, INVALID_PARAMS, f"Unknown tool: {name!r}")
                text = self.tools.handle(name, params.get("arguments") or {})
                return _result(req_id, {"content": [{"type": "text", "text": text}],
                                        "isError": "error" in json.loads(text)})
            return _error(req_id, METHOD_NOT_FOUND, f"Method not found: {method}")
        except Exception as exc:  # noqa: BLE001 - report, keep serving
            return _error(req_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")

    def handle_line(self, line: str) -> Optional[str]:
        line = line.strip()
        if not line:
            return None
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            return json.dumps(_error(None, PARSE_ERROR, f"Parse error: {exc}"))
        if isinstance(message, list):  # batch
            replies: List[Dict[str, Any]] = [r for r in map(self.handle, message) if r is not None]
            return json.dumps(replies) if replies else None
        reply = self.handle(message)
        return json.dumps(reply) if reply is not None else None

    def serve(self, stdin: Optional[IO[str]] = None, stdout: Optional[IO[str]] = None) -> None:
        """Read newline-delimited JSON-RPC from ``stdin`` until EOF. Logs go to stderr, never stdout."""
        stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
        for line in stdin:
            reply = self.handle_line(line)
            if reply is not None:
                stdout.write(reply + "\n")
                stdout.flush()


def _result(req_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}
