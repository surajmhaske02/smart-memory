"""Let the model manage its own memory through function calling.

    tools = MemoryTools(mem.scope(user_id="alice", agent_id="support"))
    request(tools=tools.schemas("anthropic"))      # or "openai", or "json" for Gemini-style declarations
    result = tools.handle(call.name, call.arguments)  # returns a JSON string for the tool result

``forget`` is off by default: enable it only if the model should be able to delete memories.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Union

if TYPE_CHECKING:  # pragma: no cover
    from .memory import ScopedMemory

_SPECS: List[Dict[str, Any]] = [
    {
        "name": "remember",
        "description": (
            "Save information worth keeping across conversations: user facts and preferences, decisions, "
            "outcomes of actions (especially failures), and step-by-step procedures. Do not save small talk "
            "or anything already in memory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "One self-contained statement."},
                "memory_type": {
                    "type": "string", "enum": ["semantic", "episodic", "procedural"],
                    "description": "semantic = facts/preferences, episodic = something that happened, "
                                   "procedural = how to do something. Omit to auto-detect.",
                },
                "key": {
                    "type": "string",
                    "description": "Stable id for a fact that can change, e.g. 'user.city'. "
                                   "Reusing a key replaces the old value.",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "recall",
        "description": "Search long-term memory. Use before answering questions about the user, past events "
                       "or how a task was done before.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in natural language."},
                "k": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Max results (default 5)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "forget",
        "description": "Permanently delete one memory by id, e.g. when the user asks you to forget something.",
        "parameters": {
            "type": "object",
            "properties": {"memory_id": {"type": "string", "description": "Id returned by recall."}},
            "required": ["memory_id"],
        },
    },
]


class MemoryTools:
    def __init__(self, memory: ScopedMemory, allow_forget: bool = False, max_k: int = 20) -> None:
        self.memory = memory
        self.allow_forget = allow_forget
        self.max_k = max_k

    @property
    def names(self) -> List[str]:
        return [s["name"] for s in _SPECS if s["name"] != "forget" or self.allow_forget]

    def schemas(self, fmt: str = "json") -> List[Dict[str, Any]]:
        """Tool definitions in ``"anthropic"``, ``"openai"`` or plain ``"json"`` (name/description/parameters)."""
        specs = [json.loads(json.dumps(s)) for s in _SPECS if s["name"] in self.names]
        if fmt == "json":
            return specs
        if fmt == "anthropic":
            return [{"name": s["name"], "description": s["description"], "input_schema": s["parameters"]}
                    for s in specs]
        if fmt == "openai":
            return [{"type": "function", "function": s} for s in specs]
        raise ValueError("fmt must be 'json', 'anthropic' or 'openai'")

    def handle(self, name: str, arguments: Union[str, Mapping[str, Any], None]) -> str:
        """Run a tool call and return a JSON string. Errors come back as ``{"error": ...}``, never raise."""
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
            if name not in self.names:
                raise ValueError(f"Unknown or disabled tool {name!r}")
            return json.dumps(getattr(self, f"_{name}")(**args))
        except Exception as exc:  # noqa: BLE001 - the model gets the error and can retry
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    def _remember(self, content: str, memory_type: Any = None, key: Any = None) -> Dict[str, Any]:
        record = self.memory.add(content, memory_type or None, key=key or None)
        return {"id": record.id, "memory_type": record.memory_type.value, "kind": record.kind}

    def _recall(self, query: str, k: int = 5) -> Dict[str, Any]:
        k = max(1, min(int(k), self.max_k))
        hits = self.memory.search(query, k=k)
        return {"results": [{"id": h.id, "content": h.content, "memory_type": h.record.memory_type.value,
                             "score": round(h.score, 3)} for h in hits]}

    def _forget(self, memory_id: str) -> Dict[str, Any]:
        return {"deleted": self.memory.delete(memory_id)}
