"""CrewAI adapter (``pip install crewai``): memory as tools a crew's agents can call.

    from smart_memory.integrations.crewai import memory_tools

    agent = Agent(role="Support agent", goal="...", backstory="...", tools=memory_tools(alice))

Tools are built from the same specs as :class:`MemoryTools`, so they behave exactly like
the MCP and function-calling tools: ``remember``, ``recall``, ``context`` (+ ``forget`` opt-in).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

try:
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field, create_model
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("The CrewAI adapter needs crewai: pip install crewai") from exc

from ..memory import ScopedMemory
from ..tools import MemoryTools

_TYPES = {"string": str, "integer": int, "number": float, "boolean": bool}


def _args_model(name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
    required = set(schema.get("required", []))
    fields: Dict[str, Any] = {}
    for prop, spec in schema["properties"].items():
        py_type = _TYPES.get(spec.get("type", "string"), str)
        if prop in required:
            fields[prop] = (py_type, Field(..., description=spec.get("description", "")))
        else:
            fields[prop] = (Optional[py_type], Field(None, description=spec.get("description", "")))
    return create_model(f"{name.title()}Args", **fields)


def memory_tools(memory: ScopedMemory, allow_forget: bool = False, prefix: str = "memory_") -> List[BaseTool]:
    """CrewAI tools bound to one scope. Each returns the same JSON string as :class:`MemoryTools`."""
    toolkit = MemoryTools(memory, allow_forget=allow_forget)
    tools: List[BaseTool] = []
    for spec in toolkit.schemas("json"):
        tool_name = spec["name"]

        def _run(self: BaseTool, _tool: str = tool_name, **kwargs: Any) -> str:
            return toolkit.handle(_tool, {k: v for k, v in kwargs.items() if v is not None})

        cls = type(
            f"SmartMemory{tool_name.title()}Tool",
            (BaseTool,),
            {
                "__module__": __name__,
                "__annotations__": {"name": str, "description": str, "args_schema": Type[BaseModel]},
                "name": f"{prefix}{tool_name}",
                "description": spec["description"],
                "args_schema": _args_model(tool_name, spec["parameters"]),
                "_run": _run,
            },
        )
        tools.append(cls())
    return tools
