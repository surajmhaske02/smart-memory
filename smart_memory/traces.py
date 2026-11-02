"""Memory traces: a structured log of every routing, retrieval, feedback and refine decision.

Traces are the raw material for continual learning: export them to fine-tune the
router (``routing_dataset``), train rerankers on retrieval + feedback pairs, or audit
what an agent knew when it acted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Union

from .storage.base import StorageBackend
from .types import Scope


class TraceLog:
    def __init__(self, store: StorageBackend, enabled: bool = True,
                 clock: Callable[[], float] = time.time) -> None:
        self._store = store
        self.enabled = enabled
        self._clock = clock

    def log(self, event: str, scope: Scope, **payload: Any) -> None:
        if self.enabled:
            self._store.add_trace(event, self._clock(), scope, payload)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return self._store.iter_traces()

    def events(self, event: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        return self._store.iter_traces(event)

    def export_jsonl(self, path: Union[str, Path], event: Optional[str] = None) -> int:
        count = 0
        with open(path, "w", encoding="utf-8") as fh:
            for trace in self._store.iter_traces(event):
                fh.write(json.dumps(trace, default=str) + "\n")
                count += 1
        return count

    def routing_dataset(self, path: Optional[Union[str, Path]] = None) -> list:
        """``[{"text", "label", "source"}]`` for training a router; human corrections override router output."""
        labels: Dict[str, Dict[str, Any]] = {}
        for trace in self._store.iter_traces("route"):
            p = trace["payload"]
            labels[p["memory_id"]] = {"text": p["text"], "label": p["memory_type"], "source": p.get("router", "")}
        for trace in self._store.iter_traces("route_correction"):
            p = trace["payload"]
            if p["memory_id"] in labels:
                labels[p["memory_id"]].update(label=p["new_type"], source="correction")
        rows = list(labels.values())
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
        return rows
