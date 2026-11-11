"""LlamaIndex adapter (``pip install llama-index-core``).

    from smart_memory.integrations.llamaindex import SmartMemoryLlamaRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine

    retriever = SmartMemoryLlamaRetriever(alice, k=5)
    engine = RetrieverQueryEngine.from_args(retriever, llm=my_llm)
"""

from __future__ import annotations

from typing import Any, List

try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("The LlamaIndex adapter needs llama-index-core: pip install llama-index-core") from exc

from ..memory import ScopedMemory


class SmartMemoryLlamaRetriever(BaseRetriever):
    """Hybrid memory search as a LlamaIndex retriever. Extra keyword args go to ``search()``."""

    def __init__(self, memory: ScopedMemory, k: int = 5, **search_kwargs: Any) -> None:
        super().__init__()
        self.memory = memory
        self.k = k
        self.search_kwargs = search_kwargs

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        hits = self.memory.search(query_bundle.query_str, k=self.k, **self.search_kwargs)
        return [
            NodeWithScore(
                node=TextNode(id_=h.id, text=h.content,
                              metadata={"memory_type": h.record.memory_type.value, "kind": h.record.kind,
                                        "created_at": h.record.created_at}),
                score=h.score,
            )
            for h in hits
        ]
