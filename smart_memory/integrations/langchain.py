"""LangChain adapters (``pip install langchain-core``).

    from smart_memory.integrations.langchain import (
        SmartMemoryRetriever, SmartMemoryChatHistory, context_message)

    retriever = SmartMemoryRetriever(memory=alice, k=5)          # any LangChain chain/agent
    history = SmartMemoryChatHistory(alice)                       # RunnableWithMessageHistory
    messages = [context_message(alice, question), HumanMessage(question)]
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.retrievers import BaseRetriever
    from pydantic import ConfigDict
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("The LangChain adapter needs langchain-core: pip install langchain-core") from exc

from ..memory import ScopedMemory


def _metadata(hit: Any) -> Dict[str, Any]:
    r = hit.record
    return {"id": r.id, "memory_type": r.memory_type.value, "kind": r.kind, "score": hit.score,
            "created_at": r.created_at, **{k: v for k, v in r.metadata.items() if k != "search_text"}}


class SmartMemoryRetriever(BaseRetriever):
    """Hybrid memory search as a LangChain retriever. Extra ``search_kwargs`` go to ``search()``."""

    memory: Any
    k: int = 5
    search_kwargs: Dict[str, Any] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        hits = self.memory.search(query, k=self.k, **self.search_kwargs)
        return [Document(page_content=h.content, metadata=_metadata(h)) for h in hits]


_ROLE = {"human": "user", "ai": "assistant", "system": "system"}
_MESSAGE = {"user": HumanMessage, "assistant": AIMessage, "system": SystemMessage}


class SmartMemoryChatHistory(BaseChatMessageHistory):
    """Chat history backed by working memory; user messages are also retained long-term.

    Args:
        memory: A scope with a ``thread_id`` (one history per conversation).
        retain_user_messages: Also ``add()`` each user message so the router can keep the facts.
    """

    def __init__(self, memory: ScopedMemory, retain_user_messages: bool = True) -> None:
        self.memory = memory
        self.retain_user_messages = retain_user_messages

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore[override]
        return [_MESSAGE.get(t.role, HumanMessage)(content=t.content) for t in self.memory.working.turns()]

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        for message in messages:
            text = message.content if isinstance(message.content, str) else str(message.content)
            role = _ROLE.get(message.type, message.type)
            self.memory.add_turn(role, text)
            if role == "user" and self.retain_user_messages and text.strip():
                self.memory.add(text)

    def clear(self) -> None:
        self.memory.working.clear()


def context_message(memory: ScopedMemory, query: str, max_tokens: int = 800, **kwargs: Any) -> SystemMessage:
    """A ``SystemMessage`` holding the Context Card for ``query``."""
    card = memory.context(query, max_tokens=max_tokens, **kwargs)
    return SystemMessage(content=card.render() or "## Memory\n(no relevant memories)")
