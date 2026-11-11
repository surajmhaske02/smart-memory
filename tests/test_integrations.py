"""Framework adapters, tested against the real libraries (skipped when not installed)."""

import json

import pytest


def test_langchain_retriever_history_and_context(alice):
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from smart_memory.integrations.langchain import SmartMemoryChatHistory, SmartMemoryRetriever, context_message

    alice.add("Alice prefers email, never phone calls")
    alice.record_event("Refund API timed out for order 4411", outcome="failure")

    docs = SmartMemoryRetriever(memory=alice, k=3).invoke("how to contact Alice")
    assert docs[0].page_content == "Alice prefers email, never phone calls"
    assert docs[0].metadata["memory_type"] == "semantic" and "search_text" not in docs[0].metadata
    filtered = SmartMemoryRetriever(memory=alice, search_kwargs={"memory_types": ["episodic"]}).invoke("refund")
    assert [d.metadata["kind"] for d in filtered] == ["failure"]

    history = SmartMemoryChatHistory(alice)
    history.add_messages([HumanMessage("I moved to Berlin"), AIMessage("Noted!")])
    assert [type(m) for m in history.messages] == [HumanMessage, AIMessage]
    assert alice.search("Berlin")[0].content == "I moved to Berlin"  # user turn retained long-term
    history.clear()
    assert history.messages == []

    msg = context_message(alice, "refund for Alice")
    assert isinstance(msg, SystemMessage) and "Refund API timed out" in msg.content
    assert "no relevant memories" in context_message(alice, "zzz qqq").content


def test_llamaindex_retriever(alice):
    pytest.importorskip("llama_index.core")
    from smart_memory.integrations.llamaindex import SmartMemoryLlamaRetriever

    alice.add_many(["Alice lives in Pune", "The board meeting is in March"])
    nodes = SmartMemoryLlamaRetriever(alice, k=2).retrieve("where does Alice live")
    assert nodes[0].node.get_content() == "Alice lives in Pune"
    assert nodes[0].score > 0 and nodes[0].node.metadata["memory_type"] == "semantic"


def test_crewai_tools(alice):
    pytest.importorskip("crewai")
    from crewai.tools import BaseTool

    from smart_memory.integrations.crewai import memory_tools

    tools = memory_tools(alice)
    assert [t.name for t in tools] == ["memory_remember", "memory_recall", "memory_context"]
    assert all(isinstance(t, BaseTool) for t in tools)
    remember, recall, context = tools
    saved = json.loads(remember.run(content="Priya prefers WhatsApp"))
    assert saved["memory_type"] == "semantic"
    assert json.loads(recall.run(query="WhatsApp"))["results"][0]["content"] == "Priya prefers WhatsApp"
    assert "Priya prefers WhatsApp" in json.loads(context.run(query="how to message Priya"))["context"]
    assert [t.name for t in memory_tools(alice, allow_forget=True)][-1] == "memory_forget"
