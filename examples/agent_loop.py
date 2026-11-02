"""A provider-agnostic agent loop with smart_memory.

Replace ``call_llm`` with your model client (Anthropic, OpenAI, Gemini, a local model...).
The fake below just echoes, so the example runs offline: python examples/agent_loop.py
"""

from __future__ import annotations

from typing import Dict, List

from smart_memory import CallableRouter, Memory

SYSTEM = "You are a helpful support agent. Use the memory below; it is the source of truth.\n\n{memory}"


def call_llm(system: str, messages: List[Dict[str, str]]) -> str:
    """Plug your model in here. Must return the assistant's text."""
    if system.startswith("Summarize"):  # fake summary: the user's lines
        lines = [ln[6:] for ln in messages[-1]["content"].splitlines() if ln.startswith("user: ")]
        return "In this session the user said: " + " / ".join(lines)
    return f"(fake reply to: {messages[-1]['content']!r}, using {system.count(chr(10) + '- ')} memory lines)"


def summarize(text: str) -> str:
    """Used to compress a session or a cluster of episodes. Usually one cheap LLM call."""
    return call_llm("Summarize the key facts and outcomes in 1-2 sentences.", [{"role": "user", "content": text}])


def classify(text: str) -> str:
    """Optional: route memories with a small fine-tuned model or an LLM instead of rules.

    Return "episodic", "semantic" or "procedural" (optionally with a confidence tuple).
    Exceptions and low confidence fall back to the rule router automatically.
    """
    raise NotImplementedError  # falls back to RuleRouter in this demo


def chat(memory, user_text: str, history: List[Dict[str, str]]) -> str:
    card = memory.context(user_text, max_tokens=600)        # REUSE: only what this turn needs
    history.append({"role": "user", "content": user_text})
    reply = call_llm(SYSTEM.format(memory=card.render()), history)
    history.append({"role": "assistant", "content": reply})

    memory.add_turn("user", user_text)                       # working memory
    memory.add_turn("assistant", reply)
    memory.add(user_text)                                    # RETAIN: router decides the store
    memory.feedback(card.memory_ids, helpful=True)           # REFINE: reward what was used
    return reply


if __name__ == "__main__":
    mem = Memory("demo_agent.db", router=CallableRouter(classify))
    session = mem.scope(user_id="u-42", agent_id="support", thread_id="2026-11-03")
    history: List[Dict[str, str]] = []
    for text in ["Hi, I'm Priya. I prefer replies in Hindi.",
                 "My order 7781 arrived broken yesterday.",
                 "Can you refund order 7781?"]:
        print("user:", text)
        print("bot: ", chat(session, text, history))

    session.end_session(summarizer=summarize)                # working -> episodic
    print(session.refine(summarizer=summarize))              # dedupe, decay, promote

    # Next week, a new thread still remembers the user.
    later = session.with_thread("2026-11-10")
    print(later.context("What language does the user prefer?").render())
    mem.traces.export_jsonl("demo_traces.jsonl")             # data for fine-tuning router/reranker
    mem.close()
