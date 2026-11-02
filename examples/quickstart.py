"""Tour of smart_memory in ~60 lines. Runs offline: python examples/quickstart.py"""

from smart_memory import Memory

mem = Memory()  # Memory("agent.db") to persist

# Global knowledge every user can read, only the global scope can change.
org = mem.scope()
org.glossary.define("refund window", "30 days from delivery date", aliases=["return window"])

# Everything else is scoped to a user + agent (+ conversation thread).
alice = mem.scope(user_id="alice", agent_id="support", thread_id="chat-001")

# RETAIN - the router picks the store (semantic / episodic / procedural).
alice.add("Alice prefers email, never phone calls")
alice.add("Always confirm the order number before issuing a refund")
alice.record_event("Refund API call for order 4411 timed out", outcome="failure")
alice.add_procedure(
    "issue_refund",
    ["Confirm order number", "Check the refund window", "Call refund API with retry + backoff"],
    when="customer asks for money back",
)
alice.add("Alice is a premium customer", pin=True)  # always opens the Context Card
alice.add("Alice lives in Pune", key="alice.city")
alice.add("Alice lives in Berlin", key="alice.city")  # supersedes Pune, history kept

# Ontology: explicit relationships instead of keyword guesses.
alice.graph.relate("Alice", "customer_of", "Acme Corp")
alice.graph.relate("Acme Corp", "plan", "Enterprise", exclusive=True)

# Working memory for the current conversation.
alice.working.set("goal", "refund order 4411")

# RECALL - hybrid search (vector + keyword + recency + importance + reuse).
for hit in alice.search("where does alice live"):
    print(f"{hit.score:.2f}  {hit.content}")

# REUSE - a compact, token-budgeted Context Card for your system prompt.
card = alice.context("Alice wants her money back for order 4411. What's the return window?", max_tokens=400)
print("\n" + card.render())
print(f"\n~{card.token_estimate} tokens, {len(card.memory_ids)} memories")
stable, dynamic = card.render_split()  # send `stable` as a cached prompt prefix

# REFINE - learn from outcomes and usage.
alice.feedback(card.memory_ids, helpful=True)
alice.record_outcome("issue_refund", success=True)
print("\n", alice.refine())

# Other agents of the same user can't read Alice's support-agent memories, other users can't either.
print("sales agent sees:", mem.scope("alice", "sales").search("email phone"))
print("bob sees:", mem.scope("bob", "support").search("Alice email"))
print("bob still sees the org glossary:", mem.scope("bob").glossary.lookup("return window"))
print("cache:", mem.cache_info())
