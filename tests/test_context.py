import pytest

from smart_memory import ContextBuilder, Memory


@pytest.fixture
def loaded(alice):
    alice.glossary.define("refund window", "30 days from delivery", aliases=["return window"])
    alice.graph.relate("Alice", "customer_of", "Acme Corp")
    alice.add_procedure("issue_refund", ["Verify order", "Check refund window", "Call refund API"],
                        when="customer asks for a refund")
    alice.add("Alice prefers email over phone calls")
    alice.record_event("Refund API call for Alice timed out", outcome="failure")
    alice.add("The office coffee machine is broken")
    alice.working.set("goal", "process refund for order 123")
    return alice


def test_card_has_every_relevant_section(loaded):
    card = loaded.context("Alice wants a refund. What is the return window?")
    titles = {s.title: s for s in card.sections if s.lines}
    assert set(titles) == {"Current state", "Definitions", "Procedures & instructions", "Facts", "Past events"}
    assert titles["Current state"].lines == ["goal: process refund for order 123"]
    assert titles["Definitions"].lines == ["refund window (aka return window): 30 days from delivery"]
    assert titles["Procedures & instructions"].lines[0].startswith("issue_refund (use when: customer asks")
    assert "Alice customer of Acme Corp" in titles["Facts"].lines
    assert titles["Past events"].lines[0].startswith("(just now) [failure] Refund API")
    assert "coffee" not in card.render()
    assert card.entities == ["Alice"]
    assert len(card.memory_ids) == len(set(card.memory_ids))


def test_card_respects_token_budget(loaded):
    for i in range(40):
        loaded.add(f"Alice refund note number {i}: customer contacted support about a refund delay", "semantic")
    for budget in (40, 120, 400):
        card = loaded.context("Alice refund", max_tokens=budget, k=50)
        assert card.token_estimate <= budget
    big = loaded.context("Alice refund", max_tokens=2000, k=50)
    small = loaded.context("Alice refund", max_tokens=120, k=50)
    assert small.truncated and len(small.memory_ids) < len(big.memory_ids)
    # Priority: state and definitions survive even tiny budgets.
    tiny = loaded.context("Alice refund return window", max_tokens=40)
    assert [s.title for s in tiny.sections if s.lines][0] == "Current state"


def test_render_formats(loaded):
    card = loaded.context("refund window")
    md = card.render()
    assert md.startswith("## Memory\n### Current state\n- goal:")
    assert str(card) == md
    xml = card.render("xml")
    assert xml.startswith("<memory>") and "<definitions>" in xml and xml.endswith("</memory>")
    with pytest.raises(ValueError):
        card.render("html")
    data = card.to_dict()
    assert data["query"] == "refund window" and data["sections"]


def test_empty_card(mem):
    card = mem.scope("nobody").context("anything")
    assert card.is_empty and card.render() == "" and card.token_estimate == 0


def test_recent_turns_and_working_toggle(alice):
    alice.add_turn("user", "hi, I need help with billing")
    alice.add_turn("assistant", "Happy to help")
    card = alice.context("billing", recent_turns=1)
    conv = [s for s in card.sections if s.title == "Recent conversation"][0]
    assert conv.lines == ["assistant: Happy to help"]
    assert alice.context("billing", recent_turns=1, include_working=False).is_empty


def test_context_touches_and_traces(loaded, mem):
    card = loaded.context("Alice refund")
    record = loaded.get(card.memory_ids[0])
    assert record.access_count >= 1
    (trace,) = list(mem.traces.events("context"))
    assert trace["payload"]["memory_ids"] == card.memory_ids


def test_custom_token_counter(clock):
    words = lambda text: len(text.split())  # noqa: E731
    with Memory(clock=clock, context_builder=ContextBuilder(token_counter=words)) as m:
        s = m.scope("u")
        s.add("Alice likes long walks on the beach")
        card = s.context("Alice walks beach", max_tokens=12)
        assert card.token_estimate == len(card.render().split())
