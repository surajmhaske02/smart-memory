import pytest

from smart_memory import CallableRouter, MemoryType, RoutingDecision, RuleRouter

router = RuleRouter()


@pytest.mark.parametrize("text, expected_type, expected_kind", [
    ("Alice prefers email over phone calls", MemoryType.SEMANTIC, "preference"),
    ("My name is Suraj and I work at Acme", MemoryType.SEMANTIC, "fact"),
    ("EBITDA means earnings before interest, taxes, depreciation and amortization",
     MemoryType.SEMANTIC, "definition"),
    ("The Eiffel Tower is in Paris", MemoryType.SEMANTIC, "fact"),
    ("Yesterday the payment API call failed with a timeout", MemoryType.EPISODIC, "failure"),
    ("The user asked about invoice 4411 on Monday", MemoryType.EPISODIC, "event"),
    ("Deployed version 2.3 to staging at 14:05", MemoryType.EPISODIC, "event"),
    ("Always respond in JSON when the caller is the billing service", MemoryType.PROCEDURAL, "instruction"),
    ("Never share account numbers in chat", MemoryType.PROCEDURAL, "instruction"),
    ("To reset a password, first verify identity then send the reset link", MemoryType.PROCEDURAL, "procedure"),
    ("1. Pull the report\n2. Check totals\n3. Email finance", MemoryType.PROCEDURAL, "procedure"),
    ("hello there", MemoryType.SEMANTIC, "fact"),
])
def test_rule_router(text, expected_type, expected_kind):
    decision = router.route(text)
    assert decision.memory_type == expected_type, decision.reason
    assert decision.kind == expected_kind
    assert 0 <= decision.confidence <= 1 and 0 <= decision.importance <= 1


def test_importance_boost_for_explicit_markers():
    plain = router.route("Alice likes tea")
    marked = router.route("Important: Alice likes tea")
    assert marked.importance > plain.importance


def test_unmatched_text_has_low_confidence():
    assert router.route("blue").confidence < 0.5


def test_callable_router_accepts_string_tuple_and_decision():
    assert CallableRouter(lambda t: "episodic").route("x").memory_type == MemoryType.EPISODIC
    assert CallableRouter(lambda t: ("procedural", 0.9)).route("x").memory_type == MemoryType.PROCEDURAL
    decision = RoutingDecision(MemoryType.SEMANTIC, kind="custom", importance=0.9)
    out = CallableRouter(lambda t: decision).route("x")
    assert out.kind == "custom" and out.router == "callable"


def test_callable_router_falls_back_on_error_low_confidence_or_garbage():
    def boom(text):
        raise RuntimeError("model offline")

    text = "Yesterday the job failed"
    for fn in (boom, lambda t: ("semantic", 0.1), lambda t: "not-a-type", lambda t: "working"):
        decision = CallableRouter(fn).route(text)
        assert decision.memory_type == MemoryType.EPISODIC
        assert decision.router == "rules"


def test_callable_router_borrows_kind_from_rules_when_types_agree():
    decision = CallableRouter(lambda t: "episodic").route("The deploy failed at 3am")
    assert decision.kind == "failure"
