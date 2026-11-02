import pytest


def test_relate_is_idempotent_and_queryable(alice):
    r1 = alice.graph.relate("Acme Corp", "supplier_of", "Globex")
    r2 = alice.graph.relate("acme corp", "SUPPLIER_OF", "globex")
    assert r1.id == r2.id
    alice.graph.relate("Globex", "subsidiary_of", "Initech")
    assert len(alice.graph.relations()) == 2
    assert [r.object for r in alice.graph.relations(subject="ACME CORP")] == ["Globex"]
    assert len(alice.graph.relations(entity="globex")) == 2
    assert alice.graph.entities() == ["Acme Corp", "Globex", "Initech"]
    assert r1.sentence == "Acme Corp supplier of Globex"


def test_exclusive_relation_replaces_object(alice):
    alice.graph.relate("Alice", "lives_in", "Pune", exclusive=True)
    alice.graph.relate("Alice", "lives_in", "Berlin", exclusive=True)
    assert [r.object for r in alice.graph.relations(subject="Alice")] == ["Berlin"]


def test_neighbors_and_multi_hop(alice):
    g = alice.graph
    g.relate("Acme", "supplier_of", "Globex")
    g.relate("Globex", "subsidiary_of", "Initech")
    g.relate("Initech", "based_in", "Austin")
    assert g.neighbors("Acme") == {"Globex"}
    assert g.neighbors("acme", depth=2) == {"Globex", "Initech"}
    assert {r.object for r in g.facts_about("Acme", depth=3)} == {"Globex", "Initech", "Austin"}


def test_unrelate(alice):
    alice.graph.relate("A", "knows", "B")
    assert alice.graph.unrelate("a", "knows", "b") is True
    assert alice.graph.relations() == []
    assert alice.graph.unrelate("a", "knows", "b") is False


def test_relate_validates(alice):
    with pytest.raises(ValueError):
        alice.graph.relate("A", " ", "B")


def test_find_entities(alice):
    alice.graph.relate("Acme Corp", "supplier_of", "Globex")
    assert alice.graph.find_entities("What is our exposure to acme corp and Globex?") == ["Acme Corp", "Globex"]
    assert alice.graph.find_entities("nothing relevant") == []


def test_graph_is_scoped(mem):
    mem.scope("alice").graph.relate("Alice", "manages", "Project X")
    assert mem.scope("bob").graph.relations() == []


def test_glossary_define_lookup_find(alice):
    g = alice.glossary
    g.define("concentration", "Share of exposure held by the top 10 counterparties", aliases=["exposure concentration"])
    g.define("ARR", "Annual recurring revenue")
    assert g.lookup("Concentration").definition.startswith("Share")
    assert g.lookup("exposure concentration").term == "concentration"
    assert g.lookup("missing") is None
    assert [t.term for t in g.find_terms("What is our concentration risk and ARR?")] == ["concentration", "ARR"]
    assert g.find_terms("arr-based forecast") != []  # case-insensitive whole-word match
    assert g.find_terms("carry on") == []  # no substring false positives


def test_glossary_redefine_and_remove(alice):
    g = alice.glossary
    g.define("churn", "Customers lost per month")
    g.define("churn", "Share of customers lost per quarter")
    assert len(g.terms()) == 1 and g.lookup("churn").definition.startswith("Share")
    assert g.remove("churn") is True and g.terms() == []
    assert g.remove("churn") is False
