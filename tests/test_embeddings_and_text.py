import math

import pytest

from smart_memory import FunctionEmbedder, HashingEmbedder
from smart_memory.embeddings import cosine
from smart_memory.retrieval import bm25_scores
from smart_memory.text import estimate_tokens, stem, tokenize, truncate


@pytest.mark.parametrize("words", [
    ("prefers", "preferred", "preferences", "prefer"),
    ("running", "runs", "run"),
    ("failed", "fails", "fail"),
    ("policies", "policy"),
])
def test_stemmer_groups_inflections(words):
    assert len({stem(w) for w in words}) == 1


def test_tokenize_drops_stopwords():
    assert tokenize("What is the refund policy?") == ["refund", "policy"]


def test_estimate_and_truncate():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd" * 10) == 10
    assert truncate("a  b   c", 100) == "a b c"
    assert truncate("x" * 50, 10) == "xxxxxxx..."


def test_hashing_embedder_is_deterministic_and_normalized():
    e = HashingEmbedder(dim=128)
    v1, v2 = e.embed(["refund policy", "refund policy"])
    assert v1 == v2 and len(v1) == 128
    assert math.isclose(sum(x * x for x in v1), 1.0, rel_tol=1e-9)
    assert HashingEmbedder(dim=128).embed(["x y z"]) == e.embed(["x y z"])  # stable across instances


def test_hashing_embedder_similarity_ordering():
    e = HashingEmbedder()
    q, near, far = e.embed(["customer refund request failed", "the refund for the customer failed",
                            "quarterly revenue grew in Europe"])
    assert cosine(q, near) > 0.5 > cosine(q, far)


def test_empty_text_embeds_to_zero_vector():
    (v,) = HashingEmbedder(dim=32).embed(["the of and"])
    assert v == [0.0] * 32


def test_function_embedder_normalizes_and_validates():
    e = FunctionEmbedder(lambda texts: [[3.0, 4.0] for _ in texts], dim=2, name="fake", batch_size=2)
    assert e.embed(["a", "b", "c"]) == [[0.6, 0.8]] * 3
    with pytest.raises(ValueError, match="dim"):
        FunctionEmbedder(lambda texts: [[1.0] for _ in texts], dim=2, name="bad").embed(["a"])
    with pytest.raises(ValueError, match="vectors"):
        FunctionEmbedder(lambda texts: [], dim=2, name="bad").embed(["a"])


def test_cosine_dimension_mismatch():
    with pytest.raises(ValueError):
        cosine([1.0], [1.0, 0.0])


def test_bm25_prefers_documents_matching_more_query_terms():
    docs = [tokenize("refund policy for orders"), tokenize("refund"), tokenize("weather today")]
    scores = bm25_scores(tokenize("refund policy"), docs)
    assert scores[0] > scores[1] > scores[2] == 0.0
    assert bm25_scores([], docs) == [0.0, 0.0, 0.0]
