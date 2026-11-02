"""Embedding providers.

Any object with ``name``, ``dim`` and ``embed(texts) -> list[list[float]]`` works.
``HashingEmbedder`` is the zero-dependency default: deterministic, offline, fast, and
good at lexical/morphological similarity. For real semantic recall in production, plug
in a neural model with :class:`FunctionEmbedder` or :class:`SentenceTransformerEmbedder`.
"""

from __future__ import annotations

import hashlib
import math
from typing import Callable, List, Optional, Protocol, Sequence, runtime_checkable

from .text import tokenize

Vector = List[float]


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> List[Vector]: ...


def normalize(vec: Sequence[float]) -> Vector:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return [0.0] * len(vec)
    return [x / norm for x in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Vectors stored by smart_memory are pre-normalized, so this is a dot product."""
    if len(a) != len(b):
        raise ValueError(f"Embedding dimension mismatch: {len(a)} != {len(b)}")
    return sum(x * y for x, y in zip(a, b))


class HashingEmbedder:
    """Feature-hashing embedder over stemmed words, word bigrams and character trigrams.

    No model download and no dependencies. Similar wording maps to similar vectors;
    paraphrases with no shared vocabulary will not. Use a neural embedder when that matters.
    """

    def __init__(self, dim: int = 512, word_weight: float = 1.0, bigram_weight: float = 0.5,
                 char_weight: float = 0.25) -> None:
        if dim < 16:
            raise ValueError("dim must be >= 16")
        self.dim = dim
        self.name = f"hashing-v1-{dim}"
        self._weights = (word_weight, bigram_weight, char_weight)

    def _bucket(self, feature: str) -> tuple:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def _embed_one(self, text: str) -> Vector:
        vec = [0.0] * self.dim
        tokens = tokenize(text)
        word_w, bigram_w, char_w = self._weights
        for tok in tokens:
            idx, sign = self._bucket("w:" + tok)
            vec[idx] += sign * word_w
            padded = f"#{tok}#"
            for i in range(len(padded) - 2):
                idx, sign = self._bucket("c:" + padded[i:i + 3])
                vec[idx] += sign * char_w
        for a, b in zip(tokens, tokens[1:]):
            idx, sign = self._bucket(f"b:{a}_{b}")
            vec[idx] += sign * bigram_w
        return normalize(vec)

    def embed(self, texts: Sequence[str]) -> List[Vector]:
        return [self._embed_one(t) for t in texts]


class FunctionEmbedder:
    """Wrap any callable ``fn(list[str]) -> list[list[float]]`` (OpenAI, Voyage, Cohere, local model...).

    Example::

        from openai import OpenAI
        client = OpenAI()

        def embed(texts):
            resp = client.embeddings.create(model="text-embedding-3-small", input=list(texts))
            return [d.embedding for d in resp.data]

        embedder = FunctionEmbedder(embed, dim=1536, name="openai-3-small")
    """

    def __init__(self, fn: Callable[[List[str]], Sequence[Sequence[float]]], dim: int, name: str,
                 batch_size: int = 64) -> None:
        self._fn = fn
        self.dim = dim
        self.name = name
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> List[Vector]:
        out: List[Vector] = []
        texts = list(texts)
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            vectors = self._fn(batch)
            if len(vectors) != len(batch):
                raise ValueError(f"Embedder {self.name!r} returned {len(vectors)} vectors for {len(batch)} texts")
            for v in vectors:
                if len(v) != self.dim:
                    raise ValueError(f"Embedder {self.name!r} returned dim {len(v)}, expected {self.dim}")
                out.append(normalize([float(x) for x in v]))
        return out


class SentenceTransformerEmbedder:
    """Local neural embeddings via ``sentence-transformers`` (``pip install smart-memory[local]``)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "SentenceTransformerEmbedder needs sentence-transformers: pip install 'smart-memory[local]'"
            ) from exc
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"st-{model_name}"

    def embed(self, texts: Sequence[str]) -> List[Vector]:  # pragma: no cover - optional dependency
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [normalize([float(x) for x in v]) for v in vectors]
