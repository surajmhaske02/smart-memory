"""Small, dependency-free text utilities: tokenization, stemming, token estimates."""

from __future__ import annotations

import math
import re
from typing import List

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for with from as into onto over under
    is are was were be been being am do does did doing have has had having it its it's this that
    these those there here i me my mine we us our ours you your yours he him his she her hers they
    them their theirs what which who whom whose when where why how all any both each few more most
    other some such no nor not only own same so than too very can will just should would could
    may might must shall about above below up down out off again further once also s t
    """.split()
)

# Ordered longest-first so the most specific suffix wins.
_SUFFIXES = ("ational", "ization", "fulness", "ousness", "iveness", "ingly", "ation", "ment",
             "ness", "ence", "ance", "edly", "ing", "ied", "ed", "ly")


def stem(word: str) -> str:
    """A tiny suffix-stripping stemmer. Maps 'prefers'/'preferred'/'preferences' to 'prefer'."""
    if len(word) <= 3 or word.isdigit():
        return word
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + ("y" if suffix == "ied" else "")
            # "preferr" -> "prefer", "runn" -> "run"
            if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "lsz":
                word = word[:-1]
            break
    return word


def words(text: str) -> List[str]:
    """Lowercased word tokens, stopwords included."""
    return _WORD_RE.findall(text.lower())


def tokenize(text: str) -> List[str]:
    """Lowercased, stemmed content tokens with stopwords removed."""
    return [stem(w.replace("'", "")) for w in words(text) if w not in STOPWORDS]


def estimate_tokens(text: str) -> int:
    """Rough LLM token estimate (~4 chars per token). Swap in a real tokenizer if you need precision."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def truncate(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."
