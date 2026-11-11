"""Small, dependency-free text utilities: tokenization, stemming, token estimates.

Tokenization is Unicode-aware: words in any script (Devanagari, Latin with accents,
Cyrillic, ...) are kept whole, including combining vowel signs. English suffix stemming
and irregular-verb lemmas apply only to ASCII words.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Dict, List


def _mark_class() -> str:
    """Character-class ranges for every combining mark (category M*) in the BMP.

    Python's ``\\w`` treats marks such as the Devanagari vowel signs in "प्रिया" as word
    breaks, which shreds Indic words. Adding marks back keeps them whole.
    """
    ranges, start, prev = [], None, None
    for cp in range(0x300, 0x10000):
        if unicodedata.category(chr(cp)).startswith("M"):
            if start is None:
                start = cp
            elif cp != prev + 1:
                ranges.append((start, prev))
                start = cp
            prev = cp
    if start is not None:
        ranges.append((start, prev))
    return "".join(f"\\u{a:04x}-\\u{b:04x}" if a != b else f"\\u{a:04x}" for a, b in ranges)


_WORD_CHAR = rf"(?:[^\W_]|[{_mark_class()}])"
_WORD_RE = re.compile(rf"{_WORD_CHAR}+(?:['\u2019]{_WORD_CHAR}+)?")
_NUMBER_RE = re.compile(r"\d+(?:[.,:/-]\d+)*")
_NEGATIONS = frozenset({"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "cannot",
                        "can't", "don't", "doesn't", "didn't", "won't", "isn't", "aren't", "wasn't",
                        "weren't", "shouldn't", "wouldn't", "couldn't", "hasn't", "haven't", "hadn't",
                        # Hindi / Marathi
                        "नहीं", "नही", "मत", "ना", "नाही", "नको", "न"})

STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for with from as into onto over under
    is are was were be been being am do does did doing have has had having it its it's this that
    these those there here i me my mine we us our ours you your yours he him his she her hers they
    them their theirs what which who whom whose when where why how all any both each few more most
    other some such no nor not only own same so than too very can will just should would could
    may might must shall about above below up down out off again further once also s t
    है हैं था थी थे में को का की के से पर और या यह वह ये वो भी तो ही ने एक इस उस कि
    आहे आहेत होता होती होते आणि मध्ये ला चा ची चे हे ते की व पण
    """.split()
)

# Irregular English forms -> base word, so "bought" matches "buy". Extend freely.
LEMMAS: Dict[str, str] = {
    "bought": "buy", "brought": "bring", "thought": "think", "caught": "catch", "taught": "teach",
    "sought": "seek", "fought": "fight", "went": "go", "gone": "go", "came": "come", "made": "make",
    "took": "take", "taken": "take", "gave": "give", "given": "give", "got": "get", "gotten": "get",
    "saw": "see", "seen": "see", "said": "say", "told": "tell", "knew": "know", "known": "know",
    "found": "find", "left": "leave", "felt": "feel", "kept": "keep", "held": "hold", "met": "meet",
    "paid": "pay", "sent": "send", "spent": "spend", "built": "build", "lost": "lose", "won": "win",
    "sold": "sell", "wrote": "write", "written": "write", "ran": "run", "began": "begin", "begun": "begin",
    "broke": "break", "broken": "break", "chose": "choose", "chosen": "choose", "drove": "drive",
    "driven": "drive", "ate": "eat", "eaten": "eat", "fell": "fall", "fallen": "fall", "flew": "fly",
    "flown": "fly", "forgot": "forget", "forgotten": "forget", "grew": "grow", "grown": "grow",
    "hid": "hide", "hidden": "hide", "led": "lead", "lent": "lend", "meant": "mean", "rode": "ride",
    "rose": "rise", "risen": "rise", "sang": "sing", "sung": "sing", "sat": "sit", "slept": "sleep",
    "spoke": "speak", "spoken": "speak", "stood": "stand", "stole": "steal", "stolen": "steal",
    "swam": "swim", "threw": "throw", "thrown": "throw", "understood": "understand", "woke": "wake",
    "wore": "wear", "worn": "wear", "children": "child", "people": "person", "men": "man", "women": "woman",
    "mice": "mouse", "feet": "foot", "teeth": "tooth", "bigger": "big", "biggest": "big", "better": "good",
    "best": "good", "worse": "bad", "worst": "bad",
}

# Ordered longest-first so the most specific suffix wins.
_SUFFIXES = ("ational", "ization", "fulness", "ousness", "iveness", "ingly", "ation", "ment",
             "ness", "ence", "ance", "edly", "ing", "ied", "ed", "ly")


def stem(word: str) -> str:
    """A tiny English stemmer: 'prefers'/'preferred'/'preferences' -> 'prefer', 'bought' -> 'buy'.

    Non-ASCII words (Hindi, Marathi, accented Latin, ...) are returned unchanged.
    """
    if word in LEMMAS:
        return LEMMAS[word]
    if len(word) <= 3 or word.isdigit() or not word.isascii():
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
    """Case-folded word tokens in any script, stopwords included."""
    return _WORD_RE.findall(unicodedata.normalize("NFC", text).casefold().replace("\u2019", "'"))


def tokenize(text: str) -> List[str]:
    """Case-folded, stemmed content tokens with stopwords removed."""
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


def same_specifics(a: str, b: str) -> bool:
    """True when two texts carry the same numbers and the same negation.

    Embeddings barely move when "order 1234" becomes "order 1235" or "likes" becomes
    "doesn't like", so similarity alone must never merge such pairs.
    """
    if sorted(_NUMBER_RE.findall(a)) != sorted(_NUMBER_RE.findall(b)):
        return False
    neg_a = sum(w in _NEGATIONS for w in words(a)) % 2
    neg_b = sum(w in _NEGATIONS for w in words(b)) % 2
    return neg_a == neg_b
