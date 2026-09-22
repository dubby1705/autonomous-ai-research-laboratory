"""
AARL Discovery — small text utilities
=====================================
Deterministic text helpers shared by the discovery layer. Nothing here is
semantic: every function is a documented, reproducible string operation so a
direction label can always be traced back to the exact characters it came from.
"""

from __future__ import annotations

import re
from typing import List

#: Words removed before any keyword matching.
STOPWORDS = frozenset(
    """
    the a an of and or to in on for with by is are was were be been it its that this
    as at from than then which we can may will would should could how what why when
    where design build create develop make do does new novel better best improved
    improve improving optimize optimise optimised optimized increase decrease reduce
    reducing using use used use better higher lower more less most least very
    into over under about between across during before after above below
    """.split()
)

#: Leading imperative verbs removed when extracting the object of study.
_LEADING_VERBS = (
    "design", "designing", "build", "building", "create", "creating", "develop",
    "developing", "make", "making", "improve", "improving", "optimize", "optimise",
    "investigate", "investigating", "explore", "exploring", "find", "finding",
    "propose", "proposing", "study", "studying", "engineer", "engineering",
    "how", "can", "could", "we", "to", "a", "an", "the",
)

#: Adjectives skipped when looking for the object of study.
_SKIPPABLE_ADJECTIVES = frozenset(
    """
    better best new novel improved improving faster cheaper efficient optimal
    advanced high low next general efficient sustainable scalable robust
    lightweight compact cheaper safer stronger
    """.split()
)

#: Tokens that are real acronyms but describe the *purpose*, not the object.
_ACRONYM_IGNORE = frozenset({"ai", "ml", "iot", "api", "ui", "os"})

_WORD = re.compile(r"[A-Za-z0-9_\-]+")
_ACRONYM = re.compile(r"\b([A-Z][A-Z0-9]{1,7})\b")


def content_tokens(text: str) -> List[str]:
    """Lower-cased content words (stopwords and 1-character tokens removed)."""
    words = [w.lower() for w in _WORD.findall(str(text or ""))]
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def content_token_set(text: str) -> set:
    return set(content_tokens(text))


def overlap_ratio(left: str, right: str) -> float:
    """Fraction of ``left``'s content tokens that appear in ``right`` (0..1)."""
    a, b = content_token_set(left), content_token_set(right)
    if not a:
        return 0.0
    return len(a & b) / float(len(a))


def subject_of(problem: str) -> str:
    """Best-effort *object of study* label for a research problem.

    Heuristic, in order:

    1. the longest ALL-CAPS acronym token of 2..7 characters (``GPU`` in
       "Design a GPU for AI") — acronyms are the strongest signal;
    2. otherwise the first content token after leading verbs/adjectives
       (``battery`` in "Design a better battery with higher energy density");
    3. otherwise the first content token, else ``"the system"``.

    The result is used for *labelling directions only*; it is never treated as
    an established fact about the problem.
    """
    text = str(problem or "")
    acronyms = [
        a for a in _ACRONYM.findall(text) if a.lower() not in _ACRONYM_IGNORE
    ]
    if acronyms:
        return max(acronyms, key=len)

    tokens = [w.lower() for w in _WORD.findall(text)]
    index = 0
    while index < len(tokens) and tokens[index] in _LEADING_VERBS:
        index += 1
    while index < len(tokens) and (
        tokens[index] in _SKIPPABLE_ADJECTIVES or tokens[index] in STOPWORDS
    ):
        index += 1
    if index < len(tokens):
        return tokens[index]
    fallback = content_tokens(text)
    return fallback[0] if fallback else "the system"


def display_term(token: str) -> str:
    """Title-case a single token for display, preserving pure acronyms."""
    token = str(token or "").strip()
    if not token:
        return ""
    if token.isupper() and len(token) <= 8:
        return token
    return token[:1].upper() + token[1:]


def clean_label(text: str, limit: int = 90) -> str:
    """One-line, whitespace-collapsed label."""
    value = " ".join(str(text or "").split())
    return value[:limit].strip()


def sentence(text: str) -> str:
    """Whitespace-collapsed text with a trailing period, for report prose."""
    value = " ".join(str(text or "").split()).strip()
    if not value:
        return ""
    return value if value.endswith((".", "?", "!")) else value + "."
