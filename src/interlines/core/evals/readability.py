"""
Readability heuristics for editor/validator checks.

This module implements a lightweight readability metric that does not
depend on external libraries. The goal is not to match any specific
linguistic standard, but to obtain a stable score in [0.0, 1.0] that
correlates with how easy a piece of text is to read.

Heuristics
----------
We combine three simple components:

1. Sentence length (syntactic simplicity)
   - Shorter sentences are usually easier to read.
   - We compute the average number of words per sentence.

2. Word length (lexical complexity)
   - Shorter words tend to be more familiar.
   - We compute the average number of characters per word.

3. Sentence-length variation (a "burstiness"-like signal)
   - Human writing often alternates longer and shorter sentences.
   - We compute the coefficient of variation of sentence lengths:
     std(sentence_len) / mean(sentence_len).

Each component is mapped into a sub-score in [0.0, 1.0] and the final
readability score is a weighted average:

    score = 0.5 * sentence_score
          + 0.3 * lexical_score
          + 0.2 * variation_score

This is deliberately simple and deterministic so tests can rely on it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _clamp_01(value: float) -> float:
    """Clamp a floating-point value into the closed interval [0.0, 1.0]."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _sentence_lengths(text: str) -> list[int]:
    """Split text into sentences and return their lengths in words.

    Sentences are split on `.`, `?`, and `!`. Word-like tokens are
    matched with :data:`_WORD_RE`.
    """
    stripped = text.strip()
    if not stripped:
        return []

    parts = re.split(r"[.!?]+", stripped)
    lengths: list[int] = []

    for part in parts:
        tokens = _WORD_RE.findall(part)
        if not tokens:
            continue
        lengths.append(len(tokens))

    return lengths


def _word_lengths(text: str) -> list[int]:
    """Return a list of word lengths (in characters) for the given text."""
    tokens = _WORD_RE.findall(text)
    return [len(token) for token in tokens]


def _sentence_length_score(avg_len: float) -> float:
    """Map average sentence length to a score in [0.0, 1.0].

    Intuition:
    - Very short sentences (<= 8 words) are extremely easy -> 1.0
    - Around 12-18 words is a comfortable range -> ~0.8
    - Very long sentences (>= 35 words) are hard -> 0.0
    """
    if avg_len <= 8.0:
        return 1.0
    if avg_len >= 35.0:
        return 0.0
    # Between 8 and 18 words we gently reduce from 1.0 to 0.8,
    # then from 18 to 35 down to 0.0.
    if avg_len <= 18.0:
        # 8 -> 1.0, 18 -> 0.8
        ratio = (avg_len - 8.0) / (18.0 - 8.0)
        return _clamp_01(1.0 - 0.2 * ratio)

    # 18 -> 0.8, 35 -> 0.0
    ratio = (avg_len - 18.0) / (35.0 - 18.0)
    return _clamp_01(0.8 * (1.0 - ratio))


def _lexical_score(avg_word_len: float) -> float:
    """Map average word length to a score in [0.0, 1.0].

    Intuition:
    - Around 3.5-5 characters per word is typical/easy -> close to 1.0
    - Very long average words (>= 8 chars) are harder -> near 0.0
    - Extremely short averages (< 3 chars) may indicate fragmented or
      unnatural text, also slightly penalised.
    """
    if avg_word_len <= 3.0:
        # Slight penalty for extremely short tokens.
        return 0.8
    if avg_word_len <= 5.0:
        # 3 -> 0.8, 5 -> 1.0
        ratio = (avg_word_len - 3.0) / (5.0 - 3.0)
        return _clamp_01(0.8 + 0.2 * ratio)
    if avg_word_len >= 8.0:
        return 0.0

    # 5 -> 1.0, 8 -> 0.0
    ratio = (avg_word_len - 5.0) / (8.0 - 5.0)
    return _clamp_01(1.0 * (1.0 - ratio))


def _variation_score(lengths: list[int]) -> float:
    """Compute a simple 'burstiness-like' score for sentence length variation.

    We use the coefficient of variation (CV) of sentence lengths:

        CV = std(len) / mean(len)

    Intuition:
    - If there is only one sentence, we return a neutral 0.7.
    - Very low CV (sentences almost identical in length) scores around
      0.4 - monotonous rhythm.
    - A moderate CV (roughly 0.3-0.8) scores higher (~0.8).
    - Extremely high CV (> 1.2) is slightly penalised (~0.5) as it may
      indicate erratic structure.
    """
    n = len(lengths)
    if n == 0:
        return 0.0
    if n == 1:
        return 0.7

    mean_len = sum(lengths) / n
    if mean_len <= 0.0:
        return 0.0

    var = sum((length - mean_len) ** 2 for length in lengths) / n
    std = math.sqrt(var)
    cv = std / mean_len

    # Piecewise mapping of CV to [0, 1].
    if cv <= 0.1:
        return 0.4
    if cv >= 1.2:
        return 0.5
    if 0.3 <= cv <= 0.8:
        # Best zone: map 0.3 -> 0.7, 0.8 -> 0.9
        ratio = (cv - 0.3) / (0.8 - 0.3)
        return _clamp_01(0.7 + 0.2 * ratio)

    # Intermediate zone: smoothly interpolate between edges.
    # For cv in (0.1, 0.3): 0.4 -> 0.7
    if cv < 0.3:
        ratio = (cv - 0.1) / (0.3 - 0.1)
        return _clamp_01(0.4 + 0.3 * ratio)

    # For cv in (0.8, 1.2): 0.9 -> 0.5
    ratio = (cv - 0.8) / (1.2 - 0.8)
    return _clamp_01(0.9 - 0.4 * ratio)


def readability_score(text: str) -> float:
    """Compute a readability score in [0.0, 1.0] for a single text string.

    Parameters
    ----------
    text:
        The input text to analyse. It may contain multiple sentences.

    Returns
    -------
    float
        A score in [0.0, 1.0], where higher values indicate easier
        reading. Empty or whitespace-only text returns 0.0.

    Notes
    -----
    - The score combines sentence length, lexical complexity, and
      sentence-length variation.
    - The function is deterministic and does not rely on any external
      models or libraries.
    """
    stripped = text.strip()
    if not stripped:
        return 0.0

    sent_lengths = _sentence_lengths(stripped)
    if not sent_lengths:
        return 0.0

    word_lengths = _word_lengths(stripped)
    if not word_lengths:
        return 0.0

    avg_sent_len = sum(sent_lengths) / len(sent_lengths)
    avg_word_len = sum(word_lengths) / len(word_lengths)

    sentence_score = _sentence_length_score(avg_sent_len)
    lexical = _lexical_score(avg_word_len)
    variation = _variation_score(sent_lengths)

    # Weighted combination: sentence structure is most important,
    # lexical complexity second, variation third.
    score = 0.5 * sentence_score + 0.3 * lexical + 0.2 * variation
    return _clamp_01(score)


def aggregate_readability(texts: Sequence[str] | None) -> float:
    """Compute a readability score for a sequence of text segments.

    Parameters
    ----------
    texts:
        A sequence of strings (may contain empty strings). If ``None``
        or empty, the function returns 0.0.

    Returns
    -------
    float
        Readability score in [0.0, 1.0] for the concatenated text.
    """
    if not texts:
        return 0.0

    merged_parts = [text for text in texts if isinstance(text, str) and text.strip()]
    if not merged_parts:
        return 0.0

    merged = " ".join(merged_parts)
    return readability_score(merged)


__all__ = ["readability_score", "aggregate_readability"]
