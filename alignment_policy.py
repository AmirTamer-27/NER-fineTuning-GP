"""Policies for projecting character spans onto XLM-R tokens."""

from __future__ import annotations

import unicodedata


def is_punctuation_or_symbol(value):
    """Return True when every character is Unicode punctuation or a symbol."""
    return bool(value) and all(
        unicodedata.category(character)[0] in {"P", "S"}
        for character in value
    )


def classify_boundary_conflict(text, token_start, token_end, entity_start, entity_end):
    """Classify a token that overlaps but is not contained by an entity span."""
    if entity_start <= token_start < token_end <= entity_end:
        return "exact", ""

    outside = ""
    if token_start < entity_start:
        outside += text[token_start:min(token_end, entity_start)]
    if token_end > entity_end:
        outside += text[max(token_start, entity_end):token_end]

    if is_punctuation_or_symbol(outside):
        return "punctuation", outside
    return "ambiguous", outside
