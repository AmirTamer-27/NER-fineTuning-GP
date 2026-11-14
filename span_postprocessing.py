"""Conservative entity-specific cleanup for decoded character spans.

This operates after BIO decoding. Internal punctuation is never removed; only
punctuation or symbols at the two outer boundaries can be trimmed.
"""

from __future__ import annotations

import unicodedata


# These entity types normally contain words or digits at their outer edges.
# The set is intentionally explicit so it can be reviewed as the ontology
# evolves. Punctuation inside a span (dates, hyphenated names, etc.) is kept.
TRIM_BOUNDARY_PUNCTUATION_FOR = frozenset({
    "AGE",
    "APPLICATION_REFERENCE",
    "BANK_ACCOUNT_NUMBER",
    "CITY",
    "COUNTRY",
    "CUSTOMER_ID",
    "DATE",
    "DISPUTE_ID",
    "EMAIL_ADDRESS",
    "IBAN",
    "JOB_TITLE",
    "NATIONALITY",
    "NATIONAL_ID_NUMBER",
    "PARTIAL_CARD_NUMBER",
    "PASSPORT_NUMBER",
    "PAYMENT_CARD_NUMBER",
    "PERSON_NAME",
    "QUANTITY",
    "SERVICE_REQUEST_ID",
    "TIME",
    "TRANSACTION_REFERENCE",
})


def is_punctuation_or_symbol(character):
    return unicodedata.category(character)[0] in {"P", "S"}


def trim_decoded_span(text, start, end, label):
    """Return adjusted ``(start, end)`` without changing the document text."""
    if label not in TRIM_BOUNDARY_PUNCTUATION_FOR:
        return start, end
    while start < end and is_punctuation_or_symbol(text[start]):
        start += 1
    while end > start and is_punctuation_or_symbol(text[end - 1]):
        end -= 1
    return start, end
