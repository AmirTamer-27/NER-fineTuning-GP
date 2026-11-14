"""Unit tests for token/span boundary policies."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alignment_policy import classify_boundary_conflict
from span_postprocessing import trim_decoded_span


class AlignmentPolicyTests(unittest.TestCase):
    def test_punctuation_only_crossing_is_supervisable(self):
        kind, outside = classify_boundary_conflict("10-", 0, 3, 0, 2)
        self.assertEqual((kind, outside), ("punctuation", "-"))

    def test_alphanumeric_crossing_is_ambiguous(self):
        kind, outside = classify_boundary_conflict("CustomerABC", 0, 11, 8, 11)
        self.assertEqual((kind, outside), ("ambiguous", "Customer"))

    def test_postprocessing_keeps_internal_punctuation(self):
        text = "(14/11/2026),"
        self.assertEqual(
            trim_decoded_span(text, 0, len(text), "DATE"),
            (1, len(text) - 2),
        )
        start, end = trim_decoded_span(text, 0, len(text), "DATE")
        self.assertEqual(text[start:end], "14/11/2026")

    def test_phone_punctuation_is_not_trimmed_automatically(self):
        text = "+20-123-4567"
        self.assertEqual(
            trim_decoded_span(text, 0, len(text), "PHONE_NUMBER"),
            (0, len(text)),
        )


if __name__ == "__main__":
    unittest.main()
