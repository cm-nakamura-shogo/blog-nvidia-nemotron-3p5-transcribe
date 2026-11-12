import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from calculate_cer import edit_counts, has_suspicious_repetition, score_pair, score_rows


class CalculateCerTest(unittest.TestCase):
    def test_edit_types(self):
        self.assertEqual(edit_counts("abc", "axc").substitutions, 1)
        self.assertEqual(edit_counts("abc", "ac").deletions, 1)
        self.assertEqual(edit_counts("abc", "abbc").insertions, 1)

    def test_cer(self):
        self.assertAlmostEqual(score_pair("日本語", "日本")["cer"], 1 / 3)

    def test_corpus_cer_is_micro_average(self):
        rows = [
            {"reference": "a", "hypothesis": ""},
            {"reference": "bbbbbbbbb", "hypothesis": "bbbbbbbbb"},
        ]
        _, summary = score_rows(rows)
        self.assertAlmostEqual(summary["raw_cer"], 0.1)
        self.assertEqual(summary["empty_output_rate"], 0.5)

    def test_zero_thresholds_are_recorded(self):
        _, summary = score_rows([{"reference": "正解", "hypothesis": "正解"}])
        self.assertEqual(summary["normalized_cer_ge_10pct"], {"count": 0, "rate": 0.0})
        self.assertEqual(summary["normalized_cer_ge_20pct"], {"count": 0, "rate": 0.0})
        self.assertEqual(summary["normalized_cer_ge_50pct"], {"count": 0, "rate": 0.0})

    def test_suspicious_repetition_heuristic(self):
        self.assertTrue(has_suspicious_repetition("音声認識です音声認識です音声認識です"))
        self.assertFalse(has_suspicious_repetition("これは普通の日本語音声です"))


if __name__ == "__main__":
    unittest.main()
