import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from normalize_ja import normalize_ja


class NormalizeJaTest(unittest.TestCase):
    def test_nfkc_case_space_and_punctuation(self):
        self.assertEqual(normalize_ja(" ＡＳＲ、Test！\n"), "asrtest")

    def test_does_not_unify_lexical_forms(self):
        self.assertNotEqual(normalize_ja("一万円"), normalize_ja("10,000円"))
        self.assertNotEqual(normalize_ja("日本"), normalize_ja("にほん"))

    def test_removes_general_symbols(self):
        self.assertEqual(normalize_ja("価格は¥1,000です。"), "価格は1000です")


if __name__ == "__main__":
    unittest.main()

