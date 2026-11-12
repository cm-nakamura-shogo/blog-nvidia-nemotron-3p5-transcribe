#!/usr/bin/env python3
"""Summarize how transcription prompts change number and punctuation formatting."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from calculate_cer import score_rows
from common import read_jsonl, write_json

DIGIT = re.compile(r"[0-9０-９]")
KANJI_NUMERAL = re.compile(r"[〇一二三四五六七八九十百千万億]")
PUNCTUATION = re.compile(r"[、。，．,.!?！？]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = {}
    for path in args.predictions:
        rows = read_jsonl(path)
        _, accuracy = score_rows(
            [{"reference": row["reference"], "hypothesis": row["pred_text"]} for row in rows]
        )
        texts = [row["pred_text"] for row in rows]
        summary[path.stem] = {
            "files": len(rows),
            "normalized_cer": accuracy["normalized_cer"],
            "outputs_with_digits": sum(bool(DIGIT.search(text)) for text in texts),
            "outputs_with_kanji_numerals": sum(bool(KANJI_NUMERAL.search(text)) for text in texts),
            "punctuation_marks": sum(len(PUNCTUATION.findall(text)) for text in texts),
            "outputs_without_punctuation": sum(not PUNCTUATION.search(text) for text in texts),
        }
        print(path.stem, summary[path.stem])
    write_json(args.output, summary)


if __name__ == "__main__":
    main()
