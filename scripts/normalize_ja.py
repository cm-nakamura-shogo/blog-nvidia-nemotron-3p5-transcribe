#!/usr/bin/env python3
"""Shared Japanese text normalization used by every model."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata


def normalize_ja(text: str) -> str:
    """Normalize presentation differences without changing lexical forms.

    The function intentionally does not unify kanji/kana, spoken/abbreviated
    forms, or kanji and Arabic numerals.
    """

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        char
        for char in normalized
        if not char.isspace()
        and unicodedata.category(char)[0] not in {"P", "S"}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    text = args.text if args.text is not None else sys.stdin.read()
    result = normalize_ja(text)
    if args.as_json:
        print(json.dumps({"input": text, "normalized": result}, ensure_ascii=False))
    else:
        print(result)


if __name__ == "__main__":
    main()

