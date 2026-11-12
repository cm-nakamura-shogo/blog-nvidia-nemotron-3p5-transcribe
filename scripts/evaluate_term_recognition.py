#!/usr/bin/env python3
"""台本CSVとNeMo推論結果を突き合わせ、対象語の認識結果と誤挿入を集計する。"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


LANG_TAG = re.compile(r"\s*<[a-z]{2}(?:-[A-Z]{2})?>\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--split",
        help="split列がある台本CSVから指定splitだけを集計する",
    )
    return parser.parse_args()


def normalize_strict(text: str) -> str:
    text = LANG_TAG.sub("", text)
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    return re.sub(r"\s+", " ", text)


def normalize_loose(text: str) -> str:
    return "".join(
        char
        for char in normalize_strict(text)
        if not char.isspace()
        and not unicodedata.category(char).startswith(("P", "Z"))
    )


def main() -> None:
    args = parse_args()
    with args.sentences.open(encoding="utf-8", newline="") as source:
        sentences = list(csv.DictReader(source))
    if args.split:
        if sentences and "split" not in sentences[0]:
            raise ValueError("--splitを使うには台本CSVにsplit列が必要です")
        sentences = [row for row in sentences if row["split"] == args.split]
        if not sentences:
            raise ValueError(f"指定splitの台本がありません: {args.split}")
    with args.predictions.open(encoding="utf-8") as source:
        predictions = [json.loads(line) for line in source if line.strip()]

    if len(sentences) != len(predictions):
        raise ValueError(
            f"台本数({len(sentences)})と推論結果数({len(predictions)})が一致しません"
        )

    target_terms = sorted(
        {row["target_term"] for row in sentences if row["target_term"]},
        key=len,
        reverse=True,
    )
    details = []
    by_term = defaultdict(lambda: {"attempts": 0, "strict_hits": 0, "loose_hits": 0})
    control_insertions = 0
    target_other_term_insertions = 0

    for sentence, prediction in zip(sentences, predictions, strict=True):
        hypothesis = str(prediction["pred_text"])
        target = sentence["target_term"]
        strict_hit = bool(target) and normalize_strict(target) in normalize_strict(hypothesis)
        loose_hit = bool(target) and normalize_loose(target) in normalize_loose(hypothesis)
        inserted_terms = [
            term
            for term in target_terms
            if normalize_loose(term) in normalize_loose(hypothesis)
            and normalize_loose(term) not in normalize_loose(sentence["reference_text"])
        ]

        if target:
            by_term[target]["attempts"] += 1
            by_term[target]["strict_hits"] += int(strict_hit)
            by_term[target]["loose_hits"] += int(loose_hit)
            target_other_term_insertions += int(bool(inserted_terms))
        else:
            control_insertions += int(bool(inserted_terms))

        details.append(
            {
                "id": sentence["id"],
                "role": sentence["role"],
                "category": sentence["category"],
                "target_term": target,
                "reference_text": sentence["reference_text"],
                "pred_text": hypothesis,
                "strict_hit": strict_hit if target else "",
                "loose_hit": loose_hit if target else "",
                "inserted_terms": "|".join(inserted_terms),
            }
        )

    term_results = []
    for term in sorted(by_term):
        value = by_term[term]
        term_results.append(
            {
                "target_term": term,
                **value,
                "strict_recall": value["strict_hits"] / value["attempts"],
                "loose_recall": value["loose_hits"] / value["attempts"],
            }
        )

    target_rows = sum(item["attempts"] for item in by_term.values())
    strict_hits = sum(item["strict_hits"] for item in by_term.values())
    loose_hits = sum(item["loose_hits"] for item in by_term.values())
    control_rows = sum(row["role"] == "negative_control" for row in sentences)
    summary = {
        "sentences": len(sentences),
        "target_sentences": target_rows,
        "strict_hits": strict_hits,
        "strict_recall": strict_hits / max(target_rows, 1),
        "loose_hits": loose_hits,
        "loose_recall": loose_hits / max(target_rows, 1),
        "negative_controls": control_rows,
        "controls_with_target_insertion": control_insertions,
        "control_false_insertion_rate": control_insertions / max(control_rows, 1),
        "target_rows_with_other_term_insertion": target_other_term_insertions,
        "target_other_term_insertion_rate": target_other_term_insertions
        / max(target_rows, 1),
        "failed_terms_strict": [
            row["target_term"] for row in term_results if row["strict_recall"] < 1.0
        ],
        "term_results": term_results,
        "metric_notes": {
            "strict": "NFKC・casefold・連続空白の統一後に、target_termの表記が含まれるか",
            "loose": "strictに加え、空白とUnicode句読点を除去してtarget_termが含まれるか",
        },
    }

    args.details.parent.mkdir(parents=True, exist_ok=True)
    with args.details.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=details[0].keys())
        writer.writeheader()
        writer.writerows(details)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
