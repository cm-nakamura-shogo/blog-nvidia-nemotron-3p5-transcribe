#!/usr/bin/env python3
"""Calculate raw and normalized character error metrics from JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import read_jsonl, write_json, write_jsonl
from normalize_ja import normalize_ja


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    hits: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def has_suspicious_repetition(value: str, *, minimum_unit: int = 4, repeats: int = 3) -> bool:
    """Flag a character span repeated consecutively at least three times.

    This is deliberately a narrow, reproducible heuristic for ASR looping. It
    does not attempt to classify ordinary lexical repetition.
    """

    normalized = normalize_ja(value)
    maximum_unit = min(24, len(normalized) // repeats)
    for unit_length in range(minimum_unit, maximum_unit + 1):
        for start in range(0, len(normalized) - unit_length * repeats + 1):
            unit = normalized[start : start + unit_length]
            if unit * repeats == normalized[start : start + unit_length * repeats]:
                return True
    return False


def edit_counts(reference: str, hypothesis: str) -> EditCounts:
    """Return deterministic Levenshtein counts for character sequences."""

    # Each cell stores (total errors, substitutions, deletions, insertions, hits).
    previous = [(index, 0, index, 0, 0) for index in range(len(reference) + 1)]
    for hyp_char in hypothesis:
        current = [(previous[0][0] + 1, 0, 0, previous[0][3] + 1, 0)]
        for ref_index, ref_char in enumerate(reference, start=1):
            if ref_char == hyp_char:
                diagonal = previous[ref_index - 1]
                candidates = [
                    (diagonal[0], diagonal[1], diagonal[2], diagonal[3], diagonal[4] + 1)
                ]
            else:
                diagonal = previous[ref_index - 1]
                candidates = [
                    (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3], diagonal[4])
                ]

            deletion = current[ref_index - 1]
            insertion = previous[ref_index]
            candidates.extend(
                [
                    (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3], deletion[4]),
                    (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1, insertion[4]),
                ]
            )
            # Prefer hits/substitutions over delete+insert when edit totals tie.
            current.append(min(candidates, key=lambda item: (item[0], item[2] + item[3], item[1])))
        previous = current

    _, substitutions, deletions, insertions, hits = previous[-1]
    return EditCounts(substitutions, deletions, insertions, hits)


def score_pair(reference: str, hypothesis: str) -> dict[str, Any]:
    counts = edit_counts(reference, hypothesis)
    denominator = len(reference)
    cer = counts.errors / denominator if denominator else (0.0 if not hypothesis else 1.0)
    return {
        "reference_characters": denominator,
        "substitutions": counts.substitutions,
        "deletions": counts.deletions,
        "insertions": counts.insertions,
        "hits": counts.hits,
        "errors": counts.errors,
        "cer": cer,
    }


def score_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    details: list[dict[str, Any]] = []
    totals = Counter()
    normalized_totals = Counter()
    thresholds = Counter()
    empty_outputs = 0
    exact_matches = 0
    suspicious_repetitions = 0

    for row in rows:
        reference = row.get("reference", row.get("raw_reference", ""))
        hypothesis = row.get("hypothesis", "")
        raw = score_pair(reference.strip(), hypothesis.strip())
        normalized_reference = normalize_ja(reference)
        normalized_hypothesis = normalize_ja(hypothesis)
        normalized = score_pair(normalized_reference, normalized_hypothesis)

        for key in ("reference_characters", "substitutions", "deletions", "insertions", "hits", "errors"):
            totals[key] += raw[key]
            normalized_totals[key] += normalized[key]

        for threshold in (0.10, 0.20, 0.50):
            if normalized["cer"] >= threshold:
                thresholds[f"normalized_cer_ge_{int(threshold * 100)}pct"] += 1
        empty_outputs += int(not hypothesis.strip())
        exact_matches += int(normalized_reference == normalized_hypothesis)
        repetition = has_suspicious_repetition(hypothesis)
        suspicious_repetitions += int(repetition)

        detail = dict(row)
        detail.update(
            {
                "normalized_reference": normalized_reference,
                "normalized_hypothesis": normalized_hypothesis,
                "raw_cer": raw["cer"],
                "normalized_cer": normalized["cer"],
                "raw_edits": raw,
                "normalized_edits": normalized,
                "suspicious_repetition": repetition,
            }
        )
        details.append(detail)

    count = len(rows)
    raw_denominator = totals["reference_characters"]
    normalized_denominator = normalized_totals["reference_characters"]
    summary = {
        "files": count,
        "raw_cer": totals["errors"] / raw_denominator if raw_denominator else None,
        "normalized_cer": (
            normalized_totals["errors"] / normalized_denominator if normalized_denominator else None
        ),
        "exact_match_rate": exact_matches / count if count else None,
        "empty_output_rate": empty_outputs / count if count else None,
        "suspicious_repetition_count": suspicious_repetitions,
        "suspicious_repetition_rate": suspicious_repetitions / count if count else None,
        "raw_counts": dict(totals),
        "normalized_counts": dict(normalized_totals),
        **{
            key: {"count": thresholds[key], "rate": thresholds[key] / count if count else None}
            for key in (
                "normalized_cer_ge_10pct",
                "normalized_cer_ge_20pct",
                "normalized_cer_ge_50pct",
            )
        },
    }
    return details, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    details, summary = score_rows(read_jsonl(args.input))
    write_jsonl(args.details, details)
    write_json(args.summary, summary)


if __name__ == "__main__":
    main()
