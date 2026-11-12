#!/usr/bin/env python3
"""Verify available benchmark runs or audit a complete planned matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import load_yaml, read_jsonl, validate_manifest, write_json


def resolve_prediction_path(metadata_path: Path, value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = metadata_path.parent / path.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Prediction file not found for {metadata_path}: {value}")


def expected_combinations(
    model_ids: list[str],
    batch_sizes: list[int],
    repeats: int,
) -> set[tuple[str, int, int]]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    return {
        (model_id, int(batch_size), repeat)
        for model_id in model_ids
        for batch_size in batch_sizes
        for repeat in range(1, repeats + 1)
    }


def expected_runs(
    matrix: dict[str, Any],
    *,
    batch_sizes: list[int] | None = None,
    repeats: int | None = None,
) -> tuple[set[tuple[str, int, int]], dict[str, str]]:
    model_revisions: dict[str, str] = {}
    for config_value in matrix["configs"]:
        config = load_yaml(Path(config_value))
        model_revisions[config["id"]] = config["model_revision"]
    selected_batch_sizes = batch_sizes or [int(value) for value in matrix["batch_sizes"]]
    selected_repeats = repeats if repeats is not None else int(matrix["repeats"])
    combinations = expected_combinations(
        list(model_revisions),
        selected_batch_sizes,
        selected_repeats,
    )
    return combinations, model_revisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--status-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument("--repeats", type=int)
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="Verify only completed metadata found in input-dir instead of a planned matrix",
    )
    args = parser.parse_args()

    matrix = load_yaml(args.matrix)
    manifest_path = Path(matrix["manifest"])
    manifest = read_jsonl(manifest_path)
    validate_manifest(manifest)
    expected_sample_ids = {row["sample_id"] for row in manifest}
    selected_batch_sizes = args.batch_sizes or [int(value) for value in matrix["batch_sizes"]]
    selected_repeats = args.repeats if args.repeats is not None else int(matrix["repeats"])
    expected, model_revisions = expected_runs(
        matrix,
        batch_sizes=selected_batch_sizes,
        repeats=selected_repeats,
    )

    if args.available_only:
        completed_statuses: set[tuple[str, int, int]] | None = None
        if args.status_dir is not None:
            completed_statuses = set()
            for status_path in sorted(args.status_dir.glob("*.json")):
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if status.get("status") != "completed":
                    continue
                completed_statuses.add(
                    (
                        status["model_id"],
                        int(status["batch_size"]),
                        int(status["repeat"]),
                    )
                )

        found: dict[tuple[str, int, int], Path] = {}
        checked_predictions = 0
        for metadata_path in sorted(args.input_dir.rglob("*.meta.json")):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            key = (
                metadata["model_id"],
                int(metadata["batch_size"]),
                int(metadata["repeat"]),
            )
            if key[0] not in model_revisions:
                raise ValueError(f"Unknown model in available result {key}: {metadata_path}")
            if key in found:
                raise ValueError(f"Duplicate available result {key}: {found[key]} and {metadata_path}")
            if completed_statuses is not None and key not in completed_statuses:
                raise ValueError(f"Available result has no completed status {key}: {metadata_path}")
            if metadata.get("model_revision") != model_revisions[key[0]]:
                raise ValueError(
                    f"Model revision mismatch for {key}: {metadata.get('model_revision')}"
                )
            prediction_path = resolve_prediction_path(metadata_path, metadata["predictions"])
            predictions = read_jsonl(prediction_path)
            sample_ids = [row["sample_id"] for row in predictions]
            if len(sample_ids) != len(expected_sample_ids):
                raise ValueError(
                    f"Expected {len(expected_sample_ids)} predictions for {key}, got {len(sample_ids)}"
                )
            if len(sample_ids) != len(set(sample_ids)):
                raise ValueError(f"Duplicate prediction sample IDs for {key}")
            if set(sample_ids) != expected_sample_ids:
                missing = sorted(expected_sample_ids - set(sample_ids))[:5]
                extra = sorted(set(sample_ids) - expected_sample_ids)[:5]
                raise ValueError(
                    f"Prediction sample IDs differ for {key}; missing={missing}, extra={extra}"
                )
            found[key] = metadata_path
            checked_predictions += len(predictions)

        if not found:
            raise ValueError(f"No completed results found under {args.input_dir}")
        write_json(
            args.output,
            {
                "status": "passed",
                "mode": "available_only",
                "matrix": str(args.matrix),
                "manifest": str(manifest_path),
                "models": sorted({key[0] for key in found}),
                "batch_sizes": sorted({key[1] for key in found}),
                "repeats": sorted({key[2] for key in found}),
                "verified_runs": len(found),
                "samples_per_run": len(expected_sample_ids),
                "verified_predictions": checked_predictions,
            },
        )
        print(args.output)
        return

    if args.status_dir is None:
        parser.error("--status-dir is required unless --available-only is used")

    statuses: dict[tuple[str, int, int], dict[str, Any]] = {}
    oom_at: dict[str, tuple[int, int]] = {}
    for status_path in sorted(args.status_dir.glob("*.json")):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        key = (status["model_id"], int(status["batch_size"]), int(status["repeat"]))
        if key not in expected:
            raise ValueError(f"Unexpected status {key}: {status_path}")
        if key in statuses:
            raise ValueError(f"Duplicate status for {key}: {status_path}")
        state = status.get("status")
        if state not in {"completed", "oom"}:
            raise ValueError(f"Benchmark run {key} has non-terminal status {state!r}")
        statuses[key] = status
        if state == "oom":
            marker = (key[1], key[2])
            oom_at[key[0]] = min(marker, oom_at.get(key[0], marker))

    missing_statuses = []
    skipped_after_oom = []
    for key in sorted(expected):
        if key in statuses:
            continue
        marker = oom_at.get(key[0])
        if marker is not None and (key[1], key[2]) > marker:
            skipped_after_oom.append(key)
            continue
        missing_statuses.append(key)
    if missing_statuses:
        raise ValueError(
            f"Missing {len(missing_statuses)} terminal statuses; first={missing_statuses[:5]}"
        )
    completed = {key for key, value in statuses.items() if value["status"] == "completed"}

    found: dict[tuple[str, int, int], Path] = {}
    checked_predictions = 0
    for metadata_path in sorted(args.input_dir.rglob("*.meta.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        key = (
            metadata["model_id"],
            int(metadata["batch_size"]),
            int(metadata["repeat"]),
        )
        if key not in expected:
            raise ValueError(f"Unexpected benchmark run {key}: {metadata_path}")
        if key not in completed:
            raise ValueError(f"Metadata exists without completed status for {key}: {metadata_path}")
        if key in found:
            raise ValueError(f"Duplicate benchmark run {key}: {found[key]} and {metadata_path}")
        if metadata.get("model_revision") != model_revisions[key[0]]:
            raise ValueError(
                f"Model revision mismatch for {key}: {metadata.get('model_revision')}"
            )
        prediction_path = resolve_prediction_path(metadata_path, metadata["predictions"])
        predictions = read_jsonl(prediction_path)
        sample_ids = [row["sample_id"] for row in predictions]
        if len(sample_ids) != len(expected_sample_ids):
            raise ValueError(
                f"Expected {len(expected_sample_ids)} predictions for {key}, got {len(sample_ids)}"
            )
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"Duplicate prediction sample IDs for {key}")
        if set(sample_ids) != expected_sample_ids:
            missing = sorted(expected_sample_ids - set(sample_ids))[:5]
            extra = sorted(set(sample_ids) - expected_sample_ids)[:5]
            raise ValueError(f"Prediction sample IDs differ for {key}; missing={missing}, extra={extra}")
        found[key] = metadata_path
        checked_predictions += len(predictions)

    missing_runs = sorted(completed - set(found))
    if missing_runs:
        raise ValueError(f"Missing {len(missing_runs)} benchmark runs; first={missing_runs[:5]}")

    write_json(
        args.output,
        {
            "status": "passed",
            "matrix": str(args.matrix),
            "manifest": str(manifest_path),
            "models": sorted(model_revisions),
            "batch_sizes": selected_batch_sizes,
            "repeats": selected_repeats,
            "planned_runs": len(expected),
            "completed_runs": len(completed),
            "verified_runs": len(found),
            "oom_runs": [
                {"model_id": model_id, "batch_size": marker[0], "repeat": marker[1]}
                for model_id, marker in sorted(oom_at.items())
            ],
            "skipped_runs_after_oom": len(skipped_after_oom),
            "samples_per_run": len(expected_sample_ids),
            "verified_predictions": checked_predictions,
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
