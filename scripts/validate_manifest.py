#!/usr/bin/env python3
"""Validate manifest identity, WAV headers, durations, and content uniqueness."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import soundfile as sf

from common import read_jsonl, resolve_manifest_paths, validate_manifest, write_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-files", type=int, required=True)
    parser.add_argument("--expected-sample-rate", type=int, default=16_000)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    resolve_manifest_paths(rows, args.manifest)
    validate_manifest(rows)
    missing_paths = []
    header_mismatches = []
    hashes = []
    actual_total_sec = 0.0
    for row in rows:
        path = Path(row["audio_filepath"])
        if not path.is_file():
            missing_paths.append(str(path))
            continue
        info = sf.info(path)
        actual_duration = info.frames / info.samplerate
        actual_total_sec += actual_duration
        if (
            info.samplerate != args.expected_sample_rate
            or info.channels != 1
            or abs(actual_duration - float(row["duration_sec"])) > 1e-6
        ):
            header_mismatches.append(row["sample_id"])
        hashes.append(sha256_file(path))

    result = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "rows": len(rows),
        "unique_sample_ids": len({row["sample_id"] for row in rows}),
        "unique_audio_paths": len({row["audio_filepath"] for row in rows}),
        "existing_audio_files": len(rows) - len(missing_paths),
        "unique_audio_sha256": len(set(hashes)),
        "missing_paths": missing_paths,
        "header_mismatches": header_mismatches,
        "declared_total_audio_sec": sum(float(row["duration_sec"]) for row in rows),
        "actual_total_audio_sec": actual_total_sec,
        "expected_files": args.expected_files,
        "expected_sample_rate": args.expected_sample_rate,
    }
    passed = (
        len(rows) == args.expected_files
        and result["unique_sample_ids"] == args.expected_files
        and result["unique_audio_paths"] == args.expected_files
        and result["existing_audio_files"] == args.expected_files
        and result["unique_audio_sha256"] == args.expected_files
        and not missing_paths
        and not header_mismatches
    )
    result["passed"] = passed
    write_json(args.output, result)
    if not passed:
        raise SystemExit(f"Manifest validation failed; see {args.output}")
    print(args.output)


if __name__ == "__main__":
    main()
