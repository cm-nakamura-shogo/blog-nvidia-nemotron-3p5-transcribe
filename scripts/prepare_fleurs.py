#!/usr/bin/env python3
"""Download pinned FLEURS test data and create a portable manifest."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.signal import resample_poly

from common import write_json, write_jsonl


DATASET_ID = "google/fleurs"
DATASET_CONFIG = "ja_jp"
DATASET_SPLIT = "test"
DATASET_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
TARGET_SAMPLE_RATE = 16_000


def decode_audio(value: dict) -> tuple[np.ndarray, int]:
    source = io.BytesIO(value["bytes"]) if value.get("bytes") is not None else value["path"]
    audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    waveform = audio.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = np.gcd(sample_rate, TARGET_SAMPLE_RATE)
        waveform = resample_poly(
            waveform,
            TARGET_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        ).astype(np.float32, copy=False)
        sample_rate = TARGET_SAMPLE_RATE
    return np.ascontiguousarray(waveform, dtype=np.float32), sample_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio-dir", type=Path, default=Path("work/fleurs-ja_jp-test/audio")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("manifests/fleurs-ja_jp-test.jsonl")
    )
    parser.add_argument(
        "--metadata", type=Path, default=Path("manifests/fleurs-ja_jp-test.metadata.json")
    )
    # 翻訳評価で英語音声（en_us）も使うため、言語を指定できるようにした。
    parser.add_argument("--config", default=DATASET_CONFIG)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    dataset = load_dataset(
        DATASET_ID,
        args.config,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
    ).cast_column("audio", Audio(decode=False))
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    args.audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, item in enumerate(dataset):
        # FLEURS `id` is not unique within a split. Include dataset position so
        # every row owns a distinct WAV path and cannot overwrite another row.
        sample_id = f"{args.config}-test-{index:04d}-id-{int(item['id']):05d}"
        audio_path = (args.audio_dir / f"{sample_id}.wav").resolve()
        waveform, sample_rate = decode_audio(item["audio"])
        sf.write(audio_path, waveform, sample_rate, subtype="PCM_16")
        raw_reference = item.get("raw_transcription") or item["transcription"]
        rows.append(
            {
                "sample_id": sample_id,
                "dataset_index": index,
                "fleurs_id": int(item["id"]),
                "audio_filepath": str(audio_path),
                "duration_sec": len(waveform) / sample_rate,
                "reference": raw_reference,
                "fleurs_transcription": item["transcription"],
                "raw_transcription": raw_reference,
                "gender": int(item["gender"]),
                "language": item["language"],
                "sample_rate": sample_rate,
            }
        )

    sample_ids = [row["sample_id"] for row in rows]
    audio_paths = [row["audio_filepath"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Generated sample_id values are not unique")
    if len(audio_paths) != len(set(audio_paths)):
        raise RuntimeError("Generated audio paths are not unique")

    write_jsonl(args.manifest, rows)
    write_json(
        args.metadata,
        {
            "dataset_id": DATASET_ID,
            "dataset_config": args.config,
            "dataset_split": DATASET_SPLIT,
            "dataset_revision": DATASET_REVISION,
            "license": "CC-BY-4.0",
            "files": len(rows),
            "total_audio_sec": sum(row["duration_sec"] for row in rows),
            "sample_rate": TARGET_SAMPLE_RATE,
            "sample_id_scheme": f"{args.config}-test-{{dataset_index:04d}}-id-{{fleurs_id:05d}}",
            "manifest": str(args.manifest.resolve()),
        },
    )
    print(json.dumps({"manifest": str(args.manifest), "files": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
