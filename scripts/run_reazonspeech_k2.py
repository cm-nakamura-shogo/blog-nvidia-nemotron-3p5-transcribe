#!/usr/bin/env python3
"""Benchmark ReazonSpeech K2 v2 with sherpa-onnx's CUDA provider."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from audio_io import load_audio_mono
from calculate_cer import score_rows
from common import (
    NvidiaSmiMonitor,
    batches,
    benchmark_summary,
    latency_summary,
    load_yaml,
    ordered_rows,
    package_version,
    read_jsonl,
    resolve_manifest_paths,
    run_identity,
    set_reproducible_seed,
    validate_manifest,
    write_json,
    write_jsonl,
)


MODEL_FILES = {
    "encoder": "encoder-epoch-99-avg-1.onnx",
    "decoder": "decoder-epoch-99-avg-1.onnx",
    "joiner": "joiner-epoch-99-avg-1.onnx",
    "tokens": "tokens.txt",
}


def pad_audio_for_model(
    audio: np.ndarray, sample_rate: int, pad_seconds: float
) -> np.ndarray:
    """Match ReazonSpeech's official 0.9-second padding on both sides."""
    if pad_seconds < 0:
        raise ValueError("pad_seconds must be non-negative")
    return np.pad(audio, pad_width=int(sample_rate * pad_seconds))


def process_batch(
    batch: list[dict[str, Any]],
    recognizer: Any,
    sample_rate: int,
    pad_seconds: float,
) -> tuple[list[str], dict[str, float]]:
    batch_started = time.perf_counter()
    load_started = time.perf_counter()
    audios = [
        pad_audio_for_model(
            load_audio_mono(row["audio_filepath"], sample_rate),
            sample_rate,
            pad_seconds,
        )
        for row in batch
    ]
    load_elapsed = time.perf_counter() - load_started

    streams = []
    for audio in audios:
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        streams.append(stream)
    inference_started = time.perf_counter()
    # decode_streams is synchronous, so completion is the CUDA timing boundary.
    recognizer.decode_streams(streams)
    inference_elapsed = time.perf_counter() - inference_started
    hypotheses = [stream.result.text.strip() for stream in streams]
    return hypotheses, {
        "audio_load_sec": load_elapsed,
        "inference_sec": inference_elapsed,
        "end_to_end_sec": time.perf_counter() - batch_started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("results/raw"))
    args = parser.parse_args()

    import sherpa_onnx
    from huggingface_hub import hf_hub_download

    set_reproducible_seed(args.seed)
    config = load_yaml(args.config)
    rows = read_jsonl(args.manifest)
    resolve_manifest_paths(rows, args.manifest)
    validate_manifest(rows)
    if args.limit is not None:
        rows = rows[: args.limit]
    rows = ordered_rows(rows, config["batch_order"])

    load_started = time.perf_counter()
    model_paths = {
        name: hf_hub_download(
            repo_id=config["model_id"],
            filename=filename,
            revision=config["model_revision"],
        )
        for name, filename in MODEL_FILES.items()
    }
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=model_paths["encoder"],
        decoder=model_paths["decoder"],
        joiner=model_paths["joiner"],
        tokens=model_paths["tokens"],
        num_threads=int(config["num_threads"]),
        sample_rate=int(config["sample_rate"]),
        feature_dim=int(config["feature_dim"]),
        decoding_method=config["decoding_method"],
        provider=config["provider"],
    )
    model_load_sec = time.perf_counter() - load_started

    warmup_rows = rows[: min(len(rows), int(config["warmup_samples"]))]
    for warmup_batch in batches(warmup_rows, args.batch_size):
        process_batch(
            warmup_batch,
            recognizer,
            int(config["sample_rate"]),
            float(config["pad_seconds"]),
        )

    predictions: list[dict[str, Any]] = []
    batch_latencies: list[float] = []
    inference_total = 0.0
    measured_started = time.perf_counter()
    with NvidiaSmiMonitor() as monitor:
        for batch_index, batch in enumerate(batches(rows, args.batch_size)):
            hypotheses, timings = process_batch(
                batch,
                recognizer,
                int(config["sample_rate"]),
                float(config["pad_seconds"]),
            )
            if len(hypotheses) != len(batch):
                raise RuntimeError(
                    f"sherpa-onnx returned {len(hypotheses)} hypotheses for {len(batch)} inputs"
                )
            inference_total += timings["inference_sec"]
            batch_latencies.append(timings["end_to_end_sec"])
            for row, hypothesis in zip(batch, hypotheses):
                predictions.append(
                    {
                        "model_id": config["id"],
                        "model_revision": config["model_revision"],
                        "runtime": "sherpa-onnx CUDA",
                        "sample_id": row["sample_id"],
                        "audio_filepath": row["audio_filepath"],
                        "audio_duration_sec": row["duration_sec"],
                        "reference": row["reference"],
                        "hypothesis": hypothesis,
                        "batch_size": args.batch_size,
                        "batch_index": batch_index,
                        "batch_audio_load_sec": timings["audio_load_sec"],
                        "batch_inference_time_sec": timings["inference_sec"],
                        "batch_end_to_end_sec": timings["end_to_end_sec"],
                    }
                )
    measured_elapsed = time.perf_counter() - measured_started

    details, accuracy = score_rows(predictions)
    run_id = run_identity(config["id"], args.batch_size, args.repeat)
    output_path = args.output_dir / f"{run_id}.jsonl"
    metadata_path = args.output_dir / f"{run_id}.meta.json"
    write_jsonl(output_path, details)

    metadata = {
        "experiment_id": run_id,
        "model_id": config["id"],
        "hub_model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "runtime": "sherpa-onnx CUDA",
        "runtime_versions": {"sherpa-onnx": package_version("sherpa-onnx")},
        "precision": config["precision"],
        "comparison_scope": config["comparison_scope"],
        "provider": config["provider"],
        "decoding_method": config["decoding_method"],
        "pad_seconds_per_side": config["pad_seconds"],
        "batch_size": args.batch_size,
        "repeat": args.repeat,
        "batch_order": config["batch_order"],
        "model_load_sec": model_load_sec,
        "warmup_samples": len(warmup_rows),
        "vram_baseline_mib_nvidia_smi": monitor.baseline_mib,
        "peak_vram_mib_nvidia_smi": monitor.peak_mib,
        "vram_monitor": monitor.backend,
        "peak_allocated_mib_torch": None,
        "peak_reserved_mib_torch": None,
        "latency": latency_summary(batch_latencies) if args.batch_size == 1 else None,
        "accuracy": accuracy,
        "predictions": str(output_path),
        "fleurs_training_overlap_disclosed": config[
            "fleurs_training_overlap_disclosed"
        ],
        **benchmark_summary(
            total_files=len(rows),
            total_audio_sec=sum(float(row["duration_sec"]) for row in rows),
            end_to_end_elapsed_sec=measured_elapsed,
            gpu_inference_elapsed_sec=inference_total,
        ),
    }
    write_json(metadata_path, metadata)
    print(metadata_path)


if __name__ == "__main__":
    main()
