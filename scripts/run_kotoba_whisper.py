#!/usr/bin/env python3
"""Benchmark Kotoba-Whisper v2.2 ASR weights without its custom pipeline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from audio_io import TARGET_SAMPLE_RATE, load_audio_mono
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


def process_batch(
    batch: list[dict[str, Any]], model: Any, processor: Any, config: dict[str, Any]
) -> tuple[list[str], dict[str, float]]:
    import torch

    batch_started = time.perf_counter()
    load_started = time.perf_counter()
    audios = [load_audio_mono(row["audio_filepath"]) for row in batch]
    load_elapsed = time.perf_counter() - load_started

    feature_started = time.perf_counter()
    inputs = processor(
        audios,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding="longest",
        return_attention_mask=True,
    )
    inputs = inputs.to(model.device, dtype=model.dtype)
    feature_elapsed = time.perf_counter() - feature_started

    torch.cuda.synchronize()
    inference_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            language=config["language"],
            task=config["task"],
            num_beams=int(config["num_beams"]),
            do_sample=bool(config["do_sample"]),
            max_new_tokens=int(config["max_new_tokens"]),
            return_timestamps=False,
        )
    torch.cuda.synchronize()
    inference_elapsed = time.perf_counter() - inference_started
    hypotheses = [text.strip() for text in processor.batch_decode(generated, skip_special_tokens=True)]
    return hypotheses, {
        "audio_load_sec": load_elapsed,
        "feature_extraction_sec": feature_elapsed,
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

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    set_reproducible_seed(args.seed)
    config = load_yaml(args.config)
    if config.get("use_custom_pipeline"):
        raise ValueError("The benchmark must not use Kotoba's diarization/punctuation pipeline")
    rows = read_jsonl(args.manifest)
    resolve_manifest_paths(rows, args.manifest)
    validate_manifest(rows)
    if args.limit is not None:
        rows = rows[: args.limit]
    rows = ordered_rows(rows, config["batch_order"])

    dtype = torch.float16
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        config["model_id"], revision=config["model_revision"]
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=config["attn_implementation"],
    ).to("cuda:0")
    model.eval()
    model_load_sec = time.perf_counter() - load_started

    warmup_rows = rows[: min(len(rows), int(config["warmup_samples"]))]
    for warmup_batch in batches(warmup_rows, args.batch_size):
        process_batch(warmup_batch, model, processor, config)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    predictions: list[dict[str, Any]] = []
    batch_latencies = []
    inference_total = 0.0
    measured_started = time.perf_counter()
    with NvidiaSmiMonitor() as monitor:
        for batch_index, batch in enumerate(batches(rows, args.batch_size)):
            hypotheses, timings = process_batch(batch, model, processor, config)
            inference_total += timings["inference_sec"]
            batch_latencies.append(timings["end_to_end_sec"])
            for row, hypothesis in zip(batch, hypotheses):
                predictions.append(
                    {
                        "model_id": config["id"],
                        "model_revision": config["model_revision"],
                        "runtime": "Transformers",
                        "sample_id": row["sample_id"],
                        "audio_filepath": row["audio_filepath"],
                        "audio_duration_sec": row["duration_sec"],
                        "reference": row["reference"],
                        "hypothesis": hypothesis,
                        "batch_size": args.batch_size,
                        "batch_index": batch_index,
                        "batch_audio_load_sec": timings["audio_load_sec"],
                        "batch_feature_extraction_sec": timings["feature_extraction_sec"],
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

    total_audio_sec = sum(float(row["duration_sec"]) for row in rows)
    metadata = {
        "experiment_id": run_id,
        "model_id": config["id"],
        "hub_model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "runtime": "Transformers",
        "runtime_versions": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
        },
        "precision": config["precision"],
        "comparison_scope": config.get("comparison_scope", "main"),
        "attention_implementation": config["attn_implementation"],
        "custom_pipeline": False,
        "external_punctuation": False,
        "diarization": False,
        "batch_size": args.batch_size,
        "repeat": args.repeat,
        "batch_order": config["batch_order"],
        "model_load_sec": model_load_sec,
        "warmup_samples": len(warmup_rows),
        "vram_baseline_mib_nvidia_smi": monitor.baseline_mib,
        "peak_vram_mib_nvidia_smi": monitor.peak_mib,
        "vram_monitor": monitor.backend,
        "peak_allocated_mib_torch": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib_torch": torch.cuda.max_memory_reserved() / 1024**2,
        "latency": latency_summary(batch_latencies) if args.batch_size == 1 else None,
        "accuracy": accuracy,
        "predictions": str(output_path),
        **benchmark_summary(
            total_files=len(rows),
            total_audio_sec=total_audio_sec,
            end_to_end_elapsed_sec=measured_elapsed,
            gpu_inference_elapsed_sec=inference_total,
        ),
    }
    write_json(metadata_path, metadata)
    print(metadata_path)


if __name__ == "__main__":
    main()
