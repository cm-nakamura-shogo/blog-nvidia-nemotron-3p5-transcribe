#!/usr/bin/env python3
"""Benchmark the Japanese Parakeet hybrid TDT-CTC model with NeMo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

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


def hypothesis_texts(value: Any) -> list[str]:
    """Normalize the list/tuple forms returned by NeMo hybrid models."""

    if isinstance(value, tuple):
        value = value[0]
    if not isinstance(value, list):
        raise TypeError(f"Unexpected NeMo transcription result: {type(value)!r}")
    return [str(item.text if hasattr(item, "text") else item).strip() for item in value]


def process_rows(
    rows: list[dict[str, Any]], model: Any, batch_size: int
) -> list[tuple[list[str], dict[str, float]]]:
    import torch
    from nemo.collections.asr.parts.mixins.transcription import TranscribeConfig

    transcribe_config = TranscribeConfig(
        use_lhotse=True,
        batch_size=batch_size,
        return_hypotheses=True,
        num_workers=0,
        verbose=False,
        timestamps=False,
    )
    generator = model.transcribe_generator(
        [row["audio_filepath"] for row in rows],
        override_config=transcribe_config,
    )
    results: list[tuple[list[str], dict[str, float]]] = []
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        while True:
            torch.cuda.synchronize()
            batch_started = time.perf_counter()
            try:
                value = next(generator)
            except StopIteration:
                break
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - batch_started
            results.append(
                (
                    hypothesis_texts(value),
                    {"inference_sec": elapsed, "end_to_end_sec": elapsed},
                )
            )
    return results


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
    from huggingface_hub import hf_hub_download
    from nemo.collections.asr.models import ASRModel

    set_reproducible_seed(args.seed)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    config = load_yaml(args.config)
    rows = read_jsonl(args.manifest)
    resolve_manifest_paths(rows, args.manifest)
    validate_manifest(rows)
    if args.limit is not None:
        rows = rows[: args.limit]
    rows = ordered_rows(rows, config["batch_order"])

    load_started = time.perf_counter()
    checkpoint = hf_hub_download(
        repo_id=config["model_id"],
        filename=config["model_filename"],
        revision=config["model_revision"],
    )
    model = ASRModel.restore_from(checkpoint, map_location="cpu")
    decoding = model.cfg.decoding
    if "strategy" in decoding:
        decoding.strategy = config["decoding_strategy"]
    if hasattr(model, "cur_decoder"):
        model.change_decoding_strategy(decoding, decoder_type=config["decoder_type"])
    else:
        model.change_decoding_strategy(decoding)
    model = model.to(device="cuda:0", dtype=torch.float32)
    model.eval()
    model_load_sec = time.perf_counter() - load_started

    warmup_rows = rows[: min(len(rows), int(config["warmup_samples"]))]
    process_rows(warmup_rows, model, args.batch_size)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    predictions: list[dict[str, Any]] = []
    batch_latencies: list[float] = []
    inference_total = 0.0
    measured_started = time.perf_counter()
    with NvidiaSmiMonitor() as monitor:
        transcribed_batches = process_rows(rows, model, args.batch_size)
        row_batches = list(batches(rows, args.batch_size))
        if len(transcribed_batches) != len(row_batches):
            raise RuntimeError(
                f"NeMo returned {len(transcribed_batches)} batches for "
                f"{len(row_batches)} expected batches"
            )
        for batch_index, (batch, result) in enumerate(zip(row_batches, transcribed_batches)):
            hypotheses, timings = result
            if len(hypotheses) != len(batch):
                raise RuntimeError(
                    f"NeMo returned {len(hypotheses)} hypotheses for {len(batch)} inputs"
                )
            inference_total += timings["inference_sec"]
            batch_latencies.append(timings["end_to_end_sec"])
            for row, hypothesis in zip(batch, hypotheses):
                predictions.append(
                    {
                        "model_id": config["id"],
                        "model_revision": config["model_revision"],
                        "runtime": "NVIDIA NeMo",
                        "sample_id": row["sample_id"],
                        "audio_filepath": row["audio_filepath"],
                        "audio_duration_sec": row["duration_sec"],
                        "reference": row["reference"],
                        "hypothesis": hypothesis,
                        "batch_size": args.batch_size,
                        "batch_index": batch_index,
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
        "runtime": "NVIDIA NeMo",
        "runtime_versions": {
            "torch": package_version("torch"),
            "nemo-toolkit": package_version("nemo-toolkit"),
        },
        "precision": config["precision"],
        "comparison_scope": config["comparison_scope"],
        "model_weight_dtype": config["model_weight_dtype"],
        "decoder_type": config["decoder_type"],
        "transducer_type": config["transducer_type"],
        "decoding_strategy": config["decoding_strategy"],
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
