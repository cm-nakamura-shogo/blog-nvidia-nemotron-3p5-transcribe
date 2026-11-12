#!/usr/bin/env python3
"""Benchmark true multi-file CTranslate2 batches through faster-whisper."""

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


def make_decoder(model: Any, config: dict[str, Any]) -> tuple[Any, Any, Any]:
    from faster_whisper import BatchedInferencePipeline
    from faster_whisper.tokenizer import Tokenizer
    from faster_whisper.transcribe import TranscriptionOptions, get_suppressed_tokens

    pipeline = BatchedInferencePipeline(model=model)
    tokenizer = Tokenizer(
        model.hf_tokenizer,
        model.model.is_multilingual,
        task=config["task"],
        language=config["language"],
    )
    options = TranscriptionOptions(
        beam_size=int(config["beam_size"]),
        best_of=1,
        patience=1.0,
        length_penalty=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        log_prob_threshold=None,
        no_speech_threshold=None,
        compression_ratio_threshold=None,
        condition_on_previous_text=False,
        prompt_reset_on_temperature=0.5,
        temperatures=[float(config["temperature"])],
        initial_prompt=None,
        prefix=None,
        suppress_blank=True,
        suppress_tokens=get_suppressed_tokens(tokenizer, [-1]),
        without_timestamps=bool(config["without_timestamps"]),
        max_initial_timestamp=0.0,
        word_timestamps=False,
        prepend_punctuations='"\'“¿([{-',
        append_punctuations='"\'.。,，!！?？:：”)]}、',
        multilingual=False,
        max_new_tokens=None,
        clip_timestamps=None,
        hallucination_silence_threshold=None,
        hotwords=None,
    )
    return pipeline, tokenizer, options


def process_batch(
    batch: list[dict[str, Any]], pipeline: Any, tokenizer: Any, options: Any
) -> tuple[list[str], dict[str, float]]:
    from faster_whisper.audio import pad_or_trim

    batch_started = time.perf_counter()
    load_started = time.perf_counter()
    audios = [load_audio_mono(row["audio_filepath"]) for row in batch]
    load_elapsed = time.perf_counter() - load_started

    feature_started = time.perf_counter()
    features = np.stack(
        [pad_or_trim(pipeline.model.feature_extractor(audio)[..., :-1]) for audio in audios]
    )
    feature_elapsed = time.perf_counter() - feature_started

    chunks_metadata = [
        {"offset": 0.0, "duration": float(row["duration_sec"]), "segments": []}
        for row in batch
    ]
    inference_started = time.perf_counter()
    segmented = pipeline.forward(features, tokenizer, chunks_metadata, options)
    hypotheses = ["".join(segment["text"] for segment in segments).strip() for segments in segmented]
    inference_elapsed = time.perf_counter() - inference_started
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

    from faster_whisper import WhisperModel

    set_reproducible_seed(args.seed)
    config = load_yaml(args.config)
    rows = read_jsonl(args.manifest)
    resolve_manifest_paths(rows, args.manifest)
    validate_manifest(rows)
    if args.limit is not None:
        rows = rows[: args.limit]
    rows = ordered_rows(rows, config["batch_order"])

    load_started = time.perf_counter()
    model = WhisperModel(
        config["model_id"],
        device="cuda",
        device_index=0,
        compute_type=config["compute_type"],
        revision=config["model_revision"],
        num_workers=1,
    )
    pipeline, tokenizer, options = make_decoder(model, config)
    model_load_sec = time.perf_counter() - load_started

    warmup_rows = rows[: min(len(rows), int(config["warmup_samples"]))]
    for warmup_batch in batches(warmup_rows, args.batch_size):
        process_batch(warmup_batch, pipeline, tokenizer, options)

    predictions: list[dict[str, Any]] = []
    batch_latencies = []
    inference_total = 0.0
    measured_started = time.perf_counter()
    with NvidiaSmiMonitor() as monitor:
        for batch_index, batch in enumerate(batches(rows, args.batch_size)):
            hypotheses, timings = process_batch(batch, pipeline, tokenizer, options)
            inference_total += timings["inference_sec"]
            batch_latencies.append(timings["end_to_end_sec"])
            for row, hypothesis in zip(batch, hypotheses):
                predictions.append(
                    {
                        "model_id": config["id"],
                        "model_revision": config["model_revision"],
                        "runtime": "faster-whisper/CTranslate2",
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
        "upstream_model_id": config["upstream_model_id"],
        "upstream_model_revision": config["upstream_model_revision"],
        "runtime": "faster-whisper/CTranslate2",
        "runtime_versions": {
            "faster-whisper": package_version("faster-whisper"),
            "ctranslate2": package_version("ctranslate2"),
        },
        "precision": config["compute_type"],
        "comparison_scope": config.get("comparison_scope", "main"),
        "batch_size": args.batch_size,
        "repeat": args.repeat,
        "batch_order": config["batch_order"],
        "model_load_sec": model_load_sec,
        "warmup_samples": len(warmup_rows),
        "vram_baseline_mib_nvidia_smi": monitor.baseline_mib,
        "peak_vram_mib_nvidia_smi": monitor.peak_mib,
        "vram_monitor": monitor.backend,
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
