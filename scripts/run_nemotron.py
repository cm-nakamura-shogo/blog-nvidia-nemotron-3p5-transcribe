#!/usr/bin/env python3
# Portions adapted from NVIDIA NeMo:
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0. See THIRD_PARTY_NOTICES.md.

"""Benchmark Nemotron 3.5 cache-aware streaming batches with NVIDIA NeMo.

The streaming loop is adapted from NeMo's Apache-2.0 licensed
speech_to_text_cache_aware_streaming_infer.py so the model can remain loaded
across warm-up and the measured dataset pass.
"""

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


def extract_transcriptions(hypotheses: list[Any]) -> list[str]:
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

    if hypotheses and isinstance(hypotheses[0], Hypothesis):
        return [hypothesis.text for hypothesis in hypotheses]
    return [str(hypothesis) for hypothesis in hypotheses]


def perform_streaming(model: Any, streaming_buffer: Any, pad_and_drop_preencoded: bool) -> list[str]:
    import torch

    batch_size = len(streaming_buffer.streams_length)
    cache_last_channel, cache_last_time, cache_last_channel_len = (
        model.encoder.get_initial_cache_state(batch_size=batch_size)
    )
    previous_hypotheses = None
    previous_prediction = None
    transcribed_texts = []

    for step_number, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer):
        drop_extra = (
            0
            if step_number == 0 and not pad_and_drop_preencoded
            else model.encoder.streaming_cfg.drop_extra_pre_encoded
        )
        with torch.inference_mode():
            (
                previous_prediction,
                transcribed_texts,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
                previous_hypotheses,
            ) = model.conformer_stream_step(
                processed_signal=chunk_audio.to(torch.float32),
                processed_signal_length=chunk_lengths,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=streaming_buffer.is_buffer_empty(),
                previous_hypotheses=previous_hypotheses,
                previous_pred_out=previous_prediction,
                drop_extra_pre_encoded=drop_extra,
                return_transcription=True,
            )
    return extract_transcriptions(transcribed_texts)


def process_batch(
    batch: list[dict[str, Any]], model: Any, config: dict[str, Any]
) -> tuple[list[str], dict[str, float]]:
    import torch
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    batch_started = time.perf_counter()
    buffer_started = time.perf_counter()
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=model,
        online_normalization=False,
        pad_and_drop_preencoded=bool(config["pad_and_drop_preencoded"]),
    )
    for row in batch:
        streaming_buffer.append_audio_file(row["audio_filepath"], stream_id=-1)
    buffer_elapsed = time.perf_counter() - buffer_started

    torch.cuda.synchronize()
    inference_started = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        hypotheses = perform_streaming(
            model,
            streaming_buffer,
            pad_and_drop_preencoded=bool(config["pad_and_drop_preencoded"]),
        )
    torch.cuda.synchronize()
    inference_elapsed = time.perf_counter() - inference_started
    return [hypothesis.strip() for hypothesis in hypotheses], {
        "audio_buffer_setup_sec": buffer_elapsed,
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
    from huggingface_hub import hf_hub_download
    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig

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
        filename="nemotron-3.5-asr-streaming-0.6b.nemo",
        revision=config["model_revision"],
    )
    model = ASRModel.restore_from(checkpoint, map_location="cpu")
    attention_context = [int(value) for value in config["att_context_size"]]
    model.encoder.set_default_att_context_size(att_context_size=attention_context)
    decoding = RNNTDecodingConfig(fused_batch_size=-1)
    if hasattr(model, "cur_decoder"):
        model.change_decoding_strategy(decoding, decoder_type=config["decoder_type"])
    else:
        model.change_decoding_strategy(decoding)
    model.set_inference_prompt(config["target_lang"])
    model.decoding.set_strip_lang_tags(bool(config["strip_lang_tags"]))
    model = model.to(device="cuda:0", dtype=torch.float32)
    model.eval()
    model_load_sec = time.perf_counter() - load_started

    warmup_rows = rows[: min(len(rows), int(config["warmup_samples"]))]
    for warmup_batch in batches(warmup_rows, args.batch_size):
        process_batch(warmup_batch, model, config)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    predictions: list[dict[str, Any]] = []
    batch_latencies = []
    inference_total = 0.0
    measured_started = time.perf_counter()
    with NvidiaSmiMonitor() as monitor:
        for batch_index, batch in enumerate(batches(rows, args.batch_size)):
            hypotheses, timings = process_batch(batch, model, config)
            if len(hypotheses) != len(batch):
                raise RuntimeError(
                    f"NeMo returned {len(hypotheses)} hypotheses for a batch of {len(batch)}"
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
                        "batch_audio_buffer_setup_sec": timings["audio_buffer_setup_sec"],
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
        "runtime": "NVIDIA NeMo",
        "runtime_versions": {
            "torch": package_version("torch"),
            "nemo-toolkit": package_version("nemo-toolkit"),
        },
        "precision": config["precision"],
        "comparison_scope": config.get("comparison_scope", "streaming_reference"),
        "model_weight_dtype": config["model_weight_dtype"],
        "target_lang": config["target_lang"],
        "att_context_size": attention_context,
        "chunk_size_ms": config["chunk_size_ms"],
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
        "fleurs_training_overlap_disclosed": True,
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
