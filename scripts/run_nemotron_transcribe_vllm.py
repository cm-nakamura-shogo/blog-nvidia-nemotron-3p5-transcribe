#!/usr/bin/env python3
"""Benchmark Nemotron 3.5 Transcribe through a running vLLM OpenAI-compatible server.

vLLM schedules requests with continuous batching, so ``--batch-size`` is the
number of in-flight client requests rather than a synchronous batch. With
batch size 1 this is identical to sequential per-file inference.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from calculate_cer import score_rows
from common import (
    NvidiaSmiMonitor,
    benchmark_summary,
    latency_summary,
    load_yaml,
    ordered_rows,
    read_jsonl,
    resolve_manifest_paths,
    run_identity,
    validate_manifest,
    write_json,
    write_jsonl,
)


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 600) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def build_payload(audio_path: str, config: dict[str, Any]) -> dict[str, Any]:
    audio = base64.b64encode(Path(audio_path).read_bytes()).decode("ascii")
    messages: list[dict[str, Any]] = []
    if config.get("system_prompt"):
        messages.append({"role": "system", "content": config["system_prompt"]})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": config["user_prompt"]},
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio}"}},
            ],
        }
    )
    return {
        "model": config["served_model_name"],
        "messages": messages,
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": int(config["max_tokens"]),
        "chat_template_kwargs": {"enable_thinking": False},
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "seed": 0,
    }


def transcribe(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    payload = build_payload(row["audio_filepath"], config)
    response = http_json(f"{config['server_url']}/v1/chat/completions", payload)
    elapsed = time.perf_counter() - started
    choice = response["choices"][0]
    return {
        "hypothesis": (choice["message"].get("content") or "").strip(),
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": response.get("usage", {}).get("completion_tokens"),
        "request_sec": elapsed,
    }


def run_requests(
    rows: list[dict[str, Any]], config: dict[str, Any], concurrency: int
) -> list[dict[str, Any]]:
    """Keep ``concurrency`` requests in flight and return results in input order."""

    results: list[dict[str, Any] | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        pending: dict[Future[dict[str, Any]], int] = {}
        next_index = 0
        while next_index < len(rows) or pending:
            while next_index < len(rows) and len(pending) < concurrency:
                pending[executor.submit(transcribe, rows[next_index], config)] = next_index
                next_index += 1
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                results[pending.pop(future)] = future.result()
    return [result for result in results if result is not None]


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

    config = load_yaml(args.config)
    rows = read_jsonl(args.manifest)
    resolve_manifest_paths(rows, args.manifest)
    validate_manifest(rows)
    if args.limit is not None:
        rows = rows[: args.limit]
    rows = ordered_rows(rows, config["batch_order"])

    served = http_json(f"{config['server_url']}/v1/models", timeout=30)
    served_ids = [model["id"] for model in served.get("data", [])]
    if config["served_model_name"] not in served_ids:
        raise RuntimeError(f"Server does not serve {config['served_model_name']}: {served_ids}")
    try:
        vllm_version = http_json(f"{config['server_url']}/version", timeout=30).get("version")
    except Exception:
        vllm_version = None

    warmup_rows = rows[: min(len(rows), int(config["warmup_samples"]))]
    run_requests(warmup_rows, config, args.batch_size)

    measured_started = time.perf_counter()
    with NvidiaSmiMonitor() as monitor:
        results = run_requests(rows, config, args.batch_size)
    measured_elapsed = time.perf_counter() - measured_started
    if len(results) != len(rows):
        raise RuntimeError(f"Got {len(results)} results for {len(rows)} requests")

    predictions: list[dict[str, Any]] = []
    for index, (row, result) in enumerate(zip(rows, results)):
        predictions.append(
            {
                "model_id": config["id"],
                "model_revision": config["model_revision"],
                "runtime": "vLLM OpenAI-compatible server",
                "sample_id": row["sample_id"],
                "audio_filepath": row["audio_filepath"],
                "audio_duration_sec": row["duration_sec"],
                "reference": row["reference"],
                "hypothesis": result["hypothesis"],
                "finish_reason": result["finish_reason"],
                "completion_tokens": result["completion_tokens"],
                "batch_size": args.batch_size,
                "batch_index": index // args.batch_size,
                "request_sec": result["request_sec"],
            }
        )

    details, accuracy = score_rows(predictions)
    run_id = run_identity(config["id"], args.batch_size, args.repeat)
    output_path = args.output_dir / f"{run_id}.jsonl"
    metadata_path = args.output_dir / f"{run_id}.meta.json"
    write_jsonl(output_path, details)

    request_latencies = [result["request_sec"] for result in results]
    total_audio_sec = sum(float(row["duration_sec"]) for row in rows)
    metadata = {
        "experiment_id": run_id,
        "model_id": config["id"],
        "hub_model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "runtime": "vLLM OpenAI-compatible server",
        "runtime_versions": {"vllm": vllm_version},
        "precision": config["precision"],
        "comparison_scope": config.get("comparison_scope", "main"),
        "model_weight_dtype": config["model_weight_dtype"],
        "request_mode": "concurrent",
        "user_prompt": config["user_prompt"],
        "speculative_decoding": config.get("speculative_decoding"),
        "mamba_backend": config.get("mamba_backend"),
        "batch_size": args.batch_size,
        "repeat": args.repeat,
        "batch_order": config["batch_order"],
        "model_load_sec": None,
        "warmup_samples": len(warmup_rows),
        "vram_baseline_mib_nvidia_smi": monitor.baseline_mib,
        "peak_vram_mib_nvidia_smi": monitor.peak_mib,
        "vram_monitor": monitor.backend,
        "vram_note": "vLLM preallocates KV cache by --gpu-memory-utilization",
        "latency": latency_summary(request_latencies) if args.batch_size == 1 else None,
        "request_latency_all": latency_summary(request_latencies),
        "finish_reason_length": sum(1 for r in results if r["finish_reason"] == "length"),
        "empty_hypotheses": sum(1 for r in results if not r["hypothesis"]),
        "accuracy": accuracy,
        "predictions": str(output_path),
        "fleurs_training_overlap_disclosed": True,
        **benchmark_summary(
            total_files=len(rows),
            total_audio_sec=total_audio_sec,
            end_to_end_elapsed_sec=measured_elapsed,
            gpu_inference_elapsed_sec=measured_elapsed,
        ),
    }
    write_json(metadata_path, metadata)
    print(metadata_path)


if __name__ == "__main__":
    main()
