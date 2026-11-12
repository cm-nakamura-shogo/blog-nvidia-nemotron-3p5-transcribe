#!/usr/bin/env python3
"""Send one manifest to the vLLM server under several prompt variants.

Used for the translation and prompt-control checks. Each variant writes one
JSONL in manifest order, with ``pred_text`` so that term evaluation from the
earlier fine-tuning article can read it unchanged.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from common import load_yaml, read_jsonl, resolve_manifest_paths, write_json, write_jsonl
from run_nemotron_transcribe_vllm import run_requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Server/model config YAML")
    parser.add_argument("--probes", type=Path, required=True, help="Prompt variants YAML")
    parser.add_argument("--only", nargs="+", help="Probe names to run")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("results/probes"))
    args = parser.parse_args()

    base_config = load_yaml(args.config)
    probes: dict[str, Any] = load_yaml(args.probes)["probes"]
    for name, probe in probes.items():
        if args.only and name not in args.only:
            continue
        manifest = Path(probe["manifest"])
        rows = read_jsonl(manifest)
        resolve_manifest_paths(rows, manifest)
        if args.limit is not None:
            rows = rows[: args.limit]
        config = {
            **base_config,
            "system_prompt": probe.get("system_prompt", ""),
            "user_prompt": probe["user_prompt"],
            "max_tokens": probe.get("max_tokens", base_config["max_tokens"]),
        }

        print(f"probe: {name} ({len(rows)} files)", flush=True)
        started = time.perf_counter()
        results = run_requests(rows, config, args.concurrency)
        elapsed = time.perf_counter() - started

        outputs = []
        for row, result in zip(rows, results, strict=True):
            outputs.append(
                {
                    "probe": name,
                    "sample_id": row["sample_id"],
                    "fleurs_id": row.get("fleurs_id"),
                    "audio_filepath": row["audio_filepath"],
                    "reference": row.get("reference", row.get("text")),
                    "pred_text": result["hypothesis"],
                    "finish_reason": result["finish_reason"],
                    "completion_tokens": result["completion_tokens"],
                    "request_sec": result["request_sec"],
                }
            )
        write_jsonl(args.output_dir / f"{name}.jsonl", outputs)
        write_json(
            args.output_dir / f"{name}.meta.json",
            {
                "probe": name,
                "hub_model_id": base_config["model_id"],
                "model_revision": base_config["model_revision"],
                "manifest": str(manifest),
                "files": len(rows),
                "concurrency": args.concurrency,
                "system_prompt": config["system_prompt"],
                "user_prompt": config["user_prompt"],
                "max_tokens": config["max_tokens"],
                "elapsed_sec": elapsed,
                "finish_reason_length": sum(r["finish_reason"] == "length" for r in results),
                "empty_outputs": sum(not r["hypothesis"] for r in results),
            },
        )


if __name__ == "__main__":
    main()
