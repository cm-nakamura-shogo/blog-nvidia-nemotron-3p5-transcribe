#!/usr/bin/env python3
"""Run the configured model/batch/repeat matrix with resumable status files."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from common import load_yaml, write_json


RUNNERS = {
    "faster_whisper": "run_faster_whisper.py",
    "transformers_whisper": "run_kotoba_whisper.py",
    "transformers_rnnt": "run_nemotron_transformers.py",
    "nemo_streaming": "run_nemotron.py",
    "nemo_offline": "run_parakeet.py",
    "sherpa_onnx_transducer": "run_reazonspeech_k2.py",
    "vllm_openai": "run_nemotron_transcribe_vllm.py",
}


def safe_name(value: str) -> str:
    return value.replace("/", "-")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only", nargs="+", help="Config IDs to run")
    parser.add_argument("--output-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--status-dir", type=Path, default=Path("results/status"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    matrix = load_yaml(args.matrix)
    batch_sizes = args.batch_sizes or [int(value) for value in matrix["batch_sizes"]]
    repeats = args.repeats or int(matrix["repeats"])
    manifest = Path(matrix["manifest"])
    script_dir = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.status_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.status_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for config_value in matrix["configs"]:
        config_path = Path(config_value)
        config: dict[str, Any] = load_yaml(config_path)
        if args.only and config["id"] not in args.only:
            continue
        runner = RUNNERS[config["runner"]]
        stop_after_oom = False
        for batch_size in batch_sizes:
            if stop_after_oom:
                break
            for repeat in range(1, repeats + 1):
                stem = f"{safe_name(config['id'])}-bs{batch_size}-run{repeat}"
                status_path = args.status_dir / f"{stem}.json"
                if args.resume and status_path.exists():
                    try:
                        import json

                        existing_status = json.loads(status_path.read_text(encoding="utf-8")).get(
                            "status"
                        )
                        if existing_status == "completed":
                            print(f"skip completed: {stem}", flush=True)
                            continue
                        if existing_status == "oom":
                            print(f"skip known OOM and larger batches: {stem}", flush=True)
                            stop_after_oom = True
                            break
                    except (ValueError, OSError):
                        pass

                python_executable = str(config.get("python_executable", sys.executable))
                command = [
                    python_executable,
                    str(script_dir / runner),
                    "--config",
                    str(config_path),
                    "--manifest",
                    str(manifest),
                    "--batch-size",
                    str(batch_size),
                    "--repeat",
                    str(repeat),
                    "--seed",
                    str(matrix["seed"]),
                    "--output-dir",
                    str(args.output_dir),
                ]
                if args.limit is not None:
                    command.extend(["--limit", str(args.limit)])

                log_path = log_dir / f"{stem}.log"
                started = time.time()
                print(f"run: {stem}", flush=True)
                with log_path.open("w", encoding="utf-8") as log_handle:
                    completed = subprocess.run(
                        command,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                if completed.returncode == 0:
                    write_json(
                        status_path,
                        {
                            "status": "completed",
                            "model_id": config["id"],
                            "batch_size": batch_size,
                            "repeat": repeat,
                            "elapsed_sec_including_model_load": time.time() - started,
                            "log": str(log_path),
                        },
                    )
                    print(log_text[-2000:], flush=True)
                    continue

                is_oom = "out of memory" in log_text.casefold()
                write_json(
                    status_path,
                    {
                        "status": "oom" if is_oom else "failed",
                        "model_id": config["id"],
                        "batch_size": batch_size,
                        "repeat": repeat,
                        "returncode": completed.returncode,
                        "elapsed_sec_including_model_load": time.time() - started,
                        "log": str(log_path),
                    },
                )
                print(log_text[-8000:], file=sys.stderr, flush=True)
                if is_oom:
                    stop_after_oom = True
                    break
                raise SystemExit(f"Runner failed: {stem}; see {log_path}")


if __name__ == "__main__":
    main()
