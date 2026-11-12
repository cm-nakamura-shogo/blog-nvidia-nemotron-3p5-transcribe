"""Shared manifest, timing, batching, and GPU measurement helpers."""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def resolve_manifest_paths(rows: list[dict[str, Any]], manifest: Path) -> None:
    base = manifest.resolve().parent
    for row in rows:
        audio_path = Path(row["audio_filepath"])
        if not audio_path.is_absolute():
            row["audio_filepath"] = str((base / audio_path).resolve())


def validate_manifest(rows: Sequence[dict[str, Any]]) -> None:
    """Reject manifests that could alias multiple samples to one result/audio path."""

    required = {"sample_id", "audio_filepath", "duration_sec", "reference"}
    for index, row in enumerate(rows):
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Manifest row {index} is missing fields: {sorted(missing)}")
    sample_ids = [str(row["sample_id"]) for row in rows]
    audio_paths = [str(row["audio_filepath"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Manifest sample_id values must be unique")
    if len(audio_paths) != len(set(audio_paths)):
        raise ValueError("Manifest audio_filepath values must be unique")


def ordered_rows(rows: Sequence[dict[str, Any]], order: str) -> list[dict[str, Any]]:
    if order == "manifest":
        return list(rows)
    if order == "bucket":
        return sorted(rows, key=lambda row: (-float(row["duration_sec"]), row["sample_id"]))
    raise ValueError(f"Unsupported batch order: {order}")


def batches(rows: Sequence[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(rows), batch_size):
        yield list(rows[start : start + batch_size])


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def latency_summary(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "mean_sec": statistics.fmean(values) if values else None,
        "p50_sec": percentile(values, 0.50),
        "p95_sec": percentile(values, 0.95),
        "p99_sec": percentile(values, 0.99),
    }


def benchmark_summary(
    *,
    total_files: int,
    total_audio_sec: float,
    end_to_end_elapsed_sec: float,
    gpu_inference_elapsed_sec: float,
) -> dict[str, float | int]:
    if end_to_end_elapsed_sec <= 0 or total_audio_sec <= 0:
        raise ValueError("elapsed time and audio duration must be positive")
    rtf = end_to_end_elapsed_sec / total_audio_sec
    return {
        "total_files": total_files,
        "total_audio_sec": total_audio_sec,
        "end_to_end_elapsed_sec": end_to_end_elapsed_sec,
        "gpu_inference_elapsed_sec": gpu_inference_elapsed_sec,
        "rtf": rtf,
        "audio_hours_per_hour": 1.0 / rtf,
        "files_per_sec": total_files / end_to_end_elapsed_sec,
    }


@dataclass
class GpuMemorySample:
    timestamp: float
    used_mib: int


class NvidiaSmiMonitor:
    """Poll total device memory independently of the inference framework.

    NVML is preferred because spawning nvidia-smi at high frequency can distort
    short ASR measurements. A lower-frequency nvidia-smi fallback is retained.
    """

    def __init__(self, interval_sec: float = 0.05) -> None:
        self.interval_sec = interval_sec
        self.samples: list[GpuMemorySample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.backend = "nvidia-smi"
        self._nvml = None
        self._nvml_handle = None

        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.backend = "NVML"
        except Exception:
            self.interval_sec = max(interval_sec, 0.2)

    def read_used_mib(self) -> int | None:
        if self._nvml is not None and self._nvml_handle is not None:
            try:
                memory = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                return int(memory.used / 1024**2)
            except Exception:
                pass
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "--id=0",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return int(completed.stdout.strip().splitlines()[0])
        except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
            return None

    def _poll(self) -> None:
        while not self._stop.is_set():
            value = self.read_used_mib()
            if value is not None:
                self.samples.append(GpuMemorySample(time.perf_counter(), value))
            self._stop.wait(self.interval_sec)

    def __enter__(self) -> "NvidiaSmiMonitor":
        self.samples = []
        self._stop.clear()
        initial = self.read_used_mib()
        if initial is not None:
            self.samples.append(GpuMemorySample(time.perf_counter(), initial))
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_sec * 3))
        final = self.read_used_mib()
        if final is not None:
            self.samples.append(GpuMemorySample(time.perf_counter(), final))
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    @property
    def baseline_mib(self) -> int | None:
        return self.samples[0].used_mib if self.samples else None

    @property
    def peak_mib(self) -> int | None:
        return max((sample.used_mib for sample in self.samples), default=None)


def package_version(distribution: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(distribution)
    except Exception:
        return None


def run_identity(model_id: str, batch_size: int, repeat: int) -> str:
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    safe_model_id = model_id.replace("/", "-")
    return f"{timestamp}-{safe_model_id}-bs{batch_size}-run{repeat}"


def set_reproducible_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
