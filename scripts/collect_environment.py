#!/usr/bin/env python3
"""Collect reproducibility metadata without failing on unavailable tools."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from common import package_version, write_json


def command_output(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except subprocess.SubprocessError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("environment/environment-info.json"))
    args = parser.parse_args()

    packages = [
        "torch",
        "transformers",
        "faster-whisper",
        "ctranslate2",
        "nemo-toolkit",
        "datasets",
        "huggingface-hub",
        "numpy",
        "soundfile",
        "scipy",
        "matplotlib",
        "sherpa-onnx",
    ]
    value = {
        "platform": platform.platform(),
        "system": platform.system(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "nvidia_smi_query": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "nvcc": command_output(["nvcc", "--version"]),
        "ffmpeg": command_output(["ffmpeg", "-version"]),
        "lsb_release": command_output(["lsb_release", "-a"]),
        "lscpu": command_output(["lscpu"]),
        "memory": command_output(["free", "-h"]),
        "disk": command_output(["df", "-h", "."]),
        "packages": {package: package_version(package) for package in packages},
    }
    try:
        import torch

        value["torch_cuda_version"] = torch.version.cuda
        value["cudnn_version"] = torch.backends.cudnn.version()
        value["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass
    write_json(args.output, value)


if __name__ == "__main__":
    main()
