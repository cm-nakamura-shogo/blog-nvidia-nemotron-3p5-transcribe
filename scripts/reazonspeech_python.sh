#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${ASR_BENCHMARK_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${ASR_BENCHMARK_REAZONSPEECH_VENV:-${PROJECT_DIR}/work/.venv-reazonspeech}"
PYTHON_BIN="${VENV_DIR}/bin/python"

NVIDIA_LIBRARY_PATHS="$(${PYTHON_BIN} -c '
from pathlib import Path
import site

root = Path(site.getsitepackages()[0]) / "nvidia"
print(":".join(str(path) for path in sorted(root.glob("*/lib"))))
')"
export LD_LIBRARY_PATH="${NVIDIA_LIBRARY_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${PYTHON_BIN}" "$@"
