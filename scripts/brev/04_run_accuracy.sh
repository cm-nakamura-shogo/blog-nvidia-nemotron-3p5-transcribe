#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${ASR_BENCHMARK_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${ASR_BENCHMARK_PYTHON_BIN:-${PROJECT_DIR}/work/.venv/bin/python}"
cd "${PROJECT_DIR}"

"${PYTHON_BIN}" scripts/run_matrix.py \
  --matrix configs/matrix.yaml \
  --batch-sizes 1 \
  --repeats 1 \
  --output-dir results/accuracy \
  --status-dir results/status-accuracy \
  --resume

